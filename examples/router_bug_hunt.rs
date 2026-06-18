//! Router bug hunt (Sprint 383 follow-up, loop iteration 1).
//!
//! Deterministic repro + structural diagnostics for the routable-but-incorrect
//! placement class found by the HLS corpus: `hls_mix6` (48 gates) under the SA
//! layout routes cleanly but fails physical verification, violating the
//! "routable ⟹ correct" invariant. The mapped netlist is provably correct, so
//! the defect is in routing legality or export.
//!
//! Diagnostics:
//!   A. find failing (input combo, output) pairs (AIG/netlist vs placed tiles);
//!   B1. cells occupied by more than one net (illegal short under no_crossings);
//!   B2. via columns (x, y) used by more than one net — ViaUp/ViaDown read the
//!       same column, so two nets sharing a column interfere even on different
//!       layer pairs (the "Dual-Via Conflict" rule the router may not model);
//!   B3. route cells that collide with placed gate tiles.
//!
//! Run: `cargo run --release --example router_bug_hunt`

use std::collections::{BTreeSet, HashMap};

use engine::synth::RouteConfig;
use engine::synth::alphafabric::{AnnealConfig, PlacementEnv, anneal, hls_instances};
use engine::synth::export::{evaluate_exported, export_to_simulation};
use engine::synth::mapping::{evaluate_aig, evaluate_netlist};
use engine::synth::routing::route_placed_netlist;
use engine::tiles::TileType;

fn main() {
    let instances = hls_instances();
    let inst = instances
        .iter()
        .find(|i| i.circuit.name == "hls_mix6")
        .expect("hls_mix6 in corpus");
    let c = &inst.circuit;

    // Deterministic repro: same env defaults + SA seed as the corpus example.
    let mut env = PlacementEnv::new(c).expect("env");
    let r = anneal(
        &mut env,
        &AnnealConfig {
            iterations: 2000,
            ..AnnealConfig::default()
        },
    );
    println!(
        "hls_mix6: gates={} routable={} verify_physical={}",
        c.num_gates(),
        r.best.metrics.routable,
        env.verify_physical()
    );

    // Re-route the SA-best layout ourselves for introspection (same config as env).
    let placed = env.placed();
    let route_config = RouteConfig {
        prefer_horizontal_first: true,
        no_crossings: true,
        max_z: 3,
    };
    let routed = route_placed_netlist(&c.netlist, &placed, &route_config).expect("routes");
    let mut export = export_to_simulation(&routed, &c.netlist);

    // --- A. failing (combo, output) pairs ---
    let n_in = c.num_inputs();
    let mask = (1u64 << n_in) - 1;
    let mut s = 0x0123_4567_89AB_CDEFu64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut combos: Vec<u64> = vec![0, mask];
    for _ in 0..2000 {
        combos.push(next() & mask);
    }
    let mut fails: Vec<(u64, usize)> = Vec::new();
    for &combo in &combos {
        let bits: Vec<bool> = (0..n_in).map(|i| (combo >> i) & 1 == 1).collect();
        let want = evaluate_aig(&c.aig, &bits);
        let nl = evaluate_netlist(&c.netlist, &c.lib, &bits);
        assert_eq!(want, nl, "netlist itself diverges at {combo:#x}?!");
        let got = evaluate_exported(&mut export, &bits);
        for (o, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            if w != g {
                fails.push((combo, o));
            }
        }
    }
    let bad_outputs: BTreeSet<usize> = fails.iter().map(|&(_, o)| o).collect();
    println!(
        "\nA. {} failing (combo,out) pairs over {} combos; distinct bad outputs: {:?}",
        fails.len(),
        combos.len(),
        bad_outputs
    );
    for &(combo, o) in fails.iter().take(8) {
        println!("   combo={combo:#07x} out={o}");
    }

    // --- B1. cells shared by >1 net ---
    let mut cell: HashMap<(usize, usize, usize), Vec<(u32, TileType)>> = HashMap::new();
    for seg in &routed.routes {
        cell.entry((seg.x, seg.y, seg.z))
            .or_default()
            .push((seg.net_id, seg.tile_type));
    }
    let mut shared = 0usize;
    for ((x, y, z), v) in &cell {
        let nets: BTreeSet<u32> = v.iter().map(|e| e.0).collect();
        if nets.len() > 1 {
            shared += 1;
            if shared <= 12 {
                println!("B1. CELL SHARED ({x},{y},{z}): {v:?}");
            }
        }
    }
    println!("B1. cells shared by >1 net: {shared}");

    // --- B2. via columns used by >1 net ---
    let mut viacol: HashMap<(usize, usize), BTreeSet<u32>> = HashMap::new();
    for seg in &routed.routes {
        if format!("{:?}", seg.tile_type).contains("Via") {
            viacol.entry((seg.x, seg.y)).or_default().insert(seg.net_id);
        }
    }
    let multi: Vec<_> = viacol.iter().filter(|(_, nets)| nets.len() > 1).collect();
    println!("\nB2. via columns used by >1 net: {}", multi.len());
    for ((x, y), nets) in multi.iter().take(10) {
        println!("   VIA COLUMN ({x},{y}) nets={nets:?} — segments:");
        for seg in routed.routes.iter().filter(|s| s.x == *x && s.y == *y) {
            println!("     net {:>3} z={} {:?}", seg.net_id, seg.z, seg.tile_type);
        }
    }

    // --- B3. route cells colliding with gate tiles ---
    let gate_at: HashMap<(usize, usize, usize), usize> = routed
        .gates
        .iter()
        .map(|g| ((g.x, g.y, g.z), g.gate_idx))
        .collect();
    let mut gate_hits = 0usize;
    for seg in &routed.routes {
        if let Some(gidx) = gate_at.get(&(seg.x, seg.y, seg.z)) {
            gate_hits += 1;
            if gate_hits <= 10 {
                println!(
                    "B3. ROUTE-OVER-GATE ({},{},{}): net {} {:?} on gate[{}]",
                    seg.x, seg.y, seg.z, seg.net_id, seg.tile_type, gidx
                );
            }
        }
    }
    println!("\nB3. route cells colliding with gate tiles: {gate_hits}");

    // --- C. per-tile truth on one failing combo ---
    // Compute every net's expected boolean value, evaluate the export on the
    // failing combo, then flag tiles whose logic disagrees with their net's
    // value. The earliest wrong tile (topologically) is the defect site.
    let Some(&(combo, bad_out)) = fails.first() else {
        println!("\nC. no failing combo to dissect");
        return;
    };
    println!("\nC. dissecting combo={combo:#07x} (bad output {bad_out})");
    let bits: Vec<bool> = (0..n_in).map(|i| (combo >> i) & 1 == 1).collect();

    // Per-net expected values (replicates mapping::evaluate_netlist internals;
    // the netlist is inversion-materialized so the flags are all false).
    let num_pi = c.netlist.primary_inputs.len();
    let mut net_values = vec![false; c.netlist.num_nets as usize];
    net_values[..n_in].copy_from_slice(&bits);
    for (gate_idx, gate) in c.netlist.gates.iter().enumerate() {
        let cellinfo = c.lib.cell(gate.cell_idx);
        let mut assignment = 0u32;
        for (pin_idx, &fanin_net) in gate.fanins.iter().enumerate() {
            if net_values[fanin_net as usize] {
                assignment |= 1 << pin_idx;
            }
        }
        net_values[num_pi + gate_idx] = (cellinfo.truth_table >> assignment) & 1 != 0;
    }

    let _ = evaluate_exported(&mut export, &bits);
    println!(
        "   converged={} rounds={}",
        export.last_converged, export.last_convergence_rounds
    );

    // Gate output tiles vs net values.
    let mut wrong_gates = Vec::new();
    for g in &routed.gates {
        let actual = export.sim.get_logic_at_3d(g.x, g.y, g.z) != 0;
        let expect = net_values[num_pi + g.gate_idx];
        if actual != expect {
            wrong_gates.push((g.gate_idx, g.x, g.y, g.z, expect, actual));
        }
    }
    println!("   wrong GATE outputs: {}", wrong_gates.len());
    for (gi, x, y, z, e, a) in wrong_gates.iter().take(10) {
        println!("     gate[{gi}] at ({x},{y},{z}) expect {e} got {a}");
    }

    // Route segment tiles vs net values.
    let mut wrong_segs: Vec<(u32, usize, usize, usize, TileType)> = Vec::new();
    for seg in &routed.routes {
        let actual = export.sim.get_logic_at_3d(seg.x, seg.y, seg.z) != 0;
        let expect = net_values[seg.net_id as usize];
        if actual != expect {
            wrong_segs.push((seg.net_id, seg.x, seg.y, seg.z, seg.tile_type));
        }
    }
    println!("   wrong ROUTE tiles: {}", wrong_segs.len());

    // --- C1. classify wrong gates: root (all pins read correctly) vs cascade ---
    use engine::synth::PinSide;
    let side = |x: usize, y: usize, s: PinSide| -> (usize, usize) {
        match s {
            PinSide::Left => (x - 1, y),
            PinSide::Right => (x + 1, y),
            PinSide::Up => (x, y - 1),
            PinSide::Down => (x, y + 1),
        }
    };
    println!("\nC1. wrong-gate pin analysis (root vs cascade):");
    for &(gi, gx, gy, gz, _e, _a) in &wrong_gates {
        let g = routed.gates.iter().find(|g| g.gate_idx == gi).unwrap();
        let fanins = &c.netlist.gates[gi].fanins;
        let mut any_bad_pin = false;
        let mut pins = String::new();
        for (p, &net) in fanins.iter().enumerate() {
            let (px, py) = side(gx, gy, g.input_pins[p]);
            let actual = export.sim.get_logic_at_3d(px, py, gz) != 0;
            let expect = net_values[net as usize];
            if actual != expect {
                any_bad_pin = true;
            }
            pins.push_str(&format!(
                " pin{p}[{:?}@({px},{py})] net{net} expect {expect} got {actual};",
                g.input_pins[p]
            ));
        }
        println!(
            "   gate[{gi}] ({},{}) {} —{pins}",
            gx,
            gy,
            if any_bad_pin { "CASCADE" } else { "ROOT" }
        );
    }

    // --- C2. full chain trace of net 11 (the corrupted PI net) ---
    println!("\nC2. full route of net 11 (every segment, with neighbors):");
    for seg in routed.routes.iter().filter(|s| s.net_id == 11) {
        let v = export.sim.get_logic_at_3d(seg.x, seg.y, seg.z) != 0;
        let mut nb = String::new();
        for (dx, dy, dz, label) in [
            (-1i64, 0i64, 0i64, "L"),
            (1, 0, 0, "R"),
            (0, -1, 0, "U"),
            (0, 1, 0, "D"),
            (0, 0, -1, "below"),
            (0, 0, 1, "above"),
        ] {
            let (nx, ny, nz) = (seg.x as i64 + dx, seg.y as i64 + dy, seg.z as i64 + dz);
            if nx < 0
                || ny < 0
                || nz < 0
                || nx as usize >= export.sim.width()
                || ny as usize >= export.sim.height()
                || nz as usize >= export.sim.num_layers()
            {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            let tt = export.sim.tilemap.tiles[export.sim.tilemap.index_3d(nx, ny, nz).unwrap()]
                .meta
                .tile_type;
            if tt == TileType::Const {
                continue;
            }
            let nv = export.sim.get_logic_at_3d(nx, ny, nz) != 0;
            nb.push_str(&format!(" {label}={tt:?}:{}", nv as u8));
        }
        // What the export actually placed at this cell (may differ from seg type).
        let placed_tt = export.sim.tilemap.tiles
            [export.sim.tilemap.index_3d(seg.x, seg.y, seg.z).unwrap()]
        .meta
        .tile_type;
        println!(
            "   ({},{},{}) routed={:?} exported={:?} logic={} |{nb}",
            seg.x, seg.y, seg.z, seg.tile_type, placed_tt, v as u8
        );
    }
    let anchor = routed.input_anchors[11];
    println!(
        "   PI 11 anchor at ({},{}) logic={}",
        anchor.0,
        anchor.1,
        export.sim.get_logic_at(anchor.0, anchor.1) != 0
    );

    // Group by net for readability.
    let mut by_net: HashMap<u32, Vec<(u32, usize, usize, usize, TileType)>> = HashMap::new();
    for s in &wrong_segs {
        by_net.entry(s.0).or_default().push(*s);
    }
    let mut nets: Vec<_> = by_net.keys().copied().collect();
    nets.sort();
    for net in nets.iter().take(6) {
        let segs = &by_net[net];
        println!(
            "     net {} expect {}: {} wrong tiles, e.g.:",
            net,
            net_values[*net as usize],
            segs.len()
        );
        for &(_, x, y, z, tt) in segs.iter().take(6) {
            let v = export.sim.get_logic_at_3d(x, y, z);
            println!("       ({x},{y},{z}) {tt:?} logic={}", v != 0);
        }
        // Neighborhood dump around the first wrong tile of this net.
        let s0 = (segs[0].1, segs[0].2, segs[0].3);
        println!("       neighborhood of ({},{},{}):", s0.0, s0.1, s0.2);
        for (dx, dy, dz, label) in [
            (-1i64, 0i64, 0i64, "left "),
            (1, 0, 0, "right"),
            (0, -1, 0, "up   "),
            (0, 1, 0, "down "),
            (0, 0, -1, "below"),
            (0, 0, 1, "above"),
        ] {
            let (nx, ny, nz) = (s0.0 as i64 + dx, s0.1 as i64 + dy, s0.2 as i64 + dz);
            if nx < 0
                || ny < 0
                || nz < 0
                || nx as usize >= export.sim.width()
                || ny as usize >= export.sim.height()
                || nz as usize >= export.sim.num_layers()
            {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            let tt = export.sim.tilemap.tiles[export.sim.tilemap.index_3d(nx, ny, nz).unwrap()]
                .meta
                .tile_type;
            let v = export.sim.get_logic_at_3d(nx, ny, nz) != 0;
            // Which net (if any) owns this neighbor cell?
            let owner: Vec<u32> = routed
                .routes
                .iter()
                .filter(|r| r.x == nx && r.y == ny && r.z == nz)
                .map(|r| r.net_id)
                .collect();
            let gate = gate_at.get(&(nx, ny, nz));
            println!(
                "         {label} ({nx},{ny},{nz}) {tt:?} logic={v} route_nets={owner:?} gate={gate:?}"
            );
        }
    }
}
