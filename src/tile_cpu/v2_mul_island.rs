//! Sprint 395: MUL island — iterative radix-2 shift-add multiplier as placed tiles.
//!
//! The last computation-authority gap (roadmap Rev 9) is MUL: `wrapping_mul` computed
//! in software and injected. This module places a physical shift-add datapath:
//!
//! ```text
//!   P' = mul_active ? P + (B&1 ? A : 0) : P      (accumulator)
//!   A' = mul_active ? A << 1            : A      (multiplicand)
//!   B' = mul_active ? B >> 1            : B      (multiplier)
//! ```
//!
//! One iteration per global clock cycle; 64 active cycles compute `P = A0 * B0`
//! (wrapping). Registers are permanent Register64 tiles (capture LEFT on the rising
//! edge); each register's next-value Mux sits at its LEFT with RIGHT reading the
//! register itself, so holds are free (the S125 LR-mux pattern) and un-clocked settle
//! is harmless by construction (`mul_active = 0` ⇒ every register recaptures itself).
//!
//! Layout is two-layer Manhattan: compute tiles + horizontal corridors on layer `zh`,
//! vertical runs on layer `zv = zh + 1`, via turns between them — horizontal/vertical
//! crossings cannot collide by construction. The placer panics on any cell reuse and
//! on any second via reading the same source (the one-via-per-source rule), and
//! `verify_structure` re-reads the grid to assert every consumer port cell holds the
//! tile the netlist expects.
//!
//! S395 scope: standalone (bare `Simulation`, no CPU integration — that is S396).
//! Deliberately NOT used: `TileType::Mul` (a single-tile `wrapping_mul` primitive
//! exists in the fabric; whether it counts as library-tier physical is a pending
//! user decision — see `User notes/WAKE_UP_2026-07-09.md`. This island satisfies the
//! stricter built-from-parts standard either way).

use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use std::collections::{HashMap, HashSet};

/// Placed MUL island: tile indices for the harness/integration.
pub struct MulIsland {
    pub p_idx: usize,
    pub a_idx: usize,
    pub b_idx: usize,
    /// The mul_active Const cell (write MAX to iterate, 0 to hold).
    pub act_idx: usize,
    /// Every placed tile (dirty-scope integration in S396).
    pub all_indices: Vec<usize>,
    /// (index, coords, expected TileType) for structural verification.
    checks: Vec<(usize, (usize, usize, usize), TileType)>,
}

/// Micro-placer: occupancy + via-source bookkeeping with panic-on-collision.
struct Plc<'a> {
    sim: &'a mut Simulation,
    used: HashSet<(usize, usize, usize)>,
    via_sources: HashMap<(usize, usize, usize), (usize, usize, usize)>,
    indices: Vec<usize>,
    checks: Vec<(usize, (usize, usize, usize), TileType)>,
}

impl<'a> Plc<'a> {
    fn new(sim: &'a mut Simulation) -> Self {
        Plc {
            sim,
            used: HashSet::new(),
            via_sources: HashMap::new(),
            indices: Vec::new(),
            checks: Vec::new(),
        }
    }

    fn put(&mut self, x: usize, y: usize, z: usize, tt: TileType) -> usize {
        assert!(
            self.used.insert((x, y, z)),
            "mul island placer: cell collision at ({x},{y},z{z}) placing {tt:?}"
        );
        // Site must be virgin fabric (default Wire) — protects CPU integration from
        // silently overwriting existing structures (S392 occupancy-dump discipline,
        // enforced mechanically).
        let existing = self.sim.tile_type_3d(x, y, z);
        assert_eq!(
            existing,
            TileType::Wire,
            "mul island placer: site ({x},{y},z{z}) already holds {existing:?}"
        );
        // One-via-per-source: ViaUp at z reads (x,y,z+1); ViaDown at z reads (x,y,z-1).
        let source = match tt {
            TileType::ViaUp => Some((x, y, z + 1)),
            TileType::ViaDown => Some((x, y, z - 1)),
            _ => None,
        };
        if let Some(src) = source {
            assert!(
                self.via_sources.insert(src, (x, y, z)).is_none(),
                "mul island placer: second via at ({x},{y},z{z}) reading source {src:?}"
            );
        }
        self.sim.set_tile_3d(x, y, z, tt);
        let idx = z * self.sim.width() * self.sim.height() + y * self.sim.width() + x;
        self.indices.push(idx);
        self.checks.push((idx, (x, y, z), tt));
        idx
    }

    fn put_val(&mut self, x: usize, y: usize, z: usize, tt: TileType, v: u64) -> usize {
        let idx = self.put(x, y, z, tt);
        self.sim.set_logic_value_by_idx(idx, v);
        idx
    }

    /// Horizontal chain of directional wires on zh, inclusive columns.
    fn hchain(&mut self, x0: usize, x1: usize, y: usize, z: usize, tt: TileType) {
        let (lo, hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        for x in lo..=hi {
            self.put(x, y, z, tt);
        }
    }

    /// Vertical chain of directional wires, inclusive rows.
    fn vchain(&mut self, x: usize, y0: usize, y1: usize, z: usize, tt: TileType) {
        let (lo, hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        for y in lo..=hi {
            self.put(x, y, z, tt);
        }
    }
}

/// Place the island with its north-west corner near `(ox, oy)` on layer `zh`
/// (vertical routing uses `zh + 1`). Needs ~30 columns × ~15 rows × 2 layers.
pub fn place_mul_island(sim: &mut Simulation, ox: usize, oy: usize, zh: usize) -> MulIsland {
    let zv = zh + 1;
    let mut p = Plc::new(sim);

    // Row oy+1: mul_active source + rail (only Mux tiles read UP; the rail passing
    // over non-Mux row-2 tiles is inert).
    let act_idx = p.put_val(ox, oy + 1, zh, TileType::Const, 0);
    p.hchain(ox + 1, ox + 24, oy + 1, zh, TileType::WireRight);

    // Row oy+2: [Pn][P] .. [An][A] .. [Bn][B] + B fanout + BitSelect(b0).
    // Register64 captures LEFT on the rising edge; next-value Mux RIGHT reads the
    // register itself (free hold).
    let sum_in = p.put(ox + 3, oy + 2, zh, TileType::WireUp); // sum arrival -> Pn.LEFT
    p.put(ox + 4, oy + 2, zh, TileType::Mux); // Pn: UP=act, LEFT=sum, RIGHT=P
    let p_idx = p.put_val(ox + 5, oy + 2, zh, TileType::Register64, 0);
    p.put(ox + 6, oy + 2, zh, TileType::WireRight); // P fanout (reads P)

    let shla_in = p.put(ox + 13, oy + 2, zh, TileType::WireUp); // shlA arrival -> An.LEFT
    p.put(ox + 14, oy + 2, zh, TileType::Mux); // An
    let a_idx = p.put_val(ox + 15, oy + 2, zh, TileType::Register64, 0);
    p.put(ox + 16, oy + 2, zh, TileType::WireRight); // A fanout
    p.put(ox + 17, oy + 2, zh, TileType::WireRight); // A fanout (cross-band tap)

    let shrb_in = p.put(ox + 23, oy + 2, zh, TileType::WireUp); // shrB arrival -> Bn.LEFT
    p.put(ox + 24, oy + 2, zh, TileType::Mux); // Bn
    let b_idx = p.put_val(ox + 25, oy + 2, zh, TileType::Register64, 0);
    p.put(ox + 26, oy + 2, zh, TileType::WireRight); // B fanout
    let bitsel = p.put(ox + 27, oy + 2, zh, TileType::BitSelect); // b0 = B bit 0
    p.put_val(ox + 28, oy + 2, zh, TileType::Const, 0); // bit index 0
    let _ = bitsel;
    let _ = (sum_in, shla_in, shrb_in);

    // Row oy+3 (band-local): register taps + shifters.
    p.put(ox + 15, oy + 3, zh, TileType::WireDown); // A tap (reads A above)
    p.put(ox + 16, oy + 3, zh, TileType::Shl); // shlA: LEFT=A tap, RIGHT=Const 1
    p.put_val(ox + 17, oy + 3, zh, TileType::Const, 1);
    p.put(ox + 25, oy + 3, zh, TileType::WireDown); // B tap
    p.put(ox + 26, oy + 3, zh, TileType::Shr); // shrB: LEFT=B tap, RIGHT=Const 1
    p.put_val(ox + 27, oy + 3, zh, TileType::Const, 1);

    // shlA -> An.LEFT: via down at (16,3), zv down to row 4, up onto zh row 4,
    // west to col 13, zv rise to row 3, zh WireUp into (13,2).
    p.put(ox + 16, oy + 3, zv, TileType::ViaDown); // reads shlA on zh
    p.put(ox + 16, oy + 4, zv, TileType::WireDown);
    p.put(ox + 16, oy + 4, zh, TileType::ViaUp); // shlA on zh row 4
    p.hchain(ox + 13, ox + 15, oy + 4, zh, TileType::WireLeft);
    p.put(ox + 13, oy + 4, zv, TileType::ViaDown); // reads (13,4,zh)
    p.put(ox + 13, oy + 3, zv, TileType::WireUp);
    p.put(ox + 13, oy + 3, zh, TileType::ViaUp); // shlA at (13,3,zh); (13,2) WireUp reads it

    // shrB -> Bn.LEFT: same shape at columns 26 -> 23.
    p.put(ox + 26, oy + 3, zv, TileType::ViaDown);
    p.put(ox + 26, oy + 4, zv, TileType::WireDown);
    p.put(ox + 26, oy + 4, zh, TileType::ViaUp);
    p.hchain(ox + 23, ox + 25, oy + 4, zh, TileType::WireLeft);
    p.put(ox + 23, oy + 4, zv, TileType::ViaDown);
    p.put(ox + 23, oy + 3, zv, TileType::WireUp);
    p.put(ox + 23, oy + 3, zh, TileType::ViaUp);

    // P -> Add.RIGHT: tap fanout (6,2) down zv col 6 to row 13, up onto zh.
    p.put(ox + 6, oy + 2, zv, TileType::ViaDown); // reads P fanout wire
    p.vchain(ox + 6, oy + 3, oy + 13, zv, TileType::WireDown);
    p.put(ox + 6, oy + 13, zh, TileType::ViaUp); // P at (6,13,zh)

    // A -> addend Mux LEFT: tap (17,2) down zv col 17 to row 15 (south of the whole
    // compute block, clear of b0's row-11 landing), west on zh row 15 to col 3,
    // rise on zv col 3 to row 12, up onto zh at (3,12).
    p.put(ox + 17, oy + 2, zv, TileType::ViaDown);
    p.vchain(ox + 17, oy + 3, oy + 15, zv, TileType::WireDown);
    p.put(ox + 17, oy + 15, zh, TileType::ViaUp);
    p.hchain(ox + 3, ox + 16, oy + 15, zh, TileType::WireLeft);
    p.put(ox + 3, oy + 15, zv, TileType::ViaDown);
    p.vchain(ox + 3, oy + 12, oy + 14, zv, TileType::WireUp);
    p.put(ox + 3, oy + 12, zh, TileType::ViaUp); // A at (3,12,zh)

    // b0 -> addend Mux UP: BitSelect (27,2) down zv col 27 to row 10, west on zh
    // row 10 to col 4, down zv to row 11, up onto zh at (4,11).
    p.put(ox + 27, oy + 2, zv, TileType::ViaDown);
    p.vchain(ox + 27, oy + 3, oy + 10, zv, TileType::WireDown);
    p.put(ox + 27, oy + 10, zh, TileType::ViaUp);
    p.hchain(ox + 4, ox + 26, oy + 10, zh, TileType::WireLeft);
    p.put(ox + 4, oy + 10, zv, TileType::ViaDown);
    p.put(ox + 4, oy + 11, zv, TileType::WireDown);
    p.put(ox + 4, oy + 11, zh, TileType::ViaUp); // b0 at (4,11,zh)

    // addend Mux at (4,12): UP=b0, LEFT=A, RIGHT=Const 0. (b0 set -> take A.)
    p.put(ox + 4, oy + 12, zh, TileType::Mux);
    p.put_val(ox + 5, oy + 12, zh, TileType::Const, 0);

    // Add at (5,13): LEFT=(4,13)=addend (WireDown reads the mux), RIGHT=(6,13)=P.
    p.put(ox + 4, oy + 13, zh, TileType::WireDown);
    p.put(ox + 5, oy + 13, zh, TileType::Add);

    // sum -> Pn.LEFT: exit south (5,14), rise on zv col 5 to row 3, onto zh row 3,
    // west to col 3, WireUp into (3,2).
    p.put(ox + 5, oy + 14, zh, TileType::WireDown); // reads the Add
    p.put(ox + 5, oy + 14, zv, TileType::ViaDown);
    p.vchain(ox + 5, oy + 3, oy + 13, zv, TileType::WireUp);
    p.put(ox + 5, oy + 3, zh, TileType::ViaUp);
    p.hchain(ox + 3, ox + 4, oy + 3, zh, TileType::WireLeft);

    let all_indices = p.indices.clone();
    let checks = p.checks.clone();
    // Via dirty-forwarding (via_fwd) is a rebuilt lookup, not automatic: without
    // this, a source tile's change never marks the via that reads it, and the
    // cross-layer paths go stale after the first settle (observed: A-riser landing
    // dead at fixpoint, zv descents stale after clock cycles).
    sim.rebuild_via_connections();
    MulIsland {
        p_idx,
        a_idx,
        b_idx,
        act_idx,
        all_indices,
        checks,
    }
}

impl MulIsland {
    /// Mark every island tile dirty. Required after ANY direct `set_logic_value`
    /// write (register load, act toggle): the delay-tick evaluator is dirty-driven
    /// and direct writes don't cascade (the "software-written Const" house rule) —
    /// without this, the combinational network never evaluates and the registers
    /// capture stale zeros.
    pub fn mark_all_dirty(&self, sim: &Simulation) {
        for &idx in &self.all_indices {
            sim.dirty.mark_dirty(idx);
        }
    }

    /// Re-read the grid and assert every placed cell still holds its tile type.
    pub fn verify_structure(&self, sim: &Simulation) {
        for (idx, (x, y, z), tt) in &self.checks {
            assert_eq!(
                sim.tile_type_3d(*x, *y, *z),
                *tt,
                "mul island structure check failed at ({x},{y},z{z}) idx {idx}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pump_cycle(sim: &mut Simulation) {
        // One full clock cycle = two edges (tick_with_delays toggles per call);
        // Register64 captures on the rising edge from the settled pre-edge values.
        sim.tick_with_delays();
        sim.tick_with_delays();
    }

    /// Settle combinational logic to fixpoint WITHOUT a clock edge. Required after
    /// direct writes: the first pump edge is a RISING edge and registers capture at
    /// delta 0 — before dirty-marked combinational tiles evaluate — so an unsettled
    /// network zeroes the registers (observed in the S395 debug dump).
    fn settle(sim: &mut Simulation) {
        for _ in 0..500 {
            if sim.propagate_combinational() == 0 {
                break;
            }
        }
    }

    fn run_mul(sim: &mut Simulation, isl: &MulIsland, a: u64, b: u64) -> u64 {
        sim.set_logic_value_by_idx(isl.act_idx, 0);
        sim.set_logic_value_by_idx(isl.p_idx, 0);
        sim.set_logic_value_by_idx(isl.a_idx, a);
        sim.set_logic_value_by_idx(isl.b_idx, b);
        isl.mark_all_dirty(sim);
        settle(sim);
        sim.set_logic_value_by_idx(isl.act_idx, u64::MAX);
        isl.mark_all_dirty(sim);
        settle(sim);
        for _ in 0..64 {
            pump_cycle(sim);
        }
        sim.set_logic_value_by_idx(isl.act_idx, 0);
        isl.mark_all_dirty(sim);
        settle(sim);
        sim.get_logic_value_by_idx(isl.p_idx)
    }

    #[test]
    fn test_mul_island_place_and_verify() {
        let mut sim = Simulation::with_size_layered(64, 64, 8);
        let isl = place_mul_island(&mut sim, 2, 2, 3);
        isl.verify_structure(&sim);
        assert!(isl.all_indices.len() > 50);
    }

    #[test]
    fn test_mul_island_debug_dump() {
        let mut sim = Simulation::with_size_layered(64, 64, 8);
        let isl = place_mul_island(&mut sim, 2, 2, 3);
        sim.initialize();
        let (ox, oy, zh, zv) = (2usize, 2usize, 3usize, 4usize);
        let g = |sim: &Simulation, dx: usize, dy: usize, z: usize| {
            let idx = z * 64 * 64 + (oy + dy) * 64 + (ox + dx);
            sim.get_logic_value_by_idx(idx)
        };
        sim.set_logic_value_by_idx(isl.act_idx, 0);
        sim.set_logic_value_by_idx(isl.p_idx, 0);
        sim.set_logic_value_by_idx(isl.a_idx, 3);
        sim.set_logic_value_by_idx(isl.b_idx, 5);
        isl.mark_all_dirty(&sim);
        settle(&mut sim);
        let dump = |sim: &Simulation, label: &str| {
            eprintln!(
                "[{label}] act_rail(4,1)={} (14,1)={} | A_tap(15,3)={} shlA(16,3)={} shlA_arr(13,2)={} \
                 | B_tap(25,3)={} shrB(26,3)={} shrB_arr(23,2)={} | Bfan(26,2)={} b0(27,2)={} \
                 | b0_zv_top(27,3,zv)={} b0_row10(26,10)={} b0_row10w(4,10)={} b0_arr(4,11)={} \
                 | Afan(16,2)={} Afan2(17,2)={} A_zv(17,3,zv)={} A_row15(16,15)={} A_row15w(3,15)={} A_arr(3,12)={} \
                 | addend(4,12)={} addend_dn(4,13)={} | Pfan(6,2)={} P_zv(6,3,zv)={} P_arr(6,13)={} \
                 | ADD(5,13)={} sum_dn(5,14)={} sum_zv(5,13,zv)={} sum_row3(5,3)={} sum_w(3,3)={} sum_arr(3,2)={} \
                 | Pn(4,2)={} An(14,2)={} Bn(24,2)={}",
                g(sim, 4, 1, zh),
                g(sim, 14, 1, zh),
                g(sim, 15, 3, zh),
                g(sim, 16, 3, zh),
                g(sim, 13, 2, zh),
                g(sim, 25, 3, zh),
                g(sim, 26, 3, zh),
                g(sim, 23, 2, zh),
                g(sim, 26, 2, zh),
                g(sim, 27, 2, zh),
                g(sim, 27, 3, zv),
                g(sim, 26, 10, zh),
                g(sim, 4, 10, zh),
                g(sim, 4, 11, zh),
                g(sim, 16, 2, zh),
                g(sim, 17, 2, zh),
                g(sim, 17, 3, zv),
                g(sim, 16, 15, zh),
                g(sim, 3, 15, zh),
                g(sim, 3, 12, zh),
                g(sim, 4, 12, zh),
                g(sim, 4, 13, zh),
                g(sim, 6, 2, zh),
                g(sim, 6, 3, zv),
                g(sim, 6, 13, zh),
                g(sim, 5, 13, zh),
                g(sim, 5, 14, zh),
                g(sim, 5, 13, zv),
                g(sim, 5, 3, zh),
                g(sim, 3, 3, zh),
                g(sim, 3, 2, zh),
                g(sim, 4, 2, zh),
                g(sim, 14, 2, zh),
                g(sim, 24, 2, zh),
            );
        };
        // First three args of dump's format are P/A/B via direct idx reads:
        eprintln!(
            "P={} A={} B={}",
            sim.get_logic_value_by_idx(isl.p_idx),
            sim.get_logic_value_by_idx(isl.a_idx),
            sim.get_logic_value_by_idx(isl.b_idx)
        );
        dump(&sim, "after settle");
        sim.set_logic_value_by_idx(isl.act_idx, u64::MAX);
        isl.mark_all_dirty(&sim);
        settle(&mut sim);
        pump_cycle(&mut sim);
        eprintln!(
            "P={} A={} B={}",
            sim.get_logic_value_by_idx(isl.p_idx),
            sim.get_logic_value_by_idx(isl.a_idx),
            sim.get_logic_value_by_idx(isl.b_idx)
        );
        dump(&sim, "after 1 active cycle");
    }

    #[test]
    fn test_mul_island_3x5_trace() {
        let mut sim = Simulation::with_size_layered(64, 64, 8);
        let isl = place_mul_island(&mut sim, 2, 2, 3);
        sim.initialize();
        // 3 * 5: after i iterations P holds the partial product of the low i bits.
        sim.set_logic_value_by_idx(isl.act_idx, 0);
        sim.set_logic_value_by_idx(isl.p_idx, 0);
        sim.set_logic_value_by_idx(isl.a_idx, 3);
        sim.set_logic_value_by_idx(isl.b_idx, 5);
        isl.mark_all_dirty(&sim);
        settle(&mut sim);
        sim.set_logic_value_by_idx(isl.act_idx, u64::MAX);
        isl.mark_all_dirty(&sim);
        settle(&mut sim);
        let mut expect_p = 0u64;
        let mut ea = 3u64;
        let mut eb = 5u64;
        for i in 0..4 {
            pump_cycle(&mut sim);
            if eb & 1 == 1 {
                expect_p = expect_p.wrapping_add(ea);
            }
            ea <<= 1;
            eb >>= 1;
            assert_eq!(
                sim.get_logic_value_by_idx(isl.p_idx),
                expect_p,
                "P wrong after iteration {i}"
            );
            assert_eq!(sim.get_logic_value_by_idx(isl.a_idx), ea, "A after {i}");
            assert_eq!(sim.get_logic_value_by_idx(isl.b_idx), eb, "B after {i}");
        }
        assert_eq!(expect_p, 15);
    }

    /// S396 I1 probe: the island placed by the V2 builder, integrated into the
    /// commit/clock scopes at wiring time, iterates once per CPU cycle when act is
    /// raised — driven entirely by the CPU's own settle + clock-edge machinery.
    #[test]
    fn test_mul_island_iterates_in_v2_cpu() {
        use crate::tile_cpu::v2_components::v2_nop;
        use crate::tile_cpu::{V2Builder, V2SynthConfig};
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let prog = vec![v2_nop(); 8];
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_wide_pc()
            .with_mul_island()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let (p_idx, a_idx, b_idx, act_idx) = cpu.mul_island_regs().unwrap();
        let mark = |sim: &Simulation, cpu: &crate::tile_cpu::TileCpuV2| {
            for &idx in cpu.mul_island_indices() {
                sim.dirty.mark_dirty(idx);
            }
        };
        // Load operands directly (I2 delivers them from the physical register
        // tiles); direct writes need the island marked dirty (house rule).
        sim.set_logic_value_by_idx(p_idx, 0);
        sim.set_logic_value_by_idx(a_idx, 3);
        sim.set_logic_value_by_idx(b_idx, 5);
        sim.set_logic_value_by_idx(act_idx, 0);
        mark(&sim, &cpu);
        cpu.run(&mut sim, 2);
        assert_eq!(sim.get_logic_value_by_idx(a_idx), 3, "hold under act=0");
        assert_eq!(sim.get_logic_value_by_idx(b_idx), 5, "hold under act=0");
        sim.set_logic_value_by_idx(act_idx, u64::MAX);
        mark(&sim, &cpu);
        cpu.run(&mut sim, 4); // 4 CPU cycles = 4 island iterations (3*5 needs 3)
        assert_eq!(
            sim.get_logic_value_by_idx(p_idx),
            15,
            "P after 4 iterations"
        );
        assert_eq!(sim.get_logic_value_by_idx(a_idx), 48, "A after 4 shifts");
        assert_eq!(sim.get_logic_value_by_idx(b_idx), 0, "B exhausted");
        sim.set_logic_value_by_idx(act_idx, 0);
        mark(&sim, &cpu);
        cpu.run(&mut sim, 2);
        assert_eq!(sim.get_logic_value_by_idx(p_idx), 15, "hold after done");
    }

    /// S396 I2: island-mode MUL executes end-to-end — the physical island computes
    /// the product over 64 real clock edges (MUL = 65-cycle instruction), the
    /// result lands in rd through the normal writeback, and the trust-verify
    /// fallback never fires (`mul_assists == 0`). Covers both register banks and
    /// a staleness case (rs rewritten immediately before MUL).
    #[test]
    fn test_mul_island_execute_in_v2_cpu() {
        use crate::tile_cpu::v2_components::{
            v2_add, v2_halt, v2_ldi, v2_mov, v2_mul, v2_with_reg_ext,
        };
        use crate::tile_cpu::{V2Builder, V2SynthConfig};
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let mut prog = vec![
            v2_ldi(1, 7),                               // R1 = 7
            v2_ldi(2, 9),                               // R2 = 9
            v2_mul(1, 2),                               // R1 = 63  (65-cycle island MUL)
            v2_ldi(3, 20),                              // R3 = 20
            v2_add(3, 3),                               // R3 = 40
            v2_with_reg_ext(v2_mov(0, 3), true, false), // R8 = 40 (high bank dest)
            v2_ldi(4, 5),                               // staleness: R4 rewritten right before use
            v2_ldi(4, 6),                               // R4 = 6
            v2_with_reg_ext(v2_mul(0, 4), true, false), // R8 = 40*6 = 240 (rd_hi)
            v2_halt(),
        ];
        prog.resize(16, 0);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_wide_pc()
            .with_mul_island()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        cpu.run(&mut sim, 400);
        assert!(cpu.is_halted(), "program should halt");
        assert_eq!(cpu.read_reg(&sim, 1), 63, "R1 = 7*9 via the island");
        assert_eq!(
            cpu.read_reg(&sim, 8),
            240,
            "R8 = 40*6 via the island (rd_hi)"
        );
        assert_eq!(cpu.mul_assists(), 0, "physical product must match software");
        assert!(
            cpu.read_cycle_count() >= 130,
            "two island MULs must stall (~65 cycles each); got {}",
            cpu.read_cycle_count()
        );
    }

    /// S397: value sweep through the ISA (wide immediates, chained squares, the
    /// zero product + Z-flag path) with an ISS END-STATE differential — island-mode
    /// MUL is a 65-cycle instruction, so cycle counts differ from the 1-cycle ISS
    /// by design; final architectural state must match exactly.
    #[test]
    fn test_v2_s397_mul_island_sweep_iss_end_state() {
        use crate::tile_cpu::v2_components::{v2_halt, v2_ldi, v2_ldi_w, v2_movz, v2_mul, v2_nop};
        use crate::tile_cpu::v2_iss::V2Iss;
        use crate::tile_cpu::{V2Builder, V2SynthConfig};
        let mut prog = vec![
            v2_ldi_w(1, 0x1234), // R1 = 0x1234
            v2_ldi_w(2, 0x00FF), // R2 = 255
            v2_mul(1, 2),        // R1 = 0x1234 * 255
            v2_mul(1, 1),        // R1 = R1^2 (large 64-bit product)
            v2_ldi(3, 0),
            v2_mul(3, 2),  // R3 = 0 * 255 = 0 -> Z set by the island result
            v2_movz(5, 2), // Z==1 -> R5 = R2 (flag path downstream of island MUL)
            v2_ldi(6, 255),
            v2_mul(6, 6), // R6 = 255*255 = 65025
            v2_halt(),
        ];
        prog.resize(16, v2_nop());

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_wide_pc()
            .with_mul_island()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        cpu.run(&mut sim, 600); // 4 island MULs ~= 260 stall cycles + overhead
        assert!(cpu.is_halted(), "physical run should halt");

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_wide_pc();
        for _ in 0..1000 {
            if iss.is_halted() {
                break;
            }
            iss.step();
        }
        assert!(iss.is_halted(), "ISS run should halt");

        for r in 0..16 {
            assert_eq!(
                cpu.read_reg(&sim, r),
                iss.state().regs[r],
                "end-state mismatch at R{r}"
            );
        }
        assert_eq!(cpu.read_reg(&sim, 5), 255, "MOVZ after zero product took Z");
        assert_eq!(cpu.mul_assists(), 0, "no fallbacks across the sweep");
    }

    /// S397 corpus: compiled recursive `fact(5)` — the source literally contains
    /// `return n * fact(n - 1)`, so under island mode this one program exercises
    /// compiled recursion returns (S393 authority) AND four island MULs together.
    #[test]
    fn test_v2_s397_compiled_fact_mul_island() {
        use crate::tile_cpu::v2_compiler::{
            Func, Program, Stmt, call, compile_to_words, int, lt, mul, sub, var,
        };
        use crate::tile_cpu::v2_components::v2_nop;
        use crate::tile_cpu::v2_iss::V2Iss;
        use crate::tile_cpu::{V2Builder, V2SynthConfig};

        let fact = Func {
            name: "fact".to_string(),
            params: vec!["n".to_string()],
            body: vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(int(1))],
                vec![Stmt::Return(mul(
                    var("n"),
                    call("fact", vec![sub(var("n"), int(1))]),
                ))],
            )],
        };
        let main = Func {
            name: "main".to_string(),
            params: vec![],
            body: vec![Stmt::Return(call("fact", vec![int(5)]))],
        };
        let program = Program::new("main", vec![main, fact]);
        let mut prog = compile_to_words(&program).expect("compile failed");
        prog.resize(prog.len().next_power_of_two().max(128), v2_nop());

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_wide_pc()
            .with_mul_island()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        cpu.run(&mut sim, 3000); // 4 island MULs + recursion overhead
        assert!(cpu.is_halted(), "compiled fact should halt");
        assert_eq!(cpu.read_reg(&sim, 0), 120, "fact(5) via island MULs");

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_wide_pc();
        for _ in 0..5000 {
            if iss.is_halted() {
                break;
            }
            iss.step();
        }
        assert!(iss.is_halted());
        assert_eq!(iss.state().regs[0], 120);
        assert_eq!(cpu.mul_assists(), 0, "all four products computed in tiles");
    }

    #[test]
    fn test_mul_island_battery_vs_wrapping_mul() {
        let mut sim = Simulation::with_size_layered(64, 64, 8);
        let isl = place_mul_island(&mut sim, 2, 2, 3);
        sim.initialize();
        let cases: [(u64, u64); 10] = [
            (0, 0),
            (0, 12345),
            (1, u64::MAX),
            (2, 0x8000_0000_0000_0001),
            (5, 7),
            (u64::MAX, u64::MAX),
            (0xDEAD_BEEF_CAFE_F00D, 0x1234_5678_9ABC_DEF0),
            (0x8000_0000_0000_0000, 3),
            (65535, 65537),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
        ];
        for (a, b) in cases {
            let got = run_mul(&mut sim, &isl, a, b);
            assert_eq!(
                got,
                a.wrapping_mul(b),
                "island mul mismatch for {a:#x} * {b:#x}"
            );
        }
    }
}
