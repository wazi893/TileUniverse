//! Milestone 2: Tile-Based Routing Substrate Demo
//!
//! Demonstrates SNN spike routing through physical tile substrate:
//! - Neurons mapped to 32×32 tile grid positions
//! - A*-routed paths between synapse endpoints
//! - Propagation delay = hop count through Wire tiles
//! - Differential arrival times from different distances
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example tile_routed_vision
//! ```

use engine::snn::router_mesh::RouterMesh;
use engine::snn::tile_substrate::NeuromorphicSubstrate;
use engine::snn::{NeuronConfig, SNNConfig, SNNNetwork, STDPConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 2: Tile-Based Routing Substrate");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Phase 1: Build RouterMesh ──────────────────────────────────────
    let n_inputs = 100;
    let n_hidden = 64;
    let n_outputs = 10;
    let total = n_inputs + n_hidden + n_outputs;

    println!(
        "Network: {} inputs, {} hidden, {} outputs ({} total)",
        n_inputs, n_hidden, n_outputs, total
    );
    println!("Grid: 32×32 tiles\n");

    let mut mesh = RouterMesh::new(32, 32);

    // Layer-clustered placement:
    //   inputs:  rows 0-3   (100 neurons, 32 cols → 4 rows)
    //   hidden:  rows 8-9   (64 neurons, 32 cols → 2 rows)
    //   outputs: row 14     (10 neurons)
    mesh.place_layer_clustered(&[
        (0, n_inputs, 0),
        (n_inputs, n_inputs + n_hidden, 8),
        (n_inputs + n_hidden, total, 14),
    ]);
    println!("Placed {} neurons in 32×32 grid.", mesh.neuron_count());

    // Build SNN to get synapse list.
    let config = SNNConfig {
        n_inputs,
        hidden_layers: vec![n_hidden],
        n_outputs,
        neurons_per_cpu: if total <= 512 { 512 } else { 1024 },
        neuron_config: NeuronConfig::default(),
        stdp_config: STDPConfig::default(),
        connection_prob: 0.3,
        ..Default::default()
    };
    let mut net = SNNNetwork::new(config);
    net.build_connectivity(42);

    // Collect synapse pairs (global src → global dst).
    let mut synapse_pairs: Vec<(usize, usize)> = Vec::new();
    for cpu in 0..net.topology.n_cpus {
        let (cpu_start, _) = net.topology.cpu_to_neurons[cpu];
        for local_src in 0..net.synapses[cpu].local.len() {
            let global_src = cpu_start + local_src;
            for syn in &net.synapses[cpu].local[local_src] {
                let global_dst = cpu_start + syn.target as usize;
                synapse_pairs.push((global_src, global_dst));
            }
        }
    }
    println!("Synapses: {} total connections", synapse_pairs.len());

    // Route all synapses through the mesh.
    let routed = mesh.route_all_synapses(&synapse_pairs);
    println!(
        "Routed: {} / {} synapse paths\n",
        routed,
        synapse_pairs.len()
    );

    // ── Phase 2: Routing statistics ────────────────────────────────────
    println!("── Routing Statistics ──");
    let hist = mesh.delay_histogram();
    let (cp_src, cp_dst, cp_delay) = mesh.critical_path();
    let total_delay: u64 = mesh.all_routes().values().map(|(h, _)| *h as u64).sum();
    let avg_delay = if routed > 0 {
        total_delay as f32 / routed as f32
    } else {
        0.0
    };

    println!(
        "  Min delay:      {} hops",
        hist.iter().position(|&c| c > 0).unwrap_or(0)
    );
    println!(
        "  Max delay:      {} hops (neuron {} → {})",
        cp_delay, cp_src, cp_dst
    );
    println!("  Avg delay:      {:.1} hops", avg_delay);
    println!(
        "  Critical path:  neuron {} → neuron {} ({} hops)",
        cp_src, cp_dst, cp_delay
    );

    println!("\n  Delay histogram:");
    let bin_size = 5;
    let max_d = hist.len();
    let mut bin_start = 0;
    while bin_start < max_d {
        let bin_end = (bin_start + bin_size).min(max_d);
        let count: u32 = hist[bin_start..bin_end].iter().sum();
        if count > 0 {
            let bar_len = (count as usize * 50) / routed.max(1);
            let bar: String = "█".repeat(bar_len);
            println!(
                "  {:>2}-{:<2}: {:>6} synapses  {}",
                bin_start,
                bin_end - 1,
                count,
                bar
            );
        }
        bin_start = bin_end;
    }

    // ── Phase 3: Physical propagation verification ─────────────────────
    println!("\n── Physical Propagation Verification ──");
    let mut sub = NeuromorphicSubstrate::new(mesh);

    // Pick 5 routed pairs with increasing A* delay to verify physical propagation.
    let mut test_pairs: Vec<(usize, usize, u8)> = sub
        .mesh
        .all_routes()
        .iter()
        .map(|(&(s, d), &(h, _))| (s, d, h))
        .collect();
    test_pairs.sort_by_key(|&(_, _, h)| h);

    // Sample at 20%, 40%, 60%, 80%, 100% of the sorted list.
    let samples: Vec<(usize, usize, u8)> = [20, 40, 60, 80, 99]
        .iter()
        .map(|&pct| {
            let idx = (pct * test_pairs.len() / 100).min(test_pairs.len() - 1);
            test_pairs[idx]
        })
        .collect();

    println!("  Physical delay vs A* delay (5 sampled pairs):");
    for &(src, dst, a_star_delay) in &samples {
        let physical_delay = sub.measure_delay(src, dst);
        let src_pos = sub.mesh.position(src).unwrap();
        let dst_pos = sub.mesh.position(dst).unwrap();
        println!(
            "    ({:>2},{:>2})→({:>2},{:>2}):  A*={:>2} hops  physical={:>2} deltas",
            src_pos.0, src_pos.1, dst_pos.0, dst_pos.1, a_star_delay, physical_delay
        );
    }

    // ── Phase 4: Apply delays to SNN ───────────────────────────────────
    println!("\n── SNN Delay Integration ──");
    net.apply_substrate_delays(&sub.mesh);

    // Show delay distribution on actual synapses.
    let mut delay_counts = [0u32; 16];
    for cpu in 0..net.topology.n_cpus {
        for local_src in 0..net.synapses[cpu].local.len() {
            for syn in &net.synapses[cpu].local[local_src] {
                delay_counts[syn.delay() as usize] += 1;
            }
        }
    }
    println!("  Synapse delay distribution after substrate calibration:");
    for d in 0..16 {
        if delay_counts[d] > 0 {
            println!("    delay={:>2}: {:>5} synapses", d, delay_counts[d]);
        }
    }

    let total_delays: u32 = delay_counts
        .iter()
        .enumerate()
        .map(|(d, &c)| d as u32 * c)
        .sum();
    let total_syns: u32 = delay_counts.iter().sum();
    let avg_syn_delay = if total_syns > 0 {
        total_delays as f32 / total_syns as f32
    } else {
        0.0
    };
    println!(
        "  Average synapse delay: {:.1} ticks ({} synapses)",
        avg_syn_delay, total_syns
    );

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results:");
    println!(
        "    ✓ {} neurons placed in 32×32 grid",
        sub.mesh.neuron_count()
    );
    println!(
        "    ✓ {} / {} synapse paths routed via A*",
        routed,
        synapse_pairs.len()
    );
    println!("    ✓ Critical path: {} hops", cp_delay);
    println!("    ✓ Average routing delay: {:.1} hops", avg_delay);
    println!(
        "    ✓ Substrate delays applied to SNN synapses (avg {:.1} ticks)",
        avg_syn_delay
    );
    println!("═══════════════════════════════════════════════════════════════");
}
