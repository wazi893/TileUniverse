//! NeuroAlphaFabric — the learned EDA placer optimizing a SPIKING NEURON's layout.
//!
//! Where the AlphaFabric timing demo optimized adders and a multiplier, this does
//! the same for a leaky integrate-and-fire neuron's next-state circuit (the
//! synaptic sum + membrane datapath that computes in tiles). It reports the
//! layout quality — total wirelength (HPWL) and the criticality-weighted
//! wirelength (the timing objective) — under the deterministic baseline
//! (row-major) placement vs the route-validated, timing-aware simulated-annealing
//! placer, and confirms the optimized layout is still physically correct on tiles.
//!
//! This is where the two tracks meet: a learned/optimized physical-design system
//! laying out a neuromorphic circuit, with functional equivalence proven by the
//! tile==reference oracle.
//!
//! Run: `cargo run --release --example snn_neuro_alphafabric`

use engine::snn::tile_lif::{QuantLif, Synapses};
use engine::synth::alphafabric::{AnnealConfig, Circuit, PlacementEnv, anneal};

fn pct(from: f64, to: f64) -> f64 {
    if from <= 0.0 {
        0.0
    } else {
        (from - to) / from * 100.0
    }
}

fn main() {
    println!("NeuroAlphaFabric — optimizing a spiking neuron's tile layout\n");

    // A small LIF neuron: 3 synapses feeding a leaky integrate-and-fire membrane.
    let lif = QuantLif::new(4, 10).with_leak_shift(1);
    let synapses = Synapses::new(vec![3, 4, 2], 4);
    let circuit = Circuit::from_aig("snn_neuron", lif.neuron_step_aig(&synapses));

    let mut env = PlacementEnv::new(&circuit).expect("placement env builds");
    let base_hpwl = env.hpwl();
    let base_timing = env.timing_cost();
    let base_ok = env.verify_physical();

    // Route-validated + timing-aware simulated annealing: find a placement that
    // routes AND minimizes the criticality-weighted wirelength.
    let cfg = AnnealConfig {
        route_validated_best: true,
        timing_weight: 4.0,
        iterations: 2500,
        ..AnnealConfig::default()
    };
    let result = anneal(&mut env, &cfg); // env now holds the best layout
    let opt_hpwl = env.hpwl();
    let opt_timing = env.timing_cost();
    let opt_ok = env.verify_physical();

    println!(
        "  neuron next-state circuit: {} gates\n",
        circuit.netlist.num_gates()
    );
    println!(
        "  {:<26} {:>12} {:>12} {:>10}",
        "metric", "baseline", "timing-SA", "improved"
    );
    println!("  {}", "─".repeat(62));
    println!(
        "  {:<26} {:>12} {:>12} {:>9.1}%",
        "HPWL (total wirelength)",
        base_hpwl,
        opt_hpwl,
        pct(base_hpwl as f64, opt_hpwl as f64),
    );
    println!(
        "  {:<26} {:>12.0} {:>12.0} {:>9.1}%",
        "criticality-weighted WL",
        base_timing,
        opt_timing,
        pct(base_timing, opt_timing),
    );
    println!(
        "  {:<26} {:>12} {:>12}",
        "physically correct", base_ok, opt_ok,
    );
    println!(
        "\n  anneal: routed-cost improvement {:.1}%, criticality-weighted WL improvement {:.1}%.",
        result.improvement() * 100.0,
        result.timing_improvement() * 100.0,
    );
    println!(
        "\n  The optimized layout is still physically correct ({opt_ok}) — the learned\n\
         placer shortened the spiking neuron's wiring while preserving its function\n\
         (tile == AIG over the exhaustive truth table). NeuroAlphaFabric: SNN meets\n\
         learned EDA."
    );
}
