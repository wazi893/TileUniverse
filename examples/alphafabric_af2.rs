//! AlphaFabric AF2 demo: simulated-annealing placer vs the row-major baseline.
//!
//! Anneals each circuit against the HPWL proxy (routing only to validate), then
//! reports the baseline vs best HPWL and routed cost, the routed-wire savings,
//! and the physical-legality gate. `mul4` is the synthesizable 4-bit multiplier
//! — the AF2 roadmap's MUL target as a placeable block.
//!
//! Run: `cargo run --release --example alphafabric_af2`

use engine::synth::alphafabric::corpus::{build_eq_comparator, build_parity, build_ripple_adder};
use engine::synth::alphafabric::{anneal, AnnealConfig, Circuit, PlacementEnv};
use engine::synth::benchmark::build_4bit_multiplier;
use std::time::Instant;

fn main() {
    let circuits = [
        Circuit::from_aig("adder4", build_ripple_adder(4)),
        Circuit::from_aig("adder8", build_ripple_adder(8)),
        Circuit::from_aig("eq8", build_eq_comparator(8)),
        Circuit::from_aig("parity8", build_parity(8)),
        Circuit::from_aig("mul4", build_4bit_multiplier()),
    ];
    let config = AnnealConfig::default();

    println!(
        "AlphaFabric AF2 — simulated annealing vs row-major baseline ({} iters)\n",
        config.iterations
    );
    println!(
        "{:<9} {:>5} {:>11} {:>7} {:>11} {:>9} {:>8}",
        "circuit", "gates", "hpwl(b>best)", "hpwl-%", "wire(b>best)", "routed-%", "phys_ok"
    );

    for c in &circuits {
        let mut env = PlacementEnv::new(c).expect("env builds");
        let t0 = Instant::now();
        let r = anneal(&mut env, &config);
        let elapsed = t0.elapsed();
        let phys_ok = env.verify_physical();

        // Routed-% is only defined when the baseline itself routed; otherwise SA
        // turned an unroutable baseline into a routable layout.
        let routed = if r.baseline.metrics.routable {
            format!("{:.1}%", r.improvement() * 100.0)
        } else {
            "base n/r".to_string()
        };
        let base_wire = if r.baseline.metrics.routable {
            r.baseline.metrics.wire_tiles.to_string()
        } else {
            "-".to_string()
        };

        println!(
            "{:<9} {:>5} {:>5}>{:<5} {:>6.1}% {:>5}>{:<5} {:>9} {:>8}   ({:?})",
            c.name,
            env.num_gates(),
            r.baseline_hpwl,
            r.best_hpwl,
            r.hpwl_improvement() * 100.0,
            base_wire,
            r.best.metrics.wire_tiles,
            routed,
            phys_ok,
            elapsed,
        );
    }

    println!(
        "\nHPWL is the inner-loop objective (routing runs only on baseline + best).\n\
         Every reported layout is physically verified on real tiles."
    );
}
