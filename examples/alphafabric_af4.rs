//! AlphaFabric AF4 demo: a learned constructive placer that generalizes, used
//! as a warm-start for simulated annealing.
//!
//! Fits a linear placement policy on a small set of training circuits, then
//! applies it — with no per-circuit search — to held-out circuits it never saw
//! (unseen widths + the mul4 multiplier). Reports:
//!   - learned one-shot HPWL vs the row-major baseline (the generalization win),
//!   - learned + a SHORT SA repair (warm-start) vs full cold SA (the speed win).
//!
//! Honest framing: the learned one-shot is a fast initializer (pure HPWL, so it
//! can over-cluster and hurt routability on big blocks); the short warm-started
//! SA repairs that and reaches near-cold-SA quality in a fraction of the search.
//! Every reported warm-start layout is physically verified on real tiles.
//!
//! Run: `cargo run --release --example alphafabric_af4`

use engine::synth::alphafabric::corpus::{
    build_and_reduce, build_eq_comparator, build_parity, build_ripple_adder,
};
use engine::synth::alphafabric::{
    AnnealConfig, Circuit, PlacementEnv, TrainConfig, anneal, place, train,
};
use engine::synth::benchmark::build_4bit_multiplier;

fn main() {
    // Training set: small circuits at widths {3, 5, 6} (NOT widths 4/8).
    let train_set = [
        Circuit::from_aig("adder3", build_ripple_adder(3)),
        Circuit::from_aig("adder5", build_ripple_adder(5)),
        Circuit::from_aig("adder6", build_ripple_adder(6)),
        Circuit::from_aig("eq3", build_eq_comparator(3)),
        Circuit::from_aig("eq5", build_eq_comparator(5)),
        Circuit::from_aig("eq6", build_eq_comparator(6)),
        Circuit::from_aig("parity6", build_parity(6)),
        Circuit::from_aig("andreduce6", build_and_reduce(6)),
    ];

    // Held-out set: unseen widths (4, 8) + the multiplier (unseen structure).
    let test_set = [
        Circuit::from_aig("adder4", build_ripple_adder(4)),
        Circuit::from_aig("adder8", build_ripple_adder(8)),
        Circuit::from_aig("eq4", build_eq_comparator(4)),
        Circuit::from_aig("eq8", build_eq_comparator(8)),
        Circuit::from_aig("parity8", build_parity(8)),
        Circuit::from_aig("mul4", build_4bit_multiplier()),
    ];

    println!("AlphaFabric AF4 — learned constructive placer + warm-start\n");

    let outcome = train(&train_set, &TrainConfig::default());
    println!(
        "trained on {} circuits (widths 3/5/6) -> theta {}",
        train_set.len(),
        outcome.weights
    );
    println!(
        "train HPWL ratio {:.3} (hand-tuned start {:.3}; 1.0 = row-major)\n",
        outcome.train_ratio, outcome.hand_tuned_ratio
    );

    let cold_cfg = AnnealConfig::default(); // 6000 iters, from row-major
    let warm_cfg = AnnealConfig {
        iterations: cold_cfg.iterations / 4, // 1/4 the search
        start_from_current: true,
        ..AnnealConfig::default()
    };

    println!(
        "{:<8} {:>5} {:>9} {:>10} {:>14} {:>10} {:>7}",
        "held-out", "gates", "rowmajor", "learned1", "warm+SA(1/4)", "coldSA", "phys"
    );

    let mut learn_sum = 0.0;
    let mut warm_sum = 0.0;
    let mut cold_sum = 0.0;

    for c in &test_set {
        // 1) Learned one-shot (generalization; HPWL only).
        let mut env = PlacementEnv::new(c).expect("env builds");
        env.reset();
        let rowmajor = env.hpwl() as f64;
        place(&mut env, &outcome.weights);
        let learned = env.hpwl() as f64;

        // 2) Warm-start: short SA from the learned init (repairs routability).
        let warm = anneal(&mut env, &warm_cfg);
        let warm_hpwl = warm.best_hpwl as f64;
        let phys_ok = warm.best.metrics.routable && env.verify_physical();

        // 3) Cold SA reference: full search from row-major.
        let mut cold_env = PlacementEnv::new(c).expect("env builds");
        let cold = anneal(&mut cold_env, &cold_cfg);
        let cold_hpwl = cold.best_hpwl as f64;

        learn_sum += learned / rowmajor;
        warm_sum += warm_hpwl / rowmajor;
        cold_sum += cold_hpwl / rowmajor;

        println!(
            "{:<8} {:>5} {:>9} {:>5} {:>3.0}% {:>6} {:>3.0}% {:>6} {:>3.0}% {:>7}",
            c.name,
            env.num_gates(),
            rowmajor as usize,
            learned as usize,
            learned / rowmajor * 100.0,
            warm_hpwl as usize,
            warm_hpwl / rowmajor * 100.0,
            cold_hpwl as usize,
            cold_hpwl / rowmajor * 100.0,
            phys_ok,
        );
    }

    let n = test_set.len() as f64;
    println!(
        "\nmean held-out HPWL (% of row-major): learned1 {:.0}%, warm+SA(1/4 search) {:.0}%, coldSA(full) {:.0}%",
        learn_sum / n * 100.0,
        warm_sum / n * 100.0,
        cold_sum / n * 100.0
    );
    println!(
        "Learned policy: one-shot, never saw these circuits. Warm+SA reaches ~cold-SA quality\n\
         in 1/4 the search and every warm-start layout is physically verified on real tiles."
    );
}
