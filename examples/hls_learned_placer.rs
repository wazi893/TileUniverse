//! AF4/AF5 learned placer × HLS corpus (bridge experiment #6, Sprint 383 follow-up).
//!
//! The HLS middle layer turns source code into an endless supply of placement
//! instances — here it becomes the learned placer's curriculum. Train the
//! route-aware policy on HLS-generated instances, then evaluate generalization on
//! **held-out widths** of the same source families (the honest AF4 win is
//! generalization, not beating SA). Every evaluated placement is routed and
//! physically verified on real tiles (post router-fix, routable ⟹ correct holds).
//!
//! Split mirrors the AlphaFabric corpus convention: test = widths divisible by 4.
//!
//! Run: `cargo run --release --example hls_learned_placer`

use engine::synth::alphafabric::{
    Circuit, Policy, TrainConfig, evaluate, hls_instances, train_route_aware,
};

fn main() {
    let instances = hls_instances();
    let mut train_set: Vec<Circuit> = Vec::new();
    let mut test_set: Vec<Circuit> = Vec::new();
    for inst in instances {
        if inst.width % 4 == 0 {
            test_set.push(inst.circuit);
        } else {
            train_set.push(inst.circuit);
        }
    }
    println!(
        "HLS placement curriculum: {} train / {} held-out (widths % 4 == 0)\n",
        train_set.len(),
        test_set.len()
    );

    // Route-aware fit (routes per evaluation — the HLS train set is small enough).
    let outcome = train_route_aware(&train_set, &TrainConfig::default());
    println!(
        "train: routed-cost ratio {:.3} (hand-tuned start {:.3})\n",
        outcome.train_cost_ratio, outcome.hand_tuned_cost_ratio
    );

    let header = format!(
        "{:<12} {:>6} {:>9} {:>9} {:>7} {:>9} {:>8}",
        "held-out", "gates", "rm_hpwl", "ln_hpwl", "ratio", "routable", "phys_ok"
    );
    for (label, policy) in [
        ("hand-tuned", Policy::hand_tuned()),
        ("learned", outcome.policy),
    ] {
        println!("policy: {label}");
        println!("{header}");
        println!("{}", "-".repeat(68));
        let evals = evaluate(&test_set, &policy);
        let mut ratios = Vec::new();
        let mut verified = 0usize;
        for e in &evals {
            ratios.push(e.ratio);
            verified += e.phys_ok as usize;
            println!(
                "{:<12} {:>6} {:>9} {:>9} {:>7.3} {:>9} {:>8}",
                e.name, e.gates, e.rowmajor_hpwl, e.learned_hpwl, e.ratio, e.routable, e.phys_ok
            );
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        println!(
            "{}\nmean held-out HPWL ratio {:.3} (<1 = beats row-major); {}/{} physically verified\n",
            "-".repeat(68),
            mean,
            verified,
            evals.len()
        );
    }
}
