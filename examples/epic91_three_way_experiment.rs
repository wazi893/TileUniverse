//! EPIC 91: Three-Way Comparison Experiment
//!
//! Does evolution discover quantum advantage? TRUE BASELINE VERSION.
//!
//! This experiment compares:
//! - Full Quantum (entangling gates allowed)
//! - Separable Quantum (no entangling gates)
//! - Classical Neural Network (no quantum at all)
//!
//! Run with: cargo run --example epic91_three_way_experiment --release

use engine::three_way_comparison::{
    ThreeWayConfig, print_three_way_results, run_three_way_comparison,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         EPIC 91: THREE-WAY QUANTUM ADVANTAGE TEST         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("PREREGISTERED EXPERIMENT");
    println!("  Architecture:  8→8→5 Classical NN (117 params)");
    println!("  Mutation:      Rate=0.7, Scale=0.1 (locked)");
    println!("  Statistics:    Paired t-test, Cohen's d, 95% CI");
    println!();

    // Use default config (100 seeds, 1000 ticks, 19 patches, value 35)
    let config = ThreeWayConfig::default();

    println!("Configuration:");
    println!(
        "  - Seeds: {}-{} (n={})",
        config.seed_start,
        config.seed_start + config.replications as u64 - 1,
        config.replications
    );
    println!("  - Ticks: {}", config.max_ticks);
    println!(
        "  - Environment: {} patches, value {}",
        config.heat_patches, config.heat_value
    );
    println!("  - Mutation rate: {}", config.forager_config.mutation_rate);
    println!();
    println!(
        "This will run {} trials ({} seeds × 3 brain types)",
        config.replications * 3,
        config.replications
    );
    println!("Estimated time: 2-4 hours");
    println!();
    println!("Starting experiment...\n");

    let start = Instant::now();
    let results = run_three_way_comparison(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Experiment complete in {:.1} minutes",
        elapsed.as_secs_f64() / 60.0
    );

    print_three_way_results(&results);

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  INTERPRETATION");
    println!("══════════════════════════════════════════════════════════════\n");

    // Automatic interpretation based on results
    let fq_mean = results.full_quantum.mean_population;
    let sq_mean = results.separable_quantum.mean_population;
    let cn_mean = results.classical_nn.mean_population;

    let fq_vs_sq_sig = results.fq_vs_sq.p_value < 0.05;
    let fq_vs_cn_sig = results.fq_vs_cn.p_value < 0.05;
    let _sq_vs_cn_sig = results.sq_vs_cn.p_value < 0.05;

    println!("Ranking: ");
    let mut groups = vec![
        ("Full Quantum", fq_mean),
        ("Separable Quantum", sq_mean),
        ("Classical NN", cn_mean),
    ];
    groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (name, mean)) in groups.iter().enumerate() {
        println!("  {}. {} ({:.1})", i + 1, name, mean);
    }

    println!();

    if fq_vs_sq_sig && fq_vs_cn_sig {
        println!("✓ QUANTUM ADVANTAGE CONFIRMED");
        println!("  Full Quantum significantly outperforms both Separable Quantum");
        println!("  and Classical NN. Entanglement provides genuine advantage.");
    } else if fq_vs_sq_sig && !fq_vs_cn_sig {
        println!("⚠ ENTANGLEMENT ADVANTAGE, NOT QUANTUM ADVANTAGE");
        println!("  Full Quantum beats Separable Quantum, but Classical NN performs");
        println!("  comparably. The advantage may be from nonlinearity, not quantumness.");
    } else if !fq_vs_sq_sig && !fq_vs_cn_sig {
        println!("✗ NO QUANTUM ADVANTAGE DETECTED");
        println!("  All three brain types perform similarly. No evidence that");
        println!("  quantum resources provide advantage in this environment.");
    } else {
        println!("? MIXED RESULTS");
        println!("  Statistical significance varies across comparisons.");
        println!("  Further investigation needed.");
    }

    println!("\n══════════════════════════════════════════════════════════════\n");
}
