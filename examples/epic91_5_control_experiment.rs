//! EPIC 91.5: Handicapped Classical NN Control Experiment
//!
//! Three-way comparison: Full Quantum vs Separable Quantum vs Handicapped Classical NN
//!
//! This experiment tests whether quantum can compete when classical has matched disadvantages:
//! - Handicapped NN: 61 parameters (vs quantum's ~50)
//! - Poor initialization: Uniform [-1, 1] (vs Xavier)
//! - Discrete mutations: Quantized to 0.1 (vs smooth Gaussian)
//!
//! Run with: cargo run --example epic91_5_control_experiment --release

use engine::brain::BrainType;
use engine::three_way_comparison::{
    ThreeWayConfig, print_three_way_results, run_three_way_comparison,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 91.5: HANDICAPPED CLASSICAL NN CONTROL EXPERIMENT  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("RESEARCH QUESTION:");
    println!("  Can quantum circuits compete when classical NN is handicapped");
    println!("  to have similar parameter count and optimization disadvantages?\n");

    println!("BRAIN TYPES:");
    println!("  1. Full Quantum (Entangling)    - ~50 params, random circuits");
    println!("  2. Separable Quantum (No CNOTs) - ~50 params, random circuits");
    println!("  3. Handicapped Classical NN     - 61 params, poor init, discrete mutations\n");

    // Full experiment config: 100 seeds, 1000 ticks
    let mut config = ThreeWayConfig::default();
    config.replications = 100;
    config.max_ticks = 1000;

    // Replace ClassicalNN with HandicappedClassicalNN
    config.brain_types = vec![
        BrainType::FullQuantum,
        BrainType::SeparableQuantum,
        BrainType::HandicappedClassicalNN,
    ];

    println!("Experiment Configuration:");
    println!(
        "  - Seeds: {} to {}",
        config.seed_start,
        config.seed_start + config.replications as u64 - 1
    );
    println!("  - Ticks per trial: {}", config.max_ticks);
    println!(
        "  - Total trials: {} (100 seeds × 3 brain types)",
        config.replications * 3
    );
    println!(
        "  - Environment: {} patches, value {}",
        config.heat_patches, config.heat_value
    );
    println!("  - Mutation rate: {}", config.forager_config.mutation_rate);
    println!();

    println!("Starting experiment...\n");

    let start = Instant::now();
    let results = run_three_way_comparison(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Experiment complete in {:.1} seconds",
        elapsed.as_secs_f64()
    );
    println!(
        "  ({:.2} seconds per trial)\n",
        elapsed.as_secs_f64() / (config.replications * 3) as f64
    );

    print_three_way_results(&results);

    println!("\n════════════════════════════════════════════════════════════");
    println!("INTERPRETATION GUIDE:");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("If Handicapped NN still wins:");
    println!("  → Classical architecture fundamentally better for this task");
    println!("  → Quantum may not have advantage in evolutionary foraging");
    println!();
    println!("If Full Quantum wins vs Handicapped NN:");
    println!("  → Original EPIC 91 result was due to optimization advantages");
    println!("  → Quantum circuits CAN compete with fair comparison");
    println!();
    println!("If results are comparable:");
    println!("  → Parameter count matters more than computation type");
    println!("  → Need deeper investigation of capacity vs architecture");
    println!();
    println!("════════════════════════════════════════════════════════════\n");
}
