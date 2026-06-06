//! EPIC 92 Sprint 92.1: Mutation Rate Sweep
//!
//! Hypothesis: 70% mutation rate is too harsh for quantum circuits.
//! Test: Sweep from 0.1 to 0.9 to find optimal rate for each brain type.
//!
//! Run with: cargo run --example epic92_1_mutation_sweep --release

use engine::mutation_rate_sweep::{
    MutationSweepConfig, print_mutation_sweep_table, run_mutation_sweep,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 92.1: MUTATION RATE SWEEP                          ║");
    println!("║   Finding Quantum's Sweet Spot                            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let config = MutationSweepConfig::default();

    println!("Experiment Design:");
    println!("  Mutation Rates: {:?}", config.mutation_rates);
    println!("  Seeds per rate: {}", config.seeds_per_rate);
    println!("  Ticks: {}", config.ticks);
    println!(
        "  Environment: {} patches, value {}",
        config.heat_patches, config.heat_value
    );
    println!(
        "  Total trials: {} ({} rates × {} seeds × {} brains)",
        config.mutation_rates.len() * config.seeds_per_rate * config.brain_types.len(),
        config.mutation_rates.len(),
        config.seeds_per_rate,
        config.brain_types.len()
    );
    println!();

    println!("Hypothesis:");
    println!("  H1: Quantum performs better at lower mutation rates (0.1-0.3)");
    println!("  H0: Classical dominates across all mutation rates\n");

    println!("Starting sweep...\n");

    let start = Instant::now();
    let results = run_mutation_sweep(&config);
    let elapsed = start.elapsed();

    println!("\n✓ Sweep complete in {:.1} seconds", elapsed.as_secs_f64());
    println!(
        "  ({:.2} seconds per trial)\n",
        elapsed.as_secs_f64()
            / (config.mutation_rates.len() * config.seeds_per_rate * config.brain_types.len())
                as f64
    );

    print_mutation_sweep_table(&results);

    println!("\n════════════════════════════════════════════════════════════");
    println!("INTERPRETATION:");
    println!("════════════════════════════════════════════════════════════");
    println!();

    if let Some(_rate) = results.quantum_wins_anywhere() {
        println!("✓ QUANTUM ADVANTAGE DETECTED!");
        println!("  → Quantum circuits CAN compete at specific mutation rates");
        println!("  → EPIC 91 result was due to suboptimal mutation rate");
        println!("  → Proceed with optimal rate for further experiments");
    } else {
        println!("✗ NO QUANTUM ADVANTAGE FOUND");
        println!("  → Classical dominates across all tested mutation rates");
        println!("  → Mutation rate is NOT the bottleneck");
        println!("  → Proceed to Sprint 92.2 (Temporal Dynamics)");
    }
    println!();
}
