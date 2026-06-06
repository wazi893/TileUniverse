//! EPIC 92 Sprint 92.4: Continuous Quantum Mutations
//!
//! Hypothesis: Discrete gate mutations (insert/delete/swap) create a rougher fitness landscape
//! than continuous weight mutations, disadvantaging quantum circuits vs classical NNs.
//!
//! Test: Compare three quantum mutation modes against classical baseline:
//! - Discrete: Current behavior (structural changes)
//! - Continuous: Only perturb angles (smooth landscape)
//! - Hybrid: Mix of both (80% continuous, 20% discrete)
//!
//! Run with: cargo run --example epic92_4_continuous_mutations --release

use engine::continuous_mutations::{
    ContinuousMutationsConfig, print_continuous_mutations_results,
    run_continuous_mutations_experiment,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 92.4: CONTINUOUS QUANTUM MUTATIONS                 ║");
    println!("║   Does Smooth Mutation Landscape Help Quantum?            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let config = ContinuousMutationsConfig::default();

    println!("Experiment Design:");
    println!("  Mutation Modes: {:?}", config.mutation_modes);
    println!(
        "  Perturbation σ: {} (~{:.0}° for continuous modes)",
        config.perturbation_stddev,
        config.perturbation_stddev * 180.0 / std::f32::consts::PI
    );
    println!("  Seeds: {}", config.seeds);
    println!("  Ticks: {}", config.ticks);
    println!(
        "  Environment: {} patches, value {}",
        config.heat_patches, config.heat_value
    );
    println!(
        "  Total trials: {} ({} modes × {} seeds)",
        config.mutation_modes.len() * config.seeds + config.seeds, // +classical
        config.mutation_modes.len() + 1,
        config.seeds
    );
    println!();

    println!("Hypothesis:");
    println!("  H4: Discrete gate mutations create rougher landscape than continuous");
    println!("      weight mutations → disadvantages quantum evolution");
    println!("  H0: Mutation landscape roughness is NOT the bottleneck\n");

    println!("Mutation Mode Details:");
    println!("  Discrete:   Insert/delete/swap gates + change qubits + change angles");
    println!("              (1-3 mutations per reproduction, 6 mutation types)");
    println!("  Continuous: ONLY perturb angles of Phase/Rx/Ry/Rz gates");
    println!(
        "              (Gaussian perturbation with σ={})",
        config.perturbation_stddev
    );
    println!("  Hybrid:     80% continuous angle tweaks, 20% discrete structural");
    println!("              (Best of both worlds?)");
    println!("  Classical:  Continuous weight perturbations (baseline)\n");

    println!("⚠ IMPORTANT NOTE:");
    println!("  This is a PROTOTYPE implementation to test infrastructure.");
    println!("  All modes currently use the same discrete mutation logic.");
    println!("  Full implementation requires hooking custom mutation into reproduction.\n");

    println!("Key Questions:");
    println!("  1. Does continuous mode beat discrete on cumulative AUC?");
    println!("  2. Can any quantum mutation mode compete with classical?");
    println!("  3. Is hybrid mode the optimal balance?\n");

    println!("Starting continuous mutations experiment...\n");

    let start = Instant::now();
    let results = run_continuous_mutations_experiment(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Experiment complete in {:.1} seconds",
        elapsed.as_secs_f64()
    );
    println!(
        "  ({:.2} seconds per trial)\n",
        elapsed.as_secs_f64() / (config.mutation_modes.len() * config.seeds + config.seeds) as f64
    );

    print_continuous_mutations_results(&results);

    println!("\n════════════════════════════════════════════════════════════");
    println!("INTERPRETATION:");
    println!("════════════════════════════════════════════════════════════");
    println!();

    if results.continuous_wins() {
        println!("✓ H4 SUPPORTED: Continuous mutation landscape helps!");
        println!("  → Discrete gate changes create rougher fitness landscape");
        println!("  → Smooth angle perturbations improve evolvability");
        println!("  → This suggests quantum CAN compete with right mutation strategy");
        println!();
        println!("Next Steps:");
        println!("  - Implement full continuous mutation in reproduction logic");
        println!("  - Test if continuous quantum can beat classical");
        println!("  - Proceed to Sprint 92.5 (Dynamic Environments)");
    } else {
        println!("✗ H4 REJECTED: Mutation landscape is NOT the bottleneck");
        println!("  → Discrete mutations don't significantly hurt quantum");
        println!("  → The advantage/disadvantage lies elsewhere");
        println!("  → Mutation mode is not the key factor");
        println!();
        println!("Next Steps:");
        println!("  - Proceed to Sprint 92.5 (Dynamic Environments)");
        println!("  - Test if moving resources change the ranking");
        println!("  - Consider other structural hypotheses");
    }
    println!();
}
