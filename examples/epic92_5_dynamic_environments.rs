//! EPIC 92 Sprint 92.5: Dynamic Environments - THE FINAL TEST
//!
//! Hypothesis: Static environments favor exploitation (classical), while dynamic
//! environments favor exploration (quantum).
//!
//! This is the last fundamental structural hypothesis in EPIC 92.
//! If this is null, we have a comprehensive negative result.
//! If this is positive, we've found quantum's ecological niche.
//!
//! Run with: cargo run --example epic92_5_dynamic_environments --release

use engine::dynamic_environments::{
    DynamicEnvironmentsConfig, print_dynamic_environments_results,
    run_dynamic_environments_experiment,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 92.5: DYNAMIC ENVIRONMENTS                         ║");
    println!("║   The Final Test for Quantum Advantage                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let config = DynamicEnvironmentsConfig::default();

    println!("Experiment Design:");
    println!(
        "  Dynamics Modes: {:?}",
        config
            .dynamics_modes
            .iter()
            .map(|d| d.display_name())
            .collect::<Vec<_>>()
    );
    println!("  Change Frequency: {} ticks", config.change_frequency);
    println!("  Seeds: {}", config.seeds);
    println!("  Ticks: {}", config.ticks);
    println!(
        "  Environment: {} patches, value {}",
        config.n_patches, config.patch_value
    );
    println!(
        "  Total trials: {} ({} modes × {} seeds × 2 brains)",
        config.dynamics_modes.len() * config.seeds * 2,
        config.dynamics_modes.len(),
        config.seeds
    );
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("THE STORY SO FAR (EPIC 92):");
    println!("═══════════════════════════════════════════════════════════");
    println!("  ❌ 92.1 Mutation Rate (0.1-0.9): Classical wins uniformly");
    println!("  ❌ 92.2 Temporal Dynamics (8 snapshots): No cumulative quantum advantage");
    println!("  ⚠️  92.4 Continuous Mutations: Classical +42.7% (prototype)");
    println!("  ⏳ 92.5 Dynamic Environments: TESTING NOW...\n");

    println!("This is the FINAL structural hypothesis:");
    println!("  If NULL:     Comprehensive negative result → Publish");
    println!("  If POSITIVE: Real discovery → Quantum's ecological niche found\n");

    println!("Hypothesis:");
    println!("  H5: Dynamic environments favor exploration (quantum) over");
    println!("      exploitation (classical) because:");
    println!("      - Static → Converge to fixed optimal policy (classical wins)");
    println!("      - Dynamic → Continuous adaptation needed (quantum wins)");
    println!("  H0: Classical dominates regardless of dynamics\n");

    println!("Environment Modes:");
    println!("  Static:      Resources never move (baseline from EPIC 91-92.4)");
    println!(
        "  RandomWalk:  Patches drift ±10 tiles every {} ticks",
        config.change_frequency
    );
    println!(
        "  Respawning:  Patches deplete & respawn elsewhere every {} ticks",
        config.change_frequency
    );
    println!("  Periodic:    Patches orbit in predictable circles (π/8 rad/update)\n");

    println!("Key Questions:");
    println!("  1. Does quantum EVER win in ANY dynamics mode?");
    println!("  2. Does dynamics help quantum (even if still loses)?");
    println!("  3. What's the quantum vs classical ranking across all modes?\n");

    println!("Expected Runtime: ~40 seconds");
    println!("Starting dynamic environments experiment...\n");

    let start = Instant::now();
    let results = run_dynamic_environments_experiment(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Experiment complete in {:.1} seconds",
        elapsed.as_secs_f64()
    );
    println!(
        "  ({:.2} seconds per trial)\n",
        elapsed.as_secs_f64() / (config.dynamics_modes.len() * config.seeds * 2) as f64
    );

    print_dynamic_environments_results(&results);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("EPIC 92 FINAL VERDICT:");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    if results.quantum_ever_wins().is_some() {
        let winning_mode = results.quantum_ever_wins().unwrap();
        println!("🎉 QUANTUM ADVANTAGE FOUND!");
        println!();
        println!("After testing:");
        println!("  • 9 mutation rates");
        println!("  • 8 temporal snapshots");
        println!("  • 3 mutation modes");
        println!("  • 4 environment dynamics");
        println!();
        println!(
            "Quantum WINS in: {} environments",
            winning_mode.display_name()
        );
        println!();
        println!("This is the ecological niche where quantum thrives.");
        println!("Classical dominates static/stable environments.");
        println!("Quantum dominates dynamic/changing environments.");
        println!();
        println!("PUBLISHABLE RESULT:");
        println!("  'Quantum advantage in evolutionary foraging emerges");
        println!("   in dynamic environments requiring continuous adaptation'");
    } else {
        println!("✗ COMPREHENSIVE NULL RESULT");
        println!();
        println!("After exhaustive testing across:");
        println!("  • 9 mutation rates (0.1-0.9)");
        println!("  • 8 temporal windows (250-5000 ticks)");
        println!("  • 3 mutation modes (discrete/continuous/hybrid)");
        println!("  • 4 environment dynamics (static/random/respawn/periodic)");
        println!();
        println!("Classical neural networks DOMINATE quantum circuits");
        println!("in evolutionary foraging across ALL tested conditions.");
        println!();
        println!("PUBLISHABLE RESULT:");
        println!("  'Comprehensive search for quantum advantage in");
        println!("   evolutionary foraging yields null result across");
        println!("   mutation rates, temporal windows, mutation modes,");
        println!("   and environmental dynamics'");
        println!();
        println!("This is VALUABLE SCIENCE:");
        println!("  → Defines boundary where quantum does NOT help");
        println!("  → Informs future quantum algorithm design");
        println!("  → Establishes benchmark for classical superiority");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("EPIC 92: QUANTUM ADVANTAGE HUNT - COMPLETE");
    println!("═══════════════════════════════════════════════════════════");
    println!();
}
