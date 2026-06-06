//! EPIC 92 Sprint 92.2: Temporal Dynamics
//!
//! Hypothesis: Quantum might lead early (exploration) but classical catches up (exploitation).
//! Test: Measure population at multiple time snapshots to find advantage windows.
//!
//! Run with: cargo run --example epic92_2_temporal --release

use engine::temporal_dynamics::{TemporalConfig, print_temporal_results, run_temporal_sweep};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 92.2: TEMPORAL DYNAMICS                            ║");
    println!("║   Does Quantum Ever Lead?                                 ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let config = TemporalConfig::default();

    println!("Experiment Design:");
    println!("  Snapshot Ticks: {:?}", config.snapshot_ticks);
    println!("  Seeds: {}", config.seeds);
    println!(
        "  Mutation Rate: {} (optimal from 92.1)",
        config.mutation_rate
    );
    println!(
        "  Environment: {} patches, value {}",
        config.heat_patches, config.heat_value
    );
    println!(
        "  Total trials: {} ({} snapshots × {} seeds × 2 brains)",
        config.snapshot_ticks.len() * config.seeds * 2,
        config.snapshot_ticks.len(),
        config.seeds
    );
    println!();

    println!("Hypothesis:");
    println!("  H2a: Quantum leads early (exploration advantage)");
    println!("  H2b: Classical catches up by tick 1000 (exploitation)");
    println!("  H0:  Classical leads at all time points\n");

    println!("Key Questions:");
    println!("  1. Does quantum EVER lead at any snapshot?");
    println!("  2. If so, when does classical overtake?");
    println!("  3. Do populations peak at different times?\n");

    println!("Starting temporal sweep...\n");

    let start = Instant::now();
    let results = run_temporal_sweep(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Temporal sweep complete in {:.1} seconds",
        elapsed.as_secs_f64()
    );

    print_temporal_results(&results);

    println!("\n════════════════════════════════════════════════════════════");
    println!("INTERPRETATION:");
    println!("════════════════════════════════════════════════════════════");
    println!();

    if results.quantum_ever_leads().is_some() {
        println!("✓ EARLY-GAME QUANTUM ADVANTAGE DETECTED!");
        println!("  → Quantum's exploration helps in the beginning");
        println!("  → Classical's policy specialization wins long-term");
        println!("  → There IS a quantum advantage window!");
        println!();
        println!("Next Steps:");
        println!("  - Optimize for early-game performance");
        println!("  - Test in more volatile environments");
        println!("  - Investigate exploration vs exploitation trade-off");
    } else {
        println!("✗ NO TEMPORAL ADVANTAGE WINDOW");
        println!("  → Classical leads from the start");
        println!("  → Quantum doesn't have early-game exploration advantage");
        println!("  → Proceed to Sprint 92.4 (Continuous Mutations)");
    }
    println!();
}
