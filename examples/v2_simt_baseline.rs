//! SIMT Throughput Roadmap — Tier 0 dashboard.
//!
//! Prints the scalar baseline every later tier must beat, and the golden oracle
//! (per-program `hash_v2_final_state`) that future lane-0 evaluations must match
//! byte-for-byte. No engine change; no extrapolated numbers — only measured runs.
//!
//! Run: `cargo run --release --example v2_simt_baseline`

use std::error::Error;

use engine::tile_cpu::run_scalar_baseline_corpus;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  V2 SIMT THROUGHPUT — TIER 0: SCALAR BASELINE + GOLDEN ORACLE                ║");
    println!("║  K independent scalar physical V2 CPUs (one Simulation each)                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let report = run_scalar_baseline_corpus()?;

    println!(
        "{:<16} {:>6} {:>9} {:>10} {:>12} {:>20}",
        "program", "words", "cycles", "R0", "step (ms)", "golden_hash"
    );
    println!("{}", "─".repeat(80));
    for r in &report.per_cpu {
        println!(
            "{:<16} {:>6} {:>9} {:>10} {:>12.3} {:>#20x}",
            r.name,
            r.program_words,
            r.cycles,
            r.r0,
            r.step_time.as_secs_f64() * 1000.0,
            r.golden_hash,
        );
    }
    println!("{}", "─".repeat(80));

    let cpus = report.per_cpu.len();
    println!("\n=== BASELINE (the number SIMT must beat) ===");
    println!("  Independent CPUs:        {}", cpus);
    println!("  Total cycles advanced:   {}", report.total_cycles);
    println!(
        "  Build time (1-time):     {:.3} ms   (SIMT amortizes this across lanes)",
        report.total_build_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Step time (steady):      {:.3} ms",
        report.total_step_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput (step-only):  {:>12.1} CPU-cycles/sec",
        report.cycles_per_sec()
    );
    println!(
        "  Throughput (with build): {:>12.1} CPU-cycles/sec",
        report.cycles_per_sec_with_build()
    );

    println!("\nThese golden_hash values are the Tier-1+ oracle: lane 0 of any lane-packed");
    println!("evaluator running the same program must reproduce them byte-for-byte.");
    println!("No '× worlds' multipliers — every number above came from a real run.");

    Ok(())
}
