//! EPIC 91: Quick smoke test (3 seeds)
//!
//! Run with: cargo run --example epic91_quick_test --release

use engine::three_way_comparison::{
    ThreeWayConfig, print_three_way_results, run_three_way_comparison,
};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         EPIC 91: QUICK SMOKE TEST (3 seeds)               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Quick test config: 3 seeds, 200 ticks
    let mut config = ThreeWayConfig::default();
    config.replications = 3;
    config.max_ticks = 200;

    println!("Quick Test Configuration:");
    println!(
        "  - Seeds: {} (n={})",
        config.seed_start, config.replications
    );
    println!("  - Ticks: {} (reduced for speed)", config.max_ticks);
    println!("  - Trials: {}", config.replications * 3);
    println!();

    let start = Instant::now();
    let results = run_three_way_comparison(&config);
    let elapsed = start.elapsed();

    println!(
        "\n✓ Smoke test complete in {:.1} seconds",
        elapsed.as_secs_f64()
    );

    print_three_way_results(&results);

    println!("\n✓ All brain types executed successfully!");
    println!("  Ready for full 100-seed experiment.");
}
