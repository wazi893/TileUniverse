// =============================================================================
// Shor's Algorithm Quick Demo
// =============================================================================
//
// Fast demonstration of Shor's quantum factoring algorithm.
//
// Run with: cargo run --example shor_demo --release
//
// =============================================================================

use engine::algorithms::factor_with_shor;
use engine::number_theory::classical_order;
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         SHOR'S ALGORITHM - QUICK DEMONSTRATION                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Classical vs Quantum Period Finding
    println!("═══ Classical Period Finding ═══");
    println!("Finding period r where 7^r ≡ 1 (mod 15)");
    let start = Instant::now();
    let period = classical_order(7, 15);
    let classical_time = start.elapsed();
    println!("✓ Period r = {}", period.unwrap());
    println!("  Time: {:?}\n", classical_time);

    // Demo 2: Factor 15
    println!("═══ Quantum Factorization of 15 ═══");
    let start = Instant::now();
    match factor_with_shor(15) {
        Some((p, q)) => {
            let quantum_time = start.elapsed();
            println!("✓ SUCCESS: 15 = {} × {}", p, q);
            println!("  Time: {:?}", quantum_time);
            println!("  Verification: {} × {} = {}\n", p, q, p * q);
        }
        None => println!("✗ Failed to factor 15\n"),
    }

    // Demo 3: Factor 21
    println!("═══ Quantum Factorization of 21 ═══");
    let start = Instant::now();
    match factor_with_shor(21) {
        Some((p, q)) => {
            let quantum_time = start.elapsed();
            println!("✓ SUCCESS: 21 = {} × {}", p, q);
            println!("  Time: {:?}", quantum_time);
            println!("  Verification: {} × {} = {}\n", p, q, p * q);
        }
        None => println!("✗ Failed to factor 21\n"),
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Shor's algorithm demonstrates exponential quantum speedup    ║");
    println!("║  for integer factorization - a foundational result in         ║");
    println!("║  quantum computing with implications for cryptography.        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}
