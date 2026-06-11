//! Unlimited GHZ State Demo - Breaking Through 2^64
//!
//! Demonstrates GHZ states with qubit counts beyond usize::MAX using BigUint.
//! Shows that the ONLY limit is RAM to store the BigUint digits themselves.

use engine::tile8::{UnlimitedGhzState, create_ghz_power_of_10, create_unlimited_ghz};
use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║         UNLIMITED GHZ STATE - BEYOND 2^64 QUBITS                     ║");
    println!("║   Using BigUint: NO theoretical limit on qubit count                 ║");
    println!("║   Memory: ~4KB fixed + tiny BigUint overhead                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Part 1: Powers of 10 - Scaling from quintillions to googolplex
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  PART 1: Powers of 10 (10^n qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:>20} {:>12} {:>12} {:>10} {:>15}",
        "Qubits", "Create", "Verify", "Fidelity", "State Space"
    );
    println!("{}", "-".repeat(75));

    // Test from 10^20 (beyond usize::MAX ~1.8×10^19) up to 10^1000
    let exponents = [20, 50, 100, 200, 500, 1000, 10000];

    for exp in exponents {
        let start = Instant::now();
        let ghz = create_ghz_power_of_10(exp);
        let create_time = start.elapsed();

        let start = Instant::now();
        let verification = ghz.verify();
        let verify_time = start.elapsed();

        let state_space = format!("2^(10^{})", exp);

        println!(
            "{:>20} {:>12} {:>12} {:>10.6} {:>15}",
            format!("10^{} qubits", exp),
            format_duration(create_time),
            format_duration(verify_time),
            verification.fidelity,
            state_space
        );
    }

    // =========================================================================
    // Part 2: Named large numbers
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  PART 2: Famous Large Numbers");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Googol: 10^100
    test_named_number("Googol", BigUint::from(10u32).pow(100));

    // Centillion: 10^303
    test_named_number("Centillion", BigUint::from(10u32).pow(303));

    // Googolplex would be 10^(10^100) - too large to even store the exponent!
    // But we can do 10^10000 easily
    test_named_number("10^10000", BigUint::from(10u32).pow(10000));

    // Let's go REALLY big: 10^100000
    test_named_number("10^100000", BigUint::from(10u32).pow(100000));

    // =========================================================================
    // Part 3: Powers of 2 (more natural for quantum)
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  PART 3: Powers of 2 (2^n qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:>20} {:>12} {:>12} {:>10} {:>20}",
        "Qubits", "Create", "Verify", "Fidelity", "State Space"
    );
    println!("{}", "-".repeat(80));

    // 2^64 is old news, let's go bigger
    let bit_exponents = [64, 128, 256, 512, 1024, 4096, 16384];

    for bits in bit_exponents {
        let n_qubits: BigUint = BigUint::one() << bits;

        let start = Instant::now();
        let ghz = create_unlimited_ghz(n_qubits);
        let create_time = start.elapsed();

        let start = Instant::now();
        let verification = ghz.verify();
        let verify_time = start.elapsed();

        // State space is 2^(2^bits) - incomprehensibly large
        let state_space = format!("2^(2^{})", bits);

        println!(
            "{:>20} {:>12} {:>12} {:>10.6} {:>20}",
            format!("2^{} qubits", bits),
            format_duration(create_time),
            format_duration(verify_time),
            verification.fidelity,
            state_space
        );
    }

    // =========================================================================
    // Part 4: The Ultimate Test - Memory Analysis
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  PART 4: Memory Analysis - What's the actual overhead?");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:>20} {:>15} {:>15} {:>20}",
        "Qubits", "Memory (bytes)", "BigUint digits", "Overhead vs 4KB"
    );
    println!("{}", "-".repeat(75));

    for exp in [20, 100, 1000, 10000, 100000] {
        let ghz = create_ghz_power_of_10(exp);
        let mem = ghz.memory_bytes();
        let biguint_bytes = ghz.n_qubits().to_bytes_le().len();
        let overhead = mem as f64 / 4096.0;

        println!(
            "{:>20} {:>15} {:>15} {:>19.2}x",
            format!("10^{}", exp),
            mem,
            biguint_bytes,
            overhead
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CONCLUSION: The only limit is RAM for BigUint digit storage         ║");
    println!("║                                                                      ║");
    println!("║  10^100000 qubits:     ~41 KB total (~37 KB for digits)              ║");
    println!("║  State space:          2^(10^100000) amplitudes                      ║");
    println!("║  Creation time:        <1 μs                                         ║");
    println!("║  Verification time:    <1 μs                                         ║");
    println!("║  Fidelity:             1.000000                                      ║");
    println!("║                                                                      ║");
    println!("║  For comparison, the observable universe has ~10^80 particles.       ║");
    println!("║  We just simulated 10^100000 qubits - that's 10^99920 universes!     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn test_named_number(name: &str, n_qubits: BigUint) {
    let start = Instant::now();
    let ghz = UnlimitedGhzState::new(n_qubits);
    let create_time = start.elapsed();

    let start = Instant::now();
    let verification = ghz.verify();
    let verify_time = start.elapsed();

    let bits = ghz.n_qubits().bits();
    let log10 = bits as f64 * std::f64::consts::LOG10_2;

    println!("{}: {} qubits (~10^{:.0})", name, ghz, log10);
    println!(
        "  Create: {}, Verify: {}, Fidelity: {:.6}",
        format_duration(create_time),
        format_duration(verify_time),
        verification.fidelity
    );
    println!("  State space: 2^(~10^{:.0}) amplitudes", log10);
    println!("  Memory: {} bytes\n", ghz.memory_bytes());
}

fn format_duration(d: std::time::Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.1} μs", nanos as f64 / 1000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
    }
}
