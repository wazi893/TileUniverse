//! Cosmological Scale GHZ - Finding the TRUE Limit
//!
//! Pushing BigUint to its absolute breaking point.
//! Where does creation time or memory actually become a problem?

use engine::tile8::{UnlimitedGhzState, create_unlimited_ghz};
use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║           COSMOLOGICAL SCALE GHZ - FINDING THE TRUE LIMIT             ║");
    println!("║   Pushing beyond all reasonable numbers into pure mathematics          ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Phase 1: Powers of 10 - Exponential climb
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  PHASE 1: Powers of 10 (10^n qubits) - Finding where it slows down");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:>25} {:>12} {:>12} {:>12} {:>15}",
        "Qubits", "Create", "Verify", "Memory", "BigUint bits"
    );
    println!("{}", "-".repeat(80));

    // Exponentially increasing exponents
    let exponents = [
        100,         // Googol
        1_000,       //
        10_000,      //
        100_000,     // Previous "limit"
        1_000_000,   // One million digits
        10_000_000,  // Ten million digits
        100_000_000, // Hundred million digits - ~400 MB just for the number!
    ];

    for exp in exponents {
        let result = test_power_of_10(exp);
        if result.is_none() {
            println!("  ⚠ Stopped at 10^{} - too slow or OOM", exp);
            break;
        }
    }

    // =========================================================================
    // Phase 2: Tower of Powers - 10^10^10^...
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("  PHASE 2: Named Cosmological Numbers");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Googol: 10^100
    named_test("Googol", BigUint::from(10u32).pow(100));

    // Googolplex is 10^googol - impossible to represent, but we can do smaller towers
    // 10^10^6 = 10^(1 million)
    named_test("10^(10^6)", BigUint::from(10u32).pow(1_000_000));

    // Shannon number (chess positions): ~10^120
    named_test("Shannon Number (~10^120)", BigUint::from(10u32).pow(120));

    // Number of atoms in observable universe: ~10^80
    named_test("Atoms in Universe (10^80)", BigUint::from(10u32).pow(80));

    // Planck volumes in observable universe: ~10^185
    named_test("Planck Volumes (10^185)", BigUint::from(10u32).pow(185));

    // Poincaré recurrence time for universe: ~10^(10^(10^(10^2.08)))
    // We can't do that, but let's do 10^(10^10) if possible
    println!("\nAttempting 10^(10^7) - ten million digit exponent...");
    named_test("10^(10^7)", BigUint::from(10u32).pow(10_000_000));

    // =========================================================================
    // Phase 3: Binary powers - 2^2^2^...
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("  PHASE 3: Powers of 2 (2^n qubits) - More natural for quantum");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:>25} {:>12} {:>12} {:>12} {:>15}",
        "Qubits", "Create", "Verify", "Memory", "BigUint bits"
    );
    println!("{}", "-".repeat(80));

    // Powers of 2
    let bit_exponents = [
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000, // 2^(100 million) qubits
    ];

    for bits in bit_exponents {
        let result = test_power_of_2(bits);
        if result.is_none() {
            println!("  ⚠ Stopped at 2^{} - too slow or OOM", bits);
            break;
        }
    }

    // =========================================================================
    // Phase 4: The Ultimate - How big can we go in 10 seconds?
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("  PHASE 4: Finding the Practical Limit (< 10 second creation)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Binary search for the largest 10^n we can create in under 10 seconds
    let mut low = 1_000_000u32;
    let mut high = 1_000_000_000u32;
    let mut best = low;

    println!("Binary searching for largest 10^n creatable in <10 seconds...\n");

    while low <= high {
        let mid = low + (high - low) / 2;

        print!("  Testing 10^{}... ", mid);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let start = Instant::now();
        let _n_qubits = BigUint::from(10u32).pow(mid);
        let elapsed = start.elapsed();

        if elapsed.as_secs() < 10 {
            println!("OK ({:.2?})", elapsed);
            best = mid;
            low = mid + 1;
        } else {
            println!("TOO SLOW ({:.2?})", elapsed);
            high = mid - 1;
        }

        // Safety: if we're taking more than 30 seconds, abort
        if elapsed.as_secs() > 30 {
            break;
        }
    }

    println!("\n  ✓ Largest 10^n creatable in <10 seconds: 10^{}", best);

    // Now create and verify at that scale
    println!("\n  Creating and verifying 10^{} qubit GHZ state...", best);

    let start = Instant::now();
    let n_qubits = BigUint::from(10u32).pow(best);
    let biguint_time = start.elapsed();

    let start = Instant::now();
    let ghz = create_unlimited_ghz(n_qubits);
    let create_time = start.elapsed();

    let start = Instant::now();
    let verification = ghz.verify();
    let verify_time = start.elapsed();

    let mem_mb = ghz.memory_bytes() as f64 / (1024.0 * 1024.0);
    let bits = ghz.n_qubits().bits();

    println!("\n  ╔══════════════════════════════════════════════════════════════════╗");
    println!("  ║  PRACTICAL LIMIT ACHIEVED: 10^{} QUBITS", best);
    println!("  ╠══════════════════════════════════════════════════════════════════╣");
    println!("  ║  BigUint creation: {:>10.2?}", biguint_time);
    println!("  ║  GHZ creation:     {:>10.2?}", create_time);
    println!("  ║  Verification:     {:>10.2?}", verify_time);
    println!("  ║  Memory:           {:>10.2} MB", mem_mb);
    println!("  ║  BigUint bits:     {:>10}", bits);
    println!("  ║  Fidelity:         {:>10.6}", verification.fidelity);
    println!("  ║  State space:      2^(10^{}) amplitudes", best);
    println!("  ╚══════════════════════════════════════════════════════════════════╝");

    // =========================================================================
    // Summary
    // =========================================================================

    println!("\n╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  THE TRUE LIMIT IS BIGUINT EXPONENTIATION, NOT GHZ SIMULATION         ║");
    println!("║                                                                        ║");
    println!("║  GHZ state creation/verification: O(1) - always microseconds          ║");
    println!("║  Bottleneck: Computing 10^n as BigUint takes O(n * log(n)) time        ║");
    println!("║                                                                        ║");
    println!("║  If we pre-computed or streamed the BigUint, the GHZ simulation       ║");
    println!("║  itself would STILL be O(1) regardless of qubit count!                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
}

fn test_power_of_10(exp: u32) -> Option<()> {
    let start = Instant::now();
    let n_qubits = BigUint::from(10u32).pow(exp);
    let biguint_time = start.elapsed();

    // Timeout check
    if biguint_time.as_secs() > 60 {
        return None;
    }

    let start = Instant::now();
    let ghz = UnlimitedGhzState::new(n_qubits);
    let create_time = start.elapsed();

    let start = Instant::now();
    let _verification = ghz.verify();
    let verify_time = start.elapsed();

    let mem = ghz.memory_bytes();
    let bits = ghz.n_qubits().bits();

    let mem_str = if mem >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", mem as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if mem >= 1024 * 1024 {
        format!("{:.1} MB", mem as f64 / (1024.0 * 1024.0))
    } else if mem >= 1024 {
        format!("{:.1} KB", mem as f64 / 1024.0)
    } else {
        format!("{} B", mem)
    };

    println!(
        "{:>25} {:>12} {:>12} {:>12} {:>15}",
        format!("10^{}", exp),
        format_duration(create_time),
        format_duration(verify_time),
        mem_str,
        format_big_num(bits),
    );

    // Also show BigUint creation time if significant
    if biguint_time.as_millis() > 10 {
        println!(
            "        (BigUint 10^{} took {})",
            exp,
            format_duration(biguint_time)
        );
    }

    Some(())
}

fn test_power_of_2(exp: u32) -> Option<()> {
    let start = Instant::now();
    let n_qubits: BigUint = BigUint::one() << exp;
    let biguint_time = start.elapsed();

    if biguint_time.as_secs() > 60 {
        return None;
    }

    let start = Instant::now();
    let ghz = UnlimitedGhzState::new(n_qubits);
    let create_time = start.elapsed();

    let start = Instant::now();
    let _verification = ghz.verify();
    let verify_time = start.elapsed();

    let mem = ghz.memory_bytes();
    let bits = ghz.n_qubits().bits();

    let mem_str = if mem >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", mem as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if mem >= 1024 * 1024 {
        format!("{:.1} MB", mem as f64 / (1024.0 * 1024.0))
    } else if mem >= 1024 {
        format!("{:.1} KB", mem as f64 / 1024.0)
    } else {
        format!("{} B", mem)
    };

    println!(
        "{:>25} {:>12} {:>12} {:>12} {:>15}",
        format!("2^{}", exp),
        format_duration(create_time),
        format_duration(verify_time),
        mem_str,
        format_big_num(bits),
    );

    if biguint_time.as_millis() > 10 {
        println!(
            "        (BigUint 2^{} took {})",
            exp,
            format_duration(biguint_time)
        );
    }

    Some(())
}

fn named_test(name: &str, n_qubits: BigUint) {
    let bits = n_qubits.bits();

    let start = Instant::now();
    let ghz = UnlimitedGhzState::new(n_qubits);
    let create_time = start.elapsed();

    let start = Instant::now();
    let verification = ghz.verify();
    let verify_time = start.elapsed();

    let mem = ghz.memory_bytes();
    let log10 = bits as f64 * std::f64::consts::LOG10_2;

    let mem_str = if mem >= 1024 * 1024 {
        format!("{:.1} MB", mem as f64 / (1024.0 * 1024.0))
    } else if mem >= 1024 {
        format!("{:.1} KB", mem as f64 / 1024.0)
    } else {
        format!("{} B", mem)
    };

    println!("  {}", name);
    println!("    Qubits: ~10^{:.0} ({} bits to store)", log10, bits);
    println!(
        "    Create: {}, Verify: {}, Fidelity: {:.6}",
        format_duration(create_time),
        format_duration(verify_time),
        verification.fidelity
    );
    println!(
        "    Memory: {}, State space: 2^(~10^{:.0})\n",
        mem_str, log10
    );
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

fn format_big_num(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
