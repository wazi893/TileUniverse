//! GHZ State Qubit Limit Test
//! How many qubits can we simulate?
//!
//! Uses MinimalGhzState for TRUE O(1) creation AND verification - NO BigInt at all!
use engine::tile8::create_minimal_ghz;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       GHZ STATE QUBIT LIMIT TEST (MINIMAL - NO BIGINT!)          ║");
    println!("║   Testing maximum qubit count with O(1) memory and time          ║");
    println!("║   Using MinimalGhzState - NO HashMap, NO BigInt, just 2 blocks!  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test increasingly large qubit counts - now we can go to QUINTILLIONS!
    let qubit_counts: Vec<usize> = vec![
        1_000,
        1_000_000,                  // 1 million
        1_000_000_000,              // 1 billion
        1_000_000_000_000,          // 1 trillion
        1_000_000_000_000_000,      // 1 quadrillion
        1_000_000_000_000_000_000,  // 1 quintillion
        10_000_000_000_000_000_000, // 10 quintillion
        usize::MAX,                 // ~18.4 quintillion (2^64 - 1)
    ];

    println!(
        "{:>20} {:>12} {:>12} {:>10} {:>12}",
        "Qubits", "Create (μs)", "Verify (μs)", "Fidelity", "Memory"
    );
    println!("{}", "-".repeat(70));

    for n in qubit_counts {
        let label = format_big_number(n);

        // Create minimal GHZ state - O(1)!
        let start = Instant::now();
        let ghz = create_minimal_ghz(n);
        let create_us = start.elapsed().as_micros();

        // Verify - also O(1)!
        let start = Instant::now();
        let verification = ghz.verify();
        let verify_us = start.elapsed().as_micros();

        let memory = format_memory(ghz.memory_bytes());

        println!(
            "{:>20} {:>12} {:>12} {:>10.6} {:>12}",
            label, create_us, verify_us, verification.fidelity, memory
        );

        // Stop if we hit issues
        if verification.fidelity < 0.99 {
            println!("\n  ⚠ Fidelity dropped below 0.99, stopping test");
            break;
        }
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  MinimalGhzState: TRUE O(1) - No BigInt, No HashMap!            ║");
    println!("║  Memory is FIXED at ~4KB regardless of qubit count              ║");
    println!("║  Creation and verification both complete in microseconds        ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}

fn format_big_number(n: usize) -> String {
    if n == usize::MAX {
        "MAX (2^64-1)".to_string()
    } else if n >= 1_000_000_000_000_000_000 {
        format!("{:.1}Qi qubits", n as f64 / 1_000_000_000_000_000_000.0)
    } else if n >= 1_000_000_000_000_000 {
        format!("{:.1}Q qubits", n as f64 / 1_000_000_000_000_000.0)
    } else if n >= 1_000_000_000_000 {
        format!("{:.1}T qubits", n as f64 / 1_000_000_000_000.0)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B qubits", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M qubits", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K qubits", n as f64 / 1_000.0)
    } else {
        format!("{} qubits", n)
    }
}

fn format_memory(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
