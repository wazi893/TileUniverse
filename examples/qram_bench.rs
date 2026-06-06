//! QRAM Performance Benchmark
//!
//! Sprint 46.0: Measures query latency for various QRAM sizes.
//!
//! Run with:
//! ```
//! cargo run --release --example qram_bench
//! ```
//!
//! Note: Full quantum simulation limited to depth 2-3 (18 qubits max)
//! due to 32-qubit QState limit. Classical queries work for any depth.

use engine::qram::QRAMQuery;
use std::time::Instant;

/// Maximum depth for full quantum simulation (total_qubits <= 32)
/// depth=3: 3 + 1 + 2*7 = 18 qubits
const MAX_QUANTUM_DEPTH: usize = 3;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           QRAM Performance Benchmark (Sprint 46.0)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Print system info
    println!("Benchmark Configuration:");
    println!("  Classical iterations: 10,000");
    println!(
        "  Quantum basis iterations: 100 (depth <= {})",
        MAX_QUANTUM_DEPTH
    );
    println!(
        "  Superposition iterations: 10 (depth <= {})",
        MAX_QUANTUM_DEPTH
    );
    println!("  Note: Quantum sim limited to 32 qubits (depth <= 3)");
    println!();

    // Full benchmark table for small depths (quantum works)
    println!("Full Quantum Benchmark (depth 2-{}):", MAX_QUANTUM_DEPTH);
    println!(
        "┌────────┬────────┬─────────┬─────────┬──────────────┬────────────────┬──────────────────┐"
    );
    println!(
        "│ Depth  │ Cells  │ Routers │ Qubits  │ Classical    │ Quantum Basis  │ Superposition    │"
    );
    println!(
        "│        │        │         │         │ (ns/query)   │ (ns/query)     │ (ns/query)       │"
    );
    println!(
        "├────────┼────────┼─────────┼─────────┼──────────────┼────────────────┼──────────────────┤"
    );

    for depth in 2..=MAX_QUANTUM_DEPTH {
        let size = 1 << depth;
        let mut qram = QRAMQuery::new(depth);

        // Load test data
        let data: Vec<u64> = (0..size).map(|i| i as u64 * 100).collect();
        qram.load(&data);

        // Benchmark classical queries
        let iterations = 10_000;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = qram.query_classical(i % size);
        }
        let classical_ns = start.elapsed().as_nanos() / iterations as u128;

        // Benchmark quantum basis queries
        let iterations = 100;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = qram.query_basis(i % size, i as u64);
        }
        let quantum_ns = start.elapsed().as_nanos() / iterations as u128;

        // Benchmark superposition queries
        let iterations = 10;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = qram.query_superposition(i as u64);
        }
        let superpos_ns = start.elapsed().as_nanos() / iterations as u128;

        // Format with commas for readability
        println!(
            "│ {:>6} │ {:>6} │ {:>7} │ {:>7} │ {:>12} │ {:>14} │ {:>16} │",
            depth,
            size,
            qram.gate_count() / 3, // routers = gates / 3
            qram.total_qubits(),
            format_ns(classical_ns),
            format_ns(quantum_ns),
            format_ns(superpos_ns)
        );
    }

    println!(
        "└────────┴────────┴─────────┴─────────┴──────────────┴────────────────┴──────────────────┘"
    );
    println!();

    // Classical-only benchmark for larger depths
    println!("Classical-Only Benchmark (larger depths):");
    println!("┌────────┬────────┬─────────┬─────────┬──────────────┐");
    println!("│ Depth  │ Cells  │ Routers │ Qubits  │ Classical    │");
    println!("│        │        │         │         │ (ns/query)   │");
    println!("├────────┼────────┼─────────┼─────────┼──────────────┤");

    for depth in [4, 6, 8, 10, 12] {
        let size = 1 << depth;
        let mut qram = QRAMQuery::new(depth);

        let data: Vec<u64> = (0..size).map(|i| i as u64 * 100).collect();
        qram.load(&data);

        let iterations = 10_000;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = qram.query_classical(i % size);
        }
        let classical_ns = start.elapsed().as_nanos() / iterations as u128;

        println!(
            "│ {:>6} │ {:>6} │ {:>7} │ {:>7} │ {:>12} │",
            depth,
            size,
            qram.gate_count() / 3,
            qram.total_qubits(),
            format_ns(classical_ns)
        );
    }

    println!("└────────┴────────┴─────────┴─────────┴──────────────┘");
    println!();

    // Additional analysis
    println!("Analysis:");
    println!("─────────");

    // Gate count scaling
    println!();
    println!("Gate Scaling (Fredkin = 3 gates per router):");
    for depth in [2, 4, 6, 8] {
        let qram = QRAMQuery::new(depth);
        println!(
            "  Depth {}: {} gates, {} circuit depth",
            depth,
            qram.gate_count(),
            qram.circuit_depth()
        );
    }

    // Memory scaling
    println!();
    println!("Qubit Scaling (depth + 1 + 2*routers):");
    for depth in [4, 8, 10, 12] {
        if depth <= 12 {
            let qram = QRAMQuery::new(depth);
            println!(
                "  Depth {:>2}: {:>6} cells, {:>5} routers, {:>5} qubits",
                depth,
                qram.size(),
                qram.gate_count() / 3,
                qram.total_qubits()
            );
        }
    }

    // Measurement distribution test
    println!();
    println!("Measurement Distribution (depth=3, 1000 trials):");
    let mut qram = QRAMQuery::new(3);
    qram.load(&[1, 2, 3, 4, 5, 6, 7, 8]);

    let stats = qram.query_statistics(1000, 42);
    let expected = 1000.0 / 8.0;

    for (addr, count) in &stats {
        let deviation = (*count as f64 - expected) / expected * 100.0;
        let bar_len = (*count as f64 / 20.0) as usize;
        let bar: String = "█".repeat(bar_len.min(30));
        println!(
            "  Address {}: {:>4} ({:>+6.1}%) {}",
            addr, count, deviation, bar
        );
    }

    println!();
    println!("Benchmark complete.");
}

/// Format nanoseconds with commas
fn format_ns(ns: u128) -> String {
    let s = ns.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
