//! GHZ State Throughput Benchmark
//!
//! Measures how many GHZ states can be created and measured per second.
//!
//! Run: cargo run --release --example ghz_throughput_bench

use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              GHZ STATE THROUGHPUT BENCHMARK                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test configurations
    let configs = [
        (10, 1_000_000, "10 qubits × 1M iterations"),
        (100, 1_000_000, "100 qubits × 1M iterations"),
        (1_000, 100_000, "1K qubits × 100K iterations"),
        (10_000, 10_000, "10K qubits × 10K iterations"),
        (100_000, 1_000, "100K qubits × 1K iterations"),
        (1_000_000, 1_000, "1M qubits × 1K iterations"),
    ];

    println!(
        "{:>20} {:>12} {:>14} {:>14}",
        "Configuration", "Total Time", "Throughput", "Per-State"
    );
    println!("{}", "-".repeat(65));

    for (n_qubits, iterations, label) in configs {
        // Create grid once
        let mut grid = SparseQuantumGridHybrid::new(n_qubits);

        // Warmup
        for _ in 0..10 {
            grid.create_ghz_state();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            grid.create_ghz_state();
        }
        let elapsed = start.elapsed();

        let elapsed_sec = elapsed.as_secs_f64();
        let throughput = iterations as f64 / elapsed_sec;
        let per_state_ns = elapsed.as_nanos() as f64 / iterations as f64;

        let throughput_str = if throughput >= 1_000_000.0 {
            format!("{:.2}M/s", throughput / 1_000_000.0)
        } else if throughput >= 1_000.0 {
            format!("{:.2}K/s", throughput / 1_000.0)
        } else {
            format!("{:.2}/s", throughput)
        };

        println!(
            "{:>20} {:>10.3}s {:>14} {:>11.0} ns",
            label, elapsed_sec, throughput_str, per_state_ns
        );
    }

    // Detailed 1M qubit benchmark
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Detailed 1 Million Qubit GHZ Analysis                          │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let mut grid = SparseQuantumGridHybrid::new(1_000_000);

    // Multiple runs for accuracy
    let mut create_times = Vec::new();
    let mut verify_times = Vec::new();

    for _ in 0..100 {
        let t_create = grid.create_ghz_state();
        create_times.push(t_create as f64 / 1_000.0); // to ms

        let start = Instant::now();
        let verification = grid.verify_ghz_fidelity();
        verify_times.push(start.elapsed().as_secs_f64() * 1000.0);

        // Verify correctness on first run
        if create_times.len() == 1 {
            println!("Verification Results:");
            println!("  Fidelity:       {:.10}", verification.fidelity);
            println!("  |00...0⟩ amp²:  {:.10}", verification.amp_zero_state);
            println!("  |11...1⟩ amp²:  {:.10}", verification.amp_one_state);
            println!("  Expected:       {:.10}", verification.expected_amplitude);
            println!();
        }
    }

    let avg_create = create_times.iter().sum::<f64>() / create_times.len() as f64;
    let avg_verify = verify_times.iter().sum::<f64>() / verify_times.len() as f64;
    let min_create = create_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_create = create_times.iter().cloned().fold(0.0, f64::max);

    println!("GHZ State Creation (1M qubits, 100 runs):");
    println!("  Average:  {:.4} ms", avg_create);
    println!("  Min:      {:.4} ms", min_create);
    println!("  Max:      {:.4} ms", max_create);
    println!("  Throughput: {:.0} states/sec", 1000.0 / avg_create);

    println!("\nFidelity Verification (1M qubits, 100 runs):");
    println!("  Average:  {:.4} ms", avg_verify);

    // Compare old vs new
    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║                    PERFORMANCE COMPARISON                       ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║  Old baseline:   ~53 seconds for 1M qubit GHZ                   ║");
    println!(
        "║  New optimized:  {:.4} ms for 1M qubit GHZ                    ║",
        avg_create
    );
    let speedup = 53.0 * 1000.0 / avg_create;
    println!(
        "║  Speedup:        {:.0}x faster!                              ║",
        speedup
    );
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
