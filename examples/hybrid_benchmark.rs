//! Hybrid Mode Benchmark: u128 vs BigInt Performance Comparison
//!
//! This benchmark demonstrates the performance advantage of the hybrid mode
//! which automatically selects u128 block IDs for n <= 135 qubits.
//!
//! Run: cargo run --release --example hybrid_benchmark

use engine::tile8::sparse_quantum_bigint::SparseQuantumGridBigInt;
use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     HYBRID MODE BENCHMARK - u128 vs BigInt Performance           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test various qubit counts
    let qubit_counts = [20, 50, 100, 128, 135];

    // Warm up
    {
        let mut grid = SparseQuantumGridHybrid::new(20);
        grid.create_ghz_state();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("GHZ STATE CREATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>8} | {:>12} | {:>12} | {:>10} | {:>8}",
        "Qubits", "Hybrid (µs)", "BigInt (µs)", "Speedup", "Mode"
    );
    println!("{}", "-".repeat(65));

    for &n in &qubit_counts {
        // Benchmark Hybrid mode
        let mut total_hybrid = 0u64;
        let iterations = 100;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridHybrid::new(n);
            let start = Instant::now();
            grid.create_ghz_state();
            total_hybrid += start.elapsed().as_micros() as u64;
        }
        let avg_hybrid = total_hybrid / iterations as u64;

        // Benchmark BigInt mode
        let mut total_bigint = 0u64;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridBigInt::new(n);
            let start = Instant::now();
            grid.create_ghz_state();
            total_bigint += start.elapsed().as_micros() as u64;
        }
        let avg_bigint = total_bigint / iterations as u64;

        let speedup = if avg_hybrid > 0 {
            avg_bigint as f64 / avg_hybrid as f64
        } else {
            f64::INFINITY
        };

        let mode = if n <= 135 { "u128" } else { "BigInt" };

        println!(
            "{:>8} | {:>12} | {:>12} | {:>9.2}x | {:>8}",
            n, avg_hybrid, avg_bigint, speedup, mode
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("W-STATE CREATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>8} | {:>12} | {:>12} | {:>10} | {:>8}",
        "Qubits", "Hybrid (µs)", "BigInt (µs)", "Speedup", "Amps"
    );
    println!("{}", "-".repeat(65));

    for &n in &qubit_counts {
        // Benchmark Hybrid mode
        let mut total_hybrid = 0u64;
        let iterations = 100;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridHybrid::new(n);
            let start = Instant::now();
            grid.create_w_state();
            total_hybrid += start.elapsed().as_micros() as u64;
        }
        let avg_hybrid = total_hybrid / iterations as u64;

        // Benchmark BigInt mode
        let mut total_bigint = 0u64;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridBigInt::new(n);
            let start = Instant::now();
            grid.create_w_state();
            total_bigint += start.elapsed().as_micros() as u64;
        }
        let avg_bigint = total_bigint / iterations as u64;

        let speedup = if avg_hybrid > 0 {
            avg_bigint as f64 / avg_hybrid as f64
        } else {
            f64::INFINITY
        };

        println!(
            "{:>8} | {:>12} | {:>12} | {:>9.2}x | {:>8}",
            n, avg_hybrid, avg_bigint, speedup, n
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DICKE STATE |D_n^2⟩ CREATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>8} | {:>12} | {:>12} | {:>10} | {:>8}",
        "Qubits", "Hybrid (µs)", "BigInt (µs)", "Speedup", "Amps"
    );
    println!("{}", "-".repeat(65));

    for &n in &qubit_counts {
        let k = 2; // Dicke state with k=2 ones
        let num_amps = binomial(n, k);

        // Benchmark Hybrid mode
        let mut total_hybrid = 0u64;
        let iterations = 50;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridHybrid::new(n);
            let start = Instant::now();
            grid.create_dicke_state(k);
            total_hybrid += start.elapsed().as_micros() as u64;
        }
        let avg_hybrid = total_hybrid / iterations as u64;

        // Benchmark BigInt mode
        let mut total_bigint = 0u64;

        for _ in 0..iterations {
            let mut grid = SparseQuantumGridBigInt::new(n);
            let start = Instant::now();
            grid.create_dicke_state(k);
            total_bigint += start.elapsed().as_micros() as u64;
        }
        let avg_bigint = total_bigint / iterations as u64;

        let speedup = if avg_hybrid > 0 {
            avg_bigint as f64 / avg_hybrid as f64
        } else {
            f64::INFINITY
        };

        println!(
            "{:>8} | {:>12} | {:>12} | {:>9.2}x | {:>8}",
            n, avg_hybrid, avg_bigint, speedup, num_amps
        );
    }

    // Fidelity verification
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FIDELITY VERIFICATION (Hybrid Mode)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for &n in &[50, 100, 135] {
        let mut grid = SparseQuantumGridHybrid::new(n);
        grid.create_ghz_state();
        let ghz_fidelity = grid.verify_ghz_fidelity();

        let mut grid = SparseQuantumGridHybrid::new(n);
        grid.create_w_state();
        let w_fidelity = grid.verify_w_fidelity();

        println!(
            "{:>3} qubits: GHZ fidelity = {:.15}, W fidelity = {:.15}",
            n, ghz_fidelity.fidelity, w_fidelity.fidelity
        );
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ The hybrid mode automatically selects the optimal representation:║");
    println!("║                                                                  ║");
    println!("║ • n ≤ 135 qubits: Uses u128 block IDs (O(1) operations)         ║");
    println!("║ • n > 135 qubits: Uses BigUint block IDs (O(n) operations)      ║");
    println!("║                                                                  ║");
    println!("║ This eliminates the O(n^1.5) BigInt arithmetic overhead for     ║");
    println!("║ the common case while maintaining unlimited scalability.         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}
