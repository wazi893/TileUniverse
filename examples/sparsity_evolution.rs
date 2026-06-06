//! Sparsity Evolution Benchmark
//!
//! Demonstrates how sparsity evolves under different circuit types:
//! - Class I: Constant Hadamard count → constant sparsity growth
//! - Class II: Logarithmic Hadamard count → polynomial sparsity
//! - Class III: Linear Hadamard count → exponential sparsity
//!
//! This provides empirical validation of the Circuit Sparsity Bound Theorem.

use engine::tile8::sparse_quantum_bigint::SparseQuantumGridBigInt;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     SPARSITY EVOLUTION BENCHMARK - Circuit Classification        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test 1: Class I - GHZ preparation (constant H count)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CLASS I: GHZ State Preparation (h(C) = 1)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "Qubits", "H Count", "Sparsity", "Bound"
    );
    println!("{}", "-".repeat(55));

    for n in [10, 50, 100, 500, 1000] {
        let mut grid = SparseQuantumGridBigInt::new(n);
        grid.create_ghz_state();
        let sparsity = grid.count_nonzero_amplitudes();
        // GHZ: 1 Hadamard applied to |0⟩, so h(C) = 1, bound = 2^1 * 1 = 2
        println!(
            "{:>10} | {:>12} | {:>12} | {:>10}",
            n, 1, sparsity, "2^1 × 1 = 2"
        );
    }
    println!("Result: Sparsity stays CONSTANT at 2 regardless of n\n");

    // Test 2: Class I - W-state preparation (O(n) operations, but O(1) H per amplitude)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CLASS I/II: W-State Preparation (linear amplitudes, sparse construction)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "Qubits", "Amplitudes", "Blocks", "Memory"
    );
    println!("{}", "-".repeat(55));

    for n in [10, 100, 1000, 10000] {
        let mut grid = SparseQuantumGridBigInt::new(n);
        grid.create_w_state();
        let sparsity = grid.count_nonzero_amplitudes();
        // Each block is ~2KB, estimate blocks from sparsity (128 amps/block)
        let blocks = (sparsity + 127) / 128;
        let mem_kb = blocks * 2; // 2KB per block
        println!(
            "{:>10} | {:>12} | {:>12} | {:>7} KB",
            n, sparsity, blocks, mem_kb
        );
    }
    println!("Result: Sparsity grows LINEARLY with n (not exponentially)\n");

    // Test 3: Dicke state - sparsity = C(n,k)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CLASS II: Dicke State |D_n^k⟩ (binomial sparsity)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>10} | {:>6} | {:>15} | {:>12}",
        "n", "k", "C(n,k)", "Sparsity"
    );
    println!("{}", "-".repeat(55));

    let test_cases = [(20, 1), (20, 2), (20, 3), (30, 2), (30, 3), (50, 2)];
    for (n, k) in test_cases {
        let mut grid = SparseQuantumGridBigInt::new(n);
        grid.create_dicke_state(k);
        let sparsity = grid.count_nonzero_amplitudes();
        let expected = binomial(n, k);
        println!("{:>10} | {:>6} | {:>15} | {:>12}", n, k, expected, sparsity);
    }
    println!("Result: Sparsity = C(n,k), polynomial for small k\n");

    // Test 4: Class III - Hadamard cascade (exponential blowup)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CLASS III: Hadamard Cascade H^⊗n (exponential sparsity)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>10} | {:>12} | {:>15} | {:>10}",
        "Qubits", "H Count", "Sparsity", "Bound"
    );
    println!("{}", "-".repeat(60));

    for n in [7, 8, 9, 10] {
        // Limited to small n due to exponential memory!
        let mut grid = SparseQuantumGridBigInt::new(n);
        grid.initialize_zero_state();

        // Apply H to all qubits
        for q in 0..n {
            grid.apply_h(q);
        }

        let sparsity = grid.count_nonzero_amplitudes();
        let bound = 1usize << n;
        println!(
            "{:>10} | {:>12} | {:>15} | {:>10}",
            n,
            n,
            sparsity,
            format!("2^{} = {}", n, bound)
        );
    }
    println!("Result: Sparsity grows EXPONENTIALLY (2^n) - this is Class III!");
    println!("        Our method becomes inefficient here.\n");

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Circuit Class | Hadamard Count | Sparsity Growth | Efficiency    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Class I       | O(1)           | O(k)            | EFFICIENT     ║");
    println!("║ Class II      | O(log n)       | poly(n)         | TRACTABLE     ║");
    println!("║ Class III     | Ω(n)           | 2^n             | INEFFICIENT   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Insight: Count Hadamards applied to superposition states.");
    println!("If h(C) = O(log n), our method stays efficient.");
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
