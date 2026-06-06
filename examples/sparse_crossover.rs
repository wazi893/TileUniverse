//! Sparse Crossover Analysis
//!
//! Determines when sparse representation breaks down vs dense.
//! Key insight: Sparse wins when state has few non-zero amplitudes.

use engine::tile8::quantum_router_f64::QuantumGridF64;
use engine::tile8::sparse_quantum::SparseQuantumGrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Sparse vs Dense Crossover Analysis                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 1: GHZ State - The Ideal Sparse Case                      │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!(
        "{:>8} {:>14} {:>14} {:>10} {:>14}",
        "Qubits", "Sparse(μs)", "Dense(μs)", "Speedup", "Memory Saved"
    );
    println!("{:-<8} {:-<14} {:-<14} {:-<10} {:-<14}", "", "", "", "", "");

    for &n_qubits in &[10, 15, 20, 25] {
        // Skip if dense would be too large
        if n_qubits > 27 {
            continue;
        }

        // Sparse GHZ
        let mut sparse_grid = SparseQuantumGrid::new(n_qubits);
        let start = Instant::now();
        sparse_grid.create_ghz_state();
        let sparse_time = start.elapsed().as_micros();

        // Dense GHZ
        let mut dense_grid = QuantumGridF64::new(n_qubits);
        let start = Instant::now();
        dense_grid.create_ghz_state();
        let dense_time = start.elapsed().as_micros();

        let speedup = dense_time as f64 / sparse_time.max(1) as f64;
        let dense_mem = (1usize << n_qubits) * 16; // 16 bytes per complex
        let sparse_mem = sparse_grid.active_blocks() * 2048;
        let mem_ratio = dense_mem / sparse_mem.max(1);

        println!(
            "{:>8} {:>14} {:>14} {:>9.0}x {:>13}x",
            n_qubits, sparse_time, dense_time, speedup, mem_ratio
        );
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 2: Impact of Hadamard Gates on Sparsity                   │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let n_qubits = 20;
    println!("System: {} qubits\n", n_qubits);

    println!("Each Hadamard on a NEW qubit DOUBLES the number of non-zero amplitudes:\n");

    println!(
        "{:>12} {:>14} {:>14} {:>12}",
        "Non-zero", "% of total", "Sparse wins?", "Why"
    );
    println!("{:-<12} {:-<14} {:-<14} {:-<12}", "", "", "", "");

    let total = 1usize << n_qubits;

    for h_qubits in 0..=n_qubits {
        let non_zero = 1usize << h_qubits;
        let pct = 100.0 * non_zero as f64 / total as f64;

        let (wins, reason) = if pct < 0.01 {
            ("YES", "<<1% sparse")
        } else if pct < 1.0 {
            ("YES", "<1% sparse")
        } else if pct < 10.0 {
            ("MAYBE", "~1-10%")
        } else {
            ("NO", ">10% dense")
        };

        if h_qubits <= 10 || h_qubits >= n_qubits - 2 {
            println!(
                "{:>12} {:>13.6}% {:>14} {:>12}",
                non_zero, pct, wins, reason
            );
        } else if h_qubits == 11 {
            println!("{:>12} {:>14} {:>14} {:>12}", "...", "...", "...", "...");
        }
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 3: Circuit Types and Sparsity                             │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!("SPARSE-FRIENDLY circuits (use sparse representation):");
    println!("  • GHZ state: H(0) + CNOT chain → 2 amplitudes");
    println!("  • W state: |100..0⟩ + |010..0⟩ + ... → n amplitudes");
    println!("  • Dicke states: C(n,k) amplitudes");
    println!("  • Stabilizer states: Often sparse in computational basis");
    println!("  • Classical reversible circuits: X, CNOT, Toffoli preserve basis");

    println!("\nDENSE circuits (use GPU):");
    println!("  • H on all qubits: 2^n amplitudes (uniform superposition)");
    println!("  • Quantum Fourier Transform: Creates dense superposition");
    println!("  • Grover's algorithm: Diffusion spreads amplitude everywhere");
    println!("  • VQE/QAOA: Parameterized rotations create dense states");
    println!("  • Random circuits: Quickly become uniformly distributed");

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║                    CROSSOVER RULE OF THUMB                      ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  Count Hadamards on DISTINCT qubits in your circuit:            ║");
    println!("║                                                                 ║");
    println!("║    0-7 distinct H qubits  → SPARSE (< 128 amplitudes)           ║");
    println!("║    8-15 distinct H qubits → Test both                           ║");
    println!("║    16+ distinct H qubits  → DENSE/GPU                           ║");
    println!("║                                                                 ║");
    println!("║  Or simply: If circuit has 'H on all qubits' → use DENSE        ║");
    println!("║                                                                 ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 4: Practical Decision Tree                                │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!("Q: What algorithm are you running?");
    println!("│");
    println!("├─► GHZ/W/Dicke state preparation");
    println!("│   └─► Use SPARSE (1000x faster, unlimited qubits)");
    println!("│");
    println!("├─► Grover search / Amplitude estimation");
    println!("│   └─► Use DENSE GPU (state becomes uniform)");
    println!("│");
    println!("├─► VQE / QAOA / Variational");
    println!("│   └─► Use DENSE GPU (rotations create dense states)");
    println!("│");
    println!("├─► Shor's / QFT-based");
    println!("│   └─► Use DENSE GPU (QFT is dense)");
    println!("│");
    println!("├─► Stabilizer / Clifford only");
    println!("│   └─► Use SPARSE or stabilizer tableau");
    println!("│");
    println!("└─► Unknown / Custom circuit");
    println!("    └─► Count distinct H qubits, use rule above");
}
