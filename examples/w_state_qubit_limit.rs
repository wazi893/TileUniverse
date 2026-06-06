//! W State Qubit Limit Test
//! W state has n amplitudes (linear memory), so more constrained than GHZ
use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           W STATE QUBIT LIMIT TEST                               ║");
    println!("║   |W_n⟩ has n amplitudes - linear memory scaling                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // W state memory scales linearly - be conservative
    let qubit_counts: Vec<usize> = vec![100, 1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000];

    println!(
        "{:>14} {:>12} {:>12} {:>10} {:>10} {:>12}",
        "Qubits", "Create (ms)", "Verify (ms)", "Fidelity", "Blocks", "Memory"
    );
    println!("{}", "-".repeat(75));

    for n in qubit_counts {
        let label = format_number(n);

        let _start = Instant::now();
        let mut grid = SparseQuantumGridHybrid::new(n);

        // Create W state
        let create_us = grid.create_w_state();
        let create_ms = create_us as f64 / 1000.0;

        // Verify
        let start_verify = Instant::now();
        let verification = grid.verify_w_fidelity();
        let verify_ms = start_verify.elapsed().as_secs_f64() * 1000.0;

        let blocks = grid.active_blocks();
        let memory_kb = blocks * 2; // ~2KB per block
        let memory_str = if memory_kb >= 1024 {
            format!("{:.1} MB", memory_kb as f64 / 1024.0)
        } else {
            format!("{} KB", memory_kb)
        };

        println!(
            "{:>14} {:>12.3} {:>12.3} {:>10.6} {:>10} {:>12}",
            label, create_ms, verify_ms, verification.fidelity, blocks, memory_str
        );

        // Stop if fidelity drops
        if verification.fidelity < 0.99 {
            println!("\n  ⚠ Fidelity dropped below 0.99, stopping test");
            break;
        }
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  W state |W_n⟩ = (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √n    ║");
    println!("║  Memory: O(n) - one amplitude per qubit position                ║");
    println!("║  Each block holds 128 amplitudes (~2KB)                         ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M qubits", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K qubits", n as f64 / 1_000.0)
    } else {
        format!("{} qubits", n)
    }
}
