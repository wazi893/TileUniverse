//! Sparse vs Dense Quantum State Benchmark
//!
//! Compares:
//! 1. Dense GPU (QuantumGridF64Gpu) - 16GB for 30 qubits
//! 2. Sparse CPU (SparseQuantumGrid) - 4KB for any qubit count
//!
//! For GHZ states, sparse representation is O(1) in memory regardless of qubit count.

use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
use engine::tile8::sparse_quantum::SparseQuantumGrid;
use logic_fabric_core::cuda::CudaRuntime;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Sparse vs Dense GHZ State Benchmark                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA for dense GPU tests
    let rt = CudaRuntime::new().expect("CUDA init failed");
    if let Ok(name) = rt.device_name() {
        println!("GPU: {}", name);
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 1: Dense GPU vs Sparse CPU (30 qubits)                    │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // 30-qubit comparison
    let n_qubits = 30;
    let num_amplitudes = 1u64 << n_qubits;
    println!(
        "Configuration: {} qubits, {} amplitudes",
        n_qubits, num_amplitudes
    );
    println!(
        "Dense memory: {} GB",
        num_amplitudes * 16 / 1024 / 1024 / 1024
    );
    println!("Sparse memory: ~4 KB (2 blocks)\n");

    // Dense GPU benchmark
    println!(
        "Allocating dense GPU state ({} GB)...",
        num_amplitudes * 16 / 1024 / 1024 / 1024
    );
    let mut dense_grid = QuantumGridF64Gpu::new(&rt, n_qubits).expect("GPU alloc failed");

    // Warmup
    let _ = dense_grid.create_ghz_state_fused();

    // Benchmark dense GPU (fused - the fastest dense approach)
    let mut dense_times = Vec::new();
    for _ in 0..5 {
        let t = dense_grid.create_ghz_state_fused().unwrap();
        dense_times.push(t as f64 / 1_000.0); // Convert to ms
    }
    let dense_avg_ms = dense_times.iter().sum::<f64>() / dense_times.len() as f64;
    let dense_fidelity = dense_grid.verify_ghz_fidelity().unwrap().fidelity;
    println!(
        "Dense GPU (fused):    {:>8.3} ms  fidelity: {:.6}",
        dense_avg_ms, dense_fidelity
    );

    // Sparse CPU benchmark
    let mut sparse_grid = SparseQuantumGrid::new(n_qubits);

    // Warmup
    let _ = sparse_grid.create_ghz_state();

    // Benchmark sparse CPU
    let mut sparse_times = Vec::new();
    for _ in 0..5 {
        let t = sparse_grid.create_ghz_state();
        sparse_times.push(t as f64 / 1_000.0); // Convert to ms
    }
    let sparse_avg_ms = sparse_times.iter().sum::<f64>() / sparse_times.len() as f64;
    let sparse_verify = sparse_grid.verify_ghz_fidelity();
    println!(
        "Sparse CPU:           {:>8.3} ms  fidelity: {:.6}",
        sparse_avg_ms, sparse_verify.fidelity
    );
    println!("  Blocks allocated: {}", sparse_grid.active_blocks());

    let speedup = dense_avg_ms / sparse_avg_ms;
    println!("\nSparse speedup: {:.1}x faster than dense GPU", speedup);

    // Memory efficiency
    let dense_bytes = num_amplitudes * 16;
    let sparse_bytes = sparse_grid.active_blocks() as u64 * 2048;
    println!(
        "Memory efficiency: {:.0}x less memory",
        dense_bytes as f64 / sparse_bytes as f64
    );

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 2: Sparse CPU Scaling (u128 version, up to 135 qubits)   │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!(
        "{:>8} {:>12} {:>12} {:>8}",
        "Qubits", "Time (ms)", "Blocks", "Memory"
    );
    println!("{:-<8} {:-<12} {:-<12} {:-<8}", "", "", "", "");

    for &n in &[30, 50, 100, 128, 135] {
        let mut grid = SparseQuantumGrid::new(n);

        // Warmup
        let _ = grid.create_ghz_state();

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..3 {
            let t = grid.create_ghz_state();
            times.push(t as f64 / 1_000.0);
        }
        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        let blocks = grid.active_blocks();
        let memory_kb = blocks * 2; // 2KB per block

        println!(
            "{:>8} {:>12.3} {:>12} {:>6} KB",
            n, avg_ms, blocks, memory_kb
        );
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 3: Sparse BigInt for Million+ Qubits                      │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Use the BigInt version for very large qubit counts
    use engine::tile8::sparse_quantum_bigint::SparseQuantumGridBigInt;

    println!(
        "{:>10} {:>12} {:>12} {:>8}",
        "Qubits", "Time (ms)", "Blocks", "Memory"
    );
    println!("{:-<10} {:-<12} {:-<12} {:-<8}", "", "", "", "");

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let mut grid = SparseQuantumGridBigInt::new(n);

        // Single run for large systems
        let t = grid.create_ghz_state();
        let time_ms = t as f64 / 1_000.0;
        let blocks = grid.active_blocks();
        let memory_kb = blocks * 2;

        println!(
            "{:>10} {:>12.3} {:>12} {:>6} KB",
            n, time_ms, blocks, memory_kb
        );
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                 ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║ For GHZ states, sparse representation:                          ║");
    println!("║ - Uses constant memory (4 KB) regardless of qubit count         ║");
    println!("║ - Faster than dense GPU even at 30 qubits                       ║");
    println!("║ - Scales to 1,000,000+ qubits on commodity hardware             ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
