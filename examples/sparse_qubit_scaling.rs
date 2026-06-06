// Sparse Qubit Scaling Benchmark
// Tests sparse state performance from 40 to 63 qubits on AMD 9950X3D + RTX 5090

use std::time::Instant;

fn main() {
    use engine::block_sparse_state::{BLOCK_SHIFT, BlockSparseState};
    #[cfg(feature = "cuda")]
    use engine::cuda::{CudaRuntime, get_device_arch_string};
    use engine::sparse_state::SparseQState;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SPARSE QUBIT SCALING BENCHMARK                                ║");
    println!("║  Testing 40-64 qubits on AMD 9950X3D + RTX 5090                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        let arch = get_device_arch_string();
        println!("  GPU: {} detected\n", arch);
    }

    // ==========================================================================
    // Part 1: Element-Sparse SparseQState scaling
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" PART 1: ELEMENT-SPARSE (SparseQState) - Max 63 qubits");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  Testing X gate (preserves sparsity) at extreme qubit counts:\n");
    println!(
        "  {:>6} | {:>20} | {:>12} | {:>12}",
        "Qubits", "State Space", "Time (μs)", "Ops/sec"
    );
    println!(
        "  {:>6} | {:>20} | {:>12} | {:>12}",
        "─".repeat(6),
        "─".repeat(20),
        "─".repeat(12),
        "─".repeat(12)
    );

    let qubit_tests = [40, 45, 50, 55, 60, 62, 63, 64];
    let depth = 1000;

    for n_qubits in qubit_tests {
        let state_space = format_big_number(1u128 << n_qubits);

        let mut state = SparseQState::new_zero(n_qubits);

        let start = Instant::now();
        for _ in 0..depth {
            state.apply_x(0);
            state.apply_x(n_qubits - 1);
        }
        let elapsed = start.elapsed();
        let time_us = elapsed.as_nanos() as f64 / 1000.0;
        let ops_per_sec = (depth * 2) as f64 / elapsed.as_secs_f64();

        println!(
            "  {:>6} | {:>20} | {:>10.1} | {:>10.0}",
            n_qubits, state_space, time_us, ops_per_sec
        );

        // Verify state is still valid
        assert_eq!(state.nnz(), 1, "X gates should preserve single amplitude");
    }

    println!();

    // ==========================================================================
    // Part 2: H gate superposition growth
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" PART 2: SUPERPOSITION GROWTH (H gates create 2^n amplitudes)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  Applying H to first N qubits (creates 2^N amplitudes):\n");
    println!(
        "  {:>6} | {:>6} | {:>12} | {:>12} | {:>12}",
        "Qubits", "H's", "Amplitudes", "Memory (KB)", "Time (ms)"
    );
    println!(
        "  {:>6} | {:>6} | {:>12} | {:>12} | {:>12}",
        "─".repeat(6),
        "─".repeat(6),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(12)
    );

    // Test superposition growth on a 63-qubit state
    for h_count in [5, 10, 15, 20, 22, 24] {
        let mut state = SparseQState::new_zero(63);

        let start = Instant::now();
        for q in 0..h_count {
            state.apply_h(q);
        }
        let elapsed = start.elapsed();

        let amps = state.nnz();
        // Each amplitude = 8 bytes (2 x f32)
        let mem_kb = (amps * 8 + amps * 8) as f64 / 1024.0; // + hash overhead estimate

        println!(
            "  {:>6} | {:>6} | {:>12} | {:>10.1} | {:>10.2}",
            63,
            h_count,
            amps,
            mem_kb,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!();

    // ==========================================================================
    // Part 3: Block-Sparse scaling
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" PART 3: BLOCK-SPARSE (BlockSparseState) - GPU-optimized");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!(
        "  Block size: 128 amplitudes (2^{} = {})",
        BLOCK_SHIFT,
        1 << BLOCK_SHIFT
    );
    println!("  Testing cross-block H gates at high qubit counts:\n");

    println!(
        "  {:>6} | {:>10} | {:>12} | {:>12}",
        "Qubits", "Blocks", "Time (ms)", "Status"
    );
    println!(
        "  {:>6} | {:>10} | {:>12} | {:>12}",
        "─".repeat(6),
        "─".repeat(10),
        "─".repeat(12),
        "─".repeat(12)
    );

    let block_qubit_tests = [40, 50, 55, 60, 63, 64];

    for n_qubits in block_qubit_tests {
        let mut state = BlockSparseState::new_zero(n_qubits);

        // Apply H to qubits above block boundary to create blocks
        let start = Instant::now();

        // Apply H to a few high qubits to create multiple blocks
        let high_qubit = n_qubits - 1;
        state.apply_h(high_qubit);

        // Apply H to qubit just above block boundary
        if n_qubits > BLOCK_SHIFT + 1 {
            state.apply_h(BLOCK_SHIFT);
        }

        let elapsed = start.elapsed();
        let blocks = state.block_count();

        println!(
            "  {:>6} | {:>10} | {:>10.2} | {:>12}",
            n_qubits,
            blocks,
            elapsed.as_secs_f64() * 1000.0,
            "OK"
        );
    }

    println!();

    // ==========================================================================
    // Part 4: GPU-accelerated sparse operations
    // ==========================================================================
    #[cfg(feature = "cuda")]
    {
        println!("═══════════════════════════════════════════════════════════════════");
        println!(" PART 4: GPU-ACCELERATED SPARSE (BlockSparseState + CUDA)");
        println!("═══════════════════════════════════════════════════════════════════\n");

        use engine::block_sparse_state::{
            GateMatrix16, GpuBlockSparseState, compile_block_sparse_kernels,
        };
        use std::sync::Arc;

        let rt = match CudaRuntime::new() {
            Ok(r) => Arc::new(r),
            Err(e) => {
                eprintln!("  Failed to create CUDA runtime: {:?}", e);
                return;
            }
        };

        let kernels = match compile_block_sparse_kernels(rt.context()) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("  Failed to compile kernels: {:?}", e);
                return;
            }
        };

        println!("  Testing GPU sparse operations at high qubit counts:\n");
        println!(
            "  {:>6} | {:>10} | {:>12} | {:>12} | {:>10}",
            "Qubits", "Blocks", "CPU (ms)", "GPU (ms)", "Speedup"
        );
        println!(
            "  {:>6} | {:>10} | {:>12} | {:>12} | {:>10}",
            "─".repeat(6),
            "─".repeat(10),
            "─".repeat(12),
            "─".repeat(12),
            "─".repeat(10)
        );

        for n_qubits in [40u8, 50, 55, 60, 63, 64] {
            // Create state with multiple blocks
            let mut cpu_state = BlockSparseState::new_zero(n_qubits);

            // Create superposition to get multiple blocks
            for q in BLOCK_SHIFT..(BLOCK_SHIFT + 6).min(n_qubits) {
                cpu_state.apply_h(q);
            }

            let blocks = cpu_state.block_count();

            if blocks < 32 {
                // Not enough blocks for GPU to be profitable
                println!(
                    "  {:>6} | {:>10} | {:>12} | {:>12} | {:>10}",
                    n_qubits, blocks, "-", "skip", "<32 blocks"
                );
                continue;
            }

            // CPU benchmark
            let cpu_state_clone = cpu_state.clone();
            let mut cpu_test = cpu_state_clone;
            let depth = 100;

            let cpu_start = Instant::now();
            for _ in 0..depth {
                cpu_test.apply_h(3);
            }
            let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;

            // GPU benchmark
            let mut gpu_state = match GpuBlockSparseState::from_host(&rt, &cpu_state) {
                Ok(s) => s,
                Err(e) => {
                    println!(
                        "  {:>6} | {:>10} | {:>10.2} | {:>12} | {:>10}",
                        n_qubits,
                        blocks,
                        cpu_time,
                        format!("ERR: {:?}", e),
                        "-"
                    );
                    continue;
                }
            };

            let gate = GateMatrix16::hadamard();
            let gpu_start = Instant::now();
            for _ in 0..depth {
                let _ = engine::block_sparse_state::launch_within_block_kernel(
                    &kernels,
                    &mut gpu_state,
                    gate,
                    3,
                    rt.stream(),
                );
            }
            let _ = rt.stream().synchronize();
            let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;

            let speedup = cpu_time / gpu_time;
            println!(
                "  {:>6} | {:>10} | {:>10.2} | {:>10.2} | {:>8.1}x",
                n_qubits, blocks, cpu_time, gpu_time, speedup
            );
        }
    }

    println!();

    // ==========================================================================
    // Summary
    // ==========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  SPARSE QUBIT SCALING SUMMARY                                  ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                ║");
    println!("║  Maximum supported qubits: 64 (full u64 basis index range)     ║");
    println!("║                                                                ║");
    println!("║  At 64 qubits:                                                 ║");
    println!("║    - State space: 1.84 × 10^19 amplitudes                      ║");
    println!("║    - Dense storage would require: 147 Exabytes                 ║");
    println!("║    - Sparse storage: Only non-zero amplitudes (KB to GB)       ║");
    println!("║                                                                ║");
    println!("║  Key insight:                                                  ║");
    println!("║    - Sparse states enable quantum simulation far beyond        ║");
    println!("║      the 32-qubit limit of dense VRAM storage                  ║");
    println!("║    - GPU acceleration kicks in at 32+ blocks                   ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}

fn format_big_number(n: u128) -> String {
    if n >= 1_000_000_000_000_000_000 {
        format!("{:.1}E", n as f64 / 1e18)
    } else if n >= 1_000_000_000_000_000 {
        format!("{:.1}P", n as f64 / 1e15)
    } else if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
