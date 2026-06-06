// Sparse GPU Threshold Benchmark
// Finds the optimal MIN_BLOCKS_FOR_GPU for RTX 5090

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::block_sparse_state::{
        BLOCK_SHIFT, BlockSparseState, GateMatrix16, GpuBlockSparseState,
        compile_block_sparse_kernels,
    };
    use engine::cuda::{CudaRuntime, get_device_arch_string};
    use std::sync::Arc;
    use std::time::Instant;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SPARSE GPU THRESHOLD BENCHMARK                                ║");
    println!("║  Finding optimal MIN_BLOCKS_FOR_GPU for RTX 5090               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let arch = get_device_arch_string();
    println!("  Detected architecture: {}\n", arch);

    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));
    let kernels = compile_block_sparse_kernels(rt.context()).expect("compile kernels");

    // Test different qubit counts which determines block counts
    // block_count = 2^(n_qubits - 7) after full superposition
    // 14 qubits -> 128 blocks, 17 qubits -> 1024 blocks, etc.
    let qubit_tests: [(u8, &str); 8] = [
        (10, "8 blocks"),
        (12, "32 blocks"),
        (14, "128 blocks"),
        (16, "512 blocks"),
        (17, "1024 blocks"),
        (18, "2048 blocks"),
        (19, "4096 blocks"),
        (20, "8192 blocks"),
    ];
    let depth = 100; // Number of gates to apply

    println!("  Testing CPU vs GPU performance at different block counts...\n");
    println!(
        "  {:>8} | {:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "Qubits", "Blocks", "CPU (ms)", "GPU (ms)", "Speedup", "Winner"
    );
    println!(
        "  {:>8} | {:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
        "─".repeat(8),
        "─".repeat(6),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(8)
    );

    let mut crossover_found = None;

    for (n_qubits, desc) in qubit_tests {
        // Create sparse state and apply H to high qubit to create 2 blocks
        // Then apply more H gates to expand to full superposition
        let mut cpu_state = BlockSparseState::new_zero(n_qubits);

        // Apply H gates to qubits BLOCK_SHIFT and above to create blocks
        for q in BLOCK_SHIFT..n_qubits {
            cpu_state.apply_h(q);
        }

        let block_count = cpu_state.block_count();

        // Clone for GPU test
        let mut gpu_test_state = cpu_state.clone();

        // CPU benchmark
        let cpu_start = Instant::now();
        for _ in 0..depth {
            cpu_state.apply_h(3); // Within-block Hadamard
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // GPU benchmark
        let gpu_start = Instant::now();

        // Upload
        let mut gpu_state = match GpuBlockSparseState::from_host(&rt, &gpu_test_state) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "  {:>8} | {:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
                    n_qubits,
                    block_count,
                    format!("{:.2}", cpu_time),
                    "FAILED",
                    "-",
                    "CPU"
                );
                eprintln!("    Upload error: {:?}", e);
                continue;
            }
        };

        // Execute gates
        let gate = GateMatrix16::hadamard();
        for _ in 0..depth {
            if let Err(e) = engine::block_sparse_state::launch_within_block_kernel(
                &kernels,
                &mut gpu_state,
                gate,
                3, // target qubit 3
                rt.stream(),
            ) {
                println!(
                    "  {:>8} | {:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
                    n_qubits,
                    block_count,
                    format!("{:.2}", cpu_time),
                    "FAILED",
                    "-",
                    "CPU"
                );
                eprintln!("    Kernel error: {:?}", e);
                continue;
            }
        }

        // Sync
        if let Err(e) = rt.stream().synchronize() {
            println!(
                "  {:>8} | {:>6} | {:>10} | {:>10} | {:>10} | {:>8}",
                n_qubits,
                block_count,
                format!("{:.2}", cpu_time),
                "FAILED",
                "-",
                "CPU"
            );
            eprintln!("    Sync error: {:?}", e);
            continue;
        }

        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;
        let speedup = cpu_time / gpu_time;
        let winner = if speedup > 1.0 { "GPU" } else { "CPU" };

        println!(
            "  {:>8} | {:>6} | {:>9.2} | {:>9.2} | {:>9.2}x | {:>8}",
            n_qubits, block_count, cpu_time, gpu_time, speedup, winner
        );

        // Track crossover point
        if speedup > 1.0 && crossover_found.is_none() {
            crossover_found = Some(block_count);
        }
    }

    println!();
    if let Some(threshold) = crossover_found {
        println!(
            "  📊 RECOMMENDATION: Set MIN_BLOCKS_FOR_GPU = {} for RTX 5090",
            threshold
        );
        println!("     (GPU becomes faster at {} blocks)", threshold);
    } else {
        println!("  📊 NOTE: GPU was not faster than CPU at any tested block count.");
        println!("     Consider keeping MIN_BLOCKS_FOR_GPU = 1024 (default)");
    }

    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Requires --features cuda,perf-bench");
}
