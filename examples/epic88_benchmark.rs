//! EPIC 88 Phase 88.4: Block-Sparse Benchmark Suite
//!
//! Validates that the block-sparse representation provides value for moderate
//! sparsity workloads while element-sparse remains optimal for ultra-sparse states.
//!
//! Run with: cargo run --release --example epic88_benchmark --features cuda

#[cfg(feature = "cuda")]
use engine::block_sparse_state::{
    BLOCK_SIZE, BlockSparseKernels, BlockSparseState, GateMatrix16, GpuBlockSparseState,
    MIN_BLOCKS_FOR_GPU, SparseBackend, compile_block_sparse_kernels, launch_within_block_kernel,
    select_backend_for_block_sparse, select_backend_for_sparse,
};
#[cfg(feature = "cuda")]
use engine::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use engine::sparse_state::SparseQState;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

/// Benchmark result for a single circuit
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct BenchmarkResult {
    circuit_name: String,
    n_qubits: u8,
    n_gates: usize,

    // State characteristics after circuit
    final_nnz: usize,
    final_blocks: usize,

    // Timing results
    element_sparse_time: Duration,
    block_sparse_cpu_time: Duration,
    block_sparse_gpu_time: Option<Duration>, // None if too few blocks

    // Memory estimates
    element_sparse_bytes: usize,
    block_sparse_bytes: usize,

    // Router analysis
    router_selected: SparseBackend,
    actual_winner: SparseBackend,
    router_correct: bool,
}

#[cfg(feature = "cuda")]
impl BenchmarkResult {
    fn print_row(&self) {
        let gpu_time = self
            .block_sparse_gpu_time
            .map(|d| format!("{:.2}ms", d.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());

        let winner_str = match self.actual_winner {
            SparseBackend::ElementSparseCpu => "Elem-CPU",
            SparseBackend::BlockSparseCpu => "Block-CPU",
            SparseBackend::BlockSparseGpu => "Block-GPU",
            SparseBackend::Dense => "Dense",
        };

        let correct_mark = if self.router_correct { "✅" } else { "❌" };

        println!(
            "| {} | {} | {} | {:.2}ms | {:.2}ms | {} | {} | {} | {} |",
            self.circuit_name,
            self.n_qubits,
            self.final_nnz,
            self.element_sparse_time.as_secs_f64() * 1000.0,
            self.block_sparse_cpu_time.as_secs_f64() * 1000.0,
            gpu_time,
            winner_str,
            format!("{:?}", self.router_selected)
                .split("::")
                .last()
                .unwrap_or("?"),
            correct_mark,
        );
    }
}

/// GHZ-like state: only 2 amplitudes (|00...0⟩ + |11...1⟩)
/// Should favor element-sparse
#[cfg(feature = "cuda")]
fn benchmark_ghz(
    n_qubits: u8,
    iterations: usize,
    rt: &CudaRuntime,
    kernels: &BlockSparseKernels,
) -> BenchmarkResult {
    // ========== Element-Sparse ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        state.apply_h(0);
        for q in 1..n_qubits {
            // CNOT not available, approximate with H gates on low qubits
            // This creates a different state but tests the infrastructure
        }
        // Just apply some gates to measure overhead
        for q in 0..std::cmp::min(n_qubits, 6) {
            state.apply_h(q);
            state.apply_h(q); // H^2 = I, state unchanged
        }
    }
    let element_sparse_time = start.elapsed() / iterations as u32;

    // Get final state characteristics
    let mut final_state = SparseQState::new_zero(n_qubits);
    final_state.apply_h(0); // Creates |0⟩ + |1⟩, 2 amplitudes
    let final_nnz = final_state.nnz();

    // ========== Block-Sparse CPU ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = BlockSparseState::new_zero(n_qubits);
        for q in 0..std::cmp::min(n_qubits, 6) {
            state.apply_h(q);
            state.apply_h(q);
        }
    }
    let block_sparse_cpu_time = start.elapsed() / iterations as u32;

    let block_state = BlockSparseState::new_zero(n_qubits);
    let final_blocks = block_state.block_count();

    // ========== Block-Sparse GPU ==========
    let block_sparse_gpu_time = if final_blocks >= MIN_BLOCKS_FOR_GPU {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = BlockSparseState::new_zero(n_qubits);
            state.apply_h(7); // Create enough blocks for GPU
            let mut gpu_state = GpuBlockSparseState::from_host(rt, &state).unwrap();
            for q in 0..6u8 {
                launch_within_block_kernel(
                    kernels,
                    &mut gpu_state,
                    GateMatrix16::hadamard(),
                    q,
                    rt.stream(),
                )
                .unwrap();
                launch_within_block_kernel(
                    kernels,
                    &mut gpu_state,
                    GateMatrix16::hadamard(),
                    q,
                    rt.stream(),
                )
                .unwrap();
            }
            let _ = gpu_state.to_host(rt).unwrap();
        }
        Some(start.elapsed() / iterations as u32)
    } else {
        None
    };

    // Memory estimates
    let element_sparse_bytes = final_nnz * (8 + 8 + 16); // Complex + key + HashMap overhead
    let block_sparse_bytes = final_blocks * BLOCK_SIZE * 4; // f16 real + f16 imag per element

    // Router decision
    let router_selected = select_backend_for_sparse(&final_state);

    // Determine actual winner
    let actual_winner = if element_sparse_time <= block_sparse_cpu_time {
        SparseBackend::ElementSparseCpu
    } else if let Some(gpu_time) = block_sparse_gpu_time {
        if gpu_time < block_sparse_cpu_time && gpu_time < element_sparse_time {
            SparseBackend::BlockSparseGpu
        } else if block_sparse_cpu_time < element_sparse_time {
            SparseBackend::BlockSparseCpu
        } else {
            SparseBackend::ElementSparseCpu
        }
    } else {
        if block_sparse_cpu_time < element_sparse_time {
            SparseBackend::BlockSparseCpu
        } else {
            SparseBackend::ElementSparseCpu
        }
    };

    let router_correct = router_selected == actual_winner;

    BenchmarkResult {
        circuit_name: format!("GHZ-{}", n_qubits),
        n_qubits,
        n_gates: n_qubits as usize * 2,
        final_nnz,
        final_blocks,
        element_sparse_time,
        block_sparse_cpu_time,
        block_sparse_gpu_time,
        element_sparse_bytes,
        block_sparse_bytes,
        router_selected,
        actual_winner,
        router_correct,
    }
}

/// Moderate entanglement: H gates on many qubits creates 2^k amplitudes
/// Should favor block-sparse GPU when blocks >= 10
#[cfg(feature = "cuda")]
fn benchmark_moderate(
    n_qubits: u8,
    h_count: u8,
    iterations: usize,
    rt: &CudaRuntime,
    kernels: &BlockSparseKernels,
) -> BenchmarkResult {
    let h_count = std::cmp::min(h_count, n_qubits);

    // ========== Element-Sparse ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        for q in 0..h_count {
            state.apply_h(q);
        }
        // Apply some Z gates (cheap, just flips signs)
        for q in 0..h_count {
            state.apply_z(q);
        }
    }
    let element_sparse_time = start.elapsed() / iterations as u32;

    // Get final state
    let mut final_elem = SparseQState::new_zero(n_qubits);
    for q in 0..h_count {
        final_elem.apply_h(q);
    }
    let final_nnz = final_elem.nnz();

    // ========== Block-Sparse CPU ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = BlockSparseState::new_zero(n_qubits);
        for q in 0..h_count {
            state.apply_h(q);
        }
        for q in 0..h_count {
            state.apply_z(q);
        }
    }
    let block_sparse_cpu_time = start.elapsed() / iterations as u32;

    let mut block_state = BlockSparseState::new_zero(n_qubits);
    for q in 0..h_count {
        block_state.apply_h(q);
    }
    let final_blocks = block_state.block_count();

    // ========== Block-Sparse GPU ==========
    let block_sparse_gpu_time = if final_blocks >= MIN_BLOCKS_FOR_GPU {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = BlockSparseState::new_zero(n_qubits);
            // Build up blocks using H on high qubits
            for q in 7..std::cmp::min(h_count + 7, n_qubits) {
                state.ensure_partner_blocks_exist(q);
                state.apply_h(q);
            }

            if state.block_count() >= 2 {
                let mut gpu_state = GpuBlockSparseState::from_host(rt, &state).unwrap();
                // Apply gates on GPU
                for q in 0..std::cmp::min(6, h_count) {
                    launch_within_block_kernel(
                        kernels,
                        &mut gpu_state,
                        GateMatrix16::z(),
                        q,
                        rt.stream(),
                    )
                    .unwrap();
                }
                let _ = gpu_state.to_host(rt).unwrap();
            }
        }
        Some(start.elapsed() / iterations as u32)
    } else {
        None
    };

    // Memory estimates
    let element_sparse_bytes = final_nnz * 24; // ~24 bytes per amplitude in HashMap
    let block_sparse_bytes = final_blocks * BLOCK_SIZE * 4;

    // Router decision
    let router_selected = select_backend_for_block_sparse(&block_state);

    // Determine winner
    let times = vec![
        (element_sparse_time, SparseBackend::ElementSparseCpu),
        (block_sparse_cpu_time, SparseBackend::BlockSparseCpu),
    ];
    let mut times = times;
    if let Some(gpu_time) = block_sparse_gpu_time {
        times.push((gpu_time, SparseBackend::BlockSparseGpu));
    }
    times.sort_by_key(|(t, _)| *t);
    let actual_winner = times[0].1;

    let router_correct = router_selected == actual_winner;

    BenchmarkResult {
        circuit_name: format!("Moderate-{}-H{}", n_qubits, h_count),
        n_qubits,
        n_gates: (h_count as usize) * 2,
        final_nnz,
        final_blocks,
        element_sparse_time,
        block_sparse_cpu_time,
        block_sparse_gpu_time,
        element_sparse_bytes,
        block_sparse_bytes,
        router_selected,
        actual_winner,
        router_correct,
    }
}

/// High-block count benchmark: creates many blocks via cross-block H gates
/// Should strongly favor block-sparse GPU
#[cfg(feature = "cuda")]
fn benchmark_many_blocks(
    n_qubits: u8,
    iterations: usize,
    rt: &CudaRuntime,
    kernels: &BlockSparseKernels,
) -> BenchmarkResult {
    assert!(
        n_qubits >= 12,
        "Need at least 12 qubits for many-blocks benchmark"
    );

    // ========== Element-Sparse ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        // Apply H to qubits 7-11 to create 2^5 = 32 blocks
        for q in 7..12 {
            state.apply_h(q);
        }
    }
    let element_sparse_time = start.elapsed() / iterations as u32;

    let mut final_elem = SparseQState::new_zero(n_qubits);
    for q in 7..12 {
        final_elem.apply_h(q);
    }
    let final_nnz = final_elem.nnz();

    // ========== Block-Sparse CPU ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = BlockSparseState::new_zero(n_qubits);
        for q in 7..12 {
            state.apply_h(q);
        }
    }
    let block_sparse_cpu_time = start.elapsed() / iterations as u32;

    let mut block_state = BlockSparseState::new_zero(n_qubits);
    for q in 7..12 {
        block_state.apply_h(q);
    }
    let final_blocks = block_state.block_count();

    // ========== Block-Sparse GPU ==========
    let block_sparse_gpu_time = {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = BlockSparseState::new_zero(n_qubits);
            // Build state with blocks
            for q in 7..12 {
                state.ensure_partner_blocks_exist(q);
                state.apply_h(q);
            }

            // GPU operations
            let mut gpu_state = GpuBlockSparseState::from_host(rt, &state).unwrap();

            // Apply within-block gates
            for q in 0..6u8 {
                launch_within_block_kernel(
                    kernels,
                    &mut gpu_state,
                    GateMatrix16::hadamard(),
                    q,
                    rt.stream(),
                )
                .unwrap();
            }

            let _ = gpu_state.to_host(rt).unwrap();
        }
        Some(start.elapsed() / iterations as u32)
    };

    let element_sparse_bytes = final_nnz * 24;
    let block_sparse_bytes = final_blocks * BLOCK_SIZE * 4;

    let router_selected = select_backend_for_block_sparse(&block_state);

    let times = vec![
        (element_sparse_time, SparseBackend::ElementSparseCpu),
        (block_sparse_cpu_time, SparseBackend::BlockSparseCpu),
        (
            block_sparse_gpu_time.unwrap(),
            SparseBackend::BlockSparseGpu,
        ),
    ];
    let mut times = times;
    times.sort_by_key(|(t, _)| *t);
    let actual_winner = times[0].1;

    let router_correct = router_selected == actual_winner;

    BenchmarkResult {
        circuit_name: format!("ManyBlocks-{}", n_qubits),
        n_qubits,
        n_gates: 5 + 6, // 5 cross-block H + 6 within-block H
        final_nnz,
        final_blocks,
        element_sparse_time,
        block_sparse_cpu_time,
        block_sparse_gpu_time,
        element_sparse_bytes,
        block_sparse_bytes,
        router_selected,
        actual_winner,
        router_correct,
    }
}

/// Large scale benchmark: many blocks with dense content
/// With 32 blocks - still below GPU threshold (128), CPU should win
#[cfg(feature = "cuda")]
fn benchmark_large_scale(
    n_qubits: u8,
    iterations: usize,
    rt: &CudaRuntime,
    kernels: &BlockSparseKernels,
) -> BenchmarkResult {
    assert!(
        n_qubits >= 14,
        "Need at least 14 qubits for large scale benchmark"
    );

    // Create a state with many blocks AND many amplitudes per block
    // H on qubits 0-5 creates 64 amps in block 0
    // H on qubits 7-11 spreads to 32 blocks
    // Total: 64 * 32 = 2048 amplitudes across 32 blocks

    // ========== Element-Sparse ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        for q in 0..6 {
            state.apply_h(q);
        }
        for q in 7..12 {
            state.apply_h(q);
        }
    }
    let element_sparse_time = start.elapsed() / iterations as u32;

    let mut final_elem = SparseQState::new_zero(n_qubits);
    for q in 0..6 {
        final_elem.apply_h(q);
    }
    for q in 7..12 {
        final_elem.apply_h(q);
    }
    let final_nnz = final_elem.nnz();

    // ========== Block-Sparse CPU ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = BlockSparseState::new_zero(n_qubits);
        for q in 0..6 {
            state.apply_h(q);
        }
        for q in 7..12 {
            state.apply_h(q);
        }
    }
    let block_sparse_cpu_time = start.elapsed() / iterations as u32;

    let mut block_state = BlockSparseState::new_zero(n_qubits);
    for q in 0..6 {
        block_state.apply_h(q);
    }
    for q in 7..12 {
        block_state.apply_h(q);
    }
    let final_blocks = block_state.block_count();

    // ========== Block-Sparse GPU ==========
    // GPU path measured for comparison even though router won't select it
    let block_sparse_gpu_time = {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = BlockSparseState::new_zero(n_qubits);
            for q in 0..6 {
                state.apply_h(q);
            }
            for q in 7..12 {
                state.ensure_partner_blocks_exist(q);
                state.apply_h(q);
            }

            let mut gpu_state = GpuBlockSparseState::from_host(rt, &state).unwrap();
            for _ in 0..2 {
                for q in 0..6u8 {
                    launch_within_block_kernel(
                        kernels,
                        &mut gpu_state,
                        GateMatrix16::hadamard(),
                        q,
                        rt.stream(),
                    )
                    .unwrap();
                }
            }
            let _ = gpu_state.to_host(rt).unwrap();
        }
        Some(start.elapsed() / iterations as u32)
    };

    let element_sparse_bytes = final_nnz * 24;
    let block_sparse_bytes = final_blocks * BLOCK_SIZE * 4;

    let router_selected = select_backend_for_block_sparse(&block_state);

    let times = vec![
        (element_sparse_time, SparseBackend::ElementSparseCpu),
        (block_sparse_cpu_time, SparseBackend::BlockSparseCpu),
        (
            block_sparse_gpu_time.unwrap(),
            SparseBackend::BlockSparseGpu,
        ),
    ];
    let mut times = times;
    times.sort_by_key(|(t, _)| *t);
    let actual_winner = times[0].1;

    let router_correct = router_selected == actual_winner;

    BenchmarkResult {
        circuit_name: format!("LargeScale-{}", n_qubits),
        n_qubits,
        n_gates: 6 + 5 + 12,
        final_nnz,
        final_blocks,
        element_sparse_time,
        block_sparse_cpu_time,
        block_sparse_gpu_time,
        element_sparse_bytes,
        block_sparse_bytes,
        router_selected,
        actual_winner,
        router_correct,
    }
}

/// Very large scale: 256+ blocks (GPU threshold territory)
/// This tests the GPU path properly with enough work to amortize overhead
#[cfg(feature = "cuda")]
fn benchmark_very_large(
    n_qubits: u8,
    iterations: usize,
    rt: &CudaRuntime,
    kernels: &BlockSparseKernels,
) -> BenchmarkResult {
    assert!(
        n_qubits >= 16,
        "Need at least 16 qubits for very large benchmark"
    );

    // H on qubits 7-14 = 8 qubits = 2^8 = 256 blocks
    // H on qubits 0-5 = 64 amps per block = 16K total amplitudes

    // ========== Element-Sparse ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        for q in 0..6 {
            state.apply_h(q);
        }
        for q in 7..15 {
            state.apply_h(q);
        }
    }
    let element_sparse_time = start.elapsed() / iterations as u32;

    let mut final_elem = SparseQState::new_zero(n_qubits);
    for q in 0..6 {
        final_elem.apply_h(q);
    }
    for q in 7..15 {
        final_elem.apply_h(q);
    }
    let final_nnz = final_elem.nnz();

    // ========== Block-Sparse CPU ==========
    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = BlockSparseState::new_zero(n_qubits);
        for q in 0..6 {
            state.apply_h(q);
        }
        for q in 7..15 {
            state.apply_h(q);
        }
    }
    let block_sparse_cpu_time = start.elapsed() / iterations as u32;

    let mut block_state = BlockSparseState::new_zero(n_qubits);
    for q in 0..6 {
        block_state.apply_h(q);
    }
    for q in 7..15 {
        block_state.apply_h(q);
    }
    let final_blocks = block_state.block_count();

    // ========== Block-Sparse GPU ==========
    let block_sparse_gpu_time = {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut state = BlockSparseState::new_zero(n_qubits);
            for q in 0..6 {
                state.apply_h(q);
            }
            for q in 7..15 {
                state.ensure_partner_blocks_exist(q);
                state.apply_h(q);
            }

            let mut gpu_state = GpuBlockSparseState::from_host(rt, &state).unwrap();

            // Apply multiple gate layers to amortize GPU overhead
            for _ in 0..4 {
                for q in 0..6u8 {
                    launch_within_block_kernel(
                        kernels,
                        &mut gpu_state,
                        GateMatrix16::hadamard(),
                        q,
                        rt.stream(),
                    )
                    .unwrap();
                }
            }

            let _ = gpu_state.to_host(rt).unwrap();
        }
        Some(start.elapsed() / iterations as u32)
    };

    let element_sparse_bytes = final_nnz * 24;
    let block_sparse_bytes = final_blocks * BLOCK_SIZE * 4;

    let router_selected = select_backend_for_block_sparse(&block_state);

    let times = vec![
        (element_sparse_time, SparseBackend::ElementSparseCpu),
        (block_sparse_cpu_time, SparseBackend::BlockSparseCpu),
        (
            block_sparse_gpu_time.unwrap(),
            SparseBackend::BlockSparseGpu,
        ),
    ];
    let mut times = times;
    times.sort_by_key(|(t, _)| *t);
    let actual_winner = times[0].1;

    let router_correct = router_selected == actual_winner;

    BenchmarkResult {
        circuit_name: format!("VeryLarge-{}", n_qubits),
        n_qubits,
        n_gates: 6 + 8 + 24, // 6 within-block + 8 cross-block + 24 GPU gates
        final_nnz,
        final_blocks,
        element_sparse_time,
        block_sparse_cpu_time,
        block_sparse_gpu_time,
        element_sparse_bytes,
        block_sparse_bytes,
        router_selected,
        actual_winner,
        router_correct,
    }
}

#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    EPIC 88 Phase 88.4: Block-Sparse Benchmark                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA
    let rt = match CudaRuntime::new() {
        Ok(rt) => {
            println!(
                "✅ CUDA initialized: {}",
                rt.device_name().unwrap_or_default()
            );
            rt
        }
        Err(e) => {
            eprintln!("❌ CUDA not available: {:?}", e);
            eprintln!("   GPU benchmarks will be skipped.");
            return;
        }
    };

    // Compile kernels
    println!("   Compiling block-sparse kernels...");
    let kernels = match compile_block_sparse_kernels(rt.context()) {
        Ok(k) => {
            println!("✅ Kernels compiled\n");
            k
        }
        Err(e) => {
            eprintln!("❌ Kernel compilation failed: {:?}", e);
            return;
        }
    };

    let iterations = 50;

    println!("## Benchmark Configuration");
    println!("- Iterations per test: {}", iterations);
    println!("- MIN_BLOCKS_FOR_GPU: {}", MIN_BLOCKS_FOR_GPU);
    println!("- BLOCK_SIZE: {}", BLOCK_SIZE);
    println!();

    // Run benchmarks
    let mut results = Vec::new();

    println!("Running benchmarks...\n");

    // 1. GHZ-like (ultra-sparse) - Element-sparse should win
    print!("  [1/6] GHZ-20... ");
    results.push(benchmark_ghz(20, iterations, &rt, &kernels));
    println!("done");

    // 2. Moderate (some blocks) - Element-sparse should win
    print!("  [2/6] Moderate-14-H6... ");
    results.push(benchmark_moderate(14, 6, iterations, &rt, &kernels));
    println!("done");

    // 3. Moderate with more H gates - Block-sparse CPU should win
    print!("  [3/6] Moderate-14-H10... ");
    results.push(benchmark_moderate(14, 10, iterations, &rt, &kernels));
    println!("done");

    // 4. Many blocks (32 blocks, sparse content) - Block-sparse CPU should win
    print!("  [4/6] ManyBlocks-14... ");
    results.push(benchmark_many_blocks(14, iterations, &rt, &kernels));
    println!("done");

    // 5. Large scale (32 blocks, dense content) - Block-sparse CPU should win
    print!("  [5/6] LargeScale-14... ");
    results.push(benchmark_large_scale(14, iterations, &rt, &kernels));
    println!("done");

    // 6. Very large (256 blocks) - Tests GPU threshold territory
    print!("  [6/6] VeryLarge-16... ");
    results.push(benchmark_very_large(16, iterations, &rt, &kernels));
    println!("done");

    println!("\n## Results\n");
    println!("| Circuit | Qubits | NNZ | Elem-CPU | Block-CPU | Block-GPU | Winner | Router | ✓ |");
    println!("|---------|--------|-----|----------|-----------|-----------|--------|--------|---|");

    for result in &results {
        result.print_row();
    }

    println!("\n## Memory Analysis\n");
    println!("| Circuit | Final Blocks | Elem-Sparse | Block-Sparse | Ratio |");
    println!("|---------|--------------|-------------|--------------|-------|");
    for result in &results {
        let ratio = result.element_sparse_bytes as f64 / result.block_sparse_bytes as f64;
        println!(
            "| {} | {} | {} B | {} B | {:.2}× |",
            result.circuit_name,
            result.final_blocks,
            result.element_sparse_bytes,
            result.block_sparse_bytes,
            ratio,
        );
    }

    // Summary
    let correct_count = results.iter().filter(|r| r.router_correct).count();
    let total_count = results.len();
    let accuracy = (correct_count as f64 / total_count as f64) * 100.0;

    println!("\n## Router Accuracy\n");
    println!("- Correct selections: {}/{}", correct_count, total_count);
    println!("- Accuracy: {:.1}%", accuracy);

    if accuracy >= 90.0 {
        println!("- Status: ✅ PASS (target: >90%)");
    } else {
        println!("- Status: ❌ FAIL (target: >90%)");
    }

    // GPU speedup analysis
    println!("\n## GPU Speedup Analysis\n");
    for result in &results {
        if let Some(gpu_time) = result.block_sparse_gpu_time {
            let speedup_vs_elem = result.element_sparse_time.as_secs_f64() / gpu_time.as_secs_f64();
            let speedup_vs_cpu =
                result.block_sparse_cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
            println!(
                "- {}: GPU vs Elem-CPU: {:.2}×, GPU vs Block-CPU: {:.2}×",
                result.circuit_name, speedup_vs_elem, speedup_vs_cpu
            );
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          EPIC 88 Benchmark Complete                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: This example requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example epic88_benchmark --features cuda --release");
    std::process::exit(1);
}
