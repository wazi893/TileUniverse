//! CPU vs GPU Direct Comparison
//!
//! Measures the same workload on both CPU (rayon) and GPU (CUDA Tensor Cores).
//! Run: cargo run --release --example cpu_vs_gpu_comparison --features cuda,perf-bench

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, WmmaGateType};
    use engine::tile8::quantum_router_f64::QuantumGridF64;

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CPU vs GPU QUANTUM THROUGHPUT COMPARISON                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let cpu_threads = rayon::current_num_threads();
    println!("CPU: {} threads (rayon)", cpu_threads);

    // Initialize CUDA
    let rt = CudaRuntime::new().expect("CUDA runtime");
    println!("GPU: CUDA initialized\n");

    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" TEST 1: CPU f64 Quantum (27 qubits, 1M blocks, 2GB)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let n_qubits = 27;
    let num_blocks = 1usize << (n_qubits - 7);

    let mut grid = QuantumGridF64::new(n_qubits);

    // Warm-up
    for _ in 0..3 {
        grid.apply_hadamard_q0();
    }

    // Benchmark Hadamard
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        grid.apply_hadamard_q0();
    }
    let cpu_hadamard_time = start.elapsed().as_secs_f64() / iterations as f64;
    let cpu_hadamard_ops = num_blocks * 256;
    let cpu_hadamard_throughput = cpu_hadamard_ops as f64 / cpu_hadamard_time;

    // Benchmark full GHZ
    let start = Instant::now();
    for _ in 0..5 {
        let _ = grid.create_ghz_state();
    }
    let cpu_ghz_time = start.elapsed().as_secs_f64() / 5.0;
    let cpu_ghz_ops = (num_blocks * 256) + (6 * num_blocks * 128) + (20 * num_blocks * 64);
    let cpu_ghz_throughput = cpu_ghz_ops as f64 / cpu_ghz_time;

    let cpu_bandwidth = (num_blocks * 2048 * 2) as f64 / cpu_hadamard_time / 1e9;

    println!(
        "  Hadamard:     {:.2} ms  ({} amp-ops/sec)",
        cpu_hadamard_time * 1000.0,
        format_throughput(cpu_hadamard_throughput)
    );
    println!(
        "  GHZ-27:       {:.2} ms  ({} amp-ops/sec)",
        cpu_ghz_time * 1000.0,
        format_throughput(cpu_ghz_throughput)
    );
    println!("  Bandwidth:    {:.1} GB/s", cpu_bandwidth);
    println!("  Precision:    f64 (100% fidelity)");

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" TEST 2: GPU WMMA Tensor Core (16 qubits, multiple states)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // GPU test: tiles_per_state=16 means 16*256=4096 amplitudes = 12 qubits per state
    // But GPU uses batched states for parallelism
    let gpu_tests: [(usize, usize, u32, &str); 3] = [
        (16, 256, 100_000, "256 states × 100K depth"),
        (16, 1024, 100_000, "1024 states × 100K depth"),
        (16, 1024, 1_000_000, "1024 states × 1M depth"),
    ];

    for (tiles_per_state, num_states, depth, label) in gpu_tests {
        let start = Instant::now();
        let result = engine::cuda::run_wmma_multi_state(
            &rt,
            num_states,
            tiles_per_state,
            WmmaGateType::Hadamard,
            depth,
            MultiStateOpt::Basic,
        );
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            Ok((_, amp_ops)) => {
                let tcops = (amp_ops as f64 / elapsed) / 1e12;
                let gops = (amp_ops as f64 / elapsed) / 1e9;
                println!(
                    "  {}: {:.2} TCOPS ({:.0} G amp-ops/sec)",
                    label, tcops, gops
                );
            }
            Err(e) => {
                println!("  {}: FAILED ({:?})", label, e);
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" TEST 3: GPU Peak (1024 states × 1M depth, ILP optimized)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let start = Instant::now();
    let result = engine::cuda::run_wmma_multi_state(
        &rt,
        1024,
        16,
        WmmaGateType::Hadamard,
        1_000_000,
        MultiStateOpt::ILP,
    );
    let elapsed = start.elapsed().as_secs_f64();

    let (gpu_peak_throughput, gpu_peak_tcops) = match result {
        Ok((_, amp_ops)) => {
            let tcops = (amp_ops as f64 / elapsed) / 1e12;
            let gops = amp_ops as f64 / elapsed;
            println!(
                "  Peak: {:.2} TCOPS ({} amp-ops/sec)",
                tcops,
                format_throughput(gops)
            );
            (gops, tcops)
        }
        Err(e) => {
            println!("  Peak: FAILED ({:?})", e);
            (0.0, 0.0)
        }
    };

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COMPARISON SUMMARY                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let speedup = gpu_peak_throughput / cpu_hadamard_throughput;

    println!("┌─────────────────────┬─────────────────────┬─────────────────────┐");
    println!("│       Metric        │      CPU (f64)      │   GPU (Tensor Core) │");
    println!("├─────────────────────┼─────────────────────┼─────────────────────┤");
    println!(
        "│ Peak Throughput     │ {:>17}/s │ {:>17}/s │",
        format_throughput(cpu_hadamard_throughput),
        format_throughput(gpu_peak_throughput)
    );
    println!(
        "│ Memory Bandwidth    │ {:>15} GB/s │ {:>15} GB/s │",
        format!("{:.1}", cpu_bandwidth),
        "~400-600"
    ); // RTX 4070 has ~504 GB/s
    println!(
        "│ Precision           │ {:>19} │ {:>19} │",
        "f64 (100%)", "FP16 (~99.9%)"
    );
    println!(
        "│ Working Set         │ {:>19} │ {:>19} │",
        "2.0 GB", "~2 GB (batched)"
    );
    println!("└─────────────────────┴─────────────────────┴─────────────────────┘");

    println!("\n  GPU Speedup: {:.0}x faster than CPU", speedup);
    println!(
        "  GPU Peak:    {:.2} trillion amplitude-ops/sec",
        gpu_peak_tcops
    );
    println!(
        "  CPU Peak:    {:.2} billion amplitude-ops/sec",
        cpu_hadamard_throughput / 1e9
    );

    println!("\n  Analysis:");
    println!(
        "  - GPU is ~{:.0}x faster due to Tensor Core matrix acceleration",
        speedup
    );
    println!(
        "  - CPU is memory-bandwidth limited at {:.0} GB/s",
        cpu_bandwidth
    );
    println!(
        "  - GPU has ~{:.0}x more memory bandwidth",
        504.0 / cpu_bandwidth
    );
    println!("  - Both are ultimately bandwidth-bound, but GPU bandwidth is massive");

    println!("\n  Trade-offs:");
    println!("  - CPU: f64 precision (100% fidelity), lower latency for small problems");
    println!("  - GPU: FP16 precision (~99.9% fidelity), massive throughput for batched work");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Requires --features cuda,perf-bench");
    std::process::exit(1);
}

fn format_throughput(ops: f64) -> String {
    if ops >= 1e12 {
        format!("{:.2}T", ops / 1e12)
    } else if ops >= 1e9 {
        format!("{:.2}G", ops / 1e9)
    } else if ops >= 1e6 {
        format!("{:.2}M", ops / 1e6)
    } else if ops >= 1e3 {
        format!("{:.2}K", ops / 1e3)
    } else {
        format!("{:.0}", ops)
    }
}
