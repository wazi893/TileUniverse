//! CPU vs GPU Benchmark for F64 Quantum Simulation
//!
//! Compares performance of:
//! - CPU: QuantumGridF64 with rayon parallelization
//! - GPU: QuantumGridF64Gpu with CUDA acceleration
//!
//! Usage:
//!   cargo run --release --features cuda --example cpu_vs_gpu_f64_bench

// CPU implementation
use engine::tile8::quantum_router_f64::QuantumGridF64;

// GPU implementation (requires cuda feature)
#[cfg(feature = "cuda")]
use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use std::time::Instant;

fn benchmark_cpu(n_qubits: usize, iterations: usize) -> (f64, f64) {
    let mut grid = QuantumGridF64::new(n_qubits);

    // Warmup
    let _ = grid.create_ghz_state();

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let time_us = grid.create_ghz_state();
        times.push(time_us as f64 / 1000.0); // Convert to ms
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);

    // Verify correctness
    let verification = grid.verify_ghz_fidelity();
    assert!(
        verification.fidelity > 0.999999,
        "CPU fidelity: {}",
        verification.fidelity
    );

    (avg, min)
}

#[cfg(feature = "cuda")]
fn benchmark_gpu(rt: &CudaRuntime, n_qubits: usize, iterations: usize) -> (f64, f64) {
    let mut grid = QuantumGridF64Gpu::new(rt, n_qubits).expect("Failed to create GPU grid");

    // Warmup
    let _ = grid.create_ghz_state();

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let time_us = grid.create_ghz_state().expect("GHZ failed");
        times.push(time_us as f64 / 1000.0); // Convert to ms
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);

    // Verify correctness
    let verification = grid.verify_ghz_fidelity().expect("Verify failed");
    assert!(
        verification.fidelity > 0.999999,
        "GPU fidelity: {}",
        verification.fidelity
    );

    (avg, min)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           CPU vs GPU F64 Quantum Simulation Benchmark            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This benchmark requires the 'cuda' feature.");
        println!("Run with: cargo run --release --features cuda --example cpu_vs_gpu_f64_bench");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        // Initialize CUDA
        let rt = match CudaRuntime::new() {
            Ok(rt) => {
                if let Ok(name) = rt.device_name() {
                    println!("GPU: {}", name);
                }
                if let Ok((major, minor)) = rt.get_compute_capability() {
                    println!("Compute Capability: {}.{}", major, minor);
                }
                rt
            }
            Err(e) => {
                println!("CUDA initialization failed: {:?}", e);
                println!("Running CPU-only benchmark...\n");

                // CPU-only fallback
                println!(
                    "{:>8} | {:>12} | {:>12}",
                    "Qubits", "CPU Avg (ms)", "Memory"
                );
                println!("{}", "-".repeat(40));

                for n_qubits in [15, 18, 20, 22, 24, 25, 26, 27] {
                    let memory_mb = (1usize << n_qubits) * 16 / 1024 / 1024;
                    if memory_mb > 8192 {
                        println!(
                            "{:>8} | {:>12} | {:>9} MB (skipped - too large)",
                            n_qubits, "-", memory_mb
                        );
                        continue;
                    }

                    let (cpu_avg, _cpu_min) = benchmark_cpu(n_qubits, 5);
                    println!("{:>8} | {:>12.2} | {:>9} MB", n_qubits, cpu_avg, memory_mb);
                }
                return;
            }
        };

        println!();
        println!(
            "{:>8} | {:>12} | {:>12} | {:>10} | {:>12}",
            "Qubits", "CPU (ms)", "GPU (ms)", "Speedup", "Memory"
        );
        println!("{}", "-".repeat(70));

        let qubit_counts = [15, 18, 20, 22, 24, 25, 26, 27];

        for n_qubits in qubit_counts {
            let memory_mb = (1usize << n_qubits) * 16 / 1024 / 1024;

            // Skip if too large for available memory
            if memory_mb > 16384 {
                println!(
                    "{:>8} | {:>12} | {:>12} | {:>10} | {:>9} MB (skipped)",
                    n_qubits, "-", "-", "-", memory_mb
                );
                continue;
            }

            // Benchmark CPU
            let start = Instant::now();
            let (cpu_avg, _cpu_min) = benchmark_cpu(n_qubits, 5);
            let cpu_total = start.elapsed();

            // Benchmark GPU
            let gpu_result = benchmark_gpu(&rt, n_qubits, 5);
            let (gpu_avg, _gpu_min) = gpu_result;

            let speedup = cpu_avg / gpu_avg;

            println!(
                "{:>8} | {:>12.2} | {:>12.2} | {:>9.1}x | {:>9} MB",
                n_qubits, cpu_avg, gpu_avg, speedup, memory_mb
            );
        }

        println!("{}", "-".repeat(70));
        println!("\nNotes:");
        println!("  - Times are for GHZ state creation (H + n-1 CNOTs)");
        println!("  - CPU uses rayon parallelization");
        println!("  - GPU uses CUDA f64 kernels");
        println!("  - Memory = 2^n * 16 bytes (real + imag, f64)");

        // Additional detailed benchmark at 25 qubits
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║                  Detailed 25-Qubit Benchmark                     ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        let n = 25;
        let iters = 20;

        println!(
            "Running {} iterations at {} qubits ({} MB)...\n",
            iters,
            n,
            (1usize << n) * 16 / 1024 / 1024
        );

        // CPU detailed
        let mut cpu_times = Vec::with_capacity(iters);
        let mut cpu_grid = QuantumGridF64::new(n);
        for _ in 0..iters {
            let t = cpu_grid.create_ghz_state();
            cpu_times.push(t as f64 / 1000.0);
        }

        // GPU detailed
        let mut gpu_times = Vec::with_capacity(iters);
        let mut gpu_grid = QuantumGridF64Gpu::new(&rt, n).expect("GPU grid failed");
        for _ in 0..iters {
            let t = gpu_grid.create_ghz_state().expect("GHZ failed");
            gpu_times.push(t as f64 / 1000.0);
        }

        // Statistics
        let cpu_avg: f64 = cpu_times.iter().sum::<f64>() / iters as f64;
        let cpu_min = cpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let cpu_max = cpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let gpu_avg: f64 = gpu_times.iter().sum::<f64>() / iters as f64;
        let gpu_min = gpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let gpu_max = gpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("CPU (rayon):");
        println!("  Average: {:.2} ms", cpu_avg);
        println!("  Min:     {:.2} ms", cpu_min);
        println!("  Max:     {:.2} ms", cpu_max);

        println!("\nGPU (CUDA):");
        println!("  Average: {:.2} ms", gpu_avg);
        println!("  Min:     {:.2} ms", gpu_min);
        println!("  Max:     {:.2} ms", gpu_max);

        println!(
            "\nSpeedup: {:.1}x (avg), {:.1}x (best)",
            cpu_avg / gpu_avg,
            cpu_min / gpu_min
        );

        // Throughput calculation
        let amps = 1u64 << n;
        let ops_per_ghz = amps + (n as u64 - 1) * amps; // H + (n-1) CNOTs

        let cpu_throughput = ops_per_ghz as f64 / (cpu_avg / 1000.0) / 1e9;
        let gpu_throughput = ops_per_ghz as f64 / (gpu_avg / 1000.0) / 1e9;

        println!("\nThroughput:");
        println!("  CPU: {:.2} Gops/sec", cpu_throughput);
        println!("  GPU: {:.2} Gops/sec", gpu_throughput);

        // Verify both produce correct results
        let cpu_fidelity = cpu_grid.verify_ghz_fidelity().fidelity;
        let gpu_fidelity = gpu_grid
            .verify_ghz_fidelity()
            .expect("verify failed")
            .fidelity;

        println!("\nFidelity verification:");
        println!("  CPU: {:.15}", cpu_fidelity);
        println!("  GPU: {:.15}", gpu_fidelity);
        println!("  Match: {}", (cpu_fidelity - gpu_fidelity).abs() < 1e-14);
    }
}
