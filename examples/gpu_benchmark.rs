// =============================================================================
// Sprint 62.0: GPU Benchmark Demo
// =============================================================================
//
// Compares CPU vs GPU quantum simulation performance using the hybrid
// orchestration layer.
//
// This example demonstrates:
// - GpuSubstrate for GPU-accelerated quantum simulation
// - AutoRouter for intelligent substrate selection
// - Performance comparison across different circuit sizes
//
// Run with:
//   cargo run --example gpu_benchmark --release --features cuda
//
// Without CUDA (CPU only comparison):
//   cargo run --example gpu_benchmark --release
//
// =============================================================================

use std::time::Instant;

use engine::hybrid::{
    AutoRouter, HybridScheduler, JobSpec, MeasurementSpec, QuantumCircuit, QuantumSpec,
    QuantumSubstrate, SchedulerConfig, StageOutputs, SubstrateRouteInfo, SubstrateType,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 62.0: GPU BENCHMARK DEMO");
    println!("        CPU vs GPU Quantum Simulation");
    println!("{}\n", "=".repeat(80));

    // Check CUDA availability
    #[cfg(feature = "cuda")]
    let cuda_available = check_cuda_available();
    #[cfg(not(feature = "cuda"))]
    let cuda_available = false;

    if cuda_available {
        println!("CUDA: Available");
    } else {
        println!("CUDA: Not available (running CPU-only benchmarks)");
    }

    // Demo 1: CPU substrate performance
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 1: CPU Quantum Substrate");
    println!("{}\n", "-".repeat(80));
    benchmark_cpu_substrate();

    // Demo 2: GPU substrate performance (if CUDA available)
    #[cfg(feature = "cuda")]
    if cuda_available {
        println!("\n{}", "-".repeat(80));
        println!("BENCHMARK 2: GPU Quantum Substrate");
        println!("{}\n", "-".repeat(80));
        benchmark_gpu_substrate();
    }

    // Demo 3: Auto-router demonstration
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: Auto-Router Substrate Selection");
    println!("{}\n", "-".repeat(80));
    demo_auto_router();

    // Demo 4: Scaling comparison
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 4: Scaling Comparison");
    println!("{}\n", "-".repeat(80));
    benchmark_scaling(cuda_available);

    println!("\n{}", "=".repeat(80));
    println!("                    GPU BENCHMARK COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Benchmark CPU quantum substrate.
fn benchmark_cpu_substrate() {
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "CPU-Quantum", 20)));

    let qubit_counts = [4, 8, 12, 16];
    let shots = 1000;

    println!("Running quantum circuits on CPU substrate:\n");
    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "Qubits", "Gates", "Time (ms)", "Ops/sec"
    );
    println!("{}", "-".repeat(55));

    for &n_qubits in &qubit_counts {
        let circuit = create_benchmark_circuit(n_qubits, 10);
        let gate_count = circuit.gate_count();

        let spec = QuantumSpec::new(n_qubits, circuit)
            .with_shots(shots)
            .with_measurement(MeasurementSpec::Computational);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Quantum(spec));
        let _ = scheduler.wait(job_id);
        let elapsed = start.elapsed();

        let time_ms = elapsed.as_secs_f64() * 1000.0;
        let state_size = 1u64 << n_qubits;
        let ops = state_size * gate_count as u64;
        let ops_per_sec = ops as f64 / elapsed.as_secs_f64();

        println!(
            "{:>8} {:>12} {:>14.2} {:>14.2e}",
            n_qubits, gate_count, time_ms, ops_per_sec
        );
    }
}

/// Benchmark GPU quantum substrate.
#[cfg(feature = "cuda")]
fn benchmark_gpu_substrate() {
    use engine::hybrid::GpuSubstrate;

    let gpu_substrate = match GpuSubstrate::new(2, "GPU-Quantum") {
        Some(s) => s,
        None => {
            println!("Failed to create GPU substrate");
            return;
        }
    };

    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(gpu_substrate));

    let qubit_counts = [4, 8, 12, 16, 20];
    let shots = 1000;

    println!("Running quantum circuits on GPU substrate:\n");
    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "Qubits", "Gates", "Time (ms)", "Ops/sec"
    );
    println!("{}", "-".repeat(55));

    for &n_qubits in &qubit_counts {
        let circuit = create_benchmark_circuit(n_qubits, 10);
        let gate_count = circuit.gate_count();

        let spec = QuantumSpec::new(n_qubits, circuit)
            .with_shots(shots)
            .with_measurement(MeasurementSpec::Computational);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Quantum(spec));
        let _ = scheduler.wait(job_id);
        let elapsed = start.elapsed();

        let time_ms = elapsed.as_secs_f64() * 1000.0;
        let state_size = 1u64 << n_qubits;
        let ops = state_size * gate_count as u64;
        let ops_per_sec = ops as f64 / elapsed.as_secs_f64();

        println!(
            "{:>8} {:>12} {:>14.2} {:>14.2e}",
            n_qubits, gate_count, time_ms, ops_per_sec
        );
    }
}

/// Demonstrate auto-router functionality.
fn demo_auto_router() {
    let router = AutoRouter::new().with_gpu_threshold(16);

    // Simulate substrate info
    let cpu_info = SubstrateRouteInfo {
        id: engine::hybrid::SubstrateId(1),
        substrate_type: SubstrateType::Quantum,
        gpu_available: false,
        max_qubits: Some(20),
        capacity: 1.0,
    };

    let gpu_info = SubstrateRouteInfo {
        id: engine::hybrid::SubstrateId(2),
        substrate_type: SubstrateType::Quantum,
        gpu_available: true,
        max_qubits: Some(26),
        capacity: 1.0,
    };

    let substrates = vec![cpu_info, gpu_info];

    println!("Auto-router configuration:");
    println!("  GPU qubit threshold: 16");
    println!("\nRouting decisions:\n");

    for n_qubits in [8, 12, 16, 20, 24] {
        let spec = JobSpec::Quantum(QuantumSpec::new(n_qubits, QuantumCircuit::new(n_qubits)));
        let selection = router.select_substrate(&spec, &substrates);

        let substrate_name = match selection {
            Some(id) if id.0 == 1 => "CPU",
            Some(id) if id.0 == 2 => "GPU",
            _ => "None",
        };

        let should_gpu = router.should_use_gpu(n_qubits);
        println!(
            "  {:>2} qubits: {} (prefer GPU: {})",
            n_qubits, substrate_name, should_gpu
        );
    }
}

/// Benchmark scaling comparison.
fn benchmark_scaling(cuda_available: bool) {
    println!("Measuring circuit execution scaling:\n");

    // CPU benchmark
    let cpu_times = benchmark_substrate_scaling(false);
    println!("\nCPU Results:");
    print_scaling_results(&cpu_times);

    // GPU benchmark (if available)
    #[cfg(feature = "cuda")]
    if cuda_available {
        let gpu_times = benchmark_substrate_scaling(true);
        println!("\nGPU Results:");
        print_scaling_results(&gpu_times);

        // Speedup comparison
        println!("\nGPU Speedup:");
        for i in 0..cpu_times.len() {
            if gpu_times[i].1 > 0.0 {
                let speedup = cpu_times[i].1 / gpu_times[i].1;
                println!("  {:>2} qubits: {:.2}x", cpu_times[i].0, speedup);
            }
        }
    }
}

fn benchmark_substrate_scaling(_use_gpu: bool) -> Vec<(usize, f64)> {
    let mut results = Vec::new();
    let mut scheduler = HybridScheduler::new();

    #[cfg(feature = "cuda")]
    if _use_gpu {
        use engine::hybrid::GpuSubstrate;
        if let Some(gpu) = GpuSubstrate::new(1, "GPU") {
            scheduler.register_substrate(Box::new(gpu));
        } else {
            return results;
        }
    } else {
        scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "CPU", 20)));
    }

    #[cfg(not(feature = "cuda"))]
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "CPU", 20)));

    for n_qubits in [4, 6, 8, 10, 12, 14, 16] {
        let circuit = create_benchmark_circuit(n_qubits, 5);
        let spec = QuantumSpec::new(n_qubits, circuit)
            .with_shots(100)
            .with_measurement(MeasurementSpec::Computational);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Quantum(spec));
        let _ = scheduler.wait(job_id);
        let elapsed = start.elapsed();

        results.push((n_qubits, elapsed.as_secs_f64() * 1000.0));
    }

    results
}

fn print_scaling_results(results: &[(usize, f64)]) {
    println!("{:>8} {:>12}", "Qubits", "Time (ms)");
    println!("{}", "-".repeat(25));
    for (qubits, time_ms) in results {
        println!("{:>8} {:>11.2}", qubits, time_ms);
    }
}

/// Create a benchmark circuit with specified depth.
fn create_benchmark_circuit(n_qubits: usize, depth: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);

    for _ in 0..depth {
        // Layer of Hadamards
        for q in 0..n_qubits {
            circuit.h(q);
        }

        // Layer of CNOTs
        for q in 0..n_qubits.saturating_sub(1) {
            circuit.cnot(q, q + 1);
        }

        // Layer of rotations
        for q in 0..n_qubits {
            circuit.ry(q, std::f64::consts::PI / 4.0);
        }
    }

    circuit
}

#[cfg(feature = "cuda")]
fn check_cuda_available() -> bool {
    use engine::hybrid::GpuSubstrate;
    GpuSubstrate::new(0, "test").is_some()
}
