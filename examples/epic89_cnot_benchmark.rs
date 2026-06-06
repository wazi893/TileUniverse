//! EPIC 89: CNOT WMMA Batching Benchmark
//!
//! Benchmarks two-qubit gate execution via WMMA tensor cores compared to
//! sequential gate application.
//!
//! Run with: cargo run --example epic89_cnot_benchmark --features cuda --release

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, WmmaState, is_cuda_available, run_wmma_batched_gates};
#[cfg(feature = "cuda")]
use engine::quantum::QGate;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              EPIC 89: CNOT WMMA Batching Benchmark                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("❌ CUDA not available - skipping benchmark");
        return;
    }

    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

    // Benchmark configurations
    let depths = [10, 50, 100, 500, 1000];
    let warmup_iters = 5;
    let bench_iters = 20;

    println!("## Configuration");
    println!("- Warmup iterations: {}", warmup_iters);
    println!("- Benchmark iterations: {}", bench_iters);
    println!();

    // Test 1: H-CNOT alternating pattern (VQE-like)
    println!("## Test 1: H-CNOT Alternating Pattern (VQE-like)\n");
    println!("| Depth | Gates | Time (us) | Gates/us | GCOPS |");
    println!("|-------|-------|-----------|----------|-------|");

    for &depth in &depths {
        let gates: Vec<QGate> = (0..depth)
            .flat_map(|_| vec![QGate::H(0), QGate::CNot(0, 1)])
            .collect();

        let result = benchmark_wmma(&rt, &gates, warmup_iters, bench_iters);
        let gates_per_us = result.total_gates as f64 / result.avg_time_us;
        let gcops = gates_per_us * 1000.0; // Gates * 1e6 / time_us = GCOPS

        println!(
            "| {:>5} | {:>5} | {:>9.2} | {:>8.1} | {:>5.2} |",
            depth,
            result.total_gates,
            result.avg_time_us,
            gates_per_us,
            gcops / 1e9
        );
    }

    // Test 2: CNOT ladder pattern
    println!("\n## Test 2: CNOT Ladder Pattern\n");
    println!("| Depth | Gates | Time (us) | Gates/us | GCOPS |");
    println!("|-------|-------|-----------|----------|-------|");

    for &depth in &depths {
        let gates: Vec<QGate> = (0..depth)
            .flat_map(|_| vec![QGate::CNot(0, 1), QGate::CNot(1, 2), QGate::CNot(2, 3)])
            .collect();

        let result = benchmark_wmma(&rt, &gates, warmup_iters, bench_iters);
        let gates_per_us = result.total_gates as f64 / result.avg_time_us;
        let gcops = gates_per_us * 1000.0;

        println!(
            "| {:>5} | {:>5} | {:>9.2} | {:>8.1} | {:>5.2} |",
            depth,
            result.total_gates,
            result.avg_time_us,
            gates_per_us,
            gcops / 1e9
        );
    }

    // Test 3: Mixed single+two-qubit gates (H, X, Z, CNOT)
    println!("\n## Test 3: Mixed Single + Two-Qubit Gates (H, X, Z, CNOT)\n");
    println!("| Depth | Gates | Time (us) | Gates/us | GCOPS |");
    println!("|-------|-------|-----------|----------|-------|");

    for &depth in &depths {
        let gates: Vec<QGate> = (0..depth)
            .flat_map(|_| {
                // Mixed gate layer pattern using supported gates
                vec![
                    QGate::H(0),
                    QGate::X(1),
                    QGate::Z(2),
                    QGate::CNot(0, 1),
                    QGate::CNot(1, 2),
                    QGate::CNot(2, 3),
                ]
            })
            .collect();

        let result = benchmark_wmma(&rt, &gates, warmup_iters, bench_iters);
        let gates_per_us = result.total_gates as f64 / result.avg_time_us;
        let gcops = gates_per_us * 1000.0;

        println!(
            "| {:>5} | {:>5} | {:>9.2} | {:>8.1} | {:>5.2} |",
            depth,
            result.total_gates,
            result.avg_time_us,
            gates_per_us,
            gcops / 1e9
        );
    }

    // Test 4: CZ gates (QAOA-like)
    println!("\n## Test 4: CZ Gate Pattern (QAOA-like)\n");
    println!("| Depth | Gates | Time (us) | Gates/us | GCOPS |");
    println!("|-------|-------|-----------|----------|-------|");

    for &depth in &depths {
        let gates: Vec<QGate> = (0..depth)
            .flat_map(|i| {
                vec![
                    QGate::H(0),
                    QGate::H(1),
                    QGate::CZ(0, 1),
                    QGate::CZ(1, 2),
                    QGate::Phase(0, 0.1 * i as f32),
                ]
            })
            .collect();

        let result = benchmark_wmma(&rt, &gates, warmup_iters, bench_iters);
        let gates_per_us = result.total_gates as f64 / result.avg_time_us;
        let gcops = gates_per_us * 1000.0;

        println!(
            "| {:>5} | {:>5} | {:>9.2} | {:>8.1} | {:>5.2} |",
            depth,
            result.total_gates,
            result.avg_time_us,
            gates_per_us,
            gcops / 1e9
        );
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    EPIC 89 Benchmark Complete                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
struct BenchmarkResult {
    total_gates: usize,
    avg_time_us: f64,
    #[allow(dead_code)]
    min_time_us: f64,
    #[allow(dead_code)]
    max_time_us: f64,
}

#[cfg(feature = "cuda")]
fn benchmark_wmma(
    rt: &CudaRuntime,
    gates: &[QGate],
    warmup: usize,
    iters: usize,
) -> BenchmarkResult {
    // Initialize state
    let mut init_data = vec![half::f16::ZERO; 256];
    init_data[0] = half::f16::ONE;

    // Warmup
    for _ in 0..warmup {
        let mut state = WmmaState::from_host(rt, &init_data).expect("WmmaState creation");
        run_wmma_batched_gates(rt, &mut state, gates).expect("WMMA gates");
    }

    // Benchmark
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut state = WmmaState::from_host(rt, &init_data).expect("WmmaState creation");

        let start = Instant::now();
        run_wmma_batched_gates(rt, &mut state, gates).expect("WMMA gates");
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f64() * 1e6); // Convert to microseconds
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    BenchmarkResult {
        total_gates: gates.len(),
        avg_time_us: avg,
        min_time_us: min,
        max_time_us: max,
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: This example requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example epic89_cnot_benchmark --features cuda --release");
    std::process::exit(1);
}
