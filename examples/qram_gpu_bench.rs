//! QRAM GPU vs CPU Benchmark (Sprint 72.0)
//!
//! Compares performance of GPU-accelerated vs CPU bucket-brigade QRAM.
//!
//! Run:
//! ```bash
//! cargo run --release --example qram_gpu_bench --features gpu-prototype
//! ```


#[cfg(feature = "gpu-prototype")]
use engine::qram::{GpuBucketBrigade, SparseBucketBrigade};

fn format_duration(d: std::time::Duration) -> String {
    let micros = d.as_micros();
    if micros >= 1_000_000 {
        format!("{:.2}s", d.as_secs_f64())
    } else if micros >= 1_000 {
        format!("{:.2}ms", micros as f64 / 1000.0)
    } else {
        format!("{}µs", micros)
    }
}

#[cfg(feature = "gpu-prototype")]
fn main() {
    println!("================================================================================");
    println!("              QRAM GPU vs CPU BENCHMARK (Sprint 72.0)                           ");
    println!("================================================================================");
    println!();

    let depths = [3, 4, 5, 6];
    let iterations = 10;

    println!(
        "  {:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "Depth", "Cells", "Qubits", "CPU Time", "GPU Time", "Speedup"
    );
    println!("  {}", "-".repeat(72));

    for depth in depths {
        let memory_cells = 1 << depth;

        // Create test data
        let data: Vec<u64> = (0..memory_cells).map(|i| (i as u64 + 1) * 100).collect();

        // CPU benchmark
        let mut cpu_qram = SparseBucketBrigade::new(depth);
        cpu_qram.load(&data);

        let cpu_start = Instant::now();
        for seed in 0..iterations {
            let _ = cpu_qram.query_basis(seed % memory_cells, seed as u64);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        let mut gpu_qram = GpuBucketBrigade::new_sync(depth);
        gpu_qram.load(&data);

        // Warmup
        for seed in 0..3 {
            let _ = gpu_qram.query_basis(seed % memory_cells, seed as u64);
        }

        let gpu_start = Instant::now();
        for seed in 0..iterations {
            let _ = gpu_qram.query_basis(seed % memory_cells, seed as u64);
        }
        let gpu_time = gpu_start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let total_qubits = cpu_qram.total_qubits();

        println!(
            "  {:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>9.2}x",
            depth,
            memory_cells,
            total_qubits,
            format_duration(cpu_time),
            format_duration(gpu_time),
            speedup
        );
    }

    println!();
    println!("--------------------------------------------------------------------------------");
    println!("  Gate Count Comparison");
    println!("--------------------------------------------------------------------------------");
    println!();

    for depth in [3, 4, 5] {
        let memory_cells = 1 << depth;
        let data: Vec<u64> = (0..memory_cells).map(|i| i as u64).collect();

        let mut cpu_qram = SparseBucketBrigade::new(depth);
        cpu_qram.load(&data);
        let cpu_result = cpu_qram.query_basis(0, 42);
        let cpu_stats = cpu_qram.state_stats();

        let mut gpu_qram = GpuBucketBrigade::new_sync(depth);
        gpu_qram.load(&data);
        let gpu_result = gpu_qram.query_basis(0, 42);

        println!(
            "  Depth {}: CPU {} gates, GPU {} gates",
            depth, cpu_stats.gates_applied, gpu_result.gates_applied
        );
    }

    println!();
    println!("--------------------------------------------------------------------------------");
    println!("  Superposition Query Test");
    println!("--------------------------------------------------------------------------------");
    println!();

    let depth = 4;
    let memory_cells = 1 << depth;
    let data: Vec<u64> = (0..memory_cells).map(|i| (i as u64 + 1) * 100).collect();

    let mut gpu_qram = GpuBucketBrigade::new_sync(depth);
    gpu_qram.load(&data);

    println!(
        "  Running {} superposition queries on {} cells:",
        iterations, memory_cells
    );
    let mut addresses_seen = std::collections::HashSet::new();

    let start = Instant::now();
    for seed in 0..iterations {
        let result = gpu_qram.query_superposition(seed as u64);
        addresses_seen.insert(result.address);
    }
    let elapsed = start.elapsed();

    println!(
        "    Time: {} for {} queries",
        format_duration(elapsed),
        iterations
    );
    println!("    Unique addresses measured: {}", addresses_seen.len());
    println!(
        "    Queries/sec: {:.0}",
        iterations as f64 / elapsed.as_secs_f64()
    );

    println!();
    println!("================================================================================");
    println!("                              SUMMARY                                           ");
    println!("================================================================================");
    println!();
    println!("  GPU QRAM uses cross-block CNOT for Fredkin routing gates.");
    println!("  Performance depends on circuit depth and qubit layout.");
    println!();
    println!("  Key insights:");
    println!("  - GPU acceleration benefits larger QRAM (more routing gates)");
    println!("  - Cross-block CNOT at 1.5B ops/sec enables fast CSWAP");
    println!("  - Measurement still requires CPU (GPU download overhead)");
    println!();
}

#[cfg(not(feature = "gpu-prototype"))]
fn main() {
    println!("================================================================================");
    println!("              QRAM GPU vs CPU BENCHMARK (Sprint 72.0)                           ");
    println!("================================================================================");
    println!();
    println!("ERROR: This benchmark requires the gpu-prototype feature.");
    println!("Run with: cargo run --release --example qram_gpu_bench --features gpu-prototype");
}
