//! GPU Sparse Evaluation Benchmark
//!
//! Demonstrates the GPU sparse evaluation system which only evaluates
//! tiles that have changed, providing massive speedups for circuits
//! where most tiles are stable.
//!
//! Run with: cargo run --example gpu_sparse_benchmark --release --features cuda
//!
//! The sparse evaluation system:
//! 1. Tracks which tiles are "dirty" (need re-evaluation) via a bitset
//! 2. Builds a compact work list of dirty tile indices on GPU
//! 3. Only evaluates tiles in the work list
//! 4. Propagates dirty flags to neighbors when outputs change
//!
//! For circuits with low activity (e.g., 0.1% of tiles changing per tick),
//! this provides ~1000x speedup over dense evaluation.

#[cfg(feature = "cuda")]
use engine::cuda_tiles::{SparseBenchmarkResult, benchmark_gpu_sparse_eval, benchmark_sparse_eval};
#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::CudaRuntime;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  GPU SPARSE EVALUATION BENCHMARK                              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This example requires the 'cuda' feature.");
        println!("Run with: cargo run --example gpu_sparse_benchmark --release --features cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if let Err(e) = run_gpu_benchmark() {
            eprintln!("GPU benchmark failed: {}", e);
            println!();
            println!("Falling back to CPU sparse benchmark...");
            run_cpu_sparse_benchmark();
        }
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let rt = CudaRuntime::new()?;

    println!(
        "GPU Device: {}",
        rt.device_name().unwrap_or("Unknown".into())
    );
    println!();

    // Test configurations: (width, height, ticks, activity_ratio)
    let configs = [
        (512, 512, 100, 0.001),  // 0.1% activity - best case for sparse
        (512, 512, 100, 0.01),   // 1% activity
        (512, 512, 100, 0.05),   // 5% activity
        (1024, 1024, 50, 0.001), // Large grid, low activity
        (1024, 1024, 50, 0.01),  // Large grid, moderate activity
    ];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  SPARSE VS DENSE COMPARISON");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:>8} {:>8} {:>6} {:>8} {:>12} {:>12} {:>10}",
        "Width", "Height", "Ticks", "Activity", "Sparse Eval", "Dense Equiv", "Speedup"
    );
    println!(
        "{:>8} {:>8} {:>6} {:>8} {:>12} {:>12} {:>10}",
        "--------", "--------", "------", "--------", "------------", "------------", "----------"
    );

    let mut results: Vec<SparseBenchmarkResult> = Vec::new();

    for (width, height, ticks, activity) in configs {
        match benchmark_gpu_sparse_eval(&rt, width, height, ticks, activity, 42) {
            Ok(result) => {
                println!(
                    "{:>8} {:>8} {:>6} {:>7.1}% {:>12} {:>12} {:>9.1}x",
                    result.width,
                    result.height,
                    result.num_ticks,
                    result.activity_ratio * 100.0,
                    format_count(result.total_sparse_evals),
                    format_count(result.total_dense_evals),
                    result.speedup_vs_dense
                );
                results.push(result);
            }
            Err(e) => {
                println!(
                    "{:>8} {:>8} {:>6} {:>7.1}% FAILED: {}",
                    width,
                    height,
                    ticks,
                    activity * 100.0,
                    e
                );
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  THROUGHPUT ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    for result in &results {
        println!(
            "Grid: {}x{} ({} tiles)",
            result.width,
            result.height,
            result.width * result.height
        );
        println!(
            "  Activity:           {:.2}%",
            result.activity_ratio * 100.0
        );
        println!(
            "  Sparse throughput:  {:.2e} evals/sec",
            result.sparse_throughput
        );
        println!(
            "  Effective throughput: {:.2e} tiles/sec (counting skipped as evaluated)",
            result.effective_throughput
        );
        println!(
            "  Time for {} ticks: {:.3} ms",
            result.num_ticks,
            result.elapsed_secs * 1000.0
        );
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  KEY INSIGHTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  • Sparse evaluation only processes tiles that might have changed");
    println!("  • For 0.1% activity, we evaluate 1000x fewer tiles");
    println!("  • Effective throughput = what dense would need to match our speed");
    println!("  • Best for: stable circuits, localized changes, long simulations");
    println!("  • Break-even: ~50% activity (below this, sparse wins)");
    println!();

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cpu_sparse_benchmark() {
    use logic_fabric_core::cuda::CudaRuntime;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  CPU SPARSE BENCHMARK (CUDA unavailable)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Use the CPU-based sparse benchmark as fallback
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(_) => {
            println!("Could not initialize CUDA runtime");
            return;
        }
    };

    let configs = [
        (256, 256, 50, 0.001),
        (256, 256, 50, 0.01),
        (512, 512, 20, 0.001),
    ];

    for (width, height, ticks, activity) in configs {
        match benchmark_sparse_eval(&rt, width, height, ticks, 100, activity, 42) {
            Ok(result) => {
                println!("{}", result);
            }
            Err(e) => {
                println!("Benchmark {}x{} failed: {}", width, height, e);
            }
        }
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
