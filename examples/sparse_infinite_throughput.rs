//! EPIC 122: Sparse Evaluation - Approaching Infinite Throughput
//!
//! This example demonstrates the power of sparse/delta evaluation.
//! Instead of evaluating ALL tiles every tick, we only evaluate
//! tiles that might have changed.
//!
//! For circuits that are mostly stable, this provides massive speedups:
//! - 1% active tiles = 100× speedup
//! - 0.1% active tiles = 1000× speedup
//! - 0.01% active tiles = 10000× speedup
//!
//! Run with:
//!   cargo run --release --example sparse_infinite_throughput --features cuda,perf-bench

use engine::cuda::{CudaRuntime, is_cuda_available};
use engine::cuda_tiles::benchmark_sparse_eval;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 122: SPARSE EVALUATION - INFINITE THROUGHPUT DEMO       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    if !is_cuda_available() {
        eprintln!("CUDA is not available. This example requires a CUDA-capable GPU.");
        return;
    }

    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create CUDA runtime: {:?}", e);
            return;
        }
    };

    println!("GPU: CUDA runtime initialized");
    println!();

    // Test parameters
    let width = 1024;
    let height = 1024;
    let tile_count = width * height;
    let num_ticks = 100;
    let max_delta = 50;

    println!("Grid: {}×{} = {} tiles", width, height, tile_count);
    println!("Ticks: {}", num_ticks);
    println!("Max delta cycles per tick: {}", max_delta);
    println!();

    // Test with different activity levels
    let activity_levels = [0.5, 0.1, 0.01, 0.001, 0.0001];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Activity%    Sparse Evals    Dense Equiv    Speedup");
    println!("═══════════════════════════════════════════════════════════════");

    for &activity in &activity_levels {
        match benchmark_sparse_eval(&rt, width, height, num_ticks, max_delta, activity, 42) {
            Ok(result) => {
                println!(
                    "  {:>7.3}%    {:>12}    {:>12}    {:>8.1}×",
                    activity * 100.0,
                    result.total_sparse_evals,
                    result.total_dense_evals,
                    result.speedup_vs_dense
                );
            }
            Err(e) => {
                eprintln!("  {:>7.3}%    ERROR: {:?}", activity * 100.0, e);
            }
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Detailed run for the lowest activity level
    println!("Detailed results for 0.01% activity:");
    println!();

    match benchmark_sparse_eval(&rt, width, height, num_ticks, max_delta, 0.0001, 12345) {
        Ok(result) => {
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }

    // Demonstrate with larger grid at very low activity
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SCALING TEST: 4096×4096 grid (16M tiles) at 0.001% activity");
    println!("═══════════════════════════════════════════════════════════════");

    let big_width = 4096;
    let big_height = 4096;

    match benchmark_sparse_eval(&rt, big_width, big_height, 50, max_delta, 0.00001, 99) {
        Ok(result) => {
            println!();
            println!("{}", result);

            if result.speedup_vs_dense > 10000.0 {
                println!();
                println!("╔═══════════════════════════════════════════════════════════════╗");
                println!("║  >>> INFINITE THROUGHPUT ACHIEVED! <<<                        ║");
                println!("║                                                               ║");
                println!(
                    "║  Effective throughput: {:.2e} tiles/sec              ║",
                    result.effective_throughput
                );
                println!(
                    "║  This would take {:.1}× longer with dense evaluation        ║",
                    result.speedup_vs_dense
                );
                println!("╚═══════════════════════════════════════════════════════════════╝");
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
}
