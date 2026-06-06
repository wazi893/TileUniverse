//! EPIC 94: GPU Neural Network Benchmark
//!
//! Measures GPU vs CPU speedup for batched neural network inference

#[cfg(feature = "cuda")]
use engine::brain::{BrainWeights, forward_batch_shared_weights};
#[cfg(feature = "cuda")]
use engine::gpu_nn::gpu_nn::GpuNNAccelerator;
#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          EPIC 94: GPU NN Accelerator Benchmark              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This benchmark requires CUDA support!");
        println!(
            "Build with: cargo run --example epic94_gpu_nn_benchmark --features cuda --release"
        );
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_benchmark();
    }
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
    // Test sizes: 1k → 1M organisms
    let batch_sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    println!("┌──────────┬────────────────────────────────────────────────────┐");
    println!("│   Size   │   CPU Time  │  GPU Time  │  Speedup  │  COPS/sec │");
    println!("├──────────┼────────────────────────────────────────────────────┤");

    for &n in &batch_sizes {
        // Generate test data
        let mut sensors = Vec::with_capacity(n);
        for i in 0..n {
            let val = (i as f32 / n as f32) * 2.0 - 1.0;
            sensors.push([val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        let weights = BrainWeights::random(12345);

        // CPU Baseline
        let cpu_start = Instant::now();
        let cpu_moves = forward_batch_shared_weights(&sensors, &weights);
        let cpu_time = cpu_start.elapsed();

        // GPU Accelerated
        let gpu_result = std::panic::catch_unwind(|| {
            let mut gpu_acc =
                GpuNNAccelerator::new(1_100_000).expect("Failed to create GPU accelerator");

            gpu_acc
                .upload_weights(&weights)
                .expect("Failed to upload weights");

            // Warmup
            let _ = gpu_acc.forward_batch(&sensors[..100.min(n)]);

            // Timed run
            let gpu_start = Instant::now();
            let gpu_moves = gpu_acc.forward_batch(&sensors).expect("GPU forward failed");
            let gpu_time = gpu_start.elapsed();

            // Verify correctness
            assert_eq!(cpu_moves.len(), gpu_moves.len());
            let mismatches: usize = cpu_moves
                .iter()
                .zip(gpu_moves.iter())
                .filter(|(cpu, gpu)| cpu != gpu)
                .count();

            if mismatches > 0 {
                println!("WARNING: {} / {} mismatches!", mismatches, n);
            }

            (gpu_time, gpu_moves)
        });

        match gpu_result {
            Ok((gpu_time, _gpu_moves)) => {
                let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

                // Compute operations:
                // - Layer 1: 8 neurons × (8 MACs + 1 bias + 1 tanh) = 80 ops
                // - Layer 2: 5 neurons × (8 MACs + 1 bias + 1 tanh) = 50 ops
                // - Argmax: 4 comparisons = 4 ops
                // Total: ~134 ops per organism
                let ops_per_organism = 134.0;
                let total_ops = n as f64 * ops_per_organism;
                let cops_per_sec = total_ops / gpu_time.as_secs_f64();

                println!(
                    "│ {:>8} │ {:>10.2}ms │ {:>9.2}ms │  {:>6.1}× │ {:>10.2}M │",
                    format_number(n),
                    cpu_time.as_secs_f64() * 1000.0,
                    gpu_time.as_secs_f64() * 1000.0,
                    speedup,
                    cops_per_sec / 1e6
                );
            }
            Err(e) => {
                println!(
                    "│ {:>8} │ {:>10.2}ms │   GPU FAIL │    N/A    │     N/A    │",
                    format_number(n),
                    cpu_time.as_secs_f64() * 1000.0
                );
                println!("    Error: {:?}", e);
            }
        }
    }

    println!("└──────────┴────────────────────────────────────────────────────┘\n");

    println!("Performance Analysis:");
    println!("  • Target: 100-1000× speedup over CPU");
    println!("  • Bottlenecks: H2D/D2H transfer overhead at small batch sizes");
    println!("  • Optimal: 100k+ organisms (amortizes PCIe latency)");
    println!("\n✓ EPIC 94 GPU NN Accelerator Complete!");
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}
