//! EPIC 122: Register-Resident Packed Tile Benchmark
//!
//! Benchmarks the new register-resident kernels targeting 100+ trillion tiles/sec.
//!
//! Run: cargo run --example packed_register_benchmark --features cuda,perf-bench --release

use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        use engine::cuda::{CudaRuntime, is_cuda_available};
        use engine::cuda_tiles::{PackedKernelVariant, benchmark_packed_tiles};

        if !is_cuda_available() {
            eprintln!("CUDA not available. This benchmark requires a CUDA GPU.");
            return;
        }

        println!("\n");
        println!("╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║  EPIC 122: Register-Resident Ultra-High-Throughput Benchmark              ║");
        println!("║  Target: 100+ trillion tiles/second                                       ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

        let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

        // Configuration - try different params to hit 100T
        let args: Vec<String> = std::env::args().collect();
        let width = args
            .iter()
            .position(|a| a == "--width")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(16384);
        let height = args
            .iter()
            .position(|a| a == "--height")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(16384);
        let depth = args
            .iter()
            .position(|a| a == "--depth")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000); // Higher depth = more compute per memory access
        let steps = args
            .iter()
            .position(|a| a == "--steps")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(20000);
        let warmup = args
            .iter()
            .position(|a| a == "--warmup")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000);

        let tile_count = width * height;
        let words = ((width + 63) / 64) * height;

        println!("Configuration:");
        println!(
            "  Grid:          {}×{} = {} tiles ({:.1}M)",
            width,
            height,
            tile_count,
            tile_count as f64 / 1e6
        );
        println!(
            "  Packed words:  {} ({:.2} MB)",
            words,
            (words * 8) as f64 / 1e6
        );
        println!("  Depth/launch:  {} steps", depth);
        println!("  Total steps:   {}", steps);
        println!("  Warmup:        {}\n", warmup);

        // Variants to benchmark
        let variants = [
            ("Shuffle (baseline)", PackedKernelVariant::Shuffle),
            ("Register V1 (16 rows)", PackedKernelVariant::Register),
            (
                "Register V2 (8 rows + coop)",
                PackedKernelVariant::RegisterV2,
            ),
            (
                "Register V3 (32 rows, 100T target)",
                PackedKernelVariant::RegisterV3,
            ),
        ];

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│  BENCHMARK RESULTS                                                          │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");

        let mut results = Vec::new();

        for (name, variant) in variants {
            print!("  Benchmarking {}...", name);
            std::io::Write::flush(&mut std::io::stdout()).ok();

            match benchmark_packed_tiles(&rt, width, height, steps, depth, warmup, variant) {
                Ok((total, elapsed, tiles_per_sec, _memory_mb)) => {
                    let trillion = tiles_per_sec / 1e12;
                    results.push((name, trillion, elapsed));
                    println!(" {:.2}T tiles/sec ({:.3}s)", trillion, elapsed);
                }
                Err(e) => {
                    println!(" FAILED: {:?}", e);
                    results.push((name, 0.0, 0.0));
                }
            }
        }

        println!(
            "└─────────────────────────────────────────────────────────────────────────────┘\n"
        );

        // Summary
        if let Some((best_name, best_throughput, _)) = results
            .iter()
            .filter(|(_, t, _)| *t > 0.0)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            let baseline = results
                .iter()
                .find(|(n, _, _)| n.contains("Shuffle"))
                .map(|(_, t, _)| *t)
                .unwrap_or(1.0);

            println!(
                "═══════════════════════════════════════════════════════════════════════════════"
            );
            println!("  SUMMARY");
            println!(
                "═══════════════════════════════════════════════════════════════════════════════"
            );
            println!("  Best kernel:     {}", best_name);
            println!("  Peak throughput: {:.2}T tiles/sec", best_throughput);
            println!("  vs Shuffle:      {:.1}×", best_throughput / baseline);

            if *best_throughput >= 100.0 {
                println!("\n  ✓ TARGET ACHIEVED: 100T+ tiles/sec!");
            } else {
                println!(
                    "\n  Target: 100T tiles/sec (current: {:.1}%)",
                    best_throughput
                );
            }
            println!(
                "═══════════════════════════════════════════════════════════════════════════════\n"
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This benchmark requires --features cuda");
    }
}
