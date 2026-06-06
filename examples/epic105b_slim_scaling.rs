//! EPIC 105b: Slim Simulation Scaling Benchmark
//!
//! Tests how far we can push with memory-optimized SlimSimulation.
//! Target: 4x larger grids than full Simulation in same RAM.
//!
//! Run with: cargo run --release --example epic105b_slim_scaling

use engine::slim_simulation::SlimSimulation;
use engine::tile_meta::TileType;
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 105b: Slim Simulation - Maximum Scale Test              ║");
    println!("║  Memory-optimized: ~45 bytes/tile vs ~197 bytes/tile          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Test progressively larger grids until we hit the wall
    let configs = [
        (512, 512, 100),   // 262K tiles - baseline
        (1024, 1024, 50),  // 1M tiles
        (2048, 2048, 20),  // 4M tiles
        (4096, 4096, 10),  // 16M tiles
        (8192, 8192, 5),   // 67M tiles
        (16384, 16384, 2), // 268M tiles
        (24576, 24576, 1), // 604M tiles - target for 11.5GB
        (32768, 32768, 1), // 1B tiles - probably OOM
    ];

    let mut results = Vec::new();

    for (width, height, iterations) in configs {
        let tile_count = width * height;
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "Testing {}x{} grid ({:.2}M tiles, {} iterations)",
            width,
            height,
            tile_count as f64 / 1_000_000.0,
            iterations
        );
        println!("═══════════════════════════════════════════════════════════════");

        // Try to allocate - this might fail for large grids
        let alloc_start = Instant::now();
        let alloc_result = std::panic::catch_unwind(|| SlimSimulation::with_size(width, height));

        let mut sim = match alloc_result {
            Ok(s) => s,
            Err(_) => {
                println!("  ✗ ALLOCATION FAILED - hit memory wall");
                break;
            }
        };

        let alloc_time = alloc_start.elapsed();
        println!("  Allocation time: {:.3}s", alloc_time.as_secs_f64());

        // Get actual memory usage
        let mem_mb = sim.memory_usage_mb();
        let mem_gb = mem_mb / 1024.0;
        let bytes_per_tile = sim.memory_usage_bytes() as f64 / tile_count as f64;
        println!("  Actual memory: {:.1} MB ({:.2} GB)", mem_mb, mem_gb);
        println!("  Bytes per tile: {:.1}", bytes_per_tile);

        // Verify dimensions
        assert_eq!(sim.width(), width);
        assert_eq!(sim.height(), height);

        // Setup workload
        let setup_start = Instant::now();
        setup_benchmark_world(&mut sim, width, height);
        let setup_time = setup_start.elapsed();
        println!("  Setup time: {:.3}s", setup_time.as_secs_f64());

        // Warmup
        println!("  Warming up...");
        for _ in 0..3 {
            sim.eval_full_grid();
        }

        // Benchmark
        println!("  Running {} iterations...", iterations);
        let bench_start = Instant::now();
        let mut total_evals: u64 = 0;
        for _ in 0..iterations {
            let (evals, _) = sim.eval_full_grid();
            total_evals += evals as u64;
        }
        let bench_elapsed = bench_start.elapsed().as_secs_f64();

        let evals_per_sec = total_evals as f64 / bench_elapsed;
        let mevals_per_sec = evals_per_sec / 1_000_000.0;

        println!("\n  Results:");
        println!("    Total evaluations: {}", total_evals);
        println!("    Elapsed time: {:.3}s", bench_elapsed);
        println!(
            "    Throughput: {:.2}M evals/sec ({:.3}B evals/sec)",
            mevals_per_sec,
            evals_per_sec / 1e9
        );
        println!(
            "    Time per iteration: {:.3}ms",
            (bench_elapsed / iterations as f64) * 1000.0
        );

        results.push((
            width,
            height,
            tile_count,
            mem_mb,
            iterations,
            bench_elapsed,
            evals_per_sec,
        ));
    }

    // Summary
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:>12} {:>12} {:>10} {:>10} {:>12}",
        "Grid", "Tiles", "Mem(GB)", "Iters", "B evals/sec"
    );
    println!(
        "{:->12} {:->12} {:->10} {:->10} {:->12}",
        "", "", "", "", ""
    );

    let baseline_throughput = results.get(0).map(|r| r.6).unwrap_or(1.0);
    for (width, height, tiles, mem_mb, iters, _elapsed, throughput) in &results {
        let ratio = *throughput / baseline_throughput;
        println!(
            "{:>5}x{:<6} {:>12} {:>10.2} {:>10} {:>10.3}  ({:.1}x)",
            width,
            height,
            tiles,
            mem_mb / 1024.0,
            iters,
            throughput / 1e9,
            ratio
        );
    }

    // Find max achieved
    if let Some((w, h, tiles, mem, _, _, _)) = results.last() {
        println!(
            "\n✓ Maximum achieved: {}x{} ({:.1}M tiles) using {:.2} GB",
            w,
            h,
            *tiles as f64 / 1e6,
            *mem / 1024.0
        );
    }

    // Compare with full simulation
    println!("\n=== Memory Comparison ===");
    println!("Full Simulation:  ~197 bytes/tile");
    println!("Slim Simulation:  ~45 bytes/tile");
    println!("Improvement:      ~4.4x more tiles in same RAM");
}

/// Set up a representative benchmark workload
fn setup_benchmark_world(sim: &mut SlimSimulation, width: usize, height: usize) {
    // Place gates at regular intervals
    let spacing = 16.min(width / 8).max(1);
    for y in (8..height.saturating_sub(8)).step_by(spacing) {
        for x in (8..width.saturating_sub(8)).step_by(spacing) {
            let gate_type = match (x + y) % 4 {
                0 => TileType::And,
                1 => TileType::Or,
                2 => TileType::Xor,
                _ => TileType::Add,
            };
            sim.set_tile(x, y, gate_type);
        }
    }

    // Set some initial logic values
    let value_spacing = 8.min(width / 16).max(1);
    for y in (4..height.saturating_sub(4)).step_by(value_spacing) {
        for x in (4..width.saturating_sub(4)).step_by(value_spacing) {
            sim.set_logic_value(x, y, 1);
        }
    }
}
