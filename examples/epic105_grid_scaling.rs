//! EPIC 105: Grid Scaling Benchmark
//!
//! Tests performance at different grid sizes to find scaling limits.
//! Measures tile evaluations per second at 512x512, 1024x1024, 2048x2048, 4096x4096.
//!
//! Run with: cargo run --release --example epic105_grid_scaling --features perf-bench

use engine::simulation::Simulation;
use engine::tile_meta::TileType;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 105: Grid Scaling Benchmark                             ║");
    println!("║  Testing tile evaluation performance at different grid sizes  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Test configurations: (width, height, iterations)
    // Let's find the actual wall!
    let configs = [
        (512, 512, 100),   // 262K tiles - baseline
        (1024, 1024, 50),  // 1M tiles
        (2048, 2048, 20),  // 4M tiles
        (4096, 4096, 10),  // 16M tiles
        (8192, 8192, 5),   // 67M tiles
        (16384, 16384, 2), // 268M tiles - quarter billion!
        (32768, 32768, 1), // 1B tiles - ONE BILLION
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

        // Create simulation with custom size
        let alloc_start = Instant::now();
        let mut sim = Simulation::with_size(width, height);
        let alloc_time = alloc_start.elapsed();
        println!("  Allocation time: {:.3}s", alloc_time.as_secs_f64());

        // Verify dimensions
        assert_eq!(sim.width(), width);
        assert_eq!(sim.height(), height);
        assert_eq!(sim.tile_count(), tile_count);

        // Create a representative workload
        let setup_start = Instant::now();
        setup_benchmark_world(&mut sim, width, height);
        let setup_time = setup_start.elapsed();
        println!("  Setup time: {:.3}s", setup_time.as_secs_f64());

        // Memory estimate (rough)
        let mem_estimate_mb = estimate_memory_mb(width, height);
        println!("  Estimated memory: {:.1} MB", mem_estimate_mb);

        // Warmup - use eval_logic_full_grid_bench which doesn't have delta cycle limits
        println!("  Warming up...");
        #[cfg(feature = "perf-bench")]
        for _ in 0..5 {
            sim.eval_logic_full_grid_bench();
        }
        #[cfg(not(feature = "perf-bench"))]
        {
            println!("    (warmup skipped - requires --features perf-bench)");
        }

        // Benchmark - full grid evaluation
        println!("  Running {} iterations...", iterations);
        let bench_start = Instant::now();
        let mut total_evals_actual: u64 = 0;
        #[cfg(feature = "perf-bench")]
        for _ in 0..iterations {
            let (evals, _changes) = sim.eval_logic_full_grid_bench();
            total_evals_actual += evals as u64;
        }
        #[cfg(not(feature = "perf-bench"))]
        {
            println!("    ERROR: This benchmark requires --features perf-bench");
            continue;
        }
        let bench_elapsed = bench_start.elapsed().as_secs_f64();

        let total_evals = (tile_count as u64) * (iterations as u64);
        let evals_per_sec = total_evals as f64 / bench_elapsed;
        let mevals_per_sec = evals_per_sec / 1_000_000.0;
        let gevals_per_sec = evals_per_sec / 1_000_000_000.0;

        println!("\n  Results:");
        println!(
            "    Total evaluations: {} (actual: {})",
            total_evals, total_evals_actual
        );
        println!("    Elapsed time: {:.3}s", bench_elapsed);
        println!(
            "    Throughput: {:.2}M evals/sec ({:.3}B evals/sec)",
            mevals_per_sec, gevals_per_sec
        );
        println!(
            "    Time per iteration: {:.3}ms",
            (bench_elapsed / iterations as f64) * 1000.0
        );

        results.push((
            width,
            height,
            tile_count,
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
        "{:>10} {:>12} {:>10} {:>12} {:>14}",
        "Grid", "Tiles", "Iters", "Time(s)", "B evals/sec"
    );
    println!(
        "{:->10} {:->12} {:->10} {:->12} {:->14}",
        "", "", "", "", ""
    );

    let baseline_throughput = results[0].5;
    for (width, height, tiles, iters, elapsed, throughput) in &results {
        let ratio = *throughput / baseline_throughput;
        println!(
            "{:>4}x{:<5} {:>12} {:>10} {:>12.3} {:>12.3}  ({:.1}x)",
            width,
            height,
            tiles,
            iters,
            elapsed,
            throughput / 1e9,
            ratio
        );
    }

    // Scaling analysis
    println!("\n\nScaling Analysis:");
    if results.len() >= 2 {
        let (_, _, t1, _, _, tp1) = results[0];
        let (_, _, t2, _, _, tp2) = results[results.len() - 1];
        let tile_ratio = t2 as f64 / t1 as f64;
        let throughput_ratio = tp2 / tp1;
        println!("  Grid size increased: {:.0}x", tile_ratio);
        println!("  Throughput change: {:.2}x", throughput_ratio);
        if throughput_ratio >= 0.9 {
            println!("  ✓ Excellent scaling - throughput maintained!");
        } else if throughput_ratio >= 0.7 {
            println!("  ◐ Good scaling - some throughput loss");
        } else if throughput_ratio >= 0.5 {
            println!("  ◑ Moderate scaling - noticeable throughput loss");
        } else {
            println!("  ✗ Poor scaling - significant throughput degradation");
        }
    }

    println!("\n✓ Grid scaling benchmark complete\n");
}

/// Set up a representative benchmark workload with gates and wires
fn setup_benchmark_world(sim: &mut Simulation, width: usize, height: usize) {
    // Place clock tiles at regular intervals
    let clock_spacing = 64.min(width / 4);
    for y in (32..height.saturating_sub(32)).step_by(clock_spacing.max(1)) {
        for x in (32..width.saturating_sub(32)).step_by(clock_spacing.max(1)) {
            sim.set_tile(x, y, TileType::ClockGlobal);
        }
    }

    // Place gates at regular intervals
    let gate_spacing = 16.min(width / 8);
    for y in (8..height.saturating_sub(8)).step_by(gate_spacing.max(1)) {
        for x in (8..width.saturating_sub(8)).step_by(gate_spacing.max(1)) {
            // Alternate gate types
            let gate_type = match (x + y) % 4 {
                0 => TileType::And,
                1 => TileType::Or,
                2 => TileType::Xor,
                _ => TileType::Not,
            };
            sim.set_tile(x, y, gate_type);
        }
    }

    // Set some initial logic values to create activity
    let value_spacing = 8.min(width / 16);
    for y in (4..height.saturating_sub(4)).step_by(value_spacing.max(1)) {
        for x in (4..width.saturating_sub(4)).step_by(value_spacing.max(1)) {
            sim.tilemap.set_value_at(x, y, 1);
        }
    }
}

/// Rough memory estimate in MB
fn estimate_memory_mb(width: usize, height: usize) -> f64 {
    let tile_count = width * height;

    // Per tile: 8 bytes logic (AtomicU64) + ~1 byte meta + padding
    let tile_bytes = tile_count * 16;

    // Neighbor table: 4 x u32 per tile
    let neighbor_bytes = tile_count * 16;

    // Fields: multiple FieldGrid instances (u8, u32, u64)
    // Roughly: 10+ fields * average 4 bytes per cell
    let field_bytes = tile_count * 40;

    // Dirty bitset: ~1 bit per tile
    let dirty_bytes = tile_count / 8;

    // last_change: Option<ChangeInfo> per tile
    let change_bytes = tile_count * 32;

    (tile_bytes + neighbor_bytes + field_bytes + dirty_bytes + change_bytes) as f64
        / (1024.0 * 1024.0)
}
