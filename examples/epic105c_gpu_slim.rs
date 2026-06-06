//! EPIC 105c: GPU Slim Simulation Benchmark
//!
//! Tests GPU-native slim simulation for maximum throughput.
//! Target: 14K x 14K grid at 40B+ evals/sec on RTX 4070M
//!
//! Run with: cargo run --release --example epic105c_gpu_slim --features cuda

#[cfg(all(feature = "cuda", feature = "gpu-prototype"))]
fn main() {
    use engine::cuda::CudaRuntime;
    use engine::gpu_slim::GpuSlimSimulation;
    use engine::tile_meta::TileType;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 105c: GPU Slim Simulation - Maximum Throughput          ║");
    println!("║  Target: 14K x 14K grid (196M tiles) at 40B+ evals/sec        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA
    println!("Initializing CUDA...");
    let rt = match CudaRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            println!("Failed to initialize CUDA: {:?}", e);
            return;
        }
    };
    println!("  CUDA initialized successfully\n");

    // Calculate max grid size for 8GB VRAM (leave 1GB headroom)
    let usable_vram_mb = 7 * 1024; // 7GB
    let (max_side, _) = GpuSlimSimulation::max_grid_size(usable_vram_mb * 1024 * 1024);
    println!(
        "Max grid size for 7GB VRAM: {}x{} ({:.1}M tiles)\n",
        max_side,
        max_side,
        (max_side * max_side) as f64 / 1e6
    );

    // Test configurations - adjusted for 8GB VRAM (leave headroom)
    // OOM at 14336x14336 (6.32GB allocation + temp buffers)
    let configs = [
        (512, 512, 1000),   // 262K tiles - warmup
        (1024, 1024, 500),  // 1M tiles
        (2048, 2048, 200),  // 4M tiles
        (4096, 4096, 100),  // 16M tiles
        (6144, 6144, 50),   // 38M tiles
        (8192, 8192, 30),   // 67M tiles
        (10240, 10240, 20), // 105M tiles
        (12288, 12288, 10), // 151M tiles - max stable
    ];

    let mut results = Vec::new();

    for (width, height, steps) in configs {
        let tile_count = width * height;
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "Testing {}x{} grid ({:.2}M tiles, {} steps)",
            width,
            height,
            tile_count as f64 / 1e6,
            steps
        );
        println!("═══════════════════════════════════════════════════════════════");

        // Create GPU simulation
        let alloc_start = std::time::Instant::now();
        let mut sim = match GpuSlimSimulation::with_size(&rt, width, height) {
            Ok(s) => s,
            Err(e) => {
                println!("  ✗ ALLOCATION FAILED: {:?}", e);
                println!("  Hit VRAM limit");
                break;
            }
        };
        let alloc_time = alloc_start.elapsed();
        println!("  Allocation time: {:.3}s", alloc_time.as_secs_f64());

        // Report VRAM usage
        let vram_mb = sim.vram_usage_mb();
        println!(
            "  VRAM usage: {:.1} MB ({:.2} GB)",
            vram_mb,
            vram_mb / 1024.0
        );

        // Setup workload (upload tile types)
        println!("  Setting up workload...");
        let mut types = vec![TileType::Wire as u8; tile_count];
        let spacing = 16.min(width / 8).max(1);
        for y in (8..height.saturating_sub(8)).step_by(spacing) {
            for x in (8..width.saturating_sub(8)).step_by(spacing) {
                let idx = y * width + x;
                types[idx] = match (x + y) % 4 {
                    0 => TileType::And as u8,
                    1 => TileType::Or as u8,
                    2 => TileType::Xor as u8,
                    _ => TileType::Add as u8,
                };
            }
        }
        if let Err(e) = sim.set_tile_types(&rt, &types) {
            println!("  Failed to upload tile types: {:?}", e);
            continue;
        }

        // Set some initial logic values
        let mut logic = vec![0u64; tile_count];
        let value_spacing = 8.min(width / 16).max(1);
        for y in (4..height.saturating_sub(4)).step_by(value_spacing) {
            for x in (4..width.saturating_sub(4)).step_by(value_spacing) {
                logic[y * width + x] = 1;
            }
        }
        if let Err(e) = sim.set_logic(&rt, &logic) {
            println!("  Failed to upload logic: {:?}", e);
            continue;
        }

        // Benchmark
        println!("  Running {} steps (with 10 warmup)...", steps);
        let result = sim.benchmark(&rt, steps, 10);

        match result {
            Ok((total_evals, elapsed, evals_per_sec)) => {
                let gevals = evals_per_sec / 1e9;
                println!("\n  Results:");
                println!("    Total evaluations: {}", total_evals);
                println!("    Elapsed time: {:.3}s", elapsed);
                println!("    Throughput: {:.2}B evals/sec", gevals);
                println!(
                    "    Time per step: {:.3}ms",
                    (elapsed / steps as f64) * 1000.0
                );

                results.push((
                    width,
                    height,
                    tile_count,
                    vram_mb,
                    steps,
                    elapsed,
                    evals_per_sec,
                ));
            }
            Err(e) => {
                println!("  Benchmark failed: {:?}", e);
            }
        }
    }

    // Summary
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:>12} {:>12} {:>10} {:>10} {:>12}",
        "Grid", "Tiles", "VRAM(GB)", "Steps", "B evals/sec"
    );
    println!(
        "{:->12} {:->12} {:->10} {:->10} {:->12}",
        "", "", "", "", ""
    );

    for (width, height, tiles, vram_mb, steps, _elapsed, throughput) in &results {
        println!(
            "{:>5}x{:<6} {:>12} {:>10.2} {:>10} {:>12.2}",
            width,
            height,
            tiles,
            vram_mb / 1024.0,
            steps,
            throughput / 1e9
        );
    }

    // Find peak
    if let Some((w, h, tiles, _, _, _, throughput)) =
        results.iter().max_by(|a, b| a.6.partial_cmp(&b.6).unwrap())
    {
        println!(
            "\n✓ Peak throughput: {:.2}B evals/sec at {}x{} ({:.1}M tiles)",
            throughput / 1e9,
            w,
            h,
            *tiles as f64 / 1e6
        );
    }

    // Compare with CPU
    println!("\n=== GPU vs CPU Comparison ===");
    println!("CPU SlimSimulation: ~400M evals/sec (best case)");
    if let Some((_, _, _, _, _, _, peak)) =
        results.iter().max_by(|a, b| a.6.partial_cmp(&b.6).unwrap())
    {
        let speedup = peak / 400_000_000.0;
        println!("GPU SlimSimulation: {:.2}B evals/sec", peak / 1e9);
        println!("Speedup: {:.1}x", speedup);
    }
}

#[cfg(not(all(feature = "cuda", feature = "gpu-prototype")))]
fn main() {
    println!("This example requires both 'cuda' and 'gpu-prototype' features.");
    println!(
        "Run with: cargo run --release --example epic105c_gpu_slim --features \"cuda gpu-prototype\""
    );
}
