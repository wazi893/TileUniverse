//! Tile8 Ising Mode Demo
//!
//! Demonstrates approaches to Ising simulation on the tile substrate:
//! 1. IsingGrid: High-level API with proper Gibbs sampling and annealing
//! 2. Native tiles: IsingNode tile type using SlimSimulation infrastructure
//! 3. GPU acceleration: CUDA-accelerated Ising simulation (with --features cuda)
//!
//! Run with: cargo run --release --example tile8_ising_demo
//! Run with GPU: cargo run --release --features cuda --example tile8_ising_demo

use engine::slim_simulation::SlimSimulation;
use engine::tile_meta::TileType;
use engine::tile8::{GridMaxCut, IsingConfig, IsingGrid};
use std::time::Instant;

#[cfg(feature = "cuda")]
use engine::tile8::{GpuGridMaxCut, GpuIsingGrid};

fn main() {
    println!("=== Tile8 Ising Mode Demo ===\n");

    // Demo 1: IsingGrid high-level API
    demo_ising_grid();

    // Demo 2: GridMaxCut benchmark
    demo_grid_maxcut();

    // Demo 3: Native IsingNode tiles
    demo_native_ising_tiles();

    // Demo 4: Performance comparison (CPU)
    demo_performance();

    // Demo 5: GPU acceleration (if CUDA feature enabled)
    #[cfg(feature = "cuda")]
    demo_gpu_ising();

    // Demo 6: GPU vs CPU comparison (if CUDA feature enabled)
    #[cfg(feature = "cuda")]
    demo_gpu_vs_cpu();
}

fn demo_ising_grid() {
    println!("--- Demo 1: IsingGrid API ---\n");

    // Create a small ferromagnetic Ising model
    let config = IsingConfig::new()
        .with_beta(2.0)
        .with_coupling(1.0) // Ferromagnetic
        .with_seed(42);

    let mut grid = IsingGrid::new(16, 16, config);
    println!(
        "Created {}x{} Ising grid ({} spins)",
        grid.width(),
        grid.height(),
        grid.spin_count()
    );

    // Randomize and measure initial state
    grid.randomize();
    println!("Initial energy: {:.2}", grid.energy());
    println!("Initial magnetization: {:.3}", grid.magnetization());

    // Run annealing
    let result = grid.anneal(500, 0.1, 5.0);

    println!("\nAfter annealing:");
    println!("  Final energy: {:.2}", result.final_energy);
    println!("  Best energy: {:.2}", result.best_energy);
    println!("  Total sweeps: {}", result.total_sweeps);
    println!("  Flip rate: {:.1}%", grid.flip_rate() * 100.0);
    println!("  Final magnetization: {:.3}", grid.magnetization());

    // Ferromagnetic should align (high |M|)
    let mag = grid.magnetization().abs();
    if mag > 0.8 {
        println!("  ✓ Spins aligned (ferromagnetic ground state)");
    }
    println!();
}

fn demo_grid_maxcut() {
    println!("--- Demo 2: GridMaxCut Benchmark ---\n");

    let sizes = [(8, 8), (16, 16), (32, 32)];

    for (w, h) in sizes {
        let mut problem = GridMaxCut::new(w, h, 42);
        let start = Instant::now();
        let result = problem.solve(1000, 0.1, 5.0);
        let elapsed = start.elapsed();

        println!(
            "{}x{} grid: {} / {} edges cut ({:.1}%) in {:?}",
            w,
            h,
            result.cut_value,
            result.max_edges,
            result.cut_fraction * 100.0,
            elapsed
        );
    }
    println!();
}

fn demo_native_ising_tiles() {
    println!("--- Demo 3: Native IsingNode Tiles ---\n");

    // Create a SlimSimulation and fill with IsingNode tiles
    let size = 16;
    let mut sim = SlimSimulation::with_size(size, size);

    // Initialize all tiles as IsingNode with random spins
    let mut rng_state = 12345u64;
    for y in 0..size {
        for x in 0..size {
            sim.set_tile(x, y, TileType::IsingNode);
            // Random initial spin (0 or 1)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let spin = (rng_state >> 63);
            sim.set_logic_value(x, y, spin);
        }
    }

    // Count initial spins
    let initial_up = count_spins_up(&sim, size);
    println!(
        "Initial state: {} spins up, {} spins down",
        initial_up,
        size * size - initial_up
    );

    // Run tile evaluation passes
    // The IsingNode tiles will converge to a low-energy configuration
    println!("\nRunning tile evaluations...");
    for pass in 0..100 {
        let (evals, changes) = sim.eval_full_grid();
        if pass < 5 || pass % 20 == 0 {
            let up = count_spins_up(&sim, size);
            println!(
                "  Pass {}: {} evals, {} changes, {} spins up",
                pass, evals, changes, up
            );
        }
        if changes == 0 {
            println!("  Converged at pass {}!", pass);
            break;
        }
    }

    // Check final state
    let final_up = count_spins_up(&sim, size);
    println!(
        "\nFinal state: {} spins up, {} spins down",
        final_up,
        size * size - final_up
    );

    // Count cut edges (antiferromagnetic IsingNode prefers opposite neighbors)
    let cut = count_cut_edges(&sim, size);
    let max_edges = 2 * size * (size - 1);
    println!(
        "Cut edges: {} / {} ({:.1}%)",
        cut,
        max_edges,
        100.0 * cut as f64 / max_edges as f64
    );
    println!();
}

fn demo_performance() {
    println!("--- Demo 4: Performance Comparison ---\n");

    let size = 64;
    let sweeps = 100;

    // IsingGrid performance
    let config = IsingConfig::new().antiferromagnetic().with_seed(42);
    let mut grid = IsingGrid::new(size, size, config);

    let start = Instant::now();
    for _ in 0..sweeps {
        grid.sweep();
    }
    let ising_time = start.elapsed();

    // Native tile performance
    let mut sim = SlimSimulation::with_size(size, size);
    for y in 0..size {
        for x in 0..size {
            sim.set_tile(x, y, TileType::IsingNode);
            sim.set_logic_value(x, y, ((x + y) % 2) as u64);
        }
    }

    let start = Instant::now();
    for _ in 0..sweeps {
        sim.eval_full_grid();
    }
    let tile_time = start.elapsed();

    let spin_count = size * size;
    let ising_rate = (sweeps * spin_count) as f64 / ising_time.as_secs_f64();
    let tile_rate = (sweeps * spin_count) as f64 / tile_time.as_secs_f64();

    println!(
        "{}x{} grid ({} spins), {} sweeps:",
        size, size, spin_count, sweeps
    );
    println!(
        "  IsingGrid:     {:?} ({:.2}M updates/sec)",
        ising_time,
        ising_rate / 1e6
    );
    println!(
        "  Native tiles:  {:?} ({:.2}M evals/sec)",
        tile_time,
        tile_rate / 1e6
    );
    println!("  Speedup:       {:.1}x", tile_rate / ising_rate);
    println!();

    println!("Note: Native tiles use deterministic updates (no RNG)");
    println!("      IsingGrid uses proper Gibbs sampling with stochastic updates");
}

fn count_spins_up(sim: &SlimSimulation, size: usize) -> usize {
    let mut count = 0;
    for y in 0..size {
        for x in 0..size {
            if sim.get_logic_at(x, y) != 0 {
                count += 1;
            }
        }
    }
    count
}

fn count_cut_edges(sim: &SlimSimulation, size: usize) -> usize {
    let mut cut = 0;
    for y in 0..size {
        for x in 0..size {
            let s_i = sim.get_logic_at(x, y) != 0;

            // Right neighbor
            if x + 1 < size {
                let s_j = sim.get_logic_at(x + 1, y) != 0;
                if s_i != s_j {
                    cut += 1;
                }
            }

            // Down neighbor
            if y + 1 < size {
                let s_j = sim.get_logic_at(x, y + 1) != 0;
                if s_i != s_j {
                    cut += 1;
                }
            }
        }
    }
    cut
}

// ============== GPU Demos (CUDA feature) ==============

#[cfg(feature = "cuda")]
fn demo_gpu_ising() {
    println!("--- Demo 5: GPU Ising Acceleration ---\n");

    // Test GPU MaxCut on various grid sizes
    let sizes = [(32, 32), (64, 64), (128, 128), (256, 256)];

    for (w, h) in sizes {
        match GpuGridMaxCut::new(w, h, 42) {
            Ok(mut problem) => {
                let start = Instant::now();
                match problem.solve(1000, 0.1, 5.0) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        println!(
                            "GPU {}x{}: {} / {} edges ({:.1}%) in {:?}",
                            w,
                            h,
                            result.cut_value,
                            result.max_edges,
                            result.cut_fraction * 100.0,
                            elapsed
                        );
                    }
                    Err(e) => println!("GPU {}x{}: solve error: {:?}", w, h, e),
                }
            }
            Err(e) => {
                println!("GPU {}x{}: creation error: {:?}", w, h, e);
                break;
            }
        }
    }
    println!();
}

#[cfg(feature = "cuda")]
fn demo_gpu_vs_cpu() {
    println!("--- Demo 6: GPU vs CPU Performance ---\n");

    let size = 128;
    let sweeps = 500;
    let spin_count = size * size;

    // CPU IsingGrid
    let config = IsingConfig::new().antiferromagnetic().with_seed(42);
    let mut cpu_grid = IsingGrid::new(size, size, config);
    cpu_grid.randomize();

    let start = Instant::now();
    for _ in 0..sweeps {
        cpu_grid.sweep();
    }
    let cpu_time = start.elapsed();
    let cpu_rate = (sweeps * spin_count) as f64 / cpu_time.as_secs_f64();

    // GPU IsingGrid
    match GpuIsingGrid::new_antiferromagnetic(size, size, 1.0, 42) {
        Ok(mut gpu_grid) => {
            gpu_grid.set_uniform_field(0.0).unwrap();
            gpu_grid.randomize().unwrap();

            let start = Instant::now();
            for _ in 0..sweeps {
                gpu_grid.sweep_checkerboard().unwrap();
            }
            // Synchronize to get accurate timing
            let _ = gpu_grid.compute_energy();
            let gpu_time = start.elapsed();
            let gpu_rate = (sweeps * spin_count) as f64 / gpu_time.as_secs_f64();

            println!(
                "{}x{} grid ({} spins), {} sweeps:",
                size, size, spin_count, sweeps
            );
            println!(
                "  CPU IsingGrid:  {:?} ({:.2}M updates/sec)",
                cpu_time,
                cpu_rate / 1e6
            );
            println!(
                "  GPU IsingGrid:  {:?} ({:.2}M updates/sec)",
                gpu_time,
                gpu_rate / 1e6
            );
            println!("  GPU Speedup:    {:.1}x", gpu_rate / cpu_rate);
        }
        Err(e) => {
            println!("GPU initialization failed: {:?}", e);
            println!("Falling back to CPU-only results:");
            println!(
                "  CPU IsingGrid: {:?} ({:.2}M updates/sec)",
                cpu_time,
                cpu_rate / 1e6
            );
        }
    }

    println!();

    // Large grid comparison
    let large_size = 512;
    let large_sweeps = 100;
    let large_spin_count = large_size * large_size;

    println!(
        "Large grid benchmark ({}x{}, {} spins, {} sweeps):",
        large_size, large_size, large_spin_count, large_sweeps
    );

    match GpuIsingGrid::new_antiferromagnetic(large_size, large_size, 1.0, 42) {
        Ok(mut gpu_grid) => {
            gpu_grid.set_uniform_field(0.0).unwrap();
            gpu_grid.randomize().unwrap();

            let start = Instant::now();
            for _ in 0..large_sweeps {
                gpu_grid.sweep_checkerboard().unwrap();
            }
            let _ = gpu_grid.compute_energy();
            let gpu_time = start.elapsed();
            let gpu_rate = (large_sweeps * large_spin_count) as f64 / gpu_time.as_secs_f64();

            println!("  GPU: {:?} ({:.2}M updates/sec)", gpu_time, gpu_rate / 1e6);

            // Get final energy to verify computation
            let energy = gpu_grid.compute_energy().unwrap();
            println!("  Final energy: {:.2}", energy);
        }
        Err(e) => println!("  GPU error: {:?}", e),
    }

    println!();
}
