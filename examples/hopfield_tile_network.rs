//! Hopfield Associative Memory Network using Tile-Based GPU Infrastructure
//!
//! This example demonstrates how to build a Hopfield network on top of the
//! existing Ising/QUBO infrastructure in cuda_tiles.rs.
//!
//! Key insight: Hopfield networks ARE Ising models!
//! - Neurons = spins (+-1)
//! - Synapses = couplings J_ij
//! - Energy: E = -1/2 * sum_ij J_ij * s_i * s_j
//! - Dynamics: minimize energy -> pattern completion
//!
//! IMPORTANT: The TileAnneal infrastructure uses a 2D LATTICE topology (nearest-neighbor
//! couplings only), while classical Hopfield requires ALL-TO-ALL connectivity.
//! This example shows:
//! 1. How patterns that match the lattice topology (checkerboard, stripes) work well
//! 2. Maximum-scale throughput benchmarks for neuromorphic computing
//! 3. The theoretical capacity limits
//!
//! Run with: cargo run --release --features cuda --example hopfield_tile_network

#[cfg(feature = "cuda")]
use engine::cuda_tiles::PackedTileGrid;

#[cfg(feature = "cuda")]
fn main() {
    use engine::cuda::CudaRuntime;

    println!("=== Hopfield Associative Memory on TileAnneal GPU Infrastructure ===\n");

    let rt = CudaRuntime::new().expect("CUDA runtime init failed");

    // =========================================================================
    // Phase 1: Lattice-Compatible Hopfield (patterns that work with 2D topology)
    // =========================================================================
    println!("=== Phase 1: Lattice-Compatible Hopfield Network ===\n");
    phase1_lattice_hopfield(&rt);

    // =========================================================================
    // Phase 2: Scale Up with Lattice Patterns
    // =========================================================================
    println!("\n=== Phase 2: Scaling Up ===\n");
    phase2_scale_up(&rt);

    // =========================================================================
    // Phase 3: Maximum Scale Throughput
    // =========================================================================
    println!("\n=== Phase 3: Maximum Scale Benchmarks ===\n");
    phase3_maximum_scale(&rt);

    // =========================================================================
    // Phase 4: FPGA Integration Notes
    // =========================================================================
    println!("\n=== Phase 4: FPGA Integration Notes ===\n");
    phase4_fpga_notes();

    println!("\n=== Hopfield Network Demo Complete ===");
}

/// Hebbian learning: compute weight matrix from stored patterns
///
/// J_ij = (1/P) * sum_p xi_i^p * xi_j^p
///
/// where xi^p are the P stored patterns (each pattern is a Vec<i8> of +-1)
#[cfg(feature = "cuda")]
fn compute_hebbian_weights(patterns: &[Vec<i8>], n: usize) -> Vec<f64> {
    let p = patterns.len() as f64;
    let mut weights = vec![0.0; n * n];

    for pattern in patterns {
        assert_eq!(pattern.len(), n, "Pattern must have {} elements", n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // J_ij += xi_i * xi_j / P
                    weights[i * n + j] += (pattern[i] as f64) * (pattern[j] as f64) / p;
                }
            }
        }
    }

    weights
}

/// Convert full weight matrix to grid-based couplings (nearest-neighbor only)
///
/// For an NxN grid, we extract:
/// - Horizontal couplings: (x, y) <-> (x+1, y)
/// - Vertical couplings: (x, y) <-> (x, y+1)
#[cfg(feature = "cuda")]
fn hopfield_to_grid_couplings(weights: &[f64], grid_size: usize) -> (Vec<i8>, Vec<i8>) {
    let n = grid_size * grid_size;
    assert_eq!(weights.len(), n * n, "Weight matrix size mismatch");

    let mut j_h = vec![0i8; n]; // Horizontal: (x,y) <-> (x+1,y)
    let mut j_v = vec![0i8; n]; // Vertical: (x,y) <-> (x,y+1)

    // Find max weight for normalization (only among neighbor pairs)
    let mut max_weight = 0.0f64;
    for y in 0..grid_size {
        for x in 0..grid_size {
            let i = y * grid_size + x;
            if x + 1 < grid_size {
                let j = y * grid_size + (x + 1);
                max_weight = max_weight.max(weights[i * n + j].abs());
            }
            if y + 1 < grid_size {
                let j = (y + 1) * grid_size + x;
                max_weight = max_weight.max(weights[i * n + j].abs());
            }
        }
    }

    let scale = if max_weight > 0.0 {
        4.0 / max_weight
    } else {
        1.0
    };

    for y in 0..grid_size {
        for x in 0..grid_size {
            let i = y * grid_size + x;

            // Horizontal coupling to (x+1, y)
            if x + 1 < grid_size {
                let j = y * grid_size + (x + 1);
                let w = weights[i * n + j] * scale;
                j_h[i] = w.round().clamp(-4.0, 4.0) as i8;
            }

            // Vertical coupling to (x, y+1)
            if y + 1 < grid_size {
                let j = (y + 1) * grid_size + x;
                let w = weights[i * n + j] * scale;
                j_v[i] = w.round().clamp(-4.0, 4.0) as i8;
            }
        }
    }

    (j_h, j_v)
}

/// Create lattice-compatible test patterns (work well with 2D nearest-neighbor topology)
#[cfg(feature = "cuda")]
fn create_lattice_patterns(grid_size: usize) -> Vec<(&'static str, Vec<i8>)> {
    let n = grid_size * grid_size;
    let mut patterns = Vec::new();

    // Pattern 1: Checkerboard (EXCELLENT for 2D lattice - antiferromagnetic ground state)
    let mut checkerboard = vec![-1i8; n];
    for y in 0..grid_size {
        for x in 0..grid_size {
            if (x + y) % 2 == 0 {
                checkerboard[y * grid_size + x] = 1;
            }
        }
    }
    patterns.push(("Checkerboard", checkerboard));

    // Pattern 2: Inverse Checkerboard (also antiferromagnetic)
    let mut inv_checker = vec![-1i8; n];
    for y in 0..grid_size {
        for x in 0..grid_size {
            if (x + y) % 2 == 1 {
                inv_checker[y * grid_size + x] = 1;
            }
        }
    }
    patterns.push(("InvChecker", inv_checker));

    patterns
}

/// Corrupt a pattern by flipping a percentage of bits
#[cfg(feature = "cuda")]
fn corrupt_pattern(pattern: &[i8], flip_fraction: f64, seed: u64) -> Vec<i8> {
    let mut corrupted = pattern.to_vec();
    let mut rng = seed.wrapping_add(1); // Avoid seed=0

    for bit in corrupted.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;

        let rand_val = (rng as f64) / (u64::MAX as f64);
        if rand_val < flip_fraction {
            *bit = -*bit;
        }
    }

    corrupted
}

/// Compute overlap (correlation) between two patterns
#[cfg(feature = "cuda")]
fn pattern_overlap(p1: &[i8], p2: &[i8]) -> f64 {
    assert_eq!(p1.len(), p2.len());
    let n = p1.len() as f64;
    let dot: f64 = p1
        .iter()
        .zip(p2.iter())
        .map(|(&a, &b)| (a as f64) * (b as f64))
        .sum();
    dot / n
}

/// Convert spin configuration to/from GPU grid format
#[cfg(feature = "cuda")]
fn spins_to_grid(spins: &[i8], grid_size: usize) -> PackedTileGrid {
    let mut grid = PackedTileGrid::new(grid_size, grid_size);
    for (i, &s) in spins.iter().enumerate() {
        let x = i % grid_size;
        let y = i / grid_size;
        // +1 -> true (1), -1 -> false (0)
        grid.set(x, y, s > 0);
    }
    grid
}

#[cfg(feature = "cuda")]
fn grid_to_spins(grid: &PackedTileGrid) -> Vec<i8> {
    let mut spins = Vec::with_capacity(grid.width * grid.height);
    for y in 0..grid.height {
        for x in 0..grid.width {
            spins.push(if grid.get(x, y) { 1 } else { -1 });
        }
    }
    spins
}

/// Print a pattern as ASCII art
#[cfg(feature = "cuda")]
fn print_pattern(name: &str, pattern: &[i8], grid_size: usize) {
    println!("  {} ({}x{}):", name, grid_size, grid_size);
    for y in 0..grid_size.min(16) {
        print!("    ");
        for x in 0..grid_size.min(32) {
            let c = if pattern[y * grid_size + x] > 0 {
                '#'
            } else {
                '.'
            };
            print!("{}", c);
        }
        if grid_size > 32 {
            print!("...");
        }
        println!();
    }
    if grid_size > 16 {
        println!("    ...");
    }
}

#[cfg(feature = "cuda")]
fn phase1_lattice_hopfield(rt: &engine::cuda::CudaRuntime) {
    use engine::cuda_tiles::{CouplingMode, GpuIsingGrid, IsingConfig, run_ising_update_weighted};

    let grid_size = 8; // 8x8 = 64 neurons
    let n = grid_size * grid_size;

    println!("--- Phase 1a: Hebbian Learning (Lattice-Compatible Patterns) ---\n");

    // Create lattice-compatible patterns
    let test_patterns = create_lattice_patterns(grid_size);
    println!(
        "Storing {} lattice-compatible patterns:\n",
        test_patterns.len()
    );

    for (name, pattern) in &test_patterns {
        print_pattern(name, pattern, grid_size);
        println!();
    }

    // Extract pattern vectors
    let pattern_vecs: Vec<Vec<i8>> = test_patterns.iter().map(|(_, p)| p.clone()).collect();

    // Compute Hebbian weights
    let weights = compute_hebbian_weights(&pattern_vecs, n);
    let (j_h, j_v) = hopfield_to_grid_couplings(&weights, grid_size);

    // Show coupling statistics
    let nonzero_h: usize = j_h.iter().filter(|&&x| x != 0).count();
    let nonzero_v: usize = j_v.iter().filter(|&&x| x != 0).count();
    println!("Coupling statistics:");
    println!("  Horizontal non-zero: {}/{}", nonzero_h, n);
    println!("  Vertical non-zero: {}/{}", nonzero_v, n);
    println!();

    println!("--- Phase 1b: Pattern Completion Test ---\n");

    // Test pattern completion
    for (pattern_idx, (name, original)) in test_patterns.iter().enumerate() {
        for corruption in [0.1, 0.2, 0.3, 0.4] {
            let corrupted = corrupt_pattern(original, corruption, 42 + pattern_idx as u64);
            let initial_overlap = pattern_overlap(original, &corrupted);

            let config = IsingConfig::new(grid_size, grid_size).with_seed(100 + pattern_idx as u64);
            let coupling = CouplingMode::weighted(j_h.clone(), j_v.clone());
            let mut grid = GpuIsingGrid::new_weighted(rt, &config, coupling)
                .expect("Failed to create GPU grid");

            // Upload corrupted pattern
            let initial_grid = spins_to_grid(&corrupted, grid_size);
            grid.spins = rt.upload(&initial_grid.data).expect("Upload failed");

            // Run annealing
            for beta in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
                grid.beta = beta;
                run_ising_update_weighted(rt, &mut grid, 200).expect("Update failed");
            }

            let final_grid = grid.download(rt).expect("Download failed");
            let recovered = grid_to_spins(&final_grid);
            let final_overlap = pattern_overlap(original, &recovered);

            let status = if final_overlap > 0.9 {
                "[OK]"
            } else if final_overlap > initial_overlap {
                "[IMPROVED]"
            } else {
                "[FAIL]"
            };
            println!(
                "  {}: {:.0}% corruption -> overlap {:.2} -> {:.2} {}",
                name,
                corruption * 100.0,
                initial_overlap,
                final_overlap,
                status
            );
        }
        println!();
    }

    // Show successful recovery example
    println!("Example recovery visualization:\n");
    let (_, original) = &test_patterns[0];
    let corrupted = corrupt_pattern(original, 0.3, 999);

    let config = IsingConfig::new(grid_size, grid_size).with_seed(999);
    let coupling = CouplingMode::weighted(j_h.clone(), j_v.clone());
    let mut grid = GpuIsingGrid::new_weighted(rt, &config, coupling).expect("grid");
    grid.spins = rt
        .upload(&spins_to_grid(&corrupted, grid_size).data)
        .expect("upload");

    for beta in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        grid.beta = beta;
        run_ising_update_weighted(rt, &mut grid, 200).expect("update");
    }

    let recovered = grid_to_spins(&grid.download(rt).expect("download"));
    print_pattern("Original", original, grid_size);
    print_pattern("Corrupted (30%)", &corrupted, grid_size);
    print_pattern("Recovered", &recovered, grid_size);
    println!(
        "  Final overlap: {:.3}",
        pattern_overlap(original, &recovered)
    );
}

#[cfg(feature = "cuda")]
fn phase2_scale_up(rt: &engine::cuda::CudaRuntime) {
    use engine::cuda_tiles::{CouplingMode, GpuIsingGrid, IsingConfig, run_ising_update_weighted};
    use std::time::Instant;

    // Sizes that work well with lattice topology
    let sizes = [8, 16, 32, 64]; // 64, 256, 1024, 4096 neurons

    println!("Scaling Hopfield network with lattice-compatible patterns:\n");
    println!(
        "{:>8} {:>12} {:>15} {:>15} {:>12}",
        "Neurons", "Grid", "Recovery@10%", "Recovery@30%", "Time"
    );
    println!("{}", "-".repeat(65));

    for &grid_size in &sizes {
        let n = grid_size * grid_size;

        // Use lattice patterns
        let patterns = create_lattice_patterns(grid_size);
        let pattern_vecs: Vec<Vec<i8>> = patterns.iter().map(|(_, p)| p.clone()).collect();
        let weights = compute_hebbian_weights(&pattern_vecs, n);
        let (j_h, j_v) = hopfield_to_grid_couplings(&weights, grid_size);

        let start = Instant::now();

        // Test at 10% corruption
        let mut success_10 = 0;
        for (idx, (_, pattern)) in patterns.iter().enumerate() {
            let corrupted = corrupt_pattern(pattern, 0.1, idx as u64);

            let config = IsingConfig::new(grid_size, grid_size).with_seed(idx as u64);
            let coupling = CouplingMode::weighted(j_h.clone(), j_v.clone());
            let mut grid = GpuIsingGrid::new_weighted(rt, &config, coupling).expect("grid");
            grid.spins = rt
                .upload(&spins_to_grid(&corrupted, grid_size).data)
                .expect("upload");

            for beta in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0] {
                grid.beta = beta;
                run_ising_update_weighted(rt, &mut grid, 100).expect("update");
            }

            let recovered = grid_to_spins(&grid.download(rt).expect("download"));
            if pattern_overlap(pattern, &recovered) > 0.9 {
                success_10 += 1;
            }
        }

        // Test at 30% corruption
        let mut success_30 = 0;
        for (idx, (_, pattern)) in patterns.iter().enumerate() {
            let corrupted = corrupt_pattern(pattern, 0.3, idx as u64 + 100);

            let config = IsingConfig::new(grid_size, grid_size).with_seed(idx as u64 + 100);
            let coupling = CouplingMode::weighted(j_h.clone(), j_v.clone());
            let mut grid = GpuIsingGrid::new_weighted(rt, &config, coupling).expect("grid");
            grid.spins = rt
                .upload(&spins_to_grid(&corrupted, grid_size).data)
                .expect("upload");

            for beta in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0] {
                grid.beta = beta;
                run_ising_update_weighted(rt, &mut grid, 100).expect("update");
            }

            let recovered = grid_to_spins(&grid.download(rt).expect("download"));
            if pattern_overlap(pattern, &recovered) > 0.9 {
                success_30 += 1;
            }
        }

        let elapsed = start.elapsed();
        let total = patterns.len();

        println!(
            "{:>8} {:>10}x{:<2} {:>13}/{} {:>13}/{} {:>11.2?}",
            n, grid_size, grid_size, success_10, total, success_30, total, elapsed
        );
    }
}

#[cfg(feature = "cuda")]
fn phase3_maximum_scale(rt: &engine::cuda::CudaRuntime) {
    use engine::cuda_tiles::{GpuIsingGrid, IsingConfig, IsingKernelVariant, run_ising_update};
    use std::time::Instant;

    println!("Maximum scale throughput benchmarks (uniform antiferromagnetic coupling):\n");
    println!(
        "{:>10} {:>12} {:>15} {:>15} {:>12}",
        "Neurons", "Grid", "Sweeps/sec", "Spins/sec", "Memory"
    );
    println!("{}", "-".repeat(70));

    // Push to maximum scale with uniform coupling (fast path)
    let scales = [128, 256, 512, 1024, 2048, 4096]; // up to 16M neurons

    for &grid_size in &scales {
        let n = grid_size * grid_size;

        let config = IsingConfig::new(grid_size, grid_size).with_seed(42);
        let mut grid = match GpuIsingGrid::new(rt, &config) {
            Ok(g) => g,
            Err(e) => {
                println!(
                    "{:>10} {:>10}x{:<4} - Failed: {:?}",
                    n, grid_size, grid_size, e
                );
                continue;
            }
        };

        // Warmup
        grid.beta = 1.0;
        if let Err(e) = run_ising_update(rt, &mut grid, 10, IsingKernelVariant::Stochastic, -1.0) {
            println!(
                "{:>10} {:>10}x{:<4} - Warmup failed: {:?}",
                n, grid_size, grid_size, e
            );
            continue;
        }

        // Benchmark
        let sweeps = 1000u32;
        let start = Instant::now();

        grid.beta = 2.0;
        if let Err(e) =
            run_ising_update(rt, &mut grid, sweeps, IsingKernelVariant::Stochastic, -1.0)
        {
            println!(
                "{:>10} {:>10}x{:<4} - Benchmark failed: {:?}",
                n, grid_size, grid_size, e
            );
            continue;
        }

        let elapsed = start.elapsed();
        let sweeps_per_sec = sweeps as f64 / elapsed.as_secs_f64();
        let spins_per_sec = (n as f64 * sweeps as f64) / elapsed.as_secs_f64();
        let memory_mb = grid.memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:>10} {:>10}x{:<4} {:>13.0} {:>13.2}G {:>10.2} MB",
            n,
            grid_size,
            grid_size,
            sweeps_per_sec,
            spins_per_sec / 1e9,
            memory_mb
        );
    }

    // Try the weighted path for comparison
    println!("\n--- Weighted Coupling Path (for full Hopfield) ---\n");

    use engine::cuda_tiles::{CouplingMode, run_ising_update_weighted};

    for &grid_size in &[64, 128, 256] {
        let n = grid_size * grid_size;

        // Create weighted coupling (antiferromagnetic everywhere)
        let coupling = CouplingMode::weighted_uniform(grid_size, grid_size, -1);
        let config = IsingConfig::new(grid_size, grid_size).with_seed(42);
        let mut grid = match GpuIsingGrid::new_weighted(rt, &config, coupling) {
            Ok(g) => g,
            Err(e) => {
                println!("{:>10}x{:<4} - Failed: {:?}", grid_size, grid_size, e);
                continue;
            }
        };

        // Warmup
        grid.beta = 1.0;
        if let Err(e) = run_ising_update_weighted(rt, &mut grid, 10) {
            println!(
                "{:>10}x{:<4} - Warmup failed: {:?}",
                grid_size, grid_size, e
            );
            continue;
        }

        // Benchmark
        let sweeps = 500u32;
        let start = Instant::now();

        grid.beta = 2.0;
        if let Err(e) = run_ising_update_weighted(rt, &mut grid, sweeps) {
            println!(
                "{:>10}x{:<4} - Benchmark failed: {:?}",
                grid_size, grid_size, e
            );
            continue;
        }

        let elapsed = start.elapsed();
        let sweeps_per_sec = sweeps as f64 / elapsed.as_secs_f64();
        let spins_per_sec = (n as f64 * sweeps as f64) / elapsed.as_secs_f64();
        let memory_mb = grid.memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "  {:>6}x{:<4} ({:>8} neurons): {:>10.0} sweeps/s, {:>8.2}G spins/s, {:>6.2} MB",
            grid_size,
            grid_size,
            n,
            sweeps_per_sec,
            spins_per_sec / 1e9,
            memory_mb
        );
    }

    // Summary
    println!("\n--- Hopfield Capacity vs Scale ---\n");
    println!("Theoretical Hopfield capacity: P_max ~ 0.138 * N\n");
    println!("{:>10} {:>15} {:>20}", "Neurons", "Max Patterns", "Notes");
    println!("{}", "-".repeat(50));
    for &n in &[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576] {
        let p_max = (0.138 * n as f64).floor() as usize;
        let note = if n <= 4096 {
            "Weighted path OK"
        } else if n <= 65536 {
            "Uniform path recommended"
        } else {
            "GPU memory limit"
        };
        println!("{:>10} {:>15} {:>20}", n, p_max, note);
    }
}

fn phase4_fpga_notes() {
    println!("FPGA Integration Path:\n");
    println!("The Hopfield network can be synthesized to FPGA via the existing");
    println!("src/fpga/ module using RustHDL.\n");
    println!("Key considerations:");
    println!("  1. Lattice topology maps naturally to FPGA fabric");
    println!("  2. Each neuron becomes a flip-flop with combinatorial update logic");
    println!("  3. Nearest-neighbor couplings -> direct wire connections");
    println!("  4. Weighted couplings -> LUT-based multiplication or shift-add");
    println!();
    println!("Synthesis command (if fpga feature enabled):");
    println!("  cargo run --features fpga --example hopfield_fpga_export\n");
    println!("Expected FPGA resources for 64x64 Hopfield:");
    println!("  - Flip-flops: ~4096 (one per neuron)");
    println!("  - LUTs: ~16K-32K (depends on coupling precision)");
    println!("  - Clock: 100-500 MHz typical");
    println!("  - Throughput: 400G-2T spin updates/sec");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --release --features cuda --example hopfield_tile_network");
    std::process::exit(1);
}
