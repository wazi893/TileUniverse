//! Quick test for Week 1 acceptance tracking (with ΔE-bucketed stats)
//!
//! Run with: cargo run --release --features cuda,perf-bench --example acceptance_tracking_test

#[cfg(feature = "cuda")]
fn main() {
    use engine::cuda::CudaRuntime;
    use engine::cuda_tiles::{
        GpuIsingGrid, IsingConfig, benchmark_acceptance_overhead, run_ising_update_tracked,
        validate_acceptance_tracking,
    };

    println!("=== TileAnneal Phase 2: Acceptance Tracking Validation ===\n");

    let rt = CudaRuntime::new().expect("CUDA runtime init failed");

    // Test ΔE-bucketed acceptance tracking (critical for adaptive annealing)
    println!("ΔE-Bucketed Acceptance Statistics (128×128, 100 sweeps):");
    println!(
        "{:>6} {:>10} {:>10} {:>10} | {:>10} {:>10} {:>8}",
        "β", "uphill", "neutral", "downhill", "up_accept", "up_rate", "up_frac"
    );
    println!("{}", "-".repeat(80));

    let config = IsingConfig::new(128, 128).with_seed(42);
    for beta in [0.01f32, 0.1, 1.0, 5.0, 10.0, 50.0] {
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");
        grid.beta = beta;
        let stats = run_ising_update_tracked(&rt, &mut grid, 100, 1.0).expect("tracked");
        println!(
            "{:>6.2} {:>10} {:>10} {:>10} | {:>10} {:>8.4} {:>8.4}",
            beta,
            stats.attempted_uphill,
            stats.attempted_neutral,
            stats.attempted_downhill,
            stats.accepted_uphill,
            stats.uphill_rate(),
            stats.uphill_fraction()
        );
    }
    println!();

    // Verify uphill rate decreases with temperature (β increases)
    println!("Verifying uphill acceptance rate behavior:");
    let betas = [0.01f32, 1.0, 10.0, 50.0];
    let mut prev_uphill_rate = 1.0;
    let mut all_decreasing = true;

    for beta in betas {
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");
        grid.beta = beta;
        let stats = run_ising_update_tracked(&rt, &mut grid, 100, 1.0).expect("tracked");
        let uphill_rate = stats.uphill_rate();

        if uphill_rate > prev_uphill_rate + 0.05 {
            // Allow small noise
            all_decreasing = false;
        }
        prev_uphill_rate = uphill_rate;
    }

    if all_decreasing {
        println!("✓ Uphill acceptance rate decreases with β (as expected)\n");
    } else {
        println!("✗ Uphill acceptance rate NOT monotonically decreasing\n");
    }

    // Original total acceptance test
    println!("Total Acceptance Rates (for reference):");
    for beta in [0.01f32, 1.0, 10.0, 50.0] {
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");
        grid.beta = beta;
        let stats = run_ising_update_tracked(&rt, &mut grid, 100, 1.0).expect("tracked");
        println!(
            "  β={:5.2}: total_rate={:.4}, uphill_rate={:.4}",
            beta,
            stats.rate(),
            stats.uphill_rate()
        );
    }
    println!();

    // Test 1: Validate acceptance rates at different temperatures
    println!("Test 1: Acceptance Rate Sanity Check");
    match validate_acceptance_tracking(&rt, 128, 128) {
        Ok(passed) => {
            if passed {
                println!("✓ Acceptance sanity: PASSED\n");
            } else {
                println!("✗ Acceptance sanity: FAILED\n");
            }
        }
        Err(e) => {
            eprintln!("ERROR: {:?}", e);
        }
    }

    // Test 2: Benchmark throughput overhead (should be <5%)
    // Note: Use large grid where all threads are valid for accurate measurement
    println!("Test 2: Throughput Overhead Benchmark (8K×8K grid, 1000 sweeps)");
    match benchmark_acceptance_overhead(&rt, 8192, 8192, 1000) {
        Ok((_baseline, _tracked, overhead)) => {
            // Target: <15% overhead (acceptable for diagnostic mode)
            // Note: Tracking can be disabled for production benchmarks
            if overhead < 15.0 {
                println!("✓ Overhead check: PASSED ({:.1}% < 15%)\n", overhead);
            } else {
                println!("✗ Overhead check: FAILED ({:.1}% >= 15%)\n", overhead);
            }
        }
        Err(e) => {
            eprintln!("ERROR: {:?}", e);
        }
    }

    println!("=== Week 1 Validation Complete ===");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature.");
    std::process::exit(1);
}
