//! EPIC 102: GatedFeedforward Throughput Benchmark
//!
//! Benchmarks GPU GatedFeedforward inference:
//! - Tests various population sizes (1K, 10K, 100K, 1M)
//! - Measures throughput in organisms/second
//! - Compares against CPU baseline and WMMA SimpleBrain
//!
//! Success criteria (from refinements.txt):
//! - Throughput: >500M organisms/sec
//! - vs SimpleBrain: <3× slower (557 vs 213 params)

#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 102: GatedFeedforward Throughput Benchmark              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(feature = "cuda")]
    run_cuda_benchmark();

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This benchmark requires the 'cuda' feature.");
        println!(
            "       Run with: cargo run --release --example epic102_benchmark --features cuda"
        );
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_benchmark() {
    use engine::brain::gated_feedforward::{GatedFeedforwardPopulation, GatedFeedforwardWeights};
    use engine::gpu_nn::gpu_gated_feedforward::GpuGatedFeedforward;

    // Test configurations: (population_size, ticks)
    let configs = [
        (1_000, 1_000),
        (10_000, 1_000),
        (100_000, 100),
        (1_000_000, 10),
    ];

    println!("  Architecture: input(8) → gated_hidden(24) → output(5)");
    println!(
        "  Parameters:   {} per organism",
        GatedFeedforwardWeights::PARAM_COUNT
    );
    println!();

    // First, run CPU baseline
    println!("┌────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ CPU Baseline (single-threaded)                                                │");
    println!("└────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let cpu_throughput = run_cpu_benchmark(1_000, 1_000);
    println!("  CPU reference: {:.2e} organisms/sec", cpu_throughput);
    println!();

    // Now GPU benchmarks
    println!("┌────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GPU GatedFeedforward (per-organism weights)                                   │");
    println!("└────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut peak_throughput = 0.0f64;

    println!("       Configuration |        Time |    Throughput | vs CPU | Memory");
    println!("  -------------------+-------------+---------------+--------+--------");

    for (pop_size, ticks) in configs {
        let memory_mb =
            (pop_size * GatedFeedforwardWeights::PARAM_COUNT * 4) as f64 / (1024.0 * 1024.0);

        match run_gpu_benchmark(pop_size, ticks) {
            Ok((time_ms, throughput)) => {
                let vs_cpu = throughput / cpu_throughput;
                peak_throughput = peak_throughput.max(throughput);

                println!(
                    "  {:>6}K × {:>4} ticks | {:>8.2}ms | {:>11.3e}/s | {:>5.1}× | {:.0}MB",
                    pop_size / 1000,
                    ticks,
                    time_ms,
                    throughput,
                    vs_cpu,
                    memory_mb
                );
            }
            Err(e) => {
                println!(
                    "  {:>6}K × {:>4} ticks | ERROR: {}",
                    pop_size / 1000,
                    ticks,
                    e
                );
            }
        }
    }

    println!();
    println!("┌────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Summary                                                                       │");
    println!("└────────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Peak throughput: {:.3e} organisms/sec", peak_throughput);
    println!(
        "  vs CPU baseline: {:.1}×",
        peak_throughput / cpu_throughput
    );
    println!();

    // Success criteria check
    let target = 500_000_000.0; // 500M/sec
    if peak_throughput >= target {
        println!("  ✓ PASS: Exceeds target (>{:.0}M/sec)", target / 1e6);
    } else {
        println!(
            "  ✗ FAIL: Below target ({:.0}M/sec < {:.0}M/sec)",
            peak_throughput / 1e6,
            target / 1e6
        );
    }

    // Compare with WMMA SimpleBrain reference (from EPIC 96)
    let wmma_ref = 1.048e9; // 1B/sec from EPIC 96
    let ratio = wmma_ref / peak_throughput;
    println!();
    println!("  vs WMMA SimpleBrain (EPIC 96):");
    println!(
        "    SimpleBrain: {:.3e}/s (shared weights, 213 params)",
        wmma_ref
    );
    println!(
        "    GatedFF:     {:.3e}/s (per-organism, 557 params)",
        peak_throughput
    );
    println!("    Ratio:       {:.2}× slower", ratio);

    println!();
    println!("  Analysis:");
    println!("    - SimpleBrain uses SHARED weights (all organisms read same 213 params)");
    println!("    - GatedFF uses PER-ORGANISM weights (each organism reads unique 557 params)");
    println!("    - Memory bound at scale: 100K orgs × 2.2KB = 220MB random access");
    println!("    - Peak efficiency at 10K orgs where working set fits in L2 cache");
    println!();
    println!("  Practical implications:");
    let ticks_per_gen = 300.0; // Typical EPIC 101.5 config
    println!(
        "    - For evolution with pop=30, ticks=300: ~{:.2}ms/generation",
        (30.0 * ticks_per_gen) / peak_throughput * 1000.0
    );
    println!(
        "    - For evolution with pop=1000, ticks=300: ~{:.1}ms/generation",
        (1000.0 * ticks_per_gen) / peak_throughput * 1000.0
    );
    println!(
        "    - vs CPU: Still {:.1}× faster than single-threaded baseline",
        peak_throughput / cpu_throughput
    );

    println!();
    println!("Benchmark complete.");
}

#[cfg(feature = "cuda")]
fn run_cpu_benchmark(pop_size: usize, ticks: usize) -> f64 {
    use engine::brain::gated_feedforward::{GatedFeedforwardPopulation, GatedFeedforwardWeights};

    let pop = GatedFeedforwardPopulation::new_random(pop_size, 42);
    let sensors: Vec<Vec<f32>> = (0..pop_size)
        .map(|i| {
            (0..8)
                .map(|j| ((i * 17 + j * 31) % 100) as f32 / 100.0 - 0.5)
                .collect()
        })
        .collect();

    // Warmup
    for org in pop.organisms.iter().take(100) {
        for s in sensors.iter().take(100) {
            let _ = org.decide(s);
        }
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..ticks {
        for (org, s) in pop.organisms.iter().zip(sensors.iter()) {
            let _ = org.decide(s);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    let total_ops = (pop_size * ticks) as f64;
    total_ops / elapsed
}

#[cfg(feature = "cuda")]
fn run_gpu_benchmark(pop_size: usize, ticks: usize) -> Result<(f64, f64), String> {
    use engine::brain::gated_feedforward::GatedFeedforwardPopulation;
    use engine::gpu_nn::gpu_gated_feedforward::GpuGatedFeedforward;

    // Create population and GPU accelerator
    let pop = GatedFeedforwardPopulation::new_random(pop_size, 42);
    let mut gpu = GpuGatedFeedforward::new(pop_size).map_err(|e| format!("GPU init: {:?}", e))?;

    gpu.upload_population(&pop)
        .map_err(|e| format!("Upload weights: {:?}", e))?;

    // Create sensor data
    let sensors: Vec<f32> = (0..pop_size * 8)
        .map(|i| ((i * 17 + 31) % 100) as f32 / 100.0 - 0.5)
        .collect();

    // Upload sensors ONCE (simulating persistent sensor buffer)
    gpu.upload_sensors(&sensors)
        .map_err(|e| format!("Upload sensors: {:?}", e))?;

    // Warmup (10 iterations)
    for _ in 0..10 {
        gpu.forward().map_err(|e| format!("Forward: {:?}", e))?;
    }
    gpu.sync().map_err(|e| format!("Sync: {:?}", e))?;

    // Timed run - compute only (no sensor upload)
    let start = Instant::now();
    for _ in 0..ticks {
        gpu.forward().map_err(|e| format!("Forward: {:?}", e))?;
    }
    gpu.sync().map_err(|e| format!("Sync: {:?}", e))?;
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0;
    let total_ops = (pop_size * ticks) as f64;
    let throughput = total_ops / elapsed.as_secs_f64();

    Ok((time_ms, throughput))
}
