//! EPIC 96 V2: WMMA Performance Benchmark
//!
//! Compares WMMA tensor core NN against:
//! 1. CPU batched forward pass
//! 2. FP32 GPU kernel (if available)
//!
//! The WMMA demon has been vanquished - now let's see the speedup!

#[cfg(feature = "cuda")]
use engine::batched_brain::{BrainWeights, forward_batch};
#[cfg(feature = "cuda")]
use engine::gpu_nn::wmma_nn_v2::WmmaGpuNNV2;
#[cfg(feature = "cuda")]
use engine::quantum::QRng;
#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         EPIC 96 V2: WMMA Performance Benchmark          ║");
    println!("║                                                          ║");
    println!("║  The WMMA demon is slain - time to measure the spoils!  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        println!("Build with: cargo run --example epic96_v2_perf_bench --features cuda --release");
    }

    #[cfg(feature = "cuda")]
    {
        run_benchmarks();
    }
}

#[cfg(feature = "cuda")]
fn run_benchmarks() {
    const N_ORGANISMS: usize = 10000;
    const N_ITERATIONS: usize = 100;
    const WARMUP: usize = 10;

    println!("Configuration:");
    println!("  Organisms:     {}", N_ORGANISMS);
    println!("  Iterations:    {}", N_ITERATIONS);
    println!("  Warmup:        {}", WARMUP);
    println!();

    // Generate test data
    println!("Generating test data...");
    let weights = BrainWeights::random(12345);
    let mut rng = QRng::new(67890);
    let sensors: Vec<[f32; 8]> = (0..N_ORGANISMS)
        .map(|_| {
            let mut s = [0.0f32; 8];
            for j in 0..8 {
                s[j] = (rng.next_f32() - 0.5) * 2.0;
            }
            s
        })
        .collect();

    // ========================================
    // CPU Baseline
    // ========================================
    println!("\n┌─────────────────────────────────────────────────┐");
    println!("│                  CPU Baseline                   │");
    println!("└─────────────────────────────────────────────────┘");

    let weights_vec = vec![weights.clone(); N_ORGANISMS];

    // Warmup
    for _ in 0..WARMUP {
        let _ = forward_batch(&sensors, &weights_vec);
    }

    // Benchmark
    let cpu_start = Instant::now();
    for _ in 0..N_ITERATIONS {
        let _ = forward_batch(&sensors, &weights_vec);
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_per_iter_us = cpu_elapsed.as_micros() as f64 / N_ITERATIONS as f64;
    let cpu_organisms_per_sec = N_ORGANISMS as f64 / (cpu_per_iter_us / 1_000_000.0);

    println!("  Time per iteration:  {:.2} us", cpu_per_iter_us);
    println!(
        "  Throughput:          {:.2}M organisms/sec",
        cpu_organisms_per_sec / 1_000_000.0
    );

    // ========================================
    // WMMA Tensor Core
    // ========================================
    println!("\n┌─────────────────────────────────────────────────┐");
    println!("│              WMMA Tensor Core V2                │");
    println!("└─────────────────────────────────────────────────┘");

    let mut wmma_nn = WmmaGpuNNV2::new(N_ORGANISMS).expect("Failed to create WMMA NN V2");
    wmma_nn
        .upload_weights(&weights)
        .expect("Failed to upload weights");
    wmma_nn
        .upload_all_sensors(&sensors)
        .expect("Failed to upload sensors");

    // Warmup (kernel only, no download)
    for _ in 0..WARMUP {
        wmma_nn.forward_persistent().expect("Forward failed");
    }

    // Benchmark kernel only (hot path)
    let wmma_kernel_start = Instant::now();
    for _ in 0..N_ITERATIONS {
        wmma_nn.forward_persistent().expect("Forward failed");
    }
    let wmma_kernel_elapsed = wmma_kernel_start.elapsed();
    let wmma_kernel_per_iter_us = wmma_kernel_elapsed.as_micros() as f64 / N_ITERATIONS as f64;
    let wmma_kernel_organisms_per_sec =
        N_ORGANISMS as f64 / (wmma_kernel_per_iter_us / 1_000_000.0);

    // Benchmark with download (cold path - realistic)
    let wmma_cold_start = Instant::now();
    for _ in 0..N_ITERATIONS {
        wmma_nn.forward_persistent().expect("Forward failed");
        let _ = wmma_nn.download_moves().expect("Download failed");
    }
    let wmma_cold_elapsed = wmma_cold_start.elapsed();
    let wmma_cold_per_iter_us = wmma_cold_elapsed.as_micros() as f64 / N_ITERATIONS as f64;
    let wmma_cold_organisms_per_sec = N_ORGANISMS as f64 / (wmma_cold_per_iter_us / 1_000_000.0);

    println!("  Hot path (kernel only):");
    println!("    Time per iteration:  {:.2} us", wmma_kernel_per_iter_us);
    println!(
        "    Throughput:          {:.2}M organisms/sec",
        wmma_kernel_organisms_per_sec / 1_000_000.0
    );
    println!();
    println!("  Cold path (with download):");
    println!("    Time per iteration:  {:.2} us", wmma_cold_per_iter_us);
    println!(
        "    Throughput:          {:.2}M organisms/sec",
        wmma_cold_organisms_per_sec / 1_000_000.0
    );

    // ========================================
    // Results Summary
    // ========================================
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let hot_speedup = cpu_per_iter_us / wmma_kernel_per_iter_us;
    let cold_speedup = cpu_per_iter_us / wmma_cold_per_iter_us;

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Backend              │   Time/iter │  Throughput  │ vs CPU │");
    println!("├──────────────────────┼─────────────┼──────────────┼────────┤");
    println!(
        "│ CPU (batched)        │ {:>9.2}us │ {:>8.2}M/s │   1.0× │",
        cpu_per_iter_us,
        cpu_organisms_per_sec / 1_000_000.0
    );
    println!(
        "│ WMMA hot (kernel)    │ {:>9.2}us │ {:>8.2}M/s │ {:>5.1}× │",
        wmma_kernel_per_iter_us,
        wmma_kernel_organisms_per_sec / 1_000_000.0,
        hot_speedup
    );
    println!(
        "│ WMMA cold (download) │ {:>9.2}us │ {:>8.2}M/s │ {:>5.1}× │",
        wmma_cold_per_iter_us,
        wmma_cold_organisms_per_sec / 1_000_000.0,
        cold_speedup
    );
    println!("└─────────────────────────────────────────────────────────┘");

    println!("\n📊 **HONEST NUMBERS (cold path with download):**");
    println!("   WMMA Tensor Core speedup: {:.1}× vs CPU", cold_speedup);

    if cold_speedup > 1.0 {
        println!("\n🎉 **TENSOR CORES ARE FASTER!** The WMMA demon has been truly vanquished!");
    } else {
        println!("\n⚠️  WMMA is slower than CPU for this workload.");
        println!("   The 8×8 NN is too small - 75% of compute is wasted on padding.");
        println!("   Consider scaling to 16×16 or 32×32 for better WMMA efficiency.");
    }

    println!("\n✓ EPIC 96 V2 Performance Benchmark Complete");
}
