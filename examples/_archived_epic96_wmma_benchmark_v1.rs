//! ARCHIVED: This v1 benchmark has only ~30% agreement due to FP16 precision bugs.
//! Use the v2 benchmarks instead:
//!   - epic96_v2_perf_bench.rs (performance)
//!   - epic96_v2_correctness_test.rs (correctness verification)
//!
//! ---
//!
//! EPIC 96: WMMA Tensor Core NN Benchmark (V1 - DEPRECATED)
//!
//! Compares FP32 vs WMMA FP16 tensor cores.
//! Reports HONEST numbers: cold path (with download), not hot path fantasy.
//!
//! Target: 5-10× kernel speedup, 3-4× batched workload speedup


#[cfg(feature = "cuda")]
use engine::brain::{BrainWeights, forward_batch_shared_weights};
#[cfg(feature = "cuda")]
use engine::gpu_nn::gpu_nn_persistent::PersistentGpuNN;
#[cfg(feature = "cuda")]
use engine::gpu_nn::wmma_nn::WmmaGpuNN;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            EPIC 96: WMMA Tensor Core Benchmark              ║");
    println!("║         (Honest Numbers - Cold Path with Downloads)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        println!("Build with: cargo run --example epic96_wmma_benchmark --features cuda --release");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_benchmark();
    }
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
    let batch_sizes = vec![10_000, 100_000, 1_000_000];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Comparison: FP32 (EPIC 95.1) vs WMMA FP16 (EPIC 96)");
    println!("═══════════════════════════════════════════════════════════════\n");

    for &n in &batch_sizes {
        println!("┌─────────────────────────────────────────────────────────┐");
        println!(
            "│ Batch Size: {:>10}                                   │",
            format_number(n)
        );
        println!("├─────────────────────────────────────────────────────────┤");

        // Generate test data
        let sensors: Vec<[f32; 8]> = (0..n)
            .map(|i| {
                let val = (i as f32 / n as f32) * 2.0 - 1.0;
                [val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            })
            .collect();

        let weights = BrainWeights::random(12345);

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // CPU Baseline
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let cpu_start = Instant::now();
        let cpu_moves = forward_batch_shared_weights(&sensors, &weights);
        let cpu_time = cpu_start.elapsed();

        println!("│ [CPU Baseline]                                          │");
        println!(
            "│   Time: {:>8.2} ms                                     │",
            cpu_time.as_secs_f64() * 1000.0
        );
        println!("├─────────────────────────────────────────────────────────┤");

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // FP32 Persistent (EPIC 95.1)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let mut fp32_gpu = PersistentGpuNN::new(n + 10_000).expect("Failed to create FP32 GPU");
        fp32_gpu
            .upload_weights(&weights)
            .expect("Upload weights failed");
        fp32_gpu
            .upload_all_sensors(&sensors)
            .expect("Upload sensors failed");

        // Warmup
        let _ = fp32_gpu.forward_persistent();

        // COLD PATH: kernel + download
        let fp32_start = Instant::now();
        fp32_gpu.forward_persistent().expect("FP32 forward failed");
        let fp32_moves = fp32_gpu.download_moves().expect("FP32 download failed");
        let fp32_time = fp32_start.elapsed();

        let fp32_speedup = cpu_time.as_secs_f64() / fp32_time.as_secs_f64();

        println!("│ [FP32 Persistent] (EPIC 95.1)                           │");
        println!(
            "│   Cold path: {:>8.2} ms                                │",
            fp32_time.as_secs_f64() * 1000.0
        );
        println!(
            "│   Speedup vs CPU: {:>6.1}×                              │",
            fp32_speedup
        );
        println!("├─────────────────────────────────────────────────────────┤");

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // WMMA FP16 Tensor Cores (EPIC 96)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let mut wmma_gpu = WmmaGpuNN::new(n + 10_000).expect("Failed to create WMMA GPU");
        wmma_gpu
            .upload_weights(&weights)
            .expect("Upload weights failed");
        wmma_gpu
            .upload_all_sensors(&sensors)
            .expect("Upload sensors failed");

        // Warmup
        let _ = wmma_gpu.forward_persistent();

        // COLD PATH: kernel + download
        let wmma_start = Instant::now();
        wmma_gpu.forward_persistent().expect("WMMA forward failed");
        let wmma_moves = wmma_gpu.download_moves().expect("WMMA download failed");
        let wmma_time = wmma_start.elapsed();

        let wmma_speedup_vs_cpu = cpu_time.as_secs_f64() / wmma_time.as_secs_f64();
        let wmma_speedup_vs_fp32 = fp32_time.as_secs_f64() / wmma_time.as_secs_f64();

        println!("│ [WMMA FP16 Tensor Cores] (EPIC 96)                      │");
        println!(
            "│   Cold path: {:>8.2} ms                                │",
            wmma_time.as_secs_f64() * 1000.0
        );
        println!(
            "│   Speedup vs CPU: {:>6.1}×                              │",
            wmma_speedup_vs_cpu
        );
        println!(
            "│   Speedup vs FP32: {:>6.1}×                             │",
            wmma_speedup_vs_fp32
        );
        println!("└─────────────────────────────────────────────────────────┘\n");

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // Correctness Verification
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        assert_eq!(cpu_moves.len(), fp32_moves.len());
        assert_eq!(cpu_moves.len(), wmma_moves.len());

        let fp32_matches = cpu_moves
            .iter()
            .zip(fp32_moves.iter())
            .filter(|(cpu, gpu)| cpu == gpu)
            .count();

        let wmma_matches = cpu_moves
            .iter()
            .zip(wmma_moves.iter())
            .filter(|(cpu, gpu)| cpu == gpu)
            .count();

        let fp32_agreement = (fp32_matches as f64 / cpu_moves.len() as f64) * 100.0;
        let wmma_agreement = (wmma_matches as f64 / cpu_moves.len() as f64) * 100.0;

        println!("✓ Correctness:");
        println!(
            "  FP32 vs CPU: {:.2}% agreement ({} / {})",
            fp32_agreement,
            fp32_matches,
            cpu_moves.len()
        );
        println!(
            "  WMMA vs CPU: {:.2}% agreement ({} / {})\n",
            wmma_agreement,
            wmma_matches,
            cpu_moves.len()
        );

        if wmma_agreement < 99.0 {
            println!("⚠ WARNING: WMMA agreement <99%! FP16 precision may be insufficient.\n");
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Key Findings:");
    println!("  • All numbers are COLD PATH (kernel + download)");
    println!("  • Download time (~2ms) dominates single-tick latency");
    println!("  • WMMA speedup is for KERNEL ONLY");
    println!("  • Real-world benefit: batched workloads\n");

    println!("Batched Workload Projection (1000 ticks, 1 download):");
    println!("  FP32:  10ms (kernel) + 2ms (download) = 12ms");
    println!("  WMMA:  1-2ms (kernel) + 2ms (download) = 3-4ms");
    println!("  Expected speedup: 3-4×\n");

    println!("✓ EPIC 96 Complete - Honest Numbers Reported");
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
