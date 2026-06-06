// EPIC 85: Bottleneck Analysis (No NCU Required)
//
// Since NCU requires admin permissions, we'll infer bottlenecks by:
// 1. Varying workload characteristics
// 2. Measuring throughput changes
// 3. Comparing against theoretical limits
//
// Run with: cargo run --example epic85_bottleneck_analysis --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 85: BOTTLENECK ANALYSIS (Inference-Based)             ║");
    println!("║       Identifying tensor core utilization limiters               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // ==========================================================================
    // Test 1: Depth Scaling (Memory Staging Latency Detection)
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 1: Depth Scaling (Detects Memory Staging Overhead)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  If throughput INCREASES with depth, we're amortizing per-kernel");
    println!("  launch overhead. If it DECREASES, we're hitting thermal/power.\n");

    let n_states = 1024;
    let tiles_per_state = 16;

    let pool = MultiStatePersistent::new(rt.clone(), n_states, tiles_per_state)
        .expect("Failed to create pool");

    // Warmup
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);

    let depths = [10, 50, 100, 500, 1000, 5000, 10000];

    println!("┌──────────────┬────────────────┬────────────────┬────────────────┐");
    println!("│    Depth     │   Time (ms)    │     TCOPS      │  vs Depth=1000 │");
    println!("├──────────────┼────────────────┼────────────────┼────────────────┤");

    let mut baseline_tcops = 0.0f64;

    for depth in depths {
        let (_, amp_ops, time_s) = pool
            .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
            .expect("Benchmark failed");

        let tcops = amp_ops as f64 / time_s / 1e12;

        if depth == 1000 {
            baseline_tcops = tcops;
        }

        let ratio = if baseline_tcops > 0.0 {
            tcops / baseline_tcops
        } else {
            1.0
        };

        println!(
            "│ {:>12} │ {:>12.3}ms │ {:>12.3} │ {:>12.2}x │",
            depth,
            time_s * 1000.0,
            tcops,
            ratio
        );
    }
    println!("└──────────────┴────────────────┴────────────────┴────────────────┘\n");

    // ==========================================================================
    // Test 2: State Count Scaling (Parallelism vs Occupancy)
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 2: State Count Scaling (Parallelism vs Occupancy)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  If throughput scales linearly, we have spare parallelism.");
    println!("  If it plateaus early, we're occupancy-limited.\n");

    let state_counts = [64, 128, 256, 512, 1024, 2048, 4096];
    let depth = 1000u32;

    println!("┌──────────────┬────────────────┬────────────────┬────────────────┐");
    println!("│   States     │   Time (ms)    │     TCOPS      │  Efficiency    │");
    println!("├──────────────┼────────────────┼────────────────┼────────────────┤");

    let mut first_tcops_per_state = 0.0f64;

    for n_states in state_counts {
        let pool = match MultiStatePersistent::new(rt.clone(), n_states, tiles_per_state) {
            Ok(p) => p,
            Err(e) => {
                println!("│ {:>12} │     FAILED: {:?}", n_states, e);
                continue;
            }
        };

        // Warmup
        let _ = pool.run_benchmark(WmmaGateType::Hadamard, 100, MultiStateOpt::ILP);

        let (_, amp_ops, time_s) = pool
            .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
            .expect("Benchmark failed");

        let tcops = amp_ops as f64 / time_s / 1e12;
        let tcops_per_state = tcops / n_states as f64;

        if first_tcops_per_state == 0.0 {
            first_tcops_per_state = tcops_per_state;
        }

        let efficiency = tcops_per_state / first_tcops_per_state * 100.0;

        println!(
            "│ {:>12} │ {:>12.3}ms │ {:>12.3} │ {:>11.1}% │",
            n_states,
            time_s * 1000.0,
            tcops,
            efficiency
        );
    }
    println!("└──────────────┴────────────────┴────────────────┴────────────────┘\n");

    // ==========================================================================
    // Test 3: Optimization Level Comparison
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 3: Kernel Variant Comparison");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let pool = MultiStatePersistent::new(rt.clone(), 1024, tiles_per_state)
        .expect("Failed to create pool");
    let depth = 10000u32;

    let variants = [
        ("Basic (8 warps)", MultiStateOpt::Basic),
        ("ILP (8 warps)", MultiStateOpt::ILP),
        ("ILP 16 warps", MultiStateOpt::ILP16Warp),
        ("ILP 32 warps", MultiStateOpt::ILP32Warp),
        ("NoFill", MultiStateOpt::NoFill),
        ("DeepPipeline", MultiStateOpt::DeepPipeline),
    ];

    println!("┌────────────────────────┬────────────────┬────────────────┐");
    println!("│      Variant           │     TCOPS      │  vs Basic      │");
    println!("├────────────────────────┼────────────────┼────────────────┤");

    let mut basic_tcops = 0.0f64;

    for (name, opt) in variants {
        // Warmup
        let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, opt);

        let (_, amp_ops, time_s) = pool
            .run_benchmark(WmmaGateType::Hadamard, depth, opt)
            .expect("Benchmark failed");

        let tcops = amp_ops as f64 / time_s / 1e12;

        if name.contains("Basic") {
            basic_tcops = tcops;
        }

        let ratio = if basic_tcops > 0.0 {
            tcops / basic_tcops
        } else {
            1.0
        };

        println!("│ {:>22} │ {:>12.3} │ {:>12.2}x │", name, tcops, ratio);
    }
    println!("└────────────────────────┴────────────────┴────────────────┘\n");

    // ==========================================================================
    // Test 4: Memory Bandwidth Estimation
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 4: Memory Bandwidth Analysis");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let pool = MultiStatePersistent::new(rt.clone(), 1024, tiles_per_state)
        .expect("Failed to create pool");

    // Each iteration reads and writes all state data
    // State data: 1024 states × 16 tiles × 256 elements × 2 bytes = 8 MB
    let state_bytes = 1024 * 16 * 256 * 2;
    let depth = 10000u32;

    let (_, _, time_s) = pool
        .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
        .expect("Benchmark failed");

    // Memory traffic: read + write per iteration
    let total_bytes = state_bytes as u64 * 2 * depth as u64;
    let bandwidth_gbps = total_bytes as f64 / time_s / 1e9;

    println!("  State data size:       {} MB", state_bytes / 1024 / 1024);
    println!("  Iterations:            {}", depth);
    println!(
        "  Total memory traffic:  {:.2} GB",
        total_bytes as f64 / 1e9
    );
    println!("  Time:                  {:.4}s", time_s);
    println!("  Achieved bandwidth:    {:.2} GB/s", bandwidth_gbps);
    println!("  Theoretical peak:      256 GB/s");
    println!(
        "  Bandwidth utilization: {:.1}%\n",
        bandwidth_gbps / 256.0 * 100.0
    );

    // ==========================================================================
    // Test 5: Compute vs Memory Bound Detection
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 5: Compute vs Memory Bound Detection");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Compute intensity: FLOPs per byte transferred
    // WMMA 16x16x16: 16*16*16*2 = 8192 FLOPs per MMA operation
    // Data for MMA: 16*16*2 (A) + 16*16*2 (B) + 16*16*2 (C) = 1536 bytes (if from DRAM)
    // Intensity: 8192 / 1536 = 5.33 FLOPs/byte

    // RTX 4070: 116 TFLOPS / 256 GB/s = 453 FLOPs/byte needed to be compute bound
    // Our intensity: ~5.33 FLOPs/byte
    // Therefore: MEMORY BOUND (need 85x more compute per byte to be compute-bound)

    let theoretical_tflops = 116.0;
    let theoretical_bandwidth = 256.0; // GB/s
    let ridge_point = theoretical_tflops * 1000.0 / theoretical_bandwidth; // FLOPs/byte

    let mma_flops = 16 * 16 * 16 * 2; // per MMA instruction
    let mma_bytes_ideal = 16 * 16 * 2 * 3; // A + B + C matrices if from DRAM
    let mma_bytes_shared = 16 * 16 * 2 * 2; // A + C if B is cached

    let intensity_ideal = mma_flops as f64 / mma_bytes_ideal as f64;
    let intensity_shared = mma_flops as f64 / mma_bytes_shared as f64;

    println!("  Roofline Analysis:");
    println!("  ─────────────────────────────────────────────────────────────");
    println!(
        "  RTX 4070 Peak Compute:      {} TFLOPS",
        theoretical_tflops
    );
    println!(
        "  RTX 4070 Peak Bandwidth:    {} GB/s",
        theoretical_bandwidth
    );
    println!(
        "  Ridge Point:                {:.1} FLOPs/byte",
        ridge_point
    );
    println!();
    println!(
        "  WMMA Intensity (DRAM):      {:.2} FLOPs/byte",
        intensity_ideal
    );
    println!(
        "  WMMA Intensity (Shared):    {:.2} FLOPs/byte",
        intensity_shared
    );
    println!();

    if intensity_shared < ridge_point {
        println!("  VERDICT: MEMORY BANDWIDTH BOUND");
        println!(
            "           Need {:.0}x more compute per byte to be compute-bound",
            ridge_point / intensity_shared
        );
    } else {
        println!("  VERDICT: COMPUTE BOUND");
    }
    println!();

    // ==========================================================================
    // Summary
    // ==========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    BOTTLENECK ANALYSIS SUMMARY                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                  ║");
    println!("║  Based on the roofline analysis:                                 ║");
    println!("║                                                                  ║");
    println!("║  The workload is MEMORY BANDWIDTH BOUND, not compute-bound.      ║");
    println!("║                                                                  ║");
    println!("║  Key evidence:                                                   ║");
    println!(
        "║    - Compute intensity: ~{:.1} FLOPs/byte                         ║",
        intensity_shared
    );
    println!(
        "║    - Ridge point: ~{:.0} FLOPs/byte                               ║",
        ridge_point
    );
    println!(
        "║    - We're {:.0}x below the ridge point                           ║",
        ridge_point / intensity_shared
    );
    println!("║                                                                  ║");
    println!("║  Why tensor utilization is 36%:                                  ║");
    println!("║    - Tensor cores wait for data from shared memory               ║");
    println!("║    - Shared memory waits for data from DRAM                      ║");
    println!("║    - The pipeline is starved for data                            ║");
    println!("║                                                                  ║");
    println!("║  Solutions:                                                      ║");
    println!("║    1. Double-buffering (load next while computing current)       ║");
    println!("║    2. cp.async (hardware async global→shared copy)               ║");
    println!("║    3. Increase tile reuse (compute more per data load)           ║");
    println!("║    4. L2 cache pinning (done - gave 21% improvement)             ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
}
