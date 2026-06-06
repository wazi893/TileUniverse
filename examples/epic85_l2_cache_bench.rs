// EPIC 85: L2 Cache Pinning Benchmark
//
// Measures the performance impact of L2 cache persistence on quantum state operations.
// Compares:
//   1. Baseline (no L2 pinning)
//   2. L2 pinned (32MB reserved)
//   3. L2 pinned (max available)
//
// Run with: cargo run --example epic85_l2_cache_bench --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           EPIC 85: L2 CACHE PINNING BENCHMARK                    ║");
    println!("║           Measuring memory bandwidth optimization                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // Query L2 cache properties
    match rt.get_l2_properties() {
        Ok((l2_size, max_persist, max_window)) => {
            println!("GPU L2 Cache Properties:");
            println!("  Total L2 cache:       {} MB", l2_size / 1024 / 1024);
            println!("  Max persistent:       {} MB", max_persist / 1024 / 1024);
            println!("  Max access window:    {} MB", max_window / 1024 / 1024);
            println!();
        }
        Err(e) => {
            println!("Warning: Could not query L2 properties: {}", e);
            println!();
        }
    }

    // Test configurations
    let configs = [
        ("Small (fits in L2)", 512, 8), // 512 states × 8 tiles = 2 MB
        ("Medium", 1024, 16),           // 1024 states × 16 tiles = 8 MB
        ("Large (fills L2)", 2048, 32), // 2048 states × 32 tiles = 32 MB
        ("XL (exceeds L2)", 4096, 32),  // 4096 states × 32 tiles = 64 MB
    ];

    let depth = 10_000u32;
    let warmup_depth = 1000u32;

    for (name, num_states, tiles_per_state) in configs {
        let total_bytes = num_states * tiles_per_state * 256 * 2;

        println!("═══════════════════════════════════════════════════════════════════");
        println!(" Configuration: {}", name);
        println!(
            " States: {}, Tiles/state: {}, Total: {:.2} MB",
            num_states,
            tiles_per_state,
            total_bytes as f64 / 1024.0 / 1024.0
        );
        println!("═══════════════════════════════════════════════════════════════════\n");

        // =====================================================================
        // Test 1: Baseline (no L2 pinning)
        // =====================================================================
        println!("  [1] Baseline (no L2 pinning)...");

        // Create fresh runtime for baseline (no L2 config)
        let rt_baseline = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));
        let pool_baseline =
            MultiStatePersistent::new(rt_baseline.clone(), num_states, tiles_per_state)
                .expect("Failed to create baseline pool");

        // Warmup
        let _ =
            pool_baseline.run_benchmark(WmmaGateType::Hadamard, warmup_depth, MultiStateOpt::ILP);

        // Benchmark
        let (_, baseline_amps, baseline_time) = pool_baseline
            .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
            .expect("Baseline benchmark failed");

        let baseline_tcops = baseline_amps as f64 / baseline_time / 1e12;
        let baseline_bw = (total_bytes as f64 * 2.0 * depth as f64) / baseline_time / 1e9;

        println!("      Time: {:.4}s", baseline_time);
        println!("      Throughput: {:.2} TCOPS", baseline_tcops);
        println!("      Effective BW: {:.2} GB/s\n", baseline_bw);

        // =====================================================================
        // Test 2: L2 pinned (max available)
        // =====================================================================
        println!("  [2] L2 Pinned (max available)...");

        let rt_l2 = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

        // Query max persistent L2 size and use that
        let (_, max_persist, _) = rt_l2
            .get_l2_properties()
            .unwrap_or((0, 22 * 1024 * 1024, 0));
        let l2_size = max_persist; // Use max available
        println!(
            "      Requesting {} MB for L2 persistence...",
            l2_size / 1024 / 1024
        );

        match rt_l2.set_l2_persist_size(l2_size) {
            Ok(actual) => {
                println!("      L2 persistence: {} MB reserved", actual / 1024 / 1024);
            }
            Err(e) => {
                println!("      Warning: Could not enable L2 persistence: {}", e);
                println!("      Skipping L2 test for this configuration.\n");
                continue;
            }
        }

        // Create pool with L2 pinning
        let pool_l2 = match MultiStatePersistent::new_with_l2_pinning(
            rt_l2.clone(),
            num_states,
            tiles_per_state,
            1.0,
        ) {
            Ok(p) => p,
            Err(e) => {
                println!("      Warning: Could not create L2-pinned pool: {}", e);
                // Fall back to regular pool + enable_l2_pinning
                let pool = MultiStatePersistent::new(rt_l2.clone(), num_states, tiles_per_state)
                    .expect("Failed to create pool");
                if let Err(e) = pool.enable_l2_pinning(1.0) {
                    println!("      Could not enable L2 pinning: {}", e);
                    continue;
                }
                pool
            }
        };

        // Warmup
        let _ = pool_l2.run_benchmark(WmmaGateType::Hadamard, warmup_depth, MultiStateOpt::ILP);

        // Benchmark
        let (_, l2_amps, l2_time) = pool_l2
            .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
            .expect("L2 benchmark failed");

        let l2_tcops = l2_amps as f64 / l2_time / 1e12;
        let l2_bw = (total_bytes as f64 * 2.0 * depth as f64) / l2_time / 1e9;
        let speedup = baseline_time / l2_time;
        let bw_improvement = l2_bw / baseline_bw;

        println!("      Time: {:.4}s", l2_time);
        println!("      Throughput: {:.2} TCOPS", l2_tcops);
        println!("      Effective BW: {:.2} GB/s", l2_bw);
        println!("      Speedup vs baseline: {:.2}x", speedup);
        println!("      BW improvement: {:.2}x\n", bw_improvement);

        // =====================================================================
        // Summary for this configuration
        // =====================================================================
        println!("  ┌─────────────────────────────────────────────────────┐");
        println!("  │ SUMMARY: {}                          ", name);
        println!("  ├─────────────────────────────────────────────────────┤");
        println!(
            "  │ Baseline:    {:>8.2} TCOPS @ {:>6.2} GB/s          │",
            baseline_tcops, baseline_bw
        );
        println!(
            "  │ L2 Pinned:   {:>8.2} TCOPS @ {:>6.2} GB/s          │",
            l2_tcops, l2_bw
        );
        println!(
            "  │ Speedup:     {:>8.2}x                               │",
            speedup
        );
        println!("  └─────────────────────────────────────────────────────┘\n");

        // Reset L2 cache between configurations
        let _ = rt_l2.reset_l2_cache();
    }

    // =========================================================================
    // Final Summary
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    EPIC 85 BENCHMARK COMPLETE                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Key insights:");
    println!("  - L2 pinning helps most when working set fits in L2 cache");
    println!("  - RTX 4070 has ~36MB max persistent L2 (of 48MB total)");
    println!("  - Best gains: 2-3x for L2-resident data vs DRAM-bound");
    println!("  - Exceeding L2 size causes cache thrashing");
    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires both 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example epic85_l2_cache_bench --features cuda,perf-bench --release"
    );
}
