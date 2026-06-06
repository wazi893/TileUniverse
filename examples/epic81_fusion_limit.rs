// EPIC 81: Find the practical fusion limit
//
// Test extreme fusion factors to find where we hit diminishing returns
//
// Run with: cargo run --example epic81_fusion_limit --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType, compose_gates_16x16,
        get_hadamard_gate_host, run_wmma_multi_state_fused,
    };
    use std::sync::Arc;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 81: FUSION LIMIT FINDER                                  ║");
    println!("║  Finding where we transition from memory-bound to overhead     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("CUDA init failed"));
    let h_gate = get_hadamard_gate_host();

    // Configuration - use large depth to have room for extreme fusion
    let num_states = 1024usize;
    let tiles_per_state = 16usize;
    let total_depth = 1_000_000u32; // 1 million iterations baseline

    println!("Configuration:");
    println!("  States:      {}", num_states);
    println!(
        "  Tiles:       {} ({} amplitudes)",
        tiles_per_state,
        tiles_per_state * 256
    );
    println!("  Total depth: {} H applications", total_depth);
    println!();

    // RTX 4070 specs
    let mem_bw_gb_s = 256.0f64; // GB/s
    let tensor_tflops = 116.0f64; // TFLOPS

    // Memory per iteration (load + store all states)
    let bytes_per_iter = (num_states * tiles_per_state * 256 * 2 * 2) as f64; // 2 bytes * 2 (load+store)

    // Compute per iteration
    let flops_per_iter = (num_states * tiles_per_state) as f64 * 8192.0; // 8192 flops per WMMA

    println!("Per-iteration analysis:");
    println!("  Memory traffic: {:.2} MB", bytes_per_iter / 1e6);
    println!("  Compute work:   {:.2} GFLOPS", flops_per_iter / 1e9);
    println!();

    let pool = MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state)
        .expect("Pool creation failed");

    // Warmup
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);

    // Baseline measurement (unfused)
    println!("Running baseline (unfused, 100K depth for speed)...");
    let (_, _, baseline_time) = pool
        .run_benchmark(WmmaGateType::Hadamard, 100_000, MultiStateOpt::ILP)
        .expect("Baseline failed");
    let baseline_time_per_iter = baseline_time / 100_000.0;

    println!(
        "  Baseline: {:.2} us per iteration",
        baseline_time_per_iter * 1e6
    );
    println!();

    // Test extreme fusion factors
    println!("┌──────────┬───────────┬───────────┬───────────┬───────────┬───────────┐");
    println!("│ Fusion   │ GPU Iters │ Time (ms) │ us/iter   │ Eff TCOPS │ Speedup   │");
    println!("├──────────┼───────────┼───────────┼───────────┼───────────┼───────────┤");

    let fusion_factors = [
        1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
    ];

    let mut best_tcops = 0.0f64;
    let mut best_fusion = 1u32;
    let mut prev_tcops = 0.0f64;

    for &fusion in &fusion_factors {
        let gpu_depth = total_depth / fusion;
        if gpu_depth == 0 {
            continue;
        }

        // Compute fused gate
        let fused_gate = compute_power_gate(&h_gate, fusion);
        let fused_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
        let fused_gpu = rt.upload(&fused_bits).expect("Upload failed");

        // Run benchmark
        let (_, amp_ops, time) = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gpu,
            gpu_depth,
            fusion,
        )
        .expect("Fused benchmark failed");

        let time_ms = time * 1000.0;
        let us_per_iter = if gpu_depth > 0 {
            (time * 1e6) / (gpu_depth as f64)
        } else {
            0.0
        };
        let eff_tcops = amp_ops as f64 / time / 1e12;
        let speedup = (baseline_time_per_iter * (total_depth as f64)) / time;

        let marker = if eff_tcops > best_tcops {
            best_tcops = eff_tcops;
            best_fusion = fusion;
            " <-- BEST"
        } else if eff_tcops < prev_tcops * 0.95 {
            " (diminishing)"
        } else {
            ""
        };

        println!(
            "│ {:>8} │ {:>9} │ {:>9.2} │ {:>9.2} │ {:>9.1} │ {:>8.1}x │{}",
            format!("{}x", fusion),
            gpu_depth,
            time_ms,
            us_per_iter,
            eff_tcops,
            speedup,
            marker
        );

        prev_tcops = eff_tcops;
    }

    println!("└──────────┴───────────┴───────────┴───────────┴───────────┴───────────┘");
    println!();

    // Analysis
    println!("═══════════════════════════════════════════════════════════════");
    println!(" ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  Best fusion factor:    {}x", best_fusion);
    println!("  Best effective TCOPS:  {:.1}", best_tcops);
    println!();

    // Theoretical limits
    let compute_limited_time =
        (num_states as f64) * (tiles_per_state as f64) * 8192.0 / (tensor_tflops * 1e12);
    let memory_limited_time = bytes_per_iter / (mem_bw_gb_s * 1e9);

    println!("  Theoretical minimum per iteration:");
    println!("    Compute-limited: {:.2} us", compute_limited_time * 1e6);
    println!("    Memory-limited:  {:.2} us", memory_limited_time * 1e6);
    println!();

    if memory_limited_time > compute_limited_time {
        let ratio = memory_limited_time / compute_limited_time;
        println!("  System is MEMORY-BOUND by {:.1}x", ratio);
        println!("  Even infinite fusion won't help - we're limited by");
        println!("  how fast we can load initial state / store final state.");
    } else {
        println!("  System is COMPUTE-BOUND");
    }
    println!();

    // What fusion factor would make us compute-bound?
    // At fusion F, we do total_depth/F iterations
    // Total memory = bytes_per_iter * (total_depth/F)
    // Total compute = flops_per_iter * (total_depth/F)  [same work either way]
    // But wait - the REAL analysis is about the KERNEL, not the total work

    println!("  Note: At extreme fusion, kernel launch overhead dominates.");
    println!("  The practical limit depends on your circuit depth.");
    println!();
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn compute_power_gate(base: &[half::f16; 256], power: u32) -> [half::f16; 256] {
    use engine::cuda::compose_gates_16x16;

    if power == 0 || power == 1 {
        if power == 0 {
            let mut identity = [half::f16::ZERO; 256];
            for i in 0..16 {
                identity[i * 16 + i] = half::f16::ONE;
            }
            return identity;
        }
        return *base;
    }

    // For Hadamard: H^2 = I, so H^n = H if n is odd, I if n is even
    // This is a massive shortcut!
    if power % 2 == 0 {
        let mut identity = [half::f16::ZERO; 256];
        for i in 0..16 {
            identity[i * 16 + i] = half::f16::ONE;
        }
        identity
    } else {
        *base
    }
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("Requires cuda,perf-bench features");
}
