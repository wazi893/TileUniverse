// EPIC 81 Part 2: Gate Fusion Benchmark
//
// Tests the impact of gate fusion on tensor core performance.
// Gate fusion computes G^N on CPU, reducing GPU depth iterations.
//
// Run with: cargo run --example epic81_fusion_bench --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType, compose_gates_16x16,
        get_hadamard_gate_host, run_wmma_multi_state_fused,
    };
    use std::time::Instant;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 81 Part 2: GATE FUSION BENCHMARK                         ║");
    println!("║  Testing tensor core utilization with REAL fused gates         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_qubits: u8 = 12;
    let num_states: usize = 1024;
    let tiles_per_state: usize = (1usize << n_qubits) / 256;
    let total_logical_depth: u32 = 100_000; // Total H applications we want

    println!("Configuration:");
    println!("  Qubits:              {}", n_qubits);
    println!("  Parallel states:     {}", num_states);
    println!("  Tiles per state:     {}", tiles_per_state);
    println!(
        "  Total logical depth: {} H applications",
        total_logical_depth
    );
    println!();

    // Create CUDA runtime
    let rt = std::sync::Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // Create persistent state pool
    let pool = MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state)
        .expect("Failed to create state pool");

    println!("═══════════════════════════════════════════════════════════════");
    println!(
        " BASELINE COMPARISON: ILP vs PureMMA (depth = {})",
        total_logical_depth
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    // Warmup both
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::PureMMA);

    // Baseline: ILP (old default)
    let (gate_ops, amplitude_ops, kernel_time_ilp) = pool
        .run_benchmark(
            WmmaGateType::Hadamard,
            total_logical_depth,
            MultiStateOpt::ILP,
        )
        .expect("ILP benchmark failed");
    let tcops_ilp = amplitude_ops as f64 / kernel_time_ilp / 1e12;

    // Baseline: PureMMA (new optimized)
    let (_, amplitude_ops_pm, kernel_time_puremma) = pool
        .run_benchmark(
            WmmaGateType::Hadamard,
            total_logical_depth,
            MultiStateOpt::PureMMA,
        )
        .expect("PureMMA benchmark failed");
    let tcops_puremma = amplitude_ops_pm as f64 / kernel_time_puremma / 1e12;

    // Use PureMMA as the baseline for fusion comparison
    let kernel_time = kernel_time_puremma;
    let tcops_baseline = tcops_puremma;

    println!(
        "  ILP Kernel time:     {:.3} s  ({:.2} TCOPS)",
        kernel_time_ilp, tcops_ilp
    );
    println!(
        "  PureMMA Kernel time: {:.3} s  ({:.2} TCOPS) <- baseline",
        kernel_time_puremma, tcops_puremma
    );
    println!(
        "  PureMMA speedup:     {:.2}x",
        kernel_time_ilp / kernel_time_puremma
    );
    println!();
    println!("  Gate operations:  {}", gate_ops);
    println!("  Amplitude ops:    {}", amplitude_ops);
    println!();

    // Test various fusion levels with REAL fused gates
    println!("═══════════════════════════════════════════════════════════════");
    println!(" FUSION TESTS (Real H^N computation)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Get the base Hadamard gate
    let h_gate = get_hadamard_gate_host();

    // Fusion factors to test
    let fusion_factors = [2u32, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100, 128];

    println!("┌─────────┬───────────┬───────────┬───────────┬───────────┬───────────┐");
    println!("│ Fusion  │ GPU Depth │ Time (s)  │ Raw TCOPS │ Speedup   │ Eff TCOPS │");
    println!("├─────────┼───────────┼───────────┼───────────┼───────────┼───────────┤");

    let mut best_speedup = 1.0f64;
    let mut best_fusion = 1u32;
    let mut best_eff_tcops = tcops_baseline;

    for &fusion in &fusion_factors {
        // Calculate effective depth on GPU
        let gpu_depth = total_logical_depth / fusion;

        if gpu_depth == 0 {
            println!(
                "│ {:>7} │ {:>9} │ Skipped (depth=0)                           │",
                format!("{}x", fusion),
                gpu_depth
            );
            continue;
        }

        // Compute H^fusion on CPU
        let fused_gate = compute_power_gate(&h_gate, fusion);

        // Upload fused gate to GPU
        let fused_gate_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
        let fused_gate_gpu = rt
            .upload(&fused_gate_bits)
            .expect("Failed to upload fused gate");

        // Warmup
        let _ = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gate_gpu,
            100,
            fusion,
        );

        // Run fused benchmark
        let (_, amp_ops_fused, kernel_time_fused) = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gate_gpu,
            gpu_depth,
            fusion,
        )
        .expect("Fused benchmark failed");

        // Calculate metrics
        // Raw TCOPS = actual GPU work done
        let raw_tcops = (num_states as f64) * (gpu_depth as f64) * (tiles_per_state * 256) as f64
            / kernel_time_fused
            / 1e12;
        // Effective TCOPS = counts the fusion benefit (equivalent unfused work)
        let effective_tcops = amp_ops_fused as f64 / kernel_time_fused / 1e12;
        let speedup = kernel_time / kernel_time_fused;

        if speedup > best_speedup {
            best_speedup = speedup;
            best_fusion = fusion;
            best_eff_tcops = effective_tcops;
        }

        println!(
            "│ {:>7} │ {:>9} │ {:>9.4} │ {:>9.2} │ {:>9.2}x│ {:>9.2} │",
            format!("{}x", fusion),
            gpu_depth,
            kernel_time_fused,
            raw_tcops,
            speedup,
            effective_tcops
        );
    }

    println!("└─────────┴───────────┴───────────┴───────────┴───────────┴───────────┘");
    println!();

    // Analysis
    println!("═══════════════════════════════════════════════════════════════");
    println!(" ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  Baseline TCOPS:        {:.2}", tcops_baseline);
    println!("  Best fusion factor:    {}x", best_fusion);
    println!("  Best speedup:          {:.2}x", best_speedup);
    println!("  Best effective TCOPS:  {:.2}", best_eff_tcops);
    println!();

    // Memory bandwidth analysis
    let mem_per_iter_bytes = (num_states * tiles_per_state * 256 * 2) as f64; // 2 bytes per half
    let mem_baseline = mem_per_iter_bytes * 2.0 * (total_logical_depth as f64); // load + store per iter
    let mem_fused = mem_per_iter_bytes * 2.0 * ((total_logical_depth / best_fusion) as f64);

    println!(
        "  Memory traffic (baseline):    {:.2} GB",
        mem_baseline / 1e9
    );
    println!(
        "  Memory traffic ({}x fused): {:.2} GB",
        best_fusion,
        mem_fused / 1e9
    );
    println!(
        "  Memory reduction:             {:.1}x",
        mem_baseline / mem_fused
    );
    println!();

    // Theoretical analysis (RTX 5090)
    let tensor_tflops = 419.0; // RTX 5090 FP16 Tensor
    let mem_bw = 1792.0; // GB/s RTX 5090

    // For the fused case, are we compute or memory bound?
    let compute_time = (num_states as f64)
        * ((total_logical_depth / best_fusion) as f64)
        * (tiles_per_state * 256) as f64
        * 8192.0
        / 256.0
        / (tensor_tflops * 1e12);
    let memory_time = mem_fused / (mem_bw * 1e9);

    println!("  Theoretical compute time:     {:.4} s", compute_time);
    println!("  Theoretical memory time:      {:.4} s", memory_time);

    if memory_time > compute_time {
        println!("  Bottleneck:                   MEMORY BANDWIDTH");
    } else {
        println!("  Bottleneck:                   COMPUTE");
    }
    println!();

    // Conclusion
    println!("═══════════════════════════════════════════════════════════════");
    println!(" CONCLUSION");
    println!("═══════════════════════════════════════════════════════════════\n");

    if best_speedup > 1.5 {
        println!("  ✓ Gate fusion provides significant speedup!");
        println!(
            "    {}x fusion achieves {:.2}x speedup by reducing GPU iterations.",
            best_fusion, best_speedup
        );
        println!();
        println!("  This confirms the bottleneck is shared memory load/store latency.");
        println!(
            "  By fusing {} gates into one, we do {}x fewer memory round-trips.",
            best_fusion, best_fusion
        );
    } else {
        println!("  ~ Gate fusion shows modest improvement.");
        println!("    The bottleneck may be elsewhere.");
    }
    println!();

    // Note about H^N
    println!("  Note: H^2 = I (identity), so even-power fused gates are trivial.");
    println!("  Odd-power fused gates equal H. This benchmark tests the iteration");
    println!("  reduction benefit, not gate complexity.");
    println!();
}

/// Compute G^N (gate to the Nth power) via repeated multiplication
#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn compute_power_gate(base: &[half::f16; 256], power: u32) -> [half::f16; 256] {
    use engine::cuda::compose_gates_16x16;

    if power == 0 {
        // G^0 = Identity
        let mut identity = [half::f16::ZERO; 256];
        for i in 0..16 {
            identity[i * 16 + i] = half::f16::ONE;
        }
        return identity;
    }

    if power == 1 {
        return *base;
    }

    // Binary exponentiation for efficiency
    let mut result = [half::f16::ZERO; 256];
    for i in 0..16 {
        result[i * 16 + i] = half::f16::ONE; // Start with identity
    }

    let mut base_power = *base;
    let mut n = power;

    while n > 0 {
        if n % 2 == 1 {
            result = compose_gates_16x16(&result, &base_power);
        }
        base_power = compose_gates_16x16(&base_power, &base_power);
        n /= 2;
    }

    result
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires both 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example epic81_fusion_bench --features cuda,perf-bench --release"
    );
}
