//! EPIC 114: FP8 Tensor Core Benchmark
//!
//! Benchmarks FP8 Tensor Core kernels against FP16 baseline.
//! FP8 provides 4x theoretical throughput (838 TFLOPS vs 209.5 TFLOPS on RTX 5090).
//!
//! Run: cargo run --release --example epic114_fp8_bench --features cuda

#[cfg(feature = "cuda")]
fn main() {
    use engine::cuda::{
        CudaRuntime, Fp8Opt, MultiStateOpt, WmmaGateType, analyze_fp8_gate_error,
        fp8_theoretical_throughput, get_hadamard_gate_host, run_fp8_multi_state,
        run_wmma_multi_state,
    };
    use std::time::Instant;

    println!("\n{}", "=".repeat(74));
    println!("  EPIC 114: FP8 TENSOR CORE BENCHMARK");
    println!("{}\n", "=".repeat(74));

    // Initialize CUDA
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {:?}", e);
            return;
        }
    };

    println!("CUDA initialized successfully\n");

    // =========================================================================
    // Section 1: Theoretical Performance
    // =========================================================================
    println!("{}", "-".repeat(74));
    println!("  THEORETICAL PERFORMANCE (RTX 5090 Specs)");
    println!("{}\n", "-".repeat(74));

    let (fp8_tflops, speedup) = fp8_theoretical_throughput();
    println!("  FP16 Tensor Core:  209.5 TFLOPS");
    println!("  FP8 Tensor Core:   {:.1} TFLOPS", fp8_tflops);
    println!("  Theoretical Speedup: {:.1}x\n", speedup);

    // =========================================================================
    // Section 2: FP8 Quantization Error Analysis
    // =========================================================================
    println!("{}", "-".repeat(74));
    println!("  FP8 QUANTIZATION ERROR ANALYSIS");
    println!("{}\n", "-".repeat(74));

    let hadamard_f16 = get_hadamard_gate_host();
    let (max_abs, mean_abs, max_rel) = analyze_fp8_gate_error(&hadamard_f16);

    println!("  Hadamard Gate (16x16) Quantization to FP8 E5M2:");
    println!("    Max absolute error:  {:.6}", max_abs);
    println!("    Mean absolute error: {:.6}", mean_abs);
    println!("    Max relative error:  {:.2}%\n", max_rel * 100.0);

    if max_rel < 0.01 {
        println!("  [OK] Quantization error < 1% - acceptable for quantum simulation\n");
    } else {
        println!("  [WARN] Quantization error > 1% - may affect fidelity\n");
    }

    // =========================================================================
    // Section 3: FP16 Baseline Benchmark
    // =========================================================================
    println!("{}", "-".repeat(74));
    println!("  FP16 BASELINE BENCHMARK");
    println!("{}\n", "-".repeat(74));

    let num_states = 1024;
    let tiles_per_state = 16;
    let depth = 100_000u32;

    println!("  Configuration:");
    println!("    States:          {}", num_states);
    println!(
        "    Tiles/state:     {} (= {} amplitudes)",
        tiles_per_state,
        tiles_per_state * 256
    );
    println!("    Depth:           {} gates", depth);
    println!();

    // FP16 ILP benchmark
    let start = Instant::now();
    let fp16_result = run_wmma_multi_state(
        &rt,
        num_states,
        tiles_per_state,
        WmmaGateType::Hadamard,
        depth,
        MultiStateOpt::ILP,
    );
    let fp16_elapsed = start.elapsed().as_secs_f64();

    let fp16_tcops = match &fp16_result {
        Ok((_, amp_ops)) => (*amp_ops as f64 / fp16_elapsed) / 1e12,
        Err(e) => {
            eprintln!("  FP16 benchmark failed: {:?}", e);
            0.0
        }
    };

    println!("  FP16 ILP Result:");
    println!("    Time:       {:.3} s", fp16_elapsed);
    println!("    Throughput: {:.2} TCOPS\n", fp16_tcops);

    // =========================================================================
    // Section 4: FP8 Benchmark Suite
    // =========================================================================
    println!("{}", "-".repeat(74));
    println!("  FP8 BENCHMARK SUITE");
    println!("{}\n", "-".repeat(74));

    let fp8_configs = [
        (Fp8Opt::Basic, "Basic (no renorm)"),
        (Fp8Opt::ILP, "ILP (4x unroll + shared mem)"),
        (Fp8Opt::Renorm, "With Renormalization"),
        (Fp8Opt::RenormILP, "RenormILP (ILP + warp shuffle)"),
        (Fp8Opt::Fused, "Fused Gate"),
        (Fp8Opt::PureMMA, "Pure MMA (compute only)"),
    ];

    let mut fp8_results: Vec<(Fp8Opt, f64, f64)> = Vec::new();

    for (opt, name) in &fp8_configs {
        print!("  Testing {:.<30}", format!("{} ", name));

        let start = Instant::now();
        let result = run_fp8_multi_state(&rt, num_states, tiles_per_state, depth, *opt);
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            Ok((_, amp_ops)) => {
                let tcops = (amp_ops as f64 / elapsed) / 1e12;
                println!(" {:.2} TCOPS ({:.3}s)", tcops, elapsed);
                fp8_results.push((*opt, tcops, elapsed));
            }
            Err(e) => {
                println!(" FAILED: {:?}", e);
            }
        }
    }

    // =========================================================================
    // Section 5: Comparison Summary
    // =========================================================================
    println!("\n{}", "-".repeat(74));
    println!("  PERFORMANCE COMPARISON");
    println!("{}\n", "-".repeat(74));

    println!(
        "  {:<25} {:>12} {:>12} {:>12}",
        "Kernel", "TCOPS", "vs FP16", "vs Theory"
    );
    println!("  {}", "-".repeat(61));

    println!(
        "  {:<25} {:>12.2} {:>12} {:>12.1}%",
        "FP16 ILP (baseline)",
        fp16_tcops,
        "1.00x",
        (fp16_tcops / 209.5) * 100.0
    );

    for (opt, tcops, _) in &fp8_results {
        let vs_fp16 = if fp16_tcops > 0.0 {
            tcops / fp16_tcops
        } else {
            0.0
        };
        let vs_theory = (tcops / fp8_tflops) * 100.0;
        let name = match opt {
            Fp8Opt::Basic => "FP8 Basic",
            Fp8Opt::ILP => "FP8 ILP",
            Fp8Opt::Renorm => "FP8 Renorm",
            Fp8Opt::RenormILP => "FP8 RenormILP",
            Fp8Opt::Fused => "FP8 Fused",
            Fp8Opt::PureMMA => "FP8 Pure MMA",
        };
        println!(
            "  {:<25} {:>12.2} {:>11.2}x {:>12.1}%",
            name, tcops, vs_fp16, vs_theory
        );
    }

    // =========================================================================
    // Section 6: Scaling Test
    // =========================================================================
    println!("\n{}", "-".repeat(74));
    println!("  DEPTH SCALING TEST (FP8 Fused)");
    println!("{}\n", "-".repeat(74));

    let depths = [10_000u32, 50_000, 100_000, 500_000, 1_000_000];

    println!(
        "  {:<12} {:>12} {:>12} {:>12}",
        "Depth", "Time (s)", "TCOPS", "Amp Ops"
    );
    println!("  {}", "-".repeat(52));

    for d in depths {
        let start = Instant::now();
        let result = run_fp8_multi_state(&rt, num_states, tiles_per_state, d, Fp8Opt::Fused);
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            Ok((_, amp_ops)) => {
                let tcops = (amp_ops as f64 / elapsed) / 1e12;
                println!(
                    "  {:<12} {:>12.3} {:>12.2} {:>12}",
                    format_with_suffix(d as u64),
                    elapsed,
                    tcops,
                    format_with_suffix(amp_ops)
                );
            }
            Err(e) => {
                println!("  {:<12} FAILED: {:?}", format_with_suffix(d as u64), e);
            }
        }
    }

    // =========================================================================
    // Section 7: State Count Scaling
    // =========================================================================
    println!("\n{}", "-".repeat(74));
    println!("  STATE COUNT SCALING TEST (FP8 Basic)");
    println!("{}\n", "-".repeat(74));

    let state_counts = [256, 512, 1024, 2048, 4096];
    let test_depth = 50_000u32;

    println!(
        "  {:<12} {:>12} {:>12} {:>12}",
        "States", "Time (s)", "TCOPS", "Memory (MB)"
    );
    println!("  {}", "-".repeat(52));

    for states in state_counts {
        let mem_mb = (states * tiles_per_state * 256 * 2) as f64 / 1e6; // 2 bytes per FP16

        let start = Instant::now();
        let result = run_fp8_multi_state(&rt, states, tiles_per_state, test_depth, Fp8Opt::Basic);
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            Ok((_, amp_ops)) => {
                let tcops = (amp_ops as f64 / elapsed) / 1e12;
                println!(
                    "  {:<12} {:>12.3} {:>12.2} {:>12.1}",
                    states, elapsed, tcops, mem_mb
                );
            }
            Err(e) => {
                println!("  {:<12} FAILED: {:?}", states, e);
            }
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(74));
    println!("  SUMMARY");
    println!("{}\n", "=".repeat(74));

    let best_fp8 = fp8_results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((opt, tcops, _)) = best_fp8 {
        let speedup_vs_fp16 = if fp16_tcops > 0.0 {
            tcops / fp16_tcops
        } else {
            0.0
        };

        println!("  Best FP8 kernel: {:?}", opt);
        println!("  Best FP8 throughput: {:.2} TCOPS", tcops);
        println!("  Speedup vs FP16: {:.2}x", speedup_vs_fp16);
        println!(
            "  Theoretical FP8 utilization: {:.1}%",
            (tcops / fp8_tflops) * 100.0
        );

        if speedup_vs_fp16 >= 2.0 {
            println!("\n  [SUCCESS] FP8 provides significant speedup over FP16!");
        } else if speedup_vs_fp16 >= 1.5 {
            println!("\n  [GOOD] FP8 provides moderate speedup over FP16");
        } else {
            println!("\n  [NOTE] FP8 speedup limited - likely memory-bound");
            println!("         Consider gate fusion to increase arithmetic intensity");
        }
    }

    println!("\n{}\n", "=".repeat(74));
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --release --example epic114_fp8_bench --features cuda");
    std::process::exit(1);
}

fn format_with_suffix(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
