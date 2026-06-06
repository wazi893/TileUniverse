// Blackwell (SM 100) Tuning Benchmark
// Tests all kernel optimization variants on RTX 5090 to find optimal configuration

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType, get_device_arch_string,
    };
    use std::sync::Arc;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  BLACKWELL (SM 100) KERNEL OPTIMIZATION SWEEP                  ║");
    println!("║  Testing all MultiStateOpt variants for RTX 5090               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let arch = get_device_arch_string();
    println!("  Detected architecture: {}", arch);

    // Verify we're on Blackwell
    if !arch.contains("12") && !arch.contains("10") {
        println!("  ⚠ Not running on RTX 5090 (compute_120), results may differ");
    }

    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));

    // Test L2 cache sizes - RTX 5090 has ~100MB L2, much larger than Ada's 48MB
    let l2_sizes_mb = [0, 32, 64, 96];

    // All optimization variants to test
    let opts = [
        (MultiStateOpt::Basic, "Basic"),
        (MultiStateOpt::ILP, "ILP"),
        (MultiStateOpt::ILP16Warp, "ILP16Warp"),
        (MultiStateOpt::ILP32Warp, "ILP32Warp"),
        (MultiStateOpt::NoFill, "NoFill"),
        (MultiStateOpt::DeepPipeline, "DeepPipeline"),
        (MultiStateOpt::Interleaved, "Interleaved"),
        (MultiStateOpt::Unroll8x, "Unroll8x"),
        (MultiStateOpt::Unroll8x16Warp, "Unroll8x16Warp"),
        (MultiStateOpt::PureMMA, "PureMMA"),
        (MultiStateOpt::Swizzled, "Swizzled"),
    ];

    // Test configurations: (qubits, states, depth)
    let configs = [
        (12, 1024, 50_000, "12q - L2 friendly"),
        (16, 256, 50_000, "16q - Mixed"),
        (20, 64, 20_000, "20q - Memory bound"),
        (24, 16, 10_000, "24q - Large state"),
    ];

    let mut all_results: Vec<(String, u8, f64)> = Vec::new();

    // First, test without L2 persistence
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(" PHASE 1: KERNEL VARIANT SWEEP (no L2 persistence)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for (qubits, num_states, depth, desc) in &configs {
        let tiles_per_state = 1usize << (qubits - 8);

        println!("──────────────────────────────────────────────────────────────────");
        println!(
            " {} qubits, {} states, depth {} ({})",
            qubits, num_states, depth, desc
        );
        println!("──────────────────────────────────────────────────────────────────\n");

        let persistent = match MultiStatePersistent::new(rt.clone(), *num_states, tiles_per_state) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  ❌ Failed to allocate: {:?}", e);
                continue;
            }
        };

        let mut best_opt = "";
        let mut best_tcops = 0.0f64;

        println!(
            "  {:16} | {:>10} | {:>10} | {:>8}",
            "Variant", "TCOPS", "Gates/s", "Warps"
        );
        println!(
            "  {:16} | {:>10} | {:>10} | {:>8}",
            "─".repeat(16),
            "─".repeat(10),
            "─".repeat(10),
            "─".repeat(8)
        );

        for (opt, name) in &opts {
            let warps = match opt {
                MultiStateOpt::ILP16Warp | MultiStateOpt::Unroll8x16Warp => 16,
                MultiStateOpt::ILP32Warp => 32,
                _ => 8,
            };

            match persistent.run_benchmark(WmmaGateType::Hadamard, *depth, *opt) {
                Ok((g, a, elapsed)) => {
                    let tcops = (a as f64 / elapsed) / 1e12;
                    let gates_m = g as f64 / elapsed / 1e6;

                    let marker = if tcops > best_tcops { "★" } else { " " };
                    println!(
                        "  {:16} | {:>9.2}T | {:>9.1}M | {:>7}w {}",
                        name, tcops, gates_m, warps, marker
                    );

                    if tcops > best_tcops {
                        best_tcops = tcops;
                        best_opt = name;
                    }

                    all_results.push((name.to_string(), *qubits, tcops));
                }
                Err(e) => {
                    println!(
                        "  {:16} | {:>10} | {:>10} | {:>8}",
                        name, "FAILED", "-", warps
                    );
                    eprintln!("    Error: {:?}", e);
                }
            }
        }

        println!(
            "\n  ★ Best for {} qubits: {} @ {:.2} TCOPS\n",
            qubits, best_opt, best_tcops
        );
    }

    // Phase 2: L2 cache persistence testing
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(" PHASE 2: L2 CACHE PERSISTENCE IMPACT");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Use the best-performing variants for L2 testing
    let l2_test_opts = [
        (MultiStateOpt::ILP, "ILP"),
        (MultiStateOpt::ILP16Warp, "ILP16Warp"),
        (MultiStateOpt::Unroll8x16Warp, "Unroll8x16Warp"),
    ];

    // Test at 12 qubits where L2 should have biggest impact
    let (qubits, num_states, depth) = (12, 1024, 50_000);
    let tiles_per_state = 1usize << (qubits - 8);

    println!(
        "  Testing at {} qubits, {} states (L2-friendly workload)\n",
        qubits, num_states
    );
    println!(
        "  {:16} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Variant", "0 MB", "32 MB", "64 MB", "96 MB"
    );
    println!(
        "  {:16} | {:>10} | {:>10} | {:>10} | {:>10}",
        "─".repeat(16),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10)
    );

    for (opt, name) in &l2_test_opts {
        let mut row = format!("  {:16} |", name);

        for l2_mb in l2_sizes_mb {
            // Set L2 persistence
            let l2_bytes = l2_mb * 1024 * 1024;
            if l2_bytes > 0 {
                match rt.set_l2_persist_size(l2_bytes) {
                    Ok(_) => {}
                    Err(e) => {
                        row.push_str(&format!(" {:>9} |", "ERR"));
                        eprintln!("    L2 {} MB error: {:?}", l2_mb, e);
                        continue;
                    }
                }
            }

            let persistent =
                match MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state) {
                    Ok(p) => p,
                    Err(_) => {
                        row.push_str(&format!(" {:>9} |", "ALLOC"));
                        continue;
                    }
                };

            match persistent.run_benchmark(WmmaGateType::Hadamard, depth, *opt) {
                Ok((_, a, elapsed)) => {
                    let tcops = (a as f64 / elapsed) / 1e12;
                    row.push_str(&format!(" {:>8.2}T |", tcops));
                }
                Err(_) => {
                    row.push_str(&format!(" {:>9} |", "FAIL"));
                }
            }

            // Reset L2 persistence
            let _ = rt.set_l2_persist_size(0);
        }

        println!("{}", row);
    }

    // Phase 3: High warp count testing for SM 100
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(" PHASE 3: BLACKWELL OCCUPANCY ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  RTX 5090 (SM 100) specifications:");
    println!("    - 192 SMs (doubled from Ada's 76)");
    println!("    - 2048 threads/SM max");
    println!("    - 64 warps/SM max");
    println!("    - ~100 MB L2 cache");
    println!("    - 1792 GB/s memory bandwidth\n");

    // Compare warp scaling
    let warp_tests = [
        (MultiStateOpt::ILP, "8 warps (ILP)"),
        (MultiStateOpt::ILP16Warp, "16 warps (ILP16Warp)"),
        (MultiStateOpt::ILP32Warp, "32 warps (ILP32Warp)"),
    ];

    println!("  Warp scaling comparison at 20 qubits:\n");
    println!(
        "  {:24} | {:>10} | {:>12} | {:>10}",
        "Config", "TCOPS", "Occupancy", "Efficiency"
    );
    println!(
        "  {:24} | {:>10} | {:>12} | {:>10}",
        "─".repeat(24),
        "─".repeat(10),
        "─".repeat(12),
        "─".repeat(10)
    );

    let (qubits, num_states, depth) = (20, 64, 20_000);
    let tiles_per_state = 1usize << (qubits - 8);

    let persistent = match MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  ❌ Failed to allocate: {:?}", e);
            return;
        }
    };

    let mut base_tcops = 0.0f64;
    for (opt, name) in &warp_tests {
        let warps_per_block: u32 = match opt {
            MultiStateOpt::ILP16Warp => 16,
            MultiStateOpt::ILP32Warp => 32,
            _ => 8,
        };

        // Estimate occupancy: warps_per_block / 64 (max warps per SM)
        let occupancy = (warps_per_block as f64 / 64.0) * 100.0;

        match persistent.run_benchmark(WmmaGateType::Hadamard, depth, *opt) {
            Ok((_, a, elapsed)) => {
                let tcops = (a as f64 / elapsed) / 1e12;
                if base_tcops == 0.0 {
                    base_tcops = tcops;
                }
                let efficiency = (tcops / base_tcops) * 100.0;
                println!(
                    "  {:24} | {:>8.2}T | {:>10.1}% | {:>9.1}%",
                    name, tcops, occupancy, efficiency
                );
            }
            Err(e) => {
                println!(
                    "  {:24} | {:>10} | {:>12} | {:>10}",
                    name, "FAILED", "-", "-"
                );
                eprintln!("    Error: {:?}", e);
            }
        }
    }

    // Summary
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  BLACKWELL TUNING SUMMARY                                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Find best per qubit count
    let qubit_counts: Vec<u8> = all_results.iter().map(|(_, q, _)| *q).collect();
    let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();

    println!("  Best kernel variant per qubit count:\n");
    println!("  {:8} | {:20} | {:>10}", "Qubits", "Best Variant", "TCOPS");
    println!(
        "  {:8} | {:20} | {:>10}",
        "─".repeat(8),
        "─".repeat(20),
        "─".repeat(10)
    );

    for q in qubit_counts {
        if seen.contains(&q) {
            continue;
        }
        seen.insert(q);

        let best = all_results
            .iter()
            .filter(|(_, qb, _)| *qb == q)
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        if let Some((name, _, tcops)) = best {
            println!("  {:>8} | {:20} | {:>8.2}T", q, name, tcops);
        }
    }

    // Find overall peak
    if let Some((name, qubits, tcops)) = all_results
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    {
        println!(
            "\n  🚀 PEAK PERFORMANCE: {:.2} TCOPS using {} at {} qubits",
            tcops, name, qubits
        );
    }

    // Memory bandwidth analysis
    let peak_tcops = all_results
        .iter()
        .map(|(_, _, t)| *t)
        .fold(0.0f64, f64::max);
    let bytes_per_amp_op = 32.0; // read + write complex f32
    let achieved_bw = peak_tcops * 1e12 * bytes_per_amp_op / 1e9;
    let theoretical_bw = 1792.0; // RTX 5090 theoretical

    println!("\n  📊 MEMORY BANDWIDTH UTILIZATION:");
    println!("     Theoretical: {:.0} GB/s", theoretical_bw);
    println!(
        "     Achieved:    {:.0} GB/s ({:.1}%)",
        achieved_bw,
        achieved_bw / theoretical_bw * 100.0
    );

    // Recommendations
    println!("\n  📋 RECOMMENDATIONS FOR SM 100:");
    if achieved_bw / theoretical_bw < 0.25 {
        println!("     - Consider higher warp counts (ILP16Warp or ILP32Warp)");
        println!("     - Enable L2 persistence for L2-friendly workloads");
        println!("     - Test with larger state batches for better occupancy");
    } else {
        println!("     - Memory bandwidth is well utilized");
        println!("     - Focus on reducing kernel launch overhead for deep circuits");
    }

    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Requires --features cuda,perf-bench");
}
