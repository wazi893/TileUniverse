// RTX 5090 Benchmark - Push to maximum qubit counts
// Tests 12-32 qubits with PureMMA kernel (optimal for Blackwell SM 100)

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType, get_device_arch_string,
    };
    use std::sync::Arc;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  RTX 5090 MAXIMUM QUBIT BENCHMARK (PureMMA Optimized)          ║");
    println!("║  32GB VRAM - Pushing the Limits!                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let arch = get_device_arch_string();
    println!("  Detected architecture: {}\n", arch);

    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));

    // Test configurations: (qubits, states, depth)
    // Scale down states as qubits increase to fit in VRAM
    // Memory per state = 2^qubits * 8 bytes (complex f32)
    let tests = vec![
        (12, 1024, 100_000), // 32 KB/state, 32 MB total
        (16, 256, 100_000),  // 512 KB/state, 128 MB total
        (20, 64, 100_000),   // 8 MB/state, 512 MB total
        (24, 16, 25_000),    // 128 MB/state, 2 GB total
        (26, 8, 10_000),     // 512 MB/state, 4 GB total
        (28, 2, 5_000),      // 2 GB/state, 4 GB total
        (30, 1, 2_000),      // 8 GB/state, 8 GB total
        (31, 1, 500),        // 16 GB/state, 16 GB total
        (32, 1, 100),        // 32 GB/state, 32 GB total - RTX 5090 MAXIMUM!
    ];

    // Results: (qubits, tcops_ilp, tcops_puremma, mem_mb, status)
    let mut results: Vec<(u8, f64, f64, f64, &str)> = Vec::new();

    for (qubits, num_states, depth) in tests {
        let tiles_per_state = 1usize << (qubits - 8); // 2^(qubits-8) tiles
        let amps = 1u64 << qubits;
        let mem_per_state_mb = (amps * 8) as f64 / (1024.0 * 1024.0);
        let total_mem_mb = mem_per_state_mb * num_states as f64;

        println!("══════════════════════════════════════════════════════════════");
        println!(
            " {} QUBITS: {} amplitudes, {} states",
            qubits, amps, num_states
        );
        println!(
            " Memory: {:.1} MB/state × {} = {:.1} MB total",
            mem_per_state_mb, num_states, total_mem_mb
        );
        println!("══════════════════════════════════════════════════════════════\n");

        let persistent = match MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  ❌ Failed to allocate: {:?}\n", e);
                results.push((qubits, 0.0, 0.0, total_mem_mb, "FAILED"));
                continue;
            }
        };

        // Run ILP benchmark (baseline)
        let tcops_ilp =
            match persistent.run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP) {
                Ok((_, a, elapsed)) => {
                    let tcops = (a as f64 / elapsed) / 1e12;
                    println!("  ILP:     {:.2} TCOPS", tcops);
                    tcops
                }
                Err(e) => {
                    eprintln!("  ILP:     FAILED ({:?})", e);
                    0.0
                }
            };

        // Run PureMMA benchmark (optimized for Blackwell)
        let tcops_puremma =
            match persistent.run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::PureMMA) {
                Ok((_, a, elapsed)) => {
                    let tcops = (a as f64 / elapsed) / 1e12;
                    println!("  PureMMA: {:.2} TCOPS", tcops);
                    tcops
                }
                Err(e) => {
                    eprintln!("  PureMMA: FAILED ({:?})", e);
                    0.0
                }
            };

        let speedup = if tcops_ilp > 0.0 {
            tcops_puremma / tcops_ilp
        } else {
            0.0
        };
        println!("  Speedup: {:.2}x\n", speedup);

        let status = if tcops_puremma > 0.0 { "OK" } else { "FAILED" };
        results.push((qubits, tcops_ilp, tcops_puremma, total_mem_mb, status));
    }

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  RTX 5090 QUBIT SCALING RESULTS                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("┌─────────┬────────────┬───────────┬────────────┬─────────┬────────┐");
    println!("│ Qubits  │ VRAM Used  │  ILP TCOPS│PureMMA TCOPS│ Speedup │ Status │");
    println!("├─────────┼────────────┼───────────┼────────────┼─────────┼────────┤");

    for (qubits, tcops_ilp, tcops_puremma, mem_mb, status) in &results {
        let mem_str = if *mem_mb >= 1024.0 {
            format!("{:.1} GB", mem_mb / 1024.0)
        } else {
            format!("{:.0} MB", mem_mb)
        };

        let ilp_str = if *tcops_ilp > 0.0 {
            format!("{:.2}", tcops_ilp)
        } else {
            "---".to_string()
        };

        let puremma_str = if *tcops_puremma > 0.0 {
            format!("{:.2}", tcops_puremma)
        } else {
            "---".to_string()
        };

        let speedup_str = if *tcops_ilp > 0.0 && *tcops_puremma > 0.0 {
            format!("{:.2}x", tcops_puremma / tcops_ilp)
        } else {
            "---".to_string()
        };

        println!(
            "│   {:2}    │ {:>10} │ {:>9} │ {:>10} │ {:>7} │ {:>6} │",
            qubits, mem_str, ilp_str, puremma_str, speedup_str, status
        );
    }

    println!("└─────────┴────────────┴───────────┴────────────┴─────────┴────────┘");

    // Find peak PureMMA
    let peak = results
        .iter()
        .filter(|(_, _, tcops, _, status)| *status == "OK" && *tcops > 0.0)
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    if let Some((qubits, _, tcops, _, _)) = peak {
        println!(
            "\n  🚀 PEAK PERFORMANCE: {:.2} TCOPS at {} qubits (PureMMA)",
            tcops, qubits
        );
    }

    // Average speedup
    let speedups: Vec<f64> = results
        .iter()
        .filter(|(_, ilp, puremma, _, _)| *ilp > 0.0 && *puremma > 0.0)
        .map(|(_, ilp, puremma, _, _)| puremma / ilp)
        .collect();

    if !speedups.is_empty() {
        let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
        println!("  ⚡ Average PureMMA vs ILP speedup: {:.2}x", avg_speedup);
    }

    // Memory bandwidth calculation (corrected)
    // RTX 5090: 1792 GB/s theoretical
    // For WMMA operations: each tile = 256 amplitudes × 4 bytes = 1KB read + 1KB write
    let peak_tcops = results
        .iter()
        .map(|(_, _, t, _, _)| *t)
        .fold(0.0f64, f64::max);

    // Each amplitude requires ~4 bytes read + 4 bytes write (FP16 storage)
    let bytes_per_amp = 8.0; // FP16 complex = 4 bytes, read + write
    let achieved_bw = peak_tcops * 1e12 * bytes_per_amp / 1e9;

    println!("\n  📊 MEMORY BANDWIDTH ANALYSIS:");
    println!("     Theoretical: 1792 GB/s");
    println!(
        "     Achieved:    {:.0} GB/s ({:.1}% efficiency)",
        achieved_bw,
        achieved_bw / 1792.0 * 100.0
    );

    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Requires --features cuda,perf-bench");
}
