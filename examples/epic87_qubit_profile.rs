//! EPIC 87 Phase 87.1: Per-Qubit Performance Profiling
//!
//! This benchmark measures GPU vs CPU performance for gates on different qubits.
//! The goal is to determine if qubits 4-7 can benefit from GPU acceleration despite
//! worse memory access patterns.
//!
//! Key insight: Qubits 0-3 fit naturally in 16x16 WMMA tiles (stride 1-8).
//! Qubits 4+ have stride ≥16, meaning pairs of interacting amplitudes span
//! multiple tiles.
//!
//! Decision gate criteria:
//! - Qubit 4 GPU < 5× slower than Qubit 0 GPU → Proceed
//! - Qubit 4 GPU faster than CPU → Proceed
//! - Qubit 4 GPU > 5× slower than Qubit 0 → Abort GPU path
//! - Qubit 4 GPU > 20% slower than CPU → Abort GPU path
//!
//! Run with: cargo run --example epic87_qubit_profile --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 87 PHASE 87.1: PER-QUBIT PERFORMANCE PROFILING        ║");
    println!("║       Measuring GPU vs CPU for qubits 0-7                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // Configuration
    let n_qubits = 12u8; // 4096 amplitudes = 16 tiles of 256
    let cpu_iterations = 1000;
    let warmup = 100;

    println!("Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!(
        "  State size: {} amplitudes ({} tiles)",
        1 << n_qubits,
        (1 << n_qubits) / 256
    );
    println!("  CPU iterations: {}", cpu_iterations);
    println!("  Warmup: {}\n", warmup);

    // =========================================================================
    // TEST 1: CPU Baseline for ALL qubits (0-7)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 1: CPU Scalar Performance (Baseline)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let cpu_times = measure_cpu_performance(n_qubits, cpu_iterations, warmup);

    println!("┌─────────┬──────────────┬─────────────┬──────────────┐");
    println!("│  Qubit  │  Time (μs)   │   Stride    │  vs Qubit 0  │");
    println!("├─────────┼──────────────┼─────────────┼──────────────┤");

    let cpu_q0_time = cpu_times[0];
    for (q, &time) in cpu_times.iter().enumerate() {
        let stride = 1usize << q;
        let ratio = time / cpu_q0_time;
        println!(
            "│    {}    │ {:>10.2}   │ {:>9}   │ {:>10.2}×  │",
            q,
            time * 1e6,
            stride,
            ratio
        );
    }
    println!("└─────────┴──────────────┴─────────────┴──────────────┘\n");

    // =========================================================================
    // TEST 2: GPU WMMA Performance - High-level benchmark using MultiStatePersistent
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 2: GPU WMMA Baseline Performance");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Use the well-tested MultiStatePersistent API
    let n_states = 16; // Match 16 tiles for 12-qubit state
    let tiles_per_state = 16;
    let gpu_depth = 1000u32;

    let pool = MultiStatePersistent::new(rt.clone(), n_states, tiles_per_state)
        .expect("Failed to create pool");

    // Warmup
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 100, MultiStateOpt::ILP);

    // Measure GPU baseline (H⊗4 on all 4 tile qubits)
    let (_, amp_ops, time_s) = pool
        .run_benchmark(WmmaGateType::Hadamard, gpu_depth, MultiStateOpt::ILP)
        .expect("GPU benchmark failed");

    let tcops = amp_ops as f64 / time_s / 1e12;
    let gpu_time_per_op = time_s / gpu_depth as f64;

    println!("  GPU H⊗4 (all 4 qubits in tile): {:.3} TCOPS", tcops);
    println!(
        "  Time per gate application: {:.3} μs",
        gpu_time_per_op * 1e6
    );
    println!(
        "  Total ops: {}, Time: {:.3} ms\n",
        amp_ops,
        time_s * 1000.0
    );

    // =========================================================================
    // TEST 3: Memory Access Pattern Analysis
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 3: Memory Access Pattern Analysis");
    println!("═══════════════════════════════════════════════════════════════════\n");

    analyze_memory_patterns(n_qubits);

    // =========================================================================
    // TEST 4: Per-Qubit GPU Timing Analysis
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 4: Per-Qubit Analysis - Theoretical Projections");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  Current state: WMMA supports qubits 0-3 in a single 16x16 tile.");
    println!("  For qubits 4+, pairs span multiple tiles requiring gather/scatter.\n");

    // Calculate theoretical projections based on existing measurements
    let gpu_base_time = gpu_time_per_op; // Time for H⊗4 application

    // For single-qubit gates, the GPU time should be similar since we're still
    // doing a 16x16 matrix multiply, just with different structure
    // The gather/scatter overhead for qubits 4+ is the key question

    println!("┌─────────┬──────────────┬──────────────┬──────────────┬───────────────┐");
    println!("│  Qubit  │  CPU (μs)    │  GPU Est.    │  Speedup     │  WMMA Status  │");
    println!("├─────────┼──────────────┼──────────────┼──────────────┼───────────────┤");

    for q in 0..8 {
        let cpu_time = cpu_times[q];

        // Estimate GPU time based on qubit position
        let (gpu_est, status) = if q <= 3 {
            // Qubits 0-3: Direct WMMA support
            // For single-qubit, ~same as H⊗4 (16x16 multiply)
            (gpu_base_time, "✓ Direct")
        } else {
            // Qubits 4+: Would need gather/scatter
            // Estimate: base_time + 2 memory passes (gather + scatter)
            // Each memory pass ~= 2x base time for small data
            let gather_scatter_overhead = gpu_base_time * 2.0;
            (gpu_base_time + gather_scatter_overhead, "? Gather/Scatter")
        };

        let speedup = cpu_time / gpu_est;

        println!(
            "│    {}    │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │ {:>13} │",
            q,
            cpu_time * 1e6,
            gpu_est * 1e6,
            speedup,
            status
        );
    }
    println!("└─────────┴──────────────┴──────────────┴──────────────┴───────────────┘\n");

    // =========================================================================
    // TEST 5: Actual WMMA Per-Qubit Test (Qubits 0-3)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 5: Actual WMMA Per-Qubit Measurement (Qubits 0-3)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Test actual single-qubit WMMA performance using run_wmma_batched_gates
    let actual_gpu_times = measure_actual_wmma_performance(&rt, n_qubits);

    println!("┌─────────┬──────────────┬──────────────┬──────────────┐");
    println!("│  Qubit  │  GPU (μs)    │  CPU (μs)    │  GPU/CPU     │");
    println!("├─────────┼──────────────┼──────────────┼──────────────┤");

    for q in 0..4 {
        let gpu_time = actual_gpu_times[q];
        let cpu_time = cpu_times[q];
        let ratio = gpu_time / cpu_time;
        let status = if ratio < 1.0 {
            "✓ GPU wins"
        } else {
            "✗ CPU wins"
        };

        println!(
            "│    {}    │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │  {}",
            q,
            gpu_time * 1e6,
            cpu_time * 1e6,
            ratio,
            status
        );
    }
    println!("└─────────┴──────────────┴──────────────┴──────────────┘\n");

    // =========================================================================
    // FINAL SUMMARY AND DECISION
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" FINAL SUMMARY: EPIC 87 GO/NO-GO DECISION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    print_decision_summary(&cpu_times, &actual_gpu_times, gpu_base_time);
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn measure_cpu_performance(n_qubits: u8, iterations: usize, warmup: usize) -> Vec<f64> {
    use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};

    let mut times = Vec::with_capacity(8);

    for target_qubit in 0u8..8 {
        if target_qubit >= n_qubits {
            times.push(0.0);
            continue;
        }

        let mut state = QState::new_zero(n_qubits);
        let mut rng = QRng { state: 12345 };
        let gate = QGate::H(target_qubit);

        // Warmup
        for _ in 0..warmup {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }
        let elapsed = start.elapsed().as_secs_f64();

        times.push(elapsed / iterations as f64);
    }

    times
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn analyze_memory_patterns(n_qubits: u8) {
    println!(
        "  For a {}-qubit state ({} amplitudes):\n",
        n_qubits,
        1 << n_qubits
    );

    println!("  │ Qubit │ Stride │ Elements per 16x16 tile │ WMMA Compatible │");
    println!("  ├───────┼────────┼─────────────────────────┼─────────────────┤");

    for q in 0..8 {
        let stride = 1usize << q;
        let (pairs_info, compatible) = if stride <= 8 {
            (format!("{} pairs in tile", 128 / stride), "✓ Yes (direct)")
        } else {
            ("Spans multiple tiles".to_string(), "✗ No (needs G/S)")
        };

        println!(
            "  │   {}   │ {:>6} │ {:>23} │ {:>15} │",
            q, stride, pairs_info, compatible
        );
    }
    println!("  └───────┴────────┴─────────────────────────┴─────────────────┘\n");

    println!("  Legend:");
    println!("    G/S = Gather/Scatter (data reorganization required)");
    println!("    Stride = Distance between paired amplitudes\n");
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn measure_actual_wmma_performance(
    rt: &std::sync::Arc<engine::cuda::CudaRuntime>,
    n_qubits: u8,
) -> Vec<f64> {
    use engine::cuda::{WmmaState, run_wmma_batched_gates};
    use engine::quantum::QGate;

    let num_tiles = (1usize << n_qubits) / 256;
    let iterations = 100; // Fewer iterations for direct API
    let warmup = 20;

    let mut times = Vec::with_capacity(4);

    for target_qubit in 0u8..4 {
        // Create host data - |0⟩ state for each tile
        let host_data: Vec<half::f16> = (0..num_tiles * 256)
            .map(|i| {
                if i % 256 == 0 {
                    half::f16::ONE
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        let mut state = WmmaState::from_host(rt, &host_data).expect("Failed to create WMMA state");

        // Create gate sequence - 10 H gates on this qubit
        let gates: Vec<QGate> = (0..10).map(|_| QGate::H(target_qubit)).collect();

        // Warmup
        for _ in 0..warmup {
            let _ = run_wmma_batched_gates(rt, &mut state, &gates);
        }
        rt.synchronize().expect("Sync after warmup");

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = run_wmma_batched_gates(rt, &mut state, &gates);
        }
        rt.synchronize().expect("Sync after measurement");
        let elapsed = start.elapsed().as_secs_f64();

        // Time per single H gate = total / (iterations * 10 gates)
        times.push(elapsed / (iterations * 10) as f64);
    }

    times
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn print_decision_summary(cpu_times: &[f64], actual_gpu_times: &[f64], gpu_base_time: f64) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│                         DECISION MATRIX                                     │");
    println!("├─────────┬──────────────┬──────────────┬──────────────┬───────────────────────┤");
    println!("│  Qubit  │  GPU (μs)    │  CPU (μs)    │  Speedup     │  Recommendation       │");
    println!("├─────────┼──────────────┼──────────────┼──────────────┼───────────────────────┤");

    let mut max_profitable_qubit = 3; // At minimum, qubits 0-3 should work

    for q in 0..8 {
        let gpu_time = if q < 4 {
            actual_gpu_times[q]
        } else {
            // Estimate: base + 2x overhead for gather/scatter
            gpu_base_time * 3.0
        };

        let cpu_time = cpu_times[q];
        let speedup = cpu_time / gpu_time;

        let (status, prefix) = if q < 4 {
            if speedup > 1.0 {
                ("✓ WMMA (measured)", "")
            } else {
                ("✗ CPU faster", "")
            }
        } else {
            if speedup > 1.5 {
                max_profitable_qubit = q;
                ("✓ PROCEED (projected)", "*")
            } else if speedup > 1.0 {
                ("? MARGINAL (projected)", "*")
            } else {
                ("✗ ABORT (projected)", "*")
            }
        };

        println!(
            "│    {}{}   │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │ {:>21} │",
            q,
            prefix,
            gpu_time * 1e6,
            cpu_time * 1e6,
            speedup,
            status
        );
    }
    println!("└─────────┴──────────────┴──────────────┴──────────────┴───────────────────────┘");
    println!("  * = Projected (requires gather/scatter implementation)\n");

    // Calculate key metrics for decision
    let q0_gpu = actual_gpu_times[0];
    let q0_cpu = cpu_times[0];
    let q4_cpu = cpu_times[4];
    let q4_gpu_projected = gpu_base_time * 3.0; // Base + 2x overhead

    let q0_speedup = q0_cpu / q0_gpu;
    let q4_projected_speedup = q4_cpu / q4_gpu_projected;

    // Decision criteria from EPIC 87 v2
    let q4_vs_q0 = q4_gpu_projected / q0_gpu;
    let q4_faster_than_cpu = q4_projected_speedup > 1.0;
    let q4_within_5x_q0 = q4_vs_q0 < 5.0;

    println!("  ═══════════════════════════════════════════════════════════════════");
    println!("  DECISION CRITERIA (from EPIC 87 v2):");
    println!("  ───────────────────────────────────────────────────────────────────");
    println!(
        "  1. Qubit 0 GPU speedup: {:.2}× (target: >1.0×)",
        q0_speedup
    );
    println!(
        "  2. Qubit 4 projected speedup: {:.2}× (target: >1.0×)",
        q4_projected_speedup
    );
    println!(
        "  3. Qubit 4 vs Qubit 0 GPU: {:.2}× (must be <5.0×)",
        q4_vs_q0
    );
    println!(
        "  4. Qubit 4 faster than CPU: {} (required)",
        if q4_faster_than_cpu {
            "✓ Yes"
        } else {
            "✗ No"
        }
    );
    println!("  ═══════════════════════════════════════════════════════════════════\n");

    // Final decision
    println!("  ╔═══════════════════════════════════════════════════════════════════╗");
    if q4_faster_than_cpu && q4_within_5x_q0 {
        println!("  ║  DECISION: ✓ PROCEED TO PHASE 87.2                               ║");
        println!("  ║                                                                   ║");
        println!("  ║  Projected qubit 4 performance is viable.                        ║");
        println!(
            "  ║  Implement gather/scatter approach for qubits 4-{}.              ║",
            max_profitable_qubit
        );
        println!("  ║                                                                   ║");
        println!("  ║  Next steps:                                                      ║");
        println!("  ║  1. Implement gather kernel (GPU)                                 ║");
        println!("  ║  2. Implement scatter kernel (GPU)                                ║");
        println!("  ║  3. Measure actual overhead vs projections                        ║");
    } else if !q4_faster_than_cpu {
        println!("  ║  DECISION: ✗ ABORT EPIC 87 GPU PATH                               ║");
        println!("  ║                                                                   ║");
        println!("  ║  Qubit 4 projected GPU time is SLOWER than CPU.                  ║");
        println!("  ║  Gather/scatter overhead too high to compensate.                 ║");
        println!("  ║  Keep qubits 4+ on CPU fallback path.                            ║");
    } else {
        println!("  ║  DECISION: ⚠ INVESTIGATE FURTHER                                  ║");
        println!("  ║                                                                   ║");
        println!("  ║  Qubit 4 GPU is >5× slower than Qubit 0.                         ║");
        println!("  ║  This suggests severe memory bandwidth bottleneck.               ║");
        println!("  ║  Consider alternative approaches before implementing.            ║");
    }
    println!("  ╚═══════════════════════════════════════════════════════════════════╝\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires --features cuda,perf-bench");
    println!(
        "Run with: cargo run --example epic87_qubit_profile --features cuda,perf-bench --release"
    );
}
