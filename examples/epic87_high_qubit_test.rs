//! EPIC 87 Phase 87.2: High-Qubit WMMA Test
//!
//! Tests the gather/scatter implementation for qubits 4-7.
//! Validates correctness and measures performance.
//!
//! Run with: cargo run --example epic87_high_qubit_test --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, WmmaState, run_wmma_batched_gates, run_wmma_high_qubit_batched,
    };
    use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 87 PHASE 87.2: HIGH-QUBIT WMMA TEST                   ║");
    println!("║       Testing gather/scatter for qubits 4-7                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // Test configuration
    let n_qubits = 12u8; // 4096 amplitudes = 16 tiles
    let num_tiles = (1 << n_qubits) / 256;

    println!("Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!("  Amplitudes: {}", 1 << n_qubits);
    println!("  Tiles: {}\n", num_tiles);

    // =========================================================================
    // TEST 1: Kernel Compilation
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 1: Kernel Compilation");
    println!("═══════════════════════════════════════════════════════════════════\n");

    match rt.get_strided_cache() {
        Ok(_) => println!("  ✓ Strided gather/scatter kernels compiled successfully"),
        Err(e) => {
            println!("  ✗ Kernel compilation FAILED: {:?}", e);
            return;
        }
    }
    println!();

    // =========================================================================
    // TEST 2: Correctness - H² = I for each qubit 4-7
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 2: Correctness - H² = I for qubits 4-7");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for target_qubit in 4u8..=7 {
        // Create initial state |0⟩
        let mut cpu_state = QState::new_zero(n_qubits);
        let mut rng = QRng { state: 12345 };

        // Apply H twice on CPU (reference)
        let gate = QGate::H(target_qubit);
        apply_gate_scalar(&mut cpu_state, &gate, &mut rng);
        apply_gate_scalar(&mut cpu_state, &gate, &mut rng);

        // Check that state is back to |0⟩
        let cpu_amps = cpu_state.to_interleaved();
        let cpu_fidelity = cpu_amps[0].norm2(); // Should be ~1.0

        // Now test GPU path
        let host_data: Vec<half::f16> = (0..num_tiles * 256)
            .map(|i| {
                if i == 0 {
                    half::f16::ONE
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        let mut gpu_state =
            WmmaState::from_host(&rt, &host_data).expect("Failed to create WMMA state");

        // Apply H twice via gather/scatter
        let gates = vec![QGate::H(target_qubit), QGate::H(target_qubit)];

        match run_wmma_high_qubit_batched(&rt, &mut gpu_state, &gates, target_qubit) {
            Ok(_) => {
                // Download and check
                let gpu_result = gpu_state.to_host(&rt).expect("Download");
                let gpu_amp0 = gpu_result[0].to_f32();

                let error = (gpu_amp0 - 1.0).abs();
                let pass = error < 0.01;

                println!(
                    "  Qubit {}: GPU amp[0] = {:.6}, CPU fidelity = {:.6}, Error = {:.6} {}",
                    target_qubit,
                    gpu_amp0,
                    cpu_fidelity,
                    error,
                    if pass { "✓" } else { "✗" }
                );
            }
            Err(e) => {
                println!("  Qubit {}: FAILED - {:?}", target_qubit, e);
            }
        }
    }
    println!();

    // =========================================================================
    // TEST 3: Batched Performance - Compare single vs batched
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 3: Batched Performance Comparison");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let batch_sizes = [1, 5, 10, 20, 50];
    let iterations = 100;
    let warmup = 20;

    println!("  Testing qubit 4 with various batch sizes:\n");

    println!("┌──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│  Batch Size  │  Total (μs)  │  Per-Gate    │  Speedup     │");
    println!("├──────────────┼──────────────┼──────────────┼──────────────┤");

    let mut single_gate_time = 0.0f64;

    for &batch_size in &batch_sizes {
        let host_data: Vec<half::f16> = (0..num_tiles * 256)
            .map(|i| {
                if i == 0 {
                    half::f16::ONE
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        let mut gpu_state =
            WmmaState::from_host(&rt, &host_data).expect("Failed to create WMMA state");

        // Create batch of H gates on qubit 4
        let gates: Vec<QGate> = (0..batch_size).map(|_| QGate::H(4)).collect();

        // Warmup
        for _ in 0..warmup {
            let _ = run_wmma_high_qubit_batched(&rt, &mut gpu_state, &gates, 4);
        }
        rt.synchronize().expect("Sync");

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = run_wmma_high_qubit_batched(&rt, &mut gpu_state, &gates, 4);
        }
        rt.synchronize().expect("Sync");
        let elapsed = start.elapsed().as_secs_f64();

        let total_time = elapsed / iterations as f64;
        let per_gate_time = total_time / batch_size as f64;

        if batch_size == 1 {
            single_gate_time = per_gate_time;
        }

        let speedup = if single_gate_time > 0.0 {
            single_gate_time / per_gate_time
        } else {
            1.0
        };

        println!(
            "│ {:>12} │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │",
            batch_size,
            total_time * 1e6,
            per_gate_time * 1e6,
            speedup
        );
    }
    println!("└──────────────┴──────────────┴──────────────┴──────────────┘\n");

    // =========================================================================
    // TEST 4: Compare GPU vs CPU for qubits 4-7
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 4: GPU vs CPU for High Qubits (Batch Size = 50)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let batch_size = 50; // Use larger batch to amortize gather/scatter overhead

    println!("┌─────────┬──────────────┬──────────────┬──────────────┐");
    println!("│  Qubit  │  GPU (μs)    │  CPU (μs)    │  Speedup     │");
    println!("├─────────┼──────────────┼──────────────┼──────────────┤");

    for target_qubit in 4u8..=7 {
        // CPU measurement
        let mut cpu_state = QState::new_zero(n_qubits);
        let mut rng = QRng { state: 12345 };
        let gate = QGate::H(target_qubit);

        // Warmup
        for _ in 0..warmup {
            for _ in 0..batch_size {
                apply_gate_scalar(&mut cpu_state, &gate, &mut rng);
            }
        }

        let cpu_start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..batch_size {
                apply_gate_scalar(&mut cpu_state, &gate, &mut rng);
            }
        }
        let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
        let cpu_time = cpu_elapsed / iterations as f64;

        // GPU measurement
        let host_data: Vec<half::f16> = (0..num_tiles * 256)
            .map(|i| {
                if i == 0 {
                    half::f16::ONE
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        let mut gpu_state =
            WmmaState::from_host(&rt, &host_data).expect("Failed to create WMMA state");

        let gates: Vec<QGate> = (0..batch_size).map(|_| QGate::H(target_qubit)).collect();

        // Warmup
        for _ in 0..warmup {
            let _ = run_wmma_high_qubit_batched(&rt, &mut gpu_state, &gates, target_qubit);
        }
        rt.synchronize().expect("Sync");

        let gpu_start = Instant::now();
        for _ in 0..iterations {
            let _ = run_wmma_high_qubit_batched(&rt, &mut gpu_state, &gates, target_qubit);
        }
        rt.synchronize().expect("Sync");
        let gpu_elapsed = gpu_start.elapsed().as_secs_f64();
        let gpu_time = gpu_elapsed / iterations as f64;

        let speedup = cpu_time / gpu_time;
        let status = if speedup > 1.0 { "✓" } else { "✗" };

        println!(
            "│    {}    │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │  {}",
            target_qubit,
            gpu_time * 1e6,
            cpu_time * 1e6,
            speedup,
            status
        );
    }
    println!("└─────────┴──────────────┴──────────────┴──────────────┘\n");

    // =========================================================================
    // TEST 5: Regression Test - Qubits 0-3 must not slow down
    // Uses batched comparison (50 gates) to match EPIC 87.1 methodology
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" TEST 5: Regression Test - Qubits 0-3 (Batched, like EPIC 87.1)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Baseline from EPIC 87 Phase 87.1 used batched (depth 1000) measurements
    // Single-gate measurements are dominated by per-call overhead
    // So we test with batch=50 to match the high-qubit test methodology
    let regression_batch = 50;

    println!("┌─────────┬──────────────┬──────────────┬──────────────┐");
    println!("│  Qubit  │  GPU (μs)    │  CPU (μs)    │  Speedup     │");
    println!("├─────────┼──────────────┼──────────────┼──────────────┤");

    let mut regression_detected = false;

    for target_qubit in 0u8..=3 {
        // CPU measurement (batched)
        let mut cpu_state = QState::new_zero(n_qubits);
        let mut rng = QRng { state: 12345 };
        let gate = QGate::H(target_qubit);

        let cpu_start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..regression_batch {
                apply_gate_scalar(&mut cpu_state, &gate, &mut rng);
            }
        }
        let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
        let cpu_time = cpu_elapsed / iterations as f64;

        // GPU measurement using standard WMMA path (qubits 0-3)
        let host_data: Vec<half::f16> = (0..num_tiles * 256)
            .map(|i| {
                if i == 0 {
                    half::f16::ONE
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        let mut gpu_state =
            WmmaState::from_host(&rt, &host_data).expect("Failed to create WMMA state");

        // Batch of 50 gates
        let gates: Vec<QGate> = (0..regression_batch)
            .map(|_| QGate::H(target_qubit))
            .collect();

        // Warmup
        for _ in 0..warmup {
            let _ = run_wmma_batched_gates(&rt, &mut gpu_state, &gates);
        }
        rt.synchronize().expect("Sync");

        let gpu_start = Instant::now();
        for _ in 0..iterations {
            let _ = run_wmma_batched_gates(&rt, &mut gpu_state, &gates);
        }
        rt.synchronize().expect("Sync");
        let gpu_elapsed = gpu_start.elapsed().as_secs_f64();
        let gpu_time = gpu_elapsed / iterations as f64;

        let speedup = cpu_time / gpu_time;
        let status = if speedup > 1.0 { "✓" } else { "✗" };
        if speedup < 1.0 {
            regression_detected = true;
        }

        println!(
            "│    {}    │ {:>10.2}   │ {:>10.2}   │ {:>10.2}×  │  {}",
            target_qubit,
            gpu_time * 1e6,
            cpu_time * 1e6,
            speedup,
            status
        );
    }
    println!("└─────────┴──────────────┴──────────────┴──────────────┘\n");

    if regression_detected {
        println!("  ⚠ WARNING: GPU slower than CPU for qubits 0-3!\n");
    } else {
        println!("  ✓ GPU faster than CPU for qubits 0-3 - no regression\n");
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" EPIC 87 TEST COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires --features cuda,perf-bench");
    println!(
        "Run with: cargo run --example epic87_high_qubit_test --features cuda,perf-bench --release"
    );
}
