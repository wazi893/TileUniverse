//! EPIC 86 End-to-End Validation
//!
//! Tests real VQE/QAOA circuits through the full EPIC 83+84+86 stack.
//! This is the "take a breath and see actual numbers" benchmark.
//!
//! Run with:
//! ```
//! cargo run --example epic86_e2e_validation --release --features cuda,perf-bench
//! ```

use engine::algebraic_fusion::{
    MegaGate, generate_qaoa_pattern, generate_vqe_pattern, mega_fuse_advanced,
};
use engine::fusion::{FusedOp, fuse_program};
use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(70));
    println!(" EPIC 83+84+86 END-TO-END VALIDATION");
    println!(" Testing real circuits through the full optimization stack");
    println!("{}\n", "=".repeat(70));

    // Test 1: VQE circuit compression and fusion analysis
    test_vqe_compression();

    // Test 2: QAOA circuit compression and fusion analysis
    test_qaoa_compression();

    // Test 3: Homogeneous H^N (EPIC 83 algebraic win)
    test_algebraic_elimination();

    // Test 4: Real execution timing (CPU baseline vs fused)
    test_execution_timing();

    // Test 5: GPU execution if available
    #[cfg(feature = "cuda")]
    test_gpu_execution();

    println!("\n{}", "=".repeat(70));
    println!(" VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
}

/// Test VQE circuit compression through the fusion pipeline
fn test_vqe_compression() {
    println!("--- TEST 1: VQE Circuit Compression ---\n");

    for layers in [1, 5, 10, 20] {
        let gates = generate_vqe_pattern(4, layers);
        let original = gates.len();

        // Fusion with mega-fusion enabled (default)
        let fused = fuse_program(&gates);

        // Also test mega_fuse_advanced directly
        let mega_result = mega_fuse_advanced(&gates);

        println!("  VQE (4 qubits, {} layers):", layers);
        println!("    Original gates:    {}", original);
        println!("    Fused ops:         {}", fused.ops.len());
        println!("    Eliminated:        {}", fused.eliminated);
        println!(
            "    Sequences fused:   {}",
            mega_result.sequences_compressed
        );
        println!(
            "    Compression:       {:.2}x",
            mega_result.compression_ratio
        );
        println!();
    }
}

/// Test QAOA circuit compression through the fusion pipeline
fn test_qaoa_compression() {
    println!("--- TEST 2: QAOA Circuit Compression ---\n");

    for p in [1, 3, 5, 10] {
        let gates = generate_qaoa_pattern(4, p);
        let original = gates.len();

        let fused = fuse_program(&gates);
        let mega_result = mega_fuse_advanced(&gates);

        println!("  QAOA (4 qubits, p={}):", p);
        println!("    Original gates:    {}", original);
        println!("    Fused ops:         {}", fused.ops.len());
        println!("    Eliminated:        {}", fused.eliminated);
        println!(
            "    Sequences fused:   {}",
            mega_result.sequences_compressed
        );
        println!(
            "    Compression:       {:.2}x",
            mega_result.compression_ratio
        );
        println!();
    }
}

/// Test algebraic elimination (H^N = I for even N)
fn test_algebraic_elimination() {
    println!("--- TEST 3: Algebraic Elimination (EPIC 83) ---\n");

    // Test H^N patterns
    for n in [2, 10, 100, 1000] {
        let gates: Vec<QGate> = (0..n).map(|_| QGate::H(0)).collect();
        let original = gates.len();
        let fused = fuse_program(&gates);

        let effective = if n % 2 == 0 {
            "eliminated (H^even = I)"
        } else {
            "single H (H^odd = H)"
        };

        println!(
            "  H^{}: {} gates -> {} ops [{}]",
            n,
            original,
            fused.ops.len(),
            effective
        );
    }
    println!();
}

/// Apply MegaGate to a QState
fn apply_mega_gate(state: &mut QState, mega: &MegaGate) {
    mega.apply(
        state.real.as_mut_slice(),
        state.imag.as_mut_slice(),
        state.n_qubits,
    );
}

/// Test actual execution timing: naive vs fused
fn test_execution_timing() {
    println!("--- TEST 4: Execution Timing (CPU) ---\n");

    let n_qubits = 8;
    let layers = 10;
    let iterations = 100;

    let gates = generate_vqe_pattern(n_qubits, layers);
    let original_count = gates.len();

    println!(
        "  Circuit: VQE {} qubits, {} layers ({} gates)",
        n_qubits, layers, original_count
    );
    println!("  Iterations: {}\n", iterations);

    // Naive execution: apply each gate sequentially
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
        for gate in &gates {
            apply_gate_scalar(&mut state, gate, &mut rng);
        }
    }
    let naive_time = start.elapsed();

    // Fused execution: use the fusion pipeline
    let fused = fuse_program(&gates);
    let fused_count = fused.ops.len();

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
        // Execute fused ops
        for op in &fused.ops {
            match op {
                FusedOp::Single(gate) => {
                    apply_gate_scalar(&mut state, gate, &mut rng);
                }
                FusedOp::Batched { gate, count } => {
                    for _ in 0..*count {
                        apply_gate_scalar(&mut state, gate, &mut rng);
                    }
                }
                FusedOp::Mega(mega) => {
                    apply_mega_gate(&mut state, mega);
                }
                _ => {}
            }
        }
    }
    let fused_time = start.elapsed();

    println!("  Naive:  {:?} ({} gates/iter)", naive_time, original_count);
    println!("  Fused:  {:?} ({} ops/iter)", fused_time, fused_count);
    println!(
        "  Speedup: {:.2}x",
        naive_time.as_secs_f64() / fused_time.as_secs_f64()
    );
    println!(
        "  Compression: {:.2}x",
        original_count as f64 / fused_count as f64
    );
    println!();
}

/// Test GPU execution through the fusion pipeline
#[cfg(feature = "cuda")]
fn test_gpu_execution() {
    use engine::cuda::{CudaRuntime, GpuQState, is_cuda_available, is_wmma_available};
    use engine::fusion::execute_fused_clustered;

    println!("--- TEST 5: GPU Execution ---\n");

    if !is_cuda_available() {
        println!("  CUDA not available, skipping GPU tests\n");
        return;
    }

    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            println!("  Failed to create CUDA runtime: {:?}\n", e);
            return;
        }
    };

    let wmma_available = is_wmma_available(&rt);
    println!("  CUDA: available");
    println!(
        "  WMMA: {}\n",
        if wmma_available {
            "available"
        } else {
            "not available"
        }
    );

    // Test with different circuit sizes
    for (n_qubits, layers) in [(4, 5), (8, 10), (10, 20)] {
        let gates = generate_vqe_pattern(n_qubits, layers);
        let original_count = gates.len();
        let fused = fuse_program(&gates);

        // Count different op types
        let mut single_count = 0;
        let mut batched_count = 0;
        let mut mega_count = 0;
        let mut batched_layer_count = 0;

        for op in &fused.ops {
            match op {
                FusedOp::Single(_) => single_count += 1,
                FusedOp::Batched { .. } => batched_count += 1,
                FusedOp::Mega(_) => mega_count += 1,
                FusedOp::BatchedLayer { .. } => batched_layer_count += 1,
                _ => {}
            }
        }

        println!("  VQE ({} qubits, {} layers):", n_qubits, layers);
        println!("    Original: {} gates", original_count);
        println!("    Fused:    {} ops", fused.ops.len());
        println!("      - Single:       {}", single_count);
        println!("      - Batched:      {}", batched_count);
        println!("      - MegaGate:     {}", mega_count);
        println!("      - BatchedLayer: {}", batched_layer_count);

        // Check which ops would use WMMA (EPIC 86B: qubits 0-3)
        let wmma_eligible = fused
            .ops
            .iter()
            .filter(|op| match op {
                FusedOp::Mega(m) => m.qubit <= 3,
                FusedOp::BatchedLayer { gates, .. } => gates.iter().all(|g| match g {
                    QGate::H(q) | QGate::X(q) | QGate::Z(q) | QGate::Phase(q, _) if *q <= 3 => true,
                    _ => false,
                }),
                _ => false,
            })
            .count();

        println!("    WMMA-eligible ops: {} (qubits 0-3)", wmma_eligible);
        println!();
    }

    // Test clustered GPU execution (the correct path)
    println!("  GPU Clustered Execution (EPIC 73):\n");

    // Use 12 qubits (4096 amplitudes) to better exercise GPU
    let n_qubits = 12u8;
    let layers = 10;
    let iterations = 10;

    let gates = generate_vqe_pattern(n_qubits, layers);
    let fused = fuse_program(&gates);
    let tiles = 1usize << n_qubits; // 2^n_qubits = state size / 16

    println!(
        "    Circuit: {} gates -> {} ops",
        gates.len(),
        fused.ops.len()
    );

    // Time clustered execution
    let mut cpu_state = QState::new_zero(n_qubits);

    let start = Instant::now();
    for _ in 0..iterations {
        cpu_state = QState::new_zero(n_qubits);
        match execute_fused_clustered(&rt, &mut cpu_state, &fused, n_qubits, tiles) {
            Ok(_result) => {}
            Err(e) => {
                println!("    Clustered execution error: {:?}", e);
                return;
            }
        }
    }
    let exec_time = start.elapsed();
    println!(
        "    Clustered execution ({}x): {:?} ({:?}/iter)",
        iterations,
        exec_time,
        exec_time / iterations as u32
    );

    // Get detailed stats from last run
    cpu_state = QState::new_zero(n_qubits);
    match execute_fused_clustered(&rt, &mut cpu_state, &fused, n_qubits, tiles) {
        Ok(result) => {
            println!("\n    Execution stats:");
            println!("      Total ops: {}", result.total_ops);
            println!("      GPU clusters: {}", result.clusters_executed);
            println!("      CPU ops: {}", result.cpu_ops_executed);
            println!("      WMMA ops: {}", result.wmma_ops);
            println!("      WMMA Packed: {}", result.wmma_packed_ops);
            println!("      FP32 ops: {}", result.fp32_ops);
            println!("      FP16 ops: {}", result.fp16_ops);
            println!("      Sync points: {}", result.sync_points);
        }
        Err(e) => {
            println!("    Error getting stats: {:?}", e);
        }
    }
    println!();
}
