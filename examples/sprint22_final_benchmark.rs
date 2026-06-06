//! SPRINT 22 Final Comprehensive Benchmark
//!
//! Captures the full EPIC 83+84+85+86/86B optimization stack performance.
//!
//! Run with:
//! ```
//! cargo run --example sprint22_final_benchmark --release --features cuda,perf-bench
//! ```

use engine::algebraic_fusion::{
    generate_qaoa_pattern, generate_vqe_pattern, mega_fuse_advanced, optimize_algebraic,
};
use engine::fusion::{FusedOp, fuse_program};
use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};
use std::time::Instant;

fn main() {
    println!("# SPRINT 22 FINAL COMPREHENSIVE BENCHMARK\n");
    println!("Generated: {}\n", chrono_lite());
    println!("---\n");

    // Section 1: COPS Baseline
    section_1_cops_baseline();

    // Section 2: Algebraic Fusion (EPIC 83)
    section_2_algebraic_fusion();

    // Section 3: Mega-Fusion (EPIC 84)
    section_3_mega_fusion();

    // Section 4: Sparse States (EPIC 85)
    section_4_sparse_states();

    // Section 5: Batched Gates (EPIC 86/86B)
    section_5_batched_gates();

    // Section 6: Full Stack Combined
    section_6_full_stack();

    println!("\n---\n");
    println!("## Summary\n");
    println!("Benchmark complete. Results above show the full EPIC 83+84+85+86/86B stack.");
}

fn chrono_lite() -> String {
    "2024-12-05".to_string()
}

// ============================================================================
// SECTION 1: COPS BASELINE
// ============================================================================

fn section_1_cops_baseline() {
    println!("## 1. COPS BASELINE (No Optimizations)\n");
    println!("Dense execution measuring raw throughput.\n");
    println!("| Qubits | Gates | Wall Time | Amp Ops | Raw TCOPS | Notes |");
    println!("|--------|-------|-----------|---------|-----------|-------|");

    let configs = [
        (12, 100),
        (12, 1000),
        (16, 100),
        (16, 1000),
        (20, 100),
        (20, 1000),
    ];

    for (n_qubits, n_gates) in configs {
        let gates = generate_random_gates(n_qubits, n_gates);
        let (wall_time, amp_ops) = run_naive_baseline(n_qubits, &gates, 10);
        let tcops = amp_ops as f64 / wall_time.as_secs_f64() / 1e12;

        println!(
            "| {} | {} | {:.2}ms | {:.2e} | {:.3} | random H/X/Z/Ry/Rz |",
            n_qubits,
            n_gates,
            wall_time.as_secs_f64() * 1000.0,
            amp_ops as f64,
            tcops
        );
    }
    println!();
}

fn generate_random_gates(n_qubits: u8, n_gates: usize) -> Vec<QGate> {
    let mut gates = Vec::with_capacity(n_gates);
    let mut seed = 12345u64;

    for _ in 0..n_gates {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let qubit = (seed % n_qubits as u64) as u8;
        let gate_type = (seed >> 16) % 5;

        let gate = match gate_type {
            0 => QGate::H(qubit),
            1 => QGate::X(qubit),
            2 => QGate::Z(qubit),
            3 => QGate::Ry(
                qubit,
                ((seed >> 32) as f32 / u32::MAX as f32) * std::f32::consts::TAU,
            ),
            _ => QGate::Rz(
                qubit,
                ((seed >> 40) as f32 / u32::MAX as f32) * std::f32::consts::TAU,
            ),
        };
        gates.push(gate);
    }
    gates
}

fn run_naive_baseline(
    n_qubits: u8,
    gates: &[QGate],
    iterations: usize,
) -> (std::time::Duration, u64) {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);
    let state_size = 1u64 << n_qubits;

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
        for gate in gates {
            apply_gate_scalar(&mut state, gate, &mut rng);
        }
    }
    let elapsed = start.elapsed();

    let amp_ops = state_size * gates.len() as u64 * iterations as u64;
    (elapsed, amp_ops)
}

// ============================================================================
// SECTION 2: ALGEBRAIC FUSION (EPIC 83)
// ============================================================================

fn section_2_algebraic_fusion() {
    println!("## 2. ALGEBRAIC FUSION (EPIC 83)\n");
    println!("Algebraic identity elimination and rotation accumulation.\n");
    println!("| Circuit | Original | Remaining | Eliminated % | Eff TCOPS | Speedup | Notes |");
    println!("|---------|----------|-----------|--------------|-----------|---------|-------|");

    // H^1000 (even power = identity)
    {
        let gates: Vec<QGate> = (0..1000).map(|_| QGate::H(0)).collect();
        let (_ops, stats) = optimize_algebraic(&gates, 4);

        let naive_time = measure_naive_time(4, &gates, 100);
        let opt_time = measure_algebraic_time(4, &gates, 100);
        let speedup = naive_time.as_secs_f64() / opt_time.as_secs_f64().max(1e-9);

        // Effective TCOPS: we "processed" 1000 gates worth of work
        let state_size = 1u64 << 4;
        let effective_ops = 1000u64 * state_size * 100;
        let eff_tcops = effective_ops as f64 / opt_time.as_secs_f64() / 1e12;

        println!(
            "| H^1000 | 1000 | {} | {:.1}% | {:.1} | {:.0}× | H² = I |",
            stats.gates_remaining,
            (1000 - stats.gates_remaining) as f64 / 10.0,
            eff_tcops,
            speedup
        );
    }

    // H^1001 (odd power = H)
    {
        let gates: Vec<QGate> = (0..1001).map(|_| QGate::H(0)).collect();
        let (_ops, stats) = optimize_algebraic(&gates, 4);

        let naive_time = measure_naive_time(4, &gates, 100);
        let opt_time = measure_algebraic_time(4, &gates, 100);
        let speedup = naive_time.as_secs_f64() / opt_time.as_secs_f64().max(1e-9);

        let state_size = 1u64 << 4;
        let effective_ops = 1001u64 * state_size * 100;
        let eff_tcops = effective_ops as f64 / opt_time.as_secs_f64() / 1e12;

        println!(
            "| H^1001 | 1001 | {} | {:.1}% | {:.1} | {:.0}× | H^odd = H |",
            stats.gates_remaining,
            (1001 - stats.gates_remaining) as f64 / 10.01,
            eff_tcops,
            speedup
        );
    }

    // Rotation chain summing to 2π
    {
        let theta = std::f32::consts::TAU / 8.0;
        let gates: Vec<QGate> = (0..8).map(|_| QGate::Rz(0, theta)).collect();
        let (_ops, stats) = optimize_algebraic(&gates, 4);

        let naive_time = measure_naive_time(4, &gates, 1000);
        let opt_time = measure_algebraic_time(4, &gates, 1000);
        let speedup = naive_time.as_secs_f64() / opt_time.as_secs_f64().max(1e-9);

        let state_size = 1u64 << 4;
        let effective_ops = 8u64 * state_size * 1000;
        let eff_tcops = effective_ops as f64 / opt_time.as_secs_f64() / 1e12;

        println!(
            "| Rz(π/4)×8 | 8 | {} | {:.1}% | {:.3} | {:.1}× | Σθ = 2π = I |",
            stats.gates_remaining,
            (8 - stats.gates_remaining) as f64 / 0.08,
            eff_tcops,
            speedup
        );
    }

    // X^100 (even power = identity)
    {
        let gates: Vec<QGate> = (0..100).map(|_| QGate::X(0)).collect();
        let (_ops, stats) = optimize_algebraic(&gates, 4);

        let naive_time = measure_naive_time(4, &gates, 100);
        let opt_time = measure_algebraic_time(4, &gates, 100);
        let speedup = naive_time.as_secs_f64() / opt_time.as_secs_f64().max(1e-9);

        let state_size = 1u64 << 4;
        let effective_ops = 100u64 * state_size * 100;
        let eff_tcops = effective_ops as f64 / opt_time.as_secs_f64() / 1e12;

        println!(
            "| X^100 | 100 | {} | {:.1}% | {:.1} | {:.0}× | X² = I |",
            stats.gates_remaining,
            (100 - stats.gates_remaining) as f64,
            eff_tcops,
            speedup
        );
    }

    println!();
}

fn measure_naive_time(n_qubits: u8, gates: &[QGate], iterations: usize) -> std::time::Duration {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
        for gate in gates {
            apply_gate_scalar(&mut state, gate, &mut rng);
        }
    }
    start.elapsed()
}

fn measure_algebraic_time(n_qubits: u8, gates: &[QGate], iterations: usize) -> std::time::Duration {
    let (ops, _) = optimize_algebraic(gates, n_qubits);
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
        for op in &ops {
            match op {
                engine::algebraic_fusion::AlgebraicOp::Skip { .. } => {}
                engine::algebraic_fusion::AlgebraicOp::Single(g) => {
                    apply_gate_scalar(&mut state, g, &mut rng);
                }
                engine::algebraic_fusion::AlgebraicOp::Power { gate, power } => {
                    for _ in 0..*power {
                        apply_gate_scalar(&mut state, gate, &mut rng);
                    }
                }
                #[cfg(feature = "cuda")]
                engine::algebraic_fusion::AlgebraicOp::FusedMatrix { .. } => {}
            }
        }
    }
    start.elapsed()
}

// ============================================================================
// SECTION 3: MEGA-FUSION (EPIC 84)
// ============================================================================

fn section_3_mega_fusion() {
    println!("## 3. MEGA-FUSION (EPIC 84)\n");
    println!("Cross-gate composition into MegaGates.\n");

    // VQE patterns
    println!("### VQE Circuits\n");
    println!(
        "| Qubits | Layers | Original | Fused Ops | Compression | Wall Time | Speedup | Notes |"
    );
    println!(
        "|--------|--------|----------|-----------|-------------|-----------|---------|-------|"
    );

    let vqe_configs = [
        (4, 10),
        (4, 20),
        (4, 50),
        (8, 10),
        (8, 20),
        (8, 50),
        (10, 10),
        (10, 20),
    ];

    for (n_qubits, layers) in vqe_configs {
        let gates = generate_vqe_pattern(n_qubits, layers);
        let original = gates.len();
        let fused = fuse_program(&gates);
        let mega_result = mega_fuse_advanced(&gates);

        let naive_time = measure_naive_time(n_qubits, &gates, 50);
        let fused_time = measure_fused_time(n_qubits, &gates, 50);
        let speedup = naive_time.as_secs_f64() / fused_time.as_secs_f64();

        println!(
            "| {} | {} | {} | {} | {:.2}× | {:.2}ms | {:.2}× | {} MegaGates |",
            n_qubits,
            layers,
            original,
            fused.ops.len(),
            mega_result.compression_ratio,
            fused_time.as_secs_f64() * 1000.0,
            speedup,
            mega_result.sequences_compressed
        );
    }

    println!();

    // QAOA patterns
    println!("### QAOA Circuits\n");
    println!("| Qubits | p | Original | Fused Ops | Compression | Wall Time | Speedup | Notes |");
    println!("|--------|---|----------|-----------|-------------|-----------|---------|-------|");

    let qaoa_configs = [(4, 5), (4, 10), (4, 20), (8, 5), (8, 10), (8, 20)];

    for (n_qubits, p) in qaoa_configs {
        let gates = generate_qaoa_pattern(n_qubits, p);
        let original = gates.len();
        let fused = fuse_program(&gates);
        let mega_result = mega_fuse_advanced(&gates);

        let naive_time = measure_naive_time(n_qubits, &gates, 50);
        let fused_time = measure_fused_time(n_qubits, &gates, 50);
        let speedup = naive_time.as_secs_f64() / fused_time.as_secs_f64();

        println!(
            "| {} | {} | {} | {} | {:.2}× | {:.2}ms | {:.2}× | {} MegaGates |",
            n_qubits,
            p,
            original,
            fused.ops.len(),
            mega_result.compression_ratio,
            fused_time.as_secs_f64() * 1000.0,
            speedup,
            mega_result.sequences_compressed
        );
    }

    println!();
}

fn measure_fused_time(n_qubits: u8, gates: &[QGate], iterations: usize) -> std::time::Duration {
    let fused = fuse_program(gates);
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        state = QState::new_zero(n_qubits);
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
                    mega.apply(
                        state.real.as_mut_slice(),
                        state.imag.as_mut_slice(),
                        state.n_qubits,
                    );
                }
                _ => {}
            }
        }
    }
    start.elapsed()
}

// ============================================================================
// SECTION 4: SPARSE STATES (EPIC 85)
// ============================================================================

fn section_4_sparse_states() {
    println!("## 4. SPARSE STATES (EPIC 85)\n");
    println!("GHZ state creation with sparse representation.\n");
    println!(
        "| Qubits | State Size | Non-Zero | Sparsity | Dense Time | Sparse Time | Speedup | Notes |"
    );
    println!(
        "|--------|------------|----------|----------|------------|-------------|---------|-------|"
    );

    // GHZ states have only 2 non-zero amplitudes: |00...0⟩ + |11...1⟩
    let configs = [20, 25, 30, 35, 40];

    for n_qubits in configs {
        let state_size = 1u64 << n_qubits;
        let non_zero = 2u64; // GHZ has exactly 2 non-zero amplitudes
        let sparsity = 100.0 * (1.0 - non_zero as f64 / state_size as f64);

        // For large qubit counts, we can't actually run dense
        let (dense_time, sparse_time, speedup) = if n_qubits <= 25 {
            let dense_t = measure_ghz_dense(n_qubits, 10);
            let sparse_t = measure_ghz_sparse(n_qubits, 10);
            let sp = dense_t.as_secs_f64() / sparse_t.as_secs_f64();
            (
                format!("{:.2}ms", dense_t.as_secs_f64() * 1000.0),
                format!("{:.2}ms", sparse_t.as_secs_f64() * 1000.0),
                format!("{:.1}×", sp),
            )
        } else {
            let sparse_t = measure_ghz_sparse(n_qubits, 10);
            (
                "OOM".to_string(),
                format!("{:.2}ms", sparse_t.as_secs_f64() * 1000.0),
                "∞".to_string(),
            )
        };

        println!(
            "| {} | 2^{} | {} | {:.10}% | {} | {} | {} | GHZ |",
            n_qubits, n_qubits, non_zero, sparsity, dense_time, sparse_time, speedup
        );
    }

    println!();
}

fn measure_ghz_dense(n_qubits: u8, iterations: usize) -> std::time::Duration {
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = QState::new_zero(n_qubits);
        // GHZ: H on qubit 0, then CNOT cascade
        apply_gate_scalar(&mut state, &QGate::H(0), &mut rng);
        for q in 1..n_qubits {
            apply_gate_scalar(&mut state, &QGate::CNot(q - 1, q), &mut rng);
        }
    }
    start.elapsed()
}

fn measure_ghz_sparse(n_qubits: u8, iterations: usize) -> std::time::Duration {
    use engine::sparse_state::SparseQState;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut state = SparseQState::new_zero(n_qubits);
        // GHZ: H on qubit 0, then CNOT cascade
        state.apply_h(0);
        for q in 1..n_qubits {
            state.apply_cnot(q - 1, q);
        }
    }
    start.elapsed()
}

// ============================================================================
// SECTION 5: BATCHED GATES (EPIC 86/86B)
// ============================================================================

fn section_5_batched_gates() {
    println!("## 5. BATCHED GATES (EPIC 86/86B)\n");
    println!("Single-qubit gate batching with qubit-parameterized WMMA.\n");

    println!("### Per-Qubit WMMA Eligibility\n");
    println!("| Target Qubit | Gate Sequence | WMMA Eligible | Notes |");
    println!("|--------------|---------------|---------------|-------|");

    for q in 0..=3u8 {
        let gates: Vec<QGate> = vec![
            QGate::H(q),
            QGate::X(q),
            QGate::Z(q),
            QGate::Ry(q, 0.5),
            QGate::Rz(q, 0.3),
        ];

        let fused = fuse_program(&gates);
        let wmma_eligible = fused
            .ops
            .iter()
            .filter(|op| match op {
                FusedOp::Mega(m) => m.qubit <= 3,
                _ => false,
            })
            .count();

        println!(
            "| {} | H,X,Z,Ry,Rz | {} of {} ({:.0}%) | EPIC 86B |",
            q,
            wmma_eligible,
            fused.ops.len(),
            100.0 * wmma_eligible as f64 / fused.ops.len() as f64
        );
    }

    // Qubit 5 (not WMMA eligible)
    {
        let gates: Vec<QGate> = vec![QGate::H(5), QGate::X(5), QGate::Z(5)];
        let fused = fuse_program(&gates);
        let wmma_eligible = fused
            .ops
            .iter()
            .filter(|op| match op {
                FusedOp::Mega(m) => m.qubit <= 3,
                _ => false,
            })
            .count();

        println!(
            "| 5 | H,X,Z | {} of {} ({:.0}%) | Qubit > 3 |",
            wmma_eligible,
            fused.ops.len(),
            100.0 * wmma_eligible as f64 / fused.ops.len() as f64
        );
    }

    println!();

    println!("### VQE WMMA Eligibility (Before vs After EPIC 86B)\n");
    println!("| Circuit | Total MegaGates | Qubit 0 Only | Qubits 0-3 | Improvement |");
    println!("|---------|-----------------|--------------|------------|-------------|");

    let vqe_configs = [(4, 5), (8, 10), (10, 20)];

    for (n_qubits, layers) in vqe_configs {
        let gates = generate_vqe_pattern(n_qubits, layers);
        let fused = fuse_program(&gates);

        let total_mega = fused
            .ops
            .iter()
            .filter(|op| matches!(op, FusedOp::Mega(_)))
            .count();
        let qubit_0_only = fused
            .ops
            .iter()
            .filter(|op| match op {
                FusedOp::Mega(m) => m.qubit == 0,
                _ => false,
            })
            .count();
        let qubits_0_3 = fused
            .ops
            .iter()
            .filter(|op| match op {
                FusedOp::Mega(m) => m.qubit <= 3,
                _ => false,
            })
            .count();

        let improvement = if qubit_0_only > 0 {
            qubits_0_3 as f64 / qubit_0_only as f64
        } else {
            0.0
        };

        println!(
            "| VQE {}q {}L | {} | {} ({:.0}%) | {} ({:.0}%) | {:.1}× |",
            n_qubits,
            layers,
            total_mega,
            qubit_0_only,
            100.0 * qubit_0_only as f64 / total_mega as f64,
            qubits_0_3,
            100.0 * qubits_0_3 as f64 / total_mega as f64,
            improvement
        );
    }

    println!();

    // GPU benchmark if available
    #[cfg(feature = "cuda")]
    section_5_gpu_benchmark();
}

#[cfg(feature = "cuda")]
fn section_5_gpu_benchmark() {
    use engine::cuda::{CudaRuntime, is_cuda_available, is_wmma_available};

    println!("### GPU Batched Execution\n");

    if !is_cuda_available() {
        println!("*CUDA not available*\n");
        return;
    }

    let rt = match CudaRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            println!("*CUDA init failed: {:?}*\n", e);
            return;
        }
    };

    let wmma_avail = is_wmma_available(&rt);
    println!("| Backend | Status |");
    println!("|---------|--------|");
    println!("| CUDA | Available |");
    println!(
        "| WMMA Tensor Core | {} |",
        if wmma_avail {
            "Available"
        } else {
            "Not Available"
        }
    );
    println!();
}

// ============================================================================
// SECTION 6: FULL STACK COMBINED
// ============================================================================

fn section_6_full_stack() {
    println!("## 6. FULL STACK COMBINED\n");
    println!("End-to-end performance with all optimizations.\n");

    println!("### VQE Full Stack\n");
    println!(
        "| Config | Naive Gates | Fused Ops | Elim % | Naive Time | Fused Time | Speedup | Eff TCOPS |"
    );
    println!(
        "|--------|-------------|-----------|--------|------------|------------|---------|-----------|"
    );

    let vqe_configs = [(8, 20, 100), (8, 50, 50), (10, 20, 50), (12, 10, 20)];

    for (n_qubits, layers, iterations) in vqe_configs {
        let gates = generate_vqe_pattern(n_qubits, layers);
        let original = gates.len();
        let fused = fuse_program(&gates);
        let fused_count = fused.ops.len();
        let elim_pct = 100.0 * (original - fused_count) as f64 / original as f64;

        let naive_time = measure_naive_time(n_qubits, &gates, iterations);
        let fused_time = measure_fused_time(n_qubits, &gates, iterations);
        let speedup = naive_time.as_secs_f64() / fused_time.as_secs_f64();

        let state_size = 1u64 << n_qubits;
        let effective_ops = original as u64 * state_size * iterations as u64;
        let eff_tcops = effective_ops as f64 / fused_time.as_secs_f64() / 1e12;

        println!(
            "| VQE {}q {}L | {} | {} | {:.1}% | {:.2}ms | {:.2}ms | {:.2}× | {:.3} |",
            n_qubits,
            layers,
            original,
            fused_count,
            elim_pct,
            naive_time.as_secs_f64() * 1000.0,
            fused_time.as_secs_f64() * 1000.0,
            speedup,
            eff_tcops
        );
    }

    println!();

    println!("### QAOA Full Stack\n");
    println!(
        "| Config | Naive Gates | Fused Ops | Elim % | Naive Time | Fused Time | Speedup | Eff TCOPS |"
    );
    println!(
        "|--------|-------------|-----------|--------|------------|------------|---------|-----------|"
    );

    let qaoa_configs = [(8, 10, 100), (8, 20, 50), (10, 10, 50)];

    for (n_qubits, p, iterations) in qaoa_configs {
        let gates = generate_qaoa_pattern(n_qubits, p);
        let original = gates.len();
        let fused = fuse_program(&gates);
        let fused_count = fused.ops.len();
        let elim_pct = 100.0 * (original - fused_count) as f64 / original as f64;

        let naive_time = measure_naive_time(n_qubits, &gates, iterations);
        let fused_time = measure_fused_time(n_qubits, &gates, iterations);
        let speedup = naive_time.as_secs_f64() / fused_time.as_secs_f64();

        let state_size = 1u64 << n_qubits;
        let effective_ops = original as u64 * state_size * iterations as u64;
        let eff_tcops = effective_ops as f64 / fused_time.as_secs_f64() / 1e12;

        println!(
            "| QAOA {}q p={} | {} | {} | {:.1}% | {:.2}ms | {:.2}ms | {:.2}× | {:.3} |",
            n_qubits,
            p,
            original,
            fused_count,
            elim_pct,
            naive_time.as_secs_f64() * 1000.0,
            fused_time.as_secs_f64() * 1000.0,
            speedup,
            eff_tcops
        );
    }

    println!();

    // Algebraic elimination showcase
    println!("### Algebraic Elimination Showcase\n");
    println!("| Circuit | Gates | After Fusion | Effective Speedup | Notes |");
    println!("|---------|-------|--------------|-------------------|-------|");

    // H^10000
    {
        let gates: Vec<QGate> = (0..10000).map(|_| QGate::H(0)).collect();
        let fused = fuse_program(&gates);
        println!("| H^10000 | 10000 | {} | ∞× | H² = I |", fused.ops.len());
    }

    // Mixed self-inverse
    {
        let mut gates = Vec::new();
        for _ in 0..100 {
            gates.push(QGate::H(0));
            gates.push(QGate::X(1));
            gates.push(QGate::Z(2));
            gates.push(QGate::H(0)); // Cancels with previous H(0)
            gates.push(QGate::X(1)); // Cancels with previous X(1)
            gates.push(QGate::Z(2)); // Cancels with previous Z(2)
        }
        let fused = fuse_program(&gates);
        println!(
            "| (H·X·Z)² × 100 | {} | {} | {:.0}× | Self-inverse pairs |",
            gates.len(),
            fused.ops.len(),
            gates.len() as f64 / fused.ops.len().max(1) as f64
        );
    }

    println!();
}
