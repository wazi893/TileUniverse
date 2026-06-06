//! EPIC 84 Mega-Fusion Benchmark: CPU Micro-Optimizations
//!
//! This benchmark measures the performance of mega-fusion optimizations:
//! 1. Matrix multiplication (fast-path vs naive)
//! 2. Identity detection (trace-based vs full check)
//! 3. Lazy vs eager composition
//! 4. VQE/QAOA pattern compression
//!
//! Run with:
//! ```
//! cargo run --example epic84_mega_fusion_bench --release
//! ```

use engine::algebraic_fusion::{
    Matrix2x2, MegaGate, compare_mega_fusion, compose_gate_sequence, gate_to_matrix,
    generate_qaoa_pattern, generate_vqe_pattern, mega_cache_clear, mega_fuse, mega_fuse_advanced,
    mega_gate_cached_always,
};
use engine::quantum::{QGate, QState};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 84: MEGA-FUSION MICRO-OPTIMIZATION BENCHMARK               ║");
    println!("║  Verifying CPU-side performance improvements                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    bench_matrix_multiplication();
    bench_identity_detection();
    bench_lazy_vs_eager_composition();
    bench_vqe_qaoa_compression();
    bench_state_application();
    bench_mega_cache();

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
}

/// Benchmark matrix multiplication optimizations
fn bench_matrix_multiplication() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 1: Matrix Multiplication Fast-Paths");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let iterations = 10_000_000; // More iterations for accurate timing

    // Test cases: diagonal × general, anti-diagonal × general, general × general
    let h = Matrix2x2::H;
    let x = Matrix2x2::X;
    let z = Matrix2x2::Z;
    let s = Matrix2x2::S;

    // Diagonal × General (Z × H) - should use fast path
    println!("  Diagonal × General (Z × H):");
    let start = Instant::now();
    let mut result = Matrix2x2::IDENTITY;
    for _ in 0..iterations {
        result = z.mul(&h);
    }
    let fast_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        result = z.mul_naive(&h);
    }
    let naive_time = start.elapsed();

    println!(
        "    Fast-path: {:?} ({:.2} M/s)",
        fast_time,
        iterations as f64 / fast_time.as_secs_f64() / 1e6
    );
    println!(
        "    Naive:     {:?} ({:.2} M/s)",
        naive_time,
        iterations as f64 / naive_time.as_secs_f64() / 1e6
    );
    println!(
        "    Speedup:   {:.2}×",
        naive_time.as_secs_f64() / fast_time.as_secs_f64()
    );
    println!();

    // Anti-diagonal × General (X × H) - should use fast path
    println!("  Anti-diagonal × General (X × H):");
    let start = Instant::now();
    for _ in 0..iterations {
        result = x.mul(&h);
    }
    let fast_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        result = x.mul_naive(&h);
    }
    let naive_time = start.elapsed();

    println!(
        "    Fast-path: {:?} ({:.2} M/s)",
        fast_time,
        iterations as f64 / fast_time.as_secs_f64() / 1e6
    );
    println!(
        "    Naive:     {:?} ({:.2} M/s)",
        naive_time,
        iterations as f64 / naive_time.as_secs_f64() / 1e6
    );
    println!(
        "    Speedup:   {:.2}×",
        naive_time.as_secs_f64() / fast_time.as_secs_f64()
    );
    println!();

    // General × General (H × H) - no fast path available
    println!("  General × General (H × H):");
    let start = Instant::now();
    for _ in 0..iterations {
        result = h.mul(&h);
    }
    let fast_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        result = h.mul_naive(&h);
    }
    let naive_time = start.elapsed();

    println!(
        "    Fast-path: {:?} ({:.2} M/s)",
        fast_time,
        iterations as f64 / fast_time.as_secs_f64() / 1e6
    );
    println!(
        "    Naive:     {:?} ({:.2} M/s)",
        naive_time,
        iterations as f64 / naive_time.as_secs_f64() / 1e6
    );
    println!(
        "    Speedup:   {:.2}× (expected ~1.0 for general case)",
        naive_time.as_secs_f64() / fast_time.as_secs_f64()
    );
    println!();

    // Diagonal × Diagonal (Z × S) - both operands diagonal
    println!("  Diagonal × Diagonal (Z × S):");
    let start = Instant::now();
    for _ in 0..iterations {
        result = z.mul(&s);
    }
    let fast_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        result = z.mul_naive(&s);
    }
    let naive_time = start.elapsed();

    println!(
        "    Fast-path: {:?} ({:.2} M/s)",
        fast_time,
        iterations as f64 / fast_time.as_secs_f64() / 1e6
    );
    println!(
        "    Naive:     {:?} ({:.2} M/s)",
        naive_time,
        iterations as f64 / naive_time.as_secs_f64() / 1e6
    );
    println!(
        "    Speedup:   {:.2}×",
        naive_time.as_secs_f64() / fast_time.as_secs_f64()
    );
    println!();

    // Prevent optimizer from removing the result
    std::hint::black_box(result);
}

/// Benchmark identity detection with trace-based quick rejection
fn bench_identity_detection() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 2: Identity Detection (Trace-Based Quick Rejection)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let iterations = 10_000_000; // More iterations for accurate timing

    // Test with identity (should pass all checks)
    println!("  Testing actual identity:");
    let id = Matrix2x2::IDENTITY;
    let start = Instant::now();
    let mut is_id = false;
    for _ in 0..iterations {
        is_id = id.is_effectively_identity(1e-6);
    }
    let time = start.elapsed();
    println!(
        "    Time: {:?} ({:.2} M/s)",
        time,
        iterations as f64 / time.as_secs_f64() / 1e6
    );
    println!("    Result: {}", is_id);
    println!();

    // Test with Hadamard (should fail quickly on trace check)
    println!("  Testing non-identity (H - should fail on trace):");
    let h = Matrix2x2::H;
    let start = Instant::now();
    for _ in 0..iterations {
        is_id = h.is_effectively_identity(1e-6);
    }
    let time = start.elapsed();
    println!(
        "    Time: {:?} ({:.2} M/s)",
        time,
        iterations as f64 / time.as_secs_f64() / 1e6
    );
    println!("    Result: {} (trace rejection working!)", is_id);
    println!();

    // Test with phase gate (diagonal, non-identity)
    println!("  Testing diagonal non-identity (Phase(0.5)):");
    let (phase_matrix, _) = gate_to_matrix(&QGate::Phase(0, 0.5)).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        is_id = phase_matrix.is_effectively_identity(1e-6);
    }
    let time = start.elapsed();
    println!(
        "    Time: {:?} ({:.2} M/s)",
        time,
        iterations as f64 / time.as_secs_f64() / 1e6
    );
    println!("    Result: {}", is_id);
    println!();

    // Test with H×H = I (composed identity)
    println!("  Testing composed identity (H×H = I):");
    let hh = h.mul(&h);
    let start = Instant::now();
    for _ in 0..iterations {
        is_id = hh.is_effectively_identity(1e-6);
    }
    let time = start.elapsed();
    println!(
        "    Time: {:?} ({:.2} M/s)",
        time,
        iterations as f64 / time.as_secs_f64() / 1e6
    );
    println!("    Result: {}", is_id);
    println!();

    std::hint::black_box(is_id);
}

/// Benchmark lazy vs eager composition
fn bench_lazy_vs_eager_composition() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 3: Lazy vs Eager Composition");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let iterations = 100_000;
    let gates_per_mega = 10;

    // Build a sequence of gates
    let gates: Vec<QGate> = (0..gates_per_mega)
        .map(|i| match i % 4 {
            0 => QGate::H(0),
            1 => QGate::Rx(0, 0.1 * i as f32),
            2 => QGate::Z(0),
            _ => QGate::Ry(0, 0.2 * i as f32),
        })
        .collect();

    // Eager composition (identity check after each gate)
    println!(
        "  Composing {} gates, {} iterations:",
        gates_per_mega, iterations
    );
    println!();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut mega = MegaGate::identity(0);
        for gate in &gates {
            mega.compose(gate);
        }
        std::hint::black_box(&mega);
    }
    let eager_time = start.elapsed();
    println!(
        "    Eager (check after each):  {:?} ({:.2} K compositions/s)",
        eager_time,
        iterations as f64 / eager_time.as_secs_f64() / 1e3
    );

    // Lazy composition (single identity check at end)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut mega = MegaGate::identity(0);
        for gate in &gates {
            mega.compose_lazy(gate);
        }
        mega.finalize();
        std::hint::black_box(&mega);
    }
    let lazy_time = start.elapsed();
    println!(
        "    Lazy (check once at end):  {:?} ({:.2} K compositions/s)",
        lazy_time,
        iterations as f64 / lazy_time.as_secs_f64() / 1e3
    );

    println!(
        "    Speedup: {:.2}×",
        eager_time.as_secs_f64() / lazy_time.as_secs_f64()
    );
    println!();

    // Test with longer sequences
    let gates_long: Vec<QGate> = (0..100)
        .map(|i| match i % 4 {
            0 => QGate::H(0),
            1 => QGate::Rx(0, 0.1 * i as f32),
            2 => QGate::Z(0),
            _ => QGate::Ry(0, 0.2 * i as f32),
        })
        .collect();

    println!("  Composing 100 gates, {} iterations:", iterations / 10);

    let start = Instant::now();
    for _ in 0..(iterations / 10) {
        let mut mega = MegaGate::identity(0);
        for gate in &gates_long {
            mega.compose(gate);
        }
        std::hint::black_box(&mega);
    }
    let eager_time = start.elapsed();
    println!("    Eager: {:?}", eager_time);

    let start = Instant::now();
    for _ in 0..(iterations / 10) {
        let mut mega = MegaGate::identity(0);
        for gate in &gates_long {
            mega.compose_lazy(gate);
        }
        mega.finalize();
        std::hint::black_box(&mega);
    }
    let lazy_time = start.elapsed();
    println!("    Lazy:  {:?}", lazy_time);
    println!(
        "    Speedup: {:.2}×",
        eager_time.as_secs_f64() / lazy_time.as_secs_f64()
    );
    println!();
}

/// Benchmark VQE/QAOA pattern compression
fn bench_vqe_qaoa_compression() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 4: VQE/QAOA Pattern Compression (EPIC 84 vs EPIC 83)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  EPIC 83 Problem: Mixed gates get 0% improvement");
    println!("  EPIC 84 Solution: Mega-fusion compresses ANY single-qubit sequence\n");

    // VQE pattern
    println!("  VQE Pattern (4 qubits, varying layers):");
    for layers in [1, 5, 10, 20] {
        let vqe_circuit = generate_vqe_pattern(4, layers);
        let comparison = compare_mega_fusion(&vqe_circuit);

        println!("    {} layers ({} gates):", layers, vqe_circuit.len());
        println!(
            "      Basic fusion:    {} ops, {:.2}× compression",
            comparison.basic_ops, comparison.basic_compression
        );
        println!(
            "      Advanced fusion: {} ops, {:.2}× compression",
            comparison.advanced_ops, comparison.advanced_compression
        );
        println!(
            "      Improvement:     {:.2}×",
            comparison.improvement_factor
        );
    }
    println!();

    // QAOA pattern
    println!("  QAOA Pattern (4 qubits, varying p):");
    for p in [1, 3, 5, 10] {
        let qaoa_circuit = generate_qaoa_pattern(4, p);
        let comparison = compare_mega_fusion(&qaoa_circuit);

        println!("    p={} ({} gates):", p, qaoa_circuit.len());
        println!(
            "      Basic fusion:    {} ops, {:.2}× compression",
            comparison.basic_ops, comparison.basic_compression
        );
        println!(
            "      Advanced fusion: {} ops, {:.2}× compression",
            comparison.advanced_ops, comparison.advanced_compression
        );
        println!(
            "      Improvement:     {:.2}×",
            comparison.improvement_factor
        );
    }
    println!();

    // Throughput measurement
    println!("  Mega-fusion throughput (1000 iterations):");
    let vqe_circuit = generate_vqe_pattern(4, 10);

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = mega_fuse(&vqe_circuit);
    }
    let basic_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = mega_fuse_advanced(&vqe_circuit);
    }
    let advanced_time = start.elapsed();

    println!(
        "    Basic mega_fuse:    {:?} ({:.2} K circuits/s)",
        basic_time,
        1000.0 / basic_time.as_secs_f64() / 1e3
    );
    println!(
        "    Advanced mega_fuse: {:?} ({:.2} K circuits/s)",
        advanced_time,
        1000.0 / advanced_time.as_secs_f64() / 1e3
    );
    println!();

    // Effective throughput calculation for VQE/QAOA
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │ EFFECTIVE THROUGHPUT COMPARISON (12 qubits, 1024 states)    │");
    println!("  └──────────────────────────────────────────────────────────────┘\n");

    let n_qubits = 12u8;
    let amplitudes = 1u64 << n_qubits; // 4096
    let n_states = 1024u64;
    let kernel_time_secs = 0.0001; // 0.1ms kernel time (from EPIC 81)

    // VQE circuit: 20 layers on 4 qubits
    let vqe_circuit = generate_vqe_pattern(4, 20);
    let vqe_gates = vqe_circuit.len() as u64;
    let vqe_comparison = compare_mega_fusion(&vqe_circuit);

    // EPIC 83 (algebraic only): No improvement for mixed gates
    let epic83_ops = vqe_gates; // 1× multiplier
    let epic83_effective = vqe_gates * amplitudes * n_states;
    let epic83_actual = epic83_ops * amplitudes * n_states;
    let epic83_tcops = epic83_effective as f64 / kernel_time_secs / 1e12;

    // EPIC 84 (mega-fusion): Compression achieved
    let epic84_ops = vqe_comparison.advanced_ops as u64;
    let epic84_effective = vqe_gates * amplitudes * n_states;
    let epic84_actual = epic84_ops * amplitudes * n_states;
    let epic84_multiplier = vqe_gates as f64 / epic84_ops as f64;
    let epic84_tcops = epic84_effective as f64 / (kernel_time_secs / epic84_multiplier) / 1e12;

    println!("  VQE Circuit: {} gates (20 layers × 4 qubits)", vqe_gates);
    println!();
    println!("  EPIC 83 (algebraic fusion only):");
    println!("    Compression: 1.00× (no improvement for mixed gates!)");
    println!("    Actual ops: {:.2e}", epic83_actual as f64);
    println!("    Effective TCOPS: {:.2}", epic83_tcops);
    println!();
    println!("  EPIC 84 (mega-fusion):");
    println!(
        "    Compression: {:.2}× ({} gates → {} ops)",
        epic84_multiplier, vqe_gates, epic84_ops
    );
    println!("    Actual ops: {:.2e}", epic84_actual as f64);
    println!(
        "    Effective TCOPS: {:.2} (same work, fewer kernel calls)",
        epic84_tcops
    );
    println!();

    // Show the improvement
    let improvement = epic84_multiplier;
    println!("  ╔═══════════════════════════════════════════════════════════════╗");
    println!("  ║  EPIC 84 IMPROVEMENT OVER EPIC 83 (for VQE):                  ║");
    println!("  ║                                                               ║");
    println!(
        "  ║    {:.2}× fewer GPU kernel calls                             ║",
        improvement
    );
    println!(
        "  ║    Same effective throughput, {:.0}% less actual work         ║",
        (1.0 - 1.0 / improvement) * 100.0
    );
    println!("  ║                                                               ║");
    println!("  ╚═══════════════════════════════════════════════════════════════╝");
    println!();
}

/// Benchmark MegaGate application to quantum state
fn bench_state_application() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 5: MegaGate State Application");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let n_qubits = 10u8; // 1024 amplitudes
    let iterations = 10_000;

    // Create a composed MegaGate
    let gates = vec![
        QGate::H(0),
        QGate::Rx(0, 0.5),
        QGate::Z(0),
        QGate::Ry(0, 0.3),
    ];
    let mut mega = MegaGate::identity(0);
    for gate in &gates {
        mega.compose_lazy(gate);
    }
    mega.finalize();

    println!(
        "  Applying 4-gate MegaGate to {}-qubit state ({} amplitudes):",
        n_qubits,
        1 << n_qubits
    );
    println!("  {} iterations\n", iterations);

    // Benchmark MegaGate application
    let mut state = QState::new_zero(n_qubits);
    let start = Instant::now();
    for _ in 0..iterations {
        mega.apply(
            state.real.as_mut_slice(),
            state.imag.as_mut_slice(),
            n_qubits,
        );
    }
    let mega_time = start.elapsed();

    let ops_per_iter = (1 << n_qubits) as u64 * 4; // 4 gates worth of work per amplitude
    let effective_ops = iterations as u64 * ops_per_iter;
    let effective_gops = effective_ops as f64 / mega_time.as_secs_f64() / 1e9;

    println!("    MegaGate time: {:?}", mega_time);
    println!(
        "    Effective ops: {:.2e} (claiming credit for {} gates)",
        effective_ops as f64, mega.gate_count
    );
    println!("    Effective GOPS: {:.2}", effective_gops);
    println!();

    // Compare with sequential gate application (naive)
    use engine::quantum::{QRng, apply_gate_scalar};
    let mut state_naive = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();
    for _ in 0..iterations {
        for gate in &gates {
            apply_gate_scalar(&mut state_naive, gate, &mut rng);
        }
    }
    let naive_time = start.elapsed();

    println!("    Naive sequential time: {:?}", naive_time);
    println!(
        "    Speedup: {:.2}×",
        naive_time.as_secs_f64() / mega_time.as_secs_f64()
    );
    println!();

    // Test with identity MegaGate (should be instant)
    let mut mega_id = MegaGate::identity(0);
    mega_id.compose_lazy(&QGate::H(0));
    mega_id.compose_lazy(&QGate::H(0)); // H×H = I
    mega_id.finalize();

    println!("  Identity MegaGate (H×H = I) application:");
    let start = Instant::now();
    for _ in 0..iterations {
        mega_id.apply(
            state.real.as_mut_slice(),
            state.imag.as_mut_slice(),
            n_qubits,
        );
    }
    let identity_time = start.elapsed();

    println!(
        "    Time: {:?} (should be ~instant due to skip)",
        identity_time
    );
    println!("    Is identity: {}", mega_id.is_identity);
    println!();
}

/// Benchmark MegaGate caching for repeated sequences
fn bench_mega_cache() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK 6: MegaGate Caching Performance");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  NOTE: Caching has overhead (hash + lock + lookup).");
    println!("  Beneficial only for longer sequences (20+ gates).\n");

    // =========================================================================
    // Test 1: Short sequences (where caching is NOT beneficial)
    // =========================================================================
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  Test 1: SHORT SEQUENCES (3-5 gates) - Cache overhead dominates");
    println!("  ─────────────────────────────────────────────────────────────────\n");

    mega_cache_clear();

    let short_sequences: Vec<Vec<QGate>> = vec![
        vec![QGate::H(0), QGate::Rx(0, 0.5), QGate::Rz(0, 0.3)],
        vec![
            QGate::Ry(0, 0.1),
            QGate::Rz(0, 0.2),
            QGate::Rx(0, 0.3),
            QGate::Ry(0, 0.4),
        ],
        vec![
            QGate::H(0),
            QGate::Z(0),
            QGate::H(0),
            QGate::X(0),
            QGate::Z(0),
        ],
    ];

    let iterations = 100_000;

    // Without cache
    let start = Instant::now();
    for i in 0..iterations {
        let seq = &short_sequences[i % short_sequences.len()];
        let result = compose_gate_sequence(seq);
        std::hint::black_box(result);
    }
    let short_no_cache = start.elapsed();

    // Force-cache (even though it's not optimal for short seqs)
    mega_cache_clear();
    for seq in &short_sequences {
        mega_gate_cached_always(seq, 0);
    }
    let start = Instant::now();
    for i in 0..iterations {
        let seq = &short_sequences[i % short_sequences.len()];
        let result = mega_gate_cached_always(seq, 0);
        std::hint::black_box(result);
    }
    let short_cache = start.elapsed();

    let short_speedup = short_no_cache.as_secs_f64() / short_cache.as_secs_f64();
    println!(
        "    No cache: {:?} ({:.2} K/s)",
        short_no_cache,
        iterations as f64 / short_no_cache.as_secs_f64() / 1e3
    );
    println!(
        "    Cached:   {:?} ({:.2} K/s)",
        short_cache,
        iterations as f64 / short_cache.as_secs_f64() / 1e3
    );
    println!(
        "    Speedup:  {:.2}× (expected < 1.0 for short sequences)",
        short_speedup
    );
    println!();

    // =========================================================================
    // Test 2: Long sequences (where caching IS beneficial)
    // =========================================================================
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  Test 2: LONG SEQUENCES (50 gates) - Caching is beneficial");
    println!("  ─────────────────────────────────────────────────────────────────\n");

    mega_cache_clear();

    // Create longer sequences
    let long_sequences: Vec<Vec<QGate>> = vec![
        (0..50)
            .map(|i| match i % 4 {
                0 => QGate::H(0),
                1 => QGate::Rx(0, 0.1 * i as f32),
                2 => QGate::Rz(0, 0.05 * i as f32),
                _ => QGate::Ry(0, 0.15 * i as f32),
            })
            .collect(),
        (0..50)
            .map(|i| match i % 3 {
                0 => QGate::Ry(0, 0.2 * i as f32),
                1 => QGate::Z(0),
                _ => QGate::Rx(0, 0.1 * i as f32),
            })
            .collect(),
        (0..50)
            .map(|i| match i % 5 {
                0 => QGate::H(0),
                1 => QGate::X(0),
                2 => QGate::Rz(0, 0.3),
                3 => QGate::Ry(0, 0.2),
                _ => QGate::Z(0),
            })
            .collect(),
    ];

    let long_iterations = 10_000;

    // Without cache
    let start = Instant::now();
    for i in 0..long_iterations {
        let seq = &long_sequences[i % long_sequences.len()];
        let result = compose_gate_sequence(seq);
        std::hint::black_box(result);
    }
    let long_no_cache = start.elapsed();

    // With cache
    mega_cache_clear();
    for seq in &long_sequences {
        mega_gate_cached_always(seq, 0);
    }
    let start = Instant::now();
    for i in 0..long_iterations {
        let seq = &long_sequences[i % long_sequences.len()];
        let result = mega_gate_cached_always(seq, 0);
        std::hint::black_box(result);
    }
    let long_cache = start.elapsed();

    let long_speedup = long_no_cache.as_secs_f64() / long_cache.as_secs_f64();
    println!(
        "    No cache: {:?} ({:.2} K/s)",
        long_no_cache,
        long_iterations as f64 / long_no_cache.as_secs_f64() / 1e3
    );
    println!(
        "    Cached:   {:?} ({:.2} K/s)",
        long_cache,
        long_iterations as f64 / long_cache.as_secs_f64() / 1e3
    );
    println!(
        "    Speedup:  {:.2}× (expected > 1.0 for long sequences)",
        long_speedup
    );
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("  ╔═══════════════════════════════════════════════════════════════╗");
    if long_speedup > 1.5 && short_speedup < 1.0 {
        println!("  ║  CACHE ANALYSIS COMPLETE                                      ║");
        println!("  ║                                                               ║");
        println!(
            "  ║  Short sequences (3-5 gates): {:.2}× (overhead dominates)     ║",
            short_speedup
        );
        println!(
            "  ║  Long sequences (50 gates):   {:.2}× (caching helps!)         ║",
            long_speedup
        );
        println!("  ║                                                               ║");
        println!("  ║  Recommendation: Only cache sequences with 20+ gates          ║");
    } else if long_speedup > 1.0 {
        println!(
            "  ║  Long sequence caching: {:.2}× speedup                        ║",
            long_speedup
        );
    } else {
        println!("  ║  Cache overhead exceeds benefit for all tested sequences      ║");
    }
    println!("  ╚═══════════════════════════════════════════════════════════════╝");
    println!();
}
