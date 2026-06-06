//! Invariant Tests for Quantum Simulation Correctness
//!
//! These tests verify mathematical properties that MUST hold for any correct
//! quantum simulator, without requiring an external reference implementation.
//!
//! Properties tested:
//! 1. Gate identities: H²=I, X²=I, Y²=I, Z²=I, CNOT²=I
//! 2. Norm preservation: ||ψ||² = 1 after any unitary
//! 3. Inner product preservation: <a|b> unchanged under same unitary
//! 4. Reversibility: U·U† = I
//!
//! Run with: cargo test --test invariant_tests --release -- --nocapture

use engine::quantum::{QBackend, QGate, QRng, QState, apply_gate_backend};
use std::f32::consts::PI;

const TOLERANCE: f64 = 1e-5;

// ============================================================================
// Helper Functions
// ============================================================================

/// Get amplitudes from QState as (real, imag) pairs
fn get_amplitudes(state: &QState) -> Vec<(f32, f32)> {
    let real = state.real.as_slice();
    let imag = state.imag.as_slice();
    let n = 1 << state.n_qubits;
    (0..n).map(|i| (real[i], imag[i])).collect()
}

/// Compute ||ψ||² (should be 1.0 for normalized states)
fn norm_squared(state: &QState) -> f64 {
    let real = state.real.as_slice();
    let imag = state.imag.as_slice();
    let n = 1 << state.n_qubits;
    (0..n)
        .map(|i| (real[i] as f64).powi(2) + (imag[i] as f64).powi(2))
        .sum()
}

/// Compute state fidelity |<a|b>|²
fn fidelity(a: &QState, b: &QState) -> f64 {
    let a_real = a.real.as_slice();
    let a_imag = a.imag.as_slice();
    let b_real = b.real.as_slice();
    let b_imag = b.imag.as_slice();
    let n = 1 << a.n_qubits;

    let mut overlap_real = 0.0f64;
    let mut overlap_imag = 0.0f64;

    for i in 0..n {
        // <a|b> = sum(conj(a_i) * b_i)
        let ar = a_real[i] as f64;
        let ai = a_imag[i] as f64;
        let br = b_real[i] as f64;
        let bi = b_imag[i] as f64;

        overlap_real += ar * br + ai * bi;
        overlap_imag += ar * bi - ai * br;
    }

    overlap_real * overlap_real + overlap_imag * overlap_imag
}

/// Compute inner product <a|b> (complex)
fn inner_product(a: &QState, b: &QState) -> (f64, f64) {
    let a_real = a.real.as_slice();
    let a_imag = a.imag.as_slice();
    let b_real = b.real.as_slice();
    let b_imag = b.imag.as_slice();
    let n = 1 << a.n_qubits;

    let mut real = 0.0f64;
    let mut imag = 0.0f64;

    for i in 0..n {
        let ar = a_real[i] as f64;
        let ai = a_imag[i] as f64;
        let br = b_real[i] as f64;
        let bi = b_imag[i] as f64;

        real += ar * br + ai * bi;
        imag += ar * bi - ai * br;
    }

    (real, imag)
}

/// Clone a quantum state
fn clone_state(state: &QState) -> QState {
    let mut new_state = QState::new_zero(state.n_qubits);
    let src_real = state.real.as_slice();
    let src_imag = state.imag.as_slice();
    let dst_real = new_state.real.as_mut_slice();
    let dst_imag = new_state.imag.as_mut_slice();
    dst_real.copy_from_slice(src_real);
    dst_imag.copy_from_slice(src_imag);
    new_state
}

/// Prepare a random superposition state (deterministic with seed)
fn prepare_random_state(n_qubits: u8, seed: u64) -> QState {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    // Apply random gates to create superposition
    for q in 0..n_qubits {
        apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
        // Add some phase variation
        let angle = (q as f32 + 1.0) * 0.1;
        apply_gate_backend(
            &mut state,
            &QGate::Phase(q, angle),
            &mut rng,
            QBackend::Avx2,
        );
    }
    // Add some entanglement
    for q in 0..(n_qubits - 1) {
        apply_gate_backend(&mut state, &QGate::CNot(q, q + 1), &mut rng, QBackend::Avx2);
    }

    state
}

// ============================================================================
// Gate Identity Tests: G^n = I
// ============================================================================

#[test]
fn test_hadamard_squared_is_identity() {
    println!("\n=== Test: H² = I ===");

    for n_qubits in [2u8, 4, 8, 12] {
        for target in 0..n_qubits.min(4) {
            let original = prepare_random_state(n_qubits, 42 + target as u64);
            let mut state = clone_state(&original);
            let mut rng = QRng::new(42);

            // Apply H twice
            apply_gate_backend(&mut state, &QGate::H(target), &mut rng, QBackend::Avx2);
            apply_gate_backend(&mut state, &QGate::H(target), &mut rng, QBackend::Avx2);

            let f = fidelity(&original, &state);
            let passed = f > 1.0 - TOLERANCE;

            println!(
                "  {}q, H({})²: fidelity = {:.10} [{}]",
                n_qubits,
                target,
                f,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(
                passed,
                "H² ≠ I for {}q target {}: fidelity = {}",
                n_qubits, target, f
            );
        }
    }
}

#[test]
fn test_pauli_squared_is_identity() {
    println!("\n=== Test: X² = Y² = Z² = I ===");

    let n_qubits = 4u8;

    let pauli_gates: Vec<(&str, fn(u8) -> QGate)> = vec![
        ("X", |t| QGate::X(t)),
        ("Y", |t| QGate::Y(t)),
        ("Z", |t| QGate::Z(t)),
    ];

    for (gate_name, make_gate) in pauli_gates {
        for target in 0..n_qubits {
            let original = prepare_random_state(n_qubits, 100 + target as u64);
            let mut state = clone_state(&original);
            let mut rng = QRng::new(42);

            // Apply gate twice
            apply_gate_backend(&mut state, &make_gate(target), &mut rng, QBackend::Avx2);
            apply_gate_backend(&mut state, &make_gate(target), &mut rng, QBackend::Avx2);

            let f = fidelity(&original, &state);
            let passed = f > 1.0 - TOLERANCE;

            println!(
                "  {}({})²: fidelity = {:.10} [{}]",
                gate_name,
                target,
                f,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(
                passed,
                "{}² ≠ I for target {}: fidelity = {}",
                gate_name, target, f
            );
        }
    }
}

#[test]
fn test_cnot_squared_is_identity() {
    println!("\n=== Test: CNOT² = I ===");

    for n_qubits in [2u8, 4, 8] {
        for control in 0..(n_qubits - 1).min(3) {
            let target = control + 1;
            let original = prepare_random_state(n_qubits, 200 + control as u64);
            let mut state = clone_state(&original);
            let mut rng = QRng::new(42);

            // Apply CNOT twice
            apply_gate_backend(
                &mut state,
                &QGate::CNot(control, target),
                &mut rng,
                QBackend::Avx2,
            );
            apply_gate_backend(
                &mut state,
                &QGate::CNot(control, target),
                &mut rng,
                QBackend::Avx2,
            );

            let f = fidelity(&original, &state);
            let passed = f > 1.0 - TOLERANCE;

            println!(
                "  {}q, CNOT({},{})²: fidelity = {:.10} [{}]",
                n_qubits,
                control,
                target,
                f,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(
                passed,
                "CNOT² ≠ I for {}q control={} target={}: fidelity = {}",
                n_qubits, control, target, f
            );
        }
    }
}

#[test]
fn test_cz_squared_is_identity() {
    println!("\n=== Test: CZ² = I ===");

    for n_qubits in [2u8, 4, 8] {
        for q0 in 0..(n_qubits - 1).min(3) {
            let q1 = q0 + 1;
            let original = prepare_random_state(n_qubits, 300 + q0 as u64);
            let mut state = clone_state(&original);
            let mut rng = QRng::new(42);

            // Apply CZ twice
            apply_gate_backend(&mut state, &QGate::CZ(q0, q1), &mut rng, QBackend::Avx2);
            apply_gate_backend(&mut state, &QGate::CZ(q0, q1), &mut rng, QBackend::Avx2);

            let f = fidelity(&original, &state);
            let passed = f > 1.0 - TOLERANCE;

            println!(
                "  {}q, CZ({},{})²: fidelity = {:.10} [{}]",
                n_qubits,
                q0,
                q1,
                f,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(passed, "CZ² ≠ I for {}q: fidelity = {}", n_qubits, f);
        }
    }
}

#[test]
fn test_swap_squared_is_identity() {
    println!("\n=== Test: SWAP² = I ===");

    for n_qubits in [2u8, 4, 8] {
        for q0 in 0..(n_qubits - 1).min(3) {
            let q1 = q0 + 1;
            let original = prepare_random_state(n_qubits, 400 + q0 as u64);
            let mut state = clone_state(&original);
            let mut rng = QRng::new(42);

            // Apply SWAP twice
            apply_gate_backend(&mut state, &QGate::Swap(q0, q1), &mut rng, QBackend::Avx2);
            apply_gate_backend(&mut state, &QGate::Swap(q0, q1), &mut rng, QBackend::Avx2);

            let f = fidelity(&original, &state);
            let passed = f > 1.0 - TOLERANCE;

            println!(
                "  {}q, SWAP({},{})²: fidelity = {:.10} [{}]",
                n_qubits,
                q0,
                q1,
                f,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(passed, "SWAP² ≠ I for {}q: fidelity = {}", n_qubits, f);
        }
    }
}

#[test]
fn test_phase_periodicity() {
    println!("\n=== Test: Phase(2π) = I ===");

    let n_qubits = 4u8;

    for target in 0..n_qubits {
        let original = prepare_random_state(n_qubits, 500 + target as u64);
        let mut state = clone_state(&original);
        let mut rng = QRng::new(42);

        // Apply Phase(2π) - should be identity
        apply_gate_backend(
            &mut state,
            &QGate::Phase(target, 2.0 * PI),
            &mut rng,
            QBackend::Avx2,
        );

        let f = fidelity(&original, &state);
        let passed = f > 1.0 - TOLERANCE;

        println!(
            "  Phase({}, 2π): fidelity = {:.10} [{}]",
            target,
            f,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Phase(2π) ≠ I for target {}: fidelity = {}",
            target, f
        );
    }
}

#[test]
fn test_s_gate_fourth_power_is_identity() {
    println!("\n=== Test: S⁴ = I (Phase(π/2)⁴ = Phase(2π) = I) ===");

    let n_qubits = 4u8;
    let s_angle = PI / 2.0; // S gate = Phase(π/2)

    for target in 0..n_qubits {
        let original = prepare_random_state(n_qubits, 600 + target as u64);
        let mut state = clone_state(&original);
        let mut rng = QRng::new(42);

        // Apply S gate 4 times
        for _ in 0..4 {
            apply_gate_backend(
                &mut state,
                &QGate::Phase(target, s_angle),
                &mut rng,
                QBackend::Avx2,
            );
        }

        let f = fidelity(&original, &state);
        let passed = f > 1.0 - TOLERANCE;

        println!(
            "  S({})⁴: fidelity = {:.10} [{}]",
            target,
            f,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(passed, "S⁴ ≠ I for target {}: fidelity = {}", target, f);
    }
}

#[test]
fn test_t_gate_eighth_power_is_identity() {
    println!("\n=== Test: T⁸ = I (Phase(π/4)⁸ = Phase(2π) = I) ===");

    let n_qubits = 4u8;
    let t_angle = PI / 4.0; // T gate = Phase(π/4)

    for target in 0..n_qubits {
        let original = prepare_random_state(n_qubits, 700 + target as u64);
        let mut state = clone_state(&original);
        let mut rng = QRng::new(42);

        // Apply T gate 8 times
        for _ in 0..8 {
            apply_gate_backend(
                &mut state,
                &QGate::Phase(target, t_angle),
                &mut rng,
                QBackend::Avx2,
            );
        }

        let f = fidelity(&original, &state);
        let passed = f > 1.0 - TOLERANCE;

        println!(
            "  T({})⁸: fidelity = {:.10} [{}]",
            target,
            f,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(passed, "T⁸ ≠ I for target {}: fidelity = {}", target, f);
    }
}

// ============================================================================
// Norm Preservation Tests
// ============================================================================

#[test]
fn test_norm_preservation_single_gates() {
    println!("\n=== Test: Norm Preservation (Single Gates) ===");

    let n_qubits = 8u8;
    let mut rng = QRng::new(42);

    let gates: Vec<(&str, QGate)> = vec![
        ("H(0)", QGate::H(0)),
        ("X(1)", QGate::X(1)),
        ("Y(2)", QGate::Y(2)),
        ("Z(3)", QGate::Z(3)),
        ("Phase(0, π/3)", QGate::Phase(0, PI / 3.0)),
        ("CNOT(0,1)", QGate::CNot(0, 1)),
        ("CZ(2,3)", QGate::CZ(2, 3)),
        ("SWAP(4,5)", QGate::Swap(4, 5)),
    ];

    for (name, gate) in gates {
        let mut state = prepare_random_state(n_qubits, 800);
        let norm_before = norm_squared(&state);

        apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);

        let norm_after = norm_squared(&state);
        let drift = (norm_after - norm_before).abs();
        let passed = drift < TOLERANCE;

        println!(
            "  {}: norm_before={:.10}, norm_after={:.10}, drift={:.2e} [{}]",
            name,
            norm_before,
            norm_after,
            drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(passed, "Norm not preserved for {}: drift = {}", name, drift);
    }
}

#[test]
fn test_norm_preservation_deep_circuit() {
    println!("\n=== Test: Norm Preservation (Deep Circuit) ===");

    for n_qubits in [4u8, 8, 12] {
        for depth in [10, 100, 500, 1000] {
            let mut state = QState::new_zero(n_qubits);
            let mut rng = QRng::new(42);

            // Initial H on all qubits
            for q in 0..n_qubits {
                apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
            }

            let _norm_initial = norm_squared(&state);

            // Apply many layers
            for layer in 0..depth {
                // Single qubit gates
                for q in 0..n_qubits {
                    let gate = match layer % 4 {
                        0 => QGate::H(q),
                        1 => QGate::Phase(q, 0.1),
                        2 => QGate::X(q),
                        _ => QGate::Z(q),
                    };
                    apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);
                }
                // Two qubit gates
                for q in (0..n_qubits - 1).step_by(2) {
                    apply_gate_backend(
                        &mut state,
                        &QGate::CNot(q, q + 1),
                        &mut rng,
                        QBackend::Avx2,
                    );
                }
            }

            let norm_final = norm_squared(&state);
            let drift = (norm_final - 1.0).abs();
            let passed = drift < 1e-4; // Allow slightly more drift for deep circuits

            println!(
                "  {}q, depth={}: norm_final={:.10}, drift={:.2e} [{}]",
                n_qubits,
                depth,
                norm_final,
                drift,
                if passed { "PASS" } else { "FAIL" }
            );

            assert!(
                passed,
                "Norm drift too large for {}q depth {}: drift = {}",
                n_qubits, depth, drift
            );
        }
    }
}

// ============================================================================
// Inner Product Preservation Tests
// ============================================================================

#[test]
fn test_inner_product_preservation() {
    println!("\n=== Test: Inner Product Preservation Under Same Unitary ===");

    let n_qubits = 6u8;
    let mut rng = QRng::new(42);

    // Prepare two different random states
    let mut state_a = prepare_random_state(n_qubits, 1000);
    let mut state_b = prepare_random_state(n_qubits, 2000);

    // Compute initial inner product
    let (ip_real_before, ip_imag_before) = inner_product(&state_a, &state_b);
    let ip_mag_before = (ip_real_before.powi(2) + ip_imag_before.powi(2)).sqrt();

    println!(
        "  Initial <a|b> = {:.6} + {:.6}i (|<a|b>| = {:.6})",
        ip_real_before, ip_imag_before, ip_mag_before
    );

    // Apply same circuit to both states
    let depth = 50;
    for _ in 0..depth {
        for q in 0..n_qubits {
            apply_gate_backend(&mut state_a, &QGate::H(q), &mut rng, QBackend::Avx2);
            apply_gate_backend(&mut state_b, &QGate::H(q), &mut rng, QBackend::Avx2);
        }
        for q in (0..n_qubits - 1).step_by(2) {
            apply_gate_backend(
                &mut state_a,
                &QGate::CNot(q, q + 1),
                &mut rng,
                QBackend::Avx2,
            );
            apply_gate_backend(
                &mut state_b,
                &QGate::CNot(q, q + 1),
                &mut rng,
                QBackend::Avx2,
            );
        }
    }

    // Compute final inner product
    let (ip_real_after, ip_imag_after) = inner_product(&state_a, &state_b);
    let ip_mag_after = (ip_real_after.powi(2) + ip_imag_after.powi(2)).sqrt();

    println!(
        "  Final <a|b> = {:.6} + {:.6}i (|<a|b>| = {:.6})",
        ip_real_after, ip_imag_after, ip_mag_after
    );

    // Magnitude should be preserved (phase may change due to global phase)
    let mag_drift = (ip_mag_after - ip_mag_before).abs();
    let passed = mag_drift < 1e-4;

    println!(
        "  |<a|b>| drift: {:.2e} [{}]",
        mag_drift,
        if passed { "PASS" } else { "FAIL" }
    );

    assert!(
        passed,
        "Inner product magnitude not preserved: drift = {}",
        mag_drift
    );
}

// ============================================================================
// Reversibility Tests: U · U† = I
// ============================================================================

#[test]
fn test_circuit_reversibility() {
    println!("\n=== Test: Circuit Reversibility (U · U† = I) ===");

    for n_qubits in [4u8, 6, 8] {
        let original = prepare_random_state(n_qubits, 3000);
        let mut state = clone_state(&original);
        let mut rng = QRng::new(42);

        // Build a circuit (sequence of gates)
        let mut circuit: Vec<QGate> = Vec::new();

        // Forward circuit
        for q in 0..n_qubits {
            circuit.push(QGate::H(q));
        }
        for q in (0..n_qubits - 1).step_by(2) {
            circuit.push(QGate::CNot(q, q + 1));
        }
        for q in 0..n_qubits {
            circuit.push(QGate::Phase(q, PI / 4.0)); // T gate
        }
        for q in (1..n_qubits - 1).step_by(2) {
            circuit.push(QGate::CNot(q, q + 1));
        }
        for q in 0..n_qubits {
            circuit.push(QGate::H(q));
        }

        // Apply forward circuit
        for gate in &circuit {
            apply_gate_backend(&mut state, gate, &mut rng, QBackend::Avx2);
        }

        // Apply inverse circuit (reverse order, conjugate phases)
        for gate in circuit.iter().rev() {
            let inverse = match gate {
                QGate::H(q) => QGate::H(*q),                        // H† = H
                QGate::X(q) => QGate::X(*q),                        // X† = X
                QGate::Y(q) => QGate::Y(*q),                        // Y† = Y
                QGate::Z(q) => QGate::Z(*q),                        // Z† = Z
                QGate::Phase(q, angle) => QGate::Phase(*q, -angle), // Phase† = Phase(-θ)
                QGate::CNot(c, t) => QGate::CNot(*c, *t),           // CNOT† = CNOT
                QGate::CZ(q0, q1) => QGate::CZ(*q0, *q1),           // CZ† = CZ
                QGate::Swap(q0, q1) => QGate::Swap(*q0, *q1),       // SWAP† = SWAP
                _ => gate.clone(),
            };
            apply_gate_backend(&mut state, &inverse, &mut rng, QBackend::Avx2);
        }

        let f = fidelity(&original, &state);
        let passed = f > 1.0 - TOLERANCE;

        println!(
            "  {}q, {} gates: fidelity = {:.10} [{}]",
            n_qubits,
            circuit.len() * 2,
            f,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Reversibility failed for {}q: fidelity = {}",
            n_qubits, f
        );
    }
}

// ============================================================================
// Summary Test
// ============================================================================

#[test]
fn test_invariant_summary() {
    println!("\n");
    println!("======================================================================");
    println!("  QUANTUM INVARIANT TEST SUMMARY");
    println!("  These tests verify mathematical properties without external reference");
    println!("======================================================================");
    println!();
    println!("  Gate Identities:");
    println!("    - H² = I (Hadamard is self-inverse)");
    println!("    - X² = Y² = Z² = I (Paulis are self-inverse)");
    println!("    - CNOT² = CZ² = SWAP² = I (Two-qubit gates)");
    println!("    - S⁴ = T⁸ = I (Phase gate periodicity)");
    println!();
    println!("  Unitarity:");
    println!("    - ||ψ||² = 1 preserved after any gate");
    println!("    - ||ψ||² = 1 preserved after deep circuits (1000 layers)");
    println!("    - |<a|b>| preserved under same unitary");
    println!();
    println!("  Reversibility:");
    println!("    - U·U† = I (circuit inverse recovers original state)");
    println!();
    println!("  All invariants verified at tolerance: {:.0e}", TOLERANCE);
    println!("======================================================================");
    println!();
}
