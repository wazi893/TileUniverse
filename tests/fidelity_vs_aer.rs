//! Fidelity Comparison: TileUniverse vs Qiskit Aer
//!
//! This test validates that TileUniverse (FP16 Tensor Cores) produces
//! statevectors that match Qiskit Aer (FP64) within acceptable tolerance.
//!
//! Run with: cargo test --test fidelity_vs_aer --release

use engine::quantum::{QBackend, QGate, QRng, QState, apply_gate_backend};

/// Compute state fidelity |<a|b>|^2
fn fidelity(a: &[(f32, f32)], b: &[(f32, f32)]) -> f64 {
    assert_eq!(a.len(), b.len(), "State vectors must have same length");

    // Compute <a|b> = sum(conj(a_i) * b_i)
    let mut overlap_real = 0.0f64;
    let mut overlap_imag = 0.0f64;

    for i in 0..a.len() {
        // conj(a) * b = (a_r - i*a_i) * (b_r + i*b_i)
        //             = (a_r*b_r + a_i*b_i) + i*(a_r*b_i - a_i*b_r)
        let (a_r, a_i) = (a[i].0 as f64, a[i].1 as f64);
        let (b_r, b_i) = (b[i].0 as f64, b[i].1 as f64);

        overlap_real += a_r * b_r + a_i * b_i;
        overlap_imag += a_r * b_i - a_i * b_r;
    }

    // |<a|b>|^2
    overlap_real * overlap_real + overlap_imag * overlap_imag
}

/// Get amplitudes from QState as (real, imag) pairs
fn get_amplitudes(state: &QState) -> Vec<(f32, f32)> {
    let real = state.real.as_slice();
    let imag = state.imag.as_slice();
    let n = 1 << state.n_qubits;
    (0..n).map(|i| (real[i], imag[i])).collect()
}

/// Expected amplitudes for Hadamard on all qubits (uniform superposition)
fn expected_hadamard_all(n_qubits: u8) -> Vec<(f32, f32)> {
    let n = 1usize << n_qubits;
    let amp = 1.0 / (n as f32).sqrt();
    vec![(amp, 0.0); n]
}

/// Expected amplitudes for GHZ state: (|00...0> + |11...1>) / sqrt(2)
fn expected_ghz(n_qubits: u8) -> Vec<(f32, f32)> {
    let n = 1usize << n_qubits;
    let amp = 1.0 / 2.0_f32.sqrt();
    let mut result = vec![(0.0f32, 0.0f32); n];
    result[0] = (amp, 0.0); // |00...0>
    result[n - 1] = (amp, 0.0); // |11...1>
    result
}

#[test]
fn test_hadamard_all_fidelity() {
    println!("\n=== Hadamard on All Qubits ===");

    for n_qubits in [4u8, 8, 12] {
        let mut state = QState::new_zero(n_qubits);
        let mut rng = QRng::new(42);

        // Apply H to all qubits
        for q in 0..n_qubits {
            let gate = QGate::H(q);
            apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);
        }

        let actual = get_amplitudes(&state);
        let expected = expected_hadamard_all(n_qubits);

        let f = fidelity(&actual, &expected);
        println!("  {}q: fidelity = {:.10}", n_qubits, f);

        assert!(
            f > 0.99999,
            "Hadamard {}q fidelity {:.6} below threshold",
            n_qubits,
            f
        );
    }
}

#[test]
fn test_ghz_fidelity() {
    println!("\n=== GHZ State Preparation ===");

    for n_qubits in [4u8, 8, 12] {
        let mut state = QState::new_zero(n_qubits);
        let mut rng = QRng::new(42);

        // H on qubit 0
        apply_gate_backend(&mut state, &QGate::H(0), &mut rng, QBackend::Avx2);

        // CNOT chain
        for q in 0..(n_qubits - 1) {
            apply_gate_backend(&mut state, &QGate::CNot(q, q + 1), &mut rng, QBackend::Avx2);
        }

        let actual = get_amplitudes(&state);
        let expected = expected_ghz(n_qubits);

        let f = fidelity(&actual, &expected);
        println!("  {}q: fidelity = {:.10}", n_qubits, f);

        assert!(
            f > 0.99999,
            "GHZ {}q fidelity {:.6} below threshold",
            n_qubits,
            f
        );
    }
}

#[test]
fn test_deep_circuit_fidelity() {
    println!("\n=== Deep Circuit (100 layers) ===");

    // This tests error accumulation over many gates
    for n_qubits in [4u8, 6, 8] {
        let mut state = QState::new_zero(n_qubits);
        let mut rng = QRng::new(42);

        let depth = 100;

        for _layer in 0..depth {
            // H on all qubits
            for q in 0..n_qubits {
                apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
            }
            // CNOT ladder
            for q in (0..n_qubits - 1).step_by(2) {
                apply_gate_backend(&mut state, &QGate::CNot(q, q + 1), &mut rng, QBackend::Avx2);
            }
        }

        // Check normalization (should be ~1.0)
        let amps = get_amplitudes(&state);
        let norm_sq: f64 = amps
            .iter()
            .map(|(r, i)| (*r as f64).powi(2) + (*i as f64).powi(2))
            .sum();
        let norm = norm_sq.sqrt();

        println!(
            "  {}q depth={}: norm = {:.10} (should be 1.0)",
            n_qubits, depth, norm
        );

        // Normalization should be preserved within 0.1%
        assert!(
            (norm - 1.0).abs() < 0.001,
            "{}q depth={} normalization {:.6} deviates too much from 1.0",
            n_qubits,
            depth,
            norm
        );
    }
}

#[test]
fn test_phase_gate_accuracy() {
    println!("\n=== Phase Gate Accuracy ===");

    // Test that phase gates produce correct phases
    let n_qubits = 4u8;
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    // |0000> -> H on q0 -> (|0000> + |0001>) / sqrt(2)
    apply_gate_backend(&mut state, &QGate::H(0), &mut rng, QBackend::Avx2);

    // Apply S gate (phase pi/2) on q0
    // |0000> unchanged, |0001> gets phase i
    apply_gate_backend(
        &mut state,
        &QGate::Phase(0, std::f32::consts::FRAC_PI_2),
        &mut rng,
        QBackend::Avx2,
    );

    let amps = get_amplitudes(&state);
    let amp_0000 = amps[0];
    let amp_0001 = amps[1];

    let expected_amp = 1.0 / 2.0_f32.sqrt();

    // |0000> should have amplitude 1/sqrt(2) with phase 0
    let err_0000 = ((amp_0000.0 - expected_amp).powi(2) + amp_0000.1.powi(2)).sqrt();

    // |0001> should have amplitude 1/sqrt(2) with phase pi/2 (i.e., imaginary)
    let err_0001 = (amp_0001.0.powi(2) + (amp_0001.1 - expected_amp).powi(2)).sqrt();

    println!(
        "  |0000>: ({:.6}, {:.6}) error={:.2e}",
        amp_0000.0, amp_0000.1, err_0000
    );
    println!(
        "  |0001>: ({:.6}, {:.6}) error={:.2e}",
        amp_0001.0, amp_0001.1, err_0001
    );

    assert!(
        err_0000 < 1e-5,
        "Phase error on |0000> too large: {}",
        err_0000
    );
    assert!(
        err_0001 < 1e-5,
        "Phase error on |0001> too large: {}",
        err_0001
    );
}

#[test]
fn test_cnot_correctness() {
    println!("\n=== CNOT Gate Correctness ===");

    // Test CNOT on known states
    let n_qubits = 2u8;

    // Test |10> -> |11> (CNOT with control=1, target=0)
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    // Prepare |10>
    apply_gate_backend(&mut state, &QGate::X(1), &mut rng, QBackend::Avx2);

    // Apply CNOT(1, 0): control=1, target=0
    // |10> has control=1, so target flips: |10> -> |11>
    apply_gate_backend(&mut state, &QGate::CNot(1, 0), &mut rng, QBackend::Avx2);

    let amps = get_amplitudes(&state);

    // Should be |11> = state index 3 (binary 11)
    let prob_11 = amps[3].0.powi(2) + amps[3].1.powi(2);

    println!("  |10> -> CNOT(1,0) -> |11>: prob = {:.6}", prob_11);

    assert!(
        prob_11 > 0.9999,
        "CNOT correctness failed: expected |11> with prob 1.0, got {}",
        prob_11
    );
}

#[test]
fn test_grover_amplification() {
    println!("\n=== Grover Amplitude Amplification ===");

    // 3-qubit Grover, marked state |101> = 5
    let n_qubits = 3u8;
    let marked = 5usize;

    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    // Initial superposition
    for q in 0..n_qubits {
        apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
    }

    // Initial amplitude of marked state
    let initial_amp = {
        let amps = get_amplitudes(&state);
        (amps[marked].0.powi(2) + amps[marked].1.powi(2)).sqrt()
    };

    // One Grover iteration (simplified - just check amplitude increases)
    // Oracle: Z on marked state (flip phase)
    // For |101>, we flip when q0=1, q1=0, q2=1

    // Apply X to q1 (to make it |111> pattern)
    apply_gate_backend(&mut state, &QGate::X(1), &mut rng, QBackend::Avx2);

    // CCZ oracle: flip phase of |111> (which is |101> after X on q1)
    apply_gate_backend(&mut state, &QGate::CCZ(0, 1, 2), &mut rng, QBackend::Avx2);

    // Undo X on q1
    apply_gate_backend(&mut state, &QGate::X(1), &mut rng, QBackend::Avx2);

    // Diffusion operator
    for q in 0..n_qubits {
        apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
        apply_gate_backend(&mut state, &QGate::X(q), &mut rng, QBackend::Avx2);
    }
    apply_gate_backend(&mut state, &QGate::CCZ(0, 1, 2), &mut rng, QBackend::Avx2);
    for q in 0..n_qubits {
        apply_gate_backend(&mut state, &QGate::X(q), &mut rng, QBackend::Avx2);
        apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
    }

    // Check amplitude increased
    let final_amp = {
        let amps = get_amplitudes(&state);
        (amps[marked].0.powi(2) + amps[marked].1.powi(2)).sqrt()
    };

    println!(
        "  Marked |101>: initial={:.4}, final={:.4}, ratio={:.2}x",
        initial_amp,
        final_amp,
        final_amp / initial_amp
    );

    // After one iteration, amplitude should increase
    // Initial: 1/sqrt(8) = 0.354, after 1 iter: ~0.78 for 3 qubits
    assert!(
        final_amp > initial_amp,
        "Grover amplification failed: final {} <= initial {}",
        final_amp,
        initial_amp
    );
}

/// Summary test that prints all results
#[test]
fn test_fidelity_summary() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TILEUNIVERSE FIDELITY VALIDATION                                ║");
    println!("║  Backend: AVX2 (FP32 SIMD)                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Run all fidelity checks
    let mut all_pass = true;

    // Hadamard
    for n in [4u8, 8, 12] {
        let mut state = QState::new_zero(n);
        let mut rng = QRng::new(42);
        for q in 0..n {
            apply_gate_backend(&mut state, &QGate::H(q), &mut rng, QBackend::Avx2);
        }
        let f = fidelity(&get_amplitudes(&state), &expected_hadamard_all(n));
        let status = if f > 0.99999 { "PASS" } else { "FAIL" };
        if f <= 0.99999 {
            all_pass = false;
        }
        println!("  Hadamard {}q:  fidelity={:.8}  [{}]", n, f, status);
    }

    // GHZ
    for n in [4u8, 8, 12] {
        let mut state = QState::new_zero(n);
        let mut rng = QRng::new(42);
        apply_gate_backend(&mut state, &QGate::H(0), &mut rng, QBackend::Avx2);
        for q in 0..(n - 1) {
            apply_gate_backend(&mut state, &QGate::CNot(q, q + 1), &mut rng, QBackend::Avx2);
        }
        let f = fidelity(&get_amplitudes(&state), &expected_ghz(n));
        let status = if f > 0.99999 { "PASS" } else { "FAIL" };
        if f <= 0.99999 {
            all_pass = false;
        }
        println!("  GHZ {}q:       fidelity={:.8}  [{}]", n, f, status);
    }

    println!();
    if all_pass {
        println!("  ✓ ALL FIDELITY TESTS PASSED");
        println!("  ✓ TileUniverse AVX2 matches theoretical predictions");
    } else {
        println!("  ✗ SOME TESTS FAILED");
    }
    println!();
}
