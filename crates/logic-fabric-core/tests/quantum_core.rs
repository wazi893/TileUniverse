//! Integration test gate for the `logic-fabric-core` public API.
//!
//! This crate shipped without an integration `tests/` gate; the inline unit
//! tests live next to the code they cover, but nothing exercises the crate the
//! way a downstream consumer does (through the re-exported surface). These
//! tests drive only the public API — `QState`, `QGate`, `apply_gate_scalar`,
//! `QRng` (from `quantum`) and the re-exported `Fixed8` / `Complex8` — so they
//! double as a protective regression gate and a usage reference.
//!
//! CPU-only (scalar backend); no CUDA / GPU paths are touched.

use logic_fabric_core::quantum::{apply_gate_scalar, Complex32, GateOutcome, QGate, QRng, QState};
use logic_fabric_core::{Complex8, Fixed8};

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Generous tolerance for f32 amplitude comparisons after a few scalar gates.
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

/// Sum of squared magnitudes — must stay ~1 for any physical pure state.
fn total_probability(s: &QState) -> f32 {
    let re = s.real.as_slice();
    let im = s.imag.as_slice();
    (0..s.len).map(|i| re[i] * re[i] + im[i] * im[i]).sum()
}

// ---------------------------------------------------------------------------
// QState construction
// ---------------------------------------------------------------------------

#[test]
fn new_zero_is_computational_basis_state() {
    for n in 1u8..=4 {
        let s = QState::new_zero(n);
        assert_eq!(s.n_qubits, n);
        assert_eq!(s.len, 1usize << n);
        assert!(approx(s.real.as_slice()[0], 1.0));
        for i in 1..s.len {
            assert!(approx(s.real.as_slice()[i], 0.0));
            assert!(approx(s.imag.as_slice()[i], 0.0));
        }
        assert!(approx(total_probability(&s), 1.0));
    }
}

#[test]
fn interleaved_roundtrip_preserves_amplitudes() {
    let amps = [
        Complex32::new(0.5, 0.0),
        Complex32::new(0.0, 0.5),
        Complex32::new(-0.5, 0.0),
        Complex32::new(0.0, -0.5),
    ];
    let s = QState::from_interleaved(2, &amps);
    let back = s.to_interleaved();
    assert_eq!(back.len(), amps.len());
    for (got, want) in back.iter().zip(amps.iter()) {
        assert!(approx(got.re, want.re));
        assert!(approx(got.im, want.im));
    }
}

// ---------------------------------------------------------------------------
// Single-qubit gates
// ---------------------------------------------------------------------------

#[test]
fn x_gate_flips_and_is_involution() {
    let mut s = QState::new_zero(1);
    let mut rng = QRng::new(1);
    apply_gate_scalar(&mut s, &QGate::X(0), &mut rng);
    assert!(approx(s.real.as_slice()[0], 0.0));
    assert!(approx(s.real.as_slice()[1], 1.0));
    // X·X = I
    apply_gate_scalar(&mut s, &QGate::X(0), &mut rng);
    assert!(approx(s.real.as_slice()[0], 1.0));
    assert!(approx(s.real.as_slice()[1], 0.0));
}

#[test]
fn hadamard_superposition_and_involution() {
    let mut s = QState::new_zero(1);
    let mut rng = QRng::new(7);
    apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
    assert!(approx(s.real.as_slice()[0], INV_SQRT2));
    assert!(approx(s.real.as_slice()[1], INV_SQRT2));
    assert!(approx(total_probability(&s), 1.0));
    // H·H = I returns to |0>
    apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
    assert!(approx(s.real.as_slice()[0], 1.0));
    assert!(approx(s.real.as_slice()[1], 0.0));
}

#[test]
fn z_gate_flips_phase_of_one() {
    let mut s = QState::new_zero(1);
    let mut rng = QRng::new(3);
    // Prepare |1>, then Z should send it to -|1>.
    apply_gate_scalar(&mut s, &QGate::X(0), &mut rng);
    apply_gate_scalar(&mut s, &QGate::Z(0), &mut rng);
    assert!(approx(s.real.as_slice()[1], -1.0));
    // Z leaves |0> untouched.
    let mut s0 = QState::new_zero(1);
    apply_gate_scalar(&mut s0, &QGate::Z(0), &mut rng);
    assert!(approx(s0.real.as_slice()[0], 1.0));
}

#[test]
fn y_gate_maps_zero_to_i_one() {
    // Y|0> = i|1>: real parts vanish, imag of |1> is +1.
    let mut s = QState::new_zero(1);
    let mut rng = QRng::new(9);
    apply_gate_scalar(&mut s, &QGate::Y(0), &mut rng);
    assert!(approx(s.real.as_slice()[0], 0.0));
    assert!(approx(s.imag.as_slice()[0], 0.0));
    assert!(approx(s.real.as_slice()[1], 0.0));
    assert!(approx(s.imag.as_slice()[1], 1.0));
}

// ---------------------------------------------------------------------------
// Multi-qubit gates / entanglement
// ---------------------------------------------------------------------------

#[test]
fn bell_state_is_entangled() {
    let mut s = QState::new_zero(2);
    let mut rng = QRng::new(42);
    apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
    apply_gate_scalar(&mut s, &QGate::CNot(0, 1), &mut rng);
    // (|00> + |11>)/√2
    assert!(approx(s.real.as_slice()[0], INV_SQRT2));
    assert!(approx(s.real.as_slice()[3], INV_SQRT2));
    assert!(approx(s.real.as_slice()[1], 0.0));
    assert!(approx(s.real.as_slice()[2], 0.0));
    assert!(approx(total_probability(&s), 1.0));
}

#[test]
fn ghz3_state() {
    let mut s = QState::new_zero(3);
    let mut rng = QRng::new(11);
    apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
    apply_gate_scalar(&mut s, &QGate::CNot(0, 1), &mut rng);
    apply_gate_scalar(&mut s, &QGate::CNot(1, 2), &mut rng);
    // (|000> + |111>)/√2 — only indices 0 and 7 populated.
    assert!(approx(s.real.as_slice()[0], INV_SQRT2));
    assert!(approx(s.real.as_slice()[7], INV_SQRT2));
    for i in 1..7 {
        assert!(approx(s.real.as_slice()[i], 0.0));
        assert!(approx(s.imag.as_slice()[i], 0.0));
    }
    assert!(approx(total_probability(&s), 1.0));
}

#[test]
fn cnot_with_control_zero_is_identity() {
    // Control |0> ⇒ target unchanged.
    let mut s = QState::new_zero(2);
    let mut rng = QRng::new(5);
    apply_gate_scalar(&mut s, &QGate::CNot(0, 1), &mut rng);
    assert!(approx(s.real.as_slice()[0], 1.0));
    for i in 1..s.len {
        assert!(approx(s.real.as_slice()[i], 0.0));
    }
}

#[test]
fn swap_exchanges_qubits() {
    // Put qubit 0 into |1>, then SWAP(0,1) ⇒ qubit 1 holds the |1>.
    let mut s = QState::new_zero(2);
    let mut rng = QRng::new(8);
    apply_gate_scalar(&mut s, &QGate::X(0), &mut rng); // |01> (index 1)
    apply_gate_scalar(&mut s, &QGate::Swap(0, 1), &mut rng);
    // Now amplitude should sit at |10> (index 2).
    assert!(approx(s.real.as_slice()[2], 1.0));
    assert!(approx(s.real.as_slice()[1], 0.0));
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[test]
fn measuring_basis_state_is_deterministic() {
    // |0> measured on qubit 0 is always 0, for any RNG seed.
    for seed in [0u64, 1, 7, 123_456_789, u64::MAX] {
        let mut s = QState::new_zero(1);
        let mut rng = QRng::new(seed);
        let out = apply_gate_scalar(&mut s, &QGate::Measure(0), &mut rng);
        assert_eq!(out, GateOutcome::Measured { qubit: 0, bit: 0 });
        // Collapsed state stays normalized.
        assert!(approx(total_probability(&s), 1.0));
    }
}

#[test]
fn measurement_is_seed_reproducible_and_collapses() {
    let run = |seed: u64| -> (GateOutcome, QState) {
        let mut s = QState::new_zero(1);
        let mut rng = QRng::new(seed);
        apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
        let out = apply_gate_scalar(&mut s, &QGate::Measure(0), &mut rng);
        (out, s)
    };
    let (out_a, state_a) = run(55_555);
    let (out_b, state_b) = run(55_555);
    assert_eq!(out_a, out_b);
    // After collapse the state is a normalized basis state.
    assert!(approx(total_probability(&state_a), 1.0));
    let bit = match out_a {
        GateOutcome::Measured { bit, .. } => bit as usize,
        GateOutcome::None => panic!("Measure must report an outcome"),
    };
    assert!(approx(state_a.real.as_slice()[bit], 1.0));
    assert!(approx(state_a.real.as_slice()[1 - bit], 0.0));
    // Reproducible across runs.
    for i in 0..state_a.len {
        assert!(approx(
            state_a.real.as_slice()[i],
            state_b.real.as_slice()[i]
        ));
    }
}

#[test]
fn norm_preserved_through_unitary_circuit() {
    let mut s = QState::new_zero(3);
    let mut rng = QRng::new(2024);
    let circuit = [
        QGate::H(0),
        QGate::Phase(1, 0.3),
        QGate::Rx(2, 0.7),
        QGate::CNot(0, 1),
        QGate::Ry(0, 1.1),
        QGate::CZ(1, 2),
        QGate::T(2),
    ];
    for g in &circuit {
        apply_gate_scalar(&mut s, g, &mut rng);
    }
    assert!(approx(total_probability(&s), 1.0));
}

// ---------------------------------------------------------------------------
// Fixed-point primitives (re-exported Fixed8 / Complex8)
// ---------------------------------------------------------------------------

#[test]
fn fixed8_f32_roundtrip_and_arithmetic() {
    // Quantization step is 1/128, so mid-range values round-trip within it.
    for v in [-0.5f32, -0.25, 0.0, 0.25, 0.5, 0.703125] {
        let q = Fixed8::from_f32(v).to_f32();
        assert!((q - v).abs() <= 1.0 / 128.0, "{v} round-tripped to {q}");
    }
    // 0.25 + 0.25 = 0.5 exactly on the 1/128 grid.
    let a = Fixed8::from_f32(0.25);
    assert!((a.add(a).to_f32() - 0.5).abs() < 1e-6);
    // 0.5 - 0.25 = 0.25.
    let half = Fixed8::from_f32(0.5);
    assert!((half.sub(a).to_f32() - 0.25).abs() < 1e-6);
    // 0.5 * 0.5 = 0.25 (within one quantization step).
    assert!((half.mul(half).to_f32() - 0.25).abs() <= 1.0 / 128.0);
    // Negation is exact.
    assert!((a.neg().to_f32() + 0.25).abs() < 1e-6);
}

#[test]
fn complex8_arithmetic_conjugate_and_norm() {
    let z = Complex8::from_f32(0.5, 0.5);
    let w = Complex8::from_f32(0.25, -0.25);

    let (sr, si) = z.add(w).to_f32();
    assert!((sr - 0.75).abs() <= 1.0 / 128.0);
    assert!((si - 0.25).abs() <= 1.0 / 128.0);

    // Conjugate flips the imaginary sign, leaves real intact.
    let (cr, ci) = z.conj().to_f32();
    assert!((cr - 0.5).abs() < 1e-6);
    assert!((ci + 0.5).abs() < 1e-6);

    // |0.5 + 0.5i|² = 0.5 (within fixed-point precision).
    let n2 = z.norm2().to_f32();
    assert!((n2 - 0.5).abs() <= 2.0 / 128.0, "norm2 was {n2}");

    // i * i = -1: stays within the representable range (ONE ≈ 127/128).
    let prod = Complex8::I.mul(Complex8::I).to_f32();
    assert!((prod.0 + 1.0).abs() <= 2.0 / 128.0);
    assert!(prod.1.abs() < 1e-6);
}
