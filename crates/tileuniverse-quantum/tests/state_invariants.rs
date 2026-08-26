//! Invariant and cross-representation parity tests for the sparse state families.
//!
//! These guard the published API: GHZ endpoint representations (usize / BigUint /
//! symbolic) and the materialized W-state grid. The crate stores only the known
//! nonzero amplitudes, so the invariants under test are:
//!   - unit fidelity against the ideal state, independent of qubit count,
//!   - the two endpoint (or n excitation) amplitudes equal 1/sqrt(2) (or 1/sqrt(n)),
//!   - no spurious amplitude leaks into the unused slots,
//!   - serial and parallel construction/verification agree exactly.

use num_bigint::BigUint;
use tileuniverse_quantum::*;

const FIDELITY_EPS: f64 = 1e-10;
const AMP_EPS: f64 = 1e-4;
const PROB_EPS: f64 = 1e-9;

// === MinimalGhzState (usize qubit count) ===

#[test]
fn minimal_ghz_invariants_across_scales() {
    // Includes the small-n path (terminal addr = 2^n - 1, n < BLOCK_SHIFT) and
    // the saturated path (terminal addr = BLOCK_SIZE - 1).
    for &n in &[1usize, 2, 6, 7, 8, 128, 1_000, 1_000_000, 1_000_000_000] {
        let ghz = MinimalGhzState::new(n);
        assert_eq!(ghz.n_qubits(), n, "n_qubits accessor for n={n}");

        let v = ghz.verify();
        assert!(
            (v.fidelity - 1.0).abs() < FIDELITY_EPS,
            "fidelity for n={n}: {}",
            v.fidelity
        );
        assert!(
            (v.amp_zero_state - std::f64::consts::FRAC_1_SQRT_2).abs() < AMP_EPS,
            "|0..0> amp for n={n}: {}",
            v.amp_zero_state
        );
        assert_eq!(
            v.amp_zero_state, v.amp_one_state,
            "endpoint symmetry for n={n}"
        );
        assert_eq!(v.max_spurious, 0.0, "no spurious amplitude for n={n}");
    }
}

#[test]
fn minimal_ghz_memory_is_constant() {
    // O(1) representation: struct size does not depend on the qubit count.
    let small = MinimalGhzState::new(8).memory_bytes();
    let huge = MinimalGhzState::new(1_000_000_000).memory_bytes();
    assert_eq!(
        small, huge,
        "endpoint storage must be qubit-count independent"
    );
}

#[test]
fn create_minimal_ghz_matches_new() {
    let a = MinimalGhzState::new(12345);
    let b = create_minimal_ghz(12345);
    assert_eq!(a.n_qubits(), b.n_qubits());
    assert_eq!(a.verify().fidelity, b.verify().fidelity);
}

#[test]
#[should_panic(expected = "at least one qubit")]
fn minimal_ghz_rejects_zero_qubits() {
    let _ = MinimalGhzState::new(0);
}

// === UnlimitedGhzState (BigUint qubit count) ===

#[test]
fn unlimited_ghz_googol_has_unit_fidelity() {
    let googol = BigUint::from(10u32).pow(100);
    let v = UnlimitedGhzState::new(googol).verify();
    assert!(
        (v.fidelity - 1.0).abs() < FIDELITY_EPS,
        "fidelity: {}",
        v.fidelity
    );
}

#[test]
fn create_ghz_power_of_10_matches_explicit_biguint() {
    let explicit = UnlimitedGhzState::new(BigUint::from(10u32).pow(42));
    let helper = create_ghz_power_of_10(42);
    assert_eq!(explicit.n_qubits, helper.n_qubits);
    assert_eq!(explicit.verify().fidelity, helper.verify().fidelity);
}

#[test]
fn minimal_and_unlimited_agree_for_shared_counts() {
    // The two endpoint backends must produce the same verification for any count
    // representable in both (covers the small-n and saturated terminal-addr paths).
    for &n in &[3u64, 7, 200, 1_000_000] {
        let minimal = MinimalGhzState::new(n as usize).verify();
        let unlimited = UnlimitedGhzState::new(BigUint::from(n)).verify();
        assert_eq!(
            minimal.amp_zero_state, unlimited.amp_zero_state,
            "endpoint amp mismatch for n={n}"
        );
        assert_eq!(
            minimal.fidelity, unlimited.fidelity,
            "fidelity mismatch for n={n}"
        );
        assert_eq!(
            minimal.max_spurious, unlimited.max_spurious,
            "spurious mismatch for n={n}"
        );
    }
}

// === SparseQuantumGridVec (materialized W-state) ===

#[test]
fn w_state_block_geometry() {
    assert_eq!(SparseQuantumGridVec::new(7).n_blocks(), 1);
    assert_eq!(SparseQuantumGridVec::new(128).n_blocks(), 1);
    assert_eq!(SparseQuantumGridVec::new(129).n_blocks(), 2);
    assert_eq!(SparseQuantumGridVec::new(256).n_blocks(), 2);
    assert_eq!(SparseQuantumGridVec::new(257).n_blocks(), 3);
}

#[test]
#[should_panic(expected = "Minimum 7 qubits")]
fn w_state_rejects_too_few_qubits() {
    let _ = SparseQuantumGridVec::new(6);
}

#[test]
fn w_state_creation_invariants() {
    for &n in &[7usize, 10, 100, 128, 300, 10_000] {
        let mut grid = SparseQuantumGridVec::new(n);
        grid.create_w_state();
        let v = grid.verify_w_fidelity();

        assert_eq!(v.correct_amplitudes, n, "correct amplitudes for n={n}");
        assert!(
            (v.fidelity - 1.0).abs() < FIDELITY_EPS,
            "fidelity for n={n}: {}",
            v.fidelity
        );
        assert!(
            (v.total_probability - 1.0).abs() < PROB_EPS,
            "total probability for n={n}: {}",
            v.total_probability
        );
        assert_eq!(v.max_spurious_amplitude, 0.0, "spurious tail for n={n}");
    }
}

#[test]
fn w_state_serial_and_parallel_construction_agree() {
    // create_w_state writes a complex amplitude; create_w_state_parallel writes only
    // the real part (imag stays 0). They must yield identical amplitudes everywhere.
    let n = 5_000;
    let mut serial = SparseQuantumGridVec::new(n);
    serial.create_w_state();
    let mut parallel = SparseQuantumGridVec::new(n);
    parallel.create_w_state_parallel();

    for q in 0..n {
        let a = serial.get_amplitude(q);
        let b = parallel.get_amplitude(q);
        assert_eq!(a.re, b.re, "re mismatch at qubit {q}");
        assert_eq!(a.im, b.im, "im mismatch at qubit {q}");
    }
}

#[test]
fn w_state_serial_and_parallel_verification_agree() {
    let n = 5_000;
    let mut grid = SparseQuantumGridVec::new(n);
    grid.create_w_state();
    let serial = grid.verify_w_fidelity();
    let parallel = grid.verify_w_fidelity_parallel();

    // Integer count is exact; the floating sums differ only by reduction order.
    assert_eq!(serial.correct_amplitudes, parallel.correct_amplitudes);
    assert_eq!(serial.max_amplitude_error, parallel.max_amplitude_error);
    assert!(
        (serial.fidelity - parallel.fidelity).abs() < PROB_EPS,
        "fidelity {} vs {}",
        serial.fidelity,
        parallel.fidelity
    );
    assert!(
        (serial.total_probability - parallel.total_probability).abs() < PROB_EPS,
        "total probability {} vs {}",
        serial.total_probability,
        parallel.total_probability
    );
}

#[test]
fn w_state_amplitude_roundtrip_and_clear() {
    let mut grid = SparseQuantumGridVec::new(100);
    grid.set_amplitude(42, Complex64::new(0.5, -0.25));
    let amp = grid.get_amplitude(42);
    assert_eq!(amp.re, 0.5);
    assert_eq!(amp.im, -0.25);

    grid.clear();
    let cleared = grid.get_amplitude(42);
    assert_eq!(cleared.re, 0.0);
    assert_eq!(cleared.im, 0.0);
}

// === SymbolicGhzState / SymbolicWState ===

#[test]
fn symbolic_ghz_unit_fidelity_and_size_class() {
    let cases: [(SymbolicGhzState, &str); 4] = [
        (SymbolicGhzState::graham(), "Graham-class"),
        (SymbolicGhzState::tree(3), "TREE-class"),
        (SymbolicGhzState::infinite(), "infinite"),
        (
            SymbolicGhzState::new(SymbolicNumber::Literal(1000)),
            "small",
        ),
    ];
    for (ghz, class) in &cases {
        let v = ghz.verify();
        assert!(
            (v.fidelity - 1.0).abs() < FIDELITY_EPS,
            "fidelity for {class}: {}",
            v.fidelity
        );
        assert_eq!(
            v.amp_zero_state, v.amp_one_state,
            "endpoint symmetry for {class}"
        );
        assert_eq!(&v.size_class, class, "size class label");
    }
}

#[test]
fn symbolic_w_state_is_algebraically_normalized() {
    // Symbolic W-states check their stored 1/n formula exactly: sum of n copies of
    // 1/n equals 1, regardless of how astronomically large n is labelled.
    for w in [create_graham_w(), create_tree_w(3), create_infinite_w()] {
        let v = w.verify();
        assert!(v.is_valid, "symbolic W should validate");
        assert!(
            v.total_probability.is_one(),
            "total probability must be exactly 1"
        );
    }
}
