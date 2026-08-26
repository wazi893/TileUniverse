//! Circuit metrics and pattern detection for backend routing.
//!
//! This module provides O(n) analysis functions that detect circuit patterns
//! and predict sparsity without running full algebraic optimization.

use logic_fabric_core::quantum::QGate;
use std::collections::HashSet;
use std::f32::consts::PI;

use super::SparsityPrediction;

/// Detect GHZ state preparation pattern.
///
/// GHZ pattern: H(0) followed by CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)
/// This creates the state (|00...0⟩ + |11...1⟩) / sqrt(2) with exactly 2 amplitudes.
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
///
/// # Returns
/// `true` if the circuit matches the GHZ preparation pattern
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::detect_ghz_pattern;
///
/// let ghz_3 = vec![
///     QGate::H(0),
///     QGate::CNot(0, 1),
///     QGate::CNot(1, 2),
/// ];
/// assert!(detect_ghz_pattern(&ghz_3));
/// ```
pub fn detect_ghz_pattern(circuit: &[QGate]) -> bool {
    if circuit.is_empty() {
        return false;
    }

    // First gate must be H(0)
    if !matches!(circuit[0], QGate::H(0)) {
        return false;
    }

    // Must have at least 2 gates for a non-trivial GHZ
    if circuit.len() < 2 {
        return false;
    }

    // Rest must be CNOT chain: CNOT(0,1), CNOT(1,2), ...
    for (i, gate) in circuit.iter().enumerate().skip(1) {
        let expected_control = (i - 1) as u8;
        let expected_target = i as u8;

        match gate {
            QGate::CNot(control, target)
                if *control == expected_control && *target == expected_target => {}
            _ => return false,
        }
    }

    true
}

/// Detect W state preparation pattern.
///
/// W state creates superposition of single-excitation states:
/// |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / sqrt(n)
///
/// Common patterns:
/// 1. X(0) followed by controlled rotations spreading amplitude
/// 2. Specific ancilla-based preparation circuits
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
///
/// # Returns
/// `true` if the circuit matches a W state preparation pattern
pub fn detect_w_state_pattern(circuit: &[QGate]) -> bool {
    if circuit.len() < 2 {
        return false;
    }

    // Pattern 1: X(0) followed by controlled rotations
    // This is a common W-state preparation starting with |10...0⟩
    let starts_with_x = matches!(circuit[0], QGate::X(0));
    if !starts_with_x {
        return false;
    }

    // Check for controlled rotation pattern (Ry rotations with specific angles)
    // W-state uses Ry(2*arccos(sqrt(1/k))) for k = n, n-1, ..., 2
    let mut has_controlled_rotations = false;
    let mut has_cnots = false;

    for gate in circuit.iter().skip(1) {
        match gate {
            QGate::CRz(_, _, _) | QGate::CPhase(_, _, _) => {
                has_controlled_rotations = true;
            }
            QGate::Ry(_, _) => {
                // Ry gates are used in W-state preparation
                has_controlled_rotations = true;
            }
            QGate::CNot(_, _) => {
                has_cnots = true;
            }
            _ => {}
        }
    }

    // W-state typically needs rotations and CNOTs to spread amplitude
    has_controlled_rotations || has_cnots
}

/// Count the number of distinct qubits that have Hadamard gates applied.
///
/// This is a key metric for sparsity prediction:
/// - h Hadamard qubits → base of 2^h amplitudes
/// - Entanglement (2-qubit gates) spreads these amplitudes
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
///
/// # Returns
/// Number of unique qubits with at least one H gate
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::count_distinct_hadamard_qubits;
///
/// let circuit = vec![
///     QGate::H(0),
///     QGate::H(1),
///     QGate::H(0),  // Same qubit, doesn't increase count
///     QGate::CNot(0, 1),
/// ];
/// assert_eq!(count_distinct_hadamard_qubits(&circuit), 2);
/// ```
pub fn count_distinct_hadamard_qubits(circuit: &[QGate]) -> usize {
    let mut h_qubits = HashSet::new();

    for gate in circuit {
        if let QGate::H(q) = gate {
            h_qubits.insert(*q);
        }
    }

    h_qubits.len()
}

/// Count the number of two-qubit gates in the circuit.
///
/// Two-qubit gates (CNot, CZ, Swap, etc.) spread entanglement
/// and can increase the number of non-zero amplitudes.
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
///
/// # Returns
/// Total count of two-qubit gates
pub fn count_two_qubit_gates(circuit: &[QGate]) -> usize {
    circuit
        .iter()
        .filter(|gate| is_two_qubit_gate(gate))
        .count()
}

/// Check if a gate is a two-qubit gate.
#[inline]
fn is_two_qubit_gate(gate: &QGate) -> bool {
    matches!(
        gate,
        QGate::CNot(_, _)
            | QGate::CZ(_, _)
            | QGate::Swap(_, _)
            | QGate::CPhase(_, _, _)
            | QGate::CRz(_, _, _)
    )
}

/// Predict sparsity of the final state vector.
///
/// Uses an entanglement-aware formula:
/// - Base: 2^h amplitudes from h Hadamard qubits
/// - Adjustment: entanglement factor spreads amplitudes
///
/// # Key Insight
/// Entanglement destroys sparsity faster than superposition creates it.
/// Each 2-qubit gate can potentially double the number of non-zero entries.
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
/// * `n_qubits` - Total number of qubits
///
/// # Returns
/// [`SparsityPrediction`] indicating expected sparsity
pub fn predict_sparsity(circuit: &[QGate], n_qubits: u8) -> SparsityPrediction {
    // Special case: empty circuit
    if circuit.is_empty() {
        return SparsityPrediction::Sparse { expected_nnz: 1 };
    }

    // Check for special patterns first
    if detect_ghz_pattern(circuit) {
        return SparsityPrediction::GHZ;
    }

    if detect_w_state_pattern(circuit) {
        return SparsityPrediction::W {
            n_qubits: n_qubits as usize,
        };
    }

    // General sparsity prediction
    let h_qubits = count_distinct_hadamard_qubits(circuit);
    let two_qubit_count = count_two_qubit_gates(circuit);

    // Entanglement factor: how much 2-qubit gates spread amplitude
    // Capped at 1.0 (fully entangled)
    let entanglement_factor = if n_qubits > 0 {
        (two_qubit_count as f64 / n_qubits as f64).min(1.0)
    } else {
        0.0
    };

    // Base sparsity from Hadamards: 2^h amplitudes
    // We cap h to avoid overflow for large h values
    let effective_h = h_qubits.min(30);
    let base_nnz = 1usize << effective_h;

    // Adjusted for entanglement spreading
    // Each unit of entanglement_factor adds 50% more amplitudes
    let expected_nnz = (base_nnz as f64 * (1.0 + entanglement_factor * 0.5)) as usize;

    // Sparse threshold: if expected < 1000, use sparse backend
    if expected_nnz < 1000 {
        SparsityPrediction::Sparse { expected_nnz }
    } else {
        SparsityPrediction::Dense
    }
}

// =========================================================================
// CLIFFORD DETECTION (Phase 77.1)
// =========================================================================

/// Check if an angle is a Clifford angle (multiple of pi/2).
///
/// Clifford angles: 0, pi/2, pi, 3*pi/2, 2*pi, ...
/// These correspond to rotations that map Paulis to Paulis.
///
/// # Arguments
/// * `angle` - The rotation angle in radians
///
/// # Returns
/// `true` if the angle is a multiple of pi/2 (within epsilon tolerance)
///
/// # Example
/// ```ignore
/// use std::f32::consts::PI;
/// use engine::math_accelerator::is_clifford_angle;
///
/// assert!(is_clifford_angle(PI / 2.0));      // true: 90 degrees
/// assert!(is_clifford_angle(PI));             // true: 180 degrees
/// assert!(!is_clifford_angle(PI / 4.0));     // false: T gate angle
/// ```
pub fn is_clifford_angle(angle: f32) -> bool {
    // Normalize to positive angle and check if it's a multiple of pi/2
    let normalized = angle.abs() % (PI / 2.0);
    // Check if close to 0 or close to pi/2
    normalized < 1e-5 || (PI / 2.0 - normalized).abs() < 1e-5
}

/// Check if a gate is non-Clifford.
///
/// Non-Clifford gates are those that cannot be efficiently simulated
/// using stabilizer formalism. The key non-Clifford gates are:
/// - T gate (pi/8 rotation)
/// - Rotation gates with non-Clifford angles
///
/// # Arguments
/// * `gate` - The quantum gate to check
///
/// # Returns
/// `true` if the gate is non-Clifford
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::is_non_clifford_gate;
///
/// assert!(is_non_clifford_gate(&QGate::T(0)));           // T gate is non-Clifford
/// assert!(!is_non_clifford_gate(&QGate::H(0)));          // H is Clifford
/// assert!(!is_non_clifford_gate(&QGate::CNot(0, 1)));    // CNOT is Clifford
/// ```
pub fn is_non_clifford_gate(gate: &QGate) -> bool {
    match gate {
        // T and Tdg are the canonical non-Clifford gates
        QGate::T(_) | QGate::Tdg(_) => true,

        // Rotation gates with non-Clifford angles
        QGate::Rz(_, angle) | QGate::Rx(_, angle) | QGate::Ry(_, angle) => {
            !is_clifford_angle(*angle)
        }

        // Phase gate with non-Clifford angle
        QGate::Phase(_, angle) => !is_clifford_angle(*angle),

        // Controlled phase with non-Clifford angle
        QGate::CPhase(_, _, angle) => !is_clifford_angle(*angle),

        // Controlled Rz with non-Clifford angle
        QGate::CRz(_, _, angle) => !is_clifford_angle(*angle),

        // U3 gate: all three angles must be Clifford
        QGate::U3(_, theta, phi, lambda) => {
            !is_clifford_angle(*theta) || !is_clifford_angle(*phi) || !is_clifford_angle(*lambda)
        }

        // All other gates are Clifford: H, X, Y, Z, CNot, CZ, Swap, Toffoli, CCZ, Measure
        _ => false,
    }
}

/// Check if a circuit is entirely Clifford (no T gates or non-Clifford rotations).
///
/// Clifford circuits can be efficiently simulated using stabilizer formalism
/// (Aaronson-Gottesman algorithm) in O(n^2) time.
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
///
/// # Returns
/// `true` if all gates in the circuit are Clifford gates
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::is_clifford_circuit;
///
/// // Clifford circuit: H-CNOT-CZ
/// let clifford = vec![
///     QGate::H(0),
///     QGate::CNot(0, 1),
///     QGate::CZ(1, 2),
/// ];
/// assert!(is_clifford_circuit(&clifford));
///
/// // Non-Clifford: contains T gate
/// let non_clifford = vec![
///     QGate::H(0),
///     QGate::T(0),
///     QGate::CNot(0, 1),
/// ];
/// assert!(!is_clifford_circuit(&non_clifford));
/// ```
pub fn is_clifford_circuit(circuit: &[QGate]) -> bool {
    !circuit.iter().any(is_non_clifford_gate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghz_pattern_detection() {
        // Valid 3-qubit GHZ
        let ghz_3 = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CNot(1, 2)];
        assert!(detect_ghz_pattern(&ghz_3));

        // Valid 2-qubit GHZ (Bell state)
        let bell = vec![QGate::H(0), QGate::CNot(0, 1)];
        assert!(detect_ghz_pattern(&bell));

        // Invalid: H not on qubit 0
        let invalid1 = vec![QGate::H(1), QGate::CNot(0, 1)];
        assert!(!detect_ghz_pattern(&invalid1));

        // Invalid: wrong CNOT chain
        let invalid2 = vec![QGate::H(0), QGate::CNot(0, 2), QGate::CNot(1, 2)];
        assert!(!detect_ghz_pattern(&invalid2));

        // Empty circuit
        assert!(!detect_ghz_pattern(&[]));
    }

    #[test]
    fn test_hadamard_qubit_counting() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(0), // Duplicate
            QGate::CNot(0, 1),
            QGate::H(2),
        ];
        assert_eq!(count_distinct_hadamard_qubits(&circuit), 3);

        // No Hadamards
        let no_h = vec![QGate::X(0), QGate::CNot(0, 1)];
        assert_eq!(count_distinct_hadamard_qubits(&no_h), 0);
    }

    #[test]
    fn test_two_qubit_gate_counting() {
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CZ(1, 2),
            QGate::X(0),
            QGate::Swap(0, 2),
        ];
        assert_eq!(count_two_qubit_gates(&circuit), 3);
    }

    #[test]
    fn test_sparsity_prediction_ghz() {
        let ghz = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
        ];
        assert_eq!(predict_sparsity(&ghz, 4), SparsityPrediction::GHZ);
    }

    #[test]
    fn test_sparsity_prediction_sparse() {
        // Single H gate: 2 amplitudes expected
        let single_h = vec![QGate::H(0)];
        match predict_sparsity(&single_h, 4) {
            SparsityPrediction::Sparse { expected_nnz } => {
                assert!(expected_nnz <= 4); // 2^1 * small factor
            }
            _ => panic!("Expected sparse prediction"),
        }
    }

    #[test]
    fn test_sparsity_prediction_dense() {
        // Many H gates: should be dense
        let many_h: Vec<QGate> = (0..12).map(QGate::H).collect();
        assert_eq!(predict_sparsity(&many_h, 12), SparsityPrediction::Dense);
    }

    // =========================================================================
    // CLIFFORD DETECTION TESTS (Phase 77.1)
    // =========================================================================

    #[test]
    fn test_clifford_angle_multiples_of_pi_over_2() {
        // Clifford angles: multiples of pi/2
        assert!(is_clifford_angle(0.0));
        assert!(is_clifford_angle(PI / 2.0));
        assert!(is_clifford_angle(PI));
        assert!(is_clifford_angle(3.0 * PI / 2.0));
        assert!(is_clifford_angle(2.0 * PI));
        assert!(is_clifford_angle(-PI / 2.0));
        assert!(is_clifford_angle(-PI));
    }

    #[test]
    fn test_clifford_angle_non_clifford() {
        // Non-Clifford angles
        assert!(!is_clifford_angle(PI / 4.0)); // T gate angle
        assert!(!is_clifford_angle(PI / 8.0));
        assert!(!is_clifford_angle(PI / 3.0));
        assert!(!is_clifford_angle(0.1));
        assert!(!is_clifford_angle(-PI / 4.0));
    }

    #[test]
    fn test_non_clifford_gate_t_gates() {
        // T and Tdg are non-Clifford
        assert!(is_non_clifford_gate(&QGate::T(0)));
        assert!(is_non_clifford_gate(&QGate::Tdg(0)));
    }

    #[test]
    fn test_non_clifford_gate_rotations() {
        // Non-Clifford rotations (pi/4 is T gate angle)
        assert!(is_non_clifford_gate(&QGate::Rz(0, PI / 4.0)));
        assert!(is_non_clifford_gate(&QGate::Rx(0, PI / 4.0)));
        assert!(is_non_clifford_gate(&QGate::Ry(0, PI / 4.0)));

        // Clifford rotations (multiples of pi/2)
        assert!(!is_non_clifford_gate(&QGate::Rz(0, PI / 2.0)));
        assert!(!is_non_clifford_gate(&QGate::Rx(0, PI)));
        assert!(!is_non_clifford_gate(&QGate::Ry(0, 0.0)));
    }

    #[test]
    fn test_non_clifford_gate_phase() {
        // Non-Clifford phase (pi/4)
        assert!(is_non_clifford_gate(&QGate::Phase(0, PI / 4.0)));

        // Clifford phase (pi/2 = S gate)
        assert!(!is_non_clifford_gate(&QGate::Phase(0, PI / 2.0)));
    }

    #[test]
    fn test_non_clifford_gate_clifford_gates() {
        // Standard Clifford gates
        assert!(!is_non_clifford_gate(&QGate::H(0)));
        assert!(!is_non_clifford_gate(&QGate::X(0)));
        assert!(!is_non_clifford_gate(&QGate::Y(0)));
        assert!(!is_non_clifford_gate(&QGate::Z(0)));
        assert!(!is_non_clifford_gate(&QGate::CNot(0, 1)));
        assert!(!is_non_clifford_gate(&QGate::CZ(0, 1)));
        assert!(!is_non_clifford_gate(&QGate::Swap(0, 1)));
    }

    #[test]
    fn test_clifford_circuit_all_clifford() {
        // H-CNOT-CZ circuit is Clifford
        let clifford = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CZ(1, 2)];
        assert!(is_clifford_circuit(&clifford));

        // Bell state preparation is Clifford
        let bell = vec![QGate::H(0), QGate::CNot(0, 1)];
        assert!(is_clifford_circuit(&bell));

        // Empty circuit is Clifford
        assert!(is_clifford_circuit(&[]));
    }

    #[test]
    fn test_clifford_circuit_with_t_gate() {
        // Circuit with T gate is non-Clifford
        let non_clifford = vec![QGate::H(0), QGate::T(0), QGate::CNot(0, 1)];
        assert!(!is_clifford_circuit(&non_clifford));
    }

    #[test]
    fn test_clifford_circuit_with_non_clifford_rotation() {
        // Circuit with non-Clifford rotation is non-Clifford
        let non_clifford = vec![
            QGate::H(0),
            QGate::Rz(0, 0.1), // arbitrary angle
            QGate::CNot(0, 1),
        ];
        assert!(!is_clifford_circuit(&non_clifford));
    }
}
