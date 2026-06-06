//! Backend selection logic using mathematical circuit analysis.
//!
//! This module provides the core decision logic that routes circuits
//! to appropriate backends based on their mathematical structure.
//!
//! # Decision Hierarchy
//!
//! 1. **Algebraic Skip**: Circuit reduces to identity → no execution needed
//! 2. **Sparse Backend**: GHZ/W patterns or low expected NNZ → hash-based simulation
//! 3. **Dense CPU**: High reduction rate → avoid GPU transfer overhead
//! 4. **Dense GPU**: Large circuit, low reduction → leverage parallelism

use logic_fabric_core::quantum::QGate;

use super::{
    CircuitCharacteristics, CircuitPattern, QuickClassification, SparsityPrediction,
    metrics::{
        count_distinct_hadamard_qubits, count_two_qubit_gates, is_clifford_circuit,
        predict_sparsity,
    },
    quick_classify,
};
use crate::tensor_network::{DEFAULT_MEMORY_BUDGET, is_tensor_network_viable};

/// Backend decision for circuit execution.
///
/// This enum represents the routing decision made by mathematical analysis.
/// The router uses this to select the appropriate substrate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendDecision {
    /// Entire circuit reduces to identity via algebraic optimization.
    /// No execution needed; return initial state.
    AlgebraicSkip,

    /// Use sparse state vector simulation.
    /// Efficient when expected non-zero entries << 2^n.
    SparseStateVector {
        /// Which sparse implementation variant to use.
        variant: SparseVariant,
    },

    /// Use dense state vector simulation.
    DenseStateVector {
        /// Whether to use GPU acceleration.
        /// `false` preferred when reduction rate is high (transfer overhead > compute savings).
        use_gpu: bool,
    },

    // Sprint 77: New routing variants
    /// Use tensor network simulation.
    /// Efficient for shallow circuits (10-50 qubits) with amplitude queries.
    TensorNetwork {
        /// Estimated peak memory in bytes during contraction.
        estimated_peak_memory: usize,
    },

    /// Use stabilizer (Clifford) simulation.
    /// Efficient for Clifford-only circuits using Aaronson-Gottesman tableau.
    /// O(n^2) simulation for n qubits, regardless of circuit depth.
    Stabilizer,
}

/// Variants of sparse state vector simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseVariant {
    /// Element-sparse: HashMap<u64, Complex>
    /// Good for general sparse states with random access patterns.
    ElementSparse,

    /// Block-sparse: Hierarchical block structure.
    /// Good for states with clustered non-zero regions.
    BlockSparse,

    /// Vec-based: Compact representation for special states.
    /// Optimal for GHZ (2 entries) and W (n entries) states.
    VecBased,
}

/// Select the optimal backend for a circuit based on mathematical analysis.
///
/// This is the main entry point for backend selection. It performs:
/// 1. Quick O(n) classification
/// 2. Pattern detection (GHZ, W)
/// 3. Sparsity prediction
/// 4. Algebraic reduction estimation (if needed)
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
/// * `n_qubits` - Total number of qubits
/// * `gpu_available` - Whether GPU backend is available
///
/// # Returns
/// [`BackendDecision`] indicating the recommended backend
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::{select_backend, BackendDecision, SparseVariant};
///
/// // GHZ circuit routes to sparse
/// let ghz = vec![
///     QGate::H(0),
///     QGate::CNot(0, 1),
///     QGate::CNot(1, 2),
/// ];
/// let decision = select_backend(&ghz, 3, true);
/// assert!(matches!(decision,
///     BackendDecision::SparseStateVector { variant: SparseVariant::VecBased }
/// ));
/// ```
pub fn select_backend(circuit: &[QGate], n_qubits: u8, gpu_available: bool) -> BackendDecision {
    // Step 1: Quick classification
    let quick = quick_classify(circuit, n_qubits);

    match quick {
        QuickClassification::Small => {
            // Small circuits: use simple CPU backend
            // Analysis overhead > execution time
            return BackendDecision::DenseStateVector { use_gpu: false };
        }
        QuickClassification::GHZ => {
            // GHZ pattern: exactly 2 amplitudes, use VecBased
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::VecBased,
            };
        }
        QuickClassification::WState => {
            // W state pattern: n amplitudes, use VecBased
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::VecBased,
            };
        }
        QuickClassification::NeedsFullAnalysis => {
            // Continue to detailed analysis
        }
    }

    // Sprint 77: Check Clifford-only for stabilizer routing
    // Clifford circuits can be simulated in O(n^2) using stabilizer formalism
    if is_clifford_circuit(circuit) {
        return BackendDecision::Stabilizer;
    }

    // Sprint 77: Check tensor network viability for small circuits
    // TN is beneficial for shallow circuits with amplitude queries
    if n_qubits <= 25 {
        let viability = is_tensor_network_viable(circuit, n_qubits, DEFAULT_MEMORY_BUDGET);
        if viability.is_viable {
            return BackendDecision::TensorNetwork {
                estimated_peak_memory: viability.estimated_peak_memory,
            };
        }
    }

    // Step 2: Compute circuit characteristics
    let characteristics = analyze_circuit(circuit, n_qubits);

    // Step 3: Check algebraic reduction
    if characteristics.is_identity() {
        return BackendDecision::AlgebraicSkip;
    }

    // High reduction rate: prefer CPU (avoid GPU transfer overhead)
    if characteristics.has_high_reduction() {
        return BackendDecision::DenseStateVector { use_gpu: false };
    }

    // Step 4: Check sparsity prediction
    match &characteristics.predicted_sparsity {
        SparsityPrediction::GHZ | SparsityPrediction::W { .. } => {
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::VecBased,
            };
        }
        SparsityPrediction::Sparse { expected_nnz } if *expected_nnz < 1000 => {
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::ElementSparse,
            };
        }
        _ => {}
    }

    // Step 5: Fall back to qubit-count based decision
    // GPU generally better for > 16 qubits
    let use_gpu = gpu_available && n_qubits > 16;
    BackendDecision::DenseStateVector { use_gpu }
}

/// Analyze circuit to compute full characteristics.
///
/// This function computes all metrics needed for backend selection:
/// - Gate counts (total, two-qubit)
/// - Hadamard qubit count
/// - Sparsity prediction
/// - Pattern detection
///
/// Note: This does NOT run full algebraic optimization (expensive).
/// Algebraic stats can be added separately if needed.
pub fn analyze_circuit(circuit: &[QGate], n_qubits: u8) -> CircuitCharacteristics {
    let gate_count = circuit.len();
    let two_qubit_gate_count = count_two_qubit_gates(circuit);
    let distinct_hadamard_qubits = count_distinct_hadamard_qubits(circuit);
    let predicted_sparsity = predict_sparsity(circuit, n_qubits);

    // Detect specific patterns
    let detected_pattern = detect_pattern(circuit);

    CircuitCharacteristics {
        algebraic_stats: None, // Not computed here to avoid overhead
        gate_count,
        two_qubit_gate_count,
        distinct_hadamard_qubits,
        predicted_sparsity,
        detected_pattern,
    }
}

/// Detect known circuit patterns.
fn detect_pattern(circuit: &[QGate]) -> Option<CircuitPattern> {
    use super::metrics::{detect_ghz_pattern, detect_w_state_pattern};

    if detect_ghz_pattern(circuit) {
        return Some(CircuitPattern::GHZPreparation);
    }

    if detect_w_state_pattern(circuit) {
        return Some(CircuitPattern::WStatePreparation);
    }

    // TODO: Add Grover and VQE pattern detection in Sprint 77
    None
}

/// Select backend with pre-computed characteristics.
///
/// Use this when characteristics have already been computed and cached
/// (e.g., in JobSpec).
pub fn select_backend_with_characteristics(
    characteristics: &CircuitCharacteristics,
    n_qubits: u8,
    gpu_available: bool,
) -> BackendDecision {
    // Check algebraic reduction
    if characteristics.is_identity() {
        return BackendDecision::AlgebraicSkip;
    }

    if characteristics.has_high_reduction() {
        return BackendDecision::DenseStateVector { use_gpu: false };
    }

    // Check sparsity
    match &characteristics.predicted_sparsity {
        SparsityPrediction::GHZ | SparsityPrediction::W { .. } => {
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::VecBased,
            };
        }
        SparsityPrediction::Sparse { expected_nnz } if *expected_nnz < 1000 => {
            return BackendDecision::SparseStateVector {
                variant: SparseVariant::ElementSparse,
            };
        }
        _ => {}
    }

    // Check detected pattern
    if let Some(pattern) = &characteristics.detected_pattern {
        match pattern {
            CircuitPattern::GHZPreparation | CircuitPattern::WStatePreparation => {
                return BackendDecision::SparseStateVector {
                    variant: SparseVariant::VecBased,
                };
            }
            _ => {}
        }
    }

    // Fall back to qubit-count decision
    let use_gpu = gpu_available && n_qubits > 16;
    BackendDecision::DenseStateVector { use_gpu }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghz_routes_to_sparse() {
        let ghz = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
        ];
        let decision = select_backend(&ghz, 4, true);
        assert!(matches!(
            decision,
            BackendDecision::SparseStateVector {
                variant: SparseVariant::VecBased
            }
        ));
    }

    #[test]
    fn test_small_circuit_uses_cpu() {
        let small = vec![QGate::H(0), QGate::X(1)];
        let decision = select_backend(&small, 2, true);
        assert!(matches!(
            decision,
            BackendDecision::DenseStateVector { use_gpu: false }
        ));
    }

    #[test]
    fn test_large_circuit_with_gpu() {
        // Large circuit without special patterns
        // Use 30 qubits to bypass TN viability check (> 25 qubit limit)
        let large: Vec<QGate> = (0..100)
            .flat_map(|i| {
                vec![
                    QGate::H((i % 30) as u8),
                    QGate::Rx((i % 30) as u8, 0.5),
                    QGate::CNot((i % 30) as u8, ((i + 1) % 30) as u8),
                ]
            })
            .collect();

        let decision = select_backend(&large, 30, true);
        // Should use GPU for large circuits (30 qubits > 16 threshold)
        assert!(matches!(
            decision,
            BackendDecision::DenseStateVector { use_gpu: true }
        ));
    }

    #[test]
    fn test_sparse_prediction_routes_correctly() {
        // Single H gate: very sparse
        let sparse = vec![QGate::H(0)];
        let decision = select_backend(&sparse, 10, true);
        // Small circuit, uses CPU dense (analysis overhead not worth it)
        assert!(matches!(
            decision,
            BackendDecision::DenseStateVector { use_gpu: false }
        ));
    }

    #[test]
    fn test_analyze_circuit() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 1),
            QGate::CZ(1, 2),
            QGate::X(2),
        ];
        let chars = analyze_circuit(&circuit, 3);

        assert_eq!(chars.gate_count, 5);
        assert_eq!(chars.two_qubit_gate_count, 2);
        assert_eq!(chars.distinct_hadamard_qubits, 2);
    }

    // =========================================================================
    // Sprint 77.4: Stabilizer and Tensor Network Routing Tests
    // =========================================================================

    #[test]
    fn test_clifford_circuit_routes_to_stabilizer() {
        // Large Clifford circuit (>= 50 gates) that isn't a GHZ pattern
        // This ensures it passes through NeedsFullAnalysis and hits the Clifford check
        let mut circuit = Vec::new();
        for i in 0..10 {
            circuit.push(QGate::H(i));
        }
        // Add CNOTs in a non-GHZ pattern (random connections)
        for i in 0..30 {
            circuit.push(QGate::CNot((i % 10) as u8, ((i + 3) % 10) as u8));
        }
        // Add more single-qubit Clifford gates
        for i in 0..15 {
            circuit.push(QGate::X((i % 10) as u8));
            circuit.push(QGate::Z((i % 10) as u8));
        }
        // Total: 10 + 30 + 30 = 70 gates (> 50 threshold)
        assert!(circuit.len() >= 50);

        let decision = select_backend(&circuit, 10, true);
        assert!(
            matches!(decision, BackendDecision::Stabilizer),
            "Expected Stabilizer, got {:?}",
            decision
        );
    }

    #[test]
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
    fn test_non_clifford_circuit_not_stabilizer() {
        // Large circuit with T gate (non-Clifford)
        let mut circuit = Vec::new();
        for i in 0..10 {
            circuit.push(QGate::H(i));
        }
        circuit.push(QGate::T(0)); // Non-Clifford!
        for i in 0..45 {
            circuit.push(QGate::CNot((i % 10) as u8, ((i + 1) % 10) as u8));
        }
        // Total: 10 + 1 + 45 = 56 gates (> 50 threshold)
        assert!(circuit.len() >= 50);

        let decision = select_backend(&circuit, 10, true);
        assert!(
            !matches!(decision, BackendDecision::Stabilizer),
            "Should NOT be Stabilizer due to T gate, got {:?}",
            decision
        );
    }

    #[test]
    fn test_large_circuit_routes_to_tn_when_viable() {
        // Large shallow circuit with non-Clifford gates that should be TN-viable
        // Must be >= 50 gates to trigger NeedsFullAnalysis
        // Use only TN-supported gates: H, X, Y, Z, T, Tdg, Rx, Ry, Rz, CNot, CZ
        let mut circuit = Vec::new();
        // Spread H gates across qubits to avoid dense patterns
        for i in 0..15 {
            circuit.push(QGate::H(i % 8));
        }
        circuit.push(QGate::T(0)); // Non-Clifford, prevents Stabilizer routing
        // Shallow CNOT layer
        for i in 0..40 {
            circuit.push(QGate::CNot((i % 7) as u8, ((i % 7) + 1) as u8));
        }
        // Total: 15 + 1 + 40 = 56 gates (> 50 threshold)
        assert!(circuit.len() >= 50);

        // Use 8 qubits (small enough for TN to be viable with reasonable memory)
        let decision = select_backend(&circuit, 8, true);
        // Non-Clifford (has T gate), 8 qubits <= 25 limit
        // TN should be viable for this shallow circuit
        assert!(
            matches!(decision, BackendDecision::TensorNetwork { .. }),
            "Expected TensorNetwork, got {:?}",
            decision
        );
    }

    #[test]
    fn test_tn_not_used_for_large_qubit_count() {
        // Large circuit with more than 25 qubits should not route to TN
        let mut circuit = Vec::new();
        for i in 0..30 {
            circuit.push(QGate::H(i));
        }
        circuit.push(QGate::T(0)); // Non-Clifford
        for i in 0..30 {
            circuit.push(QGate::CNot(i, (i + 1) % 30));
        }
        // Total: 30 + 1 + 30 = 61 gates (> 50 threshold)
        assert!(circuit.len() >= 50);

        let decision = select_backend(&circuit, 30, true);
        // Should NOT be TN (qubit limit exceeded) and NOT Stabilizer (has T gate)
        assert!(
            !matches!(decision, BackendDecision::TensorNetwork { .. }),
            "Should NOT be TensorNetwork for 30 qubits, got {:?}",
            decision
        );
        assert!(
            !matches!(decision, BackendDecision::Stabilizer),
            "Should NOT be Stabilizer due to T gate, got {:?}",
            decision
        );
    }

    #[test]
    fn test_clifford_with_clifford_rotation_routes_to_stabilizer() {
        // Large circuit with Clifford rotations (pi/2 angles)
        use std::f32::consts::PI;
        let mut circuit = Vec::new();
        for i in 0..15 {
            circuit.push(QGate::H(i % 10));
            circuit.push(QGate::Rz(i % 10, PI / 2.0)); // Clifford angle
        }
        for i in 0..25 {
            circuit.push(QGate::CNot((i % 10) as u8, ((i + 1) % 10) as u8));
        }
        // Total: 30 + 25 = 55 gates (> 50 threshold)
        assert!(circuit.len() >= 50);

        let decision = select_backend(&circuit, 10, true);
        assert!(
            matches!(decision, BackendDecision::Stabilizer),
            "Expected Stabilizer for Clifford-only circuit, got {:?}",
            decision
        );
    }

    #[test]
    fn test_small_circuit_bypasses_clifford_check() {
        // Small circuits (< 50 gates) bypass the Clifford check and go to CPU
        // This is intentional: small circuits execute faster than analysis overhead
        let small_clifford = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CZ(1, 2), QGate::X(2)];
        assert!(small_clifford.len() < 50);

        let decision = select_backend(&small_clifford, 3, true);
        // Small circuit goes to CPU dense (fast path)
        assert!(
            matches!(
                decision,
                BackendDecision::DenseStateVector { use_gpu: false }
            ),
            "Small circuits should use CPU, got {:?}",
            decision
        );
    }
}
