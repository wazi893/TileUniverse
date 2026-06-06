//! Module-level tests for math_accelerator.
//!
//! These tests verify the integration between components.

use logic_fabric_core::quantum::QGate;

use super::*;

#[test]
fn test_full_pipeline_ghz() {
    // Create a 10-qubit GHZ circuit
    let mut circuit = vec![QGate::H(0)];
    for i in 0..9 {
        circuit.push(QGate::CNot(i, i + 1));
    }

    // Quick classify should detect GHZ
    let quick = quick_classify(&circuit, 10);
    assert_eq!(quick, QuickClassification::GHZ);

    // Backend selection should route to sparse
    let decision = select_backend(&circuit, 10, true);
    assert!(matches!(
        decision,
        BackendDecision::SparseStateVector {
            variant: SparseVariant::VecBased
        }
    ));

    // Sparsity prediction should be GHZ
    let sparsity = predict_sparsity(&circuit, 10);
    assert_eq!(sparsity, SparsityPrediction::GHZ);
}

#[test]
fn test_full_pipeline_dense() {
    // Create a circuit that's clearly dense (many H gates)
    // Must have >= 50 gates to trigger NeedsFullAnalysis
    let mut circuit = Vec::new();
    for i in 0..20 {
        circuit.push(QGate::H(i as u8));
    }
    for i in 0..19 {
        circuit.push(QGate::CNot(i as u8, (i + 1) as u8));
    }
    // Add more gates to exceed threshold (39 so far, need 11 more)
    for i in 0..12 {
        circuit.push(QGate::X((i % 20) as u8));
    }

    // Verify we have enough gates
    assert!(
        circuit.len() >= 50,
        "Circuit has {} gates, need >= 50",
        circuit.len()
    );

    // Should require full analysis (not small, not GHZ/W)
    let quick = quick_classify(&circuit, 20);
    assert_eq!(quick, QuickClassification::NeedsFullAnalysis);

    // Should predict dense (20 H gates = 2^20 = 1M amplitudes)
    let sparsity = predict_sparsity(&circuit, 20);
    assert_eq!(sparsity, SparsityPrediction::Dense);
}

#[test]
fn test_characteristics_methods() {
    use logic_fabric_core::algebraic_fusion::AlgebraicStats;

    // Test is_identity
    let mut chars = CircuitCharacteristics::for_small_circuit(10);
    assert!(!chars.is_identity());

    // Add stats with gates_remaining = 0
    chars.algebraic_stats = Some(AlgebraicStats {
        original_gates: 100,
        gates_eliminated: 100,
        gates_remaining: 0,
        throughput_multiplier: f64::INFINITY,
    });
    assert!(chars.is_identity());

    // Test has_high_reduction
    let mut chars2 = CircuitCharacteristics::for_small_circuit(10);
    chars2.algebraic_stats = Some(AlgebraicStats {
        original_gates: 100,
        gates_eliminated: 85,
        gates_remaining: 15,
        throughput_multiplier: 6.67,
    });
    assert!(chars2.has_high_reduction()); // 85% > 80%

    let mut chars3 = CircuitCharacteristics::for_small_circuit(10);
    chars3.algebraic_stats = Some(AlgebraicStats {
        original_gates: 100,
        gates_eliminated: 50,
        gates_remaining: 50,
        throughput_multiplier: 2.0,
    });
    assert!(!chars3.has_high_reduction()); // 50% < 80%
}

#[test]
fn test_sparsity_prediction_methods() {
    // GHZ always has 2 NNZ
    let ghz = SparsityPrediction::GHZ;
    assert!(ghz.is_sparse());
    assert_eq!(ghz.expected_nnz(), Some(2));

    // W state has n NNZ
    let w = SparsityPrediction::W { n_qubits: 100 };
    assert!(w.is_sparse());
    assert_eq!(w.expected_nnz(), Some(100));

    // Dense has no expected NNZ
    let dense = SparsityPrediction::Dense;
    assert!(!dense.is_sparse());
    assert_eq!(dense.expected_nnz(), None);

    // Sparse has specific NNZ
    let sparse = SparsityPrediction::Sparse { expected_nnz: 500 };
    assert!(sparse.is_sparse());
    assert_eq!(sparse.expected_nnz(), Some(500));
}
