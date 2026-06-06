//! QRAM Integration Tests
//!
//! End-to-end tests for the QRAM module (Sprint 46.0).
//! Extended with sparse QRAM tests (Sprint 49.0).

use engine::qram::{
    GraphRouter, QRAMQuery, QRAMRouter, QRAMTree, RouterState, SparseBucketBrigade, StabilizerQRAM,
};

// =============================================================================
// Classical Functionality Tests
// =============================================================================

#[test]
fn test_qram_classical_correctness() {
    let mut qram = QRAMQuery::new(3);
    qram.load(&[100, 200, 300, 400, 500, 600, 700, 800]);

    for addr in 0..8 {
        let result = qram.query_classical(addr);
        assert_eq!(result, Some((addr as u64 + 1) * 100));
    }
}

#[test]
fn test_qram_large_data() {
    let mut qram = QRAMQuery::new(6); // 64 cells

    let data: Vec<u64> = (0..64).map(|i| i * 1000).collect();
    qram.load(&data);

    for addr in 0..64 {
        let result = qram.query_classical(addr);
        assert_eq!(result, Some(addr as u64 * 1000));
    }
}

// =============================================================================
// Quantum Query Tests
// =============================================================================

#[test]
fn test_qram_basis_query_all_addresses() {
    let mut qram = QRAMQuery::new(3);
    qram.load(&[10, 20, 30, 40, 50, 60, 70, 80]);

    for addr in 0..8 {
        let result = qram.query_basis(addr, 12345 + addr as u64).unwrap();
        assert_eq!(result.address, addr);
        assert_eq!(result.data, (addr as u64 + 1) * 10);
        assert!((result.probability - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_qram_superposition_uniform() {
    let mut qram = QRAMQuery::new(3);
    qram.load(&[1, 2, 3, 4, 5, 6, 7, 8]);

    let results = qram.query_superposition(42);

    // Should have 8 results
    assert_eq!(results.len(), 8);

    // Total probability should be 1
    let total_prob: f64 = results.iter().map(|r| r.probability).sum();
    assert!(
        (total_prob - 1.0).abs() < 0.01,
        "Total probability {} should be ~1.0",
        total_prob
    );

    // Each should have ~12.5% probability (1/8)
    let expected = 1.0 / 8.0;
    for result in &results {
        assert!(
            (result.probability - expected).abs() < 0.05,
            "Address {} has probability {}, expected ~{}",
            result.address,
            result.probability,
            expected
        );
    }
}

#[test]
fn test_qram_measurement_distribution() {
    let mut qram = QRAMQuery::new(2);
    qram.load(&[1, 2, 3, 4]);

    // Run 400 measurements
    let stats = qram.query_statistics(400, 0);

    // All 4 addresses should appear
    let total: usize = stats.iter().map(|(_, count)| *count).sum();
    assert_eq!(total, 400);

    // Check roughly uniform distribution (each ~25% = 100 ± 50)
    for (addr, count) in &stats {
        assert!(
            *count >= 50 && *count <= 150,
            "Address {} appeared {} times, expected ~100",
            addr,
            count
        );
    }
}

// =============================================================================
// Router and Tree Tests
// =============================================================================

#[test]
fn test_router_state_transitions() {
    let mut router = QRAMRouter::new(0, 0);

    // Initial state
    assert_eq!(router.state(), RouterState::Idle);

    // Route left
    let (left, right) = router.route(RouterState::Left);
    assert!(left && !right);
    assert_eq!(router.state(), RouterState::Left);

    // Reset and route right
    router.reset();
    let (left, right) = router.route(RouterState::Right);
    assert!(!left && right);
    assert_eq!(router.state(), RouterState::Right);

    // Reset and superposition
    router.reset();
    let (left, right) = router.route(RouterState::Superposition);
    assert!(left && right);
    assert_eq!(router.state(), RouterState::Superposition);
}

#[test]
fn test_tree_structure_depth_4() {
    let tree = QRAMTree::new(4);

    assert_eq!(tree.size(), 16);
    assert_eq!(tree.depth(), 4);
    assert_eq!(tree.router_count(), 15); // 1 + 2 + 4 + 8

    // Verify qubit formula: depth + 1 + 2*routers
    assert_eq!(tree.total_qubits(), 4 + 1 + 2 * 15);
}

#[test]
fn test_classical_routing_all_paths() {
    let mut tree = QRAMTree::new(4);
    let data: Vec<u64> = (0..16).collect();
    tree.load_data(&data);

    for addr in 0..16 {
        tree.reset_routers();
        let cell = tree.route_classical(addr);
        assert_eq!(cell, addr);
        assert_eq!(tree.get(addr), Some(addr as u64));
    }
}

// =============================================================================
// Circuit Generation Tests
// =============================================================================

#[test]
fn test_circuit_gate_types() {
    let qram = QRAMQuery::new(2);
    let circuit = qram.circuit();

    // Should contain CNot and Toffoli gates (from Fredkin decomposition)
    let mut has_cnot = false;
    let mut has_toffoli = false;

    for gate in &circuit {
        match gate {
            engine::quantum::QGate::CNot(_, _) => has_cnot = true,
            engine::quantum::QGate::Toffoli(_, _, _) => has_toffoli = true,
            _ => {}
        }
    }

    assert!(has_cnot, "Circuit should contain CNOT gates");
    assert!(has_toffoli, "Circuit should contain Toffoli gates");
}

#[test]
fn test_circuit_scaling() {
    // Verify circuit metrics scale correctly
    for depth in 1..=5 {
        let qram = QRAMQuery::new(depth);

        let routers = (1 << depth) - 1;
        let expected_gates = 3 * routers; // 3 gates per Fredkin
        let expected_depth = 3 * depth;

        assert_eq!(
            qram.gate_count(),
            expected_gates,
            "Depth {}: gate count mismatch",
            depth
        );
        assert_eq!(
            qram.circuit_depth(),
            expected_depth,
            "Depth {}: circuit depth mismatch",
            depth
        );
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_minimum_depth() {
    let mut qram = QRAMQuery::new(1);
    qram.load(&[42, 99]);

    assert_eq!(qram.query_classical(0), Some(42));
    assert_eq!(qram.query_classical(1), Some(99));

    let results = qram.query_superposition(123);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_zero_data() {
    let mut qram = QRAMQuery::new(2);
    // All zeros (default)

    for addr in 0..4 {
        assert_eq!(qram.query_classical(addr), Some(0));
    }

    let results = qram.query_superposition(42);
    for result in &results {
        assert_eq!(result.data, 0);
    }
}

#[test]
fn test_out_of_bounds_queries() {
    let mut qram = QRAMQuery::new(2); // 4 cells

    assert!(qram.query_classical(4).is_none());
    assert!(qram.query_classical(100).is_none());
    assert!(qram.query_basis(4, 42).is_none());
}

// =============================================================================
// Performance Sanity Checks
// =============================================================================

#[test]
fn test_reasonable_query_time() {
    use std::time::Instant;

    // Use depth=3 (8 cells, 18 qubits) to stay within 32-qubit dense limit
    let mut qram = QRAMQuery::new(3);
    let data: Vec<u64> = (0..8).collect();
    qram.load(&data);

    // Classical queries should be fast
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = qram.query_classical(5);
    }
    let classical_time = start.elapsed();

    // 1000 classical queries should take < 10ms
    assert!(
        classical_time.as_millis() < 10,
        "Classical queries too slow: {:?}",
        classical_time
    );

    // Quantum queries (depth 3 = 7 routers = 21 gates)
    let start = Instant::now();
    for i in 0..10 {
        let _ = qram.query_superposition(i as u64);
    }
    let quantum_time = start.elapsed();

    // 10 quantum queries should take < 2s (sanity check, allows for system variance)
    assert!(
        quantum_time.as_secs() < 2,
        "Quantum queries too slow: {:?}",
        quantum_time
    );
}

// =============================================================================
// Sparse Bucket-Brigade QRAM Tests (Sprint 49.0)
// =============================================================================

#[test]
fn test_sparse_qram_creation() {
    let qram = SparseBucketBrigade::new(4);
    assert_eq!(qram.depth(), 4);
    assert_eq!(qram.memory_cells(), 16);
    assert_eq!(qram.total_qubits(), 35); // 4 + 1 + 2*15
}

#[test]
fn test_sparse_qram_classical_query() {
    let mut qram = SparseBucketBrigade::new(4);
    let data: Vec<u64> = (0..16).map(|i| (i + 1) * 100).collect();
    qram.load(&data);

    for addr in 0..16 {
        assert_eq!(qram.query_classical(addr), Some((addr as u64 + 1) * 100));
    }
}

#[test]
fn test_sparse_qram_basis_query_correctness() {
    let mut qram = SparseBucketBrigade::new(3); // 8 cells
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 10).collect();
    qram.load(&data);

    // Each basis query should return the exact address we initialized
    for addr in 0..8 {
        let result = qram.query_basis(addr, addr as u64 * 1000);
        assert_eq!(
            result.address, addr,
            "Basis query for addr {} returned {}",
            addr, result.address
        );
        assert_eq!(result.data, (addr as u64 + 1) * 10);
    }
}

#[test]
fn test_sparse_qram_superposition_query_validity() {
    let mut qram = SparseBucketBrigade::new(3);
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 100).collect();
    qram.load(&data);

    // Multiple superposition queries should return valid addresses
    for seed in 0..20 {
        let result = qram.query_superposition(seed);
        assert!(
            result.address < 8,
            "Address {} out of range",
            result.address
        );
        assert_eq!(
            result.data,
            (result.address as u64 + 1) * 100,
            "Data mismatch for address {}",
            result.address
        );
    }
}

#[test]
fn test_sparse_qram_large_scale_creation() {
    // Test that we can create QRAMs with many cells
    // (not full simulation, just structure)

    let qram_256 = SparseBucketBrigade::new(8); // 256 cells
    assert_eq!(qram_256.memory_cells(), 256);
    assert_eq!(qram_256.total_qubits(), 519);

    let qram_1k = SparseBucketBrigade::new(10); // 1024 cells
    assert_eq!(qram_1k.memory_cells(), 1024);
    assert_eq!(qram_1k.total_qubits(), 2057);

    // Even larger (just creation test)
    let qram_4k = SparseBucketBrigade::new(12);
    assert_eq!(qram_4k.memory_cells(), 4096);
    assert_eq!(qram_4k.total_qubits(), 8203);
}

#[test]
fn test_sparse_vs_dense_parity() {
    // Verify sparse and dense implementations produce same classical results
    let depth = 3;
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 111).collect();

    let mut dense = QRAMQuery::new(depth);
    dense.load(&data);

    let mut sparse = SparseBucketBrigade::new(depth);
    sparse.load(&data);

    // Classical queries should match exactly
    for addr in 0..8 {
        let dense_result = dense.query_classical(addr);
        let sparse_result = sparse.query_classical(addr);
        assert_eq!(
            dense_result, sparse_result,
            "Mismatch at address {}: dense={:?}, sparse={:?}",
            addr, dense_result, sparse_result
        );
    }
}

#[test]
fn test_sparse_qram_resource_comparison() {
    let qram = SparseBucketBrigade::new(8); // 256 cells, 519 qubits
    let comparison = qram.resource_comparison();

    assert_eq!(comparison.memory_cells, 256);
    assert_eq!(comparison.total_qubits, 519);
    assert!(!comparison.is_feasible_dense); // 519 >> 32 qubits
    assert!(comparison.sparse_blocks >= 1);
    assert!(comparison.sparse_bytes > 0);
}

#[test]
fn test_sparse_qram_stats_tracking() {
    let mut qram = SparseBucketBrigade::new(3);
    qram.load(&[1, 2, 3, 4, 5, 6, 7, 8]);

    assert_eq!(qram.queries_performed(), 0);

    let _ = qram.query_basis(0, 42);
    assert_eq!(qram.queries_performed(), 1);

    let _ = qram.query_superposition(123);
    assert_eq!(qram.queries_performed(), 2);

    let stats = qram.state_stats();
    assert!(stats.gates_applied > 0, "Should have applied gates");
}

// =============================================================================
// Stabilizer QRAM Tests (Sprint 50.0)
// =============================================================================

#[test]
fn test_stabilizer_qram_creation() {
    let qram = StabilizerQRAM::new(4);
    assert_eq!(qram.depth(), 4);
    assert_eq!(qram.memory_cells(), 16);
    assert_eq!(qram.total_qubits(), 35); // 4 + 1 + 2*15
}

#[test]
fn test_stabilizer_qram_classical_query() {
    let mut qram = StabilizerQRAM::new(4);
    let data: Vec<u64> = (0..16).map(|i| (i + 1) * 100).collect();
    qram.load(&data);

    for addr in 0..16 {
        assert_eq!(qram.query_classical(addr), Some((addr as u64 + 1) * 100));
    }
}

#[test]
fn test_stabilizer_qram_clifford_query() {
    let mut qram = StabilizerQRAM::new(3);
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 10).collect();
    qram.load(&data);

    // Clifford queries should return valid addresses
    for addr in 0..8 {
        let result = qram.query_clifford(addr, addr as u64 * 1000);
        assert!(
            result.address < 8,
            "Address {} out of range",
            result.address
        );
        assert!(
            result.deterministic,
            "Basis state query should be deterministic"
        );
    }
}

#[test]
fn test_stabilizer_qram_superposition() {
    let mut qram = StabilizerQRAM::new(3);
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 100).collect();
    qram.load(&data);

    // Multiple superposition queries should return valid but varying addresses
    let mut addresses = std::collections::HashSet::new();
    for seed in 0..20 {
        let result = qram.query_clifford_superposition(seed);
        addresses.insert(result.address);
        assert!(
            result.address < 8,
            "Address {} out of range",
            result.address
        );
        assert!(
            !result.deterministic,
            "Superposition query should not be deterministic"
        );
    }

    // Should see multiple different addresses
    assert!(addresses.len() > 1, "Should measure different addresses");
}

#[test]
fn test_stabilizer_qram_resource_comparison() {
    let qram = StabilizerQRAM::new(8); // 256 cells
    let comparison = qram.resource_comparison();

    assert_eq!(comparison.memory_cells, 256);
    assert_eq!(comparison.depth, 8);
    assert!(!comparison.is_feasible_dense); // 519 qubits >> 32
    assert!(comparison.stabilizer_bytes > 0);
    assert!(comparison.compression_ratio > 1.0);
}

#[test]
fn test_stabilizer_qram_frame_tracking() {
    let mut qram = StabilizerQRAM::new(3);
    qram.load(&[1, 2, 3, 4, 5, 6, 7, 8]);

    // Query with frame tracking
    let result = qram.query_with_frame(3, 42);
    assert!(result.address < 8);

    // Should have tracked non-Clifford gates
    let stats = qram.stats();
    assert!(
        stats.non_clifford_gates > 0,
        "Should track non-Clifford gates"
    );
}

#[test]
fn test_stabilizer_vs_sparse_parity() {
    // Both should return same classical data
    let depth = 3;
    let data: Vec<u64> = (0..8).map(|i| (i + 1) * 111).collect();

    let mut stabilizer = StabilizerQRAM::new(depth);
    stabilizer.load(&data);

    let mut sparse = SparseBucketBrigade::new(depth);
    sparse.load(&data);

    // Classical queries should match exactly
    for addr in 0..8 {
        let stab_result = stabilizer.query_classical(addr);
        let sparse_result = sparse.query_classical(addr);
        assert_eq!(
            stab_result, sparse_result,
            "Mismatch at address {}: stabilizer={:?}, sparse={:?}",
            addr, stab_result, sparse_result
        );
    }
}

// =============================================================================
// Graph Router Tests (Sprint 50.0)
// =============================================================================

#[test]
fn test_graph_router_bucket_brigade() {
    let mut router = GraphRouter::new_bucket_brigade(3);
    let result = router.route_and_measure(5, 42);
    assert!(result.measured_address < 8);
    assert!(!result.path.is_empty());
}

#[test]
fn test_graph_router_superposition() {
    let mut router = GraphRouter::new_bucket_brigade(3);

    // Multiple routes should give varying addresses
    let mut addresses = std::collections::HashSet::new();
    for seed in 0..20 {
        let result = router.route_superposition(seed);
        addresses.insert(result.measured_address);
        assert!(result.measured_address < 8);
    }

    assert!(addresses.len() > 1, "Should route to different addresses");
}

#[test]
fn test_graph_router_topologies() {
    // Test different graph topologies can be created
    let bucket = GraphRouter::new_bucket_brigade(4);
    let star = GraphRouter::new_star(4);
    let path = GraphRouter::new_path(5);

    // Each should have valid structure
    assert!(bucket.graph().edge_count() > 0);
    assert_eq!(star.graph().edge_count(), 4); // Hub to 4 spokes
    assert_eq!(path.graph().edge_count(), 4); // 5 vertices, 4 edges
}
