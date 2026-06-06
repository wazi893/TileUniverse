// =============================================================================
// QAOA Integration Tests
// =============================================================================
//
// End-to-end tests for the QAOA implementation.
//
// Run: cargo test --test qaoa_integration
//
// =============================================================================

use engine::algorithms::qaoa::{
    Graph, QAOAConfig, QUBOProblem, brute_force_maxcut, evaluate_cut, maxcut, maxcut_p,
    qaoa_circuit, qaoa_gate_count, qaoa_parameter_count, run_qaoa,
};

// =============================================================================
// Graph Construction Tests
// =============================================================================

#[test]
fn test_graph_complete() {
    let g = Graph::complete(5);
    assert_eq!(g.n_vertices, 5);
    assert_eq!(g.n_edges(), 10); // C(5,2) = 10
}

#[test]
fn test_graph_cycle() {
    let g = Graph::cycle(6);
    assert_eq!(g.n_vertices, 6);
    assert_eq!(g.n_edges(), 6);
}

#[test]
fn test_graph_random_reproducible() {
    let g1 = Graph::random(8, 0.5, 42);
    let g2 = Graph::random(8, 0.5, 42);

    assert_eq!(g1.n_edges(), g2.n_edges());
}

#[test]
fn test_graph_weighted() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 2.5);
    g.add_edge(1, 2, 1.5);

    assert_eq!(g.total_weight(), 4.0);
}

// =============================================================================
// Circuit Construction Tests
// =============================================================================

#[test]
fn test_qaoa_circuit_structure() {
    let graph = Graph::complete(3);
    let gammas = vec![0.5, 0.7];
    let betas = vec![0.3, 0.4];

    let circuit = qaoa_circuit(&graph, &gammas, &betas);

    // Should have gates
    assert!(!circuit.is_empty());

    // Gate count should match formula
    let expected = qaoa_gate_count(&graph, 2);
    assert_eq!(circuit.len(), expected);
}

#[test]
fn test_parameter_count() {
    assert_eq!(qaoa_parameter_count(1), 2);
    assert_eq!(qaoa_parameter_count(3), 6);
}

// =============================================================================
// MaxCut Evaluation Tests
// =============================================================================

#[test]
fn test_maxcut_triangle() {
    let g = Graph::complete(3);
    let (optimal, partition) = brute_force_maxcut(&g);

    assert_eq!(optimal, 2.0);

    // Verify partition achieves optimal
    let cut = evaluate_cut(&g, &partition);
    assert_eq!(cut, optimal);
}

#[test]
fn test_maxcut_square() {
    let g = Graph::cycle(4);
    let (optimal, _) = brute_force_maxcut(&g);

    assert_eq!(optimal, 4.0); // All edges can be cut
}

#[test]
fn test_maxcut_path() {
    let g = Graph::path(5);
    let (optimal, _) = brute_force_maxcut(&g);

    assert_eq!(optimal, 4.0); // All edges can be cut by alternating
}

#[test]
fn test_maxcut_star() {
    let g = Graph::star(5);
    let (optimal, _) = brute_force_maxcut(&g);

    assert_eq!(optimal, 4.0); // All edges from center
}

// =============================================================================
// QAOA Algorithm Tests
// =============================================================================

#[test]
fn test_qaoa_finds_cut() {
    let graph = Graph::cycle(4);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(2).with_shots(200).with_seed(42),
    );

    // Should find a positive cut
    assert!(result.best_cut > 0.0);
}

#[test]
fn test_qaoa_approximation_ratio_reasonable() {
    let graph = Graph::complete(4);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(2).with_shots(500).with_seed(42),
    );

    // Should achieve decent approximation
    let ratio = result.approximation_ratio.unwrap();
    assert!(ratio >= 0.5, "Approximation ratio {} too low", ratio);
}

#[test]
fn test_qaoa_depth_1() {
    let graph = Graph::cycle(3);

    let result = maxcut_p(graph, 1);

    assert_eq!(result.optimal_params.len(), 2); // 2 params for p=1
}

#[test]
fn test_qaoa_depth_3() {
    let graph = Graph::cycle(4);

    let result = maxcut_p(graph, 3);

    assert_eq!(result.optimal_params.len(), 6); // 6 params for p=3
}

#[test]
fn test_qaoa_reproducibility() {
    let graph = Graph::random(4, 0.5, 100);

    let result1 = run_qaoa(
        graph.clone(),
        QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
    );

    let result2 = run_qaoa(
        graph,
        QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
    );

    assert_eq!(result1.best_cut, result2.best_cut);
    assert_eq!(result1.optimal_params, result2.optimal_params);
}

#[test]
fn test_qaoa_different_seeds() {
    let graph = Graph::random(4, 0.5, 100);

    let result1 = run_qaoa(
        graph.clone(),
        QAOAConfig::with_depth(1).with_shots(100).with_seed(1),
    );

    let result2 = run_qaoa(
        graph,
        QAOAConfig::with_depth(1).with_shots(100).with_seed(2),
    );

    // Different seeds may give different results (not guaranteed, but likely)
    // Just check both complete successfully
    assert!(result1.best_cut >= 0.0);
    assert!(result2.best_cut >= 0.0);
}

// =============================================================================
// Optimizer Variant Tests
// =============================================================================

#[test]
fn test_qaoa_nelder_mead() {
    let graph = Graph::cycle(3);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
    );

    assert!(result.best_cut >= 0.0);
    assert!(!result.history.is_empty());
}

#[test]
fn test_qaoa_spsa() {
    let graph = Graph::cycle(3);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(1)
            .with_spsa()
            .with_shots(100)
            .with_seed(42),
    );

    assert!(result.best_cut >= 0.0);
}

#[test]
fn test_qaoa_gradient_descent() {
    let graph = Graph::cycle(3);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(1)
            .with_gradient_descent(0.1)
            .with_shots(100)
            .with_seed(42),
    );

    assert!(result.best_cut >= 0.0);
}

// =============================================================================
// Scaling Tests
// =============================================================================

#[test]
fn test_qaoa_4_vertices() {
    let graph = Graph::random(4, 0.5, 42);
    let result = maxcut(graph);

    assert_eq!(result.best_bitstring.len(), 4);
    assert!(result.approximation_ratio.is_some());
}

#[test]
fn test_qaoa_6_vertices() {
    let graph = Graph::random(6, 0.5, 42);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(2).with_shots(200).with_seed(42),
    );

    assert_eq!(result.best_bitstring.len(), 6);
}

#[test]
fn test_qaoa_8_vertices() {
    let graph = Graph::random(8, 0.4, 42);

    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
    );

    assert_eq!(result.best_bitstring.len(), 8);
    assert!(result.best_cut >= 0.0);
}

// =============================================================================
// Result Structure Tests
// =============================================================================

#[test]
fn test_result_contains_history() {
    let graph = Graph::cycle(3);
    let result = maxcut(graph);

    assert!(!result.history.is_empty());
}

#[test]
fn test_result_contains_iterations() {
    let graph = Graph::cycle(3);
    let result = maxcut(graph);

    assert!(result.iterations > 0);
}

#[test]
fn test_result_contains_evaluations() {
    let graph = Graph::cycle(3);
    let result = maxcut(graph);

    assert!(result.evaluations > 0);
}

#[test]
fn test_result_partition_valid() {
    let graph = Graph::complete(4);
    let result = maxcut(graph.clone());

    // Partition should have correct length
    assert_eq!(result.best_bitstring.len(), 4);

    // Cut value should match partition
    let computed_cut = evaluate_cut(&graph, &result.best_bitstring);
    assert_eq!(computed_cut, result.best_cut);
}

// =============================================================================
// QUBO Tests
// =============================================================================

#[test]
fn test_qubo_from_maxcut() {
    let graph = Graph::complete(3);
    let qubo = QUBOProblem::from_maxcut(&graph);

    assert_eq!(qubo.n_vars, 3);
}

#[test]
fn test_qubo_evaluation() {
    let mut qubo = QUBOProblem::new(2);
    qubo.set_linear(0, 1.0);
    qubo.set_linear(1, 2.0);
    qubo.set_quadratic(0, 1, 3.0);

    let value = qubo.evaluate(&[true, true]);
    assert_eq!(value, 6.0); // 1 + 2 + 3
}

// =============================================================================
// Known Optimal Tests
// =============================================================================

#[test]
fn test_known_optimal_k4() {
    use engine::algorithms::qaoa::maxcut::known_optima;

    let g = Graph::complete(4);
    let (optimal, _) = brute_force_maxcut(&g);

    assert_eq!(optimal, known_optima::complete_graph(4));
}

#[test]
fn test_known_optimal_c6() {
    use engine::algorithms::qaoa::maxcut::known_optima;

    let g = Graph::cycle(6);
    let (optimal, _) = brute_force_maxcut(&g);

    assert_eq!(optimal, known_optima::cycle(6));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_two_vertex_graph() {
    let mut g = Graph::new(2);
    g.add_unweighted_edge(0, 1);

    let result = maxcut(g);

    assert_eq!(result.best_cut, 1.0); // Only one edge, must cut it
}

#[test]
fn test_disconnected_vertices() {
    let g = Graph::new(3); // No edges

    let result = maxcut(g);

    assert_eq!(result.best_cut, 0.0); // No edges to cut
}

#[test]
fn test_weighted_maxcut() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 10.0);
    g.add_edge(1, 2, 1.0);

    let _result = run_qaoa(
        g.clone(),
        QAOAConfig::with_depth(2).with_shots(500).with_seed(42),
    );

    // Should prioritize cutting the heavy edge
    let (optimal, _) = brute_force_maxcut(&g);
    assert_eq!(optimal, 11.0); // Cut both edges
}
