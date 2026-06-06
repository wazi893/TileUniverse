// =============================================================================
// QAOA Circuit Construction
// =============================================================================
//
// Builds parameterized QAOA circuits for optimization problems.
//
// QAOA circuit structure:
// 1. Initial state: |+⟩^⊗n (uniform superposition)
// 2. p layers of:
//    - Cost unitary:  U_C(γ) = exp(-iγH_C)
//    - Mixer unitary: U_M(β) = exp(-iβH_M)
//
// For MaxCut:
// - H_C = Σ_{(i,j)∈E} w_ij (1 - Z_i Z_j) / 2
// - H_M = Σ_i X_i
//
// =============================================================================

use crate::algorithms::qaoa::problem::Graph;
use crate::quantum::QGate;

/// Generate cost unitary gates for MaxCut on a graph
///
/// U_C(γ) = exp(-iγH_C) where H_C = Σ w_ij (1 - Z_i Z_j) / 2
///
/// For each edge (i,j) with weight w:
/// exp(-iγw Z_i Z_j / 2) = CNOT(i,j) · Rz(j, γw) · CNOT(i,j)
///
/// The constant term exp(-iγw/2) is a global phase and ignored.
pub fn cost_unitary(graph: &Graph, gamma: f64) -> Vec<QGate> {
    let mut gates = Vec::new();

    for &(u, v, weight) in &graph.edges {
        // ZZ rotation: exp(-i γ w Z_i Z_j / 2)
        // Decomposition: CNOT - Rz - CNOT
        gates.push(QGate::CNot(u as u8, v as u8));
        gates.push(QGate::Rz(v as u8, (gamma * weight) as f32));
        gates.push(QGate::CNot(u as u8, v as u8));
    }

    gates
}

/// Generate mixer unitary gates
///
/// U_M(β) = exp(-iβH_M) where H_M = Σ_i X_i
///
/// This decomposes as: Π_i exp(-iβ X_i) = Π_i Rx(2β)
pub fn mixer_unitary(n_qubits: usize, beta: f64) -> Vec<QGate> {
    (0..n_qubits)
        .map(|q| QGate::Rx(q as u8, (2.0 * beta) as f32))
        .collect()
}

/// Generate initial state preparation (|+⟩^⊗n)
pub fn initial_state(n_qubits: usize) -> Vec<QGate> {
    (0..n_qubits).map(|q| QGate::H(q as u8)).collect()
}

/// Build complete QAOA circuit with p layers
///
/// # Arguments
/// * `graph` - The optimization problem graph
/// * `gammas` - Cost unitary parameters (one per layer)
/// * `betas` - Mixer unitary parameters (one per layer)
///
/// # Returns
/// Complete gate sequence for QAOA circuit
pub fn qaoa_circuit(graph: &Graph, gammas: &[f64], betas: &[f64]) -> Vec<QGate> {
    assert_eq!(
        gammas.len(),
        betas.len(),
        "Must have same number of γ and β parameters"
    );

    let p = gammas.len();
    let n = graph.n_vertices;

    let mut circuit = Vec::new();

    // Initial state: |+⟩^⊗n
    circuit.extend(initial_state(n));

    // p layers of cost + mixer
    for layer in 0..p {
        circuit.extend(cost_unitary(graph, gammas[layer]));
        circuit.extend(mixer_unitary(n, betas[layer]));
    }

    circuit
}

/// Count gates in QAOA circuit
pub fn qaoa_gate_count(graph: &Graph, p: usize) -> usize {
    let n = graph.n_vertices;
    let m = graph.n_edges();

    // Initial: n Hadamards
    // Per layer: 3*m (ZZ rotations) + n (Rx mixers)
    n + p * (3 * m + n)
}

/// Count parameters in QAOA circuit
pub fn qaoa_parameter_count(p: usize) -> usize {
    2 * p // p gammas + p betas
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let gates = initial_state(4);
        assert_eq!(gates.len(), 4);
        for (i, gate) in gates.iter().enumerate() {
            assert!(matches!(gate, QGate::H(q) if *q == i as u8));
        }
    }

    #[test]
    fn test_mixer_unitary() {
        let gates = mixer_unitary(3, 0.5);
        assert_eq!(gates.len(), 3);

        for gate in &gates {
            match gate {
                QGate::Rx(_, angle) => {
                    assert!((*angle - 1.0).abs() < 1e-6); // 2 * 0.5 = 1.0
                }
                _ => panic!("Expected Rx gate"),
            }
        }
    }

    #[test]
    fn test_cost_unitary_single_edge() {
        let mut graph = Graph::new(2);
        graph.add_unweighted_edge(0, 1);

        let gates = cost_unitary(&graph, 0.5);

        // Should have 3 gates: CNOT, Rz, CNOT
        assert_eq!(gates.len(), 3);
        assert!(matches!(gates[0], QGate::CNot(0, 1)));
        assert!(matches!(gates[1], QGate::Rz(1, _)));
        assert!(matches!(gates[2], QGate::CNot(0, 1)));
    }

    #[test]
    fn test_cost_unitary_multiple_edges() {
        let graph = Graph::cycle(4); // 4 edges

        let gates = cost_unitary(&graph, 0.5);

        // 3 gates per edge * 4 edges = 12 gates
        assert_eq!(gates.len(), 12);

        // Count gate types
        let cnot_count = gates
            .iter()
            .filter(|g| matches!(g, QGate::CNot(_, _)))
            .count();
        let rz_count = gates
            .iter()
            .filter(|g| matches!(g, QGate::Rz(_, _)))
            .count();

        assert_eq!(cnot_count, 8); // 2 per edge
        assert_eq!(rz_count, 4); // 1 per edge
    }

    #[test]
    fn test_cost_unitary_weighted() {
        let mut graph = Graph::new(2);
        graph.add_edge(0, 1, 2.0);

        let gates = cost_unitary(&graph, 0.5);

        // Rz angle should be gamma * weight = 0.5 * 2.0 = 1.0
        match gates[1] {
            QGate::Rz(_, angle) => assert!((angle - 1.0).abs() < 1e-6),
            _ => panic!("Expected Rz gate"),
        }
    }

    #[test]
    fn test_qaoa_circuit_structure() {
        let graph = Graph::cycle(3); // Triangle
        let gammas = vec![0.5, 0.7];
        let betas = vec![0.3, 0.4];

        let circuit = qaoa_circuit(&graph, &gammas, &betas);

        // Initial: 3 H gates
        // Layer 1: 9 cost gates (3 edges * 3) + 3 mixer gates
        // Layer 2: 9 cost gates + 3 mixer gates
        // Total: 3 + 12 + 12 = 27
        assert_eq!(circuit.len(), 27);

        // First 3 gates should be Hadamards
        for i in 0..3 {
            assert!(matches!(circuit[i], QGate::H(_)));
        }
    }

    #[test]
    fn test_qaoa_circuit_p1() {
        let mut graph = Graph::new(2);
        graph.add_unweighted_edge(0, 1);

        let circuit = qaoa_circuit(&graph, &[0.5], &[0.3]);

        // Initial: 2 H gates
        // Layer: 3 cost gates + 2 mixer gates
        // Total: 7
        assert_eq!(circuit.len(), 7);
    }

    #[test]
    fn test_gate_count() {
        let graph = Graph::complete(4); // 6 edges

        // p=1: 4 + 1*(3*6 + 4) = 4 + 22 = 26
        assert_eq!(qaoa_gate_count(&graph, 1), 26);

        // p=2: 4 + 2*(3*6 + 4) = 4 + 44 = 48
        assert_eq!(qaoa_gate_count(&graph, 2), 48);
    }

    #[test]
    fn test_parameter_count() {
        assert_eq!(qaoa_parameter_count(1), 2);
        assert_eq!(qaoa_parameter_count(3), 6);
        assert_eq!(qaoa_parameter_count(5), 10);
    }

    #[test]
    #[should_panic(expected = "Must have same number")]
    fn test_mismatched_parameters() {
        let graph = Graph::cycle(3);
        let _ = qaoa_circuit(&graph, &[0.5, 0.6], &[0.3]); // Different lengths
    }
}
