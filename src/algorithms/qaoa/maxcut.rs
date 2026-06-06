// =============================================================================
// MaxCut Problem Implementation
// =============================================================================
//
// MaxCut: Given a graph G=(V,E), find a partition of V into two sets
// that maximizes the number (or weight) of edges crossing the partition.
//
// This is NP-hard in general, but QAOA can provide approximate solutions.
//
// =============================================================================

use crate::algorithms::qaoa::problem::Graph;

/// Evaluate the cut value for a given partition
///
/// # Arguments
/// * `graph` - The graph
/// * `partition` - Boolean vector where true = set A, false = set B
///
/// # Returns
/// Sum of edge weights crossing the partition
pub fn evaluate_cut(graph: &Graph, partition: &[bool]) -> f64 {
    assert_eq!(
        partition.len(),
        graph.n_vertices,
        "Partition size must match graph vertices"
    );

    graph
        .edges
        .iter()
        .filter(|(u, v, _)| partition[*u] != partition[*v])
        .map(|(_, _, w)| w)
        .sum()
}

/// Find optimal MaxCut by brute force enumeration
///
/// WARNING: Exponential in number of vertices! Only use for n <= 20
///
/// # Returns
/// (optimal_cut_value, optimal_partition)
pub fn brute_force_maxcut(graph: &Graph) -> (f64, Vec<bool>) {
    let n = graph.n_vertices;
    assert!(n <= 20, "Brute force only feasible for n <= 20");

    let mut best_cut = 0.0;
    let mut best_partition = vec![false; n];

    // Enumerate all 2^n partitions (but we can skip half by symmetry)
    // We fix vertex 0 to set A (false) without loss of generality
    for bits in 0..(1u64 << (n - 1)) {
        let mut partition = vec![false; n];
        for i in 1..n {
            partition[i] = (bits >> (i - 1)) & 1 == 1;
        }

        let cut = evaluate_cut(graph, &partition);
        if cut > best_cut {
            best_cut = cut;
            best_partition = partition;
        }
    }

    (best_cut, best_partition)
}

/// Compute the MaxCut approximation ratio
///
/// Ratio = achieved_cut / optimal_cut
pub fn approximation_ratio(graph: &Graph, partition: &[bool]) -> f64 {
    let achieved = evaluate_cut(graph, partition);
    let (optimal, _) = brute_force_maxcut(graph);

    if optimal == 0.0 {
        if achieved == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        achieved / optimal
    }
}

/// Known optimal MaxCut values for standard graphs
pub mod known_optima {
    /// Complete graph K_n: optimal cut = floor(n^2/4)
    pub fn complete_graph(n: usize) -> f64 {
        (n * n / 4) as f64
    }

    /// Cycle C_n: optimal cut = n - (n % 2) for n >= 3
    /// Even cycle: n edges, cut n (all edges)
    /// Odd cycle: n edges, cut n-1
    pub fn cycle(n: usize) -> f64 {
        if n % 2 == 0 { n as f64 } else { (n - 1) as f64 }
    }

    /// Star S_n: optimal cut = n-1 (all edges)
    pub fn star(n: usize) -> f64 {
        (n - 1) as f64
    }

    /// Path P_n: optimal cut = n-1 (all edges, by alternating partition)
    pub fn path(n: usize) -> f64 {
        (n - 1) as f64
    }
}

/// Convert measurement bitstring to partition
pub fn bitstring_to_partition(bitstring: u64, n_qubits: usize) -> Vec<bool> {
    (0..n_qubits).map(|i| (bitstring >> i) & 1 == 1).collect()
}

/// Sample multiple bitstrings and return the one with best cut
pub fn best_of_samples(graph: &Graph, samples: &[u64]) -> (f64, Vec<bool>) {
    let mut best_cut = f64::NEG_INFINITY;
    let mut best_partition = vec![false; graph.n_vertices];

    for &bits in samples {
        let partition = bitstring_to_partition(bits, graph.n_vertices);
        let cut = evaluate_cut(graph, &partition);

        if cut > best_cut {
            best_cut = cut;
            best_partition = partition;
        }
    }

    (best_cut, best_partition)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_cut_simple() {
        // Triangle graph
        let mut g = Graph::new(3);
        g.add_unweighted_edge(0, 1);
        g.add_unweighted_edge(1, 2);
        g.add_unweighted_edge(0, 2);

        // Partition {0} vs {1, 2}
        let partition = vec![false, true, true];
        let cut = evaluate_cut(&g, &partition);
        assert_eq!(cut, 2.0); // Edges (0,1) and (0,2) are cut
    }

    #[test]
    fn test_evaluate_cut_none() {
        let mut g = Graph::new(3);
        g.add_unweighted_edge(0, 1);
        g.add_unweighted_edge(1, 2);

        // All in same partition
        let partition = vec![true, true, true];
        let cut = evaluate_cut(&g, &partition);
        assert_eq!(cut, 0.0);
    }

    #[test]
    fn test_evaluate_cut_weighted() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 3.5);

        let partition = vec![true, false];
        let cut = evaluate_cut(&g, &partition);
        assert_eq!(cut, 3.5);
    }

    #[test]
    fn test_brute_force_triangle() {
        let mut g = Graph::new(3);
        g.add_unweighted_edge(0, 1);
        g.add_unweighted_edge(1, 2);
        g.add_unweighted_edge(0, 2);

        let (optimal, _) = brute_force_maxcut(&g);
        assert_eq!(optimal, 2.0); // Best is 2 edges
    }

    #[test]
    fn test_brute_force_square() {
        let g = Graph::cycle(4);

        let (optimal, partition) = brute_force_maxcut(&g);
        assert_eq!(optimal, 4.0); // All 4 edges can be cut

        // Verify partition is valid
        let cut = evaluate_cut(&g, &partition);
        assert_eq!(cut, optimal);
    }

    #[test]
    fn test_brute_force_complete_k4() {
        let g = Graph::complete(4);

        let (optimal, _) = brute_force_maxcut(&g);
        assert_eq!(optimal, known_optima::complete_graph(4)); // 4
    }

    #[test]
    fn test_known_optima_complete() {
        assert_eq!(known_optima::complete_graph(2), 1.0); // K_2: 1 edge
        assert_eq!(known_optima::complete_graph(3), 2.0); // K_3: floor(9/4) = 2
        assert_eq!(known_optima::complete_graph(4), 4.0); // K_4: floor(16/4) = 4
        assert_eq!(known_optima::complete_graph(5), 6.0); // K_5: floor(25/4) = 6
    }

    #[test]
    fn test_known_optima_cycle() {
        assert_eq!(known_optima::cycle(3), 2.0); // Odd cycle
        assert_eq!(known_optima::cycle(4), 4.0); // Even cycle
        assert_eq!(known_optima::cycle(5), 4.0); // Odd cycle
        assert_eq!(known_optima::cycle(6), 6.0); // Even cycle
    }

    #[test]
    fn test_approximation_ratio() {
        let g = Graph::complete(4);
        let partition = vec![true, true, false, false]; // 2-2 split

        let ratio = approximation_ratio(&g, &partition);
        assert!((ratio - 1.0).abs() < 1e-6); // This is optimal for K_4
    }

    #[test]
    fn test_bitstring_to_partition() {
        let partition = bitstring_to_partition(0b1010, 4);
        assert_eq!(partition, vec![false, true, false, true]);
    }

    #[test]
    fn test_best_of_samples() {
        let g = Graph::complete(3);
        let samples = vec![0b000, 0b001, 0b011, 0b111];

        let (best_cut, _) = best_of_samples(&g, &samples);
        assert_eq!(best_cut, 2.0); // 0b001 or 0b011 give cut of 2
    }
}
