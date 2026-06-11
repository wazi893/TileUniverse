// =============================================================================
// QAOA Problem Representation
// =============================================================================
//
// Data structures for combinatorial optimization problems.
//
// Supports:
// - Undirected weighted graphs (for MaxCut, graph coloring)
// - QUBO (Quadratic Unconstrained Binary Optimization)
//
// =============================================================================

use std::collections::HashMap;

/// Undirected weighted graph for optimization problems
#[derive(Clone, Debug)]
pub struct Graph {
    /// Number of vertices
    pub n_vertices: usize,
    /// Edges as (u, v, weight) tuples
    pub edges: Vec<(usize, usize, f64)>,
}

impl Graph {
    /// Create an empty graph with n vertices
    pub fn new(n_vertices: usize) -> Self {
        Self {
            n_vertices,
            edges: Vec::new(),
        }
    }

    /// Add an edge between vertices u and v with given weight
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        assert!(u < self.n_vertices, "Vertex u out of bounds");
        assert!(v < self.n_vertices, "Vertex v out of bounds");
        assert!(u != v, "Self-loops not allowed");
        self.edges.push((u, v, weight));
    }

    /// Add an unweighted edge (weight = 1.0)
    pub fn add_unweighted_edge(&mut self, u: usize, v: usize) {
        self.add_edge(u, v, 1.0);
    }

    /// Create graph from adjacency matrix
    /// Only upper triangle is used (symmetric/undirected graph)
    pub fn from_adjacency_matrix(adj: &[Vec<f64>]) -> Self {
        let n = adj.len();
        let mut graph = Self::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] != 0.0 {
                    graph.add_edge(i, j, adj[i][j]);
                }
            }
        }

        graph
    }

    /// Total number of edges
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Total weight of all edges
    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|(_, _, w)| w).sum()
    }

    // =========================================================================
    // Standard Graph Generators
    // =========================================================================

    /// Complete graph K_n (all pairs connected)
    pub fn complete(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                graph.add_unweighted_edge(i, j);
            }
        }
        graph
    }

    /// Cycle graph C_n (ring topology)
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3, "Cycle requires at least 3 vertices");
        let mut graph = Self::new(n);
        for i in 0..n {
            graph.add_unweighted_edge(i, (i + 1) % n);
        }
        graph
    }

    /// Path graph P_n (linear chain)
    pub fn path(n: usize) -> Self {
        assert!(n >= 2, "Path requires at least 2 vertices");
        let mut graph = Self::new(n);
        for i in 0..(n - 1) {
            graph.add_unweighted_edge(i, i + 1);
        }
        graph
    }

    /// Star graph S_n (one hub connected to all others)
    pub fn star(n: usize) -> Self {
        assert!(n >= 2, "Star requires at least 2 vertices");
        let mut graph = Self::new(n);
        for i in 1..n {
            graph.add_unweighted_edge(0, i);
        }
        graph
    }

    /// Random Erdos-Renyi graph G(n, p)
    /// Each edge exists independently with probability p
    pub fn random(n: usize, p: f64, seed: u64) -> Self {
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

        let mut graph = Self::new(n);
        let mut rng_state = seed;

        for i in 0..n {
            for j in (i + 1)..n {
                // Simple LCG for reproducibility
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

                if rand_val < p {
                    graph.add_unweighted_edge(i, j);
                }
            }
        }

        graph
    }

    /// Random weighted graph with weights in [0, 1]
    pub fn random_weighted(n: usize, p: f64, seed: u64) -> Self {
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

        let mut graph = Self::new(n);
        let mut rng_state = seed;

        for i in 0..n {
            for j in (i + 1)..n {
                // Edge existence
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

                if rand_val < p {
                    // Edge weight
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let weight = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    graph.add_edge(i, j, weight);
                }
            }
        }

        graph
    }

    /// k-regular graph (each vertex has exactly k neighbors)
    /// Note: n*k must be even for this to be possible
    pub fn regular(n: usize, k: usize, _seed: u64) -> Self {
        assert!(k < n, "Degree k must be less than n");
        assert!(
            (n * k).is_multiple_of(2),
            "n*k must be even for regular graph"
        );

        // Simple construction: cycle + chords
        // For k=2, this is just a cycle
        // For higher k, we add additional regular connections
        let mut graph = Self::new(n);

        // Add edges to form k-regular structure
        for offset in 1..=(k / 2) {
            for i in 0..n {
                let j = (i + offset) % n;
                graph.add_unweighted_edge(i.min(j), i.max(j));
            }
        }

        // If k is odd, need perfect matching (only works for even n)
        if k % 2 == 1 {
            assert!(n.is_multiple_of(2), "Odd k requires even n");
            for i in 0..(n / 2) {
                graph.add_unweighted_edge(i, i + n / 2);
            }
        }

        // Remove duplicate edges that might have been added
        let mut seen = std::collections::HashSet::new();
        graph.edges.retain(|(u, v, _)| {
            let key = (*u.min(v), *u.max(v));
            seen.insert(key)
        });

        graph
    }
}

// =============================================================================
// QUBO Problem Representation
// =============================================================================

/// Quadratic Unconstrained Binary Optimization problem
///
/// Minimize: x^T Q x + c^T x
/// where x ∈ {0, 1}^n
///
/// Can represent MaxCut, graph coloring, TSP, etc.
#[derive(Clone, Debug)]
pub struct QUBOProblem {
    /// Number of binary variables
    pub n_vars: usize,
    /// Quadratic terms Q_ij (only upper triangle stored)
    pub quadratic: HashMap<(usize, usize), f64>,
    /// Linear terms c_i
    pub linear: Vec<f64>,
}

impl QUBOProblem {
    /// Create empty QUBO with n variables
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            quadratic: HashMap::new(),
            linear: vec![0.0; n_vars],
        }
    }

    /// Set linear coefficient for variable i
    pub fn set_linear(&mut self, i: usize, coeff: f64) {
        assert!(i < self.n_vars, "Variable index out of bounds");
        self.linear[i] = coeff;
    }

    /// Set quadratic coefficient for variables i, j
    pub fn set_quadratic(&mut self, i: usize, j: usize, coeff: f64) {
        assert!(
            i < self.n_vars && j < self.n_vars,
            "Variable index out of bounds"
        );
        let key = (i.min(j), i.max(j));
        self.quadratic.insert(key, coeff);
    }

    /// Evaluate QUBO for a given binary assignment
    pub fn evaluate(&self, x: &[bool]) -> f64 {
        assert_eq!(x.len(), self.n_vars, "Assignment size mismatch");

        let mut value = 0.0;

        // Linear terms
        for (i, &coeff) in self.linear.iter().enumerate() {
            if x[i] {
                value += coeff;
            }
        }

        // Quadratic terms
        for (&(i, j), &coeff) in &self.quadratic {
            if x[i] && x[j] {
                value += coeff;
            }
        }

        value
    }

    /// Convert MaxCut on a graph to QUBO
    ///
    /// MaxCut: maximize Σ_{(i,j)∈E} w_ij * (x_i ⊕ x_j)
    /// Equivalent QUBO: minimize Σ_{(i,j)∈E} w_ij * (2*x_i*x_j - x_i - x_j)
    pub fn from_maxcut(graph: &Graph) -> Self {
        let mut qubo = Self::new(graph.n_vertices);

        for &(i, j, weight) in &graph.edges {
            // Quadratic term: 2 * w_ij * x_i * x_j
            qubo.set_quadratic(i, j, 2.0 * weight);

            // Linear terms: -w_ij * x_i - w_ij * x_j
            qubo.linear[i] -= weight;
            qubo.linear[j] -= weight;
        }

        qubo
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);

        assert_eq!(g.n_vertices, 4);
        assert_eq!(g.n_edges(), 2);
        assert_eq!(g.total_weight(), 3.0);
    }

    #[test]
    fn test_complete_graph() {
        let g = Graph::complete(4);
        assert_eq!(g.n_vertices, 4);
        assert_eq!(g.n_edges(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_cycle_graph() {
        let g = Graph::cycle(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.n_edges(), 5);
    }

    #[test]
    fn test_path_graph() {
        let g = Graph::path(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.n_edges(), 4);
    }

    #[test]
    fn test_star_graph() {
        let g = Graph::star(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.n_edges(), 4);
    }

    #[test]
    fn test_random_graph_reproducibility() {
        let g1 = Graph::random(10, 0.5, 42);
        let g2 = Graph::random(10, 0.5, 42);

        assert_eq!(g1.n_edges(), g2.n_edges());
        for (e1, e2) in g1.edges.iter().zip(g2.edges.iter()) {
            assert_eq!(e1, e2);
        }
    }

    #[test]
    fn test_random_graph_density() {
        // With p=1.0, should be complete graph
        let g = Graph::random(5, 1.0, 42);
        assert_eq!(g.n_edges(), 10); // C(5,2) = 10

        // With p=0.0, should be empty
        let g = Graph::random(5, 0.0, 42);
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn test_adjacency_matrix() {
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 2.0],
            vec![0.0, 2.0, 0.0],
        ];
        let g = Graph::from_adjacency_matrix(&adj);

        assert_eq!(g.n_vertices, 3);
        assert_eq!(g.n_edges(), 2);
    }

    #[test]
    fn test_regular_graph() {
        // 4-regular graph on 8 vertices
        let g = Graph::regular(8, 4, 42);
        assert_eq!(g.n_vertices, 8);
        // Each vertex has degree 4, so total degree = 32, edges = 16
        assert_eq!(g.n_edges(), 16);
    }

    #[test]
    fn test_qubo_creation() {
        let mut qubo = QUBOProblem::new(3);
        qubo.set_linear(0, 1.0);
        qubo.set_quadratic(0, 1, 2.0);

        assert_eq!(qubo.n_vars, 3);
        assert_eq!(qubo.linear[0], 1.0);
        assert_eq!(qubo.quadratic.get(&(0, 1)), Some(&2.0));
    }

    #[test]
    fn test_qubo_evaluate() {
        let mut qubo = QUBOProblem::new(3);
        qubo.set_linear(0, 1.0);
        qubo.set_linear(1, 2.0);
        qubo.set_quadratic(0, 1, 3.0);

        // x = [1, 0, 0]: value = 1.0
        assert_eq!(qubo.evaluate(&[true, false, false]), 1.0);

        // x = [0, 1, 0]: value = 2.0
        assert_eq!(qubo.evaluate(&[false, true, false]), 2.0);

        // x = [1, 1, 0]: value = 1.0 + 2.0 + 3.0 = 6.0
        assert_eq!(qubo.evaluate(&[true, true, false]), 6.0);
    }

    #[test]
    fn test_qubo_from_maxcut() {
        // Triangle graph
        let mut g = Graph::new(3);
        g.add_unweighted_edge(0, 1);
        g.add_unweighted_edge(1, 2);
        g.add_unweighted_edge(0, 2);

        let qubo = QUBOProblem::from_maxcut(&g);
        assert_eq!(qubo.n_vars, 3);

        // MaxCut on triangle: optimal is 2 edges
        // Partition {0} vs {1, 2}: cuts edges (0,1) and (0,2) = 2
        // In QUBO formulation, we minimize, so need to negate for MaxCut
    }
}
