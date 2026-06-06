//! Graph State Support for Sparse Quantum Simulation
//!
//! Graph states are quantum states defined by an undirected graph G = (V, E):
//! 1. Initialize: |+⟩^⊗n  (all qubits in |+⟩ = (|0⟩+|1⟩)/√2)
//! 2. Apply CZ gates: For each edge (i,j) ∈ E, apply CZ_ij
//! 3. Result: |G⟩ (the graph state)
//!
//! Properties:
//! - Star graph S_n: 2 non-zero amplitudes (equivalent to GHZ!)
//! - Path graph P_n: 2 non-zero amplitudes
//! - Cycle graph C_n: 4 non-zero amplitudes
//! - 2D cluster state (grid): 2^O(√n) amplitudes (polynomial for fixed width)
//!
//! Connection to Stabilizer Formalism:
//! Graph states are a subset of stabilizer states. Each vertex has a stabilizer:
//!   K_i = X_i ∏_(j∈N(i)) Z_j
//! where N(i) is the set of neighbors of vertex i.
//!
//! The Gottesman-Knill theorem states that Clifford circuits on stabilizer states
//! can be simulated in polynomial time. Our block-sparse method provides an
//! alternative representation that works for ALL sparse states, not just stabilizers.

use std::collections::HashSet;

/// Graph representation for graph states
#[derive(Clone, Debug)]
pub struct Graph {
    /// Number of vertices (qubits)
    pub n_vertices: usize,
    /// Edges as (i, j) pairs where i < j
    pub edges: Vec<(usize, usize)>,
}

impl Graph {
    /// Create an empty graph with n vertices and no edges
    pub fn empty(n: usize) -> Self {
        Self {
            n_vertices: n,
            edges: Vec::new(),
        }
    }

    /// Create a star graph S_n (one central vertex connected to all others)
    ///
    /// Properties:
    /// - Edges: (0,1), (0,2), ..., (0,n-1)
    /// - Chromatic number: 2
    /// - Max degree: n-1
    /// - Graph state sparsity: 2 amplitudes (equivalent to GHZ!)
    ///
    /// ```text
    ///       1
    ///       |
    ///   2 - 0 - 4
    ///       |
    ///       3
    /// ```
    pub fn star(n: usize) -> Self {
        assert!(n >= 2, "Star graph requires at least 2 vertices");
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a path graph P_n (linear chain)
    ///
    /// Properties:
    /// - Edges: (0,1), (1,2), ..., (n-2,n-1)
    /// - Chromatic number: 2
    /// - Max degree: 2
    /// - Graph state sparsity: 2 amplitudes
    ///
    /// ```text
    /// 0 - 1 - 2 - 3 - ... - n-1
    /// ```
    pub fn path(n: usize) -> Self {
        assert!(n >= 2, "Path graph requires at least 2 vertices");
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a cycle graph C_n (circular)
    ///
    /// Properties:
    /// - Edges: (0,1), (1,2), ..., (n-2,n-1), (0,n-1)
    /// - Chromatic number: 2 (even n) or 3 (odd n)
    /// - Max degree: 2
    /// - Graph state sparsity: 4 amplitudes
    ///
    /// ```text
    ///     0
    ///    / \
    ///   4   1
    ///   |   |
    ///   3 - 2
    /// ```
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3, "Cycle graph requires at least 3 vertices");
        let mut edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        edges.push((0, n - 1));
        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a complete graph K_n (all pairs connected)
    ///
    /// Properties:
    /// - Edges: C(n,2) = n(n-1)/2 edges
    /// - Chromatic number: n
    /// - Max degree: n-1
    /// - Graph state sparsity: 2^(n-1) amplitudes (exponential!)
    ///
    /// Warning: Complete graphs lead to dense states, defeating sparsity.
    pub fn complete(n: usize) -> Self {
        assert!(n >= 2, "Complete graph requires at least 2 vertices");
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a 2D grid graph (cluster state substrate)
    ///
    /// Properties:
    /// - Vertices: rows × cols
    /// - Chromatic number: 2 (bipartite)
    /// - Max degree: 4 (interior), 2-3 (edges/corners)
    /// - Graph state sparsity: O(2^min(rows,cols)) typically
    ///
    /// Applications: Measurement-based quantum computation (MBQC)
    ///
    /// ```text
    /// 0 - 1 - 2
    /// |   |   |
    /// 3 - 4 - 5
    /// |   |   |
    /// 6 - 7 - 8
    /// ```
    pub fn grid_2d(rows: usize, cols: usize) -> Self {
        assert!(rows >= 1 && cols >= 1, "Grid must have positive dimensions");
        let n = rows * cols;
        let mut edges = Vec::new();

        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                // Horizontal edge
                if c + 1 < cols {
                    edges.push((v, v + 1));
                }
                // Vertical edge
                if r + 1 < rows {
                    edges.push((v, v + cols));
                }
            }
        }

        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a random Erdős-Rényi graph G(n, p)
    ///
    /// Each possible edge exists with probability p.
    /// Uses deterministic seeding for reproducibility.
    ///
    /// Properties:
    /// - Expected edges: p × C(n,2)
    /// - Threshold for connectivity: p ≈ ln(n)/n
    pub fn random_erdos_renyi(n: usize, p: f64, seed: u64) -> Self {
        assert!(n >= 2, "Random graph requires at least 2 vertices");
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

        let mut edges = Vec::new();

        // Simple LCG random number generator for reproducibility
        let mut rng_state = seed;
        let next_random = |state: &mut u64| -> f64 {
            // LCG parameters from Numerical Recipes
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state as f64) / (u64::MAX as f64)
        };

        for i in 0..n {
            for j in (i + 1)..n {
                if next_random(&mut rng_state) < p {
                    edges.push((i, j));
                }
            }
        }

        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a ladder graph (two parallel paths connected by rungs)
    ///
    /// ```text
    /// 0 - 2 - 4 - 6
    /// |   |   |   |
    /// 1 - 3 - 5 - 7
    /// ```
    pub fn ladder(n_rungs: usize) -> Self {
        assert!(n_rungs >= 2, "Ladder requires at least 2 rungs");
        let n = 2 * n_rungs;
        let mut edges = Vec::new();

        for i in 0..n_rungs {
            // Rung
            edges.push((2 * i, 2 * i + 1));
            // Horizontal connections
            if i + 1 < n_rungs {
                edges.push((2 * i, 2 * (i + 1)));
                edges.push((2 * i + 1, 2 * (i + 1) + 1));
            }
        }

        Self {
            n_vertices: n,
            edges,
        }
    }

    /// Create a hypercube graph Q_n (n-dimensional hypercube)
    ///
    /// Properties:
    /// - Vertices: 2^n
    /// - Edges: n × 2^(n-1)
    /// - Each vertex connected to n others (differ by one bit)
    pub fn hypercube(dimension: usize) -> Self {
        assert!(dimension >= 1, "Hypercube dimension must be at least 1");
        assert!(dimension <= 20, "Hypercube too large (2^n vertices)");

        let n = 1 << dimension;
        let mut edges = Vec::new();

        for v in 0..n {
            for bit in 0..dimension {
                let neighbor = v ^ (1 << bit);
                if v < neighbor {
                    edges.push((v, neighbor));
                }
            }
        }

        Self {
            n_vertices: n,
            edges,
        }
    }

    // =========================================================================
    // Graph Properties
    // =========================================================================

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, v: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for &(i, j) in &self.edges {
            if i == v {
                result.push(j);
            } else if j == v {
                result.push(i);
            }
        }
        result.sort();
        result
    }

    /// Get adjacency list representation
    pub fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.n_vertices];
        for &(i, j) in &self.edges {
            adj[i].push(j);
            adj[j].push(i);
        }
        for neighbors in &mut adj {
            neighbors.sort();
        }
        adj
    }

    /// Compute the degree of each vertex
    pub fn degrees(&self) -> Vec<usize> {
        let mut deg = vec![0; self.n_vertices];
        for &(i, j) in &self.edges {
            deg[i] += 1;
            deg[j] += 1;
        }
        deg
    }

    /// Maximum degree in the graph
    pub fn max_degree(&self) -> usize {
        self.degrees().into_iter().max().unwrap_or(0)
    }

    /// Approximate chromatic number using greedy coloring
    ///
    /// Returns an upper bound on the chromatic number.
    /// For many graph families, this is tight or near-tight.
    pub fn chromatic_number_upper_bound(&self) -> usize {
        if self.n_vertices == 0 {
            return 0;
        }
        if self.edges.is_empty() {
            return 1;
        }

        let adj = self.adjacency_list();
        let mut colors: Vec<Option<usize>> = vec![None; self.n_vertices];
        let mut max_color = 0;

        // Greedy coloring in order of decreasing degree
        let mut order: Vec<usize> = (0..self.n_vertices).collect();
        let degrees = self.degrees();
        order.sort_by_key(|&v| std::cmp::Reverse(degrees[v]));

        for v in order {
            // Find colors used by neighbors
            let used: HashSet<usize> = adj[v].iter().filter_map(|&u| colors[u]).collect();

            // Assign smallest available color
            let color = (0..).find(|c| !used.contains(c)).unwrap();
            colors[v] = Some(color);
            max_color = max_color.max(color);
        }

        max_color + 1
    }

    /// Check if the graph is bipartite (2-colorable)
    pub fn is_bipartite(&self) -> bool {
        if self.n_vertices == 0 {
            return true;
        }

        let adj = self.adjacency_list();
        let mut color: Vec<Option<bool>> = vec![None; self.n_vertices];

        // BFS from each unvisited component
        for start in 0..self.n_vertices {
            if color[start].is_some() {
                continue;
            }

            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            color[start] = Some(true);

            while let Some(v) = queue.pop_front() {
                let my_color = color[v].unwrap();
                for &u in &adj[v] {
                    match color[u] {
                        None => {
                            color[u] = Some(!my_color);
                            queue.push_back(u);
                        }
                        Some(c) if c == my_color => return false,
                        _ => {}
                    }
                }
            }
        }

        true
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.n_vertices <= 1 {
            return true;
        }

        let adj = self.adjacency_list();
        let mut visited = vec![false; self.n_vertices];
        let mut stack = vec![0];
        let mut count = 0;

        while let Some(v) = stack.pop() {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            count += 1;
            for &u in &adj[v] {
                if !visited[u] {
                    stack.push(u);
                }
            }
        }

        count == self.n_vertices
    }
}

/// Verification result for graph states
#[derive(Debug, Clone)]
pub struct GraphStateVerification {
    /// Graph type description
    pub graph_type: String,
    /// Number of vertices/qubits
    pub n_vertices: usize,
    /// Number of edges
    pub n_edges: usize,
    /// Number of non-zero amplitudes found
    pub nonzero_amplitudes: usize,
    /// Number of stabilizer checks passed
    pub stabilizers_passed: usize,
    /// Total number of stabilizers (should equal n_vertices)
    pub total_stabilizers: usize,
    /// Total probability (should be 1.0)
    pub total_probability: f64,
    /// Fidelity with expected graph state
    pub fidelity: f64,
    /// Number of active blocks
    pub active_blocks: usize,
    /// Memory in bytes
    pub memory_bytes: usize,
    /// Creation time in microseconds
    pub time_us: u64,
}

impl std::fmt::Display for GraphStateVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph State Verification: {}", self.graph_type)?;
        writeln!(f, "  Vertices (qubits):    {}", self.n_vertices)?;
        writeln!(f, "  Edges:                {}", self.n_edges)?;
        writeln!(f, "  Non-zero amplitudes:  {}", self.nonzero_amplitudes)?;
        writeln!(
            f,
            "  Stabilizers passed:   {}/{}",
            self.stabilizers_passed, self.total_stabilizers
        )?;
        writeln!(f, "  Total probability:    {:.6}", self.total_probability)?;
        writeln!(f, "  Fidelity:             {:.2}%", self.fidelity * 100.0)?;
        writeln!(f, "  Active blocks:        {}", self.active_blocks)?;
        writeln!(
            f,
            "  Memory:               {} bytes ({:.2} KB)",
            self.memory_bytes,
            self.memory_bytes as f64 / 1024.0
        )?;
        writeln!(
            f,
            "  Time:                 {}μs ({:.2}ms)",
            self.time_us,
            self.time_us as f64 / 1000.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_graph() {
        let g = Graph::star(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.max_degree(), 4); // Center vertex
        assert!(g.is_bipartite());
        assert!(g.is_connected());
    }

    #[test]
    fn test_path_graph() {
        let g = Graph::path(10);
        assert_eq!(g.n_vertices, 10);
        assert_eq!(g.num_edges(), 9);
        assert_eq!(g.max_degree(), 2);
        assert!(g.is_bipartite());
        assert!(g.is_connected());
    }

    #[test]
    fn test_cycle_graph() {
        let g = Graph::cycle(6);
        assert_eq!(g.n_vertices, 6);
        assert_eq!(g.num_edges(), 6);
        assert_eq!(g.max_degree(), 2);
        assert!(g.is_bipartite()); // Even cycle
        assert!(g.is_connected());

        let g_odd = Graph::cycle(5);
        assert!(!g_odd.is_bipartite()); // Odd cycle
    }

    #[test]
    fn test_grid_2d() {
        let g = Graph::grid_2d(3, 4);
        assert_eq!(g.n_vertices, 12);
        assert_eq!(g.num_edges(), 17); // 2*3 + 3*2 + 1*3 + 1*2 = 17
        assert!(g.is_bipartite());
        assert!(g.is_connected());
    }

    #[test]
    fn test_complete_graph() {
        let g = Graph::complete(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.num_edges(), 10); // C(5,2)
        assert_eq!(g.max_degree(), 4);
        assert_eq!(g.chromatic_number_upper_bound(), 5);
    }

    #[test]
    fn test_hypercube() {
        let g = Graph::hypercube(3);
        assert_eq!(g.n_vertices, 8);
        assert_eq!(g.num_edges(), 12); // 3 * 2^2
        assert!(g.is_bipartite());
    }

    #[test]
    fn test_neighbors() {
        let g = Graph::star(5);
        assert_eq!(g.neighbors(0), vec![1, 2, 3, 4]);
        assert_eq!(g.neighbors(1), vec![0]);
    }
}
