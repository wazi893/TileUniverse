//! Graph-Based Quantum Router for QRAM
//!
//! SPRINT 50.0 Phase 2: Graph State Router
//!
//! This module implements QRAM routing using graph states, where routing
//! relationships are encoded as edges in a graph. Graph states are stabilizer
//! states, enabling efficient polynomial-time simulation.
//!
//! # Graph State QRAM Insight
//!
//! The bucket-brigade routing tree can be viewed as a graph:
//!
//! ```text
//! Address qubits:    a₀ ─── a₁ ─── a₂ ─── a₃
//!                     │      │      │      │
//! Router tree:       R₀ ─── R₁ ─── R₂ ─── ...
//!                     │      │      │
//! Memory cells:      D₀    D₁     D₂    ...
//! ```
//!
//! Graph state preparation:
//! 1. Initialize all qubits in |+⟩ (H on |0⟩)
//! 2. Apply CZ for each edge in the graph
//!
//! The resulting state is a stabilizer state that encodes the routing
//! structure. Measurements collapse paths and retrieve data.
//!
//! # Advantages
//!
//! - O(n²) simulation via stabilizer formalism
//! - All operations are Clifford (CZ gates)
//! - Exact representation of routing correlations
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::GraphRouter;
//!
//! let mut router = GraphRouter::new_bucket_brigade(4);
//! router.prepare_graph_state();
//! let path = router.route_and_measure(0b1010, 42);
//! ```

use crate::qec::stabilizer::StabilizerTableau;
use crate::quantum::QRng;
use std::fmt;

/// Graph-based quantum router using stabilizer formalism
///
/// Represents the QRAM routing structure as a graph where:
/// - Vertices are qubits (address, routers, memory)
/// - Edges represent CZ entanglement (routing relationships)
#[derive(Clone)]
pub struct GraphRouter {
    /// Graph structure
    graph: RoutingGraph,
    /// Stabilizer tableau for simulation
    tableau: StabilizerTableau,
    /// Configuration
    config: GraphRouterConfig,
    /// Statistics
    stats: GraphRouterStats,
}

/// Configuration for graph router
#[derive(Clone, Debug)]
pub struct GraphRouterConfig {
    /// Number of address bits
    pub depth: usize,
    /// Include bus qubit in graph
    pub include_bus: bool,
    /// Graph topology type
    pub topology: GraphTopology,
}

/// Graph topology types
#[derive(Clone, Debug, PartialEq)]
pub enum GraphTopology {
    /// Star graph: central hub connected to all others
    Star,
    /// Path graph: linear chain
    Path,
    /// Binary tree (standard bucket-brigade)
    BinaryTree,
    /// Complete graph (all-to-all)
    Complete,
    /// Custom edge list
    Custom,
}

/// Statistics for graph router operations
#[derive(Clone, Debug, Default)]
pub struct GraphRouterStats {
    /// Graphs prepared
    pub graphs_prepared: usize,
    /// Routes performed
    pub routes_performed: usize,
    /// CZ gates applied
    pub cz_gates: usize,
    /// Measurements performed
    pub measurements: usize,
}

/// Result from graph routing
#[derive(Clone, Debug)]
pub struct GraphRouteResult {
    /// Input address (before routing)
    pub input_address: usize,
    /// Measured address (after routing)
    pub measured_address: usize,
    /// Path taken through router tree
    pub path: Vec<RouterStep>,
    /// Was measurement deterministic?
    pub deterministic: bool,
}

/// A single step in the routing path
#[derive(Clone, Debug)]
pub struct RouterStep {
    /// Router index
    pub router: usize,
    /// Level in tree
    pub level: usize,
    /// Direction taken (false=left, true=right)
    pub direction: bool,
}

/// Graph structure for routing
#[derive(Clone, Debug)]
pub struct RoutingGraph {
    /// Total number of vertices (qubits)
    pub n_vertices: usize,
    /// Address vertex indices
    pub address_vertices: Vec<usize>,
    /// Router vertex indices
    pub router_vertices: Vec<usize>,
    /// Bus vertex index (if any)
    pub bus_vertex: Option<usize>,
    /// Edges as (vertex_a, vertex_b) pairs
    pub edges: Vec<(usize, usize)>,
}

impl RoutingGraph {
    /// Create empty graph
    pub fn new(n_vertices: usize) -> Self {
        Self {
            n_vertices,
            address_vertices: Vec::new(),
            router_vertices: Vec::new(),
            bus_vertex: None,
            edges: Vec::new(),
        }
    }

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, a: usize, b: usize) {
        if a < self.n_vertices && b < self.n_vertices && a != b {
            // Avoid duplicates
            let edge = if a < b { (a, b) } else { (b, a) };
            if !self.edges.contains(&edge) {
                self.edges.push(edge);
            }
        }
    }

    /// Get edges incident to a vertex
    pub fn edges_of(&self, vertex: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(a, b)| {
                if a == vertex {
                    Some(b)
                } else if b == vertex {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get degree (number of edges) of a vertex
    pub fn degree(&self, vertex: usize) -> usize {
        self.edges_of(vertex).len()
    }

    /// Total edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl GraphRouter {
    /// Create a new graph router for bucket-brigade topology
    pub fn new_bucket_brigade(depth: usize) -> Self {
        let config = GraphRouterConfig {
            depth,
            include_bus: true,
            topology: GraphTopology::BinaryTree,
        };

        let graph = Self::build_bucket_brigade_graph(depth, true);
        let n_qubits = graph.n_vertices;

        Self {
            graph,
            tableau: StabilizerTableau::new(n_qubits),
            config,
            stats: GraphRouterStats::default(),
        }
    }

    /// Create a star graph router (efficient for certain patterns)
    pub fn new_star(depth: usize) -> Self {
        let config = GraphRouterConfig {
            depth,
            include_bus: true,
            topology: GraphTopology::Star,
        };

        let graph = Self::build_star_graph(depth);
        let n_qubits = graph.n_vertices;

        Self {
            graph,
            tableau: StabilizerTableau::new(n_qubits),
            config,
            stats: GraphRouterStats::default(),
        }
    }

    /// Create a path graph router
    pub fn new_path(depth: usize) -> Self {
        let config = GraphRouterConfig {
            depth,
            include_bus: false,
            topology: GraphTopology::Path,
        };

        let graph = Self::build_path_graph(depth);
        let n_qubits = graph.n_vertices;

        Self {
            graph,
            tableau: StabilizerTableau::new(n_qubits),
            config,
            stats: GraphRouterStats::default(),
        }
    }

    /// Build bucket-brigade graph structure
    fn build_bucket_brigade_graph(depth: usize, include_bus: bool) -> RoutingGraph {
        let memory_cells = 1 << depth;
        let routers = memory_cells - 1;

        // Qubit layout:
        // 0..depth: address qubits
        // depth: bus qubit (if included)
        // depth+1..: router qubits (2 per router)
        let bus_offset = if include_bus { 1 } else { 0 };
        let n_qubits = depth + bus_offset + 2 * routers;

        let mut graph = RoutingGraph::new(n_qubits);

        // Address vertices
        graph.address_vertices = (0..depth).collect();

        // Bus vertex
        if include_bus {
            graph.bus_vertex = Some(depth);
        }

        // Router vertices
        graph.router_vertices = (0..2 * routers).map(|i| depth + bus_offset + i).collect();

        // Add edges for bucket-brigade structure
        // Each address qubit connects to routers at its level
        for level in 0..depth {
            let routers_at_level = 1 << level;
            let addr_qubit = level;

            for pos in 0..routers_at_level {
                let router_idx = (1 << level) - 1 + pos;
                let left = depth + bus_offset + 2 * router_idx;
                let right = left + 1;

                // Address controls routing direction
                graph.add_edge(addr_qubit, left);
                graph.add_edge(addr_qubit, right);

                // Routers at same level connected
                if pos > 0 {
                    let prev_left = depth + bus_offset + 2 * (router_idx - 1);
                    graph.add_edge(prev_left + 1, left);
                }
            }
        }

        // Bus connects to all router paths
        if include_bus {
            let bus = depth;
            let router_qubits: Vec<usize> = graph.router_vertices.clone();
            for router_qubit in router_qubits {
                graph.add_edge(bus, router_qubit);
            }
        }

        graph
    }

    /// Build star graph: central hub connected to all address qubits
    fn build_star_graph(depth: usize) -> RoutingGraph {
        // Star: 1 hub + depth spokes = depth + 1 qubits
        let n_qubits = depth + 1;
        let mut graph = RoutingGraph::new(n_qubits);

        // Hub is vertex 0, spokes are 1..depth
        graph.address_vertices = (1..=depth).collect();
        graph.bus_vertex = Some(0);

        // Connect hub to all spokes
        for i in 1..=depth {
            graph.add_edge(0, i);
        }

        graph
    }

    /// Build path graph: linear chain of address qubits
    fn build_path_graph(depth: usize) -> RoutingGraph {
        let mut graph = RoutingGraph::new(depth);
        graph.address_vertices = (0..depth).collect();

        // Linear chain
        for i in 0..depth - 1 {
            graph.add_edge(i, i + 1);
        }

        graph
    }

    /// Get the underlying graph structure
    pub fn graph(&self) -> &RoutingGraph {
        &self.graph
    }

    /// Get current statistics
    pub fn stats(&self) -> &GraphRouterStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &GraphRouterConfig {
        &self.config
    }

    /// Prepare the graph state via CZ gates
    ///
    /// 1. Initialize all qubits in |+⟩ (H on |0⟩)
    /// 2. Apply CZ for each edge
    pub fn prepare_graph_state(&mut self) {
        // Reset to |0⟩^⊗n
        self.tableau = StabilizerTableau::new(self.graph.n_vertices);

        // Apply H to all qubits → |+⟩^⊗n
        for q in 0..self.graph.n_vertices {
            self.tableau.hadamard(q);
        }

        // Apply CZ for each edge
        for &(a, b) in &self.graph.edges {
            self.tableau.cz(a, b);
            self.stats.cz_gates += 1;
        }

        self.stats.graphs_prepared += 1;
    }

    /// Initialize address register to a specific basis state
    pub fn set_address(&mut self, address: usize) {
        for (i, &qubit) in self.graph.address_vertices.iter().enumerate() {
            if (address >> i) & 1 == 1 {
                // X gate to flip from |+⟩ to |-⟩
                self.tableau.pauli_x(qubit);
            }
        }
    }

    /// Route and measure, returning the routing path
    pub fn route_and_measure(&mut self, address: usize, seed: u64) -> GraphRouteResult {
        // Prepare graph state
        self.prepare_graph_state();

        // Set address
        self.set_address(address);

        // Measure address qubits
        let mut rng = QRng::new(seed);
        let mut measured_address = 0usize;
        let mut path = Vec::new();
        let mut all_deterministic = true;

        for (i, &qubit) in self.graph.address_vertices.iter().enumerate() {
            let random_bit = rng.next_f32() > 0.5;
            let (outcome, is_random) = self.tableau.measure_z(qubit, random_bit);
            self.stats.measurements += 1;

            if is_random {
                all_deterministic = false;
            }

            if outcome == 1 {
                measured_address |= 1 << i;
            }

            // Record routing step
            if i < self.config.depth {
                path.push(RouterStep {
                    router: (1 << i) - 1 + (measured_address >> i),
                    level: i,
                    direction: outcome == 1,
                });
            }
        }

        self.stats.routes_performed += 1;

        GraphRouteResult {
            input_address: address,
            measured_address,
            path,
            deterministic: all_deterministic,
        }
    }

    /// Route from superposition (all addresses equally likely)
    pub fn route_superposition(&mut self, seed: u64) -> GraphRouteResult {
        // Prepare graph state (already in superposition)
        self.prepare_graph_state();

        // Don't set any address - stay in uniform superposition
        // Measure address qubits
        let mut rng = QRng::new(seed);
        let mut measured_address = 0usize;
        let mut path = Vec::new();

        for (i, &qubit) in self.graph.address_vertices.iter().enumerate() {
            let random_bit = rng.next_f32() > 0.5;
            let (outcome, _) = self.tableau.measure_z(qubit, random_bit);
            self.stats.measurements += 1;

            if outcome == 1 {
                measured_address |= 1 << i;
            }

            if i < self.config.depth {
                path.push(RouterStep {
                    router: (1 << i) - 1 + (measured_address >> i),
                    level: i,
                    direction: outcome == 1,
                });
            }
        }

        self.stats.routes_performed += 1;

        GraphRouteResult {
            input_address: 0, // Was in superposition
            measured_address,
            path,
            deterministic: false,
        }
    }

    /// Get resource comparison
    pub fn resource_comparison(&self) -> GraphRouterResourceComparison {
        let n_qubits = self.graph.n_vertices;

        // Dense state would need 2^n * 16 bytes
        let dense_bytes = if n_qubits <= 40 {
            (1u64 << n_qubits) * 16
        } else {
            u64::MAX
        };

        // Stabilizer: 2n generators * (n/64) words * 8 bytes
        let n_words = (n_qubits + 63) / 64;
        let stabilizer_bytes = 2 * n_qubits * n_words * 8;

        GraphRouterResourceComparison {
            depth: self.config.depth,
            n_vertices: self.graph.n_vertices,
            n_edges: self.graph.edge_count(),
            topology: self.config.topology.clone(),
            dense_state_bytes: dense_bytes,
            stabilizer_bytes,
            compression_ratio: if stabilizer_bytes > 0 {
                dense_bytes as f64 / stabilizer_bytes as f64
            } else {
                f64::INFINITY
            },
            is_feasible_dense: n_qubits <= 32,
        }
    }
}

impl fmt::Display for GraphRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraphRouter({:?}, depth={}, vertices={}, edges={})",
            self.config.topology,
            self.config.depth,
            self.graph.n_vertices,
            self.graph.edge_count()
        )
    }
}

/// Resource comparison for graph router
#[derive(Clone, Debug)]
pub struct GraphRouterResourceComparison {
    /// Routing depth
    pub depth: usize,
    /// Number of graph vertices (qubits)
    pub n_vertices: usize,
    /// Number of graph edges (CZ gates)
    pub n_edges: usize,
    /// Graph topology
    pub topology: GraphTopology,
    /// Dense state bytes (may be u64::MAX)
    pub dense_state_bytes: u64,
    /// Stabilizer tableau bytes
    pub stabilizer_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Whether dense is feasible
    pub is_feasible_dense: bool,
}

impl fmt::Display for GraphRouterResourceComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dense_str = if self.dense_state_bytes == u64::MAX {
            "IMPOSSIBLE".to_string()
        } else if self.dense_state_bytes >= 1_000_000_000 {
            format!("{:.1} GB", self.dense_state_bytes as f64 / 1e9)
        } else {
            format!("{} KB", self.dense_state_bytes / 1024)
        };

        write!(
            f,
            "GraphRouterComparison({:?}, vertices={}, edges={}, dense={}, stabilizer={}KB)",
            self.topology,
            self.n_vertices,
            self.n_edges,
            dense_str,
            self.stabilizer_bytes / 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_brigade_creation() {
        let router = GraphRouter::new_bucket_brigade(4);
        assert_eq!(router.config.depth, 4);
        assert_eq!(router.graph.address_vertices.len(), 4);
        assert!(router.graph.bus_vertex.is_some());
    }

    #[test]
    fn test_star_graph() {
        let router = GraphRouter::new_star(4);
        assert_eq!(router.graph.n_vertices, 5); // 1 hub + 4 spokes
        assert_eq!(router.graph.edge_count(), 4); // Hub to each spoke
    }

    #[test]
    fn test_path_graph() {
        let router = GraphRouter::new_path(5);
        assert_eq!(router.graph.n_vertices, 5);
        assert_eq!(router.graph.edge_count(), 4); // Linear chain
    }

    #[test]
    fn test_prepare_graph_state() {
        let mut router = GraphRouter::new_bucket_brigade(2);
        router.prepare_graph_state();
        assert_eq!(router.stats.graphs_prepared, 1);
        assert!(router.stats.cz_gates > 0);
    }

    #[test]
    fn test_route_and_measure() {
        let mut router = GraphRouter::new_bucket_brigade(3);
        let result = router.route_and_measure(5, 42);
        assert!(result.measured_address < 8);
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_route_superposition() {
        let mut router = GraphRouter::new_bucket_brigade(3);

        // Run multiple routes to verify randomness
        let mut addresses = std::collections::HashSet::new();
        for seed in 0..20 {
            let result = router.route_superposition(seed);
            addresses.insert(result.measured_address);
            assert!(result.measured_address < 8);
            assert!(!result.deterministic);
        }

        // Should see multiple different addresses
        assert!(addresses.len() > 1, "Should measure different addresses");
    }

    #[test]
    fn test_graph_degree() {
        let router = GraphRouter::new_star(4);
        // Hub (vertex 0) should have degree 4
        assert_eq!(router.graph.degree(0), 4);
        // Spokes should have degree 1
        for i in 1..=4 {
            assert_eq!(router.graph.degree(i), 1);
        }
    }

    #[test]
    fn test_resource_comparison() {
        let router = GraphRouter::new_bucket_brigade(4);
        let comparison = router.resource_comparison();

        assert_eq!(comparison.depth, 4);
        assert!(comparison.stabilizer_bytes > 0);
        println!("{}", comparison);
    }

    #[test]
    fn test_large_scale() {
        // Create large graph router
        let router = GraphRouter::new_bucket_brigade(10);
        let comparison = router.resource_comparison();

        assert!(!comparison.is_feasible_dense);
        println!("{}", router);
        println!("{}", comparison);
    }

    #[test]
    fn test_display() {
        let router = GraphRouter::new_bucket_brigade(3);
        let s = format!("{}", router);
        assert!(s.contains("GraphRouter"));
        assert!(s.contains("BinaryTree"));
    }
}
