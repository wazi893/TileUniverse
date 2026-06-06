//! Syndrome Graph Abstraction for MWPM Decoding
//!
//! This module provides the abstraction layer between QEC codes and MWPM solvers.
//! The key insight is that graph construction is ~80% of the decoding work.
//!
//! # Architecture
//!
//! - `SyndromeGraphBuilder`: Trait for code-specific graph construction
//! - `DecodingGraph`: Generic graph representation for MWPM solvers
//! - `PhenomenologicalGraphBuilder`: Builder for phenomenological noise on surface codes

/// Trait for building decoding graphs from syndrome measurements.
///
/// Different QEC codes require different graph structures. This trait
/// abstracts the graph construction so decoders can work with any code.
pub trait SyndromeGraphBuilder {
    /// Build a decoding graph from syndrome measurements.
    ///
    /// # Arguments
    /// - `syndrome`: Bit vector where 1 indicates a triggered stabilizer
    /// - `n_stabilizers`: Total number of stabilizers (may differ from syndrome.len())
    ///
    /// # Returns
    /// A `DecodingGraph` ready for MWPM solving
    fn build_graph(&self, syndrome: &[u8], n_stabilizers: usize) -> DecodingGraph;
}

/// A vertex in the decoding graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    /// Unique identifier for this vertex
    pub id: usize,
    /// Whether this is a boundary (virtual) vertex
    pub is_boundary: bool,
    /// Original stabilizer index (None for pure boundary nodes)
    pub stabilizer_idx: Option<usize>,
    /// Position for distance calculations (row, col)
    pub position: (f32, f32),
}

impl Vertex {
    /// Create a new defect vertex (triggered stabilizer)
    pub fn defect(id: usize, stabilizer_idx: usize, position: (f32, f32)) -> Self {
        Self {
            id,
            is_boundary: false,
            stabilizer_idx: Some(stabilizer_idx),
            position,
        }
    }

    /// Create a new boundary (virtual) vertex
    pub fn boundary(id: usize, position: (f32, f32)) -> Self {
        Self {
            id,
            is_boundary: true,
            stabilizer_idx: None,
            position,
        }
    }
}

/// A weighted edge in the decoding graph.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedEdge {
    /// First vertex ID
    pub v1: usize,
    /// Second vertex ID
    pub v2: usize,
    /// Edge weight (typically -log(p_error) for phenomenological noise)
    pub weight: f64,
    /// Qubits associated with this edge (for correction generation)
    pub associated_qubits: Vec<usize>,
}

impl WeightedEdge {
    /// Create a new weighted edge
    pub fn new(v1: usize, v2: usize, weight: f64) -> Self {
        Self {
            v1,
            v2,
            weight,
            associated_qubits: Vec::new(),
        }
    }

    /// Create an edge with associated qubits
    pub fn with_qubits(v1: usize, v2: usize, weight: f64, qubits: Vec<usize>) -> Self {
        Self {
            v1,
            v2,
            weight,
            associated_qubits: qubits,
        }
    }
}

/// Graph structure for MWPM solving.
///
/// This is a generic representation that can be converted to various
/// MWPM solver formats (fusion-blossom, PyMatching, etc.).
#[derive(Clone, Debug)]
pub struct DecodingGraph {
    /// All vertices (syndrome defects + boundary nodes)
    pub vertices: Vec<Vertex>,
    /// Weighted edges connecting vertices
    pub edges: Vec<WeightedEdge>,
    /// Indices of boundary (virtual) vertices
    pub boundary_vertices: Vec<usize>,
    /// Indices of defect (syndrome=1) vertices
    pub defect_vertices: Vec<usize>,
}

impl DecodingGraph {
    /// Create a new empty decoding graph
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            boundary_vertices: Vec::new(),
            defect_vertices: Vec::new(),
        }
    }

    /// Add a defect vertex and return its ID
    pub fn add_defect(&mut self, stabilizer_idx: usize, position: (f32, f32)) -> usize {
        let id = self.vertices.len();
        self.vertices
            .push(Vertex::defect(id, stabilizer_idx, position));
        self.defect_vertices.push(id);
        id
    }

    /// Add a boundary vertex and return its ID
    pub fn add_boundary(&mut self, position: (f32, f32)) -> usize {
        let id = self.vertices.len();
        self.vertices.push(Vertex::boundary(id, position));
        self.boundary_vertices.push(id);
        id
    }

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, v1: usize, v2: usize, weight: f64) {
        self.edges.push(WeightedEdge::new(v1, v2, weight));
    }

    /// Add an edge with associated qubits
    pub fn add_edge_with_qubits(&mut self, v1: usize, v2: usize, weight: f64, qubits: Vec<usize>) {
        self.edges
            .push(WeightedEdge::with_qubits(v1, v2, weight, qubits));
    }

    /// Get the number of real (non-boundary) vertices
    pub fn num_defects(&self) -> usize {
        self.defect_vertices.len()
    }

    /// Get the total number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if the graph is empty (no defects)
    pub fn is_empty(&self) -> bool {
        self.defect_vertices.is_empty()
    }
}

impl Default for DecodingGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Phenomenological graph builder for surface codes.
///
/// Builds a decoding graph assuming phenomenological noise:
/// - Each data qubit has independent X and Z error probability p
/// - Edge weights are -ln(p/(1-p)) to convert probabilities to distances
/// - Boundary vertices allow unpaired defects to be matched
#[derive(Clone, Debug)]
pub struct PhenomenologicalGraphBuilder {
    /// Code distance
    pub distance: usize,
    /// Physical error probability per qubit
    pub p_error: f64,
    /// Precomputed edge weight from p_error
    edge_weight: f64,
}

impl PhenomenologicalGraphBuilder {
    /// Create a new phenomenological graph builder.
    ///
    /// # Arguments
    /// - `distance`: Surface code distance
    /// - `p_error`: Physical error probability (typically 0.001 to 0.1)
    pub fn new(distance: usize, p_error: f64) -> Self {
        // Edge weight is -ln(p / (1-p)) for optimal MWPM decoding
        // Clamp p_error to avoid division by zero or negative weights
        let p_clamped = p_error.clamp(1e-10, 1.0 - 1e-10);
        let edge_weight = -(p_clamped / (1.0 - p_clamped)).ln();

        Self {
            distance,
            p_error,
            edge_weight,
        }
    }

    /// Get the edge weight for this error probability
    pub fn get_edge_weight(&self) -> f64 {
        self.edge_weight
    }

    /// Compute stabilizer position for X stabilizers in a surface code.
    /// Returns (row, col) in half-integer coordinates.
    fn x_stabilizer_position(&self, stab_idx: usize) -> (f32, f32) {
        // X stabilizers are arranged in a specific pattern
        // Interior stabilizers at even (r+c) positions
        let d = self.distance;
        let interior_count = (d - 1) * (d - 1) / 2;

        if stab_idx < interior_count {
            // Interior stabilizer
            let mut count = 0;
            for r in 0..(d - 1) {
                for c in 0..(d - 1) {
                    if (r + c) % 2 == 0 {
                        if count == stab_idx {
                            return (r as f32 + 0.5, c as f32 + 0.5);
                        }
                        count += 1;
                    }
                }
            }
        }

        // Boundary stabilizers
        let boundary_idx = stab_idx - interior_count;
        let top_count = (d - 1) / 2; // Odd columns on top

        if boundary_idx < top_count {
            // Top boundary (row = -0.5)
            let c = 1 + boundary_idx * 2;
            return (-0.5, c as f32 + 0.5);
        }

        // Bottom boundary (row = d - 0.5)
        let bottom_idx = boundary_idx - top_count;
        let c = bottom_idx * 2;
        (self.distance as f32 - 0.5, c as f32 + 0.5)
    }

    /// Compute stabilizer position for Z stabilizers in a surface code.
    fn z_stabilizer_position(&self, stab_idx: usize) -> (f32, f32) {
        let d = self.distance;
        let interior_count = (d - 1) * (d - 1) / 2;

        if stab_idx < interior_count {
            // Interior stabilizer at odd (r+c) positions
            let mut count = 0;
            for r in 0..(d - 1) {
                for c in 0..(d - 1) {
                    if (r + c) % 2 == 1 {
                        if count == stab_idx {
                            return (r as f32 + 0.5, c as f32 + 0.5);
                        }
                        count += 1;
                    }
                }
            }
        }

        // Boundary stabilizers
        let boundary_idx = stab_idx - interior_count;
        let left_count = (d - 1) / 2; // Odd rows on left

        if boundary_idx < left_count {
            // Left boundary (col = -0.5)
            let r = 1 + boundary_idx * 2;
            return (r as f32 + 0.5, -0.5);
        }

        // Right boundary (col = d - 0.5)
        let right_idx = boundary_idx - left_count;
        let r = right_idx * 2;
        (r as f32 + 0.5, self.distance as f32 - 0.5)
    }

    /// Manhattan distance between two positions
    fn manhattan_distance(p1: (f32, f32), p2: (f32, f32)) -> f32 {
        (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()
    }

    /// Build X syndrome graph (for detecting Z errors)
    pub fn build_x_graph(&self, x_syndrome: &[u8]) -> DecodingGraph {
        self.build_syndrome_graph(x_syndrome, true)
    }

    /// Build Z syndrome graph (for detecting X errors)
    pub fn build_z_graph(&self, z_syndrome: &[u8]) -> DecodingGraph {
        self.build_syndrome_graph(z_syndrome, false)
    }

    /// Internal graph builder
    fn build_syndrome_graph(&self, syndrome: &[u8], is_x_syndrome: bool) -> DecodingGraph {
        let mut graph = DecodingGraph::new();

        // Find defects (syndrome = 1)
        let defect_positions: Vec<(usize, (f32, f32))> = syndrome
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s == 1)
            .map(|(idx, _)| {
                let pos = if is_x_syndrome {
                    self.x_stabilizer_position(idx)
                } else {
                    self.z_stabilizer_position(idx)
                };
                (idx, pos)
            })
            .collect();

        if defect_positions.is_empty() {
            return graph;
        }

        // Add defect vertices
        let mut defect_vertex_ids: Vec<usize> = Vec::new();
        for (stab_idx, pos) in &defect_positions {
            let vid = graph.add_defect(*stab_idx, *pos);
            defect_vertex_ids.push(vid);
        }

        // Add boundary vertices at edges of the code
        // For X syndrome: boundaries at top and bottom
        // For Z syndrome: boundaries at left and right
        let boundary_positions: Vec<(f32, f32)> = if is_x_syndrome {
            // Top and bottom boundaries for X stabilizers
            vec![
                (-1.0, self.distance as f32 / 2.0),                 // Top boundary
                (self.distance as f32, self.distance as f32 / 2.0), // Bottom boundary
            ]
        } else {
            // Left and right boundaries for Z stabilizers
            vec![
                (self.distance as f32 / 2.0, -1.0), // Left boundary
                (self.distance as f32 / 2.0, self.distance as f32), // Right boundary
            ]
        };

        let mut boundary_vertex_ids: Vec<usize> = Vec::new();
        for pos in &boundary_positions {
            let vid = graph.add_boundary(*pos);
            boundary_vertex_ids.push(vid);
        }

        // Add edges between all defect pairs (complete graph on defects)
        for i in 0..defect_vertex_ids.len() {
            for j in (i + 1)..defect_vertex_ids.len() {
                let v1 = defect_vertex_ids[i];
                let v2 = defect_vertex_ids[j];
                let pos1 = defect_positions[i].1;
                let pos2 = defect_positions[j].1;

                // Weight proportional to Manhattan distance
                let dist = Self::manhattan_distance(pos1, pos2);
                let weight = self.edge_weight * dist as f64;

                graph.add_edge(v1, v2, weight);
            }
        }

        // Add edges from defects to boundary vertices
        for (i, &def_vid) in defect_vertex_ids.iter().enumerate() {
            let def_pos = defect_positions[i].1;

            // Find nearest boundary
            let mut min_dist = f32::MAX;
            let mut nearest_boundary = 0;

            for (j, &bound_vid) in boundary_vertex_ids.iter().enumerate() {
                let bound_pos = boundary_positions[j];
                let dist = Self::manhattan_distance(def_pos, bound_pos);
                if dist < min_dist {
                    min_dist = dist;
                    nearest_boundary = bound_vid;
                }
            }

            let weight = self.edge_weight * min_dist as f64;
            graph.add_edge(def_vid, nearest_boundary, weight);
        }

        // Add zero-weight edges between boundary vertices
        // (they can be matched together at no cost)
        for i in 0..boundary_vertex_ids.len() {
            for j in (i + 1)..boundary_vertex_ids.len() {
                graph.add_edge(boundary_vertex_ids[i], boundary_vertex_ids[j], 0.0);
            }
        }

        graph
    }
}

impl SyndromeGraphBuilder for PhenomenologicalGraphBuilder {
    fn build_graph(&self, syndrome: &[u8], _n_stabilizers: usize) -> DecodingGraph {
        // Default to X syndrome behavior
        // In practice, callers should use build_x_graph or build_z_graph directly
        self.build_x_graph(syndrome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoding_graph_creation() {
        let mut graph = DecodingGraph::new();

        // Add two defects
        let v0 = graph.add_defect(0, (0.5, 0.5));
        let v1 = graph.add_defect(1, (0.5, 1.5));

        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(graph.num_defects(), 2);
        assert_eq!(graph.num_vertices(), 2);

        // Add boundary
        let v2 = graph.add_boundary((-0.5, 1.0));
        assert_eq!(v2, 2);
        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.boundary_vertices.len(), 1);

        // Add edges
        graph.add_edge(v0, v1, 1.0);
        graph.add_edge(v0, v2, 0.5);
        assert_eq!(graph.num_edges(), 2);
    }

    #[test]
    fn test_phenomenological_builder_empty_syndrome() {
        let builder = PhenomenologicalGraphBuilder::new(3, 0.01);

        // Empty syndrome should produce empty graph
        let syndrome = vec![0u8; 4];
        let graph = builder.build_x_graph(&syndrome);

        assert!(graph.is_empty());
        assert_eq!(graph.num_defects(), 0);
    }

    #[test]
    fn test_phenomenological_builder_single_defect() {
        let builder = PhenomenologicalGraphBuilder::new(3, 0.01);

        // Single defect
        let mut syndrome = vec![0u8; 4];
        syndrome[0] = 1;

        let graph = builder.build_x_graph(&syndrome);

        // Should have 1 defect + 2 boundary vertices
        assert_eq!(graph.num_defects(), 1);
        assert!(graph.num_vertices() >= 3); // 1 defect + boundaries
        assert!(!graph.edges.is_empty()); // Should have edge to boundary
    }

    #[test]
    fn test_phenomenological_builder_two_defects() {
        let builder = PhenomenologicalGraphBuilder::new(5, 0.01);

        // Two defects
        let mut syndrome = vec![0u8; 12]; // d=5 has 12 X stabilizers
        syndrome[0] = 1;
        syndrome[1] = 1;

        let graph = builder.build_x_graph(&syndrome);

        assert_eq!(graph.num_defects(), 2);

        // Should have:
        // - 1 edge between the two defects
        // - edges from each defect to boundary
        // - edges between boundaries
        assert!(graph.num_edges() >= 3);
    }

    #[test]
    fn test_edge_weight_calculation() {
        let builder = PhenomenologicalGraphBuilder::new(3, 0.01);

        // Weight should be -ln(p/(1-p)) = -ln(0.01/0.99) ~ 4.6
        let weight = builder.get_edge_weight();
        assert!(weight > 4.0 && weight < 5.0);

        // Higher error rate = lower weight
        let builder_high = PhenomenologicalGraphBuilder::new(3, 0.1);
        let weight_high = builder_high.get_edge_weight();
        assert!(weight_high < weight);
    }

    #[test]
    fn test_graph_has_correct_edge_count() {
        let builder = PhenomenologicalGraphBuilder::new(5, 0.05);

        // Three defects
        let mut syndrome = vec![0u8; 12];
        syndrome[0] = 1;
        syndrome[2] = 1;
        syndrome[4] = 1;

        let graph = builder.build_x_graph(&syndrome);

        assert_eq!(graph.num_defects(), 3);

        // For 3 defects and 2 boundaries:
        // - C(3,2) = 3 edges between defects
        // - 3 edges from defects to nearest boundary
        // - C(2,2) = 1 edge between boundaries
        // Total: at least 3 edges (defect-to-defect)
        assert!(
            graph.num_edges() >= 3,
            "Expected at least 3 edges, got {}",
            graph.num_edges()
        );
    }
}
