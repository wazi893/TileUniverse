//! Device Topology Definitions
//!
//! SPRINT 52.0 Phase 1: Device Topologies
//!
//! Defines physical qubit connectivity for various quantum hardware types:
//! - 2D grids (superconducting processors)
//! - Linear chains (ion traps)
//! - Heavy-hex lattices (IBM specific)
//! - Custom topologies
//!
//! # Example
//!
//! ```ignore
//! use engine::hardware::{DeviceTopology, EnhancedCouplingMap};
//!
//! let topology = DeviceTopology::grid_2d(5, 5);
//! let coupling = topology.to_coupling_map();
//! assert_eq!(coupling.qubit_count(), 25);
//! assert!(coupling.are_adjacent(0, 1));  // Horizontal neighbor
//! assert!(coupling.are_adjacent(0, 5));  // Vertical neighbor
//! ```

use std::collections::{HashSet, VecDeque};
use std::fmt;

/// Physical device topology defining qubit connectivity
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceTopology {
    /// 2D grid with nearest-neighbor coupling (superconducting)
    Grid2D { width: usize, height: usize },

    /// Linear chain with all-to-all connectivity (ion trap)
    LinearChain { length: usize },

    /// Heavy-hex lattice (IBM Falcon/Heron style)
    /// Hexagonal cells with some vertices removed
    HeavyHex { rows: usize, cols: usize },

    /// Star topology with central hub
    Star { spokes: usize },

    /// Ring topology
    Ring { size: usize },

    /// Fully connected (theoretical/small systems)
    FullyConnected { size: usize },

    /// Custom topology from edge list
    Custom {
        qubit_count: usize,
        edges: Vec<(usize, usize)>,
    },
}

impl DeviceTopology {
    // === Constructors ===

    /// Create 2D grid topology
    pub fn grid_2d(width: usize, height: usize) -> Self {
        Self::Grid2D { width, height }
    }

    /// Create linear chain topology
    pub fn linear_chain(length: usize) -> Self {
        Self::LinearChain { length }
    }

    /// Create heavy-hex topology (IBM style)
    pub fn heavy_hex(rows: usize, cols: usize) -> Self {
        Self::HeavyHex { rows, cols }
    }

    /// Create star topology
    pub fn star(spokes: usize) -> Self {
        Self::Star { spokes }
    }

    /// Create ring topology
    pub fn ring(size: usize) -> Self {
        Self::Ring { size }
    }

    /// Create fully connected topology
    pub fn fully_connected(size: usize) -> Self {
        Self::FullyConnected { size }
    }

    /// Create custom topology from edges
    pub fn custom(qubit_count: usize, edges: Vec<(usize, usize)>) -> Self {
        Self::Custom { qubit_count, edges }
    }

    // === Properties ===

    /// Get total qubit count
    pub fn qubit_count(&self) -> usize {
        match self {
            Self::Grid2D { width, height } => width * height,
            Self::LinearChain { length } => *length,
            Self::HeavyHex { rows, cols } => Self::heavy_hex_qubit_count(*rows, *cols),
            Self::Star { spokes } => spokes + 1, // Hub + spokes
            Self::Ring { size } => *size,
            Self::FullyConnected { size } => *size,
            Self::Custom { qubit_count, .. } => *qubit_count,
        }
    }

    /// Get all edges (pairs of connected qubits)
    pub fn edges(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Grid2D { width, height } => Self::grid_2d_edges(*width, *height),
            Self::LinearChain { length } => Self::linear_chain_edges(*length),
            Self::HeavyHex { rows, cols } => Self::heavy_hex_edges(*rows, *cols),
            Self::Star { spokes } => Self::star_edges(*spokes),
            Self::Ring { size } => Self::ring_edges(*size),
            Self::FullyConnected { size } => Self::fully_connected_edges(*size),
            Self::Custom { edges, .. } => edges.clone(),
        }
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        match self {
            Self::Grid2D { width, height } => {
                let w = *width;
                let h = *height;
                (w - 1) * h + w * (h - 1) // Horizontal + vertical edges
            }
            Self::LinearChain { length } => length.saturating_sub(1),
            Self::HeavyHex { rows, cols } => Self::heavy_hex_edges(*rows, *cols).len(),
            Self::Star { spokes } => *spokes,
            Self::Ring { size } => *size,
            Self::FullyConnected { size } => size * (size - 1) / 2,
            Self::Custom { edges, .. } => edges.len(),
        }
    }

    /// Get maximum degree (max connections per qubit)
    pub fn max_degree(&self) -> usize {
        match self {
            Self::Grid2D { .. } => 4,
            Self::LinearChain { length } => {
                if *length <= 2 {
                    1
                } else {
                    2
                }
            }
            Self::HeavyHex { .. } => 3,
            Self::Star { spokes } => *spokes, // Hub has all connections
            Self::Ring { .. } => 2,
            Self::FullyConnected { size } => size.saturating_sub(1),
            Self::Custom { .. } => {
                let coupling = self.to_coupling_map();
                (0..self.qubit_count())
                    .map(|q| coupling.neighbors(q).len())
                    .max()
                    .unwrap_or(0)
            }
        }
    }

    /// Build enhanced coupling map with precomputed distances
    pub fn to_coupling_map(&self) -> EnhancedCouplingMap {
        EnhancedCouplingMap::from_topology(self)
    }

    // === Edge Generation ===

    fn grid_2d_edges(width: usize, height: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                // Right neighbor
                if col + 1 < width {
                    edges.push((idx, idx + 1));
                }
                // Down neighbor
                if row + 1 < height {
                    edges.push((idx, idx + width));
                }
            }
        }
        edges
    }

    fn linear_chain_edges(length: usize) -> Vec<(usize, usize)> {
        (0..length.saturating_sub(1)).map(|i| (i, i + 1)).collect()
    }

    fn heavy_hex_edges(rows: usize, cols: usize) -> Vec<(usize, usize)> {
        // Heavy-hex: hexagonal lattice with alternating connectivity
        // Simplified model: similar to grid but with some edges removed
        let mut edges = Vec::new();
        let _width = cols * 2; // Approximate width

        for row in 0..rows {
            for col in 0..cols {
                let base = row * cols + col;

                // Horizontal connections (all rows)
                if col + 1 < cols {
                    edges.push((base, base + 1));
                }

                // Vertical connections (alternating pattern)
                if row + 1 < rows {
                    // Only connect every other column vertically
                    if col % 2 == row % 2 {
                        edges.push((base, base + cols));
                    }
                }
            }
        }

        // Add cross connections for hex pattern
        for row in 0..rows.saturating_sub(1) {
            for col in 0..cols.saturating_sub(1) {
                let base = row * cols + col;
                // Diagonal connections in alternating pattern
                if (row + col) % 2 == 0 {
                    edges.push((base, base + cols + 1));
                }
            }
        }

        edges
    }

    fn heavy_hex_qubit_count(rows: usize, cols: usize) -> usize {
        rows * cols
    }

    fn star_edges(spokes: usize) -> Vec<(usize, usize)> {
        // Hub is qubit 0, spokes are 1..=spokes
        (1..=spokes).map(|s| (0, s)).collect()
    }

    fn ring_edges(size: usize) -> Vec<(usize, usize)> {
        let mut edges: Vec<_> = (0..size.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        if size > 2 {
            edges.push((size - 1, 0)); // Close the ring
        }
        edges
    }

    fn fully_connected_edges(size: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push((i, j));
            }
        }
        edges
    }

    /// Check if topology supports efficient all-to-all operations
    pub fn has_full_connectivity(&self) -> bool {
        matches!(self, Self::FullyConnected { .. } | Self::LinearChain { .. })
    }

    /// Get topology name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grid2D { .. } => "Grid2D",
            Self::LinearChain { .. } => "LinearChain",
            Self::HeavyHex { .. } => "HeavyHex",
            Self::Star { .. } => "Star",
            Self::Ring { .. } => "Ring",
            Self::FullyConnected { .. } => "FullyConnected",
            Self::Custom { .. } => "Custom",
        }
    }
}

impl fmt::Display for DeviceTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grid2D { width, height } => {
                write!(f, "Grid2D({}x{}, {} qubits)", width, height, width * height)
            }
            Self::LinearChain { length } => {
                write!(f, "LinearChain({} qubits)", length)
            }
            Self::HeavyHex { rows, cols } => {
                write!(
                    f,
                    "HeavyHex({}x{}, {} qubits)",
                    rows,
                    cols,
                    self.qubit_count()
                )
            }
            Self::Star { spokes } => {
                write!(f, "Star({} spokes, {} qubits)", spokes, spokes + 1)
            }
            Self::Ring { size } => {
                write!(f, "Ring({} qubits)", size)
            }
            Self::FullyConnected { size } => {
                write!(f, "FullyConnected({} qubits)", size)
            }
            Self::Custom { qubit_count, edges } => {
                write!(f, "Custom({} qubits, {} edges)", qubit_count, edges.len())
            }
        }
    }
}

/// Enhanced coupling map with precomputed shortest paths
#[derive(Clone, Debug)]
pub struct EnhancedCouplingMap {
    /// Number of qubits
    qubit_count: usize,
    /// Adjacency list (qubit -> neighbors)
    adjacency: Vec<Vec<usize>>,
    /// Precomputed shortest distances (Floyd-Warshall)
    distances: Vec<Vec<usize>>,
    /// Edge set for O(1) adjacency check
    edge_set: HashSet<(usize, usize)>,
}

impl EnhancedCouplingMap {
    /// Create from topology
    pub fn from_topology(topology: &DeviceTopology) -> Self {
        let qubit_count = topology.qubit_count();
        let edges = topology.edges();

        // Build adjacency list
        let mut adjacency = vec![Vec::new(); qubit_count];
        let mut edge_set = HashSet::new();

        for &(a, b) in &edges {
            if a < qubit_count && b < qubit_count {
                adjacency[a].push(b);
                adjacency[b].push(a);
                edge_set.insert((a.min(b), a.max(b)));
            }
        }

        // Sort adjacency lists for consistent ordering
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        // Compute shortest distances using BFS (more efficient than Floyd-Warshall for sparse graphs)
        let distances = Self::compute_distances(&adjacency, qubit_count);

        Self {
            qubit_count,
            adjacency,
            distances,
            edge_set,
        }
    }

    /// Create from explicit edges
    pub fn from_edges(qubit_count: usize, edges: &[(usize, usize)]) -> Self {
        let topology = DeviceTopology::Custom {
            qubit_count,
            edges: edges.to_vec(),
        };
        Self::from_topology(&topology)
    }

    /// Compute all-pairs shortest paths using BFS
    fn compute_distances(adjacency: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
        let mut distances = vec![vec![usize::MAX; n]; n];

        for start in 0..n {
            distances[start][start] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[start][current];
                for &neighbor in &adjacency[current] {
                    if distances[start][neighbor] == usize::MAX {
                        distances[start][neighbor] = current_dist + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    // === Queries ===

    /// Get qubit count
    pub fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    /// Check if two qubits are adjacent (connected by edge)
    pub fn are_adjacent(&self, q1: usize, q2: usize) -> bool {
        let key = (q1.min(q2), q1.max(q2));
        self.edge_set.contains(&key)
    }

    /// Get neighbors of a qubit
    pub fn neighbors(&self, qubit: usize) -> &[usize] {
        if qubit < self.qubit_count {
            &self.adjacency[qubit]
        } else {
            &[]
        }
    }

    /// Get shortest distance between two qubits
    pub fn distance(&self, q1: usize, q2: usize) -> usize {
        if q1 < self.qubit_count && q2 < self.qubit_count {
            self.distances[q1][q2]
        } else {
            usize::MAX
        }
    }

    /// Get SWAP count needed to make qubits adjacent
    /// SWAP count = distance - 1 (need to move one qubit to neighbor of other)
    pub fn swap_count(&self, q1: usize, q2: usize) -> usize {
        let dist = self.distance(q1, q2);
        if dist == usize::MAX || dist == 0 {
            0
        } else {
            dist.saturating_sub(1)
        }
    }

    /// Get shortest path between two qubits
    pub fn shortest_path(&self, q1: usize, q2: usize) -> Vec<usize> {
        if q1 >= self.qubit_count || q2 >= self.qubit_count {
            return vec![];
        }
        if q1 == q2 {
            return vec![q1];
        }

        // BFS to find path
        let mut prev = vec![None; self.qubit_count];
        let mut visited = vec![false; self.qubit_count];
        let mut queue = VecDeque::new();

        visited[q1] = true;
        queue.push_back(q1);

        while let Some(current) = queue.pop_front() {
            if current == q2 {
                break;
            }
            for &neighbor in &self.adjacency[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    prev[neighbor] = Some(current);
                    queue.push_back(neighbor);
                }
            }
        }

        // Reconstruct path
        if !visited[q2] {
            return vec![]; // No path exists
        }

        let mut path = vec![q2];
        let mut current = q2;
        while let Some(p) = prev[current] {
            path.push(p);
            current = p;
        }
        path.reverse();
        path
    }

    /// Get diameter (maximum distance between any two qubits)
    pub fn diameter(&self) -> usize {
        self.distances
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&d| d != usize::MAX)
            .max()
            .copied()
            .unwrap_or(0)
    }

    /// Get average distance between qubits
    pub fn average_distance(&self) -> f64 {
        let mut sum = 0usize;
        let mut count = 0usize;

        for i in 0..self.qubit_count {
            for j in (i + 1)..self.qubit_count {
                let d = self.distances[i][j];
                if d != usize::MAX {
                    sum += d;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Check if graph is connected
    pub fn is_connected(&self) -> bool {
        if self.qubit_count == 0 {
            return true;
        }
        self.distances[0].iter().all(|&d| d != usize::MAX)
    }

    /// Get degree of a qubit
    pub fn degree(&self, qubit: usize) -> usize {
        if qubit < self.qubit_count {
            self.adjacency[qubit].len()
        } else {
            0
        }
    }

    /// Get all edges
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.edge_set.iter().copied().collect()
    }

    /// Find a connected subgraph of given size starting from a qubit
    pub fn find_connected_subgraph(&self, start: usize, size: usize) -> Option<Vec<usize>> {
        if start >= self.qubit_count || size == 0 {
            return None;
        }
        if size > self.qubit_count {
            return None;
        }

        // BFS to find connected qubits
        let mut result = Vec::with_capacity(size);
        let mut visited = vec![false; self.qubit_count];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);
        result.push(start);

        while result.len() < size {
            if let Some(current) = queue.pop_front() {
                for &neighbor in &self.adjacency[current] {
                    if !visited[neighbor] && result.len() < size {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                        result.push(neighbor);
                    }
                }
            } else {
                return None; // Not enough connected qubits
            }
        }

        Some(result)
    }
}

impl fmt::Display for EnhancedCouplingMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CouplingMap({} qubits, {} edges, diameter={}, avg_dist={:.2})",
            self.qubit_count,
            self.edge_set.len(),
            self.diameter(),
            self.average_distance()
        )
    }
}

/// Common device topologies
pub mod common {
    use super::DeviceTopology;

    /// IBM Falcon (27 qubits, heavy-hex approximation)
    pub fn ibm_falcon() -> DeviceTopology {
        DeviceTopology::heavy_hex(5, 6) // Approximate
    }

    /// IBM Heron (133 qubits, heavy-hex)
    pub fn ibm_heron() -> DeviceTopology {
        DeviceTopology::heavy_hex(11, 13) // Approximate
    }

    /// Google Sycamore (53 qubits, 6x9 grid with some removed)
    pub fn google_sycamore() -> DeviceTopology {
        DeviceTopology::grid_2d(6, 9) // Simplified
    }

    /// IonQ Aria (25 qubits, all-to-all within chain)
    pub fn ionq_aria() -> DeviceTopology {
        DeviceTopology::fully_connected(25)
    }

    /// Quantinuum H2 (56 qubits, QCCD - modeled as all-to-all)
    pub fn quantinuum_h2() -> DeviceTopology {
        DeviceTopology::fully_connected(56)
    }

    /// QuEra Aquila (256 atoms, 2D array)
    pub fn quera_aquila() -> DeviceTopology {
        DeviceTopology::grid_2d(16, 16)
    }

    /// Small test grid (5x5)
    pub fn test_grid_5x5() -> DeviceTopology {
        DeviceTopology::grid_2d(5, 5)
    }

    /// Small test chain (10 qubits)
    pub fn test_chain_10() -> DeviceTopology {
        DeviceTopology::linear_chain(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_2d() {
        let topo = DeviceTopology::grid_2d(3, 3);
        assert_eq!(topo.qubit_count(), 9);
        assert_eq!(topo.edge_count(), 12); // 6 horizontal + 6 vertical

        let coupling = topo.to_coupling_map();
        assert!(coupling.are_adjacent(0, 1)); // Horizontal
        assert!(coupling.are_adjacent(0, 3)); // Vertical
        assert!(!coupling.are_adjacent(0, 4)); // Diagonal - not adjacent
    }

    #[test]
    fn test_linear_chain() {
        let topo = DeviceTopology::linear_chain(5);
        assert_eq!(topo.qubit_count(), 5);
        assert_eq!(topo.edge_count(), 4);

        let coupling = topo.to_coupling_map();
        assert!(coupling.are_adjacent(0, 1));
        assert!(coupling.are_adjacent(3, 4));
        assert!(!coupling.are_adjacent(0, 4));
        assert_eq!(coupling.distance(0, 4), 4);
    }

    #[test]
    fn test_fully_connected() {
        let topo = DeviceTopology::fully_connected(5);
        assert_eq!(topo.qubit_count(), 5);
        assert_eq!(topo.edge_count(), 10); // 5*4/2

        let coupling = topo.to_coupling_map();
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(coupling.are_adjacent(i, j));
                    assert_eq!(coupling.distance(i, j), 1);
                }
            }
        }
    }

    #[test]
    fn test_star_topology() {
        let topo = DeviceTopology::star(4);
        assert_eq!(topo.qubit_count(), 5); // Hub + 4 spokes
        assert_eq!(topo.edge_count(), 4);

        let coupling = topo.to_coupling_map();
        // Hub (0) connected to all spokes
        for spoke in 1..=4 {
            assert!(coupling.are_adjacent(0, spoke));
        }
        // Spokes not connected to each other
        assert!(!coupling.are_adjacent(1, 2));
        // Distance between spokes is 2 (through hub)
        assert_eq!(coupling.distance(1, 2), 2);
    }

    #[test]
    fn test_ring_topology() {
        let topo = DeviceTopology::ring(6);
        assert_eq!(topo.qubit_count(), 6);
        assert_eq!(topo.edge_count(), 6);

        let coupling = topo.to_coupling_map();
        assert!(coupling.are_adjacent(0, 1));
        assert!(coupling.are_adjacent(5, 0)); // Ring closes
        assert_eq!(coupling.diameter(), 3); // Max distance in ring of 6
    }

    #[test]
    fn test_shortest_path() {
        let topo = DeviceTopology::grid_2d(3, 3);
        let coupling = topo.to_coupling_map();

        let path = coupling.shortest_path(0, 8);
        assert_eq!(path.len(), 5); // 0 -> 1 -> 2 -> 5 -> 8 or similar
        assert_eq!(path[0], 0);
        assert_eq!(path[4], 8);
    }

    #[test]
    fn test_swap_count() {
        let topo = DeviceTopology::grid_2d(4, 4);
        let coupling = topo.to_coupling_map();

        // Adjacent qubits need 0 swaps
        assert_eq!(coupling.swap_count(0, 1), 0);

        // Distance 2 needs 1 swap
        assert_eq!(coupling.swap_count(0, 2), 1);

        // Corner to corner
        assert_eq!(coupling.distance(0, 15), 6);
        assert_eq!(coupling.swap_count(0, 15), 5);
    }

    #[test]
    fn test_find_connected_subgraph() {
        let topo = DeviceTopology::grid_2d(5, 5);
        let coupling = topo.to_coupling_map();

        let subgraph = coupling.find_connected_subgraph(0, 9);
        assert!(subgraph.is_some());
        let sg = subgraph.unwrap();
        assert_eq!(sg.len(), 9);
        assert!(sg.contains(&0));
    }

    #[test]
    fn test_common_topologies() {
        let falcon = common::ibm_falcon();
        assert!(falcon.qubit_count() >= 25);

        let aria = common::ionq_aria();
        assert_eq!(aria.qubit_count(), 25);

        let coupling = aria.to_coupling_map();
        assert_eq!(coupling.diameter(), 1); // All-to-all
    }

    #[test]
    fn test_display() {
        let topo = DeviceTopology::grid_2d(4, 4);
        let s = format!("{}", topo);
        assert!(s.contains("Grid2D"));
        assert!(s.contains("16 qubits"));

        let coupling = topo.to_coupling_map();
        let s = format!("{}", coupling);
        assert!(s.contains("16 qubits"));
    }

    #[test]
    fn test_connectivity() {
        let connected = DeviceTopology::grid_2d(3, 3);
        assert!(connected.to_coupling_map().is_connected());

        // Create disconnected graph
        let disconnected = DeviceTopology::Custom {
            qubit_count: 4,
            edges: vec![(0, 1)], // Only 0-1 connected, 2-3 isolated
        };
        assert!(!disconnected.to_coupling_map().is_connected());
    }
}
