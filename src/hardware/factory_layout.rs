//! Magic State Factory Layout
//!
//! SPRINT 52.0 Phase 3: Factory Layout
//!
//! Defines spatial footprints for magic state distillation factories
//! and provides layout optimization for placing multiple factories
//! on hardware devices.
//!
//! # Example
//!
//! ```ignore
//! use engine::hardware::{FactoryFootprint, FactoryLayoutOptimizer, profiles};
//!
//! let footprint = FactoryFootprint::fifteen_to_one();
//! let profile = profiles::ibm_heron();
//!
//! let optimizer = FactoryLayoutOptimizer::new(&profile);
//! let placements = optimizer.place_factories(&footprint, 2);
//! ```

use std::fmt;

use super::profile::HardwareProfile;
use super::topology::EnhancedCouplingMap;

/// Spatial footprint of a magic state distillation factory
///
/// Describes the physical layout requirements for a distillation circuit:
/// - Dimensions (width x height in qubits)
/// - Internal connectivity requirements
/// - Input/output qubit positions
#[derive(Clone, Debug)]
pub struct FactoryFootprint {
    /// Factory name/protocol
    pub name: String,
    /// Width in qubits
    pub width: usize,
    /// Height in qubits
    pub height: usize,
    /// Required internal edges (relative to origin)
    pub internal_edges: Vec<((usize, usize), (usize, usize))>,
    /// Input qubit positions (relative to origin)
    pub input_positions: Vec<(usize, usize)>,
    /// Output qubit position (relative to origin)
    pub output_position: (usize, usize),
    /// Ancilla qubit positions
    pub ancilla_positions: Vec<(usize, usize)>,
}

impl FactoryFootprint {
    /// Create footprint for 15-to-1 Bravyi-Kitaev protocol
    ///
    /// Layout: 5x4 grid with 15 input qubits + 5 ancillas
    /// ```text
    /// [I0][I1][I2][I3][I4]
    /// [I5][I6][I7][I8][I9]
    /// [I10][I11][I12][A0][A1]
    /// [A2][A3][A4][OUT][ ]
    /// ```
    pub fn fifteen_to_one() -> Self {
        let width = 5;
        let height = 4;

        // Input positions (15 qubits for input magic states)
        let input_positions: Vec<_> = (0..15).map(|i| (i % width, i / width)).collect();

        // Ancilla positions (5 qubits)
        let ancilla_positions = vec![
            (3, 2),
            (4, 2), // Row 2
            (0, 3),
            (1, 3),
            (2, 3), // Row 3
        ];

        // Output position
        let output_position = (3, 3);

        // Internal edges (grid connectivity + distillation specific)
        let mut internal_edges = Vec::new();

        // Grid connections
        for y in 0..height {
            for x in 0..width {
                if x + 1 < width {
                    internal_edges.push(((x, y), (x + 1, y)));
                }
                if y + 1 < height {
                    internal_edges.push(((x, y), (x, y + 1)));
                }
            }
        }

        Self {
            name: "15-to-1".to_string(),
            width,
            height,
            internal_edges,
            input_positions,
            output_position,
            ancilla_positions,
        }
    }

    /// Create footprint for 10-to-2 protocol
    ///
    /// More compact layout: 4x4 grid
    pub fn ten_to_two() -> Self {
        let width = 4;
        let height = 4;

        // 10 input positions
        let input_positions: Vec<_> = (0..10).map(|i| (i % width, i / width)).collect();

        // Ancilla positions (4 qubits)
        let ancilla_positions = vec![(2, 2), (3, 2), (0, 3), (1, 3)];

        // Two output positions
        let output_position = (2, 3); // Primary output

        let mut internal_edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if x + 1 < width {
                    internal_edges.push(((x, y), (x + 1, y)));
                }
                if y + 1 < height {
                    internal_edges.push(((x, y), (x, y + 1)));
                }
            }
        }

        Self {
            name: "10-to-2".to_string(),
            width,
            height,
            internal_edges,
            input_positions,
            output_position,
            ancilla_positions,
        }
    }

    /// Create footprint for 7-to-1 protocol (Steane-based)
    ///
    /// Compact layout: 3x4 grid
    pub fn seven_to_one() -> Self {
        let width = 3;
        let height = 4;

        let input_positions: Vec<_> = (0..7).map(|i| (i % width, i / width)).collect();

        let ancilla_positions = vec![(0, 2), (1, 2), (2, 2)];
        let output_position = (1, 3);

        let mut internal_edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if x + 1 < width {
                    internal_edges.push(((x, y), (x + 1, y)));
                }
                if y + 1 < height {
                    internal_edges.push(((x, y), (x, y + 1)));
                }
            }
        }

        Self {
            name: "7-to-1".to_string(),
            width,
            height,
            internal_edges,
            input_positions,
            output_position,
            ancilla_positions,
        }
    }

    /// Create custom rectangular footprint
    pub fn custom(name: &str, width: usize, height: usize, n_inputs: usize) -> Self {
        let input_positions: Vec<_> = (0..n_inputs).map(|i| (i % width, i / width)).collect();

        let mut internal_edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if x + 1 < width {
                    internal_edges.push(((x, y), (x + 1, y)));
                }
                if y + 1 < height {
                    internal_edges.push(((x, y), (x, y + 1)));
                }
            }
        }

        Self {
            name: name.to_string(),
            width,
            height,
            internal_edges,
            input_positions,
            output_position: (width / 2, height - 1),
            ancilla_positions: vec![],
        }
    }

    // === Properties ===

    /// Total qubit count
    pub fn qubit_count(&self) -> usize {
        self.width * self.height
    }

    /// Number of input qubits
    pub fn input_count(&self) -> usize {
        self.input_positions.len()
    }

    /// Number of ancilla qubits
    pub fn ancilla_count(&self) -> usize {
        self.ancilla_positions.len()
    }

    /// Get all qubit positions
    pub fn all_positions(&self) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                positions.push((x, y));
            }
        }
        positions
    }

    /// Check if a position is within the footprint
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Convert (x, y) position to linear index
    pub fn to_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Convert linear index to (x, y) position
    pub fn to_position(&self, index: usize) -> (usize, usize) {
        (index % self.width, index / self.width)
    }
}

impl fmt::Display for FactoryFootprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FactoryFootprint({}, {}x{}, {} inputs, {} qubits)",
            self.name,
            self.width,
            self.height,
            self.input_count(),
            self.qubit_count()
        )
    }
}

/// Placement of a factory on a device
#[derive(Clone, Debug)]
pub struct FactoryPlacement {
    /// Factory footprint
    pub footprint: FactoryFootprint,
    /// Physical qubits assigned (footprint index -> device qubit)
    pub qubit_mapping: Vec<usize>,
    /// Origin position on device (if grid-based)
    pub origin: Option<(usize, usize)>,
    /// SWAP overhead for internal operations
    pub swap_overhead: usize,
    /// Estimated additional cycles from routing
    pub routing_cycles: usize,
    /// Placement validity
    pub valid: bool,
}

impl FactoryPlacement {
    /// Create new placement with qubit mapping
    pub fn new(footprint: FactoryFootprint, qubit_mapping: Vec<usize>) -> Self {
        Self {
            footprint,
            qubit_mapping,
            origin: None,
            swap_overhead: 0,
            routing_cycles: 0,
            valid: true,
        }
    }

    /// Get physical qubit for footprint position
    pub fn physical_qubit(&self, x: usize, y: usize) -> Option<usize> {
        let idx = self.footprint.to_index(x, y);
        self.qubit_mapping.get(idx).copied()
    }

    /// Get physical qubit for input slot
    pub fn input_qubit(&self, input_idx: usize) -> Option<usize> {
        self.footprint
            .input_positions
            .get(input_idx)
            .and_then(|&(x, y)| self.physical_qubit(x, y))
    }

    /// Get physical output qubit
    pub fn output_qubit(&self) -> Option<usize> {
        let (x, y) = self.footprint.output_position;
        self.physical_qubit(x, y)
    }

    /// Get all physical qubits used
    pub fn physical_qubits(&self) -> &[usize] {
        &self.qubit_mapping
    }

    /// Check if placement uses a specific qubit
    pub fn uses_qubit(&self, qubit: usize) -> bool {
        self.qubit_mapping.contains(&qubit)
    }
}

impl fmt::Display for FactoryPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FactoryPlacement({}, {} qubits, {} SWAP overhead, {})",
            self.footprint.name,
            self.qubit_mapping.len(),
            self.swap_overhead,
            if self.valid { "valid" } else { "invalid" }
        )
    }
}

/// Factory layout optimizer for placing multiple factories on a device
pub struct FactoryLayoutOptimizer<'a> {
    profile: &'a HardwareProfile,
    coupling: EnhancedCouplingMap,
}

impl<'a> FactoryLayoutOptimizer<'a> {
    /// Create new optimizer for a hardware profile
    pub fn new(profile: &'a HardwareProfile) -> Self {
        let coupling = profile.coupling_map();
        Self { profile, coupling }
    }

    /// Place a single factory, finding best connected subgraph
    pub fn place_single(&self, footprint: &FactoryFootprint) -> Option<FactoryPlacement> {
        let needed = footprint.qubit_count();

        if needed > self.profile.qubit_count {
            return None;
        }

        // Find connected subgraph of required size
        // Try starting from different qubits to find best placement
        let mut best_placement: Option<FactoryPlacement> = None;
        let mut best_overhead = usize::MAX;

        for start in 0..self.profile.qubit_count {
            if let Some(qubits) = self.coupling.find_connected_subgraph(start, needed) {
                let placement = self.create_placement(footprint, &qubits);
                if placement.valid && placement.swap_overhead < best_overhead {
                    best_overhead = placement.swap_overhead;
                    best_placement = Some(placement);
                }
            }
        }

        best_placement
    }

    /// Place multiple factories without overlap
    pub fn place_factories(
        &self,
        footprint: &FactoryFootprint,
        count: usize,
    ) -> Vec<FactoryPlacement> {
        let needed_per_factory = footprint.qubit_count();
        let total_needed = needed_per_factory * count;

        if total_needed > self.profile.qubit_count {
            // Not enough qubits for all factories
            let max_factories = self.profile.qubit_count / needed_per_factory;
            return self.place_factories(footprint, max_factories);
        }

        let mut placements = Vec::new();
        let mut used_qubits = std::collections::HashSet::new();

        for _ in 0..count {
            // Find connected subgraph that doesn't overlap with used qubits
            let mut best_placement: Option<FactoryPlacement> = None;
            let mut best_overhead = usize::MAX;

            for start in 0..self.profile.qubit_count {
                if used_qubits.contains(&start) {
                    continue;
                }

                if let Some(qubits) =
                    self.find_non_overlapping_subgraph(start, needed_per_factory, &used_qubits)
                {
                    let placement = self.create_placement(footprint, &qubits);
                    if placement.valid && placement.swap_overhead < best_overhead {
                        best_overhead = placement.swap_overhead;
                        best_placement = Some(placement);
                    }
                }
            }

            if let Some(placement) = best_placement {
                for &q in &placement.qubit_mapping {
                    used_qubits.insert(q);
                }
                placements.push(placement);
            } else {
                break; // Can't place more factories
            }
        }

        placements
    }

    /// Find connected subgraph that doesn't overlap with excluded qubits
    fn find_non_overlapping_subgraph(
        &self,
        start: usize,
        size: usize,
        excluded: &std::collections::HashSet<usize>,
    ) -> Option<Vec<usize>> {
        use std::collections::VecDeque;

        let mut result = Vec::with_capacity(size);
        let mut visited = vec![false; self.profile.qubit_count];
        let mut queue = VecDeque::new();

        if excluded.contains(&start) {
            return None;
        }

        visited[start] = true;
        queue.push_back(start);
        result.push(start);

        while result.len() < size {
            if let Some(current) = queue.pop_front() {
                for &neighbor in self.coupling.neighbors(current) {
                    if !visited[neighbor] && !excluded.contains(&neighbor) && result.len() < size {
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

    /// Create placement from qubit list and calculate overhead
    fn create_placement(&self, footprint: &FactoryFootprint, qubits: &[usize]) -> FactoryPlacement {
        let mut placement = FactoryPlacement::new(footprint.clone(), qubits.to_vec());

        // Calculate SWAP overhead for internal edges
        let mut total_swaps = 0;
        for &((x1, y1), (x2, y2)) in &footprint.internal_edges {
            let idx1 = footprint.to_index(x1, y1);
            let idx2 = footprint.to_index(x2, y2);

            if idx1 < qubits.len() && idx2 < qubits.len() {
                let q1 = qubits[idx1];
                let q2 = qubits[idx2];
                total_swaps += self.coupling.swap_count(q1, q2);
            }
        }

        placement.swap_overhead = total_swaps;
        placement.routing_cycles = total_swaps * self.profile.swap_cost;
        placement.valid = true;

        placement
    }

    /// Calculate total resources for all placements
    pub fn total_resources(&self, placements: &[FactoryPlacement]) -> LayoutResources {
        let total_qubits: usize = placements.iter().map(|p| p.qubit_mapping.len()).sum();
        let total_swaps: usize = placements.iter().map(|p| p.swap_overhead).sum();
        let total_routing_cycles: usize = placements.iter().map(|p| p.routing_cycles).sum();

        LayoutResources {
            factory_count: placements.len(),
            total_qubits,
            total_swaps,
            total_routing_cycles,
            device_utilization: total_qubits as f64 / self.profile.qubit_count as f64,
        }
    }

    /// Check if footprint can fit on device
    pub fn can_fit(&self, footprint: &FactoryFootprint) -> bool {
        footprint.qubit_count() <= self.profile.qubit_count
    }

    /// Calculate maximum number of factories that can fit
    pub fn max_factories(&self, footprint: &FactoryFootprint) -> usize {
        self.profile.qubit_count / footprint.qubit_count()
    }
}

/// Resources used by factory layout
#[derive(Clone, Debug)]
pub struct LayoutResources {
    /// Number of factories placed
    pub factory_count: usize,
    /// Total qubits used
    pub total_qubits: usize,
    /// Total SWAP gates needed
    pub total_swaps: usize,
    /// Total routing cycles
    pub total_routing_cycles: usize,
    /// Device utilization (0-1)
    pub device_utilization: f64,
}

impl fmt::Display for LayoutResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LayoutResources({} factories, {} qubits, {} SWAPs, {:.1}% utilization)",
            self.factory_count,
            self.total_qubits,
            self.total_swaps,
            self.device_utilization * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::profile::profiles;

    #[test]
    fn test_fifteen_to_one_footprint() {
        let footprint = FactoryFootprint::fifteen_to_one();
        assert_eq!(footprint.width, 5);
        assert_eq!(footprint.height, 4);
        assert_eq!(footprint.qubit_count(), 20);
        assert_eq!(footprint.input_count(), 15);
    }

    #[test]
    fn test_ten_to_two_footprint() {
        let footprint = FactoryFootprint::ten_to_two();
        assert_eq!(footprint.qubit_count(), 16);
        assert_eq!(footprint.input_count(), 10);
    }

    #[test]
    fn test_seven_to_one_footprint() {
        let footprint = FactoryFootprint::seven_to_one();
        assert_eq!(footprint.qubit_count(), 12);
        assert_eq!(footprint.input_count(), 7);
    }

    #[test]
    fn test_place_single_on_heron() {
        let profile = profiles::ibm_heron();
        let optimizer = FactoryLayoutOptimizer::new(&profile);
        let footprint = FactoryFootprint::fifteen_to_one();

        let placement = optimizer.place_single(&footprint);
        assert!(placement.is_some());
        let p = placement.unwrap();
        assert_eq!(p.qubit_mapping.len(), 20);
        assert!(p.valid);
    }

    #[test]
    fn test_place_multiple_factories() {
        let profile = profiles::ibm_heron();
        let optimizer = FactoryLayoutOptimizer::new(&profile);
        let footprint = FactoryFootprint::fifteen_to_one();

        let placements = optimizer.place_factories(&footprint, 3);
        assert!(!placements.is_empty());

        // Check no overlap
        let mut all_qubits = std::collections::HashSet::new();
        for p in &placements {
            for &q in &p.qubit_mapping {
                assert!(
                    all_qubits.insert(q),
                    "Qubit {} used by multiple factories",
                    q
                );
            }
        }
    }

    #[test]
    fn test_ion_trap_no_swap_overhead() {
        let profile = profiles::ionq_aria();
        let optimizer = FactoryLayoutOptimizer::new(&profile);
        let footprint = FactoryFootprint::fifteen_to_one();

        let placement = optimizer.place_single(&footprint);
        assert!(placement.is_some());
        let p = placement.unwrap();
        // All-to-all connectivity means no SWAP overhead
        assert_eq!(p.swap_overhead, 0);
    }

    #[test]
    fn test_layout_resources() {
        let profile = profiles::ibm_heron();
        let optimizer = FactoryLayoutOptimizer::new(&profile);
        let footprint = FactoryFootprint::fifteen_to_one();

        let placements = optimizer.place_factories(&footprint, 2);
        let resources = optimizer.total_resources(&placements);

        assert_eq!(resources.factory_count, placements.len());
        assert_eq!(resources.total_qubits, placements.len() * 20);
    }

    #[test]
    fn test_max_factories() {
        let profile = profiles::ibm_falcon(); // ~27 qubits
        let optimizer = FactoryLayoutOptimizer::new(&profile);
        let footprint = FactoryFootprint::fifteen_to_one(); // 20 qubits

        assert_eq!(optimizer.max_factories(&footprint), 1);
    }

    #[test]
    fn test_placement_display() {
        let footprint = FactoryFootprint::fifteen_to_one();
        let placement = FactoryPlacement::new(footprint, (0..20).collect());
        let s = format!("{}", placement);
        assert!(s.contains("15-to-1"));
    }
}
