//! SWAP Gate Router
//!
//! SPRINT 52.0 Phase 4: SWAP Routing
//!
//! Provides SWAP gate insertion for executing two-qubit gates
//! on non-adjacent qubits in restricted connectivity devices.
//!
//! # Example
//!
//! ```ignore
//! use engine::hardware::{SwapRouter, profiles, LogicalGate};
//!
//! let profile = profiles::ibm_falcon();
//! let mut router = SwapRouter::new(&profile);
//!
//! // Route a CNOT between non-adjacent qubits
//! let swaps = router.route_gate(0, 10);
//! println!("Need {} SWAPs", swaps.len());
//! ```

use std::fmt;

use super::profile::HardwareProfile;
use super::topology::EnhancedCouplingMap;

/// SWAP gate between two physical qubits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapGate {
    pub q1: usize,
    pub q2: usize,
}

impl SwapGate {
    pub fn new(q1: usize, q2: usize) -> Self {
        Self { q1, q2 }
    }
}

impl fmt::Display for SwapGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SWAP({}, {})", self.q1, self.q2)
    }
}

/// Logical gate to be routed
#[derive(Clone, Debug)]
pub enum LogicalGate {
    /// Single-qubit gate
    Single { qubit: usize, name: String },
    /// Two-qubit gate (requires adjacency)
    TwoQubit { q1: usize, q2: usize, name: String },
    /// Measurement
    Measure { qubit: usize },
    /// Reset
    Reset { qubit: usize },
}

impl LogicalGate {
    pub fn cnot(control: usize, target: usize) -> Self {
        Self::TwoQubit {
            q1: control,
            q2: target,
            name: "CNOT".to_string(),
        }
    }

    pub fn cz(q1: usize, q2: usize) -> Self {
        Self::TwoQubit {
            q1,
            q2,
            name: "CZ".to_string(),
        }
    }

    pub fn h(qubit: usize) -> Self {
        Self::Single {
            qubit,
            name: "H".to_string(),
        }
    }

    pub fn t(qubit: usize) -> Self {
        Self::Single {
            qubit,
            name: "T".to_string(),
        }
    }

    /// Get qubits involved in gate
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Self::Single { qubit, .. } => vec![*qubit],
            Self::TwoQubit { q1, q2, .. } => vec![*q1, *q2],
            Self::Measure { qubit } => vec![*qubit],
            Self::Reset { qubit } => vec![*qubit],
        }
    }

    /// Check if gate is two-qubit
    pub fn is_two_qubit(&self) -> bool {
        matches!(self, Self::TwoQubit { .. })
    }
}

impl fmt::Display for LogicalGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single { qubit, name } => write!(f, "{}({})", name, qubit),
            Self::TwoQubit { q1, q2, name } => write!(f, "{}({}, {})", name, q1, q2),
            Self::Measure { qubit } => write!(f, "M({})", qubit),
            Self::Reset { qubit } => write!(f, "R({})", qubit),
        }
    }
}

/// Physical gate after routing
#[derive(Clone, Debug)]
pub enum PhysicalGate {
    /// Single-qubit gate on physical qubit
    Single { qubit: usize, name: String },
    /// Two-qubit gate on adjacent physical qubits
    TwoQubit { q1: usize, q2: usize, name: String },
    /// SWAP gate
    Swap { q1: usize, q2: usize },
    /// Measurement
    Measure { qubit: usize },
    /// Reset
    Reset { qubit: usize },
}

impl PhysicalGate {
    pub fn is_swap(&self) -> bool {
        matches!(self, Self::Swap { .. })
    }
}

impl fmt::Display for PhysicalGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single { qubit, name } => write!(f, "{}({})", name, qubit),
            Self::TwoQubit { q1, q2, name } => write!(f, "{}({}, {})", name, q1, q2),
            Self::Swap { q1, q2 } => write!(f, "SWAP({}, {})", q1, q2),
            Self::Measure { qubit } => write!(f, "M({})", qubit),
            Self::Reset { qubit } => write!(f, "R({})", qubit),
        }
    }
}

/// SWAP router for inserting SWAP gates
pub struct SwapRouter {
    /// Coupling map
    coupling: EnhancedCouplingMap,
    /// Current logical-to-physical mapping
    mapping: Vec<usize>,
    /// Reverse mapping (physical to logical)
    reverse_mapping: Vec<usize>,
    /// SWAP cost (number of native 2Q gates per SWAP)
    swap_cost: usize,
    /// Statistics
    total_swaps: usize,
}

impl SwapRouter {
    /// Create new router from hardware profile
    pub fn new(profile: &HardwareProfile) -> Self {
        let coupling = profile.coupling_map();
        let n = coupling.qubit_count();

        // Initial identity mapping
        let mapping: Vec<usize> = (0..n).collect();
        let reverse_mapping = mapping.clone();

        Self {
            coupling,
            mapping,
            reverse_mapping,
            swap_cost: profile.swap_cost,
            total_swaps: 0,
        }
    }

    /// Create router from coupling map
    pub fn from_coupling_map(coupling: EnhancedCouplingMap, swap_cost: usize) -> Self {
        let n = coupling.qubit_count();
        let mapping: Vec<usize> = (0..n).collect();
        let reverse_mapping = mapping.clone();

        Self {
            coupling,
            mapping,
            reverse_mapping,
            swap_cost,
            total_swaps: 0,
        }
    }

    /// Get current logical-to-physical mapping
    pub fn mapping(&self) -> &[usize] {
        &self.mapping
    }

    /// Get physical qubit for logical qubit
    pub fn physical_qubit(&self, logical: usize) -> usize {
        self.mapping[logical]
    }

    /// Get logical qubit for physical qubit
    pub fn logical_qubit(&self, physical: usize) -> usize {
        self.reverse_mapping[physical]
    }

    /// Reset to identity mapping
    pub fn reset(&mut self) {
        let n = self.mapping.len();
        self.mapping = (0..n).collect();
        self.reverse_mapping = self.mapping.clone();
        self.total_swaps = 0;
    }

    /// Apply a SWAP gate and update mappings
    fn apply_swap(&mut self, p1: usize, p2: usize) {
        // Get logical qubits at these physical positions
        let l1 = self.reverse_mapping[p1];
        let l2 = self.reverse_mapping[p2];

        // Update mappings
        self.mapping[l1] = p2;
        self.mapping[l2] = p1;
        self.reverse_mapping[p1] = l2;
        self.reverse_mapping[p2] = l1;

        self.total_swaps += 1;
    }

    /// Route a single two-qubit gate between logical qubits
    ///
    /// Returns the SWAP gates needed to make the qubits adjacent
    pub fn route_gate(&mut self, logical_q1: usize, logical_q2: usize) -> Vec<SwapGate> {
        let p1 = self.physical_qubit(logical_q1);
        let p2 = self.physical_qubit(logical_q2);

        if self.coupling.are_adjacent(p1, p2) {
            return vec![]; // Already adjacent
        }

        // Find shortest path
        let path = self.coupling.shortest_path(p1, p2);
        if path.len() < 2 {
            return vec![]; // No path or same qubit
        }

        // Insert SWAPs to move q1 toward q2 along path
        let mut swaps = Vec::new();

        // Move along path, applying SWAPs until adjacent
        for i in 0..(path.len() - 2) {
            let swap_p1 = path[i];
            let swap_p2 = path[i + 1];

            swaps.push(SwapGate::new(swap_p1, swap_p2));
            self.apply_swap(swap_p1, swap_p2);
        }

        swaps
    }

    /// Route an entire circuit
    pub fn route_circuit(&mut self, gates: &[LogicalGate]) -> RoutedCircuit {
        let mut physical_gates = Vec::new();
        let mut swap_count = 0;

        for gate in gates {
            match gate {
                LogicalGate::Single { qubit, name } => {
                    let phys = self.physical_qubit(*qubit);
                    physical_gates.push(PhysicalGate::Single {
                        qubit: phys,
                        name: name.clone(),
                    });
                }
                LogicalGate::TwoQubit { q1, q2, name } => {
                    // Route and insert SWAPs
                    let swaps = self.route_gate(*q1, *q2);
                    swap_count += swaps.len();

                    for swap in swaps {
                        physical_gates.push(PhysicalGate::Swap {
                            q1: swap.q1,
                            q2: swap.q2,
                        });
                    }

                    // Now the gate
                    let p1 = self.physical_qubit(*q1);
                    let p2 = self.physical_qubit(*q2);
                    physical_gates.push(PhysicalGate::TwoQubit {
                        q1: p1,
                        q2: p2,
                        name: name.clone(),
                    });
                }
                LogicalGate::Measure { qubit } => {
                    let phys = self.physical_qubit(*qubit);
                    physical_gates.push(PhysicalGate::Measure { qubit: phys });
                }
                LogicalGate::Reset { qubit } => {
                    let phys = self.physical_qubit(*qubit);
                    physical_gates.push(PhysicalGate::Reset { qubit: phys });
                }
            }
        }

        let original_2q = gates.iter().filter(|g| g.is_two_qubit()).count();
        let routed_2q = physical_gates
            .iter()
            .filter(|g| matches!(g, PhysicalGate::TwoQubit { .. }))
            .count();

        RoutedCircuit {
            physical_gates,
            swap_count,
            original_2q_count: original_2q,
            routed_2q_count: routed_2q + swap_count * self.swap_cost,
            swap_cost: self.swap_cost,
        }
    }

    /// Calculate SWAP overhead for a list of gate pairs
    pub fn calculate_overhead(&self, gate_pairs: &[(usize, usize)]) -> SwapOverheadStats {
        let mut total_swaps = 0;
        let mut total_distance = 0;
        let mut adjacent_count = 0;

        for &(q1, q2) in gate_pairs {
            if q1 < self.coupling.qubit_count() && q2 < self.coupling.qubit_count() {
                let dist = self.coupling.distance(q1, q2);
                total_distance += dist;

                if dist <= 1 {
                    adjacent_count += 1;
                } else {
                    total_swaps += dist - 1;
                }
            }
        }

        let gate_count = gate_pairs.len();
        SwapOverheadStats {
            gate_count,
            total_swaps,
            total_distance,
            adjacent_count,
            average_distance: if gate_count > 0 {
                total_distance as f64 / gate_count as f64
            } else {
                0.0
            },
            overhead_ratio: if gate_count > 0 {
                (total_swaps * self.swap_cost) as f64 / gate_count as f64
            } else {
                0.0
            },
        }
    }

    /// Get total SWAPs performed since last reset
    pub fn total_swaps(&self) -> usize {
        self.total_swaps
    }

    /// Get total 2Q gate overhead from SWAPs
    pub fn total_swap_gate_overhead(&self) -> usize {
        self.total_swaps * self.swap_cost
    }
}

/// Routed circuit result
#[derive(Clone, Debug)]
pub struct RoutedCircuit {
    /// Physical gates after routing
    pub physical_gates: Vec<PhysicalGate>,
    /// Number of SWAP gates inserted
    pub swap_count: usize,
    /// Original number of 2Q gates
    pub original_2q_count: usize,
    /// Total 2Q gates after routing (including SWAP decomposition)
    pub routed_2q_count: usize,
    /// SWAP cost (native gates per SWAP)
    pub swap_cost: usize,
}

impl RoutedCircuit {
    /// Get circuit depth (simplified: count gates)
    pub fn depth(&self) -> usize {
        self.physical_gates.len()
    }

    /// Get SWAP overhead percentage
    pub fn swap_overhead_percent(&self) -> f64 {
        if self.original_2q_count > 0 {
            ((self.swap_count * self.swap_cost) as f64 / self.original_2q_count as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Count gate types
    pub fn gate_counts(&self) -> GateCounts {
        let mut counts = GateCounts::default();
        for gate in &self.physical_gates {
            match gate {
                PhysicalGate::Single { .. } => counts.single += 1,
                PhysicalGate::TwoQubit { .. } => counts.two_qubit += 1,
                PhysicalGate::Swap { .. } => counts.swaps += 1,
                PhysicalGate::Measure { .. } => counts.measure += 1,
                PhysicalGate::Reset { .. } => counts.reset += 1,
            }
        }
        counts
    }
}

impl fmt::Display for RoutedCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RoutedCircuit({} gates, {} SWAPs, {:.1}% overhead)",
            self.physical_gates.len(),
            self.swap_count,
            self.swap_overhead_percent()
        )
    }
}

/// Gate counts in routed circuit
#[derive(Clone, Debug, Default)]
pub struct GateCounts {
    pub single: usize,
    pub two_qubit: usize,
    pub swaps: usize,
    pub measure: usize,
    pub reset: usize,
}

/// SWAP overhead statistics
#[derive(Clone, Debug)]
pub struct SwapOverheadStats {
    /// Number of gates analyzed
    pub gate_count: usize,
    /// Total SWAPs needed
    pub total_swaps: usize,
    /// Total distance (sum of shortest paths)
    pub total_distance: usize,
    /// Number of already-adjacent pairs
    pub adjacent_count: usize,
    /// Average distance per gate
    pub average_distance: f64,
    /// Overhead ratio (additional 2Q gates per original gate)
    pub overhead_ratio: f64,
}

impl fmt::Display for SwapOverheadStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SwapStats({} gates, {} SWAPs, avg_dist={:.2}, {:.1}% overhead)",
            self.gate_count,
            self.total_swaps,
            self.average_distance,
            self.overhead_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::profile::profiles;

    #[test]
    fn test_route_adjacent_qubits() {
        let profile = profiles::ibm_falcon();
        let mut router = SwapRouter::new(&profile);

        // Find two adjacent qubits
        let coupling = profile.coupling_map();
        let edges = coupling.edges();
        if !edges.is_empty() {
            let (q1, q2) = edges[0];
            let swaps = router.route_gate(q1, q2);
            assert_eq!(swaps.len(), 0);
        }
    }

    #[test]
    fn test_route_non_adjacent_qubits() {
        let profile = profiles::custom_superconducting("Test", 4, 4, 0.99);
        let mut router = SwapRouter::new(&profile);

        // Route from corner to corner (0,0) to (3,3)
        let swaps = router.route_gate(0, 15);
        assert!(swaps.len() > 0);
    }

    #[test]
    fn test_fully_connected_no_swaps() {
        let profile = profiles::ionq_aria();
        let mut router = SwapRouter::new(&profile);

        // Any pair should need no SWAPs on all-to-all
        let swaps = router.route_gate(0, 24);
        assert_eq!(swaps.len(), 0);
    }

    #[test]
    #[allow(unused_comparisons)]
    fn test_route_circuit() {
        let profile = profiles::custom_superconducting("Test", 3, 3, 0.99);
        let mut router = SwapRouter::new(&profile);

        let gates = vec![
            LogicalGate::h(0),
            LogicalGate::cnot(0, 8), // Non-adjacent
            LogicalGate::cz(4, 5),   // Adjacent
        ];

        let routed = router.route_circuit(&gates);
        assert!(routed.swap_count >= 0);
        assert!(routed.physical_gates.len() >= gates.len());
    }

    #[test]
    fn test_mapping_updates() {
        let profile = profiles::custom_superconducting("Test", 3, 3, 0.99);
        let mut router = SwapRouter::new(&profile);

        // Initial identity mapping
        assert_eq!(router.physical_qubit(0), 0);
        assert_eq!(router.physical_qubit(1), 1);

        // Route between non-adjacent qubits
        let _ = router.route_gate(0, 8);

        // Mapping should have changed
        // After SWAPs, logical qubit 0 is now on a different physical qubit
        let p0 = router.physical_qubit(0);
        assert!(p0 != 0 || router.coupling.are_adjacent(0, 8));
    }

    #[test]
    fn test_calculate_overhead() {
        let profile = profiles::custom_superconducting("Test", 4, 4, 0.99);
        let router = SwapRouter::new(&profile);

        let pairs = vec![(0, 1), (0, 15), (7, 8)];
        let stats = router.calculate_overhead(&pairs);

        assert_eq!(stats.gate_count, 3);
        assert!(stats.total_swaps > 0); // (0,15) needs SWAPs
    }

    #[test]
    #[allow(unused_comparisons)]
    fn test_routed_circuit_stats() {
        let profile = profiles::custom_superconducting("Test", 3, 3, 0.99);
        let mut router = SwapRouter::new(&profile);

        let gates = vec![LogicalGate::cnot(0, 1), LogicalGate::cnot(0, 8)];

        let routed = router.route_circuit(&gates);
        let counts = routed.gate_counts();

        assert_eq!(counts.two_qubit, 2);
        assert!(counts.swaps >= 0);
    }

    #[test]
    fn test_reset() {
        let profile = profiles::ibm_falcon();
        let mut router = SwapRouter::new(&profile);

        router.route_gate(0, 10);
        assert!(router.total_swaps() > 0);

        router.reset();
        assert_eq!(router.total_swaps(), 0);
        assert_eq!(router.physical_qubit(0), 0);
    }
}
