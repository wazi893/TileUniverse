//! QRAM Binary Tree: Bucket-brigade router organization
//!
//! Arranges QRAM routers in a binary tree structure for address-based routing.
//!
//! # Tree Structure
//!
//! For n address bits (depth n), the tree has:
//! - 2^n memory cells (leaves)
//! - 2^n - 1 routers (internal nodes)
//! - n levels (0 = root, n-1 = bottom)
//!
//! ```text
//!          Level 0:     [R_0]           (1 router)
//!                       /    \
//!          Level 1:  [R_1]   [R_2]      (2 routers)
//!                    /   \   /   \
//!          Data:   [D_0][D_1][D_2][D_3]  (4 memory cells)
//! ```

use super::router::{QRAMRouter, RouterState, fredkin_gates};
use crate::quantum::QGate;

/// Binary tree of QRAM routers
///
/// Organizes routers hierarchically for bucket-brigade addressing.
/// Each level corresponds to one address bit.
#[derive(Clone, Debug)]
pub struct QRAMTree {
    /// Number of address bits (tree depth)
    depth: usize,
    /// Routers organized by level: routers[level][index]
    routers: Vec<Vec<QRAMRouter>>,
    /// Data register values (classical memory cells)
    data: Vec<u64>,
}

impl QRAMTree {
    /// Create a new QRAM tree with 2^depth memory cells
    ///
    /// # Arguments
    /// * `depth` - Number of address bits (1-20)
    ///
    /// # Panics
    /// Panics if depth is 0 or greater than 20
    pub fn new(depth: usize) -> Self {
        assert!(depth > 0 && depth <= 20, "depth must be 1-20");

        let mut routers = Vec::with_capacity(depth);
        for level in 0..depth {
            let count = 1 << level; // 2^level routers at each level
            let level_routers: Vec<_> = (0..count).map(|idx| QRAMRouter::new(level, idx)).collect();
            routers.push(level_routers);
        }

        let data = vec![0u64; 1 << depth];

        Self {
            depth,
            routers,
            data,
        }
    }

    /// Load classical data into memory cells
    ///
    /// Copies up to `size()` elements from the input slice.
    pub fn load_data(&mut self, data: &[u64]) {
        let n = data.len().min(self.data.len());
        self.data[..n].copy_from_slice(&data[..n]);
    }

    /// Get data at address (classical lookup)
    pub fn get(&self, address: usize) -> Option<u64> {
        self.data.get(address).copied()
    }

    /// Set data at address
    pub fn set(&mut self, address: usize, value: u64) {
        if address < self.data.len() {
            self.data[address] = value;
        }
    }

    /// Number of memory cells (2^depth)
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Tree depth (number of address bits)
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Total number of routers (2^depth - 1)
    pub fn router_count(&self) -> usize {
        (1 << self.depth) - 1
    }

    /// Get router at specific position
    pub fn get_router(&self, level: usize, index: usize) -> Option<&QRAMRouter> {
        self.routers.get(level).and_then(|lvl| lvl.get(index))
    }

    /// Get mutable router at specific position
    pub fn get_router_mut(&mut self, level: usize, index: usize) -> Option<&mut QRAMRouter> {
        self.routers
            .get_mut(level)
            .and_then(|lvl| lvl.get_mut(index))
    }

    /// Reset all routers to idle state
    pub fn reset_routers(&mut self) {
        for level in &mut self.routers {
            for router in level {
                router.reset();
            }
        }
    }

    /// Route a classical address through the tree
    ///
    /// Activates routers along the path from root to the addressed memory cell.
    /// Returns the memory cell index reached.
    ///
    /// # Arguments
    /// * `address` - The address to route (0 to size()-1)
    pub fn route_classical(&mut self, address: usize) -> usize {
        let mut current_idx = 0;

        for level in 0..self.depth {
            // Extract bit for this level (MSB first)
            let bit = (address >> (self.depth - 1 - level)) & 1;

            let router = &mut self.routers[level][current_idx];
            let state = if bit == 0 {
                RouterState::Left
            } else {
                RouterState::Right
            };
            router.route(state);

            // Navigate to child: left child = 2*idx, right child = 2*idx + 1
            current_idx = 2 * current_idx + bit;
        }

        // After traversing all levels, current_idx is the data cell index
        address // The address directly maps to the cell
    }

    /// Route with superposition (all paths active)
    ///
    /// Activates all routers in superposition mode.
    /// Used for quantum queries where address is in superposition.
    pub fn route_superposition(&mut self) {
        for level in &mut self.routers {
            for router in level {
                router.route(RouterState::Superposition);
            }
        }
    }

    // =========================================================================
    // Quantum Circuit Generation
    // =========================================================================

    /// Total qubits needed for quantum simulation
    ///
    /// Layout:
    /// - Address register: qubits 0..(depth-1)
    /// - Bus qubit: qubit depth
    /// - Router path qubits: depth+1.. (2 per router for left/right paths)
    pub fn total_qubits(&self) -> usize {
        // address (depth) + bus (1) + router paths (2 per router)
        self.depth + 1 + 2 * self.router_count()
    }

    /// Calculate qubit index for router path at (level, index)
    ///
    /// Each router has 2 path qubits (left and right).
    fn router_qubit_base(&self, level: usize, index: usize) -> usize {
        // After address qubits and bus qubit
        let base = self.depth + 1;
        // Count routers in all previous levels: sum of 2^i for i in 0..level
        let prev_routers: usize = if level == 0 { 0 } else { (1 << level) - 1 };
        // Each router uses 2 qubits
        base + (prev_routers + index) * 2
    }

    /// Generate quantum circuit for QRAM query
    ///
    /// The circuit implements bucket-brigade routing:
    /// 1. Address qubits control which path is taken at each level
    /// 2. Fredkin gates conditionally swap router paths
    ///
    /// # Qubit Layout
    /// - Qubits 0..depth: Address register
    /// - Qubit depth: Bus qubit (carries data out)
    /// - Qubits depth+1..: Router path qubits (2 per router)
    pub fn query_circuit(&self) -> Vec<QGate> {
        let mut gates = Vec::new();

        // For each level, apply Fredkin gates controlled by address qubit
        for level in 0..self.depth {
            let address_qubit = level as u8;
            let routers_at_level = 1 << level;

            for router_idx in 0..routers_at_level {
                // Get qubit indices for this router
                let path_base = self.router_qubit_base(level, router_idx);
                let left_path = path_base as u8;
                let right_path = (path_base + 1) as u8;

                // Fredkin gate: address_qubit controls swap of left/right paths
                gates.extend(fredkin_gates(address_qubit, left_path, right_path));
            }
        }

        gates
    }

    /// Generate initialization circuit
    ///
    /// Prepares the bus qubit and router paths for a query.
    /// For a superposition query, applies Hadamard to all address qubits.
    pub fn init_superposition_circuit(&self) -> Vec<QGate> {
        let mut gates = Vec::with_capacity(self.depth);

        // Apply Hadamard to all address qubits for uniform superposition
        for q in 0..self.depth {
            gates.push(QGate::H(q as u8));
        }

        gates
    }

    /// Generate circuit for specific address (basis state query)
    ///
    /// Prepares address register to encode a specific address.
    pub fn init_address_circuit(&self, address: usize) -> Vec<QGate> {
        let mut gates = Vec::new();

        // Set address qubits based on address bits (MSB first)
        for bit in 0..self.depth {
            if (address >> (self.depth - 1 - bit)) & 1 == 1 {
                gates.push(QGate::X(bit as u8));
            }
        }

        gates
    }

    /// Get the data qubit indices for each memory cell
    ///
    /// In a full QRAM implementation, each memory cell would have
    /// a data qubit. This returns the qubit indices for the leaf level.
    pub fn data_qubit_indices(&self) -> Vec<usize> {
        let base = self.router_qubit_base(self.depth - 1, 0);
        let last_level_routers = 1 << (self.depth - 1);

        // Data qubits come after all router path qubits
        let data_base = base + last_level_routers * 2;

        (0..self.size()).map(|i| data_base + i).collect()
    }

    /// Circuit depth estimate
    ///
    /// Each level adds one Fredkin gate depth (3 gates each).
    /// Total depth ≈ 3 * depth
    pub fn circuit_depth(&self) -> usize {
        3 * self.depth
    }

    /// Total gate count estimate
    ///
    /// Each router contributes 3 gates (Fredkin decomposition).
    /// Total = 3 * router_count = 3 * (2^depth - 1)
    pub fn gate_count(&self) -> usize {
        3 * self.router_count()
    }
}

// =============================================================================
// Display implementation for debugging
// =============================================================================

impl std::fmt::Display for QRAMTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "QRAMTree(depth={}, cells={}, routers={})",
            self.depth,
            self.size(),
            self.router_count()
        )?;

        for level in 0..self.depth {
            write!(f, "  Level {}: ", level)?;
            for (idx, router) in self.routers[level].iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "[{:?}]", router.state())?;
            }
            writeln!(f)?;
        }

        write!(f, "  Data: [")?;
        for (i, val) in self.data.iter().take(8).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        if self.data.len() > 8 {
            write!(f, ", ...")?;
        }
        writeln!(f, "]")
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree = QRAMTree::new(3);
        assert_eq!(tree.size(), 8);
        assert_eq!(tree.depth(), 3);
        assert_eq!(tree.router_count(), 7); // 1 + 2 + 4 = 7
    }

    #[test]
    fn test_tree_creation_depth_1() {
        let tree = QRAMTree::new(1);
        assert_eq!(tree.size(), 2);
        assert_eq!(tree.depth(), 1);
        assert_eq!(tree.router_count(), 1);
    }

    #[test]
    fn test_tree_creation_depth_2() {
        let tree = QRAMTree::new(2);
        assert_eq!(tree.size(), 4);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.router_count(), 3); // 1 + 2 = 3
    }

    #[test]
    fn test_tree_data_operations() {
        let mut tree = QRAMTree::new(2);

        // Load data
        tree.load_data(&[10, 20, 30, 40]);

        assert_eq!(tree.get(0), Some(10));
        assert_eq!(tree.get(1), Some(20));
        assert_eq!(tree.get(2), Some(30));
        assert_eq!(tree.get(3), Some(40));
        assert_eq!(tree.get(4), None); // Out of bounds

        // Set individual value
        tree.set(1, 200);
        assert_eq!(tree.get(1), Some(200));
    }

    #[test]
    fn test_tree_partial_load() {
        let mut tree = QRAMTree::new(3); // 8 cells

        // Load only 3 values
        tree.load_data(&[100, 200, 300]);

        assert_eq!(tree.get(0), Some(100));
        assert_eq!(tree.get(1), Some(200));
        assert_eq!(tree.get(2), Some(300));
        assert_eq!(tree.get(3), Some(0)); // Default
        assert_eq!(tree.get(7), Some(0)); // Default
    }

    #[test]
    fn test_classical_routing_depth_2() {
        let mut tree = QRAMTree::new(2);
        tree.load_data(&[10, 20, 30, 40]);

        // Address 0 = 0b00: Left, Left
        tree.reset_routers();
        let cell = tree.route_classical(0);
        assert_eq!(cell, 0);
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Left);
        assert_eq!(tree.get_router(1, 0).unwrap().state(), RouterState::Left);

        // Address 1 = 0b01: Left, Right
        tree.reset_routers();
        let cell = tree.route_classical(1);
        assert_eq!(cell, 1);
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Left);
        assert_eq!(tree.get_router(1, 0).unwrap().state(), RouterState::Right);

        // Address 2 = 0b10: Right, Left
        tree.reset_routers();
        let cell = tree.route_classical(2);
        assert_eq!(cell, 2);
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Right);
        assert_eq!(tree.get_router(1, 1).unwrap().state(), RouterState::Left);

        // Address 3 = 0b11: Right, Right
        tree.reset_routers();
        let cell = tree.route_classical(3);
        assert_eq!(cell, 3);
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Right);
        assert_eq!(tree.get_router(1, 1).unwrap().state(), RouterState::Right);
    }

    #[test]
    fn test_classical_routing_depth_3() {
        let mut tree = QRAMTree::new(3);

        // Address 5 = 0b101
        tree.reset_routers();
        let cell = tree.route_classical(5);
        assert_eq!(cell, 5);

        // Level 0: bit 2 = 1 -> Right
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Right);
        // Level 1: bit 1 = 0 -> Left (router index 1 because we went right)
        assert_eq!(tree.get_router(1, 1).unwrap().state(), RouterState::Left);
        // Level 2: bit 0 = 1 -> Right (router index 2)
        assert_eq!(tree.get_router(2, 2).unwrap().state(), RouterState::Right);
    }

    #[test]
    fn test_superposition_routing() {
        let mut tree = QRAMTree::new(2);
        tree.route_superposition();

        // All routers should be in superposition
        assert_eq!(
            tree.get_router(0, 0).unwrap().state(),
            RouterState::Superposition
        );
        assert_eq!(
            tree.get_router(1, 0).unwrap().state(),
            RouterState::Superposition
        );
        assert_eq!(
            tree.get_router(1, 1).unwrap().state(),
            RouterState::Superposition
        );
    }

    #[test]
    fn test_router_reset() {
        let mut tree = QRAMTree::new(2);

        tree.route_classical(3); // Set some states
        tree.reset_routers();

        // All should be idle
        assert_eq!(tree.get_router(0, 0).unwrap().state(), RouterState::Idle);
        assert_eq!(tree.get_router(1, 0).unwrap().state(), RouterState::Idle);
        assert_eq!(tree.get_router(1, 1).unwrap().state(), RouterState::Idle);
    }

    #[test]
    fn test_qubit_count() {
        // Depth 2: 2 address + 1 bus + 2*3 router paths = 9
        let tree = QRAMTree::new(2);
        assert_eq!(tree.total_qubits(), 2 + 1 + 2 * 3);
        assert_eq!(tree.total_qubits(), 9);

        // Depth 3: 3 address + 1 bus + 2*7 router paths = 18
        let tree = QRAMTree::new(3);
        assert_eq!(tree.total_qubits(), 3 + 1 + 2 * 7);
        assert_eq!(tree.total_qubits(), 18);

        // Depth 1: 1 address + 1 bus + 2*1 router paths = 4
        let tree = QRAMTree::new(1);
        assert_eq!(tree.total_qubits(), 1 + 1 + 2 * 1);
        assert_eq!(tree.total_qubits(), 4);
    }

    #[test]
    fn test_router_qubit_base() {
        let tree = QRAMTree::new(3);

        // Level 0, index 0: base = depth + 1 = 4
        assert_eq!(tree.router_qubit_base(0, 0), 4);

        // Level 1, index 0: base = 4 + 1*2 = 6
        assert_eq!(tree.router_qubit_base(1, 0), 6);

        // Level 1, index 1: base = 4 + 2*2 = 8
        assert_eq!(tree.router_qubit_base(1, 1), 8);

        // Level 2, index 0: base = 4 + 3*2 = 10
        assert_eq!(tree.router_qubit_base(2, 0), 10);

        // Level 2, index 3: base = 4 + 6*2 = 16
        assert_eq!(tree.router_qubit_base(2, 3), 16);
    }

    #[test]
    fn test_query_circuit_gate_count() {
        let tree = QRAMTree::new(2);
        let circuit = tree.query_circuit();

        // 3 routers * 3 gates per Fredkin = 9 gates
        assert_eq!(circuit.len(), 9);
        assert_eq!(tree.gate_count(), 9);
    }

    #[test]
    fn test_query_circuit_depth_3() {
        let tree = QRAMTree::new(3);
        let circuit = tree.query_circuit();

        // 7 routers * 3 gates per Fredkin = 21 gates
        assert_eq!(circuit.len(), 21);
        assert_eq!(tree.gate_count(), 21);
    }

    #[test]
    fn test_circuit_structure() {
        let tree = QRAMTree::new(2);
        let circuit = tree.query_circuit();

        // For depth 2:
        // - Address qubits: 0, 1
        // - Bus qubit: 2
        // - router_qubit_base(0, 0) = 2 + 1 + 0 = 3
        // So first Fredkin: control=0, left=3, right=4
        // Fredkin = CNOT(right, left) + Toffoli(ctrl, left, right) + CNOT(right, left)
        match &circuit[0] {
            QGate::CNot(c, t) => {
                assert_eq!(*c, 4); // right path controls
                assert_eq!(*t, 3); // left path target
            }
            _ => panic!("Expected CNOT"),
        }
        match &circuit[1] {
            QGate::Toffoli(c1, c2, t) => {
                assert_eq!(*c1, 0); // address qubit
                assert_eq!(*c2, 3); // left path
                assert_eq!(*t, 4); // right path
            }
            _ => panic!("Expected Toffoli"),
        }
    }

    #[test]
    fn test_init_superposition_circuit() {
        let tree = QRAMTree::new(3);
        let circuit = tree.init_superposition_circuit();

        assert_eq!(circuit.len(), 3);
        for (i, gate) in circuit.iter().enumerate() {
            match gate {
                QGate::H(q) => assert_eq!(*q, i as u8),
                _ => panic!("Expected Hadamard"),
            }
        }
    }

    #[test]
    fn test_init_address_circuit() {
        let tree = QRAMTree::new(3);

        // Address 0 = 0b000: no X gates
        let circuit = tree.init_address_circuit(0);
        assert_eq!(circuit.len(), 0);

        // Address 7 = 0b111: X on all qubits
        let circuit = tree.init_address_circuit(7);
        assert_eq!(circuit.len(), 3);

        // Address 5 = 0b101: X on qubits 0 and 2
        let circuit = tree.init_address_circuit(5);
        assert_eq!(circuit.len(), 2);
        // MSB first: bit 2 -> qubit 0, bit 0 -> qubit 2
        match &circuit[0] {
            QGate::X(q) => assert_eq!(*q, 0), // MSB
            _ => panic!("Expected X"),
        }
        match &circuit[1] {
            QGate::X(q) => assert_eq!(*q, 2), // LSB
            _ => panic!("Expected X"),
        }
    }

    #[test]
    fn test_circuit_depth_estimate() {
        let tree = QRAMTree::new(3);
        assert_eq!(tree.circuit_depth(), 9); // 3 * 3

        let tree = QRAMTree::new(5);
        assert_eq!(tree.circuit_depth(), 15); // 3 * 5
    }

    #[test]
    fn test_display() {
        let mut tree = QRAMTree::new(2);
        tree.load_data(&[10, 20, 30, 40]);
        tree.route_classical(2);

        let display = format!("{}", tree);
        assert!(display.contains("QRAMTree"));
        assert!(display.contains("depth=2"));
        assert!(display.contains("cells=4"));
        assert!(display.contains("routers=3"));
    }

    #[test]
    fn test_qubit_scaling() {
        // Verify qubit count formula for various depths
        for depth in 1..=6 {
            let tree = QRAMTree::new(depth);
            let routers = tree.router_count();
            let qubits = tree.total_qubits();

            // Routers = 2^depth - 1
            assert_eq!(routers, (1 << depth) - 1);

            // Qubits = depth + 1 + 2*routers
            assert_eq!(qubits, depth + 1 + 2 * routers);

            // Print for reference
            println!(
                "Depth {}: {} cells, {} routers, {} qubits",
                depth,
                tree.size(),
                routers,
                qubits
            );
        }
    }

    #[test]
    #[should_panic(expected = "depth must be 1-20")]
    fn test_invalid_depth_zero() {
        QRAMTree::new(0);
    }

    #[test]
    #[should_panic(expected = "depth must be 1-20")]
    fn test_invalid_depth_too_large() {
        QRAMTree::new(21);
    }
}
