//! QRAM Router: Fundamental routing unit for bucket-brigade QRAM
//!
//! A QRAM router is a 3-qubit device implementing a controlled-SWAP (Fredkin gate):
//! - Control qubit (c): Determines routing direction
//! - Left path (L): Data passes through when c=|0⟩
//! - Right path (R): Data passes through when c=|1⟩
//!
//! The router implements: |c⟩|L⟩|R⟩ → |c⟩ ⊗ SWAP^c(|L⟩|R⟩)

use crate::quantum::QGate;

/// State of a single QRAM router
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RouterState {
    /// Idle - waiting for address qubit
    #[default]
    Idle,
    /// Routing left (control was |0⟩)
    Left,
    /// Routing right (control was |1⟩)
    Right,
    /// Superposition (control was |+⟩ or similar)
    Superposition,
}

/// A single QRAM router unit (Fredkin gate based)
///
/// Routers are arranged in a binary tree structure. Each router
/// conditionally swaps its two data paths based on a control qubit.
#[derive(Clone, Debug)]
pub struct QRAMRouter {
    /// Level in the tree (0 = root)
    level: usize,
    /// Index within the level (0..2^level)
    index: usize,
    /// Current routing state
    state: RouterState,
}

impl QRAMRouter {
    /// Create a new router at the specified position in the tree
    pub fn new(level: usize, index: usize) -> Self {
        Self {
            level,
            index,
            state: RouterState::Idle,
        }
    }

    /// Route based on the control qubit state
    ///
    /// Returns (left_active, right_active):
    /// - Left: (true, false)
    /// - Right: (false, true)
    /// - Superposition: (true, true) - both paths active
    /// - Idle: (false, false)
    pub fn route(&mut self, control_state: RouterState) -> (bool, bool) {
        self.state = control_state;
        match control_state {
            RouterState::Idle => (false, false),
            RouterState::Left => (true, false),
            RouterState::Right => (false, true),
            RouterState::Superposition => (true, true),
        }
    }

    /// Reset router to idle state
    pub fn reset(&mut self) {
        self.state = RouterState::Idle;
    }

    /// Get current routing state
    pub fn state(&self) -> RouterState {
        self.state
    }

    /// Get tree level (0 = root)
    pub fn level(&self) -> usize {
        self.level
    }

    /// Get index within level
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the qubit indices this router operates on
    ///
    /// Given a base qubit offset, returns (control, left_path, right_path)
    pub fn qubit_indices(&self, base_offset: usize) -> (usize, usize, usize) {
        // Control qubit is the address bit for this level
        let control = self.level;
        // Path qubits are allocated after address and bus qubits
        let left = base_offset + self.index * 2;
        let right = base_offset + self.index * 2 + 1;
        (control, left, right)
    }
}

// =============================================================================
// Fredkin (CSWAP) Gate Decomposition
// =============================================================================

/// Generate gates for Fredkin (CSWAP) on qubits (control, target1, target2)
///
/// The Fredkin gate swaps target1 and target2 when control is |1⟩.
///
/// Decomposition: CSWAP = CNOT(t2,t1) · Toffoli(c,t1,t2) · CNOT(t2,t1)
///
/// This uses 2 CNOTs + 1 Toffoli = 2 + 6 = 8 two-qubit gates minimum.
pub fn fredkin_gates(control: u8, target1: u8, target2: u8) -> Vec<QGate> {
    let mut gates = Vec::with_capacity(3);

    // CSWAP decomposition using Toffoli:
    // CSWAP(c, t1, t2) = CNOT(t2, t1) · Toffoli(c, t1, t2) · CNOT(t2, t1)
    //
    // This works because:
    // 1. First CNOT: t1 ← t1 ⊕ t2
    // 2. Toffoli: t2 ← t2 ⊕ (c ∧ t1) = t2 ⊕ (c ∧ (t1 ⊕ t2))
    // 3. Second CNOT: t1 ← t1 ⊕ t2 (now t2 is the swapped value)
    //
    // When c=1: effectively swaps t1 and t2
    // When c=0: Toffoli does nothing, CNOTs cancel out

    // First CNOT: t2 controls t1
    gates.push(QGate::CNot(target2, target1));

    // Toffoli: c and t1 control t2
    gates.push(QGate::Toffoli(control, target1, target2));

    // Second CNOT: t2 controls t1
    gates.push(QGate::CNot(target2, target1));

    gates
}

/// Generate gates for Fredkin using CCZ decomposition (alternative)
///
/// CSWAP = (I⊗CNOT) · CCZ · (I⊗CNOT) · (H⊗I⊗I) · ...
///
/// This version may be more efficient for some architectures as CCZ
/// can be implemented with fewer T-gates than Toffoli in some contexts.
pub fn fredkin_gates_ccz(control: u8, target1: u8, target2: u8) -> Vec<QGate> {
    let mut gates = Vec::with_capacity(7);

    // Alternative decomposition using CCZ:
    // CSWAP = CNOT(t1,t2) · CNOT(t2,t1) · CCZ(c,t1,t2) · CNOT(t2,t1)

    gates.push(QGate::CNot(target1, target2));
    gates.push(QGate::CNot(target2, target1));
    gates.push(QGate::CCZ(control, target1, target2));
    gates.push(QGate::CNot(target2, target1));

    gates
}

/// Count of gates in Fredkin decomposition
pub fn fredkin_gate_count() -> usize {
    3 // 2 CNOTs + 1 Toffoli
}

/// Estimated depth of Fredkin decomposition
///
/// The 3 gates are sequential, so depth = 3
/// (Toffoli itself decomposes further, but we treat it as native)
pub fn fredkin_depth() -> usize {
    3
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = QRAMRouter::new(0, 0);
        assert_eq!(router.level(), 0);
        assert_eq!(router.index(), 0);
        assert_eq!(router.state(), RouterState::Idle);
    }

    #[test]
    fn test_router_at_different_levels() {
        let root = QRAMRouter::new(0, 0);
        assert_eq!(root.level(), 0);

        let level1_left = QRAMRouter::new(1, 0);
        let level1_right = QRAMRouter::new(1, 1);
        assert_eq!(level1_left.level(), 1);
        assert_eq!(level1_right.index(), 1);

        let level2 = QRAMRouter::new(2, 3);
        assert_eq!(level2.level(), 2);
        assert_eq!(level2.index(), 3);
    }

    #[test]
    fn test_router_classical_routing_left() {
        let mut router = QRAMRouter::new(0, 0);

        let (left, right) = router.route(RouterState::Left);
        assert!(left, "Left path should be active");
        assert!(!right, "Right path should be inactive");
        assert_eq!(router.state(), RouterState::Left);
    }

    #[test]
    fn test_router_classical_routing_right() {
        let mut router = QRAMRouter::new(0, 0);

        let (left, right) = router.route(RouterState::Right);
        assert!(!left, "Left path should be inactive");
        assert!(right, "Right path should be active");
        assert_eq!(router.state(), RouterState::Right);
    }

    #[test]
    fn test_router_superposition_routing() {
        let mut router = QRAMRouter::new(0, 0);

        let (left, right) = router.route(RouterState::Superposition);
        assert!(left, "Left path should be active in superposition");
        assert!(right, "Right path should be active in superposition");
        assert_eq!(router.state(), RouterState::Superposition);
    }

    #[test]
    fn test_router_idle_routing() {
        let mut router = QRAMRouter::new(0, 0);

        let (left, right) = router.route(RouterState::Idle);
        assert!(!left, "Left path should be inactive when idle");
        assert!(!right, "Right path should be inactive when idle");
    }

    #[test]
    fn test_router_reset() {
        let mut router = QRAMRouter::new(0, 0);
        router.route(RouterState::Right);
        assert_eq!(router.state(), RouterState::Right);

        router.reset();
        assert_eq!(router.state(), RouterState::Idle);
    }

    #[test]
    fn test_router_qubit_indices() {
        let router = QRAMRouter::new(1, 0);
        // With base offset 5 (after address + bus qubits)
        let (ctrl, left, right) = router.qubit_indices(5);
        assert_eq!(ctrl, 1); // Level 1 = address qubit 1
        assert_eq!(left, 5); // First path qubit
        assert_eq!(right, 6); // Second path qubit

        let router2 = QRAMRouter::new(1, 1);
        let (ctrl2, left2, right2) = router2.qubit_indices(5);
        assert_eq!(ctrl2, 1); // Same level
        assert_eq!(left2, 7); // index 1 * 2 + 5
        assert_eq!(right2, 8);
    }

    #[test]
    fn test_fredkin_gate_structure() {
        let gates = fredkin_gates(0, 1, 2);

        assert_eq!(gates.len(), 3, "Fredkin should decompose to 3 gates");

        // First gate: CNOT(t2, t1) = CNOT(2, 1)
        match gates[0] {
            QGate::CNot(c, t) => {
                assert_eq!(c, 2);
                assert_eq!(t, 1);
            }
            _ => panic!("First gate should be CNOT"),
        }

        // Second gate: Toffoli(c, t1, t2) = Toffoli(0, 1, 2)
        match gates[1] {
            QGate::Toffoli(c1, c2, t) => {
                assert_eq!(c1, 0);
                assert_eq!(c2, 1);
                assert_eq!(t, 2);
            }
            _ => panic!("Second gate should be Toffoli"),
        }

        // Third gate: CNOT(t2, t1) = CNOT(2, 1)
        match gates[2] {
            QGate::CNot(c, t) => {
                assert_eq!(c, 2);
                assert_eq!(t, 1);
            }
            _ => panic!("Third gate should be CNOT"),
        }
    }

    #[test]
    fn test_fredkin_ccz_variant() {
        let gates = fredkin_gates_ccz(0, 1, 2);

        assert_eq!(gates.len(), 4, "CCZ-based Fredkin should have 4 gates");

        // Should contain CCZ
        let has_ccz = gates.iter().any(|g| matches!(g, QGate::CCZ(_, _, _)));
        assert!(has_ccz, "Should contain CCZ gate");
    }

    #[test]
    fn test_fredkin_different_qubits() {
        // Test with non-adjacent qubits
        let gates = fredkin_gates(5, 10, 15);

        assert_eq!(gates.len(), 3);

        match gates[1] {
            QGate::Toffoli(c1, c2, t) => {
                assert_eq!(c1, 5);
                assert_eq!(c2, 10);
                assert_eq!(t, 15);
            }
            _ => panic!("Should have Toffoli with correct qubits"),
        }
    }

    #[test]
    fn test_gate_count_and_depth() {
        assert_eq!(fredkin_gate_count(), 3);
        assert_eq!(fredkin_depth(), 3);
    }

    #[test]
    fn test_router_state_default() {
        let state: RouterState = Default::default();
        assert_eq!(state, RouterState::Idle);
    }
}
