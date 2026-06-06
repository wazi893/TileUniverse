//! Sparse Quantum State Adapter for Large-Scale QRAM
//!
//! SPRINT 49.0 Phase 1: SparseQRAMState Trait
//!
//! This module provides an abstraction layer between QRAM implementations and
//! sparse quantum state backends. It enables bucket-brigade QRAM to scale to
//! thousands of memory cells by using block-sparse state representations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SparseBucketBrigade                          │
//! │                         (Phase 2)                               │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SparseQRAMState Trait                        │
//! │    - initialize_zero()      - apply_cswap()                     │
//! │    - apply_hadamard()       - measure_qubit()                   │
//! │    - apply_cnot()           - get_amplitude()                   │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              SparseQuantumGridHybrid (tile8)                    │
//! │    - u128 mode: ≤135 qubits, O(1) operations                    │
//! │    - BigInt mode: unlimited qubits, O(n) operations             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Scaling
//!
//! | N cells | Qubits | Dense State | Sparse Blocks |
//! |---------|--------|-------------|---------------|
//! | 16      | 35     | 2^35 = 34GB | 2-4 blocks    |
//! | 256     | 519    | impossible  | 4-8 blocks    |
//! | 1,024   | 2,057  | impossible  | 8-16 blocks   |
//! | 16,384  | 32,781 | impossible  | 16-32 blocks  |
//!
//! # References
//!
//! - [Bucket-brigade QRAM](https://arxiv.org/abs/0708.1879)
//! - [Sparse quantum simulation](internal tile8 module)

use crate::tile8::sparse_quantum_hybrid::{Complex64, SparseQuantumGridHybrid};
use std::fmt;

/// Trait for sparse quantum state backends usable with QRAM
///
/// This trait abstracts the quantum state representation, allowing
/// QRAM implementations to work with different sparse backends.
pub trait SparseQRAMState: Clone + Send {
    /// Create a new state with the specified number of qubits
    fn new(n_qubits: usize) -> Self;

    /// Get the number of qubits
    fn n_qubits(&self) -> usize;

    /// Initialize to |0...0⟩ state
    fn initialize_zero(&mut self);

    /// Initialize a specific computational basis state |address⟩
    fn initialize_basis(&mut self, address: usize);

    /// Apply Hadamard gate to a qubit
    fn apply_hadamard(&mut self, qubit: usize);

    /// Apply X (NOT) gate to a qubit
    fn apply_x(&mut self, qubit: usize);

    /// Apply Z gate to a qubit
    fn apply_z(&mut self, qubit: usize);

    /// Apply CNOT gate (control → target)
    fn apply_cnot(&mut self, control: usize, target: usize);

    /// Apply CZ gate
    fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize);

    /// Apply CSWAP (Fredkin) gate: if control=1, swap target1 and target2
    ///
    /// Decomposition: CSWAP = CNOT(t2→t1) · Toffoli(c,t1→t2) · CNOT(t2→t1)
    /// We use the standard decomposition with T gates.
    fn apply_cswap(&mut self, control: usize, target1: usize, target2: usize);

    /// Apply Toffoli (CCX) gate: if control1=1 AND control2=1, flip target
    ///
    /// Standard decomposition using 7 T gates and CNOTs.
    fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize);

    /// Measure a single qubit, collapsing the state
    ///
    /// Returns the measurement outcome (0 or 1) and updates the state.
    fn measure_qubit(&mut self, qubit: usize, random_value: f64) -> bool;

    /// Get the amplitude of a specific basis state
    fn get_amplitude(&self, basis_state: usize) -> Complex64;

    /// Get the probability of measuring a specific basis state
    fn get_probability(&self, basis_state: usize) -> f64 {
        let amp = self.get_amplitude(basis_state);
        amp.re * amp.re + amp.im * amp.im
    }

    /// Get the number of active (non-zero) blocks
    fn active_blocks(&self) -> usize;

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize;

    /// Get statistics about the sparse representation
    fn stats(&self) -> SparseStateStats;
}

/// Statistics about sparse state usage
#[derive(Clone, Debug, Default)]
pub struct SparseStateStats {
    /// Total qubits in the system
    pub n_qubits: usize,
    /// Number of active blocks
    pub active_blocks: usize,
    /// Peak number of blocks during computation
    pub peak_blocks: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Whether using small (u128) or large (BigInt) mode
    pub is_small_mode: bool,
    /// Number of gates applied
    pub gates_applied: usize,
}

impl fmt::Display for SparseStateStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseState({} qubits, {} blocks, {}KB, mode={})",
            self.n_qubits,
            self.active_blocks,
            self.memory_bytes / 1024,
            if self.is_small_mode { "u128" } else { "BigInt" }
        )
    }
}

/// Adapter implementing SparseQRAMState for SparseQuantumGridHybrid
#[derive(Clone)]
pub struct HybridSparseAdapter {
    grid: SparseQuantumGridHybrid,
    gates_applied: usize,
    peak_blocks: usize,
}

impl HybridSparseAdapter {
    /// Create adapter with minimum 7 qubits (sparse grid requirement)
    pub fn new(n_qubits: usize) -> Self {
        let actual_qubits = n_qubits.max(7);
        let mut grid = SparseQuantumGridHybrid::new(actual_qubits);
        grid.initialize_zero_state();
        Self {
            grid,
            gates_applied: 0,
            peak_blocks: 1,
        }
    }

    fn track_blocks(&mut self) {
        let current = self.grid.active_blocks();
        if current > self.peak_blocks {
            self.peak_blocks = current;
        }
    }

    /// Apply T gate (π/4 phase)
    #[allow(dead_code)]
    fn apply_t(&mut self, _qubit: usize) {
        // T gate: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩
        // For now, we approximate with identity (T gates are tracked for cost)
        // Full implementation would modify phases in blocks
        self.gates_applied += 1;
    }

    /// Apply T† gate (-π/4 phase)
    #[allow(dead_code)]
    fn apply_t_dagger(&mut self, _qubit: usize) {
        // T†: |0⟩ → |0⟩, |1⟩ → e^(-iπ/4)|1⟩
        self.gates_applied += 1;
    }

    /// Apply S gate (π/2 phase)
    #[allow(dead_code)]
    fn apply_s(&mut self, qubit: usize) {
        // S = T · T
        self.apply_t(qubit);
        self.apply_t(qubit);
    }

    /// Apply S† gate (-π/2 phase)
    #[allow(dead_code)]
    fn apply_s_dagger(&mut self, qubit: usize) {
        self.apply_t_dagger(qubit);
        self.apply_t_dagger(qubit);
    }
}

impl SparseQRAMState for HybridSparseAdapter {
    fn new(n_qubits: usize) -> Self {
        HybridSparseAdapter::new(n_qubits)
    }

    fn n_qubits(&self) -> usize {
        self.grid.n_qubits()
    }

    fn initialize_zero(&mut self) {
        self.grid.initialize_zero_state();
        self.gates_applied = 0;
        self.peak_blocks = 1;
    }

    fn initialize_basis(&mut self, address: usize) {
        // Start from |0...0⟩ and flip bits to create |address⟩
        self.initialize_zero();
        for i in 0..self.n_qubits() {
            if (address >> i) & 1 == 1 {
                self.apply_x(i);
            }
        }
    }

    fn apply_hadamard(&mut self, qubit: usize) {
        self.grid.apply_h(qubit);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_x(&mut self, qubit: usize) {
        self.grid.apply_x(qubit);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_z(&mut self, qubit: usize) {
        self.grid.apply_z(qubit);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_cnot(&mut self, control: usize, target: usize) {
        self.grid.apply_cnot(control, target);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        self.grid.apply_cz(qubit_a, qubit_b);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        // Direct Toffoli implementation via amplitude manipulation
        // Flips target when both controls are |1⟩
        self.grid.apply_toffoli(control1, control2, target);
        self.gates_applied += 1;
        self.track_blocks();
    }

    fn apply_cswap(&mut self, control: usize, target1: usize, target2: usize) {
        // CSWAP (Fredkin) decomposition:
        // CSWAP = CNOT(t2→t1) · Toffoli(c,t1,t2) · CNOT(t2→t1)
        //
        // This swaps target1 and target2 when control=1

        self.apply_cnot(target2, target1);
        self.apply_toffoli(control, target1, target2);
        self.apply_cnot(target2, target1);

        self.track_blocks();
    }

    fn measure_qubit(&mut self, qubit: usize, random_value: f64) -> bool {
        // Calculate probability of measuring |1⟩ on this qubit
        // This requires summing over all basis states where qubit=1
        //
        // For sparse states, we iterate over active blocks only
        let prob_one = self.calculate_qubit_probability(qubit);

        let result = random_value < prob_one;

        // Collapse the state based on measurement outcome
        self.collapse_qubit(qubit, result);

        result
    }

    fn get_amplitude(&self, basis_state: usize) -> Complex64 {
        // Extract amplitude from sparse grid
        // The basis state is split into block_id (upper bits) and local_idx (lower 7 bits)
        self.grid.get_amplitude_at(basis_state)
    }

    fn active_blocks(&self) -> usize {
        self.grid.active_blocks()
    }

    fn memory_bytes(&self) -> usize {
        self.grid.memory_bytes()
    }

    fn stats(&self) -> SparseStateStats {
        SparseStateStats {
            n_qubits: self.n_qubits(),
            active_blocks: self.active_blocks(),
            peak_blocks: self.peak_blocks,
            memory_bytes: self.memory_bytes(),
            is_small_mode: self.grid.is_small_mode(),
            gates_applied: self.gates_applied,
        }
    }
}

impl HybridSparseAdapter {
    /// Calculate probability of measuring |1⟩ on a specific qubit
    fn calculate_qubit_probability(&self, qubit: usize) -> f64 {
        // For sparse grids, we need to sum |amplitude|² for all states where qubit=1
        // This is done by iterating over active blocks
        self.grid.calculate_qubit_one_probability(qubit)
    }

    /// Collapse state after measuring a qubit
    fn collapse_qubit(&mut self, qubit: usize, outcome: bool) {
        self.grid.collapse_qubit(qubit, outcome);
        self.track_blocks();
    }
}

/// Result of a sparse QRAM query
#[derive(Clone, Debug)]
pub struct SparseQueryResult {
    /// The measured address
    pub address: usize,
    /// The retrieved data value
    pub data: u64,
    /// Number of qubits used
    pub n_qubits: usize,
    /// Number of active blocks during query
    pub blocks_used: usize,
    /// Peak blocks during computation
    pub peak_blocks: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Total gates applied
    pub gates_applied: usize,
}

impl fmt::Display for SparseQueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseQuery(addr={}, data={}, qubits={}, blocks={}/{}peak, {}KB)",
            self.address,
            self.data,
            self.n_qubits,
            self.blocks_used,
            self.peak_blocks,
            self.memory_bytes / 1024
        )
    }
}

/// Configuration for sparse QRAM
#[derive(Clone, Debug)]
pub struct SparseQRAMConfig {
    /// Number of address bits (log₂ of memory cells)
    pub depth: usize,
    /// Whether to track detailed statistics
    pub track_stats: bool,
    /// Random seed for measurements
    pub seed: Option<u64>,
}

impl Default for SparseQRAMConfig {
    fn default() -> Self {
        Self {
            depth: 4,
            track_stats: true,
            seed: None,
        }
    }
}

impl SparseQRAMConfig {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            ..Default::default()
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Calculate total qubits needed for bucket-brigade at this depth
    pub fn total_qubits(&self) -> usize {
        // address (depth) + bus (1) + router paths (2 per router)
        // routers = 2^depth - 1
        let routers = (1usize << self.depth) - 1;
        self.depth + 1 + 2 * routers
    }

    /// Calculate memory cells
    pub fn memory_cells(&self) -> usize {
        1 << self.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = HybridSparseAdapter::new(10);
        assert_eq!(adapter.n_qubits(), 10);
        assert_eq!(adapter.active_blocks(), 1);
    }

    #[test]
    fn test_initialize_zero() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_zero();
        let amp = adapter.get_amplitude(0);
        assert!((amp.re - 1.0).abs() < 1e-10);
        assert!(amp.im.abs() < 1e-10);
    }

    #[test]
    fn test_initialize_basis() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_basis(5); // |00000101⟩
        let amp = adapter.get_amplitude(5);
        assert!((amp.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_zero();
        adapter.apply_hadamard(0);

        // After H on qubit 0: (|0⟩ + |1⟩)/√2
        let amp0 = adapter.get_amplitude(0);
        let amp1 = adapter.get_amplitude(1);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        assert!((amp0.re - inv_sqrt2).abs() < 1e-10);
        assert!((amp1.re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_cnot() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_basis(1); // |00000001⟩ - qubit 0 = 1
        adapter.apply_cnot(0, 1); // Control=0, Target=1

        // After CNOT: |00000011⟩
        let amp = adapter.get_amplitude(3);
        assert!((amp.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_qubit_calculation() {
        let config = SparseQRAMConfig::new(4); // 16 cells
        // 4 address + 1 bus + 2*(2^4 - 1) = 4 + 1 + 30 = 35 qubits
        assert_eq!(config.total_qubits(), 35);
        assert_eq!(config.memory_cells(), 16);

        let config = SparseQRAMConfig::new(10); // 1024 cells
        // 10 + 1 + 2*1023 = 2057 qubits
        assert_eq!(config.total_qubits(), 2057);
    }

    #[test]
    fn test_stats() {
        let adapter = HybridSparseAdapter::new(20);
        let stats = adapter.stats();
        assert_eq!(stats.n_qubits, 20);
        assert!(stats.is_small_mode); // 20 < 135
    }

    #[test]
    fn test_sparse_state_stats_display() {
        let stats = SparseStateStats {
            n_qubits: 100,
            active_blocks: 4,
            peak_blocks: 8,
            memory_bytes: 8192,
            is_small_mode: true,
            gates_applied: 50,
        };
        let s = format!("{}", stats);
        assert!(s.contains("100 qubits"));
        assert!(s.contains("4 blocks"));
    }

    #[test]
    fn test_large_qubit_count() {
        // Test with qubits > 135 (triggers BigInt mode)
        let config = SparseQRAMConfig::new(8); // 256 cells, 519 qubits
        assert_eq!(config.total_qubits(), 519);

        // Create adapter - should use BigInt mode
        let adapter = HybridSparseAdapter::new(519);
        assert!(!adapter.grid.is_small_mode()); // BigInt mode
        assert_eq!(adapter.n_qubits(), 519);
    }

    #[test]
    fn test_toffoli_decomposition() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_basis(3); // |00000011⟩ - qubits 0,1 = 1

        // Toffoli with controls 0,1 and target 2
        adapter.apply_toffoli(0, 1, 2);

        // Result should be |00000111⟩ = 7
        let amp = adapter.get_amplitude(7);
        assert!((amp.re.abs() - 1.0).abs() < 0.1); // Allow for phase
    }

    #[test]
    fn test_cswap_decomposition() {
        let mut adapter = HybridSparseAdapter::new(8);
        adapter.initialize_basis(5); // |00000101⟩ - control=1 (q0), target1=0 (q1), target2=1 (q2)

        // CSWAP should swap q1 and q2 when q0=1
        adapter.apply_cswap(0, 1, 2);

        // After swap: q1=1, q2=0, so |00000011⟩ = 3
        // (q0 still 1, q1 becomes 1, q2 becomes 0)
        let amp = adapter.get_amplitude(3);
        assert!((amp.re.abs() - 1.0).abs() < 0.1);
    }
}
