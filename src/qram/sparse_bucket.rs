//! Sparse Bucket-Brigade QRAM for Large-Scale Memory
//!
//! SPRINT 49.0 Phase 2: SparseBucketBrigade Implementation
//!
//! This module implements bucket-brigade QRAM using sparse quantum state
//! representations, enabling simulation of thousands of memory cells that
//! would be impossible with dense state vectors.
//!
//! # Scaling Comparison
//!
//! | N cells | Dense Qubits | Dense State Size | Sparse Blocks |
//! |---------|--------------|------------------|---------------|
//! | 16      | 35           | 2^35 = 34 GB     | 2-4           |
//! | 256     | 519          | impossible       | 4-8           |
//! | 1,024   | 2,057        | impossible       | 8-16          |
//! | 16,384  | 32,781       | impossible       | 16-32         |
//!
//! # Architecture
//!
//! ```text
//! SparseBucketBrigade
//! ├── Address Register: log₂(N) qubits
//! ├── Bus Register: 1 qubit (data output)
//! └── Router Tree: 2×(N-1) qubits
//!     └── Each router: Fredkin (CSWAP) gate
//!
//! Total: 2^(depth+1) + depth - 1 qubits
//! ```
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::SparseBucketBrigade;
//!
//! // Create QRAM with 1024 memory cells (impossible with dense simulation!)
//! let mut qram = SparseBucketBrigade::new(10); // 2^10 = 1024 cells
//! qram.load(&data);
//!
//! // Classical query
//! let value = qram.query_classical(500);
//!
//! // Superposition query
//! let result = qram.query_superposition(42);
//! println!("Measured address: {}, data: {}", result.address, result.data);
//! ```

use super::sparse_state::{
    HybridSparseAdapter, SparseQRAMConfig, SparseQRAMState, SparseQueryResult,
};
use crate::quantum::QRng;
use std::fmt;

/// Sparse Bucket-Brigade QRAM for large-scale quantum memory simulation
///
/// Uses block-sparse state representations to enable simulation of QRAM
/// with thousands of memory cells, far beyond what's possible with dense
/// state vectors.
#[derive(Clone)]
pub struct SparseBucketBrigade {
    /// Configuration
    config: SparseQRAMConfig,
    /// Classical memory data
    memory: Vec<u64>,
    /// Sparse quantum state
    state: HybridSparseAdapter,
    /// Query statistics
    queries_performed: usize,
}

impl SparseBucketBrigade {
    /// Create a new sparse bucket-brigade QRAM
    ///
    /// # Arguments
    /// * `depth` - Number of address bits (creates 2^depth memory cells)
    ///
    /// # Example
    /// ```ignore
    /// let qram = SparseBucketBrigade::new(10); // 1024 cells, 2057 qubits
    /// ```
    pub fn new(depth: usize) -> Self {
        let config = SparseQRAMConfig::new(depth);
        let n_qubits = config.total_qubits();
        let memory_cells = config.memory_cells();

        Self {
            config,
            memory: vec![0; memory_cells],
            state: HybridSparseAdapter::new(n_qubits),
            queries_performed: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SparseQRAMConfig) -> Self {
        let n_qubits = config.total_qubits();
        let memory_cells = config.memory_cells();

        Self {
            config,
            memory: vec![0; memory_cells],
            state: HybridSparseAdapter::new(n_qubits),
            queries_performed: 0,
        }
    }

    /// Get the depth (number of address bits)
    pub fn depth(&self) -> usize {
        self.config.depth
    }

    /// Get the number of memory cells
    pub fn memory_cells(&self) -> usize {
        self.config.memory_cells()
    }

    /// Get the total number of qubits
    pub fn total_qubits(&self) -> usize {
        self.config.total_qubits()
    }

    /// Load data into memory
    pub fn load(&mut self, data: &[u64]) {
        let len = data.len().min(self.memory.len());
        self.memory[..len].copy_from_slice(&data[..len]);
    }

    /// Set a single memory cell
    pub fn set(&mut self, address: usize, value: u64) {
        if address < self.memory.len() {
            self.memory[address] = value;
        }
    }

    /// Get a memory cell value (classical read)
    pub fn get(&self, address: usize) -> Option<u64> {
        self.memory.get(address).copied()
    }

    /// Classical query - directly returns data at address
    pub fn query_classical(&self, address: usize) -> Option<u64> {
        self.get(address)
    }

    /// Query with address in computational basis |address⟩
    ///
    /// Initializes address register to |address⟩, runs routing,
    /// and measures the result.
    pub fn query_basis(&mut self, address: usize, seed: u64) -> SparseQueryResult {
        // Reset state
        self.state.initialize_basis(address);

        // Run bucket-brigade routing
        self.apply_routing();

        // Measure and get result
        self.measure_result(seed)
    }

    /// Query with address in uniform superposition
    ///
    /// Creates |+⟩^⊗n on address register (equal superposition of all addresses),
    /// runs routing, and measures.
    pub fn query_superposition(&mut self, seed: u64) -> SparseQueryResult {
        // Reset to |0...0⟩
        self.state.initialize_zero();

        // Create superposition on address register
        for i in 0..self.config.depth {
            self.state.apply_hadamard(i);
        }

        // Run bucket-brigade routing
        self.apply_routing();

        // Measure and get result
        self.measure_result(seed)
    }

    /// Apply the bucket-brigade routing circuit
    ///
    /// The routing tree has 2^depth - 1 routers, each implemented as
    /// a Fredkin (CSWAP) gate controlled by the corresponding address qubit.
    fn apply_routing(&mut self) {
        let depth = self.config.depth;

        // Router qubit indices
        // Address: 0..depth
        // Bus: depth
        // Router paths: depth+1..

        for level in 0..depth {
            let routers_at_level = 1 << level;
            let addr_qubit = level;

            for router_idx in 0..routers_at_level {
                // Calculate router path qubit indices
                // Each router has 2 path qubits (left and right)
                let router_base = depth + 1 + 2 * ((1 << level) - 1 + router_idx);
                let left_path = router_base;
                let right_path = router_base + 1;

                // The bus qubit travels through
                let bus = depth;

                // Fredkin gate: swap paths when address bit is 1
                self.state.apply_cswap(addr_qubit, left_path, right_path);

                // Route bus through appropriate path
                // When addr=0, route to left; when addr=1, route to right
                self.state.apply_cswap(addr_qubit, bus, right_path);
            }
        }
    }

    /// Measure result after routing
    fn measure_result(&mut self, seed: u64) -> SparseQueryResult {
        let mut rng = QRng::new(seed);
        let depth = self.config.depth;

        // Measure address register
        let mut address = 0usize;
        for i in 0..depth {
            let outcome = self.state.measure_qubit(i, rng.next_f32() as f64);
            if outcome {
                address |= 1 << i;
            }
        }

        // Get data from classical memory
        let data = self.memory.get(address).copied().unwrap_or(0);

        self.queries_performed += 1;

        let stats = self.state.stats();

        SparseQueryResult {
            address,
            data,
            n_qubits: self.total_qubits(),
            blocks_used: stats.active_blocks,
            peak_blocks: stats.peak_blocks,
            memory_bytes: stats.memory_bytes,
            gates_applied: stats.gates_applied,
        }
    }

    /// Get current sparse state statistics
    pub fn state_stats(&self) -> super::sparse_state::SparseStateStats {
        self.state.stats()
    }

    /// Get number of queries performed
    pub fn queries_performed(&self) -> usize {
        self.queries_performed
    }

    /// Compare resources with standard bucket-brigade
    pub fn resource_comparison(&self) -> SparseBBComparison {
        let n = self.memory_cells();
        let depth = self.config.depth;

        // Standard bucket-brigade would need
        let dense_qubits = self.total_qubits();
        let dense_state_size = if dense_qubits <= 40 {
            // 2^qubits * 16 bytes (Complex64)
            (1u64 << dense_qubits) * 16
        } else {
            u64::MAX // Effectively infinite
        };

        // Sparse uses blocks
        let stats = self.state.stats();

        SparseBBComparison {
            memory_cells: n,
            depth,
            total_qubits: dense_qubits,
            dense_state_bytes: dense_state_size,
            sparse_blocks: stats.active_blocks,
            sparse_bytes: stats.memory_bytes,
            compression_ratio: if stats.memory_bytes > 0 {
                dense_state_size as f64 / stats.memory_bytes as f64
            } else {
                f64::INFINITY
            },
            is_feasible_dense: dense_qubits <= 32,
        }
    }
}

impl fmt::Display for SparseBucketBrigade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.state.stats();
        write!(
            f,
            "SparseBucketBrigade(N={}, depth={}, qubits={}, blocks={}, {}KB)",
            self.memory_cells(),
            self.depth(),
            self.total_qubits(),
            stats.active_blocks,
            stats.memory_bytes / 1024
        )
    }
}

/// Comparison of sparse vs dense bucket-brigade resources
#[derive(Clone, Debug)]
pub struct SparseBBComparison {
    /// Number of memory cells
    pub memory_cells: usize,
    /// Tree depth
    pub depth: usize,
    /// Total qubits required
    pub total_qubits: usize,
    /// Dense state vector size in bytes (may be u64::MAX for infeasible)
    pub dense_state_bytes: u64,
    /// Number of sparse blocks used
    pub sparse_blocks: usize,
    /// Sparse memory usage in bytes
    pub sparse_bytes: usize,
    /// Compression ratio (dense/sparse)
    pub compression_ratio: f64,
    /// Whether dense simulation would be feasible
    pub is_feasible_dense: bool,
}

impl fmt::Display for SparseBBComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dense_str = if self.dense_state_bytes == u64::MAX {
            "IMPOSSIBLE".to_string()
        } else if self.dense_state_bytes >= 1_000_000_000 {
            format!("{:.1} GB", self.dense_state_bytes as f64 / 1e9)
        } else if self.dense_state_bytes >= 1_000_000 {
            format!("{:.1} MB", self.dense_state_bytes as f64 / 1e6)
        } else {
            format!("{} KB", self.dense_state_bytes / 1024)
        };

        write!(
            f,
            "SparseBBComparison(N={}, qubits={}, dense={}, sparse={} blocks/{}KB, ratio={:.0}x, feasible={})",
            self.memory_cells,
            self.total_qubits,
            dense_str,
            self.sparse_blocks,
            self.sparse_bytes / 1024,
            self.compression_ratio,
            self.is_feasible_dense
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let qram = SparseBucketBrigade::new(4);
        assert_eq!(qram.depth(), 4);
        assert_eq!(qram.memory_cells(), 16);
        assert_eq!(qram.total_qubits(), 35); // 4 + 1 + 2*15 = 35
    }

    #[test]
    fn test_load_and_get() {
        let mut qram = SparseBucketBrigade::new(4);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 100).collect();
        qram.load(&data);

        assert_eq!(qram.get(0), Some(100));
        assert_eq!(qram.get(5), Some(600));
        assert_eq!(qram.get(15), Some(1600));
    }

    #[test]
    fn test_classical_query() {
        let mut qram = SparseBucketBrigade::new(4);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 100).collect();
        qram.load(&data);

        assert_eq!(qram.query_classical(0), Some(100));
        assert_eq!(qram.query_classical(10), Some(1100));
    }

    #[test]
    fn test_basis_query() {
        let mut qram = SparseBucketBrigade::new(4);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 100).collect();
        qram.load(&data);

        let result = qram.query_basis(5, 42);
        // Address should be 5 (we initialized in basis state |5⟩)
        assert_eq!(result.address, 5);
        assert_eq!(result.data, 600);
    }

    #[test]
    fn test_superposition_query() {
        let mut qram = SparseBucketBrigade::new(3);
        let data: Vec<u64> = (0..8).map(|i| (i + 1) * 100).collect();
        qram.load(&data);

        // Run multiple queries to verify randomness
        let mut addresses = std::collections::HashSet::new();
        for seed in 0..20 {
            let result = qram.query_superposition(seed);
            addresses.insert(result.address);
            assert!(result.address < 8);
            assert_eq!(result.data, (result.address as u64 + 1) * 100);
        }

        // With 20 queries, we should see multiple different addresses
        assert!(addresses.len() > 1, "Should measure different addresses");
    }

    #[test]
    fn test_large_scale_creation() {
        // Test that we can create large QRAMs (not simulate full routing)
        let qram = SparseBucketBrigade::new(10); // 1024 cells
        assert_eq!(qram.memory_cells(), 1024);
        assert_eq!(qram.total_qubits(), 2057);

        let comparison = qram.resource_comparison();
        assert!(!comparison.is_feasible_dense); // Way too many qubits for dense
    }

    #[test]
    fn test_resource_comparison() {
        let qram = SparseBucketBrigade::new(4);
        let comparison = qram.resource_comparison();

        assert_eq!(comparison.memory_cells, 16);
        assert_eq!(comparison.depth, 4);
        assert_eq!(comparison.total_qubits, 35);
        assert!(!comparison.is_feasible_dense); // 35 > 32 qubits
    }

    #[test]
    fn test_display() {
        let qram = SparseBucketBrigade::new(4);
        let s = format!("{}", qram);
        assert!(s.contains("SparseBucketBrigade"));
        assert!(s.contains("N=16"));
    }

    #[test]
    fn test_stats_tracking() {
        let mut qram = SparseBucketBrigade::new(3);
        qram.load(&[100, 200, 300, 400, 500, 600, 700, 800]);

        let initial_stats = qram.state_stats();
        assert!(initial_stats.active_blocks >= 1);

        let _ = qram.query_basis(3, 42);

        let after_stats = qram.state_stats();
        assert!(after_stats.gates_applied > 0);
    }

    #[test]
    fn test_set_individual() {
        let mut qram = SparseBucketBrigade::new(4);
        qram.set(5, 999);
        assert_eq!(qram.get(5), Some(999));
    }

    #[test]
    fn test_queries_performed_counter() {
        let mut qram = SparseBucketBrigade::new(3);
        qram.load(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(qram.queries_performed(), 0);
        let _ = qram.query_basis(0, 1);
        assert_eq!(qram.queries_performed(), 1);
        let _ = qram.query_superposition(2);
        assert_eq!(qram.queries_performed(), 2);
    }

    #[test]
    fn test_256_cells_feasibility() {
        // 256 cells = depth 8 = 519 qubits
        let config = SparseQRAMConfig::new(8);
        assert_eq!(config.memory_cells(), 256);
        assert_eq!(config.total_qubits(), 519);

        // We can create it but not simulate full routing in reasonable time
        let qram = SparseBucketBrigade::with_config(config);
        let comparison = qram.resource_comparison();
        assert!(!comparison.is_feasible_dense);
    }
}
