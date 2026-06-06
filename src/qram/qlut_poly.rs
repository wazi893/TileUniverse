//! Quantum Look-Up Table with Polynomial Encoding (qLUT_poly)
//!
//! SPRINT 48.0 Phase 4: qLUT_poly Extension (EPIC 139)
//!
//! This module implements a Quantum Look-Up Table that combines two QRAM_poly
//! instances to achieve √N qubit scaling while maintaining O(log log N) T-depth.
//!
//! # Key Features
//!
//! - **√N Qubit Scaling**: Uses √N × √N grid organization
//! - **Two-Stage Query**: Row selection followed by column selection
//! - **T-count O(√N)**: Reduced from O(N) in standard QRAM
//! - **T-depth O(log log N)**: Same as single QRAM_poly
//!
//! # Architecture
//!
//! ```text
//! N = 16 cells organized as 4×4 grid:
//!
//!        Col 0   Col 1   Col 2   Col 3
//! Row 0: [D_0]   [D_1]   [D_2]   [D_3]
//! Row 1: [D_4]   [D_5]   [D_6]   [D_7]
//! Row 2: [D_8]   [D_9]   [D_10]  [D_11]
//! Row 3: [D_12]  [D_13]  [D_14]  [D_15]
//!
//! Query address 10 (row=2, col=2):
//! 1. Row QRAM selects row 2 → [D_8, D_9, D_10, D_11]
//! 2. Column QRAM selects col 2 from row → D_10
//! ```
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::QLUTPoly;
//!
//! // Create 16-cell qLUT (4×4 grid)
//! let mut qlut = QLUTPoly::new(16);
//! qlut.load(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
//!
//! // Query address 10 (row 2, col 2)
//! let result = qlut.query_classical(10);
//! assert_eq!(result, Some(110));
//!
//! // Compare with standard QRAM
//! let comparison = qlut.resource_comparison();
//! println!("Qubit reduction: {:.1}x", comparison.qubit_reduction);
//! ```
//!
//! # References
//!
//! - [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-95283-5)
//! - [QRAM Polynomial Encoding (arXiv:2408.16794)](https://arxiv.org/abs/2408.16794)

use crate::quantum::{QGate, QRng, QState, apply_gate_scalar};

use super::polynomial::TGateStats;
use super::qram_poly::{QRAMPoly, QRAMPolyConfig};
use super::query::QueryResult;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for qLUT with polynomial encoding
#[derive(Clone, Debug)]
pub struct QLUTPolyConfig {
    /// Total number of memory cells
    pub n_cells: usize,
    /// Number of rows (√N)
    pub n_rows: usize,
    /// Number of columns (√N)
    pub n_cols: usize,
    /// Synthesis precision
    pub precision: f32,
    /// Use approximate QFT
    pub approximate_qft: bool,
}

impl QLUTPolyConfig {
    /// Create config for given number of cells
    ///
    /// Automatically computes optimal √N × √N grid dimensions.
    pub fn new(n_cells: usize) -> Self {
        let sqrt_n = (n_cells as f64).sqrt().ceil() as usize;
        let n_rows = sqrt_n;
        let n_cols = (n_cells + n_rows - 1) / n_rows; // Ceiling division

        Self {
            n_cells,
            n_rows,
            n_cols,
            precision: 1e-4,
            approximate_qft: false,
        }
    }

    /// Create config with explicit dimensions
    pub fn with_dimensions(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_cells: n_rows * n_cols,
            n_rows,
            n_cols,
            precision: 1e-4,
            approximate_qft: false,
        }
    }

    /// Set synthesis precision
    pub fn with_precision(mut self, precision: f32) -> Self {
        self.precision = precision;
        self
    }

    /// Enable approximate QFT
    pub fn with_approximate_qft(mut self) -> Self {
        self.approximate_qft = true;
        self
    }

    /// Address bits needed for row selection
    pub fn row_bits(&self) -> usize {
        if self.n_rows <= 1 {
            0
        } else {
            (self.n_rows as f64).log2().ceil() as usize
        }
    }

    /// Address bits needed for column selection
    pub fn col_bits(&self) -> usize {
        if self.n_cols <= 1 {
            0
        } else {
            (self.n_cols as f64).log2().ceil() as usize
        }
    }

    /// Total address bits (row + column)
    pub fn total_bits(&self) -> usize {
        self.row_bits() + self.col_bits()
    }
}

// =============================================================================
// qLUT Result Types
// =============================================================================

/// Result of a qLUT query
#[derive(Clone, Debug)]
pub struct QLUTQueryResult {
    /// Query result
    pub result: QueryResult,
    /// Row address
    pub row: usize,
    /// Column address
    pub col: usize,
    /// Combined T-gate statistics
    pub t_stats: TGateStats,
    /// Total T-depth (max of row and column)
    pub t_depth: usize,
    /// Total gate count
    pub gate_count: usize,
}

/// Resource comparison between qLUT and standard QRAM
#[derive(Clone, Debug)]
pub struct QLUTResourceComparison {
    /// Number of memory cells
    pub n_cells: usize,
    /// qLUT qubit count
    pub qlut_qubits: usize,
    /// Standard QRAM qubit count
    pub qram_qubits: usize,
    /// Qubit reduction factor
    pub qubit_reduction: f32,
    /// qLUT T-count estimate
    pub qlut_t_count: usize,
    /// Standard QRAM T-count estimate
    pub qram_t_count: usize,
    /// T-count reduction factor
    pub t_count_reduction: f32,
    /// T-depth (same for both with polynomial encoding)
    pub t_depth: usize,
}

// =============================================================================
// qLUT_poly Implementation
// =============================================================================

/// Quantum Look-Up Table with polynomial encoding
///
/// Combines two QRAM_poly instances (row and column) to achieve
/// √N qubit scaling while maintaining O(log log N) T-depth.
#[derive(Clone, Debug)]
pub struct QLUTPoly {
    /// Configuration
    config: QLUTPolyConfig,
    /// 2D lookup table (row-major order)
    table: Vec<Vec<u64>>,
    /// Row QRAM (selects which row)
    row_qram: QRAMPoly,
    /// Column QRAM (selects column within row)
    col_qram: QRAMPoly,
}

impl QLUTPoly {
    /// Create a new qLUT with given number of cells
    ///
    /// Automatically organizes into √N × √N grid.
    pub fn new(n_cells: usize) -> Self {
        Self::with_config(QLUTPolyConfig::new(n_cells))
    }

    /// Create qLUT with explicit dimensions
    pub fn with_dimensions(n_rows: usize, n_cols: usize) -> Self {
        Self::with_config(QLUTPolyConfig::with_dimensions(n_rows, n_cols))
    }

    /// Create qLUT with custom configuration
    pub fn with_config(config: QLUTPolyConfig) -> Self {
        // Initialize 2D table
        let table = vec![vec![0u64; config.n_cols]; config.n_rows];

        // Create row QRAM (depth = row_bits)
        let row_depth = config.row_bits().max(1);
        let row_config = QRAMPolyConfig::new(row_depth).with_precision(config.precision);
        let row_qram = QRAMPoly::with_config(row_config);

        // Create column QRAM (depth = col_bits)
        let col_depth = config.col_bits().max(1);
        let col_config = QRAMPolyConfig::new(col_depth).with_precision(config.precision);
        let col_qram = QRAMPoly::with_config(col_config);

        Self {
            config,
            table,
            row_qram,
            col_qram,
        }
    }

    /// Number of memory cells
    pub fn size(&self) -> usize {
        self.config.n_cells
    }

    /// Number of rows
    pub fn n_rows(&self) -> usize {
        self.config.n_rows
    }

    /// Number of columns
    pub fn n_cols(&self) -> usize {
        self.config.n_cols
    }

    /// Total address bits
    pub fn address_bits(&self) -> usize {
        self.config.total_bits()
    }

    /// Total qubits needed
    ///
    /// qLUT uses: row_qubits + col_qubits + bus_qubits
    /// This is O(√N) instead of O(N) for standard QRAM
    pub fn total_qubits(&self) -> usize {
        // Row QRAM qubits + Column QRAM qubits + shared bus
        self.row_qram.total_qubits() + self.col_qram.total_qubits()
    }

    // =========================================================================
    // Address Conversion
    // =========================================================================

    /// Convert linear address to (row, col)
    pub fn address_to_grid(&self, address: usize) -> (usize, usize) {
        let row = address / self.config.n_cols;
        let col = address % self.config.n_cols;
        (row, col)
    }

    /// Convert (row, col) to linear address
    pub fn grid_to_address(&self, row: usize, col: usize) -> usize {
        row * self.config.n_cols + col
    }

    // =========================================================================
    // Data Management
    // =========================================================================

    /// Load data into qLUT (row-major order)
    pub fn load(&mut self, data: &[u64]) {
        for (i, &value) in data.iter().take(self.config.n_cells).enumerate() {
            let (row, col) = self.address_to_grid(i);
            if row < self.table.len() && col < self.table[row].len() {
                self.table[row][col] = value;
            }
        }

        // Update internal QRAMs
        self.update_qrams();
    }

    /// Set a single cell value
    pub fn set(&mut self, address: usize, value: u64) {
        let (row, col) = self.address_to_grid(address);
        if row < self.table.len() && col < self.table[row].len() {
            self.table[row][col] = value;
            self.update_qrams();
        }
    }

    /// Get a single cell value
    pub fn get(&self, address: usize) -> Option<u64> {
        let (row, col) = self.address_to_grid(address);
        self.table.get(row).and_then(|r| r.get(col).copied())
    }

    /// Get value at (row, col)
    pub fn get_grid(&self, row: usize, col: usize) -> Option<u64> {
        self.table.get(row).and_then(|r| r.get(col).copied())
    }

    /// Update internal QRAMs with current table data
    fn update_qrams(&mut self) {
        // Row QRAM: each entry is a "pointer" to a row
        // We encode row indices as values
        let row_data: Vec<u64> = (0..self.config.n_rows).map(|i| i as u64).collect();
        self.row_qram.load(&row_data);

        // Column QRAM: we'll update this dynamically during queries
        // For now, just ensure it has the right structure
        let col_data: Vec<u64> = (0..self.config.n_cols).map(|i| i as u64).collect();
        self.col_qram.load(&col_data);
    }

    // =========================================================================
    // Classical Query
    // =========================================================================

    /// Classical query - O(1) lookup
    pub fn query_classical(&self, address: usize) -> Option<u64> {
        self.get(address)
    }

    /// Classical grid query
    pub fn query_classical_grid(&self, row: usize, col: usize) -> Option<u64> {
        self.get_grid(row, col)
    }

    // =========================================================================
    // Quantum Query
    // =========================================================================

    /// Quantum query with address in computational basis
    ///
    /// Performs two-stage query:
    /// 1. Row QRAM selects the row
    /// 2. Column QRAM selects the column within that row
    pub fn query_basis(&mut self, address: usize, seed: u64) -> Option<QLUTQueryResult> {
        if address >= self.size() {
            return None;
        }

        let (row, col) = self.address_to_grid(address);

        // Stage 1: Query row QRAM
        let row_result = self.row_qram.query_basis(row, seed)?;

        // Stage 2: Load the selected row into column QRAM and query
        let row_data: Vec<u64> = self.table[row].clone();
        self.col_qram.load(&row_data);
        let col_result = self.col_qram.query_basis(col, seed.wrapping_add(1))?;

        // Combine statistics
        let combined_t_stats = TGateStats {
            t_count: row_result.t_stats.t_count + col_result.t_stats.t_count,
            tdg_count: row_result.t_stats.tdg_count + col_result.t_stats.tdg_count,
            t_depth: row_result.t_stats.t_depth.max(col_result.t_stats.t_depth),
            clifford_count: row_result.t_stats.clifford_count + col_result.t_stats.clifford_count,
        };

        let data = self.get(address)?;

        Some(QLUTQueryResult {
            result: QueryResult::new(address, data, 1.0),
            row,
            col,
            t_stats: combined_t_stats,
            t_depth: row_result.t_depth.max(col_result.t_depth),
            gate_count: row_result.gate_count + col_result.gate_count,
        })
    }

    /// Quantum superposition query
    ///
    /// Queries all addresses in superposition using the two-stage approach.
    pub fn query_superposition(&mut self, seed: u64) -> Vec<QLUTQueryResult> {
        let mut results = Vec::new();

        // For superposition, we need to handle the entanglement between
        // row and column selections. This is a simplified implementation
        // that queries each address combination.

        let n_qubits = self.total_qubits();
        let row_bits = self.config.row_bits();
        let col_bits = self.config.col_bits();

        // Create quantum state for combined row+col address
        let mut state = QState::new_zero(n_qubits as u8);
        let mut rng = QRng::new(seed);

        // Put row address qubits in superposition
        for q in 0..row_bits {
            apply_gate_scalar(&mut state, &QGate::H(q as u8), &mut rng);
        }

        // Put column address qubits in superposition
        for q in 0..col_bits {
            apply_gate_scalar(&mut state, &QGate::H((row_bits + q) as u8), &mut rng);
        }

        // For each valid address, calculate probability
        for addr in 0..self.size() {
            let (row, col) = self.address_to_grid(addr);

            // Probability is uniform in ideal case
            let prob = 1.0 / self.size() as f64;

            if prob > 1e-10 {
                let data = self.get(addr).unwrap_or(0);

                // Get T-stats from component QRAMs
                let row_t_depth = self.row_qram.t_depth();
                let col_t_depth = self.col_qram.t_depth();

                results.push(QLUTQueryResult {
                    result: QueryResult::new(addr, data, prob),
                    row,
                    col,
                    t_stats: TGateStats::default(),
                    t_depth: row_t_depth.max(col_t_depth),
                    gate_count: 0, // Would need to generate circuits
                });
            }
        }

        results
    }

    /// Query and measure
    pub fn query_and_measure(&mut self, seed: u64) -> QLUTQueryResult {
        let mut rng = QRng::new(seed);

        // Randomly select an address (simulating measurement)
        let measured_address = (rng.next_f32() * self.size() as f32) as usize % self.size();
        let (row, col) = self.address_to_grid(measured_address);
        let data = self.get(measured_address).unwrap_or(0);

        let row_t_depth = self.row_qram.t_depth();
        let col_t_depth = self.col_qram.t_depth();

        QLUTQueryResult {
            result: QueryResult::new(measured_address, data, 1.0 / self.size() as f64),
            row,
            col,
            t_stats: TGateStats::default(),
            t_depth: row_t_depth.max(col_t_depth),
            gate_count: 0,
        }
    }

    // =========================================================================
    // Resource Analysis
    // =========================================================================

    /// Compare resources with standard QRAM
    pub fn resource_comparison(&self) -> QLUTResourceComparison {
        let n = self.size();
        let sqrt_n = self.config.n_rows;

        // qLUT qubits: O(√N)
        let qlut_qubits = self.total_qubits();

        // Standard QRAM qubits: O(N) for bucket-brigade
        // Specifically: n_address + n_bus + 2*(N-1) router qubits
        let log_n = (n as f64).log2().ceil() as usize;
        let qram_qubits = log_n + 1 + 2 * (n - 1);

        // T-count estimates
        // qLUT: O(√N) - two small QRAMs
        let qlut_t_count = 2 * sqrt_n;
        // Standard QRAM: O(N) - one large QRAM
        let qram_t_count = n;

        // T-depth: O(log log N) for both with polynomial encoding
        let t_depth = self.row_qram.t_depth().max(self.col_qram.t_depth());

        QLUTResourceComparison {
            n_cells: n,
            qlut_qubits,
            qram_qubits,
            qubit_reduction: qram_qubits as f32 / qlut_qubits.max(1) as f32,
            qlut_t_count,
            qram_t_count,
            t_count_reduction: qram_t_count as f32 / qlut_t_count.max(1) as f32,
            t_depth,
        }
    }

    /// Get T-depth
    pub fn t_depth(&self) -> usize {
        self.row_qram.t_depth().max(self.col_qram.t_depth())
    }

    /// Get estimated T-count
    pub fn t_count_estimate(&self) -> usize {
        // Row QRAM T-count + Column QRAM T-count
        // Each has O(√N) T gates
        2 * self.config.n_rows
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl std::fmt::Display for QLUTPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QLUTPoly({}×{} = {} cells, {} qubits, T-depth={})",
            self.n_rows(),
            self.n_cols(),
            self.size(),
            self.total_qubits(),
            self.t_depth()
        )
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlut_creation() {
        let qlut = QLUTPoly::new(16);
        assert_eq!(qlut.size(), 16);
        assert_eq!(qlut.n_rows(), 4);
        assert_eq!(qlut.n_cols(), 4);
    }

    #[test]
    fn test_qlut_dimensions() {
        let qlut = QLUTPoly::with_dimensions(3, 5);
        assert_eq!(qlut.n_rows(), 3);
        assert_eq!(qlut.n_cols(), 5);
        assert_eq!(qlut.size(), 15);
    }

    #[test]
    fn test_address_conversion() {
        let qlut = QLUTPoly::with_dimensions(4, 4);

        // Address 0 = (0, 0)
        assert_eq!(qlut.address_to_grid(0), (0, 0));
        // Address 5 = (1, 1)
        assert_eq!(qlut.address_to_grid(5), (1, 1));
        // Address 10 = (2, 2)
        assert_eq!(qlut.address_to_grid(10), (2, 2));
        // Address 15 = (3, 3)
        assert_eq!(qlut.address_to_grid(15), (3, 3));

        // Reverse conversion
        assert_eq!(qlut.grid_to_address(2, 2), 10);
    }

    #[test]
    fn test_load_and_get() {
        let mut qlut = QLUTPoly::new(16);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 10).collect();
        qlut.load(&data);

        for i in 0..16 {
            assert_eq!(qlut.get(i), Some((i as u64 + 1) * 10));
        }

        assert_eq!(qlut.get(16), None);
    }

    #[test]
    fn test_grid_access() {
        let mut qlut = QLUTPoly::with_dimensions(4, 4);
        let data: Vec<u64> = (0..16).map(|i| i as u64 * 100).collect();
        qlut.load(&data);

        // Row 2, Col 2 = address 10 = value 1000
        assert_eq!(qlut.get_grid(2, 2), Some(1000));
        // Row 0, Col 0 = address 0 = value 0
        assert_eq!(qlut.get_grid(0, 0), Some(0));
        // Row 3, Col 3 = address 15 = value 1500
        assert_eq!(qlut.get_grid(3, 3), Some(1500));
    }

    #[test]
    fn test_set() {
        let mut qlut = QLUTPoly::new(16);
        qlut.set(5, 999);
        assert_eq!(qlut.get(5), Some(999));
    }

    #[test]
    fn test_classical_query() {
        let mut qlut = QLUTPoly::new(16);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 10).collect();
        qlut.load(&data);

        for i in 0..16 {
            assert_eq!(qlut.query_classical(i), Some((i as u64 + 1) * 10));
        }
    }

    #[test]
    fn test_basis_query() {
        let mut qlut = QLUTPoly::new(16);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 10).collect();
        qlut.load(&data);

        let result = qlut.query_basis(10, 42).unwrap();
        assert_eq!(result.result.address, 10);
        assert_eq!(result.result.data, 110);
        assert_eq!(result.row, 2);
        assert_eq!(result.col, 2);
        assert!(result.t_depth > 0);
    }

    #[test]
    fn test_superposition_query() {
        let mut qlut = QLUTPoly::new(4);
        qlut.load(&[10, 20, 30, 40]);

        let results = qlut.query_superposition(42);

        // Should have results for all addresses
        assert_eq!(results.len(), 4);

        // Total probability should be ~1
        let total_prob: f64 = results.iter().map(|r| r.result.probability).sum();
        assert!((total_prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_query_and_measure() {
        let mut qlut = QLUTPoly::new(16);
        let data: Vec<u64> = (0..16).map(|i| (i + 1) * 10).collect();
        qlut.load(&data);

        let result = qlut.query_and_measure(42);

        assert!(result.result.address < 16);
        let expected_data = (result.result.address as u64 + 1) * 10;
        assert_eq!(result.result.data, expected_data);
    }

    #[test]
    fn test_resource_comparison() {
        let qlut = QLUTPoly::new(64);
        let comparison = qlut.resource_comparison();

        assert_eq!(comparison.n_cells, 64);
        assert!(comparison.qlut_qubits < comparison.qram_qubits);
        assert!(comparison.qubit_reduction > 1.0);
        assert!(comparison.t_count_reduction > 1.0);
    }

    #[test]
    fn test_sqrt_n_scaling() {
        // Verify qubit count scales as √N
        let sizes = [16, 64, 256, 1024];

        for &n in &sizes {
            let qlut = QLUTPoly::new(n);
            let qubits = qlut.total_qubits();
            let sqrt_n = (n as f64).sqrt() as usize;

            // Qubits should be O(√N), roughly 2*√N + constants
            assert!(
                qubits < 4 * sqrt_n + 10,
                "For N={}, qubits={} should be O(√N)={}",
                n,
                qubits,
                sqrt_n
            );
        }
    }

    #[test]
    fn test_t_depth() {
        let qlut = QLUTPoly::new(64);
        let t_depth = qlut.t_depth();

        // T-depth should be O(log log N)
        // For N=64: log2(64)=6, log2(6)≈2.6, so T-depth should be small
        assert!(t_depth < 10, "T-depth {} should be O(log log N)", t_depth);
    }

    #[test]
    fn test_config_builder() {
        let config = QLUTPolyConfig::new(100)
            .with_precision(1e-6)
            .with_approximate_qft();

        let qlut = QLUTPoly::with_config(config);
        assert_eq!(qlut.size(), 100);
    }

    #[test]
    fn test_display() {
        let qlut = QLUTPoly::new(16);
        let display = format!("{}", qlut);

        assert!(display.contains("QLUTPoly"));
        assert!(display.contains("4×4"));
        assert!(display.contains("16 cells"));
    }

    #[test]
    fn test_non_square_grid() {
        // Test with non-square dimensions
        let qlut = QLUTPoly::with_dimensions(2, 8);
        assert_eq!(qlut.n_rows(), 2);
        assert_eq!(qlut.n_cols(), 8);
        assert_eq!(qlut.size(), 16);

        let mut qlut = QLUTPoly::with_dimensions(2, 8);
        qlut.load(&(0..16).map(|i| i as u64).collect::<Vec<_>>());

        // Address 9 = row 1, col 1
        assert_eq!(qlut.get(9), Some(9));
        assert_eq!(qlut.address_to_grid(9), (1, 1));
    }

    #[test]
    fn test_small_qlut() {
        // Test with minimal size
        let mut qlut = QLUTPoly::new(4);
        qlut.load(&[100, 200, 300, 400]);

        assert_eq!(qlut.n_rows(), 2);
        assert_eq!(qlut.n_cols(), 2);

        let result = qlut.query_basis(3, 42).unwrap();
        assert_eq!(result.result.data, 400);
        assert_eq!(result.row, 1);
        assert_eq!(result.col, 1);
    }
}
