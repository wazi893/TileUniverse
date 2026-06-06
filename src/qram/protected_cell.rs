//! Protected Memory Cells for Fault-Tolerant QRAM
//!
//! Sprint 47.0: Wraps QRAM data cells with Steane [[7,1,3]] encoding
//! for single-error correction capability.
//!
//! Each logical memory cell uses 7 physical qubits and can correct
//! any single-qubit X, Y, or Z error.

use crate::qec::{SimpleRng, SteaneCode};

/// Result of an error correction cycle
#[derive(Clone, Debug)]
pub struct CorrectionResult {
    /// Qubit where error was detected (if any)
    pub error_qubit: Option<usize>,
    /// Type of error detected ('X', 'Y', 'Z', or 'I' for identity)
    pub error_type: char,
    /// Whether correction succeeded (syndrome cleared)
    pub success: bool,
}

/// Statistics for a protected memory cell
#[derive(Clone, Debug, Default)]
pub struct CellStatistics {
    /// Total errors injected
    pub errors_injected: usize,
    /// Total correction cycles run
    pub correction_cycles: usize,
    /// Successful corrections
    pub successful_corrections: usize,
    /// Failed corrections (syndrome not cleared)
    pub failed_corrections: usize,
}

impl CellStatistics {
    /// Success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.correction_cycles == 0 {
            100.0
        } else {
            100.0 * self.successful_corrections as f64 / self.correction_cycles as f64
        }
    }
}

/// A memory cell protected by Steane [[7,1,3]] code
///
/// The Steane code encodes 1 logical qubit into 7 physical qubits,
/// providing distance-3 error correction (corrects any single error).
///
/// For QRAM, we use the logical qubit to store the "presence" of data
/// at this memory location. The actual classical data value is stored
/// separately and associated with the logical state.
#[derive(Clone)]
pub struct ProtectedMemoryCell {
    /// Classical data value stored at this cell
    data: u64,

    /// Steane code protecting the cell's quantum state
    code: SteaneCode,

    /// Statistics for this cell
    stats: CellStatistics,
}

impl ProtectedMemoryCell {
    /// Create a new protected memory cell with given data
    ///
    /// The cell starts in the |0_L⟩ logical state (no errors).
    pub fn new(data: u64) -> Self {
        Self {
            data,
            code: SteaneCode::new(),
            stats: CellStatistics::default(),
        }
    }

    /// Get the classical data value
    pub fn data(&self) -> u64 {
        self.data
    }

    /// Set the classical data value
    pub fn set_data(&mut self, data: u64) {
        self.data = data;
    }

    /// Get statistics for this cell
    pub fn stats(&self) -> &CellStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CellStatistics::default();
    }

    /// Inject a specific error on a specific qubit
    ///
    /// # Arguments
    /// * `qubit` - Physical qubit index (0-6)
    /// * `error_type` - 'X', 'Y', or 'Z'
    pub fn inject_error(&mut self, qubit: usize, error_type: char) {
        assert!(qubit < 7, "Steane code has 7 qubits (0-6)");
        self.code.apply_error(qubit, error_type);
        self.stats.errors_injected += 1;
    }

    /// Inject depolarizing noise on all qubits
    ///
    /// Each qubit independently has probability `p` of receiving
    /// a random Pauli error (X, Y, or Z with equal probability).
    pub fn inject_depolarizing_noise(&mut self, p: f64, rng: &mut SimpleRng) {
        for qubit in 0..7 {
            if rng.next_f64() < p {
                let error = match rng.next_usize(3) {
                    0 => 'X',
                    1 => 'Y',
                    _ => 'Z',
                };
                self.code.apply_error(qubit, error);
                self.stats.errors_injected += 1;
            }
        }
    }

    /// Inject bit-flip (X) noise on all qubits
    pub fn inject_bit_flip_noise(&mut self, p: f64, rng: &mut SimpleRng) {
        for qubit in 0..7 {
            if rng.next_f64() < p {
                self.code.apply_error(qubit, 'X');
                self.stats.errors_injected += 1;
            }
        }
    }

    /// Inject phase-flip (Z) noise on all qubits
    pub fn inject_phase_flip_noise(&mut self, p: f64, rng: &mut SimpleRng) {
        for qubit in 0..7 {
            if rng.next_f64() < p {
                self.code.apply_error(qubit, 'Z');
                self.stats.errors_injected += 1;
            }
        }
    }

    /// Run error correction cycle
    ///
    /// Measures syndrome, decodes to identify error, and applies correction.
    /// Returns information about what was detected and corrected.
    pub fn correct_errors(&mut self) -> CorrectionResult {
        let (qubit, error_type, success) = self.code.error_correction_cycle();

        self.stats.correction_cycles += 1;
        if success {
            self.stats.successful_corrections += 1;
        } else {
            self.stats.failed_corrections += 1;
        }

        CorrectionResult {
            error_qubit: qubit,
            error_type,
            success,
        }
    }

    /// Measure the current syndrome without correcting
    ///
    /// Returns (x_syndrome, z_syndrome) where:
    /// - x_syndrome detects Z errors (phase flips)
    /// - z_syndrome detects X errors (bit flips)
    pub fn measure_syndrome(&self) -> ([u8; 3], [u8; 3]) {
        self.code.measure_syndrome()
    }

    /// Check if the cell has any detectable errors
    ///
    /// Returns true if any syndrome bit is non-zero.
    pub fn has_errors(&self) -> bool {
        let (x_syn, z_syn) = self.code.measure_syndrome();
        x_syn.iter().any(|&s| s != 0) || z_syn.iter().any(|&s| s != 0)
    }

    /// Get access to the underlying Steane code (for advanced operations)
    pub fn steane_code(&self) -> &SteaneCode {
        &self.code
    }

    /// Get mutable access to the underlying Steane code
    pub fn steane_code_mut(&mut self) -> &mut SteaneCode {
        &mut self.code
    }
}

/// A protected memory bank containing multiple Steane-encoded cells
pub struct ProtectedMemoryBank {
    cells: Vec<ProtectedMemoryCell>,
}

impl ProtectedMemoryBank {
    /// Create a new memory bank with given size
    pub fn new(size: usize) -> Self {
        Self {
            cells: (0..size)
                .map(|i| ProtectedMemoryCell::new(i as u64))
                .collect(),
        }
    }

    /// Create from existing data
    pub fn from_data(data: &[u64]) -> Self {
        Self {
            cells: data.iter().map(|&d| ProtectedMemoryCell::new(d)).collect(),
        }
    }

    /// Number of cells
    pub fn size(&self) -> usize {
        self.cells.len()
    }

    /// Total physical qubits (7 per cell)
    pub fn total_qubits(&self) -> usize {
        self.cells.len() * 7
    }

    /// Get cell at address
    pub fn get(&self, address: usize) -> Option<&ProtectedMemoryCell> {
        self.cells.get(address)
    }

    /// Get mutable cell at address
    pub fn get_mut(&mut self, address: usize) -> Option<&mut ProtectedMemoryCell> {
        self.cells.get_mut(address)
    }

    /// Read data at address
    pub fn read(&self, address: usize) -> Option<u64> {
        self.cells.get(address).map(|c| c.data())
    }

    /// Write data at address
    pub fn write(&mut self, address: usize, data: u64) -> bool {
        if let Some(cell) = self.cells.get_mut(address) {
            cell.set_data(data);
            true
        } else {
            false
        }
    }

    /// Inject depolarizing noise on all cells
    pub fn inject_noise(&mut self, p: f64, rng: &mut SimpleRng) {
        for cell in &mut self.cells {
            cell.inject_depolarizing_noise(p, rng);
        }
    }

    /// Run error correction on all cells
    ///
    /// Returns (total_corrections, successful_corrections)
    pub fn correct_all(&mut self) -> (usize, usize) {
        let mut total = 0;
        let mut successful = 0;

        for cell in &mut self.cells {
            if cell.has_errors() {
                let result = cell.correct_errors();
                total += 1;
                if result.success {
                    successful += 1;
                }
            }
        }

        (total, successful)
    }

    /// Get aggregate statistics
    pub fn aggregate_stats(&self) -> CellStatistics {
        let mut agg = CellStatistics::default();
        for cell in &self.cells {
            let s = cell.stats();
            agg.errors_injected += s.errors_injected;
            agg.correction_cycles += s.correction_cycles;
            agg.successful_corrections += s.successful_corrections;
            agg.failed_corrections += s.failed_corrections;
        }
        agg
    }

    /// Count cells with errors
    pub fn cells_with_errors(&self) -> usize {
        self.cells.iter().filter(|c| c.has_errors()).count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protected_cell_creation() {
        let cell = ProtectedMemoryCell::new(42);
        assert_eq!(cell.data(), 42);
        assert!(!cell.has_errors());
    }

    #[test]
    fn test_single_x_error_correction() {
        let mut cell = ProtectedMemoryCell::new(100);

        // Inject X error on qubit 3
        cell.inject_error(3, 'X');
        assert!(cell.has_errors());

        // Correct
        let result = cell.correct_errors();
        assert!(result.success);
        assert_eq!(result.error_qubit, Some(3));
        assert_eq!(result.error_type, 'X');
        assert!(!cell.has_errors());
    }

    #[test]
    fn test_single_z_error_correction() {
        let mut cell = ProtectedMemoryCell::new(100);

        // Inject Z error on qubit 5
        cell.inject_error(5, 'Z');
        assert!(cell.has_errors());

        // Correct
        let result = cell.correct_errors();
        assert!(result.success);
        assert_eq!(result.error_qubit, Some(5));
        assert_eq!(result.error_type, 'Z');
        assert!(!cell.has_errors());
    }

    #[test]
    fn test_single_y_error_correction() {
        let mut cell = ProtectedMemoryCell::new(100);

        // Inject Y error on qubit 2
        cell.inject_error(2, 'Y');
        assert!(cell.has_errors());

        // Correct
        let result = cell.correct_errors();
        assert!(result.success);
        assert_eq!(result.error_qubit, Some(2));
        assert_eq!(result.error_type, 'Y');
        assert!(!cell.has_errors());
    }

    #[test]
    fn test_all_qubit_positions() {
        // Test correction at every qubit position
        for qubit in 0..7 {
            for error in ['X', 'Y', 'Z'] {
                let mut cell = ProtectedMemoryCell::new(0);
                cell.inject_error(qubit, error);
                let result = cell.correct_errors();
                assert!(
                    result.success,
                    "Failed to correct {} error on qubit {}",
                    error, qubit
                );
            }
        }
    }

    #[test]
    fn test_no_error_correction() {
        let mut cell = ProtectedMemoryCell::new(0);

        // No error injected
        assert!(!cell.has_errors());

        let result = cell.correct_errors();
        assert!(result.success);
        assert_eq!(result.error_qubit, None);
        assert_eq!(result.error_type, 'I');
    }

    #[test]
    fn test_depolarizing_noise() {
        let mut cell = ProtectedMemoryCell::new(0);
        let mut rng = SimpleRng::new(42);

        // High error rate to ensure some errors
        cell.inject_depolarizing_noise(0.5, &mut rng);

        // Should have injected some errors
        assert!(cell.stats().errors_injected > 0);
    }

    #[test]
    fn test_memory_bank() {
        let data = vec![100, 200, 300, 400];
        let bank = ProtectedMemoryBank::from_data(&data);

        assert_eq!(bank.size(), 4);
        assert_eq!(bank.total_qubits(), 28); // 4 * 7

        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(bank.read(i), Some(expected));
        }
    }

    #[test]
    fn test_memory_bank_error_correction() {
        let mut bank = ProtectedMemoryBank::new(8);
        let _rng = SimpleRng::new(123);

        // Inject low-rate noise (single errors expected)
        for i in 0..8 {
            if i % 2 == 0 {
                bank.get_mut(i).unwrap().inject_error(i % 7, 'X');
            }
        }

        // Count cells with errors
        assert_eq!(bank.cells_with_errors(), 4);

        // Correct all
        let (total, successful) = bank.correct_all();
        assert_eq!(total, 4);
        assert_eq!(successful, 4);
        assert_eq!(bank.cells_with_errors(), 0);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut cell = ProtectedMemoryCell::new(0);

        // Inject and correct multiple times
        for i in 0..5 {
            cell.inject_error(i % 7, 'X');
            cell.correct_errors();
        }

        let stats = cell.stats();
        assert_eq!(stats.errors_injected, 5);
        assert_eq!(stats.correction_cycles, 5);
        assert_eq!(stats.successful_corrections, 5);
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_two_error_detection() {
        let mut cell = ProtectedMemoryCell::new(0);

        // Inject TWO errors (beyond correction capability)
        cell.inject_error(0, 'X');
        cell.inject_error(1, 'X');

        // Still detectable
        assert!(cell.has_errors());

        // Correction may fail (Steane corrects only 1 error)
        let result = cell.correct_errors();
        // Result depends on error pattern - may or may not succeed
        // The important thing is we detected something
        assert!(result.error_qubit.is_some() || !result.success);
    }

    #[test]
    fn test_threshold_behavior() {
        // Run multiple trials at different error rates
        let trials = 100;
        let mut rng = SimpleRng::new(42);

        for &p in &[0.01, 0.05, 0.1, 0.2] {
            let mut successes = 0;

            for _ in 0..trials {
                let mut cell = ProtectedMemoryCell::new(0);
                cell.inject_depolarizing_noise(p, &mut rng);

                if cell.has_errors() {
                    let result = cell.correct_errors();
                    if result.success {
                        successes += 1;
                    }
                } else {
                    successes += 1; // No error = success
                }
            }

            let rate = 100.0 * successes as f64 / trials as f64;
            // At low error rates, should have high success
            if p <= 0.05 {
                assert!(rate > 80.0, "Success rate {} too low at p={}", rate, p);
            }
        }
    }
}
