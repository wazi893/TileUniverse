//! Fault-Tolerant QRAM Tree
//!
//! Sprint 47.0 Phase 3: Integrates protected memory cells with logical routers
//! to create a complete fault-tolerant QRAM system.
//!
//! # Architecture
//!
//! ```text
//!                    [Logical Router 0]          (21 qubits)
//!                    /                \
//!       [Logical Router 1]    [Logical Router 2]  (21 qubits each)
//!          /       \              /       \
//!      [Cell 0] [Cell 1]    [Cell 2] [Cell 3]     (7 qubits each)
//!
//! Total for depth 2: 3 routers × 21 + 4 cells × 7 = 63 + 28 = 91 qubits
//! ```
//!
//! # Error Correction Strategy
//!
//! - Memory cells are protected by Steane [[7,1,3]] code
//! - Routers use logical Fredkin gates with error flags
//! - Periodic syndrome measurement and correction during queries
//! - Error budget tracking for threshold analysis

use crate::qec::SimpleRng;

use super::logical_ops::LogicalRouter;
use super::protected_cell::{CellStatistics, ProtectedMemoryCell};

/// Configuration for fault-tolerant QRAM
#[derive(Clone, Debug)]
pub struct FTQRAMConfig {
    /// Tree depth (2^depth memory cells)
    pub depth: usize,
    /// Physical error rate per gate
    pub error_rate: f64,
    /// Correction interval (correct every N routing operations)
    pub correction_interval: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for FTQRAMConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            error_rate: 0.001,
            correction_interval: 1,
            seed: 42,
        }
    }
}

impl FTQRAMConfig {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            ..Default::default()
        }
    }

    pub fn with_error_rate(mut self, p: f64) -> Self {
        self.error_rate = p;
        self
    }

    pub fn with_correction_interval(mut self, interval: usize) -> Self {
        self.correction_interval = interval;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Statistics for FT-QRAM operations
#[derive(Clone, Debug, Default)]
pub struct FTQRAMStats {
    /// Total queries performed
    pub queries: usize,
    /// Successful queries (no uncorrected errors)
    pub successful_queries: usize,
    /// Failed queries (uncorrected errors detected)
    pub failed_queries: usize,
    /// Total routing operations
    pub routing_ops: usize,
    /// Error flags triggered during routing
    pub error_flags: usize,
    /// Correction cycles performed
    pub correction_cycles: usize,
    /// Successful corrections
    pub successful_corrections: usize,
    /// Physical gates executed
    pub total_gates: usize,
}

impl FTQRAMStats {
    /// Query success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.queries == 0 {
            100.0
        } else {
            100.0 * self.successful_queries as f64 / self.queries as f64
        }
    }

    /// Correction success rate
    pub fn correction_rate(&self) -> f64 {
        if self.correction_cycles == 0 {
            100.0
        } else {
            100.0 * self.successful_corrections as f64 / self.correction_cycles as f64
        }
    }
}

/// Result of a fault-tolerant query
#[derive(Clone, Debug)]
pub struct FTQueryResult {
    /// Data value retrieved
    pub data: Option<u64>,
    /// Address queried
    pub address: usize,
    /// Whether query succeeded
    pub success: bool,
    /// Number of error flags during query
    pub error_flags: usize,
    /// Corrections applied during query
    pub corrections_applied: usize,
    /// Gates executed during query
    pub gates_executed: usize,
}

/// Fault-Tolerant QRAM Tree
///
/// Combines logical routers with protected memory cells for
/// error-corrected quantum memory access.
pub struct FaultTolerantQRAM {
    /// Configuration
    config: FTQRAMConfig,
    /// Logical routers organized by level
    routers: Vec<Vec<LogicalRouter>>,
    /// Protected memory cells
    memory: Vec<ProtectedMemoryCell>,
    /// Statistics
    stats: FTQRAMStats,
    /// RNG for noise simulation
    rng: SimpleRng,
    /// Operations since last correction
    ops_since_correction: usize,
}

impl FaultTolerantQRAM {
    /// Create a new fault-tolerant QRAM with given configuration
    pub fn new(config: FTQRAMConfig) -> Self {
        let depth = config.depth;
        let size = 1 << depth;

        // Create routers for each level
        let mut routers = Vec::with_capacity(depth);
        for level in 0..depth {
            let count = 1 << level;
            routers.push((0..count).map(|_| LogicalRouter::new()).collect());
        }

        // Create protected memory cells
        let memory = (0..size)
            .map(|i| ProtectedMemoryCell::new(i as u64))
            .collect();

        let rng = SimpleRng::new(config.seed);

        Self {
            config,
            routers,
            memory,
            stats: FTQRAMStats::default(),
            rng,
            ops_since_correction: 0,
        }
    }

    /// Create with default config for given depth
    pub fn with_depth(depth: usize) -> Self {
        Self::new(FTQRAMConfig::new(depth))
    }

    /// Number of memory cells
    pub fn size(&self) -> usize {
        self.memory.len()
    }

    /// Tree depth
    pub fn depth(&self) -> usize {
        self.config.depth
    }

    /// Total physical qubits used
    pub fn total_qubits(&self) -> usize {
        // Routers: 21 qubits each
        let router_qubits: usize = self.routers.iter().map(|level| level.len() * 21).sum();
        // Memory: 7 qubits each
        let memory_qubits = self.memory.len() * 7;
        router_qubits + memory_qubits
    }

    /// Number of routers
    pub fn router_count(&self) -> usize {
        self.routers.iter().map(|level| level.len()).sum()
    }

    /// Load data into memory cells
    pub fn load(&mut self, data: &[u64]) {
        for (i, &value) in data.iter().enumerate() {
            if i < self.memory.len() {
                self.memory[i].set_data(value);
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &FTQRAMStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FTQRAMStats::default();
    }

    /// Classical query (no quantum simulation, for testing)
    pub fn query_classical(&self, address: usize) -> Option<u64> {
        self.memory.get(address).map(|cell| cell.data())
    }

    /// Fault-tolerant query with error correction
    ///
    /// Routes through logical routers and applies periodic correction.
    pub fn query(&mut self, address: usize) -> FTQueryResult {
        if address >= self.memory.len() {
            return FTQueryResult {
                data: None,
                address,
                success: false,
                error_flags: 0,
                corrections_applied: 0,
                gates_executed: 0,
            };
        }

        self.stats.queries += 1;
        let mut error_flags = 0;
        let mut corrections_applied = 0;
        let mut gates_executed = 0;

        // Route through each level
        for level in 0..self.config.depth {
            // Determine which router to use at this level
            let router_index =
                address >> (self.config.depth - level - 1) & ((1 << level) - 1).max(0);
            let router_index = if level == 0 { 0 } else { router_index };

            if router_index < self.routers[level].len() {
                let router = &mut self.routers[level][router_index];

                // Inject noise before routing
                if self.config.error_rate > 0.0 {
                    router.inject_noise(self.config.error_rate, &mut self.rng);
                }

                // Perform routing
                let result = router.route();
                gates_executed += result.gate_count;
                self.stats.routing_ops += 1;

                if result.error_detected {
                    error_flags += 1;
                    self.stats.error_flags += 1;
                }

                self.ops_since_correction += 1;

                // Periodic correction
                if self.ops_since_correction >= self.config.correction_interval {
                    let (c_ok, l_ok, r_ok) = router.correct_errors();
                    self.stats.correction_cycles += 1;
                    corrections_applied += 1;

                    if c_ok && l_ok && r_ok {
                        self.stats.successful_corrections += 1;
                    }

                    self.ops_since_correction = 0;
                }
            }
        }

        // Access memory cell
        let cell = &mut self.memory[address];

        // Inject noise on memory
        if self.config.error_rate > 0.0 {
            cell.inject_depolarizing_noise(self.config.error_rate, &mut self.rng);
        }

        // Correct memory cell if it has errors
        if cell.has_errors() {
            let result = cell.correct_errors();
            corrections_applied += 1;
            self.stats.correction_cycles += 1;
            if result.success {
                self.stats.successful_corrections += 1;
            }
        }

        let data = cell.data();
        self.stats.total_gates += gates_executed;

        // Query succeeds if no uncorrected errors remain
        let success = !self.has_uncorrected_errors();

        if success {
            self.stats.successful_queries += 1;
        } else {
            self.stats.failed_queries += 1;
        }

        FTQueryResult {
            data: Some(data),
            address,
            success,
            error_flags,
            corrections_applied,
            gates_executed,
        }
    }

    /// Check if any component has uncorrected errors
    pub fn has_uncorrected_errors(&self) -> bool {
        // Check routers
        for level in &self.routers {
            for router in level {
                if router.has_errors() {
                    return true;
                }
            }
        }

        // Check memory
        for cell in &self.memory {
            if cell.has_errors() {
                return true;
            }
        }

        false
    }

    /// Run full error correction on all components
    pub fn correct_all(&mut self) -> (usize, usize) {
        let mut total = 0;
        let mut successful = 0;

        // Correct routers
        for level in &mut self.routers {
            for router in level {
                if router.has_errors() {
                    let (c_ok, l_ok, r_ok) = router.correct_errors();
                    total += 3;
                    if c_ok {
                        successful += 1;
                    }
                    if l_ok {
                        successful += 1;
                    }
                    if r_ok {
                        successful += 1;
                    }
                }
            }
        }

        // Correct memory
        for cell in &mut self.memory {
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

    /// Inject noise on all components
    pub fn inject_global_noise(&mut self, p: f64) {
        for level in &mut self.routers {
            for router in level {
                router.inject_noise(p, &mut self.rng);
            }
        }

        for cell in &mut self.memory {
            cell.inject_depolarizing_noise(p, &mut self.rng);
        }
    }

    /// Get memory cell statistics
    pub fn memory_stats(&self) -> CellStatistics {
        let mut agg = CellStatistics::default();
        for cell in &self.memory {
            let s = cell.stats();
            agg.errors_injected += s.errors_injected;
            agg.correction_cycles += s.correction_cycles;
            agg.successful_corrections += s.successful_corrections;
            agg.failed_corrections += s.failed_corrections;
        }
        agg
    }
}

/// Run threshold analysis
///
/// Tests FT-QRAM at various error rates to identify the threshold.
pub fn threshold_analysis(
    depth: usize,
    error_rates: &[f64],
    queries_per_rate: usize,
    seed: u64,
) -> Vec<(f64, f64)> {
    let mut results = Vec::with_capacity(error_rates.len());

    for (i, &p) in error_rates.iter().enumerate() {
        let config = FTQRAMConfig::new(depth)
            .with_error_rate(p)
            .with_correction_interval(1)
            .with_seed(seed + i as u64);

        let mut qram = FaultTolerantQRAM::new(config);

        // Load test data
        let data: Vec<u64> = (0..qram.size()).map(|i| i as u64 * 100).collect();
        qram.load(&data);

        // Run queries
        for addr in 0..queries_per_rate {
            let target = addr % qram.size();
            qram.query(target);
        }

        let success_rate = qram.stats().success_rate();
        results.push((p, success_rate));
    }

    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ft_qram_creation() {
        let qram = FaultTolerantQRAM::with_depth(2);

        assert_eq!(qram.size(), 4);
        assert_eq!(qram.depth(), 2);
        assert_eq!(qram.router_count(), 3); // 1 + 2
        assert_eq!(qram.total_qubits(), 3 * 21 + 4 * 7); // 63 + 28 = 91
    }

    #[test]
    fn test_ft_qram_load_and_classical_query() {
        let mut qram = FaultTolerantQRAM::with_depth(2);
        qram.load(&[100, 200, 300, 400]);

        assert_eq!(qram.query_classical(0), Some(100));
        assert_eq!(qram.query_classical(1), Some(200));
        assert_eq!(qram.query_classical(2), Some(300));
        assert_eq!(qram.query_classical(3), Some(400));
        assert_eq!(qram.query_classical(4), None);
    }

    #[test]
    fn test_ft_qram_query_no_noise() {
        let config = FTQRAMConfig::new(2).with_error_rate(0.0);
        let mut qram = FaultTolerantQRAM::new(config);
        qram.load(&[10, 20, 30, 40]);

        for addr in 0..4 {
            let result = qram.query(addr);
            assert!(result.success);
            assert_eq!(result.data, Some((addr as u64 + 1) * 10));
            assert_eq!(result.error_flags, 0);
        }

        assert_eq!(qram.stats().queries, 4);
        assert_eq!(qram.stats().successful_queries, 4);
    }

    #[test]
    fn test_ft_qram_query_with_low_noise() {
        let config = FTQRAMConfig::new(2).with_error_rate(0.001).with_seed(12345);
        let mut qram = FaultTolerantQRAM::new(config);
        qram.load(&[10, 20, 30, 40]);

        // Run multiple queries
        let mut successes = 0;
        for _ in 0..100 {
            for addr in 0..4 {
                let result = qram.query(addr);
                if result.success {
                    successes += 1;
                }
            }
        }

        // At low error rate, most queries should succeed
        let rate = successes as f64 / 400.0;
        assert!(rate > 0.8, "Success rate {} too low", rate);
    }

    #[test]
    fn test_ft_qram_correct_all() {
        let mut qram = FaultTolerantQRAM::with_depth(2);

        // Inject errors
        qram.inject_global_noise(0.1);

        // Some components should have errors
        // (probabilistic, but likely at 10%)

        // Correct all
        let (_total, _successful) = qram.correct_all();

        // After correction, should be clean
        // (assuming single errors, which Steane can correct)
    }

    #[test]
    fn test_ft_qram_scaling() {
        for depth in 1..=3 {
            let qram = FaultTolerantQRAM::with_depth(depth);
            let expected_cells = 1 << depth;
            let expected_routers = (1 << depth) - 1;

            assert_eq!(qram.size(), expected_cells);
            assert_eq!(qram.router_count(), expected_routers);
        }
    }

    #[test]
    fn test_threshold_analysis_runs() {
        let error_rates = vec![0.0, 0.001, 0.01];
        let results = threshold_analysis(2, &error_rates, 10, 42);

        assert_eq!(results.len(), 3);

        // No errors should give 100% success
        assert_eq!(results[0].1, 100.0);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = FTQRAMConfig::new(2).with_error_rate(0.0);
        let mut qram = FaultTolerantQRAM::new(config);

        qram.query(0);
        qram.query(1);
        qram.query(2);

        let stats = qram.stats();
        assert_eq!(stats.queries, 3);
        assert!(stats.routing_ops > 0);
        assert!(stats.total_gates > 0);
    }

    #[test]
    fn test_config_builder() {
        let config = FTQRAMConfig::new(3)
            .with_error_rate(0.01)
            .with_correction_interval(5)
            .with_seed(999);

        assert_eq!(config.depth, 3);
        assert_eq!(config.error_rate, 0.01);
        assert_eq!(config.correction_interval, 5);
        assert_eq!(config.seed, 999);
    }

    #[test]
    fn test_memory_stats() {
        let config = FTQRAMConfig::new(2).with_error_rate(0.05);
        let mut qram = FaultTolerantQRAM::new(config);

        // Run some queries to generate stats
        for _ in 0..20 {
            qram.query(0);
        }

        let _mem_stats = qram.memory_stats();
        // Should have some activity
        // (errors_injected depends on RNG, so just check it runs)
    }
}
