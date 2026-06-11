//! Magic State Budgeting for Fault-Tolerant QRAM
//!
//! SPRINT 51.0 Phase 1: Magic State Budget Core
//!
//! This module provides comprehensive resource estimation for fault-tolerant
//! quantum operations, tracking T-gate consumption, magic state distillation
//! overhead, and factory resource requirements.
//!
//! # Background
//!
//! In fault-tolerant quantum computing:
//! - Clifford gates (H, S, CNOT) are "cheap" - transversal implementation
//! - Non-Clifford gates (T, Toffoli) are "expensive" - require magic states
//! - Magic states must be distilled from noisy physical states
//! - Distillation has significant overhead (15:1 to 225:1 ratios)
//!
//! # QRAM Magic State Cost
//!
//! For bucket-brigade QRAM with N cells:
//! ```text
//! Per Query:
//!   Routers traversed: log₂(N)
//!   Toffoli gates: log₂(N)
//!   Bare T-count: 7 × log₂(N)
//!
//! With 15-to-1 distillation (2 levels for 10⁻¹² error):
//!   Magic states = 7 × log₂(N) × 225
//! ```
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::{MagicStateBudget, DistillationConfig};
//!
//! // Create budget from gate counts
//! let budget = MagicStateBudget::from_gate_counts(8, 0, 0); // 8 Toffolis
//!
//! // Add fault-tolerant overhead
//! let ft_budget = budget.with_distillation(DistillationConfig::for_error_rate(1e-12));
//!
//! println!("Total magic states: {}", ft_budget.total_magic_states);
//! println!("Factory qubits: {}", ft_budget.factory_qubits);
//! ```

use std::fmt;

/// Comprehensive magic state resource budget
///
/// Tracks all aspects of non-Clifford resource consumption including
/// raw gate counts, distillation overhead, and factory requirements.
#[derive(Clone, Debug, Default)]
pub struct MagicStateBudget {
    // === Raw Gate Counts ===
    /// Direct T gates (not from Toffoli decomposition)
    pub direct_t_gates: usize,
    /// Direct T† gates
    pub direct_tdg_gates: usize,
    /// Toffoli gate count
    pub toffoli_count: usize,
    /// Total theoretical T-count (Toffoli×7 + direct)
    pub total_t_count_bare: usize,

    // === T-depth Analysis ===
    /// Maximum T-depth (sequential T gates on critical path)
    pub t_depth: usize,
    /// T-gates that can be parallelized
    pub parallel_t_gates: usize,

    // === Resource Breakdown by Context ===
    /// T-gates from QRAM routing (Fredkin/Toffoli gates)
    pub routing_t_gates: usize,
    /// T-gates from error correction operations
    pub correction_t_gates: usize,
    /// T-gates from data manipulation
    pub data_t_gates: usize,

    // === Fault-Tolerant Configuration ===
    /// Target code distance (0 if not FT)
    pub code_distance: usize,
    /// Physical error rate assumption
    pub physical_error_rate: f64,
    /// Target logical error rate per operation
    pub target_logical_error_rate: f64,

    // === Distillation Overhead ===
    /// Number of distillation levels
    pub distillation_levels: usize,
    /// Noisy magic states consumed per clean state
    pub noisy_states_per_clean: usize,
    /// Total magic states needed (after distillation accounting)
    pub total_magic_states: usize,

    // === Factory Resources ===
    /// Qubits dedicated to magic state factory
    pub factory_qubits: usize,
    /// Cycles per magic state production
    pub cycles_per_magic_state: usize,
    /// Total factory cycles needed
    pub total_factory_cycles: u64,

    // === Cost Metrics ===
    /// Total qubit-cycles (space × time)
    pub qubit_cycles: u64,
    /// Estimated runtime in seconds (at given cycle rate)
    pub estimated_seconds: f64,
}

impl MagicStateBudget {
    /// Create a new empty budget
    pub fn new() -> Self {
        Self::default()
    }

    /// Create budget from raw gate counts
    ///
    /// Uses standard Toffoli decomposition (7 T gates per Toffoli)
    pub fn from_gate_counts(toffoli: usize, t_gates: usize, tdg_gates: usize) -> Self {
        let total_t = toffoli * 7 + t_gates + tdg_gates;

        Self {
            direct_t_gates: t_gates,
            direct_tdg_gates: tdg_gates,
            toffoli_count: toffoli,
            total_t_count_bare: total_t,
            t_depth: total_t,             // Conservative: assume sequential
            routing_t_gates: toffoli * 7, // Assume all Toffolis are routing
            ..Default::default()
        }
    }

    /// Create budget for QRAM routing operations
    ///
    /// # Arguments
    /// * `depth` - QRAM depth (log₂ of cell count)
    /// * `queries` - Number of queries to budget for
    pub fn for_qram_routing(depth: usize, queries: usize) -> Self {
        // Each query traverses `depth` routers, each with 1 Toffoli
        let toffolis_per_query = depth;
        let total_toffolis = toffolis_per_query * queries;
        let t_per_query = toffolis_per_query * 7;

        Self {
            toffoli_count: total_toffolis,
            total_t_count_bare: total_toffolis * 7,
            routing_t_gates: total_toffolis * 7,
            t_depth: t_per_query, // Per-query depth
            parallel_t_gates: if queries > 1 {
                total_toffolis * 7 - t_per_query
            } else {
                0
            },
            ..Default::default()
        }
    }

    /// Add fault-tolerant distillation overhead
    ///
    /// # Arguments
    /// * `config` - Distillation configuration
    pub fn with_distillation(mut self, config: &DistillationConfig) -> Self {
        self.code_distance = config.code_distance;
        self.physical_error_rate = config.physical_error_rate;
        self.target_logical_error_rate = config.target_error_rate;
        self.distillation_levels = config.levels;
        self.noisy_states_per_clean = config.overhead_ratio();

        // Calculate total magic states needed
        self.total_magic_states = self.total_t_count_bare * self.noisy_states_per_clean;

        // Estimate factory resources
        let factory = config.factory_estimate(self.total_t_count_bare);
        self.factory_qubits = factory.qubits;
        self.cycles_per_magic_state = factory.cycles_per_state;
        self.total_factory_cycles = factory.total_cycles;

        // Calculate qubit-cycles
        self.qubit_cycles = self.factory_qubits as u64 * self.total_factory_cycles;

        // Estimate time (assuming 1 MHz cycle rate)
        self.estimated_seconds = self.total_factory_cycles as f64 / 1_000_000.0;

        self
    }

    /// Set routing context (all Toffolis from QRAM routing)
    ///
    /// This labels the T-gates as coming from QRAM routing operations
    pub fn with_routing_context(mut self, routing_toffolis: usize) -> Self {
        self.routing_t_gates = routing_toffolis * 7;
        self
    }

    /// Set correction context (T-gates from error correction)
    pub fn with_correction_context(mut self, correction_t: usize) -> Self {
        self.correction_t_gates = correction_t;
        self
    }

    /// Set data operation context
    pub fn with_data_context(mut self, data_t: usize) -> Self {
        self.data_t_gates = data_t;
        self
    }

    /// Get effective T-count including distillation overhead
    pub fn effective_t_count(&self) -> usize {
        if self.noisy_states_per_clean > 0 {
            self.total_magic_states
        } else {
            self.total_t_count_bare
        }
    }

    /// Check if fault-tolerant overhead is configured
    pub fn is_ft_configured(&self) -> bool {
        self.distillation_levels > 0
    }

    /// Get overhead ratio (effective / bare)
    pub fn overhead_ratio(&self) -> f64 {
        if self.total_t_count_bare > 0 {
            self.effective_t_count() as f64 / self.total_t_count_bare as f64
        } else {
            1.0
        }
    }

    /// Merge another budget into this one (for combining operations)
    pub fn merge(&mut self, other: &MagicStateBudget) {
        self.direct_t_gates += other.direct_t_gates;
        self.direct_tdg_gates += other.direct_tdg_gates;
        self.toffoli_count += other.toffoli_count;
        self.total_t_count_bare += other.total_t_count_bare;
        self.routing_t_gates += other.routing_t_gates;
        self.correction_t_gates += other.correction_t_gates;
        self.data_t_gates += other.data_t_gates;
        self.total_magic_states += other.total_magic_states;
        self.total_factory_cycles += other.total_factory_cycles;
        self.qubit_cycles += other.qubit_cycles;

        // T-depth is max of sequential paths
        self.t_depth = self.t_depth.max(other.t_depth);
    }

    /// Create summary string for display
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("=== Magic State Budget ===\n"));
        s.push_str(&format!("Toffoli gates: {}\n", self.toffoli_count));
        s.push_str(&format!(
            "Direct T/T†: {}/{}\n",
            self.direct_t_gates, self.direct_tdg_gates
        ));
        s.push_str(&format!("Bare T-count: {}\n", self.total_t_count_bare));
        s.push_str(&format!("T-depth: {}\n", self.t_depth));

        if self.is_ft_configured() {
            s.push_str(&format!("\n--- Fault-Tolerant Overhead ---\n"));
            s.push_str(&format!("Code distance: {}\n", self.code_distance));
            s.push_str(&format!(
                "Target error: {:.0e}\n",
                self.target_logical_error_rate
            ));
            s.push_str(&format!(
                "Distillation levels: {}\n",
                self.distillation_levels
            ));
            s.push_str(&format!(
                "Overhead ratio: {}:1\n",
                self.noisy_states_per_clean
            ));
            s.push_str(&format!(
                "Total magic states: {}\n",
                self.total_magic_states
            ));
            s.push_str(&format!("Factory qubits: {}\n", self.factory_qubits));
            s.push_str(&format!("Qubit-cycles: {}\n", self.qubit_cycles));
            s.push_str(&format!("Est. time: {:.2}s\n", self.estimated_seconds));
        }

        s
    }
}

impl fmt::Display for MagicStateBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ft_configured() {
            write!(
                f,
                "MagicStateBudget(T={}, magic={}, factory={}q, {:.1}s)",
                self.total_t_count_bare,
                self.total_magic_states,
                self.factory_qubits,
                self.estimated_seconds
            )
        } else {
            write!(
                f,
                "MagicStateBudget(Toffoli={}, T={}, depth={})",
                self.toffoli_count, self.total_t_count_bare, self.t_depth
            )
        }
    }
}

/// Configuration for magic state distillation
#[derive(Clone, Debug)]
pub struct DistillationConfig {
    /// Target code distance
    pub code_distance: usize,
    /// Physical qubit error rate
    pub physical_error_rate: f64,
    /// Target logical error rate per T-gate
    pub target_error_rate: f64,
    /// Number of distillation levels
    pub levels: usize,
    /// Protocol used at each level
    pub protocol: DistillationProtocolType,
}

/// Type of distillation protocol
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillationProtocolType {
    /// 15-to-1 Bravyi-Kitaev protocol
    FifteenToOne,
    /// 10-to-2 improved protocol
    TenToTwo,
    /// Reed-Muller based (very low error)
    ReedMuller,
}

impl DistillationConfig {
    /// Create config for target error rate with default physical rate
    pub fn for_error_rate(target: f64) -> Self {
        Self::new(1e-3, target)
    }

    /// Create config with physical and target error rates
    pub fn new(physical: f64, target: f64) -> Self {
        // Calculate required distillation levels
        let levels = Self::calculate_levels(physical, target);
        let code_distance = Self::calculate_code_distance(target);

        Self {
            code_distance,
            physical_error_rate: physical,
            target_error_rate: target,
            levels,
            protocol: DistillationProtocolType::FifteenToOne,
        }
    }

    /// Create config with specific number of levels
    pub fn with_levels(physical: f64, target: f64, levels: usize) -> Self {
        let code_distance = Self::calculate_code_distance(target);

        Self {
            code_distance,
            physical_error_rate: physical,
            target_error_rate: target,
            levels,
            protocol: DistillationProtocolType::FifteenToOne,
        }
    }

    /// Calculate distillation levels needed
    ///
    /// 15-to-1 protocol: p_out ≈ 35 × p_in³
    fn calculate_levels(physical: f64, target: f64) -> usize {
        if physical <= target {
            return 0;
        }

        let mut current_error = physical;
        let mut levels = 0;

        // 15-to-1: output error ≈ 35 * input³
        while current_error > target && levels < 5 {
            current_error = 35.0 * current_error.powi(3);
            levels += 1;
        }

        levels
    }

    /// Calculate code distance for target error rate
    ///
    /// Approximate: d ≈ 2 * log₁₀(1/target) + 3
    fn calculate_code_distance(target: f64) -> usize {
        let log_inv = (-target.log10()).ceil() as usize;
        let d = 2 * log_inv + 3;
        // Round up to odd number (surface code requirement)
        if d.is_multiple_of(2) { d + 1 } else { d }
    }

    /// Get overhead ratio (noisy states per clean state)
    pub fn overhead_ratio(&self) -> usize {
        match self.protocol {
            DistillationProtocolType::FifteenToOne => {
                // 15^levels noisy states per clean state
                15usize.pow(self.levels as u32)
            }
            DistillationProtocolType::TenToTwo => {
                // 5^levels (produces 2 states, so 10/2 = 5 effective)
                5usize.pow(self.levels as u32)
            }
            DistillationProtocolType::ReedMuller => {
                // More efficient at high levels
                8usize.pow(self.levels as u32)
            }
        }
    }

    /// Estimate factory resources for N clean magic states
    pub fn factory_estimate(&self, n_clean_states: usize) -> FactoryEstimate {
        let overhead = self.overhead_ratio();
        let total_noisy = n_clean_states * overhead;

        // Factory qubit estimate based on protocol and levels
        let base_qubits = match self.protocol {
            DistillationProtocolType::FifteenToOne => 20, // 15 data + ancillas
            DistillationProtocolType::TenToTwo => 15,
            DistillationProtocolType::ReedMuller => 30,
        };

        // Scale by levels (each level needs its own factory stage)
        let factory_qubits = base_qubits * self.levels * 10; // Rough scaling

        // Cycles per magic state (distillation rounds)
        let cycles_per_state = 15 * self.levels; // ~15 cycles per level

        // Total cycles
        let total_cycles = (n_clean_states as u64) * (cycles_per_state as u64);

        FactoryEstimate {
            qubits: factory_qubits,
            cycles_per_state: cycles_per_state,
            total_cycles,
            noisy_states_consumed: total_noisy,
        }
    }

    /// Get error rate achieved after distillation
    pub fn achieved_error_rate(&self) -> f64 {
        let mut error = self.physical_error_rate;
        for _ in 0..self.levels {
            error = 35.0 * error.powi(3);
        }
        error
    }
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self::for_error_rate(1e-10)
    }
}

impl fmt::Display for DistillationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DistillationConfig(d={}, levels={}, {:?}, target={:.0e})",
            self.code_distance, self.levels, self.protocol, self.target_error_rate
        )
    }
}

/// Factory resource estimate
#[derive(Clone, Debug)]
pub struct FactoryEstimate {
    /// Qubits in magic state factory
    pub qubits: usize,
    /// Cycles per clean magic state
    pub cycles_per_state: usize,
    /// Total cycles needed
    pub total_cycles: u64,
    /// Noisy states consumed
    pub noisy_states_consumed: usize,
}

impl fmt::Display for FactoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Factory({}q, {} cycles/state, {} total)",
            self.qubits, self.cycles_per_state, self.total_cycles
        )
    }
}

/// Quick estimate functions for common scenarios
pub mod estimates {
    use super::*;

    /// Estimate for QRAM query at given depth and error rate
    pub fn qram_query(depth: usize, target_error: f64) -> MagicStateBudget {
        MagicStateBudget::for_qram_routing(depth, 1)
            .with_distillation(&DistillationConfig::for_error_rate(target_error))
    }

    /// Estimate for N QRAM queries
    pub fn qram_queries(depth: usize, queries: usize, target_error: f64) -> MagicStateBudget {
        MagicStateBudget::for_qram_routing(depth, queries)
            .with_distillation(&DistillationConfig::for_error_rate(target_error))
    }

    /// Compare costs across different error targets
    pub fn compare_error_targets(
        toffoli_count: usize,
        targets: &[f64],
    ) -> Vec<(f64, MagicStateBudget)> {
        let base = MagicStateBudget::from_gate_counts(toffoli_count, 0, 0);
        targets
            .iter()
            .map(|&t| {
                let config = DistillationConfig::for_error_rate(t);
                (t, base.clone().with_distillation(&config))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_gate_counts() {
        let budget = MagicStateBudget::from_gate_counts(10, 5, 3);
        assert_eq!(budget.toffoli_count, 10);
        assert_eq!(budget.direct_t_gates, 5);
        assert_eq!(budget.direct_tdg_gates, 3);
        assert_eq!(budget.total_t_count_bare, 10 * 7 + 5 + 3); // 78
    }

    #[test]
    fn test_qram_routing_budget() {
        let budget = MagicStateBudget::for_qram_routing(8, 1);
        assert_eq!(budget.toffoli_count, 8);
        assert_eq!(budget.total_t_count_bare, 56); // 8 * 7
        assert_eq!(budget.routing_t_gates, 56);
    }

    #[test]
    fn test_distillation_levels() {
        // 10^-3 physical, 10^-10 target should need ~2 levels
        let config = DistillationConfig::new(1e-3, 1e-10);
        assert!(config.levels >= 1 && config.levels <= 3);
    }

    #[test]
    fn test_overhead_ratio() {
        let config = DistillationConfig::with_levels(1e-3, 1e-10, 2);
        assert_eq!(config.overhead_ratio(), 225); // 15^2
    }

    #[test]
    fn test_with_distillation() {
        let budget = MagicStateBudget::from_gate_counts(8, 0, 0);
        let config = DistillationConfig::with_levels(1e-3, 1e-10, 2);
        let ft_budget = budget.with_distillation(&config);

        assert_eq!(ft_budget.total_t_count_bare, 56);
        assert_eq!(ft_budget.noisy_states_per_clean, 225);
        assert_eq!(ft_budget.total_magic_states, 56 * 225);
        assert!(ft_budget.factory_qubits > 0);
    }

    #[test]
    fn test_code_distance_calculation() {
        // 10^-10 target should give ~d=23 or so
        let d = DistillationConfig::calculate_code_distance(1e-10);
        assert!(d >= 21 && d <= 25);
        assert!(d % 2 == 1); // Must be odd
    }

    #[test]
    fn test_achieved_error_rate() {
        let config = DistillationConfig::with_levels(1e-3, 1e-15, 2);
        let achieved = config.achieved_error_rate();
        // 35 * (35 * (10^-3)^3)^3 ≈ very small
        assert!(achieved < 1e-10);
    }

    #[test]
    fn test_merge() {
        let mut budget1 = MagicStateBudget::from_gate_counts(5, 0, 0);
        let budget2 = MagicStateBudget::from_gate_counts(3, 2, 1);
        budget1.merge(&budget2);

        assert_eq!(budget1.toffoli_count, 8);
        assert_eq!(budget1.direct_t_gates, 2);
        assert_eq!(budget1.total_t_count_bare, 5 * 7 + 3 * 7 + 2 + 1);
    }

    #[test]
    fn test_display() {
        let budget = MagicStateBudget::from_gate_counts(8, 0, 0);
        let s = format!("{}", budget);
        assert!(s.contains("Toffoli=8"));
        assert!(s.contains("T=56"));
    }

    #[test]
    fn test_estimates_module() {
        let budget = estimates::qram_query(8, 1e-10);
        assert!(budget.is_ft_configured());
        assert!(budget.total_magic_states > budget.total_t_count_bare);
    }

    #[test]
    fn test_compare_error_targets() {
        let comparisons = estimates::compare_error_targets(10, &[1e-6, 1e-10, 1e-15]);
        assert_eq!(comparisons.len(), 3);

        // Lower error target should have more overhead
        let (_, budget_6) = &comparisons[0];
        let (_, budget_15) = &comparisons[2];
        assert!(budget_15.total_magic_states > budget_6.total_magic_states);
    }
}
