//! Hardware-Aware Magic State Budget
//!
//! SPRINT 52.0 Phase 5: Hardware-Aware Budgeting
//!
//! Extends the abstract MagicStateBudget with hardware-specific constraints:
//! - SWAP overhead from routing on restricted topologies
//! - Physical qubit requirements with overhead
//! - Fidelity estimates considering gate errors
//! - Factory placement information
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::MagicStateBudget;
//! use engine::hardware::{HardwareAwareBudget, profiles};
//!
//! let budget = MagicStateBudget::for_qram_routing(8, 100);
//! let profile = profiles::ibm_heron();
//!
//! let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 2);
//! println!("Physical qubits: {}", hw_budget.physical_qubits);
//! println!("SWAP overhead: {}", hw_budget.swap_overhead);
//! ```

use std::fmt;

use crate::qram::MagicStateBudget;

use super::factory_layout::{
    FactoryFootprint, FactoryLayoutOptimizer, FactoryPlacement, LayoutResources,
};
use super::profile::HardwareProfile;

/// Hardware-aware magic state budget
///
/// Extends the abstract MagicStateBudget with physical device constraints
#[derive(Clone, Debug)]
pub struct HardwareAwareBudget {
    /// Base abstract budget
    pub base: MagicStateBudget,
    /// Hardware profile name
    pub profile_name: String,
    /// Hardware vendor
    pub vendor: String,

    // === Physical Resources ===
    /// Physical qubits needed (including factory overhead)
    pub physical_qubits: usize,
    /// Device qubit count
    pub device_qubits: usize,
    /// Device utilization (0-1)
    pub device_utilization: f64,

    // === SWAP Overhead ===
    /// Total SWAP gates needed for factory operations
    pub swap_gates: usize,
    /// Additional 2Q gates from SWAP decomposition
    pub swap_2q_overhead: usize,
    /// Routing cycle overhead
    pub routing_cycles: usize,

    // === Adjusted Metrics ===
    /// Adjusted cycle count (base + routing overhead)
    pub adjusted_cycles: u64,
    /// Adjusted time estimate (seconds)
    pub adjusted_time_seconds: f64,
    /// Cycle time scaling factor vs abstract
    pub time_scaling_factor: f64,

    // === Fidelity Estimates ===
    /// Estimated output fidelity
    pub estimated_fidelity: f64,
    /// Dominant error source
    pub bottleneck: ErrorSource,
    /// Fidelity breakdown by source
    pub fidelity_breakdown: FidelityBreakdown,

    // === Layout Information ===
    /// Number of factories placed
    pub factory_count: usize,
    /// Factory placements (if available)
    pub placements: Vec<FactoryPlacement>,
    /// Layout resources
    pub layout_resources: Option<LayoutResources>,
}

/// Dominant error source
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorSource {
    /// Two-qubit gate errors
    TwoQubitGates,
    /// SWAP routing overhead
    SwapOverhead,
    /// Decoherence during circuit
    Decoherence,
    /// Measurement errors
    Measurement,
    /// Single-qubit gate errors
    SingleQubitGates,
}

impl fmt::Display for ErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TwoQubitGates => write!(f, "2Q Gates"),
            Self::SwapOverhead => write!(f, "SWAP Routing"),
            Self::Decoherence => write!(f, "Decoherence"),
            Self::Measurement => write!(f, "Measurement"),
            Self::SingleQubitGates => write!(f, "1Q Gates"),
        }
    }
}

/// Fidelity breakdown by error source
#[derive(Clone, Debug, Default)]
pub struct FidelityBreakdown {
    /// Error from single-qubit gates
    pub single_qubit_error: f64,
    /// Error from two-qubit gates
    pub two_qubit_error: f64,
    /// Error from SWAP routing
    pub swap_error: f64,
    /// Error from decoherence
    pub decoherence_error: f64,
    /// Error from measurement
    pub measurement_error: f64,
}

impl FidelityBreakdown {
    /// Total error (multiplicative model, capped at 0.9999)
    ///
    /// Uses F_total = F_1q * F_2q * F_swap * F_dec * F_meas
    /// where F_i = 1 - error_i (capped to avoid numerical issues)
    pub fn total_error(&self) -> f64 {
        // Cap individual errors to avoid numerical issues
        let cap = |e: f64| e.min(0.999);

        // Multiplicative fidelity model
        let total_fidelity = (1.0 - cap(self.single_qubit_error))
            * (1.0 - cap(self.two_qubit_error))
            * (1.0 - cap(self.swap_error))
            * (1.0 - cap(self.decoherence_error))
            * (1.0 - cap(self.measurement_error));

        // Return error, capped to ensure fidelity stays positive
        (1.0 - total_fidelity).min(0.9999)
    }

    /// Dominant error source
    pub fn dominant(&self) -> ErrorSource {
        let errors = [
            (self.single_qubit_error, ErrorSource::SingleQubitGates),
            (self.two_qubit_error, ErrorSource::TwoQubitGates),
            (self.swap_error, ErrorSource::SwapOverhead),
            (self.decoherence_error, ErrorSource::Decoherence),
            (self.measurement_error, ErrorSource::Measurement),
        ];

        errors
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|&(_, src)| src)
            .unwrap_or(ErrorSource::TwoQubitGates)
    }
}

impl HardwareAwareBudget {
    /// Create hardware-aware budget from abstract budget
    ///
    /// # Arguments
    /// * `budget` - Abstract magic state budget
    /// * `profile` - Hardware profile
    /// * `factory_count` - Number of factories to use
    pub fn from_budget(
        budget: &MagicStateBudget,
        profile: &HardwareProfile,
        factory_count: usize,
    ) -> Self {
        let footprint = FactoryFootprint::fifteen_to_one();
        Self::from_budget_with_footprint(budget, profile, &footprint, factory_count)
    }

    /// Create hardware-aware budget with custom factory footprint
    pub fn from_budget_with_footprint(
        budget: &MagicStateBudget,
        profile: &HardwareProfile,
        footprint: &FactoryFootprint,
        factory_count: usize,
    ) -> Self {
        // Place factories on device
        let optimizer = FactoryLayoutOptimizer::new(profile);
        let placements = optimizer.place_factories(footprint, factory_count);
        let layout_resources = Some(optimizer.total_resources(&placements));

        // Calculate SWAP overhead
        let total_swaps: usize = placements.iter().map(|p| p.swap_overhead).sum();
        let swap_2q_overhead = total_swaps * profile.swap_cost;
        let routing_cycles: usize = placements.iter().map(|p| p.routing_cycles).sum();

        // Physical qubits
        let physical_qubits: usize = placements.iter().map(|p| p.qubit_mapping.len()).sum();
        let device_utilization = physical_qubits as f64 / profile.qubit_count as f64;

        // Adjusted cycles
        let base_cycles = budget.total_factory_cycles;
        let adjusted_cycles = base_cycles + routing_cycles as u64;

        // Time scaling
        let abstract_cycle_ns = 1000.0; // 1µs abstract cycle
        let device_cycle_ns = profile.t2q_time_ns;
        let time_scaling = device_cycle_ns / abstract_cycle_ns;
        let adjusted_time = adjusted_cycles as f64 * device_cycle_ns / 1e9;

        // Fidelity estimation
        let fidelity_breakdown = Self::estimate_fidelity_breakdown(
            budget,
            profile,
            swap_2q_overhead,
            adjusted_time * 1e9, // Convert to ns
        );
        let estimated_fidelity = 1.0 - fidelity_breakdown.total_error();
        let bottleneck = fidelity_breakdown.dominant();

        Self {
            base: budget.clone(),
            profile_name: profile.name.clone(),
            vendor: profile.vendor.clone(),
            physical_qubits,
            device_qubits: profile.qubit_count,
            device_utilization,
            swap_gates: total_swaps,
            swap_2q_overhead,
            routing_cycles,
            adjusted_cycles,
            adjusted_time_seconds: adjusted_time,
            time_scaling_factor: time_scaling,
            estimated_fidelity,
            bottleneck,
            fidelity_breakdown,
            factory_count: placements.len(),
            placements,
            layout_resources,
        }
    }

    /// Estimate fidelity breakdown
    ///
    /// Uses numerically stable calculation for large gate counts:
    /// F^n = e^(n * ln(F)), error = 1 - F^n
    fn estimate_fidelity_breakdown(
        budget: &MagicStateBudget,
        profile: &HardwareProfile,
        swap_2q_count: usize,
        time_ns: f64,
    ) -> FidelityBreakdown {
        // Helper for stable fidelity exponentiation
        // Returns error = 1 - F^n, handling edge cases
        let compute_error = |fidelity: f64, n: usize| -> f64 {
            if n == 0 {
                return 0.0;
            }
            if fidelity >= 1.0 {
                return 0.0;
            }
            if fidelity <= 0.0 {
                return 1.0;
            }
            // Use log for numerical stability: F^n = e^(n * ln(F))
            let log_fidelity = fidelity.ln();
            let combined_fidelity = (n as f64 * log_fidelity).exp();
            1.0 - combined_fidelity
        };

        // Estimate gate counts from T-gates
        // Each T-gate in distillation involves multiple 1Q and 2Q gates
        let t_gates = budget.total_t_count_bare;
        let estimated_1q = t_gates * 3; // Rough estimate
        let estimated_2q = t_gates * 2; // Rough estimate

        let single_qubit_error = compute_error(profile.t1q_fidelity, estimated_1q);
        let two_qubit_error = compute_error(profile.t2q_fidelity, estimated_2q);
        let swap_error = compute_error(profile.t2q_fidelity, swap_2q_count);

        let t2_ns = profile.t2_us * 1000.0;
        let decoherence_error = if t2_ns > 0.0 {
            1.0 - (-time_ns / t2_ns).exp()
        } else {
            0.0
        };

        // Measurement error (per magic state produced)
        let measurements = budget.total_magic_states.max(1);
        let measurement_error = compute_error(profile.measure_fidelity, measurements);

        FidelityBreakdown {
            single_qubit_error,
            two_qubit_error,
            swap_error,
            decoherence_error,
            measurement_error,
        }
    }

    /// Check if the budget is feasible on the device
    pub fn is_feasible(&self) -> bool {
        self.factory_count > 0 && self.device_utilization <= 1.0 && self.estimated_fidelity > 0.5
    }

    /// Get overhead ratio compared to abstract budget
    pub fn overhead_ratio(&self) -> f64 {
        if self.base.total_factory_cycles > 0 {
            self.adjusted_cycles as f64 / self.base.total_factory_cycles as f64
        } else {
            1.0
        }
    }

    /// Compare to another hardware-aware budget
    pub fn compare_to(&self, other: &HardwareAwareBudget) -> BudgetComparison {
        BudgetComparison {
            profile_a: self.profile_name.clone(),
            profile_b: other.profile_name.clone(),
            qubits_ratio: self.physical_qubits as f64 / other.physical_qubits.max(1) as f64,
            time_ratio: self.adjusted_time_seconds / other.adjusted_time_seconds.max(1e-12),
            fidelity_ratio: self.estimated_fidelity / other.estimated_fidelity.max(1e-12),
            swap_ratio: self.swap_gates as f64 / other.swap_gates.max(1) as f64,
        }
    }
}

impl fmt::Display for HardwareAwareBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Hardware-Aware Magic State Budget ===")?;
        writeln!(f, "Device: {} ({})", self.profile_name, self.vendor)?;
        writeln!(f)?;
        writeln!(f, "--- Physical Resources ---")?;
        writeln!(
            f,
            "Physical qubits: {} / {} ({:.1}% utilization)",
            self.physical_qubits,
            self.device_qubits,
            self.device_utilization * 100.0
        )?;
        writeln!(f, "Factories: {}", self.factory_count)?;
        writeln!(f)?;
        writeln!(f, "--- SWAP Overhead ---")?;
        writeln!(f, "SWAP gates: {}", self.swap_gates)?;
        writeln!(f, "2Q overhead: {} gates", self.swap_2q_overhead)?;
        writeln!(f, "Routing cycles: {}", self.routing_cycles)?;
        writeln!(f)?;
        writeln!(f, "--- Timing ---")?;
        writeln!(f, "Adjusted cycles: {}", self.adjusted_cycles)?;
        writeln!(f, "Adjusted time: {:.3}s", self.adjusted_time_seconds)?;
        writeln!(f, "Scaling factor: {:.2}x", self.time_scaling_factor)?;
        writeln!(f)?;
        writeln!(f, "--- Fidelity ---")?;
        writeln!(f, "Estimated fidelity: {:.4}", self.estimated_fidelity)?;
        writeln!(f, "Bottleneck: {}", self.bottleneck)?;
        Ok(())
    }
}

/// Comparison between two hardware-aware budgets
#[derive(Clone, Debug)]
pub struct BudgetComparison {
    pub profile_a: String,
    pub profile_b: String,
    pub qubits_ratio: f64,
    pub time_ratio: f64,
    pub fidelity_ratio: f64,
    pub swap_ratio: f64,
}

impl fmt::Display for BudgetComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} vs {}: qubits {:.2}x, time {:.2}x, fidelity {:.2}x, SWAPs {:.2}x",
            self.profile_a,
            self.profile_b,
            self.qubits_ratio,
            self.time_ratio,
            self.fidelity_ratio,
            self.swap_ratio
        )
    }
}

/// Compare abstract budget across multiple hardware profiles
pub fn compare_on_hardware(
    budget: &MagicStateBudget,
    profiles: &[&HardwareProfile],
    factory_count: usize,
) -> Vec<HardwareAwareBudget> {
    profiles
        .iter()
        .map(|p| HardwareAwareBudget::from_budget(budget, p, factory_count))
        .collect()
}

/// Format hardware comparison as table
pub fn format_hardware_comparison(budgets: &[HardwareAwareBudget]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:>15} {:>10} {:>10} {:>10} {:>10} {:>12}\n",
        "Device", "Qubits", "Factories", "SWAPs", "Time(s)", "Fidelity"
    ));
    s.push_str(&format!("{}\n", "-".repeat(75)));

    for b in budgets {
        s.push_str(&format!(
            "{:>15} {:>10} {:>10} {:>10} {:>10.3} {:>12.4}\n",
            b.profile_name,
            b.physical_qubits,
            b.factory_count,
            b.swap_gates,
            b.adjusted_time_seconds,
            b.estimated_fidelity
        ));
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::profile::profiles;
    use crate::qram::DistillationConfig;

    #[test]
    fn test_hardware_aware_budget() {
        let budget = MagicStateBudget::for_qram_routing(8, 10)
            .with_distillation(&DistillationConfig::for_error_rate(1e-10));

        let profile = profiles::ibm_heron();
        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 2);

        assert!(hw_budget.physical_qubits > 0);
        assert!(hw_budget.factory_count > 0);
        assert!(hw_budget.estimated_fidelity > 0.0);
    }

    #[test]
    fn test_ion_trap_low_swap() {
        let budget = MagicStateBudget::for_qram_routing(4, 1);
        let profile = profiles::ionq_aria();
        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 1);

        // All-to-all connectivity should have zero SWAP overhead
        assert_eq!(hw_budget.swap_gates, 0);
    }

    #[test]
    fn test_fidelity_breakdown() {
        let budget = MagicStateBudget::for_qram_routing(4, 1);
        let profile = profiles::ibm_falcon();
        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 1);

        assert!(hw_budget.fidelity_breakdown.total_error() > 0.0);
        assert!(hw_budget.fidelity_breakdown.total_error() < 1.0);
    }

    #[test]
    fn test_compare_hardware() {
        let budget = MagicStateBudget::for_qram_routing(6, 10);

        let profiles = vec![profiles::ibm_falcon(), profiles::ionq_aria()];

        let budgets = compare_on_hardware(&budget, &profiles.iter().collect::<Vec<_>>(), 1);
        assert_eq!(budgets.len(), 2);
    }

    #[test]
    fn test_feasibility() {
        let budget = MagicStateBudget::for_qram_routing(4, 1);
        let profile = profiles::ibm_heron();
        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 1);

        assert!(hw_budget.is_feasible());
    }

    #[test]
    fn test_display() {
        let budget = MagicStateBudget::for_qram_routing(4, 1);
        let profile = profiles::ibm_falcon();
        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 1);

        let s = format!("{}", hw_budget);
        assert!(s.contains("Hardware-Aware"));
        assert!(s.contains("Falcon"));
    }
}
