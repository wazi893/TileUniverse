//! Resource Estimation Dashboard
//!
//! SPRINT 56.0: Comprehensive Resource Analysis
//!
//! Combines all circuit metrics into actionable planning data:
//! - Logical circuit metrics (gates, depth, T-count)
//! - Fault-tolerant overhead (physical qubits, code distance)
//! - Timing projections (NISQ and fault-tolerant)
//! - Report generation (JSON, Markdown)
//!
//! # Example
//!
//! ```ignore
//! use engine::transpile::{ResourceEstimator, FaultTolerantConfig};
//! use logic_fabric_core::quantum::QGate;
//!
//! let circuit = vec![QGate::H(0), QGate::T(0), QGate::CNot(0, 1)];
//!
//! let estimator = ResourceEstimator::new()
//!     .with_ft_config(FaultTolerantConfig::for_error_rate(1e-12));
//!
//! let estimate = estimator.estimate(&circuit, "my_circuit");
//! println!("{}", estimate);
//! println!("{}", estimate.to_markdown());
//! ```

use std::collections::HashMap;
use std::fmt;

use super::analysis::analyze_abstract_circuit;
use super::t_count::{TGateAnalysis, analyze_t_gates};
use super::t_depth::{TDepthAnalysis, analyze_t_depth};
use crate::hardware::HardwareProfile;
use crate::qram::DistillationConfig;
use logic_fabric_core::quantum::QGate;

/// Format a number with thousands separators
fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Comprehensive resource estimate
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// Circuit name
    pub name: String,

    // === Logical Circuit Metrics ===
    /// Number of logical qubits
    pub logical_qubits: usize,
    /// Total gate count
    pub total_gates: usize,
    /// Circuit depth
    pub depth: usize,
    /// Single-qubit gate count
    pub single_qubit_gates: usize,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
    /// Three-qubit gate count (Toffoli, CCZ)
    pub three_qubit_gates: usize,
    /// Gate counts by type
    pub gate_breakdown: HashMap<String, usize>,

    // === T-Gate Analysis ===
    /// T-gate analysis
    pub t_analysis: TGateAnalysis,
    /// T-depth analysis
    pub t_depth_analysis: TDepthAnalysis,

    // === Fault-Tolerant Resources ===
    /// Fault-tolerant configuration used
    pub ft_config: Option<FaultTolerantConfig>,
    /// Physical resource estimate
    pub physical_estimate: Option<PhysicalResourceEstimate>,

    // === Timing Estimates ===
    /// NISQ timing estimate (if hardware profile provided)
    pub nisq_timing: Option<NISQTiming>,
    /// Fault-tolerant timing estimate
    pub ft_timing: Option<FaultTolerantTiming>,
}

/// Configuration for fault-tolerant resource estimation
#[derive(Clone, Debug)]
pub struct FaultTolerantConfig {
    /// Target logical error rate
    pub target_error_rate: f64,
    /// Physical qubit error rate
    pub physical_error_rate: f64,
    /// Error correction code (e.g., "surface", "steane")
    pub code_type: String,
    /// Magic state distillation configuration
    pub distillation: DistillationConfig,
    /// Physical gate time in nanoseconds
    pub physical_gate_time_ns: f64,
    /// Surface code cycle time in nanoseconds
    pub code_cycle_time_ns: f64,
}

impl FaultTolerantConfig {
    /// Create config for target error rate with surface code
    pub fn for_error_rate(target_error: f64) -> Self {
        Self {
            target_error_rate: target_error,
            physical_error_rate: 1e-3, // Typical superconducting
            code_type: "surface".to_string(),
            distillation: DistillationConfig::for_error_rate(target_error),
            physical_gate_time_ns: 50.0, // ~50ns for superconducting
            code_cycle_time_ns: 1000.0,  // ~1µs per code cycle
        }
    }

    /// Create config for superconducting qubits
    pub fn superconducting() -> Self {
        Self::for_error_rate(1e-12)
    }

    /// Create config for trapped ion qubits
    pub fn trapped_ion() -> Self {
        Self {
            target_error_rate: 1e-12,
            physical_error_rate: 1e-4, // Better than superconducting
            code_type: "surface".to_string(),
            distillation: DistillationConfig::for_error_rate(1e-12),
            physical_gate_time_ns: 10_000.0, // ~10µs for trapped ion
            code_cycle_time_ns: 100_000.0,   // ~100µs per code cycle
        }
    }

    /// Calculate required code distance for target error rate
    pub fn required_code_distance(&self, logical_ops: usize) -> usize {
        // Surface code: p_L ≈ 0.1 * (p/p_th)^((d+1)/2)
        // For p_th ≈ 1%, we need d such that error per operation is < target/ops

        let error_per_op = self.target_error_rate / logical_ops.max(1) as f64;
        let p = self.physical_error_rate;
        let p_th = 0.01; // Threshold ~1%

        if p >= p_th {
            return 99; // Error rate too high
        }

        // Solve for d: 0.1 * (p/p_th)^((d+1)/2) < error_per_op
        let ratio = p / p_th;
        let log_target = (error_per_op / 0.1).ln();
        let log_ratio = ratio.ln();

        let d_float = 2.0 * log_target / log_ratio - 1.0;
        let d = (d_float.ceil() as usize).max(3);

        // Round up to odd number
        if d % 2 == 0 { d + 1 } else { d }
    }
}

impl Default for FaultTolerantConfig {
    fn default() -> Self {
        Self::for_error_rate(1e-12)
    }
}

/// Physical resource estimate for fault-tolerant execution
#[derive(Clone, Debug)]
pub struct PhysicalResourceEstimate {
    /// Required code distance
    pub code_distance: usize,
    /// Physical qubits per logical qubit
    pub physical_per_logical: usize,
    /// Total physical qubits for data
    pub data_qubits: usize,
    /// Physical qubits for magic state factories
    pub factory_qubits: usize,
    /// Number of magic state factories
    pub num_factories: usize,
    /// Total physical qubits
    pub total_physical_qubits: usize,
    /// Magic states required
    pub magic_states_required: usize,
    /// Factory cycles needed
    pub factory_cycles: usize,
}

impl PhysicalResourceEstimate {
    /// Calculate from circuit metrics and FT config
    pub fn calculate(
        logical_qubits: usize,
        t_count: usize,
        t_depth: usize,
        config: &FaultTolerantConfig,
    ) -> Self {
        // Estimate total logical operations
        let logical_ops = t_count + logical_qubits * 100; // Rough estimate

        // Calculate code distance
        let code_distance = config.required_code_distance(logical_ops);

        // Physical qubits per logical qubit (surface code: 2*d^2)
        let physical_per_logical = 2 * code_distance * code_distance;

        // Data qubits
        let data_qubits = logical_qubits * physical_per_logical;

        // Magic state factory sizing
        // Each factory produces ~1 state per code cycle
        // We want enough factories to not bottleneck on T-depth
        let num_factories = if t_depth > 0 {
            (t_count / t_depth).max(1)
        } else {
            1
        };

        // Each factory uses ~15*d^2 physical qubits (15-to-1 distillation)
        let factory_qubits = num_factories * 15 * code_distance * code_distance;

        // Magic states needed (with distillation overhead)
        let distillation_overhead = config.distillation.overhead_ratio();
        let magic_states_required = t_count * distillation_overhead;

        // Factory cycles = ceil(magic_states / factories)
        let factory_cycles = if num_factories > 0 {
            (magic_states_required + num_factories - 1) / num_factories
        } else {
            magic_states_required
        };

        let total_physical_qubits = data_qubits + factory_qubits;

        Self {
            code_distance,
            physical_per_logical,
            data_qubits,
            factory_qubits,
            num_factories,
            total_physical_qubits,
            magic_states_required,
            factory_cycles,
        }
    }
}

/// NISQ timing estimate
#[derive(Clone, Debug)]
pub struct NISQTiming {
    /// Hardware target name
    pub hardware: String,
    /// Execution time in nanoseconds
    pub duration_ns: f64,
    /// Estimated success probability
    pub success_probability: f64,
    /// Expected shots for 99% confidence
    pub shots_for_confidence: usize,

    // SPRINT 57.0: Error mitigation tracking
    /// Mitigation strategy (if any)
    pub mitigation_strategy: Option<crate::mitigation::MitigationStrategy>,
    /// Mitigated success probability (after error mitigation)
    pub mitigated_success_probability: Option<f64>,
}

/// Fault-tolerant timing estimate
#[derive(Clone, Debug)]
pub struct FaultTolerantTiming {
    /// Time per code cycle (ns)
    pub cycle_time_ns: f64,
    /// Total cycles for T-gates
    pub t_gate_cycles: usize,
    /// Total cycles for Clifford gates
    pub clifford_cycles: usize,
    /// Factory cycles (parallel with computation)
    pub factory_cycles: usize,
    /// Total execution time (ns)
    pub total_time_ns: f64,
    /// Dominant time contributor
    pub bottleneck: String,
}

impl FaultTolerantTiming {
    /// Calculate from circuit metrics and FT config
    pub fn calculate(
        _t_count: usize,
        t_depth: usize,
        clifford_count: usize,
        physical_estimate: &PhysicalResourceEstimate,
        config: &FaultTolerantConfig,
    ) -> Self {
        let cycle_time_ns = config.code_cycle_time_ns;

        // T-gates: each T-gate takes ~d cycles for lattice surgery
        let t_gate_cycles = t_depth * physical_estimate.code_distance;

        // Clifford gates: ~d cycles each in lattice surgery
        let clifford_cycles = (clifford_count / 10).max(1) * physical_estimate.code_distance;

        // Factory runs in parallel, but may still bottleneck
        let factory_cycles = physical_estimate.factory_cycles;

        // Total time is max of computation and factory
        let compute_cycles = t_gate_cycles + clifford_cycles;
        let total_cycles = compute_cycles.max(factory_cycles);
        let total_time_ns = total_cycles as f64 * cycle_time_ns;

        let bottleneck = if factory_cycles > compute_cycles {
            "Magic state factory".to_string()
        } else if t_gate_cycles > clifford_cycles {
            "T-gate depth".to_string()
        } else {
            "Clifford gates".to_string()
        };

        Self {
            cycle_time_ns,
            t_gate_cycles,
            clifford_cycles,
            factory_cycles,
            total_time_ns,
            bottleneck,
        }
    }
}

/// Resource estimator builder
#[derive(Clone, Debug, Default)]
pub struct ResourceEstimator {
    /// Fault-tolerant configuration
    ft_config: Option<FaultTolerantConfig>,
    /// NISQ hardware profile
    nisq_profile: Option<HardwareProfile>,
    /// Error mitigation strategy (SPRINT 57.0)
    mitigation_strategy: Option<crate::mitigation::MitigationStrategy>,
}

impl ResourceEstimator {
    /// Create a new estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add fault-tolerant configuration
    pub fn with_ft_config(mut self, config: FaultTolerantConfig) -> Self {
        self.ft_config = Some(config);
        self
    }

    /// Add NISQ hardware profile
    pub fn with_nisq_profile(mut self, profile: HardwareProfile) -> Self {
        self.nisq_profile = Some(profile);
        self
    }

    /// Add error mitigation strategy (SPRINT 57.0)
    pub fn with_mitigation(mut self, strategy: crate::mitigation::MitigationStrategy) -> Self {
        self.mitigation_strategy = Some(strategy);
        self
    }

    /// Estimate resources for a circuit
    pub fn estimate(&self, circuit: &[QGate], name: &str) -> ResourceEstimate {
        // Basic circuit analysis
        let analysis = analyze_abstract_circuit(circuit, name);
        let t_analysis = analyze_t_gates(circuit);
        let t_depth_analysis = analyze_t_depth(circuit);

        // Count gate types
        let mut three_qubit_gates = 0;
        for gate in circuit {
            if matches!(gate, QGate::Toffoli(_, _, _) | QGate::CCZ(_, _, _)) {
                three_qubit_gates += 1;
            }
        }

        // Fault-tolerant resources
        let physical_estimate = self.ft_config.as_ref().map(|config| {
            PhysicalResourceEstimate::calculate(
                analysis.num_qubits,
                t_analysis.t_count,
                t_analysis.t_depth,
                config,
            )
        });

        // Fault-tolerant timing
        let ft_timing = match (&self.ft_config, &physical_estimate) {
            (Some(config), Some(phys)) => Some(FaultTolerantTiming::calculate(
                t_analysis.t_count,
                t_analysis.t_depth,
                t_analysis.clifford_count,
                phys,
                config,
            )),
            _ => None,
        };

        // NISQ timing
        let nisq_timing = self.nisq_profile.as_ref().map(|profile| {
            // Estimate execution time
            let gate_time = profile.t1q_time_ns * analysis.single_qubit_gates as f64
                + profile.t2q_time_ns * analysis.two_qubit_gates as f64;

            // Estimate success probability
            let fidelity_1q = profile
                .t1q_fidelity
                .powf(analysis.single_qubit_gates as f64);
            let fidelity_2q = profile.t2q_fidelity.powf(analysis.two_qubit_gates as f64);
            let success_probability = fidelity_1q * fidelity_2q;

            // Shots for 99% confidence (Chernoff bound approximation)
            let shots = if success_probability > 0.0 {
                (5.0 / success_probability).ceil() as usize
            } else {
                1_000_000
            };

            // SPRINT 57.0: Compute mitigated success probability
            let (mit_strategy, mit_probability) = match &self.mitigation_strategy {
                Some(crate::mitigation::MitigationStrategy::ZNE(config)) => {
                    // ZNE typically improves fidelity by 30-50% (recovering lost fidelity)
                    let error_rate = 1.0 - success_probability;
                    let improvement_factor = 0.4; // 40% error reduction
                    let recovered_error = error_rate * improvement_factor;
                    let improved = (success_probability + recovered_error).min(1.0);
                    (
                        Some(crate::mitigation::MitigationStrategy::ZNE(config.clone())),
                        Some(improved),
                    )
                }
                Some(crate::mitigation::MitigationStrategy::Readout(cal)) => {
                    // REM corrects measurement errors - estimate based on measure_fidelity
                    let measurement_error = 1.0 - profile.measure_fidelity as f64;
                    let correction_factor = 0.3; // 30% measurement error reduction
                    let improved =
                        (success_probability + measurement_error * correction_factor).min(1.0);
                    (
                        Some(crate::mitigation::MitigationStrategy::Readout(cal.clone())),
                        Some(improved),
                    )
                }
                Some(crate::mitigation::MitigationStrategy::ZNEPlusReadout { zne, readout }) => {
                    // Combined: apply both improvements
                    let error_rate = 1.0 - success_probability;
                    let measurement_error = 1.0 - profile.measure_fidelity as f64;
                    let zne_improvement = error_rate * 0.4;
                    let rem_improvement = measurement_error * 0.3;
                    let improved =
                        (success_probability + zne_improvement + rem_improvement).min(1.0);
                    (
                        Some(crate::mitigation::MitigationStrategy::ZNEPlusReadout {
                            zne: zne.clone(),
                            readout: readout.clone(),
                        }),
                        Some(improved),
                    )
                }
                _ => (None, None),
            };

            NISQTiming {
                hardware: profile.name.clone(),
                duration_ns: gate_time,
                success_probability,
                shots_for_confidence: shots,
                mitigation_strategy: mit_strategy,
                mitigated_success_probability: mit_probability,
            }
        });

        ResourceEstimate {
            name: name.to_string(),
            logical_qubits: analysis.num_qubits,
            total_gates: analysis.total_gates,
            depth: analysis.depth,
            single_qubit_gates: analysis.single_qubit_gates,
            two_qubit_gates: analysis.two_qubit_gates,
            three_qubit_gates,
            gate_breakdown: analysis.gate_counts,
            t_analysis,
            t_depth_analysis,
            ft_config: self.ft_config.clone(),
            physical_estimate,
            nisq_timing,
            ft_timing,
        }
    }
}

/// Estimate resources for a circuit (convenience function)
pub fn estimate_resources(circuit: &[QGate], name: &str) -> ResourceEstimate {
    ResourceEstimator::new()
        .with_ft_config(FaultTolerantConfig::default())
        .estimate(circuit, name)
}

/// Estimate resources with custom FT config
pub fn estimate_resources_ft(
    circuit: &[QGate],
    name: &str,
    config: FaultTolerantConfig,
) -> ResourceEstimate {
    ResourceEstimator::new()
        .with_ft_config(config)
        .estimate(circuit, name)
}

impl ResourceEstimate {
    /// Generate JSON report
    pub fn to_json(&self) -> String {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"name\": \"{}\",\n", self.name));
        json.push_str("  \"logical_circuit\": {\n");
        json.push_str(&format!("    \"qubits\": {},\n", self.logical_qubits));
        json.push_str(&format!("    \"total_gates\": {},\n", self.total_gates));
        json.push_str(&format!("    \"depth\": {},\n", self.depth));
        json.push_str(&format!(
            "    \"single_qubit_gates\": {},\n",
            self.single_qubit_gates
        ));
        json.push_str(&format!(
            "    \"two_qubit_gates\": {},\n",
            self.two_qubit_gates
        ));
        json.push_str(&format!(
            "    \"three_qubit_gates\": {}\n",
            self.three_qubit_gates
        ));
        json.push_str("  },\n");

        json.push_str("  \"t_gate_analysis\": {\n");
        json.push_str(&format!("    \"t_count\": {},\n", self.t_analysis.t_count));
        json.push_str(&format!("    \"t_depth\": {},\n", self.t_analysis.t_depth));
        json.push_str(&format!(
            "    \"clifford_count\": {},\n",
            self.t_analysis.clifford_count
        ));
        json.push_str(&format!(
            "    \"t_ratio\": {:.4}\n",
            self.t_analysis.t_ratio()
        ));
        json.push_str("  },\n");

        json.push_str("  \"t_depth_analysis\": {\n");
        json.push_str(&format!(
            "    \"original_depth\": {},\n",
            self.t_depth_analysis.original_t_depth
        ));
        json.push_str(&format!(
            "    \"optimized_depth\": {},\n",
            self.t_depth_analysis.optimized_t_depth
        ));
        json.push_str(&format!(
            "    \"parallelism\": {:.2}\n",
            self.t_depth_analysis.parallelization_factor()
        ));
        json.push_str("  },\n");

        if let Some(phys) = &self.physical_estimate {
            json.push_str("  \"fault_tolerant\": {\n");
            json.push_str(&format!("    \"code_distance\": {},\n", phys.code_distance));
            json.push_str(&format!(
                "    \"physical_per_logical\": {},\n",
                phys.physical_per_logical
            ));
            json.push_str(&format!("    \"data_qubits\": {},\n", phys.data_qubits));
            json.push_str(&format!(
                "    \"factory_qubits\": {},\n",
                phys.factory_qubits
            ));
            json.push_str(&format!("    \"num_factories\": {},\n", phys.num_factories));
            json.push_str(&format!(
                "    \"total_physical_qubits\": {},\n",
                phys.total_physical_qubits
            ));
            json.push_str(&format!(
                "    \"magic_states_required\": {},\n",
                phys.magic_states_required
            ));
            json.push_str(&format!(
                "    \"factory_cycles\": {}\n",
                phys.factory_cycles
            ));
            json.push_str("  },\n");
        }

        if let Some(timing) = &self.ft_timing {
            json.push_str("  \"ft_timing\": {\n");
            json.push_str(&format!(
                "    \"cycle_time_ns\": {},\n",
                timing.cycle_time_ns
            ));
            json.push_str(&format!(
                "    \"t_gate_cycles\": {},\n",
                timing.t_gate_cycles
            ));
            json.push_str(&format!(
                "    \"clifford_cycles\": {},\n",
                timing.clifford_cycles
            ));
            json.push_str(&format!(
                "    \"factory_cycles\": {},\n",
                timing.factory_cycles
            ));
            json.push_str(&format!(
                "    \"total_time_ns\": {},\n",
                timing.total_time_ns
            ));
            json.push_str(&format!("    \"bottleneck\": \"{}\"\n", timing.bottleneck));
            json.push_str("  },\n");
        }

        if let Some(nisq) = &self.nisq_timing {
            json.push_str("  \"nisq_timing\": {\n");
            json.push_str(&format!("    \"hardware\": \"{}\",\n", nisq.hardware));
            json.push_str(&format!("    \"duration_ns\": {},\n", nisq.duration_ns));
            json.push_str(&format!(
                "    \"success_probability\": {:.6},\n",
                nisq.success_probability
            ));
            json.push_str(&format!(
                "    \"shots_for_confidence\": {}\n",
                nisq.shots_for_confidence
            ));
            json.push_str("  },\n");
        }

        // Remove trailing comma
        if json.ends_with(",\n") {
            json.truncate(json.len() - 2);
            json.push('\n');
        }

        json.push_str("}\n");
        json
    }

    /// Generate Markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Resource Estimate: {}\n\n", self.name));

        // Logical Circuit
        md.push_str("## Logical Circuit\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Logical Qubits | {} |\n", self.logical_qubits));
        md.push_str(&format!("| Total Gates | {} |\n", self.total_gates));
        md.push_str(&format!("| Circuit Depth | {} |\n", self.depth));
        md.push_str(&format!("| 1Q Gates | {} |\n", self.single_qubit_gates));
        md.push_str(&format!("| 2Q Gates | {} |\n", self.two_qubit_gates));
        md.push_str(&format!("| 3Q Gates | {} |\n", self.three_qubit_gates));
        md.push('\n');

        // T-Gate Analysis
        md.push_str("## T-Gate Analysis\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| T-Count | {} |\n", self.t_analysis.t_count));
        md.push_str(&format!("| T-Depth | {} |\n", self.t_analysis.t_depth));
        md.push_str(&format!(
            "| Clifford Count | {} |\n",
            self.t_analysis.clifford_count
        ));
        md.push_str(&format!(
            "| T-Ratio | {:.2}% |\n",
            self.t_analysis.t_ratio() * 100.0
        ));
        if self.t_analysis.t_count > 0 {
            md.push_str(&format!(
                "| Parallelism | {:.2}x |\n",
                self.t_depth_analysis.parallelization_factor()
            ));
        }
        md.push('\n');

        // Fault-Tolerant Resources
        if let Some(phys) = &self.physical_estimate {
            md.push_str("## Fault-Tolerant Resources\n\n");
            md.push_str("| Metric | Value |\n");
            md.push_str("|--------|-------|\n");
            md.push_str(&format!("| Code Distance | {} |\n", phys.code_distance));
            md.push_str(&format!(
                "| Physical/Logical | {} |\n",
                phys.physical_per_logical
            ));
            md.push_str(&format!(
                "| Data Qubits | {} |\n",
                format_with_commas(phys.data_qubits)
            ));
            md.push_str(&format!(
                "| Factory Qubits | {} |\n",
                format_with_commas(phys.factory_qubits)
            ));
            md.push_str(&format!(
                "| **Total Physical Qubits** | **{}** |\n",
                format_with_commas(phys.total_physical_qubits)
            ));
            md.push_str(&format!(
                "| Magic States | {} |\n",
                format_with_commas(phys.magic_states_required)
            ));
            md.push_str(&format!("| Factories | {} |\n", phys.num_factories));
            md.push('\n');
        }

        // Timing
        if let Some(timing) = &self.ft_timing {
            md.push_str("## Fault-Tolerant Timing\n\n");
            let time_str = format_time(timing.total_time_ns);
            md.push_str(&format!("**Estimated Execution Time:** {}\n\n", time_str));
            md.push_str(&format!("**Bottleneck:** {}\n\n", timing.bottleneck));

            md.push_str("| Component | Cycles |\n");
            md.push_str("|-----------|--------|\n");
            md.push_str(&format!(
                "| T-gates | {} |\n",
                format_with_commas(timing.t_gate_cycles)
            ));
            md.push_str(&format!(
                "| Clifford | {} |\n",
                format_with_commas(timing.clifford_cycles)
            ));
            md.push_str(&format!(
                "| Factory | {} |\n",
                format_with_commas(timing.factory_cycles)
            ));
            md.push('\n');
        }

        if let Some(nisq) = &self.nisq_timing {
            md.push_str("## NISQ Timing\n\n");
            md.push_str(&format!("**Hardware:** {}\n\n", nisq.hardware));
            let time_str = format_time(nisq.duration_ns);
            md.push_str(&format!("**Execution Time:** {}\n\n", time_str));
            md.push_str(&format!(
                "**Success Probability:** {:.4}%\n\n",
                nisq.success_probability * 100.0
            ));
            md.push_str(&format!(
                "**Shots for 99% Confidence:** {}\n\n",
                format_with_commas(nisq.shots_for_confidence)
            ));
        }

        // Gate Breakdown
        if !self.gate_breakdown.is_empty() {
            md.push_str("## Gate Breakdown\n\n");
            md.push_str("| Gate | Count |\n");
            md.push_str("|------|-------|\n");

            let mut gates: Vec<_> = self.gate_breakdown.iter().collect();
            gates.sort_by(|a, b| b.1.cmp(a.1));

            for (gate, count) in gates {
                md.push_str(&format!("| {} | {} |\n", gate, count));
            }
        }

        md
    }

    /// Check if circuit is Clifford-only
    pub fn is_clifford(&self) -> bool {
        self.t_analysis.is_clifford()
    }

    /// Get total physical qubit count (if FT enabled)
    pub fn physical_qubits(&self) -> Option<usize> {
        self.physical_estimate
            .as_ref()
            .map(|p| p.total_physical_qubits)
    }

    /// Get execution time estimate (FT or NISQ)
    pub fn execution_time_ns(&self) -> Option<f64> {
        self.ft_timing
            .as_ref()
            .map(|t| t.total_time_ns)
            .or_else(|| self.nisq_timing.as_ref().map(|t| t.duration_ns))
    }
}

/// Format time in human-readable units
fn format_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:.2} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if ns < 60_000_000_000.0 {
        format!("{:.2} s", ns / 1_000_000_000.0)
    } else if ns < 3_600_000_000_000.0 {
        format!("{:.2} min", ns / 60_000_000_000.0)
    } else if ns < 86_400_000_000_000.0 {
        format!("{:.2} hours", ns / 3_600_000_000_000.0)
    } else {
        format!("{:.2} days", ns / 86_400_000_000_000.0)
    }
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════╗"
        )?;
        writeln!(f, "║         RESOURCE ESTIMATE: {:^24}   ║", self.name)?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════╣"
        )?;

        writeln!(
            f,
            "║ LOGICAL CIRCUIT                                          ║"
        )?;
        writeln!(
            f,
            "║   Qubits: {:>6}    Gates: {:>8}    Depth: {:>6}      ║",
            self.logical_qubits, self.total_gates, self.depth
        )?;
        writeln!(
            f,
            "║   1Q: {:>6}  2Q: {:>6}  3Q: {:>6}                       ║",
            self.single_qubit_gates, self.two_qubit_gates, self.three_qubit_gates
        )?;

        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ T-GATE ANALYSIS                                          ║"
        )?;
        writeln!(
            f,
            "║   T-Count: {:>8}    T-Depth: {:>8}                   ║",
            self.t_analysis.t_count, self.t_analysis.t_depth
        )?;
        writeln!(
            f,
            "║   Clifford: {:>7}    T-Ratio: {:>6.2}%                   ║",
            self.t_analysis.clifford_count,
            self.t_analysis.t_ratio() * 100.0
        )?;

        if let Some(phys) = &self.physical_estimate {
            writeln!(
                f,
                "╠══════════════════════════════════════════════════════════╣"
            )?;
            writeln!(
                f,
                "║ FAULT-TOLERANT RESOURCES                                 ║"
            )?;
            writeln!(
                f,
                "║   Code Distance: {:>4}    Physical/Logical: {:>6}       ║",
                phys.code_distance, phys.physical_per_logical
            )?;
            writeln!(
                f,
                "║   Data Qubits: {:>12}                               ║",
                format_with_commas(phys.data_qubits)
            )?;
            writeln!(
                f,
                "║   Factory Qubits: {:>9}  ({} factories)              ║",
                format_with_commas(phys.factory_qubits),
                phys.num_factories
            )?;
            writeln!(
                f,
                "║   TOTAL PHYSICAL: {:>12}                          ║",
                format_with_commas(phys.total_physical_qubits)
            )?;
            writeln!(
                f,
                "║   Magic States: {:>11}                            ║",
                format_with_commas(phys.magic_states_required)
            )?;
        }

        if let Some(timing) = &self.ft_timing {
            writeln!(
                f,
                "╠══════════════════════════════════════════════════════════╣"
            )?;
            writeln!(
                f,
                "║ EXECUTION TIME                                           ║"
            )?;
            let time_str = format_time(timing.total_time_ns);
            writeln!(
                f,
                "║   Fault-Tolerant: {:>20}                   ║",
                time_str
            )?;
            writeln!(
                f,
                "║   Bottleneck: {:>24}               ║",
                timing.bottleneck
            )?;
        }

        if let Some(nisq) = &self.nisq_timing {
            let time_str = format_time(nisq.duration_ns);
            writeln!(
                f,
                "║   NISQ ({}): {:>20}                 ║",
                &nisq.hardware[..nisq.hardware.len().min(8)],
                time_str
            )?;
            writeln!(
                f,
                "║   Success Rate: {:>6.2}%                                  ║",
                nisq.success_probability * 100.0
            )?;
        }

        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════╝"
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_estimate() {
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::Measure(0),
        ];

        let estimate = estimate_resources(&circuit, "test");

        assert_eq!(estimate.logical_qubits, 2);
        assert_eq!(estimate.total_gates, 4);
        assert_eq!(estimate.t_analysis.t_count, 1);
        assert!(estimate.physical_estimate.is_some());
    }

    #[test]
    fn test_clifford_only() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::X(1)];

        let estimate = estimate_resources(&circuit, "clifford");
        assert!(estimate.is_clifford());
        assert_eq!(estimate.t_analysis.t_count, 0);
    }

    #[test]
    fn test_code_distance() {
        let config = FaultTolerantConfig::for_error_rate(1e-12);

        // Small circuit needs lower distance
        let d1 = config.required_code_distance(100);
        // Large circuit needs higher distance
        let d2 = config.required_code_distance(1_000_000);

        assert!(d1 >= 3);
        assert!(d2 > d1);
        assert!(d1 % 2 == 1); // Odd
        assert!(d2 % 2 == 1);
    }

    #[test]
    fn test_physical_estimate() {
        let config = FaultTolerantConfig::for_error_rate(1e-12);
        let phys = PhysicalResourceEstimate::calculate(10, 100, 10, &config);

        assert!(phys.code_distance >= 3);
        assert!(phys.total_physical_qubits > 0);
        assert!(phys.num_factories > 0);
    }

    #[test]
    fn test_json_output() {
        let circuit = vec![QGate::H(0), QGate::T(0)];
        let estimate = estimate_resources(&circuit, "test");
        let json = estimate.to_json();

        assert!(json.contains("\"name\": \"test\""));
        assert!(json.contains("\"t_count\": 1"));
    }

    #[test]
    fn test_markdown_output() {
        let circuit = vec![QGate::H(0), QGate::T(0)];
        let estimate = estimate_resources(&circuit, "test");
        let md = estimate.to_markdown();

        assert!(md.contains("# Resource Estimate: test"));
        assert!(md.contains("| T-Count |"));
    }

    #[test]
    fn test_toffoli_circuit() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::Toffoli(0, 1, 2),
            QGate::Measure(2),
        ];

        let estimate = estimate_resources(&circuit, "toffoli_test");

        assert_eq!(estimate.logical_qubits, 3);
        assert_eq!(estimate.three_qubit_gates, 1);
    }

    #[test]
    fn test_format_time() {
        assert!(format_time(500.0).contains("ns"));
        assert!(format_time(5000.0).contains("µs"));
        assert!(format_time(5_000_000.0).contains("ms"));
        assert!(format_time(5_000_000_000.0).contains(" s"));
    }

    #[test]
    fn test_ft_timing() {
        let config = FaultTolerantConfig::superconducting();
        let phys = PhysicalResourceEstimate::calculate(5, 50, 10, &config);
        let timing = FaultTolerantTiming::calculate(50, 10, 100, &phys, &config);

        assert!(timing.total_time_ns > 0.0);
        assert!(!timing.bottleneck.is_empty());
    }
}
