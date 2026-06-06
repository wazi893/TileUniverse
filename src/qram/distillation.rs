//! Magic State Distillation Protocols
//!
//! SPRINT 51.0 Phase 2: Distillation Protocols
//!
//! This module implements various magic state distillation protocols used
//! in fault-tolerant quantum computing. Distillation converts many noisy
//! magic states into fewer, higher-quality magic states.
//!
//! # Overview
//!
//! Magic states |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 are required for T-gates.
//! Physical |T⟩ states have errors; distillation reduces error rate.
//!
//! # Protocols
//!
//! - **15-to-1 (Bravyi-Kitaev)**: Standard protocol, p_out ≈ 35p³
//! - **10-to-2**: Improved efficiency for some error rates
//! - **Reed-Muller**: Very low output error, higher overhead
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::distillation::{DistillationProtocol, DistillationFactory};
//!
//! // Create factory for target error rate
//! let factory = DistillationFactory::for_target_error(1e-3, 1e-12);
//!
//! // Get resources needed for 100 clean magic states
//! let resources = factory.resources_for(100);
//! println!("Factory qubits: {}", resources.total_qubits);
//! ```

use std::fmt;

/// A magic state distillation protocol
///
/// Describes how many input states are consumed to produce output states,
/// along with the error transformation and resource overhead.
#[derive(Clone, Debug)]
pub struct DistillationProtocol {
    /// Protocol name
    pub name: &'static str,
    /// Number of input (noisy) magic states
    pub input_states: usize,
    /// Number of output (cleaner) magic states
    pub output_states: usize,
    /// Maximum input error rate for protocol to work
    pub max_input_error: f64,
    /// Minimum achievable output error
    pub min_output_error: f64,
    /// Qubit overhead per distillation round
    pub qubit_overhead: usize,
    /// Cycle overhead per distillation round
    pub cycle_overhead: usize,
    /// Error suppression exponent (p_out ~ p_in^exponent)
    pub error_exponent: u32,
    /// Error coefficient (p_out ~ coefficient * p_in^exponent)
    pub error_coefficient: f64,
}

impl DistillationProtocol {
    /// Standard 15-to-1 protocol (Bravyi-Kitaev)
    ///
    /// Uses 15 noisy |T⟩ states to produce 1 clean |T⟩ state.
    /// Error transformation: p_out ≈ 35 × p_in³
    pub fn fifteen_to_one() -> Self {
        Self {
            name: "15-to-1",
            input_states: 15,
            output_states: 1,
            max_input_error: 0.14, // Protocol fails above this
            min_output_error: 1e-15,
            qubit_overhead: 20,
            cycle_overhead: 15,
            error_exponent: 3,
            error_coefficient: 35.0,
        }
    }

    /// 10-to-2 protocol (improved Meier-Eastin-Knill)
    ///
    /// Uses 10 noisy states to produce 2 clean states.
    /// More efficient than 15-to-1 for some error ranges.
    pub fn ten_to_two() -> Self {
        Self {
            name: "10-to-2",
            input_states: 10,
            output_states: 2,
            max_input_error: 0.10,
            min_output_error: 1e-14,
            qubit_overhead: 15,
            cycle_overhead: 12,
            error_exponent: 3,
            error_coefficient: 20.0,
        }
    }

    /// Reed-Muller based protocol
    ///
    /// Higher overhead but achieves very low error rates.
    /// Based on punctured Reed-Muller codes.
    pub fn reed_muller() -> Self {
        Self {
            name: "Reed-Muller",
            input_states: 16,
            output_states: 1,
            max_input_error: 0.05,
            min_output_error: 1e-20,
            qubit_overhead: 32,
            cycle_overhead: 20,
            error_exponent: 4,
            error_coefficient: 100.0,
        }
    }

    /// 7-to-1 protocol (Steane-based)
    ///
    /// Lower overhead but less error suppression.
    pub fn seven_to_one() -> Self {
        Self {
            name: "7-to-1",
            input_states: 7,
            output_states: 1,
            max_input_error: 0.20,
            min_output_error: 1e-10,
            qubit_overhead: 12,
            cycle_overhead: 10,
            error_exponent: 2,
            error_coefficient: 7.0,
        }
    }

    /// Calculate output error rate from input error rate
    pub fn distill(&self, input_error: f64) -> f64 {
        if input_error > self.max_input_error {
            // Protocol doesn't work at this error rate
            return 1.0;
        }

        let output = self.error_coefficient * input_error.powi(self.error_exponent as i32);
        output.max(self.min_output_error)
    }

    /// Calculate effective ratio (input consumed per output produced)
    pub fn effective_ratio(&self) -> f64 {
        self.input_states as f64 / self.output_states as f64
    }

    /// Check if protocol works at given input error
    pub fn is_valid_for(&self, input_error: f64) -> bool {
        input_error <= self.max_input_error
    }

    /// Get resource cost for producing N clean states
    pub fn resources_for(&self, n_clean: usize) -> ProtocolResources {
        let rounds = (n_clean + self.output_states - 1) / self.output_states;
        let input_consumed = rounds * self.input_states;
        let qubits = self.qubit_overhead;
        let cycles = rounds * self.cycle_overhead;

        ProtocolResources {
            rounds,
            input_consumed,
            output_produced: rounds * self.output_states,
            qubits,
            cycles,
        }
    }
}

impl fmt::Display for DistillationProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}-to-{}, p_out≈{:.0}p^{})",
            self.name,
            self.input_states,
            self.output_states,
            self.error_coefficient,
            self.error_exponent
        )
    }
}

/// Resources for a single protocol level
#[derive(Clone, Debug)]
pub struct ProtocolResources {
    /// Distillation rounds needed
    pub rounds: usize,
    /// Input states consumed
    pub input_consumed: usize,
    /// Output states produced
    pub output_produced: usize,
    /// Qubits used
    pub qubits: usize,
    /// Total cycles
    pub cycles: usize,
}

/// Multi-level distillation factory
///
/// Chains multiple distillation protocols to achieve very low error rates.
/// Each level takes the output of the previous level as input.
#[derive(Clone, Debug)]
pub struct DistillationFactory {
    /// Protocols at each level (level 0 = closest to physical)
    levels: Vec<DistillationProtocol>,
    /// Physical T-state error rate
    physical_error_rate: f64,
    /// Target output error rate
    target_error_rate: f64,
}

impl DistillationFactory {
    /// Create factory with specified levels
    pub fn new(physical_rate: f64, levels: Vec<DistillationProtocol>) -> Self {
        let target = levels.iter().fold(physical_rate, |e, p| p.distill(e));

        Self {
            levels,
            physical_error_rate: physical_rate,
            target_error_rate: target,
        }
    }

    /// Create factory to achieve target error rate
    ///
    /// Automatically determines number of 15-to-1 levels needed.
    pub fn for_target_error(physical_rate: f64, target_rate: f64) -> Self {
        let mut levels = Vec::new();
        let mut current_error = physical_rate;
        let protocol = DistillationProtocol::fifteen_to_one();

        while current_error > target_rate && levels.len() < 5 {
            if !protocol.is_valid_for(current_error) {
                // Need to use lower-overhead protocol first
                let starter = DistillationProtocol::seven_to_one();
                levels.push(starter.clone());
                current_error = starter.distill(current_error);
            } else {
                levels.push(protocol.clone());
                current_error = protocol.distill(current_error);
            }
        }

        Self {
            levels,
            physical_error_rate: physical_rate,
            target_error_rate: current_error,
        }
    }

    /// Create factory with mixed protocols for efficiency
    pub fn optimized(physical_rate: f64, target_rate: f64) -> Self {
        let mut levels = Vec::new();
        let mut current_error = physical_rate;

        // Use 10-to-2 when possible (more efficient)
        let ten_to_two = DistillationProtocol::ten_to_two();
        let fifteen_to_one = DistillationProtocol::fifteen_to_one();

        while current_error > target_rate && levels.len() < 5 {
            if ten_to_two.is_valid_for(current_error)
                && ten_to_two.distill(current_error) < fifteen_to_one.distill(current_error) * 0.5
            {
                levels.push(ten_to_two.clone());
                current_error = ten_to_two.distill(current_error);
            } else if fifteen_to_one.is_valid_for(current_error) {
                levels.push(fifteen_to_one.clone());
                current_error = fifteen_to_one.distill(current_error);
            } else {
                // Error rate too high, start with 7-to-1
                let starter = DistillationProtocol::seven_to_one();
                levels.push(starter.clone());
                current_error = starter.distill(current_error);
            }
        }

        Self {
            levels,
            physical_error_rate: physical_rate,
            target_error_rate: current_error,
        }
    }

    /// Get number of distillation levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get error rate at each level
    pub fn error_rates(&self) -> Vec<f64> {
        let mut rates = vec![self.physical_error_rate];
        let mut current = self.physical_error_rate;

        for protocol in &self.levels {
            current = protocol.distill(current);
            rates.push(current);
        }

        rates
    }

    /// Get achieved error rate
    pub fn achieved_error_rate(&self) -> f64 {
        self.target_error_rate
    }

    /// Calculate total overhead ratio (physical states per clean state)
    pub fn total_overhead(&self) -> usize {
        self.levels
            .iter()
            .map(|p| p.input_states / p.output_states)
            .product()
    }

    /// Calculate resources needed for N clean magic states
    pub fn resources_for(&self, n_clean_states: usize) -> FactoryResources {
        if self.levels.is_empty() {
            return FactoryResources {
                total_qubits: 0,
                total_cycles: 0,
                noisy_states_consumed: n_clean_states,
                per_level: vec![],
                achieved_error: self.physical_error_rate,
            };
        }

        // Work backwards from output to input
        let mut states_needed = vec![n_clean_states];

        for protocol in self.levels.iter().rev() {
            let output_needed = *states_needed.last().unwrap();
            let input_needed =
                ((output_needed as f64) * protocol.effective_ratio()).ceil() as usize;
            states_needed.push(input_needed);
        }

        states_needed.reverse();

        // Calculate resources per level
        let mut per_level = Vec::new();
        let mut total_qubits = 0;
        let mut total_cycles = 0;

        for (i, protocol) in self.levels.iter().enumerate() {
            let input = states_needed[i];
            let resources =
                protocol.resources_for((input as f64 / protocol.effective_ratio()).ceil() as usize);
            total_qubits = total_qubits.max(resources.qubits * (i + 1)); // Stages can overlap
            total_cycles += resources.cycles;
            per_level.push(resources);
        }

        FactoryResources {
            total_qubits,
            total_cycles,
            noisy_states_consumed: *states_needed.first().unwrap(),
            per_level,
            achieved_error: self.target_error_rate,
        }
    }

    /// Get protocols used
    pub fn protocols(&self) -> &[DistillationProtocol] {
        &self.levels
    }
}

impl fmt::Display for DistillationFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DistillationFactory({} levels, {:.0e} → {:.0e}, {}:1 overhead)",
            self.levels.len(),
            self.physical_error_rate,
            self.target_error_rate,
            self.total_overhead()
        )
    }
}

/// Resources for entire factory
#[derive(Clone, Debug)]
pub struct FactoryResources {
    /// Total qubits in factory
    pub total_qubits: usize,
    /// Total cycles needed
    pub total_cycles: usize,
    /// Noisy physical states consumed
    pub noisy_states_consumed: usize,
    /// Resources per level
    pub per_level: Vec<ProtocolResources>,
    /// Achieved error rate
    pub achieved_error: f64,
}

impl fmt::Display for FactoryResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FactoryResources({}q, {} cycles, {} noisy → {:.0e} error)",
            self.total_qubits, self.total_cycles, self.noisy_states_consumed, self.achieved_error
        )
    }
}

/// Compare different factory configurations
pub fn compare_factories(
    physical_rate: f64,
    targets: &[f64],
    n_states: usize,
) -> Vec<FactoryComparison> {
    targets
        .iter()
        .map(|&target| {
            let factory = DistillationFactory::for_target_error(physical_rate, target);
            let resources = factory.resources_for(n_states);

            FactoryComparison {
                target_error: target,
                achieved_error: factory.achieved_error_rate(),
                levels: factory.num_levels(),
                overhead: factory.total_overhead(),
                qubits: resources.total_qubits,
                cycles: resources.total_cycles,
            }
        })
        .collect()
}

/// Factory comparison result
#[derive(Clone, Debug)]
pub struct FactoryComparison {
    /// Target error rate
    pub target_error: f64,
    /// Achieved error rate
    pub achieved_error: f64,
    /// Number of levels
    pub levels: usize,
    /// Overhead ratio
    pub overhead: usize,
    /// Factory qubits
    pub qubits: usize,
    /// Total cycles
    pub cycles: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fifteen_to_one() {
        let protocol = DistillationProtocol::fifteen_to_one();
        assert_eq!(protocol.input_states, 15);
        assert_eq!(protocol.output_states, 1);

        // Test error transformation
        let output = protocol.distill(1e-3);
        assert!(output < 1e-6); // 35 * (10^-3)^3 = 3.5e-8
    }

    #[test]
    fn test_ten_to_two() {
        let protocol = DistillationProtocol::ten_to_two();
        assert_eq!(protocol.effective_ratio(), 5.0);
    }

    #[test]
    fn test_protocol_validity() {
        let protocol = DistillationProtocol::fifteen_to_one();
        assert!(protocol.is_valid_for(0.1));
        assert!(!protocol.is_valid_for(0.2));
    }

    #[test]
    fn test_factory_for_target() {
        let factory = DistillationFactory::for_target_error(1e-3, 1e-10);
        assert!(factory.num_levels() >= 1);
        assert!(factory.achieved_error_rate() <= 1e-10);
    }

    #[test]
    fn test_factory_overhead() {
        let factory = DistillationFactory::for_target_error(1e-3, 1e-10);
        let overhead = factory.total_overhead();
        assert!(overhead >= 15); // At least one 15-to-1 level
    }

    #[test]
    fn test_factory_resources() {
        let factory = DistillationFactory::for_target_error(1e-3, 1e-10);
        let resources = factory.resources_for(100);

        assert!(resources.total_qubits > 0);
        assert!(resources.total_cycles > 0);
        assert!(resources.noisy_states_consumed >= 100 * 15);
    }

    #[test]
    fn test_error_rates_chain() {
        let factory = DistillationFactory::for_target_error(1e-3, 1e-15);
        let rates = factory.error_rates();

        // Each level should decrease error
        for i in 1..rates.len() {
            assert!(rates[i] < rates[i - 1]);
        }
    }

    #[test]
    fn test_optimized_factory() {
        let standard = DistillationFactory::for_target_error(1e-3, 1e-10);
        let optimized = DistillationFactory::optimized(1e-3, 1e-10);

        // Optimized should achieve similar error with potentially different structure
        assert!(optimized.achieved_error_rate() <= 1e-9);

        let std_resources = standard.resources_for(100);
        let opt_resources = optimized.resources_for(100);

        // Optimized might use fewer noisy states
        println!(
            "Standard: {} noisy, Optimized: {} noisy",
            std_resources.noisy_states_consumed, opt_resources.noisy_states_consumed
        );
    }

    #[test]
    fn test_compare_factories() {
        let comparisons = compare_factories(1e-3, &[1e-6, 1e-10, 1e-15], 50);
        assert_eq!(comparisons.len(), 3);

        // Higher target (lower error) should need more overhead
        assert!(comparisons[2].overhead >= comparisons[0].overhead);
    }

    #[test]
    fn test_display() {
        let protocol = DistillationProtocol::fifteen_to_one();
        let s = format!("{}", protocol);
        assert!(s.contains("15-to-1"));

        let factory = DistillationFactory::for_target_error(1e-3, 1e-10);
        let s = format!("{}", factory);
        assert!(s.contains("levels"));
    }
}
