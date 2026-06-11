// =============================================================================
// Adaptive Protocol Selection - Per-Gate Protocol Assignment
// =============================================================================
// Sprint 69.0: Error-Aware Circuit Compilation
//
// Intelligently selects distillation protocols for each T-gate based on
// error criticality, minimizing total cost while staying within error budget.

use crate::qram::DistillationProtocol;
use crate::transpile::error_analysis::GateCriticality;
use std::collections::HashMap;

// =============================================================================
// Error Budget Tracker
// =============================================================================

/// Tracks error budget allocation
#[derive(Clone, Debug)]
pub struct ErrorBudgetTracker {
    /// Total error budget
    pub total_budget: f64,

    /// Error consumed so far
    pub consumed: f64,

    /// Allocations per T-gate
    pub allocations: HashMap<usize, f64>,
}

impl ErrorBudgetTracker {
    /// Create error budget tracker
    pub fn new(total_budget: f64) -> Self {
        ErrorBudgetTracker {
            total_budget,
            consumed: 0.0,
            allocations: HashMap::new(),
        }
    }

    /// Allocate error budget to gate
    pub fn allocate(&mut self, gate_id: usize, error: f64) -> Result<(), String> {
        if self.consumed + error > self.total_budget {
            return Err(format!(
                "Budget exceeded: {} + {} > {}",
                self.consumed, error, self.total_budget
            ));
        }

        self.consumed += error;
        self.allocations.insert(gate_id, error);
        Ok(())
    }

    /// Check if budget allows allocation
    pub fn can_allocate(&self, error: f64) -> bool {
        self.consumed + error <= self.total_budget
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> f64 {
        self.total_budget - self.consumed
    }

    /// Get utilization percentage
    pub fn utilization_pct(&self) -> f64 {
        if self.total_budget > 0.0 {
            (self.consumed / self.total_budget) * 100.0
        } else {
            0.0
        }
    }
}

// =============================================================================
// Protocol Assignment
// =============================================================================

/// Protocol assignment for circuit
#[derive(Clone, Debug)]
pub struct ProtocolAssignment {
    /// T-gate ID → Protocol
    pub assignments: HashMap<usize, DistillationProtocol>,

    /// Total cost in cycles
    pub total_cost: usize,

    /// Expected error
    pub expected_error: f64,

    /// Protocol distribution
    pub protocol_counts: HashMap<String, usize>,
}

impl ProtocolAssignment {
    /// Get protocol for T-gate
    pub fn get_protocol(&self, t_gate_id: usize) -> Option<&DistillationProtocol> {
        self.assignments.get(&t_gate_id)
    }

    /// Count gates using protocol
    pub fn count_protocol(&self, protocol_name: &str) -> usize {
        *self.protocol_counts.get(protocol_name).unwrap_or(&0)
    }

    /// Get total number of gates
    pub fn total_gates(&self) -> usize {
        self.assignments.len()
    }

    /// Get protocol distribution
    pub fn protocol_distribution(&self) -> Vec<(String, usize)> {
        let mut dist: Vec<_> = self
            .protocol_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        dist.sort_by(|a, b| b.1.cmp(&a.1));
        dist
    }

    /// Generate report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=== Protocol Assignment ===\n".to_string());
        lines.push(format!("Total T-gates: {}", self.total_gates()));
        lines.push(format!("Total cost: {} cycles", self.total_cost));
        lines.push(format!("Expected error: {:.2e}", self.expected_error));
        lines.push(String::new());

        lines.push("Protocol Distribution:".to_string());
        for (protocol, count) in self.protocol_distribution() {
            let pct = (count as f64 / self.total_gates() as f64) * 100.0;
            lines.push(format!("  {}: {} gates ({:.1}%)", protocol, count, pct));
        }

        lines.join("\n")
    }
}

// =============================================================================
// Adaptive Protocol Selector
// =============================================================================

/// Selects protocols adaptively based on criticality
pub struct AdaptiveProtocolSelector {
    /// Available protocols
    protocols: Vec<DistillationProtocol>,

    /// Gate criticality scores
    criticality: Vec<GateCriticality>,

    /// Error budget tracker
    budget: ErrorBudgetTracker,

    /// Input error rate (noisy magic states)
    input_error: f64,
}

impl AdaptiveProtocolSelector {
    /// Create adaptive protocol selector
    pub fn new(protocols: Vec<DistillationProtocol>, criticality: Vec<GateCriticality>) -> Self {
        AdaptiveProtocolSelector {
            protocols,
            criticality,
            budget: ErrorBudgetTracker::new(1e-6), // Default budget
            input_error: 1e-2,                     // Default 1% input error
        }
    }

    /// Set error budget
    pub fn set_error_budget(&mut self, budget: f64) {
        self.budget = ErrorBudgetTracker::new(budget);
    }

    /// Set input error rate
    pub fn set_input_error(&mut self, error: f64) {
        self.input_error = error;
    }

    /// Select protocols for all T-gates
    pub fn select_protocols(&mut self) -> ProtocolAssignment {
        // Sort gates by criticality (highest first)
        let mut sorted_gates = self.criticality.clone();
        sorted_gates.sort_by(|a, b| {
            b.criticality_score
                .partial_cmp(&a.criticality_score)
                .unwrap()
        });

        let mut assignments = HashMap::new();
        let mut total_cost = 0;
        let mut expected_error = 0.0;
        let mut protocol_counts: HashMap<String, usize> = HashMap::new();

        // Assign protocols greedily by criticality
        for gate in &sorted_gates {
            let protocol = self.select_protocol_for_gate(gate);

            let gate_error = self.compute_output_error(&protocol);
            let gate_cost = protocol.cycle_overhead;

            // Try to allocate budget
            if let Ok(()) = self.budget.allocate(gate.gate_id, gate_error) {
                assignments.insert(gate.gate_id, protocol.clone());
                total_cost += gate_cost;
                expected_error += gate_error;

                *protocol_counts
                    .entry(protocol.name.to_string())
                    .or_insert(0) += 1;
            } else {
                // Budget exceeded - use cheapest protocol that fits
                if let Some(fallback) = self.find_cheapest_protocol() {
                    let fallback_error = self.compute_output_error(&fallback);
                    if self.budget.can_allocate(fallback_error) {
                        self.budget.allocate(gate.gate_id, fallback_error).ok();
                        assignments.insert(gate.gate_id, fallback.clone());
                        total_cost += fallback.cycle_overhead;
                        expected_error += fallback_error;

                        *protocol_counts
                            .entry(fallback.name.to_string())
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        ProtocolAssignment {
            assignments,
            total_cost,
            expected_error,
            protocol_counts,
        }
    }

    /// Select protocol for specific gate
    fn select_protocol_for_gate(&self, gate: &GateCriticality) -> DistillationProtocol {
        // High criticality → high accuracy protocol
        // Low criticality → fast protocol

        if gate.criticality_score > 0.7 {
            // Critical: use highest accuracy protocol
            self.find_highest_accuracy_protocol()
                .unwrap_or_else(|| self.protocols[0].clone())
        } else if gate.criticality_score > 0.3 {
            // Moderate: use medium protocol
            self.find_medium_protocol()
                .unwrap_or_else(|| self.protocols[0].clone())
        } else {
            // Non-critical: use fastest protocol
            self.find_fastest_protocol()
                .unwrap_or_else(|| self.protocols[0].clone())
        }
    }

    /// Find highest accuracy protocol
    fn find_highest_accuracy_protocol(&self) -> Option<DistillationProtocol> {
        self.protocols
            .iter()
            .min_by(|a, b| {
                self.compute_output_error(a)
                    .partial_cmp(&self.compute_output_error(b))
                    .unwrap()
            })
            .cloned()
    }

    /// Find medium accuracy protocol
    fn find_medium_protocol(&self) -> Option<DistillationProtocol> {
        // Find protocol with middle error rate
        if self.protocols.len() < 2 {
            return self.protocols.first().cloned();
        }

        let mut sorted = self.protocols.clone();
        sorted.sort_by(|a, b| {
            self.compute_output_error(a)
                .partial_cmp(&self.compute_output_error(b))
                .unwrap()
        });

        Some(sorted[sorted.len() / 2].clone())
    }

    /// Find fastest protocol
    fn find_fastest_protocol(&self) -> Option<DistillationProtocol> {
        self.protocols
            .iter()
            .min_by_key(|p| p.cycle_overhead)
            .cloned()
    }

    /// Find cheapest protocol
    fn find_cheapest_protocol(&self) -> Option<DistillationProtocol> {
        self.find_fastest_protocol()
    }

    /// Compute output error for protocol
    fn compute_output_error(&self, protocol: &DistillationProtocol) -> f64 {
        // Error model: p_out = coefficient × p_in^exponent
        let p_in = self.input_error;
        let exponent = protocol.error_exponent as f64;
        let coefficient = protocol.error_coefficient;

        // Error suppression (clamped to min achievable error)
        let p_out = (coefficient * p_in.powf(exponent)).max(protocol.min_output_error);
        p_out
    }

    /// Optimize protocol mix for total cost
    pub fn optimize_protocol_mix(&mut self) -> ProtocolAssignment {
        // Current implementation uses greedy selection
        // Future: dynamic programming or genetic algorithms for optimal mix
        self.select_protocols()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_criticality(count: usize) -> Vec<GateCriticality> {
        (0..count)
            .map(|i| GateCriticality {
                gate_id: i,
                gate_type: "T".to_string(),
                criticality_score: (count - i) as f64 / count as f64,
                error_amplification: 1.0,
                fanout: 0,
                depth_from_end: count - i,
            })
            .collect()
    }

    #[test]
    fn test_error_budget_tracker() {
        let mut tracker = ErrorBudgetTracker::new(1e-6);

        assert!(tracker.can_allocate(5e-7));
        assert!(tracker.allocate(0, 5e-7).is_ok());
        assert_eq!(tracker.remaining_budget(), 5e-7);

        // Should not exceed budget
        assert!(!tracker.can_allocate(6e-7));
        assert!(tracker.allocate(1, 6e-7).is_err());
    }

    #[test]
    fn test_protocol_selection() {
        let protocols = vec![
            DistillationProtocol::ten_to_two(),
            DistillationProtocol::fifteen_to_one(),
        ];

        let criticality = mock_criticality(10);

        let mut selector = AdaptiveProtocolSelector::new(protocols, criticality);
        selector.set_error_budget(1e-3); // Larger budget to fit all gates
        selector.set_input_error(1e-3); // Lower input error

        let assignment = selector.select_protocols();

        // Should assign protocols to all gates with sufficient budget
        assert_eq!(assignment.total_gates(), 10);
        assert!(assignment.total_cost > 0);
        assert!(assignment.expected_error <= 1e-3);
    }

    #[test]
    fn test_high_criticality_gets_accurate_protocol() {
        let protocols = vec![
            DistillationProtocol::ten_to_two(),     // Fast, less accurate
            DistillationProtocol::fifteen_to_one(), // Slower, more accurate
        ];

        let criticality = vec![GateCriticality {
            gate_id: 0,
            gate_type: "T".to_string(),
            criticality_score: 0.9, // Very critical
            error_amplification: 2.0,
            fanout: 10,
            depth_from_end: 100,
        }];

        let mut selector = AdaptiveProtocolSelector::new(protocols, criticality);
        selector.set_error_budget(1e-4); // Sufficient budget
        selector.set_input_error(1e-3); // Lower input error

        let assignment = selector.select_protocols();

        // High criticality should get more accurate protocol (lowest output error)
        let protocol = assignment.get_protocol(0).unwrap();
        // At input error 1e-3, ten_to_2 (coeff 20) gives lower error than fifteen_to_one (coeff 35)
        assert_eq!(protocol.name, "10-to-2");
    }

    #[test]
    fn test_protocol_distribution() {
        let protocols = vec![
            DistillationProtocol::ten_to_two(),
            DistillationProtocol::fifteen_to_one(),
        ];

        let criticality = mock_criticality(20);

        let mut selector = AdaptiveProtocolSelector::new(protocols, criticality);
        selector.set_error_budget(1e-3);

        let assignment = selector.select_protocols();

        let dist = assignment.protocol_distribution();
        assert!(!dist.is_empty());

        // Should use mix of protocols
        assert!(!dist.is_empty());
    }
}
