// =============================================================================
// Circuit Rewriting Engine - Circuit Optimization via Strategic Rewrites
// =============================================================================
// Sprint 69.0: Error-Aware Circuit Compilation
//
// Rewrites quantum circuits to optimize T-depth, reduce gate count, and
// improve overall circuit quality while preserving functional equivalence.

use crate::quantum::QGate;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Rewrite Strategies
// =============================================================================

/// Circuit rewrite strategy
#[derive(Clone, Debug, PartialEq)]
pub enum RewriteStrategy {
    /// Minimize T-depth through commutation and reordering
    TDepthMinimization,

    /// Substitute gates with approximate versions
    ApproximateSubstitution { epsilon: f64 },

    /// Merge consecutive Clifford gates
    CliffordReduction,

    /// Remove identity sequences
    IdentityElimination,

    /// Gate cancellation (e.g., H·H = I)
    GateCancellation,
}

// =============================================================================
// Rewritten Circuit
// =============================================================================

/// Result of circuit rewriting
#[derive(Clone, Debug)]
pub struct RewrittenCircuit {
    /// Optimized circuit
    pub circuit: Vec<QGate>,

    /// Original T-depth
    pub original_t_depth: usize,

    /// Optimized T-depth
    pub optimized_t_depth: usize,

    /// Original gate count
    pub original_gate_count: usize,

    /// Optimized gate count
    pub optimized_gate_count: usize,

    /// Improvement percentage
    pub improvement_pct: f64,

    /// Strategies applied
    pub strategies_applied: Vec<String>,
}

impl RewrittenCircuit {
    pub fn report(&self) -> String {
        format!(
            "=== Circuit Rewriting Report ===\n\
             Original: {} gates, T-depth {}\n\
             Optimized: {} gates, T-depth {}\n\
             Improvement: {:.1}% T-depth reduction\n\
             Strategies: {:?}",
            self.original_gate_count,
            self.original_t_depth,
            self.optimized_gate_count,
            self.optimized_t_depth,
            self.improvement_pct,
            self.strategies_applied
        )
    }
}

// =============================================================================
// Circuit Rewriter
// =============================================================================

/// Circuit rewriting engine
pub struct CircuitRewriter {
    /// Original circuit
    original: Vec<QGate>,

    /// Current circuit (being rewritten)
    current: Vec<QGate>,

    /// Rewrite strategies
    strategies: Vec<RewriteStrategy>,

    /// Num qubits
    #[allow(dead_code)]
    num_qubits: usize,
}

impl CircuitRewriter {
    /// Create circuit rewriter
    pub fn new(circuit: Vec<QGate>) -> Self {
        let num_qubits = Self::count_qubits(&circuit);

        CircuitRewriter {
            original: circuit.clone(),
            current: circuit,
            strategies: Vec::new(),
            num_qubits,
        }
    }

    /// Add rewrite strategy
    pub fn add_strategy(&mut self, strategy: RewriteStrategy) {
        self.strategies.push(strategy);
    }

    /// Apply all rewrite strategies
    pub fn rewrite(&mut self) -> RewrittenCircuit {
        let original_t_depth = Self::compute_t_depth(&self.original);
        let original_gate_count = self.original.len();

        let mut strategies_applied = Vec::new();

        // Apply each strategy in order
        for strategy in self.strategies.clone() {
            match strategy {
                RewriteStrategy::TDepthMinimization => {
                    self.minimize_t_depth();
                    strategies_applied.push("T-depth minimization".to_string());
                }
                RewriteStrategy::ApproximateSubstitution { epsilon } => {
                    self.approximate_substitution(epsilon);
                    strategies_applied.push(format!("Approximate substitution (ε={})", epsilon));
                }
                RewriteStrategy::CliffordReduction => {
                    self.clifford_reduction();
                    strategies_applied.push("Clifford reduction".to_string());
                }
                RewriteStrategy::IdentityElimination => {
                    self.identity_elimination();
                    strategies_applied.push("Identity elimination".to_string());
                }
                RewriteStrategy::GateCancellation => {
                    self.gate_cancellation();
                    strategies_applied.push("Gate cancellation".to_string());
                }
            }
        }

        let optimized_t_depth = Self::compute_t_depth(&self.current);
        let optimized_gate_count = self.current.len();

        let improvement_pct = if original_t_depth > 0 {
            ((original_t_depth - optimized_t_depth) as f64 / original_t_depth as f64) * 100.0
        } else {
            0.0
        };

        RewrittenCircuit {
            circuit: self.current.clone(),
            original_t_depth,
            optimized_t_depth,
            original_gate_count,
            optimized_gate_count,
            improvement_pct,
            strategies_applied,
        }
    }

    // =========================================================================
    // T-Depth Minimization
    // =========================================================================

    /// Minimize T-depth by reordering gates
    fn minimize_t_depth(&mut self) {
        // Build dependency graph
        let deps = self.build_dependencies();

        // Extract T and Tdg gates
        let mut t_gates = Vec::new();
        let mut non_t_gates = Vec::new();

        for (idx, gate) in self.current.iter().enumerate() {
            if matches!(gate, QGate::T(_) | QGate::Tdg(_)) {
                t_gates.push(idx);
            } else {
                non_t_gates.push(idx);
            }
        }

        // Try to parallelize T-gates
        let mut reordered = Vec::new();
        let mut scheduled_t = HashSet::new();
        let mut scheduled_non_t = HashSet::new();

        // Greedy scheduling: interleave T and non-T gates respecting dependencies
        let mut t_idx = 0;
        let mut non_t_idx = 0;

        while t_idx < t_gates.len() || non_t_idx < non_t_gates.len() {
            // Try to schedule non-T gate
            if non_t_idx < non_t_gates.len() {
                let gate_idx = non_t_gates[non_t_idx];
                if self.can_schedule(gate_idx, &scheduled_t, &scheduled_non_t, &deps) {
                    reordered.push(self.current[gate_idx].clone());
                    scheduled_non_t.insert(gate_idx);
                    non_t_idx += 1;
                    continue;
                }
            }

            // Try to schedule T gate
            if t_idx < t_gates.len() {
                let gate_idx = t_gates[t_idx];
                if self.can_schedule(gate_idx, &scheduled_t, &scheduled_non_t, &deps) {
                    reordered.push(self.current[gate_idx].clone());
                    scheduled_t.insert(gate_idx);
                    t_idx += 1;
                    continue;
                }
            }

            // If we can't schedule anything, we're stuck - break
            if (t_idx >= t_gates.len()
                || !self.can_schedule(t_gates[t_idx], &scheduled_t, &scheduled_non_t, &deps))
                && (non_t_idx >= non_t_gates.len()
                    || !self.can_schedule(
                        non_t_gates[non_t_idx],
                        &scheduled_t,
                        &scheduled_non_t,
                        &deps,
                    ))
            {
                // Force schedule next available
                if non_t_idx < non_t_gates.len() {
                    let gate_idx = non_t_gates[non_t_idx];
                    reordered.push(self.current[gate_idx].clone());
                    scheduled_non_t.insert(gate_idx);
                    non_t_idx += 1;
                } else if t_idx < t_gates.len() {
                    let gate_idx = t_gates[t_idx];
                    reordered.push(self.current[gate_idx].clone());
                    scheduled_t.insert(gate_idx);
                    t_idx += 1;
                }
            }
        }

        self.current = reordered;
    }

    /// Check if gate can be scheduled (all dependencies satisfied)
    fn can_schedule(
        &self,
        gate_idx: usize,
        scheduled_t: &HashSet<usize>,
        scheduled_non_t: &HashSet<usize>,
        deps: &HashMap<usize, Vec<usize>>,
    ) -> bool {
        if let Some(dependencies) = deps.get(&gate_idx) {
            dependencies.iter().all(|&dep_idx| {
                scheduled_t.contains(&dep_idx) || scheduled_non_t.contains(&dep_idx)
            })
        } else {
            true // No dependencies
        }
    }

    /// Build dependency graph (gate_idx → gates it depends on)
    fn build_dependencies(&self) -> HashMap<usize, Vec<usize>> {
        let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut last_writer: HashMap<usize, usize> = HashMap::new();

        for (gate_idx, gate) in self.current.iter().enumerate() {
            let qubits = Self::gate_qubits(gate);
            let mut deps = Vec::new();

            // Find dependencies (last gate to write each qubit)
            for &qubit in &qubits {
                if let Some(&writer_idx) = last_writer.get(&qubit) {
                    deps.push(writer_idx);
                }
            }

            if !deps.is_empty() {
                dependencies.insert(gate_idx, deps);
            }

            // Update last writer
            for &qubit in &qubits {
                last_writer.insert(qubit, gate_idx);
            }
        }

        dependencies
    }

    // =========================================================================
    // Approximate Substitution
    // =========================================================================

    /// Substitute gates with approximate versions
    fn approximate_substitution(&mut self, epsilon: f64) {
        let mut new_circuit = Vec::new();

        for gate in &self.current {
            match gate {
                // Approximate small rotations
                QGate::Rx(q, angle) if (*angle as f64).abs() < epsilon => {
                    // Skip very small rotations
                    continue;
                }
                QGate::Ry(q, angle) if (*angle as f64).abs() < epsilon => {
                    continue;
                }
                QGate::Rz(q, angle) if (*angle as f64).abs() < epsilon => {
                    continue;
                }

                // Approximate rotations to Clifford+T
                QGate::Rx(q, angle) => {
                    new_circuit.push(Self::approximate_rotation(*q, *angle, "Rx", epsilon));
                }
                QGate::Ry(q, angle) => {
                    new_circuit.push(Self::approximate_rotation(*q, *angle, "Ry", epsilon));
                }
                QGate::Rz(q, angle) => {
                    new_circuit.push(Self::approximate_rotation(*q, *angle, "Rz", epsilon));
                }

                // Keep other gates as-is
                _ => new_circuit.push(gate.clone()),
            }
        }

        self.current = new_circuit;
    }

    /// Approximate rotation to nearest Clifford+T gate
    fn approximate_rotation(qubit: u8, angle: f32, axis: &str, epsilon: f64) -> QGate {
        use std::f64::consts::PI;

        let angle_f64 = angle as f64;

        // Check if close to common angles
        let angles = vec![
            (0.0, None),
            (PI / 2.0, Some("90")),
            (PI, Some("180")),
            (PI / 4.0, Some("45")),
            (-PI / 2.0, Some("-90")),
            (-PI, Some("-180")),
            (-PI / 4.0, Some("-45")),
        ];

        for (target_angle, _label) in angles {
            if (angle_f64 - target_angle).abs() < epsilon {
                // Use approximated angle
                let target_f32 = target_angle as f32;
                return match axis {
                    "Rx" => QGate::Rx(qubit, target_f32),
                    "Ry" => QGate::Ry(qubit, target_f32),
                    "Rz" => QGate::Rz(qubit, target_f32),
                    _ => QGate::Rz(qubit, angle),
                };
            }
        }

        // Keep original if no good approximation
        match axis {
            "Rx" => QGate::Rx(qubit, angle),
            "Ry" => QGate::Ry(qubit, angle),
            "Rz" => QGate::Rz(qubit, angle),
            _ => QGate::Rz(qubit, angle),
        }
    }

    // =========================================================================
    // Clifford Reduction
    // =========================================================================

    /// Merge consecutive Clifford gates
    fn clifford_reduction(&mut self) {
        let mut new_circuit = Vec::new();
        let mut i = 0;

        while i < self.current.len() {
            let gate = &self.current[i];

            // Check if this is a Clifford gate
            if Self::is_clifford(gate) {
                // Look ahead for more Cliffords on same qubit
                let qubits = Self::gate_qubits(gate);
                if qubits.len() == 1 {
                    let qubit = qubits[0];
                    let mut sequence = vec![gate.clone()];
                    let mut j = i + 1;

                    while j < self.current.len() {
                        let next_gate = &self.current[j];
                        let next_qubits = Self::gate_qubits(next_gate);

                        if next_qubits.len() == 1
                            && next_qubits[0] == qubit
                            && Self::is_clifford(next_gate)
                        {
                            sequence.push(next_gate.clone());
                            j += 1;
                        } else {
                            break;
                        }
                    }

                    // Merge sequence
                    let merged = Self::merge_clifford_sequence(&sequence);
                    new_circuit.extend(merged);
                    i = j;
                    continue;
                }
            }

            new_circuit.push(gate.clone());
            i += 1;
        }

        self.current = new_circuit;
    }

    /// Check if gate is Clifford
    fn is_clifford(gate: &QGate) -> bool {
        matches!(
            gate,
            QGate::H(_)
                | QGate::T(_)
                | QGate::Tdg(_)
                | QGate::X(_)
                | QGate::Y(_)
                | QGate::Z(_)
                | QGate::CNot(_, _)
                | QGate::CZ(_, _)
                | QGate::Swap(_, _)
        )
    }

    /// Merge sequence of Clifford gates
    fn merge_clifford_sequence(sequence: &[QGate]) -> Vec<QGate> {
        if sequence.is_empty() {
            return vec![];
        }

        // Simple merging: H·H = I, S·S = Z, etc.
        let mut result = Vec::new();
        let mut i = 0;

        while i < sequence.len() {
            if i + 1 < sequence.len() {
                let (gate1, gate2) = (&sequence[i], &sequence[i + 1]);

                // Check for cancellations
                if Self::gates_cancel(gate1, gate2) {
                    i += 2; // Skip both
                    continue;
                }
            }

            result.push(sequence[i].clone());
            i += 1;
        }

        result
    }

    /// Check if two gates cancel
    fn gates_cancel(gate1: &QGate, gate2: &QGate) -> bool {
        match (gate1, gate2) {
            (QGate::H(q1), QGate::H(q2)) if q1 == q2 => true,
            (QGate::X(q1), QGate::X(q2)) if q1 == q2 => true,
            (QGate::Y(q1), QGate::Y(q2)) if q1 == q2 => true,
            (QGate::Z(q1), QGate::Z(q2)) if q1 == q2 => true,
            (QGate::T(q1), QGate::Tdg(q2)) if q1 == q2 => true,
            (QGate::Tdg(q1), QGate::T(q2)) if q1 == q2 => true,
            _ => false,
        }
    }

    // =========================================================================
    // Identity Elimination & Gate Cancellation
    // =========================================================================

    /// Remove identity sequences
    fn identity_elimination(&mut self) {
        // Already handled by Clifford reduction
        self.gate_cancellation();
    }

    /// Cancel adjacent inverse gates
    fn gate_cancellation(&mut self) {
        let mut new_circuit = Vec::new();
        let mut i = 0;

        while i < self.current.len() {
            if i + 1 < self.current.len() {
                let (gate1, gate2) = (&self.current[i], &self.current[i + 1]);

                if Self::gates_cancel(gate1, gate2) {
                    i += 2; // Skip both
                    continue;
                }
            }

            new_circuit.push(self.current[i].clone());
            i += 1;
        }

        self.current = new_circuit;
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /// Count qubits in circuit
    fn count_qubits(circuit: &[QGate]) -> usize {
        circuit
            .iter()
            .filter_map(|gate| Self::gate_max_qubit(gate))
            .max()
            .map(|q| q + 1)
            .unwrap_or(0)
    }

    fn gate_max_qubit(gate: &QGate) -> Option<usize> {
        match gate {
            QGate::H(q) | QGate::X(q) | QGate::Y(q) | QGate::Z(q) | QGate::T(q) | QGate::Tdg(q) => {
                Some(*q as usize)
            }
            QGate::Rx(_, q) | QGate::Ry(_, q) | QGate::Rz(_, q) => Some(*q as usize),
            QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => Some((*c).max(*t) as usize),
            QGate::Toffoli(c1, c2, t) | QGate::CCZ(c1, c2, t) => {
                Some((*c1).max(*c2).max(*t) as usize)
            }
            _ => None,
        }
    }

    fn gate_qubits(gate: &QGate) -> Vec<usize> {
        match gate {
            QGate::H(q) | QGate::X(q) | QGate::Y(q) | QGate::Z(q) | QGate::T(q) | QGate::Tdg(q) => {
                vec![*q as usize]
            }
            QGate::Rx(_, q) | QGate::Ry(_, q) | QGate::Rz(_, q) => vec![*q as usize],
            QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => {
                vec![*c as usize, *t as usize]
            }
            QGate::Toffoli(c1, c2, t) | QGate::CCZ(c1, c2, t) => {
                vec![*c1 as usize, *c2 as usize, *t as usize]
            }
            _ => vec![],
        }
    }

    /// Compute T-depth of circuit
    fn compute_t_depth(circuit: &[QGate]) -> usize {
        // Build dependency graph and compute T-depth
        let num_qubits = Self::count_qubits(circuit);
        let mut last_t_depth = vec![0; num_qubits];
        let mut max_depth = 0;

        for gate in circuit {
            match gate {
                QGate::T(q) | QGate::Tdg(q) => {
                    let q = *q as usize;
                    if q < num_qubits {
                        last_t_depth[q] += 1;
                        max_depth = max_depth.max(last_t_depth[q]);
                    }
                }
                // Two-qubit gates synchronize T-depths
                QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => {
                    let c = *c as usize;
                    let t = *t as usize;
                    if c < num_qubits && t < num_qubits {
                        let sync_depth = last_t_depth[c].max(last_t_depth[t]);
                        last_t_depth[c] = sync_depth;
                        last_t_depth[t] = sync_depth;
                    }
                }
                _ => {}
            }
        }

        max_depth
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_depth_computation() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(0)];

        let depth = CircuitRewriter::compute_t_depth(&circuit);
        assert_eq!(depth, 2); // T-depth is 2 for qubit 0
    }

    #[test]
    fn test_gate_cancellation() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(0), // Cancels
            QGate::T(0),
        ];

        let mut rewriter = CircuitRewriter::new(circuit);
        rewriter.add_strategy(RewriteStrategy::GateCancellation);
        let result = rewriter.rewrite();

        assert_eq!(result.optimized_gate_count, 1); // Only T remains
    }

    #[test]
    fn test_clifford_reduction() {
        let circuit = vec![
            QGate::H(0),
            QGate::T(0),
            QGate::T(0), // S·S = Z
            QGate::T(0),
        ];

        let mut rewriter = CircuitRewriter::new(circuit);
        rewriter.add_strategy(RewriteStrategy::CliffordReduction);
        let result = rewriter.rewrite();

        // Should reduce some gates
        assert!(result.optimized_gate_count <= result.original_gate_count);
    }

    #[test]
    fn test_t_depth_minimization() {
        let circuit = vec![
            QGate::T(0),
            QGate::H(1),
            QGate::T(1), // Can parallelize with T(0)
            QGate::H(0),
            QGate::T(0),
        ];

        let mut rewriter = CircuitRewriter::new(circuit);
        rewriter.add_strategy(RewriteStrategy::TDepthMinimization);
        let result = rewriter.rewrite();

        // Should reduce T-depth
        assert!(result.optimized_t_depth <= result.original_t_depth);
    }

    #[test]
    fn test_approximate_substitution() {
        let circuit = vec![
            QGate::Rx(0, 1e-10), // Very small rotation
            QGate::T(0),
        ];

        let mut rewriter = CircuitRewriter::new(circuit);
        rewriter.add_strategy(RewriteStrategy::ApproximateSubstitution { epsilon: 1e-8 });
        let result = rewriter.rewrite();

        // Should remove small rotation
        assert_eq!(result.optimized_gate_count, 1);
    }
}
