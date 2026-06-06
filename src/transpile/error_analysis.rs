// =============================================================================
// Error Analysis Engine - Quantum Error Propagation and Criticality Analysis
// =============================================================================
// Sprint 69.0: Error-Aware Circuit Compilation
//
// Analyzes how errors propagate through quantum circuits and identifies
// gates where errors have the most impact on algorithm success.

use crate::quantum::QGate;
use std::collections::HashMap;

// =============================================================================
// Error Models
// =============================================================================

/// Quantum error model
#[derive(Clone, Debug)]
pub enum ErrorModel {
    /// Depolarizing channel: ρ → (1-p)ρ + p·I/d
    Depolarizing { error_rate: f64 },

    /// Amplitude damping: T₁ relaxation
    AmplitudeDamping { gamma: f64 },

    /// Phase damping: T₂ dephasing
    PhaseDamping { gamma: f64 },

    /// Combined model
    Combined {
        depolarizing: f64,
        amplitude_damping: f64,
        phase_damping: f64,
    },
}

impl ErrorModel {
    /// Create depolarizing error model
    pub fn depolarizing(error_rate: f64) -> Self {
        ErrorModel::Depolarizing { error_rate }
    }

    /// Create amplitude damping model
    pub fn amplitude_damping(gamma: f64) -> Self {
        ErrorModel::AmplitudeDamping { gamma }
    }

    /// Create phase damping model
    pub fn phase_damping(gamma: f64) -> Self {
        ErrorModel::PhaseDamping { gamma }
    }

    /// Get effective error rate for single-qubit gate
    pub fn single_qubit_error(&self) -> f64 {
        match self {
            ErrorModel::Depolarizing { error_rate } => *error_rate,
            ErrorModel::AmplitudeDamping { gamma } => *gamma,
            ErrorModel::PhaseDamping { gamma } => *gamma,
            ErrorModel::Combined {
                depolarizing,
                amplitude_damping,
                phase_damping,
            } => {
                // Combined error (simplified additive model)
                depolarizing + amplitude_damping + phase_damping
            }
        }
    }

    /// Get effective error rate for two-qubit gate (typically higher)
    pub fn two_qubit_error(&self) -> f64 {
        // Two-qubit gates typically have 10-100× higher error rates
        self.single_qubit_error() * 10.0
    }
}

// =============================================================================
// Error Propagation
// =============================================================================

/// Error propagation through circuit
#[derive(Clone, Debug)]
pub struct ErrorPropagation {
    /// Number of qubits
    pub num_qubits: usize,

    /// Number of layers (circuit depth)
    pub num_layers: usize,

    /// Error probability per qubit per layer
    /// error_state[layer][qubit] = error probability
    pub error_state: Vec<Vec<f64>>,

    /// Final error probability (algorithm output error)
    pub final_error: f64,

    /// Total error accumulated
    pub total_error: f64,

    /// Error by gate type
    pub error_by_gate_type: HashMap<String, f64>,
}

impl ErrorPropagation {
    /// Generate report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=== Error Propagation Analysis ===\n".to_string());
        lines.push(format!(
            "Circuit: {} qubits, {} layers",
            self.num_qubits, self.num_layers
        ));
        lines.push(format!("Final error: {:.2e}", self.final_error));
        lines.push(format!("Total error accumulated: {:.2e}", self.total_error));
        lines.push(String::new());

        lines.push("Error by Gate Type:".to_string());
        let mut sorted_gates: Vec<_> = self.error_by_gate_type.iter().collect();
        sorted_gates.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (gate_type, error) in sorted_gates {
            lines.push(format!("  {}: {:.2e}", gate_type, error));
        }

        lines.join("\n")
    }
}

// =============================================================================
// Gate Criticality
// =============================================================================

/// Criticality score for a gate
#[derive(Clone, Debug)]
pub struct GateCriticality {
    /// Gate ID in circuit
    pub gate_id: usize,

    /// Gate type
    pub gate_type: String,

    /// Criticality score: 0.0 (unimportant) to 1.0 (critical)
    pub criticality_score: f64,

    /// Error amplification factor
    pub error_amplification: f64,

    /// Number of gates that depend on this gate's output
    pub fanout: usize,

    /// Depth from circuit end (reverse topological distance)
    pub depth_from_end: usize,
}

impl GateCriticality {
    /// Check if gate is critical (score > 0.7)
    pub fn is_critical(&self) -> bool {
        self.criticality_score > 0.7
    }

    /// Check if gate is moderately critical (0.3 < score < 0.7)
    pub fn is_moderate(&self) -> bool {
        self.criticality_score > 0.3 && self.criticality_score <= 0.7
    }

    /// Check if gate is non-critical (score < 0.3)
    pub fn is_non_critical(&self) -> bool {
        self.criticality_score <= 0.3
    }
}

// =============================================================================
// Error Analyzer
// =============================================================================

/// Analyzes error propagation through quantum circuits
pub struct ErrorAnalyzer {
    /// Circuit to analyze
    circuit: Vec<QGate>,

    /// Error model
    error_model: ErrorModel,

    /// Number of qubits
    num_qubits: usize,

    /// Error state (per qubit per layer)
    error_state: Vec<Vec<f64>>,

    /// Gate criticality scores
    criticality: Vec<GateCriticality>,
}

impl ErrorAnalyzer {
    /// Create error analyzer
    pub fn new(circuit: Vec<QGate>, error_model: ErrorModel) -> Self {
        let num_qubits = Self::count_qubits(&circuit);

        ErrorAnalyzer {
            circuit,
            error_model,
            num_qubits,
            error_state: Vec::new(),
            criticality: Vec::new(),
        }
    }

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

    /// Propagate errors through circuit
    pub fn propagate_errors(&mut self, initial_error: f64) -> ErrorPropagation {
        // Initialize error state
        let mut current_error = vec![initial_error; self.num_qubits];
        let mut layer_errors = Vec::new();
        layer_errors.push(current_error.clone());

        let mut error_by_gate_type: HashMap<String, f64> = HashMap::new();
        let mut total_error = 0.0;

        // Propagate through each gate
        for gate in &self.circuit {
            let gate_error = self.apply_gate_error(&mut current_error, gate);
            total_error += gate_error;

            // Track error by gate type
            let gate_name = Self::gate_name(gate);
            *error_by_gate_type.entry(gate_name).or_insert(0.0) += gate_error;

            layer_errors.push(current_error.clone());
        }

        // Final error is max error across all qubits
        let final_error = current_error.iter().copied().fold(0.0, f64::max);

        self.error_state = layer_errors.clone();

        ErrorPropagation {
            num_qubits: self.num_qubits,
            num_layers: layer_errors.len(),
            error_state: layer_errors,
            final_error,
            total_error,
            error_by_gate_type,
        }
    }

    /// Apply error from gate to current error state
    fn apply_gate_error(&self, current_error: &mut [f64], gate: &QGate) -> f64 {
        match gate {
            // Single-qubit gates
            QGate::H(q) | QGate::X(q) | QGate::Y(q) | QGate::Z(q) | QGate::T(q) | QGate::Tdg(q) => {
                let q = *q as usize;
                if q < current_error.len() {
                    let gate_error = self.error_model.single_qubit_error();
                    current_error[q] = self.combine_errors(current_error[q], gate_error);
                    gate_error
                } else {
                    0.0
                }
            }

            // Rotation gates
            QGate::Rx(_, q) | QGate::Ry(_, q) | QGate::Rz(_, q) => {
                let q = *q as usize;
                if q < current_error.len() {
                    let gate_error = self.error_model.single_qubit_error();
                    current_error[q] = self.combine_errors(current_error[q], gate_error);
                    gate_error
                } else {
                    0.0
                }
            }

            // Two-qubit gates
            QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => {
                let c = *c as usize;
                let t = *t as usize;
                let gate_error = self.error_model.two_qubit_error();

                if c < current_error.len() {
                    current_error[c] = self.combine_errors(current_error[c], gate_error);
                }
                if t < current_error.len() {
                    current_error[t] = self.combine_errors(current_error[t], gate_error);
                }

                // Two-qubit gates create correlations - simplified model
                gate_error * 2.0
            }

            // Three-qubit gates
            QGate::Toffoli(c1, c2, t) | QGate::CCZ(c1, c2, t) => {
                let c1 = *c1 as usize;
                let c2 = *c2 as usize;
                let t = *t as usize;
                let gate_error = self.error_model.two_qubit_error() * 2.0; // Worse for 3-qubit

                if c1 < current_error.len() {
                    current_error[c1] = self.combine_errors(current_error[c1], gate_error);
                }
                if c2 < current_error.len() {
                    current_error[c2] = self.combine_errors(current_error[c2], gate_error);
                }
                if t < current_error.len() {
                    current_error[t] = self.combine_errors(current_error[t], gate_error);
                }

                gate_error * 3.0
            }

            _ => 0.0,
        }
    }

    /// Combine two error probabilities (simplified model: 1 - (1-p1)(1-p2))
    fn combine_errors(&self, error1: f64, error2: f64) -> f64 {
        1.0 - (1.0 - error1) * (1.0 - error2)
    }

    /// Get gate name for tracking
    fn gate_name(gate: &QGate) -> String {
        match gate {
            QGate::H(_) => "H".to_string(),
            QGate::X(_) => "X".to_string(),
            QGate::Y(_) => "Y".to_string(),
            QGate::Z(_) => "Z".to_string(),
            QGate::T(_) => "T".to_string(),
            QGate::Tdg(_) => "Tdg".to_string(),
            QGate::Rx(_, _) => "Rx".to_string(),
            QGate::Ry(_, _) => "Ry".to_string(),
            QGate::Rz(_, _) => "Rz".to_string(),
            QGate::CNot(_, _) => "CNot".to_string(),
            QGate::CZ(_, _) => "CZ".to_string(),
            QGate::Swap(_, _) => "Swap".to_string(),
            QGate::Toffoli(_, _, _) => "Toffoli".to_string(),
            QGate::CCZ(_, _, _) => "CCZ".to_string(),
            _ => "Other".to_string(),
        }
    }

    /// Identify critical gates based on error amplification
    pub fn identify_critical_gates(&mut self) -> Vec<GateCriticality> {
        if self.error_state.is_empty() {
            // Need to propagate errors first
            self.propagate_errors(1e-10);
        }

        let mut criticality = Vec::new();

        // Build dependency graph (which gates depend on which)
        let dependencies = self.build_dependency_graph();

        // Analyze each gate
        for (gate_id, gate) in self.circuit.iter().enumerate() {
            let gate_type = Self::gate_name(gate);

            // Calculate error amplification
            let error_before = if gate_id > 0 {
                self.get_gate_error_state(gate_id - 1, gate)
            } else {
                1e-10
            };

            let error_after = self.get_gate_error_state(gate_id, gate);
            let error_amplification = if error_before > 0.0 {
                error_after / error_before
            } else {
                1.0
            };

            // Calculate fanout (how many gates depend on this)
            let fanout = dependencies.get(&gate_id).map(|v| v.len()).unwrap_or(0);

            // Calculate depth from end
            let depth_from_end = self.circuit.len() - gate_id;

            // Compute criticality score
            // Factors: error amplification, fanout, depth from end
            let amp_score = (error_amplification - 1.0).max(0.0).min(1.0);
            let fanout_score = (fanout as f64 / 10.0).min(1.0);
            let depth_score = (depth_from_end as f64 / self.circuit.len() as f64).min(1.0);

            // Weighted combination
            let criticality_score = amp_score * 0.5 + fanout_score * 0.3 + depth_score * 0.2;

            criticality.push(GateCriticality {
                gate_id,
                gate_type,
                criticality_score,
                error_amplification,
                fanout,
                depth_from_end,
            });
        }

        self.criticality = criticality.clone();
        criticality
    }

    /// Get error state for gate's qubits
    fn get_gate_error_state(&self, gate_id: usize, gate: &QGate) -> f64 {
        if gate_id >= self.error_state.len() {
            return 0.0;
        }

        let state = &self.error_state[gate_id];
        let qubits = Self::gate_qubits(gate);

        qubits
            .iter()
            .filter_map(|&q| state.get(q))
            .copied()
            .fold(0.0, f64::max)
    }

    /// Get qubits involved in gate
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

    /// Build dependency graph (gate_id → dependent gate IDs)
    fn build_dependency_graph(&self) -> HashMap<usize, Vec<usize>> {
        let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new();

        // Track last gate to write each qubit
        let mut last_writer: HashMap<usize, usize> = HashMap::new();

        for (gate_id, gate) in self.circuit.iter().enumerate() {
            let qubits = Self::gate_qubits(gate);

            // For each qubit this gate uses, find last writer
            for &qubit in &qubits {
                if let Some(&writer_id) = last_writer.get(&qubit) {
                    dependencies.entry(writer_id).or_default().push(gate_id);
                }
            }

            // Update last writer for all qubits
            for &qubit in &qubits {
                last_writer.insert(qubit, gate_id);
            }
        }

        dependencies
    }

    /// Compute error margins (budget - actual)
    pub fn compute_margins(&self, target_error: f64) -> ErrorMargins {
        if self.error_state.is_empty() {
            return ErrorMargins {
                target_error,
                actual_error: 0.0,
                margin: target_error,
                margin_pct: 100.0,
                within_budget: true,
            };
        }

        let actual_error = self
            .error_state
            .last()
            .unwrap()
            .iter()
            .copied()
            .fold(0.0, f64::max);

        let margin = target_error - actual_error;
        let margin_pct = if target_error > 0.0 {
            (margin / target_error) * 100.0
        } else {
            0.0
        };

        ErrorMargins {
            target_error,
            actual_error,
            margin,
            margin_pct,
            within_budget: actual_error <= target_error,
        }
    }
}

// =============================================================================
// Error Margins
// =============================================================================

/// Error budget margins
#[derive(Clone, Debug)]
pub struct ErrorMargins {
    /// Target error budget
    pub target_error: f64,

    /// Actual error
    pub actual_error: f64,

    /// Margin (target - actual)
    pub margin: f64,

    /// Margin as percentage
    pub margin_pct: f64,

    /// Within budget?
    pub within_budget: bool,
}

impl ErrorMargins {
    pub fn report(&self) -> String {
        format!(
            "Error Budget: {:.2e} (target) vs {:.2e} (actual)\n\
             Margin: {:.2e} ({:.1}%)\n\
             Status: {}",
            self.target_error,
            self.actual_error,
            self.margin,
            self.margin_pct,
            if self.within_budget {
                "✓ Within budget"
            } else {
                "✗ EXCEEDED budget"
            }
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_model() {
        let model = ErrorModel::depolarizing(1e-4);
        assert_eq!(model.single_qubit_error(), 1e-4);
        assert_eq!(model.two_qubit_error(), 1e-3);
    }

    #[test]
    fn test_error_propagation() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::T(0), QGate::T(1)];

        let model = ErrorModel::depolarizing(1e-4);
        let mut analyzer = ErrorAnalyzer::new(circuit, model);

        let propagation = analyzer.propagate_errors(1e-10);

        assert!(propagation.final_error > 1e-10);
        assert!(propagation.final_error < 1e-2);
        assert_eq!(propagation.num_qubits, 2);
    }

    #[test]
    fn test_critical_gates() {
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::CNot(0, 1),
            QGate::T(0),
        ];

        let model = ErrorModel::depolarizing(1e-4);
        let mut analyzer = ErrorAnalyzer::new(circuit, model);

        let criticality = analyzer.identify_critical_gates();

        assert_eq!(criticality.len(), 5);

        // Gates later in circuit should have higher depth_from_end
        assert!(criticality[4].depth_from_end < criticality[0].depth_from_end);
    }

    #[test]
    fn test_error_margins() {
        let circuit = vec![QGate::H(0), QGate::T(0)];

        let model = ErrorModel::depolarizing(1e-4);
        let mut analyzer = ErrorAnalyzer::new(circuit, model);

        analyzer.propagate_errors(1e-10);
        let margins = analyzer.compute_margins(1e-3);

        assert!(margins.within_budget);
        assert!(margins.margin > 0.0);
        assert!(margins.margin_pct > 0.0);
    }

    #[test]
    fn test_combine_errors() {
        let circuit = vec![];
        let model = ErrorModel::depolarizing(1e-4);
        let analyzer = ErrorAnalyzer::new(circuit, model);

        // Test error combination
        let combined = analyzer.combine_errors(0.1, 0.1);
        assert!((combined - 0.19).abs() < 1e-10); // 1 - 0.9 * 0.9 = 0.19
    }

    #[test]
    fn test_dependency_graph() {
        let circuit = vec![QGate::H(0), QGate::T(0), QGate::CNot(0, 1), QGate::T(1)];

        let model = ErrorModel::depolarizing(1e-4);
        let analyzer = ErrorAnalyzer::new(circuit, model);

        let deps = analyzer.build_dependency_graph();

        // Gate 0 (H on 0) should be depended on by gate 1 (T on 0)
        assert!(deps.get(&0).unwrap().contains(&1));

        // Gate 2 (CNot 0,1) should be depended on by gate 3 (T on 1)
        assert!(deps.get(&2).unwrap().contains(&3));
    }
}
