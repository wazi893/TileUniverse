//! T-Gate Counting and Analysis
//!
//! SPRINT 53.0: Analyze T-gate content for magic state budgeting
//!
//! T-gates (and T†) are the only non-Clifford gates in the Clifford+T
//! universal gate set. Each T-gate requires a magic state for fault-tolerant
//! execution, making T-count a critical resource metric.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use crate::qram::{DistillationConfig, MagicStateBudget};
use logic_fabric_core::quantum::QGate;

/// T-gate analysis result
#[derive(Clone, Debug, Default)]
pub struct TGateAnalysis {
    /// Total number of T and T† gates
    pub t_count: usize,
    /// T-depth (T-gates on critical path)
    pub t_depth: usize,
    /// Number of Clifford gates
    pub clifford_count: usize,
    /// Total gate count
    pub total_gates: usize,
    /// Positions of T-gates in circuit
    pub t_positions: Vec<usize>,
    /// Breakdown by gate type
    pub gate_counts: HashMap<String, usize>,
    /// Qubits that have T-gates
    pub t_qubits: HashSet<usize>,
}

impl TGateAnalysis {
    /// Ratio of T-gates to total gates
    pub fn t_ratio(&self) -> f64 {
        if self.total_gates == 0 {
            0.0
        } else {
            self.t_count as f64 / self.total_gates as f64
        }
    }

    /// Check if circuit is Clifford-only (no T-gates)
    pub fn is_clifford(&self) -> bool {
        self.t_count == 0
    }

    /// Estimated magic states needed (1:1 without distillation overhead)
    pub fn raw_magic_states(&self) -> usize {
        self.t_count
    }
}

/// Analyze a circuit for T-gate content
pub fn analyze_t_gates(circuit: &[QGate]) -> TGateAnalysis {
    let mut analysis = TGateAnalysis {
        total_gates: circuit.len(),
        ..Default::default()
    };

    // Track qubit depths for T-depth calculation
    let mut qubit_t_depths: HashMap<usize, usize> = HashMap::new();

    for (idx, gate) in circuit.iter().enumerate() {
        let gate_name = gate_type_name(gate);
        *analysis
            .gate_counts
            .entry(gate_name.to_string())
            .or_insert(0) += 1;

        if is_t_gate(gate) {
            analysis.t_count += 1;
            analysis.t_positions.push(idx);

            // Track T-qubit
            if let Some(q) = get_gate_qubit(gate) {
                analysis.t_qubits.insert(q);

                // Update T-depth
                let current_depth = qubit_t_depths.get(&q).copied().unwrap_or(0);
                let new_depth = current_depth + 1;
                qubit_t_depths.insert(q, new_depth);
            }
        } else {
            analysis.clifford_count += 1;

            // For 2-qubit gates, synchronize depths
            if let Some((q1, q2)) = get_two_qubit_operands(gate) {
                let d1 = qubit_t_depths.get(&q1).copied().unwrap_or(0);
                let d2 = qubit_t_depths.get(&q2).copied().unwrap_or(0);
                let max_d = d1.max(d2);
                qubit_t_depths.insert(q1, max_d);
                qubit_t_depths.insert(q2, max_d);
            }
        }
    }

    // T-depth is maximum across all qubits
    analysis.t_depth = qubit_t_depths.values().copied().max().unwrap_or(0);

    analysis
}

/// Check if a gate is a T-gate (or T†)
fn is_t_gate(gate: &QGate) -> bool {
    match gate {
        QGate::T(_) | QGate::Tdg(_) => true,
        // Rz(π/4) is equivalent to T
        // Use 1e-6 tolerance to accommodate f32 to f64 precision loss
        QGate::Rz(_, angle) => {
            let a = *angle as f64;
            (a - PI / 4.0).abs() < 1e-6 || (a + PI / 4.0).abs() < 1e-6
        }
        // Phase(π/4) is also T
        QGate::Phase(_, angle) => {
            let a = *angle as f64;
            (a - PI / 4.0).abs() < 1e-6 || (a + PI / 4.0).abs() < 1e-6
        }
        _ => false,
    }
}

/// Get the qubit for single-qubit gates
fn get_gate_qubit(gate: &QGate) -> Option<usize> {
    match gate {
        QGate::T(q)
        | QGate::Tdg(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::H(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::Phase(q, _) => Some(*q as usize),
        QGate::U3(q, _, _, _) => Some(*q as usize),
        _ => None,
    }
}

/// Get operands for two-qubit gates
fn get_two_qubit_operands(gate: &QGate) -> Option<(usize, usize)> {
    match gate {
        QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => Some((*c as usize, *t as usize)),
        QGate::CPhase(c, t, _) | QGate::CRz(c, t, _) => Some((*c as usize, *t as usize)),
        _ => None,
    }
}

/// Get gate type name for statistics
fn gate_type_name(gate: &QGate) -> &'static str {
    match gate {
        QGate::X(_) => "X",
        QGate::Y(_) => "Y",
        QGate::Z(_) => "Z",
        QGate::H(_) => "H",
        QGate::T(_) => "T",
        QGate::Tdg(_) => "Tdg",
        QGate::Rx(_, _) => "Rx",
        QGate::Ry(_, _) => "Ry",
        QGate::Rz(_, _) => "Rz",
        QGate::Phase(_, _) => "Phase",
        QGate::U3(_, _, _, _) => "U3",
        QGate::CNot(_, _) => "CNot",
        QGate::CZ(_, _) => "CZ",
        QGate::Swap(_, _) => "Swap",
        QGate::CPhase(_, _, _) => "CPhase",
        QGate::CRz(_, _, _) => "CRz",
        QGate::Toffoli(_, _, _) => "Toffoli",
        QGate::CCZ(_, _, _) => "CCZ",
        QGate::Measure(_) => "Measure",
    }
}

/// Estimate magic state cost from T-gate analysis
///
/// Connects T-count to Sprint 51's magic state budgeting
pub fn estimate_magic_cost(
    analysis: &TGateAnalysis,
    distillation: &DistillationConfig,
) -> MagicStateCost {
    // Each T-gate requires one magic state
    let raw_magic_states = analysis.t_count;

    // Apply distillation overhead using the config's overhead_ratio
    let overhead = distillation.overhead_ratio();
    let distilled_states = raw_magic_states * overhead;

    // Factory cycles based on T-depth (parallelism)
    let factory_cycles = if analysis.t_depth > 0 {
        analysis.t_count.div_ceil(analysis.t_depth)
    } else {
        0
    };

    // Get protocol name
    let protocol_name = format!("{:?}", distillation.protocol);

    MagicStateCost {
        t_count: analysis.t_count,
        t_depth: analysis.t_depth,
        raw_magic_states,
        distilled_magic_states: distilled_states,
        distillation_input_states: distilled_states,
        factory_cycles,
        distillation_protocol: protocol_name,
    }
}

/// Magic state cost breakdown
#[derive(Clone, Debug)]
pub struct MagicStateCost {
    /// Number of T-gates
    pub t_count: usize,
    /// T-depth (critical path)
    pub t_depth: usize,
    /// Raw magic states needed (= t_count)
    pub raw_magic_states: usize,
    /// Magic states after distillation
    pub distilled_magic_states: usize,
    /// Input states for distillation
    pub distillation_input_states: usize,
    /// Factory cycles needed
    pub factory_cycles: usize,
    /// Distillation protocol used
    pub distillation_protocol: String,
}

impl MagicStateCost {
    /// Convert to Sprint 51 MagicStateBudget
    pub fn to_budget(&self, distillation: &DistillationConfig) -> MagicStateBudget {
        MagicStateBudget::from_gate_counts(0, self.t_count, 0).with_distillation(distillation)
    }
}

impl std::fmt::Display for MagicStateCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Magic State Cost ===")?;
        writeln!(f, "T-count: {}", self.t_count)?;
        writeln!(f, "T-depth: {}", self.t_depth)?;
        writeln!(f, "Raw magic states: {}", self.raw_magic_states)?;
        writeln!(
            f,
            "Distilled states: {} ({})",
            self.distilled_magic_states, self.distillation_protocol
        )?;
        writeln!(f, "Input states: {}", self.distillation_input_states)?;
        writeln!(f, "Factory cycles: {}", self.factory_cycles)
    }
}

impl std::fmt::Display for TGateAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== T-Gate Analysis ===")?;
        writeln!(f, "Total gates: {}", self.total_gates)?;
        writeln!(f, "T-count: {}", self.t_count)?;
        writeln!(f, "T-depth: {}", self.t_depth)?;
        writeln!(f, "Clifford gates: {}", self.clifford_count)?;
        writeln!(f, "T-ratio: {:.2}%", self.t_ratio() * 100.0)?;
        writeln!(f, "Clifford-only: {}", self.is_clifford())?;

        if !self.gate_counts.is_empty() {
            writeln!(f, "\nGate breakdown:")?;
            let mut counts: Vec<_> = self.gate_counts.iter().collect();
            counts.sort_by(|a, b| b.1.cmp(a.1));
            for (gate, count) in counts {
                writeln!(f, "  {}: {}", gate, count)?;
            }
        }

        Ok(())
    }
}

/// Analyze T-gates in a Toffoli decomposition
///
/// Standard Toffoli decomposition uses 7 T-gates
pub fn toffoli_t_count() -> usize {
    7
}

/// Analyze T-gates in a CCZ decomposition
///
/// CCZ = H(c) · Toffoli · H(c), same T-count as Toffoli
pub fn ccz_t_count() -> usize {
    7
}

/// Estimate T-count for multi-controlled Z gate
///
/// MCZ on n controls uses approximately 8(n-1) T-gates with ancillas
pub fn mcz_t_count(num_controls: usize) -> usize {
    if num_controls <= 1 {
        0 // CZ is Clifford
    } else if num_controls == 2 {
        7 // CCZ
    } else {
        8 * (num_controls - 1) // Approximate for MCZ with ancillas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_clifford_circuit() {
        // Use Phase(π/2) for S gate equivalent
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::Phase(0, std::f32::consts::FRAC_PI_2), // S = Phase(π/2)
            QGate::X(1),
        ];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.t_count, 0);
        assert!(analysis.is_clifford());
        assert_eq!(analysis.clifford_count, 4);
    }

    #[test]
    fn test_analyze_t_gates() {
        let circuit = vec![
            QGate::H(0),
            QGate::T(0),
            QGate::CNot(0, 1),
            QGate::T(1),
            QGate::Tdg(0),
        ];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.t_count, 3);
        assert_eq!(analysis.t_depth, 2); // T(0), Tdg(0) sequential; T(1) parallel
        assert!(!analysis.is_clifford());
    }

    #[test]
    fn test_t_depth_calculation() {
        // All T-gates on same qubit - sequential
        let circuit = vec![QGate::T(0), QGate::T(0), QGate::T(0)];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.t_count, 3);
        assert_eq!(analysis.t_depth, 3);
    }

    #[test]
    fn test_t_depth_parallel() {
        // T-gates on different qubits - parallel
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2)];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.t_count, 3);
        assert_eq!(analysis.t_depth, 1); // All parallel
    }

    #[test]
    fn test_rz_pi4_is_t() {
        let circuit = vec![
            QGate::Rz(0, (PI / 4.0) as f32),
            QGate::Rz(1, (PI / 2.0) as f32), // S gate equivalent, not T
        ];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.t_count, 1);
    }

    #[test]
    fn test_magic_cost() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::CNot(0, 1), QGate::T(0)];
        let analysis = analyze_t_gates(&circuit);

        let distillation = DistillationConfig::for_error_rate(1e-10);
        let cost = estimate_magic_cost(&analysis, &distillation);

        assert_eq!(cost.t_count, 3);
        assert_eq!(cost.raw_magic_states, 3);
        assert!(cost.distilled_magic_states >= cost.raw_magic_states);
    }

    #[test]
    fn test_toffoli_t_count() {
        assert_eq!(toffoli_t_count(), 7);
    }

    #[test]
    fn test_mcz_t_count() {
        assert_eq!(mcz_t_count(1), 0); // CZ is Clifford
        assert_eq!(mcz_t_count(2), 7); // CCZ
        assert_eq!(mcz_t_count(3), 16); // 3-controlled Z
    }

    #[test]
    fn test_gate_breakdown() {
        let circuit = vec![QGate::H(0), QGate::H(1), QGate::T(0), QGate::CNot(0, 1)];
        let analysis = analyze_t_gates(&circuit);

        assert_eq!(analysis.gate_counts.get("H"), Some(&2));
        assert_eq!(analysis.gate_counts.get("T"), Some(&1));
        assert_eq!(analysis.gate_counts.get("CNot"), Some(&1));
    }
}
