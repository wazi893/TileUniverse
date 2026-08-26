//! Circuit Analysis
//!
//! SPRINT 53.0: Analyze transpiled circuits for resources and fidelity
//!
//! Provides comprehensive analysis of transpiled circuits including:
//! - Gate counts and depth
//! - Timing estimates
//! - Fidelity estimation
//! - Resource comparison across hardware targets

use std::collections::HashMap;
use std::fmt;

use super::native_gates::{IBMCircuit, IonQCircuit};
use super::t_count::{TGateAnalysis, analyze_t_gates};
use crate::hardware::HardwareProfile;
use logic_fabric_core::quantum::QGate;

/// Comprehensive circuit analysis
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    /// Circuit name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Total gate count
    pub total_gates: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate counts by type
    pub gate_counts: HashMap<String, usize>,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
    /// Single-qubit gate count
    pub single_qubit_gates: usize,
    /// T-gate analysis
    pub t_analysis: TGateAnalysis,
    /// Estimated duration in nanoseconds
    pub duration_ns: f64,
    /// Estimated fidelity
    pub estimated_fidelity: f64,
    /// Fidelity breakdown
    pub fidelity_breakdown: FidelityBreakdown,
}

/// Fidelity breakdown by error source
#[derive(Clone, Debug, Default)]
pub struct FidelityBreakdown {
    /// Error from single-qubit gates
    pub single_qubit_error: f64,
    /// Error from two-qubit gates
    pub two_qubit_error: f64,
    /// Error from decoherence
    pub decoherence_error: f64,
    /// Error from measurement
    pub measurement_error: f64,
}

impl FidelityBreakdown {
    /// Total error (multiplicative model)
    pub fn total_error(&self) -> f64 {
        let cap = |e: f64| e.min(0.999);

        let fidelity = (1.0 - cap(self.single_qubit_error))
            * (1.0 - cap(self.two_qubit_error))
            * (1.0 - cap(self.decoherence_error))
            * (1.0 - cap(self.measurement_error));

        (1.0 - fidelity).min(0.9999)
    }

    /// Dominant error source
    pub fn dominant(&self) -> &'static str {
        let errors = [
            (self.single_qubit_error, "1Q gates"),
            (self.two_qubit_error, "2Q gates"),
            (self.decoherence_error, "Decoherence"),
            (self.measurement_error, "Measurement"),
        ];

        errors
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, name)| *name)
            .unwrap_or("Unknown")
    }
}

/// Analyze an IBM transpiled circuit
pub fn analyze_ibm_circuit(circuit: &IBMCircuit, profile: &HardwareProfile) -> CircuitAnalysis {
    let mut gate_counts = HashMap::new();
    let mut two_qubit_gates = 0;
    let mut single_qubit_gates = 0;

    for gate in &circuit.gates {
        *gate_counts.entry(gate.name().to_string()).or_insert(0) += 1;

        if gate.qubit_count() == 2 {
            two_qubit_gates += 1;
        } else if gate.qubit_count() == 1 {
            single_qubit_gates += 1;
        }
    }

    // Calculate fidelity breakdown
    let single_qubit_error = compute_gate_error(single_qubit_gates, profile.t1q_fidelity);
    let two_qubit_error = compute_gate_error(two_qubit_gates, profile.t2q_fidelity);

    let duration_ns = circuit.total_duration_ns();
    let t2_ns = profile.t2_us * 1000.0;
    let decoherence_error = if t2_ns > 0.0 {
        1.0 - (-duration_ns / t2_ns).exp()
    } else {
        0.0
    };

    let measure_count = circuit.count("measure");
    let measurement_error = compute_gate_error(measure_count, profile.measure_fidelity);

    let fidelity_breakdown = FidelityBreakdown {
        single_qubit_error,
        two_qubit_error,
        decoherence_error,
        measurement_error,
    };

    let estimated_fidelity = 1.0 - fidelity_breakdown.total_error();

    CircuitAnalysis {
        name: circuit.name.clone(),
        num_qubits: circuit.num_qubits,
        total_gates: circuit.gate_count(),
        depth: circuit.depth(),
        gate_counts,
        two_qubit_gates,
        single_qubit_gates,
        t_analysis: TGateAnalysis::default(), // IBM circuit doesn't have T directly
        duration_ns,
        estimated_fidelity,
        fidelity_breakdown,
    }
}

/// Analyze an IonQ transpiled circuit
pub fn analyze_ionq_circuit(circuit: &IonQCircuit, profile: &HardwareProfile) -> CircuitAnalysis {
    let mut gate_counts = HashMap::new();
    let mut two_qubit_gates = 0;
    let mut single_qubit_gates = 0;

    for gate in &circuit.gates {
        *gate_counts.entry(gate.name().to_string()).or_insert(0) += 1;

        if gate.qubit_count() == 2 {
            two_qubit_gates += 1;
        } else if gate.qubit_count() == 1 {
            single_qubit_gates += 1;
        }
    }

    // Calculate fidelity breakdown
    let single_qubit_error = compute_gate_error(single_qubit_gates, profile.t1q_fidelity);
    let two_qubit_error = compute_gate_error(two_qubit_gates, profile.t2q_fidelity);

    let duration_ns = circuit.total_duration_ns();
    let t2_ns = profile.t2_us * 1000.0;
    let decoherence_error = if t2_ns > 0.0 {
        1.0 - (-duration_ns / t2_ns).exp()
    } else {
        0.0
    };

    let measure_count = circuit.count("measure");
    let measurement_error = compute_gate_error(measure_count, profile.measure_fidelity);

    let fidelity_breakdown = FidelityBreakdown {
        single_qubit_error,
        two_qubit_error,
        decoherence_error,
        measurement_error,
    };

    let estimated_fidelity = 1.0 - fidelity_breakdown.total_error();

    CircuitAnalysis {
        name: circuit.name.clone(),
        num_qubits: circuit.num_qubits,
        total_gates: circuit.gate_count(),
        depth: circuit.depth(),
        gate_counts,
        two_qubit_gates,
        single_qubit_gates,
        t_analysis: TGateAnalysis::default(),
        duration_ns,
        estimated_fidelity,
        fidelity_breakdown,
    }
}

/// Analyze an abstract circuit (QGate)
pub fn analyze_abstract_circuit(circuit: &[QGate], name: &str) -> CircuitAnalysis {
    let t_analysis = analyze_t_gates(circuit);

    let gate_counts = t_analysis.gate_counts.clone();
    let mut two_qubit_gates = 0;
    let mut single_qubit_gates = 0;
    let mut max_qubit = 0;

    for gate in circuit {
        let qubits = gate_qubits(gate);
        max_qubit = max_qubit.max(qubits.iter().max().copied().unwrap_or(0));

        if qubits.len() == 2 {
            two_qubit_gates += 1;
        } else if qubits.len() == 1 {
            single_qubit_gates += 1;
        }
    }

    // Calculate depth
    let depth = calculate_depth(circuit);

    CircuitAnalysis {
        name: name.to_string(),
        num_qubits: max_qubit + 1,
        total_gates: circuit.len(),
        depth,
        gate_counts,
        two_qubit_gates,
        single_qubit_gates,
        t_analysis,
        duration_ns: 0.0, // No timing for abstract circuit
        estimated_fidelity: 1.0,
        fidelity_breakdown: FidelityBreakdown::default(),
    }
}

/// Compute cumulative gate error
fn compute_gate_error(count: usize, fidelity: f64) -> f64 {
    if count == 0 || fidelity >= 1.0 {
        return 0.0;
    }
    if fidelity <= 0.0 {
        return 1.0;
    }

    // F^n = e^(n * ln(F))
    let log_fidelity = fidelity.ln();
    let combined_fidelity = (count as f64 * log_fidelity).exp();
    1.0 - combined_fidelity
}

/// Get qubits involved in a gate
fn gate_qubits(gate: &QGate) -> Vec<usize> {
    match gate {
        QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::H(q)
        | QGate::T(q)
        | QGate::Tdg(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::Phase(q, _)
        | QGate::Measure(q) => vec![*q as usize],

        QGate::CNot(c, t)
        | QGate::CZ(c, t)
        | QGate::Swap(c, t)
        | QGate::CRz(c, t, _)
        | QGate::CPhase(c, t, _) => vec![*c as usize, *t as usize],

        QGate::Toffoli(a, b, c) | QGate::CCZ(a, b, c) => {
            vec![*a as usize, *b as usize, *c as usize]
        }

        QGate::U3(q, _, _, _) => vec![*q as usize],
    }
}

/// Calculate circuit depth
fn calculate_depth(circuit: &[QGate]) -> usize {
    if circuit.is_empty() {
        return 0;
    }

    let max_qubit = circuit.iter().flat_map(gate_qubits).max().unwrap_or(0);

    let mut qubit_depths = vec![0usize; max_qubit + 1];

    for gate in circuit {
        let qubits = gate_qubits(gate);
        if qubits.is_empty() {
            continue;
        }

        let max_depth = qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
        let new_depth = max_depth + 1;

        for &q in &qubits {
            qubit_depths[q] = new_depth;
        }
    }

    *qubit_depths.iter().max().unwrap_or(&0)
}

impl fmt::Display for CircuitAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Circuit Analysis: {} ===", self.name)?;
        writeln!(f, "Qubits: {}", self.num_qubits)?;
        writeln!(f, "Total gates: {}", self.total_gates)?;
        writeln!(f, "Depth: {}", self.depth)?;
        writeln!(f, "1Q gates: {}", self.single_qubit_gates)?;
        writeln!(f, "2Q gates: {}", self.two_qubit_gates)?;

        if self.duration_ns > 0.0 {
            if self.duration_ns < 1000.0 {
                writeln!(f, "Duration: {:.2} ns", self.duration_ns)?;
            } else if self.duration_ns < 1_000_000.0 {
                writeln!(f, "Duration: {:.2} µs", self.duration_ns / 1000.0)?;
            } else {
                writeln!(f, "Duration: {:.2} ms", self.duration_ns / 1_000_000.0)?;
            }
        }

        writeln!(f, "Est. fidelity: {:.4}", self.estimated_fidelity)?;

        if self.fidelity_breakdown.total_error() > 0.0 {
            writeln!(f, "Dominant error: {}", self.fidelity_breakdown.dominant())?;
        }

        if self.t_analysis.t_count > 0 {
            writeln!(f, "T-count: {}", self.t_analysis.t_count)?;
            writeln!(f, "T-depth: {}", self.t_analysis.t_depth)?;
        }

        Ok(())
    }
}

/// Compare circuits across hardware targets
#[derive(Clone, Debug)]
pub struct HardwareComparison {
    /// Circuit name
    pub name: String,
    /// Abstract circuit analysis
    pub abstract_analysis: CircuitAnalysis,
    /// IBM analysis (if available)
    pub ibm_analysis: Option<CircuitAnalysis>,
    /// IonQ analysis (if available)
    pub ionq_analysis: Option<CircuitAnalysis>,
}

impl HardwareComparison {
    /// Create comparison with just abstract analysis
    pub fn new(name: &str, abstract_analysis: CircuitAnalysis) -> Self {
        Self {
            name: name.to_string(),
            abstract_analysis,
            ibm_analysis: None,
            ionq_analysis: None,
        }
    }

    /// Add IBM analysis
    pub fn with_ibm(mut self, analysis: CircuitAnalysis) -> Self {
        self.ibm_analysis = Some(analysis);
        self
    }

    /// Add IonQ analysis
    pub fn with_ionq(mut self, analysis: CircuitAnalysis) -> Self {
        self.ionq_analysis = Some(analysis);
        self
    }

    /// Get speedup of IBM over IonQ
    pub fn ibm_speedup(&self) -> Option<f64> {
        match (&self.ibm_analysis, &self.ionq_analysis) {
            (Some(ibm), Some(ionq)) if ionq.duration_ns > 0.0 => {
                Some(ionq.duration_ns / ibm.duration_ns)
            }
            _ => None,
        }
    }

    /// Get fidelity advantage of IonQ over IBM
    pub fn ionq_fidelity_advantage(&self) -> Option<f64> {
        match (&self.ibm_analysis, &self.ionq_analysis) {
            (Some(ibm), Some(ionq)) => Some(ionq.estimated_fidelity - ibm.estimated_fidelity),
            _ => None,
        }
    }
}

impl fmt::Display for HardwareComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Hardware Comparison: {} ===", self.name)?;
        writeln!(f)?;

        writeln!(f, "--- Abstract ---")?;
        writeln!(f, "Gates: {}", self.abstract_analysis.total_gates)?;
        writeln!(f, "Depth: {}", self.abstract_analysis.depth)?;
        writeln!(f, "T-count: {}", self.abstract_analysis.t_analysis.t_count)?;
        writeln!(f)?;

        if let Some(ibm) = &self.ibm_analysis {
            writeln!(f, "--- IBM ---")?;
            writeln!(f, "Gates: {}", ibm.total_gates)?;
            writeln!(f, "Depth: {}", ibm.depth)?;
            writeln!(f, "CX count: {}", ibm.two_qubit_gates)?;
            writeln!(f, "Duration: {:.2} µs", ibm.duration_ns / 1000.0)?;
            writeln!(f, "Fidelity: {:.4}", ibm.estimated_fidelity)?;
            writeln!(f)?;
        }

        if let Some(ionq) = &self.ionq_analysis {
            writeln!(f, "--- IonQ ---")?;
            writeln!(f, "Gates: {}", ionq.total_gates)?;
            writeln!(f, "Depth: {}", ionq.depth)?;
            writeln!(f, "MS count: {}", ionq.two_qubit_gates)?;
            writeln!(f, "Duration: {:.2} ms", ionq.duration_ns / 1_000_000.0)?;
            writeln!(f, "Fidelity: {:.4}", ionq.estimated_fidelity)?;
            writeln!(f)?;
        }

        if let Some(speedup) = self.ibm_speedup() {
            writeln!(f, "IBM speedup: {:.1}x faster", speedup)?;
        }

        if let Some(adv) = self.ionq_fidelity_advantage() {
            if adv > 0.0 {
                writeln!(f, "IonQ fidelity advantage: +{:.4}", adv)?;
            } else {
                writeln!(f, "IBM fidelity advantage: +{:.4}", -adv)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_abstract_circuit() {
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::Measure(0),
        ];
        let analysis = analyze_abstract_circuit(&circuit, "test");

        assert_eq!(analysis.total_gates, 4);
        assert_eq!(analysis.two_qubit_gates, 1);
        assert_eq!(analysis.single_qubit_gates, 3);
        assert_eq!(analysis.t_analysis.t_count, 1);
    }

    #[test]
    fn test_fidelity_breakdown() {
        let breakdown = FidelityBreakdown {
            single_qubit_error: 0.001,
            two_qubit_error: 0.01,
            decoherence_error: 0.005,
            measurement_error: 0.02,
        };

        assert!(breakdown.total_error() > 0.0);
        assert!(breakdown.total_error() < 1.0);
        assert_eq!(breakdown.dominant(), "Measurement");
    }

    #[test]
    fn test_compute_gate_error() {
        // 0 gates = 0 error
        assert_eq!(compute_gate_error(0, 0.99), 0.0);

        // 1 gate with 99% fidelity = 1% error
        let error = compute_gate_error(1, 0.99);
        assert!((error - 0.01).abs() < 0.001);

        // 100 gates with 99% fidelity
        let error_100 = compute_gate_error(100, 0.99);
        assert!(error_100 > 0.5); // Should be significant
    }

    #[test]
    fn test_circuit_depth() {
        // Sequential gates on same qubit (using Phase for S equivalent)
        let circuit = vec![
            QGate::H(0),
            QGate::T(0),
            QGate::Phase(0, std::f32::consts::FRAC_PI_2),
        ];
        let analysis = analyze_abstract_circuit(&circuit, "test");
        assert_eq!(analysis.depth, 3);

        // Parallel gates on different qubits
        let circuit = vec![QGate::H(0), QGate::H(1), QGate::H(2)];
        let analysis = analyze_abstract_circuit(&circuit, "test");
        assert_eq!(analysis.depth, 1);
    }

    #[test]
    fn test_hardware_comparison() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let abstract_analysis = analyze_abstract_circuit(&circuit, "bell");

        let comparison = HardwareComparison::new("bell", abstract_analysis);

        assert_eq!(comparison.name, "bell");
        assert!(comparison.ibm_speedup().is_none());
    }
}
