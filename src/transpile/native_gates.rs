//! Native Gate Set Definitions
//!
//! SPRINT 53.0: Hardware-specific gate sets for transpilation
//!
//! Defines native gate sets for:
//! - IBM Falcon/Heron (superconducting): SX, X, Rz, CX
//! - IonQ Aria (trapped ion): GPI, GPI2, MS

use std::f64::consts::PI;
use std::fmt;

// ============================================================================
// IBM Native Gates (Falcon/Heron)
// ============================================================================

/// IBM native gate set
///
/// IBM quantum computers use a basis of:
/// - X: Pauli X
/// - SX: √X (square root of X)
/// - Rz: Z rotation (virtual, 0 duration)
/// - CX: CNOT (echoed cross-resonance)
#[derive(Clone, Debug, PartialEq)]
pub enum IBMGate {
    /// Identity (for explicit delays)
    Id(usize),
    /// Pauli X gate
    X(usize),
    /// √X gate (square root of X)
    SX(usize),
    /// Z rotation (virtual gate, 0 duration)
    Rz(usize, f64),
    /// CNOT gate (control, target)
    CX(usize, usize),
    /// Reset qubit to |0⟩
    Reset(usize),
    /// Measure qubit to classical bit
    Measure(usize, usize),
    /// Barrier (for circuit structure)
    Barrier(Vec<usize>),
}

impl IBMGate {
    /// Gate name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Id(_) => "id",
            Self::X(_) => "x",
            Self::SX(_) => "sx",
            Self::Rz(..) => "rz",
            Self::CX(..) => "cx",
            Self::Reset(_) => "reset",
            Self::Measure(..) => "measure",
            Self::Barrier(_) => "barrier",
        }
    }

    /// Gate duration in nanoseconds (IBM Heron specs)
    pub fn duration_ns(&self) -> f64 {
        match self {
            Self::Id(_) => 35.56,
            Self::X(_) => 35.56,
            Self::SX(_) => 35.56,
            Self::Rz(..) => 0.0, // Virtual Z - no duration
            Self::CX(..) => 68.0,
            Self::Reset(_) => 1000.0,
            Self::Measure(..) => 1000.0,
            Self::Barrier(_) => 0.0,
        }
    }

    /// Gate fidelity (IBM Heron specs)
    pub fn fidelity(&self) -> f64 {
        match self {
            Self::Id(_) => 1.0,
            Self::X(_) | Self::SX(_) => 0.9996,
            Self::Rz(..) => 1.0, // Virtual Z - perfect
            Self::CX(..) => 0.995,
            Self::Reset(_) => 0.99,
            Self::Measure(..) => 0.98,
            Self::Barrier(_) => 1.0,
        }
    }

    /// Number of qubits involved
    pub fn qubit_count(&self) -> usize {
        match self {
            Self::CX(..) => 2,
            Self::Barrier(qubits) => qubits.len(),
            _ => 1,
        }
    }

    /// Get qubits involved
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Self::Id(q) | Self::X(q) | Self::SX(q) | Self::Rz(q, _) | Self::Reset(q) => vec![*q],
            Self::Measure(q, _) => vec![*q],
            Self::CX(c, t) => vec![*c, *t],
            Self::Barrier(qubits) => qubits.clone(),
        }
    }
}

impl fmt::Display for IBMGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(q) => write!(f, "id q[{}]", q),
            Self::X(q) => write!(f, "x q[{}]", q),
            Self::SX(q) => write!(f, "sx q[{}]", q),
            Self::Rz(q, angle) => write!(f, "rz({:.6}) q[{}]", angle, q),
            Self::CX(c, t) => write!(f, "cx q[{}], q[{}]", c, t),
            Self::Reset(q) => write!(f, "reset q[{}]", q),
            Self::Measure(q, c) => write!(f, "measure q[{}] -> c[{}]", q, c),
            Self::Barrier(qubits) => {
                write!(f, "barrier ")?;
                for (i, q) in qubits.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "q[{}]", q)?;
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// IonQ Native Gates (Aria)
// ============================================================================

/// IonQ native gate set
///
/// IonQ trapped ion computers use:
/// - GPI: Single-qubit rotation around axis in XY plane
/// - GPI2: π/2 rotation around axis in XY plane
/// - MS: Mølmer-Sørensen entangling gate
#[derive(Clone, Debug, PartialEq)]
pub enum IonQGate {
    /// GPI gate: rotation by π around axis at angle φ in XY plane
    /// GPI(φ) = [[0, e^{-iφ}], [e^{iφ}, 0]]
    GPI(usize, f64),
    /// GPI2 gate: rotation by π/2 around axis at angle φ in XY plane
    /// GPI2(φ) = (1/√2) [[1, -ie^{-iφ}], [-ie^{iφ}, 1]]
    GPI2(usize, f64),
    /// Mølmer-Sørensen gate: XX interaction
    /// MS(φ₀, φ₁) = exp(-iπ/4 · σ_φ₀ ⊗ σ_φ₁)
    MS(usize, usize, f64, f64),
    /// Measure qubit
    Measure(usize),
}

impl IonQGate {
    /// Gate name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::GPI(..) => "gpi",
            Self::GPI2(..) => "gpi2",
            Self::MS(..) => "ms",
            Self::Measure(_) => "measure",
        }
    }

    /// Gate duration in nanoseconds (IonQ Aria specs)
    pub fn duration_ns(&self) -> f64 {
        match self {
            Self::GPI(..) => 135_000.0,  // 135 µs
            Self::GPI2(..) => 135_000.0, // 135 µs
            Self::MS(..) => 600_000.0,   // 600 µs
            Self::Measure(_) => 300_000.0,
        }
    }

    /// Gate fidelity (IonQ Aria specs)
    pub fn fidelity(&self) -> f64 {
        match self {
            Self::GPI(..) | Self::GPI2(..) => 0.9998,
            Self::MS(..) => 0.995,
            Self::Measure(_) => 0.995,
        }
    }

    /// Number of qubits involved
    pub fn qubit_count(&self) -> usize {
        match self {
            Self::MS(..) => 2,
            _ => 1,
        }
    }

    /// Get qubits involved
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Self::GPI(q, _) | Self::GPI2(q, _) | Self::Measure(q) => vec![*q],
            Self::MS(q1, q2, _, _) => vec![*q1, *q2],
        }
    }
}

impl fmt::Display for IonQGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GPI(q, phi) => write!(f, "gpi({:.6}) q[{}]", phi, q),
            Self::GPI2(q, phi) => write!(f, "gpi2({:.6}) q[{}]", phi, q),
            Self::MS(q1, q2, phi0, phi1) => {
                write!(f, "ms({:.6}, {:.6}) q[{}], q[{}]", phi0, phi1, q1, q2)
            }
            Self::Measure(q) => write!(f, "measure q[{}]", q),
        }
    }
}

// ============================================================================
// Transpiled Circuit Types
// ============================================================================

/// IBM transpiled circuit
#[derive(Clone, Debug)]
pub struct IBMCircuit {
    /// Circuit name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_clbits: usize,
    /// Gates in order
    pub gates: Vec<IBMGate>,
}

impl IBMCircuit {
    /// Create new empty circuit
    pub fn new(name: &str, num_qubits: usize) -> Self {
        Self {
            name: name.to_string(),
            num_qubits,
            num_clbits: num_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate
    pub fn add(&mut self, gate: IBMGate) {
        self.gates.push(gate);
    }

    /// Total gate count
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Count of specific gate type
    pub fn count(&self, name: &str) -> usize {
        self.gates.iter().filter(|g| g.name() == name).count()
    }

    /// Total circuit duration in nanoseconds
    pub fn total_duration_ns(&self) -> f64 {
        // Simple sequential model (actual parallel execution would be less)
        self.gates.iter().map(|g| g.duration_ns()).sum()
    }

    /// Estimated circuit fidelity (product of gate fidelities)
    pub fn estimated_fidelity(&self) -> f64 {
        self.gates.iter().map(|g| g.fidelity()).product()
    }

    /// Circuit depth (layers of parallel gates)
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        let mut qubit_depths = vec![0usize; self.num_qubits];

        for gate in &self.gates {
            let qubits = gate.qubits();
            let max_depth = qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
            let new_depth = max_depth + 1;
            for &q in &qubits {
                qubit_depths[q] = new_depth;
            }
        }

        *qubit_depths.iter().max().unwrap_or(&0)
    }

    /// Convert to OpenQASM 2.0 string
    pub fn to_qasm(&self) -> String {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 2.0;\n");
        qasm.push_str("include \"qelib1.inc\";\n\n");
        qasm.push_str(&format!("qreg q[{}];\n", self.num_qubits));
        qasm.push_str(&format!("creg c[{}];\n\n", self.num_clbits));

        for gate in &self.gates {
            qasm.push_str(&format!("{};\n", gate));
        }

        qasm
    }
}

impl fmt::Display for IBMCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IBMCircuit '{}' ({} qubits)", self.name, self.num_qubits)?;
        writeln!(f, "  Gates: {}", self.gate_count())?;
        writeln!(f, "  Depth: {}", self.depth())?;
        writeln!(f, "  CX count: {}", self.count("cx"))?;
        writeln!(f, "  Duration: {:.2} µs", self.total_duration_ns() / 1000.0)?;
        writeln!(f, "  Est. fidelity: {:.4}", self.estimated_fidelity())
    }
}

/// IonQ transpiled circuit
#[derive(Clone, Debug)]
pub struct IonQCircuit {
    /// Circuit name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates in order
    pub gates: Vec<IonQGate>,
}

impl IonQCircuit {
    /// Create new empty circuit
    pub fn new(name: &str, num_qubits: usize) -> Self {
        Self {
            name: name.to_string(),
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate
    pub fn add(&mut self, gate: IonQGate) {
        self.gates.push(gate);
    }

    /// Total gate count
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Count of specific gate type
    pub fn count(&self, name: &str) -> usize {
        self.gates.iter().filter(|g| g.name() == name).count()
    }

    /// Total circuit duration in nanoseconds
    pub fn total_duration_ns(&self) -> f64 {
        self.gates.iter().map(|g| g.duration_ns()).sum()
    }

    /// Estimated circuit fidelity
    pub fn estimated_fidelity(&self) -> f64 {
        self.gates.iter().map(|g| g.fidelity()).product()
    }

    /// Circuit depth
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        let mut qubit_depths = vec![0usize; self.num_qubits];

        for gate in &self.gates {
            let qubits = gate.qubits();
            let max_depth = qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
            let new_depth = max_depth + 1;
            for &q in &qubits {
                qubit_depths[q] = new_depth;
            }
        }

        *qubit_depths.iter().max().unwrap_or(&0)
    }
}

impl fmt::Display for IonQCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "IonQCircuit '{}' ({} qubits)",
            self.name, self.num_qubits
        )?;
        writeln!(f, "  Gates: {}", self.gate_count())?;
        writeln!(f, "  Depth: {}", self.depth())?;
        writeln!(f, "  MS count: {}", self.count("ms"))?;
        writeln!(f, "  Duration: {:.2} ms", self.total_duration_ns() / 1e6)?;
        writeln!(f, "  Est. fidelity: {:.4}", self.estimated_fidelity())
    }
}

// ============================================================================
// Common Gate Conversions
// ============================================================================

/// Convert standard gate angle to IBM Rz + SX sequence
pub mod ibm_decompositions {
    use super::*;

    /// H = Rz(π/2) · SX · Rz(π/2)
    pub fn hadamard(q: usize) -> Vec<IBMGate> {
        vec![
            IBMGate::Rz(q, PI / 2.0),
            IBMGate::SX(q),
            IBMGate::Rz(q, PI / 2.0),
        ]
    }

    /// T = Rz(π/4)
    pub fn t_gate(q: usize) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, PI / 4.0)]
    }

    /// T† = Rz(-π/4)
    pub fn t_dagger(q: usize) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, -PI / 4.0)]
    }

    /// S = Rz(π/2)
    pub fn s_gate(q: usize) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, PI / 2.0)]
    }

    /// S† = Rz(-π/2)
    pub fn s_dagger(q: usize) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, -PI / 2.0)]
    }

    /// Z = Rz(π)
    pub fn z_gate(q: usize) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, PI)]
    }

    /// Y = Rz(π) · X (or SX · Rz(π) · SX · Rz(π))
    pub fn y_gate(q: usize) -> Vec<IBMGate> {
        vec![
            IBMGate::Rz(q, PI),
            IBMGate::SX(q),
            IBMGate::Rz(q, PI),
            IBMGate::SX(q),
        ]
    }

    /// Rx(θ) = Rz(-π/2) · SX · Rz(θ+π) · SX · Rz(π/2 - π)
    /// Simplified: Rz(π/2) · SX · Rz(θ+π) · SX · Rz(-π/2)
    pub fn rx(q: usize, theta: f64) -> Vec<IBMGate> {
        vec![
            IBMGate::Rz(q, PI / 2.0),
            IBMGate::SX(q),
            IBMGate::Rz(q, theta + PI),
            IBMGate::SX(q),
            IBMGate::Rz(q, -PI / 2.0),
        ]
    }

    /// Ry(θ) = Rz(-π/2) · SX · Rz(θ+π) · SX · Rz(3π/2)
    /// This is derived from Ry = S† · H · Rz(θ) · H · S
    pub fn ry(q: usize, theta: f64) -> Vec<IBMGate> {
        // Simpler: SX · Rz(θ+π) · SX
        vec![IBMGate::SX(q), IBMGate::Rz(q, theta + PI), IBMGate::SX(q)]
    }

    /// Rz(θ) - native gate
    pub fn rz(q: usize, theta: f64) -> Vec<IBMGate> {
        vec![IBMGate::Rz(q, theta)]
    }

    /// U3(θ, φ, λ) = Rz(φ-π/2) · SX · Rz(θ+π) · SX · Rz(λ-π/2)
    pub fn u3(q: usize, theta: f64, phi: f64, lambda: f64) -> Vec<IBMGate> {
        vec![
            IBMGate::Rz(q, phi - PI / 2.0),
            IBMGate::SX(q),
            IBMGate::Rz(q, theta + PI),
            IBMGate::SX(q),
            IBMGate::Rz(q, lambda - PI / 2.0),
        ]
    }
}

/// Convert standard gates to IonQ native set
pub mod ionq_decompositions {
    use super::*;

    /// X = GPI(0)
    pub fn x_gate(q: usize) -> Vec<IonQGate> {
        vec![IonQGate::GPI(q, 0.0)]
    }

    /// Y = GPI(π/2)
    pub fn y_gate(q: usize) -> Vec<IonQGate> {
        vec![IonQGate::GPI(q, PI / 2.0)]
    }

    /// Z = GPI2(0) · GPI2(0)
    pub fn z_gate(q: usize) -> Vec<IonQGate> {
        vec![IonQGate::GPI2(q, 0.0), IonQGate::GPI2(q, 0.0)]
    }

    /// H = GPI2(0) · GPI(π/4)
    pub fn hadamard(q: usize) -> Vec<IonQGate> {
        vec![IonQGate::GPI2(q, 0.0), IonQGate::GPI(q, PI / 4.0)]
    }

    /// CNOT decomposition using MS gate
    /// CX = GPI2(t, -π/2) · MS(c, t, 0, 0) · GPI(c, -π/2) · GPI2(c, π/2) · GPI2(t, π/2)
    pub fn cnot(control: usize, target: usize) -> Vec<IonQGate> {
        vec![
            IonQGate::GPI2(target, -PI / 2.0),
            IonQGate::MS(control, target, 0.0, 0.0),
            IonQGate::GPI(control, -PI / 2.0),
            IonQGate::GPI2(control, PI / 2.0),
            IonQGate::GPI2(target, PI / 2.0),
        ]
    }

    /// Rz(θ) = GPI(π/2) · GPI(θ/2 - π/2)
    /// Or using virtual Z: GPI2(-π/2) · GPI2(θ - π/2)
    pub fn rz(q: usize, theta: f64) -> Vec<IonQGate> {
        // Using phase tracking - this is approximate
        vec![
            IonQGate::GPI2(q, -PI / 2.0),
            IonQGate::GPI2(q, theta - PI / 2.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_gate_properties() {
        let cx = IBMGate::CX(0, 1);
        assert_eq!(cx.name(), "cx");
        assert_eq!(cx.qubit_count(), 2);
        assert_eq!(cx.duration_ns(), 68.0);
        assert!(cx.fidelity() > 0.99);
    }

    #[test]
    fn test_ibm_circuit() {
        let mut circuit = IBMCircuit::new("test", 2);
        circuit.add(IBMGate::SX(0));
        circuit.add(IBMGate::Rz(0, PI / 4.0));
        circuit.add(IBMGate::CX(0, 1));

        assert_eq!(circuit.gate_count(), 3);
        assert_eq!(circuit.count("cx"), 1);
        assert_eq!(circuit.count("sx"), 1);
        assert!(circuit.depth() >= 2);
    }

    #[test]
    fn test_ionq_gate_properties() {
        let ms = IonQGate::MS(0, 1, 0.0, 0.0);
        assert_eq!(ms.name(), "ms");
        assert_eq!(ms.qubit_count(), 2);
        assert_eq!(ms.duration_ns(), 600_000.0); // 600 µs
    }

    #[test]
    fn test_ibm_hadamard_decomposition() {
        let h = ibm_decompositions::hadamard(0);
        assert_eq!(h.len(), 3); // Rz, SX, Rz
        assert_eq!(h[1], IBMGate::SX(0));
    }

    #[test]
    fn test_ionq_cnot_decomposition() {
        let cx = ionq_decompositions::cnot(0, 1);
        assert!(cx.iter().any(|g| matches!(g, IonQGate::MS(..))));
        assert_eq!(cx.len(), 5);
    }

    #[test]
    fn test_ibm_circuit_qasm() {
        let mut circuit = IBMCircuit::new("bell", 2);
        circuit.add(IBMGate::Rz(0, PI / 2.0));
        circuit.add(IBMGate::SX(0));
        circuit.add(IBMGate::Rz(0, PI / 2.0));
        circuit.add(IBMGate::CX(0, 1));

        let qasm = circuit.to_qasm();
        assert!(qasm.contains("OPENQASM 2.0"));
        assert!(qasm.contains("qreg q[2]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }
}
