//! OpenQASM 3.0 Abstract Syntax Tree
//!
//! Defines the AST types for representing parsed QASM3 programs.

use std::fmt;

/// A complete QASM3 program
#[derive(Clone, Debug)]
pub struct QasmProgram {
    /// QASM version (e.g., "3.0")
    pub version: String,
    /// Include statements (e.g., "stdgates.inc")
    pub includes: Vec<String>,
    /// Number of qubits declared
    pub num_qubits: usize,
    /// Number of classical bits declared
    pub num_clbits: usize,
    /// Qubit register name (default "q")
    pub qubit_register: String,
    /// Classical register name (default "c")
    pub clbit_register: String,
    /// Statements in the program
    pub statements: Vec<QasmStatement>,
}

impl Default for QasmProgram {
    fn default() -> Self {
        Self {
            version: "3.0".to_string(),
            includes: vec!["stdgates.inc".to_string()],
            num_qubits: 0,
            num_clbits: 0,
            qubit_register: "q".to_string(),
            clbit_register: "c".to_string(),
            statements: Vec::new(),
        }
    }
}

impl QasmProgram {
    /// Create a new empty program
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            num_clbits: num_qubits,
            ..Default::default()
        }
    }

    /// Add a gate statement
    pub fn add_gate(&mut self, gate: QasmGate) {
        self.statements.push(QasmStatement::Gate(gate));
    }

    /// Add a measurement statement
    pub fn add_measurement(&mut self, qubit: usize, clbit: usize) {
        self.statements
            .push(QasmStatement::Measure { qubit, clbit });
    }

    /// Get all gate statements
    pub fn gates(&self) -> impl Iterator<Item = &QasmGate> {
        self.statements.iter().filter_map(|s| match s {
            QasmStatement::Gate(g) => Some(g),
            _ => None,
        })
    }
}

/// A statement in a QASM3 program
#[derive(Clone, Debug, PartialEq)]
pub enum QasmStatement {
    /// Gate application
    Gate(QasmGate),
    /// Measurement: measure qubit -> clbit
    Measure { qubit: usize, clbit: usize },
    /// Reset qubit to |0⟩
    Reset(usize),
    /// Barrier (for visualization/optimization boundary)
    Barrier(Vec<usize>),
    /// Comment (preserved for round-trip)
    Comment(String),
}

/// A gate in QASM3
#[derive(Clone, Debug, PartialEq)]
pub enum QasmGate {
    // Single-qubit gates (no parameters)
    /// Hadamard gate
    H(usize),
    /// Pauli X gate
    X(usize),
    /// Pauli Y gate
    Y(usize),
    /// Pauli Z gate
    Z(usize),
    /// T gate (π/8 rotation)
    T(usize),
    /// T-dagger gate
    Tdg(usize),
    /// S gate (π/4 rotation)
    S(usize),
    /// S-dagger gate
    Sdg(usize),
    /// SX gate (√X)
    SX(usize),
    /// SXdg gate (√X†)
    SXdg(usize),

    // Single-qubit rotation gates (with parameters)
    /// Rx rotation
    Rx(usize, f64),
    /// Ry rotation
    Ry(usize, f64),
    /// Rz rotation
    Rz(usize, f64),
    /// Phase gate P(θ) = diag(1, e^{iθ})
    P(usize, f64),
    /// U1 gate (equivalent to P)
    U1(usize, f64),
    /// U2 gate U2(φ,λ) = U3(π/2, φ, λ)
    U2(usize, f64, f64),
    /// U3 gate (universal single-qubit)
    U3(usize, f64, f64, f64),
    /// U gate (alias for U3)
    U(usize, f64, f64, f64),

    // Two-qubit gates
    /// CNOT (CX) gate
    CX(usize, usize),
    /// CZ gate
    CZ(usize, usize),
    /// SWAP gate
    Swap(usize, usize),
    /// iSWAP gate
    ISwap(usize, usize),
    /// Controlled-Rx
    CRx(usize, usize, f64),
    /// Controlled-Ry
    CRy(usize, usize, f64),
    /// Controlled-Rz
    CRz(usize, usize, f64),
    /// Controlled-Phase
    CP(usize, usize, f64),
    /// RZZ gate
    RZZ(usize, usize, f64),
    /// RXX gate
    RXX(usize, usize, f64),
    /// RYY gate
    RYY(usize, usize, f64),

    // Three-qubit gates
    /// Toffoli (CCX) gate
    CCX(usize, usize, usize),
    /// CCZ gate
    CCZ(usize, usize, usize),
    /// CSWAP (Fredkin) gate
    CSwap(usize, usize, usize),

    // Identity (for explicit delays)
    /// Identity gate
    Id(usize),
}

impl QasmGate {
    /// Get the QASM3 gate name
    pub fn name(&self) -> &'static str {
        match self {
            Self::H(_) => "h",
            Self::X(_) => "x",
            Self::Y(_) => "y",
            Self::Z(_) => "z",
            Self::T(_) => "t",
            Self::Tdg(_) => "tdg",
            Self::S(_) => "s",
            Self::Sdg(_) => "sdg",
            Self::SX(_) => "sx",
            Self::SXdg(_) => "sxdg",
            Self::Rx(..) => "rx",
            Self::Ry(..) => "ry",
            Self::Rz(..) => "rz",
            Self::P(..) => "p",
            Self::U1(..) => "u1",
            Self::U2(..) => "u2",
            Self::U3(..) => "u3",
            Self::U(..) => "u",
            Self::CX(..) => "cx",
            Self::CZ(..) => "cz",
            Self::Swap(..) => "swap",
            Self::ISwap(..) => "iswap",
            Self::CRx(..) => "crx",
            Self::CRy(..) => "cry",
            Self::CRz(..) => "crz",
            Self::CP(..) => "cp",
            Self::RZZ(..) => "rzz",
            Self::RXX(..) => "rxx",
            Self::RYY(..) => "ryy",
            Self::CCX(..) => "ccx",
            Self::CCZ(..) => "ccz",
            Self::CSwap(..) => "cswap",
            Self::Id(_) => "id",
        }
    }

    /// Get the qubits this gate operates on
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            // Single-qubit
            Self::H(q)
            | Self::X(q)
            | Self::Y(q)
            | Self::Z(q)
            | Self::T(q)
            | Self::Tdg(q)
            | Self::S(q)
            | Self::Sdg(q)
            | Self::SX(q)
            | Self::SXdg(q)
            | Self::Rx(q, _)
            | Self::Ry(q, _)
            | Self::Rz(q, _)
            | Self::P(q, _)
            | Self::U1(q, _)
            | Self::U2(q, _, _)
            | Self::U3(q, _, _, _)
            | Self::U(q, _, _, _)
            | Self::Id(q) => vec![*q],

            // Two-qubit
            Self::CX(a, b)
            | Self::CZ(a, b)
            | Self::Swap(a, b)
            | Self::ISwap(a, b)
            | Self::CRx(a, b, _)
            | Self::CRy(a, b, _)
            | Self::CRz(a, b, _)
            | Self::CP(a, b, _)
            | Self::RZZ(a, b, _)
            | Self::RXX(a, b, _)
            | Self::RYY(a, b, _) => vec![*a, *b],

            // Three-qubit
            Self::CCX(a, b, c) | Self::CCZ(a, b, c) | Self::CSwap(a, b, c) => vec![*a, *b, *c],
        }
    }

    /// Check if this gate has parameters
    pub fn has_parameters(&self) -> bool {
        matches!(
            self,
            Self::Rx(..)
                | Self::Ry(..)
                | Self::Rz(..)
                | Self::P(..)
                | Self::U1(..)
                | Self::U2(..)
                | Self::U3(..)
                | Self::U(..)
                | Self::CRx(..)
                | Self::CRy(..)
                | Self::CRz(..)
                | Self::CP(..)
                | Self::RZZ(..)
                | Self::RXX(..)
                | Self::RYY(..)
        )
    }
}

impl fmt::Display for QasmGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Single-qubit no params
            Self::H(q) => write!(f, "h q[{}]", q),
            Self::X(q) => write!(f, "x q[{}]", q),
            Self::Y(q) => write!(f, "y q[{}]", q),
            Self::Z(q) => write!(f, "z q[{}]", q),
            Self::T(q) => write!(f, "t q[{}]", q),
            Self::Tdg(q) => write!(f, "tdg q[{}]", q),
            Self::S(q) => write!(f, "s q[{}]", q),
            Self::Sdg(q) => write!(f, "sdg q[{}]", q),
            Self::SX(q) => write!(f, "sx q[{}]", q),
            Self::SXdg(q) => write!(f, "sxdg q[{}]", q),
            Self::Id(q) => write!(f, "id q[{}]", q),

            // Single-qubit with params
            Self::Rx(q, theta) => write!(f, "rx({}) q[{}]", theta, q),
            Self::Ry(q, theta) => write!(f, "ry({}) q[{}]", theta, q),
            Self::Rz(q, theta) => write!(f, "rz({}) q[{}]", theta, q),
            Self::P(q, theta) => write!(f, "p({}) q[{}]", theta, q),
            Self::U1(q, lambda) => write!(f, "u1({}) q[{}]", lambda, q),
            Self::U2(q, phi, lambda) => write!(f, "u2({}, {}) q[{}]", phi, lambda, q),
            Self::U3(q, theta, phi, lambda) => {
                write!(f, "u3({}, {}, {}) q[{}]", theta, phi, lambda, q)
            }
            Self::U(q, theta, phi, lambda) => {
                write!(f, "u({}, {}, {}) q[{}]", theta, phi, lambda, q)
            }

            // Two-qubit
            Self::CX(c, t) => write!(f, "cx q[{}], q[{}]", c, t),
            Self::CZ(a, b) => write!(f, "cz q[{}], q[{}]", a, b),
            Self::Swap(a, b) => write!(f, "swap q[{}], q[{}]", a, b),
            Self::ISwap(a, b) => write!(f, "iswap q[{}], q[{}]", a, b),
            Self::CRx(c, t, theta) => write!(f, "crx({}) q[{}], q[{}]", theta, c, t),
            Self::CRy(c, t, theta) => write!(f, "cry({}) q[{}], q[{}]", theta, c, t),
            Self::CRz(c, t, theta) => write!(f, "crz({}) q[{}], q[{}]", theta, c, t),
            Self::CP(c, t, theta) => write!(f, "cp({}) q[{}], q[{}]", theta, c, t),
            Self::RZZ(a, b, theta) => write!(f, "rzz({}) q[{}], q[{}]", theta, a, b),
            Self::RXX(a, b, theta) => write!(f, "rxx({}) q[{}], q[{}]", theta, a, b),
            Self::RYY(a, b, theta) => write!(f, "ryy({}) q[{}], q[{}]", theta, a, b),

            // Three-qubit
            Self::CCX(a, b, c) => write!(f, "ccx q[{}], q[{}], q[{}]", a, b, c),
            Self::CCZ(a, b, c) => write!(f, "ccz q[{}], q[{}], q[{}]", a, b, c),
            Self::CSwap(a, b, c) => write!(f, "cswap q[{}], q[{}], q[{}]", a, b, c),
        }
    }
}

impl fmt::Display for QasmStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gate(g) => write!(f, "{};", g),
            Self::Measure { qubit, clbit } => write!(f, "c[{}] = measure q[{}];", clbit, qubit),
            Self::Reset(q) => write!(f, "reset q[{}];", q),
            Self::Barrier(qubits) => {
                let qs: Vec<String> = qubits.iter().map(|q| format!("q[{}]", q)).collect();
                write!(f, "barrier {};", qs.join(", "))
            }
            Self::Comment(c) => write!(f, "// {}", c),
        }
    }
}

impl fmt::Display for QasmProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OPENQASM {};", self.version)?;
        for inc in &self.includes {
            writeln!(f, "include \"{}\";", inc)?;
        }
        writeln!(f)?;
        writeln!(f, "qubit[{}] {};", self.num_qubits, self.qubit_register)?;
        if self.num_clbits > 0 {
            writeln!(f, "bit[{}] {};", self.num_clbits, self.clbit_register)?;
        }
        writeln!(f)?;
        for stmt in &self.statements {
            writeln!(f, "{}", stmt)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)] // 1.5708 is literal display-format test data, not π/2
    fn test_gate_display() {
        assert_eq!(QasmGate::H(0).to_string(), "h q[0]");
        assert_eq!(QasmGate::CX(0, 1).to_string(), "cx q[0], q[1]");
        assert_eq!(QasmGate::Rz(2, 1.5708).to_string(), "rz(1.5708) q[2]");
    }

    #[test]
    fn test_program_display() {
        let mut prog = QasmProgram::new(2);
        prog.add_gate(QasmGate::H(0));
        prog.add_gate(QasmGate::CX(0, 1));

        let output = prog.to_string();
        assert!(output.contains("OPENQASM 3.0"));
        assert!(output.contains("qubit[2] q"));
        assert!(output.contains("h q[0]"));
        assert!(output.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_gate_qubits() {
        assert_eq!(QasmGate::H(0).qubits(), vec![0]);
        assert_eq!(QasmGate::CX(0, 1).qubits(), vec![0, 1]);
        assert_eq!(QasmGate::CCX(0, 1, 2).qubits(), vec![0, 1, 2]);
    }
}
