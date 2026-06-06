//! OpenQASM 3.0 Emitter
//!
//! Converts QGate circuits to QASM3 source text.

use super::ast::{QasmGate, QasmProgram, QasmStatement};
use logic_fabric_core::quantum::QGate;
use std::f64::consts::PI;

/// Emission options
#[derive(Clone, Debug)]
pub struct EmitOptions {
    /// Include OPENQASM header
    pub include_header: bool,
    /// Include stdgates.inc
    pub include_stdgates: bool,
    /// Qubit register name
    pub qubit_register: String,
    /// Classical register name
    pub clbit_register: String,
    /// Add comments for circuit structure
    pub add_comments: bool,
    /// Pretty-print with indentation
    pub pretty_print: bool,
    /// Use pi notation for common angles
    pub use_pi_notation: bool,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            include_header: true,
            include_stdgates: true,
            qubit_register: "q".to_string(),
            clbit_register: "c".to_string(),
            add_comments: false,
            pretty_print: true,
            use_pi_notation: true,
        }
    }
}

impl EmitOptions {
    /// Create minimal options (no header, compact)
    pub fn minimal() -> Self {
        Self {
            include_header: false,
            include_stdgates: false,
            add_comments: false,
            pretty_print: false,
            ..Default::default()
        }
    }

    /// Create options for Qiskit compatibility
    pub fn qiskit() -> Self {
        Self {
            include_header: true,
            include_stdgates: true,
            add_comments: false,
            pretty_print: true,
            use_pi_notation: true,
            ..Default::default()
        }
    }
}

/// Emit QGate circuit to QASM3
///
/// # Example
///
/// ```ignore
/// use engine::transpile::qasm3::emit_qasm3;
/// use logic_fabric_core::quantum::QGate;
///
/// let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
/// let qasm = emit_qasm3(&circuit, "bell_state");
///
/// assert!(qasm.contains("h q[0]"));
/// assert!(qasm.contains("cx q[0], q[1]"));
/// ```
pub fn emit_qasm3(circuit: &[QGate], name: &str) -> String {
    emit_qasm3_with_options(circuit, name, &EmitOptions::default())
}

/// Emit QGate circuit to QASM3 with options
pub fn emit_qasm3_with_options(circuit: &[QGate], name: &str, options: &EmitOptions) -> String {
    let num_qubits = count_qubits(circuit);
    let num_clbits = count_measurements(circuit);

    let mut output = String::new();

    // Header
    if options.include_header {
        output.push_str("OPENQASM 3.0;\n");
        if options.include_stdgates {
            output.push_str("include \"stdgates.inc\";\n");
        }
        output.push('\n');
    }

    // Comment with circuit name
    if options.add_comments {
        output.push_str(&format!("// Circuit: {}\n", name));
    }

    // Qubit declaration
    output.push_str(&format!(
        "qubit[{}] {};\n",
        num_qubits, options.qubit_register
    ));

    // Classical bit declaration
    if num_clbits > 0 {
        output.push_str(&format!(
            "bit[{}] {};\n",
            num_clbits, options.clbit_register
        ));
    }

    if options.pretty_print {
        output.push('\n');
    }

    // Gates
    let mut measurement_idx = 0;
    for gate in circuit {
        let line = emit_gate(gate, options, &mut measurement_idx);
        output.push_str(&line);
        output.push('\n');
    }

    output
}

/// Emit a single gate to QASM3 format
fn emit_gate(gate: &QGate, options: &EmitOptions, measurement_idx: &mut usize) -> String {
    let q = &options.qubit_register;
    let c = &options.clbit_register;

    match gate {
        // Single-qubit gates
        QGate::H(qubit) => format!("h {}[{}];", q, qubit),
        QGate::X(qubit) => format!("x {}[{}];", q, qubit),
        QGate::Y(qubit) => format!("y {}[{}];", q, qubit),
        QGate::Z(qubit) => format!("z {}[{}];", q, qubit),
        QGate::T(qubit) => format!("t {}[{}];", q, qubit),
        QGate::Tdg(qubit) => format!("tdg {}[{}];", q, qubit),

        // Rotation gates
        QGate::Rx(qubit, angle) => {
            format!(
                "rx({}) {}[{}];",
                format_angle(*angle as f64, options),
                q,
                qubit
            )
        }
        QGate::Ry(qubit, angle) => {
            format!(
                "ry({}) {}[{}];",
                format_angle(*angle as f64, options),
                q,
                qubit
            )
        }
        QGate::Rz(qubit, angle) => {
            format!(
                "rz({}) {}[{}];",
                format_angle(*angle as f64, options),
                q,
                qubit
            )
        }
        QGate::Phase(qubit, angle) => {
            // Check for S and Sdg
            let a = *angle as f64;
            if options.use_pi_notation {
                if (a - PI / 2.0).abs() < 1e-6 {
                    return format!("s {}[{}];", q, qubit);
                } else if (a + PI / 2.0).abs() < 1e-6 {
                    return format!("sdg {}[{}];", q, qubit);
                }
            }
            format!("p({}) {}[{}];", format_angle(a, options), q, qubit)
        }
        QGate::U3(qubit, theta, phi, lambda) => {
            format!(
                "u3({}, {}, {}) {}[{}];",
                format_angle(*theta as f64, options),
                format_angle(*phi as f64, options),
                format_angle(*lambda as f64, options),
                q,
                qubit
            )
        }

        // Two-qubit gates
        QGate::CNot(ctrl, tgt) => format!("cx {}[{}], {}[{}];", q, ctrl, q, tgt),
        QGate::CZ(a, b) => format!("cz {}[{}], {}[{}];", q, a, q, b),
        QGate::Swap(a, b) => format!("swap {}[{}], {}[{}];", q, a, q, b),
        QGate::CPhase(ctrl, tgt, angle) => {
            format!(
                "cp({}) {}[{}], {}[{}];",
                format_angle(*angle as f64, options),
                q,
                ctrl,
                q,
                tgt
            )
        }
        QGate::CRz(ctrl, tgt, angle) => {
            format!(
                "crz({}) {}[{}], {}[{}];",
                format_angle(*angle as f64, options),
                q,
                ctrl,
                q,
                tgt
            )
        }

        // Three-qubit gates
        QGate::Toffoli(a, b, c_qubit) => {
            format!("ccx {}[{}], {}[{}], {}[{}];", q, a, q, b, q, c_qubit)
        }
        QGate::CCZ(a, b, c_qubit) => {
            format!("ccz {}[{}], {}[{}], {}[{}];", q, a, q, b, q, c_qubit)
        }

        // Measurement
        QGate::Measure(qubit) => {
            let idx = *measurement_idx;
            *measurement_idx += 1;
            format!("{}[{}] = measure {}[{}];", c, idx, q, qubit)
        }
    }
}

/// Format an angle, optionally using pi notation
fn format_angle(angle: f64, options: &EmitOptions) -> String {
    if !options.use_pi_notation {
        return format!("{:.6}", angle);
    }

    // Check for common fractions of pi
    let ratio = angle / PI;

    // Exact pi multiples
    if (ratio - 1.0).abs() < 1e-6 {
        return "pi".to_string();
    }
    if (ratio + 1.0).abs() < 1e-6 {
        return "-pi".to_string();
    }
    if (ratio - 2.0).abs() < 1e-6 {
        return "2*pi".to_string();
    }
    if (ratio + 2.0).abs() < 1e-6 {
        return "-2*pi".to_string();
    }

    // Common fractions
    for (num, denom) in [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 6),
        (1, 8),
        (2, 3),
        (3, 4),
        (3, 2),
    ] {
        let frac = num as f64 / denom as f64;
        if (ratio - frac).abs() < 1e-6 {
            if num == 1 {
                return format!("pi/{}", denom);
            } else {
                return format!("{}*pi/{}", num, denom);
            }
        }
        if (ratio + frac).abs() < 1e-6 {
            if num == 1 {
                return format!("-pi/{}", denom);
            } else {
                return format!("-{}*pi/{}", num, denom);
            }
        }
    }

    // Not a common fraction, use numeric
    format!("{:.6}", angle)
}

/// Count qubits used in circuit
fn count_qubits(circuit: &[QGate]) -> usize {
    let mut max_qubit = 0u8;
    for gate in circuit {
        let qubits = get_qubits(gate);
        for q in qubits {
            max_qubit = max_qubit.max(q);
        }
    }
    max_qubit as usize + 1
}

/// Get qubits used by a gate
fn get_qubits(gate: &QGate) -> Vec<u8> {
    match gate {
        QGate::H(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::T(q)
        | QGate::Tdg(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::Phase(q, _)
        | QGate::U3(q, _, _, _)
        | QGate::Measure(q) => vec![*q],
        QGate::CNot(a, b)
        | QGate::CZ(a, b)
        | QGate::Swap(a, b)
        | QGate::CPhase(a, b, _)
        | QGate::CRz(a, b, _) => vec![*a, *b],
        QGate::Toffoli(a, b, c) | QGate::CCZ(a, b, c) => vec![*a, *b, *c],
    }
}

/// Count measurements in circuit
fn count_measurements(circuit: &[QGate]) -> usize {
    circuit
        .iter()
        .filter(|g| matches!(g, QGate::Measure(_)))
        .count()
}

/// Convert Vec<QGate> to QasmProgram AST
#[allow(dead_code)]
pub fn qgate_to_program(circuit: &[QGate], _name: &str) -> QasmProgram {
    let mut program = QasmProgram::new(count_qubits(circuit));
    program.num_clbits = count_measurements(circuit);

    let mut measurement_idx = 0;
    for gate in circuit {
        match gate {
            QGate::Measure(q) => {
                program.statements.push(QasmStatement::Measure {
                    qubit: *q as usize,
                    clbit: measurement_idx,
                });
                measurement_idx += 1;
            }
            _ => {
                if let Some(qasm_gate) = qgate_to_qasm_gate(gate) {
                    program.statements.push(QasmStatement::Gate(qasm_gate));
                }
            }
        }
    }

    program
}

/// Convert a single QGate to QasmGate
#[allow(dead_code)]
fn qgate_to_qasm_gate(gate: &QGate) -> Option<QasmGate> {
    let result = match gate {
        QGate::H(q) => QasmGate::H(*q as usize),
        QGate::X(q) => QasmGate::X(*q as usize),
        QGate::Y(q) => QasmGate::Y(*q as usize),
        QGate::Z(q) => QasmGate::Z(*q as usize),
        QGate::T(q) => QasmGate::T(*q as usize),
        QGate::Tdg(q) => QasmGate::Tdg(*q as usize),
        QGate::Rx(q, angle) => QasmGate::Rx(*q as usize, *angle as f64),
        QGate::Ry(q, angle) => QasmGate::Ry(*q as usize, *angle as f64),
        QGate::Rz(q, angle) => QasmGate::Rz(*q as usize, *angle as f64),
        QGate::Phase(q, angle) => {
            // Check for S/Sdg
            let a = *angle as f64;
            if (a - PI / 2.0).abs() < 1e-6 {
                QasmGate::S(*q as usize)
            } else if (a + PI / 2.0).abs() < 1e-6 {
                QasmGate::Sdg(*q as usize)
            } else {
                QasmGate::P(*q as usize, a)
            }
        }
        QGate::U3(q, theta, phi, lambda) => {
            QasmGate::U3(*q as usize, *theta as f64, *phi as f64, *lambda as f64)
        }
        QGate::CNot(c, t) => QasmGate::CX(*c as usize, *t as usize),
        QGate::CZ(a, b) => QasmGate::CZ(*a as usize, *b as usize),
        QGate::Swap(a, b) => QasmGate::Swap(*a as usize, *b as usize),
        QGate::CPhase(c, t, angle) => QasmGate::CP(*c as usize, *t as usize, *angle as f64),
        QGate::CRz(c, t, angle) => QasmGate::CRz(*c as usize, *t as usize, *angle as f64),
        QGate::Toffoli(a, b, c) => QasmGate::CCX(*a as usize, *b as usize, *c as usize),
        QGate::CCZ(a, b, c) => QasmGate::CCZ(*a as usize, *b as usize, *c as usize),
        QGate::Measure(_) => return None, // Handled separately
    };

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_bell_state() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let qasm = emit_qasm3(&circuit, "bell");

        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[2] q"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_emit_rotations() {
        let circuit = vec![
            QGate::Rx(0, std::f32::consts::FRAC_PI_2),
            QGate::Ry(0, std::f32::consts::PI),
            QGate::Rz(0, std::f32::consts::FRAC_PI_4),
        ];
        let qasm = emit_qasm3(&circuit, "rotations");

        assert!(qasm.contains("rx(pi/2) q[0]"));
        assert!(qasm.contains("ry(pi) q[0]"));
        assert!(qasm.contains("rz(pi/4) q[0]"));
    }

    #[test]
    fn test_emit_s_gates() {
        let circuit = vec![
            QGate::Phase(0, std::f32::consts::FRAC_PI_2),
            QGate::Phase(0, -std::f32::consts::FRAC_PI_2),
        ];
        let qasm = emit_qasm3(&circuit, "s_gates");

        assert!(qasm.contains("s q[0]"));
        assert!(qasm.contains("sdg q[0]"));
    }

    #[test]
    fn test_emit_t_gates() {
        let circuit = vec![QGate::T(0), QGate::Tdg(1)];
        let qasm = emit_qasm3(&circuit, "t_gates");

        assert!(qasm.contains("t q[0]"));
        assert!(qasm.contains("tdg q[1]"));
    }

    #[test]
    fn test_emit_three_qubit() {
        let circuit = vec![QGate::Toffoli(0, 1, 2), QGate::CCZ(0, 1, 2)];
        let qasm = emit_qasm3(&circuit, "three_qubit");

        assert!(qasm.contains("ccx q[0], q[1], q[2]"));
        assert!(qasm.contains("ccz q[0], q[1], q[2]"));
    }

    #[test]
    fn test_emit_measurement() {
        let circuit = vec![QGate::H(0), QGate::Measure(0)];
        let qasm = emit_qasm3(&circuit, "measure");

        assert!(qasm.contains("bit[1] c"));
        assert!(qasm.contains("c[0] = measure q[0]"));
    }

    #[test]
    fn test_emit_minimal() {
        let circuit = vec![QGate::H(0)];
        let qasm = emit_qasm3_with_options(&circuit, "test", &EmitOptions::minimal());

        // Minimal should not have header
        assert!(!qasm.contains("OPENQASM"));
        assert!(qasm.contains("qubit[1] q"));
        assert!(qasm.contains("h q[0]"));
    }

    #[test]
    fn test_format_angle_pi() {
        let options = EmitOptions::default();

        assert_eq!(format_angle(PI, &options), "pi");
        assert_eq!(format_angle(-PI, &options), "-pi");
        assert_eq!(format_angle(PI / 2.0, &options), "pi/2");
        assert_eq!(format_angle(PI / 4.0, &options), "pi/4");
        assert_eq!(format_angle(-PI / 2.0, &options), "-pi/2");
    }

    #[test]
    fn test_roundtrip() {
        use super::super::parser::parse_qasm3;

        let original = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::Rz(1, std::f32::consts::FRAC_PI_4),
        ];

        let qasm = emit_qasm3(&original, "roundtrip");
        let parsed = parse_qasm3(&qasm).unwrap();

        assert_eq!(original.len(), parsed.len());

        // Check gate types match
        assert!(matches!(parsed[0], QGate::H(0)));
        assert!(matches!(parsed[1], QGate::CNot(0, 1)));
        assert!(matches!(parsed[2], QGate::T(0)));
        assert!(matches!(parsed[3], QGate::Rz(1, _)));
    }

    #[test]
    fn test_qgate_to_program() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::Measure(0)];
        let program = qgate_to_program(&circuit, "test");

        assert_eq!(program.num_qubits, 2);
        assert_eq!(program.num_clbits, 1);
        assert_eq!(program.statements.len(), 3);
    }
}
