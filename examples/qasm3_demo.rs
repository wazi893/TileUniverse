//! OpenQASM 3.0 Import/Export Demo
//!
//! Sprint 55.0: Demonstrates QASM3 parsing and emission for
//! interoperability with Qiskit, Cirq, and other quantum tools.
//!
//! Run with: cargo run --example qasm3_demo

use engine::transpile::qasm3::{EmitOptions, emit_qasm3, emit_qasm3_with_options, parse_qasm3};
use logic_fabric_core::quantum::QGate;

fn main() {
    println!("===========================================");
    println!(" SPRINT 55.0: OpenQASM 3.0 Import/Export");
    println!("===========================================\n");

    // Example 1: Parse QASM3 to QGate circuit
    demo_parse_qasm3();

    // Example 2: Emit QGate circuit to QASM3
    demo_emit_qasm3();

    // Example 3: Round-trip conversion
    demo_roundtrip();

    // Example 4: Complex circuit with rotations
    demo_rotations();

    // Example 5: T-gate circuit for magic state analysis
    demo_t_gates();
}

fn demo_parse_qasm3() {
    println!("=== 1. Parse QASM3 to QGate ===\n");

    let qasm = r#"
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[1] c;

// Create Bell state
h q[0];
cx q[0], q[1];

// Add entanglement with third qubit
cx q[1], q[2];

// Measurement
measure q[0] -> c[0];
"#;

    println!("Input QASM3:");
    println!("{}", qasm);

    match parse_qasm3(qasm) {
        Ok(circuit) => {
            println!("Parsed {} gates:", circuit.len());
            for (i, gate) in circuit.iter().enumerate() {
                println!("  {}: {:?}", i + 1, gate);
            }
        }
        Err(e) => {
            println!("Parse error: {}", e);
        }
    }
    println!();
}

fn demo_emit_qasm3() {
    println!("=== 2. Emit QGate to QASM3 ===\n");

    // Create a GHZ state preparation circuit
    let circuit = vec![
        QGate::H(0),
        QGate::CNot(0, 1),
        QGate::CNot(1, 2),
        QGate::CNot(2, 3),
        QGate::Measure(0),
        QGate::Measure(1),
        QGate::Measure(2),
        QGate::Measure(3),
    ];

    println!("QGate circuit:");
    for gate in &circuit {
        println!("  {:?}", gate);
    }

    let qasm = emit_qasm3(&circuit, "ghz_state");
    println!("\nEmitted QASM3:");
    println!("{}", qasm);
}

fn demo_roundtrip() {
    println!("=== 3. Round-trip Conversion ===\n");

    // Start with a QGate circuit
    let original = vec![
        QGate::H(0),
        QGate::T(0),
        QGate::CNot(0, 1),
        QGate::Rz(1, std::f32::consts::FRAC_PI_4),
        QGate::CZ(0, 1),
    ];

    println!("Original circuit:");
    for gate in &original {
        println!("  {:?}", gate);
    }

    // Convert to QASM3
    let qasm = emit_qasm3(&original, "roundtrip_test");
    println!("\nConverted to QASM3:");
    println!("{}", qasm);

    // Parse back to QGate
    match parse_qasm3(&qasm) {
        Ok(parsed) => {
            println!("Parsed back ({} gates):", parsed.len());
            for gate in &parsed {
                println!("  {:?}", gate);
            }

            // Verify same number of gates
            if original.len() == parsed.len() {
                println!("\nRound-trip successful: gate counts match");
            }
        }
        Err(e) => {
            println!("Round-trip failed: {}", e);
        }
    }
    println!();
}

fn demo_rotations() {
    println!("=== 4. Rotation Gates ===\n");

    let qasm = r#"
OPENQASM 3.0;
qubit[2] q;

// Various rotation gates with pi expressions
rx(pi/2) q[0];
ry(pi/4) q[0];
rz(pi) q[0];

// U3 universal gate
u3(pi/2, 0, pi) q[1];

// Controlled rotations
crz(pi/4) q[0], q[1];
"#;

    println!("Input QASM3:");
    println!("{}", qasm);

    match parse_qasm3(qasm) {
        Ok(circuit) => {
            println!("Parsed gates:");
            for gate in &circuit {
                println!("  {:?}", gate);
            }
        }
        Err(e) => {
            println!("Parse error: {}", e);
        }
    }
    println!();
}

fn demo_t_gates() {
    println!("=== 5. T-Gate Circuit (Magic State Analysis) ===\n");

    // Create a circuit with T gates for magic state analysis
    let circuit = vec![
        QGate::H(0),
        QGate::H(1),
        QGate::T(0),
        QGate::CNot(0, 1),
        QGate::T(1),
        QGate::Tdg(0),
        QGate::H(0),
        QGate::T(0),
    ];

    println!("T-gate circuit:");
    for gate in &circuit {
        println!("  {:?}", gate);
    }

    // Emit with minimal options (no header)
    let minimal_qasm = emit_qasm3_with_options(&circuit, "t_circuit", &EmitOptions::minimal());
    println!("\nMinimal QASM3 (no header):");
    println!("{}", minimal_qasm);

    // Emit with Qiskit-compatible options
    let qiskit_qasm = emit_qasm3_with_options(&circuit, "t_circuit", &EmitOptions::qiskit());
    println!("Qiskit-compatible QASM3:");
    println!("{}", qiskit_qasm);

    // Count T gates
    let t_count = circuit
        .iter()
        .filter(|g| matches!(g, QGate::T(_) | QGate::Tdg(_)))
        .count();
    println!("T-gate count: {}", t_count);
    println!("(Connects to Sprint 51 magic state budgeting)\n");
}
