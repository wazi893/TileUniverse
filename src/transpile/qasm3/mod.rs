//! OpenQASM 3.0 Import/Export Module
//!
//! SPRINT 55.0: OpenQASM 3.0 Support
//!
//! This module provides parsing and emission of OpenQASM 3.0 circuits,
//! enabling interoperability with Qiskit, Cirq, and other quantum tools.
//!
//! # OpenQASM 3.0 Overview
//!
//! OpenQASM 3.0 is the standard quantum assembly language for describing
//! quantum circuits. Key features supported:
//!
//! - Qubit and classical bit declarations
//! - Standard gates (h, x, y, z, cx, cz, swap, etc.)
//! - Rotation gates (rx, ry, rz, u3)
//! - T and Tdg gates
//! - Multi-controlled gates (ccx/toffoli, ccz)
//! - Measurement
//!
//! # Example
//!
//! ```ignore
//! use engine::transpile::qasm3::{parse_qasm3, emit_qasm3};
//! use logic_fabric_core::quantum::QGate;
//!
//! // Parse QASM3 to QGate circuit
//! let qasm = r#"
//!     OPENQASM 3.0;
//!     qubit[2] q;
//!     h q[0];
//!     cx q[0], q[1];
//! "#;
//! let circuit = parse_qasm3(qasm).unwrap();
//!
//! // Emit QGate circuit to QASM3
//! let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
//! let qasm_out = emit_qasm3(&circuit, "bell_state");
//! ```
//!
//! # Supported Gates
//!
//! | QASM3 | QGate | Notes |
//! |-------|-------|-------|
//! | `h q[0]` | `H(0)` | Hadamard |
//! | `x q[0]` | `X(0)` | Pauli X |
//! | `y q[0]` | `Y(0)` | Pauli Y |
//! | `z q[0]` | `Z(0)` | Pauli Z |
//! | `t q[0]` | `T(0)` | T gate (π/8) |
//! | `tdg q[0]` | `Tdg(0)` | T-dagger |
//! | `s q[0]` | `Phase(0, π/2)` | S gate |
//! | `sdg q[0]` | `Phase(0, -π/2)` | S-dagger |
//! | `rx(θ) q[0]` | `Rx(0, θ)` | X rotation |
//! | `ry(θ) q[0]` | `Ry(0, θ)` | Y rotation |
//! | `rz(θ) q[0]` | `Rz(0, θ)` | Z rotation |
//! | `p(θ) q[0]` | `Phase(0, θ)` | Phase gate |
//! | `u3(θ,φ,λ) q[0]` | `U3(0,θ,φ,λ)` | Universal |
//! | `cx q[0], q[1]` | `CNot(0, 1)` | CNOT |
//! | `cz q[0], q[1]` | `CZ(0, 1)` | Controlled-Z |
//! | `swap q[0], q[1]` | `Swap(0, 1)` | SWAP |
//! | `ccx q[0], q[1], q[2]` | `Toffoli(0,1,2)` | Toffoli |
//! | `ccz q[0], q[1], q[2]` | `CCZ(0,1,2)` | CCZ |
//! | `measure q[0] -> c[0]` | `Measure(0)` | Measurement |

mod ast;
mod emitter;
mod lexer;
mod parser;

pub use ast::{QasmGate, QasmProgram, QasmStatement};
pub use emitter::{EmitOptions, emit_qasm3, emit_qasm3_with_options};
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::{ParseError, ParseResult, parse_qasm3, parse_qasm3_strict};

/// QASM3 version supported
pub const QASM3_VERSION: &str = "3.0";

/// Standard library include (stdgates.inc)
pub const STDGATES_INC: &str = "stdgates.inc";
