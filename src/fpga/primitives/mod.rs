//! SPRINT 80: Hardware primitive implementations
//!
//! This module contains RustHDL LogicBlock implementations for all
//! TileType variants that can be synthesized to FPGA.
//!
//! # Organization
//!
//! Primitives are organized by function:
//! - `combinational`: Logic gates (And, Or, Xor, Not) and routing (Wire, Cross)
//! - `arithmetic`: Math operations (Add, Sub, Mul, Div, etc.)
//! - `comparison`: Relational operators (Lt, Gt, Eq, etc.)
//! - `sequential`: Registers, latches, and clocked elements
//! - `cpu`: CPU building blocks (Decoder, Mux8to1, ProgramCounter)

pub mod arithmetic;
pub mod combinational;
pub mod comparison;
pub mod cpu;
pub mod sequential;

// Re-export all primitives
pub use arithmetic::*;
pub use combinational::*;
pub use comparison::*;
pub use cpu::*;
pub use sequential::*;
