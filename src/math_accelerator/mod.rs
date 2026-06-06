//! Sprint 76.0: Mathematics as First-Class Hardware Accelerator
//!
//! This module connects algebraic analysis to backend routing decisions.
//! Mathematical structure determines execution path, not hardware defaults.
//!
//! # Philosophy (EPIC 83)
//!
//! > "Mathematics is the ultimate accelerator. When algebra can prove work
//! > equals identity, no hardware can beat doing nothing."
//!
//! The extension is:
//!
//! > "When mathematical analysis can predict structure, route to the backend
//! > that exploits it."
//!
//! # Module Organization
//!
//! - [`characteristics`]: Circuit characteristics data structures
//! - [`quick_analysis`]: Fast O(n) pre-analysis for routing decisions
//! - [`metrics`]: GHZ/W state pattern detection, sparsity prediction
//! - [`selector`]: Backend selection logic using algebraic stats

mod characteristics;
mod metrics;
mod quick_analysis;
mod selector;

pub use characteristics::{
    CircuitCharacteristics, CircuitPattern, QuickClassification, SparsityPrediction,
};
pub use metrics::{
    count_distinct_hadamard_qubits,
    count_two_qubit_gates,
    detect_ghz_pattern,
    detect_w_state_pattern,
    // Phase 77.1: Clifford detection
    is_clifford_angle,
    is_clifford_circuit,
    is_non_clifford_gate,
    predict_sparsity,
};
pub use quick_analysis::quick_classify;
pub use selector::{
    BackendDecision, SparseVariant, analyze_circuit, select_backend,
    select_backend_with_characteristics,
};

#[cfg(test)]
mod tests;
