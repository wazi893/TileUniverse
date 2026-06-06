// =============================================================================
// src/algorithms/mod.rs
// =============================================================================
//
// Quantum-classical hybrid algorithms that demonstrate the feedback loop.
//
// SPRINT 2.1b: Grover's search (quadratic speedup)
// SPRINT 2.3: Deutsch-Jozsa (exponential speedup)
// SPRINT 2.3: Bernstein-Vazirani (linear → constant speedup)
// Shor's factoring algorithm (exponential speedup)
// SPRINT 40.0: VQE (Variational Quantum Eigensolver)
// SPRINT 42.0: QAOA (Quantum Approximate Optimization Algorithm)
//

pub mod deutsch_jozsa;
pub mod grover;
pub mod qaoa;
pub mod shor;
pub mod vqe;

// Grover's algorithm
pub use grover::{
    grover_circuit, grover_program_for_tile, grover_success_rate, optimal_iterations, run_grover,
};

// Deutsch-Jozsa algorithm
pub use deutsch_jozsa::{DJOracleType, deutsch_jozsa_circuit, run_deutsch_jozsa};

// Bernstein-Vazirani algorithm
pub use deutsch_jozsa::{bernstein_vazirani_circuit, run_bernstein_vazirani};

// Shor's algorithm
pub use shor::{
    factor_with_shor, inverse_qft_circuit, modular_exp_circuit, qft_circuit, run_shor_quantum,
    shor_circuit,
};

// QAOA (Quantum Approximate Optimization Algorithm)
pub use qaoa::{
    Graph, QAOAConfig, QAOAOptimizer, QAOAResult, QUBOProblem, maxcut, maxcut_p, run_qaoa,
};
