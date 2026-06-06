//! Tensor network simulation for quantum circuits.
//!
//! This module provides tensor network-based simulation for computing
//! amplitudes and simulating shallow circuits (10-50 qubits, depth < 100)
//! where tensor network methods outperform state vector simulation.
//!
//! # Key Components
//!
//! - [`Tensor`] - A tensor with explicit indices and complex data
//! - [`TensorIndex`] - Index label with unique ID and dimension
//! - [`TensorNetwork`] - Collection of tensors forming a network
//! - [`ContractionPath`] - Ordered sequence of tensor contractions
//!
//! # Example
//!
//! ```ignore
//! use engine::tensor_network::{circuit_to_network, compute_amplitude};
//! use logic_fabric_core::quantum::QGate;
//!
//! // Create a Bell state circuit
//! let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
//! let network = circuit_to_network(&circuit, 2);
//!
//! // Compute amplitude <00|Bell|00>
//! let amp = compute_amplitude(&network, 0b00, 0b00);
//! // amp ≈ 1/√2 for the Bell state
//! ```

mod amplitude;
mod circuit_convert;
mod contraction;
mod greedy;
mod network;
mod path;
mod path_optimizer;
mod slicing;
mod tensor;
mod viability;

// Re-export main types and functions
pub use amplitude::{
    circuit_amplitude, compute_all_output_amplitudes, compute_amplitude, compute_amplitudes,
};
pub use circuit_convert::circuit_to_network;
pub use contraction::{contract_as_matmul, contract_tensors, contract_tensors_smart};
pub use greedy::{GreedyMetric, execute_path, find_greedy_order};
pub use network::TensorNetwork;
pub use path::{
    ContractionPath, ContractionStep, PeakMemoryProfile, compute_peak_memory,
    compute_peak_memory_profile, verify_path_peak_memory,
};
pub use path_optimizer::{ContractionTree, PathConfig, TreeStats, find_optimal_path};
pub use slicing::{SliceMode, SliceSpec, SlicedPath, execute_sliced_path, plan_slicing};
pub use tensor::{Tensor, TensorIndex};
pub use viability::{
    DEFAULT_MEMORY_BUDGET, TensorNetworkViability, ViabilityReason, is_tensor_network_viable,
};
