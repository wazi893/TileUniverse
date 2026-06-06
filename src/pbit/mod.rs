//! Probabilistic Bit (P-bit) Simulation Module
//!
//! P-bits are probabilistic computing elements that fluctuate between 0 and 1
//! with tunable probability. They provide a room-temperature alternative to
//! quantum computing for solving combinatorial optimization problems.
//!
//! # Architecture
//!
//! - **PBit**: Single probabilistic bit with bias control
//! - **PBitNetwork**: Coupled p-bit array with Ising model dynamics
//! - **Sampler**: Gibbs sampling, simulated annealing, parallel tempering
//! - **Ising**: Problem encoding for optimization (MaxCut, TSP, etc.)
//!
//! # Physical Analogy
//!
//! Real p-bits are implemented using:
//! - Low-barrier magnetic tunnel junctions (MTJs)
//! - Thermal fluctuations at room temperature (~1 GHz flip rate)
//! - Spin-orbit torque for bias control
//!
//! # Example
//!
//! ```ignore
//! use engine::pbit::{PBitNetwork, PBitConfig, IsingProblem};
//!
//! // Create a 100-pbit network for MaxCut
//! let config = PBitConfig::new(100).with_beta(1.0);
//! let mut network = PBitNetwork::new(config);
//!
//! // Encode a MaxCut problem
//! let problem = IsingProblem::maxcut_random(100, 0.5, 42);
//! network.set_problem(&problem);
//!
//! // Run Gibbs sampling for 10000 steps
//! for _ in 0..10000 {
//!     network.step_gibbs();
//! }
//!
//! // Get best solution found
//! let (solution, energy) = network.best_solution();
//! ```

pub mod ising;
pub mod network;
pub mod pbit;
pub mod sampler;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports
pub use ising::{IsingProblem, MaxCutGraph, QuboMatrix};
pub use network::{PBitConfig, PBitNetwork, UpdateOrder};
pub use pbit::PBit;
pub use sampler::{
    AnnealingResult, AnnealingSchedule, GibbsSampler, ParallelTempering, SimulatedAnnealing,
};

#[cfg(feature = "cuda")]
pub use gpu::{GpuAnnealingResult, GpuPBitNetwork, GpuSimulatedAnnealing};
