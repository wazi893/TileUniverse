//! SPRINT 65.0: Quantum-Enhanced Neural Search Engine
//!
//! A hybrid search system combining:
//! - Massive parallelism (268M candidates)
//! - Spiking neural networks (fitness → spikes)
//! - Quantum interference (Grover amplification)
//! - Hive mesh communication (neighbor gradients)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Search Substrate (1M-268M candidates)                       │
//! │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                           │
//! │  │ C_0 │─│ C_1 │─│ C_2 │─│ ... │  (Hive mesh)              │
//! │  └──┬──┘ └──┬──┘ └──┬──┘ └─────┘                           │
//! │     └───────┴───────┴─── Fitness Evaluation                 │
//! │                    ↓                                        │
//! │              Spike Encoding (SNN)                           │
//! │                    ↓                                        │
//! │           Quantum Interference (Grover)                     │
//! │                    ↓                                        │
//! │             Selection & Mutation                            │
//! │                    ↓                                        │
//! │               Next Generation                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use engine::search::{QNSConfig, QuantumNeuralSearchEngine, OneMax};
//!
//! // Create engine with OneMax problem (count 1-bits)
//! let config = QNSConfig::default()
//!     .with_population(1_000_000)
//!     .with_problem(OneMax::new(64));
//!
//! let mut engine = QuantumNeuralSearchEngine::new(config);
//! let result = engine.run();
//!
//! println!("Best fitness: {}", result.best.fitness);
//! println!("Generations: {}", result.generations);
//! ```
//!
//! # Modules
//!
//! - [`candidate`] - Solution candidate encoding (64-bit)
//! - [`fitness`] - Fitness functions (OneMax, NK-Landscape, MaxSAT)
//! - [`substrate`] - Population manager with parallel evaluation
//! - [`selection`] - Selection strategies (quantum + classical)
//! - [`baseline`] - Classical GA baseline for comparison
//! - [`telemetry`] - Per-generation instrumentation
//! - [`spike`] - SNN spike encoding (Phase 2)
//! - [`hive`] - Hive mesh communication (Phase 3)
//! - [`quantum`] - Quantum Grover amplification (Phase 4)

pub mod baseline;
pub mod candidate;
pub mod fitness;
pub mod hive; // Phase 3: Hive mesh communication
pub mod quantum;
pub mod selection;
pub mod spike; // Phase 2: SNN spike encoding
pub mod substrate;
pub mod telemetry; // Phase 4: Quantum Grover amplification

// Re-exports for convenience
pub use baseline::{RouletteSelection, TournamentSelection, TruncationSelection};
pub use candidate::Candidate;
pub use fitness::{FitnessFunction, MaxSat, NKLandscape, OneMax};
pub use hive::{HiveConfig, HiveMesh, HiveSelector, NeighborGradient};
pub use quantum::{
    GroverSelector, GroverStats, HybridQuantumSelector, QuantumConfig, QuantumState,
};
pub use selection::SelectionStrategy;
pub use spike::{FitnessToSpikeEncoder, LIFNeuron, PopulationSpikeLayer, SpikeConfig, SpikeTrain};
pub use substrate::{SearchSubstrate, SubstrateStats};
pub use telemetry::{GenerationTelemetry, TelemetryCollector};

/// Default population size for CPU mode (1M for rapid iteration)
pub const DEFAULT_POPULATION_CPU: usize = 1_000_000;

/// Target population size for GPU mode (268M)
pub const TARGET_POPULATION_GPU: usize = 268_000_000;

/// Default candidate bit width
pub const DEFAULT_CANDIDATE_BITS: usize = 64;
