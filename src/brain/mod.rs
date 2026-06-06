// Brain/Neural Network module - various brain architectures for organisms

pub mod batched;
pub mod classical;
pub mod configurable;
pub mod gated_feedforward;
pub mod handicapped;
pub mod hybrid_brain; // Hybrid NN+QUBO brain (exploitation + exploration)
mod mod_brain; // Unified brain abstraction (BrainType, Brain)
pub mod quantum;
pub mod qubo_brain; // QUBO brain using p-bit annealing
pub mod sensors;
pub mod spiking; // CHUNGUS 4: Spiking Neural Network brain

#[cfg(feature = "burn-compute")]
pub mod burn_brain;

// Re-export from mod_brain (unified abstraction)
pub use mod_brain::{Brain, BrainType};

// Re-export commonly used types from classical
pub use classical::{ClassicalBrain, Move};

// Re-export from handicapped
pub use handicapped::HandicappedClassicalBrain;

// Re-export from quantum brain
pub use quantum::{
    CircuitStats, FusedCircuitResult, MoveAction, MutationType, SensorReadings, SimpleRng,
    analyze_circuit, circuit_hash, circuit_hash_exact, count_entangling_gates, decode_action,
    decode_action_with_stay, encode_as_rotations, mutate_circuit, quantum_ness, random_circuit,
    run_circuit, run_circuit_auto, run_circuit_fused, run_circuit_noisy, sense_local_fields,
};

// Re-export from batched
pub use batched::{BrainWeights, forward_batch, forward_batch_shared_weights};

// Re-export from configurable
pub use configurable::{ConfigurableBrain, NNConfig};

// Re-export from gated_feedforward
pub use gated_feedforward::{GatedFeedforwardPopulation, GatedFeedforwardWeights};

// Re-export from sensors
pub use sensors::extract_sensors;

// Re-export from spiking (CHUNGUS 4: Neuromorphic)
pub use spiking::{SpikingBrain, SpikingBrainStats};

// Re-export from qubo_brain (p-bit annealing)
pub use qubo_brain::{DecisionBudget, QuboAblation, QuboBrain};

// Re-export from hybrid_brain (NN exploitation + QUBO exploration)
pub use hybrid_brain::{HybridConfig, HybridExplorationBrain};
