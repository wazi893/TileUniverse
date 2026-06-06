//! CHUNGUS 4: Neuromorphic Spiking Neural Network Module
//!
//! A million-neuron spiking neural network system built on the TILE-8 CPU substrate.
//!
//! # Architecture
//!
//! - **Neurons**: Leaky Integrate-and-Fire (LIF) model with 8-byte state
//! - **Synapses**: 4-byte weighted connections with delays
//! - **Learning**: STDP + Reward-modulated STDP (R-STDP)
//! - **Scale**: 128 neurons per TILE-8 CPU, supports 1M+ CPUs
//! - **Routing**: Cross-CPU spike propagation via direct tile access
//!
//! # Example
//!
//! ```ignore
//! use engine::snn::{SNNConfig, SNNNetwork, LIFNeuron};
//!
//! // Create a small network
//! let config = SNNConfig::new(8, vec![64, 64], 5);
//! let mut network = SNNNetwork::new(config);
//!
//! // Run for 100 ticks
//! for tick in 0..100 {
//!     network.step(tick);
//! }
//! ```

pub mod config;
pub mod neuron;
pub mod stdp;
pub mod synapse;

// Phase 2+
pub mod network;
pub mod spike_router;

// Phase 3+
#[cfg(feature = "cuda")]
pub mod gpu_kernels;

// Phase 3b: EPIC 97-style fused GPU kernels
#[cfg(feature = "cuda")]
pub mod gpu_fused;

// Phase 6: Quantum-SNN Hybrid
pub mod quantum_hybrid;

// Phase 7: Curiosity-Driven Intrinsic Motivation (EPIC 121)
pub mod curiosity;

// Phase 8: TD-Style Credit Assignment (EPIC 122)
pub mod value_function;

// Phase 9: Population Coding (EPIC 123)
pub mod population_coding;

// Phase 10: Stabilizer Quantum Neurons (EPIC 130)
pub mod stabilizer_network;
pub mod stabilizer_neuron;

// Phase 10b: GPU-Accelerated Stabilizer Network (EPIC 135)
#[cfg(feature = "cuda")]
pub mod gpu_stabilizer_network;

// Phase 11: Hybrid Stabilizer-Cluster Network (EPIC 134)
pub mod stabilizer_hybrid;

// Phase 12: Quantum-Neural Analysis (Sprint 59.0 EPIC 136)
pub mod quantum_analysis;

// Phase 13: Quantum Eligibility Traces (Sprint 59.0 EPIC 137)
pub mod quantum_eligibility;

// Phase 14: Variational Quantum Circuits (Sprint 59.0 EPIC 138)
pub mod variational_cluster;

// Phase 15: Block Sparse Synapses for GPU (Sprint 59.0 EPIC 140)
pub mod block_sparse_synapses;

// Phase 16: Column-Major Stabilizer with CSR Synapses (Sprint 59.0 EPIC 141)
// GPU-optimal memory layout for batch CZ operations
#[cfg(feature = "cuda")]
pub mod column_major_stabilizer;

// Phase 17: E-prop (Eligibility Propagation) Learning Rule
// Based on Bellec et al. (2020) - can learn from scratch without pre-training
pub mod eprop;

// Milestone 1: DVS Retina Encoder for Neuromorphic Vision
pub mod dvs_retina;

// Milestone 1: MNIST Dataset Loader
pub mod mnist;

// Milestone 2: Tile-Based Routing Substrate
pub mod router_mesh;
pub mod tile_substrate;

// Milestone 3: GPU-Batched SNN Training
#[cfg(feature = "cuda")]
pub mod gpu_batch;

// Milestone 10: GPU MLP Readout (eliminates CPU bottleneck)
#[cfg(feature = "cuda")]
pub mod gpu_mlp;

// Milestone 11: MLP weights + CPU inference (no cuda dependency)
pub mod mlp_weights;

// Convergence S4: threshold neuron realized as a single ThresholdVia tile
pub mod tile_neuron;

// Re-exports
pub use config::{NeuronConfig, SNNConfig, STDPConfig};
pub use neuron::LIFNeuron;
pub use stdp::{apply_rstdp, apply_stdp};
pub use synapse::{CrossCpuSynapse, Synapse, SynapseFlags};

pub use network::{SNNNetwork, SNNTopology};
pub use tile_neuron::TileNeuron;
pub use spike_router::{SpikePacket, SpikeRouter};

#[cfg(feature = "cuda")]
pub use gpu_kernels::GpuSNNState;

#[cfg(feature = "cuda")]
pub use gpu_fused::{
    GpuFusedSNN,
    SNNSnapshot,
    SynapseCSRData,
    discriminative_2layer_thresholds,
    discriminative_3layer_thresholds,
    discriminative_channel_thresholds,
    // Milestone 4: Fisher LDA scoring + cross-class inhibition
    fisher_discriminative_scores,
    generate_columnar_csr,
    generate_dense_feedforward_csr,
    generate_direct_csr,
    // Milestone 3 fix: 2-layer readout (hidden = output)
    generate_discriminative_2layer_csr,
    generate_discriminative_2layer_csr_inhibitory,
    generate_discriminative_2layer_csr_weighted,
    // Milestone 6: 3-layer discriminative CSR + thresholds
    generate_discriminative_3layer_csr,
    // Milestone 6 Phase 3b: 3-layer with cross-class lateral inhibition in readout
    generate_discriminative_3layer_csr_inhibitory,
    // Milestone 10: 3-layer with within-class hidden recurrence
    generate_discriminative_3layer_csr_recurrent,
    // Milestone 3: discriminative channel connectivity
    generate_discriminative_channel_csr,
    generate_layered_csr,
    generate_random_csr,
};

#[cfg(feature = "cuda")]
pub use gpu_batch::GpuBatchSNN;

// MLP weights + live SNN model (unconditional — used by tile_cpu MMIO SNN bridge)
pub use mlp_weights::{CachedRates, LiveSnnModel, MlpWeights};

#[cfg(feature = "cuda")]
pub use gpu_mlp::GpuMLP;

#[cfg(feature = "cuda")]
pub use quantum_hybrid::GpuQuantumSNN;
pub use quantum_hybrid::{InterferenceMode, QuantumSNN, QuantumSNNConfig};

pub use curiosity::{CuriosityModule, PredictionErrorCuriosity, StateVisitationTracker};
pub use population_coding::{OutputEncoding, PopulationEncoder};
pub use value_function::{TDCreditAssignment, TDEligibilityTrace, ValueFunction};

// Stabilizer quantum neurons (EPIC 130)
pub use stabilizer_network::{
    StabilizerNetworkConfig, StabilizerNeuronNetwork, StabilizerTopology,
};
pub use stabilizer_neuron::{
    EntanglingGate, LayerType, StabilizerLayer, StabilizerNeuron, StabilizerNeuronConfig,
    StabilizerSynapse,
};

// GPU-accelerated stabilizer network (EPIC 135)
#[cfg(feature = "cuda")]
pub use gpu_stabilizer_network::{GpuStabilizerNetwork, StabilizerBackend};

// Hybrid stabilizer-cluster network (EPIC 134)
pub use stabilizer_hybrid::{
    AdaptiveClusterConfig, Complex64, FeedbackMode, FullQuantumCluster, HybridNetworkConfig,
    HybridQuantumNetwork, MagicState,
};

// Cluster-to-cluster bridges (Sprint 59.0 EPIC 139)
pub use stabilizer_hybrid::{BridgeGate, ClusterBridge};

// Quantum-neural analysis (Sprint 59.0 EPIC 136)
pub use quantum_analysis::{
    ClusterSnapshot, QuantumNeuralRecorder, QuantumNeuralSnapshot, RecordingStatistics,
};

// Quantum eligibility traces (Sprint 59.0 EPIC 137)
pub use quantum_eligibility::{
    EligibilityTrace, QuantumEligibilityConfig, QuantumEligibilityTraces, QuantumTraceApplicator,
};

// Variational quantum circuits (Sprint 59.0 EPIC 138)
pub use variational_cluster::{
    EntanglePattern, VariationalCircuit, VariationalConfig, VariationalLayer,
};

// E-prop learning (Phase 17)
pub use eprop::{EpropConfig, EpropNetwork, SurrogateGradient};

// DVS retina encoder (Milestone 1)
pub use dvs_retina::{DvsEvent, DvsRetina, EventStatistics, Polarity};

// MNIST dataset loader (Milestone 1)
pub use mnist::{Batch, BatchIterator, MnistDataset};
