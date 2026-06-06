//! Quantum Error Correction (QEC) Module
//!
//! This module provides efficient simulation of stabilizer codes for QEC:
//! - Stabilizer tableau representation (Aaronson-Gottesman)
//! - Clifford gate operations (H, S, CNOT, etc.)
//! - Pauli measurements with syndrome extraction
//! - Common QEC codes (repetition, Steane, surface code)
//!
//! # Fault-Tolerant Computing Layer
//!
//! The module also includes a complete fault-tolerant computing layer:
//! - Logical qubits with Pauli frame tracking
//! - Surface code error model with proper scaling
//! - Magic state supply with throughput/latency awareness
//! - T-gate injection (abstract and sampled modes)
//! - Resource estimation for large-scale circuits
//!
//! # Performance
//!
//! Stabilizer simulation enables:
//! - O(n²) memory for ANY n-qubit stabilizer state
//! - O(n²) per Clifford gate
//! - O(n²) per Pauli measurement
//! - Millions of qubits for QEC simulation
//!
//! Fault-tolerant layer enables:
//! - Fast resource estimation for any circuit size
//! - Pauli frame simulation for control logic verification
//! - Accurate spacetime volume calculations

// Core QEC modules
pub mod bacon_shor;
pub mod codes;
pub mod decoder;
pub mod gate_noise;
pub mod noise;
pub mod stabilizer;
pub mod union_find;
pub mod union_find_dn;

// MWPM decoder modules (Phase 78.1)
pub mod mwpm_decoder;
pub mod syndrome_graph;

// Fault-tolerant computing modules
pub mod ft_processor;
pub mod ft_types;
pub mod gate_decomp;
pub mod injection;
pub mod magic_supply;
pub mod qec_model;
pub mod scheduler;

#[cfg(feature = "cuda")]
pub mod gpu_stabilizer;

// Core QEC exports
pub use bacon_shor::{BaconShorCode, BaconShorDecoder, GaugeOperator, GaugePauliType};
pub use codes::{RepetitionCode, SteaneCode, SurfaceCode};
pub use decoder::UnionFindDecoder;
pub use gate_noise::{
    GateError, GateErrorModel, Pauli, sample_measurement_error, sample_preparation_error,
    sample_single_qubit_error, sample_two_qubit_error,
};
pub use noise::SimpleRng;
pub use stabilizer::{PauliOperator, StabilizerError, StabilizerTableau};
pub use union_find::UnionFindDecoderV2;
pub use union_find_dn::UnionFindDecoderDN;

// MWPM decoder exports (Phase 78.1)
pub use mwpm_decoder::MWPMDecoderFB;
pub use syndrome_graph::{
    DecodingGraph, PhenomenologicalGraphBuilder, SyndromeGraphBuilder, Vertex, WeightedEdge,
};

// Fault-tolerant computing exports
pub use ft_processor::{
    ExecutionResult, FTConfig, FTStats, FactoryAwareEstimate, FaultTolerantProcessor,
    MeasurementBasis, MeasurementOutcome, ResourceEstimate, StallSimulationResult, StepResult,
};
pub use ft_types::{
    DecoderChoice, DemandBin, KnownBasis, LogicalBasis, LogicalErrorBudget, LogicalGate,
    LogicalQubit, PauliFrame, ResourceDelta, ResourceEvent, ResourceLog, ResourceSummary,
    SimulationFidelity, TimedOp, UnknownReason,
};
pub use gate_decomp::{
    CCZExpansion, ExpansionConfig, ToffoliExpansion, expand_circuit, expand_gate,
};
pub use injection::{BatchInjectionResult, InjectionModel, InjectionResult};
pub use magic_supply::{
    BackpressureReport, FactoryConfig, FactoryFleet, FactoryPipeline, FleetBackpressureReport,
    FleetRequest, FleetStrategy, MagicStateResult, MagicSupply, MagicSupplyCost, PipelineRequest,
    StallEvent, SupplyBottleneck, estimate_factory_config,
};
pub use qec_model::{DecoderModel, PhenomenologicalNoise, QecModel};
pub use scheduler::{
    CycleCosts, ParallelSchedule, ScheduleAnalysis, ScheduledGate, Scheduler, compute_t_depth,
    compute_t_layers,
};

#[cfg(feature = "cuda")]
pub use gpu_stabilizer::GpuStabilizerTableau;
