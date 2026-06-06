//! Sprint 61.0: Hybrid Orchestration Layer
//!
//! Unifies classical, quantum, and observer substrates under a single
//! orchestration framework, inspired by RIKEN's JHPC-Quantum architecture.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      HybridScheduler                            │
//! │  - Job queue management                                         │
//! │  - Resource allocation                                          │
//! │  - Substrate routing                                            │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!               ┌───────────────┼───────────────┐
//!               ▼               ▼               ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │ ClassicalEngine │ │  QuantumEngine  │ │ ObserverEngine  │
//! │ impl Substrate  │ │ impl Substrate  │ │ impl Substrate  │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::{HybridScheduler, JobSpec, VariationalSpec};
//!
//! let mut scheduler = HybridScheduler::new()
//!     .with_classical(ClassicalSubstrate::new())
//!     .with_quantum(QuantumSubstrate::new(20));
//!
//! let job_id = scheduler.submit(JobSpec::Variational(VariationalSpec {
//!     ansatz: circuit,
//!     optimizer: Optimizer::COBYLA { maxiter: 100 },
//!     cost_function: CostFunction::ExpectationValue(hamiltonian),
//!     ..Default::default()
//! }));
//!
//! let result = scheduler.wait(job_id);
//! ```

mod bridge;
mod job;
mod result;
mod scheduler;
mod substrate;

// Phase 2: Concrete substrate implementations
mod classical;
mod observer;
mod quantum;

// Sprint 62.0: GPU-accelerated substrate
#[cfg(feature = "cuda")]
mod gpu;

// Sprint 62.0: Auto-router for substrate selection
mod router;

// Sprint 62.0: Circuit batching
mod batch;

// Sprint 62.0: Checkpointing for variational (requires serde)
#[cfg(feature = "config")]
mod checkpoint;

// Sprint 67.0: Variational algorithm support
mod expectation;
mod gradient;
mod optimizer;
mod variational;

pub use bridge::{BridgeData, BridgeDataType, DataBridge};
pub use job::{
    ClassicalOp, ClassicalSpec, CostFunction, Gate, JobSpec, MeasurementBasis, MeasurementSpec,
    ObserverSpec, ObserverTrigger, Optimizer, PauliTerm, QuantumCircuit, QuantumSpec,
    VariationalSpec,
};
pub use result::{
    DataRef, HybridJob, JobId, JobMetrics, JobResult, JobStage, JobState, StageMetrics,
    StageOutputs, StageResult,
};
pub use scheduler::{HybridScheduler, JobHandle, SchedulerConfig, SchedulerStats, SubstrateInfo};
pub use substrate::{Substrate, SubstrateCapabilities, SubstrateError, SubstrateId, SubstrateType};

// Concrete substrate implementations
pub use classical::ClassicalSubstrate;
pub use observer::ObserverSubstrate;
pub use quantum::QuantumSubstrate;

// Sprint 62.0: GPU substrate (requires cuda feature)
#[cfg(feature = "cuda")]
pub use gpu::GpuSubstrate;

// Sprint 62.0: Auto-router
pub use router::{AutoRouter, SubstrateRouteInfo};

// Sprint 62.0: Circuit batching
pub use batch::{BatchEntry, BatchResult, CircuitBatch, execute_batch, linspace, parameter_grid};

// Sprint 62.0: Checkpointing (requires config feature)
#[cfg(feature = "config")]
pub use checkpoint::{CheckpointManager, CheckpointMetadata, VariationalCheckpoint};

// Sprint 67.0: Variational algorithm support
pub use expectation::{
    ExpectationEvaluator, GroupedMeasurement, H2_GROUND_STATE_ENERGY, MeasurementBasisType,
    PauliGroup, PauliGrouping, h2_hamiltonian,
};
pub use gradient::{BatchGradientEstimator, GradientEstimator, NaturalGradientEstimator};
pub use optimizer::{ClassicalOptimizer, OptimizerState};
pub use variational::{
    VariationalConfig, VariationalExecutor, VariationalResult, execute_variational,
};
