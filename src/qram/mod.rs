//! Quantum Random Access Memory (QRAM)
//!
//! SPRINT 46.0: QRAM Foundations
//! SPRINT 47.0: Fault-Tolerant QRAM with QEC
//!
//! Implements bucket-brigade QRAM architecture for superposition queries.
//! QRAM enables quantum algorithms to query classical data in superposition,
//! providing the basis for quantum speedups in algorithms like Grover search,
//! HHL, and quantum machine learning.
//!
//! # Architecture
//!
//! The bucket-brigade QRAM uses a binary tree of quantum routers:
//! ```text
//!          Level 0:     [R_0]           (1 router)
//!                       /    \
//!          Level 1:  [R_1]   [R_2]      (2 routers)
//!                    /   \   /   \
//!          Data:   [D_0][D_1][D_2][D_3]  (4 memory cells)
//! ```
//!
//! Each router implements a Fredkin (CSWAP) gate that conditionally routes
//! data based on the address qubit state.
//!
//! # Fault-Tolerant QRAM (Sprint 47.0)
//!
//! Memory cells can be protected using Steane [[7,1,3]] encoding:
//! ```text
//! Physical Cell:    [data]     (1 qubit)
//! Protected Cell:   [d0][d1][d2][d3][d4][d5][d6]  (7 qubits)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::{QRAMQuery, ProtectedMemoryCell};
//!
//! // Create QRAM with 2^3 = 8 memory cells
//! let mut qram = QRAMQuery::new(3);
//! qram.load(&[100, 200, 300, 400, 500, 600, 700, 800]);
//!
//! // Classical query
//! let data = qram.query_classical(5);
//! assert_eq!(data, Some(600));
//!
//! // Protected memory cell with error correction
//! let mut cell = ProtectedMemoryCell::new(42);
//! cell.inject_error(3, 'X');  // Single-qubit error
//! let result = cell.correct_errors();  // Steane correction
//! assert!(result.success);
//! ```
//!
//! # References
//!
//! - [Bucket-brigade QRAM (Giovannetti et al.)](https://arxiv.org/abs/0708.1879)
//! - [Experimental QRAM 2025](https://arxiv.org/abs/2506.16682)
//! - [QRAM Polynomial Encoding (arXiv:2408.16794)](https://arxiv.org/abs/2408.16794)
//!
//! # Sprint 49.0: Large-Scale Sparse QRAM
//!
//! The sparse_state module enables bucket-brigade QRAM with thousands of cells
//! by using block-sparse quantum state representations instead of dense vectors.

pub mod distillation;
pub mod ft_qram;
#[cfg(feature = "gpu-prototype")]
pub mod gpu_sparse;
pub mod graph_router;
pub mod logical_ops;
pub mod magic_budget;
pub mod packed_frame;
pub mod pauli_frame;
pub mod poly_router;
pub mod polynomial;
pub mod protected_cell;
pub mod qlut_poly;
pub mod qram_poly;
pub mod query;
pub mod router;
pub mod sparse_bucket;
pub mod sparse_state;
pub mod stabilizer_qram;
pub mod toffoli_decomp;
pub mod tree;

// Sprint 63.0: Magic State Factory Scheduling
pub mod factory_pool;
pub mod factory_scheduler;
pub mod schedule_optimizer;
pub mod throughput_model;

pub use distillation::{DistillationFactory, DistillationProtocol, FactoryResources};
pub use ft_qram::{
    FTQRAMConfig, FTQRAMStats, FTQueryResult, FaultTolerantQRAM, threshold_analysis,
};
pub use graph_router::{
    GraphRouteResult, GraphRouter, GraphRouterConfig, GraphRouterResourceComparison, GraphTopology,
    RoutingGraph,
};
pub use logical_ops::{
    LogicalGateResult, LogicalRouter, flagged_toffoli, logical_fredkin, transversal_cnot,
};
pub use magic_budget::{
    DistillationConfig, DistillationProtocolType, FactoryEstimate, MagicStateBudget,
};
pub use packed_frame::{PackedFrameStats, PackedPauliFrame64, SurrogateModel};
pub use pauli_frame::{
    MeasurementDependency, PauliFrame, PauliType, ResolvedFrame, TrackedOperation,
};
pub use poly_router::{
    PolyRouteResult, PolyRouter, PolyRouterConfig, TDepthComparison, encode_memory_as_phases,
    identity_phases,
};
pub use polynomial::{
    AddressPolynomial, CliffordTSynthesizer, SynthesisResult, TGateStats, analyze_t_gates,
    estimate_bucket_brigade_t_depth, estimate_polynomial_t_depth,
};
pub use protected_cell::{
    CellStatistics, CorrectionResult, ProtectedMemoryBank, ProtectedMemoryCell,
};
pub use qlut_poly::{QLUTPoly, QLUTPolyConfig, QLUTQueryResult, QLUTResourceComparison};
pub use qram_poly::{PolyQueryResult, QRAMComparison, QRAMPoly, QRAMPolyConfig};
pub use query::{QRAMQuery, QueryResult};
pub use router::{QRAMRouter, RouterState, fredkin_gates};
pub use sparse_bucket::{SparseBBComparison, SparseBucketBrigade};
pub use sparse_state::{
    HybridSparseAdapter, SparseQRAMConfig, SparseQRAMState, SparseQueryResult, SparseStateStats,
};
pub use stabilizer_qram::{
    FullResourceComparison, StabilizerQRAM, StabilizerQRAMStats, StabilizerQueryResult,
    StabilizerResourceComparison,
};
pub use toffoli_decomp::{ToffoliContext, ToffoliDecomposition, ToffoliTracker, TrackedToffoli};
pub use tree::QRAMTree;

// Sprint 72.0: GPU-Accelerated Sparse QRAM
#[cfg(feature = "gpu-prototype")]
pub use gpu_sparse::{GpuBucketBrigade, GpuQueryResult, GpuSparseAdapter};

// Sprint 63.0: Magic State Factory Scheduling
pub use factory_pool::{
    AutoConfigStrategy, FactoryPool, FactoryPoolBuilder, FactoryPoolStats,
    binary_search_factory_count, estimate_schedule_length, select_protocol_for_error,
};
pub use factory_scheduler::{
    DependencyGraph, DependencyGraphStats, Factory, FactoryScheduler, MagicStateRequest, Schedule,
    ScheduleMetrics, TGateNode, TGateType, UtilizationStats,
};
pub use schedule_optimizer::{
    OptimizationComparison, OptimizationStrategy, PriorityRule, ScheduleOptimizer, ScheduleResult,
    compare_strategies,
};
pub use throughput_model::{
    BottleneckType, ExecutionEstimate, ScalingAnalysis, TDistribution, ThroughputModel,
};
