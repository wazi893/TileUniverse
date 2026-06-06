// Distributed computing module - multi-core and multi-node tile farms

#[cfg(feature = "quantum_jit")]
pub mod tile_farm;

#[cfg(feature = "quantum_jit")]
pub mod distributed_farm;

#[cfg(feature = "perf-bench")]
pub mod kernel_farm;

// SPRINT 39.0 Phase 2: Multi-core block-sparse quantum execution
pub mod quantum_worker_pool;

// Re-export commonly used types
#[cfg(feature = "quantum_jit")]
pub use tile_farm::{
    AutoConfigChoice, AutoConfigInput, BackendKind, KernelJob, KernelSpec, KernelType, TileFarm,
    TileRange, WorkerMode, choose_auto_config, jit_worker_count, jit_worker_mode,
};

#[cfg(feature = "quantum_jit")]
pub use distributed_farm::{
    AggregatedMetrics, ChannelTransport, ClusterConfig, ClusterMeta, DistributedJob,
    DistributedResult, DistributedTileFarm, NodeConfig, NodeHandle, NodeHealth, NodeHealthInfo,
    NodeMetrics, NodeTransport, TcpTransport, aggregate_results_parallel, run_pipelined_batches,
    run_pipelined_batches_with_queue_depth, tcp_worker_main,
};

#[cfg(feature = "perf-bench")]
pub use kernel_farm::{KernelCtx, KernelFarm, KernelKind, KernelRef, KernelStats};

// SPRINT 39.0 Phase 2: Re-export quantum worker pool types
pub use quantum_worker_pool::{
    AutoConfigInput as QuantumAutoConfigInput, BlockRange, QuantumWorkerPool, WorkerConfig,
    auto_config_for_block_sparse, partition_blocks,
};
