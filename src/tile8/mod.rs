//! EPIC 108: TILE-8 CPU Toolchain
//!
//! A minimal 8-bit CPU that runs on the tile simulation fabric.
//!
//! Components:
//! - `asm`: Assembler (text → binary)
//! - `cpu`: CPU builder (binary → tile layout)
//! - `run`: Execution harness
//!
//! Architecture:
//! - 8-bit data width
//! - 256-byte address space (8-bit addresses)
//! - 4 general-purpose registers (R0-R3)
//! - 16 instructions (4-bit opcode)
//! - Zero and Carry flags

pub mod asm;
pub mod circuit;
pub mod cpu;
pub mod graph_state;
pub mod grid_allocator;
pub mod grover;
pub mod hybrid_search; // Sprint 66: Hybrid Fitness-Grover Integration
pub mod isa;
pub mod ising_mode; // Tile8 Ising Mode: P-bit dynamics on tile substrate
#[cfg(feature = "cuda")]
pub mod ising_mode_gpu;
pub mod observer_grid;
pub mod physical;
pub mod primitives;
pub mod profiler;
pub mod quantum_ops;
pub mod quantum_router;
pub mod quantum_router_f64;
#[cfg(feature = "cuda")]
pub mod quantum_router_f64_gpu;
pub mod sparse_noise;
pub mod sparse_quantum;
pub mod sparse_quantum_bigint;
#[cfg(feature = "gpu-prototype")]
pub mod sparse_quantum_gpu;
pub mod sparse_quantum_hybrid;
pub mod sparse_quantum_vec; // GPU-accelerated Ising mode

#[cfg(test)]
mod quantum_test;

pub use asm::Assembler;
pub use cpu::Tile8Cpu;
pub use grover::{GroverConfig, GroverResult, GroverSearch};
pub use hybrid_search::{
    Candidate, EvolutionResult, GenerationStats, HybridConfig, HybridSearch, SelectionStats,
    Xorshift64,
};
pub use isa::{Instruction, Opcode, Register};
pub use ising_mode::{AnnealResult, GridMaxCut, IsingConfig, IsingGrid, IsingRng, MaxCutResult};
#[cfg(feature = "cuda")]
pub use ising_mode_gpu::{GpuAnnealResult, GpuGridMaxCut, GpuIsingGrid, GpuMaxCutResult};
pub use observer_grid::{
    AnomalyDetector, BLOCK_SIZE, BlockDistribution, DistributionType, ObservationRule,
    ObservationState, ObservationStats, Observer, ObserverGrid, ObserverGridConfig,
};
pub use physical::{
    CpuInstance, CpuState, GpuCpuInstance, PhysicalCpu, calculate_grid_size, create_cpu_states,
    instantiate_cpus, instantiate_cpus_gpu, instantiate_cpus_gpu_direct,
    instantiate_cpus_gpu_lightweight, run_all_cpus, run_all_cpus_fast, run_parallel, step_parallel,
};
pub use primitives::{
    TileComponent, TilePlacement, connect_wire, eval_full_adder, get_output, place_component,
    set_input,
};
pub use profiler::{OperationStats, Profiler};
pub use sparse_noise::{
    AdaptivePruning, ErrorBranch, NoiseStats, Pauli, PauliString, SparseNoisyState,
};
pub use sparse_quantum::{GhzVerification, SparseBlock, SparseGridStats, SparseQuantumGrid};
pub use sparse_quantum_bigint::{
    GhzVerificationBigInt, GhzVerificationFast, SparseQuantumGridBigInt,
};
pub use sparse_quantum_hybrid::{SparseQuantumGridHybrid, SparseQuantumGridSmall};
pub use sparse_quantum_vec::{
    MinimalGhzState,
    MinimalGhzVerification,
    SimplifiedBinomial,
    SparseQuantumGridVec,
    // Symbolic W-state types and functions
    SymbolicAmplitude,
    // Symbolic Dicke state types and functions
    SymbolicBinomial,
    SymbolicDickeState,
    SymbolicDickeVerification,
    SymbolicEntropy,
    SymbolicFraction,
    SymbolicGhzState,
    SymbolicGhzVerification,
    SymbolicNumber,
    SymbolicWState,
    SymbolicWVerification,
    UnlimitedGhzState,
    UnlimitedGhzVerification,
    WStateVerificationVec,
    create_ghz_power_of_10,
    create_graham_dicke,
    create_graham_ghz,
    create_graham_w,
    create_infinite_dicke,
    create_infinite_ghz,
    create_infinite_w,
    create_minimal_ghz,
    create_symbolic_dicke,
    create_symbolic_ghz,
    create_symbolic_w,
    create_tree_dicke,
    create_tree_ghz,
    create_tree_w,
    create_unlimited_ghz,
};
