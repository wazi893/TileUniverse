pub mod algorithms;
pub mod assembler;
pub mod bench_api;
pub mod blueprint;
pub mod brain;
pub mod bus;
pub mod chungus; // Chungus neural architecture compiler
pub mod circuits_mod;
pub mod clock_domain; // Priority 6: Multi-Clock Domains
pub mod component;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda_tiles;
#[cfg(feature = "cuda")]
pub mod cuda_life;
#[cfg(feature = "cuda")]
pub mod cellular_player;
pub mod debug_macros;
pub mod demo;
pub mod distributed;
pub mod epic101;
pub mod experiments;
pub mod external_api;
pub mod field;
#[cfg(feature = "fpga")]
pub mod fpga; // SPRINT 80: FPGA synthesis via RustHDL
#[cfg(feature = "gpu-prototype")]
pub mod gpu_engine;
#[cfg(feature = "cuda")]
pub mod gpu_nn; // GPU neural network implementations
pub mod math_accelerator; // SPRINT 76.0: Mathematics as First-Class Hardware Accelerator
pub mod memory; // Priority 3: Memory Controller
pub mod net; // Network communication
pub mod quantum_ext; // Quantum JIT and Hamiltonians

pub mod hardware; // SPRINT 52.0: Hardware mapping for magic state factories
pub mod hybrid; // SPRINT 61.0: Hybrid orchestration layer (RIKEN-inspired)
pub mod lint;
pub mod metrics;
pub mod mitigation; // SPRINT 57.0: Error mitigation for NISQ (ZNE, readout mitigation)

pub mod number_theory;
pub mod organisms_mod;
pub mod pbit; // Probabilistic bit simulation for Ising machines
pub mod physics;
pub mod probe;
pub mod qec; // Quantum Error Correction: stabilizer simulation, codes, decoders
pub mod qram; // SPRINT 46.0: Quantum Random Access Memory

pub mod scripts;
pub mod search; // SPRINT 65.0: Quantum-Enhanced Neural Search Engine
pub mod simulation;
pub mod snn; // CHUNGUS 4: Neuromorphic Spiking Neural Network
pub mod synth; // Logic synthesis: AIG core, cut enumeration, technology mapping
pub mod tensor_network; // Tensor network simulation for shallow circuits
pub mod tile8;
pub mod tile_cpu; // Tile-based CPU: gate-level processor from simulation tiles
pub mod tiles; // Tile primitives and tilemap
pub mod transpile; // SPRINT 53.0: Hardware transpilation & T-gate tracking
pub mod universe;

// Re-exports for legacy compatibility
pub use physics::charge;
pub use physics::coupling;
pub use physics::diffuse;
pub use physics::feedback;
pub use physics::fieldstep;
pub use physics::heat;
pub use physics::interact;
pub use physics::reaction;

pub use brain as quantum_brain; // Legacy alias
pub use brain::sensors as rich_sensors; // Legacy alias for examples

// Legacy module aliases
pub use brain::batched as batched_brain;
pub use brain::classical as classical_brain;
pub use brain::configurable as configurable_nn;
pub use brain::handicapped as handicapped_classical_brain;
pub use circuits_mod as circuits;
pub use circuits_mod::constrained as constrained_circuits;
pub use circuits_mod::patterns;
pub use debug_macros as debug;
pub use experiments::circuit_archaeology;
pub use experiments::continuous_mutations;
pub use experiments::dynamic_environments;
pub use experiments::evolution as evolution_experiments;
pub use experiments::mutation_sweep;
pub use experiments::mutation_sweep as mutation_rate_sweep; // Alias for example
pub use experiments::noise_model;
pub use experiments::quantum_vs_classical;
pub use experiments::temporal_dynamics;
pub use experiments::three_way_comparison;
pub use gui::api as gui_api;
pub use gui::backend as gui_backend;
pub use gui::recorder;
pub use gui::render;
pub use organisms_mod as organisms;
pub use organisms_mod::ecosystem;
pub use organisms_mod::forager as quantum_forager;
pub use organisms_mod::quantum as organisms_quantum;
pub use physics::experiments as physics_experiments;
pub use physics::feedback as physics_feedback;
pub use physics::interact as physics_interact;
pub use physics::scenarios as physics_scenarios;

pub mod agent;

pub mod debugger;
pub mod diagnostics;
pub mod gui;
pub mod hdl; // Priority 4: HDL-to-Blueprint DSL compiler
pub mod slim_simulation;

// Backward compatibility re-exports for moved modules
pub use chungus::chungus2_assembler;
pub use chungus::chungus2_cpu;
pub use chungus::chungus3_compiler;
#[cfg(feature = "gpu-prototype")]
pub use gpu_engine::gpu_simulation;
#[cfg(feature = "gpu-prototype")]
pub use gpu_engine::gpu_slim;
pub use quantum_ext::hamiltonians;
pub use quantum_ext::quantum_jit;
pub use tiles::dirty_bitset;
pub use tiles::tile;
pub use tiles::tile_meta;
pub use tiles::tilemap;

// Re-export core modules
pub use logic_fabric_core::algebraic_fusion;
pub use logic_fabric_core::block_sparse_state;
#[cfg(feature = "cuda")]
pub use logic_fabric_core::cuda; // Core CUDA (Runtime, Quantum Kernels)
pub use logic_fabric_core::fusion;
pub use logic_fabric_core::hybrid_state;
pub use logic_fabric_core::qasm;
pub use logic_fabric_core::quantum;
pub use logic_fabric_core::sparse_state;
