//! TileFabric CPU - A processor built from simulation tiles
//!
//! Unlike the tile8 module which uses software emulation, TileFabric CPUs
//! execute by propagating signals through actual tile connections. Every
//! gate, register, and wire is a discrete tile evaluated by the simulation
//! engine.
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +------------------+
//! |   Program ROM    |---->|   Instruction    |
//! |   (Const tiles)  |     |   Decoder        |
//! +------------------+     +------------------+
//!          ^                       |
//!          |                       v
//! +------------------+     +------------------+
//! | Program Counter  |<----|   Control Unit   |
//! | (ProgramCounter) |     |   (And/Or/Mux)   |
//! +------------------+     +------------------+
//!                                  |
//!          +----------+------------+----------+
//!          |          |            |          |
//!          v          v            v          v
//!     +--------+  +--------+  +--------+  +--------+
//!     | Reg 0  |  | Reg 1  |  | Reg 2  |  | Reg 3  |
//!     +--------+  +--------+  +--------+  +--------+
//!          |          |            |          |
//!          +----------+-----+------+----------+
//!                           |
//!                           v
//!                    +------------+
//!                    |    ALU     |
//!                    | Add/Sub/.. |
//!                    +------------+
//! ```
//!
//! # Key Difference from tile8
//!
//! - tile8: `cpu.step()` runs a Rust match statement
//! - tile_cpu: `cpu.tick(sim)` calls `sim.tick_with_delays()` - tiles execute
//!
//! # Example
//!
//! ```rust,ignore
//! use engine::tile_cpu::{TileCpuBuilder, TileCpu};
//! use engine::simulation::Simulation;
//!
//! let mut sim = Simulation::new();
//! let cpu = TileCpuBuilder::new()
//!     .with_program(&[0x12, 0x17, 0x31]) // LDI R0,#2; LDI R1,#3; ADD R0,R1
//!     .build(&mut sim);
//!
//! // Execute via tile simulation - no software shortcuts
//! for _ in 0..10 {
//!     let stats = cpu.tick(&mut sim);
//!     println!("Critical path: {} deltas", stats.critical_path_deltas);
//! }
//!
//! assert_eq!(cpu.read_reg(&sim, 0), 5); // 2 + 3 = 5
//! ```

mod builder;
mod components;
mod datapath;
mod execute;
mod layout;
pub mod minimal;
pub mod synth_driver;
#[cfg(feature = "cranelift_jit")]
pub mod tile_jit;
pub mod v2_assembler;
pub mod v2_benchmarks;
mod v2_builder;
pub mod v2_compiler;
pub mod v2_compiler_bench;
pub mod v2_components;
pub mod v2_dense_stepper;
pub mod v2_device_cycle;
#[cfg(feature = "cuda")]
pub mod v2_device_gpu;
pub mod v2_disassembler;
pub mod v2_execute;
pub mod v2_fast;
pub mod v2_fast_array;
pub mod v2_full_phase_prover;
pub mod v2_ground_truth;
pub mod v2_hls_accel;
pub mod v2_iss;
#[cfg(feature = "cranelift_jit")]
pub mod v2_jit;
pub mod v2_mmio;
pub mod v2_mmio_devices;
pub mod v2_mul_island;
pub mod v2_parser;
pub mod v2_player;
pub mod v2_replay;
pub mod v2_route_materialize;
pub mod v2_routing;
pub mod v2_showcase;
pub mod v2_simt_baseline;
pub mod v2_simt_eval;
pub mod v2_simt_fabric;
#[cfg(feature = "cuda")]
pub mod v2_simt_gpu;
pub mod v2_stdlib;
pub mod v2_trace;
pub mod v2_visualization;
mod v2_wiring;
mod wiring;

pub use builder::TileCpuBuilder;
pub use components::{AluOp, ControlSignals};
pub use datapath::TileCpuDatapath;
pub use execute::{PhysicalCpu, TileCpu, TileCpuMetrics};
pub use layout::{PlacedComponent, TileLayout, WireRoute};
pub use minimal::MinimalCpu;
pub use v2_assembler::{V2AsmError, assemble_v2};
pub use v2_benchmarks::{
    V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2BenchmarkCase, V2BenchmarkDetailedOutcome,
    V2BenchmarkOutcome, V2CycleMetrics, V2FastBenchmarkOutcome, benchmark_cases,
    expected_cycles_for_case, expected_hash_for_case, hash_v2_final_state, long_benchmark_cases,
    run_v2_benchmark_case, run_v2_benchmark_detailed, run_v2_benchmark_fast,
    validate_benchmark_outcome, validate_benchmark_performance,
};
pub use v2_builder::{
    V2ArrayTopology, V2Builder, V2SynthConfig, build_v2_array, build_v2_array_with_synth,
    finalize_multi_cpu_chains,
};
pub use v2_compiler::{
    ArithOp, CmpOp, CompileError, Cond, Expr, Func, Program, Stmt, compile_program,
    compile_to_words,
};
pub use v2_compiler_bench::{
    V2_COMPILED_BENCHMARKS, V2CompiledBenchCase, V2CompiledBenchOutcome, run_v2_compiled_benchmark,
};
pub use v2_dense_stepper::{
    DensePipelinePreflight, DensePipelineRunOutcome, DensePipelineStepOutcome,
    DensePipelineStepper, DensePipelineUnsupportedOp,
};
pub use v2_device_cycle::{
    CommitSwapIncidence, DEVICE_PHASES, DeviceCyclePlan, DeviceCycleState, ReadbackTaps,
};
pub use v2_disassembler::{V2DecodedInstruction, decode_v2_word, disassemble_v2_word};
pub use v2_execute::{
    TileCpuV2, V2DebugBreakpoint, V2DebugRunResult, V2DebugSnapshot, V2DebugStopReason,
    V2HybridAssistCounters,
};
pub use v2_fast::{V2FastCpu, V2FastTickResult, hash_v2_iss_state};
pub use v2_fast_array::{
    CpuSnapshot, V2FastCpuPool, V2FastFabric, V2FastFabricTopology, V2ParallelFabric,
};
pub use v2_full_phase_prover::{
    CHARTER_PROGRAMS, CharterProgramReport, LaneCpuState, PHASE_NAMES, PhaseProfile, StaticProfile,
    UnifiedSlotUniverse, charter_cases, prove_charter,
};
pub use v2_ground_truth::{
    V2_GROUND_TRUTH_SCHEMA_VERSION, capture_v2_ground_truth_jsonl, write_v2_ground_truth_jsonl,
};
pub use v2_hls_accel::{
    V2MmioHlsAccelDevice, V2SocProgram, compile_source_with_accel, eval_accel_func_ref,
};
pub use v2_iss::{
    V2DiffConfig, V2DiffMismatch, V2DiffStats, V2Iss, V2IssState, run_v2_differential,
    run_v2_differential_dataset, run_v2_differential_mmio, run_v2_differential_with,
};
pub use v2_mmio::{
    V2_MMIO_BASE, V2_MMIO_END, V2_MMIO_SIZE, V2MmioDevice, V2MmioHandle, is_v2_mmio_addr,
};
pub use v2_mmio_devices::{
    DISPLAY_HEIGHT, DISPLAY_PIXELS, DISPLAY_WIDTH, DS_ERR_INVALID_CMD, DS_ERR_NO_SAMPLE,
    DS_ERR_OOB, DS_OK, DatasetSample, InferenceModel, MMIO_ACCEL_ARG_DATA, MMIO_ACCEL_ARG_SELECT,
    MMIO_ACCEL_RESULT, MMIO_CONSOLE_COUNT, MMIO_CONSOLE_DATA, MMIO_DATASET_CMD, MMIO_DATASET_DATA,
    MMIO_DATASET_STATUS, MMIO_DISPLAY_CMD, MMIO_DISPLAY_STATUS, MMIO_MAILBOX_IN, MMIO_MAILBOX_OUT,
    MMIO_MATH_A, MMIO_MATH_B, MMIO_MATH_CMD, MMIO_MATH_RESULT, MMIO_PBIT_CTRL, MMIO_PBIT_RESULT,
    MMIO_QUANTUM_CMD, MMIO_QUANTUM_DATA, MMIO_QUANTUM_PARAM, MMIO_QUANTUM_QUBIT, MMIO_RNG_DATA,
    MMIO_SNN_CMD, MMIO_SNN_DATA, MMIO_TIMER_CYCLE, V2_MMIO_DISPLAY_SNAPSHOT_KIND,
    V2_MMIO_MATH_SNAPSHOT_KIND, V2_MMIO_PBIT_SNAPSHOT_KIND, V2_MMIO_QUANTUM_SNAPSHOT_KIND,
    V2_MMIO_REF_SNAPSHOT_KIND, V2_MMIO_SNN_SNAPSHOT_KIND, V2LinkMailboxDevice,
    V2MmioCombinedDevice, V2MmioDatasetDevice, V2MmioDisplayDevice, V2MmioMathDevice,
    V2MmioPbitBridgeDevice, V2MmioQuantumDevice, V2MmioRefDevicePack, V2MmioSnnBridgeDevice,
};
pub use v2_parser::{ParseError, compile_source, parse_program};
pub use v2_player::{
    V2PlayerProgram, build_player_html, default_player_programs, export_default_player,
    write_player_html,
};
pub use v2_replay::{
    V2_REPLAY_MANIFEST_FILE, V2_REPLAY_MMIO_FINAL_FILE, V2_REPLAY_MMIO_INITIAL_FILE,
    V2_REPLAY_MMIO_MODE_NONE, V2_REPLAY_MMIO_MODE_SNAPSHOT, V2_REPLAY_PROGRAM_FILE,
    V2_REPLAY_SCHEMA_VERSION, V2_REPLAY_SCHEMA_VERSION_V1, V2_REPLAY_SCHEMA_VERSION_V2,
    V2_REPLAY_TRACE_FILE, V2ReplayBundle, capture_v2_replay_bundle,
    capture_v2_replay_bundle_with_mmio_snapshot, hash_trace_lines, read_v2_replay_bundle_dir,
    verify_v2_replay_bundle, verify_v2_replay_bundle_with_mmio, write_v2_replay_bundle_dir,
};
pub use v2_route_materialize::{
    V2MaterializeError, V2MaterializedRoute, materialize_route, materialize_routes, settle_route,
    validate_materialized_route,
};
pub use v2_routing::{
    V2EndpointClass, V2MultiRouteResult, V2RouteBounds, V2RouteCoord, V2RouteNet, V2RouteNetClass,
    V2RoutingConfig, V2RoutingDb,
};
pub use v2_simt_baseline::{
    SimtBaselineReport, SimtBaselineResult, run_scalar_baseline, run_scalar_baseline_case,
    run_scalar_baseline_corpus,
};
pub use v2_simt_eval::{LaneKernel, LanePack};
pub use v2_simt_fabric::V2SimtFabric;
pub use v2_stdlib::{
    V2_ARG0, V2_ARG1, V2_ARG2, V2_ARG3, V2_RET, V2_SP, V2_STDLIB_ABS, V2_STDLIB_CLAMP,
    V2_STDLIB_MAX, V2_STDLIB_MEMSET4, V2_STDLIB_MIN,
};
pub use v2_trace::{
    V2_TRACE_CSV_HEADER, V2_TRACE_ENTRY_FIELDS, V2_TRACE_SCHEMA_NAME, V2_TRACE_SCHEMA_VERSION,
    V2TraceDocument, V2TraceEntry, V2TraceLog, V2TraceMemEvent, V2TraceParseError, V2TraceProgram,
    V2TraceProgramRow, V2TraceRegWrite, V2TraceState,
};
pub use v2_visualization::{
    V2_TRACE_VIZ_INDEX_FILE, V2_TRACE_VIZ_SUMMARY_FILE, V2_TRACE_VIZ_TRACE_FILE,
    V2TraceVizDeltaKind, V2TraceVizStepSummary, V2TraceVizTimeline, V2VizCoord, V2VizFlowEdge,
    V2VizFrame, V2VizFrameMetrics, V2VizLayerLayout, V2VizRenderParams, V2VizSession,
    capture_v2_trace_viz_timeline, export_v2_trace_viz_bundle, load_v2_trace_viz_bundle_summaries,
    load_v2_trace_viz_trace_text, v2_viz_find_next_by_asm, v2_viz_find_next_by_delta,
    v2_viz_find_next_by_pc, v2_viz_find_prev_by_delta, v2_viz_frame_to_ppm_bytes,
    write_v2_trace_viz_summary_tsv, write_v2_viz_frame_ppm,
};
pub use wiring::{MuxTree4to1, MuxTree8to1, WiringContext};

/// Width of the CPU datapath in bits
pub const DATAPATH_WIDTH: usize = 8;

/// Number of general-purpose registers
pub const NUM_REGISTERS: usize = 4;

/// Maximum ROM size in bytes
pub const MAX_ROM_SIZE: usize = 64;

/// Maximum RAM size in cells (Sprint 189: expanded from 64 to 128)
pub const MAX_RAM_SIZE: usize = 128;

/// Maximum propagation deltas before declaring non-convergence
pub const MAX_PROPAGATION_DELTAS: u32 = 1000;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod v2_tests;
