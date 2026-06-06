//! Hardware Transpilation Module
//!
//! SPRINT 53.0: Hardware Transpilation & T-Gate Tracking
//! SPRINT 54.0: Optimization Pipeline Integration
//! SPRINT 55.0: OpenQASM 3.0 Support
//! SPRINT 56.0: Resource Estimation Dashboard
//! SPRINT 63.0: Dynamic Resource Estimation with Factory Scheduling
//! SPRINT 64.0: Topology-Aware Resource Estimation
//! SPRINT 69.0: Error-Aware Circuit Compilation
//!
//! This module provides circuit transpilation to hardware-native gate sets,
//! T-gate analysis for magic state resource estimation, OpenQASM 3.0
//! import/export for interoperability, comprehensive resource estimation
//! for fault-tolerant quantum computing planning, and error-aware compilation
//! with adaptive protocol selection.
//!
//! # Overview
//!
//! The transpilation system consists of:
//! - **Native Gates**: IBM (SX, Rz, CX) and IonQ (GPI, GPI2, MS) gate sets
//! - **IBM Transpiler**: Convert abstract circuits to IBM Falcon/Heron
//! - **IonQ Transpiler**: Convert abstract circuits to IonQ Aria
//! - **T-Gate Counting**: Analyze T-gate content for magic state budgeting
//! - **Circuit Analysis**: Fidelity estimation, timing, resource counting
//! - **Pre-Transpile Optimization**: Gate cancellation, rotation merging (Sprint 54)
//! - **T-Depth Minimization**: Parallelize T gates for factory throughput (Sprint 54)
//! - **Post-Transpile Optimization**: Hardware-native gate cancellation (Sprint 54)
//! - **OpenQASM 3.0**: Parse and emit QASM3 for interoperability (Sprint 55)
//! - **Resource Estimation**: Physical qubit counts, timing, FT overhead (Sprint 56)
//!
//! # Example
//!
//! ```ignore
//! use engine::transpile::{IBMTranspiler, IonQTranspiler, analyze_t_gates};
//! use engine::transpile::analysis::{analyze_ibm_circuit, HardwareComparison};
//! use engine::hardware::profiles;
//! use logic_fabric_core::quantum::QGate;
//!
//! // Create a simple circuit
//! let circuit = vec![
//!     QGate::H(0),
//!     QGate::CNot(0, 1),
//!     QGate::T(0),
//!     QGate::Measure(0),
//! ];
//!
//! // Analyze T-gates
//! let t_analysis = analyze_t_gates(&circuit);
//! println!("T-count: {}, T-depth: {}", t_analysis.t_count, t_analysis.t_depth);
//!
//! // Transpile to IBM
//! let ibm_transpiler = IBMTranspiler::heron();
//! let ibm_circuit = ibm_transpiler.transpile(&circuit, "my_circuit");
//! println!("IBM gates: {}, CX count: {}", ibm_circuit.gate_count(), ibm_circuit.count("cx"));
//!
//! // Transpile to IonQ
//! let ionq_transpiler = IonQTranspiler::aria();
//! let ionq_circuit = ionq_transpiler.transpile(&circuit, "my_circuit");
//! println!("IonQ gates: {}, MS count: {}", ionq_circuit.gate_count(), ionq_circuit.count("ms"));
//!
//! // Analyze transpiled circuit
//! let analysis = analyze_ibm_circuit(&ibm_circuit, &profiles::ibm_heron());
//! println!("Fidelity: {:.4}, Duration: {:.2}µs",
//!     analysis.estimated_fidelity,
//!     analysis.duration_ns / 1000.0);
//! ```
//!
//! # Integration with Magic State Budgeting
//!
//! ```ignore
//! use engine::transpile::{analyze_t_gates, estimate_magic_cost};
//! use engine::qram::DistillationConfig;
//!
//! let circuit = /* your circuit */;
//! let t_analysis = analyze_t_gates(&circuit);
//!
//! // Connect to Sprint 51 magic state budgeting
//! let distillation = DistillationConfig::for_error_rate(1e-12);
//! let magic_cost = estimate_magic_cost(&t_analysis, &distillation);
//!
//! println!("T-count: {}", magic_cost.t_count);
//! println!("Magic states needed: {}", magic_cost.raw_magic_states);
//! println!("Factory cycles: {}", magic_cost.factory_cycles);
//! ```

pub mod analysis;
pub mod ibm;
pub mod ionq;
pub mod native_gates;
pub mod native_optimize;
pub mod optimize;
pub mod qasm3;
pub mod resource_estimate;
pub mod t_count;
pub mod t_depth;

// Sprint 63.0: Dynamic Resource Estimation with Factory Scheduling
pub mod dynamic_estimate;

// Sprint 64.0: Topology-Aware Resource Estimation
pub mod topology_aware_estimate;

// Sprint 69.0: Error-Aware Circuit Compilation
pub mod adaptive_protocols;
pub mod circuit_rewrite;
pub mod error_analysis;
pub mod error_aware_compiler;
pub mod multi_objective;

// Re-exports for convenient access
pub use analysis::{
    CircuitAnalysis, FidelityBreakdown, HardwareComparison, analyze_abstract_circuit,
    analyze_ibm_circuit, analyze_ionq_circuit,
};
pub use ibm::{IBMTranspiler, IBMTranspilerConfig, ToffoliDecomposition, TranspileStats};
pub use ionq::{IonQTranspileStats, IonQTranspiler, IonQTranspilerConfig};
pub use native_gates::{
    IBMCircuit, IBMGate, IonQCircuit, IonQGate, ibm_decompositions, ionq_decompositions,
};
pub use native_optimize::{
    IBMOptimizationStats, IonQOptimizationStats, optimize_ibm_circuit, optimize_ionq_circuit,
};
pub use optimize::{
    OptimizationConfig, OptimizationLevel, OptimizationResult, OptimizationStats, optimize,
    optimize_aggressive, optimize_circuit,
};
pub use t_count::{
    MagicStateCost, TGateAnalysis, analyze_t_gates, ccz_t_count, estimate_magic_cost, mcz_t_count,
    toffoli_t_count,
};
pub use t_depth::{TDepthAnalysis, analyze_t_depth, minimize_t_depth};

// QASM3 re-exports (Sprint 55)
pub use qasm3::{
    EmitOptions, Lexer, ParseError, ParseResult, QasmGate, QasmProgram, QasmStatement, Token,
    TokenKind, emit_qasm3, emit_qasm3_with_options, parse_qasm3, parse_qasm3_strict,
};

// Resource Estimation re-exports (Sprint 56)
pub use resource_estimate::{
    FaultTolerantConfig, FaultTolerantTiming, NISQTiming, PhysicalResourceEstimate,
    ResourceEstimate, ResourceEstimator, estimate_resources, estimate_resources_ft,
};

// Dynamic Resource Estimation re-exports (Sprint 63)
pub use dynamic_estimate::{
    DynamicResourceEstimate, DynamicResourceEstimator, FaultTolerantConfig as DynamicFTConfig,
    StaticComparison,
};

// Topology-Aware Resource Estimation re-exports (Sprint 64)
pub use topology_aware_estimate::{
    TopologyAwareEstimate, TopologyAwareEstimator, TopologyComparison, TopologyType,
};

// Error-Aware Circuit Compilation re-exports (Sprint 69)
pub use adaptive_protocols::{AdaptiveProtocolSelector, ErrorBudgetTracker, ProtocolAssignment};
pub use circuit_rewrite::{CircuitRewriter, RewriteStrategy, RewrittenCircuit};
pub use error_analysis::{ErrorAnalyzer, ErrorModel, ErrorPropagation, GateCriticality};
pub use error_aware_compiler::{
    CompilationResult, CompilerConfig, ErrorAwareCircuitCompiler, OptimizationMetrics,
    OptimizationObjective,
};
pub use multi_objective::{
    ConfigurationSpace, Constraints, MultiObjectiveOptimizer, NormalizedMetrics, ParetoFrontier,
    ParetoPoint, TradeoffAnalysis,
};
