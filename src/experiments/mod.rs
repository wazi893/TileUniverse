// Experiments module - research experiments and analysis tools

pub mod circuit_archaeology;
pub mod continuous_mutations;
pub mod dynamic_environments;
pub mod evolution;
pub mod facade;
pub mod lineage;
pub mod mutation_sweep;
pub mod noise_model;
pub mod phase_diagram;
pub mod quantum_vs_classical;
pub mod temporal_dynamics;
pub mod three_way_comparison;

// Re-export from facade (unified experiment interface)
pub use facade::{UnifiedExperiment, UnifiedResult, run_unified};

// Re-export from evolution
pub use evolution::{
    AggregatedStats, ComparisonRow, ComparisonSummary, ExperimentConfig, ExperimentResult,
    ExperimentSuite, RunResult as EvolutionRunResult, export_csv, export_timeseries_csv,
    mutation_rate_suite, quantum_advantage_suite, resource_scarcity_suite, run_experiment,
};

// Re-export from quantum_vs_classical
pub use quantum_vs_classical::{
    ComparisonConfig, ComparisonResult, ComparisonSummary as QvcComparisonSummary,
    ConstrainedForager, ConstrainedPopulation, RunResult as QvcRunResult, print_comparison,
    run_quantum_vs_classical,
};

// Re-export from three_way_comparison
pub use three_way_comparison::{
    BrainForager, BrainPopulation, GroupResult, PairwiseComparison, ThreeWayConfig,
    ThreeWayResults, TrialResult, print_three_way_results, run_three_way_comparison,
};

// Re-export from mutation_sweep
pub use mutation_sweep::{
    MutationRateResult, MutationSweepConfig, MutationSweepResults, print_mutation_sweep_table,
    run_mutation_sweep,
};

// Re-export from temporal_dynamics
pub use temporal_dynamics::{
    SnapshotStatistics, TemporalConfig, TemporalResult, TemporalSnapshot, TemporalSweepResults,
    TrialSnapshot, print_temporal_results, run_temporal_sweep,
};

// Re-export from continuous_mutations
pub use continuous_mutations::{
    ContinuousMutationPopulation, ContinuousMutationsConfig, ContinuousMutationsResults,
    MutationModeResult, QuantumMutationMode, mutate_circuit_continuous, mutate_circuit_hybrid,
    print_continuous_mutations_results, run_continuous_mutations_experiment,
};

// Re-export from dynamic_environments
pub use dynamic_environments::{
    DynamicEnvironment, DynamicEnvironmentsConfig, DynamicEnvironmentsResults, DynamicsModeResult,
    ResourceDynamics, ResourcePatch, print_dynamic_environments_results,
    run_dynamic_environments_experiment,
};

// Re-export from phase_diagram
pub use phase_diagram::{
    ParameterRange, ParameterSweep1D, PhaseDiagram, PhaseDiagramConfig, PhaseDiagramSummary,
    PhasePoint, SweepParameter, generate_phase_diagram, movement_vs_metabolism_config,
    mutation_vs_time_config, patches_vs_value_config, qubits_vs_length_config, run_parameter_sweep,
    time_vs_scarcity_config,
};

// Re-export from lineage
pub use lineage::{LineageNode, LineageSummary, LineageTracker, OrganismSnapshot};

// Re-export from circuit_archaeology
pub use circuit_archaeology::{
    CircuitComparison, PatternAnalysis, PopulationPatternAnalysis, QuantumPattern,
    analyze_population_patterns, circuit_diagram, circuit_to_string, compare_circuits,
    detect_patterns,
};

// Re-export from noise_model
pub use noise_model::{
    NoiseConfig, NoiseResilience, NoiseRng, apply_measurement_error, apply_noise_to_circuit,
    categorize_resilience, expected_success_probability,
};
