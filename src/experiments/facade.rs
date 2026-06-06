// =============================================================================
// src/experiments.rs
// =============================================================================
//
// Unified experiment framework facade.
//
// This module re-exports two experiment domains:
//
// 1. Physics experiments (physics_experiments.rs)
//    - Parameter sweeps for heat, charge, diffusion
//    - Scenario-based physics simulations
//
// 2. Evolution experiments (evolution_experiments.rs)
//    - Quantum forager evolution studies
//    - Quantum vs classical comparisons
//    - Population dynamics analysis
//
// Future: Cross-domain experiments combining physics and evolution.
//
// =============================================================================

// Re-export physics experiments
pub use crate::physics_experiments::{
    ExperimentKind, ExperimentParam, ExperimentResult as PhysicsExperimentResult, Range1D,
    RunResult as PhysicsRunResult, run_experiment as run_physics_experiment,
};

// Re-export evolution experiments
pub use crate::evolution_experiments::{
    AggregatedStats, ComparisonRow, ComparisonSummary, ExperimentConfig, ExperimentResult,
    ExperimentSuite, RunResult as EvolutionRunResult, export_csv, export_timeseries_csv,
    mutation_rate_suite, quantum_advantage_suite, resource_scarcity_suite,
    run_experiment as run_evolution_experiment,
};

// Re-export quantum vs classical (THE experiment)
pub use crate::quantum_vs_classical::{
    ComparisonConfig, ComparisonResult, ComparisonSummary as QvCComparisonSummary,
    ConstrainedForager, ConstrainedPopulation, print_comparison, run_quantum_vs_classical,
};

// Re-export circuit constraints
pub use crate::constrained_circuits::{
    CircuitConstraints, EntanglementLevel, categorize_entanglement, count_entangling_gates,
    entanglement_fraction, is_entangling_gate, mutate_circuit_constrained,
    random_circuit_constrained,
};

// =============================================================================
// Unified Experiment Types (for future cross-domain work)
// =============================================================================

/// Unified experiment kind covering both domains.
#[derive(Clone, Debug)]
pub enum UnifiedExperiment {
    /// Physics parameter sweep
    Physics(ExperimentKind),

    /// Evolution experiment
    Evolution(ExperimentConfig),

    /// Quantum vs classical comparison
    QuantumVsClassical(ComparisonConfig),
}

/// Run any unified experiment.
pub fn run_unified(experiment: &UnifiedExperiment) -> UnifiedResult {
    match experiment {
        UnifiedExperiment::Physics(kind) => {
            let mut sim = crate::simulation::Simulation::new();
            let result = crate::physics_experiments::run_experiment(&mut sim, kind);
            UnifiedResult::Physics(result)
        }
        UnifiedExperiment::Evolution(config) => {
            let result = crate::evolution_experiments::run_experiment(config);
            UnifiedResult::Evolution(result)
        }
        UnifiedExperiment::QuantumVsClassical(config) => {
            let result = crate::quantum_vs_classical::run_quantum_vs_classical(config);
            UnifiedResult::QuantumVsClassical(result)
        }
    }
}

/// Result from any unified experiment.
#[derive(Clone, Debug)]
pub enum UnifiedResult {
    Physics(PhysicsExperimentResult),
    Evolution(ExperimentResult),
    QuantumVsClassical(ComparisonResult),
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_reexport() {
        // Just verify the re-exports work
        let _range = Range1D {
            start: 0,
            end: 10,
            steps: 3,
        };
        let _param = ExperimentParam::MaxHeat;
    }

    #[test]
    fn test_evolution_reexport() {
        let _constraints = CircuitConstraints::full_quantum();
        assert!(_constraints.allows_entanglement());
    }
}
