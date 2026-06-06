// Organisms module - artificial life, foragers, and ecosystem simulation

pub mod base;
pub mod ecosystem;
pub mod forager;
pub mod quantum;

// Re-export from base
pub use base::{
    OrganismError, OrganismKind, OrganismParams, OrganismResult, OrganismTemplate, Region,
    ZoneRole, ZoneSpec, list_organisms, place_organism, run_organism,
};

// Re-export from quantum ecosystem
pub use quantum::{
    GenomeSummary, QuantumEcosystemConfig, QuantumEcosystemResult, print_ecosystem_summary,
    run_quantum_ecosystem, test_ecosystem_config,
};

// Re-export from forager
pub use forager::{
    BatchStats, ForagerPopulation, ForagerRegion, ForagerTickResult, MIN_BATCH_FOR_GPU,
    MIN_GATES_FOR_GPU, OrganismBatch, PopulationStats, QuantumForager, QuantumForagerConfig,
    TickMetrics, group_organisms_by_circuit,
};

// Re-export from ecosystem
pub use ecosystem::{
    Axis, ClaimMode, EcosystemConfig, EcosystemResult, InteractionRules, OrganismInstance,
    ResourceBudgets, SpeciesConfig, SpeciesPolicy, Thresholds, run_ecosystem,
};
// Note: ecosystem::Region is shadowed by base::Region - use ecosystem::Region explicitly if needed
