// =============================================================================
// src/experiments.rs
// =============================================================================
//
// SPRINT 3.2a: Experiment Runner & Analysis Framework
//
// This module provides infrastructure for running controlled experiments
// on quantum forager evolution. It enables:
//
// - Running multiple replications of the same configuration
// - Comparing different experimental conditions
// - Statistical aggregation across runs
// - Export for external analysis
//
// The deeper purpose: This is our telescope into a computational universe.
// We're not just measuring performance — we're observing what evolution
// discovers when physics, computation, and survival intertwine.
//
// =============================================================================

use crate::organisms_quantum::{
    GenomeSummary, QuantumEcosystemConfig, QuantumEcosystemResult, run_quantum_ecosystem,
};
use crate::quantum_forager::{ForagerRegion, QuantumForagerConfig};
// quantum_ness, analyze_circuit, CircuitStats available if needed for future analysis
use crate::simulation::Simulation;

// =============================================================================
// Experiment Configuration
// =============================================================================

/// A single experiment configuration.
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    /// Human-readable name for this experiment
    pub name: String,

    /// Description of what this experiment tests
    pub description: String,

    /// The ecosystem configuration to run
    pub ecosystem_config: QuantumEcosystemConfig,

    /// Base seed (each replication adds its index)
    pub base_seed: u64,

    /// Number of times to run this configuration
    pub replications: usize,
}

impl ExperimentConfig {
    /// Create a new experiment with default settings.
    pub fn new(name: &str, ecosystem_config: QuantumEcosystemConfig) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            ecosystem_config,
            base_seed: 42,
            replications: 5,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set the number of replications.
    pub fn with_replications(mut self, n: usize) -> Self {
        self.replications = n;
        self
    }

    /// Set the base seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.base_seed = seed;
        self
    }
}

// =============================================================================
// Experiment Results
// =============================================================================

/// Results from a single run of an experiment.
#[derive(Clone, Debug)]
pub struct RunResult {
    /// Which replication this was (0-indexed)
    pub replication: usize,

    /// Seed used for this run
    pub seed: u64,

    /// The raw ecosystem result
    pub ecosystem_result: QuantumEcosystemResult,

    /// Final population quantum-ness
    pub final_quantum_ness: f32,

    /// Peak population reached
    pub peak_population: usize,

    /// Average survival time (ticks alive)
    pub avg_survival_time: f32,
}

/// Aggregated statistics across multiple runs.
#[derive(Clone, Debug, Default)]
pub struct AggregatedStats {
    /// Number of runs that went extinct
    pub extinction_count: usize,

    /// Mean final population (excluding extinctions)
    pub mean_final_population: f32,
    pub std_final_population: f32,

    /// Mean max generation reached
    pub mean_max_generation: f32,
    pub std_max_generation: f32,

    /// Mean final quantum-ness
    pub mean_quantum_ness: f32,
    pub std_quantum_ness: f32,

    /// Mean peak population
    pub mean_peak_population: f32,

    /// Quantum-ness trend (final - initial, averaged)
    pub quantum_ness_trend: f32,

    /// Total ticks simulated across all runs
    pub total_ticks: u64,
}

/// Complete results from an experiment (all replications).
#[derive(Clone, Debug)]
pub struct ExperimentResult {
    /// The experiment configuration
    pub config: ExperimentConfig,

    /// Results from each replication
    pub runs: Vec<RunResult>,

    /// Aggregated statistics
    pub stats: AggregatedStats,

    /// Top genomes across all runs (by fitness)
    pub top_genomes: Vec<GenomeSummary>,
}

// =============================================================================
// Experiment Suite
// =============================================================================

/// A collection of experiments to run and compare.
#[derive(Clone, Debug)]
pub struct ExperimentSuite {
    /// Name of the suite
    pub name: String,

    /// Individual experiments
    pub experiments: Vec<ExperimentConfig>,

    /// Results after running
    pub results: Vec<ExperimentResult>,
}

impl ExperimentSuite {
    /// Create a new empty suite.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            experiments: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add an experiment to the suite.
    pub fn add_experiment(&mut self, config: ExperimentConfig) {
        self.experiments.push(config);
    }

    /// Run all experiments in the suite.
    pub fn run(&mut self) {
        self.results.clear();

        for config in &self.experiments {
            let result = run_experiment(config);
            self.results.push(result);
        }
    }

    /// Get a comparison summary between experiments.
    pub fn comparison_summary(&self) -> ComparisonSummary {
        ComparisonSummary::from_results(&self.results)
    }
}

// =============================================================================
// Core Experiment Runner
// =============================================================================

/// Run a single experiment with all its replications.
pub fn run_experiment(config: &ExperimentConfig) -> ExperimentResult {
    let mut runs = Vec::with_capacity(config.replications);
    let mut all_genomes: Vec<GenomeSummary> = Vec::new();

    for rep in 0..config.replications {
        let seed = config.base_seed.wrapping_add(rep as u64 * 10007);
        let mut sim = Simulation::new();

        let ecosystem_result = run_quantum_ecosystem(&mut sim, &config.ecosystem_config, seed);

        // Calculate additional metrics
        let final_quantum_ness = ecosystem_result
            .stats_history
            .last()
            .map(|s| s.avg_quantum_ness)
            .unwrap_or(0.0);

        let peak_population = ecosystem_result
            .stats_history
            .iter()
            .map(|s| s.size)
            .max()
            .unwrap_or(0);

        // Estimate average survival time from stats
        let avg_survival = estimate_avg_survival(&ecosystem_result);

        // Collect top genomes
        all_genomes.extend(ecosystem_result.top_genomes.clone());

        runs.push(RunResult {
            replication: rep,
            seed,
            ecosystem_result,
            final_quantum_ness,
            peak_population,
            avg_survival_time: avg_survival,
        });
    }

    // Aggregate statistics
    let stats = aggregate_stats(&runs);

    // Sort and take top genomes across all runs
    all_genomes.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_genomes.truncate(10);

    ExperimentResult {
        config: config.clone(),
        runs,
        stats,
        top_genomes: all_genomes,
    }
}

/// Estimate average survival time from population statistics.
fn estimate_avg_survival(result: &QuantumEcosystemResult) -> f32 {
    if result.stats_history.is_empty() {
        return 0.0;
    }

    // Use average age from final stats as proxy
    result
        .stats_history
        .last()
        .map(|s| s.avg_age)
        .unwrap_or(0.0)
}

/// Aggregate statistics across multiple runs.
fn aggregate_stats(runs: &[RunResult]) -> AggregatedStats {
    if runs.is_empty() {
        return AggregatedStats::default();
    }

    let n = runs.len() as f32;

    // Count extinctions
    let extinction_count = runs.iter().filter(|r| r.ecosystem_result.extinct).count();

    // Final populations (non-extinct only)
    let final_pops: Vec<f32> = runs
        .iter()
        .filter(|r| !r.ecosystem_result.extinct)
        .map(|r| r.ecosystem_result.final_population as f32)
        .collect();

    let (mean_final_population, std_final_population) = mean_std(&final_pops);

    // Max generations
    let max_gens: Vec<f32> = runs
        .iter()
        .map(|r| r.ecosystem_result.max_generation as f32)
        .collect();

    let (mean_max_generation, std_max_generation) = mean_std(&max_gens);

    // Quantum-ness
    let quantum_ness: Vec<f32> = runs.iter().map(|r| r.final_quantum_ness).collect();

    let (mean_quantum_ness, std_quantum_ness) = mean_std(&quantum_ness);

    // Peak populations
    let peak_pops: Vec<f32> = runs.iter().map(|r| r.peak_population as f32).collect();

    let mean_peak_population = peak_pops.iter().sum::<f32>() / n;

    // Quantum-ness trend
    let trends: Vec<f32> = runs
        .iter()
        .filter_map(|r| {
            let history = &r.ecosystem_result.stats_history;
            if history.len() >= 2 {
                let first = history.first()?.avg_quantum_ness;
                let last = history.last()?.avg_quantum_ness;
                Some(last - first)
            } else {
                None
            }
        })
        .collect();

    let quantum_ness_trend = if trends.is_empty() {
        0.0
    } else {
        trends.iter().sum::<f32>() / trends.len() as f32
    };

    // Total ticks
    let total_ticks: u64 = runs.iter().map(|r| r.ecosystem_result.ticks_run).sum();

    AggregatedStats {
        extinction_count,
        mean_final_population,
        std_final_population,
        mean_max_generation,
        std_max_generation,
        mean_quantum_ness,
        std_quantum_ness,
        mean_peak_population,
        quantum_ness_trend,
        total_ticks,
    }
}

/// Calculate mean and standard deviation.
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

    (mean, variance.sqrt())
}

// =============================================================================
// Comparison Analysis
// =============================================================================

/// Summary comparing multiple experiments.
#[derive(Clone, Debug)]
pub struct ComparisonSummary {
    /// Names of experiments being compared
    pub experiment_names: Vec<String>,

    /// Which experiment had highest mean population
    pub best_population: Option<String>,

    /// Which experiment had highest max generation
    pub best_evolution: Option<String>,

    /// Which experiment had highest quantum-ness
    pub most_quantum: Option<String>,

    /// Which experiment had lowest extinction rate
    pub most_stable: Option<String>,

    /// Raw comparison data
    pub comparison_data: Vec<ComparisonRow>,
}

/// One row of comparison data.
#[derive(Clone, Debug)]
pub struct ComparisonRow {
    pub name: String,
    pub mean_population: f32,
    pub extinction_rate: f32,
    pub mean_generation: f32,
    pub mean_quantum_ness: f32,
    pub quantum_trend: f32,
}

impl ComparisonSummary {
    /// Create comparison from experiment results.
    pub fn from_results(results: &[ExperimentResult]) -> Self {
        let rows: Vec<ComparisonRow> = results
            .iter()
            .map(|r| {
                let extinction_rate = r.stats.extinction_count as f32 / r.runs.len() as f32;
                ComparisonRow {
                    name: r.config.name.clone(),
                    mean_population: r.stats.mean_final_population,
                    extinction_rate,
                    mean_generation: r.stats.mean_max_generation,
                    mean_quantum_ness: r.stats.mean_quantum_ness,
                    quantum_trend: r.stats.quantum_ness_trend,
                }
            })
            .collect();

        let experiment_names: Vec<String> = rows.iter().map(|r| r.name.clone()).collect();

        // Find best in each category
        let best_population = rows
            .iter()
            .max_by(|a, b| a.mean_population.partial_cmp(&b.mean_population).unwrap())
            .map(|r| r.name.clone());

        let best_evolution = rows
            .iter()
            .max_by(|a, b| a.mean_generation.partial_cmp(&b.mean_generation).unwrap())
            .map(|r| r.name.clone());

        let most_quantum = rows
            .iter()
            .max_by(|a, b| {
                a.mean_quantum_ness
                    .partial_cmp(&b.mean_quantum_ness)
                    .unwrap()
            })
            .map(|r| r.name.clone());

        let most_stable = rows
            .iter()
            .min_by(|a, b| a.extinction_rate.partial_cmp(&b.extinction_rate).unwrap())
            .map(|r| r.name.clone());

        ComparisonSummary {
            experiment_names,
            best_population,
            best_evolution,
            most_quantum,
            most_stable,
            comparison_data: rows,
        }
    }

    /// Print a formatted comparison table.
    pub fn print_table(&self) {
        println!("\n{}", "=".repeat(80));
        println!("  Experiment Comparison");
        println!("{}\n", "=".repeat(80));

        println!(
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Experiment", "Population", "Extinct%", "MaxGen", "Quantum", "Q-Trend"
        );
        println!("{:-<80}", "");

        for row in &self.comparison_data {
            println!(
                "{:<20} {:>10.1} {:>9.0}% {:>10.1} {:>10.3} {:>+10.3}",
                truncate(&row.name, 20),
                row.mean_population,
                row.extinction_rate * 100.0,
                row.mean_generation,
                row.mean_quantum_ness,
                row.quantum_trend
            );
        }

        println!("\nWinners:");
        if let Some(ref name) = self.best_population {
            println!("  Highest Population: {}", name);
        }
        if let Some(ref name) = self.best_evolution {
            println!("  Deepest Evolution:  {}", name);
        }
        if let Some(ref name) = self.most_quantum {
            println!("  Most Quantum:       {}", name);
        }
        if let Some(ref name) = self.most_stable {
            println!("  Most Stable:        {}", name);
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// Pre-built Experiment Templates
// =============================================================================

/// Create the "Quantum Advantage" experiment suite.
///
/// This is THE experiment: Does evolution discover quantum advantage?
pub fn quantum_advantage_suite() -> ExperimentSuite {
    let mut suite = ExperimentSuite::new("Quantum Advantage Comparison");

    // Base configuration
    let base_config = QuantumEcosystemConfig {
        forager_config: QuantumForagerConfig {
            initial_population: 40,
            initial_energy: 100.0,
            n_qubits: 4,
            initial_circuit_length: 6,
            max_circuit_length: 25,
            movement_cost: 0.5,
            metabolism_cost: 0.3,
            reproduction_threshold: 140.0,
            reproduction_cost: 55.0,
            mutation_rate: 0.85,
            heat_energy_factor: 0.6,
            charge_energy_factor: 0.3,
            consume_resources: true,
            max_population: 200,
            region: ForagerRegion {
                x0: 30,
                y0: 30,
                width: 100,
                height: 100,
            },
        },
        max_ticks: 500,
        resource_spawn_interval: 8,
        heat_patches_per_spawn: 25,
        heat_per_patch: 45,
        run_physics: false,
        log_interval: 50,
    };

    // Experiment 1: Full quantum (all gates allowed)
    suite.add_experiment(
        ExperimentConfig::new("Full Quantum", base_config.clone())
            .with_description("All quantum gates allowed, including entangling gates")
            .with_replications(10),
    );

    // For classical comparison, we'd need to modify the circuit generation
    // This is a placeholder - Sprint 3.2d will implement constrained circuits

    suite
}

/// Create a resource scarcity comparison suite.
pub fn resource_scarcity_suite() -> ExperimentSuite {
    let mut suite = ExperimentSuite::new("Resource Scarcity Effects");

    let base_config = QuantumEcosystemConfig::default();

    // Scarce resources
    let mut scarce = base_config.clone();
    scarce.heat_patches_per_spawn = 10;
    scarce.heat_per_patch = 25;
    suite.add_experiment(
        ExperimentConfig::new("Scarce", scarce)
            .with_description("Low resource availability")
            .with_replications(5),
    );

    // Moderate resources
    let mut moderate = base_config.clone();
    moderate.heat_patches_per_spawn = 25;
    moderate.heat_per_patch = 45;
    suite.add_experiment(
        ExperimentConfig::new("Moderate", moderate)
            .with_description("Medium resource availability")
            .with_replications(5),
    );

    // Abundant resources
    let mut abundant = base_config.clone();
    abundant.heat_patches_per_spawn = 50;
    abundant.heat_per_patch = 80;
    suite.add_experiment(
        ExperimentConfig::new("Abundant", abundant)
            .with_description("High resource availability")
            .with_replications(5),
    );

    suite
}

/// Create a mutation rate comparison suite.
pub fn mutation_rate_suite() -> ExperimentSuite {
    let mut suite = ExperimentSuite::new("Mutation Rate Effects");

    let base_config = QuantumEcosystemConfig::default();

    for rate in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let mut config = base_config.clone();
        config.forager_config.mutation_rate = rate;

        suite.add_experiment(
            ExperimentConfig::new(&format!("Mutation {:.0}%", rate * 100.0), config)
                .with_description(&format!("Mutation rate = {}", rate))
                .with_replications(5),
        );
    }

    suite
}

// =============================================================================
// Export Functions
// =============================================================================

/// Export experiment results to CSV format.
pub fn export_csv(results: &[ExperimentResult]) -> String {
    let mut csv = String::new();

    // Header
    csv.push_str(
        "experiment,replication,seed,extinct,final_pop,max_gen,quantum_ness,peak_pop,ticks\n",
    );

    // Data rows
    for result in results {
        for run in &result.runs {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{:.4},{},{}\n",
                result.config.name,
                run.replication,
                run.seed,
                run.ecosystem_result.extinct,
                run.ecosystem_result.final_population,
                run.ecosystem_result.max_generation,
                run.final_quantum_ness,
                run.peak_population,
                run.ecosystem_result.ticks_run,
            ));
        }
    }

    csv
}

/// Export time series data (quantum-ness over time) to CSV.
pub fn export_timeseries_csv(results: &[ExperimentResult]) -> String {
    let mut csv = String::new();

    // Header
    csv.push_str(
        "experiment,replication,sample,tick,population,avg_energy,quantum_ness,avg_circuit_len\n",
    );

    // Data rows
    for result in results {
        for run in &result.runs {
            let log_interval = result.config.ecosystem_config.log_interval;
            for (i, stats) in run.ecosystem_result.stats_history.iter().enumerate() {
                let tick = i as u64 * log_interval;
                csv.push_str(&format!(
                    "{},{},{},{},{},{:.2},{:.4},{:.2}\n",
                    result.config.name,
                    run.replication,
                    i,
                    tick,
                    stats.size,
                    stats.avg_energy,
                    stats.avg_quantum_ness,
                    stats.avg_circuit_length,
                ));
            }
        }
    }

    csv
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_config() -> QuantumEcosystemConfig {
        QuantumEcosystemConfig {
            forager_config: QuantumForagerConfig {
                initial_population: 10,
                max_population: 50,
                ..Default::default()
            },
            max_ticks: 100,
            log_interval: 20,
            ..Default::default()
        }
    }

    #[test]
    fn test_run_experiment() {
        let config = ExperimentConfig::new("Test", quick_config()).with_replications(2);

        let result = run_experiment(&config);

        assert_eq!(result.runs.len(), 2);
        assert_eq!(result.config.name, "Test");
    }

    #[test]
    fn test_aggregated_stats() {
        let config = ExperimentConfig::new("Test", quick_config()).with_replications(3);

        let result = run_experiment(&config);

        // Stats should be populated
        assert!(result.stats.total_ticks > 0);
    }

    #[test]
    fn test_experiment_suite() {
        let mut suite = ExperimentSuite::new("Test Suite");

        suite.add_experiment(ExperimentConfig::new("Exp1", quick_config()).with_replications(2));
        suite.add_experiment(ExperimentConfig::new("Exp2", quick_config()).with_replications(2));

        suite.run();

        assert_eq!(suite.results.len(), 2);
    }

    #[test]
    fn test_comparison_summary() {
        let mut suite = ExperimentSuite::new("Comparison Test");

        let mut config1 = quick_config();
        config1.heat_patches_per_spawn = 50; // More resources

        let mut config2 = quick_config();
        config2.heat_patches_per_spawn = 10; // Fewer resources

        suite.add_experiment(ExperimentConfig::new("Abundant", config1).with_replications(2));
        suite.add_experiment(ExperimentConfig::new("Scarce", config2).with_replications(2));

        suite.run();

        let comparison = suite.comparison_summary();
        assert_eq!(comparison.experiment_names.len(), 2);
        assert_eq!(comparison.comparison_data.len(), 2);
    }

    #[test]
    fn test_mean_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&values);

        assert!((mean - 5.0).abs() < 0.01);
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_csv_export() {
        let config = ExperimentConfig::new("Test", quick_config()).with_replications(2);

        let result = run_experiment(&config);
        let csv = export_csv(&[result]);

        assert!(csv.contains("experiment,replication"));
        assert!(csv.contains("Test"));
    }

    #[test]
    fn test_timeseries_export() {
        let config = ExperimentConfig::new("Test", quick_config()).with_replications(1);

        let result = run_experiment(&config);
        let csv = export_timeseries_csv(&[result]);

        assert!(csv.contains("quantum_ness"));
    }

    #[test]
    fn test_resource_scarcity_suite() {
        let suite = resource_scarcity_suite();
        assert_eq!(suite.experiments.len(), 3);
    }

    #[test]
    fn test_mutation_rate_suite() {
        let suite = mutation_rate_suite();
        assert_eq!(suite.experiments.len(), 5);
    }
}
