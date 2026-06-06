// =============================================================================
// src/organisms_quantum.rs
// =============================================================================
//
// SPRINT 3.1: Integration between quantum foragers and existing organism system
//
// This module provides:
// 1. OrganismKind::QuantumForager variant addition
// 2. Bridge between ecosystem.rs and quantum_forager.rs
// 3. Resource spawning utilities
//
// =============================================================================

use crate::quantum_forager::{
    ForagerPopulation, ForagerRegion, PopulationStats, QuantumForagerConfig,
};
use crate::simulation::Simulation;

// =============================================================================
// Ecosystem Integration
// =============================================================================

/// Configuration for running quantum foragers within the ecosystem.
#[derive(Clone, Debug)]
pub struct QuantumEcosystemConfig {
    /// Forager population configuration
    pub forager_config: QuantumForagerConfig,

    /// Number of ticks to run
    pub max_ticks: u64,

    /// Resource spawning interval (spawn every N ticks)
    pub resource_spawn_interval: u64,

    /// Heat patches to spawn per interval
    pub heat_patches_per_spawn: usize,

    /// Heat value per patch
    pub heat_per_patch: u32,

    /// Whether to run physics simulation each tick
    pub run_physics: bool,

    /// Logging interval (log stats every N ticks)
    pub log_interval: u64,
}

impl Default for QuantumEcosystemConfig {
    fn default() -> Self {
        Self {
            forager_config: QuantumForagerConfig::default(),
            max_ticks: 1000,
            resource_spawn_interval: 10,
            heat_patches_per_spawn: 20,
            heat_per_patch: 50,
            run_physics: true,
            log_interval: 100,
        }
    }
}

/// Result of running a quantum ecosystem simulation.
#[derive(Debug, Clone)]
pub struct QuantumEcosystemResult {
    /// Total ticks run
    pub ticks_run: u64,

    /// Whether population went extinct
    pub extinct: bool,

    /// Final population size
    pub final_population: usize,

    /// Maximum generation reached
    pub max_generation: u32,

    /// Statistics history (sampled at log_interval)
    pub stats_history: Vec<PopulationStats>,

    /// Final top performers
    pub top_genomes: Vec<GenomeSummary>,
}

/// Summary of a genome for export.
#[derive(Debug, Clone)]
pub struct GenomeSummary {
    pub id: u64,
    pub generation: u32,
    pub fitness: f32,
    pub circuit_length: usize,
    pub entangling_gates: usize,
    pub quantum_ness: f32,
}

/// Run a quantum ecosystem simulation.
///
/// This is the main entry point for running quantum forager evolution.
pub fn run_quantum_ecosystem(
    sim: &mut Simulation,
    config: &QuantumEcosystemConfig,
    seed: u64,
) -> QuantumEcosystemResult {
    let mut pop = ForagerPopulation::new(config.forager_config.clone(), seed);
    let mut stats_history = Vec::new();

    // Initial resource spawning
    pop.spawn_resources(
        sim,
        config.heat_patches_per_spawn * 5,
        config.heat_per_patch,
    );

    for tick in 0..config.max_ticks {
        // Spawn resources periodically
        if tick > 0 && tick % config.resource_spawn_interval == 0 {
            pop.spawn_resources(sim, config.heat_patches_per_spawn, config.heat_per_patch);
        }

        // Run physics if enabled
        if config.run_physics {
            sim.tick();
        }

        // Step population
        pop.step(sim);

        // Log stats periodically
        if tick % config.log_interval == 0 {
            stats_history.push(pop.stats().clone());
        }

        // Check for extinction
        if pop.is_extinct() {
            let max_gen = pop.stats().max_generation;
            return QuantumEcosystemResult {
                ticks_run: tick + 1,
                extinct: true,
                final_population: 0,
                max_generation: max_gen,
                stats_history,
                top_genomes: vec![],
            };
        }
    }

    // Collect final stats
    let final_stats = pop.stats().clone();
    stats_history.push(final_stats.clone());

    // Get top performers
    let top = pop.top_by_fitness(10);
    let top_genomes: Vec<GenomeSummary> = top
        .iter()
        .map(|f| {
            let stats = f.circuit_stats();
            GenomeSummary {
                id: f.id,
                generation: f.generation,
                fitness: f.energy * (f.age as f32 + 1.0),
                circuit_length: f.genome.len(),
                entangling_gates: stats.entangling_gates(),
                quantum_ness: crate::quantum_brain::quantum_ness(&f.genome),
            }
        })
        .collect();

    QuantumEcosystemResult {
        ticks_run: config.max_ticks,
        extinct: false,
        final_population: pop.size(),
        max_generation: final_stats.max_generation,
        stats_history,
        top_genomes,
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a small test ecosystem configuration.
pub fn test_ecosystem_config() -> QuantumEcosystemConfig {
    QuantumEcosystemConfig {
        forager_config: QuantumForagerConfig {
            initial_population: 20,
            initial_energy: 100.0,
            n_qubits: 4,
            initial_circuit_length: 6,
            max_circuit_length: 20,
            movement_cost: 0.5,
            metabolism_cost: 0.2,
            reproduction_threshold: 130.0,
            reproduction_cost: 50.0,
            mutation_rate: 0.9,
            heat_energy_factor: 0.8,
            charge_energy_factor: 0.4,
            consume_resources: true,
            max_population: 200,
            region: ForagerRegion {
                x0: 50,
                y0: 50,
                width: 100,
                height: 100,
            },
        },
        max_ticks: 500,
        resource_spawn_interval: 5,
        heat_patches_per_spawn: 30,
        heat_per_patch: 40,
        run_physics: false, // Skip physics for faster testing
        log_interval: 50,
    }
}

/// Print a summary of ecosystem results.
pub fn print_ecosystem_summary(result: &QuantumEcosystemResult) {
    println!("=== Quantum Ecosystem Results ===");
    println!("Ticks run: {}", result.ticks_run);
    println!("Extinct: {}", result.extinct);
    println!("Final population: {}", result.final_population);
    println!("Max generation: {}", result.max_generation);
    println!();

    if !result.stats_history.is_empty() {
        println!("=== Population History ===");
        for (i, stats) in result.stats_history.iter().enumerate() {
            println!(
                "Sample {}: pop={}, avg_energy={:.1}, avg_age={:.1}, avg_circuit={:.1}, quantum_ness={:.3}",
                i,
                stats.size,
                stats.avg_energy,
                stats.avg_age,
                stats.avg_circuit_length,
                stats.avg_quantum_ness
            );
        }
        println!();
    }

    if !result.top_genomes.is_empty() {
        println!("=== Top Performers ===");
        for (i, genome) in result.top_genomes.iter().enumerate() {
            println!(
                "#{}: id={}, gen={}, fitness={:.1}, gates={}, entangling={}, qness={:.3}",
                i + 1,
                genome.id,
                genome.generation,
                genome.fitness,
                genome.circuit_length,
                genome.entangling_gates,
                genome.quantum_ness
            );
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_ecosystem_runs() {
        let mut sim = Simulation::new();
        let config = test_ecosystem_config();

        let result = run_quantum_ecosystem(&mut sim, &config, 12345);

        // Should run without panicking
        assert!(result.ticks_run > 0);

        // Should have recorded some history
        assert!(!result.stats_history.is_empty());
    }

    #[test]
    fn test_quantum_ecosystem_evolution() {
        let mut sim = Simulation::new();
        let mut config = test_ecosystem_config();
        config.max_ticks = 200;
        config.log_interval = 20;

        let result = run_quantum_ecosystem(&mut sim, &config, 54321);

        if !result.extinct {
            // Check that evolution happened
            assert!(
                result.max_generation > 0,
                "Should have multiple generations"
            );

            // Check that stats were recorded
            assert!(result.stats_history.len() >= 2);
        }
    }

    #[test]
    fn test_ecosystem_with_scarce_resources() {
        let mut sim = Simulation::new();
        let mut config = test_ecosystem_config();
        config.heat_patches_per_spawn = 5; // Scarce resources
        config.heat_per_patch = 20;
        config.max_ticks = 300;

        let result = run_quantum_ecosystem(&mut sim, &config, 11111);

        // With scarce resources, population should struggle
        // Either extinct or small
        assert!(result.extinct || result.final_population < 50);
    }

    #[test]
    fn test_ecosystem_with_abundant_resources() {
        let mut sim = Simulation::new();
        let mut config = test_ecosystem_config();
        config.heat_patches_per_spawn = 100; // Abundant resources
        config.heat_per_patch = 100;
        config.forager_config.max_population = 300;
        config.max_ticks = 200;

        let result = run_quantum_ecosystem(&mut sim, &config, 22222);

        // With abundant resources, population should grow
        if !result.extinct {
            assert!(result.final_population > config.forager_config.initial_population);
        }
    }

    #[test]
    fn test_genome_summary() {
        let mut sim = Simulation::new();
        let config = test_ecosystem_config();

        let result = run_quantum_ecosystem(&mut sim, &config, 33333);

        if !result.extinct && !result.top_genomes.is_empty() {
            let top = &result.top_genomes[0];
            assert!(top.fitness > 0.0);
            assert!(top.circuit_length > 0);
            assert!(top.quantum_ness >= 0.0 && top.quantum_ness <= 1.0);
        }
    }
}
