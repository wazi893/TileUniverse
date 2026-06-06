// =============================================================================
// src/quantum_vs_classical.rs
// =============================================================================
//
// SPRINT 3.2: THE EXPERIMENT
//
// "Does evolution discover quantum advantage?"
//
// This module runs the critical comparison:
// - Group A: Full quantum organisms (all gates allowed)
// - Group B: Classical-only organisms (no entangling gates)
//
// Same environment, same selection pressure, same everything —
// except one group can entangle qubits and the other cannot.
//
// If Group A outperforms Group B, evolution has discovered that
// quantum correlations help in this universe.
//
// If they perform equally, quantum resources aren't beneficial here.
//
// Either result is meaningful. We're not hoping for a specific outcome;
// we're asking the universe a question and listening for the answer.
//
// =============================================================================

use crate::constrained_circuits::{
    CircuitConstraints, entanglement_fraction, mutate_circuit_constrained,
    random_circuit_constrained,
};
use crate::quantum_brain::{decode_action, encode_as_rotations, run_circuit, sense_local_fields};
use crate::quantum_forager::{
    ForagerRegion, PopulationStats, QuantumForager, QuantumForagerConfig,
};
use crate::simulation::Simulation;

use std::collections::HashMap;

// =============================================================================
// Constrained Forager (respects circuit constraints)
// =============================================================================

/// A forager with circuit generation constraints.
#[derive(Clone, Debug)]
pub struct ConstrainedForager {
    /// The underlying forager
    pub forager: QuantumForager,

    /// Circuit constraints for this lineage
    pub constraints: CircuitConstraints,
}

impl ConstrainedForager {
    /// Create a new constrained forager with random genome.
    pub fn new_random(
        id: u64,
        x: u32,
        y: u32,
        config: &QuantumForagerConfig,
        constraints: CircuitConstraints,
        seed: u64,
    ) -> Self {
        let genome = random_circuit_constrained(
            config.n_qubits,
            config.initial_circuit_length,
            &constraints,
            seed,
        );

        Self {
            forager: QuantumForager {
                id,
                generation: 0,
                parent_id: None,
                x,
                y,
                energy: config.initial_energy,
                genome,
                n_qubits: config.n_qubits,
                age: 0,
                total_consumed: 0.0,
                offspring_count: 0,
                decision_seed: seed.wrapping_mul(0x517cc1b727220a95),
            },
            constraints,
        }
    }

    /// Reproduce with constrained mutation.
    pub fn reproduce(
        &mut self,
        offspring_id: u64,
        config: &QuantumForagerConfig,
        mutation_seed: u64,
    ) -> Option<ConstrainedForager> {
        if self.forager.energy < config.reproduction_threshold {
            return None;
        }

        self.forager.energy -= config.reproduction_cost;
        self.forager.offspring_count += 1;

        // Mutate with constraints
        let offspring_genome = mutate_circuit_constrained(
            &self.forager.genome,
            self.forager.n_qubits,
            &self.constraints,
            mutation_seed,
        );

        Some(ConstrainedForager {
            forager: QuantumForager {
                id: offspring_id,
                generation: self.forager.generation + 1,
                parent_id: Some(self.forager.id),
                x: self.forager.x,
                y: self.forager.y,
                energy: config.reproduction_cost,
                genome: offspring_genome,
                n_qubits: self.forager.n_qubits,
                age: 0,
                total_consumed: 0.0,
                offspring_count: 0,
                decision_seed: mutation_seed.wrapping_mul(0x9e3779b97f4a7c15),
            },
            constraints: self.constraints.clone(),
        })
    }
}

// =============================================================================
// Constrained Population
// =============================================================================

/// Population with circuit constraints.
pub struct ConstrainedPopulation {
    foragers: HashMap<u64, ConstrainedForager>,
    next_id: u64,
    config: QuantumForagerConfig,
    constraints: CircuitConstraints,
    tick: u64,
    seed: u64,
    last_stats: PopulationStats,
}

impl ConstrainedPopulation {
    /// Create a new constrained population.
    pub fn new(config: QuantumForagerConfig, constraints: CircuitConstraints, seed: u64) -> Self {
        let mut pop = Self {
            foragers: HashMap::new(),
            next_id: 0,
            config,
            constraints,
            tick: 0,
            seed,
            last_stats: PopulationStats::default(),
        };

        pop.spawn_initial();
        pop
    }

    fn spawn_initial(&mut self) {
        let mut rng = crate::quantum_brain::SimpleRng::new(self.seed);

        for _ in 0..self.config.initial_population {
            let x = self.config.region.x0 + rng.gen_range(0, self.config.region.width);
            let y = self.config.region.y0 + rng.gen_range(0, self.config.region.height);
            let id = self.next_id;
            self.next_id += 1;

            let forager = ConstrainedForager::new_random(
                id,
                x,
                y,
                &self.config,
                self.constraints.clone(),
                rng.next_u64(),
            );

            self.foragers.insert(id, forager);
        }
    }

    pub fn size(&self) -> usize {
        self.foragers.len()
    }

    pub fn is_extinct(&self) -> bool {
        self.foragers.is_empty()
    }

    pub fn stats(&self) -> &PopulationStats {
        &self.last_stats
    }
    /// Extract all circuits from living organisms.
    pub fn extract_all_circuits(&self) -> Vec<Vec<crate::quantum::QGate>> {
        self.foragers
            .values()
            .map(|cf| cf.forager.genome.clone())
            .collect()
    }

    /// Step the population.
    pub fn step(&mut self, sim: &mut Simulation) -> &PopulationStats {
        self.tick += 1;

        let mut deaths = Vec::new();
        let mut births = Vec::new();
        let mut total_consumed = 0.0f32;

        // Cache values needed for reproduction check
        let max_population = self.config.max_population;
        let current_population = self.foragers.len();

        let ids: Vec<u64> = self.foragers.keys().copied().collect();

        for id in ids {
            let cf = self.foragers.get_mut(&id).unwrap();
            let forager = &mut cf.forager;

            // Age and metabolism
            forager.age += 1;
            forager.energy -= self.config.metabolism_cost;

            if forager.energy <= 0.0 {
                deaths.push(id);
                continue;
            }

            // Sense
            let readings = sense_local_fields(sim, forager.x, forager.y);

            // Decide
            let sensor_gates = encode_as_rotations(&readings, forager.n_qubits);
            let measurement = run_circuit(
                &forager.genome,
                &sensor_gates,
                forager.n_qubits,
                forager.decision_seed.wrapping_add(forager.age as u64),
            );
            let action = decode_action(measurement);

            // Move
            let new_x = (forager.x as i32 + action.dx).max(0) as u32;
            let new_y = (forager.y as i32 + action.dy).max(0) as u32;
            let (cx, cy) = self.config.region.clamp(new_x as i32, new_y as i32);

            if cx != forager.x || cy != forager.y {
                forager.x = cx;
                forager.y = cy;
                forager.energy -= self.config.movement_cost;
            }

            // Consume
            if self.config.consume_resources {
                let heat = sim.get_heat_field_for_test(forager.x as usize, forager.y as usize);
                let charge = sim.get_charge_field_for_test(forager.x as usize, forager.y as usize);

                let energy_gain = heat as f32 * self.config.heat_energy_factor
                    + charge as f32 * self.config.charge_energy_factor;

                forager.energy += energy_gain;
                forager.total_consumed += energy_gain;
                total_consumed += energy_gain;

                // Deplete
                if heat > 0 {
                    let new_heat =
                        heat.saturating_sub((energy_gain / self.config.heat_energy_factor) as u32);
                    sim.set_heat_field_for_test(forager.x as usize, forager.y as usize, new_heat);
                }
            }

            // Reproduce
            if forager.energy >= self.config.reproduction_threshold
                && current_population + births.len() < max_population
            {
                let offspring_id = self.next_id;
                self.next_id += 1;

                let mutation_seed = self.seed.wrapping_add(self.tick).wrapping_mul(id);

                if let Some(offspring) = cf.reproduce(offspring_id, &self.config, mutation_seed) {
                    births.push(offspring);
                }
            }
        }

        // Remove dead
        for id in &deaths {
            self.foragers.remove(id);
        }

        // Add births
        for offspring in births {
            self.foragers.insert(offspring.forager.id, offspring);
        }

        // Update stats
        self.update_stats(deaths.len(), total_consumed);

        &self.last_stats
    }

    fn update_stats(&mut self, deaths: usize, consumed: f32) {
        let size = self.foragers.len();

        if size == 0 {
            self.last_stats = PopulationStats::default();
            return;
        }

        let total_energy: f32 = self.foragers.values().map(|cf| cf.forager.energy).sum();
        let total_age: u32 = self.foragers.values().map(|cf| cf.forager.age).sum();
        let total_len: usize = self
            .foragers
            .values()
            .map(|cf| cf.forager.genome.len())
            .sum();
        let total_qness: f32 = self
            .foragers
            .values()
            .map(|cf| entanglement_fraction(&cf.forager.genome))
            .sum();
        let max_gen = self
            .foragers
            .values()
            .map(|cf| cf.forager.generation)
            .max()
            .unwrap_or(0);

        self.last_stats = PopulationStats {
            size,
            total_energy,
            avg_energy: total_energy / size as f32,
            avg_age: total_age as f32 / size as f32,
            avg_circuit_length: total_len as f32 / size as f32,
            avg_quantum_ness: total_qness / size as f32,
            max_generation: max_gen,
            births_this_tick: 0, // Could track
            deaths_this_tick: deaths,
            total_consumed_this_tick: consumed,
        };
    }

    /// Spawn resources.
    pub fn spawn_resources(&self, sim: &mut Simulation, patches: usize, value: u32) {
        let mut rng = crate::quantum_brain::SimpleRng::new(self.seed.wrapping_add(self.tick * 997));

        for _ in 0..patches {
            let x = self.config.region.x0 + rng.gen_range(0, self.config.region.width);
            let y = self.config.region.y0 + rng.gen_range(0, self.config.region.height);

            let current = sim.get_heat_field_for_test(x as usize, y as usize);
            sim.set_heat_field_for_test(x as usize, y as usize, current.saturating_add(value));
        }
    }
}

// =============================================================================
// THE EXPERIMENT
// =============================================================================

/// Configuration for quantum vs classical comparison.
#[derive(Clone, Debug)]
pub struct ComparisonConfig {
    /// Base forager configuration (same for both groups)
    pub forager_config: QuantumForagerConfig,

    /// Number of ticks to run
    pub max_ticks: u64,

    /// Resource spawn interval
    pub resource_interval: u64,

    /// Heat patches per spawn
    pub heat_patches: usize,

    /// Heat value per patch
    pub heat_value: u32,

    /// Logging interval
    pub log_interval: u64,

    /// Number of replications per group
    pub replications: usize,

    /// Base seed
    pub seed: u64,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
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
            resource_interval: 8,
            heat_patches: 25,
            heat_value: 45,
            log_interval: 50,
            replications: 5,
            seed: 42,
        }
    }
}

/// Results from one run.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub extinct: bool,
    pub final_population: usize,
    pub max_generation: u32,
    pub final_quantum_ness: f32,
    pub stats_history: Vec<PopulationStats>,
}

/// Results from the comparison experiment.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    /// Results from quantum group
    pub quantum_runs: Vec<RunResult>,

    /// Results from classical group
    pub classical_runs: Vec<RunResult>,

    /// Summary statistics
    pub summary: ComparisonSummary,
}

/// Summary of comparison.
#[derive(Clone, Debug, Default)]
pub struct ComparisonSummary {
    // Quantum group
    pub quantum_mean_population: f32,
    pub quantum_extinction_rate: f32,
    pub quantum_mean_generation: f32,
    pub quantum_mean_qness: f32,

    // Classical group
    pub classical_mean_population: f32,
    pub classical_extinction_rate: f32,
    pub classical_mean_generation: f32,

    // Comparison
    pub population_advantage: f32, // quantum - classical
    pub generation_advantage: f32,
    pub winner: String,
}

/// Run THE experiment.
pub fn run_quantum_vs_classical(config: &ComparisonConfig) -> ComparisonResult {
    let mut quantum_runs = Vec::new();
    let mut classical_runs = Vec::new();

    // Run quantum group
    for rep in 0..config.replications {
        let seed = config.seed.wrapping_add(rep as u64 * 10007);
        let result = run_single(config, CircuitConstraints::full_quantum(), seed);
        quantum_runs.push(result);
    }

    // Run classical group (same seeds for fair comparison)
    for rep in 0..config.replications {
        let seed = config.seed.wrapping_add(rep as u64 * 10007);
        let result = run_single(config, CircuitConstraints::classical_only(), seed);
        classical_runs.push(result);
    }

    // Compute summary
    let summary = compute_summary(&quantum_runs, &classical_runs);

    ComparisonResult {
        quantum_runs,
        classical_runs,
        summary,
    }
}

fn run_single(config: &ComparisonConfig, constraints: CircuitConstraints, seed: u64) -> RunResult {
    let mut sim = Simulation::new();
    let mut pop = ConstrainedPopulation::new(config.forager_config.clone(), constraints, seed);

    let mut stats_history = Vec::new();

    // Initial resources
    pop.spawn_resources(&mut sim, config.heat_patches * 5, config.heat_value);

    for tick in 0..config.max_ticks {
        if tick > 0 && tick % config.resource_interval == 0 {
            pop.spawn_resources(&mut sim, config.heat_patches, config.heat_value);
        }

        pop.step(&mut sim);

        if tick % config.log_interval == 0 {
            stats_history.push(pop.stats().clone());
        }

        if pop.is_extinct() {
            let max_gen = pop.stats().max_generation;
            return RunResult {
                extinct: true,
                final_population: 0,
                max_generation: max_gen,
                final_quantum_ness: 0.0,
                stats_history,
            };
        }
    }

    let final_stats = pop.stats().clone();
    stats_history.push(final_stats.clone());

    RunResult {
        extinct: false,
        final_population: final_stats.size,
        max_generation: final_stats.max_generation,
        final_quantum_ness: final_stats.avg_quantum_ness,
        stats_history,
    }
}

fn compute_summary(quantum: &[RunResult], classical: &[RunResult]) -> ComparisonSummary {
    let n_q = quantum.len() as f32;
    let n_c = classical.len() as f32;

    let q_ext = quantum.iter().filter(|r| r.extinct).count() as f32 / n_q;
    let c_ext = classical.iter().filter(|r| r.extinct).count() as f32 / n_c;

    let q_pop: f32 = quantum
        .iter()
        .filter(|r| !r.extinct)
        .map(|r| r.final_population as f32)
        .sum::<f32>()
        / quantum.iter().filter(|r| !r.extinct).count().max(1) as f32;

    let c_pop: f32 = classical
        .iter()
        .filter(|r| !r.extinct)
        .map(|r| r.final_population as f32)
        .sum::<f32>()
        / classical.iter().filter(|r| !r.extinct).count().max(1) as f32;

    let q_gen: f32 = quantum.iter().map(|r| r.max_generation as f32).sum::<f32>() / n_q;
    let c_gen: f32 = classical
        .iter()
        .map(|r| r.max_generation as f32)
        .sum::<f32>()
        / n_c;

    let q_qness: f32 = quantum
        .iter()
        .filter(|r| !r.extinct)
        .map(|r| r.final_quantum_ness)
        .sum::<f32>()
        / quantum.iter().filter(|r| !r.extinct).count().max(1) as f32;

    let pop_adv = q_pop - c_pop;
    let gen_adv = q_gen - c_gen;

    let winner = if pop_adv > 10.0 {
        "QUANTUM (significant population advantage)".to_string()
    } else if pop_adv < -10.0 {
        "CLASSICAL (significant population advantage)".to_string()
    } else if gen_adv > 2.0 {
        "QUANTUM (deeper evolution)".to_string()
    } else if gen_adv < -2.0 {
        "CLASSICAL (deeper evolution)".to_string()
    } else {
        "TIE (no significant difference)".to_string()
    };

    ComparisonSummary {
        quantum_mean_population: q_pop,
        quantum_extinction_rate: q_ext,
        quantum_mean_generation: q_gen,
        quantum_mean_qness: q_qness,
        classical_mean_population: c_pop,
        classical_extinction_rate: c_ext,
        classical_mean_generation: c_gen,
        population_advantage: pop_adv,
        generation_advantage: gen_adv,
        winner,
    }
}

/// Print comparison results.
pub fn print_comparison(result: &ComparisonResult) {
    let s = &result.summary;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     QUANTUM VS CLASSICAL: THE EXPERIMENT                   ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("                    QUANTUM        CLASSICAL      DIFFERENCE");
    println!("─────────────────────────────────────────────────────────────");
    println!(
        "Mean Population     {:>8.1}        {:>8.1}        {:>+8.1}",
        s.quantum_mean_population, s.classical_mean_population, s.population_advantage
    );
    println!(
        "Extinction Rate     {:>7.0}%        {:>7.0}%",
        s.quantum_extinction_rate * 100.0,
        s.classical_extinction_rate * 100.0
    );
    println!(
        "Mean Max Gen        {:>8.1}        {:>8.1}        {:>+8.1}",
        s.quantum_mean_generation, s.classical_mean_generation, s.generation_advantage
    );
    println!(
        "Final Quantum-ness  {:>8.3}           N/A",
        s.quantum_mean_qness
    );
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  WINNER: {}", s.winner);
    println!("═══════════════════════════════════════════════════════════════");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained_circuits::count_entangling_gates;

    fn quick_config() -> ComparisonConfig {
        ComparisonConfig {
            forager_config: QuantumForagerConfig {
                initial_population: 10,
                max_population: 30,
                ..Default::default()
            },
            max_ticks: 50,
            log_interval: 10,
            replications: 2,
            ..Default::default()
        }
    }

    #[test]
    fn test_run_comparison() {
        let config = quick_config();
        let result = run_quantum_vs_classical(&config);

        assert_eq!(result.quantum_runs.len(), 2);
        assert_eq!(result.classical_runs.len(), 2);
    }

    #[test]
    fn test_constrained_forager() {
        let config = QuantumForagerConfig::default();
        let constraints = CircuitConstraints::classical_only();

        let forager =
            ConstrainedForager::new_random(1, 50, 50, &config, constraints.clone(), 12345);

        // Should have no entangling gates
        assert_eq!(count_entangling_gates(&forager.forager.genome), 0);
    }

    #[test]
    fn test_constrained_reproduction() {
        let mut config = QuantumForagerConfig::default();
        config.reproduction_threshold = 50.0;

        let constraints = CircuitConstraints::classical_only();
        let mut forager =
            ConstrainedForager::new_random(1, 50, 50, &config, constraints.clone(), 12345);
        forager.forager.energy = 100.0;

        let offspring = forager.reproduce(2, &config, 54321);
        assert!(offspring.is_some());

        let child = offspring.unwrap();
        // Child should also have no entangling gates
        assert_eq!(count_entangling_gates(&child.forager.genome), 0);
    }
}
