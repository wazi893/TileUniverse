//! EPIC 91: Three-Way Comparison - Full Quantum vs Separable Quantum vs Classical NN
//!
//! This module runs the definitive three-way comparison:
//! - Group A: Full quantum organisms (entangling gates allowed)
//! - Group B: Separable quantum organisms (no entangling gates)
//! - Group C: Classical neural network organisms
//!
//! Same environment, same selection pressure, same everything —
//! except the brain type. If Full Quantum outperforms both Separable and Classical,
//! we have genuine quantum advantage evidence.

use crate::brain::{Brain, BrainType};
use crate::classical_brain::Move;
use crate::quantum::QRng;
use crate::quantum_brain::sense_local_fields;
use crate::quantum_forager::{ForagerRegion, PopulationStats, QuantumForagerConfig};
use crate::simulation::Simulation;
use std::collections::HashMap;

// =============================================================================
// Brain-Based Forager
// =============================================================================

/// A forager organism with a Brain (quantum or classical NN).
#[derive(Clone, Debug)]
pub struct BrainForager {
    /// Unique identifier
    pub id: u64,

    /// Generation number (0 for initial population)
    pub generation: u32,

    /// Parent's ID (None for initial population)
    pub parent_id: Option<u64>,

    /// Current X position
    pub x: u32,

    /// Current Y position
    pub y: u32,

    /// Current energy level
    pub energy: f32,

    /// Brain (quantum or classical)
    pub brain: Brain,

    /// Age in ticks
    pub age: u32,

    /// Total energy consumed from environment
    pub total_consumed: f32,

    /// Number of offspring produced
    pub offspring_count: u32,

    /// Decision seed (for determinism)
    pub decision_seed: u64,
}

impl BrainForager {
    /// Create a new random forager with specified brain type.
    pub fn new_random(
        id: u64,
        x: u32,
        y: u32,
        energy: f32,
        brain_type: BrainType,
        seed: u64,
    ) -> Self {
        let mut rng = QRng::new(seed);
        let brain = Brain::random(brain_type, 4, 8, &mut rng); // 4 qubits, 8 gates initial

        Self {
            id,
            generation: 0,
            parent_id: None,
            x,
            y,
            energy,
            brain,
            age: 0,
            total_consumed: 0.0,
            offspring_count: 0,
            decision_seed: seed.wrapping_mul(0x517cc1b727220a95),
        }
    }

    /// Reproduce with mutation.
    pub fn reproduce(
        &mut self,
        offspring_id: u64,
        reproduction_cost: f32,
        reproduction_threshold: f32,
        mutation_rate: f32,
        seed: u64,
    ) -> Option<BrainForager> {
        if self.energy < reproduction_threshold {
            return None;
        }

        self.energy -= reproduction_cost;
        self.offspring_count += 1;

        // Mutate brain
        let mut rng = QRng::new(seed);
        let offspring_brain = self.brain.reproduce(&mut rng, mutation_rate, 25);

        Some(BrainForager {
            id: offspring_id,
            generation: self.generation + 1,
            parent_id: Some(self.id),
            x: self.x,
            y: self.y,
            energy: reproduction_cost,
            brain: offspring_brain,
            age: 0,
            total_consumed: 0.0,
            offspring_count: 0,
            decision_seed: seed.wrapping_mul(0x9e3779b97f4a7c15),
        })
    }
}

// =============================================================================
// Brain-Based Population
// =============================================================================

/// Population of brain-based foragers.
pub struct BrainPopulation {
    foragers: HashMap<u64, BrainForager>,
    next_id: u64,
    config: QuantumForagerConfig,
    brain_type: BrainType,
    tick: u64,
    seed: u64,
    last_stats: PopulationStats,
}

impl BrainPopulation {
    /// Create a new brain-based population.
    pub fn new(config: QuantumForagerConfig, brain_type: BrainType, seed: u64) -> Self {
        let mut pop = Self {
            foragers: HashMap::new(),
            next_id: 0,
            config,
            brain_type,
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

            let forager = BrainForager::new_random(
                id,
                x,
                y,
                self.config.initial_energy,
                self.brain_type,
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

    /// Spawn heat resources.
    pub fn spawn_resources(&self, sim: &mut Simulation, n_patches: u32, value: u32) {
        let mut rng =
            crate::quantum_brain::SimpleRng::new(self.seed.wrapping_add(self.tick * 123456789));

        for _ in 0..n_patches {
            let x = (self.config.region.x0 + rng.gen_range(0, self.config.region.width)) as usize;
            let y = (self.config.region.y0 + rng.gen_range(0, self.config.region.height)) as usize;

            sim.set_heat_field_for_test(x, y, value);
        }
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
            let forager = self.foragers.get_mut(&id).unwrap();

            // Age and metabolism
            forager.age += 1;
            forager.energy -= self.config.metabolism_cost;

            if forager.energy <= 0.0 {
                deaths.push(id);
                continue;
            }

            // Sense
            let readings = sense_local_fields(sim, forager.x, forager.y);

            // Decide using Brain
            let mut rng = QRng::new(forager.decision_seed.wrapping_add(forager.age as u64));
            let movement = forager.brain.decide(&readings, &mut rng);

            // Convert Move to deltas
            let (dx, dy) = match movement {
                Move::Up => (0, -1),
                Move::Down => (0, 1),
                Move::Left => (-1, 0),
                Move::Right => (1, 0),
                Move::Stay => (0, 0),
            };

            // Move
            let new_x = (forager.x as i32 + dx).max(0) as u32;
            let new_y = (forager.y as i32 + dy).max(0) as u32;
            let (cx, cy) = self.config.region.clamp(new_x as i32, new_y as i32);

            forager.x = cx;
            forager.y = cy;
            forager.energy -= self.config.movement_cost;

            if forager.energy <= 0.0 {
                deaths.push(id);
                continue;
            }

            // Consume resources
            if self.config.consume_resources {
                let xu = forager.x as usize;
                let yu = forager.y as usize;

                let heat = sim.get_heat_field_for_test(xu, yu);
                if heat > 0 {
                    let gain = heat as f32 * self.config.heat_energy_factor;
                    forager.energy += gain;
                    forager.total_consumed += gain;
                    total_consumed += gain;
                    sim.set_heat_field_for_test(xu, yu, 0);
                }

                let charge = sim.get_charge_field_for_test(xu, yu);
                if charge > 0 {
                    let gain = charge as f32 * self.config.charge_energy_factor;
                    forager.energy += gain;
                    forager.total_consumed += gain;
                    total_consumed += gain;
                    sim.set_charge_field_for_test(xu, yu, 0);
                }
            }

            // Reproduction
            if current_population + births.len() < max_population {
                let offspring_id = self.next_id;
                let reproduction_seed = self.seed.wrapping_add(self.tick * 9999991 + id * 7919);

                if let Some(offspring) = forager.reproduce(
                    offspring_id,
                    self.config.reproduction_cost,
                    self.config.reproduction_threshold,
                    self.config.mutation_rate,
                    reproduction_seed,
                ) {
                    births.push(offspring);
                    self.next_id += 1;
                }
            }
        }

        // Apply deaths
        for id in deaths {
            self.foragers.remove(&id);
        }

        // Apply births
        for offspring in births {
            self.foragers.insert(offspring.id, offspring);
        }

        // Update statistics
        self.update_stats(total_consumed);

        &self.last_stats
    }

    fn update_stats(&mut self, total_consumed: f32) {
        let population = self.foragers.len() as u32;
        let max_generation = self
            .foragers
            .values()
            .map(|f| f.generation)
            .max()
            .unwrap_or(0);

        let mean_quantumness = if population > 0 {
            let sum: f32 = self.foragers.values().map(|f| f.brain.quantumness()).sum();
            sum / population as f32
        } else {
            0.0
        };

        let total_energy: f32 = self.foragers.values().map(|f| f.energy).sum();
        let avg_energy = if population > 0 {
            total_energy / population as f32
        } else {
            0.0
        };
        let avg_age = if population > 0 {
            let total_age: u32 = self.foragers.values().map(|f| f.age).sum();
            total_age as f32 / population as f32
        } else {
            0.0
        };
        let avg_circuit_length = if population > 0 {
            let total_len: usize = self
                .foragers
                .values()
                .map(|f| f.brain.genome_length())
                .sum();
            total_len as f32 / population as f32
        } else {
            0.0
        };

        self.last_stats = PopulationStats {
            size: population as usize,
            total_energy,
            avg_energy,
            avg_age,
            avg_circuit_length,
            avg_quantum_ness: mean_quantumness,
            max_generation,
            births_this_tick: 0, // Not tracked in this simplified version
            deaths_this_tick: 0, // Not tracked in this simplified version
            total_consumed_this_tick: total_consumed,
        };
    }
}

// =============================================================================
// Experiment Configuration
// =============================================================================

/// Configuration for three-way comparison experiment.
#[derive(Clone, Debug)]
pub struct ThreeWayConfig {
    pub forager_config: QuantumForagerConfig,
    pub max_ticks: u64,
    pub resource_interval: u64,
    pub heat_patches: u32,
    pub heat_value: u32,
    pub log_interval: u64,
    pub replications: usize, // Number of seeds to run
    pub seed_start: u64,
    pub brain_types: Vec<BrainType>, // Which brain types to compare
}

impl Default for ThreeWayConfig {
    fn default() -> Self {
        Self {
            forager_config: QuantumForagerConfig {
                initial_population: 30,
                initial_energy: 100.0,
                n_qubits: 4,
                initial_circuit_length: 8,
                max_circuit_length: 25,
                movement_cost: 0.5,
                metabolism_cost: 0.25,
                reproduction_threshold: 140.0,
                reproduction_cost: 55.0,
                mutation_rate: 0.7, // Preregistered
                heat_energy_factor: 0.7,
                charge_energy_factor: 0.3,
                consume_resources: true,
                max_population: 150,
                region: ForagerRegion {
                    x0: 30,
                    y0: 30,
                    width: 100,
                    height: 100,
                },
            },
            max_ticks: 1000,
            resource_interval: 8,
            heat_patches: 19,
            heat_value: 35,
            log_interval: 100,
            replications: 100,
            seed_start: 1,
            brain_types: vec![
                BrainType::FullQuantum,
                BrainType::SeparableQuantum,
                BrainType::ClassicalNN,
            ],
        }
    }
}

impl ThreeWayConfig {
    /// Create a four-way comparison config that includes QUBO brain
    pub fn four_way() -> Self {
        let mut config = Self::default();
        config.brain_types = vec![
            BrainType::FullQuantum,
            BrainType::SeparableQuantum,
            BrainType::ClassicalNN,
            BrainType::QuboBrain,
        ];
        config
    }

    /// Create a config comparing just QUBO vs Classical
    pub fn qubo_vs_classical() -> Self {
        let mut config = Self::default();
        config.brain_types = vec![
            BrainType::QuboBrain,
            BrainType::ClassicalNN,
            BrainType::ClassicalNN,
        ];
        config
    }
}

// =============================================================================
// Results
// =============================================================================

/// Result from a single trial.
#[derive(Clone, Debug)]
pub struct TrialResult {
    pub seed: u64,
    pub brain_type: BrainType,
    pub final_population: u32,
    pub max_generation: u32,
    pub extinct: bool,
    pub final_quantumness: f32,
    // EPIC 92: Robust outcome metrics
    pub population_auc: f64, // Area-under-curve: sum of population at each tick
    pub mean_energy: f32,    // Average energy per organism-tick
    pub total_reproductions: u32, // Total successful reproduction events
    pub ticks_to_extinction: Option<u64>, // First tick when population hit 0 (None if never)
    pub viability_fraction: f32, // Fraction of ticks with population >= 10
}

/// Aggregated results for one brain type.
#[derive(Clone, Debug)]
pub struct GroupResult {
    pub brain_type: BrainType,
    pub trials: Vec<TrialResult>,
    pub mean_population: f64,
    pub std_population: f64,
    pub median_population: f64,
    pub extinctions: usize,
    pub mean_quantumness: f64,
    // EPIC 92: Robust outcome metrics
    pub mean_population_auc: f64,
    pub mean_energy: f64,
    pub mean_reproductions: f64,
    pub mean_viability_fraction: f64,
    pub median_ticks_to_extinction: Option<f64>, // Median among extinct trials
}

/// Pairwise comparison statistics.
#[derive(Clone, Debug)]
pub struct PairwiseComparison {
    pub group_a: String,
    pub group_b: String,
    pub mean_difference: f64,
    pub std_difference: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub cohens_d: f64,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
}

/// Full three-way results.
#[derive(Clone, Debug)]
pub struct ThreeWayResults {
    pub config: ThreeWayConfig,
    pub full_quantum: GroupResult,
    pub separable_quantum: GroupResult,
    pub classical_nn: GroupResult,
    pub fq_vs_sq: PairwiseComparison,
    pub fq_vs_cn: PairwiseComparison,
    pub sq_vs_cn: PairwiseComparison,
}

// =============================================================================
// Experiment Runner
// =============================================================================

/// Run a single trial for one brain type.
fn run_single_trial(seed: u64, brain_type: BrainType, config: &ThreeWayConfig) -> TrialResult {
    let mut sim = Simulation::new();
    let mut pop = BrainPopulation::new(config.forager_config.clone(), brain_type, seed);

    // Initial resources
    pop.spawn_resources(&mut sim, config.heat_patches * 5, config.heat_value);

    // EPIC 92: Track metrics over time
    let mut population_auc = 0.0_f64;
    let mut total_energy = 0.0_f64;
    let mut total_organism_ticks = 0_u64;
    let mut total_reproductions = 0_u32;
    let mut ticks_viable = 0_u64; // Ticks with population >= 10
    let mut ticks_to_extinction: Option<u64> = None;
    let mut already_extinct = false;

    for tick in 0..config.max_ticks {
        if tick > 0 && tick % config.resource_interval == 0 {
            pop.spawn_resources(&mut sim, config.heat_patches, config.heat_value);
        }

        let pop_before = pop.size();
        pop.step(&mut sim);
        let pop_after = pop.size();

        // Track metrics
        population_auc += pop_after as f64;

        if pop_after >= 10 {
            ticks_viable += 1;
        }

        // Track first extinction
        if pop_after == 0 && !already_extinct {
            ticks_to_extinction = Some(tick);
            already_extinct = true;
        }

        // Count reproductions (population increase)
        if pop_after > pop_before {
            total_reproductions += (pop_after - pop_before) as u32;
        }

        // Sum energy across all organisms
        for organism in pop.foragers.values() {
            total_energy += organism.energy as f64;
            total_organism_ticks += 1;
        }
    }

    let stats = pop.stats();
    let mean_energy = if total_organism_ticks > 0 {
        (total_energy / total_organism_ticks as f64) as f32
    } else {
        0.0
    };
    let viability_fraction = ticks_viable as f32 / config.max_ticks as f32;

    TrialResult {
        seed,
        brain_type,
        final_population: stats.size as u32,
        max_generation: stats.max_generation,
        extinct: pop.is_extinct(),
        final_quantumness: stats.avg_quantum_ness,
        population_auc,
        mean_energy,
        total_reproductions,
        ticks_to_extinction,
        viability_fraction,
    }
}

/// Run full three-way comparison.
pub fn run_three_way_comparison(config: &ThreeWayConfig) -> ThreeWayResults {
    let mut all_results: HashMap<BrainType, Vec<TrialResult>> = HashMap::new();

    for &brain_type in &config.brain_types {
        println!("\n=== Running {} ===", brain_type.display_name());
        let mut trials = Vec::new();

        for i in 0..config.replications {
            let seed = config.seed_start + i as u64;

            if (i + 1) % 10 == 0 {
                println!("  Completed {}/{} seeds", i + 1, config.replications);
            }

            let result = run_single_trial(seed, brain_type, config);
            trials.push(result);
        }

        all_results.insert(brain_type, trials);
    }

    // Aggregate by group - use first 3 brain types for compatibility
    let full_quantum = aggregate_group(all_results.get(&config.brain_types[0]).unwrap());
    let separable_quantum = aggregate_group(all_results.get(&config.brain_types[1]).unwrap());
    let classical_nn = aggregate_group(all_results.get(&config.brain_types[2]).unwrap());

    // Compute pairwise comparisons
    let fq_vs_sq = compute_pairwise(&full_quantum, &separable_quantum);
    let fq_vs_cn = compute_pairwise(&full_quantum, &classical_nn);
    let sq_vs_cn = compute_pairwise(&separable_quantum, &classical_nn);

    ThreeWayResults {
        config: config.clone(),
        full_quantum,
        separable_quantum,
        classical_nn,
        fq_vs_sq,
        fq_vs_cn,
        sq_vs_cn,
    }
}

// =============================================================================
// Statistics
// =============================================================================

fn aggregate_group(trials: &[TrialResult]) -> GroupResult {
    let pops: Vec<f64> = trials.iter().map(|t| t.final_population as f64).collect();
    let mut sorted_pops = pops.clone();
    sorted_pops.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_pop = mean(&pops);
    let std_pop = std_dev(&pops);
    let median_pop = sorted_pops[sorted_pops.len() / 2];
    let extinctions = trials.iter().filter(|t| t.extinct).count();
    let mean_qness = mean(
        &trials
            .iter()
            .map(|t| t.final_quantumness as f64)
            .collect::<Vec<_>>(),
    );

    // EPIC 92: Aggregate robust outcome metrics
    let aucs: Vec<f64> = trials.iter().map(|t| t.population_auc).collect();
    let energies: Vec<f64> = trials.iter().map(|t| t.mean_energy as f64).collect();
    let reproductions: Vec<f64> = trials
        .iter()
        .map(|t| t.total_reproductions as f64)
        .collect();
    let viability: Vec<f64> = trials.iter().map(|t| t.viability_fraction as f64).collect();

    // Median ticks to extinction (among extinct trials only)
    let mut extinction_ticks: Vec<f64> = trials
        .iter()
        .filter_map(|t| t.ticks_to_extinction.map(|x| x as f64))
        .collect();
    extinction_ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_extinction = if !extinction_ticks.is_empty() {
        Some(extinction_ticks[extinction_ticks.len() / 2])
    } else {
        None
    };

    GroupResult {
        brain_type: trials[0].brain_type,
        trials: trials.to_vec(),
        mean_population: mean_pop,
        std_population: std_pop,
        median_population: median_pop,
        extinctions,
        mean_quantumness: mean_qness,
        mean_population_auc: mean(&aucs),
        mean_energy: mean(&energies),
        mean_reproductions: mean(&reproductions),
        mean_viability_fraction: mean(&viability),
        median_ticks_to_extinction: median_extinction,
    }
}

fn compute_pairwise(group_a: &GroupResult, group_b: &GroupResult) -> PairwiseComparison {
    // Paired comparison (same seed → same environment)
    let differences: Vec<f64> = group_a
        .trials
        .iter()
        .zip(group_b.trials.iter())
        .map(|(a, b)| a.final_population as f64 - b.final_population as f64)
        .collect();

    let mean_diff = mean(&differences);
    let std_diff = std_dev(&differences);
    let n = differences.len() as f64;
    let se = std_diff / n.sqrt();
    let t_stat = if se > 0.0 { mean_diff / se } else { 0.0 };

    // 95% CI (t-distribution approximation for df=99 ≈ 1.984)
    let t_critical = 1.984;
    let ci_lower = mean_diff - t_critical * se;
    let ci_upper = mean_diff + t_critical * se;

    // Cohen's d (paired)
    let cohens_d = if std_diff > 0.0 {
        mean_diff / std_diff
    } else {
        0.0
    };

    // p-value approximation (two-tailed)
    let p_value = if t_stat.abs() > 3.39 {
        0.001
    } else if t_stat.abs() > 2.63 {
        0.01
    } else if t_stat.abs() > 1.98 {
        0.05
    } else {
        0.10
    };

    // Win/loss/tie counts (threshold ±10)
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;

    for diff in &differences {
        if *diff > 10.0 {
            wins += 1;
        } else if *diff < -10.0 {
            losses += 1;
        } else {
            ties += 1;
        }
    }

    PairwiseComparison {
        group_a: group_a.brain_type.display_name().to_string(),
        group_b: group_b.brain_type.display_name().to_string(),
        mean_difference: mean_diff,
        std_difference: std_diff,
        ci_lower,
        ci_upper,
        t_statistic: t_stat,
        p_value,
        cohens_d,
        wins,
        losses,
        ties,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

// =============================================================================
// Output Formatting
// =============================================================================

/// Print three-way results to console.
pub fn print_three_way_results(results: &ThreeWayResults) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   THREE-WAY COMPARISON: Quantum Advantage Verification    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Configuration:");
    println!(
        "  - Seeds: {}-{} (n={})",
        results.config.seed_start,
        results.config.seed_start + results.config.replications as u64 - 1,
        results.config.replications
    );
    println!("  - Ticks: {}", results.config.max_ticks);
    println!(
        "  - Environment: {} patches, value {}",
        results.config.heat_patches, results.config.heat_value
    );

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  RESULTS BY GROUP");
    println!("══════════════════════════════════════════════════════════════\n");

    println!("                      FULL Q     SEP Q     CLASSICAL");
    println!("────────────────────────────────────────────────────────");
    println!(
        "Mean Population      {:6.1}     {:6.1}     {:6.1}",
        results.full_quantum.mean_population,
        results.separable_quantum.mean_population,
        results.classical_nn.mean_population
    );
    println!(
        "Std Dev              {:6.1}     {:6.1}     {:6.1}",
        results.full_quantum.std_population,
        results.separable_quantum.std_population,
        results.classical_nn.std_population
    );
    println!(
        "Median               {:6.1}     {:6.1}     {:6.1}",
        results.full_quantum.median_population,
        results.separable_quantum.median_population,
        results.classical_nn.median_population
    );
    println!(
        "Extinctions          {:6}     {:6}     {:6}",
        results.full_quantum.extinctions,
        results.separable_quantum.extinctions,
        results.classical_nn.extinctions
    );
    println!(
        "Mean Quantumness     {:6.3}     {:6.3}     {:6.3}",
        results.full_quantum.mean_quantumness,
        results.separable_quantum.mean_quantumness,
        results.classical_nn.mean_quantumness
    );

    // EPIC 92: Display robust outcome metrics
    println!("\n────────── Robust Outcome Metrics (EPIC 92) ─────────");
    println!(
        "Pop AUC (×1000)      {:6.1}     {:6.1}     {:6.1}",
        results.full_quantum.mean_population_auc / 1000.0,
        results.separable_quantum.mean_population_auc / 1000.0,
        results.classical_nn.mean_population_auc / 1000.0
    );
    println!(
        "Mean Energy          {:6.1}     {:6.1}     {:6.1}",
        results.full_quantum.mean_energy,
        results.separable_quantum.mean_energy,
        results.classical_nn.mean_energy
    );
    println!(
        "Total Reproductions  {:6.0}     {:6.0}     {:6.0}",
        results.full_quantum.mean_reproductions,
        results.separable_quantum.mean_reproductions,
        results.classical_nn.mean_reproductions
    );
    println!(
        "Viability Fraction   {:6.3}     {:6.3}     {:6.3}",
        results.full_quantum.mean_viability_fraction,
        results.separable_quantum.mean_viability_fraction,
        results.classical_nn.mean_viability_fraction
    );

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  PAIRWISE COMPARISONS");
    println!("══════════════════════════════════════════════════════════════\n");

    print_pairwise(&results.fq_vs_sq);
    println!();
    print_pairwise(&results.fq_vs_cn);
    println!();
    print_pairwise(&results.sq_vs_cn);
}

fn print_pairwise(comp: &PairwiseComparison) {
    println!("{} vs {}:", comp.group_a, comp.group_b);
    println!("  Mean Difference:  {:+.1}", comp.mean_difference);
    println!(
        "  95% CI:           [{:+.1}, {:+.1}]",
        comp.ci_lower, comp.ci_upper
    );
    println!("  t-statistic:      {:.2}", comp.t_statistic);
    println!("  p-value:          {:.3}", comp.p_value);
    println!("  Cohen's d:        {:.2}", comp.cohens_d);
    println!(
        "  Win/Loss/Tie:     {}/{}/{}",
        comp.wins, comp.losses, comp.ties
    );

    let significance = if comp.p_value < 0.001 {
        "***"
    } else if comp.p_value < 0.01 {
        "**"
    } else if comp.p_value < 0.05 {
        "*"
    } else {
        "ns"
    };
    println!(
        "  Significance:     {} {}",
        significance,
        if comp.p_value < 0.05 { "✓" } else { "" }
    );
}
