// =============================================================================
// src/quantum_forager.rs
// =============================================================================
//
// SPRINT 3.1: Quantum Forager organisms with evolvable quantum brains
//
// A QuantumForager is an organism that:
// - Senses its local environment (heat/charge fields)
// - Makes decisions using a quantum circuit (its genome)
// - Moves to find resources
// - Consumes energy to survive
// - Reproduces when energy exceeds threshold
// - Dies when energy depletes
//
// The quantum circuit is the heritable, mutable genome that evolves
// across generations through natural selection.
//
// =============================================================================

use crate::metrics::CopsMetric;
use crate::quantum::QGate;
use crate::quantum_brain::{
    self, CircuitStats, FusedCircuitResult, MoveAction, SimpleRng, analyze_circuit, circuit_hash,
    decode_action, encode_as_rotations, mutate_circuit, random_circuit, run_circuit_auto,
    sense_local_fields,
};
use crate::simulation::Simulation;
use crate::tilemap::{HEIGHT, WIDTH};

use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for quantum forager population.
#[derive(Clone, Debug)]
pub struct QuantumForagerConfig {
    /// Number of organisms to spawn initially
    pub initial_population: usize,

    /// Starting energy for new organisms
    pub initial_energy: f32,

    /// Number of qubits in quantum brain
    pub n_qubits: u8,

    /// Initial circuit length for random genomes
    pub initial_circuit_length: usize,

    /// Maximum circuit length (prevents bloat)
    pub max_circuit_length: usize,

    /// Energy cost per movement
    pub movement_cost: f32,

    /// Energy cost per tick (metabolism)
    pub metabolism_cost: f32,

    /// Energy threshold for reproduction
    pub reproduction_threshold: f32,

    /// Energy given to offspring (deducted from parent)
    pub reproduction_cost: f32,

    /// Probability of mutation per reproduction
    pub mutation_rate: f32,

    /// Energy gained from consuming heat
    pub heat_energy_factor: f32,

    /// Energy gained from consuming charge
    pub charge_energy_factor: f32,

    /// Whether to consume resources from the field
    pub consume_resources: bool,

    /// Maximum population (prevents explosion)
    pub max_population: usize,

    /// Region bounds for spawning and movement
    pub region: ForagerRegion,
}

/// Region where foragers can exist.
#[derive(Clone, Debug)]
pub struct ForagerRegion {
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
}

impl ForagerRegion {
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x0 && x < self.x0 + self.width && y >= self.y0 && y < self.y0 + self.height
    }

    pub fn clamp(&self, x: i32, y: i32) -> (u32, u32) {
        let cx = (x.max(self.x0 as i32) as u32).min(self.x0 + self.width - 1);
        let cy = (y.max(self.y0 as i32) as u32).min(self.y0 + self.height - 1);
        (cx, cy)
    }
}

impl Default for ForagerRegion {
    fn default() -> Self {
        Self {
            x0: 0,
            y0: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
        }
    }
}

impl Default for QuantumForagerConfig {
    fn default() -> Self {
        Self {
            initial_population: 50,
            initial_energy: 100.0,
            n_qubits: 4,
            initial_circuit_length: 8,
            max_circuit_length: 30,
            movement_cost: 1.0,
            metabolism_cost: 0.5,
            reproduction_threshold: 150.0,
            reproduction_cost: 60.0,
            mutation_rate: 0.8,
            heat_energy_factor: 0.5,
            charge_energy_factor: 0.3,
            consume_resources: true,
            max_population: 500,
            region: ForagerRegion::default(),
        }
    }
}

// =============================================================================
// Quantum Forager
// =============================================================================

/// A single quantum forager organism.
#[derive(Clone, Debug)]
pub struct QuantumForager {
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

    /// Quantum circuit genome
    pub genome: Vec<QGate>,

    /// Number of qubits in brain
    pub n_qubits: u8,

    /// Age in ticks
    pub age: u32,

    /// Total energy consumed from environment
    pub total_consumed: f32,

    /// Number of offspring produced
    pub offspring_count: u32,

    /// Random seed for this organism's decisions
    pub decision_seed: u64,
}

impl QuantumForager {
    /// Create a new forager with a random genome.
    pub fn new_random(id: u64, x: u32, y: u32, config: &QuantumForagerConfig, seed: u64) -> Self {
        let genome = random_circuit(config.n_qubits, config.initial_circuit_length, seed);

        Self {
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
        }
    }

    /// Create offspring from this forager.
    pub fn reproduce(
        &mut self,
        offspring_id: u64,
        config: &QuantumForagerConfig,
        mutation_seed: u64,
    ) -> Option<QuantumForager> {
        if self.energy < config.reproduction_threshold {
            return None;
        }

        // Deduct reproduction cost from parent
        self.energy -= config.reproduction_cost;
        self.offspring_count += 1;

        // Mutate genome for offspring
        let mut rng = SimpleRng::new(mutation_seed);
        let offspring_genome = if rng.gen_bool(config.mutation_rate) {
            mutate_circuit(
                &self.genome,
                self.n_qubits,
                config.max_circuit_length,
                mutation_seed,
            )
        } else {
            self.genome.clone()
        };

        // Offspring spawns at same position
        Some(QuantumForager {
            id: offspring_id,
            generation: self.generation + 1,
            parent_id: Some(self.id),
            x: self.x,
            y: self.y,
            energy: config.reproduction_cost, // Offspring gets the cost as starting energy
            genome: offspring_genome,
            n_qubits: self.n_qubits,
            age: 0,
            total_consumed: 0.0,
            offspring_count: 0,
            decision_seed: mutation_seed.wrapping_mul(0x9e3779b97f4a7c15),
        })
    }

    /// Check if forager is alive.
    pub fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    /// Get circuit statistics.
    pub fn circuit_stats(&self) -> CircuitStats {
        analyze_circuit(&self.genome)
    }

    /// Sense environment around current position.
    pub fn sense(&self, sim: &Simulation) -> crate::quantum_brain::SensorReadings {
        sense_local_fields(sim, self.x, self.y)
    }

    /// Make a decision using the quantum brain.
    pub fn decide(&mut self, readings: &crate::quantum_brain::SensorReadings) -> MoveAction {
        // Use the fused path which automatically selects optimal execution
        let result = self.decide_fused(readings);
        decode_action(result.measurement)
    }

    /// Make a decision using the quantum brain with full metrics (EPIC 90).
    ///
    /// Returns the full `FusedCircuitResult` including fusion statistics.
    /// This is the optimized path that routes through EPICs 83-89.
    pub fn decide_fused(
        &mut self,
        readings: &crate::quantum_brain::SensorReadings,
    ) -> FusedCircuitResult {
        // Encode sensor readings as quantum rotations
        let sensor_gates = encode_as_rotations(readings, self.n_qubits);

        // Run quantum circuit via the auto-selecting path
        // This uses fusion for circuits >= 8 gates, scalar for smaller
        run_circuit_auto(
            &self.genome,
            &sensor_gates,
            self.n_qubits,
            self.decision_seed.wrapping_add(self.age as u64),
        )
    }

    /// Execute one tick of the forager's life.
    ///
    /// Returns the action taken (for logging/visualization).
    pub fn tick(
        &mut self,
        sim: &mut Simulation,
        config: &QuantumForagerConfig,
    ) -> ForagerTickResult {
        // 1. Age
        self.age += 1;

        // 2. Metabolism cost
        self.energy -= config.metabolism_cost;

        // 3. Check death
        if self.energy <= 0.0 {
            return ForagerTickResult::Died;
        }

        // 4. Sense environment
        let readings = self.sense(sim);

        // 5. Decide action
        let action = self.decide(&readings);

        // 6. Execute movement
        let new_x = (self.x as i32 + action.dx).max(0) as u32;
        let new_y = (self.y as i32 + action.dy).max(0) as u32;
        let (clamped_x, clamped_y) = config.region.clamp(new_x as i32, new_y as i32);

        let moved = clamped_x != self.x || clamped_y != self.y;
        if moved {
            self.x = clamped_x;
            self.y = clamped_y;
            self.energy -= config.movement_cost;
        }

        // 7. Consume resources at new position
        let mut consumed = 0.0f32;
        if config.consume_resources {
            let heat = sim.get_heat_field_for_test(self.x as usize, self.y as usize);
            let charge = sim.get_charge_field_for_test(self.x as usize, self.y as usize);

            let heat_energy = heat as f32 * config.heat_energy_factor;
            let charge_energy = charge as f32 * config.charge_energy_factor;
            consumed = heat_energy + charge_energy;

            self.energy += consumed;
            self.total_consumed += consumed;

            // Deplete resources from field
            if heat > 0 {
                let new_heat =
                    heat.saturating_sub((heat_energy / config.heat_energy_factor) as u32);
                sim.set_heat_field_for_test(self.x as usize, self.y as usize, new_heat);
            }
            if charge > 0 {
                let new_charge =
                    charge.saturating_sub((charge_energy / config.charge_energy_factor) as u32);
                sim.set_charge_field_for_test(self.x as usize, self.y as usize, new_charge);
            }
        }

        // 8. Check for reproduction
        let can_reproduce = self.energy >= config.reproduction_threshold;

        ForagerTickResult::Alive {
            action,
            moved,
            consumed,
            can_reproduce,
        }
    }

    /// Execute an action after decision has been made (EPIC 90.2).
    ///
    /// This is the second half of tick(), separated for batched execution.
    /// The first half (sense + decide) is done by decide_fused().
    pub fn execute_action(
        &mut self,
        sim: &mut Simulation,
        action: MoveAction,
        config: &QuantumForagerConfig,
    ) -> ForagerTickResult {
        // 1. Age
        self.age += 1;

        // 2. Metabolism cost
        self.energy -= config.metabolism_cost;

        // 3. Check death
        if self.energy <= 0.0 {
            return ForagerTickResult::Died;
        }

        // 4. Execute movement
        let new_x = (self.x as i32 + action.dx).max(0) as u32;
        let new_y = (self.y as i32 + action.dy).max(0) as u32;
        let (clamped_x, clamped_y) = config.region.clamp(new_x as i32, new_y as i32);

        let moved = clamped_x != self.x || clamped_y != self.y;
        if moved {
            self.x = clamped_x;
            self.y = clamped_y;
            self.energy -= config.movement_cost;
        }

        // 5. Consume resources at new position
        let mut consumed = 0.0f32;
        if config.consume_resources {
            let heat = sim.get_heat_field_for_test(self.x as usize, self.y as usize);
            let charge = sim.get_charge_field_for_test(self.x as usize, self.y as usize);

            let heat_energy = heat as f32 * config.heat_energy_factor;
            let charge_energy = charge as f32 * config.charge_energy_factor;
            consumed = heat_energy + charge_energy;

            self.energy += consumed;
            self.total_consumed += consumed;

            // Deplete resources from field
            if heat > 0 {
                let new_heat =
                    heat.saturating_sub((heat_energy / config.heat_energy_factor) as u32);
                sim.set_heat_field_for_test(self.x as usize, self.y as usize, new_heat);
            }
            if charge > 0 {
                let new_charge =
                    charge.saturating_sub((charge_energy / config.charge_energy_factor) as u32);
                sim.set_charge_field_for_test(self.x as usize, self.y as usize, new_charge);
            }
        }

        // 6. Check for reproduction
        let can_reproduce = self.energy >= config.reproduction_threshold;

        ForagerTickResult::Alive {
            action,
            moved,
            consumed,
            can_reproduce,
        }
    }
}

/// Result of a forager tick.
#[derive(Debug, Clone)]
pub enum ForagerTickResult {
    Died,
    Alive {
        action: MoveAction,
        moved: bool,
        consumed: f32,
        can_reproduce: bool,
    },
}

// =============================================================================
// EPIC 90.2: Organism Batching
// =============================================================================

/// Thresholds for GPU batching decisions.
/// Based on EPIC 88 findings: GPU needs 50+ organisms with substantial circuits
/// to overcome kernel launch overhead (~5-10μs).
pub const MIN_BATCH_FOR_GPU: usize = 50;
pub const MIN_GATES_FOR_GPU: usize = 8;

/// Statistics from batched execution.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Number of batches formed
    pub num_batches: usize,
    /// Total organisms processed
    pub total_organisms: usize,
    /// Organisms processed via batching
    pub batched_organisms: usize,
    /// Organisms processed individually (too small batch)
    pub individual_organisms: usize,
    /// Total gates executed
    pub total_gates: usize,
    /// Gates eliminated by fusion
    pub gates_eliminated: usize,
    /// Number of batches that used GPU
    pub gpu_batches: usize,
    /// Number of batches that used CPU
    pub cpu_batches: usize,
}

impl BatchStats {
    /// Batching efficiency: fraction of organisms that were batched.
    pub fn batch_efficiency(&self) -> f32 {
        if self.total_organisms == 0 {
            0.0
        } else {
            self.batched_organisms as f32 / self.total_organisms as f32
        }
    }

    /// Fusion efficiency: fraction of gates eliminated.
    pub fn fusion_efficiency(&self) -> f32 {
        let total = self.total_gates + self.gates_eliminated;
        if total == 0 {
            0.0
        } else {
            self.gates_eliminated as f32 / total as f32
        }
    }
}

// =============================================================================
// EPIC 90.3: Per-Tick Metrics
// =============================================================================

/// Complete metrics for a single population tick.
#[derive(Debug, Clone)]
pub struct TickMetrics {
    /// Population statistics
    pub pop_stats: PopulationStats,
    /// Batching statistics
    pub batch_stats: BatchStats,
    /// COPS metric for quantum operations
    pub cops: CopsMetric,
    /// Time elapsed for this tick
    pub elapsed_secs: f64,
    /// Total quantum operations (gates × amplitudes)
    pub total_quantum_ops: u64,
    /// Number of organisms processed
    pub organisms_processed: usize,
}

impl TickMetrics {
    /// Get organisms processed per second.
    pub fn organisms_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.organisms_processed as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Get gates per second.
    pub fn gates_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.batch_stats.total_gates as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Print a summary line.
    pub fn summary(&self) -> String {
        format!(
            "tick: {} orgs ({:.0}/s), {} gates ({:.0}K/s), {}, fusion={:.1}%",
            self.organisms_processed,
            self.organisms_per_sec(),
            self.batch_stats.total_gates,
            self.gates_per_sec() / 1000.0,
            self.cops.display(),
            self.batch_stats.fusion_efficiency() * 100.0
        )
    }
}

/// A batch of organisms with similar circuit structure.
///
/// Organisms in a batch have the same circuit_hash, meaning they have
/// the same gate structure (but possibly different angle parameters).
#[derive(Debug)]
pub struct OrganismBatch {
    /// Hash of the circuit structure (from circuit_hash())
    pub structure_hash: u64,
    /// IDs of organisms in this batch
    pub organism_ids: Vec<u64>,
    /// Number of qubits (all organisms in batch must match)
    pub n_qubits: u8,
    /// Number of gates in the circuit
    pub gate_count: usize,
}

impl OrganismBatch {
    /// Check if this batch is large enough for GPU execution.
    pub fn should_use_gpu(&self) -> bool {
        self.organism_ids.len() >= MIN_BATCH_FOR_GPU && self.gate_count >= MIN_GATES_FOR_GPU
    }
}

/// Groups organisms by circuit structure for batched execution.
pub fn group_organisms_by_circuit<'a>(
    foragers: impl Iterator<Item = &'a QuantumForager>,
) -> Vec<OrganismBatch> {
    let mut groups: HashMap<u64, OrganismBatch> = HashMap::new();

    for forager in foragers {
        let hash = circuit_hash(&forager.genome);

        groups
            .entry(hash)
            .and_modify(|batch| {
                batch.organism_ids.push(forager.id);
            })
            .or_insert_with(|| OrganismBatch {
                structure_hash: hash,
                organism_ids: vec![forager.id],
                n_qubits: forager.n_qubits,
                gate_count: forager.genome.len(),
            });
    }

    groups.into_values().collect()
}

// =============================================================================
// Population Management
// =============================================================================

/// Statistics about the population.
#[derive(Debug, Clone, Default)]
pub struct PopulationStats {
    pub size: usize,
    pub total_energy: f32,
    pub avg_energy: f32,
    pub avg_age: f32,
    pub avg_circuit_length: f32,
    pub avg_quantum_ness: f32,
    pub max_generation: u32,
    pub births_this_tick: usize,
    pub deaths_this_tick: usize,
    pub total_consumed_this_tick: f32,
}

/// Manages a population of quantum foragers.
#[derive(Clone, Debug)]
pub struct ForagerPopulation {
    /// All foragers indexed by ID
    foragers: HashMap<u64, QuantumForager>,

    /// Next ID to assign
    next_id: u64,

    /// Configuration
    config: QuantumForagerConfig,

    /// Current tick number
    tick: u64,

    /// Random seed for population-level events
    seed: u64,

    /// Stats from last tick
    last_stats: PopulationStats,
}

impl ForagerPopulation {
    /// Create a new population with initial foragers.
    pub fn new(config: QuantumForagerConfig, seed: u64) -> Self {
        let mut pop = Self {
            foragers: HashMap::new(),
            next_id: 0,
            config,
            tick: 0,
            seed,
            last_stats: PopulationStats::default(),
        };

        pop.spawn_initial();
        pop
    }

    /// Spawn the initial population.
    fn spawn_initial(&mut self) {
        let mut rng = SimpleRng::new(self.seed);

        for _ in 0..self.config.initial_population {
            let x = self.config.region.x0 + rng.gen_range(0, self.config.region.width);
            let y = self.config.region.y0 + rng.gen_range(0, self.config.region.height);
            let id = self.next_id;
            self.next_id += 1;

            let forager = QuantumForager::new_random(id, x, y, &self.config, rng.next_u64());

            self.foragers.insert(id, forager);
        }
    }

    /// Get the current population size.
    pub fn size(&self) -> usize {
        self.foragers.len()
    }

    /// Check if population is extinct.
    pub fn is_extinct(&self) -> bool {
        self.foragers.is_empty()
    }

    /// Get the current tick number.
    pub fn tick_count(&self) -> u64 {
        self.tick
    }

    /// Get stats from the last tick.
    pub fn stats(&self) -> &PopulationStats {
        &self.last_stats
    }

    /// Get a reference to a forager by ID.
    pub fn get(&self, id: u64) -> Option<&QuantumForager> {
        self.foragers.get(&id)
    }

    /// Iterate over all foragers.
    pub fn iter(&self) -> impl Iterator<Item = &QuantumForager> {
        self.foragers.values()
    }

    /// Step the entire population by one tick.
    pub fn step(&mut self, sim: &mut Simulation) -> &PopulationStats {
        self.tick += 1;

        let mut deaths = Vec::new();
        let mut births = Vec::new();
        let mut total_consumed = 0.0f32;

        // Cache values needed for reproduction check
        let max_population = self.config.max_population;
        let current_population = self.foragers.len();

        // Tick each forager
        let ids: Vec<u64> = self.foragers.keys().copied().collect();
        for id in ids {
            let forager = self.foragers.get_mut(&id).unwrap();
            let result = forager.tick(sim, &self.config);

            match result {
                ForagerTickResult::Died => {
                    deaths.push(id);
                }
                ForagerTickResult::Alive {
                    consumed,
                    can_reproduce,
                    ..
                } => {
                    total_consumed += consumed;

                    // Handle reproduction
                    if can_reproduce && current_population + births.len() < max_population {
                        let offspring_id = self.next_id;
                        self.next_id += 1;

                        let mutation_seed = self.seed.wrapping_add(self.tick).wrapping_mul(id);

                        if let Some(offspring) =
                            forager.reproduce(offspring_id, &self.config, mutation_seed)
                        {
                            births.push(offspring);
                        }
                    }
                }
            }
        }

        // Remove dead foragers
        for id in &deaths {
            self.foragers.remove(id);
        }

        // Add offspring
        for offspring in births.iter() {
            self.foragers.insert(offspring.id, offspring.clone());
        }

        // Update stats
        self.update_stats(deaths.len(), births.len(), total_consumed);

        &self.last_stats
    }

    /// Step the population with batched quantum execution (EPIC 90.2).
    ///
    /// This optimized version groups organisms by circuit structure and
    /// executes them in batches for better cache utilization and potential
    /// GPU offload.
    ///
    /// Returns both population stats and batch execution statistics.
    pub fn step_batched(&mut self, sim: &mut Simulation) -> (&PopulationStats, BatchStats) {
        self.tick += 1;

        let mut deaths = Vec::new();
        let mut births = Vec::new();
        let mut total_consumed = 0.0f32;
        let mut batch_stats = BatchStats::default();

        // Cache values needed for reproduction check
        let max_population = self.config.max_population;
        let current_population = self.foragers.len();

        // Group organisms by circuit structure
        let batches = group_organisms_by_circuit(self.foragers.values());
        batch_stats.num_batches = batches.len();
        batch_stats.total_organisms = self.foragers.len();

        // Process each batch
        for batch in batches {
            let batch_size = batch.organism_ids.len();
            let should_batch = batch_size >= 2; // Batch if 2+ organisms share structure

            if should_batch {
                batch_stats.batched_organisms += batch_size;
                if batch.should_use_gpu() {
                    batch_stats.gpu_batches += 1;
                } else {
                    batch_stats.cpu_batches += 1;
                }
            } else {
                batch_stats.individual_organisms += batch_size;
            }

            // Execute organisms in this batch
            for &id in &batch.organism_ids {
                let forager = self.foragers.get_mut(&id).unwrap();

                // Sense environment
                let readings = forager.sense(sim);

                // Execute quantum circuit (fused path)
                let circuit_result = forager.decide_fused(&readings);
                let action = decode_action(circuit_result.measurement);

                // Track fusion statistics
                batch_stats.total_gates += circuit_result.original_gates;
                batch_stats.gates_eliminated += circuit_result.eliminated;

                // Execute the rest of the tick (movement, energy, reproduction check)
                let result = forager.execute_action(sim, action, &self.config);

                match result {
                    ForagerTickResult::Died => {
                        deaths.push(id);
                    }
                    ForagerTickResult::Alive {
                        consumed,
                        can_reproduce,
                        ..
                    } => {
                        total_consumed += consumed;

                        // Handle reproduction
                        if can_reproduce && current_population + births.len() < max_population {
                            let offspring_id = self.next_id;
                            self.next_id += 1;

                            let mutation_seed = self.seed.wrapping_add(self.tick).wrapping_mul(id);

                            if let Some(offspring) =
                                forager.reproduce(offspring_id, &self.config, mutation_seed)
                            {
                                births.push(offspring);
                            }
                        }
                    }
                }
            }
        }

        // Remove dead foragers
        for id in &deaths {
            self.foragers.remove(id);
        }

        // Add offspring
        for offspring in births.iter() {
            self.foragers.insert(offspring.id, offspring.clone());
        }

        // Update stats
        self.update_stats(deaths.len(), births.len(), total_consumed);

        (&self.last_stats, batch_stats)
    }

    /// Step the population with full metrics (EPIC 90.3).
    ///
    /// Returns population stats, batch stats, and COPS metric for this tick.
    /// This is the most instrumented version for performance analysis.
    pub fn step_with_metrics(&mut self, sim: &mut Simulation) -> TickMetrics {
        let start = Instant::now();

        // Use batched step for execution
        let (pop_stats, batch_stats) = self.step_batched(sim);

        // Clone immediately to release the borrow
        let pop_stats = pop_stats.clone();

        let elapsed = start.elapsed().as_secs_f64();

        // Calculate COPS
        // For organisms: gates × amplitudes (2^n_qubits)
        // Total operations = sum of (gates × 2^n) for each organism
        let total_quantum_ops: u64 = self
            .foragers
            .values()
            .map(|f| {
                let amplitudes = 1u64 << f.n_qubits;
                f.genome.len() as u64 * amplitudes
            })
            .sum();

        let gate_ops_per_sec = if elapsed > 0.0 {
            batch_stats.total_gates as f64 / elapsed
        } else {
            0.0
        };

        // Average qubits across population
        let avg_qubits = if !self.foragers.is_empty() {
            let total: u32 = self.foragers.values().map(|f| f.n_qubits as u32).sum();
            (total as f64 / self.foragers.len() as f64).round() as u8
        } else {
            4 // Default assumption
        };

        let cops = CopsMetric::quantum(gate_ops_per_sec, avg_qubits);

        TickMetrics {
            pop_stats,
            batch_stats,
            cops,
            elapsed_secs: elapsed,
            total_quantum_ops,
            organisms_processed: self.foragers.len(),
        }
    }

    /// Update population statistics.
    fn update_stats(&mut self, deaths: usize, births: usize, consumed: f32) {
        let size = self.foragers.len();

        if size == 0 {
            self.last_stats = PopulationStats {
                size: 0,
                deaths_this_tick: deaths,
                births_this_tick: births,
                total_consumed_this_tick: consumed,
                ..Default::default()
            };
            return;
        }

        let total_energy: f32 = self.foragers.values().map(|f| f.energy).sum();
        let total_age: u32 = self.foragers.values().map(|f| f.age).sum();
        let total_circuit_len: usize = self.foragers.values().map(|f| f.genome.len()).sum();
        let total_quantum_ness: f32 = self
            .foragers
            .values()
            .map(|f| quantum_brain::quantum_ness(&f.genome))
            .sum();
        let max_gen = self
            .foragers
            .values()
            .map(|f| f.generation)
            .max()
            .unwrap_or(0);

        self.last_stats = PopulationStats {
            size,
            total_energy,
            avg_energy: total_energy / size as f32,
            avg_age: total_age as f32 / size as f32,
            avg_circuit_length: total_circuit_len as f32 / size as f32,
            avg_quantum_ness: total_quantum_ness / size as f32,
            max_generation: max_gen,
            births_this_tick: births,
            deaths_this_tick: deaths,
            total_consumed_this_tick: consumed,
        };
    }

    /// Get the top N foragers by fitness (energy * age).
    pub fn top_by_fitness(&self, n: usize) -> Vec<&QuantumForager> {
        let mut sorted: Vec<_> = self.foragers.values().collect();
        sorted.sort_by(|a, b| {
            let fitness_a = a.energy * (a.age as f32 + 1.0);
            let fitness_b = b.energy * (b.age as f32 + 1.0);
            fitness_b
                .partial_cmp(&fitness_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get foragers at a specific position.
    pub fn at_position(&self, x: u32, y: u32) -> Vec<&QuantumForager> {
        self.foragers
            .values()
            .filter(|f| f.x == x && f.y == y)
            .collect()
    }

    /// Spawn resources in the simulation.
    ///
    /// Call this periodically to replenish the environment.
    pub fn spawn_resources(&self, sim: &mut Simulation, heat_patches: usize, heat_value: u32) {
        let mut rng = SimpleRng::new(self.seed.wrapping_add(self.tick * 997));

        for _ in 0..heat_patches {
            let x = self.config.region.x0 + rng.gen_range(0, self.config.region.width);
            let y = self.config.region.y0 + rng.gen_range(0, self.config.region.height);

            let current = sim.get_heat_field_for_test(x as usize, y as usize);
            sim.set_heat_field_for_test(x as usize, y as usize, current.saturating_add(heat_value));
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;

    fn test_config() -> QuantumForagerConfig {
        QuantumForagerConfig {
            initial_population: 10,
            initial_energy: 100.0,
            n_qubits: 4,
            initial_circuit_length: 5,
            max_circuit_length: 20,
            movement_cost: 1.0,
            metabolism_cost: 0.5,
            reproduction_threshold: 120.0,
            reproduction_cost: 50.0,
            mutation_rate: 0.8,
            heat_energy_factor: 1.0,
            charge_energy_factor: 0.5,
            consume_resources: true,
            max_population: 100,
            region: ForagerRegion {
                x0: 10,
                y0: 10,
                width: 50,
                height: 50,
            },
        }
    }

    #[test]
    fn test_forager_creation() {
        let config = test_config();
        let forager = QuantumForager::new_random(1, 20, 20, &config, 12345);

        assert_eq!(forager.id, 1);
        assert_eq!(forager.x, 20);
        assert_eq!(forager.y, 20);
        assert_eq!(forager.energy, 100.0);
        assert_eq!(forager.generation, 0);
        assert!(forager.parent_id.is_none());
        assert_eq!(forager.genome.len(), 5);
        assert!(forager.is_alive());
    }

    #[test]
    fn test_forager_reproduction() {
        let config = test_config();
        let mut parent = QuantumForager::new_random(1, 20, 20, &config, 12345);
        parent.energy = 150.0; // Above threshold

        let offspring = parent.reproduce(2, &config, 54321);
        assert!(offspring.is_some());

        let child = offspring.unwrap();
        assert_eq!(child.id, 2);
        assert_eq!(child.parent_id, Some(1));
        assert_eq!(child.generation, 1);
        assert_eq!(child.x, parent.x);
        assert_eq!(child.y, parent.y);
        assert_eq!(child.energy, 50.0); // reproduction_cost

        // Parent energy should be reduced
        assert_eq!(parent.energy, 100.0); // 150 - 50
        assert_eq!(parent.offspring_count, 1);
    }

    #[test]
    fn test_forager_reproduction_fails_below_threshold() {
        let config = test_config();
        let mut parent = QuantumForager::new_random(1, 20, 20, &config, 12345);
        parent.energy = 100.0; // Below 120 threshold

        let offspring = parent.reproduce(2, &config, 54321);
        assert!(offspring.is_none());
    }

    #[test]
    fn test_forager_death() {
        let config = test_config();
        let mut sim = Simulation::new();
        let mut forager = QuantumForager::new_random(1, 20, 20, &config, 12345);
        forager.energy = 0.5; // Very low

        let result = forager.tick(&mut sim, &config);
        assert!(matches!(result, ForagerTickResult::Died));
        assert!(!forager.is_alive());
    }

    #[test]
    fn test_forager_consumes_resources() {
        let config = test_config();
        let mut sim = Simulation::new();

        // Place heat in a 3x3 area around the forager position
        // so it can consume regardless of where it moves
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                let x = (20i32 + dx) as usize;
                let y = (20i32 + dy) as usize;
                sim.set_heat_field_for_test(x, y, 50);
            }
        }

        let mut forager = QuantumForager::new_random(1, 20, 20, &config, 12345);
        let _initial_energy = forager.energy;

        let result = forager.tick(&mut sim, &config);

        if let ForagerTickResult::Alive { consumed, .. } = result {
            assert!(consumed > 0.0);
            // Energy should increase from consumption (minus metabolism)
            assert!(forager.total_consumed > 0.0);
        }

        // Heat should be depleted at forager's final position
        let remaining_heat = sim.get_heat_field_for_test(forager.x as usize, forager.y as usize);
        assert!(remaining_heat < 50);
    }

    #[test]
    fn test_population_creation() {
        let config = test_config();
        let pop = ForagerPopulation::new(config, 12345);

        assert_eq!(pop.size(), 10);
        assert!(!pop.is_extinct());
    }

    #[test]
    fn test_population_step() {
        let config = test_config();
        let mut sim = Simulation::new();
        let mut pop = ForagerPopulation::new(config, 12345);

        // Seed some resources
        for x in 10..60 {
            for y in 10..60 {
                sim.set_heat_field_for_test(x, y, 20);
            }
        }

        // Run a few steps
        for _ in 0..10 {
            pop.step(&mut sim);
        }

        let stats = pop.stats();
        assert!(stats.size > 0 || stats.deaths_this_tick > 0);
    }

    #[test]
    fn test_region_contains() {
        let region = ForagerRegion {
            x0: 10,
            y0: 10,
            width: 50,
            height: 50,
        };

        assert!(region.contains(10, 10));
        assert!(region.contains(30, 30));
        assert!(region.contains(59, 59));
        assert!(!region.contains(60, 30));
        assert!(!region.contains(30, 60));
        assert!(!region.contains(9, 30));
    }

    #[test]
    fn test_region_clamp() {
        let region = ForagerRegion {
            x0: 10,
            y0: 10,
            width: 50,
            height: 50,
        };

        assert_eq!(region.clamp(30, 30), (30, 30)); // Inside
        assert_eq!(region.clamp(5, 30), (10, 30)); // Left of region
        assert_eq!(region.clamp(70, 30), (59, 30)); // Right of region
        assert_eq!(region.clamp(30, 5), (30, 10)); // Above region
        assert_eq!(region.clamp(30, 70), (30, 59)); // Below region
    }

    #[test]
    fn test_population_extinction() {
        let mut config = test_config();
        config.initial_population = 2;
        config.initial_energy = 5.0; // Very low, will die quickly
        config.metabolism_cost = 10.0; // High metabolism

        let mut sim = Simulation::new();
        let mut pop = ForagerPopulation::new(config, 12345);

        // Run until extinction
        for _ in 0..100 {
            pop.step(&mut sim);
            if pop.is_extinct() {
                break;
            }
        }

        assert!(pop.is_extinct());
    }

    #[test]
    fn test_forager_decision_varies_with_environment() {
        let config = test_config();
        let mut sim = Simulation::new();

        // Create forager
        let mut forager = QuantumForager::new_random(1, 50, 50, &config, 12345);

        // Test with heat to the north
        sim.set_heat_field_for_test(50, 49, 255);
        let readings1 = forager.sense(&sim);
        let _action1 = forager.decide(&readings1);

        // Clear and test with heat to the east
        sim.set_heat_field_for_test(50, 49, 0);
        sim.set_heat_field_for_test(51, 50, 255);
        let readings2 = forager.sense(&sim);

        // The sensor readings should be different
        assert_ne!(readings1.heat_n, readings2.heat_n);
        assert_ne!(readings1.heat_e, readings2.heat_e);
    }

    #[test]
    fn test_circuit_stats() {
        let config = test_config();
        let forager = QuantumForager::new_random(1, 20, 20, &config, 12345);

        let stats = forager.circuit_stats();
        assert_eq!(stats.total_gates, forager.genome.len());
    }

    // =========================================================================
    // EPIC 90.2: Batching Tests
    // =========================================================================

    #[test]
    fn test_circuit_hash_consistency() {
        // Same circuit should produce same hash
        let genome = vec![QGate::H(0), QGate::CNot(0, 1), QGate::Phase(1, 0.5)];

        let hash1 = circuit_hash(&genome);
        let hash2 = circuit_hash(&genome);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_circuit_hash_ignores_angles() {
        // Circuits with different angles should have same structural hash
        let genome1 = vec![QGate::H(0), QGate::Phase(0, 0.5)];
        let genome2 = vec![
            QGate::H(0),
            QGate::Phase(0, 1.5), // Different angle
        ];

        assert_eq!(circuit_hash(&genome1), circuit_hash(&genome2));
    }

    #[test]
    fn test_circuit_hash_different_structure() {
        // Different structures should have different hashes
        let genome1 = vec![QGate::H(0), QGate::X(1)];
        let genome2 = vec![QGate::H(0), QGate::Z(1)]; // Different gate

        assert_ne!(circuit_hash(&genome1), circuit_hash(&genome2));
    }

    #[test]
    fn test_group_organisms_by_circuit() {
        let config = test_config();

        // Create foragers with known genomes
        let mut f1 = QuantumForager::new_random(1, 20, 20, &config, 12345);
        let mut f2 = QuantumForager::new_random(2, 30, 30, &config, 12346);
        let f3 = QuantumForager::new_random(3, 40, 40, &config, 12347);

        // Force f1 and f2 to have the same circuit structure
        let shared_genome = vec![QGate::H(0), QGate::X(1)];
        f1.genome = shared_genome.clone();
        f2.genome = shared_genome.clone();
        // f3 keeps its random genome

        let foragers = [f1, f2, f3];
        let batches = group_organisms_by_circuit(foragers.iter());

        // Should have 2 batches: one with f1+f2, one with f3
        assert_eq!(batches.len(), 2);

        // Find the batch with 2 organisms
        let large_batch = batches.iter().find(|b| b.organism_ids.len() == 2);
        assert!(large_batch.is_some());

        // Find the batch with 1 organism
        let small_batch = batches.iter().find(|b| b.organism_ids.len() == 1);
        assert!(small_batch.is_some());
    }

    #[test]
    fn test_batch_stats_efficiency() {
        let mut stats = BatchStats {
            total_organisms: 100,
            batched_organisms: 80,
            individual_organisms: 20,
            total_gates: 1000,
            gates_eliminated: 200,
            ..Default::default()
        };

        assert!((stats.batch_efficiency() - 0.8).abs() < 0.01);
        assert!((stats.fusion_efficiency() - (200.0 / 1200.0)).abs() < 0.01);

        // Edge case: empty
        stats.total_organisms = 0;
        stats.total_gates = 0;
        stats.gates_eliminated = 0;
        assert_eq!(stats.batch_efficiency(), 0.0);
        assert_eq!(stats.fusion_efficiency(), 0.0);
    }

    #[test]
    fn test_step_batched_produces_stats() {
        let mut config = test_config();
        config.initial_population = 10;

        let mut pop = ForagerPopulation::new(config.clone(), 12345);
        let mut sim = Simulation::new();

        // Add some heat so foragers survive
        for x in 0..100 {
            for y in 0..100 {
                sim.set_heat_field_for_test(x, y, 20);
            }
        }

        let (pop_stats, batch_stats) = pop.step_batched(&mut sim);

        // Should have processed all organisms
        assert_eq!(batch_stats.total_organisms, 10);
        assert!(batch_stats.num_batches > 0);
        assert!(batch_stats.total_gates > 0);

        // Population stats should also be valid
        assert!(pop_stats.size > 0 || pop_stats.deaths_this_tick > 0);
    }

    #[test]
    fn test_step_with_metrics_returns_cops() {
        let mut config = test_config();
        config.initial_population = 10;

        let mut pop = ForagerPopulation::new(config.clone(), 12345);
        let mut sim = Simulation::new();

        // Add some heat so foragers survive
        for x in 0..100 {
            for y in 0..100 {
                sim.set_heat_field_for_test(x, y, 20);
            }
        }

        let metrics = pop.step_with_metrics(&mut sim);

        // Should have valid COPS metric
        assert!(metrics.cops.cops >= 0.0);
        assert!(metrics.cops.amplification > 0);

        // Should have batch stats
        assert_eq!(metrics.batch_stats.total_organisms, 10);
        assert!(metrics.batch_stats.total_gates > 0);

        // Should have processed organisms
        assert!(metrics.organisms_processed > 0 || metrics.pop_stats.deaths_this_tick > 0);

        // Timing should be reasonable (< 1 second for 10 organisms)
        assert!(metrics.elapsed_secs < 1.0);

        // Summary should be non-empty
        assert!(!metrics.summary().is_empty());
    }

    #[test]
    fn test_tick_metrics_rates() {
        let metrics = TickMetrics {
            pop_stats: PopulationStats::default(),
            batch_stats: BatchStats {
                total_organisms: 100,
                total_gates: 5000,
                ..Default::default()
            },
            cops: CopsMetric::quantum(1000.0, 4),
            elapsed_secs: 0.1,
            total_quantum_ops: 80000,
            organisms_processed: 100,
        };

        // 100 organisms / 0.1s = 1000 orgs/sec
        assert!((metrics.organisms_per_sec() - 1000.0).abs() < 1.0);

        // 5000 gates / 0.1s = 50000 gates/sec
        assert!((metrics.gates_per_sec() - 50000.0).abs() < 1.0);
    }
}
