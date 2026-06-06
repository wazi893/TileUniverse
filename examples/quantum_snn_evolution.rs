//! Quantum-SNN Evolution Demo
//!
//! Compares classical SNN evolution with Quantum-SNN hybrid evolution
//! for a foraging task. The quantum hybrid uses probabilistic decision
//! making for natural exploration during evolution.
//!
//! Run with: cargo run --release --example quantum_snn_evolution

use engine::classical_brain::Move;
use engine::quantum::QRng;
use engine::quantum_brain::SensorReadings;
use engine::snn::{NeuronConfig, QuantumSNN, QuantumSNNConfig, SNNConfig, STDPConfig};
use std::time::Instant;

// ============================================================================
// Quantum Spiking Brain - wrapper for evolution compatibility
// ============================================================================

/// Quantum-enhanced spiking brain for evolution
///
/// Uses QuantumSNN internally but provides the same interface as SpikingBrain
/// for drop-in replacement in evolution scenarios.
struct QuantumSpikingBrain {
    qsnn: QuantumSNN,
    seed: u64,
    config: QuantumSNNConfig,
}

impl QuantumSpikingBrain {
    /// Create a new quantum brain with given config
    fn new(config: QuantumSNNConfig, seed: u64) -> Self {
        // Create with seed and randomize weights for variation
        let mut qsnn = QuantumSNN::with_seed(config.clone(), seed);
        qsnn.snn_mut().randomize_weights(seed.wrapping_add(12345));

        Self { qsnn, seed, config }
    }

    /// Create a random quantum brain (matching SpikingBrain::random interface)
    fn random(rng: &mut QRng) -> Self {
        let config = QuantumSNNConfig {
            n_inputs: 6,                // Same as SensorReadings fields
            hidden_layers: vec![16, 8], // Compact network
            n_outputs: 5,               // 5 moves: Up, Down, Left, Right, Stay
            interference_depth: 1,
            ticks_per_decision: 10,
            learning_enabled: true,
            ..Default::default()
        };
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;
        Self::new(config, seed)
    }

    /// Convert sensor readings to input bytes
    fn sensors_to_inputs(&self, sensors: &SensorReadings) -> Vec<u8> {
        vec![
            sensors.heat_n.min(255) as u8,
            sensors.heat_s.min(255) as u8,
            sensors.heat_e.min(255) as u8,
            sensors.heat_w.min(255) as u8,
            sensors.heat_local.min(255) as u8,
            sensors.charge_local.min(255) as u8,
        ]
    }

    /// Convert action index to Move
    fn action_to_move(&self, action: usize) -> Move {
        match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        }
    }

    /// Make a decision based on sensor readings (matching SpikingBrain interface)
    fn decide(&mut self, sensors: &SensorReadings) -> Move {
        let inputs = self.sensors_to_inputs(sensors);
        self.qsnn.set_inputs(&inputs);
        let action = self.qsnn.decide();
        self.action_to_move(action)
    }

    /// Apply reward for R-STDP learning
    fn apply_reward(&mut self, reward: f32) {
        // Convert f32 reward to i8 for SNN
        let reward_i8 = (reward * 10.0).clamp(-127.0, 127.0) as i8;
        self.qsnn.set_reward(reward_i8);
    }

    /// Clone the brain
    fn clone(&self) -> Self {
        // Create new with same config and seed
        Self::new(self.config.clone(), self.seed)
    }

    /// Reproduce with mutation
    fn reproduce(&self, rng: &mut QRng, mutation_rate: f32) -> Self {
        // Create child with mutated seed
        let seed_mutation = (rng.next_f32() * 1000.0) as u64;
        let child_seed = self.seed.wrapping_add(seed_mutation);

        let mut child = Self::new(self.config.clone(), child_seed);

        // Mutate weights
        child.qsnn.snn_mut().mutate_weights(rng, mutation_rate);

        child
    }

    /// Get neuron count
    fn neuron_count(&self) -> usize {
        self.config.n_inputs
            + self.config.hidden_layers.iter().sum::<usize>()
            + self.config.n_outputs
    }

    /// Get parameter count (approximate synapse count)
    fn parameter_count(&self) -> usize {
        // Rough estimate based on connection probability
        self.neuron_count() * 5
    }

    /// Get quantum exploration metric (entropy of last decision)
    fn exploration_entropy(&self) -> f64 {
        let probs = self.qsnn.last_probabilities();
        if probs.is_empty() {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &p in probs {
            if p > 1e-10 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Get quantum blocks used
    fn quantum_blocks(&self) -> usize {
        self.qsnn.quantum_blocks()
    }
}

// ============================================================================
// Classical Spiking Brain - simplified for comparison
// ============================================================================

/// Classical spiking brain without quantum enhancement
struct ClassicalSpikingBrain {
    snn: engine::snn::SNNNetwork,
    seed: u64,
    config: SNNConfig,
    rng_state: u64, // For fair exploration when no spikes
}

impl ClassicalSpikingBrain {
    fn new(config: SNNConfig, seed: u64) -> Self {
        let mut snn = engine::snn::SNNNetwork::new(config.clone());
        snn.build_connectivity(seed); // Build synapses first
        snn.randomize_weights(seed.wrapping_add(12345)); // Then randomize weights
        Self {
            snn,
            seed,
            config,
            rng_state: seed,
        }
    }

    fn random(rng: &mut QRng) -> Self {
        let config = SNNConfig {
            n_inputs: 6,
            hidden_layers: vec![16, 8],
            n_outputs: 5,
            connection_prob: 0.5,
            neurons_per_cpu: 32,
            recurrent: false,
            neuron_config: NeuronConfig::default(),
            stdp_config: STDPConfig::default(),
            use_reward_modulation: true,
            duplicate_inputs: false,
        };
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;
        Self::new(config, seed)
    }

    fn sensors_to_inputs(&self, sensors: &SensorReadings) -> Vec<u8> {
        vec![
            sensors.heat_n.min(255) as u8,
            sensors.heat_s.min(255) as u8,
            sensors.heat_e.min(255) as u8,
            sensors.heat_w.min(255) as u8,
            sensors.heat_local.min(255) as u8,
            sensors.charge_local.min(255) as u8,
        ]
    }

    fn action_to_move(&self, action: usize) -> Move {
        match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        }
    }

    fn decide(&mut self, sensors: &SensorReadings) -> Move {
        let inputs = self.sensors_to_inputs(sensors);
        self.snn.set_inputs(&inputs);

        // Run for same number of ticks as quantum version
        self.snn.reset_output_counts();
        for _ in 0..10 {
            self.snn.step();
        }

        // Fair comparison: use softmax exploration like quantum does
        // When all counts are zero, this becomes uniform random (like quantum superposition)
        let counts = &self.snn.output_counts;
        let total: u32 = counts.iter().sum();

        let action = if total == 0 {
            // No spikes: uniform random (matches quantum's uniform superposition)
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((self.rng_state >> 32) as usize) % 5
        } else {
            // Sample proportionally to spike counts (like quantum probability)
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let r = ((self.rng_state >> 33) as f64) / (1u64 << 31) as f64;

            let mut cumulative = 0.0;
            let mut selected = 0;
            for (i, &count) in counts.iter().enumerate() {
                cumulative += count as f64 / total as f64;
                if r < cumulative {
                    selected = i;
                    break;
                }
            }
            selected
        };

        self.action_to_move(action)
    }

    fn apply_reward(&mut self, reward: f32) {
        let reward_i8 = (reward * 10.0).clamp(-127.0, 127.0) as i8;
        self.snn.set_reward(reward_i8);
    }

    fn clone(&self) -> Self {
        Self::new(self.config.clone(), self.seed)
    }

    fn reproduce(&self, rng: &mut QRng, mutation_rate: f32) -> Self {
        let seed_mutation = (rng.next_f32() * 1000.0) as u64;
        let child_seed = self.seed.wrapping_add(seed_mutation);

        let mut child = Self::new(self.config.clone(), child_seed);
        child.snn.mutate_weights(rng, mutation_rate);
        child
    }

    fn neuron_count(&self) -> usize {
        self.config.n_inputs
            + self.config.hidden_layers.iter().sum::<usize>()
            + self.config.n_outputs
    }
}

// ============================================================================
// Environment (same as snn_evolution.rs)
// ============================================================================

struct Environment {
    width: i32,
    height: i32,
    food_positions: Vec<(i32, i32)>,
}

impl Environment {
    fn new(width: i32, height: i32, n_food: usize, rng: &mut QRng) -> Self {
        let mut food_positions = Vec::with_capacity(n_food);
        for _ in 0..n_food {
            let x = (rng.next_f32() * width as f32) as i32;
            let y = (rng.next_f32() * height as f32) as i32;
            food_positions.push((x, y));
        }
        Self {
            width,
            height,
            food_positions,
        }
    }

    fn get_sensors(&self, x: i32, y: i32) -> SensorReadings {
        SensorReadings {
            heat_n: self.food_heat(x, y - 1),
            heat_s: self.food_heat(x, y + 1),
            heat_e: self.food_heat(x + 1, y),
            heat_w: self.food_heat(x - 1, y),
            heat_local: self.food_heat(x, y),
            charge_local: 0,
        }
    }

    fn food_heat(&self, x: i32, y: i32) -> u32 {
        let mut max_heat = 0u32;
        for &(fx, fy) in &self.food_positions {
            let dx = (x - fx).abs();
            let dy = (y - fy).abs();
            let dist = dx + dy;
            let heat = if dist == 0 {
                255
            } else {
                (255 / (dist + 1).max(1)) as u32
            };
            max_heat = max_heat.max(heat);
        }
        max_heat.min(255)
    }

    fn check_food(&mut self, x: i32, y: i32) -> bool {
        if let Some(idx) = self
            .food_positions
            .iter()
            .position(|&(fx, fy)| fx == x && fy == y)
        {
            self.food_positions.remove(idx);
            true
        } else {
            false
        }
    }

    fn wrap(&self, x: i32, y: i32) -> (i32, i32) {
        let wx = ((x % self.width) + self.width) % self.width;
        let wy = ((y % self.height) + self.height) % self.height;
        (wx, wy)
    }
}

// ============================================================================
// Evolution runner
// ============================================================================

/// Run evolution with quantum brains
fn run_quantum_evolution(
    population_size: usize,
    generations: usize,
    env_width: i32,
    env_height: i32,
    food_per_env: usize,
    steps_per_eval: usize,
    mutation_rate: f32,
) -> (Vec<f32>, f64) {
    let mut rng = QRng::new(42);
    let mut best_history = Vec::new();
    let mut total_entropy = 0.0;
    let mut entropy_count = 0;

    // Create initial population
    let mut population: Vec<QuantumSpikingBrain> = (0..population_size)
        .map(|_| QuantumSpikingBrain::random(&mut rng))
        .collect();

    for generation in 0..generations {
        let env_seed = generation as u64 * 1000;

        // Evaluate fitness
        let mut fitness_scores: Vec<(usize, f32)> = population
            .iter_mut()
            .enumerate()
            .map(|(i, brain)| {
                let mut total_fitness = 0.0;
                for trial in 0..3 {
                    let mut trial_rng = QRng::new(env_seed + trial as u64);
                    let mut env =
                        Environment::new(env_width, env_height, food_per_env, &mut trial_rng);

                    let (mut x, mut y) = (env_width / 2, env_height / 2);
                    let mut food_eaten = 0u32;
                    let mut steps = 0u32;

                    for _ in 0..steps_per_eval {
                        let sensors = env.get_sensors(x, y);
                        let action = brain.decide(&sensors);

                        // Track exploration entropy
                        total_entropy += brain.exploration_entropy();
                        entropy_count += 1;

                        let (dx, dy) = match action {
                            Move::Up => (0, -1),
                            Move::Down => (0, 1),
                            Move::Left => (-1, 0),
                            Move::Right => (1, 0),
                            Move::Stay => (0, 0),
                        };

                        let (new_x, new_y) = env.wrap(x + dx, y + dy);
                        x = new_x;
                        y = new_y;
                        steps += 1;

                        if env.check_food(x, y) {
                            food_eaten += 1;
                            brain.apply_reward(1.0);
                        } else if action != Move::Stay {
                            brain.apply_reward(-0.1);
                        }
                    }

                    let fitness =
                        food_eaten as f32 * 10.0 + (food_eaten as f32 / steps.max(1) as f32) * 5.0;
                    total_fitness += fitness;
                }
                (i, total_fitness / 3.0)
            })
            .collect();

        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        best_history.push(fitness_scores[0].1);

        // Selection and reproduction
        let n_elite = (population_size as f32 * 0.2) as usize;
        let n_elite = n_elite.max(1);

        let mut new_population = Vec::with_capacity(population_size);

        // Keep elite
        for i in 0..n_elite {
            let idx = fitness_scores[i].0;
            new_population.push(population[idx].clone());
        }

        // Reproduce
        while new_population.len() < population_size {
            let mut tournament: Vec<(usize, f32)> = (0..3)
                .map(|_| {
                    let idx = (rng.next_f32() * fitness_scores.len() as f32) as usize;
                    fitness_scores[idx.min(fitness_scores.len() - 1)]
                })
                .collect();
            tournament.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let parent_idx = tournament[0].0;
            let child = population[parent_idx].reproduce(&mut rng, mutation_rate);
            new_population.push(child);
        }

        population = new_population;
    }

    let avg_entropy = if entropy_count > 0 {
        total_entropy / entropy_count as f64
    } else {
        0.0
    };

    (best_history, avg_entropy)
}

/// Run evolution with classical brains
fn run_classical_evolution(
    population_size: usize,
    generations: usize,
    env_width: i32,
    env_height: i32,
    food_per_env: usize,
    steps_per_eval: usize,
    mutation_rate: f32,
) -> Vec<f32> {
    let mut rng = QRng::new(42);
    let mut best_history = Vec::new();

    let mut population: Vec<ClassicalSpikingBrain> = (0..population_size)
        .map(|_| ClassicalSpikingBrain::random(&mut rng))
        .collect();

    for generation in 0..generations {
        let env_seed = generation as u64 * 1000;

        let mut fitness_scores: Vec<(usize, f32)> = population
            .iter_mut()
            .enumerate()
            .map(|(i, brain)| {
                let mut total_fitness = 0.0;
                for trial in 0..3 {
                    let mut trial_rng = QRng::new(env_seed + trial as u64);
                    let mut env =
                        Environment::new(env_width, env_height, food_per_env, &mut trial_rng);

                    let (mut x, mut y) = (env_width / 2, env_height / 2);
                    let mut food_eaten = 0u32;
                    let mut steps = 0u32;

                    for _ in 0..steps_per_eval {
                        let sensors = env.get_sensors(x, y);
                        let action = brain.decide(&sensors);

                        let (dx, dy) = match action {
                            Move::Up => (0, -1),
                            Move::Down => (0, 1),
                            Move::Left => (-1, 0),
                            Move::Right => (1, 0),
                            Move::Stay => (0, 0),
                        };

                        let (new_x, new_y) = env.wrap(x + dx, y + dy);
                        x = new_x;
                        y = new_y;
                        steps += 1;

                        if env.check_food(x, y) {
                            food_eaten += 1;
                            brain.apply_reward(1.0);
                        } else if action != Move::Stay {
                            brain.apply_reward(-0.1);
                        }
                    }

                    let fitness =
                        food_eaten as f32 * 10.0 + (food_eaten as f32 / steps.max(1) as f32) * 5.0;
                    total_fitness += fitness;
                }
                (i, total_fitness / 3.0)
            })
            .collect();

        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        best_history.push(fitness_scores[0].1);

        let n_elite = (population_size as f32 * 0.2) as usize;
        let n_elite = n_elite.max(1);

        let mut new_population = Vec::with_capacity(population_size);

        for i in 0..n_elite {
            let idx = fitness_scores[i].0;
            new_population.push(population[idx].clone());
        }

        while new_population.len() < population_size {
            let mut tournament: Vec<(usize, f32)> = (0..3)
                .map(|_| {
                    let idx = (rng.next_f32() * fitness_scores.len() as f32) as usize;
                    fitness_scores[idx.min(fitness_scores.len() - 1)]
                })
                .collect();
            tournament.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let parent_idx = tournament[0].0;
            let child = population[parent_idx].reproduce(&mut rng, mutation_rate);
            new_population.push(child);
        }

        population = new_population;
    }

    best_history
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("================================================================");
    println!("  QUANTUM-SNN EVOLUTION COMPARISON");
    println!("  Quantum vs Classical Decision Making in Evolution");
    println!("================================================================\n");

    let population_size = 30;
    let generations = 25;
    let env_width = 20;
    let env_height = 20;
    let food_per_env = 15;
    let steps_per_eval = 80;
    let mutation_rate = 0.15;

    println!("Configuration:");
    println!("  Population: {}", population_size);
    println!("  Generations: {}", generations);
    println!(
        "  Environment: {}x{} with {} food",
        env_width, env_height, food_per_env
    );
    println!("  Steps per eval: {}", steps_per_eval);
    println!("  Mutation rate: {:.0}%\n", mutation_rate * 100.0);

    // Run quantum evolution
    println!("--- Running Quantum-SNN Evolution ---");
    let start_q = Instant::now();
    let (quantum_history, quantum_entropy) = run_quantum_evolution(
        population_size,
        generations,
        env_width,
        env_height,
        food_per_env,
        steps_per_eval,
        mutation_rate,
    );
    let elapsed_q = start_q.elapsed();

    // Run classical evolution
    println!("--- Running Classical SNN Evolution ---");
    let start_c = Instant::now();
    let classical_history = run_classical_evolution(
        population_size,
        generations,
        env_width,
        env_height,
        food_per_env,
        steps_per_eval,
        mutation_rate,
    );
    let elapsed_c = start_c.elapsed();

    // Display results
    println!("\n--- Evolution Progress ---");
    println!("{:>5} {:>15} {:>15}", "Gen", "Quantum", "Classical");
    println!("{:-<40}", "");

    for (i, (q, c)) in quantum_history
        .iter()
        .zip(classical_history.iter())
        .enumerate()
    {
        if i % 5 == 0 || i == generations - 1 {
            println!("{:>5} {:>15.2} {:>15.2}", i, q, c);
        }
    }

    // Final comparison
    println!("\n--- Final Results ---\n");

    let q_initial = quantum_history.first().copied().unwrap_or(0.0);
    let q_final = quantum_history.last().copied().unwrap_or(0.0);
    let q_improvement = if q_initial > 0.0 {
        (q_final - q_initial) / q_initial * 100.0
    } else {
        0.0
    };

    let c_initial = classical_history.first().copied().unwrap_or(0.0);
    let c_final = classical_history.last().copied().unwrap_or(0.0);
    let c_improvement = if c_initial > 0.0 {
        (c_final - c_initial) / c_initial * 100.0
    } else {
        0.0
    };

    println!("Quantum-SNN:");
    println!("  Initial fitness: {:.2}", q_initial);
    println!("  Final fitness:   {:.2}", q_final);
    println!("  Improvement:     {:.1}%", q_improvement);
    println!("  Avg exploration: {:.3} bits", quantum_entropy);
    println!("  Time:            {:.2}s", elapsed_q.as_secs_f64());

    println!("\nClassical SNN:");
    println!("  Initial fitness: {:.2}", c_initial);
    println!("  Final fitness:   {:.2}", c_final);
    println!("  Improvement:     {:.1}%", c_improvement);
    println!("  Exploration:     softmax sampling (comparable to quantum)");
    println!("  Time:            {:.2}s", elapsed_c.as_secs_f64());

    // Advantage analysis
    println!("\n--- Quantum Advantage Analysis ---\n");

    let fitness_advantage = q_final - c_final;
    let improvement_advantage = q_improvement - c_improvement;

    if fitness_advantage > 0.0 {
        println!("  Quantum WINS by {:.2} fitness points", fitness_advantage);
        println!("  ({:.1}% better improvement)", improvement_advantage);
    } else if fitness_advantage < 0.0 {
        println!(
            "  Classical WINS by {:.2} fitness points",
            -fitness_advantage
        );
        println!("  ({:.1}% better improvement)", -improvement_advantage);
    } else {
        println!("  TIE - both methods achieved same fitness");
    }

    println!(
        "\n  Quantum exploration entropy: {:.3} bits",
        quantum_entropy
    );
    println!("  Note: Both methods now use probabilistic exploration.");
    println!("  Quantum uses interference patterns on spike amplitudes.");
    println!("  Classical uses softmax sampling on spike counts.");

    println!("\n================================================================");
    println!("  EVOLUTION COMPARISON COMPLETE");
    println!("================================================================");
}
