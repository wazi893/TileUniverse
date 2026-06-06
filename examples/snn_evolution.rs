//! CHUNGUS 4: Spiking Neural Network Evolution Demo
//!
//! Demonstrates evolving SNNs for a simple foraging task where
//! organisms must navigate toward food sources based on sensor input.
//!
//! Run with: cargo run --release --example snn_evolution

use engine::brain::spiking::SpikingBrain;
use engine::classical_brain::Move;
use engine::quantum::QRng;
use engine::quantum_brain::SensorReadings;
use engine::snn::{NeuronConfig, SNNConfig, STDPConfig};
use std::time::Instant;

/// Simple 2D environment for foraging
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

    /// Get sensor readings at a position (heat = proximity to food)
    fn get_sensors(&self, x: i32, y: i32) -> SensorReadings {
        // Calculate heat based on distance to nearest food in each direction
        let heat_n = self.food_heat(x, y - 1);
        let heat_s = self.food_heat(x, y + 1);
        let heat_e = self.food_heat(x + 1, y);
        let heat_w = self.food_heat(x - 1, y);
        let heat_local = self.food_heat(x, y);

        SensorReadings {
            heat_n,
            heat_s,
            heat_e,
            heat_w,
            heat_local,
            charge_local: 0, // Not used
        }
    }

    /// Calculate "heat" (food proximity) at a position
    fn food_heat(&self, x: i32, y: i32) -> u32 {
        let mut max_heat = 0u32;
        for &(fx, fy) in &self.food_positions {
            let dx = (x - fx).abs();
            let dy = (y - fy).abs();
            let dist = dx + dy; // Manhattan distance
            let heat = if dist == 0 {
                255
            } else {
                (255 / (dist + 1).max(1)) as u32
            };
            max_heat = max_heat.max(heat);
        }
        max_heat.min(255)
    }

    /// Check if position has food and consume it
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

    /// Wrap position within bounds
    fn wrap(&self, x: i32, y: i32) -> (i32, i32) {
        let wx = ((x % self.width) + self.width) % self.width;
        let wy = ((y % self.height) + self.height) % self.height;
        (wx, wy)
    }
}

/// An evolving organism with a spiking brain
struct Organism {
    brain: SpikingBrain,
    x: i32,
    y: i32,
    food_eaten: u32,
    steps_taken: u32,
}

impl Organism {
    fn new(brain: SpikingBrain, x: i32, y: i32) -> Self {
        Self {
            brain,
            x,
            y,
            food_eaten: 0,
            steps_taken: 0,
        }
    }

    fn step(&mut self, env: &mut Environment) {
        // Get sensor readings
        let sensors = env.get_sensors(self.x, self.y);

        // Brain decides movement
        let action = self.brain.decide(&sensors);

        // Apply movement
        let (dx, dy) = match action {
            Move::Up => (0, -1),
            Move::Down => (0, 1),
            Move::Left => (-1, 0),
            Move::Right => (1, 0),
            Move::Stay => (0, 0),
        };

        let (new_x, new_y) = env.wrap(self.x + dx, self.y + dy);
        self.x = new_x;
        self.y = new_y;
        self.steps_taken += 1;

        // Check for food
        if env.check_food(self.x, self.y) {
            self.food_eaten += 1;
            // Apply positive reward for eating
            self.brain.apply_reward(1.0);
        } else if action != Move::Stay {
            // Small negative reward for moving without eating
            self.brain.apply_reward(-0.1);
        }
    }

    fn fitness(&self) -> f32 {
        // Fitness = food eaten (primary) + efficiency bonus
        let base_fitness = self.food_eaten as f32 * 10.0;
        let efficiency = if self.steps_taken > 0 {
            self.food_eaten as f32 / self.steps_taken as f32
        } else {
            0.0
        };
        base_fitness + efficiency * 5.0
    }
}

/// Evolution parameters
struct EvolutionConfig {
    population_size: usize,
    generations: usize,
    env_width: i32,
    env_height: i32,
    food_per_env: usize,
    steps_per_eval: usize,
    mutation_rate: f32,
    elite_fraction: f32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 30,
            env_width: 20,
            env_height: 20,
            food_per_env: 15,
            steps_per_eval: 100,
            mutation_rate: 0.10, // Lower mutation for more stable evolution
            elite_fraction: 0.2, // Keep more elites
        }
    }
}

/// Run evolution
fn run_evolution(config: &EvolutionConfig) -> Vec<f32> {
    let mut rng = QRng::new(42);
    let mut best_fitness_history = Vec::new();

    println!("===========================================");
    println!("  CHUNGUS 4: SNN Evolution");
    println!("===========================================\n");

    println!("Configuration:");
    println!("  Population size: {}", config.population_size);
    println!("  Generations: {}", config.generations);
    println!("  Environment: {}x{}", config.env_width, config.env_height);
    println!("  Food per environment: {}", config.food_per_env);
    println!("  Steps per evaluation: {}", config.steps_per_eval);
    println!("  Mutation rate: {:.1}%", config.mutation_rate * 100.0);
    println!();

    // Create initial population
    println!("Creating initial population...");
    let mut population: Vec<SpikingBrain> = (0..config.population_size)
        .map(|_| SpikingBrain::random(&mut rng))
        .collect();

    let start_time = Instant::now();

    // Evolution loop
    println!(
        "\n{:>5} {:>12} {:>12} {:>12} {:>12}",
        "Gen", "Best", "Avg", "Worst", "Neurons"
    );
    println!("{:-<60}", "");

    for generation in 0..config.generations {
        // Use consistent environment seed per generation for fair comparison
        let env_seed = generation as u64 * 1000;

        // Evaluate fitness - average over multiple trials for stability
        let mut fitness_scores: Vec<(usize, f32)> = population
            .iter_mut()
            .enumerate()
            .map(|(i, brain)| {
                let mut total_fitness = 0.0;
                // Run 3 trials with consistent environments across all organisms
                for trial in 0..3 {
                    let trial_seed = env_seed + trial as u64;
                    let mut trial_rng = QRng::new(trial_seed);
                    let mut env = Environment::new(
                        config.env_width,
                        config.env_height,
                        config.food_per_env,
                        &mut trial_rng,
                    );

                    // Fixed starting position for fair comparison
                    let start_x = config.env_width / 2;
                    let start_y = config.env_height / 2;

                    let mut organism = Organism::new(brain.clone(), start_x, start_y);

                    // Run simulation
                    for _ in 0..config.steps_per_eval {
                        organism.step(&mut env);
                    }

                    total_fitness += organism.fitness();
                }
                (i, total_fitness / 3.0)
            })
            .collect();

        // Sort by fitness (descending)
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_fitness = fitness_scores[0].1;
        let worst_fitness = fitness_scores.last().unwrap().1;
        let avg_fitness: f32 =
            fitness_scores.iter().map(|(_, f)| f).sum::<f32>() / fitness_scores.len() as f32;

        best_fitness_history.push(best_fitness);

        // Get stats from best brain
        let best_idx = fitness_scores[0].0;
        let best_neurons = population[best_idx].neuron_count();

        println!(
            "{:>5} {:>12.2} {:>12.2} {:>12.2} {:>12}",
            generation, best_fitness, avg_fitness, worst_fitness, best_neurons
        );

        // Selection and reproduction
        let n_elite = (config.population_size as f32 * config.elite_fraction) as usize;
        let n_elite = n_elite.max(1);

        let mut new_population = Vec::with_capacity(config.population_size);

        // Keep elite
        for i in 0..n_elite {
            let idx = fitness_scores[i].0;
            new_population.push(population[idx].clone());
        }

        // Tournament selection and reproduction
        while new_population.len() < config.population_size {
            // Tournament selection (size 3)
            let mut tournament: Vec<(usize, f32)> = (0..3)
                .map(|_| {
                    let idx = (rng.next_f32() * fitness_scores.len() as f32) as usize;
                    fitness_scores[idx.min(fitness_scores.len() - 1)]
                })
                .collect();
            tournament.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let parent_idx = tournament[0].0;
            let child = population[parent_idx].reproduce(&mut rng, config.mutation_rate);
            new_population.push(child);
        }

        population = new_population;
    }

    let elapsed = start_time.elapsed();
    println!("{:-<60}", "");
    println!("\nEvolution completed in {:.2}s", elapsed.as_secs_f64());

    // Final statistics
    println!("\n--- Final Best Brain Stats ---");
    let best = &population[0];
    let stats = best.stats();
    println!("  Neurons: {}", stats.neurons);
    println!("  Synapses: {}", stats.synapses);
    println!("  CPUs: {}", stats.cpus);
    println!("  Recurrent: {}", stats.recurrent);
    println!("  Ticks per decision: {}", stats.ticks_per_decision);

    // Show improvement
    if best_fitness_history.len() >= 2 {
        let initial = best_fitness_history[0];
        let final_best = *best_fitness_history.last().unwrap();
        let improvement = if initial > 0.0 {
            (final_best - initial) / initial * 100.0
        } else {
            0.0
        };
        println!("\n--- Evolution Progress ---");
        println!("  Initial best: {:.2}", initial);
        println!("  Final best:   {:.2}", final_best);
        println!("  Improvement:  {:.1}%", improvement);
    }

    best_fitness_history
}

/// Compare SNN evolution with different configurations
fn compare_configurations() {
    println!("\n===========================================");
    println!("  Configuration Comparison");
    println!("===========================================\n");

    // Use architectures similar to SpikingBrain::random() for proper activity
    let configs = [
        (
            "Small SNN",
            SNNConfig {
                n_inputs: 6,
                hidden_layers: vec![16, 8], // Same as random()
                n_outputs: 5,
                connection_prob: 0.5,
                neurons_per_cpu: 32,
                recurrent: false,
                neuron_config: NeuronConfig::default(),
                stdp_config: STDPConfig::default(),
                use_reward_modulation: true,
                duplicate_inputs: false,
            },
        ),
        (
            "Medium SNN",
            SNNConfig {
                n_inputs: 6,
                hidden_layers: vec![24, 16],
                n_outputs: 5,
                connection_prob: 0.5,
                neurons_per_cpu: 32,
                recurrent: true,
                neuron_config: NeuronConfig::default(),
                stdp_config: STDPConfig::default(),
                use_reward_modulation: true,
                duplicate_inputs: false,
            },
        ),
        (
            "Large SNN",
            SNNConfig {
                n_inputs: 6,
                hidden_layers: vec![32, 24, 16],
                n_outputs: 5,
                connection_prob: 0.4,
                neurons_per_cpu: 64,
                recurrent: true,
                neuron_config: NeuronConfig::default(),
                stdp_config: STDPConfig::default(),
                use_reward_modulation: true,
                duplicate_inputs: false,
            },
        ),
    ];

    println!(
        "{:<15} {:>10} {:>10} {:>12} {:>12}",
        "Config", "Neurons", "Synapses", "Best Fit", "Time (s)"
    );
    println!("{:-<65}", "");

    for (name, snn_config) in configs {
        let mut rng = QRng::new(42);

        // Create population with this config
        let pop_size = 30;
        let generations = 20;

        let mut population: Vec<SpikingBrain> = (0..pop_size)
            .map(|_| {
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                SpikingBrain::with_config(snn_config.clone(), seed)
            })
            .collect();

        let start = Instant::now();
        let mut best_overall = 0.0f32;

        // Quick evolution
        for _ in 0..generations {
            let mut fitness_scores: Vec<(usize, f32)> = population
                .iter_mut()
                .enumerate()
                .map(|(i, brain)| {
                    // Average over multiple trials for stability
                    let mut total_fitness = 0.0;
                    for trial in 0..3 {
                        let mut env =
                            Environment::new(15, 15, 12, &mut QRng::new(trial as u64 * 1000));
                        let mut organism = Organism::new(brain.clone(), 7, 7);
                        for _ in 0..80 {
                            organism.step(&mut env);
                        }
                        total_fitness += organism.fitness();
                    }
                    (i, total_fitness / 3.0)
                })
                .collect();

            fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            best_overall = best_overall.max(fitness_scores[0].1);

            let mut new_pop = Vec::with_capacity(pop_size);
            // Keep top 3
            for i in 0..3.min(pop_size) {
                new_pop.push(population[fitness_scores[i].0].clone());
            }

            while new_pop.len() < pop_size {
                let idx = (rng.next_f32() * pop_size as f32 / 2.0) as usize;
                let parent_idx = fitness_scores[idx.min(fitness_scores.len() - 1)].0;
                new_pop.push(population[parent_idx].reproduce(&mut rng, 0.2));
            }

            population = new_pop;
        }

        let elapsed = start.elapsed();

        let neurons = population[0].neuron_count();
        let synapses = population[0].parameter_count();

        println!(
            "{:<15} {:>10} {:>10} {:>12.2} {:>12.3}",
            name,
            neurons,
            synapses,
            best_overall,
            elapsed.as_secs_f64()
        );
    }
}

/// Demonstrate R-STDP learning during lifetime
fn demo_lifetime_learning() {
    println!("\n===========================================");
    println!("  Lifetime Learning Demo (R-STDP)");
    println!("===========================================\n");

    let mut rng = QRng::new(12345);
    let mut brain = SpikingBrain::random(&mut rng);

    println!("Testing R-STDP learning over organism lifetime...\n");

    // Run multiple "lives" and track improvement
    println!("{:>5} {:>12} {:>12}", "Life", "Food Eaten", "Efficiency");
    println!("{:-<35}", "");

    for life in 0..10 {
        let mut env = Environment::new(20, 20, 20, &mut rng);
        let mut organism = Organism::new(brain.clone(), 10, 10);

        // Run for 200 steps
        for _ in 0..200 {
            organism.step(&mut env);
        }

        let efficiency = if organism.steps_taken > 0 {
            organism.food_eaten as f32 / organism.steps_taken as f32
        } else {
            0.0
        };

        println!(
            "{:>5} {:>12} {:>12.4}",
            life, organism.food_eaten, efficiency
        );

        // Use the brain from this organism for next life (with learned weights)
        brain = organism.brain;
    }

    println!("\nNote: R-STDP modifies synaptic weights based on reward signals,");
    println!("allowing the brain to improve within a single lifetime.");
}

fn main() {
    // Run main evolution
    let config = EvolutionConfig::default();
    let _history = run_evolution(&config);

    // Compare different SNN sizes
    compare_configurations();

    // Demo lifetime learning
    demo_lifetime_learning();

    println!("\n===========================================");
    println!("  SNN Evolution Demo Complete!");
    println!("===========================================");
}
