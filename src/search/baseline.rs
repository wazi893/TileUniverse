//! Classical genetic algorithm baseline for comparison.
//!
//! Provides a standard GA implementation to benchmark against
//! quantum-enhanced variants.

use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::candidate::{Candidate, Rng, Xorshift64};
use super::fitness::FitnessFunction;
use super::selection::{SelectionStrategy, Selector, create_selector};
use super::substrate::{SearchSubstrate, SubstrateStats};
use super::telemetry::{GenerationTelemetry, TelemetryCollector};

/// Configuration for the classical GA baseline
#[derive(Clone, Debug)]
pub struct BaselineConfig {
    /// Population size
    pub population_size: usize,
    /// Maximum generations
    pub max_generations: usize,
    /// Selection strategy
    pub selection: SelectionStrategy,
    /// Crossover probability (0.0-1.0)
    pub crossover_rate: f32,
    /// Mutation probability per bit
    pub mutation_rate: f32,
    /// Elite count (preserved without modification)
    pub elite_count: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Stop if optimal fitness reached
    pub stop_on_optimal: bool,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            population_size: 1_000_000,
            max_generations: 1000,
            selection: SelectionStrategy::Tournament { tournament_size: 7 },
            crossover_rate: 0.9,
            mutation_rate: 0.01,
            elite_count: 10,
            seed: 42,
            stop_on_optimal: true,
        }
    }
}

impl BaselineConfig {
    /// Create config with specified population size
    pub fn with_population(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set maximum generations
    pub fn with_max_generations(mut self, max: usize) -> Self {
        self.max_generations = max;
        self
    }

    /// Set selection strategy
    pub fn with_selection(mut self, strategy: SelectionStrategy) -> Self {
        self.selection = strategy;
        self
    }

    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f32) -> Self {
        assert!((0.0..=1.0).contains(&rate), "Crossover rate must be [0, 1]");
        self.crossover_rate = rate;
        self
    }

    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f32) -> Self {
        assert!((0.0..=1.0).contains(&rate), "Mutation rate must be [0, 1]");
        self.mutation_rate = rate;
        self
    }

    /// Set elite count
    pub fn with_elite_count(mut self, count: usize) -> Self {
        self.elite_count = count;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set whether to stop on optimal
    pub fn with_stop_on_optimal(mut self, stop: bool) -> Self {
        self.stop_on_optimal = stop;
        self
    }
}

/// Result of a baseline GA run
#[derive(Clone, Debug)]
pub struct BaselineResult {
    /// Best candidate found
    pub best: Candidate,
    /// Generation where best was found
    pub best_generation: u64,
    /// Total generations run
    pub generations: u64,
    /// Whether optimal was reached
    pub optimal_reached: bool,
    /// Total runtime
    pub runtime: Duration,
    /// Evaluations per second
    pub evals_per_second: f64,
    /// Final population statistics
    pub final_stats: SubstrateStats,
    /// Telemetry data for all generations
    pub telemetry: Vec<GenerationTelemetry>,
}

/// Classical genetic algorithm baseline engine
pub struct ClassicalGA {
    config: BaselineConfig,
    substrate: SearchSubstrate,
    selector: Box<dyn Selector>,
    telemetry: TelemetryCollector,
    rng: Xorshift64,
}

impl ClassicalGA {
    /// Create a new classical GA with the given fitness function
    pub fn new(config: BaselineConfig, fitness_fn: impl FitnessFunction + 'static) -> Self {
        let mut substrate = SearchSubstrate::new(config.population_size, fitness_fn);
        substrate.initialize_random(config.seed);

        let selector = create_selector(&config.selection);
        let telemetry = TelemetryCollector::new();
        let rng = Xorshift64::new(config.seed.wrapping_add(0x9E3779B97F4A7C15));

        Self {
            config,
            substrate,
            selector,
            telemetry,
            rng,
        }
    }

    /// Run the GA until termination
    pub fn run(&mut self) -> BaselineResult {
        let start = Instant::now();

        // Initial evaluation
        self.substrate.evaluate_all();

        let mut best = *self.substrate.best_candidate();
        let mut best_generation = 0u64;
        let mut optimal_reached = false;

        for gen_idx in 0..self.config.max_generations {
            let gen_start = Instant::now();

            // Record pre-generation telemetry
            let stats = self.substrate.statistics();

            // Check for improvement
            let current_best = self.substrate.best_candidate();
            if current_best.fitness > best.fitness {
                best = *current_best;
                best_generation = gen_idx as u64;
            }

            // Check termination
            if self.config.stop_on_optimal && self.substrate.is_solved() {
                optimal_reached = true;
                self.telemetry.record(GenerationTelemetry {
                    generation: gen_idx as u64,
                    best_fitness: stats.best_fitness,
                    mean_fitness: stats.mean_fitness,
                    std_fitness: stats.std_fitness,
                    diversity: self.substrate.compute_diversity(100),
                    eval_time: Duration::ZERO,
                    generation_time: gen_start.elapsed(),
                    selection_pressure: stats.selection_pressure(),
                    improvements: 0,
                });
                break;
            }

            // Evolve one generation
            let (eval_time, improvements) = self.evolve_generation();

            // Record telemetry
            self.telemetry.record(GenerationTelemetry {
                generation: gen_idx as u64,
                best_fitness: stats.best_fitness,
                mean_fitness: stats.mean_fitness,
                std_fitness: stats.std_fitness,
                diversity: self.substrate.compute_diversity(100),
                eval_time,
                generation_time: gen_start.elapsed(),
                selection_pressure: stats.selection_pressure(),
                improvements,
            });

            self.substrate.next_generation();
        }

        let runtime = start.elapsed();
        let total_evals = (self.telemetry.len() as u64 + 1) * self.config.population_size as u64;
        let evals_per_second = total_evals as f64 / runtime.as_secs_f64();

        BaselineResult {
            best,
            best_generation,
            generations: self.telemetry.len() as u64,
            optimal_reached,
            runtime,
            evals_per_second,
            final_stats: self.substrate.statistics(),
            telemetry: self.telemetry.all().to_vec(),
        }
    }

    /// Evolve one generation
    fn evolve_generation(&mut self) -> (Duration, u32) {
        let n = self.substrate.len();
        let elite_count = self.config.elite_count.min(n);

        // Get elite indices (best individuals)
        let mut elite_indices: Vec<usize> = (0..n).collect();
        elite_indices.sort_unstable_by(|&a, &b| {
            self.substrate
                .get(b)
                .fitness
                .cmp(&self.substrate.get(a).fitness)
        });
        let elite_indices: Vec<usize> = elite_indices.into_iter().take(elite_count).collect();

        // Select parents for rest of population
        let offspring_count = n - elite_count;
        let parent_indices =
            self.selector
                .select(&self.substrate, offspring_count * 2, &mut self.rng);

        // Create offspring via crossover and mutation
        let crossover_rate = self.config.crossover_rate;
        let mutation_rate = self.config.mutation_rate;
        let candidates = self.substrate.candidates();

        // Prepare parent pairs for parallel processing
        let parent_pairs: Vec<(usize, usize)> = (0..offspring_count)
            .map(|i| (parent_indices[i * 2], parent_indices[i * 2 + 1]))
            .collect();

        // Generate offspring in parallel
        let chunk_size = (offspring_count / rayon::current_num_threads()).max(1000);
        let offspring: Vec<Candidate> = parent_pairs
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let chunk_seed = self
                    .config
                    .seed
                    .wrapping_add(self.substrate.generation())
                    .wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let mut rng = Xorshift64::new(chunk_seed);

                chunk
                    .iter()
                    .map(|&(p1_idx, p2_idx)| {
                        let p1 = &candidates[p1_idx];
                        let p2 = &candidates[p2_idx];

                        // Crossover
                        let mut child = if rng.next_f32() < crossover_rate {
                            let point = (rng.next_u64() as usize) % 64;
                            let (c1, _) = p1.crossover_single_point(p2, point);
                            c1
                        } else {
                            *p1
                        };

                        // Mutation
                        child.mutate(mutation_rate, &mut rng);

                        child
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Build new population: elites + offspring
        let mut new_population = Vec::with_capacity(n);

        // Add elites (unchanged)
        for &idx in &elite_indices {
            let mut elite = *self.substrate.get(idx);
            elite.flags.set_elite(true);
            new_population.push(elite);
        }

        // Add offspring
        new_population.extend(offspring);

        // Replace population
        self.substrate.replace_all(new_population);

        // Evaluate new population
        let eval_start = Instant::now();
        self.substrate.evaluate_all();
        let eval_time = eval_start.elapsed();

        // Count improvements (offspring better than population mean before)
        let new_stats = self.substrate.statistics();
        let improvements = self
            .substrate
            .candidates()
            .iter()
            .skip(elite_count)
            .filter(|c| c.fitness as f32 > new_stats.mean_fitness)
            .count() as u32;

        (eval_time, improvements)
    }

    /// Get current generation statistics
    pub fn statistics(&self) -> SubstrateStats {
        self.substrate.statistics()
    }

    /// Get the substrate for inspection
    pub fn substrate(&self) -> &SearchSubstrate {
        &self.substrate
    }
}

// Re-export selection types for convenience
pub use super::selection::{
    RouletteSelector as RouletteSelection, TournamentSelector as TournamentSelection,
    TruncationSelector as TruncationSelection,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::OneMax;

    #[test]
    fn test_baseline_config() {
        let config = BaselineConfig::default()
            .with_population(10000)
            .with_max_generations(50)
            .with_crossover_rate(0.8)
            .with_mutation_rate(0.02)
            .with_elite_count(5);

        assert_eq!(config.population_size, 10000);
        assert_eq!(config.max_generations, 50);
        assert!((config.crossover_rate - 0.8).abs() < 0.001);
        assert!((config.mutation_rate - 0.02).abs() < 0.001);
        assert_eq!(config.elite_count, 5);
    }

    #[test]
    fn test_classical_ga_small() {
        let config = BaselineConfig::default()
            .with_population(100)
            .with_max_generations(10)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::new(8));
        let result = ga.run();

        assert!(result.best.fitness > 0);
        assert!(result.generations <= 10);
        assert!(!result.telemetry.is_empty());
    }

    #[test]
    fn test_classical_ga_converges() {
        // Small population, easy problem
        let config = BaselineConfig::default()
            .with_population(1000)
            .with_max_generations(100)
            .with_mutation_rate(0.05)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::new(16));
        let result = ga.run();

        // Should find near-optimal solution
        assert!(
            result.best.fitness >= 200,
            "Expected fitness >= 200, got {}",
            result.best.fitness
        );
    }

    #[test]
    fn test_selection_strategies() {
        let strategies = vec![
            SelectionStrategy::Tournament { tournament_size: 5 },
            SelectionStrategy::Roulette,
            SelectionStrategy::Truncation { survival_rate: 0.2 },
            SelectionStrategy::Rank { pressure: 1.5 },
            SelectionStrategy::SUS,
        ];

        for strategy in strategies {
            let config = BaselineConfig::default()
                .with_population(100)
                .with_max_generations(5)
                .with_selection(strategy.clone())
                .with_seed(123);

            let mut ga = ClassicalGA::new(config, OneMax::new(8));
            let result = ga.run();

            assert!(
                result.best.fitness > 0,
                "Strategy {:?} produced zero fitness",
                strategy.name()
            );
        }
    }

    #[test]
    fn test_elitism() {
        let config = BaselineConfig::default()
            .with_population(100)
            .with_max_generations(10)
            .with_elite_count(5)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::new(8));

        // Run one generation and check elites are preserved
        ga.substrate.evaluate_all();
        let best_before = ga.substrate.best_candidate().fitness;

        let _ = ga.run();

        // Best should never decrease due to elitism
        assert!(ga.substrate.best_candidate().fitness >= best_before);
    }

    #[test]
    fn test_telemetry_recording() {
        let config = BaselineConfig::default()
            .with_population(100)
            .with_max_generations(10)
            .with_stop_on_optimal(false)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::new(8));
        let result = ga.run();

        // Should have telemetry for each generation
        assert_eq!(result.telemetry.len(), 10);

        // Telemetry should be ordered by generation
        for (i, t) in result.telemetry.iter().enumerate() {
            assert_eq!(t.generation, i as u64);
        }
    }

    #[test]
    fn test_1m_scale_baseline() {
        // Test at 1M scale (Phase 1 gate check target)
        let config = BaselineConfig::default()
            .with_population(1_000_000)
            .with_max_generations(5)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::standard());

        let start = std::time::Instant::now();
        let result = ga.run();
        let duration = start.elapsed();

        println!(
            "1M baseline: {} generations in {:?}",
            result.generations, duration
        );
        println!("Evals/sec: {:.2e}", result.evals_per_second);
        println!("Best fitness: {}", result.best.fitness);

        // Should complete in reasonable time
        assert!(
            duration.as_secs() < 30,
            "1M baseline took too long: {:?}",
            duration
        );
    }
}
