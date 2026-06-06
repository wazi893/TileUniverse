//! Evolution framework for EPIC 101 experiments

use super::organism::Organism;
use super::task::Task;
use super::world::World;

/// Statistics for one generation
#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub generation: usize,
    pub mean_fitness: f32,
    pub max_fitness: f32,
    pub min_fitness: f32,
    pub std_fitness: f32,
    pub survival_rate: f32,
    pub total_food_consumed: f32,
    pub mean_ticks_survived: f32,
}

/// Configuration for evolution
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    pub organism_count: usize,
    pub ticks_per_generation: usize,
    pub generations: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub elite_count: usize,
    pub tournament_size: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            organism_count: 200,
            ticks_per_generation: 2000,
            generations: 50,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: 10,
            tournament_size: 5,
        }
    }
}

/// Evolution experiment runner
///
/// This is a simplified evolution framework that:
/// 1. Runs organisms through a task
/// 2. Evaluates fitness
/// 3. Selects and reproduces
/// 4. Mutates offspring
pub struct Evolution {
    config: EvolutionConfig,
    world: World,
    organisms: Vec<Organism>,
    generation: usize,
    rng_state: u64,
}

impl Evolution {
    pub fn new(config: EvolutionConfig, grid_size: usize, seed: u64) -> Self {
        let mut evo = Self {
            config,
            world: World::new(grid_size),
            organisms: Vec::new(),
            generation: 0,
            rng_state: seed,
        };
        evo.init_population(grid_size as f32);
        evo
    }

    /// Simple PRNG for reproducibility
    fn next_rand(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    fn next_rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_rand() * (max - min)
    }

    /// Initialize population with random positions
    fn init_population(&mut self, grid_size: f32) {
        self.organisms.clear();
        for i in 0..self.config.organism_count {
            let x = self.next_rand_range(10.0, grid_size - 10.0);
            let y = self.next_rand_range(10.0, grid_size - 10.0);
            self.organisms.push(Organism::new(i, x, y));
        }
    }

    /// Reset organisms for new generation
    fn reset_organisms(&mut self, grid_size: f32) {
        // Pre-generate random positions to avoid borrow issues
        let positions: Vec<(f32, f32)> = (0..self.organisms.len())
            .map(|_| {
                let x = self.next_rand_range(10.0, grid_size - 10.0);
                let y = self.next_rand_range(10.0, grid_size - 10.0);
                (x, y)
            })
            .collect();

        for (org, (x, y)) in self.organisms.iter_mut().zip(positions) {
            org.reset(x, y, 100.0);
        }
    }

    /// Run one generation with given brain decisions
    ///
    /// `decide_fn` takes (organism_idx, sensors) and returns action index
    pub fn run_generation<F>(&mut self, task: &mut dyn Task, mut decide_fn: F) -> GenerationStats
    where
        F: FnMut(usize, &[f32]) -> usize,
    {
        // Setup world for this generation
        let seed = self.rng_state;
        task.reset_for_generation(&mut self.world, seed);

        // Reset organisms
        let grid_size = self.world.width;
        self.reset_organisms(grid_size);

        // Run simulation
        for tick in 0..self.config.ticks_per_generation {
            task.tick_world(&mut self.world, tick);

            // Process each organism
            for org_idx in 0..self.organisms.len() {
                let organism = &self.organisms[org_idx];
                if !task.is_alive(organism) {
                    continue;
                }

                // Get sensors
                let sensors = task.get_sensors(organism, &self.world);

                // Get action from brain
                let action = decide_fn(org_idx, &sensors);

                // Apply action
                let organism = &mut self.organisms[org_idx];
                task.apply_action(organism, action, &mut self.world);
            }
        }

        // Calculate fitness
        for org in &mut self.organisms {
            org.fitness = task.calculate_fitness(org);
        }

        // Compute statistics
        let stats = self.compute_stats();
        self.generation += 1;

        stats
    }

    /// Compute generation statistics
    fn compute_stats(&self) -> GenerationStats {
        let fitnesses: Vec<f32> = self.organisms.iter().map(|o| o.fitness).collect();
        let n = fitnesses.len() as f32;

        let sum: f32 = fitnesses.iter().sum();
        let mean = sum / n;

        let variance: f32 = fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let max = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);

        let alive_count = self.organisms.iter().filter(|o| o.is_alive()).count();
        let survival_rate = alive_count as f32 / n;

        let total_food: f32 = self.organisms.iter().map(|o| o.state.food_consumed).sum();
        let mean_ticks: f32 = self
            .organisms
            .iter()
            .map(|o| o.state.ticks_survived as f32)
            .sum::<f32>()
            / n;

        GenerationStats {
            generation: self.generation,
            mean_fitness: mean,
            max_fitness: max,
            min_fitness: min,
            std_fitness: std,
            survival_rate,
            total_food_consumed: total_food,
            mean_ticks_survived: mean_ticks,
        }
    }

    /// Get current generation number
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get organisms for inspection
    pub fn organisms(&self) -> &[Organism] {
        &self.organisms
    }

    /// Get world for inspection
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Get mutable world reference
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epic101::StationaryTask;
    use crate::epic101::stationary::StationaryConfig;

    #[test]
    fn test_evolution_basic() {
        let evo_config = EvolutionConfig {
            organism_count: 20,
            ticks_per_generation: 100,
            generations: 2,
            ..Default::default()
        };

        let task_config = StationaryConfig {
            grid_size: 64,
            food_patch_count: 5,
            ..Default::default()
        };

        let mut task = StationaryTask::new(task_config);
        let mut evo = Evolution::new(evo_config, 64, 42);

        // Random actions for testing
        let mut rng = 12345u64;
        let stats = evo.run_generation(&mut task, |_, _| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng % 5) as usize
        });

        assert!(stats.mean_fitness >= 0.0);
        assert!(stats.survival_rate >= 0.0 && stats.survival_rate <= 1.0);
    }
}
