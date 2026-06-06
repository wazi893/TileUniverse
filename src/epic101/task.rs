//! Task trait definition for EPIC 101 experiments

use super::organism::Organism;
use super::world::World;

/// Common configuration for all tasks
#[derive(Clone, Debug)]
pub struct TaskConfig {
    /// Grid size (square)
    pub grid_size: usize,
    /// Number of organisms
    pub organism_count: usize,
    /// Ticks per generation
    pub ticks_per_generation: usize,
    /// Number of generations to run
    pub generations: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            grid_size: 256,
            organism_count: 200,
            ticks_per_generation: 2000,
            generations: 50,
            seed: 42,
        }
    }
}

/// Trait defining a task for evolution experiments
pub trait Task: Send + Sync {
    /// Task name for logging
    fn name(&self) -> &str;

    /// Number of sensor inputs this task provides
    fn sensor_count(&self) -> usize;

    /// Number of action outputs expected
    fn action_count(&self) -> usize {
        5
    } // Default: 5 actions (N, S, E, W, Stay)

    /// Initialize the world for this task
    fn setup_world(&self, world: &mut World, seed: u64);

    /// Update world state each tick (e.g., move food, update timers)
    fn tick_world(&mut self, world: &mut World, tick: usize);

    /// Extract sensor values for an organism
    /// Returns a vector of sensor values in [-1, 1] range
    fn get_sensors(&self, organism: &Organism, world: &World) -> Vec<f32>;

    /// Apply an action to an organism
    /// action_idx: 0=North, 1=South, 2=East, 3=West, 4=Stay
    fn apply_action(&self, organism: &mut Organism, action_idx: usize, world: &mut World);

    /// Calculate fitness for an organism at end of generation
    fn calculate_fitness(&self, organism: &Organism) -> f32;

    /// Check if organism is still alive
    fn is_alive(&self, organism: &Organism) -> bool {
        organism.energy > 0.0
    }

    /// Reset task state for new generation (keep learned patterns)
    fn reset_for_generation(&mut self, world: &mut World, seed: u64);
}

/// Direction enum for movement
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
    Stay,
}

impl Direction {
    pub fn from_action(action: usize) -> Self {
        match action {
            0 => Direction::North,
            1 => Direction::South,
            2 => Direction::East,
            3 => Direction::West,
            _ => Direction::Stay,
        }
    }

    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
            Direction::Stay => (0, 0),
        }
    }
}
