//! Task 1: Stationary Foraging (Control)
//!
//! Food doesn't move. No memory needed.
//! Expected: Simple NN ≈ LSTM (no memory advantage)

use super::organism::Organism;
use super::task::{Direction, Task};
use super::world::World;

/// Configuration for Stationary Foraging task
#[derive(Clone, Debug)]
pub struct StationaryConfig {
    pub grid_size: usize,
    pub food_patch_count: usize,
    pub food_radius: f32,
    pub food_value: f32,
    pub sensor_range: f32,
}

impl Default for StationaryConfig {
    fn default() -> Self {
        Self {
            grid_size: 256,
            food_patch_count: 30,
            food_radius: 15.0,
            food_value: 10.0,
            sensor_range: 100.0,
        }
    }
}

/// Task 1: Stationary Foraging
///
/// Simple foraging task where food patches don't move.
/// This serves as a control - LSTM should NOT have an advantage here.
pub struct StationaryTask {
    config: StationaryConfig,
}

impl StationaryTask {
    pub fn new(config: StationaryConfig) -> Self {
        Self { config }
    }
}

impl Task for StationaryTask {
    fn name(&self) -> &str {
        "Task1_Stationary"
    }

    fn sensor_count(&self) -> usize {
        8 // 4 directions × (density + distance)
    }

    fn setup_world(&self, world: &mut World, seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        world.clear();

        // Simple seeded random placement
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let next_rand = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) as f32) / (u32::MAX as f32)
        };

        // Place food patches randomly
        for _ in 0..self.config.food_patch_count {
            let x = next_rand(&mut rng_state)
                * (self.config.grid_size as f32 - 2.0 * self.config.food_radius)
                + self.config.food_radius;
            let y = next_rand(&mut rng_state)
                * (self.config.grid_size as f32 - 2.0 * self.config.food_radius)
                + self.config.food_radius;
            world.add_food(x, y, self.config.food_radius, self.config.food_value);
        }
    }

    fn tick_world(&mut self, world: &mut World, _tick: usize) {
        world.tick = _tick;
        // Stationary task: food doesn't move or change
    }

    fn get_sensors(&self, organism: &Organism, world: &World) -> Vec<f32> {
        let mut sensors = vec![0.0; 8];
        let range = self.config.sensor_range;

        // Direction vectors: N, S, E, W
        let directions = [
            (0.0, -1.0), // North
            (0.0, 1.0),  // South
            (1.0, 0.0),  // East
            (-1.0, 0.0), // West
        ];

        for (i, (dx, dy)) in directions.iter().enumerate() {
            // Sensor i: Food density in direction
            sensors[i] = world.food_density_in_direction(organism.x, organism.y, *dx, *dy, range);

            // Sensor i+4: Distance to nearest food in direction (inverted: 1 = close, 0 = far)
            let dist =
                world.nearest_food_distance_in_direction(organism.x, organism.y, *dx, *dy, range);
            sensors[i + 4] = 1.0 - dist;
        }

        sensors
    }

    fn apply_action(&self, organism: &mut Organism, action_idx: usize, world: &mut World) {
        let dir = Direction::from_action(action_idx);
        let (dx, dy) = dir.delta();

        organism.move_by(dx as f32, dy as f32, world.width, world.height);
        organism.tick_metabolism();

        // Check if on food
        if let Some(food_idx) = world.is_on_food(organism.x, organism.y) {
            let value = world.food_patches[food_idx].value;
            organism.eat(value);
            // Food is consumed but respawns (stationary = infinite food)
            // For true stationary, we don't deplete it
        }
    }

    fn calculate_fitness(&self, organism: &Organism) -> f32 {
        // Fitness = food consumed + survival bonus
        organism.state.food_consumed + (organism.state.ticks_survived as f32 * 0.01)
    }

    fn reset_for_generation(&mut self, world: &mut World, seed: u64) {
        // Regenerate food positions for variety
        self.setup_world(world, seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_task_setup() {
        let config = StationaryConfig::default();
        let task = StationaryTask::new(config.clone());
        let mut world = World::new(config.grid_size);

        task.setup_world(&mut world, 42);

        assert_eq!(world.food_patches.len(), config.food_patch_count);
        for food in &world.food_patches {
            assert!(food.x >= 0.0 && food.x < config.grid_size as f32);
            assert!(food.y >= 0.0 && food.y < config.grid_size as f32);
        }
    }

    #[test]
    fn test_stationary_sensors() {
        let config = StationaryConfig {
            grid_size: 100,
            food_patch_count: 1,
            food_radius: 10.0,
            food_value: 10.0,
            sensor_range: 50.0,
        };
        let task = StationaryTask::new(config);
        let mut world = World::new(100);

        // Place food directly east of organism
        world.add_food(80.0, 50.0, 10.0, 10.0);

        let organism = Organism::new(0, 50.0, 50.0);
        let sensors = task.get_sensors(&organism, &world);

        assert_eq!(sensors.len(), 8);
        // East direction should have highest density
        assert!(sensors[2] > sensors[0]); // East > North
        assert!(sensors[2] > sensors[1]); // East > South
        assert!(sensors[2] > sensors[3]); // East > West
    }
}
