//! Task 5: Delayed Reward (Key Test)
//!
//! Must visit beacon BEFORE food becomes accessible.
//! Expected: LSTM dominates (requires memory of beacon visit)
//!
//! This is THE critical test for LSTM value - if LSTM can't win here,
//! it can't win anywhere.

use super::organism::Organism;
use super::task::{Direction, Task};
use super::world::World;

/// Configuration for Delayed Reward task
#[derive(Clone, Debug)]
pub struct DelayedRewardConfig {
    pub grid_size: usize,
    pub beacon_x: f32,
    pub beacon_y: f32,
    pub beacon_radius: f32,
    pub food_x: f32,
    pub food_y: f32,
    pub food_radius: f32,
    pub food_value: f32,
    /// How long food stays accessible after beacon visit
    pub unlock_duration: usize,
    /// Sensor range
    pub sensor_range: f32,
}

impl Default for DelayedRewardConfig {
    fn default() -> Self {
        let grid_size = 256;
        Self {
            grid_size,
            // Beacon in upper-middle area
            beacon_x: (grid_size as f32) / 2.0,
            beacon_y: (grid_size as f32) / 4.0,
            beacon_radius: 20.0,
            // Food in lower-middle area
            food_x: (grid_size as f32) / 2.0,
            food_y: (grid_size as f32) * 3.0 / 4.0,
            food_radius: 25.0,
            food_value: 50.0, // High value to make it worth the effort
            unlock_duration: 500,
            sensor_range: 150.0,
        }
    }
}

/// Task 5: Delayed Reward
///
/// The organism must:
/// 1. Visit the beacon first
/// 2. Then go to the food (which is now accessible)
/// 3. Food only stays accessible for unlock_duration ticks
///
/// Simple NN cannot remember the beacon visit.
/// LSTM should be able to maintain the "beacon visited" state internally.
pub struct DelayedRewardTask {
    config: DelayedRewardConfig,
}

impl DelayedRewardTask {
    pub fn new(config: DelayedRewardConfig) -> Self {
        Self { config }
    }

    /// Check if food is accessible for this organism
    fn is_food_accessible(&self, organism: &Organism, current_tick: usize) -> bool {
        if let Some(visit_tick) = organism.state.beacon_visit_tick {
            // Food is accessible for unlock_duration ticks after beacon visit
            current_tick <= visit_tick + self.config.unlock_duration
        } else {
            false
        }
    }
}

impl Task for DelayedRewardTask {
    fn name(&self) -> &str {
        "Task5_DelayedReward"
    }

    fn sensor_count(&self) -> usize {
        10 // See get_sensors for breakdown
    }

    fn setup_world(&self, world: &mut World, _seed: u64) {
        world.clear();

        // Add beacon
        world.add_beacon(
            self.config.beacon_x,
            self.config.beacon_y,
            self.config.beacon_radius,
        );

        // Add food (will be conditionally accessible)
        world.add_food(
            self.config.food_x,
            self.config.food_y,
            self.config.food_radius,
            self.config.food_value,
        );
    }

    fn tick_world(&mut self, world: &mut World, tick: usize) {
        world.tick = tick;
        // Food and beacon positions don't change
    }

    fn get_sensors(&self, organism: &Organism, world: &World) -> Vec<f32> {
        let mut sensors = vec![0.0; 10];

        // Sensors 0-1: Direction to beacon (normalized)
        if let Some(beacon) = world.beacons.first() {
            let (dx, dy) = organism.direction_to(beacon.x, beacon.y);
            sensors[0] = dx; // [-1, 1]
            sensors[1] = dy; // [-1, 1]

            // Sensor 2: Distance to beacon (normalized: 0 = at beacon, 1 = far)
            let dist = organism.distance_to(beacon.x, beacon.y);
            sensors[2] = (dist / self.config.sensor_range).min(1.0);
        }

        // Sensor 3: Beacon visited? (THIS IS THE KEY)
        // Note: Simple NN gets this info but can't remember it changed!
        // LSTM should maintain internal state even when sensor goes back to 0
        sensors[3] = if organism.state.beacon_visited {
            1.0
        } else {
            0.0
        };

        // Sensors 4-5: Direction to food (normalized)
        if let Some(food) = world.food_patches.first() {
            let (dx, dy) = organism.direction_to(food.x, food.y);
            sensors[4] = dx;
            sensors[5] = dy;

            // Sensor 6: Distance to food (normalized)
            let dist = organism.distance_to(food.x, food.y);
            sensors[6] = (dist / self.config.sensor_range).min(1.0);
        }

        // Sensor 7: Food currently accessible?
        let accessible = self.is_food_accessible(organism, world.tick);
        sensors[7] = if accessible { 1.0 } else { 0.0 };

        // Sensors 8-9: Random noise (to test robustness)
        // Using deterministic "noise" based on position for reproducibility
        sensors[8] = ((organism.x * 0.1).sin() + 1.0) / 2.0;
        sensors[9] = ((organism.y * 0.1).cos() + 1.0) / 2.0;

        sensors
    }

    fn apply_action(&self, organism: &mut Organism, action_idx: usize, world: &mut World) {
        let dir = Direction::from_action(action_idx);
        let (dx, dy) = dir.delta();

        organism.move_by(dx as f32, dy as f32, world.width, world.height);
        organism.tick_metabolism();

        // Check if organism reached beacon
        if !organism.state.beacon_visited {
            if world.is_on_beacon(organism.x, organism.y).is_some() {
                organism.state.beacon_visited = true;
                organism.state.beacon_visit_tick = Some(world.tick);
            }
        }

        // Check if organism can eat food
        if self.is_food_accessible(organism, world.tick) {
            if world.is_on_food(organism.x, organism.y).is_some() {
                let value = self.config.food_value;
                organism.eat(value);
            }
        }
    }

    fn calculate_fitness(&self, organism: &Organism) -> f32 {
        // Primary: food consumed (only possible if visited beacon first)
        let food_fitness = organism.state.food_consumed * 10.0;

        // Secondary: beacon visit bonus (encourages exploration)
        let beacon_bonus = if organism.state.beacon_visited {
            20.0
        } else {
            0.0
        };

        // Tertiary: survival time
        let survival_bonus = organism.state.ticks_survived as f32 * 0.01;

        food_fitness + beacon_bonus + survival_bonus
    }

    fn reset_for_generation(&mut self, world: &mut World, seed: u64) {
        self.setup_world(world, seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delayed_reward_setup() {
        let config = DelayedRewardConfig::default();
        let task = DelayedRewardTask::new(config);
        let mut world = World::new(256);

        task.setup_world(&mut world, 42);

        assert_eq!(world.beacons.len(), 1);
        assert_eq!(world.food_patches.len(), 1);
    }

    #[test]
    fn test_delayed_reward_mechanics() {
        let config = DelayedRewardConfig {
            grid_size: 100,
            beacon_x: 25.0,
            beacon_y: 25.0,
            beacon_radius: 10.0,
            food_x: 75.0,
            food_y: 75.0,
            food_radius: 10.0,
            food_value: 50.0,
            unlock_duration: 100,
            sensor_range: 100.0,
        };
        let task = DelayedRewardTask::new(config);
        let mut world = World::new(100);
        task.setup_world(&mut world, 42);

        let mut organism = Organism::new(0, 50.0, 50.0);

        // Initially, food should not be accessible
        assert!(!task.is_food_accessible(&organism, 0));

        // Visit beacon
        organism.x = 25.0;
        organism.y = 25.0;
        organism.state.beacon_visited = true;
        organism.state.beacon_visit_tick = Some(10);

        // Now food should be accessible
        assert!(task.is_food_accessible(&organism, 10));
        assert!(task.is_food_accessible(&organism, 110)); // Still within duration

        // After unlock_duration, food becomes inaccessible again
        assert!(!task.is_food_accessible(&organism, 111));
    }

    #[test]
    fn test_delayed_reward_sensors() {
        let config = DelayedRewardConfig::default();
        let task = DelayedRewardTask::new(config.clone());
        let mut world = World::new(config.grid_size);
        task.setup_world(&mut world, 42);

        let organism = Organism::new(
            0,
            config.grid_size as f32 / 2.0,
            config.grid_size as f32 / 2.0,
        );
        let sensors = task.get_sensors(&organism, &world);

        assert_eq!(sensors.len(), 10);
        // Beacon not visited initially
        assert_eq!(sensors[3], 0.0);
        // Food not accessible initially
        assert_eq!(sensors[7], 0.0);
    }
}
