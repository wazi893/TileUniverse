//! Organism representation for EPIC 101 experiments

/// Per-organism state that persists across ticks
#[derive(Clone, Debug, Default)]
pub struct OrganismState {
    /// Has this organism visited the beacon? (for Task 5)
    pub beacon_visited: bool,
    /// Tick when beacon was visited
    pub beacon_visit_tick: Option<usize>,
    /// Total food consumed
    pub food_consumed: f32,
    /// Ticks survived
    pub ticks_survived: usize,
}

/// An organism in the simulation
#[derive(Clone, Debug)]
pub struct Organism {
    /// Unique ID
    pub id: usize,
    /// Position X
    pub x: f32,
    /// Position Y
    pub y: f32,
    /// Current energy (dies at 0)
    pub energy: f32,
    /// Maximum energy capacity
    pub max_energy: f32,
    /// Energy cost per tick
    pub metabolism: f32,
    /// Movement speed (pixels per tick)
    pub speed: f32,
    /// Task-specific state
    pub state: OrganismState,
    /// Fitness score (calculated at end of generation)
    pub fitness: f32,
}

impl Organism {
    pub fn new(id: usize, x: f32, y: f32) -> Self {
        Self {
            id,
            x,
            y,
            energy: 100.0,
            max_energy: 100.0,
            metabolism: 0.1, // Lose 0.1 energy per tick
            speed: 2.0,
            state: OrganismState::default(),
            fitness: 0.0,
        }
    }

    /// Apply metabolism cost
    pub fn tick_metabolism(&mut self) {
        self.energy = (self.energy - self.metabolism).max(0.0);
        self.state.ticks_survived += 1;
    }

    /// Consume food and gain energy
    pub fn eat(&mut self, value: f32) {
        self.energy = (self.energy + value).min(self.max_energy);
        self.state.food_consumed += value;
    }

    /// Move in a direction
    pub fn move_by(&mut self, dx: f32, dy: f32, world_width: f32, world_height: f32) {
        self.x = (self.x + dx * self.speed).max(0.0).min(world_width - 1.0);
        self.y = (self.y + dy * self.speed).max(0.0).min(world_height - 1.0);
    }

    /// Check if alive
    pub fn is_alive(&self) -> bool {
        self.energy > 0.0
    }

    /// Reset for new generation (keep position, reset state)
    pub fn reset(&mut self, x: f32, y: f32, energy: f32) {
        self.x = x;
        self.y = y;
        self.energy = energy;
        self.state = OrganismState::default();
        self.fitness = 0.0;
    }

    /// Distance to a point
    pub fn distance_to(&self, x: f32, y: f32) -> f32 {
        let dx = x - self.x;
        let dy = y - self.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Direction to a point (normalized)
    pub fn direction_to(&self, x: f32, y: f32) -> (f32, f32) {
        let dx = x - self.x;
        let dy = y - self.y;
        let dist = (dx * dx + dy * dy).sqrt().max(0.001);
        (dx / dist, dy / dist)
    }
}
