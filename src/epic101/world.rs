//! World representation for EPIC 101 experiments

/// A food patch in the world
#[derive(Clone, Debug)]
pub struct FoodPatch {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub value: f32,
    pub active: bool,
}

impl FoodPatch {
    pub fn new(x: f32, y: f32, radius: f32, value: f32) -> Self {
        Self {
            x,
            y,
            radius,
            value,
            active: true,
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        if !self.active {
            return false;
        }
        let dx = px - self.x;
        let dy = py - self.y;
        (dx * dx + dy * dy).sqrt() <= self.radius
    }

    pub fn distance_to(&self, px: f32, py: f32) -> f32 {
        let dx = px - self.x;
        let dy = py - self.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A beacon (for Task 5: Delayed Reward)
#[derive(Clone, Debug)]
pub struct Beacon {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub active: bool,
}

impl Beacon {
    pub fn new(x: f32, y: f32, radius: f32) -> Self {
        Self {
            x,
            y,
            radius,
            active: true,
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        if !self.active {
            return false;
        }
        let dx = px - self.x;
        let dy = py - self.y;
        (dx * dx + dy * dy).sqrt() <= self.radius
    }

    pub fn distance_to(&self, px: f32, py: f32) -> f32 {
        let dx = px - self.x;
        let dy = py - self.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn direction_to(&self, px: f32, py: f32) -> (f32, f32) {
        let dx = self.x - px;
        let dy = self.y - py;
        let dist = (dx * dx + dy * dy).sqrt().max(0.001);
        (dx / dist, dy / dist)
    }
}

/// The world containing food, beacons, and environment state
#[derive(Clone, Debug)]
pub struct World {
    pub width: f32,
    pub height: f32,
    pub food_patches: Vec<FoodPatch>,
    pub beacons: Vec<Beacon>,
    pub tick: usize,
}

impl World {
    pub fn new(size: usize) -> Self {
        Self {
            width: size as f32,
            height: size as f32,
            food_patches: Vec::new(),
            beacons: Vec::new(),
            tick: 0,
        }
    }

    pub fn clear(&mut self) {
        self.food_patches.clear();
        self.beacons.clear();
        self.tick = 0;
    }

    pub fn add_food(&mut self, x: f32, y: f32, radius: f32, value: f32) {
        self.food_patches.push(FoodPatch::new(x, y, radius, value));
    }

    pub fn add_beacon(&mut self, x: f32, y: f32, radius: f32) {
        self.beacons.push(Beacon::new(x, y, radius));
    }

    /// Get food density in a direction from position
    /// Returns value in [0, 1] based on nearby food
    pub fn food_density_in_direction(&self, px: f32, py: f32, dx: f32, dy: f32, range: f32) -> f32 {
        let mut density = 0.0;
        let norm = (dx * dx + dy * dy).sqrt().max(0.001);
        let ndx = dx / norm;
        let ndy = dy / norm;

        for food in &self.food_patches {
            if !food.active {
                continue;
            }
            let fx = food.x - px;
            let fy = food.y - py;
            let dist = (fx * fx + fy * fy).sqrt();

            if dist > range {
                continue;
            }

            // Check if food is in the specified direction (dot product > 0)
            let dot = fx * ndx + fy * ndy;
            if dot > 0.0 {
                // Weight by proximity and alignment
                let alignment = dot / dist.max(0.001);
                let proximity = 1.0 - (dist / range);
                density += alignment * proximity * food.value / 10.0;
            }
        }

        density.min(1.0)
    }

    /// Get distance to nearest food in a direction
    /// Returns normalized distance [0, 1] where 0 = at food, 1 = no food in range
    pub fn nearest_food_distance_in_direction(
        &self,
        px: f32,
        py: f32,
        dx: f32,
        dy: f32,
        range: f32,
    ) -> f32 {
        let norm = (dx * dx + dy * dy).sqrt().max(0.001);
        let ndx = dx / norm;
        let ndy = dy / norm;

        let mut nearest_dist = range;

        for food in &self.food_patches {
            if !food.active {
                continue;
            }
            let fx = food.x - px;
            let fy = food.y - py;
            let dist = (fx * fx + fy * fy).sqrt();

            // Check if food is in the specified direction
            let dot = fx * ndx + fy * ndy;
            if dot > 0.0 && dist < nearest_dist {
                nearest_dist = dist;
            }
        }

        nearest_dist / range
    }

    /// Check if position is inside any active food patch
    pub fn is_on_food(&self, px: f32, py: f32) -> Option<usize> {
        for (i, food) in self.food_patches.iter().enumerate() {
            if food.contains(px, py) {
                return Some(i);
            }
        }
        None
    }

    /// Check if position is inside any beacon
    pub fn is_on_beacon(&self, px: f32, py: f32) -> Option<usize> {
        for (i, beacon) in self.beacons.iter().enumerate() {
            if beacon.contains(px, py) {
                return Some(i);
            }
        }
        None
    }

    /// Clamp position to world bounds
    pub fn clamp_position(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x.max(0.0).min(self.width - 1.0),
            y.max(0.0).min(self.height - 1.0),
        )
    }
}
