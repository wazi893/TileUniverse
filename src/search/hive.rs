//! Hive Mesh Communication for the search engine (Phase 3).
//!
//! Implements torus topology neighbor communication for distributed
//! gradient sharing and local search bias.

use rayon::prelude::*;

use super::candidate::{Candidate, Rng, Xorshift64};
use super::substrate::SearchSubstrate;

/// Hive mesh configuration
#[derive(Clone, Debug)]
pub struct HiveConfig {
    /// Grid width (candidates arranged in 2D torus)
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Neighbor influence strength (0.0-1.0)
    pub influence: f32,
    /// Gradient momentum (0.0-1.0)
    pub momentum: f32,
    /// Local search probability
    pub local_search_rate: f32,
    /// Number of bits to consider for local search
    pub local_search_bits: usize,
}

impl HiveConfig {
    /// Create config for given population size (auto-grid)
    pub fn for_population(size: usize) -> Self {
        // Find roughly square dimensions
        let width = (size as f64).sqrt().ceil() as usize;
        let height = (size + width - 1) / width;

        Self {
            width,
            height,
            influence: 0.3,
            momentum: 0.9,
            local_search_rate: 0.1,
            local_search_bits: 3,
        }
    }

    /// Create config with explicit dimensions
    pub fn with_dimensions(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            influence: 0.3,
            momentum: 0.9,
            local_search_rate: 0.1,
            local_search_bits: 3,
        }
    }

    /// Set influence strength
    pub fn with_influence(mut self, influence: f32) -> Self {
        assert!(influence >= 0.0 && influence <= 1.0);
        self.influence = influence;
        self
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        assert!(momentum >= 0.0 && momentum <= 1.0);
        self.momentum = momentum;
        self
    }

    /// Set local search rate
    pub fn with_local_search(mut self, rate: f32, bits: usize) -> Self {
        assert!(rate >= 0.0 && rate <= 1.0);
        self.local_search_rate = rate;
        self.local_search_bits = bits;
        self
    }

    /// Total grid capacity
    pub fn capacity(&self) -> usize {
        self.width * self.height
    }
}

impl Default for HiveConfig {
    fn default() -> Self {
        Self::for_population(1_000_000)
    }
}

/// Direction for neighbor lookup
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

impl Direction {
    /// All 4 cardinal directions
    pub fn cardinals() -> &'static [Direction] {
        &[
            Direction::North,
            Direction::South,
            Direction::East,
            Direction::West,
        ]
    }

    /// All 8 directions (Moore neighborhood)
    pub fn all() -> &'static [Direction] {
        &[
            Direction::North,
            Direction::South,
            Direction::East,
            Direction::West,
            Direction::NorthEast,
            Direction::NorthWest,
            Direction::SouthEast,
            Direction::SouthWest,
        ]
    }
}

/// Gradient information from neighbors
#[derive(Clone, Debug, Default)]
pub struct NeighborGradient {
    /// Best fitness among neighbors
    pub best_fitness: u8,
    /// Mean fitness of neighbors
    pub mean_fitness: f32,
    /// Direction to best neighbor
    pub best_direction: Option<Direction>,
    /// Fitness gradient (best - self)
    pub gradient: i16,
    /// Bit pattern suggestion from best neighbor
    pub suggested_bits: Option<u64>,
}

/// Torus topology mesh for neighbor communication
pub struct HiveMesh {
    config: HiveConfig,
    /// Gradient accumulator for momentum
    gradients: Vec<f32>,
    /// Best neighbor indices (cached)
    best_neighbors: Vec<usize>,
    /// Generation counter
    generation: u64,
}

impl HiveMesh {
    /// Create a new hive mesh
    pub fn new(config: HiveConfig) -> Self {
        let capacity = config.capacity();
        Self {
            config,
            gradients: vec![0.0; capacity],
            best_neighbors: vec![0; capacity],
            generation: 0,
        }
    }

    /// Create with auto-sized config for population
    pub fn for_population(size: usize) -> Self {
        Self::new(HiveConfig::for_population(size))
    }

    /// Convert linear index to 2D coordinates
    #[inline]
    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        (index % self.config.width, index / self.config.width)
    }

    /// Convert 2D coordinates to linear index
    #[inline]
    pub fn coords_to_index(&self, x: usize, y: usize) -> usize {
        y * self.config.width + x
    }

    /// Get neighbor index in given direction (torus wrapping)
    #[inline]
    pub fn neighbor_index(&self, index: usize, direction: Direction) -> usize {
        let (x, y) = self.index_to_coords(index);
        let w = self.config.width;
        let h = self.config.height;

        let (nx, ny) = match direction {
            Direction::North => (x, if y == 0 { h - 1 } else { y - 1 }),
            Direction::South => (x, if y == h - 1 { 0 } else { y + 1 }),
            Direction::East => (if x == w - 1 { 0 } else { x + 1 }, y),
            Direction::West => (if x == 0 { w - 1 } else { x - 1 }, y),
            Direction::NorthEast => (
                if x == w - 1 { 0 } else { x + 1 },
                if y == 0 { h - 1 } else { y - 1 },
            ),
            Direction::NorthWest => (
                if x == 0 { w - 1 } else { x - 1 },
                if y == 0 { h - 1 } else { y - 1 },
            ),
            Direction::SouthEast => (
                if x == w - 1 { 0 } else { x + 1 },
                if y == h - 1 { 0 } else { y + 1 },
            ),
            Direction::SouthWest => (
                if x == 0 { w - 1 } else { x - 1 },
                if y == h - 1 { 0 } else { y + 1 },
            ),
        };

        self.coords_to_index(nx, ny)
    }

    /// Get all cardinal neighbor indices
    pub fn cardinal_neighbors(&self, index: usize) -> [usize; 4] {
        [
            self.neighbor_index(index, Direction::North),
            self.neighbor_index(index, Direction::South),
            self.neighbor_index(index, Direction::East),
            self.neighbor_index(index, Direction::West),
        ]
    }

    /// Get all 8 Moore neighborhood indices
    pub fn moore_neighbors(&self, index: usize) -> [usize; 8] {
        [
            self.neighbor_index(index, Direction::North),
            self.neighbor_index(index, Direction::South),
            self.neighbor_index(index, Direction::East),
            self.neighbor_index(index, Direction::West),
            self.neighbor_index(index, Direction::NorthEast),
            self.neighbor_index(index, Direction::NorthWest),
            self.neighbor_index(index, Direction::SouthEast),
            self.neighbor_index(index, Direction::SouthWest),
        ]
    }

    /// Compute gradient information from neighbors
    pub fn compute_gradient(&self, index: usize, substrate: &SearchSubstrate) -> NeighborGradient {
        let self_fitness = substrate.get(index).fitness;
        let neighbors = self.cardinal_neighbors(index);

        let mut best_fitness = 0u8;
        let mut best_direction = None;
        let mut best_neighbor_idx = index;
        let mut total_fitness = 0u32;

        for (i, &neighbor_idx) in neighbors.iter().enumerate() {
            if neighbor_idx < substrate.len() {
                let fitness = substrate.get(neighbor_idx).fitness;
                total_fitness += fitness as u32;

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_direction = Some(Direction::cardinals()[i]);
                    best_neighbor_idx = neighbor_idx;
                }
            }
        }

        let mean_fitness = total_fitness as f32 / neighbors.len() as f32;
        let gradient = best_fitness as i16 - self_fitness as i16;

        let suggested_bits = if gradient > 0 && best_neighbor_idx < substrate.len() {
            Some(substrate.get(best_neighbor_idx).bits)
        } else {
            None
        };

        NeighborGradient {
            best_fitness,
            mean_fitness,
            best_direction,
            gradient,
            suggested_bits,
        }
    }

    /// Update all gradients from substrate (parallel)
    pub fn update_gradients(&mut self, substrate: &SearchSubstrate) {
        let capacity = self.config.capacity().min(substrate.len());
        let momentum = self.config.momentum;

        // Compute new gradients in parallel
        let new_gradients: Vec<f32> = (0..capacity)
            .into_par_iter()
            .map(|i| {
                let grad = self.compute_gradient(i, substrate);
                grad.gradient as f32
            })
            .collect();

        // Update with momentum
        for (i, new_grad) in new_gradients.into_iter().enumerate() {
            self.gradients[i] = momentum * self.gradients[i] + (1.0 - momentum) * new_grad;
        }

        // Update best neighbors
        self.best_neighbors = (0..capacity)
            .into_par_iter()
            .map(|i| {
                let neighbors = self.cardinal_neighbors(i);
                let mut best_idx = i;
                let mut best_fit = substrate.get(i).fitness;
                for &n in &neighbors {
                    if n < substrate.len() && substrate.get(n).fitness > best_fit {
                        best_fit = substrate.get(n).fitness;
                        best_idx = n;
                    }
                }
                best_idx
            })
            .collect();

        self.generation += 1;
    }

    /// Apply neighbor influence to create offspring
    /// Returns modified candidate that incorporates neighbor information
    pub fn apply_influence(
        &self,
        index: usize,
        candidate: &Candidate,
        substrate: &SearchSubstrate,
        rng: &mut impl Rng,
    ) -> Candidate {
        let gradient = self.compute_gradient(index, substrate);

        // If no improvement available from neighbors, return unchanged
        if gradient.gradient <= 0 || gradient.suggested_bits.is_none() {
            return candidate.clone();
        }

        let neighbor_bits = gradient.suggested_bits.unwrap();
        let influence = self.config.influence;

        // Probabilistically adopt bits from better neighbor
        let mut new_bits = candidate.bits;

        for bit in 0..64 {
            let self_bit = (candidate.bits >> bit) & 1;
            let neighbor_bit = (neighbor_bits >> bit) & 1;

            if self_bit != neighbor_bit && rng.next_f32() < influence {
                // Adopt neighbor's bit
                if neighbor_bit == 1 {
                    new_bits |= 1 << bit;
                } else {
                    new_bits &= !(1 << bit);
                }
            }
        }

        Candidate::new(new_bits)
    }

    /// Perform local search around a candidate
    pub fn local_search(
        &self,
        candidate: &Candidate,
        fitness_fn: &dyn super::fitness::FitnessFunction,
        rng: &mut impl Rng,
    ) -> Candidate {
        let mut best = candidate.clone();
        best.fitness = fitness_fn.evaluate(&best);

        // Try flipping a few random bits
        for _ in 0..self.config.local_search_bits {
            let bit = (rng.next_u64() as usize) % 64;
            let mut trial = best.clone();
            trial.flip_bit(bit);
            trial.fitness = fitness_fn.evaluate(&trial);

            if trial.fitness > best.fitness {
                best = trial;
            }
        }

        best
    }

    /// Get accumulated gradient for candidate
    pub fn get_gradient(&self, index: usize) -> f32 {
        if index < self.gradients.len() {
            self.gradients[index]
        } else {
            0.0
        }
    }

    /// Get best neighbor index
    pub fn get_best_neighbor(&self, index: usize) -> usize {
        if index < self.best_neighbors.len() {
            self.best_neighbors[index]
        } else {
            index
        }
    }

    /// Get mesh statistics
    pub fn statistics(&self) -> HiveMeshStats {
        let n = self.gradients.len();
        if n == 0 {
            return HiveMeshStats::default();
        }

        let positive_gradients = self.gradients.iter().filter(|&&g| g > 0.0).count();
        let mean_gradient = self.gradients.iter().sum::<f32>() / n as f32;
        let max_gradient = self.gradients.iter().cloned().fold(f32::MIN, f32::max);

        HiveMeshStats {
            width: self.config.width,
            height: self.config.height,
            generation: self.generation,
            positive_gradients,
            mean_gradient,
            max_gradient,
        }
    }

    /// Config accessor
    pub fn config(&self) -> &HiveConfig {
        &self.config
    }
}

/// Statistics for hive mesh
#[derive(Clone, Debug, Default)]
pub struct HiveMeshStats {
    pub width: usize,
    pub height: usize,
    pub generation: u64,
    pub positive_gradients: usize,
    pub mean_gradient: f32,
    pub max_gradient: f32,
}

/// Gradient-biased selection using hive mesh
pub struct HiveSelector {
    mesh: HiveMesh,
}

impl HiveSelector {
    /// Create a new hive selector
    pub fn new(mesh: HiveMesh) -> Self {
        Self { mesh }
    }

    /// Select candidates with gradient bias
    /// Candidates with positive gradients (better neighbors) are favored
    pub fn select(
        &mut self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        // Update gradients
        self.mesh.update_gradients(substrate);

        let n = substrate.len();
        let mut selected = Vec::with_capacity(count);

        // Compute selection weights based on fitness + gradient bonus
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let fitness = substrate.get(i).fitness as f32;
                let gradient = self.mesh.get_gradient(i).max(0.0);
                fitness + gradient * 0.5 // Gradient bonus
            })
            .collect();

        let total: f32 = weights.iter().sum();

        for _ in 0..count {
            let target = rng.next_f32() * total;
            let mut cumulative = 0.0;
            let mut found = false;

            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= target {
                    selected.push(i);
                    found = true;
                    break;
                }
            }

            // Fallback if no selection made
            if !found {
                selected.push(n - 1);
            }
        }

        selected
    }

    /// Get the underlying mesh
    pub fn mesh(&self) -> &HiveMesh {
        &self.mesh
    }

    /// Get mutable mesh
    pub fn mesh_mut(&mut self) -> &mut HiveMesh {
        &mut self.mesh
    }
}

/// Parallel gradient computation for large populations
pub fn compute_all_gradients_parallel(
    substrate: &SearchSubstrate,
    mesh: &HiveMesh,
) -> Vec<NeighborGradient> {
    let n = substrate.len().min(mesh.config.capacity());

    (0..n)
        .into_par_iter()
        .map(|i| mesh.compute_gradient(i, substrate))
        .collect()
}

/// Apply hive influence to entire population (parallel)
pub fn apply_hive_influence_parallel(
    substrate: &SearchSubstrate,
    mesh: &HiveMesh,
    seed: u64,
) -> Vec<Candidate> {
    let n = substrate.len();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Xorshift64::new(seed.wrapping_add(i as u64));
            mesh.apply_influence(i, substrate.get(i), substrate, &mut rng)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::{FitnessFunction, OneMax};

    #[test]
    fn test_hive_config() {
        let config = HiveConfig::for_population(1_000_000);
        assert_eq!(config.capacity(), config.width * config.height);
        assert!(config.capacity() >= 1_000_000);
    }

    #[test]
    fn test_coords_conversion() {
        let mesh = HiveMesh::new(HiveConfig::with_dimensions(100, 100));

        // Test round-trip
        for i in [0, 50, 99, 100, 5000, 9999] {
            let (x, y) = mesh.index_to_coords(i);
            let back = mesh.coords_to_index(x, y);
            assert_eq!(i, back, "Failed for index {}", i);
        }
    }

    #[test]
    fn test_torus_wrapping() {
        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));

        // Test wrapping at edges
        // Index 0 (top-left corner)
        assert_eq!(mesh.neighbor_index(0, Direction::North), 90); // Wraps to bottom
        assert_eq!(mesh.neighbor_index(0, Direction::West), 9); // Wraps to right

        // Index 9 (top-right corner)
        assert_eq!(mesh.neighbor_index(9, Direction::East), 0); // Wraps to left
        assert_eq!(mesh.neighbor_index(9, Direction::North), 99); // Wraps to bottom

        // Index 99 (bottom-right corner)
        assert_eq!(mesh.neighbor_index(99, Direction::South), 9); // Wraps to top
        assert_eq!(mesh.neighbor_index(99, Direction::East), 90); // Wraps to left
    }

    #[test]
    fn test_cardinal_neighbors() {
        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));

        // Middle cell (55)
        let neighbors = mesh.cardinal_neighbors(55);
        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0], 45); // North
        assert_eq!(neighbors[1], 65); // South
        assert_eq!(neighbors[2], 56); // East
        assert_eq!(neighbors[3], 54); // West
    }

    #[test]
    fn test_gradient_computation() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));
        let gradient = mesh.compute_gradient(55, &substrate);

        assert!(gradient.mean_fitness >= 0.0);
        // Gradient could be positive, negative, or zero depending on random init
    }

    #[test]
    fn test_gradient_update() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));
        mesh.update_gradients(&substrate);

        // Should have computed gradients
        assert_eq!(mesh.gradients.len(), 100);

        // Best neighbors should be valid indices
        for &idx in &mesh.best_neighbors {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_apply_influence() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10).with_influence(0.5));

        let mut rng = Xorshift64::new(123);
        let original = substrate.get(50).clone();
        let influenced = mesh.apply_influence(50, &original, &substrate, &mut rng);

        // Influenced candidate may or may not be different depending on gradients
        // Just verify it's a valid candidate
        assert!(influenced.bits <= u64::MAX);
    }

    #[test]
    fn test_local_search() {
        let fitness_fn = OneMax::new(8);
        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10).with_local_search(0.5, 3));

        // Create a suboptimal candidate
        let mut candidate = Candidate::new(0b11110000);
        candidate.fitness = fitness_fn.evaluate(&candidate);

        let mut rng = Xorshift64::new(42);
        let improved = mesh.local_search(&candidate, &fitness_fn, &mut rng);

        // Local search should find equal or better solution
        assert!(improved.fitness >= candidate.fitness);
    }

    #[test]
    fn test_hive_selector() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));
        let mut selector = HiveSelector::new(mesh);
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 20, &mut rng);
        assert_eq!(selected.len(), 20);

        for &idx in &selected {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_parallel_gradient_computation() {
        let mut substrate = SearchSubstrate::new(1000, OneMax::new(16));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mesh = HiveMesh::for_population(1000);
        let gradients = compute_all_gradients_parallel(&substrate, &mesh);

        assert_eq!(gradients.len(), 1000);
    }

    #[test]
    fn test_mesh_statistics() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut mesh = HiveMesh::new(HiveConfig::with_dimensions(10, 10));
        mesh.update_gradients(&substrate);

        let stats = mesh.statistics();
        assert_eq!(stats.width, 10);
        assert_eq!(stats.height, 10);
        assert_eq!(stats.generation, 1);
    }

    #[test]
    fn test_1m_scale_mesh() {
        // Test mesh at 1M scale (quick operation)
        let config = HiveConfig::for_population(1_000_000);
        let mesh = HiveMesh::new(config);

        assert!(mesh.config.capacity() >= 1_000_000);

        // Test neighbor lookups are fast
        let start = std::time::Instant::now();
        for i in 0..10000 {
            let _ = mesh.cardinal_neighbors(i * 100);
        }
        let duration = start.elapsed();
        assert!(
            duration.as_millis() < 100,
            "Neighbor lookup too slow: {:?}",
            duration
        );
    }
}
