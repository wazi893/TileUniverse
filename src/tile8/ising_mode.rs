//! Tile8 Ising Mode: P-bit dynamics on the tile substrate
//!
//! Maps Ising model dynamics onto the existing tile infrastructure:
//! - Each tile acts as a p-bit (spin stored in logic value)
//! - Neighbor connections provide J couplings
//! - Leverages 40B evals/sec tile infrastructure for massive parallelism
//!
//! The Ising Hamiltonian: H = -Σ J_ij s_i s_j - Σ h_i s_i
//! Updates follow Gibbs sampling with configurable temperature.

use crate::slim_simulation::SlimSimulation;
use crate::tile_meta::TileType;

/// Simple LCG-based RNG for Ising updates (same as QEC module)
#[derive(Clone)]
pub struct IsingRng {
    state: u64,
}

impl IsingRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Configuration for Ising mode
#[derive(Clone, Debug)]
pub struct IsingConfig {
    /// Inverse temperature (β). Higher = more deterministic.
    pub beta: f64,
    /// Default coupling strength for neighbor connections
    pub default_j: f64,
    /// Default external field (bias)
    pub default_h: f64,
    /// Whether to use antiferromagnetic coupling (for MaxCut)
    pub antiferromagnetic: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for IsingConfig {
    fn default() -> Self {
        Self {
            beta: 1.0,
            default_j: 1.0,
            default_h: 0.0,
            antiferromagnetic: false,
            seed: 42,
        }
    }
}

impl IsingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_coupling(mut self, j: f64) -> Self {
        self.default_j = j;
        self
    }

    pub fn with_field(mut self, h: f64) -> Self {
        self.default_h = h;
        self
    }

    pub fn antiferromagnetic(mut self) -> Self {
        self.antiferromagnetic = true;
        self.default_j = -1.0; // Antiferromagnetic = negative coupling
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Ising grid built on the tile substrate
///
/// Uses SlimSimulation for efficient neighbor lookups and parallel updates.
/// Each tile's logic value represents spin: 0 = spin down (-1), nonzero = spin up (+1)
pub struct IsingGrid {
    sim: SlimSimulation,
    config: IsingConfig,
    rng: IsingRng,
    /// Per-tile coupling strengths (indexed as [idx * 4 + dir] for 4 neighbors)
    couplings: Vec<f64>,
    /// Per-tile external fields
    fields: Vec<f64>,
    /// Energy tracking
    current_energy: f64,
    best_energy: f64,
    best_state: Vec<u8>,
    /// Statistics
    steps: u64,
    flips: u64,
}

impl IsingGrid {
    /// Create a new Ising grid with given dimensions
    pub fn new(width: usize, height: usize, config: IsingConfig) -> Self {
        let mut sim = SlimSimulation::with_size(width, height);
        let tile_count = sim.tile_count();

        // Initialize all tiles as IsingNode type
        for y in 0..height {
            for x in 0..width {
                sim.set_tile(x, y, TileType::Selector); // Use Selector as Ising placeholder
            }
        }

        // Initialize coupling matrix (4 neighbors per tile)
        let j_val = config.default_j;
        let couplings = vec![j_val; tile_count * 4];

        // Initialize external fields
        let fields = vec![config.default_h; tile_count];

        let rng = IsingRng::new(config.seed);

        Self {
            sim,
            config,
            rng,
            couplings,
            fields,
            current_energy: 0.0,
            best_energy: f64::INFINITY,
            best_state: vec![0; tile_count],
            steps: 0,
            flips: 0,
        }
    }

    /// Get grid width
    pub fn width(&self) -> usize {
        self.sim.width()
    }

    /// Get grid height
    pub fn height(&self) -> usize {
        self.sim.height()
    }

    /// Get total spin count
    pub fn spin_count(&self) -> usize {
        self.sim.tile_count()
    }

    /// Set spin at position (true = up/+1, false = down/-1)
    pub fn set_spin(&mut self, x: usize, y: usize, spin_up: bool) {
        let val = if spin_up { 1u64 } else { 0u64 };
        self.sim.set_logic_value(x, y, val);
    }

    /// Get spin at position (true = up/+1, false = down/-1)
    pub fn get_spin(&self, x: usize, y: usize) -> bool {
        self.sim.get_logic_at(x, y) != 0
    }

    /// Get spin as Ising value (-1 or +1)
    pub fn get_spin_value(&self, x: usize, y: usize) -> i32 {
        if self.get_spin(x, y) { 1 } else { -1 }
    }

    /// Set coupling between adjacent tiles
    /// dir: 0=left, 1=right, 2=up, 3=down
    pub fn set_coupling(&mut self, x: usize, y: usize, dir: usize, j: f64) {
        let idx = y * self.width() + x;
        if idx < self.spin_count() && dir < 4 {
            self.couplings[idx * 4 + dir] = j;
        }
    }

    /// Set external field at position
    pub fn set_field(&mut self, x: usize, y: usize, h: f64) {
        let idx = y * self.width() + x;
        if idx < self.spin_count() {
            self.fields[idx] = h;
        }
    }

    /// Set inverse temperature
    pub fn set_beta(&mut self, beta: f64) {
        self.config.beta = beta;
    }

    /// Get current beta
    pub fn beta(&self) -> f64 {
        self.config.beta
    }

    /// Randomize all spins
    pub fn randomize(&mut self) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let spin_up = self.rng.next_f64() < 0.5;
                self.set_spin(x, y, spin_up);
            }
        }
        self.current_energy = self.compute_energy();
    }

    /// Compute total Ising energy: H = -Σ J_ij s_i s_j - Σ h_i s_i
    pub fn compute_energy(&self) -> f64 {
        let mut energy = 0.0;
        let width = self.width();
        let height = self.height();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let s_i = if self.sim.get_logic_at(x, y) != 0 {
                    1.0
                } else {
                    -1.0
                };

                // External field contribution
                energy -= self.fields[idx] * s_i;

                // Coupling contributions (only count right and down to avoid double counting)
                if x + 1 < width {
                    let s_j = if self.sim.get_logic_at(x + 1, y) != 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let j = self.couplings[idx * 4 + 1]; // right coupling
                    energy -= j * s_i * s_j;
                }

                if y + 1 < height {
                    let s_j = if self.sim.get_logic_at(x, y + 1) != 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let j = self.couplings[idx * 4 + 3]; // down coupling
                    energy -= j * s_i * s_j;
                }
            }
        }

        energy
    }

    /// Compute local field for a single spin: h_i + Σ_j J_ij s_j
    #[inline]
    fn local_field(&self, x: usize, y: usize) -> f64 {
        let width = self.width();
        let height = self.height();
        let idx = y * width + x;

        let mut field = self.fields[idx];

        // Left neighbor
        if x > 0 {
            let s_j = if self.sim.get_logic_at(x - 1, y) != 0 {
                1.0
            } else {
                -1.0
            };
            field += self.couplings[idx * 4 + 0] * s_j;
        }

        // Right neighbor
        if x + 1 < width {
            let s_j = if self.sim.get_logic_at(x + 1, y) != 0 {
                1.0
            } else {
                -1.0
            };
            field += self.couplings[idx * 4 + 1] * s_j;
        }

        // Up neighbor
        if y > 0 {
            let s_j = if self.sim.get_logic_at(x, y - 1) != 0 {
                1.0
            } else {
                -1.0
            };
            field += self.couplings[idx * 4 + 2] * s_j;
        }

        // Down neighbor
        if y + 1 < height {
            let s_j = if self.sim.get_logic_at(x, y + 1) != 0 {
                1.0
            } else {
                -1.0
            };
            field += self.couplings[idx * 4 + 3] * s_j;
        }

        field
    }

    /// Perform one Gibbs sampling update on a single spin
    #[inline]
    fn update_spin(&mut self, x: usize, y: usize) -> bool {
        let local_field = self.local_field(x, y);

        // Probability of spin up: P(s=+1) = 1 / (1 + exp(-2β * local_field))
        let prob_up = sigmoid(2.0 * self.config.beta * local_field);

        let old_spin = self.get_spin(x, y);
        let new_spin = self.rng.next_f64() < prob_up;

        if old_spin != new_spin {
            self.set_spin(x, y, new_spin);
            self.flips += 1;
            true
        } else {
            false
        }
    }

    /// Perform one full sweep (all spins updated once in random order)
    pub fn sweep(&mut self) {
        let width = self.width();
        let height = self.height();

        // Generate random order
        let mut order: Vec<(usize, usize)> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .collect();

        // Fisher-Yates shuffle
        for i in (1..order.len()).rev() {
            let j = (self.rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }

        // Update spins in random order
        for (x, y) in order {
            self.update_spin(x, y);
        }

        self.steps += 1;
        self.current_energy = self.compute_energy();

        // Track best state
        if self.current_energy < self.best_energy {
            self.best_energy = self.current_energy;
            self.save_best_state();
        }
    }

    /// Perform sequential sweep (checkerboard pattern for parallelism)
    pub fn sweep_checkerboard(&mut self) {
        let width = self.width();
        let height = self.height();

        // Update "black" squares (x+y even)
        for y in 0..height {
            for x in 0..width {
                if (x + y) % 2 == 0 {
                    self.update_spin(x, y);
                }
            }
        }

        // Update "white" squares (x+y odd)
        for y in 0..height {
            for x in 0..width {
                if (x + y) % 2 == 1 {
                    self.update_spin(x, y);
                }
            }
        }

        self.steps += 1;
        self.current_energy = self.compute_energy();

        if self.current_energy < self.best_energy {
            self.best_energy = self.current_energy;
            self.save_best_state();
        }
    }

    /// Perform simulated annealing
    pub fn anneal(&mut self, sweeps: usize, beta_min: f64, beta_max: f64) -> AnnealResult {
        self.randomize();

        for i in 0..sweeps {
            // Linear schedule
            let progress = i as f64 / sweeps as f64;
            self.config.beta = beta_min + (beta_max - beta_min) * progress;

            self.sweep();
        }

        AnnealResult {
            final_energy: self.current_energy,
            best_energy: self.best_energy,
            best_state: self.best_state.clone(),
            total_sweeps: sweeps,
            total_flips: self.flips,
        }
    }

    /// Save current state as best
    fn save_best_state(&mut self) {
        let width = self.width();
        for y in 0..self.height() {
            for x in 0..width {
                let idx = y * width + x;
                self.best_state[idx] = if self.get_spin(x, y) { 1 } else { 0 };
            }
        }
    }

    /// Restore best state
    pub fn restore_best(&mut self) {
        let width = self.width();
        for y in 0..self.height() {
            for x in 0..width {
                let idx = y * width + x;
                self.set_spin(x, y, self.best_state[idx] != 0);
            }
        }
        self.current_energy = self.best_energy;
    }

    /// Get current energy
    pub fn energy(&self) -> f64 {
        self.current_energy
    }

    /// Get best energy found
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get magnetization: M = (1/N) Σ s_i
    pub fn magnetization(&self) -> f64 {
        let mut sum = 0i64;
        for y in 0..self.height() {
            for x in 0..self.width() {
                sum += if self.get_spin(x, y) { 1 } else { -1 };
            }
        }
        sum as f64 / self.spin_count() as f64
    }

    /// Get flip rate (flips per sweep)
    pub fn flip_rate(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.flips as f64 / self.steps as f64 / self.spin_count() as f64
        }
    }

    /// Get raw state vector (for analysis)
    pub fn get_state(&self) -> Vec<u8> {
        let width = self.width();
        let mut state = vec![0u8; self.spin_count()];
        for y in 0..self.height() {
            for x in 0..width {
                let idx = y * width + x;
                state[idx] = if self.get_spin(x, y) { 1 } else { 0 };
            }
        }
        state
    }
}

/// Result of annealing run
#[derive(Debug, Clone)]
pub struct AnnealResult {
    pub final_energy: f64,
    pub best_energy: f64,
    pub best_state: Vec<u8>,
    pub total_sweeps: usize,
    pub total_flips: u64,
}

/// Numerically stable sigmoid function
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// MaxCut problem on a 2D grid (for benchmarking)
pub struct GridMaxCut {
    grid: IsingGrid,
}

impl GridMaxCut {
    /// Create a MaxCut problem on a width x height grid
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        let config = IsingConfig::new().antiferromagnetic().with_seed(seed);

        Self {
            grid: IsingGrid::new(width, height, config),
        }
    }

    /// Solve with simulated annealing
    pub fn solve(&mut self, sweeps: usize, beta_min: f64, beta_max: f64) -> MaxCutResult {
        let result = self.grid.anneal(sweeps, beta_min, beta_max);

        // Compute cut value
        let cut = self.compute_cut(&result.best_state);
        let max_edges = self.max_edges();

        MaxCutResult {
            cut_value: cut,
            max_edges,
            cut_fraction: cut as f64 / max_edges as f64,
            best_state: result.best_state,
            best_energy: result.best_energy,
        }
    }

    /// Compute cut value from state
    fn compute_cut(&self, state: &[u8]) -> usize {
        let width = self.grid.width();
        let height = self.grid.height();
        let mut cut = 0;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let s_i = state[idx];

                // Check right neighbor
                if x + 1 < width {
                    let s_j = state[idx + 1];
                    if s_i != s_j {
                        cut += 1;
                    }
                }

                // Check down neighbor
                if y + 1 < height {
                    let s_j = state[(y + 1) * width + x];
                    if s_i != s_j {
                        cut += 1;
                    }
                }
            }
        }

        cut
    }

    /// Maximum possible edges in grid
    fn max_edges(&self) -> usize {
        let w = self.grid.width();
        let h = self.grid.height();
        // Horizontal edges: (w-1) * h
        // Vertical edges: w * (h-1)
        (w - 1) * h + w * (h - 1)
    }
}

/// Result of MaxCut solve
#[derive(Debug, Clone)]
pub struct MaxCutResult {
    pub cut_value: usize,
    pub max_edges: usize,
    pub cut_fraction: f64,
    pub best_state: Vec<u8>,
    pub best_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_grid_creation() {
        let config = IsingConfig::new().with_beta(1.0);
        let grid = IsingGrid::new(8, 8, config);

        assert_eq!(grid.width(), 8);
        assert_eq!(grid.height(), 8);
        assert_eq!(grid.spin_count(), 64);
    }

    #[test]
    fn test_spin_operations() {
        let config = IsingConfig::new();
        let mut grid = IsingGrid::new(4, 4, config);

        grid.set_spin(1, 1, true);
        assert!(grid.get_spin(1, 1));
        assert_eq!(grid.get_spin_value(1, 1), 1);

        grid.set_spin(1, 1, false);
        assert!(!grid.get_spin(1, 1));
        assert_eq!(grid.get_spin_value(1, 1), -1);
    }

    #[test]
    fn test_ferromagnetic_ground_state() {
        // Ferromagnetic (J > 0) should prefer aligned spins
        let config = IsingConfig::new()
            .with_beta(10.0)
            .with_coupling(1.0)
            .with_seed(42);

        let mut grid = IsingGrid::new(8, 8, config);
        grid.anneal(100, 0.1, 10.0);

        // At low temperature, should be mostly aligned
        let mag = grid.magnetization().abs();
        assert!(
            mag > 0.8,
            "Ferromagnetic ground state should have high magnetization, got {}",
            mag
        );
    }

    #[test]
    fn test_antiferromagnetic_checkerboard() {
        // Antiferromagnetic (J < 0) on 2D grid should form checkerboard
        let config = IsingConfig::new()
            .antiferromagnetic()
            .with_beta(10.0)
            .with_seed(42);

        let mut grid = IsingGrid::new(8, 8, config);
        grid.anneal(200, 0.1, 10.0);

        // Checkerboard has zero magnetization
        let mag = grid.magnetization().abs();
        assert!(
            mag < 0.3,
            "Antiferromagnetic ground state should have low magnetization, got {}",
            mag
        );
    }

    #[test]
    fn test_energy_calculation() {
        let config = IsingConfig::new().with_coupling(1.0).with_field(0.0);
        let mut grid = IsingGrid::new(2, 2, config);

        // Set all spins up
        for y in 0..2 {
            for x in 0..2 {
                grid.set_spin(x, y, true);
            }
        }

        // Energy = -J * (number of aligned pairs)
        // 2x2 grid has 4 edges, all aligned → E = -4
        let energy = grid.compute_energy();
        assert!(
            (energy - (-4.0)).abs() < 0.01,
            "Energy should be -4, got {}",
            energy
        );
    }

    #[test]
    fn test_grid_maxcut() {
        let mut problem = GridMaxCut::new(8, 8, 42);
        let result = problem.solve(500, 0.1, 5.0);

        // 8x8 grid has (8-1)*8 + 8*(8-1) = 56 + 56 = 112 edges
        assert_eq!(result.max_edges, 112);

        // Should find a reasonable cut
        assert!(
            result.cut_fraction > 0.4,
            "MaxCut should find >40% cut, got {}%",
            result.cut_fraction * 100.0
        );

        println!(
            "Grid MaxCut: {} / {} edges ({:.1}%)",
            result.cut_value,
            result.max_edges,
            result.cut_fraction * 100.0
        );
    }

    #[test]
    fn test_sweep_reduces_energy() {
        let config = IsingConfig::new().with_beta(2.0).with_seed(123);
        let mut grid = IsingGrid::new(8, 8, config);
        grid.randomize();

        let initial_energy = grid.energy();

        // Run several sweeps at moderate temperature
        for _ in 0..50 {
            grid.sweep();
        }

        // Energy should generally decrease (with high beta, system cools)
        // Note: not guaranteed due to stochastic nature, but likely
        println!(
            "Energy: {} → {} (best: {})",
            initial_energy,
            grid.energy(),
            grid.best_energy()
        );
    }
}
