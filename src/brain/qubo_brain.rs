//! QUBO Brain - Probabilistic bit (p-bit) based decision making
//!
//! This module implements a brain that uses QUBO (Quadratic Unconstrained Binary Optimization)
//! formulation with p-bit annealing for making foraging decisions.
//!
//! # Architecture
//!
//! The QUBO brain maps sensor readings to an Ising problem whose ground state
//! encodes the optimal action. The key insight is that action selection becomes
//! an optimization problem that p-bits can solve via thermal sampling.
//!
//! ## Variables
//!
//! - Bits 0-4: One-hot action encoding (Up, Down, Left, Right, Stay)
//! - Bits 5+: Auxiliary/hidden variables for richer computation
//!
//! ## Energy Function
//!
//! E(x) = x^T Q x + b^T x + lambda * (sum(x[0:5]) - 1)^2
//!
//! where:
//! - Q is the coupling matrix (learned/evolved)
//! - b is the bias vector (computed from sensor features)
//! - lambda enforces one-hot constraint on action bits
//!
//! # Genome Representation
//!
//! The evolvable genome consists of:
//! 1. `couplings`: Upper triangle of Q matrix (n*(n+1)/2 entries)
//! 2. `sensor_weights`: Maps 20 expanded features to n_vars biases

use crate::brain::classical::Move;
use crate::pbit::{IsingProblem, PBitConfig, PBitNetwork, UpdateOrder};
use crate::qec::noise::SimpleRng;
use crate::quantum_brain::SensorReadings;

/// Decision budget for QUBO brain - controls compute per decision
#[derive(Debug, Clone, Copy)]
pub struct DecisionBudget {
    /// Number of Gibbs sampling sweeps to perform
    pub sweeps: usize,
    /// Inverse temperature (beta) for annealing
    pub beta: f64,
}

impl Default for DecisionBudget {
    fn default() -> Self {
        Self {
            sweeps: 200,
            beta: 8.0,
        }
    }
}

/// Ablation variants for studying QUBO brain components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuboAblation {
    /// Full model with all features
    Full,
    /// Only biases from sensors, no couplings
    BiasesOnly,
    /// Only couplings, fixed bias
    CouplingsOnly { fixed_bias: i32 },
    /// No one-hot constraint penalty
    NoConstraint,
    /// Sparse local connectivity (only nearby bits coupled)
    SparseLocal { radius: usize },
    /// Linear sensor features only (no quadratic expansion)
    LinearSensors,
}

/// Number of expanded sensor features
/// 6 raw + 4 gradients + 6 squares + 4 products = 20
pub const N_FEATURES: usize = 20;

/// Number of action bits (5 actions: up, down, left, right, stay)
pub const N_ACTION_BITS: usize = 5;

/// QUBO Brain using p-bit annealing for decision making
#[derive(Clone, Debug)]
pub struct QuboBrain {
    /// Number of binary variables (action bits + auxiliary)
    pub n_vars: usize,

    /// Coupling weights - upper triangle only: n*(n+1)/2 entries
    /// Index (i,j) where i <= j stored at position tri_idx(i, j)
    pub couplings: Vec<f32>,

    /// Sensor-to-bias weights: n_vars rows x N_FEATURES columns
    /// Each variable's bias is computed as dot(sensor_weights[var], features)
    pub sensor_weights: Vec<f32>,

    /// One-hot constraint penalty (lambda)
    /// Higher values enforce stricter one-hot on action bits
    pub one_hot_penalty: f32,

    /// Ablation mode for experiments
    pub ablation: QuboAblation,

    // ========== Preallocated buffers (not part of genome) ==========
    /// Buffer for computed biases
    biases_buffer: Vec<f64>,

    /// Buffer for expanded features
    features_buffer: Vec<f32>,
}

impl QuboBrain {
    /// Create a new QUBO brain with zero-initialized weights
    pub fn new(n_vars: usize, one_hot_penalty: f32) -> Self {
        assert!(n_vars >= N_ACTION_BITS, "Need at least 5 vars for actions");

        let n_couplings = n_vars * (n_vars + 1) / 2;
        let n_sensor_weights = n_vars * N_FEATURES;

        Self {
            n_vars,
            couplings: vec![0.0; n_couplings],
            sensor_weights: vec![0.0; n_sensor_weights],
            one_hot_penalty,
            ablation: QuboAblation::Full,
            biases_buffer: vec![0.0; n_vars],
            features_buffer: vec![0.0; N_FEATURES],
        }
    }

    /// Create a QUBO brain with random initialization
    ///
    /// Uses STRONG fixed directional weights - no noise on the key directional sensors.
    /// Only couplings are random/evolvable. This ensures the brain can always sense direction.
    pub fn random(n_vars: usize, one_hot_penalty: f32, rng: &mut SimpleRng) -> Self {
        let mut brain = Self::new(n_vars, one_hot_penalty);

        // Initialize couplings with small random values (these are evolvable)
        let coupling_scale = 0.5;
        for c in &mut brain.couplings {
            *c = (rng.next_f64() as f32 * 2.0 - 1.0) * coupling_scale;
        }

        // Initialize sensor weights with FIXED strong directional bias
        // Key insight: directional weights must be strong enough to overcome
        // one-hot penalty even with weak sensor signals (e.g., heat=50 vs 40)
        //
        // Feature layout:
        //   0: heat_n, 1: heat_s, 2: heat_e, 3: heat_w, 4: heat_local, 5: charge_local
        //   6: n-s gradient, 7: e-w gradient, 8: n-local, 9: e-local
        //
        // Action layout:
        //   0: Up, 1: Down, 2: Left, 3: Right, 4: Stay

        let directional_strength = 20.0; // Very strong to handle subtle gradients
        let gradient_strength = 30.0; // Gradients are more reliable

        for var in 0..n_vars {
            for feat in 0..N_FEATURES {
                let idx = var * N_FEATURES + feat;

                // FIXED directional bias for action bits - no noise on key features
                let weight = if var < N_ACTION_BITS {
                    match (var, feat) {
                        // Up (var 0): prefer north
                        (0, 0) => directional_strength, // heat_n
                        (0, 6) => gradient_strength,    // n-s gradient

                        // Down (var 1): prefer south
                        (1, 1) => directional_strength, // heat_s
                        (1, 6) => -gradient_strength,   // n-s gradient

                        // Left (var 2): prefer west
                        (2, 3) => directional_strength, // heat_w
                        (2, 7) => -gradient_strength,   // e-w gradient

                        // Right (var 3): prefer east
                        (3, 2) => directional_strength, // heat_e
                        (3, 7) => gradient_strength,    // e-w gradient

                        // Stay (var 4): prefer high local
                        (4, 4) => directional_strength, // heat_local
                        (4, 5) => directional_strength * 0.5, // charge_local

                        // Non-key features get small random noise
                        _ => (rng.next_f64() as f32 * 2.0 - 1.0) * 0.5,
                    }
                } else {
                    // Auxiliary bits (if any) get random weights
                    (rng.next_f64() as f32 * 2.0 - 1.0) * 1.0
                };

                brain.sensor_weights[idx] = weight;
            }
        }

        brain
    }

    /// Create a QUBO brain with specific ablation configuration
    pub fn with_ablation(n_vars: usize, one_hot_penalty: f32, ablation: QuboAblation) -> Self {
        let mut brain = Self::new(n_vars, one_hot_penalty);
        brain.ablation = ablation;
        brain
    }

    /// Get the triangular index for coupling (i, j) where i <= j
    #[inline]
    pub fn tri_idx(i: usize, j: usize) -> usize {
        debug_assert!(i <= j, "tri_idx requires i <= j");
        // Row-major upper triangle: sum of (n-k) for k=0..i, plus (j-i)
        // = i*n - i*(i+1)/2 + j
        // But simpler formula: j*(j+1)/2 + i (column-major upper tri)
        j * (j + 1) / 2 + i
    }

    /// Get coupling J_ij (symmetric)
    #[inline]
    pub fn get_coupling(&self, i: usize, j: usize) -> f32 {
        if i <= j {
            self.couplings[Self::tri_idx(i, j)]
        } else {
            self.couplings[Self::tri_idx(j, i)]
        }
    }

    /// Set coupling J_ij (symmetric)
    #[inline]
    pub fn set_coupling(&mut self, i: usize, j: usize, val: f32) {
        if i <= j {
            self.couplings[Self::tri_idx(i, j)] = val;
        } else {
            self.couplings[Self::tri_idx(j, i)] = val;
        }
    }

    /// Get sensor weight for variable `var` and feature `feat`
    #[inline]
    pub fn get_sensor_weight(&self, var: usize, feat: usize) -> f32 {
        self.sensor_weights[var * N_FEATURES + feat]
    }

    /// Set sensor weight for variable `var` and feature `feat`
    #[inline]
    pub fn set_sensor_weight(&mut self, var: usize, feat: usize, val: f32) {
        self.sensor_weights[var * N_FEATURES + feat] = val;
    }

    /// Expand sensor readings to feature vector
    ///
    /// Features (20 total):
    /// 0-5: Raw sensors (heat_n, heat_s, heat_e, heat_w, heat_local, charge_local)
    /// 6-9: Gradients (n-s, e-w, n-local, e-local)
    /// 10-15: Squares (heat_n^2, heat_s^2, heat_e^2, heat_w^2, heat_local^2, charge_local^2)
    /// 16-19: Products (n*s, e*w, heat*charge, max_heat*min_heat)
    pub fn expand_features(&mut self, sensors: &SensorReadings) {
        let f = &mut self.features_buffer;

        // Normalize to [0, 1] range
        let norm = |v: u32| (v as f32) / 255.0;

        let hn = norm(sensors.heat_n);
        let hs = norm(sensors.heat_s);
        let he = norm(sensors.heat_e);
        let hw = norm(sensors.heat_w);
        let hl = norm(sensors.heat_local);
        let cl = norm(sensors.charge_local);

        // Raw sensors (0-5)
        f[0] = hn;
        f[1] = hs;
        f[2] = he;
        f[3] = hw;
        f[4] = hl;
        f[5] = cl;

        // Gradients (6-9)
        f[6] = hn - hs; // North-South gradient
        f[7] = he - hw; // East-West gradient
        f[8] = hn - hl; // North vs local
        f[9] = he - hl; // East vs local

        // Squares (10-15) - for nonlinear response
        f[10] = hn * hn;
        f[11] = hs * hs;
        f[12] = he * he;
        f[13] = hw * hw;
        f[14] = hl * hl;
        f[15] = cl * cl;

        // Products (16-19) - interaction terms
        f[16] = hn * hs; // Vertical total
        f[17] = he * hw; // Horizontal total
        f[18] = hl * cl; // Local heat-charge interaction
        let max_h = hn.max(hs).max(he).max(hw);
        let min_h = hn.min(hs).min(he).min(hw);
        f[19] = max_h * min_h; // Contrast measure
    }

    /// Compute biases from expanded features
    ///
    /// For each variable i: bias[i] = sum_j(sensor_weights[i,j] * features[j])
    pub fn compute_biases(&mut self, sensors: &SensorReadings) {
        // First expand features
        self.expand_features(sensors);

        // Handle ablations
        match self.ablation {
            QuboAblation::BiasesOnly | QuboAblation::Full | QuboAblation::NoConstraint => {
                // Compute biases normally
                for var in 0..self.n_vars {
                    let mut bias = 0.0f64;
                    for feat in 0..N_FEATURES {
                        bias += self.get_sensor_weight(var, feat) as f64
                            * self.features_buffer[feat] as f64;
                    }
                    self.biases_buffer[var] = bias;
                }
            }
            QuboAblation::CouplingsOnly { fixed_bias } => {
                // Use fixed bias for all variables
                for var in 0..self.n_vars {
                    self.biases_buffer[var] = fixed_bias as f64;
                }
            }
            QuboAblation::SparseLocal { .. } => {
                // Normal bias computation
                for var in 0..self.n_vars {
                    let mut bias = 0.0f64;
                    for feat in 0..N_FEATURES {
                        bias += self.get_sensor_weight(var, feat) as f64
                            * self.features_buffer[feat] as f64;
                    }
                    self.biases_buffer[var] = bias;
                }
            }
            QuboAblation::LinearSensors => {
                // Only use first 6 (raw) features
                for var in 0..self.n_vars {
                    let mut bias = 0.0f64;
                    for feat in 0..6 {
                        bias += self.get_sensor_weight(var, feat) as f64
                            * self.features_buffer[feat] as f64;
                    }
                    self.biases_buffer[var] = bias;
                }
            }
        }
    }

    /// Build the Ising problem from current genome and sensor-derived biases
    ///
    /// Includes one-hot constraint penalty on action bits 0-4
    pub fn build_problem(&self) -> IsingProblem {
        let mut problem = IsingProblem::new(self.n_vars);

        // Set biases (h_i)
        for i in 0..self.n_vars {
            problem.set_bias(i, self.biases_buffer[i]);
        }

        // Set couplings (J_ij)
        match self.ablation {
            QuboAblation::BiasesOnly => {
                // No couplings
            }
            QuboAblation::SparseLocal { radius } => {
                // Only couple variables within radius
                for i in 0..self.n_vars {
                    for j in (i + 1)..self.n_vars {
                        if j - i <= radius {
                            problem.set_coupling(i, j, self.get_coupling(i, j) as f64);
                        }
                    }
                }
            }
            _ => {
                // Full couplings
                for i in 0..self.n_vars {
                    for j in (i + 1)..self.n_vars {
                        problem.set_coupling(i, j, self.get_coupling(i, j) as f64);
                    }
                }
            }
        }

        // Add one-hot constraint penalty on action bits 0-4
        // Penalty: lambda * (x0 + x1 + x2 + x3 + x4 - 1)^2
        //
        // QUBO expansion: λ * (Σx - 1)² = λ * [2*Σ_{i<j}x_i·x_j - Σx_i + 1]
        //
        // Converting to Ising (x = (s+1)/2):
        //
        // Off-diagonal term 2λ·x_i·x_j = 2λ·(s_i+1)(s_j+1)/4 = λ/2·(1 + s_i + s_j + s_i·s_j)
        //   To ADD this to energy E (where E = -Σ J_ij s_i s_j - Σ h_i s_i):
        //   - s_i·s_j term: want +λ/2·s_i·s_j in energy, so need -J_ij = +λ/2 → J_ij = -λ/2
        //   - s_i term: want +λ/2·s_i in energy, so need -h_i = +λ/2 → h contribution = -λ/2
        //   - Each bit is in 4 pairs, so total h contribution from off-diagonal = -4·λ/2 = -2λ
        //
        // Diagonal term -λ·x_i = -λ·(s_i+1)/2 = -λ/2·s_i - λ/2
        //   - s_i term: want -λ/2·s_i in energy, so need -h_i = -λ/2 → h contribution = +λ/2
        //
        // Net h[i] from one-hot: +λ/2 - 2λ = -3λ/2 (SUBTRACT from bias)
        // J_ij from one-hot: -λ/2 (anti-ferromagnetic)

        if self.ablation != QuboAblation::NoConstraint {
            let lambda = self.one_hot_penalty as f64;

            // CORRECTED one-hot constraint:
            // Diagonal: Net effect is -3λ/2 per action bit
            for i in 0..N_ACTION_BITS {
                problem.h[i] -= 1.5 * lambda;
            }

            // Off-diagonal: J_ij = -λ/2 (anti-ferromagnetic)
            for i in 0..N_ACTION_BITS {
                for j in (i + 1)..N_ACTION_BITS {
                    let current = problem.coupling(i, j);
                    problem.set_coupling(i, j, current - 0.5 * lambda);
                }
            }
        }

        problem
    }

    /// Make a decision given sensor readings
    ///
    /// Returns the chosen action based on p-bit annealing
    pub fn decide(
        &mut self,
        sensors: &SensorReadings,
        rng: &mut SimpleRng,
        budget: &DecisionBudget,
    ) -> Move {
        // Compute biases from sensors
        self.compute_biases(sensors);

        // Build the Ising problem
        let problem = self.build_problem();

        // Create p-bit network
        let config = PBitConfig::new(self.n_vars)
            .with_beta(budget.beta)
            .with_update_order(UpdateOrder::Sequential)
            .with_seed(rng.next_u64());

        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        // Randomize initial state
        network.randomize();

        // Run Gibbs sampling
        network.run(budget.sweeps as u64);

        // Get best state found
        let best_state = network.best_state();

        // Decode action from bits 0-4
        self.decode_action(best_state)
    }

    /// Decode action from binary state
    ///
    /// Uses one-hot encoding on bits 0-4:
    /// - Bit 0: Up
    /// - Bit 1: Down
    /// - Bit 2: Left
    /// - Bit 3: Right
    /// - Bit 4: Stay
    ///
    /// If multiple bits are set (constraint violation), pick highest priority
    /// If no bits are set, default to Stay
    pub fn decode_action(&self, state: &[u8]) -> Move {
        // Check action bits in priority order
        if state.len() > 0 && state[0] == 1 {
            Move::Up
        } else if state.len() > 1 && state[1] == 1 {
            Move::Down
        } else if state.len() > 2 && state[2] == 1 {
            Move::Left
        } else if state.len() > 3 && state[3] == 1 {
            Move::Right
        } else {
            Move::Stay
        }
    }

    /// Create mutated offspring
    ///
    /// Only mutates couplings - sensor weights are fixed to preserve directional sensing.
    pub fn reproduce(&self, rng: &mut SimpleRng, mutation_rate: f32, mutation_scale: f32) -> Self {
        let mut child = self.clone();

        // Mutate couplings (the main evolvable genome)
        for c in &mut child.couplings {
            if rng.next_f64() < mutation_rate as f64 {
                *c += (rng.next_f64() as f32 * 2.0 - 1.0) * mutation_scale;
                *c = c.clamp(-5.0, 5.0);
            }
        }

        // DON'T mutate sensor weights - they encode fixed directional structure
        // This ensures the brain always knows how to sense direction
        // Evolution happens through couplings which modulate action selection

        // Occasionally mutate one-hot penalty
        if rng.next_f64() < 0.05 {
            child.one_hot_penalty *= 0.9 + rng.next_f64() as f32 * 0.2;
            child.one_hot_penalty = child.one_hot_penalty.clamp(0.1, 10.0);
        }

        child
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.couplings.len() + self.sensor_weights.len()
    }

    /// Get computed biases (for debugging)
    pub fn get_biases(&self) -> &[f64] {
        &self.biases_buffer
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.couplings.len() * std::mem::size_of::<f32>()
            + self.sensor_weights.len() * std::mem::size_of::<f32>()
            + self.biases_buffer.len() * std::mem::size_of::<f64>()
            + self.features_buffer.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_brain_creation() {
        let brain = QuboBrain::new(8, 2.0);

        assert_eq!(brain.n_vars, 8);
        assert_eq!(brain.couplings.len(), 8 * 9 / 2); // 36
        assert_eq!(brain.sensor_weights.len(), 8 * N_FEATURES); // 160
        assert_eq!(brain.one_hot_penalty, 2.0);
    }

    #[test]
    fn test_triangular_indexing() {
        // Test tri_idx produces unique indices for upper triangle
        let n = 5;
        let mut indices = Vec::new();

        for i in 0..n {
            for j in i..n {
                let idx = QuboBrain::tri_idx(i, j);
                assert!(
                    !indices.contains(&idx),
                    "Duplicate index {} for ({}, {})",
                    idx,
                    i,
                    j
                );
                indices.push(idx);
            }
        }

        // Should have n*(n+1)/2 unique indices
        assert_eq!(indices.len(), n * (n + 1) / 2);
    }

    #[test]
    fn test_coupling_symmetry() {
        let mut brain = QuboBrain::new(8, 2.0);

        brain.set_coupling(2, 5, 0.75);
        assert!((brain.get_coupling(2, 5) - 0.75).abs() < 1e-6);
        assert!((brain.get_coupling(5, 2) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_feature_expansion() {
        let mut brain = QuboBrain::new(8, 2.0);

        let sensors = SensorReadings {
            heat_n: 255,
            heat_s: 0,
            heat_e: 128,
            heat_w: 64,
            heat_local: 100,
            charge_local: 50,
        };

        brain.expand_features(&sensors);

        // Check raw features
        assert!((brain.features_buffer[0] - 1.0).abs() < 1e-4); // heat_n normalized
        assert!((brain.features_buffer[1] - 0.0).abs() < 1e-4); // heat_s normalized

        // Check gradient
        assert!((brain.features_buffer[6] - 1.0).abs() < 1e-4); // N-S gradient = 1.0 - 0.0
    }

    #[test]
    fn test_random_brain() {
        let mut rng = SimpleRng::new(42);
        let brain = QuboBrain::random(8, 2.0, &mut rng);

        // Should have non-zero weights
        let non_zero_couplings = brain.couplings.iter().filter(|&&c| c.abs() > 1e-6).count();
        let non_zero_weights = brain
            .sensor_weights
            .iter()
            .filter(|&&w| w.abs() > 1e-6)
            .count();

        assert!(non_zero_couplings > 0, "Should have non-zero couplings");
        assert!(non_zero_weights > 0, "Should have non-zero sensor weights");
    }

    #[test]
    fn test_decision_determinism() {
        let mut rng = SimpleRng::new(42);
        let brain = QuboBrain::random(8, 2.0, &mut rng);

        let sensors = SensorReadings {
            heat_n: 200,
            heat_s: 50,
            heat_e: 100,
            heat_w: 100,
            heat_local: 75,
            charge_local: 25,
        };

        let budget = DecisionBudget {
            sweeps: 100,
            beta: 5.0,
        };

        // Same seed should give same result
        let mut brain1 = brain.clone();
        let mut rng1 = SimpleRng::new(12345);
        let move1 = brain1.decide(&sensors, &mut rng1, &budget);

        let mut brain2 = brain.clone();
        let mut rng2 = SimpleRng::new(12345);
        let move2 = brain2.decide(&sensors, &mut rng2, &budget);

        assert_eq!(move1, move2);
    }

    #[test]
    fn test_decode_action() {
        let brain = QuboBrain::new(8, 2.0);

        // One-hot encodings
        assert_eq!(brain.decode_action(&[1, 0, 0, 0, 0, 0, 0, 0]), Move::Up);
        assert_eq!(brain.decode_action(&[0, 1, 0, 0, 0, 0, 0, 0]), Move::Down);
        assert_eq!(brain.decode_action(&[0, 0, 1, 0, 0, 0, 0, 0]), Move::Left);
        assert_eq!(brain.decode_action(&[0, 0, 0, 1, 0, 0, 0, 0]), Move::Right);
        assert_eq!(brain.decode_action(&[0, 0, 0, 0, 1, 0, 0, 0]), Move::Stay);

        // All zeros -> Stay
        assert_eq!(brain.decode_action(&[0, 0, 0, 0, 0, 0, 0, 0]), Move::Stay);

        // Multiple bits (priority: Up > Down > Left > Right > Stay)
        assert_eq!(brain.decode_action(&[1, 1, 0, 0, 0, 0, 0, 0]), Move::Up);
        assert_eq!(brain.decode_action(&[0, 1, 1, 0, 0, 0, 0, 0]), Move::Down);
    }

    #[test]
    fn test_reproduction() {
        let mut rng = SimpleRng::new(42);
        let parent = QuboBrain::random(8, 2.0, &mut rng);

        let child = parent.reproduce(&mut rng, 1.0, 0.1); // 100% mutation rate

        // Child should be different from parent
        let mut different = false;
        for (p, c) in parent.couplings.iter().zip(child.couplings.iter()) {
            if (p - c).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        assert!(different, "Child should have mutated weights");
    }

    #[test]
    fn test_parameter_count() {
        let brain = QuboBrain::new(8, 2.0);

        // Couplings: 8*9/2 = 36
        // Sensor weights: 8 * 20 = 160
        // Total: 196
        assert_eq!(brain.parameter_count(), 36 + 160);
    }

    #[test]
    fn test_ablation_biases_only() {
        let mut rng = SimpleRng::new(42);
        let mut brain = QuboBrain::random(8, 2.0, &mut rng);
        brain.ablation = QuboAblation::BiasesOnly;

        let sensors = SensorReadings {
            heat_n: 200,
            heat_s: 50,
            heat_e: 100,
            heat_w: 100,
            heat_local: 75,
            charge_local: 25,
        };

        brain.compute_biases(&sensors);
        let problem = brain.build_problem();

        // All couplings should be zero (except one-hot penalty)
        let mut found_non_penalty_coupling = false;
        for i in N_ACTION_BITS..brain.n_vars {
            for j in (i + 1)..brain.n_vars {
                if problem.coupling(i, j).abs() > 1e-6 {
                    found_non_penalty_coupling = true;
                }
            }
        }
        assert!(
            !found_non_penalty_coupling,
            "BiasesOnly should have no non-penalty couplings"
        );
    }

    #[test]
    fn test_ablation_no_constraint() {
        let mut rng = SimpleRng::new(42);
        let mut brain = QuboBrain::random(8, 2.0, &mut rng);
        brain.ablation = QuboAblation::NoConstraint;

        let sensors = SensorReadings::default();
        brain.compute_biases(&sensors);
        let problem = brain.build_problem();

        // Action bit couplings should be from genome only, not penalty
        // With zero sensor inputs and random genome, we should see genome couplings
        // but not the +2*lambda penalty pattern specifically

        // Just verify it runs without error
        assert_eq!(problem.n, 8);
    }
}
