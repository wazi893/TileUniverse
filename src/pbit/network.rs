//! P-bit Network with Ising Model Coupling
//!
//! A network of coupled p-bits that evolve according to Ising model dynamics.
//! The energy function is:
//!
//! E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
//!
//! where J is the coupling matrix, h is the bias vector, and s ∈ {-1, +1}.

use super::ising::IsingProblem;
use super::pbit::PBit;
use crate::qec::noise::SimpleRng;

/// Configuration for a p-bit network
#[derive(Debug, Clone)]
pub struct PBitConfig {
    /// Number of p-bits
    pub n_pbits: usize,
    /// Inverse temperature (higher = more deterministic)
    pub beta: f64,
    /// Update order: Sequential (Gibbs) or Random
    pub update_order: UpdateOrder,
    /// Seed for RNG
    pub seed: u64,
}

/// Order in which p-bits are updated each step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOrder {
    /// Update p-bits 0, 1, 2, ... in sequence
    Sequential,
    /// Update p-bits in random order each step
    Random,
    /// Update all p-bits in parallel (less accurate but faster)
    Parallel,
}

impl PBitConfig {
    /// Create a new configuration
    pub fn new(n_pbits: usize) -> Self {
        Self {
            n_pbits,
            beta: 1.0,
            update_order: UpdateOrder::Sequential,
            seed: 42,
        }
    }

    /// Set inverse temperature
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set update order
    pub fn with_update_order(mut self, order: UpdateOrder) -> Self {
        self.update_order = order;
        self
    }

    /// Set RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// A network of coupled probabilistic bits
#[derive(Debug)]
pub struct PBitNetwork {
    /// Configuration
    config: PBitConfig,
    /// Array of p-bits
    pbits: Vec<PBit>,
    /// Coupling matrix J (symmetric, stored as dense for now)
    /// J[i * n + j] = coupling between p-bit i and j
    couplings: Vec<f64>,
    /// Bias vector h
    biases: Vec<f64>,
    /// Random number generator
    rng: SimpleRng,
    /// Current energy
    energy: f64,
    /// Best energy found
    best_energy: f64,
    /// Best state found
    best_state: Vec<u8>,
    /// Number of steps taken
    steps: u64,
    /// Update order buffer (for random ordering)
    update_indices: Vec<usize>,
}

impl PBitNetwork {
    /// Create a new p-bit network
    pub fn new(config: PBitConfig) -> Self {
        let n = config.n_pbits;
        let rng = SimpleRng::new(config.seed);

        Self {
            pbits: vec![PBit::new(); n],
            couplings: vec![0.0; n * n],
            biases: vec![0.0; n],
            rng,
            energy: 0.0,
            best_energy: f64::INFINITY,
            best_state: vec![0; n],
            steps: 0,
            update_indices: (0..n).collect(),
            config,
        }
    }

    /// Load an Ising problem into the network
    pub fn set_problem(&mut self, problem: &IsingProblem) {
        assert_eq!(problem.n, self.config.n_pbits);

        // Copy couplings and biases
        self.couplings.copy_from_slice(&problem.j);
        self.biases.copy_from_slice(&problem.h);

        // Update p-bit biases
        for (i, pbit) in self.pbits.iter_mut().enumerate() {
            pbit.bias = self.biases[i];
        }

        // Recalculate energy
        self.recalculate_energy();
    }

    /// Set coupling between p-bits i and j
    pub fn set_coupling(&mut self, i: usize, j: usize, value: f64) {
        let n = self.config.n_pbits;
        debug_assert!(i < n && j < n);
        self.couplings[i * n + j] = value;
        self.couplings[j * n + i] = value; // Symmetric
    }

    /// Get coupling between p-bits i and j
    #[inline]
    pub fn coupling(&self, i: usize, j: usize) -> f64 {
        self.couplings[i * self.config.n_pbits + j]
    }

    /// Set bias for p-bit i
    pub fn set_bias(&mut self, i: usize, value: f64) {
        self.biases[i] = value;
        self.pbits[i].bias = value;
    }

    /// Get the number of p-bits
    #[inline]
    pub fn n_pbits(&self) -> usize {
        self.config.n_pbits
    }

    /// Get current state as a vector of 0/1
    pub fn state(&self) -> Vec<u8> {
        self.pbits.iter().map(|p| p.state).collect()
    }

    /// Get current state as spins (-1/+1)
    pub fn spins(&self) -> Vec<i8> {
        self.pbits.iter().map(|p| p.spin()).collect()
    }

    /// Set the state directly
    pub fn set_state(&mut self, state: &[u8]) {
        assert_eq!(state.len(), self.config.n_pbits);
        for (pbit, &s) in self.pbits.iter_mut().zip(state.iter()) {
            pbit.set_state(s);
        }
        self.recalculate_energy();
    }

    /// Randomize all p-bit states
    pub fn randomize(&mut self) {
        for pbit in &mut self.pbits {
            pbit.state = if self.rng.next_f64() < 0.5 { 0 } else { 1 };
        }
        self.recalculate_energy();
    }

    /// Compute the local field (input) for p-bit i
    ///
    /// I_i = h_i + Σⱼ J_ij s_j
    #[inline]
    fn local_field(&self, i: usize) -> f64 {
        let n = self.config.n_pbits;
        let mut field = self.biases[i];

        for j in 0..n {
            if i != j {
                field += self.couplings[i * n + j] * self.pbits[j].spin_f64();
            }
        }

        field
    }

    /// Perform one Gibbs sampling step
    ///
    /// Updates each p-bit once according to the configured order
    pub fn step(&mut self) {
        let n = self.config.n_pbits;
        let beta = self.config.beta;

        match self.config.update_order {
            UpdateOrder::Sequential => {
                for i in 0..n {
                    let field = self.local_field(i);
                    self.pbits[i].update_with_beta(field, beta, &mut self.rng);
                }
            }
            UpdateOrder::Random => {
                // Fisher-Yates shuffle
                for i in (1..n).rev() {
                    let j = self.rng.next_usize(i + 1);
                    self.update_indices.swap(i, j);
                }
                for &i in &self.update_indices {
                    let field = self.local_field(i);
                    self.pbits[i].update_with_beta(field, beta, &mut self.rng);
                }
            }
            UpdateOrder::Parallel => {
                // Compute all fields first, then update all at once
                let fields: Vec<f64> = (0..n).map(|i| self.local_field(i)).collect();
                for (i, field) in fields.into_iter().enumerate() {
                    self.pbits[i].update_with_beta(field, beta, &mut self.rng);
                }
            }
        }

        self.steps += 1;
        self.update_best();
    }

    /// Run multiple steps
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Recalculate the Ising energy from scratch
    fn recalculate_energy(&mut self) {
        let n = self.config.n_pbits;
        let mut energy = 0.0;

        // Coupling term: -Σᵢ<ⱼ Jᵢⱼ sᵢ sⱼ
        for i in 0..n {
            for j in (i + 1)..n {
                energy -=
                    self.couplings[i * n + j] * self.pbits[i].spin_f64() * self.pbits[j].spin_f64();
            }
        }

        // Bias term: -Σᵢ hᵢ sᵢ
        for i in 0..n {
            energy -= self.biases[i] * self.pbits[i].spin_f64();
        }

        self.energy = energy;
    }

    /// Update best solution if current is better
    fn update_best(&mut self) {
        self.recalculate_energy();
        if self.energy < self.best_energy {
            self.best_energy = self.energy;
            self.best_state = self.state();
        }
    }

    /// Get current energy
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// Get best energy found
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get best state found
    pub fn best_state(&self) -> &[u8] {
        &self.best_state
    }

    /// Get number of steps taken
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Get reference to a p-bit
    pub fn pbit(&self, i: usize) -> &PBit {
        &self.pbits[i]
    }

    /// Set inverse temperature (for annealing)
    pub fn set_beta(&mut self, beta: f64) {
        self.config.beta = beta;
    }

    /// Get current inverse temperature
    pub fn beta(&self) -> f64 {
        self.config.beta
    }

    /// Reset statistics (best energy, step count) while keeping state
    pub fn reset_stats(&mut self) {
        self.best_energy = f64::INFINITY;
        self.best_state = vec![0; self.config.n_pbits];
        self.steps = 0;
        self.update_best();
    }

    /// Get magnetization (average spin)
    pub fn magnetization(&self) -> f64 {
        let sum: i32 = self.pbits.iter().map(|p| p.spin() as i32).sum();
        sum as f64 / self.config.n_pbits as f64
    }

    /// Get flip rate (fraction of p-bits that changed in last step)
    /// Useful for monitoring equilibration
    pub fn measure_flip_rate(&mut self) -> f64 {
        let old_state = self.state();
        self.step();
        let new_state = self.state();

        let flips = old_state
            .iter()
            .zip(new_state.iter())
            .filter(|(a, b)| a != b)
            .count();

        flips as f64 / self.config.n_pbits as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = PBitConfig::new(10);
        let network = PBitNetwork::new(config);
        assert_eq!(network.n_pbits(), 10);
    }

    #[test]
    fn test_coupling_symmetry() {
        let config = PBitConfig::new(5);
        let mut network = PBitNetwork::new(config);

        network.set_coupling(1, 3, 0.5);
        assert!((network.coupling(1, 3) - 0.5).abs() < 1e-10);
        assert!((network.coupling(3, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ferromagnetic_ground_state() {
        // Ferromagnetic couplings should prefer aligned spins
        let config = PBitConfig::new(4).with_beta(5.0);
        let mut network = PBitNetwork::new(config);

        // All-to-all ferromagnetic coupling
        for i in 0..4 {
            for j in (i + 1)..4 {
                network.set_coupling(i, j, 1.0); // Positive = ferromagnetic
            }
        }

        // Run many steps at high beta (low temperature)
        network.run(1000);

        // Should find ground state (all same spin)
        let spins = network.spins();
        let all_up = spins.iter().all(|&s| s == 1);
        let all_down = spins.iter().all(|&s| s == -1);
        assert!(
            all_up || all_down,
            "Expected aligned spins, got {:?}",
            spins
        );
    }

    #[test]
    fn test_antiferromagnetic_alternating() {
        // 1D chain with antiferromagnetic coupling
        let config = PBitConfig::new(4).with_beta(5.0);
        let mut network = PBitNetwork::new(config);

        // Nearest-neighbor antiferromagnetic
        for i in 0..3 {
            network.set_coupling(i, i + 1, -1.0);
        }

        network.run(1000);

        // Should alternate: +-+- or -+-+
        let spins = network.spins();
        let alternating = (0..3).all(|i| spins[i] != spins[i + 1]);
        assert!(alternating, "Expected alternating, got {:?}", spins);
    }

    #[test]
    fn test_energy_calculation() {
        let config = PBitConfig::new(2);
        let mut network = PBitNetwork::new(config);

        network.set_coupling(0, 1, 1.0);
        network.set_state(&[1, 1]); // Both +1

        // E = -J * s0 * s1 = -1 * 1 * 1 = -1
        assert!((network.energy() - (-1.0)).abs() < 1e-10);

        network.set_state(&[0, 1]); // -1 and +1
        // E = -1 * (-1) * 1 = 1
        assert!((network.energy() - 1.0).abs() < 1e-10);
    }
}
