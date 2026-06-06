//! Sampling Algorithms for P-bit Networks
//!
//! Provides various sampling strategies:
//! - Gibbs sampling (basic MCMC)
//! - Simulated annealing
//! - Parallel tempering (replica exchange)

use super::network::PBitNetwork;
use crate::qec::noise::SimpleRng;

/// Gibbs sampler wrapper for p-bit networks
pub struct GibbsSampler<'a> {
    network: &'a mut PBitNetwork,
    /// Number of equilibration steps
    pub equilibration_steps: u64,
    /// Number of sampling steps
    pub sampling_steps: u64,
    /// Sample every N steps (for decorrelation)
    pub thin: u64,
}

impl<'a> GibbsSampler<'a> {
    pub fn new(network: &'a mut PBitNetwork) -> Self {
        Self {
            network,
            equilibration_steps: 1000,
            sampling_steps: 10000,
            thin: 1,
        }
    }

    /// Set equilibration steps
    pub fn with_equilibration(mut self, steps: u64) -> Self {
        self.equilibration_steps = steps;
        self
    }

    /// Set sampling steps
    pub fn with_sampling(mut self, steps: u64) -> Self {
        self.sampling_steps = steps;
        self
    }

    /// Set thinning interval
    pub fn with_thin(mut self, thin: u64) -> Self {
        self.thin = thin;
        self
    }

    /// Run sampling and return collected samples
    pub fn run(&mut self) -> Vec<Vec<u8>> {
        // Equilibration
        self.network.run(self.equilibration_steps);

        // Collect samples
        let mut samples = Vec::new();
        for step in 0..self.sampling_steps {
            self.network.step();
            if step % self.thin == 0 {
                samples.push(self.network.state());
            }
        }

        samples
    }

    /// Run and return only the best solution
    pub fn find_minimum(&mut self) -> (Vec<u8>, f64) {
        self.network.reset_stats();
        self.network
            .run(self.equilibration_steps + self.sampling_steps);
        (
            self.network.best_state().to_vec(),
            self.network.best_energy(),
        )
    }
}

/// Annealing schedule for simulated annealing
#[derive(Debug, Clone)]
pub enum AnnealingSchedule {
    /// Linear: beta(t) = beta_min + (beta_max - beta_min) * t / T
    Linear { beta_min: f64, beta_max: f64 },
    /// Exponential: beta(t) = beta_min * (beta_max / beta_min)^(t/T)
    Exponential { beta_min: f64, beta_max: f64 },
    /// Logarithmic: beta(t) = beta_0 * ln(1 + t) (slow but guaranteed)
    Logarithmic { beta_0: f64 },
    /// Custom schedule from a function
    Custom(Vec<f64>),
}

impl AnnealingSchedule {
    /// Default linear schedule
    pub fn linear() -> Self {
        Self::Linear {
            beta_min: 0.1,
            beta_max: 10.0,
        }
    }

    /// Default exponential schedule
    pub fn exponential() -> Self {
        Self::Exponential {
            beta_min: 0.1,
            beta_max: 10.0,
        }
    }

    /// Compute beta at step t out of total_steps
    pub fn beta_at(&self, t: u64, total_steps: u64) -> f64 {
        let progress = t as f64 / total_steps as f64;

        match self {
            AnnealingSchedule::Linear { beta_min, beta_max } => {
                beta_min + (beta_max - beta_min) * progress
            }
            AnnealingSchedule::Exponential { beta_min, beta_max } => {
                beta_min * (beta_max / beta_min).powf(progress)
            }
            AnnealingSchedule::Logarithmic { beta_0 } => beta_0 * (1.0 + t as f64).ln(),
            AnnealingSchedule::Custom(betas) => {
                let idx = ((betas.len() - 1) as f64 * progress) as usize;
                betas[idx.min(betas.len() - 1)]
            }
        }
    }
}

/// Simulated annealing optimizer
pub struct SimulatedAnnealing<'a> {
    network: &'a mut PBitNetwork,
    schedule: AnnealingSchedule,
    total_steps: u64,
}

impl<'a> SimulatedAnnealing<'a> {
    pub fn new(network: &'a mut PBitNetwork) -> Self {
        Self {
            network,
            schedule: AnnealingSchedule::exponential(),
            total_steps: 10000,
        }
    }

    /// Set annealing schedule
    pub fn with_schedule(mut self, schedule: AnnealingSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set total steps
    pub fn with_steps(mut self, steps: u64) -> Self {
        self.total_steps = steps;
        self
    }

    /// Run annealing and return best solution
    pub fn run(&mut self) -> AnnealingResult {
        self.network.reset_stats();
        self.network.randomize();

        let mut energy_history = Vec::with_capacity((self.total_steps / 100) as usize);

        for t in 0..self.total_steps {
            let beta = self.schedule.beta_at(t, self.total_steps);
            self.network.set_beta(beta);
            self.network.step();

            // Record energy periodically
            if t % 100 == 0 {
                energy_history.push(self.network.energy());
            }
        }

        AnnealingResult {
            best_state: self.network.best_state().to_vec(),
            best_energy: self.network.best_energy(),
            final_state: self.network.state(),
            final_energy: self.network.energy(),
            energy_history,
        }
    }

    /// Run multiple restarts and return best
    pub fn run_with_restarts(&mut self, restarts: usize) -> AnnealingResult {
        let mut best_result = self.run();

        for _ in 1..restarts {
            let result = self.run();
            if result.best_energy < best_result.best_energy {
                best_result = result;
            }
        }

        best_result
    }
}

/// Result of simulated annealing
#[derive(Debug, Clone)]
pub struct AnnealingResult {
    /// Best state found
    pub best_state: Vec<u8>,
    /// Best energy found
    pub best_energy: f64,
    /// Final state
    pub final_state: Vec<u8>,
    /// Final energy
    pub final_energy: f64,
    /// Energy history (sampled every 100 steps)
    pub energy_history: Vec<f64>,
}

/// Parallel tempering (replica exchange Monte Carlo)
///
/// Runs multiple replicas at different temperatures and periodically
/// attempts to swap configurations between adjacent temperatures.
pub struct ParallelTempering {
    /// Number of replicas
    n_replicas: usize,
    /// Inverse temperatures for each replica (sorted ascending)
    betas: Vec<f64>,
    /// Networks for each replica
    networks: Vec<PBitNetwork>,
    /// RNG for swap decisions
    rng: SimpleRng,
    /// Total swap attempts
    swap_attempts: u64,
    /// Successful swaps
    swap_successes: u64,
}

impl ParallelTempering {
    /// Create parallel tempering with geometric temperature spacing
    pub fn new(template: &PBitNetwork, n_replicas: usize, beta_min: f64, beta_max: f64) -> Self {
        // Geometric spacing of betas
        let betas: Vec<f64> = (0..n_replicas)
            .map(|i| {
                let t = i as f64 / (n_replicas - 1) as f64;
                beta_min * (beta_max / beta_min).powf(t)
            })
            .collect();

        // Create replica networks
        let mut networks = Vec::with_capacity(n_replicas);
        let mut rng = SimpleRng::new(42);

        for &beta in &betas {
            let config = super::network::PBitConfig::new(template.n_pbits())
                .with_beta(beta)
                .with_seed(rng.next_u64());
            let mut network = PBitNetwork::new(config);

            // Copy problem from template
            // (In practice we'd copy couplings/biases directly)
            network.randomize();
            networks.push(network);
        }

        Self {
            n_replicas,
            betas,
            networks,
            rng: SimpleRng::new(12345),
            swap_attempts: 0,
            swap_successes: 0,
        }
    }

    /// Load the same problem into all replicas
    pub fn set_problem(&mut self, problem: &super::ising::IsingProblem) {
        for network in &mut self.networks {
            network.set_problem(problem);
        }
    }

    /// Run one sweep: update all replicas then attempt swaps
    pub fn step(&mut self) {
        // Update each replica
        for network in &mut self.networks {
            network.step();
        }

        // Attempt swaps between adjacent replicas
        self.attempt_swaps();
    }

    /// Attempt swaps between adjacent temperature replicas
    fn attempt_swaps(&mut self) {
        for i in 0..(self.n_replicas - 1) {
            self.swap_attempts += 1;

            let e_i = self.networks[i].energy();
            let e_j = self.networks[i + 1].energy();
            let beta_i = self.betas[i];
            let beta_j = self.betas[i + 1];

            // Metropolis criterion for swap
            let delta = (beta_j - beta_i) * (e_i - e_j);
            let accept = delta < 0.0 || self.rng.next_f64() < (-delta).exp();

            if accept {
                self.swap_successes += 1;
                // Swap states
                let state_i = self.networks[i].state();
                let state_j = self.networks[i + 1].state();
                self.networks[i].set_state(&state_j);
                self.networks[i + 1].set_state(&state_i);
            }
        }
    }

    /// Run for given number of sweeps
    pub fn run(&mut self, sweeps: u64) {
        for _ in 0..sweeps {
            self.step();
        }
    }

    /// Get best solution across all replicas
    pub fn best_solution(&self) -> (Vec<u8>, f64) {
        let mut best_state = self.networks[0].best_state().to_vec();
        let mut best_energy = self.networks[0].best_energy();

        for network in &self.networks[1..] {
            if network.best_energy() < best_energy {
                best_energy = network.best_energy();
                best_state = network.best_state().to_vec();
            }
        }

        (best_state, best_energy)
    }

    /// Get swap acceptance rate
    pub fn swap_rate(&self) -> f64 {
        if self.swap_attempts == 0 {
            0.0
        } else {
            self.swap_successes as f64 / self.swap_attempts as f64
        }
    }

    /// Get reference to coldest (highest beta) replica
    pub fn coldest_replica(&self) -> &PBitNetwork {
        &self.networks[self.n_replicas - 1]
    }

    /// Get replica at given index
    pub fn replica(&self, i: usize) -> &PBitNetwork {
        &self.networks[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbit::ising::MaxCutGraph;
    use crate::pbit::network::PBitConfig;

    #[test]
    fn test_annealing_schedule() {
        let schedule = AnnealingSchedule::Linear {
            beta_min: 1.0,
            beta_max: 10.0,
        };

        assert!((schedule.beta_at(0, 100) - 1.0).abs() < 1e-10);
        assert!((schedule.beta_at(50, 100) - 5.5).abs() < 1e-10);
        assert!((schedule.beta_at(100, 100) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulated_annealing_maxcut() {
        // Small MaxCut instance
        let graph = MaxCutGraph::complete(6);
        let problem = graph.to_ising();

        let config = PBitConfig::new(6);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let mut annealer = SimulatedAnnealing::new(&mut network).with_steps(5000);

        let result = annealer.run();

        // Complete graph with 6 nodes: optimal cut = 9 edges (3-3 split)
        let cut = graph.cut_value(&result.best_state);
        assert!(cut >= 7.0, "Expected good cut, got {}", cut);
    }

    #[test]
    fn test_gibbs_sampler() {
        let config = PBitConfig::new(4).with_beta(2.0);
        let mut network = PBitNetwork::new(config);

        // Ferromagnetic coupling
        for i in 0..4 {
            for j in (i + 1)..4 {
                network.set_coupling(i, j, 1.0);
            }
        }

        let mut sampler = GibbsSampler::new(&mut network)
            .with_equilibration(500)
            .with_sampling(1000)
            .with_thin(10);

        let samples = sampler.run();
        assert!(!samples.is_empty());

        // At low temperature, samples should be mostly aligned
        let all_aligned: usize = samples
            .iter()
            .filter(|s| s.iter().all(|&x| x == s[0]))
            .count();

        assert!(
            all_aligned > samples.len() / 2,
            "Expected mostly aligned samples"
        );
    }
}
