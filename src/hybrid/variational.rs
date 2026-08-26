//! Sprint 67.0: Variational Algorithm Execution
//!
//! Provides the execution engine for variational hybrid algorithms (VQE, QAOA)
//! using the hybrid orchestration layer.
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::{HybridScheduler, VariationalSpec, VariationalExecutor};
//!
//! let mut scheduler = HybridScheduler::new();
//! scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 20)));
//!
//! let spec = VariationalSpec::new(
//!     QuantumCircuit::hardware_efficient(4, 2),
//!     CostFunction::ExpectationValue(hamiltonian_terms),
//! )
//! .with_optimizer(Optimizer::SPSA { a: 0.1, c: 0.1, maxiter: 100 })
//! .with_shots(1000);
//!
//! let mut executor = VariationalExecutor::new(&mut scheduler);
//! let result = executor.run(&spec);
//! println!("Optimal energy: {}", result.optimal_cost);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::expectation::ExpectationEvaluator;
use super::job::{
    CostFunction, JobSpec, MeasurementSpec, PauliTerm, QuantumCircuit, QuantumSpec, VariationalSpec,
};
use super::optimizer::{ClassicalOptimizer, OptimizerState};
use super::result::StageOutputs;
use super::scheduler::HybridScheduler;

#[cfg(feature = "config")]
use super::checkpoint::{CheckpointManager, VariationalCheckpoint};

/// Configuration for variational execution.
#[derive(Clone, Debug)]
pub struct VariationalConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence threshold (stop if cost change < threshold)
    pub convergence_threshold: f64,

    /// Window size for convergence detection
    pub convergence_window: usize,

    /// Number of shots per cost evaluation
    pub shots: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Checkpoint interval (iterations between saves)
    #[cfg(feature = "config")]
    pub checkpoint_interval: Option<usize>,

    /// Enable gradient-based optimization
    pub use_gradient: bool,

    /// Verbose output during optimization
    pub verbose: bool,
}

impl Default for VariationalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            convergence_window: 5,
            shots: 1024,
            seed: None,
            #[cfg(feature = "config")]
            checkpoint_interval: None,
            use_gradient: false,
            verbose: false,
        }
    }
}

impl VariationalConfig {
    /// Create config from VariationalSpec.
    pub fn from_spec(spec: &VariationalSpec) -> Self {
        Self {
            max_iterations: spec.max_iterations,
            convergence_threshold: spec.tolerance,
            shots: spec.shots,
            seed: spec.seed,
            ..Default::default()
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set number of shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Enable gradient-based optimization.
    pub fn with_gradient(mut self, use_gradient: bool) -> Self {
        self.use_gradient = use_gradient;
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Result of variational optimization.
#[derive(Clone, Debug)]
pub struct VariationalResult {
    /// Optimal parameters found
    pub optimal_params: Vec<f64>,

    /// Optimal cost (energy) value
    pub optimal_cost: f64,

    /// Number of iterations completed
    pub iterations: usize,

    /// Whether optimization converged
    pub converged: bool,

    /// Cost history (one entry per iteration)
    pub history: Vec<f64>,

    /// Parameter history (optional, for analysis)
    pub param_history: Option<Vec<Vec<f64>>>,

    /// Total circuits executed
    pub total_circuits: usize,

    /// Total measurement shots
    pub total_shots: usize,

    /// Total execution time
    pub duration: Duration,

    /// Final optimizer state (for resumption)
    pub optimizer_state: Option<OptimizerState>,
}

impl VariationalResult {
    /// Create a new result.
    pub fn new(optimal_params: Vec<f64>, optimal_cost: f64) -> Self {
        Self {
            optimal_params,
            optimal_cost,
            iterations: 0,
            converged: false,
            history: Vec::new(),
            param_history: None,
            total_circuits: 0,
            total_shots: 0,
            duration: Duration::ZERO,
            optimizer_state: None,
        }
    }

    /// Check if optimization improved from initial cost.
    pub fn improved(&self) -> bool {
        if let Some(&first) = self.history.first() {
            self.optimal_cost < first
        } else {
            false
        }
    }

    /// Get the improvement ratio.
    pub fn improvement_ratio(&self) -> Option<f64> {
        self.history.first().map(|&first| {
            if first.abs() < 1e-10 {
                0.0
            } else {
                (first - self.optimal_cost) / first.abs()
            }
        })
    }

    /// Average cost per iteration.
    pub fn avg_cost(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        }
    }
}

/// Executor for variational algorithms.
pub struct VariationalExecutor<'a> {
    /// Reference to the hybrid scheduler
    scheduler: &'a mut HybridScheduler,

    /// Configuration
    config: VariationalConfig,

    /// Expectation value evaluator
    expectation_eval: Option<ExpectationEvaluator>,

    /// Checkpoint manager (if enabled)
    #[cfg(feature = "config")]
    checkpoint_manager: Option<CheckpointManager>,
}

impl<'a> VariationalExecutor<'a> {
    /// Create a new variational executor.
    pub fn new(scheduler: &'a mut HybridScheduler) -> Self {
        Self {
            scheduler,
            config: VariationalConfig::default(),
            expectation_eval: None,
            #[cfg(feature = "config")]
            checkpoint_manager: None,
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: VariationalConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable checkpointing.
    #[cfg(feature = "config")]
    pub fn with_checkpointing(mut self, base_path: &str, interval: usize) -> Self {
        self.checkpoint_manager = Some(CheckpointManager::new(base_path).with_interval(interval));
        self.config.checkpoint_interval = Some(interval);
        self
    }

    /// Run the variational optimization.
    pub fn run(&mut self, spec: &VariationalSpec) -> VariationalResult {
        let start = Instant::now();

        // Extract config from spec
        self.config = VariationalConfig::from_spec(spec);

        // Set up expectation evaluator
        self.expectation_eval = Some(ExpectationEvaluator::from_cost_function(
            &spec.cost_function,
        ));

        // Initialize parameters
        let n_params = spec.ansatz.parameter_count();
        let mut params = spec
            .initial_params
            .clone()
            .unwrap_or_else(|| self.initialize_params(n_params, spec.seed));

        // Initialize optimizer
        let mut optimizer = ClassicalOptimizer::from_spec(&spec.optimizer, n_params);

        // Initialize tracking
        let mut history = Vec::with_capacity(self.config.max_iterations);
        let mut param_history = if self.config.verbose {
            Some(Vec::with_capacity(self.config.max_iterations))
        } else {
            None
        };

        let mut best_params = params.clone();
        let mut best_cost = f64::INFINITY;
        let mut total_circuits = 0usize;
        let mut total_shots = 0usize;
        let mut converged = false;

        // Optimization loop
        for iteration in 0..self.config.max_iterations {
            // Evaluate cost at current parameters
            let cost = self.evaluate_cost(&spec.ansatz, &params, &spec.cost_function);
            total_circuits += 1;
            total_shots += self.config.shots;

            // Update best
            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            // Record history
            history.push(cost);
            if let Some(ref mut ph) = param_history {
                ph.push(params.clone());
            }

            // Check convergence
            if self.check_convergence(&history) {
                converged = true;
                if self.config.verbose {
                    println!(
                        "Converged at iteration {} with cost {:.6}",
                        iteration + 1,
                        best_cost
                    );
                }
                break;
            }

            // Compute gradient if needed
            let gradient = if self.config.use_gradient || optimizer.needs_gradient() {
                let grad = self.estimate_gradient(&spec.ansatz, &params, &spec.cost_function);
                total_circuits += 2 * n_params; // Parameter-shift uses 2 evaluations per param
                total_shots += 2 * n_params * self.config.shots;
                Some(grad)
            } else {
                None
            };

            // Update parameters
            params = optimizer.step(&params, cost, gradient.as_deref());

            // Checkpoint if enabled
            #[cfg(feature = "config")]
            if let Some(ref mut cm) = self.checkpoint_manager {
                if cm.should_save(iteration) {
                    let checkpoint = VariationalCheckpoint::new(
                        params.clone(),
                        cost,
                        iteration,
                        history.clone(),
                    )
                    .with_algorithm("VQE");
                    let _ = cm.save(&checkpoint);
                }
            }

            // Verbose output
            if self.config.verbose && (iteration % 10 == 0 || iteration == 0) {
                println!(
                    "Iteration {:>4}: cost = {:>10.6}, best = {:>10.6}",
                    iteration + 1,
                    cost,
                    best_cost
                );
            }
        }

        VariationalResult {
            optimal_params: best_params,
            optimal_cost: best_cost,
            iterations: history.len(),
            converged,
            history,
            param_history,
            total_circuits,
            total_shots,
            duration: start.elapsed(),
            optimizer_state: Some(optimizer.state()),
        }
    }

    /// Initialize random parameters.
    fn initialize_params(&self, n_params: usize, seed: Option<u64>) -> Vec<f64> {
        let mut params = Vec::with_capacity(n_params);
        let mut rng_state = seed.unwrap_or(42);

        for _ in 0..n_params {
            // Simple LCG for reproducibility
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng_state as f64) / (u64::MAX as f64);
            // Initialize in [0, 2π)
            params.push(r * 2.0 * std::f64::consts::PI);
        }

        params
    }

    /// Evaluate cost function at given parameters.
    fn evaluate_cost(
        &mut self,
        ansatz: &QuantumCircuit,
        params: &[f64],
        cost_fn: &CostFunction,
    ) -> f64 {
        match cost_fn {
            CostFunction::ExpectationValue(terms) => {
                self.evaluate_expectation(ansatz, params, terms)
            }
            CostFunction::MaxCut { edges } => self.evaluate_maxcut(ansatz, params, edges),
            CostFunction::Zero => 0.0,
            CostFunction::Custom { .. } => {
                // For custom cost functions, just return Z expectation
                self.evaluate_z_expectation(ansatz, params)
            }
        }
    }

    /// Evaluate expectation value of Pauli Hamiltonian.
    fn evaluate_expectation(
        &mut self,
        ansatz: &QuantumCircuit,
        params: &[f64],
        terms: &[PauliTerm],
    ) -> f64 {
        // Bind parameters to circuit
        let circuit = ansatz.bind_parameters(params);

        // Execute circuit
        let counts = self.execute_circuit(&circuit, ansatz.n_qubits);

        // Compute expectation value
        if let Some(ref evaluator) = self.expectation_eval {
            evaluator.evaluate(&counts)
        } else {
            ExpectationEvaluator::compute_pauli_expectation(&counts, terms)
        }
    }

    /// Evaluate MaxCut cost function.
    fn evaluate_maxcut(
        &mut self,
        ansatz: &QuantumCircuit,
        params: &[f64],
        edges: &[(usize, usize, f64)],
    ) -> f64 {
        let circuit = ansatz.bind_parameters(params);
        let counts = self.execute_circuit(&circuit, ansatz.n_qubits);

        // MaxCut: count = -sum over edges of w_ij * (1 - z_i * z_j) / 2
        // We want to maximize cut, so return negative
        let total: usize = counts.values().sum();
        if total == 0 {
            return 0.0;
        }

        let mut cost = 0.0;
        for (bitstring, &count) in &counts {
            let bits: Vec<bool> = bitstring.chars().rev().map(|c| c == '1').collect();
            let mut cut_value = 0.0;

            for &(i, j, weight) in edges {
                if i < bits.len() && j < bits.len() && bits[i] != bits[j] {
                    cut_value += weight;
                }
            }

            cost += cut_value * (count as f64 / total as f64);
        }

        // Return negative because we minimize (want to maximize cut)
        -cost
    }

    /// Evaluate simple Z expectation (for custom cost functions).
    fn evaluate_z_expectation(&mut self, ansatz: &QuantumCircuit, params: &[f64]) -> f64 {
        let circuit = ansatz.bind_parameters(params);
        let counts = self.execute_circuit(&circuit, ansatz.n_qubits);

        let total: usize = counts.values().sum();
        if total == 0 {
            return 0.0;
        }

        // Sum of Z expectation on all qubits
        let mut expectation = 0.0;
        for (bitstring, &count) in &counts {
            let mut z_sum = 0.0;
            for c in bitstring.chars() {
                z_sum += if c == '0' { 1.0 } else { -1.0 };
            }
            expectation += z_sum * (count as f64 / total as f64);
        }

        expectation / ansatz.n_qubits as f64
    }

    /// Execute a circuit and return measurement counts.
    fn execute_circuit(
        &mut self,
        circuit: &QuantumCircuit,
        n_qubits: usize,
    ) -> HashMap<String, usize> {
        let spec = QuantumSpec::new(n_qubits, circuit.clone())
            .with_shots(self.config.shots)
            .with_measurement(MeasurementSpec::Computational);

        let spec = if let Some(seed) = self.config.seed {
            spec.with_seed(seed)
        } else {
            spec
        };

        let job_id = self.scheduler.submit(JobSpec::Quantum(spec));

        if let Some(result) = self.scheduler.wait(job_id)
            && let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            return counts.clone();
        }

        HashMap::new()
    }

    /// Estimate gradient using parameter-shift rule.
    fn estimate_gradient(
        &mut self,
        ansatz: &QuantumCircuit,
        params: &[f64],
        cost_fn: &CostFunction,
    ) -> Vec<f64> {
        let shift = std::f64::consts::PI / 2.0;
        let n_params = params.len();
        let mut gradient = vec![0.0; n_params];

        for i in 0..n_params {
            // f(θ + π/2)
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;
            let cost_plus = self.evaluate_cost(ansatz, &params_plus, cost_fn);

            // f(θ - π/2)
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;
            let cost_minus = self.evaluate_cost(ansatz, &params_minus, cost_fn);

            // Gradient: (f(θ + π/2) - f(θ - π/2)) / 2
            gradient[i] = (cost_plus - cost_minus) / 2.0;
        }

        gradient
    }

    /// Check if optimization has converged.
    fn check_convergence(&self, history: &[f64]) -> bool {
        if history.len() < self.config.convergence_window {
            return false;
        }

        let window: Vec<f64> = history
            .iter()
            .rev()
            .take(self.config.convergence_window)
            .cloned()
            .collect();

        let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = window.iter().cloned().fold(f64::INFINITY, f64::min);

        (max - min).abs() < self.config.convergence_threshold
    }
}

/// Execute a variational job specification through the scheduler.
pub fn execute_variational(
    scheduler: &mut HybridScheduler,
    spec: &VariationalSpec,
) -> VariationalResult {
    let mut executor = VariationalExecutor::new(scheduler);
    executor.run(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::QuantumSubstrate;

    fn create_test_scheduler() -> HybridScheduler {
        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Test", 10)));
        scheduler
    }

    #[test]
    fn test_variational_config_default() {
        let config = VariationalConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.shots, 1024);
        assert!(!config.use_gradient);
    }

    #[test]
    fn test_variational_config_builder() {
        let config = VariationalConfig::default()
            .with_max_iterations(50)
            .with_shots(2000)
            .with_gradient(true);

        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.shots, 2000);
        assert!(config.use_gradient);
    }

    #[test]
    fn test_variational_result_new() {
        let result = VariationalResult::new(vec![0.1, 0.2], -1.5);

        assert_eq!(result.optimal_params, vec![0.1, 0.2]);
        assert_eq!(result.optimal_cost, -1.5);
        assert!(!result.converged);
    }

    #[test]
    fn test_variational_result_improved() {
        let mut result = VariationalResult::new(vec![0.1], -1.5);
        result.history = vec![-1.0, -1.2, -1.5];

        assert!(result.improved());
    }

    #[test]
    fn test_convergence_detection() {
        let mut scheduler = create_test_scheduler();
        let executor = VariationalExecutor::new(&mut scheduler)
            .with_config(VariationalConfig::default().with_convergence_threshold(0.01));

        // Not converged - not enough history
        assert!(!executor.check_convergence(&[1.0, 1.0]));

        // Not converged - large variation
        assert!(!executor.check_convergence(&[1.0, 0.5, 0.8, 0.3, 0.9]));

        // Converged - small variation
        assert!(executor.check_convergence(&[1.0, 1.001, 0.999, 1.002, 0.998]));
    }

    #[test]
    fn test_param_initialization() {
        let mut scheduler = create_test_scheduler();
        let executor = VariationalExecutor::new(&mut scheduler);

        let params = executor.initialize_params(5, Some(42));
        assert_eq!(params.len(), 5);

        // Should be reproducible
        let params2 = executor.initialize_params(5, Some(42));
        assert_eq!(params, params2);

        // Different seed should give different params
        let params3 = executor.initialize_params(5, Some(123));
        assert_ne!(params, params3);
    }

    #[test]
    fn test_simple_variational_run() {
        let mut scheduler = create_test_scheduler();

        let ansatz = QuantumCircuit::hardware_efficient(2, 1);
        let spec = VariationalSpec::new(ansatz, CostFunction::Zero)
            .with_max_iterations(5)
            .with_shots(100);

        let result = execute_variational(&mut scheduler, &spec);

        assert!(result.iterations <= 5);
        assert!(!result.history.is_empty());
    }
}
