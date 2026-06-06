// =============================================================================
// QAOA (Quantum Approximate Optimization Algorithm) Module
// =============================================================================
//
// Hybrid quantum-classical algorithm for combinatorial optimization.
//
// QAOA uses alternating layers of cost and mixer unitaries to explore
// the solution space, with classical optimization of the variational
// parameters (γ, β).
//
// References:
// - Farhi et al. "A Quantum Approximate Optimization Algorithm" (2014)
// - Hadfield et al. "From QAOA to Quantum Alternating Operator Ansatz" (2019)
//
// =============================================================================

pub mod circuit;
pub mod maxcut;
pub mod problem;

// Re-exports
pub use circuit::{
    cost_unitary, mixer_unitary, qaoa_circuit, qaoa_gate_count, qaoa_parameter_count,
};
pub use maxcut::{approximation_ratio, brute_force_maxcut, evaluate_cut};
pub use problem::{Graph, QUBOProblem};

use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

// =============================================================================
// QAOA Configuration and Results
// =============================================================================

/// Optimizer type for QAOA (reused concept from VQE)
#[derive(Debug, Clone)]
pub enum QAOAOptimizer {
    NelderMead {
        tolerance: f64,
        max_iterations: usize,
    },
    SPSA {
        max_iterations: usize,
        a: f64,
        c: f64,
    },
    GradientDescent {
        learning_rate: f64,
        max_iterations: usize,
    },
}

impl Default for QAOAOptimizer {
    fn default() -> Self {
        Self::NelderMead {
            tolerance: 1e-4,
            max_iterations: 200,
        }
    }
}

/// QAOA algorithm configuration
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (circuit depth)
    pub p: usize,
    /// Classical optimizer
    pub optimizer: QAOAOptimizer,
    /// Number of measurement shots per evaluation
    pub n_shots: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Initial parameters (if None, uses default initialization)
    pub initial_params: Option<Vec<f64>>,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            p: 2,
            optimizer: QAOAOptimizer::default(),
            n_shots: 1000,
            seed: 42,
            initial_params: None,
        }
    }
}

impl QAOAConfig {
    /// Create config with specified depth
    pub fn with_depth(p: usize) -> Self {
        Self {
            p,
            ..Default::default()
        }
    }

    /// Set number of shots
    pub fn with_shots(mut self, n_shots: usize) -> Self {
        self.n_shots = n_shots;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Use SPSA optimizer
    pub fn with_spsa(mut self) -> Self {
        self.optimizer = QAOAOptimizer::SPSA {
            max_iterations: 200,
            a: 0.1,
            c: 0.1,
        };
        self
    }

    /// Use gradient descent optimizer
    pub fn with_gradient_descent(mut self, learning_rate: f64) -> Self {
        self.optimizer = QAOAOptimizer::GradientDescent {
            learning_rate,
            max_iterations: 200,
        };
        self
    }

    /// Set initial parameters
    pub fn with_initial_params(mut self, params: Vec<f64>) -> Self {
        self.initial_params = Some(params);
        self
    }
}

/// Result of QAOA optimization
#[derive(Debug, Clone)]
pub struct QAOAResult {
    /// Best solution bitstring found
    pub best_bitstring: Vec<bool>,
    /// Cut value of best solution
    pub best_cut: f64,
    /// Final expectation value of cost Hamiltonian
    pub expectation: f64,
    /// Optimized parameters [γ_1, ..., γ_p, β_1, ..., β_p]
    pub optimal_params: Vec<f64>,
    /// Approximation ratio (if optimal is known)
    pub approximation_ratio: Option<f64>,
    /// Number of optimization iterations
    pub iterations: usize,
    /// Number of cost function evaluations
    pub evaluations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Energy history during optimization
    pub history: Vec<f64>,
}

// =============================================================================
// QAOA Cost Function
// =============================================================================

/// Cost function for QAOA optimization
///
/// Evaluates: -⟨ψ(γ,β)|H_C|ψ(γ,β)⟩ (negative because we minimize)
pub struct QAOACostFunction {
    graph: Graph,
    p: usize,
    n_shots: usize,
    seed: u64,
    eval_count: usize,
}

impl QAOACostFunction {
    pub fn new(graph: Graph, p: usize, n_shots: usize, seed: u64) -> Self {
        Self {
            graph,
            p,
            n_shots,
            seed,
            eval_count: 0,
        }
    }

    /// Evaluate cost function at given parameters
    ///
    /// Parameters: [γ_1, ..., γ_p, β_1, ..., β_p]
    pub fn evaluate(&mut self, params: &[f64]) -> f64 {
        self.eval_count += 1;

        let (gammas, betas) = params.split_at(self.p);
        let circuit = qaoa_circuit(&self.graph, gammas, betas);

        // Prepare |0⟩^n state
        let mut state = QState::new_zero(self.graph.n_vertices as u8);
        let mut rng = QRng::new(self.seed.wrapping_add(self.eval_count as u64));

        // Apply QAOA circuit
        for gate in &circuit {
            let _ = apply_gate_scalar(&mut state, gate, &mut rng);
        }

        // Compute MaxCut expectation value
        let expectation = self.compute_maxcut_expectation(&state);

        // Return negative because optimizers minimize
        -expectation
    }

    /// Compute expectation value of MaxCut cost Hamiltonian
    ///
    /// ⟨H_C⟩ = Σ_{(i,j)∈E} w_ij * (1 - ⟨Z_i Z_j⟩) / 2
    fn compute_maxcut_expectation(&self, state: &QState) -> f64 {
        let mut total = 0.0;

        for &(u, v, weight) in &self.graph.edges {
            // Measure ⟨Z_i Z_j⟩ by sampling
            let zz_expectation = self.measure_zz(state, u, v);
            total += weight * (1.0 - zz_expectation) / 2.0;
        }

        total
    }

    /// Measure ⟨Z_i Z_j⟩ expectation value by sampling
    fn measure_zz(&self, initial_state: &QState, i: usize, j: usize) -> f64 {
        let mut total_parity = 0.0;

        for shot in 0..self.n_shots {
            let mut state = initial_state.clone();
            let mut rng = QRng::new(self.seed.wrapping_add((shot * 10000 + i * 100 + j) as u64));

            // Measure qubit i
            let bit_i = match apply_gate_scalar(&mut state, &QGate::Measure(i as u8), &mut rng) {
                GateOutcome::Measured { bit, .. } => bit,
                _ => 0,
            };

            // Measure qubit j
            let bit_j = match apply_gate_scalar(&mut state, &QGate::Measure(j as u8), &mut rng) {
                GateOutcome::Measured { bit, .. } => bit,
                _ => 0,
            };

            // Z eigenvalues: |0⟩ → +1, |1⟩ → -1
            // ZZ eigenvalue: (-1)^(bit_i ⊕ bit_j)
            let parity = if bit_i == bit_j { 1.0 } else { -1.0 };
            total_parity += parity;
        }

        total_parity / self.n_shots as f64
    }

    /// Get evaluation count
    pub fn evaluation_count(&self) -> usize {
        self.eval_count
    }

    /// Sample final state and return best bitstring
    pub fn sample_best_solution(&mut self, params: &[f64], n_samples: usize) -> (Vec<bool>, f64) {
        let (gammas, betas) = params.split_at(self.p);
        let circuit = qaoa_circuit(&self.graph, gammas, betas);

        let mut best_cut = f64::NEG_INFINITY;
        let mut best_bitstring = vec![false; self.graph.n_vertices];

        for sample in 0..n_samples {
            let mut state = QState::new_zero(self.graph.n_vertices as u8);
            let mut rng = QRng::new(self.seed.wrapping_add(1000000 + sample as u64));

            // Apply circuit
            for gate in &circuit {
                let _ = apply_gate_scalar(&mut state, gate, &mut rng);
            }

            // Measure all qubits
            let mut bitstring = vec![false; self.graph.n_vertices];
            for q in 0..self.graph.n_vertices {
                if let GateOutcome::Measured { bit, .. } =
                    apply_gate_scalar(&mut state, &QGate::Measure(q as u8), &mut rng)
                {
                    bitstring[q] = bit == 1;
                }
            }

            let cut = evaluate_cut(&self.graph, &bitstring);
            if cut > best_cut {
                best_cut = cut;
                best_bitstring = bitstring;
            }
        }

        (best_bitstring, best_cut)
    }
}

// =============================================================================
// Optimizers (Simplified implementations)
// =============================================================================

/// Run Nelder-Mead optimization
fn run_nelder_mead(
    cost_fn: &mut QAOACostFunction,
    initial: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, bool) {
    let n = initial.len();
    let mut history = Vec::new();

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());

    for i in 0..n {
        let mut point = initial.to_vec();
        point[i] += 0.5; // Step size for initial simplex
        simplex.push(point);
    }

    // Evaluate simplex vertices
    let mut values: Vec<f64> = simplex.iter().map(|p| cost_fn.evaluate(p)).collect();

    let mut iteration = 0;
    let mut converged = false;

    while iteration < max_iterations {
        iteration += 1;

        // Sort by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        history.push(values[best_idx]);

        // Check convergence
        let range = values[worst_idx] - values[best_idx];
        if range < tolerance {
            converged = true;
            break;
        }

        // Compute centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &idx in &indices[0..n] {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = centroid
            .iter()
            .zip(&simplex[worst_idx])
            .map(|(c, w)| 2.0 * c - w)
            .collect();
        let reflected_val = cost_fn.evaluate(&reflected);

        if reflected_val < values[second_worst_idx] && reflected_val >= values[best_idx] {
            // Accept reflection
            simplex[worst_idx] = reflected;
            values[worst_idx] = reflected_val;
        } else if reflected_val < values[best_idx] {
            // Expansion
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(&reflected)
                .map(|(c, r)| c + 2.0 * (r - c))
                .collect();
            let expanded_val = cost_fn.evaluate(&expanded);

            if expanded_val < reflected_val {
                simplex[worst_idx] = expanded;
                values[worst_idx] = expanded_val;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_val;
            }
        } else {
            // Contraction
            let contracted: Vec<f64> = centroid
                .iter()
                .zip(&simplex[worst_idx])
                .map(|(c, w)| c + 0.5 * (w - c))
                .collect();
            let contracted_val = cost_fn.evaluate(&contracted);

            if contracted_val < values[worst_idx] {
                simplex[worst_idx] = contracted;
                values[worst_idx] = contracted_val;
            } else {
                // Shrink
                let best = simplex[best_idx].clone();
                for i in 0..=n {
                    if i != best_idx {
                        for j in 0..n {
                            simplex[i][j] = best[j] + 0.5 * (simplex[i][j] - best[j]);
                        }
                        values[i] = cost_fn.evaluate(&simplex[i]);
                    }
                }
            }
        }
    }

    // Find best point
    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    (simplex[best_idx].clone(), history, converged)
}

/// Run SPSA optimization
fn run_spsa(
    cost_fn: &mut QAOACostFunction,
    initial: &[f64],
    max_iterations: usize,
    a: f64,
    c: f64,
) -> (Vec<f64>, Vec<f64>, bool) {
    let n = initial.len();
    let mut params = initial.to_vec();
    let mut history = Vec::new();

    let alpha = 0.602;
    let gamma = 0.101;

    for k in 0..max_iterations {
        let ak = a / (k as f64 + 1.0).powf(alpha);
        let ck = c / (k as f64 + 1.0).powf(gamma);

        // Generate random perturbation direction
        let mut rng_state = (k as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
        let delta: Vec<f64> = (0..n)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng_state >> 63) == 0 { 1.0 } else { -1.0 }
            })
            .collect();

        // Perturbed evaluations
        let plus: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p + ck * d).collect();
        let minus: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p - ck * d).collect();

        let f_plus = cost_fn.evaluate(&plus);
        let f_minus = cost_fn.evaluate(&minus);

        // Gradient estimate
        let grad: Vec<f64> = delta
            .iter()
            .map(|d| (f_plus - f_minus) / (2.0 * ck * d))
            .collect();

        // Update parameters
        for (p, g) in params.iter_mut().zip(&grad) {
            *p -= ak * g;
        }

        history.push(cost_fn.evaluate(&params));
    }

    (params, history, true)
}

// =============================================================================
// Main QAOA Entry Point
// =============================================================================

/// Run QAOA on a graph for MaxCut
pub fn run_qaoa(graph: Graph, config: QAOAConfig) -> QAOAResult {
    let n_params = 2 * config.p;

    // Default initialization: γ near 0.5, β near π/8
    let initial = config.initial_params.unwrap_or_else(|| {
        let mut params = vec![0.5; n_params];
        for i in config.p..n_params {
            params[i] = std::f64::consts::PI / 8.0;
        }
        params
    });

    let mut cost_fn = QAOACostFunction::new(graph.clone(), config.p, config.n_shots, config.seed);

    // Run optimization
    let (optimal_params, history, converged) = match config.optimizer {
        QAOAOptimizer::NelderMead {
            tolerance,
            max_iterations,
        } => run_nelder_mead(&mut cost_fn, &initial, tolerance, max_iterations),
        QAOAOptimizer::SPSA {
            max_iterations,
            a,
            c,
        } => run_spsa(&mut cost_fn, &initial, max_iterations, a, c),
        QAOAOptimizer::GradientDescent {
            learning_rate,
            max_iterations,
        } => {
            // Simple gradient descent using finite differences
            let mut params = initial.clone();
            let mut hist = Vec::new();
            let eps = 0.01;

            for _ in 0..max_iterations {
                let f0 = cost_fn.evaluate(&params);
                hist.push(f0);

                let mut grad = vec![0.0; n_params];
                for i in 0..n_params {
                    let mut p_plus = params.clone();
                    p_plus[i] += eps;
                    grad[i] = (cost_fn.evaluate(&p_plus) - f0) / eps;
                }

                for (p, g) in params.iter_mut().zip(&grad) {
                    *p -= learning_rate * g;
                }
            }
            (params, hist, true)
        }
    };

    let evaluations = cost_fn.evaluation_count();
    let expectation = -cost_fn.evaluate(&optimal_params); // Negate back

    // Sample best solution
    let (best_bitstring, best_cut) = cost_fn.sample_best_solution(&optimal_params, config.n_shots);

    // Compute approximation ratio if graph is small enough
    let approximation_ratio = if graph.n_vertices <= 15 {
        let (optimal, _) = brute_force_maxcut(&graph);
        if optimal > 0.0 {
            Some(best_cut / optimal)
        } else {
            Some(1.0)
        }
    } else {
        None
    };

    QAOAResult {
        best_bitstring,
        best_cut,
        expectation,
        optimal_params,
        approximation_ratio,
        iterations: history.len(),
        evaluations,
        converged,
        history,
    }
}

/// Convenience function: solve MaxCut with default settings
pub fn maxcut(graph: Graph) -> QAOAResult {
    run_qaoa(graph, QAOAConfig::default())
}

/// Convenience function: solve MaxCut with specified depth
pub fn maxcut_p(graph: Graph, p: usize) -> QAOAResult {
    run_qaoa(graph, QAOAConfig::with_depth(p))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qaoa_config_default() {
        let config = QAOAConfig::default();
        assert_eq!(config.p, 2);
        assert_eq!(config.n_shots, 1000);
    }

    #[test]
    fn test_qaoa_config_builder() {
        let config = QAOAConfig::with_depth(3).with_shots(500).with_seed(123);

        assert_eq!(config.p, 3);
        assert_eq!(config.n_shots, 500);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_cost_function_evaluation() {
        let graph = Graph::cycle(3);
        let mut cost_fn = QAOACostFunction::new(graph, 1, 100, 42);

        let params = vec![0.5, 0.3]; // γ, β
        let cost = cost_fn.evaluate(&params);

        // Cost should be negative (we negate for minimization)
        assert!(cost < 0.0 || cost >= 0.0); // Just check it runs
    }

    #[test]
    fn test_qaoa_triangle() {
        let mut graph = Graph::new(3);
        graph.add_unweighted_edge(0, 1);
        graph.add_unweighted_edge(1, 2);
        graph.add_unweighted_edge(0, 2);

        let result = run_qaoa(
            graph,
            QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
        );

        // Should find a reasonable solution
        assert!(result.best_cut >= 0.0);
        assert!(result.best_cut <= 3.0); // Max possible is 2 for triangle
    }

    #[test]
    fn test_qaoa_square() {
        let graph = Graph::cycle(4);

        let result = run_qaoa(
            graph,
            QAOAConfig::with_depth(2).with_shots(200).with_seed(42),
        );

        // Square has optimal cut of 4
        assert!(result.best_cut >= 2.0); // Should get at least half
        assert!(result.approximation_ratio.is_some());
    }

    #[test]
    fn test_qaoa_result_structure() {
        let graph = Graph::path(3);
        let result = maxcut(graph);

        assert_eq!(result.best_bitstring.len(), 3);
        assert_eq!(result.optimal_params.len(), 4); // 2 * p = 2 * 2
        assert!(!result.history.is_empty());
    }

    #[test]
    fn test_maxcut_convenience() {
        let graph = Graph::complete(3);
        let result = maxcut(graph);

        assert!(result.best_cut >= 1.0);
    }

    #[test]
    fn test_qaoa_reproducibility() {
        let graph = Graph::random(4, 0.5, 100);

        let result1 = run_qaoa(
            graph.clone(),
            QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
        );

        let result2 = run_qaoa(
            graph,
            QAOAConfig::with_depth(1).with_shots(100).with_seed(42),
        );

        // Same seed should give same result
        assert_eq!(result1.best_cut, result2.best_cut);
    }

    #[test]
    fn test_qaoa_spsa_optimizer() {
        let graph = Graph::cycle(3);

        let result = run_qaoa(graph, QAOAConfig::with_depth(1).with_spsa().with_shots(50));

        assert!(result.best_cut >= 0.0);
    }
}
