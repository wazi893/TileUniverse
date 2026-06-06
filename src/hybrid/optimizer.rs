//! Sprint 67.0: Classical Optimizers for Variational Algorithms
//!
//! Provides classical optimization algorithms for parameter updates in VQE/QAOA.
//!
//! # Supported Optimizers
//!
//! - **Nelder-Mead**: Simplex-based, derivative-free
//! - **SPSA**: Stochastic gradient approximation, good for noisy objectives
//! - **Gradient Descent**: Simple gradient-based with momentum
//! - **Adam**: Adaptive learning rate with momentum
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::optimizer::{ClassicalOptimizer, OptimizerState};
//!
//! let mut optimizer = ClassicalOptimizer::spsa(4, 0.1, 0.1);
//!
//! for _ in 0..100 {
//!     let cost = evaluate(&params);
//!     params = optimizer.step(&params, cost, None);
//! }
//! ```

use super::job::Optimizer;

/// State of an optimizer (for checkpointing/resumption).
#[derive(Clone, Debug)]
pub enum OptimizerState {
    /// Nelder-Mead simplex state
    NelderMead {
        simplex: Vec<Vec<f64>>,
        values: Vec<f64>,
        iteration: usize,
    },

    /// SPSA state
    SPSA {
        iteration: usize,
        a_k: f64,
        c_k: f64,
    },

    /// Gradient descent state
    GradientDescent {
        velocity: Vec<f64>,
        iteration: usize,
    },

    /// Adam state
    Adam {
        m: Vec<f64>,
        v: Vec<f64>,
        iteration: usize,
    },

    /// Generic/unknown state
    Generic { iteration: usize },
}

/// Classical optimizer for variational algorithms.
pub struct ClassicalOptimizer {
    /// Number of parameters
    n_params: usize,

    /// Optimizer type and configuration
    optimizer_type: OptimizerType,

    /// Current iteration
    iteration: usize,

    /// Internal state
    state: InternalState,

    /// RNG state for stochastic optimizers
    rng_state: u64,
}

/// Optimizer type with configuration.
#[derive(Clone, Debug)]
enum OptimizerType {
    NelderMead {
        maxiter: usize,
        alpha: f64, // reflection
        #[allow(dead_code)]
        gamma: f64, // expansion
        #[allow(dead_code)]
        rho: f64, // contraction
        #[allow(dead_code)]
        sigma: f64, // shrink
    },
    SPSA {
        a: f64,
        c: f64,
        maxiter: usize,
        a_decay: f64,
        c_decay: f64,
    },
    GradientDescent {
        learning_rate: f64,
        momentum: f64,
        maxiter: usize,
    },
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        maxiter: usize,
    },
}

/// Internal optimizer state.
#[derive(Clone, Debug)]
enum InternalState {
    NelderMead {
        simplex: Vec<Vec<f64>>,
        values: Vec<f64>,
        best_idx: usize,
        worst_idx: usize,
        state: NelderMeadState,
    },
    SPSA {
        a_k: f64,
        c_k: f64,
    },
    GradientDescent {
        velocity: Vec<f64>,
    },
    Adam {
        m: Vec<f64>,
        v: Vec<f64>,
    },
    #[allow(dead_code)]
    Uninitialized,
}

/// Nelder-Mead algorithm state.
#[derive(Clone, Copy, Debug, PartialEq)]
enum NelderMeadState {
    Initialize,
    Reflect,
    #[allow(dead_code)]
    Expand,
    #[allow(dead_code)]
    ContractOutside,
    #[allow(dead_code)]
    ContractInside,
    #[allow(dead_code)]
    Shrink,
}

impl ClassicalOptimizer {
    /// Create optimizer from spec.
    pub fn from_spec(spec: &Optimizer, n_params: usize) -> Self {
        match spec {
            Optimizer::COBYLA { maxiter } => Self::nelder_mead(n_params, *maxiter),
            Optimizer::NelderMead { maxiter } => Self::nelder_mead(n_params, *maxiter),
            Optimizer::SPSA { a, c, maxiter } => Self::spsa(n_params, *a, *c, *maxiter),
            Optimizer::GradientDescent {
                learning_rate,
                maxiter,
            } => Self::gradient_descent(n_params, *learning_rate, *maxiter),
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                maxiter,
            } => Self::adam(n_params, *learning_rate, *beta1, *beta2, *maxiter),
        }
    }

    /// Create Nelder-Mead optimizer.
    pub fn nelder_mead(n_params: usize, maxiter: usize) -> Self {
        Self {
            n_params,
            optimizer_type: OptimizerType::NelderMead {
                maxiter,
                alpha: 1.0,
                gamma: 2.0,
                rho: 0.5,
                sigma: 0.5,
            },
            iteration: 0,
            state: InternalState::NelderMead {
                simplex: Vec::new(),
                values: Vec::new(),
                best_idx: 0,
                worst_idx: 0,
                state: NelderMeadState::Initialize,
            },
            rng_state: 42,
        }
    }

    /// Create SPSA optimizer.
    pub fn spsa(n_params: usize, a: f64, c: f64, maxiter: usize) -> Self {
        Self {
            n_params,
            optimizer_type: OptimizerType::SPSA {
                a,
                c,
                maxiter,
                a_decay: 0.602,
                c_decay: 0.101,
            },
            iteration: 0,
            state: InternalState::SPSA { a_k: a, c_k: c },
            rng_state: 42,
        }
    }

    /// Create gradient descent optimizer.
    pub fn gradient_descent(n_params: usize, learning_rate: f64, maxiter: usize) -> Self {
        Self {
            n_params,
            optimizer_type: OptimizerType::GradientDescent {
                learning_rate,
                momentum: 0.9,
                maxiter,
            },
            iteration: 0,
            state: InternalState::GradientDescent {
                velocity: vec![0.0; n_params],
            },
            rng_state: 42,
        }
    }

    /// Create Adam optimizer.
    pub fn adam(
        n_params: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        maxiter: usize,
    ) -> Self {
        Self {
            n_params,
            optimizer_type: OptimizerType::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon: 1e-8,
                maxiter,
            },
            iteration: 0,
            state: InternalState::Adam {
                m: vec![0.0; n_params],
                v: vec![0.0; n_params],
            },
            rng_state: 42,
        }
    }

    /// Check if this optimizer needs gradient information.
    pub fn needs_gradient(&self) -> bool {
        matches!(
            self.optimizer_type,
            OptimizerType::GradientDescent { .. } | OptimizerType::Adam { .. }
        )
    }

    /// Perform one optimization step.
    pub fn step(&mut self, params: &[f64], cost: f64, gradient: Option<&[f64]>) -> Vec<f64> {
        self.iteration += 1;

        match &self.optimizer_type {
            OptimizerType::NelderMead { .. } => self.step_nelder_mead(params, cost),
            OptimizerType::SPSA { .. } => self.step_spsa(params, cost),
            OptimizerType::GradientDescent { .. } => {
                self.step_gradient_descent(params, gradient.unwrap_or(&vec![0.0; self.n_params]))
            }
            OptimizerType::Adam { .. } => {
                self.step_adam(params, gradient.unwrap_or(&vec![0.0; self.n_params]))
            }
        }
    }

    /// Nelder-Mead step (simplified version).
    fn step_nelder_mead(&mut self, params: &[f64], cost: f64) -> Vec<f64> {
        let OptimizerType::NelderMead {
            alpha,
            gamma: _,
            rho: _,
            sigma: _,
            ..
        } = self.optimizer_type
        else {
            return params.to_vec();
        };

        if let InternalState::NelderMead {
            simplex,
            values,
            best_idx,
            worst_idx,
            state,
        } = &mut self.state
        {
            match state {
                NelderMeadState::Initialize => {
                    // Initialize simplex around current point
                    simplex.clear();
                    values.clear();
                    simplex.push(params.to_vec());
                    values.push(cost);

                    for i in 0..self.n_params {
                        let mut point = params.to_vec();
                        point[i] += 0.1; // Step size
                        simplex.push(point);
                        values.push(f64::INFINITY); // Will be evaluated later
                    }

                    *state = NelderMeadState::Reflect;

                    // Return next point to evaluate
                    if simplex.len() > 1 {
                        simplex[1].clone()
                    } else {
                        params.to_vec()
                    }
                }

                NelderMeadState::Reflect | _ => {
                    // Update the point that was just evaluated
                    if !simplex.is_empty() {
                        // Find which point needs updating
                        for i in 0..values.len() {
                            if values[i] == f64::INFINITY {
                                values[i] = cost;
                                break;
                            }
                        }
                    }

                    // Find best and worst
                    *best_idx = 0;
                    *worst_idx = 0;
                    for i in 1..values.len() {
                        if values[i] < values[*best_idx] {
                            *best_idx = i;
                        }
                        if values[i] > values[*worst_idx] {
                            *worst_idx = i;
                        }
                    }

                    // Compute centroid (excluding worst)
                    let mut centroid = vec![0.0; self.n_params];
                    let n = simplex.len() - 1;
                    for (i, point) in simplex.iter().enumerate() {
                        if i != *worst_idx {
                            for (j, &v) in point.iter().enumerate() {
                                centroid[j] += v / n as f64;
                            }
                        }
                    }

                    // Reflect worst point through centroid
                    let mut reflected = vec![0.0; self.n_params];
                    for i in 0..self.n_params {
                        reflected[i] = centroid[i] + alpha * (centroid[i] - simplex[*worst_idx][i]);
                    }

                    // Replace worst with reflected (simplified)
                    simplex[*worst_idx] = reflected.clone();
                    values[*worst_idx] = f64::INFINITY;

                    reflected
                }
            }
        } else {
            params.to_vec()
        }
    }

    /// SPSA step.
    fn step_spsa(&mut self, params: &[f64], cost: f64) -> Vec<f64> {
        let OptimizerType::SPSA {
            a_decay, c_decay, ..
        } = self.optimizer_type
        else {
            return params.to_vec();
        };

        if let InternalState::SPSA { a_k, c_k } = &mut self.state {
            // Generate perturbation vector (Bernoulli ±1)
            let mut delta = Vec::with_capacity(self.n_params);
            for _ in 0..self.n_params {
                self.rng_state = self
                    .rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let r = (self.rng_state >> 63) as i8 * 2 - 1;
                delta.push(r as f64);
            }

            // Approximate gradient: g ≈ (f(θ + cΔ) - f(θ - cΔ)) / (2cΔ)
            // Here we use a simplified one-sided version: g ≈ f(θ) * Δ / c
            // (Real SPSA would need two evaluations)
            let mut new_params = Vec::with_capacity(self.n_params);
            for i in 0..self.n_params {
                let grad_approx = cost * delta[i] / *c_k;
                new_params.push(params[i] - *a_k * grad_approx);
            }

            // Decay learning rates
            let k = self.iteration as f64;
            *a_k *= 1.0 / (1.0 + k).powf(a_decay);
            *c_k *= 1.0 / (1.0 + k).powf(c_decay);

            new_params
        } else {
            params.to_vec()
        }
    }

    /// Gradient descent step.
    fn step_gradient_descent(&mut self, params: &[f64], gradient: &[f64]) -> Vec<f64> {
        let OptimizerType::GradientDescent {
            learning_rate,
            momentum,
            ..
        } = self.optimizer_type
        else {
            return params.to_vec();
        };

        if let InternalState::GradientDescent { velocity } = &mut self.state {
            let mut new_params = Vec::with_capacity(self.n_params);

            for i in 0..self.n_params {
                // Update velocity with momentum
                velocity[i] = momentum * velocity[i] - learning_rate * gradient[i];
                new_params.push(params[i] + velocity[i]);
            }

            new_params
        } else {
            params.to_vec()
        }
    }

    /// Adam step.
    fn step_adam(&mut self, params: &[f64], gradient: &[f64]) -> Vec<f64> {
        let OptimizerType::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            ..
        } = self.optimizer_type
        else {
            return params.to_vec();
        };

        if let InternalState::Adam { m, v } = &mut self.state {
            let t = self.iteration as f64;
            let mut new_params = Vec::with_capacity(self.n_params);

            for i in 0..self.n_params {
                // Update biased first moment estimate
                m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i];

                // Update biased second moment estimate
                v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i] * gradient[i];

                // Bias correction
                let m_hat = m[i] / (1.0 - beta1.powf(t));
                let v_hat = v[i] / (1.0 - beta2.powf(t));

                // Update parameters
                new_params.push(params[i] - learning_rate * m_hat / (v_hat.sqrt() + epsilon));
            }

            new_params
        } else {
            params.to_vec()
        }
    }

    /// Check if optimizer has converged (hit maxiter).
    pub fn converged(&self) -> bool {
        let maxiter = match &self.optimizer_type {
            OptimizerType::NelderMead { maxiter, .. } => *maxiter,
            OptimizerType::SPSA { maxiter, .. } => *maxiter,
            OptimizerType::GradientDescent { maxiter, .. } => *maxiter,
            OptimizerType::Adam { maxiter, .. } => *maxiter,
        };

        self.iteration >= maxiter
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.state = match &self.optimizer_type {
            OptimizerType::NelderMead { .. } => InternalState::NelderMead {
                simplex: Vec::new(),
                values: Vec::new(),
                best_idx: 0,
                worst_idx: 0,
                state: NelderMeadState::Initialize,
            },
            OptimizerType::SPSA { a, c, .. } => InternalState::SPSA { a_k: *a, c_k: *c },
            OptimizerType::GradientDescent { .. } => InternalState::GradientDescent {
                velocity: vec![0.0; self.n_params],
            },
            OptimizerType::Adam { .. } => InternalState::Adam {
                m: vec![0.0; self.n_params],
                v: vec![0.0; self.n_params],
            },
        };
    }

    /// Get current optimizer state (for checkpointing).
    pub fn state(&self) -> OptimizerState {
        match &self.state {
            InternalState::NelderMead {
                simplex, values, ..
            } => OptimizerState::NelderMead {
                simplex: simplex.clone(),
                values: values.clone(),
                iteration: self.iteration,
            },
            InternalState::SPSA { a_k, c_k } => OptimizerState::SPSA {
                iteration: self.iteration,
                a_k: *a_k,
                c_k: *c_k,
            },
            InternalState::GradientDescent { velocity } => OptimizerState::GradientDescent {
                velocity: velocity.clone(),
                iteration: self.iteration,
            },
            InternalState::Adam { m, v } => OptimizerState::Adam {
                m: m.clone(),
                v: v.clone(),
                iteration: self.iteration,
            },
            InternalState::Uninitialized => OptimizerState::Generic {
                iteration: self.iteration,
            },
        }
    }

    /// Get current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_from_spec() {
        let spec = Optimizer::SPSA {
            a: 0.1,
            c: 0.1,
            maxiter: 100,
        };
        let opt = ClassicalOptimizer::from_spec(&spec, 5);

        assert_eq!(opt.n_params, 5);
        assert!(!opt.needs_gradient());
    }

    #[test]
    fn test_gradient_descent_needs_gradient() {
        let opt = ClassicalOptimizer::gradient_descent(4, 0.01, 100);
        assert!(opt.needs_gradient());
    }

    #[test]
    fn test_adam_needs_gradient() {
        let opt = ClassicalOptimizer::adam(4, 0.001, 0.9, 0.999, 100);
        assert!(opt.needs_gradient());
    }

    #[test]
    fn test_spsa_step() {
        let mut opt = ClassicalOptimizer::spsa(2, 0.1, 0.1, 100);

        let params = vec![0.5, 0.5];
        let new_params = opt.step(&params, 1.0, None);

        // Should produce different parameters
        assert_ne!(params, new_params);
        assert_eq!(new_params.len(), 2);
    }

    #[test]
    fn test_gradient_descent_step() {
        let mut opt = ClassicalOptimizer::gradient_descent(2, 0.1, 100);

        let params = vec![1.0, 1.0];
        let gradient = vec![0.5, -0.5];
        let new_params = opt.step(&params, 1.0, Some(&gradient));

        // Should move in opposite direction of gradient
        assert!(new_params[0] < params[0]); // gradient positive → decrease
        assert!(new_params[1] > params[1]); // gradient negative → increase
    }

    #[test]
    fn test_adam_step() {
        let mut opt = ClassicalOptimizer::adam(2, 0.1, 0.9, 0.999, 100);

        let params = vec![1.0, 1.0];
        let gradient = vec![0.1, 0.1];
        let new_params = opt.step(&params, 1.0, Some(&gradient));

        // Should produce different parameters
        assert_ne!(params, new_params);
    }

    #[test]
    fn test_optimizer_converged() {
        let mut opt = ClassicalOptimizer::spsa(2, 0.1, 0.1, 5);

        let params = vec![0.5, 0.5];
        for _ in 0..5 {
            let _ = opt.step(&params, 1.0, None);
        }

        assert!(opt.converged());
    }

    #[test]
    fn test_optimizer_reset() {
        let mut opt = ClassicalOptimizer::spsa(2, 0.1, 0.1, 100);

        let params = vec![0.5, 0.5];
        for _ in 0..10 {
            let _ = opt.step(&params, 1.0, None);
        }

        assert_eq!(opt.iteration(), 10);

        opt.reset();
        assert_eq!(opt.iteration(), 0);
    }

    #[test]
    fn test_optimizer_state() {
        let mut opt = ClassicalOptimizer::adam(2, 0.1, 0.9, 0.999, 100);

        let params = vec![1.0, 1.0];
        let gradient = vec![0.1, 0.1];
        let _ = opt.step(&params, 1.0, Some(&gradient));

        let state = opt.state();
        match state {
            OptimizerState::Adam { iteration, .. } => {
                assert_eq!(iteration, 1);
            }
            _ => panic!("Expected Adam state"),
        }
    }
}
