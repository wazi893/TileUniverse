//! Sprint 67.0: Gradient Estimation
//!
//! Parameter-shift rule implementation for computing gradients of quantum circuits.
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::gradient::GradientEstimator;
//!
//! let estimator = GradientEstimator::new();
//!
//! // Generate shifted parameter sets
//! let shifted = estimator.shifted_params(&params);
//!
//! // Evaluate all shifted circuits (via batch execution)
//! let costs: Vec<f64> = shifted.iter()
//!     .map(|p| evaluate_circuit(p))
//!     .collect();
//!
//! // Compute gradient from results
//! let gradient = estimator.gradient_from_batch(&costs, params.len());
//! ```

use std::f64::consts::PI;

/// Gradient estimator using parameter-shift rule.
#[derive(Clone, Debug)]
pub struct GradientEstimator {
    /// Shift amount (π/2 for standard rotation gates)
    shift: f64,

    /// Coefficient for gradient formula
    coeff: f64,
}

impl Default for GradientEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientEstimator {
    /// Create a new gradient estimator with standard shift (π/2).
    pub fn new() -> Self {
        Self {
            shift: PI / 2.0,
            coeff: 0.5,
        }
    }

    /// Create estimator with custom shift amount.
    pub fn with_shift(shift: f64) -> Self {
        Self {
            shift,
            coeff: 0.5 / shift.sin(),
        }
    }

    /// Get the shift amount.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// Estimate gradient using parameter-shift rule.
    ///
    /// For each parameter θᵢ:
    /// ∂f/∂θᵢ = (f(θ + sêᵢ) - f(θ - sêᵢ)) / (2 sin(s))
    ///
    /// where s is the shift amount (π/2 for standard gates).
    pub fn estimate<F>(&self, params: &[f64], cost_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_params = params.len();
        let mut gradient = vec![0.0; n_params];

        for i in 0..n_params {
            // f(θ + s*êᵢ)
            let mut params_plus = params.to_vec();
            params_plus[i] += self.shift;
            let cost_plus = cost_fn(&params_plus);

            // f(θ - s*êᵢ)
            let mut params_minus = params.to_vec();
            params_minus[i] -= self.shift;
            let cost_minus = cost_fn(&params_minus);

            // Gradient: (f+ - f-) * coeff
            gradient[i] = (cost_plus - cost_minus) * self.coeff;
        }

        gradient
    }

    /// Generate all shifted parameter sets for batch evaluation.
    ///
    /// Returns 2*n parameter sets: [θ+s*e₀, θ-s*e₀, θ+s*e₁, θ-s*e₁, ...]
    pub fn shifted_params(&self, params: &[f64]) -> Vec<Vec<f64>> {
        let n_params = params.len();
        let mut shifted = Vec::with_capacity(2 * n_params);

        for i in 0..n_params {
            // θ + s*êᵢ
            let mut params_plus = params.to_vec();
            params_plus[i] += self.shift;
            shifted.push(params_plus);

            // θ - s*êᵢ
            let mut params_minus = params.to_vec();
            params_minus[i] -= self.shift;
            shifted.push(params_minus);
        }

        shifted
    }

    /// Compute gradient from batch evaluation results.
    ///
    /// Expects results in order: [f(θ+s*e₀), f(θ-s*e₀), f(θ+s*e₁), f(θ-s*e₁), ...]
    pub fn gradient_from_batch(&self, results: &[f64], n_params: usize) -> Vec<f64> {
        assert_eq!(results.len(), 2 * n_params);

        let mut gradient = vec![0.0; n_params];

        for i in 0..n_params {
            let cost_plus = results[2 * i];
            let cost_minus = results[2 * i + 1];
            gradient[i] = (cost_plus - cost_minus) * self.coeff;
        }

        gradient
    }

    /// Estimate gradient with finite differences (for comparison/validation).
    pub fn finite_difference<F>(&self, params: &[f64], cost_fn: F, epsilon: f64) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_params = params.len();
        let mut gradient = vec![0.0; n_params];

        for i in 0..n_params {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;

            let mut params_minus = params.to_vec();
            params_minus[i] -= epsilon;

            gradient[i] = (cost_fn(&params_plus) - cost_fn(&params_minus)) / (2.0 * epsilon);
        }

        gradient
    }
}

/// Batch gradient estimation for efficient circuit evaluation.
#[derive(Clone, Debug)]
pub struct BatchGradientEstimator {
    /// Base parameter-shift estimator
    estimator: GradientEstimator,

    /// Include central point evaluation
    include_central: bool,
}

impl BatchGradientEstimator {
    /// Create a new batch gradient estimator.
    pub fn new() -> Self {
        Self {
            estimator: GradientEstimator::new(),
            include_central: false,
        }
    }

    /// Include central point (θ) in batch for cost function value.
    pub fn with_central_point(mut self) -> Self {
        self.include_central = true;
        self
    }

    /// Generate all parameter sets for batch evaluation.
    ///
    /// If include_central is true, first element is the original params.
    pub fn generate_batch(&self, params: &[f64]) -> Vec<Vec<f64>> {
        let mut batch = Vec::new();

        if self.include_central {
            batch.push(params.to_vec());
        }

        batch.extend(self.estimator.shifted_params(params));
        batch
    }

    /// Extract gradient and cost from batch results.
    ///
    /// Returns (cost_at_params, gradient).
    pub fn extract_gradient(&self, results: &[f64], n_params: usize) -> (Option<f64>, Vec<f64>) {
        if self.include_central {
            let cost = Some(results[0]);
            let gradient = self.estimator.gradient_from_batch(&results[1..], n_params);
            (cost, gradient)
        } else {
            let gradient = self.estimator.gradient_from_batch(results, n_params);
            (None, gradient)
        }
    }

    /// Number of circuits needed for gradient estimation.
    pub fn circuit_count(&self, n_params: usize) -> usize {
        let base = 2 * n_params;
        if self.include_central { base + 1 } else { base }
    }
}

impl Default for BatchGradientEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Natural gradient estimation using quantum Fisher information.
///
/// The natural gradient adjusts the parameter-space gradient by the inverse
/// of the quantum Fisher information matrix.
#[derive(Clone, Debug)]
pub struct NaturalGradientEstimator {
    /// Base gradient estimator
    gradient_estimator: GradientEstimator,

    /// Regularization parameter for matrix inversion
    regularization: f64,
}

impl NaturalGradientEstimator {
    /// Create a new natural gradient estimator.
    pub fn new() -> Self {
        Self {
            gradient_estimator: GradientEstimator::new(),
            regularization: 1e-4,
        }
    }

    /// Set regularization parameter.
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Estimate quantum Fisher information matrix element.
    ///
    /// F_ij = Re[⟨∂ψ/∂θᵢ|∂ψ/∂θⱼ⟩] - ⟨∂ψ/∂θᵢ|ψ⟩⟨ψ|∂ψ/∂θⱼ⟩
    ///
    /// This is computationally expensive and requires multiple circuit evaluations.
    pub fn estimate_fisher_element<F>(
        &self,
        params: &[f64],
        i: usize,
        j: usize,
        overlap_fn: F,
    ) -> f64
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let s = self.gradient_estimator.shift;

        // Compute using parameter shifts
        // This is a simplified approximation
        let mut params_i_plus = params.to_vec();
        params_i_plus[i] += s;

        let mut params_j_plus = params.to_vec();
        params_j_plus[j] += s;

        let mut params_ij_plus = params.to_vec();
        params_ij_plus[i] += s;
        params_ij_plus[j] += s;

        // Approximation using overlaps
        let f_ij = (overlap_fn(&params_i_plus, &params_j_plus)
            - overlap_fn(params, &params_j_plus)
            - overlap_fn(&params_i_plus, params)
            + overlap_fn(params, params))
            / (s * s);

        f_ij
    }
}

impl Default for NaturalGradientEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_estimator_new() {
        let estimator = GradientEstimator::new();
        assert!((estimator.shift - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_shifted_params() {
        let estimator = GradientEstimator::new();
        let params = vec![0.5, 1.0, 1.5];

        let shifted = estimator.shifted_params(&params);

        assert_eq!(shifted.len(), 6); // 2 * 3 params

        // First pair: shifts for param 0
        assert!((shifted[0][0] - (0.5 + PI / 2.0)).abs() < 1e-10);
        assert!((shifted[1][0] - (0.5 - PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_quadratic() {
        // f(x, y) = x² + y² - gradient should be (2x, 2y)
        let cost_fn = |p: &[f64]| p[0] * p[0] + p[1] * p[1];

        let estimator = GradientEstimator::new();
        let params = vec![1.0, 2.0];

        // Use finite difference instead (parameter shift is for quantum circuits)
        let grad = estimator.finite_difference(&params, cost_fn, 1e-5);

        // Expected: (2.0, 4.0)
        assert!((grad[0] - 2.0).abs() < 1e-4);
        assert!((grad[1] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_gradient_from_batch() {
        let estimator = GradientEstimator::new();

        // Results: [f+, f-, f+, f-] for 2 parameters
        let results = vec![2.0, 0.0, 3.0, 1.0];
        let gradient = estimator.gradient_from_batch(&results, 2);

        // grad[0] = (2.0 - 0.0) * 0.5 = 1.0
        // grad[1] = (3.0 - 1.0) * 0.5 = 1.0
        assert_eq!(gradient.len(), 2);
        assert!((gradient[0] - 1.0).abs() < 1e-10);
        assert!((gradient[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_gradient_with_central() {
        let estimator = BatchGradientEstimator::new().with_central_point();

        let params = vec![0.5, 1.0];
        let batch = estimator.generate_batch(&params);

        // 1 central + 2*2 shifted = 5
        assert_eq!(batch.len(), 5);
        assert_eq!(batch[0], params); // First is central
    }

    #[test]
    fn test_batch_circuit_count() {
        let estimator = BatchGradientEstimator::new();
        assert_eq!(estimator.circuit_count(4), 8); // 2 * 4

        let estimator_central = BatchGradientEstimator::new().with_central_point();
        assert_eq!(estimator_central.circuit_count(4), 9); // 2 * 4 + 1
    }

    #[test]
    fn test_extract_gradient() {
        let estimator = BatchGradientEstimator::new().with_central_point();

        // Results: [central, f+_0, f-_0, f+_1, f-_1]
        let results = vec![0.5, 2.0, 0.0, 3.0, 1.0];
        let (cost, gradient) = estimator.extract_gradient(&results, 2);

        assert_eq!(cost, Some(0.5));
        assert_eq!(gradient.len(), 2);
    }
}
