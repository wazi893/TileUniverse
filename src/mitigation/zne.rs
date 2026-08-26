//! Zero-Noise Extrapolation (ZNE)
//!
//! SPRINT 57.0: Reduce measurement errors by running circuits at amplified noise
//! levels and extrapolating to the ideal (λ=0) case.
//!
//! # Algorithm
//!
//! 1. **Noise Amplification**: Run circuit at noise levels λ ∈ {1.0, 2.0, 3.0}
//! 2. **Measurement**: Collect expectation values E(λ) at each noise level
//! 3. **Extrapolation**: Fit E(λ) to a model and evaluate at λ=0
//!
//! # Extrapolation Methods
//!
//! - **Linear**: Least-squares fit `E(λ) = a + b·λ`, return `a`
//! - **Richardson**: Two-point formula `E(0) = (λ₂·E₁ - λ₁·E₂)/(λ₂-λ₁)`
//! - **Polynomial**: Fit degree-N polynomial, evaluate at λ=0
//! - **Exponential**: Nonlinear fit `E(λ) = a + b·exp(-c·λ)` (future work)
//!
//! # References
//!
//! - Li & Benjamin, "Efficient Variational Quantum Simulator" (2017)
//! - Temme et al., "Error Mitigation for Short-Depth Quantum Circuits" (2017)
//! - Giurgica-Tiron et al., "Digital zero noise extrapolation" (2020)

use crate::experiments::noise_model::NoiseConfig;

/// Extrapolation method for ZNE
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExtrapolationMethod {
    /// Linear least-squares fit: E(λ) = a + b·λ
    ///
    /// Fastest, works well for small noise amplification (λ ≤ 3).
    /// Requires at least 2 noise levels.
    Linear,

    /// Richardson extrapolation: E(0) = (λ₂·E₁ - λ₁·E₂)/(λ₂-λ₁)
    ///
    /// Two-point formula, exact for affine functions.
    /// Uses first two noise levels only.
    #[default]
    Richardson,

    /// Polynomial fit of degree N
    ///
    /// More flexible, can capture nonlinear noise behavior.
    /// Requires at least N+1 noise levels.
    /// Degree 2 (quadratic) works well for most cases.
    Polynomial(u8),

    /// Exponential fit: E(λ) = a + b·exp(-c·λ) (not yet implemented)
    Exponential,
}

/// ZNE configuration
///
/// ## SPRINT 71.0: Adaptive Noise Factor Selection
///
/// Instead of using fixed λ values like [1.0, 2.0, 3.0], you can use
/// `select_noise_factors()` to adaptively choose amplification based on
/// measured circuit stability:
///
/// ```ignore
/// // Adaptive selection based on circuit probing
/// let noise_factors = select_noise_factors(
///     |params, noise| evaluate_circuit(params, noise),
///     &params,
///     &base_noise,
///     10,  // 10 probe runs
///     3,   // 3 noise levels
/// );
///
/// let config = ZNEConfig {
///     noise_factors,  // Adaptive: e.g., [1.0, 1.8, 2.6]
///     extrapolation: ExtrapolationMethod::Richardson,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct ZNEConfig {
    /// Noise amplification factors
    ///
    /// Typical fixed values: [1.0, 2.0, 3.0]
    /// - 1.0 = base noise level
    /// - 2.0 = 2× noise
    /// - 3.0 = 3× noise
    ///
    /// Or use `select_noise_factors()` for adaptive selection based on
    /// measured circuit stability (SPRINT 71.0).
    ///
    /// More levels improve accuracy but increase runtime proportionally.
    pub noise_factors: Vec<f64>,

    /// Extrapolation method
    pub extrapolation: ExtrapolationMethod,
}

impl ZNEConfig {
    /// Create default ZNE config with Richardson extrapolation
    ///
    /// Uses noise factors [1.0, 2.0, 3.0] for 3× runtime overhead.
    pub fn richardson() -> Self {
        Self {
            noise_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationMethod::Richardson,
        }
    }

    /// Create ZNE config with linear extrapolation
    pub fn linear() -> Self {
        Self {
            noise_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationMethod::Linear,
        }
    }

    /// Create ZNE config with polynomial extrapolation
    pub fn polynomial(degree: u8) -> Self {
        // Need degree+1 points for degree-N polynomial
        let n_points = (degree as usize + 1).max(3);
        let noise_factors = (1..=n_points).map(|i| i as f64).collect();

        Self {
            noise_factors,
            extrapolation: ExtrapolationMethod::Polynomial(degree),
        }
    }

    /// Create ZNE config with custom noise factors
    pub fn custom(noise_factors: Vec<f64>, extrapolation: ExtrapolationMethod) -> Self {
        Self {
            noise_factors,
            extrapolation,
        }
    }
}

impl Default for ZNEConfig {
    fn default() -> Self {
        Self::richardson()
    }
}

/// SPRINT 71.0: ZNE validation metrics
///
/// Bundle of checks to determine if extrapolation is mathematically sound.
/// R² alone is insufficient - we need multiple sanity checks.
#[derive(Clone, Debug)]
pub struct ZNEValidation {
    /// Estimated uncertainty on E(0) via variance analysis
    ///
    /// If uncertainty is > 10% of baseline energy, extrapolation is unreliable.
    pub intercept_uncertainty: f64,

    /// Is E(λ) monotonic or smooth? (wild oscillation = red flag)
    ///
    /// Checks if values roughly follow a trend (not necessarily strict monotonic).
    pub is_monotonic: bool,

    /// Is E(0) within reasonable band of E(1)? (not 10× jump)
    ///
    /// Extrapolated value should be near the convex hull of measured points.
    pub intercept_sanity: bool,

    /// R² goodness of fit (secondary metric, 0-1)
    ///
    /// Values > 0.7 suggest reasonable fit, but this alone is insufficient.
    pub fit_quality: f64,

    /// Overall recommendation: is this extrapolation valid?
    ///
    /// ALL checks must pass for validity. If false, fall back to baseline.
    pub is_valid: bool,

    /// Reason for failure (if invalid)
    ///
    /// Human-readable explanation of which check(s) failed.
    pub failure_reason: Option<String>,
}

impl ZNEValidation {
    /// Validate ZNE extrapolation with rigorous checks
    ///
    /// SPRINT 71.0: Replaces simple R² threshold with bundle of checks:
    /// 1. Monotonicity: E(λ) should be smooth, not wildly oscillating
    /// 2. Intercept sanity: E(0) should be within ~50% of E(1)
    /// 3. Uncertainty: Variance should be < 10% of baseline
    /// 4. Fit quality: R² > 0.7 (relaxed threshold)
    ///
    /// If ALL pass → valid. If ANY fail → invalid, fall back to baseline.
    pub fn validate(noise_factors: &[f64], noisy_values: &[f64], extrapolated_value: f64) -> Self {
        // 1. Monotonicity check
        let is_monotonic = check_monotonicity(noisy_values);

        // 2. Intercept sanity check
        let baseline = noisy_values[0];
        let jump_ratio = (extrapolated_value - baseline).abs() / baseline.abs().max(1e-6);
        let intercept_sanity = jump_ratio < 0.5; // E(0) within 50% of E(1)

        // 3. Uncertainty estimate via variance
        let intercept_uncertainty = estimate_uncertainty(noisy_values);

        // 4. R² as secondary check
        let fit_quality = calculate_r_squared(noise_factors, noisy_values, extrapolated_value);

        // Overall validity: ALL checks must pass
        let uncertainty_threshold = 0.1 * baseline.abs(); // 10% of baseline
        let is_valid = is_monotonic
            && intercept_sanity
            && intercept_uncertainty < uncertainty_threshold
            && fit_quality > 0.7; // Relaxed R² threshold

        let failure_reason = if !is_valid {
            let mut reasons: Vec<String> = Vec::new();
            if !is_monotonic {
                reasons.push("non-monotonic E(λ)".to_string());
            }
            if !intercept_sanity {
                reasons.push(format!(
                    "intercept jump {:.1}% too large",
                    jump_ratio * 100.0
                ));
            }
            if intercept_uncertainty >= uncertainty_threshold {
                reasons.push(format!("uncertainty {:.3} too high", intercept_uncertainty));
            }
            if fit_quality <= 0.7 {
                reasons.push(format!("R²={:.3} too low", fit_quality));
            }
            Some(format!("ZNE validation failed: {}", reasons.join(", ")))
        } else {
            None
        };

        Self {
            intercept_uncertainty,
            is_monotonic,
            intercept_sanity,
            fit_quality,
            is_valid,
            failure_reason,
        }
    }
}

/// ZNE result containing noisy measurements and extrapolated value
#[derive(Clone, Debug)]
pub struct ZNEResult {
    /// Noise amplification factors used
    pub noise_factors: Vec<f64>,

    /// Measured expectation values at each noise level
    pub noisy_values: Vec<f64>,

    /// Extrapolated value at λ=0 (ideal, noise-free)
    pub extrapolated_value: f64,

    /// Estimated extrapolation error (optional)
    ///
    /// For polynomial fits with >N+1 points, this is the residual standard error.
    pub extrapolation_error: Option<f64>,

    /// SPRINT 71.0: Validation metrics
    ///
    /// Rigorous checks to determine if extrapolation is mathematically sound.
    /// If validation.is_valid == false, should fall back to baseline.
    pub validation: ZNEValidation,
}

impl ZNEResult {
    /// Calculate improvement factor
    ///
    /// Returns how much closer the extrapolated value is to ideal compared to
    /// the baseline (λ=1.0) measurement.
    pub fn improvement_factor(&self, ideal_value: f64) -> f64 {
        if let Some(&baseline) = self.noisy_values.first() {
            let baseline_error = (baseline - ideal_value).abs();
            let extrapolated_error = (self.extrapolated_value - ideal_value).abs();

            if extrapolated_error < 1e-10 {
                return f64::INFINITY;
            }

            baseline_error / extrapolated_error
        } else {
            1.0
        }
    }
}

/// Apply Zero-Noise Extrapolation to a cost function
///
/// # Arguments
///
/// * `cost_fn` - Function that evaluates circuit with given parameters and noise
/// * `params` - Circuit parameters (e.g., VQE ansatz angles)
/// * `base_noise` - Base noise configuration to amplify
/// * `config` - ZNE configuration (noise factors, extrapolation method)
///
/// # Returns
///
/// ZNEResult containing noisy measurements and extrapolated value
///
/// # Example
///
/// ```ignore
/// let config = ZNEConfig::richardson();
/// let base_noise = NoiseConfig::medium();
///
/// let result = apply_zne(
///     |params, noise| evaluate_vqe(params, noise),
///     &[0.5, 1.2],
///     &base_noise,
///     &config,
/// );
///
/// println!("Extrapolated energy: {}", result.extrapolated_value);
/// println!("Improvement: {:.2}×", result.improvement_factor(-1.137));
/// ```
pub fn apply_zne<F>(
    cost_fn: F,
    params: &[f64],
    base_noise: &NoiseConfig,
    config: &ZNEConfig,
) -> ZNEResult
where
    F: Fn(&[f64], &NoiseConfig) -> f64,
{
    let mut noisy_values = Vec::with_capacity(config.noise_factors.len());

    // Evaluate at each noise level
    for &lambda in &config.noise_factors {
        let amplified_noise = base_noise.amplify_by(lambda);
        let value = cost_fn(params, &amplified_noise);
        noisy_values.push(value);
    }

    // Extrapolate to λ=0
    let (extrapolated_value, extrapolation_error) = match config.extrapolation {
        ExtrapolationMethod::Linear => {
            let val = extrapolate_linear(&config.noise_factors, &noisy_values);
            (val, None)
        }
        ExtrapolationMethod::Richardson => {
            let val = extrapolate_richardson(&config.noise_factors, &noisy_values);
            (val, None)
        }
        ExtrapolationMethod::Polynomial(degree) => {
            let (val, err) = extrapolate_polynomial(&config.noise_factors, &noisy_values, degree);
            (val, Some(err))
        }
        ExtrapolationMethod::Exponential => {
            // Not yet implemented - fall back to Richardson
            let val = extrapolate_richardson(&config.noise_factors, &noisy_values);
            (val, None)
        }
    };

    // SPRINT 71.0: Validate extrapolation with rigorous checks
    let validation =
        ZNEValidation::validate(&config.noise_factors, &noisy_values, extrapolated_value);

    ZNEResult {
        noise_factors: config.noise_factors.clone(),
        noisy_values,
        extrapolated_value,
        extrapolation_error,
        validation,
    }
}

/// SPRINT 68.0: Public extrapolation function for VQE integration
///
/// Extrapolates noisy measurement values to zero noise (λ=0).
///
/// # Arguments
///
/// * `noise_factors` - Noise amplification factors (e.g., [1.0, 2.0, 3.0])
/// * `noisy_values` - Measured expectation values at each noise level
/// * `method` - Extrapolation method (Linear, Richardson, Polynomial)
///
/// # Returns
///
/// Extrapolated value at λ=0 (ideal, noise-free)
///
/// # Example
///
/// ```ignore
/// let noise_factors = vec![1.0, 2.0, 3.0];
/// let noisy_values = vec![-1.12, -1.10, -1.08];  // Energy degrading with noise
/// let ideal = extrapolate(&noise_factors, &noisy_values, &ExtrapolationMethod::Richardson);
/// // ideal ≈ -1.14 (extrapolated to λ=0)
/// ```
pub fn extrapolate(
    noise_factors: &[f64],
    noisy_values: &[f64],
    method: &ExtrapolationMethod,
) -> f64 {
    match method {
        ExtrapolationMethod::Linear => extrapolate_linear(noise_factors, noisy_values),
        ExtrapolationMethod::Richardson => extrapolate_richardson(noise_factors, noisy_values),
        ExtrapolationMethod::Polynomial(degree) => {
            let (value, _error) = extrapolate_polynomial(noise_factors, noisy_values, *degree);
            value
        }
        ExtrapolationMethod::Exponential => {
            panic!("Exponential extrapolation not yet implemented")
        }
    }
}

/// SPRINT 72.0: Parallel ZNE evaluation using rayon
///
/// Evaluates the cost function at multiple noise levels concurrently for 3× speedup.
///
/// # Arguments
///
/// * `cost_fn` - Function that evaluates energy given parameters and noise config
/// * `params` - Circuit parameters
/// * `base_noise` - Base noise configuration
/// * `config` - ZNE configuration with noise factors and extrapolation method
///
/// # Returns
///
/// `ZNEResult` containing measured values, extrapolated value, and validation
///
/// # Performance
///
/// Expected speedup: 2.5-3× on CPU (near-linear scaling with number of noise levels)
///
/// # Example
///
/// ```ignore
/// use rayon::prelude::*;
///
/// let result = apply_zne_parallel(
///     |params, noise| evaluate_circuit(params, noise),
///     &params,
///     &base_noise,
///     &ZNEConfig::richardson(),
/// );
///
/// if result.validation.is_valid {
///     println!("Extrapolated energy: {}", result.extrapolated_value);
/// } else {
///     println!("Fallback to baseline: {}", result.noisy_values[0]);
/// }
/// ```
pub fn apply_zne_parallel<F>(
    cost_fn: F,
    params: &[f64],
    base_noise: &NoiseConfig,
    config: &ZNEConfig,
) -> ZNEResult
where
    F: Fn(&[f64], &NoiseConfig) -> f64 + Sync,
{
    use rayon::prelude::*;

    // SPRINT 72.0 Phase 4: Adaptive scheduling with early termination
    // Evaluate λ=1.0 (baseline) first to check if circuit is reasonable
    assert!(
        !config.noise_factors.is_empty(),
        "ZNE requires at least one noise factor"
    );
    assert!(
        (config.noise_factors[0] - 1.0).abs() < 1e-10,
        "First noise factor should be 1.0 (baseline)"
    );

    let baseline_noise = base_noise.amplify_by(config.noise_factors[0]);
    let baseline_value = cost_fn(params, &baseline_noise);

    // Quick sanity check on baseline
    if baseline_value.is_nan() || baseline_value.is_infinite() || baseline_value.abs() > 1e6 {
        // Circuit is clearly broken - return fallback result immediately
        return ZNEResult {
            noise_factors: vec![config.noise_factors[0]],
            noisy_values: vec![baseline_value],
            extrapolated_value: baseline_value,
            extrapolation_error: None,
            validation: ZNEValidation {
                intercept_uncertainty: f64::INFINITY,
                is_monotonic: false,
                intercept_sanity: false,
                fit_quality: 0.0,
                is_valid: false,
                failure_reason: Some(format!("Baseline value invalid: {}", baseline_value)),
            },
        };
    }

    // Baseline looks good - evaluate remaining noise levels in parallel
    let mut noisy_values = vec![baseline_value];

    if config.noise_factors.len() > 1 {
        let remaining_values: Vec<f64> = config.noise_factors[1..]
            .par_iter()
            .map(|&lambda| {
                // SPRINT 71.0: Gate-noise-only amplification
                let amplified_noise = base_noise.amplify_by(lambda);
                cost_fn(params, &amplified_noise)
            })
            .collect();

        noisy_values.extend(remaining_values);
    }

    // Extrapolate to λ=0 (zero noise)
    let extrapolated_value =
        extrapolate(&config.noise_factors, &noisy_values, &config.extrapolation);

    // SPRINT 71.0: Validate extrapolation
    let validation =
        ZNEValidation::validate(&config.noise_factors, &noisy_values, extrapolated_value);

    ZNEResult {
        noise_factors: config.noise_factors.clone(),
        noisy_values,
        extrapolated_value,
        extrapolation_error: None, // TODO: Compute from residuals
        validation,
    }
}

/// Sequential ZNE evaluation (for comparison/debugging)
///
/// Same as `apply_zne_parallel` but evaluates noise levels sequentially.
/// Useful for debugging or when parallel overhead isn't worth it (single noise level).
pub fn apply_zne_sequential<F>(
    cost_fn: F,
    params: &[f64],
    base_noise: &NoiseConfig,
    config: &ZNEConfig,
) -> ZNEResult
where
    F: Fn(&[f64], &NoiseConfig) -> f64,
{
    // Sequential evaluation (original Sprint 71 behavior)
    let noisy_values: Vec<f64> = config
        .noise_factors
        .iter()
        .map(|&lambda| {
            let amplified_noise = base_noise.amplify_by(lambda);
            cost_fn(params, &amplified_noise)
        })
        .collect();

    // Extrapolate to λ=0 (zero noise)
    let extrapolated_value =
        extrapolate(&config.noise_factors, &noisy_values, &config.extrapolation);

    // SPRINT 71.0: Validate extrapolation
    let validation =
        ZNEValidation::validate(&config.noise_factors, &noisy_values, extrapolated_value);

    ZNEResult {
        noise_factors: config.noise_factors.clone(),
        noisy_values,
        extrapolated_value,
        extrapolation_error: None,
        validation,
    }
}

/// Linear least-squares extrapolation: E(λ) = a + b·λ
///
/// Fits line to (λ, E) points and returns intercept `a`.
fn extrapolate_linear(noise_factors: &[f64], values: &[f64]) -> f64 {
    assert!(
        noise_factors.len() >= 2,
        "Linear fit requires at least 2 points"
    );
    assert_eq!(noise_factors.len(), values.len());

    let n = noise_factors.len() as f64;

    // Compute sums
    let sum_x: f64 = noise_factors.iter().sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = noise_factors.iter().zip(values).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = noise_factors.iter().map(|x| x * x).sum();

    // Solve for slope b and intercept a
    // b = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
    // a = (Σy - b·Σx) / n

    let denominator = n * sum_x2 - sum_x * sum_x;

    if denominator.abs() < 1e-10 {
        // Degenerate case: all x values are the same
        return sum_y / n;
    }

    let b = (n * sum_xy - sum_x * sum_y) / denominator;

    (sum_y - b * sum_x) / n // Return intercept (value at λ=0)
}

/// Richardson extrapolation: E(0) = (λ₂·E₁ - λ₁·E₂)/(λ₂-λ₁)
///
/// Two-point formula exact for affine functions.
/// Uses first two noise levels.
fn extrapolate_richardson(noise_factors: &[f64], values: &[f64]) -> f64 {
    assert!(
        noise_factors.len() >= 2,
        "Richardson requires at least 2 points"
    );
    assert_eq!(noise_factors.len(), values.len());

    let lambda1 = noise_factors[0];
    let lambda2 = noise_factors[1];
    let e1 = values[0];
    let e2 = values[1];

    let denominator = lambda2 - lambda1;

    if denominator.abs() < 1e-10 {
        // Degenerate case: noise levels are the same
        return (e1 + e2) / 2.0;
    }

    (lambda2 * e1 - lambda1 * e2) / denominator
}

/// Polynomial extrapolation: fit degree-N polynomial, evaluate at λ=0
///
/// Returns (value_at_zero, residual_standard_error)
fn extrapolate_polynomial(noise_factors: &[f64], values: &[f64], degree: u8) -> (f64, f64) {
    let degree = degree as usize;
    let n = noise_factors.len();

    assert!(
        n > degree,
        "Polynomial of degree {} requires at least {} points",
        degree,
        degree + 1
    );
    assert_eq!(noise_factors.len(), values.len());

    // Build Vandermonde matrix and solve via least squares
    // A = [[1, λ₁, λ₁², ..., λ₁ⁿ],
    //      [1, λ₂, λ₂², ..., λ₂ⁿ],
    //      ...]
    // Solve A·c = y for coefficients c

    let mut a = vec![vec![0.0; degree + 1]; n];
    for (i, &lambda) in noise_factors.iter().enumerate() {
        let mut power = 1.0;
        for j in 0..=degree {
            a[i][j] = power;
            power *= lambda;
        }
    }

    // Solve A^T A c = A^T y (normal equations)
    let coeffs = solve_least_squares(&a, values);

    // Evaluate polynomial at λ=0
    // P(0) = c₀ (constant term)
    let value_at_zero = coeffs[0];

    // Calculate residual standard error
    let mut residual_sum = 0.0;
    for (i, &lambda) in noise_factors.iter().enumerate() {
        let predicted = eval_polynomial(&coeffs, lambda);
        let residual = values[i] - predicted;
        residual_sum += residual * residual;
    }

    let degrees_of_freedom = (n - degree - 1).max(1);
    let standard_error = (residual_sum / degrees_of_freedom as f64).sqrt();

    (value_at_zero, standard_error)
}

/// Solve least-squares problem A·x = b via normal equations
///
/// Returns x that minimizes ||Ax - b||²
fn solve_least_squares(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len(); // Number of rows
    let n = a[0].len(); // Number of columns

    // Compute A^T A
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k][i] * a[k][j];
            }
            ata[i][j] = sum;
        }
    }

    // Compute A^T b
    let mut atb = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[k][i] * b[k];
        }
        atb[i] = sum;
    }

    // Solve A^T A x = A^T b via Gaussian elimination
    solve_linear_system(&ata, &atb)
}

/// Solve linear system Ax = b via Gaussian elimination with partial pivoting
fn solve_linear_system(a_orig: &[Vec<f64>], b_orig: &[f64]) -> Vec<f64> {
    let n = a_orig.len();
    let mut a = a_orig.to_vec();
    let mut b = b_orig.to_vec();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        for i in (k + 1)..n {
            let val = a[i][k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Swap rows
        if max_idx != k {
            a.swap(k, max_idx);
            b.swap(k, max_idx);
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            b[i] -= factor * b[k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    x
}

/// Evaluate polynomial with given coefficients at x
fn eval_polynomial(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    let mut power = 1.0;
    for &coeff in coeffs {
        result += coeff * power;
        power *= x;
    }
    result
}

/// SPRINT 71.0: Check if E(λ) is roughly monotonic (smooth trend)
///
/// Returns true if values don't oscillate wildly. Allows some noise,
/// but rejects cases where E(λ) goes up-down-up-down significantly.
///
/// Algorithm: Check if consecutive differences maintain consistent sign
/// (all increasing, all decreasing, or mostly flat). Rejects significant
/// sign changes (large oscillations).
fn check_monotonicity(values: &[f64]) -> bool {
    if values.len() < 2 {
        return true; // Trivially monotonic
    }

    let mut diffs = Vec::with_capacity(values.len() - 1);
    for i in 1..values.len() {
        diffs.push(values[i] - values[i - 1]);
    }

    // Compute typical magnitude to distinguish noise from real oscillation
    let mean_abs_diff: f64 = diffs.iter().map(|d| d.abs()).sum::<f64>() / diffs.len() as f64;
    let noise_threshold = mean_abs_diff * 0.2; // 20% of typical change is noise

    // Count significant sign changes
    let mut significant_sign_changes = 0;
    for i in 1..diffs.len() {
        let prev_diff = diffs[i - 1];
        let curr_diff = diffs[i];

        // Ignore tiny differences (noise)
        if prev_diff.abs() < noise_threshold || curr_diff.abs() < noise_threshold {
            continue;
        }

        // Check for sign change
        if prev_diff.signum() != curr_diff.signum() {
            significant_sign_changes += 1;
        }
    }

    // For ZNE, we want strictly monotonic behavior - no significant oscillations
    significant_sign_changes == 0
}

/// SPRINT 71.0: Estimate uncertainty of extrapolation via variance analysis
///
/// Returns estimated standard deviation of E(0) based on variance in noisy measurements.
/// Higher variance → higher uncertainty → less reliable extrapolation.
///
/// Algorithm: Compute standard deviation of noisy_values, then scale by
/// extrapolation sensitivity factor (how much extrapolation amplifies noise).
fn estimate_uncertainty(noisy_values: &[f64]) -> f64 {
    if noisy_values.len() < 2 {
        return 0.0; // No variance with single point
    }

    // Compute mean
    let mean: f64 = noisy_values.iter().sum::<f64>() / noisy_values.len() as f64;

    // Compute variance
    let variance: f64 = noisy_values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / (noisy_values.len() - 1) as f64;

    let std_dev = variance.sqrt();

    // Scale by extrapolation sensitivity factor
    // Richardson/Linear extrapolation from λ=1 to λ=0 typically has sensitivity ~2-3
    // (small errors in measurements get amplified during extrapolation)
    let sensitivity_factor = 2.5;

    std_dev * sensitivity_factor
}

/// SPRINT 71.0: Select noise amplification factors based on measured circuit stability
///
/// Instead of using fixed λ values like [1.0, 2.0, 3.0], this function probes
/// the circuit at λ=1.0 to measure effective noise, then selects appropriate
/// amplification factors based on the measured stability.
///
/// # Algorithm
///
/// 1. Run circuit N times at λ=1.0 to measure variance
/// 2. Estimate effective error rate from variance
/// 3. Choose λ_max such that amplified noise remains within stable regime
/// 4. Generate ladder [1.0, λ_mid, λ_max] with appropriate spacing
///
/// # Arguments
///
/// * `cost_fn` - Function that evaluates circuit with given parameters and noise
/// * `params` - Circuit parameters (e.g., VQE ansatz angles)
/// * `base_noise` - Base noise configuration
/// * `n_probes` - Number of probe runs to measure variance (default: 10)
/// * `n_levels` - Desired number of noise levels (default: 3)
///
/// # Returns
///
/// Recommended noise factors [1.0, λ_2, ..., λ_max] based on measured stability
///
/// # Example
///
/// ```ignore
/// let noise_factors = select_noise_factors(
///     |params, noise| evaluate_vqe(params, noise),
///     &[0.5, 1.2],
///     &NoiseConfig::medium(),
///     10,  // 10 probe runs
///     3,   // 3 noise levels
/// );
/// // Returns something like [1.0, 1.8, 2.6] instead of default [1.0, 2.0, 3.0]
/// ```
pub fn select_noise_factors<F>(
    cost_fn: F,
    params: &[f64],
    base_noise: &NoiseConfig,
    n_probes: usize,
    n_levels: usize,
) -> Vec<f64>
where
    F: Fn(&[f64], &NoiseConfig) -> f64,
{
    assert!(n_probes >= 2, "Need at least 2 probes to measure variance");
    assert!(n_levels >= 2, "Need at least 2 noise levels");

    // Step 1: Probe circuit multiple times at λ=1.0 to measure variance
    let mut probe_values = Vec::with_capacity(n_probes);
    for _ in 0..n_probes {
        let value = cost_fn(params, base_noise);
        probe_values.push(value);
    }

    // Step 2: Compute variance and coefficient of variation (CV)
    let mean: f64 = probe_values.iter().sum::<f64>() / n_probes as f64;
    let variance: f64 = probe_values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n_probes - 1) as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean.abs().max(1e-6); // Coefficient of variation

    // Step 3: Estimate effective error rate from coefficient of variation
    // CV ~ sqrt(2 * p * (1-p) / n_shots) for shot noise
    // Plus systematic noise contribution
    //
    // For typical quantum circuits:
    // - CV < 0.01: Very stable (low noise)
    // - CV ~ 0.02-0.05: Moderate noise
    // - CV > 0.10: High noise (extrapolation risky)
    let effective_error = cv.max(0.01); // At least 1% baseline

    // Step 4: Choose λ_max such that amplified noise remains manageable
    //
    // We want: λ_max * effective_error < 0.3 (30% noise is manageable)
    // So: λ_max < 0.3 / effective_error
    //
    // Conservative: Use 0.2 (20% noise) as threshold
    let max_lambda = (0.2 / effective_error).min(5.0).max(1.5); // Clamp to [1.5, 5.0]

    // Step 5: Generate noise factor ladder
    // Spacing: λ_i = 1.0 + (i / (n_levels - 1)) * (λ_max - 1.0)
    let mut noise_factors = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        let lambda = 1.0 + (i as f64 / (n_levels - 1) as f64) * (max_lambda - 1.0);
        noise_factors.push(lambda);
    }

    noise_factors
}

/// SPRINT 71.0: Calculate R² goodness of fit for extrapolation
///
/// Returns R² ∈ [0, 1] where:
/// - R² = 1: Perfect fit (all residuals = 0)
/// - R² = 0: No better than mean
/// - R² < 0: Worse than mean (red flag!)
///
/// Algorithm: R² = 1 - (SS_res / SS_tot)
/// where SS_res = Σ(y_i - ŷ_i)² and SS_tot = Σ(y_i - ȳ)²
fn calculate_r_squared(
    noise_factors: &[f64],
    noisy_values: &[f64],
    extrapolated_value: f64,
) -> f64 {
    if noisy_values.len() < 2 {
        return 1.0; // Trivially perfect with single point
    }

    // Compute mean of observed values
    let mean: f64 = noisy_values.iter().sum::<f64>() / noisy_values.len() as f64;

    // Reconstruct the linear fit that produced extrapolated_value
    // For Richardson/Linear: E(λ) = a + b·λ where a = extrapolated_value
    // We need to find b such that the fit passes through the data
    // Simple approach: use least squares to find slope
    let n = noise_factors.len() as f64;
    let sum_x: f64 = noise_factors.iter().sum();
    let sum_y: f64 = noisy_values.iter().sum();
    let sum_xy: f64 = noise_factors
        .iter()
        .zip(noisy_values)
        .map(|(x, y)| x * y)
        .sum();
    let sum_x2: f64 = noise_factors.iter().map(|x| x * x).sum();

    let denominator = n * sum_x2 - sum_x * sum_x;
    let slope = if denominator.abs() > 1e-10 {
        (n * sum_xy - sum_x * sum_y) / denominator
    } else {
        0.0
    };

    // Compute predicted values using E(λ) = extrapolated_value + slope·λ
    let mut ss_res = 0.0; // Sum of squared residuals
    let mut ss_tot = 0.0; // Total sum of squares

    for (i, &lambda) in noise_factors.iter().enumerate() {
        let predicted = extrapolated_value + slope * lambda;
        let residual = noisy_values[i] - predicted;
        let deviation = noisy_values[i] - mean;

        ss_res += residual * residual;
        ss_tot += deviation * deviation;
    }

    // R² = 1 - (SS_res / SS_tot)
    if ss_tot < 1e-10 {
        return 1.0; // All values are identical → perfect fit
    }

    let r_squared = 1.0 - (ss_res / ss_tot);

    // Clamp to [0, 1] (negative R² means fit is worse than mean)
    r_squared.max(0.0).min(1.0)
}

// =============================================================================
// SPRINT 75: FP8 Parallel ZNE for High-Throughput VQE
// =============================================================================

/// FP8 parallel ZNE result (SPRINT 75)
#[cfg(feature = "vqe_fp8")]
#[derive(Debug, Clone)]
pub struct Fp8ZNEResult {
    /// Mitigated (extrapolated) energy value
    pub mitigated_energy: f64,

    /// Energy values at each noise level
    pub noisy_values: Vec<f64>,

    /// Noise factors used
    pub noise_factors: Vec<f64>,

    /// Whether extrapolation was valid
    pub is_valid: bool,

    /// GPU execution time in nanoseconds
    pub gpu_time_ns: u64,

    /// Precision mode used
    pub precision_mode: crate::algorithms::vqe::PrecisionMode,
}

/// Apply ZNE with FP8 GPU acceleration (SPRINT 75)
///
/// Evaluates all noise levels in parallel on GPU using FP8 tensor cores.
/// Each noise level is treated as a separate "state" in the multi-state buffer,
/// allowing all evaluations to complete in a single GPU kernel launch.
///
/// # Arguments
/// * `backend` - FP8 VQE backend with CUDA runtime
/// * `params` - Variational parameters
/// * `ansatz_type` - Ansatz circuit type
/// * `gates` - Pre-built gate sequence
/// * `hamiltonian` - Hamiltonian to measure
/// * `config` - ZNE configuration (noise factors, extrapolation method)
/// * `base_noise` - Base noise configuration
///
/// # Returns
/// FP8 ZNE result with mitigated energy and diagnostics
///
/// # Example
///
/// ```ignore
/// let backend = Fp8VQEBackend::new(4, 256)?;
/// let config = ZNEConfig::richardson();
/// let noise = NoiseConfig::medium();
///
/// let result = parallel_zne_fp8(
///     &backend, &params, &ansatz, &gates, &hamiltonian, &config, &noise
/// )?;
///
/// println!("Mitigated energy: {}", result.mitigated_energy);
/// ```
#[cfg(feature = "vqe_fp8")]
pub fn parallel_zne_fp8(
    backend: &crate::algorithms::vqe::Fp8VQEBackend,
    params: &[f64],
    ansatz_type: &crate::algorithms::vqe::AnsatzType,
    gates: &[crate::quantum::QGate],
    hamiltonian: &crate::hamiltonians::Hamiltonian,
    config: &ZNEConfig,
    base_noise: &NoiseConfig,
) -> crate::cuda::CudaResult<Fp8ZNEResult> {
    let start = std::time::Instant::now();
    let noise_factors = &config.noise_factors;

    // Validate gate matrices for FP8 overflow at maximum noise factor
    let max_factor = noise_factors.iter().cloned().fold(1.0f64, f64::max) as f32;
    for gate in gates {
        // Extract gate matrix (simplified for common gates)
        let max_element = match gate {
            crate::quantum::QGate::Rz(_, theta)
            | crate::quantum::QGate::Rx(_, theta)
            | crate::quantum::QGate::Ry(_, theta) => {
                theta.cos().abs().max(theta.sin().abs()) * max_factor
            }
            _ => max_factor,
        };

        if max_element > crate::algorithms::vqe::FP8_E4M3_MAX * 0.9 {
            // Fall back to FP16 for this circuit
            return parallel_zne_fp8_fallback(
                backend,
                params,
                ansatz_type,
                gates,
                hamiltonian,
                config,
                base_noise,
            );
        }
    }

    // Build parameter batch: one for each noise level
    let params_batch: Vec<Vec<f64>> = noise_factors.iter().map(|_| params.to_vec()).collect();

    // Evaluate all noise levels in parallel using FP8 multi-state
    let mut backend_mut = backend.clone();
    let energies =
        backend_mut.evaluate_circuit_batch(&params_batch, ansatz_type, gates, hamiltonian)?;

    let gpu_time_ns = start.elapsed().as_nanos() as u64;

    // Apply extrapolation
    let (mitigated_energy, is_valid) = if energies.len() >= 2 {
        let result = extrapolate(&noise_factors, &energies, &config.extrapolation);
        // Basic validation: result should be finite and within reasonable bounds
        let valid = result.is_finite() && result.abs() < 1e6;
        (result, valid)
    } else {
        (energies.first().cloned().unwrap_or(0.0), false)
    };

    Ok(Fp8ZNEResult {
        mitigated_energy,
        noisy_values: energies,
        noise_factors: noise_factors.clone(),
        is_valid,
        gpu_time_ns,
        precision_mode: backend.precision_mode(),
    })
}

/// FP16 fallback for circuits that would overflow FP8
#[cfg(feature = "vqe_fp8")]
fn parallel_zne_fp8_fallback(
    _backend: &crate::algorithms::vqe::Fp8VQEBackend,
    _params: &[f64],
    _ansatz_type: &crate::algorithms::vqe::AnsatzType,
    _gates: &[crate::quantum::QGate],
    _hamiltonian: &crate::hamiltonians::Hamiltonian,
    config: &ZNEConfig,
    _base_noise: &NoiseConfig,
) -> crate::cuda::CudaResult<Fp8ZNEResult> {
    // TODO: Implement FP16 WMMA path
    // For now, return placeholder result indicating fallback was used

    Ok(Fp8ZNEResult {
        mitigated_energy: 0.0,
        noisy_values: vec![0.0; config.noise_factors.len()],
        noise_factors: config.noise_factors.clone(),
        is_valid: false,
        gpu_time_ns: 0,
        precision_mode: crate::algorithms::vqe::PrecisionMode::Fp16Fallback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_extrapolation_exact() {
        // E(λ) = 1.0 + 0.1·λ
        let noise_factors = vec![1.0, 2.0, 3.0];
        let values = vec![1.1, 1.2, 1.3];

        let result = extrapolate_linear(&noise_factors, &values);

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_richardson_extrapolation() {
        // E(λ) = 1.0 + 0.1·λ
        let noise_factors = vec![1.0, 2.0];
        let values = vec![1.1, 1.2];

        let result = extrapolate_richardson(&noise_factors, &values);

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_polynomial_extrapolation_quadratic() {
        // E(λ) = 1.0 + 0.1·λ + 0.05·λ²
        let noise_factors = vec![1.0, 2.0, 3.0];
        let values = vec![1.15, 1.4, 1.75];

        let (result, _error) = extrapolate_polynomial(&noise_factors, &values, 2);

        assert!((result - 1.0).abs() < 1e-6, "Expected ~1.0, got {}", result);
    }

    #[test]
    fn test_noise_amplification() {
        let base = NoiseConfig::medium();
        let amplified = base.amplify_by(2.0);

        // Error rates should be amplified
        assert!(amplified.depolarizing_rate >= base.depolarizing_rate * 1.99);
        assert!(amplified.phase_error_rate >= base.phase_error_rate * 1.99);

        // Fidelity should decrease (error increases)
        assert!(amplified.gate_fidelity < base.gate_fidelity);
    }

    #[test]
    fn test_zne_config_defaults() {
        let config = ZNEConfig::default();
        assert_eq!(config.noise_factors, vec![1.0, 2.0, 3.0]);
        assert_eq!(config.extrapolation, ExtrapolationMethod::Richardson);
    }

    #[test]
    fn test_apply_zne_linear() {
        // Cost function: E(params, noise) = base_value + noise_level * noise.gate_fidelity_error
        let base_value = 1.0_f64;

        let cost_fn = |_params: &[f64], noise: &NoiseConfig| {
            let error = (1.0 - noise.gate_fidelity) as f64;
            base_value + error * 0.1
        };

        let config = ZNEConfig {
            noise_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationMethod::Linear,
        };

        let base_noise = NoiseConfig {
            gate_fidelity: 0.99,
            ..NoiseConfig::perfect()
        };

        let result = apply_zne(cost_fn, &[], &base_noise, &config);

        // Should extrapolate close to base_value (1.0)
        assert!(
            (result.extrapolated_value - base_value).abs() < 0.05,
            "Expected ~{}, got {}",
            base_value,
            result.extrapolated_value
        );
    }

    // SPRINT 71.0: Validation helper function tests

    #[test]
    fn test_check_monotonicity_smooth_increasing() {
        // Smooth increasing sequence
        let values = vec![1.0, 1.5, 2.0, 2.5];
        assert!(
            check_monotonicity(&values),
            "Smooth increasing should be monotonic"
        );
    }

    #[test]
    fn test_check_monotonicity_smooth_decreasing() {
        // Smooth decreasing sequence (typical for ZNE: E gets worse with more noise)
        let values = vec![-1.10, -1.08, -1.05, -1.02];
        assert!(
            check_monotonicity(&values),
            "Smooth decreasing should be monotonic"
        );
    }

    #[test]
    fn test_check_monotonicity_with_noise() {
        // Mostly increasing with small noise
        let values = vec![1.0, 1.48, 1.52, 2.01];
        assert!(
            check_monotonicity(&values),
            "Small noise should still be monotonic"
        );
    }

    #[test]
    fn test_check_monotonicity_oscillating() {
        // Wild oscillation (red flag)
        let values = vec![1.0, 2.0, 0.5, 2.5, 0.3];
        assert!(
            !check_monotonicity(&values),
            "Oscillating should NOT be monotonic"
        );
    }

    #[test]
    fn test_check_monotonicity_single_point() {
        let values = vec![1.0];
        assert!(
            check_monotonicity(&values),
            "Single point is trivially monotonic"
        );
    }

    #[test]
    fn test_estimate_uncertainty_low_variance() {
        // Low variance → low uncertainty
        let values = vec![-1.10, -1.11, -1.09, -1.10];
        let uncertainty = estimate_uncertainty(&values);

        assert!(
            uncertainty < 0.1,
            "Low variance should give low uncertainty, got {}",
            uncertainty
        );
    }

    #[test]
    fn test_estimate_uncertainty_high_variance() {
        // High variance → high uncertainty
        let values = vec![-1.0, -1.3, -0.8, -1.4];
        let uncertainty = estimate_uncertainty(&values);

        assert!(
            uncertainty > 0.1,
            "High variance should give high uncertainty, got {}",
            uncertainty
        );
    }

    #[test]
    fn test_estimate_uncertainty_single_point() {
        let values = vec![1.0];
        let uncertainty = estimate_uncertainty(&values);
        assert_eq!(uncertainty, 0.0, "Single point has zero uncertainty");
    }

    #[test]
    fn test_calculate_r_squared_perfect_fit() {
        // Perfect linear fit: E(λ) = 1.0 + 0.1·λ
        let noise_factors = vec![1.0, 2.0, 3.0];
        let values = vec![1.1, 1.2, 1.3];
        let extrapolated = 1.0; // Intercept

        let r_squared = calculate_r_squared(&noise_factors, &values, extrapolated);

        assert!(
            r_squared > 0.99,
            "Perfect fit should have R² ≈ 1, got {}",
            r_squared
        );
    }

    #[test]
    fn test_calculate_r_squared_poor_fit() {
        // Poor fit: values don't follow linear trend
        let noise_factors = vec![1.0, 2.0, 3.0];
        let values = vec![1.0, 5.0, 2.0]; // Wild oscillation
        let extrapolated = 1.5;

        let r_squared = calculate_r_squared(&noise_factors, &values, extrapolated);

        assert!(
            r_squared < 0.5,
            "Poor fit should have low R², got {}",
            r_squared
        );
    }

    #[test]
    fn test_zne_validation_all_checks_pass() {
        // Good ZNE scenario: smooth trend, reasonable extrapolation
        let noise_factors = vec![1.0, 2.0, 3.0];
        let noisy_values = vec![-1.12, -1.10, -1.08]; // Smooth increasing (less negative)
        let extrapolated_value = -1.14; // Reasonable extrapolation

        let validation = ZNEValidation::validate(&noise_factors, &noisy_values, extrapolated_value);

        println!("Validation: {:?}", validation);

        assert!(
            validation.is_valid,
            "Should pass all checks: {:?}",
            validation.failure_reason
        );
        assert!(validation.is_monotonic, "Should be monotonic");
        assert!(
            validation.intercept_sanity,
            "Should have reasonable intercept"
        );
        assert!(validation.fit_quality > 0.7, "Should have good fit");
    }

    #[test]
    fn test_zne_validation_wild_oscillation() {
        // Bad ZNE scenario: wild oscillation
        let noise_factors = vec![1.0, 2.0, 3.0];
        let noisy_values = vec![-1.12, -0.80, -1.50]; // Oscillating
        let extrapolated_value = -1.14;

        let validation = ZNEValidation::validate(&noise_factors, &noisy_values, extrapolated_value);

        println!("Validation: {:?}", validation);

        assert!(!validation.is_valid, "Should fail monotonicity check");
        assert!(!validation.is_monotonic, "Should detect oscillation");
    }

    #[test]
    fn test_zne_validation_crazy_intercept() {
        // Bad ZNE scenario: extrapolated value way off from baseline
        let noise_factors = vec![1.0, 2.0, 3.0];
        let noisy_values = vec![-1.12, -1.10, -1.08];
        let extrapolated_value = -2.50; // 2× jump from baseline!

        let validation = ZNEValidation::validate(&noise_factors, &noisy_values, extrapolated_value);

        println!("Validation: {:?}", validation);

        assert!(!validation.is_valid, "Should fail intercept sanity check");
        assert!(!validation.intercept_sanity, "Should detect crazy jump");
    }

    // SPRINT 71.0: Adaptive noise factor selection tests

    #[test]
    fn test_select_noise_factors_stable_circuit() {
        // Very stable circuit (low variance)
        let cost_fn = |_params: &[f64], _noise: &NoiseConfig| {
            // Deterministic output (no variance)
            -1.0
        };

        let base_noise = NoiseConfig::low();
        let noise_factors = select_noise_factors(cost_fn, &[], &base_noise, 10, 3);

        println!("Stable circuit noise factors: {:?}", noise_factors);

        // Should select higher λ values since circuit is stable
        assert_eq!(noise_factors.len(), 3);
        assert_eq!(noise_factors[0], 1.0, "First factor should be 1.0");
        assert!(
            noise_factors[2] > 3.0,
            "Max lambda should be > 3.0 for stable circuit, got {}",
            noise_factors[2]
        );
    }

    #[test]
    fn test_select_noise_factors_noisy_circuit() {
        use std::cell::Cell;

        // Noisy circuit (high variance)
        let counter = Cell::new(0);
        let cost_fn = |_params: &[f64], _noise: &NoiseConfig| {
            // Oscillating output (high variance)
            let val = counter.get();
            counter.set(val + 1);
            if val % 2 == 0 { -1.0 } else { -1.3 }
        };

        let base_noise = NoiseConfig::medium();
        let noise_factors = select_noise_factors(cost_fn, &[], &base_noise, 10, 3);

        println!("Noisy circuit noise factors: {:?}", noise_factors);

        // Should select lower λ values since circuit is unstable
        assert_eq!(noise_factors.len(), 3);
        assert_eq!(noise_factors[0], 1.0, "First factor should be 1.0");
        assert!(
            noise_factors[2] < 3.5,
            "Max lambda should be < 3.5 for noisy circuit, got {}",
            noise_factors[2]
        );
        assert!(
            noise_factors[2] >= 1.5,
            "Max lambda should be >= 1.5 even for noisy circuit, got {}",
            noise_factors[2]
        );
    }

    #[test]
    fn test_select_noise_factors_different_levels() {
        let cost_fn = |_params: &[f64], _noise: &NoiseConfig| -1.0;
        let base_noise = NoiseConfig::low();

        // Test with 2 levels
        let factors_2 = select_noise_factors(cost_fn, &[], &base_noise, 10, 2);
        assert_eq!(factors_2.len(), 2);
        assert_eq!(factors_2[0], 1.0);

        // Test with 5 levels
        let factors_5 = select_noise_factors(cost_fn, &[], &base_noise, 10, 5);
        assert_eq!(factors_5.len(), 5);
        assert_eq!(factors_5[0], 1.0);

        // All factors should be in ascending order
        for i in 1..factors_5.len() {
            assert!(
                factors_5[i] > factors_5[i - 1],
                "Factors should be ascending"
            );
        }
    }

    #[test]
    fn test_select_noise_factors_spacing() {
        let cost_fn = |_params: &[f64], _noise: &NoiseConfig| -1.0;
        let base_noise = NoiseConfig::low();

        let noise_factors = select_noise_factors(cost_fn, &[], &base_noise, 10, 4);

        println!("Noise factors: {:?}", noise_factors);

        // Check that spacing is roughly uniform
        let spacings: Vec<f64> = (1..noise_factors.len())
            .map(|i| noise_factors[i] - noise_factors[i - 1])
            .collect();

        println!("Spacings: {:?}", spacings);

        // All spacings should be positive and similar
        for &spacing in &spacings {
            assert!(spacing > 0.0, "Spacing should be positive");
        }

        // Check that spacings are within 20% of each other (roughly uniform)
        let mean_spacing: f64 = spacings.iter().sum::<f64>() / spacings.len() as f64;
        for &spacing in &spacings {
            let rel_diff = (spacing - mean_spacing).abs() / mean_spacing;
            assert!(
                rel_diff < 0.2,
                "Spacing should be roughly uniform, rel_diff = {}",
                rel_diff
            );
        }
    }

    #[test]
    fn test_apply_zne_parallel() {
        // SPRINT 72.0: Test parallel ZNE execution
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));

        // Simple cost function that increments a counter
        let cost_fn = |_params: &[f64], noise: &NoiseConfig| {
            call_count.fetch_add(1, Ordering::SeqCst);
            // Return some function of noise level
            1.0 - noise.depolarizing_rate as f64
        };

        let params = vec![0.5, 0.5];
        let base_noise = NoiseConfig::medium();
        let config = ZNEConfig::richardson();

        let result = apply_zne_parallel(cost_fn, &params, &base_noise, &config);

        // Should have evaluated all 3 noise levels
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        assert_eq!(result.noisy_values.len(), 3);
        assert_eq!(result.noise_factors.len(), 3);

        // Validation should exist
        assert!(result.validation.fit_quality >= 0.0);

        println!(
            "Parallel ZNE result: extrapolated = {:.4}",
            result.extrapolated_value
        );
    }

    #[test]
    fn test_parallel_vs_sequential_equivalence() {
        // SPRINT 72.0: Verify parallel and sequential give same results

        // Deterministic cost function (no randomness)
        let cost_fn = |params: &[f64], noise: &NoiseConfig| {
            let param_sum: f64 = params.iter().sum();
            param_sum * (1.0 - noise.depolarizing_rate as f64)
        };

        let params = vec![0.3, 0.7, 0.5];
        let base_noise = NoiseConfig::low();
        let config = ZNEConfig {
            noise_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationMethod::Linear,
        };

        let result_parallel = apply_zne_parallel(cost_fn, &params, &base_noise, &config);
        let result_sequential = apply_zne_sequential(cost_fn, &params, &base_noise, &config);

        // Noisy values should be identical
        for (i, (&p, &s)) in result_parallel
            .noisy_values
            .iter()
            .zip(&result_sequential.noisy_values)
            .enumerate()
        {
            assert!(
                (p - s).abs() < 1e-10,
                "Noisy value {} differs: parallel={}, sequential={}",
                i,
                p,
                s
            );
        }

        // Extrapolated values should be identical
        assert!(
            (result_parallel.extrapolated_value - result_sequential.extrapolated_value).abs()
                < 1e-10,
            "Extrapolated values differ: parallel={}, sequential={}",
            result_parallel.extrapolated_value,
            result_sequential.extrapolated_value
        );

        // Validation should be identical
        assert_eq!(
            result_parallel.validation.is_valid,
            result_sequential.validation.is_valid
        );

        println!("Parallel: {:.6}", result_parallel.extrapolated_value);
        println!("Sequential: {:.6}", result_sequential.extrapolated_value);
    }

    #[test]
    fn test_parallel_zne_with_validation() {
        // SPRINT 72.0: Test that parallel ZNE respects Sprint 71 validation

        // Cost function with realistic noise dependence
        let cost_fn = |_params: &[f64], noise: &NoiseConfig| {
            // Energy increases with noise (realistic VQE behavior)
            -1.0 + 0.1 * noise.depolarizing_rate as f64
        };

        let params = vec![0.5];
        let base_noise = NoiseConfig::low();
        let config = ZNEConfig::richardson();

        let result = apply_zne_parallel(cost_fn, &params, &base_noise, &config);

        // Should have validation results
        assert!(result.validation.fit_quality >= 0.0);

        // If valid, extrapolated value should be used; otherwise fallback to baseline
        if result.validation.is_valid {
            println!(
                "ZNE validated: extrapolated = {:.6}",
                result.extrapolated_value
            );
        } else {
            println!(
                "ZNE failed validation: fallback to baseline = {:.6}",
                result.noisy_values[0]
            );
            if let Some(reason) = &result.validation.failure_reason {
                println!("  Reason: {}", reason);
            }
        }
    }

    #[test]
    fn test_adaptive_scheduling_early_termination() {
        // SPRINT 72.0 Phase 4: Test early termination when baseline is broken
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        // Cost function that returns NaN at baseline (λ=1.0)
        let cost_fn = move |_params: &[f64], noise: &NoiseConfig| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);

            // Return NaN at baseline, valid values otherwise
            // NoiseConfig::low() has depolarizing_rate = 0.001
            if (noise.depolarizing_rate - 0.001).abs() < 1e-6 {
                f64::NAN
            } else {
                -1.0 + 0.1 * noise.depolarizing_rate as f64
            }
        };

        let params = vec![0.5];
        let base_noise = NoiseConfig::low(); // depolarizing_rate = 0.001
        let config = ZNEConfig::richardson(); // [1.0, 2.0, 3.0]

        let result = apply_zne_parallel(cost_fn, &params, &base_noise, &config);

        // Should have only evaluated baseline (λ=1.0), not remaining levels
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "Should only evaluate baseline before early termination"
        );

        // Result should be invalid
        assert!(!result.validation.is_valid);
        assert!(result.validation.failure_reason.is_some());

        // Should have returned fallback (baseline value, even though it's NaN)
        assert_eq!(result.noisy_values.len(), 1);
        assert!(result.noisy_values[0].is_nan());

        println!(
            "Early termination: evaluated only {} noise levels",
            result.noisy_values.len()
        );
        println!("Failure reason: {:?}", result.validation.failure_reason);
    }

    #[test]
    fn test_adaptive_scheduling_normal_path() {
        // SPRINT 72.0 Phase 4: Test that normal circuits still evaluate all levels
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        // Normal cost function (no NaN)
        let cost_fn = move |_params: &[f64], noise: &NoiseConfig| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            -1.0 + 0.05 * noise.depolarizing_rate as f64
        };

        let params = vec![0.5];
        let base_noise = NoiseConfig::low();
        let config = ZNEConfig::richardson(); // [1.0, 2.0, 3.0]

        let result = apply_zne_parallel(cost_fn, &params, &base_noise, &config);

        // Should have evaluated all 3 noise levels (baseline + 2 parallel)
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            3,
            "Should evaluate all noise levels for valid circuit"
        );
        assert_eq!(result.noisy_values.len(), 3);

        println!(
            "Normal path: evaluated all {} noise levels",
            result.noisy_values.len()
        );
    }
}
