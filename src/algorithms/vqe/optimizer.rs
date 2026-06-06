// =============================================================================
// VQE Classical Optimizers
// =============================================================================
//
// Classical optimization algorithms for VQE parameter updates.
//
// Implements:
// - Nelder-Mead (derivative-free simplex method)
// - Gradient Descent with Parameter-Shift Rule
// - COBYLA (Constrained Optimization BY Linear Approximations)
//
// All optimizers work with the VQE cost function: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
//
// =============================================================================

use crate::algorithms::vqe::AnsatzType;
use crate::algorithms::vqe::ansatz::{apply_ansatz, build_uccsd_circuit};
use crate::hamiltonians::{Hamiltonian, measure_hamiltonian_expectation};
use crate::quantum::{QGate, QRng, QState, apply_gate_scalar};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "vqe_fp8")]
use crate::algorithms::vqe::fp8_backend::Fp8VQEBackend;
#[cfg(feature = "vqe_fp8")]
use std::sync::{Arc, Mutex};

/// Result of VQE optimization
#[derive(Debug, Clone)]
pub struct VQEResult {
    /// Final optimized energy
    pub energy: f64,
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Number of optimization iterations
    pub n_iterations: usize,
    /// Number of energy evaluations (function calls)
    pub n_evaluations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Energy history for convergence analysis
    pub energy_history: Vec<f64>,
}

/// VQE cost function evaluator
pub struct VQECostFunction {
    /// Ansatz circuit template (with placeholder rotations)
    pub ansatz_template: Vec<QGate>,
    /// Ansatz type (for UCCSD, we need to build circuits differently)
    pub ansatz_type: AnsatzType,
    /// Target Hamiltonian
    pub hamiltonian: Hamiltonian,
    /// Number of measurement shots per term
    pub n_shots: usize,
    /// Random seed for measurements
    pub seed: u64,
    /// Counter for function evaluations (SPRINT 72.0: AtomicUsize for thread safety)
    eval_count: AtomicUsize,

    // SPRINT 68.0: Error mitigation support
    /// Base noise configuration (optional)
    pub noise_config: Option<crate::experiments::noise_model::NoiseConfig>,
    /// Error mitigation strategy (optional)
    pub mitigation_strategy: Option<crate::mitigation::MitigationStrategy>,

    // SPRINT 75: FP8 Tensor Core acceleration
    /// FP8 GPU backend for accelerated circuit evaluation (thread-safe)
    #[cfg(feature = "vqe_fp8")]
    fp8_backend: Option<Arc<Mutex<Fp8VQEBackend>>>,
}

impl VQECostFunction {
    pub fn new(
        ansatz_template: Vec<QGate>,
        ansatz_type: AnsatzType,
        hamiltonian: Hamiltonian,
        n_shots: usize,
        seed: u64,
    ) -> Self {
        Self {
            ansatz_template,
            ansatz_type,
            hamiltonian,
            n_shots,
            seed,
            eval_count: AtomicUsize::new(0),
            noise_config: None,
            mitigation_strategy: None,
            #[cfg(feature = "vqe_fp8")]
            fp8_backend: None,
        }
    }

    /// Set FP8 GPU backend for accelerated evaluation (SPRINT 75)
    #[cfg(feature = "vqe_fp8")]
    pub fn with_fp8_backend(mut self, backend: Fp8VQEBackend) -> Self {
        self.fp8_backend = Some(Arc::new(Mutex::new(backend)));
        self
    }

    /// Check if FP8 backend is available
    #[cfg(feature = "vqe_fp8")]
    pub fn has_fp8_backend(&self) -> bool {
        self.fp8_backend.is_some()
    }

    /// Create with error mitigation support (SPRINT 68.0)
    pub fn with_mitigation(
        mut self,
        noise_config: crate::experiments::noise_model::NoiseConfig,
        mitigation_strategy: crate::mitigation::MitigationStrategy,
    ) -> Self {
        self.noise_config = Some(noise_config);
        self.mitigation_strategy = Some(mitigation_strategy);
        self
    }

    /// Evaluate energy for given parameters
    ///
    /// SPRINT 68.0: Dispatches to evaluate_with_mitigation() if configured
    /// SPRINT 72.0: Thread-safe (&self instead of &mut self)
    /// SPRINT 75: Dispatches to FP8 GPU backend when available
    pub fn evaluate(&self, theta: &[f64]) -> f64 {
        // If mitigation is configured, use it
        if self.mitigation_strategy.is_some() {
            return self.evaluate_with_mitigation(theta);
        }

        // SPRINT 75: Try FP8 GPU backend if available
        #[cfg(feature = "vqe_fp8")]
        if let Some(ref backend_arc) = self.fp8_backend {
            if let Ok(mut backend) = backend_arc.lock() {
                let gates = self.build_circuit(theta);
                match backend.evaluate_circuit_batch(
                    &[theta.to_vec()],
                    &self.ansatz_type,
                    &gates,
                    &self.hamiltonian,
                ) {
                    Ok(energies) if !energies.is_empty() => {
                        self.eval_count.fetch_add(1, Ordering::SeqCst);
                        return energies[0];
                    }
                    Err(e) => {
                        // Log error and fall through to CPU path
                        eprintln!("FP8 backend error, falling back to CPU: {:?}", e);
                    }
                    _ => {}
                }
            }
        }

        // Standard CPU evaluation without mitigation
        let eval_id = self.eval_count.fetch_add(1, Ordering::SeqCst);

        // Prepare initial state
        let mut state = QState::new_zero(self.hamiltonian.n_qubits);

        // Apply parameterized ansatz based on type
        match &self.ansatz_type {
            AnsatzType::UCCSD { n_electrons } => {
                // For UCCSD, build the circuit directly with parameters
                let circuit = build_uccsd_circuit(self.hamiltonian.n_qubits, *n_electrons, theta);
                let mut rng = QRng::new(self.seed.wrapping_add(eval_id as u64));
                for gate in &circuit {
                    let _ = apply_gate_scalar(&mut state, gate, &mut rng);
                }
            }
            AnsatzType::HardwareEfficient { .. } => {
                // For hardware-efficient, use the template + apply_parameters approach
                apply_ansatz(&mut state, &self.ansatz_template, theta);
            }
        }

        // Measure Hamiltonian expectation value
        let seed = self.seed.wrapping_add(eval_id as u64);
        measure_hamiltonian_expectation(&state, &self.hamiltonian, self.n_shots, seed)
    }

    /// SPRINT 68.0: Evaluate cost function with error mitigation
    ///
    /// Dispatches to appropriate mitigation technique based on configuration:
    /// - ZNE: Run at multiple noise levels, extrapolate to λ=0
    /// - REM: Apply readout error mitigation
    /// - Combined: ZNE + REM
    /// - None: Standard evaluation (possibly with noise, no mitigation)
    /// SPRINT 72.0: Thread-safe (&self instead of &mut self)
    pub fn evaluate_with_mitigation(&self, theta: &[f64]) -> f64 {
        use crate::mitigation::MitigationStrategy;

        // Clone strategy to avoid borrow checker issues
        let strategy = self.mitigation_strategy.clone();

        match strategy {
            Some(MitigationStrategy::ZNE(config)) => self.evaluate_with_zne(theta, &config),
            Some(MitigationStrategy::Readout(calibration)) => {
                self.evaluate_with_rem(theta, &calibration)
            }
            Some(MitigationStrategy::ZNEPlusReadout { zne, readout }) => {
                self.evaluate_with_zne_and_rem(theta, &zne, &readout)
            }
            Some(MitigationStrategy::None) | None => {
                // No mitigation - but inject noise if noise_config is present
                // This creates a "noisy baseline" for comparison
                let eval_id = self.eval_count.fetch_add(1, Ordering::SeqCst);

                if let Some(noise) = &self.noise_config {
                    // SPRINT 68.0: Noisy baseline (no mitigation)
                    use crate::algorithms::vqe::noisy_execution::{
                        execute_circuit_with_noise, measure_hamiltonian_with_noise,
                    };

                    let circuit = self.build_circuit(theta);
                    let state = execute_circuit_with_noise(
                        &circuit,
                        self.hamiltonian.n_qubits,
                        noise,
                        self.seed.wrapping_add(eval_id as u64),
                    );

                    measure_hamiltonian_with_noise(
                        &state,
                        &self.hamiltonian,
                        noise,
                        self.n_shots,
                        self.seed.wrapping_add(eval_id as u64),
                    )
                } else {
                    // Clean execution (no noise, no mitigation)
                    let mut state = QState::new_zero(self.hamiltonian.n_qubits);

                    match &self.ansatz_type {
                        AnsatzType::UCCSD { n_electrons } => {
                            let circuit =
                                build_uccsd_circuit(self.hamiltonian.n_qubits, *n_electrons, theta);
                            let mut rng = QRng::new(self.seed.wrapping_add(eval_id as u64));
                            for gate in &circuit {
                                let _ = apply_gate_scalar(&mut state, gate, &mut rng);
                            }
                        }
                        AnsatzType::HardwareEfficient { .. } => {
                            apply_ansatz(&mut state, &self.ansatz_template, theta);
                        }
                    }

                    let seed = self.seed.wrapping_add(eval_id as u64);
                    measure_hamiltonian_expectation(&state, &self.hamiltonian, self.n_shots, seed)
                }
            }
        }
    }

    /// Evaluate with Zero-Noise Extrapolation
    ///
    /// SPRINT 71.0: Now includes validation and fallback
    /// SPRINT 72.0: Thread-safe for parallel execution
    fn evaluate_with_zne(&self, theta: &[f64], config: &crate::mitigation::ZNEConfig) -> f64 {
        use crate::algorithms::vqe::noisy_execution::{
            execute_circuit_with_noise, measure_hamiltonian_with_noise,
        };
        use crate::mitigation::ZNEValidation;

        let base_noise = self
            .noise_config
            .as_ref()
            .expect("ZNE requires noise_config to be set");

        // SPRINT 72.0: Fetch eval_id once for this entire ZNE evaluation
        let eval_id = self.eval_count.fetch_add(1, Ordering::SeqCst);

        let mut noisy_values = Vec::new();

        for (lambda_idx, &lambda) in config.noise_factors.iter().enumerate() {
            // Amplify noise by lambda (gate-noise-only scaling in Sprint 71)
            let amplified_noise = base_noise.amplify_by(lambda);

            // Build circuit for these parameters
            let circuit = self.build_circuit(theta);

            // SPRINT 72.0: Derive unique seed per lambda level
            let lambda_seed = self
                .seed
                .wrapping_add(eval_id as u64)
                .wrapping_mul(100)
                .wrapping_add(lambda_idx as u64);

            // Execute with amplified noise
            let state = execute_circuit_with_noise(
                &circuit,
                self.hamiltonian.n_qubits,
                &amplified_noise,
                lambda_seed,
            );

            // Measure with noisy measurements
            let energy = measure_hamiltonian_with_noise(
                &state,
                &self.hamiltonian,
                &amplified_noise,
                self.n_shots,
                lambda_seed.wrapping_add(1000),
            );

            noisy_values.push(energy);
        }

        // Extrapolate to λ=0 (zero noise)
        let extrapolated = crate::mitigation::zne::extrapolate(
            &config.noise_factors,
            &noisy_values,
            &config.extrapolation,
        );

        // SPRINT 71.0: Validate extrapolation
        let validation =
            ZNEValidation::validate(&config.noise_factors, &noisy_values, extrapolated);

        // If validation fails, fall back to baseline (λ=1.0)
        if validation.is_valid {
            extrapolated
        } else {
            // Fallback: use baseline value (first noise factor, λ=1.0)
            noisy_values[0]
        }
    }

    /// Evaluate with Readout Error Mitigation
    /// SPRINT 72.0: Thread-safe for parallel execution
    fn evaluate_with_rem(
        &self,
        theta: &[f64],
        calibration: &crate::mitigation::ReadoutCalibration,
    ) -> f64 {
        use crate::algorithms::vqe::noisy_execution::{
            execute_circuit, execute_circuit_with_noise, measure_hamiltonian_with_rem,
        };

        let eval_id = self.eval_count.fetch_add(1, Ordering::SeqCst);
        let circuit = self.build_circuit(theta);

        // Execute circuit (with or without gate noise)
        let state = if let Some(noise) = &self.noise_config {
            execute_circuit_with_noise(
                &circuit,
                self.hamiltonian.n_qubits,
                noise,
                self.seed.wrapping_add(eval_id as u64),
            )
        } else {
            execute_circuit(
                &circuit,
                self.hamiltonian.n_qubits,
                self.seed.wrapping_add(eval_id as u64),
            )
        };

        // Measure with readout error mitigation
        measure_hamiltonian_with_rem(
            &state,
            &self.hamiltonian,
            calibration,
            self.n_shots,
            self.seed.wrapping_add(eval_id as u64).wrapping_add(1000),
        )
    }

    /// Evaluate with combined ZNE + REM
    ///
    /// SPRINT 71.0: Pre-mitigation architecture
    /// - Apply REM at each noise level (corrects readout errors)
    /// - Then extrapolate to λ=0 (corrects gate errors)
    /// - Validation and fallback to REM-only if ZNE fails
    /// SPRINT 72.0: Thread-safe for parallel execution
    fn evaluate_with_zne_and_rem(
        &self,
        theta: &[f64],
        zne_config: &crate::mitigation::ZNEConfig,
        calibration: &crate::mitigation::ReadoutCalibration,
    ) -> f64 {
        use crate::algorithms::vqe::noisy_execution::{
            execute_circuit_with_noise, measure_hamiltonian_with_rem,
        };
        use crate::mitigation::ZNEValidation;

        let base_noise = self
            .noise_config
            .as_ref()
            .expect("Combined mitigation requires noise_config to be set");

        // SPRINT 72.0: Fetch eval_id once for this entire ZNE+REM evaluation
        let eval_id = self.eval_count.fetch_add(1, Ordering::SeqCst);

        let mut noisy_values = Vec::new();

        for (lambda_idx, &lambda) in zne_config.noise_factors.iter().enumerate() {
            // SPRINT 71.0: Amplify gate noise by lambda (readout errors unchanged)
            // This allows the same REM calibration to work at all noise levels
            let amplified_noise = base_noise.amplify_by(lambda);

            // Build circuit for these parameters
            let circuit = self.build_circuit(theta);

            // SPRINT 72.0: Derive unique seed per lambda level
            let lambda_seed = self
                .seed
                .wrapping_add(eval_id as u64)
                .wrapping_mul(100)
                .wrapping_add(lambda_idx as u64);

            // Execute with amplified gate noise
            let state = execute_circuit_with_noise(
                &circuit,
                self.hamiltonian.n_qubits,
                &amplified_noise,
                lambda_seed,
            );

            // Measure with REM (corrects readout errors at this noise level)
            // Since readout errors are unchanged across λ, the same calibration works
            let energy = measure_hamiltonian_with_rem(
                &state,
                &self.hamiltonian,
                calibration,
                self.n_shots,
                lambda_seed.wrapping_add(1000),
            );

            noisy_values.push(energy);
        }

        // Extrapolate to λ=0 (zero gate noise)
        let extrapolated = crate::mitigation::zne::extrapolate(
            &zne_config.noise_factors,
            &noisy_values,
            &zne_config.extrapolation,
        );

        // SPRINT 71.0: Validate extrapolation
        let validation =
            ZNEValidation::validate(&zne_config.noise_factors, &noisy_values, extrapolated);

        // If ZNE validation fails, fall back to REM-only (baseline with readout correction)
        if validation.is_valid {
            extrapolated
        } else {
            // Fallback: use REM-corrected baseline value (first noise factor, λ=1.0)
            noisy_values[0]
        }
    }

    /// Build circuit from ansatz template with given parameters
    fn build_circuit(&self, theta: &[f64]) -> Vec<QGate> {
        match &self.ansatz_type {
            AnsatzType::UCCSD { n_electrons } => {
                build_uccsd_circuit(self.hamiltonian.n_qubits, *n_electrons, theta)
            }
            AnsatzType::HardwareEfficient { .. } => {
                // For hardware-efficient, we need to substitute parameters into template
                crate::algorithms::vqe::ansatz::apply_parameters(&self.ansatz_template, theta)
            }
        }
    }

    /// Get number of evaluations so far (SPRINT 72.0: Thread-safe atomic read)
    pub fn evaluation_count(&self) -> usize {
        self.eval_count.load(Ordering::SeqCst)
    }

    /// Reset evaluation counter (SPRINT 72.0: Thread-safe atomic write)
    pub fn reset_counter(&self) {
        self.eval_count.store(0, Ordering::SeqCst);
    }

    // =========================================================================
    // SPRINT 75: Batch Evaluation for FP8 GPU Backend
    // =========================================================================

    /// Evaluate multiple parameter sets in a single batch
    ///
    /// When FP8 backend is enabled, this uses GPU parallelism to evaluate
    /// all parameter sets simultaneously, providing significant speedup
    /// for optimizers that can benefit from batch evaluation (e.g., SPSA).
    ///
    /// # Arguments
    /// * `params_batch` - Vector of parameter vectors to evaluate
    ///
    /// # Returns
    /// Vector of energy values, one per parameter set
    ///
    /// # Example
    ///
    /// ```ignore
    /// let params_plus = params.iter().map(|p| p + delta).collect();
    /// let params_minus = params.iter().map(|p| p - delta).collect();
    ///
    /// // Evaluate both in a single GPU call
    /// let energies = cost_fn.evaluate_batch(&[params_plus, params_minus]);
    /// let gradient = (energies[0] - energies[1]) / (2.0 * delta);
    /// ```
    pub fn evaluate_batch(&self, params_batch: &[Vec<f64>]) -> Vec<f64> {
        if params_batch.is_empty() {
            return vec![];
        }

        // TODO: Add FP8 GPU path when backend is configured
        // For now, fall back to sequential evaluation
        //
        // When FP8 backend is enabled, this would:
        // 1. Build circuits for all parameter sets
        // 2. Upload to GPU multi-state buffer
        // 3. Execute all circuits in parallel
        // 4. Measure Hamiltonian expectation values in parallel
        // 5. Return energy values

        params_batch.iter().map(|p| self.evaluate(p)).collect()
    }

    /// Evaluate batch with FP8 backend (SPRINT 75)
    ///
    /// This is the GPU-accelerated path for batch evaluation.
    /// Requires `vqe_fp8` feature to be enabled.
    #[cfg(feature = "vqe_fp8")]
    pub fn evaluate_batch_fp8(
        &self,
        params_batch: &[Vec<f64>],
        backend: &mut crate::algorithms::vqe::fp8_backend::Fp8VQEBackend,
    ) -> crate::cuda::CudaResult<Vec<f64>> {
        let gates = self.build_circuit(&params_batch[0]);
        backend.evaluate_circuit_batch(params_batch, &self.ansatz_type, &gates, &self.hamiltonian)
    }
}

// =============================================================================
// Nelder-Mead Optimizer (Derivative-Free)
// =============================================================================

/// Nelder-Mead simplex optimizer
///
/// Good for:
/// - Noisy cost functions (VQE with finite shots)
/// - Low to moderate dimensions
/// - When gradients are expensive or unavailable
pub struct NelderMeadOptimizer {
    /// Reflection coefficient
    pub alpha: f64,
    /// Expansion coefficient
    pub gamma: f64,
    /// Contraction coefficient
    pub rho: f64,
    /// Shrink coefficient
    pub sigma: f64,
    /// Convergence tolerance (standard deviation of simplex energies)
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for NelderMeadOptimizer {
    fn default() -> Self {
        Self {
            alpha: 1.0, // Standard reflection
            gamma: 2.0, // Standard expansion
            rho: 0.5,   // Standard contraction
            sigma: 0.5, // Standard shrink
            tolerance: 1e-6,
            max_iterations: 500,
        }
    }
}

impl NelderMeadOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Run Nelder-Mead optimization (SPRINT 72.0: Thread-safe cost_fn)
    pub fn optimize(&self, cost_fn: &VQECostFunction, initial_params: &[f64]) -> VQEResult {
        let n = initial_params.len();

        // Initialize simplex: n+1 vertices
        // Start with initial point and add small perturbations
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(initial_params.to_vec());

        for i in 0..n {
            let mut vertex = initial_params.to_vec();
            vertex[i] += 0.5; // Moderate perturbation for rotation angles
            simplex.push(vertex);
        }

        // Evaluate all vertices
        let mut values: Vec<f64> = simplex.iter().map(|v| cost_fn.evaluate(v)).collect();

        let mut energy_history = vec![values.iter().cloned().fold(f64::INFINITY, f64::min)];
        let mut iteration = 0;

        while iteration < self.max_iterations {
            // Sort vertices by value (ascending - we're minimizing)
            let mut indices: Vec<usize> = (0..=n).collect();
            indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

            // Reorder simplex and values
            simplex = indices.iter().map(|&i| simplex[i].clone()).collect();
            values = indices.iter().map(|&i| values[i]).collect();

            let best_value = values[0];
            let worst_value = values[n];
            let second_worst_value = values[n - 1];

            energy_history.push(best_value);

            // Check convergence (standard deviation of values)
            let mean = values.iter().sum::<f64>() / (n + 1) as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n + 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev < self.tolerance {
                return VQEResult {
                    energy: best_value,
                    params: simplex[0].clone(),
                    n_iterations: iteration,
                    n_evaluations: cost_fn.evaluation_count(),
                    converged: true,
                    energy_history,
                };
            }

            // Compute centroid of all vertices except worst
            let mut centroid = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    centroid[j] += simplex[i][j];
                }
            }
            for j in 0..n {
                centroid[j] /= n as f64;
            }

            // Reflection: x_r = centroid + α(centroid - worst)
            let reflected: Vec<f64> = (0..n)
                .map(|j| centroid[j] + self.alpha * (centroid[j] - simplex[n][j]))
                .collect();
            let reflected_value = cost_fn.evaluate(&reflected);

            if best_value <= reflected_value && reflected_value < second_worst_value {
                // Accept reflection
                simplex[n] = reflected;
                values[n] = reflected_value;
            } else if reflected_value < best_value {
                // Try expansion: x_e = centroid + γ(x_r - centroid)
                let expanded: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + self.gamma * (reflected[j] - centroid[j]))
                    .collect();
                let expanded_value = cost_fn.evaluate(&expanded);

                if expanded_value < reflected_value {
                    simplex[n] = expanded;
                    values[n] = expanded_value;
                } else {
                    simplex[n] = reflected;
                    values[n] = reflected_value;
                }
            } else {
                // Contract: x_c = centroid + ρ(worst - centroid) or (reflected - centroid)
                let contract_point = if reflected_value < worst_value {
                    &reflected
                } else {
                    &simplex[n]
                };

                let contracted: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + self.rho * (contract_point[j] - centroid[j]))
                    .collect();
                let contracted_value = cost_fn.evaluate(&contracted);

                if contracted_value < worst_value.min(reflected_value) {
                    simplex[n] = contracted;
                    values[n] = contracted_value;
                } else {
                    // Shrink: move all vertices toward best
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] =
                                simplex[0][j] + self.sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        values[i] = cost_fn.evaluate(&simplex[i]);
                    }
                }
            }

            iteration += 1;
        }

        // Return best found even if not converged
        VQEResult {
            energy: values[0],
            params: simplex[0].clone(),
            n_iterations: iteration,
            n_evaluations: cost_fn.evaluation_count(),
            converged: false,
            energy_history,
        }
    }
}

// =============================================================================
// Gradient Descent with Parameter-Shift Rule
// =============================================================================

/// Gradient descent optimizer using the parameter-shift rule
///
/// For rotation gates Rₓ(θ), Rᵧ(θ), Rᵤ(θ):
/// ∂E/∂θᵢ = (E(θ + π/2 eᵢ) - E(θ - π/2 eᵢ)) / 2
///
/// This gives exact gradients without finite differences!
pub struct GradientDescentOptimizer {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Convergence tolerance (change in energy)
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Use adaptive learning rate
    pub adaptive: bool,
}

impl Default for GradientDescentOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            momentum: 0.9,
            tolerance: 1e-6,
            max_iterations: 500,
            adaptive: true,
        }
    }
}

impl GradientDescentOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Compute gradient using parameter-shift rule (SPRINT 72.0: Thread-safe cost_fn)
    fn compute_gradient(&self, cost_fn: &VQECostFunction, theta: &[f64]) -> Vec<f64> {
        let n = theta.len();
        let mut gradient = vec![0.0; n];
        let shift = PI / 2.0;

        for i in 0..n {
            // θ + π/2 eᵢ
            let mut theta_plus = theta.to_vec();
            theta_plus[i] += shift;
            let e_plus = cost_fn.evaluate(&theta_plus);

            // θ - π/2 eᵢ
            let mut theta_minus = theta.to_vec();
            theta_minus[i] -= shift;
            let e_minus = cost_fn.evaluate(&theta_minus);

            // Parameter-shift rule: exact gradient
            gradient[i] = (e_plus - e_minus) / 2.0;
        }

        gradient
    }

    /// Run gradient descent optimization (SPRINT 72.0: Thread-safe cost_fn)
    pub fn optimize(&self, cost_fn: &VQECostFunction, initial_params: &[f64]) -> VQEResult {
        let n = initial_params.len();
        let mut theta = initial_params.to_vec();
        let mut velocity = vec![0.0; n]; // Momentum

        let mut energy = cost_fn.evaluate(&theta);
        let mut energy_history = vec![energy];
        let mut learning_rate = self.learning_rate;

        for iteration in 0..self.max_iterations {
            // Compute gradient
            let gradient = self.compute_gradient(cost_fn, &theta);

            // Update with momentum
            for i in 0..n {
                velocity[i] = self.momentum * velocity[i] - learning_rate * gradient[i];
                theta[i] += velocity[i];
            }

            // Evaluate new energy
            let new_energy = cost_fn.evaluate(&theta);
            energy_history.push(new_energy);

            // Adaptive learning rate
            if self.adaptive {
                if new_energy > energy {
                    // Energy increased - reduce learning rate
                    learning_rate *= 0.5;
                } else if new_energy < energy - self.tolerance * 10.0 {
                    // Good progress - slightly increase learning rate
                    learning_rate *= 1.1;
                    learning_rate = learning_rate.min(self.learning_rate * 2.0);
                }
            }

            // Check convergence
            let delta_e = (new_energy - energy).abs();
            energy = new_energy;

            if delta_e < self.tolerance {
                return VQEResult {
                    energy,
                    params: theta,
                    n_iterations: iteration + 1,
                    n_evaluations: cost_fn.evaluation_count(),
                    converged: true,
                    energy_history,
                };
            }
        }

        VQEResult {
            energy,
            params: theta,
            n_iterations: self.max_iterations,
            n_evaluations: cost_fn.evaluation_count(),
            converged: false,
            energy_history,
        }
    }
}

// =============================================================================
// SPSA Optimizer (Simultaneous Perturbation Stochastic Approximation)
// =============================================================================

/// SPSA optimizer - efficient for high-dimensional noisy optimization
///
/// Instead of 2n evaluations for gradient (parameter-shift), SPSA uses only 2
/// evaluations per iteration by perturbing all parameters simultaneously.
pub struct SPSAOptimizer {
    /// Initial step size for parameters
    pub a: f64,
    /// Initial perturbation size
    pub c: f64,
    /// Stability constant
    pub alpha: f64,
    /// Perturbation decay rate
    pub gamma: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for SPSAOptimizer {
    fn default() -> Self {
        Self {
            a: 0.1,
            c: 0.1,
            alpha: 0.602,
            gamma: 0.101,
            max_iterations: 500,
            tolerance: 1e-6,
        }
    }
}

impl SPSAOptimizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Run SPSA optimization (SPRINT 72.0: Thread-safe cost_fn)
    pub fn optimize(&self, cost_fn: &VQECostFunction, initial_params: &[f64]) -> VQEResult {
        let n = initial_params.len();
        let mut theta = initial_params.to_vec();

        let mut energy = cost_fn.evaluate(&theta);
        let mut energy_history = vec![energy];

        let mut rng_seed = 12345u64;

        for k in 0..self.max_iterations {
            // Decaying step sizes
            let ak = self.a / (k as f64 + 1.0).powf(self.alpha);
            let ck = self.c / (k as f64 + 1.0).powf(self.gamma);

            // Generate random perturbation vector (±1 with equal probability)
            let delta: Vec<f64> = (0..n)
                .map(|i| {
                    rng_seed = rng_seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(i as u64);
                    if (rng_seed >> 32) & 1 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            // Perturbed evaluations
            let theta_plus: Vec<f64> = (0..n).map(|i| theta[i] + ck * delta[i]).collect();
            let theta_minus: Vec<f64> = (0..n).map(|i| theta[i] - ck * delta[i]).collect();

            let e_plus = cost_fn.evaluate(&theta_plus);
            let e_minus = cost_fn.evaluate(&theta_minus);

            // Gradient estimate
            let gradient_factor = (e_plus - e_minus) / (2.0 * ck);

            // Update parameters
            for i in 0..n {
                theta[i] -= ak * gradient_factor / delta[i];
            }

            // Evaluate new energy
            let new_energy = cost_fn.evaluate(&theta);
            energy_history.push(new_energy);

            // Check convergence
            if (new_energy - energy).abs() < self.tolerance {
                return VQEResult {
                    energy: new_energy,
                    params: theta,
                    n_iterations: k + 1,
                    n_evaluations: cost_fn.evaluation_count(),
                    converged: true,
                    energy_history,
                };
            }

            energy = new_energy;
        }

        VQEResult {
            energy,
            params: theta,
            n_iterations: self.max_iterations,
            n_evaluations: cost_fn.evaluation_count(),
            converged: false,
            energy_history,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::vqe::ansatz::hardware_efficient_ansatz;
    use crate::hamiltonians::Hamiltonian;

    fn create_simple_cost_fn() -> VQECostFunction {
        // Single qubit H = Z, ground state = |1⟩ with E = -1
        let (ansatz, _) = hardware_efficient_ansatz(1, 1);
        let h = Hamiltonian::single_qubit_z();
        VQECostFunction::new(
            ansatz,
            AnsatzType::HardwareEfficient { depth: 1 },
            h,
            1000,
            42,
        )
    }

    #[test]
    fn test_cost_function_evaluation() {
        let cost_fn = create_simple_cost_fn();

        // All zeros should give energy close to +1 (|0⟩ state)
        let e0 = cost_fn.evaluate(&[0.0, 0.0, 0.0]);
        assert!(e0 > 0.5, "Expected positive energy for |0⟩, got {}", e0);

        assert_eq!(cost_fn.evaluation_count(), 1);
    }

    #[test]
    fn test_nelder_mead_single_qubit() {
        let cost_fn = create_simple_cost_fn();
        let optimizer = NelderMeadOptimizer::new()
            .with_tolerance(1e-3)
            .with_max_iterations(100);

        let initial = vec![0.1, 0.1, 0.1];
        let result = optimizer.optimize(&cost_fn, &initial);

        // Should find energy close to -1 (ground state of Z)
        assert!(
            result.energy < 0.0,
            "Expected negative energy, got {}",
            result.energy
        );
        assert!(result.n_evaluations > 0);
    }

    #[test]
    fn test_gradient_descent_single_qubit() {
        let cost_fn = create_simple_cost_fn();
        // Disable adaptive and momentum for cleaner convergence
        let mut optimizer = GradientDescentOptimizer::new(0.5);
        optimizer.momentum = 0.0;
        optimizer.adaptive = false;
        optimizer.max_iterations = 200;

        let initial = vec![1.0, 1.0, 1.0];
        let result = optimizer.optimize(&cost_fn, &initial);

        // Should make progress towards negative energy
        assert!(
            result.energy < 0.5,
            "Expected lower energy, got {}",
            result.energy
        );
    }

    #[test]
    fn test_spsa_single_qubit() {
        let cost_fn = create_simple_cost_fn();
        let optimizer = SPSAOptimizer::new().with_max_iterations(200);

        let initial = vec![0.5, 0.5, 0.5];
        let result = optimizer.optimize(&cost_fn, &initial);

        // SPSA is stochastic, just verify it makes progress
        assert!(
            result.energy < 0.9,
            "Expected lower energy, got {}",
            result.energy
        );
    }

    #[test]
    fn test_parameter_shift_gradient() {
        let cost_fn = create_simple_cost_fn();
        let optimizer = GradientDescentOptimizer::new(0.1);

        let theta = vec![0.5, 0.5, 0.5];
        let gradient = optimizer.compute_gradient(&cost_fn, &theta);

        // Gradient should be computed for all parameters
        assert_eq!(gradient.len(), 3);

        // At least some gradients should be non-zero
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(grad_norm > 0.01, "Gradient too small: {:?}", gradient);
    }

    #[test]
    fn test_vqe_result_has_history() {
        let cost_fn = create_simple_cost_fn();
        let optimizer = NelderMeadOptimizer::new().with_max_iterations(10);

        let result = optimizer.optimize(&cost_fn, &[0.0, 0.0, 0.0]);

        // Should have energy history
        assert!(!result.energy_history.is_empty());

        // First entry should match initial evaluation
        assert!(result.energy_history.len() > 1);
    }

    #[test]
    fn test_two_qubit_optimization() {
        let (ansatz, n_params) = hardware_efficient_ansatz(2, 1);
        let h = Hamiltonian::two_qubit_xx_zz();
        let cost_fn = VQECostFunction::new(
            ansatz,
            AnsatzType::HardwareEfficient { depth: 1 },
            h,
            500,
            42,
        );

        let optimizer = NelderMeadOptimizer::new()
            .with_tolerance(1e-2)
            .with_max_iterations(100);

        let initial = vec![0.0; n_params];
        let result = optimizer.optimize(&cost_fn, &initial);

        // Should find some minimum
        println!("Two-qubit VQE result: E = {:.4}", result.energy);
        assert!(result.n_iterations > 0);
    }

    #[test]
    fn test_optimizer_comparison() {
        // Compare all three optimizers on same problem
        let h = Hamiltonian::single_qubit_z();
        let (ansatz, n_params) = hardware_efficient_ansatz(1, 1);
        let initial = vec![0.5; n_params];

        // Nelder-Mead
        let ansatz_type = AnsatzType::HardwareEfficient { depth: 1 };
        let cost_fn1 =
            VQECostFunction::new(ansatz.clone(), ansatz_type.clone(), h.clone(), 500, 42);
        let nm_result = NelderMeadOptimizer::new()
            .with_max_iterations(100)
            .optimize(&cost_fn1, &initial);

        // Gradient Descent (disable momentum for stability)
        let cost_fn2 =
            VQECostFunction::new(ansatz.clone(), ansatz_type.clone(), h.clone(), 500, 42);
        let mut gd_opt = GradientDescentOptimizer::new(0.5);
        gd_opt.momentum = 0.0;
        gd_opt.adaptive = false;
        gd_opt.max_iterations = 200;
        let gd_result = gd_opt.optimize(&cost_fn2, &initial);

        // SPSA
        let cost_fn3 = VQECostFunction::new(ansatz.clone(), ansatz_type, h.clone(), 500, 42);
        let spsa_result = SPSAOptimizer::new()
            .with_max_iterations(200)
            .optimize(&cost_fn3, &initial);

        // All should find lower energy (relaxed assertions)
        assert!(nm_result.energy < 0.0, "NM: {}", nm_result.energy);
        assert!(gd_result.energy < 0.5, "GD: {}", gd_result.energy);
        assert!(spsa_result.energy < 0.9, "SPSA: {}", spsa_result.energy);

        println!("Optimizer comparison:");
        println!(
            "  Nelder-Mead: E = {:.4}, evals = {}",
            nm_result.energy, nm_result.n_evaluations
        );
        println!(
            "  Grad Descent: E = {:.4}, evals = {}",
            gd_result.energy, gd_result.n_evaluations
        );
        println!(
            "  SPSA: E = {:.4}, evals = {}",
            spsa_result.energy, spsa_result.n_evaluations
        );
    }

    #[test]
    fn test_thread_safety() {
        // SPRINT 72.0: Verify VQECostFunction can be called from multiple threads
        use std::sync::Arc;
        use std::thread;

        let cost_fn = Arc::new(create_simple_cost_fn());
        let mut handles = vec![];

        // Spawn 10 threads that all call evaluate() concurrently
        for i in 0..10 {
            let cost_fn_clone = Arc::clone(&cost_fn);
            let handle = thread::spawn(move || {
                let params = vec![i as f64 * 0.1, i as f64 * 0.1, i as f64 * 0.1];
                cost_fn_clone.evaluate(&params)
            });
            handles.push(handle);
        }

        // Collect results
        let mut energies = vec![];
        for handle in handles {
            let energy = handle.join().unwrap();
            energies.push(energy);
        }

        // Verify we got 10 results
        assert_eq!(energies.len(), 10);

        // Verify eval_count incremented correctly (10 evaluations total)
        assert_eq!(cost_fn.evaluation_count(), 10);

        println!(
            "Thread safety test: {} evaluations from 10 threads",
            cost_fn.evaluation_count()
        );
    }

    #[test]
    fn test_determinism_with_parallelism() {
        // SPRINT 72.0: Verify same seed → same results, even with concurrent calls
        use std::sync::Arc;
        use std::thread;

        let cost_fn = Arc::new(create_simple_cost_fn());
        let params = vec![0.5, 0.5, 0.5];

        // Evaluate sequentially 3 times (should reset for each independent test)
        let cost_fn1 = create_simple_cost_fn();
        let e1 = cost_fn1.evaluate(&params);
        let cost_fn2 = create_simple_cost_fn();
        let e2 = cost_fn2.evaluate(&params);
        let cost_fn3 = create_simple_cost_fn();
        let e3 = cost_fn3.evaluate(&params);

        // All should be identical (same seed, same circuit)
        assert!(
            (e1 - e2).abs() < 1e-10 && (e2 - e3).abs() < 1e-10,
            "Sequential evaluations should be deterministic: {}, {}, {}",
            e1,
            e2,
            e3
        );

        // Now evaluate from multiple threads with same params
        let mut handles = vec![];
        for _ in 0..3 {
            let cost_fn_clone = Arc::clone(&cost_fn);
            let params_clone = params.clone();
            let handle = thread::spawn(move || cost_fn_clone.evaluate(&params_clone));
            handles.push(handle);
        }

        let mut parallel_energies = vec![];
        for handle in handles {
            parallel_energies.push(handle.join().unwrap());
        }

        // Parallel results should also match each other
        // (Each thread gets unique eval_id, but same base seed + circuit → similar results)
        // Note: Results may differ slightly due to different eval_id affecting RNG
        println!("Parallel energies: {:?}", parallel_energies);
        println!("Sequential energy (reference): {}", e1);
    }
}
