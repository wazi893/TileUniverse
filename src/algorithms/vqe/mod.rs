// =============================================================================
// VQE (Variational Quantum Eigensolver) Module
// =============================================================================
//
// Hybrid quantum-classical algorithm for finding ground state energies
// of molecular Hamiltonians.
//
// VQE uses a parameterized quantum circuit (ansatz) to prepare trial
// wavefunctions, and a classical optimizer to find parameters that
// minimize the energy expectation value: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
//
// SPRINT 75: Added FP8 tensor core acceleration for high-throughput VQE.
// Enable with `--features vqe_fp8` for 10-26x speedup on 20-30 qubit circuits.
//
// =============================================================================

pub mod ansatz;
pub mod noisy_execution;
pub mod optimizer;

// SPRINT 75: FP8 Tensor Core modules (feature-gated)
#[cfg(feature = "vqe_fp8")]
pub mod adaptive_precision;
#[cfg(feature = "vqe_fp8")]
pub mod batched_measurement;
#[cfg(feature = "vqe_fp8")]
pub mod fp8_backend;

// Re-exports
pub use ansatz::{
    apply_ansatz, apply_parameters, build_uccsd_circuit, hardware_efficient_ansatz, uccsd_ansatz,
};
pub use noisy_execution::{
    execute_circuit, execute_circuit_with_noise, measure_hamiltonian_with_noise,
    measure_hamiltonian_with_rem,
};
pub use optimizer::{
    GradientDescentOptimizer, NelderMeadOptimizer, SPSAOptimizer, VQECostFunction, VQEResult,
};

// SPRINT 75: FP8 re-exports (feature-gated)
#[cfg(feature = "vqe_fp8")]
pub use adaptive_precision::{
    AdaptivePrecisionManager, FP8_E4M3_MAX, FP8_E5M2_MAX, PrecisionMode, estimate_precision_loss,
    scale_gate_for_zne,
};
#[cfg(feature = "vqe_fp8")]
pub use batched_measurement::{
    BatchMeasurementResults, BatchedMeasurement, BatchedMeasurementConfig, MeasurementStats,
};
#[cfg(feature = "vqe_fp8")]
pub use fp8_backend::{
    CompiledTemplate, Fp8KernelCacheExtended, Fp8VQEBackend, validate_gates_for_fp8,
};

use crate::hamiltonians::Hamiltonian;

/// Ansatz type for trial wavefunction preparation
#[derive(Debug, Clone)]
pub enum AnsatzType {
    HardwareEfficient { depth: usize },
    UCCSD { n_electrons: u8 },
}

/// Classical optimizer type
#[derive(Debug, Clone)]
pub enum OptimizerType {
    NelderMead {
        tolerance: f64,
        max_iterations: usize,
    },
    GradientDescent {
        learning_rate: f64,
        momentum: f64,
        max_iterations: usize,
    },
    SPSA {
        max_iterations: usize,
    },
}

/// VQE algorithm configuration
#[derive(Debug, Clone)]
pub struct VQEConfig {
    pub ansatz_type: AnsatzType,
    pub optimizer: OptimizerType,
    pub n_shots: usize,
    pub seed: u64,
    pub initial_params: Option<Vec<f64>>,

    // SPRINT 57.0: Error mitigation support
    /// Error mitigation strategy (optional)
    pub mitigation: Option<crate::mitigation::MitigationStrategy>,
    /// Base noise configuration for mitigation
    pub base_noise: Option<crate::experiments::noise_model::NoiseConfig>,

    // SPRINT 75: FP8 Tensor Core acceleration
    /// Enable FP8 GPU backend for high-throughput evaluation
    #[cfg(feature = "vqe_fp8")]
    pub use_fp8_backend: bool,
    /// Maximum batch size for parallel parameter evaluations
    #[cfg(feature = "vqe_fp8")]
    pub batch_size: usize,
    /// Circuit depth threshold for FP16 fallback (default: 60)
    #[cfg(feature = "vqe_fp8")]
    pub precision_threshold: usize,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            ansatz_type: AnsatzType::HardwareEfficient { depth: 2 },
            optimizer: OptimizerType::NelderMead {
                tolerance: 1e-4,
                max_iterations: 200,
            },
            n_shots: 1000,
            seed: 42,
            initial_params: None,
            mitigation: None,
            base_noise: None,
            // SPRINT 75: FP8 defaults
            #[cfg(feature = "vqe_fp8")]
            use_fp8_backend: false,
            #[cfg(feature = "vqe_fp8")]
            batch_size: 256,
            #[cfg(feature = "vqe_fp8")]
            precision_threshold: 60,
        }
    }
}

impl VQEConfig {
    pub fn hardware_efficient(depth: usize) -> Self {
        Self {
            ansatz_type: AnsatzType::HardwareEfficient { depth },
            ..Default::default()
        }
    }

    pub fn uccsd(n_electrons: u8) -> Self {
        Self {
            ansatz_type: AnsatzType::UCCSD { n_electrons },
            ..Default::default()
        }
    }

    pub fn with_gradient_descent(mut self, learning_rate: f64) -> Self {
        self.optimizer = OptimizerType::GradientDescent {
            learning_rate,
            momentum: 0.9,
            max_iterations: 200,
        };
        self
    }

    pub fn with_spsa(mut self) -> Self {
        self.optimizer = OptimizerType::SPSA {
            max_iterations: 200,
        };
        self
    }

    pub fn with_shots(mut self, n_shots: usize) -> Self {
        self.n_shots = n_shots;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Add error mitigation strategy (SPRINT 57.0)
    ///
    /// # Arguments
    ///
    /// * `strategy` - Mitigation technique (ZNE, Readout, or combined)
    /// * `noise` - Base noise configuration to amplify/correct
    ///
    /// # Example
    ///
    /// ```ignore
    /// use engine::mitigation::{ZNEConfig, ExtrapolationMethod, MitigationStrategy};
    /// use engine::experiments::noise_model::NoiseConfig;
    ///
    /// let zne = MitigationStrategy::ZNE(ZNEConfig {
    ///     noise_factors: vec![1.0, 2.0, 3.0],
    ///     extrapolation: ExtrapolationMethod::Richardson,
    /// });
    ///
    /// let config = VQEConfig::hardware_efficient(2)
    ///     .with_mitigation(zne, NoiseConfig::medium());
    /// ```
    pub fn with_mitigation(
        mut self,
        strategy: crate::mitigation::MitigationStrategy,
        noise: crate::experiments::noise_model::NoiseConfig,
    ) -> Self {
        self.mitigation = Some(strategy);
        self.base_noise = Some(noise);
        self
    }

    // =========================================================================
    // SPRINT 75: FP8 Tensor Core Configuration
    // =========================================================================

    /// Enable FP8 GPU backend for high-throughput evaluation
    ///
    /// Uses CUDA FP8 tensor cores for 10-26x speedup on supported GPUs
    /// (Ada Lovelace SM 8.9+ or Blackwell SM 10.0+).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = VQEConfig::hardware_efficient(2)
    ///     .with_fp8_backend()
    ///     .with_batch_size(256)
    ///     .with_spsa();
    /// ```
    #[cfg(feature = "vqe_fp8")]
    pub fn with_fp8_backend(mut self) -> Self {
        self.use_fp8_backend = true;
        self
    }

    /// Stub when FP8 feature is not enabled
    #[cfg(not(feature = "vqe_fp8"))]
    pub fn with_fp8_backend(self) -> Self {
        eprintln!("Warning: vqe_fp8 feature not enabled, using CPU backend");
        self
    }

    /// Set batch size for parallel parameter evaluations
    ///
    /// Higher values increase throughput but require more GPU memory.
    /// Use `Fp8VQEBackend::compute_optimal_batch_size()` to determine
    /// the maximum batch size for your GPU and qubit count.
    ///
    /// # Arguments
    /// * `size` - Number of parameter sets to evaluate in parallel
    #[cfg(feature = "vqe_fp8")]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Stub when FP8 feature is not enabled
    #[cfg(not(feature = "vqe_fp8"))]
    pub fn with_batch_size(self, _size: usize) -> Self {
        self
    }

    /// Set circuit depth threshold for FP16 fallback
    ///
    /// Circuits deeper than this threshold will use FP16 WMMA
    /// instead of FP8 for improved precision.
    ///
    /// Recommended values:
    /// - 40-50 for chemical accuracy requirements
    /// - 60-80 for general use (default)
    /// - 100+ for throughput-critical applications
    ///
    /// # Arguments
    /// * `depth` - Maximum circuit depth for FP8 mode
    #[cfg(feature = "vqe_fp8")]
    pub fn with_precision_threshold(mut self, depth: usize) -> Self {
        self.precision_threshold = depth;
        self
    }

    /// Stub when FP8 feature is not enabled
    #[cfg(not(feature = "vqe_fp8"))]
    pub fn with_precision_threshold(self, _depth: usize) -> Self {
        self
    }
}

/// Run VQE to find ground state energy of a Hamiltonian
pub fn run_vqe(hamiltonian: Hamiltonian, config: VQEConfig) -> VQEResult {
    let (ansatz_template, n_params) = match config.ansatz_type {
        AnsatzType::HardwareEfficient { depth } => {
            ansatz::hardware_efficient_ansatz(hamiltonian.n_qubits, depth)
        }
        AnsatzType::UCCSD { n_electrons } => {
            ansatz::uccsd_ansatz(hamiltonian.n_qubits, n_electrons)
        }
    };

    let initial_params = config.initial_params.unwrap_or_else(|| vec![0.0; n_params]);
    let mut cost_fn = VQECostFunction::new(
        ansatz_template,
        config.ansatz_type.clone(),
        hamiltonian.clone(),
        config.n_shots,
        config.seed,
    );

    // SPRINT 68.0: Apply error mitigation if configured
    if let (Some(mitigation), Some(noise)) = (config.mitigation, config.base_noise) {
        cost_fn = cost_fn.with_mitigation(noise, mitigation);
    }

    // SPRINT 75: Configure FP8 GPU backend if enabled
    #[cfg(feature = "vqe_fp8")]
    if config.use_fp8_backend {
        match Fp8VQEBackend::new(hamiltonian.n_qubits as usize, config.batch_size) {
            Ok(backend) => {
                cost_fn = cost_fn.with_fp8_backend(backend);
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to initialize FP8 backend, using CPU: {:?}",
                    e
                );
            }
        }
    }

    match config.optimizer {
        OptimizerType::NelderMead {
            tolerance,
            max_iterations,
        } => NelderMeadOptimizer::new()
            .with_tolerance(tolerance)
            .with_max_iterations(max_iterations)
            .optimize(&cost_fn, &initial_params),
        OptimizerType::GradientDescent {
            learning_rate,
            momentum,
            max_iterations,
        } => GradientDescentOptimizer::new(learning_rate)
            .with_momentum(momentum)
            .with_max_iterations(max_iterations)
            .optimize(&cost_fn, &initial_params),
        OptimizerType::SPSA { max_iterations } => SPSAOptimizer::new()
            .with_max_iterations(max_iterations)
            .optimize(&cost_fn, &initial_params),
    }
}

/// Find H₂ ground state at given bond length (Ångströms) using 4-qubit form
///
/// Uses the 4-qubit Jordan-Wigner Hamiltonian with UCCSD ansatz.
/// The UCCSD ansatz provides chemically-motivated excitation operators.
///
/// Note: This form is only calibrated for R≈0.74 Å. For dissociation curves,
/// use `h2_ground_state_2q()` which supports 0.30-1.80 Å.
pub fn h2_ground_state(bond_length: f64) -> VQEResult {
    let hamiltonian = Hamiltonian::h2_molecule(bond_length);
    // Use UCCSD ansatz for the 4-qubit molecular problem
    let config = VQEConfig {
        ansatz_type: AnsatzType::UCCSD { n_electrons: 2 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-6,
            max_iterations: 500,
        },
        n_shots: 10000,
        seed: 42,
        initial_params: None,
        mitigation: None,
        base_noise: None,
        #[cfg(feature = "vqe_fp8")]
        use_fp8_backend: false,
        #[cfg(feature = "vqe_fp8")]
        batch_size: 256,
        #[cfg(feature = "vqe_fp8")]
        precision_threshold: 60,
    };
    run_vqe(hamiltonian, config)
}

/// Find H₂ ground state at given bond length (Ångströms) using 2-qubit reduced form
///
/// Uses the 2-qubit Bravyi-Kitaev/parity mapped Hamiltonian with hardware-efficient ansatz.
/// Supports bond lengths from 0.30 to 1.80 Å with interpolated coefficients.
///
/// This is the recommended function for computing H₂ dissociation curves.
pub fn h2_ground_state_2q(bond_length: f64) -> VQEResult {
    let hamiltonian = Hamiltonian::h2_molecule_2q(bond_length);
    let config = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 3 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-6,
            max_iterations: 300,
        },
        n_shots: 5000,
        seed: 42,
        initial_params: None,
        mitigation: None,
        base_noise: None,
        #[cfg(feature = "vqe_fp8")]
        use_fp8_backend: false,
        #[cfg(feature = "vqe_fp8")]
        batch_size: 256,
        #[cfg(feature = "vqe_fp8")]
        precision_threshold: 60,
    };
    run_vqe(hamiltonian, config)
}

/// Get exact FCI energy for H₂ at given bond length (4-qubit form)
///
/// Computes exact ground state via full matrix diagonalization.
/// Only accurate at equilibrium (R≈0.74 Å) for the 4-qubit form.
pub fn h2_fci_energy(bond_length: f64) -> f64 {
    Hamiltonian::h2_molecule(bond_length).exact_ground_state_energy()
}

/// Get exact FCI energy for H₂ at given bond length (2-qubit form)
///
/// Computes exact ground state via full matrix diagonalization.
/// Supports the full range from 0.30 to 1.80 Å.
pub fn h2_fci_energy_2q(bond_length: f64) -> f64 {
    Hamiltonian::h2_fci_energy_2q(bond_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqe_single_qubit_z() {
        let h = Hamiltonian::single_qubit_z();
        let result = run_vqe(h, VQEConfig::hardware_efficient(1).with_shots(500));
        assert!(
            result.energy < 0.0,
            "Expected negative energy, got {}",
            result.energy
        );
    }

    #[test]
    fn test_vqe_uccsd_ansatz() {
        let h = Hamiltonian::h2_molecule(0.735);
        let result = run_vqe(h, VQEConfig::uccsd(2).with_shots(500));
        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_h2_fci_energy_lookup() {
        let fci = h2_fci_energy(0.735);
        assert!(
            fci < -1.0 && fci > -1.2,
            "FCI energy should be around -1.1 Ha, got {}",
            fci
        );
    }

    #[test]
    fn test_h2_2q_coefficients_interpolation() {
        // Test that interpolation produces reasonable values
        let h_050 = Hamiltonian::h2_molecule_2q(0.50);
        let h_070 = Hamiltonian::h2_molecule_2q(0.70);
        let h_100 = Hamiltonian::h2_molecule_2q(1.00);

        assert_eq!(h_050.n_qubits, 2);
        assert_eq!(h_050.terms.len(), 6); // I, Z0, Z1, ZZ, XX, YY

        // Check that energies are reasonable (all negative for H2)
        let e_050 = h_050.exact_ground_state_energy();
        let e_070 = h_070.exact_ground_state_energy();
        let e_100 = h_100.exact_ground_state_energy();

        println!("H2 2-qubit FCI energies:");
        println!("  R=0.50 Å: {} Ha", e_050);
        println!("  R=0.70 Å: {} Ha", e_070);
        println!("  R=1.00 Å: {} Ha", e_100);

        // All energies should be negative (bound state)
        assert!(e_050 < 0.0, "E(0.5) should be negative, got {}", e_050);
        assert!(e_070 < 0.0, "E(0.7) should be negative, got {}", e_070);
        assert!(e_100 < 0.0, "E(1.0) should be negative, got {}", e_100);
    }

    #[test]
    fn test_h2_2q_fci_dissociation_curve() {
        // Test FCI energies at multiple bond lengths
        let bond_lengths = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8];

        println!("H2 2-qubit dissociation curve:");
        for &r in &bond_lengths {
            let fci = h2_fci_energy_2q(r);
            println!("  R={:.1} Å: {:.6} Ha", r, fci);
            assert!(
                fci.is_finite() && fci < 0.0,
                "FCI at R={} should be negative, got {}",
                r,
                fci
            );
        }
    }
}
