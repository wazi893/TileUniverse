// =============================================================================
// tests/vqe_integration.rs
// =============================================================================
//
// Integration tests for the Variational Quantum Eigensolver (VQE) algorithm.
// These tests verify the full quantum-classical hybrid optimization loop works
// correctly for ground state energy estimation.
//
// =============================================================================

use engine::algorithms::vqe::{AnsatzType, OptimizerType, VQEConfig, h2_fci_energy_2q, run_vqe};
use engine::hamiltonians::{Hamiltonian, Pauli, PauliTerm};

// Keep default `cargo test` practical: use a lighter 2-qubit H2 setup for
// integration checks while still exercising the VQE H2 path.
fn h2_ground_state_fast(bond_length: f64) -> engine::algorithms::vqe::VQEResult {
    let h = Hamiltonian::h2_molecule_2q(bond_length);
    let config = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 2 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-3,
            max_iterations: 60,
        },
        n_shots: 256,
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
    run_vqe(h, config)
}

/// Test that VQE finds the correct ground state for H = Z (single qubit).
/// Ground state is |1⟩ with energy -1.
#[test]
fn test_vqe_single_qubit_z_ground_state() {
    let h = Hamiltonian::single_qubit_z();
    let config = VQEConfig::hardware_efficient(1)
        .with_shots(1000)
        .with_seed(42);

    let result = run_vqe(h, config);

    // Should find energy very close to -1.0
    assert!(
        result.energy < -0.9,
        "Single qubit H=Z should have ground state energy near -1.0, got {}",
        result.energy
    );
    assert!(result.converged, "VQE should converge for simple problem");
    assert!(result.n_iterations > 0, "Should have taken some iterations");
}

/// Test that VQE energy decreases monotonically (or nearly so).
#[test]
fn test_vqe_energy_decreases() {
    let h = Hamiltonian::single_qubit_z();
    let config = VQEConfig::hardware_efficient(1).with_shots(500);

    let result = run_vqe(h, config);

    // Check that final energy is less than or equal to initial energy
    if !result.energy_history.is_empty() {
        let initial = result.energy_history[0];
        let final_e = result.energy;
        assert!(
            final_e <= initial + 0.1, // Allow small tolerance for noise
            "Energy should decrease: initial={}, final={}",
            initial,
            final_e
        );
    }
}

/// Test VQE with two-qubit XX + ZZ Hamiltonian.
#[test]
fn test_vqe_two_qubit_xx_zz() {
    let h = Hamiltonian::two_qubit_xx_zz();
    let exact = h.exact_ground_state_energy();

    let config = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 2 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-4,
            max_iterations: 300,
        },
        n_shots: 1000,
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

    let result = run_vqe(h, config);

    // VQE should find a negative energy (lower than trivial |00⟩)
    // Note: accuracy depends on ansatz expressibility and optimizer convergence
    assert!(
        result.energy < 0.0,
        "Two-qubit VQE should find negative energy, got {}",
        result.energy
    );

    println!(
        "Two-qubit XX+ZZ: VQE={:.4}, exact={:.4}",
        result.energy, exact
    );
}

/// Test that UCCSD ansatz produces valid circuits with H₂ Hamiltonian.
#[test]
fn test_vqe_uccsd_ansatz_runs() {
    // Use the H₂ molecule Hamiltonian (4 qubits)
    let h = Hamiltonian::h2_molecule(0.735);

    let config = VQEConfig {
        ansatz_type: AnsatzType::UCCSD { n_electrons: 2 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-3,
            max_iterations: 100,
        },
        n_shots: 500,
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

    let result = run_vqe(h, config);

    // Verify it runs and produces a result
    assert!(
        result.energy.is_finite(),
        "UCCSD should produce finite energy"
    );
    assert!(result.n_iterations > 0, "Should have run some iterations");

    // VQE energy should be negative for H₂
    println!("UCCSD VQE energy: {:.4} Ha", result.energy);
}

/// Test H2 molecule ground state at equilibrium distance.
#[test]
fn test_vqe_h2_equilibrium() {
    let bond_length = 0.735;
    let result = h2_ground_state_fast(bond_length);
    let fci = h2_fci_energy_2q(bond_length);

    // This fast test uses the reduced 2-qubit H2 model.
    // Just verify it produces a reasonable negative energy
    assert!(
        result.energy < 0.0,
        "H2 ground state energy should be negative, got {}",
        result.energy
    );
    assert!(
        result.n_iterations > 0,
        "Should have performed optimization"
    );

    println!(
        "H2 at r={}: VQE={:.4}, FCI={:.4}",
        bond_length, result.energy, fci
    );
}

/// Test that VQE works with different optimizers.
#[test]
fn test_vqe_optimizer_variants() {
    let h = Hamiltonian::single_qubit_z();

    // Test Nelder-Mead
    let config_nm = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-3,
            max_iterations: 100,
        },
        n_shots: 500,
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
    let result_nm = run_vqe(h.clone(), config_nm);
    assert!(
        result_nm.energy < 0.0,
        "Nelder-Mead should find negative energy"
    );

    // Test Gradient Descent
    let config_gd = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::GradientDescent {
            learning_rate: 0.5,
            momentum: 0.0,
            max_iterations: 100,
        },
        n_shots: 500,
        seed: 42,
        initial_params: Some(vec![1.0, 1.0, 1.0]),
        mitigation: None,
        base_noise: None,
        #[cfg(feature = "vqe_fp8")]
        use_fp8_backend: false,
        #[cfg(feature = "vqe_fp8")]
        batch_size: 256,
        #[cfg(feature = "vqe_fp8")]
        precision_threshold: 60,
    };
    let result_gd = run_vqe(h.clone(), config_gd);
    assert!(
        result_gd.energy < 0.5,
        "Gradient Descent should improve from initial, got {}",
        result_gd.energy
    );

    // Test SPSA
    let config_spsa = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::SPSA {
            max_iterations: 100,
        },
        n_shots: 500,
        seed: 42,
        initial_params: Some(vec![0.5, 0.5, 0.5]),
        mitigation: None,
        base_noise: None,
        #[cfg(feature = "vqe_fp8")]
        use_fp8_backend: false,
        #[cfg(feature = "vqe_fp8")]
        batch_size: 256,
        #[cfg(feature = "vqe_fp8")]
        precision_threshold: 60,
    };
    let result_spsa = run_vqe(h.clone(), config_spsa);
    assert!(
        result_spsa.energy.is_finite(),
        "SPSA should produce finite result"
    );
}

/// Test that VQE respects seed for reproducibility.
#[test]
fn test_vqe_reproducibility() {
    let h = Hamiltonian::single_qubit_z();

    let config = VQEConfig::hardware_efficient(1)
        .with_shots(500)
        .with_seed(12345);

    let result1 = run_vqe(h.clone(), config.clone());
    let result2 = run_vqe(h.clone(), config);

    // With same seed, should get same energy
    assert!(
        (result1.energy - result2.energy).abs() < 1e-10,
        "Same seed should give same energy: {} vs {}",
        result1.energy,
        result2.energy
    );
}

/// Test VQE with custom initial parameters.
#[test]
fn test_vqe_custom_initial_params() {
    let h = Hamiltonian::single_qubit_z();

    // Start near the optimal (pi rotation to |1⟩)
    let good_initial = vec![std::f64::consts::PI, 0.0, 0.0];

    let config = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-4,
            max_iterations: 50,
        },
        n_shots: 500,
        seed: 42,
        initial_params: Some(good_initial),
        mitigation: None,
        base_noise: None,
        #[cfg(feature = "vqe_fp8")]
        use_fp8_backend: false,
        #[cfg(feature = "vqe_fp8")]
        batch_size: 256,
        #[cfg(feature = "vqe_fp8")]
        precision_threshold: 60,
    };

    let result = run_vqe(h, config);

    // Should converge quickly from good initial guess
    assert!(
        result.energy < -0.9,
        "From good initial params, should find ground state: {}",
        result.energy
    );
}

/// Test Hamiltonian construction and measurement.
#[test]
fn test_hamiltonian_measurement() {
    // H = Z has eigenvalues +1 (|0⟩) and -1 (|1⟩)
    let h = Hamiltonian::single_qubit_z();
    assert_eq!(h.n_qubits, 1);
    assert_eq!(h.terms.len(), 1);

    let exact = h.exact_ground_state_energy();
    assert!(
        (exact - (-1.0)).abs() < 1e-10,
        "Exact ground state of Z is -1"
    );
}

/// Test multi-term Hamiltonian.
#[test]
fn test_multi_term_hamiltonian() {
    let h = Hamiltonian::two_qubit_xx_zz();

    // Should have XX and ZZ terms
    assert!(h.terms.len() >= 2, "Should have multiple terms");
    assert_eq!(h.n_qubits, 2);

    let exact = h.exact_ground_state_energy();
    assert!(exact < 0.0, "Ground state should be negative");
}

/// Test custom Hamiltonian construction.
#[test]
fn test_custom_hamiltonian() {
    // H = 0.5*X + 0.5*Z (single qubit)
    let h = Hamiltonian {
        n_qubits: 1,
        terms: vec![
            PauliTerm {
                coefficient: 0.5,
                paulis: vec![Pauli::X],
            },
            PauliTerm {
                coefficient: 0.5,
                paulis: vec![Pauli::Z],
            },
        ],
    };

    let config = VQEConfig::hardware_efficient(1).with_shots(500);
    let result = run_vqe(h.clone(), config);

    let exact = h.exact_ground_state_energy();

    assert!(
        result.energy < 0.0,
        "Ground state should be negative: {}",
        result.energy
    );
    println!("Custom H: VQE={:.4}, exact={:.4}", result.energy, exact);
}

/// Test VQE with depth scaling.
#[test]
fn test_vqe_depth_scaling() {
    let h = Hamiltonian::single_qubit_z();

    for depth in 1..=3 {
        let config = VQEConfig::hardware_efficient(depth)
            .with_shots(500)
            .with_seed(42);

        let result = run_vqe(h.clone(), config);

        assert!(
            result.energy < -0.5,
            "Depth {} should find good solution: {}",
            depth,
            result.energy
        );
    }
}

/// Test that VQE handles larger qubit counts.
#[test]
fn test_vqe_scaling_qubits() {
    // 3-qubit Hamiltonian: H = Z0 + Z1 + Z2
    let h = Hamiltonian {
        n_qubits: 3,
        terms: vec![
            PauliTerm {
                coefficient: 1.0,
                paulis: vec![Pauli::Z, Pauli::I, Pauli::I],
            },
            PauliTerm {
                coefficient: 1.0,
                paulis: vec![Pauli::I, Pauli::Z, Pauli::I],
            },
            PauliTerm {
                coefficient: 1.0,
                paulis: vec![Pauli::I, Pauli::I, Pauli::Z],
            },
        ],
    };

    // Ground state: |111⟩ with E = -3
    let config = VQEConfig::hardware_efficient(2).with_shots(1000);
    let result = run_vqe(h.clone(), config);

    let exact = h.exact_ground_state_energy();
    assert!(
        (exact - (-3.0)).abs() < 1e-6,
        "Exact should be -3: {}",
        exact
    );

    // VQE should get reasonably close
    assert!(
        result.energy < -1.5,
        "3-qubit VQE should find low energy: {}",
        result.energy
    );
}

/// Test H2 dissociation curve points.
#[test]
fn test_h2_dissociation_curve() {
    let bond_lengths = [0.5, 0.735, 1.0, 1.5];

    for &r in &bond_lengths {
        let result = h2_ground_state_fast(r);
        let fci = h2_fci_energy_2q(r);

        assert!(
            result.energy.is_finite(),
            "Energy at r={} should be finite",
            r
        );
        assert!(
            result.energy < 0.0,
            "H2 should have negative energy at r={}: {}",
            r,
            result.energy
        );

        println!("H2 r={}: VQE={:.4}, FCI={:.4}", r, result.energy, fci);
    }
}

/// Test that VQE result contains history.
#[test]
fn test_vqe_result_structure() {
    let h = Hamiltonian::single_qubit_z();
    let config = VQEConfig::hardware_efficient(1).with_shots(500);

    let result = run_vqe(h, config);

    assert!(result.params.len() > 0, "Should have parameters");
    assert!(result.n_iterations > 0, "Should have iterations");
    assert!(result.n_evaluations > 0, "Should have evaluations");
    // Energy history may be empty depending on optimizer
}

/// Stress test: Run VQE multiple times with different seeds.
#[test]
fn test_vqe_statistical_consistency() {
    let h = Hamiltonian::single_qubit_z();
    let n_trials = 10;
    let mut energies = Vec::new();

    for seed in 0..n_trials {
        let config = VQEConfig::hardware_efficient(1)
            .with_shots(500)
            .with_seed(seed as u64);

        let result = run_vqe(h.clone(), config);
        energies.push(result.energy);
    }

    // All trials should find approximately the same ground state
    let mean: f64 = energies.iter().sum::<f64>() / n_trials as f64;
    let max_deviation = energies
        .iter()
        .map(|&e| (e - mean).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_deviation < 0.5,
        "VQE results should be consistent: mean={}, max_dev={}",
        mean,
        max_deviation
    );
    assert!(
        mean < -0.5,
        "Mean energy should be close to ground state: {}",
        mean
    );
}
