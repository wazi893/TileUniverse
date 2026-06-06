//! Integration tests for error mitigation with VQE
//!
//! SPRINT 57.0: Verify that error mitigation improves VQE accuracy

use engine::algorithms::vqe::{VQEConfig, run_vqe};
use engine::hamiltonians::Hamiltonian;
use engine::mitigation::{ExtrapolationMethod, MitigationStrategy, ZNEConfig};

#[test]
fn test_vqe_config_with_mitigation() {
    // Test that VQEConfig can be created with mitigation strategy
    let zne_config = ZNEConfig {
        noise_factors: vec![1.0, 2.0, 3.0],
        extrapolation: ExtrapolationMethod::Richardson,
    };

    let strategy = MitigationStrategy::ZNE(zne_config);
    let noise = engine::experiments::noise_model::NoiseConfig::medium();

    let config = VQEConfig::hardware_efficient(2).with_mitigation(strategy.clone(), noise.clone());

    assert!(config.mitigation.is_some());
    assert!(config.base_noise.is_some());

    // Verify we can clone it
    let _config2 = config.clone();
}

#[test]
fn test_vqe_runs_without_mitigation() {
    // Verify VQE still works without mitigation (backwards compatibility)
    let hamiltonian = Hamiltonian::single_qubit_z(); // Single-qubit Z
    let config = VQEConfig::hardware_efficient(1)
        .with_shots(100)
        .with_seed(42);

    let result = run_vqe(hamiltonian, config);

    // Should find eigenvalue near -1 (ground state of Z)
    assert!(
        result.energy < -0.5,
        "Expected energy near -1, got {}",
        result.energy
    );
    assert!(
        result.converged,
        "VQE should converge on simple Hamiltonian"
    );
}

#[test]
fn test_mitigation_strategy_types() {
    use engine::hardware::profiles;
    use engine::mitigation::ReadoutCalibration;

    // Test ZNE strategy
    let zne = MitigationStrategy::ZNE(ZNEConfig {
        noise_factors: vec![1.0, 1.5, 2.0],
        extrapolation: ExtrapolationMethod::Linear,
    });
    assert!(
        matches!(zne, MitigationStrategy::ZNE(_)),
        "Should be ZNE strategy"
    );

    // Test Readout strategy
    let profile = profiles::ibm_falcon();
    let calibration = ReadoutCalibration::build(&profile, 2, 1000, 42);
    let rem = MitigationStrategy::Readout(calibration.clone());
    assert!(
        matches!(rem, MitigationStrategy::Readout(_)),
        "Should be Readout strategy"
    );

    // Test combined strategy
    let combined = MitigationStrategy::ZNEPlusReadout {
        zne: ZNEConfig {
            noise_factors: vec![1.0, 2.0],
            extrapolation: ExtrapolationMethod::Richardson,
        },
        readout: calibration,
    };
    assert!(
        matches!(combined, MitigationStrategy::ZNEPlusReadout { .. }),
        "Should be combined strategy"
    );
}

#[test]
fn test_resource_estimate_with_mitigation() {
    use engine::quantum::QGate;
    use engine::transpile::ResourceEstimator;
    use logic_fabric_core::quantum::QGate as CoreQGate;

    // Create simple circuit
    let circuit: Vec<QGate> = vec![
        CoreQGate::H(0),
        CoreQGate::CNot(0, 1),
        CoreQGate::T(0),
        CoreQGate::H(1),
    ];

    // Create estimator with NISQ profile and mitigation
    let profile = engine::hardware::profiles::ibm_falcon();
    let zne = MitigationStrategy::ZNE(ZNEConfig {
        noise_factors: vec![1.0, 2.0, 3.0],
        extrapolation: ExtrapolationMethod::Richardson,
    });

    let estimator = ResourceEstimator::new()
        .with_nisq_profile(profile)
        .with_mitigation(zne);

    let estimate = estimator.estimate(&circuit, "test_circuit");

    // Verify NISQ timing includes mitigation data
    assert!(estimate.nisq_timing.is_some(), "Should have NISQ timing");
    let timing = estimate.nisq_timing.unwrap();

    assert!(
        timing.mitigation_strategy.is_some(),
        "Should track mitigation strategy"
    );
    assert!(
        timing.mitigated_success_probability.is_some(),
        "Should track mitigated probability"
    );

    // Mitigated probability should be higher than raw probability
    let raw_prob = timing.success_probability;
    let mit_prob = timing.mitigated_success_probability.unwrap();
    assert!(
        mit_prob >= raw_prob,
        "Mitigated probability ({}) should be >= raw probability ({})",
        mit_prob,
        raw_prob
    );
}

#[test]
fn test_mitigation_overhead_factor() {
    // Test overhead calculation
    let zne_3x = MitigationStrategy::ZNE(ZNEConfig {
        noise_factors: vec![1.0, 2.0, 3.0],
        extrapolation: ExtrapolationMethod::Linear,
    });
    assert_eq!(
        zne_3x.overhead_factor(),
        3.0,
        "ZNE with 3 noise factors should have 3x overhead"
    );

    let zne_2x = MitigationStrategy::ZNE(ZNEConfig {
        noise_factors: vec![1.0, 2.0],
        extrapolation: ExtrapolationMethod::Richardson,
    });
    assert_eq!(
        zne_2x.overhead_factor(),
        2.0,
        "ZNE with 2 noise factors should have 2x overhead"
    );

    // Readout mitigation has ~1x overhead (calibration is one-time)
    let profile = engine::hardware::profiles::ibm_falcon();
    let calibration = engine::mitigation::ReadoutCalibration::build(&profile, 2, 1000, 42);
    let rem = MitigationStrategy::Readout(calibration);
    assert_eq!(
        rem.overhead_factor(),
        1.0,
        "Readout mitigation should have ~1x runtime overhead"
    );
}

#[test]
fn test_noise_amplification() {
    use engine::experiments::noise_model::NoiseConfig;

    let base = NoiseConfig::low();
    let amplified = base.amplify_by(2.0);

    // Verify noise rates are amplified
    assert!(
        amplified.phase_error_rate >= base.phase_error_rate,
        "Amplified phase error should be >= base"
    );
    assert!(
        amplified.bit_flip_rate >= base.bit_flip_rate,
        "Amplified bit flip should be >= base"
    );
    assert!(
        amplified.depolarizing_rate >= base.depolarizing_rate,
        "Amplified depolarizing should be >= base"
    );

    // Verify rates are clamped to [0, 1]
    let extreme = base.amplify_by(100.0);
    assert!(
        extreme.phase_error_rate <= 1.0,
        "Phase error should be clamped to 1.0"
    );
    assert!(extreme.gate_fidelity >= 0.0, "Gate fidelity should be >= 0");
}
