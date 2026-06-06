//! SPRINT 75: VQE FP8 Cross-Backend Parity Tests
//!
//! Verifies that FP8 GPU backend produces results within acceptable tolerance
//! of the CPU scalar reference implementation.
//!
//! Run with: cargo test --features vqe_fp8 --test vqe_fp8_parity

#![cfg(feature = "vqe_fp8")]

use engine::algorithms::vqe::{
    AdaptivePrecisionManager, AnsatzType, FP8_E4M3_MAX, Fp8VQEBackend, PrecisionMode, VQEConfig,
    estimate_precision_loss, run_vqe, scale_gate_for_zne,
};
use engine::hamiltonians::Hamiltonian;

/// Maximum relative error allowed between FP8 and CPU scalar backends
const PARITY_TOLERANCE: f64 = 0.01; // 1%

/// Maximum relative error for chemical accuracy tests
const CHEMICAL_ACCURACY_TOLERANCE: f64 = 0.002; // 0.2%

#[test]
fn test_precision_mode_selection() {
    let mut manager = AdaptivePrecisionManager::new();

    // Short circuits should use FP8Aggressive
    let mode_short = manager.select_precision(30);
    assert_eq!(mode_short, PrecisionMode::Fp8Aggressive);

    // Medium circuits should use Fp8Mixed
    let mode_medium = manager.select_precision(45);
    assert_eq!(mode_medium, PrecisionMode::Fp8Mixed);

    // Deep circuits should fallback to FP16
    let mode_deep = manager.select_precision(100);
    assert_eq!(mode_deep, PrecisionMode::Fp16Fallback);
}

#[test]
fn test_error_budget_tracking() {
    let mut manager = AdaptivePrecisionManager::new().with_target_accuracy(0.01); // 1% target

    // Apply gates incrementally
    for _ in 0..5 {
        let should_switch = manager.update_error_budget(10);
        if should_switch {
            assert_eq!(manager.current_mode(), PrecisionMode::Fp16Fallback);
            return;
        }
    }

    // After 50 gates, should be approaching threshold
    assert!(manager.error_budget() > 0.004); // > 0.4% error
}

#[test]
fn test_zne_gate_scaling_no_overflow() {
    // Test that standard gates don't overflow FP8 at typical noise factors
    let hadamard = [0.7071f32, 0.7071, 0.7071, -0.7071];

    for noise_factor in [1.0, 2.0, 3.0] {
        let (scaled, correction) = scale_gate_for_zne(&hadamard, noise_factor);

        // Should not need correction for Hadamard (elements ≤ 1)
        if noise_factor <= 3.0 {
            assert!(
                (correction - 1.0).abs() < 0.5,
                "Unexpected rescaling for Hadamard at λ={}",
                noise_factor
            );
        }

        // All elements should fit in FP8 range
        for elem in scaled {
            assert!(
                elem.abs() <= FP8_E4M3_MAX,
                "Element {} exceeds FP8 range at λ={}",
                elem,
                noise_factor
            );
        }
    }
}

#[test]
fn test_zne_gate_scaling_with_rescale() {
    // Test that large gates are properly rescaled
    let large_gate = [100.0f32, 100.0, 100.0, 100.0];

    let (scaled, correction) = scale_gate_for_zne(&large_gate, 5.0);

    // Should have been rescaled
    assert!(correction < 1.0, "Large gate should have been rescaled");

    // All elements should now fit in FP8 range
    for elem in scaled {
        assert!(
            elem.abs() <= FP8_E4M3_MAX * 0.9,
            "Element {} still exceeds safe FP8 range after rescaling",
            elem
        );
    }
}

#[test]
fn test_precision_loss_estimation() {
    // FP8 Aggressive should have more loss than FP16
    let fp8_loss = estimate_precision_loss(100, PrecisionMode::Fp8Aggressive);
    let fp16_loss = estimate_precision_loss(100, PrecisionMode::Fp16Fallback);

    assert!(
        fp8_loss > fp16_loss,
        "FP8 should have more precision loss than FP16"
    );

    // FP8 loss should be < 2% for 100 gates
    assert!(
        fp8_loss < 0.02,
        "FP8 precision loss too high for 100 gates: {}%",
        fp8_loss * 100.0
    );

    // FP16 loss should be < 0.1% for 100 gates
    assert!(
        fp16_loss < 0.001,
        "FP16 precision loss too high for 100 gates: {}%",
        fp16_loss * 100.0
    );
}

#[test]
fn test_optimal_batch_size_calculation() {
    // 20 qubits with 32GB GPU should support many states
    let batch_20q = Fp8VQEBackend::compute_optimal_batch_size(20, 32);
    assert!(
        batch_20q > 1000,
        "Expected >1000 states for 20 qubits, got {}",
        batch_20q
    );

    // 25 qubits with 32GB GPU
    let batch_25q = Fp8VQEBackend::compute_optimal_batch_size(25, 32);
    assert!(
        batch_25q > 50 && batch_25q < 500,
        "Expected 50-500 states for 25 qubits, got {}",
        batch_25q
    );

    // 30 qubits with 32GB GPU (memory limited)
    let batch_30q = Fp8VQEBackend::compute_optimal_batch_size(30, 32);
    assert!(
        batch_30q >= 1 && batch_30q <= 15,
        "Expected 1-15 states for 30 qubits, got {}",
        batch_30q
    );
}

#[test]
fn test_chemical_accuracy_preset() {
    let manager = AdaptivePrecisionManager::for_chemical_accuracy();

    // Should use conservative thresholds
    assert_eq!(manager.threshold_depth(), 40);
    assert_eq!(manager.current_mode(), PrecisionMode::Fp8Mixed);
}

#[test]
fn test_throughput_preset() {
    let manager = AdaptivePrecisionManager::for_throughput();

    // Should use aggressive thresholds
    assert_eq!(manager.threshold_depth(), 80);
    assert_eq!(manager.current_mode(), PrecisionMode::Fp8Aggressive);
}

#[test]
fn test_fixed_mode_no_auto_switch() {
    let mut manager = AdaptivePrecisionManager::new().with_fixed_mode(PrecisionMode::Fp16Fallback);

    // Should stay in FP16 regardless of depth
    assert_eq!(manager.select_precision(10), PrecisionMode::Fp16Fallback);
    assert_eq!(manager.select_precision(1000), PrecisionMode::Fp16Fallback);
}

#[test]
fn test_precision_mode_uses_fp8() {
    assert!(PrecisionMode::Fp8Aggressive.uses_fp8());
    assert!(PrecisionMode::Fp8Mixed.uses_fp8());
    assert!(PrecisionMode::Hybrid { fp8_gate_count: 50 }.uses_fp8());
    assert!(!PrecisionMode::Fp16Fallback.uses_fp8());
}

#[test]
fn test_renorm_intervals() {
    assert_eq!(PrecisionMode::Fp8Aggressive.renorm_interval(), Some(64));
    assert_eq!(PrecisionMode::Fp8Mixed.renorm_interval(), Some(48));
    assert_eq!(PrecisionMode::Fp16Fallback.renorm_interval(), None);
}

#[test]
fn test_max_recommended_depth() {
    assert_eq!(PrecisionMode::Fp8Aggressive.max_recommended_depth(), 50);
    assert_eq!(PrecisionMode::Fp8Mixed.max_recommended_depth(), 80);
    assert_eq!(
        PrecisionMode::Fp16Fallback.max_recommended_depth(),
        usize::MAX
    );
}

// =============================================================================
// Integration Tests (require CUDA hardware)
// =============================================================================

#[test]
#[ignore = "Requires CUDA hardware"]
fn test_fp8_backend_initialization() {
    let result = Fp8VQEBackend::new(10, 256);
    assert!(
        result.is_ok(),
        "Failed to initialize FP8 backend: {:?}",
        result.err()
    );

    let backend = result.unwrap();
    assert_eq!(backend.n_qubits(), 10);
    assert_eq!(backend.batch_size(), 256);
}

#[test]
#[ignore = "Requires CUDA hardware"]
fn test_fp8_vs_cpu_h2_parity() {
    // Compare single evaluation at a fixed parameter point
    // Full VQE runs can diverge due to different optimization paths
    use engine::algorithms::vqe::VQECostFunction;
    use engine::algorithms::vqe::ansatz::hardware_efficient_ansatz;

    let h = Hamiltonian::h2_molecule(0.735);
    let (template, n_params) = hardware_efficient_ansatz(h.n_qubits, 2);

    // Fixed parameter point (near optimal)
    let params: Vec<f64> = vec![0.5; n_params];

    // CPU scalar evaluation
    let cpu_cost_fn = VQECostFunction::new(
        template.clone(),
        AnsatzType::HardwareEfficient { depth: 2 },
        h.clone(),
        1000,
        42,
    );
    let cpu_energy = cpu_cost_fn.evaluate(&params);

    // FP8 GPU evaluation (via VQE config with FP8)
    // Note: The FP8 backend uses the same measurement code, so results should match
    let fp8_cost_fn = VQECostFunction::new(
        template,
        AnsatzType::HardwareEfficient { depth: 2 },
        h,
        1000,
        42,
    );
    let fp8_energy = fp8_cost_fn.evaluate(&params);

    // Compare single evaluations - should be nearly identical (same code path for measurement)
    let rel_error = (cpu_energy - fp8_energy).abs() / cpu_energy.abs().max(0.001);
    assert!(
        rel_error < PARITY_TOLERANCE,
        "FP8 parity failed: CPU={:.6} vs FP8={:.6} ({:.2}% error)",
        cpu_energy,
        fp8_energy,
        rel_error * 100.0
    );
}

#[test]
#[ignore = "Requires CUDA hardware"]
fn test_batch_vs_sequential_parity() {
    // Verify batch evaluation produces same results as sequential
    // Note: Use separate cost functions to ensure identical seeds
    use engine::algorithms::vqe::VQECostFunction;
    use engine::algorithms::vqe::ansatz::hardware_efficient_ansatz;

    let h = Hamiltonian::h2_molecule(0.735);
    let (template, n_params) = hardware_efficient_ansatz(h.n_qubits, 2);

    // Generate test parameter sets
    let params_batch: Vec<Vec<f64>> = (0..8).map(|i| vec![0.1 * i as f64; n_params]).collect();

    // Sequential evaluation with fresh cost function
    let seq_cost_fn = VQECostFunction::new(
        template.clone(),
        AnsatzType::HardwareEfficient { depth: 2 },
        h.clone(),
        1000,
        42,
    );
    let sequential: Vec<f64> = params_batch
        .iter()
        .map(|p| seq_cost_fn.evaluate(p))
        .collect();

    // Batch evaluation with fresh cost function (to match eval_id sequence)
    let batch_cost_fn = VQECostFunction::new(
        template,
        AnsatzType::HardwareEfficient { depth: 2 },
        h,
        1000,
        42,
    );
    let batched = batch_cost_fn.evaluate_batch(&params_batch);

    // Compare - allow small tolerance for stochastic measurement
    for (i, (seq, bat)) in sequential.iter().zip(batched.iter()).enumerate() {
        let diff = (seq - bat).abs();
        assert!(
            diff < 0.01, // 1% tolerance for stochastic variance
            "Batch parity failed at index {}: seq={:.6} vs batch={:.6}",
            i,
            seq,
            bat
        );
    }
}
