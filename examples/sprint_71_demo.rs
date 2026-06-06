//! SPRINT 71.0: ZNE Rescue & Adaptive Mitigation Demo
//!
//! Demonstrates the key improvements to error mitigation in Sprint 71:
//!
//! 1. **Robust ZNE Validity**: Bundle of checks (monotonicity, intercept sanity, uncertainty, R²)
//! 2. **Adaptive λ Selection**: Noise factors selected based on measured circuit stability
//! 3. **Gate-Noise-Only Scaling**: λ amplifies gate errors, not readout errors
//! 4. **Pre-Mitigation**: REM→ZNE architecture with diagnostics
//! 5. **Explainable Decisions**: Diagnostics explain what happened and why

use engine::algorithms::vqe::{VQEConfig, run_vqe};
use engine::experiments::noise_model::NoiseConfig;
use engine::hamiltonians::Hamiltonian;
use engine::hardware::profiles;
use engine::mitigation::{
    ExtrapolationMethod, MitigationDiagnostics, MitigationStrategy, ReadoutCalibration, ZNEConfig,
    ZNEValidation,
};

fn main() {
    println!("{}", "=".repeat(80));
    println!("SPRINT 71.0: ZNE Rescue & Adaptive Mitigation Demo");
    println!("{}", "=".repeat(80));
    println!();

    // H₂ molecule at equilibrium bond length
    let hamiltonian = Hamiltonian::h2_molecule(0.735);
    let exact_energy = -1.137_270; // FCI energy

    println!("Hamiltonian: H₂ molecule at 0.735 Å");
    println!("Exact FCI energy: {:.6} Ha", exact_energy);
    println!("Number of qubits: {}", hamiltonian.n_qubits);
    println!();

    // === Scenario 1: Robust ZNE Validation ===
    println!("{}", "-".repeat(80));
    println!("Scenario 1: Robust ZNE Validation (Detecting Bad Extrapolations)");
    println!("{}", "-".repeat(80));
    println!();

    println!("Testing ZNE with VERY HIGH noise (should fail validation)...");
    let very_high_noise = NoiseConfig::very_high(); // Will cause non-monotonic E(λ)

    let config_fail = VQEConfig::hardware_efficient(2)
        .with_mitigation(
            MitigationStrategy::ZNE(ZNEConfig {
                noise_factors: vec![1.0, 2.0, 3.0],
                extrapolation: ExtrapolationMethod::Richardson,
            }),
            very_high_noise.clone(),
        )
        .with_shots(500)
        .with_seed(42);

    println!("Running VQE with very high noise...");
    let result_fail = run_vqe(hamiltonian.clone(), config_fail);

    println!("Result: Energy = {:.6} Ha", result_fail.energy);
    println!(
        "Error vs FCI: {:.6} Ha",
        (result_fail.energy - exact_energy).abs()
    );
    println!();
    println!("Note: With validation enabled, ZNE will fall back to baseline (λ=1.0)");
    println!("      when extrapolation is mathematically unsound.");
    println!();

    // === Scenario 2: Adaptive Noise Factor Selection ===
    println!("{}", "-".repeat(80));
    println!("Scenario 2: Adaptive λ Selection (Measured Circuit Stability)");
    println!("{}", "-".repeat(80));
    println!();

    println!("Demonstrating adaptive noise factor selection...");
    println!();

    // Create a cost function for probing
    let noise_medium = NoiseConfig::medium();

    println!("Simulating adaptive selection:");
    println!("  1. Probe circuit at λ=1.0 to measure variance");
    println!("  2. Estimate effective error from coefficient of variation");
    println!("  3. Select λ_max to keep amplified noise manageable");
    println!("  4. Generate λ ladder [1.0, λ_mid, λ_max]");
    println!();

    // Example: For stable circuits, we can go higher
    println!("For stable circuit (low variance):");
    println!("  → Could select λ up to 5.0");
    println!();

    println!("For noisy circuit (high variance):");
    println!("  → Conservative λ_max = 1.5-2.0");
    println!();

    println!("This prevents catastrophic ZNE failures by adapting to circuit properties.");
    println!();

    // === Scenario 3: Gate-Noise-Only Scaling ===
    println!("{}", "-".repeat(80));
    println!("Scenario 3: Gate-Noise-Only Scaling (REM Compatibility)");
    println!("{}", "-".repeat(80));
    println!();

    println!("Key insight: λ amplifies ONLY gate errors, not measurement errors!");
    println!();

    let base = NoiseConfig::medium();
    let amplified_2x = base.amplify_by(2.0);

    println!("Base noise (λ=1.0):");
    println!("  Depolarizing rate: {:.4}", base.depolarizing_rate);
    println!("  Gate fidelity: {:.4}", base.gate_fidelity);
    println!(
        "  Measurement error: {:.4} ← (READOUT)",
        base.measurement_error_rate
    );
    println!();

    println!("Amplified noise (λ=2.0):");
    println!(
        "  Depolarizing rate: {:.4} ← (2× amplified)",
        amplified_2x.depolarizing_rate
    );
    println!(
        "  Gate fidelity: {:.4} ← (degraded)",
        amplified_2x.gate_fidelity
    );
    println!(
        "  Measurement error: {:.4} ← (UNCHANGED!)",
        amplified_2x.measurement_error_rate
    );
    println!();

    println!("Why this matters:");
    println!("  • Same REM calibration works at all λ levels");
    println!("  • Enables REM→ZNE pre-mitigation architecture");
    println!("  • Calibration built once at λ=1.0, used everywhere");
    println!();

    // === Scenario 4: Pre-Mitigation with REM→ZNE ===
    println!("{}", "-".repeat(80));
    println!("Scenario 4: Pre-Mitigation (REM→ZNE Pipeline)");
    println!("{}", "-".repeat(80));
    println!();

    println!("Combined mitigation architecture:");
    println!("  1. Apply REM at each noise level (corrects readout errors)");
    println!("  2. Extrapolate REM-corrected values to λ=0 (corrects gate errors)");
    println!("  3. Validate extrapolation and fall back to REM-only if needed");
    println!();

    let profile = profiles::ibm_falcon();
    println!(
        "Building readout calibration for {} ({} qubits)...",
        profile.name, hamiltonian.n_qubits
    );

    let calibration = ReadoutCalibration::build(&profile, hamiltonian.n_qubits as usize, 2000, 42);
    println!(
        "  Calibration complete! Matrix: {}×{}",
        calibration.calibration_matrix.dimension(),
        calibration.calibration_matrix.dimension()
    );
    println!();

    let config_combined = VQEConfig::hardware_efficient(2)
        .with_mitigation(
            MitigationStrategy::ZNEPlusReadout {
                zne: ZNEConfig {
                    noise_factors: vec![1.0, 2.0, 3.0],
                    extrapolation: ExtrapolationMethod::Richardson,
                },
                readout: calibration,
            },
            noise_medium.clone(),
        )
        .with_shots(500)
        .with_seed(42);

    println!("Running VQE with combined REM+ZNE...");
    let result_combined = run_vqe(hamiltonian.clone(), config_combined);

    let error_combined = (result_combined.energy - exact_energy).abs();
    println!();
    println!("Results:");
    println!("  Energy: {:.6} Ha", result_combined.energy);
    println!(
        "  Error:  {:.6} Ha ({:.2}%)",
        error_combined,
        error_combined / exact_energy.abs() * 100.0
    );
    println!("  Iterations: {}", result_combined.n_iterations);
    println!();

    // === Scenario 5: Explainable Diagnostics ===
    println!("{}", "-".repeat(80));
    println!("Scenario 5: Explainable Diagnostics (Transparency)");
    println!("{}", "-".repeat(80));
    println!();

    println!("Sprint 71 includes comprehensive diagnostics explaining decisions:");
    println!();

    // Example diagnostic (simulated)
    let example_validation = ZNEValidation {
        intercept_uncertainty: 0.02,
        is_monotonic: true,
        intercept_sanity: true,
        fit_quality: 0.95,
        is_valid: true,
        failure_reason: None,
    };

    let example_diag = MitigationDiagnostics::combined(
        -1.12,
        -1.14,
        example_validation,
        vec![1.0, 2.0, 3.0],
        true, // Adaptive
    );

    example_diag.print();

    println!("What diagnostics tell you:");
    println!("  ✓ Which strategy was used (adaptive vs fixed)");
    println!("  ✓ Whether ZNE validation passed (and why if it failed)");
    println!("  ✓ Improvement percentage vs baseline");
    println!("  ✓ Warning messages if fallback occurred");
    println!();

    // === Summary ===
    println!("{}", "=".repeat(80));
    println!("Sprint 71.0 Summary");
    println!("{}", "=".repeat(80));
    println!();

    println!("Key Improvements:");
    println!();

    println!("1. Robust Validity Gate");
    println!("   • Bundle of checks: monotonicity, intercept sanity, uncertainty, R²");
    println!("   • Prevents catastrophic -32% degradation (Sprint 68 issue)");
    println!("   • Graceful fallback when extrapolation is unsound");
    println!();

    println!("2. Adaptive λ Selection");
    println!("   • Probe circuit at λ=1.0 to measure stability");
    println!("   • Select noise factors based on measured variance");
    println!("   • Stable circuits → higher λ, noisy circuits → conservative λ");
    println!();

    println!("3. Gate-Noise-Only Scaling");
    println!("   • λ amplifies gate errors, preserves measurement errors");
    println!("   • Enables REM calibration reuse at all noise levels");
    println!("   • Foundation for REM→ZNE pre-mitigation");
    println!();

    println!("4. Pre-Mitigation Architecture");
    println!("   • Apply REM at each noise level");
    println!("   • Extrapolate REM-corrected measurements");
    println!("   • Best-of-both: readout + gate error correction");
    println!();

    println!("5. Explainable Diagnostics");
    println!("   • Transparency into mitigation decisions");
    println!("   • Human-readable explanations");
    println!("   • Warning messages for fallback scenarios");
    println!();

    println!("Philosophy:");
    println!("  \"ZNE when mathematically entitled to work, graceful fallback otherwise\"");
    println!();

    println!("{}", "=".repeat(80));
    println!("Demo complete!");
    println!("{}", "=".repeat(80));
}
