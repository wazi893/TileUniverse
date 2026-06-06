//! Error Mitigation Demo
//!
//! SPRINT 57.0: Demonstrate Zero-Noise Extrapolation (ZNE) and
//! Readout Error Mitigation (REM) on VQE H₂ ground state.
//!
//! This example shows:
//! - VQE without mitigation (baseline with errors)
//! - VQE with ZNE (extrapolation to zero noise)
//! - VQE with Readout Error Mitigation (calibration-based correction)
//! - Performance comparison and improvement percentages

use engine::algorithms::vqe::{VQEConfig, run_vqe};
use engine::experiments::noise_model::NoiseConfig;
use engine::hamiltonians::Hamiltonian;
use engine::hardware::profiles;
use engine::mitigation::{ExtrapolationMethod, MitigationStrategy, ReadoutCalibration, ZNEConfig};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Error Mitigation Demo: VQE H₂ Ground State");
    println!("SPRINT 57.0: Zero-Noise Extrapolation + Readout Error Mitigation");
    println!("{}", "=".repeat(70));
    println!();

    // H₂ molecule at equilibrium bond length (0.735 Å)
    let hamiltonian = Hamiltonian::h2_molecule(0.735);
    let exact_energy = -1.137_270; // FCI energy

    println!("Hamiltonian: H₂ molecule at 0.735 Å");
    println!("Exact FCI energy: {:.6} Ha", exact_energy);
    println!("Number of qubits: {}", hamiltonian.n_qubits);
    println!();

    // Baseline configuration (no mitigation)
    let base_config = VQEConfig::hardware_efficient(2)
        .with_shots(1000)
        .with_seed(42);

    // SPRINT 68.0: Define noise model for all experiments
    let noise_config = NoiseConfig::medium();

    println!("VQE Configuration:");
    println!("  Ansatz: Hardware-Efficient (depth 2)");
    println!("  Optimizer: Nelder-Mead");
    println!("  Shots: 1000");
    println!();

    println!("Noise Model: Medium");
    println!("  Depolarizing rate: {:.4}", noise_config.depolarizing_rate);
    println!("  Phase error rate: {:.4}", noise_config.phase_error_rate);
    println!("  Bit flip rate: {:.4}", noise_config.bit_flip_rate);
    println!(
        "  Measurement error: {:.4}",
        noise_config.measurement_error_rate
    );
    println!("  Gate fidelity: {:.4}", noise_config.gate_fidelity);
    println!();

    // === 1. Baseline: Noisy execution, No Mitigation ===
    println!("{}", "-".repeat(70));
    println!("1. Baseline: Noisy Execution (NO Mitigation)");
    println!("{}", "-".repeat(70));

    // Create baseline config with noise but no mitigation
    let baseline_config = base_config
        .clone()
        .with_mitigation(MitigationStrategy::None, noise_config.clone());

    let result_baseline = run_vqe(hamiltonian.clone(), baseline_config);
    let error_baseline = (result_baseline.energy - exact_energy).abs();

    println!("Energy: {:.6} Ha", result_baseline.energy);
    println!(
        "Error:  {:.6} Ha ({:.2}%)",
        error_baseline,
        error_baseline / exact_energy.abs() * 100.0
    );
    println!(
        "Converged: {} ({} iterations)",
        result_baseline.converged, result_baseline.n_iterations
    );
    println!();

    // === 2. Zero-Noise Extrapolation ===
    println!("{}", "-".repeat(70));
    println!("2. Zero-Noise Extrapolation (ZNE)");
    println!("{}", "-".repeat(70));

    let zne_config = ZNEConfig {
        noise_factors: vec![1.0, 2.0, 3.0],
        extrapolation: ExtrapolationMethod::Richardson,
    };

    println!("ZNE Configuration:");
    println!("  Noise factors: {:?}", zne_config.noise_factors);
    println!("  Extrapolation: Richardson");
    println!("  Overhead: {}x runtime", zne_config.noise_factors.len());
    println!();

    let strategy_zne = MitigationStrategy::ZNE(zne_config.clone());
    let config_zne = base_config
        .clone()
        .with_mitigation(strategy_zne, noise_config.clone());

    let result_zne = run_vqe(hamiltonian.clone(), config_zne);
    let error_zne = (result_zne.energy - exact_energy).abs();
    let improvement_zne = ((error_baseline - error_zne) / error_baseline * 100.0).max(0.0);

    println!("Results:");
    println!("  Energy: {:.6} Ha", result_zne.energy);
    println!(
        "  Error:  {:.6} Ha ({:.2}%)",
        error_zne,
        error_zne / exact_energy.abs() * 100.0
    );
    println!(
        "  Improvement: {:.1}% error reduction vs baseline",
        improvement_zne
    );
    println!();

    // === 3. Readout Error Mitigation ===
    println!("{}", "-".repeat(70));
    println!("3. Readout Error Mitigation (REM)");
    println!("{}", "-".repeat(70));

    let profile = profiles::ibm_falcon();
    println!(
        "Hardware: {} ({} qubits)",
        profile.name, profile.qubit_count
    );
    println!("Measurement fidelity: {:.4}", profile.measure_fidelity);
    println!();

    println!("Building calibration matrix...");
    println!("  Qubits: {}", hamiltonian.n_qubits);
    println!("  Shots per basis state: 5000");
    println!("  Total basis states: {}", 1 << hamiltonian.n_qubits);

    let calibration = ReadoutCalibration::build(&profile, hamiltonian.n_qubits as usize, 5000, 42);

    println!("  Calibration complete!");
    println!(
        "  Matrix dimension: {}×{}",
        calibration.calibration_matrix.dimension(),
        calibration.calibration_matrix.dimension()
    );
    println!(
        "  Expected improvement: {:.2}x",
        calibration.expected_improvement()
    );
    println!();

    let strategy_rem = MitigationStrategy::Readout(calibration);
    let config_rem = base_config
        .clone()
        .with_mitigation(strategy_rem, noise_config.clone());

    let result_rem = run_vqe(hamiltonian.clone(), config_rem);
    let error_rem = (result_rem.energy - exact_energy).abs();
    let improvement_rem = ((error_baseline - error_rem) / error_baseline * 100.0).max(0.0);

    println!("Results:");
    println!("  Energy: {:.6} Ha", result_rem.energy);
    println!(
        "  Error:  {:.6} Ha ({:.2}%)",
        error_rem,
        error_rem / exact_energy.abs() * 100.0
    );
    println!(
        "  Improvement: {:.1}% error reduction vs baseline",
        improvement_rem
    );
    println!();

    // === 4. Combined ZNE + REM ===
    println!("{}", "-".repeat(70));
    println!("4. Combined: ZNE + Readout Mitigation");
    println!("{}", "-".repeat(70));

    let calibration2 = ReadoutCalibration::build(&profile, hamiltonian.n_qubits as usize, 5000, 43);
    let strategy_combined = MitigationStrategy::ZNEPlusReadout {
        zne: zne_config.clone(),
        readout: calibration2,
    };

    println!("Configuration:");
    println!("  ZNE: Richardson extrapolation, 3 noise levels");
    println!("  REM: Full calibration, {} qubits", hamiltonian.n_qubits);
    println!(
        "  Overhead: {}x runtime",
        strategy_combined.overhead_factor()
    );
    println!();

    let config_combined = base_config
        .clone()
        .with_mitigation(strategy_combined, noise_config);

    let result_combined = run_vqe(hamiltonian, config_combined);
    let error_combined = (result_combined.energy - exact_energy).abs();
    let improvement_combined =
        ((error_baseline - error_combined) / error_baseline * 100.0).max(0.0);

    println!("Results:");
    println!("  Energy: {:.6} Ha", result_combined.energy);
    println!(
        "  Error:  {:.6} Ha ({:.2}%)",
        error_combined,
        error_combined / exact_energy.abs() * 100.0
    );
    println!(
        "  Improvement: {:.1}% error reduction vs baseline",
        improvement_combined
    );
    println!();

    // === Summary ===
    println!("{}", "=".repeat(70));
    println!("Summary: Error Mitigation Performance");
    println!("{}", "=".repeat(70));
    println!();

    println!(
        "{:<25} {:>15} {:>15} {:>15}",
        "Method", "Energy (Ha)", "Error (Ha)", "Improvement"
    );
    println!("{}", "-".repeat(72));
    println!(
        "{:<25} {:>15.6} {:>15.6} {:>15}",
        "Baseline (no mitigation)", result_baseline.energy, error_baseline, "-"
    );
    println!(
        "{:<25} {:>15.6} {:>15.6} {:>14.1}%",
        "ZNE (Richardson)", result_zne.energy, error_zne, improvement_zne
    );
    println!(
        "{:<25} {:>15.6} {:>15.6} {:>14.1}%",
        "Readout Mitigation", result_rem.energy, error_rem, improvement_rem
    );
    println!(
        "{:<25} {:>15.6} {:>15.6} {:>14.1}%",
        "Combined (ZNE + REM)", result_combined.energy, error_combined, improvement_combined
    );
    println!("{}", "-".repeat(72));
    println!("Exact FCI:               {:>15.6}", exact_energy);
    println!();

    // Key takeaways
    println!("Key Takeaways:");
    println!("• ZNE reduces errors by amplifying noise and extrapolating to λ=0");
    println!("• REM corrects measurement errors via calibration matrix inversion");
    println!("• Combined techniques provide additive error reduction");
    println!(
        "• Overhead: ZNE is {}x, REM is ~1x (calibration is one-time)",
        zne_config.noise_factors.len()
    );
    println!("• Practical for NISQ devices with n ≤ 10 qubits (REM memory limit)");
    println!();

    println!("{}", "=".repeat(70));
    println!("Demo complete!");
    println!("{}", "=".repeat(70));
}
