// =============================================================================
// VQE (Variational Quantum Eigensolver) Demo
// =============================================================================
//
// Demonstrates the VQE algorithm for finding molecular ground state energies.
//
// VQE is a hybrid quantum-classical algorithm:
// 1. Quantum: Prepare trial wavefunctions |ψ(θ)⟩ with parameterized circuits
// 2. Classical: Optimize parameters θ to minimize energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
//
// Run with:
//   cargo run --example vqe_demo --release
//
// =============================================================================

use engine::algorithms::vqe::{
    AnsatzType, OptimizerType, VQEConfig, h2_fci_energy, h2_ground_state, run_vqe,
};
use engine::hamiltonians::Hamiltonian;
use std::time::Instant;

fn main() {
    print_header();

    // Demo 1: Simple single-qubit Hamiltonian (validation)
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO 1: Single Qubit H = Z                                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    demo_single_qubit();

    // Demo 2: Two-qubit Hamiltonian
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO 2: Two-Qubit XX + ZZ Hamiltonian                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    demo_two_qubit();

    // Demo 3: H₂ molecule at equilibrium
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO 3: H₂ Molecule Ground State (Equilibrium)                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    demo_h2_equilibrium();

    // Demo 4: H₂ dissociation curve
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO 4: H₂ Dissociation Curve                                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    demo_h2_dissociation();

    // Demo 5: Optimizer comparison
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  DEMO 5: Optimizer Comparison                                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    demo_optimizer_comparison();

    print_footer();
}

fn print_header() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║              VARIATIONAL QUANTUM EIGENSOLVER                   ║");
    println!("║                      (VQE) DEMO                                ║");
    println!("║                                                                ║");
    println!("║  Finding molecular ground states with quantum-classical        ║");
    println!("║  hybrid optimization                                           ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn print_footer() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                      DEMO COMPLETE                             ║");
    println!("║                                                                ║");
    println!("║  VQE successfully calculates molecular ground states!          ║");
    println!("║  This is a key milestone for quantum chemistry.                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}

fn demo_single_qubit() {
    // H = Z, ground state = |1⟩, E = -1
    let h = Hamiltonian::single_qubit_z();
    let config = VQEConfig::hardware_efficient(1).with_shots(1000);

    println!("  Hamiltonian: H = Z");
    println!("  Expected ground state: |1⟩ with E = -1");
    println!("  Ansatz: Hardware-efficient (depth 1)\n");

    let start = Instant::now();
    let result = run_vqe(h, config);
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    VQE energy:     {:.6}", result.energy);
    println!("    Exact energy:   -1.0");
    println!("    Error:          {:.6}", (result.energy - (-1.0)).abs());
    println!("    Converged:      {}", result.converged);
    println!("    Iterations:     {}", result.n_iterations);
    println!("    Evaluations:    {}", result.n_evaluations);
    println!("    Time:           {:?}", elapsed);
}

fn demo_two_qubit() {
    let h = Hamiltonian::two_qubit_xx_zz();
    let config = VQEConfig::hardware_efficient(2).with_shots(1000);

    println!("  Hamiltonian: H = XX + 0.5 ZZ");
    println!("  Ansatz: Hardware-efficient (depth 2)\n");

    let start = Instant::now();
    let result = run_vqe(h.clone(), config);
    let elapsed = start.elapsed();

    // Compute exact ground state for comparison
    let exact = h.exact_ground_state_energy();

    println!("  Results:");
    println!("    VQE energy:     {:.6}", result.energy);
    println!("    Exact energy:   {:.6}", exact);
    println!("    Error:          {:.6}", (result.energy - exact).abs());
    println!("    Iterations:     {}", result.n_iterations);
    println!("    Evaluations:    {}", result.n_evaluations);
    println!("    Time:           {:?}", elapsed);
}

fn demo_h2_equilibrium() {
    let bond_length = 0.735; // Ångström (equilibrium geometry)

    println!("  Molecule: H₂ (hydrogen molecule)");
    println!("  Bond length: {} Å (equilibrium)", bond_length);
    println!("  Ansatz: UCCSD (chemically-inspired)");
    println!("  Optimizer: Nelder-Mead\n");

    let start = Instant::now();
    let result = h2_ground_state(bond_length);
    let elapsed = start.elapsed();

    let exact = h2_fci_energy(bond_length);
    let error_hartree = (result.energy - exact).abs();
    let error_kcal_mol = error_hartree * 627.5; // Convert to kcal/mol

    println!("  Results:");
    println!("    VQE energy:       {:.6} Hartree", result.energy);
    println!("    FCI energy:       {:.6} Hartree", exact);
    println!("    Error:            {:.6} Hartree", error_hartree);
    println!("    Error:            {:.4} kcal/mol", error_kcal_mol);
    println!(
        "    Chemical accuracy: {}",
        if error_kcal_mol < 1.6 { "YES" } else { "NO" }
    );
    println!("    (< 1.6 kcal/mol)");
    println!("    Converged:        {}", result.converged);
    println!("    Iterations:       {}", result.n_iterations);
    println!("    Time:             {:?}", elapsed);
}

fn demo_h2_dissociation() {
    let bond_lengths = [0.5, 0.735, 1.0, 1.5, 2.0];

    println!("  Computing H₂ potential energy surface...\n");
    println!("  Bond Length (Å) │ VQE Energy │ FCI Energy │ Error");
    println!("  ────────────────┼────────────┼────────────┼───────────");

    let start = Instant::now();

    for &r in &bond_lengths {
        let result = h2_ground_state(r);
        let exact = h2_fci_energy(r);
        let error = (result.energy - exact).abs();

        println!(
            "      {:.3}        │  {:.6}  │  {:.6}  │  {:.6}",
            r, result.energy, exact, error
        );
    }

    let elapsed = start.elapsed();

    println!("\n  Total time: {:?}", elapsed);
    println!(
        "  Average time per point: {:?}",
        elapsed / bond_lengths.len() as u32
    );
}

fn demo_optimizer_comparison() {
    let h = Hamiltonian::single_qubit_z();
    let (_ansatz, _) = engine::algorithms::vqe::hardware_efficient_ansatz(1, 1);

    println!("  Comparing optimizers on H = Z...\n");

    // Nelder-Mead
    let config_nm = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::NelderMead {
            tolerance: 1e-4,
            max_iterations: 200,
        },
        n_shots: 500,
        seed: 42,
        initial_params: None,
        mitigation: None,
        base_noise: None,
    };

    let start = Instant::now();
    let result_nm = run_vqe(h.clone(), config_nm);
    let time_nm = start.elapsed();

    // Gradient Descent
    let config_gd = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::GradientDescent {
            learning_rate: 0.5,
            momentum: 0.0,
            max_iterations: 200,
        },
        n_shots: 500,
        seed: 42,
        initial_params: Some(vec![1.0, 1.0, 1.0]),
        mitigation: None,
        base_noise: None,
    };

    let start = Instant::now();
    let result_gd = run_vqe(h.clone(), config_gd);
    let time_gd = start.elapsed();

    // SPSA
    let config_spsa = VQEConfig {
        ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
        optimizer: OptimizerType::SPSA {
            max_iterations: 200,
        },
        n_shots: 500,
        seed: 42,
        initial_params: Some(vec![0.5, 0.5, 0.5]),
        mitigation: None,
        base_noise: None,
    };

    let start = Instant::now();
    let result_spsa = run_vqe(h.clone(), config_spsa);
    let time_spsa = start.elapsed();

    println!("  Optimizer      │ Energy  │ Iters │ Evals  │ Time");
    println!("  ───────────────┼─────────┼───────┼────────┼─────────");
    println!(
        "  Nelder-Mead    │ {:+.4}  │ {:>5} │ {:>6} │ {:?}",
        result_nm.energy, result_nm.n_iterations, result_nm.n_evaluations, time_nm
    );
    println!(
        "  Grad Descent   │ {:+.4}  │ {:>5} │ {:>6} │ {:?}",
        result_gd.energy, result_gd.n_iterations, result_gd.n_evaluations, time_gd
    );
    println!(
        "  SPSA           │ {:+.4}  │ {:>5} │ {:>6} │ {:?}",
        result_spsa.energy, result_spsa.n_iterations, result_spsa.n_evaluations, time_spsa
    );

    println!("\n  Notes:");
    println!("    - Nelder-Mead: Derivative-free, robust, good for noisy problems");
    println!("    - Gradient Descent: Uses parameter-shift rule for exact gradients");
    println!("    - SPSA: Stochastic, efficient for high-dimensional problems");
}
