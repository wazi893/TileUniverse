// =============================================================================
// Sprint 67.0: VQE H₂ Ground State Demo
// =============================================================================
//
// Demonstrates variational quantum eigensolver (VQE) for finding the ground
// state energy of the hydrogen molecule using the hybrid orchestration layer.
//
// This example shows:
// - Hardware-efficient ansatz construction
// - H₂ Hamiltonian creation
// - VQE optimization with different optimizers
// - Comparison with known ground state energy
//
// Run with:
//   cargo run --example vqe_h2_hybrid --release
//
// =============================================================================

use std::time::Instant;

use engine::hybrid::{
    CostFunction, H2_GROUND_STATE_ENERGY, HybridScheduler, Optimizer, QuantumCircuit,
    QuantumSubstrate, VariationalConfig, VariationalExecutor, VariationalSpec, h2_hamiltonian,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 67.0: VQE H₂ GROUND STATE DEMO");
    println!("        Variational Quantum Eigensolver");
    println!("{}\n", "=".repeat(80));

    // Create scheduler with quantum substrate
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 10)));

    // Demo 1: Simple VQE with SPSA optimizer
    println!("{}", "-".repeat(80));
    println!("DEMO 1: VQE with SPSA Optimizer");
    println!("{}\n", "-".repeat(80));
    demo_vqe_spsa(&mut scheduler);

    // Demo 2: VQE with Nelder-Mead optimizer
    println!("\n{}", "-".repeat(80));
    println!("DEMO 2: VQE with Nelder-Mead Optimizer");
    println!("{}\n", "-".repeat(80));
    demo_vqe_nelder_mead(&mut scheduler);

    // Demo 3: VQE with gradient descent
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: VQE with Gradient Descent");
    println!("{}\n", "-".repeat(80));
    demo_vqe_gradient_descent(&mut scheduler);

    // Demo 4: Bond length scan
    println!("\n{}", "-".repeat(80));
    println!("DEMO 4: H₂ Bond Length Scan");
    println!("{}\n", "-".repeat(80));
    demo_bond_length_scan(&mut scheduler);

    println!("\n{}", "=".repeat(80));
    println!("                    VQE H₂ DEMO COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Demo VQE with SPSA optimizer.
fn demo_vqe_spsa(scheduler: &mut HybridScheduler) {
    let ansatz = QuantumCircuit::hardware_efficient(2, 2);
    let hamiltonian = h2_hamiltonian(1.4); // Equilibrium distance in Bohr

    println!("Configuration:");
    println!("  Qubits: 2");
    println!("  Ansatz: Hardware-efficient (2 layers)");
    println!("  Parameters: {}", ansatz.parameter_count());
    println!("  Optimizer: SPSA");
    println!("  Target energy: {:.6} Ha", H2_GROUND_STATE_ENERGY);
    println!();

    let spec = VariationalSpec::new(ansatz, CostFunction::ExpectationValue(hamiltonian))
        .with_optimizer(Optimizer::SPSA {
            a: 0.1,
            c: 0.1,
            maxiter: 100,
        })
        .with_max_iterations(50)
        .with_shots(500)
        .with_tolerance(1e-4);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler)
        .with_config(VariationalConfig::from_spec(&spec).with_verbose(true));
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Final energy: {:.6} Ha", result.optimal_cost);
    println!(
        "  Error from exact: {:.6} Ha",
        (result.optimal_cost - H2_GROUND_STATE_ENERGY).abs()
    );
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!("  Total time: {:?}", elapsed);
    println!("  Circuits executed: {}", result.total_circuits);
}

/// Demo VQE with Nelder-Mead optimizer.
fn demo_vqe_nelder_mead(scheduler: &mut HybridScheduler) {
    let ansatz = QuantumCircuit::hardware_efficient(2, 2);
    let hamiltonian = h2_hamiltonian(1.4);

    println!("Optimizer: Nelder-Mead (simplex method)");
    println!();

    let spec = VariationalSpec::new(ansatz, CostFunction::ExpectationValue(hamiltonian))
        .with_optimizer(Optimizer::NelderMead { maxiter: 100 })
        .with_max_iterations(50)
        .with_shots(500);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler);
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Final energy: {:.6} Ha", result.optimal_cost);
    println!(
        "  Error: {:.6} Ha",
        (result.optimal_cost - H2_GROUND_STATE_ENERGY).abs()
    );
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:?}", elapsed);
}

/// Demo VQE with gradient descent.
fn demo_vqe_gradient_descent(scheduler: &mut HybridScheduler) {
    let ansatz = QuantumCircuit::hardware_efficient(2, 1); // Simpler ansatz
    let hamiltonian = h2_hamiltonian(1.4);

    println!("Optimizer: Gradient Descent (learning rate = 0.1)");
    println!("Note: Uses parameter-shift rule for gradient estimation");
    println!();

    let spec = VariationalSpec::new(ansatz, CostFunction::ExpectationValue(hamiltonian))
        .with_optimizer(Optimizer::GradientDescent {
            learning_rate: 0.1,
            maxiter: 100,
        })
        .with_max_iterations(30)
        .with_shots(500);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler)
        .with_config(VariationalConfig::from_spec(&spec).with_gradient(true));
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Final energy: {:.6} Ha", result.optimal_cost);
    println!(
        "  Error: {:.6} Ha",
        (result.optimal_cost - H2_GROUND_STATE_ENERGY).abs()
    );
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:?}", elapsed);
    println!("  Circuits (incl. gradient): {}", result.total_circuits);
}

/// Demo bond length scan.
fn demo_bond_length_scan(scheduler: &mut HybridScheduler) {
    println!("Scanning H₂ potential energy surface...\n");
    println!(
        "{:>12} {:>14} {:>14}",
        "Distance (a₀)", "Energy (Ha)", "From Exact"
    );
    println!("{}", "-".repeat(45));

    // Scan from 0.5 to 3.0 Bohr
    let bond_lengths = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0];

    let mut best_energy = f64::INFINITY;
    let mut best_distance = 0.0;

    for &r in &bond_lengths {
        let ansatz = QuantumCircuit::hardware_efficient(2, 1);
        let hamiltonian = h2_hamiltonian(r);

        let spec = VariationalSpec::new(ansatz, CostFunction::ExpectationValue(hamiltonian))
            .with_optimizer(Optimizer::SPSA {
                a: 0.1,
                c: 0.1,
                maxiter: 50,
            })
            .with_max_iterations(20)
            .with_shots(200);

        let mut executor = VariationalExecutor::new(scheduler);
        let result = executor.run(&spec);

        let error = result.optimal_cost - H2_GROUND_STATE_ENERGY;
        println!("{:>12.2} {:>14.6} {:>14.6}", r, result.optimal_cost, error);

        if result.optimal_cost < best_energy {
            best_energy = result.optimal_cost;
            best_distance = r;
        }
    }

    println!();
    println!("Best result:");
    println!("  Equilibrium distance: {:.2} a₀", best_distance);
    println!("  Minimum energy: {:.6} Ha", best_energy);
    println!("  (Expected: ~1.4 a₀, {:.6} Ha)", H2_GROUND_STATE_ENERGY);
}
