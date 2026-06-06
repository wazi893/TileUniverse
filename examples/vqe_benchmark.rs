// =============================================================================
// VQE Comprehensive Benchmark
// =============================================================================
//
// Benchmarks VQE performance across different configurations:
// - Problem sizes (1-4 qubits)
// - Ansatz types (Hardware-Efficient, UCCSD)
// - Optimizer types (Nelder-Mead, Gradient Descent, SPSA)
// - Shot counts
//
// Run with:
//   cargo run --example vqe_benchmark --release
//
// =============================================================================

use engine::algorithms::vqe::{AnsatzType, OptimizerType, VQEConfig, h2_fci_energy, run_vqe};
use engine::hamiltonians::Hamiltonian;
use std::time::{Duration, Instant};

/// Benchmark result for a single configuration
#[derive(Debug)]
struct BenchResult {
    name: String,
    n_qubits: usize,
    n_params: usize,
    energy: f64,
    exact_energy: f64,
    error: f64,
    converged: bool,
    iterations: usize,
    evaluations: usize,
    time: Duration,
    evals_per_sec: f64,
}

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("                     VQE COMPREHENSIVE BENCHMARK");
    println!("{}\n", "=".repeat(80));

    // Benchmark 1: Qubit scaling with hardware-efficient ansatz
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 1: Qubit Scaling (Hardware-Efficient Ansatz)");
    println!("{}\n", "-".repeat(80));
    benchmark_qubit_scaling();

    // Benchmark 2: Optimizer comparison
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 2: Optimizer Comparison");
    println!("{}\n", "-".repeat(80));
    benchmark_optimizers();

    // Benchmark 3: Ansatz depth scaling
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 3: Ansatz Depth Scaling");
    println!("{}\n", "-".repeat(80));
    benchmark_depth_scaling();

    // Benchmark 4: Shot count impact
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 4: Shot Count Impact");
    println!("{}\n", "-".repeat(80));
    benchmark_shot_scaling();

    // Benchmark 5: H2 molecule dissociation curve
    println!("\n{}", "-".repeat(80));
    println!("BENCHMARK 5: H2 Dissociation Curve Performance");
    println!("{}\n", "-".repeat(80));
    benchmark_h2_curve();

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("                           BENCHMARK COMPLETE");
    println!("{}\n", "=".repeat(80));
}

fn benchmark_qubit_scaling() {
    let qubit_counts = [1, 2, 3, 4];
    let mut results = Vec::new();

    for &n_qubits in &qubit_counts {
        let h = create_random_hamiltonian(n_qubits);
        let exact = h.exact_ground_state_energy();
        let depth = 2;
        let n_params = 3 * n_qubits * depth;

        let config = VQEConfig {
            ansatz_type: AnsatzType::HardwareEfficient { depth },
            optimizer: OptimizerType::NelderMead {
                tolerance: 1e-4,
                max_iterations: 200,
            },
            n_shots: 1000,
            seed: 42,
            initial_params: None,
            mitigation: None,
            base_noise: None,
        };

        let start = Instant::now();
        let result = run_vqe(h, config);
        let elapsed = start.elapsed();

        let evals_per_sec = result.n_evaluations as f64 / elapsed.as_secs_f64();

        results.push(BenchResult {
            name: format!("{}-qubit HW-Eff", n_qubits),
            n_qubits,
            n_params,
            energy: result.energy,
            exact_energy: exact,
            error: (result.energy - exact).abs(),
            converged: result.converged,
            iterations: result.n_iterations,
            evaluations: result.n_evaluations,
            time: elapsed,
            evals_per_sec,
        });
    }

    print_results_table(&results);
}

fn benchmark_optimizers() {
    let h = Hamiltonian::two_qubit_xx_zz();
    let exact = h.exact_ground_state_energy();

    let optimizers: Vec<(&str, OptimizerType)> = vec![
        (
            "Nelder-Mead",
            OptimizerType::NelderMead {
                tolerance: 1e-4,
                max_iterations: 200,
            },
        ),
        (
            "Gradient Descent",
            OptimizerType::GradientDescent {
                learning_rate: 0.3,
                momentum: 0.0,
                max_iterations: 200,
            },
        ),
        (
            "SPSA",
            OptimizerType::SPSA {
                max_iterations: 200,
            },
        ),
    ];

    let mut results = Vec::new();

    for (name, opt) in optimizers {
        let config = VQEConfig {
            ansatz_type: AnsatzType::HardwareEfficient { depth: 2 },
            optimizer: opt,
            n_shots: 1000,
            seed: 42,
            initial_params: Some(vec![0.5; 12]), // 3 * 2 qubits * 2 depth
            mitigation: None,
            base_noise: None,
        };

        let start = Instant::now();
        let result = run_vqe(h.clone(), config);
        let elapsed = start.elapsed();

        let evals_per_sec = result.n_evaluations as f64 / elapsed.as_secs_f64();

        results.push(BenchResult {
            name: name.to_string(),
            n_qubits: 2,
            n_params: 12,
            energy: result.energy,
            exact_energy: exact,
            error: (result.energy - exact).abs(),
            converged: result.converged,
            iterations: result.n_iterations,
            evaluations: result.n_evaluations,
            time: elapsed,
            evals_per_sec,
        });
    }

    print_results_table(&results);
}

fn benchmark_depth_scaling() {
    let h = Hamiltonian::single_qubit_z();
    let exact = -1.0;
    let depths = [1, 2, 3, 4];

    let mut results = Vec::new();

    for &depth in &depths {
        let n_params = 3 * 1 * depth; // 3 rotations per qubit per layer

        let config = VQEConfig {
            ansatz_type: AnsatzType::HardwareEfficient { depth },
            optimizer: OptimizerType::NelderMead {
                tolerance: 1e-4,
                max_iterations: 200,
            },
            n_shots: 1000,
            seed: 42,
            initial_params: None,
            mitigation: None,
            base_noise: None,
        };

        let start = Instant::now();
        let result = run_vqe(h.clone(), config);
        let elapsed = start.elapsed();

        let evals_per_sec = result.n_evaluations as f64 / elapsed.as_secs_f64();

        results.push(BenchResult {
            name: format!("Depth {}", depth),
            n_qubits: 1,
            n_params,
            energy: result.energy,
            exact_energy: exact,
            error: (result.energy - exact).abs(),
            converged: result.converged,
            iterations: result.n_iterations,
            evaluations: result.n_evaluations,
            time: elapsed,
            evals_per_sec,
        });
    }

    print_results_table(&results);
}

fn benchmark_shot_scaling() {
    let h = Hamiltonian::single_qubit_z();
    let exact = -1.0;
    let shot_counts = [100, 500, 1000, 5000, 10000];

    let mut results = Vec::new();

    for &n_shots in &shot_counts {
        let config = VQEConfig {
            ansatz_type: AnsatzType::HardwareEfficient { depth: 1 },
            optimizer: OptimizerType::NelderMead {
                tolerance: 1e-4,
                max_iterations: 100,
            },
            n_shots,
            seed: 42,
            initial_params: None,
            mitigation: None,
            base_noise: None,
        };

        let start = Instant::now();
        let result = run_vqe(h.clone(), config);
        let elapsed = start.elapsed();

        let evals_per_sec = result.n_evaluations as f64 / elapsed.as_secs_f64();

        results.push(BenchResult {
            name: format!("{} shots", n_shots),
            n_qubits: 1,
            n_params: 3,
            energy: result.energy,
            exact_energy: exact,
            error: (result.energy - exact).abs(),
            converged: result.converged,
            iterations: result.n_iterations,
            evaluations: result.n_evaluations,
            time: elapsed,
            evals_per_sec,
        });
    }

    print_results_table(&results);
}

fn benchmark_h2_curve() {
    let bond_lengths = [0.5, 0.735, 1.0, 1.5, 2.0];
    let mut total_time = Duration::ZERO;

    println!(
        "  {:^12} | {:^12} | {:^12} | {:^10} | {:^10} | {:^10}",
        "Bond (A)", "VQE Energy", "FCI Energy", "Error", "Iters", "Time"
    );
    println!("  {}", "-".repeat(72));

    for &r in &bond_lengths {
        let h = Hamiltonian::h2_molecule(r);
        let exact = h2_fci_energy(r);

        let config = VQEConfig {
            ansatz_type: AnsatzType::UCCSD { n_electrons: 2 },
            optimizer: OptimizerType::NelderMead {
                tolerance: 1e-5,
                max_iterations: 300,
            },
            n_shots: 2000,
            seed: 42,
            initial_params: None,
            mitigation: None,
            base_noise: None,
        };

        let start = Instant::now();
        let result = run_vqe(h, config);
        let elapsed = start.elapsed();
        total_time += elapsed;

        let error = (result.energy - exact).abs();

        println!(
            "  {:^12.3} | {:^12.6} | {:^12.6} | {:^10.6} | {:^10} | {:^10.2?}",
            r, result.energy, exact, error, result.n_iterations, elapsed
        );
    }

    println!("  {}", "-".repeat(72));
    println!(
        "  Total time: {:?}  |  Avg per point: {:?}",
        total_time,
        total_time / bond_lengths.len() as u32
    );
}

fn create_random_hamiltonian(n_qubits: usize) -> Hamiltonian {
    match n_qubits {
        1 => Hamiltonian::single_qubit_z(),
        2 => Hamiltonian::two_qubit_xx_zz(),
        3 => create_three_qubit_hamiltonian(),
        4 => Hamiltonian::h2_molecule(0.735), // H2 is naturally 4 qubits in STO-3G
        _ => Hamiltonian::single_qubit_z(),
    }
}

fn create_three_qubit_hamiltonian() -> Hamiltonian {
    use engine::hamiltonians::{Pauli, PauliTerm};
    // H = Z0 + Z1 + Z2 + 0.5 X0X1 + 0.5 X1X2
    Hamiltonian {
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
            PauliTerm {
                coefficient: 0.5,
                paulis: vec![Pauli::X, Pauli::X, Pauli::I],
            },
            PauliTerm {
                coefficient: 0.5,
                paulis: vec![Pauli::I, Pauli::X, Pauli::X],
            },
        ],
    }
}

fn print_results_table(results: &[BenchResult]) {
    println!(
        "  {:^18} | {:^6} | {:^6} | {:^10} | {:^10} | {:^8} | {:^6} | {:^10}",
        "Config", "Qubits", "Params", "Energy", "Error", "Conv", "Iters", "Time"
    );
    println!("  {}", "-".repeat(90));

    for r in results {
        println!(
            "  {:^18} | {:^6} | {:^6} | {:^+10.4} | {:^10.6} | {:^8} | {:^6} | {:^10.2?}",
            r.name,
            r.n_qubits,
            r.n_params,
            r.energy,
            r.error,
            if r.converged { "Yes" } else { "No" },
            r.iterations,
            r.time
        );
    }

    // Summary stats
    let total_time: Duration = results.iter().map(|r| r.time).sum();
    let avg_evals: f64 =
        results.iter().map(|r| r.evals_per_sec).sum::<f64>() / results.len() as f64;

    println!("  {}", "-".repeat(90));
    println!(
        "  Total: {:?}  |  Avg evals/sec: {:.1}",
        total_time, avg_evals
    );
}
