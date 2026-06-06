//! SPRINT 72.0: Parallel ZNE Benchmark
//!
//! Measures speedup of parallel vs sequential ZNE execution.
//!
//! Expected results:
//! - CPU parallel: 2.5-3× speedup (rayon)
//! - GPU batch: 5-10× speedup (if CUDA available)

use engine::algorithms::vqe::ansatz::hardware_efficient_ansatz;
use engine::experiments::noise_model::NoiseConfig;
use engine::hamiltonians::Hamiltonian;
use engine::mitigation::zne::{
    ExtrapolationMethod, ZNEConfig, apply_zne_parallel, apply_zne_sequential,
};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(80));
    println!("SPRINT 72.0: Parallel ZNE Benchmark");
    println!("{}", "=".repeat(80));
    println!();

    // Test parameters (larger circuit for meaningful parallel speedup)
    let n_qubits: u8 = 4; // Match H2 molecule qubit count
    let depth: usize = 4; // Increased depth for more gates
    let hamiltonian = Hamiltonian::h2_molecule(0.735);
    let (ansatz, _) = hardware_efficient_ansatz(n_qubits, depth);
    let params = vec![0.5; (n_qubits as usize) * depth * 3]; // Rough parameter count

    let base_noise = NoiseConfig::medium();
    let config = ZNEConfig {
        noise_factors: vec![1.0, 2.0, 3.0],
        extrapolation: ExtrapolationMethod::Richardson,
    };

    println!("Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!("  Ansatz depth: {}", depth);
    println!("  Noise factors: {:?}", config.noise_factors);
    println!("  Base noise: medium");
    println!();

    // Create a cost function that simulates realistic VQE evaluation
    let cost_fn = |params: &[f64], noise: &NoiseConfig| -> f64 {
        use engine::hamiltonians::measure_hamiltonian_expectation;
        use engine::quantum::{QRng, QState, apply_gate_scalar};

        let mut state = QState::new_zero(hamiltonian.n_qubits);
        let mut rng = QRng::new(42);

        // Apply ansatz gates with parameters
        for (i, gate) in ansatz.iter().enumerate() {
            let gate_with_param = match gate {
                engine::quantum::QGate::Rz(q, _) => {
                    engine::quantum::QGate::Rz(*q, params[i % params.len()] as f32)
                }
                engine::quantum::QGate::Ry(q, _) => {
                    engine::quantum::QGate::Ry(*q, params[i % params.len()] as f32)
                }
                _ => gate.clone(),
            };
            let _ = apply_gate_scalar(&mut state, &gate_with_param, &mut rng);
        }

        // Measure Hamiltonian expectation (increased shots for more realistic timing)
        let energy = measure_hamiltonian_expectation(&state, &hamiltonian, 5000, 42);

        // Apply noise effect (simple multiplicative model)
        energy * (1.0 + noise.depolarizing_rate as f64)
    };

    // Benchmark 1: Sequential ZNE
    println!("{}", "-".repeat(80));
    println!("Benchmark 1: Sequential ZNE");
    println!("{}", "-".repeat(80));

    let n_runs = 10;
    let mut sequential_times = Vec::new();

    for run in 0..n_runs {
        let start = Instant::now();
        let result = apply_zne_sequential(&cost_fn, &params, &base_noise, &config);
        let elapsed = start.elapsed();
        sequential_times.push(elapsed.as_secs_f64());

        if run == 0 {
            println!("  First run result:");
            println!("    Extrapolated energy: {:.6}", result.extrapolated_value);
            println!("    Valid: {}", result.validation.is_valid);
        }
    }

    let avg_sequential = sequential_times.iter().sum::<f64>() / n_runs as f64;
    let min_sequential = sequential_times
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_sequential = sequential_times
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  Average time: {:.3} ms", avg_sequential * 1000.0);
    println!("  Min time: {:.3} ms", min_sequential * 1000.0);
    println!("  Max time: {:.3} ms", max_sequential * 1000.0);
    println!();

    // Benchmark 2: Parallel ZNE (Rayon)
    println!("{}", "-".repeat(80));
    println!("Benchmark 2: Parallel ZNE (Rayon)");
    println!("{}", "-".repeat(80));

    let mut parallel_times = Vec::new();

    for run in 0..n_runs {
        let start = Instant::now();
        let result = apply_zne_parallel(&cost_fn, &params, &base_noise, &config);
        let elapsed = start.elapsed();
        parallel_times.push(elapsed.as_secs_f64());

        if run == 0 {
            println!("  First run result:");
            println!("    Extrapolated energy: {:.6}", result.extrapolated_value);
            println!("    Valid: {}", result.validation.is_valid);
        }
    }

    let avg_parallel = parallel_times.iter().sum::<f64>() / n_runs as f64;
    let min_parallel = parallel_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_parallel = parallel_times
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  Average time: {:.3} ms", avg_parallel * 1000.0);
    println!("  Min time: {:.3} ms", min_parallel * 1000.0);
    println!("  Max time: {:.3} ms", max_parallel * 1000.0);
    println!();

    // Speedup analysis
    println!("{}", "=".repeat(80));
    println!("Speedup Analysis");
    println!("{}", "=".repeat(80));

    let speedup_avg = avg_sequential / avg_parallel;
    let speedup_best = max_sequential / min_parallel;
    let speedup_worst = min_sequential / max_parallel;

    println!("  Average speedup: {:.2}×", speedup_avg);
    println!("  Best speedup: {:.2}×", speedup_best);
    println!("  Worst speedup: {:.2}×", speedup_worst);
    println!();

    let theoretical_speedup = config.noise_factors.len() as f64;
    let efficiency = (speedup_avg / theoretical_speedup) * 100.0;

    println!(
        "  Theoretical max speedup: {:.2}× (perfect scaling)",
        theoretical_speedup
    );
    println!("  Parallel efficiency: {:.1}%", efficiency);
    println!();

    // Performance breakdown
    println!("{}", "=".repeat(80));
    println!("Performance Breakdown");
    println!("{}", "=".repeat(80));

    let per_level_time_sequential = avg_sequential / config.noise_factors.len() as f64;
    let overhead_parallel = avg_parallel - (avg_sequential / theoretical_speedup);
    let overhead_percent = (overhead_parallel / avg_parallel) * 100.0;

    println!(
        "  Time per noise level (sequential): {:.3} ms",
        per_level_time_sequential * 1000.0
    );
    println!(
        "  Parallel overhead: {:.3} ms ({:.1}%)",
        overhead_parallel * 1000.0,
        overhead_percent
    );
    println!();

    // Summary
    println!("{}", "=".repeat(80));
    println!("Summary");
    println!("{}", "=".repeat(80));
    println!();

    if speedup_avg >= 2.0 {
        println!(
            "✅ SUCCESS: Achieved {:.2}× speedup (target: ≥2.0×)",
            speedup_avg
        );
    } else {
        println!(
            "⚠️  WARNING: Only {:.2}× speedup (target: ≥2.0×)",
            speedup_avg
        );
    }

    if efficiency >= 80.0 {
        println!("✅ EXCELLENT: {:.1}% parallel efficiency", efficiency);
    } else if efficiency >= 60.0 {
        println!("✅ GOOD: {:.1}% parallel efficiency", efficiency);
    } else {
        println!("⚠️  MODERATE: {:.1}% parallel efficiency", efficiency);
    }

    println!();
    println!("Parallel ZNE is recommended for:");
    println!("  • VQE optimization (many evaluations per iteration)");
    println!("  • QAOA with multiple noise levels");
    println!("  • Circuit benchmarking and validation");
    println!();

    println!("{}", "=".repeat(80));
}
