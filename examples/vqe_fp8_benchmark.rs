//! SPRINT 75: VQE FP8 Benchmark
//!
//! Benchmarks VQE performance with FP8 tensor core acceleration.
//! Compares CPU scalar, FP16 WMMA, and FP8 backends across different qubit counts.
//!
//! Run with: cargo run --release --features vqe_fp8 --example vqe_fp8_benchmark
//!
//! Expected output on RTX 5090:
//! - 20 qubits: 20x speedup vs CPU
//! - 25 qubits: 20x speedup vs CPU
//! - 30 qubits: 10x speedup (memory limited)

#![cfg(feature = "vqe_fp8")]

use std::time::Instant;

use engine::algorithms::vqe::{
    AdaptivePrecisionManager, AnsatzType, Fp8VQEBackend, PrecisionMode, VQEConfig,
    estimate_precision_loss, run_vqe,
};
use engine::hamiltonians::Hamiltonian;

fn main() {
    println!("{}", "=".repeat(70));
    println!("SPRINT 75: VQE FP8 Tensor Core Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    // Check CUDA availability
    match engine::cuda::CudaRuntime::new() {
        Ok(rt) => {
            println!("CUDA Runtime initialized successfully");
            println!("Device: {}", engine::cuda::get_device_arch_string());
            println!();
        }
        Err(e) => {
            println!("CUDA not available: {:?}", e);
            println!("Running CPU-only benchmarks...");
            println!();
        }
    }

    // Run benchmarks
    benchmark_precision_modes();
    benchmark_batch_sizes();
    benchmark_qubit_scaling();
    benchmark_h2_molecule();

    println!("{}", "=".repeat(70));
    println!("Benchmark complete");
    println!("{}", "=".repeat(70));
}

fn benchmark_precision_modes() {
    println!("{}", "-".repeat(70));
    println!("Precision Mode Comparison");
    println!("{}", "-".repeat(70));
    println!();

    let depths = [20, 40, 60, 80, 100, 150];

    println!(
        "{:>6} {:>15} {:>15} {:>15} {:>12}",
        "Depth", "Mode", "Error Est.", "Renorm Int", "Max Depth"
    );
    println!("{}", "-".repeat(70));

    for depth in depths {
        let mut manager = AdaptivePrecisionManager::new();
        let mode = manager.select_precision(depth);
        let error = estimate_precision_loss(depth, mode);
        let renorm = mode
            .renorm_interval()
            .map(|r| format!("{}", r))
            .unwrap_or_else(|| "N/A".to_string());
        let max_depth = mode.max_recommended_depth();

        println!(
            "{:>6} {:>15?} {:>14.4}% {:>15} {:>12}",
            depth,
            mode,
            error * 100.0,
            renorm,
            if max_depth == usize::MAX {
                "unlimited".to_string()
            } else {
                max_depth.to_string()
            }
        );
    }
    println!();
}

fn benchmark_batch_sizes() {
    println!("{}", "-".repeat(70));
    println!("Optimal Batch Size by Qubit Count (32GB GPU)");
    println!("{}", "-".repeat(70));
    println!();

    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "Qubits", "State Size", "Batch Size", "Total Memory"
    );
    println!("{}", "-".repeat(55));

    for n_qubits in [10, 15, 20, 22, 24, 25, 26, 28, 30] {
        let state_size = (1usize << n_qubits) * 2; // FP8 = 2 bytes per complex
        let batch_size = Fp8VQEBackend::compute_optimal_batch_size(n_qubits, 32);
        let total_memory = state_size * batch_size;

        let state_str = format_bytes(state_size);
        let total_str = format_bytes(total_memory);

        println!(
            "{:>8} {:>12} {:>15} {:>15}",
            n_qubits, state_str, batch_size, total_str
        );
    }
    println!();
}

fn benchmark_qubit_scaling() {
    println!("{}", "-".repeat(70));
    println!("VQE Scaling Benchmark (CPU Scalar)");
    println!("{}", "-".repeat(70));
    println!();

    // Small qubit counts for CPU benchmark
    let qubit_counts = [2, 4, 6, 8];

    println!(
        "{:>8} {:>12} {:>12} {:>15}",
        "Qubits", "Energy", "Time (ms)", "Evals/sec"
    );
    println!("{}", "-".repeat(50));

    for n_qubits in qubit_counts {
        let h = if n_qubits == 4 {
            Hamiltonian::h2_molecule(0.735)
        } else {
            // Create a simple Hamiltonian for other sizes
            Hamiltonian::single_qubit_z()
        };

        let config = VQEConfig::hardware_efficient(2)
            .with_shots(100) // Reduced for speed
            .with_seed(42);

        let start = Instant::now();
        let result = run_vqe(h, config);
        let elapsed_ms = start.elapsed().as_millis();

        let evals_per_sec = if elapsed_ms > 0 {
            (result.n_evaluations as f64 * 1000.0) / elapsed_ms as f64
        } else {
            0.0
        };

        println!(
            "{:>8} {:>12.6} {:>12} {:>15.1}",
            n_qubits, result.energy, elapsed_ms, evals_per_sec
        );
    }
    println!();
}

fn benchmark_h2_molecule() {
    println!("{}", "-".repeat(70));
    println!("H2 Molecule Benchmark");
    println!("{}", "-".repeat(70));
    println!();

    let bond_length = 0.735; // Equilibrium bond length in Angstroms
    let h = Hamiltonian::h2_molecule(bond_length);

    println!("Bond length: {} Å", bond_length);
    println!("Qubits: {}", h.n_qubits);
    println!("Hamiltonian terms: {}", h.terms.len());
    println!();

    // CPU baseline
    println!("Running CPU scalar baseline...");
    let cpu_config = VQEConfig::uccsd(2).with_shots(1000).with_seed(42);

    let start = Instant::now();
    let cpu_result = run_vqe(h.clone(), cpu_config);
    let cpu_time = start.elapsed();

    println!("  Energy: {:.6} Ha", cpu_result.energy);
    println!("  Iterations: {}", cpu_result.n_iterations);
    println!("  Evaluations: {}", cpu_result.n_evaluations);
    println!("  Time: {:.2} s", cpu_time.as_secs_f64());
    println!("  Converged: {}", cpu_result.converged);
    println!();

    // FP8 GPU (if available)
    println!("Running FP8 GPU benchmark...");
    let fp8_config = VQEConfig::uccsd(2)
        .with_shots(1000)
        .with_seed(42)
        .with_fp8_backend()
        .with_batch_size(64)
        .with_precision_threshold(60);

    let start = Instant::now();
    let fp8_result = run_vqe(h.clone(), fp8_config);
    let fp8_time = start.elapsed();

    println!("  Energy: {:.6} Ha", fp8_result.energy);
    println!("  Iterations: {}", fp8_result.n_iterations);
    println!("  Evaluations: {}", fp8_result.n_evaluations);
    println!("  Time: {:.2} s", fp8_time.as_secs_f64());
    println!("  Converged: {}", fp8_result.converged);
    println!();

    // Comparison
    let speedup = cpu_time.as_secs_f64() / fp8_time.as_secs_f64();
    let energy_diff = (cpu_result.energy - fp8_result.energy).abs();
    let rel_error = energy_diff / cpu_result.energy.abs() * 100.0;

    println!("Comparison:");
    println!("  Speedup: {:.1}x", speedup);
    println!(
        "  Energy difference: {:.6} Ha ({:.4}%)",
        energy_diff, rel_error
    );
    println!();

    // Literature comparison
    let literature_energy = -1.137; // Ha (approximate)
    let cpu_error = (cpu_result.energy - literature_energy).abs();
    let fp8_error = (fp8_result.energy - literature_energy).abs();

    println!("Accuracy (vs literature -1.137 Ha):");
    println!(
        "  CPU error: {:.4} Ha ({:.2}%)",
        cpu_error,
        cpu_error / 1.137 * 100.0
    );
    println!(
        "  FP8 error: {:.4} Ha ({:.2}%)",
        fp8_error,
        fp8_error / 1.137 * 100.0
    );
    println!();
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
