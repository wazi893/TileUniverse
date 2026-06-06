//! Sparse Noise Model Benchmark
//!
//! Demonstrates the sparse noise model's efficiency for quantum error simulation.
//!
//! Run: cargo run --release --example noise_benchmark

use engine::tile8::sparse_noise::SparseNoisyState;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         SPARSE NOISE MODEL BENCHMARK                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test various qubit counts and error rates
    let qubit_counts = [10, 20, 30, 50];
    let error_rates = [0.001, 0.01, 0.05];

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("NOISY GHZ STATE PREPARATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>12}",
        "Qubits", "Err Rate", "Branches", "Fidelity", "Time (ms)", "Memory (KB)"
    );
    println!("{}", "-".repeat(75));

    for &n in &qubit_counts {
        for &p in &error_rates {
            let start = Instant::now();

            let mut noisy = SparseNoisyState::new(n);
            noisy.set_prob_threshold(1e-10);
            noisy.set_max_branches(10000);

            // Prepare noisy GHZ state
            noisy.prepare_ghz_noisy(p);

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let fidelity = noisy.verify_ghz_fidelity();
            let memory_kb = noisy.memory_bytes() / 1024;

            println!(
                "{:>8} | {:>7.3}% | {:>10} | {:>10.6} | {:>10.2} | {:>12}",
                n,
                p * 100.0,
                noisy.num_branches(),
                fidelity,
                elapsed_ms,
                memory_kb
            );
        }
    }

    // Detailed analysis for a specific case
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DETAILED ANALYSIS: 20 Qubits, 1% Error Rate");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut noisy = SparseNoisyState::new(20);
    noisy.set_prob_threshold(1e-12);
    noisy.set_max_branches(50000);
    noisy.prepare_ghz_noisy(0.01);

    let stats = noisy.stats();
    println!("\nNoise Statistics:");
    println!("  Total branches created:    {}", stats.branches_created);
    println!("  Total branches pruned:     {}", stats.branches_pruned);
    println!("  Total branches merged:     {}", stats.branches_merged);
    println!("  Peak branches:             {}", stats.peak_branches);
    println!("  Noise applications:        {}", stats.noise_applications);
    println!(
        "  Probability lost to prune: {:.2e}",
        stats.probability_pruned
    );

    println!("\nState Analysis:");
    println!("  Current branches:          {}", noisy.num_branches());
    println!(
        "  Total probability:         {:.10}",
        noisy.total_probability()
    );
    println!(
        "  Fidelity (no error):       {:.10}",
        noisy.fidelity_no_error()
    );
    println!(
        "  Average error weight:      {:.4}",
        noisy.average_error_weight()
    );
    println!(
        "  Memory usage:              {} KB",
        noisy.memory_bytes() / 1024
    );

    // Error distribution
    println!("\nError Distribution (top 10):");
    let mut error_dist = noisy.error_distribution();
    error_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (error, prob)) in error_dist.iter().take(10).enumerate() {
        let error_str = if error.is_identity() {
            "Identity (no error)".to_string()
        } else {
            let terms: Vec<String> = error
                .iter()
                .map(|(&q, &p)| format!("{:?}_{}", p, q))
                .collect();
            terms.join(" ⊗ ")
        };
        println!("  {:>2}. {:40} : {:.6}", i + 1, error_str, prob);
    }

    // Theoretical comparison
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("THEORETICAL COMPARISON: Sparse vs Dense");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:>8} | {:>16} | {:>16} | {:>12}",
        "Qubits", "Dense Memory", "Sparse Memory", "Compression"
    );
    println!("{}", "-".repeat(60));

    for &n in &[10, 20, 30, 50, 100] {
        // Dense: 2^n * 16 bytes (complex128) per error branch, 4^n branches
        let dense_per_branch = 1u128 << n; // 2^n amplitudes
        let dense_branches = 4u128.pow(n as u32); // 4^n error patterns (very rough)
        let dense_total_bytes = dense_per_branch
            .saturating_mul(dense_branches)
            .saturating_mul(16);

        // Sparse: estimate based on actual simulation
        let mut noisy = SparseNoisyState::new(n);
        noisy.set_prob_threshold(1e-8);
        noisy.set_max_branches(10000);
        noisy.prepare_ghz_noisy(0.01);
        let sparse_bytes = noisy.memory_bytes() as u128;

        let compression = if sparse_bytes > 0 && dense_total_bytes < u128::MAX / 100 {
            dense_total_bytes as f64 / sparse_bytes as f64
        } else {
            f64::INFINITY
        };

        let dense_str = format_bytes(dense_total_bytes);
        let sparse_str = format!("{} KB", sparse_bytes / 1024);
        let comp_str = if compression.is_infinite() {
            "∞".to_string()
        } else if compression > 1e12 {
            format!("{:.1e}x", compression)
        } else {
            format!("{:.0}x", compression)
        };

        println!(
            "{:>8} | {:>16} | {:>16} | {:>12}",
            n, dense_str, sparse_str, comp_str
        );
    }

    // QEC syndrome extraction demo
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("QEC SYNDROME DISTRIBUTION (10 qubits, 5% error rate)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut noisy = SparseNoisyState::new(10);
    noisy.set_prob_threshold(1e-10);
    noisy.prepare_ghz_noisy(0.05);

    let syndromes = noisy.syndrome_distribution();
    let mut syndrome_list: Vec<(Vec<bool>, f64)> = syndromes.into_iter().collect();
    syndrome_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top syndromes (qubit error pattern → probability):");
    for (syndrome, prob) in syndrome_list.iter().take(10) {
        let weight: usize = syndrome.iter().filter(|&&b| b).count();
        let error_qubits: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|&(_, b)| *b)
            .map(|(i, _)| i)
            .collect();
        let error_str = if error_qubits.is_empty() {
            "None".to_string()
        } else {
            format!("{:?}", error_qubits)
        };
        println!(
            "  Weight {} | Qubits {:20} : {:.6}",
            weight, error_str, prob
        );
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ The sparse noise model efficiently tracks quantum errors by:     ║");
    println!("║                                                                  ║");
    println!("║ • Exploiting Pauli error sparsity preservation                   ║");
    println!("║ • Using adaptive pruning to manage branch explosion              ║");
    println!("║ • Maintaining O(k) memory per branch (k = non-zero amplitudes)   ║");
    println!("║                                                                  ║");
    println!("║ This enables noisy QEC simulation at scales impossible with      ║");
    println!("║ traditional density matrix methods.                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn format_bytes(bytes: u128) -> String {
    if bytes >= 1 << 60 {
        format!("{:.1} EB", bytes as f64 / (1u128 << 60) as f64)
    } else if bytes >= 1 << 50 {
        format!("{:.1} PB", bytes as f64 / (1u128 << 50) as f64)
    } else if bytes >= 1 << 40 {
        format!("{:.1} TB", bytes as f64 / (1u128 << 40) as f64)
    } else if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u128 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u128 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1u128 << 10) as f64)
    } else {
        format!("{} B", bytes)
    }
}
