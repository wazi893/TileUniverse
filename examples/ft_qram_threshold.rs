//! Fault-Tolerant QRAM Threshold Analysis
//!
//! Sprint 47.0 Phase 4: Characterizes FT-QRAM performance vs noise.
//!
//! This benchmark sweeps physical error rates to identify the threshold
//! where error correction breaks down.
//!
//! Run with:
//! ```
//! cargo run --release --example ft_qram_threshold
//! ```

use engine::qram::{FTQRAMConfig, FaultTolerantQRAM, threshold_analysis};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      Fault-Tolerant QRAM Threshold Analysis (Sprint 47.0)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // System overview
    println!("System Configuration:");
    println!("  Memory encoding: Steane [[7,1,3]] code");
    println!("  Router encoding: Logical Fredkin (21 qubits per router)");
    println!("  Error correction: Syndrome-based, periodic");
    println!();

    // Qubit scaling
    println!("Qubit Scaling:");
    println!("┌───────┬────────┬──────────┬─────────────┬──────────────┐");
    println!("│ Depth │ Cells  │ Routers  │ Qubits/Cell │ Total Qubits │");
    println!("├───────┼────────┼──────────┼─────────────┼──────────────┤");
    for depth in 1..=4 {
        let qram = FaultTolerantQRAM::with_depth(depth);
        println!(
            "│ {:>5} │ {:>6} │ {:>8} │ {:>11} │ {:>12} │",
            depth,
            qram.size(),
            qram.router_count(),
            7,
            qram.total_qubits()
        );
    }
    println!("└───────┴────────┴──────────┴─────────────┴──────────────┘");
    println!();

    // Threshold analysis
    println!("Threshold Analysis:");
    println!("  Running {} queries per error rate...", 100);
    println!();

    let error_rates = vec![
        0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
    ];

    println!("┌───────────┬───────────────┬───────────────┬───────────────┐");
    println!("│ Error Rate│ Depth 1 (%)   │ Depth 2 (%)   │ Depth 3 (%)   │");
    println!("├───────────┼───────────────┼───────────────┼───────────────┤");

    for &p in &error_rates {
        let r1 = run_threshold_test(1, p, 100, 42);
        let r2 = run_threshold_test(2, p, 100, 42);
        let r3 = run_threshold_test(3, p, 100, 42);

        println!(
            "│ {:>9.4} │ {:>13.1} │ {:>13.1} │ {:>13.1} │",
            p, r1, r2, r3
        );
    }

    println!("└───────────┴───────────────┴───────────────┴───────────────┘");
    println!();

    // Detailed depth-2 analysis
    println!("Detailed Analysis (Depth 2, 1000 queries):");
    detailed_analysis(2, 1000);
    println!();

    // Performance metrics
    println!("Performance Metrics:");
    performance_benchmark();
    println!();

    // Threshold identification
    println!("Threshold Estimation:");
    estimate_threshold();

    println!();
    println!("Benchmark complete.");
}

fn run_threshold_test(depth: usize, error_rate: f64, queries: usize, seed: u64) -> f64 {
    let results = threshold_analysis(depth, &[error_rate], queries, seed);
    results[0].1
}

fn detailed_analysis(depth: usize, queries: usize) {
    let error_rates = vec![0.001, 0.005, 0.01, 0.02];

    println!("┌───────────┬─────────┬────────────┬────────────┬────────────┐");
    println!("│ Error Rate│ Success │ Err Flags  │ Corrections│ Gates/Query│");
    println!("├───────────┼─────────┼────────────┼────────────┼────────────┤");

    for p in error_rates {
        let config = FTQRAMConfig::new(depth)
            .with_error_rate(p)
            .with_correction_interval(1)
            .with_seed(12345);

        let mut qram = FaultTolerantQRAM::new(config);
        let data: Vec<u64> = (0..qram.size()).map(|i| i as u64 * 100).collect();
        qram.load(&data);

        for i in 0..queries {
            qram.query(i % qram.size());
        }

        let stats = qram.stats();
        let gates_per_query = if stats.queries > 0 {
            stats.total_gates as f64 / stats.queries as f64
        } else {
            0.0
        };

        println!(
            "│ {:>9.4} │ {:>6.1}% │ {:>10} │ {:>10} │ {:>10.1} │",
            p,
            stats.success_rate(),
            stats.error_flags,
            stats.correction_cycles,
            gates_per_query
        );
    }

    println!("└───────────┴─────────┴────────────┴────────────┴────────────┘");
}

fn performance_benchmark() {
    let depths = [1, 2, 3];
    let queries = 100;

    println!("┌───────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Depth │ Time (µs)    │ Queries/sec  │ Gates/sec    │");
    println!("├───────┼──────────────┼──────────────┼──────────────┤");

    for depth in depths {
        let config = FTQRAMConfig::new(depth).with_error_rate(0.001);
        let mut qram = FaultTolerantQRAM::new(config);
        let data: Vec<u64> = (0..qram.size()).map(|i| i as u64).collect();
        qram.load(&data);

        let start = Instant::now();
        for i in 0..queries {
            qram.query(i % qram.size());
        }
        let elapsed = start.elapsed();

        let time_us = elapsed.as_micros();
        let qps = if time_us > 0 {
            queries as f64 * 1_000_000.0 / time_us as f64
        } else {
            0.0
        };
        let gps = qram.stats().total_gates as f64 * 1_000_000.0 / time_us.max(1) as f64;

        println!(
            "│ {:>5} │ {:>12} │ {:>12.0} │ {:>12.0} │",
            depth, time_us, qps, gps
        );
    }

    println!("└───────┴──────────────┴──────────────┴──────────────┘");
}

fn estimate_threshold() {
    // Run fine-grained sweep to find threshold
    let error_rates: Vec<f64> = (0..20).map(|i| 0.001 * (i + 1) as f64).collect();
    let queries = 200;

    println!(
        "Fine-grained sweep (depth 2, {} queries per point):",
        queries
    );
    println!();

    let mut prev_rate = 100.0;
    let mut threshold_estimate = 0.0;

    for &p in &error_rates {
        let success = run_threshold_test(2, p, queries, 99);

        let marker = if success < 90.0 && prev_rate >= 90.0 {
            threshold_estimate = p;
            " <-- threshold region"
        } else {
            ""
        };

        let bar_len = (success / 5.0) as usize;
        let bar: String = "█".repeat(bar_len);

        println!("  p={:.4}: {:>5.1}% {}{}", p, success, bar, marker);

        prev_rate = success;
    }

    println!();
    if threshold_estimate > 0.0 {
        println!(
            "Estimated threshold: ~{:.4} (success drops below 90%)",
            threshold_estimate
        );
    } else {
        println!("Threshold not reached in tested range.");
    }

    println!();
    println!("Note: Steane code theoretical threshold ~1-3% for CSS codes.");
    println!("      Actual threshold depends on gate implementation and");
    println!("      correction strategy.");
}
