//! Stabilizer Neuron Maximum Scale Benchmark
//!
//! Benchmarks stabilizer quantum neurons at practical scales.
//!
//! Run with: cargo run --release --example stabilizer_max_bench

use engine::snn::{StabilizerNetworkConfig, StabilizerNeuronNetwork};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(70));
    println!("  Stabilizer Neuron Maximum Scale Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    println!("Memory scaling (O(n^2) bits for stabilizer tableau):");
    println!("  10K neurons  -> 12.5 MB");
    println!("  100K neurons -> 1.25 GB");
    println!();

    bench_network_creation();
    bench_decision_throughput();
    bench_peak_performance();

    println!("{}", "=".repeat(70));
    println!("  Benchmark Complete!");
    println!("{}", "=".repeat(70));
}

fn bench_network_creation() {
    println!("{}", "-".repeat(70));
    println!("Benchmark 1: Network Creation (scales to 100K+)");
    println!("{}", "-".repeat(70));

    let scales = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000];

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Neurons", "Memory", "Time", "Synapses", "Rate"
    );
    println!("{}", "-".repeat(60));

    for &n in &scales {
        let config = StabilizerNetworkConfig {
            n_inputs: 10,
            hidden_layers: vec![n - 20],
            n_outputs: 10,
            connectivity: 0.001,
            ..Default::default()
        };

        let start = Instant::now();
        let network = StabilizerNeuronNetwork::new(config, 42);
        let elapsed = start.elapsed();

        let mem_mb = (n * n) as f64 / 8.0 / 1_000_000.0;
        let rate = n as f64 / elapsed.as_secs_f64();

        println!(
            "{:>10} {:>10.1} MB {:>12.2?} {:>12} {:>10.0}/s",
            n,
            mem_mb,
            elapsed,
            network.num_synapses(),
            rate
        );
    }
    println!();
}

fn bench_decision_throughput() {
    println!("{}", "-".repeat(70));
    println!("Benchmark 2: Decision Throughput");
    println!("{}", "-".repeat(70));

    // Practical sizes - measurement is O(n) so keep networks small
    let configs = [
        (16, 100_000, "16 neurons"),
        (32, 50_000, "32 neurons"),
        (64, 20_000, "64 neurons"),
        (128, 5_000, "128 neurons"),
        (256, 1_000, "256 neurons"),
    ];

    println!(
        "{:>15} {:>12} {:>12} {:>15}",
        "Network", "Decisions", "Time", "Decisions/sec"
    );
    println!("{}", "-".repeat(56));

    for (n, iters, desc) in configs {
        let config = StabilizerNetworkConfig {
            n_inputs: n / 4,
            hidden_layers: vec![n / 2],
            n_outputs: n / 4,
            connectivity: 0.5,
            ..Default::default()
        };

        let mut network = StabilizerNeuronNetwork::new(config, 42);
        let inputs: Vec<u8> = (0..n / 4).map(|i| ((i * 50) % 256) as u8).collect();

        // Warmup
        for _ in 0..50 {
            let _ = network.decide(&inputs);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = network.decide(&inputs);
        }
        let elapsed = start.elapsed();

        let rate = iters as f64 / elapsed.as_secs_f64();

        println!(
            "{:>15} {:>12} {:>12.2?} {:>15.0}",
            desc, iters, elapsed, rate
        );
    }
    println!();
}

fn bench_peak_performance() {
    println!("{}", "-".repeat(70));
    println!("Benchmark 3: Peak Performance");
    println!("{}", "-".repeat(70));

    // Minimal network for maximum throughput
    let config = StabilizerNetworkConfig {
        n_inputs: 4,
        hidden_layers: vec![8],
        n_outputs: 4,
        connectivity: 0.8,
        ..Default::default()
    };

    let mut network = StabilizerNeuronNetwork::new(config, 42);
    let inputs = [200u8, 150, 100, 50];

    println!(
        "Network: 4 -> 8 -> 4 (16 neurons, {} synapses)",
        network.num_synapses()
    );
    println!();

    // Peak decisions/sec
    let iterations = 500_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = network.decide(&inputs);
    }
    let elapsed = start.elapsed();
    let decisions_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("  Iterations:    {:>12}", iterations);
    println!("  Time:          {:>12.2?}", elapsed);
    println!("  Rate:          {:>12.0} decisions/sec", decisions_per_sec);
    println!();

    // Quantum ops estimate
    let synapses = network.num_synapses();
    let ops_per_decision = 4 + synapses + 4; // H + CZ + measure
    let quantum_ops_sec = ops_per_decision as f64 * decisions_per_sec;

    println!("  Quantum ops/decision: {}", ops_per_decision);
    println!("  Quantum ops/sec:      {:.2e}", quantum_ops_sec);
    println!();
}
