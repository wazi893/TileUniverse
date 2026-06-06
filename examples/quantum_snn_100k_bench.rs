//! Quantum-SNN Hybrid 100K Neuron Benchmark
//!
//! Phase 2: GPU-accelerated hybrid system at scale.
//!
//! Run with: cargo run --release --features cuda --example quantum_snn_100k_bench

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, is_cuda_available};
#[cfg(feature = "cuda")]
use engine::snn::{GpuQuantumSNN, QuantumSNNConfig};
#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("  QUANTUM-SNN HYBRID: 100K NEURON BENCHMARK (Phase 2)");
    println!("================================================================\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This benchmark requires the 'cuda' feature.");
        println!("Run with: cargo run --release --features cuda --example quantum_snn_100k_bench");
        return;
    }

    #[cfg(feature = "cuda")]
    run_benchmark();
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
    // Check CUDA availability
    if !is_cuda_available() {
        println!("ERROR: No CUDA GPU available.");
        return;
    }

    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            println!("ERROR: Failed to initialize CUDA: {:?}", e);
            return;
        }
    };

    println!("CUDA runtime initialized.\n");

    // Test configurations: (name, n_inputs, hidden, n_outputs, ticks_per_decision)
    let configs = [
        ("Small (1K)", 100, vec![850], 10, 20),
        ("Medium (10K)", 500, vec![9000], 10, 20),
        ("Large (50K)", 1000, vec![48000], 10, 20),
        ("XLarge (100K)", 2000, vec![97000], 10, 20),
        ("XXLarge (200K)", 4000, vec![194000], 10, 20),
    ];

    println!(
        "{:<18} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Config", "Neurons", "Synapses", "Decision(ms)", "Throughput", "Q-Blocks"
    );
    println!("{:-<84}", "");

    for (name, n_inputs, hidden, n_outputs, ticks) in configs {
        let config = QuantumSNNConfig {
            n_inputs,
            hidden_layers: hidden.clone(),
            n_outputs,
            interference_depth: 1,
            ticks_per_decision: ticks,
            learning_enabled: false,
            ..Default::default()
        };

        match benchmark_config(&rt, name, config) {
            Ok((neurons, synapses, decision_ms, throughput, q_blocks)) => {
                println!(
                    "{:<18} {:>12} {:>12} {:>12.2} {:>10.0} /s {:>10}",
                    name, neurons, synapses, decision_ms, throughput, q_blocks
                );
            }
            Err(e) => {
                println!("{:<18} ERROR: {:?}", name, e);
            }
        }
    }

    println!("\n--- Quantum vs Classical Comparison at 100K Scale ---\n");
    compare_quantum_classical(&rt);

    println!("\n--- Sparse Quantum Efficiency at Scale ---\n");
    sparse_efficiency_test(&rt);

    println!("\n================================================================");
    println!("  BENCHMARK COMPLETE");
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn benchmark_config(
    rt: &CudaRuntime,
    _name: &str,
    config: QuantumSNNConfig,
) -> Result<(usize, usize, f64, f64, usize), engine::cuda::CudaError> {
    let n_inputs = config.n_inputs;

    // Create GPU hybrid
    let mut hybrid = GpuQuantumSNN::new(rt, config)?;
    let (n_neurons, n_synapses, _) = hybrid.network_info();

    // Set random inputs
    let inputs: Vec<u8> = (0..n_inputs)
        .map(|i| ((i * 17 + 123) % 256) as u8)
        .collect();
    hybrid.set_inputs(rt, &inputs)?;

    // Warmup
    for _ in 0..3 {
        hybrid.decide(rt)?;
    }

    // Timed runs
    let n_decisions = 20;
    let start = Instant::now();
    for _ in 0..n_decisions {
        hybrid.decide(rt)?;
    }
    let elapsed = start.elapsed();

    let decision_ms = elapsed.as_secs_f64() * 1000.0 / n_decisions as f64;
    let throughput = 1000.0 / decision_ms;
    let q_blocks = hybrid.quantum_blocks();

    Ok((n_neurons, n_synapses, decision_ms, throughput, q_blocks))
}

#[cfg(feature = "cuda")]
fn compare_quantum_classical(rt: &CudaRuntime) {
    use std::collections::HashMap;

    let config = QuantumSNNConfig {
        n_inputs: 2000,
        hidden_layers: vec![97000],
        n_outputs: 10,
        interference_depth: 1,
        ticks_per_decision: 20,
        learning_enabled: false,
        ..Default::default()
    };

    let mut hybrid = match GpuQuantumSNN::new(rt, config.clone()) {
        Ok(h) => h,
        Err(e) => {
            println!("Failed to create hybrid: {:?}", e);
            return;
        }
    };

    let inputs: Vec<u8> = (0..2000).map(|i| ((i * 7) % 256) as u8).collect();
    if hybrid.set_inputs(rt, &inputs).is_err() {
        return;
    }

    let n_trials = 50;
    let mut quantum_counts: HashMap<usize, usize> = HashMap::new();
    let mut classical_counts: HashMap<usize, usize> = HashMap::new();

    println!("Running {} decisions at 100K neuron scale...\n", n_trials);

    for _ in 0..n_trials {
        if let Ok(q_action) = hybrid.decide(rt) {
            *quantum_counts.entry(q_action).or_insert(0) += 1;
        }
        if let Ok(c_action) = hybrid.classical_action(rt) {
            *classical_counts.entry(c_action).or_insert(0) += 1;
        }
    }

    println!("Quantum decision distribution (100K neurons):");
    for action in 0..10 {
        let count = quantum_counts.get(&action).unwrap_or(&0);
        let pct = (*count as f64 / n_trials as f64) * 100.0;
        println!("  Action {}: {:>3} times ({:>5.1}%)", action, count, pct);
    }

    println!("\nClassical WTA distribution (100K neurons):");
    for action in 0..10 {
        let count = classical_counts.get(&action).unwrap_or(&0);
        let pct = (*count as f64 / n_trials as f64) * 100.0;
        println!("  Action {}: {:>3} times ({:>5.1}%)", action, count, pct);
    }

    // Calculate entropy
    let q_entropy = entropy(&quantum_counts, n_trials);
    let c_entropy = entropy(&classical_counts, n_trials);

    println!("\nDecision entropy:");
    println!("  Quantum:   {:.3} bits (exploration)", q_entropy);
    println!("  Classical: {:.3} bits (exploitation)", c_entropy);
}

#[cfg(feature = "cuda")]
fn sparse_efficiency_test(rt: &CudaRuntime) {
    println!(
        "{:<12} {:>15} {:>15} {:>12}",
        "Outputs", "Dense Amps", "Quantum Blocks", "Ratio"
    );
    println!("{:-<58}", "");

    // Only test up to 16 outputs - beyond that, interference creates too many amplitudes
    for n_outputs in [10, 16] {
        let config = QuantumSNNConfig {
            n_inputs: 1000,
            hidden_layers: vec![50000],
            n_outputs,
            interference_depth: 1,
            ticks_per_decision: 10,
            learning_enabled: false,
            ..Default::default()
        };

        if let Ok(mut hybrid) = GpuQuantumSNN::new(rt, config.clone()) {
            let inputs: Vec<u8> = (0..1000).map(|i| ((i * 3) % 256) as u8).collect();
            if hybrid.set_inputs(rt, &inputs).is_ok() {
                if hybrid.decide(rt).is_ok() {
                    let dense_amps = 1usize << n_outputs.min(30); // Cap at 2^30 for display
                    let q_blocks = hybrid.quantum_blocks();
                    let sparse_amps = q_blocks * 128;
                    let ratio = if sparse_amps > 0 {
                        dense_amps as f64 / sparse_amps as f64
                    } else {
                        0.0
                    };

                    let dense_str = if n_outputs <= 30 {
                        format!("{}", dense_amps)
                    } else {
                        format!("2^{}", n_outputs)
                    };

                    println!(
                        "{:<12} {:>15} {:>15} {:>11.0}x",
                        n_outputs,
                        dense_str,
                        q_blocks,
                        ratio.max(1.0)
                    );
                }
            }
        }
    }
}

fn entropy(counts: &std::collections::HashMap<usize, usize>, total: usize) -> f64 {
    let mut h = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total as f64;
            h -= p * p.log2();
        }
    }
    h
}
