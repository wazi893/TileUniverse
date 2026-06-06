//! Quantum-SNN Hybrid Demo
//!
//! Demonstrates the combination of classical Spiking Neural Networks with
//! sparse quantum simulation for decision making.
//!
//! Run with: cargo run --release --example quantum_snn_demo

use engine::snn::{QuantumSNN, QuantumSNNConfig};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("==========================================================");
    println!("  QUANTUM-SNN HYBRID PROOF OF CONCEPT");
    println!("  Phase 6: Combining Sparse Quantum with Neuromorphic SNN");
    println!("==========================================================\n");

    // Demo 1: Basic operation
    demo_basic_operation();

    // Demo 2: Quantum vs Classical comparison
    demo_quantum_vs_classical();

    // Demo 3: XOR task learning
    demo_xor_task();

    // Demo 4: Sparse efficiency
    demo_sparse_efficiency();

    // Demo 5: Scaling test
    demo_scaling();

    println!("\n==========================================================");
    println!("  DEMO COMPLETE");
    println!("==========================================================");
}

/// Demo 1: Basic operation of the hybrid system
fn demo_basic_operation() {
    println!("--- Demo 1: Basic Operation ---\n");

    let config = QuantumSNNConfig::small();
    let mut qsnn = QuantumSNN::new(config);

    println!("Created Quantum-SNN hybrid:");
    println!("  - 16 input neurons");
    println!("  - 32 hidden neurons");
    println!("  - 4 output qubits");
    println!();

    // Set input pattern
    let inputs: Vec<u8> = vec![
        255, 128, 64, 32, 200, 150, 100, 50, 25, 12, 6, 3, 180, 90, 45, 22,
    ];
    qsnn.set_inputs(&inputs);

    // Make a decision
    let action = qsnn.decide();
    let probs = qsnn.last_probabilities();

    println!("Decision made:");
    println!("  - Action selected: {}", action);
    println!("  - Probability distribution: {:?}", probs);
    println!(
        "  - Classical WTA would choose: {}",
        qsnn.classical_action()
    );
    println!("  - Output spike counts: {:?}", qsnn.output_spike_counts());
    println!("  - Active quantum blocks: {}", qsnn.quantum_blocks());
    println!();
}

/// Demo 2: Compare quantum probabilistic vs classical deterministic decisions
fn demo_quantum_vs_classical() {
    println!("--- Demo 2: Quantum vs Classical Decision Making ---\n");

    let config = QuantumSNNConfig {
        n_inputs: 16,
        hidden_layers: vec![32],
        n_outputs: 5,
        interference_depth: 1,
        ticks_per_decision: 15,
        learning_enabled: false,
        ..Default::default()
    };
    let mut qsnn = QuantumSNN::new(config);

    // Test pattern that should cause some uncertainty
    let inputs: Vec<u8> = vec![
        200, 200, 180, 180, 160, 160, 140, 140, 120, 120, 100, 100, 80, 80, 60, 60,
    ];
    qsnn.set_inputs(&inputs);

    let n_trials = 50;
    let mut quantum_counts: HashMap<usize, usize> = HashMap::new();
    let mut classical_counts: HashMap<usize, usize> = HashMap::new();

    println!("Running {} decision trials...\n", n_trials);

    for _ in 0..n_trials {
        let q_action = qsnn.decide();
        let c_action = qsnn.classical_action();

        *quantum_counts.entry(q_action).or_insert(0) += 1;
        *classical_counts.entry(c_action).or_insert(0) += 1;
    }

    println!("Quantum decision distribution:");
    for action in 0..5 {
        let count = quantum_counts.get(&action).unwrap_or(&0);
        let pct = (*count as f64 / n_trials as f64) * 100.0;
        println!("  Action {}: {:>3} times ({:>5.1}%)", action, count, pct);
    }

    println!("\nClassical WTA distribution:");
    for action in 0..5 {
        let count = classical_counts.get(&action).unwrap_or(&0);
        let pct = (*count as f64 / n_trials as f64) * 100.0;
        println!("  Action {}: {:>3} times ({:>5.1}%)", action, count, pct);
    }

    let q_entropy = entropy(&quantum_counts, n_trials);
    let c_entropy = entropy(&classical_counts, n_trials);

    println!("\nDecision entropy (higher = more exploration):");
    println!("  Quantum: {:.3} bits", q_entropy);
    println!("  Classical: {:.3} bits", c_entropy);
    println!();
}

/// Demo 3: XOR task with learning
fn demo_xor_task() {
    println!("--- Demo 3: XOR Task Learning ---\n");

    let config = QuantumSNNConfig::xor();
    let mut qsnn = QuantumSNN::new(config);

    // XOR truth table: (A, B) -> A XOR B
    let xor_patterns = [
        ([0u8, 0], 0),   // 0 XOR 0 = 0
        ([255, 0], 1),   // 1 XOR 0 = 1
        ([0, 255], 1),   // 0 XOR 1 = 1
        ([255, 255], 0), // 1 XOR 1 = 0
    ];

    println!("Training on XOR patterns with R-STDP...\n");

    let n_epochs = 100; // More epochs for better learning
    let trials_per_epoch = 20;

    for epoch in 0..n_epochs {
        let mut correct = 0;

        for _ in 0..trials_per_epoch {
            for (inputs, expected) in &xor_patterns {
                // Encode inputs (spread across input neurons)
                let mut full_inputs = vec![0u8; 4];
                full_inputs[0] = inputs[0];
                full_inputs[1] = inputs[1];
                full_inputs[2] = if inputs[0] > 0 { 128 } else { 0 };
                full_inputs[3] = if inputs[1] > 0 { 128 } else { 0 };

                qsnn.set_inputs(&full_inputs);
                let action = qsnn.decide();

                // Reward correct, punish incorrect
                if action == *expected {
                    qsnn.learn(100); // Strong positive reward
                    correct += 1;
                } else {
                    qsnn.learn(-80); // Moderate negative reward
                }
            }
        }

        let accuracy = (correct as f64 / (trials_per_epoch * 4) as f64) * 100.0;
        if epoch % 20 == 0 || epoch == n_epochs - 1 {
            println!("Epoch {:>3}: Accuracy = {:.1}%", epoch, accuracy);
        }
    }

    // Final evaluation
    println!("\nFinal XOR evaluation:");
    for (inputs, expected) in &xor_patterns {
        let mut full_inputs = vec![0u8; 4];
        full_inputs[0] = inputs[0];
        full_inputs[1] = inputs[1];

        qsnn.set_inputs(&full_inputs);
        let action = qsnn.decide();
        let probs = qsnn.last_probabilities();

        let a = if inputs[0] > 0 { 1 } else { 0 };
        let b = if inputs[1] > 0 { 1 } else { 0 };
        println!(
            "  {} XOR {} = {} (expected {}, probs: [{:.2}, {:.2}])",
            a, b, action, expected, probs[0], probs[1]
        );
    }
    println!();
}

/// Demo 4: Sparse efficiency of quantum layer
fn demo_sparse_efficiency() {
    println!("--- Demo 4: Sparse Quantum Efficiency ---\n");

    println!(
        "{:>10} {:>12} {:>12} {:>15} {:>12}",
        "Outputs", "Dense (2^n)", "Sparse", "Ratio", "Memory"
    );
    println!("{:-<65}", "");

    for n_outputs in [4, 8, 10, 12, 16, 20] {
        let config = QuantumSNNConfig {
            n_inputs: n_outputs * 4,
            hidden_layers: vec![n_outputs * 8],
            n_outputs,
            interference_depth: 1,
            ticks_per_decision: 10,
            learning_enabled: false,
            ..Default::default()
        };

        let mut qsnn = QuantumSNN::new(config);

        // Random inputs
        let inputs: Vec<u8> = (0..(n_outputs * 4))
            .map(|i| ((i * 17 + 123) % 256) as u8)
            .collect();
        qsnn.set_inputs(&inputs);
        qsnn.decide();

        let dense_amps = 1usize << n_outputs;
        let sparse_blocks = qsnn.quantum_blocks();
        let sparse_amps = sparse_blocks * 128; // 128 amplitudes per block
        let ratio = dense_amps as f64 / sparse_amps.max(1) as f64;
        let memory_kb = (sparse_blocks * 2048) as f64 / 1024.0;

        println!(
            "{:>10} {:>12} {:>12} {:>14.1}x {:>10.1} KB",
            n_outputs, dense_amps, sparse_amps, ratio, memory_kb
        );
    }
    println!();
}

/// Demo 5: Scaling test
fn demo_scaling() {
    println!("--- Demo 5: Scaling Test ---\n");

    println!(
        "{:>12} {:>12} {:>12} {:>15} {:>12}",
        "Neurons", "Outputs", "Blocks", "Decision (us)", "Throughput"
    );
    println!("{:-<70}", "");

    for (n_hidden, n_outputs) in [(100, 10), (500, 20), (1000, 10), (2000, 10)] {
        let config = QuantumSNNConfig {
            n_inputs: 64,
            hidden_layers: vec![n_hidden],
            n_outputs,
            interference_depth: 1,
            ticks_per_decision: 20,
            learning_enabled: false,
            ..Default::default()
        };

        let mut qsnn = QuantumSNN::new(config);

        let inputs: Vec<u8> = (0..64).map(|i| ((i * 4) % 256) as u8).collect();
        qsnn.set_inputs(&inputs);

        // Warmup
        qsnn.decide();

        // Timed runs
        let n_runs = 10;
        let start = Instant::now();
        for _ in 0..n_runs {
            qsnn.decide();
        }
        let elapsed = start.elapsed();
        let per_decision_us = elapsed.as_micros() as f64 / n_runs as f64;
        let throughput = 1_000_000.0 / per_decision_us;

        let total_neurons = 64 + n_hidden + n_outputs;
        let blocks = qsnn.quantum_blocks();

        println!(
            "{:>12} {:>12} {:>12} {:>14.1} {:>10.0} /s",
            total_neurons, n_outputs, blocks, per_decision_us, throughput
        );
    }
    println!();
}

/// Calculate entropy of a distribution (in bits)
fn entropy(counts: &HashMap<usize, usize>, total: usize) -> f64 {
    let mut h = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total as f64;
            h -= p * p.log2();
        }
    }
    h
}
