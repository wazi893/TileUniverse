//! Trace the ACTUAL quantum interference in QuantumSNN
//!
//! Run with: cargo run --release --example actual_quantum_trace

use engine::snn::{QuantumSNN, QuantumSNNConfig};

fn main() {
    println!("================================================================");
    println!("  ACTUAL QUANTUM INTERFERENCE TRACE");
    println!("  What does QuantumSNN REALLY do?");
    println!("================================================================\n");

    // Create a minimal QuantumSNN to trace its behavior
    let config = QuantumSNNConfig {
        n_inputs: 8,
        hidden_layers: vec![16],
        n_outputs: 5,
        interference_depth: 2, // Same as A/B test
        ticks_per_decision: 10,
        learning_enabled: false,
        ..Default::default()
    };

    // Test with various input patterns
    let test_inputs: Vec<(&str, [u8; 8])> = vec![
        ("All zero", [0, 0, 0, 0, 0, 0, 0, 0]),
        ("Uniform", [100, 100, 100, 100, 100, 100, 100, 100]),
        ("Strong up", [200, 50, 50, 50, 100, 100, 100, 100]),
        ("Strong right", [50, 50, 50, 200, 100, 100, 100, 100]),
        ("Predator close", [255, 0, 0, 0, 128, 128, 128, 128]),
    ];

    for (name, inputs) in &test_inputs {
        println!("=== {} ===", name);
        println!("Inputs: {:?}\n", inputs);

        // Run multiple times to see distribution
        let mut action_counts = [0u32; 5];
        let mut classical_counts = [0u32; 5];
        let n_samples = 100;

        for seed in 0..n_samples {
            let mut qsnn = QuantumSNN::with_seed(config.clone(), seed as u64 * 137);
            qsnn.set_inputs(inputs);

            let q_action = qsnn.decide();
            action_counts[q_action] += 1;

            let c_action = qsnn.classical_action();
            classical_counts[c_action] += 1;
        }

        // Show distributions
        println!("Quantum decisions over {} samples:", n_samples);
        for (i, &count) in action_counts.iter().enumerate() {
            let pct = count as f64 / n_samples as f64 * 100.0;
            let bar = "#".repeat((pct / 5.0) as usize);
            println!("  Action {}: {:>5.1}% {}", i, pct, bar);
        }

        println!("\nClassical WTA over {} samples:", n_samples);
        for (i, &count) in classical_counts.iter().enumerate() {
            let pct = count as f64 / n_samples as f64 * 100.0;
            let bar = "#".repeat((pct / 5.0) as usize);
            println!("  Action {}: {:>5.1}% {}", i, pct, bar);
        }

        // Entropy comparison
        let q_entropy = entropy(&action_counts, n_samples);
        let c_entropy = entropy(&classical_counts, n_samples);
        println!(
            "\nEntropy - Quantum: {:.3}, Classical: {:.3}",
            q_entropy, c_entropy
        );
        println!();
    }

    // Now trace a single decision in detail
    println!("================================================================");
    println!("  SINGLE DECISION TRACE");
    println!("================================================================\n");

    let mut qsnn = QuantumSNN::with_seed(config.clone(), 42);
    qsnn.set_inputs(&[200, 50, 50, 50, 100, 100, 100, 100]);

    println!("Before decide():");
    println!("  SNN tick: {}", qsnn.tick());

    let action = qsnn.decide();

    println!("\nAfter decide():");
    println!("  SNN tick: {}", qsnn.tick());
    println!("  Chosen action: {}", action);
    println!("  Quantum blocks: {}", qsnn.quantum_blocks());

    println!("\nSpike counts from SNN:");
    for (i, &count) in qsnn.output_spike_counts().iter().enumerate() {
        println!("  Output {}: {} spikes", i, count);
    }

    println!("\nProbability distribution after interference:");
    let probs = qsnn.last_probabilities();
    for (i, &p) in probs.iter().enumerate() {
        let bar = "#".repeat((p * 50.0) as usize);
        println!("  Action {}: {:>6.2}% {}", i, p * 100.0, bar);
    }

    // Compare with what proportional would give
    let counts = qsnn.output_spike_counts();
    let total: u32 = counts.iter().sum();
    if total > 0 {
        println!("\nProportional (no interference) would give:");
        for (i, &c) in counts.iter().enumerate() {
            let p = c as f64 / total as f64;
            let bar = "#".repeat((p * 50.0) as usize);
            println!("  Action {}: {:>6.2}% {}", i, p * 100.0, bar);
        }
    }

    // Calculate the difference
    println!("\nDifference (Quantum - Proportional):");
    if total > 0 {
        for (i, (&q_prob, &count)) in probs.iter().zip(counts.iter()).enumerate() {
            let p_prob = count as f64 / total as f64;
            let diff = q_prob - p_prob;
            let arrow = if diff > 0.01 {
                "↑"
            } else if diff < -0.01 {
                "↓"
            } else {
                "="
            };
            println!("  Action {}: {:+6.2}% {}", i, diff * 100.0, arrow);
        }
    }

    println!("\n================================================================");
    println!("  INTERFERENCE DEPTH COMPARISON");
    println!("================================================================\n");

    for depth in 0..=3 {
        let mut cfg = config.clone();
        cfg.interference_depth = depth;

        let mut action_counts = [0u32; 5];
        let n_samples = 100;

        for seed in 0..n_samples {
            let mut qsnn = QuantumSNN::with_seed(cfg.clone(), seed as u64 * 137);
            qsnn.set_inputs(&[200, 50, 50, 50, 100, 100, 100, 100]);
            let action = qsnn.decide();
            action_counts[action] += 1;
        }

        let ent = entropy(&action_counts, n_samples);
        println!(
            "Depth {}: entropy = {:.3}, distribution = {:?}",
            depth,
            ent,
            action_counts
                .iter()
                .map(|&c| format!("{}%", c))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

fn entropy(counts: &[u32], total: u32) -> f64 {
    let mut h = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            h -= p * p.log2();
        }
    }
    h
}
