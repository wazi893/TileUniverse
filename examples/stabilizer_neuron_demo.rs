//! Stabilizer Quantum Neuron Demo (EPIC 130, 134)
//!
//! Demonstrates the stabilizer-based quantum neural network system where
//! each neuron IS a qubit, using the Aaronson-Gottesman tableau for efficient
//! simulation at scale.
//!
//! Run with: cargo run --example stabilizer_neuron_demo

use engine::snn::{
    HybridNetworkConfig, HybridQuantumNetwork, StabilizerNetworkConfig, StabilizerNeuronNetwork,
    StabilizerTopology,
};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(70));
    println!("  Stabilizer Quantum Neuron Demo (EPIC 130, 134)");
    println!("{}", "=".repeat(70));
    println!();

    // Demo 1: Basic stabilizer network
    demo_basic_network();

    // Demo 2: Decision making
    demo_decision_making();

    // Demo 3: Memory efficiency at scale
    demo_memory_efficiency();

    // Demo 4: Hybrid network with quantum clusters
    demo_hybrid_network();

    // Demo 5: Learning with reward signal
    demo_learning();

    // Demo 6: Bell state correlations (quantum behavior)
    demo_quantum_correlations();

    println!("{}", "=".repeat(70));
    println!("  Demo Complete!");
    println!("{}", "=".repeat(70));
}

/// Demo 1: Basic stabilizer network creation and topology
fn demo_basic_network() {
    println!("{}", "-".repeat(70));
    println!("Demo 1: Basic Stabilizer Network");
    println!("{}", "-".repeat(70));

    // Create a minimal network: 4 inputs -> 8 hidden -> 3 outputs
    let config = StabilizerNetworkConfig {
        n_inputs: 4,
        hidden_layers: vec![8],
        n_outputs: 3,
        connectivity: 0.5,
        ..Default::default()
    };

    let network = StabilizerNeuronNetwork::new(config.clone(), 42);

    println!("Network topology:");
    println!("  Input neurons:  {}", config.n_inputs);
    println!("  Hidden layers:  {:?}", config.hidden_layers);
    println!("  Output neurons: {}", config.n_outputs);
    println!("  Total neurons:  {}", network.num_neurons());
    println!("  Total synapses: {}", network.num_synapses());
    println!();

    // Show topology details
    let topology = StabilizerTopology::from_config(&config);
    println!("Layer structure:");
    for (i, layer) in topology.layers.iter().enumerate() {
        println!(
            "  Layer {}: qubits [{}, {}), type: {:?}",
            i,
            layer.start,
            layer.end(),
            layer.layer_type
        );
    }
    println!();
}

/// Demo 2: Making decisions with the network
fn demo_decision_making() {
    println!("{}", "-".repeat(70));
    println!("Demo 2: Decision Making");
    println!("{}", "-".repeat(70));

    // Create network for simple classification task
    let mut network = StabilizerNeuronNetwork::feedforward(4, &[16, 8], 4, 0.4, 42);

    println!("Network: 4 inputs -> [16, 8] hidden -> 4 outputs");
    println!();

    // Test with different input patterns
    let test_inputs = [
        ([255u8, 0, 0, 0], "Strong input 0"),
        ([0, 255, 0, 0], "Strong input 1"),
        ([0, 0, 255, 0], "Strong input 2"),
        ([0, 0, 0, 255], "Strong input 3"),
        ([128, 128, 128, 128], "Uniform inputs"),
        ([200, 50, 200, 50], "Alternating pattern"),
    ];

    println!("Single-step decisions:");
    for (inputs, description) in &test_inputs {
        let decision = network.decide(inputs);
        println!("  {} -> Output {}", description, decision);
    }
    println!();

    // Integrated decision (multiple ticks)
    println!("Integrated decisions (10 ticks):");
    for (inputs, description) in &test_inputs {
        let decision = network.decide_integrated(inputs, 10);
        println!("  {} -> Output {}", description, decision);
    }
    println!();
}

/// Demo 3: Memory efficiency at scale
fn demo_memory_efficiency() {
    println!("{}", "-".repeat(70));
    println!("Demo 3: Memory Efficiency at Scale");
    println!("{}", "-".repeat(70));

    let scales = [100, 1_000, 10_000, 50_000];

    println!("Stabilizer network memory usage (O(n^2) bits):");
    println!(
        "{:>10} {:>15} {:>15} {:>15}",
        "Neurons", "Tableau (est)", "Time to create", "Synapses"
    );
    println!("{}", "-".repeat(60));

    for &n in &scales {
        // Configure network with n neurons total
        let hidden_size = n - 10; // 5 inputs + 5 outputs + hidden
        if hidden_size < 1 {
            continue;
        }

        let config = StabilizerNetworkConfig {
            n_inputs: 5,
            hidden_layers: vec![hidden_size],
            n_outputs: 5,
            connectivity: 0.01, // Low connectivity for speed
            ..Default::default()
        };

        let start = Instant::now();
        let network = StabilizerNeuronNetwork::new(config, 42);
        let elapsed = start.elapsed();

        // Estimate tableau memory: O(n^2) bits = n^2 / 8 bytes
        let tableau_bytes = (n * n) / 8;
        let tableau_str = if tableau_bytes > 1_000_000 {
            format!("{:.1} MB", tableau_bytes as f64 / 1_000_000.0)
        } else if tableau_bytes > 1_000 {
            format!("{:.1} KB", tableau_bytes as f64 / 1_000.0)
        } else {
            format!("{} B", tableau_bytes)
        };

        println!(
            "{:>10} {:>15} {:>15.2?} {:>15}",
            n,
            tableau_str,
            elapsed,
            network.num_synapses()
        );
    }

    println!();
    println!("Compare to full quantum state (2^n amplitudes):");
    println!("  20 qubits: 2^20 * 16 bytes = 16 MB");
    println!("  30 qubits: 2^30 * 16 bytes = 16 GB");
    println!("  40 qubits: 2^40 * 16 bytes = 16 TB (impossible!)");
    println!();
    println!("Stabilizer formalism: 10K neurons = ~12 MB (tractable!)");
    println!();
}

/// Demo 4: Hybrid stabilizer-cluster network
fn demo_hybrid_network() {
    println!("{}", "-".repeat(70));
    println!("Demo 4: Hybrid Stabilizer-Cluster Network");
    println!("{}", "-".repeat(70));

    // Create hybrid network: stabilizer backbone + quantum clusters
    let backbone_config = StabilizerNetworkConfig {
        n_inputs: 8,
        hidden_layers: vec![32],
        n_outputs: 4,
        connectivity: 0.3,
        ..Default::default()
    };

    let config = HybridNetworkConfig {
        backbone: backbone_config.clone(),
        n_clusters: 2,
        cluster_size: 8, // 8 qubits per cluster (2^8 = 256 amplitudes)
        ..Default::default()
    };

    let mut network = HybridQuantumNetwork::new(config.clone(), 42);

    println!("Hybrid architecture:");
    println!(
        "  Stabilizer backbone: {} -> {:?} -> {} neurons",
        backbone_config.n_inputs, backbone_config.hidden_layers, backbone_config.n_outputs
    );
    println!(
        "  Quantum clusters: {} clusters x {} qubits each",
        config.n_clusters, config.cluster_size
    );
    println!("  Total memory: backbone O(n^2) + clusters O(2^k) per cluster");
    println!();

    // Make decisions with hybrid network
    let inputs = [200u8, 150, 100, 50, 200, 150, 100, 50];

    println!("Decision making with hybrid network:");
    for i in 0..5 {
        let decision = network.decide(&inputs);
        let entropy = network.cluster_entropy(0);
        println!(
            "  Step {}: decision = {}, cluster 0 entropy = {:.3} bits",
            i + 1,
            decision,
            entropy
        );
    }
    println!();

    // Demonstrate T-gate (non-Clifford) on cluster
    println!("Applying T-gate (non-Clifford) to cluster 0...");
    if let Some(cluster) = network.cluster_mut(0) {
        cluster.t(0); // Apply T gate to qubit 0 in cluster
    }
    let entropy_after = network.cluster_entropy(0);
    println!(
        "  Cluster 0 entropy after T-gate: {:.3} bits",
        entropy_after
    );
    println!();
}

/// Demo 5: Learning with reward signal
fn demo_learning() {
    println!("{}", "-".repeat(70));
    println!("Demo 5: Reward-Modulated Learning");
    println!("{}", "-".repeat(70));

    let mut network = StabilizerNeuronNetwork::feedforward(4, &[16], 2, 0.5, 42);

    // Training data: XOR-like problem
    let training_data = [
        ([200u8, 200, 50, 50], 0usize), // High-High-Low-Low -> 0
        ([50, 50, 200, 200], 1),        // Low-Low-High-High -> 1
        ([200, 50, 200, 50], 0),        // Alternating A -> 0
        ([50, 200, 50, 200], 1),        // Alternating B -> 1
    ];

    println!("Training on pattern classification task...");
    println!("  Pattern 1: [H,H,L,L] -> 0");
    println!("  Pattern 2: [L,L,H,H] -> 1");
    println!("  Pattern 3: [H,L,H,L] -> 0");
    println!("  Pattern 4: [L,H,L,H] -> 1");
    println!();

    // Initial accuracy
    let mut correct = 0;
    for (inputs, expected) in &training_data {
        let decision = network.decide_integrated(inputs, 5);
        if decision == *expected {
            correct += 1;
        }
    }
    println!("Initial accuracy: {}/{}", correct, training_data.len());

    // Training loop
    let epochs = 50;
    for epoch in 0..epochs {
        let mut epoch_correct = 0;

        for (inputs, expected) in &training_data {
            let decision = network.decide_integrated(inputs, 5);

            // Reward signal: +50 for correct, -30 for incorrect
            let reward = if decision == *expected {
                epoch_correct += 1;
                50i8
            } else {
                -30i8
            };

            network.learn(reward);
        }

        if (epoch + 1) % 10 == 0 {
            println!(
                "  Epoch {}: accuracy {}/{}",
                epoch + 1,
                epoch_correct,
                training_data.len()
            );
        }
    }

    // Final accuracy
    let mut correct = 0;
    for (inputs, expected) in &training_data {
        let decision = network.decide_integrated(inputs, 5);
        if decision == *expected {
            correct += 1;
        }
    }
    println!("Final accuracy: {}/{}", correct, training_data.len());
    println!();
}

/// Demo 6: Bell state correlations (quantum behavior)
fn demo_quantum_correlations() {
    println!("{}", "-".repeat(70));
    println!("Demo 6: Quantum Correlations (Bell States)");
    println!("{}", "-".repeat(70));

    // Create minimal network to observe Bell correlations
    let mut network = StabilizerNeuronNetwork::feedforward(2, &[2], 2, 1.0, 42);

    println!("Creating Bell pairs via CNOT entanglement...");
    println!();

    // Run many trials and collect correlation statistics
    let trials = 1000;
    let mut same_outcome = 0;
    let mut different_outcome = 0;

    for _ in 0..trials {
        // Apply Hadamard-like input to first qubit
        let inputs = [255u8, 128]; // Strong superposition on first, weaker on second

        // Run and observe correlations
        network.step(&inputs);
        let counts = network.get_output_spike_counts();

        // Check if outputs show correlation
        if counts.len() >= 2 {
            if (counts[0] > 0) == (counts[1] > 0) {
                same_outcome += 1;
            } else {
                different_outcome += 1;
            }
        }

        network.reset_output_counts();
    }

    println!("Correlation statistics over {} trials:", trials);
    println!(
        "  Same outcome:      {} ({:.1}%)",
        same_outcome,
        100.0 * same_outcome as f64 / trials as f64
    );
    println!(
        "  Different outcome: {} ({:.1}%)",
        different_outcome,
        100.0 * different_outcome as f64 / trials as f64
    );
    println!();

    println!("Note: In a fully entangled Bell state (|00> + |11>)/sqrt(2),");
    println!("measurements on entangled qubits show perfect correlation.");
    println!("The stabilizer tableau efficiently tracks these correlations!");
    println!();
}
