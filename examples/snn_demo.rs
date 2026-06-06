//! CHUNGUS 4: Spiking Neural Network Demo
//!
//! Demonstrates the neuromorphic SNN system with:
//! - LIF neuron dynamics
//! - STDP learning
//! - Multi-CPU spike routing
//!
//! Run with: cargo run --release --example snn_demo

use engine::snn::{LIFNeuron, SNNConfig, SNNNetwork, STDPConfig};

fn main() {
    println!("===========================================");
    println!("  CHUNGUS 4: Neuromorphic SNN Demo");
    println!("===========================================\n");

    // Demo 1: Single LIF Neuron
    demo_single_neuron();

    // Demo 2: Small Network
    demo_small_network();

    // Demo 3: STDP Learning
    demo_stdp_learning();

    // Demo 4: Scalability
    demo_scalability();

    println!("\n===========================================");
    println!("  CHUNGUS 4 Demo Complete!");
    println!("===========================================");
}

fn demo_single_neuron() {
    println!("--- Demo 1: Single LIF Neuron ---\n");

    let mut neuron = LIFNeuron::new();
    println!("Created LIF neuron:");
    println!("  Threshold: {:.2}", neuron.threshold_f32());
    println!("  Leak: {:.3}", neuron.leak as f32 / 256.0);
    println!("  Size: {} bytes\n", std::mem::size_of::<LIFNeuron>());

    // Simulate with varying input
    println!("Simulating 20 ticks with oscillating input:");
    println!(
        "{:>5} {:>10} {:>8} {:>8}",
        "Tick", "Input", "V_mem", "Spiked"
    );
    println!("{:-<40}", "");

    for tick in 0..20u8 {
        // Oscillating input current
        let input = if tick % 4 < 2 { 150i16 } else { 50 };

        let fired = neuron.step(input, tick);

        if tick < 10 || fired {
            println!(
                "{:>5} {:>10} {:>8.2} {:>8}",
                tick,
                input,
                neuron.v_mem_f32(),
                if fired { "SPIKE!" } else { "" }
            );
        }
    }
    println!();
}

fn demo_small_network() {
    println!("--- Demo 2: Small Network ---\n");

    let config = SNNConfig {
        n_inputs: 4,
        hidden_layers: vec![16],
        n_outputs: 3,
        connection_prob: 0.5,
        recurrent: true,
        ..SNNConfig::default()
    };

    println!("Network configuration:");
    println!("  Inputs: {}", config.n_inputs);
    println!("  Hidden layers: {:?}", config.hidden_layers);
    println!("  Outputs: {}", config.n_outputs);
    println!("  Total neurons: {}", config.total_neurons());
    println!("  Required CPUs: {}", config.required_cpus());
    println!("  Est. synapses: {}\n", config.estimated_synapses());

    let mut network = SNNNetwork::new(config);
    network.build_connectivity(42);

    println!("Actual synapses created: {}\n", network.n_synapses());

    // Run simulation
    println!("Running 100 ticks with random inputs...");
    let mut total_spikes = 0u64;

    for tick in 0..100 {
        // Random-ish input rates
        let inputs = [
            ((tick * 7) % 256) as u8,
            ((tick * 11 + 50) % 256) as u8,
            ((tick * 13 + 100) % 256) as u8,
            ((tick * 17 + 150) % 256) as u8,
        ];
        network.set_inputs(&inputs);
        network.step();
        total_spikes += network.spike_count() as u64;

        // Report every 20 ticks
        if (tick + 1) % 20 == 0 {
            let action = network.get_action();
            println!(
                "  Tick {:>3}: spikes={:>3}, action={}",
                tick + 1,
                network.spike_count(),
                action
            );
        }
    }

    println!("\nTotal spikes: {}", total_spikes);
    println!("Avg spikes/tick: {:.1}", total_spikes as f64 / 100.0);
    println!();
}

fn demo_stdp_learning() {
    println!("--- Demo 3: STDP Learning ---\n");

    use engine::snn::{Synapse, apply_stdp};

    let config = STDPConfig::default();
    println!("STDP configuration:");
    println!("  tau_plus: {} ticks", config.tau_plus);
    println!("  tau_minus: {} ticks", config.tau_minus);
    println!("  a_plus: {}", config.a_plus);
    println!("  a_minus: {}", config.a_minus);
    println!("  Weight bounds: [{}, {}]\n", config.w_min, config.w_max);

    // Create test synapse
    let mut synapse = Synapse::excitatory(0, 0, 0); // Start at weight 0

    println!("Testing STDP weight updates:");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "Pre Time", "Post Time", "Delta", "New Weight"
    );
    println!("{:-<52}", "");

    // LTP: Post after pre
    let scenarios = [
        (10u8, 12u8, "LTP (+2)"),
        (10, 15, "LTP (+5)"),
        (10, 20, "LTP (+10)"),
        (20, 15, "LTD (-5)"),
        (20, 10, "LTD (-10)"),
    ];

    for (pre, post, desc) in scenarios {
        let old_weight = synapse.weight;
        apply_stdp(&mut synapse, pre, post, &config);
        let delta = synapse.weight - old_weight;
        println!(
            "{:>12} {:>12} {:>12} {:>12} ({})",
            pre, post, delta, synapse.weight, desc
        );
    }
    println!();
}

fn demo_scalability() {
    println!("--- Demo 4: Scalability Test ---\n");

    let configs = [
        ("Tiny", SNNConfig::new(4, vec![16], 2)),
        ("Small", SNNConfig::new(8, vec![64, 32], 5)),
        ("Medium", SNNConfig::new(16, vec![128, 64], 8)),
        ("Large", SNNConfig::new(32, vec![256, 128, 64], 16)),
    ];

    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>12}",
        "Size", "Neurons", "CPUs", "Synapses", "Time/100"
    );
    println!("{:-<60}", "");

    for (name, config) in configs {
        let neurons = config.total_neurons();
        let cpus = config.required_cpus();

        let mut network = SNNNetwork::new(config);
        network.build_connectivity(123);
        let synapses = network.n_synapses();

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..100 {
            network.set_inputs(&[128; 32]);
            network.step();
        }
        let elapsed = start.elapsed();

        println!(
            "{:<10} {:>10} {:>10} {:>12} {:>9.2}ms",
            name,
            neurons,
            cpus,
            synapses,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!();

    // Million neuron projection
    println!("Projected scaling to 1M+ neurons:");
    let mega_config = SNNConfig {
        n_inputs: 128,
        hidden_layers: vec![32768, 16384, 8192],
        n_outputs: 64,
        connection_prob: 0.01, // Very sparse
        neurons_per_cpu: 128,
        ..SNNConfig::default()
    };

    println!("  Neurons: {}", mega_config.total_neurons());
    println!("  CPUs required: {}", mega_config.required_cpus());
    println!("  Est. synapses: {}", mega_config.estimated_synapses());
}
