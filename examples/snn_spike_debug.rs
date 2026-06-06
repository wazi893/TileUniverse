//! Debug: Why aren't neurons spiking?
//!
//! Run with: cargo run --release --example snn_spike_debug

use engine::snn::{NeuronConfig, SNNConfig, SNNNetwork, STDPConfig};

fn main() {
    println!("================================================================");
    println!("  SNN SPIKE DEBUG: Why aren't neurons firing?");
    println!("================================================================\n");

    let config = SNNConfig {
        n_inputs: 4,
        hidden_layers: vec![8],
        n_outputs: 2,
        connection_prob: 0.8,
        neurons_per_cpu: 32,
        recurrent: false,
        neuron_config: NeuronConfig::default(),
        stdp_config: STDPConfig::default(),
        use_reward_modulation: false,
        duplicate_inputs: false,
    };

    let mut snn = SNNNetwork::new(config.clone());
    snn.build_connectivity(42);
    snn.randomize_weights(12345);

    println!(
        "Network: {} inputs → {} hidden → {} outputs",
        config.n_inputs, config.hidden_layers[0], config.n_outputs
    );
    println!("Total neurons: {}", config.total_neurons());
    println!(
        "Synapse counts: {:?}",
        snn.synapses
            .iter()
            .map(|s| s.total_count)
            .collect::<Vec<_>>()
    );

    // Check neuron configuration
    println!("\nNeuron Config:");
    let nc = &config.neuron_config;
    println!("  leak: {}", nc.leak);
    println!(
        "  threshold: {} (Q8.8 → {:.2})",
        nc.threshold,
        nc.threshold as f64 / 256.0
    );
    println!("  refractory_period: {}", nc.refractory_period);
    println!("  v_reset: {}", nc.v_reset);

    // Set max inputs
    let inputs = vec![255u8; config.n_inputs];
    snn.set_inputs(&inputs);

    println!("\n--- Running 50 ticks with max input (255) ---\n");

    for tick in 0..50 {
        snn.step();

        // Count spikes in each population
        let mut total_spikes = 0u32;
        for pop in &snn.populations {
            total_spikes += pop.spike_count;
        }

        if tick < 10 || tick % 10 == 0 || total_spikes > 0 {
            println!(
                "Tick {:>3}: Spikes this tick: {:>3}, Output counts: {:?}",
                tick, total_spikes, snn.output_counts
            );
        }
    }

    // Trace first few neurons
    println!("\n--- Neuron State Sample (first population) ---");
    if let Some(pop) = snn.populations.first() {
        for (i, neuron) in pop.neurons.iter().enumerate().take(8) {
            println!(
                "Neuron {}: v_mem = {:>6} ({:>6.2}), spiked = {}, refractory = {}",
                i,
                neuron.v_mem,
                neuron.v_mem as f64 / 256.0,
                neuron.spiked,
                neuron.refractory
            );
        }
    }

    // Try with stronger weights
    println!("\n================================================================");
    println!("  Testing with stronger weights (+100)");
    println!("================================================================\n");

    let mut snn2 = SNNNetwork::new(config.clone());
    snn2.build_connectivity(42);

    // Manually set stronger weights
    for synapse_table in &mut snn2.synapses {
        for syn_list in &mut synapse_table.local {
            for syn in syn_list {
                syn.weight = 100; // Strong excitatory
            }
        }
    }

    snn2.set_inputs(&[255, 255, 255, 255]);

    for tick in 0..20 {
        snn2.step();
        let mut total_spikes = 0u32;
        for pop in &snn2.populations {
            total_spikes += pop.spike_count;
        }
        println!(
            "Tick {:>3}: Spikes: {:>3}, Output counts = {:?}",
            tick, total_spikes, snn2.output_counts
        );
    }

    // Check actual spike generation for inputs
    println!("\n================================================================");
    println!("  Input Encoding Analysis");
    println!("================================================================\n");

    for rate in [0u8, 64, 128, 192, 255] {
        let mut snn3 = SNNNetwork::new(config.clone());
        snn3.build_connectivity(42);
        snn3.randomize_weights(12345);
        snn3.set_inputs(&[rate, rate, rate, rate]);

        let mut total_population_spikes = 0u64;

        for _tick in 0..100 {
            snn3.step();
            for pop in &snn3.populations {
                total_population_spikes += pop.spike_count as u64;
            }
        }

        println!(
            "Rate {:>3} → {} total spikes over 100 ticks ({:.1} per tick)",
            rate,
            total_population_spikes,
            total_population_spikes as f64 / 100.0
        );
    }

    // Check if the issue is input neurons not generating spikes
    println!("\n================================================================");
    println!("  Checking Input Neuron Behavior");
    println!("================================================================\n");

    let mut snn4 = SNNNetwork::new(config.clone());
    snn4.build_connectivity(42);
    snn4.randomize_weights(12345);
    snn4.set_inputs(&[255, 255, 255, 255]);

    println!("Input rates set: {:?}", snn4.input_rates);

    for tick in 0..10 {
        snn4.step();

        // Check input neuron voltages
        if let Some(pop) = snn4.populations.first() {
            let input_voltages: Vec<i16> = pop
                .neurons
                .iter()
                .take(config.n_inputs)
                .map(|n| n.v_mem)
                .collect();
            let input_spiked: Vec<u8> = pop
                .neurons
                .iter()
                .take(config.n_inputs)
                .map(|n| n.spiked)
                .collect();

            println!("Tick {}: Input voltages: {:?}", tick, input_voltages);
            println!("        Input spiked:   {:?}", input_spiked);
        }
    }
}
