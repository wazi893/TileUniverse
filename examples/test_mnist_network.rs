//! Minimal test: Does the DVS MNIST network structure work at all?

use engine::snn::{SNNConfig, SNNNetwork};

fn main() {
    println!("=== Testing DVS MNIST Network Structure ===\n");

    let config = SNNConfig::for_dvs_mnist();
    let mut network = SNNNetwork::new(config.clone());

    println!("Config:");
    println!("  Inputs: {}", config.n_inputs);
    println!("  Hidden: {:?}", config.hidden_layers);
    println!("  Outputs: {}", config.n_outputs);
    println!("  Total neurons: {}", config.total_neurons());
    println!("  Connection prob: {}", config.connection_prob);

    network.build_simple_connectivity(42);

    println!("\nNetwork built:");
    println!("  Populations: {}", network.populations.len());
    println!("  Synapses tables: {}", network.synapses.len());

    // Set ALL inputs to max
    let inputs = vec![255u8; config.n_inputs];
    println!("\nSetting all {} inputs to 255...", inputs.len());
    network.set_inputs(&inputs);

    println!("\nRunning 20 ticks...\n");

    for tick in 0..20 {
        network.step();

        let spike_count = network.spike_count();
        let max_v: Vec<i16> = network
            .populations
            .iter()
            .map(|p| p.neurons.iter().map(|n| n.v_mem).max().unwrap_or(0))
            .collect();

        if tick < 10 || spike_count > 0 {
            println!(
                "Tick {:>2}: Spikes = {:>4}, Max V = {:?}",
                tick,
                spike_count,
                &max_v[..3.min(max_v.len())]
            );
        }
    }

    println!("\nFinal output counts: {:?}", network.output_counts);
}
