//! Debug SNN connectivity - check if synapses are created properly

use engine::snn::{SNNConfig, SNNNetwork};

fn main() {
    println!("=== SNN Connectivity Debug ===\n");

    let config = SNNConfig::for_dvs_mnist();
    let mut network = SNNNetwork::new(config.clone());

    println!("Before build_connectivity:");
    println!("  Total synapses: {}", network.n_synapses());
    for (cpu, table) in network.synapses.iter().enumerate() {
        let local_count: usize = table.local.iter().map(|v| v.len()).sum();
        println!("  CPU {}: {} local synapses", cpu, local_count);
    }

    println!("\nBuilding connectivity with seed=42...");
    network.build_connectivity(42);

    println!("\nAfter build_connectivity:");
    println!("  Total synapses: {}", network.n_synapses());
    println!("  Estimated synapses: {}", config.estimated_synapses());

    // Check cross-CPU synapses from input layer
    let mut cross_cpu_count = 0;
    let n_inputs = config.actual_input_count();
    for cpu in 0..network.topology.n_cpus {
        let (start, end) = network.topology.cpu_to_neurons[cpu];
        for neuron_id in start..end {
            if neuron_id < n_inputs {
                let local_idx = network
                    .topology
                    .local_index(neuron_id, config.neurons_per_cpu);
                if let Some(syns) = network.router.routers[cpu].get_synapses(local_idx as u8) {
                    cross_cpu_count += syns.len();
                }
            }
        }
    }
    println!("  Cross-CPU synapses from inputs: {}", cross_cpu_count);

    // Check synapses from input layer (neurons 0..1567)
    let mut input_synapse_count = 0;

    for cpu in 0..network.topology.n_cpus {
        let (start, end) = network.topology.cpu_to_neurons[cpu];

        // Count synapses from this CPU
        let local_count: usize = network.synapses[cpu].local.iter().map(|v| v.len()).sum();

        // Check how many are from input neurons
        for neuron_id in start..end {
            if neuron_id < n_inputs {
                let local_idx = neuron_id - start;
                if local_idx < network.synapses[cpu].local.len() {
                    input_synapse_count += network.synapses[cpu].local[local_idx].len();
                }
            }
        }

        println!(
            "  CPU {}: {} local synapses (neurons {}-{})",
            cpu,
            local_count,
            start,
            end.saturating_sub(1)
        );
    }

    println!("\n  Synapses from input layer: {}", input_synapse_count);
    println!(
        "  Average synapses per input: {:.1}",
        input_synapse_count as f32 / n_inputs as f32
    );

    // Sample a few input neurons and show their connectivity
    println!("\nSample input neuron connectivity:");
    for sample_id in [0, 100, 500, 1000, 1567] {
        if sample_id >= n_inputs {
            continue;
        }

        let cpu = network.topology.neuron_to_cpu[sample_id];
        let local = network
            .topology
            .local_index(sample_id, config.neurons_per_cpu);

        if cpu < network.synapses.len() && local < network.synapses[cpu].local.len() {
            let syns = &network.synapses[cpu].local[local];
            println!("  Input neuron {}: {} synapses", sample_id, syns.len());
            if !syns.is_empty() {
                let weights: Vec<i16> = syns.iter().map(|s| s.weight as i16).collect();
                let targets: Vec<u16> = syns.iter().take(5).map(|s| s.target).collect();
                println!(
                    "    Weight range: {}..{}",
                    weights.iter().min().unwrap_or(&0),
                    weights.iter().max().unwrap_or(&0)
                );
                println!("    First 5 targets (local): {:?}", targets);
            }
        }
    }

    println!("\n=== Testing with manual spike ===");

    // Manually inject a spike on first input neuron
    let input_neuron = &mut network.populations[0].neurons[0];
    input_neuron.spiked = 1;
    input_neuron.v_mem = -128;
    input_neuron.refractory = 0; // Don't set refractory yet

    println!(
        "Injected spike on neuron 0 (spiked={}, v_mem={}, refractory={})",
        input_neuron.spiked, input_neuron.v_mem, input_neuron.refractory
    );

    // Check which neurons it connects to
    let cpu0_fired: Vec<usize> = network.populations[0]
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.fired())
        .map(|(i, _)| i)
        .collect();
    println!("Fired neurons in CPU 0: {:?}", cpu0_fired);

    // Accumulate currents
    let mut currents = vec![0i16; network.populations[0].len()];
    network.synapses[0].accumulate_currents(&cpu0_fired, &mut currents);

    let non_zero: Vec<(usize, i16)> = currents
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, c)| (i, *c))
        .collect();

    println!("Non-zero currents in CPU 0: {} neurons", non_zero.len());
    if !non_zero.is_empty() {
        println!(
            "  Sample currents: {:?}",
            &non_zero[..non_zero.len().min(10)]
        );
    }

    // Check if hidden layer neurons would receive current
    let hidden_start = config.actual_input_count();
    let hidden_currents: Vec<(usize, i16)> = currents
        .iter()
        .enumerate()
        .filter(|(i, c)| *i >= hidden_start && **c != 0)
        .map(|(i, c)| (i, *c))
        .collect();

    if hidden_currents.is_empty() {
        println!("\nWARNING: No currents reaching hidden layer!");
        println!("Hidden layer starts at neuron {}", hidden_start);
        println!("This suggests synapses aren't connected properly.");
    } else {
        println!(
            "\nHidden layer receiving current: {} neurons",
            hidden_currents.len()
        );
        println!(
            "  Sample: {:?}",
            &hidden_currents[..hidden_currents.len().min(5)]
        );
    }

    println!("\n=== Testing with full step() (cross-CPU routing) ===");

    // Set high input rates to trigger many spikes
    let inputs = vec![255u8; config.n_inputs];
    network.set_inputs(&inputs);

    println!("Set all {} inputs to rate 255", inputs.len());

    // Run one step
    network.step();

    // Check voltages after step
    let max_v: Vec<i16> = network
        .populations
        .iter()
        .map(|p| p.neurons.iter().map(|n| n.v_mem).max().unwrap_or(0))
        .collect();
    let spike_counts: Vec<u32> = network.populations.iter().map(|p| p.spike_count).collect();

    println!("\nAfter 1 step with all inputs=255:");
    println!("  Total spikes: {}", network.spike_count());
    println!(
        "  Spike counts per CPU: {:?}",
        &spike_counts[..spike_counts.len().min(5)]
    );
    println!("  Max V per CPU: {:?}", &max_v[..max_v.len().min(5)]);

    // Check hidden layer specifically (CPU 12-13)
    if network.populations.len() > 12 {
        let hidden_voltages: Vec<i16> = network.populations[12]
            .neurons
            .iter()
            .take(10)
            .map(|n| n.v_mem)
            .collect();
        println!(
            "\n  Hidden layer (CPU 12) first 10 voltages: {:?}",
            hidden_voltages
        );

        let hidden_max = network.populations[12]
            .neurons
            .iter()
            .map(|n| n.v_mem)
            .max()
            .unwrap_or(0);
        let hidden_min = network.populations[12]
            .neurons
            .iter()
            .map(|n| n.v_mem)
            .min()
            .unwrap_or(0);
        println!(
            "  Hidden layer voltage range: {}..{}",
            hidden_min, hidden_max
        );

        if hidden_max == -128 && hidden_min == -128 {
            println!("\n  ERROR: Hidden layer stuck at v_reset=-128!");
            println!("  This means no currents are reaching hidden neurons.");
        } else {
            println!("\n  SUCCESS: Hidden neurons accumulating voltage!");
        }
    }
}
