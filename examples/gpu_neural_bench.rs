//! GPU vs CPU Stabilizer Neural Network Benchmark (EPIC 135)
//!
//! Compares performance of CPU and GPU-accelerated stabilizer neural networks.
//!
//! Run with: cargo run --release --example gpu_neural_bench --features cuda

#[cfg(feature = "cuda")]
use engine::snn::{
    GpuStabilizerNetwork, StabilizerBackend, StabilizerNetworkConfig, StabilizerNeuronNetwork,
};

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    println!("=== GPU vs CPU Stabilizer Neural Network Benchmark ===\n");
    println!("Note: Stabilizer operations are O(n²), so larger networks are slow.\n");

    // Test smaller network sizes with reduced ticks for large ones
    let configs = vec![
        // (inputs, hidden, outputs, ticks, name)
        (10, vec![64], 5, 100, "Small (79 neurons)"),
        (32, vec![128], 10, 50, "Medium (170 neurons)"),
        (64, vec![256], 16, 20, "Large (336 neurons)"),
        (128, vec![512], 32, 10, "XL (672 neurons)"),
        (256, vec![1024], 64, 5, "XXL (1344 neurons)"),
    ];

    let warmup_ticks = 2;

    for (n_inputs, hidden_layers, n_outputs, ticks, name) in configs {
        println!("--- {} ---", name);

        let config = StabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden_layers.clone(),
            n_outputs,
            connectivity: 0.2, // Lower connectivity for faster benchmarks
            recurrent: false,
            ..Default::default()
        };

        let total_neurons: usize = n_inputs + hidden_layers.iter().sum::<usize>() + n_outputs;

        // Generate random inputs
        let inputs: Vec<u8> = (0..n_inputs).map(|i| ((i * 37) % 256) as u8).collect();

        // Benchmark CPU network
        let mut cpu_network = StabilizerNeuronNetwork::new(config.clone(), 42);

        // Warmup
        for _ in 0..warmup_ticks {
            cpu_network.step(&inputs);
        }
        cpu_network.reset_state();

        let cpu_start = Instant::now();
        for _ in 0..ticks {
            cpu_network.step(&inputs);
        }
        let cpu_duration = cpu_start.elapsed();

        // Benchmark GPU network (auto backend)
        let mut gpu_network_auto =
            GpuStabilizerNetwork::with_backend(config.clone(), 42, StabilizerBackend::Auto)
                .expect("Failed to create auto-backend network");

        let uses_gpu = gpu_network_auto.is_gpu_accelerated();

        // Warmup
        for _ in 0..warmup_ticks {
            gpu_network_auto.step(&inputs).unwrap();
        }
        gpu_network_auto.reset_state().unwrap();

        let gpu_auto_start = Instant::now();
        for _ in 0..ticks {
            gpu_network_auto.step(&inputs).unwrap();
        }
        let gpu_auto_duration = gpu_auto_start.elapsed();

        // Calculate metrics
        let cpu_per_tick = cpu_duration.as_secs_f64() * 1000.0 / ticks as f64;
        let gpu_auto_per_tick = gpu_auto_duration.as_secs_f64() * 1000.0 / ticks as f64;
        let speedup = cpu_per_tick / gpu_auto_per_tick;

        println!("  Neurons: {}", total_neurons);
        println!("  Synapses: ~{}", cpu_network.num_synapses());
        println!("  Ticks: {}", ticks);
        println!("  CPU: {:.3} ms/tick", cpu_per_tick);
        println!(
            "  GPU (Auto={}): {:.3} ms/tick ({:.2}x)",
            if uses_gpu { "GPU" } else { "CPU" },
            gpu_auto_per_tick,
            speedup
        );
        println!();
    }

    // Quick decision benchmark with smaller network
    println!("=== Decision Benchmark ===\n");

    let config = StabilizerNetworkConfig {
        n_inputs: 64,
        hidden_layers: vec![256],
        n_outputs: 16,
        connectivity: 0.2,
        ..Default::default()
    };

    let inputs: Vec<u8> = (0..64).map(|i| ((i * 37) % 256) as u8).collect();
    let decisions = 10;
    let ticks_per_decision = 5;

    // CPU
    let mut cpu_net = StabilizerNeuronNetwork::new(config.clone(), 42);
    let cpu_start = Instant::now();
    for _ in 0..decisions {
        cpu_net.decide_integrated(&inputs, ticks_per_decision);
    }
    let cpu_decision_time = cpu_start.elapsed();

    // GPU (auto)
    let mut gpu_net =
        GpuStabilizerNetwork::with_backend(config.clone(), 42, StabilizerBackend::Auto)
            .expect("Failed to create network");
    let gpu_start = Instant::now();
    for _ in 0..decisions {
        gpu_net
            .decide_integrated(&inputs, ticks_per_decision)
            .unwrap();
    }
    let gpu_decision_time = gpu_start.elapsed();

    let speedup = cpu_decision_time.as_secs_f64() / gpu_decision_time.as_secs_f64();

    println!(
        "336 neuron network, {} decisions ({} ticks each):",
        decisions, ticks_per_decision
    );
    println!(
        "  CPU: {:.3} ms/decision",
        cpu_decision_time.as_secs_f64() * 1000.0 / decisions as f64
    );
    println!(
        "  GPU (auto): {:.3} ms/decision ({:.2}x)",
        gpu_decision_time.as_secs_f64() * 1000.0 / decisions as f64,
        speedup
    );

    println!("\n=== Summary ===");
    println!(
        "Current implementation uses CPU for gates (fast) and prepares for GPU batched measurements."
    );
    println!(
        "The O(n²) stabilizer tableau dominates - GPU batched measurement helps at 10K+ qubits."
    );
    println!("See gpu_stabilizer_bench for raw stabilizer performance comparison.");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This benchmark requires the 'cuda' feature.");
    println!("Run with: cargo run --release --example gpu_neural_bench --features cuda");
}
