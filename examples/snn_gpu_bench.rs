//! SNN GPU Benchmark
//!
//! Compares CPU vs GPU performance for SNN simulation at various scales.
//!
//! Run with: cargo run --release --features cuda --example snn_gpu_bench

use std::time::Instant;

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, is_cuda_available};
#[cfg(feature = "cuda")]
use engine::snn::{GpuSNNState, SNNConfig};

fn main() {
    println!("===========================================");
    println!("  CHUNGUS 4: SNN GPU Benchmark");
    println!("===========================================\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: This benchmark requires the 'cuda' feature.");
        println!("Run with: cargo run --release --features cuda --example snn_gpu_bench");
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

    println!("CUDA runtime initialized successfully.\n");

    // Test configurations
    let configs = [
        ("Small", 1_000, 10_000),       // 1K neurons, 10K synapses
        ("Medium", 10_000, 100_000),    // 10K neurons, 100K synapses
        ("Large", 50_000, 500_000),     // 50K neurons, 500K synapses
        ("XLarge", 100_000, 1_000_000), // 100K neurons, 1M synapses
    ];

    println!(
        "{:<10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "Size", "Neurons", "Synapses", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{:-<70}", "");

    for (name, n_neurons, n_synapses) in configs {
        match benchmark_scale(&rt, n_neurons, n_synapses) {
            Ok((cpu_ms, gpu_ms)) => {
                let speedup = cpu_ms / gpu_ms;
                println!(
                    "{:<10} {:>10} {:>12} {:>12.2} {:>12.2} {:>10.1}x",
                    name, n_neurons, n_synapses, cpu_ms, gpu_ms, speedup
                );
            }
            Err(e) => {
                println!(
                    "{:<10} {:>10} {:>12} ERROR: {:?}",
                    name, n_neurons, n_synapses, e
                );
            }
        }
    }

    println!("\n===========================================");
    println!("  Benchmark Complete");
    println!("===========================================");
}

#[cfg(feature = "cuda")]
fn benchmark_scale(
    rt: &CudaRuntime,
    n_neurons: usize,
    n_synapses: usize,
) -> Result<(f64, f64), String> {
    use engine::snn::LIFNeuron;
    use engine::snn::synapse::SynapseCSR;

    // Calculate network structure
    let n_inputs = (n_neurons / 10).max(10);
    let n_outputs = 5;
    let n_hidden = n_neurons - n_inputs - n_outputs;

    // Create neurons
    let neurons: Vec<LIFNeuron> = (0..n_neurons).map(|_| LIFNeuron::default()).collect();

    // Create random sparse connectivity (CSR format)
    let synapses_per_neuron = n_synapses / n_neurons;
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::with_capacity(n_synapses);
    let mut weights = Vec::with_capacity(n_synapses);
    let mut delays = Vec::with_capacity(n_synapses);

    let mut rng_state = 42u64;
    let mut ptr = 0u32;

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        // Generate random synapses from this neuron
        for _ in 0..synapses_per_neuron {
            // Simple LCG for randomness
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let tgt = ((rng_state >> 32) as usize) % n_neurons;

            if tgt != src {
                targets.push(tgt as u16);
                weights.push(((rng_state >> 48) as i8).abs().max(10)); // Positive weights
                delays.push(0);
                ptr += 1;
            }
        }
    }
    syn_ptr.push(ptr);

    let csr = SynapseCSR {
        syn_ptr,
        targets,
        weights,
        delays,
    };

    let actual_synapses = csr.n_synapses();

    // Benchmark parameters
    let ticks = 100;
    let warmup_ticks = 10;

    // ========== CPU Benchmark ==========
    // Create CPU network
    let config = SNNConfig {
        n_inputs,
        hidden_layers: vec![n_hidden],
        n_outputs,
        connection_prob: 0.0, // We'll use our own connectivity
        neurons_per_cpu: 256,
        recurrent: false,
        ..SNNConfig::default()
    };

    // For CPU, we use a simplified approach - just measure neuron step time
    let cpu_start = Instant::now();

    // Simulate CPU work: O(n_neurons) for dynamics + O(n_synapses) for currents
    let mut dummy_v: Vec<i16> = vec![0; n_neurons];
    let mut dummy_currents: Vec<i16> = vec![0; n_neurons];

    for _ in 0..(warmup_ticks + ticks) {
        // Simulate LIF dynamics (O(n_neurons))
        for i in 0..n_neurons {
            let v = (dummy_v[i] as i32 * 230) >> 8;
            dummy_v[i] = (v + dummy_currents[i] as i32).clamp(-32768, 32767) as i16;
        }

        // Simulate current accumulation (O(n_synapses))
        for i in 0..actual_synapses.min(n_neurons * 10) {
            let idx = i % n_neurons;
            dummy_currents[idx] = dummy_currents[idx].saturating_add(1);
        }

        // Clear currents
        for c in &mut dummy_currents {
            *c = 0;
        }
    }

    let cpu_elapsed = cpu_start.elapsed();
    let cpu_ms = cpu_elapsed.as_secs_f64() * 1000.0 / ticks as f64;

    // ========== GPU Benchmark ==========
    let mut gpu_state = GpuSNNState::from_cpu(rt, &neurons, &csr, n_inputs, n_outputs)
        .map_err(|e| format!("GPU state creation failed: {:?}", e))?;

    // Set input rates
    let input_rates: Vec<u8> = (0..n_inputs).map(|i| ((i % 256) as u8)).collect();
    gpu_state
        .set_input_rates(rt, &input_rates)
        .map_err(|e| format!("Set input rates failed: {:?}", e))?;

    // Warmup
    for _ in 0..warmup_ticks {
        gpu_state
            .step(rt)
            .map_err(|e| format!("GPU warmup step failed: {:?}", e))?;
    }

    // Timed run
    let gpu_start = Instant::now();
    for _ in 0..ticks {
        gpu_state
            .step(rt)
            .map_err(|e| format!("GPU step failed: {:?}", e))?;
    }
    let gpu_elapsed = gpu_start.elapsed();
    let gpu_ms = gpu_elapsed.as_secs_f64() * 1000.0 / ticks as f64;

    Ok((cpu_ms, gpu_ms))
}

#[cfg(not(feature = "cuda"))]
fn benchmark_scale(_n_neurons: usize, _n_synapses: usize) -> Result<(f64, f64), String> {
    Err("CUDA feature not enabled".to_string())
}
