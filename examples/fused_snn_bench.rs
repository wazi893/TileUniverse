//! Fused SNN Benchmark
//!
//! Compares EPIC 97-style fused kernels vs separate kernel approach.
//!
//! Run with: cargo run --release --features cuda --example fused_snn_bench

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, is_cuda_available};
#[cfg(feature = "cuda")]
use engine::snn::{
    GpuFusedSNN, generate_columnar_csr, generate_dense_feedforward_csr, generate_layered_csr,
};
#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       FUSED SNN BENCHMARK (EPIC 97-Style Architecture)           ║");
    println!("║                                                                  ║");
    println!("║  Zero PCIe transfers per tick - fire-and-forget simulation       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        println!("Build with: cargo run --example fused_snn_bench --features cuda --release");
        return;
    }

    #[cfg(feature = "cuda")]
    run_benchmark();
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
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

    // ========================================================================
    // Test 1: Correctness - verify spikes are generated
    // ========================================================================
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│                    TEST 1: Correctness                         │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    {
        let n_neurons = 1000;
        let n_inputs = 100;
        let n_outputs = 10;
        // Use dense feedforward for guaranteed spike propagation
        let synapses = generate_dense_feedforward_csr(n_neurons, n_inputs, n_outputs, 42);

        let mut snn =
            GpuFusedSNN::new(&rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses).unwrap();

        // High spike rates to ensure activity
        let rates: Vec<u8> = (0..n_inputs).map(|i| 128 + ((i % 64) as u8)).collect();
        snn.set_input_rates(&rt, &rates).unwrap();

        // Run 100 ticks with output counting to accumulate spike totals
        snn.tick_many_with_counting(&rt, 100).unwrap();
        snn.synchronize(&rt).unwrap();

        let snapshot = snn.snapshot(&rt).unwrap();
        let total_output_spikes: u32 = snapshot.output_counts.iter().sum();
        let active_neurons = snapshot.spiked.iter().filter(|&&s| s == 1).count();

        println!("  Neurons:            {}", n_neurons);
        println!("  Synapses:           {}", synapses.n_synapses());
        println!("  Ticks:              100");
        println!("  Output spikes:      {}", total_output_spikes);
        println!("  Active last tick:   {}", active_neurons);
        println!(
            "  Status:             {}",
            if total_output_spikes > 0 {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
    }
    println!();

    // ========================================================================
    // Test 2: Scaling Performance
    // ========================================================================
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│                  TEST 2: Scaling Performance                   │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    let test_configs = [
        (1_000, 10, "1K"),
        (10_000, 10, "10K"),
        (50_000, 10, "50K"),
        (100_000, 10, "100K"),
        (200_000, 10, "200K"),
        (500_000, 10, "500K"),
        (1_000_000, 10, "1M"),
    ];

    println!(
        "{:>10} {:>12} {:>12} {:>14} {:>16}",
        "Neurons", "Synapses", "Ticks/sec", "Time/tick", "Neuron-ticks/s"
    );
    println!("{:-<66}", "");

    let mut best_throughput = 0.0f64;
    let mut best_config = "";

    for (n_neurons, _syn_per_neuron, label) in test_configs {
        let n_inputs = (n_neurons / 10).max(100);
        let n_outputs = 10;

        // Use layered connectivity for proper spike propagation
        let synapses = generate_layered_csr(n_neurons, n_inputs, n_outputs, 20, 12345);
        let n_synapses = synapses.n_synapses();

        let mut snn =
            match GpuFusedSNN::new(&rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses) {
                Ok(s) => s,
                Err(e) => {
                    println!("{:>10} FAILED: {:?}", label, e);
                    continue;
                }
            };

        // Set input rates
        let rates: Vec<u8> = (0..n_inputs).map(|i| ((i * 7) % 200) as u8 + 50).collect();
        if snn.set_input_rates(&rt, &rates).is_err() {
            println!("{:>10} Failed to set inputs", label);
            continue;
        }

        // Warmup
        let warmup_ticks = 10;
        snn.tick_many(&rt, warmup_ticks).unwrap();
        snn.synchronize(&rt).unwrap();

        // Benchmark - fire and forget!
        let bench_ticks = if n_neurons >= 500_000 { 100 } else { 500 };
        let start = Instant::now();
        snn.tick_many(&rt, bench_ticks).unwrap();
        snn.synchronize(&rt).unwrap();
        let elapsed = start.elapsed();

        let ticks_per_sec = bench_ticks as f64 / elapsed.as_secs_f64();
        let time_per_tick_us = elapsed.as_micros() as f64 / bench_ticks as f64;
        let neuron_ticks_per_sec = (n_neurons as f64) * ticks_per_sec;

        if neuron_ticks_per_sec > best_throughput {
            best_throughput = neuron_ticks_per_sec;
            best_config = label;
        }

        println!(
            "{:>10} {:>12} {:>10.1}/s {:>12.2} us {:>14.2}B/s",
            label,
            format_num(n_synapses),
            ticks_per_sec,
            time_per_tick_us,
            neuron_ticks_per_sec / 1e9
        );
    }

    println!();

    // ========================================================================
    // Test 3: Fire-and-Forget Demo
    // ========================================================================
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│              TEST 3: Fire-and-Forget Demo                      │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    {
        let n_neurons = 200_000;
        let n_inputs = 20_000;
        let n_outputs = 10;
        // Dense feedforward connectivity for reliable spike propagation
        let synapses = generate_dense_feedforward_csr(n_neurons, n_inputs, n_outputs, 99999);

        let mut snn =
            GpuFusedSNN::new(&rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses).unwrap();

        let rates: Vec<u8> = (0..n_inputs).map(|i| ((i * 3) % 200) as u8 + 50).collect();
        snn.set_input_rates(&rt, &rates).unwrap();

        let mega_ticks = 10_000;

        println!(
            "  Running {} ticks with {} neurons...",
            mega_ticks,
            format_num(n_neurons)
        );
        println!("  (No CPU involvement during simulation!)");
        println!();

        let start = Instant::now();
        snn.tick_many_with_counting(&rt, mega_ticks).unwrap();
        snn.synchronize(&rt).unwrap();
        let elapsed = start.elapsed();

        let ticks_per_sec = mega_ticks as f64 / elapsed.as_secs_f64();
        let neuron_ticks = n_neurons * mega_ticks;
        let neuron_ticks_per_sec = neuron_ticks as f64 / elapsed.as_secs_f64();

        let snapshot = snn.snapshot(&rt).unwrap();
        let total_output_spikes: u32 = snapshot.output_counts.iter().sum();

        println!("  Total time:           {:.2}s", elapsed.as_secs_f64());
        println!("  Ticks per second:     {:.0}", ticks_per_sec);
        println!("  Neuron-ticks/sec:     {:.2}B", neuron_ticks_per_sec / 1e9);
        println!("  Output spikes:        {}", total_output_spikes);
    }
    println!();

    // ========================================================================
    // Test 4: Decision Making (RL-style)
    // ========================================================================
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│             TEST 4: Decision Making (RL-style)                 │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    {
        let n_neurons = 100_000;
        let n_inputs = 10_000;
        let n_outputs = 5;
        // Dense feedforward connectivity for varied output activity
        let synapses = generate_dense_feedforward_csr(n_neurons, n_inputs, n_outputs, 77777);

        let mut snn =
            GpuFusedSNN::new(&rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses).unwrap();

        // Apply threshold asymmetry to break uniform output distribution
        // Feedback: "Uniform spike counts mean your network is a maximum entropy policy"
        // Breaking symmetry via threshold variation allows varied decisions
        snn.apply_output_threshold_asymmetry(&rt, 100, 30, 12345)
            .unwrap();

        let ticks_per_decision = 20;
        let n_decisions = 100;

        println!(
            "  {} neurons, {} ticks/decision, {} decisions",
            format_num(n_neurons),
            ticks_per_decision,
            n_decisions
        );
        println!("  (with threshold asymmetry for varied decisions)");
        println!();

        let start = Instant::now();
        let mut action_counts = vec![0u32; n_outputs];

        for i in 0..n_decisions {
            // Vary inputs each decision
            let rates: Vec<u8> = (0..n_inputs)
                .map(|j| ((j + i * 100) % 200) as u8 + 50)
                .collect();
            snn.set_input_rates(&rt, &rates).unwrap();

            // Reset output counts
            snn.reset_output_counts(&rt).unwrap();

            // Run decision ticks with output counting
            snn.tick_many_with_counting(&rt, ticks_per_decision)
                .unwrap();
            snn.synchronize(&rt).unwrap();

            // Get decision
            let action = snn.get_decision(&rt).unwrap();
            action_counts[action] += 1;
        }

        let elapsed = start.elapsed();
        let decisions_per_sec = n_decisions as f64 / elapsed.as_secs_f64();

        // Get last decision's output spike counts to verify activity
        let last_counts = snn.get_output_counts(&rt).unwrap();
        let total_output_spikes: u32 = last_counts.iter().sum();

        println!("  Decisions per second: {:.1}", decisions_per_sec);
        println!("  Last decision spike counts: {:?}", last_counts);
        println!("  Total output spikes (last): {}", total_output_spikes);
        println!("  Action distribution:");
        for (i, count) in action_counts.iter().enumerate() {
            println!(
                "    Action {}: {} times ({:.1}%)",
                i,
                count,
                (*count as f64 / n_decisions as f64) * 100.0
            );
        }
        if total_output_spikes > 0 {
            println!("  Status: ✓ Outputs firing correctly");
        } else {
            println!("  Status: ⚠ No output spikes detected");
        }
    }
    println!();

    // ========================================================================
    // Test 5: Columnar vs Random Connectivity
    // ========================================================================
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│         TEST 5: Columnar vs Random Connectivity                │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("  Feedback insight: 'GPUs reward structure'");
    println!("  Columnar organization gives predictable memory access.\n");

    {
        let n_neurons = 200_000;
        let n_inputs = 20_000;
        let n_outputs = 10;

        // Random (layered) connectivity
        let synapses_random = generate_layered_csr(n_neurons, n_inputs, n_outputs, 20, 11111);
        let mut snn_random = GpuFusedSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            100,
            230,
            &synapses_random,
        )
        .unwrap();
        let rates: Vec<u8> = (0..n_inputs).map(|i| ((i * 3) % 200) as u8 + 50).collect();
        snn_random.set_input_rates(&rt, &rates).unwrap();

        // Warmup
        snn_random.tick_many(&rt, 10).unwrap();
        snn_random.synchronize(&rt).unwrap();

        // Benchmark random
        let bench_ticks = 500;
        let start = Instant::now();
        snn_random.tick_many(&rt, bench_ticks).unwrap();
        snn_random.synchronize(&rt).unwrap();
        let elapsed_random = start.elapsed();
        let throughput_random = (n_neurons * bench_ticks) as f64 / elapsed_random.as_secs_f64();

        // Columnar connectivity (aligned with GPU block size 256)
        let synapses_columnar = generate_columnar_csr(
            n_neurons, n_inputs, n_outputs, 256, // column_size = GPU block size
            0.1, // 10% intra-column density
            5,   // 5 inter-column connections
            22222,
        );
        let mut snn_columnar = GpuFusedSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            100,
            230,
            &synapses_columnar,
        )
        .unwrap();
        snn_columnar.set_input_rates(&rt, &rates).unwrap();

        // Warmup
        snn_columnar.tick_many(&rt, 10).unwrap();
        snn_columnar.synchronize(&rt).unwrap();

        // Benchmark columnar
        let start = Instant::now();
        snn_columnar.tick_many(&rt, bench_ticks).unwrap();
        snn_columnar.synchronize(&rt).unwrap();
        let elapsed_columnar = start.elapsed();
        let throughput_columnar = (n_neurons * bench_ticks) as f64 / elapsed_columnar.as_secs_f64();

        let speedup = throughput_columnar / throughput_random;

        println!(
            "  {:>20} {:>15} {:>15}",
            "Connectivity", "Synapses", "Throughput"
        );
        println!("  {:->20} {:->15} {:->15}", "", "", "");
        println!(
            "  {:>20} {:>15} {:>12.2}B/s",
            "Layered (random)",
            format_num(synapses_random.n_synapses()),
            throughput_random / 1e9
        );
        println!(
            "  {:>20} {:>15} {:>12.2}B/s",
            "Columnar (structured)",
            format_num(synapses_columnar.n_synapses()),
            throughput_columnar / 1e9
        );
        println!();
        println!("  Speedup: {:.2}x", speedup);

        // Verify outputs still fire with columnar connectivity
        snn_columnar.tick_many_with_counting(&rt, 100).unwrap();
        snn_columnar.synchronize(&rt).unwrap();
        let snapshot = snn_columnar.snapshot(&rt).unwrap();
        let total_output_spikes: u32 = snapshot.output_counts.iter().sum();
        println!(
            "  Columnar output spikes (100 ticks): {}",
            total_output_spikes
        );
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("  FUSED SNN KERNEL (EPIC 97-Style)");
    println!();
    println!("  Key features:");
    println!("    ✓ Zero PCIe transfers per tick");
    println!("    ✓ All state GPU-resident");
    println!("    ✓ Fused input + dynamics kernel");
    println!("    ✓ Fire-and-forget tick_many()");
    println!();
    println!("  Peak performance:");
    println!(
        "    - {:.2}B neuron-ticks/sec @ {} neurons",
        best_throughput / 1e9,
        best_config
    );
    println!();

    if best_throughput > 5e9 {
        println!("  🎉 Exceeded 5B neuron-ticks/sec - EPIC 97-style optimization working!");
    } else if best_throughput > 2e9 {
        println!("  ✓ Good performance - consider tuning block sizes for more speed.");
    } else {
        println!("  ⚠️  Performance lower than expected - check GPU utilization.");
    }
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}
