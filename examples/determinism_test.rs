//! Determinism Test Suite for Fused SNN
//!
//! Critical invariant: Same seed + same inputs = identical outputs (bit-for-bit)
//!
//! This test suite verifies determinism at multiple scales before any learning
//! work begins. As the feedback states: "If you add plasticity without a
//! determinism harness, you'll be chasing ghost regressions forever."
//!
//! Run with: cargo run --release --features cuda --example determinism_test

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, is_cuda_available};
#[cfg(feature = "cuda")]
use engine::snn::{GpuFusedSNN, SNNSnapshot, generate_columnar_csr, generate_layered_csr};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           DETERMINISM TEST SUITE - Fused SNN Kernel              ║");
    println!("║                                                                  ║");
    println!("║  Invariant: same seed + same inputs = identical outputs          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        return;
    }

    #[cfg(feature = "cuda")]
    run_determinism_tests();
}

#[cfg(feature = "cuda")]
fn run_determinism_tests() {
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

    let mut all_passed = true;

    // Test at multiple scales
    let test_configs = [
        (1_000, 100, 10, "1K neurons"),
        (10_000, 1_000, 10, "10K neurons"),
        (100_000, 10_000, 10, "100K neurons"),
        (500_000, 50_000, 10, "500K neurons"),
    ];

    for (n_neurons, n_inputs, n_outputs, label) in test_configs {
        println!("┌────────────────────────────────────────────────────────────────┐");
        println!("│  Testing: {:>50} │", label);
        println!("└────────────────────────────────────────────────────────────────┘\n");

        let passed = test_determinism_at_scale(&rt, n_neurons, n_inputs, n_outputs);

        if passed {
            println!("  ✓ PASSED: {} determinism verified\n", label);
        } else {
            println!("  ✗ FAILED: {} determinism broken!\n", label);
            all_passed = false;
        }
    }

    // Test with different connectivity patterns
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  Testing: Columnar connectivity determinism                    │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    let passed = test_columnar_determinism(&rt);
    if passed {
        println!("  ✓ PASSED: Columnar connectivity determinism verified\n");
    } else {
        println!("  ✗ FAILED: Columnar connectivity determinism broken!\n");
        all_passed = false;
    }

    // Test decision-making determinism
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  Testing: Decision sequence determinism                        │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    let passed = test_decision_determinism(&rt);
    if passed {
        println!("  ✓ PASSED: Decision sequence determinism verified\n");
    } else {
        println!("  ✗ FAILED: Decision sequence determinism broken!\n");
        all_passed = false;
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════╗");
    if all_passed {
        println!("║  ALL DETERMINISM TESTS PASSED                                   ║");
        println!("║                                                                  ║");
        println!("║  The kernel is safe for learning work.                          ║");
    } else {
        println!("║  DETERMINISM TESTS FAILED                                        ║");
        println!("║                                                                  ║");
        println!("║  DO NOT proceed with learning until this is fixed!              ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn test_determinism_at_scale(
    rt: &CudaRuntime,
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
) -> bool {
    let seed = 42u64;
    let n_ticks = 100;

    // Create identical networks
    let synapses = generate_layered_csr(n_neurons, n_inputs, n_outputs, 20, seed);

    let mut snn1 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 1");
    let mut snn2 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 2");

    // Set identical input rates
    let rates: Vec<u8> = (0..n_inputs)
        .map(|i| ((i * 7 + 42) % 200) as u8 + 50)
        .collect();
    snn1.set_input_rates(rt, &rates).unwrap();
    snn2.set_input_rates(rt, &rates).unwrap();

    // Run both networks
    snn1.tick_many_with_counting(rt, n_ticks).unwrap();
    snn2.tick_many_with_counting(rt, n_ticks).unwrap();

    snn1.synchronize(rt).unwrap();
    snn2.synchronize(rt).unwrap();

    // Compare snapshots
    let snap1 = snn1.snapshot(rt).unwrap();
    let snap2 = snn2.snapshot(rt).unwrap();

    compare_snapshots(&snap1, &snap2, n_neurons)
}

#[cfg(feature = "cuda")]
fn test_columnar_determinism(rt: &CudaRuntime) -> bool {
    let n_neurons = 50_000;
    let n_inputs = 5_000;
    let n_outputs = 10;
    let seed = 12345u64;
    let n_ticks = 100;

    // Create identical columnar networks
    let synapses = generate_columnar_csr(
        n_neurons, n_inputs, n_outputs, 256, // column_size
        0.1, // intra_column_density
        5,   // inter_column_synapses
        seed,
    );

    let mut snn1 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 1");
    let mut snn2 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 2");

    let rates: Vec<u8> = (0..n_inputs)
        .map(|i| ((i * 13 + 7) % 200) as u8 + 50)
        .collect();
    snn1.set_input_rates(rt, &rates).unwrap();
    snn2.set_input_rates(rt, &rates).unwrap();

    snn1.tick_many_with_counting(rt, n_ticks).unwrap();
    snn2.tick_many_with_counting(rt, n_ticks).unwrap();

    snn1.synchronize(rt).unwrap();
    snn2.synchronize(rt).unwrap();

    let snap1 = snn1.snapshot(rt).unwrap();
    let snap2 = snn2.snapshot(rt).unwrap();

    compare_snapshots(&snap1, &snap2, n_neurons)
}

#[cfg(feature = "cuda")]
fn test_decision_determinism(rt: &CudaRuntime) -> bool {
    let n_neurons = 20_000;
    let n_inputs = 2_000;
    let n_outputs = 5;
    let seed = 99999u64;
    let n_decisions = 50;
    let ticks_per_decision = 20;

    let synapses = generate_layered_csr(n_neurons, n_inputs, n_outputs, 20, seed);

    let mut snn1 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 1");
    let mut snn2 = GpuFusedSNN::new(rt, n_neurons, n_inputs, n_outputs, 100, 230, &synapses)
        .expect("Failed to create SNN 2");

    let mut decisions1 = Vec::with_capacity(n_decisions);
    let mut decisions2 = Vec::with_capacity(n_decisions);
    let mut counts1 = Vec::with_capacity(n_decisions);
    let mut counts2 = Vec::with_capacity(n_decisions);

    for i in 0..n_decisions {
        // Same input pattern for both
        let rates: Vec<u8> = (0..n_inputs)
            .map(|j| ((j + i * 100) % 200) as u8 + 50)
            .collect();

        // SNN 1
        snn1.set_input_rates(rt, &rates).unwrap();
        snn1.reset_output_counts(rt).unwrap();
        snn1.tick_many_with_counting(rt, ticks_per_decision)
            .unwrap();
        snn1.synchronize(rt).unwrap();
        decisions1.push(snn1.get_decision(rt).unwrap());
        counts1.push(snn1.get_output_counts(rt).unwrap());

        // SNN 2
        snn2.set_input_rates(rt, &rates).unwrap();
        snn2.reset_output_counts(rt).unwrap();
        snn2.tick_many_with_counting(rt, ticks_per_decision)
            .unwrap();
        snn2.synchronize(rt).unwrap();
        decisions2.push(snn2.get_decision(rt).unwrap());
        counts2.push(snn2.get_output_counts(rt).unwrap());
    }

    // Compare decision sequences
    let decisions_match = decisions1 == decisions2;
    let counts_match = counts1 == counts2;

    if !decisions_match {
        println!("  ERROR: Decision sequences differ!");
        for i in 0..n_decisions {
            if decisions1[i] != decisions2[i] {
                println!("    Decision {}: {} vs {}", i, decisions1[i], decisions2[i]);
            }
        }
    }

    if !counts_match {
        println!("  ERROR: Output counts differ!");
        for i in 0..n_decisions {
            if counts1[i] != counts2[i] {
                println!("    Decision {}: {:?} vs {:?}", i, counts1[i], counts2[i]);
            }
        }
    }

    decisions_match && counts_match
}

#[cfg(feature = "cuda")]
fn compare_snapshots(snap1: &SNNSnapshot, snap2: &SNNSnapshot, n_neurons: usize) -> bool {
    let mut all_match = true;

    // Compare membrane potentials (bit-for-bit)
    let v_mem_match = snap1.v_mem == snap2.v_mem;
    if !v_mem_match {
        let diffs: Vec<_> = snap1
            .v_mem
            .iter()
            .zip(snap2.v_mem.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(10)
            .collect();
        println!(
            "  ERROR: Membrane potentials differ at {} locations",
            snap1
                .v_mem
                .iter()
                .zip(snap2.v_mem.iter())
                .filter(|(a, b)| a != b)
                .count()
        );
        for (i, (a, b)) in diffs {
            println!("    v_mem[{}]: {} vs {}", i, a, b);
        }
        all_match = false;
    } else {
        println!("  ✓ Membrane potentials match (bit-for-bit)");
    }

    // Compare spike flags (bit-for-bit)
    let spiked_match = snap1.spiked == snap2.spiked;
    if !spiked_match {
        let diffs: Vec<_> = snap1
            .spiked
            .iter()
            .zip(snap2.spiked.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(10)
            .collect();
        println!(
            "  ERROR: Spike flags differ at {} locations",
            snap1
                .spiked
                .iter()
                .zip(snap2.spiked.iter())
                .filter(|(a, b)| a != b)
                .count()
        );
        for (i, (a, b)) in diffs {
            println!("    spiked[{}]: {} vs {}", i, a, b);
        }
        all_match = false;
    } else {
        let total_spikes: usize = snap1.spiked.iter().map(|&s| s as usize).sum();
        println!("  ✓ Spike flags match ({} neurons spiked)", total_spikes);
    }

    // Compare output counts (bit-for-bit)
    let counts_match = snap1.output_counts == snap2.output_counts;
    if !counts_match {
        println!("  ERROR: Output counts differ!");
        println!("    Run 1: {:?}", snap1.output_counts);
        println!("    Run 2: {:?}", snap2.output_counts);
        all_match = false;
    } else {
        let total: u32 = snap1.output_counts.iter().sum();
        println!("  ✓ Output counts match ({} total spikes)", total);
    }

    // Compare tick counts
    if snap1.tick != snap2.tick {
        println!(
            "  ERROR: Tick counts differ: {} vs {}",
            snap1.tick, snap2.tick
        );
        all_match = false;
    } else {
        println!("  ✓ Tick counts match ({})", snap1.tick);
    }

    all_match
}
