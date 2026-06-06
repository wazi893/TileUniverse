//! Synthetic 4-Quadrant Pattern Test — Phase 1.3 Architecture Validation
//!
//! Creates 4 trivially-distinct patterns (one quadrant activated per class)
//! and trains the network. Expected result: >90% accuracy.
//!
//! Diagnostic questions:
//!   A) Do different patterns already produce different output firing? (pre-training)
//!   B) Does learning improve accuracy? (post-training)
//!   C) Is it consistent across seeds?
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example synthetic_quadrant_test
//! ```

use engine::snn::config::STDPConfig;
use engine::snn::{NeuronConfig, SNNConfig, SNNNetwork};

// ─────────────────────────────────────────────────────────────────────────────
// Data generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a 28×28 image with one quadrant fully lit, rest dark.
/// class 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
fn quadrant_image(class: usize) -> Vec<u8> {
    let (x0, x1, y0, y1) = match class {
        0 => (0, 14, 0, 14),
        1 => (14, 28, 0, 14),
        2 => (0, 14, 14, 28),
        _ => (14, 28, 14, 28),
    };
    let mut img = vec![0u8; 28 * 28];
    for y in y0..y1 {
        for x in x0..x1 {
            img[y * 28 + x] = 220;
        }
    }
    img
}

/// Add salt-and-pepper noise to an image (in-place), seeded deterministically.
fn add_noise(img: &mut Vec<u8>, noise_level: f32, seed: u64) {
    let mut rng = seed;
    for p in img.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((rng >> 33) as f32) / (u32::MAX as f32);
        if r < noise_level {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *p = ((rng >> 33) as f32 / u32::MAX as f32 * 255.0) as u8;
        }
    }
}

/// Reorder 28×28 pixels from row-major to quadrant-major order:
///   pixels 0-195   = top-left quadrant    (class 0's active region)
///   pixels 196-391 = top-right quadrant   (class 1's active region)
///   pixels 392-587 = bottom-left quadrant (class 2's active region)
///   pixels 588-783 = bottom-right quadrant(class 3's active region)
///
/// This alignment lets build_connectivity create spatially coherent
/// receptive fields: channel k's hidden neurons will connect primarily
/// to the k-th quadrant's pixels.
fn quadrant_reorder(img: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(784);
    // Quadrant order: (col_start, row_start)
    let regions = [(0usize, 0usize), (14, 0), (0, 14), (14, 14)];
    for &(x0, y0) in &regions {
        for y in y0..y0 + 14 {
            for x in x0..x0 + 14 {
                out.push(img[y * 28 + x]);
            }
        }
    }
    out
}

/// Convert pixel brightness to spike rate (0-255 → 0-200).
fn to_rates(img: &[u8]) -> Vec<u8> {
    img.iter()
        .map(|&p| ((p as u32 * 200) / 255) as u8)
        .collect()
}

/// Build dataset: n_per_class samples with added noise, shuffled.
fn build_dataset(n_per_class: usize, noise: f32, seed: u64) -> Vec<(Vec<u8>, usize)> {
    let mut data: Vec<(Vec<u8>, usize)> = Vec::with_capacity(n_per_class * 4);
    for class in 0..4usize {
        for i in 0..n_per_class {
            let mut img = quadrant_image(class);
            add_noise(
                &mut img,
                noise,
                seed.wrapping_add((class * 10000 + i) as u64),
            );
            // Reorder pixels so channel-aligned connectivity maps to correct spatial regions:
            // ch0=top-left(0-195), ch1=top-right(196-391), ch2=BL(392-587), ch3=BR(588-783)
            let img = quadrant_reorder(&img);
            data.push((img, class));
        }
    }
    // Fisher-Yates shuffle
    let mut rng = seed.wrapping_add(99999);
    for i in (1..data.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
    data
}

// ─────────────────────────────────────────────────────────────────────────────
// Network construction
// ─────────────────────────────────────────────────────────────────────────────

fn build_network(seed: u64) -> SNNNetwork {
    let config = SNNConfig {
        n_inputs: 28 * 28,       // 784 pixels (quadrant-reordered before use)
        hidden_layers: vec![64], // 16 hidden neurons per channel (64/4)
        n_outputs: 4,            // One per quadrant
        // Channel-aligned connectivity:
        //   same-channel prob = 0.5 × 1.5 = 0.75  → ~147 excitatory connections per hidden neuron
        //   cross-channel prob = 0.5 × 0.3 = 0.15 → inhibitory, suppresses wrong channels
        connection_prob: 0.5,
        recurrent: false,
        neuron_config: NeuronConfig::fast_spiking(),
        stdp_config: STDPConfig {
            a_plus: 20,
            a_minus: 15,
            w_min: -128,
            w_max: 127,
            ..STDPConfig::default()
        },
        neurons_per_cpu: 128,
        use_reward_modulation: true,
        duplicate_inputs: false,
    };

    let mut net = SNNNetwork::new(config);
    // Channel-aligned connectivity exploits the quadrant-reordered input ordering:
    // inputs 0-195 (top-left) → hidden[0..16] (ch 0) → output[0]
    // inputs 196-391 (top-right) → hidden[16..32] (ch 1) → output[1]  etc.
    net.build_connectivity(seed);
    net
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic: output differentiation before training
// ─────────────────────────────────────────────────────────────────────────────

/// Returns average spike counts per output neuron, broken down by class.
/// Shape: [class][output_neuron] → mean spike count over presentation.
fn measure_class_outputs(network: &mut SNNNetwork, ticks: u32) -> [[f32; 4]; 4] {
    let mut sums = [[0u64; 4]; 4];
    let mut counts = [0u32; 4];

    // One clean example per class (no noise, no state carry-over matters much here)
    for class in 0..4usize {
        network.reset_output_counts();
        // Quadrant-reorder before rate coding so channels align with spatial quadrants
        let rates = to_rates(&quadrant_reorder(&quadrant_image(class)));
        network.set_inputs(&rates);
        for _ in 0..ticks {
            network.step();
        }
        for out in 0..4 {
            sums[class][out] += network.output_counts[out] as u64;
        }
        counts[class] += 1;
    }

    let mut avgs = [[0f32; 4]; 4];
    for c in 0..4 {
        for o in 0..4 {
            avgs[c][o] = sums[c][o] as f32 / counts[c].max(1) as f32;
        }
    }
    avgs
}

/// Print a matrix of [class][output] spike counts.
fn print_output_matrix(avgs: &[[f32; 4]; 4], label: &str) {
    println!("  {} output spike counts [class × output neuron]:", label);
    print!("           ");
    for o in 0..4 {
        print!(" Out{}", o);
    }
    println!("   Winner");
    for c in 0..4 {
        print!("  Class {:?}: ", c);
        let mut winner = 0usize;
        let mut max_spikes = 0f32;
        for o in 0..4 {
            print!(" {:>5.1}", avgs[c][o]);
            if avgs[c][o] > max_spikes {
                max_spikes = avgs[c][o];
                winner = o;
            }
        }
        println!("   → Out{}", winner);
    }
}

/// Count how many classes have a distinct winner output.
fn count_distinct_winners(avgs: &[[f32; 4]; 4]) -> usize {
    let mut winners = [0usize; 4];
    for c in 0..4 {
        let mut best = 0;
        for o in 1..4 {
            if avgs[c][o] > avgs[c][best] {
                best = o;
            }
        }
        winners[c] = best;
    }
    let mut seen = [false; 4];
    let mut distinct = 0;
    for &w in &winners {
        if !seen[w] {
            distinct += 1;
            seen[w] = true;
        }
    }
    distinct
}

// ─────────────────────────────────────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────────────────────────────────────

/// Run one training epoch. Returns (accuracy, per-class accuracy).
fn train_epoch(
    network: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
) -> (f32, [f32; 4]) {
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; 4];
    let mut class_total = [0u32; 4];

    for (img, label) in dataset {
        network.reset_output_counts();

        let rates = to_rates(img);
        network.set_inputs(&rates);
        for _ in 0..ticks {
            network.step();
        }

        let pred = network.get_action();

        // Supervised STDP: always strengthen correct output, weaken wrong output.
        // Avoids R-STDP weight death caused by net-negative rewards on mispredictions.
        network.apply_learning_supervised(pred, *label);

        if pred == *label {
            correct += 1;
            class_correct[*label] += 1;
        }
        total += 1;
        class_total[*label] += 1;
    }

    let acc = correct as f32 / total.max(1) as f32;
    let mut per_class = [0f32; 4];
    for c in 0..4 {
        per_class[c] = class_correct[c] as f32 / class_total[c].max(1) as f32;
    }
    (acc, per_class)
}

/// Evaluate without learning.
fn evaluate(network: &mut SNNNetwork, dataset: &[(Vec<u8>, usize)], ticks: u32) -> f32 {
    // Disable learning temporarily
    let prev_enabled = network.runtime.learning_enabled;
    network.runtime.learning_enabled = false;

    let mut correct = 0usize;
    let mut total = 0usize;

    for (img, label) in dataset {
        network.reset_output_counts();
        let rates = to_rates(img);
        network.set_inputs(&rates);
        for _ in 0..ticks {
            network.step();
        }
        if network.get_action() == *label {
            correct += 1;
        }
        total += 1;
    }

    network.runtime.learning_enabled = prev_enabled;
    correct as f32 / total.max(1) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Sanity check: 4-input one-hot patterns → confirms learning rule works at all
// ─────────────────────────────────────────────────────────────────────────────

fn run_minimal_test() -> bool {
    println!("─────────────────────────────────────────────────────────────────────────");
    println!("SANITY CHECK: 4-input one-hot patterns (channel-aligned, single CPU)");
    println!("─────────────────────────────────────────────────────────────────────────");
    println!("Network: 4 → 8 → 4  |  Patterns: [200,0,0,0]→C0, [0,200,0,0]→C1, ...");
    println!("Connectivity: build_connectivity (channel-aligned, conn_prob=0.8)");
    println!("  → ch0: input[0]→hidden[0,1]→output[0] (same-channel excitatory)");
    println!("  → ch1: input[1]→hidden[2,3]→output[1]  etc.");

    // Build 4→8→4 network — total 16 neurons, all in CPU 0.
    // Uses channel-aligned connectivity so each input has a direct excitatory
    // pathway to the corresponding output via dedicated hidden neurons.
    // conn_prob=0.8 ensures same-channel prob = 0.8×1.5 = 1.2 → always connected.
    let config = SNNConfig {
        n_inputs: 4,
        hidden_layers: vec![8],
        n_outputs: 4,
        connection_prob: 0.8, // Same-channel: 0.8*1.5 = 1.2 → 100% connected
        recurrent: false,
        neuron_config: NeuronConfig::fast_spiking(), // threshold=200, leak=245
        stdp_config: STDPConfig {
            a_plus: 25,
            a_minus: 15,
            w_min: -128,
            w_max: 127,
            ..STDPConfig::default()
        },
        neurons_per_cpu: 128,
        use_reward_modulation: true,
        duplicate_inputs: false,
    };

    let ticks = 30; // More ticks for reliable output firing
    let n_per_class = 40; // 160 total samples per epoch
    let n_epochs = 100;
    let patterns: &[(Vec<u8>, usize)] = &[
        (vec![200, 0, 0, 0], 0),
        (vec![0, 200, 0, 0], 1),
        (vec![0, 0, 200, 0], 2),
        (vec![0, 0, 0, 200], 3),
    ];

    let mut pass_count = 0usize;
    let seeds = [42u64, 123, 999, 31415, 7777];

    for &seed in &seeds {
        let mut net = SNNNetwork::new(config.clone());
        // Channel-aligned: input[i] → hidden[2i..2i+2] → output[i]
        // Ensures each class has a dedicated excitatory pathway
        net.build_connectivity(seed);

        // Quick diagnostic after first tick to confirm hidden neurons fire
        let diag_rates = vec![200u8, 0, 0, 0];
        net.set_inputs(&diag_rates);
        net.step();
        let hidden_spikes: u32 = net.populations[0].neurons[4..12]
            .iter()
            .map(|n| n.spiked as u32)
            .sum();
        let output_spikes: u32 = net.populations[0].neurons[12..16]
            .iter()
            .map(|n| n.spiked as u32)
            .sum();
        if seed == 42 {
            println!(
                "  [seed 42 tick-1 diag] hidden fires: {}  output fires: {}",
                hidden_spikes, output_spikes
            );
        }
        // Reset state before real training
        for pop in &mut net.populations {
            for n in &mut pop.neurons {
                n.v_mem = 0;
                n.refractory = 0;
                n.spiked = 0;
                n.last_spike_time = 0;
            }
        }
        net.reset_output_counts();

        // Build training set (repeat each pattern n_per_class times, shuffled)
        let mut dataset: Vec<(Vec<u8>, usize)> = Vec::new();
        for _ in 0..n_per_class {
            for (rates, label) in patterns.iter() {
                dataset.push((rates.clone(), *label));
            }
        }
        let mut rng = seed.wrapping_add(42);
        for i in (1..dataset.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            dataset.swap(i, j);
        }

        let mut best = 0.0f32;
        for _epoch in 0..n_epochs {
            train_epoch(&mut net, &dataset, ticks);
            let eval_acc = evaluate(&mut net, &patterns.to_vec(), ticks);
            if eval_acc > best {
                best = eval_acc;
            }
            if best >= 0.9 {
                break;
            }
        }
        let ok = if best >= 0.9 {
            pass_count += 1;
            "✓"
        } else {
            "✗"
        };
        println!(
            "  Seed {:>5}: best eval = {:.0}%  {}",
            seed,
            best * 100.0,
            ok
        );
    }

    let passed = pass_count >= 4;
    println!("  Minimal test: {}/{} seeds pass", pass_count, seeds.len());
    if passed {
        println!("  → Supervised STDP learning works. Issue is with larger input space.\n");
    } else {
        println!("  → Learning rule itself is broken. Fix before scaling up.\n");
    }
    passed
}

fn main() {
    println!("=== Phase 1.3: Synthetic 4-Quadrant Architecture Validation ===\n");
    println!("Patterns: 4 trivially distinct spatial quadrants (28×28)");
    println!("Hypothesis: if >90% → architecture OK; if ~25% → WTA broken\n");

    // Run minimal sanity check first
    let minimal_ok = run_minimal_test();
    if !minimal_ok {
        println!("✗ Minimal sanity check failed — learning rule broken");
        println!("  Stopping here. Fix learning rule before testing larger patterns.");
        return;
    }
    println!();

    let n_per_class = 50; // 200 total samples
    let ticks = 30; // ticks per presentation
    let n_epochs = 25;
    let noise = 0.05f32; // 5% salt-and-pepper noise

    let seeds: &[u64] = &[42, 123, 999];
    let mut all_best: Vec<f32> = Vec::new();

    for &seed in seeds {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Seed {}", seed);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let train_data = build_dataset(n_per_class, noise, seed);
        let eval_data = build_dataset(20, 0.0, seed.wrapping_add(7777)); // Clean eval

        let mut net = build_network(seed);
        println!(
            "  Neurons: {}  Synapses: {}",
            net.n_neurons(),
            net.n_synapses()
        );

        // ── Pre-training diagnostic ──────────────────────────────────────────
        // Measure output differentiation with learning disabled
        let mut diag_net = net.clone();
        diag_net.runtime.learning_enabled = false;
        let pre_matrix = measure_class_outputs(&mut diag_net, ticks);
        print_output_matrix(&pre_matrix, "Pre-training");
        let pre_distinct = count_distinct_winners(&pre_matrix);
        println!("  Pre-training distinct winners: {}/4\n", pre_distinct);

        // ── Training loop ────────────────────────────────────────────────────
        let mut best_eval = 0.0f32;
        let mut best_epoch = 0;

        for epoch in 0..n_epochs {
            let (train_acc, per_class) = train_epoch(&mut net, &train_data, ticks);
            let eval_acc = evaluate(&mut net, &eval_data, ticks);

            if eval_acc > best_eval {
                best_eval = eval_acc;
                best_epoch = epoch + 1;
            }

            let log =
                epoch == 0 || (epoch + 1) % 5 == 0 || epoch == n_epochs - 1 || eval_acc >= 0.9;

            if log {
                println!(
                    "  Ep {:>2}: train={:.0}%  eval={:.0}%  |  C0={:.0}% C1={:.0}% C2={:.0}% C3={:.0}%",
                    epoch + 1,
                    train_acc * 100.0,
                    eval_acc * 100.0,
                    per_class[0] * 100.0,
                    per_class[1] * 100.0,
                    per_class[2] * 100.0,
                    per_class[3] * 100.0,
                );
                if eval_acc >= 0.9 {
                    break;
                } // early success
            }
        }

        // ── Post-training diagnostic ─────────────────────────────────────────
        let post_matrix = measure_class_outputs(&mut net, ticks);
        print_output_matrix(&post_matrix, "Post-training");
        let post_distinct = count_distinct_winners(&post_matrix);

        println!(
            "\n  Best eval: {:.0}% @ epoch {}  |  distinct winners: {} → {}",
            best_eval * 100.0,
            best_epoch,
            pre_distinct,
            post_distinct
        );

        let verdict = if best_eval >= 0.9 {
            "✓ PASS"
        } else if best_eval >= 0.5 {
            "? PARTIAL"
        } else {
            "✗ FAIL"
        };
        println!("  Result: {}\n", verdict);

        all_best.push(best_eval);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════");
    let avg = all_best.iter().sum::<f32>() / all_best.len() as f32;
    let pass_count = all_best.iter().filter(|&&a| a >= 0.9).count();
    let partial_count = all_best.iter().filter(|&&a| a >= 0.5 && a < 0.9).count();

    for (&seed, &best) in seeds.iter().zip(all_best.iter()) {
        println!(
            "  Seed {:>5}: {:.0}%  {}",
            seed,
            best * 100.0,
            if best >= 0.9 {
                "✓"
            } else if best >= 0.5 {
                "?"
            } else {
                "✗"
            }
        );
    }
    println!(
        "  Average: {:.0}%  ({}/{} pass, {}/{} partial)",
        avg * 100.0,
        pass_count,
        seeds.len(),
        partial_count,
        seeds.len()
    );

    println!();
    if pass_count == seeds.len() {
        println!("✓ ARCHITECTURE VALID");
        println!("  Network can learn simple spatial patterns with R-STDP.");
        println!("  → Next: Phase 1.1 Binary MNIST (0 vs 1)");
    } else if pass_count > 0 || partial_count > 0 {
        println!("? PARTIAL CONVERGENCE");
        println!("  Some seeds work. R-STDP is sensitive to initialisation.");
        println!("  → Next: Phase 2 — implement temporal competition WTA");
        println!("         OR Phase 3 — perceptron supervised fallback");
    } else {
        println!("✗ ARCHITECTURE BROKEN");
        println!("  Cannot learn even trivial 4-quadrant patterns.");
        println!("  → Must implement supervised learning (Strategy B)");
        println!("  → Try: apply_learning_competitive or perceptron updates");
    }
}
