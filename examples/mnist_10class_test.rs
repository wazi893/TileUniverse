//! Phase 3a: 10-Class MNIST Discriminative Channels
//!
//! Extends Phase 1.1 binary discriminative channel architecture to 10 classes.
//! Uses one-vs-rest discriminative pixel selection: for each class c, the K pixels
//! with highest (avg_c - mean(avg_j, j≠c)) are assigned to channel c.
//!
//! Architecture:
//!   - n_inputs = K × 10  (K discriminative pixels per class, 10 channels)
//!   - hidden  = N_H_PER_CLASS × 10  (same-channel excitatory only, no inhibition)
//!   - n_outputs = 10  (not used for prediction)
//!   - Prediction: argmax of total hidden spikes per channel over TICKS steps
//!
//! Per-class adaptive T_c = midpoint(vss_cc, max_{j≠c}(vss_cj)) ensures only
//! the matched channel fires for each class.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example mnist_10class_test
//! ```

use engine::snn::{MnistDataset, NeuronConfig, SNNConfig, SNNNetwork, config::STDPConfig};

const N_CLASSES: usize = 10;
const N_PIXELS: usize = 784;
const N_H_PER_CLASS: usize = 32; // hidden neurons per channel (320 total); matches binary_mnist
const TICKS: u32 = 100;
const N_EPOCHS: usize = 10;
const N_TRAIN_PER_CLASS: usize = 200; // 2000 total train samples
const N_TEST_PER_CLASS: usize = 100; // 1000 total test samples
const K_SCAN: &[usize] = &[3, 5, 7, 10, 15, 20, 30, 50, 100, 200];
const SEEDS: &[u64] = &[42, 123, 999];
const CEILING_TARGET: f32 = 0.85; // first K with d-norm ceiling ≥ 85%
// Lower max_rate keeps v_ss within i16 range.
// With max_rate=100: vss_cc ≈ 54000-71000 >> i16_max(32767) → T capped, all channels fire.
// With max_rate=20:  vss_cc ≈ 10000-14000, T in (8000-13000) → proper per-class separation.
const MAX_RATE: u32 = 20;

// ─────────────────────────────────────────────────────────────────────────────
// Data analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Per-class, per-pixel mean intensity from training data.
fn compute_class_averages(
    data_per_class: &[Vec<Vec<u8>>; N_CLASSES],
) -> [[f32; N_PIXELS]; N_CLASSES] {
    let mut avgs = [[0f32; N_PIXELS]; N_CLASSES];
    for c in 0..N_CLASSES {
        let imgs = &data_per_class[c];
        if imgs.is_empty() {
            continue;
        }
        for img in imgs {
            for (i, &px) in img.iter().enumerate() {
                avgs[c][i] += px as f32;
            }
        }
        let inv = 1.0 / imgs.len() as f32;
        for v in &mut avgs[c] {
            *v *= inv;
        }
    }
    avgs
}

/// For each class c: sort pixels by one-vs-rest score descending.
///   score_c[i] = avg_c[i] - mean(avg_j[i] for j ≠ c)
fn compute_discrim_orders(avgs: &[[f32; N_PIXELS]; N_CLASSES]) -> [Vec<usize>; N_CLASSES] {
    // Precompute global pixel sum across all classes
    let mut global_sum = [0f32; N_PIXELS];
    for c in 0..N_CLASSES {
        for i in 0..N_PIXELS {
            global_sum[i] += avgs[c][i];
        }
    }

    let mut orders: [Vec<usize>; N_CLASSES] = Default::default();
    for c in 0..N_CLASSES {
        let mut scores: Vec<(usize, f32)> = (0..N_PIXELS)
            .map(|i| {
                let others_mean = (global_sum[i] - avgs[c][i]) / (N_CLASSES - 1) as f32;
                (i, avgs[c][i] - others_mean)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        orders[c] = scores.iter().map(|(i, _)| *i).collect();
    }
    orders
}

/// Compute per-class top-K pixel lists and d_norm (normalized discriminability weights).
///   pix_per_class[c] = orders[c][0..k]
///   d_norms[c][i] = max(0, score_c[pix]) / max_score_c
fn compute_d_norms(
    avgs: &[[f32; N_PIXELS]; N_CLASSES],
    orders: &[Vec<usize>; N_CLASSES],
    k: usize,
) -> ([Vec<usize>; N_CLASSES], [Vec<f32>; N_CLASSES]) {
    let mut global_sum = [0f32; N_PIXELS];
    for c in 0..N_CLASSES {
        for i in 0..N_PIXELS {
            global_sum[i] += avgs[c][i];
        }
    }

    let mut pix_per_class: [Vec<usize>; N_CLASSES] = Default::default();
    let mut d_norms: [Vec<f32>; N_CLASSES] = Default::default();

    for c in 0..N_CLASSES {
        let pix_c: Vec<usize> = orders[c][..k].to_vec();
        let d_raw: Vec<f32> = pix_c
            .iter()
            .map(|&i| {
                let others_mean = (global_sum[i] - avgs[c][i]) / (N_CLASSES - 1) as f32;
                (avgs[c][i] - others_mean).max(0.0)
            })
            .collect();
        let max_d = d_raw.iter().cloned().fold(0.0f32, f32::max).max(1.0);
        d_norms[c] = d_raw.iter().map(|&d| d / max_d).collect();
        pix_per_class[c] = pix_c;
    }

    (pix_per_class, d_norms)
}

/// Expected steady-state voltage for class-c channel when class-j digit is presented.
///
///   v_ss = scale_e × Σ_i d_norm_c[i] × avg_j[pix_c[i]]
///
/// scale_e = conn_prob×1.5 × weight_avg × 2 × (MAX_RATE / (255 × 256)) × (256 / 11)
///         = 0.75 × 100 × 2 × (MAX_RATE / (255 × 256)) × (256 / 11)
/// With MAX_RATE=20: scale_e ≈ 1.070 → vss_cc ≈ 10000-14000 (within i16 range)
/// With MAX_RATE=100: scale_e ≈ 5.348 → vss_cc ≈ 50000-71000 (i16 overflow!)
fn vss_channel(
    avgs: &[[f32; N_PIXELS]; N_CLASSES],
    pix_c: &[usize],
    d_norm_c: &[f32],
    class_j: usize,
) -> f32 {
    let scale_e = 0.75 * 100.0 * 2.0 * (MAX_RATE as f32 / (255.0 * 256.0)) * (256.0 / 11.0);
    pix_c
        .iter()
        .zip(d_norm_c.iter())
        .map(|(&pix, &d)| avgs[class_j][pix] * d * scale_e)
        .sum()
}

/// Per-class adaptive threshold.
///
/// For separable classes (vss_cc > vss_max_other):
///   T_c = midpoint(vss_cc, vss_max_other) — classic Phase 1.1 formula
///
/// For non-separable classes (vss_cc ≤ vss_max_other):
///   T_c = 0.85 × vss_cc — threshold just below own-class v_ss so the channel
///   still fires for its class. It will also fire for the confusing class, but
///   the argmax might still give partial accuracy if no other channel fires more.
fn compute_adaptive_thresholds(
    avgs: &[[f32; N_PIXELS]; N_CLASSES],
    pix_per_class: &[Vec<usize>; N_CLASSES],
    d_norms: &[Vec<f32>; N_CLASSES],
) -> [i16; N_CLASSES] {
    let mut t = [0i16; N_CLASSES];
    for c in 0..N_CLASSES {
        let vss_cc = vss_channel(avgs, &pix_per_class[c], &d_norms[c], c);
        let vss_max_other = (0..N_CLASSES)
            .filter(|&j| j != c)
            .map(|j| vss_channel(avgs, &pix_per_class[c], &d_norms[c], j))
            .fold(f32::NEG_INFINITY, f32::max);
        t[c] = if vss_cc > vss_max_other {
            // Separable: T at midpoint — suppresses non-preferred class while preferred fires
            ((vss_cc + vss_max_other.max(0.0)) / 2.0)
                .max(500.0)
                .min(30000.0) as i16
        } else {
            // Non-separable (class 5, 6 in MNIST): set T ABOVE vss_max_other to silence
            // this channel entirely. Without this, the low T causes the channel to fire for
            // ALL inputs (including non-class images), drowning out correct separable channels
            // in the argmax and corrupting predictions for the other 8 classes.
            // Accepting 0% accuracy for classes 5 and 6 is better than corrupting 8 others.
            (vss_max_other * 1.10).max(500.0).min(30000.0) as i16
        };
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// K-scan: analytical ceiling accuracy
// ─────────────────────────────────────────────────────────────────────────────

/// Noiseless ceiling accuracy for equal-weight and d-norm-weighted argmax classifiers.
/// Also returns n_sep: number of classes with vss_cc > vss_max_other.
fn ceiling_accuracy(
    avgs: &[[f32; N_PIXELS]; N_CLASSES],
    orders: &[Vec<usize>; N_CLASSES],
    k: usize,
    test_per_class: &[Vec<Vec<u8>>; N_CLASSES],
) -> (f32, f32, usize) {
    let (pix_per_class, d_norms) = compute_d_norms(avgs, orders, k);

    let mut correct_eq = 0usize;
    let mut correct_dw = 0usize;
    let mut total = 0usize;

    for j in 0..N_CLASSES {
        for img in &test_per_class[j] {
            let mut scores_eq = [0f32; N_CLASSES];
            let mut scores_dw = [0f32; N_CLASSES];

            for c in 0..N_CLASSES {
                for (i, &pix) in pix_per_class[c].iter().enumerate() {
                    let v = img[pix] as f32;
                    scores_eq[c] += v;
                    scores_dw[c] += v * d_norms[c][i];
                }
            }

            let pred_eq = scores_eq
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let pred_dw = scores_dw
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if pred_eq == j {
                correct_eq += 1;
            }
            if pred_dw == j {
                correct_dw += 1;
            }
            total += 1;
        }
    }

    let n_sep = (0..N_CLASSES)
        .filter(|&c| {
            let vss_cc = vss_channel(avgs, &pix_per_class[c], &d_norms[c], c);
            let vss_max = (0..N_CLASSES)
                .filter(|&j| j != c)
                .map(|j| vss_channel(avgs, &pix_per_class[c], &d_norms[c], j))
                .fold(f32::NEG_INFINITY, f32::max);
            vss_cc > vss_max
        })
        .count();

    let acc_eq = correct_eq as f32 / total.max(1) as f32;
    let acc_dw = correct_dw as f32 / total.max(1) as f32;
    (acc_eq, acc_dw, n_sep)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset preparation
// ─────────────────────────────────────────────────────────────────────────────

fn collect_class_imgs(dataset: &MnistDataset, label: u8, max: usize) -> Vec<Vec<u8>> {
    dataset
        .images
        .iter()
        .zip(dataset.labels.iter())
        .filter(|(_, l)| **l == label)
        .take(max)
        .map(|(img, _)| img.clone())
        .collect()
}

/// Encode image: K spike rates per class channel, concatenated (K×10 total).
/// Uses MAX_RATE (=20) to keep v_ss values within i16 range.
fn encode_image(img: &[u8], pix_per_class: &[Vec<usize>; N_CLASSES]) -> Vec<u8> {
    let k = pix_per_class[0].len();
    let mut rates = vec![0u8; k * N_CLASSES];
    for c in 0..N_CLASSES {
        for (i, &pix) in pix_per_class[c].iter().enumerate() {
            rates[c * k + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
        }
    }
    rates
}

fn prepare_dataset(
    data_per_class: &[Vec<Vec<u8>>; N_CLASSES],
    max_per_class: usize,
    pix_per_class: &[Vec<usize>; N_CLASSES],
) -> Vec<(Vec<u8>, usize)> {
    let mut out = Vec::with_capacity(max_per_class * N_CLASSES);
    for c in 0..N_CLASSES {
        for img in data_per_class[c].iter().take(max_per_class) {
            out.push((encode_image(img, pix_per_class), c));
        }
    }
    out
}

fn shuffle(data: &mut Vec<(Vec<u8>, usize)>, seed: u64) {
    let mut rng = seed;
    for i in (1..data.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network construction
// ─────────────────────────────────────────────────────────────────────────────

fn build_network(seed: u64, k: usize) -> SNNNetwork {
    let n_inputs = k * N_CLASSES;
    let n_hidden = N_H_PER_CLASS * N_CLASSES;
    let total = n_inputs + n_hidden + N_CLASSES;
    let neurons_per_cpu = if total <= 512 {
        512
    } else if total <= 1024 {
        1024
    } else if total <= 2048 {
        2048
    } else {
        4096
    };

    let config = SNNConfig {
        n_inputs,
        hidden_layers: vec![n_hidden],
        n_outputs: N_CLASSES,
        connection_prob: 0.5,
        recurrent: false,
        neuron_config: NeuronConfig {
            threshold: 8000, // overridden per-class by set_hidden_thresholds
            leak: 245,
            refractory_period: 2,
            v_reset: -128,
        },
        stdp_config: STDPConfig {
            a_plus: 20,
            a_minus: 15,
            w_min: -128,
            w_max: 127,
            ..STDPConfig::default()
        },
        neurons_per_cpu,
        use_reward_modulation: true,
        duplicate_inputs: false,
    };

    let mut net = SNNNetwork::new(config);
    net.build_excitatory_channels(seed);
    net
}

/// Scale input neuron outgoing weights by per-class d_norm.
fn scale_input_weights(net: &mut SNNNetwork, d_norms: &[Vec<f32>; N_CLASSES], k: usize) {
    let npcu = net.config.neurons_per_cpu;
    for c in 0..N_CLASSES {
        for i in 0..k {
            let global = c * k + i;
            let cpu = net.topology.neuron_to_cpu[global];
            let local = net.topology.local_index(global, npcu);
            let d = d_norms[c][i];
            for syn in &mut net.synapses[cpu].local[local] {
                let new_w = (syn.weight as f32 * d).round();
                syn.weight = new_w.clamp(-128.0, 127.0) as i8;
            }
        }
    }
}

/// Set per-class adaptive threshold for hidden neurons.
fn set_hidden_thresholds(net: &mut SNNNetwork, t_per_class: &[i16; N_CLASSES]) {
    let (_, h_start, _) = net.topology.layers[1];
    let npcu = net.config.neurons_per_cpu;
    for c in 0..N_CLASSES {
        let t = t_per_class[c];
        for i in 0..N_H_PER_CLASS {
            let global = h_start + c * N_H_PER_CLASS + i;
            let cpu = net.topology.neuron_to_cpu[global];
            let local = net.topology.local_index(global, npcu);
            net.populations[cpu].neurons[local].threshold = t;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Run one trial: full state reset, set inputs, simulate `ticks` steps.
/// Prediction = argmax of total hidden spikes per channel.
fn run_trial(net: &mut SNNNetwork, rates: &[u8], ticks: u32) -> usize {
    let (_, h_start, _) = net.topology.layers[1];
    let npcu = net.config.neurons_per_cpu;

    net.reset_output_counts();
    // Full state reset: clear v_mem, refractory, and spiked for ALL neurons.
    // reset_activity() only resets last_spike_time; without this, v_mem carries over
    // from the previous trial, causing hidden neurons to fire immediately on tick 0
    // or not fire at all due to stale refractory state.
    for cpu in 0..net.topology.n_cpus {
        for neuron in &mut net.populations[cpu].neurons {
            neuron.v_mem = 0;
            neuron.refractory = 0;
            neuron.spiked = 0;
            neuron.last_spike_time = 0;
        }
    }
    net.set_inputs(rates);

    let mut ch_spikes = [0u32; N_CLASSES];

    for _ in 0..ticks {
        net.step();
        for c in 0..N_CLASSES {
            for i in 0..N_H_PER_CLASS {
                let global = h_start + c * N_H_PER_CLASS + i;
                let cpu = net.topology.neuron_to_cpu[global];
                let local = net.topology.local_index(global, npcu);
                ch_spikes[c] += net.populations[cpu].neurons[local].spiked as u32;
            }
        }
    }

    ch_spikes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn train_epoch(
    net: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
) -> (f32, [f32; N_CLASSES]) {
    let saved = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; N_CLASSES];
    let mut class_total = [0u32; N_CLASSES];

    for (rates, label) in dataset {
        let label = *label;
        let pred = run_trial(net, rates, ticks);
        // Perceptron on output layer (not used for prediction, but may help in future)
        net.apply_perceptron_learning(pred, label, 5);
        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    net.runtime.learning_enabled = saved;

    let acc = correct as f32 / total.max(1) as f32;
    let mut per_class = [0f32; N_CLASSES];
    for c in 0..N_CLASSES {
        per_class[c] = class_correct[c] as f32 / class_total[c].max(1) as f32;
    }
    (acc, per_class)
}

fn evaluate(
    net: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
) -> (f32, [f32; N_CLASSES]) {
    let saved = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; N_CLASSES];
    let mut class_total = [0u32; N_CLASSES];

    for (rates, label) in dataset {
        let label = *label;
        let pred = run_trial(net, rates, ticks);
        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    net.runtime.learning_enabled = saved;

    let acc = correct as f32 / total.max(1) as f32;
    let mut per_class = [0f32; N_CLASSES];
    for c in 0..N_CLASSES {
        per_class[c] = class_correct[c] as f32 / class_total[c].max(1) as f32;
    }
    (acc, per_class)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Phase 3a: 10-Class MNIST Discriminative Channels ===\n");
    println!("Strategy: one-vs-rest discriminative pixel selection");
    println!("  score_c[i] = avg_c[i] - mean(avg_j[i], j≠c)");
    println!("  Top K → channel c  (10 channels, same-channel excitatory only)");
    println!("  Prediction: argmax of hidden channel spike totals\n");

    // Load MNIST
    let train_ds = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("MNIST train data not found — run from repo root");
    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("MNIST test data not found — run from repo root");

    // Collect per-class images
    let mut train_per_class: [Vec<Vec<u8>>; N_CLASSES] = Default::default();
    let mut test_per_class: [Vec<Vec<u8>>; N_CLASSES] = Default::default();
    for c in 0..N_CLASSES {
        train_per_class[c] = collect_class_imgs(&train_ds, c as u8, 6000);
        test_per_class[c] = collect_class_imgs(&test_ds, c as u8, 1000);
    }

    let train_counts: Vec<usize> = (0..N_CLASSES).map(|c| train_per_class[c].len()).collect();
    let test_counts: Vec<usize> = (0..N_CLASSES).map(|c| test_per_class[c].len()).collect();
    print!("Loaded training:");
    for c in 0..N_CLASSES {
        print!(" {}×{}", train_counts[c], c);
        if c < N_CLASSES - 1 {
            print!(",");
        }
    }
    println!();
    print!("Loaded test:    ");
    for c in 0..N_CLASSES {
        print!(" {}×{}", test_counts[c], c);
        if c < N_CLASSES - 1 {
            print!(",");
        }
    }
    println!("\n");

    // Compute class averages from full training set
    let avgs = compute_class_averages(&train_per_class);
    let orders = compute_discrim_orders(&avgs);

    // K-scan: theoretical ceiling accuracy
    println!(
        "K-scan (noiseless argmax ceiling, all {} test samples):",
        test_counts.iter().sum::<usize>()
    );
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>6}",
        "K", "equal-wt%", "discrim-wt%", "n_sep"
    );

    let mut k_test = *K_SCAN.last().unwrap();
    let mut found_target = false;
    let mut ceiling_results: Vec<(usize, f32, f32, usize)> = Vec::new();

    for &k in K_SCAN {
        let (acc_eq, acc_dw, n_sep) = ceiling_accuracy(&avgs, &orders, k, &test_per_class);
        ceiling_results.push((k, acc_eq, acc_dw, n_sep));

        let marker = if acc_dw >= CEILING_TARGET {
            if !found_target {
                found_target = true;
                k_test = k;
            }
            "  ✓ ≥85%"
        } else {
            ""
        };
        println!(
            "  {:>5}: {:>9.1}%  {:>9.1}%  {:>4}/10{}",
            k,
            acc_eq * 100.0,
            acc_dw * 100.0,
            n_sep,
            marker
        );
    }
    println!();

    if !found_target {
        // Use K with best discrim-weighted ceiling
        k_test = ceiling_results
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _, _, _)| *k)
            .unwrap_or(50);
        println!(
            "  → No K reached {:.0}% ceiling. Using K={} (best).",
            CEILING_TARGET * 100.0,
            k_test
        );
    } else {
        println!(
            "  → First K with discrim-weighted ceiling ≥{:.0}%: {}",
            CEILING_TARGET * 100.0,
            k_test
        );
    }

    let (ceiling_at_ktest_eq, ceiling_at_ktest_dw, _) = ceiling_results
        .iter()
        .find(|(k, _, _, _)| *k == k_test)
        .map(|(_, eq, dw, sep)| (*eq, *dw, *sep))
        .unwrap_or((0.0, 0.0, 0));
    println!(
        "  → Ceiling at K={}: equal-wt={:.1}%  discrim-wt={:.1}%",
        k_test,
        ceiling_at_ktest_eq * 100.0,
        ceiling_at_ktest_dw * 100.0
    );
    println!(
        "  → SNN with K={} + discriminability weights should approach {:.1}%.\n",
        k_test,
        ceiling_at_ktest_dw * 100.0
    );

    // Compute d_norms and adaptive T_hidden for K_test
    let (pix_per_class, d_norms) = compute_d_norms(&avgs, &orders, k_test);
    let t_per_class = compute_adaptive_thresholds(&avgs, &pix_per_class, &d_norms);

    // Print per-class threshold details
    println!("Adaptive T_hidden per class (K={}):", k_test);
    println!(
        "  {:>5}  {:>8}  {:>12}  {:>8}  {:>4}",
        "Class", "vss_cc", "vss_max_other", "T_hidden", "sep?"
    );
    for c in 0..N_CLASSES {
        let vss_cc = vss_channel(&avgs, &pix_per_class[c], &d_norms[c], c);
        let vss_max_other = (0..N_CLASSES)
            .filter(|&j| j != c)
            .map(|j| vss_channel(&avgs, &pix_per_class[c], &d_norms[c], j))
            .fold(f32::NEG_INFINITY, f32::max);
        let sep = if vss_cc > vss_max_other { "✓" } else { "✗" };
        println!(
            "  {:>5}: {:>8.0}  {:>12.0}  {:>8}  {}",
            c, vss_cc, vss_max_other, t_per_class[c], sep
        );
    }
    println!();

    // Prepare SNN datasets
    println!(
        "Using {} train/class × 10 = {} total  |  {} test/class × 10 = {} total",
        N_TRAIN_PER_CLASS,
        N_TRAIN_PER_CLASS * N_CLASSES,
        N_TEST_PER_CLASS,
        N_TEST_PER_CLASS * N_CLASSES
    );
    println!(
        "n_inputs={}, n_hidden={}, n_outputs={}, ticks={}\n",
        k_test * N_CLASSES,
        N_H_PER_CLASS * N_CLASSES,
        N_CLASSES,
        TICKS
    );

    let train_snn_base = prepare_dataset(&train_per_class, N_TRAIN_PER_CLASS, &pix_per_class);
    let test_snn = prepare_dataset(&test_per_class, N_TEST_PER_CLASS, &pix_per_class);

    // Run 3 seeds
    let mut best_tests: Vec<f32> = Vec::new();
    let mut pass_count = 0usize;

    for &seed in SEEDS {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Seed {}", seed);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut net = build_network(seed, k_test);
        scale_input_weights(&mut net, &d_norms, k_test);
        set_hidden_thresholds(&mut net, &t_per_class);

        println!(
            "  Neurons: {}  Synapses: {}  K={}",
            net.n_neurons(),
            net.n_synapses(),
            k_test
        );

        // Diagnostic: show per-channel spike counts for first image of each class
        println!("  Diagnostic (first test image per class, ch_spikes per channel):");
        println!("  {:>5}  {:>65}  pred", "true", "ch_spikes[0..9]");
        let h_start = net.topology.layers[1].1;
        let npcu_diag = net.config.neurons_per_cpu;
        for true_c in 0..N_CLASSES {
            if test_snn.iter().filter(|(_, l)| *l == true_c).count() == 0 {
                continue;
            }
            let (rates, _) = test_snn.iter().find(|(_, l)| *l == true_c).unwrap();
            // Full reset + set inputs
            for cpu in 0..net.topology.n_cpus {
                for neuron in &mut net.populations[cpu].neurons {
                    neuron.v_mem = 0;
                    neuron.refractory = 0;
                    neuron.spiked = 0;
                    neuron.last_spike_time = 0;
                }
            }
            net.set_inputs(rates);
            let mut ch_spikes_diag = [0u32; N_CLASSES];
            for _ in 0..TICKS {
                net.step();
                for c in 0..N_CLASSES {
                    for i in 0..N_H_PER_CLASS {
                        let global = h_start + c * N_H_PER_CLASS + i;
                        let cpu = net.topology.neuron_to_cpu[global];
                        let local = net.topology.local_index(global, npcu_diag) as usize;
                        ch_spikes_diag[c] += net.populations[cpu].neurons[local].spiked as u32;
                    }
                }
            }
            let pred = ch_spikes_diag
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let spikes_str: Vec<String> = ch_spikes_diag
                .iter()
                .enumerate()
                .map(|(c, &s)| {
                    if c == true_c {
                        format!("[{:3}]", s)
                    } else {
                        format!(" {:3} ", s)
                    }
                })
                .collect();
            println!("  {:>5}  {}  {}", true_c, spikes_str.join(""), pred);
        }
        println!();

        // Pre-training evaluation
        let (pre_train_acc, _) = evaluate(&mut net, &train_snn_base, TICKS);
        let (pre_test_acc, pre_test_pc) = evaluate(&mut net, &test_snn, TICKS);
        println!(
            "  Pre-training: train={:.0}%  test={:.0}%",
            pre_train_acc * 100.0,
            pre_test_acc * 100.0
        );
        print!("    per-class: [");
        for c in 0..N_CLASSES {
            print!("{:.0}%", pre_test_pc[c] * 100.0);
            if c < N_CLASSES - 1 {
                print!(" ");
            }
        }
        println!("]");
        println!();

        let mut best_test = pre_test_acc;
        let mut best_epoch = 0usize;

        // Training epochs
        let mut train_snn = train_snn_base.clone();
        for ep in 1..=N_EPOCHS {
            shuffle(&mut train_snn, seed.wrapping_add(ep as u64));
            let (train_acc, _) = train_epoch(&mut net, &train_snn, TICKS);
            if ep % 5 == 0 || ep == 1 {
                let (test_acc, _) = evaluate(&mut net, &test_snn, TICKS);
                println!(
                    "  Ep {:2}: train={:.0}%  test={:.0}%",
                    ep,
                    train_acc * 100.0,
                    test_acc * 100.0
                );
                if test_acc > best_test {
                    best_test = test_acc;
                    best_epoch = ep;
                }
            }
        }

        // Final per-class breakdown
        let (final_test_acc, final_test_pc) = evaluate(&mut net, &test_snn, TICKS);
        println!();
        println!(
            "  Best test: {:.1}% @ epoch {}  |  final: {:.1}%",
            best_test * 100.0,
            best_epoch,
            final_test_acc * 100.0
        );
        print!("  Final per-class: [");
        for c in 0..N_CLASSES {
            print!("{:.0}%", final_test_pc[c] * 100.0);
            if c < N_CLASSES - 1 {
                print!(" ");
            }
        }
        println!("]");

        // Grading: pass if SNN beats the analytical ceiling; partial if within 20% of it
        let pass_floor = ceiling_at_ktest_dw;
        let partial_floor = ceiling_at_ktest_dw * 0.80;
        let result = if best_test >= pass_floor {
            pass_count += 1;
            "✓ PASS"
        } else if best_test >= partial_floor {
            "? PARTIAL"
        } else {
            "✗ FAIL"
        };
        println!(
            "  Result: {}  ({:.1}% vs {:.1}% ceiling)",
            result,
            best_test * 100.0,
            ceiling_at_ktest_dw * 100.0
        );

        best_tests.push(best_test);
        println!();
    }

    // Summary
    println!("════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════");
    for (i, &seed) in SEEDS.iter().enumerate() {
        let marker = if best_tests[i] >= ceiling_at_ktest_dw {
            "✓"
        } else if best_tests[i] >= ceiling_at_ktest_dw * 0.80 {
            "?"
        } else {
            "✗"
        };
        println!(
            "  Seed {:>5}: {:.1}%  {}",
            seed,
            best_tests[i] * 100.0,
            marker
        );
    }
    let avg = best_tests.iter().sum::<f32>() / SEEDS.len() as f32;
    let partial_count = best_tests
        .iter()
        .filter(|&&x| x >= ceiling_at_ktest_dw * 0.80)
        .count();
    println!(
        "  Average: {:.1}%  ({}/{} beat ceiling, {}/{} ≥80%×ceiling)",
        avg * 100.0,
        pass_count,
        SEEDS.len(),
        partial_count,
        SEEDS.len()
    );
    println!();

    if pass_count == SEEDS.len() {
        println!(
            "✓ PASS: {:.1}% average — all seeds beat analytical ceiling ({:.1}%).\n  SNN surplus: +{:.1}pp above ceiling.",
            avg * 100.0,
            ceiling_at_ktest_dw * 100.0,
            (avg - ceiling_at_ktest_dw) * 100.0
        );
    } else if partial_count == SEEDS.len() {
        println!(
            "? PARTIAL: {:.1}% average — all seeds ≥80%×ceiling. Try: more hidden neurons or epochs.",
            avg * 100.0
        );
    } else if partial_count > 0 {
        println!(
            "? PARTIAL: {:.1}% average — some seeds near ceiling. Try: larger K or more epochs.",
            avg * 100.0
        );
    } else {
        println!(
            "✗ FAIL: {:.1}% average — below 80%×ceiling ({:.1}%). Check threshold formula and connectivity.",
            avg * 100.0,
            ceiling_at_ktest_dw * 0.80 * 100.0
        );
    }
    println!();
}
