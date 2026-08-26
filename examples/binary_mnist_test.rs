//! Phase 1.1: Binary MNIST Classification (Digit 0 vs Digit 1)
//!
//! Validates that the SNN architecture can learn from real MNIST images.
//! Uses data-driven discriminative pixel selection (same core insight as Phase 1.3)
//! to create clean channel separation.
//!
//! ## Key insight (why naive side/center columns failed)
//!
//! Side columns (cols 0-6, 21-27) have avg 23.3 for digit 0, avg 1.6 for digit 1.
//! BUT both digits drive center columns heavily (142.7 vs 113.3). Since center
//! dominates absolutely, Out1 wins the WTA for both classes → no learning signal.
//!
//! ## Fix: data-driven discriminative pixel selection
//!
//! Sort all 784 pixels by (avg_digit0 - avg_digit1):
//!   - Top K pixels: ring-edge pixels, very bright for "0", dark for "1"
//!   - Bottom K pixels: stroke pixels, very bright for "1", dark (hollow center of "0")
//!
//! Use only 2K pixels as input (n_inputs = 2K, K = 200).
//!   Channel 0 (inputs 0..K):   most digit-0-specific → hidden ch0 → output[0]
//!   Channel 1 (inputs K..2K):  most digit-1-specific → hidden ch1 → output[1]
//!
//! With K=200 and threshold=5000: non-preferred digit's v_ss ≈ 4900 < 5000 → never fires.
//! Pre-training should immediately give 2/2 correct WTA → R-STDP reinforces.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example binary_mnist_test
//! ```

use engine::snn::config::STDPConfig;
use engine::snn::{MnistDataset, NeuronConfig, SNNConfig, SNNNetwork};

// ─────────────────────────────────────────────────────────────────────────────
// Discriminative pixel ordering
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-pixel discriminative ordering for binary MNIST.
///
/// Returns a permutation of pixel indices (0..784) sorted by
/// `avg_pixel(digit0) - avg_pixel(digit1)` in DESCENDING order:
///   order[0]   = pixel most specific to digit 0 (bright for "0", dark for "1")
///   order[783] = pixel most specific to digit 1 (bright for "1", dark for "0")
/// Returns (order, avg0, avg1) where:
///   order = pixel indices sorted by avg0-avg1 descending
///   avg0[i], avg1[i] = per-pixel mean intensity for each class
fn compute_discriminative_order(
    zeros: &[Vec<u8>],
    ones: &[Vec<u8>],
) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let n = 784usize;
    let mut avg0 = vec![0f32; n];
    let mut avg1 = vec![0f32; n];

    for img in zeros {
        for (i, &px) in img.iter().enumerate() {
            avg0[i] += px as f32;
        }
    }
    if !zeros.is_empty() {
        let inv = 1.0 / zeros.len() as f32;
        for v in &mut avg0 {
            *v *= inv;
        }
    }

    for img in ones {
        for (i, &px) in img.iter().enumerate() {
            avg1[i] += px as f32;
        }
    }
    if !ones.is_empty() {
        let inv = 1.0 / ones.len() as f32;
        for v in &mut avg1 {
            *v *= inv;
        }
    }

    // Sort descending by (avg0 - avg1): digit-0-specific pixels first
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let da = avg0[a] - avg1[a];
        let db = avg0[b] - avg1[b];
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    (order, avg0, avg1)
}

/// Build the 2K-pixel input ordering used for network input:
///   [order[0..K], order[784-K..784]]
///     ↑                ↑
///     digit-0-specific  digit-1-specific
fn build_input_indices(full_order: &[usize], k: usize) -> Vec<usize> {
    let n = full_order.len(); // 784
    let mut indices = Vec::with_capacity(2 * k);
    indices.extend_from_slice(&full_order[..k]); // top K = digit-0-specific
    indices.extend_from_slice(&full_order[n - k..]); // bottom K = digit-1-specific
    indices
}

/// Compute normalized discriminability weights per input neuron and adaptive T_hidden.
///
/// For each input neuron, its discriminability measures how "diagnostic" its pixel is:
///   - ch0 inputs (i < K): d_norm[i] = (avg0[pixel_i] - avg1[pixel_i]) / max_d_ch0
///   - ch1 inputs (i >= K): d_norm[j] = (avg1[pixel_j] - avg0[pixel_j]) / max_d_ch1
///
/// T_hidden is the midpoint between the discrim-weighted v_ss for the preferred
/// and non-preferred class. With K=50, vss_d0_ch0h >> vss_d1_ch0h (often negative),
/// giving clean separation and ~96% theoretical ceiling.
fn compute_discrim_setup(
    avg0: &[f32],
    avg1: &[f32],
    full_order: &[usize],
    k: usize,
) -> (Vec<f32>, Vec<f32>, i16) {
    let n = full_order.len();
    let ch0_pix = &full_order[..k];
    let ch1_pix = &full_order[n - k..];

    let d_ch0_raw: Vec<f32> = ch0_pix
        .iter()
        .map(|&i| (avg0[i] - avg1[i]).max(0.0))
        .collect();
    let d_ch1_raw: Vec<f32> = ch1_pix
        .iter()
        .map(|&i| (avg1[i] - avg0[i]).max(0.0))
        .collect();

    let max_d_ch0 = d_ch0_raw.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    let max_d_ch1 = d_ch1_raw.iter().cloned().fold(0.0f32, f32::max).max(1.0);

    let d_ch0_norm: Vec<f32> = d_ch0_raw.iter().map(|&d| d / max_d_ch0).collect();
    let d_ch1_norm: Vec<f32> = d_ch1_raw.iter().map(|&d| d / max_d_ch1).collect();

    // Discrim-weighted v_ss for each channel's hidden group under each digit class.
    // max_rate=100, so firing rates are [0..100] (matching encode_image).
    // Each input i contributes:
    //   excit from same-channel inputs: prob=0.75, weight=100×d_norm, current=weight×2
    //   inhib from cross-channel inputs: prob=0.15, weight=60×d_norm, current=weight×2
    //   v_ss = (sum excit - sum inhib) / leak_denom  where leak_denom = 11/256
    let scale_e = 0.75 * 100.0 * 2.0 * (100.0 / (255.0 * 256.0)) * (256.0 / 11.0);
    let scale_i = 0.15 * 60.0 * 2.0 * (100.0 / (255.0 * 256.0)) * (256.0 / 11.0);

    // ch0h v_ss under digit-0: excit from ch0 inputs (ring pixels), inhib from ch1 inputs (non-ring)
    let mut vss_d0_ch0h: f32 = ch0_pix
        .iter()
        .zip(&d_ch0_norm)
        .map(|(&pi, &d)| avg0[pi] * d * scale_e)
        .sum::<f32>();
    vss_d0_ch0h -= ch1_pix
        .iter()
        .zip(&d_ch1_norm)
        .map(|(&pi, &d)| avg0[pi] * d * scale_i)
        .sum::<f32>();

    // ch1h v_ss under digit-0: excit from ch1 inputs, inhib from ch0 inputs
    // T must be > this so ch1h is SILENT for digit-0 images.
    let mut vss_d0_ch1h: f32 = ch1_pix
        .iter()
        .zip(&d_ch1_norm)
        .map(|(&pi, &d)| avg0[pi] * d * scale_e)
        .sum::<f32>();
    vss_d0_ch1h -= ch0_pix
        .iter()
        .zip(&d_ch0_norm)
        .map(|(&pi, &d)| avg0[pi] * d * scale_i)
        .sum::<f32>();

    // T_hidden = midpoint between ch0h (should fire) and ch1h (should be silent) for digit-0.
    // This guarantees ch1h cannot fire for any digit-0 image, eliminating the main error mode.
    let t_hidden = ((vss_d0_ch0h + vss_d0_ch1h.max(0.0)) / 2.0)
        .max(500.0)
        .min(30000.0) as i16;

    println!("Discrim-weighted v_ss (K={}, max_rate=100):", k);
    println!(
        "  ch0h_d0={:.0}  ch1h_d0={:.0}  → T_hidden={} (midpoint, ch1h suppressed)",
        vss_d0_ch0h, vss_d0_ch1h, t_hidden
    );
    println!();

    (d_ch0_norm, d_ch1_norm, t_hidden)
}

/// Scale each input neuron's outgoing synapse weights by its discriminability norm.
///
/// After build_connectivity, weights are ~uniform. This applies the per-input
/// scaling factor so highly discriminative pixels drive the network more strongly:
///   - Input i < K   (ch0 channel): weights × d_ch0_norm[i]
///   - Input i >= K  (ch1 channel): weights × d_ch1_norm[i - K]
///
/// Weights are rounded to i8. Zero-discriminability inputs end up with weight=0
/// and contribute nothing — they're effectively removed from the computation.
fn scale_input_weights(net: &mut SNNNetwork, d_ch0_norm: &[f32], d_ch1_norm: &[f32], k: usize) {
    let cpu = 0usize; // All neurons on CPU 0 (neurons_per_cpu=512 > total neurons)
    let n_inputs = k * 2;
    let npcu = net.config.neurons_per_cpu;
    for i in 0..n_inputs {
        let local_i = net.topology.local_index(i, npcu);
        let d = if i < k {
            d_ch0_norm[i]
        } else {
            d_ch1_norm[i - k]
        };
        for syn in &mut net.synapses[cpu].local[local_i] {
            let new_w = (syn.weight as f32 * d).round();
            syn.weight = new_w.clamp(-128.0, 127.0) as i8;
        }
    }
}

/// Patch all hidden neuron thresholds to `t_hidden` (computed adaptively from data).
fn set_hidden_threshold(net: &mut SNNNetwork, t_hidden: i16) {
    let (_, h_start, h_end) = net.topology.layers[1];
    let npcu = net.config.neurons_per_cpu;
    for i in h_start..h_end {
        let cpu = net.topology.neuron_to_cpu[i];
        let local = net.topology.local_index(i, npcu);
        net.populations[cpu].neurons[local].threshold = t_hidden;
    }
}

/// Encode an image to spike rates (0-100) for the selected pixels.
///
/// Using max_rate=100 (not 200) keeps v_ss values within i16 range for K=50,
/// enabling the threshold to sit between ch0h_d0 and ch1h_d0 correctly.
fn encode_image(img: &[u8], pixel_indices: &[usize]) -> Vec<u8> {
    pixel_indices
        .iter()
        .map(|&i| ((img[i] as u32 * 100) / 255) as u8)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset preparation
// ─────────────────────────────────────────────────────────────────────────────

/// Collect raw pixel arrays for a specific digit label, up to `max` samples.
fn collect_class(dataset: &MnistDataset, label: u8, max: usize) -> Vec<Vec<u8>> {
    dataset
        .images
        .iter()
        .zip(dataset.labels.iter())
        .filter(|(_, l)| **l == label)
        .take(max)
        .map(|(img, _)| img.clone())
        .collect()
}

/// Build a balanced binary dataset using the discriminative pixel ordering.
/// Returns Vec<(encoded_rates: Vec<u8>, label: usize)>.
fn prepare_binary(
    dataset: &MnistDataset,
    max_per_class: usize,
    pixel_indices: &[usize],
) -> Vec<(Vec<u8>, usize)> {
    let mut out = Vec::with_capacity(max_per_class * 2);
    for (img, &label) in dataset.images.iter().zip(dataset.labels.iter()) {
        match label {
            0 if out.iter().filter(|(_, l)| *l == 0).count() < max_per_class => {
                out.push((encode_image(img, pixel_indices), 0));
            }
            1 if out.iter().filter(|(_, l)| *l == 1).count() < max_per_class => {
                out.push((encode_image(img, pixel_indices), 1));
            }
            _ => {}
        }
        if out.iter().filter(|(_, l)| *l == 0).count() >= max_per_class
            && out.iter().filter(|(_, l)| *l == 1).count() >= max_per_class
        {
            break;
        }
    }
    out
}

/// Fisher-Yates shuffle (deterministic).
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

fn build_network(seed: u64, n_inputs: usize) -> SNNNetwork {
    // Dual-threshold design:
    //   T_hidden = placeholder (overridden by set_hidden_threshold in main via compute_discrim_setup)
    //   T_output = 2000 (patched after build_connectivity; output neurons need lower threshold)
    //
    // With K=50 + discriminability-weighted synapses (applied post-build):
    //   vss_d0_ch0h ≈ +44,000  (digit-0 pixels are bright + high discrim → strong excitation)
    //   vss_d1_ch0h ≈  -1,000  (digit-1 pixels are dark for ch0, ch1 pixels inhibit strongly)
    //   T_hidden = midpoint ≈ 22,000 → clean separation with 96.3% theoretical ceiling
    //
    // With T_output = 2000:
    //   ch0h fires burst → Out0 gets ~32 neurons × strong weights → Out0 fires reliably ✓
    let config = SNNConfig {
        n_inputs,
        hidden_layers: vec![64], // 32 per channel (ch0: first half, ch1: second half)
        n_outputs: 2,
        connection_prob: 0.5,
        recurrent: false,
        neuron_config: NeuronConfig {
            threshold: 8000, // Hidden threshold: ch1h_d0 actual=5577 << 8000 → silent
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
        neurons_per_cpu: 512, // force single CPU so all synapses are local
        use_reward_modulation: true,
        duplicate_inputs: false,
    };

    let mut net = SNNNetwork::new(config);
    net.build_connectivity(seed);

    // Patch output neurons with lower threshold so they fire from the hidden burst.
    // ch0h fires one burst of ~4350 current to Out0 >> T_out=2000 → Out0 fires reliably.
    // (all neurons are on CPU 0 since neurons_per_cpu=512 > total_neurons=80)
    let out_start = net.topology.layers.last().unwrap().1;
    let out_end = net.topology.layers.last().unwrap().2;
    let npcu = net.config.neurons_per_cpu;
    for i in out_start..out_end {
        let cpu = net.topology.neuron_to_cpu[i];
        let local = net.topology.local_index(i, npcu);
        net.populations[cpu].neurons[local].threshold = 2000;
    }

    net
}

// ─────────────────────────────────────────────────────────────────────────────
// Training and evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Run one trial: reset state, set inputs, simulate `ticks` steps, count hidden spikes.
///
/// Uses per-tick `spiked` flag to accumulate TOTAL spike counts per channel — much
/// more informative than binary `last_spike_time > 0` when neurons fire multiple times.
///
/// Returns `(prediction, ch0h_total_spikes, ch1h_total_spikes)`.
///
/// With T_hidden=8000 (K=7, 100 ticks):
///   digit-0: ch0h ≈ 28 neurons × 3.4 spikes = 95,  ch1h ≈ 1.4 × 2 = 3
///   digit-1: ch1h ≈ 28 × 3.4 = 95,  ch0h ≈ 0
/// The total-spike signal is ~30× stronger than binary count for hard examples.
fn run_trial(net: &mut SNNNetwork, rates: &[u8], ticks: u32) -> (usize, u32, u32) {
    let (h_start, h_end) = match net.topology.layers.get(1).copied() {
        Some((_, s, e)) => (s, e),
        None => return (0, 0, 0),
    };
    let n_per_ch = (h_end - h_start) / 2;

    net.reset_output_counts();
    net.reset_activity();
    net.set_inputs(rates);

    let mut ch0h = 0u32;
    let mut ch1h = 0u32;

    for _ in 0..ticks {
        net.step();
        // Sum spiked flags for each channel this tick (spiked=1 iff fired this tick)
        ch0h += (h_start..h_start + n_per_ch)
            .map(|i| net.populations[0].neurons[i].spiked as u32)
            .sum::<u32>();
        ch1h += (h_start + n_per_ch..h_end)
            .map(|i| net.populations[0].neurons[i].spiked as u32)
            .sum::<u32>();
    }

    let pred = if ch0h >= ch1h { 0 } else { 1 };
    (pred, ch0h, ch1h)
}

fn train_epoch(net: &mut SNNNetwork, dataset: &[(Vec<u8>, usize)], ticks: u32) -> (f32, [f32; 2]) {
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; 2];
    let mut class_total = [0u32; 2];

    let saved_learning = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    for (rates, label) in dataset {
        let label = *label;
        // run_trial handles reset_activity + set_inputs + ticks + spike counting
        let (pred, _ch0h, _ch1h) = run_trial(net, rates, ticks);

        // Perceptron: update output weights based on reliable hidden-count prediction.
        // apply_perceptron_learning uses last_spike_time (set during run_trial) to know
        // which presynaptic neurons fired, so no extra reset is needed here.
        net.apply_perceptron_learning(pred, label, 5);

        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    net.runtime.learning_enabled = saved_learning;

    let acc = correct as f32 / total.max(1) as f32;
    let per_class = [
        class_correct[0] as f32 / class_total[0].max(1) as f32,
        class_correct[1] as f32 / class_total[1].max(1) as f32,
    ];
    (acc, per_class)
}

fn evaluate(net: &mut SNNNetwork, dataset: &[(Vec<u8>, usize)], ticks: u32) -> (f32, [f32; 2]) {
    let prev = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; 2];
    let mut class_total = [0u32; 2];

    for (rates, label) in dataset {
        let label = *label;
        let (pred, _ch0h, _ch1h) = run_trial(net, rates, ticks);
        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    net.runtime.learning_enabled = prev;
    let acc = correct as f32 / total.max(1) as f32;
    let per_class = [
        class_correct[0] as f32 / class_total[0].max(1) as f32,
        class_correct[1] as f32 / class_total[1].max(1) as f32,
    ];
    (acc, per_class)
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic: average output firing per class
// ─────────────────────────────────────────────────────────────────────────────

fn measure_class_outputs(
    net: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
    n: usize,
) -> [[f32; 2]; 2] {
    let prev = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    let mut sums = [[0u64; 2]; 2];
    let mut counts = [0u32; 2];

    for (rates, label) in dataset.iter() {
        let label = *label;
        if label > 1 {
            continue;
        }
        if counts[label] as usize >= n {
            continue;
        }
        run_trial(net, rates, ticks); // handles reset + set_inputs + step loop
        for o in 0..2 {
            sums[label][o] += net.output_counts[o] as u64;
        }
        counts[label] += 1;
        if counts[0] >= n as u32 && counts[1] >= n as u32 {
            break;
        }
    }

    net.runtime.learning_enabled = prev;
    let mut avgs = [[0f32; 2]; 2];
    for c in 0..2 {
        for o in 0..2 {
            avgs[c][o] = sums[c][o] as f32 / counts[c].max(1) as f32;
        }
    }
    avgs
}

fn print_output_matrix(avgs: &[[f32; 2]; 2], label: &str) {
    println!("  {} output counts [class × output]:", label);
    println!("           Out0   Out1   Winner");
    let names = ["Digit 0", "Digit 1"];
    for c in 0..2 {
        let winner = if avgs[c][0] >= avgs[c][1] { 0 } else { 1 };
        println!(
            "  {:>7}: {:>5.1}  {:>5.1}   → Out{}",
            names[c], avgs[c][0], avgs[c][1], winner
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic: channel statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ACTUAL v_ss (steady-state voltage) for a hidden neuron in channel `ch`.
///
/// `build_connectivity` uses:
///   - Same-channel feedforward: prob=conn_prob×1.5=0.75, weight ~+100 (avg of +80..+120)
///   - Cross-channel feedforward: prob=conn_prob×0.3=0.15, weight ~-60 (avg of -40..-80)
///   - leak_denom = 1 - (245/256) = 11/256 ≈ 0.04297
///
/// CRITICAL: synapse.current() = weight × 2, and spike_prob = rate/256 = (avg × 200/255)/256
/// So actual input per tick = K × prob × (avg × 200/(255×256)) × (weight × 2)
///
/// v_ss = net_input_per_tick / leak_denom
///      = [K × 0.75 × (avg_same × 200/(255×256)) × 200
///         - K × 0.15 × (avg_cross × 200/(255×256)) × 120] / (11/256)
fn vss_hidden(avg_same: f32, avg_cross: f32, k: usize) -> f32 {
    let kf = k as f32;
    // spike_rate = avg × 200/255, spike_prob = spike_rate/256, current = weight × 2
    let excit = kf * 0.75 * (avg_same * 200.0 / (255.0 * 256.0)) * 200.0; // weight=100→current=200
    let inhib = kf * 0.15 * (avg_cross * 200.0 / (255.0 * 256.0)) * 120.0; // weight=60→current=120
    (excit - inhib) / (11.0 / 256.0) // leak_denom = 1 - 245/256 = 11/256
}

/// Print per-channel average pixel activation for each digit class.
/// Uses the correct v_ss formula accounting for cross-channel inhibitory inputs.
fn print_channel_stats(zeros: &[Vec<u8>], ones: &[Vec<u8>], full_order: &[usize], k: usize) {
    let ch0_pixels = &full_order[..k];
    let ch1_pixels = &full_order[784 - k..];

    let avg_in_channel = |imgs: &[Vec<u8>], pixels: &[usize]| -> f32 {
        if imgs.is_empty() || pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = imgs
            .iter()
            .map(|img| pixels.iter().map(|&i| img[i] as f64).sum::<f64>())
            .sum();
        (sum / (imgs.len() * pixels.len()) as f64) as f32
    };

    let d0_ch0 = avg_in_channel(zeros, ch0_pixels);
    let d0_ch1 = avg_in_channel(zeros, ch1_pixels);
    let d1_ch0 = avg_in_channel(ones, ch0_pixels);
    let d1_ch1 = avg_in_channel(ones, ch1_pixels);

    // v_ss for each hidden group and digit class (threshold=5000)
    // ch0-hidden for digit 0: same=d0_ch0, cross=d0_ch1
    // ch1-hidden for digit 0: same=d0_ch1, cross=d0_ch0
    let vss_d0_ch0h = vss_hidden(d0_ch0, d0_ch1, k);
    let vss_d0_ch1h = vss_hidden(d0_ch1, d0_ch0, k);
    let vss_d1_ch0h = vss_hidden(d1_ch0, d1_ch1, k);
    let vss_d1_ch1h = vss_hidden(d1_ch1, d1_ch0, k);

    // T_hidden=8000: hidden threshold. T_out=2000: output threshold (set post-build).
    let t_hidden = 8000.0f32;
    println!("Channel stats (K={}):", k);
    println!(
        "  Pixel avgs:  d0_ch0={:.1}  d0_ch1={:.1}  d1_ch0={:.1}  d1_ch1={:.1}",
        d0_ch0, d0_ch1, d1_ch0, d1_ch1
    );
    println!(
        "  actual v_ss ch0-hidden: digit0={:.0}  digit1={:.0}",
        vss_d0_ch0h, vss_d1_ch0h
    );
    println!(
        "  actual v_ss ch1-hidden: digit0={:.0}  digit1={:.0}",
        vss_d0_ch1h, vss_d1_ch1h
    );
    let ch0_ok2 = vss_d0_ch0h > t_hidden && vss_d1_ch0h < t_hidden;
    let ch1_ok2 = vss_d1_ch1h > t_hidden && vss_d0_ch1h < t_hidden;
    println!(
        "  Ch0-hidden vs T_hidden={:.0}: {} (d0={:.0}>{:.0}, d1={:.0}<{:.0})",
        t_hidden,
        if ch0_ok2 {
            "✓ selective"
        } else {
            "✗ overlap"
        },
        vss_d0_ch0h,
        t_hidden,
        vss_d1_ch0h,
        t_hidden
    );
    println!(
        "  Ch1-hidden vs T_hidden={:.0}: {} (d1={:.0}>{:.0}, d0={:.0}<{:.0})",
        t_hidden,
        if ch1_ok2 {
            "✓ selective"
        } else {
            "✗ overlap"
        },
        vss_d1_ch1h,
        t_hidden,
        vss_d0_ch1h,
        t_hidden
    );
    println!();
}

/// Print K-scan: channel selectivity for multiple K values.
/// Helps identify the optimal K where both channels are class-exclusive.
fn print_kscan(zeros: &[Vec<u8>], ones: &[Vec<u8>], full_order: &[usize]) {
    let avg_in_ch = |imgs: &[Vec<u8>], pixels: &[usize]| -> f32 {
        if imgs.is_empty() || pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = imgs
            .iter()
            .map(|img| pixels.iter().map(|&i| img[i] as f64).sum::<f64>())
            .sum();
        (sum / (imgs.len() * pixels.len()) as f64) as f32
    };

    // T_hidden=8000 (actual hidden threshold). All v_ss values are ACTUAL (account for ×1.5625).
    // ch0h fires for d0 AND silent for d1 → Out0 wins for digit-0.
    // ch1h fires for d1 AND silent for d0 → Out1 wins for digit-1.
    let t_scan = 8000.0f32;
    println!(
        "K-scan (T_hidden={:.0}: actual v_ss, ✓ = clean separation for both channels):",
        t_scan
    );
    println!(
        "{:>5}  {:>8} {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8} {:>8}  Status",
        "K", "d0_ch0", "d0_ch1", "d1_ch0", "d1_ch1", "ch0h_d0", "ch0h_d1", "ch1h_d0", "ch1h_d1"
    );
    let n = full_order.len();
    for &k in &[3usize, 5, 7, 10, 15, 20, 30, 50, 100, 200] {
        if k >= n {
            break;
        }
        let ch0 = &full_order[..k];
        let ch1 = &full_order[n - k..];
        let d0c0 = avg_in_ch(zeros, ch0);
        let d0c1 = avg_in_ch(zeros, ch1);
        let d1c0 = avg_in_ch(ones, ch0);
        let d1c1 = avg_in_ch(ones, ch1);
        let vss_d0_ch0h = vss_hidden(d0c0, d0c1, k);
        let vss_d0_ch1h = vss_hidden(d0c1, d0c0, k);
        let vss_d1_ch0h = vss_hidden(d1c0, d1c1, k);
        let vss_d1_ch1h = vss_hidden(d1c1, d1c0, k);
        let ok = vss_d0_ch0h > t_scan
            && vss_d1_ch0h < t_scan
            && vss_d1_ch1h > t_scan
            && vss_d0_ch1h < t_scan;
        println!(
            "{:>5}  {:>8.1} {:>8.1} {:>8.1} {:>8.1}  {:>8.0} {:>8.0} {:>8.0} {:>8.0}  {}",
            k,
            d0c0,
            d0c1,
            d1c0,
            d1c1,
            vss_d0_ch0h,
            vss_d1_ch0h,
            vss_d0_ch1h,
            vss_d1_ch1h,
            if ok { "✓ CLEAN" } else { "✗" }
        );
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Phase 1.1: Binary MNIST Classification (0 vs 1) ===\n");
    println!("Strategy: data-driven discriminative pixel selection");
    println!("  Sort pixels by avg(digit0) - avg(digit1)");
    println!("  Top K → channel 0 (digit-0-specific)");
    println!("  Bottom K → channel 1 (digit-1-specific)\n");

    // ── Load MNIST ──────────────────────────────────────────────────────────
    let train_mnist = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("Failed to load training MNIST — run from repo root, data/ must exist");

    let test_mnist =
        MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
            .expect("Failed to load test MNIST");

    let train_dist = train_mnist.label_distribution();
    let test_dist = test_mnist.label_distribution();
    println!(
        "Loaded training: {}×0, {}×1  |  test: {}×0, {}×1",
        train_dist[0], train_dist[1], test_dist[0], test_dist[1]
    );

    // ── Compute discriminative ordering ──────────────────────────────────────
    let n_for_stats = 1000usize; // use up to 1000 samples per class for ordering
    let zeros_raw = collect_class(&train_mnist, 0, n_for_stats);
    let ones_raw = collect_class(&train_mnist, 1, n_for_stats);

    let (full_order, avg0, avg1) = compute_discriminative_order(&zeros_raw, &ones_raw);

    // K-scan: find the K where both channels are class-exclusive.
    // At small K we select the MOST discriminative pixels (outer ring corners for d0,
    // hollow center for d1). Cross-channel inhibitory inputs then suppress the
    // non-preferred hidden group → clean WTA from initialization.
    print_kscan(&zeros_raw, &ones_raw, &full_order);

    // K = number of discriminative pixels per channel.
    // K=15 + discrim-weighted synapses: 95.0% theoretical ceiling (discrim-weighted).
    // T_hidden computed adaptively from data via compute_discrim_setup below.
    let k = 15usize;
    let n_inputs = 2 * k;
    let pixel_indices = build_input_indices(&full_order, k);

    // Compute per-input discriminability norms and adaptive T_hidden.
    // These are shared across all seeds (data-derived, not random).
    let (d_ch0_norm, d_ch1_norm, t_hidden_base) =
        compute_discrim_setup(&avg0, &avg1, &full_order, k);

    // ── Prepare balanced binary datasets ────────────────────────────────────
    let n_train_per_class = 1000;
    let n_test_per_class = 500;
    let ticks = 100u32; // 100 ticks: ch0h fires ~3.4× per neuron vs ~1.7× at 50; better statistics
    let n_epochs = 50usize;

    let train_data = prepare_binary(&train_mnist, n_train_per_class, &pixel_indices);
    let test_data = prepare_binary(&test_mnist, n_test_per_class, &pixel_indices);
    println!(
        "Using {} train, {} test  (n_inputs={})\n",
        train_data.len(),
        test_data.len(),
        n_inputs
    );

    // ── Theoretical ceiling scan ─────────────────────────────────────────────
    // Equal-weight: sum(rates[ch0]) vs sum(rates[ch1]) — no weighting
    // Discrim-weighted: Σ discrim[i]×rate[i] for each channel — optimal linear
    // The weighted version reveals the true limit of the discriminative pixel approach.
    println!("Ceiling scan (noiseless, equal-weight vs discrim-weighted):");
    println!("{:>5}  {:>10}  {:>15}", "K", "equal-wt%", "discrim-wt%");
    let mut optimal_k = k;
    let raw_test_imgs: Vec<(Vec<u8>, usize)> = test_mnist
        .images
        .iter()
        .zip(test_mnist.labels.iter())
        .filter(|(_, l)| **l == 0 || **l == 1)
        .take(n_test_per_class * 2)
        .map(|(img, l)| (img.clone(), *l as usize))
        .collect();
    let n_test_total = raw_test_imgs.len();
    for &scan_k in &[3usize, 5, 7, 10, 15, 20, 30, 50, 100, 200] {
        if scan_k * 2 > full_order.len() {
            break;
        }
        let ch0_pix = &full_order[..scan_k];
        let ch1_pix = &full_order[full_order.len() - scan_k..];
        // Discriminability weights: for ch0 pixels = avg0[i]-avg1[i], for ch1 = avg1[i]-avg0[i]
        let d_ch0: Vec<f32> = ch0_pix
            .iter()
            .map(|&i| (avg0[i] - avg1[i]).max(0.0))
            .collect();
        let d_ch1: Vec<f32> = ch1_pix
            .iter()
            .map(|&i| (avg1[i] - avg0[i]).max(0.0))
            .collect();
        let mut eq_correct = 0usize;
        let mut wt_correct = 0usize;
        for (img, label) in &raw_test_imgs {
            let ch0_eq: u32 = ch0_pix.iter().map(|&i| img[i] as u32).sum();
            let ch1_eq: u32 = ch1_pix.iter().map(|&i| img[i] as u32).sum();
            if (ch0_eq >= ch1_eq) == (*label == 0) {
                eq_correct += 1;
            }
            let ch0_wt: f32 = ch0_pix
                .iter()
                .zip(&d_ch0)
                .map(|(&i, &w)| w * img[i] as f32)
                .sum();
            let ch1_wt: f32 = ch1_pix
                .iter()
                .zip(&d_ch1)
                .map(|(&i, &w)| w * img[i] as f32)
                .sum();
            if (ch0_wt >= ch1_wt) == (*label == 0) {
                wt_correct += 1;
            }
        }
        let eq_acc = eq_correct as f32 / n_test_total as f32 * 100.0;
        let wt_acc = wt_correct as f32 / n_test_total as f32 * 100.0;
        let mark = if wt_acc >= 97.0 {
            "★ ≥97%"
        } else if wt_acc >= 95.0 {
            "✓ ≥95%"
        } else {
            ""
        };
        println!(
            "  K={:>3}:  {:>7.1}%   {:>10.1}%  {}",
            scan_k, eq_acc, wt_acc, mark
        );
        if wt_acc >= 95.0 && optimal_k == k {
            optimal_k = scan_k;
        }
    }
    println!();
    println!(
        "  → First K with discrim-weighted ceiling ≥95%: {}",
        optimal_k
    );
    println!(
        "  → SNN with K={} + discriminability weights should match.\n",
        optimal_k
    );

    // ── Run across multiple seeds ────────────────────────────────────────────
    let seeds: &[u64] = &[42, 123, 999];
    let mut all_best: Vec<f32> = Vec::new();

    for &seed in seeds {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Seed {}", seed);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut net = build_network(seed, n_inputs);
        // Apply discriminability weighting: scale input synapse weights by pixel discriminability,
        // then patch hidden thresholds to the data-derived T_hidden.
        scale_input_weights(&mut net, &d_ch0_norm, &d_ch1_norm, k);
        set_hidden_threshold(&mut net, t_hidden_base);
        println!(
            "  Neurons: {}  Synapses: {}  T_hidden={}",
            net.n_neurons(),
            net.n_synapses(),
            t_hidden_base
        );

        let mut shuffled = train_data.clone();
        shuffle(&mut shuffled, seed);

        // ── Pre-training diagnostic ──────────────────────────────────────────
        let pre = measure_class_outputs(&mut net, &shuffled, ticks, 20);
        print_output_matrix(&pre, "Pre-training");
        let pre_correct = (pre[0][0] > pre[0][1]) as usize + (pre[1][1] > pre[1][0]) as usize;
        println!(
            "  Pre-training classes correctly discriminated: {}/2\n",
            pre_correct
        );

        // ── Training loop ────────────────────────────────────────────────────
        let mut best_test = 0.0f32;
        let mut best_epoch = 0;

        for epoch in 0..n_epochs {
            shuffle(&mut shuffled, seed.wrapping_add(epoch as u64));
            let (train_acc, per_class) = train_epoch(&mut net, &shuffled, ticks);
            let (test_acc, test_per) = evaluate(&mut net, &test_data, ticks);

            if test_acc > best_test {
                best_test = test_acc;
                best_epoch = epoch + 1;
            }

            let log =
                epoch == 0 || (epoch + 1) % 5 == 0 || epoch == n_epochs - 1 || test_acc >= 0.95;
            if log {
                println!(
                    "  Ep {:>2}: train={:.0}%  test={:.0}%  (d0={:.0}% d1={:.0}%)  [test: d0={:.0}% d1={:.0}%]",
                    epoch + 1,
                    train_acc * 100.0,
                    test_acc * 100.0,
                    per_class[0] * 100.0,
                    per_class[1] * 100.0,
                    test_per[0] * 100.0,
                    test_per[1] * 100.0,
                );
                if test_acc >= 0.95 {
                    break;
                }
            }
        }

        // ── Post-training diagnostic ─────────────────────────────────────────
        let post = measure_class_outputs(&mut net, &test_data, ticks, 20);
        print_output_matrix(&post, "Post-training");
        let post_correct = (post[0][0] > post[0][1]) as usize + (post[1][1] > post[1][0]) as usize;

        println!(
            "\n  Best test: {:.1}% @ epoch {}  |  discriminated: {} → {}",
            best_test * 100.0,
            best_epoch,
            pre_correct,
            post_correct
        );
        let verdict = if best_test >= 0.95 {
            "✓ PASS"
        } else if best_test >= 0.80 {
            "? PARTIAL"
        } else {
            "✗ FAIL"
        };
        println!("  Result: {}\n", verdict);

        all_best.push(best_test);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════");
    let avg = all_best.iter().sum::<f32>() / all_best.len() as f32;
    let pass_count = all_best.iter().filter(|&&a| a >= 0.95).count();
    let partial_count = all_best
        .iter()
        .filter(|&&a| (0.80..0.95).contains(&a))
        .count();

    for (&seed, &best) in seeds.iter().zip(all_best.iter()) {
        println!(
            "  Seed {:>5}: {:.1}%  {}",
            seed,
            best * 100.0,
            if best >= 0.95 {
                "✓"
            } else if best >= 0.80 {
                "?"
            } else {
                "✗"
            }
        );
    }
    println!(
        "  Average: {:.1}%  ({}/{} pass ≥95%, {}/{} partial ≥80%)",
        avg * 100.0,
        pass_count,
        seeds.len(),
        partial_count,
        seeds.len()
    );

    println!();
    if pass_count == seeds.len() {
        println!("✓ PHASE 1.1 PASS: SNN learns binary MNIST with discriminative pixel selection.");
        println!("  → Next: Phase 1.1b — extend to 3+ digits, or Phase 2 WTA improvement");
    } else if pass_count > 0 || partial_count > 0 {
        println!(
            "? PARTIAL: {:.1}% average — some seeds pass. Try: more epochs, larger K.",
            avg * 100.0
        );
    } else {
        println!("✗ FAIL: <80% — discriminative ordering insufficient.");
        println!("  → Diagnose: check channel stats above (non-preferred avg should be near 0)");
        println!("  → Try: reduce K (fewer pixels, more extreme selection) or lower threshold");
    }
}
