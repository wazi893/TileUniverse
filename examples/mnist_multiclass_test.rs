//! Phase 1.2 + Phase 2: 10-class MNIST — Larger Network & Temporal WTA
//!
//! ## Phase 1.2: Larger Network Diagnostic
//! Network: 49 inputs (7×7 downsample) → [128] hidden → 10 outputs
//! Prediction: argmax(output_counts) after 30 ticks — no WTA
//! Expected: ≈10% (random chance) — confirms WTA is the bottleneck, not capacity
//!
//! ## Phase 2: Temporal Competition WTA
//! Same network + first output to fire wins, all others suppressed for rest of trial.
//! + Perceptron learning on output weights.
//! Expected: 30-60% after 10 epochs
//!
//! ## Why 7×7 downsampling?
//! 784 inputs with sparse connectivity → ~20k synapses → ~14B ops/epoch (too slow).
//! 49 inputs with dense connectivity → ~3k synapses → ~300M ops/epoch (~2s total).
//! At 7×7 resolution digits are still clearly distinguishable.
//!
//! ## GPU note
//! Full 784-input training would benefit enormously from GPU-batched SNN inference
//! (LIF neuron update + sparse synapse routing → ~256× speedup on RTX 5090).
//! This is tracked as a future milestone: "GPU-batched SNN step() for MNIST training".
//!
//! ## Usage
//! cargo run --release --example mnist_multiclass_test

use engine::snn::config::STDPConfig;
use engine::snn::{MnistDataset, NeuronConfig, SNNConfig, SNNNetwork};

// ─────────────────────────────────────────────────────────────────────────────
// Input encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Spatially downsample a 28×28 MNIST image to 7×7 by averaging each 4×4 block.
///
/// This reduces n_inputs from 784 → 49, cutting synapses from ~20k → ~3k and
/// making CPU training tractable without GPU acceleration.
/// Digit shapes remain recognizable at 7×7 resolution.
fn downsample_7x7(img: &[u8]) -> [u8; 49] {
    let mut out = [0u8; 49];
    for row in 0..7usize {
        for col in 0..7usize {
            let mut sum = 0u32;
            let mut cnt = 0u32;
            for dr in 0..4usize {
                for dc in 0..4usize {
                    let r = row * 4 + dr;
                    let c = col * 4 + dc;
                    if r < 28 && c < 28 {
                        sum += img[r * 28 + c] as u32;
                        cnt += 1;
                    }
                }
            }
            out[row * 7 + col] = (sum / cnt.max(1)) as u8;
        }
    }
    out
}

/// Rate-encode a downsampled image: averaged pixel → spike probability 0-100.
fn encode_image(img: &[u8]) -> [u8; 49] {
    let small = downsample_7x7(img);
    let mut rates = [0u8; 49];
    for (r, &px) in rates.iter_mut().zip(small.iter()) {
        *r = ((px as u32 * 100) / 255) as u8;
    }
    rates
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset helpers
// ─────────────────────────────────────────────────────────────────────────────

fn collect_balanced(dataset: &MnistDataset, max_per_class: usize) -> Vec<(Vec<u8>, usize)> {
    let mut counts = [0usize; 10];
    let mut out = Vec::new();
    for (img, &label) in dataset.images.iter().zip(dataset.labels.iter()) {
        let c = label as usize;
        if counts[c] < max_per_class {
            out.push((img.clone(), c));
            counts[c] += 1;
            if counts.iter().all(|&x| x >= max_per_class) {
                break;
            }
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

/// Build the 10-class network.
///
/// ## Parameter choices (7×7 input)
///
/// connection_prob = 0.5 (dense — reasonable for 49 inputs):
///   Each hidden neuron: ~49 × 0.5 × 0.8 = 19.6 excit + 4.9 inhib connections.
///   v_ss = (19.6×120 - 4.9×60) × (avg_rate/256) / (11/256)
///        ≈ (2352 - 294) × 0.051 / 0.043 ≈ 2442
///   → hidden threshold = 2000: fires selectively (~5% per tick)
///
/// Each output gets 128 × 0.15 × 0.8 ≈ 15 excitatory hidden connections.
///   At 5% hidden firing: 15 × 120 × 0.05 = 90 current/tick → v_ss ≈ 2090
///   → output threshold = 1000: fires about every 12 ticks
///
/// neurons_per_cpu = 256 > 49+128+10 = 187 → single CPU, no cross-CPU overhead.
fn build_network(seed: u64) -> SNNNetwork {
    let config = SNNConfig {
        n_inputs: 49,
        hidden_layers: vec![128],
        n_outputs: 10,
        connection_prob: 0.5,
        recurrent: false,
        neuron_config: NeuronConfig {
            threshold: 2000,
            leak: 245,
            refractory_period: 2,
            v_reset: -128,
        },
        stdp_config: STDPConfig {
            a_plus: 15,
            a_minus: 12,
            w_min: -128,
            w_max: 127,
            ..STDPConfig::default()
        },
        neurons_per_cpu: 256,
        use_reward_modulation: false,
        duplicate_inputs: false,
    };

    let mut net = SNNNetwork::new(config);
    net.build_simple_connectivity(seed);

    // Lower output threshold — hidden→output connections are sparse (0.5×0.3=0.15 prob)
    // so each output only gets ~15 excitatory connections. Need threshold below v_ss.
    let (_, out_start, out_end) = *net.topology.layers.last().unwrap();
    let npcu = net.config.neurons_per_cpu;
    for i in out_start..out_end {
        let cpu = net.topology.neuron_to_cpu[i];
        let local = net.topology.local_index(i, npcu) as usize;
        net.populations[cpu].neurons[local].threshold = 1000;
    }

    net
}

// ─────────────────────────────────────────────────────────────────────────────
// Trial execution
// ─────────────────────────────────────────────────────────────────────────────

/// Full neuron state reset before each trial.
///
/// reset_activity() only clears last_spike_time. WTA suppression sets refractory=200
/// on losers — those must be cleared before the next trial or they stay suppressed.
fn full_reset(net: &mut SNNNetwork) {
    net.reset_output_counts();
    net.reset_activity();
    for pop in &mut net.populations {
        for neuron in &mut pop.neurons {
            neuron.v_mem = 0;
            neuron.refractory = 0;
            neuron.spiked = 0;
        }
    }
}

/// Phase 1.2: run without WTA — argmax of output spike counts.
fn run_trial(net: &mut SNNNetwork, rates: &[u8], ticks: u32) -> usize {
    full_reset(net);
    net.set_inputs(rates);
    for _ in 0..ticks {
        net.step();
    }
    net.get_action()
}

/// Phase 2: temporal competition WTA — returns (prediction, Option<wta_winner_index>).
///
/// First output to fire wins. All others are suppressed (refractory=200) for
/// the remainder of the presentation. Perceptron training then strengthens the
/// correct output path so it fires first more reliably after training.
///
/// Returns `(pred, Some(winner))` if an output fired, `(pred, None)` if all silent.
fn run_trial_wta_tracked(net: &mut SNNNetwork, rates: &[u8], ticks: u32) -> (usize, Option<usize>) {
    full_reset(net);
    net.set_inputs(rates);

    let (_, out_start, out_end) = *net.topology.layers.last().unwrap();
    let n_out = out_end - out_start;
    let npcu = net.config.neurons_per_cpu;

    let mut winner: Option<usize> = None;

    for _ in 0..ticks {
        net.step();

        if winner.is_none() {
            for i in 0..n_out {
                let nidx = out_start + i;
                let cpu = net.topology.neuron_to_cpu[nidx];
                let local = net.topology.local_index(nidx, npcu) as usize;
                if net.populations[cpu].neurons[local].spiked == 1 {
                    winner = Some(i);
                    for j in 0..n_out {
                        if j != i {
                            let jidx = out_start + j;
                            let jcpu = net.topology.neuron_to_cpu[jidx];
                            let jlocal = net.topology.local_index(jidx, npcu) as usize;
                            net.populations[jcpu].neurons[jlocal].v_mem = -128;
                            net.populations[jcpu].neurons[jlocal].refractory = 200;
                        }
                    }
                    break;
                }
            }
        }
    }

    let pred = winner.unwrap_or_else(|| net.get_action());
    (pred, winner)
}

/// Convenience wrapper for evaluation (no win tracking needed).
fn run_trial_wta(net: &mut SNNNetwork, rates: &[u8], ticks: u32) -> usize {
    run_trial_wta_tracked(net, rates, ticks).0
}

// ─────────────────────────────────────────────────────────────────────────────
// Training and evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Returns (accuracy, per_class_acc, wta_win_counts).
/// `wta_win_counts[i]` = how many trials output i won the WTA race this epoch.
fn train_epoch(
    net: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
    use_wta: bool,
    lr: i8,
) -> (f32, [f32; 10], [u32; 10]) {
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; 10];
    let mut class_total = [0u32; 10];
    let mut wta_wins = [0u32; 10];

    for (img, label) in dataset {
        let label = *label;
        let rates = encode_image(img);
        let (pred, wta_winner) = if use_wta {
            let (p, w) = run_trial_wta_tracked(net, &rates, ticks);
            (p, w)
        } else {
            (run_trial(net, &rates, ticks), None)
        };
        if let Some(w) = wta_winner {
            wta_wins[w] += 1;
        }
        net.apply_perceptron_learning(pred, label, lr);
        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    let acc = correct as f32 / total.max(1) as f32;
    let per_class = std::array::from_fn(|c| class_correct[c] as f32 / class_total[c].max(1) as f32);
    (acc, per_class, wta_wins)
}

/// Homeostatic threshold adaptation.
///
/// After each training epoch, adjust output thresholds to keep firing rates balanced:
///   - Output fires > 2× target → raise threshold by `homeo_lr` (suppress dominant)
///   - Output fires 0 times     → lower threshold by `homeo_lr` (activate dormant)
///
/// This prevents single-output monopoly (mode collapse in WTA networks).
/// Without homeostasis, one randomly-favored output wins 90%+ of trials and
/// perceptron training just hops from monopoly to monopoly.
fn apply_homeostasis(net: &mut SNNNetwork, wta_wins: &[u32; 10], n_samples: usize, homeo_lr: i16) {
    let target = (n_samples / 10).max(1) as u32;
    let (_, out_start, out_end) = *net.topology.layers.last().unwrap();
    let n_out = out_end - out_start;
    let npcu = net.config.neurons_per_cpu;
    for i in 0..n_out {
        let nidx = out_start + i;
        let cpu = net.topology.neuron_to_cpu[nidx];
        let local = net.topology.local_index(nidx, npcu) as usize;
        let t = net.populations[cpu].neurons[local].threshold as i16;
        let new_t = if wta_wins[i] > target * 2 {
            (t + homeo_lr).min(5000) // over-firing: raise threshold
        } else if wta_wins[i] == 0 {
            (t - homeo_lr).max(100) // never fires: lower threshold
        } else {
            t
        };
        net.populations[cpu].neurons[local].threshold = new_t as i16;
    }
}

fn evaluate(
    net: &mut SNNNetwork,
    dataset: &[(Vec<u8>, usize)],
    ticks: u32,
    use_wta: bool,
) -> (f32, [f32; 10]) {
    let prev = net.runtime.learning_enabled;
    net.runtime.learning_enabled = false;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct = [0u32; 10];
    let mut class_total = [0u32; 10];

    for (img, label) in dataset {
        let label = *label;
        let rates = encode_image(img);
        let pred = if use_wta {
            run_trial_wta(net, &rates, ticks)
        } else {
            run_trial(net, &rates, ticks)
        };
        if pred == label {
            correct += 1;
            class_correct[label] += 1;
        }
        total += 1;
        class_total[label] += 1;
    }

    net.runtime.learning_enabled = prev;
    let acc = correct as f32 / total.max(1) as f32;
    let per_class = std::array::from_fn(|c| class_correct[c] as f32 / class_total[c].max(1) as f32);
    (acc, per_class)
}

// ─────────────────────────────────────────────────────────────────────────────
// Experiment runner
// ─────────────────────────────────────────────────────────────────────────────

fn run_experiment(
    label: &str,
    seed: u64,
    train_data: &[(Vec<u8>, usize)],
    test_data: &[(Vec<u8>, usize)],
    n_epochs: usize,
    ticks: u32,
    use_wta: bool,
    lr: i8,
) -> f32 {
    let mut net = build_network(seed);
    println!(
        "  Neurons: {}  Synapses: {}  [{label}]",
        net.n_neurons(),
        net.n_synapses()
    );

    // Pre-training baseline
    let (pre_acc, pre_class) = evaluate(&mut net, test_data, ticks, use_wta);
    let class_str: String = pre_class
        .iter()
        .map(|&a| format!("{:2.0}", a * 100.0))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  Pre:  {:.1}%  [{}]", pre_acc * 100.0, class_str);

    let mut shuffled = train_data.to_vec();
    shuffle(&mut shuffled, seed);

    let mut best_test = 0.0f32;
    let mut best_epoch = 0;

    for epoch in 0..n_epochs {
        shuffle(&mut shuffled, seed.wrapping_add(epoch as u64 + 1));
        let (train_acc, _, wta_wins) = train_epoch(&mut net, &shuffled, ticks, use_wta, lr);

        // Homeostasis: balance output firing rates each epoch to prevent monopoly.
        // Without this, one randomly-favored output dominates 90%+ of trials and
        // perceptron training can never assign class structure to the other outputs.
        if use_wta {
            apply_homeostasis(&mut net, &wta_wins, shuffled.len(), 150);
        }

        let (test_acc, per_class) = evaluate(&mut net, test_data, ticks, use_wta);

        if test_acc > best_test {
            best_test = test_acc;
            best_epoch = epoch + 1;
        }

        let should_log = epoch == 0 || (epoch + 1) % 5 == 0 || epoch == n_epochs - 1;
        if should_log {
            let class_str: String = per_class
                .iter()
                .map(|&a| format!("{:2.0}", a * 100.0))
                .collect::<Vec<_>>()
                .join(" ");
            // Show WTA diversity: how many distinct outputs are winning (ideally 10/10)
            let active_outputs = wta_wins.iter().filter(|&&w| w > 0).count();
            println!(
                "  Ep {:>2}: train={:.1}%  test={:.1}%  WTA={}/10  [{}]",
                epoch + 1,
                train_acc * 100.0,
                test_acc * 100.0,
                active_outputs,
                class_str
            );
        }
    }

    println!("  Best: {:.1}% @ epoch {}", best_test * 100.0, best_epoch);
    best_test
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Phase 1.2 + Phase 2: 10-class MNIST ===\n");
    println!("Input:   7×7 downsampled (49 pixels per image)");
    println!("Network: 49 → [128] hidden → 10 outputs");
    println!("Note:    Full 784-input training needs GPU-batched SNN step()");
    println!("         (tracked as future milestone — RTX 5090 ~256× speedup)\n");

    // ── Load MNIST ────────────────────────────────────────────────────────────
    let train_mnist = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("MNIST not found — run from repo root, data/ must exist");
    let test_mnist =
        MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
            .expect("MNIST test data not found");

    // Balanced subsets — large enough for qualitative signal, small enough for CPU
    let train_data = collect_balanced(&train_mnist, 200); // 2000 total
    let test_data = collect_balanced(&test_mnist, 100); // 1000 total
    println!(
        "Using {} train ({}/class), {} test ({}/class)\n",
        train_data.len(),
        200,
        test_data.len(),
        100
    );

    let seeds: &[u64] = &[42, 123, 999];
    let ticks = 30u32;
    let n_epochs = 10usize;
    let lr = 5i8;

    // ── Phase 1.2: No WTA (diagnostic) ───────────────────────────────────────
    println!("════════════════════════════════════════════");
    println!("Phase 1.2: Larger Network — No WTA");
    println!("  Prediction: argmax(output_counts) after {} ticks", ticks);
    println!("  Expected: ≈10% — WTA is the bottleneck, not capacity");
    println!("════════════════════════════════════════════\n");

    let mut res12 = Vec::new();
    for &seed in seeds {
        println!("── Seed {} ──", seed);
        let best = run_experiment(
            "no WTA",
            seed,
            &train_data,
            &test_data,
            n_epochs,
            ticks,
            false,
            lr,
        );
        res12.push(best);
        println!();
    }

    let avg12 = res12.iter().sum::<f32>() / res12.len() as f32;
    println!("Phase 1.2 avg: {:.1}%", avg12 * 100.0);
    for (&s, &b) in seeds.iter().zip(res12.iter()) {
        println!(
            "  Seed {:>5}: {:.1}%  {}",
            s,
            b * 100.0,
            if b >= 0.30 {
                "★ above chance"
            } else {
                "(≈chance)"
            }
        );
    }
    let wta_confirmed = avg12 < 0.20;
    println!(
        "→ {}\n",
        if wta_confirmed {
            "WTA bottleneck confirmed. Capacity alone doesn't help."
        } else {
            "Larger capacity shows some benefit even without WTA."
        }
    );

    // ── Phase 2: Temporal Competition WTA ────────────────────────────────────
    println!("════════════════════════════════════════════");
    println!("Phase 2: Temporal Competition WTA");
    println!("  First output to fire wins; others → refractory=200");
    println!("  Perceptron learning: lr={}, {} epochs", lr, n_epochs);
    println!("  Expected: 30-60% after training");
    println!("════════════════════════════════════════════\n");

    let mut res2 = Vec::new();
    for &seed in seeds {
        println!("── Seed {} ──", seed);
        let best = run_experiment(
            "WTA",
            seed,
            &train_data,
            &test_data,
            n_epochs,
            ticks,
            true,
            lr,
        );
        res2.push(best);
        println!();
    }

    let avg2 = res2.iter().sum::<f32>() / res2.len() as f32;
    let pass2 = res2.iter().filter(|&&a| a >= 0.50).count();
    let gain = avg2 - avg12;

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════");
    println!("  Phase 1.2 (no WTA):  {:.1}% avg", avg12 * 100.0);
    println!(
        "  Phase 2   (WTA):     {:.1}% avg  ({}/{} seeds ≥50%)  gain={:+.1}%",
        avg2 * 100.0,
        pass2,
        seeds.len(),
        gain * 100.0
    );
    println!();
    for (&seed, (&b12, &b2)) in seeds.iter().zip(res12.iter().zip(res2.iter())) {
        println!(
            "  Seed {:>5}: no-WTA={:.1}%  WTA={:.1}%  Δ={:+.1}%",
            seed,
            b12 * 100.0,
            b2 * 100.0,
            (b2 - b12) * 100.0
        );
    }

    println!();
    if avg2 >= 0.60 {
        println!("✓ Phase 2 PASS: WTA + homeostasis achieves ≥60%.");
        println!("  → Next: Phase 3 supervised pre-train, or scale to 28×28 with GPU batching.");
    } else if avg2 >= 0.30 {
        println!(
            "? Phase 2 PARTIAL: {:.1}% — above chance, but limited by random features.",
            avg2 * 100.0
        );
        println!("  Root cause: random hidden neurons fire for all digits → perceptron saturates");
        println!("  all weights to w_max, so WTA winner is determined by connection count bias.");
        println!("  → Next: Phase 3 — extend discriminative channel architecture to 10 classes");
        println!("            (Phase 1.1 approach: K pixels/class, channel-aligned connectivity)");
    } else {
        println!(
            "✗ Phase 2 FAIL: WTA + homeostasis insufficient (Δ={:+.1}%).",
            gain * 100.0
        );
        println!();
        println!("  Diagnosis: random hidden features with flat class tuning curves.");
        println!("  - Hidden neurons fire for most digit classes (random 49→128 projection)");
        println!("  - Perceptron strengthens h→all_outputs equally → weights saturate at w_max");
        println!("  - WTA winner = output with most initial connections (unchanged by training)");
        println!("  - Homeostasis forces balanced firing rates but can't fix weight saturation");
        println!();
        println!("  Fix options:");
        println!("  A) Phase 3a: Extend discriminative channels to 10 classes (proven approach)");
        println!("     K pixels/class → 10 channel inputs → 10 hidden groups → 10 outputs");
        println!("     Same architecture as Phase 1.1 (96.1%), scaled from 2 to 10 classes");
        println!("  B) Phase 3b: Supervised pre-train → use correct label to drive weight updates");
        println!("     Penalise ALL wrong outputs (not just WTA winner) → avoid weight saturation");
        println!("  C) Phase 4: GPU-batched SNN on full 28×28 with deeper supervised training");
    }

    println!();
    println!("Future work: GPU-batched SNN step()");
    println!("  Current: 1 sample × 187 neurons × 30 ticks (CPU sequential)");
    println!("  Target:  256 samples × 187 neurons × 30 ticks (CUDA parallel)");
    println!("  Benefit: ~256× speedup → full 784-input 60K MNIST training feasible");
    println!("  Impact:  Enables Phase 3/4 on full MNIST without downsampling");
}
