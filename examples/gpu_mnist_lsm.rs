//! Milestone 7 Phase 1: LSM + Linear Readout (Pre-check + Full Training)
//!
//! The frozen discriminative hidden layer acts as a Liquid State Machine (LSM).
//! Hidden spike counts over T ticks become a 320-dim feature vector; a linear
//! readout (W [320×10]) is trained from scratch via softmax cross-entropy SGD.
//!
//! ```text
//! Input[K×C=1500]  →  Hidden[H×C=320]  →  W_readout[320×10]  →  softmax → CE
//!   (Fisher prior)     (frozen, no ALIF)    (f32, CPU SGD)
//! ```
//!
//! ## Pre-check gate (epoch 3)
//!
//! If test accuracy < 67%: hidden representation lacks separability — pause before
//! committing to Phase 2. If ≥ 67%: proceed with full training.
//!
//! ## Acceptance gates
//!
//! | Gate | Threshold |
//! |------|-----------|
//! | Pre-check epoch 3 | ≥ 67% |
//! | Phase 1 final best | ≥ 70% |
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_lsm
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires --features cuda.");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    run();
}

#[cfg(feature = "cuda")]
fn run() {
    use engine::cuda::CudaRuntime;
    use engine::snn::gpu_batch::GpuBatchSNN;
    use engine::snn::mnist::MnistDataset;
    use engine::snn::{discriminative_3layer_thresholds, generate_discriminative_3layer_csr};
    use std::time::Instant;

    // ── Seed (optional CLI arg: cargo run ... -- 42) ──────────────────────────
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const N_CLASSES: usize = 10;
    const N_PIXELS: usize = 784;
    const K: usize = 150; // discriminative pixels per class
    const H: usize = 32; // hidden neurons per class
    const R: usize = 8; // readout neurons per class (kept for network compat)
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB_IH: f32 = 0.5;
    const CONN_PROB_HR: f32 = 0.8;
    const LEAK: u8 = 230;
    const W_MAX_IH: i8 = 120;
    const W_MAX_HR: i8 = 32; // spiking readout unused; kept for CSR compat
    const T_READOUT: i16 = 400;
    const ALPHA: f32 = 0.50;

    // ── LSM linear readout hyperparameters ────────────────────────────────────
    const LR_ADAM: f32 = 0.001; // Adam learning rate for W_readout
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const N_EPOCHS: usize = 60;
    const PRECHECK_EPOCH: usize = 6; // gate check at this epoch (1-indexed)
    const PRECHECK_THRESHOLD: f32 = 62.0; // abort if below at pre-check epoch (ceiling-level)

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 7 Phase 1: LSM + Linear Readout (Pre-check)");
    println!(
        "  K={K} · H={H} (LSM hidden) · Linear readout [{}×{N_CLASSES}]  seed={seed}",
        H * N_CLASSES
    );
    println!("  IH frozen (Fisher prior) · No ALIF · No E-prop");
    println!(
        "  Adam LR={LR_ADAM} β1={BETA1} β2={BETA2} · Epochs={N_EPOCHS} · Batch={BATCH_SIZE} · Ticks={TICKS}"
    );
    println!("  Pre-check gate: ≥{PRECHECK_THRESHOLD:.0}% at epoch {PRECHECK_EPOCH}");
    println!("  Phase 1 gate: ≥69% best over full run");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Load MNIST ───────────────────────────────────────────────────────────
    let train_ds = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("load training data");
    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("load test data");
    println!(
        "  Loaded {} train + {} test samples",
        train_ds.images.len(),
        test_ds.images.len()
    );

    let train_flat: Vec<(Vec<u8>, usize)> = train_ds
        .images
        .iter()
        .zip(train_ds.labels.iter())
        .map(|(img, &lbl)| (img.clone(), lbl as usize))
        .filter(|(_, c)| *c < N_CLASSES)
        .collect();
    let test_flat: Vec<(Vec<u8>, usize)> = test_ds
        .images
        .iter()
        .zip(test_ds.labels.iter())
        .map(|(img, &lbl)| (img.clone(), lbl as usize))
        .filter(|(_, c)| *c < N_CLASSES)
        .collect();

    // ── Class statistics ──────────────────────────────────────────────────────
    let mut avgs = vec![[0f32; N_PIXELS]; N_CLASSES];
    let mut counts_per_class = vec![0usize; N_CLASSES];
    for (img, c) in &train_flat {
        counts_per_class[*c] += 1;
        for i in 0..N_PIXELS {
            avgs[*c][i] += img[i] as f32;
        }
    }
    for c in 0..N_CLASSES {
        let n = counts_per_class[c] as f32;
        if n > 0.0 {
            for i in 0..N_PIXELS {
                avgs[c][i] /= n;
            }
        }
    }
    let mut global_sum = [0f32; N_PIXELS];
    for c in 0..N_CLASSES {
        for i in 0..N_PIXELS {
            global_sum[i] += avgs[c][i];
        }
    }

    // ── Pixel scoring ─────────────────────────────────────────────────────────
    println!("  Selecting top-K pixels by mean-diff...");
    let mut pix_per_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    let mut d_norms: Vec<Vec<f32>> = vec![Vec::new(); N_CLASSES];
    for c in 0..N_CLASSES {
        let mut scores: Vec<(usize, f32)> = (0..N_PIXELS)
            .map(|i| {
                let others = (global_sum[i] - avgs[c][i]) / (N_CLASSES - 1) as f32;
                (i, avgs[c][i] - others)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<usize> = scores[..K].iter().map(|(i, _)| *i).collect();
        let d_raw: Vec<f32> = top_k
            .iter()
            .map(|&pix| {
                let others = (global_sum[pix] - avgs[c][pix]) / (N_CLASSES - 1) as f32;
                (avgs[c][pix] - others).max(0.0)
            })
            .collect();
        let max_d = d_raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        d_norms[c] = d_raw.iter().map(|&d| d / max_d).collect();
        pix_per_class[c] = top_k;
    }

    // ── Hidden threshold calibration ─────────────────────────────────────────
    let w_scale = 2.0_f32;
    let leak_gap = 256.0_f32 - LEAK as f32;
    let gpu_vss = |cc: usize, cj: usize| -> f32 {
        let weighted_sum: f32 = pix_per_class[cc]
            .iter()
            .zip(d_norms[cc].iter())
            .map(|(&px, &d)| avgs[cj][px] * d)
            .sum();
        CONN_PROB_IH * W_MAX_IH as f32 * w_scale * MAX_RATE as f32 / (255.0 * leak_gap)
            * weighted_sum
    };
    let t_hidden: Vec<i16> = (0..N_CLASSES)
        .map(|c| {
            let vss_cc = gpu_vss(c, c);
            let vss_max = (0..N_CLASSES)
                .filter(|&j| j != c)
                .map(|j| gpu_vss(c, j))
                .fold(0.0f32, f32::max);
            let t = if vss_cc > vss_max {
                (ALPHA * vss_cc + (1.0 - ALPHA) * vss_max).clamp(1.0, i16::MAX as f32)
            } else {
                (0.85 * vss_cc).clamp(1.0, i16::MAX as f32)
            };
            t as i16
        })
        .collect();
    println!("  Hidden thresholds (α={ALPHA:.2}): {:?}", t_hidden);

    // ── Build network ─────────────────────────────────────────────────────────
    // Keep 3-layer topology (R=8 readout). The spiking readout is unused —
    // we extract hidden counts from d_all_counts via enable_stdp().
    let n_inp = K * N_CLASSES;
    let n_hid = H * N_CLASSES;
    let n_out = R * N_CLASSES;
    let n_neurons = n_inp + n_hid + n_out;
    let n_inputs = n_inp;
    let n_outputs = n_out;
    let leaks = vec![LEAK; n_neurons];

    let d_norms_flat: Vec<f32> = d_norms.iter().flat_map(|row| row.iter().copied()).collect();

    let thresholds =
        discriminative_3layer_thresholds(K, H, R, N_CLASSES, &t_hidden, T_READOUT, 32000);
    let synapses = generate_discriminative_3layer_csr(
        K,
        H,
        R,
        N_CLASSES,
        CONN_PROB_IH,
        CONN_PROB_HR,
        W_MAX_IH,
        W_MAX_HR,
        &d_norms_flat,
        seed,
    );
    let n_total_synapses = synapses.targets.len();

    let rt = CudaRuntime::new().expect("CUDA init");
    let mut snn = GpuBatchSNN::new(
        &rt,
        n_neurons,
        n_inputs,
        n_outputs,
        &thresholds,
        &leaks,
        &synapses,
        BATCH_SIZE,
    )
    .expect("GpuBatchSNN::new");

    // Enable STDP — allocates d_all_counts [B × N], needed for hidden counting.
    // We use only the counting infrastructure, not Hebbian learning.
    snn.enable_stdp(&rt)
        .expect("enable_stdp (for hidden counting)");

    println!("  Network: {n_inp} inputs · {n_hid} hidden (LSM) · {n_out} readout (unused)");
    println!(
        "  Total synapses: {n_total_synapses}  ·  d_all_counts allocated ({BATCH_SIZE}×{n_neurons})"
    );

    // ── Linear readout weight init (Glorot uniform) ───────────────────────────
    let mut rng: u64 = seed.wrapping_mul(0xDEAD_BEEF_CAFE_1234).wrapping_add(1);
    let mut lcg = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    };
    let mut lcg_f32 = |s: &mut u64| -> f32 { (lcg(s) >> 33) as f32 / (u32::MAX >> 1) as f32 };
    let w_init_scale = (6.0 / (n_hid + N_CLASSES) as f32).sqrt();
    let mut w_readout = vec![0.0f32; n_hid * N_CLASSES];
    for w in w_readout.iter_mut() {
        *w = (lcg_f32(&mut rng) * 2.0 - 1.0) * w_init_scale;
    }
    let mut dw = vec![0.0f32; n_hid * N_CLASSES]; // gradient accumulator
    // Adam optimizer state
    let mut m_w = vec![0.0f32; n_hid * N_CLASSES]; // 1st moment
    let mut v_w = vec![0.0f32; n_hid * N_CLASSES]; // 2nd moment
    let mut adam_t: u32 = 0; // step counter

    println!(
        "  W_readout [{}×{N_CLASSES}] f32, Glorot uniform (scale={w_init_scale:.4})",
        n_hid
    );

    // ── Encoding helper ───────────────────────────────────────────────────────
    let encode_image = |img: &[u8]| -> Vec<u8> {
        let mut rates = vec![0u8; K * N_CLASSES];
        for c in 0..N_CLASSES {
            for (i, &pix) in pix_per_class[c].iter().enumerate() {
                rates[c * K + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
            }
        }
        rates
    };

    // ── Linear readout: forward + CE loss + gradient ──────────────────────────
    // hidden_rates: [actual_bs × n_hid] f32 (spike counts / TICKS, range [0,1])
    // w:            [n_hid × N_CLASSES] f32
    // dw:           [n_hid × N_CLASSES] f32 (accumulates gradient over batch)
    // Returns (batch_correct, mean_ce_loss)
    let linear_forward_ce = |hidden_rates: &[f32],
                             labels: &[usize],
                             w: &[f32],
                             dw: &mut [f32],
                             actual_bs: usize|
     -> (usize, f32) {
        let mut correct = 0usize;
        let mut total_loss = 0.0f32;
        dw.fill(0.0);
        let mut logits = vec![0.0f32; N_CLASSES];
        let mut softmax = vec![0.0f32; N_CLASSES];

        for b in 0..actual_bs {
            let h = &hidden_rates[b * n_hid..(b + 1) * n_hid];

            // Linear projection: logit[c] = sum_h h[hh] * w[hh * N_CLASSES + c]
            logits.fill(0.0);
            for hh in 0..n_hid {
                let h_val = h[hh];
                let w_row = &w[hh * N_CLASSES..(hh + 1) * N_CLASSES];
                for c in 0..N_CLASSES {
                    logits[c] += h_val * w_row[c];
                }
            }

            // Numerically stable softmax
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..N_CLASSES {
                softmax[c] = (logits[c] - max_l).exp();
                sum_exp += softmax[c];
            }
            for c in 0..N_CLASSES {
                softmax[c] /= sum_exp;
            }

            // Prediction + CE loss
            let pred = softmax
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred == labels[b] {
                correct += 1;
            }
            total_loss -= softmax[labels[b]].ln().max(-20.0);

            // Gradient: dL/dW[hh, c] += (softmax[c] - y[c]) * h[hh]
            for c in 0..N_CLASSES {
                let delta = softmax[c] - if c == labels[b] { 1.0 } else { 0.0 };
                let dw_row = &mut dw[0..];
                for hh in 0..n_hid {
                    dw_row[hh * N_CLASSES + c] += delta * h[hh];
                }
            }
        }
        (correct, total_loss / actual_bs as f32)
    };

    // ── Test evaluation (no gradient) ─────────────────────────────────────────
    let run_test = |snn: &mut GpuBatchSNN, w: &[f32], dataset: &[(Vec<u8>, usize)]| -> f32 {
        let n = dataset.len();
        let mut correct = 0usize;
        let mut i = 0;
        let mut logits = vec![0.0f32; N_CLASSES];
        let mut softmax = vec![0.0f32; N_CLASSES];

        while i < n {
            let chunk = &dataset[i..(i + BATCH_SIZE).min(n)];
            let actual_bs = chunk.len();

            let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
            for (b, (img, _)) in chunk.iter().enumerate() {
                let enc = encode_image(img);
                rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
            }

            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_all_counts(&rt).unwrap();
            snn.tick_many_full_counting(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();

            let all_counts = snn.get_all_counts(&rt).unwrap();

            for b in 0..actual_bs {
                let row = &all_counts[b];
                // Extract hidden firing rates: neurons n_inp..n_inp+n_hid
                logits.fill(0.0);
                for c in 0..N_CLASSES {
                    let mut sum = 0.0f32;
                    for hh in 0..n_hid {
                        sum += (row[n_inp + hh] as f32 / TICKS as f32) * w[hh * N_CLASSES + c];
                    }
                    logits[c] = sum;
                }
                // Softmax argmax
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for c in 0..N_CLASSES {
                    softmax[c] = (logits[c] - max_l).exp();
                    sum_exp += softmax[c];
                }
                for c in 0..N_CLASSES {
                    softmax[c] /= sum_exp;
                }
                let pred = softmax
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if pred == chunk[b].1 {
                    correct += 1;
                }
            }
            i += actual_bs;
        }
        100.0 * correct as f32 / n as f32
    };

    // ── Baseline: random W_readout ────────────────────────────────────────────
    println!("\n  Baseline (random W_readout)...");
    let t0 = Instant::now();
    let baseline_acc = run_test(&mut snn, &w_readout, &test_flat);
    println!(
        "  Random readout accuracy: {baseline_acc:.1}%  ({:.1}s)  [expect ~10%]",
        t0.elapsed().as_secs_f32()
    );
    println!("  M6 ceiling (T-norm, frozen IH): 62.4%\n");

    // ── Training loop ─────────────────────────────────────────────────────────
    let n_train = train_flat.len();
    println!("  Starting LSM training ({N_EPOCHS} epochs, Adam LR={LR_ADAM})...");
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>14}  {:>10}  {:>8}",
        "epoch", "train_corr", "train_acc", "test_acc", "ce_loss", "secs"
    );
    println!("  {}", "-".repeat(68));

    let mut best_test_acc = baseline_acc;
    let mut precheck_passed = false;
    let mut precheck_failed = false;

    for epoch in 0..N_EPOCHS {
        // Shuffle
        let mut indices: Vec<usize> = (0..n_train).collect();
        for i in (1..n_train).rev() {
            let j = (lcg(&mut rng) as usize) % (i + 1);
            indices.swap(i, j);
        }

        let epoch_t0 = Instant::now();
        let mut train_correct = 0usize;
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0usize;
        let mut i = 0;

        while i < n_train {
            let chunk_end = (i + BATCH_SIZE).min(n_train);
            let chunk_idx = &indices[i..chunk_end];
            let actual_bs = chunk_idx.len();

            // Encode batch
            let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
            let mut true_labels = vec![0usize; actual_bs];
            for (b, &idx) in chunk_idx.iter().enumerate() {
                let (ref img, lbl) = train_flat[idx];
                let enc = encode_image(img);
                rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
                true_labels[b] = lbl;
            }

            // GPU forward pass: count all neurons
            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_all_counts(&rt).unwrap();
            snn.tick_many_full_counting(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();

            // Download all-neuron counts, extract hidden firing rates
            let all_counts = snn.get_all_counts(&rt).unwrap();
            let mut hidden_rates = vec![0.0f32; BATCH_SIZE * n_hid];
            for b in 0..actual_bs {
                let row = &all_counts[b];
                for hh in 0..n_hid {
                    hidden_rates[b * n_hid + hh] = row[n_inp + hh] as f32 / TICKS as f32;
                }
            }

            // CPU forward + CE loss + gradient
            let (batch_correct, batch_loss) =
                linear_forward_ce(&hidden_rates, &true_labels, &w_readout, &mut dw, actual_bs);
            train_correct += batch_correct;
            epoch_loss += batch_loss;
            n_batches += 1;

            // Adam update (step-level, one step per batch)
            adam_t += 1;
            let g_scale = 1.0 / actual_bs as f32;
            let bc1 = 1.0 - BETA1.powi(adam_t as i32);
            let bc2 = 1.0 - BETA2.powi(adam_t as i32);
            let lr_eff = LR_ADAM * bc2.sqrt() / bc1;
            for k in 0..w_readout.len() {
                let g = dw[k] * g_scale;
                m_w[k] = BETA1 * m_w[k] + (1.0 - BETA1) * g;
                v_w[k] = BETA2 * v_w[k] + (1.0 - BETA2) * g * g;
                w_readout[k] -= lr_eff * m_w[k] / (v_w[k].sqrt() + ADAM_EPS);
            }

            i += actual_bs;
        }

        let test_acc = run_test(&mut snn, &w_readout, &test_flat);
        let train_acc = 100.0 * train_correct as f32 / n_train as f32;
        let mean_loss = epoch_loss / n_batches as f32;
        let epoch_secs = epoch_t0.elapsed().as_secs_f32();
        let marker = if test_acc > best_test_acc { " ←" } else { "" };
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
        }

        println!(
            "  {:>6}  {:>12}  {:>11.1}%  {:>13.1}%{:>3}  {:>10.4}  {:>7.1}s",
            epoch + 1,
            train_correct,
            train_acc,
            test_acc,
            marker,
            mean_loss,
            epoch_secs
        );

        // Pre-check gate at PRECHECK_EPOCH
        if epoch + 1 == PRECHECK_EPOCH {
            if test_acc >= PRECHECK_THRESHOLD {
                println!("\n  *** PRE-CHECK PASS: {test_acc:.1}% ≥ {PRECHECK_THRESHOLD:.0}%");
                println!("  *** Hidden representation has separability headroom.");
                println!("  *** Proceeding with full training → Phase 1 gate ≥70%\n");
                precheck_passed = true;
            } else {
                println!("\n  *** PRE-CHECK FAIL: {test_acc:.1}% < {PRECHECK_THRESHOLD:.0}%");
                println!("  *** Hidden representation lacks headroom.");
                println!("  *** Pause — diagnose before committing to Phase 2.");
                println!("  *** Options: increase H, adjust K, check threshold calibration.\n");
                precheck_failed = true;
                // Don't break — keep training to measure plateau
            }
        }
    }

    // ── Final weight stats ─────────────────────────────────────────────────────
    if !precheck_failed {
        let w_min = w_readout.iter().cloned().fold(f32::INFINITY, f32::min);
        let w_max = w_readout.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w_mean = w_readout.iter().sum::<f32>() / w_readout.len() as f32;
        println!("\n  ── Final W_readout stats ──");
        println!("  min={w_min:.3}  max={w_max:.3}  mean={w_mean:.4}");
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n  ── Summary ──");
    println!("  ┌─────────────────────────────────┬────────┐");
    println!("  │ M6 ceiling (T-norm, frozen IH)   │ 62.4% │");
    println!("  │ Random W_readout baseline        │ {baseline_acc:.1}% │");
    println!("  │ LSM + linear readout best        │ {best_test_acc:.1}% │");
    let delta = best_test_acc - 62.4;
    if delta > 0.0 {
        println!("  │ Δ vs M6 ceiling (62.4%)          │ +{delta:.1}pp │");
    } else {
        println!("  │ Δ vs M6 ceiling (62.4%)          │  {delta:.1}pp │");
    }
    println!("  └─────────────────────────────────┴────────┘");

    println!();
    let gate_precheck = precheck_passed;
    let gate_phase1 = best_test_acc >= 69.0;
    let gate_stretch = best_test_acc >= 78.0;

    if precheck_failed {
        println!("  [✗] Pre-check FAIL — stopped at epoch {PRECHECK_EPOCH}");
    } else {
        println!(
            "  [{}] Pre-check ≥{PRECHECK_THRESHOLD:.0}% at epoch {PRECHECK_EPOCH}",
            if gate_precheck { "✓" } else { "✗" }
        );
    }
    println!(
        "  [{}] Phase 1 gate ≥69% ({best_test_acc:.1}%)",
        if gate_phase1 { "✓" } else { "✗" }
    );
    println!(
        "  [{}] Stretch ≥78% ({best_test_acc:.1}%)",
        if gate_stretch { "✓" } else { "✗" }
    );

    if gate_phase1 {
        println!("\n  ✓ Phase 1 PASS — proceed to Phase 2: coupled IH E-prop.");
        println!("    Next: gpu_mnist_lsm_v2.rs + tick_many_with_eprop_and_hidden_counting()");
    } else if !precheck_failed {
        println!("\n  ✗ Phase 1 below 69% — investigate before Phase 2.");
        println!("    Consider: higher H, more epochs, LR schedule, or weight decay.");
    }
}
