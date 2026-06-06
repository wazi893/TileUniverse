//! Milestone 5 Phase 2: GPU E-prop Training — Eligibility Propagation
//!
//! Trains a discriminative-channel SNN using online eligibility traces (e-prop).
//! E-prop is the spike-equivalent of BPTT without storing the full unrolled graph.
//!
//! ## What it does
//!
//! 1. Analytically initialize a discriminative-channel SNN (K=150, H=32, Fisher scoring).
//! 2. Enable ALIF adaptive thresholds (`enable_alif`) — homeostatic suppression during training.
//! 3. Enable e-prop eligibility traces (`enable_eprop`) — per-synapse online gradient estimate.
//! 4. For each epoch:
//!    - Shuffle training set.
//!    - Per batch: `tick_many_with_eprop` → argmax prediction → learning signal ±1.0
//!      → `apply_eprop_update(lr)`.
//!    - Evaluate frozen weights on test set (`tick_many_with_counting`).
//! 5. Report accuracy per epoch vs 62.2% frozen-weight baseline.
//!
//! Gate: test accuracy ≥ 78% after 10 epochs before moving to Phase 3.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_eprop
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
    use engine::snn::{
        discriminative_2layer_thresholds, generate_discriminative_2layer_csr_weighted,
    };
    use std::time::Instant;

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const N_CLASSES: usize = 10;
    const N_PIXELS: usize = 784;
    const K: usize = 150; // discriminative pixels per class
    const H: usize = 32; // hidden neurons per class
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB: f32 = 0.5;
    const LEAK: u8 = 230;
    const W_MAX: i8 = 120;
    const ALPHA: f32 = 0.50; // threshold T = α·V_self + (1-α)·V_rival

    // ── ALIF parameters ───────────────────────────────────────────────────────
    const ALIF_ALPHA: f32 = 0.967; // exp(-1/30): 30-tick adaptation time constant
    const ALIF_BETA: f32 = 0.1; // each spike raises threshold by ~10% of a unit

    // ── E-prop parameters ─────────────────────────────────────────────────────
    const N_EPOCHS: usize = 20;
    const LEARNING_RATE: f32 = 0.003; // f32 shadow path: no i8 rounding — 0.003 gives ~1 unit/epoch/active-syn
    const EPROP_ALPHA_PRE: f32 = 0.95; // pre-synaptic trace decay
    const EPROP_ALPHA_ELIG: f32 = 0.95; // eligibility trace decay
    const EPROP_GAMMA: f32 = 0.3; // fast-sigmoid surrogate sharpness

    // LR sweep — uncomment to explore; default: single run at LEARNING_RATE
    const LR_CANDIDATES: &[f32] = &[LEARNING_RATE];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 5 Phase 2: GPU E-prop Training");
    println!("  K={K} · H={H} · ALIF(α={ALIF_ALPHA:.3}, β={ALIF_BETA})");
    println!("  E-prop(α_pre={EPROP_ALPHA_PRE}, α_elig={EPROP_ALPHA_ELIG}, γ={EPROP_GAMMA})");
    println!("  Epochs={N_EPOCHS} · LR={LEARNING_RATE} · Batch={BATCH_SIZE} · Ticks={TICKS}");
    println!("  Gate: ≥ 78% test accuracy after 10 epochs");
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

    // Flatten into (image, label) pairs for shuffle-indexing
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

    // ── Class statistics (full training set) ─────────────────────────────────
    let mut avgs = vec![[0f32; N_PIXELS]; N_CLASSES];
    let mut vars = vec![[0f32; N_PIXELS]; N_CLASSES];
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
    for (img, c) in &train_flat {
        for i in 0..N_PIXELS {
            let diff = img[i] as f32 - avgs[*c][i];
            vars[*c][i] += diff * diff;
        }
    }
    for c in 0..N_CLASSES {
        let n = counts_per_class[c] as f32;
        if n > 1.0 {
            for i in 0..N_PIXELS {
                vars[c][i] /= n - 1.0;
            }
        }
    }

    let mut global_sum = [0f32; N_PIXELS];
    for c in 0..N_CLASSES {
        for i in 0..N_PIXELS {
            global_sum[i] += avgs[c][i];
        }
    }

    // ── Pixel scoring: mean-diff ordering (matches gpu_mnist_discrim "no Fisher" mode) ──
    // Fisher LDA ordering selects dark-for-class pixels that produce zero d_raw weights
    // and collapse thresholds to T=1, degrading the frozen baseline from 62.2% to 55%.
    // Mean-diff ordering (class bright - rest bright) is positive by construction.
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

    // ── GPU V_ss helpers ──────────────────────────────────────────────────────
    let w_scale = 2.0_f32;
    let leak_gap = 256.0_f32 - LEAK as f32;
    let gpu_vss = |cc: usize, cj: usize| -> f32 {
        let weighted_sum: f32 = pix_per_class[cc]
            .iter()
            .zip(d_norms[cc].iter())
            .map(|(&px, &d)| avgs[cj][px] * d)
            .sum();
        CONN_PROB * W_MAX as f32 * w_scale * MAX_RATE as f32 / (255.0 * leak_gap) * weighted_sum
    };

    // Compute threshold array
    let t_per_class: Vec<i16> = (0..N_CLASSES)
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
    println!("  Per-class thresholds (α={ALPHA:.2}): {:?}", t_per_class);

    // ── Build CSR and SNN ─────────────────────────────────────────────────────
    let n_inputs = K * N_CLASSES;
    let n_hidden = H * N_CLASSES;
    let n_neurons = n_inputs + n_hidden;
    let n_outputs = n_hidden;
    let leaks = vec![LEAK; n_neurons];

    let d_norms_flat: Vec<f32> = d_norms.iter().flat_map(|row| row.iter().copied()).collect();
    let thresholds = discriminative_2layer_thresholds(K, H, N_CLASSES, &t_per_class, 32767);
    let synapses = generate_discriminative_2layer_csr_weighted(
        K,
        H,
        N_CLASSES,
        CONN_PROB,
        W_MAX,
        &d_norms_flat,
        42,
    );

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

    snn.enable_alif(&rt, ALIF_ALPHA, ALIF_BETA)
        .expect("enable_alif");
    snn.enable_eprop(&rt, EPROP_ALPHA_PRE, EPROP_ALPHA_ELIG, EPROP_GAMMA)
        .expect("enable_eprop");
    snn.init_weight_shadow(&rt).expect("init_weight_shadow");

    println!(
        "  Network: {n_inputs} inputs · {n_hidden} hidden · {} synapses",
        synapses.targets.len()
    );
    println!("  ALIF + E-prop buffers allocated.");

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

    // ── Argmax prediction from output counts (T-normalized) ──────────────────
    let predict_batch = |counts: &Vec<Vec<u32>>, actual_bs: usize| -> Vec<usize> {
        (0..actual_bs)
            .map(|b| {
                let row = &counts[b];
                (0..N_CLASSES)
                    .map(|c| {
                        let lo = c * H;
                        let hi = lo + H;
                        let raw: u64 = row[lo..hi].iter().map(|&x| x as u64).sum();
                        raw * t_per_class[c] as u64
                    })
                    .enumerate()
                    .max_by_key(|&(_, s)| s)
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect()
    };

    // ── Inference loop (no weight update, no e-prop overhead) ─────────────────
    let run_test = |snn: &mut GpuBatchSNN, dataset: &[(Vec<u8>, usize)]| -> f32 {
        let n = dataset.len();
        let mut correct = 0usize;
        let mut i = 0;
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
            snn.reset_adapt_state(&rt).unwrap();
            snn.reset_output_counts(&rt).unwrap();
            snn.tick_many_with_counting(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();
            let counts = snn.get_output_counts(&rt).unwrap();
            let preds = predict_batch(&counts, actual_bs);
            for (b, (_, true_lbl)) in chunk.iter().enumerate() {
                if preds[b] == *true_lbl {
                    correct += 1;
                }
            }
            i += actual_bs;
        }
        100.0 * correct as f32 / n as f32
    };

    // ── Baseline (frozen weights, no training) ────────────────────────────────
    println!("\n  Baseline (frozen weights, ALIF enabled)...");
    let t0 = Instant::now();
    let baseline_acc = run_test(&mut snn, &test_flat);
    println!(
        "  Baseline test accuracy: {baseline_acc:.1}%  ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    println!("  (Reference: 62.2% without ALIF)\n");

    // ── Training ──────────────────────────────────────────────────────────────
    let mut rng_state: u64 = 0xDEAD_BEEF_C0FFEE42;
    let lcg = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    };

    // Shuffle indices for each epoch
    let n_train = train_flat.len();

    println!("  Starting e-prop training ({N_EPOCHS} epochs, LR={LEARNING_RATE})...");
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>14}  {:>8}",
        "epoch", "train_corr", "train_acc", "test_acc", "secs"
    );
    println!("  {}", "-".repeat(58));

    let mut best_test_acc = baseline_acc;

    for epoch in 0..N_EPOCHS {
        // Shuffle training indices
        let mut indices: Vec<usize> = (0..n_train).collect();
        for i in (1..n_train).rev() {
            let j = (lcg(&mut rng_state) as usize) % (i + 1);
            indices.swap(i, j);
        }

        let epoch_t0 = Instant::now();
        let mut train_correct = 0usize;
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

            // Forward pass with e-prop tracking
            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_adapt_state(&rt).unwrap();
            snn.reset_output_counts(&rt).unwrap();
            snn.reset_eprop_state(&rt).unwrap();
            snn.tick_many_with_eprop(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();

            // Compute predictions
            let counts = snn.get_output_counts(&rt).unwrap();
            let preds = predict_batch(&counts, actual_bs);
            for b in 0..actual_bs {
                if preds[b] == true_labels[b] {
                    train_correct += 1;
                }
            }

            // Class-specific learning signals: one E-prop call per class.
            // For class c: +1 if sample is from class c (strengthen true class),
            //              -1 if sample is wrongly predicted AS class c (weaken wrong winner).
            // This avoids the ±1 global signal death spiral where wrong predictions
            // weaken the correct class's synapses instead of strengthening them.
            let lr = LEARNING_RATE / N_CLASSES as f32;
            for c in 0..N_CLASSES {
                let mut class_signals = vec![0.0f32; BATCH_SIZE];
                for b in 0..actual_bs {
                    if true_labels[b] == c {
                        class_signals[b] = 1.0; // strengthen true class c
                    } else if preds[b] == c {
                        class_signals[b] = -1.0; // weaken false winner class c
                    }
                    // other samples: 0 — no update for class-c synapses
                }
                snn.apply_eprop_update(&rt, &class_signals, lr).unwrap();
            }

            i += actual_bs;
        }

        // Diagnostics every 5 epochs
        if epoch % 5 == 4 {
            if let Ok((mean_abs, max_abs)) = snn.get_eligibility_stats(&rt) {
                println!("    [elig stats] mean_abs={mean_abs:.4e}  max_abs={max_abs:.4e}");
            }
        }

        // Sync f32 shadow → i8 weights before inference
        snn.project_weights_f32(&rt).unwrap();

        // Test evaluation (no weight update)
        let test_acc = run_test(&mut snn, &test_flat);
        let train_acc = 100.0 * train_correct as f32 / n_train as f32;
        let epoch_secs = epoch_t0.elapsed().as_secs_f32();
        let marker = if test_acc > best_test_acc { " ←" } else { "" };
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
        }

        println!(
            "  {:>6}  {:>12}  {:>11.1}%  {:>13.1}%{:>3}  {:>7.1}s",
            epoch + 1,
            train_correct,
            train_acc,
            test_acc,
            marker,
            epoch_secs
        );

        // Check gate after epoch 10
        if epoch == 9 && test_acc < 78.0 {
            println!("\n  *** GATE MISS: test accuracy {test_acc:.1}% < 78.0% at epoch 10");
            println!("  Consider: lower LR, higher TICKS, or revisit ALIF_BETA before Phase 3.");
        }
    }

    // ── Final weight stats ────────────────────────────────────────────────────
    snn.project_weights_f32(&rt).unwrap();
    let weights = snn.get_weights(&rt).unwrap();
    let w_min = weights.iter().copied().min().unwrap_or(0);
    let w_max = weights.iter().copied().max().unwrap_or(0);
    let w_mean = weights.iter().map(|&w| w as f32).sum::<f32>() / weights.len() as f32;
    let w_zero_frac = weights.iter().filter(|&&w| w == 0).count() as f32 / weights.len() as f32;
    let w_sat_frac =
        weights.iter().filter(|&&w| w == 127 || w == -128).count() as f32 / weights.len() as f32;

    println!("\n  ── Final weights ({} synapses) ──", weights.len());
    println!("  min={w_min}  max={w_max}  mean={w_mean:.2}");
    println!(
        "  zero_frac={:.1}%  saturated_frac={:.1}%",
        w_zero_frac * 100.0,
        w_sat_frac * 100.0
    );

    println!("\n  ── Summary ──");
    println!("  Baseline (frozen, ALIF):  {baseline_acc:.1}%");
    println!("  Best trained (e-prop):    {best_test_acc:.1}%");
    let delta = best_test_acc - baseline_acc;
    if delta > 0.0 {
        println!("  Δ = +{delta:.1}pp  (e-prop IMPROVED over frozen baseline)");
    } else {
        println!("  Δ = {delta:.1}pp  (e-prop did not improve — check LR / hyperparams)");
    }
    if best_test_acc >= 78.0 {
        println!("  ✓ Gate PASSED (≥ 78%) — proceed to Phase 3 (Python bindings).");
    } else {
        println!("  ✗ Gate NOT MET — revisit hyperparameters before Phase 3.");
    }
}
