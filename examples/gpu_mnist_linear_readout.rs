//! Milestone 4 Phase 1: GPU SNN + Linear Readout — Prove Hidden Feature Separability
//!
//! Runs all 60K training images through the frozen discriminative-channel SNN and
//! collects [60K × 320] spike count feature vectors.  A 10-class logistic regression
//! is then trained (CPU, SGD, 50 epochs) on those features, then evaluated on 10K test
//! images.
//!
//! The gap between the SNN argmax (62.2%) and the logistic readout tells us how much
//! separability the hidden representation contains that the threshold argmax leaves on
//! the table.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_linear_readout
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

    // ── Fixed hyperparameters (same as gpu_mnist_discrim.rs best config) ─────
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

    // ── Logistic regression hyperparameters ──────────────────────────────────
    const SGD_EPOCHS: usize = 50;
    const SGD_LR: f32 = 0.01;
    const SGD_BATCH: usize = 1024;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 4 Phase 1: GPU SNN + Linear Readout");
    println!(
        "  K={K} · H={H} · features={} · logreg epochs={SGD_EPOCHS}",
        H * N_CLASSES
    );
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

    // Group by class
    let mut train_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    let mut test_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    for (img, &lbl) in train_ds.images.iter().zip(train_ds.labels.iter()) {
        let c = lbl as usize;
        if c < N_CLASSES {
            train_per_class[c].push(img.clone());
        }
    }
    for (img, &lbl) in test_ds.images.iter().zip(test_ds.labels.iter()) {
        let c = lbl as usize;
        if c < N_CLASSES {
            test_per_class[c].push(img.clone());
        }
    }

    // ── Discriminative feature analysis (CPU) ────────────────────────────────
    let mut avgs = vec![[0f32; N_PIXELS]; N_CLASSES];
    for c in 0..N_CLASSES {
        for img in &train_per_class[c] {
            for i in 0..N_PIXELS {
                avgs[c][i] += img[i] as f32;
            }
        }
        let n = train_per_class[c].len() as f32;
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

    // GPU-accurate per-class thresholds
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
    let t_per_class: Vec<i16> = (0..N_CLASSES)
        .map(|c| {
            let vss_cc = gpu_vss(c, c);
            let vss_max = (0..N_CLASSES)
                .filter(|&j| j != c)
                .map(|j| gpu_vss(c, j))
                .fold(0.0f32, f32::max);
            let t = if vss_cc > vss_max {
                (0.5 * (vss_cc + vss_max)).clamp(1.0, i16::MAX as f32)
            } else {
                (0.85 * vss_cc).clamp(1.0, i16::MAX as f32)
            };
            t as i16
        })
        .collect();

    // Input encoder
    let encode_image = |img: &[u8]| -> Vec<u8> {
        let mut rates = vec![0u8; K * N_CLASSES];
        for c in 0..N_CLASSES {
            for (i, &pix) in pix_per_class[c].iter().enumerate() {
                rates[c * K + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
            }
        }
        rates
    };

    // ── Build SNN ────────────────────────────────────────────────────────────
    let n_inputs = K * N_CLASSES;
    let n_hidden = H * N_CLASSES;
    let n_neurons = n_inputs + n_hidden;
    let n_outputs = n_hidden;

    let d_norms_flat: Vec<f32> = d_norms.iter().flat_map(|row| row.iter().copied()).collect();
    let synapses = generate_discriminative_2layer_csr_weighted(
        K,
        H,
        N_CLASSES,
        CONN_PROB,
        W_MAX,
        &d_norms_flat,
        42,
    );
    let thresholds = discriminative_2layer_thresholds(K, H, N_CLASSES, &t_per_class, 32767);
    let leaks = vec![LEAK; n_neurons];

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

    println!(
        "  Network: {} inputs · {} hidden · {} synapses",
        n_inputs,
        n_hidden,
        synapses.targets.len()
    );

    // ── Flat train/test datasets ──────────────────────────────────────────────
    let train_flat: Vec<(Vec<u8>, usize)> = train_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();
    let test_flat: Vec<(Vec<u8>, usize)> = test_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();
    let n_train = train_flat.len();
    let n_test = test_flat.len();
    let n_feat = n_outputs; // 320

    // ── Collect GPU spike features (training set) ─────────────────────────────
    println!(
        "\n  Collecting GPU spike features for {} training samples...",
        n_train
    );
    let t_feat = Instant::now();

    let mut x_train: Vec<f32> = Vec::with_capacity(n_train * n_feat);
    let mut y_train: Vec<usize> = Vec::with_capacity(n_train);

    let mut i = 0usize;
    while i < n_train {
        let chunk_end = (i + BATCH_SIZE).min(n_train);
        let chunk = &train_flat[i..chunk_end];
        let actual_bs = chunk.len();

        let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
        for (b, (img, _)) in chunk.iter().enumerate() {
            let enc = encode_image(img);
            rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
        }

        snn.upload_input_batch(&rt, &rates_buf).unwrap();
        snn.reset_state(&rt).unwrap();
        snn.reset_output_counts(&rt).unwrap();
        snn.tick_many_with_counting(&rt, TICKS).unwrap();
        rt.synchronize().unwrap();

        let flat = snn.get_output_counts_flat(&rt).unwrap();
        // flat is [BATCH_SIZE × n_feat]; only use the actual_bs rows
        for b in 0..actual_bs {
            let start = b * n_feat;
            for f in 0..n_feat {
                x_train.push(flat[start + f] as f32);
            }
            y_train.push(chunk[b].1);
        }

        i += actual_bs;
    }
    println!(
        "  Feature collection: {:.2}s  ({:.0} samples/sec)",
        t_feat.elapsed().as_secs_f32(),
        n_train as f32 / t_feat.elapsed().as_secs_f32()
    );

    // ── SNN baseline: T-normalized argmax on train set (same features) ────────
    {
        let mut correct = 0usize;
        for idx in 0..n_train {
            let x = &x_train[idx * n_feat..(idx + 1) * n_feat];
            let pred = (0..N_CLASSES)
                .map(|c| {
                    let raw: f32 = x[c * H..(c + 1) * H].iter().sum();
                    (raw * t_per_class[c] as f32) as u64
                })
                .enumerate()
                .max_by_key(|&(_, s)| s)
                .map(|(c, _)| c)
                .unwrap_or(0);
            if pred == y_train[idx] {
                correct += 1;
            }
        }
        println!(
            "  SNN T-argmax on train: {:.1}%  ({}/{})",
            100.0 * correct as f32 / n_train as f32,
            correct,
            n_train
        );
    }

    // ── Train logistic regression (multi-class softmax, mini-batch SGD) ───────
    println!(
        "\n  Training logistic regression ({} epochs, lr={}, batch={})...",
        SGD_EPOCHS, SGD_LR, SGD_BATCH
    );

    // W: [N_CLASSES × n_feat],  b: [N_CLASSES]
    let mut w = vec![0.0f32; N_CLASSES * n_feat];
    let mut b = vec![0.0f32; N_CLASSES];

    let t_train = Instant::now();
    let mut rng = 12345u64;
    let mut lcg = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f32 / u32::MAX as f32
    };

    // Shuffle index
    let mut order: Vec<usize> = (0..n_train).collect();

    for epoch in 0..SGD_EPOCHS {
        // Fisher-Yates shuffle
        for idx in (1..n_train).rev() {
            let j = (lcg() * (idx + 1) as f32) as usize % (idx + 1);
            order.swap(idx, j);
        }

        let mut batch_start = 0;
        while batch_start < n_train {
            let batch_end = (batch_start + SGD_BATCH).min(n_train);
            let bs = batch_end - batch_start;

            // Accumulate gradients
            let mut dw = vec![0.0f32; N_CLASSES * n_feat];
            let mut db = vec![0.0f32; N_CLASSES];

            for b_idx in batch_start..batch_end {
                let sample = order[b_idx];
                let x = &x_train[sample * n_feat..(sample + 1) * n_feat];
                let label = y_train[sample];

                // Logits = W·x + b
                let mut logits = [0.0f32; 10];
                for c in 0..N_CLASSES {
                    let wc = &w[c * n_feat..(c + 1) * n_feat];
                    logits[c] = b[c];
                    for f in 0..n_feat {
                        logits[c] += wc[f] * x[f];
                    }
                }

                // Softmax
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut p = [0.0f32; 10];
                let mut sum_p = 0.0f32;
                for c in 0..N_CLASSES {
                    p[c] = (logits[c] - max_l).exp();
                    sum_p += p[c];
                }
                for c in 0..N_CLASSES {
                    p[c] /= sum_p;
                }

                // Gradient
                for c in 0..N_CLASSES {
                    let grad = p[c] - if c == label { 1.0 } else { 0.0 };
                    db[c] += grad;
                    let dw_c = &mut dw[c * n_feat..(c + 1) * n_feat];
                    for f in 0..n_feat {
                        dw_c[f] += grad * x[f];
                    }
                }
            }

            // SGD update
            let scale = SGD_LR / bs as f32;
            for c in 0..N_CLASSES {
                b[c] -= scale * db[c];
                let wc = &mut w[c * n_feat..(c + 1) * n_feat];
                let dw_c = &dw[c * n_feat..(c + 1) * n_feat];
                for f in 0..n_feat {
                    wc[f] -= scale * dw_c[f];
                }
            }

            batch_start = batch_end;
        }

        // Log every 10 epochs (quick train acc on 2K samples)
        if (epoch + 1) % 10 == 0 {
            let check = n_train.min(2000);
            let mut correct = 0usize;
            for idx in 0..check {
                let x = &x_train[idx * n_feat..(idx + 1) * n_feat];
                let mut logits = [0.0f32; 10];
                for c in 0..N_CLASSES {
                    let wc = &w[c * n_feat..(c + 1) * n_feat];
                    logits[c] = b[c];
                    for f in 0..n_feat {
                        logits[c] += wc[f] * x[f];
                    }
                }
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(c, _)| c)
                    .unwrap_or(0);
                if pred == y_train[idx] {
                    correct += 1;
                }
            }
            println!(
                "    Epoch {:2}: train[2K] {:.1}%",
                epoch + 1,
                100.0 * correct as f32 / check as f32
            );
        }
    }
    println!("  Training: {:.2}s", t_train.elapsed().as_secs_f32());

    // ── Collect GPU spike features (test set) ─────────────────────────────────
    println!(
        "\n  Collecting GPU spike features for {} test samples...",
        n_test
    );
    let t_test = Instant::now();

    let mut x_test: Vec<f32> = Vec::with_capacity(n_test * n_feat);
    let mut y_test: Vec<usize> = Vec::with_capacity(n_test);

    let mut snn_correct_test = 0usize; // T-argmax accuracy on test for comparison
    let mut i = 0usize;
    while i < n_test {
        let chunk_end = (i + BATCH_SIZE).min(n_test);
        let chunk = &test_flat[i..chunk_end];
        let actual_bs = chunk.len();

        let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
        for (b, (img, _)) in chunk.iter().enumerate() {
            let enc = encode_image(img);
            rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
        }

        snn.upload_input_batch(&rt, &rates_buf).unwrap();
        snn.reset_state(&rt).unwrap();
        snn.reset_output_counts(&rt).unwrap();
        snn.tick_many_with_counting(&rt, TICKS).unwrap();
        rt.synchronize().unwrap();

        let flat = snn.get_output_counts_flat(&rt).unwrap();

        for b in 0..actual_bs {
            let start = b * n_feat;
            let x: Vec<f32> = flat[start..start + n_feat]
                .iter()
                .map(|&v| v as f32)
                .collect();

            // SNN T-argmax
            let snn_pred = (0..N_CLASSES)
                .map(|c| {
                    let raw: f32 = x[c * H..(c + 1) * H].iter().sum();
                    (raw * t_per_class[c] as f32) as u64
                })
                .enumerate()
                .max_by_key(|&(_, s)| s)
                .map(|(c, _)| c)
                .unwrap_or(0);
            if snn_pred == chunk[b].1 {
                snn_correct_test += 1;
            }

            x_test.extend_from_slice(&x);
            y_test.push(chunk[b].1);
        }

        i += actual_bs;
    }

    // ── Evaluate logistic regression on test set ──────────────────────────────
    let mut correct = 0usize;
    let mut correct_per_class = vec![0usize; N_CLASSES];
    let mut total_per_class = vec![0usize; N_CLASSES];

    for idx in 0..n_test {
        let x = &x_test[idx * n_feat..(idx + 1) * n_feat];
        let true_lbl = y_test[idx];

        let mut logits = [0.0f32; 10];
        for c in 0..N_CLASSES {
            let wc = &w[c * n_feat..(c + 1) * n_feat];
            logits[c] = b[c];
            for f in 0..n_feat {
                logits[c] += wc[f] * x[f];
            }
        }
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(c, _)| c)
            .unwrap_or(0);

        total_per_class[true_lbl] += 1;
        if pred == true_lbl {
            correct += 1;
            correct_per_class[true_lbl] += 1;
        }
    }

    let elapsed_test = t_test.elapsed().as_secs_f32();
    let acc_logreg = 100.0 * correct as f32 / n_test as f32;
    let acc_snn = 100.0 * snn_correct_test as f32 / n_test as f32;

    // ── Report ────────────────────────────────────────────────────────────────
    println!("\n  Per-class accuracy (logistic readout vs SNN argmax):");
    for c in 0..N_CLASSES {
        let acc = if total_per_class[c] > 0 {
            100.0 * correct_per_class[c] as f32 / total_per_class[c] as f32
        } else {
            0.0
        };
        println!(
            "    Class {c}: {acc:5.1}%  ({}/{} correct)",
            correct_per_class[c], total_per_class[c]
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Results:");
    println!("    SNN T-argmax (baseline):   {acc_snn:.1}%");
    println!("    Logistic readout:          {acc_logreg:.1}%");
    if acc_logreg > acc_snn {
        println!(
            "    Hidden features add:       +{:.1}pp separability",
            acc_logreg - acc_snn
        );
    } else {
        println!(
            "    Hidden features add:       {:.1}pp (no improvement)",
            acc_logreg - acc_snn
        );
    }
    println!("    CPU baseline (Phase 3a):   61.3%");
    println!("    GPU SNN M3 best:           62.2%");
    println!(
        "    Feature collection:        {:.2}s",
        t_feat.elapsed().as_secs_f32()
    );
    println!(
        "    Logreg training:           {:.2}s",
        t_train.elapsed().as_secs_f32()
    );
    println!("    Test inference+eval:       {:.2}s", elapsed_test);
    println!("═══════════════════════════════════════════════════════════════");

    // Gate check
    if acc_logreg >= 67.0 {
        println!("  ✓ Gate PASSED (≥67%): hidden features have extra separability.");
        println!("    Proceed to Phase 2 (Fisher scoring + threshold α sweep).");
    } else if acc_logreg >= 65.0 {
        println!("  △ Gate MARGINAL (65-67%): some extra separability present.");
        println!("    Consider Phase 2 (calibration) before Phase 4 (STDP).");
    } else {
        println!("  ✗ Gate FAILED (<65%): revisit K or H before continuing.");
    }
    println!("═══════════════════════════════════════════════════════════════");
}
