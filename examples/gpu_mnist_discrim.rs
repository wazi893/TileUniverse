//! Milestone 3 Phase 3 / Milestone 4 Phases 2-3:
//! GPU Discriminative-Channel SNN — Full MNIST (10-class)
//!
//! Uses a **2-layer** (input → hidden) discriminative channel SNN where the
//! hidden layer IS the readout.  For each test image the predicted class is:
//!
//! ```text
//! predicted = argmax over c of  (sum(output_counts[c*H .. (c+1)*H]) × T_c)
//! ```
//!
//! ## Milestone 4 additions vs original Phase 3 example
//!
//! - **Validation split**: 10% of training data (stratified) used for calibration.
//!   Test set is touched only for final reporting.
//! - **Confusion matrix**: 10×10 table + top confused pairs.
//! - **Fisher LDA scoring**: Optional (FISHER=true) — replaces raw mean-diff pixel
//!   selection with Fisher ratio (mean_diff² / variance_within+variance_between).
//! - **Threshold α**: Sweep from calibration; `T = α·V_self + (1-α)·V_rival`.
//!   Best α locked from validation before test evaluation.
//! - **Cross-class inhibition**: Optional (INHIBIT=true) — sparse inhibitory
//!   synapses from each class's hidden pool to rival classes.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_discrim
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
        discriminative_2layer_thresholds, fisher_discriminative_scores,
        generate_discriminative_2layer_csr_inhibitory, generate_discriminative_2layer_csr_weighted,
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

    // ── Milestone 4 flags ────────────────────────────────────────────────────
    // Set FISHER=true to use Fisher LDA pixel scoring (Phase 2).
    // Set INHIBIT=true to add cross-class inhibitory synapses (Phase 3).
    const FISHER: bool = true;
    const INHIBIT: bool = false; // enable after validating confusion matrix
    const INHIBIT_CROSS: f32 = 0.3; // inhibitory weight as fraction of w_max
    const INHIBIT_PROB: f32 = 0.1; // sparse cross-channel inhibitory connections

    // ── Milestone 5 flags ────────────────────────────────────────────────────
    // Set ALIF=true to validate adaptive threshold doesn't regress accuracy.
    // Gate: accuracy must be ≥ 61.2% with ALIF enabled (≤ 1pp regression).
    const ALIF: bool = true;
    const ALIF_ALPHA: f32 = 0.967; // exp(-1/30): 30-tick adaptation time constant
    const ALIF_BETA: f32 = 0.1; // each spike raises threshold by 10% of a unit

    // α sweep values for threshold calibration (best locked from validation).
    // T = α·V_ss_self + (1-α)·V_ss_max_rival
    const ALPHA_VALUES: &[f32] = &[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 4: GPU Discriminative-Channel SNN (Full MNIST)");
    println!("  K={K} · H={H} · Fisher={FISHER} · Inhibit={INHIBIT}");
    println!(
        "  Batch={BATCH_SIZE} · Ticks={TICKS} · α-sweep: {} values",
        ALPHA_VALUES.len()
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
    let mut all_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    let mut test_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    for (img, &lbl) in train_ds.images.iter().zip(train_ds.labels.iter()) {
        let c = lbl as usize;
        if c < N_CLASSES {
            all_per_class[c].push(img.clone());
        }
    }
    for (img, &lbl) in test_ds.images.iter().zip(test_ds.labels.iter()) {
        let c = lbl as usize;
        if c < N_CLASSES {
            test_per_class[c].push(img.clone());
        }
    }

    // ── Stratified train / validation split (90% / 10%) ──────────────────────
    // Validation: every 10th sample per class (sorted by index), rest is train.
    let mut train_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    let mut val_per_class: Vec<Vec<Vec<u8>>> = vec![Vec::new(); N_CLASSES];
    for c in 0..N_CLASSES {
        for (idx, img) in all_per_class[c].iter().enumerate() {
            if idx % 10 == 0 {
                val_per_class[c].push(img.clone());
            } else {
                train_per_class[c].push(img.clone());
            }
        }
    }
    let n_train_total: usize = train_per_class.iter().map(|v| v.len()).sum();
    let n_val_total: usize = val_per_class.iter().map(|v| v.len()).sum();
    println!(
        "  Split: {} train, {} val, {} test",
        n_train_total,
        n_val_total,
        test_per_class.iter().map(|v| v.len()).sum::<usize>()
    );

    // ── Class statistics (use train split only) ───────────────────────────────
    let mut avgs = vec![[0f32; N_PIXELS]; N_CLASSES];
    let mut vars = vec![[0f32; N_PIXELS]; N_CLASSES];
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

        // Variance (unbiased)
        for img in &train_per_class[c] {
            for i in 0..N_PIXELS {
                let diff = img[i] as f32 - avgs[c][i];
                vars[c][i] += diff * diff;
            }
        }
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

    // ── Pixel scoring (Fisher LDA or raw mean-diff) ───────────────────────────
    let fisher_scores_opt = if FISHER {
        Some(fisher_discriminative_scores(
            &avgs, &vars, N_CLASSES, N_PIXELS,
        ))
    } else {
        None
    };

    let mut pix_per_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    let mut d_norms: Vec<Vec<f32>> = vec![Vec::new(); N_CLASSES];

    for c in 0..N_CLASSES {
        let mut scores: Vec<(usize, f32)> = if let Some(ref fs) = fisher_scores_opt {
            // Fisher LDA: high ratio = large mean_diff AND low within-class variance
            (0..N_PIXELS).map(|i| (i, fs[c][i])).collect()
        } else {
            // Original mean-diff (one-vs-rest)
            (0..N_PIXELS)
                .map(|i| {
                    let others = (global_sum[i] - avgs[c][i]) / (N_CLASSES - 1) as f32;
                    (i, avgs[c][i] - others)
                })
                .collect()
        };
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<usize> = scores[..K].iter().map(|(i, _)| *i).collect();
        // d_norm derived from raw mean-diff (not Fisher score) — maintains GPU V_ss semantics
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

    if FISHER {
        println!("  Using Fisher LDA pixel scoring.");
    } else {
        println!("  Using mean-diff pixel scoring (original).");
    }

    // ── GPU V_ss function ─────────────────────────────────────────────────────
    let w_scale = 2.0_f32;
    let leak_gap = 256.0_f32 - LEAK as f32;
    let gpu_vss = |cc: usize, cj: usize, pix: &[Vec<usize>], dn: &[Vec<f32>]| -> f32 {
        let weighted_sum: f32 = pix[cc]
            .iter()
            .zip(dn[cc].iter())
            .map(|(&px, &d)| avgs[cj][px] * d)
            .sum();
        CONN_PROB * W_MAX as f32 * w_scale * MAX_RATE as f32 / (255.0 * leak_gap) * weighted_sum
    };

    // ── α calibration on validation set ──────────────────────────────────────
    println!("\n  Calibrating threshold α on validation set...");

    // Build SNN once (architecture doesn't change with α)
    let n_inputs = K * N_CLASSES;
    let n_hidden = H * N_CLASSES;
    let n_neurons = n_inputs + n_hidden;
    let n_outputs = n_hidden;

    let d_norms_flat: Vec<f32> = d_norms.iter().flat_map(|row| row.iter().copied()).collect();

    let synapses = if INHIBIT {
        generate_discriminative_2layer_csr_inhibitory(
            K,
            H,
            N_CLASSES,
            CONN_PROB,
            W_MAX,
            &d_norms_flat,
            INHIBIT_CROSS,
            INHIBIT_PROB,
            42,
        )
    } else {
        generate_discriminative_2layer_csr_weighted(
            K,
            H,
            N_CLASSES,
            CONN_PROB,
            W_MAX,
            &d_norms_flat,
            42,
        )
    };

    let leaks = vec![LEAK; n_neurons];
    let rt = CudaRuntime::new().expect("CUDA init");

    let val_flat: Vec<(Vec<u8>, usize)> = val_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();

    let encode_image = |img: &[u8]| -> Vec<u8> {
        let mut rates = vec![0u8; K * N_CLASSES];
        for c in 0..N_CLASSES {
            for (i, &pix) in pix_per_class[c].iter().enumerate() {
                rates[c * K + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
            }
        }
        rates
    };

    let run_inference = |snn: &mut GpuBatchSNN,
                         dataset: &[(Vec<u8>, usize)],
                         t_per_class: &[i16]|
     -> (usize, Vec<Vec<usize>>) {
        let n = dataset.len();
        let mut correct = 0usize;
        let mut confusion = vec![vec![0usize; N_CLASSES]; N_CLASSES];
        let mut i = 0;
        while i < n {
            let chunk_end = (i + BATCH_SIZE).min(n);
            let chunk = &dataset[i..chunk_end];
            let actual_bs = chunk.len();
            let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
            for (b, (img, _)) in chunk.iter().enumerate() {
                let enc = encode_image(img);
                rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
            }
            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_adapt_state(&rt).unwrap(); // no-op if ALIF not enabled
            snn.reset_output_counts(&rt).unwrap();
            snn.tick_many_with_counting(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();
            let counts = snn.get_output_counts(&rt).unwrap();
            for (b, (_, true_lbl)) in chunk.iter().enumerate() {
                let row = &counts[b];
                let pred = (0..N_CLASSES)
                    .map(|c| {
                        let lo = c * H;
                        let hi = lo + H;
                        let raw: u64 = row[lo..hi].iter().map(|&x| x as u64).sum();
                        raw * t_per_class[c] as u64
                    })
                    .enumerate()
                    .max_by_key(|&(_, s)| s)
                    .map(|(c, _)| c)
                    .unwrap_or(0);
                confusion[*true_lbl][pred] += 1;
                if pred == *true_lbl {
                    correct += 1;
                }
            }
            i += actual_bs;
        }
        (correct, confusion)
    };

    // Sweep α on validation set — find best threshold scaling
    let mut best_alpha = 0.5f32;
    let mut best_val_acc = 0.0f32;
    println!("    {:>5}  {:>10}  {:>10}", "alpha", "val_acc", "");
    for &alpha in ALPHA_VALUES {
        let t_per_class: Vec<i16> = (0..N_CLASSES)
            .map(|c| {
                let vss_cc = gpu_vss(c, c, &pix_per_class, &d_norms);
                let vss_max = (0..N_CLASSES)
                    .filter(|&j| j != c)
                    .map(|j| gpu_vss(c, j, &pix_per_class, &d_norms))
                    .fold(0.0f32, f32::max);
                let t = if vss_cc > vss_max {
                    (alpha * vss_cc + (1.0 - alpha) * vss_max).clamp(1.0, i16::MAX as f32)
                } else {
                    (0.85 * vss_cc).clamp(1.0, i16::MAX as f32)
                };
                t as i16
            })
            .collect();

        let thresholds = discriminative_2layer_thresholds(K, H, N_CLASSES, &t_per_class, 32767);
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
        if ALIF {
            snn.enable_alif(&rt, ALIF_ALPHA, ALIF_BETA)
                .expect("enable_alif");
        }

        let (correct, _) = run_inference(&mut snn, &val_flat, &t_per_class);
        let val_acc = 100.0 * correct as f32 / val_flat.len() as f32;
        let mark = if val_acc > best_val_acc {
            " ← best"
        } else {
            ""
        };
        println!("    {:>5.2}  {:>9.1}%{}", alpha, val_acc, mark);
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_alpha = alpha;
        }
    }
    println!("  Locked: α={best_alpha:.2}  val_acc={best_val_acc:.1}%\n");

    // ── Build final model with best α ─────────────────────────────────────────
    let t_per_class: Vec<i16> = (0..N_CLASSES)
        .map(|c| {
            let vss_cc = gpu_vss(c, c, &pix_per_class, &d_norms);
            let vss_max = (0..N_CLASSES)
                .filter(|&j| j != c)
                .map(|j| gpu_vss(c, j, &pix_per_class, &d_norms))
                .fold(0.0f32, f32::max);
            let t = if vss_cc > vss_max {
                (best_alpha * vss_cc + (1.0 - best_alpha) * vss_max).clamp(1.0, i16::MAX as f32)
            } else {
                (0.85 * vss_cc).clamp(1.0, i16::MAX as f32)
            };
            t as i16
        })
        .collect();

    let thresholds = discriminative_2layer_thresholds(K, H, N_CLASSES, &t_per_class, 32767);
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
    if ALIF {
        snn.enable_alif(&rt, ALIF_ALPHA, ALIF_BETA)
            .expect("enable_alif");
    }

    println!(
        "  Network: {} inputs · {} hidden · {} synapses",
        n_inputs,
        n_hidden,
        synapses.targets.len()
    );
    println!("  Per-class thresholds: {:?}", t_per_class);

    // ── Inference on test set ─────────────────────────────────────────────────
    let test_flat: Vec<(Vec<u8>, usize)> = test_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();

    println!(
        "\n  Running GPU inference on {} test samples...",
        test_flat.len()
    );
    let t0 = Instant::now();
    let (correct, confusion) = run_inference(&mut snn, &test_flat, &t_per_class);
    let elapsed = t0.elapsed().as_secs_f32();
    let accuracy = 100.0 * correct as f32 / test_flat.len() as f32;
    let sps = test_flat.len() as f32 / elapsed;

    // ── Per-class accuracy ────────────────────────────────────────────────────
    println!("\n  Per-class accuracy:");
    let total_per_class: Vec<usize> = (0..N_CLASSES).map(|c| confusion[c].iter().sum()).collect();
    for c in 0..N_CLASSES {
        let acc = if total_per_class[c] > 0 {
            100.0 * confusion[c][c] as f32 / total_per_class[c] as f32
        } else {
            0.0
        };
        println!(
            "    Class {c}: {acc:5.1}%  ({}/{} correct)",
            confusion[c][c], total_per_class[c]
        );
    }

    // ── Confusion matrix ──────────────────────────────────────────────────────
    println!("\n  Confusion matrix (row=true, col=pred):");
    print!("       ");
    for c in 0..N_CLASSES {
        print!("{:5}", c);
    }
    println!();
    for r in 0..N_CLASSES {
        print!("  {:2}:  ", r);
        for c in 0..N_CLASSES {
            if r == c {
                print!("{:5}", confusion[r][c]);
            } else {
                print!("{:5}", confusion[r][c]);
            }
        }
        println!();
    }

    // Top confused pairs
    let mut off_diag: Vec<(usize, usize, usize)> = Vec::new();
    for r in 0..N_CLASSES {
        for c in 0..N_CLASSES {
            if r != c && confusion[r][c] > 0 {
                off_diag.push((confusion[r][c], r, c));
            }
        }
    }
    off_diag.sort_by(|a, b| b.0.cmp(&a.0));
    println!("\n  Top confused pairs (true→pred: count):");
    for &(cnt, r, c) in off_diag.iter().take(5) {
        println!("    {}→{}: {} errors", r, c, cnt);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Overall accuracy: {accuracy:.1}%  ({correct}/{} correct)",
        test_flat.len()
    );
    println!("  Throughput:       {sps:.0} samples/sec");
    println!("  Elapsed:          {elapsed:.2}s");
    println!("  Best α (val):     {best_alpha:.2}  (val_acc={best_val_acc:.1}%)");
    println!("  Fisher scoring:   {FISHER}");
    println!("  Cross inhibit:    {INHIBIT} (scale={INHIBIT_CROSS}, prob={INHIBIT_PROB})");
    println!("  CPU baseline:     61.3%  (7×7 input, H=32)");
    println!("  GPU SNN M3 best:  62.2%  (K=150, α=0.5, no Fisher, no inhibit)");
    if accuracy > 62.2 {
        println!("  ✓ Beats M3 GPU baseline by +{:.1}pp", accuracy - 62.2);
    } else {
        println!("  △ Below M3 GPU baseline by {:.1}pp", 62.2 - accuracy);
    }
    println!("═══════════════════════════════════════════════════════════════");
}
