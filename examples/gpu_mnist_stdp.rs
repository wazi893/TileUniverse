//! Milestone 4 Phase 4: Batch STDP Training — Train→Deploy Path
//!
//! Demonstrates the complete train→deploy loop using `GpuBatchSNN`:
//!
//! 1. Build a discriminative-channel SNN (K=150, H=32) with analytic initialization.
//! 2. Enable STDP weight updates via `enable_stdp()`.
//! 3. Train for N epochs using reward-modulated Hebbian updates:
//!    - reward[b] = +1.0 if prediction correct, -1.0 if wrong
//!    - `apply_hebbian_reward(lr=0.001)` after each batch
//! 4. Evaluate accuracy on test set each epoch.
//!
//! This closes the architectural gap between `GpuFusedSNN` (single-instance R-STDP)
//! and `GpuBatchSNN` (batch inference). After training, `get_weights()` can export
//! weights for deployment in a separate inference-only `GpuBatchSNN`.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_stdp
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
    const K: usize = 150;
    const H: usize = 32;
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB: f32 = 0.5;
    const LEAK: u8 = 230;
    const W_MAX: i8 = 120;

    // ── STDP training parameters ──────────────────────────────────────────────
    const N_EPOCHS: usize = 10;
    const LEARNING_RATE: f32 = 0.001;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 4 Phase 4: Batch STDP Training (Train→Deploy)");
    println!("  K={K} · H={H} · Ticks={TICKS} · Batch={BATCH_SIZE}");
    println!("  Epochs={N_EPOCHS} · LR={LEARNING_RATE}");
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

    // ── Discriminative feature analysis (use full training set) ──────────────
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

    // GPU-accurate per-class thresholds (α=0.5, same as M3 baseline)
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

    // Enable STDP — allocates [B × N] spike count buffer
    snn.enable_stdp(&rt).expect("enable_stdp");
    println!(
        "  STDP enabled. Allocated {:.1} MB for all-neuron spike counts.",
        BATCH_SIZE as f32 * n_neurons as f32 * 4.0 / 1e6
    );

    // ── Flat datasets ────────────────────────────────────────────────────────
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

    // ── Evaluate function (no weight updates) ────────────────────────────────
    let evaluate = |snn: &mut GpuBatchSNN, dataset: &[(Vec<u8>, usize)]| -> f32 {
        let n = dataset.len();
        let mut correct = 0usize;
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
                if pred == *true_lbl {
                    correct += 1;
                }
            }
            i += actual_bs;
        }
        100.0 * correct as f32 / n as f32
    };

    // ── Epoch 0: baseline (no training) ─────────────────────────────────────
    let baseline = evaluate(&mut snn, &test_flat);
    println!("\n  Epoch  0 (baseline):  test={:.1}%", baseline);

    // ── Training loop ────────────────────────────────────────────────────────
    let mut rng = 42u64;
    let mut order: Vec<usize> = (0..n_train).collect();

    println!();
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>10}",
        "Epoch", "Train correct", "Test acc", "Time"
    );
    println!("  {}", "─".repeat(48));

    for epoch in 1..=N_EPOCHS {
        let t_epoch = Instant::now();

        // Fisher-Yates shuffle
        for idx in (1..n_train).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (idx + 1);
            order.swap(idx, j);
        }

        let mut train_correct = 0usize;
        let mut i = 0;

        while i < n_train {
            let chunk_end = (i + BATCH_SIZE).min(n_train);
            let chunk = &train_flat[i..chunk_end];
            let actual_bs = chunk.len();

            // Build rate buffer
            let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
            for (b, &sample_idx) in order[i..chunk_end].iter().enumerate() {
                let (img, _) = &train_flat[sample_idx];
                let enc = encode_image(img);
                rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
            }

            // Forward pass with full spike count tracking
            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_output_counts(&rt).unwrap();
            snn.reset_all_counts(&rt).unwrap();
            snn.tick_many_full_counting(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();

            // Compute predictions + per-sample reward
            let counts = snn.get_output_counts(&rt).unwrap();
            let mut rewards = vec![0.0f32; BATCH_SIZE];
            for (b, &sample_idx) in order[i..chunk_end].iter().enumerate() {
                let (_, true_lbl) = &train_flat[sample_idx];
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
                if pred == *true_lbl {
                    rewards[b] = 1.0;
                    train_correct += 1;
                } else {
                    rewards[b] = -1.0;
                }
            }
            // Zero-out reward for padding slots (last chunk may be shorter)
            for b in actual_bs..BATCH_SIZE {
                rewards[b] = 0.0;
            }

            // Apply Hebbian weight update
            snn.apply_hebbian_reward(&rt, &rewards, TICKS, LEARNING_RATE)
                .unwrap();

            i += actual_bs;
        }

        rt.synchronize().unwrap();

        // Test evaluation (no weight updates)
        let test_acc = evaluate(&mut snn, &test_flat);
        println!(
            "  {:>6}  {:>9}/{:>5}  {:>11.1}%  {:>8.2}s",
            epoch,
            train_correct,
            n_train,
            test_acc,
            t_epoch.elapsed().as_secs_f32()
        );
    }

    // ── Final weight stats ───────────────────────────────────────────────────
    let weights = snn.get_weights(&rt).unwrap();
    let w_min = weights.iter().copied().min().unwrap_or(0);
    let w_max = weights.iter().copied().max().unwrap_or(0);
    let w_mean = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
    let zero_frac = weights.iter().filter(|&&w| w == 0).count() as f32 / weights.len() as f32;

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Final weight stats:  min={w_min}  max={w_max}  mean={w_mean:.2}  zero={:.1}%",
        100.0 * zero_frac
    );
    println!("  Baseline (no STDP):  {:.1}%", baseline);
    println!("  After {} epochs:     see table above", N_EPOCHS);
    println!("  M3 GPU best:         62.2%  (frozen weights, same architecture)");
    println!(
        "  Architecture:        {} inputs → {} hidden",
        n_inputs, n_hidden
    );
    println!("═══════════════════════════════════════════════════════════════");

    println!("\n  [Train→Deploy path demonstrated: weights updated in-place on GPU,");
    println!("   readable via get_weights() for deployment in inference-only SNN]");
}
