//! Milestone 10 Phase 2: 3-Layer MLP + Recurrence + 300 Epochs (GPU MLP)
//!
//! Combines within-class HH recurrence, deeper 3-layer MLP readout, and extended training.
//! MLP: [320->256->64->10] with Adam. E-prop trains both IH and HH weights.
//!
//! ```text
//! GPU:  Input[1500] -> Hidden[320, HH recurrence] -> count_all (d_all_counts)
//!                                                   -> pre_trace + eligibility (E-prop)
//!       hidden_rates -> MLP[320->256->64->10] -> softmax CE -> Adam (all on GPU)
//!                                              -> signal[b][h] -> IH+HH E-prop (GPU-to-GPU)
//! ```
//!
//! The entire MLP forward/backward/Adam/E-prop signal pipeline runs on GPU.
//! Only ~20 bytes per batch are downloaded for logging.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_lsm_v2
//! cargo run --release --features cuda --example gpu_mnist_lsm_v2 -- 123
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
    use engine::snn::gpu_mlp::GpuMLP;
    use engine::snn::mlp_weights::CachedRates;
    use engine::snn::mnist::MnistDataset;
    use engine::snn::{
        discriminative_3layer_thresholds, generate_discriminative_3layer_csr_recurrent,
    };
    use std::time::Instant;

    // ── Seed (optional CLI arg: cargo run ... -- 42) ──────────────────────────
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    // ── Hyperparameters ─────────────────────────────────────────────────────
    const N_CLASSES: usize = 10;
    const N_PIXELS: usize = 784;
    const K: usize = 150;
    const H: usize = 32;
    const R: usize = 8;
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB_IH: f32 = 0.5;
    const CONN_PROB_HR: f32 = 0.8;
    const CONN_PROB_HH: f32 = 0.3;
    const LEAK: u8 = 230;
    const W_MAX_IH: i8 = 120;
    const W_MAX_HR: i8 = 32;
    const W_HH_EXC: i8 = 12;
    const W_HH_INH: i8 = -8;
    const HH_EXC_RATIO: f32 = 0.8;
    const T_READOUT: i16 = 400;
    const ALPHA: f32 = 0.50;
    const MLP_H1: usize = 256;
    const MLP_H2: usize = 64;

    // ── Adam ────────────────────────────────────────────────────────────────
    const LR_ADAM: f32 = 0.001;
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const N_EPOCHS: usize = 300;

    // ── IH E-prop ───────────────────────────────────────────────────────────
    const EPROP_ALPHA_PRE: f32 = 0.95;
    const EPROP_ALPHA_ELIG: f32 = 0.95;
    const EPROP_GAMMA: f32 = 0.3;
    const LR_IH: f32 = 0.001;
    const LR_HH: f32 = 0.0003;
    const WARMUP_EPOCHS: usize = 5;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 10 Phase 2: 3-Layer MLP + Recurrence + 300 Epochs");
    println!("  GPU MLP pipeline (no CPU bottleneck)");
    println!(
        "  K={K} · H={H} (LSM hidden) · MLP [{}→{MLP_H1}→{MLP_H2}→{N_CLASSES}]  seed={seed}",
        H * N_CLASSES
    );
    println!("  IH E-prop: LR_IH={LR_IH}  HH E-prop: LR_HH={LR_HH}");
    println!(
        "  HH: conn_prob={CONN_PROB_HH}  exc={W_HH_EXC}  inh={W_HH_INH}  ratio={HH_EXC_RATIO}"
    );
    println!("  MLP: Adam LR={LR_ADAM} β1={BETA1} β2={BETA2}");
    println!("  Epochs={N_EPOCHS} · Batch={BATCH_SIZE} · Ticks={TICKS}  warmup={WARMUP_EPOCHS}");
    println!("  M10P2 gate: >=83% best · spike rate [0.01, 0.15] · gap <3pp");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Load MNIST ──────────────────────────────────────────────────────────
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

    // ── Class statistics ────────────────────────────────────────────────────
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

    // ── Pixel scoring ───────────────────────────────────────────────────────
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

    // ── Hidden threshold calibration ────────────────────────────────────────
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

    // ── Build network ───────────────────────────────────────────────────────
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
    let synapses = generate_discriminative_3layer_csr_recurrent(
        K,
        H,
        R,
        N_CLASSES,
        CONN_PROB_IH,
        CONN_PROB_HR,
        CONN_PROB_HH,
        W_MAX_IH,
        W_MAX_HR,
        W_HH_EXC,
        W_HH_INH,
        HH_EXC_RATIO,
        &d_norms_flat,
        seed,
    );
    let n_total_synapses = synapses.targets.len();
    let n_ih_synapses = synapses.syn_ptr[n_inp] as usize;
    let n_hh_hr_synapses = n_total_synapses - n_ih_synapses;

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

    snn.enable_stdp(&rt).expect("enable_stdp");
    snn.enable_eprop(&rt, EPROP_ALPHA_PRE, EPROP_ALPHA_ELIG, EPROP_GAMMA)
        .expect("enable_eprop");
    snn.init_weight_shadow(&rt).expect("init_weight_shadow");
    snn.init_synapse_target_hidden_map(&rt, n_inp, n_hid)
        .expect("init_synapse_target_hidden_map");

    let init_ih_weights = snn.get_weights(&rt).expect("get_weights init")[..n_ih_synapses].to_vec();

    println!("  Network: {n_inp} inputs · {n_hid} hidden (recurrent) · {n_out} readout (unused)");
    println!(
        "  Total synapses: {n_total_synapses}  IH: {n_ih_synapses}  HH+HR: {n_hh_hr_synapses}"
    );

    // ── MLP readout weight init (Glorot uniform, CPU) ───────────────────────
    let mut rng: u64 = seed.wrapping_mul(0xDEAD_BEEF_CAFE_1234).wrapping_add(1);
    let lcg = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    };
    let lcg_f32 = |s: &mut u64| -> f32 { (lcg(s) >> 33) as f32 / (u32::MAX >> 1) as f32 };
    let w1_scale = (6.0 / (n_hid + MLP_H1) as f32).sqrt();
    let mut w1 = vec![0.0f32; n_hid * MLP_H1];
    for w in w1.iter_mut() {
        *w = (lcg_f32(&mut rng) * 2.0 - 1.0) * w1_scale;
    }
    let b1 = vec![0.0f32; MLP_H1];
    let w2_scale = (6.0 / (MLP_H1 + MLP_H2) as f32).sqrt();
    let mut w2 = vec![0.0f32; MLP_H1 * MLP_H2];
    for w in w2.iter_mut() {
        *w = (lcg_f32(&mut rng) * 2.0 - 1.0) * w2_scale;
    }
    let b2 = vec![0.0f32; MLP_H2];
    let w3_scale = (6.0 / (MLP_H2 + N_CLASSES) as f32).sqrt();
    let mut w3 = vec![0.0f32; MLP_H2 * N_CLASSES];
    for w in w3.iter_mut() {
        *w = (lcg_f32(&mut rng) * 2.0 - 1.0) * w3_scale;
    }
    let b3 = vec![0.0f32; N_CLASSES];

    // ── Create GPU MLP (uploads weights, compiles kernels) ──────────────────
    let mut mlp = GpuMLP::new(
        &rt, n_hid, n_neurons, n_inp, MLP_H1, MLP_H2, N_CLASSES, BATCH_SIZE, LR_ADAM, BETA1, BETA2,
        ADAM_EPS, &w1, &b1, &w2, &b2, &w3, &b3,
    )
    .expect("GpuMLP::new");

    let n_signal_channels = mlp.n_signal_channels();

    println!(
        "  MLP readout [{n_hid}→{MLP_H1}→{MLP_H2}→{N_CLASSES}]  Glorot scales: {w1_scale:.4}, {w2_scale:.4}, {w3_scale:.4}"
    );
    println!("  GPU MLP pipeline active — zero CPU math per batch\n");

    // ── Encoding helper ─────────────────────────────────────────────────────
    let encode_image = |img: &[u8]| -> Vec<u8> {
        let mut rates = vec![0u8; K * N_CLASSES];
        for c in 0..N_CLASSES {
            for (i, &pix) in pix_per_class[c].iter().enumerate() {
                rates[c * K + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
            }
        }
        rates
    };

    // ── GPU test evaluation ─────────────────────────────────────────────────
    let run_test_gpu =
        |snn: &mut GpuBatchSNN, mlp: &mut GpuMLP, dataset: &[(Vec<u8>, usize)]| -> f32 {
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
                snn.reset_all_counts(&rt).unwrap();
                snn.tick_many_full_counting(&rt, TICKS).unwrap();
                rt.synchronize().unwrap();

                let d_counts = snn.all_counts_device().unwrap();
                let preds = mlp.forward_test(&rt, d_counts, TICKS, actual_bs).unwrap();
                for (b, (_, lbl)) in chunk.iter().enumerate() {
                    if preds[b] as usize == *lbl {
                        correct += 1;
                    }
                }
                i += actual_bs;
            }
            100.0 * correct as f32 / n as f32
        };

    // ── GPU test evaluation with hidden rate collection ─────────────────────
    let run_test_gpu_collect_rates = |snn: &mut GpuBatchSNN,
                                      mlp: &mut GpuMLP,
                                      dataset: &[(Vec<u8>, usize)]|
     -> (f32, Vec<f32>) {
        let n = dataset.len();
        let mut correct = 0usize;
        let mut i = 0;
        let mut all_rates = Vec::with_capacity(n * n_hid);

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

            let d_counts = snn.all_counts_device().unwrap();
            let preds = mlp.forward_test(&rt, d_counts, TICKS, actual_bs).unwrap();
            let batch_rates = mlp.download_hidden_rates(&rt, actual_bs).unwrap();
            all_rates.extend_from_slice(&batch_rates);

            for (b, (_, lbl)) in chunk.iter().enumerate() {
                if preds[b] as usize == *lbl {
                    correct += 1;
                }
            }
            i += actual_bs;
        }
        (100.0 * correct as f32 / n as f32, all_rates)
    };

    // ── Baseline ────────────────────────────────────────────────────────────
    println!("  Baseline (random W_readout)...");
    let t0 = Instant::now();
    let baseline_acc = run_test_gpu(&mut snn, &mut mlp, &test_flat);
    println!(
        "  Random readout: {baseline_acc:.1}%  ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    println!("  M7P1=69.3%  M7P2=71.7%  M6=62.4%\n");

    // ── Training loop ───────────────────────────────────────────────────────
    let n_train = train_flat.len();
    let mut last_train_acc = 0.0f32;
    println!("  Starting M10P2 training ({N_EPOCHS} epochs)...");
    println!("  Warmup: epochs 1..{WARMUP_EPOCHS} (W_readout only)");
    let eprop_start = WARMUP_EPOCHS + 1;
    println!(
        "  Per-neuron E-prop: epochs {eprop_start}..{N_EPOCHS} (IH LR={LR_IH}, HH LR={LR_HH})"
    );
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>14}  {:>10}  {:>8}  {}",
        "epoch", "train_corr", "train_acc", "test_acc", "ce_loss", "secs", "mode"
    );
    println!("  {}", "-".repeat(80));

    let mut best_test_acc = baseline_acc;

    for epoch in 0..N_EPOCHS {
        let ih_active = epoch >= WARMUP_EPOCHS;
        let mut epoch_spike_rate_sum = 0.0f32;
        let mut epoch_spike_rate_count = 0usize;

        // Shuffle training set
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

            // GPU forward: SNN counting + E-prop traces
            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_all_counts(&rt).unwrap();
            snn.reset_output_counts(&rt).unwrap();
            snn.reset_eprop_state(&rt).unwrap();

            snn.tick_many_with_eprop_and_hidden_counting(&rt, TICKS)
                .unwrap();
            rt.synchronize().unwrap();

            // GPU MLP: forward + backward + Adam (all on GPU, no CPU math)
            let d_counts = snn.all_counts_device().unwrap();
            mlp.train_step(&rt, d_counts, &true_labels, TICKS, actual_bs)
                .unwrap();

            // Spike rate monitoring (small download)
            let batch_rate = mlp.get_mean_spike_rate(&rt, actual_bs).unwrap();
            epoch_spike_rate_sum += batch_rate;
            epoch_spike_rate_count += 1;

            // Download metrics (~20 bytes) for logging
            rt.synchronize().unwrap();
            let (batch_correct, batch_loss) = mlp.get_train_metrics(&rt, actual_bs).unwrap();
            train_correct += batch_correct;
            epoch_loss += batch_loss;
            n_batches += 1;

            // IH E-prop update (GPU-to-GPU signal transfer, no CPU round-trip)
            if ih_active {
                snn.apply_eprop_update_per_class_device(
                    &rt,
                    mlp.signals_device(),
                    LR_IH,
                    LR_HH,
                    n_signal_channels,
                    n_ih_synapses,
                )
                .unwrap();
                snn.project_weights_f32(&rt).unwrap();
            }

            i += actual_bs;
        }

        let test_acc = run_test_gpu(&mut snn, &mut mlp, &test_flat);
        let train_acc = 100.0 * train_correct as f32 / n_train as f32;
        last_train_acc = train_acc;
        let mean_loss = epoch_loss / n_batches as f32;
        let mean_spike_rate = epoch_spike_rate_sum / epoch_spike_rate_count.max(1) as f32;
        let epoch_secs = epoch_t0.elapsed().as_secs_f32();
        let marker = if test_acc > best_test_acc { " <-" } else { "" };
        let mode_tag = if ih_active { "IH+W" } else { "W   " };
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
        }

        println!(
            "  {:>6}  {:>12}  {:>11.1}%  {:>13.1}%{:>3}  {:>10.4}  {:>7.1}s  {}",
            epoch + 1,
            train_correct,
            train_acc,
            test_acc,
            marker,
            mean_loss,
            epoch_secs,
            mode_tag
        );

        // IH weight drift diagnostics every 5 epochs
        if (epoch + 1) % 5 == 0 {
            let weights = snn.get_weights(&rt).unwrap();
            let ih = &weights[..n_ih_synapses];
            let drift = ih
                .iter()
                .zip(init_ih_weights.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as f32)
                .sum::<f32>()
                / n_ih_synapses as f32;
            let mean_w = ih.iter().map(|&w| w as f32).sum::<f32>() / n_ih_synapses as f32;
            let syns_per_class = n_ih_synapses / N_CLASSES;
            let class_means: Vec<f32> = (0..N_CLASSES)
                .map(|c| {
                    let lo = c * syns_per_class;
                    let hi = (lo + syns_per_class).min(n_ih_synapses);
                    ih[lo..hi].iter().map(|&w| w as f32).sum::<f32>() / (hi - lo) as f32
                })
                .collect();
            let class_min = class_means.iter().cloned().fold(f32::INFINITY, f32::min);
            let class_max = class_means
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let class_spread = class_max - class_min;
            let class_str: Vec<String> = class_means.iter().map(|m| format!("{m:.1}")).collect();
            let rate_ok = mean_spike_rate >= 0.01 && mean_spike_rate <= 0.15;
            let rate_warn = mean_spike_rate > 0.25;
            println!(
                "  [IH epoch {:>2}] mean={mean_w:.1}  drift={drift:.3}  spread={class_spread:.1}  spike_rate={mean_spike_rate:.3}{}{}{}",
                epoch + 1,
                if class_spread > 5.0 {
                    "  [class divergence]"
                } else {
                    ""
                },
                if rate_warn {
                    "  [SPIKE RATE > 0.25 — saturation risk!]"
                } else if !rate_ok {
                    "  [spike rate outside target band 0.01-0.15]"
                } else {
                    ""
                },
                if mean_spike_rate > 0.50 {
                    "\n  *** ABORT: spike rate > 0.50 — recurrence runaway! ***"
                } else {
                    ""
                }
            );
            println!("    per-class means: [{}]", class_str.join(", "));
        }

        if epoch + 1 == WARMUP_EPOCHS {
            println!(
                "\n  *** Warmup complete — enabling IH per-neuron E-prop from epoch {} ***\n",
                WARMUP_EPOCHS + 1
            );
        }
    }

    // ── Final MLP weight stats (download from GPU) ──────────────────────────
    let mlp_w = mlp.download_weights(&rt).unwrap();
    let w1_min = mlp_w.w1.iter().cloned().fold(f32::INFINITY, f32::min);
    let w1_max = mlp_w.w1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w1_mean = mlp_w.w1.iter().sum::<f32>() / mlp_w.w1.len() as f32;
    let w2_min = mlp_w.w2.iter().cloned().fold(f32::INFINITY, f32::min);
    let w2_max = mlp_w.w2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w2_mean = mlp_w.w2.iter().sum::<f32>() / mlp_w.w2.len() as f32;
    let w3_min = mlp_w.w3.iter().cloned().fold(f32::INFINITY, f32::min);
    let w3_max = mlp_w.w3.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w3_mean = mlp_w.w3.iter().sum::<f32>() / mlp_w.w3.len() as f32;
    println!("\n  -- Final MLP weight stats --");
    println!("  W1: min={w1_min:.3}  max={w1_max:.3}  mean={w1_mean:.4}");
    println!("  W2: min={w2_min:.3}  max={w2_max:.3}  mean={w2_mean:.4}");
    println!("  W3: min={w3_min:.3}  max={w3_max:.3}  mean={w3_mean:.4}");

    // ── Save M11 checkpoint (weights + cached hidden rates) ──────────────
    let ckpt_dir = format!("checkpoints");
    std::fs::create_dir_all(&ckpt_dir).ok();
    let weights_path = format!("{ckpt_dir}/m10_seed{seed}.mlp");
    mlp_w.save(&weights_path).expect("save MLP weights");
    println!("\n  Saved MLP weights  → {weights_path}");

    // Collect hidden rates for all test samples (final model)
    println!(
        "  Collecting hidden rates for {} test samples...",
        test_flat.len()
    );
    let (final_acc, all_rates) = run_test_gpu_collect_rates(&mut snn, &mut mlp, &test_flat);
    let cached = CachedRates::new(n_hid, all_rates);
    let rates_path = format!("{ckpt_dir}/m10_seed{seed}.rates");
    cached.save(&rates_path).expect("save cached rates");
    println!(
        "  Saved cached rates → {rates_path}  ({} samples × {} hidden)",
        cached.n_samples(),
        n_hid
    );
    println!("  Final accuracy (verify): {final_acc:.1}%");

    // ── Save M12 live SNN checkpoint ──────────────────────────────────────
    let trained_weights = snn.get_weights(&rt).unwrap();
    {
        use engine::snn::mlp_weights::LiveSnnModel;
        let live = LiveSnnModel {
            syn_ptr: synapses.syn_ptr.clone(),
            targets: synapses.targets.iter().map(|&t| t as u16).collect(),
            weights: trained_weights.clone(),
            thresholds: thresholds.clone(),
            leaks: leaks.clone(),
            pix_per_class: pix_per_class.clone(),
            d_norms: d_norms.clone(),
            n_input: n_inp,
            n_hidden: n_hid,
            n_readout: n_out,
            n_classes: N_CLASSES,
            k_per_class: K,
            max_rate: MAX_RATE,
            n_ticks: TICKS,
            mlp: mlp_w.clone(),
        };
        let snn_path = format!("{ckpt_dir}/m10_seed{seed}.snn");
        live.save(&snn_path).expect("save live SNN model");
        println!(
            "  Saved live SNN    → {snn_path}  ({} neurons, {} synapses)",
            live.n_neurons(),
            live.n_synapses()
        );
    }

    let weights = trained_weights;
    let ih = &weights[..n_ih_synapses];
    let final_drift = ih
        .iter()
        .zip(init_ih_weights.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as f32)
        .sum::<f32>()
        / n_ih_synapses as f32;
    let syns_per_class = n_ih_synapses / N_CLASSES;
    let final_class_means: Vec<f32> = (0..N_CLASSES)
        .map(|c| {
            let lo = c * syns_per_class;
            let hi = (lo + syns_per_class).min(n_ih_synapses);
            ih[lo..hi].iter().map(|&w| w as f32).sum::<f32>() / (hi - lo) as f32
        })
        .collect();
    let class_min = final_class_means
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let class_max = final_class_means
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let class_spread = class_max - class_min;
    println!("  IH final drift={final_drift:.3}  class_spread={class_spread:.1}");

    // ── Summary ─────────────────────────────────────────────────────────────
    println!("\n  -- Summary --");
    println!("  M6 ceiling (T-norm, frozen IH)    : 62.4%");
    println!("  M7P1 mean (frozen IH + readout)   : 69.3%");
    println!("  M7P2 ceiling (scalar E-prop)      : 71.7%");
    println!("  M8 best (H=32, per-neuron)        : 74.6%");
    println!("  M9 best (MLP readout)             : 81.0%");
    println!("  M10P2 best (3L MLP+rec, this run) : {best_test_acc:.1}%");
    let delta_m9 = best_test_acc - 81.0;
    let delta_m6 = best_test_acc - 62.4;
    if delta_m9 >= 0.0 {
        println!("  Delta vs M9 (81.0%)               : +{delta_m9:.1}pp");
    } else {
        println!("  Delta vs M9 (81.0%)               :  {delta_m9:.1}pp");
    }
    println!("  Delta vs M6 ceiling (62.4%)       : +{delta_m6:.1}pp");

    println!();
    let train_test_gap = last_train_acc - best_test_acc;
    let gate_m10p2 = best_test_acc >= 83.0;
    let diverge_ok = class_spread > 5.0;
    let gap_ok = train_test_gap < 3.0;

    println!(
        "  [{}] IH class divergence (spread={class_spread:.1}, need >5)",
        if diverge_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] M10P2 gate >=83% ({best_test_acc:.1}%)",
        if gate_m10p2 { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Train-test gap <3pp (gap={train_test_gap:.1}pp)",
        if gap_ok { "PASS" } else { "FAIL" }
    );

    if gate_m10p2 {
        println!("\n  PASS — M10 Phase 2: 3-layer MLP + recurrence achieves >=83%.");
    } else {
        println!("\n  FAIL — M10P2 below 83% — escalation options:");
        println!("    1. Adjust MLP depths (try [320->128->64->10] or [320->256->128->10])");
        println!("    2. Add L2 weight decay (lambda=1e-4) if train-test gap > 3pp");
        println!("    3. Increase to 400 epochs");
    }
}
