//! Milestone 6 Phase 3: 3-Layer Discriminative E-prop Training
//!
//! Adds a true readout layer on top of the 2-layer discriminative network:
//!
//! ```text
//! Input [k×C]  →  Hidden [h×C]  →  Readout [r×C]
//!  (Fisher prior)  (discriminative)   (E-prop trains from scratch)
//! ```
//!
//! E-prop trains the hidden→readout synapses from uniform initialization,
//! while the Fisher-calibrated input→hidden weights are frozen (LR_IH=0).
//!
//! ## Acceptance gates
//!
//! | Check | Threshold |
//! |-------|-----------|
//! | 3-layer frozen ≥ 60% | ≥ 60% |
//! | Eligibility mean_abs > 0.001 at epoch 5 | yes |
//! | 3-layer E-prop > 3-layer frozen | > 0pp delta |
//! | Stretch: ≥ +2pp vs Gate A best (62.5%) | ≥ 64.5% |
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_3layer
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

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const N_CLASSES: usize = 10;
    const N_PIXELS: usize = 784;
    const K: usize = 150; // discriminative pixels per class
    const H: usize = 32; // hidden neurons per class
    const R: usize = 8; // readout neurons per class
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB_IH: f32 = 0.5; // input→hidden connection probability
    const CONN_PROB_HR: f32 = 0.8; // hidden→readout connection probability
    const LEAK: u8 = 230;
    const W_MAX_IH: i8 = 120; // Fisher-scaled input→hidden max weight
    const W_MAX_HR: i8 = 32; // uniform hidden→readout initial weight
    // T_READOUT: readout threshold. T-norm argmax multiplies raw counts by T_hidden[c],
    // so T_READOUT only matters for absolute firing rate, not relative class ranking.
    const T_READOUT: i16 = 400;
    const ALPHA: f32 = 0.50; // threshold alpha for hidden calibration

    // ── E-prop parameters ─────────────────────────────────────────────────────
    const N_EPOCHS: usize = 20;
    // Layered learning rates:
    // IH layer: FROZEN (0.0) — Fisher-calibrated weights must not drift.
    // HR layer: 0.003 — E-prop trains hidden→readout from uniform initialization.
    const LR_IH: f32 = 0.0;
    const LR_HR: f32 = 0.003;
    const EPROP_ALPHA_PRE: f32 = 0.95;
    const EPROP_ALPHA_ELIG: f32 = 0.95;
    const EPROP_GAMMA: f32 = 0.3;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 6 Phase 3: 3-Layer Discriminative E-prop");
    println!("  K={K} · H={H} · R={R} · T_readout={T_READOUT}");
    println!("  W_MAX_IH={W_MAX_IH} · W_MAX_HR={W_MAX_HR} (uniform init)");
    println!("  E-prop(α_pre={EPROP_ALPHA_PRE}, α_elig={EPROP_ALPHA_ELIG}, γ={EPROP_GAMMA})");
    println!(
        "  Epochs={N_EPOCHS} · IH frozen · LR_HR={LR_HR} · Batch={BATCH_SIZE} · Ticks={TICKS}"
    );
    println!("  Eval: T-norm argmax (score[c] = sum_counts[c] × T_hidden[c])");
    println!("  Gate: 3-layer E-prop > 3-layer frozen; stretch ≥ +2pp vs Gate A 62.5%");
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

    // ── Pixel scoring: mean-diff ordering ────────────────────────────────────
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

    // ── GPU V_ss helpers for hidden threshold calibration ─────────────────────
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

    // ── Build 3-layer network ─────────────────────────────────────────────────
    let n_inp = K * N_CLASSES;
    let n_hid = H * N_CLASSES;
    let n_out = R * N_CLASSES;
    let n_neurons = n_inp + n_hid + n_out;
    let n_inputs = n_inp; // GpuBatchSNN: first n_inputs neurons get rate inputs
    let n_outputs = n_out; // GpuBatchSNN: last n_outputs neurons are the readout
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
        42,
    );

    // Count IH vs HR synapses for layer-split weight stats
    let n_ih_synapses = synapses.syn_ptr[n_inp] as usize;
    let n_total_synapses = synapses.targets.len();
    let n_hr_synapses = n_total_synapses - n_ih_synapses;

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

    // No ALIF: with ALIF enabled, theta_eff grows far above T_READOUT_base, making
    // the surrogate gradient psi = 1/(1 + gamma*|v/T_base - 1|)^2 ≈ 0 for readout
    // neurons. Without ALIF, psi uses T_READOUT=400 directly, which is near the
    // actual operating point, giving psi ≈ 1 and nonzero HR eligibility traces.
    snn.enable_eprop(&rt, EPROP_ALPHA_PRE, EPROP_ALPHA_ELIG, EPROP_GAMMA)
        .expect("enable_eprop");
    snn.init_weight_shadow(&rt).expect("init_weight_shadow");

    println!(
        "  Network: {n_inp} inputs · {n_hid} hidden · {n_out} readout · {} total synapses",
        n_total_synapses
    );
    println!(
        "  IH synapses: {} · HR synapses: {}",
        n_ih_synapses, n_hr_synapses
    );
    println!("  E-prop + f32-shadow buffers allocated (no ALIF).");

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

    // ── T-normalised argmax: score[c] = sum(counts[c*R..(c+1)*R]) × T_hidden[c]
    // T_hidden[c] compensates for the per-class threshold difference, so low-T
    // classes (e.g. class 5: T=3378) don't dominate raw counts.
    let predict_batch = |counts: &Vec<Vec<u32>>, actual_bs: usize| -> Vec<usize> {
        (0..actual_bs)
            .map(|b| {
                let row = &counts[b];
                (0..N_CLASSES)
                    .map(|c| {
                        let lo = c * R;
                        let hi = lo + R;
                        let raw: u64 = row[lo..hi].iter().map(|&x| x as u64).sum();
                        raw * (t_hidden[c] as u64)
                    })
                    .enumerate()
                    .max_by_key(|&(_, s)| s)
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect()
    };

    // ── Inference loop ────────────────────────────────────────────────────────
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

    // ── 3-layer frozen baseline ────────────────────────────────────────────────
    println!("\n  3-layer frozen baseline (T-norm argmax, no ALIF)...");
    let t0 = Instant::now();
    let baseline_acc = run_test(&mut snn, &test_flat);
    println!(
        "  3-layer frozen accuracy (T-norm): {baseline_acc:.1}%  ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    if baseline_acc < 60.0 {
        println!(
            "  *** WARNING: 3-layer frozen < 60% — threshold calibration may need adjustment."
        );
    }
    println!("  (M5 Gate A reference: 62.5% frozen+ALIF)\n");

    // ── Training ──────────────────────────────────────────────────────────────
    let mut rng_state: u64 = 0xDEAD_BEEF_C0FFEE42;
    let lcg = |s: &mut u64| -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    };

    let n_train = train_flat.len();

    println!("  Starting 3-layer E-prop training ({N_EPOCHS} epochs, IH frozen, LR_HR={LR_HR})...");
    println!("  Training signal: all-samples T-norm (+1 true class, -1 wrong pred).");
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>14}  {:>8}",
        "epoch", "train_corr", "train_acc", "test_acc", "secs"
    );
    println!("  {}", "-".repeat(58));

    let mut best_test_acc = baseline_acc;
    let two_layer_gate_a_best = 62.5_f32; // Gate A Run 5 best (= frozen baseline)

    for epoch in 0..N_EPOCHS {
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

            let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
            let mut true_labels = vec![0usize; actual_bs];
            for (b, &idx) in chunk_idx.iter().enumerate() {
                let (ref img, lbl) = train_flat[idx];
                let enc = encode_image(img);
                rates_buf[b * n_inputs..(b + 1) * n_inputs].copy_from_slice(&enc);
                true_labels[b] = lbl;
            }

            snn.upload_input_batch(&rt, &rates_buf).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.reset_output_counts(&rt).unwrap();
            snn.reset_eprop_state(&rt).unwrap();
            snn.tick_many_with_eprop(&rt, TICKS).unwrap();
            rt.synchronize().unwrap();

            let counts = snn.get_output_counts(&rt).unwrap();
            let preds = predict_batch(&counts, actual_bs);
            for b in 0..actual_bs {
                if preds[b] == true_labels[b] {
                    train_correct += 1;
                }
            }

            // All-samples class-specific learning signals (T-norm predictions):
            //   true class c → +1 (strengthen class-c readout)
            //   wrong pred c → -1 (weaken over-active class-c readout)
            // Per-class HR weight differentiation:
            //   Net_c = +1 × FN_c − 1 × FP_c  (FN=false negatives, FP=false positives)
            //   Classes with FP > FN → HR[c] ↓; classes with FN > FP → HR[c] ↑.
            let lr_ih = LR_IH / N_CLASSES as f32; // = 0.0 (frozen)
            let lr_hr = LR_HR / N_CLASSES as f32;
            for c in 0..N_CLASSES {
                let mut class_signals = vec![0.0f32; BATCH_SIZE];
                for b in 0..actual_bs {
                    if true_labels[b] == c {
                        class_signals[b] = 1.0;
                    } else if preds[b] == c {
                        class_signals[b] = -1.0;
                    }
                }
                snn.apply_eprop_update_layered(&rt, &class_signals, lr_ih, lr_hr, n_ih_synapses)
                    .unwrap();
            }

            i += actual_bs;
        }

        // Elig stats + per-class HR mean every 5 epochs
        if epoch % 5 == 4 {
            if let Ok((mean_abs, max_abs)) = snn.get_eligibility_stats(&rt) {
                println!("    [elig] mean_abs={mean_abs:.4e}  max_abs={max_abs:.4e}");
                if mean_abs < 0.001 {
                    println!(
                        "    *** WARNING: elig mean_abs very low — possible kernel/init issue"
                    );
                }
            }
            // Per-class HR mean (diagnostic: should diverge if training differentiates HR)
            let w = snn.get_weights(&rt).unwrap();
            let hr = &w[n_ih_synapses..];
            let hr_f32: Vec<f32> = hr.iter().map(|&x| x as f32).collect();
            let syns_per_class = n_hr_synapses / N_CLASSES;
            let hr_means: Vec<String> = (0..N_CLASSES)
                .map(|c| {
                    let lo = c * syns_per_class;
                    let hi = (lo + syns_per_class).min(hr_f32.len());
                    let m = if hi > lo {
                        hr_f32[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
                    } else {
                        0.0
                    };
                    format!("{m:.1}")
                })
                .collect();
            println!("    [HR mean/class] {:?}", hr_means);
        }

        // Sync f32 shadow → i8 before inference
        snn.project_weights_f32(&rt).unwrap();

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
    }

    // ── Final layer-split weight stats ────────────────────────────────────────
    snn.project_weights_f32(&rt).unwrap();
    let weights = snn.get_weights(&rt).unwrap();

    let weight_stats = |w: &[i8], label: &str| {
        if w.is_empty() {
            return;
        }
        let w_min = w.iter().copied().min().unwrap_or(0);
        let w_max = w.iter().copied().max().unwrap_or(0);
        let w_mean = w.iter().map(|&x| x as f32).sum::<f32>() / w.len() as f32;
        let w_zero = w.iter().filter(|&&x| x == 0).count() as f32 / w.len() as f32;
        let w_sat = w.iter().filter(|&&x| x == 127 || x == -128).count() as f32 / w.len() as f32;
        println!(
            "  {label}: min={w_min}  max={w_max}  mean={w_mean:.2}  \
                  zero={:.1}%  sat={:.1}%",
            w_zero * 100.0,
            w_sat * 100.0
        );
    };

    println!("\n  ── Final weight stats ──");
    weight_stats(&weights[..n_ih_synapses], "  Input→Hidden ");
    weight_stats(&weights[n_ih_synapses..], "  Hidden→Readout");

    // ── Summary table ─────────────────────────────────────────────────────────
    println!("\n  ── Summary ──");
    println!("  ┌─────────────────────────────────┬────────┐");
    println!("  │ 2-layer frozen (Gate A ref)      │ {:.1}% │", 62.5);
    println!(
        "  │ 3-layer frozen (baseline)        │ {:.1}% │",
        baseline_acc
    );
    println!(
        "  │ 3-layer E-prop best              │ {:.1}% │",
        best_test_acc
    );
    let delta_vs_frozen = best_test_acc - baseline_acc;
    let delta_vs_gate_a = best_test_acc - two_layer_gate_a_best;
    if delta_vs_frozen > 0.0 {
        println!(
            "  │ Δ vs 3-layer frozen              │ +{:.1}pp │",
            delta_vs_frozen
        );
    } else {
        println!(
            "  │ Δ vs 3-layer frozen              │  {:.1}pp │",
            delta_vs_frozen
        );
    }
    if delta_vs_gate_a > 0.0 {
        println!(
            "  │ Δ vs 2-layer Gate A (62.5%)      │ +{:.1}pp │",
            delta_vs_gate_a
        );
    } else {
        println!(
            "  │ Δ vs 2-layer Gate A (62.5%)      │  {:.1}pp │",
            delta_vs_gate_a
        );
    }
    println!("  └─────────────────────────────────┴────────┘");

    // ── Gate checks ───────────────────────────────────────────────────────────
    println!();
    let gate_frozen = baseline_acc >= 60.0;
    let gate_delta = best_test_acc > baseline_acc;
    let gate_stretch = best_test_acc >= 64.5;

    println!(
        "  [{}] 3-layer frozen ≥ 60% ({:.1}%)",
        if gate_frozen { "✓" } else { "✗" },
        baseline_acc
    );
    println!(
        "  [{}] E-prop > frozen baseline ({:.1}% > {:.1}%)",
        if gate_delta { "✓" } else { "✗" },
        best_test_acc,
        baseline_acc
    );
    println!(
        "  [{}] Stretch: ≥ 64.5% ({:.1}%)",
        if gate_stretch { "✓" } else { "✗" },
        best_test_acc
    );
}
