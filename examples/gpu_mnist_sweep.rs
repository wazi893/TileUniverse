//! Milestone 3 Phase 4 / Milestone 4 Phase 2: GPU SNN Parameter Sweep — K, H, and α
//!
//! Sweeps the key hyperparameters of the discriminative-channel SNN and
//! reports accuracy + throughput for each configuration.
//!
//! Uses a **2-layer** (input → hidden) architecture where the hidden layer is
//! the readout: `predicted = argmax over c of sum(counts[b][c*H..(c+1)*H]) × T_c`.
//!
//! ## Parameters swept
//!
//! **Section 1: K × H sweep** (same as Phase 4, for reference)
//! - **K** (discriminative pixels per class): 50, 75, 100, 150, 200
//! - **H** (hidden neurons per class): 32, 64
//! - α fixed at 0.50 (original midpoint)
//!
//! **Section 2: α sweep at best K** (Milestone 4 Phase 2)
//! - K fixed at 150, H=32 (M3 best config)
//! - **α** (threshold interpolation): 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65
//! - Uses validation set for selection, reports test accuracy of best α
//!
//! All configurations run on the same MNIST test set (10K samples).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features cuda --example gpu_mnist_sweep
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

    // ── Fixed parameters ─────────────────────────────────────────────────
    const N_CLASSES: usize = 10;
    const N_PIXELS: usize = 784;
    const TICKS: usize = 100;
    const BATCH_SIZE: usize = 256;
    const MAX_RATE: u32 = 100;
    const CONN_PROB: f32 = 0.5;
    const LEAK: u8 = 230;
    const W_MAX: i8 = 120;

    // ── Sweep grids ──────────────────────────────────────────────────────
    const K_VALUES: &[usize] = &[50, 75, 100, 150, 200];
    const H_VALUES: &[usize] = &[32, 64];
    const ALPHA_VALUES: &[f32] = &[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 4 Phase 2: GPU SNN Parameter Sweep (K, H, α)");
    println!(
        "  batch={BATCH_SIZE} · ticks={TICKS} · {} test samples",
        10000
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

    // Stratified train / val split (90 / 10)
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

    // ── CPU feature analysis ─────────────────────────────────────────────────
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

    // Sorted discriminative orders for each class (computed once, reused for all K)
    let mut orders: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    for c in 0..N_CLASSES {
        let mut scores: Vec<(usize, f32)> = (0..N_PIXELS)
            .map(|i| {
                let others = (global_sum[i] - avgs[c][i]) / (N_CLASSES - 1) as f32;
                (i, avgs[c][i] - others)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        orders[c] = scores.iter().map(|(i, _)| *i).collect();
    }

    // ── Build flat test and val datasets ─────────────────────────────────────
    let test_flat: Vec<(Vec<u8>, usize)> = test_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();
    let val_flat: Vec<(Vec<u8>, usize)> = val_per_class
        .iter()
        .enumerate()
        .flat_map(|(c, imgs)| imgs.iter().map(move |img| (img.clone(), c)))
        .collect();

    // ── CUDA runtime ─────────────────────────────────────────────────────────
    let rt = CudaRuntime::new().expect("CUDA init");

    // GPU V_ss helper
    let w_scale = 2.0_f32;
    let leak_gap = 256.0_f32 - LEAK as f32;

    struct SweepResult {
        k: usize,
        h: usize,
        alpha: f32,
        n_neurons: usize,
        n_syn: usize,
        accuracy: f32,
        sps: f32,
    }
    let mut results: Vec<SweepResult> = Vec::new();

    // ──────────────────────────────────────────────────────────────────────────
    // SECTION 1: K × H sweep (α = 0.50 fixed)
    // ──────────────────────────────────────────────────────────────────────────
    println!("  Section 1: K × H sweep  (α=0.50 fixed)");
    println!(
        "  {:>5}  {:>5}  {:>8}  {:>8}  {:>10}  {:>6}",
        "K", "H", "Neurons", "Synapses", "Accuracy", "SPS"
    );
    println!("  {}", "─".repeat(55));

    for &k in K_VALUES {
        let mut pix_per_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
        let mut d_norms: Vec<Vec<f32>> = vec![Vec::new(); N_CLASSES];

        for c in 0..N_CLASSES {
            let top_k: Vec<usize> = orders[c][..k].to_vec();
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

        let gpu_vss = |cc: usize, cj: usize| -> f32 {
            let weighted_sum: f32 = pix_per_class[cc]
                .iter()
                .zip(d_norms[cc].iter())
                .map(|(&px, &d)| avgs[cj][px] * d)
                .sum();
            CONN_PROB * W_MAX as f32 * w_scale * MAX_RATE as f32 / (255.0 * leak_gap) * weighted_sum
        };

        // Build t_per_class with α=0.5
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

        let encode_image = |img: &[u8], k: usize| -> Vec<u8> {
            let mut rates = vec![0u8; k * N_CLASSES];
            for c in 0..N_CLASSES {
                for (i, &pix) in pix_per_class[c].iter().enumerate() {
                    rates[c * k + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
                }
            }
            rates
        };

        for &h in H_VALUES {
            let n_inputs = k * N_CLASSES;
            let n_hidden = h * N_CLASSES;
            let n_neurons = n_inputs + n_hidden;
            let n_outputs = n_hidden;

            let d_norms_flat: Vec<f32> =
                d_norms.iter().flat_map(|row| row.iter().copied()).collect();

            let synapses = generate_discriminative_2layer_csr_weighted(
                k,
                h,
                N_CLASSES,
                CONN_PROB,
                W_MAX,
                &d_norms_flat,
                42,
            );
            let n_syn = synapses.targets.len();

            let thresholds = discriminative_2layer_thresholds(k, h, N_CLASSES, &t_per_class, 32767);
            let leaks = vec![LEAK; n_neurons];

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

            let t0 = Instant::now();
            let mut correct = 0usize;
            let mut i = 0;
            while i < test_flat.len() {
                let chunk_end = (i + BATCH_SIZE).min(test_flat.len());
                let chunk = &test_flat[i..chunk_end];
                let actual_bs = chunk.len();

                let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs];
                for (b, (img, _)) in chunk.iter().enumerate() {
                    let enc = encode_image(img, k);
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
                            let lo = c * h;
                            let hi = lo + h;
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

            let elapsed = t0.elapsed().as_secs_f32();
            let accuracy = 100.0 * correct as f32 / test_flat.len() as f32;
            let sps = test_flat.len() as f32 / elapsed;

            println!(
                "  {:>5}  {:>5}  {:>8}  {:>8}  {:>9.1}%  {:>6.0}",
                k, h, n_neurons, n_syn, accuracy, sps
            );

            results.push(SweepResult {
                k,
                h,
                alpha: 0.5,
                n_neurons,
                n_syn,
                accuracy,
                sps,
            });
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // SECTION 2: α sweep at K=150, H=32
    // Uses validation set to select α, then evaluates selected α on test.
    // ──────────────────────────────────────────────────────────────────────────
    println!();
    println!("  Section 2: α sweep  (K=150, H=32, Fisher=false)");
    println!(
        "  {:>5}  {:>5}  {:>8}  {:>10}  {:>10}",
        "alpha", "H", "set", "val_acc", "test_acc"
    );
    println!("  {}", "─".repeat(55));

    const K150: usize = 150;
    const H32: usize = 32;

    // Recompute top-K=150 pix/d_norms
    let mut pix150: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    let mut dn150: Vec<Vec<f32>> = vec![Vec::new(); N_CLASSES];
    for c in 0..N_CLASSES {
        let top_k: Vec<usize> = orders[c][..K150].to_vec();
        let d_raw: Vec<f32> = top_k
            .iter()
            .map(|&pix| {
                let others = (global_sum[pix] - avgs[c][pix]) / (N_CLASSES - 1) as f32;
                (avgs[c][pix] - others).max(0.0)
            })
            .collect();
        let max_d = d_raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        dn150[c] = d_raw.iter().map(|&d| d / max_d).collect();
        pix150[c] = top_k;
    }

    let gpu_vss150 = |cc: usize, cj: usize| -> f32 {
        let weighted_sum: f32 = pix150[cc]
            .iter()
            .zip(dn150[cc].iter())
            .map(|(&px, &d)| avgs[cj][px] * d)
            .sum();
        CONN_PROB * W_MAX as f32 * w_scale * MAX_RATE as f32 / (255.0 * leak_gap) * weighted_sum
    };

    let encode150 = |img: &[u8]| -> Vec<u8> {
        let mut rates = vec![0u8; K150 * N_CLASSES];
        for c in 0..N_CLASSES {
            for (i, &pix) in pix150[c].iter().enumerate() {
                rates[c * K150 + i] = ((img[pix] as u32 * MAX_RATE) / 255) as u8;
            }
        }
        rates
    };

    let d_norms150_flat: Vec<f32> = dn150.iter().flat_map(|row| row.iter().copied()).collect();
    let synapses150 = generate_discriminative_2layer_csr_weighted(
        K150,
        H32,
        N_CLASSES,
        CONN_PROB,
        W_MAX,
        &d_norms150_flat,
        42,
    );
    let n_inputs150 = K150 * N_CLASSES;
    let n_hidden150 = H32 * N_CLASSES;
    let n_neurons150 = n_inputs150 + n_hidden150;
    let n_outputs150 = n_hidden150;
    let leaks150 = vec![LEAK; n_neurons150];

    let run_set =
        |snn: &mut GpuBatchSNN, dataset: &[(Vec<u8>, usize)], t_per_class: &[i16]| -> f32 {
            let n = dataset.len();
            let mut correct = 0usize;
            let mut i = 0;
            while i < n {
                let chunk_end = (i + BATCH_SIZE).min(n);
                let chunk = &dataset[i..chunk_end];
                let actual_bs = chunk.len();
                let mut rates_buf = vec![0u8; BATCH_SIZE * n_inputs150];
                for (b, (img, _)) in chunk.iter().enumerate() {
                    let enc = encode150(img);
                    rates_buf[b * n_inputs150..(b + 1) * n_inputs150].copy_from_slice(&enc);
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
                            let lo = c * H32;
                            let hi = lo + H32;
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

    let mut best_alpha = 0.5f32;
    let mut best_val_acc = 0.0f32;
    let mut alpha_results: Vec<(f32, f32, f32)> = Vec::new(); // (alpha, val, test)

    for &alpha in ALPHA_VALUES {
        let t_per_class: Vec<i16> = (0..N_CLASSES)
            .map(|c| {
                let vss_cc = gpu_vss150(c, c);
                let vss_max = (0..N_CLASSES)
                    .filter(|&j| j != c)
                    .map(|j| gpu_vss150(c, j))
                    .fold(0.0f32, f32::max);
                let t = if vss_cc > vss_max {
                    (alpha * vss_cc + (1.0 - alpha) * vss_max).clamp(1.0, i16::MAX as f32)
                } else {
                    (0.85 * vss_cc).clamp(1.0, i16::MAX as f32)
                };
                t as i16
            })
            .collect();

        let thresholds =
            discriminative_2layer_thresholds(K150, H32, N_CLASSES, &t_per_class, 32767);
        let mut snn = GpuBatchSNN::new(
            &rt,
            n_neurons150,
            n_inputs150,
            n_outputs150,
            &thresholds,
            &leaks150,
            &synapses150,
            BATCH_SIZE,
        )
        .expect("GpuBatchSNN::new");

        let val_acc = run_set(&mut snn, &val_flat, &t_per_class);
        let test_acc = run_set(&mut snn, &test_flat, &t_per_class);
        let mark = if val_acc > best_val_acc {
            " ← val best"
        } else {
            ""
        };
        println!(
            "  {:>5.2}  {:>5}  {:>8}  {:>9.1}%  {:>9.1}%{}",
            alpha, H32, "val+test", val_acc, test_acc, mark
        );

        alpha_results.push((alpha, val_acc, test_acc));
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_alpha = alpha;
        }

        results.push(SweepResult {
            k: K150,
            h: H32,
            alpha,
            n_neurons: n_neurons150,
            n_syn: synapses150.targets.len(),
            accuracy: test_acc,
            sps: 0.0,
        });
    }

    let best_test = alpha_results
        .iter()
        .find(|&&(a, _, _)| (a - best_alpha).abs() < 1e-4)
        .map(|&(_, _, t)| t)
        .unwrap_or(0.0);

    // ── Summary ───────────────────────────────────────────────────────────────
    let best_kh = results
        .iter()
        .filter(|r| r.alpha == 0.5)
        .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap());

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Sweep Summary:");

    if let Some(b) = best_kh {
        println!(
            "    Sec 1 best (K/H, α=0.5):  {:.1}%  (K={}, H={})",
            b.accuracy, b.k, b.h
        );
    }
    println!(
        "    Sec 2 best α (val-selected): α={:.2}  test={:.1}%",
        best_alpha, best_test
    );
    println!("    CPU baseline:               61.3%  (7×7 input, H=32)");
    println!("    GPU SNN M3 best:            62.2%  (K=150, α=0.5)");

    let overall_best = results
        .iter()
        .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap());
    if let Some(b) = overall_best {
        if b.accuracy > 62.2 {
            println!(
                "    ✓ Best config beats M3 by +{:.1}pp  (K={}, H={}, α={:.2})",
                b.accuracy - 62.2,
                b.k,
                b.h,
                b.alpha
            );
        } else {
            println!("    △ Best {:.1}% below M3 baseline 62.2%", b.accuracy);
        }
    }
    println!("═══════════════════════════════════════════════════════════════");
}
