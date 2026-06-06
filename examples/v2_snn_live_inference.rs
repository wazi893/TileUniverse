//! Milestone 12: End-to-End V2 Live SNN Inference Demo
//!
//! Loads MNIST test set + M10 `.snn` checkpoint (full SNN model + MLP weights),
//! builds a V2 CPU with combined MMIO, assembles an inference loop using live
//! SNN simulation commands (LOAD_IMAGE → SNN_RUN → INFER_LIVE), and runs to
//! completion.
//!
//! Unlike M11 (which used pre-computed cached rates), M12 runs the actual
//! LIF SNN simulation for each sample through MMIO commands.
//!
//! ## Usage
//!
//! ```bash
//! # Run on first 100 test samples (quick demo)
//! cargo run --release --example v2_snn_live_inference
//!
//! # Run with specific seed
//! cargo run --release --example v2_snn_live_inference -- --seed 42
//!
//! # Run on more samples
//! cargo run --release --example v2_snn_live_inference -- --count 500
//! ```

fn main() {
    use engine::simulation::Simulation;
    use engine::snn::mlp_weights::LiveSnnModel;
    use engine::snn::mnist::MnistDataset;
    use engine::tile_cpu::{
        DatasetSample, V2Builder, V2MmioCombinedDevice, V2MmioDatasetDevice, V2MmioHandle,
        V2MmioSnnBridgeDevice, assemble_v2,
    };
    use std::rc::Rc;
    use std::time::Instant;

    // ── Parse CLI args ───────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 42;
    let mut count: usize = 100;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid seed");
            }
            "--count" => {
                i += 1;
                count = args[i].parse().expect("invalid count");
            }
            _ => {
                if let Ok(s) = args[i].parse::<u64>() {
                    seed = s;
                }
            }
        }
        i += 1;
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 12: V2 + Live SNN MMIO Inference Demo");
    println!("  seed={seed}  count={count}");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Load MNIST test set ────────────────────────────────────────────────
    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("load MNIST test data — ensure data/ directory contains MNIST files");
    let test_labels: Vec<usize> = test_ds.labels.iter().map(|&l| l as usize).collect();
    let test_images: Vec<Vec<u8>> = test_ds.images.clone();
    println!("  Loaded {} test images + labels", test_labels.len());

    // ── Load M10 live SNN checkpoint ───────────────────────────────────────
    let snn_path = format!("checkpoints/m10_seed{seed}.snn");
    let model = LiveSnnModel::load(&snn_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: Cannot load live SNN model from '{snn_path}': {e}");
        eprintln!("  Run M10 training first:");
        eprintln!("    cargo run --release --features cuda --example gpu_mnist_lsm_v2 -- {seed}");
        std::process::exit(1);
    });
    println!(
        "  Loaded live SNN model: {} neurons, {} synapses",
        model.n_neurons(),
        model.n_synapses()
    );
    let (n_hid, mlp_h1, mlp_h2, n_classes) = model.mlp.dims();
    println!(
        "  MLP readout: [{n_hid}→{mlp_h1}→{mlp_h2}→{n_classes}]  ticks={}",
        model.n_ticks
    );

    // ── Prepare demo subset ────────────────────────────────────────────────
    let demo_count = count.min(test_labels.len());
    let demo_images: Vec<Vec<u8>> = test_images[..demo_count].to_vec();

    // ── Direct CPU live inference (reference path) ─────────────────────────
    println!("\n  Running direct CPU live inference on {demo_count} samples...");
    let direct_t0 = Instant::now();
    let mut direct_correct = 0usize;
    let mut total_spike_rate_sum = 0.0f64;
    for idx in 0..demo_count {
        let rates = model.encode_image(&demo_images[idx]);
        let n_total = model.n_neurons();
        let mut v_mem = vec![0i16; n_total];
        let mut refract = vec![0u8; n_total];
        let mut spiked = vec![0u8; n_total];
        let mut hidden_counts = vec![0u32; model.n_hidden];
        let rng_seed = seed as u32;

        for tick in 0..model.n_ticks {
            let tick_seed = rng_seed.wrapping_add(tick as u32 * 1000003);
            for i in 0..model.n_input {
                if refract[i] > 0 {
                    refract[i] -= 1;
                    spiked[i] = 0;
                    continue;
                }
                let mut rng = tick_seed ^ (i as u32).wrapping_mul(2654435761);
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let rand_val = (rng >> 24) as u8;
                if rand_val < rates[i] {
                    spiked[i] = 1;
                    v_mem[i] = -128;
                    refract[i] = 2;
                } else {
                    spiked[i] = 0;
                }
            }

            let mut currents = vec![0i32; n_total];
            for src in 0..n_total {
                if spiked[src] != 0 {
                    let start = model.syn_ptr[src] as usize;
                    let end = model.syn_ptr[src + 1] as usize;
                    for j in start..end {
                        let tgt = model.targets[j] as usize;
                        if tgt < n_total {
                            currents[tgt] += model.weights[j] as i32 * 2;
                        }
                    }
                }
            }

            for i in model.n_input..n_total {
                if refract[i] > 0 {
                    refract[i] -= 1;
                    spiked[i] = 0;
                    continue;
                }
                let v = ((v_mem[i] as i32 * model.leaks[i] as i32) >> 8) + currents[i];
                let v = v.clamp(-32768, 32767);
                if v >= model.thresholds[i] as i32 {
                    spiked[i] = 1;
                    v_mem[i] = -128;
                    refract[i] = 2;
                } else {
                    spiked[i] = 0;
                    v_mem[i] = v as i16;
                }
            }

            for h in 0..model.n_hidden {
                if spiked[model.n_input + h] != 0 {
                    hidden_counts[h] += 1;
                }
            }
            for i in 0..model.n_input {
                spiked[i] = 0;
            }
        }

        let firing_rates: Vec<f32> = hidden_counts
            .iter()
            .map(|&c| c as f32 / model.n_ticks as f32)
            .collect();
        let mean_rate: f64 =
            firing_rates.iter().map(|&r| r as f64).sum::<f64>() / firing_rates.len() as f64;
        total_spike_rate_sum += mean_rate;
        let pred = model.mlp.forward_cpu(&firing_rates);
        if pred == test_labels[idx] {
            direct_correct += 1;
        }
    }
    let direct_elapsed = direct_t0.elapsed();
    let direct_acc = 100.0 * direct_correct as f32 / demo_count as f32;
    let mean_spike_rate = total_spike_rate_sum / demo_count as f64;
    println!(
        "  Direct CPU: {direct_correct}/{demo_count} = {direct_acc:.1}%  ({:.1}s, spike_rate={mean_spike_rate:.4})",
        direct_elapsed.as_secs_f32()
    );

    // ── Create dataset samples ─────────────────────────────────────────────
    let samples: Vec<DatasetSample> = (0..demo_count)
        .map(|i| DatasetSample {
            features: 0,
            label: test_labels[i] as u64,
        })
        .collect();

    // ── Build MMIO devices ─────────────────────────────────────────────────
    let snn = V2MmioSnnBridgeDevice::with_live_model(
        model.n_input,
        model.n_hidden,
        model.n_readout,
        seed,
        model,
        demo_images,
    );
    let dataset = V2MmioDatasetDevice::from_samples(samples);
    let combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(
        seed, snn, dataset,
    ));
    let mmio_handle = V2MmioHandle::from_rc(combined.clone());

    // ── V2 Assembly: live SNN inference loop ───────────────────────────────
    let program_source = r#"
; M12: V2 + Live SNN MMIO Inference Loop
; For each sample: LOAD_SAMPLE → LOAD_IMAGE → SNN_RUN → INFER_LIVE → SUBMIT
; Registers: R0=idx, R1=total, R2=scratch, R3=prediction, R4=correct
;
; Get sample count
    LDI R2, 3                  ; cmd GET_COUNT
    STB [DATASET_CMD], R2
    LDB R1, [DATASET_DATA]    ; R1 = total samples
    LDI R0, 0                  ; R0 = sample index
loop:
    CMP R0, R1
    JZ done
; LOAD_SAMPLE(R0)
    STB [DATASET_DATA], R0
    LDI R2, 0                  ; cmd LOAD_SAMPLE
    STB [DATASET_CMD], R2
; LOAD_IMAGE(R0) — encode pixels into Poisson rates, reset SNN
    STB [SNN_DATA], R0
    LDI R2, 8                  ; cmd LOAD_IMAGE
    STB [SNN_CMD], R2
; SNN_RUN — run n_ticks LIF simulation
    LDI R2, 9                  ; cmd SNN_RUN
    STB [SNN_CMD], R2
; INFER_LIVE — compute rates from spike counts, MLP forward
    LDI R2, 10                 ; cmd INFER_LIVE
    STB [SNN_CMD], R2
; Read prediction
    LDB R3, [SNN_DATA]
; SUBMIT_PREDICTION(R3)
    STB [DATASET_DATA], R3
    LDI R2, 5                  ; cmd SUBMIT_PREDICTION
    STB [DATASET_CMD], R2
; Next sample
    INC R0
    JMP loop
done:
; GET_CORRECT
    LDI R2, 4                  ; cmd GET_CORRECT
    STB [DATASET_CMD], R2
    LDB R4, [DATASET_DATA]    ; R4 = correct count
    HALT
"#;

    let words = assemble_v2(program_source).expect("assemble M12 inference program");
    println!("\n  Assembled {} instruction words", words.len());

    // ── Build V2 CPU ────────────────────────────────────────────────────────
    println!("  Building V2 CPU (128x128x4 grid)...");
    let build_t0 = Instant::now();
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&words)
        .with_rom_size(64)
        .with_ram_size(64)
        .with_mmio(mmio_handle)
        .build(&mut sim);
    println!("  Built in {:.1}s", build_t0.elapsed().as_secs_f32());

    // ── Execute ─────────────────────────────────────────────────────────────
    println!("\n  Running live SNN inference on {demo_count} samples...");
    let max_cycles: u64 = (demo_count as u64) * 200 + 1000;
    let run_t0 = Instant::now();
    let mut cycles: u64 = 0;
    let mut instructions: u64 = 0;
    let mut last_progress = 0;

    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
        cycles += 1;
        if cpu.last_stage_x_valid() {
            instructions += 1;
        }

        let r0 = cpu.read_reg(&sim, 0) as usize;
        let pct = if demo_count > 0 {
            r0 * 100 / demo_count
        } else {
            100
        };
        if pct >= last_progress + 10 && pct <= 100 {
            let elapsed = run_t0.elapsed().as_secs_f32();
            print!("  [{pct:>3}%] sample {r0}/{demo_count}  cycles={cycles}  {elapsed:.1}s\r");
            last_progress = pct;
        }
    }
    let elapsed = run_t0.elapsed();
    println!();

    // ── Results ─────────────────────────────────────────────────────────────
    let halted = cpu.is_halted();
    let v2_correct = cpu.read_reg(&sim, 4);
    let v2_acc = 100.0 * v2_correct as f32 / demo_count as f32;
    let per_sample_ms = if demo_count > 0 {
        elapsed.as_secs_f64() * 1000.0 / demo_count as f64
    } else {
        0.0
    };

    println!("\n  ── M12 Results ──────────────────────────────────────────────");
    println!("  Halted:           {halted}");
    println!("  Cycles:           {cycles}");
    println!("  Instructions:     {instructions}");
    println!("  Wall-clock:       {:.2}s", elapsed.as_secs_f32());
    println!("  Per-sample:       {per_sample_ms:.2}ms");
    println!();
    println!("  V2 live accuracy: {v2_correct}/{demo_count} = {v2_acc:.1}%");
    println!("  Direct CPU:       {direct_correct}/{demo_count} = {direct_acc:.1}%");
    let match_ok = v2_correct as usize == direct_correct;
    println!(
        "  Parity:           {} (V2={v2_correct}, direct={direct_correct})",
        if match_ok { "MATCH" } else { "MISMATCH" }
    );
    println!("  Mean spike rate:  {mean_spike_rate:.4}");

    // ── Gate ────────────────────────────────────────────────────────────────
    println!("\n  ── M12 Gate Checks ─────────────────────────────────────────");
    println!(
        "  [{}] V2 halted normally",
        if halted { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Parity: V2 == direct CPU",
        if match_ok { "PASS" } else { "FAIL" }
    );
    let acc_gate = direct_acc >= 75.0;
    println!(
        "  [{}] Live accuracy >= 75% ({direct_acc:.1}%)",
        if acc_gate { "PASS" } else { "FAIL" }
    );
    let rate_ok = mean_spike_rate >= 0.01 && mean_spike_rate <= 0.30;
    println!(
        "  [{}] Spike rate in [0.01, 0.30] ({mean_spike_rate:.4})",
        if rate_ok { "PASS" } else { "FAIL" }
    );

    let all_pass = halted && match_ok && acc_gate && rate_ok;
    if all_pass {
        println!("\n  PASS — M12: V2 + Live SNN MMIO inference end-to-end verified.");
    } else {
        println!("\n  FAIL — see above for details.");
        std::process::exit(1);
    }
}
