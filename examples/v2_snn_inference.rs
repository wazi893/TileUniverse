//! Milestone 11: End-to-End V2 MMIO Inference Demo
//!
//! Loads MNIST test set + M10 checkpoint (MLP weights + cached hidden rates),
//! builds a V2 CPU with combined MMIO (dataset + SNN with model), assembles
//! an inference loop, and runs it to completion.
//!
//! The V2 program uses ~20 instructions to:
//!   1. LOAD_SAMPLE(idx) via dataset MMIO
//!   2. SNN_CMD_INFER(idx) via SNN MMIO → runs MLP forward_cpu on cached rates
//!   3. SUBMIT_PREDICTION via dataset MMIO
//!
//! ## Usage
//!
//! ```bash
//! # Run on first 100 test samples (quick demo, ~30s)
//! cargo run --release --example v2_snn_inference
//!
//! # Run on first 100 samples with specific seed
//! cargo run --release --example v2_snn_inference -- --seed 42
//!
//! # Run on more samples (slower — V2 is tile-level simulation)
//! cargo run --release --example v2_snn_inference -- --count 500
//! ```

fn main() {
    use engine::simulation::Simulation;
    use engine::snn::mlp_weights::{CachedRates, MlpWeights};
    use engine::snn::mnist::MnistDataset;
    use engine::tile_cpu::{
        DatasetSample, InferenceModel, V2Builder, V2MmioCombinedDevice, V2MmioDatasetDevice,
        V2MmioHandle, V2MmioSnnBridgeDevice, assemble_v2,
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
                // Try parsing as bare seed for compatibility
                if let Ok(s) = args[i].parse::<u64>() {
                    seed = s;
                }
            }
        }
        i += 1;
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Milestone 11: V2 + SNN MMIO Inference Demo");
    println!("  seed={seed}  count={count}");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ── Load MNIST test labels ───────────────────────────────────────────────
    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("load MNIST test data — ensure data/ directory contains MNIST files");
    let test_labels: Vec<usize> = test_ds.labels.iter().map(|&l| l as usize).collect();
    println!("  Loaded {} test labels", test_labels.len());

    // ── Load M10 checkpoint ──────────────────────────────────────────────────
    let weights_path = format!("checkpoints/m10_seed{seed}.mlp");
    let rates_path = format!("checkpoints/m10_seed{seed}.rates");
    let weights = MlpWeights::load(&weights_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: Cannot load weights from '{weights_path}': {e}");
        eprintln!("  Run M10 training first:");
        eprintln!("    cargo run --release --features cuda --example gpu_mnist_lsm_v2 -- {seed}");
        std::process::exit(1);
    });
    let cached_rates = CachedRates::load(&rates_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: Cannot load cached rates from '{rates_path}': {e}");
        eprintln!("  Run M10 training first:");
        eprintln!("    cargo run --release --features cuda --example gpu_mnist_lsm_v2 -- {seed}");
        std::process::exit(1);
    });
    let (n_hid, mlp_h1, mlp_h2, n_classes) = weights.dims();
    println!("  Loaded MLP weights: [{n_hid}→{mlp_h1}→{mlp_h2}→{n_classes}]");
    println!(
        "  Loaded cached rates: {} samples × {n_hid} hidden",
        cached_rates.n_samples()
    );

    // ── Verify: direct forward_cpu on full test set ──────────────────────────
    let n_total = cached_rates.n_samples().min(test_labels.len());
    let direct_correct: usize = (0..n_total)
        .filter(|&i| weights.forward_cpu(cached_rates.get(i)) == test_labels[i])
        .count();
    let direct_acc = 100.0 * direct_correct as f32 / n_total as f32;
    println!(
        "\n  Direct forward_cpu (all {n_total}): {direct_correct}/{n_total} = {direct_acc:.1}%"
    );

    // ── Prepare V2 demo subset ───────────────────────────────────────────────
    let demo_count = count.min(n_total);
    let direct_subset: usize = (0..demo_count)
        .filter(|&i| weights.forward_cpu(cached_rates.get(i)) == test_labels[i])
        .count();
    let direct_subset_acc = 100.0 * direct_subset as f32 / demo_count as f32;
    println!(
        "  Direct forward_cpu (first {demo_count}): {direct_subset}/{demo_count} = {direct_subset_acc:.1}%"
    );

    // ── Create dataset samples (labels only — features unused for M11) ───────
    let samples: Vec<DatasetSample> = (0..demo_count)
        .map(|i| DatasetSample {
            features: 0, // SNN INFER uses cached rates, not features
            label: test_labels[i] as u64,
        })
        .collect();

    // ── Create inference model ───────────────────────────────────────────────
    let model = InferenceModel {
        weights,
        cached_rates,
    };

    // ── Build MMIO devices ───────────────────────────────────────────────────
    let snn = V2MmioSnnBridgeDevice::with_model(8, 4, 2, seed, model);
    let dataset = V2MmioDatasetDevice::from_samples(samples);
    let combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(
        seed, snn, dataset,
    ));
    let mmio_handle = V2MmioHandle::from_rc(combined.clone());

    // ── V2 Assembly: inference loop ──────────────────────────────────────────
    let program_source = r#"
; M11: V2+SNN MMIO Inference Loop
; For each sample: LOAD_SAMPLE → SNN INFER → SUBMIT_PREDICTION
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
; SNN INFER(R0) — runs MLP forward_cpu on cached rates[R0]
    STB [SNN_DATA], R0
    LDI R2, 7                  ; cmd INFER
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

    let words = assemble_v2(program_source).expect("assemble M11 inference program");
    println!("\n  Assembled {} instruction words", words.len());

    // ── Build V2 CPU ─────────────────────────────────────────────────────────
    println!("  Building V2 CPU (128×128×4 grid)...");
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

    // ── Execute ──────────────────────────────────────────────────────────────
    println!("\n  Running inference on {demo_count} samples...");
    let max_cycles: u64 = (demo_count as u64) * 200 + 1000; // ~200 cycles per sample + overhead
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

        // Progress indicator every 10%
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

    // ── Results ──────────────────────────────────────────────────────────────
    let halted = cpu.is_halted();
    let v2_correct = cpu.read_reg(&sim, 4);
    let v2_acc = 100.0 * v2_correct as f32 / demo_count as f32;
    let per_sample_ms = if demo_count > 0 {
        elapsed.as_secs_f64() * 1000.0 / demo_count as f64
    } else {
        0.0
    };

    println!("\n  ── M11 Results ──────────────────────────────────────────────");
    println!("  Halted:           {halted}");
    println!("  Cycles:           {cycles}");
    println!("  Instructions:     {instructions}");
    println!("  Wall-clock:       {:.2}s", elapsed.as_secs_f32());
    println!("  Per-sample:       {per_sample_ms:.2}ms");
    println!();
    println!("  V2 MMIO accuracy: {v2_correct}/{demo_count} = {v2_acc:.1}%");
    println!("  Direct CPU:       {direct_subset}/{demo_count} = {direct_subset_acc:.1}%");
    let match_ok = v2_correct as usize == direct_subset;
    println!(
        "  Parity:           {} (V2={v2_correct}, direct={direct_subset})",
        if match_ok { "MATCH" } else { "MISMATCH" }
    );
    println!();
    println!(
        "  Full test set:    {direct_correct}/{n_total} = {direct_acc:.1}% (direct forward_cpu)"
    );

    // ── Gate ─────────────────────────────────────────────────────────────────
    println!("\n  ── M11 Gate Checks ─────────────────────────────────────────");
    println!(
        "  [{}] V2 halted normally",
        if halted { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Parity: V2 == direct forward_cpu",
        if match_ok { "PASS" } else { "FAIL" }
    );
    let acc_gate = direct_acc >= 80.0;
    println!(
        "  [{}] Full accuracy >= 80% ({direct_acc:.1}%)",
        if acc_gate { "PASS" } else { "FAIL" }
    );

    let all_pass = halted && match_ok && acc_gate;
    if all_pass {
        println!("\n  PASS — M11: V2+SNN MMIO inference end-to-end verified.");
    } else {
        println!("\n  FAIL — see above for details.");
        std::process::exit(1);
    }
}
