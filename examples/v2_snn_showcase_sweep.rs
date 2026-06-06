//! V2 SNN MNIST inference cross-seed sweep.
//!
//! Runs both M11 (`SHOWCASE_SNN_CACHED_MNIST`) and M12 (`SHOWCASE_SNN_LIVE_MNIST`)
//! over all three M10 checkpoint seeds {42, 123, 777} at 500 samples each, and
//! emits a single reproducibility table. Validates parity against the direct
//! CPU reference path for every (seed, mode) cell.
//!
//! Designed as the canonical "is the SNN-on-V2 bridge still solid" smoke test
//! and as the headline result for the SHOWCASE-promotion polish pass.
//!
//! Usage:
//! ```bash
//! cargo run --release --example v2_snn_showcase_sweep
//! cargo run --release --example v2_snn_showcase_sweep -- --count 100
//! ```

fn main() {
    use engine::simulation::Simulation;
    use engine::snn::mlp_weights::{CachedRates, LiveSnnModel, MlpWeights};
    use engine::snn::mnist::MnistDataset;
    use engine::tile_cpu::v2_showcase::{
        SHOWCASE_SNN_CACHED_MNIST, SHOWCASE_SNN_LIVE_MNIST, V2ShowcaseProgram,
    };
    use engine::tile_cpu::{
        DatasetSample, InferenceModel, V2Builder, V2MmioCombinedDevice, V2MmioDatasetDevice,
        V2MmioHandle, V2MmioSnnBridgeDevice, assemble_v2,
    };
    use std::rc::Rc;
    use std::time::Instant;

    // ── Args ────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut count: usize = 500;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--count" {
            i += 1;
            count = args[i].parse().expect("invalid --count");
        }
        i += 1;
    }
    let seeds: &[u64] = &[42, 123, 777];

    println!("===============================================================");
    println!("  V2 SNN MNIST Showcase Sweep");
    println!("  Seeds: {:?}   Samples per run: {}", seeds, count);
    println!("===============================================================\n");

    // ── Load MNIST once ─────────────────────────────────────────────────────
    let test_ds = MnistDataset::load(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    )
    .expect("load MNIST test data — ensure data/ directory contains MNIST files");
    let test_labels: Vec<usize> = test_ds.labels.iter().map(|&l| l as usize).collect();
    let demo_count = count.min(test_labels.len());

    // ── Result row ──────────────────────────────────────────────────────────
    #[derive(Clone)]
    struct Row {
        mode: &'static str,
        seed: u64,
        v2_correct: u64,
        direct_correct: usize,
        cycles: u64,
        wallclock_ms: f64,
    }
    let mut rows: Vec<Row> = Vec::new();

    // ── Helper: assemble + run one showcase, accept any pre-built MMIO ──────
    fn run_showcase(
        program: &V2ShowcaseProgram,
        mmio: V2MmioHandle,
        demo_count: usize,
    ) -> (bool, u64, u64, f64) {
        let words = assemble_v2(program.source).expect("assemble showcase program");
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&words)
            .with_rom_size(64)
            .with_ram_size(64)
            .with_mmio(mmio)
            .build(&mut sim);

        let max_cycles: u64 = (demo_count as u64) * 200 + 1000;
        let t0 = Instant::now();
        let mut cycles = 0u64;
        for _ in 0..max_cycles {
            if cpu.is_halted() { break; }
            cpu.step(&mut sim);
            cycles += 1;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let v2_correct = cpu.read_reg(&sim, 4);
        (cpu.is_halted(), v2_correct, cycles, elapsed_ms)
    }

    // ── Per-seed loop ───────────────────────────────────────────────────────
    for &seed in seeds {
        println!("─── seed {seed} ───");

        // M11 cached-rate path
        {
            let weights_path = format!("checkpoints/m10_seed{seed}.mlp");
            let rates_path   = format!("checkpoints/m10_seed{seed}.rates");
            let weights = MlpWeights::load(&weights_path)
                .unwrap_or_else(|e| panic!("M11: {weights_path}: {e}"));
            let cached_rates = CachedRates::load(&rates_path)
                .unwrap_or_else(|e| panic!("M11: {rates_path}: {e}"));

            // Direct reference: argmax through MLP forward_cpu on first demo_count samples
            let direct_correct: usize = (0..demo_count)
                .filter(|&i| weights.forward_cpu(cached_rates.get(i)) == test_labels[i])
                .count();

            let samples: Vec<DatasetSample> = (0..demo_count)
                .map(|i| DatasetSample { features: 0, label: test_labels[i] as u64 })
                .collect();
            let model = InferenceModel { weights, cached_rates };
            let snn = V2MmioSnnBridgeDevice::with_model(8, 4, 2, seed, model);
            let dataset = V2MmioDatasetDevice::from_samples(samples);
            let combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(seed, snn, dataset));
            let mmio = V2MmioHandle::from_rc(combined);

            let (halted, v2_correct, cycles, wallclock_ms) =
                run_showcase(&SHOWCASE_SNN_CACHED_MNIST, mmio, demo_count);

            let v2_acc = 100.0 * v2_correct as f32 / demo_count as f32;
            let direct_acc = 100.0 * direct_correct as f32 / demo_count as f32;
            let parity = v2_correct as usize == direct_correct;
            println!(
                "  M11 cached  halted={halted}  acc={:.2}% (direct={:.2}%)  parity={}  cyc={cycles} ({:.2}ms)",
                v2_acc, direct_acc,
                if parity { "MATCH" } else { "MISMATCH" },
                wallclock_ms / demo_count as f64,
            );
            rows.push(Row {
                mode: "M11 cached", seed,
                v2_correct, direct_correct, cycles, wallclock_ms,
            });
        }

        // M12 live LIF path
        {
            let snn_path = format!("checkpoints/m10_seed{seed}.snn");
            let model = match LiveSnnModel::load(&snn_path) {
                Ok(m) => m,
                Err(e) => {
                    println!("  M12 live    SKIP — checkpoints/m10_seed{seed}.snn not loadable: {e}");
                    continue;
                }
            };

            // Direct reference: full live LIF sim + MLP forward, matches the bridge implementation.
            let demo_images: Vec<Vec<u8>> = test_ds.images[..demo_count].to_vec();
            let direct_correct = {
                let mut ok = 0usize;
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
                            if refract[i] > 0 { refract[i] -= 1; spiked[i] = 0; continue; }
                            let mut rng = tick_seed ^ (i as u32).wrapping_mul(2654435761);
                            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                            let rand_val = (rng >> 24) as u8;
                            if rand_val < rates[i] {
                                spiked[i] = 1; v_mem[i] = -128; refract[i] = 2;
                            } else { spiked[i] = 0; }
                        }
                        let mut currents = vec![0i32; n_total];
                        for src in 0..n_total {
                            if spiked[src] != 0 {
                                let s = model.syn_ptr[src] as usize;
                                let e = model.syn_ptr[src + 1] as usize;
                                for j in s..e {
                                    let tgt = model.targets[j] as usize;
                                    if tgt < n_total { currents[tgt] += model.weights[j] as i32 * 2; }
                                }
                            }
                        }
                        for i in model.n_input..n_total {
                            if refract[i] > 0 { refract[i] -= 1; spiked[i] = 0; continue; }
                            let v = ((v_mem[i] as i32 * model.leaks[i] as i32) >> 8) + currents[i];
                            let v = v.clamp(-32768, 32767);
                            if v >= model.thresholds[i] as i32 {
                                spiked[i] = 1; v_mem[i] = -128; refract[i] = 2;
                            } else { spiked[i] = 0; v_mem[i] = v as i16; }
                        }
                        for h in 0..model.n_hidden {
                            if spiked[model.n_input + h] != 0 { hidden_counts[h] += 1; }
                        }
                        for i in 0..model.n_input { spiked[i] = 0; }
                    }
                    let rates_f: Vec<f32> = hidden_counts.iter()
                        .map(|&c| c as f32 / model.n_ticks as f32).collect();
                    if model.mlp.forward_cpu(&rates_f) == test_labels[idx] { ok += 1; }
                }
                ok
            };

            let samples: Vec<DatasetSample> = (0..demo_count)
                .map(|i| DatasetSample { features: 0, label: test_labels[i] as u64 })
                .collect();
            let snn = V2MmioSnnBridgeDevice::with_live_model(
                model.n_input, model.n_hidden, model.n_readout, seed, model, demo_images,
            );
            let dataset = V2MmioDatasetDevice::from_samples(samples);
            let combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(seed, snn, dataset));
            let mmio = V2MmioHandle::from_rc(combined);

            let (halted, v2_correct, cycles, wallclock_ms) =
                run_showcase(&SHOWCASE_SNN_LIVE_MNIST, mmio, demo_count);

            let v2_acc = 100.0 * v2_correct as f32 / demo_count as f32;
            let direct_acc = 100.0 * direct_correct as f32 / demo_count as f32;
            let parity = v2_correct as usize == direct_correct;
            println!(
                "  M12 live    halted={halted}  acc={:.2}% (direct={:.2}%)  parity={}  cyc={cycles} ({:.2}ms/sample)",
                v2_acc, direct_acc,
                if parity { "MATCH" } else { "MISMATCH" },
                wallclock_ms / demo_count as f64,
            );
            rows.push(Row {
                mode: "M12 live", seed,
                v2_correct, direct_correct, cycles, wallclock_ms,
            });
        }
        println!();
    }

    // ── Summary table ───────────────────────────────────────────────────────
    println!("===============================================================");
    println!("  Cross-Seed Summary ({} samples)", demo_count);
    println!("===============================================================");
    println!("{:<12} {:>6} {:>9} {:>9} {:>8} {:>8} {:>14}",
             "Mode", "Seed", "V2 acc", "Direct", "Parity", "Cyc/smp", "ms/sample");
    println!("{:-<74}", "");
    let mut all_parity = true;
    for r in &rows {
        let v2_acc = 100.0 * r.v2_correct as f32 / demo_count as f32;
        let direct_acc = 100.0 * r.direct_correct as f32 / demo_count as f32;
        let parity = r.v2_correct as usize == r.direct_correct;
        all_parity &= parity;
        println!("{:<12} {:>6} {:>8.2}% {:>8.2}% {:>8} {:>8} {:>13.3}",
            r.mode, r.seed, v2_acc, direct_acc,
            if parity { "MATCH" } else { "MISMATCH" },
            r.cycles / demo_count as u64,
            r.wallclock_ms / demo_count as f64,
        );
    }

    // ── Per-mode mean ───────────────────────────────────────────────────────
    for &mode in ["M11 cached", "M12 live"].iter() {
        let mode_rows: Vec<&Row> = rows.iter().filter(|r| r.mode == mode).collect();
        if mode_rows.is_empty() { continue; }
        let mean_acc: f32 = mode_rows.iter()
            .map(|r| 100.0 * r.v2_correct as f32 / demo_count as f32)
            .sum::<f32>() / mode_rows.len() as f32;
        let mean_var: f32 = mode_rows.iter()
            .map(|r| {
                let a = 100.0 * r.v2_correct as f32 / demo_count as f32;
                (a - mean_acc).powi(2)
            })
            .sum::<f32>() / mode_rows.len() as f32;
        let std = mean_var.sqrt();
        println!("  {:<12}  mean={:.2}% (±{:.2}pp across {} seeds)",
                 mode, mean_acc, std, mode_rows.len());
    }

    println!();
    println!("Overall parity:  {}",
        if all_parity { "ALL MATCH (V2 == direct reference for every row)" }
        else { "MISMATCH DETECTED — investigate" });
}
