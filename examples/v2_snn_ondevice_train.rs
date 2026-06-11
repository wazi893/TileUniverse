//! Option 2 — On-device readout training driven by V2 assembly.
//!
//! Loads M10 seed-42 live SNN checkpoint as a frozen feature extractor; trains
//! a fresh linear readout `W[n_hid × 10] + b[10]` from random init using the
//! delta rule (`SNN_CMD_TRAIN_ONE`) issued from V2 machine code. The bridge
//! device performs the f32 weight update; V2 assembly only orchestrates the
//! per-sample LOAD_IMAGE → SNN_RUN → TRAIN_ONE → SUBMIT loop.
//!
//! Per-epoch accuracy is measured by reading the dataset device's GET_CORRECT
//! counter before and after each `SHOWCASE_SNN_TRAIN_EPOCH` execution.
//!
//! Comparison baseline: M7 Phase 1 — frozen IH + linear readout, GPU + Adam,
//! 69.3% MNIST test accuracy after 60 epochs.
//!
//! Usage:
//! ```bash
//! cargo run --release --example v2_snn_ondevice_train
//! cargo run --release --example v2_snn_ondevice_train -- --epochs 10 --train 500
//! ```

fn main() {
    use engine::simulation::Simulation;
    use engine::snn::mlp_weights::LiveSnnModel;
    use engine::snn::mnist::MnistDataset;
    use engine::tile_cpu::v2_mmio_devices::{MMIO_DATASET_CMD, MMIO_DATASET_DATA};
    use engine::tile_cpu::v2_showcase::{SHOWCASE_SNN_EVAL_READOUT, SHOWCASE_SNN_TRAIN_EPOCH};
    use engine::tile_cpu::{
        DatasetSample, V2Builder, V2MmioCombinedDevice, V2MmioDatasetDevice, V2MmioDevice,
        V2MmioHandle, V2MmioSnnBridgeDevice, assemble_v2,
    };
    use std::rc::Rc;
    use std::time::Instant;

    // ── Args ────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 42;
    let mut n_train: usize = 1000;
    let mut n_test: usize = 500;
    let mut n_epochs: usize = 10;
    let mut lr: f32 = 0.01;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("seed");
            }
            "--train" => {
                i += 1;
                n_train = args[i].parse().expect("train count");
            }
            "--test" => {
                i += 1;
                n_test = args[i].parse().expect("test count");
            }
            "--epochs" => {
                i += 1;
                n_epochs = args[i].parse().expect("epochs");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("lr");
            }
            _ => {}
        }
        i += 1;
    }

    println!("===============================================================");
    println!("  Option 2: V2 + On-Device Readout Training");
    println!("  seed={seed}  n_train={n_train}  n_test={n_test}  epochs={n_epochs}  lr={lr}");
    println!("  Baseline to match: M7 Phase 1 = 69.3% MNIST (GPU + Adam, 60ep)");
    println!("===============================================================\n");

    // ── Load MNIST + checkpoint ─────────────────────────────────────────────
    let train_ds = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("load MNIST train data");
    let train_imgs: Vec<Vec<u8>> = train_ds.images[..n_train.min(train_ds.images.len())].to_vec();
    let train_lbls: Vec<usize> = train_ds.labels[..n_train.min(train_ds.labels.len())]
        .iter()
        .map(|&l| l as usize)
        .collect();
    let n_train = train_imgs.len();
    println!("  Loaded {} train samples", n_train);

    let ckpt_path = format!("checkpoints/m10_seed{seed}.snn");
    let model = LiveSnnModel::load(&ckpt_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: cannot load {ckpt_path}: {e}");
        eprintln!("  Train it first:");
        eprintln!("    cargo run --release --features cuda --example gpu_mnist_lsm_v2 -- {seed}");
        std::process::exit(1);
    });
    let n_hid = model.n_hidden;
    let n_classes = model.n_classes;
    println!(
        "  Loaded SNN model: {} neurons, {} synapses, n_hidden={n_hid}, n_classes={n_classes}",
        model.n_neurons(),
        model.n_synapses()
    );

    // ── Build bridge with trainable readout enabled ─────────────────────────
    let mut snn = V2MmioSnnBridgeDevice::with_live_model(
        model.n_input,
        model.n_hidden,
        model.n_readout,
        seed,
        model,
        train_imgs.clone(),
    );
    snn.enable_trainable_readout(n_hid, n_classes, lr, seed);
    println!("  Trainable readout: W[{n_hid}×{n_classes}] Glorot-uniform, lr={lr}");

    // ── Dataset device for label lookup / SUBMIT_PREDICTION ─────────────────
    let samples: Vec<DatasetSample> = (0..n_train)
        .map(|i| DatasetSample {
            features: 0,
            label: train_lbls[i] as u64,
        })
        .collect();
    let dataset = V2MmioDatasetDevice::from_samples(samples);
    let combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(
        seed, snn, dataset,
    ));

    // ── Pre-training sanity check: GET_CORRECT before any epoch ─────────────
    let dataset_ref = combined.dataset.as_ref().expect("dataset device present");
    let read_correct = || -> u64 {
        dataset_ref.write(MMIO_DATASET_CMD, 4); // GET_CORRECT
        dataset_ref.read(MMIO_DATASET_DATA)
    };
    let initial_correct = read_correct();
    if initial_correct != 0 {
        eprintln!("  WARN: dataset.correct={initial_correct} before training (expected 0)");
    }

    // ── Assemble training program once ──────────────────────────────────────
    let words =
        assemble_v2(SHOWCASE_SNN_TRAIN_EPOCH.source).expect("assemble SHOWCASE_SNN_TRAIN_EPOCH");
    println!(
        "  Assembled SHOWCASE_SNN_TRAIN_EPOCH: {} instruction words",
        words.len()
    );

    // ── Epoch loop ──────────────────────────────────────────────────────────
    println!("\n  Training trajectory:");
    println!(
        "  {:<6} {:>10} {:>12} {:>10}",
        "epoch", "correct", "acc", "wall (s)"
    );
    println!("  {:-<44}", "");

    let mut accs: Vec<f32> = Vec::with_capacity(n_epochs);
    let mut prev_correct = initial_correct;
    let train_t0 = Instant::now();

    for ep in 0..n_epochs {
        // Build a fresh V2 CPU per epoch — bridge state (W, b) and dataset counter
        // persist via Rc<Combined>, only the CPU PC + registers reset.
        let mmio_handle = V2MmioHandle::from_rc(combined.clone());
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&words)
            .with_rom_size(64)
            .with_ram_size(64)
            .with_mmio(mmio_handle)
            .build(&mut sim);

        let max_cycles: u64 = (n_train as u64) * 200 + 1000;
        let ep_t0 = Instant::now();
        for _ in 0..max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
        }
        let ep_secs = ep_t0.elapsed().as_secs_f32();
        if !cpu.is_halted() {
            eprintln!("  WARN: epoch {ep} did not halt within max_cycles");
        }

        let post = read_correct();
        let epoch_correct = post - prev_correct;
        let epoch_acc = 100.0 * epoch_correct as f32 / n_train as f32;
        accs.push(epoch_acc);
        prev_correct = post;
        println!(
            "  {:<6} {:>10} {:>11.2}% {:>9.2}s",
            ep + 1,
            epoch_correct,
            epoch_acc,
            ep_secs
        );
    }
    let total_secs = train_t0.elapsed().as_secs_f32();

    // ── Held-out test evaluation ────────────────────────────────────────────
    // Transfer trained readout weights to a fresh bridge configured with the
    // test images, run SHOWCASE_SNN_EVAL_READOUT (no weight updates), count
    // correct predictions. This gives the apples-to-apples test number to
    // compare against M7P1's 69.3% test accuracy.
    let (trained_w, trained_b) = combined.snn.readout_weights();

    // Load test images from disk and reload the model checkpoint (LiveSnnModel
    // is consumed by the bridge constructor, so we deserialize twice).
    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("load MNIST test data");
    let n_test = n_test.min(test_ds.images.len());
    let test_imgs: Vec<Vec<u8>> = test_ds.images[..n_test].to_vec();
    let test_lbls: Vec<usize> = test_ds.labels[..n_test]
        .iter()
        .map(|&l| l as usize)
        .collect();
    let test_model = LiveSnnModel::load(&ckpt_path).expect("re-load checkpoint");

    let mut test_snn = V2MmioSnnBridgeDevice::with_live_model(
        test_model.n_input,
        test_model.n_hidden,
        test_model.n_readout,
        seed,
        test_model,
        test_imgs,
    );
    test_snn.set_readout_weights(n_hid, n_classes, trained_w, trained_b);
    let test_samples: Vec<DatasetSample> = (0..n_test)
        .map(|i| DatasetSample {
            features: 0,
            label: test_lbls[i] as u64,
        })
        .collect();
    let test_dataset = V2MmioDatasetDevice::from_samples(test_samples);
    let test_combined = Rc::new(V2MmioCombinedDevice::with_snn_and_dataset(
        seed,
        test_snn,
        test_dataset,
    ));

    let eval_words =
        assemble_v2(SHOWCASE_SNN_EVAL_READOUT.source).expect("assemble SHOWCASE_SNN_EVAL_READOUT");
    let mmio_handle = V2MmioHandle::from_rc(test_combined.clone());
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&eval_words)
        .with_rom_size(64)
        .with_ram_size(64)
        .with_mmio(mmio_handle)
        .build(&mut sim);

    let test_t0 = Instant::now();
    let max_cycles: u64 = (n_test as u64) * 200 + 1000;
    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
    }
    let test_secs = test_t0.elapsed().as_secs_f32();
    let test_dataset_ref = test_combined.dataset.as_ref().expect("test dataset");
    test_dataset_ref.write(MMIO_DATASET_CMD, 4);
    let test_correct = test_dataset_ref.read(MMIO_DATASET_DATA);
    let test_acc = 100.0 * test_correct as f32 / n_test as f32;
    println!("\n  Test-set eval: {test_correct}/{n_test} = {test_acc:.2}%  ({test_secs:.1}s)");

    // ── Summary ─────────────────────────────────────────────────────────────
    let final_acc = *accs.last().unwrap_or(&0.0);
    let best_acc = accs.iter().cloned().fold(0.0f32, f32::max);
    let first_acc = *accs.first().unwrap_or(&0.0);

    println!("\n===============================================================");
    println!("  Summary");
    println!("===============================================================");
    println!(
        "  Total train wall-clock:  {:.1}s ({:.2}s/epoch avg)",
        total_secs,
        total_secs / n_epochs as f32
    );
    println!(
        "  First-epoch train acc:   {:.2}% (random readout init)",
        first_acc
    );
    println!("  Final-epoch train acc:   {:.2}%", final_acc);
    println!("  Best-epoch train acc:    {:.2}%", best_acc);
    println!("  Δ first→last (train):    {:+.2}pp", final_acc - first_acc);
    println!();
    println!(
        "  >>> Held-out test accuracy: {:.2}% ({} samples) <<<",
        test_acc, n_test
    );
    println!();
    println!("  Reference baselines (test accuracy):");
    println!("    Chance (10 classes)                  = 10.0%");
    println!("    M6 ceiling (T-norm, frozen IH)       = 62.4%");
    println!("    M7P1 (frozen IH + linear, GPU+Adam)  = 69.3%   ← apples-to-apples target");
    println!("    M10 frozen MLP readout (M12 path)    = 82.2-84.0% (3 seeds)");
    let delta_m7p1 = test_acc - 69.3;
    println!();
    if test_acc >= 55.0 {
        println!("  PASS gate 1 (>=55%, learning verified): {:.2}%", test_acc);
    } else {
        println!("  FAIL gate 1 (>=55%, learning verified): {:.2}%", test_acc);
    }
    if test_acc >= 65.0 {
        println!(
            "  PASS gate 2 (>=65%, close to M7P1):     {:.2}%  (Δ vs M7P1: {:+.1}pp)",
            test_acc, delta_m7p1
        );
    } else {
        println!(
            "  -- gate 2 (>=65%, close to M7P1):       {:.2}%  (Δ vs M7P1: {:+.1}pp)",
            test_acc, delta_m7p1
        );
    }
}
