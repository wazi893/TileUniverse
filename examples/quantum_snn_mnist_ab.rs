//! Quantum-SNN vs Classical SNN — MNIST A/B Sanity Check
//!
//! Same SNN backbone (n_inputs=784, hidden=[128], n_outputs=10), vary only
//! the interference mode used for decision making + learning. R-STDP reward
//! is +50 on correct, -50 on incorrect.
//!
//! Goal: see whether any quantum interference mode gives a reproducible lift
//! over classical winner-take-all on real digit classification.
//!
//! Run with: cargo run --release --example quantum_snn_mnist_ab

use engine::snn::quantum_hybrid::{InterferenceMode, QuantumSNN, QuantumSNNConfig};
use engine::snn::MnistDataset;
use std::time::Instant;

const N_INPUTS: usize = 784;
const HIDDEN: &[usize] = &[128];
const N_OUTPUTS: usize = 10;
const TICKS_PER_DECISION: usize = 30;
const MIX_ANGLE: f64 = 0.1;

const N_TRAIN: usize = 1000; // 100 per class on average
const N_TEST: usize = 500;
const N_EPOCHS: usize = 5;
const SEEDS: &[u64] = &[42, 123, 777];

fn make_config(mode: InterferenceMode) -> QuantumSNNConfig {
    QuantumSNNConfig {
        n_inputs: N_INPUTS,
        hidden_layers: HIDDEN.to_vec(),
        n_outputs: N_OUTPUTS,
        interference_mode: mode,
        interference_depth: 1,
        mix_angle: MIX_ANGLE,
        ticks_per_decision: TICKS_PER_DECISION,
        learning_enabled: true,
        trigger_threshold: 50,
        trigger_count: 5,
        duplicate_inputs: false,
    }
}

/// Balanced subset: roughly equal samples per class, deterministic order.
fn balanced_subset(ds: &MnistDataset, per_class: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); 10];
    for (i, &lbl) in ds.labels.iter().enumerate() {
        if (lbl as usize) < 10 {
            by_class[lbl as usize].push(i);
        }
    }
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for c in 0..10 {
        for &idx in by_class[c].iter().take(per_class) {
            images.push(ds.images[idx].clone());
            labels.push(c as u8);
        }
    }
    (images, labels)
}

/// Deterministic Fisher-Yates shuffle.
fn shuffle(state: &mut u64, indices: &mut [usize]) {
    for i in (1..indices.len()).rev() {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((*state >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn train_and_eval(
    mode_name: &str,
    mode: InterferenceMode,
    use_classical_decision: bool,
    train_imgs: &[Vec<u8>],
    train_lbls: &[u8],
    test_imgs: &[Vec<u8>],
    test_lbls: &[u8],
    seed: u64,
) -> f32 {
    let mut qsnn = QuantumSNN::with_seed(make_config(mode), seed);
    qsnn.snn_mut().randomize_weights(seed.wrapping_add(0xABCD));

    let mut order: Vec<usize> = (0..train_imgs.len()).collect();
    let mut rng_state = seed.wrapping_add(0xDEAD_BEEF);

    for _epoch in 0..N_EPOCHS {
        shuffle(&mut rng_state, &mut order);
        for &idx in &order {
            qsnn.set_inputs(&train_imgs[idx]);
            let predicted = if use_classical_decision {
                let _ = qsnn.decide(); // still run forward + update internal state
                qsnn.classical_action()
            } else {
                qsnn.decide()
            };
            let reward: i8 = if predicted == train_lbls[idx] as usize { 50 } else { -50 };
            qsnn.learn(reward);
        }
    }

    // Evaluate (no learning)
    let mut correct = 0usize;
    for (img, &lbl) in test_imgs.iter().zip(test_lbls.iter()) {
        qsnn.set_inputs(img);
        let predicted = if use_classical_decision {
            let _ = qsnn.decide();
            qsnn.classical_action()
        } else {
            qsnn.decide()
        };
        if predicted == lbl as usize {
            correct += 1;
        }
    }
    let acc = correct as f32 / test_imgs.len() as f32 * 100.0;
    println!("    [{}] seed={} acc={:.1}%", mode_name, seed, acc);
    acc
}

fn main() {
    println!("================================================================");
    println!("  QUANTUM-SNN vs CLASSICAL — MNIST A/B SANITY CHECK");
    println!("================================================================");

    let train_ds = MnistDataset::load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("Failed to load MNIST train set");
    let test_ds = MnistDataset::load(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    )
    .expect("Failed to load MNIST test set");

    let (train_imgs, train_lbls) = balanced_subset(&train_ds, N_TRAIN / 10);
    let (test_imgs, test_lbls) = balanced_subset(&test_ds, N_TEST / 10);

    println!("Config:");
    println!("  n_inputs={}, hidden={:?}, n_outputs={}", N_INPUTS, HIDDEN, N_OUTPUTS);
    println!("  ticks_per_decision={}, mix_angle={}", TICKS_PER_DECISION, MIX_ANGLE);
    println!(
        "  train={} ({}/class), test={} ({}/class), epochs={}",
        train_imgs.len(),
        N_TRAIN / 10,
        test_imgs.len(),
        N_TEST / 10,
        N_EPOCHS
    );
    println!("  seeds={:?}\n", SEEDS);

    // Modes to compare. `use_classical_decision` toggles whether the action
    // selection uses quantum measurement (false) or WTA argmax (true).
    let modes: &[(&str, InterferenceMode, bool)] = &[
        ("Classical-WTA", InterferenceMode::None, true),
        ("NoInterf-Prop", InterferenceMode::None, false),
        ("SoftMix",       InterferenceMode::SoftMix, false),
        ("Grover",        InterferenceMode::Grover, false),
        ("AntiGrover",    InterferenceMode::AntiGrover, false),
        ("Adaptive",      InterferenceMode::Adaptive, false),
    ];

    let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
    for &(name, mode, use_classical) in modes {
        println!("--- Mode: {} ---", name);
        let t0 = Instant::now();
        let mut accs = Vec::new();
        for &seed in SEEDS {
            let acc = train_and_eval(
                name, mode, use_classical,
                &train_imgs, &train_lbls,
                &test_imgs, &test_lbls,
                seed,
            );
            accs.push(acc);
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let mean: f32 = accs.iter().sum::<f32>() / accs.len() as f32;
        let var: f32 = accs.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32;
        let std = var.sqrt();
        println!("  -> mean={:.1}% (±{:.1}pp)  [{:.1}s for {} seeds]\n", mean, std, elapsed, SEEDS.len());
        results.push((name, accs));
    }

    println!("================================================================");
    println!("  SUMMARY (test accuracy, mean ± std across {} seeds)", SEEDS.len());
    println!("================================================================");
    println!("{:<18} {:>10} {:>10} {:>10} {:>14}", "Mode", "Seed42", "Seed123", "Seed777", "Mean ± Std");
    println!("{:-<70}", "");
    for (name, accs) in &results {
        let mean: f32 = accs.iter().sum::<f32>() / accs.len() as f32;
        let var: f32 = accs.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32;
        let std = var.sqrt();
        print!("{:<18}", name);
        for &a in accs {
            print!(" {:>9.1}%", a);
        }
        println!(" {:>9.1}% ± {:.1}pp", mean, std);
    }

    // Pick the best non-classical mode, compare against classical
    let classical_mean: f32 = {
        let r = &results[0].1;
        r.iter().sum::<f32>() / r.len() as f32
    };
    let mut best: Option<(&str, f32)> = None;
    for (name, accs) in results.iter().skip(1) {
        let m: f32 = accs.iter().sum::<f32>() / accs.len() as f32;
        if best.map_or(true, |(_, bm)| m > bm) {
            best = Some((name, m));
        }
    }
    if let Some((bn, bm)) = best {
        let delta = bm - classical_mean;
        println!(
            "\nBest non-classical: {} at {:.1}% (Δ vs Classical-WTA: {:+.1}pp)",
            bn, bm, delta
        );
    }
}
