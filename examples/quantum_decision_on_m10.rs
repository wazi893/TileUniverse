//! Quantum decision policies on top of the M10 MNIST classifier.
//!
//! Isolates the question: does quantum interference do anything useful as a
//! decision policy when applied to a classifier that is already working?
//!
//! Pipeline:
//!   M10 cached hidden rates → MLP forward → 10 logits
//!     → 5 decision policies (argmax / proportional / Grover / AntiGrover / SoftMix)
//!     → top-1 accuracy on 10k MNIST test samples × 3 seeds.
//!
//! This is the honest follow-up to `quantum_snn_mnist_ab` (R-STDP didn't learn).
//! Here the underlying classifier already hits ~83% — so we measure only the
//! decision-policy effect, not the learning rule.
//!
//! Run with: cargo run --release --example quantum_decision_on_m10

use engine::snn::MnistDataset;
use engine::snn::mlp_weights::{CachedRates, MlpWeights};
use std::time::Instant;

const SEEDS: &[u64] = &[42, 123, 777];
const N_CLASSES: usize = 10;
const SOFTMAX_TEMP: f32 = 1.0;
// Grover-style amplification strength (matches QuantumSNN default of 0.5).
const GROVER_G: f32 = 0.5;
const ANTIGROVER_G: f32 = -0.5;
const SOFTMIX_ANGLE: f32 = 0.1;

/// MLP forward returning the 10-dim logit vector (not just argmax).
fn mlp_logits(mlp: &MlpWeights, hidden_rates: &[f32]) -> [f32; N_CLASSES] {
    let n_hid = hidden_rates.len();
    let mlp_h1 = mlp.b1.len();
    let mlp_h2 = mlp.b2.len();

    let mut a1 = vec![0.0f32; mlp_h1];
    for j in 0..mlp_h1 {
        let mut sum = mlp.b1[j];
        for h in 0..n_hid {
            sum += hidden_rates[h] * mlp.w1[h * mlp_h1 + j];
        }
        a1[j] = sum.max(0.0);
    }

    let mut a2 = vec![0.0f32; mlp_h2];
    for k in 0..mlp_h2 {
        let mut sum = mlp.b2[k];
        for j in 0..mlp_h1 {
            sum += a1[j] * mlp.w2[j * mlp_h2 + k];
        }
        a2[k] = sum.max(0.0);
    }

    let mut logits = [0.0f32; N_CLASSES];
    for c in 0..N_CLASSES {
        let mut sum = mlp.b3[c];
        for k in 0..mlp_h2 {
            sum += a2[k] * mlp.w3[k * N_CLASSES + c];
        }
        logits[c] = sum;
    }
    logits
}

/// Numerically stable softmax with temperature.
fn softmax(logits: &[f32; N_CLASSES], temp: f32) -> [f32; N_CLASSES] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = [0.0f32; N_CLASSES];
    let mut sum = 0.0f32;
    for c in 0..N_CLASSES {
        exps[c] = ((logits[c] - max) / temp).exp();
        sum += exps[c];
    }
    if sum > 0.0 {
        for c in 0..N_CLASSES {
            exps[c] /= sum;
        }
    }
    exps
}

/// Argmax over a probability vector (or any score vector).
fn argmax(v: &[f32; N_CLASSES]) -> usize {
    let mut best = 0;
    let mut best_v = v[0];
    for c in 1..N_CLASSES {
        if v[c] > best_v {
            best_v = v[c];
            best = c;
        }
    }
    best
}

/// LCG step → [0,1) sample.
fn rng_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / ((1u64 << 31) as f32)
}

/// Sample from a probability distribution.
fn sample_categorical(probs: &[f32; N_CLASSES], rng: &mut u64) -> usize {
    let r = rng_next(rng);
    let mut c = 0.0f32;
    for i in 0..N_CLASSES {
        c += probs[i];
        if r < c {
            return i;
        }
    }
    N_CLASSES - 1
}

/// Grover-style amplification: amp' = amp + g·(amp − mean). g>0 amplifies the
/// peak, g<0 amplifies the trough (anti-Grover).
/// Matches `QuantumSNN::apply_grover_with_strength` in `quantum_hybrid.rs`.
fn grover_amplification(probs: &[f32; N_CLASSES], g: f32) -> [f32; N_CLASSES] {
    let amps: [f32; N_CLASSES] = std::array::from_fn(|i| probs[i].sqrt());
    let mean: f32 = amps.iter().sum::<f32>() / N_CLASSES as f32;
    let mut new_amps = [0.0f32; N_CLASSES];
    for i in 0..N_CLASSES {
        new_amps[i] = amps[i] + g * (amps[i] - mean);
    }
    // Renormalize amplitudes, then square to get probs.
    let norm_sq: f32 = new_amps.iter().map(|a| a * a).sum();
    let mut new_probs = [0.0f32; N_CLASSES];
    if norm_sq > 1e-12 {
        for i in 0..N_CLASSES {
            new_probs[i] = (new_amps[i] * new_amps[i]) / norm_sq;
        }
    }
    new_probs
}

/// Soft mixing: new_amp[i] = cos(θ)·amp[i] + sin(θ)/√2 · (amp[i-1] + amp[i+1]).
/// Matches `QuantumSNN::apply_soft_mix`.
fn softmix(probs: &[f32; N_CLASSES], angle: f32) -> [f32; N_CLASSES] {
    let amps: [f32; N_CLASSES] = std::array::from_fn(|i| probs[i].sqrt());
    let cos_a = angle.cos();
    let sin_a = angle.sin() / 2.0f32.sqrt();
    let mut new_amps = amps;
    for i in 0..N_CLASSES {
        let left = if i > 0 { amps[i - 1] } else { 0.0 };
        let right = if i + 1 < N_CLASSES { amps[i + 1] } else { 0.0 };
        new_amps[i] = cos_a * amps[i] + sin_a * (left + right);
    }
    let norm_sq: f32 = new_amps.iter().map(|a| a * a).sum();
    let mut new_probs = [0.0f32; N_CLASSES];
    if norm_sq > 1e-12 {
        for i in 0..N_CLASSES {
            new_probs[i] = (new_amps[i] * new_amps[i]) / norm_sq;
        }
    }
    new_probs
}

#[derive(Clone, Copy)]
enum Policy {
    /// argmax(logits) — the classical M10 baseline.
    Argmax,
    /// Sample categorically from softmax(logits) — pure proportional sampling.
    Proportional,
    /// Sample from softmax, but with Grover amplification applied (peak-amplifying).
    Grover,
    /// Sample from softmax, with anti-Grover amplification (trough-amplifying).
    AntiGrover,
    /// Sample from softmax, with soft-mix interference (neighbor blending).
    SoftMix,
    /// Take argmax of Grover-amplified distribution (deterministic Grover).
    GroverArgmax,
}

impl Policy {
    fn name(&self) -> &'static str {
        match self {
            Policy::Argmax => "Argmax (M10)",
            Policy::Proportional => "Prop-sample",
            Policy::Grover => "Grover-sample",
            Policy::AntiGrover => "AntiGrover-samp",
            Policy::SoftMix => "SoftMix-sample",
            Policy::GroverArgmax => "Grover-argmax",
        }
    }
    fn decide(&self, logits: &[f32; N_CLASSES], rng: &mut u64) -> usize {
        match self {
            Policy::Argmax => argmax(logits),
            Policy::Proportional => {
                let p = softmax(logits, SOFTMAX_TEMP);
                sample_categorical(&p, rng)
            }
            Policy::Grover => {
                let p = softmax(logits, SOFTMAX_TEMP);
                let q = grover_amplification(&p, GROVER_G);
                sample_categorical(&q, rng)
            }
            Policy::AntiGrover => {
                let p = softmax(logits, SOFTMAX_TEMP);
                let q = grover_amplification(&p, ANTIGROVER_G);
                sample_categorical(&q, rng)
            }
            Policy::SoftMix => {
                let p = softmax(logits, SOFTMAX_TEMP);
                let q = softmix(&p, SOFTMIX_ANGLE);
                sample_categorical(&q, rng)
            }
            Policy::GroverArgmax => {
                let p = softmax(logits, SOFTMAX_TEMP);
                let q = grover_amplification(&p, GROVER_G);
                argmax(&q)
            }
        }
    }
}

fn evaluate(
    policies: &[Policy],
    mlp: &MlpWeights,
    rates: &CachedRates,
    labels: &[u8],
    seed: u64,
) -> Vec<f32> {
    let mut rng = seed.wrapping_add(0xABCD_1234);
    let n = rates.n_samples();
    let mut correct = vec![0usize; policies.len()];
    for i in 0..n {
        let logits = mlp_logits(mlp, rates.get(i));
        let label = labels[i] as usize;
        for (p_idx, policy) in policies.iter().enumerate() {
            let pred = policy.decide(&logits, &mut rng);
            if pred == label {
                correct[p_idx] += 1;
            }
        }
    }
    correct
        .into_iter()
        .map(|c| c as f32 / n as f32 * 100.0)
        .collect()
}

fn main() {
    println!("================================================================");
    println!("  QUANTUM DECISION POLICIES ON M10 (MNIST test, 3 seeds)");
    println!("================================================================");

    let test_ds = MnistDataset::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
        .expect("Failed to load MNIST test set");
    println!("Loaded test set: {} samples", test_ds.labels.len());

    let policies = [
        Policy::Argmax,
        Policy::Proportional,
        Policy::Grover,
        Policy::AntiGrover,
        Policy::SoftMix,
        Policy::GroverArgmax,
    ];

    let mut per_seed: Vec<Vec<f32>> = Vec::new();
    let mut elapsed_total = 0.0f32;

    for &seed in SEEDS {
        let mlp_path = format!("checkpoints/m10_seed{}.mlp", seed);
        let rate_path = format!("checkpoints/m10_seed{}.rates", seed);
        let mlp = MlpWeights::load(&mlp_path)
            .unwrap_or_else(|e| panic!("Failed to load {mlp_path}: {e}"));
        let rates = CachedRates::load(&rate_path)
            .unwrap_or_else(|e| panic!("Failed to load {rate_path}: {e}"));

        if rates.n_samples() != test_ds.labels.len() {
            panic!(
                "rates n_samples ({}) != test labels ({}) for seed {}",
                rates.n_samples(),
                test_ds.labels.len(),
                seed
            );
        }

        let t0 = Instant::now();
        let accs = evaluate(&policies, &mlp, &rates, &test_ds.labels, seed);
        let elapsed = t0.elapsed().as_secs_f32();
        elapsed_total += elapsed;
        println!("\n--- seed {} (eval {:.1}s) ---", seed, elapsed);
        for (p, a) in policies.iter().zip(accs.iter()) {
            println!("  {:<18} {:>6.2}%", p.name(), a);
        }
        per_seed.push(accs);
    }

    println!("\n================================================================");
    println!(
        "  SUMMARY — mean ± std across {} seeds (eval {:.1}s)",
        SEEDS.len(),
        elapsed_total
    );
    println!("================================================================");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>14}",
        "Policy",
        format!("Seed{}", SEEDS[0]),
        format!("Seed{}", SEEDS[1]),
        format!("Seed{}", SEEDS[2]),
        "Mean ± Std"
    );
    println!("{:-<72}", "");
    for (p_idx, policy) in policies.iter().enumerate() {
        let accs: Vec<f32> = per_seed.iter().map(|v| v[p_idx]).collect();
        let mean: f32 = accs.iter().sum::<f32>() / accs.len() as f32;
        let var: f32 = accs.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32;
        let std = var.sqrt();
        print!("{:<18}", policy.name());
        for &a in &accs {
            print!(" {:>9.2}%", a);
        }
        println!(" {:>9.2}% ± {:.2}pp", mean, std);
    }

    // Reference comparison
    let argmax_mean: f32 = per_seed.iter().map(|v| v[0]).sum::<f32>() / SEEDS.len() as f32;
    println!("\nDeltas vs Argmax (M10 baseline = {:.2}%):", argmax_mean);
    for (p_idx, policy) in policies.iter().enumerate().skip(1) {
        let mean: f32 = per_seed.iter().map(|v| v[p_idx]).sum::<f32>() / SEEDS.len() as f32;
        println!("  {:<18} {:+.2}pp", policy.name(), mean - argmax_mean);
    }
}
