//! Quantum vs Softmax A/B Test (v2)
//!
//! Rigorous comparison using the ACTUAL QuantumSNN implementation vs
//! classical softmax sampling on the same SNN outputs.
//!
//! Run with: cargo run --release --example quantum_vs_softmax_ab_test

use engine::snn::{NeuronConfig, QuantumSNN, QuantumSNNConfig, SNNConfig, SNNNetwork, STDPConfig};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Classical Decision Mechanisms for Comparison
// ============================================================================

/// Softmax with temperature: p_i = exp(count_i / T) / Σ exp(count_j / T)
fn softmax_sample(spike_counts: &[u32], temperature: f64, rng: &mut u64) -> usize {
    let n = spike_counts.len();
    let total: u32 = spike_counts.iter().sum();

    if total == 0 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        return ((*rng >> 32) as usize) % n;
    }

    let max_count = *spike_counts.iter().max().unwrap() as f64;

    let exp_vals: Vec<f64> = spike_counts
        .iter()
        .map(|&c| ((c as f64 - max_count) / temperature).exp())
        .collect();

    let sum: f64 = exp_vals.iter().sum();
    let probs: Vec<f64> = exp_vals.iter().map(|&e| e / sum).collect();

    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = (*rng >> 33) as f64 / (1u64 << 31) as f64;

    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    n - 1
}

/// Proportional sampling: p_i = count_i / total
fn proportional_sample(spike_counts: &[u32], rng: &mut u64) -> usize {
    let n = spike_counts.len();
    let total: u32 = spike_counts.iter().sum();

    if total == 0 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        return ((*rng >> 32) as usize) % n;
    }

    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = (*rng >> 33) as f64 / (1u64 << 31) as f64;

    let mut cumulative = 0.0;
    for (i, &count) in spike_counts.iter().enumerate() {
        cumulative += count as f64 / total as f64;
        if r < cumulative {
            return i;
        }
    }
    n - 1
}

/// Winner-take-all (deterministic)
fn wta_sample(spike_counts: &[u32], rng: &mut u64) -> usize {
    let n = spike_counts.len();
    let max_count = *spike_counts.iter().max().unwrap_or(&0);

    if max_count == 0 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        return ((*rng >> 32) as usize) % n;
    }

    // Find all with max count
    let max_actions: Vec<usize> = spike_counts
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c == max_count)
        .map(|(i, _)| i)
        .collect();

    // Tiebreaker
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    max_actions[(*rng >> 32) as usize % max_actions.len()]
}

// ============================================================================
// Predator-Prey Environment
// ============================================================================

struct Predator {
    x: i32,
    y: i32,
    transition_counts: HashMap<(usize, usize), u32>,
    total_counts: [u32; 5],
    history: Vec<usize>,
    correct_predictions: u32,
    total_predictions: u32,
}

impl Predator {
    fn new(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
            transition_counts: HashMap::new(),
            total_counts: [1; 5],
            history: Vec::new(),
            correct_predictions: 0,
            total_predictions: 0,
        }
    }

    fn observe(&mut self, action: usize) {
        if let Some(&last) = self.history.last() {
            *self.transition_counts.entry((last, action)).or_insert(0) += 1;
            self.total_counts[last] += 1;
        }
        self.history.push(action);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    fn predict(&self) -> usize {
        if let Some(&last) = self.history.last() {
            (0..5)
                .max_by_key(|&next| {
                    self.transition_counts
                        .get(&(last, next))
                        .copied()
                        .unwrap_or(0)
                })
                .unwrap_or(4)
        } else {
            4
        }
    }

    fn check_catch_and_learn(&mut self, prey_x: i32, prey_y: i32, prey_action: usize) -> bool {
        // Track prediction accuracy
        let prediction = self.predict();
        self.total_predictions += 1;
        if prediction == prey_action {
            self.correct_predictions += 1;
        }

        // Check if caught
        (self.x - prey_x).abs() <= 1 && (self.y - prey_y).abs() <= 1
    }

    fn hunt(&mut self, prey_x: i32, prey_y: i32, width: i32, height: i32) {
        let prediction = self.predict();
        let (dx, dy) = match prediction {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        let target_x = (prey_x + dx + width) % width;
        let target_y = (prey_y + dy + height) % height;

        self.x += (target_x - self.x).signum();
        self.y += (target_y - self.y).signum();
        self.x = (self.x + width) % width;
        self.y = (self.y + height) % height;
    }

    fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }
}

// ============================================================================
// Trial Results
// ============================================================================

#[derive(Debug)]
struct TrialResult {
    survival: u32,
    predator_accuracy: f64,
    decision_entropy: f64,
}

fn calculate_entropy(counts: &[u32]) -> f64 {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ============================================================================
// TEST 1: Quantum (actual implementation) vs Softmax
// ============================================================================

fn run_quantum_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> TrialResult {
    let config = QuantumSNNConfig {
        n_inputs: 8,
        hidden_layers: vec![32, 16],
        n_outputs: 5,
        interference_depth: 2, // Key: Quantum interference enabled
        ticks_per_decision: 10,
        learning_enabled: false,
        ..Default::default()
    };

    let mut qsnn = QuantumSNN::with_seed(config, seed);
    let mut predator = Predator::new(width - 5, height - 5);

    let (mut prey_x, mut prey_y) = (5, 5);
    let mut action_counts = [0u32; 5];

    for step in 0..max_steps {
        // Compute sensors
        let dx = predator.x - prey_x;
        let dy = predator.y - prey_y;

        let sensors = [
            if dy < 0 {
                ((-dy).min(10) * 25) as u8
            } else {
                0
            },
            if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 },
            if dx < 0 {
                ((-dx).min(10) * 25) as u8
            } else {
                0
            },
            if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 },
            ((height - prey_y).min(5) * 50) as u8,
            (prey_y.min(5) * 50) as u8,
            (prey_x.min(5) * 50) as u8,
            ((width - prey_x).min(5) * 50) as u8,
        ];

        qsnn.set_inputs(&sensors);
        let action = qsnn.decide(); // Uses quantum interference
        action_counts[action] += 1;

        // Apply move
        let (mx, my) = match action {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        prey_x = (prey_x + mx + width) % width;
        prey_y = (prey_y + my + height) % height;

        // Check catch
        if predator.check_catch_and_learn(prey_x, prey_y, action) {
            return TrialResult {
                survival: step,
                predator_accuracy: predator.accuracy(),
                decision_entropy: calculate_entropy(&action_counts),
            };
        }

        predator.observe(action);
        predator.hunt(prey_x, prey_y, width, height);
    }

    TrialResult {
        survival: max_steps,
        predator_accuracy: predator.accuracy(),
        decision_entropy: calculate_entropy(&action_counts),
    }
}

fn run_softmax_trial(
    seed: u64,
    temperature: f64,
    width: i32,
    height: i32,
    max_steps: u32,
) -> TrialResult {
    // Same SNN as quantum, but softmax decision instead
    let snn_config = SNNConfig {
        n_inputs: 8,
        hidden_layers: vec![32, 16],
        n_outputs: 5,
        connection_prob: 0.3, // Match QuantumSNN default
        neurons_per_cpu: 32,
        recurrent: false,
        neuron_config: NeuronConfig::default(),
        stdp_config: STDPConfig::default(),
        use_reward_modulation: false,
        duplicate_inputs: false,
    };

    let mut snn = SNNNetwork::new(snn_config);
    snn.build_connectivity(seed); // Same seed = same network structure

    let mut predator = Predator::new(width - 5, height - 5);
    let mut rng = seed.wrapping_add(99999);

    let (mut prey_x, mut prey_y) = (5, 5);
    let mut action_counts = [0u32; 5];

    for step in 0..max_steps {
        // Same sensors as quantum trial
        let dx = predator.x - prey_x;
        let dy = predator.y - prey_y;

        let sensors = [
            if dy < 0 {
                ((-dy).min(10) * 25) as u8
            } else {
                0
            },
            if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 },
            if dx < 0 {
                ((-dx).min(10) * 25) as u8
            } else {
                0
            },
            if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 },
            ((height - prey_y).min(5) * 50) as u8,
            (prey_y.min(5) * 50) as u8,
            (prey_x.min(5) * 50) as u8,
            ((width - prey_x).min(5) * 50) as u8,
        ];

        snn.set_inputs(&sensors);
        snn.reset_output_counts();
        for _ in 0..10 {
            // Same ticks_per_decision
            snn.step();
        }

        // Softmax decision instead of quantum
        let action = softmax_sample(&snn.output_counts, temperature, &mut rng);
        action_counts[action] += 1;

        // Apply move
        let (mx, my) = match action {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        prey_x = (prey_x + mx + width) % width;
        prey_y = (prey_y + my + height) % height;

        // Check catch
        if predator.check_catch_and_learn(prey_x, prey_y, action) {
            return TrialResult {
                survival: step,
                predator_accuracy: predator.accuracy(),
                decision_entropy: calculate_entropy(&action_counts),
            };
        }

        predator.observe(action);
        predator.hunt(prey_x, prey_y, width, height);
    }

    TrialResult {
        survival: max_steps,
        predator_accuracy: predator.accuracy(),
        decision_entropy: calculate_entropy(&action_counts),
    }
}

fn run_proportional_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> TrialResult {
    let snn_config = SNNConfig {
        n_inputs: 8,
        hidden_layers: vec![32, 16],
        n_outputs: 5,
        connection_prob: 0.3,
        neurons_per_cpu: 32,
        recurrent: false,
        neuron_config: NeuronConfig::default(),
        stdp_config: STDPConfig::default(),
        use_reward_modulation: false,
        duplicate_inputs: false,
    };

    let mut snn = SNNNetwork::new(snn_config);
    snn.build_connectivity(seed);

    let mut predator = Predator::new(width - 5, height - 5);
    let mut rng = seed.wrapping_add(99999);

    let (mut prey_x, mut prey_y) = (5, 5);
    let mut action_counts = [0u32; 5];

    for step in 0..max_steps {
        let dx = predator.x - prey_x;
        let dy = predator.y - prey_y;

        let sensors = [
            if dy < 0 {
                ((-dy).min(10) * 25) as u8
            } else {
                0
            },
            if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 },
            if dx < 0 {
                ((-dx).min(10) * 25) as u8
            } else {
                0
            },
            if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 },
            ((height - prey_y).min(5) * 50) as u8,
            (prey_y.min(5) * 50) as u8,
            (prey_x.min(5) * 50) as u8,
            ((width - prey_x).min(5) * 50) as u8,
        ];

        snn.set_inputs(&sensors);
        snn.reset_output_counts();
        for _ in 0..10 {
            snn.step();
        }

        // Proportional sampling (same as quantum without interference)
        let action = proportional_sample(&snn.output_counts, &mut rng);
        action_counts[action] += 1;

        let (mx, my) = match action {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        prey_x = (prey_x + mx + width) % width;
        prey_y = (prey_y + my + height) % height;

        if predator.check_catch_and_learn(prey_x, prey_y, action) {
            return TrialResult {
                survival: step,
                predator_accuracy: predator.accuracy(),
                decision_entropy: calculate_entropy(&action_counts),
            };
        }

        predator.observe(action);
        predator.hunt(prey_x, prey_y, width, height);
    }

    TrialResult {
        survival: max_steps,
        predator_accuracy: predator.accuracy(),
        decision_entropy: calculate_entropy(&action_counts),
    }
}

fn run_wta_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> TrialResult {
    let snn_config = SNNConfig {
        n_inputs: 8,
        hidden_layers: vec![32, 16],
        n_outputs: 5,
        connection_prob: 0.3,
        neurons_per_cpu: 32,
        recurrent: false,
        neuron_config: NeuronConfig::default(),
        stdp_config: STDPConfig::default(),
        use_reward_modulation: false,
        duplicate_inputs: false,
    };

    let mut snn = SNNNetwork::new(snn_config);
    snn.build_connectivity(seed);

    let mut predator = Predator::new(width - 5, height - 5);
    let mut rng = seed.wrapping_add(99999);

    let (mut prey_x, mut prey_y) = (5, 5);
    let mut action_counts = [0u32; 5];

    for step in 0..max_steps {
        let dx = predator.x - prey_x;
        let dy = predator.y - prey_y;

        let sensors = [
            if dy < 0 {
                ((-dy).min(10) * 25) as u8
            } else {
                0
            },
            if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 },
            if dx < 0 {
                ((-dx).min(10) * 25) as u8
            } else {
                0
            },
            if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 },
            ((height - prey_y).min(5) * 50) as u8,
            (prey_y.min(5) * 50) as u8,
            (prey_x.min(5) * 50) as u8,
            ((width - prey_x).min(5) * 50) as u8,
        ];

        snn.set_inputs(&sensors);
        snn.reset_output_counts();
        for _ in 0..10 {
            snn.step();
        }

        // Winner-take-all (deterministic)
        let action = wta_sample(&snn.output_counts, &mut rng);
        action_counts[action] += 1;

        let (mx, my) = match action {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        prey_x = (prey_x + mx + width) % width;
        prey_y = (prey_y + my + height) % height;

        if predator.check_catch_and_learn(prey_x, prey_y, action) {
            return TrialResult {
                survival: step,
                predator_accuracy: predator.accuracy(),
                decision_entropy: calculate_entropy(&action_counts),
            };
        }

        predator.observe(action);
        predator.hunt(prey_x, prey_y, width, height);
    }

    TrialResult {
        survival: max_steps,
        predator_accuracy: predator.accuracy(),
        decision_entropy: calculate_entropy(&action_counts),
    }
}

// ============================================================================
// Main A/B Test
// ============================================================================

fn main() {
    println!("================================================================");
    println!("  QUANTUM vs SOFTMAX A/B TEST (v2)");
    println!("  Using actual QuantumSNN implementation");
    println!("================================================================\n");

    let width = 30;
    let height = 30;
    let max_steps = 500;
    let n_trials = 200; // More trials for statistical power

    println!("Configuration:");
    println!("  Arena: {}x{}", width, height);
    println!("  Max steps: {}", max_steps);
    println!("  Trials per method: {}", n_trials);
    println!();

    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let start = Instant::now();

    // Collect results for each method
    let mut quantum_results: Vec<TrialResult> = Vec::new();
    let mut proportional_results: Vec<TrialResult> = Vec::new();
    let mut wta_results: Vec<TrialResult> = Vec::new();
    let mut softmax_1_results: Vec<TrialResult> = Vec::new();
    let mut softmax_2_results: Vec<TrialResult> = Vec::new();
    let mut softmax_5_results: Vec<TrialResult> = Vec::new();
    let mut softmax_10_results: Vec<TrialResult> = Vec::new();

    print!("Running trials");
    for trial in 0..n_trials {
        let seed = base_seed.wrapping_add(trial as u64 * 137);

        quantum_results.push(run_quantum_trial(seed, width, height, max_steps));
        proportional_results.push(run_proportional_trial(seed, width, height, max_steps));
        wta_results.push(run_wta_trial(seed, width, height, max_steps));
        softmax_1_results.push(run_softmax_trial(seed, 1.0, width, height, max_steps));
        softmax_2_results.push(run_softmax_trial(seed, 2.0, width, height, max_steps));
        softmax_5_results.push(run_softmax_trial(seed, 5.0, width, height, max_steps));
        softmax_10_results.push(run_softmax_trial(seed, 10.0, width, height, max_steps));

        if trial % 10 == 9 {
            print!(".");
        }
    }
    println!(" done!");

    let elapsed = start.elapsed();

    // Aggregate results
    let methods: Vec<(&str, &[TrialResult])> = vec![
        ("Quantum (H+CZ)", &quantum_results),
        ("Proportional", &proportional_results),
        ("WTA", &wta_results),
        ("Softmax T=1", &softmax_1_results),
        ("Softmax T=2", &softmax_2_results),
        ("Softmax T=5", &softmax_5_results),
        ("Softmax T=10", &softmax_10_results),
    ];

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>12}",
        "Method", "Survival", "Std Dev", "Entropy", "Pred Acc"
    );
    println!("{:-<64}", "");

    let mut stats: Vec<(&str, f64, f64, f64, f64)> = Vec::new();

    for (name, results) in &methods {
        let survivals: Vec<f64> = results.iter().map(|r| r.survival as f64).collect();
        let avg_survival = survivals.iter().sum::<f64>() / n_trials as f64;
        let variance = survivals
            .iter()
            .map(|&s| (s - avg_survival).powi(2))
            .sum::<f64>()
            / n_trials as f64;
        let std_dev = variance.sqrt();

        let avg_entropy = results.iter().map(|r| r.decision_entropy).sum::<f64>() / n_trials as f64;
        let avg_pred_acc =
            results.iter().map(|r| r.predator_accuracy).sum::<f64>() / n_trials as f64;

        println!(
            "{:<18} {:>10.1} {:>10.1} {:>10.3} {:>11.1}%",
            name,
            avg_survival,
            std_dev,
            avg_entropy,
            avg_pred_acc * 100.0
        );

        stats.push((name, avg_survival, std_dev, avg_entropy, avg_pred_acc));
    }

    // Analysis
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  COMPARISON                                                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let quantum = stats
        .iter()
        .find(|(n, _, _, _, _)| *n == "Quantum (H+CZ)")
        .unwrap();
    let proportional = stats
        .iter()
        .find(|(n, _, _, _, _)| *n == "Proportional")
        .unwrap();
    let best_softmax = stats
        .iter()
        .filter(|(n, _, _, _, _)| n.starts_with("Softmax"))
        .max_by(|(_, a, _, _, _), (_, b, _, _, _)| a.partial_cmp(b).unwrap())
        .unwrap();
    let wta = stats.iter().find(|(n, _, _, _, _)| *n == "WTA").unwrap();

    println!("QUANTUM vs PROPORTIONAL (isolates interference effect):");
    let q_vs_p = quantum.1 - proportional.1;
    println!("  Quantum:      {:.1} steps", quantum.1);
    println!("  Proportional: {:.1} steps", proportional.1);
    println!("  Difference:   {:+.1} steps", q_vs_p);
    if q_vs_p > 10.0 {
        println!("  → Interference provides significant benefit");
    } else if q_vs_p > 2.0 {
        println!("  → Interference provides modest benefit");
    } else if q_vs_p > -2.0 {
        println!("  → No meaningful difference (interference has no effect)");
    } else {
        println!("  → Proportional actually outperforms quantum");
    }

    println!("\nQUANTUM vs BEST SOFTMAX ({}):", best_softmax.0);
    let q_vs_s = quantum.1 - best_softmax.1;
    println!("  Quantum:  {:.1} steps", quantum.1);
    println!("  Softmax:  {:.1} steps", best_softmax.1);
    println!("  Difference: {:+.1} steps", q_vs_s);

    // Effect size
    let pooled_std = ((quantum.2.powi(2) + best_softmax.2.powi(2)) / 2.0).sqrt();
    let effect_size = if pooled_std > 0.0 {
        q_vs_s / pooled_std
    } else {
        0.0
    };
    println!("  Cohen's d: {:.2}", effect_size);

    println!("\nQUANTUM vs WTA (deterministic baseline):");
    let q_vs_wta = quantum.1 - wta.1;
    println!("  Quantum: {:.1} steps", quantum.1);
    println!("  WTA:     {:.1} steps", wta.1);
    println!("  Difference: {:+.1} steps", q_vs_wta);

    // Predator accuracy comparison (lower = harder to predict)
    println!("\nPREDATABILITY (lower predator accuracy = better):");
    let mut by_pred: Vec<_> = stats.iter().collect();
    by_pred.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
    for (i, (name, _, _, _, pred_acc)) in by_pred.iter().enumerate() {
        let marker = if i == 0 { "★" } else { " " };
        println!(
            "  {} {:<18} Predator accuracy: {:.1}%",
            marker,
            name,
            pred_acc * 100.0
        );
    }

    // Final verdict
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL VERDICT                                                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut by_survival: Vec<_> = stats.iter().collect();
    by_survival.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("RANKING BY SURVIVAL:");
    for (i, (name, surv, std, _, _)) in by_survival.iter().enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!("  {} {:<18} {:.1} ± {:.1}", medal, name, surv, std);
    }

    let winner = by_survival[0].0;

    println!();
    if winner == "Quantum (H+CZ)" {
        if effect_size > 0.5 {
            println!("CONCLUSION: Quantum interference provides GENUINE advantage.");
            println!(
                "            Effect size {:.2} indicates meaningful improvement",
                effect_size
            );
            println!("            over best classical alternative.");
        } else if effect_size > 0.2 {
            println!("CONCLUSION: Quantum provides SMALL advantage.");
            println!(
                "            Effect size {:.2} suggests marginal improvement.",
                effect_size
            );
        } else {
            println!("CONCLUSION: Quantum wins but by NEGLIGIBLE margin.");
            println!(
                "            Effect size {:.2} - could be within noise.",
                effect_size
            );
        }
    } else if winner.starts_with("Softmax") {
        println!("CONCLUSION: Classical softmax OUTPERFORMS quantum.");
        println!(
            "            Temperature {} achieves better survival.",
            winner
        );
        println!("            Quantum interference provides NO advantage in this task.");
    } else if winner == "Proportional" {
        println!("CONCLUSION: Simple proportional sampling wins.");
        println!("            Both quantum and softmax add unnecessary complexity.");
    } else {
        println!("CONCLUSION: WTA (deterministic) wins.");
        println!("            Stochastic decision-making doesn't help here.");
    }

    println!("\n================================================================");
    println!(
        "  Test completed in {:.2}s ({} trials × 7 methods)",
        elapsed.as_secs_f64(),
        n_trials
    );
    println!("================================================================");
}
