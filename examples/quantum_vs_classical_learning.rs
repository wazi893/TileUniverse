//! Rigorous Quantum vs Classical Learning Speed Comparison
//!
//! This experiment compares learning speed between:
//! - Quantum: Uses probabilistic measurement for action selection (exploration)
//! - Classical: Uses winner-take-all for action selection (exploitation)
//!
//! Methodology:
//! - Same SNN architecture and initial weights for both
//! - Same STDP learning parameters
//! - Multiple random seeds for statistical significance
//! - Track accuracy over epochs, report mean ± std
//!
//! Run with: cargo run --release --example quantum_vs_classical_learning

use engine::snn::quantum_hybrid::{InterferenceMode, QuantumSNN, QuantumSNNConfig};
use std::time::Instant;

/// Decision mode for action selection
#[derive(Clone, Copy, Debug)]
enum DecisionMode {
    /// SoftMix: gentle blending that preserves spike info
    SoftMix,
    /// Grover: amplitude amplification of dominant action
    Grover,
    /// No interference: pure spike-proportional sampling
    NoInterference,
    /// Classical: winner-take-all (deterministic)
    Classical,
    /// HadamardCZ: original broken circuit (for comparison)
    HadamardCZ,
    /// Hybrid: train with classical WTA, deploy with Grover quantum
    HybridGrover,
    /// Hybrid: train with classical WTA, deploy with spike-proportional
    HybridProportional,
}

/// Results from a single learning run
#[derive(Clone, Debug)]
struct LearningRun {
    seed: u64,
    accuracies: Vec<f64>, // Accuracy at each checkpoint
    final_accuracy: f64,
    epochs_to_70: Option<usize>, // Epochs to reach 70% accuracy
    epochs_to_80: Option<usize>, // Epochs to reach 80% accuracy
}

/// Aggregated results across multiple runs
#[derive(Debug)]
struct AggregatedResults {
    mean_accuracies: Vec<f64>,
    std_accuracies: Vec<f64>,
    mean_final: f64,
    std_final: f64,
    mean_epochs_to_70: Option<f64>,
    mean_epochs_to_80: Option<f64>,
    success_rate_70: f64, // % of runs that reached 70%
    success_rate_80: f64, // % of runs that reached 80%
}

fn main() {
    println!("================================================================");
    println!("  RIGOROUS QUANTUM VS CLASSICAL LEARNING COMPARISON");
    println!("================================================================\n");

    // Experiment parameters
    let n_seeds = 10; // Number of random seeds for statistical significance
    let n_epochs = 1000;
    let checkpoint_interval = 100;
    let trials_per_epoch = 20;
    let eval_trials = 50;

    println!("Experiment Configuration:");
    println!("  - Random seeds: {}", n_seeds);
    println!("  - Epochs per run: {}", n_epochs);
    println!("  - Checkpoint interval: {} epochs", checkpoint_interval);
    println!("  - Training trials per epoch: {}", trials_per_epoch);
    println!("  - Evaluation trials: {}", eval_trials);
    println!();

    // Task: 4-input pattern classification (2 classes)
    // Pattern A: inputs 0,2 active → Class 0
    // Pattern B: inputs 1,3 active → Class 1
    let pattern_a = [255u8, 0, 255, 0];
    let pattern_b = [0u8, 255, 0, 255];

    println!("Task: Binary Pattern Classification");
    println!("  Pattern A [255,0,255,0] → Class 0");
    println!("  Pattern B [0,255,0,255] → Class 1");
    println!();

    // First, diagnose what quantum is doing
    println!("--- Diagnostic: What are the quantum probabilities? ---\n");
    diagnose_quantum_probabilities(&pattern_a, &pattern_b);

    // Run experiments
    println!("Running {} experiments per group...\n", n_seeds);

    // Test 1: Grover (amplitude amplification)
    let grover_results = run_experiment_group(
        "Grover",
        n_seeds,
        n_epochs,
        checkpoint_interval,
        trials_per_epoch,
        eval_trials,
        &pattern_a,
        &pattern_b,
        DecisionMode::Grover,
    );

    // Test 3: No interference (pure spike-proportional)
    let none_results = run_experiment_group(
        "No Interf.",
        n_seeds,
        n_epochs,
        checkpoint_interval,
        trials_per_epoch,
        eval_trials,
        &pattern_a,
        &pattern_b,
        DecisionMode::NoInterference,
    );

    // Test 4: Classical WTA
    let classical_results = run_experiment_group(
        "Classical",
        n_seeds,
        n_epochs,
        checkpoint_interval,
        trials_per_epoch,
        eval_trials,
        &pattern_a,
        &pattern_b,
        DecisionMode::Classical,
    );

    // Test 5: Hybrid Grover (train classical, deploy quantum Grover)
    let hybrid_grover_results = run_experiment_group(
        "Hybrid-Grover",
        n_seeds,
        n_epochs,
        checkpoint_interval,
        trials_per_epoch,
        eval_trials,
        &pattern_a,
        &pattern_b,
        DecisionMode::HybridGrover,
    );

    // Test 6: Hybrid Proportional (train classical, deploy spike-proportional)
    let hybrid_prop_results = run_experiment_group(
        "Hybrid-Prop",
        n_seeds,
        n_epochs,
        checkpoint_interval,
        trials_per_epoch,
        eval_trials,
        &pattern_a,
        &pattern_b,
        DecisionMode::HybridProportional,
    );

    // Print comparison
    print_six_way_comparison(
        &grover_results,
        &none_results,
        &classical_results,
        &hybrid_grover_results,
        &hybrid_prop_results,
        checkpoint_interval,
    );
}

/// Diagnose what each interference mode does to the probabilities
fn diagnose_quantum_probabilities(pattern_a: &[u8; 4], pattern_b: &[u8; 4]) {
    let modes = [
        ("None (spike-proportional)", InterferenceMode::None),
        ("SoftMix (gentle blend)", InterferenceMode::SoftMix),
        ("Grover (amplification)", InterferenceMode::Grover),
        ("HadamardCZ (original)", InterferenceMode::HadamardCZ),
    ];

    for (name, mode) in &modes {
        let config = QuantumSNNConfig {
            n_inputs: 4,
            hidden_layers: vec![16],
            n_outputs: 2,
            interference_mode: *mode,
            interference_depth: 1,
            mix_angle: 0.15,
            ticks_per_decision: 30,
            learning_enabled: false,
            ..Default::default()
        };

        let mut qsnn = QuantumSNN::with_seed(config, 42);

        // Check pattern A
        qsnn.set_inputs(pattern_a);
        qsnn.decide();
        let probs_a = qsnn.last_probabilities().to_vec();
        let spikes_a = qsnn.output_spike_counts().to_vec();

        // Check pattern B
        qsnn.set_inputs(pattern_b);
        qsnn.decide();
        let probs_b = qsnn.last_probabilities().to_vec();
        let spikes_b = qsnn.output_spike_counts().to_vec();

        println!(
            "{:20} Pattern A: spikes={:?} → probs={:.2?}",
            name, spikes_a, probs_a
        );
        println!(
            "{:20} Pattern B: spikes={:?} → probs={:.2?}",
            name, spikes_b, probs_b
        );
        println!();
    }
}

fn print_six_way_comparison(
    grover: &AggregatedResults,
    none: &AggregatedResults,
    classical: &AggregatedResults,
    hybrid_grover: &AggregatedResults,
    hybrid_prop: &AggregatedResults,
    interval: usize,
) {
    println!("\n================================================================");
    println!("  COMPARISON: PURE vs HYBRID MODES");
    println!("================================================================\n");

    println!("Training strategy:");
    println!("  - Pure modes: Train AND evaluate with same method");
    println!("  - Hybrid modes: Train with Classical WTA, evaluate with Quantum\n");

    println!("LEARNING CURVES (mean accuracy):\n");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Epoch", "Grover", "NoInterf", "Classical", "Hyb-Grover", "Hyb-Prop"
    );
    println!("{:-<78}", "");

    for i in 0..grover.mean_accuracies.len() {
        let epoch = (i + 1) * interval;
        println!(
            "{:>8} {:>9.1}% {:>9.1}% {:>9.1}% {:>11.1}% {:>11.1}%",
            epoch,
            grover.mean_accuracies[i],
            none.mean_accuracies[i],
            classical.mean_accuracies[i],
            hybrid_grover.mean_accuracies[i],
            hybrid_prop.mean_accuracies[i],
        );
    }

    println!("\nFINAL RESULTS:\n");
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "", "Grover", "NoInterf", "Classical", "Hyb-Grover", "Hyb-Prop"
    );
    println!("{:-<90}", "");
    println!(
        "{:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>11.1}% {:>11.1}%",
        "Mean Final Accuracy",
        grover.mean_final,
        none.mean_final,
        classical.mean_final,
        hybrid_grover.mean_final,
        hybrid_prop.mean_final,
    );
    println!(
        "{:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>11.1}% {:>11.1}%",
        "Success Rate (≥70%)",
        grover.success_rate_70,
        none.success_rate_70,
        classical.success_rate_70,
        hybrid_grover.success_rate_70,
        hybrid_prop.success_rate_70,
    );
    println!(
        "{:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>11.1}% {:>11.1}%",
        "Success Rate (≥80%)",
        grover.success_rate_80,
        none.success_rate_80,
        classical.success_rate_80,
        hybrid_grover.success_rate_80,
        hybrid_prop.success_rate_80,
    );

    // Find best mode
    let results = [
        ("Grover (pure)", grover.mean_final),
        ("NoInterference (pure)", none.mean_final),
        ("Classical", classical.mean_final),
        ("Hybrid-Grover", hybrid_grover.mean_final),
        ("Hybrid-Proportional", hybrid_prop.mean_final),
    ];
    let best = results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\nBEST MODE: {} with {:.1}% accuracy", best.0, best.1);

    // Summary analysis
    println!("\nANALYSIS:");
    if hybrid_grover.mean_final > grover.mean_final {
        println!(
            "  ✓ Hybrid-Grover ({:.1}%) beats pure Grover ({:.1}%) - classical training helps!",
            hybrid_grover.mean_final, grover.mean_final
        );
    }
    if hybrid_prop.mean_final > none.mean_final {
        println!(
            "  ✓ Hybrid-Prop ({:.1}%) beats pure NoInterf ({:.1}%) - classical training helps!",
            hybrid_prop.mean_final, none.mean_final
        );
    }
    if hybrid_grover.mean_final > classical.mean_final {
        println!(
            "  ★ Hybrid-Grover beats Classical! Quantum exploration + classical learning works!"
        );
    } else if (classical.mean_final - hybrid_grover.mean_final).abs() < 5.0 {
        println!(
            "  ≈ Hybrid-Grover nearly matches Classical - quantum adds exploration at low cost"
        );
    }
}

fn run_experiment_group(
    name: &str,
    n_seeds: usize,
    n_epochs: usize,
    checkpoint_interval: usize,
    trials_per_epoch: usize,
    eval_trials: usize,
    pattern_a: &[u8; 4],
    pattern_b: &[u8; 4],
    mode: DecisionMode,
) -> AggregatedResults {
    println!("--- {} Group ---", name);

    let mut all_runs: Vec<LearningRun> = Vec::new();
    let start = Instant::now();

    for seed_idx in 0..n_seeds {
        let seed = 1000 + seed_idx as u64 * 7919; // Prime-spaced seeds

        let run = run_single_experiment(
            seed,
            n_epochs,
            checkpoint_interval,
            trials_per_epoch,
            eval_trials,
            pattern_a,
            pattern_b,
            mode,
        );

        print!("  Seed {:>5}: final={:>5.1}%", seed, run.final_accuracy);
        if let Some(e70) = run.epochs_to_70 {
            print!("  @70%={:>4}", e70);
        } else {
            print!("  @70%=  - ");
        }
        if let Some(e80) = run.epochs_to_80 {
            print!("  @80%={:>4}", e80);
        } else {
            print!("  @80%=  - ");
        }
        println!();

        all_runs.push(run);
    }

    let elapsed = start.elapsed();
    println!("  Completed in {:.2}s\n", elapsed.as_secs_f64());

    aggregate_results(&all_runs)
}

fn run_single_experiment(
    seed: u64,
    n_epochs: usize,
    checkpoint_interval: usize,
    trials_per_epoch: usize,
    eval_trials: usize,
    pattern_a: &[u8; 4],
    pattern_b: &[u8; 4],
    mode: DecisionMode,
) -> LearningRun {
    // Create network with specific seed
    // For hybrid modes, we use Grover/None for deployment but train with classical
    let (interference_mode, mix_angle) = match mode {
        DecisionMode::SoftMix => (InterferenceMode::SoftMix, 0.15),
        DecisionMode::Grover | DecisionMode::HybridGrover => (InterferenceMode::Grover, 0.0),
        DecisionMode::NoInterference | DecisionMode::HybridProportional => {
            (InterferenceMode::None, 0.0)
        }
        DecisionMode::Classical => (InterferenceMode::None, 0.0),
        DecisionMode::HadamardCZ => (InterferenceMode::HadamardCZ, 0.0),
    };

    let config = QuantumSNNConfig {
        n_inputs: 4,
        hidden_layers: vec![16],
        n_outputs: 2,
        interference_mode,
        interference_depth: 1,
        mix_angle,
        ticks_per_decision: 30,
        learning_enabled: true,
        ..Default::default()
    };

    let mut qsnn = QuantumSNN::with_seed(config, seed);

    let mut accuracies = Vec::new();
    let mut epochs_to_70 = None;
    let mut epochs_to_80 = None;

    // Debug: track how often rewards differ
    let mut action_differences = 0usize;
    let mut reward_sum = 0i64;
    let mut total_trials = 0usize;

    for epoch in 0..n_epochs {
        // Training phase
        for trial in 0..trials_per_epoch {
            let (pattern, target) = if trial % 2 == 0 {
                (pattern_a, 0usize)
            } else {
                (pattern_b, 1usize)
            };

            qsnn.set_inputs(pattern);

            // Run SNN to get spike counts
            qsnn.decide();

            let quantum_action = qsnn.last_action();
            let classical_action = qsnn.classical_action();

            // Track differences
            if quantum_action != classical_action {
                action_differences += 1;
            }
            total_trials += 1;

            // For training: determine reward based on mode
            let training_action = match mode {
                DecisionMode::HybridGrover
                | DecisionMode::HybridProportional
                | DecisionMode::Classical => {
                    classical_action // Use WTA for training reward
                }
                _ => {
                    quantum_action // Use quantum-sampled action
                }
            };

            // R-STDP learning based on training action
            let reward = if training_action == target {
                127i8
            } else {
                -100i8
            };
            reward_sum += reward as i64;
            qsnn.learn(reward);
        }

        // Evaluation phase (at checkpoints)
        if epoch % checkpoint_interval == checkpoint_interval - 1 {
            // Evaluate with same mode as training for consistency
            let accuracy = evaluate(&mut qsnn, pattern_a, pattern_b, eval_trials, mode);
            accuracies.push(accuracy);

            // Track when thresholds are reached
            if epochs_to_70.is_none() && accuracy >= 70.0 {
                epochs_to_70 = Some(epoch + 1);
            }
            if epochs_to_80.is_none() && accuracy >= 80.0 {
                epochs_to_80 = Some(epoch + 1);
            }
        }
    }

    let final_accuracy = *accuracies.last().unwrap_or(&50.0);

    // Print stats for first seed only
    if seed == 1000 {
        let diff_pct = 100.0 * action_differences as f64 / total_trials as f64;
        let avg_reward = reward_sum as f64 / total_trials as f64;

        // For hybrid modes, show both classical (what learned) and quantum (deployment) accuracy
        if matches!(
            mode,
            DecisionMode::HybridGrover | DecisionMode::HybridProportional
        ) {
            let classical_acc = evaluate(
                &mut qsnn,
                pattern_a,
                pattern_b,
                100,
                DecisionMode::Classical,
            );
            let quantum_acc = evaluate(&mut qsnn, pattern_a, pattern_b, 100, mode);
            println!(
                "    [{:?}] learned={:.0}%, deploy={:.0}% (Q!=C:{:.0}%)",
                mode, classical_acc, quantum_acc, diff_pct
            );
        } else {
            println!(
                "    [{:?}] Q!=C: {:.1}%, avg_reward: {:.1}",
                mode, diff_pct, avg_reward
            );
        }
    }

    LearningRun {
        seed,
        accuracies,
        final_accuracy,
        epochs_to_70,
        epochs_to_80,
    }
}

fn evaluate(
    qsnn: &mut QuantumSNN,
    pattern_a: &[u8; 4],
    pattern_b: &[u8; 4],
    n_trials: usize,
    mode: DecisionMode,
) -> f64 {
    let mut correct = 0;

    for trial in 0..n_trials {
        let (pattern, target) = if trial % 2 == 0 {
            (pattern_a, 0usize)
        } else {
            (pattern_b, 1usize)
        };

        qsnn.set_inputs(pattern);

        // For evaluation: Classical uses WTA, everything else uses quantum sampling
        qsnn.decide();
        let action = match mode {
            DecisionMode::Classical => qsnn.classical_action(),
            _ => qsnn.last_action(), // Quantum-sampled (includes hybrid modes)
        };

        if action == target {
            correct += 1;
        }
    }

    100.0 * correct as f64 / n_trials as f64
}

fn aggregate_results(runs: &[LearningRun]) -> AggregatedResults {
    let n = runs.len() as f64;

    // Aggregate accuracies at each checkpoint
    let n_checkpoints = runs[0].accuracies.len();
    let mut mean_accuracies = vec![0.0; n_checkpoints];
    let mut std_accuracies = vec![0.0; n_checkpoints];

    for i in 0..n_checkpoints {
        let values: Vec<f64> = runs.iter().map(|r| r.accuracies[i]).collect();
        mean_accuracies[i] = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|v| (v - mean_accuracies[i]).powi(2))
            .sum::<f64>()
            / n;
        std_accuracies[i] = variance.sqrt();
    }

    // Final accuracy stats
    let final_values: Vec<f64> = runs.iter().map(|r| r.final_accuracy).collect();
    let mean_final = final_values.iter().sum::<f64>() / n;
    let variance = final_values
        .iter()
        .map(|v| (v - mean_final).powi(2))
        .sum::<f64>()
        / n;
    let std_final = variance.sqrt();

    // Epochs to threshold stats
    let epochs_70: Vec<f64> = runs
        .iter()
        .filter_map(|r| r.epochs_to_70.map(|e| e as f64))
        .collect();
    let epochs_80: Vec<f64> = runs
        .iter()
        .filter_map(|r| r.epochs_to_80.map(|e| e as f64))
        .collect();

    let mean_epochs_to_70 = if epochs_70.is_empty() {
        None
    } else {
        Some(epochs_70.iter().sum::<f64>() / epochs_70.len() as f64)
    };

    let mean_epochs_to_80 = if epochs_80.is_empty() {
        None
    } else {
        Some(epochs_80.iter().sum::<f64>() / epochs_80.len() as f64)
    };

    let success_rate_70 = 100.0 * epochs_70.len() as f64 / n;
    let success_rate_80 = 100.0 * epochs_80.len() as f64 / n;

    AggregatedResults {
        mean_accuracies,
        std_accuracies,
        mean_final,
        std_final,
        mean_epochs_to_70,
        mean_epochs_to_80,
        success_rate_70,
        success_rate_80,
    }
}

fn print_comparison(quantum: &AggregatedResults, classical: &AggregatedResults, interval: usize) {
    println!("================================================================");
    println!("  RESULTS");
    println!("================================================================\n");

    // Learning curves
    println!("LEARNING CURVES (mean ± std):\n");
    println!("{:>8} {:>20} {:>20}", "Epoch", "Quantum", "Classical");
    println!("{:-<52}", "");

    for (i, (q_mean, q_std, c_mean, c_std)) in quantum
        .mean_accuracies
        .iter()
        .zip(quantum.std_accuracies.iter())
        .zip(classical.mean_accuracies.iter())
        .zip(classical.std_accuracies.iter())
        .map(|(((a, b), c), d)| (a, b, c, d))
        .enumerate()
    {
        let epoch = (i + 1) * interval;
        println!(
            "{:>8} {:>8.1}% ± {:>5.1}% {:>8.1}% ± {:>5.1}%",
            epoch, q_mean, q_std, c_mean, c_std
        );
    }

    // Summary statistics
    println!("\nSUMMARY STATISTICS:\n");
    println!("{:<25} {:>15} {:>15}", "", "Quantum", "Classical");
    println!("{:-<57}", "");

    println!(
        "{:<25} {:>8.1}% ± {:>4.1}% {:>8.1}% ± {:>4.1}%",
        "Final Accuracy",
        quantum.mean_final,
        quantum.std_final,
        classical.mean_final,
        classical.std_final
    );

    println!(
        "{:<25} {:>15.0}% {:>15.0}%",
        "Success Rate (≥70%)", quantum.success_rate_70, classical.success_rate_70
    );

    println!(
        "{:<25} {:>15.0}% {:>15.0}%",
        "Success Rate (≥80%)", quantum.success_rate_80, classical.success_rate_80
    );

    if let (Some(q), Some(c)) = (quantum.mean_epochs_to_70, classical.mean_epochs_to_70) {
        println!("{:<25} {:>15.0} {:>15.0}", "Mean Epochs to 70%", q, c);
    }

    if let (Some(q), Some(c)) = (quantum.mean_epochs_to_80, classical.mean_epochs_to_80) {
        println!("{:<25} {:>15.0} {:>15.0}", "Mean Epochs to 80%", q, c);
    }

    // Statistical significance (simple t-test approximation)
    println!("\nSTATISTICAL ANALYSIS:\n");

    let diff = quantum.mean_final - classical.mean_final;
    let pooled_std = ((quantum.std_final.powi(2) + classical.std_final.powi(2)) / 2.0).sqrt();
    let effect_size = if pooled_std > 0.0 {
        diff / pooled_std
    } else {
        0.0
    };

    println!("Difference (Q - C): {:+.1}%", diff);
    println!("Effect size (Cohen's d): {:.2}", effect_size);

    let interpretation = if effect_size.abs() < 0.2 {
        "negligible"
    } else if effect_size.abs() < 0.5 {
        "small"
    } else if effect_size.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    println!("Effect interpretation: {}", interpretation);

    // Conclusion
    println!("\nCONCLUSION:\n");

    if diff > 2.0 && quantum.success_rate_70 > classical.success_rate_70 {
        println!("✓ Quantum decision-making shows FASTER learning:");
        println!(
            "  - Higher final accuracy ({:.1}% vs {:.1}%)",
            quantum.mean_final, classical.mean_final
        );
        println!(
            "  - Higher success rate reaching 70% ({:.0}% vs {:.0}%)",
            quantum.success_rate_70, classical.success_rate_70
        );
    } else if diff < -2.0 && classical.success_rate_70 > quantum.success_rate_70 {
        println!("✗ Classical decision-making shows FASTER learning:");
        println!(
            "  - Higher final accuracy ({:.1}% vs {:.1}%)",
            classical.mean_final, quantum.mean_final
        );
        println!(
            "  - Higher success rate reaching 70% ({:.0}% vs {:.0}%)",
            classical.success_rate_70, quantum.success_rate_70
        );
    } else {
        println!("≈ No significant difference in learning speed:");
        println!(
            "  - Similar final accuracies ({:.1}% vs {:.1}%)",
            quantum.mean_final, classical.mean_final
        );
        println!(
            "  - Effect size is {} (d={:.2})",
            interpretation, effect_size
        );
    }
}
