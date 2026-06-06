//! Hybrid Quantum-Classical RL with Sparse Rewards
//!
//! Tests whether quantum exploration helps discover sparse rewards that
//! greedy classical policies might miss.
//!
//! Task: "Hidden Treasure" - 4 actions, only ONE gives reward per context
//! - Context A: Only action 2 gives +100, others give 0
//! - Context B: Only action 0 gives +100, others give 0
//!
//! Challenge: Initial random weights make network prefer wrong actions.
//! Exploration is needed to discover the rewarding actions.
//!
//! Run with: cargo run --release --example hybrid_sparse_rl

use engine::snn::quantum_hybrid::{InterferenceMode, QuantumSNN, QuantumSNNConfig};
use std::collections::HashMap;

/// Training mode
#[derive(Clone, Copy, Debug, PartialEq)]
enum TrainMode {
    /// Classical: train and act with WTA (greedy)
    Classical,
    /// Pure Quantum (Grover): train and act with Grover amplification
    PureQuantum,
    /// Hybrid Grover: train with classical, act with Grover
    Hybrid,
    /// Anti-Grover: curiosity-driven, amplifies LEAST preferred actions
    AntiGrover,
    /// Hybrid Anti-Grover: train classical, act with anti-grover
    HybridAntiGrover,
    /// Adaptive: starts anti-grover, transitions to grover
    Adaptive,
    /// Triggered: anti-grover until high reward, then grover
    Triggered,
    /// Hybrid Triggered: train classical, triggered exploration→exploitation
    HybridTriggered,
    /// Epsilon-greedy: classical with 30% random exploration
    EpsilonGreedy,
}

fn main() {
    println!("================================================================");
    println!("  SPARSE REWARD RL: QUANTUM EXPLORATION TEST");
    println!("================================================================\n");

    // Test 1: Pure sparse rewards
    test_sparse_rewards();

    // Test 2: Simple 2-action distractor escape
    test_two_action_distractor();
}

fn test_sparse_rewards() {
    println!("=== TEST 1: PURE SPARSE REWARDS ===\n");

    println!("Task: Hidden Treasure (4 actions)");
    println!("  - Context A [255,0,0,0]: Only action 2 gives +100");
    println!("  - Context B [0,255,0,0]: Only action 0 gives +100");
    println!("  - All other actions give 0 (sparse rewards!)\n");

    println!("Challenge: Random initial weights → network prefers wrong actions");
    println!("Goal: Discover the hidden rewarding actions through exploration\n");

    let n_seeds = 10;
    let n_episodes = 500;
    let trials_per_episode = 20;

    println!(
        "Running {} seeds, {} episodes each...\n",
        n_seeds, n_episodes
    );

    // Test all modes including Triggered
    let modes = [
        TrainMode::Classical,
        TrainMode::PureQuantum,
        TrainMode::Hybrid,
        TrainMode::Triggered,
        TrainMode::HybridTriggered,
        TrainMode::EpsilonGreedy,
    ];

    let mut all_results: HashMap<&str, Vec<f64>> = HashMap::new();

    for mode in &modes {
        let name = match mode {
            TrainMode::Classical => "Classical (greedy)",
            TrainMode::PureQuantum => "Pure Quantum",
            TrainMode::Hybrid => "Hybrid (train classical, act quantum)",
            TrainMode::AntiGrover => "AntiGrover (curiosity)",
            TrainMode::HybridAntiGrover => "Hybrid AntiGrover",
            TrainMode::Adaptive => "Adaptive (time-based)",
            TrainMode::Triggered => "Triggered (reward-based)",
            TrainMode::HybridTriggered => "Hybrid Triggered",
            TrainMode::EpsilonGreedy => "Epsilon-Greedy (ε=0.3)",
        };

        let mut final_rewards: Vec<f64> = Vec::new();
        let mut discovery_episodes: Vec<Option<usize>> = Vec::new();

        for seed_idx in 0..n_seeds {
            let seed = 1000 + seed_idx as u64 * 7919;
            let (final_reward, discovery_ep) =
                run_sparse_rl(seed, n_episodes, trials_per_episode, *mode);
            final_rewards.push(final_reward);
            discovery_episodes.push(discovery_ep);
        }

        let mean_reward: f64 = final_rewards.iter().sum::<f64>() / n_seeds as f64;
        let discovery_count = discovery_episodes.iter().filter(|d| d.is_some()).count();
        let mean_discovery: f64 = discovery_episodes
            .iter()
            .filter_map(|d| d.map(|e| e as f64))
            .sum::<f64>()
            / discovery_count.max(1) as f64;

        println!("--- {} ---", name);
        println!("  Mean final reward:    {:.1}/episode", mean_reward);
        println!(
            "  Discovery rate:       {}/{}  ({:.0}%)",
            discovery_count,
            n_seeds,
            100.0 * discovery_count as f64 / n_seeds as f64
        );
        if discovery_count > 0 {
            println!("  Mean discovery episode: {:.0}", mean_discovery);
        }
        println!();

        all_results.insert(name, final_rewards);
    }

    // Print comparison table
    println!("================================================================");
    println!("  RESULTS SUMMARY");
    println!("================================================================\n");

    println!("{:<35} {:>12} {:>12}", "Mode", "Mean Reward", "Std Dev");
    println!("{:-<60}", "");

    for mode in &modes {
        let name = match mode {
            TrainMode::Classical => "Classical (greedy)",
            TrainMode::PureQuantum => "Pure Quantum",
            TrainMode::Hybrid => "Hybrid (train classical, act quantum)",
            TrainMode::AntiGrover => "AntiGrover (curiosity)",
            TrainMode::HybridAntiGrover => "Hybrid AntiGrover",
            TrainMode::Adaptive => "Adaptive (time-based)",
            TrainMode::Triggered => "Triggered (reward-based)",
            TrainMode::HybridTriggered => "Hybrid Triggered",
            TrainMode::EpsilonGreedy => "Epsilon-Greedy (ε=0.3)",
        };

        if let Some(rewards) = all_results.get(name) {
            let mean: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
            let variance: f64 =
                rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
            let std_dev = variance.sqrt();
            println!("{:<35} {:>12.1} {:>12.1}", name, mean, std_dev);
        }
    }

    println!("\nANALYSIS:");

    let hybrid_mean: f64 = all_results
        .get("Hybrid (train classical, act quantum)")
        .map(|r| r.iter().sum::<f64>() / r.len() as f64)
        .unwrap_or(0.0);
    let classical_mean: f64 = all_results
        .get("Classical (greedy)")
        .map(|r| r.iter().sum::<f64>() / r.len() as f64)
        .unwrap_or(0.0);
    let epsilon_mean: f64 = all_results
        .get("Epsilon-Greedy (ε=0.3)")
        .map(|r| r.iter().sum::<f64>() / r.len() as f64)
        .unwrap_or(0.0);

    if hybrid_mean > classical_mean * 1.1 {
        println!(
            "  ✓ Hybrid outperforms Classical by {:.0}%",
            100.0 * (hybrid_mean - classical_mean) / classical_mean.max(1.0)
        );
        println!("    → Quantum exploration helps discover sparse rewards!");
    }
    if hybrid_mean > epsilon_mean {
        println!("  ✓ Hybrid beats Epsilon-Greedy - structured exploration > random");
    }
    println!();
}

/// Test 2: Simple 2-action distractor escape
/// Clear test of whether exploration helps escape suboptimal local optima
fn test_two_action_distractor() {
    println!("\n=== TEST 2: TWO-ACTION DISTRACTOR ESCAPE ===\n");

    println!("Task: 2 actions, initial bias towards distractor");
    println!("  - Action 0: +20 (distractor - network initially prefers this)");
    println!("  - Action 1: +100 (optimal - requires exploration to discover)");
    println!();

    println!("Modes:");
    println!("  - Grover: amplifies current best (exploitation-biased)");
    println!("  - AntiGrover: amplifies current WORST (curiosity-driven)");
    println!("  - Adaptive: AntiGrover early → Grover late");
    println!();

    let n_seeds = 10;
    let n_episodes = 1000; // More episodes to see convergence
    let trials_per_episode = 20;

    println!(
        "Running {} seeds, {} episodes each...\n",
        n_seeds, n_episodes
    );

    let modes = [
        TrainMode::Classical,
        TrainMode::PureQuantum,     // Grover
        TrainMode::AntiGrover,      // Curiosity
        TrainMode::Triggered,       // Anti-grover → grover on reward
        TrainMode::HybridTriggered, // Train classical + triggered exploration
        TrainMode::Adaptive,        // Time-based explore → exploit
        TrainMode::EpsilonGreedy,
    ];

    for mode in &modes {
        let name = match mode {
            TrainMode::Classical => "Classical (greedy)",
            TrainMode::PureQuantum => "Grover (exploitation)",
            TrainMode::Hybrid => "Hybrid Grover",
            TrainMode::AntiGrover => "AntiGrover (curiosity)",
            TrainMode::HybridAntiGrover => "Hybrid AntiGrover",
            TrainMode::Adaptive => "Adaptive (time-based)",
            TrainMode::Triggered => "Triggered (reward-based)",
            TrainMode::HybridTriggered => "Hybrid Triggered",
            TrainMode::EpsilonGreedy => "Epsilon-Greedy (ε=0.3)",
        };

        let mut final_rewards: Vec<f64> = Vec::new();
        let mut optimal_rates: Vec<f64> = Vec::new();
        let mut escape_episodes: Vec<Option<usize>> = Vec::new();

        for seed_idx in 0..n_seeds {
            let seed = 1000 + seed_idx as u64 * 7919;
            let (final_reward, optimal_rate, escape_ep) =
                run_two_action_rl(seed, n_episodes, trials_per_episode, *mode);
            final_rewards.push(final_reward);
            optimal_rates.push(optimal_rate);
            escape_episodes.push(escape_ep);
        }

        let mean_reward: f64 = final_rewards.iter().sum::<f64>() / n_seeds as f64;
        let mean_optimal: f64 = optimal_rates.iter().sum::<f64>() / n_seeds as f64;
        let escape_count = escape_episodes.iter().filter(|e| e.is_some()).count();

        println!("--- {} ---", name);
        println!("  Mean final reward:    {:.1}/episode", mean_reward);
        println!("  Optimal action rate:  {:.1}%", mean_optimal * 100.0);
        println!("  Escaped distractor:   {}/{} seeds", escape_count, n_seeds);
        println!();
    }
}

/// Run 2-action distractor experiment
fn run_two_action_rl(
    seed: u64,
    n_episodes: usize,
    trials_per_episode: usize,
    mode: TrainMode,
) -> (f64, f64, Option<usize>) {
    let interference_mode = match mode {
        TrainMode::Classical | TrainMode::EpsilonGreedy => InterferenceMode::None,
        TrainMode::PureQuantum | TrainMode::Hybrid => InterferenceMode::Grover,
        TrainMode::AntiGrover | TrainMode::HybridAntiGrover => InterferenceMode::AntiGrover,
        TrainMode::Adaptive => InterferenceMode::Adaptive,
        TrainMode::Triggered | TrainMode::HybridTriggered => InterferenceMode::Triggered,
    };

    let config = QuantumSNNConfig {
        n_inputs: 4,
        hidden_layers: vec![16],
        n_outputs: 2, // Only 2 actions
        interference_mode,
        interference_depth: 1,
        mix_angle: 0.0,
        ticks_per_decision: 20,
        learning_enabled: true,
        trigger_threshold: 50, // Switch to exploitation after reward >= 50
        trigger_count: 20,     // Require 20 consecutive high rewards (ensure learning first)
        duplicate_inputs: false,
    };

    let mut qsnn = QuantumSNN::with_seed(config, seed);

    // Single context
    let context: [u8; 4] = [255, 128, 64, 32];

    let mut eps_rng = seed;
    let mut last_episode_reward = 0.0;
    let mut optimal_count = 0usize;
    let mut total_count = 0usize;
    let mut escape_episode: Option<usize> = None;

    for episode in 0..n_episodes {
        let mut episode_reward = 0i64;
        let mut episode_optimal = 0;

        for _trial in 0..trials_per_episode {
            qsnn.set_inputs(&context);
            qsnn.decide();

            let action = match mode {
                TrainMode::Classical => qsnn.classical_action(),
                // Triggered modes: quantum while exploring, classical after trigger
                TrainMode::Triggered | TrainMode::HybridTriggered => {
                    if qsnn.is_exploitation_triggered() {
                        qsnn.classical_action() // Deterministic after trigger
                    } else {
                        qsnn.last_action() // Probabilistic while exploring
                    }
                }
                // All other quantum modes use quantum sampling
                TrainMode::PureQuantum
                | TrainMode::Hybrid
                | TrainMode::AntiGrover
                | TrainMode::HybridAntiGrover
                | TrainMode::Adaptive => qsnn.last_action(),
                TrainMode::EpsilonGreedy => {
                    eps_rng = eps_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (eps_rng >> 33) as f64 / (1u64 << 31) as f64;
                    if r < 0.3 {
                        ((eps_rng >> 40) % 2) as usize
                    } else {
                        qsnn.classical_action()
                    }
                }
            };

            // Reward: action 0 = +20 (distractor), action 1 = +100 (optimal)
            let reward = if action == 1 { 100i8 } else { 20i8 };
            episode_reward += reward as i64;

            if action == 1 {
                episode_optimal += 1;
            }

            // Track stats for last 100 episodes
            if episode >= n_episodes - 100 {
                total_count += 1;
                if action == 1 {
                    optimal_count += 1;
                }
            }

            // Learning: use classical action for credit assignment in Hybrid modes
            let training_action = match mode {
                TrainMode::Hybrid | TrainMode::HybridAntiGrover | TrainMode::HybridTriggered => {
                    qsnn.classical_action()
                }
                _ => action,
            };
            let training_reward = if training_action == 1 { 100i8 } else { 20i8 };
            qsnn.learn(training_reward);
        }

        // Check if escaped (>70% optimal in this episode)
        if episode_optimal > trials_per_episode * 7 / 10 && escape_episode.is_none() {
            escape_episode = Some(episode);
        }

        last_episode_reward = episode_reward as f64 / trials_per_episode as f64;
    }

    let optimal_rate = optimal_count as f64 / total_count.max(1) as f64;
    (last_episode_reward, optimal_rate, escape_episode)
}

#[allow(dead_code)]
/// Test with distractors (4 actions) - keeping for reference
fn test_with_distractors() {
    println!("\n=== TEST 2: SPARSE REWARDS WITH DISTRACTORS ===\n");

    println!("Task: Hidden Treasure with Distractors (4 actions)");
    println!("  - Context A: Action 0 gives +20 (distractor), Action 2 gives +100 (optimal)");
    println!("  - Context B: Action 1 gives +20 (distractor), Action 3 gives +100 (optimal)");
    println!("  - Other actions give 0\n");

    println!("Challenge: Agent might get stuck exploiting +20 distractor");
    println!("Goal: Discover optimal +100 actions through exploration\n");

    let n_seeds = 10;
    let n_episodes = 500;
    let trials_per_episode = 20;

    println!(
        "Running {} seeds, {} episodes each...\n",
        n_seeds, n_episodes
    );

    let modes = [
        TrainMode::Classical,
        TrainMode::PureQuantum,
        TrainMode::Hybrid,
        TrainMode::EpsilonGreedy,
    ];

    let mut all_results: HashMap<&str, (Vec<f64>, Vec<f64>)> = HashMap::new();

    for mode in &modes {
        let name = match mode {
            TrainMode::Classical => "Classical (greedy)",
            TrainMode::PureQuantum => "Pure Quantum",
            TrainMode::Hybrid => "Hybrid",
            TrainMode::AntiGrover => "AntiGrover (curiosity)",
            TrainMode::HybridAntiGrover => "Hybrid AntiGrover",
            TrainMode::Adaptive => "Adaptive (time-based)",
            TrainMode::Triggered => "Triggered (reward-based)",
            TrainMode::HybridTriggered => "Hybrid Triggered",
            TrainMode::EpsilonGreedy => "Epsilon-Greedy (ε=0.3)",
        };

        let mut final_rewards: Vec<f64> = Vec::new();
        let mut optimal_rates: Vec<f64> = Vec::new();

        for seed_idx in 0..n_seeds {
            let seed = 1000 + seed_idx as u64 * 7919;
            let (final_reward, optimal_rate) =
                run_distractor_rl(seed, n_episodes, trials_per_episode, *mode);
            final_rewards.push(final_reward);
            optimal_rates.push(optimal_rate);
        }

        let mean_reward: f64 = final_rewards.iter().sum::<f64>() / n_seeds as f64;
        let mean_optimal: f64 = optimal_rates.iter().sum::<f64>() / n_seeds as f64;

        println!("--- {} ---", name);
        println!("  Mean final reward:    {:.1}/episode", mean_reward);
        println!("  Optimal action rate:  {:.1}%", mean_optimal * 100.0);
        println!();

        all_results.insert(name, (final_rewards, optimal_rates));
    }

    // Summary
    println!("================================================================");
    println!("  TEST 2 SUMMARY: ESCAPING LOCAL OPTIMA");
    println!("================================================================\n");

    println!(
        "{:<25} {:>12} {:>15}",
        "Mode", "Mean Reward", "Optimal Rate"
    );
    println!("{:-<55}", "");

    for mode in &modes {
        let name = match mode {
            TrainMode::Classical => "Classical (greedy)",
            TrainMode::PureQuantum => "Pure Quantum",
            TrainMode::Hybrid => "Hybrid",
            TrainMode::AntiGrover => "AntiGrover (curiosity)",
            TrainMode::HybridAntiGrover => "Hybrid AntiGrover",
            TrainMode::Adaptive => "Adaptive (time-based)",
            TrainMode::Triggered => "Triggered (reward-based)",
            TrainMode::HybridTriggered => "Hybrid Triggered",
            TrainMode::EpsilonGreedy => "Epsilon-Greedy (ε=0.3)",
        };

        if let Some((rewards, optimals)) = all_results.get(name) {
            let mean_r: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
            let mean_o: f64 = optimals.iter().sum::<f64>() / optimals.len() as f64;
            println!("{:<25} {:>12.1} {:>14.1}%", name, mean_r, mean_o * 100.0);
        }
    }

    // Analysis
    let hybrid_optimal = all_results
        .get("Hybrid")
        .map(|(_, o)| o.iter().sum::<f64>() / o.len() as f64)
        .unwrap_or(0.0);
    let pure_optimal = all_results
        .get("Pure Quantum")
        .map(|(_, o)| o.iter().sum::<f64>() / o.len() as f64)
        .unwrap_or(0.0);
    let classical_optimal = all_results
        .get("Classical (greedy)")
        .map(|(_, o)| o.iter().sum::<f64>() / o.len() as f64)
        .unwrap_or(0.0);

    println!("\nANALYSIS:");
    if hybrid_optimal > pure_optimal {
        println!("  ✓ Hybrid finds optimal actions more often than Pure Quantum");
        println!("    → Classical credit assignment helps escape distractors!");
    }
    if hybrid_optimal > classical_optimal {
        println!("  ✓ Hybrid beats Classical - exploration finds optimal rewards");
    }
}

/// Run distractor RL experiment
fn run_distractor_rl(
    seed: u64,
    n_episodes: usize,
    trials_per_episode: usize,
    mode: TrainMode,
) -> (f64, f64) {
    let interference_mode = match mode {
        TrainMode::Classical | TrainMode::EpsilonGreedy => InterferenceMode::None,
        TrainMode::PureQuantum | TrainMode::Hybrid => InterferenceMode::Grover,
        TrainMode::AntiGrover | TrainMode::HybridAntiGrover => InterferenceMode::AntiGrover,
        TrainMode::Adaptive => InterferenceMode::Adaptive,
        TrainMode::Triggered | TrainMode::HybridTriggered => InterferenceMode::Triggered,
    };

    let config = QuantumSNNConfig {
        n_inputs: 4,
        hidden_layers: vec![16],
        n_outputs: 4,
        interference_mode,
        interference_depth: 1,
        mix_angle: 0.0,
        ticks_per_decision: 20,
        learning_enabled: true,
        trigger_threshold: 50,
        trigger_count: 20,
        duplicate_inputs: false,
    };

    let mut qsnn = QuantumSNN::with_seed(config, seed);

    let context_a: [u8; 4] = [255, 0, 0, 0];
    let context_b: [u8; 4] = [0, 255, 0, 0];

    // Rewards: distractor gives +20, optimal gives +100
    // Context A: action 0 = +20 (distractor), action 2 = +100 (optimal)
    // Context B: action 1 = +20 (distractor), action 3 = +100 (optimal)

    let mut eps_rng = seed;
    let mut last_episode_reward = 0.0;
    let mut optimal_count = 0usize;
    let mut total_count = 0usize;

    for _episode in 0..n_episodes {
        let mut episode_reward = 0i64;

        for trial in 0..trials_per_episode {
            let (context, distractor, optimal) = if trial % 2 == 0 {
                (&context_a, 0, 2)
            } else {
                (&context_b, 1, 3)
            };

            qsnn.set_inputs(context);
            qsnn.decide();

            let action = match mode {
                TrainMode::Classical => qsnn.classical_action(),
                TrainMode::PureQuantum
                | TrainMode::Hybrid
                | TrainMode::AntiGrover
                | TrainMode::HybridAntiGrover
                | TrainMode::Adaptive
                | TrainMode::Triggered
                | TrainMode::HybridTriggered => qsnn.last_action(),
                TrainMode::EpsilonGreedy => {
                    eps_rng = eps_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (eps_rng >> 33) as f64 / (1u64 << 31) as f64;
                    if r < 0.3 {
                        ((eps_rng >> 40) % 4) as usize
                    } else {
                        qsnn.classical_action()
                    }
                }
            };

            // Reward function with distractors
            let reward = if action == optimal {
                100i8
            } else if action == distractor {
                20i8
            } else {
                0i8
            };
            episode_reward += reward as i64;

            // Track optimal action rate (last 100 episodes)
            if _episode >= n_episodes - 100 {
                total_count += 1;
                if action == optimal {
                    optimal_count += 1;
                }
            }

            // Learning: use classical for Hybrid modes
            let training_action = match mode {
                TrainMode::Hybrid | TrainMode::HybridAntiGrover | TrainMode::HybridTriggered => {
                    qsnn.classical_action()
                }
                _ => action,
            };
            let training_reward = if training_action == optimal {
                100i8
            } else if training_action == distractor {
                20i8
            } else {
                0i8
            };
            qsnn.learn(training_reward);
        }

        last_episode_reward = episode_reward as f64 / trials_per_episode as f64;
    }

    let optimal_rate = optimal_count as f64 / total_count.max(1) as f64;
    (last_episode_reward, optimal_rate)
}

/// Run one sparse RL experiment
/// Returns (final_episode_reward, episode_when_both_treasures_discovered)
fn run_sparse_rl(
    seed: u64,
    n_episodes: usize,
    trials_per_episode: usize,
    mode: TrainMode,
) -> (f64, Option<usize>) {
    // Create network with 4 outputs (actions)
    let interference_mode = match mode {
        TrainMode::Classical | TrainMode::EpsilonGreedy => InterferenceMode::None,
        TrainMode::PureQuantum | TrainMode::Hybrid => InterferenceMode::Grover,
        TrainMode::AntiGrover | TrainMode::HybridAntiGrover => InterferenceMode::AntiGrover,
        TrainMode::Adaptive => InterferenceMode::Adaptive,
        TrainMode::Triggered | TrainMode::HybridTriggered => InterferenceMode::Triggered,
    };

    let config = QuantumSNNConfig {
        n_inputs: 4,
        hidden_layers: vec![16],
        n_outputs: 4, // 4 actions
        interference_mode,
        interference_depth: 1,
        mix_angle: 0.0,
        ticks_per_decision: 20,
        learning_enabled: true,
        trigger_threshold: 50,
        trigger_count: 20,
        duplicate_inputs: false,
    };

    let mut qsnn = QuantumSNN::with_seed(config, seed);

    // Contexts and their hidden treasure actions
    let context_a: [u8; 4] = [255, 0, 0, 0];
    let context_b: [u8; 4] = [0, 255, 0, 0];
    let treasure_a = 2; // Action 2 is treasure for context A
    let treasure_b = 0; // Action 0 is treasure for context B

    // Track discovery
    let mut found_treasure_a = false;
    let mut found_treasure_b = false;
    let mut discovery_episode: Option<usize> = None;

    // Epsilon-greedy RNG
    let mut eps_rng = seed;

    let mut last_episode_reward = 0.0;

    for episode in 0..n_episodes {
        let mut episode_reward = 0i64;

        for trial in 0..trials_per_episode {
            // Alternate contexts
            let (context, treasure) = if trial % 2 == 0 {
                (&context_a, treasure_a)
            } else {
                (&context_b, treasure_b)
            };

            qsnn.set_inputs(context);
            qsnn.decide();

            // Select action based on mode
            let action = match mode {
                TrainMode::Classical => qsnn.classical_action(),
                // Triggered modes: quantum while exploring, classical after trigger
                TrainMode::Triggered | TrainMode::HybridTriggered => {
                    if qsnn.is_exploitation_triggered() {
                        qsnn.classical_action() // Deterministic after trigger
                    } else {
                        qsnn.last_action() // Probabilistic while exploring
                    }
                }
                // All other quantum modes use quantum sampling
                TrainMode::PureQuantum
                | TrainMode::Hybrid
                | TrainMode::AntiGrover
                | TrainMode::HybridAntiGrover
                | TrainMode::Adaptive => qsnn.last_action(),
                TrainMode::EpsilonGreedy => {
                    // 30% random exploration
                    eps_rng = eps_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (eps_rng >> 33) as f64 / (1u64 << 31) as f64;
                    if r < 0.3 {
                        ((eps_rng >> 40) % 4) as usize // Random action
                    } else {
                        qsnn.classical_action() // Greedy
                    }
                }
            };

            // Sparse reward: only treasure action gives reward
            let reward = if action == treasure { 100i8 } else { 0i8 };
            episode_reward += reward as i64;

            // Track discovery
            if action == treasure {
                if trial % 2 == 0 {
                    found_treasure_a = true;
                } else {
                    found_treasure_b = true;
                }
            }

            // Learning: use classical action for credit assignment in Hybrid modes
            let training_action = match mode {
                TrainMode::Hybrid | TrainMode::HybridAntiGrover | TrainMode::HybridTriggered => {
                    qsnn.classical_action()
                }
                _ => action,
            };
            let training_reward = if training_action == treasure {
                100i8
            } else {
                0i8
            };
            qsnn.learn(training_reward);
        }

        // Check if both treasures discovered this episode
        if found_treasure_a && found_treasure_b && discovery_episode.is_none() {
            discovery_episode = Some(episode);
        }

        last_episode_reward = episode_reward as f64 / trials_per_episode as f64;
    }

    (last_episode_reward, discovery_episode)
}
