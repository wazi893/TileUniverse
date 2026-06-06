//! Four-Way Brain Comparison: Full Quantum vs Separable Quantum vs Classical NN vs QUBO
//!
//! Run with: cargo run --release --example four_way_qubo_experiment
//!
//! This experiment compares all four brain types under identical conditions.

use engine::brain::BrainType;
use engine::experiments::three_way_comparison::{BrainPopulation, ThreeWayConfig};
use engine::simulation::Simulation;
use std::collections::HashMap;
use std::time::Instant;

/// Result from a single trial
#[derive(Debug, Clone)]
struct TrialResult {
    seed: u64,
    brain_type: BrainType,
    final_population: u32,
    population_auc: f64,
    extinct: bool,
    max_generation: u32,
}

/// Run a single trial
fn run_trial(seed: u64, brain_type: BrainType, config: &ThreeWayConfig) -> TrialResult {
    let mut sim = Simulation::new();
    let mut pop = BrainPopulation::new(config.forager_config.clone(), brain_type, seed);

    // Initial resources
    pop.spawn_resources(&mut sim, config.heat_patches * 5, config.heat_value);

    let mut population_sum = 0u64;
    let mut max_gen = 0u32;

    for tick in 0..config.max_ticks {
        // Spawn resources periodically
        if tick > 0 && tick % config.resource_interval == 0 {
            pop.spawn_resources(&mut sim, config.heat_patches, config.heat_value);
        }

        let stats = pop.step(&mut sim);
        population_sum += stats.size as u64;
        max_gen = max_gen.max(stats.max_generation);

        if pop.is_extinct() {
            break;
        }
    }

    let stats = pop.stats();
    TrialResult {
        seed,
        brain_type,
        final_population: stats.size as u32,
        population_auc: population_sum as f64,
        extinct: pop.is_extinct(),
        max_generation: max_gen,
    }
}

/// Aggregated statistics for a brain type
#[derive(Debug)]
struct GroupStats {
    brain_type: BrainType,
    n_trials: usize,
    mean_pop: f64,
    std_pop: f64,
    median_pop: f64,
    mean_auc: f64,
    extinctions: usize,
    mean_max_gen: f64,
}

fn compute_stats(trials: &[TrialResult]) -> GroupStats {
    let n = trials.len() as f64;
    let pops: Vec<f64> = trials.iter().map(|t| t.final_population as f64).collect();
    let aucs: Vec<f64> = trials.iter().map(|t| t.population_auc).collect();
    let gens: Vec<f64> = trials.iter().map(|t| t.max_generation as f64).collect();

    let mean_pop = pops.iter().sum::<f64>() / n;
    let variance = pops.iter().map(|p| (p - mean_pop).powi(2)).sum::<f64>() / n;
    let std_pop = variance.sqrt();

    let mut sorted = pops.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_pop = sorted[sorted.len() / 2];

    let mean_auc = aucs.iter().sum::<f64>() / n;
    let extinctions = trials.iter().filter(|t| t.extinct).count();
    let mean_max_gen = gens.iter().sum::<f64>() / n;

    GroupStats {
        brain_type: trials[0].brain_type,
        n_trials: trials.len(),
        mean_pop,
        std_pop,
        median_pop,
        mean_auc,
        extinctions,
        mean_max_gen,
    }
}

/// Compute paired t-test statistics
fn paired_comparison(a: &[TrialResult], b: &[TrialResult]) -> (f64, f64, f64) {
    // Returns (mean_diff, t_statistic, cohen_d)
    let n = a.len() as f64;
    let diffs: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.population_auc - y.population_auc)
        .collect();

    let mean_diff = diffs.iter().sum::<f64>() / n;
    let var_diff = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0);
    let std_diff = var_diff.sqrt();

    let t_stat = if std_diff > 0.0 {
        mean_diff / (std_diff / n.sqrt())
    } else {
        0.0
    };

    // Pooled std for Cohen's d
    let a_pops: Vec<f64> = a.iter().map(|t| t.population_auc).collect();
    let b_pops: Vec<f64> = b.iter().map(|t| t.population_auc).collect();
    let a_mean = a_pops.iter().sum::<f64>() / n;
    let b_mean = b_pops.iter().sum::<f64>() / n;
    let a_var = a_pops.iter().map(|p| (p - a_mean).powi(2)).sum::<f64>() / (n - 1.0);
    let b_var = b_pops.iter().map(|p| (p - b_mean).powi(2)).sum::<f64>() / (n - 1.0);
    let pooled_std = ((a_var + b_var) / 2.0).sqrt();
    let cohen_d = if pooled_std > 0.0 {
        mean_diff / pooled_std
    } else {
        0.0
    };

    (mean_diff, t_stat, cohen_d)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  FOUR-WAY BRAIN COMPARISON EXPERIMENT                          ║");
    println!("║  Full Quantum vs Separable Quantum vs Classical NN vs QUBO     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Create config
    let mut config = ThreeWayConfig::default();
    config.replications = 30; // Quick run - increase to 100 for full power
    config.max_ticks = 1000;

    let brain_types = vec![
        BrainType::FullQuantum,
        BrainType::SeparableQuantum,
        BrainType::ClassicalNN,
        BrainType::QuboBrain,
    ];

    println!("Configuration:");
    println!("  Replications: {}", config.replications);
    println!("  Max ticks: {}", config.max_ticks);
    println!(
        "  Initial population: {}",
        config.forager_config.initial_population
    );
    println!("  Mutation rate: {}", config.forager_config.mutation_rate);
    println!();

    let total_trials = config.replications * brain_types.len();
    println!("Running {} total trials...", total_trials);
    println!();

    let start = Instant::now();

    // Run all trials
    let mut all_results: HashMap<BrainType, Vec<TrialResult>> = HashMap::new();

    for &brain_type in &brain_types {
        println!("=== {} ===", brain_type.display_name());
        let mut trials = Vec::new();

        for i in 0..config.replications {
            let seed = config.seed_start + i as u64;

            if (i + 1) % 10 == 0 {
                print!("  {}/{}", i + 1, config.replications);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }

            let result = run_trial(seed, brain_type, &config);
            trials.push(result);
        }
        println!("  Done!");

        all_results.insert(brain_type, trials);
    }

    let elapsed = start.elapsed();

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!(
        "RESULTS (completed in {:.1} seconds)",
        elapsed.as_secs_f64()
    );
    println!("════════════════════════════════════════════════════════════════");
    println!();

    // Compute and print stats for each group
    let mut stats: Vec<GroupStats> = brain_types
        .iter()
        .map(|bt| compute_stats(all_results.get(bt).unwrap()))
        .collect();

    // Sort by mean AUC (best first)
    stats.sort_by(|a, b| b.mean_auc.partial_cmp(&a.mean_auc).unwrap());

    println!("GROUP STATISTICS (sorted by AUC):");
    println!(
        "┌─────────────────────────┬──────────┬──────────┬──────────┬────────────┬────────────┬──────────┐"
    );
    println!(
        "│ Brain Type              │ Mean Pop │ Std Pop  │ Median   │ Mean AUC   │ Extinctions│ Max Gen  │"
    );
    println!(
        "├─────────────────────────┼──────────┼──────────┼──────────┼────────────┼────────────┼──────────┤"
    );

    for s in &stats {
        println!(
            "│ {:23} │ {:8.1} │ {:8.1} │ {:8.1} │ {:10.0} │ {:4}/{:4}   │ {:8.1} │",
            s.brain_type.display_name(),
            s.mean_pop,
            s.std_pop,
            s.median_pop,
            s.mean_auc,
            s.extinctions,
            s.n_trials,
            s.mean_max_gen,
        );
    }
    println!(
        "└─────────────────────────┴──────────┴──────────┴──────────┴────────────┴────────────┴──────────┘"
    );

    // Pairwise comparisons
    println!();
    println!("PAIRWISE COMPARISONS (AUC-based):");
    println!("┌─────────────────────────────────────┬────────────┬──────────┬──────────┐");
    println!("│ Comparison                          │ Mean Diff  │ t-stat   │ Cohen's d│");
    println!("├─────────────────────────────────────┼────────────┼──────────┼──────────┤");

    let comparisons = [
        (
            BrainType::QuboBrain,
            BrainType::ClassicalNN,
            "QUBO vs Classical NN",
        ),
        (
            BrainType::QuboBrain,
            BrainType::FullQuantum,
            "QUBO vs Full Quantum",
        ),
        (
            BrainType::ClassicalNN,
            BrainType::FullQuantum,
            "Classical vs Full Quantum",
        ),
        (
            BrainType::FullQuantum,
            BrainType::SeparableQuantum,
            "Full Q vs Separable Q",
        ),
    ];

    for (a_type, b_type, label) in &comparisons {
        let a_trials = all_results.get(a_type).unwrap();
        let b_trials = all_results.get(b_type).unwrap();
        let (mean_diff, t_stat, cohen_d) = paired_comparison(a_trials, b_trials);

        let significance = if t_stat.abs() > 2.756 {
            "***"
        }
        // p < 0.01
        else if t_stat.abs() > 2.045 {
            "**"
        }
        // p < 0.05
        else if t_stat.abs() > 1.699 {
            "*"
        }
        // p < 0.10
        else {
            ""
        };

        println!(
            "│ {:35} │ {:+10.0} │ {:+7.2}{} │ {:+8.2} │",
            label, mean_diff, t_stat, significance, cohen_d,
        );
    }
    println!("└─────────────────────────────────────┴────────────┴──────────┴──────────┘");
    println!(
        "Significance: * p<0.10, ** p<0.05, *** p<0.01 (df={})",
        config.replications - 1
    );

    // Winner summary
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════════════════════════");

    let winner = &stats[0];
    let runner_up = &stats[1];

    println!(
        "🥇 Winner: {} (AUC: {:.0})",
        winner.brain_type.display_name(),
        winner.mean_auc
    );
    println!(
        "🥈 Runner-up: {} (AUC: {:.0})",
        runner_up.brain_type.display_name(),
        runner_up.mean_auc
    );

    let qubo_stats = stats
        .iter()
        .find(|s| s.brain_type == BrainType::QuboBrain)
        .unwrap();
    let classical_stats = stats
        .iter()
        .find(|s| s.brain_type == BrainType::ClassicalNN)
        .unwrap();

    let qubo_vs_classical_pct = if classical_stats.mean_auc > 0.0 {
        ((qubo_stats.mean_auc - classical_stats.mean_auc) / classical_stats.mean_auc) * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "QUBO Brain vs Classical NN: {:+.1}% AUC",
        qubo_vs_classical_pct
    );

    if qubo_vs_classical_pct > 5.0 {
        println!("→ QUBO shows promising advantage over classical!");
    } else if qubo_vs_classical_pct < -5.0 {
        println!("→ Classical NN dominates (as expected from EPIC 91)");
    } else {
        println!("→ No significant difference between QUBO and Classical");
    }
}
