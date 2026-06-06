//! EPIC 92 Sprint 92.2: Temporal Dynamics
//!
//! Test H2: Quantum might lead early but classical catches up.
//! Measure population at multiple time snapshots to find advantage windows.
//!
//! EPIC 92.5 Enhancement: Add statistical rigor to distinguish:
//! - Instantaneous population advantage vs cumulative AUC advantage
//! - Statistical significance (p-value, Cohen's d, 95% CI)

use crate::brain::BrainType;
use crate::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use crate::simulation::Simulation;
use crate::three_way_comparison::BrainPopulation;

/// Raw data from a single trial at a specific snapshot.
#[derive(Clone, Debug)]
pub struct TrialSnapshot {
    pub tick: u64,
    pub population: usize,
    pub energy: f32,
    pub quantumness: f32,
    pub cumulative_auc: f64, // Total population-ticks up to this point
}

/// Statistical analysis for a snapshot comparison.
#[derive(Clone, Debug)]
pub struct SnapshotStatistics {
    pub tick: u64,

    // Population stats
    pub quantum_pop_mean: f32,
    pub quantum_pop_std: f32,
    pub classical_pop_mean: f32,
    pub classical_pop_std: f32,
    pub pop_delta: f32,
    pub pop_pvalue: f32,
    pub pop_cohens_d: f32,
    pub pop_ci_lower: f32,
    pub pop_ci_upper: f32,

    // Cumulative AUC stats
    pub quantum_auc_mean: f64,
    pub quantum_auc_std: f64,
    pub classical_auc_mean: f64,
    pub classical_auc_std: f64,
    pub auc_delta: f64,
    pub auc_pvalue: f32,
    pub auc_cohens_d: f32,

    // Winner flags
    pub pop_winner: &'static str,
    pub auc_winner: &'static str,
}

/// Snapshot of population state at a specific tick.
#[derive(Clone, Debug)]
pub struct TemporalSnapshot {
    pub tick: u64,
    pub population: usize,
    pub mean_energy: f32,
    pub mean_quantumness: f32,
    pub extinctions: usize, // How many trials extinct at this tick
}

/// Result for one brain type across all time snapshots.
#[derive(Clone, Debug)]
pub struct TemporalResult {
    pub brain_type: BrainType,
    pub snapshots: Vec<TemporalSnapshot>,
    pub seeds_tested: usize,
}

impl TemporalResult {
    /// Find the tick where this brain type performs best.
    pub fn peak_tick(&self) -> (u64, usize) {
        self.snapshots
            .iter()
            .map(|s| (s.tick, s.population))
            .max_by_key(|(_, pop)| *pop)
            .unwrap()
    }

    /// Check if this brain leads at any snapshot.
    pub fn ever_leads(&self, other: &TemporalResult) -> Option<u64> {
        for (self_snap, other_snap) in self.snapshots.iter().zip(other.snapshots.iter()) {
            if self_snap.population > other_snap.population {
                return Some(self_snap.tick);
            }
        }
        None
    }
}

/// Configuration for temporal dynamics experiment.
#[derive(Clone, Debug)]
pub struct TemporalConfig {
    pub snapshot_ticks: Vec<u64>,
    pub seeds: usize,
    pub mutation_rate: f32,
    pub heat_patches: u32,
    pub heat_value: u32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            snapshot_ticks: vec![250, 500, 750, 1000, 1500, 2000, 3000, 5000],
            seeds: 50,
            mutation_rate: 0.7, // Use optimal from 92.1
            heat_patches: 19,
            heat_value: 35,
        }
    }
}

/// Full temporal sweep results.
#[derive(Clone, Debug)]
pub struct TemporalSweepResults {
    pub config: TemporalConfig,
    pub full_quantum: TemporalResult,
    pub classical_nn: TemporalResult,
    pub statistics: Vec<SnapshotStatistics>, // Statistical analysis at each snapshot
}

impl TemporalSweepResults {
    /// Find crossover point where classical overtakes quantum (if any).
    pub fn crossover_tick(&self) -> Option<u64> {
        for (q_snap, c_snap) in self
            .full_quantum
            .snapshots
            .iter()
            .zip(self.classical_nn.snapshots.iter())
        {
            if c_snap.population > q_snap.population {
                return Some(c_snap.tick);
            }
        }
        None
    }

    /// Check if quantum ever leads (on instantaneous population).
    pub fn quantum_ever_leads(&self) -> Option<u64> {
        self.full_quantum.ever_leads(&self.classical_nn)
    }

    /// Check if quantum ever leads on cumulative AUC.
    pub fn quantum_auc_ever_leads(&self) -> Option<u64> {
        for stat in &self.statistics {
            if stat.quantum_auc_mean > stat.classical_auc_mean {
                return Some(stat.tick);
            }
        }
        None
    }

    /// Find the first tick where quantum has statistically significant advantage.
    pub fn first_significant_advantage(&self, alpha: f32) -> Option<(u64, &'static str)> {
        for stat in &self.statistics {
            if stat.pop_pvalue < alpha && stat.pop_delta > 0.0 {
                return Some((stat.tick, "population"));
            }
            if stat.auc_pvalue < alpha && stat.auc_delta > 0.0 {
                return Some((stat.tick, "AUC"));
            }
        }
        None
    }
}

/// Run temporal dynamics sweep with enhanced statistical analysis.
pub fn run_temporal_sweep(config: &TemporalConfig) -> TemporalSweepResults {
    let brain_types = vec![BrainType::FullQuantum, BrainType::ClassicalNN];
    let max_ticks = *config.snapshot_ticks.iter().max().unwrap();

    // Store raw trial data: [brain_type][snapshot_idx][trial_idx]
    let mut quantum_trials: Vec<Vec<TrialSnapshot>> = vec![Vec::new(); config.snapshot_ticks.len()];
    let mut classical_trials: Vec<Vec<TrialSnapshot>> =
        vec![Vec::new(); config.snapshot_ticks.len()];

    for &brain_type in &brain_types {
        println!("\n=== Running {} ===", brain_type.display_name());

        for seed_idx in 0..config.seeds {
            let seed = (seed_idx + 1) as u64;

            if (seed_idx + 1) % 10 == 0 {
                println!("  Completed {}/{} seeds", seed_idx + 1, config.seeds);
            }

            // Run single trial
            let mut sim = Simulation::new();
            let forager_config = QuantumForagerConfig {
                initial_population: 30,
                initial_energy: 100.0,
                n_qubits: 4,
                initial_circuit_length: 8,
                max_circuit_length: 25,
                movement_cost: 0.5,
                metabolism_cost: 0.25,
                reproduction_threshold: 140.0,
                reproduction_cost: 55.0,
                mutation_rate: config.mutation_rate,
                heat_energy_factor: 0.7,
                charge_energy_factor: 0.3,
                consume_resources: true,
                max_population: 150,
                region: ForagerRegion {
                    x0: 30,
                    y0: 30,
                    width: 100,
                    height: 100,
                },
            };

            let mut pop = BrainPopulation::new(forager_config, brain_type, seed);
            pop.spawn_resources(&mut sim, config.heat_patches * 5, config.heat_value);

            let mut snapshot_idx = 0;
            let mut cumulative_auc = 0.0_f64;

            for tick in 0..max_ticks {
                if tick > 0 && tick % 8 == 0 {
                    pop.spawn_resources(&mut sim, config.heat_patches, config.heat_value);
                }

                pop.step(&mut sim);

                let stats = pop.stats();
                cumulative_auc += stats.size as f64;

                // Check if we've hit a snapshot tick
                if snapshot_idx < config.snapshot_ticks.len()
                    && tick == config.snapshot_ticks[snapshot_idx]
                {
                    let trial_snap = TrialSnapshot {
                        tick: config.snapshot_ticks[snapshot_idx],
                        population: stats.size,
                        energy: stats.avg_energy,
                        quantumness: stats.avg_quantum_ness,
                        cumulative_auc,
                    };

                    if brain_type == BrainType::FullQuantum {
                        quantum_trials[snapshot_idx].push(trial_snap);
                    } else {
                        classical_trials[snapshot_idx].push(trial_snap);
                    }

                    snapshot_idx += 1;
                }
            }
        }
    }

    // Compute statistics at each snapshot
    let statistics =
        compute_snapshot_statistics(&config.snapshot_ticks, &quantum_trials, &classical_trials);

    // Build legacy TemporalResult structures for backward compatibility
    let quantum_snapshots = build_temporal_snapshots(&config.snapshot_ticks, &quantum_trials);
    let classical_snapshots = build_temporal_snapshots(&config.snapshot_ticks, &classical_trials);

    TemporalSweepResults {
        config: config.clone(),
        full_quantum: TemporalResult {
            brain_type: BrainType::FullQuantum,
            snapshots: quantum_snapshots,
            seeds_tested: config.seeds,
        },
        classical_nn: TemporalResult {
            brain_type: BrainType::ClassicalNN,
            snapshots: classical_snapshots,
            seeds_tested: config.seeds,
        },
        statistics,
    }
}

/// Build legacy TemporalSnapshot aggregates from raw trial data.
fn build_temporal_snapshots(ticks: &[u64], trials: &[Vec<TrialSnapshot>]) -> Vec<TemporalSnapshot> {
    ticks
        .iter()
        .enumerate()
        .map(|(idx, &tick)| {
            let pops: Vec<usize> = trials[idx].iter().map(|t| t.population).collect();
            let energies: Vec<f32> = trials[idx].iter().map(|t| t.energy).collect();
            let qnesses: Vec<f32> = trials[idx].iter().map(|t| t.quantumness).collect();

            let mean_pop = if !pops.is_empty() {
                pops.iter().sum::<usize>() as f32 / pops.len() as f32
            } else {
                0.0
            };
            let mean_energy = if !energies.is_empty() {
                energies.iter().sum::<f32>() / energies.len() as f32
            } else {
                0.0
            };
            let mean_qness = if !qnesses.is_empty() {
                qnesses.iter().sum::<f32>() / qnesses.len() as f32
            } else {
                0.0
            };
            let extinctions = pops.iter().filter(|&&p| p == 0).count();

            TemporalSnapshot {
                tick,
                population: mean_pop as usize,
                mean_energy,
                mean_quantumness: mean_qness,
                extinctions,
            }
        })
        .collect()
}

/// Compute mean of a slice.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute mean of f64 slice.
fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation.
fn std_dev(values: &[f32], mean_val: f32) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff
        })
        .sum::<f32>()
        / (values.len() - 1) as f32;
    variance.sqrt()
}

/// Compute standard deviation for f64.
fn std_dev_f64(values: &[f64], mean_val: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Paired t-test (returns approximate p-value using t-distribution).
fn paired_t_test(group1: &[f32], group2: &[f32]) -> f32 {
    if group1.len() != group2.len() || group1.is_empty() {
        return 1.0; // No significance
    }

    let n = group1.len() as f32;
    let differences: Vec<f32> = group1
        .iter()
        .zip(group2.iter())
        .map(|(a, b)| a - b)
        .collect();

    let mean_diff = mean(&differences);
    let std_diff = std_dev(&differences, mean_diff);

    if std_diff < 1e-9 {
        return if mean_diff.abs() < 1e-9 { 1.0 } else { 0.0 };
    }

    let t_stat = (mean_diff * n.sqrt()) / std_diff;
    let df = n - 1.0;

    // Approximate p-value using two-tailed test
    // For simplicity, use normal approximation if df > 30, else conservative estimate
    if df > 30.0 {
        let z = t_stat.abs();
        // Approximate p-value for two-tailed normal
        if z > 2.576 {
            0.01
        } else if z > 1.96 {
            0.05
        } else if z > 1.645 {
            0.10
        } else {
            0.20
        }
    } else {
        // Conservative: use t-critical values for small samples
        let t_abs = t_stat.abs();
        if t_abs > 2.75 {
            0.01
        } else if t_abs > 2.04 {
            0.05
        } else if t_abs > 1.70 {
            0.10
        } else {
            0.20
        }
    }
}

/// Paired t-test for f64 values (for AUC).
fn paired_t_test_f64(group1: &[f64], group2: &[f64]) -> f32 {
    if group1.len() != group2.len() || group1.is_empty() {
        return 1.0;
    }

    let n = group1.len() as f64;
    let differences: Vec<f64> = group1
        .iter()
        .zip(group2.iter())
        .map(|(a, b)| a - b)
        .collect();

    let mean_diff = mean_f64(&differences);
    let std_diff = std_dev_f64(&differences, mean_diff);

    if std_diff < 1e-9 {
        return if mean_diff.abs() < 1e-9 { 1.0 } else { 0.0 };
    }

    let t_stat = (mean_diff * n.sqrt()) / std_diff;
    let df = n - 1.0;

    if df > 30.0 {
        let z = t_stat.abs();
        if z > 2.576 {
            0.01
        } else if z > 1.96 {
            0.05
        } else if z > 1.645 {
            0.10
        } else {
            0.20
        }
    } else {
        let t_abs = t_stat.abs();
        if t_abs > 2.75 {
            0.01
        } else if t_abs > 2.04 {
            0.05
        } else if t_abs > 1.70 {
            0.10
        } else {
            0.20
        }
    }
}

/// Cohen's d effect size.
fn cohens_d(group1: &[f32], group2: &[f32]) -> f32 {
    if group1.is_empty() || group2.is_empty() {
        return 0.0;
    }

    let mean1 = mean(group1);
    let mean2 = mean(group2);
    let std1 = std_dev(group1, mean1);
    let std2 = std_dev(group2, mean2);

    let n1 = group1.len() as f32;
    let n2 = group2.len() as f32;

    // Pooled standard deviation
    let pooled_std =
        (((n1 - 1.0) * std1 * std1 + (n2 - 1.0) * std2 * std2) / (n1 + n2 - 2.0)).sqrt();

    if pooled_std < 1e-9 {
        return 0.0;
    }

    (mean1 - mean2) / pooled_std
}

/// Cohen's d for f64 values.
fn cohens_d_f64(group1: &[f64], group2: &[f64]) -> f32 {
    if group1.is_empty() || group2.is_empty() {
        return 0.0;
    }

    let mean1 = mean_f64(group1);
    let mean2 = mean_f64(group2);
    let std1 = std_dev_f64(group1, mean1);
    let std2 = std_dev_f64(group2, mean2);

    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;

    let pooled_std =
        (((n1 - 1.0) * std1 * std1 + (n2 - 1.0) * std2 * std2) / (n1 + n2 - 2.0)).sqrt();

    if pooled_std < 1e-9 {
        return 0.0;
    }

    ((mean1 - mean2) / pooled_std) as f32
}

/// Compute statistics at each snapshot.
fn compute_snapshot_statistics(
    ticks: &[u64],
    quantum_trials: &[Vec<TrialSnapshot>],
    classical_trials: &[Vec<TrialSnapshot>],
) -> Vec<SnapshotStatistics> {
    ticks
        .iter()
        .enumerate()
        .map(|(idx, &tick)| {
            let q_pops: Vec<f32> = quantum_trials[idx]
                .iter()
                .map(|t| t.population as f32)
                .collect();
            let c_pops: Vec<f32> = classical_trials[idx]
                .iter()
                .map(|t| t.population as f32)
                .collect();
            let q_aucs: Vec<f64> = quantum_trials[idx]
                .iter()
                .map(|t| t.cumulative_auc)
                .collect();
            let c_aucs: Vec<f64> = classical_trials[idx]
                .iter()
                .map(|t| t.cumulative_auc)
                .collect();

            let q_pop_mean = mean(&q_pops);
            let c_pop_mean = mean(&c_pops);
            let q_pop_std = std_dev(&q_pops, q_pop_mean);
            let c_pop_std = std_dev(&c_pops, c_pop_mean);

            let q_auc_mean = mean_f64(&q_aucs);
            let c_auc_mean = mean_f64(&c_aucs);
            let q_auc_std = std_dev_f64(&q_aucs, q_auc_mean);
            let c_auc_std = std_dev_f64(&c_aucs, c_auc_mean);

            let pop_delta = q_pop_mean - c_pop_mean;
            let auc_delta = q_auc_mean - c_auc_mean;

            let pop_pvalue = paired_t_test(&q_pops, &c_pops);
            let auc_pvalue = paired_t_test_f64(&q_aucs, &c_aucs);

            let pop_cohens_d = cohens_d(&q_pops, &c_pops);
            let auc_cohens_d = cohens_d_f64(&q_aucs, &c_aucs);

            // 95% CI for difference (using t-distribution approximation)
            let se = (q_pop_std * q_pop_std / q_pops.len() as f32
                + c_pop_std * c_pop_std / c_pops.len() as f32)
                .sqrt();
            let t_critical = 1.96; // Approximate for large n
            let pop_ci_lower = pop_delta - t_critical * se;
            let pop_ci_upper = pop_delta + t_critical * se;

            let pop_winner = if pop_delta > 0.0 {
                "Quantum"
            } else {
                "Classical"
            };
            let auc_winner = if auc_delta > 0.0 {
                "Quantum"
            } else {
                "Classical"
            };

            SnapshotStatistics {
                tick,
                quantum_pop_mean: q_pop_mean,
                quantum_pop_std: q_pop_std,
                classical_pop_mean: c_pop_mean,
                classical_pop_std: c_pop_std,
                pop_delta,
                pop_pvalue,
                pop_cohens_d,
                pop_ci_lower,
                pop_ci_upper,
                quantum_auc_mean: q_auc_mean,
                quantum_auc_std: q_auc_std,
                classical_auc_mean: c_auc_mean,
                classical_auc_std: c_auc_std,
                auc_delta,
                auc_pvalue,
                auc_cohens_d,
                pop_winner,
                auc_winner,
            }
        })
        .collect()
}

/// Print formatted temporal results with statistical analysis.
pub fn print_temporal_results(results: &TemporalSweepResults) {
    println!(
        "\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║   EPIC 92.5: TEMPORAL DYNAMICS WITH STATISTICAL RIGOR                          ║");
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("INSTANTANEOUS POPULATION ANALYSIS");
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("Tick    Q_mean±σ      C_mean±σ      Δ        p-val   d      95%CI           Winner");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    for stat in &results.statistics {
        let sig = if stat.pop_pvalue < 0.05 { "*" } else { " " };
        let note = if stat.tick == 1000 { " (EPIC 91)" } else { "" };

        println!(
            "{:<7} {:>5.1}±{:<4.1} {:>5.1}±{:<4.1} {:>+7.1}  {:<6.2}  {:>+4.2}  [{:>+5.1},{:>+5.1}]  {}{}{}",
            stat.tick,
            stat.quantum_pop_mean,
            stat.quantum_pop_std,
            stat.classical_pop_mean,
            stat.classical_pop_std,
            stat.pop_delta,
            stat.pop_pvalue,
            stat.pop_cohens_d,
            stat.pop_ci_lower,
            stat.pop_ci_upper,
            stat.pop_winner,
            sig,
            note
        );
    }

    println!("\n\nCUMULATIVE AUC (Total Organism-Ticks Lived)");
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("Tick    Q_AUC          C_AUC          Δ_AUC       p-val   d      Winner");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    for stat in &results.statistics {
        let sig = if stat.auc_pvalue < 0.05 { "*" } else { " " };
        let note = if stat.tick == 1000 { " (EPIC 91)" } else { "" };

        println!(
            "{:<7} {:>12.0}   {:>12.0}   {:>+11.0}  {:<6.2}  {:>+4.2}  {}{}{}",
            stat.tick,
            stat.quantum_auc_mean,
            stat.classical_auc_mean,
            stat.auc_delta,
            stat.auc_pvalue,
            stat.auc_cohens_d,
            stat.auc_winner,
            sig,
            note
        );
    }

    println!("\n─────────────────────────────────────────────────────────────────────────────────");
    println!("* indicates p < 0.05 (statistically significant difference)");
    println!("d = Cohen's d effect size (|d| > 0.5 = medium, |d| > 0.8 = large effect)");

    // Key findings
    println!(
        "\n\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║   KEY FINDINGS                                                                 ║");
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Check for instantaneous population advantage
    if let Some(lead_tick) = results.quantum_ever_leads() {
        let stat = results
            .statistics
            .iter()
            .find(|s| s.tick == lead_tick)
            .unwrap();
        println!("✓ QUANTUM LEADS ON INSTANTANEOUS POPULATION:");
        println!("  First occurrence: tick {}", lead_tick);
        println!(
            "  At tick {}: Q={:.1}±{:.1} vs C={:.1}±{:.1}, Δ={:+.1}",
            lead_tick,
            stat.quantum_pop_mean,
            stat.quantum_pop_std,
            stat.classical_pop_mean,
            stat.classical_pop_std,
            stat.pop_delta
        );
        println!(
            "  p={:.3}, d={:+.2}, 95%CI=[{:+.1}, {:+.1}]",
            stat.pop_pvalue, stat.pop_cohens_d, stat.pop_ci_lower, stat.pop_ci_upper
        );

        if stat.pop_pvalue < 0.05 {
            println!("  ✓ Statistically significant (p < 0.05)");
        } else {
            println!(
                "  ⚠ NOT statistically significant (p = {:.3})",
                stat.pop_pvalue
            );
        }
    } else {
        println!("✗ Classical leads on instantaneous population at ALL time points");
    }

    println!();

    // Check for cumulative AUC advantage
    if let Some(auc_lead_tick) = results.quantum_auc_ever_leads() {
        let stat = results
            .statistics
            .iter()
            .find(|s| s.tick == auc_lead_tick)
            .unwrap();
        println!("✓ QUANTUM LEADS ON CUMULATIVE AUC:");
        println!("  First occurrence: tick {}", auc_lead_tick);
        println!(
            "  At tick {}: Q_AUC={:.0} vs C_AUC={:.0}, Δ={:+.0}",
            auc_lead_tick, stat.quantum_auc_mean, stat.classical_auc_mean, stat.auc_delta
        );
        println!("  p={:.3}, d={:+.2}", stat.auc_pvalue, stat.auc_cohens_d);

        if stat.auc_pvalue < 0.05 {
            println!("  ✓ Statistically significant (p < 0.05)");
        } else {
            println!(
                "  ⚠ NOT statistically significant (p = {:.3})",
                stat.auc_pvalue
            );
        }
    } else {
        println!("✗ Classical leads on cumulative AUC at ALL time points");
    }

    println!();

    // Check for any statistically significant advantage
    if let Some((tick, metric)) = results.first_significant_advantage(0.05) {
        println!("✓ FIRST STATISTICALLY SIGNIFICANT QUANTUM ADVANTAGE:");
        println!("  Tick: {}, Metric: {}", tick, metric);
    } else {
        println!("✗ NO STATISTICALLY SIGNIFICANT QUANTUM ADVANTAGE DETECTED");
        println!("  Classical dominates on all metrics across all time points");
    }

    println!(
        "\n\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║   INTERPRETATION                                                               ║");
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let pop_advantage = results.quantum_ever_leads();
    let auc_advantage = results.quantum_auc_ever_leads();
    let sig_advantage = results.first_significant_advantage(0.05);

    match (pop_advantage, auc_advantage, sig_advantage) {
        (Some(_), Some(_), Some(_)) => {
            println!("Story: GENUINE QUANTUM ADVANTAGE");
            println!("  → Quantum shows both instantaneous and cumulative advantage");
            println!("  → Advantage is statistically significant");
            println!("  → This suggests quantum has a real ecological niche");
        }
        (Some(tick), None, _) => {
            println!("Story: QUANTUM REBOUND (Mid-Game Spike)");
            println!("  → Quantum population spikes at tick {}", tick);
            println!("  → BUT Classical has higher cumulative survival (total organism-hours)");
            println!("  → Classical wins early, accumulates more total biomass");
            println!("  → Quantum finds robust late policies but can't overcome early deficit");
        }
        (None, _, _) => {
            println!("Story: CLASSICAL DOMINANCE");
            println!("  → Classical leads from the start and maintains advantage");
            println!("  → No temporal window where quantum outperforms");
            println!("  → Mutation rate is NOT the bottleneck");
            println!("  → Proceed to Sprint 92.3 (Environmental Pressure)");
        }
        _ => {
            println!("Story: AMBIGUOUS / EDGE CASE");
        }
    }

    println!();
}
