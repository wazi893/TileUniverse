//! EPIC 92 Sprint 92.1: Mutation Rate Sweep
//!
//! Test H1: 70% mutation rate is too harsh for quantum circuits.
//! Sweeps from 0.1 to 0.9 to find optimal rate for each brain type.

use crate::brain::BrainType;
use crate::three_way_comparison::{GroupResult, ThreeWayConfig, run_three_way_comparison};

/// Configuration for mutation rate sweep.
#[derive(Clone, Debug)]
pub struct MutationSweepConfig {
    pub mutation_rates: Vec<f32>,
    pub seeds_per_rate: usize,
    pub ticks: u64,
    pub heat_patches: u32,
    pub heat_value: u32,
    pub brain_types: Vec<BrainType>,
}

impl Default for MutationSweepConfig {
    fn default() -> Self {
        Self {
            mutation_rates: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            seeds_per_rate: 50,
            ticks: 1000,
            heat_patches: 19,
            heat_value: 35,
            brain_types: vec![
                BrainType::FullQuantum,
                BrainType::SeparableQuantum,
                BrainType::ClassicalNN,
            ],
        }
    }
}

/// Result for one brain type at one mutation rate.
#[derive(Clone, Debug)]
pub struct MutationRateResult {
    pub brain_type: BrainType,
    pub mutation_rate: f32,
    pub group_result: GroupResult,
}

/// Full sweep results across all rates and brain types.
#[derive(Clone, Debug)]
pub struct MutationSweepResults {
    pub config: MutationSweepConfig,
    pub results: Vec<MutationRateResult>,
}

impl MutationSweepResults {
    /// Get results for a specific brain type.
    pub fn for_brain(&self, brain_type: BrainType) -> Vec<&MutationRateResult> {
        self.results
            .iter()
            .filter(|r| r.brain_type == brain_type)
            .collect()
    }

    /// Find optimal mutation rate for a brain type (by population AUC).
    pub fn optimal_rate(&self, brain_type: BrainType) -> (f32, f64) {
        let results = self.for_brain(brain_type);
        let (rate, auc) = results
            .iter()
            .map(|r| (r.mutation_rate, r.group_result.mean_population_auc))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        (rate, auc)
    }

    /// Check if quantum ever beats classical at any mutation rate.
    pub fn quantum_wins_anywhere(&self) -> Option<f32> {
        let quantum_results = self.for_brain(BrainType::FullQuantum);
        let classical_results = self.for_brain(BrainType::ClassicalNN);

        for (q, c) in quantum_results.iter().zip(classical_results.iter()) {
            if q.group_result.mean_population_auc > c.group_result.mean_population_auc {
                return Some(q.mutation_rate);
            }
        }
        None
    }
}

/// Run mutation rate sweep.
pub fn run_mutation_sweep(config: &MutationSweepConfig) -> MutationSweepResults {
    let mut results = Vec::new();

    for &mutation_rate in &config.mutation_rates {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!(
            "║   MUTATION RATE: {:.1}                                       ║",
            mutation_rate
        );
        println!("╚════════════════════════════════════════════════════════════╝");

        // Configure three-way comparison with this mutation rate
        let mut three_way_config = ThreeWayConfig::default();
        three_way_config.replications = config.seeds_per_rate;
        three_way_config.max_ticks = config.ticks;
        three_way_config.heat_patches = config.heat_patches;
        three_way_config.heat_value = config.heat_value;
        three_way_config.brain_types = config.brain_types.clone();

        // Set mutation rate for all brain types
        three_way_config.forager_config.mutation_rate = mutation_rate;

        let comparison_results = run_three_way_comparison(&three_way_config);

        // Extract group results for each brain type
        for &brain_type in &config.brain_types {
            let group_result = match brain_type {
                BrainType::FullQuantum => comparison_results.full_quantum.clone(),
                BrainType::SeparableQuantum => comparison_results.separable_quantum.clone(),
                BrainType::ClassicalNN => comparison_results.classical_nn.clone(),
                BrainType::HandicappedClassicalNN
                | BrainType::SpikingNN
                | BrainType::QuboBrain
                | BrainType::HybridExploration => {
                    // Handle handicapped NN, SpikingNN, QuboBrain, and HybridExploration if present
                    // For now, skip or use classical_nn as fallback
                    continue;
                }
            };

            results.push(MutationRateResult {
                brain_type,
                mutation_rate,
                group_result,
            });
        }

        // Print quick summary
        let fq = results
            .iter()
            .find(|r| r.brain_type == BrainType::FullQuantum && r.mutation_rate == mutation_rate)
            .unwrap();
        let sq = results
            .iter()
            .find(|r| {
                r.brain_type == BrainType::SeparableQuantum && r.mutation_rate == mutation_rate
            })
            .unwrap();
        let cn = results
            .iter()
            .find(|r| r.brain_type == BrainType::ClassicalNN && r.mutation_rate == mutation_rate)
            .unwrap();

        println!("\nQuick Summary (rate={:.1}):", mutation_rate);
        println!(
            "  Full Q:      {:.1} (AUC: {:.0})",
            fq.group_result.mean_population, fq.group_result.mean_population_auc
        );
        println!(
            "  Separable Q: {:.1} (AUC: {:.0})",
            sq.group_result.mean_population, sq.group_result.mean_population_auc
        );
        println!(
            "  Classical:   {:.1} (AUC: {:.0})",
            cn.group_result.mean_population, cn.group_result.mean_population_auc
        );
    }

    MutationSweepResults {
        config: config.clone(),
        results,
    }
}

/// Print formatted results table.
pub fn print_mutation_sweep_table(results: &MutationSweepResults) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   MUTATION RATE SWEEP: Finding Quantum's Sweet Spot          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Mutation   Full Q    Sep Q    Classical   FQ vs CN   Winner");
    println!("Rate       Pop(AUC)  Pop(AUC) Pop(AUC)    Δ AUC      (by AUC)");
    println!("────────────────────────────────────────────────────────────────");

    for &rate in &results.config.mutation_rates {
        let fq = results
            .results
            .iter()
            .find(|r| r.brain_type == BrainType::FullQuantum && r.mutation_rate == rate)
            .unwrap();
        let sq = results
            .results
            .iter()
            .find(|r| r.brain_type == BrainType::SeparableQuantum && r.mutation_rate == rate)
            .unwrap();
        let cn = results
            .results
            .iter()
            .find(|r| r.brain_type == BrainType::ClassicalNN && r.mutation_rate == rate)
            .unwrap();

        let delta_auc = fq.group_result.mean_population_auc - cn.group_result.mean_population_auc;
        let winner = if fq.group_result.mean_population_auc > cn.group_result.mean_population_auc {
            "Quantum!"
        } else {
            "Classical"
        };

        println!(
            "{:.1}      {:.1}({:.0})  {:.1}({:.0})  {:.1}({:.0})   {:+.0}      {}",
            rate,
            fq.group_result.mean_population,
            fq.group_result.mean_population_auc / 1000.0,
            sq.group_result.mean_population,
            sq.group_result.mean_population_auc / 1000.0,
            cn.group_result.mean_population,
            cn.group_result.mean_population_auc / 1000.0,
            delta_auc / 1000.0,
            winner
        );
    }

    println!("\n────────────────────────────────────────────────────────────────");

    // Print optimal rates
    for &brain_type in &results.config.brain_types {
        let (optimal_rate, optimal_auc) = results.optimal_rate(brain_type);
        println!(
            "Optimal for {}: rate={:.1} (AUC={:.0})",
            brain_type.display_name(),
            optimal_rate,
            optimal_auc
        );
    }

    // Check for crossover
    if let Some(crossover_rate) = results.quantum_wins_anywhere() {
        println!(
            "\n🎉 QUANTUM ADVANTAGE FOUND at mutation rate {:.1}!",
            crossover_rate
        );
    } else {
        println!("\n📊 Classical dominates across all mutation rates tested.");
    }
}
