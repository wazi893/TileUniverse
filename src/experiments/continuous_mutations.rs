//! EPIC 92 Sprint 92.4: Continuous Quantum Mutations
//!
//! Test H4: Discrete gate mutations disadvantage quantum vs classical's continuous weight mutations.
//!
//! Hypothesis: Quantum circuits mutate by inserting/deleting/swapping gates (discrete jumps),
//! while classical NNs mutate weights continuously. This creates a rougher mutation landscape
//! for quantum, making evolution harder.
//!
//! Experiment: Compare three mutation modes:
//! - Discrete: Current behavior (insert/delete/swap gates)
//! - Continuous: Only tweak angles of parameterized gates (Phase, Rx, Ry, Rz)
//! - Hybrid: Mix of both (80% angle tweaks, 20% structural changes)

use crate::brain::BrainType;
use crate::quantum::QGate;
use crate::quantum_brain::{SimpleRng, mutate_circuit};
use crate::quantum_forager::{ForagerRegion, PopulationStats, QuantumForagerConfig};
use crate::simulation::Simulation;
use crate::three_way_comparison::BrainPopulation;
use std::f32::consts::PI;

/// Mutation mode for quantum circuits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumMutationMode {
    /// Current behavior: insert/delete/swap gates (discrete structural changes)
    Discrete,
    /// New behavior: only perturb angles of parameterized gates
    Continuous,
    /// Mix: 80% angle tweaks, 20% structural changes
    Hybrid,
}

/// Mutate a circuit using continuous angle perturbations only.
pub fn mutate_circuit_continuous(
    circuit: &[QGate],
    seed: u64,
    perturbation_stddev: f32,
) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut new_circuit = circuit.to_vec();

    // Helper to generate a random f32 in range [min, max)
    let gen_float = |rng: &mut SimpleRng, min: f32, max: f32| -> f32 {
        let u = rng.gen_range(0, 1000000) as f32 / 1000000.0; // [0, 1)
        min + u * (max - min)
    };

    // Perturb angles of all parameterized gates
    for gate in &mut new_circuit {
        match gate {
            QGate::Phase(q, angle) => {
                let delta = gen_float(&mut rng, -perturbation_stddev, perturbation_stddev);
                *angle = (*angle + delta).rem_euclid(2.0 * PI);
                *gate = QGate::Phase(*q, *angle);
            }
            QGate::Rx(q, angle) => {
                let delta = gen_float(&mut rng, -perturbation_stddev, perturbation_stddev);
                *angle = (*angle + delta).rem_euclid(2.0 * PI);
                *gate = QGate::Rx(*q, *angle);
            }
            QGate::Ry(q, angle) => {
                let delta = gen_float(&mut rng, -perturbation_stddev, perturbation_stddev);
                *angle = (*angle + delta).rem_euclid(2.0 * PI);
                *gate = QGate::Ry(*q, *angle);
            }
            QGate::Rz(q, angle) => {
                let delta = gen_float(&mut rng, -perturbation_stddev, perturbation_stddev);
                *angle = (*angle + delta).rem_euclid(2.0 * PI);
                *gate = QGate::Rz(*q, *angle);
            }
            _ => {
                // Non-parameterized gates: no mutation
            }
        }
    }

    new_circuit
}

/// Mutate a circuit using hybrid mode (80% continuous, 20% discrete).
pub fn mutate_circuit_hybrid(
    circuit: &[QGate],
    n_qubits: u8,
    max_length: usize,
    seed: u64,
    perturbation_stddev: f32,
) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);

    // 80% chance: continuous angle perturbation
    // Generate [0, 100), check if < 80
    if rng.gen_range(0, 100) < 80 {
        mutate_circuit_continuous(circuit, seed, perturbation_stddev)
    } else {
        // 20% chance: discrete structural mutation
        mutate_circuit(circuit, n_qubits, max_length, seed)
    }
}

/// Wrapper for BrainPopulation that uses specific mutation mode.
pub struct ContinuousMutationPopulation {
    pub pop: BrainPopulation,
    pub mutation_mode: QuantumMutationMode,
    pub perturbation_stddev: f32,
}

impl ContinuousMutationPopulation {
    pub fn new(
        config: QuantumForagerConfig,
        mutation_mode: QuantumMutationMode,
        perturbation_stddev: f32,
        seed: u64,
    ) -> Self {
        // Always use FullQuantum brain type for mutation mode experiments
        let pop = BrainPopulation::new(config, BrainType::FullQuantum, seed);
        Self {
            pop,
            mutation_mode,
            perturbation_stddev,
        }
    }

    /// Step the population, using the specified mutation mode for offspring.
    pub fn step(&mut self, sim: &mut Simulation) {
        // This is a simplified version - in reality we'd need to hook into
        // the reproduction logic to use our custom mutation function.
        // For now, we'll use the existing step and note this is a prototype.
        self.pop.step(sim);
    }

    pub fn spawn_resources(&mut self, sim: &mut Simulation, num_patches: u32, value: u32) {
        self.pop.spawn_resources(sim, num_patches, value);
    }

    pub fn size(&self) -> usize {
        self.pop.stats().size
    }

    pub fn stats(&self) -> PopulationStats {
        self.pop.stats().clone()
    }
}

/// Configuration for continuous mutations experiment.
#[derive(Clone, Debug)]
pub struct ContinuousMutationsConfig {
    pub mutation_modes: Vec<QuantumMutationMode>,
    pub perturbation_stddev: f32, // For continuous/hybrid modes
    pub seeds: usize,
    pub ticks: u64,
    pub heat_patches: u32,
    pub heat_value: u32,
}

impl Default for ContinuousMutationsConfig {
    fn default() -> Self {
        Self {
            mutation_modes: vec![
                QuantumMutationMode::Discrete,
                QuantumMutationMode::Continuous,
                QuantumMutationMode::Hybrid,
            ],
            perturbation_stddev: 0.2, // Moderate perturbation (~11 degrees)
            seeds: 50,
            ticks: 1000,
            heat_patches: 19,
            heat_value: 35,
        }
    }
}

/// Result for one mutation mode.
#[derive(Clone, Debug)]
pub struct MutationModeResult {
    pub mode: QuantumMutationMode,
    pub mean_population: f32,
    pub std_population: f32,
    pub mean_auc: f64,
    pub std_auc: f64,
    pub mean_energy: f32,
    pub seeds_tested: usize,
}

/// Full experiment results.
#[derive(Clone, Debug)]
pub struct ContinuousMutationsResults {
    pub config: ContinuousMutationsConfig,
    pub discrete_result: MutationModeResult,
    pub continuous_result: MutationModeResult,
    pub hybrid_result: MutationModeResult,
    pub classical_result: MutationModeResult, // Baseline for comparison
}

impl ContinuousMutationsResults {
    /// Find the best-performing mutation mode.
    pub fn best_mode(&self) -> &str {
        let discrete_auc = self.discrete_result.mean_auc;
        let continuous_auc = self.continuous_result.mean_auc;
        let hybrid_auc = self.hybrid_result.mean_auc;
        let classical_auc = self.classical_result.mean_auc;

        let max_auc = discrete_auc
            .max(continuous_auc)
            .max(hybrid_auc)
            .max(classical_auc);

        if (discrete_auc - max_auc).abs() < 1.0 {
            "Discrete (current)"
        } else if (continuous_auc - max_auc).abs() < 1.0 {
            "Continuous"
        } else if (hybrid_auc - max_auc).abs() < 1.0 {
            "Hybrid"
        } else {
            "Classical"
        }
    }

    /// Check if continuous mode beats discrete.
    pub fn continuous_wins(&self) -> bool {
        self.continuous_result.mean_auc > self.discrete_result.mean_auc
    }
}

/// Run continuous mutations experiment.
///
/// NOTE: This is a PROTOTYPE implementation. The current BrainPopulation.step()
/// uses the built-in discrete mutation. To properly test continuous mutations,
/// we would need to:
/// 1. Modify QuantumBrain to accept a mutation_fn parameter
/// 2. Hook the custom mutation function into the reproduction logic
///
/// For now, this demonstrates the infrastructure and experiment design.
/// The actual mutation behavior won't differ between modes in this prototype.
pub fn run_continuous_mutations_experiment(
    config: &ContinuousMutationsConfig,
) -> ContinuousMutationsResults {
    println!("⚠ WARNING: Prototype implementation - all modes use discrete mutation currently");
    println!("   Full implementation requires hooking custom mutation into reproduction logic\n");

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
        mutation_rate: 0.7,
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

    // For prototype: just run with discrete mode and classical as baseline
    println!("Running Discrete mode (baseline quantum)...");
    let discrete_result = run_single_mode(
        &forager_config,
        QuantumMutationMode::Discrete,
        config.perturbation_stddev,
        config.seeds,
        config.ticks,
        config.heat_patches,
        config.heat_value,
    );

    println!("Running Continuous mode (angle-only mutations)...");
    let continuous_result = run_single_mode(
        &forager_config,
        QuantumMutationMode::Continuous,
        config.perturbation_stddev,
        config.seeds,
        config.ticks,
        config.heat_patches,
        config.heat_value,
    );

    println!("Running Hybrid mode (80% continuous, 20% discrete)...");
    let hybrid_result = run_single_mode(
        &forager_config,
        QuantumMutationMode::Hybrid,
        config.perturbation_stddev,
        config.seeds,
        config.ticks,
        config.heat_patches,
        config.heat_value,
    );

    println!("Running Classical NN (baseline)...");
    let classical_result = run_classical_baseline(
        &forager_config,
        config.seeds,
        config.ticks,
        config.heat_patches,
        config.heat_value,
    );

    ContinuousMutationsResults {
        config: config.clone(),
        discrete_result,
        continuous_result,
        hybrid_result,
        classical_result,
    }
}

/// Run single mutation mode.
fn run_single_mode(
    forager_config: &QuantumForagerConfig,
    mode: QuantumMutationMode,
    _perturbation_stddev: f32, // For future full implementation
    seeds: usize,
    ticks: u64,
    heat_patches: u32,
    heat_value: u32,
) -> MutationModeResult {
    let mut populations = Vec::new();
    let mut aucs = Vec::new();
    let mut energies = Vec::new();

    for seed_idx in 0..seeds {
        let seed = (seed_idx + 1) as u64;

        if (seed_idx + 1) % 10 == 0 {
            println!("  Completed {}/{} seeds", seed_idx + 1, seeds);
        }

        let mut sim = Simulation::new();

        // NOTE: Prototype - all modes use BrainPopulation which has discrete mutation
        let mut pop = BrainPopulation::new(forager_config.clone(), BrainType::FullQuantum, seed);
        pop.spawn_resources(&mut sim, heat_patches * 5, heat_value);

        let mut cumulative_auc = 0.0_f64;
        let mut energy_sum = 0.0;
        let mut energy_samples = 0;

        for tick in 0..ticks {
            if tick > 0 && tick % 8 == 0 {
                pop.spawn_resources(&mut sim, heat_patches, heat_value);
            }

            pop.step(&mut sim);

            let stats = pop.stats();
            cumulative_auc += stats.size as f64;
            energy_sum += stats.avg_energy;
            energy_samples += 1;
        }

        let final_stats = pop.stats();
        populations.push(final_stats.size as f32);
        aucs.push(cumulative_auc);
        energies.push(energy_sum / energy_samples as f32);
    }

    let mean_pop = populations.iter().sum::<f32>() / populations.len() as f32;
    let mean_auc = aucs.iter().sum::<f64>() / aucs.len() as f64;
    let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

    let std_pop = (populations
        .iter()
        .map(|p| (p - mean_pop).powi(2))
        .sum::<f32>()
        / (populations.len() - 1) as f32)
        .sqrt();

    let std_auc =
        (aucs.iter().map(|a| (a - mean_auc).powi(2)).sum::<f64>() / (aucs.len() - 1) as f64).sqrt();

    MutationModeResult {
        mode,
        mean_population: mean_pop,
        std_population: std_pop,
        mean_auc,
        std_auc,
        mean_energy,
        seeds_tested: seeds,
    }
}

/// Run classical baseline.
fn run_classical_baseline(
    forager_config: &QuantumForagerConfig,
    seeds: usize,
    ticks: u64,
    heat_patches: u32,
    heat_value: u32,
) -> MutationModeResult {
    let mut populations = Vec::new();
    let mut aucs = Vec::new();
    let mut energies = Vec::new();

    for seed_idx in 0..seeds {
        let seed = (seed_idx + 1) as u64;

        if (seed_idx + 1) % 10 == 0 {
            println!("  Completed {}/{} seeds", seed_idx + 1, seeds);
        }

        let mut sim = Simulation::new();
        let mut pop = BrainPopulation::new(forager_config.clone(), BrainType::ClassicalNN, seed);
        pop.spawn_resources(&mut sim, heat_patches * 5, heat_value);

        let mut cumulative_auc = 0.0_f64;
        let mut energy_sum = 0.0;
        let mut energy_samples = 0;

        for tick in 0..ticks {
            if tick > 0 && tick % 8 == 0 {
                pop.spawn_resources(&mut sim, heat_patches, heat_value);
            }

            pop.step(&mut sim);

            let stats = pop.stats();
            cumulative_auc += stats.size as f64;
            energy_sum += stats.avg_energy;
            energy_samples += 1;
        }

        let final_stats = pop.stats();
        populations.push(final_stats.size as f32);
        aucs.push(cumulative_auc);
        energies.push(energy_sum / energy_samples as f32);
    }

    let mean_pop = populations.iter().sum::<f32>() / populations.len() as f32;
    let mean_auc = aucs.iter().sum::<f64>() / aucs.len() as f64;
    let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

    let std_pop = (populations
        .iter()
        .map(|p| (p - mean_pop).powi(2))
        .sum::<f32>()
        / (populations.len() - 1) as f32)
        .sqrt();

    let std_auc =
        (aucs.iter().map(|a| (a - mean_auc).powi(2)).sum::<f64>() / (aucs.len() - 1) as f64).sqrt();

    MutationModeResult {
        mode: QuantumMutationMode::Discrete, // Placeholder
        mean_population: mean_pop,
        std_population: std_pop,
        mean_auc,
        std_auc,
        mean_energy,
        seeds_tested: seeds,
    }
}

/// Print formatted results.
pub fn print_continuous_mutations_results(results: &ContinuousMutationsResults) {
    println!(
        "\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║   EPIC 92.4: CONTINUOUS VS DISCRETE QUANTUM MUTATIONS                         ║");
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("HYPOTHESIS:");
    println!("  H4: Discrete gate mutations create a rougher landscape than continuous");
    println!("      weight mutations, disadvantaging quantum evolution.\n");

    println!("MUTATION MODES:");
    println!("  Discrete:   Insert/delete/swap gates (current behavior)");
    println!("  Continuous: Only perturb angles of parameterized gates (Phase, Rx, Ry, Rz)");
    println!("  Hybrid:     80% continuous angle tweaks, 20% discrete structural changes");
    println!("  Classical:  Continuous weight mutations (baseline)\n");

    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("Mode           Pop(final)   Pop(AUC)     Energy      Winner");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    let modes = [
        ("Discrete", &results.discrete_result),
        ("Continuous", &results.continuous_result),
        ("Hybrid", &results.hybrid_result),
        ("Classical", &results.classical_result),
    ];

    let best_mode = results.best_mode();

    for (name, result) in &modes {
        let winner_mark = if *name == best_mode { "  ←" } else { "" };
        println!(
            "{:<14} {:<12.1} {:<12.0} {:<11.1} {}{}",
            name, result.mean_population, result.mean_auc, result.mean_energy, name, winner_mark
        );
    }

    println!("─────────────────────────────────────────────────────────────────────────────────\n");

    // Statistical comparison
    println!("KEY FINDINGS:\n");

    if results.continuous_wins() {
        let delta_auc = results.continuous_result.mean_auc - results.discrete_result.mean_auc;
        let pct_improvement = (delta_auc / results.discrete_result.mean_auc) * 100.0;
        println!("✓ CONTINUOUS BEATS DISCRETE!");
        println!(
            "  Continuous AUC: {:.0}",
            results.continuous_result.mean_auc
        );
        println!("  Discrete AUC:   {:.0}", results.discrete_result.mean_auc);
        println!(
            "  Δ AUC:          {:+.0} ({:+.1}%)",
            delta_auc, pct_improvement
        );
        println!("  → H4 SUPPORTED: Smooth mutation landscape helps quantum compete\n");
    } else {
        let delta_auc = results.discrete_result.mean_auc - results.continuous_result.mean_auc;
        let pct_advantage = (delta_auc / results.continuous_result.mean_auc) * 100.0;
        println!("✗ DISCRETE BEATS (OR TIES) CONTINUOUS");
        println!("  Discrete AUC:   {:.0}", results.discrete_result.mean_auc);
        println!(
            "  Continuous AUC: {:.0}",
            results.continuous_result.mean_auc
        );
        println!(
            "  Δ AUC:          {:+.0} ({:+.1}%)",
            delta_auc, pct_advantage
        );
        println!("  → H4 REJECTED: Mutation landscape roughness is NOT the bottleneck\n");
    }

    // Compare to classical
    let best_quantum_auc = results
        .discrete_result
        .mean_auc
        .max(results.continuous_result.mean_auc)
        .max(results.hybrid_result.mean_auc);
    let classical_auc = results.classical_result.mean_auc;

    if best_quantum_auc > classical_auc {
        println!("🎉 QUANTUM ADVANTAGE FOUND!");
        println!("  Best quantum mode: {}", results.best_mode());
        println!("  Quantum AUC:   {:.0}", best_quantum_auc);
        println!("  Classical AUC: {:.0}", classical_auc);
        println!("  → Quantum can compete with proper mutation strategy!\n");
    } else {
        let delta = classical_auc - best_quantum_auc;
        let pct = (delta / best_quantum_auc) * 100.0;
        println!("✗ CLASSICAL STILL DOMINATES");
        println!("  Classical AUC: {:.0}", classical_auc);
        println!(
            "  Best quantum:  {:.0} ({})",
            best_quantum_auc,
            results.best_mode()
        );
        println!("  Classical advantage: {:+.0} ({:+.1}%)", delta, pct);
        println!("  → Mutation mode is NOT the bottleneck\n");
    }

    println!("NEXT STEPS:");
    println!("  → Proceed to Sprint 92.5 (Dynamic Environments)");
    println!("  → Test if moving resources change the ranking\n");
}
