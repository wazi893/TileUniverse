//! EPIC 92 Sprint 92.5: Dynamic Environments
//!
//! Test H5: Static environments favor exploitation (classical), while dynamic
//! environments favor exploration (quantum).
//!
//! Hypothesis: Quantum's "parallel exploration" of policy space provides advantage
//! when resources move, requiring continuous adaptation rather than convergence
//! to a fixed optimal policy.

use crate::brain::BrainType;
use crate::quantum_brain::SimpleRng;
use crate::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use crate::simulation::Simulation;
use crate::three_way_comparison::BrainPopulation;

/// Resource dynamics modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceDynamics {
    /// Static patches (baseline)
    Static,
    /// Patches move randomly every N ticks
    RandomWalk,
    /// Patches deplete and respawn elsewhere
    Respawning,
    /// Patches move in predictable circular patterns
    Periodic,
}

impl ResourceDynamics {
    pub fn display_name(&self) -> &'static str {
        match self {
            ResourceDynamics::Static => "Static",
            ResourceDynamics::RandomWalk => "RandomWalk",
            ResourceDynamics::Respawning => "Respawning",
            ResourceDynamics::Periodic => "Periodic",
        }
    }
}

/// A resource patch with location and value.
#[derive(Clone, Debug)]
pub struct ResourcePatch {
    pub x: u32,
    pub y: u32,
    pub value: u32,
    /// For periodic movement: phase angle
    pub angle: f32,
    /// For periodic movement: radius from center
    pub radius: f32,
}

/// Dynamic environment manager.
pub struct DynamicEnvironment {
    pub patches: Vec<ResourcePatch>,
    pub dynamics: ResourceDynamics,
    pub change_frequency: u64, // Ticks between changes
    pub region: ForagerRegion,
    pub seed: u64,
    pub tick: u64,
}

impl DynamicEnvironment {
    /// Create new dynamic environment.
    pub fn new(
        n_patches: u32,
        patch_value: u32,
        dynamics: ResourceDynamics,
        change_frequency: u64,
        region: ForagerRegion,
        seed: u64,
    ) -> Self {
        let mut rng = SimpleRng::new(seed);
        let mut patches = Vec::new();

        // Initialize patches with random positions
        for _ in 0..n_patches {
            let x = region.x0 + rng.gen_range(0, region.width);
            let y = region.y0 + rng.gen_range(0, region.height);

            // For periodic mode: assign orbit parameters
            let center_x = region.x0 + region.width / 2;
            let center_y = region.y0 + region.height / 2;
            let dx = x as i32 - center_x as i32;
            let dy = y as i32 - center_y as i32;
            let radius = ((dx * dx + dy * dy) as f32).sqrt();
            let angle = (dy as f32).atan2(dx as f32);

            patches.push(ResourcePatch {
                x,
                y,
                value: patch_value,
                angle,
                radius,
            });
        }

        Self {
            patches,
            dynamics,
            change_frequency,
            region,
            seed,
            tick: 0,
        }
    }

    /// Update environment (move patches if needed).
    pub fn update(&mut self, sim: &mut Simulation) {
        self.tick += 1;

        // Check if it's time to update
        if self.tick % self.change_frequency != 0 {
            return;
        }

        let mut rng = SimpleRng::new(self.seed.wrapping_add(self.tick * 12345));

        match self.dynamics {
            ResourceDynamics::Static => {
                // No change
            }
            ResourceDynamics::RandomWalk => {
                // Move each patch by small random displacement
                for patch in &mut self.patches {
                    // Clear old location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, 0);

                    // Random walk: -10..+10 tiles
                    let dx = rng.gen_range(0, 21) as i32 - 10;
                    let dy = rng.gen_range(0, 21) as i32 - 10;

                    patch.x = ((patch.x as i32 + dx)
                        .max(self.region.x0 as i32)
                        .min((self.region.x0 + self.region.width - 1) as i32))
                        as u32;
                    patch.y = ((patch.y as i32 + dy)
                        .max(self.region.y0 as i32)
                        .min((self.region.y0 + self.region.height - 1) as i32))
                        as u32;

                    // Set new location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, patch.value);
                }
            }
            ResourceDynamics::Respawning => {
                // Deplete old patches and spawn at new random locations
                for patch in &mut self.patches {
                    // Clear old location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, 0);

                    // Respawn at random location
                    patch.x = self.region.x0 + rng.gen_range(0, self.region.width);
                    patch.y = self.region.y0 + rng.gen_range(0, self.region.height);

                    // Set new location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, patch.value);
                }
            }
            ResourceDynamics::Periodic => {
                // Move patches in circular orbit around center
                let center_x = self.region.x0 + self.region.width / 2;
                let center_y = self.region.y0 + self.region.height / 2;

                for patch in &mut self.patches {
                    // Clear old location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, 0);

                    // Advance angle by π/8 radians per update
                    patch.angle += std::f32::consts::PI / 8.0;

                    // Compute new position on orbit
                    let new_x = center_x as f32 + patch.radius * patch.angle.cos();
                    let new_y = center_y as f32 + patch.radius * patch.angle.sin();

                    patch.x = (new_x as u32)
                        .max(self.region.x0)
                        .min(self.region.x0 + self.region.width - 1);
                    patch.y = (new_y as u32)
                        .max(self.region.y0)
                        .min(self.region.y0 + self.region.height - 1);

                    // Set new location
                    sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, patch.value);
                }
            }
        }
    }

    /// Apply patches to simulation (initial setup or after static respawn).
    pub fn apply_patches(&self, sim: &mut Simulation) {
        for patch in &self.patches {
            sim.set_heat_field_for_test(patch.x as usize, patch.y as usize, patch.value);
        }
    }
}

/// Configuration for dynamic environments experiment.
#[derive(Clone, Debug)]
pub struct DynamicEnvironmentsConfig {
    pub dynamics_modes: Vec<ResourceDynamics>,
    pub change_frequency: u64, // Ticks between environment changes
    pub seeds: usize,
    pub ticks: u64,
    pub n_patches: u32,
    pub patch_value: u32,
}

impl Default for DynamicEnvironmentsConfig {
    fn default() -> Self {
        Self {
            dynamics_modes: vec![
                ResourceDynamics::Static,
                ResourceDynamics::RandomWalk,
                ResourceDynamics::Respawning,
                ResourceDynamics::Periodic,
            ],
            change_frequency: 150, // Changes every 150 ticks
            seeds: 50,
            ticks: 1000,
            n_patches: 19,
            patch_value: 35,
        }
    }
}

/// Result for one dynamics mode.
#[derive(Clone, Debug)]
pub struct DynamicsModeResult {
    pub dynamics: ResourceDynamics,
    pub quantum_mean_pop: f32,
    pub quantum_std_pop: f32,
    pub quantum_mean_auc: f64,
    pub quantum_std_auc: f64,
    pub classical_mean_pop: f32,
    pub classical_std_pop: f32,
    pub classical_mean_auc: f64,
    pub classical_std_auc: f64,
    pub seeds_tested: usize,
}

impl DynamicsModeResult {
    /// Check if quantum wins on AUC.
    pub fn quantum_wins(&self) -> bool {
        self.quantum_mean_auc > self.classical_mean_auc
    }

    /// Get delta AUC (quantum - classical).
    pub fn delta_auc(&self) -> f64 {
        self.quantum_mean_auc - self.classical_mean_auc
    }

    /// Get percent advantage/disadvantage.
    pub fn percent_delta(&self) -> f64 {
        (self.delta_auc() / self.classical_mean_auc) * 100.0
    }
}

/// Full experiment results.
#[derive(Clone, Debug)]
pub struct DynamicEnvironmentsResults {
    pub config: DynamicEnvironmentsConfig,
    pub results: Vec<DynamicsModeResult>,
}

impl DynamicEnvironmentsResults {
    /// Find best dynamics mode for quantum.
    pub fn best_for_quantum(&self) -> &DynamicsModeResult {
        self.results
            .iter()
            .max_by(|a, b| a.quantum_mean_auc.partial_cmp(&b.quantum_mean_auc).unwrap())
            .unwrap()
    }

    /// Check if quantum ever wins in any dynamics mode.
    pub fn quantum_ever_wins(&self) -> Option<ResourceDynamics> {
        for result in &self.results {
            if result.quantum_wins() {
                return Some(result.dynamics);
            }
        }
        None
    }

    /// Get static baseline for comparison.
    pub fn static_baseline(&self) -> &DynamicsModeResult {
        self.results
            .iter()
            .find(|r| r.dynamics == ResourceDynamics::Static)
            .unwrap()
    }
}

/// Run dynamic environments experiment.
pub fn run_dynamic_environments_experiment(
    config: &DynamicEnvironmentsConfig,
) -> DynamicEnvironmentsResults {
    let mut results = Vec::new();

    for &dynamics in &config.dynamics_modes {
        println!("\n=== Running {} mode ===", dynamics.display_name());

        let mode_result = run_single_dynamics_mode(
            dynamics,
            config.change_frequency,
            config.seeds,
            config.ticks,
            config.n_patches,
            config.patch_value,
        );

        results.push(mode_result);
    }

    DynamicEnvironmentsResults {
        config: config.clone(),
        results,
    }
}

/// Run single dynamics mode.
fn run_single_dynamics_mode(
    dynamics: ResourceDynamics,
    change_frequency: u64,
    seeds: usize,
    ticks: u64,
    n_patches: u32,
    patch_value: u32,
) -> DynamicsModeResult {
    // Run quantum trials
    println!("  Running Quantum trials...");
    let (q_pops, q_aucs) = run_trials(
        BrainType::FullQuantum,
        dynamics,
        change_frequency,
        seeds,
        ticks,
        n_patches,
        patch_value,
    );

    // Run classical trials
    println!("  Running Classical trials...");
    let (c_pops, c_aucs) = run_trials(
        BrainType::ClassicalNN,
        dynamics,
        change_frequency,
        seeds,
        ticks,
        n_patches,
        patch_value,
    );

    // Compute statistics
    let quantum_mean_pop = q_pops.iter().sum::<f32>() / q_pops.len() as f32;
    let classical_mean_pop = c_pops.iter().sum::<f32>() / c_pops.len() as f32;
    let quantum_mean_auc = q_aucs.iter().sum::<f64>() / q_aucs.len() as f64;
    let classical_mean_auc = c_aucs.iter().sum::<f64>() / c_aucs.len() as f64;

    let quantum_std_pop = (q_pops
        .iter()
        .map(|p| (p - quantum_mean_pop).powi(2))
        .sum::<f32>()
        / (q_pops.len() - 1) as f32)
        .sqrt();

    let classical_std_pop = (c_pops
        .iter()
        .map(|p| (p - classical_mean_pop).powi(2))
        .sum::<f32>()
        / (c_pops.len() - 1) as f32)
        .sqrt();

    let quantum_std_auc = (q_aucs
        .iter()
        .map(|a| (a - quantum_mean_auc).powi(2))
        .sum::<f64>()
        / (q_aucs.len() - 1) as f64)
        .sqrt();

    let classical_std_auc = (c_aucs
        .iter()
        .map(|a| (a - classical_mean_auc).powi(2))
        .sum::<f64>()
        / (c_aucs.len() - 1) as f64)
        .sqrt();

    DynamicsModeResult {
        dynamics,
        quantum_mean_pop,
        quantum_std_pop,
        quantum_mean_auc,
        quantum_std_auc,
        classical_mean_pop,
        classical_std_pop,
        classical_mean_auc,
        classical_std_auc,
        seeds_tested: seeds,
    }
}

/// Run trials for one brain type.
fn run_trials(
    brain_type: BrainType,
    dynamics: ResourceDynamics,
    change_frequency: u64,
    seeds: usize,
    ticks: u64,
    n_patches: u32,
    patch_value: u32,
) -> (Vec<f32>, Vec<f64>) {
    let mut populations = Vec::new();
    let mut aucs = Vec::new();

    for seed_idx in 0..seeds {
        let seed = (seed_idx + 1) as u64;

        if (seed_idx + 1) % 10 == 0 {
            println!("    Completed {}/{} seeds", seed_idx + 1, seeds);
        }

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

        let mut sim = Simulation::new();
        let mut pop = BrainPopulation::new(forager_config.clone(), brain_type, seed);

        // Create dynamic environment
        let mut env = DynamicEnvironment::new(
            n_patches,
            patch_value,
            dynamics,
            change_frequency,
            forager_config.region,
            seed,
        );

        // Initial resource setup
        env.apply_patches(&mut sim);

        let mut cumulative_auc = 0.0_f64;

        for _tick in 0..ticks {
            // Update environment (move patches if needed)
            env.update(&mut sim);

            // Step population
            pop.step(&mut sim);

            // Track metrics
            let stats = pop.stats();
            cumulative_auc += stats.size as f64;
        }

        let final_stats = pop.stats();
        populations.push(final_stats.size as f32);
        aucs.push(cumulative_auc);
    }

    (populations, aucs)
}

/// Print formatted results.
pub fn print_dynamic_environments_results(results: &DynamicEnvironmentsResults) {
    println!(
        "\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║   EPIC 92.5: DYNAMIC ENVIRONMENTS - THE FINAL TEST                            ║");
    println!(
        "╚════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("HYPOTHESIS:");
    println!(
        "  H5: Dynamic environments favor exploration (quantum) over exploitation (classical)\n"
    );

    println!("ENVIRONMENT MODES:");
    println!("  Static:      Resources never move (baseline)");
    println!(
        "  RandomWalk:  Patches drift randomly every {} ticks (±10 tiles)",
        results.config.change_frequency
    );
    println!(
        "  Respawning:  Patches deplete and respawn elsewhere every {} ticks",
        results.config.change_frequency
    );
    println!("  Periodic:    Patches orbit in predictable circular patterns\n");

    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("Dynamics       Q_Pop    Q_AUC      C_Pop    C_AUC      Δ_AUC       %Δ      Winner");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    for result in &results.results {
        let winner = if result.quantum_wins() {
            "Quantum"
        } else {
            "Classical"
        };
        let winner_mark = if result.quantum_wins() { "  ←" } else { "" };

        println!(
            "{:<14} {:<8.1} {:<10.0} {:<8.1} {:<10.0} {:>+10.0} {:>+7.1}%  {}{}",
            result.dynamics.display_name(),
            result.quantum_mean_pop,
            result.quantum_mean_auc,
            result.classical_mean_pop,
            result.classical_mean_auc,
            result.delta_auc(),
            result.percent_delta(),
            winner,
            winner_mark
        );
    }

    println!("─────────────────────────────────────────────────────────────────────────────────\n");

    // Analysis
    println!("KEY FINDINGS:\n");

    if let Some(winning_mode) = results.quantum_ever_wins() {
        println!("✓ QUANTUM ADVANTAGE FOUND!");
        let winner = results
            .results
            .iter()
            .find(|r| r.dynamics == winning_mode)
            .unwrap();
        println!("  Mode: {}", winning_mode.display_name());
        println!("  Quantum AUC: {:.0}", winner.quantum_mean_auc);
        println!("  Classical AUC: {:.0}", winner.classical_mean_auc);
        println!(
            "  Advantage: {:+.0} ({:+.1}%)",
            winner.delta_auc(),
            winner.percent_delta()
        );
        println!();
        println!("  → H5 SUPPORTED: Dynamic environments favor quantum exploration!");
        println!("  → This is the ecological niche where quantum thrives!\n");
    } else {
        println!("✗ NO QUANTUM ADVANTAGE IN ANY DYNAMICS MODE");
        println!("  Classical dominates across:");
        for result in &results.results {
            println!(
                "    {}: Classical {:+.1}%",
                result.dynamics.display_name(),
                -result.percent_delta()
            );
        }
        println!();
        println!("  → H5 REJECTED: Dynamics mode is NOT the differentiator");
        println!("  → Classical wins in both static AND dynamic environments\n");
    }

    // Compare to static baseline
    let static_result = results.static_baseline();
    let best_dynamic = results
        .results
        .iter()
        .filter(|r| r.dynamics != ResourceDynamics::Static)
        .max_by(|a, b| a.quantum_mean_auc.partial_cmp(&b.quantum_mean_auc).unwrap())
        .unwrap();

    println!("STATIC VS DYNAMIC COMPARISON:");
    println!(
        "  Static Quantum AUC:  {:.0}",
        static_result.quantum_mean_auc
    );
    println!(
        "  Best Dynamic Q AUC:  {:.0} ({})",
        best_dynamic.quantum_mean_auc,
        best_dynamic.dynamics.display_name()
    );
    println!(
        "  Improvement:         {:+.0} ({:+.1}%)\n",
        best_dynamic.quantum_mean_auc - static_result.quantum_mean_auc,
        ((best_dynamic.quantum_mean_auc - static_result.quantum_mean_auc)
            / static_result.quantum_mean_auc)
            * 100.0
    );

    if best_dynamic.quantum_mean_auc > static_result.quantum_mean_auc {
        println!("  → Dynamics HELP quantum (but may not beat classical)");
    } else {
        println!("  → Dynamics HURT quantum (makes it worse)");
    }

    println!();
}
