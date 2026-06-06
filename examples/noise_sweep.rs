//! Noise Robustness Experiment
//!
//! Determines if quantum advantage survives under realistic noise conditions,
//! and finds the threshold where it disappears.
//!
//! Run with: cargo run --example noise_sweep --release

use engine::constrained_circuits::{
    CircuitConstraints, entanglement_fraction, mutate_circuit_constrained,
    random_circuit_constrained,
};
use engine::noise_model::NoiseConfig;
use engine::quantum::QGate;
use engine::quantum_brain::{
    decode_action, encode_as_rotations, run_circuit_noisy, sense_local_fields,
};
use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::simulation::Simulation;

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   NOISE ROBUSTNESS: Does Quantum Advantage Survive?        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let noise_levels = vec![0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30];
    let ticks = 1000;
    let replications = 5;
    let base_seed = 42u64;

    // Moderate scarcity configuration (goldilocks zone)
    let config = QuantumForagerConfig {
        initial_population: 30,
        initial_energy: 100.0,
        n_qubits: 4,
        initial_circuit_length: 8,
        max_circuit_length: 25,
        movement_cost: 0.5,
        metabolism_cost: 0.25,
        reproduction_threshold: 140.0,
        reproduction_cost: 55.0,
        mutation_rate: 0.85,
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

    let heat_patches = 18;
    let heat_value = 32;

    println!("Configuration:");
    println!("  - Ticks: {}", ticks);
    println!("  - Replications per noise level: {}", replications);
    println!(
        "  - Resource scarcity: {} patches, value {}",
        heat_patches, heat_value
    );
    println!(
        "  - Noise levels: {:?}",
        noise_levels
            .iter()
            .map(|n| format!("{:.0}%", n * 100.0))
            .collect::<Vec<_>>()
    );
    println!("  - Noise is APPLIED to quantum circuit execution");
    println!();

    let mut results = Vec::new();

    for &noise_level in &noise_levels {
        print!("Testing noise {:.0}%... ", noise_level * 100.0);
        std::io::stdout().flush().unwrap();

        let noise_config = if noise_level == 0.0 {
            NoiseConfig::perfect()
        } else {
            NoiseConfig::scaled(noise_level)
        };

        let mut q_pops = Vec::new();
        let mut c_pops = Vec::new();
        let mut q_extinctions = 0;
        let mut c_extinctions = 0;
        let mut q_qness_sum = 0.0f32;

        for rep in 0..replications {
            let seed = base_seed.wrapping_add(rep as u64 * 1000);

            // Run quantum with noise
            let (q_pop, q_extinct, q_qness) = run_noisy_population(
                &config,
                CircuitConstraints::full_quantum(),
                &noise_config,
                ticks,
                heat_patches,
                heat_value,
                seed,
            );
            q_pops.push(q_pop as f32);
            if q_extinct {
                q_extinctions += 1;
            }
            q_qness_sum += q_qness;

            // Run classical with noise
            let (c_pop, c_extinct, _) = run_noisy_population(
                &config,
                CircuitConstraints::classical_only(),
                &noise_config,
                ticks,
                heat_patches,
                heat_value,
                seed,
            );
            c_pops.push(c_pop as f32);
            if c_extinct {
                c_extinctions += 1;
            }
        }

        let q_mean: f32 = q_pops.iter().sum::<f32>() / replications as f32;
        let c_mean: f32 = c_pops.iter().sum::<f32>() / replications as f32;
        let advantage = q_mean - c_mean;
        let q_ext_rate = (q_extinctions as f32 / replications as f32) * 100.0;
        let c_ext_rate = (c_extinctions as f32 / replications as f32) * 100.0;
        let q_qness_avg = q_qness_sum / replications as f32;

        let winner = if advantage > 10.0 {
            "Q+"
        } else if advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        results.push(NoiseResult {
            noise_level,
            q_mean,
            c_mean,
            advantage,
            q_ext_rate,
            c_ext_rate,
            q_qness: q_qness_avg,
            winner: winner.to_string(),
        });

        println!(
            "Q={:.1} C={:.1} Δ={:+.1} {}",
            q_mean, c_mean, advantage, winner
        );
    }

    // Print results table
    println!("\n======================================================================");
    println!("  RESULTS TABLE");
    println!("======================================================================\n");

    println!("Noise    Q-Pop    C-Pop    Δ Adv    Q-Ext    C-Ext    Q-ness   Winner");
    println!("────────────────────────────────────────────────────────────────────────");

    for r in &results {
        println!(
            "{:>5.1}%   {:>6.1}   {:>6.1}   {:>+6.1}    {:>4.0}%    {:>4.0}%    {:>5.2}    {}",
            r.noise_level * 100.0,
            r.q_mean,
            r.c_mean,
            r.advantage,
            r.q_ext_rate,
            r.c_ext_rate,
            r.q_qness,
            r.winner,
        );
    }

    // Find crossover point
    println!("\n======================================================================");
    println!("  ANALYSIS");
    println!("======================================================================\n");

    let mut crossover = None;
    for i in 1..results.len() {
        if results[i - 1].advantage > 0.0 && results[i].advantage <= 0.0 {
            crossover = Some((results[i - 1].noise_level, results[i].noise_level));
            break;
        }
    }

    if let Some((before, after)) = crossover {
        println!(
            "CROSSOVER POINT: Quantum advantage disappears between {:.0}% and {:.0}% noise",
            before * 100.0,
            after * 100.0
        );
    } else if results.last().map(|r| r.advantage > 0.0).unwrap_or(false) {
        println!(
            "ROBUST: Quantum advantage persists even at {:.0}% noise!",
            noise_levels.last().unwrap() * 100.0
        );
    } else if results.first().map(|r| r.advantage <= 0.0).unwrap_or(false) {
        println!("NO ADVANTAGE: Quantum never showed advantage even at 0% noise");
    }

    // Calculate degradation rate
    let perfect_adv = results.first().map(|r| r.advantage).unwrap_or(0.0);
    let worst_adv = results.last().map(|r| r.advantage).unwrap_or(0.0);
    let degradation = perfect_adv - worst_adv;

    println!();
    println!("Perfect noise advantage:  {:+.1}", perfect_adv);
    println!("Worst noise advantage:    {:+.1}", worst_adv);
    println!("Total degradation:        {:.1}", degradation);

    // ASCII chart
    println!("\n======================================================================");
    println!("  QUANTUM ADVANTAGE vs NOISE (ASCII Chart)");
    println!("======================================================================\n");

    print_advantage_chart(&results);

    // Export CSV
    let csv_path = "noise_sweep_results.csv";
    if let Ok(mut file) = File::create(csv_path) {
        writeln!(
            file,
            "noise_pct,q_pop,c_pop,advantage,q_ext_pct,c_ext_pct,q_qness,winner"
        )
        .unwrap();
        for r in &results {
            writeln!(
                file,
                "{:.1},{:.1},{:.1},{:.1},{:.0},{:.0},{:.3},{}",
                r.noise_level * 100.0,
                r.q_mean,
                r.c_mean,
                r.advantage,
                r.q_ext_rate,
                r.c_ext_rate,
                r.q_qness,
                r.winner,
            )
            .unwrap();
        }
        println!("\nResults exported to {}", csv_path);
    }
}

#[derive(Debug)]
struct NoiseResult {
    noise_level: f32,
    q_mean: f32,
    c_mean: f32,
    advantage: f32,
    q_ext_rate: f32,
    c_ext_rate: f32,
    q_qness: f32,
    winner: String,
}

/// Organism with noisy decision-making
#[derive(Clone)]
struct NoisyForager {
    id: u64,
    generation: u32,
    x: u32,
    y: u32,
    energy: f32,
    genome: Vec<QGate>,
    n_qubits: u8,
    age: u32,
    decision_seed: u64,
    constraints: CircuitConstraints,
}

/// Run a population with noise applied to every decision
fn run_noisy_population(
    config: &QuantumForagerConfig,
    constraints: CircuitConstraints,
    noise_config: &NoiseConfig,
    ticks: u32,
    heat_patches: u32,
    heat_value: u32,
    seed: u64,
) -> (usize, bool, f32) {
    let mut sim = Simulation::new();
    let mut foragers: HashMap<u64, NoisyForager> = HashMap::new();
    let mut next_id = 0u64;

    let region = &config.region;

    // Create initial population
    for i in 0..config.initial_population {
        let fseed = seed.wrapping_add(i as u64 * 1337);
        let x = region.x0 + (fseed.wrapping_mul(0x123) as u32 % region.width);
        let y = region.y0 + (fseed.wrapping_mul(0x456) as u32 % region.height);

        let genome = random_circuit_constrained(
            config.n_qubits,
            config.initial_circuit_length,
            &constraints,
            fseed,
        );

        foragers.insert(
            next_id,
            NoisyForager {
                id: next_id,
                generation: 0,
                x,
                y,
                energy: config.initial_energy,
                genome,
                n_qubits: config.n_qubits,
                age: 0,
                decision_seed: fseed.wrapping_mul(0x9e3779b97f4a7c15),
                constraints: constraints.clone(),
            },
        );
        next_id += 1;
    }

    // Seed initial resources
    for i in 0..heat_patches {
        let rx = region.x0
            + ((seed.wrapping_mul(0x123456789).wrapping_add(i as u64 * 17)) as u32 % region.width);
        let ry = region.y0
            + ((seed.wrapping_mul(0x987654321).wrapping_add(i as u64 * 31)) as u32 % region.height);
        sim.set_heat_field_for_test(rx as usize, ry as usize, heat_value);
    }

    // Run simulation
    for tick in 0..ticks {
        // Respawn resources
        if tick % 8 == 0 {
            for i in 0..heat_patches {
                let rseed = seed.wrapping_add(tick as u64);
                let rx = region.x0
                    + ((rseed
                        .wrapping_mul(0x123456789)
                        .wrapping_add(tick as u64 * 7 + i as u64 * 17))
                        as u32
                        % region.width);
                let ry = region.y0
                    + ((rseed
                        .wrapping_mul(0x987654321)
                        .wrapping_add(tick as u64 * 13 + i as u64 * 31))
                        as u32
                        % region.height);
                sim.set_heat_field_for_test(rx as usize, ry as usize, heat_value);
            }
        }

        let mut deaths = Vec::new();
        let mut births = Vec::new();
        let current_pop = foragers.len();

        let ids: Vec<u64> = foragers.keys().copied().collect();
        for id in ids {
            let f = foragers.get_mut(&id).unwrap();

            // Age and metabolism
            f.age += 1;
            f.energy -= config.metabolism_cost;

            if f.energy <= 0.0 {
                deaths.push(id);
                continue;
            }

            // Sense environment
            let readings = sense_local_fields(&sim, f.x, f.y);

            // Make NOISY decision
            let sensor_gates = encode_as_rotations(&readings, f.n_qubits);
            let decision_seed = f.decision_seed.wrapping_add(f.age as u64);
            let measurement = run_circuit_noisy(
                &f.genome,
                &sensor_gates,
                f.n_qubits,
                decision_seed,
                noise_config,
            );
            let action = decode_action(measurement);

            // Move
            let new_x = (f.x as i32 + action.dx)
                .clamp(region.x0 as i32, (region.x0 + region.width - 1) as i32)
                as u32;
            let new_y = (f.y as i32 + action.dy)
                .clamp(region.y0 as i32, (region.y0 + region.height - 1) as i32)
                as u32;
            f.x = new_x;
            f.y = new_y;
            f.energy -= config.movement_cost;

            // Consume resources
            if config.consume_resources {
                let heat = sim.get_heat_field_for_test(f.x as usize, f.y as usize);
                if heat > 0 {
                    let consumed = (heat as f32 * config.heat_energy_factor).min(heat as f32);
                    f.energy += consumed;
                    let remaining = heat.saturating_sub(consumed as u32);
                    sim.set_heat_field_for_test(f.x as usize, f.y as usize, remaining);
                }
            }

            // Reproduce
            if f.energy >= config.reproduction_threshold
                && current_pop + births.len() < config.max_population
            {
                f.energy -= config.reproduction_cost;

                let offspring_genome = mutate_circuit_constrained(
                    &f.genome,
                    f.n_qubits,
                    &f.constraints,
                    decision_seed.wrapping_mul(0x1234567890abcdef),
                );

                births.push(NoisyForager {
                    id: next_id,
                    generation: f.generation + 1,
                    x: f.x,
                    y: f.y,
                    energy: config.reproduction_cost,
                    genome: offspring_genome,
                    n_qubits: f.n_qubits,
                    age: 0,
                    decision_seed: decision_seed.wrapping_mul(0x9e3779b97f4a7c15),
                    constraints: f.constraints.clone(),
                });
                next_id += 1;
            }
        }

        // Apply deaths and births
        for id in deaths {
            foragers.remove(&id);
        }
        for f in births {
            if foragers.len() < config.max_population {
                foragers.insert(f.id, f);
            }
        }
    }

    // Calculate final stats
    let pop = foragers.len();
    let extinct = pop == 0;
    let qness = if pop == 0 {
        0.0
    } else {
        let total: f32 = foragers
            .values()
            .map(|f| entanglement_fraction(&f.genome))
            .sum();
        total / pop as f32
    };

    (pop, extinct, qness)
}

fn print_advantage_chart(results: &[NoiseResult]) {
    let height = 12;

    let max_adv = results
        .iter()
        .map(|r| r.advantage)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_adv = results
        .iter()
        .map(|r| r.advantage)
        .fold(f32::INFINITY, f32::min);
    let range = (max_adv - min_adv).max(1.0);

    let zero_row = if max_adv > 0.0 && min_adv < 0.0 {
        ((max_adv / range) * height as f32) as usize
    } else if max_adv <= 0.0 {
        0
    } else {
        height
    };

    for row in 0..height {
        let threshold = max_adv - (row as f32 / height as f32) * range;

        if row == zero_row {
            print!("{:>+6.0} |", 0.0);
        } else {
            print!("{:>+6.0} |", threshold);
        }

        for r in results {
            let bar_height = ((r.advantage - min_adv) / range * height as f32) as usize;
            let current_row = height - row - 1;

            if current_row < bar_height {
                if r.advantage > 0.0 {
                    print!(" ## ");
                } else {
                    print!(" -- ");
                }
            } else if row == zero_row {
                print!("----");
            } else {
                print!("    ");
            }
        }
        println!();
    }

    print!("       +");
    for _ in results {
        print!("----");
    }
    println!();

    print!("        ");
    for r in results {
        print!("{:>3.0}%", r.noise_level * 100.0);
    }
    println!();
    println!();
    println!("Legend: ## = Quantum advantage, -- = Classical advantage");
}
