//! EPIC 96.5: Neural Network Architecture Scaling Experiment
//!
//! Tests whether larger NN architectures (16×16, 32×32, 64×64) can:
//! 1. Evolve successfully (convergence, fitness)
//! 2. Run fast on GPU (esp. with WMMA tensor cores)
//! 3. Produce more complex behaviors
//!
//! This determines if WMMA optimization is viable.

use engine::configurable_nn::{ConfigurableBrain, NNConfig};
use engine::quantum::QRng;
use engine::rich_sensors::extract_sensors;
use std::time::Instant;

/// Evolution metrics for one experiment
#[derive(Debug, Clone)]
struct EvolutionMetrics {
    config_name: &'static str,
    config: NNConfig,
    generations_to_90_pct: usize,
    final_fitness_mean: f64,
    final_fitness_max: f64,
    final_fitness_std: f64,
    converged: bool,
    time_per_generation_ms: f64,
}

/// Fitness function: organisms that move toward heat and away from charge do better
fn evaluate_fitness(
    brain: &ConfigurableBrain,
    world_heat: &[f32],
    world_charge: &[f32],
    width: usize,
    height: usize,
    rng: &mut QRng,
) -> f64 {
    const TRIALS: usize = 10;
    let mut total_fitness = 0.0f64;

    for _ in 0..TRIALS {
        // Random starting position
        let x = (rng.next_f32() * width as f32) as usize % width;
        let y = (rng.next_f32() * height as f32) as usize % height;

        // Extract sensors
        let sensors = extract_sensors(world_heat, world_charge, x, y, width, height, brain.config);

        // Get brain's move
        let mv = brain.select_move(&sensors);

        // Calculate new position
        let (nx, ny) = match mv {
            engine::classical_brain::Move::Up => (x, (y + height - 1) % height),
            engine::classical_brain::Move::Down => (x, (y + 1) % height),
            engine::classical_brain::Move::Left => ((x + width - 1) % width, y),
            engine::classical_brain::Move::Right => ((x + 1) % width, y),
            engine::classical_brain::Move::Stay => (x, y),
        };

        // Fitness = heat_improvement - charge_penalty
        let old_heat = world_heat[y * width + x];
        let new_heat = world_heat[ny * width + nx];
        let old_charge = world_charge[y * width + x].abs();
        let new_charge = world_charge[ny * width + nx].abs();

        let heat_improvement = (new_heat - old_heat) as f64;
        let charge_penalty = (new_charge - old_charge) as f64;

        total_fitness += heat_improvement - charge_penalty;
    }

    total_fitness / TRIALS as f64
}

/// Run evolution experiment for one configuration
fn run_evolution(
    config: NNConfig,
    config_name: &'static str,
    n_generations: usize,
    population_size: usize,
) -> EvolutionMetrics {
    println!("\n┌─────────────────────────────────────────────────────┐");
    println!("│ Testing: {:<43} │", config_name);
    println!(
        "│ Sensors: {:>3}  Hidden: {:>3}  Outputs: {:>3}           │",
        config.n_sensors, config.n_hidden, config.n_outputs
    );
    println!(
        "│ Total params: {:>5}                               │",
        config.total_params()
    );
    println!(
        "│ WMMA efficiency: {:>5.1}%                           │",
        config.wmma_efficiency()
    );
    println!("└─────────────────────────────────────────────────────┘");

    // Create simple world
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;

    let mut world_heat = vec![0.0f32; WIDTH * HEIGHT];
    let mut world_charge = vec![0.0f32; WIDTH * HEIGHT];

    // Heat gradient: higher in center
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let dx = (x as f32 - WIDTH as f32 / 2.0) / (WIDTH as f32 / 2.0);
            let dy = (y as f32 - HEIGHT as f32 / 2.0) / (HEIGHT as f32 / 2.0);
            let dist = (dx * dx + dy * dy).sqrt();
            world_heat[y * WIDTH + x] = (1.0 - dist).max(0.0);
        }
    }

    // Charge: random noise (organisms should avoid)
    let mut rng = QRng::new(12345);
    for i in 0..WIDTH * HEIGHT {
        world_charge[i] = (rng.next_f32() - 0.5) * 0.5;
    }

    // Initialize population
    let mut population: Vec<ConfigurableBrain> = (0..population_size)
        .map(|i| ConfigurableBrain::random(config, i as u64))
        .collect();

    let mut fitness_history = Vec::new();
    let start_time = Instant::now();

    for generation in 0..n_generations {
        let gen_start = Instant::now();

        // Evaluate fitness
        let mut fitness_scores: Vec<(usize, f64)> = population
            .iter()
            .enumerate()
            .map(|(i, brain)| {
                let mut eval_rng = QRng::new((generation * population_size + i) as u64);
                let fitness = evaluate_fitness(
                    brain,
                    &world_heat,
                    &world_charge,
                    WIDTH,
                    HEIGHT,
                    &mut eval_rng,
                );
                (i, fitness)
            })
            .collect();

        // Sort by fitness (descending)
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate statistics
        let top_10_pct_count = population_size / 10;
        let mean_fitness: f64 = fitness_scores[0..top_10_pct_count]
            .iter()
            .map(|(_, f)| f)
            .sum::<f64>()
            / top_10_pct_count as f64;

        let max_fitness = fitness_scores[0].1;

        // Standard deviation
        let variance: f64 = fitness_scores[0..top_10_pct_count]
            .iter()
            .map(|(_, f)| {
                let diff = f - mean_fitness;
                diff * diff
            })
            .sum::<f64>()
            / top_10_pct_count as f64;
        let std_dev = variance.sqrt();

        fitness_history.push((generation, mean_fitness, max_fitness, std_dev));

        let gen_time = gen_start.elapsed();

        if generation % 10 == 0 || generation == n_generations - 1 {
            println!(
                "Gen {:>3}: mean={:>7.4}  max={:>7.4}  std={:>6.4}  time={:>6.2}ms",
                generation,
                mean_fitness,
                max_fitness,
                std_dev,
                gen_time.as_secs_f64() * 1000.0
            );
        }

        if generation == n_generations - 1 {
            break;
        }

        // Selection: top 10%
        let survivors: Vec<_> = fitness_scores[0..top_10_pct_count]
            .iter()
            .map(|(i, _)| population[*i].clone())
            .collect();

        // Reproduction: mutation + crossover
        let mut next_gen = Vec::with_capacity(population_size);
        let mut repro_rng = QRng::new((generation + 1) as u64 * 999983);

        for _i in 0..population_size {
            if repro_rng.next_f32() < 0.1 {
                // 10% crossover
                let parent1_idx =
                    (repro_rng.next_f32() * survivors.len() as f32) as usize % survivors.len();
                let parent2_idx =
                    (repro_rng.next_f32() * survivors.len() as f32) as usize % survivors.len();
                let child =
                    survivors[parent1_idx].crossover(&survivors[parent2_idx], &mut repro_rng);
                next_gen.push(child.mutate(0.05, &mut repro_rng));
            } else {
                // 90% mutation only
                let parent_idx =
                    (repro_rng.next_f32() * survivors.len() as f32) as usize % survivors.len();
                next_gen.push(survivors[parent_idx].mutate(0.1, &mut repro_rng));
            }
        }

        population = next_gen;
    }

    let total_time = start_time.elapsed();
    let time_per_gen = total_time.as_secs_f64() * 1000.0 / n_generations as f64;

    // Find generation where we reached 90% of final fitness
    let final_mean = fitness_history.last().unwrap().1;
    let target_fitness = final_mean * 0.9;
    let gens_to_90 = fitness_history
        .iter()
        .position(|(_, mean, _, _)| *mean >= target_fitness)
        .unwrap_or(n_generations);

    let converged = final_mean > 0.1; // Arbitrary threshold

    EvolutionMetrics {
        config_name,
        config,
        generations_to_90_pct: gens_to_90,
        final_fitness_mean: fitness_history.last().unwrap().1,
        final_fitness_max: fitness_history.last().unwrap().2,
        final_fitness_std: fitness_history.last().unwrap().3,
        converged,
        time_per_generation_ms: time_per_gen,
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     EPIC 96.5: NN Architecture Scaling Experiment        ║");
    println!("║                                                           ║");
    println!("║  Question: Can larger NNs evolve AND run fast on WMMA?   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    const N_GENS: usize = 100;
    const POP_SIZE: usize = 1000;

    let configs: Vec<(&'static str, NNConfig)> = vec![
        ("A: 8-8-5 (baseline)", NNConfig::CONFIG_A),
        ("B: 16-16-5 (min WMMA)", NNConfig::CONFIG_B),
        ("C: 32-32-8 (WMMA sweet spot)", NNConfig::CONFIG_C),
        ("D: 32-64-8 (max capacity)", NNConfig::CONFIG_D),
        ("E: 64-64-8 (stretch goal)", NNConfig::CONFIG_E),
    ];

    let mut results = Vec::new();

    for (name, config) in configs {
        let metrics = run_evolution(config, name, N_GENS, POP_SIZE);
        results.push(metrics);
    }

    // Print summary table
    println!(
        "\n╔═══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              RESULTS SUMMARY                                      ║"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ Config           │ Params │ Gens→90% │ Final Fit │  Converged  │  ms/gen │ WMMA%  ║"
    );
    println!(
        "╠══════════════════╪════════╪══════════╪═══════════╪═════════════╪═════════╪════════╣"
    );

    for m in &results {
        println!(
            "║ {:<16} │ {:>6} │ {:>8} │ {:>9.4} │ {:>11} │ {:>7.2} │ {:>5.1}% ║",
            m.config_name,
            m.config.total_params(),
            m.generations_to_90_pct,
            m.final_fitness_mean,
            if m.converged { "✓ Yes" } else { "✗ No" },
            m.time_per_generation_ms,
            m.config.wmma_efficiency()
        );
    }

    println!(
        "╚══════════════════╧════════╧══════════╧═══════════╧═════════════╧═════════╧════════╝\n"
    );

    // Analysis
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                        ANALYSIS                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Evolution Performance:");
    let baseline = &results[0];
    for m in &results[1..] {
        let slowdown = m.time_per_generation_ms / baseline.time_per_generation_ms;
        let fitness_ratio = m.final_fitness_mean / baseline.final_fitness_mean;
        println!("  {} vs baseline:", m.config_name);
        println!("    Evolution speed: {:.2}× slower", slowdown);
        println!(
            "    Final fitness:   {:.2}× {}",
            fitness_ratio,
            if fitness_ratio > 1.0 {
                "better"
            } else {
                "worse"
            }
        );
        println!(
            "    Convergence:     {} (baseline: {})",
            if m.converged { "✓" } else { "✗" },
            if baseline.converged { "✓" } else { "✗" }
        );
    }

    println!("\nWMMA Viability:");
    for m in &results {
        let viable = m.converged && m.config.wmma_efficiency() > 60.0;
        if viable {
            println!(
                "  ✓ {}: VIABLE for WMMA (converged + {:.1}% efficient)",
                m.config_name,
                m.config.wmma_efficiency()
            );
        } else if !m.converged {
            println!("  ✗ {}: NOT viable (failed to converge)", m.config_name);
        } else {
            println!(
                "  ⚠ {}: Converged but low WMMA efficiency ({:.1}%)",
                m.config_name,
                m.config.wmma_efficiency()
            );
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      RECOMMENDATION                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Find best viable config
    let viable_configs: Vec<_> = results
        .iter()
        .filter(|m| m.converged && m.config.wmma_efficiency() > 60.0)
        .collect();

    if viable_configs.is_empty() {
        println!("❌ NO configurations are viable for WMMA.");
        println!("   Even small NNs failed to evolve or have terrible WMMA efficiency.");
        println!("   → Stick with EPIC 95.1 (FP32 GPU, 9.9× speedup)");
        println!("   → DO NOT pursue WMMA optimization.");
    } else {
        let best = viable_configs
            .iter()
            .max_by(|a, b| {
                let a_score = a.final_fitness_mean / a.time_per_generation_ms;
                let b_score = b.final_fitness_mean / b.time_per_generation_ms;
                a_score.partial_cmp(&b_score).unwrap()
            })
            .unwrap();

        println!("✅ RECOMMENDED: {}", best.config_name);
        println!("   Final fitness: {:.4}", best.final_fitness_mean);
        println!("   WMMA efficiency: {:.1}%", best.config.wmma_efficiency());
        println!(
            "   Evolution speed: {:.2}ms/gen",
            best.time_per_generation_ms
        );
        println!("\n   → Proceed to implement WMMA kernel for this configuration.");
        println!(
            "   → Expected speedup: {}× kernel (based on WMMA efficiency)",
            (best.config.wmma_efficiency() / 100.0 * 8.0) as usize
        );
    }

    println!("\n✓ EPIC 96.5 Experiment Complete");
}
