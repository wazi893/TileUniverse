//! EPIC 93.2: Full Simulation Loop Profiling
//!
//! Measures where time is actually spent in neural network organism simulation:
//! - NN forward pass (brain decision making)
//! - Movement execution
//! - Energy/resource management
//!
//! Decision Criteria:
//! - If NN forward >50% of tick time → GPU optimization is worth it (EPIC 94)
//! - If NN forward <10% of tick time → Optimize other parts instead
//! - If 10-50% → Consider both paths
//!
//! Run with: cargo run --example epic93_2_simulation_profile --release

use engine::batched_brain::{BrainWeights, forward_batch_shared_weights};
use engine::classical_brain::{ClassicalBrain, Move};
use engine::quantum::QRng;
use std::time::Instant;

/// Simple forager for profiling
struct Forager {
    x: u32,
    y: u32,
    energy: u16,
}

/// Simple resource patch
struct Patch {
    x: u32,
    y: u32,
    value: u16,
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   EPIC 93.2: Full Simulation Loop Profiling              ║");
    println!("║   Finding the Real Bottleneck                             ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Hypothesis:");
    println!("  We need to know WHERE time is spent in a simulation tick");
    println!("  before deciding whether to optimize NN forward pass.\n");

    println!("Decision Criteria:");
    println!("  • NN forward >50% of tick → Proceed to GPU (EPIC 94)");
    println!("  • NN forward <10% of tick → Optimize grid/physics instead");
    println!("  • NN forward 10-50%       → Consider both optimization paths\n");

    // Test configurations
    let population_sizes = [100, 1_000, 10_000];
    let ticks = 100;

    println!("Configuration:");
    println!("  Ticks: {}", ticks);
    println!("  Population sizes: {:?}\n", population_sizes);

    println!("═══════════════════════════════════════════════════════════");
    println!("TEST 1: Per-Organism NN Forward Pass (Current Approach)");
    println!("═══════════════════════════════════════════════════════════\n");

    for &n_organisms in &population_sizes {
        println!("Population: {} organisms", n_organisms);

        // Setup
        let mut rng = QRng::new(12345);
        let mut foragers: Vec<Forager> = (0..n_organisms)
            .map(|_| Forager {
                x: (rng.next_f32() * 256.0) as u32,
                y: (rng.next_f32() * 256.0) as u32,
                energy: 100,
            })
            .collect();

        let brains: Vec<ClassicalBrain> = (0..n_organisms)
            .map(|_| ClassicalBrain::random(&mut rng))
            .collect();

        let mut patches: Vec<Patch> = (0..20)
            .map(|_| Patch {
                x: (rng.next_f32() * 256.0) as u32,
                y: (rng.next_f32() * 256.0) as u32,
                value: 50,
            })
            .collect();

        // Warmup
        for _ in 0..10 {
            for (_forager, brain) in foragers.iter_mut().zip(brains.iter()) {
                let sensors = [0.5; 8]; // Dummy sensors
                let _ = brain.forward(&sensors, &mut rng);
            }
        }

        // Profile
        let mut time_nn_forward = 0.0;
        let mut time_movement = 0.0;
        let mut time_energy = 0.0;
        let mut time_total = 0.0;

        for _ in 0..ticks {
            let tick_start = Instant::now();

            // 1. NN Forward Pass
            let forward_start = Instant::now();
            let mut decisions = Vec::with_capacity(n_organisms);
            for (forager, brain) in foragers.iter().zip(brains.iter()) {
                // Generate sensors based on position
                let sensors = [
                    (forager.x as f32 / 256.0) * 2.0 - 1.0,
                    (forager.y as f32 / 256.0) * 2.0 - 1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ];
                let mv = brain.forward(&sensors, &mut rng);
                decisions.push(mv);
            }
            time_nn_forward += forward_start.elapsed().as_secs_f64();

            // 2. Movement
            let movement_start = Instant::now();
            for (forager, mv) in foragers.iter_mut().zip(decisions.iter()) {
                let (dx, dy) = match mv {
                    Move::Up => (0, 1),
                    Move::Down => (0, -1),
                    Move::Left => (-1, 0),
                    Move::Right => (1, 0),
                    Move::Stay => (0, 0),
                };

                forager.x = ((forager.x as i32 + dx).max(0).min(255)) as u32;
                forager.y = ((forager.y as i32 + dy).max(0).min(255)) as u32;
            }
            time_movement += movement_start.elapsed().as_secs_f64();

            // 3. Energy & Resources
            let energy_start = Instant::now();
            for forager in foragers.iter_mut() {
                // Movement cost
                forager.energy = forager.energy.saturating_sub(1);

                // Check resources
                for patch in patches.iter_mut() {
                    if forager.x == patch.x && forager.y == patch.y && patch.value > 0 {
                        forager.energy += patch.value;
                        patch.value = 0;
                    }
                }
            }
            time_energy += energy_start.elapsed().as_secs_f64();

            time_total += tick_start.elapsed().as_secs_f64();
        }

        // Calculate percentages
        let pct_nn = (time_nn_forward / time_total) * 100.0;
        let pct_movement = (time_movement / time_total) * 100.0;
        let pct_energy = (time_energy / time_total) * 100.0;
        let pct_other = 100.0 - pct_nn - pct_movement - pct_energy;

        println!(
            "  Total time: {:.2} ms ({:.3} ms/tick)",
            time_total * 1000.0,
            time_total * 1000.0 / ticks as f64
        );
        println!();
        println!("  Time breakdown:");
        println!(
            "    NN forward:     {:.2} ms ({:5.1}%)",
            time_nn_forward * 1000.0,
            pct_nn
        );
        println!(
            "    Movement:       {:.2} ms ({:5.1}%)",
            time_movement * 1000.0,
            pct_movement
        );
        println!(
            "    Energy/res:     {:.2} ms ({:5.1}%)",
            time_energy * 1000.0,
            pct_energy
        );
        println!(
            "    Other/overhead: {:.2} ms ({:5.1}%)",
            (time_total - time_nn_forward - time_movement - time_energy) * 1000.0,
            pct_other
        );
        println!();

        // Decision
        if pct_nn > 50.0 {
            println!("  Decision: ✓ NN is DOMINANT bottleneck ({:.0}%)", pct_nn);
            println!("           → GPU optimization (EPIC 94) is HIGH PRIORITY");
        } else if pct_nn > 25.0 {
            println!("  Decision: ⚠ NN is SIGNIFICANT component ({:.0}%)", pct_nn);
            println!("           → GPU optimization would help significantly");
        } else if pct_nn > 10.0 {
            println!("  Decision: ⚠ NN is MODERATE component ({:.0}%)", pct_nn);
            println!("           → Consider optimizing other parts first");
        } else {
            println!("  Decision: ✗ NN is MINOR component ({:.0}%)", pct_nn);
            println!("           → Optimize grid/physics instead of NN");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("TEST 2: Batched NN Forward (Shared Weights Optimization)");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Simulating early generation where all organisms have same brain\n");

    for &n_organisms in &population_sizes {
        println!("Population: {} organisms (shared weights)", n_organisms);

        // Setup
        let mut rng = QRng::new(54321);
        let foragers: Vec<Forager> = (0..n_organisms)
            .map(|_| Forager {
                x: (rng.next_f32() * 256.0) as u32,
                y: (rng.next_f32() * 256.0) as u32,
                energy: 100,
            })
            .collect();

        let shared_brain = ClassicalBrain::random(&mut rng);
        let shared_weights = BrainWeights::from(&shared_brain);

        // Warmup
        for _ in 0..10 {
            let sensors: Vec<[f32; 8]> = foragers
                .iter()
                .map(|f| {
                    [
                        (f.x as f32 / 256.0) * 2.0 - 1.0,
                        (f.y as f32 / 256.0) * 2.0 - 1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                })
                .collect();
            let _ = forward_batch_shared_weights(&sensors, &shared_weights);
        }

        // Profile
        let mut time_batched_nn = 0.0;
        let mut time_rest = 0.0;

        for _ in 0..ticks {
            // Batched NN forward
            let forward_start = Instant::now();
            let sensors: Vec<[f32; 8]> = foragers
                .iter()
                .map(|f| {
                    [
                        (f.x as f32 / 256.0) * 2.0 - 1.0,
                        (f.y as f32 / 256.0) * 2.0 - 1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                })
                .collect();
            let moves = forward_batch_shared_weights(&sensors, &shared_weights);
            time_batched_nn += forward_start.elapsed().as_secs_f64();

            // Rest of simulation
            let rest_start = Instant::now();
            // Movement and energy (simplified)
            for _ in 0..n_organisms {
                let _ = moves[0]; // Just access to prevent optimization
            }
            time_rest += rest_start.elapsed().as_secs_f64();
        }

        let time_total = time_batched_nn + time_rest;
        let pct_batched_nn = (time_batched_nn / time_total) * 100.0;

        println!(
            "  Batched NN forward: {:.2} ms ({:.1}%)",
            time_batched_nn * 1000.0,
            pct_batched_nn
        );
        println!(
            "  Rest of simulation: {:.2} ms ({:.1}%)",
            time_rest * 1000.0,
            100.0 - pct_batched_nn
        );
        println!(
            "  Total: {:.3} ms/tick\n",
            time_total * 1000.0 / ticks as f64
        );
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("EPIC 93.2 VERDICT");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Key Questions Answered:");
    println!("  1. What % of simulation time is NN forward pass?");
    println!("  2. Is GPU optimization worth pursuing?");
    println!("  3. Does population size affect the bottleneck?\n");

    println!("Next Steps Based on Results:");
    println!("  • If NN >50%  → EPIC 94 (GPU) is HIGH PRIORITY");
    println!("  • If NN 25-50% → EPIC 94 (GPU) would help significantly");
    println!("  • If NN 10-25% → Consider other optimizations first");
    println!("  • If NN <10%  → Optimize grid/physics, not NN\n");

    println!("Strategic Recommendations:");
    println!("  • Use shared weights for early evolution generations");
    println!("  • Profile scales with population size");
    println!("  • GPU benefits increase with larger populations\n");
}
