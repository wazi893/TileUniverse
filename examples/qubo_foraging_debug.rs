//! Debug QUBO brain behavior under actual foraging conditions
//!
//! Run with: cargo run --release --example qubo_foraging_debug

use engine::brain::qubo_brain::{DecisionBudget, QuboBrain};
use engine::classical_brain::Move;
use engine::qec::noise::SimpleRng;
use engine::quantum_brain::SensorReadings;

fn main() {
    println!("=== QUBO Brain Foraging Debug ===\n");

    // Foraging config values
    let heat_value = 35u32; // From ThreeWayConfig::default()

    let mut rng = SimpleRng::new(42);
    let mut brain = QuboBrain::random(5, 5.0, &mut rng); // Same as mod_brain.rs
    let budget = DecisionBudget::default(); // sweeps: 200, beta: 8.0

    println!("Brain config:");
    println!("  n_vars: {}", brain.n_vars);
    println!("  one_hot_penalty: {}", brain.one_hot_penalty);
    println!("  sweeps: {}, beta: {}", budget.sweeps, budget.beta);
    println!();

    // Test scenarios that occur in foraging
    let scenarios = [
        (
            "Empty field (all zeros)",
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            },
        ),
        (
            "On heat source",
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: heat_value,
                charge_local: 0,
            },
        ),
        (
            "Heat to north",
            SensorReadings {
                heat_n: heat_value,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            },
        ),
        (
            "Heat to south",
            SensorReadings {
                heat_n: 0,
                heat_s: heat_value,
                heat_e: 0,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            },
        ),
        (
            "Heat to east",
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: heat_value,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            },
        ),
        (
            "Heat to west",
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: 0,
                heat_w: heat_value,
                heat_local: 0,
                charge_local: 0,
            },
        ),
        (
            "Weak gradient (5 vs 0)",
            SensorReadings {
                heat_n: 5,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            },
        ),
    ];

    let action_names = ["Up", "Down", "Left", "Right", "Stay"];

    for (name, sensors) in &scenarios {
        println!("=== Scenario: {} ===", name);
        println!(
            "Sensors: n={}, s={}, e={}, w={}, local={}",
            sensors.heat_n, sensors.heat_s, sensors.heat_e, sensors.heat_w, sensors.heat_local
        );

        // Compute biases and show them
        brain.compute_biases(sensors);
        let biases = brain.get_biases();
        println!("Biases:");
        for i in 0..5 {
            println!("  {}: {:+.4}", action_names[i], biases[i]);
        }

        // Build problem and show h values (after one-hot penalty)
        let problem = brain.build_problem();
        println!("h[i] (after one-hot penalty):");
        for i in 0..5 {
            println!("  {}: {:+.4}", action_names[i], problem.h[i]);
        }

        // Run 20 decisions and count actions
        let mut counts = [0u32; 5];
        for trial in 0..20 {
            let mut test_rng = SimpleRng::new(trial as u64 * 999983);
            let action = brain.decide(sensors, &mut test_rng, &budget);
            let idx = match action {
                Move::Up => 0,
                Move::Down => 1,
                Move::Left => 2,
                Move::Right => 3,
                Move::Stay => 4,
            };
            counts[idx] += 1;
        }

        println!("Decisions (20 trials):");
        for i in 0..5 {
            let pct = counts[i] as f32 / 20.0 * 100.0;
            let bar = "#".repeat((pct / 5.0) as usize);
            println!(
                "  {:5}: {:2} ({:5.1}%) {}",
                action_names[i], counts[i], pct, bar
            );
        }
        println!();
    }

    // Simulate a mini foraging run
    println!("=== Mini Foraging Simulation ===");
    println!("Simulating 100 steps of random-then-directed movement\n");

    let mut empty_count = 0;
    let mut directed_correct = 0;
    let mut directed_total = 0;
    let mut stay_correct = 0;
    let mut stay_total = 0;

    for step in 0..100 {
        let rng_seed = step as u64 * 12345;

        // 70% of time: empty field
        // 20% of time: heat nearby
        // 10% of time: on heat
        let scenario_roll = step % 10;

        let sensors = if scenario_roll < 7 {
            empty_count += 1;
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: 0,
                charge_local: 0,
            }
        } else if scenario_roll < 9 {
            directed_total += 1;
            let dir = step % 4;
            match dir {
                0 => SensorReadings {
                    heat_n: heat_value,
                    heat_s: 0,
                    heat_e: 0,
                    heat_w: 0,
                    heat_local: 0,
                    charge_local: 0,
                },
                1 => SensorReadings {
                    heat_n: 0,
                    heat_s: heat_value,
                    heat_e: 0,
                    heat_w: 0,
                    heat_local: 0,
                    charge_local: 0,
                },
                2 => SensorReadings {
                    heat_n: 0,
                    heat_s: 0,
                    heat_e: heat_value,
                    heat_w: 0,
                    heat_local: 0,
                    charge_local: 0,
                },
                _ => SensorReadings {
                    heat_n: 0,
                    heat_s: 0,
                    heat_e: 0,
                    heat_w: heat_value,
                    heat_local: 0,
                    charge_local: 0,
                },
            }
        } else {
            stay_total += 1;
            SensorReadings {
                heat_n: 0,
                heat_s: 0,
                heat_e: 0,
                heat_w: 0,
                heat_local: heat_value,
                charge_local: 0,
            }
        };

        let mut test_rng = SimpleRng::new(rng_seed);
        let action = brain.decide(&sensors, &mut test_rng, &budget);

        // Check correctness
        if sensors.heat_n > 0 && action == Move::Up {
            directed_correct += 1;
        }
        if sensors.heat_s > 0 && action == Move::Down {
            directed_correct += 1;
        }
        if sensors.heat_e > 0 && action == Move::Right {
            directed_correct += 1;
        }
        if sensors.heat_w > 0 && action == Move::Left {
            directed_correct += 1;
        }
        if sensors.heat_local > 0 && action == Move::Stay {
            stay_correct += 1;
        }
    }

    println!("Results:");
    println!("  Empty field steps: {}", empty_count);
    println!(
        "  Directed movement: {}/{} correct ({:.1}%)",
        directed_correct,
        directed_total,
        if directed_total > 0 {
            directed_correct as f32 / directed_total as f32 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  Stay on heat: {}/{} correct ({:.1}%)",
        stay_correct,
        stay_total,
        if stay_total > 0 {
            stay_correct as f32 / stay_total as f32 * 100.0
        } else {
            0.0
        }
    );

    // What does a random brain look like?
    println!("\n=== Comparison: Fresh Random Brains ===");
    for brain_id in 0..5 {
        let mut rng = SimpleRng::new(brain_id as u64 * 777);
        let mut brain = QuboBrain::random(5, 1.0, &mut rng);

        // Test on "heat to north" scenario
        let sensors = SensorReadings {
            heat_n: heat_value,
            heat_s: 0,
            heat_e: 0,
            heat_w: 0,
            heat_local: 0,
            charge_local: 0,
        };

        let mut counts = [0u32; 5];
        for trial in 0..10 {
            let mut test_rng = SimpleRng::new(trial as u64 + brain_id as u64 * 1000);
            let action = brain.decide(&sensors, &mut test_rng, &budget);
            let idx = match action {
                Move::Up => 0,
                Move::Down => 1,
                Move::Left => 2,
                Move::Right => 3,
                Move::Stay => 4,
            };
            counts[idx] += 1;
        }

        println!(
            "  Brain {}: Up={}, Down={}, Left={}, Right={}, Stay={}",
            brain_id, counts[0], counts[1], counts[2], counts[3], counts[4]
        );
    }
}
