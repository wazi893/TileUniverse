//! Quantum-ness Evolution Experiment
//!
//! Answers: "Does evolution discover entanglement, or just preserve it?"
//!
//! Tracks quantum-ness every 100 ticks to see the trajectory:
//! - If it increases: Evolution discovered entanglement is useful
//! - If it stays constant: Initial random circuits had entanglement, preserved
//! - If it decreases: Evolution pruned useless entanglement
//!
//! Run with: cargo run --example quantumness_evolution --release

use engine::constrained_circuits::{CircuitConstraints, entanglement_fraction};
use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::ConstrainedPopulation;
use engine::simulation::Simulation;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   QUANTUM-NESS EVOLUTION: Discovery or Preservation?       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

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

    let max_ticks = 2000;
    let sample_interval = 100;
    let seed = 42u64;

    println!("Configuration:");
    println!("  - Max ticks: {}", max_ticks);
    println!("  - Sample every: {} ticks", sample_interval);
    println!("  - Qubits: {}", config.n_qubits);
    println!();

    // Run quantum population with tracking
    println!("Running QUANTUM population with quantum-ness tracking...\n");
    let quantum_trajectory = run_with_tracking(
        &config,
        CircuitConstraints::full_quantum(),
        max_ticks,
        sample_interval,
        seed,
    );

    // Run classical population for comparison
    println!("Running CLASSICAL population for comparison...\n");
    let classical_trajectory = run_with_tracking(
        &config,
        CircuitConstraints::classical_only(),
        max_ticks,
        sample_interval,
        seed,
    );

    // Print trajectory table
    println!("======================================================================");
    println!("  QUANTUM-NESS TRAJECTORY");
    println!("======================================================================\n");

    println!("Tick    Q-Pop   Q-Gen   Q-ness   C-Pop   C-Gen   C-ness   ΔQ-ness");
    println!("--------------------------------------------------------------------");

    for (q, c) in quantum_trajectory.iter().zip(classical_trajectory.iter()) {
        let delta = q.avg_quantum_ness - c.avg_quantum_ness;
        println!(
            "{:>4}    {:>5}   {:>5}   {:>6.3}   {:>5}   {:>5}   {:>6.3}   {:>+7.3}",
            q.tick,
            q.population,
            q.max_generation,
            q.avg_quantum_ness,
            c.population,
            c.max_generation,
            c.avg_quantum_ness,
            delta,
        );
    }

    // Analyze trajectory
    println!("\n======================================================================");
    println!("  TRAJECTORY ANALYSIS");
    println!("======================================================================\n");

    // Calculate trend
    let q_start = quantum_trajectory
        .first()
        .map(|s| s.avg_quantum_ness)
        .unwrap_or(0.0);
    let q_mid = quantum_trajectory
        .get(quantum_trajectory.len() / 2)
        .map(|s| s.avg_quantum_ness)
        .unwrap_or(0.0);
    let q_end = quantum_trajectory
        .last()
        .map(|s| s.avg_quantum_ness)
        .unwrap_or(0.0);

    println!("Quantum-ness trajectory:");
    println!("  Start (tick 0):    {:.3}", q_start);
    println!("  Middle (tick {}): {:.3}", max_ticks / 2, q_mid);
    println!("  End (tick {}):   {:.3}", max_ticks, q_end);
    println!();

    let early_delta = q_mid - q_start;
    let late_delta = q_end - q_mid;
    let total_delta = q_end - q_start;

    println!("Changes:");
    println!(
        "  Early phase (0 → {}):    {:+.3}",
        max_ticks / 2,
        early_delta
    );
    println!(
        "  Late phase ({} → {}): {:+.3}",
        max_ticks / 2,
        max_ticks,
        late_delta
    );
    println!("  Total change:             {:+.3}", total_delta);
    println!();

    // Interpret
    println!("======================================================================");
    println!("  INTERPRETATION");
    println!("======================================================================\n");

    if total_delta > 0.05 {
        println!(
            "[+] DISCOVERY: Quantum-ness INCREASED over evolution (+{:.1}%)",
            total_delta * 100.0
        );
        println!("    Evolution actively discovered that entanglement helps survival!");
    } else if total_delta < -0.05 {
        println!(
            "[-] PRUNING: Quantum-ness DECREASED over evolution ({:.1}%)",
            total_delta * 100.0
        );
        println!("    Evolution removed some entanglement - perhaps too much is harmful?");
    } else {
        println!(
            "[=] PRESERVATION: Quantum-ness stayed relatively STABLE ({:+.1}%)",
            total_delta * 100.0
        );
        println!("    Initial random circuits had useful entanglement, evolution preserved it.");
    }

    println!();

    // Additional insights
    if early_delta > 0.05 && late_delta < -0.05 {
        println!("[!] INTERESTING: Early increase, then late decrease!");
        println!("    Entanglement helps early exploration but may hinder late exploitation.");
    }

    if early_delta < -0.05 && late_delta > 0.05 {
        println!("[!] INTERESTING: Early decrease, then late increase!");
        println!("    Evolution first pruned, then rediscovered entanglement.");
    }

    // Peak analysis
    let peak = quantum_trajectory
        .iter()
        .max_by(|a, b| a.avg_quantum_ness.partial_cmp(&b.avg_quantum_ness).unwrap())
        .unwrap();
    let trough = quantum_trajectory
        .iter()
        .min_by(|a, b| a.avg_quantum_ness.partial_cmp(&b.avg_quantum_ness).unwrap())
        .unwrap();

    println!();
    println!(
        "Peak quantum-ness:   {:.3} at tick {}",
        peak.avg_quantum_ness, peak.tick
    );
    println!(
        "Lowest quantum-ness: {:.3} at tick {}",
        trough.avg_quantum_ness, trough.tick
    );
    println!(
        "Range: {:.3}",
        peak.avg_quantum_ness - trough.avg_quantum_ness
    );

    // ASCII chart
    println!("\n======================================================================");
    println!("  QUANTUM-NESS OVER TIME (ASCII Chart)");
    println!("======================================================================\n");

    print_ascii_chart(&quantum_trajectory, &classical_trajectory);
}

#[derive(Clone, Debug)]
struct Snapshot {
    tick: u32,
    population: usize,
    max_generation: u32,
    avg_quantum_ness: f32,
    avg_circuit_length: f32,
}

fn run_with_tracking(
    config: &QuantumForagerConfig,
    constraints: CircuitConstraints,
    max_ticks: u32,
    sample_interval: u32,
    seed: u64,
) -> Vec<Snapshot> {
    let mut sim = Simulation::new();
    let mut pop = ConstrainedPopulation::new(config.clone(), constraints, seed);
    let mut snapshots = Vec::new();

    // Seed resources
    let region = &config.region;
    for i in 0..30u64 {
        let rx = region.x0
            + ((seed.wrapping_mul(0x123456789).wrapping_add(i * 17)) as u32 % region.width);
        let ry = region.y0
            + ((seed.wrapping_mul(0x987654321).wrapping_add(i * 31)) as u32 % region.height);
        sim.set_heat_field_for_test(rx as usize, ry as usize, 45);
    }

    // Take initial snapshot
    snapshots.push(take_snapshot(&pop, 0));

    // Run simulation with periodic snapshots
    for tick in 1..=max_ticks {
        // Respawn resources periodically
        if tick % 8 == 0 {
            for i in 0..30u64 {
                let rseed = seed.wrapping_add(tick as u64);
                let rx = region.x0
                    + ((rseed
                        .wrapping_mul(0x123456789)
                        .wrapping_add(tick as u64 * 7 + i * 17)) as u32
                        % region.width);
                let ry = region.y0
                    + ((rseed
                        .wrapping_mul(0x987654321)
                        .wrapping_add(tick as u64 * 13 + i * 31)) as u32
                        % region.height);
                sim.set_heat_field_for_test(rx as usize, ry as usize, 45);
            }
        }

        pop.step(&mut sim);

        // Take snapshot at intervals
        if tick % sample_interval == 0 {
            snapshots.push(take_snapshot(&pop, tick));
        }
    }

    snapshots
}

fn take_snapshot(pop: &ConstrainedPopulation, tick: u32) -> Snapshot {
    let stats = pop.stats();
    let circuits = pop.extract_all_circuits();

    // Calculate average quantum-ness
    let avg_quantum_ness = if circuits.is_empty() {
        0.0
    } else {
        let total: f32 = circuits.iter().map(|c| entanglement_fraction(c)).sum();
        total / circuits.len() as f32
    };

    let avg_circuit_length = if circuits.is_empty() {
        0.0
    } else {
        circuits.iter().map(|c| c.len()).sum::<usize>() as f32 / circuits.len() as f32
    };

    Snapshot {
        tick,
        population: stats.size,
        max_generation: stats.max_generation,
        avg_quantum_ness,
        avg_circuit_length,
    }
}

fn print_ascii_chart(quantum: &[Snapshot], classical: &[Snapshot]) {
    let height = 15;
    let width = quantum.len().min(60);

    // Find range
    let _max_qness = quantum
        .iter()
        .map(|s| s.avg_quantum_ness)
        .fold(0.0f32, |a, b| a.max(b))
        .max(0.01);

    println!("1.0 |");

    for row in 0..height {
        let threshold = 1.0 - (row as f32 / height as f32);
        print!("{:.1} |", threshold);

        for (_i, (q, c)) in quantum.iter().zip(classical.iter()).enumerate().take(width) {
            let q_norm = q.avg_quantum_ness;
            let c_norm = c.avg_quantum_ness;

            let q_here = q_norm >= threshold && q_norm < threshold + (1.0 / height as f32);
            let c_here = c_norm >= threshold && c_norm < threshold + (1.0 / height as f32);

            if q_here && c_here {
                print!("*"); // Both
            } else if q_here {
                print!("Q"); // Quantum
            } else if c_here {
                print!("C"); // Classical
            } else if q_norm > threshold {
                print!("|"); // Quantum above
            } else {
                print!(" ");
            }
        }
        println!();
    }

    print!("0.0 +");
    for _ in 0..width {
        print!("-");
    }
    println!();
    println!(
        "     0{:>width$}",
        quantum.last().map(|s| s.tick).unwrap_or(0),
        width = width - 1
    );
    println!();
    println!("Legend: Q = Quantum, C = Classical, * = Both, | = Quantum above threshold");
}
