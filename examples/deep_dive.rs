//! Deep dive experiments: investigating the quantum advantage window
//!
//! Run with: cargo run --example deep_dive --release

use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::{ComparisonConfig, run_quantum_vs_classical};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              DEEP DIVE: QUANTUM ADVANTAGE                   ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut all_results = Vec::new();

    // =========================================================================
    // EXPERIMENT 1: Multiple seeds at the 1000-tick sweet spot
    // =========================================================================
    println!("{}", "=".repeat(70));
    println!("  EXPERIMENT 1: REPRODUCIBILITY TEST (1000 ticks, 10 different seeds)");
    println!("  Question: Is the quantum advantage at 1000 ticks consistent?");
    println!("{}\n", "=".repeat(70));

    let mut q_wins = 0;
    let mut c_wins = 0;
    let mut ties = 0;
    let mut total_q_advantage = 0.0f32;

    for seed in [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006] {
        let config = make_config_with_seed(18, 35, 1000, 4, seed);
        let result = run_quantum_vs_classical(&config);
        let s = &result.summary;

        let winner = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        match winner {
            "Q+" => q_wins += 1,
            "C+" => c_wins += 1,
            _ => ties += 1,
        }
        total_q_advantage += s.population_advantage;

        println!(
            "  Seed {}: Q={:.0} C={:.0} Δ={:+.1} {}",
            seed,
            s.quantum_mean_population,
            s.classical_mean_population,
            s.population_advantage,
            winner
        );

        all_results.push((format!("Seed_{}", seed), result));
    }

    println!(
        "\n  SUMMARY: Q wins={}, C wins={}, Ties={}",
        q_wins, c_wins, ties
    );
    println!("  Average Q advantage: {:+.1}", total_q_advantage / 10.0);

    // =========================================================================
    // EXPERIMENT 2: Time evolution - where does advantage emerge/decay?
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 2: TIME EVOLUTION (finding the advantage window)");
    println!("  Question: When does quantum advantage peak?");
    println!("{}\n", "=".repeat(70));

    let tick_series = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000];

    for ticks in tick_series {
        let config = make_config_with_seed(18, 35, ticks, 4, 42);
        let result = run_quantum_vs_classical(&config);
        let s = &result.summary;

        let winner = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        println!(
            "  {} ticks: Q={:.0} (gen={:.1}) C={:.0} (gen={:.1}) Δ={:+.1} {}",
            ticks,
            s.quantum_mean_population,
            s.quantum_mean_generation,
            s.classical_mean_population,
            s.classical_mean_generation,
            s.population_advantage,
            winner
        );

        all_results.push((format!("Ticks_{}", ticks), result));
    }

    // =========================================================================
    // EXPERIMENT 3: Goldilocks scarcity zone
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 3: GOLDILOCKS ZONE (survivable scarcity)");
    println!("  Question: Where is the sweet spot between easy and extinction?");
    println!("{}\n", "=".repeat(70));

    let scarcity_params = [
        (30, 45, "Easy"),
        (22, 38, "Medium"),
        (18, 32, "Moderate"),
        (15, 28, "Challenging"),
        (13, 24, "Hard"),
        (11, 20, "Very Hard"),
        (9, 16, "Extreme"),
    ];

    for (patches, value, label) in scarcity_params {
        let config = make_config_with_seed(patches, value, 1000, 4, 42);
        let result = run_quantum_vs_classical(&config);
        let s = &result.summary;

        let status = if s.quantum_extinction_rate > 0.5 && s.classical_extinction_rate > 0.5 {
            "EXTINCT"
        } else if s.quantum_extinction_rate > 0.5 || s.classical_extinction_rate > 0.5 {
            "PARTIAL"
        } else {
            "SURVIVE"
        };

        let winner = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        println!(
            "  {} ({},{}): Q={:.0} C={:.0} Δ={:+.1} {} [{}]",
            label,
            patches,
            value,
            s.quantum_mean_population,
            s.classical_mean_population,
            s.population_advantage,
            winner,
            status
        );

        all_results.push((format!("Scarcity_{}", label), result));
    }

    // =========================================================================
    // EXPERIMENT 4: Qubit scaling
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 4: QUBIT SCALING");
    println!("  Question: Does more qubits = more quantum advantage?");
    println!("{}\n", "=".repeat(70));

    for n_qubits in [3, 4, 5, 6] {
        let config = make_config_with_seed(18, 35, 1000, n_qubits, 42);
        let result = run_quantum_vs_classical(&config);
        let s = &result.summary;

        let winner = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        println!(
            "  {} qubits: Q={:.0} (qness={:.3}) C={:.0} Δ={:+.1} {}",
            n_qubits,
            s.quantum_mean_population,
            s.quantum_mean_qness,
            s.classical_mean_population,
            s.population_advantage,
            winner
        );

        all_results.push((format!("Qubits_{}", n_qubits), result));
    }

    // =========================================================================
    // EXPERIMENT 5: Circuit length impact
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 5: CIRCUIT COMPLEXITY");
    println!("  Question: Do longer circuits help quantum organisms?");
    println!("{}\n", "=".repeat(70));

    for (init_len, max_len) in [(4, 12), (8, 25), (12, 35), (16, 45)] {
        let mut config = make_config_with_seed(18, 35, 1000, 4, 42);
        config.forager_config.initial_circuit_length = init_len;
        config.forager_config.max_circuit_length = max_len;

        let result = run_quantum_vs_classical(&config);
        let s = &result.summary;

        let winner = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };

        println!(
            "  Circuit {}-{}: Q={:.0} (qness={:.3}) C={:.0} Δ={:+.1} {}",
            init_len,
            max_len,
            s.quantum_mean_population,
            s.quantum_mean_qness,
            s.classical_mean_population,
            s.population_advantage,
            winner
        );

        all_results.push((format!("Circuit_{}_{}", init_len, max_len), result));
    }

    // =========================================================================
    // Export and Summary
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  FINAL SUMMARY");
    println!("{}\n", "=".repeat(70));

    export_csv(&all_results, "deep_dive_results.csv");
    println!("Results exported to deep_dive_results.csv");

    // Count overall wins
    let mut total_q = 0;
    let mut total_c = 0;
    let mut total_tie = 0;
    for (_, result) in &all_results {
        let adv = result.summary.population_advantage;
        if adv > 10.0 {
            total_q += 1;
        } else if adv < -10.0 {
            total_c += 1;
        } else {
            total_tie += 1;
        }
    }

    println!("\n  OVERALL: {} experiments run", all_results.len());
    println!("  Quantum wins: {}", total_q);
    println!("  Classical wins: {}", total_c);
    println!("  Ties: {}", total_tie);
}

fn make_config_with_seed(
    heat_patches: usize,
    heat_value: u32,
    max_ticks: u64,
    n_qubits: u8,
    seed: u64,
) -> ComparisonConfig {
    ComparisonConfig {
        forager_config: QuantumForagerConfig {
            initial_population: 25,
            initial_energy: 100.0,
            n_qubits,
            initial_circuit_length: 8,
            max_circuit_length: 25,
            movement_cost: 0.5,
            metabolism_cost: 0.3,
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
        },
        max_ticks,
        resource_interval: 8,
        heat_patches,
        heat_value,
        log_interval: 50,
        replications: 5,
        seed,
    }
}

fn export_csv(
    results: &[(String, engine::quantum_vs_classical::ComparisonResult)],
    filename: &str,
) {
    let mut file = File::create(filename).expect("Failed to create CSV file");

    writeln!(
        file,
        "experiment,q_pop,q_ext,q_gen,q_qness,c_pop,c_ext,c_gen,advantage,winner"
    )
    .unwrap();

    for (name, result) in results {
        let s = &result.summary;
        let winner = if s.population_advantage > 10.0 {
            "Quantum"
        } else if s.population_advantage < -10.0 {
            "Classical"
        } else {
            "Tie"
        };
        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.2},{}",
            name,
            s.quantum_mean_population,
            s.quantum_extinction_rate,
            s.quantum_mean_generation,
            s.quantum_mean_qness,
            s.classical_mean_population,
            s.classical_extinction_rate,
            s.classical_mean_generation,
            s.population_advantage,
            winner
        )
        .unwrap();
    }
}
