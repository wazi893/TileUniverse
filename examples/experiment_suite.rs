//! Comprehensive experiment suite: resource scarcity & mutation rates
//!
//! Run with: cargo run --example experiment_suite --release

use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::{ComparisonConfig, run_quantum_vs_classical};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         COMPREHENSIVE EXPERIMENT SUITE                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut all_results = Vec::new();

    // =========================================================================
    // EXPERIMENT 1: Resource Scarcity Sweep
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("  EXPERIMENT 1: RESOURCE SCARCITY");
    println!("  Hypothesis: Quantum advantage emerges under resource pressure");
    println!("{}\n", "=".repeat(70));

    let scarcity_levels = [
        ("Abundant", 50, 60), // Easy mode
        ("Normal", 25, 40),   // Default
        ("Scarce", 12, 25),   // Challenging
        ("Harsh", 6, 15),     // Very difficult
    ];

    for (name, patches, value) in scarcity_levels.iter() {
        println!(
            "\n--- {} Resources (patches={}, value={}) ---",
            name, patches, value
        );

        let config = make_config(*patches, *value, 0.85, 500);
        let result = run_quantum_vs_classical(&config);

        let s = &result.summary;
        println!(
            "  Quantum:   pop={:.1}, ext={:.0}%, gen={:.1}",
            s.quantum_mean_population,
            s.quantum_extinction_rate * 100.0,
            s.quantum_mean_generation
        );
        println!(
            "  Classical: pop={:.1}, ext={:.0}%, gen={:.1}",
            s.classical_mean_population,
            s.classical_extinction_rate * 100.0,
            s.classical_mean_generation
        );
        println!("  Δ Population: {:+.1}", s.population_advantage);
        println!("  Winner: {}", s.winner);

        all_results.push((format!("Scarcity_{}", name), result));
    }

    // =========================================================================
    // EXPERIMENT 2: Mutation Rate Sweep
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 2: MUTATION RATE");
    println!("  Hypothesis: Optimal mutation rate differs for quantum vs classical");
    println!("{}\n", "=".repeat(70));

    let mutation_rates = [0.5, 0.7, 0.85, 0.95];

    for rate in mutation_rates.iter() {
        println!("\n--- Mutation Rate: {:.0}% ---", rate * 100.0);

        let config = make_config(20, 35, *rate, 500);
        let result = run_quantum_vs_classical(&config);

        let s = &result.summary;
        println!(
            "  Quantum:   pop={:.1}, ext={:.0}%, gen={:.1}, qness={:.3}",
            s.quantum_mean_population,
            s.quantum_extinction_rate * 100.0,
            s.quantum_mean_generation,
            s.quantum_mean_qness
        );
        println!(
            "  Classical: pop={:.1}, ext={:.0}%, gen={:.1}",
            s.classical_mean_population,
            s.classical_extinction_rate * 100.0,
            s.classical_mean_generation
        );
        println!("  Δ Population: {:+.1}", s.population_advantage);
        println!("  Winner: {}", s.winner);

        all_results.push((format!("Mutation_{:.0}pct", rate * 100.0), result));
    }

    // =========================================================================
    // EXPERIMENT 3: Longer Evolution (more generations)
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPERIMENT 3: LONGER EVOLUTION");
    println!("  Hypothesis: Quantum advantage may take time to emerge");
    println!("{}\n", "=".repeat(70));

    let tick_counts = [500, 1000, 2000];

    for ticks in tick_counts.iter() {
        println!("\n--- {} Ticks ---", ticks);

        let config = make_config(18, 35, 0.85, *ticks);
        let result = run_quantum_vs_classical(&config);

        let s = &result.summary;
        println!(
            "  Quantum:   pop={:.1}, ext={:.0}%, gen={:.1}, qness={:.3}",
            s.quantum_mean_population,
            s.quantum_extinction_rate * 100.0,
            s.quantum_mean_generation,
            s.quantum_mean_qness
        );
        println!(
            "  Classical: pop={:.1}, ext={:.0}%, gen={:.1}",
            s.classical_mean_population,
            s.classical_extinction_rate * 100.0,
            s.classical_mean_generation
        );
        println!("  Δ Population: {:+.1}", s.population_advantage);
        println!("  Winner: {}", s.winner);

        all_results.push((format!("Ticks_{}", ticks), result));
    }

    // =========================================================================
    // Export CSV
    // =========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("  EXPORTING RESULTS");
    println!("{}\n", "=".repeat(70));

    export_csv(&all_results, "experiment_results.csv");
    println!("Results exported to experiment_results.csv");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    SUMMARY                                   ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Experiment", "Q Pop", "C Pop", "Advantage"
    );
    println!("{:-<65}", "");

    for (name, result) in &all_results {
        let s = &result.summary;
        let winner_mark = if s.population_advantage > 10.0 {
            "Q+"
        } else if s.population_advantage < -10.0 {
            "C+"
        } else {
            "="
        };
        println!(
            "{:<25} {:>12.1} {:>12.1} {:>+10.1} {}",
            name,
            s.quantum_mean_population,
            s.classical_mean_population,
            s.population_advantage,
            winner_mark
        );
    }
}

fn make_config(
    heat_patches: usize,
    heat_value: u32,
    mutation_rate: f32,
    max_ticks: u64,
) -> ComparisonConfig {
    ComparisonConfig {
        forager_config: QuantumForagerConfig {
            initial_population: 25,
            initial_energy: 100.0,
            n_qubits: 4,
            initial_circuit_length: 8,
            max_circuit_length: 25,
            movement_cost: 0.5,
            metabolism_cost: 0.3,
            reproduction_threshold: 140.0,
            reproduction_cost: 55.0,
            mutation_rate,
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
        seed: 42,
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
        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.2},\"{}\"",
            name,
            s.quantum_mean_population,
            s.quantum_extinction_rate,
            s.quantum_mean_generation,
            s.quantum_mean_qness,
            s.classical_mean_population,
            s.classical_extinction_rate,
            s.classical_mean_generation,
            s.population_advantage,
            s.winner
        )
        .unwrap();
    }
}
