//! THE EXPERIMENT: Does evolution discover quantum advantage?
//!
//! Run with: cargo run --example quantum_vs_classical_experiment --release

use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::{ComparisonConfig, print_comparison, run_quantum_vs_classical};

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   THE EXPERIMENT: Does Evolution Discover Quantum Advantage?║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Configure the experiment
    let config = ComparisonConfig {
        forager_config: QuantumForagerConfig {
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
        },
        max_ticks: 500,
        resource_interval: 8,
        heat_patches: 30,
        heat_value: 45,
        log_interval: 50,
        replications: 5, // Run 5 times per group for statistical significance
        seed: 42,
    };

    println!("Configuration:");
    println!(
        "  - Initial population: {}",
        config.forager_config.initial_population
    );
    println!("  - Max ticks: {}", config.max_ticks);
    println!("  - Replications per group: {}", config.replications);
    println!(
        "  - Qubits per organism: {}",
        config.forager_config.n_qubits
    );
    println!();
    println!("Running experiment...\n");

    let result = run_quantum_vs_classical(&config);

    print_comparison(&result);

    // Additional details
    println!("\n--- Individual Run Details ---\n");

    println!("QUANTUM GROUP:");
    for (i, run) in result.quantum_runs.iter().enumerate() {
        println!(
            "  Run {}: pop={}, gen={}, extinct={}, qness={:.3}",
            i + 1,
            run.final_population,
            run.max_generation,
            run.extinct,
            run.final_quantum_ness
        );
    }

    println!("\nCLASSICAL GROUP:");
    for (i, run) in result.classical_runs.iter().enumerate() {
        println!(
            "  Run {}: pop={}, gen={}, extinct={}",
            i + 1,
            run.final_population,
            run.max_generation,
            run.extinct
        );
    }
}
