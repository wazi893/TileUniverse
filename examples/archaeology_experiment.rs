//! Circuit Archaeology Experiment
//!
//! Answers: "What circuits do quantum winners actually evolve?"
//!
//! Run with: cargo run --example archaeology_experiment --release

use engine::circuit_archaeology::{
    QuantumPattern, analyze_population_patterns, circuit_diagram, detect_patterns,
};
use engine::constrained_circuits::CircuitConstraints;
use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::ConstrainedPopulation;
use engine::simulation::Simulation;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   CIRCUIT ARCHAEOLOGY: What Did Evolution Discover?        ║");
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

    let ticks = 1000;
    let seed = 42u64;

    println!("Configuration:");
    println!("  - Ticks: {}", ticks);
    println!("  - Qubits: {}", config.n_qubits);
    println!(
        "  - Initial circuit length: {}",
        config.initial_circuit_length
    );
    println!("  - Max circuit length: {}", config.max_circuit_length);
    println!();

    // Run quantum population
    println!("Running QUANTUM population for {} ticks...", ticks);
    let (quantum_circuits, quantum_stats) =
        run_and_extract(&config, CircuitConstraints::full_quantum(), ticks, seed);
    println!("  Final population: {}", quantum_stats.0);
    println!("  Max generation: {}", quantum_stats.1);
    println!();

    // Run classical population
    println!("Running CLASSICAL population for {} ticks...", ticks);
    let (classical_circuits, classical_stats) =
        run_and_extract(&config, CircuitConstraints::classical_only(), ticks, seed);
    println!("  Final population: {}", classical_stats.0);
    println!("  Max generation: {}", classical_stats.1);
    println!();

    // Analyze quantum winners
    println!("======================================================================");
    println!(
        "  QUANTUM WINNERS - Top {} Circuits",
        quantum_circuits.len().min(10)
    );
    println!("======================================================================\n");

    for (i, circuit) in quantum_circuits.iter().take(10).enumerate() {
        println!(
            "--- Quantum Winner #{} ({} gates) ---",
            i + 1,
            circuit.len()
        );

        let analysis = detect_patterns(circuit, config.n_qubits);

        println!("Quantum-ness: {:.3}", analysis.quantum_ness);
        println!("Interference score: {:.3}", analysis.interference_score);

        if analysis.patterns.is_empty() {
            println!("Patterns: None detected");
        } else {
            println!("Patterns:");
            for (pattern, count) in &analysis.patterns {
                let name = pattern_name(*pattern);
                println!("  - {} (count: {})", name, count);
            }
        }

        println!("Circuit diagram:");
        println!("{}", circuit_diagram(circuit, config.n_qubits));
        println!();
    }

    // Analyze classical winners
    println!("======================================================================");
    println!(
        "  CLASSICAL WINNERS - Top {} Circuits",
        classical_circuits.len().min(10)
    );
    println!("======================================================================\n");

    for (i, circuit) in classical_circuits.iter().take(10).enumerate() {
        println!(
            "--- Classical Winner #{} ({} gates) ---",
            i + 1,
            circuit.len()
        );

        let analysis = detect_patterns(circuit, config.n_qubits);

        println!("Quantum-ness: {:.3}", analysis.quantum_ness);
        println!("Interference score: {:.3}", analysis.interference_score);

        if analysis.patterns.is_empty() {
            println!("Patterns: None detected");
        } else {
            println!("Patterns:");
            for (pattern, count) in &analysis.patterns {
                let name = pattern_name(*pattern);
                println!("  - {} (count: {})", name, count);
            }
        }

        println!("Circuit diagram:");
        println!("{}", circuit_diagram(circuit, config.n_qubits));
        println!();
    }

    // Summary comparison
    println!("======================================================================");
    println!("  SUMMARY COMPARISON");
    println!("======================================================================\n");

    let q_analysis = analyze_population_patterns(&quantum_circuits, config.n_qubits);
    let c_analysis = analyze_population_patterns(&classical_circuits, config.n_qubits);

    println!("                          QUANTUM      CLASSICAL");
    println!("-----------------------------------------------------");
    println!(
        "Population                {:>7}        {:>7}",
        quantum_stats.0, classical_stats.0
    );
    println!(
        "Max Generation            {:>7}        {:>7}",
        quantum_stats.1, classical_stats.1
    );
    println!(
        "Avg Circuit Length        {:>7.1}        {:>7.1}",
        q_analysis.avg_circuit_length, c_analysis.avg_circuit_length
    );
    println!(
        "Avg Quantum-ness          {:>7.3}        {:>7.3}",
        q_analysis.avg_quantum_ness, c_analysis.avg_quantum_ness
    );
    println!(
        "Avg Entangling Gates      {:>7.1}        {:>7.1}",
        q_analysis.avg_entangling_gates, c_analysis.avg_entangling_gates
    );
    println!();

    // Key findings
    println!("======================================================================");
    println!("  KEY FINDINGS");
    println!("======================================================================\n");

    if q_analysis.avg_quantum_ness > 0.4 {
        println!(
            "[+] Quantum organisms maintain high quantum-ness ({:.1}%)",
            q_analysis.avg_quantum_ness * 100.0
        );
    } else {
        println!(
            "[-] Quantum organisms have LOW quantum-ness ({:.1}%) - evolution may not be using entanglement",
            q_analysis.avg_quantum_ness * 100.0
        );
    }

    if q_analysis.avg_entangling_gates > 2.0 {
        println!(
            "[+] Quantum circuits use significant entanglement ({:.1} avg entangling gates)",
            q_analysis.avg_entangling_gates
        );
    } else {
        println!(
            "[?] Low entangling gate count ({:.1}) - advantage may come from interference alone",
            q_analysis.avg_entangling_gates
        );
    }

    if q_analysis.avg_circuit_length < c_analysis.avg_circuit_length {
        println!(
            "[+] Quantum circuits are shorter on average ({:.1} vs {:.1}) - more efficient?",
            q_analysis.avg_circuit_length, c_analysis.avg_circuit_length
        );
    } else if q_analysis.avg_circuit_length > c_analysis.avg_circuit_length {
        println!(
            "[?] Quantum circuits are longer ({:.1} vs {:.1}) - complexity needed for advantage?",
            q_analysis.avg_circuit_length, c_analysis.avg_circuit_length
        );
    }
}

fn pattern_name(pattern: QuantumPattern) -> &'static str {
    match pattern {
        QuantumPattern::BellPair => "Bell Pair",
        QuantumPattern::GHZSetup => "GHZ Setup",
        QuantumPattern::HadamardSandwich => "Hadamard Sandwich",
        QuantumPattern::PhaseKickback => "Phase Kickback",
        QuantumPattern::EntanglementLayer => "Entanglement Layer",
        QuantumPattern::UniformSuperposition => "Uniform Superposition",
        QuantumPattern::CNOTCascade => "CNOT Cascade",
        QuantumPattern::CZNetwork => "CZ Network",
        QuantumPattern::Unstructured => "Unstructured",
    }
}

/// Run a population and extract the circuits from survivors
fn run_and_extract(
    config: &QuantumForagerConfig,
    constraints: CircuitConstraints,
    ticks: u32,
    seed: u64,
) -> (Vec<Vec<engine::quantum::QGate>>, (usize, u32)) {
    let mut sim = Simulation::new();
    let mut pop = ConstrainedPopulation::new(config.clone(), constraints, seed);

    // Seed resources
    let region = &config.region;
    for i in 0..30 {
        let rx = region.x0
            + ((seed.wrapping_mul(0x123456789).wrapping_add(i * 17)) as u32 % region.width);
        let ry = region.y0
            + ((seed.wrapping_mul(0x987654321).wrapping_add(i * 31)) as u32 % region.height);
        sim.set_heat_field_for_test(rx as usize, ry as usize, 45);
    }

    // Run simulation
    for tick in 0..ticks {
        // Respawn resources periodically
        if tick % 8 == 0 {
            for i in 0..30 {
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
    }

    let stats = pop.stats();
    let circuits = pop.extract_all_circuits();

    (circuits, (stats.size, stats.max_generation))
}
