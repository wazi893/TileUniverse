//! Hardware Transpilation Benchmarks
//!
//! SPRINT 53.0: Hardware Transpilation & T-Gate Tracking
//! SPRINT 54.0: Optimization Pipeline Integration
//!
//! Compares circuit transpilation across hardware platforms:
//! 1. IBM Falcon/Heron (superconducting)
//! 2. IonQ Aria (trapped ion)
//!
//! Run: cargo run --release --example transpile_bench

use engine::hardware::profiles;
use engine::qram::DistillationConfig;
use engine::transpile::{
    IBMTranspiler, IBMTranspilerConfig, IonQTranspiler, OptimizationConfig, ToffoliDecomposition,
    analyze_abstract_circuit, analyze_ibm_circuit, analyze_ionq_circuit, analyze_t_depth,
    analyze_t_gates, estimate_magic_cost, optimize_circuit, optimize_ibm_circuit,
};
use logic_fabric_core::quantum::QGate;
use std::f32::consts::PI;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║      SPRINT 53/54: Hardware Transpilation & Optimization             ║");
    println!("║      IBM vs IonQ Circuit Compilation                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Part 1: Bell State
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Part 1: Bell State Circuit");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    transpile_bell_state();

    // Part 2: GHZ State
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 2: GHZ State Circuit (4 qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    transpile_ghz_state();

    // Part 3: T-Gate Analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 3: T-Gate Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    t_gate_analysis();

    // Part 4: Grover-like Circuit
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 4: Grover-like Circuit (3 qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    transpile_grover_like();

    // Part 5: Hardware Comparison Table
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 5: Hardware Comparison Summary");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    hardware_comparison_summary();

    // Part 6: Magic State Cost Integration
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 6: Magic State Cost Integration (Sprint 51)");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    magic_state_integration();

    // Part 7: Sprint 54 Optimization Pipeline
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 7: Sprint 54 Optimization Pipeline");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    optimization_pipeline_demo();

    // Part 8: Toffoli Decomposition Options
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 8: Toffoli Decomposition Options");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    toffoli_decomposition_demo();

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SPRINT 53/54 Hardware Transpilation & Optimization Complete!");
    println!("═══════════════════════════════════════════════════════════════════════");
}

fn transpile_bell_state() {
    let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];

    println!("Abstract circuit:");
    for gate in &circuit {
        println!("  {:?}", gate);
    }
    println!();

    // IBM transpilation
    let ibm = IBMTranspiler::heron();
    let ibm_circuit = ibm.transpile(&circuit, "bell");
    let ibm_analysis = analyze_ibm_circuit(&ibm_circuit, &profiles::ibm_heron());

    println!("IBM Heron transpilation:");
    println!("  Gates: {}", ibm_circuit.gate_count());
    println!("  CX count: {}", ibm_circuit.count("cx"));
    println!("  SX count: {}", ibm_circuit.count("sx"));
    println!("  Rz count: {}", ibm_circuit.count("rz"));
    println!("  Depth: {}", ibm_circuit.depth());
    println!("  Duration: {:.2} ns", ibm_circuit.total_duration_ns());
    println!("  Est. fidelity: {:.4}", ibm_analysis.estimated_fidelity);
    println!();

    // IonQ transpilation
    let ionq = IonQTranspiler::aria();
    let ionq_circuit = ionq.transpile(&circuit, "bell");
    let ionq_analysis = analyze_ionq_circuit(&ionq_circuit, &profiles::ionq_aria());

    println!("IonQ Aria transpilation:");
    println!("  Gates: {}", ionq_circuit.gate_count());
    println!("  MS count: {}", ionq_circuit.count("ms"));
    println!("  GPI count: {}", ionq_circuit.count("gpi"));
    println!("  GPI2 count: {}", ionq_circuit.count("gpi2"));
    println!("  Depth: {}", ionq_circuit.depth());
    println!(
        "  Duration: {:.2} µs",
        ionq_circuit.total_duration_ns() / 1000.0
    );
    println!("  Est. fidelity: {:.4}", ionq_analysis.estimated_fidelity);
    println!();

    // Speed comparison
    let speedup = ionq_circuit.total_duration_ns() / ibm_circuit.total_duration_ns();
    println!("IBM is {:.1}x faster than IonQ", speedup);
}

fn transpile_ghz_state() {
    // GHZ: H(0), CNOT(0,1), CNOT(0,2), CNOT(0,3)
    let circuit = vec![
        QGate::H(0),
        QGate::CNot(0, 1),
        QGate::CNot(0, 2),
        QGate::CNot(0, 3),
    ];

    let abstract_analysis = analyze_abstract_circuit(&circuit, "ghz4");
    println!("Abstract circuit:");
    println!("{}", abstract_analysis);

    // IBM
    let ibm = IBMTranspiler::heron();
    let ibm_circuit = ibm.transpile(&circuit, "ghz4");
    let ibm_analysis = analyze_ibm_circuit(&ibm_circuit, &profiles::ibm_heron());

    println!("IBM Heron:");
    println!("  Total gates: {}", ibm_circuit.gate_count());
    println!("  CX count: {}", ibm_circuit.count("cx"));
    println!("  Duration: {:.2} ns", ibm_circuit.total_duration_ns());
    println!("  Fidelity: {:.4}", ibm_analysis.estimated_fidelity);
    println!();

    // IonQ
    let ionq = IonQTranspiler::aria();
    let ionq_circuit = ionq.transpile(&circuit, "ghz4");
    let ionq_analysis = analyze_ionq_circuit(&ionq_circuit, &profiles::ionq_aria());

    println!("IonQ Aria:");
    println!("  Total gates: {}", ionq_circuit.gate_count());
    println!("  MS count: {}", ionq_circuit.count("ms"));
    println!(
        "  Duration: {:.2} µs",
        ionq_circuit.total_duration_ns() / 1000.0
    );
    println!("  Fidelity: {:.4}", ionq_analysis.estimated_fidelity);
}

fn t_gate_analysis() {
    // Circuit with T-gates
    let circuit = vec![
        QGate::H(0),
        QGate::T(0),
        QGate::CNot(0, 1),
        QGate::T(1),
        QGate::T(0),
        QGate::Tdg(1),
        QGate::CNot(0, 1),
        QGate::H(0),
    ];

    let t_analysis = analyze_t_gates(&circuit);
    println!("{}", t_analysis);

    // Transpile to see how T-gates become Rz
    let ibm = IBMTranspiler::heron();
    let ibm_circuit = ibm.transpile(&circuit, "t_test");

    println!("\nIBM transpilation:");
    println!(
        "  Rz gates: {} (includes T → Rz(π/4))",
        ibm_circuit.count("rz")
    );
    println!("  CX gates: {}", ibm_circuit.count("cx"));
}

fn transpile_grover_like() {
    // Simplified Grover-like circuit (oracle + diffusion)
    let circuit = vec![
        // Initialize superposition
        QGate::H(0),
        QGate::H(1),
        QGate::H(2),
        // Oracle (mark |110⟩)
        QGate::X(2),
        QGate::Toffoli(0, 1, 2),
        QGate::X(2),
        // Diffusion
        QGate::H(0),
        QGate::H(1),
        QGate::H(2),
        QGate::X(0),
        QGate::X(1),
        QGate::X(2),
        QGate::Toffoli(0, 1, 2),
        QGate::X(0),
        QGate::X(1),
        QGate::X(2),
        QGate::H(0),
        QGate::H(1),
        QGate::H(2),
    ];

    let abstract_analysis = analyze_abstract_circuit(&circuit, "grover_like");
    println!("Abstract circuit:");
    println!("  Gates: {}", abstract_analysis.total_gates);
    println!("  Depth: {}", abstract_analysis.depth);
    println!(
        "  T-count: {} (from Toffoli decomposition)",
        abstract_analysis.t_analysis.t_count
    );
    println!();

    // IBM
    let ibm = IBMTranspiler::heron();
    let ibm_circuit = ibm.transpile(&circuit, "grover_like");
    let ibm_analysis = analyze_ibm_circuit(&ibm_circuit, &profiles::ibm_heron());

    println!("IBM Heron:");
    println!("  Total gates: {}", ibm_circuit.gate_count());
    println!("  CX count: {}", ibm_circuit.count("cx"));
    println!("  Depth: {}", ibm_circuit.depth());
    println!(
        "  Duration: {:.2} µs",
        ibm_circuit.total_duration_ns() / 1000.0
    );
    println!("  Fidelity: {:.6}", ibm_analysis.estimated_fidelity);
    println!();

    // IonQ
    let ionq = IonQTranspiler::aria();
    let ionq_circuit = ionq.transpile(&circuit, "grover_like");
    let ionq_analysis = analyze_ionq_circuit(&ionq_circuit, &profiles::ionq_aria());

    println!("IonQ Aria:");
    println!("  Total gates: {}", ionq_circuit.gate_count());
    println!("  MS count: {}", ionq_circuit.count("ms"));
    println!("  Depth: {}", ionq_circuit.depth());
    println!(
        "  Duration: {:.2} ms",
        ionq_circuit.total_duration_ns() / 1_000_000.0
    );
    println!("  Fidelity: {:.6}", ionq_analysis.estimated_fidelity);
}

fn hardware_comparison_summary() {
    let circuits = vec![
        ("Bell", vec![QGate::H(0), QGate::CNot(0, 1)]),
        (
            "GHZ-4",
            vec![
                QGate::H(0),
                QGate::CNot(0, 1),
                QGate::CNot(0, 2),
                QGate::CNot(0, 3),
            ],
        ),
        (
            "T-heavy",
            vec![
                QGate::H(0),
                QGate::T(0),
                QGate::T(0),
                QGate::T(0),
                QGate::T(0),
                QGate::CNot(0, 1),
            ],
        ),
    ];

    println!(
        "{:>12} {:>8} {:>8} {:>12} {:>12} {:>10} {:>10}",
        "Circuit", "Qubits", "Gates", "IBM_dur(ns)", "IonQ_dur(µs)", "IBM_fid", "IonQ_fid"
    );
    println!("{}", "-".repeat(80));

    let ibm_transpiler = IBMTranspiler::heron();
    let ionq_transpiler = IonQTranspiler::aria();

    for (name, circuit) in circuits {
        let abstract_analysis = analyze_abstract_circuit(&circuit, name);

        let ibm_circuit = ibm_transpiler.transpile(&circuit, name);
        let ibm_analysis = analyze_ibm_circuit(&ibm_circuit, &profiles::ibm_heron());

        let ionq_circuit = ionq_transpiler.transpile(&circuit, name);
        let ionq_analysis = analyze_ionq_circuit(&ionq_circuit, &profiles::ionq_aria());

        println!(
            "{:>12} {:>8} {:>8} {:>12.1} {:>12.1} {:>10.4} {:>10.4}",
            name,
            abstract_analysis.num_qubits,
            abstract_analysis.total_gates,
            ibm_circuit.total_duration_ns(),
            ionq_circuit.total_duration_ns() / 1000.0,
            ibm_analysis.estimated_fidelity,
            ionq_analysis.estimated_fidelity
        );
    }
}

fn magic_state_integration() {
    // Circuit with multiple T-gates
    let circuit = vec![
        QGate::H(0),
        QGate::H(1),
        QGate::T(0),
        QGate::T(1),
        QGate::CNot(0, 1),
        QGate::T(0),
        QGate::Tdg(1),
        QGate::CNot(0, 1),
        QGate::T(1),
        QGate::H(0),
        QGate::H(1),
    ];

    let t_analysis = analyze_t_gates(&circuit);
    println!("Circuit T-gate analysis:");
    println!("  T-count: {}", t_analysis.t_count);
    println!("  T-depth: {}", t_analysis.t_depth);
    println!("  Clifford gates: {}", t_analysis.clifford_count);
    println!();

    // Estimate magic state costs at different error rates
    let error_rates = [1e-6, 1e-9, 1e-12, 1e-15];

    println!(
        "{:>12} {:>12} {:>15} {:>15}",
        "Target Err", "Protocol", "Magic States", "Factory Cycles"
    );
    println!("{}", "-".repeat(60));

    for &error_rate in &error_rates {
        let distillation = DistillationConfig::for_error_rate(error_rate);
        let cost = estimate_magic_cost(&t_analysis, &distillation);

        println!(
            "{:>12.0e} {:>12?} {:>15} {:>15}",
            error_rate, distillation.protocol, cost.distilled_magic_states, cost.factory_cycles
        );
    }

    println!();
    println!("Integration with Sprint 51 MagicStateBudget:");

    let distillation = DistillationConfig::for_error_rate(1e-12);
    let cost = estimate_magic_cost(&t_analysis, &distillation);
    let budget = cost.to_budget(&distillation);

    println!("  T-gates in circuit: {}", t_analysis.t_count);
    println!("  Magic states needed: {}", budget.total_magic_states);
    println!("  Factory cycles: {}", budget.total_factory_cycles);
    println!("  Distillation protocol: {:?}", distillation.protocol);
}

fn optimization_pipeline_demo() {
    println!("SPRINT 54.0: Pre-Transpile Optimization\n");

    // Circuit with redundant gates that can be optimized
    let circuit = vec![
        QGate::H(0),
        QGate::H(0), // H·H = I (should cancel)
        QGate::X(0),
        QGate::Rz(0, PI / 4.0),
        QGate::Rz(0, PI / 4.0), // Rz·Rz should merge
        QGate::CNot(0, 1),
        QGate::CNot(0, 1), // CNOT·CNOT = I (should cancel)
        QGate::X(1),
        QGate::X(1), // X·X = I (should cancel)
    ];

    println!("Original circuit ({} gates):", circuit.len());
    for gate in &circuit {
        println!("  {:?}", gate);
    }
    println!();

    // Compare optimization levels
    let levels = [
        ("None", OptimizationConfig::none()),
        ("Basic", OptimizationConfig::basic()),
        ("Standard", OptimizationConfig::standard()),
        ("Aggressive", OptimizationConfig::aggressive()),
    ];

    println!(
        "{:>12} {:>10} {:>12} {:>10}",
        "Level", "Gates", "Reduction", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for (name, config) in levels {
        let result = optimize_circuit(&circuit, &config);
        println!(
            "{:>12} {:>10} {:>11.1}% {:>9.2}x",
            name,
            result.stats.final_gates,
            result.stats.reduction_percent(),
            result.stats.speedup_factor()
        );
    }

    // Show optimized circuit
    let optimized = optimize_circuit(&circuit, &OptimizationConfig::aggressive());
    println!("\nOptimized circuit ({} gates):", optimized.circuit.len());
    for gate in &optimized.circuit {
        println!("  {:?}", gate);
    }

    // T-depth analysis
    println!("\n--- T-Depth Analysis ---\n");

    let t_circuit = vec![
        QGate::T(0),
        QGate::T(1), // Can be parallel with T(0)
        QGate::T(2), // Can be parallel with T(0) and T(1)
        QGate::CNot(0, 1),
        QGate::T(0),
        QGate::T(1),
    ];

    let t_analysis = analyze_t_depth(&t_circuit);
    println!("Circuit with {} T-gates:", t_analysis.t_count);
    println!("  Original T-depth:  {}", t_analysis.original_t_depth);
    println!("  Optimized T-depth: {}", t_analysis.optimized_t_depth);
    println!(
        "  Parallelization:   {:.1}x",
        t_analysis.parallelization_factor()
    );
    println!(
        "  T-depth reduction: {:.1}%",
        t_analysis.reduction_percent()
    );

    // Post-transpile optimization
    println!("\n--- Post-Transpile IBM Native Optimization ---\n");

    let ibm = IBMTranspiler::heron();
    let ibm_circuit = ibm.transpile(&circuit, "redundant");

    println!(
        "Before native optimization: {} gates",
        ibm_circuit.gates.len()
    );

    let (optimized_ibm, ibm_stats) = optimize_ibm_circuit(&ibm_circuit);

    println!(
        "After native optimization:  {} gates",
        optimized_ibm.gates.len()
    );
    println!("  CX cancelled: {}", ibm_stats.cx_cancelled);
    println!("  Rz merged:    {}", ibm_stats.rz_merged);
    println!("  Rz eliminated:{}", ibm_stats.rz_eliminated);
    println!("  Reduction:    {:.1}%", ibm_stats.reduction_percent());
}

fn toffoli_decomposition_demo() {
    println!("SPRINT 54.0: Toffoli Decomposition Options\n");

    // Circuit with Toffoli gate
    let circuit = vec![QGate::H(0), QGate::Toffoli(0, 1, 2), QGate::H(0)];

    println!("Abstract circuit with Toffoli:");
    for gate in &circuit {
        println!("  {:?}", gate);
    }
    println!();

    // Compare decomposition strategies
    let strategies = [
        ("Standard", ToffoliDecomposition::Standard),
        ("T-Depth-1", ToffoliDecomposition::TDepthOne),
        ("Relative", ToffoliDecomposition::RelativePhase),
    ];

    println!(
        "{:>12} {:>8} {:>8} {:>8} {:>12}",
        "Strategy", "CX", "T gates", "T-depth", "Total gates"
    );
    println!("{}", "-".repeat(55));

    for (name, strategy) in strategies {
        let config = IBMTranspilerConfig {
            toffoli_decomposition: strategy,
            ..Default::default()
        };

        let ibm = IBMTranspiler::heron().with_config(config);
        let ibm_circuit = ibm.transpile(&circuit, "toffoli_test");

        let cx_count = ibm_circuit.count("cx");

        println!(
            "{:>12} {:>8} {:>8} {:>8} {:>12}",
            name,
            cx_count,
            strategy.t_count(),
            strategy.t_depth(),
            ibm_circuit.gate_count()
        );
    }

    println!();
    println!("Trade-offs:");
    println!("  - Standard:   Balanced CX and T count");
    println!("  - T-Depth-1:  More CX, but T gates parallelizable");
    println!("  - Relative:   Fewest gates, but phase error");
}
