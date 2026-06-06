//! Resource Estimation Dashboard Demo
//!
//! Sprint 56.0: Demonstrates comprehensive resource estimation
//! combining logical circuit metrics with fault-tolerant overhead.
//!
//! Run with: cargo run --example resource_estimate_demo

use engine::hardware::profiles;
use engine::transpile::{
    FaultTolerantConfig, ResourceEstimator, analyze_t_gates, estimate_resources,
    estimate_resources_ft,
};
use logic_fabric_core::quantum::QGate;

fn main() {
    println!("===========================================");
    println!(" SPRINT 56.0: Resource Estimation Dashboard");
    println!("===========================================\n");

    // Example 1: Simple Bell state (Clifford-only)
    demo_clifford_circuit();

    // Example 2: T-gate circuit with magic states
    demo_t_gate_circuit();

    // Example 3: Grover-like circuit with Toffoli decomposition
    demo_grover_circuit();

    // Example 4: Fault-tolerant estimation
    demo_fault_tolerant();

    // Example 5: Report generation
    demo_reports();
}

fn demo_clifford_circuit() {
    println!("=== 1. Clifford Circuit (Bell State) ===\n");

    let circuit = vec![
        QGate::H(0),
        QGate::CNot(0, 1),
        QGate::Measure(0),
        QGate::Measure(1),
    ];

    let estimate = estimate_resources(&circuit, "bell_state");

    println!("{}", estimate);
    println!(
        "Clifford-only: {} (no magic states needed)\n",
        estimate.is_clifford()
    );
}

fn demo_t_gate_circuit() {
    println!("=== 2. T-Gate Circuit ===\n");

    // Circuit with T gates requiring magic state distillation
    let circuit = vec![
        QGate::H(0),
        QGate::H(1),
        QGate::T(0),
        QGate::CNot(0, 1),
        QGate::T(1),
        QGate::Tdg(0),
        QGate::H(0),
        QGate::T(0),
        QGate::Measure(0),
        QGate::Measure(1),
    ];

    let estimate = estimate_resources(&circuit, "t_gate_demo");
    let t_analysis = analyze_t_gates(&circuit);

    println!("{}", estimate);
    println!("T-Gate Analysis:");
    println!("  T-count: {}", t_analysis.t_count);
    println!("  T-depth: {}", t_analysis.t_depth);
    println!("  T-ratio: {:.2}%\n", t_analysis.t_ratio() * 100.0);
}

fn demo_grover_circuit() {
    println!("=== 3. Grover-like Circuit (Toffoli Gates) ===\n");

    // 3-qubit Grover-like circuit with Toffoli (decomposed to T gates)
    let circuit = vec![
        // Initialize superposition
        QGate::H(0),
        QGate::H(1),
        QGate::H(2),
        // Oracle (Toffoli-based marking)
        QGate::Toffoli(0, 1, 2),
        // Diffusion
        QGate::H(0),
        QGate::H(1),
        QGate::X(0),
        QGate::X(1),
        QGate::CZ(0, 1),
        QGate::X(0),
        QGate::X(1),
        QGate::H(0),
        QGate::H(1),
        QGate::Measure(0),
        QGate::Measure(1),
        QGate::Measure(2),
    ];

    let estimate = estimate_resources(&circuit, "grover_3q");

    println!("{}", estimate);
    println!("Note: Toffoli decomposes to 7 T-gates\n");
}

fn demo_fault_tolerant() {
    println!("=== 4. Fault-Tolerant Resource Estimation ===\n");

    // Larger circuit for FT estimation
    let mut circuit = Vec::new();

    // 10-qubit circuit with various gates
    for i in 0..10 {
        circuit.push(QGate::H(i));
    }
    for i in 0..9 {
        circuit.push(QGate::CNot(i, i + 1));
    }
    for i in 0..5 {
        circuit.push(QGate::T(i));
    }
    circuit.push(QGate::Toffoli(0, 1, 2));
    circuit.push(QGate::Toffoli(3, 4, 5));
    for i in 0..10 {
        circuit.push(QGate::Measure(i));
    }

    // Conservative FT configuration (1e-10 target error)
    let ft_config = FaultTolerantConfig::for_error_rate(1e-10);
    let estimate = estimate_resources_ft(&circuit, "ft_demo", ft_config.clone());

    println!("{}", estimate);

    if let Some(phys) = &estimate.physical_estimate {
        println!("Physical Resource Summary:");
        println!("  Code distance: {}", phys.code_distance);
        println!(
            "  Physical qubits per logical: {}",
            phys.physical_per_logical
        );
        println!("  Total physical qubits: {}", phys.total_physical_qubits);
        println!("  Magic state factories: {}", phys.num_factories);
    }

    if let Some(timing) = &estimate.ft_timing {
        println!("\nTiming Estimate:");
        println!("  Total time: {:.2} ms", timing.total_time_ns / 1_000_000.0);
        println!("  Bottleneck: {}", timing.bottleneck);
    }
    println!();
}

fn demo_reports() {
    println!("=== 5. Report Generation ===\n");

    let circuit = vec![
        QGate::H(0),
        QGate::T(0),
        QGate::CNot(0, 1),
        QGate::T(1),
        QGate::Rz(1, std::f32::consts::FRAC_PI_4),
        QGate::CZ(0, 1),
        QGate::Measure(0),
        QGate::Measure(1),
    ];

    // Use builder pattern with NISQ timing
    let profile = profiles::ibm_heron();
    let estimator = ResourceEstimator::new().with_nisq_profile(profile);

    let estimate = estimator.estimate(&circuit, "report_demo");

    // JSON Report
    println!("--- JSON Report ---");
    let json = estimate.to_json();
    // Pretty print first 500 chars
    let json_preview: String = json.chars().take(500).collect();
    println!("{}", json_preview);
    if json.len() > 500 {
        println!("... ({} total chars)\n", json.len());
    }

    // Markdown Report
    println!("\n--- Markdown Report ---");
    let md = estimate.to_markdown();
    println!("{}", md);
}
