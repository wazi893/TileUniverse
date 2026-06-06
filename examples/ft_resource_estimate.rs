//! Fault-Tolerant Resource Estimation Example
//!
//! Demonstrates the fast resource estimation mode of the FaultTolerantProcessor.
//! This mode computes spacetime volume, qubit requirements, and error probability
//! without simulating individual gate outcomes.
//!
//! Run with: cargo run --example ft_resource_estimate

use engine::qec::{
    CycleCosts, FTConfig, FaultTolerantProcessor, LogicalGate, QecModel, compute_t_depth,
    estimate_factory_config,
};

fn main() {
    println!("=== Fault-Tolerant Resource Estimation ===\n");

    // Example 1: Simple Bell state creation with T gates
    println!("--- Example 1: Bell State + T Gates ---");
    simple_circuit_estimate();

    // Example 2: Grover-like circuit
    println!("\n--- Example 2: Grover-like Circuit (8 qubits) ---");
    grover_like_estimate();

    // Example 3: Large-scale estimation
    println!("\n--- Example 3: Large-Scale Circuit ---");
    large_scale_estimate();

    // Example 4: Compare different code distances
    println!("\n--- Example 4: Distance Comparison ---");
    distance_comparison();
}

fn simple_circuit_estimate() {
    let config = FTConfig::new(7, 1e-3);
    let processor = FaultTolerantProcessor::new(2, config);

    // Bell state + some T gates
    let circuit = vec![
        LogicalGate::H(0),
        LogicalGate::CNOT(0, 1),
        LogicalGate::T(0),
        LogicalGate::T(1),
        LogicalGate::CNOT(0, 1),
        LogicalGate::MeasureZ(0),
        LogicalGate::MeasureZ(1),
    ];

    let estimate = processor.estimate_resources(&circuit);

    println!("Circuit: H(0), CNOT(0,1), T(0), T(1), CNOT(0,1), MZ(0), MZ(1)");
    println!("  Logical qubits:   2");
    println!("  Code distance:    7");
    println!("  Physical error:   1e-3");
    println!();
    println!("Resource Estimate:");
    println!("  Total cycles:     {}", estimate.cycles);
    println!("  Data qubits:      {}", estimate.data_qubits);
    println!("  Factory qubits:   {}", estimate.factory_qubits);
    println!("  Total qubits:     {}", estimate.total_qubits);
    println!("  Spacetime volume: {}", estimate.spacetime_volume);
    println!();
    println!("Gate Statistics:");
    println!("  T-count:          {}", estimate.t_count);
    println!("  T-depth:          {}", estimate.t_depth);
    println!("  Clifford gates:   {}", estimate.clifford_count);
    println!("  Two-qubit gates:  {}", estimate.two_qubit_count);
    println!();
    println!("Error Budget:");
    println!("  QEC error:        {:.2e}", estimate.qec_error_probability);
    println!(
        "  Injection error:  {:.2e}",
        estimate.injection_error_probability
    );
    println!(
        "  Total error:      {:.2e}",
        estimate.total_error_probability
    );
    println!();
    println!("Time Estimate:");
    println!("  Runtime:          {:.2} us", estimate.runtime_us);
}

fn grover_like_estimate() {
    // 8-qubit circuit with multiple Toffoli gates (like Grover oracle)
    let config = FTConfig::new(9, 1e-3);
    let processor = FaultTolerantProcessor::new(8, config);

    // Build a Grover-like circuit
    let mut circuit = Vec::new();

    // Initial superposition
    for q in 0..8 {
        circuit.push(LogicalGate::H(q));
    }

    // Oracle (Toffoli cascade)
    circuit.push(LogicalGate::Toffoli(0, 1, 7));
    circuit.push(LogicalGate::Toffoli(2, 3, 7));
    circuit.push(LogicalGate::Toffoli(4, 5, 7));

    // Diffusion operator (H-X-MCZ-X-H pattern simplified)
    for q in 0..8 {
        circuit.push(LogicalGate::H(q));
        circuit.push(LogicalGate::X(q));
    }
    circuit.push(LogicalGate::CCZ(0, 1, 2));
    for q in 0..8 {
        circuit.push(LogicalGate::X(q));
        circuit.push(LogicalGate::H(q));
    }

    // Measurement
    for q in 0..8 {
        circuit.push(LogicalGate::MeasureZ(q));
    }

    let estimate = processor.estimate_resources(&circuit);
    let t_depth = compute_t_depth(&circuit);

    println!("Circuit: Grover-like with 3 Toffoli + 1 CCZ");
    println!("  Logical qubits:   8");
    println!("  Code distance:    9");
    println!();
    println!("Resource Estimate:");
    println!("  Total cycles:     {}", estimate.cycles);
    println!("  Total qubits:     {} physical", estimate.total_qubits);
    println!("  Spacetime volume: {}", estimate.spacetime_volume);
    println!();
    println!("T-Gate Analysis:");
    println!(
        "  T-count:          {} (from {} Toffoli + {} CCZ)",
        estimate.t_count, 3, 1
    );
    println!("  T-depth:          {}", t_depth);
    println!(
        "  Parallelism:      {:.1}x",
        estimate.t_count as f64 / t_depth.max(1) as f64
    );
    println!();
    println!("Error: {:.2e}", estimate.total_error_probability);
}

fn large_scale_estimate() {
    // Estimate for a "serious" quantum computation
    // Similar to what might be needed for early fault-tolerant applications

    let config = FTConfig::new(15, 1e-4)
        .with_cycle_costs(CycleCosts::scaled_by_distance(15))
        .with_target_error(1e-15);
    let processor = FaultTolerantProcessor::new(100, config);

    // Build a large circuit: 100 qubits, many layers
    let mut circuit = Vec::new();

    // Initial state preparation
    for q in 0..100 {
        circuit.push(LogicalGate::H(q));
    }

    // Multiple layers of entangling gates + T gates
    for _layer in 0..10 {
        // CNOT ladder
        for q in 0..99 {
            circuit.push(LogicalGate::CNOT(q, q + 1));
        }
        // T gates on every other qubit
        for q in (0..100).step_by(2) {
            circuit.push(LogicalGate::T(q));
        }
    }

    // Final measurements
    for q in 0..100 {
        circuit.push(LogicalGate::MeasureZ(q));
    }

    let estimate = processor.estimate_resources(&circuit);

    println!("Circuit: 100 qubits, 10 layers of CNOT ladder + T gates");
    println!("  Code distance:    15");
    println!("  Physical error:   1e-4");
    println!();
    println!("Scale:");
    println!("  Total gates:      {}", circuit.len());
    println!("  T-count:          {}", estimate.t_count);
    println!("  T-depth:          {}", estimate.t_depth);
    println!();
    println!("Resources:");
    println!(
        "  Total qubits:     {} ({:.1}M)",
        estimate.total_qubits,
        estimate.total_qubits as f64 / 1e6
    );
    println!(
        "    Data:           {} ({:.1}k)",
        estimate.data_qubits,
        estimate.data_qubits as f64 / 1e3
    );
    println!(
        "    Factory:        {} ({:.1}k)",
        estimate.factory_qubits,
        estimate.factory_qubits as f64 / 1e3
    );
    println!(
        "  Spacetime vol:    {} ({:.2}M qubit-cycles)",
        estimate.spacetime_volume,
        estimate.spacetime_volume as f64 / 1e6
    );
    println!();
    println!("Time (at 1 us/cycle):");
    println!(
        "  Runtime:          {:.2} us ({:.2} ms)",
        estimate.runtime_us,
        estimate.runtime_us / 1000.0
    );
    println!();
    println!("Error Budget:");
    println!(
        "  Total error:      {:.2e}",
        estimate.total_error_probability
    );
}

fn distance_comparison() {
    println!("Comparing resources for different code distances:");
    println!("(Circuit: 10 qubits, 100 T-gates, physical error 1e-3)\n");

    let mut circuit = Vec::new();
    for q in 0..10 {
        circuit.push(LogicalGate::H(q));
    }
    for _ in 0..10 {
        for q in 0..10 {
            circuit.push(LogicalGate::T(q));
        }
    }

    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12}",
        "d", "Qubits", "Cycles", "p_logical", "Suppression"
    );
    println!("{}", "-".repeat(56));

    for distance in [5, 7, 9, 11, 13, 15] {
        let config = FTConfig::new(distance, 1e-3);
        let processor = FaultTolerantProcessor::new(10, config);
        let estimate = processor.estimate_resources(&circuit);
        let qec_model = QecModel::new(distance, 1e-3);

        println!(
            "{:>4} {:>12} {:>12} {:>12.2e} {:>12.1}x",
            distance,
            estimate.total_qubits,
            estimate.cycles,
            estimate.total_error_probability,
            qec_model.suppression_factor()
        );
    }

    println!();
    println!("Key insight: Higher distance = more qubits but exponentially lower error");

    // Factory configuration recommendation
    println!("\n--- Factory Configuration for 1000 T-gates ---");
    let config = estimate_factory_config(1000, 50, 1e-3, 1e-10, 1000);
    println!("{}", config);
}
