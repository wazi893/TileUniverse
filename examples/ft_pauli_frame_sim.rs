//! Fault-Tolerant Pauli Frame Simulation Example
//!
//! Demonstrates the sampled simulation mode of the FaultTolerantProcessor.
//! This mode tracks Pauli frame updates from T-gate injection measurements
//! and simulates the classical control flow of fault-tolerant circuits.
//!
//! Run with: cargo run --example ft_pauli_frame_sim

use engine::qec::{FTConfig, FaultTolerantProcessor, LogicalGate};

fn main() {
    println!("=== Fault-Tolerant Pauli Frame Simulation ===\n");

    // Example 1: T-gate injection and frame tracking
    println!("--- Example 1: T-Gate Frame Tracking ---");
    t_gate_frame_tracking();

    // Example 2: Teleportation circuit
    println!("\n--- Example 2: Quantum Teleportation ---");
    teleportation_circuit();

    // Example 3: Multiple runs for statistics
    println!("\n--- Example 3: Statistical Analysis ---");
    statistical_analysis();

    // Example 4: Toffoli decomposition
    println!("\n--- Example 4: Toffoli Gate Simulation ---");
    toffoli_simulation();
}

fn t_gate_frame_tracking() {
    // Use sampled injection mode with deterministic outcomes for demo
    let config = FTConfig::new(7, 1e-3).with_sampled_injection();
    let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

    println!("Simulating: |0> -> H -> T -> T -> MZ");
    println!();

    // Step through the circuit
    let circuit = vec![
        LogicalGate::H(0),
        LogicalGate::T(0),
        LogicalGate::T(0),
        LogicalGate::MeasureZ(0),
    ];

    let result = processor.execute(&circuit);

    println!("Execution Result:");
    println!("  Cycles used:      {}", result.cycles_used);
    println!("  T-gates executed: {}", result.stats.t_gates);
    println!("  Magic states:     {}", result.stats.magic_states_consumed);
    println!();

    println!("Final State:");
    println!("  Qubit 0 frame:    {}", result.final_frames[0]);
    println!();

    println!("Measurement:");
    for m in &result.measurement_outcomes {
        println!(
            "  {}: {} (frame at measurement: {})",
            m,
            if m.value { "1" } else { "0" },
            m.frame_at_measurement
        );
    }
    println!();

    println!("Error Budget:");
    println!("  Total error:      {:.2e}", result.error_budget.total());
    println!("  From QEC:         {:.2e}", result.error_budget.from_qec);
    println!(
        "  From injection:   {:.2e}",
        result.error_budget.from_injection
    );
}

fn teleportation_circuit() {
    // Quantum teleportation: transfer state from qubit 0 to qubit 2
    // Uses qubits: 0 (state to teleport), 1-2 (Bell pair)

    let config = FTConfig::new(7, 1e-3).with_sampled_injection();
    let mut processor = FaultTolerantProcessor::new(3, config).with_seed(12345);

    println!("Teleporting state |+> from qubit 0 to qubit 2");
    println!();

    let circuit = vec![
        // Prepare |+> state on qubit 0 (state to teleport)
        LogicalGate::PreparePlus(0),
        // Create Bell pair between qubits 1 and 2
        LogicalGate::H(1),
        LogicalGate::CNOT(1, 2),
        // Bell measurement on qubits 0 and 1
        LogicalGate::CNOT(0, 1),
        LogicalGate::H(0),
        LogicalGate::MeasureZ(0),
        LogicalGate::MeasureZ(1),
        // Corrections would be applied based on measurement outcomes
        // (In real FT, this is tracked in the Pauli frame)
        // Final measurement to verify teleportation
        LogicalGate::MeasureX(2), // Should give |+> state
    ];

    let result = processor.execute(&circuit);

    println!("Teleportation Circuit Executed:");
    println!("  Total cycles:     {}", result.cycles_used);
    println!();

    println!("Measurement Outcomes:");
    for m in &result.measurement_outcomes {
        let basis = match m.basis {
            engine::qec::MeasurementBasis::Z => "Z",
            engine::qec::MeasurementBasis::X => "X",
        };
        println!(
            "  Qubit {}: M{} = {} (frame: {})",
            m.qubit,
            basis,
            if m.value { "1" } else { "0" },
            m.frame_at_measurement
        );
    }
    println!();

    println!("Final Pauli Frames:");
    for (i, frame) in result.final_frames.iter().enumerate() {
        println!("  Qubit {}: {}", i, frame);
    }
    println!();

    // Interpret results
    let m0 = result.measurement_outcomes[0].value;
    let m1 = result.measurement_outcomes[1].value;
    println!("Teleportation Corrections Needed:");
    println!(
        "  X correction on qubit 2: {}",
        if m1 { "Yes" } else { "No" }
    );
    println!(
        "  Z correction on qubit 2: {}",
        if m0 { "Yes" } else { "No" }
    );
}

fn statistical_analysis() {
    // Run many iterations to see frame distribution
    let n_runs = 100;
    let mut z_frame_count = 0;
    let mut measurement_ones = 0;

    println!("Running {} simulations of: |0> -> H -> T -> MZ", n_runs);
    println!();

    for seed in 0..n_runs {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(seed);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::MeasureZ(0),
        ];

        let result = processor.execute(&circuit);

        // Count frames with Z component at measurement time
        // (final_frames is identity after measurement; we need frame_at_measurement)
        if result.measurement_outcomes[0].frame_at_measurement.z {
            z_frame_count += 1;
        }

        // Count measurement outcomes
        if result.measurement_outcomes[0].value {
            measurement_ones += 1;
        }
    }

    println!("Statistics over {} runs:", n_runs);
    println!(
        "  Z frame at measurement: {}/{} ({:.1}%)",
        z_frame_count,
        n_runs,
        100.0 * z_frame_count as f64 / n_runs as f64
    );
    println!(
        "  Measurement = 1:        {}/{} ({:.1}%)",
        measurement_ones,
        n_runs,
        100.0 * measurement_ones as f64 / n_runs as f64
    );
    println!();

    println!("Expected: ~50% Z frames (from T injection measurement)");
    println!("Expected: ~50% measurement ones (|+> state in Z basis)");
}

fn toffoli_simulation() {
    // Simulate a Toffoli gate and observe the T-gate consumption

    let config = FTConfig::new(7, 1e-3).with_sampled_injection();
    let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

    println!("Simulating Toffoli(0,1,2) on |110>");
    println!("Expected: |110> -> |111> (flips target when both controls are 1)");
    println!();

    let circuit = vec![
        // Prepare |110>
        LogicalGate::X(0),
        LogicalGate::X(1),
        // Apply Toffoli
        LogicalGate::Toffoli(0, 1, 2),
        // Measure all qubits
        LogicalGate::MeasureZ(0),
        LogicalGate::MeasureZ(1),
        LogicalGate::MeasureZ(2),
    ];

    let result = processor.execute(&circuit);

    println!("Toffoli Execution:");
    println!("  Cycles used:          {}", result.cycles_used);
    println!(
        "  T-gates consumed:     {} (standard decomposition)",
        result.stats.t_gates
    );
    println!(
        "  Magic states used:    {}",
        result.stats.magic_states_consumed
    );
    println!(
        "  Factory delays:       {} cycles",
        result.stats.factory_delays
    );
    println!();

    println!("Measurement Results:");
    for m in &result.measurement_outcomes {
        println!("  Qubit {}: {}", m.qubit, if m.value { "1" } else { "0" });
    }
    println!();

    println!("Error Accumulation:");
    println!(
        "  Total error:          {:.2e}",
        result.error_budget.total()
    );
    println!(
        "  From {} T-injections: {:.2e}",
        result.stats.t_gates, result.error_budget.from_injection
    );
    println!();

    // Compare with resource estimate
    let estimate = processor.estimate_resources(&circuit[3..4]); // Just Toffoli
    println!("Resource Estimate for Toffoli only:");
    println!("  T-count:              {}", estimate.t_count);
    println!("  Estimated cycles:     {}", estimate.cycles);
}
