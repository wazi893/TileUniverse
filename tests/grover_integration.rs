// =============================================================================
// tests/grover_integration.rs
// =============================================================================
//
// Integration tests for Grover's algorithm running on QDemo tiles.
// These tests verify the quantum→classical feedback loop works correctly
// with a real hybrid algorithm.
//
// =============================================================================

use engine::algorithms::grover::{grover_circuit, optimal_iterations};
use engine::quantum::QState;
use engine::simulation::Simulation;
use engine::tile_meta::TileType;
use std::sync::atomic::Ordering;

/// Test that Grover's algorithm finds the marked state when run on a QDemo tile.
/// This is the key integration test proving the feedback loop works.
#[test]
fn test_grover_on_qdemo_tile_finds_marked_state() {
    let mut sim = Simulation::new();

    // Setup: 3-qubit Grover searching for |101⟩
    let n_qubits = 3u8;
    let marked_state = 0b101u64;
    let iterations = optimal_iterations(n_qubits); // Should be 2

    let program = grover_circuit(n_qubits, marked_state, iterations);
    let program_len = program.len();

    // Register QDemo tile at position (50, 50)
    let state = QState::new_zero(n_qubits);
    sim.register_qdemo_tile(50, 50, state, program, 0x12345678);

    // Run simulation: one gate per tick
    for _ in 0..program_len {
        sim.tick();
    }

    // Read measurement result from tile logic output
    let result = sim.tilemap.value_at(50, 50).unwrap();

    // With the specific seed and optimal iterations, should find marked state
    // Note: Due to quantum randomness, this might occasionally fail with a
    // different result. For deterministic testing, we check that the result
    // is at least a valid 3-bit value.
    assert!(result <= 0b111, "Result should be 3-bit value");

    // For the specific seed 0x12345678, verify we get the marked state
    // (If this fails intermittently, the seed may need adjustment or
    // we should test over multiple trials)
    assert_eq!(
        result, marked_state,
        "Grover should find marked state |101⟩; got {:03b}",
        result
    );
}

/// Test that measurement results propagate to neighboring classical tiles.
#[test]
fn test_grover_result_propagates_to_neighbors() {
    let mut sim = Simulation::new();

    // Simple 2-qubit Grover to find |11⟩
    let program = grover_circuit(2, 0b11, 1);
    let program_len = program.len();

    sim.register_qdemo_tile(50, 50, QState::new_zero(2), program, 0xABCDEF);

    // Place Wire tiles as neighbors
    sim.set_tile(51, 50, TileType::Wire); // Right
    sim.set_tile(50, 51, TileType::Wire); // Down

    // Run the quantum circuit
    for _ in 0..program_len {
        sim.tick();
    }

    // Evaluate neighbors to propagate
    sim.eval_at(51, 50);
    sim.eval_at(50, 51);

    // QDemo tile output
    let qdemo_logic = sim.tilemap.value_at(50, 50).unwrap();

    // Neighbors should have received the signal
    let right_logic = sim.tilemap.value_at(51, 50).unwrap();
    let down_logic = sim.tilemap.value_at(50, 51).unwrap();

    // Wire tiles OR their inputs, so they should have the QDemo output
    assert!(
        right_logic != 0 || qdemo_logic == 0,
        "Right Wire should propagate QDemo signal"
    );
    assert!(
        down_logic != 0 || qdemo_logic == 0,
        "Down Wire should propagate QDemo signal"
    );
}

/// Test running Grover over multiple trials to measure success rate.
#[test]
fn test_grover_success_rate_3_qubits() {
    let n_qubits = 3u8;
    let marked_state = 0b101u64;
    let iterations = optimal_iterations(n_qubits);
    let trials = 50;

    let mut successes = 0;

    for trial in 0..trials {
        let mut sim = Simulation::new();
        let seed = (trial as u64) * 0x1234567890;
        let program = grover_circuit(n_qubits, marked_state, iterations);
        let program_len = program.len();

        sim.register_qdemo_tile(50, 50, QState::new_zero(n_qubits), program, seed);

        for _ in 0..program_len {
            sim.tick();
        }

        let result = sim.tilemap.value_at(50, 50).unwrap();

        if result == marked_state {
            successes += 1;
        }
    }

    let success_rate = successes as f64 / trials as f64;

    // Theoretical success rate for 3 qubits with 2 iterations is ~94.5%
    // Allow some variance for small sample size
    assert!(
        success_rate > 0.7,
        "Grover success rate {} is too low (expected >0.7)",
        success_rate
    );

    println!(
        "Grover 3-qubit success rate: {:.1}% ({}/{})",
        success_rate * 100.0,
        successes,
        trials
    );
}

/// Test that different marked states all work.
#[test]
fn test_grover_different_marked_states() {
    let n_qubits = 3u8;
    let iterations = optimal_iterations(n_qubits);

    // Test each possible marked state
    for marked in 0..8u64 {
        let mut success_count = 0;
        let trials = 20;

        for trial in 0..trials {
            let mut sim = Simulation::new();
            let seed = marked * 1000 + trial as u64;
            let program = grover_circuit(n_qubits, marked, iterations);
            let program_len = program.len();

            sim.register_qdemo_tile(50, 50, QState::new_zero(n_qubits), program, seed);

            for _ in 0..program_len {
                sim.tick();
            }

            let result = sim.tilemap.value_at(50, 50).unwrap();

            if result == marked {
                success_count += 1;
            }
        }

        let rate = success_count as f64 / trials as f64;
        assert!(
            rate > 0.5,
            "Marked state {:03b} has low success rate: {:.1}%",
            marked,
            rate * 100.0
        );
    }
}

/// Benchmark: Compare quantum vs classical search.
#[test]
fn test_grover_vs_classical_iterations() {
    // For 3 qubits (8 states):
    // - Classical random search: expected 4 queries to find marked state
    // - Grover: 2 iterations to find marked state with ~94.5% probability

    let n_qubits = 3u8;
    let marked_state = 0b101u64;

    // Grover: count circuit depth (proxy for time)
    let grover_iters = optimal_iterations(n_qubits);
    let grover_circuit = grover_circuit(n_qubits, marked_state, grover_iters);
    let grover_gates = grover_circuit.len();

    // Classical: expected queries = N/2 = 4 for random search
    let classical_expected = (1usize << n_qubits) / 2;

    println!("=== Grover vs Classical Comparison (3 qubits) ===");
    println!("Search space: {} states", 1 << n_qubits);
    println!("Grover iterations: {}", grover_iters);
    println!("Grover total gates: {}", grover_gates);
    println!("Classical expected queries: {}", classical_expected);
    println!(
        "Speedup factor: {:.1}x (in iterations)",
        classical_expected as f64 / grover_iters as f64
    );

    // Verify Grover is indeed faster (in oracle queries)
    assert!(
        grover_iters < classical_expected,
        "Grover should use fewer iterations than classical"
    );
}

/// Test multiple QDemo tiles running Grover independently.
#[test]
fn test_multiple_grover_tiles() {
    let mut sim = Simulation::new();

    // Register 4 QDemo tiles, each searching for a different state
    let targets = [0b000, 0b011, 0b101, 0b111];
    let positions = [(10, 10), (20, 10), (10, 20), (20, 20)];

    for (i, (&target, &(x, y))) in targets.iter().zip(positions.iter()).enumerate() {
        let program = grover_circuit(3, target, 2);
        let seed = (i as u64 + 1) * 0x1111111111;
        sim.register_qdemo_tile(x, y, QState::new_zero(3), program, seed);
    }

    // Run enough ticks for all circuits to complete
    // Each 3-qubit Grover with 2 iterations has roughly:
    // 3 H + 2*(oracle + diffusion) + 3 Measure gates
    let max_gates = 50; // Conservative upper bound
    for _ in 0..max_gates {
        sim.tick();
    }

    // Check each tile has a measurement result
    for (i, &(x, y)) in positions.iter().enumerate() {
        let result = sim.tilemap.value_at(x, y).unwrap();

        assert!(
            result <= 0b111,
            "Tile {} at ({},{}) has invalid result: {}",
            i,
            x,
            y,
            result
        );

        println!(
            "Tile {} searching for {:03b}: found {:03b}",
            i, targets[i], result
        );
    }
}
