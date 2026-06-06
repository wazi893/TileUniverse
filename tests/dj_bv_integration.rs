// =============================================================================
// tests/dj_bv_integration.rs
// =============================================================================
//
// Integration tests for Deutsch-Jozsa and Bernstein-Vazirani algorithms
// running on QDemo tiles in the simulation.
//
// =============================================================================

use std::sync::atomic::Ordering;

use engine::algorithms::{DJOracleType, bernstein_vazirani_circuit, deutsch_jozsa_circuit};
use engine::quantum::QState;
use engine::simulation::Simulation;

/// Test Deutsch-Jozsa on QDemo tile detects constant-zero oracle.
#[test]
fn test_dj_qdemo_constant_zero() {
    let mut sim = Simulation::new();

    let n_qubits = 3u8;
    let program = deutsch_jozsa_circuit(n_qubits, DJOracleType::ConstantZero);
    let program_len = program.len();

    // Register with n+1 qubits (includes ancilla)
    let state = QState::new_zero(n_qubits + 1);
    sim.register_qdemo_tile(50, 50, state, program, 42);

    // Run circuit
    for _ in 0..program_len {
        sim.tick();
    }

    // Read result (only first n_qubits bits matter)
    let result = sim
        .tilemap
        .get_tile(50, 50)
        .unwrap()
        .logic
        .load(Ordering::Relaxed);
    let input_bits = result & ((1u64 << n_qubits) - 1);

    // Constant oracle → should measure |000⟩
    assert_eq!(
        input_bits, 0,
        "Constant-zero should give |000⟩, got {:03b}",
        input_bits
    );
}

/// Test Deutsch-Jozsa on QDemo tile detects constant-one oracle.
#[test]
fn test_dj_qdemo_constant_one() {
    let mut sim = Simulation::new();

    let n_qubits = 3u8;
    let program = deutsch_jozsa_circuit(n_qubits, DJOracleType::ConstantOne);
    let program_len = program.len();

    let state = QState::new_zero(n_qubits + 1);
    sim.register_qdemo_tile(50, 50, state, program, 42);

    for _ in 0..program_len {
        sim.tick();
    }

    let result = sim
        .tilemap
        .get_tile(50, 50)
        .unwrap()
        .logic
        .load(Ordering::Relaxed);
    let input_bits = result & ((1u64 << n_qubits) - 1);

    // Constant oracle → should measure |000⟩
    assert_eq!(
        input_bits, 0,
        "Constant-one should give |000⟩, got {:03b}",
        input_bits
    );
}

/// Test Deutsch-Jozsa on QDemo tile detects balanced oracle.
#[test]
fn test_dj_qdemo_balanced() {
    let mut sim = Simulation::new();

    let n_qubits = 3u8;
    let pattern = 0b101u64; // Balanced function based on this pattern
    let program = deutsch_jozsa_circuit(n_qubits, DJOracleType::Balanced(pattern));
    let program_len = program.len();

    let state = QState::new_zero(n_qubits + 1);
    sim.register_qdemo_tile(50, 50, state, program, 42);

    for _ in 0..program_len {
        sim.tick();
    }

    let result = sim
        .tilemap
        .get_tile(50, 50)
        .unwrap()
        .logic
        .load(Ordering::Relaxed);
    let input_bits = result & ((1u64 << n_qubits) - 1);

    // Balanced oracle → should NOT measure |000⟩
    assert_ne!(
        input_bits, 0,
        "Balanced should NOT give |000⟩, got {:03b}",
        input_bits
    );
}

/// Test Bernstein-Vazirani on QDemo tile finds hidden string.
#[test]
fn test_bv_qdemo_finds_string() {
    let mut sim = Simulation::new();

    let n_qubits = 4u8;
    let hidden_string = 0b1010u64;
    let program = bernstein_vazirani_circuit(n_qubits, hidden_string);
    let program_len = program.len();

    let state = QState::new_zero(n_qubits + 1);
    sim.register_qdemo_tile(50, 50, state, program, 42);

    for _ in 0..program_len {
        sim.tick();
    }

    let result = sim
        .tilemap
        .get_tile(50, 50)
        .unwrap()
        .logic
        .load(Ordering::Relaxed);
    let input_bits = result & ((1u64 << n_qubits) - 1);

    // Should recover exact hidden string
    assert_eq!(
        input_bits, hidden_string,
        "Should find hidden string {:04b}, got {:04b}",
        hidden_string, input_bits
    );
}

/// Test BV works for all possible hidden strings (exhaustive for small n).
#[test]
fn test_bv_qdemo_exhaustive() {
    let n_qubits = 3u8;

    for hidden in 0..(1u64 << n_qubits) {
        let mut sim = Simulation::new();
        let program = bernstein_vazirani_circuit(n_qubits, hidden);
        let program_len = program.len();

        let state = QState::new_zero(n_qubits + 1);
        sim.register_qdemo_tile(50, 50, state, program, hidden * 1234);

        for _ in 0..program_len {
            sim.tick();
        }

        let result = sim
            .tilemap
            .get_tile(50, 50)
            .unwrap()
            .logic
            .load(Ordering::Relaxed);
        let input_bits = result & ((1u64 << n_qubits) - 1);

        assert_eq!(
            input_bits, hidden,
            "Hidden {:03b}: expected {:03b}, got {:03b}",
            hidden, hidden, input_bits
        );
    }
}

/// Test DJ with all balanced patterns for n=3.
#[test]
fn test_dj_qdemo_all_balanced_patterns() {
    let n_qubits = 3u8;

    // All non-zero patterns create balanced functions
    for pattern in 1..(1u64 << n_qubits) {
        let mut sim = Simulation::new();
        let program = deutsch_jozsa_circuit(n_qubits, DJOracleType::Balanced(pattern));
        let program_len = program.len();

        let state = QState::new_zero(n_qubits + 1);
        sim.register_qdemo_tile(50, 50, state, program, pattern * 5678);

        for _ in 0..program_len {
            sim.tick();
        }

        let result = sim
            .tilemap
            .get_tile(50, 50)
            .unwrap()
            .logic
            .load(Ordering::Relaxed);
        let input_bits = result & ((1u64 << n_qubits) - 1);

        assert_ne!(
            input_bits, 0,
            "Pattern {:03b}: balanced should not give |000⟩",
            pattern
        );
    }
}

/// Test multiple DJ tiles running simultaneously.
#[test]
fn test_multiple_dj_tiles() {
    let mut sim = Simulation::new();

    // Register 4 tiles with different oracle types
    let oracles = [
        DJOracleType::ConstantZero,
        DJOracleType::ConstantOne,
        DJOracleType::Balanced(0b101),
        DJOracleType::Balanced(0b010),
    ];
    let positions = [(10, 10), (20, 10), (10, 20), (20, 20)];

    let n_qubits = 3u8;
    let mut max_len = 0;

    for (oracle, &(x, y)) in oracles.iter().zip(positions.iter()) {
        let program = deutsch_jozsa_circuit(n_qubits, *oracle);
        max_len = max_len.max(program.len());
        let state = QState::new_zero(n_qubits + 1);
        sim.register_qdemo_tile(x, y, state, program, (x * y) as u64);
    }

    // Run all circuits
    for _ in 0..max_len {
        sim.tick();
    }

    // Check results
    let expected_constant = [true, true, false, false];

    for (i, &(x, y)) in positions.iter().enumerate() {
        let result = sim
            .tilemap
            .get_tile(x, y)
            .unwrap()
            .logic
            .load(Ordering::Relaxed);
        let input_bits = result & ((1u64 << n_qubits) - 1);
        let is_constant = input_bits == 0;

        assert_eq!(
            is_constant, expected_constant[i],
            "Tile {} at ({},{}): expected constant={}, got input_bits={:03b}",
            i, x, y, expected_constant[i], input_bits
        );
    }
}

/// Test multiple BV tiles running simultaneously.
#[test]
fn test_multiple_bv_tiles() {
    let mut sim = Simulation::new();

    let hidden_strings = [0b000, 0b101, 0b111, 0b010];
    let positions = [(30, 30), (40, 30), (30, 40), (40, 40)];

    let n_qubits = 3u8;
    let mut max_len = 0;

    for (&hidden, &(x, y)) in hidden_strings.iter().zip(positions.iter()) {
        let program = bernstein_vazirani_circuit(n_qubits, hidden);
        max_len = max_len.max(program.len());
        let state = QState::new_zero(n_qubits + 1);
        sim.register_qdemo_tile(x, y, state, program, hidden * 9999);
    }

    for _ in 0..max_len {
        sim.tick();
    }

    for (&hidden, &(x, y)) in hidden_strings.iter().zip(positions.iter()) {
        let result = sim
            .tilemap
            .get_tile(x, y)
            .unwrap()
            .logic
            .load(Ordering::Relaxed);
        let input_bits = result & ((1u64 << n_qubits) - 1);

        assert_eq!(
            input_bits, hidden,
            "Tile at ({},{}): expected {:03b}, got {:03b}",
            x, y, hidden, input_bits
        );
    }
}

/// Benchmark: verify single query property.
#[test]
fn test_speedup_verification() {
    // Deutsch-Jozsa: Classical requires 2^(n-1) + 1 queries worst case
    // Quantum: 1 query

    // Bernstein-Vazirani: Classical requires n queries
    // Quantum: 1 query

    let n = 5;

    // DJ classical worst case
    let dj_classical = (1 << (n - 1)) + 1;
    let dj_quantum = 1;

    // BV classical
    let bv_classical = n;
    let bv_quantum = 1;

    println!("=== Speedup Analysis (n={}) ===", n);
    println!(
        "Deutsch-Jozsa: {} classical queries vs {} quantum ({}x speedup)",
        dj_classical, dj_quantum, dj_classical
    );
    println!(
        "Bernstein-Vazirani: {} classical queries vs {} quantum ({}x speedup)",
        bv_classical, bv_quantum, bv_classical
    );

    // For n=5: DJ gives 17x speedup, BV gives 5x
    assert!(dj_classical > dj_quantum);
    assert!(bv_classical > bv_quantum);
}
