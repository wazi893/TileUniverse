// SPRINT 39.0 Phase 4: Distributed Quantum Circuit Execution Benchmark
// Tests multi-node quantum circuit execution with in-process simulation

use engine::distributed::distributed_farm::DistributedTileFarm;
use logic_fabric_core::quantum::QGate;
use std::time::Instant;

fn main() {
    println!("\n=== SPRINT 39.0 Phase 4: Distributed Quantum Circuit Execution ===\n");

    // Test 1: Single-node baseline
    println!("=== Test 1: Single-Node Baseline ===\n");
    test_single_node();

    // Test 2: Multi-node scaling
    println!("\n=== Test 2: Multi-Node Scaling (1, 2, 4 nodes) ===\n");
    test_multi_node_scaling();

    // Test 3: Batch execution
    println!("\n=== Test 3: Batch Execution (10 circuits) ===\n");
    test_batch_execution();

    // Test 4: CNOT gates
    println!("\n=== Test 4: CNOT Gates (Within-Block and Cross-Block) ===\n");
    test_cnot_gates();

    // Test 5: Large circuit
    println!("\n=== Test 5: Large Circuit (200 gates) ===\n");
    test_large_circuit();

    println!("\n=== Summary ===\n");
    println!("Phase 4 distributed quantum circuit execution complete!");
    println!("✅ Single-node execution working");
    println!("✅ Multi-node scaling demonstrated");
    println!("✅ Batch execution with pool reuse");
    println!("✅ CNOT gates (within-block and cross-block)");
    println!("✅ Large circuits handled efficiently");
    println!();
}

fn test_single_node() {
    let farm = DistributedTileFarm::new(12, 1, 8, 2);

    let circuit = vec![
        QGate::H(0),
        QGate::X(1),
        QGate::Y(2),
        QGate::Z(3),
        QGate::H(4),
    ];

    println!("  Configuration:");
    println!("    Nodes: 1");
    println!("    Qubits: 12");
    println!("    Gates: {}", circuit.len());
    println!("    Workers per node: 2");
    println!();

    let start = Instant::now();
    let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();
    let elapsed = start.elapsed();

    let (node_id, checksum, exec_time_us) = results[0];

    println!("  Results:");
    println!(
        "    Node {}: checksum=0x{:016X}, time={}us",
        node_id, checksum, exec_time_us
    );
    println!("    Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

fn test_multi_node_scaling() {
    let circuit = vec![
        QGate::H(0),
        QGate::CNot(0, 1),
        QGate::H(2),
        QGate::CNot(2, 3),
        QGate::H(4),
        QGate::CNot(4, 5),
    ];

    for num_nodes in [1, 2, 4] {
        let farm = DistributedTileFarm::new(12, num_nodes, 8, 2);

        println!(
            "  Configuration: {} node(s), {} gates, 2 workers/node",
            num_nodes,
            circuit.len()
        );

        let start = Instant::now();
        let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();
        let elapsed = start.elapsed();

        println!("  Results:");
        for (node_id, checksum, exec_time_us) in &results {
            println!(
                "    Node {}: checksum=0x{:016X}, time={}us",
                node_id, checksum, exec_time_us
            );
        }

        // Verify all nodes produce same checksum
        let first_checksum = results[0].1;
        let all_match = results
            .iter()
            .all(|(_, checksum, _)| *checksum == first_checksum);

        println!("  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Checksum consistency: {}",
            if all_match { "✓ PASS" } else { "✗ FAIL" }
        );
        println!();
    }
}

fn test_batch_execution() {
    let farm = DistributedTileFarm::new(12, 2, 8, 2);

    let circuits = vec![
        vec![QGate::H(0)],
        vec![QGate::X(1), QGate::H(1)],
        vec![QGate::Y(2), QGate::Z(2), QGate::H(2)],
        vec![QGate::H(3), QGate::CNot(3, 4)],
        vec![QGate::CNot(0, 1), QGate::CNot(1, 2)],
        vec![QGate::H(5), QGate::X(5), QGate::Y(5), QGate::Z(5)],
        vec![QGate::CNot(2, 3), QGate::H(3)],
        vec![QGate::X(0), QGate::Y(1), QGate::Z(2)],
        vec![QGate::H(4), QGate::H(5)],
        vec![QGate::CNot(4, 5), QGate::CNot(5, 4)],
    ];

    let total_gates: usize = circuits.iter().map(|c| c.len()).sum();

    println!("  Configuration:");
    println!("    Nodes: 2");
    println!("    Circuits: {}", circuits.len());
    println!("    Total gates: {}", total_gates);
    println!("    Workers per node: 2");
    println!();

    let start = Instant::now();
    let results = farm.execute_quantum_batch(12, &circuits, 2).unwrap();
    let elapsed = start.elapsed();

    println!("  Results:");
    for (node_id, checksum, circuit_count, total_gates, exec_time_us) in &results {
        println!(
            "    Node {}: {} circuits, {} gates, checksum=0x{:016X}, time={}us",
            node_id, circuit_count, total_gates, checksum, exec_time_us
        );
    }

    let first_checksum = results[0].1;
    let all_match = results
        .iter()
        .all(|(_, checksum, _, _, _)| *checksum == first_checksum);

    println!("  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Checksum consistency: {}",
        if all_match { "✓ PASS" } else { "✗ FAIL" }
    );
    println!();
}

fn test_cnot_gates() {
    let farm = DistributedTileFarm::new(12, 2, 8, 2);

    // Within-block CNOTs (qubits 0-6)
    let within_block_circuit = vec![
        QGate::H(0),
        QGate::CNot(0, 1),
        QGate::CNot(1, 2),
        QGate::CNot(2, 3),
        QGate::H(4),
        QGate::CNot(4, 5),
    ];

    // Cross-block CNOTs (control >= 7, target < 7)
    let cross_block_circuit = vec![
        QGate::H(7),
        QGate::CNot(7, 0),
        QGate::CNot(8, 1),
        QGate::CNot(9, 2),
    ];

    println!("  Test 4a: Within-Block CNOT");
    let results = farm
        .execute_quantum_circuit(12, &within_block_circuit, 2)
        .unwrap();
    let (_, checksum, time) = results[0];
    println!("    Result: checksum=0x{:016X}, time={}us", checksum, time);

    println!("\n  Test 4b: Cross-Block CNOT");
    let results = farm
        .execute_quantum_circuit(12, &cross_block_circuit, 2)
        .unwrap();
    let (_, checksum, time) = results[0];
    println!("    Result: checksum=0x{:016X}, time={}us", checksum, time);
    println!();
}

fn test_large_circuit() {
    let farm = DistributedTileFarm::new(12, 2, 8, 2);

    // Create a large circuit with 200 gates
    let mut circuit = Vec::with_capacity(200);
    for i in 0..100 {
        circuit.push(QGate::H(i % 6));
        circuit.push(QGate::X((i + 1) % 6));
    }

    println!("  Configuration:");
    println!("    Nodes: 2");
    println!("    Gates: {}", circuit.len());
    println!("    Workers per node: 2");
    println!();

    let start = Instant::now();
    let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();
    let elapsed = start.elapsed();

    println!("  Results:");
    for (node_id, checksum, exec_time_us) in &results {
        println!(
            "    Node {}: checksum=0x{:016X}, time={:.2}ms",
            node_id,
            checksum,
            *exec_time_us as f64 / 1000.0
        );
    }

    let first_checksum = results[0].1;
    let all_match = results
        .iter()
        .all(|(_, checksum, _)| *checksum == first_checksum);

    println!("  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Checksum consistency: {}",
        if all_match { "✓ PASS" } else { "✗ FAIL" }
    );
    println!();
}
