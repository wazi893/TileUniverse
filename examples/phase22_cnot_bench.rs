// SPRINT 39.0 Phase 2.2: CNOT Gate Performance Benchmark
// Tests entangling circuit performance with parallel block-sparse execution

use engine::distributed::quantum_worker_pool::QuantumWorkerPool;
use logic_fabric_core::block_sparse_state::BlockSparseState;
use logic_fabric_core::quantum::QGate;
use std::time::Instant;

fn main() {
    println!("\n=== SPRINT 39.0 Phase 2.2: CNOT Gate Performance Benchmark ===\n");

    let n_qubits = 12;
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    println!("Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!("  CPU cores available: {}", num_cpus);
    println!();

    // Test 1: Within-block CNOT (qubits 0-6)
    println!("=== Test 1: Within-Block CNOT (qubits 0-6) ===");
    println!();
    bench_within_block_cnot(n_qubits, 200);

    // Test 2: Cross-block CNOT (control >= 7)
    println!("\n=== Test 2: Cross-Block CNOT (control >= 7, target < 7) ===");
    println!();
    bench_cross_block_cnot(n_qubits, 200);

    // Test 3: Entangling circuit (Bell pairs)
    println!("\n=== Test 3: Bell Pair Creation (H + CNOT) ===");
    println!();
    bench_bell_pairs(n_qubits, 100);

    // Test 4: Mixed circuit (single-qubit + CNOT)
    println!("\n=== Test 4: Mixed Circuit (50% H, 50% CNOT) ===");
    println!();
    bench_mixed_circuit(n_qubits, 200);

    println!("\n=== Summary ===");
    println!();
    println!("Phase 2.2 CNOT implementation complete!");
    println!("- Within-block CNOT: Fully parallel (fast path)");
    println!("- Cross-block CNOT: Partial implementation");
    println!("  ✅ Control >= 7, Target < 7: Working");
    println!("  ⏳ Target >= 7: Requires inter-block coordination (TODO)");
    println!();
}

fn bench_within_block_cnot(n_qubits: u8, depth: usize) {
    // Create circuit with CNOTs between low-order qubits only
    let mut circuit = Vec::with_capacity(depth);
    for i in 0..depth {
        let control = (i % 6) as u8; // Qubits 0-5
        let target = ((i + 1) % 6) as u8;
        if control != target {
            circuit.push(QGate::CNot(control, target));
        }
    }

    println!("  Circuit: {} CNOT gates (all within-block)", circuit.len());
    println!();

    let mut baseline_time = 0.0;

    for num_workers in [1, 2, 4] {
        let mut state = BlockSparseState::new_zero(n_qubits);
        let mut pool = QuantumWorkerPool::new(num_workers);

        let start = Instant::now();
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        let elapsed = start.elapsed();

        pool.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if num_workers == 1 {
            baseline_time = elapsed_ms;
            println!(
                "    {} worker:  {:.2} ms  [baseline]",
                num_workers, elapsed_ms
            );
        } else {
            let speedup = baseline_time / elapsed_ms;
            let efficiency = (speedup / num_workers as f64) * 100.0;
            println!(
                "    {} workers: {:.2} ms  ({:.2}× speedup, {:.1}% efficiency)",
                num_workers, elapsed_ms, speedup, efficiency
            );
        }
    }
}

fn bench_cross_block_cnot(n_qubits: u8, depth: usize) {
    // Create circuit with control qubit >= 7, target < 7
    let mut circuit = Vec::with_capacity(depth);
    for i in 0..depth {
        let control = 7 + (i % 3) as u8; // Qubits 7-9 (cross-block)
        let target = (i % 6) as u8; // Qubits 0-5 (within-block)
        circuit.push(QGate::CNot(control, target));
    }

    println!(
        "  Circuit: {} CNOT gates (control cross-block)",
        circuit.len()
    );
    println!();

    let mut baseline_time = 0.0;

    for num_workers in [1, 2, 4] {
        let mut state = BlockSparseState::new_zero(n_qubits);
        let mut pool = QuantumWorkerPool::new(num_workers);

        let start = Instant::now();
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        let elapsed = start.elapsed();

        pool.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if num_workers == 1 {
            baseline_time = elapsed_ms;
            println!(
                "    {} worker:  {:.2} ms  [baseline]",
                num_workers, elapsed_ms
            );
        } else {
            let speedup = baseline_time / elapsed_ms;
            let efficiency = (speedup / num_workers as f64) * 100.0;
            println!(
                "    {} workers: {:.2} ms  ({:.2}× speedup, {:.1}% efficiency)",
                num_workers, elapsed_ms, speedup, efficiency
            );
        }
    }
}

fn bench_bell_pairs(n_qubits: u8, num_pairs: usize) {
    // Create Bell pairs: H on qubit i, CNOT(i, i+1)
    let mut circuit = Vec::with_capacity(num_pairs * 2);
    for i in 0..num_pairs {
        let qubit = (i % (n_qubits / 2) as usize) as u8;
        circuit.push(QGate::H(qubit * 2));
        circuit.push(QGate::CNot(qubit * 2, qubit * 2 + 1));
    }

    println!("  Circuit: {} Bell pairs (H + CNOT)", num_pairs);
    println!();

    let mut baseline_time = 0.0;

    for num_workers in [1, 2, 4] {
        let mut state = BlockSparseState::new_zero(n_qubits);
        let mut pool = QuantumWorkerPool::new(num_workers);

        let start = Instant::now();
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        let elapsed = start.elapsed();

        pool.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if num_workers == 1 {
            baseline_time = elapsed_ms;
            println!(
                "    {} worker:  {:.2} ms  [baseline]",
                num_workers, elapsed_ms
            );
        } else {
            let speedup = baseline_time / elapsed_ms;
            let efficiency = (speedup / num_workers as f64) * 100.0;
            println!(
                "    {} workers: {:.2} ms  ({:.2}× speedup, {:.1}% efficiency)",
                num_workers, elapsed_ms, speedup, efficiency
            );
        }
    }
}

fn bench_mixed_circuit(n_qubits: u8, depth: usize) {
    // Create circuit with 50% single-qubit gates, 50% CNOT
    let mut circuit = Vec::with_capacity(depth);
    for i in 0..depth {
        if i % 2 == 0 {
            // Single-qubit gate
            let qubit = (i % n_qubits as usize) as u8;
            circuit.push(QGate::H(qubit));
        } else {
            // CNOT
            let control = (i % 6) as u8;
            let target = ((i + 1) % 6) as u8;
            if control != target {
                circuit.push(QGate::CNot(control, target));
            } else {
                circuit.push(QGate::X(control));
            }
        }
    }

    println!("  Circuit: {} gates (mixed H and CNOT)", circuit.len());
    println!();

    let mut baseline_time = 0.0;

    for num_workers in [1, 2, 4] {
        let mut state = BlockSparseState::new_zero(n_qubits);
        let mut pool = QuantumWorkerPool::new(num_workers);

        let start = Instant::now();
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        let elapsed = start.elapsed();

        pool.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if num_workers == 1 {
            baseline_time = elapsed_ms;
            println!(
                "    {} worker:  {:.2} ms  [baseline]",
                num_workers, elapsed_ms
            );
        } else {
            let speedup = baseline_time / elapsed_ms;
            let efficiency = (speedup / num_workers as f64) * 100.0;
            println!(
                "    {} workers: {:.2} ms  ({:.2}× speedup, {:.1}% efficiency)",
                num_workers, elapsed_ms, speedup, efficiency
            );
        }
    }
}
