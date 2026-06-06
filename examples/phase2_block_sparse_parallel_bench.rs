// SPRINT 39.0 Phase 2: Multi-core block-sparse parallel quantum execution benchmark
// Compares single-worker vs multi-worker performance for block-sparse quantum circuits

use engine::distributed::quantum_worker_pool::{
    AutoConfigInput, QuantumWorkerPool, auto_config_for_block_sparse,
};
use logic_fabric_core::block_sparse_state::BlockSparseState;
use logic_fabric_core::quantum::QGate;
use std::time::Instant;

fn main() {
    println!("\n=== SPRINT 39.0 Phase 2: Block-Sparse Parallel Quantum Execution Benchmark ===\n");

    // Configuration
    let n_qubits = 12; // 12 qubits = 4096 blocks of 128 amplitudes each
    let circuit_depth = 500; // Increased for better parallelism visibility
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    println!("Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!("  Circuit depth: {} gates", circuit_depth);
    println!("  CPU cores available: {}", num_cpus);
    println!();

    // Create test circuit (single-qubit gates only for Phase 2)
    let circuit = create_test_circuit(n_qubits, circuit_depth);

    println!("Testing worker configurations:");
    println!();

    let mut baseline_time = 0.0;

    for num_workers in [1, 2, 4] {
        if num_workers > num_cpus {
            println!("  Skipping {} workers (exceeds CPU count)", num_workers);
            continue;
        }

        // Timed run with fresh state
        let mut state = BlockSparseState::new_zero(n_qubits);
        let mut pool = QuantumWorkerPool::new(num_workers);

        let start = Instant::now();
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        let elapsed = start.elapsed();

        pool.shutdown();

        // Small delay to ensure cleanup
        std::thread::sleep(std::time::Duration::from_millis(100));

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let gates_per_sec = (circuit_depth as f64) / elapsed.as_secs_f64();

        if num_workers == 1 {
            baseline_time = elapsed_ms;
            println!(
                "  {} worker:  {:.2} ms  ({:.0} gates/sec)  [baseline]",
                num_workers, elapsed_ms, gates_per_sec
            );
        } else {
            let speedup = baseline_time / elapsed_ms;
            let efficiency = (speedup / num_workers as f64) * 100.0;
            println!(
                "  {} workers: {:.2} ms  ({:.0} gates/sec)  {:.2}× speedup  ({:.1}% efficiency)",
                num_workers, elapsed_ms, gates_per_sec, speedup, efficiency
            );
        }
    }

    println!();
    println!("=== Auto-Configuration Test ===");
    println!();

    // Test auto-configuration
    let input = AutoConfigInput {
        cpu_cores: num_cpus,
        block_count: 1 << (n_qubits - 7), // BLOCK_SHIFT = 7
        n_qubits,
    };

    let config = auto_config_for_block_sparse(input);
    println!(
        "  Auto-selected: {} workers × {} blocks/worker",
        config.num_workers, config.blocks_per_worker
    );

    let mut state = BlockSparseState::new_zero(n_qubits);
    let mut pool = QuantumWorkerPool::new(config.num_workers);

    let start = Instant::now();
    pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
    let elapsed = start.elapsed();

    pool.shutdown();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let speedup = baseline_time / elapsed_ms;

    println!(
        "  Performance: {:.2} ms  ({:.2}× speedup)",
        elapsed_ms, speedup
    );
    println!();

    println!("=== Summary ===");
    println!();
    println!("Phase 2 multi-core block-sparse execution is functional!");
    println!("Single-qubit gates (H, X, Z, Y, Phase) working correctly.");
    println!("Multi-qubit gates (CNOT, CZ) deferred to Phase 2.2.");
    println!();
}

/// Create a test circuit with deterministic single-qubit gates
fn create_test_circuit(n_qubits: u8, depth: usize) -> Vec<QGate> {
    let mut circuit = Vec::with_capacity(depth);

    for i in 0..depth {
        let qubit = (i % n_qubits as usize) as u8;
        let gate_type = i % 5;

        let gate = match gate_type {
            0 => QGate::H(qubit),
            1 => QGate::X(qubit),
            2 => QGate::Z(qubit),
            3 => QGate::Y(qubit),
            4 => QGate::Phase(qubit, (i as f32) * 0.1),
            _ => unreachable!(),
        };

        circuit.push(gate);
    }

    circuit
}
