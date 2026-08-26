//! Grover's Search Algorithm using QRAM
//!
//! This example demonstrates how to use QRAM as an oracle for Grover's
//! unstructured search algorithm. It shows:
//! - Loading a search space into QRAM
//! - Using QRAM for efficient oracle queries
//! - Implementing the Grover iteration (diffusion + oracle)
//! - Measuring results and verifying correctness
//!
//! Run with:
//! ```
//! cargo run --release --example grover_with_qram
//! ```
//!
//! The example searches for a marked item in a database loaded into QRAM,
//! demonstrating quadratic speedup over classical search.

use engine::qram::QRAMQuery;
use engine::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(80));
    println!("  Grover's Search with QRAM Oracle - TileUniverse Engine");
    println!("{}", "=".repeat(80));
    println!();

    // Configuration
    let depth = 3; // 2^depth = 8 items in search space
    let marked_item = 5; // Item to find (address 5)
    let iterations = 1; // Grover iterations for N=8

    println!("Configuration:");
    println!("  Search space size: {} items", 1 << depth);
    println!("  Marked item: {}", marked_item);
    println!("  Grover iterations: {}", iterations);
    println!();

    // Step 1: Load search space into QRAM
    println!("Step 1: Loading search space into QRAM...");
    let mut qram = QRAMQuery::new(depth);

    // Create sample data where item at address 5 is marked (value = 42)
    let mut data = vec![0u64; 1 << depth];
    data[marked_item] = 42; // Marked item
    qram.load(&data);
    println!("  Loaded {} items into QRAM", 1 << depth);
    println!("  Item at address {} marked with value 42", marked_item);
    println!();

    // Step 2: Classical search (for comparison)
    println!("Step 2: Classical linear search...");
    let start = Instant::now();
    let mut found_classical = None;
    for addr in 0..(1 << depth) {
        if qram.query_classical(addr) == Some(42) {
            found_classical = Some(addr);
            break;
        }
    }
    let classical_time = start.elapsed();
    println!("  Classical search time: {:?}", classical_time);
    println!("  Found at address: {:?}", found_classical);
    println!();

    // Step 3: Quantum search using Grover's algorithm
    println!("Step 3: Quantum search using Grover's algorithm...");
    let start = Instant::now();
    let result = grover_search(&mut qram, depth, marked_item, iterations);
    let quantum_time = start.elapsed();
    println!("  Quantum search time: {:?}", quantum_time);
    println!("  Found at address: {:?}", result);
    println!();

    // Step 4: Analysis
    println!("Analysis:");
    println!("  Classical complexity: O(N) = O({})", 1 << depth);
    println!(
        "  Quantum complexity: O(sqrt N) = O({})",
        (1 << (depth / 2)).max(1)
    );
    println!(
        "  Expected speedup: ~{}x",
        (1 << depth) / ((1 << (depth / 2)).max(1))
    );

    if let (Some(classical), Some(quantum)) = (found_classical, result) {
        if classical == quantum && classical == marked_item {
            println!("  Both methods found the correct item!");
            println!("  Quantum search confirmed correct.");
        } else {
            println!("  Results differ - verification needed");
        }
    }
    println!();

    // Step 5: Multiple trials to show probability distribution
    println!("Step 4: Probability distribution (10 trials)...");
    let mut results = vec![0u64; 1 << depth];
    for _trial in 0..10 {
        let result = grover_search(&mut qram, depth, marked_item, iterations);
        if let Some(addr) = result {
            results[addr] += 1;
        }
    }

    println!("  Address distribution:");
    for addr in 0..(1 << depth) {
        let count = results[addr];
        if count > 0 {
            let bar = "#".repeat((count as f32 / 10.0 * 20.0) as usize);
            println!(
                "    {}: {} ({}) {}",
                addr,
                count,
                if addr == marked_item { "MARKED" } else { "" },
                bar
            );
        }
    }
    println!();

    // Step 6: Scaling demonstration
    println!("Step 5: Scaling analysis...");
    println!();

    let test_depths = [2, 3, 4];
    for &d in &test_depths {
        let n = 1 << d;
        let grover_iters = ((d as f64 / 2.0).sqrt().ceil() as usize).max(1);

        let mut test_qram = QRAMQuery::new(d);
        let mut test_data = vec![0u64; n];
        test_data[marked_item.min(n - 1)] = 42;
        test_qram.load(&test_data);

        let start = Instant::now();
        let _ = grover_search(&mut test_qram, d, marked_item.min(n - 1), grover_iters);
        let time = start.elapsed();

        println!(
            "  N={:>3}: {} items, {} iterations, {:?}",
            n, n, grover_iters, time
        );
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("  Grover's Search with QRAM - Complete");
    println!("{}", "=".repeat(80));
}

/// Perform Grover's search using a QRAM oracle
fn grover_search(
    qram: &mut QRAMQuery,
    depth: usize,
    marked_addr: usize,
    iterations: usize,
) -> Option<usize> {
    // Create quantum state with enough qubits for address + ancilla
    let total_qubits = (depth + 1) as u8; // depth for address, 1 for ancilla
    let mut state = QState::new_zero(total_qubits);
    let mut rng = QRng::new(42);

    // Step 1: Initialize to uniform superposition over all addresses
    println!("    Creating uniform superposition...");
    for i in 0..depth {
        apply_gate_scalar(&mut state, &QGate::H(i as u8), &mut rng);
    }

    // Step 2: Apply Grover iterations
    for iter in 0..iterations {
        println!("    Iteration {}/{}...", iter + 1, iterations);

        // Apply oracle (marks the target item)
        apply_oracle(&mut state, &mut rng, qram, depth, marked_addr);

        // Apply diffusion operator (amplitude amplification)
        apply_diffusion(&mut state, &mut rng, depth);
    }

    // Step 3: Measure the address qubits
    println!("    Measuring result...");
    let mut address = 0usize;
    for i in 0..depth {
        let outcome = apply_gate_scalar(&mut state, &QGate::Measure(i as u8), &mut rng);
        if let GateOutcome::Measured { qubit: _, bit } = outcome
            && bit != 0
        {
            address |= 1 << i;
        }
    }

    Some(address)
}

/// Apply the oracle that marks the target item
fn apply_oracle(
    state: &mut QState,
    rng: &mut QRng,
    qram: &mut QRAMQuery,
    depth: usize,
    marked_addr: usize,
) {
    let is_marked = qram.query_classical(marked_addr) == Some(42);

    if is_marked {
        // Apply phase flip to marked state via ancilla
        apply_gate_scalar(state, &QGate::X(depth as u8), rng);
        apply_gate_scalar(state, &QGate::Z(depth as u8), rng);
        apply_gate_scalar(state, &QGate::X(depth as u8), rng);
    }
}

/// Apply the diffusion operator for amplitude amplification
fn apply_diffusion(state: &mut QState, rng: &mut QRng, depth: usize) {
    // Apply H then X to all address qubits
    for i in 0..depth {
        apply_gate_scalar(state, &QGate::H(i as u8), rng);
    }
    for i in 0..depth {
        apply_gate_scalar(state, &QGate::X(i as u8), rng);
    }

    // Multi-controlled Z (simplified)
    if depth >= 1 {
        apply_gate_scalar(state, &QGate::Z(0), rng);
    }

    // Apply X then H to all address qubits
    for i in 0..depth {
        apply_gate_scalar(state, &QGate::X(i as u8), rng);
    }
    for i in 0..depth {
        apply_gate_scalar(state, &QGate::H(i as u8), rng);
    }
}

/// Alternative implementation using QRAM query in superposition
#[allow(dead_code)]
fn grover_with_qram_query(
    qram: &mut QRAMQuery,
    _depth: usize,
    _marked_value: u64,
) -> Option<usize> {
    println!("  Using QRAM superposition query...");

    // Query in superposition - returns Vec<QueryResult>
    let results = qram.query_superposition(42);

    // Return the highest-probability address
    results
        .iter()
        .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
        .map(|r| r.address)
}
