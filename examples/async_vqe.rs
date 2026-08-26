// =============================================================================
// Sprint 62.0: Async VQE Demo
// =============================================================================
//
// Demonstrates non-blocking execution for VQE workloads using the hybrid
// orchestration layer.
//
// This example shows:
// - JobHandle for non-blocking job submission
// - Parallel parameter evaluation using batch execution
// - Circuit batching for efficient parameter sweeps
// - Basic VQE optimization loop
//
// Run with:
//   cargo run --example async_vqe --release
//
// =============================================================================

use std::collections::HashMap;
use std::time::Instant;

use engine::hybrid::{
    CircuitBatch, HybridScheduler, JobHandle, JobSpec, MeasurementSpec, QuantumCircuit,
    QuantumSpec, QuantumSubstrate, StageOutputs, execute_batch, linspace,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 62.0: ASYNC VQE DEMO");
    println!("        Non-Blocking Quantum Optimization");
    println!("{}\n", "=".repeat(80));

    // Demo 1: Non-blocking job submission
    println!("\n{}", "-".repeat(80));
    println!("DEMO 1: Non-Blocking Job Submission");
    println!("{}\n", "-".repeat(80));
    demo_nonblocking_jobs();

    // Demo 2: Parallel job execution
    println!("\n{}", "-".repeat(80));
    println!("DEMO 2: Parallel Job Execution");
    println!("{}\n", "-".repeat(80));
    demo_parallel_jobs();

    // Demo 3: Circuit batching for VQE
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: Circuit Batching for Parameter Sweep");
    println!("{}\n", "-".repeat(80));
    demo_circuit_batching();

    // Demo 4: Simple VQE optimization
    println!("\n{}", "-".repeat(80));
    println!("DEMO 4: Simple VQE Optimization");
    println!("{}\n", "-".repeat(80));
    demo_vqe_optimization();

    println!("\n{}", "=".repeat(80));
    println!("                    ASYNC VQE DEMO COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Demo 1: Non-blocking job submission.
fn demo_nonblocking_jobs() {
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 20)));

    println!("Submitting 3 non-blocking jobs...\n");

    // Create circuits
    let circuits: Vec<QuantumCircuit> = (2..=4)
        .map(|n| {
            let mut c = QuantumCircuit::new(n);
            c.h(0);
            for q in 0..n - 1 {
                c.cnot(q, q + 1);
            }
            c
        })
        .collect();

    // Submit jobs non-blocking
    let handles: Vec<JobHandle> = circuits
        .iter()
        .enumerate()
        .map(|(i, circuit)| {
            let spec = QuantumSpec::new(circuit.n_qubits, circuit.clone())
                .with_shots(1000)
                .with_seed(42 + i as u64);
            scheduler.submit_nonblocking(JobSpec::Quantum(spec))
        })
        .collect();

    println!(
        "Jobs submitted with IDs: {:?}",
        handles.iter().map(|h| h.id()).collect::<Vec<_>>()
    );
    println!("\nWaiting for results...\n");

    // Collect results
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.wait();
        if let Some(res) = result {
            let state = if res.is_success() {
                "Success"
            } else {
                "Failed"
            };
            println!("Job {} ({}): {}", i + 1, i + 2, state);

            if let Some(stage) = res.stage_results.first()
                && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
            {
                let total: usize = counts.values().sum();
                println!("  Total measurements: {}", total);
                println!("  Unique outcomes: {}", counts.len());
            }
        }
    }
}

/// Demo 2: Parallel job execution.
fn demo_parallel_jobs() {
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 20)));

    println!("Submitting batch of 5 jobs in parallel...\n");

    // Create 5 different circuits with different rotation angles
    let angles = [0.0, 0.5, 1.0, 1.5, 2.0];
    let specs: Vec<JobSpec> = angles
        .iter()
        .map(|&theta| {
            let mut circuit = QuantumCircuit::new(3);
            circuit.h(0);
            circuit.ry(0, theta);
            circuit.cnot(0, 1);
            circuit.ry(1, theta / 2.0);
            circuit.cnot(1, 2);
            JobSpec::Quantum(
                QuantumSpec::new(3, circuit)
                    .with_shots(500)
                    .with_measurement(MeasurementSpec::Computational),
            )
        })
        .collect();

    let start = Instant::now();

    // Submit all jobs
    let handles = scheduler.submit_batch_nonblocking(specs);
    println!("Submitted {} jobs", handles.len());

    // Collect results as they complete
    let mut results = Vec::new();
    for handle in handles {
        if let Some(result) = handle.wait() {
            results.push(result);
        }
    }

    let elapsed = start.elapsed();
    println!("\nAll jobs completed in {:?}", elapsed);
    println!(
        "Successful: {}/{}",
        results.iter().filter(|r| r.is_success()).count(),
        results.len()
    );

    // Show outcome distribution for each angle
    println!("\nOutcome distributions by rotation angle:\n");
    for (angle, result) in angles.iter().zip(results.iter()) {
        print!("  θ = {:.1}: ", angle);
        if let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            let mut sorted: Vec<_> = counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let top: Vec<_> = sorted.iter().take(2).collect();
            for (bitstring, count) in top {
                print!("|{}⟩:{} ", bitstring, count);
            }
        }
        println!();
    }
}

/// Demo 3: Circuit batching for parameter sweep.
fn demo_circuit_batching() {
    // Create a hardware-efficient ansatz
    let ansatz = QuantumCircuit::hardware_efficient(2, 1);
    println!("Ansatz: {} parameters\n", ansatz.parameter_count());

    // Create parameter grid for sweep
    let param_count = ansatz.parameter_count();
    let values = linspace(0.0, std::f64::consts::PI, 4);

    // For 2 parameters, create a 2D grid
    let mut parameter_sets = Vec::new();
    for &v0 in &values {
        for &v1 in &values {
            // Pad to full parameter count
            let mut params = vec![v0, v1];
            while params.len() < param_count {
                params.push(0.0);
            }
            parameter_sets.push(params);
        }
    }

    println!("Created {} parameter sets", parameter_sets.len());

    // Create batch
    let batch = CircuitBatch::new(ansatz)
        .with_parameter_sets(parameter_sets)
        .with_shots(500)
        .with_seed(42);

    // Execute batch on quantum substrate
    let mut substrate = QuantumSubstrate::new(1, "Quantum", 20);

    let start = Instant::now();
    let result = execute_batch(&mut substrate, &batch).expect("Batch execution failed");
    let elapsed = start.elapsed();

    println!("\nBatch execution completed:");
    println!("  Total time: {:?}", elapsed);
    println!("  Circuits executed: {}", result.circuit_count);
    println!(
        "  Success rate: {}%",
        result.success_count() * 100 / result.circuit_count
    );
    println!("  Avg time per circuit: {:?}", result.avg_duration());

    // Find best result (lowest expected value of Z)
    let mut best_idx = 0;
    let mut best_expectation = 1.0;

    for (i, entry) in result.iter().enumerate() {
        if let StageOutputs::MeasurementCounts(counts) = &entry.output {
            let exp = compute_z_expectation(counts);
            if exp < best_expectation {
                best_expectation = exp;
                best_idx = i;
            }
        }
    }

    if let Some(entry) = result.get(best_idx) {
        println!("\nBest parameters found:");
        println!(
            "  Params: {:?}",
            entry.parameters.iter().take(2).collect::<Vec<_>>()
        );
        println!("  ⟨Z⟩ = {:.4}", best_expectation);
    }
}

/// Demo 4: Simple VQE optimization.
fn demo_vqe_optimization() {
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 20)));

    // Simple 2-qubit ansatz
    let n_qubits = 2;
    let n_params = 4;

    println!("VQE Configuration:");
    println!("  Qubits: {}", n_qubits);
    println!("  Parameters: {}", n_params);
    println!("  Shots per evaluation: 500");
    println!("\nRunning optimization...\n");

    // Initial parameters
    let mut params = vec![0.1; n_params];
    let mut best_cost = f64::MAX;
    let mut best_params = params.clone();
    let n_iterations = 20;

    let start = Instant::now();

    for iter in 0..n_iterations {
        // Create ansatz with current parameters
        let circuit = create_simple_ansatz(n_qubits, &params);

        let spec = QuantumSpec::new(n_qubits, circuit)
            .with_shots(500)
            .with_seed(iter as u64);

        // Evaluate cost (submit non-blocking)
        let handle = scheduler.submit_nonblocking(JobSpec::Quantum(spec));

        // Wait for result
        if let Some(result) = handle.wait()
            && let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            // Compute simple cost: expectation of Z on first qubit
            let cost = compute_z_expectation(counts);

            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            if iter % 5 == 0 || iter == n_iterations - 1 {
                println!(
                    "  Iteration {:>2}: cost = {:>7.4}, best = {:>7.4}",
                    iter + 1,
                    cost,
                    best_cost
                );
            }

            // Simple gradient-free parameter update
            for (i, p) in params.iter_mut().enumerate() {
                *p += 0.1 * (0.5 - simple_random(iter as u64 + i as u64));
            }
        }
    }

    let elapsed = start.elapsed();

    println!("\nOptimization Results:");
    println!("  Iterations: {}", n_iterations);
    println!("  Best cost: {:.4}", best_cost);
    println!("  Best params: {:?}", best_params);
    println!("  Total time: {:?}", elapsed);
    println!("  Time per iteration: {:?}", elapsed / n_iterations as u32);
}

/// Create a simple parameterized ansatz.
fn create_simple_ansatz(n_qubits: usize, params: &[f64]) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);

    // Layer 1: Single-qubit rotations
    for (q, &theta) in params.iter().take(n_qubits).enumerate() {
        circuit.ry(q, theta);
    }

    // Entangling layer
    for q in 0..n_qubits.saturating_sub(1) {
        circuit.cnot(q, q + 1);
    }

    // Layer 2: More rotations
    for (q, &theta) in params.iter().skip(n_qubits).take(n_qubits).enumerate() {
        circuit.ry(q, theta);
    }

    circuit
}

/// Compute Z expectation value from measurement counts.
fn compute_z_expectation(counts: &HashMap<String, usize>) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut expectation = 0.0;
    for (bitstring, &count) in counts {
        // Eigenvalue of Z on first qubit
        let first_bit = bitstring.chars().last().unwrap_or('0');
        let eigenvalue = if first_bit == '0' { 1.0 } else { -1.0 };
        expectation += eigenvalue * (count as f64 / total as f64);
    }

    expectation
}

/// Simple pseudo-random number generator.
fn simple_random(seed: u64) -> f64 {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    (x as f64) / (u64::MAX as f64)
}
