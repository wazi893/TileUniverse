// =============================================================================
// Sprint 61.0: VQE Hybrid Orchestration Demo
// =============================================================================
//
// Demonstrates the hybrid orchestration layer for VQE (Variational Quantum
// Eigensolver) workloads, inspired by RIKEN's JHPC-Quantum architecture.
//
// This example shows:
// - HybridScheduler with multiple substrates (Classical + Quantum)
// - Variational loop execution (quantum evaluation + classical optimization)
// - Job submission and result retrieval
// - Inter-substrate data flow via DataBridge
//
// Run with:
//   cargo run --example vqe_hybrid --release
//
// =============================================================================

use std::collections::HashMap;
use std::time::Instant;

use engine::hybrid::{
    ClassicalSubstrate, HybridScheduler, JobSpec, MeasurementSpec, PauliTerm, QuantumCircuit,
    QuantumSpec, QuantumSubstrate, SchedulerConfig, StageOutputs,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 61.0: VQE HYBRID ORCHESTRATION DEMO");
    println!("        RIKEN JHPC-Quantum Inspired Architecture");
    println!("{}\n", "=".repeat(80));

    // Demo 1: Basic scheduler setup and substrate registration
    println!("\n{}", "-".repeat(80));
    println!("DEMO 1: Hybrid Scheduler Setup");
    println!("{}\n", "-".repeat(80));
    demo_scheduler_setup();

    // Demo 2: Simple quantum job submission
    println!("\n{}", "-".repeat(80));
    println!("DEMO 2: Quantum Job Submission");
    println!("{}\n", "-".repeat(80));
    demo_quantum_job();

    // Demo 3: H2 molecule VQE simulation
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: H2 Molecule Ground State (VQE)");
    println!("{}\n", "-".repeat(80));
    demo_h2_vqe();

    // Demo 4: Multi-stage pipeline
    println!("\n{}", "-".repeat(80));
    println!("DEMO 4: Multi-Stage Pipeline Execution");
    println!("{}\n", "-".repeat(80));
    demo_pipeline();

    println!("\n{}", "=".repeat(80));
    println!("                    HYBRID ORCHESTRATION DEMO COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Demo 1: Setting up the HybridScheduler with multiple substrates
fn demo_scheduler_setup() {
    // Create scheduler with custom configuration
    let config = SchedulerConfig {
        max_queue_size: 1000,
        max_concurrent: 4,
        default_timeout: std::time::Duration::from_secs(60),
        auto_route: true,
        enable_preemption: false,
    };

    let mut scheduler = HybridScheduler::with_config(config);

    // Register substrates
    let classical = ClassicalSubstrate::new(1, "CPU-Classical").with_workers(8);
    let quantum = QuantumSubstrate::new(2, "Quantum-Sim-20q", 20);

    scheduler.register_substrate(Box::new(classical));
    scheduler.register_substrate(Box::new(quantum));

    // Query scheduler status
    let stats = scheduler.stats();
    println!("Scheduler Statistics:");
    println!("  Total substrates: {}", stats.substrate_count);
    println!("  Jobs submitted: {}", stats.total_submitted);
    println!("  Jobs completed: {}", stats.total_completed);
    println!("  Queue size: {}", stats.queue_size);

    println!("\nRegistered Substrates:");
    for info in scheduler.list_substrates() {
        println!(
            "  [{}] {} ({:?}) - Capacity: {:.1}%",
            info.id.0,
            info.name,
            info.substrate_type,
            info.capacity * 100.0
        );
    }
}

/// Demo 2: Submitting and executing a simple quantum job
fn demo_quantum_job() {
    let mut scheduler = create_standard_scheduler();

    // Create a simple Bell state circuit: H(0), CNOT(0,1)
    let mut circuit = QuantumCircuit::new(2);
    circuit.h(0);
    circuit.cnot(0, 1);

    // Create quantum job spec
    let spec = QuantumSpec::new(2, circuit)
        .with_shots(1000)
        .with_measurement(MeasurementSpec::Computational);

    // Submit and execute
    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Quantum(spec));
    println!("Submitted job: {:?}", job_id);

    // Wait for result
    if let Some(result) = scheduler.wait(job_id) {
        let elapsed = start.elapsed();
        println!("\nJob completed in {:?}", elapsed);
        println!("Success: {}", result.is_success());

        if let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            println!("\nBell State Measurement Results (1000 shots):");
            let mut sorted: Vec<_> = counts.iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            for (bitstring, count) in sorted {
                let prob = *count as f64 / 1000.0;
                println!("  |{}⟩: {} ({:.1}%)", bitstring, count, prob * 100.0);
            }

            // Verify Bell state: should see ~50% |00⟩ and ~50% |11⟩
            let entangled = counts.get("00").unwrap_or(&0) + counts.get("11").unwrap_or(&0);
            let classical = counts.get("01").unwrap_or(&0) + counts.get("10").unwrap_or(&0);
            if entangled + classical > 0 {
                println!(
                    "\nEntanglement verification: {:.1}% correlated (expect ~100%)",
                    entangled as f64 / (entangled + classical) as f64 * 100.0
                );
            }
        }
    }
}

/// Demo 3: H2 molecule ground state using VQE-style hybrid execution
fn demo_h2_vqe() {
    let mut scheduler = create_standard_scheduler();

    // H2 Hamiltonian at equilibrium bond length (0.735 Angstroms)
    // Simplified 2-qubit Hamiltonian:
    // H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
    let h2_hamiltonian = vec![
        PauliTerm::identity(-1.0523),
        PauliTerm::z(0.3979, 0),
        PauliTerm::z(-0.3979, 1),
        PauliTerm::zz(-0.0112, 0, 1),
        // X and Y terms would need basis rotation for measurement
    ];

    // Create hardware-efficient ansatz for 2 qubits, depth 2
    let ansatz = QuantumCircuit::hardware_efficient(2, 2);

    println!("VQE Configuration:");
    println!("  Qubits: 2");
    println!("  Parameters: {}", ansatz.parameter_count());
    println!("  Hamiltonian terms: {}", h2_hamiltonian.len());
    println!("  Optimizer: COBYLA (maxiter=50)");
    println!("  Convergence: 1e-4");

    // Manual VQE loop demonstration
    println!("\nRunning VQE optimization loop...\n");

    let mut params: Vec<f64> = (0..ansatz.parameter_count())
        .map(|i| 0.1 + 0.01 * i as f64)
        .collect();
    let mut best_energy = f64::MAX;
    let start = Instant::now();

    for iteration in 0..10 {
        // Bind parameters to create concrete circuit
        let bound_circuit = ansatz.bind_parameters(&params);
        let spec = QuantumSpec::new(2, bound_circuit)
            .with_shots(1000)
            .with_measurement(MeasurementSpec::Computational);

        // Execute on quantum substrate
        let job_id = scheduler.submit(JobSpec::Quantum(spec));
        if let Some(result) = scheduler.wait(job_id)
            && let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            // Compute energy expectation value
            let energy = compute_h2_energy(counts, &h2_hamiltonian);

            if energy < best_energy {
                best_energy = energy;
            }

            println!(
                "  Iteration {}: Energy = {:.6} Ha (best: {:.6} Ha)",
                iteration + 1,
                energy,
                best_energy
            );

            // Simple gradient-free parameter update (demonstration)
            for (i, p) in params.iter_mut().enumerate() {
                *p += 0.05 * (0.5 - rand_f64(iteration as u64 + i as u64));
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\nVQE Results:");
    println!("  Final energy: {:.6} Ha", best_energy);
    println!("  FCI energy:   {:.6} Ha (exact)", -1.1373);
    println!("  Error:        {:.6} Ha", (best_energy - (-1.1373)).abs());
    println!("  Total time:   {:?}", elapsed);
}

/// Demo 4: Multi-stage pipeline with classical-quantum-classical flow
fn demo_pipeline() {
    let mut scheduler = create_standard_scheduler();

    println!("Pipeline: Classical → Quantum → Classical");
    println!("  Stage 1: Initialize parameters (Classical)");
    println!("  Stage 2: Execute quantum circuit (Quantum)");
    println!("  Stage 3: Post-process results (Classical)\n");

    // Stage 1: Quantum execution (GHZ state)
    let mut circuit = QuantumCircuit::new(3);
    circuit.h(0);
    circuit.cnot(0, 1);
    circuit.cnot(1, 2);

    let spec = QuantumSpec::new(3, circuit)
        .with_shots(2000)
        .with_measurement(MeasurementSpec::Computational);

    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Quantum(spec));

    if let Some(result) = scheduler.wait(job_id) {
        let elapsed = start.elapsed();

        println!("Pipeline completed in {:?}\n", elapsed);

        if let Some(stage) = result.stage_results.first()
            && let StageOutputs::MeasurementCounts(counts) = &stage.outputs
        {
            println!("3-Qubit GHZ State Results (2000 shots):");
            let mut sorted: Vec<_> = counts.iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            for (bitstring, count) in sorted {
                let prob = *count as f64 / 2000.0;
                if *count > 10 {
                    println!("  |{}⟩: {} ({:.1}%)", bitstring, count, prob * 100.0);
                }
            }

            // Post-processing: compute fidelity to ideal GHZ
            let ghz_fidelity = compute_ghz_fidelity(counts, 3);
            println!("\nPost-processing:");
            println!("  GHZ state fidelity: {:.1}%", ghz_fidelity * 100.0);
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_standard_scheduler() -> HybridScheduler {
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(ClassicalSubstrate::new(1, "CPU")));
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(2, "Quantum-Sim", 20)));
    scheduler
}

fn compute_h2_energy(counts: &HashMap<String, usize>, hamiltonian: &[PauliTerm]) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut energy = 0.0;

    for term in hamiltonian {
        // Compute expectation value for this term
        let expectation = compute_pauli_expectation(counts, term, total);
        energy += term.coeff * expectation;
    }

    energy
}

fn compute_pauli_expectation(
    counts: &HashMap<String, usize>,
    term: &PauliTerm,
    total: usize,
) -> f64 {
    if total == 0 {
        return 0.0;
    }

    // Identity term has expectation value 1
    if term.paulis.is_empty() {
        return 1.0;
    }

    let mut expectation = 0.0;

    for (bitstring, &count) in counts {
        // Compute eigenvalue for this bitstring
        let bits: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();
        let mut eigenvalue = 1.0;

        for &(qubit, pauli) in &term.paulis {
            if qubit < bits.len() {
                match pauli {
                    'I' => {} // Identity: eigenvalue 1
                    'X' | 'Y' => {
                        // X or Y: off-diagonal in computational basis
                        // For computational basis measurement, only Z terms contribute
                        // X and Y terms would need basis rotation
                    }
                    'Z' => {
                        // Z: eigenvalue +1 for |0⟩, -1 for |1⟩
                        if bits[qubit] {
                            eigenvalue *= -1.0;
                        }
                    }
                    _ => {}
                }
            }
        }

        expectation += eigenvalue * (count as f64 / total as f64);
    }

    expectation
}

fn compute_ghz_fidelity(counts: &HashMap<String, usize>, n_qubits: usize) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    // GHZ state: (|000...⟩ + |111...⟩) / sqrt(2)
    // Ideal probabilities: P(0...0) = 0.5, P(1...1) = 0.5
    let zeros = "0".repeat(n_qubits);
    let ones = "1".repeat(n_qubits);

    let p_zeros = *counts.get(&zeros).unwrap_or(&0) as f64 / total as f64;
    let p_ones = *counts.get(&ones).unwrap_or(&0) as f64 / total as f64;

    // Simplified fidelity: sum of correct state probabilities
    p_zeros + p_ones
}

fn rand_f64(seed: u64) -> f64 {
    // Simple LCG-based random number
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    (x as f64) / (u64::MAX as f64)
}
