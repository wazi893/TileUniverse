// =============================================================================
// Sprint 67.0: QAOA MaxCut Demo
// =============================================================================
//
// Demonstrates Quantum Approximate Optimization Algorithm (QAOA) for solving
// MaxCut problems using the hybrid orchestration layer.
//
// This example shows:
// - Graph creation (cycle, complete, random)
// - QAOA ansatz construction
// - MaxCut cost function evaluation
// - Comparison with brute-force optimal solution
//
// Run with:
//   cargo run --example qaoa_maxcut_hybrid --release
//
// =============================================================================

use std::time::Instant;

use engine::hybrid::{
    CostFunction, Gate, HybridScheduler, Optimizer, QuantumCircuit, QuantumSubstrate, VariationalExecutor, VariationalSpec,
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 67.0: QAOA MAXCUT DEMO");
    println!("        Quantum Approximate Optimization Algorithm");
    println!("{}\n", "=".repeat(80));

    // Create scheduler
    let mut scheduler = HybridScheduler::new();
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(1, "Quantum", 16)));

    // Demo 1: Small cycle graph
    println!("{}", "-".repeat(80));
    println!("DEMO 1: MaxCut on Cycle Graph (4 vertices)");
    println!("{}\n", "-".repeat(80));
    demo_cycle_graph(&mut scheduler);

    // Demo 2: Complete graph
    println!("\n{}", "-".repeat(80));
    println!("DEMO 2: MaxCut on Complete Graph K4");
    println!("{}\n", "-".repeat(80));
    demo_complete_graph(&mut scheduler);

    // Demo 3: Random graph
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: MaxCut on Random Graph");
    println!("{}\n", "-".repeat(80));
    demo_random_graph(&mut scheduler);

    // Demo 4: Scaling comparison
    println!("\n{}", "-".repeat(80));
    println!("DEMO 4: QAOA Depth Comparison");
    println!("{}\n", "-".repeat(80));
    demo_depth_comparison(&mut scheduler);

    println!("\n{}", "=".repeat(80));
    println!("                    QAOA MAXCUT DEMO COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Create edges for a cycle graph.
fn cycle_graph(n: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n, 1.0));
    }
    edges
}

/// Create edges for a complete graph.
fn complete_graph(n: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j, 1.0));
        }
    }
    edges
}

/// Create edges for a random graph.
fn random_graph(n: usize, density: f64, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    let mut rng = seed;

    for i in 0..n {
        for j in (i + 1)..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng as f64) / (u64::MAX as f64);
            if r < density {
                edges.push((i, j, 1.0));
            }
        }
    }
    edges
}

/// Create QAOA ansatz circuit for MaxCut.
fn qaoa_ansatz(n_qubits: usize, edges: &[(usize, usize, f64)], depth: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);

    // Initial superposition
    for q in 0..n_qubits {
        circuit.h(q);
    }

    // QAOA layers
    for d in 0..depth {
        // Cost layer: exp(-i γ C)
        // For MaxCut: C = Σ (1 - Z_i Z_j) / 2
        // We parameterize with γ via RZZ gates
        for (i, &(u, v, _w)) in edges.iter().enumerate() {
            // ZZ interaction via CNOT-RZ-CNOT
            circuit.cnot(u, v);
            circuit
                .gates
                .push(Gate::RzParam(v, circuit.parameters.len()));
            circuit.parameters.push(format!("gamma_{}_{}", d, i));
            circuit.cnot(u, v);
        }

        // Mixer layer: exp(-i β B)
        // B = Σ X_i
        for q in 0..n_qubits {
            circuit
                .gates
                .push(Gate::RxParam(q, circuit.parameters.len()));
            circuit.parameters.push(format!("beta_{}_{}", d, q));
        }
    }

    circuit
}

/// Compute optimal MaxCut by brute force.
fn brute_force_maxcut(n: usize, edges: &[(usize, usize, f64)]) -> (usize, String) {
    let mut best_cut = 0;
    let mut best_bitstring = String::new();

    for i in 0..(1 << n) {
        let mut cut = 0;
        for &(u, v, w) in edges {
            let bu = (i >> u) & 1;
            let bv = (i >> v) & 1;
            if bu != bv {
                cut += w as usize;
            }
        }
        if cut > best_cut {
            best_cut = cut;
            best_bitstring = format!("{:0width$b}", i, width = n);
        }
    }

    (best_cut, best_bitstring)
}

/// Demo QAOA on cycle graph.
fn demo_cycle_graph(scheduler: &mut HybridScheduler) {
    let n = 4;
    let edges = cycle_graph(n);
    let depth = 1;

    println!("Graph: Cycle with {} vertices", n);
    println!("Edges: {:?}", edges);

    let ansatz = qaoa_ansatz(n, &edges, depth);
    println!("QAOA depth: {}", depth);
    println!("Parameters: {}", ansatz.parameter_count());

    // Compute optimal
    let (optimal_cut, optimal_bitstring) = brute_force_maxcut(n, &edges);
    println!(
        "Optimal cut: {} (bitstring: {})",
        optimal_cut, optimal_bitstring
    );
    println!();

    let spec = VariationalSpec::new(
        ansatz,
        CostFunction::MaxCut {
            edges: edges.clone(),
        },
    )
    .with_optimizer(Optimizer::SPSA {
        a: 0.3,
        c: 0.2,
        maxiter: 100,
    })
    .with_max_iterations(50)
    .with_shots(500);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler);
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    // Note: Our cost function returns negative of cut value (we minimize)
    let found_cut = (-result.optimal_cost).round() as usize;
    let approx_ratio = found_cut as f64 / optimal_cut as f64;

    println!("QAOA Results:");
    println!("  Found cut: {}", found_cut);
    println!("  Approximation ratio: {:.2}%", approx_ratio * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:?}", elapsed);
}

/// Demo QAOA on complete graph.
fn demo_complete_graph(scheduler: &mut HybridScheduler) {
    let n = 4;
    let edges = complete_graph(n);
    let depth = 2;

    println!("Graph: Complete K{}", n);
    println!("Edges: {} (all pairs)", edges.len());

    let ansatz = qaoa_ansatz(n, &edges, depth);
    println!("QAOA depth: {}", depth);
    println!("Parameters: {}", ansatz.parameter_count());

    let (optimal_cut, _) = brute_force_maxcut(n, &edges);
    println!("Optimal cut: {}", optimal_cut);
    println!();

    let spec = VariationalSpec::new(
        ansatz,
        CostFunction::MaxCut {
            edges: edges.clone(),
        },
    )
    .with_optimizer(Optimizer::NelderMead { maxiter: 100 })
    .with_max_iterations(50)
    .with_shots(500);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler);
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    let found_cut = (-result.optimal_cost).round() as usize;
    let approx_ratio = found_cut as f64 / optimal_cut as f64;

    println!("QAOA Results:");
    println!("  Found cut: {}", found_cut);
    println!("  Approximation ratio: {:.2}%", approx_ratio * 100.0);
    println!("  Time: {:?}", elapsed);
}

/// Demo QAOA on random graph.
fn demo_random_graph(scheduler: &mut HybridScheduler) {
    let n = 5;
    let edges = random_graph(n, 0.5, 42);
    let depth = 2;

    println!("Graph: Random with {} vertices, density 0.5", n);
    println!("Edges: {}", edges.len());

    let ansatz = qaoa_ansatz(n, &edges, depth);
    println!("QAOA depth: {}", depth);
    println!("Parameters: {}", ansatz.parameter_count());

    let (optimal_cut, optimal_bitstring) = brute_force_maxcut(n, &edges);
    println!(
        "Optimal cut: {} (bitstring: {})",
        optimal_cut, optimal_bitstring
    );
    println!();

    let spec = VariationalSpec::new(
        ansatz,
        CostFunction::MaxCut {
            edges: edges.clone(),
        },
    )
    .with_optimizer(Optimizer::SPSA {
        a: 0.2,
        c: 0.15,
        maxiter: 100,
    })
    .with_max_iterations(60)
    .with_shots(500);

    let start = Instant::now();
    let mut executor = VariationalExecutor::new(scheduler);
    let result = executor.run(&spec);
    let elapsed = start.elapsed();

    let found_cut = (-result.optimal_cost).round() as usize;
    let approx_ratio = found_cut as f64 / optimal_cut as f64;

    println!("QAOA Results:");
    println!("  Found cut: {}", found_cut);
    println!("  Approximation ratio: {:.2}%", approx_ratio * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:?}", elapsed);
}

/// Demo depth comparison.
fn demo_depth_comparison(scheduler: &mut HybridScheduler) {
    let n = 4;
    let edges = cycle_graph(n);
    let (optimal_cut, _) = brute_force_maxcut(n, &edges);

    println!("Comparing QAOA depths on cycle graph with {} vertices", n);
    println!("Optimal cut: {}\n", optimal_cut);

    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12}",
        "Depth", "Parameters", "Cut Found", "Approx %", "Time (ms)"
    );
    println!("{}", "-".repeat(60));

    for depth in 1..=4 {
        let ansatz = qaoa_ansatz(n, &edges, depth);
        let n_params = ansatz.parameter_count();

        let spec = VariationalSpec::new(
            ansatz,
            CostFunction::MaxCut {
                edges: edges.clone(),
            },
        )
        .with_optimizer(Optimizer::SPSA {
            a: 0.3,
            c: 0.2,
            maxiter: 50,
        })
        .with_max_iterations(30)
        .with_shots(300);

        let start = Instant::now();
        let mut executor = VariationalExecutor::new(scheduler);
        let result = executor.run(&spec);
        let elapsed = start.elapsed();

        let found_cut = (-result.optimal_cost).round() as usize;
        let approx_ratio = found_cut as f64 / optimal_cut as f64 * 100.0;
        let time_ms = elapsed.as_secs_f64() * 1000.0;

        println!(
            "{:>6} {:>12} {:>12} {:>11.1}% {:>11.1}",
            depth, n_params, found_cut, approx_ratio, time_ms
        );
    }

    println!();
    println!("Note: Higher depth can find better solutions but requires more optimization.");
}
