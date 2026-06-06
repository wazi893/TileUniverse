// =============================================================================
// QAOA Demo - Quantum Approximate Optimization Algorithm
// =============================================================================
//
// Demonstrates QAOA for MaxCut on various graphs.
//
// Run: cargo run --example qaoa_demo --release
//
// =============================================================================

use engine::algorithms::qaoa::{Graph, QAOAConfig, brute_force_maxcut, maxcut_p, run_qaoa};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     QAOA (Quantum Approximate Optimization Algorithm) Demo       ║");
    println!("║                    SPRINT 42.0 - MaxCut Solver                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    demo_1_triangle();
    demo_2_square();
    demo_3_random_graph();
    demo_4_depth_scaling();
    demo_5_optimizer_comparison();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      Demo Complete!                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

/// Demo 1: Simple triangle graph (3 vertices)
fn demo_1_triangle() {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Triangle Graph (K₃)                                      │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Graph: 3 vertices, 3 edges (complete triangle)");
    println!("  Optimal MaxCut: 2 edges (any single vertex vs other two)");
    println!();

    let graph = Graph::complete(3);
    let (optimal, _) = brute_force_maxcut(&graph);

    let start = Instant::now();
    let result = maxcut_p(graph, 2);
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    QAOA cut:     {:.1}", result.best_cut);
    println!("    Optimal cut:  {:.1}", optimal);
    println!(
        "    Ratio:        {:.2}",
        result.approximation_ratio.unwrap_or(0.0)
    );
    println!("    Partition:    {:?}", result.best_bitstring);
    println!("    Iterations:   {}", result.iterations);
    println!("    Time:         {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

/// Demo 2: Square graph (4-cycle)
fn demo_2_square() {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: Square Graph (C₄)                                        │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Graph: 4 vertices in a cycle (square)");
    println!("  Optimal MaxCut: 4 edges (checkerboard pattern)");
    println!();

    let graph = Graph::cycle(4);
    let (optimal, optimal_partition) = brute_force_maxcut(&graph);

    let start = Instant::now();
    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(3).with_shots(500).with_seed(42),
    );
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    QAOA cut:        {:.1}", result.best_cut);
    println!("    Optimal cut:     {:.1}", optimal);
    println!(
        "    Ratio:           {:.2}",
        result.approximation_ratio.unwrap_or(0.0)
    );
    println!("    QAOA partition:  {:?}", result.best_bitstring);
    println!("    Optimal part:    {:?}", optimal_partition);
    println!("    Converged:       {}", result.converged);
    println!(
        "    Time:            {:.1}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!();
}

/// Demo 3: Random graph
fn demo_3_random_graph() {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: Random Erdős-Rényi Graph                                 │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();

    let n = 6;
    let p = 0.5;
    let graph = Graph::random(n, p, 12345);

    println!(
        "  Graph: {} vertices, {} edges (p={})",
        n,
        graph.n_edges(),
        p
    );

    let (optimal, _) = brute_force_maxcut(&graph);
    println!("  Optimal MaxCut: {:.1} edges", optimal);
    println!();

    let start = Instant::now();
    let result = run_qaoa(
        graph,
        QAOAConfig::with_depth(3).with_shots(1000).with_seed(42),
    );
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    QAOA cut:     {:.1}", result.best_cut);
    println!("    Optimal cut:  {:.1}", optimal);
    println!(
        "    Ratio:        {:.2}",
        result.approximation_ratio.unwrap_or(0.0)
    );
    println!(
        "    Parameters:   γ={:.3}, β={:.3}",
        result.optimal_params[0],
        result.optimal_params[result.optimal_params.len() / 2]
    );
    println!("    Evaluations:  {}", result.evaluations);
    println!("    Time:         {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

/// Demo 4: Effect of QAOA depth (p)
fn demo_4_depth_scaling() {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 4: QAOA Depth Scaling                                       │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();

    let graph = Graph::random(5, 0.6, 999);
    let (optimal, _) = brute_force_maxcut(&graph);

    println!("  Graph: 5 vertices, {} edges", graph.n_edges());
    println!("  Optimal MaxCut: {:.1}", optimal);
    println!();
    println!("  Depth (p) │ Cut Value │ Ratio  │ Time");
    println!("  ──────────┼───────────┼────────┼──────────");

    for p in 1..=4 {
        let start = Instant::now();
        let result = run_qaoa(
            graph.clone(),
            QAOAConfig::with_depth(p).with_shots(500).with_seed(42),
        );
        let elapsed = start.elapsed();

        println!(
            "      {}     │   {:.1}     │ {:.2}   │ {:.1}ms",
            p,
            result.best_cut,
            result.approximation_ratio.unwrap_or(0.0),
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();
}

/// Demo 5: Optimizer comparison
fn demo_5_optimizer_comparison() {
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 5: Optimizer Comparison                                     │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();

    let graph = Graph::cycle(5);
    let (optimal, _) = brute_force_maxcut(&graph);

    println!("  Graph: Pentagon (C₅)");
    println!("  Optimal MaxCut: {:.1}", optimal);
    println!();
    println!("  Optimizer      │ Cut │ Ratio  │ Evals │ Time");
    println!("  ───────────────┼─────┼────────┼───────┼─────────");

    // Nelder-Mead
    let start = Instant::now();
    let result = run_qaoa(
        graph.clone(),
        QAOAConfig::with_depth(2).with_shots(200).with_seed(42),
    );
    let elapsed = start.elapsed();
    println!(
        "  Nelder-Mead    │ {:.1} │ {:.2}   │ {:>5} │ {:.1}ms",
        result.best_cut,
        result.approximation_ratio.unwrap_or(0.0),
        result.evaluations,
        elapsed.as_secs_f64() * 1000.0
    );

    // SPSA
    let start = Instant::now();
    let result = run_qaoa(
        graph.clone(),
        QAOAConfig::with_depth(2)
            .with_spsa()
            .with_shots(200)
            .with_seed(42),
    );
    let elapsed = start.elapsed();
    println!(
        "  SPSA           │ {:.1} │ {:.2}   │ {:>5} │ {:.1}ms",
        result.best_cut,
        result.approximation_ratio.unwrap_or(0.0),
        result.evaluations,
        elapsed.as_secs_f64() * 1000.0
    );

    // Gradient Descent
    let start = Instant::now();
    let result = run_qaoa(
        graph.clone(),
        QAOAConfig::with_depth(2)
            .with_gradient_descent(0.1)
            .with_shots(200)
            .with_seed(42),
    );
    let elapsed = start.elapsed();
    println!(
        "  Grad Descent   │ {:.1} │ {:.2}   │ {:>5} │ {:.1}ms",
        result.best_cut,
        result.approximation_ratio.unwrap_or(0.0),
        result.evaluations,
        elapsed.as_secs_f64() * 1000.0
    );

    println!();
}
