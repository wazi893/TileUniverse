//! Phase 1 Gate Check: Search Substrate Foundation
//!
//! Sprint 65.0 - Quantum-Enhanced Neural Search Engine
//!
//! Gate criteria:
//! - OneMax-64 converges (fitness >= 250) in <100 generations
//! - 1M population evaluates in reasonable time
//! - All selection strategies work correctly
//! - Telemetry captures per-generation metrics

use engine::search::{
    baseline::{BaselineConfig, ClassicalGA},
    fitness::{MaxSat, NKLandscape, OneMax},
    selection::SelectionStrategy,
    substrate::SearchSubstrate,
};
use std::time::Instant;

/// Gate Check 1: OneMax-64 converges in <100 generations at 1M scale
#[test]
fn gate_onemax64_convergence() {
    println!("\n=== PHASE 1 GATE CHECK: OneMax-64 Convergence ===\n");

    let config = BaselineConfig::default()
        .with_population(1_000_000)
        .with_max_generations(100)
        .with_selection(SelectionStrategy::Tournament { tournament_size: 7 })
        .with_crossover_rate(0.9)
        .with_mutation_rate(1.0 / 64.0) // ~1.5% per bit
        .with_elite_count(100)
        .with_seed(42)
        .with_stop_on_optimal(true);

    let mut ga = ClassicalGA::new(config, OneMax::standard());

    let start = Instant::now();
    let result = ga.run();
    let duration = start.elapsed();

    println!("Population: 1,000,000");
    println!("Generations: {}", result.generations);
    println!("Runtime: {:?}", duration);
    println!("Evals/sec: {:.2e}", result.evals_per_second);
    println!("Best fitness: {} (target: 255)", result.best.fitness);
    println!("Best found at generation: {}", result.best_generation);
    println!("Optimal reached: {}", result.optimal_reached);

    // Gate check criteria
    assert!(
        result.best.fitness >= 250,
        "Gate failed: fitness {} < 250",
        result.best.fitness
    );

    assert!(
        result.generations <= 100,
        "Gate failed: took {} generations (limit: 100)",
        result.generations
    );

    println!(
        "\n✓ GATE CHECK PASSED: OneMax-64 converged in {} generations\n",
        result.generations
    );
}

/// Gate Check 2: NK-Landscape with moderate ruggedness
#[test]
fn gate_nk_landscape() {
    println!("\n=== PHASE 1 GATE CHECK: NK-Landscape ===\n");

    // N=32, K=4 (moderate epistasis)
    let nk = NKLandscape::new(32, 4, 12345);

    let config = BaselineConfig::default()
        .with_population(100_000)
        .with_max_generations(50)
        .with_selection(SelectionStrategy::Tournament { tournament_size: 5 })
        .with_mutation_rate(0.03)
        .with_seed(42);

    let mut ga = ClassicalGA::new(config, nk);

    let start = Instant::now();
    let result = ga.run();
    let duration = start.elapsed();

    println!("NK-Landscape N=32, K=4");
    println!("Population: 100,000");
    println!("Generations: {}", result.generations);
    println!("Runtime: {:?}", duration);
    println!("Best fitness: {}", result.best.fitness);
    println!("Mean fitness: {:.2}", result.final_stats.mean_fitness);

    // NK landscape is harder - just verify it runs and improves
    assert!(
        result.best.fitness > result.final_stats.worst_fitness,
        "GA should improve fitness on NK landscape"
    );

    // Verify telemetry was collected
    assert!(!result.telemetry.is_empty(), "Should have telemetry data");

    println!("\n✓ GATE CHECK PASSED: NK-Landscape optimization working\n");
}

/// Gate Check 3: MaxSAT with planted solution
#[test]
fn gate_maxsat() {
    println!("\n=== PHASE 1 GATE CHECK: MaxSAT ===\n");

    // Create a satisfiable instance with planted solution
    let maxsat = MaxSat::random_satisfiable(32, 100, 42);

    let config = BaselineConfig::default()
        .with_population(50_000)
        .with_max_generations(100)
        .with_selection(SelectionStrategy::Truncation { survival_rate: 0.1 })
        .with_mutation_rate(0.02)
        .with_seed(42);

    let mut ga = ClassicalGA::new(config, maxsat);

    let start = Instant::now();
    let result = ga.run();
    let duration = start.elapsed();

    println!("MaxSAT: 32 vars, 100 clauses (satisfiable)");
    println!("Population: 50,000");
    println!("Generations: {}", result.generations);
    println!("Runtime: {:?}", duration);
    println!(
        "Best fitness: {} (255 = all satisfied)",
        result.best.fitness
    );

    // Should satisfy most clauses
    assert!(
        result.best.fitness >= 200,
        "Should satisfy >80% of clauses, got {}",
        result.best.fitness
    );

    println!("\n✓ GATE CHECK PASSED: MaxSAT optimization working\n");
}

/// Gate Check 4: Selection strategy comparison
#[test]
fn gate_selection_strategies() {
    println!("\n=== PHASE 1 GATE CHECK: Selection Strategies ===\n");

    let strategies = vec![
        (
            "Tournament(7)",
            SelectionStrategy::Tournament { tournament_size: 7 },
        ),
        ("Roulette", SelectionStrategy::Roulette),
        (
            "Truncation(0.2)",
            SelectionStrategy::Truncation { survival_rate: 0.2 },
        ),
        ("Rank(1.5)", SelectionStrategy::Rank { pressure: 1.5 }),
        ("SUS", SelectionStrategy::SUS),
    ];

    let mut results = Vec::new();

    for (name, strategy) in strategies {
        let config = BaselineConfig::default()
            .with_population(10_000)
            .with_max_generations(30)
            .with_selection(strategy)
            .with_seed(42);

        let mut ga = ClassicalGA::new(config, OneMax::new(32));
        let result = ga.run();

        println!(
            "{:20} -> fitness: {:3}, generations: {:3}",
            name, result.best.fitness, result.generations
        );

        results.push((name, result.best.fitness));
    }

    // All strategies should achieve reasonable fitness
    for (name, fitness) in &results {
        assert!(
            *fitness >= 180,
            "Strategy {} underperformed with fitness {}",
            name,
            fitness
        );
    }

    println!("\n✓ GATE CHECK PASSED: All selection strategies working\n");
}

/// Gate Check 5: Telemetry and instrumentation
#[test]
fn gate_telemetry() {
    println!("\n=== PHASE 1 GATE CHECK: Telemetry ===\n");

    let config = BaselineConfig::default()
        .with_population(10_000)
        .with_max_generations(20)
        .with_stop_on_optimal(false) // Run all generations for telemetry
        .with_seed(42);

    let mut ga = ClassicalGA::new(config, OneMax::new(16));
    let result = ga.run();

    // Verify telemetry was collected for all generations
    assert_eq!(
        result.telemetry.len(),
        20,
        "Should have telemetry for all 20 generations"
    );

    // Verify telemetry contents
    for (i, t) in result.telemetry.iter().enumerate() {
        assert_eq!(t.generation, i as u64, "Generation number mismatch");
        assert!(t.best_fitness > 0, "Best fitness should be > 0");
        assert!(t.mean_fitness > 0.0, "Mean fitness should be > 0");
        assert!(t.generation_time.as_nanos() > 0, "Should have timing data");
    }

    // Check fitness improvement trend
    let first_fitness = result.telemetry.first().unwrap().best_fitness;
    let last_fitness = result.telemetry.last().unwrap().best_fitness;
    assert!(
        last_fitness >= first_fitness,
        "Fitness should improve or stay same"
    );

    // Verify diversity is tracked
    let has_diversity = result.telemetry.iter().any(|t| t.diversity > 0.0);
    assert!(has_diversity, "Should track diversity");

    println!("Telemetry records: {}", result.telemetry.len());
    println!("First generation fitness: {}", first_fitness);
    println!("Last generation fitness: {}", last_fitness);
    println!("Fitness improvement: {}", last_fitness - first_fitness);

    println!("\n✓ GATE CHECK PASSED: Telemetry working correctly\n");
}

/// Gate Check 6: 1M scale performance
#[test]
fn gate_1m_performance() {
    println!("\n=== PHASE 1 GATE CHECK: 1M Scale Performance ===\n");

    // Test substrate initialization and evaluation at 1M scale
    let mut substrate = SearchSubstrate::new(1_000_000, OneMax::standard());

    let init_start = Instant::now();
    substrate.initialize_random(42);
    let init_time = init_start.elapsed();

    let eval_start = Instant::now();
    substrate.evaluate_all();
    let eval_time = eval_start.elapsed();

    let stats = substrate.statistics();

    println!("Population: 1,000,000");
    println!("Initialization time: {:?}", init_time);
    println!("Evaluation time: {:?}", eval_time);
    println!(
        "Evaluations/second: {:.2e}",
        1_000_000.0 / eval_time.as_secs_f64()
    );
    println!("Best fitness: {}", stats.best_fitness);
    println!("Mean fitness: {:.2}", stats.mean_fitness);
    println!(
        "Diversity (sample): {:.2}",
        substrate.compute_diversity(100)
    );

    // Performance criteria
    assert!(
        init_time.as_secs() < 5,
        "Initialization too slow: {:?}",
        init_time
    );
    assert!(
        eval_time.as_secs() < 2,
        "Evaluation too slow: {:?}",
        eval_time
    );

    // Statistics should be reasonable for random population
    assert!(stats.best_fitness > 50, "Should have some good candidates");
    assert!(stats.mean_fitness > 100.0, "Mean should be ~128 for random");

    println!("\n✓ GATE CHECK PASSED: 1M scale performance acceptable\n");
}

/// Gate Check 7: Full Phase 1 integration
#[test]
fn gate_phase1_integration() {
    println!("\n=== PHASE 1 GATE CHECK: Full Integration ===\n");

    // This is the definitive Phase 1 gate check
    let config = BaselineConfig::default()
        .with_population(1_000_000)
        .with_max_generations(100)
        .with_selection(SelectionStrategy::Tournament { tournament_size: 7 })
        .with_crossover_rate(0.9)
        .with_mutation_rate(1.0 / 64.0)
        .with_elite_count(100)
        .with_seed(65); // Sprint 65 seed

    let mut ga = ClassicalGA::new(config, OneMax::standard());

    let start = Instant::now();
    let result = ga.run();
    let total_time = start.elapsed();

    // Print comprehensive results
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║            SPRINT 65.0 - PHASE 1 GATE CHECK               ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Problem:       OneMax-64                                  ║");
    println!("║ Population:    1,000,000                                  ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!(
        "║ Best Fitness:  {:3} / 255                                  ║",
        result.best.fitness
    );
    println!(
        "║ Generations:   {:3} / 100                                  ║",
        result.generations
    );
    println!(
        "║ Runtime:       {:.2?}                                    ║",
        total_time
    );
    println!(
        "║ Evals/sec:     {:.2e}                              ║",
        result.evals_per_second
    );
    println!(
        "║ Optimal:       {}                                       ║",
        if result.optimal_reached { "YES" } else { "NO " }
    );
    println!("╠═══════════════════════════════════════════════════════════╣");

    // Determine gate status
    let gate_passed = result.best.fitness >= 250 && result.generations <= 100;

    if gate_passed {
        println!("║                  ✓ GATE PASSED                            ║");
    } else {
        println!("║                  ✗ GATE FAILED                            ║");
    }
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    assert!(
        gate_passed,
        "Phase 1 gate check failed: fitness={}, generations={}",
        result.best.fitness, result.generations
    );
}
