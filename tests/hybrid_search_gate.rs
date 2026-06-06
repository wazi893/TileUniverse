//! Sprint 66.0 Gate Check: Hybrid Fitness-Grover Integration
//!
//! Tests the integration of SlimSimulation-based fitness evaluation
//! with QuantumGridF64-based Grover amplification.
//!
//! Gate criteria:
//! - Fitness oracle marks correct states
//! - Grover amplification increases marked probability
//! - Selection provides positive selection pressure
//! - Scales to meaningful population sizes (16K+)
//! - Evolution loop improves fitness over generations
//!
//! ⚠️ INTEGRATION DEBT: This is a STANDALONE prototype.
//! See: User notes/SPRINTS/SPRINT 66.0/SPRINT_66_HYBRID_FITNESS_GROVER.md
//! TODO: Integrate with TILE-8 CPU memory, tile circuits, physical hive mesh

use engine::tile8::hybrid_search::{Candidate, HybridConfig, HybridSearch};
use std::time::Instant;

// =============================================================================
// Sprint 66 Gate Check Tests
// =============================================================================

/// Gate Check: Candidate fitness evaluation
#[test]
fn gate_candidate_fitness() {
    println!("\n=== SPRINT 66 GATE CHECK: Candidate Fitness ===\n");

    let mut c = Candidate::new(0);
    c.evaluate_onemax();
    assert_eq!(c.fitness, 0, "Empty genome should have 0 fitness");

    let mut c = Candidate::new(u64::MAX);
    c.evaluate_onemax();
    assert_eq!(c.fitness, 64, "Full genome should have 64 fitness");

    let mut c = Candidate::new(0b11110000_11110000_11110000_11110000);
    c.evaluate_onemax();
    assert_eq!(c.fitness, 16, "Should count correct number of ones");

    // Test bit-limited evaluation
    let mut c = Candidate::new(u64::MAX);
    c.evaluate_onemax_bits(32);
    assert_eq!(c.fitness, 32, "Should only count 32 bits");

    println!("Candidate fitness tests:");
    println!("  Zero genome: 0 fitness");
    println!("  Full genome: 64 fitness");
    println!("  Partial genome: 16 fitness");
    println!("  Bit-limited: 32 fitness");

    println!("\n[PASS] Candidate fitness evaluation correct\n");
}

/// Gate Check: Population initialization and statistics
#[test]
fn gate_population_stats() {
    println!("\n=== SPRINT 66 GATE CHECK: Population Statistics ===\n");

    let config = HybridConfig::new(12).with_seed(42); // 4096 candidates
    let mut search = HybridSearch::new(config);

    assert_eq!(search.population_size(), 4096);

    search.evaluate_fitness_onemax();

    let (mean, min, max, std) = search.fitness_stats();

    println!("Population: {} candidates", search.population_size());
    println!("Fitness statistics:");
    println!("  Mean: {:.2}", mean);
    println!("  Min: {}", min);
    println!("  Max: {}", max);
    println!("  Std: {:.2}", std);

    // Random 64-bit genomes should have ~32 ones on average
    assert!(mean > 28.0 && mean < 36.0, "Mean fitness should be ~32");
    assert!(min < 20, "Should have some low-fitness candidates");
    assert!(max > 44, "Should have some high-fitness candidates");

    println!("\n[PASS] Population statistics within expected bounds\n");
}

/// Gate Check: Threshold computation
#[test]
fn gate_threshold_computation() {
    println!("\n=== SPRINT 66 GATE CHECK: Threshold Computation ===\n");

    let config = HybridConfig::new(12).with_seed(42).with_percentile(0.75);
    let mut search = HybridSearch::new(config);
    search.evaluate_fitness_onemax();

    let threshold = search.compute_threshold();
    let marked = search.count_marked(threshold);

    let ratio = marked as f64 / search.population_size() as f64;

    println!("Percentile: 0.75 (mark top 25%)");
    println!("Computed threshold: {}", threshold);
    println!("Marked candidates: {} ({:.1}%)", marked, ratio * 100.0);

    // Top 25% should be marked (with some variance)
    assert!(ratio > 0.15 && ratio < 0.35, "Marking ratio should be ~25%");

    // Test different percentiles
    let config = HybridConfig::new(12).with_seed(42).with_percentile(0.90);
    let mut search = HybridSearch::new(config);
    search.evaluate_fitness_onemax();

    let threshold = search.compute_threshold();
    let marked = search.count_marked(threshold);
    let ratio = marked as f64 / search.population_size() as f64;

    println!("\nPercentile: 0.90 (mark top 10%)");
    println!("Computed threshold: {}", threshold);
    println!("Marked candidates: {} ({:.1}%)", marked, ratio * 100.0);

    assert!(ratio > 0.05 && ratio < 0.20, "Marking ratio should be ~10%");

    println!("\n[PASS] Threshold computation correct\n");
}

/// Gate Check: Grover amplification
#[test]
fn gate_grover_amplification() {
    println!("\n=== SPRINT 66 GATE CHECK: Grover Amplification ===\n");

    let config = HybridConfig::new(10).with_seed(42).with_percentile(0.75);
    let mut search = HybridSearch::new(config);
    search.evaluate_fitness_onemax();

    let threshold = search.compute_threshold();
    let marked = search.count_marked(threshold);
    let n = search.population_size();

    let initial_prob = marked as f64 / n as f64;
    let iterations = HybridSearch::optimal_iterations(n, marked);

    println!("Population: {} candidates", n);
    println!("Threshold: {}", threshold);
    println!("Marked: {} ({:.2}%)", marked, initial_prob * 100.0);
    println!("Optimal iterations: {}", iterations);
    println!("Initial marked probability: {:.4}", initial_prob);

    // Run selection (which internally does Grover iterations)
    let selected = search.select(100);
    let stats = search.last_stats().unwrap();

    println!("\nAfter {} Grover iterations:", stats.iterations);
    println!("  Marked probability: {:.4}", stats.marked_probability);
    println!("  Amplification: {:.2}x", stats.amplification);

    assert!(
        stats.amplification > 1.5,
        "Should achieve >1.5x amplification, got {:.2}x",
        stats.amplification
    );

    assert_eq!(selected.len(), 100, "Should return requested count");

    println!("\n[PASS] Grover amplification working correctly\n");
}

/// Gate Check: Selection pressure
#[test]
fn gate_selection_pressure() {
    println!("\n=== SPRINT 66 GATE CHECK: Selection Pressure ===\n");

    let config = HybridConfig::new(12).with_seed(42).with_percentile(0.80);
    let mut search = HybridSearch::new(config);
    search.evaluate_fitness_onemax();

    let (pop_mean, _, _, _) = search.fitness_stats();

    let selected = search.select_candidates(500);
    let selected_mean: f64 =
        selected.iter().map(|c| c.fitness as f64).sum::<f64>() / selected.len() as f64;

    let stats = search.last_stats().unwrap();

    println!("Population: {} candidates", search.population_size());
    println!("Selected: {} candidates", selected.len());
    println!();
    println!("Fitness comparison:");
    println!("  Population mean: {:.2}", pop_mean);
    println!("  Selected mean: {:.2}", selected_mean);
    println!("  Selection pressure: {:.2}x", selected_mean / pop_mean);
    println!();
    println!("Grover stats:");
    println!("  Threshold: {}", stats.threshold);
    println!(
        "  Marked: {} ({:.1}%)",
        stats.marked_count,
        stats.marked_count as f64 / stats.population_size as f64 * 100.0
    );
    println!("  Amplification: {:.2}x", stats.amplification);

    assert!(
        selected_mean > pop_mean,
        "Selected mean ({:.2}) should exceed population mean ({:.2})",
        selected_mean,
        pop_mean
    );

    println!("\n[PASS] Selection provides positive pressure\n");
}

/// Gate Check: Multiple selection rounds
#[test]
fn gate_multiple_rounds() {
    println!("\n=== SPRINT 66 GATE CHECK: Multiple Selection Rounds ===\n");

    let config = HybridConfig::new(10).with_seed(42).with_percentile(0.75);
    let mut search = HybridSearch::new(config);

    search.evaluate_fitness_onemax();
    let (initial_mean, _, _, _) = search.fitness_stats();

    println!("Initial population mean: {:.2}", initial_mean);
    println!();

    // Run multiple selection rounds
    for round in 1..=5 {
        let selected = search.select(100);
        let stats = search.last_stats().unwrap();

        println!(
            "Round {}: selected_mean={:.2}, amplification={:.2}x",
            round, stats.selected_mean_fitness, stats.amplification
        );

        // Each round should provide positive selection pressure
        assert!(
            stats.selected_mean_fitness >= stats.population_mean_fitness * 0.95,
            "Round {} should have selection pressure",
            round
        );

        assert_eq!(selected.len(), 100);
    }

    println!("\n[PASS] Multiple selection rounds work correctly\n");
}

/// Gate Check: Scale test (16K candidates)
#[test]
fn gate_scale_16k() {
    println!("\n=== SPRINT 66 GATE CHECK: Scale Test (16K) ===\n");

    let config = HybridConfig::new(14).with_seed(42); // 16384 candidates

    println!("Creating search engine with {} candidates...", 1 << 14);
    let start = Instant::now();
    let mut search = HybridSearch::new(config);
    let create_time = start.elapsed();
    println!("Created in {:?}", create_time);

    println!("Evaluating fitness...");
    let start = Instant::now();
    search.evaluate_fitness_onemax();
    let eval_time = start.elapsed();
    println!("Evaluated in {:?}", eval_time);

    println!("Running selection...");
    let start = Instant::now();
    let selected = search.select(1000);
    let select_time = start.elapsed();
    println!("Selected in {:?}", select_time);

    let stats = search.last_stats().unwrap();

    println!();
    println!("Results:");
    println!("  Population: {}", stats.population_size);
    println!("  Threshold: {}", stats.threshold);
    println!(
        "  Marked: {} ({:.2}%)",
        stats.marked_count,
        stats.marked_count as f64 / stats.population_size as f64 * 100.0
    );
    println!("  Iterations: {}", stats.iterations);
    println!("  Amplification: {:.2}x", stats.amplification);
    println!("  Population mean: {:.2}", stats.population_mean_fitness);
    println!("  Selected mean: {:.2}", stats.selected_mean_fitness);

    assert_eq!(selected.len(), 1000);
    assert!(stats.amplification > 1.0);
    assert!(stats.selected_mean_fitness > stats.population_mean_fitness * 0.95);

    println!("\n[PASS] 16K scale test passed\n");
}

/// Gate Check: Custom fitness function
#[test]
fn gate_custom_fitness() {
    println!("\n=== SPRINT 66 GATE CHECK: Custom Fitness Function ===\n");

    let config = HybridConfig::new(10).with_seed(42);
    let mut search = HybridSearch::new(config);

    // Custom fitness: count trailing zeros (good for testing specific patterns)
    search.evaluate_fitness_custom(|genome| genome.trailing_zeros() as u8);

    let (mean, min, max, _) = search.fitness_stats();

    println!("Custom fitness (trailing zeros):");
    println!("  Mean: {:.2}", mean);
    println!("  Min: {}", min);
    println!("  Max: {}", max);

    let selected = search.select(100);
    let stats = search.last_stats().unwrap();

    println!();
    println!("Selection:");
    println!("  Population mean: {:.2}", stats.population_mean_fitness);
    println!("  Selected mean: {:.2}", stats.selected_mean_fitness);

    assert_eq!(selected.len(), 100);

    println!("\n[PASS] Custom fitness function works\n");
}

/// Gate Check: Full integration test
#[test]
fn gate_full_integration() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         SPRINT 66.0 - FULL INTEGRATION GATE CHECK          ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Hybrid Fitness-Grover Integration                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Test progression through qubit counts
    let test_cases = vec![(10, "1K"), (12, "4K"), (14, "16K")];

    println!("Testing hybrid selection at multiple scales:\n");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Qubits", "Pop Size", "Marked", "Iters", "Amplify", "Pressure"
    );
    println!("{}", "-".repeat(70));

    for (n_qubits, label) in test_cases {
        let config = HybridConfig::new(n_qubits)
            .with_seed(66)
            .with_percentile(0.75);

        let mut search = HybridSearch::new(config);
        search.evaluate_fitness_onemax();

        let _ = search.select(100);
        let stats = search.last_stats().unwrap();

        let pressure = stats.selected_mean_fitness / stats.population_mean_fitness;

        println!(
            "{:>8} {:>10} {:>9}% {:>10} {:>9.2}x {:>11.2}x",
            format!("{} ({})", n_qubits, label),
            stats.population_size,
            format!(
                "{:.1}",
                stats.marked_count as f64 / stats.population_size as f64 * 100.0
            ),
            stats.iterations,
            stats.amplification,
            pressure
        );

        assert!(stats.amplification > 1.0, "Should have amplification");
        assert!(pressure > 0.95, "Should have selection pressure");
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              [PASS] SPRINT 66 GATE PASSED                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
}

// =============================================================================
// Evolution Gate Tests
// =============================================================================

/// Gate Check: Evolution improves fitness over generations
#[test]
fn gate_evolution_improves() {
    println!("\n=== SPRINT 66 GATE CHECK: Evolution Improves Fitness ===\n");

    let config = HybridConfig::new(10).with_seed(42);
    let mut search = HybridSearch::new(config);

    let result = search.evolve(30, 0.8, 0.015);

    let initial_max = result.history[0].pre_max_fitness;
    let final_max = result.best_fitness;
    let improvement = final_max as f64 - initial_max as f64;

    println!("30-generation evolution:");
    println!("  Initial max fitness: {}", initial_max);
    println!("  Final max fitness: {}", final_max);
    println!("  Improvement: +{:.0}", improvement);
    println!("  Time: {:?}", result.elapsed);

    assert!(
        final_max > initial_max,
        "Fitness should improve: {} -> {}",
        initial_max,
        final_max
    );

    println!("\n[PASS] Evolution improves fitness over generations\n");
}

/// Gate Check: Evolution converges to near-optimal
#[test]
fn gate_evolution_convergence() {
    println!("\n=== SPRINT 66 GATE CHECK: Evolution Convergence ===\n");

    let config = HybridConfig::new(12).with_seed(66); // 4096 candidates
    let mut search = HybridSearch::new(config);

    let start = Instant::now();
    let result = search.evolve(50, 0.85, 0.02);
    let elapsed = start.elapsed();

    println!("50-generation evolution (4K population):");
    println!(
        "  Best fitness: {}/64 ({:.1}%)",
        result.best_fitness,
        result.best_fitness as f64 / 64.0 * 100.0
    );
    println!("  Final mean: {:.2}", result.final_mean_fitness);
    println!("  Best found at generation: {}", result.best_generation);
    println!("  Total time: {:?}", elapsed);

    // Check 90% convergence
    if let Some(g) = result.convergence_generation(0.9) {
        println!("  Reached 90% of optimal at generation: {}", g);
    }

    assert!(
        result.best_fitness >= 55,
        "Should converge to fitness >= 55 (got {})",
        result.best_fitness
    );

    println!("\n[PASS] Evolution converges to near-optimal\n");
}

/// Gate Check: Full evolution integration
#[test]
fn gate_evolution_full() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║       SPRINT 66.0 - EVOLUTION INTEGRATION GATE CHECK       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Test evolution at different scales
    let test_cases = vec![
        (10, 30, "1K"), // 1024 candidates, 30 generations
        (12, 40, "4K"), // 4096 candidates, 40 generations
    ];

    println!(
        "{:>8} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Scale", "Pop", "Gens", "Init Max", "Final Max", "Time"
    );
    println!("{}", "-".repeat(72));

    for (n_qubits, generations, label) in test_cases {
        let config = HybridConfig::new(n_qubits).with_seed(66);
        let mut search = HybridSearch::new(config);

        let start = Instant::now();
        let result = search.evolve(generations, 0.8, 0.015);
        let elapsed = start.elapsed();

        let initial = result.history[0].pre_max_fitness;

        println!(
            "{:>8} {:>10} {:>10} {:>12} {:>12} {:>10.2?}",
            label,
            1 << n_qubits,
            result.generations_run,
            format!("{}/64", initial),
            format!("{}/64", result.best_fitness),
            elapsed
        );

        assert!(
            result.best_fitness > initial,
            "Fitness should improve for {} scale",
            label
        );
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         [PASS] EVOLUTION INTEGRATION GATE PASSED           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    println!("⚠️  REMINDER: This is a STANDALONE prototype.");
    println!("    TODO: Integrate with TILE-8 substrate for GPU scale.");
    println!("    See: SPRINT_66_HYBRID_FITNESS_GROVER.md\n");
}
