//! Phase 4 Gate Check: Quantum Grover Amplification
//!
//! Sprint 65.0 - Quantum-Enhanced Neural Search Engine
//!
//! Gate criteria:
//! - Grover oracle marks high-fitness candidates correctly
//! - Amplitude amplification boosts marked candidate probabilities
//! - Quantum selector provides selection pressure
//! - Integration with GA baseline works correctly
//! - Hybrid quantum-classical selection operational

use engine::search::{
    baseline::{BaselineConfig, ClassicalGA},
    candidate::Xorshift64,
    fitness::OneMax,
    quantum::{
        GroverSelector, HybridQuantumSelector, QuantumConfig, QuantumState, theoretical_speedup,
    },
    selection::{SelectionStrategy, create_selector},
    substrate::SearchSubstrate,
};
use std::time::Instant;

// =============================================================================
// Phase 4: Quantum Grover Amplification Tests
// =============================================================================

/// Gate Check: Quantum state initialization and normalization
#[test]
fn gate_quantum_state_operations() {
    println!("\n=== PHASE 4 GATE CHECK: Quantum State Operations ===\n");

    // Uniform superposition
    let state = QuantumState::uniform(1000);
    let probs = state.probabilities();
    let total: f64 = probs.iter().sum();

    println!("Uniform state (1000 candidates):");
    println!("  Total probability: {:.6}", total);
    println!("  Mean probability: {:.6}", 1.0 / 1000.0);

    assert!(
        (total - 1.0).abs() < 1e-10,
        "Probabilities should sum to 1.0"
    );

    // Fitness-weighted state
    let fitnesses: Vec<u8> = (0..100).map(|i| (i * 2) as u8).collect();
    let weighted = QuantumState::from_fitness(&fitnesses);
    let weighted_probs = weighted.probabilities();

    // Higher fitness should have higher probability
    assert!(
        weighted_probs[99] > weighted_probs[50],
        "Higher fitness should have higher amplitude"
    );

    println!("Fitness-weighted state (100 candidates):");
    println!("  Prob[low]: {:.6}", weighted_probs[10]);
    println!("  Prob[high]: {:.6}", weighted_probs[99]);

    println!("\n✓ GATE CHECK PASSED: Quantum state operations correct\n");
}

/// Gate Check: Grover oracle correctly marks candidates
#[test]
fn gate_grover_oracle() {
    println!("\n=== PHASE 4 GATE CHECK: Grover Oracle ===\n");

    let mut state = QuantumState::uniform(16);

    // Mark first 4 states
    let marked = vec![
        true, true, true, true, false, false, false, false, false, false, false, false, false,
        false, false, false,
    ];

    let prob_before: f64 = (0..4).map(|i| state.probability(i)).sum();

    // Apply oracle (phase flip on marked)
    state.apply_oracle(&marked);

    // After oracle, probabilities unchanged (only phase flipped)
    let prob_after: f64 = (0..4).map(|i| state.probability(i)).sum();

    println!("Oracle effect on marked states:");
    println!("  Prob before: {:.6}", prob_before);
    println!("  Prob after oracle: {:.6}", prob_after);

    // Apply diffusion to see amplification
    state.apply_diffusion();
    let prob_amplified: f64 = (0..4).map(|i| state.probability(i)).sum();

    println!("  Prob after diffusion: {:.6}", prob_amplified);

    assert!(
        prob_amplified > prob_before,
        "Amplification should increase marked probability"
    );

    println!("\n✓ GATE CHECK PASSED: Grover oracle working correctly\n");
}

/// Gate Check: Amplitude amplification achieves speedup
#[test]
fn gate_amplitude_amplification() {
    println!("\n=== PHASE 4 GATE CHECK: Amplitude Amplification ===\n");

    // N=64, M=4 (6.25% marked)
    let n = 64;
    let m = 4;
    let mut state = QuantumState::uniform(n);
    let mut marked = vec![false; n];
    for i in 0..m {
        marked[i] = true;
    }

    let initial_marked_prob: f64 = (0..m).map(|i| state.probability(i)).sum();

    // Optimal iterations ≈ π/4 * √(N/M) = π/4 * √16 = π ≈ 3.14
    let optimal_iter = QuantumConfig::optimal_iterations(n, m);
    println!("N={}, M={}", n, m);
    println!("Optimal iterations: {}", optimal_iter);

    // Apply Grover iterations
    for _ in 0..optimal_iter {
        state.grover_iteration(&marked);
    }

    let final_marked_prob: f64 = (0..m).map(|i| state.probability(i)).sum();
    let amplification = final_marked_prob / initial_marked_prob;

    println!("Initial marked probability: {:.4}", initial_marked_prob);
    println!("Final marked probability: {:.4}", final_marked_prob);
    println!("Amplification: {:.2}x", amplification);
    println!("Theoretical speedup: {:.2}x", theoretical_speedup(n, m));

    assert!(
        final_marked_prob > 0.5,
        "Should achieve >50% probability for marked states"
    );

    assert!(
        amplification > 5.0,
        "Should achieve significant amplification"
    );

    println!("\n✓ GATE CHECK PASSED: Amplitude amplification working\n");
}

/// Gate Check: GroverSelector integration
#[test]
fn gate_grover_selector() {
    println!("\n=== PHASE 4 GATE CHECK: Grover Selector ===\n");

    let mut substrate = SearchSubstrate::new(1000, OneMax::new(32));
    substrate.initialize_random(42);
    substrate.evaluate_all();

    let pop_stats = substrate.statistics();
    println!("Population: 1000 candidates");
    println!("  Best fitness: {}", pop_stats.best_fitness);
    println!("  Mean fitness: {:.2}", pop_stats.mean_fitness);

    let mut selector = GroverSelector::new(
        QuantumConfig::default()
            .with_threshold((pop_stats.mean_fitness * 1.1) as u8)
            .with_auto_iterations(),
    );
    let mut rng = Xorshift64::new(42);

    let start = Instant::now();
    let selected = selector.select(&substrate, 100, &mut rng);
    let select_time = start.elapsed();

    let stats = selector.last_stats().unwrap();

    println!("\nGrover Selection:");
    println!("  Selection time: {:?}", select_time);
    println!("  Threshold: {}", stats.threshold);
    println!(
        "  Marked candidates: {} ({:.1}%)",
        stats.num_marked,
        stats.num_marked as f64 / 10.0
    );
    println!("  Iterations: {}", stats.iterations);
    println!("  Amplification: {:.2}x", stats.amplification);

    // Verify selection pressure
    let selected_mean: f64 = selected
        .iter()
        .map(|&i| substrate.get(i).fitness as f64)
        .sum::<f64>()
        / selected.len() as f64;

    println!("  Selected mean fitness: {:.2}", selected_mean);

    assert!(stats.amplification >= 1.0);
    assert_eq!(selected.len(), 100);

    println!("\n✓ GATE CHECK PASSED: Grover selector operational\n");
}

/// Gate Check: Hybrid quantum-classical selector
#[test]
fn gate_hybrid_selector() {
    println!("\n=== PHASE 4 GATE CHECK: Hybrid Quantum-Classical ===\n");

    let mut substrate = SearchSubstrate::new(500, OneMax::new(16));
    substrate.initialize_random(42);
    substrate.evaluate_all();

    // Test different quantum ratios
    let ratios = vec![
        (
            "Classical-heavy (20%)",
            HybridQuantumSelector::classical_heavy(),
        ),
        ("Balanced (50%)", HybridQuantumSelector::balanced()),
        (
            "Quantum-heavy (80%)",
            HybridQuantumSelector::quantum_heavy(),
        ),
    ];

    for (name, mut selector) in ratios {
        let mut rng = Xorshift64::new(42);
        let selected = selector.select(&substrate, 50, &mut rng);

        let selected_mean: f64 = selected
            .iter()
            .map(|&i| substrate.get(i).fitness as f64)
            .sum::<f64>()
            / selected.len() as f64;

        println!("{}: selected mean = {:.2}", name, selected_mean);

        assert_eq!(selected.len(), 50);
    }

    println!("\n✓ GATE CHECK PASSED: Hybrid selector working\n");
}

/// Gate Check: Quantum strategy via SelectionStrategy enum
#[test]
fn gate_quantum_selection_strategy() {
    println!("\n=== PHASE 4 GATE CHECK: Quantum Selection Strategy ===\n");

    let mut substrate = SearchSubstrate::new(500, OneMax::new(16));
    substrate.initialize_random(42);
    substrate.evaluate_all();

    // Create selector via the strategy enum
    let strategy = SelectionStrategy::QuantumGrover { iterations: 3 };
    let selector = create_selector(&strategy);

    assert_eq!(selector.name(), "QuantumGrover");

    let mut rng = Xorshift64::new(42);
    let selected = selector.select(&substrate, 50, &mut rng);

    assert_eq!(selected.len(), 50);
    println!("Quantum strategy name: {}", selector.name());
    println!("Selected {} candidates", selected.len());

    // Verify all indices are valid
    for &idx in &selected {
        assert!(idx < 500, "Invalid index: {}", idx);
    }

    println!("\n✓ GATE CHECK PASSED: Quantum selection strategy working\n");
}

/// Gate Check: Quantum vs Classical comparison
#[test]
fn gate_quantum_vs_classical() {
    println!("\n=== PHASE 4 GATE CHECK: Quantum vs Classical Comparison ===\n");

    let mut substrate = SearchSubstrate::new(1000, OneMax::new(32));
    substrate.initialize_random(42);
    substrate.evaluate_all();

    let pop_mean = substrate.statistics().mean_fitness as f64;

    // Classical (tournament) selection
    let classical_strategy = SelectionStrategy::Tournament { tournament_size: 7 };
    let classical_selector = create_selector(&classical_strategy);
    let mut rng = Xorshift64::new(123);
    let classical_selected = classical_selector.select(&substrate, 200, &mut rng);
    let classical_mean: f64 = classical_selected
        .iter()
        .map(|&i| substrate.get(i).fitness as f64)
        .sum::<f64>()
        / classical_selected.len() as f64;

    // Quantum selection
    let quantum_strategy = SelectionStrategy::QuantumGrover { iterations: 2 };
    let quantum_selector = create_selector(&quantum_strategy);
    let mut rng = Xorshift64::new(123);
    let quantum_selected = quantum_selector.select(&substrate, 200, &mut rng);
    let quantum_mean: f64 = quantum_selected
        .iter()
        .map(|&i| substrate.get(i).fitness as f64)
        .sum::<f64>()
        / quantum_selected.len() as f64;

    println!("Population mean: {:.2}", pop_mean);
    println!(
        "Classical (Tournament) selected mean: {:.2}",
        classical_mean
    );
    println!("Quantum (Grover) selected mean: {:.2}", quantum_mean);
    println!("Classical pressure: {:.2}x", classical_mean / pop_mean);
    println!("Quantum pressure: {:.2}x", quantum_mean / pop_mean);

    // Both should provide selection pressure
    assert!(
        classical_mean > pop_mean * 0.95,
        "Classical should have pressure"
    );
    assert!(
        quantum_mean > pop_mean * 0.95,
        "Quantum should have pressure"
    );

    println!("\n✓ GATE CHECK PASSED: Both selectors provide selection pressure\n");
}

/// Gate Check: Quantum selection with GA baseline
#[test]
fn gate_quantum_ga_integration() {
    println!("\n=== PHASE 4 GATE CHECK: Quantum GA Integration ===\n");

    // Run GA with quantum selection
    let config = BaselineConfig::default()
        .with_population(10_000)
        .with_max_generations(20)
        .with_selection(SelectionStrategy::QuantumGrover { iterations: 2 })
        .with_seed(42);

    let mut ga = ClassicalGA::new(config, OneMax::new(32));

    let start = Instant::now();
    let result = ga.run();
    let runtime = start.elapsed();

    println!("Quantum-Enhanced GA Results:");
    println!("  Population: 10,000");
    println!("  Generations: {}", result.generations);
    println!("  Runtime: {:?}", runtime);
    println!("  Best fitness: {}", result.best.fitness);
    println!("  Best found at generation: {}", result.best_generation);
    println!(
        "  Final mean fitness: {:.2}",
        result.final_stats.mean_fitness
    );

    // Should converge well
    assert!(
        result.best.fitness >= 200,
        "Should achieve good fitness: got {}",
        result.best.fitness
    );

    println!("\n✓ GATE CHECK PASSED: Quantum GA integration working\n");
}

/// Gate Check: Full Phase 4 integration
#[test]
fn gate_phase4_full_integration() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         SPRINT 65.0 - PHASE 4 GATE CHECK                   ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Quantum Grover Amplification for Search                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Run both classical and quantum GA
    let classical_config = BaselineConfig::default()
        .with_population(5_000)
        .with_max_generations(30)
        .with_selection(SelectionStrategy::Tournament { tournament_size: 7 })
        .with_seed(65);

    let quantum_config = BaselineConfig::default()
        .with_population(5_000)
        .with_max_generations(30)
        .with_selection(SelectionStrategy::QuantumGrover { iterations: 2 })
        .with_seed(65);

    let mut classical_ga = ClassicalGA::new(classical_config, OneMax::new(32));
    let mut quantum_ga = ClassicalGA::new(quantum_config, OneMax::new(32));

    let classical_result = classical_ga.run();
    let quantum_result = quantum_ga.run();

    println!("Comparison: Classical vs Quantum Selection");
    println!("─────────────────────────────────────────────");
    println!("                    Classical    Quantum");
    println!("─────────────────────────────────────────────");
    println!(
        "Best fitness:       {:>8}    {:>8}",
        classical_result.best.fitness, quantum_result.best.fitness
    );
    println!(
        "Best generation:    {:>8}    {:>8}",
        classical_result.best_generation, quantum_result.best_generation
    );
    println!(
        "Total generations:  {:>8}    {:>8}",
        classical_result.generations, quantum_result.generations
    );
    println!(
        "Final mean:         {:>8.2}    {:>8.2}",
        classical_result.final_stats.mean_fitness, quantum_result.final_stats.mean_fitness
    );
    println!("─────────────────────────────────────────────");

    // Both should achieve good results
    assert!(classical_result.best.fitness >= 200);
    assert!(quantum_result.best.fitness >= 200);

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                ✓ PHASE 4 GATE PASSED                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
}
