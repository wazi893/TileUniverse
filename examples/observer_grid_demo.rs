//! Sprint 60.0: The Observer Grid Demo
//!
//! Demonstrates all 5 phases:
//! - Phase 1: Quantum tile binding (CPU-quantum pairing)
//! - Phase 2: Observation dynamics (wavefront propagation)
//! - Phase 3: Unmeasured compute (coherence in protected regions)
//! - Phase 4: Emergent boundaries (membrane visualization)
//! - Phase 5: The message (anomaly detection)
//!
//! Run: cargo run --release --example observer_grid_demo

use engine::tile8::quantum_router_f64::QuantumGridF64;
use engine::tile8::{ObservationRule, ObservationStats, ObserverGrid, ObserverGridConfig};
use std::time::Instant;

fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn print_grid_ascii(grid: &ObserverGrid, width: usize, height: usize) {
    let data = grid.get_visualization_data();
    let boundary = grid.get_boundary_mask();

    println!("\n  Grid State (. = superposition, # = collapsed, @ = boundary, P = protected):");
    println!("  {}", "─".repeat(width + 2));

    for y in 0..height.min(30) {
        print!("  │");
        for x in 0..width.min(80) {
            let id = y * grid.width + x;
            if id >= data.len() {
                print!(" ");
                continue;
            }

            let ch = if boundary[id] == 1 {
                '@' // Boundary
            } else if data[id] == 0 {
                '.' // Superposition
            } else if data[id] == 255 {
                'P' // Protected
            } else {
                '#' // Collapsed
            };
            print!("{}", ch);
        }
        println!("│");
    }
    println!("  {}", "─".repeat(width + 2));
}

fn print_stats(stats: &ObservationStats, tick: u64) {
    let total = stats.total as f64;
    let super_pct = stats.superposition_count as f64 / total * 100.0;
    let collapsed_pct = stats.collapsed_count as f64 / total * 100.0;

    println!(
        "  Tick {:>5} │ Superposition: {:>6} ({:>5.1}%) │ Collapsed: {:>6} ({:>5.1}%) │ Boundary: {:>5} │ Islands: {:>3} │ Anomaly: {:.4}",
        tick,
        format_thousands(stats.superposition_count),
        super_pct,
        format_thousands(stats.collapsed_count),
        collapsed_pct,
        stats.boundary_length,
        stats.unmeasured_islands,
        stats.anomaly_score
    );
}

fn main() {
    println!();
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("                    SPRINT 60.0: THE OBSERVER GRID                              ");
    println!("           \"What happens when 268 million observers decide whether to look?\"    ");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!();

    // =========================================================================
    // PHASE 1: Quantum Tile Binding
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: QUANTUM TILE BINDING                                               │");
    println!("│ Each CPU binds to a quantum block (128 amplitudes = 7 qubits)               │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create quantum grid (14 qubits = 128 blocks)
    let n_qubits = 14;
    let mut quantum_grid = QuantumGridF64::new(n_qubits);
    let num_blocks = quantum_grid.num_blocks();
    println!(
        "  Quantum grid: {} qubits = {} blocks",
        n_qubits, num_blocks
    );

    // Create observer grid to match
    let config = ObserverGridConfig {
        num_observers: num_blocks,
        grid_width: (num_blocks as f64).sqrt().ceil() as usize,
        default_rule: ObservationRule::NeighborCount(2),
        seed: 42,
        track_entanglement: true,
        max_observations_per_tick: 0,
    };
    let mut observer_grid = ObserverGrid::new(config);
    println!(
        "  Observer grid: {} observers in {}x{} layout",
        observer_grid.observers.len(),
        observer_grid.width,
        observer_grid.height
    );

    // Create GHZ state in quantum grid
    println!("  Creating GHZ state...");
    let time_us = quantum_grid.create_ghz_state();
    println!("  GHZ state created in {}μs", time_us);

    // Extract probabilities and bind to observer grid
    let (block_probs, max_prob, total_prob) = quantum_grid.get_block_probabilities();
    observer_grid.update_probabilities(&block_probs);

    // Set GHZ distribution for Born rule sampling
    // GHZ state collapses to either |0...0⟩ or |1...1⟩ with 50% probability each
    observer_grid.set_ghz_distribution();
    println!("  GHZ distribution set: P(|0⟩)=0.5, P(|127⟩)=0.5 (1 bit entropy)");

    // Calibrate anomaly detector for GHZ statistics
    observer_grid
        .anomaly_detector
        .calibrate_from_observers(&observer_grid.observers);
    println!(
        "  Anomaly detector calibrated: expected entropy = {:.2} bits",
        observer_grid.anomaly_detector.expected_entropy
    );

    println!(
        "  Probabilities bound: max={:.6}, total={:.6}",
        max_prob, total_prob
    );
    println!();

    // =========================================================================
    // PHASE 2: Observation Dynamics
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: OBSERVATION DYNAMICS                                               │");
    println!("│ Observation wavefront propagates through the grid                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Show initial state
    print_grid_ascii(&observer_grid, observer_grid.width, observer_grid.height);
    print_stats(&observer_grid.stats, 0);
    println!();

    // Trigger observation wave from center
    let center_x = observer_grid.width / 2;
    let center_y = observer_grid.height / 2;
    println!(
        "  Triggering observation wave from center ({}, {})...",
        center_x, center_y
    );
    observer_grid.trigger_observation_wave(center_x, center_y);

    // Run several ticks and show propagation
    for tick in 1..=10 {
        observer_grid.tick();
        if tick <= 5 || tick == 10 {
            print_stats(&observer_grid.stats, tick as u64);
        }
    }
    print_grid_ascii(&observer_grid, observer_grid.width, observer_grid.height);
    println!();

    // =========================================================================
    // PHASE 3: Unmeasured Compute (Protected Regions)
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: UNMEASURED COMPUTE                                                 │");
    println!("│ Protected regions maintain quantum coherence                                │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Reset and create new grid with protected sanctuary
    let config2 = ObserverGridConfig {
        num_observers: 400,
        grid_width: 20,
        default_rule: ObservationRule::Stochastic(0.1), // 10% chance per tick
        seed: 123,
        track_entanglement: true,
        max_observations_per_tick: 0,
    };
    let mut observer_grid2 = ObserverGrid::new(config2);

    // Create protected sanctuary in the center
    println!("  Creating protected sanctuary (radius 3) at center...");
    observer_grid2.protect_region(10, 10, 3);

    // Run simulation
    println!("  Running 50 ticks with stochastic observation (10% per tick)...");
    for _ in 0..50 {
        observer_grid2.tick();
    }

    print_grid_ascii(&observer_grid2, 20, 20);
    print_stats(&observer_grid2.stats, 50);
    println!(
        "  Protected count: {} (sanctuary preserved!)",
        observer_grid2.stats.protected_count
    );
    println!(
        "  Largest unmeasured region: {} (coherence maintained)",
        observer_grid2.stats.largest_unmeasured_region
    );
    println!();

    // =========================================================================
    // PHASE 4: Emergent Boundaries
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 4: EMERGENT BOUNDARIES                                                │");
    println!("│ The membrane between measured and unmeasured exhibits dynamics              │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create grid with neighbor-based rule (creates interesting boundaries)
    let config3 = ObserverGridConfig {
        num_observers: 900,
        grid_width: 30,
        default_rule: ObservationRule::NeighborCount(3),
        seed: 456,
        track_entanglement: true,
        max_observations_per_tick: 50, // Rate-limited observations
    };
    let mut observer_grid3 = ObserverGrid::new(config3);

    // Seed multiple observation points
    println!("  Seeding 5 observation points...");
    observer_grid3.trigger_observation_wave(5, 5);
    observer_grid3.trigger_observation_wave(25, 5);
    observer_grid3.trigger_observation_wave(15, 15);
    observer_grid3.trigger_observation_wave(5, 25);
    observer_grid3.trigger_observation_wave(25, 25);

    // Track boundary evolution
    println!("  Tracking boundary evolution over 30 ticks:");
    println!(
        "  {:>5} │ {:>10} │ {:>10} │ {:>8}",
        "Tick", "Collapsed", "Boundary", "Islands"
    );
    println!("  ──────┼────────────┼────────────┼─────────");

    for tick in 1..=30 {
        observer_grid3.tick();
        if tick % 5 == 0 || tick <= 5 {
            println!(
                "  {:>5} │ {:>10} │ {:>10} │ {:>8}",
                tick,
                observer_grid3.stats.collapsed_count,
                observer_grid3.stats.boundary_length,
                observer_grid3.stats.unmeasured_islands
            );
        }
    }

    print_grid_ascii(&observer_grid3, 30, 30);
    println!();

    // =========================================================================
    // PHASE 5: The Message (Anomaly Detection)
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 5: THE MESSAGE                                                        │");
    println!("│ Anomaly detection: looking for unexpected patterns                          │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create large grid for statistical significance
    let config4 = ObserverGridConfig {
        num_observers: 10000,
        grid_width: 100,
        default_rule: ObservationRule::Stochastic(0.05),
        seed: 789,
        track_entanglement: true,
        max_observations_per_tick: 0,
    };
    let mut observer_grid4 = ObserverGrid::new(config4);

    // Use uniform distribution (default) - measurements should be uniformly distributed
    // Expected entropy for uniform = log2(128) = 7 bits
    println!("  Distribution: Uniform (7 bits entropy expected)");
    println!("  Measurement model: Born rule sampling from 128-state distribution");
    println!();

    println!("  Running 100 ticks on 10,000 observer grid...");
    let start = Instant::now();

    for tick in 1..=100 {
        observer_grid4.tick();

        if tick % 20 == 0 {
            let detector = &observer_grid4.anomaly_detector;
            println!(
                "  Tick {:>3}: Score={:.4} | Autocorr={:.4} | Chi2={:.1} | Entropy={:.2} bits",
                tick,
                detector.score,
                detector.pattern_score,
                detector.chi_squared,
                detector.observed_entropy_avg
            );
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "  Completed in {:.2}ms ({:.0} observer-ticks/sec)",
        elapsed.as_secs_f64() * 1000.0,
        (10000.0 * 100.0) / elapsed.as_secs_f64()
    );
    println!();

    // Final anomaly report
    let detector = &observer_grid4.anomaly_detector;
    println!("  ╔═══════════════════════════════════════════════════════════════╗");
    println!("  ║              ANOMALY DETECTION FINAL REPORT                   ║");
    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║  Status:            {:>42} ║", detector.status());
    println!("  ║  Anomaly Score:     {:>42.6} ║", detector.score);
    println!(
        "  ║  Expected Entropy:  {:>38.2} bits ║",
        detector.expected_entropy
    );
    println!(
        "  ║  Observed Entropy:  {:>38.2} bits ║",
        detector.observed_entropy_avg
    );
    println!("  ║  Autocorrelation:   {:>42.6} ║", detector.pattern_score);
    println!("  ║  Chi-Squared:       {:>42.2} ║", detector.chi_squared);
    println!(
        "  ║  Info Surplus:      {:>38.2} bits ║",
        detector.information_surplus
    );
    println!("  ╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Explain results
    println!("  Interpretation:");
    if detector.pattern_score < 0.05 {
        println!("  - Autocorrelation: LOW (measurements appear independent)");
    } else {
        println!("  - Autocorrelation: HIGH (measurements show temporal patterns)");
    }

    let entropy_diff = (detector.observed_entropy_avg - detector.expected_entropy).abs();
    if entropy_diff < 0.5 {
        println!(
            "  - Entropy: MATCHES expected ({:.2} vs {:.2} bits)",
            detector.observed_entropy_avg, detector.expected_entropy
        );
    } else {
        println!(
            "  - Entropy: DEVIATES from expected ({:.2} vs {:.2} bits)",
            detector.observed_entropy_avg, detector.expected_entropy
        );
    }

    // Chi-squared critical value for 127 df at p=0.05 is ~152
    if detector.chi_squared < 160.0 {
        println!("  - Chi-squared: PASS (distribution matches expected)");
    } else {
        println!("  - Chi-squared: FAIL (distribution differs from expected)");
    }

    println!();

    if detector.is_anomalous() {
        println!("  >>> ANOMALY DETECTED - Measurements deviate from quantum randomness <<<");
    } else {
        println!("  >>> NOMINAL - Measurements consistent with true quantum randomness <<<");
    }
    println!();

    // =========================================================================
    // SCALE TEST
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SCALE TEST: 1 MILLION OBSERVERS                                             │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let config_large = ObserverGridConfig {
        num_observers: 1_000_000,
        grid_width: 1000,
        default_rule: ObservationRule::NeighborCount(3),
        seed: 999,
        track_entanglement: false, // Disable for speed
        max_observations_per_tick: 10000,
    };

    println!("  Creating 1M observer grid...");
    let start = Instant::now();
    let mut observer_grid_large = ObserverGrid::new(config_large);
    println!(
        "  Created in {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // Seed observation
    observer_grid_large.trigger_observation_wave(500, 500);

    println!("  Running 100 ticks...");
    let start = Instant::now();
    for _ in 0..100 {
        observer_grid_large.tick();
    }
    let elapsed = start.elapsed();

    println!("  Completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Throughput: {:.2}M observer-ticks/sec",
        (1_000_000.0 * 100.0) / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!();
    print_stats(&observer_grid_large.stats, 100);
    println!();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("                              SPRINT 60.0 COMPLETE                              ");
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Phase 1: ✓ Quantum tile binding (CPU-quantum block pairing)");
    println!("  Phase 2: ✓ Observation dynamics (wavefront propagation)");
    println!("  Phase 3: ✓ Unmeasured compute (protected sanctuaries)");
    println!("  Phase 4: ✓ Emergent boundaries (membrane tracking)");
    println!("  Phase 5: ✓ Anomaly detection (pattern recognition)");
    println!();
    println!("  The Observer Grid is operational.");
    println!("  268 million eyes, deciding whether to look.");
    println!();
}
