//! Statistical Power Experiment (100 Seeds)
//!
//! Establishes statistically significant quantum advantage (or disproves it) with p < 0.01.
//!
//! Run with: cargo run --example statistical_power --release

use engine::constrained_circuits::{CircuitConstraints, entanglement_fraction};
use engine::quantum_forager::{ForagerRegion, QuantumForagerConfig};
use engine::quantum_vs_classical::ConstrainedPopulation;
use engine::simulation::Simulation;

use std::fs::File;
use std::io::Write;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   STATISTICAL POWER: 100-Seed Quantum vs Classical         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let config = QuantumForagerConfig {
        initial_population: 30,
        initial_energy: 100.0,
        n_qubits: 4,
        initial_circuit_length: 8,
        max_circuit_length: 25,
        movement_cost: 0.5,
        metabolism_cost: 0.25,
        reproduction_threshold: 140.0,
        reproduction_cost: 55.0,
        mutation_rate: 0.85,
        heat_energy_factor: 0.7,
        charge_energy_factor: 0.3,
        consume_resources: true,
        max_population: 150,
        region: ForagerRegion {
            x0: 30,
            y0: 30,
            width: 100,
            height: 100,
        },
    };

    let ticks = 1000;
    let heat_patches = 19; // Slightly more challenging
    let heat_value = 35; // Slightly more challenging

    println!("Configuration:");
    println!("  - Seeds: 1-100");
    println!("  - Ticks: {}", ticks);
    println!("  - Qubits: {}", config.n_qubits);
    println!(
        "  - Environment: Moderate scarcity ({} patches, value {})",
        heat_patches, heat_value
    );
    println!();

    let mut results = Vec::with_capacity(100);

    for seed in 1u64..=100 {
        print!("\rRunning seed {:>3}/100...", seed);
        std::io::stdout().flush().unwrap();

        let (q_pop, q_gen, q_extinct, q_qness) = run_population(
            &config,
            CircuitConstraints::full_quantum(),
            ticks,
            heat_patches,
            heat_value,
            seed,
        );

        let (c_pop, c_gen, c_extinct, _) = run_population(
            &config,
            CircuitConstraints::classical_only(),
            ticks,
            heat_patches,
            heat_value,
            seed,
        );

        results.push(SeedResult {
            seed,
            quantum_pop: q_pop as f32,
            classical_pop: c_pop as f32,
            advantage: q_pop as f32 - c_pop as f32,
            quantum_gen: q_gen as f32,
            classical_gen: c_gen as f32,
            quantum_extinct: q_extinct,
            classical_extinct: c_extinct,
            quantum_ness: q_qness,
        });
    }
    println!("\rCompleted 100/100 seeds.    ");
    println!();

    // Compute statistics
    let summary = compute_statistics(&results);

    // Print results
    println!("======================================================================");
    println!("  RESULTS SUMMARY");
    println!("======================================================================\n");

    let q_pops: Vec<f32> = results.iter().map(|r| r.quantum_pop).collect();
    let c_pops: Vec<f32> = results.iter().map(|r| r.classical_pop).collect();

    let q_mean = q_pops.iter().sum::<f32>() / q_pops.len() as f32;
    let c_mean = c_pops.iter().sum::<f32>() / c_pops.len() as f32;
    let q_std = std_dev(&q_pops);
    let c_std = std_dev(&c_pops);
    let q_median = median(&q_pops);
    let c_median = median(&c_pops);

    println!("                        QUANTUM      CLASSICAL");
    println!("────────────────────────────────────────────────");
    println!(
        "Mean Population          {:>6.1}        {:>6.1}",
        q_mean, c_mean
    );
    println!(
        "Std Dev                  {:>6.1}        {:>6.1}",
        q_std, c_std
    );
    println!(
        "Median                   {:>6.0}        {:>6.0}",
        q_median, c_median
    );

    println!("\n======================================================================");
    println!("  WIN/LOSS TALLY");
    println!("======================================================================\n");

    println!(
        "Quantum Wins (Δ > +10):    {:>2} / 100  ({:>5.1}%)",
        summary.quantum_wins, summary.quantum_wins as f32
    );
    println!(
        "Classical Wins (Δ < -10):  {:>2} / 100  ({:>5.1}%)",
        summary.classical_wins, summary.classical_wins as f32
    );
    println!(
        "Ties (|Δ| ≤ 10):           {:>2} / 100  ({:>5.1}%)",
        summary.ties, summary.ties as f32
    );

    println!("\n======================================================================");
    println!("  QUANTUM ADVANTAGE DISTRIBUTION");
    println!("======================================================================\n");

    println!(
        "Mean Advantage:     {:>+6.1} ± {:.1}",
        summary.mean_advantage, summary.std_advantage
    );
    println!("Median Advantage:   {:>+6.1}", summary.median_advantage);
    println!("Min:                {:>+6.1}", summary.min_advantage);
    println!("Max:                {:>+6.1}", summary.max_advantage);
    println!();
    println!(
        "95% Confidence Interval: [{:+.1}, {:+.1}]",
        summary.ci_lower, summary.ci_upper
    );

    println!("\n======================================================================");
    println!("  STATISTICAL TESTS");
    println!("======================================================================\n");

    let significant_t = summary.p_value_t < 0.01;
    let significant_w = summary.p_value_wilcoxon < 0.01;

    println!("Paired t-test:");
    println!("  t-statistic: {:.2}", summary.t_statistic);
    println!("  p-value: {}", format_p_value(summary.p_value_t));
    println!(
        "  Result: {} at α = 0.01",
        if significant_t {
            "SIGNIFICANT"
        } else {
            "NOT SIGNIFICANT"
        }
    );

    println!();
    println!("Wilcoxon signed-rank test:");
    println!("  W-statistic: {:.1}", summary.w_statistic);
    println!("  p-value: {}", format_p_value(summary.p_value_wilcoxon));
    println!(
        "  Result: {} at α = 0.01",
        if significant_w {
            "SIGNIFICANT"
        } else {
            "NOT SIGNIFICANT"
        }
    );

    println!();
    let effect_interpretation = if summary.cohens_d.abs() >= 0.8 {
        "LARGE"
    } else if summary.cohens_d.abs() >= 0.5 {
        "MEDIUM"
    } else if summary.cohens_d.abs() >= 0.2 {
        "SMALL"
    } else {
        "NEGLIGIBLE"
    };
    println!("Effect Size (Cohen's d): {:.2}", summary.cohens_d);
    println!("  Interpretation: {}", effect_interpretation);

    // Histogram
    println!("\n======================================================================");
    println!("  HISTOGRAM: Distribution of Quantum Advantage");
    println!("======================================================================\n");

    print_histogram(&results);

    // Conclusion
    println!("\n======================================================================");
    println!("  CONCLUSION");
    println!("======================================================================\n");

    let confirmed =
        significant_t && significant_w && !ci_includes_zero(summary.ci_lower, summary.ci_upper);

    if confirmed && summary.mean_advantage > 0.0 {
        println!("✅ QUANTUM ADVANTAGE CONFIRMED at p < 0.01\n");
        println!("\"Across 100 independent trials, quantum organisms showed a");
        println!(
            "statistically significant advantage of {:+.1} ± {:.1}",
            summary.mean_advantage, summary.std_advantage
        );
        println!(
            "(95% CI: [{:+.1}, {:+.1}], p < 0.01, Cohen's d = {:.2})",
            summary.ci_lower, summary.ci_upper, summary.cohens_d
        );
        println!("at {} ticks under moderate resource scarcity.\"", ticks);
    } else if confirmed && summary.mean_advantage < 0.0 {
        println!("✅ CLASSICAL ADVANTAGE CONFIRMED at p < 0.01\n");
        println!("\"Across 100 independent trials, classical organisms showed a");
        println!(
            "statistically significant advantage of {:+.1} ± {:.1}\"",
            -summary.mean_advantage, summary.std_advantage
        );
    } else {
        println!("❌ NO SIGNIFICANT ADVANTAGE DETECTED\n");
        println!("\"Across 100 independent trials, no statistically significant");
        println!(
            "quantum advantage was detected (p = {}).",
            format_p_value(summary.p_value_t)
        );
        println!("The observed variance is consistent with random fluctuation.\"");
    }

    // Export CSV
    let csv_path = "100_seed_results.csv";
    if let Ok(mut file) = File::create(csv_path) {
        writeln!(file, "seed,quantum_pop,classical_pop,advantage,quantum_gen,classical_gen,quantum_extinct,classical_extinct,quantumness").unwrap();
        for r in &results {
            writeln!(
                file,
                "{},{:.0},{:.0},{:.1},{:.0},{:.0},{},{},{:.3}",
                r.seed,
                r.quantum_pop,
                r.classical_pop,
                r.advantage,
                r.quantum_gen,
                r.classical_gen,
                r.quantum_extinct,
                r.classical_extinct,
                r.quantum_ness,
            )
            .unwrap();
        }
        println!("\nResults exported to {}", csv_path);
    }
}

#[derive(Debug)]
struct SeedResult {
    seed: u64,
    quantum_pop: f32,
    classical_pop: f32,
    advantage: f32,
    quantum_gen: f32,
    classical_gen: f32,
    quantum_extinct: bool,
    classical_extinct: bool,
    quantum_ness: f32,
}

struct StatisticalSummary {
    mean_advantage: f32,
    std_advantage: f32,
    median_advantage: f32,
    min_advantage: f32,
    max_advantage: f32,
    quantum_wins: usize,
    classical_wins: usize,
    ties: usize,
    t_statistic: f32,
    p_value_t: f32,
    w_statistic: f32,
    p_value_wilcoxon: f32,
    cohens_d: f32,
    ci_lower: f32,
    ci_upper: f32,
}

fn run_population(
    config: &QuantumForagerConfig,
    constraints: CircuitConstraints,
    ticks: u32,
    heat_patches: u32,
    heat_value: u32,
    seed: u64,
) -> (usize, u32, bool, f32) {
    let mut sim = Simulation::new();
    let mut pop = ConstrainedPopulation::new(config.clone(), constraints, seed);

    let region = &config.region;

    // Seed initial resources
    for i in 0..heat_patches {
        let rx = region.x0
            + ((seed.wrapping_mul(0x123456789).wrapping_add(i as u64 * 17)) as u32 % region.width);
        let ry = region.y0
            + ((seed.wrapping_mul(0x987654321).wrapping_add(i as u64 * 31)) as u32 % region.height);
        sim.set_heat_field_for_test(rx as usize, ry as usize, heat_value);
    }

    for tick in 0..ticks {
        // Respawn resources
        if tick % 8 == 0 {
            for i in 0..heat_patches {
                let rseed = seed.wrapping_add(tick as u64);
                let rx = region.x0
                    + ((rseed
                        .wrapping_mul(0x123456789)
                        .wrapping_add(tick as u64 * 7 + i as u64 * 17))
                        as u32
                        % region.width);
                let ry = region.y0
                    + ((rseed
                        .wrapping_mul(0x987654321)
                        .wrapping_add(tick as u64 * 13 + i as u64 * 31))
                        as u32
                        % region.height);
                sim.set_heat_field_for_test(rx as usize, ry as usize, heat_value);
            }
        }

        pop.step(&mut sim);
    }

    let stats = pop.stats();
    let circuits = pop.extract_all_circuits();
    let qness = if circuits.is_empty() {
        0.0
    } else {
        circuits
            .iter()
            .map(|c| entanglement_fraction(c))
            .sum::<f32>()
            / circuits.len() as f32
    };

    (stats.size, stats.max_generation, stats.size == 0, qness)
}

fn compute_statistics(results: &[SeedResult]) -> StatisticalSummary {
    let advantages: Vec<f32> = results.iter().map(|r| r.advantage).collect();
    let n = advantages.len();

    // Mean
    let mean = advantages.iter().sum::<f32>() / n as f32;

    // Std dev
    let variance = advantages.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    let std = variance.sqrt();

    // Median
    let median_val = median(&advantages);

    // Min/Max
    let min_val = advantages.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = advantages.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Win/loss tally
    let quantum_wins = advantages.iter().filter(|&&x| x > 10.0).count();
    let classical_wins = advantages.iter().filter(|&&x| x < -10.0).count();
    let ties = n - quantum_wins - classical_wins;

    // t-statistic: t = mean / (std / sqrt(n))
    let standard_error = std / (n as f32).sqrt();
    let t_statistic = mean / standard_error;

    // p-value (approximate, two-tailed) for t-test
    let p_value_t = estimate_p_value_t(t_statistic.abs(), n);

    // Wilcoxon signed-rank test
    let (w_statistic, p_value_wilcoxon) = wilcoxon_signed_rank(&advantages);

    // Cohen's d
    let cohens_d = mean / std;

    // 95% CI (t-critical for df=99 is ~1.984)
    let t_critical = 1.984;
    let ci_lower = mean - t_critical * standard_error;
    let ci_upper = mean + t_critical * standard_error;

    StatisticalSummary {
        mean_advantage: mean,
        std_advantage: std,
        median_advantage: median_val,
        min_advantage: min_val,
        max_advantage: max_val,
        quantum_wins,
        classical_wins,
        ties,
        t_statistic,
        p_value_t,
        w_statistic,
        p_value_wilcoxon,
        cohens_d,
        ci_lower,
        ci_upper,
    }
}

fn std_dev(values: &[f32]) -> f32 {
    let n = values.len();
    let mean = values.iter().sum::<f32>() / n as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    variance.sqrt()
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn estimate_p_value_t(t_abs: f32, n: usize) -> f32 {
    // Approximate p-value for two-tailed t-test with df = n-1
    // For n=100 (df=99):
    // t > 3.39 → p < 0.001
    // t > 2.63 → p < 0.01
    // t > 1.98 → p < 0.05
    // t > 1.66 → p < 0.10
    let _ = n; // df = 99, close enough to normal
    if t_abs > 3.39 {
        0.0005
    } else if t_abs > 2.63 {
        0.005
    } else if t_abs > 1.98 {
        0.025
    } else if t_abs > 1.66 {
        0.05
    } else {
        0.20
    }
}

fn wilcoxon_signed_rank(values: &[f32]) -> (f32, f32) {
    // Wilcoxon signed-rank test for H0: median = 0
    // Returns (W+, approximate p-value)

    // Remove zeros and get absolute values with signs
    let mut ranked: Vec<(f32, i32)> = values
        .iter()
        .filter(|&&x| x.abs() > 0.001)
        .map(|&x| (x.abs(), if x > 0.0 { 1 } else { -1 }))
        .collect();

    // Sort by absolute value
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks and sum positive ranks
    let n = ranked.len();
    let mut w_plus: f32 = 0.0;
    let mut w_minus: f32 = 0.0;

    for (i, (_, sign)) in ranked.iter().enumerate() {
        let rank = (i + 1) as f32;
        if *sign > 0 {
            w_plus += rank;
        } else {
            w_minus += rank;
        }
    }

    let w = w_plus.min(w_minus);

    // For large n, W is approximately normal
    // Mean under H0: n(n+1)/4
    // Var under H0: n(n+1)(2n+1)/24
    let n_f = n as f32;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let var_w = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;
    let z = (w - mean_w) / var_w.sqrt();

    // Approximate p-value from z-score (two-tailed)
    let z_abs = z.abs();
    let p_value = if z_abs > 3.29 {
        0.0005
    } else if z_abs > 2.58 {
        0.005
    } else if z_abs > 1.96 {
        0.025
    } else if z_abs > 1.65 {
        0.05
    } else {
        0.20
    };

    (w_plus, p_value)
}

fn ci_includes_zero(lower: f32, upper: f32) -> bool {
    lower <= 0.0 && upper >= 0.0
}

fn format_p_value(p: f32) -> String {
    if p < 0.001 {
        "< 0.001".to_string()
    } else if p < 0.01 {
        "< 0.01".to_string()
    } else if p < 0.05 {
        "< 0.05".to_string()
    } else {
        format!("{:.3}", p)
    }
}

fn print_histogram(results: &[SeedResult]) {
    // Buckets: <-50, -50:-30, -30:-10, -10:10, 10:30, 30:50, >50
    let mut buckets = [0usize; 7];

    for r in results {
        let bucket = if r.advantage < -50.0 {
            0
        } else if r.advantage < -30.0 {
            1
        } else if r.advantage < -10.0 {
            2
        } else if r.advantage <= 10.0 {
            3
        } else if r.advantage <= 30.0 {
            4
        } else if r.advantage <= 50.0 {
            5
        } else {
            6
        };
        buckets[bucket] += 1;
    }

    let labels = [
        "   <-50", "-50:-30", "-30:-10", " -10:10", "  10:30", "  30:50", "    >50",
    ];
    let max_count = *buckets.iter().max().unwrap_or(&1);
    let scale = 40.0 / max_count as f32;

    for (i, (label, &count)) in labels.iter().zip(buckets.iter()).enumerate() {
        let bar_len = (count as f32 * scale) as usize;
        let bar: String = "█".repeat(bar_len);
        let marker = if i == 3 { " ← TIES" } else { "" };
        println!("  {} │{} ({}){}", label, bar, count, marker);
    }

    println!();
    println!("           Classical ←──────────────────────→ Quantum");
}
