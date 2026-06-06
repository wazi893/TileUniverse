//! EPIC 90 Phase 90.4a: Performance Benchmark
//!
//! Measures fusion speedup, batch grouping efficiency, and determines
//! if GPU batching is needed.
//!
//! Run with:
//! ```
//! cargo run --release --example epic90_benchmark --features perf-bench
//! ```

use engine::metrics::CopsMetric;
use engine::quantum_forager::{
    BatchStats, ForagerPopulation, ForagerRegion, QuantumForagerConfig, TickMetrics,
    group_organisms_by_circuit,
};
use engine::simulation::Simulation;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  EPIC 90 Phase 90.4a: Performance Benchmark");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Run scenarios
    let results = vec![
        benchmark_scenario("Small", 100, 4, 20),
        benchmark_scenario("Medium", 500, 4, 30),
        benchmark_scenario("Large", 1000, 4, 50),
    ];

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Decision Gate Analysis");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Print summary table
    println!("┌─────────┬───────────┬────────────┬───────────┬───────────┬───────────┐");
    println!("│ Scenario│ Orgs/sec  │ Fusion     │ Avg Batch │ Max Batch │ COPS      │");
    println!("├─────────┼───────────┼────────────┼───────────┼───────────┼───────────┤");
    for r in &results {
        println!(
            "│ {:7} │ {:9.0} │ {:5.2}×     │ {:9.1} │ {:9} │ {:9} │",
            r.name,
            r.organisms_per_sec,
            r.fusion_speedup,
            r.avg_batch_size,
            r.largest_batch,
            r.cops.display()
        );
    }
    println!("└─────────┴───────────┴────────────┴───────────┴───────────┴───────────┘");

    // Decision analysis
    println!();
    println!("Decision Matrix:");
    println!();

    let large = results.iter().find(|r| r.name == "Large").unwrap();

    if large.organisms_per_sec >= 100_000.0 {
        println!("  ✓ Organisms/sec >= 100K ({:.0})", large.organisms_per_sec);
        println!("  → DECISION: Ship EPIC 90 as complete");
        println!("  → GPU batching is premature optimization");
    } else if large.avg_batch_size < 20.0 {
        println!("  ✗ Organisms/sec < 100K ({:.0})", large.organisms_per_sec);
        println!("  ✗ Avg batch size < 20 ({:.1})", large.avg_batch_size);
        println!("  → DECISION: Ship EPIC 90 as complete");
        println!("  → Batches too small for GPU benefit");
    } else if large.avg_batch_size >= 50.0 {
        println!("  ✗ Organisms/sec < 100K ({:.0})", large.organisms_per_sec);
        println!("  ✓ Avg batch size >= 50 ({:.1})", large.avg_batch_size);
        println!("  → DECISION: Proceed to Phase 90.4b");
        println!("  → GPU batching will provide significant speedup");
    } else {
        println!("  ✗ Organisms/sec < 100K ({:.0})", large.organisms_per_sec);
        println!(
            "  ~ Avg batch size = {:.1} (between 20-50)",
            large.avg_batch_size
        );
        println!("  → DECISION: Consider Phase 90.4b");
        println!("  → Marginal case - GPU may or may not help");
    }

    println!();
    println!("Additional metrics:");
    println!(
        "  Fusion efficiency: {:.1}%",
        large.fusion_efficiency * 100.0
    );
    println!("  Metrics overhead: {:.1}%", large.metrics_overhead_pct);
    println!("  Unique circuits: {}", large.unique_circuits);
}

#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    organisms_per_sec: f64,
    fusion_speedup: f64,
    avg_batch_size: f64,
    largest_batch: usize,
    unique_circuits: usize,
    fusion_efficiency: f64,
    metrics_overhead_pct: f64,
    cops: CopsMetric,
}

fn benchmark_scenario(
    name: &'static str,
    organisms: usize,
    qubits: u8,
    gates: usize,
) -> BenchmarkResult {
    println!("───────────────────────────────────────────────────────────────────────");
    println!(
        "  {} Scenario: {} organisms, {} qubits, {} gates/org",
        name, organisms, qubits, gates
    );
    println!("───────────────────────────────────────────────────────────────────────");

    // Setup simulation with resources
    let mut sim = Simulation::new();
    for x in 0..200 {
        for y in 0..200 {
            sim.set_heat_field_for_test(x, y, 50);
        }
    }

    // Setup population config
    let config = QuantumForagerConfig {
        n_qubits: qubits,
        initial_circuit_length: gates,
        max_circuit_length: gates * 2,
        initial_population: organisms,
        initial_energy: 100.0,
        metabolism_cost: 0.1,
        movement_cost: 0.05,
        reproduction_threshold: 200.0, // High to prevent population explosion
        reproduction_cost: 50.0,
        heat_energy_factor: 0.1,
        charge_energy_factor: 0.1,
        consume_resources: false, // Don't deplete for consistent benchmarks
        mutation_rate: 0.0,       // No mutation for consistent circuits
        region: ForagerRegion {
            x0: 10,
            y0: 10,
            width: 180,
            height: 180,
        },
        max_population: organisms * 2,
    };

    let mut pop = ForagerPopulation::new(config, 12345);

    // Warmup
    println!("  Warming up...");
    for _ in 0..3 {
        pop.step(&mut sim);
    }

    // Benchmark: Regular step (uses fused path now)
    println!("  Benchmarking step() [10 ticks]...");
    let step_start = Instant::now();
    for _ in 0..10 {
        pop.step(&mut sim);
    }
    let step_time = step_start.elapsed().as_secs_f64() / 10.0;

    // Benchmark: Batched step
    println!("  Benchmarking step_batched() [10 ticks]...");
    let batched_start = Instant::now();
    let mut last_batch_stats = BatchStats::default();
    for _ in 0..10 {
        let (_, batch_stats) = pop.step_batched(&mut sim);
        last_batch_stats = batch_stats;
    }
    let batched_time = batched_start.elapsed().as_secs_f64() / 10.0;

    // Benchmark: With metrics
    println!("  Benchmarking step_with_metrics() [10 ticks]...");
    let metrics_start = Instant::now();
    let mut last_metrics: Option<TickMetrics> = None;
    for _ in 0..10 {
        last_metrics = Some(pop.step_with_metrics(&mut sim));
    }
    let metrics_time = metrics_start.elapsed().as_secs_f64() / 10.0;
    let tick_metrics = last_metrics.unwrap();

    // Analyze batch distribution
    let batches = group_organisms_by_circuit(pop.iter());
    let unique_circuits = batches.len();
    let largest_batch = batches
        .iter()
        .map(|b| b.organism_ids.len())
        .max()
        .unwrap_or(0);
    let current_size = pop.size();
    let avg_batch_size = if unique_circuits > 0 {
        current_size as f64 / unique_circuits as f64
    } else {
        0.0
    };

    // Calculate derived metrics
    let organisms_per_sec = current_size as f64 / batched_time;
    let fusion_speedup = step_time / batched_time;
    let metrics_overhead_pct = ((metrics_time - batched_time) / batched_time) * 100.0;

    // Report
    println!();
    println!("  Results:");
    println!(
        "    step() time:          {:.3}ms / tick",
        step_time * 1000.0
    );
    println!(
        "    step_batched() time:  {:.3}ms / tick",
        batched_time * 1000.0
    );
    println!(
        "    step_with_metrics():  {:.3}ms / tick",
        metrics_time * 1000.0
    );
    println!("    Fusion speedup:       {:.2}×", fusion_speedup);
    println!("    Metrics overhead:     {:.1}%", metrics_overhead_pct);
    println!("    Organisms/sec:        {:.0}", organisms_per_sec);
    println!("    Unique circuits:      {}", unique_circuits);
    println!("    Avg batch size:       {:.1}", avg_batch_size);
    println!("    Largest batch:        {}", largest_batch);
    println!("    Gates executed:       {}", last_batch_stats.total_gates);
    println!(
        "    Gates eliminated:     {}",
        last_batch_stats.gates_eliminated
    );
    println!(
        "    Fusion efficiency:    {:.1}%",
        last_batch_stats.fusion_efficiency() * 100.0
    );
    println!("    COPS:                 {}", tick_metrics.cops.display());
    println!();

    BenchmarkResult {
        name,
        organisms_per_sec,
        fusion_speedup,
        avg_batch_size,
        largest_batch,
        unique_circuits,
        fusion_efficiency: last_batch_stats.fusion_efficiency() as f64,
        metrics_overhead_pct,
        cops: tick_metrics.cops,
    }
}
