// =============================================================================
// Sprint 61.0: Observer Search Demo
// =============================================================================
//
// Demonstrates the hybrid orchestration layer with Observer Grid substrate
// for observer-controlled quantum collapse dynamics.
//
// This example shows:
// - HybridScheduler with Observer substrate integration
// - Observer Grid execution with various trigger conditions
// - Anomaly detection and collapse ratio monitoring
// - Multi-substrate coordination (Quantum → Observer → Classical)
//
// Run with:
//   cargo run --example observer_search --release
//
// =============================================================================

use std::time::Instant;

use engine::hybrid::{
    ClassicalSubstrate, HybridScheduler, JobSpec, ObserverSpec, ObserverSubstrate, ObserverTrigger,
    QuantumSubstrate, SchedulerConfig, StageOutputs,
};
use engine::tile8::ObserverGridConfig;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("        SPRINT 61.0: OBSERVER SEARCH DEMO");
    println!("        Observer Grid Hybrid Orchestration");
    println!("{}\n", "=".repeat(80));

    // Demo 1: Basic Observer Grid execution
    println!("\n{}", "-".repeat(80));
    println!("DEMO 1: Basic Observer Grid Execution");
    println!("{}\n", "-".repeat(80));
    demo_basic_observer();

    // Demo 2: Observer with tick-based trigger
    println!("\n{}", "-".repeat(80));
    println!("DEMO 2: Tick-Based Termination");
    println!("{}\n", "-".repeat(80));
    demo_tick_trigger();

    // Demo 3: Observer with collapse ratio trigger
    println!("\n{}", "-".repeat(80));
    println!("DEMO 3: Collapse Ratio Trigger");
    println!("{}\n", "-".repeat(80));
    demo_collapse_trigger();

    // Demo 4: Compound triggers (Any/All)
    println!("\n{}", "-".repeat(80));
    println!("DEMO 4: Compound Triggers");
    println!("{}\n", "-".repeat(80));
    demo_compound_triggers();

    // Demo 5: Multi-substrate orchestration
    println!("\n{}", "-".repeat(80));
    println!("DEMO 5: Multi-Substrate Orchestration");
    println!("{}\n", "-".repeat(80));
    demo_multi_substrate();

    // Demo 6: Observer scaling test
    println!("\n{}", "-".repeat(80));
    println!("DEMO 6: Observer Scaling Performance");
    println!("{}\n", "-".repeat(80));
    demo_observer_scaling();

    println!("\n{}", "=".repeat(80));
    println!("                    OBSERVER SEARCH DEMO COMPLETE");
    println!("{}\n", "=".repeat(80));
}

/// Demo 1: Basic Observer Grid execution
fn demo_basic_observer() {
    let mut scheduler = create_observer_scheduler();

    // Configure a small observer grid
    let config = ObserverGridConfig {
        num_observers: 100,
        ..Default::default()
    };

    // Create observer job with simple tick-based termination
    let spec = ObserverSpec::new(config)
        .with_trigger(ObserverTrigger::AfterTicks(50))
        .with_max_ticks(100);

    println!("Observer Grid Configuration:");
    println!("  Observers: 100");
    println!("  Trigger: AfterTicks(50)");
    println!("  Max ticks: 100\n");

    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Observer(spec));

    if let Some(result) = scheduler.wait(job_id) {
        let elapsed = start.elapsed();

        println!("Execution completed in {:?}", elapsed);
        println!("Success: {}", result.is_success());

        if let Some(stage) = result.stage_results.first() {
            if let StageOutputs::Observer {
                collapse_ratio,
                anomaly_score,
                ticks,
                measurements,
            } = &stage.outputs
            {
                println!("\nObserver Grid Results:");
                println!("  Ticks executed: {}", ticks);
                println!("  Collapse ratio: {:.2}%", collapse_ratio * 100.0);
                println!("  Anomaly score:  {:.4}", anomaly_score);
                println!("  Measurements:   {} values", measurements.len());

                if !measurements.is_empty() {
                    let sum: u32 = measurements.iter().map(|&x| x as u32).sum();
                    let avg = sum as f64 / measurements.len() as f64;
                    println!("  Average value:  {:.2}", avg);
                }
            }
        }
    }
}

/// Demo 2: Observer with specific tick count
fn demo_tick_trigger() {
    let mut scheduler = create_observer_scheduler();

    let tick_counts = [10, 50, 100, 200];

    println!("Testing different tick triggers:\n");
    println!(
        "{:>6} {:>12} {:>15} {:>12}",
        "Ticks", "Collapse %", "Anomaly Score", "Time"
    );
    println!("{}", "-".repeat(50));

    for &ticks in &tick_counts {
        let config = ObserverGridConfig {
            num_observers: 64,
            ..Default::default()
        };

        let spec = ObserverSpec::new(config)
            .with_trigger(ObserverTrigger::AfterTicks(ticks))
            .with_max_ticks(ticks + 50);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Observer(spec));

        if let Some(result) = scheduler.wait(job_id) {
            let elapsed = start.elapsed();

            if let Some(stage) = result.stage_results.first() {
                if let StageOutputs::Observer {
                    collapse_ratio,
                    anomaly_score,
                    ..
                } = &stage.outputs
                {
                    println!(
                        "{:>6} {:>11.1}% {:>15.4} {:>12?}",
                        ticks,
                        collapse_ratio * 100.0,
                        anomaly_score,
                        elapsed
                    );
                }
            }
        }
    }
}

/// Demo 3: Observer with collapse ratio trigger
fn demo_collapse_trigger() {
    let mut scheduler = create_observer_scheduler();

    let thresholds = [0.25, 0.50, 0.75, 0.90];

    println!("Testing collapse ratio triggers:\n");
    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "Threshold", "Achieved", "Ticks", "Time"
    );
    println!("{}", "-".repeat(50));

    for &threshold in &thresholds {
        let config = ObserverGridConfig {
            num_observers: 64,
            ..Default::default()
        };

        let spec = ObserverSpec::new(config)
            .with_trigger(ObserverTrigger::OnCollapseRatio { threshold })
            .with_max_ticks(1000);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Observer(spec));

        if let Some(result) = scheduler.wait(job_id) {
            let elapsed = start.elapsed();

            if let Some(stage) = result.stage_results.first() {
                if let StageOutputs::Observer {
                    collapse_ratio,
                    ticks,
                    ..
                } = &stage.outputs
                {
                    println!(
                        "{:>9.0}% {:>11.1}% {:>12} {:>12?}",
                        threshold * 100.0,
                        collapse_ratio * 100.0,
                        ticks,
                        elapsed
                    );
                }
            }
        }
    }
}

/// Demo 4: Compound triggers (Any/All)
fn demo_compound_triggers() {
    let mut scheduler = create_observer_scheduler();

    // Test "Any" trigger - terminate when ANY condition is met
    println!("Compound 'Any' Trigger (ticks >= 50 OR collapse >= 30%):");

    let config = ObserverGridConfig {
        num_observers: 64,
        ..Default::default()
    };

    let any_trigger = ObserverTrigger::Any(vec![
        ObserverTrigger::AfterTicks(50),
        ObserverTrigger::OnCollapseRatio { threshold: 0.30 },
    ]);

    let spec = ObserverSpec::new(config.clone())
        .with_trigger(any_trigger)
        .with_max_ticks(200);

    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Observer(spec));

    if let Some(result) = scheduler.wait(job_id) {
        if let Some(stage) = result.stage_results.first() {
            if let StageOutputs::Observer {
                collapse_ratio,
                ticks,
                ..
            } = &stage.outputs
            {
                println!(
                    "  Result: {} ticks, {:.1}% collapse (triggered by: {})",
                    ticks,
                    collapse_ratio * 100.0,
                    if *ticks >= 50 {
                        "tick count"
                    } else {
                        "collapse ratio"
                    }
                );
            }
        }
    }
    println!("  Time: {:?}\n", start.elapsed());

    // Test "All" trigger - terminate when ALL conditions are met
    println!("Compound 'All' Trigger (ticks >= 30 AND collapse >= 20%):");

    let all_trigger = ObserverTrigger::All(vec![
        ObserverTrigger::AfterTicks(30),
        ObserverTrigger::OnCollapseRatio { threshold: 0.20 },
    ]);

    let spec = ObserverSpec::new(config)
        .with_trigger(all_trigger)
        .with_max_ticks(500);

    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Observer(spec));

    if let Some(result) = scheduler.wait(job_id) {
        if let Some(stage) = result.stage_results.first() {
            if let StageOutputs::Observer {
                collapse_ratio,
                ticks,
                ..
            } = &stage.outputs
            {
                println!(
                    "  Result: {} ticks, {:.1}% collapse (both conditions met)",
                    ticks,
                    collapse_ratio * 100.0
                );
            }
        }
    }
    println!("  Time: {:?}", start.elapsed());
}

/// Demo 5: Multi-substrate coordination
fn demo_multi_substrate() {
    let mut scheduler = HybridScheduler::new();

    // Register all three substrate types
    scheduler.register_substrate(Box::new(ClassicalSubstrate::new(1, "CPU")));
    scheduler.register_substrate(Box::new(QuantumSubstrate::new(2, "Quantum-Sim", 10)));
    scheduler.register_substrate(Box::new(ObserverSubstrate::new(3, "Observer-Grid")));

    println!("Multi-substrate scheduler configured:");
    for info in scheduler.list_substrates() {
        println!(
            "  [{:?}] {} - Capacity: {:.0}%",
            info.substrate_type,
            info.name,
            info.capacity * 100.0
        );
    }

    // Run an observer job
    let config = ObserverGridConfig {
        num_observers: 128,
        ..Default::default()
    };

    let spec = ObserverSpec::new(config)
        .with_trigger(ObserverTrigger::AfterTicks(100))
        .with_max_ticks(200);

    println!("\nRunning observer job on multi-substrate scheduler...");

    let start = Instant::now();
    let job_id = scheduler.submit(JobSpec::Observer(spec));

    if let Some(result) = scheduler.wait(job_id) {
        let elapsed = start.elapsed();

        println!("\nJob completed in {:?}", elapsed);

        if let Some(stage) = result.stage_results.first() {
            if let StageOutputs::Observer {
                collapse_ratio,
                anomaly_score,
                ticks,
                measurements,
            } = &stage.outputs
            {
                println!("Results:");
                println!("  Ticks: {}", ticks);
                println!("  Collapse ratio: {:.1}%", collapse_ratio * 100.0);
                println!("  Anomaly score: {:.4}", anomaly_score);
                println!("  Measurements: {} collapsed observers", measurements.len());
            }
        }
    }

    // Check scheduler stats
    let stats = scheduler.stats();
    println!("\nScheduler statistics:");
    println!("  Jobs submitted: {}", stats.total_submitted);
    println!("  Jobs completed: {}", stats.total_completed);
    println!("  Queue size: {}", stats.queue_size);
}

/// Demo 6: Observer scaling test
fn demo_observer_scaling() {
    let mut scheduler = create_observer_scheduler();

    let observer_counts = [100, 1_000, 10_000, 100_000];

    println!("Testing observer count scaling:\n");
    println!(
        "{:>10} {:>12} {:>15} {:>12} {:>15}",
        "Observers", "Ticks", "Collapse %", "Time", "Obs/sec"
    );
    println!("{}", "-".repeat(70));

    for &count in &observer_counts {
        let config = ObserverGridConfig {
            num_observers: count,
            ..Default::default()
        };

        let spec = ObserverSpec::new(config)
            .with_trigger(ObserverTrigger::AfterTicks(100))
            .with_max_ticks(100);

        let start = Instant::now();
        let job_id = scheduler.submit(JobSpec::Observer(spec));

        if let Some(result) = scheduler.wait(job_id) {
            let elapsed = start.elapsed();

            if let Some(stage) = result.stage_results.first() {
                if let StageOutputs::Observer {
                    collapse_ratio,
                    ticks,
                    ..
                } = &stage.outputs
                {
                    let ops_per_sec = (count as f64 * *ticks as f64) / elapsed.as_secs_f64();
                    println!(
                        "{:>10} {:>12} {:>14.1}% {:>12?} {:>14.2e}",
                        count,
                        ticks,
                        collapse_ratio * 100.0,
                        elapsed,
                        ops_per_sec
                    );
                }
            }
        }
    }

    println!("\nNote: Observer operations scale linearly with count.");
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_observer_scheduler() -> HybridScheduler {
    let config = SchedulerConfig {
        max_queue_size: 100,
        max_concurrent: 4,
        default_timeout: std::time::Duration::from_secs(30),
        auto_route: true,
        enable_preemption: false,
    };

    let mut scheduler = HybridScheduler::with_config(config);

    // Register observer substrate with high capacity
    let observer = ObserverSubstrate::new(1, "Observer-Grid").with_max_observers(1_000_000);

    scheduler.register_substrate(Box::new(observer));
    scheduler
}
