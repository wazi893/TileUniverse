//! P-bit MaxCut Benchmark
//!
//! Benchmarks the p-bit Ising machine on MaxCut problems of various sizes.
//!
//! Run with:
//!   cargo run --release --example pbit_maxcut_bench
//!
//! With CUDA acceleration:
//!   cargo run --release --features cuda --example pbit_maxcut_bench

use engine::pbit::{
    AnnealingSchedule, GibbsSampler, MaxCutGraph, PBitConfig, PBitNetwork, ParallelTempering,
    SimulatedAnnealing, UpdateOrder,
};
use std::time::Instant;

#[cfg(feature = "cuda")]
use engine::pbit::{GpuPBitNetwork, GpuSimulatedAnnealing};

fn main() {
    println!("=== P-bit MaxCut Benchmark ===\n");

    // Benchmark different graph sizes
    println!("1. Scaling with graph size (random graphs, edge_prob=0.5)\n");
    println!(
        "{:>6} {:>8} {:>10} {:>12} {:>10}",
        "Nodes", "Edges", "Best Cut", "Time (ms)", "Cut/Edge"
    );
    println!("{}", "-".repeat(50));

    for n in [10, 20, 50, 100, 200] {
        let result = bench_random_graph(n, 0.5, 10000);
        println!(
            "{:>6} {:>8} {:>10} {:>12.2} {:>10.2}%",
            n,
            result.edges,
            result.best_cut,
            result.time_ms,
            100.0 * result.best_cut as f64 / result.edges as f64
        );
    }

    // Compare sampling methods
    println!("\n2. Sampling method comparison (n=100, edge_prob=0.3)\n");
    compare_methods(100, 0.3);

    // Annealing schedule comparison
    println!("\n3. Annealing schedule comparison (n=50)\n");
    compare_schedules(50);

    // Known optimal: complete graph
    println!("\n4. Complete graphs (known optimal)\n");
    bench_complete_graphs();

    // Parallel tempering demo
    println!("\n5. Parallel tempering (n=50)\n");
    bench_parallel_tempering(50);

    // GPU benchmarks (if CUDA available)
    #[cfg(feature = "cuda")]
    {
        println!("\n6. GPU Acceleration Benchmark\n");
        bench_gpu_acceleration();
    }
}

struct BenchResult {
    edges: usize,
    best_cut: u32,
    time_ms: f64,
}

fn bench_random_graph(n: usize, edge_prob: f64, steps: u64) -> BenchResult {
    let graph = MaxCutGraph::random(n, edge_prob, 42);
    let problem = graph.to_ising();
    let edges = graph.num_edges();

    let config = PBitConfig::new(n).with_seed(123);
    let mut network = PBitNetwork::new(config);
    network.set_problem(&problem);

    let start = Instant::now();
    let mut annealer = SimulatedAnnealing::new(&mut network)
        .with_steps(steps)
        .with_schedule(AnnealingSchedule::Exponential {
            beta_min: 0.1,
            beta_max: 10.0,
        });
    let result = annealer.run();
    let elapsed = start.elapsed();

    let best_cut = graph.cut_value(&result.best_state) as u32;

    BenchResult {
        edges,
        best_cut,
        time_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

fn compare_methods(n: usize, edge_prob: f64) {
    let graph = MaxCutGraph::random(n, edge_prob, 42);
    let problem = graph.to_ising();
    let edges = graph.num_edges();

    println!(
        "{:>20} {:>10} {:>12} {:>10}",
        "Method", "Best Cut", "Time (ms)", "Cut/Edge"
    );
    println!("{}", "-".repeat(55));

    // Gibbs sampling (high beta)
    {
        let config = PBitConfig::new(n).with_beta(5.0).with_seed(123);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let start = Instant::now();
        let mut sampler = GibbsSampler::new(&mut network)
            .with_equilibration(1000)
            .with_sampling(9000);
        let _ = sampler.find_minimum();
        let elapsed = start.elapsed();

        let cut = graph.cut_value(network.best_state()) as u32;
        println!(
            "{:>20} {:>10} {:>12.2} {:>10.2}%",
            "Gibbs (β=5)",
            cut,
            elapsed.as_secs_f64() * 1000.0,
            100.0 * cut as f64 / edges as f64
        );
    }

    // Simulated annealing (linear)
    {
        let config = PBitConfig::new(n).with_seed(123);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let start = Instant::now();
        let mut annealer = SimulatedAnnealing::new(&mut network)
            .with_steps(10000)
            .with_schedule(AnnealingSchedule::Linear {
                beta_min: 0.1,
                beta_max: 10.0,
            });
        let result = annealer.run();
        let elapsed = start.elapsed();

        let cut = graph.cut_value(&result.best_state) as u32;
        println!(
            "{:>20} {:>10} {:>12.2} {:>10.2}%",
            "SA (linear)",
            cut,
            elapsed.as_secs_f64() * 1000.0,
            100.0 * cut as f64 / edges as f64
        );
    }

    // Simulated annealing (exponential)
    {
        let config = PBitConfig::new(n).with_seed(123);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let start = Instant::now();
        let mut annealer = SimulatedAnnealing::new(&mut network)
            .with_steps(10000)
            .with_schedule(AnnealingSchedule::Exponential {
                beta_min: 0.1,
                beta_max: 10.0,
            });
        let result = annealer.run();
        let elapsed = start.elapsed();

        let cut = graph.cut_value(&result.best_state) as u32;
        println!(
            "{:>20} {:>10} {:>12.2} {:>10.2}%",
            "SA (exponential)",
            cut,
            elapsed.as_secs_f64() * 1000.0,
            100.0 * cut as f64 / edges as f64
        );
    }

    // SA with restarts
    {
        let config = PBitConfig::new(n).with_seed(123);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let start = Instant::now();
        let mut annealer = SimulatedAnnealing::new(&mut network).with_steps(2000);
        let result = annealer.run_with_restarts(5);
        let elapsed = start.elapsed();

        let cut = graph.cut_value(&result.best_state) as u32;
        println!(
            "{:>20} {:>10} {:>12.2} {:>10.2}%",
            "SA (5 restarts)",
            cut,
            elapsed.as_secs_f64() * 1000.0,
            100.0 * cut as f64 / edges as f64
        );
    }

    // Random update order
    {
        let config = PBitConfig::new(n)
            .with_seed(123)
            .with_update_order(UpdateOrder::Random);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let start = Instant::now();
        let mut annealer = SimulatedAnnealing::new(&mut network).with_steps(10000);
        let result = annealer.run();
        let elapsed = start.elapsed();

        let cut = graph.cut_value(&result.best_state) as u32;
        println!(
            "{:>20} {:>10} {:>12.2} {:>10.2}%",
            "SA (random order)",
            cut,
            elapsed.as_secs_f64() * 1000.0,
            100.0 * cut as f64 / edges as f64
        );
    }
}

fn compare_schedules(n: usize) {
    let graph = MaxCutGraph::random(n, 0.5, 42);
    let problem = graph.to_ising();

    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Schedule", "β_min", "β_max", "Best Cut"
    );
    println!("{}", "-".repeat(60));

    let schedules = [
        ("Linear 0.1→5", 0.1, 5.0),
        ("Linear 0.1→10", 0.1, 10.0),
        ("Linear 0.1→20", 0.1, 20.0),
        ("Linear 0.5→10", 0.5, 10.0),
        ("Linear 1.0→10", 1.0, 10.0),
    ];

    for (name, beta_min, beta_max) in schedules {
        let config = PBitConfig::new(n).with_seed(123);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let mut annealer = SimulatedAnnealing::new(&mut network)
            .with_steps(10000)
            .with_schedule(AnnealingSchedule::Linear { beta_min, beta_max });
        let result = annealer.run();

        let cut = graph.cut_value(&result.best_state) as u32;
        println!(
            "{:>20} {:>12.1} {:>12.1} {:>12}",
            name, beta_min, beta_max, cut
        );
    }
}

fn bench_complete_graphs() {
    // For complete graph K_n, optimal MaxCut = floor(n²/4)
    println!("{:>6} {:>10} {:>10} {:>10}", "n", "Optimal", "Found", "Gap");
    println!("{}", "-".repeat(40));

    for n in [4, 6, 8, 10, 12, 16, 20] {
        let graph = MaxCutGraph::complete(n);
        let problem = graph.to_ising();
        let optimal = (n * n) / 4;

        let config = PBitConfig::new(n).with_seed(42);
        let mut network = PBitNetwork::new(config);
        network.set_problem(&problem);

        let mut annealer = SimulatedAnnealing::new(&mut network)
            .with_steps(5000)
            .with_schedule(AnnealingSchedule::Exponential {
                beta_min: 0.1,
                beta_max: 15.0,
            });
        let result = annealer.run();

        let found = graph.cut_value(&result.best_state) as usize;
        let gap = optimal as i32 - found as i32;

        println!("{:>6} {:>10} {:>10} {:>10}", n, optimal, found, gap);
    }
}

fn bench_parallel_tempering(n: usize) {
    let graph = MaxCutGraph::random(n, 0.5, 42);
    let problem = graph.to_ising();

    println!("Graph: {} nodes, {} edges\n", n, graph.num_edges());

    // Create template network
    let config = PBitConfig::new(n);
    let template = PBitNetwork::new(config);

    // Run parallel tempering
    let start = Instant::now();
    let mut pt = ParallelTempering::new(&template, 8, 0.5, 10.0);
    pt.set_problem(&problem);
    pt.run(5000);
    let elapsed = start.elapsed();

    let (best_state, best_energy) = pt.best_solution();
    let best_cut = graph.cut_value(&best_state);

    println!("Replicas: 8 (β: 0.5 → 10.0)");
    println!("Sweeps: 5000");
    println!("Time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("Swap rate: {:.1}%", pt.swap_rate() * 100.0);
    println!("Best cut: {}", best_cut);
    println!("Best energy: {:.4}", best_energy);

    // Compare with single-temperature SA
    println!("\nComparison with SA (same total steps = 8 * 5000):");

    let config = PBitConfig::new(n).with_seed(42);
    let mut network = PBitNetwork::new(config);
    network.set_problem(&problem);

    let start = Instant::now();
    let mut annealer = SimulatedAnnealing::new(&mut network).with_steps(40000);
    let result = annealer.run();
    let elapsed = start.elapsed();

    let sa_cut = graph.cut_value(&result.best_state);
    println!(
        "SA best cut: {} (in {:.2} ms)",
        sa_cut,
        elapsed.as_secs_f64() * 1000.0
    );
}

/// GPU acceleration benchmark (only compiled with cuda feature)
#[cfg(feature = "cuda")]
fn bench_gpu_acceleration() {
    println!("CPU vs GPU Performance Comparison\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "N", "CPU (ms)", "GPU (ms)", "Speedup", "CPU Cut", "GPU Cut"
    );
    println!("{}", "-".repeat(70));

    let steps = 10000u64;

    for n in [100, 200, 500, 1000, 2000] {
        let graph = MaxCutGraph::random(n, 0.3, 42);
        let problem = graph.to_ising();

        // CPU benchmark
        let cpu_result = {
            let config = PBitConfig::new(n).with_seed(123);
            let mut network = PBitNetwork::new(config);
            network.set_problem(&problem);

            let start = Instant::now();
            let mut annealer = SimulatedAnnealing::new(&mut network)
                .with_steps(steps)
                .with_schedule(AnnealingSchedule::Exponential {
                    beta_min: 0.1,
                    beta_max: 10.0,
                });
            let result = annealer.run();
            let elapsed = start.elapsed();

            (
                elapsed.as_secs_f64() * 1000.0,
                graph.cut_value(&result.best_state),
            )
        };

        // GPU benchmark
        let gpu_result = match GpuPBitNetwork::new_with_seed(n, 1.0, 123) {
            Ok(mut gpu_net) => {
                if let Err(e) = gpu_net.set_problem(&problem) {
                    println!("GPU set_problem failed for n={}: {:?}", n, e);
                    continue;
                }

                let start = Instant::now();
                let mut annealer = GpuSimulatedAnnealing::new(&mut gpu_net)
                    .with_steps(steps)
                    .with_beta_range(0.1, 10.0);

                match annealer.run() {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        (
                            elapsed.as_secs_f64() * 1000.0,
                            graph.cut_value(&result.best_state),
                        )
                    }
                    Err(e) => {
                        println!("GPU annealing failed for n={}: {:?}", n, e);
                        continue;
                    }
                }
            }
            Err(e) => {
                println!("GPU initialization failed for n={}: {:?}", n, e);
                continue;
            }
        };

        let speedup = cpu_result.0 / gpu_result.0;

        println!(
            "{:>8} {:>12.2} {:>12.2} {:>12.1}x {:>10} {:>10}",
            n, cpu_result.0, gpu_result.0, speedup, cpu_result.1, gpu_result.1
        );
    }

    // Large-scale GPU-only benchmark
    println!("\nLarge-scale GPU-only benchmarks:\n");
    println!(
        "{:>8} {:>10} {:>12} {:>14} {:>12}",
        "N", "Edges", "Time (ms)", "Steps/sec", "Best Cut"
    );
    println!("{}", "-".repeat(60));

    for n in [5000, 10000] {
        let graph = MaxCutGraph::random(n, 0.1, 42); // Sparse for large n
        let problem = graph.to_ising();
        let edges = graph.num_edges();

        match GpuPBitNetwork::new_with_seed(n, 1.0, 42) {
            Ok(mut gpu_net) => {
                if let Err(e) = gpu_net.set_problem(&problem) {
                    println!("GPU set_problem failed for n={}: {:?}", n, e);
                    continue;
                }

                let steps = 5000u64;
                let start = Instant::now();
                let mut annealer = GpuSimulatedAnnealing::new(&mut gpu_net)
                    .with_steps(steps)
                    .with_beta_range(0.1, 10.0);

                match annealer.run() {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        let time_ms = elapsed.as_secs_f64() * 1000.0;
                        let steps_per_sec = (steps as f64 / elapsed.as_secs_f64()) as u64;
                        let cut = graph.cut_value(&result.best_state);

                        println!(
                            "{:>8} {:>10} {:>12.2} {:>14} {:>12}",
                            n, edges, time_ms, steps_per_sec, cut
                        );
                    }
                    Err(e) => {
                        println!("GPU annealing failed for n={}: {:?}", n, e);
                    }
                }
            }
            Err(e) => {
                println!("GPU initialization failed for n={}: {:?}", n, e);
            }
        }
    }
}
