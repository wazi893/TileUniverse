// EPIC 78 Phase 2C: ILP Optimization Test
//
// Run with: cargo run --example phase2c_test --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::bench_api::bench_api;
    use engine::cuda::MultiStateOpt;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  EPIC 78 Phase 2C: Instruction-Level Parallelism Test");
    println!("════════════════════════════════════════════════════════════════\n");

    let configs = vec![
        ("Basic (Phase 2B)", MultiStateOpt::Basic),
        ("ILP (Phase 2C)", MultiStateOpt::ILP),
    ];

    for (name, opt) in configs {
        println!("──────────────────────────────────────────────────────────────");
        println!("Testing: {}", name);
        println!("──────────────────────────────────────────────────────────────\n");

        let report = bench_api::quantum_gpu_multi_state_cops_benchmark(
            opt, 1024, // num_states
            12,   // qubits
            1000, // depth
        );

        println!("Name: {}", report.name);
        println!("Hardware: {}", report.hardware);
        println!("Elapsed: {:.3}s", report.elapsed_secs);

        if let Some(cops) = report.combined {
            println!("TCOPS: {:.2}", cops.tcops());
            println!("Display: {}", cops.display());
        }

        println!();
    }

    println!("════════════════════════════════════════════════════════════════\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: This example requires both 'cuda' and 'perf-bench' features.");
    eprintln!("Run with: cargo run --example phase2c_test --features cuda,perf-bench --release");
    std::process::exit(1);
}
