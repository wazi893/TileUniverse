// EPIC 78: Test Multi-State WMMA Kernel
//
// Run with: cargo run --example test_multistate --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::bench_api::bench_api;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  EPIC 78: Multi-State Batched WMMA Test");
    println!("  Testing GPU massive parallelism optimization (Fix 4)");
    println!("════════════════════════════════════════════════════════════════\n");

    // Test with increasing batch sizes to find L2 cache sweet spot
    let test_configs = vec![
        (16, 12, 1000),   // 16 states × 12 qubits (16 tiles each) × 1000 depth
        (64, 12, 1000),   // 64 states × 12 qubits
        (256, 12, 1000),  // 256 states × 12 qubits
        (1024, 12, 1000), // 1024 states × 12 qubits (THE MONSTER)
    ];

    println!("Testing batch sizes: 16, 64, 256, 1024 parallel states\n");

    for (num_states, qubits, depth) in test_configs {
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "Testing {} states × {} qubits × {} depth...",
            num_states, qubits, depth
        );

        let report = bench_api::quantum_gpu_multi_state_cops_benchmark(num_states, qubits, depth);

        println!("\n{}", report.name);
        println!("Hardware: {}", report.hardware);
        println!("Elapsed: {:.3}s", report.elapsed_secs);

        if let Some(cops) = report.combined {
            println!("TCOPS: {:.2}", cops.tcops());
            println!("Display: {}", cops.display());
        }

        println!();
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("  Test Complete!");
    println!("════════════════════════════════════════════════════════════════\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: This example requires both 'cuda' and 'perf-bench' features.");
    eprintln!("Run with: cargo run --example test_multistate --features cuda,perf-bench --release");
    std::process::exit(1);
}
