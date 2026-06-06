// EPIC 78 Phase 2D: Persistent Buffer Test
// Compare allocation overhead vs zero-copy persistent buffers

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 78 Phase 2D: Persistent Buffer Optimization              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));

    // Test configuration
    let num_states = 1024;
    let qubits = 12;
    let tiles_per_state = 16;
    let depth = 1_000_000;

    println!("Configuration:");
    println!("  States: {}", num_states);
    println!("  Qubits: {}", qubits);
    println!("  Depth: {}", depth);
    println!();

    // Phase 2C: With allocation overhead
    println!("═══════════════════════════════════════════════════════════════");
    println!(" Phase 2C: ILP with allocation overhead");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut total_elapsed = 0.0;
    for i in 0..5 {
        let start = std::time::Instant::now();
        let (g, a) = engine::cuda::run_wmma_multi_state(
            &rt,
            num_states,
            tiles_per_state,
            WmmaGateType::Hadamard,
            depth,
            MultiStateOpt::ILP,
        )
        .unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        total_elapsed += elapsed;
        println!(
            "  Run {}: {:.3}s ({:.2} TCOPS)",
            i + 1,
            elapsed,
            (a as f64 / elapsed) / 1e12
        );
    }
    let avg_2c = total_elapsed / 5.0;
    println!("\nAverage: {:.3}s\n", avg_2c);

    // Phase 2D: Persistent buffers (zero allocation)
    println!("═══════════════════════════════════════════════════════════════");
    println!(" Phase 2D: Persistent buffers (zero allocation overhead)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let persistent = MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state)
        .expect("Create persistent pool");

    println!();

    let mut total_elapsed = 0.0;
    for i in 0..5 {
        let (g, a, elapsed) = persistent
            .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
            .unwrap();
        total_elapsed += elapsed;
        println!(
            "  Run {}: {:.3}s ({:.2} TCOPS)",
            i + 1,
            elapsed,
            (a as f64 / elapsed) / 1e12
        );
    }
    let avg_2d = total_elapsed / 5.0;
    println!("\nAverage: {:.3}s\n", avg_2d);

    // Comparison
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2D Improvement                                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let speedup = avg_2c / avg_2d;
    println!("  Phase 2C (with alloc): {:.3}s", avg_2c);
    println!("  Phase 2D (persistent): {:.3}s", avg_2d);
    println!("  Speedup: {:.2}×", speedup);
    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: Requires cuda,perf-bench features");
    std::process::exit(1);
}
