// EPIC 78: Final REAL benchmark - no more bugs!

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, WmmaGateType};

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 78 FINAL: Real Numbers, No Bugs, No Cave Walls! 🚀      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = CudaRuntime::new().expect("CUDA runtime");

    println!("═══════════════════════════════════════════════════════════════");
    println!(" Baseline: EPIC 72 (1 state, 100M depth)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let start = std::time::Instant::now();
    let (g, a) = engine::cuda::run_wmma_multi_state(
        &rt,
        1,
        16,
        WmmaGateType::Hadamard,
        100_000_000,
        MultiStateOpt::Basic,
    )
    .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!("TCOPS: {:.2}", (a as f64 / elapsed) / 1e12);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!(" Phase 2B: Massive Parallelism (1024 states, 1M depth)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let start = std::time::Instant::now();
    let (g, a) = engine::cuda::run_wmma_multi_state(
        &rt,
        1024,
        16,
        WmmaGateType::Hadamard,
        1_000_000,
        MultiStateOpt::Basic,
    )
    .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!("TCOPS: {:.2}", (a as f64 / elapsed) / 1e12);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!(" Phase 2C: ILP Optimization (1024 states, 1M depth)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let start = std::time::Instant::now();
    let (g, a) = engine::cuda::run_wmma_multi_state(
        &rt,
        1024,
        16,
        WmmaGateType::Hadamard,
        1_000_000,
        MultiStateOpt::ILP,
    )
    .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!("TCOPS: {:.2}", (a as f64 / elapsed) / 1e12);
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Summary: $600 Laptop GPU Performance                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: Requires cuda,perf-bench features");
    std::process::exit(1);
}
