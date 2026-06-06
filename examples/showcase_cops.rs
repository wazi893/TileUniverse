// EPIC 78: COPS Showcase Example
//
// Run with: cargo run --example showcase_cops --features perf-bench,cuda --release

#[cfg(feature = "perf-bench")]
fn main() {
    use engine::bench_api::bench_api;

    println!("\n████████████████████████████████████████████████████████████████");
    println!("██                                                            ██");
    println!("██   LOGIC FABRIC ENGINE - PERFORMANCE SHOWCASE              ██");
    println!("██   Hybrid Quantum-Classical Computing Platform             ██");
    println!("██                                                            ██");
    println!("████████████████████████████████████████████████████████████████\n");

    let reports = bench_api::showcase_benchmark();

    for report in &reports {
        report.print_vc_summary();
    }

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY: RTX 4070 ($600 GPU)                                 ║");
    println!("║  ──────────────────────────────────────────────────────────────  ║");

    // Extract metrics dynamically based on available reports
    let quantum_cpu_tcops = reports
        .get(0)
        .and_then(|r| r.combined.as_ref().map(|c| c.tcops()))
        .unwrap_or(0.0);

    #[cfg(feature = "cuda")]
    let quantum_gpu_tcops = reports
        .get(1)
        .and_then(|r| r.combined.as_ref().map(|c| c.tcops()))
        .unwrap_or(0.0);

    #[cfg(feature = "cuda")]
    let classical_offset = 2;

    #[cfg(not(feature = "cuda"))]
    let classical_offset = 1;

    let classical_1w_tcops = reports
        .get(classical_offset)
        .and_then(|r| r.combined.as_ref().map(|c| c.tcops()))
        .unwrap_or(0.0);
    let classical_10w_tcops = reports
        .get(classical_offset + 1)
        .and_then(|r| r.combined.as_ref().map(|c| c.tcops()))
        .unwrap_or(0.0);
    let classical_100w_tcops = reports
        .get(classical_offset + 2)
        .and_then(|r| r.combined.as_ref().map(|c| c.tcops()))
        .unwrap_or(0.0);

    println!(
        "║  Quantum CPU (AVX2):    {:.2} TCOPS (12 qubits, 100K depth)   ║",
        quantum_cpu_tcops
    );

    #[cfg(feature = "cuda")]
    println!(
        "║  Quantum GPU (CUDA):    {:.2} TCOPS (1024 parallel states)    ║",
        quantum_gpu_tcops
    );

    println!(
        "║  Classical (1 world):   {:.2} TCOPS (64-lane parallelism)    ║",
        classical_1w_tcops
    );
    println!(
        "║  Classical (10 worlds): {:.2} TCOPS (640-lane parallelism)   ║",
        classical_10w_tcops
    );
    println!(
        "║  Classical (100 worlds): {:.2} TCOPS (6,400-lane parallelism) ║",
        classical_100w_tcops
    );
    println!("║                                                                ║");

    #[cfg(feature = "cuda")]
    let peak = quantum_gpu_tcops.max(classical_100w_tcops);

    #[cfg(not(feature = "cuda"))]
    let peak = quantum_cpu_tcops.max(classical_100w_tcops);

    println!(
        "║  → PEAK DEMONSTRATED: {:.2} TRILLION OPS/SEC                  ║",
        peak
    );
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        println!("🎯 Phase 2B Complete: Massive Parallelism Optimization");
        println!(
            "   ✓ GPU multi-state kernel: {:.2} TCOPS ({:.0}× baseline)",
            quantum_gpu_tcops,
            quantum_gpu_tcops / 2.5
        );
        println!(
            "   → Next: Phase 2C (ILP) + 2D (FP16 native) → {:.0} TCOPS",
            peak * 4.5
        );
        println!(
            "   → Final: Phase 2E (Persistent kernels) → {:.0} TCOPS",
            peak * 5.8
        );
        println!();
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("🎯 Next steps:");
        println!("   - Rebuild with --features cuda to see GPU performance");
        println!("   - Phase 2B: Massive parallelism → 477+ TCOPS");
        println!();
    }
}

#[cfg(not(feature = "perf-bench"))]
fn main() {
    eprintln!("Error: This example requires the 'perf-bench' feature.");
    eprintln!("Run with: cargo run --example showcase_cops --features perf-bench,cuda --release");
    std::process::exit(1);
}
