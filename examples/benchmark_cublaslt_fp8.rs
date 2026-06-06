// EPIC 115.3: Benchmark cuBLASLt FP8 GEMM Performance
//
// This tests TRUE FP8 tensor core operations via cuBLASLt.
// Target: 838 TFLOPS (4x over FP16's 209.5 TFLOPS on RTX 5090)
//
// Run with: cargo run --example benchmark_cublaslt_fp8 --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use logic_fabric_core::cuda::{CublasLtFp8, CudaRuntime, is_cuda_available};
    use std::time::Instant;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115.3: cuBLASLt FP8 GEMM Benchmark                       ║");
    println!("║  Target: 838 TFLOPS (4x over FP16)                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");

    // Create cuBLASLt FP8 context
    let cublaslt = match CublasLtFp8::new(&rt) {
        Ok(ctx) => {
            println!("✓ cuBLASLt FP8 context created\n");
            ctx
        }
        Err(e) => {
            println!("✗ cuBLASLt FP8 creation failed: {:?}", e);
            return;
        }
    };

    // Test configurations: (m, n, k, description)
    // M must be divisible by 16 for FP8
    let configs = vec![
        (256, 256, 256, "256x256x256 (small)"),
        (1024, 1024, 1024, "1Kx1Kx1K"),
        (2048, 2048, 2048, "2Kx2Kx2K"),
        (4096, 4096, 4096, "4Kx4Kx4K (large)"),
        (16, 16, 16, "16x16x16 (quantum gate)"),
        (256, 16, 256, "256x16x256 (batched gates)"),
    ];

    println!("Benchmarking FP8 E4M3 GEMM (D = A * B)\n");
    println!("┌──────────────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│        Configuration         │  Time (ms)   │    TFLOPS    │  Efficiency  │");
    println!("├──────────────────────────────┼──────────────┼──────────────┼──────────────┤");

    let fp8_theoretical = 838.0; // RTX 5090 FP8 theoretical TFLOPS

    for (m, n, k, desc) in configs {
        match cublaslt.benchmark_fp8_gemm(&rt, m, n, k, 100) {
            Ok((time_ms, tflops)) => {
                let efficiency = (tflops / fp8_theoretical) * 100.0;
                println!(
                    "│ {:28} │ {:12.4} │ {:12.2} │ {:10.1}%  │",
                    desc, time_ms, tflops, efficiency
                );
            }
            Err(e) => {
                println!("│ {:28} │ ERROR: {:?}", desc, e);
            }
        }
    }

    println!("└──────────────────────────────┴──────────────┴──────────────┴──────────────┘");

    println!("\n📊 Performance Analysis:");
    println!("   RTX 5090 FP8 Theoretical: {} TFLOPS", fp8_theoretical);
    println!("   RTX 5090 FP16 Theoretical: 209.5 TFLOPS");
    println!("   Expected FP8/FP16 Speedup: 4.0x");
    println!("\n   Efficiency Interpretation:");
    println!("   - >70%: Excellent - hitting FP8 tensor cores");
    println!("   - 40-70%: Good - some overhead but using FP8");
    println!("   - <40%: Poor - may not be using FP8 tensor cores");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example benchmark_cublaslt_fp8 --features cuda,perf-bench --release"
    );
}
