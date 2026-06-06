// EPIC 115.2: Benchmark Native FP8 Tensor Core Performance
//
// This tests ACTUAL FP8 tensor core instructions via inline PTX.
// Expected 4x throughput improvement over FP16 WMMA (838 TFLOPS vs 209.5 TFLOPS)
//
// Run with: cargo run --example benchmark_native_fp8 --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use logic_fabric_core::cuda::{CudaRuntime, compile_fp8_kernels_public, is_cuda_available};

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115.2: Native FP8 Tensor Core Benchmark                  ║");
    println!("║  Target: 838 TFLOPS (4x over FP16's 209.5 TFLOPS)              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");

    // Try to compile FP8 kernels
    let fp8_cache = match compile_fp8_kernels_public(&rt) {
        Ok(cache) => cache,
        Err(e) => {
            println!("✗ FP8 kernel compilation failed: {:?}", e);
            return;
        }
    };

    // Check for native FP8 support
    if !fp8_cache.has_native_fp8() {
        println!("✗ Native FP8 tensor cores NOT available on this hardware.");
        println!("  Requires SM89 (Ada Lovelace) or SM120 (Blackwell).");
        println!("\n  Your GPU is using FP16 tensor cores with FP8 simulation.");
        println!("  Current performance: ~185 TFLOPS (88% of FP16 theoretical)");
        println!("  Potential with native FP8: ~750+ TFLOPS (90% of FP8 theoretical)");
        return;
    }

    println!("✓ Native FP8 tensor core support detected!\n");

    // Benchmark configurations: (num_warps, iterations_per_warp, description)
    // We need many warps to saturate all tensor cores
    // RTX 5090: 170 SMs × 4 warps each = 680 concurrent warps minimum
    // More warps and iterations = better GPU utilization
    let configs = vec![
        (8192, 1000, "8K warps × 1K iter"),
        (16384, 1000, "16K warps × 1K iter"),
        (32768, 1000, "32K warps × 1K iter"),
        (16384, 10000, "16K warps × 10K iter"),
        (32768, 10000, "32K warps × 10K iter"),
        (32768, 100000, "32K warps × 100K iter"),
    ];

    println!("Benchmarking mma.sync.m16n8k32.f32.e4m3.e4m3.f32 (multi-warp)\n");
    println!("┌──────────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│      Configuration       │  Time (ms)   │    TFLOPS    │  Efficiency  │");
    println!("├──────────────────────────┼──────────────┼──────────────┼──────────────┤");

    let fp8_theoretical = 838.0; // RTX 5090 FP8 theoretical TFLOPS

    for (num_warps, iterations, desc) in configs {
        match fp8_cache.benchmark_native_fp8(&rt, num_warps, iterations) {
            Ok((time_ms, tflops)) => {
                let efficiency = (tflops / fp8_theoretical) * 100.0;
                println!(
                    "│ {:24} │ {:12.3} │ {:12.2} │ {:10.1}%  │",
                    desc, time_ms, tflops, efficiency
                );
            }
            Err(e) => {
                println!("│ {:24} │ ERROR: {:?}", desc, e);
            }
        }
    }

    println!("└──────────────────────────┴──────────────┴──────────────┴──────────────┘");

    println!("\n📊 Analysis:");
    println!("   FP8 Theoretical Peak: {} TFLOPS", fp8_theoretical);
    println!("   FP16 Theoretical Peak: 209.5 TFLOPS");
    println!("   Expected Speedup: 4x over FP16");
    println!("\n   If efficiency > 70%: Native FP8 is working correctly");
    println!("   If efficiency < 30%: May be falling back to FP16 emulation");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example benchmark_native_fp8 --features cuda,perf-bench --release"
    );
}
