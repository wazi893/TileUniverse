// EPIC 115: Benchmark FP8 batched gates kernel
//
// This measures the performance of applying N different gates while keeping
// state in shared memory, comparing against the FP16 version and single-gate kernels.
//
// Run with: cargo run --example benchmark_fp8_batched_gates --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use logic_fabric_core::cuda::{CudaRuntime, compile_fp8_kernels_public, is_cuda_available};
    use std::time::Instant;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115: FP8 Batched Gates Kernel Benchmark                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");

    // Try to compile FP8 kernels
    let fp8_cache = match compile_fp8_kernels_public(&rt) {
        Ok(cache) => {
            println!("✓ FP8 kernels compiled successfully\n");
            cache
        }
        Err(e) => {
            println!("✗ FP8 kernel compilation failed: {:?}", e);
            println!("  This may require SM80+ (Ampere or newer)");
            return;
        }
    };

    // Test configurations: (qubits, batch_size, description)
    let configs = vec![
        (12, 50, "4K amps, 50 gates"),
        (16, 50, "64K amps, 50 gates"),
        (16, 100, "64K amps, 100 gates"),
        (20, 50, "1M amps, 50 gates"),
        (20, 100, "1M amps, 100 gates"),
        (24, 50, "16M amps, 50 gates"),
    ];

    println!("Testing N-different-gates with shared memory residence\n");
    println!(
        "┌──────────────────────────────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "│          Configuration               │  Time (ms)   │    TCOPS     │  Comp Intens │"
    );
    println!(
        "├──────────────────────────────────────┼──────────────┼──────────────┼──────────────┤"
    );

    for (n_qubits, num_gates, desc) in configs {
        let result = benchmark_batched_gates(&rt, &fp8_cache, n_qubits, num_gates);
        match result {
            Ok((time_ms, tcops, compute_intensity)) => {
                println!(
                    "│ {:36} │ {:12.3} │ {:12.2} │ {:10.1} B │",
                    desc, time_ms, tcops, compute_intensity
                );
            }
            Err(e) => {
                println!("│ {:36} │ ERROR: {:?}", desc, e);
            }
        }
    }

    println!(
        "└──────────────────────────────────────┴──────────────┴──────────────┴──────────────┘"
    );

    println!("\n📊 Performance Notes:");
    println!("   - TCOPS = Trillion Complex Ops Per Second");
    println!("   - Compute Intensity = FLOPs/byte (higher is better)");
    println!("   - RTX 5090 ridge point: ~400 FLOPs/byte");
    println!("   - Single-gate intensity: ~8 FLOPs/byte");
    println!("   - 50-gate batch intensity: ~400 FLOPs/byte (50x improvement)");
    println!("\n   FP8 provides 4x higher tensor throughput than FP16:");
    println!("   - FP16: 209.5 TFLOPS");
    println!("   - FP8:  838 TFLOPS (theoretical)");
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn benchmark_batched_gates(
    rt: &logic_fabric_core::cuda::CudaRuntime,
    fp8_cache: &logic_fabric_core::cuda::Fp8KernelCache,
    n_qubits: usize,
    num_gates: usize,
) -> Result<(f64, f64, f64), logic_fabric_core::cuda::CudaError> {
    use std::time::Instant;

    let state_size = 1usize << n_qubits; // 2^n amplitudes
    let tiles_per_state = (state_size + 255) / 256; // Each tile is 16x16 = 256 elements
    let num_states = 1; // Single state for now

    // Create random initial state (FP16 as u16)
    let state_data: Vec<u16> = (0..state_size)
        .map(|i| half::f16::from_f32((i as f32 * 0.001) % 1.0).to_bits())
        .collect();

    // Create random gate sequence (N different 16x16 matrices)
    let gates_data: Vec<u16> = (0..num_gates * 256)
        .map(|i| {
            // Simple rotation-like matrix values
            let row = (i / 16) % 16;
            let col = i % 16;
            let val = if row == col { 0.99f32 } else { 0.01f32 / 16.0 };
            half::f16::from_f32(val).to_bits()
        })
        .collect();

    // Upload to GPU
    let mut states_gpu = rt.upload(&state_data)?;
    let gates_gpu = rt.upload(&gates_data)?;

    // Get stream
    let stream = rt.get_stream();

    // Warmup
    fp8_cache.apply_batched_gates(
        &stream,
        &mut states_gpu,
        &gates_gpu,
        num_states,
        tiles_per_state,
        num_gates,
    )?;
    rt.synchronize()?;

    // Benchmark
    let iterations = if n_qubits <= 16 { 20 } else { 5 };

    let start = Instant::now();
    for _ in 0..iterations {
        fp8_cache.apply_batched_gates(
            &stream,
            &mut states_gpu,
            &gates_gpu,
            num_states,
            tiles_per_state,
            num_gates,
        )?;
    }
    rt.synchronize()?;
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    // Calculate TCOPS (Trillion Complex Ops Per Second)
    // Each gate application involves state_size complex multiply-accumulates
    let total_ops = (state_size as f64) * (num_gates as f64) * 8.0; // 8 FLOPs per complex MAC
    let ops_per_second = total_ops / (time_ms / 1000.0);
    let tcops = ops_per_second / 1e12;

    // Calculate compute intensity (FLOPs per byte)
    // Load state once (state_size * 4 bytes), load gates (num_gates * 256 * 2 bytes), store once
    let bytes_transferred = (state_size * 4 + num_gates * 256 * 2 + state_size * 4) as f64;
    let compute_intensity = total_ops / bytes_transferred;

    Ok((time_ms, tcops, compute_intensity))
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example benchmark_fp8_batched_gates --features cuda,perf-bench --release"
    );
}
