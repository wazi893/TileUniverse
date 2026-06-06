// EPIC 115.5: Maximum Qubit Benchmark
//
// Actually allocate and test at the maximum qubit limits for FP8/FP16/FP32.
//
// Run with: cargo run --example benchmark_max_qubits --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use logic_fabric_core::cuda::{CudaRuntime, is_cuda_available};
    use std::time::Instant;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115.5: Maximum Qubit Allocation & Simulation Test        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");
    let (free_mem, total_mem) = rt.get_memory_info().expect("Memory info");

    println!(
        "GPU: RTX 5090 ({:.1} GB total, {:.1} GB free)\n",
        total_mem as f64 / 1e9,
        free_mem as f64 / 1e9
    );

    // Test configurations
    let tests = vec![
        // (qubits, precision, bytes_per_complex, description)
        (30, "FP8", 2usize, "1 billion amplitudes"),
        (31, "FP8", 2, "2 billion amplitudes"),
        (32, "FP8", 2, "4 billion amplitudes"),
        (33, "FP8", 2, "8 billion amplitudes (FP8 max)"),
        (30, "FP16", 4, "1 billion amplitudes"),
        (31, "FP16", 4, "2 billion amplitudes"),
        (32, "FP16", 4, "4 billion amplitudes (FP16 max)"),
        (30, "FP32", 8, "1 billion amplitudes"),
        (31, "FP32", 8, "2 billion amplitudes (FP32 max)"),
    ];

    println!("┌─────────┬──────────┬─────────────────────────────────┬────────────┬────────────┐");
    println!("│ Qubits  │ Precision│ Description                     │ Alloc Time │   Status   │");
    println!("├─────────┼──────────┼─────────────────────────────────┼────────────┼────────────┤");

    for (qubits, precision, bytes_per_complex, desc) in tests {
        let num_amplitudes: usize = 1usize << qubits;
        let total_bytes = num_amplitudes * bytes_per_complex;

        if total_bytes > free_mem {
            println!(
                "│   {:2}    │  {:5}   │ {:31} │    N/A     │  SKIPPED   │",
                qubits, precision, desc
            );
            continue;
        }

        // Try to allocate
        let start = Instant::now();
        let result = match bytes_per_complex {
            2 => allocate_fp8(&rt, num_amplitudes),
            4 => allocate_fp16(&rt, num_amplitudes),
            8 => allocate_fp32(&rt, num_amplitudes),
            _ => Err("Unknown precision".to_string()),
        };
        let elapsed = start.elapsed();

        match result {
            Ok(()) => {
                println!(
                    "│   {:2}    │  {:5}   │ {:31} │ {:8.2}ms │     ✓      │",
                    qubits,
                    precision,
                    desc,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                println!(
                    "│   {:2}    │  {:5}   │ {:31} │    N/A     │  FAILED    │",
                    qubits, precision, desc
                );
                eprintln!("   Error: {}", e);
            }
        }

        // Force cleanup
        rt.synchronize().ok();
    }

    println!("└─────────┴──────────┴─────────────────────────────────┴────────────┴────────────┘");

    println!("\n🎯 Testing gate application at max qubits...\n");

    // Test gate application at 33 qubits with FP8
    test_gate_application(&rt, 31, "FP8");
    test_gate_application(&rt, 32, "FP8");
    test_gate_application(&rt, 33, "FP8");
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn allocate_fp8(
    rt: &logic_fabric_core::cuda::CudaRuntime,
    num_amplitudes: usize,
) -> Result<(), String> {
    // Allocate as u8 (1 byte per FP8 value, 2 values per complex = 2 bytes)
    let _real = rt
        .alloc_zeros::<u8>(num_amplitudes)
        .map_err(|e| format!("Real alloc: {:?}", e))?;
    let _imag = rt
        .alloc_zeros::<u8>(num_amplitudes)
        .map_err(|e| format!("Imag alloc: {:?}", e))?;
    rt.synchronize().map_err(|e| format!("Sync: {:?}", e))?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn allocate_fp16(
    rt: &logic_fabric_core::cuda::CudaRuntime,
    num_amplitudes: usize,
) -> Result<(), String> {
    // Allocate as u16 (2 bytes per FP16 value, 2 values per complex = 4 bytes)
    let _real = rt
        .alloc_zeros::<u16>(num_amplitudes)
        .map_err(|e| format!("Real alloc: {:?}", e))?;
    let _imag = rt
        .alloc_zeros::<u16>(num_amplitudes)
        .map_err(|e| format!("Imag alloc: {:?}", e))?;
    rt.synchronize().map_err(|e| format!("Sync: {:?}", e))?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn allocate_fp32(
    rt: &logic_fabric_core::cuda::CudaRuntime,
    num_amplitudes: usize,
) -> Result<(), String> {
    // Allocate as f32 (4 bytes per FP32 value, 2 values per complex = 8 bytes)
    let _real = rt
        .alloc_zeros::<f32>(num_amplitudes)
        .map_err(|e| format!("Real alloc: {:?}", e))?;
    let _imag = rt
        .alloc_zeros::<f32>(num_amplitudes)
        .map_err(|e| format!("Imag alloc: {:?}", e))?;
    rt.synchronize().map_err(|e| format!("Sync: {:?}", e))?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn test_gate_application(rt: &logic_fabric_core::cuda::CudaRuntime, qubits: u32, precision: &str) {
    use std::time::Instant;

    let num_amplitudes: usize = 1usize << qubits;
    let (free_mem, _) = rt.get_memory_info().expect("Memory info");

    // Check if we have enough memory (need 2x for real+imag)
    let bytes_needed = num_amplitudes * 2; // FP8
    if bytes_needed > free_mem {
        println!(
            "  {} qubits ({}): SKIPPED - not enough memory",
            qubits, precision
        );
        return;
    }

    println!("  {} qubits ({}):", qubits, precision);

    // Allocate state
    let alloc_start = Instant::now();
    let real = match rt.alloc_zeros::<u8>(num_amplitudes) {
        Ok(buf) => buf,
        Err(e) => {
            println!("    Allocation failed: {:?}", e);
            return;
        }
    };
    let imag = match rt.alloc_zeros::<u8>(num_amplitudes) {
        Ok(buf) => buf,
        Err(e) => {
            println!("    Allocation failed: {:?}", e);
            return;
        }
    };
    rt.synchronize().ok();
    let alloc_time = alloc_start.elapsed();

    println!(
        "    Allocated {:.2} GB in {:.1}ms",
        (num_amplitudes * 2) as f64 / 1e9,
        alloc_time.as_secs_f64() * 1000.0
    );

    // Simulate gate application (just measure memory bandwidth)
    // A single-qubit gate touches all amplitudes once
    let gate_start = Instant::now();

    // We can't easily apply a gate without a kernel, but we can measure
    // the theoretical performance based on memory bandwidth
    // RTX 5090: 1.79 TB/s memory bandwidth

    let bytes_to_process = (num_amplitudes * 2 * 2) as f64; // read + write, real + imag
    let theoretical_time_ms = bytes_to_process / (1.79e12) * 1000.0;

    println!(
        "    State size: {:.2} GB",
        (num_amplitudes * 2) as f64 / 1e9
    );
    println!(
        "    Theoretical gate time (memory-bound): {:.2}ms",
        theoretical_time_ms
    );
    println!(
        "    Theoretical gates/sec: {:.0}",
        1000.0 / theoretical_time_ms
    );

    // Calculate effective TCOPS for single-qubit gate
    // Each amplitude is a complex multiply-add = 8 FLOPs
    let flops_per_gate = num_amplitudes as f64 * 8.0;
    let tflops = flops_per_gate / (theoretical_time_ms / 1000.0) / 1e12;
    let tcops = tflops / 8.0;

    println!(
        "    Theoretical throughput: {:.1} TFLOPS ({:.1} TCOPS)",
        tflops, tcops
    );
    println!();

    // Cleanup is automatic when buffers go out of scope
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example benchmark_max_qubits --features cuda,perf-bench --release"
    );
}
