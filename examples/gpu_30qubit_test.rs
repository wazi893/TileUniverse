//! 30-Qubit GPU GHZ State Benchmark
//!
//! Compares different optimization strategies for cross-block CNOT operations:
//! 1. Original (v1): 75% wasted threads, sparse access
//! 2. Compacted (v2): 4x fewer threads, all do useful work
//! 3. Shared memory (v3): Coalesced loads via shared memory staging
//! 4. Cascade: Adjacent qubit CNOT chain for memory locality
//! 5. Cascade+SharedMem: Cascade with shared memory optimization
//! 6. Fused: Direct state computation (GHZ-specific, theoretical max)
//!
//! Usage: cargo run --release --features cuda --example gpu_30qubit_test

use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
use logic_fabric_core::cuda::CudaRuntime;
use std::panic;

fn benchmark(
    name: &str,
    grid: &mut QuantumGridF64Gpu,
    f: impl Fn(&mut QuantumGridF64Gpu) -> u64,
) -> f64 {
    // Warmup
    let _ = f(grid);

    // Benchmark
    let mut times = Vec::with_capacity(5);
    for _ in 0..5 {
        let time_us = f(grid);
        times.push(time_us as f64 / 1_000_000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  {:30} avg: {:>7.3}s  min: {:>7.3}s  max: {:>7.3}s",
        name, avg, min, max
    );

    avg
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║      30-Qubit GPU Cross-Block CNOT Optimization Benchmark        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = CudaRuntime::new().expect("CUDA init failed");

    if let Ok(name) = rt.device_name() {
        println!("GPU: {}", name);
    }
    if let Ok((major, minor)) = rt.get_compute_capability() {
        println!("Compute Capability: {}.{}", major, minor);
    }

    let n_qubits = 30;
    let num_amplitudes = 1u64 << n_qubits;
    let memory_gb = num_amplitudes * 16 / 1024 / 1024 / 1024;

    println!("\nConfiguration:");
    println!("  Qubits: {}", n_qubits);
    println!("  Amplitudes: {:.2}B", num_amplitudes as f64 / 1e9);
    println!("  Memory: {} GB", memory_gb);
    println!("  Cross-block CNOTs: {} (qubits 7-29)", n_qubits - 7);

    println!("\nAllocating GPU memory...");
    let mut grid = match QuantumGridF64Gpu::new(&rt, n_qubits) {
        Ok(g) => g,
        Err(e) => {
            println!("ERROR: {:?}", e);
            return;
        }
    };

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  GHZ State Creation Benchmark (H + 29 CNOTs)                    │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Run benchmarks
    let t_v1 = benchmark("Original (v1) - 75% waste", &mut grid, |g| {
        g.create_ghz_state().unwrap()
    });

    // Verify correctness
    let v1_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    let t_v2 = benchmark("Compacted (v2) - 4x fewer threads", &mut grid, |g| {
        g.create_ghz_state_v2().unwrap()
    });

    let v2_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    let t_v3 = benchmark("Shared memory (v3) - coalesced", &mut grid, |g| {
        g.create_ghz_state_v3().unwrap()
    });

    let v3_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    // Cascade pattern: CNOT(i,i+1) instead of CNOT(0,i) for better locality
    let t_cascade = benchmark("Cascade - adjacent qubit chain", &mut grid, |g| {
        g.create_ghz_state_cascade().unwrap()
    });

    let cascade_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    let t_cascade_shared = benchmark("Cascade + shared memory", &mut grid, |g| {
        g.create_ghz_state_cascade_shared().unwrap()
    });

    let cascade_shared_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    // GEMM-based approach using cuBLAS DGEMM for matrix multiply (4x4)
    let t_gemm = benchmark("GEMM 4x4 (cuBLAS)", &mut grid, |g| {
        g.create_ghz_state_gemm(&rt).unwrap()
    });

    let gemm_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    // Fused 128x128 GEMM - combines H + 6 CNOTs into single matrix multiply
    let t_gemm_fused = benchmark("GEMM 128x128 F64 (cuBLAS)", &mut grid, |g| {
        g.create_ghz_state_fused_gemm(&rt).unwrap()
    });

    let gemm_fused_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    // FP16 Tensor Core GEMM - trades precision for speed
    // Note: cudarc's hgemm panics if library loading fails, so we use catch_unwind
    let (t_gemm_fp16, gemm_fp16_fidelity) = {
        // Use AssertUnwindSafe since we're careful to not use grid in panic state
        let grid_ptr: *mut QuantumGridF64Gpu = &mut grid;
        let rt_ptr: *const CudaRuntime = &rt;

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            let grid = &mut *grid_ptr;
            let rt = &*rt_ptr;
            match grid.create_ghz_state_fused_gemm_fp16(rt) {
                Ok(_) => {
                    let t = benchmark("GEMM 128x128 FP16 TensorCore", grid, |g| {
                        g.create_ghz_state_fused_gemm_fp16(rt).unwrap()
                    });
                    let f = grid.verify_ghz_fidelity().unwrap().fidelity;
                    Ok((t, f))
                }
                Err(e) => Err(format!("{:?}", e)),
            }
        }));

        match result {
            Ok(Ok((t, f))) => (t, f),
            Ok(Err(e)) => {
                println!("  GEMM 128x128 FP16 TensorCore   SKIPPED: {}", e);
                (0.0, 0.0)
            }
            Err(_) => {
                println!("  GEMM 128x128 FP16 TensorCore   SKIPPED: hgemm library not available");
                (0.0, 0.0)
            }
        }
    };

    // FP16 256x256 Tensor Core GEMM - H + 7 CNOTs fused
    let (t_gemm_fp16_256, gemm_fp16_256_fidelity) = {
        let grid_ptr: *mut QuantumGridF64Gpu = &mut grid;
        let rt_ptr: *const CudaRuntime = &rt;

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            let grid = &mut *grid_ptr;
            let rt = &*rt_ptr;
            match grid.create_ghz_state_fused_gemm_fp16_256(rt) {
                Ok(_) => {
                    let t = benchmark("GEMM 256x256 FP16 TensorCore", grid, |g| {
                        g.create_ghz_state_fused_gemm_fp16_256(rt).unwrap()
                    });
                    let f = grid.verify_ghz_fidelity().unwrap().fidelity;
                    Ok((t, f))
                }
                Err(e) => Err(format!("{:?}", e)),
            }
        }));

        match result {
            Ok(Ok((t, f))) => (t, f),
            Ok(Err(e)) => {
                println!("  GEMM 256x256 FP16 TensorCore   SKIPPED: {}", e);
                (0.0, 0.0)
            }
            Err(_) => {
                println!("  GEMM 256x256 FP16 TensorCore   SKIPPED: hgemm library not available");
                (0.0, 0.0)
            }
        }
    };

    // FP16 512x512 Tensor Core GEMM - H + 8 CNOTs fused
    let (t_gemm_fp16_512, gemm_fp16_512_fidelity) = {
        let grid_ptr: *mut QuantumGridF64Gpu = &mut grid;
        let rt_ptr: *const CudaRuntime = &rt;

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            let grid = &mut *grid_ptr;
            let rt = &*rt_ptr;
            match grid.create_ghz_state_fused_gemm_fp16_512(rt) {
                Ok(_) => {
                    let t = benchmark("GEMM 512x512 FP16 TensorCore", grid, |g| {
                        g.create_ghz_state_fused_gemm_fp16_512(rt).unwrap()
                    });
                    let f = grid.verify_ghz_fidelity().unwrap().fidelity;
                    Ok((t, f))
                }
                Err(e) => Err(format!("{:?}", e)),
            }
        }));

        match result {
            Ok(Ok((t, f))) => (t, f),
            Ok(Err(e)) => {
                println!("  GEMM 512x512 FP16 TensorCore   SKIPPED: {}", e);
                (0.0, 0.0)
            }
            Err(_) => {
                println!("  GEMM 512x512 FP16 TensorCore   SKIPPED: hgemm library not available");
                (0.0, 0.0)
            }
        }
    };

    let t_fused = benchmark("Fused (direct computation)", &mut grid, |g| {
        g.create_ghz_state_fused().unwrap()
    });

    let fused_fidelity = grid.verify_ghz_fidelity().unwrap().fidelity;

    // Results table
    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                                 ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:30} │ {:>8} │ {:>8} │ {:>8} ║",
        "Method", "Time", "Speedup", "Fidelity"
    );
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>7} │ {:>8.6} ║",
        "Original (v1)", t_v1, "1.00x", v1_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "Compacted (v2)",
        t_v2,
        t_v1 / t_v2,
        v2_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "Shared memory (v3)",
        t_v3,
        t_v1 / t_v3,
        v3_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "Cascade (locality opt)",
        t_cascade,
        t_v1 / t_cascade,
        cascade_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "Cascade + shared mem",
        t_cascade_shared,
        t_v1 / t_cascade_shared,
        cascade_shared_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "GEMM 4x4 (cuBLAS)",
        t_gemm,
        t_v1 / t_gemm,
        gemm_fidelity
    );
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
        "GEMM 128x128 F64",
        t_gemm_fused,
        t_v1 / t_gemm_fused,
        gemm_fused_fidelity
    );
    if t_gemm_fp16 > 0.0 {
        println!(
            "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
            "GEMM 128x128 FP16 TC",
            t_gemm_fp16,
            t_v1 / t_gemm_fp16,
            gemm_fp16_fidelity
        );
    } else {
        println!(
            "║ {:30} │ {:>8} │ {:>8} │ {:>8} ║",
            "GEMM 128x128 FP16 TC", "N/A", "N/A", "N/A"
        );
    }
    if t_gemm_fp16_256 > 0.0 {
        println!(
            "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
            "GEMM 256x256 FP16 TC",
            t_gemm_fp16_256,
            t_v1 / t_gemm_fp16_256,
            gemm_fp16_256_fidelity
        );
    } else {
        println!(
            "║ {:30} │ {:>8} │ {:>8} │ {:>8} ║",
            "GEMM 256x256 FP16 TC", "N/A", "N/A", "N/A"
        );
    }
    if t_gemm_fp16_512 > 0.0 {
        println!(
            "║ {:30} │ {:>6.3}s  │ {:>6.2}x │ {:>8.6} ║",
            "GEMM 512x512 FP16 TC",
            t_gemm_fp16_512,
            t_v1 / t_gemm_fp16_512,
            gemm_fp16_512_fidelity
        );
    } else {
        println!(
            "║ {:30} │ {:>8} │ {:>8} │ {:>8} ║",
            "GEMM 512x512 FP16 TC", "N/A", "N/A", "N/A"
        );
    }
    println!(
        "║ {:30} │ {:>6.3}s  │ {:>6.1}x │ {:>8.6} ║",
        "Fused (theoretical max)",
        t_fused,
        t_v1 / t_fused,
        fused_fidelity
    );
    println!("╚═════════════════════════════════════════════════════════════════╝");

    // Memory bandwidth analysis
    let bytes_per_cnot = memory_gb as f64 * 2.0; // Read + write full state
    let num_cross_block = (n_qubits - 7) as f64;

    println!("\nMemory Bandwidth Analysis:");
    println!(
        "  Theoretical per cross-block CNOT: {:.0} GB",
        bytes_per_cnot
    );
    println!(
        "  Total for {} cross-block CNOTs: {:.0} GB",
        num_cross_block,
        bytes_per_cnot * num_cross_block
    );
    println!();
    println!(
        "  v1 effective: {:>7.1} GB/s",
        bytes_per_cnot * num_cross_block / t_v1
    );
    println!(
        "  v2 effective: {:>7.1} GB/s",
        bytes_per_cnot * num_cross_block / t_v2
    );
    println!(
        "  v3 effective: {:>7.1} GB/s",
        bytes_per_cnot * num_cross_block / t_v3
    );
    println!(
        "  Fused:        {:>7.1} GB/s (single pass)",
        memory_gb as f64 * 2.0 / t_fused
    );

    // Summary
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  OPTIMIZATION SUMMARY                                           │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!(
        "  v1 → v2:       {:>5.2}x speedup (compacted indexing)",
        t_v1 / t_v2
    );
    println!(
        "  v1 → v3:       {:>5.2}x speedup (shared memory + coalescing)",
        t_v1 / t_v3
    );
    println!(
        "  v1 → cascade:  {:>5.2}x speedup (adjacent qubit locality)",
        t_v1 / t_cascade
    );
    println!(
        "  v1 → casc+shm: {:>5.2}x speedup (cascade + shared memory)",
        t_v1 / t_cascade_shared
    );
    println!(
        "  v1 → GEMM 4x4: {:>5.2}x speedup (cuBLAS single gate)",
        t_v1 / t_gemm
    );
    println!(
        "  v1 → GEMM128:  {:>5.2}x speedup (cuBLAS 7 gates F64)",
        t_v1 / t_gemm_fused
    );
    if t_gemm_fp16 > 0.0 {
        println!(
            "  v1 → FP16 128: {:>5.2}x speedup (Tensor Core FP16 128)",
            t_v1 / t_gemm_fp16
        );
    } else {
        println!("  v1 → FP16 128: N/A (hgemm not available)");
    }
    if t_gemm_fp16_256 > 0.0 {
        println!(
            "  v1 → FP16 256: {:>5.2}x speedup (Tensor Core FP16 256)",
            t_v1 / t_gemm_fp16_256
        );
    } else {
        println!("  v1 → FP16 256: N/A (hgemm not available)");
    }
    if t_gemm_fp16_512 > 0.0 {
        println!(
            "  v1 → FP16 512: {:>5.2}x speedup (Tensor Core FP16 512)",
            t_v1 / t_gemm_fp16_512
        );
    } else {
        println!("  v1 → FP16 512: N/A (hgemm not available)");
    }
    println!(
        "  v1 → fused:    {:>5.1}x speedup (direct computation)",
        t_v1 / t_fused
    );

    // F64 methods need high fidelity, FP16 allows some precision loss
    let f64_ok = v1_fidelity > 0.999999
        && v2_fidelity > 0.999999
        && v3_fidelity > 0.999999
        && cascade_fidelity > 0.999999
        && cascade_shared_fidelity > 0.999999
        && gemm_fidelity > 0.999999
        && gemm_fused_fidelity > 0.999999
        && fused_fidelity > 0.999999;
    let fp16_ok = (t_gemm_fp16 == 0.0 || gemm_fp16_fidelity > 0.99)
        && (t_gemm_fp16_256 == 0.0 || gemm_fp16_256_fidelity > 0.99)
        && (t_gemm_fp16_512 == 0.0 || gemm_fp16_512_fidelity > 0.99);

    if f64_ok && fp16_ok {
        println!("\n✓ ALL METHODS VERIFIED CORRECT");
        if t_gemm_fp16 > 0.0 || t_gemm_fp16_256 > 0.0 || t_gemm_fp16_512 > 0.0 {
            println!("  FP16 fidelities (reduced precision expected):");
            if t_gemm_fp16 > 0.0 {
                println!("    128x128: {:.6}", gemm_fp16_fidelity);
            }
            if t_gemm_fp16_256 > 0.0 {
                println!("    256x256: {:.6}", gemm_fp16_256_fidelity);
            }
            if t_gemm_fp16_512 > 0.0 {
                println!("    512x512: {:.6}", gemm_fp16_512_fidelity);
            }
        }
    } else {
        println!("\n✗ FIDELITY CHECK FAILED!");
        println!("  v1: {}", v1_fidelity);
        println!("  v2: {}", v2_fidelity);
        println!("  v3: {}", v3_fidelity);
        println!("  cascade: {}", cascade_fidelity);
        println!("  cascade+shm: {}", cascade_shared_fidelity);
        println!("  GEMM 4x4: {}", gemm_fidelity);
        println!("  GEMM 128 F64: {}", gemm_fused_fidelity);
        if t_gemm_fp16 > 0.0 {
            println!("  GEMM 128 FP16: {}", gemm_fp16_fidelity);
        }
        if t_gemm_fp16_256 > 0.0 {
            println!("  GEMM 256 FP16: {}", gemm_fp16_256_fidelity);
        }
        if t_gemm_fp16_512 > 0.0 {
            println!("  GEMM 512 FP16: {}", gemm_fp16_512_fidelity);
        }
        println!("  fused: {}", fused_fidelity);
    }
}
