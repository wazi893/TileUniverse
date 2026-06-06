//! Cross-Block CNOT GPU Benchmark (Sprint 72.0)
//!
//! Benchmarks the multi-pass GPU implementation of cross-block CNOT operations.
//!
//! Run with: cargo run --release --example cross_block_cnot_gpu_bench --features gpu-prototype


#[cfg(feature = "gpu-prototype")]
use engine::tile8::sparse_quantum_gpu::SparseQuantumGpu;

fn format_throughput(ops: usize, duration: std::time::Duration) -> String {
    let secs = duration.as_secs_f64();
    let ops_per_sec = ops as f64 / secs;
    if ops_per_sec >= 1e9 {
        format!("{:.2}B ops/sec", ops_per_sec / 1e9)
    } else if ops_per_sec >= 1e6 {
        format!("{:.2}M ops/sec", ops_per_sec / 1e6)
    } else if ops_per_sec >= 1e3 {
        format!("{:.2}K ops/sec", ops_per_sec / 1e3)
    } else {
        format!("{:.2} ops/sec", ops_per_sec)
    }
}

#[cfg(feature = "gpu-prototype")]
fn main() {
    println!("================================================================================");
    println!("              CROSS-BLOCK CNOT GPU BENCHMARK (Sprint 72.0)                      ");
    println!("================================================================================");
    println!();

    // Test configurations: (n_qubits, num_blocks)
    let configs = [
        (256, 2),     // 8 qubits worth of blocks
        (1024, 8),    // 10 qubits
        (4096, 32),   // 12 qubits
        (16384, 128), // 14 qubits
        (65536, 512), // 16 qubits
    ];

    let iterations = 100;
    let warmup = 10;

    for (n_qubits, expected_blocks) in configs {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "Configuration: {} amplitudes ({} blocks, ~{} qubits)",
            n_qubits,
            expected_blocks,
            (n_qubits as f64).log2() as u32
        );
        println!(
            "--------------------------------------------------------------------------------"
        );

        let gpu = pollster::block_on(SparseQuantumGpu::new(n_qubits));
        gpu.init_zero();
        gpu.sync();

        let num_blocks = gpu.num_blocks();
        println!("  Actual blocks: {}", num_blocks);
        println!();

        // ============================================================
        // Test 1: Within-Block CNOT (baseline)
        // ============================================================
        println!("  Test 1: Within-Block CNOT (control=0, target=1)");

        // Warmup
        for _ in 0..warmup {
            gpu.cnot(0, 1);
        }
        gpu.sync();

        let start = Instant::now();
        for _ in 0..iterations {
            gpu.cnot(0, 1);
        }
        gpu.sync();
        let within_block_time = start.elapsed();

        println!(
            "    Time: {:?} for {} iterations",
            within_block_time, iterations
        );
        println!("    Avg: {:?}/op", within_block_time / iterations as u32);
        println!(
            "    Throughput: {}",
            format_throughput(iterations * num_blocks as usize, within_block_time)
        );
        println!();

        // ============================================================
        // Test 2: Control-High CNOT (control >= 7, target < 7)
        // ============================================================
        if num_blocks >= 2 {
            println!("  Test 2: Control-High CNOT (control=7, target=0)");

            // Warmup
            for _ in 0..warmup {
                gpu.cnot_control_high(7, 0);
            }
            gpu.sync();

            let start = Instant::now();
            for _ in 0..iterations {
                gpu.cnot_control_high(7, 0);
            }
            gpu.sync();
            let control_high_time = start.elapsed();

            println!(
                "    Time: {:?} for {} iterations",
                control_high_time, iterations
            );
            println!("    Avg: {:?}/op", control_high_time / iterations as u32);
            println!(
                "    Throughput: {}",
                format_throughput(iterations * num_blocks as usize, control_high_time)
            );
            println!(
                "    vs Within-Block: {:.2}x",
                within_block_time.as_secs_f64() / control_high_time.as_secs_f64()
            );
            println!();
        }

        // ============================================================
        // Test 3: Target-High CNOT (control < 7, target >= 7)
        // ============================================================
        if num_blocks >= 2 {
            println!("  Test 3: Target-High CNOT (control=0, target=7)");

            // Warmup
            for _ in 0..warmup {
                gpu.cnot_target_high(0, 7);
            }
            gpu.sync();

            let start = Instant::now();
            for _ in 0..iterations {
                gpu.cnot_target_high(0, 7);
            }
            gpu.sync();
            let target_high_time = start.elapsed();

            println!(
                "    Time: {:?} for {} iterations",
                target_high_time, iterations
            );
            println!("    Avg: {:?}/op", target_high_time / iterations as u32);
            println!(
                "    Throughput: {}",
                format_throughput(iterations * num_blocks as usize, target_high_time)
            );
            println!(
                "    vs Within-Block: {:.2}x",
                within_block_time.as_secs_f64() / target_high_time.as_secs_f64()
            );
            println!();
        }

        // ============================================================
        // Test 4: Both-High CNOT (both >= 7)
        // ============================================================
        if num_blocks >= 4 {
            println!("  Test 4: Both-High CNOT (control=7, target=8)");

            // Warmup
            for _ in 0..warmup {
                gpu.cnot_both_high(7, 8);
            }
            gpu.sync();

            let start = Instant::now();
            for _ in 0..iterations {
                gpu.cnot_both_high(7, 8);
            }
            gpu.sync();
            let both_high_time = start.elapsed();

            println!(
                "    Time: {:?} for {} iterations",
                both_high_time, iterations
            );
            println!("    Avg: {:?}/op", both_high_time / iterations as u32);
            println!(
                "    Throughput: {}",
                format_throughput(iterations * num_blocks as usize, both_high_time)
            );
            println!(
                "    vs Within-Block: {:.2}x",
                within_block_time.as_secs_f64() / both_high_time.as_secs_f64()
            );
            println!();
        }

        // ============================================================
        // Test 5: GHZ State Creation with Cross-Block CNOT
        // ============================================================
        if num_blocks >= 4 {
            println!("  Test 5: Cross-Block GHZ State (qubits 0-9 via cnot_any)");

            let ghz_qubits = 10.min((num_blocks as f64).log2() as u32 + 7);

            let start = Instant::now();
            for _ in 0..iterations {
                gpu.init_zero();
                gpu.hadamard(0);
                for q in 1..ghz_qubits {
                    gpu.cnot_any(0, q);
                }
            }
            gpu.sync();
            let ghz_time = start.elapsed();

            println!("    {} qubits GHZ ({} gates)", ghz_qubits, ghz_qubits);
            println!("    Time: {:?} for {} iterations", ghz_time, iterations);
            println!("    Avg: {:?}/GHZ creation", ghz_time / iterations as u32);
            println!();
        }
    }

    // ============================================================
    // Large Scale Test
    // ============================================================
    println!("================================================================================");
    println!("                         LARGE SCALE TEST                                       ");
    println!("================================================================================");
    println!();

    let large_blocks = 65535; // Max single dispatch
    let large_qubits = large_blocks * 128;

    println!(
        "Testing with {} blocks ({} amplitudes, ~{} effective qubits)",
        large_blocks,
        large_qubits,
        7 + (large_blocks as f64).log2() as u32
    );

    let gpu = pollster::block_on(SparseQuantumGpu::new(large_qubits));
    gpu.init_zero();
    gpu.sync();

    let iterations = 10;

    // Control-High at scale
    println!();
    println!("  Control-High CNOT at {} blocks:", large_blocks);
    let start = Instant::now();
    for _ in 0..iterations {
        gpu.cnot_control_high(7, 0);
    }
    gpu.sync();
    let time = start.elapsed();
    println!("    Time: {:?} for {} iterations", time, iterations);
    println!(
        "    Throughput: {}",
        format_throughput(iterations * large_blocks as usize, time)
    );

    // Target-High at scale
    println!();
    println!("  Target-High CNOT at {} blocks:", large_blocks);
    let start = Instant::now();
    for _ in 0..iterations {
        gpu.cnot_target_high(0, 7);
    }
    gpu.sync();
    let time = start.elapsed();
    println!("    Time: {:?} for {} iterations", time, iterations);
    println!(
        "    Throughput: {}",
        format_throughput(iterations * large_blocks as usize, time)
    );

    // Both-High at scale
    println!();
    println!("  Both-High CNOT at {} blocks:", large_blocks);
    let start = Instant::now();
    for _ in 0..iterations {
        gpu.cnot_both_high(7, 8);
    }
    gpu.sync();
    let time = start.elapsed();
    println!("    Time: {:?} for {} iterations", time, iterations);
    println!(
        "    Throughput: {}",
        format_throughput(iterations * large_blocks as usize, time)
    );

    println!();
    println!("================================================================================");
    println!("                              SUMMARY                                           ");
    println!("================================================================================");
    println!();
    println!("  Cross-Block CNOT GPU Implementation:");
    println!("    - Control-High: Parallel per-block (control bit in block ID)");
    println!("    - Target-High: Block pair swap with conditional (control bit local)");
    println!("    - Both-High: Full block pair swap (both bits in block ID)");
    println!();
    println!("  All operations use single-pass GPU kernels with filtering in shader.");
    println!("  No CPU-side pair enumeration or multi-dispatch required.");
    println!();
}

#[cfg(not(feature = "gpu-prototype"))]
fn main() {
    println!("================================================================================");
    println!("              CROSS-BLOCK CNOT GPU BENCHMARK (Sprint 72.0)                      ");
    println!("================================================================================");
    println!();
    println!("ERROR: This benchmark requires the gpu-prototype feature.");
    println!(
        "Run with: cargo run --release --example cross_block_cnot_gpu_bench --features gpu-prototype"
    );
}
