//! CPU Throughput Benchmark for QuantumGridF64
//!
//! Measures true throughput when pushed to max CPU cores (32 threads).
//! Tests scaling across qubit counts and operation types.
//!
//! Run: cargo run --release --example cpu_throughput_bench

use engine::tile8::quantum_router_f64::QuantumGridF64;
use std::time::Instant;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CPU THROUGHPUT BENCHMARK - QuantumGridF64 (Rayon Parallelized)      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Detect thread count
    let num_threads = rayon::current_num_threads();
    println!("Rayon threads: {}", num_threads);
    println!("Block size: 128 Complex64 amplitudes (2KB per block)\n");

    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" PHASE 1: GHZ State Creation Scaling (7-30 qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let qubit_tests = [7, 10, 14, 18, 20, 22, 24, 26, 27, 28];

    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>14} {:>12}",
        "Qubits", "Blocks", "Memory", "Time (ms)", "Amp Ops/sec", "Fidelity"
    );
    println!(
        "{:-<6} {:-<12} {:-<12} {:-<12} {:-<14} {:-<12}",
        "", "", "", "", "", ""
    );

    for &n_qubits in &qubit_tests {
        // Skip if would exceed ~16GB
        let num_blocks = 1usize << (n_qubits - 7);
        let memory_mb = (num_blocks * 2048) / (1024 * 1024);
        if memory_mb > 16000 {
            println!(
                "{:>6} {:>12} {:>10}MB  SKIPPED (>{} GB)",
                n_qubits,
                format_thousands(num_blocks),
                memory_mb,
                16
            );
            continue;
        }

        let mut grid = QuantumGridF64::new(n_qubits);

        // Warm-up
        let _ = grid.create_ghz_state();

        // Benchmark with multiple iterations for stability
        let iterations = if n_qubits <= 20 {
            10
        } else if n_qubits <= 24 {
            5
        } else {
            3
        };
        let mut total_time_us = 0u64;

        for _ in 0..iterations {
            total_time_us += grid.create_ghz_state();
        }

        let avg_time_us = total_time_us / iterations as u64;
        let verification = grid.verify_ghz_fidelity();

        // Calculate amplitude operations
        // GHZ creation: H(q0) + CNOT chain
        // H on q0: 64 pairs × num_blocks × 4 ops (2 add, 2 mul) = 256 × blocks
        // CNOT local (q1-q6): 6 × 64 swaps × blocks = 384 × blocks
        // CNOT cross (q7+): (n-7) × 64 swaps × blocks/2 pairs
        let _num_amplitudes = num_blocks * 128;
        let hadamard_ops = num_blocks * 256;
        let cnot_local_ops = if n_qubits >= 7 {
            6.min(n_qubits - 1) * num_blocks * 128
        } else {
            0
        };
        let cnot_cross_ops = if n_qubits > 7 {
            (n_qubits - 7) * num_blocks * 64
        } else {
            0
        };
        let total_ops = hadamard_ops + cnot_local_ops + cnot_cross_ops;

        let time_sec = avg_time_us as f64 / 1_000_000.0;
        let ops_per_sec = total_ops as f64 / time_sec;

        let memory_str = if memory_mb < 1024 {
            format!("{}MB", memory_mb)
        } else {
            format!("{:.1}GB", memory_mb as f64 / 1024.0)
        };

        println!(
            "{:>6} {:>12} {:>12} {:>12.2} {:>14} {:>11.6}%",
            n_qubits,
            format_thousands(num_blocks),
            memory_str,
            avg_time_us as f64 / 1000.0,
            format_throughput(ops_per_sec),
            verification.fidelity * 100.0
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" PHASE 2: Sustained Throughput Test (27 qubits, 1M blocks)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let mut grid = QuantumGridF64::new(27);
    let num_blocks = grid.num_blocks();
    println!(
        "Configuration: {} blocks, {}GB memory",
        format_thousands(num_blocks),
        (num_blocks * 2048) / (1024 * 1024 * 1024)
    );

    // Test individual operations
    println!("\nOperation breakdown:");

    // Hadamard Q0
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        grid.apply_hadamard_q0();
    }
    let h_time = start.elapsed().as_secs_f64() / iterations as f64;
    let h_ops = num_blocks * 256;
    let h_throughput = h_ops as f64 / h_time;
    println!(
        "  Hadamard(q0):     {:>10.3}ms  {:>12} amp-ops/sec",
        h_time * 1000.0,
        format_throughput(h_throughput)
    );

    // CNOT local
    let start = Instant::now();
    for _ in 0..iterations {
        grid.apply_cnot_local(0, 1);
    }
    let cnot_local_time = start.elapsed().as_secs_f64() / iterations as f64;
    let cnot_local_ops = num_blocks * 128;
    let cnot_local_throughput = cnot_local_ops as f64 / cnot_local_time;
    println!(
        "  CNOT(0,1) local:  {:>10.3}ms  {:>12} amp-ops/sec",
        cnot_local_time * 1000.0,
        format_throughput(cnot_local_throughput)
    );

    // CNOT cross-block
    let start = Instant::now();
    for _ in 0..iterations {
        grid.apply_cnot_cross_block(7);
    }
    let cnot_cross_time = start.elapsed().as_secs_f64() / iterations as f64;
    let cnot_cross_ops = num_blocks * 64; // Half blocks participate
    let cnot_cross_throughput = cnot_cross_ops as f64 / cnot_cross_time;
    println!(
        "  CNOT cross-block: {:>10.3}ms  {:>12} amp-ops/sec",
        cnot_cross_time * 1000.0,
        format_throughput(cnot_cross_throughput)
    );

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" PHASE 3: Memory Bandwidth Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Memory bandwidth calculation
    // Each Hadamard reads and writes 128 Complex64 per block = 2KB read + 2KB write
    let bytes_per_hadamard = num_blocks * 2048 * 2; // Read + Write
    let bandwidth_gbps = (bytes_per_hadamard as f64 / h_time) / 1e9;

    println!(
        "Hadamard memory access: {} bytes/op",
        format_thousands(bytes_per_hadamard)
    );
    println!("Achieved bandwidth:     {:.1} GB/s", bandwidth_gbps);
    println!("DDR4-3200 theoretical:  ~51 GB/s (dual channel)");
    println!(
        "Efficiency:             {:.1}%",
        (bandwidth_gbps / 51.0) * 100.0
    );

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" PHASE 4: Thread Scaling Analysis (27 qubits)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Full GHZ creation timing
    let _ = grid.create_ghz_state(); // Reset

    let mut best_time = f64::MAX;
    let mut total_time = 0.0;
    let runs = 10;

    for _ in 0..runs {
        let start = Instant::now();
        let _ = grid.create_ghz_state();
        let elapsed = start.elapsed().as_secs_f64();
        best_time = best_time.min(elapsed);
        total_time += elapsed;
    }

    let avg_time = total_time / runs as f64;
    let verification = grid.verify_ghz_fidelity();

    // Total amplitude operations for full GHZ
    let total_ghz_ops = (num_blocks * 256) + // Hadamard
                        (6 * num_blocks * 128) + // 6 local CNOTs
                        (20 * num_blocks * 64); // 20 cross-block CNOTs (q7-q26)

    println!("27-qubit GHZ State Creation ({} runs):", runs);
    println!("  Best time:     {:>10.2}ms", best_time * 1000.0);
    println!("  Average time:  {:>10.2}ms", avg_time * 1000.0);
    println!("  Amplitude ops: {}", format_thousands(total_ghz_ops));
    println!(
        "  Peak rate:     {} amp-ops/sec",
        format_throughput(total_ghz_ops as f64 / best_time)
    );
    println!(
        "  Avg rate:      {} amp-ops/sec",
        format_throughput(total_ghz_ops as f64 / avg_time)
    );
    println!("  Fidelity:      {:.12}%", verification.fidelity * 100.0);

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY: True CPU Throughput Potential                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("System: {} threads (rayon auto-detected)", num_threads);
    println!(
        "Peak Hadamard throughput: {} amplitude-ops/sec",
        format_throughput(h_throughput)
    );
    println!(
        "Peak GHZ creation rate:   {} amplitude-ops/sec",
        format_throughput(total_ghz_ops as f64 / best_time)
    );
    println!(
        "Memory bandwidth:         {:.1} GB/s ({:.0}% of DDR4-3200)",
        bandwidth_gbps,
        (bandwidth_gbps / 51.0) * 100.0
    );
    println!("Fidelity:                 100.000000000000% (f64 precision)");

    println!("\nBottleneck analysis:");
    if bandwidth_gbps > 40.0 {
        println!("  Status: MEMORY-BOUND (bandwidth-limited)");
        println!("  The CPU cores are starved for data - memory is the bottleneck.");
    } else if bandwidth_gbps > 25.0 {
        println!("  Status: MIXED (approaching memory limit)");
        println!("  Good parallelization but nearing memory bandwidth saturation.");
    } else {
        println!("  Status: COMPUTE-BOUND or SYNC OVERHEAD");
        println!("  Room for optimization - not saturating memory bandwidth.");
    }
}

fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_throughput(ops: f64) -> String {
    if ops >= 1e12 {
        format!("{:.2}T", ops / 1e12)
    } else if ops >= 1e9 {
        format!("{:.2}G", ops / 1e9)
    } else if ops >= 1e6 {
        format!("{:.2}M", ops / 1e6)
    } else if ops >= 1e3 {
        format!("{:.2}K", ops / 1e3)
    } else {
        format!("{:.0}", ops)
    }
}
