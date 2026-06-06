//! 268M CpuState Scale Test
//!
//! Tests what a single CPU can handle with CpuState at 268M scale
//!
//! Run: cargo run --release --example cpustate_268m_test

use engine::tile8::{create_cpu_states, run_parallel, step_parallel};
use std::time::Instant;

fn main() {
    println!("\n=== CpuState Scale Test ===\n");

    let program = [
        0xA1, // LDI R0, #1
        0x20, // ADD R0, R0 -> 2
        0x20, // ADD R0, R0 -> 4
        0x20, // ADD R0, R0 -> 8
    ];

    // Test progressively larger scales
    let scales: Vec<usize> = vec![
        1_000_000,   // 1M   - ~32MB
        10_000_000,  // 10M  - ~320MB
        50_000_000,  // 50M  - ~1.6GB
        100_000_000, // 100M - ~3.2GB
        268_000_000, // 268M - ~8.6GB
    ];

    println!(
        "{:>12} {:>10} {:>12} {:>12} {:>10}",
        "CPUs", "Memory", "Alloc", "10 cycles", "GIPS"
    );
    println!(
        "{:-<12} {:-<10} {:-<12} {:-<12} {:-<10}",
        "", "", "", "", ""
    );

    for count in scales {
        let mem_mb = (count * 32) / (1024 * 1024);
        let mem_str = if mem_mb < 1024 {
            format!("{}MB", mem_mb)
        } else {
            format!("{:.1}GB", mem_mb as f64 / 1024.0)
        };

        // Allocation
        let start = Instant::now();
        let mut states = create_cpu_states(count, &program);
        let alloc_time = start.elapsed();

        // Warmup
        step_parallel(&mut states);

        // Benchmark 10 cycles
        let start = Instant::now();
        run_parallel(&mut states, 10);
        let run_time = start.elapsed();

        let total_ops = count as u64 * 10;
        let gips = total_ops as f64 / run_time.as_secs_f64() / 1e9;

        println!(
            "{:>12} {:>10} {:>12.2?} {:>12.2?} {:>10.2}",
            format_thousands(count),
            mem_str,
            alloc_time,
            run_time,
            gips
        );

        // Drop to free memory before next iteration
        drop(states);
    }

    println!("\nBottleneck: Memory bandwidth (~61 GB/s on this system)");
    println!("CpuState size: 32 bytes (with padding)");
    println!(
        "Theoretical max at 61 GB/s: {:.1} GIPS",
        61.0 * 1e9 / 32.0 / 1e9
    );
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
