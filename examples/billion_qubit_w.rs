//! Billion-Qubit W State Demonstration (Sprint 57.0)
//!
//! Proves O(n) memory scaling for sparse quantum states by creating and verifying
//! W states from 1,000 to 1,000,000,000 qubits.
//!
//! # Key Insight
//!
//! A W state |W_n⟩ = (1/√n)(|100...0⟩ + |010...0⟩ + ... + |000...1⟩) has only n
//! non-zero amplitudes, not 2^n. Using sparse block storage (128 amplitudes per block),
//! we need only n/128 blocks = O(n) memory.
//!
//! # Performance (Expected on 32GB+ RAM)
//!
//! | Qubits        | Blocks      | Memory     | Time       |
//! |---------------|-------------|------------|------------|
//! | 1,000         | 8           | 16 KB      | < 1 ms     |
//! | 1,000,000     | 7,813       | 15.3 MB    | ~25 ms     |
//! | 100,000,000   | 781,250     | 1.5 GB     | ~3 sec     |
//! | 1,000,000,000 | 7,812,500   | 15.3 GB    | ~30 sec    |
//!
//! # Run
//!
//! ```bash
//! cargo run --release --example billion_qubit_w
//! ```
//!
//! For maximum scale (requires 64GB+ RAM):
//! ```bash
//! cargo run --release --example billion_qubit_w -- --max-scale
//! ```

use engine::tile8::SparseQuantumGridVec;
use std::env;
use std::io::{self, Write};
use std::time::{Duration, Instant};

/// Format large numbers with commas for readability
fn format_with_commas(n: usize) -> String {
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

/// Format bytes as human-readable string
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 60.0 {
        let mins = (secs / 60.0).floor();
        let remaining = secs - mins * 60.0;
        format!("{:.0}m {:.1}s", mins, remaining)
    } else if secs >= 1.0 {
        format!("{:.2} s", secs)
    } else if secs >= 0.001 {
        format!("{:.2} ms", secs * 1000.0)
    } else {
        format!("{:.0} us", secs * 1_000_000.0)
    }
}

/// Single benchmark result
#[derive(Clone)]
struct BenchResult {
    n_qubits: usize,
    n_blocks: usize,
    memory_bytes: usize,
    create_time: Duration,
    verify_time: Duration,
    fidelity: f64,
    correct_amplitudes: usize,
    max_error: f64,
    parallel: bool,
}

/// Run a single W state benchmark
fn benchmark_w_state(n_qubits: usize, use_parallel: bool) -> BenchResult {
    // Create the sparse grid
    let mut grid = SparseQuantumGridVec::new(n_qubits);

    // Create W state
    let create_start = Instant::now();
    if use_parallel {
        grid.create_w_state_parallel();
    } else {
        grid.create_w_state();
    }
    let create_time = create_start.elapsed();

    // Verify
    let verify_start = Instant::now();
    let verification = if use_parallel {
        grid.verify_w_fidelity_parallel()
    } else {
        grid.verify_w_fidelity()
    };
    let verify_time = verify_start.elapsed();

    BenchResult {
        n_qubits,
        n_blocks: grid.n_blocks(),
        memory_bytes: verification.memory_bytes,
        create_time,
        verify_time,
        fidelity: verification.fidelity,
        correct_amplitudes: verification.correct_amplitudes,
        max_error: verification.max_amplitude_error,
        parallel: use_parallel,
    }
}

/// Print benchmark result as table row
fn print_result(result: &BenchResult) {
    let qubit_str = if result.n_qubits >= 1_000_000_000 {
        format!("{:.1}B", result.n_qubits as f64 / 1_000_000_000.0)
    } else if result.n_qubits >= 1_000_000 {
        format!("{:.0}M", result.n_qubits as f64 / 1_000_000.0)
    } else if result.n_qubits >= 1_000 {
        format!("{:.0}K", result.n_qubits as f64 / 1_000.0)
    } else {
        format!("{}", result.n_qubits)
    };

    let total_time = result.create_time + result.verify_time;
    let mem_str = format_bytes(result.memory_bytes);
    let time_str = format_duration(total_time);

    // Fidelity check - use 1e-7 threshold for f64 precision at billion-qubit scale
    let fidelity_ok = (result.fidelity - 1.0).abs() < 1e-7;
    let fidelity_mark = if fidelity_ok { "OK" } else { "FAIL" };

    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>10.8}  {:>6}",
        qubit_str,
        format_with_commas(result.n_blocks),
        mem_str,
        time_str,
        result.fidelity,
        fidelity_mark
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let max_scale = args.iter().any(|a| a == "--max-scale");
    let quick = args.iter().any(|a| a == "--quick");

    println!();
    println!("================================================================================");
    println!("                    BILLION-QUBIT W STATE DEMONSTRATION                         ");
    println!("                           Sprint 57.0 - TileUniverse                           ");
    println!("================================================================================");
    println!();
    println!("  W state: |W_n> = (1/sqrt(n)) * (|10...0> + |01...0> + ... + |00...1>)");
    println!();
    println!("  Key insight: W state has exactly n non-zero amplitudes.");
    println!("  Memory = O(n), not O(2^n). This enables billion-qubit simulation.");
    println!();

    // Define qubit counts to test
    let qubit_counts: Vec<usize> = if quick {
        vec![1_000, 10_000, 100_000, 1_000_000]
    } else if max_scale {
        vec![
            1_000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            500_000_000,
            1_000_000_000,
        ]
    } else {
        vec![
            1_000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
        ]
    };

    // Calculate memory estimate for largest size
    let max_n = *qubit_counts.last().unwrap();
    let max_blocks = max_n.div_ceil(128);
    let max_memory_bytes = max_blocks * 2048; // 128 * 8 * 2 bytes per block
    let max_memory_gb = max_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("  Target: {} qubits", format_with_commas(max_n));
    println!("  Estimated peak memory: {:.1} GB", max_memory_gb);
    println!();

    if max_memory_gb > 28.0 {
        println!("  WARNING: This will use significant RAM. Ensure you have enough.");
        println!("  Use --quick for a lighter test, or Ctrl+C to abort.");
        println!();
    }

    println!("--------------------------------------------------------------------------------");
    println!("                              SEQUENTIAL CREATION                               ");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>10}  {:>6}",
        "Qubits", "Blocks", "Memory", "Time", "Fidelity", "Status"
    );
    println!("  {}", "-".repeat(72));

    let mut sequential_results: Vec<BenchResult> = Vec::new();

    for &n in &qubit_counts {
        let result = benchmark_w_state(n, false);
        print_result(&result);
        io::stdout().flush().unwrap();
        sequential_results.push(result);
    }

    println!();
    println!("--------------------------------------------------------------------------------");
    println!("                               PARALLEL CREATION                                ");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>10}  {:>6}",
        "Qubits", "Blocks", "Memory", "Time", "Fidelity", "Status"
    );
    println!("  {}", "-".repeat(72));

    let mut parallel_results: Vec<BenchResult> = Vec::new();

    for &n in &qubit_counts {
        let result = benchmark_w_state(n, true);
        print_result(&result);
        io::stdout().flush().unwrap();
        parallel_results.push(result);
    }

    // Speedup comparison
    println!();
    println!("--------------------------------------------------------------------------------");
    println!("                            PARALLEL SPEEDUP                                    ");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>10}",
        "Qubits", "Sequential", "Parallel", "Speedup"
    );
    println!("  {}", "-".repeat(50));

    for (seq, par) in sequential_results.iter().zip(parallel_results.iter()) {
        let seq_time = seq.create_time + seq.verify_time;
        let par_time = par.create_time + par.verify_time;
        let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();

        let qubit_str = if seq.n_qubits >= 1_000_000_000 {
            format!("{:.1}B", seq.n_qubits as f64 / 1_000_000_000.0)
        } else if seq.n_qubits >= 1_000_000 {
            format!("{:.0}M", seq.n_qubits as f64 / 1_000_000.0)
        } else if seq.n_qubits >= 1_000 {
            format!("{:.0}K", seq.n_qubits as f64 / 1_000.0)
        } else {
            format!("{}", seq.n_qubits)
        };

        println!(
            "  {:>10}  {:>12}  {:>12}  {:>9.1}x",
            qubit_str,
            format_duration(seq_time),
            format_duration(par_time),
            speedup
        );
    }

    // Memory scaling analysis
    println!();
    println!("--------------------------------------------------------------------------------");
    println!("                           MEMORY SCALING PROOF                                 ");
    println!("--------------------------------------------------------------------------------");
    println!();
    println!("  If memory were O(2^n), a 1B qubit state would need 2^1000000000 bytes.");
    println!("  That's more bytes than atoms in the observable universe.");
    println!();
    println!("  With O(n) sparse storage:");
    println!();

    for result in &parallel_results {
        let bytes_per_qubit = result.memory_bytes as f64 / result.n_qubits as f64;
        let qubit_str = if result.n_qubits >= 1_000_000_000 {
            format!("{:.1}B qubits", result.n_qubits as f64 / 1_000_000_000.0)
        } else if result.n_qubits >= 1_000_000 {
            format!("{:.0}M qubits", result.n_qubits as f64 / 1_000_000.0)
        } else if result.n_qubits >= 1_000 {
            format!("{:.0}K qubits", result.n_qubits as f64 / 1_000.0)
        } else {
            format!("{} qubits", result.n_qubits)
        };

        println!(
            "    {:>15}: {:>10} ({:.1} bytes/qubit)",
            qubit_str,
            format_bytes(result.memory_bytes),
            bytes_per_qubit
        );
    }

    println!();
    println!("  Memory per qubit converges to ~16 bytes (128-amplitude block / 128 qubits × 2KB)");

    // Final summary
    println!();
    println!("================================================================================");
    println!("                                  SUMMARY                                       ");
    println!("================================================================================");

    let last = parallel_results.last().unwrap();
    println!();
    println!(
        "  Maximum achieved: {} qubits",
        format_with_commas(last.n_qubits)
    );
    println!("  Memory used: {}", format_bytes(last.memory_bytes));
    println!(
        "  Total time: {}",
        format_duration(last.create_time + last.verify_time)
    );
    println!("  Fidelity: {:.10}", last.fidelity);
    println!();

    // Verification summary - use 1e-7 threshold for f64 precision at billion-qubit scale
    let all_passed = parallel_results
        .iter()
        .all(|r| (r.fidelity - 1.0).abs() < 1e-7);

    if all_passed {
        println!("  STATUS: ALL VERIFICATIONS PASSED");
        println!();
        println!("  The sparse W state implementation correctly simulates quantum states");
        println!("  at scales impossible with dense amplitude representations.");
    } else {
        println!("  STATUS: SOME VERIFICATIONS FAILED");
        println!();
        println!("  Check results above for details.");
    }

    println!();
    println!("================================================================================");
    println!("  TileUniverse Sprint 57.0 - GPU-Native Sparse Quantum Visualizer");
    println!("================================================================================");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_w_state_1k() {
        let result = benchmark_w_state(1_000, false);
        assert!((result.fidelity - 1.0).abs() < 1e-9);
        assert_eq!(result.correct_amplitudes, 1_000);
    }

    #[test]
    fn test_w_state_1k_parallel() {
        let result = benchmark_w_state(1_000, true);
        assert!((result.fidelity - 1.0).abs() < 1e-9);
        assert_eq!(result.correct_amplitudes, 1_000);
    }

    #[test]
    fn test_memory_scaling_linear() {
        let r1 = benchmark_w_state(1_000, false);
        let r2 = benchmark_w_state(10_000, false);

        // Memory should scale roughly linearly (within 2x factor)
        let ratio = r2.memory_bytes as f64 / r1.memory_bytes as f64;
        assert!(ratio > 8.0 && ratio < 12.0, "Memory ratio: {}", ratio);
    }
}
