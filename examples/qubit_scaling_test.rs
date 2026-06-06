// EPIC 78: Qubit Scaling Test - Path to PETAOPS
// Test 12, 14, 16, 18, 20 qubits with persistent buffers

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 78: Qubit Scaling - Path to PETAOPS! 🚀                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("CUDA runtime"));

    // Test configurations: (qubits, states, depth)
    // FIXED depth = 100K to measure actual arithmetic intensity benefit!
    let tests = vec![
        (12, 1024, 100_000), // Baseline: 4096 amps, 16 tiles
        (14, 512, 100_000),  // 16384 amps, 64 tiles
        (16, 256, 100_000),  // 65536 amps, 256 tiles
        (18, 128, 100_000),  // 262144 amps, 1024 tiles
        (20, 64, 100_000),   // 1048576 amps, 4096 tiles
    ];

    let mut results = Vec::new();

    for (qubits, num_states, depth) in tests {
        let tiles_per_state = 1 << (qubits - 8); // 2^(qubits-8) tiles (each tile = 256 amps = 2^8)
        let amps = 1u64 << qubits;

        println!("══════════════════════════════════════════════════════════════");
        println!(
            " {} qubits: {} amps, {} tiles/state, {} states, {} depth",
            qubits, amps, tiles_per_state, num_states, depth
        );
        println!("══════════════════════════════════════════════════════════════\n");

        let persistent = match MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to allocate {} qubits: {:?}\n", qubits, e);
                continue;
            }
        };

        println!();

        // Run benchmark
        let (g, a, elapsed) =
            match persistent.run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Benchmark failed: {:?}\n", e);
                    continue;
                }
            };

        let tcops = (a as f64 / elapsed) / 1e12;
        println!("Elapsed: {:.3}s", elapsed);
        println!("TCOPS: {:.2}", tcops);
        println!();

        results.push((qubits, tcops));
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Qubit Scaling Summary                                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    for (qubits, tcops) in results {
        let status = if tcops >= 1000.0 {
            format!("🔥 {:.2} PETAOPS!", tcops / 1000.0)
        } else {
            format!("{:.2} TCOPS", tcops)
        };
        println!("  {} qubits: {}", qubits, status);
    }

    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: Requires cuda,perf-bench features");
    std::process::exit(1);
}
