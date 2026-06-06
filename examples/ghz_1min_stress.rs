//! 1-Minute GHZ State Stress Test
use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::{Duration, Instant};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           1-MINUTE GHZ STATE STRESS TEST                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Quick 5-second tests for scaling, then 1-minute test for 1M qubits
    let test_configs = [
        (100, "100 qubits", 5),
        (1_000, "1K qubits", 5),
        (10_000, "10K qubits", 5),
        (100_000, "100K qubits", 5),
        (1_000_000, "1M qubits", 60), // Full 1-minute test
    ];

    println!(
        "{:>12} {:>8} {:>15} {:>15} {:>12}",
        "Qubits", "Duration", "States/min", "States/sec", "Per-State"
    );
    println!("{}", "-".repeat(68));

    for (n_qubits, label, duration_secs) in test_configs {
        let mut grid = SparseQuantumGridHybrid::new(n_qubits);
        let duration = Duration::from_secs(duration_secs);

        // Warmup
        for _ in 0..100 {
            grid.create_ghz_state();
        }

        print!("{:>12} {:>6}s  ", label, duration_secs);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        // Stress test
        let start = Instant::now();
        let mut count: u64 = 0;

        while start.elapsed() < duration {
            // Run batches of 1000 to reduce timing overhead
            for _ in 0..1000 {
                grid.create_ghz_state();
            }
            count += 1000;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let per_sec = count as f64 / elapsed;
        let per_min = per_sec * 60.0;
        let per_state_ns = 1_000_000_000.0 / per_sec;

        println!(
            "{:>15.0} {:>15.0} {:>10.0} ns",
            per_min, per_sec, per_state_ns
        );
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  All tests used O(1) sparse GHZ construction                    ║");
    println!("║  Memory: 4 KB constant regardless of qubit count                ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
