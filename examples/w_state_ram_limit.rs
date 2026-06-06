//! W State RAM Saturation Test
//! Find the maximum W state size that fits in RAM
use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           W STATE RAM SATURATION TEST                            ║");
    println!("║   Finding maximum qubit count before OOM                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test increasing sizes - skip verification to avoid memory blowup
    let qubit_counts: Vec<usize> = vec![
        500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000,
        15_000_000, 20_000_000,
    ];

    println!(
        "{:>14} {:>12} {:>12} {:>14}",
        "Qubits", "Create (s)", "Blocks", "Memory (GB)"
    );
    println!("{}", "-".repeat(55));

    for n in qubit_counts {
        let label = format_number(n);

        print!("{:>14} ", label);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let _start = Instant::now();
        let mut grid = SparseQuantumGridHybrid::new(n);

        // Create W state (no verification - that's what blows memory)
        let create_us = grid.create_w_state();
        let create_sec = create_us as f64 / 1_000_000.0;

        let blocks = grid.active_blocks();
        let memory_gb = (blocks * 2) as f64 / 1024.0 / 1024.0;

        println!("{:>12.2} {:>12} {:>12.2} GB", create_sec, blocks, memory_gb);

        // Explicitly drop to free memory before next iteration
        drop(grid);
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  W state memory = n blocks × ~2KB = ~2n KB                      ║");
    println!("║  Skipped verification to avoid memory explosion                 ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M qubits", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K qubits", n as f64 / 1_000.0)
    } else {
        format!("{} qubits", n)
    }
}
