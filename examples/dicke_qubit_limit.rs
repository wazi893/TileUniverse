//! Dicke State Qubit Limit Test
//! |D_n^k⟩ has C(n,k) amplitudes - grows combinatorially with k

use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           DICKE STATE QUBIT LIMIT TEST                           ║");
    println!("║   |D_n^k⟩ = uniform superposition of C(n,k) basis states         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // k=1: Same as W state (n amplitudes) - already tested to 4B
    // k=2: Quadratic C(n,2) = n(n-1)/2 amplitudes
    // k=3: Cubic C(n,3) = n(n-1)(n-2)/6 amplitudes

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  DICKE k=2: Quadratic scaling C(n,2) = n(n-1)/2                 │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!(
        "{:>10} {:>16} {:>12} {:>12} {:>10}",
        "Qubits", "Amplitudes", "Create", "Blocks", "Memory"
    );
    println!("{}", "-".repeat(65));

    // k=2 tests - quadratic growth (capped at ~25GB)
    for n in [100, 500, 1000, 2000, 3000, 5000, 6000] {
        let cnk = (n as u64) * (n as u64 - 1) / 2;
        let _est_mem_mb = (cnk * 2 / 1024 / 128) as usize; // rough estimate

        print!("{:>10} {:>16} ", n, cnk);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let mut grid = SparseQuantumGridHybrid::new(n);

        let start = Instant::now();
        grid.create_dicke_state(2);
        let create_ms = start.elapsed().as_secs_f64() * 1000.0;

        let blocks = grid.active_blocks();
        let memory_mb = blocks * 2 / 1024;
        let mem_str = if memory_mb >= 1024 {
            format!("{:.1} GB", memory_mb as f64 / 1024.0)
        } else {
            format!("{} MB", memory_mb)
        };

        println!("{:>10.0}ms {:>12} {:>10}", create_ms, blocks, mem_str);

        drop(grid);

        // Stop if getting too slow or using too much memory
        if create_ms > 60000.0 || memory_mb > 30000 {
            println!("\n  Stopping - approaching limits");
            break;
        }
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  DICKE k=3: Cubic scaling C(n,3) = n(n-1)(n-2)/6                │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!(
        "{:>10} {:>16} {:>12} {:>12} {:>10}",
        "Qubits", "Amplitudes", "Create", "Blocks", "Memory"
    );
    println!("{}", "-".repeat(65));

    // k=3 tests - cubic growth (conservative)
    for n in [50, 100, 200, 300, 400, 500] {
        let cnk = (n as u64) * (n as u64 - 1) * (n as u64 - 2) / 6;

        print!("{:>10} {:>16} ", n, cnk);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let mut grid = SparseQuantumGridHybrid::new(n);

        let start = Instant::now();
        grid.create_dicke_state(3);
        let create_ms = start.elapsed().as_secs_f64() * 1000.0;

        let blocks = grid.active_blocks();
        let memory_mb = blocks * 2 / 1024;
        let mem_str = if memory_mb >= 1024 {
            format!("{:.1} GB", memory_mb as f64 / 1024.0)
        } else if memory_mb > 0 {
            format!("{} MB", memory_mb)
        } else {
            format!("{} KB", blocks * 2)
        };

        println!("{:>10.0}ms {:>12} {:>10}", create_ms, blocks, mem_str);

        drop(grid);

        if create_ms > 60000.0 || memory_mb > 30000 {
            println!("\n  Stopping - approaching limits");
            break;
        }
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  DICKE k=n/2: Maximum entropy (most amplitudes)                 │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!(
        "{:>10} {:>6} {:>16} {:>12} {:>12}",
        "Qubits", "k", "Amplitudes", "Create", "Memory"
    );
    println!("{}", "-".repeat(60));

    // k=n/2 tests - maximum binomial coefficient
    for n in [10, 14, 18, 20, 22, 24, 26] {
        let k = n / 2;
        let cnk = binomial(n, k);

        print!("{:>10} {:>6} {:>16} ", n, k, cnk);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let mut grid = SparseQuantumGridHybrid::new(n);

        let start = Instant::now();
        grid.create_dicke_state(k);
        let create_ms = start.elapsed().as_secs_f64() * 1000.0;

        let blocks = grid.active_blocks();
        let memory_kb = blocks * 2;

        println!("{:>10.0}ms {:>10} KB", create_ms, memory_kb);

        drop(grid);

        if create_ms > 30000.0 {
            println!("\n  Stopping - too slow");
            break;
        }
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  Dicke state scaling depends heavily on k:                      ║");
    println!("║  • k=1: O(n) - same as W state, scales to billions              ║");
    println!("║  • k=2: O(n²) - quadratic, scales to ~10K qubits                ║");
    println!("║  • k=3: O(n³) - cubic, scales to ~1K qubits                     ║");
    println!("║  • k=n/2: O(2^n/√n) - exponential, max ~26 qubits               ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / (i + 1) as u64;
    }
    result
}
