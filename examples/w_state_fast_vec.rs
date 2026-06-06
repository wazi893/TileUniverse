//! Fast W State using Vec-based storage (no BigInt overhead)
//! For W states, block IDs are sequential: 0, 1, 2, ..., n/128
//! So we can use a Vec instead of HashMap<BigUint>

use std::time::Instant;

/// Minimal block representation
#[derive(Clone)]
struct WBlock {
    re: [f64; 128],
    im: [f64; 128],
}

impl WBlock {
    fn new() -> Self {
        Self {
            re: [0.0; 128],
            im: [0.0; 128],
        }
    }

    fn set(&mut self, addr: usize, re: f64, im: f64) {
        self.re[addr] = re;
        self.im[addr] = im;
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║      FAST W STATE (Vec-based, no BigInt)                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let qubit_counts: Vec<usize> = vec![
        1_000_000_000, // 1 billion
        2_000_000_000, // 2 billion
        3_000_000_000, // 3 billion
        4_000_000_000, // 4 billion - ~60GB
    ];

    println!(
        "{:>12} {:>12} {:>10} {:>12} {:>12}",
        "Qubits", "Blocks", "Create", "Verify", "Memory"
    );
    println!("{}", "-".repeat(62));

    for n in qubit_counts {
        let label = if n >= 1_000_000_000 {
            format!("{:.1}B", n as f64 / 1_000_000_000.0)
        } else {
            format!("{}M", n / 1_000_000)
        };
        print!("{:>12} ", label);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let num_blocks = (n + 127) / 128;
        let memory_mb = (num_blocks * 2048) / (1024 * 1024);

        // Allocate
        let start = Instant::now();
        let mut blocks: Vec<WBlock> = vec![WBlock::new(); num_blocks];
        let _alloc_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Create W state
        let amplitude = 1.0 / (n as f64).sqrt();
        let start = Instant::now();
        for qubit in 0..n {
            let block_idx = qubit / 128;
            let local_addr = qubit % 128;
            blocks[block_idx].set(local_addr, amplitude, 0.0);
        }
        let create_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Verify (just check total probability)
        let start = Instant::now();
        let mut total_prob = 0.0;
        for qubit in 0..n {
            let block_idx = qubit / 128;
            let local_addr = qubit % 128;
            let re = blocks[block_idx].re[local_addr];
            let im = blocks[block_idx].im[local_addr];
            total_prob += re * re + im * im;
        }
        let verify_ms = start.elapsed().as_secs_f64() * 1000.0;
        let fidelity = total_prob;

        let mem_str = if memory_mb >= 1024 {
            format!("{:.1} GB", memory_mb as f64 / 1024.0)
        } else {
            format!("{} MB", memory_mb)
        };
        println!(
            "{:>12} {:>8.0}ms {:>8.0}ms {:>12}  fid={:.6}",
            num_blocks, create_ms, verify_ms, mem_str, fidelity
        );

        // Stop if we're using too much memory (50GB limit to be safe)
        if memory_mb > 50000 {
            println!("\n  Approaching RAM limit, stopping.");
            break;
        }

        drop(blocks);
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  Vec-based W state: O(n) memory, O(1) access per amplitude      ║");
    println!("║  No BigInt overhead - direct indexing by qubit number           ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
