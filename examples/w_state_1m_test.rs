//! W State 1 Million Qubit Test
use engine::tile8::sparse_quantum_hybrid::SparseQuantumGridHybrid;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           W STATE - 1 MILLION QUBIT TEST                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let n = 1_000_000;
    println!("Creating 1,000,000 qubit W state...");
    println!("Expected memory: ~2 GB (1M amplitudes × ~2KB/block)\n");

    let start = Instant::now();
    let mut grid = SparseQuantumGridHybrid::new(n);
    println!(
        "Grid initialized in {:.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    let _start = Instant::now();
    let create_us = grid.create_w_state();
    let create_ms = create_us as f64 / 1000.0;
    println!("W state created in {:.1} ms", create_ms);

    let blocks = grid.active_blocks();
    let memory_mb = blocks * 2 / 1024;
    println!("Active blocks: {} (~{} MB)", blocks, memory_mb);

    println!("\nVerifying fidelity (this may take a while)...");
    let start = Instant::now();
    let verification = grid.verify_w_fidelity();
    let verify_sec = start.elapsed().as_secs_f64();

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║                      RESULTS                                    ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║  Qubits:      1,000,000                                         ║");
    println!(
        "║  Create:      {:.1} ms                                         ║",
        create_ms
    );
    println!(
        "║  Verify:      {:.1} sec                                        ║",
        verify_sec
    );
    println!(
        "║  Fidelity:    {:.10}                                    ║",
        verification.fidelity
    );
    println!(
        "║  Memory:      ~{} MB                                          ║",
        memory_mb
    );
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
