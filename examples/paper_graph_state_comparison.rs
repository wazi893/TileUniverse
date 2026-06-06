//! Paper Benchmark: Graph State Comparison for Physical Review A
//!
//! This benchmark generates data for the paper's comparison between:
//! - Block-sparse method (this work): GHZ, W, Dicke states
//! - Graph states (stabilizer states): Shows where sparse method fails
//!
//! Key insight: Graph states have 2^n amplitudes in computational basis,
//! making them intractable for amplitude-sparse simulation despite being
//! efficiently simulable via stabilizer tableaux.
//!
//! Run with: cargo run --release --example paper_graph_state_comparison

use engine::tile8::graph_state::Graph;
use engine::tile8::sparse_quantum_bigint::SparseQuantumGridBigInt;
use std::time::Instant;

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("PAPER BENCHMARK: Sparse vs Graph State Comparison");
    println!("For: Physical Review A - Sparse Quantum State Simulation");
    println!("{}", "=".repeat(80));

    // Section 1: Demonstrate sparse states scale to huge qubit counts
    println!("\n## Section 1: Sparse States (Our Method Excels)");
    benchmark_sparse_states();

    // Section 2: Demonstrate graph states explode exponentially
    println!("\n## Section 2: Graph States (Our Method Fails)");
    benchmark_graph_states();

    // Section 3: Generate comparison table data
    println!("\n## Section 3: Method Comparison Table (Table 12 in paper)");
    generate_comparison_table();

    // Section 4: Sparsity evolution demonstration
    println!("\n## Section 4: Sparsity Evolution Under Circuit Operations");
    demonstrate_sparsity_evolution();

    // Section 5: Circuit class examples
    println!("\n## Section 5: Circuit Classification Examples (Table 11 in paper)");
    demonstrate_circuit_classes();

    println!("\n{}", "=".repeat(80));
    println!("Benchmark complete. Data ready for paper.");
    println!("{}", "=".repeat(80));
}

/// Benchmark sparse states: GHZ, W, Dicke at various scales
fn benchmark_sparse_states() {
    println!("\n### 1a. GHZ States (k=2 amplitudes, any n)");
    println!("{:-<70}", "");
    println!(
        "{:>10} | {:>15} | {:>10} | {:>12} | {:>10}",
        "Qubits", "Amplitudes", "Blocks", "Memory", "Time"
    );
    println!("{:-<70}", "");

    for n in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let mut grid = SparseQuantumGridBigInt::new(n);
        let start = Instant::now();
        grid.create_ghz_state();
        let elapsed = start.elapsed();

        let verification = grid.verify_ghz_fidelity();
        let memory_kb = verification.active_blocks as f64 * 2.0;

        println!(
            "{:>10} | {:>15} | {:>10} | {:>10.1} KB | {:>10}",
            format_number(n),
            "2",
            verification.active_blocks,
            memory_kb,
            format_duration(elapsed)
        );
    }

    println!("\n### 1b. W States (k=n amplitudes)");
    println!("{:-<70}", "");
    println!(
        "{:>10} | {:>15} | {:>10} | {:>12} | {:>10}",
        "Qubits", "Amplitudes", "Blocks", "Memory", "Time"
    );
    println!("{:-<70}", "");

    for n in [100, 1_000, 10_000, 100_000] {
        let mut grid = SparseQuantumGridBigInt::new(n);
        let start = Instant::now();
        grid.create_w_state();
        let elapsed = start.elapsed();

        let verification = grid.verify_w_fidelity();
        let memory_kb = verification.active_blocks as f64 * 2.0;

        println!(
            "{:>10} | {:>15} | {:>10} | {:>10.1} KB | {:>10}",
            format_number(n),
            format_number(n),
            verification.active_blocks,
            memory_kb,
            format_duration(elapsed)
        );
    }

    println!("\n### 1c. Dicke States |D_n^k> (k=C(n,j) amplitudes)");
    println!("{:-<70}", "");
    println!(
        "{:>8} | {:>4} | {:>12} | {:>10} | {:>12} | {:>10}",
        "n", "j", "C(n,j)", "Blocks", "Memory", "Time"
    );
    println!("{:-<70}", "");

    let dicke_cases = [(20, 1), (20, 2), (20, 10), (100, 2), (1000, 2)];

    for (n, j) in dicke_cases {
        let mut grid = SparseQuantumGridBigInt::new(n);
        let start = Instant::now();
        grid.create_dicke_state(j);
        let elapsed = start.elapsed();

        let verification = grid.verify_dicke_fidelity(j);
        let memory_kb = verification.memory_bytes as f64 / 1024.0;

        println!(
            "{:>8} | {:>4} | {:>12} | {:>10} | {:>10.1} KB | {:>10}",
            n,
            j,
            format_number(verification.expected_amplitudes),
            verification.active_blocks,
            memory_kb,
            format_duration(elapsed)
        );
    }
}

/// Benchmark graph states showing exponential blowup
fn benchmark_graph_states() {
    println!("\n### 2a. Graph States (2^n amplitudes - NOT sparse!)");
    println!("{:-<80}", "");
    println!(
        "{:>15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10}",
        "Graph Type", "Vertices", "Edges", "Amplitudes", "Memory", "Time"
    );
    println!("{:-<80}", "");

    // Only small graphs - beyond ~20 qubits, memory explodes
    let graph_cases: Vec<(&str, Graph)> = vec![
        ("Star S_8", Graph::star(8)),
        ("Path P_8", Graph::path(8)),
        ("Cycle C_8", Graph::cycle(8)),
        ("Grid 3x3", Graph::grid_2d(3, 3)),
        ("Star S_10", Graph::star(10)),
        ("Path P_10", Graph::path(10)),
        ("Grid 3x4", Graph::grid_2d(3, 4)),
        ("Star S_12", Graph::star(12)),
        // ("Star S_15", Graph::star(15)), // 32K amplitudes - getting slow
        // ("Star S_20", Graph::star(20)), // 1M amplitudes - ~16MB
    ];

    for (name, graph) in graph_cases {
        let n = graph.n_vertices;
        let mut grid = SparseQuantumGridBigInt::new(n);

        let start = Instant::now();
        grid.create_graph_state(&graph);
        let elapsed = start.elapsed();

        let nonzero = grid.count_nonzero_amplitudes();
        let memory_kb = grid.active_blocks() as f64 * 2.0;

        println!(
            "{:>15} | {:>8} | {:>8} | {:>12} | {:>10.1} KB | {:>10}",
            name,
            graph.n_vertices,
            graph.num_edges(),
            format_number(nonzero),
            memory_kb,
            format_duration(elapsed)
        );
    }

    println!("\n### 2b. Graph State Exponential Growth (theoretical)");
    println!("{:-<70}", "");
    println!(
        "{:>10} | {:>20} | {:>20}",
        "Qubits", "Amplitudes", "Memory Required"
    );
    println!("{:-<70}", "");

    for n in [10, 20, 30, 50, 100, 1000] {
        let amps = format!("2^{}", n);
        let memory = if n <= 30 {
            format!("{:.1} MB", (1u64 << n) as f64 * 16.0 / 1e6)
        } else if n <= 50 {
            format!("{:.1} PB", (1u128 << n) as f64 * 16.0 / 1e15)
        } else {
            format!("10^{:.0} bytes", n as f64 * 0.301 + 1.2)
        };
        println!("{:>10} | {:>20} | {:>20}", n, amps, memory);
    }
    println!("\nNote: Graph states are intractable beyond ~30 qubits for dense simulation.");
    println!("      Our sparse method handles 10^6 qubits for GHZ/W states.");
}

/// Generate comparison table for paper (Table 12)
fn generate_comparison_table() {
    println!("\n### Table 12: Block-sparse vs Tensor Network Efficiency");
    println!("{:-<80}", "");
    println!(
        "{:>15} | {:>12} | {:>15} | {:>12} | {:>12}",
        "State", "Amplitudes", "Entanglement", "Sparse", "MPS"
    );
    println!("{:-<80}", "");

    let rows = [
        ("GHZ (100Q)", "2", "Maximal", "O(1)", "O(n·4)"),
        ("W-state (100Q)", "100", "High", "O(n)", "O(n·4)"),
        ("Dicke D^2_100", "4,950", "Medium", "O(n²)", "O(n·χ²)"),
        ("1D cluster", "2^n", "Area law", "Exp", "O(n)"),
        ("2D cluster", "2^n", "O(√n)", "Exp", "O(n·2^√n)"),
        ("Random state", "2^n", "Volume law", "Exp", "Exp"),
    ];

    for (state, amps, entanglement, sparse, mps) in rows {
        println!(
            "{:>15} | {:>12} | {:>15} | {:>12} | {:>12}",
            state, amps, entanglement, sparse, mps
        );
    }

    println!("\nKey insight:");
    println!("  - Our method wins on GHZ/W/Dicke: amplitude sparsity => O(k) memory");
    println!("  - MPS wins on 1D cluster: area-law entanglement => O(n) memory");
    println!("  - Both fail on random states: volume-law entanglement + dense amplitudes");
}

/// Demonstrate how sparsity evolves under different operations
fn demonstrate_sparsity_evolution() {
    println!("\n### Sparsity Evolution Under Gate Operations");
    println!("{:-<70}", "");

    let n = 12; // Small enough to track all amplitudes

    // Start with |0...0>
    let mut grid = SparseQuantumGridBigInt::new(n);
    grid.initialize_zero_state();
    println!(
        "Initial |0...0>: {} amplitudes",
        grid.count_nonzero_amplitudes()
    );

    // Apply H to qubit 0 -> 2 amplitudes
    grid.apply_h(0);
    println!(
        "After H(0):      {} amplitudes",
        grid.count_nonzero_amplitudes()
    );

    // CNOT cascade -> still 2 amplitudes (creates GHZ)
    for i in 1..n {
        if i < 7 {
            grid.local_cnot(i as u8);
        } else {
            grid.cross_block_cnot(i);
        }
    }
    println!(
        "After CNOT cascade (GHZ): {} amplitudes",
        grid.count_nonzero_amplitudes()
    );

    // Reset and show Hadamard cascade
    grid.initialize_zero_state();
    println!(
        "\nReset to |0...0>: {} amplitudes",
        grid.count_nonzero_amplitudes()
    );

    println!("\nHadamard cascade (exponential blowup):");
    for i in 0..std::cmp::min(n, 10) {
        grid.apply_h(i);
        println!(
            "  After H({}): {} amplitudes",
            i,
            grid.count_nonzero_amplitudes()
        );
    }

    println!("\nConclusion: CNOT preserves sparsity, Hadamard doubles it.");
    println!("           GHZ preparation: 1 H + (n-1) CNOTs = 2 amplitudes");
    println!("           Graph state prep: n H + edges CZs = 2^n amplitudes");
}

/// Demonstrate circuit classification from the paper
fn demonstrate_circuit_classes() {
    println!("\n### Table 11: Circuit Classification for Sparse Simulation");
    println!("{:-<80}", "");
    println!(
        "{:>10} | {:>20} | {:>15} | {:>15}",
        "Class", "Hadamard Count", "Sparsity", "Our Method"
    );
    println!("{:-<80}", "");

    let classes = [
        ("Class I", "h(C) = O(1)", "O(k)", "Efficient"),
        ("Class II", "h(C) = O(log n)", "O(k·poly(n))", "Tractable"),
        ("Class III", "h(C) = Ω(n)", "O(k·2^n)", "Inefficient"),
    ];

    for (class, hadamard, sparsity, method) in classes {
        println!(
            "{:>10} | {:>20} | {:>15} | {:>15}",
            class, hadamard, sparsity, method
        );
    }

    println!("\n### Circuit Examples by Class:");
    println!("{:-<70}", "");

    println!("\nClass I (Efficient - h(C) = O(1)):");
    println!("  - GHZ state preparation: H on qubit 0, then CNOT cascade (h=1)");
    println!("  - Phase circuits: Z, S, T, CZ, arbitrary diagonal unitaries (h=0)");
    println!("  - Controlled operations: CNOT, Toffoli, multi-controlled gates (h=0)");

    println!("\nClass II (Tractable - h(C) = O(log n)):");
    println!("  - Dicke state preparation with logarithmic Hadamard depth");
    println!("  - QFT on sparse input (output may remain tractably sparse)");
    println!("  - Certain variational ansatze with bounded entangling layers");

    println!("\nClass III (Inefficient - h(C) = Ω(n)):");
    println!("  - Graph state preparation: H^⊗n followed by CZ (h=n)");
    println!("  - Random circuits with linear Hadamard layers");
    println!("  - Grover's algorithm: repeated H^⊗n applications");

    // Demonstrate with actual circuit
    println!("\n### Empirical Verification:");

    let n = 10;

    // Class I: GHZ
    let mut grid = SparseQuantumGridBigInt::new(n);
    grid.create_ghz_state();
    println!(
        "  GHZ({} qubits): {} amplitudes (Class I, h=1)",
        n,
        grid.count_nonzero_amplitudes()
    );

    // Class I: W-state (direct construction, no H gates in superposition)
    let mut grid = SparseQuantumGridBigInt::new(n);
    grid.create_w_state();
    println!(
        "  W-state({} qubits): {} amplitudes (Class I, direct construction)",
        n,
        grid.count_nonzero_amplitudes()
    );

    // Class III: Graph state
    let graph = Graph::star(n);
    let mut grid = SparseQuantumGridBigInt::new(n);
    grid.create_graph_state(&graph);
    println!(
        "  Star graph({} qubits): {} amplitudes (Class III, h={})",
        n,
        grid.count_nonzero_amplitudes(),
        n
    );
}

// Helper functions

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_duration(d: std::time::Duration) -> String {
    let us = d.as_micros();
    if us >= 1_000_000 {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    } else if us >= 1_000 {
        format!("{:.1}ms", us as f64 / 1_000.0)
    } else {
        format!("{}μs", us)
    }
}
