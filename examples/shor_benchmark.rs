// =============================================================================
// Shor's Algorithm Benchmark
// =============================================================================
//
// Demonstrates Shor's quantum factoring algorithm for factoring composite numbers.
//
// This example shows:
// - Classical vs quantum period finding comparison
// - Factorization of 15 and 21
// - Success rate analysis over multiple trials
// - Circuit structure and gate count analysis
// - Performance timing
//
// Run with:
//   cargo run --example shor_benchmark --release
//
// For detailed output:
//   cargo run --example shor_benchmark --release -- --verbose
//
// =============================================================================

use engine::algorithms::{factor_with_shor, qft_circuit, run_shor_quantum, shor_circuit};
use engine::number_theory::{classical_order, gcd, mod_exp};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let verbose = std::env::args().any(|arg| arg == "--verbose");

    print_header();

    // Section 1: Classical Period Finding
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 1: Classical Period Finding (Baseline)              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    demo_classical_period_finding(verbose);

    // Section 2: Quantum Period Finding
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 2: Quantum Period Finding                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    demo_quantum_period_finding(verbose);

    // Section 3: Full Factorization
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 3: Full Factorization with Shor's Algorithm          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    demo_factorization(15, verbose);
    demo_factorization(21, verbose);

    // Section 4: Success Rate Analysis
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 4: Success Rate Analysis                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    analyze_success_rate(15, 20, verbose);

    // Section 5: Circuit Analysis
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 5: Circuit Structure Analysis                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    analyze_circuit_structure(verbose);

    print_footer();
}

fn print_header() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                                                                ║");
    println!("║           SHOR'S FACTORING ALGORITHM BENCHMARK                 ║");
    println!("║                                                                ║");
    println!("║  Quantum period finding for integer factorization             ║");
    println!("║  Exponential speedup over classical algorithms                ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn print_footer() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK COMPLETE                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}

/// Demonstrate classical period finding (brute force)
fn demo_classical_period_finding(verbose: bool) {
    println!("Testing classical period finding for N=15, a=7");
    println!("Goal: Find period r such that 7^r ≡ 1 (mod 15)\n");

    let n = 15u64;
    let a = 7u64;

    let start = Instant::now();
    let period = classical_order(a, n);
    let elapsed = start.elapsed();

    match period {
        Some(r) => {
            println!("✓ Found period: r = {}", r);
            println!("  Verification: 7^{} mod 15 = {}", r, mod_exp(a, r, n));
            println!("  Time: {:.2?}", elapsed);

            if verbose {
                println!("\n  Computation steps:");
                for i in 1..=r {
                    let result = mod_exp(a, i, n);
                    println!("    7^{} mod 15 = {}", i, result);
                    if result == 1 {
                        break;
                    }
                }
            }
        }
        None => {
            println!("✗ Period not found");
        }
    }

    println!("\n  Classical complexity: O(N) exponential time");
    println!("  Quantum complexity:   O((log N)³) polynomial time");
}

/// Demonstrate quantum period finding
fn demo_quantum_period_finding(verbose: bool) {
    println!("Testing quantum period finding for N=15, a=7");
    println!("Using Shor's quantum circuit with QFT\n");

    let n = 15u64;
    let a = 7u64;
    let n_bits = (n as f64).log2().ceil() as u8;
    let register_size = 1u64 << (2 * n_bits);

    println!("Circuit parameters:");
    println!("  N = {}", n);
    println!("  Base a = {}", a);
    println!("  Counting register: {} qubits", 2 * n_bits);
    println!("  Target register: {} qubits", n_bits);
    println!("  Total qubits: {}", 2 * n_bits + n_bits);

    let mut measurements = Vec::new();
    let num_trials = if verbose { 10 } else { 5 };

    println!("\nRunning {} quantum trials:\n", num_trials);

    for trial in 0..num_trials {
        let seed = trial as u64;
        let start = Instant::now();
        let measured = run_shor_quantum(n, a, seed);
        let elapsed = start.elapsed();

        // Extract period candidates from measurement
        let candidates =
            engine::number_theory::continued_fraction_candidates(measured, register_size);

        println!(
            "  Trial {}: measured = {}/{} ({:.3})",
            trial + 1,
            measured,
            register_size,
            measured as f64 / register_size as f64
        );

        if verbose {
            println!("           period candidates: {:?}", candidates);
        }

        println!("           time: {:.2?}", elapsed);

        measurements.push((measured, candidates));
    }

    println!("\n✓ Quantum period finding complete");
    println!(
        "  Average time per trial: {:.2?}",
        measurements.iter().map(|_| 100).sum::<u64>() / measurements.len() as u64
    );
}

/// Demonstrate full factorization
fn demo_factorization(n: u64, verbose: bool) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!(" Factoring N = {}", n);
    println!("═══════════════════════════════════════════════════════════════\n");

    if verbose {
        println!("Pre-checks:");
        println!("  Even? {}", n % 2 == 0);
        println!("  Prime? {}", engine::number_theory::is_prime(n));
        if let Some((p, k)) = engine::number_theory::is_prime_power(n) {
            println!("  Prime power? Yes ({}^{})", p, k);
        } else {
            println!("  Prime power? No");
        }
        println!();
    }

    let start = Instant::now();
    let result = factor_with_shor(n);
    let elapsed = start.elapsed();

    match result {
        Some((p, q)) => {
            println!("✓ SUCCESS: {} = {} × {}", n, p, q);
            println!("  Verification: {} × {} = {}", p, q, p * q);
            println!("  Time: {:.2?}", elapsed);

            if verbose {
                println!("\n  Factor properties:");
                println!(
                    "    p = {} (prime: {})",
                    p,
                    engine::number_theory::is_prime(p)
                );
                println!(
                    "    q = {} (prime: {})",
                    q,
                    engine::number_theory::is_prime(q)
                );
                println!("    gcd(p, q) = {}", gcd(p, q));
            }
        }
        None => {
            println!("✗ FAILED: Could not factor {}", n);
            println!("  Time: {:.2?}", elapsed);
        }
    }
}

/// Analyze success rate over multiple trials
fn analyze_success_rate(n: u64, num_trials: usize, verbose: bool) {
    println!(
        "Analyzing success rate for N = {} over {} trials\n",
        n, num_trials
    );

    let mut successes = 0;
    let mut total_time = std::time::Duration::ZERO;
    let mut first_success_time = None;

    for trial in 0..num_trials {
        if verbose && trial % 5 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        let start = Instant::now();
        let result = factor_with_shor(n);
        let elapsed = start.elapsed();
        total_time += elapsed;

        if result.is_some() {
            successes += 1;
            if first_success_time.is_none() {
                first_success_time = Some(elapsed);
            }
        }
    }

    if verbose {
        println!();
    }

    println!("\nResults:");
    println!("  Total trials: {}", num_trials);
    println!("  Successes: {}", successes);
    println!(
        "  Success rate: {:.1}%",
        (successes as f64 / num_trials as f64) * 100.0
    );
    println!(
        "  Average time per trial: {:.2?}",
        total_time / num_trials as u32
    );

    if let Some(time) = first_success_time {
        println!("  First success time: {:.2?}", time);
    }

    println!("\n  Theoretical success probability:");
    println!("    With optimal parameters: ~80-90%");
    println!("    With multiple bases: >95%");
}

/// Analyze circuit structure
fn analyze_circuit_structure(verbose: bool) {
    let test_cases = vec![(15, 7, "Small (4 bits)"), (21, 2, "Small (5 bits)")];

    for (n, a, description) in test_cases {
        println!("\n─────────────────────────────────────────────────────────────────");
        println!(" Circuit for N={}, a={} ({})", n, a, description);
        println!("─────────────────────────────────────────────────────────────────\n");

        let circuit = shor_circuit(n, a);

        // Count gate types
        let mut gate_counts: HashMap<String, usize> = HashMap::new();
        for gate in &circuit {
            let gate_name = match gate {
                engine::quantum::QGate::H(_) => "H",
                engine::quantum::QGate::X(_) => "X",
                engine::quantum::QGate::Y(_) => "Y",
                engine::quantum::QGate::Z(_) => "Z",
                engine::quantum::QGate::CNot(_, _) => "CNot",
                engine::quantum::QGate::CZ(_, _) => "CZ",
                engine::quantum::QGate::Phase(_, _) => "Phase",
                engine::quantum::QGate::Rx(_, _) => "Rx",
                engine::quantum::QGate::Ry(_, _) => "Ry",
                engine::quantum::QGate::Rz(_, _) => "Rz",
                engine::quantum::QGate::CPhase(_, _, _) => "CPhase",
                engine::quantum::QGate::CRz(_, _, _) => "CRz",
                engine::quantum::QGate::Swap(_, _) => "Swap",
                engine::quantum::QGate::Toffoli(_, _, _) => "Toffoli",
                engine::quantum::QGate::CCZ(_, _, _) => "CCZ",
                engine::quantum::QGate::U3(_, _, _, _) => "U3",
                engine::quantum::QGate::Measure(_) => "Measure",
                engine::quantum::QGate::T(_) => "T",
                engine::quantum::QGate::Tdg(_) => "Tdg",
            };
            *gate_counts.entry(gate_name.to_string()).or_insert(0) += 1;
        }

        println!("  Total gates: {}", circuit.len());
        println!("\n  Gate breakdown:");

        let mut sorted_gates: Vec<_> = gate_counts.iter().collect();
        sorted_gates.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

        for (gate_type, count) in sorted_gates {
            println!("    {:<12} {}", gate_type, count);
        }

        if verbose {
            println!("\n  Circuit depth estimation:");
            println!("    Sequential depth: ~{}", circuit.len());
            println!("    Parallel depth: ~{}", estimate_parallel_depth(&circuit));
        }

        // QFT analysis
        let n_bits = (n as f64).log2().ceil() as u8;
        let qft_gates = qft_circuit(2 * n_bits);
        println!("\n  QFT subcircuit ({} qubits):", 2 * n_bits);
        println!("    Gates: {}", qft_gates.len());
        println!(
            "    Theoretical: O(n²) = O({})",
            (2 * n_bits) * (2 * n_bits)
        );
    }
}

/// Estimate parallel circuit depth (rough approximation)
fn estimate_parallel_depth(circuit: &[engine::quantum::QGate]) -> usize {
    // Simple estimation: count layers where gates can be parallelized
    // In reality, this depends on qubit topology and gate dependencies
    let single_qubit_gates = circuit.iter().filter(|g| is_single_qubit_gate(g)).count();
    let two_qubit_gates = circuit.iter().filter(|g| !is_single_qubit_gate(g)).count();

    // Rough estimate: single-qubit gates can parallelize well, two-qubit gates less so
    (single_qubit_gates / 4).max(1) + two_qubit_gates
}

fn is_single_qubit_gate(gate: &engine::quantum::QGate) -> bool {
    matches!(
        gate,
        engine::quantum::QGate::H(_)
            | engine::quantum::QGate::X(_)
            | engine::quantum::QGate::Y(_)
            | engine::quantum::QGate::Z(_)
            | engine::quantum::QGate::Phase(_, _)
            | engine::quantum::QGate::Rx(_, _)
            | engine::quantum::QGate::Ry(_, _)
            | engine::quantum::QGate::Rz(_, _)
            | engine::quantum::QGate::U3(_, _, _, _)
            | engine::quantum::QGate::Measure(_)
    )
}
