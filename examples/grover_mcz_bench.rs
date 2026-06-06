/// SPRINT 37.0: Benchmark Grover's algorithm with Qiskit MCZ synthesis
///
/// Tests Grover's search for 2-10 qubits to verify that the Qiskit-based
/// multi-controlled-Z synthesis fixes the success rate degradation.
///
/// Expected: Success rate should be >85% for all qubit counts with optimal iterations.
///
/// Usage: cargo run --example grover_mcz_bench --release
use engine::algorithms::{grover_success_rate, optimal_iterations};

fn main() {
    println!("=== Grover's Algorithm MCZ Synthesis Benchmark ===");
    println!();
    println!("Testing Grover's search with Qiskit-based MCZ synthesis");
    println!();

    let trials = 100;

    println!("| Qubits | Search Space | Optimal Iter | Success Rate | Status |");
    println!("|--------|--------------|--------------|--------------|--------|");

    for n_qubits in 2..=10 {
        let search_space = 1 << n_qubits;
        let marked_state = 0; // Find |000...0⟩
        let iterations = optimal_iterations(n_qubits);

        let success_rate = grover_success_rate(n_qubits, marked_state, iterations, trials);

        let status = if success_rate >= 0.85 {
            "✓ PASS"
        } else if success_rate >= 0.70 {
            "⚠ WARN"
        } else {
            "✗ FAIL"
        };

        println!(
            "| {:6} | {:12} | {:12} | {:11.1}% | {:6} |",
            n_qubits,
            search_space,
            iterations,
            success_rate * 100.0,
            status
        );
    }

    println!();
    println!("Notes:");
    println!("- Qubits 1-3: Use native Z, CZ, CCZ gates");
    println!("- Qubits 4+:  Use Qiskit MCZ synthesis (cached after first call)");
    println!("- Success rate >85% indicates proper MCZ implementation");
    println!("- First run for each qubit count will be slower (synthesis + caching)");
    println!("- Subsequent runs are instant (cache hits)");
}
