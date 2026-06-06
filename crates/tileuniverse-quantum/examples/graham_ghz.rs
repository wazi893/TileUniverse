//! Demonstrate a GHZ state with Graham's number of qubits.
//!
//! This state has O(1) memory regardless of the qubit count.
//! It takes the same amount of time and space as a 10-qubit GHZ state.

use tileuniverse_quantum::*;

fn main() {
    println!("=== Graham's Number GHZ State ===\n");

    let ghz = SymbolicGhzState::graham();
    println!("Created: |GHZ_Graham> = (|0...0> + |1...1>) / sqrt(2)");
    println!("n_qubits = {}", ghz.n_qubits.to_notation());
    println!("creation_time_ns = {} nanoseconds\n", ghz.creation_time_ns);

    // Verify the state algebraically -- works for Graham's number
    let v = ghz.verify();
    println!("Verification:");
    println!("  fidelity        = {:.15}", v.fidelity);
    println!("  size_class      = {}", v.size_class);
    println!("  amp_zero        = {:.15}", v.amp_zero_state);
    println!("  amp_one         = {:.15}", v.amp_one_state);
    println!("  expected_amp    = {:.15}", v.expected_amplitude);

    // GHZ normalization depends on the two stored endpoint amplitudes, not on n:
    //   2 * |1/sqrt(2)|^2 = 1
    println!("\nNormalization proof (algebraic):");
    println!("  2 * |1/sqrt(2)|^2 = 2 * (1/2)");
    println!("                    = 1.0 (exactly)\n");

    println!("{}", v);

    println!("=== Comparison ===");

    let tiny = MinimalGhzState::new(5);
    let tiny_v = tiny.verify();
    println!(
        "  5-qubit GHZ:   creation = {} us,  fidelity = {:.15}",
        tiny.creation_time_us, tiny_v.fidelity
    );

    let huge = MinimalGhzState::new(1_000_000_000);
    let huge_v = huge.verify();
    println!(
        "  1B-qubit GHZ:  creation = {} us,  fidelity = {:.15}",
        huge.creation_time_us, huge_v.fidelity
    );

    println!(
        "  Graham-qubit:  creation = {:.3} us,  fidelity = {:.15}",
        ghz.creation_time_ns as f64 / 1000.0,
        v.fidelity
    );

    println!(
        "\nAll three states use the same two-amplitude endpoint representation; only the qubit-count label differs."
    );
}
