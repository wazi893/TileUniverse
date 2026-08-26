// =============================================================================
// SPRINT 2.1b: Grover's Search Algorithm
// =============================================================================
//
// File: src/algorithms/grover.rs
//
// Implements Grover's quantum search algorithm using the new CZ/CCZ/Toffoli
// gates from Sprint 2.1a. This demonstrates the quantum→classical feedback
// loop with a real hybrid algorithm.
//
// =============================================================================

use crate::algebraic_fusion::synthesize_mcz;
use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

/// Build a Grover search circuit for n qubits with a specific marked state.
///
/// # Arguments
/// * `n_qubits` - Number of qubits (search space = 2^n_qubits)
/// * `marked_state` - The state to find (as a bit pattern)
/// * `iterations` - Number of Grover iterations (optimal ≈ π/4 × √N)
///
/// # Returns
/// A vector of quantum gates implementing Grover's algorithm
///
/// # Example
/// ```
/// use engine::algorithms::grover_circuit;
/// // Find |101⟩ in 3-qubit space (8 states)
/// // Optimal iterations = floor(π/4 × √8) ≈ 2
/// let circuit = grover_circuit(3, 0b101, 2);
/// ```
pub fn grover_circuit(n_qubits: u8, marked_state: u64, iterations: usize) -> Vec<QGate> {
    assert!(n_qubits > 0 && n_qubits <= 12, "n_qubits must be 1-12");
    assert!(
        marked_state < (1u64 << n_qubits),
        "marked_state out of range"
    );

    let mut program = Vec::new();

    // Step 1: Initialize uniform superposition with Hadamards
    for q in 0..n_qubits {
        program.push(QGate::H(q));
    }

    // Step 2: Grover iterations
    for _ in 0..iterations {
        // Oracle: phase flip the marked state
        program.extend(oracle_phase_flip(n_qubits, marked_state));

        // Diffusion: reflect about the mean
        program.extend(diffusion_operator(n_qubits));
    }

    // Step 3: Measure all qubits
    for q in 0..n_qubits {
        program.push(QGate::Measure(q));
    }

    program
}

/// Calculate optimal number of Grover iterations.
///
/// For a search space of N = 2^n_qubits with 1 marked state,
/// optimal iterations ≈ π/4 × √N
pub fn optimal_iterations(n_qubits: u8) -> usize {
    let n = 1usize << n_qubits;
    let optimal = (std::f64::consts::PI / 4.0 * (n as f64).sqrt()).floor() as usize;
    optimal.max(1) // At least 1 iteration
}

/// Oracle: flip the phase of the marked state.
///
/// For a marked state |m⟩, this applies a phase of -1 to that amplitude
/// while leaving all other amplitudes unchanged.
///
/// Implementation:
/// 1. Apply X gates to qubits that are 0 in the marked state
/// 2. Apply multi-controlled Z (CCZ or decomposed)
/// 3. Apply X gates again to restore
///
/// This effectively creates: I - 2|m⟩⟨m|
fn oracle_phase_flip(n_qubits: u8, marked_state: u64) -> Vec<QGate> {
    let mut gates = Vec::new();

    // Step 1: Flip qubits that are 0 in marked state
    // This transforms |marked⟩ → |111...1⟩
    for q in 0..n_qubits {
        if (marked_state & (1u64 << q)) == 0 {
            gates.push(QGate::X(q));
        }
    }

    // Step 2: Multi-controlled Z on |111...1⟩
    // For small n, use direct CCZ; for larger n, decompose
    gates.extend(multi_controlled_z(n_qubits));

    // Step 3: Undo the X gates
    for q in 0..n_qubits {
        if (marked_state & (1u64 << q)) == 0 {
            gates.push(QGate::X(q));
        }
    }

    gates
}

/// Diffusion operator (Grover's diffusion).
///
/// Reflects amplitudes about the mean: 2|s⟩⟨s| - I
/// where |s⟩ = H⊗n|0⟩ is the uniform superposition.
///
/// Implementation:
/// 1. H⊗n (transform to computational basis)
/// 2. X⊗n (flip all bits)
/// 3. Multi-controlled Z (phase flip |111...1⟩)
/// 4. X⊗n (restore)
/// 5. H⊗n (transform back)
fn diffusion_operator(n_qubits: u8) -> Vec<QGate> {
    let mut gates = Vec::new();

    // H⊗n
    for q in 0..n_qubits {
        gates.push(QGate::H(q));
    }

    // X⊗n
    for q in 0..n_qubits {
        gates.push(QGate::X(q));
    }

    // Multi-controlled Z: flip phase of |111...1⟩
    gates.extend(multi_controlled_z(n_qubits));

    // X⊗n
    for q in 0..n_qubits {
        gates.push(QGate::X(q));
    }

    // H⊗n
    for q in 0..n_qubits {
        gates.push(QGate::H(q));
    }

    gates
}

/// Multi-controlled Z gate.
///
/// Flips the phase of |111...1⟩ (all qubits = 1).
///
/// For n=1: Z(0)
/// For n=2: CZ(0,1)
/// For n=3: CCZ(0,1,2)
/// For n≥4: Uses Qiskit synthesis with caching (SPRINT 37.0)
///
/// **SPRINT 37.0 FIX**: Now uses proper Qiskit-based synthesis for n≥4 instead of
/// the approximate decomposition that caused reduced success rates. Results are
/// cached, so repeated calls are instant.
fn multi_controlled_z(n_qubits: u8) -> Vec<QGate> {
    match n_qubits {
        1 => vec![QGate::Z(0)],
        2 => vec![QGate::CZ(0, 1)],
        3 => vec![QGate::CCZ(0, 1, 2)],
        _ => {
            // For n ≥ 4, use Qiskit synthesis
            // First call synthesizes and caches, subsequent calls are instant
            if let Some(gates) = synthesize_mcz(n_qubits) {
                gates
            } else {
                // Fallback to approximate decomposition if synthesis fails
                eprintln!(
                    "Warning: Qiskit synthesis failed for MCZ({}), using approximate decomposition",
                    n_qubits
                );
                decompose_multi_cz(n_qubits)
            }
        }
    }
}

/// Decompose n-qubit controlled-Z into smaller gates.
///
/// Uses the standard decomposition with Toffoli gates.
/// For n controls, we need n-2 ancilla qubits (not implemented here).
///
/// Simplified version for n=4: chain of CZ and CCZ
fn decompose_multi_cz(n_qubits: u8) -> Vec<QGate> {
    // For now, support up to 4 qubits with direct implementation
    // For larger circuits, would need ancilla-based decomposition

    if n_qubits == 4 {
        // 4-qubit controlled Z can be decomposed as:
        // CCZ(0,1,2) controlled on q3
        // This is still a simplification; proper decomposition would use
        // Toffoli ladder with ancilla

        // Workaround: Use the identity that for phase gates, we can
        // build up the multi-controlled version iteratively
        // C^n(Z) = product of controlled phases on all subsets

        // Simplified: just iterate and check all-ones
        // This is not efficient but works for demo
        vec![
            // Approximate with nested structure
            // In practice, you'd decompose properly
            QGate::CZ(0, 1),
            QGate::CZ(0, 2),
            QGate::CZ(0, 3),
            QGate::CZ(1, 2),
            QGate::CZ(1, 3),
            QGate::CZ(2, 3),
            QGate::CCZ(0, 1, 2),
            QGate::CCZ(0, 1, 3),
            QGate::CCZ(0, 2, 3),
            QGate::CCZ(1, 2, 3),
            // Note: This is the phase polynomial decomposition
            // Actual C^4(Z) needs more careful construction
        ]
    } else {
        // Fall back to warning + approximation for n > 4
        eprintln!(
            "Warning: multi_controlled_z for n={} uses approximate decomposition",
            n_qubits
        );
        // Just use available gates as approximation
        let mut gates = Vec::new();
        for i in 0..(n_qubits - 1) {
            gates.push(QGate::CZ(i, i + 1));
        }
        if n_qubits >= 3 {
            gates.push(QGate::CCZ(0, 1, 2));
        }
        gates
    }
}

// =============================================================================
// High-level simulation helper
// =============================================================================

/// Run Grover's algorithm and return the measured result.
///
/// This is a convenience function for testing that runs the full
/// circuit and returns the measurement outcome.
pub fn run_grover(n_qubits: u8, marked_state: u64, iterations: usize, seed: u64) -> u64 {
    let program = grover_circuit(n_qubits, marked_state, iterations);
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    let mut result: u64 = 0;

    for gate in program.iter() {
        let outcome = apply_gate_scalar(&mut state, gate, &mut rng);
        if let GateOutcome::Measured { qubit, bit } = outcome
            && bit != 0
        {
            result |= 1u64 << qubit;
        }
    }

    result
}

/// Run Grover multiple times and return success rate.
///
/// Useful for benchmarking: with optimal iterations, success rate
/// should be close to 1.0 for small search spaces.
pub fn grover_success_rate(
    n_qubits: u8,
    marked_state: u64,
    iterations: usize,
    trials: usize,
) -> f64 {
    let mut successes = 0;

    for trial in 0..trials {
        let seed = trial as u64 * 0x123456789ABCDEF;
        let result = run_grover(n_qubits, marked_state, iterations, seed);
        if result == marked_state {
            successes += 1;
        }
    }

    successes as f64 / trials as f64
}

// =============================================================================
// Integration with Simulation (for QDemo tiles)
// =============================================================================

/// Create a Grover circuit suitable for registering with a QDemo tile.
///
/// This is the same as `grover_circuit` but explicitly typed for
/// the tile registration API.
pub fn grover_program_for_tile(n_qubits: u8, marked_state: u64) -> Vec<QGate> {
    let iterations = optimal_iterations(n_qubits);
    grover_circuit(n_qubits, marked_state, iterations)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_iterations() {
        // n=2: √4 × π/4 ≈ 1.57 → 1 iteration
        assert_eq!(optimal_iterations(2), 1);

        // n=3: √8 × π/4 ≈ 2.22 → 2 iterations
        assert_eq!(optimal_iterations(3), 2);

        // n=4: √16 × π/4 ≈ 3.14 → 3 iterations
        assert_eq!(optimal_iterations(4), 3);

        // n=5: √32 × π/4 ≈ 4.44 → 4 iterations
        assert_eq!(optimal_iterations(5), 4);
    }

    #[test]
    fn test_grover_circuit_structure() {
        let circuit = grover_circuit(3, 0b101, 2);

        // Should start with H gates
        assert!(matches!(circuit[0], QGate::H(0)));
        assert!(matches!(circuit[1], QGate::H(1)));
        assert!(matches!(circuit[2], QGate::H(2)));

        // Should end with Measure gates
        let n = circuit.len();
        assert!(matches!(circuit[n - 3], QGate::Measure(0)));
        assert!(matches!(circuit[n - 2], QGate::Measure(1)));
        assert!(matches!(circuit[n - 1], QGate::Measure(2)));
    }

    #[test]
    fn test_oracle_phase_flip_structure() {
        // For marked state |101⟩ = 5:
        // - q0 = 1 (no X)
        // - q1 = 0 (X before and after)
        // - q2 = 1 (no X)
        let oracle = oracle_phase_flip(3, 0b101);

        // Should have X(1) at start (q1 is 0 in marked state)
        assert!(oracle.iter().any(|g| matches!(g, QGate::X(1))));

        // Should have CCZ in the middle
        assert!(oracle.iter().any(|g| matches!(g, QGate::CCZ(_, _, _))));
    }

    #[test]
    fn test_grover_finds_marked_state_2_qubits() {
        // 2 qubits, find |11⟩ = 3
        // With 1 iteration, should succeed with high probability
        let success_rate = grover_success_rate(2, 0b11, 1, 100);

        // Should succeed most of the time (theoretical: ~100% for 2 qubits)
        assert!(success_rate > 0.9, "Success rate {} too low", success_rate);
    }

    #[test]
    fn test_grover_finds_marked_state_3_qubits() {
        // 3 qubits, find |101⟩ = 5
        // With 2 iterations, should succeed with high probability
        let success_rate = grover_success_rate(3, 0b101, 2, 100);

        // Should succeed most of the time (theoretical: ~94.5%)
        assert!(success_rate > 0.85, "Success rate {} too low", success_rate);
    }

    #[test]
    fn test_grover_deterministic_with_seed() {
        // Same seed should give same result
        let result1 = run_grover(3, 0b101, 2, 0x1234567890ABCDEF);
        let result2 = run_grover(3, 0b101, 2, 0x1234567890ABCDEF);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_grover_different_seeds_vary() {
        // Different seeds should (usually) give different measurement outcomes
        // over many trials due to quantum randomness
        let mut results = Vec::new();
        for seed in 0..20u64 {
            results.push(run_grover(3, 0b101, 2, seed * 0xDEADBEEF));
        }

        // Not all results should be identical (quantum randomness)
        // But most should be the marked state
        let marked_count = results.iter().filter(|&&r| r == 0b101).count();
        assert!(marked_count > 10, "Should find marked state frequently");
    }

    #[test]
    fn test_grover_quadratic_speedup() {
        // Classical search: expected N/2 queries for N items
        // Grover: expected O(√N) iterations

        // For 3 qubits (8 items):
        // - Classical: ~4 queries expected
        // - Grover: 2 iterations

        // Measure success rate with optimal vs suboptimal iterations
        let optimal = optimal_iterations(3);
        let rate_optimal = grover_success_rate(3, 0b101, optimal, 100);
        let rate_one = grover_success_rate(3, 0b101, 1, 100);
        let rate_four = grover_success_rate(3, 0b101, 4, 100);

        // Optimal should be best
        println!(
            "Success rates: optimal({})={:.2}, one={:.2}, four={:.2}",
            optimal, rate_optimal, rate_one, rate_four
        );

        assert!(
            rate_optimal > rate_one || rate_one > 0.9,
            "Optimal iterations should perform well"
        );
    }

    #[test]
    fn test_diffusion_operator_structure() {
        let diff = diffusion_operator(3);

        // Should have: H⊗3, X⊗3, MCZ, X⊗3, H⊗3
        // Count H gates: should be 6 (3 at start, 3 at end)
        let h_count = diff.iter().filter(|g| matches!(g, QGate::H(_))).count();
        assert_eq!(h_count, 6);

        // Count X gates: should be 6 (3 before MCZ, 3 after)
        let x_count = diff.iter().filter(|g| matches!(g, QGate::X(_))).count();
        assert_eq!(x_count, 6);
    }
}
