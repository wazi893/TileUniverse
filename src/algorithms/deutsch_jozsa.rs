// =============================================================================
// SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani Algorithms
// =============================================================================
//
// File: src/algorithms/deutsch_jozsa.rs
//
// Implements two algorithms demonstrating exponential quantum speedup:
//
// 1. Deutsch-Jozsa: Determine if f(x) is constant or balanced in 1 query
//    - Classical: requires 2^(n-1) + 1 queries worst case
//    - Quantum: 1 query (exponential speedup)
//
// 2. Bernstein-Vazirani: Find hidden string s where f(x) = s·x mod 2
//    - Classical: requires n queries
//    - Quantum: 1 query (linear → constant speedup)
//
// =============================================================================

use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

// =============================================================================
// Deutsch-Jozsa Algorithm
// =============================================================================

/// Oracle type for Deutsch-Jozsa algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DJOracleType {
    /// Constant oracle: f(x) = 0 for all x
    ConstantZero,
    /// Constant oracle: f(x) = 1 for all x
    ConstantOne,
    /// Balanced oracle: f(x) = 0 for half of inputs, 1 for other half
    /// The parameter specifies which pattern to use (different balanced functions)
    Balanced(u64),
}

/// Build a Deutsch-Jozsa circuit.
///
/// The Deutsch-Jozsa algorithm determines whether a function is constant
/// (always 0 or always 1) or balanced (exactly half 0, half 1) with a
/// single query to the oracle.
///
/// Classical complexity: O(2^(n-1) + 1) queries worst case
/// Quantum complexity: O(1) query - exponential speedup!
///
/// # Arguments
/// * `n_qubits` - Number of input qubits (oracle has n input bits)
/// * `oracle_type` - Type of oracle (constant or balanced)
///
/// # Returns
/// A vector of quantum gates implementing the algorithm.
/// After measurement, |0...0⟩ indicates constant, anything else indicates balanced.
///
/// # Circuit Structure
/// ```text
/// |0⟩ ─H─────Oracle─────H─────M─
/// |0⟩ ─H─────Oracle─────H─────M─
/// ...
/// |0⟩ ─H─────Oracle─────H─────M─
/// |1⟩ ─H─────Oracle───────────── (ancilla, not measured)
/// ```
pub fn deutsch_jozsa_circuit(n_qubits: u8, oracle_type: DJOracleType) -> Vec<QGate> {
    assert!(n_qubits >= 1 && n_qubits <= 10, "n_qubits must be 1-10");

    let mut program = Vec::new();
    let ancilla = n_qubits; // Ancilla qubit index
    let total_qubits = n_qubits + 1;

    // Step 1: Initialize ancilla to |1⟩
    program.push(QGate::X(ancilla));

    // Step 2: Apply Hadamard to all qubits (including ancilla)
    for q in 0..total_qubits {
        program.push(QGate::H(q));
    }

    // Step 3: Apply oracle
    program.extend(dj_oracle(n_qubits, oracle_type));

    // Step 4: Apply Hadamard to input qubits only (not ancilla)
    for q in 0..n_qubits {
        program.push(QGate::H(q));
    }

    // Step 5: Measure input qubits only
    for q in 0..n_qubits {
        program.push(QGate::Measure(q));
    }

    program
}

/// Build the Deutsch-Jozsa oracle.
///
/// For constant-zero: do nothing (identity)
/// For constant-one: flip ancilla (X on ancilla)
/// For balanced: CNOT from some input qubits to ancilla
fn dj_oracle(n_qubits: u8, oracle_type: DJOracleType) -> Vec<QGate> {
    let ancilla = n_qubits;

    match oracle_type {
        DJOracleType::ConstantZero => {
            // f(x) = 0: do nothing
            vec![]
        }
        DJOracleType::ConstantOne => {
            // f(x) = 1: flip ancilla unconditionally
            vec![QGate::X(ancilla)]
        }
        DJOracleType::Balanced(pattern) => {
            // f(x) = 1 for inputs where x·pattern is odd
            // Implemented as: CNOT from each qubit where pattern bit is 1
            let mut gates = Vec::new();
            for q in 0..n_qubits {
                if (pattern & (1u64 << q)) != 0 {
                    gates.push(QGate::CNot(q, ancilla));
                }
            }
            // If no bits set in pattern, use qubit 0 to ensure balanced
            if gates.is_empty() {
                gates.push(QGate::CNot(0, ancilla));
            }
            gates
        }
    }
}

/// Run Deutsch-Jozsa algorithm and return whether the oracle is constant.
///
/// # Returns
/// * `true` if oracle is constant (measurement = |0...0⟩)
/// * `false` if oracle is balanced (measurement ≠ |0...0⟩)
pub fn run_deutsch_jozsa(n_qubits: u8, oracle_type: DJOracleType, seed: u64) -> bool {
    let program = deutsch_jozsa_circuit(n_qubits, oracle_type);
    let total_qubits = n_qubits + 1; // Including ancilla
    let mut state = QState::new_zero(total_qubits);
    let mut rng = QRng::new(seed);

    let mut result: u64 = 0;

    for gate in program.iter() {
        let outcome = apply_gate_scalar(&mut state, gate, &mut rng);
        if let GateOutcome::Measured { qubit, bit } = outcome {
            if bit != 0 {
                result |= 1u64 << qubit;
            }
        }
    }

    // Constant if all measured qubits are 0
    result == 0
}

// =============================================================================
// Bernstein-Vazirani Algorithm
// =============================================================================

/// Build a Bernstein-Vazirani circuit.
///
/// The Bernstein-Vazirani algorithm finds a hidden string s where the
/// oracle computes f(x) = s·x mod 2 (bitwise dot product).
///
/// Classical complexity: O(n) queries (query with each basis vector)
/// Quantum complexity: O(1) query - linear → constant speedup!
///
/// # Arguments
/// * `n_qubits` - Number of bits in the hidden string
/// * `hidden_string` - The secret string s to encode in the oracle
///
/// # Returns
/// A vector of quantum gates. After measurement, the result equals the hidden string.
///
/// # Circuit Structure
/// Same as Deutsch-Jozsa:
/// ```text
/// |0⟩ ─H─────Oracle─────H─────M─  → bit 0 of s
/// |0⟩ ─H─────Oracle─────H─────M─  → bit 1 of s
/// ...
/// |1⟩ ─H─────Oracle───────────── (ancilla)
/// ```
pub fn bernstein_vazirani_circuit(n_qubits: u8, hidden_string: u64) -> Vec<QGate> {
    assert!(n_qubits >= 1 && n_qubits <= 10, "n_qubits must be 1-10");
    assert!(
        hidden_string < (1u64 << n_qubits),
        "hidden_string out of range"
    );

    let mut program = Vec::new();
    let ancilla = n_qubits;
    let total_qubits = n_qubits + 1;

    // Step 1: Initialize ancilla to |1⟩
    program.push(QGate::X(ancilla));

    // Step 2: Apply Hadamard to all qubits
    for q in 0..total_qubits {
        program.push(QGate::H(q));
    }

    // Step 3: Apply oracle (CNOT from each qubit where s has a 1)
    program.extend(bv_oracle(n_qubits, hidden_string));

    // Step 4: Apply Hadamard to input qubits only
    for q in 0..n_qubits {
        program.push(QGate::H(q));
    }

    // Step 5: Measure input qubits
    for q in 0..n_qubits {
        program.push(QGate::Measure(q));
    }

    program
}

/// Build the Bernstein-Vazirani oracle.
///
/// The oracle computes f(x) = s·x mod 2 by applying CNOT from each
/// input qubit to the ancilla where the hidden string has a 1 bit.
fn bv_oracle(n_qubits: u8, hidden_string: u64) -> Vec<QGate> {
    let ancilla = n_qubits;
    let mut gates = Vec::new();

    for q in 0..n_qubits {
        if (hidden_string & (1u64 << q)) != 0 {
            gates.push(QGate::CNot(q, ancilla));
        }
    }

    gates
}

/// Run Bernstein-Vazirani algorithm and return the discovered hidden string.
///
/// # Returns
/// The hidden string s, recovered in a single query.
pub fn run_bernstein_vazirani(n_qubits: u8, hidden_string: u64, seed: u64) -> u64 {
    let program = bernstein_vazirani_circuit(n_qubits, hidden_string);
    let total_qubits = n_qubits + 1;
    let mut state = QState::new_zero(total_qubits);
    let mut rng = QRng::new(seed);

    let mut result: u64 = 0;

    for gate in program.iter() {
        let outcome = apply_gate_scalar(&mut state, gate, &mut rng);
        if let GateOutcome::Measured { qubit, bit } = outcome {
            if bit != 0 {
                result |= 1u64 << qubit;
            }
        }
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Deutsch-Jozsa Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dj_constant_zero_detected() {
        // Constant-zero oracle should always return "constant" (true)
        for n in 1..=4 {
            let is_constant = run_deutsch_jozsa(n, DJOracleType::ConstantZero, 42);
            assert!(
                is_constant,
                "n={}: constant-zero should be detected as constant",
                n
            );
        }
    }

    #[test]
    fn test_dj_constant_one_detected() {
        // Constant-one oracle should always return "constant" (true)
        for n in 1..=4 {
            let is_constant = run_deutsch_jozsa(n, DJOracleType::ConstantOne, 42);
            assert!(
                is_constant,
                "n={}: constant-one should be detected as constant",
                n
            );
        }
    }

    #[test]
    fn test_dj_balanced_detected() {
        // Balanced oracles should always return "balanced" (false)
        for n in 1..=4 {
            for pattern in 1..(1u64 << n) {
                let is_constant = run_deutsch_jozsa(n, DJOracleType::Balanced(pattern), 42);
                assert!(
                    !is_constant,
                    "n={}, pattern={}: balanced should be detected as balanced",
                    n, pattern
                );
            }
        }
    }

    #[test]
    fn test_dj_deterministic() {
        // Same inputs should give same results
        let r1 = run_deutsch_jozsa(3, DJOracleType::Balanced(5), 12345);
        let r2 = run_deutsch_jozsa(3, DJOracleType::Balanced(5), 12345);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_dj_circuit_structure() {
        let circuit = deutsch_jozsa_circuit(3, DJOracleType::Balanced(5));

        // Should start with X on ancilla (qubit 3)
        assert!(matches!(circuit[0], QGate::X(3)));

        // Then H on all 4 qubits
        assert!(matches!(circuit[1], QGate::H(0)));
        assert!(matches!(circuit[2], QGate::H(1)));
        assert!(matches!(circuit[3], QGate::H(2)));
        assert!(matches!(circuit[4], QGate::H(3)));

        // Should end with measurements of input qubits only
        let n = circuit.len();
        assert!(matches!(circuit[n - 3], QGate::Measure(0)));
        assert!(matches!(circuit[n - 2], QGate::Measure(1)));
        assert!(matches!(circuit[n - 1], QGate::Measure(2)));
    }

    #[test]
    fn test_dj_single_qubit() {
        // Special case: n=1 (Deutsch's original algorithm)
        // Constant-zero
        assert!(run_deutsch_jozsa(1, DJOracleType::ConstantZero, 0));
        // Constant-one
        assert!(run_deutsch_jozsa(1, DJOracleType::ConstantOne, 0));
        // Balanced (only one pattern possible: 1)
        assert!(!run_deutsch_jozsa(1, DJOracleType::Balanced(1), 0));
    }

    // -------------------------------------------------------------------------
    // Bernstein-Vazirani Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bv_finds_hidden_string() {
        // Should recover the hidden string exactly
        for n in 1..=5 {
            for s in 0..(1u64 << n) {
                let result = run_bernstein_vazirani(n, s, 42);
                assert_eq!(
                    result, s,
                    "n={}, hidden={}: should recover hidden string",
                    n, s
                );
            }
        }
    }

    #[test]
    fn test_bv_zero_string() {
        // Hidden string = 0 should return 0
        let result = run_bernstein_vazirani(4, 0, 42);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_bv_all_ones_string() {
        // Hidden string = 111...1 should return all ones
        let n = 5;
        let s = (1u64 << n) - 1; // 11111
        let result = run_bernstein_vazirani(n as u8, s, 42);
        assert_eq!(result, s);
    }

    #[test]
    fn test_bv_deterministic() {
        // Same inputs should give same results
        let r1 = run_bernstein_vazirani(4, 0b1010, 12345);
        let r2 = run_bernstein_vazirani(4, 0b1010, 12345);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_bv_circuit_structure() {
        let circuit = bernstein_vazirani_circuit(3, 0b101);

        // Should start with X on ancilla (qubit 3)
        assert!(matches!(circuit[0], QGate::X(3)));

        // Then H on all 4 qubits
        assert!(matches!(circuit[1], QGate::H(0)));

        // Should end with measurements of input qubits only
        let n = circuit.len();
        assert!(matches!(circuit[n - 3], QGate::Measure(0)));
        assert!(matches!(circuit[n - 2], QGate::Measure(1)));
        assert!(matches!(circuit[n - 1], QGate::Measure(2)));
    }

    #[test]
    fn test_bv_different_seeds_same_result() {
        // BV is deterministic - different seeds should give same result
        // (no probabilistic element like Grover)
        let s = 0b10101;
        let r1 = run_bernstein_vazirani(5, s, 111);
        let r2 = run_bernstein_vazirani(5, s, 222);
        let r3 = run_bernstein_vazirani(5, s, 333);
        assert_eq!(r1, s);
        assert_eq!(r2, s);
        assert_eq!(r3, s);
    }

    // -------------------------------------------------------------------------
    // Speedup Verification Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dj_single_query() {
        // Verify the circuit uses only one oracle query
        let circuit = deutsch_jozsa_circuit(4, DJOracleType::Balanced(0b1010));

        // Count oracle gates (CNOTs to ancilla in our implementation)
        let oracle_gates: Vec<_> = circuit
            .iter()
            .filter(|g| matches!(g, QGate::CNot(_, 4)))
            .collect();

        // Should have exactly 2 CNOTs (bits 1 and 3 of pattern 0b1010)
        assert_eq!(oracle_gates.len(), 2);

        // This is ONE oracle query, not two separate queries
        // The oracle is applied as a single unitary
    }

    #[test]
    fn test_bv_single_query() {
        // Verify the circuit uses only one oracle query
        let circuit = bernstein_vazirani_circuit(4, 0b1010);

        // Count oracle gates
        let oracle_gates: Vec<_> = circuit
            .iter()
            .filter(|g| matches!(g, QGate::CNot(_, 4)))
            .collect();

        // Should have exactly 2 CNOTs (bits 1 and 3 of 0b1010)
        assert_eq!(oracle_gates.len(), 2);
    }
}
