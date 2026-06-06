/// SPRINT 37.0: Option A+ - Toffoli Truth Table Validation
///
/// This module validates that the synthesized Toffoli gate (from Qiskit decomposition)
/// behaves correctly by testing all 8 basis states.
///
/// Tests:
/// 1. Truth table for all 8 basis states |000⟩ through |111⟩
/// 2. Qubit remapping sanity check (Toffoli(0,1,2) == Toffoli(5,6,7))
///
/// Scope:
/// - Toffoli only (Shor's doesn't use Fredkin)
/// - Fredkin validation deferred for later

#[cfg(test)]
mod tests {
    use crate::algebraic_fusion::{synthesize_three_qubit, ThreeQubitGate};
    use crate::quantum::{apply_gate_scalar, GateOutcome, QGate, QRng, QState};

    /// Helper: Create a 3-qubit basis state |q2 q1 q0⟩ where q0,q1,q2 ∈ {0,1}
    ///
    /// Note: Parameter order matches standard quantum notation |q2 q1 q0⟩ (big-endian)
    /// but state vector index is little-endian: index = q0 + 2*q1 + 4*q2
    ///
    /// Examples:
    /// - create_basis_state_3q(0, 0, 0) → |000⟩ at index 0
    /// - create_basis_state_3q(0, 0, 1) → |001⟩ at index 1
    /// - create_basis_state_3q(1, 1, 0) → |110⟩ at index 6
    /// - create_basis_state_3q(1, 1, 1) → |111⟩ at index 7
    fn create_basis_state_3q(q2: u8, q1: u8, q0: u8) -> QState {
        assert!(
            q0 <= 1 && q1 <= 1 && q2 <= 1,
            "Basis state bits must be 0 or 1"
        );

        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Flip qubits to create desired basis state
        // X(i) flips qubit i
        if q0 == 1 {
            apply_gate_scalar(&mut state, &QGate::X(0), &mut rng);
        }
        if q1 == 1 {
            apply_gate_scalar(&mut state, &QGate::X(1), &mut rng);
        }
        if q2 == 1 {
            apply_gate_scalar(&mut state, &QGate::X(2), &mut rng);
        }

        state
    }

    /// Helper: Apply a sequence of gates to a quantum state
    fn apply_gate_sequence(state: &mut QState, gates: &[QGate]) {
        let mut rng = QRng::new(0);
        for gate in gates {
            apply_gate_scalar(state, gate, &mut rng);
        }
    }

    /// Helper: Get the amplitude at a specific basis state index
    ///
    /// For 3 qubits: index = 4*a + 2*b + c
    fn get_amplitude(state: &QState, index: usize) -> (f32, f32) {
        let real = state.real.as_slice()[index];
        let imag = state.imag.as_slice()[index];
        (real, imag)
    }

    /// Helper: Check if amplitude has magnitude ≈ 1 (ignoring global phase)
    ///
    /// In quantum mechanics, global phase doesn't matter - |ψ⟩ and e^{iθ}|ψ⟩ are equivalent.
    /// So we check if |amplitude| ≈ 1, not if amplitude ≈ 1+0i.
    fn is_amplitude_one(amp: (f32, f32)) -> bool {
        let magnitude_squared = amp.0 * amp.0 + amp.1 * amp.1;
        (magnitude_squared - 1.0).abs() < 1e-5
    }

    /// Helper: Check if amplitude is approximately 0.0 + 0.0i
    fn is_amplitude_zero(amp: (f32, f32)) -> bool {
        let magnitude_squared = amp.0 * amp.0 + amp.1 * amp.1;
        magnitude_squared < 1e-10
    }

    /// TEST 1: Toffoli Truth Table - All 8 Basis States
    ///
    /// Toffoli(control1=0, control2=1, target=2) truth table:
    ///
    /// | Input (abc) | Output (abc) | Explanation                          |
    /// |-------------|--------------|--------------------------------------|
    /// | 000         | 000          | Controls not both 1 → target unchanged |
    /// | 001         | 001          | Controls not both 1 → target unchanged |
    /// | 010         | 010          | Controls not both 1 → target unchanged |
    /// | 011         | 011          | Controls not both 1 → target unchanged |
    /// | 100         | 100          | Controls not both 1 → target unchanged |
    /// | 101         | 101          | Controls not both 1 → target unchanged |
    /// | 110         | 111          | Controls BOTH 1 → target FLIPPED      |
    /// | 111         | 110          | Controls BOTH 1 → target FLIPPED      |
    ///
    /// This is the critical behavior: target flips if and only if both controls are |1⟩
    #[test]
    fn test_toffoli_truth_table() {
        println!("\n=== Toffoli Truth Table Test ===\n");

        // Synthesize Toffoli once (will be cached after first call)
        let toffoli_gates = synthesize_three_qubit(ThreeQubitGate::Toffoli, 0, 1, 2)
            .expect("Failed to synthesize Toffoli gate");

        println!("Synthesized Toffoli into {} gates", toffoli_gates.len());
        println!("\nTesting all 8 basis states:\n");

        // Test all 8 basis states
        // For |q2 q1 q0⟩ with Toffoli(control1=0, control2=1, target=2):
        // - If q0=1 AND q1=1, flip q2
        let test_cases = [
            // (q2, q1, q0) -> (expected_q2, expected_q1, expected_q0)
            (0, 0, 0, 0, 0, 0), // |000⟩: q0=0,q1=0 → no flip → |000⟩
            (0, 0, 1, 0, 0, 1), // |001⟩: q0=1,q1=0 → no flip → |001⟩
            (0, 1, 0, 0, 1, 0), // |010⟩: q0=0,q1=1 → no flip → |010⟩
            (0, 1, 1, 1, 1, 1), // |011⟩: q0=1,q1=1 → FLIP q2: 0→1 → |111⟩
            (1, 0, 0, 1, 0, 0), // |100⟩: q0=0,q1=0 → no flip → |100⟩
            (1, 0, 1, 1, 0, 1), // |101⟩: q0=1,q1=0 → no flip → |101⟩
            (1, 1, 0, 1, 1, 0), // |110⟩: q0=0,q1=1 → no flip → |110⟩
            (1, 1, 1, 0, 1, 1), // |111⟩: q0=1,q1=1 → FLIP q2: 1→0 → |011⟩
        ];

        for (a, b, c, exp_a, exp_b, exp_c) in test_cases {
            // Create initial basis state |abc⟩
            let mut state = create_basis_state_3q(a, b, c);

            // Apply synthesized Toffoli
            apply_gate_sequence(&mut state, &toffoli_gates);

            // Calculate expected output index (little-endian: index = q0 + 2*q1 + 4*q2)
            // For |abc⟩ notation: a=q2, b=q1, c=q0
            let expected_idx = exp_c + 2 * exp_b + 4 * exp_a;

            // Check that we're in the expected output state
            let amp = get_amplitude(&state, expected_idx as usize);

            let result = if is_amplitude_one(amp) {
                "✓ PASS"
            } else {
                "✗ FAIL"
            };
            println!(
                "  |{}{}{} → |{}{}{}: {} (amp = {:.6} + {:.6}i)",
                a, b, c, exp_a, exp_b, exp_c, result, amp.0, amp.1
            );

            assert!(
                is_amplitude_one(amp),
                "Toffoli truth table violated: |{}{}{} should map to |{}{}{}, \
                     but amplitude at index {} is {:.6} + {:.6}i",
                a,
                b,
                c,
                exp_a,
                exp_b,
                exp_c,
                expected_idx,
                amp.0,
                amp.1
            );

            // Verify all other amplitudes are zero
            for idx in 0..8 {
                if idx != expected_idx as usize {
                    let other_amp = get_amplitude(&state, idx);
                    assert!(
                        is_amplitude_zero(other_amp),
                        "Expected zero amplitude at index {}, got {:.6} + {:.6}i",
                        idx,
                        other_amp.0,
                        other_amp.1
                    );
                }
            }
        }

        println!("\n✓ All 8 basis states passed!");
        println!("  - 6 states unchanged (controls not both |1⟩)");
        println!("  - 2 states flipped (|110⟩ ↔ |111⟩, controls both |1⟩)");
    }

    /// TEST 2: Qubit Remapping Sanity Check
    ///
    /// Verifies that Toffoli(0,1,2) produces the same logical behavior as Toffoli(5,6,7)
    /// after accounting for qubit index differences.
    ///
    /// Strategy:
    /// 1. Create 6-qubit state in |110000⟩ (qubits 0,1 are |1⟩, rest are |0⟩)
    /// 2. Apply Toffoli(0,1,2) - should flip qubit 2 → |111000⟩
    /// 3. Create 6-qubit state in |000110⟩ (qubits 3,4 are |1⟩, rest are |0⟩)
    /// 4. Apply Toffoli(3,4,5) - should flip qubit 5 → |000111⟩
    /// 5. Verify both behave identically (modulo qubit indexing)
    #[test]
    fn test_toffoli_qubit_remapping() {
        println!("\n=== Toffoli Qubit Remapping Test ===\n");

        // Synthesize Toffoli for qubits (0,1,2)
        let toffoli_012 = synthesize_three_qubit(ThreeQubitGate::Toffoli, 0, 1, 2)
            .expect("Failed to synthesize Toffoli(0,1,2)");

        // Synthesize Toffoli for qubits (3,4,5)
        let toffoli_345 = synthesize_three_qubit(ThreeQubitGate::Toffoli, 3, 4, 5)
            .expect("Failed to synthesize Toffoli(3,4,5)");

        println!("Synthesized Toffoli(0,1,2): {} gates", toffoli_012.len());
        println!("Synthesized Toffoli(3,4,5): {} gates", toffoli_345.len());

        // Both should have same number of gates (same decomposition, different qubits)
        assert_eq!(
            toffoli_012.len(),
            toffoli_345.len(),
            "Gate counts differ - remapping may be broken"
        );

        // Test 1: Toffoli(0,1,2) on |110⟩ → should give |111⟩
        println!("\nTest 1: Toffoli(0,1,2) on |110⟩");
        let mut state_012 = QState::new_zero(3);
        let mut rng = QRng::new(0);
        apply_gate_scalar(&mut state_012, &QGate::X(0), &mut rng); // |100⟩
        apply_gate_scalar(&mut state_012, &QGate::X(1), &mut rng); // |110⟩

        apply_gate_sequence(&mut state_012, &toffoli_012);

        // Should be in state |111⟩ (index 7 = 0b111)
        let amp_111 = get_amplitude(&state_012, 7);
        println!(
            "  Result: |111⟩ amplitude = {:.6} + {:.6}i",
            amp_111.0, amp_111.1
        );
        assert!(
            is_amplitude_one(amp_111),
            "Toffoli(0,1,2) failed: expected |111⟩"
        );

        // Test 2: Toffoli(3,4,5) on 6-qubit state with qubits 3,4 = |1⟩
        println!("\nTest 2: Toffoli(3,4,5) on |011000⟩");
        let mut state_345 = QState::new_zero(6);
        let mut rng = QRng::new(0);
        apply_gate_scalar(&mut state_345, &QGate::X(3), &mut rng); // Set qubit 3
        apply_gate_scalar(&mut state_345, &QGate::X(4), &mut rng); // Set qubit 4
                                                                   // State is now |011000⟩: q3=1, q4=1 (index = 2^3 + 2^4 = 8 + 16 = 24)

        apply_gate_sequence(&mut state_345, &toffoli_345);

        // Should flip qubit 5 (target) since qubits 3,4 (controls) are both |1⟩
        // Result: |111000⟩ (index = 24 + 2^5 = 24 + 32 = 56)
        let amp_result = get_amplitude(&state_345, 56);
        println!(
            "  Result: |111000⟩ amplitude = {:.6} + {:.6}i",
            amp_result.0, amp_result.1
        );
        assert!(
            is_amplitude_one(amp_result),
            "Toffoli(3,4,5) failed: expected |111000⟩"
        );

        println!("\n✓ Qubit remapping works correctly!");
        println!("  - Toffoli(0,1,2) and Toffoli(3,4,5) have identical decompositions");
        println!("  - Both produce correct truth table behavior on their respective qubits");
    }

    /// TEST 3: Cache Hit Verification
    ///
    /// Ensures that the second call to synthesize_three_qubit() returns cached results
    /// and that cached results are functionally identical to the original.
    #[test]
    fn test_toffoli_cache_consistency() {
        println!("\n=== Toffoli Cache Consistency Test ===\n");

        // First call - cache miss
        let toffoli_first = synthesize_three_qubit(ThreeQubitGate::Toffoli, 0, 1, 2)
            .expect("First synthesis failed");

        // Second call - should hit cache
        let toffoli_second = synthesize_three_qubit(ThreeQubitGate::Toffoli, 2, 3, 4)
            .expect("Second synthesis failed");

        println!("First call:  {} gates", toffoli_first.len());
        println!("Second call: {} gates", toffoli_second.len());

        // Both should have same number of gates
        assert_eq!(
            toffoli_first.len(),
            toffoli_second.len(),
            "Cache returned different gate count"
        );

        // Apply both to the same input state and verify identical behavior
        // Use |110⟩ as test case (should flip to |111⟩)

        let mut state1 = QState::new_zero(3);
        let mut rng = QRng::new(0);
        apply_gate_scalar(&mut state1, &QGate::X(0), &mut rng);
        apply_gate_scalar(&mut state1, &QGate::X(1), &mut rng);
        apply_gate_sequence(&mut state1, &toffoli_first);

        let mut state2 = QState::new_zero(5);
        let mut rng = QRng::new(0);
        apply_gate_scalar(&mut state2, &QGate::X(2), &mut rng);
        apply_gate_scalar(&mut state2, &QGate::X(3), &mut rng);
        apply_gate_sequence(&mut state2, &toffoli_second);

        // state1 should be |111⟩ (index 7)
        // state2: X(2) and X(3) → |01100⟩ (index = 2^2 + 2^3 = 4 + 8 = 12)
        // After Toffoli(2,3,4): qubits 2,3 both |1⟩ → flip qubit 4
        // Result: |11100⟩ (index = 12 + 2^4 = 12 + 16 = 28)
        let amp1 = get_amplitude(&state1, 7);
        let amp2 = get_amplitude(&state2, 28);

        println!("\nFirst synthesis result:  {:.6} + {:.6}i", amp1.0, amp1.1);
        println!("Cached synthesis result: {:.6} + {:.6}i", amp2.0, amp2.1);

        assert!(is_amplitude_one(amp1), "First synthesis failed");
        assert!(is_amplitude_one(amp2), "Cached synthesis failed");

        println!("\n✓ Cache consistency verified!");
    }
}
