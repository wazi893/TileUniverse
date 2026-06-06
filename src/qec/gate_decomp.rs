//! Gate Decomposition for Fault-Tolerant Circuits
//!
//! Provides explicit gate expansions for composite gates like Toffoli and CCZ.
//! Both estimator mode and simulation mode use these same expansions, ensuring
//! consistent T-counts and frame tracking.
//!
//! # Naming Convention
//!
//! Decompositions are named like software versions:
//! - `SevenTv1`: Standard Nielsen-Chuang (7 T, 6 CNOT)
//! - Future: `FourT_Ancilla_v1`, etc.
//!
//! # Example
//!
//! ```ignore
//! use engine::qec::gate_decomp::{ToffoliExpansion, expand_gate};
//! use engine::qec::LogicalGate;
//!
//! let gates = expand_gate(&LogicalGate::Toffoli(0, 1, 2), ToffoliExpansion::SevenTv1);
//! assert_eq!(gates.iter().filter(|g| matches!(g, LogicalGate::T(_) | LogicalGate::Tdg(_))).count(), 7);
//! ```

use super::ft_types::LogicalGate;

// ============================================================================
// Toffoli Decomposition
// ============================================================================

/// Toffoli gate decomposition strategies.
///
/// Named like software versions for clarity and future extensibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToffoliExpansion {
    /// Standard Nielsen-Chuang decomposition: 7 T gates, 6 CNOTs, T-depth 4
    ///
    /// This is the canonical exact Toffoli decomposition without ancillas.
    /// Gate sequence: H-CNOT-Tdg-CNOT-T-CNOT-Tdg-CNOT-T-T-CNOT-H-T-Tdg-CNOT
    SevenTv1,

    /// Keep as single Toffoli gate (no expansion)
    ///
    /// Use this when you want the FT processor to handle Toffoli as a primitive.
    NoExpand,
}

impl ToffoliExpansion {
    /// T-gate count for this decomposition
    pub fn t_count(&self) -> usize {
        match self {
            Self::SevenTv1 => 7,
            Self::NoExpand => 7, // Still 7 T gates worth
        }
    }

    /// CNOT count for this decomposition
    pub fn cnot_count(&self) -> usize {
        match self {
            Self::SevenTv1 => 6,
            Self::NoExpand => 0,
        }
    }

    /// T-depth for this decomposition
    pub fn t_depth(&self) -> usize {
        match self {
            Self::SevenTv1 => 4,
            Self::NoExpand => 1, // Treated as single layer
        }
    }

    /// Expand a Toffoli gate into primitive gates.
    ///
    /// Returns a sequence of H, CNOT, T, and Tdg gates that implement
    /// the Toffoli(c1, c2, target) operation.
    pub fn expand(&self, c1: usize, c2: usize, target: usize) -> Vec<LogicalGate> {
        match self {
            Self::SevenTv1 => expand_toffoli_seven_t(c1, c2, target),
            Self::NoExpand => vec![LogicalGate::Toffoli(c1, c2, target)],
        }
    }
}

impl Default for ToffoliExpansion {
    fn default() -> Self {
        Self::SevenTv1
    }
}

/// Standard Nielsen-Chuang Toffoli decomposition.
///
/// Decomposes Toffoli(c1, c2, t) into 7 T/Tdg gates and 6 CNOTs.
///
/// # Circuit Diagram
///
/// ```text
/// c1: ───────────●───────────●───T───●───────●───
///                │           │       │       │
/// c2: ───────●───┼───────●───┼───T───X───Tdg─X───
///            │   │       │   │
/// t:  ─H─────X─Tdg─X─T───X─Tdg─X─T─────H─────────
/// ```
///
/// Gate count: 2 H, 6 CNOT, 4 T, 3 Tdg = 15 gates total
fn expand_toffoli_seven_t(c1: usize, c2: usize, t: usize) -> Vec<LogicalGate> {
    vec![
        // Prepare target
        LogicalGate::H(t),
        // First CNOT-T-CNOT-T block
        LogicalGate::CNOT(c2, t),
        LogicalGate::Tdg(t),
        LogicalGate::CNOT(c1, t),
        LogicalGate::T(t),
        LogicalGate::CNOT(c2, t),
        LogicalGate::Tdg(t),
        LogicalGate::CNOT(c1, t),
        // Phase corrections on controls
        LogicalGate::T(c2),
        LogicalGate::T(t),
        LogicalGate::CNOT(c1, c2),
        // Finish target
        LogicalGate::H(t),
        // Final phase corrections
        LogicalGate::T(c1),
        LogicalGate::Tdg(c2),
        LogicalGate::CNOT(c1, c2),
    ]
}

// ============================================================================
// CCZ Decomposition
// ============================================================================

/// CCZ gate decomposition strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CCZExpansion {
    /// Convert to Toffoli with H gates: CCZ = H(c) · Toffoli · H(c)
    /// Uses SevenTv1 Toffoli internally.
    ViaToffoliv1,

    /// Keep as single CCZ gate (no expansion)
    NoExpand,
}

impl CCZExpansion {
    /// T-gate count for this decomposition
    pub fn t_count(&self) -> usize {
        match self {
            Self::ViaToffoliv1 => 7,
            Self::NoExpand => 7,
        }
    }

    /// Expand a CCZ gate into primitive gates.
    pub fn expand(&self, a: usize, b: usize, c: usize) -> Vec<LogicalGate> {
        match self {
            Self::ViaToffoliv1 => expand_ccz_via_toffoli(a, b, c),
            Self::NoExpand => vec![LogicalGate::CCZ(a, b, c)],
        }
    }
}

impl Default for CCZExpansion {
    fn default() -> Self {
        Self::ViaToffoliv1
    }
}

/// CCZ via Toffoli: CCZ(a,b,c) = H(c) · Toffoli(a,b,c) · H(c)
fn expand_ccz_via_toffoli(a: usize, b: usize, c: usize) -> Vec<LogicalGate> {
    let mut gates = vec![LogicalGate::H(c)];
    gates.extend(expand_toffoli_seven_t(a, b, c));
    gates.push(LogicalGate::H(c));
    gates
}

// ============================================================================
// Generic Gate Expansion
// ============================================================================

/// Configuration for gate expansion.
#[derive(Clone, Debug)]
pub struct ExpansionConfig {
    pub toffoli: ToffoliExpansion,
    pub ccz: CCZExpansion,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            toffoli: ToffoliExpansion::SevenTv1,
            ccz: CCZExpansion::ViaToffoliv1,
        }
    }
}

impl ExpansionConfig {
    /// Create config that doesn't expand any gates
    pub fn no_expand() -> Self {
        Self {
            toffoli: ToffoliExpansion::NoExpand,
            ccz: CCZExpansion::NoExpand,
        }
    }
}

/// Expand a gate according to the given configuration.
///
/// Returns the original gate wrapped in a Vec if no expansion is needed,
/// or the expanded gate sequence if expansion applies.
pub fn expand_gate(gate: &LogicalGate, config: &ExpansionConfig) -> Vec<LogicalGate> {
    match gate {
        LogicalGate::Toffoli(c1, c2, t) => config.toffoli.expand(*c1, *c2, *t),
        LogicalGate::CCZ(a, b, c) => config.ccz.expand(*a, *b, *c),
        // All other gates pass through unchanged
        other => vec![other.clone()],
    }
}

/// Expand an entire circuit according to the given configuration.
pub fn expand_circuit(circuit: &[LogicalGate], config: &ExpansionConfig) -> Vec<LogicalGate> {
    circuit
        .iter()
        .flat_map(|g| expand_gate(g, config))
        .collect()
}

/// Count T-gates in an expanded circuit.
pub fn count_t_gates(circuit: &[LogicalGate]) -> usize {
    circuit
        .iter()
        .filter(|g| matches!(g, LogicalGate::T(_) | LogicalGate::Tdg(_)))
        .count()
}

/// Count CNOTs in a circuit.
pub fn count_cnots(circuit: &[LogicalGate]) -> usize {
    circuit
        .iter()
        .filter(|g| matches!(g, LogicalGate::CNOT(_, _)))
        .count()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toffoli_expansion_gate_counts() {
        let expanded = ToffoliExpansion::SevenTv1.expand(0, 1, 2);

        // Count gate types
        let t_count = count_t_gates(&expanded);
        let cnot_count = count_cnots(&expanded);
        let h_count = expanded
            .iter()
            .filter(|g| matches!(g, LogicalGate::H(_)))
            .count();

        assert_eq!(t_count, 7, "Should have 7 T/Tdg gates");
        assert_eq!(cnot_count, 6, "Should have 6 CNOTs");
        assert_eq!(h_count, 2, "Should have 2 H gates");
        assert_eq!(expanded.len(), 15, "Total 15 gates");
    }

    #[test]
    fn test_toffoli_expansion_qubit_indices() {
        // Expand Toffoli(3, 5, 7) and verify qubit indices are preserved
        let expanded = ToffoliExpansion::SevenTv1.expand(3, 5, 7);

        // All gates should only touch qubits 3, 5, 7
        for gate in &expanded {
            let qubits: Vec<usize> = match gate {
                LogicalGate::H(q) => vec![*q],
                LogicalGate::T(q) => vec![*q],
                LogicalGate::Tdg(q) => vec![*q],
                LogicalGate::CNOT(c, t) => vec![*c, *t],
                _ => vec![],
            };
            for q in qubits {
                assert!(
                    q == 3 || q == 5 || q == 7,
                    "Gate {:?} touches unexpected qubit {}",
                    gate,
                    q
                );
            }
        }
    }

    #[test]
    fn test_ccz_expansion() {
        let expanded = CCZExpansion::ViaToffoliv1.expand(0, 1, 2);

        // Should be H + Toffoli + H = 2 + 15 = 17 gates
        assert_eq!(expanded.len(), 17, "CCZ should expand to 17 gates");

        // First and last should be H on the third qubit
        assert!(
            matches!(expanded.first(), Some(LogicalGate::H(2))),
            "Should start with H(2)"
        );
        assert!(
            matches!(expanded.last(), Some(LogicalGate::H(2))),
            "Should end with H(2)"
        );

        // T-count should still be 7
        assert_eq!(count_t_gates(&expanded), 7);
    }

    #[test]
    fn test_expand_circuit() {
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::Toffoli(0, 1, 2),
            LogicalGate::MeasureZ(2),
        ];

        let expanded = expand_circuit(&circuit, &ExpansionConfig::default());

        // H + 15 (Toffoli) + Measure = 17 gates
        assert_eq!(expanded.len(), 17);

        // First gate should still be H(0)
        assert!(matches!(expanded[0], LogicalGate::H(0)));

        // Last gate should still be MeasureZ(2)
        assert!(matches!(expanded.last(), Some(LogicalGate::MeasureZ(2))));
    }

    #[test]
    fn test_no_expand_config() {
        let circuit = vec![LogicalGate::Toffoli(0, 1, 2)];

        let expanded = expand_circuit(&circuit, &ExpansionConfig::no_expand());

        // Should not expand
        assert_eq!(expanded.len(), 1);
        assert!(matches!(expanded[0], LogicalGate::Toffoli(0, 1, 2)));
    }

    #[test]
    fn test_t_gate_distribution() {
        // Verify T gates are on the right qubits
        let expanded = ToffoliExpansion::SevenTv1.expand(0, 1, 2);

        let mut t_on_c1 = 0;
        let mut t_on_c2 = 0;
        let mut t_on_target = 0;

        for gate in &expanded {
            match gate {
                LogicalGate::T(0) | LogicalGate::Tdg(0) => t_on_c1 += 1,
                LogicalGate::T(1) | LogicalGate::Tdg(1) => t_on_c2 += 1,
                LogicalGate::T(2) | LogicalGate::Tdg(2) => t_on_target += 1,
                _ => {}
            }
        }

        // Standard decomposition: 1 T on c1, 2 T/Tdg on c2, 4 T/Tdg on target
        assert_eq!(t_on_c1, 1, "Control 1 should have 1 T gate");
        assert_eq!(t_on_c2, 2, "Control 2 should have 2 T/Tdg gates");
        assert_eq!(t_on_target, 4, "Target should have 4 T/Tdg gates");
    }

    // =========================================================================
    // Phase 1.2: Toffoli Truth-Table Tests
    //
    // Two categories of tests:
    // 1. PRIMITIVE Toffoli tests - verify the LogicalGate::Toffoli works correctly
    //    with the FT processor's Z-basis state tracking
    // 2. EXPANDED Toffoli tests - verify resource counts and frame consistency
    //
    // NOTE: The expanded Toffoli (7T decomposition) requires full quantum state
    // tracking through superposition (the H gates put target in X-basis), which
    // the Pauli frame simulation model doesn't support. The expanded decomposition's
    // mathematical correctness should be verified separately via unitary comparison
    // or a full state vector simulator.
    // =========================================================================

    use crate::qec::ft_processor::{FTConfig, FaultTolerantProcessor};

    /// Prepare a qubit in |0⟩ or |1⟩ state
    fn prepare_computational_basis(
        processor: &mut FaultTolerantProcessor,
        qubit: usize,
        value: bool,
    ) {
        // Reset to |0⟩ then optionally apply X
        processor.execute(&[LogicalGate::Reset(qubit)]);
        if value {
            processor.execute(&[LogicalGate::X(qubit)]);
        }
    }

    /// Run PRIMITIVE Toffoli and return corrected measurement outcomes
    fn run_primitive_toffoli(
        c1_val: bool,
        c2_val: bool,
        t_val: bool,
        seed: u64,
    ) -> (bool, bool, bool) {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(seed);

        // Prepare inputs: c1=qubit0, c2=qubit1, target=qubit2
        prepare_computational_basis(&mut processor, 0, c1_val);
        prepare_computational_basis(&mut processor, 1, c2_val);
        prepare_computational_basis(&mut processor, 2, t_val);

        // Use primitive Toffoli (Z-basis tracking works)
        processor.execute(&[LogicalGate::Toffoli(0, 1, 2)]);

        // Measure all qubits
        let measurements = processor.execute(&[
            LogicalGate::MeasureZ(0),
            LogicalGate::MeasureZ(1),
            LogicalGate::MeasureZ(2),
        ]);

        // Use corrected_value() for frame-aware interpretation
        let m0 = measurements.measurement_outcomes[0].corrected_value();
        let m1 = measurements.measurement_outcomes[1].corrected_value();
        let m2 = measurements.measurement_outcomes[2].corrected_value();

        (m0, m1, m2)
    }

    /// Toffoli truth table: target flips IFF both controls are 1
    fn toffoli_expected(c1: bool, c2: bool, t: bool) -> (bool, bool, bool) {
        let t_out = t ^ (c1 && c2);
        (c1, c2, t_out)
    }

    // =========================================================================
    // Primitive Toffoli Truth Table Tests
    // =========================================================================

    #[test]
    fn primitive_toffoli_truth_table_all_8_inputs() {
        // Test all 8 input combinations using PRIMITIVE Toffoli
        // The primitive implementation has Z-basis state tracking
        let test_cases = [
            (false, false, false), // |000⟩ → |000⟩
            (false, false, true),  // |001⟩ → |001⟩
            (false, true, false),  // |010⟩ → |010⟩
            (false, true, true),   // |011⟩ → |011⟩
            (true, false, false),  // |100⟩ → |100⟩
            (true, false, true),   // |101⟩ → |101⟩
            (true, true, false),   // |110⟩ → |111⟩ (target flips!)
            (true, true, true),    // |111⟩ → |110⟩ (target flips!)
        ];

        for (c1, c2, t) in test_cases {
            let expected = toffoli_expected(c1, c2, t);

            // Test with multiple seeds (though primitive Toffoli is deterministic for Z-basis)
            for seed in [42, 123, 456, 789, 1000] {
                let (m0, m1, m2) = run_primitive_toffoli(c1, c2, t, seed);

                assert_eq!(
                    (m0, m1, m2),
                    expected,
                    "Primitive Toffoli failed for input |{}{}{} with seed {}: \
                     expected |{}{}{}⟩, got |{}{}{}⟩",
                    c1 as u8,
                    c2 as u8,
                    t as u8,
                    seed,
                    expected.0 as u8,
                    expected.1 as u8,
                    expected.2 as u8,
                    m0 as u8,
                    m1 as u8,
                    m2 as u8
                );
            }
        }
    }

    #[test]
    fn primitive_toffoli_controls_unchanged() {
        // Verify that control qubits are never modified (only target changes)
        for seed in 0..50 {
            for c1 in [false, true] {
                for c2 in [false, true] {
                    let (m0, m1, _) = run_primitive_toffoli(c1, c2, false, seed);
                    assert_eq!(
                        m0, c1,
                        "Control 1 should be unchanged: expected {}, got {} (seed {})",
                        c1, m0, seed
                    );
                    assert_eq!(
                        m1, c2,
                        "Control 2 should be unchanged: expected {}, got {} (seed {})",
                        c2, m1, seed
                    );
                }
            }
        }
    }

    #[test]
    fn primitive_toffoli_flip_only_when_both_controls_set() {
        // The defining property of Toffoli: target XOR (c1 AND c2)
        for seed in 0..30 {
            // When at least one control is 0, target unchanged
            assert_eq!(run_primitive_toffoli(false, false, false, seed).2, false);
            assert_eq!(run_primitive_toffoli(false, false, true, seed).2, true);
            assert_eq!(run_primitive_toffoli(false, true, false, seed).2, false);
            assert_eq!(run_primitive_toffoli(false, true, true, seed).2, true);
            assert_eq!(run_primitive_toffoli(true, false, false, seed).2, false);
            assert_eq!(run_primitive_toffoli(true, false, true, seed).2, true);

            // When BOTH controls are 1, target flips
            assert_eq!(run_primitive_toffoli(true, true, false, seed).2, true);
            assert_eq!(run_primitive_toffoli(true, true, true, seed).2, false);
        }
    }

    #[test]
    fn primitive_toffoli_verifies_all_corrections() {
        // Verify that verify_correction() passes for all measurements
        let test_cases = [
            (false, false, false),
            (false, false, true),
            (false, true, false),
            (false, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
        ];

        for (c1, c2, t) in test_cases {
            for seed in 0..20 {
                let config = FTConfig::new(7, 1e-3).with_sampled_injection();
                let mut processor = FaultTolerantProcessor::new(3, config).with_seed(seed);

                prepare_computational_basis(&mut processor, 0, c1);
                prepare_computational_basis(&mut processor, 1, c2);
                prepare_computational_basis(&mut processor, 2, t);

                processor.execute(&[LogicalGate::Toffoli(0, 1, 2)]);

                let result = processor.execute(&[
                    LogicalGate::MeasureZ(0),
                    LogicalGate::MeasureZ(1),
                    LogicalGate::MeasureZ(2),
                ]);

                for (i, m) in result.measurement_outcomes.iter().enumerate() {
                    assert!(
                        m.verify_correction(),
                        "Measurement {} failed verify_correction() for input |{}{}{}⟩ seed {}",
                        i,
                        c1 as u8,
                        c2 as u8,
                        t as u8,
                        seed
                    );
                }
            }
        }
    }

    // =========================================================================
    // Primitive CCZ Truth Table Tests
    // =========================================================================

    #[test]
    fn primitive_ccz_truth_table_all_8_inputs() {
        // CCZ applies a phase flip when all 3 inputs are 1
        // For computational basis states, this is unobservable in Z-basis measurement
        // So all states should be preserved
        let test_cases = [
            (false, false, false),
            (false, false, true),
            (false, true, false),
            (false, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
        ];

        for (a, b, c) in test_cases {
            for seed in [42, 123, 456] {
                let config = FTConfig::new(7, 1e-3).with_sampled_injection();
                let mut processor = FaultTolerantProcessor::new(3, config).with_seed(seed);

                prepare_computational_basis(&mut processor, 0, a);
                prepare_computational_basis(&mut processor, 1, b);
                prepare_computational_basis(&mut processor, 2, c);

                // Use primitive CCZ
                processor.execute(&[LogicalGate::CCZ(0, 1, 2)]);

                let result = processor.execute(&[
                    LogicalGate::MeasureZ(0),
                    LogicalGate::MeasureZ(1),
                    LogicalGate::MeasureZ(2),
                ]);

                // CCZ doesn't change computational basis states (only adds phase)
                assert_eq!(
                    result.measurement_outcomes[0].corrected_value(),
                    a,
                    "CCZ changed qubit 0 for input |{}{}{}⟩ seed {}",
                    a as u8,
                    b as u8,
                    c as u8,
                    seed
                );
                assert_eq!(
                    result.measurement_outcomes[1].corrected_value(),
                    b,
                    "CCZ changed qubit 1 for input |{}{}{}⟩ seed {}",
                    a as u8,
                    b as u8,
                    c as u8,
                    seed
                );
                assert_eq!(
                    result.measurement_outcomes[2].corrected_value(),
                    c,
                    "CCZ changed qubit 2 for input |{}{}{}⟩ seed {}",
                    a as u8,
                    b as u8,
                    c as u8,
                    seed
                );
            }
        }
    }

    // =========================================================================
    // Expanded Toffoli Resource Tests
    //
    // NOTE: These tests verify resource counts and frame consistency, NOT
    // mathematical correctness. The expanded Toffoli requires full state
    // tracking through superposition which the Pauli frame model doesn't support.
    // =========================================================================

    #[test]
    fn expanded_toffoli_resource_counts() {
        // Verify expanded Toffoli consumes correct resources
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

        prepare_computational_basis(&mut processor, 0, true);
        prepare_computational_basis(&mut processor, 1, true);
        prepare_computational_basis(&mut processor, 2, false);

        let expanded = ToffoliExpansion::SevenTv1.expand(0, 1, 2);
        let result = processor.execute(&expanded);

        // Should consume exactly 7 magic states (7 T/Tdg gates)
        assert_eq!(
            result.stats.magic_states_consumed, 7,
            "Should consume 7 magic states"
        );
        assert_eq!(result.stats.t_gates, 7, "Should have 7 T-gates");
    }

    #[test]
    fn expanded_toffoli_frame_consistency() {
        // Verify frame tracking is internally consistent through expanded Toffoli
        for seed in 0..50 {
            let config = FTConfig::new(7, 1e-3).with_sampled_injection();
            let mut processor = FaultTolerantProcessor::new(3, config).with_seed(seed);

            prepare_computational_basis(&mut processor, 0, true);
            prepare_computational_basis(&mut processor, 1, false);
            prepare_computational_basis(&mut processor, 2, true);

            let expanded = ToffoliExpansion::SevenTv1.expand(0, 1, 2);
            processor.execute(&expanded);

            let result = processor.execute(&[
                LogicalGate::MeasureZ(0),
                LogicalGate::MeasureZ(1),
                LogicalGate::MeasureZ(2),
            ]);

            // All measurements should have valid frame corrections
            for (i, m) in result.measurement_outcomes.iter().enumerate() {
                assert!(
                    m.verify_correction(),
                    "Measurement {} failed verify_correction() for seed {}",
                    i,
                    seed
                );
            }
        }
    }

    #[test]
    fn expanded_toffoli_traced_injection_count() {
        // Trace through expanded Toffoli and verify 7 T-injection steps
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

        prepare_computational_basis(&mut processor, 0, true);
        prepare_computational_basis(&mut processor, 1, true);
        prepare_computational_basis(&mut processor, 2, false);

        let expanded = ToffoliExpansion::SevenTv1.expand(0, 1, 2);
        let trace = processor.execute_traced(&expanded);

        // Verify we got 15 steps (15 gates in expansion)
        assert_eq!(trace.len(), 15, "Should have 15 traced steps");

        // Count T-gates that have injection outcomes
        let t_injections: Vec<_> = trace
            .iter()
            .filter(|s| s.injection_outcome.is_some())
            .collect();

        assert_eq!(t_injections.len(), 7, "Should have 7 T-injection steps");
    }

    #[test]
    fn expanded_ccz_resource_counts() {
        // CCZ expansion = H + Toffoli + H = 17 gates, 7 T-gates
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

        prepare_computational_basis(&mut processor, 0, true);
        prepare_computational_basis(&mut processor, 1, true);
        prepare_computational_basis(&mut processor, 2, true);

        let expanded = CCZExpansion::ViaToffoliv1.expand(0, 1, 2);
        assert_eq!(expanded.len(), 17, "CCZ expansion should be 17 gates");

        let result = processor.execute(&expanded);
        assert_eq!(
            result.stats.magic_states_consumed, 7,
            "Should consume 7 magic states"
        );
        assert_eq!(result.stats.t_gates, 7, "Should have 7 T-gates");
    }
}
