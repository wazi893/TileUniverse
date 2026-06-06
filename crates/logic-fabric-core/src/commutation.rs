//! SPRINT 37.0: Comprehensive Commutation-Based Optimization
//!
//! This module implements Qiskit-style commutation analysis for quantum circuit
//! optimization. The key insight is that gates don't need to be adjacent to cancel -
//! with proper commutation analysis, we can bring any cancellable pair together.
//!
//! ## Key Features
//!
//! 1. **Extended Commutation Rules**: Comprehensive rules for when gates commute,
//!    including CNOT-CNOT, CNOT-CZ, and single-qubit through two-qubit gates.
//!
//! 2. **Long-Range Cancellation**: Finds non-adjacent cancellable pairs and uses
//!    commutation to bring them together.
//!
//! 3. **Fusion-Oriented Commutation**: Reorders gates to bring same-qubit single-qubit
//!    gates together for better fusion.
//!
//! ## Commutation Rules Summary
//!
//! | Gate A | Gate B | Commutes? | Condition |
//! |--------|--------|-----------|-----------|
//! | Any | Any | Yes | Disjoint qubits |
//! | Z-axis | Z-axis | Yes | Same qubit (Rz, Z, Phase) |
//! | X-axis | X-axis | Yes | Same qubit (Rx, X) |
//! | Y-axis | Y-axis | Yes | Same qubit (Ry, Y) |
//! | Rz/Phase | CNOT | Yes | On control qubit |
//! | Rx/X | CNOT | Yes | On target qubit |
//! | Z/Rz/Phase | CZ | Yes | Either qubit |
//! | CNOT | CNOT | Yes | Same control, different targets |
//! | CNOT | CNOT | Yes | Same target, different controls |

use crate::algebraic_fusion::{
    are_inverse_gates, cancel_inverse_fixpoint, fuse_general_single_qubit, gates_commute,
    get_gate_qubits,
};
use crate::quantum::QGate;

// ============================================================================
// EXTENDED COMMUTATION RULES
// ============================================================================

/// Extended commutation check with comprehensive rules.
///
/// This extends the basic `gates_commute` with additional rules especially for
/// gates interacting with two-qubit gates.
pub fn gates_commute_extended(a: &QGate, b: &QGate) -> bool {
    // First check the basic rules
    if gates_commute(a, b) {
        return true;
    }

    let qubits_a = get_gate_qubits(a);
    let qubits_b = get_gate_qubits(b);

    // Gates on disjoint qubits always commute
    if qubits_a.iter().all(|q| !qubits_b.contains(q)) {
        return true;
    }

    // Additional rules for comprehensive commutation
    match (a, b) {
        // CNOT commutes with CNOT on disjoint qubits (already covered by disjoint check)
        // But also: Two CNOTs with same control but different targets commute
        (QGate::CNot(c1, t1), QGate::CNot(c2, t2)) if c1 == c2 && t1 != t2 => true,

        // Two CNOTs with same target but different controls commute
        (QGate::CNot(c1, t1), QGate::CNot(c2, t2)) if t1 == t2 && c1 != c2 => true,

        // CZ commutes with CZ on any configuration (CZ is symmetric and diagonal in Z-basis)
        (QGate::CZ(_a1, _b1), QGate::CZ(_a2, _b2)) => {
            // CZs commute unless they share exactly one qubit AND would interact
            // Actually, CZs always commute with each other!
            // CZ = diag(1, 1, 1, -1) - diagonal matrices commute
            true
        }

        // CNOT with same control: control qubit experiences same Z-action
        // CNOT(c,t1) and CNOT(c,t2) commute because Z on c is unchanged

        // Additional: Ry on control does NOT commute with CNOT (unlike Rz)
        // Ry on target does NOT commute with CNOT (unlike Rx)
        _ => false,
    }
}

/// Get target qubit for single-qubit gates
pub fn get_single_qubit_target(gate: &QGate) -> Option<u8> {
    match gate {
        QGate::H(q) | QGate::X(q) | QGate::Y(q) | QGate::Z(q) => Some(*q),
        QGate::Rx(q, _) | QGate::Ry(q, _) | QGate::Rz(q, _) => Some(*q),
        QGate::Phase(q, _) | QGate::U3(q, _, _, _) => Some(*q),
        _ => None,
    }
}

// ============================================================================
// LONG-RANGE CANCELLATION
// ============================================================================

/// Check if two gates would cancel each other (are mutual inverses)
fn would_cancel(g1: &QGate, g2: &QGate) -> bool {
    are_inverse_gates(g1, g2)
}

/// Find all pairs of gates that would cancel if brought together.
/// Returns pairs of indices (earlier_idx, later_idx).
fn find_cancellable_pairs(gates: &[QGate]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();

    for i in 0..gates.len() {
        for j in (i + 1)..gates.len() {
            if would_cancel(&gates[i], &gates[j]) {
                pairs.push((i, j));
            }
        }
    }

    pairs
}

/// Try to bring a cancellable pair together by commuting intervening gates.
/// Returns the modified gate list if successful, None if not possible.
fn try_commute_to_cancel(gates: &[QGate], i: usize, j: usize) -> Option<Vec<QGate>> {
    if j <= i || j >= gates.len() {
        return None;
    }

    let g_j = &gates[j];

    // Check if all intervening gates can commute with g_j
    // We'll move g_j backward to be adjacent to g_i
    for k in (i + 1)..j {
        let intervening = &gates[k];
        if !gates_commute_extended(g_j, intervening) {
            return None;
        }
    }

    // Success! g_i and g_j cancel
    let mut result = Vec::with_capacity(gates.len() - 2);

    // Add gates before i
    result.extend(gates[0..i].iter().cloned());

    // Skip g_i (it will cancel)
    // Add intervening gates
    result.extend(gates[(i + 1)..j].iter().cloned());

    // Skip g_j (it cancels with g_i)
    // Add gates after j
    result.extend(gates[(j + 1)..].iter().cloned());

    Some(result)
}

/// Perform one round of commutation-based cancellation.
fn commutation_cancel_pass(gates: &[QGate]) -> (Vec<QGate>, bool) {
    if gates.len() < 2 {
        return (gates.to_vec(), false);
    }

    // Find all cancellable pairs
    let pairs = find_cancellable_pairs(gates);

    // Try each pair in order of proximity (closer pairs are easier)
    let mut sorted_pairs = pairs;
    sorted_pairs.sort_by_key(|(i, j)| j - i);

    for (i, j) in sorted_pairs {
        if let Some(reduced) = try_commute_to_cancel(gates, i, j) {
            return (reduced, true);
        }
    }

    (gates.to_vec(), false)
}

/// Run commutation-based cancellation until fixpoint.
pub fn commutation_cancellation_fixpoint(gates: Vec<QGate>) -> Vec<QGate> {
    let mut current = gates;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    loop {
        let (next, changed) = commutation_cancel_pass(&current);
        iterations += 1;

        if !changed || iterations >= MAX_ITERATIONS {
            return next;
        }
        current = next;
    }
}

// ============================================================================
// FUSION-ORIENTED COMMUTATION
// ============================================================================

/// Move gates to bring same-qubit single-qubit gates together.
fn commute_for_fusion_pass(gates: &[QGate]) -> (Vec<QGate>, bool) {
    if gates.len() < 3 {
        return (gates.to_vec(), false);
    }

    let mut result = gates.to_vec();
    let mut changed = false;

    // For each single-qubit gate, try to move it toward other gates on the same qubit
    for i in 0..result.len().saturating_sub(1) {
        let target_q = match get_single_qubit_target(&result[i]) {
            Some(q) => q,
            None => continue,
        };

        // Look ahead for another single-qubit gate on the same qubit
        for j in (i + 2)..result.len() {
            if let Some(q) = get_single_qubit_target(&result[j]) {
                if q == target_q {
                    // Found a gate on the same qubit. Try to swap gates[i] and gates[i+1]
                    if i + 1 < result.len() && gates_commute_extended(&result[i], &result[i + 1]) {
                        result.swap(i, i + 1);
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    (result, changed)
}

/// Run fusion-oriented commutation until fixpoint
pub fn commute_for_fusion_fixpoint(gates: Vec<QGate>) -> Vec<QGate> {
    let mut current = gates;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 50;

    loop {
        let (next, changed) = commute_for_fusion_pass(&current);
        iterations += 1;

        if !changed || iterations >= MAX_ITERATIONS {
            return next;
        }
        current = next;
    }
}

// ============================================================================
// COMPREHENSIVE OPTIMIZATION
// ============================================================================

/// Combined commutation optimization pass.
/// 1. Commute for fusion (bring same-qubit gates together)
/// 2. Fuse single-qubit gates
/// 3. Cancel inverse pairs via commutation
/// 4. Repeat until fixpoint
pub fn comprehensive_commutation_optimize(gates: Vec<QGate>) -> Vec<QGate> {
    let mut current = gates;
    let mut prev_len = usize::MAX;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;

    while current.len() < prev_len && iterations < MAX_ITERATIONS {
        prev_len = current.len();
        iterations += 1;

        // Step 1: Commute for fusion
        current = commute_for_fusion_fixpoint(current);

        // Step 2: Fuse single-qubit gates
        // SPRINT 36.0: Re-enabled after fixing global phase bug in fuse_general_single_qubit
        current = fuse_general_single_qubit(&current);

        // Step 3: Cancel inverse pairs via commutation
        current = commutation_cancellation_fixpoint(current);

        // Step 4: Adjacent cancellation (fast path)
        current = cancel_inverse_fixpoint(current);
    }

    current
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnot_same_control_commute() {
        // CNOT(0,1) and CNOT(0,2) should commute (same control, different targets)
        let a = QGate::CNot(0, 1);
        let b = QGate::CNot(0, 2);
        assert!(gates_commute_extended(&a, &b));
    }

    #[test]
    fn test_cnot_same_target_commute() {
        // CNOT(0,2) and CNOT(1,2) should commute (different controls, same target)
        let a = QGate::CNot(0, 2);
        let b = QGate::CNot(1, 2);
        assert!(gates_commute_extended(&a, &b));
    }

    #[test]
    fn test_cz_always_commute() {
        // CZs are diagonal and always commute
        let a = QGate::CZ(0, 1);
        let b = QGate::CZ(1, 2);
        assert!(gates_commute_extended(&a, &b));
    }

    #[test]
    fn test_long_range_cnot_cancellation() {
        // CNOT(0,1) - Rz(2,θ) - CNOT(0,1) should cancel the CNOTs
        // because Rz(2,θ) commutes with both CNOTs (disjoint qubits)
        let gates = vec![QGate::CNot(0, 1), QGate::Rz(2, 0.5), QGate::CNot(0, 1)];

        let result = commutation_cancellation_fixpoint(gates);

        // Should reduce to just Rz(2,θ)
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], QGate::Rz(2, _)));
    }

    #[test]
    fn test_long_range_h_cancellation() {
        // H(0) - X(1) - H(0) should cancel because X(1) commutes with H(0)
        let gates = vec![QGate::H(0), QGate::X(1), QGate::H(0)];

        let result = commutation_cancellation_fixpoint(gates);

        // Should reduce to just X(1)
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], QGate::X(1)));
    }

    #[test]
    fn test_comprehensive_optimization() {
        // Test a more complex circuit
        let gates = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::Rz(2, 0.5),
            QGate::CNot(0, 1), // Should cancel with first CNOT
            QGate::H(0),       // Should cancel with first H
        ];

        let result = comprehensive_commutation_optimize(gates);

        // Should reduce significantly
        assert!(result.len() <= 2, "Expected ≤2 gates, got {}", result.len());
    }

    #[test]
    fn test_fusion_commutation() {
        // Rz(0) - CNOT(1,2) - Rz(0) should commute Rz gates together
        // then fuse them
        let gates = vec![QGate::Rz(0, 0.5), QGate::CNot(1, 2), QGate::Rz(0, 0.3)];

        let result = comprehensive_commutation_optimize(gates);

        // Rz gates should be fused
        // Result: CNOT(1,2) + fused Rz(0, 0.8) or vice versa
        assert_eq!(result.len(), 2);
    }
}
