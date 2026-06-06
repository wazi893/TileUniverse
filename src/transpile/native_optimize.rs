//! SPRINT 54.0: Post-Transpile Native Gate Optimization
//!
//! This module optimizes circuits that have already been transpiled to
//! hardware-native gate sets. It applies hardware-specific algebraic
//! simplifications that only make sense after transpilation.
//!
//! ## IBM Native Optimizations
//!
//! | Pattern | Optimization | Savings |
//! |---------|--------------|---------|
//! | CX·CX | → Identity | 2 CX gates |
//! | SX·SX | → X | 1 SX gate (X is virtual) |
//! | Rz(θ₁)·Rz(θ₂) | → Rz(θ₁+θ₂) | 1 Rz gate |
//! | Rz(0) | → Identity | 1 Rz gate |
//! | Rz(2π) | → Identity | 1 Rz gate |
//!
//! ## IonQ Native Optimizations
//!
//! | Pattern | Optimization | Savings |
//! |---------|--------------|---------|
//! | MS·MS (same qubits, inverse) | → Identity | 2 MS gates |
//! | GPI(θ)·GPI(θ+π) | → Identity | 2 GPI gates |
//! | GPI2(θ)·GPI2(θ+π) | → Identity | 2 GPI2 gates |

use super::native_gates::{IBMCircuit, IBMGate, IonQCircuit, IonQGate};
use std::f64::consts::{PI, TAU};

/// Tolerance for angle comparison
const ANGLE_EPSILON: f64 = 1e-9;

/// Check if an angle is effectively zero (mod 2π)
fn is_zero_angle(angle: f64) -> bool {
    let normalized = angle.rem_euclid(TAU);
    normalized.abs() < ANGLE_EPSILON || (TAU - normalized).abs() < ANGLE_EPSILON
}

/// Check if two angles are equal (mod 2π)
fn angles_equal(a: f64, b: f64) -> bool {
    let diff = (a - b).rem_euclid(TAU);
    diff.abs() < ANGLE_EPSILON || (TAU - diff).abs() < ANGLE_EPSILON
}

/// Check if two angles differ by π (mod 2π)
fn angles_differ_by_pi(a: f64, b: f64) -> bool {
    let diff = (b - a).rem_euclid(TAU);
    (diff - PI).abs() < ANGLE_EPSILON
}

// ============================================================================
// IBM NATIVE OPTIMIZATION
// ============================================================================

/// Statistics from IBM circuit optimization
#[derive(Clone, Debug, Default)]
pub struct IBMOptimizationStats {
    /// Original gate count
    pub original_gates: usize,
    /// Final gate count
    pub final_gates: usize,
    /// CX gates cancelled
    pub cx_cancelled: usize,
    /// SX gates reduced
    pub sx_reduced: usize,
    /// Rz gates merged
    pub rz_merged: usize,
    /// Rz gates eliminated (zero angle)
    pub rz_eliminated: usize,
}

impl IBMOptimizationStats {
    pub fn reduction_percent(&self) -> f64 {
        if self.original_gates == 0 {
            return 0.0;
        }
        let reduced = self.original_gates.saturating_sub(self.final_gates);
        (reduced as f64 / self.original_gates as f64) * 100.0
    }
}

/// Optimize an IBM native circuit
pub fn optimize_ibm_circuit(circuit: &IBMCircuit) -> (IBMCircuit, IBMOptimizationStats) {
    let mut stats = IBMOptimizationStats {
        original_gates: circuit.gates.len(),
        ..Default::default()
    };

    if circuit.gates.is_empty() {
        stats.final_gates = 0;
        return (circuit.clone(), stats);
    }

    let mut optimized = circuit.gates.clone();
    let mut changed = true;

    // Iterate until fixpoint
    while changed {
        changed = false;

        // Pass 1: Cancel adjacent CX gates
        let (new_gates, cx_cancelled) = cancel_adjacent_cx(&optimized);
        if cx_cancelled > 0 {
            optimized = new_gates;
            stats.cx_cancelled += cx_cancelled;
            changed = true;
        }

        // Pass 2: Merge adjacent Rz gates
        let (new_gates, rz_merged) = merge_adjacent_rz(&optimized);
        if rz_merged > 0 {
            optimized = new_gates;
            stats.rz_merged += rz_merged;
            changed = true;
        }

        // Pass 3: Eliminate zero-angle Rz gates
        let (new_gates, rz_eliminated) = eliminate_zero_rz(&optimized);
        if rz_eliminated > 0 {
            optimized = new_gates;
            stats.rz_eliminated += rz_eliminated;
            changed = true;
        }

        // Pass 4: Reduce SX·SX to X (if beneficial)
        let (new_gates, sx_reduced) = reduce_sx_pairs(&optimized);
        if sx_reduced > 0 {
            optimized = new_gates;
            stats.sx_reduced += sx_reduced;
            changed = true;
        }
    }

    stats.final_gates = optimized.len();

    let result = IBMCircuit {
        name: circuit.name.clone(),
        num_qubits: circuit.num_qubits,
        num_clbits: circuit.num_clbits,
        gates: optimized,
    };

    (result, stats)
}

/// Cancel adjacent CX gates on same qubits
fn cancel_adjacent_cx(gates: &[IBMGate]) -> (Vec<IBMGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut cancelled = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            // Check for CX·CX pattern
            if let (IBMGate::CX(c1, t1), IBMGate::CX(c2, t2)) = (&gates[i], &gates[i + 1]) {
                if c1 == c2 && t1 == t2 {
                    // CX·CX = Identity, skip both
                    cancelled += 2;
                    i += 2;
                    continue;
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, cancelled)
}

/// Merge adjacent Rz gates on same qubit
fn merge_adjacent_rz(gates: &[IBMGate]) -> (Vec<IBMGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut merged = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            // Check for Rz·Rz pattern
            if let (IBMGate::Rz(q1, a1), IBMGate::Rz(q2, a2)) = (&gates[i], &gates[i + 1]) {
                if q1 == q2 {
                    // Merge: Rz(θ₁)·Rz(θ₂) = Rz(θ₁+θ₂)
                    let combined_angle = (a1 + a2).rem_euclid(TAU);
                    result.push(IBMGate::Rz(*q1, combined_angle));
                    merged += 1;
                    i += 2;
                    continue;
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, merged)
}

/// Eliminate Rz gates with zero angle
fn eliminate_zero_rz(gates: &[IBMGate]) -> (Vec<IBMGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut eliminated = 0;

    for gate in gates {
        if let IBMGate::Rz(_, angle) = gate {
            if is_zero_angle(*angle) {
                eliminated += 1;
                continue;
            }
        }
        result.push(gate.clone());
    }

    (result, eliminated)
}

/// Reduce SX·SX pairs to X
fn reduce_sx_pairs(gates: &[IBMGate]) -> (Vec<IBMGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut reduced = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            // Check for SX·SX pattern
            if let (IBMGate::SX(q1), IBMGate::SX(q2)) = (&gates[i], &gates[i + 1]) {
                if q1 == q2 {
                    // SX·SX = X (but X = Rz(π)·SX·Rz(π) in IBM basis)
                    // For simplicity, we can represent this as X directly
                    result.push(IBMGate::X(*q1));
                    reduced += 1; // Reduced 2 SX to 1 X
                    i += 2;
                    continue;
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, reduced)
}

// ============================================================================
// IONQ NATIVE OPTIMIZATION
// ============================================================================

/// Statistics from IonQ circuit optimization
#[derive(Clone, Debug, Default)]
pub struct IonQOptimizationStats {
    /// Original gate count
    pub original_gates: usize,
    /// Final gate count
    pub final_gates: usize,
    /// MS gates cancelled
    pub ms_cancelled: usize,
    /// GPI gates merged
    pub gpi_merged: usize,
    /// GPI2 gates merged
    pub gpi2_merged: usize,
    /// Identity gates eliminated
    pub identities_eliminated: usize,
}

impl IonQOptimizationStats {
    pub fn reduction_percent(&self) -> f64 {
        if self.original_gates == 0 {
            return 0.0;
        }
        let reduced = self.original_gates.saturating_sub(self.final_gates);
        (reduced as f64 / self.original_gates as f64) * 100.0
    }
}

/// Optimize an IonQ native circuit
pub fn optimize_ionq_circuit(circuit: &IonQCircuit) -> (IonQCircuit, IonQOptimizationStats) {
    let mut stats = IonQOptimizationStats {
        original_gates: circuit.gates.len(),
        ..Default::default()
    };

    if circuit.gates.is_empty() {
        stats.final_gates = 0;
        return (circuit.clone(), stats);
    }

    let mut optimized = circuit.gates.clone();
    let mut changed = true;

    // Iterate until fixpoint
    while changed {
        changed = false;

        // Pass 1: Cancel inverse GPI gates
        let (new_gates, gpi_cancelled) = cancel_inverse_gpi(&optimized);
        if gpi_cancelled > 0 {
            optimized = new_gates;
            stats.gpi_merged += gpi_cancelled;
            changed = true;
        }

        // Pass 2: Cancel inverse GPI2 gates
        let (new_gates, gpi2_cancelled) = cancel_inverse_gpi2(&optimized);
        if gpi2_cancelled > 0 {
            optimized = new_gates;
            stats.gpi2_merged += gpi2_cancelled;
            changed = true;
        }

        // Pass 3: Cancel inverse MS gates
        let (new_gates, ms_cancelled) = cancel_inverse_ms(&optimized);
        if ms_cancelled > 0 {
            optimized = new_gates;
            stats.ms_cancelled += ms_cancelled;
            changed = true;
        }
    }

    stats.final_gates = optimized.len();

    let result = IonQCircuit {
        name: circuit.name.clone(),
        num_qubits: circuit.num_qubits,
        gates: optimized,
    };

    (result, stats)
}

/// Cancel inverse GPI gates: GPI(θ)·GPI(θ+π) = Identity
fn cancel_inverse_gpi(gates: &[IonQGate]) -> (Vec<IonQGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut cancelled = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            if let (IonQGate::GPI(q1, phi1), IonQGate::GPI(q2, phi2)) = (&gates[i], &gates[i + 1]) {
                if q1 == q2 && angles_differ_by_pi(*phi1, *phi2) {
                    // GPI(φ)·GPI(φ+π) = I
                    cancelled += 2;
                    i += 2;
                    continue;
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, cancelled)
}

/// Cancel inverse GPI2 gates: GPI2(θ)·GPI2(θ+π) = Identity
fn cancel_inverse_gpi2(gates: &[IonQGate]) -> (Vec<IonQGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut cancelled = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            if let (IonQGate::GPI2(q1, phi1), IonQGate::GPI2(q2, phi2)) = (&gates[i], &gates[i + 1])
            {
                if q1 == q2 {
                    // Two GPI2 with same angle = GPI
                    if angles_equal(*phi1, *phi2) {
                        result.push(IonQGate::GPI(*q1, *phi1));
                        cancelled += 1; // Merged 2 into 1
                        i += 2;
                        continue;
                    }
                    // Two GPI2 with opposite angles = I (up to global phase)
                    if angles_differ_by_pi(*phi1, *phi2) {
                        cancelled += 2;
                        i += 2;
                        continue;
                    }
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, cancelled)
}

/// Cancel inverse MS gates: MS with opposite angles
fn cancel_inverse_ms(gates: &[IonQGate]) -> (Vec<IonQGate>, usize) {
    let mut result = Vec::with_capacity(gates.len());
    let mut cancelled = 0;
    let mut i = 0;

    while i < gates.len() {
        if i + 1 < gates.len() {
            // MS gate: MS(qubit1, qubit2, phi0, phi1)
            if let (
                IonQGate::MS(q1a, q1b, phi1_0, phi1_1),
                IonQGate::MS(q2a, q2b, phi2_0, phi2_1),
            ) = (&gates[i], &gates[i + 1])
            {
                // Same qubits (in either order)
                let same_qubits = (q1a == q2a && q1b == q2b) || (q1a == q2b && q1b == q2a);

                // Check if phases are inverse (differ by π)
                let inverse_phases =
                    angles_differ_by_pi(*phi1_0, *phi2_0) && angles_differ_by_pi(*phi1_1, *phi2_1);

                if same_qubits && inverse_phases {
                    // MS(φ)·MS(φ+π) = I
                    cancelled += 2;
                    i += 2;
                    continue;
                }
            }
        }
        result.push(gates[i].clone());
        i += 1;
    }

    (result, cancelled)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // IBM Tests

    #[test]
    fn test_ibm_cx_cancellation() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 2,
            num_clbits: 2,
            gates: vec![IBMGate::CX(0, 1), IBMGate::CX(0, 1)],
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 0);
        assert_eq!(stats.cx_cancelled, 2);
    }

    #[test]
    fn test_ibm_rz_merge() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            num_clbits: 1,
            gates: vec![IBMGate::Rz(0, PI / 4.0), IBMGate::Rz(0, PI / 4.0)],
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 1);
        assert_eq!(stats.rz_merged, 1);

        if let IBMGate::Rz(_, angle) = optimized.gates[0] {
            assert!((angle - PI / 2.0).abs() < ANGLE_EPSILON);
        }
    }

    #[test]
    fn test_ibm_zero_rz_elimination() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            num_clbits: 1,
            gates: vec![IBMGate::SX(0), IBMGate::Rz(0, 0.0), IBMGate::SX(0)],
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        // Zero Rz removed, SX·SX reduced to X
        assert!(optimized.gates.len() <= 2);
        assert_eq!(stats.rz_eliminated, 1);
    }

    #[test]
    fn test_ibm_2pi_rz_elimination() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            num_clbits: 1,
            gates: vec![IBMGate::Rz(0, TAU)], // 2π = identity
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 0);
        assert_eq!(stats.rz_eliminated, 1);
    }

    #[test]
    fn test_ibm_sx_reduction() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            num_clbits: 1,
            gates: vec![IBMGate::SX(0), IBMGate::SX(0)],
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 1);
        assert!(matches!(optimized.gates[0], IBMGate::X(0)));
        assert_eq!(stats.sx_reduced, 1);
    }

    #[test]
    fn test_ibm_complex_optimization() {
        let circuit = IBMCircuit {
            name: "test".to_string(),
            num_qubits: 2,
            num_clbits: 2,
            gates: vec![
                IBMGate::Rz(0, PI / 4.0),
                IBMGate::SX(0),
                IBMGate::Rz(0, PI / 4.0),
                IBMGate::CX(0, 1),
                IBMGate::CX(0, 1),   // Cancels
                IBMGate::Rz(1, 0.0), // Eliminated
            ],
        };

        let (optimized, stats) = optimize_ibm_circuit(&circuit);

        assert!(stats.cx_cancelled >= 2);
        assert!(stats.rz_eliminated >= 1);
        assert!(optimized.gates.len() < circuit.gates.len());
    }

    // IonQ Tests

    #[test]
    fn test_ionq_gpi2_same_angle() {
        let circuit = IonQCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            gates: vec![IonQGate::GPI2(0, PI / 2.0), IonQGate::GPI2(0, PI / 2.0)],
        };

        let (optimized, stats) = optimize_ionq_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 1);
        assert!(matches!(optimized.gates[0], IonQGate::GPI(..)));
        assert_eq!(stats.gpi2_merged, 1);
    }

    #[test]
    fn test_ionq_gpi2_opposite_angles() {
        let circuit = IonQCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            gates: vec![IonQGate::GPI2(0, 0.0), IonQGate::GPI2(0, PI)],
        };

        let (optimized, stats) = optimize_ionq_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 0);
        assert_eq!(stats.gpi2_merged, 2);
    }

    #[test]
    fn test_ionq_gpi_inverse() {
        let circuit = IonQCircuit {
            name: "test".to_string(),
            num_qubits: 1,
            gates: vec![IonQGate::GPI(0, 0.0), IonQGate::GPI(0, PI)],
        };

        let (optimized, stats) = optimize_ionq_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 0);
        assert_eq!(stats.gpi_merged, 2);
    }

    #[test]
    fn test_ionq_ms_cancellation() {
        let circuit = IonQCircuit {
            name: "test".to_string(),
            num_qubits: 2,
            gates: vec![
                IonQGate::MS(0, 1, 0.0, 0.0),
                IonQGate::MS(0, 1, PI, PI), // Inverse
            ],
        };

        let (optimized, stats) = optimize_ionq_circuit(&circuit);

        assert_eq!(optimized.gates.len(), 0);
        assert_eq!(stats.ms_cancelled, 2);
    }

    #[test]
    fn test_stats_reduction_percent() {
        let stats = IBMOptimizationStats {
            original_gates: 100,
            final_gates: 75,
            ..Default::default()
        };

        assert!((stats.reduction_percent() - 25.0).abs() < 1e-6);
    }
}
