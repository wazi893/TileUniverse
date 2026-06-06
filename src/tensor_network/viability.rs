//! Tensor network viability analysis for automatic backend routing.
//!
//! Sprint 77.0: Determines if tensor network simulation will outperform
//! state vector simulation for a given circuit.

use super::{GreedyMetric, circuit_to_network, compute_peak_memory, find_greedy_order};
use logic_fabric_core::quantum::QGate;

/// Result of tensor network viability analysis.
#[derive(Debug, Clone)]
pub struct TensorNetworkViability {
    /// Whether tensor network simulation is viable.
    pub is_viable: bool,
    /// Reason for the viability decision.
    pub reason: ViabilityReason,
    /// Estimated peak memory in bytes.
    pub estimated_peak_memory: usize,
    /// Estimated total FLOPs for contraction.
    pub estimated_flops: u64,
}

/// Reason for viability decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViabilityReason {
    /// Tensor network is viable.
    Viable,
    /// Too many qubits for efficient tensor network.
    TooManyQubits,
    /// Peak memory exceeds budget.
    PeakMemoryExceeds,
    /// Circuit contains unsupported gates.
    UnsupportedGates,
    /// Circuit is too deep for efficient contraction.
    TooDeep,
    /// Empty circuit.
    EmptyCircuit,
}

impl TensorNetworkViability {
    /// Create a viable result with estimated resources.
    pub fn viable(peak_memory: usize, flops: u64) -> Self {
        Self {
            is_viable: true,
            reason: ViabilityReason::Viable,
            estimated_peak_memory: peak_memory,
            estimated_flops: flops,
        }
    }

    /// Create a not-viable result with the given reason.
    pub fn not_viable(reason: ViabilityReason) -> Self {
        Self {
            is_viable: false,
            reason,
            estimated_peak_memory: 0,
            estimated_flops: 0,
        }
    }
}

/// Check if tensor network simulation is viable for a circuit.
///
/// # Arguments
/// * `circuit` - The quantum circuit to analyze
/// * `n_qubits` - Number of qubits
/// * `memory_budget_bytes` - Maximum memory budget (default: 2GB)
///
/// # Returns
/// Viability analysis result with estimated resources.
pub fn is_tensor_network_viable(
    circuit: &[QGate],
    n_qubits: u8,
    memory_budget_bytes: usize,
) -> TensorNetworkViability {
    // Fast reject: empty circuit
    if circuit.is_empty() {
        return TensorNetworkViability::not_viable(ViabilityReason::EmptyCircuit);
    }

    // Fast reject: too many qubits (> 30 qubits rarely beneficial)
    if n_qubits > 30 {
        return TensorNetworkViability::not_viable(ViabilityReason::TooManyQubits);
    }

    // Fast reject: very deep circuits (depth > 200 gates per qubit)
    if circuit.len() > 200 * n_qubits as usize {
        return TensorNetworkViability::not_viable(ViabilityReason::TooDeep);
    }

    // Check for unsupported gates (measurements, 3-qubit gates)
    if has_unsupported_gates(circuit) {
        return TensorNetworkViability::not_viable(ViabilityReason::UnsupportedGates);
    }

    // Build network and estimate peak memory
    let network = circuit_to_network(circuit, n_qubits);
    let path = find_greedy_order(&network, GreedyMetric::MinMemory);
    let peak_elements = compute_peak_memory(&network, &path);
    let peak_bytes = peak_elements * 8; // Complex32 = 8 bytes

    if peak_bytes > memory_budget_bytes {
        return TensorNetworkViability::not_viable(ViabilityReason::PeakMemoryExceeds);
    }

    TensorNetworkViability::viable(peak_bytes, path.total_cost)
}

/// Check for gates not supported by tensor network conversion.
///
/// The tensor network converter in `circuit_convert.rs` handles:
/// - Single-qubit: H, X, Y, Z, T, Tdg, Rx, Ry, Rz
/// - Two-qubit: CNot, CZ
///
/// Unsupported gates are:
/// - Measure (non-unitary)
/// - Three-qubit gates: Toffoli, CCZ
/// - Other gates that fall through the match: Phase, Swap, CPhase, CRz, U3
fn has_unsupported_gates(circuit: &[QGate]) -> bool {
    circuit.iter().any(|gate| {
        matches!(
            gate,
            QGate::Toffoli(_, _, _)
                | QGate::CCZ(_, _, _)
                | QGate::Measure(_)
                | QGate::Phase(_, _)
                | QGate::Swap(_, _)
                | QGate::CPhase(_, _, _)
                | QGate::CRz(_, _, _)
                | QGate::U3(_, _, _, _)
        )
    })
}

/// Default memory budget: 2GB
pub const DEFAULT_MEMORY_BUDGET: usize = 2 * 1024 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_circuit_not_viable() {
        let result = is_tensor_network_viable(&[], 4, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::EmptyCircuit);
    }

    #[test]
    fn test_too_many_qubits_not_viable() {
        let circuit = vec![QGate::H(0)];
        let result = is_tensor_network_viable(&circuit, 35, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::TooManyQubits);
    }

    #[test]
    fn test_small_circuit_viable() {
        // Bell state: H(0), CNOT(0,1)
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let result = is_tensor_network_viable(&circuit, 2, DEFAULT_MEMORY_BUDGET);
        assert!(result.is_viable);
        assert!(result.estimated_peak_memory < 1000); // Very small
    }

    #[test]
    fn test_medium_circuit_viable() {
        // 10-qubit GHZ
        let mut circuit = vec![QGate::H(0)];
        for i in 0..9 {
            circuit.push(QGate::CNot(i, i + 1));
        }
        let result = is_tensor_network_viable(&circuit, 10, DEFAULT_MEMORY_BUDGET);
        assert!(result.is_viable);
    }

    #[test]
    fn test_unsupported_gate_toffoli() {
        let circuit = vec![QGate::H(0), QGate::Toffoli(0, 1, 2)];
        let result = is_tensor_network_viable(&circuit, 3, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::UnsupportedGates);
    }

    #[test]
    fn test_unsupported_gate_ccz() {
        let circuit = vec![QGate::H(0), QGate::CCZ(0, 1, 2)];
        let result = is_tensor_network_viable(&circuit, 3, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::UnsupportedGates);
    }

    #[test]
    fn test_unsupported_gate_measure() {
        let circuit = vec![QGate::H(0), QGate::Measure(0)];
        let result = is_tensor_network_viable(&circuit, 1, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::UnsupportedGates);
    }

    #[test]
    fn test_too_deep_circuit() {
        // Create a circuit with depth > 200 per qubit
        let n_qubits = 2u8;
        let mut circuit = Vec::new();
        for _ in 0..(201 * n_qubits as usize + 1) {
            circuit.push(QGate::H(0));
        }
        let result = is_tensor_network_viable(&circuit, n_qubits, DEFAULT_MEMORY_BUDGET);
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::TooDeep);
    }

    #[test]
    fn test_memory_budget_exceeded() {
        // Use a very small memory budget
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let result = is_tensor_network_viable(&circuit, 2, 1); // 1 byte budget
        assert!(!result.is_viable);
        assert_eq!(result.reason, ViabilityReason::PeakMemoryExceeds);
    }

    #[test]
    fn test_supported_rotation_gates() {
        // All rotation gates should be supported
        let circuit = vec![
            QGate::Rx(0, 0.5),
            QGate::Ry(1, 0.5),
            QGate::Rz(2, 0.5),
            QGate::T(0),
            QGate::Tdg(1),
        ];
        let result = is_tensor_network_viable(&circuit, 3, DEFAULT_MEMORY_BUDGET);
        assert!(result.is_viable);
    }

    #[test]
    fn test_supported_two_qubit_gates() {
        // CNot and CZ should be supported
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CZ(1, 2)];
        let result = is_tensor_network_viable(&circuit, 3, DEFAULT_MEMORY_BUDGET);
        assert!(result.is_viable);
    }
}
