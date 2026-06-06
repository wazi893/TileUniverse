//! Amplitude computation via tensor network contraction.

use super::circuit_convert::circuit_to_network;
use super::greedy::{GreedyMetric, execute_path, find_greedy_order};
use super::network::TensorNetwork;
use super::tensor::Tensor;
use logic_fabric_core::quantum::{Complex32, QGate};

/// Compute a specific amplitude <out_bitstring|Circuit|in_bitstring>.
///
/// This is the main use case for tensor networks - computing specific
/// amplitudes without constructing the full state vector.
pub fn compute_amplitude(network: &TensorNetwork, in_bits: u64, out_bits: u64) -> Complex32 {
    let n_qubits = network.n_qubits;

    // Clone network and add "capping" tensors for the bitstrings
    let mut capped = network.clone();

    // The open_indices are [input_0, input_1, ..., output_0, output_1, ...]
    // First n_qubits are inputs, next n_qubits are outputs

    // Add input basis state tensors |in_bits>
    for q in 0..n_qubits {
        let bit = ((in_bits >> q) & 1) as usize;
        let idx = network.open_indices[q as usize];
        let cap = Tensor::basis_projector(idx, bit);
        capped.add_tensor(cap);
    }

    // Add output basis state tensors <out_bits|
    for q in 0..n_qubits {
        let bit = ((out_bits >> q) & 1) as usize;
        let idx = network.open_indices[(n_qubits + q) as usize];
        let cap = Tensor::basis_projector(idx, bit);
        capped.add_tensor(cap);
    }

    // Clear open indices since they're now all contracted
    capped.open_indices.clear();

    // Find contraction order and contract
    let path = find_greedy_order(&capped, GreedyMetric::MinMemory);
    let result = execute_path(&capped, &path);

    // Result should be a scalar (rank-0 tensor)
    if result.is_scalar() {
        result.data[0]
    } else {
        // If not scalar, something went wrong - return 0
        Complex32::new(0.0, 0.0)
    }
}

/// Compute amplitude directly from a circuit.
pub fn circuit_amplitude(gates: &[QGate], n_qubits: u8, in_bits: u64, out_bits: u64) -> Complex32 {
    let network = circuit_to_network(gates, n_qubits);
    compute_amplitude(&network, in_bits, out_bits)
}

/// Compute multiple amplitudes efficiently by reusing the network.
pub fn compute_amplitudes(network: &TensorNetwork, bitstrings: &[(u64, u64)]) -> Vec<Complex32> {
    bitstrings
        .iter()
        .map(|(in_bits, out_bits)| compute_amplitude(network, *in_bits, *out_bits))
        .collect()
}

/// Compute all 2^n output amplitudes for a given input (full output distribution).
/// Only practical for small n_qubits (< 20).
pub fn compute_all_output_amplitudes(network: &TensorNetwork, in_bits: u64) -> Vec<Complex32> {
    let n_qubits = network.n_qubits;
    let n_outputs = 1u64 << n_qubits;

    (0..n_outputs)
        .map(|out_bits| compute_amplitude(network, in_bits, out_bits))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_identity_amplitude() {
        // Empty circuit: <0|I|0> = 1, <1|I|0> = 0
        let circuit: Vec<QGate> = vec![];
        let network = circuit_to_network(&circuit, 1);

        let amp_00 = compute_amplitude(&network, 0, 0);
        let amp_01 = compute_amplitude(&network, 0, 1);

        assert!((amp_00.re - 1.0).abs() < 1e-6);
        assert!(amp_00.im.abs() < 1e-6);
        assert!(amp_01.re.abs() < 1e-6);
        assert!(amp_01.im.abs() < 1e-6);
    }

    #[test]
    fn test_hadamard_amplitude() {
        // H|0> = (|0> + |1>)/sqrt(2)
        let circuit = vec![QGate::H(0)];
        let network = circuit_to_network(&circuit, 1);

        let amp_0 = compute_amplitude(&network, 0, 0);
        let amp_1 = compute_amplitude(&network, 0, 1);

        assert!((amp_0.re - FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((amp_1.re - FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_bell_state_amplitude() {
        // Bell state: (|00> + |11>)/sqrt(2)
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);

        let amp_00 = compute_amplitude(&network, 0b00, 0b00);
        let amp_01 = compute_amplitude(&network, 0b00, 0b01);
        let amp_10 = compute_amplitude(&network, 0b00, 0b10);
        let amp_11 = compute_amplitude(&network, 0b00, 0b11);

        // |00> and |11> should have amplitude 1/sqrt(2)
        assert!(
            (amp_00.re - FRAC_1_SQRT_2).abs() < 1e-5,
            "amp_00 = {}",
            amp_00.re
        );
        assert!(
            (amp_11.re - FRAC_1_SQRT_2).abs() < 1e-5,
            "amp_11 = {}",
            amp_11.re
        );

        // |01> and |10> should have amplitude 0
        assert!(amp_01.re.abs() < 1e-5);
        assert!(amp_10.re.abs() < 1e-5);
    }

    #[test]
    fn test_x_gate_amplitude() {
        // X|0> = |1>
        let circuit = vec![QGate::X(0)];
        let network = circuit_to_network(&circuit, 1);

        let amp_0 = compute_amplitude(&network, 0, 0);
        let amp_1 = compute_amplitude(&network, 0, 1);

        assert!(amp_0.re.abs() < 1e-6);
        assert!((amp_1.re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_circuit_amplitude_convenience() {
        let circuit = vec![QGate::H(0)];
        let amp = circuit_amplitude(&circuit, 1, 0, 0);
        assert!((amp.re - FRAC_1_SQRT_2).abs() < 1e-6);
    }
}
