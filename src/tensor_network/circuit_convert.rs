//! Circuit to tensor network conversion.

use super::network::TensorNetwork;
use super::tensor::Tensor;
use logic_fabric_core::quantum::QGate;

/// Convert a quantum circuit to a tensor network.
///
/// Each qubit gets an input and output index. Gates create tensors
/// that connect indices. The time ordering creates a "chain" structure
/// where each gate's output becomes the next gate's input.
pub fn circuit_to_network(gates: &[QGate], n_qubits: u8) -> TensorNetwork {
    let mut network = TensorNetwork::new(n_qubits);

    // Current "wire" index for each qubit (starts with input indices)
    let mut qubit_wires: Vec<_> = (0..n_qubits).map(|_| network.new_qubit_index()).collect();

    // Record input indices as open (first n_qubits entries)
    let input_indices = qubit_wires.clone();

    for gate in gates {
        match gate {
            // Single-qubit gates
            QGate::H(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::hadamard(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::X(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::pauli_x(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Y(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::pauli_y(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Z(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::pauli_z(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::T(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::t_gate(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Tdg(q) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::tdg_gate(in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Rx(q, theta) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::rx_gate(*theta, in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Ry(q, theta) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::ry_gate(*theta, in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            QGate::Rz(q, theta) => {
                let q = *q as usize;
                let in_idx = qubit_wires[q];
                let out_idx = network.new_qubit_index();
                let tensor = Tensor::rz_gate(*theta, in_idx, out_idx);
                network.add_tensor(tensor);
                qubit_wires[q] = out_idx;
            }

            // Two-qubit gates
            QGate::CNot(ctrl, tgt) => {
                let ctrl = *ctrl as usize;
                let tgt = *tgt as usize;

                let in_ctrl = qubit_wires[ctrl];
                let in_tgt = qubit_wires[tgt];
                let out_ctrl = network.new_qubit_index();
                let out_tgt = network.new_qubit_index();

                let tensor = Tensor::cnot(in_ctrl, in_tgt, out_ctrl, out_tgt);
                network.add_tensor(tensor);

                qubit_wires[ctrl] = out_ctrl;
                qubit_wires[tgt] = out_tgt;
            }

            QGate::CZ(ctrl, tgt) => {
                let ctrl = *ctrl as usize;
                let tgt = *tgt as usize;

                let in_ctrl = qubit_wires[ctrl];
                let in_tgt = qubit_wires[tgt];
                let out_ctrl = network.new_qubit_index();
                let out_tgt = network.new_qubit_index();

                let tensor = Tensor::cz(in_ctrl, in_tgt, out_ctrl, out_tgt);
                network.add_tensor(tensor);

                qubit_wires[ctrl] = out_ctrl;
                qubit_wires[tgt] = out_tgt;
            }

            // Skip unsupported gates (measurements, 3-qubit gates)
            QGate::Measure(_) => {
                // Measurements are handled separately
            }

            _ => {
                // Other gates not yet supported in TN
                // Could add: Phase, Swap, CPhase, CRz, U3, Toffoli, CCZ
            }
        }
    }

    // Record all open indices: inputs first, then outputs
    network.open_indices = input_indices;
    network.open_indices.extend(qubit_wires.iter().copied());

    network
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_hadamard() {
        let circuit = vec![QGate::H(0)];
        let network = circuit_to_network(&circuit, 1);

        assert_eq!(network.tensor_count(), 1);
        assert_eq!(network.n_qubits, 1);
        // 2 open indices: input and output for qubit 0
        assert_eq!(network.open_indices.len(), 2);
    }

    #[test]
    fn test_bell_circuit() {
        // H(0) -> CNOT(0,1) creates Bell state
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);

        // 1 Hadamard + 1 CNOT = 2 tensors
        assert_eq!(network.tensor_count(), 2);
        // 4 open indices: 2 inputs + 2 outputs
        assert_eq!(network.open_indices.len(), 4);
    }

    #[test]
    fn test_ghz_circuit() {
        // H(0) -> CNOT(0,1) -> CNOT(1,2) creates 3-qubit GHZ
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CNot(1, 2)];
        let network = circuit_to_network(&circuit, 3);

        assert_eq!(network.tensor_count(), 3);
        // 6 open indices: 3 inputs + 3 outputs
        assert_eq!(network.open_indices.len(), 6);
    }
}
