#!/usr/bin/env python3
"""
SPRINT 37.0: Qiskit-based 2-qubit gate synthesis
Reads a 4x4 unitary matrix from stdin and outputs a decomposed circuit
"""
import sys
import json
import numpy as np

def parse_unitary(data):
    """Parse 4x4 complex matrix from JSON"""
    matrix = []
    for row in data['matrix']:
        matrix.append([complex(c['re'], c['im']) for c in row])
    return np.array(matrix, dtype=complex)

def decompose_two_qubit(unitary, qa, qb):
    """Decompose 2-qubit unitary using Qiskit"""
    try:
        from qiskit.quantum_info import Operator
        from qiskit.synthesis import TwoQubitBasisDecomposer
        from qiskit.circuit.library import CXGate

        # Create decomposer for CNOT basis
        decomposer = TwoQubitBasisDecomposer(CXGate())

        # Decompose the unitary
        circuit = decomposer(unitary)

        # Extract gates
        gates = []
        # Map circuit qubits to our qubit indices
        qubit_map = {circuit.qubits[0]: qa, circuit.qubits[1]: qb}

        for instruction in circuit.data:
            gate_name = instruction.operation.name
            qubits = [qubit_map[q] for q in instruction.qubits]

            if gate_name == 'cx':
                gates.append({'type': 'CNot', 'control': qubits[0], 'target': qubits[1]})
            elif gate_name == 'rz':
                angle = float(instruction.operation.params[0])
                gates.append({'type': 'Rz', 'qubit': qubits[0], 'angle': angle})
            elif gate_name == 'ry':
                angle = float(instruction.operation.params[0])
                gates.append({'type': 'Ry', 'qubit': qubits[0], 'angle': angle})
            elif gate_name == 'rx':
                angle = float(instruction.operation.params[0])
                gates.append({'type': 'Rx', 'qubit': qubits[0], 'angle': angle})
            elif gate_name in ['u3', 'u']:
                theta, phi, lam = [float(p) for p in instruction.operation.params]
                # Convert U/U3 to RZ-RY-RZ sequence
                # U(θ,φ,λ) = RZ(φ) · RY(θ) · RZ(λ)
                if abs(lam) > 1e-10:
                    gates.append({'type': 'Rz', 'qubit': qubits[0], 'angle': lam})
                if abs(theta) > 1e-10:
                    gates.append({'type': 'Ry', 'qubit': qubits[0], 'angle': theta})
                if abs(phi) > 1e-10:
                    gates.append({'type': 'Rz', 'qubit': qubits[0], 'angle': phi})
            else:
                # Try to extract rotation gates generically
                print(f"Warning: unknown gate {gate_name}", file=sys.stderr)

        return gates
    except ImportError:
        print("Error: qiskit not installed", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error in decomposition: {e}", file=sys.stderr)
        return None

def main():
    # Read input JSON
    try:
        data = json.load(sys.stdin)
        unitary = parse_unitary(data)
        qa = data['qa']
        qb = data['qb']

        # Decompose
        gates = decompose_two_qubit(unitary, qa, qb)

        if gates is None:
            sys.exit(1)

        # Output result
        result = {'gates': gates}
        print(json.dumps(result))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
