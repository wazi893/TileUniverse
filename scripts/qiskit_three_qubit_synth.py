#!/usr/bin/env python3
"""
SPRINT 37.0: Three-qubit gate synthesis using Qiskit

Decomposes 3-qubit gates (Toffoli, Fredkin, general unitaries) into basic gates
using Qiskit's decomposition algorithms.

Input (JSON via stdin):
{
    "gate_type": "Toffoli",  // or "Fredkin", "CCX", "CSWAP"
    "qubits": [0, 1, 2]      // Control qubits and target
}

OR for general unitary:
{
    "gate_type": "Unitary",
    "qubits": [0, 1, 2],
    "matrix": [[...], [...], ...]  // 8x8 unitary matrix
}

Output (JSON via stdout):
{
    "gates": [
        {"type": "Rz", "qubit": 0, "angle": 1.5707963267948966},
        {"type": "CNot", "control": 0, "target": 1},
        ...
    ]
}
"""

import sys
import json
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

def synthesize_toffoli(qubits):
    """
    Synthesize Toffoli (CCNOT) gate.

    Args:
        qubits: [control1, control2, target]

    Returns:
        List of decomposed gates
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)  # Toffoli on qubits 0, 1, 2
    return qc

def synthesize_fredkin(qubits):
    """
    Synthesize Fredkin (CSWAP) gate.

    Args:
        qubits: [control, target1, target2]

    Returns:
        List of decomposed gates
    """
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)  # Fredkin on qubits 0, 1, 2
    return qc

def synthesize_general_unitary(matrix, qubits):
    """
    Synthesize arbitrary 3-qubit unitary.

    Args:
        matrix: 8x8 unitary matrix (list of lists)
        qubits: [q0, q1, q2]

    Returns:
        List of decomposed gates
    """
    # Convert to numpy array
    U = np.array(matrix, dtype=complex)

    # Verify it's 8x8
    if U.shape != (8, 8):
        raise ValueError(f"Expected 8x8 matrix, got {U.shape}")

    # Create circuit with unitary
    qc = QuantumCircuit(3)
    qc.unitary(U, [0, 1, 2])

    return qc

def decompose_to_basis_gates(qc):
    """
    Decompose quantum circuit to basis gates.

    Args:
        qc: QuantumCircuit to decompose

    Returns:
        List of gate dictionaries
    """
    # Transpile to basis gates
    basis_gates = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 'sx', 's', 't', 'tdg', 'sdg']
    qc_basis = transpile(qc, basis_gates=basis_gates, optimization_level=0)

    # Convert to our gate format
    gates = []

    for instruction in qc_basis.data:
        gate_name = instruction.operation.name
        qubits = [qc_basis.find_bit(q).index for q in instruction.qubits]
        params = instruction.operation.params

        if gate_name == 'cx':
            gates.append({
                'type': 'CNot',
                'control': qubits[0],
                'target': qubits[1]
            })
        elif gate_name == 'rz':
            gates.append({
                'type': 'Rz',
                'qubit': qubits[0],
                'angle': float(params[0])
            })
        elif gate_name == 'ry':
            gates.append({
                'type': 'Ry',
                'qubit': qubits[0],
                'angle': float(params[0])
            })
        elif gate_name == 'rx':
            gates.append({
                'type': 'Rx',
                'qubit': qubits[0],
                'angle': float(params[0])
            })
        elif gate_name == 'h':
            gates.append({
                'type': 'H',
                'qubit': qubits[0]
            })
        elif gate_name == 'x':
            gates.append({
                'type': 'X',
                'qubit': qubits[0]
            })
        elif gate_name == 'y':
            gates.append({
                'type': 'Y',
                'qubit': qubits[0]
            })
        elif gate_name == 'z':
            gates.append({
                'type': 'Z',
                'qubit': qubits[0]
            })
        elif gate_name in ['sx', 's', 't', 'tdg', 'sdg']:
            # Convert special gates to rotations
            # SX = sqrt(X) = Rx(π/2)
            # S = Rz(π/2)
            # T = Rz(π/4)
            # Tdg = Rz(-π/4)
            # Sdg = Rz(-π/2)
            if gate_name == 'sx':
                gates.append({
                    'type': 'Rx',
                    'qubit': qubits[0],
                    'angle': np.pi / 2
                })
            elif gate_name == 's':
                gates.append({
                    'type': 'Rz',
                    'qubit': qubits[0],
                    'angle': np.pi / 2
                })
            elif gate_name == 't':
                gates.append({
                    'type': 'Rz',
                    'qubit': qubits[0],
                    'angle': np.pi / 4
                })
            elif gate_name == 'tdg':
                gates.append({
                    'type': 'Rz',
                    'qubit': qubits[0],
                    'angle': -np.pi / 4
                })
            elif gate_name == 'sdg':
                gates.append({
                    'type': 'Rz',
                    'qubit': qubits[0],
                    'angle': -np.pi / 2
                })
        # Skip barriers and other non-gates

    return gates

def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        gate_type = input_data['gate_type'].upper()
        qubits = input_data.get('qubits', [0, 1, 2])

        # Synthesize based on gate type
        if gate_type in ['TOFFOLI', 'CCX', 'CCNOT']:
            qc = synthesize_toffoli(qubits)
        elif gate_type in ['FREDKIN', 'CSWAP']:
            qc = synthesize_fredkin(qubits)
        elif gate_type == 'UNITARY':
            matrix = input_data['matrix']
            qc = synthesize_general_unitary(matrix, qubits)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        # Decompose to basis gates
        gates = decompose_to_basis_gates(qc)

        # Output result
        result = {'gates': gates}
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        error = {'error': str(e)}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
