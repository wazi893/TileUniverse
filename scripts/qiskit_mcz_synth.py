#!/usr/bin/env python3
"""
SPRINT 37.0: Multi-controlled-Z gate synthesis using Qiskit

Decomposes n-controlled-Z gates into CNot, Rz, Ry, Rx gates using Qiskit's
built-in decomposition algorithms.

Input (JSON via stdin):
{
    "n_qubits": 4  // Number of qubits for multi-controlled-Z
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def decompose_mcz(n_qubits):
    """
    Decompose n-qubit controlled-Z gate into basic gates.

    For n_qubits:
    - 1: Z gate
    - 2: CZ gate
    - 3+: Multi-controlled Z using Qiskit decomposition
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")

    # Create circuit with MCZ
    qc = QuantumCircuit(n_qubits)

    if n_qubits == 1:
        # Single Z gate
        qc.z(0)
    elif n_qubits == 2:
        # CZ gate
        qc.cz(0, 1)
    else:
        # Multi-controlled Z: control on all qubits except last, target on last
        # Then use phase trick to make it symmetric
        # MCZ on n qubits = H(target) + MC-X(controls, target) + H(target)
        # But we want symmetric MCZ on all n qubits

        # Use the diagonal gate approach for multi-controlled-Z
        # MCZ flips phase only when all qubits are |1⟩

        # Qiskit's multi-controlled Z implementation:
        # control_qubits = all except target, target = last qubit
        controls = list(range(n_qubits - 1))
        target = n_qubits - 1

        # Use mcx (multi-controlled X) with Hadamards to create MCZ
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

    # Decompose to basis gates using transpile
    from qiskit import transpile
    basis_gates = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z']
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
        else:
            # Skip unknown gates or barriers
            pass

    return gates

def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        n_qubits = input_data['n_qubits']

        # Decompose MCZ
        gates = decompose_mcz(n_qubits)

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
