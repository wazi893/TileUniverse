#!/usr/bin/env python3
"""
Fidelity Comparison: TileUniverse vs Qiskit Aer
================================================
Direct statevector comparison to prove FP16 accuracy.

Run this in WSL with: python3 fidelity_comparison.py
Then run the Rust comparison with the generated reference data.
"""

import json
import numpy as np
from pathlib import Path

def generate_aer_reference():
    """Generate reference statevectors using Qiskit Aer (FP64)"""
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit

    print("="*70)
    print("  Generating Aer Reference Statevectors (FP64)")
    print("="*70)

    # Use FP64 for maximum precision reference
    sim = AerSimulator(method='statevector', device='CPU', precision='double')

    test_cases = []

    # Test Case 1: Simple Hadamard on all qubits
    for n_qubits in [4, 8, 12]:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
        qc.save_statevector()

        result = sim.run(qc, shots=1).result()
        sv = result.get_statevector()

        # Convert to list of (real, imag) tuples
        amplitudes = [(float(c.real), float(c.imag)) for c in sv.data]

        test_cases.append({
            'name': f'hadamard_all_{n_qubits}q',
            'n_qubits': n_qubits,
            'gates': [{'type': 'H', 'target': q} for q in range(n_qubits)],
            'amplitudes': amplitudes,
        })
        print(f"  Generated: hadamard_all_{n_qubits}q")

    # Test Case 2: GHZ state preparation
    for n_qubits in [4, 8, 12]:
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.save_statevector()

        result = sim.run(qc, shots=1).result()
        sv = result.get_statevector()
        amplitudes = [(float(c.real), float(c.imag)) for c in sv.data]

        gates = [{'type': 'H', 'target': 0}]
        for q in range(n_qubits - 1):
            gates.append({'type': 'CNOT', 'control': q, 'target': q + 1})

        test_cases.append({
            'name': f'ghz_{n_qubits}q',
            'n_qubits': n_qubits,
            'gates': gates,
            'amplitudes': amplitudes,
        })
        print(f"  Generated: ghz_{n_qubits}q")

    # Test Case 3: QFT-like circuit (deep, many rotations)
    for n_qubits in [4, 6, 8]:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
            for k in range(q + 1, n_qubits):
                angle = np.pi / (2 ** (k - q))
                qc.cp(angle, q, k)  # Controlled phase
        qc.save_statevector()

        result = sim.run(qc, shots=1).result()
        sv = result.get_statevector()
        amplitudes = [(float(c.real), float(c.imag)) for c in sv.data]

        # Simplified gate representation
        test_cases.append({
            'name': f'qft_like_{n_qubits}q',
            'n_qubits': n_qubits,
            'gates': 'QFT-like',  # Complex to serialize, use name
            'amplitudes': amplitudes,
        })
        print(f"  Generated: qft_like_{n_qubits}q")

    # Test Case 4: Random circuit (stress test)
    np.random.seed(42)
    for n_qubits in [4, 6, 8]:
        depth = 20
        qc = QuantumCircuit(n_qubits)

        for d in range(depth):
            # Layer of single-qubit gates
            for q in range(n_qubits):
                gate = np.random.choice(['h', 'x', 'z', 's', 't'])
                if gate == 'h':
                    qc.h(q)
                elif gate == 'x':
                    qc.x(q)
                elif gate == 'z':
                    qc.z(q)
                elif gate == 's':
                    qc.s(q)
                elif gate == 't':
                    qc.t(q)

            # Layer of CNOTs
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)

        qc.save_statevector()
        result = sim.run(qc, shots=1).result()
        sv = result.get_statevector()
        amplitudes = [(float(c.real), float(c.imag)) for c in sv.data]

        test_cases.append({
            'name': f'random_depth20_{n_qubits}q',
            'n_qubits': n_qubits,
            'gates': f'random_seed42_depth{depth}',
            'amplitudes': amplitudes,
        })
        print(f"  Generated: random_depth20_{n_qubits}q")

    # Test Case 5: Grover iteration (algorithm-relevant)
    n_qubits = 4
    marked = 0b1010  # Target state |1010>

    qc = QuantumCircuit(n_qubits)
    # Initialize superposition
    for q in range(n_qubits):
        qc.h(q)

    # One Grover iteration
    # Oracle: flip phase of marked state
    for q in range(n_qubits):
        if not (marked >> q) & 1:
            qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in range(n_qubits):
        if not (marked >> q) & 1:
            qc.x(q)

    # Diffusion
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)

    qc.save_statevector()
    result = sim.run(qc, shots=1).result()
    sv = result.get_statevector()
    amplitudes = [(float(c.real), float(c.imag)) for c in sv.data]

    test_cases.append({
        'name': 'grover_1iter_4q',
        'n_qubits': n_qubits,
        'gates': 'grover_oracle_diffusion',
        'amplitudes': amplitudes,
        'marked_state': marked,
    })
    print(f"  Generated: grover_1iter_4q (marked={marked})")

    return test_cases


def compute_fidelity(sv1, sv2):
    """Compute state fidelity |<sv1|sv2>|^2"""
    # Convert to numpy complex arrays
    a1 = np.array([complex(r, i) for r, i in sv1])
    a2 = np.array([complex(r, i) for r, i in sv2])

    # Normalize (in case of numerical errors)
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)

    # Fidelity = |<a1|a2>|^2
    overlap = np.abs(np.vdot(a1, a2)) ** 2
    return float(overlap)


def compute_max_amplitude_error(sv1, sv2):
    """Compute maximum amplitude error"""
    a1 = np.array([complex(r, i) for r, i in sv1])
    a2 = np.array([complex(r, i) for r, i in sv2])
    return float(np.max(np.abs(a1 - a2)))


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Compare mode: load TileUniverse results and compare
        print("="*70)
        print("  Comparing TileUniverse vs Aer Reference")
        print("="*70)

        with open('aer_reference.json', 'r') as f:
            aer_data = json.load(f)

        with open('tileuniverse_results.json', 'r') as f:
            tu_data = json.load(f)

        print(f"\n{'Test Case':<30} {'Fidelity':>12} {'Max Error':>12} {'Status':>10}")
        print("-" * 70)

        all_pass = True
        for aer_case in aer_data:
            name = aer_case['name']
            tu_case = next((t for t in tu_data if t['name'] == name), None)

            if tu_case is None:
                print(f"{name:<30} {'N/A':>12} {'N/A':>12} {'MISSING':>10}")
                continue

            fidelity = compute_fidelity(aer_case['amplitudes'], tu_case['amplitudes'])
            max_err = compute_max_amplitude_error(aer_case['amplitudes'], tu_case['amplitudes'])

            # Fidelity > 0.9999 is excellent, > 0.999 is good, > 0.99 is acceptable
            if fidelity > 0.9999:
                status = "EXCELLENT"
            elif fidelity > 0.999:
                status = "GOOD"
            elif fidelity > 0.99:
                status = "OK"
            else:
                status = "FAIL"
                all_pass = False

            print(f"{name:<30} {fidelity:>12.8f} {max_err:>12.2e} {status:>10}")

        print()
        if all_pass:
            print("ALL TESTS PASSED - TileUniverse matches Aer within tolerance")
        else:
            print("SOME TESTS FAILED - Check precision settings")

    else:
        # Generate mode: create Aer reference data
        test_cases = generate_aer_reference()

        output_file = Path('aer_reference.json')
        with open(output_file, 'w') as f:
            json.dump(test_cases, f, indent=2)

        print(f"\nSaved {len(test_cases)} test cases to {output_file}")
        print("\nNext steps:")
        print("  1. Copy aer_reference.json to Windows")
        print("  2. Run TileUniverse on the same circuits")
        print("  3. Run: python3 fidelity_comparison.py --compare")


if __name__ == '__main__':
    main()
