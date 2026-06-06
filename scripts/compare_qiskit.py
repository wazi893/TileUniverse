#!/usr/bin/env python3
"""
Compare TileUniverse optimizer vs Qiskit -O3 on QASMBench circuits.
"""

import os
import subprocess
import csv
from pathlib import Path

# Try importing qiskit
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available")

QASMBENCH_DIR = Path("./QASMBench")
OPTIMIZER = Path("./target/release/opt_qasm.exe")
RESULTS_FILE = "qiskit_comparison.csv"

def count_gates(qc):
    """Count non-barrier, non-measure gates in a circuit."""
    count = 0
    for inst in qc.data:
        name = inst.operation.name
        if name not in ('barrier', 'measure', 'reset'):
            count += 1
    return count

def run_tileuniverse(qasm_file):
    """Run TileUniverse optimizer and return (original, optimized) gate counts."""
    try:
        out_file = "/tmp/opt_out.qasm"
        result = subprocess.run(
            [str(OPTIMIZER), str(qasm_file), out_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Parse output for gate counts
        original = None
        optimized = None
        for line in result.stdout.split('\n'):
            if 'Original gates:' in line:
                original = int(line.split(':')[1].strip())
            elif 'Final gates:' in line:
                optimized = int(line.split(':')[1].strip())

        return original, optimized
    except Exception as e:
        return None, None

def run_qiskit_o3(qasm_file):
    """Run Qiskit transpile with optimization_level=3."""
    try:
        qc = QuantumCircuit.from_qasm_file(str(qasm_file))
        original = count_gates(qc)

        # Transpile with O3
        optimized_qc = transpile(qc, optimization_level=3, basis_gates=['cx', 'u3', 'u2', 'u1', 'id'])
        optimized = count_gates(optimized_qc)

        return original, optimized
    except Exception as e:
        return None, None

def main():
    if not QISKIT_AVAILABLE:
        print("Qiskit not installed. Run: pip install qiskit")
        return

    if not OPTIMIZER.exists():
        print(f"TileUniverse optimizer not found at {OPTIMIZER}")
        return

    results = []
    categories = ['small', 'medium']

    print("=" * 80)
    print("TileUniverse vs Qiskit -O3 Comparison")
    print("=" * 80)
    print()

    for category in categories:
        cat_dir = QASMBENCH_DIR / category
        if not cat_dir.exists():
            continue

        print(f"=== {category.upper()} BENCHMARKS ===")

        for bench_dir in sorted(cat_dir.iterdir()):
            if not bench_dir.is_dir():
                continue

            # Find .qasm file
            qasm_files = list(bench_dir.glob("*.qasm"))
            if not qasm_files:
                continue
            qasm_file = qasm_files[0]
            name = bench_dir.name

            # Run both optimizers
            tu_orig, tu_opt = run_tileuniverse(qasm_file)
            qk_orig, qk_opt = run_qiskit_o3(qasm_file)

            if tu_orig is None or qk_orig is None:
                print(f"  {name}: PARSE ERROR")
                continue

            # Calculate reductions
            tu_reduction = (1 - tu_opt / tu_orig) * 100 if tu_orig > 0 else 0
            qk_reduction = (1 - qk_opt / qk_orig) * 100 if qk_orig > 0 else 0

            # Determine winner
            if tu_opt < qk_opt:
                winner = "TU"
                diff = qk_opt - tu_opt
            elif qk_opt < tu_opt:
                winner = "QK"
                diff = tu_opt - qk_opt
            else:
                winner = "TIE"
                diff = 0

            results.append({
                'benchmark': name,
                'category': category,
                'original': tu_orig,
                'tu_optimized': tu_opt,
                'tu_reduction': tu_reduction,
                'qk_optimized': qk_opt,
                'qk_reduction': qk_reduction,
                'winner': winner,
                'diff': diff
            })

            print(f"  {name:25s} | TU: {tu_orig:6d} -> {tu_opt:6d} ({tu_reduction:5.1f}%) | QK: {qk_orig:6d} -> {qk_opt:6d} ({qk_reduction:5.1f}%) | {winner} by {diff}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    tu_wins = sum(1 for r in results if r['winner'] == 'TU')
    qk_wins = sum(1 for r in results if r['winner'] == 'QK')
    ties = sum(1 for r in results if r['winner'] == 'TIE')

    tu_total_orig = sum(r['original'] for r in results)
    tu_total_opt = sum(r['tu_optimized'] for r in results)
    qk_total_opt = sum(r['qk_optimized'] for r in results)

    print(f"Benchmarks compared: {len(results)}")
    print(f"TileUniverse wins: {tu_wins}")
    print(f"Qiskit -O3 wins: {qk_wins}")
    print(f"Ties: {ties}")
    print()
    print(f"Total original gates: {tu_total_orig:,}")
    print(f"TileUniverse total: {tu_total_opt:,} ({(1-tu_total_opt/tu_total_orig)*100:.1f}% reduction)")
    print(f"Qiskit -O3 total: {qk_total_opt:,} ({(1-qk_total_opt/tu_total_orig)*100:.1f}% reduction)")
    print()

    if tu_total_opt < qk_total_opt:
        print(f">>> TileUniverse produces {qk_total_opt - tu_total_opt:,} fewer gates overall <<<")
    elif qk_total_opt < tu_total_opt:
        print(f">>> Qiskit -O3 produces {tu_total_opt - qk_total_opt:,} fewer gates overall <<<")
    else:
        print(">>> Both optimizers produce the same total gate count <<<")

    # Save CSV
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'category', 'original', 'tu_optimized', 'tu_reduction', 'qk_optimized', 'qk_reduction', 'winner', 'diff'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_FILE}")

if __name__ == '__main__':
    main()
