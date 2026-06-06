#!/usr/bin/env python3
"""
Qiskit Aer GPU Benchmark (cuStateVec)
=====================================

This script benchmarks Qiskit Aer's GPU backend (cuStateVec) with best practices:
1. Handle reuse: Single AerSimulator instance across all runs
2. Workspace reuse: Aer manages workspace internally
3. Proper sync: Use explicit barrier and result extraction
4. Warmup: Multiple warmup runs before timing
5. Consistent circuit structure: QV-style (H + CNOT ladder)

Run in WSL2 with qiskit-aer-gpu installed:
    source /opt/qiskit_env/bin/activate
    python benchmark_aer_gpu.py

Requirements:
    pip install qiskit qiskit-aer-gpu custatevec-cu12
"""

import time
import numpy as np
from typing import Dict, List, Tuple

def create_qv_circuit(n_qubits: int, depth: int):
    """Create a QV-style circuit: H on all qubits + CNOT ladder per layer"""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    for layer in range(depth):
        # Hadamard on all qubits
        for q in range(n_qubits):
            qc.h(q)

        # CNOT ladder (even pairs)
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)

        # CNOT ladder (odd pairs)
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)

    return qc


def benchmark_gpu_custatevec(
    n_qubits: int,
    depth: int,
    warmup_runs: int = 3,
    timed_runs: int = 5,
    simulator = None
) -> Dict:
    """
    Benchmark Aer GPU with cuStateVec.

    Best practices implemented:
    - Reuse simulator instance (passed as parameter)
    - Warmup before timing
    - Extract full result to ensure completion
    - Multiple runs for statistical reliability
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # Create circuit
    qc = create_qv_circuit(n_qubits, depth)
    gate_count = qc.size()

    # Add statevector save instruction
    qc_save = qc.copy()
    qc_save.save_statevector()

    # Create simulator if not provided (for handle reuse)
    if simulator is None:
        simulator = AerSimulator(
            method='statevector',
            device='GPU',
            cuStateVec_enable=True,
            # Workspace is managed internally by cuStateVec
        )

    # Warmup runs (critical for accurate timing)
    for _ in range(warmup_runs):
        job = simulator.run(qc_save, shots=1)
        result = job.result()
        _ = result.get_statevector()  # Force completion

    # Timed runs
    times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        job = simulator.run(qc_save, shots=1)
        result = job.result()
        sv = result.get_statevector()  # Force GPU completion
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Verify we got a valid statevector
        assert len(sv.data) == 2**n_qubits, f"Invalid statevector size"

    # Statistics
    min_time = min(times)
    avg_time = sum(times) / len(times)
    std_time = np.std(times)

    # Compute TCOPS (same formula as TileUniverse)
    # amplitude_ops = 2^n * gates
    # TCOPS = amplitude_ops / time / 1e12
    amp_ops = (2 ** n_qubits) * gate_count
    tcops = amp_ops / min_time / 1e12
    gops = amp_ops / min_time / 1e9

    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'gates': gate_count,
        'min_time_s': min_time,
        'avg_time_s': avg_time,
        'std_time_s': std_time,
        'tcops': tcops,
        'gops': gops,
        'times': times,
    }


def benchmark_gpu_fp32_mode(
    n_qubits: int,
    depth: int,
    warmup_runs: int = 3,
    timed_runs: int = 5,
) -> Dict:
    """
    Benchmark Aer GPU in FP32 mode (for fair comparison with FP16 TileUniverse)
    """
    from qiskit_aer import AerSimulator

    qc = create_qv_circuit(n_qubits, depth)
    qc_save = qc.copy()
    qc_save.save_statevector()

    # FP32 precision mode
    simulator = AerSimulator(
        method='statevector',
        device='GPU',
        cuStateVec_enable=True,
        precision='single',  # FP32 instead of FP64
    )

    # Warmup
    for _ in range(warmup_runs):
        job = simulator.run(qc_save, shots=1)
        result = job.result()
        _ = result.get_statevector()

    # Timed runs
    times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        job = simulator.run(qc_save, shots=1)
        result = job.result()
        sv = result.get_statevector()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    min_time = min(times)
    gate_count = qc.size()
    amp_ops = (2 ** n_qubits) * gate_count
    tcops = amp_ops / min_time / 1e12

    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'min_time_s': min_time,
        'tcops': tcops,
        'precision': 'single',
    }


def run_comprehensive_benchmark():
    """Run comprehensive GPU benchmark across qubit counts"""
    print("=" * 70)
    print("  QISKIT AER GPU BENCHMARK (cuStateVec)")
    print("  Best practices: handle reuse, warmup, proper sync")
    print("=" * 70)
    print()

    from qiskit_aer import AerSimulator

    # Check GPU availability
    try:
        test_sim = AerSimulator(device='GPU', method='statevector')
        test_result = test_sim.run(create_qv_circuit(2, 1).copy()).result()
        print("GPU detected and working")
    except Exception as e:
        print(f"GPU not available: {e}")
        print("Falling back to CPU benchmark")
        return

    # Create single simulator instance for handle reuse
    simulator = AerSimulator(
        method='statevector',
        device='GPU',
        cuStateVec_enable=True,
    )
    print(f"Simulator: {simulator}")
    print()

    # Benchmark configurations
    configs = [
        (12, 1000),
        (16, 1000),
        (20, 1000),
        (24, 1000),
        (26, 500),
        (28, 500),
    ]

    print("FP64 Mode (default):")
    print(f"{'Qubits':>8} | {'Depth':>8} | {'Gates':>10} | {'Time (s)':>10} | {'TCOPS':>10}")
    print("-" * 56)

    fp64_results = []
    for n_qubits, depth in configs:
        try:
            result = benchmark_gpu_custatevec(
                n_qubits, depth,
                warmup_runs=3,
                timed_runs=5,
                simulator=simulator
            )
            fp64_results.append(result)
            print(f"{result['n_qubits']:>8} | {result['depth']:>8} | {result['gates']:>10} | {result['min_time_s']:>10.4f} | {result['tcops']:>10.3f}")
        except Exception as e:
            print(f"{n_qubits:>8} | {depth:>8} | ERROR: {e}")

    print()
    print("FP32 Mode (for comparison with TileUniverse):")
    print(f"{'Qubits':>8} | {'Depth':>8} | {'Time (s)':>10} | {'TCOPS':>10}")
    print("-" * 46)

    for n_qubits, depth in configs[:4]:  # Only test up to 24q for FP32
        try:
            result = benchmark_gpu_fp32_mode(
                n_qubits, depth,
                warmup_runs=3,
                timed_runs=5,
            )
            print(f"{result['n_qubits']:>8} | {result['depth']:>8} | {result['min_time_s']:>10.4f} | {result['tcops']:>10.3f}")
        except Exception as e:
            print(f"{n_qubits:>8} | {depth:>8} | ERROR: {e}")

    print()
    print("=" * 70)
    print("  BENCHMARK CONFIGURATION")
    print("=" * 70)
    print("""
Best Practices Implemented:
  1. Handle Reuse: Single AerSimulator instance for all FP64 runs
  2. Warmup: 3 warmup runs before timing
  3. Proper Sync: Extract statevector to force GPU completion
  4. Statistical: 5 timed runs, report minimum

Circuit Structure:
  - QV-style: H gates on all qubits + CNOT ladder per layer
  - Same structure as TileUniverse benchmark for fair comparison

Metric:
  - TCOPS = (2^n_qubits * gate_count) / time / 1e12
  - Consistent with TileUniverse definition
    """)

    return fp64_results


def save_results_json(results: List[Dict], filename: str = "aer_gpu_results.json"):
    """Save benchmark results to JSON for comparison"""
    import json

    # Convert to serializable format
    serializable = []
    for r in results:
        s = dict(r)
        s.pop('times', None)  # Remove raw times
        serializable.append(s)

    with open(filename, 'w') as f:
        json.dump({
            'benchmark': 'Qiskit Aer GPU (cuStateVec)',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': serializable,
        }, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    if results:
        save_results_json(results)
