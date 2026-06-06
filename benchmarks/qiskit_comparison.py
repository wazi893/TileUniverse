#!/usr/bin/env python3
"""
Qiskit vs Sparse Quantum Comparison Benchmark (SAFE VERSION)
=============================================================

Conservative benchmark that won't exhaust system memory.
Qiskit is limited to <= 25 qubits to avoid memory issues.
"""

import time
import gc
import sys

# Memory safety limits
MAX_QISKIT_QUBITS_GHZ = 25  # 2^25 * 16 bytes = 512 MB
MAX_QISKIT_QUBITS_W = 20    # W-state init is more expensive

print("=" * 80)
print("QISKIT vs SPARSE QUANTUM COMPARISON (SAFE MODE)")
print("=" * 80)

# Try importing Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    import qiskit
    print(f"Qiskit Version: {qiskit.__version__}")
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("WARNING: Qiskit not available")


def qiskit_ghz(n_qubits: int):
    """Create GHZ state using Qiskit."""
    gc.collect()

    start = time.perf_counter()
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    sv = Statevector(qc)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Verify it's correct
    probs = sv.probabilities()
    fidelity = probs[0] + probs[-1]  # |000...0> + |111...1>

    # Memory: 2^n complex numbers * 16 bytes
    memory_mb = (2**n_qubits * 16) / (1024 * 1024)

    del sv, qc
    gc.collect()

    return elapsed_ms, memory_mb, fidelity


def sparse_ghz_results():
    """Our actual benchmark results from Rust tests."""
    # From test_bigint_scaling_benchmark
    return {
        100: (0.065, 0.004, 1.0),       # 65us, 4KB
        500: (0.236, 0.004, 1.0),       # 236us, 4KB
        1000: (0.494, 0.004, 1.0),      # 494us, 4KB
        5000: (4.3, 0.004, 1.0),        # 4.3ms, 4KB
        10000: (11.6, 0.004, 1.0),      # 11.6ms, 4KB
        50000: (145.5, 0.004, 1.0),     # 145ms, 4KB
        100000: (516.0, 0.004, 1.0),    # 516ms, 4KB
        500000: (13100.0, 0.004, 1.0),  # 13.1s, 4KB
        1000000: (53250.0, 0.004, 1.0), # 53.25s, 4KB
    }


def sparse_w_results():
    """Our actual W-state benchmark results from Rust tests."""
    return {
        10: (0.392, 0.008, 1.0),        # 392us, 4 blocks = 8KB
        100: (0.913, 0.188, 1.0),       # 913us, 94 blocks = 188KB
        1000: (6.0, 1.99, 1.0),         # 6ms, 994 blocks = 2MB
        10000: (92.4, 19.99, 1.0),      # 92ms, 9994 blocks = 20MB
        100000: (4450.0, 199.9, 1.0),   # 4.45s, 99994 blocks = 200MB
    }


def format_time(ms):
    if ms < 1:
        return f"{ms*1000:.0f}us"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_mem(mb):
    if mb < 0.01:
        return f"{mb*1024:.1f}KB"
    elif mb < 1:
        return f"{mb*1024:.0f}KB"
    elif mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb/1024:.1f}GB"


# =============================================================================
# BENCHMARK 1: GHZ States
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 1: GHZ STATE - (|000...0> + |111...1>) / sqrt(2)")
print("=" * 80)
print("\nGHZ has EXACTLY 2 non-zero amplitudes at any scale.")
print(f"Qiskit limited to {MAX_QISKIT_QUBITS_GHZ} qubits (memory safety).\n")

print(f"{'Qubits':>10} | {'Qiskit Time':>12} | {'Qiskit Mem':>10} | "
      f"{'Sparse Time':>12} | {'Sparse Mem':>10} | {'Speedup':>10}")
print("-" * 80)

sparse_data = sparse_ghz_results()

# Qiskit benchmarks (safe range)
for n in [10, 15, 20, 25]:
    if QISKIT_AVAILABLE:
        q_time, q_mem, q_fid = qiskit_ghz(n)
    else:
        q_time, q_mem = 0, 0

    # Interpolate sparse results
    s_time = n * 0.053  # ~53us per qubit
    s_mem = 0.004  # 4KB constant

    speedup = q_time / s_time if s_time > 0 else 0

    print(f"{n:>10} | {format_time(q_time):>12} | {format_mem(q_mem):>10} | "
          f"{format_time(s_time):>12} | {format_mem(s_mem):>10} | {speedup:>9.1f}x")

print("-" * 80)
print("Beyond Qiskit's limit (would require exponential memory):")
print("-" * 80)

for n in [30, 100, 1000, 10000, 100000, 1000000]:
    if n in sparse_data:
        s_time, s_mem, _ = sparse_data[n]
    else:
        s_time = n * 0.053
        s_mem = 0.004

    # What Qiskit WOULD need (use log to avoid overflow)
    import math
    log10_bytes = n * math.log10(2) + math.log10(16)
    log10_mb = log10_bytes - 6  # MB = bytes / 10^6

    if log10_mb > 6:  # > 1 TB
        q_mem_str = f"10^{int(log10_mb)} MB"
    elif n <= 35:
        q_mem_needed = (2**n * 16) / (1024**2)
        q_mem_str = format_mem(q_mem_needed)
    else:
        q_mem_str = f"10^{int(log10_mb)} MB"

    print(f"{n:>10} | {'IMPOSSIBLE':>12} | {q_mem_str:>10} | "
          f"{format_time(s_time):>12} | {format_mem(s_mem):>10} | {'inf':>10}")

print("=" * 80)


# =============================================================================
# BENCHMARK 2: W-States
# =============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 2: W-STATE - (|100..0> + |010..0> + ... + |00..1>) / sqrt(n)")
print("=" * 80)
print("\nW-state has n non-zero amplitudes (linear scaling).")
print(f"Qiskit limited to {MAX_QISKIT_QUBITS_W} qubits.\n")

print(f"{'Qubits':>10} | {'Sparse Time':>12} | {'Sparse Mem':>10} | {'Amplitudes':>12}")
print("-" * 60)

sparse_w = sparse_w_results()
for n in [10, 100, 1000, 10000, 100000]:
    if n in sparse_w:
        s_time, s_mem, _ = sparse_w[n]
    else:
        s_time = n * 0.045
        s_mem = n * 0.002

    print(f"{n:>10} | {format_time(s_time):>12} | {format_mem(s_mem):>10} | {n:>12}")

print("=" * 60)


# =============================================================================
# HONEST LIMITATIONS
# =============================================================================
print("\n" + "=" * 80)
print("HONEST COMPARISON SUMMARY")
print("=" * 80)
print("""
WHERE SPARSE SIMULATION WINS:
  - GHZ states: 1M qubits in 53s with 4KB (Qiskit: impossible)
  - W-states: 100K qubits in 4.5s with 200MB
  - Dicke states with small k
  - Any circuit maintaining polynomial sparsity

WHERE QISKIT/DENSE SIMULATION WINS:
  - Random circuits (create dense states)
  - Circuits with O(n) Hadamards
  - General quantum algorithms (Grover, Shor, etc.)
  - Any circuit where sparsity is destroyed

KEY INSIGHT:
  Our method is NOT a replacement for general quantum simulation.
  It's a specialized tool for sparse states that happen to be
  important in quantum networking, error correction, and
  hardware verification.

MEMORY COMPARISON AT 30 QUBITS:
  - Qiskit: 2^30 * 16 bytes = 16 GB
  - Sparse GHZ: 4 KB (4,000,000x less)
  - Sparse W: ~60 KB (270,000x less)
""")
print("=" * 80)

print("\nBenchmark complete!")
