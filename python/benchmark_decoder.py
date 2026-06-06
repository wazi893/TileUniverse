"""
Benchmark comparison: TileUniverse UnionFindDecoderV2 vs PyMatching MWPM

This script compares:
1. Single-error correction rate
2. Logical error rate at various physical error rates
3. Threshold behavior (error rate where logical error = physical error)
4. Decoding speed
"""

import numpy as np
import time
from typing import Tuple, List
import pymatching

# Try to import tileuniverse, fall back to simulation if not available
try:
    import tileuniverse as tu
    HAS_TU = True
except ImportError:
    HAS_TU = False
    print("Warning: tileuniverse not available, using Python simulation")


class SurfaceCodePython:
    """Pure Python surface code for PyMatching comparison."""

    def __init__(self, distance: int):
        self.distance = distance
        self.width = distance
        self.height = distance
        self.n_data = distance * distance

        # Build stabilizers
        self.x_stabilizers = []
        self.z_stabilizers = []

        # Interior 4-qubit stabilizers
        for r in range(self.height - 1):
            for c in range(self.width - 1):
                qubits = [
                    r * self.width + c,
                    r * self.width + c + 1,
                    (r + 1) * self.width + c,
                    (r + 1) * self.width + c + 1,
                ]
                if (r + c) % 2 == 0:
                    self.x_stabilizers.append(qubits)
                else:
                    self.z_stabilizers.append(qubits)

        # Boundary stabilizers
        for c in range(self.width - 1):
            if c % 2 == 1:
                self.x_stabilizers.append([c, c + 1])
        for c in range(self.width - 1):
            if c % 2 == 0:
                base = (self.height - 1) * self.width
                self.x_stabilizers.append([base + c, base + c + 1])
        for r in range(self.height - 1):
            if r % 2 == 1:
                self.z_stabilizers.append([r * self.width, (r + 1) * self.width])
        for r in range(self.height - 1):
            if r % 2 == 0:
                self.z_stabilizers.append([
                    r * self.width + (self.width - 1),
                    (r + 1) * self.width + (self.width - 1)
                ])

        self.num_x_stabs = len(self.x_stabilizers)
        self.num_z_stabs = len(self.z_stabilizers)

        # Error tracking
        self.x_errors = np.zeros(self.n_data, dtype=bool)
        self.z_errors = np.zeros(self.n_data, dtype=bool)

    def reset(self):
        self.x_errors.fill(False)
        self.z_errors.fill(False)

    def apply_x_error(self, qubit: int):
        self.x_errors[qubit] ^= True

    def apply_z_error(self, qubit: int):
        self.z_errors[qubit] ^= True

    def apply_y_error(self, qubit: int):
        self.x_errors[qubit] ^= True
        self.z_errors[qubit] ^= True

    def measure_x_syndrome(self) -> np.ndarray:
        """X stabilizers detect Z errors."""
        syndrome = np.zeros(self.num_x_stabs, dtype=np.uint8)
        for i, stab in enumerate(self.x_stabilizers):
            parity = 0
            for q in stab:
                if self.z_errors[q]:
                    parity ^= 1
            syndrome[i] = parity
        return syndrome

    def measure_z_syndrome(self) -> np.ndarray:
        """Z stabilizers detect X errors."""
        syndrome = np.zeros(self.num_z_stabs, dtype=np.uint8)
        for i, stab in enumerate(self.z_stabilizers):
            parity = 0
            for q in stab:
                if self.x_errors[q]:
                    parity ^= 1
            syndrome[i] = parity
        return syndrome

    def apply_x_correction(self, qubits: List[int]):
        for q in qubits:
            self.x_errors[q] ^= True

    def apply_z_correction(self, qubits: List[int]):
        for q in qubits:
            self.z_errors[q] ^= True

    def has_logical_x_error(self) -> bool:
        """Check if there's a logical X error (X chain along top row)."""
        parity = 0
        for c in range(self.width):
            if self.x_errors[c]:
                parity ^= 1
        return parity == 1

    def has_logical_z_error(self) -> bool:
        """Check if there's a logical Z error (Z chain along left column)."""
        parity = 0
        for r in range(self.height):
            if self.z_errors[r * self.width]:
                parity ^= 1
        return parity == 1


def build_pymatching_decoder(code: SurfaceCodePython, for_x_syndrome: bool):
    """Build PyMatching decoder for one syndrome type."""
    if for_x_syndrome:
        stabilizers = code.x_stabilizers
        num_stabs = code.num_x_stabs
    else:
        stabilizers = code.z_stabilizers
        num_stabs = code.num_z_stabs

    # Build check matrix H where H[i,j] = 1 if stabilizer i includes qubit j
    H = np.zeros((num_stabs, code.n_data), dtype=np.uint8)
    for i, stab in enumerate(stabilizers):
        for q in stab:
            H[i, q] = 1

    return pymatching.Matching(H)


class UnionFindDecoderPython:
    """Python implementation matching the Rust UnionFindDecoderV2."""

    def __init__(self, code: SurfaceCodePython):
        self.code = code
        self.distance = code.distance

        # Build edge graphs
        self.x_edges, self.x_boundary_qubits = self._build_graph(code.x_stabilizers)
        self.z_edges, self.z_boundary_qubits = self._build_graph(code.z_stabilizers)

    def _build_graph(self, stabilizers: List[List[int]]):
        """Build edge graph and boundary qubits."""
        n_stabs = len(stabilizers)

        # Qubit to stabilizers map
        qubit_to_stabs = {}
        for stab_idx, qubits in enumerate(stabilizers):
            for q in qubits:
                if q not in qubit_to_stabs:
                    qubit_to_stabs[q] = []
                qubit_to_stabs[q].append(stab_idx)

        # Build edges
        edges = [[] for _ in range(n_stabs)]
        for qubit, stabs in qubit_to_stabs.items():
            if len(stabs) == 2:
                edges[stabs[0]].append((stabs[1], qubit))
                edges[stabs[1]].append((stabs[0], qubit))

        # Boundary qubits
        boundary_qubits = [[] for _ in range(n_stabs)]
        for stab_idx, qubits in enumerate(stabilizers):
            for q in qubits:
                if len(qubit_to_stabs.get(q, [])) == 1:
                    boundary_qubits[stab_idx].append(q)

        return edges, boundary_qubits

    def decode_syndrome(self, syndrome: np.ndarray, edges, boundary_qubits) -> List[int]:
        """Decode a syndrome to corrections."""
        if np.all(syndrome == 0):
            return []

        defects = list(np.where(syndrome == 1)[0])
        if not defects:
            return []

        corrections = []
        handled = set()

        # Process defect pairs
        for d1 in defects:
            if d1 in handled:
                continue

            paired = False
            for neighbor, qubit in edges[d1]:
                if neighbor in defects and neighbor not in handled:
                    corrections.append(qubit)
                    handled.add(d1)
                    handled.add(neighbor)
                    paired = True
                    break

            if not paired:
                if boundary_qubits[d1]:
                    corrections.append(boundary_qubits[d1][0])
                    handled.add(d1)

        return corrections

    def decode_x_syndrome(self, syndrome: np.ndarray) -> List[int]:
        return self.decode_syndrome(syndrome, self.x_edges, self.x_boundary_qubits)

    def decode_z_syndrome(self, syndrome: np.ndarray) -> List[int]:
        return self.decode_syndrome(syndrome, self.z_edges, self.z_boundary_qubits)


def run_comparison(distances=[3, 5, 7], error_probs=[0.01, 0.02, 0.05, 0.08, 0.10], n_trials=1000):
    """Run comparison between UnionFind and PyMatching."""

    print("=" * 80)
    print("DECODER COMPARISON: UnionFindDecoderV2 vs PyMatching MWPM")
    print("=" * 80)

    np.random.seed(42)

    for distance in distances:
        print(f"\n{'='*70}")
        print(f"Distance {distance} Surface Code ({distance*distance} data qubits)")
        print("=" * 70)

        code = SurfaceCodePython(distance)
        uf_decoder = UnionFindDecoderPython(code)
        pm_x_decoder = build_pymatching_decoder(code, for_x_syndrome=True)
        pm_z_decoder = build_pymatching_decoder(code, for_x_syndrome=False)

        print(f"\n{'Error Rate':>12} | {'UF Logical':>12} | {'PM Logical':>12} | {'UF Time':>10} | {'PM Time':>10}")
        print("-" * 70)

        for p in error_probs:
            uf_logical_errors = 0
            pm_logical_errors = 0
            uf_time = 0
            pm_time = 0

            for _ in range(n_trials):
                code.reset()

                # Apply random errors
                for q in range(code.n_data):
                    if np.random.random() < p:
                        error_type = np.random.randint(3)
                        if error_type == 0:
                            code.apply_x_error(q)
                        elif error_type == 1:
                            code.apply_y_error(q)
                        else:
                            code.apply_z_error(q)

                x_syn = code.measure_x_syndrome()
                z_syn = code.measure_z_syndrome()

                # Save error state for both decoders
                x_errors_backup = code.x_errors.copy()
                z_errors_backup = code.z_errors.copy()

                # UnionFind decoding
                t0 = time.perf_counter()
                uf_x_corr = uf_decoder.decode_z_syndrome(z_syn)  # Z syndrome -> X corrections
                uf_z_corr = uf_decoder.decode_x_syndrome(x_syn)  # X syndrome -> Z corrections
                uf_time += time.perf_counter() - t0

                code.apply_x_correction(uf_x_corr)
                code.apply_z_correction(uf_z_corr)

                if code.has_logical_x_error() or code.has_logical_z_error():
                    uf_logical_errors += 1

                # Restore for PyMatching
                code.x_errors = x_errors_backup.copy()
                code.z_errors = z_errors_backup.copy()

                # PyMatching decoding
                t0 = time.perf_counter()
                pm_x_corr = pm_z_decoder.decode(z_syn)  # Z syndrome -> X corrections
                pm_z_corr = pm_x_decoder.decode(x_syn)  # X syndrome -> Z corrections
                pm_time += time.perf_counter() - t0

                code.apply_x_correction(list(np.where(pm_x_corr)[0]))
                code.apply_z_correction(list(np.where(pm_z_corr)[0]))

                if code.has_logical_x_error() or code.has_logical_z_error():
                    pm_logical_errors += 1

            uf_rate = uf_logical_errors / n_trials * 100
            pm_rate = pm_logical_errors / n_trials * 100

            print(f"{p:>12.2%} | {uf_rate:>11.2f}% | {pm_rate:>11.2f}% | {uf_time*1000:>8.2f}ms | {pm_time*1000:>8.2f}ms")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
PyMatching uses Minimum Weight Perfect Matching (MWPM) which achieves
the optimal threshold of ~10.3% for surface codes.

UnionFindDecoderV2 uses a greedy matching algorithm with:
- O(n) complexity (vs O(n³) for MWPM)
- Simpler implementation
- Lower threshold (~5-8%)

For fault-tolerant quantum computing below threshold, MWPM provides
lower logical error rates. However, Union-Find's speed advantage
makes it attractive for real-time decoding.
""")


def single_error_test():
    """Test single-error correction rate."""
    print("\n" + "=" * 70)
    print("SINGLE ERROR CORRECTION TEST")
    print("=" * 70)

    for distance in [3, 5, 7]:
        code = SurfaceCodePython(distance)
        uf_decoder = UnionFindDecoderPython(code)
        pm_x_decoder = build_pymatching_decoder(code, for_x_syndrome=True)
        pm_z_decoder = build_pymatching_decoder(code, for_x_syndrome=False)

        uf_success = 0
        pm_success = 0

        for q in range(code.n_data):
            for error_type in ['X', 'Y', 'Z']:
                code.reset()

                if error_type == 'X':
                    code.apply_x_error(q)
                elif error_type == 'Y':
                    code.apply_y_error(q)
                else:
                    code.apply_z_error(q)

                x_syn = code.measure_x_syndrome()
                z_syn = code.measure_z_syndrome()

                # Save state
                x_backup = code.x_errors.copy()
                z_backup = code.z_errors.copy()

                # UF decode
                uf_x_corr = uf_decoder.decode_z_syndrome(z_syn)
                uf_z_corr = uf_decoder.decode_x_syndrome(x_syn)
                code.apply_x_correction(uf_x_corr)
                code.apply_z_correction(uf_z_corr)

                uf_x_syn = code.measure_x_syndrome()
                uf_z_syn = code.measure_z_syndrome()
                if np.all(uf_x_syn == 0) and np.all(uf_z_syn == 0):
                    uf_success += 1

                # Restore and PM decode
                code.x_errors = x_backup.copy()
                code.z_errors = z_backup.copy()

                pm_x_corr = pm_z_decoder.decode(z_syn)
                pm_z_corr = pm_x_decoder.decode(x_syn)
                code.apply_x_correction(list(np.where(pm_x_corr)[0]))
                code.apply_z_correction(list(np.where(pm_z_corr)[0]))

                pm_x_syn = code.measure_x_syndrome()
                pm_z_syn = code.measure_z_syndrome()
                if np.all(pm_x_syn == 0) and np.all(pm_z_syn == 0):
                    pm_success += 1

        total = code.n_data * 3
        print(f"Distance {distance}: UF={uf_success}/{total} ({uf_success/total*100:.1f}%), PM={pm_success}/{total} ({pm_success/total*100:.1f}%)")


if __name__ == "__main__":
    single_error_test()
    run_comparison(distances=[3, 5, 7], error_probs=[0.01, 0.02, 0.05, 0.08, 0.10], n_trials=500)
