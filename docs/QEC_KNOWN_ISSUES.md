# QEC Module Known Issues and Improvement Roadmap

**Last Updated**: Sprint 78 (January 2026)

This document tracks known limitations, gaps, and planned improvements for the QEC module.

---

## Current Limitations

### 1. MWPM Graph Builder is Phenomenological Only

**Status**: Known limitation

**Description**: The `PhenomenologicalGraphBuilder` constructs decoding graphs assuming phenomenological noise (errors applied post-circuit). For circuit-level noise, the detector graph needs space-time correlations that track where errors can propagate during the actual measurement circuit.

**Impact**: Using MWPM with circuit-level noise (`GateErrorModel`) currently applies a phenomenological graph, which is an approximation. This may underestimate error correction performance for circuit-level simulations.

**Workaround**: For accurate circuit-level MWPM, use Union-Find decoder instead (it's graph-agnostic).

**Future Work**: Implement `CircuitLevelGraphBuilder` that constructs detector graphs from the actual syndrome measurement circuit with proper time-like edges.

---

### 2. Threshold Numbers Are Model-Dependent

**Status**: By design

**Description**: QEC thresholds (e.g., "~10.3% for surface code with MWPM") are specific to:
- Noise model (phenomenological, circuit-level, biased)
- Code lattice (rotated vs unrotated surface code)
- Decoder graph construction
- Number of syndrome measurement rounds

**Impact**: Tests should not assert hard-coded threshold values. Instead, they should verify:
- Correctness on hand-constructed syndromes
- Monotonic improvement (MWPM > Union-Find > greedy)
- Single-error correction = 100% success

**Policy**: Benchmark tests may print estimated thresholds but should not assert specific values.

---

### 3. No Correlated or Leakage Errors

**Status**: Not implemented

**Description**: The `GateErrorModel` implements independent depolarizing channels. Real hardware exhibits:
- **Correlated errors**: Two-qubit gates may cause correlated failures
- **Leakage**: Qubits may transition to non-computational states (|2⟩, |3⟩, ...)
- **Coherent errors**: Systematic over/under-rotations
- **Biased noise**: Z errors more likely than X errors (or vice versa)

**Impact**: Simulations with current noise models are optimistic compared to real hardware.

**Future Work**:
- Add `CorrelatedNoiseModel` with spatial correlation functions
- Add `LeakageModel` with leakage and seepage rates
- Add `BiasedNoiseModel` with configurable X/Z bias

---

### 4. Missing Code Families

**Status**: Partial coverage

**Implemented**:
- Repetition code (bit-flip only)
- Steane [[7,1,3]] CSS code
- Surface code (rotated layout)
- Bacon-Shor subsystem code

**Not Implemented**:
| Code Family | Why It Matters |
|-------------|----------------|
| **Color codes** | Lower overhead for some operations, transversal gates |
| **Floquet codes** | Time-dynamic codes, potentially higher thresholds |
| **LDPC codes** | Good asymptotic scaling, relevant for quantum memory |
| **Concatenated codes** | Hierarchical error correction |
| **Foliated codes** | 3D extensions for specific architectures |

---

### 5. Missing Decoders

**Status**: Partial coverage

**Implemented**:
- Union-Find (multiple variants)
- MWPM via weighted matching
- Lookup (for small codes)
- 1D repetition (for Bacon-Shor)

**Not Implemented**:
| Decoder | Why It Matters |
|---------|----------------|
| **Belief Propagation (BP)** | Fast, parallelizable, good for LDPC |
| **Integer Linear Programming (ILP)** | Optimal decoding (slow but accurate) |
| **Neural network decoders** | Potentially faster than MWPM |
| **Sliding window decoders** | Necessary for real-time decoding |

---

### 6. Bacon-Shor Decoder is Simple

**Status**: Known limitation

**Description**: The Bacon-Shor decoder uses 1D repetition decoding on rows/columns. This is correct but not optimal. More sophisticated decoders exist that exploit the full gauge structure.

**Impact**: Threshold may be lower than theoretically achievable.

**Future Work**: Implement proper Bacon-Shor matching decoder.

---

## Resolved Issues (Sprint 78)

### MWPM Was Greedy Placeholder

**Resolved in**: Sprint 78.1

**Original Issue**: `MWPMDecoder` at `decoder.rs:94-160` used greedy sequential matching, not true minimum-weight perfect matching.

**Resolution**: Added `MWPMDecoderFB` with `SyndromeGraphBuilder` abstraction. The old `MWPMDecoder` is now deprecated.

---

### No Circuit-Level Noise

**Resolved in**: Sprint 78.2

**Original Issue**: Only phenomenological noise existed. No error injection during gate execution.

**Resolution**: Added `GateErrorModel` with per-gate Pauli channel injection via `execute_gate_noisy()` and `execute_circuit_noisy()`.

---

### No Subsystem Codes

**Resolved in**: Sprint 78.3

**Original Issue**: Only stabilizer codes implemented. No subsystem code support.

**Resolution**: Added `BaconShorCode` with weight-2 gauge measurements and 1D repetition decoder.

---

## Improvement Roadmap

### Near-Term (Sprint 79-80)

1. **CircuitLevelGraphBuilder** - Space-time correlations for MWPM
2. **Biased noise model** - Configurable X/Z bias ratio
3. **Measurement rounds** - Multiple syndrome extraction rounds

### Medium-Term (Sprint 81-85)

4. **Color codes** - [[4,2,2]] and hexagonal layouts
5. **Belief propagation decoder** - For LDPC-style codes
6. **GPU-accelerated decoding** - CUDA kernels for Union-Find

### Long-Term

7. **LDPC codes** - qLDPC with good asymptotic scaling
8. **Real-time decoder interface** - Streaming syndrome input
9. **Hardware noise characterization** - Import from device calibration

---

## Testing Philosophy

### Do

- Test algebraic correctness (gauge products = stabilizers)
- Test single-error correction (100% for d >= 3)
- Test monotonic improvement (better decoder > worse decoder)
- Test logical operator properties (anticommutation)

### Don't

- Assert specific threshold percentages
- Hardcode expected error rates
- Assume one noise model applies to all scenarios

---

## Contributing

When adding new codes or decoders:

1. Follow existing patterns in `src/qec/codes.rs`
2. Add unit tests for algebraic properties
3. Add integration tests for error correction
4. Update this document and `QEC_STATUS.md`
5. Do NOT claim specific threshold values without extensive validation
