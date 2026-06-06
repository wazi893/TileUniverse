# QEC Module Status

**Last Updated**: Sprint 78 (January 2026)

This document provides an honest assessment of the Quantum Error Correction module's current capabilities and limitations.

## Overview

The QEC module provides stabilizer-based quantum error correction simulation with O(n^2) scaling via the Aaronson-Gottesman tableau representation. It supports multiple code families and decoders.

## Supported Codes

| Code | Type | Parameters | Status |
|------|------|------------|--------|
| **Repetition** | Stabilizer | [[n, 1, n]] | Production |
| **Steane** | CSS | [[7, 1, 3]] | Production |
| **Surface** | Topological | [[d^2, 1, d]] | Production |
| **Bacon-Shor** | Subsystem | [[d^2, 1, d]] | Production (Sprint 78) |

### Code Details

**Repetition Code**
- Corrects bit-flip (X) errors only
- Simple majority-vote decoder
- Good for testing and educational purposes

**Steane [[7,1,3]] Code**
- Full CSS code with X and Z syndrome measurements
- Hamming-code based decoding
- Corrects any single-qubit error (X, Y, or Z)

**Surface Code**
- Rotated layout with proper boundary handling
- Supports distances d = 3, 5, 7, ...
- Multiple decoder options (Union-Find, MWPM)

**Bacon-Shor Code** (New in Sprint 78)
- Subsystem code with weight-2 gauge measurements
- XX gauges (horizontal) detect Z errors
- ZZ gauges (vertical) detect X errors
- 1D repetition decoder per row/column

## Decoder Matrix

| Code | MWPM | Union-Find | Lookup | Repetition |
|------|------|------------|--------|------------|
| Surface | Available | Default | - | - |
| Steane | - | - | Default | - |
| Repetition | - | - | - | Default |
| Bacon-Shor | - | - | - | Default |

### Decoder Details

**MWPMDecoderFB** (New in Sprint 78)
- True minimum-weight perfect matching
- Uses `SyndromeGraphBuilder` abstraction for graph construction
- `PhenomenologicalGraphBuilder` for standard surface code decoding
- Available for surface codes

**Union-Find Decoders**
- `UnionFindDecoder`: Basic weighted cluster growth
- `UnionFindDecoderV2`: Improved edge-weighted matching
- `UnionFindDecoderDN`: Delfosse-Nickerson style

**Lookup Decoder**
- Pre-computed syndrome-to-correction mapping
- Fast for small codes (Steane)

## Noise Models Supported

| Model | Status | Implementation |
|-------|--------|----------------|
| Phenomenological (post-circuit) | Supported | `DepolarizingNoise`, `BitFlipNoise`, `PhaseFlipNoise` |
| Gate-level depolarizing | Supported | `GateErrorModel` in `execute_gate_noisy()` |
| Measurement bit-flip | Supported | Via `p_meas` in `GateErrorModel` |
| State prep bit-flip | Supported | Via `p_prep` in `GateErrorModel` |
| Correlated errors | Not implemented | - |
| Leakage | Not implemented | - |
| Coherent over-rotations | Not implemented | - |
| Biased noise | Not implemented | - |

### Circuit-Level Noise (Sprint 78)

The `GateErrorModel` provides per-gate error injection:

```rust
pub struct GateErrorModel {
    pub p_single: f64,    // 1Q depolarizing: X/Y/Z each with p/3
    pub p_two: f64,       // 2Q depolarizing: 15 non-II Paulis each with p/15
    pub p_meas: f64,      // Measurement: classical bit-flip
    pub p_prep: f64,      // State prep: classical bit-flip
    pub p_idle: f64,      // Idle error per cycle
    pub cycle_duration_ns: Option<f64>,  // What is a "cycle"?
}
```

**Preset Models**:
- `GateErrorModel::superconducting_ballpark()`: ~0.1% single, ~1% two-qubit
- `GateErrorModel::ion_trap_ballpark()`: ~0.01% single, ~0.1% two-qubit
- `GateErrorModel::ideal()`: No errors (for baseline comparison)

**Important**: These are illustrative ballpark numbers, not canonical hardware specifications.

## Critical Caveats

### MWPM Graph Builder Limitation

> **The MWPM graph builder (`PhenomenologicalGraphBuilder`) is calibrated for phenomenological noise.**
>
> Circuit-level noise changes the detector graph structure (space-time correlations). Using MWPM with circuit-level noise currently uses the phenomenological graph, which is a **known approximation**.

A future `CircuitLevelGraphBuilder` would be needed for accurate circuit-level MWPM decoding.

### Threshold Numbers Are Model-Dependent

Do not trust hard-coded threshold assertions in tests. Threshold behavior depends on:
- Noise model (phenomenological vs circuit-level)
- Code family and distance
- Decoder choice and configuration
- Graph construction assumptions

Tests verify correctness and monotonic improvement, not specific threshold values.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Stabilizer simulation | O(n^2) memory | Aaronson-Gottesman tableau |
| Clifford gates | O(n) | Per gate |
| Measurement | O(n) | Deterministic fast path |
| MWPM decoding | O(n^3) worst | Average O(n) for low error rates |
| Union-Find decoding | O(n * alpha(n)) | Nearly linear |

## API Quick Reference

### Bacon-Shor Code
```rust
use engine::qec::bacon_shor::{BaconShorCode, BaconShorDecoder};

let mut code = BaconShorCode::new(3);  // d=3 (9 qubits)
code.apply_error(4, 'X');              // Apply X error to center qubit
let success = code.error_correction_round();
assert!(success);
```

### Circuit-Level Noise
```rust
use engine::qec::gate_noise::GateErrorModel;
use engine::qec::stabilizer::StabilizerTableau;

let model = GateErrorModel::superconducting_ballpark();
let mut tableau = StabilizerTableau::new(5);
let mut rng = SimpleRng::new(42);

// Execute circuit with noise
let errors = tableau.execute_circuit_noisy(&circuit, Some(&model), &mut rng)?;
```

### MWPM Decoder
```rust
use engine::qec::mwpm_decoder::MWPMDecoderFB;

let mut decoder = MWPMDecoderFB::for_surface_code(5, 0.01);
let correction = decoder.decode(&x_syndrome, &z_syndrome);
```

## Test Coverage

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| `bacon_shor` | 16 | 3 |
| `gate_noise` | 13 | - |
| `mwpm_decoder` | 8 | - |
| `syndrome_graph` | 6 | - |
| `codes` | 20+ | - |
| `stabilizer` | 20+ | - |

## File Reference

| File | Purpose |
|------|---------|
| `src/qec/bacon_shor.rs` | Bacon-Shor subsystem code |
| `src/qec/gate_noise.rs` | Circuit-level noise model |
| `src/qec/mwpm_decoder.rs` | MWPM decoder wrapper |
| `src/qec/syndrome_graph.rs` | Graph builders for MWPM |
| `src/qec/stabilizer.rs` | Core stabilizer tableau |
| `src/qec/codes.rs` | Repetition, Steane, Surface codes |
| `src/qec/decoder.rs` | Union-Find and lookup decoders |
