# CLAUDE.md - TileUniverse

## What This Is

TileUniverse is a **high-performance quantum/cellular simulation engine** in Rust with Python bindings.

**Core capabilities:**
- **Quantum**: 15.8 TCOPS on RTX 5090, multi-backend (CPU scalar/AVX/JIT, GPU CUDA/Tensor Core)
- **Cellular**: 115T tiles/sec packed 1-bit evaluation (register-resident), sparse eval for stable circuits
- **Sparse Quantum**: O(1) GHZ states up to infinite qubits (see below)
- **QEC**: Stabilizer simulation, Surface/Steane codes, Union-Find decoder
- **SNN**: Quantum-SNN hybrid with CUDA acceleration

Design priorities: deterministic, reproducible, cross-backend parity.

## Key Insight: Sparse Quantum Simulation

The `tile8` module achieves **unlimited-scale GHZ states** through sparse representation.

**Why it works:** GHZ state |GHZ_n⟩ = (|00...0⟩ + |11...1⟩)/√2 has exactly **2 non-zero amplitudes** regardless of n. We store only those two amplitudes; the qubit count is metadata.

| Struct | Limit | Use Case |
|--------|-------|----------|
| `MinimalGhzState` | 2^64 qubits | Fast path for "normal" large states |
| `UnlimitedGhzState` | 10^100M qubits | BigUint qubit count |
| `SymbolicGhzState` | ∞ (ℵ₀) | Graham's number, TREE(3), infinite |
| `SparseQuantumGridVec` | Billions | W-states with O(n) memory |

Location: `src/tile8/sparse_quantum_vec.rs`

## Key Insight: Packed Tile Evaluation

The `cuda_tiles.rs` module packs 64 boolean tiles per u64, achieving up to 375× speedup over naive u64-per-tile.

**Why it works:** Warp shuffles for horizontal neighbors (zero-cost), L2 cache for vertical, single-instruction evaluation (`output = left | right | up | down` processes 64 tiles).

**Kernel variants and benchmarks (RTX 5090):**

| Variant | Grid | Throughput | Notes |
|---------|------|------------|-------|
| Basic | 8K×8K | 25.89 T/s | Simple, portable |
| Register-V2 | 8K×8K | 54.12 T/s | 8 rows + coop halo |
| Register-V3 | 8K×8K | 75.00 T/s | 32 rows register-resident |
| Register-V3 | 16K×16K | 98.93 T/s | Larger grid |
| **Register-V3** | **32K×32K** | **115.61 T/s** | 1B tiles, peak throughput |
| Heterogeneous | — | — | Mixed AND/OR/XOR per-word |

Run: `cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3`

## Key Insight: Sparse Evaluation

For circuits where 99%+ of tiles are stable, evaluate only active tiles.

**The math:** Dense evaluates ALL tiles every tick. Sparse evaluates only dirty tiles. For 0.1% activity → 1000× speedup.

**GPU-native optimizations:**
- **Hierarchical dirty tracking** — L0/L1/L2 summary bitsets for O(active_regions) worklist construction
- **GPU prefix sum** — Blelloch scan eliminates GPU↔CPU transfers (was 120MB round-trip for 1B tiles)
- **Warp-aggregated propagation** — `__ballot_sync`/`__shfl_sync` reduces atomic contention 4-16×
- **GPU popcount reduction** — Stats collection without downloading bitsets

**Sparse evaluation benchmarks (RTX 5090):**

| Grid | Per-Tick Latency | Effective Throughput |
|------|------------------|---------------------|
| 1M tiles | 0.19 ms | 5.3 Gtiles/s |
| 16M tiles | 0.43 ms | 37 Gtiles/s |
| 67M tiles | 1.04 ms | 64 Gtiles/s |
| 268M tiles | 3.68 ms | 73 Gtiles/s |

Key structs in `cuda_tiles.rs`: `GpuSparseContext`, `run_sparse_tick`, hierarchical kernels.

Run: `cargo run --release --features cuda --example sparse_perf_check`

## Key Insight: Timing-Aware Simulation

The CPU simulation path (`simulation.rs`) models propagation delays for realistic critical path analysis.

**Distance-based wire delays:** Wire delay scales with length: `delay = 1 + length/10`. Use `compute_wire_delays()` to auto-calculate via BFS from signal sources.

**Timing features:**
- Per-tile `wire_delay` storage with `effective_delay()` lookup
- Critical path tracking with `trace_critical_path()`
- Glitch detection for setup/hold violations
- `TimingStats` with convergence metrics

## Repository Structure

```
src/
├── lib.rs              # Module exports
├── quantum.rs          # Core quantum state, gates, scalar backend
├── cuda.rs             # CUDA kernels (FP32, WMMA Tensor Core)
├── cuda_tiles.rs       # Packed tiles, sparse eval, Ising/QUBO
├── fusion.rs           # Gate fusion, identity elimination
├── simulation.rs       # Classical tile evaluation
├── tile8/              # Sparse quantum (GHZ, W-state)
├── qec/                # Quantum error correction
├── snn/                # Spiking neural networks
├── algorithms/         # Grover, DJ, BV, Shor, VQE, QAOA
└── bin/                # CLI, benchmarks

python/
├── tileuniverse/       # Python package
│   ├── __init__.py     # Main API
│   └── rl/             # Gymnasium integration
└── tests/

tests/                  # Integration tests
examples/               # Demos and benchmarks
```

## Build Commands

```bash
# Standard
cargo build
cargo test

# With GPU
cargo build --features cuda

# Full features
cargo build --release --features cuda,quantum_jit,cranelift_jit,perf-bench

# Python bindings
cd python && maturin develop --release
```

## Feature Flags

| Feature | Purpose |
|---------|---------|
| `cuda` | CUDA GPU backend (requires Toolkit 12.0+) |
| `quantum_jit` | JIT compilation infrastructure |
| `cranelift_jit` | Cranelift JIT backend |
| `perf-bench` | Benchmarking APIs |
| `cluster` | Distributed/cluster mode |

## Architecture Principles

**Determinism:** Classical logic runs in stable order. Measurement uses scalar backend only. No hidden nondeterminism.

**Cross-Backend Parity:** Scalar is authoritative reference. AVX/JIT/GPU must match within ε ≈ 1e-6. See `tests/compare3_parity.rs`.

**One Gate Per Tile Per Tick:** Core evolution applies single quantum gate per tile each tick. Depth batching happens at fusion level.

## Files to Read First

When working on this codebase:
- `src/lib.rs` - Module structure
- `Cargo.toml` - Dependencies and features
- `src/quantum.rs` - Quantum primitives
- `src/cuda.rs` - GPU backend
- `src/cuda_tiles.rs` - Packed tiles, sparse eval
- `src/tile8/sparse_quantum_vec.rs` - Unlimited GHZ

For specific domains:
- QEC: `src/qec/mod.rs`
- SNN: `src/snn/mod.rs`
- Algorithms: `src/algorithms/`
- Python: `python/tileuniverse/__init__.py`
