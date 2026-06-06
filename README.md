<p align="center">
  <img src="python/assets/logo.svg" alt="TileUniverse" width="400">
</p>

<h1 align="center">TileUniverse</h1>

<p align="center">
  <strong>High-Performance Quantum & Cellular Simulation Engine</strong>
</p>

<p align="center">
  <a href="#performance">15.8 TCOPS</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-2024-orange.svg" alt="Rust 2024">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/cuda-12.0+-green.svg" alt="CUDA 12.0+">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

---

## Overview

TileUniverse is a high-performance simulation engine written in Rust with Python bindings. It provides two substrates for massively parallel computation:

- **Quantum Substrate**: 15.8 TCOPS on RTX 5090, 2.5 TCOPS on RTX 4070 (CUDA Tensor Cores)
- **Cellular Substrate**: 200B logic evals/sec on RTX 5090, 40B on RTX 4070

The engine achieves these numbers through optimized CUDA kernels, gate fusion, algebraic optimization, and depth-batched execution on consumer GPUs.

---

## TileUniverse in Numbers

| Metric | Value |
|---------|---|
| Quantum Simulation Throughput | **15.8 TCOPS** on RTX 5090 |
| Cellular Logic | **115T tiles/sec** packed 1-bit evaluation |
| Sparse Quantum States | **Unlimited qubits** — GHZ states with O(1) memory (2 amplitudes regardless of scale) |
| TileCpuV2 Physical Authority | **~99.8%** of execution is physical tiles (only MUL 0.2% remains as software) |
| Tests Passed | **3,175** with zero library warnings |
| GPU Benchmarks | **15.8 TCOPS** (CUDA) + **2.5 TCOPS** (Ada Lovelace) |
| Packed Eval | 375× speedup over naive u64-per-tile via 64-tile-wide bit-packing |
| Sparse Eval | O(n) memory for W-states — **billions** of stable qubits on consumer GPU |
| TileCpuV2 ISA | 32 opcodes, 16 registers, 128 ROM entries, 128 RAM cells |
| Synth Pipeline | AIG → NPN4 → placement → routing, fully automated |

---

## Performance

Verified benchmarks (January 2026):

### RTX 5090 (32GB, Blackwell)

| Substrate | Metric | Value |
|-----------|--------|-------|
| Quantum | Peak Throughput | **15.8 TCOPS** (PureMMA, 24 qubits) |
| Quantum | 32-qubit Simulation | **15.4 TCOPS** (full 32GB VRAM) |
| Cellular | Logic Evaluations | **200B evals/sec** (10+ worlds) |

### RTX 4070 (12GB, Ada Lovelace)

| Substrate | Metric | Value |
|-----------|--------|-------|
| Quantum | Throughput | **2.5 TCOPS** (WMMA) |
| Cellular | Logic Evaluations | **40B evals/sec** |
| | CPU → GPU Speedup | ~46x |

### Both GPUs

| Feature | Value |
|---------|-------|
| Parallel Worlds | 100+ concurrent |
| World Dimensions | Up to 4096x4096 |

---

## Features

### Multi-Backend Execution

The engine provides multiple execution backends with guaranteed cross-backend parity:

| Backend | Description |
|---------|-------------|
| **Scalar** | Reference implementation, used for measurement/collapse |
| **AVX2/AVX512** | Vectorized CPU kernels with tail-safe masked lanes |
| **Cranelift JIT** | Runtime code generation for quantum kernels |
| **CUDA FP32** | GPU-accelerated batched gate execution |
| **CUDA WMMA** | Tensor Core acceleration with FP16 (15x over FP32) |

### Gate Fusion & Optimization

Intelligent IR-level optimizations reduce kernel launches:

- **Identity Elimination**: H(q) → H(q) = I (removed)
- **Gate Cancellation**: X(q) → X(q) = I, Z(q) → Z(q) = I
- **Depth Batching**: N identical gates → single batched kernel
- **Layer Fusion**: Parallel gates on different qubits → fused layer
- **Mega-Fusion**: H·X·Z·Rx(θ) → single 2x2 unitary matrix

### Quantum Algorithms

Built-in implementations with correctness guarantees:

- Grover's Search (2-3 qubits: 90%+ success rate)
- Deutsch-Jozsa Algorithm
- Bernstein-Vazirani Algorithm
- VQE/QAOA circuit support

### Python Integration

Native PyO3 bindings with NumPy interop and Gymnasium RL support:

```python
import tileuniverse as tu

engine = tu.Engine(worlds=100, size=(256, 256), ruleset="gol")
engine.evolve(1000)
state = engine.get_world(0)  # Returns numpy array
```

Sprint 212 also exposes the V2 tile CPU and the synth pipeline directly from Python:

```python
import tileuniverse as tu

cpu = tu.V2Cpu.from_asm("LDI R0, 42\nHALT")
cpu.run(100)
assert cpu.reg(0) == 42

result = tu.synthesize(truth_table=0xE8, num_inputs=3)
print(result.summary())
```

---

## Quick Start

### Rust

```bash
# Clone repository
git clone https://github.com/wazi893/TileUniverse.git
cd TileUniverse/engine

# Build with CUDA support
cargo build --release --features cuda,perf-bench

# Run tests
cargo test

# Run benchmark
cargo run --release --example showcase_cops --features perf-bench
```

### Python

```bash
cd python
pip install maturin
maturin develop --release
# Optional GPU extras:
# maturin develop --release --features cuda

# Verify installation
python -c "import tileuniverse as tu; print(tu.V2Cpu.from_asm('HALT'))"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                                │
│                   tileuniverse.Engine                            │
├─────────────────────────────────────────────────────────────────┤
│                     Rust Core Engine                             │
│         PyO3 Bindings • Gate Fusion • State Management           │
├─────────────────────────────────────────────────────────────────┤
│                    Execution Backends                            │
│     Scalar │ AVX2/512 │ Cranelift JIT │ CUDA FP32 │ WMMA        │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Kernels                                 │
│     Depth-Batched Execution • Tensor Cores • L2 Optimization    │
├─────────────────────────────────────────────────────────────────┤
│                   Parallel Worlds                                │
│           World_0 │ World_1 │ World_2 │ ... │ World_n           │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `quantum.rs` | Core quantum state (SoA layout), gates, scalar backend |
| `cuda.rs` | CUDA kernels: FP32 batched, WMMA Tensor Core FP16 |
| `fusion.rs` | Gate fusion, identity elimination, depth batching |
| `algebraic_fusion.rs` | Mega-fusion: compose gate sequences into single matrices |
| `simulation.rs` | Classical logic simulation, tile evaluation |
| `algorithms/` | Grover's search, Deutsch-Jozsa, Bernstein-Vazirani |

---

## Feature Flags

```toml
[features]
cuda = []           # CUDA GPU backend (requires CUDA Toolkit 12.0+)
quantum_jit = []    # JIT compilation infrastructure
cranelift_jit = []  # Cranelift JIT backend
perf-bench = []     # Benchmarking APIs
cluster = []        # Distributed/cluster mode
config = []         # JSON/YAML config file support
```

---

## Testing

```bash
# All tests
cargo test

# Specific module tests
cargo test fusion::tests
cargo test quantum::tests

# Integration tests
cargo test --test grover_integration
cargo test --test dj_bv_integration

# Cross-backend parity tests
cargo test --test compare3_parity
```

---

## Project Structure

```
TileUniverse/
├── src/                    # Rust core engine
│   ├── lib.rs              # Module declarations
│   ├── quantum.rs          # Quantum state vectors, gates
│   ├── cuda.rs             # CUDA GPU backend
│   ├── fusion.rs           # Gate fusion & dispatch
│   ├── algorithms/         # Quantum algorithms
│   └── bin/                # CLI and benchmarks
├── crates/
│   └── logic-fabric-core/  # Core primitives (quantum, CUDA, fusion)
├── python/                 # Python bindings (PyO3 + maturin)
├── examples/               # Benchmarks and demos
├── tests/                  # Integration tests
└── User notes/SPRINTS/     # Development history (39 sprints)
```

---

## Citation

If TileUniverse contributes to your research, please cite:

```bibtex
@software{tileuniverse2024,
  title     = {TileUniverse: GPU-Accelerated Quantum and Cellular Simulation Engine},
  author    = {Aziz, Waheed},
  year      = {2024},
  url       = {https://github.com/wazi893/TileUniverse},
  note      = {2.5T amplitude ops/sec on consumer GPU hardware}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please read the codebase conventions in `.claude/CLAUDE.md` before submitting PRs.

---

<p align="center">
  <sub>Built for researchers who need simulation throughput, not simulation overhead.</sub>
</p>
