<p align="center">
  <img src="python/assets/logo.svg" alt="TileUniverse" width="400">
</p>

<h1 align="center">TileUniverse</h1>

<p align="center">
  <strong>A logic fabric where a CPU, a C compiler, and a learned placement AI all run on simulated gates — verified bit-exact.</strong>
</p>

<p align="center">
  <a href="#what-is-this">What it is</a> •
  <a href="#tileuniverse-in-numbers">Numbers</a> •
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

<p align="center"><img src="python/assets/demo_tilecpu.gif" alt="TileUniverse — C source compiled and running on a CPU built from logic gates" width="720"></p>

## What is this?

TileUniverse is a from-scratch, GPU-accelerated, **multi-backend** simulation engine in Rust, engineered so every execution backend agrees bit-for-bit. Three things make it worth a look:

- ⚡ **Many accelerator backends, one verified result** — CPU scalar/AVX, Cranelift JIT, CUDA FP32, and Tensor-Core paths, all held to **1e-6 cross-backend parity** against a scalar reference. A register-resident packed kernel reaches **114.5 trillion tile-evals/sec on an RTX 5090 — 573× over the repo's u64-per-tile baseline**.
- 🖥️ **A CPU built out of simulated logic gates** — plus a C-like compiler that targets it. Programs run **bit-identical** on a software reference and the physical tile fabric, across **3,000+ differential tests**. (A Brainfuck interpreter runs on it.)
- 🤖 **A learned circuit-placement AI** (AlphaChip-style) that **generalizes to unseen circuits one-shot**, with every layout proven correct by a hardware oracle.

The through-line: **if it computes, it computes in real tiles — and it's verified.** Correctness is a hard gate (golden-hash regression + reference-vs-fabric differential), not an afterthought.

---

## TileUniverse in Numbers

| Metric | Value |
|---------|---|
| Quantum Simulation Throughput | **15.56 TCOPS** on RTX 5090 (PureMMA, live 2026-08-25) |
| Cellular Logic | **114.5T tiles/sec** packed 1-bit eval (573× over repo u64-per-tile baseline) |
| Cross-Backend Parity | **ε ≈ 1e-6** across scalar / AVX / JIT / CUDA / Tensor-Core |
| Sparse Quantum States | O(1) memory for *structured* states — GHZ stored as 2 amplitudes regardless of qubit count |
| TileCpuV2 Physical Authority | Island-mode MUL computes in tiles (`with_mul_island()`); default mode keeps software MUL for goldens |
| Tests Passed | **3,175** with zero library warnings |
| Packed Eval | 573× speedup over repo u64-per-tile baseline via 64-tile-wide bit-packing |
| Sparse Eval | O(n) memory for W-states — **billions** of stable qubits on consumer GPU |
| TileCpuV2 ISA | 32 opcodes, 16 registers, 128 ROM entries, 128 RAM cells |
| Synth Pipeline | AIG → NPN4 → placement → routing, fully automated |

---

## Performance

Verified benchmarks (packed + PureMMA live-confirmed 2026-08-25 on RTX 5090; 2026-07-03 under-ran at 97.07T / 13.13 TCOPS — see `benchmarks/results/FELLOWSHIP_BENCH_CONFIRMATION_2026-07-03.md`):

### RTX 5090 (32GB, Blackwell)

| Substrate | Metric | Value |
|-----------|--------|-------|
| Quantum | Peak Throughput | **15.56 TCOPS** (PureMMA, 24 qubits) |
| Quantum | 32-qubit Simulation | **14.94 TCOPS** (full 32GB VRAM) |
| Cellular | Packed 1-bit Eval | **114.5T tiles/sec** (register-v3, 32K x 32K) |

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

### A CPU & Compiler, in Tiles

The flagship: a working processor whose datapath *is* simulated logic.

- A **32-bit CPU** whose fetch / decode / execute / ALU / write-back run as **physical tile circuits**, not an interpreter shortcut.
- A **C-like compiler toolchain** (lexer → recursive-descent parser → register allocation → codegen) that targets its ISA.
- Source compiles and runs **bit-identical** on a software ISA reference *and* the physical fabric — a **Brainfuck interpreter**, recursion, and arrays all execute on the simulated CPU, enforced by golden-hash differential testing.

### Learned Physical Design (AlphaFabric)

An AlphaChip-style learned placement system lays out logic circuits on the fabric:

- A policy trained on small circuits **generalizes one-shot to unseen circuit widths** at ~60% of the naive baseline's wirelength — zero per-instance search.
- The deterministic **simulated-annealing baseline** reduces routed wirelength by 21–45% versus row-major placement.
- **Correctness oracle**: a layout is accepted only after placed-and-routed execution matches the AIG reference on real simulated tiles — exhaustively through 12 inputs and with a deterministic 1,024-vector check above that.

### Scheduling Safety Frontier

The `sched_eval` example turns the same verification discipline into a workload scheduler:
real compiled tile-CPU jobs are placed onto racks, adversary/victim overlap is scored by the
`control_eval` covert-channel oracle, and a deterministic optimizer traces the throughput-vs-safety
frontier. The presentation snapshot is in
[`benchmarks/results/SCHED_OPTIMIZER_FRONTIER.md`](benchmarks/results/SCHED_OPTIMIZER_FRONTIER.md):
`balanced_backfill` reaches zero leakage at +0% makespan, while `victim_contention` measures a real
+1524-cycle safety price (+33.58%) when trusted backfill is unavailable.

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
cd TileUniverse

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
└── tests/                  # Integration tests
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
  note      = {15.56 TCOPS quantum and 114.5T packed tile-evals/sec on consumer GPU hardware}
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
