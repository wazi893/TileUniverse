# TileUniverse - Comprehensive System Documentation

**Version**: 0.5.0
**Last Updated**: January 2026
**Sprints Completed**: 71+
**Total Codebase**: 220+ Rust files, ~1.5MB core quantum code

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Systems](#core-systems)
4. [Quantum Computing Stack](#quantum-computing-stack)
5. [Advanced Modules](#advanced-modules)
6. [Hardware & Performance](#hardware--performance)
7. [Python Integration](#python-integration)
8. [Development Guide](#development-guide)
9. [Known Issues & Limitations](#known-issues--limitations)
10. [Roadmap & Future Work](#roadmap--future-work)

---

## Executive Summary

### What is TileUniverse?

TileUniverse is a **high-performance quantum and cellular simulation engine** written in Rust with Python bindings. It provides a complete stack for:

- **Quantum Simulation**: Up to 30 qubits dense, billion-qubit sparse states
- **Quantum Algorithms**: Grover, Shor, VQE, QAOA, Deutsch-Jozsa
- **Quantum Error Correction**: Steane, Surface codes, Union-Find decoder
- **Hardware Transpilation**: IBM/IonQ native gates, OpenQASM 3.0 support
- **Resource Estimation**: Fault-tolerant QC resource planning
- **Quantum-SNN Hybrid**: Neuromorphic computing with quantum interference
- **QRAM**: Fault-tolerant quantum memory with magic state budgeting
- **Cellular Automata**: Parallel world simulation (40B evals/sec)

### Performance Highlights

| Metric | Value | Hardware |
|--------|-------|----------|
| **Quantum Substrate** | 15.8T amplitude ops/sec | RTX 5090 PureMMA |
| **Quantum Substrate** | 2.5T amplitude ops/sec | RTX 4070 WMMA |
| **Cellular Substrate** | 200B logic evals/sec | RTX 5090 |
| **Cellular Substrate** | 40B logic evals/sec | RTX 4070 |
| **1-Bit Packed Tiles** | 27.5T tile evals/sec | RTX 5090 (138× vs baseline) |
| **GPU Speedup** | 46.8× over CPU | RTX 4070 |
| **Tensor Core Speedup** | 15.1× over FP32 | FP16 WMMA |
| **FP8 Speedup** | 26× over baseline | Hopper+ GPUs |
| **Cross-Block CNOT** | 210M block-ops/sec | GPU sparse quantum |
| **SNN Scale** | 200K neurons | CUDA acceleration |
| **Sparse States** | 2^64 qubit GHZ | O(1) MinimalGhzState |
| **Ising Mode GPU** | 1.2B updates/sec | 29.2× speedup |

### Key Differentiators

1. **Cross-Backend Parity**: Deterministic results across Scalar, AVX2, AVX512, JIT, CUDA (ε ≈ 1e-6)
2. **Fault-Tolerant Ready**: Full FT processor with Pauli frame tracking, magic state factories, gate scheduling
3. **Error Mitigation**: ZNE rescue with adaptive lambda, REM, combined pipeline with 94% error reduction
4. **Hardware Interoperability**: OpenQASM 3.0, native IBM/IonQ transpilation, error-aware compilation
5. **Production-Grade**: Edition 2024 Rust, comprehensive test suite, 71+ sprints
6. **Hybrid Architecture**: Quantum + classical + neuromorphic + probabilistic computing
7. **Physics-Logic Coupling**: Bidirectional heat/charge/power affecting tile computation
8. **Billion-Qubit Scale**: O(1) GHZ states, billion-qubit W states, cross-block GPU entanglement

---

## Architecture Overview

### Repository Structure

```
TileUniverse/
├── src/                          # Rust core engine (201 files)
│   ├── lib.rs                    # Module declarations, public API
│   ├── quantum.rs                # Quantum primitives (QState, QGate)
│   ├── fusion.rs                 # Gate fusion & optimization (331KB)
│   ├── cuda.rs                   # GPU backend (315KB, FP32/WMMA/FP8)
│   ├── algebraic_fusion.rs       # Algebraic optimization (310KB)
│   ├── simulation.rs             # Classical logic simulation
│   ├── universe.rs               # High-level Universe API
│   ├── config.rs                 # JSON/YAML config loading
│   ├── metrics.rs                # COPS metrics tracking
│   │
│   ├── algorithms/               # Quantum algorithms
│   │   ├── grover.rs             # Grover's search
│   │   ├── deutsch_jozsa.rs      # DJ, BV algorithms
│   │   ├── shor.rs               # Shor's factoring
│   │   ├── vqe/                  # VQE (ansatz, optimizer)
│   │   └── qaoa/                 # QAOA (circuit, maxcut)
│   │
│   ├── qec/                      # Quantum Error Correction (15+ files)
│   │   ├── stabilizer.rs         # Aaronson-Gottesman tableau
│   │   ├── codes.rs              # Steane, Surface codes
│   │   ├── decoder.rs            # Syndrome decoding
│   │   ├── union_find.rs         # Union-Find decoder
│   │   ├── union_find_dn.rs      # Delfosse-Nickerson weighted
│   │   ├── noise.rs              # Noise models
│   │   ├── ft_processor.rs       # FaultTolerantProcessor (Sprint 71)
│   │   ├── ft_types.rs           # LogicalQubit, KnownBasis, PauliFrame
│   │   ├── gate_decomp.rs        # Toffoli→7T, CCZ expansion
│   │   ├── injection.rs          # Abstract/Sampled injection models
│   │   ├── magic_supply.rs       # FactoryFleet orchestration
│   │   ├── qec_model.rs          # Surface code error scaling
│   │   ├── scheduler.rs          # Gate-to-cycle mapping
│   │   └── gpu_stabilizer.rs     # CUDA batch_rowmult, O(n²) memory
│   │
│   ├── qram/                     # Quantum RAM (25+ files)
│   │   ├── bucket_brigade.rs     # Binary tree architecture
│   │   ├── ft_qram.rs            # Fault-tolerant QRAM
│   │   ├── polynomial.rs         # Polynomial encoding
│   │   ├── sparse_bucket.rs      # Sparse bucket-brigade
│   │   ├── stabilizer_qram.rs    # Gottesman-Knill QRAM
│   │   ├── magic_budget.rs       # Magic state budgeting
│   │   ├── distillation.rs       # State distillation factories
│   │   ├── packed_frame.rs       # PackedPauliFrame64 ensemble tracking
│   │   ├── factory_scheduler.rs  # T-gate dependency analysis
│   │   ├── factory_pool.rs       # Multi-factory orchestration
│   │   ├── schedule_optimizer.rs # Greedy, list, simulated annealing
│   │   └── throughput_model.rs   # Auto-config minimize latency/qubits
│   │
│   ├── snn/                      # Spiking Neural Networks (20+ files)
│   │   ├── neuron.rs             # Leaky Integrate-and-Fire
│   │   ├── synapse.rs            # Weighted connections
│   │   ├── stdp.rs               # STDP & R-STDP learning
│   │   ├── network.rs            # SNNNetwork builder
│   │   ├── quantum_hybrid.rs     # QuantumSNN, InterferenceMode
│   │   ├── curiosity.rs          # Intrinsic motivation
│   │   ├── value_function.rs     # TD credit assignment
│   │   ├── population_coding.rs  # Population encoding
│   │   ├── gpu_fused.rs          # Zero-PCIe fused kernels (EPIC 97)
│   │   ├── block_sparse_synapses.rs # 2:4/4:8 structured sparsity
│   │   ├── stabilizer_neuron.rs  # Clifford-based quantum neurons
│   │   ├── stabilizer_network.rs # StabilizerNetwork (87× speedup)
│   │   ├── stabilizer_hybrid.rs  # HybridQuantumNetwork
│   │   └── gpu_stabilizer_network.rs # 7.98× GPU speedup at 100K
│   │
│   ├── transpile/                # Hardware transpilation (18+ files)
│   │   ├── ibm.rs                # IBM native gates (SX, Rz, CX)
│   │   ├── ionq.rs               # IonQ native gates (GPI, MS)
│   │   ├── native_gates.rs       # Native circuit representations
│   │   ├── optimizer.rs          # Optimization pipeline
│   │   ├── t_count.rs            # T-gate analysis
│   │   ├── t_depth.rs            # T-depth minimization
│   │   ├── qasm3/                # OpenQASM 3.0 support
│   │   ├── resource_estimate.rs  # Physical resource estimation
│   │   ├── analysis.rs           # Circuit analysis
│   │   ├── error_analysis.rs     # Error propagation through circuits
│   │   ├── circuit_rewrite.rs    # T-depth minimization reordering
│   │   ├── adaptive_protocols.rs # Per-T-gate distillation selection
│   │   ├── error_aware_compiler.rs # Multi-objective optimization
│   │   ├── multi_objective.rs    # Pareto frontier optimization
│   │   ├── topology_aware_estimate.rs # SWAP overhead accounting
│   │   └── dynamic_estimate.rs   # VQE 4× speedup with factories
│   │
│   ├── hardware/                 # Hardware mapping (10+ files)
│   │   ├── profile.rs            # IBM Heron, IonQ Aria profiles
│   │   ├── topology.rs           # Device connectivity graphs
│   │   ├── swap_router.rs        # SWAP routing
│   │   ├── factory_layout.rs     # Distillation factory placement
│   │   ├── aware_budget.rs       # Hardware-adjusted estimates
│   │   ├── routing.rs            # Dijkstra/A* shortest path
│   │   ├── placement.rs          # Greedy/SA/GA factory placement
│   │   ├── layout_viz.rs         # ASCII congestion heatmaps
│   │   └── congestion.rs         # SWAP chain congestion analysis
│   │
│   ├── tile8/                    # TILE-8 CPU toolchain (25+ files)
│   │   ├── asm.rs                # Assembler (text → binary)
│   │   ├── cpu.rs                # Tile8Cpu builder
│   │   ├── isa.rs                # 16-instruction ISA
│   │   ├── sparse_quantum.rs     # SparseQuantumGrid, GHZ
│   │   ├── sparse_quantum_vec.rs # Vec-based W states + MinimalGhzState
│   │   ├── sparse_quantum_gpu.rs # Cross-block CNOT GPU kernels
│   │   ├── sparse_quantum_bigint.rs # BigInt verify_ghz_fast
│   │   ├── sparse_quantum_hybrid.rs # Hybrid dispatch
│   │   ├── grover.rs             # Grover on TILE-8
│   │   ├── hybrid_search.rs      # Hybrid fitness-Grover (3.65× amplify)
│   │   └── ising_mode_gpu.rs     # GpuIsingGrid (29.2× speedup)
│   │
│   ├── distributed/              # Parallel/distributed execution
│   │   ├── tile_farm.rs          # Multi-core tile farm
│   │   ├── distributed_farm.rs   # Multi-node distributed
│   │   └── quantum_worker_pool.rs # Block-sparse workers
│   │
│   ├── physics/                  # Physics simulation (10 files)
│   │   └── logic_coupling.rs     # Physics-to-logic coupling (heat/charge/power)
│   ├── pbit/                     # Probabilistic computing (6 files)
│   │   ├── pbit.rs               # Single p-bit with sigmoid activation
│   │   ├── network.rs            # Coupled p-bit array
│   │   ├── ising.rs              # Problem encoding (MaxCut, QUBO, SK)
│   │   ├── sampler.rs            # Gibbs, annealing, parallel tempering
│   │   └── gpu.rs                # CUDA 15-26× speedup
│   ├── tensor_network/           # Tensor network contraction (11 files)
│   │   ├── tensor.rs             # Core tensor with explicit indices
│   │   ├── contraction.rs        # Matrix-based 10-100× faster
│   │   ├── path_optimizer.rs     # Beam search optimization
│   │   └── slicing.rs            # Memory-bounded slicing
│   ├── search/                   # Quantum-enhanced neural search (12 files)
│   │   ├── candidate.rs          # Xorshift64 RNG encoding
│   │   ├── fitness.rs            # OneMax, NK-Landscape, MaxSAT
│   │   ├── substrate.rs          # 1M-268M parallel evaluation
│   │   ├── selection.rs          # Tournament, Roulette, Rank, SUS
│   │   ├── snn_encoding.rs       # LIF neuron spike encoding
│   │   ├── hive_mesh.rs          # Torus topology communication
│   │   └── grover_amplify.rs     # O(√N) quantum speedup
│   ├── mitigation/               # Error mitigation (Sprint 68-71)
│   │   ├── zne.rs                # Zero-noise extrapolation with rescue
│   │   └── mod.rs                # REM, combined pipeline
│   ├── organisms_mod/            # Evolutionary organisms (4 files)
│   ├── brain/                    # Neural networks (10 files)
│   ├── circuits_mod/             # Circuit patterns (4 files)
│   ├── gui/                      # GUI backend (4 files)
│   └── experiments/              # Research experiments (11 files)
│
├── crates/logic-fabric-core/     # Performance-critical core (14 files)
│   ├── src/
│   │   ├── quantum.rs            # Core quantum (96KB)
│   │   ├── fusion.rs             # Gate fusion (331KB)
│   │   ├── cuda.rs               # GPU kernels (315KB)
│   │   ├── algebraic_fusion.rs   # Algebraic optimization (310KB)
│   │   ├── block_sparse_state.rs # Block-sparse states
│   │   ├── sparse_state.rs       # Element-sparse states
│   │   ├── hybrid_state.rs       # Hybrid dense/sparse
│   │   ├── qasm/                 # OpenQASM 2.0
│   │   ├── commutation.rs        # Gate commutation rules
│   │   ├── fixed_point.rs        # Fixed-point arithmetic
│   │   └── hardware.rs           # Hardware abstraction
│
├── python/                       # Python bindings (PyO3 + maturin)
│   ├── Cargo.toml                # PyO3 cdylib config
│   ├── src/lib.rs                # Rust FFI glue
│   ├── tileuniverse/             # Python package
│   │   ├── __init__.py           # Main API (v0.4.0)
│   │   ├── algorithms.py         # Algorithm wrappers
│   │   └── rl/                   # Gymnasium RL integration
│   │       ├── quantum_agent.py  # QuantumSNNAgent
│   │       ├── vec_env.py        # Vectorized environments
│   │       └── sb3_wrapper.py    # Stable-Baselines3 wrapper
│   └── tests/                    # Python test suite
│
├── tests/                        # Rust integration tests (70+ files)
│   ├── compare3_parity.rs        # Cross-backend parity validation
│   ├── grover_integration.rs     # Grover end-to-end tests
│   ├── dj_bv_integration.rs      # DJ/BV tests
│   ├── vqe_integration.rs        # VQE tests (23 tests)
│   ├── qaoa_integration.rs       # QAOA tests (16 tests)
│   ├── cli_*.rs                  # CLI command tests
│   ├── coupling_gpu_parity.rs    # Physics coupling GPU parity
│   ├── physics_coupling_integration.rs # Physics coupling tests
│   ├── mitigation_validation.rs  # ZNE/REM validation
│   ├── packed_frame_validation.rs # PackedPauliFrame64 tests
│   └── hybrid_search_gate.rs     # Hybrid Grover tests
│
├── examples/                     # Benchmarks & demos (100+ examples)
│   ├── showcase_cops.rs          # COPS demonstration
│   ├── comprehensive_bench.rs    # Full backend comparison
│   ├── vqe_h2_molecule.rs        # VQE molecular ground state
│   ├── qaoa_maxcut.rs            # QAOA MaxCut
│   ├── epic114_fp8_bench.rs      # FP8 Tensor Core benchmark
│   ├── w_state_gpu_max.rs        # Billion-qubit W state
│   ├── cross_block_cnot_gpu_bench.rs # Cross-block GPU CNOT
│   ├── sparse_quantum_gpu_bench.rs # GPU sparse quantum
│   ├── sprint_71_demo.rs         # ZNE/mitigation demo
│   ├── segmentation_visualizer.rs # TileAnneal visualizer
│   └── gpu_neural_bench.rs       # Stabilizer network GPU bench
│
├── User notes/                   # Sprint documentation
│   ├── SPRINTS/                  # Sprints 0.6 through 57.0
│   └── Project Glossary/         # EPICs, technical terms
│
├── GUI/                          # GUI recording/playback data
├── Cargo.toml                    # Main workspace config
└── CLAUDE.md                     # AI assistant guide

**Total**: 201 Rust source files, comprehensive test coverage, production-ready build system
```

### Design Principles

#### 1. Determinism First
- Classical logic runs in stable order every tick
- Measurement/collapse **always** uses scalar backend (never SIMD/JIT/GPU)
- No hidden nondeterminism, no multithreading in critical paths
- Reproducible results with seed control

#### 2. Cross-Backend Parity
- **Scalar backend** is the authoritative reference implementation
- AVX2, AVX512, JIT, and GPU must match scalar within ε ≈ 1e-6
- Comprehensive parity tests in `tests/compare3_parity.rs`
- Verified across all gate types and circuit depths

#### 3. One Gate Per Tile Per Tick
- Core evolution applies single quantum gate per tile each tick
- Enables consistent performance measurement
- Depth batching at fusion level, not execution level
- Predictable resource consumption

#### 4. Modular Architecture
- Core quantum primitives isolated in `logic-fabric-core` crate
- Algorithm implementations separate from execution backends
- Clean separation: simulation ↔ quantum ↔ hardware ↔ transpile
- Feature flags for optional dependencies (CUDA, JIT, etc.)

---

## Core Systems

### 1. Quantum State Representation

#### Dense State (`QState` in `quantum.rs`)

**Structure-of-Arrays (SoA) Layout:**
```rust
pub struct QState {
    pub n_qubits: u32,
    pub real: AlignedVecF32,  // 64-byte aligned
    pub imag: AlignedVecF32,  // 64-byte aligned
    // Guard words for buffer overrun detection (EPIC 62Q)
}
```

**Characteristics:**
- **Capacity**: Up to 30 qubits (1 billion amplitudes, ~8GB memory)
- **Alignment**: 64-byte for cache friendliness and SIMD operations
- **Guard Words**: Canary-based detection of buffer overruns (0xDEADBEEF pattern)
- **Backend Support**: All backends (Scalar, AVX2, AVX512, JIT, CUDA)

#### Sparse States

**1. Element-Sparse (`SparseQState` - EPIC 85):**
```rust
pub struct SparseQState {
    n_qubits: u32,
    amplitudes: HashMap<u64, Complex32>,  // index → amplitude
}
```
- **Capacity**: Up to 64 qubits
- **Use Case**: States with few non-zero amplitudes (< 10% occupancy)
- **Performance**: O(nnz) for operations where nnz = non-zero count

**2. Block-Sparse (`BlockSparseState` - EPIC 88):**
```rust
pub struct BlockSparseState {
    n_qubits: u32,
    blocks: Vec<Block128>,  // 128 complex amplitudes per block
}
```
- **Block Size**: 128 amplitudes = 1 WMMA tile (16×16 matrix for 2×2 gate)
- **GPU Optimization**: Direct WMMA execution on RTX 5090 (32+ blocks profitable)
- **Qubit Organization**:
  - Qubits 0-6: Operations within block (stride 1-64)
  - Qubits 7+: Gather/scatter operations across blocks
- **Use Case**: Medium-sparse states, GPU-accelerated execution

**3. Hybrid Dense/Sparse (`HybridState`):**
```rust
pub struct HybridState {
    dense_part: QState,      // Frequently accessed amplitudes
    sparse_part: SparseQState,  // Rarely accessed amplitudes
}
```
- **Adaptive**: Switches between dense/sparse based on occupancy
- **Use Case**: States with varying sparsity during evolution

**4. Fast Sparse States (TILE-8 module):**
```rust
// O(1) sparse GHZ - only 2 amplitudes regardless of n
pub struct FastGHZState {
    n_qubits: usize,
    // |00...0⟩ + |11...1⟩ represented implicitly
}

// Vec-based W state - billion-qubit capable
pub struct FastWState {
    n_qubits: usize,
    amplitudes: Vec<Complex32>,  // n+1 amplitudes
}
```
- **GHZ**: O(1) memory, instant creation for any qubit count
- **W**: O(n) memory, verified for 1 billion qubits
- **Use Case**: Specific entangled states for QEC/quantum communication

### 2. Gate Set

#### Supported Gates (23 Variants in `QGate` Enum)

**Single-Qubit Clifford:**
- `H(q)` - Hadamard: |0⟩ ↔ (|0⟩+|1⟩)/√2
- `X(q)` - Pauli-X (bit flip): |0⟩ ↔ |1⟩
- `Y(q)` - Pauli-Y: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
- `Z(q)` - Pauli-Z (phase flip): |1⟩ → -|1⟩
- `S(q)` - Phase gate: |1⟩ → i|1⟩
- `Sdg(q)` - S-dagger: |1⟩ → -i|1⟩

**Single-Qubit Non-Clifford:**
- `T(q)` - π/8 gate: |1⟩ → e^(iπ/4)|1⟩ (SPRINT 48.0)
- `Tdg(q)` - T-dagger: |1⟩ → e^(-iπ/4)|1⟩

**Single-Qubit Rotation:**
- `Phase(q, θ)` - Phase rotation: |1⟩ → e^(iθ)|1⟩
- `Rx(q, θ)` - X-axis rotation (EPIC 83)
- `Ry(q, θ)` - Y-axis rotation
- `Rz(q, θ)` - Z-axis rotation
- `U3(q, θ, φ, λ)` - Universal single-qubit gate (SPRINT 32.0)

**Two-Qubit Gates:**
- `CNot(c, t)` - Controlled-NOT (fundamental entangling gate)
- `CZ(c, t)` - Controlled-Z (SPRINT 2.1a, symmetric)
- `Swap(a, b)` - SWAP (exchanges qubit states)
- `CPhase(c, t, θ)` - Controlled phase rotation
- `CRz(c, t, θ)` - Controlled Rz rotation

**Three-Qubit Gates:**
- `Toffoli(c1, c2, t)` - CCX (Controlled-Controlled-NOT)
- `CCZ(c1, c2, t)` - Controlled-Controlled-Z

**Measurement:**
- `Measure(q)` - Projective measurement (collapses to |0⟩ or |1⟩)

**Batching & Fusion:**
- `BatchedGate(gate, count)` - Depth batching (N identical gates)
- `FusedLayer(gates)` - Parallel gates on different qubits
- `MegaGate(qubit, matrix)` - Arbitrary single-qubit 2×2 matrix (EPIC 84)

#### Gate Implementation Backends

| Gate | Scalar | AVX2 | AVX512 | JIT | CUDA FP32 | WMMA FP16 | WMMA FP8 |
|------|--------|------|--------|-----|-----------|-----------|----------|
| H, X, Y, Z | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rx, Ry, Rz | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CNot | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CZ, Swap | ✅ | ✅ | - | ✅ | ✅ | ✅ | - |
| T, Tdg | ✅ | ✅ | - | ✅ | ✅ | ✅ | - |
| U3 | ✅ | ✅ | - | ✅ | ✅ | ✅ | - |
| Toffoli | ✅ | - | - | - | ✅ | - | - |
| MegaGate | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |

### 3. Gate Fusion & Optimization

#### Fusion Engine (`fusion.rs` - 331KB)

**Phase 1: Identity Elimination**
```
H·H = I  →  (removed)
X·X = I  →  (removed)
Z·Z = I  →  (removed)
CZ·CZ = I  →  (removed)
```

**Phase 2: Algebraic Optimization (EPIC 83)**

Self-Inverse Gates:
```
H^N = { I  if N even
      { H  if N odd
```

Rotation Gate Spectral Decomposition:
```
Rx(θ) = cos(θ/2)·I + i·sin(θ/2)·X
Rx(θ)^N = Rx(N×θ)  (angle accumulation)
Rx(2πk) = I  (periodicity, k ∈ ℤ)
```

**Phase 3: Mega-Fusion (EPIC 84)**

Compose arbitrary single-qubit gate sequences:
```
Circuit: H → X → Z → Rx(θ)
Mega:    MegaGate(q, M)  where M = Rx(θ)·Z·X·H
```

**Benefits:**
- Reduces N gates → 1 operation
- Effective throughput: N × amplitudes × states
- Critical for VQE/QAOA circuits with varied gate types
- Demonstrated 26+ PCOPS effective throughput

**Phase 4: Depth Batching**
```
Circuit: Rx(θ)·Rx(θ)·Rx(θ)·Rx(θ)·Rx(θ)
Batched: BatchedGate(Rx(θ), 5)
```
- Single kernel launch for N identical gates
- Reduces overhead, improves GPU utilization

**Phase 5: Layer Fusion**
```
Circuit: H(0), H(1), H(2), H(3)
Fused:   FusedLayer([H(0), H(1), H(2), H(3)])
```
- Parallel execution on independent qubits
- Exploits SIMD/GPU parallelism

#### Backend Selection Heuristics

**Rule-Based Selection (EPIC 87):**

| Circuit Properties | Selected Backend | Rationale |
|--------------------|------------------|-----------|
| n < 6 qubits | CPU (JIT or AVX2) | Small state, CPU overhead acceptable |
| 6 ≤ n ≤ 10, depth < 100 | CPU JIT | JIT compilation profitable for short circuits |
| 6 ≤ n ≤ 10, depth ≥ 100 | GPU FP32 Batched | Amortize GPU transfer cost |
| n > 10 | GPU WMMA (FP16/FP8) | Exponential state size, GPU essential |
| Sparse state | Element-sparse or Block-sparse | Adaptive based on occupancy |

**Per-Qubit GPU Speedup Model:**
- Qubits 0-3: **1.5-1.8×** (direct WMMA on matrix elements)
- Qubits 4-7: **3-4×** (gather/scatter + WMMA)
- Qubits 8+: **10-50×** (large state, GPU dominates)

**Profitability Threshold:**
- Min batch size: **20 operations** for GPU transfer overhead
- Block-sparse: **32+ blocks** profitable on RTX 5090

### 4. CUDA Backend

#### Architecture (`cuda.rs` - 315KB)

**CUDA Toolkit Integration:**
- Auto-detection of CUDA versions: v13.1, v13.0, v12.6, v12.0
- Dynamic PTX compilation for target architecture
- Supported compute capabilities: 89 (Ada Lovelace), 90 (Hopper), 100 (Blackwell)
- cudarc-based Rust-native bindings (no nvcc at build time)

**Kernel Variants:**

**1. FP32 Batched Kernels:**
```cuda
// Standard precision, baseline performance
__global__ void apply_h_kernel_fp32(float* real, float* imag, int n, int qubit)
```
- Performance: 821M ops/sec (RTX 4070)
- Use case: Baseline, compatibility, validation

**2. WMMA FP16 Tensor Cores (EPIC 67):**
```cuda
// Matrix operations using Tensor Cores
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, matrix_a, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```
- Performance: **15.1× speedup** over FP32
- Throughput: 12.4 PCOPS (trillion ops/sec)
- Use case: Production workloads, standard precision

**3. WMMA FP8 Tensor Cores (EPIC 114):**
```cuda
// Ultra-high throughput with FP8
wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_fp8_e4m3, wmma::row_major> a_frag;
// ILP (Instruction-Level Parallelism) variant
```
- Performance: **26× speedup** over FP32 baseline
- Throughput: 21.3 PCOPS
- Requirements: Hopper+ GPUs (compute_90, compute_100)
- Use case: Maximum throughput, research workloads

**4. ILP (Instruction-Level Parallelism) Kernels:**
```cuda
// Unrolled loops for memory-bound operations
#pragma unroll 4
for (int i = 0; i < batch_size; i += 4) { ... }
```
- Reduces memory latency impact
- Variants: RenormILP (renormalization-aware)

**CUDA Features:**
- **CUDA Graphs** (EPIC 72): Reduced kernel launch overhead
- **Pinned Memory**: Fast host-device transfers
- **Async Streams**: Overlap computation and communication
- **Device Selection**: Multi-GPU support (select via device ID)

**Runtime Management:**
```rust
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    state_buf: CudaSlice<f32>,
    kernel_cache: HashMap<String, CudaFunction>,
}
```

---

## Quantum Computing Stack

### 1. Quantum Algorithms (`algorithms/`)

#### Grover's Search Algorithm

**Purpose**: Quadratic speedup for unstructured search
**Complexity**: O(√N) vs O(N) classical

**Implementation** (`grover.rs`):
```rust
pub fn run_grover(
    n_qubits: u32,
    target: u32,
    backend: BackendMode,
    seed: Option<u64>
) -> GroverResult
```

**Components:**
1. **Superposition**: Apply H to all qubits → uniform superposition
2. **Oracle**: Mark target state with phase flip
3. **Diffusion**: Amplitude amplification (Grover operator)
4. **Iterate**: Repeat oracle + diffusion for optimal_iterations = ⌊π/4 × √(2^n)⌋
5. **Measure**: Collapse to target state with high probability

**Performance:**
- **2-3 qubits**: 90-95% success rate ✅
- **4+ qubits**: Reduced success rate due to approximate MCZ decomposition ⚠️
- See [Known Issues](#known-issues--limitations) for details

**Usage:**
```rust
let result = run_grover(3, 5, BackendMode::Cuda, Some(42));
println!("Found: {} (success: {})", result.measured, result.success);
```

#### Deutsch-Jozsa & Bernstein-Vazirani

**Purpose**: Determine oracle properties with single query (exponential speedup)

**Implementation** (`deutsch_jozsa.rs`):
```rust
pub fn run_deutsch_jozsa(n_qubits: u32, oracle_type: OracleType) -> DJResult
pub fn run_bernstein_vazirani(n_qubits: u32, secret: u32) -> BVResult
```

**Deutsch-Jozsa:**
- Oracle types: Constant (always 0 or 1), Balanced (half 0s, half 1s)
- Result: Single measurement distinguishes constant vs balanced
- Classical: 2^(n-1) + 1 queries required, Quantum: 1 query

**Bernstein-Vazirani:**
- Hidden bitstring s: Oracle computes f(x) = s·x (mod 2)
- Result: Single measurement reveals entire bitstring s
- Classical: n queries required, Quantum: 1 query

**Status**: Works correctly for all sizes (max 3-qubit MCZ limitation) ✅

#### Shor's Factoring Algorithm

**Purpose**: Integer factorization in polynomial time (exponential speedup)
**Complexity**: O((log N)³) quantum vs O(exp((log N)^(1/3))) classical (sub-exponential)

**Implementation** (`shor.rs`):
```rust
pub fn factor_with_shor(N: u64, backend: BackendMode) -> Option<(u64, u64)>
```

**Components:**
1. **Classical Pre-processing**: GCD checks, power detection
2. **Quantum Period Finding**: QFT-based order finding
3. **Modular Exponentiation**: Controlled-U^(2^k) mod N gates
4. **Inverse QFT**: Extract period from phase
5. **Classical Post-processing**: Convert period to factors

**Quantum Subroutines:**
```rust
pub fn qft_circuit(n_qubits: u32) -> Vec<QGate>
pub fn inverse_qft_circuit(n_qubits: u32) -> Vec<QGate>
pub fn modular_exp_circuit(a: u64, N: u64, n_qubits: u32) -> Vec<QGate>
```

**Limitations:**
- Requires many qubits (2 × ceil(log₂ N))
- Deep circuits (QFT depth O(n²))
- Currently demo-scale (< 20 qubits)

#### VQE (Variational Quantum Eigensolver) - SPRINT 40.0

**Purpose**: Find ground state energy of molecules (hybrid quantum-classical)
**Application**: Drug discovery, materials science, quantum chemistry

**Implementation** (`vqe/`):
```rust
pub fn run_vqe(hamiltonian: Hamiltonian, config: VQEConfig) -> VQEResult
pub fn h2_ground_state(bond_length: f64) -> VQEResult  // Convenience for H₂
```

**Ansatz Types:**

**1. Hardware-Efficient Ansatz** (Recommended for NISQ):
```
Layer: Ry(θ₁) - Ry(θ₂) - CNOT(0,1) - CNOT(1,0)
Repeat L layers with different parameters
```
- Parameters: n_qubits × n_layers
- Circuit depth: O(n_layers)
- Status: ✅ Works correctly, widely used in practice

**2. UCCSD Ansatz** (Simplified):
```
Single excitations: Ry(θ) - CNOT - CNOT - Ry(-θ)
Double excitations: (simplified, 5 gates instead of 16)
```
- Parameters: n_single + n_double excitations
- Status: ⚠️ Simplified (see [Known Issues](#known-issues--limitations))
- Use case: Research, proof-of-concept

**Optimizers:**

| Optimizer | Method | Best For | Convergence |
|-----------|--------|----------|-------------|
| **Nelder-Mead** | Simplex search | Default, robust | Slow but reliable |
| **Gradient Descent** | ∇E with finite differences | Smooth landscapes | Fast if initialized well |
| **SPSA** | Stochastic perturbation | Noisy objectives | Robust to noise |

**Molecular Hamiltonians:**
```rust
use engine::hamiltonians::Hamiltonian;

let h2 = Hamiltonian::h2_molecule(0.735);  // Bond length in Ångströms
// Returns: 4-qubit Hamiltonian with Pauli operators
```

**Example:**
```rust
use engine::algorithms::vqe::{run_vqe, VQEConfig, h2_ground_state};

// H₂ molecule at equilibrium
let result = h2_ground_state(0.735);
println!("Energy: {} Ha", result.energy);  // Expected: ~-1.14 Ha (FCI)

// Custom Hamiltonian with configuration
let config = VQEConfig::hardware_efficient(2)  // 2 layers
    .with_shots(5000)
    .with_seed(42)
    .with_optimizer(OptimizerType::NelderMead);
let result = run_vqe(hamiltonian, config);
```

**Performance:**
- 2-qubit Hamiltonians: Converges to machine precision (E = -1.0)
- H₂ molecule (4 qubits): ~0.72 Ha error with UCCSD (simplified), accurate with hardware-efficient
- Optimization: 50-200 iterations typical

#### QAOA (Quantum Approximate Optimization) - SPRINT 42.0

**Purpose**: Solve combinatorial optimization problems (approximate solutions)
**Application**: MaxCut, traveling salesman, portfolio optimization, logistics

**Implementation** (`qaoa/`):
```rust
pub fn run_qaoa(graph: Graph, config: QAOAConfig) -> QAOAResult
pub fn maxcut(graph: Graph) -> MaxCutResult  // Convenience for MaxCut problem
```

**Algorithm:**
1. **Problem Encoding**: QUBO → Ising Hamiltonian → Phase gates
2. **Ansatz Circuit**:
   ```
   |+⟩^⊗n → [Problem(γ) → Mixer(β)]^p → Measure
   ```
   - Problem layer: CZ gates encoding graph edges
   - Mixer layer: Rx gates on all qubits
3. **Classical Optimization**: Find (γ, β) parameters maximizing objective
4. **Sampling**: Measure circuit to get candidate solutions

**Graph Utilities:**
```rust
let cycle = Graph::cycle(4);          // Square graph (4 vertices, 4 edges)
let complete = Graph::complete(5);    // K₅ (5 vertices, 10 edges)
let path = Graph::path(10);           // Linear chain (10 vertices, 9 edges)
let random = Graph::random(8, 0.5, 42);  // 8 vertices, 50% edge density, seed 42
```

**Configuration:**
```rust
let config = QAOAConfig::with_depth(3)  // p=3 layers
    .with_shots(1000)
    .with_spsa()  // SPSA optimizer
    .with_seed(42);
```

**MaxCut Problem:**
```rust
use engine::algorithms::qaoa::{maxcut, Graph};

let graph = Graph::cycle(4);
let result = maxcut(graph);
println!("Best cut: {}", result.best_cut);  // e.g., 4 (maximum for cycle)
println!("Approximation ratio: {:?}", result.approximation_ratio);  // e.g., Some(1.0)
```

**Performance:**
- Small graphs (n ≤ 10): Finds optimal solution with p=2-3
- Medium graphs (n ≤ 20): Approximation ratio 0.8-0.95
- Optimization: 100-500 iterations

### 2. Quantum Error Correction (`qec/`)

#### Stabilizer Simulation

**Theory**: Aaronson-Gottesman algorithm for efficient Clifford simulation
**Complexity**: O(n²) memory, O(n²) per gate (vs O(2^n) for full state)

**Implementation** (`stabilizer.rs`):
```rust
pub struct StabilizerState {
    n_qubits: usize,
    tableau: Vec<bool>,  // (2n+1) × 2n tableau
}
```

**Supported Operations:**
- **Clifford Gates**: H, S, CNOT, CZ
- **Pauli Measurements**: X, Y, Z basis
- **State Preparation**: |0⟩, |+⟩, computational basis
- **Stabilizer Checks**: Verify stabilizer group membership

**Capabilities:**
- **Scale**: Millions of qubits (limited by memory, not exponential blowup)
- **Use Case**: QEC simulation, verification, decoder testing

#### Error Correction Codes

**1. Repetition Code** (Bit-flip or Phase-flip):
```
Encoding: |0⟩ → |000⟩, |1⟩ → |111⟩
Distance: 3 (corrects 1 error)
Syndrome: Measure Z₀Z₁ and Z₁Z₂
```

**2. Steane Code [[7,1,3]]**:
```
Encoding: |0⟩ → |0_L⟩, |1⟩ → |1_L⟩
Distance: 3 (corrects 1 X or Z error)
Stabilizers: 6 operators (3 X-type, 3 Z-type)
Logical operations: Transversal gates (fault-tolerant)
```

**Implementation** (`codes.rs`):
```rust
use engine::qec::SteaneCode;

let mut steane = SteaneCode::new();
steane.encode_zero();        // |0_L⟩
steane.apply_logical_x();    // |1_L⟩
steane.inject_error(3, 'X'); // Bit flip on qubit 3
let syndrome = steane.measure_syndrome();  // [0,0,1,0,1,1]
let corrected = steane.decode_and_correct(syndrome);  // Corrects error
```

**3. Surface Code**:
```
Encoding: 2D lattice of data and ancilla qubits
Distance: d (corrects ⌊(d-1)/2⌋ errors)
Stabilizers: Plaquette (Z-type) and vertex (X-type) checks
```

**Grid Sizes:**
- Distance 3: 9 data qubits, 8 ancilla (17 total)
- Distance 5: 25 data qubits, 24 ancilla (49 total)
- Distance d: d² data qubits, (d²-1) ancilla

#### Syndrome Decoding

**Minimum-Weight Perfect Matching (MWPM):**
- Find error chain with minimum weight matching syndrome
- Complexity: O(n³) with Blossom algorithm

**Union-Find Decoder** (`union_find.rs`):
```rust
pub struct UnionFindDecoder<'a> {
    surface_code: &'a SurfaceCode,
    union_find: UnionFind,
}

pub fn decode(&self, syndrome: &[u8]) -> Vec<(usize, usize)>
```

**Algorithm:**
1. Cluster syndrome defects using Union-Find data structure
2. Grow clusters until they merge
3. Decode correction from cluster structure

**Complexity**: O(n²) average case (vs O(n³) for MWPM)

**Delfosse-Nickerson Weighted Union-Find** (`union_find_dn.rs`):
- Weighted edges for biased noise (pₓ ≠ p_z)
- Improved performance on asymmetric noise

#### Fault-Tolerant Computing Layer (Sprint 71.0)

**FaultTolerantProcessor** (`qec/ft_processor.rs`):
```rust
pub struct FaultTolerantProcessor {
    mode: ProcessorMode,           // ResourceEstimation or PauliFrameSimulation
    logical_qubits: Vec<LogicalQubit>,
    magic_supply: MagicSupply,
    scheduler: Scheduler,
}

pub enum ProcessorMode {
    ResourceEstimation,    // Count qubits, cycles, T-gates
    PauliFrameSimulation,  // Track Pauli frames through execution
}
```

**Key Components:**

1. **LogicalQubit** (`ft_types.rs`):
```rust
pub struct LogicalQubit {
    basis: KnownBasis,        // Z, X, or Unknown
    pauli_frame: PauliFrame,  // Tracked Pauli corrections
}

pub enum KnownBasis {
    Z,       // |0⟩ or |1⟩ eigenstates
    X,       // |+⟩ or |-⟩ eigenstates
    Unknown, // General superposition
}
```

2. **MagicSupply** (`magic_supply.rs`):
```rust
pub struct MagicSupply {
    factory_fleet: FactoryFleet,
    throughput: f64,       // T-states/cycle
    latency: usize,        // Cycles to produce T-state
}

pub struct FactoryFleet {
    factories: Vec<DistillationFactory>,
    strategy: AllocationStrategy,  // RoundRobin, LeastLoaded, FillFirst
}
```

3. **Scheduler** (`scheduler.rs`):
```rust
pub struct Scheduler {
    cycle_map: HashMap<GateId, usize>,  // Gate → execution cycle
    dependencies: DependencyGraph,
}

pub fn schedule_circuit(&self, circuit: &[QGate]) -> Schedule {
    // Maps gates to cycles, respecting T-gate factory latency
}
```

4. **Gate Decomposition** (`gate_decomp.rs`):
```rust
pub fn decompose_toffoli() -> Vec<QGate> {
    // Toffoli → 7 T-gates (standard decomposition)
}

pub fn decompose_ccz() -> Vec<QGate> {
    // CCZ → Hadamard sandwich around Toffoli
}
```

5. **InjectionModel** (`injection.rs`):
```rust
pub enum InjectionModel {
    Abstract,   // Perfect T-state injection (for estimates)
    Sampled,    // Probabilistic injection with error rates
}
```

**Resource Estimation Output:**
```rust
pub struct FTResourceEstimate {
    pub logical_qubits: usize,
    pub physical_qubits: usize,
    pub spacetime_volume: u64,     // Qubit-cycles
    pub t_count: usize,
    pub t_depth: usize,
    pub factories_needed: usize,
    pub execution_cycles: usize,
}
```

**GPU-Accelerated Stabilizer Tableau** (`gpu_stabilizer.rs`):
- CUDA kernels: `batch_rowmult`, `find_anticommuting`, `batch_popcount`
- O(n²) bit memory layout with column-major GPU format
- Coalesced access for efficient parallel simulation

#### Noise Models (`noise.rs`)

**Supported Noise Types:**

| Model | Description | Parameters |
|-------|-------------|------------|
| **Depolarizing** | Random Pauli error | p (error rate) |
| **Bit-flip** | X error only | p (flip probability) |
| **Phase-flip** | Z error only | p (flip probability) |
| **Amplitude damping** | T₁ relaxation | γ (damping rate) |
| **Phase damping** | T₂ dephasing | γ (damping rate) |

**Usage:**
```rust
use engine::qec::noise::{NoiseModel, apply_noise};

let noise = NoiseModel::depolarizing(0.001);  // 0.1% error rate
apply_noise(&mut state, &noise);
```

### 3. Error Mitigation (`mitigation/`) - Sprint 68-71

**Purpose**: Reduce errors in NISQ circuits without full QEC overhead

#### Zero-Noise Extrapolation (ZNE) with Rescue

**Implementation** (`zne.rs`):
```rust
pub struct ZNEConfig {
    pub scale_factors: Vec<f64>,    // e.g., [1.0, 1.5, 2.0, 2.5]
    pub extrapolation: Extrapolation,
    pub gate_noise_only: bool,      // Preserve measurement error rate
}

pub enum Extrapolation {
    Linear,
    Polynomial(usize),  // Degree
    Exponential,
}

pub fn mitigate_zne(
    circuit: &[QGate],
    expectation_fn: impl Fn(&[QGate]) -> f64,
    config: &ZNEConfig,
) -> ZNEResult
```

**ZNE Rescue Validation** (Sprint 71.0):
```rust
pub struct ZNEValidation {
    pub monotonicity_ok: bool,      // E(λ) increases with λ
    pub intercept_sane: bool,       // E(0) within physical bounds
    pub uncertainty_ok: bool,       // Error bars acceptable
    pub r_squared: f64,             // Fit quality
}

pub fn validate_extrapolation(samples: &[(f64, f64)]) -> ZNEValidation {
    // If validation fails, gracefully fallback to baseline
}
```

**Adaptive Lambda Selection**:
```rust
pub fn select_scale_factors(
    circuit: &[QGate],
    variance_budget: f64,
) -> Vec<f64> {
    // Probe circuit variance, select lambdas that don't exceed budget
}
```

#### Readout Error Mitigation (REM)

```rust
pub struct REMConfig {
    pub calibration_shots: usize,
    pub method: REMMethod,
}

pub enum REMMethod {
    MatrixInversion,      // Invert confusion matrix
    LeastSquares,         // Constrained least squares
    MaximumLikelihood,    // ML estimation
}

pub fn calibrate_rem(backend: &impl Backend) -> ConfusionMatrix
pub fn mitigate_rem(counts: &Counts, calibration: &ConfusionMatrix) -> Counts
```

#### Combined Mitigation Pipeline

```rust
pub struct MitigationPipeline {
    rem: Option<REMConfig>,
    zne: Option<ZNEConfig>,
}

// Pre-mitigation architecture: REM → ZNE
pub fn mitigate(
    circuit: &[QGate],
    pipeline: &MitigationPipeline,
) -> MitigatedResult
```

**MitigationDiagnostics** (Human-readable explanations):
```rust
pub struct MitigationDiagnostics {
    pub strategy_used: String,
    pub fallback_reason: Option<String>,
    pub confidence: f64,
    pub explanation: String,  // "ZNE extrapolation failed monotonicity; using baseline"
}
```

**Performance** (H₂ Molecule VQE):
- REM alone: **94.1% error reduction**
- ZNE alone: ~60-80% error reduction (circuit-dependent)
- Combined: Up to 98% error reduction

### 4. Hardware Transpilation (`transpile/`)

#### Native Gate Sets

**IBM Quantum (Falcon, Heron chips):**
```
Native: {SX, Rz, CX}
SX = √X = [[1+i, 1-i], [1-i, 1+i]] / 2
Rz(θ) = diag(e^(-iθ/2), e^(iθ/2))
CX = CNOT
```

**IonQ Trapped-Ion (Aria chip):**
```
Native: {GPI(φ), GPI2(φ), MS(φ)}
GPI(φ) = [[0, e^(-iφ)], [e^(iφ), 0]]  (π rotation)
GPI2(φ) = [[1, -ie^(-iφ)], [-ie^(iφ), 1]] / √2  (π/2 rotation)
MS(φ) = Mølmer-Sørensen entangling gate (2-qubit)
```

#### Transpilers

**IBM Transpiler** (`ibm.rs`):
```rust
pub struct IBMTranspiler {
    toffoli_strategy: ToffoliStrategy,
}

pub enum ToffoliStrategy {
    Decompose6CX,    // 6 CX decomposition (standard)
    Decompose3CX,    // 3 CX with ancilla (requires extra qubit)
}

pub fn transpile(&self, circuit: &[QGate]) -> IBMCircuit
```

**Decomposition Examples:**
```
H = SX → Rz(π/2) → SX
X = SX → SX
Rx(θ) = Rz(-π/2) → SX → Rz(θ) → SX → Rz(π/2)
```

**IonQ Transpiler** (`ionq.rs`):
```rust
pub struct IonQTranspiler;

pub fn transpile(&self, circuit: &[QGate]) -> IonQCircuit
```

**Decomposition Examples:**
```
H = GPI2(0) → GPI(π/2)
X = GPI(0)
Ry(θ) = GPI2(0) → Rz(θ) → GPI2(0)
```

#### Optimization Pipeline (SPRINT 54.0)

**Stage 1: Pre-Transpile Optimization**
1. **Gate Cancellation**: H·H = I, X·X = I, etc.
2. **Rotation Merging**: Rz(θ₁)·Rz(θ₂) = Rz(θ₁+θ₂)
3. **Commutation**: Move gates to enable cancellation

**Stage 2: Transpile to Native Gates**
- Convert universal gates → native gate set
- Preserve circuit semantics

**Stage 3: T-Depth Minimization**
- **Goal**: Reduce T-gate depth (critical for fault-tolerant QC)
- **Method**: Parallelize T gates using commutativity
- **Impact**: Reduces magic state factory time (bottleneck in FT circuits)

**Stage 4: Post-Transpile Optimization**
- Gate cancellation in native gate language
- Hardware-specific optimization (e.g., cross-resonance duration on IBM)

**API:**
```rust
use engine::transpile::optimizer::OptimizationPipeline;

let pipeline = OptimizationPipeline::default();
let optimized = pipeline.optimize(&circuit);
println!("Original depth: {}, Optimized depth: {}", circuit.len(), optimized.len());
```

#### T-Gate Analysis

**Motivation**: T gates require **magic state distillation** in fault-tolerant QC
**Cost**: 15-to-1 distillation → 15 qubits + 15 time steps per T gate

**T-Count** (`t_count.rs`):
```rust
pub fn analyze_t_gates(circuit: &[QGate]) -> TGateAnalysis {
    TGateAnalysis {
        t_count: usize,
        tdg_count: usize,
        toffoli_count: usize,  // 7 T gates per Toffoli
        total_t_equivalent: usize,
    }
}
```

**T-Depth** (`t_depth.rs`):
```rust
pub fn minimize_t_depth(circuit: &[QGate]) -> Vec<QGate>
```
- Parallelize commuting T gates
- Example: T(0)·T(1) → depth 1 (parallel), T(0)·T(0) → depth 2 (sequential)

**Magic State Budgeting:**
```rust
let analysis = analyze_t_gates(&circuit);
let factories_needed = analysis.total_t_equivalent / factory_throughput;
let distillation_time = analysis.t_depth * distillation_latency;
```

#### Error-Aware Circuit Compilation (Sprint 69)

**Purpose**: Optimize circuits considering error propagation and gate criticality

**Error Propagation Analysis** (`error_analysis.rs`):
```rust
pub struct ErrorAnalysis {
    pub error_amplification: HashMap<GateId, f64>,
    pub fanout_factor: HashMap<GateId, usize>,
    pub depth_position: HashMap<GateId, usize>,
}

pub fn analyze_error_propagation(circuit: &[QGate]) -> ErrorAnalysis {
    // Track how errors at each gate propagate to measurement
}
```

**Gate Criticality Scoring**:
```rust
pub struct GateCriticality {
    pub error_amplification: f64,  // How much this gate amplifies errors
    pub fanout: usize,             // Number of dependent gates
    pub depth: usize,              // Distance from measurement
    pub score: f64,                // Combined criticality metric
}

pub fn score_criticality(analysis: &ErrorAnalysis) -> Vec<GateCriticality>
```

**T-Depth Minimization** (`circuit_rewrite.rs`):
```rust
pub fn minimize_t_depth(circuit: &[QGate]) -> Vec<QGate> {
    // Reorder gates respecting dependencies
    // Parallelize T-gates where possible
    // Reduces magic state factory time
}
```

**Adaptive Distillation Protocols** (`adaptive_protocols.rs`):
```rust
pub enum DistillationProtocol {
    FifteenToOne,      // Standard: 15 noisy → 1 clean
    TwentyToFour,      // High-throughput: 20 → 4
    TriorthogonalCode, // Lower overhead for specific circuits
}

pub fn select_protocol(
    gate_criticality: &GateCriticality,
    target_error: f64,
) -> DistillationProtocol {
    // Per-T-gate adaptive selection
}
```

**Multi-Objective Pareto Optimization** (`multi_objective.rs`):
```rust
pub struct OptimizationObjectives {
    pub minimize_t_depth: bool,
    pub minimize_physical_qubits: bool,
    pub minimize_error_rate: bool,
    pub minimize_execution_time: bool,
}

pub fn pareto_optimize(
    circuit: &[QGate],
    objectives: &OptimizationObjectives,
) -> Vec<OptimizedCircuit> {
    // Returns Pareto frontier of non-dominated solutions
}
```

#### OpenQASM 3.0 Support (SPRINT 55.0)

**Full Implementation** (`qasm3/`):
- **Lexer**: Token stream from source text
- **Parser**: AST (Abstract Syntax Tree) construction
- **Emitter**: AST → OpenQASM 3.0 text
- **Interoperability**: Import/export with Qiskit, Cirq, Q#

**Supported Features:**
- Qubit declarations: `qubit[5] q;`
- Standard gates: `h q[0]; cx q[0], q[1];`
- Measurements: `bit[5] c; c = measure q;`
- Gate definitions: `gate custom a, b { ... }`
- Parameters: `gate rx(θ) q { ... }`

**API:**
```rust
use engine::transpile::qasm3::{parse_qasm3, emit_qasm3};

// Parse OpenQASM → internal circuit
let circuit = parse_qasm3(qasm_source)?;

// Emit circuit → OpenQASM
let qasm_output = emit_qasm3(&circuit);
println!("{}", qasm_output);
```

**Example:**
```qasm
// Input: OpenQASM 3.0
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;

// After roundtrip: identical circuit
```

#### Resource Estimation (SPRINT 56.0)

**Purpose**: Estimate physical resources for fault-tolerant quantum computing

**NISQ Resource Estimation** (`resource_estimate.rs`):
```rust
pub struct ResourceEstimate {
    pub logical_qubits: usize,
    pub circuit_depth: usize,
    pub gate_count: usize,
    pub t_count: usize,
    pub execution_time_us: f64,  // On NISQ hardware
}

pub fn estimate_resources(circuit: &[QGate]) -> ResourceEstimate
```

**Fault-Tolerant Resource Estimation:**
```rust
pub struct FTResourceEstimate {
    pub logical_qubits: usize,
    pub physical_qubits: usize,     // Includes QEC overhead
    pub code_distance: usize,        // Surface code distance
    pub t_factories: usize,          // Magic state factories needed
    pub execution_time_sec: f64,     // Wall-clock time
    pub error_budget: f64,           // Total error probability
}

pub fn estimate_resources_ft(
    circuit: &[QGate],
    code_distance: usize,
    physical_error_rate: f64
) -> FTResourceEstimate
```

**Hardware Comparison** (`analysis.rs`):
```rust
pub fn analyze_ibm_circuit(circuit: &IBMCircuit) -> CircuitAnalysis {
    CircuitAnalysis {
        cx_count: usize,
        single_qubit_count: usize,
        depth: usize,
        fidelity_estimate: f64,  // Based on hardware error rates
    }
}
```

**Example:**
```rust
use engine::transpile::{estimate_resources, estimate_resources_ft};

let nisq = estimate_resources(&circuit);
println!("NISQ: {} qubits, {} μs", nisq.logical_qubits, nisq.execution_time_us);

let ft = estimate_resources_ft(&circuit, 15, 1e-3);  // d=15, p=0.1%
println!("FT: {} physical qubits, {} factories, {:.2} sec",
    ft.physical_qubits, ft.t_factories, ft.execution_time_sec);
```

**Typical Results:**
- **Shor N=2048**: ~4000 logical qubits, ~20M physical qubits (d=15), ~100 T factories
- **VQE H₂**: 4 logical qubits, ~1600 physical qubits (d=15), 1 T factory
- **QAOA MaxCut n=20**: 20 logical qubits, ~8000 physical qubits (d=15), 2 T factories

### 4. Hardware Mapping (`hardware/`)

#### Hardware Profiles (`profile.rs`)

**Pre-defined Profiles:**
```rust
pub fn ibm_heron() -> HardwareProfile {
    HardwareProfile {
        name: "IBM Heron",
        qubit_count: 133,
        topology: TopologyType::HeavyHex,
        t1_us: 100.0,
        t2_us: 80.0,
        single_qubit_error: 0.0001,
        two_qubit_error: 0.005,
        readout_error: 0.01,
        gate_times: GateTimes { ... },
    }
}

pub fn ionq_aria() -> HardwareProfile {
    HardwareProfile {
        name: "IonQ Aria",
        qubit_count: 25,
        topology: TopologyType::AllToAll,  // Trapped ions
        t1_us: 100_000.0,  // Long coherence
        t2_us: 10_000.0,
        single_qubit_error: 0.00001,
        two_qubit_error: 0.001,
        readout_error: 0.001,
        gate_times: GateTimes { ... },
    }
}
```

#### Device Topology (`topology.rs`)

**Topology Types:**

| Type | Description | Connectivity | Example Devices |
|------|-------------|--------------|-----------------|
| **Grid** | 2D rectangular grid | 4 neighbors (up/down/left/right) | Google Sycamore |
| **HeavyHex** | IBM hexagonal lattice | 2-3 neighbors (optimized layout) | IBM Heron, Condor |
| **Chain** | 1D linear chain | 2 neighbors (prev/next) | Early superconducting |
| **AllToAll** | Fully connected | All pairs | IonQ (trapped ions) |
| **Custom** | User-defined graph | Arbitrary | Research devices |

**API:**
```rust
use engine::hardware::topology::{Topology, TopologyType};

let topology = Topology::new(TopologyType::HeavyHex, 133);
let neighbors = topology.neighbors(42);  // Qubits adjacent to qubit 42
let distance = topology.distance(10, 50);  // Shortest path length
```

#### SWAP Routing (`swap_router.rs`)

**Problem**: Map logical circuit to physical device with limited connectivity

**Algorithm**:
1. **Initial Placement**: Assign logical qubits → physical qubits (heuristic or random)
2. **Routing**: For each 2-qubit gate:
   - If qubits adjacent: Execute directly
   - Else: Insert SWAP gates to bring qubits together
3. **Optimization**: Minimize SWAP count using lookahead

**Implementation:**
```rust
pub struct SwapRouter {
    topology: Topology,
}

pub fn route(&self, circuit: &[QGate], initial_layout: &[usize]) -> Vec<QGate>
```

**Example:**
```
Logical circuit: CNOT(0, 5)
Physical layout: q0 → p10, q5 → p42
Topology: Grid (distance 8 between p10 and p42)

Routed circuit:
SWAP(p10, p11), SWAP(p11, p12), ..., SWAP(p41, p42),  // 8 SWAPs
CNOT(p10, p42)  // Now adjacent after SWAPs
```

**SWAP Cost:**
- Each SWAP = 3 CNOT gates (standard decomposition)
- Overhead: 8 SWAPs × 3 = 24 CNOT gates for CNOT(0, 5) in example above

#### Magic State Factory Layout (`factory_layout.rs`)

**Fault-Tolerant Context**: T gates require distilled magic states

**Factory Architecture:**
- **Input**: Noisy T states (error rate ~1%)
- **Distillation Protocol**: 15-to-1 or 20-to-4 (Reed-Muller codes)
- **Output**: Clean T states (error rate < 10⁻⁸)

**Layout Problem**: Place distillation factories on 2D chip

**Constraints:**
- Factory size: 15-50 physical qubits per factory
- Communication: Deliver magic states to computation region
- Parallelism: Multiple factories for T-depth > 1 circuits

**Implementation:**
```rust
pub struct FactoryLayout {
    chip_size: (usize, usize),  // 2D chip dimensions
    factories: Vec<FactoryPlacement>,
}

pub fn optimize_layout(
    circuit_t_depth: usize,
    available_area: (usize, usize)
) -> FactoryLayout
```

#### Hardware-Aware Resource Budgeting (`aware_budget.rs`)

**Purpose**: Adjust resource estimates for specific hardware

**Example:**
```rust
use engine::hardware::{HardwareAwareBudget, profiles};

let budget = estimate_resources_ft(&circuit, 15, 1e-3);
let ibm = profiles::ibm_heron();
let hw_budget = HardwareAwareBudget::from_budget(&budget, &ibm, 2);

println!("Logical qubits: {}", hw_budget.logical_qubits);
println!("Physical qubits: {}", hw_budget.physical_qubits);
println!("SWAP overhead: {} gates", hw_budget.swap_gates);
println!("Fidelity: {:.6}", hw_budget.estimated_fidelity);
```

**Adjustments:**
- **Topology**: Add SWAP overhead based on connectivity
- **Error Rates**: Adjust code distance for target fidelity
- **Gate Times**: Calculate execution time from gate durations

---

## Advanced Modules

### 1. QRAM (Quantum Random Access Memory) - `qram/`

**Purpose**: Superposition queries to quantum memory
**Applications**: Quantum databases, Grover with memory, quantum machine learning

#### Bucket-Brigade Architecture (SPRINT 46.0)

**Structure**: Binary tree of quantum routers (Fredkin/CSWAP gates)

```
Address qubits: |a₂ a₁ a₀⟩
Memory cells: M[0], M[1], ..., M[7]

Operation: |a⟩|0⟩ → |a⟩|M[a]⟩
Superposition: (|0⟩ + |5⟩)/√2 |0⟩ → (|0⟩|M[0]⟩ + |5⟩|M[5]⟩)/√2
```

**Implementation** (`bucket_brigade.rs`):
```rust
pub struct QRAMQuery {
    n_address_qubits: u32,
    memory: Vec<u8>,  // 2^n cells
}

pub fn query_classical(&self, address: usize) -> Option<u8>
pub fn query_quantum(&mut self, address_state: &QState) -> QState
```

**Complexity:**
- **Circuit Depth**: O(n) for n address qubits
- **Ancilla**: O(2^n) for routing tree
- **Gates**: O(2^n) Fredkin gates

#### Fault-Tolerant QRAM (SPRINT 47.0)

**Protection**: Steane [[7,1,3]] encoding on memory cells

**Implementation** (`ft_qram.rs`, `protected_cell.rs`):
```rust
pub struct ProtectedMemoryCell {
    logical_value: u8,
    physical_qubits: [u8; 7],  // Steane encoding
}

pub struct FaultTolerantQRAM {
    cells: Vec<ProtectedMemoryCell>,
    error_correction_enabled: bool,
}

pub fn query_with_correction(&mut self, address: usize) -> Result<u8, QECError>
```

**Features:**
- Syndrome measurement between operations
- Automatic error correction
- Transversal logical operations (fault-tolerant by construction)

**Cost:**
- **Physical Qubits**: 7 per logical qubit
- **Latency**: +syndrome measurement time per query

#### Polynomial Encoding (SPRINTs 48-49)

**Idea**: Encode memory as polynomial coefficients

```
M[0], M[1], ..., M[N-1]  →  P(x) = Σᵢ M[i]·xⁱ
Query address a: Evaluate P(a) using quantum arithmetic
```

**Advantages:**
- **Compact**: O(n) qubits for 2^n cells (vs O(2^n) for bucket-brigade)
- **No Ancilla**: Phase-encoding eliminates routing qubits

**Implementation** (`polynomial.rs`, `poly_router.rs`):
```rust
pub struct PolynomialQRAM {
    coefficients: Vec<u8>,  // Polynomial coefficients
    n_address_qubits: u32,
}

pub fn query_phase_encoded(&self, address: u32) -> f64  // Phase φ = P(address)
```

**Limitations:**
- **Read-only**: Cannot update memory easily
- **Quantum Query**: Classical queries not efficient with this encoding

#### Sparse Bucket-Brigade (SPRINT 49.0)

**Use Case**: Large address space, few non-zero cells

**Implementation** (`sparse_bucket.rs`):
```rust
pub struct SparseQRAM {
    occupied_cells: HashMap<usize, u8>,  // Only store non-zero cells
    n_address_qubits: u32,
}
```

**Benefits:**
- **Memory**: O(k) for k occupied cells (vs O(2^n) for dense)
- **Circuit**: Only route to occupied cells (pruned tree)

#### Stabilizer-Based QRAM (SPRINT 50.0)

**Optimization**: Use Gottesman-Knill theorem for efficient simulation

**Implementation** (`stabilizer_qram.rs`):
```rust
pub struct StabilizerQRAM {
    state: StabilizerState,
    memory: Vec<u8>,
}
```

**Features:**
- **Pauli Frame Tracking** (`pauli_frame.rs`): Track Pauli corrections without measurement
- **Graph-Based Routing** (`graph_router.rs`): Arbitrary routing topologies

**Simulation Performance:**
- **Classical**: O(n²) per query (vs O(2^n) for full state)
- **Qubits**: Tested up to 10,000 address qubits (classical simulation)

#### Magic State Budgeting (SPRINT 51.0)

**Motivation**: QRAM routing uses many Toffoli gates → many T gates

**Magic State Cost**:
- 1 Toffoli = 7 T gates (standard decomposition)
- 1 T gate = 15 physical qubits × 15 time steps (15-to-1 distillation)
- 1 Toffoli ≈ 105 qubit-time units

**Implementation** (`magic_budget.rs`, `distillation.rs`):
```rust
pub struct MagicStateBudget {
    pub t_count: usize,
    pub toffoli_count: usize,
    pub factories_needed: usize,
    pub distillation_time_us: f64,
}

pub fn MagicStateBudget::for_qram_routing(n_address_qubits: usize) -> Self

pub struct DistillationConfig {
    pub protocol: Protocol,  // FifteenToOne, TwentyToFour
    pub physical_error_rate: f64,
    pub target_error_rate: f64,
}
```

**Example:**
```rust
let budget = MagicStateBudget::for_qram_routing(8);  // 2^8 = 256 cells
println!("Toffoli count: {}", budget.toffoli_count);  // ~255 for binary tree
println!("T count: {}", budget.t_count);  // ~1785
println!("Factories: {}", budget.factories_needed);  // Depends on T-depth
```

### 2. Spiking Neural Networks (`snn/`)

#### Neuromorphic Computing - CHUNGUS 4

**Architecture**: Event-driven neural networks mimicking biological neurons

**Neuron Model**: Leaky Integrate-and-Fire (LIF)
```
dV/dt = (V_rest - V) / τ + I_syn / C
If V ≥ V_threshold: Emit spike, V ← V_reset
```

**Implementation** (`neuron.rs`):
```rust
pub struct LIFNeuron {
    pub voltage: i16,           // Membrane potential (mV × 10)
    pub threshold: i16,         // Spike threshold
    pub reset: i16,             // Reset voltage after spike
    pub leak_tau: u8,           // Time constant (ticks)
    pub refractory: u8,         // Refractory period (ticks)
}

impl LIFNeuron {
    pub fn tick(&mut self, input_current: i16) -> bool {
        // Returns true if spike emitted
    }
}
```

**State Size**: 8 bytes per neuron (optimized for cache)

#### Synaptic Connections (`synapse.rs`)

**Synapse Model**:
```rust
pub struct Synapse {
    pub weight: i8,             // Synaptic strength (-127 to 127)
    pub delay: u8,              // Propagation delay (ticks)
    pub pre_neuron: u16,        // Source neuron ID
    pub post_neuron: u16,       // Target neuron ID
}
```

**Cross-CPU Synapses**:
```rust
pub struct CrossCpuSynapse {
    pub synapse: Synapse,
    pub target_cpu: u16,        // For distributed SNN
}
```

**State Size**: 4 bytes per synapse

#### Learning Rules (`stdp.rs`)

**STDP (Spike-Timing-Dependent Plasticity)**:
```
Δw = { +A_plus × exp(-Δt / τ_plus)   if t_post > t_pre (potentiation)
     { -A_minus × exp(Δt / τ_minus)  if t_post < t_pre (depression)
```

**R-STDP (Reward-Modulated STDP)**:
```
Δw = STDP(Δt) × R
where R = reward signal (-127 to 127)
```

**Implementation**:
```rust
pub fn apply_stdp(
    pre_spike_time: u32,
    post_spike_time: u32,
    current_weight: i8,
    a_plus: f32,
    a_minus: f32,
) -> i8
```

#### Action-Specific R-STDP Learning (Sprint 71.0)

**GPU Kernels for RL Credit Assignment** (`gpu_fused.rs`):

1. **Eligibility Trace Update**:
```cuda
// snn_fused_update_eligibility kernel
// STDP: Pre + Post spike → LTP (+ltp_rate)
// STDP: Pre spike alone → LTD (-ltd_rate)
// Decay: e *= tau_decay
```

2. **Action-Specific Reward** (Winner-Takes-All):
```rust
pub fn learn_action(
    &mut self,
    chosen_action: usize,
    reward: f64,
    anti_reward_factor: f64,  // e.g., 0.5
) {
    // Full reward to chosen action's output neuron
    // Anti-reward to other output neurons
    // Δw = learning_rate × eligibility × reward
}
```

3. **Pathway-Based Learning** (Softer Credit):
```rust
pub fn learn_pathway(
    &mut self,
    hidden_scale_factor: f64,  // e.g., 0.7
) {
    // Scale credit for hidden layer connections
    // Prevents signal dilution in deep networks
}
```

**Direct Topology for Simple RL**:
```rust
pub fn generate_direct_csr(
    n_inputs: usize,
    n_outputs: usize,
) -> CSRSynapses {
    // Direct input→output connections
    // Bypasses hidden layers for MountainCar-style tasks
}
```

**Python API**:
```python
snn = tu.FusedSNN(n_neurons=5000, n_inputs=7, n_outputs=3)
snn.enable_learning()
snn.set_learning_params(
    tau_decay=0.85,
    ltp_rate=0.2,
    ltd_rate=0.1,
    learning_rate=0.05
)

# Training loop
action = snn.decide(spike_rates)
env_reward = env.step(action)
snn.learn_action(action, env_reward, anti_reward_factor=0.5)
```

**MountainCar Example** (from `test_mountaincar_rstdp.py`):
- 7-input state encoding (position, velocity, slope, energy)
- 3-output actions (left, neutral, right)
- Goal: Average <110 steps (vs random ~200)

#### Network Architecture (`network.rs`)

**SNNNetwork Builder**:
```rust
pub struct SNNNetwork {
    pub neurons: Vec<LIFNeuron>,
    pub synapses: Vec<Synapse>,
    pub topology: SNNTopology,
}

pub enum SNNTopology {
    Feedforward { layers: Vec<usize> },
    Recurrent { n_neurons: usize, connectivity: f32 },
    Reservoir { input_size: usize, reservoir_size: usize, output_size: usize },
}
```

**Example**:
```rust
let network = SNNNetwork::feedforward(vec![784, 128, 10]);  // MNIST-like
network.add_spike(0, 100);  // Input spike to neuron 0 at tick 100
network.tick();  // Advance 1 time step
let output_spikes = network.get_output_spikes();
```

#### CUDA Acceleration (`gpu_kernels.rs`)

**GPU Kernels**:
- Neuron state update (parallel across all neurons)
- Spike propagation (parallel across synapses)
- STDP weight updates (parallel across recent spike pairs)

**Performance**:
- **CPU**: ~10K neurons at 1000 ticks/sec
- **GPU**: ~200K neurons at 1000 ticks/sec (RTX 4070)
- **Speedup**: ~20× for large networks

#### GPU-Resident Fused SNN Architecture (EPIC 97)

**Zero-PCIe-Transfer Simulation** (`gpu_fused.rs`):
```rust
pub struct FusedGpuSNN {
    weights: CudaBuffer<f32>,      // GPU-resident
    neuron_states: CudaBuffer<f32>,
    spike_buffer: CudaBuffer<u8>,
    eligibility: CudaBuffer<f32>,  // For R-STDP
}

// Single fused kernel: input → LIF → spike detection
pub fn tick_fused(&mut self, inputs: &[f32]) -> Vec<bool>
```

**Structured Sparsity** (`block_sparse_synapses.rs`):
- 2:4 sparsity: 2 non-zeros per 4-element block
- 4:8 sparsity: 4 non-zeros per 8-element block
- GPU-optimal memory layout for Tensor Core acceleration

#### Stabilizer Quantum Neurons (EPICs 130, 134, 135)

**StabilizerNeuron** (`stabilizer_neuron.rs`):
```rust
pub struct StabilizerNeuron {
    tableau: StabilizerTableau,  // Clifford-based quantum state
    threshold: f64,
}

impl StabilizerNeuron {
    pub fn integrate(&mut self, input: f64)
    pub fn fire(&mut self) -> bool  // Clifford-based spike decision
}
```

**StabilizerNetwork** (`stabilizer_network.rs`):
```rust
pub struct StabilizerNetwork {
    neurons: Vec<StabilizerNeuron>,
    synapses: Vec<CliffordSynapse>,
}
```
- **87× speedup** for 300K neurons (vs full quantum simulation)

**HybridQuantumNetwork** (`stabilizer_hybrid.rs`):
- Combines stabilizer neurons with cluster states
- Efficient simulation of quantum-enhanced decision making

**GPU-Accelerated Stabilizer Network** (`gpu_stabilizer_network.rs`):
```rust
pub struct GpuStabilizerNetwork {
    backend: StabilizerBackend,  // Cpu, Gpu, Auto
}

pub enum StabilizerBackend {
    Cpu,
    Gpu,
    Auto,  // GPU for networks ≥ 4K neurons
}
```
- **7.98× speedup** at 100K neurons

#### Quantum-SNN Hybrid (`quantum_hybrid.rs`)

**QuantumSNN**: Decision-making layer combining SNN and quantum interference

**Architecture**:
```
Input (spike rates) → SNN Layer → Quantum Layer → Measurement → Action
                                      ↓
                                   Reward
```

**Interference Modes** (`InterferenceMode`):

| Mode | Description | Performance (vs Epsilon-Greedy) |
|------|-------------|--------------------------------|
| **Epsilon-Greedy** | Classical baseline (ε=0.1) | Baseline (0%) |
| **Continuous** | Quantum interference every tick | +15% |
| **Triggered** | Interference at decision points | **+69%** ✅ |

**Implementation**:
```rust
pub struct QuantumSNN {
    n_inputs: usize,
    hidden_layers: Vec<usize>,
    n_outputs: usize,
    mode: InterferenceMode,
    quantum_state: QState,
    snn_state: SNNNetwork,
}

pub fn decide(&mut self, inputs: &[u8]) -> usize {
    // inputs: spike rates [0-255] for each input neuron
    // returns: action index [0, n_outputs)
}

pub fn learn(&mut self, reward: i8) {
    // reward: [-127, 127] for R-STDP
}
```

**Configuration**:
```rust
pub struct QuantumSNNConfig {
    pub n_inputs: usize,
    pub hidden_layers: Vec<usize>,
    pub n_outputs: usize,
    pub mode: InterferenceMode,
    pub ticks_per_decision: usize,  // SNN ticks before quantum measurement
}
```

**Python API**:
```python
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode

config = QuantumSNNConfig(
    n_inputs=4,
    hidden_layers=[32],
    n_outputs=3,
    mode=InterferenceMode.triggered(),
    ticks_per_decision=20,
)
brain = QuantumSNN(config, seed=42)

action = brain.decide([128, 64, 200, 50])  # Spike rates [0-255]
brain.learn(reward=50)  # Reward [-127, 127]
```

#### Advanced SNN Features (EPICs 121-123)

**Curiosity-Driven Exploration** (`curiosity.rs`):
```rust
pub struct CuriosityModule {
    state_visitation: HashMap<StateHash, usize>,
    prediction_error: Vec<f32>,
}

pub fn intrinsic_reward(&self, state: &[u8]) -> f32 {
    // Reward for visiting novel states
}
```

**Value Function & Credit Assignment** (`value_function.rs`):
```rust
pub struct ValueFunction {
    state_values: HashMap<StateHash, f32>,
    eligibility_traces: Vec<f32>,
    gamma: f32,  // Discount factor
}

pub fn td_error(&self, state: &[u8], reward: f32, next_state: &[u8]) -> f32
```

**Population Coding** (`population_coding.rs`):
```rust
pub fn encode_population(value: f32, min: f32, max: f32, n_neurons: usize) -> Vec<u8> {
    // Encode scalar value as spike rates across neuron population
    // Uses Gaussian tuning curves
}

pub fn decode_population(spike_rates: &[u8]) -> f32 {
    // Decode population activity to scalar value
}
```

#### SNN Scale & Performance

**TILE-8 CPU Integration**:
- 128 neurons per TILE-8 CPU
- 1M+ CPUs supported on single machine
- Cross-CPU spike routing via `spike_router.rs`

**Scaling Results**:
- **10K neurons**: Real-time simulation on CPU
- **100K neurons**: Real-time with multi-core parallelism
- **200K neurons**: Real-time with CUDA acceleration
- **1M+ neurons**: Offline simulation (slower than real-time)

### 3. TILE-8 CPU Toolchain (`tile8/`)

#### Minimal 8-bit Architecture

**ISA (Instruction Set Architecture)**:
- **Data Width**: 8 bits
- **Address Space**: 256 bytes
- **Registers**: 4 general-purpose (R0-R3)
- **Flags**: Zero (Z), Carry (C)
- **Instructions**: 16 opcodes (4-bit opcode)

**Instruction Set** (`isa.rs`):

| Opcode | Mnemonic | Operation | Flags |
|--------|----------|-----------|-------|
| 0x0 | NOP | No operation | - |
| 0x1 | LOAD Rn, [addr] | Rn ← M[addr] | Z |
| 0x2 | STORE [addr], Rn | M[addr] ← Rn | - |
| 0x3 | ADD Rn, Rm | Rn ← Rn + Rm | Z, C |
| 0x4 | SUB Rn, Rm | Rn ← Rn - Rm | Z, C |
| 0x5 | AND Rn, Rm | Rn ← Rn & Rm | Z |
| 0x6 | OR Rn, Rm | Rn ← Rn \| Rm | Z |
| 0x7 | XOR Rn, Rm | Rn ← Rn ^ Rm | Z |
| 0x8 | CMP Rn, Rm | Rn - Rm (no store) | Z, C |
| 0x9 | JMP addr | PC ← addr | - |
| 0xA | JZ addr | If Z: PC ← addr | - |
| 0xB | JC addr | If C: PC ← addr | - |
| 0xC | CALL addr | Push PC; PC ← addr | - |
| 0xD | RET | Pop PC | - |
| 0xE | PUSH Rn | Push Rn to stack | - |
| 0xF | POP Rn | Pop stack to Rn | Z |

#### Assembler (`asm.rs`)

**Assembly Syntax**:
```asm
; Example: Add two numbers
LOAD R0, [0x80]    ; Load from address 0x80
LOAD R1, [0x81]
ADD R0, R1         ; R0 = R0 + R1
STORE [0x82], R0   ; Store result to 0x82
```

**API**:
```rust
pub fn assemble(source: &str) -> Result<Vec<u8>, AssemblerError>
```

#### CPU Builder (`cpu.rs`)

**Tile8Cpu**: Builds TILE-8 CPU on tilemap

**API**:
```rust
pub struct Tile8Cpu {
    origin: (i32, i32),
    binary: Vec<u8>,
}

impl Tile8Cpu {
    pub fn new(origin: (i32, i32), binary: Vec<u8>) -> Self;
    pub fn place_on_tilemap(&self, tilemap: &mut Tilemap);
    pub fn run(&mut self, ticks: usize);
}
```

**Physical Variants**:
- **CPU Tile**: Logic gate implementation on classical tilemap
- **GPU Variant**: GPU-accelerated TILE-8 (EPIC 120 prototype)

#### Sparse Quantum Extensions

**SparseQuantumGrid** (`sparse_quantum.rs`):
```rust
pub struct SparseQuantumGrid {
    qubits: Vec<(i32, i32)>,  // Qubit positions on grid
    state: SparseQState,
}

pub fn create_ghz_state(n_qubits: usize) -> SparseQuantumGrid
pub fn verify_ghz(&self) -> bool  // Checks |00...0⟩ + |11...1⟩ structure
```

**Vec-Based W State** (`sparse_quantum_vec.rs`):
```rust
pub struct FastWState {
    n_qubits: usize,
    amplitudes: Vec<Complex32>,  // n+1 amplitudes
}

pub fn create_w_state(n_qubits: usize) -> FastWState
// W_n = (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √n
```

**Performance**:
- GHZ: O(1) creation, O(1) verification for any n
- W: O(n) creation, O(n) memory, tested up to 1 billion qubits

**MinimalGhzState** (Sprint 72.0) - O(1) for 2^64 Qubits:
```rust
pub struct MinimalGhzState {
    n_qubits: usize,
    blocks: Vec<VecBlock>,  // Always 2 blocks (4KB fixed)
}

pub fn create_minimal_ghz(n_qubits: usize) -> MinimalGhzState
pub fn verify_ghz_fast(&self) -> MinimalGhzVerification  // O(1) without BigInt
```

| Qubits | Creation | Verification | Memory |
|--------|----------|--------------|--------|
| 1M     | 0μs      | 0μs          | 4KB    |
| 100M   | 0μs      | 0μs          | 4KB    |
| 1B     | 0μs      | 0μs          | 4KB    |
| 2^64   | 0μs      | 0μs          | 4KB    |

**Cross-Block GPU CNOT** (Sprint 72.0):
```rust
// Enable entanglement across 128-amplitude block boundaries
pub async fn cnot_any(&self, control: usize, target: usize)
pub async fn cnot_control_high(&self, control: usize, target: usize)
pub async fn cnot_target_high(&self, control: usize, target: usize)
pub async fn cnot_both_high(&self, control: usize, target: usize)
```

**Three GPU Shader Kernels**:
1. **cnot_control_high**: Control qubit ≥7, target <7 (parallel per-block)
2. **cnot_target_high**: Control <7, target ≥7 (conditional block swaps)
3. **cnot_both_high**: Both ≥7 (full block swaps)

**Performance** (65K blocks = 8.4M qubits):
- Control-high: **210M block-ops/sec**
- Target-high: **204M block-ops/sec**
- Both-high: **97M block-ops/sec**

**GPU Ising Mode** (`ising_mode_gpu.rs`):
```rust
pub struct GpuIsingGrid {
    grid: Vec<i8>,
    couplings: Vec<f32>,
}

pub struct GpuGridMaxCut {
    // GPU-accelerated MaxCut solver
}
```
- Gibbs sampling with checkerboard update pattern
- **29.2× speedup** on 128×128 grid
- **1.2B updates/sec**

**Hybrid Fitness-Grover** (`hybrid_search.rs`):
```rust
pub fn hybrid_evolutionary_selection(
    population: &mut Population,
    fitness_fn: impl Fn(&Candidate) -> f64,
) -> SelectionResult
```
- 3.65-3.94× amplification on marked states
- 1.14-1.15× selection pressure per generation

**Grover on TILE-8** (`grover.rs`):
- Grover's algorithm implemented on TILE-8 substrate
- Sparse state representation for large search spaces

#### Noisy Quantum Simulation (`sparse_noise.rs`)

**Features**:
- Pauli noise channels (X, Y, Z errors)
- Depolarizing noise
- Measurement errors
- Sparse state preservation under noise (amplitude threshold)

### 4. Probabilistic Computing (`pbit/`)

**Purpose**: Classical probabilistic bits for optimization and sampling

#### P-Bit Architecture

**Single P-Bit** (`pbit.rs`):
```rust
pub struct PBit {
    state: i8,           // Current state (-1 or +1)
    bias: f64,           // Local bias field
    temperature: f64,    // Effective temperature
}

impl PBit {
    pub fn update(&mut self, input: f64) {
        // Sigmoid activation: P(+1) = σ(input / T)
        self.state = if random() < sigmoid(input / self.temperature) { 1 } else { -1 };
    }
}
```

**P-Bit Network** (`network.rs`):
```rust
pub struct PBitNetwork {
    pbits: Vec<PBit>,
    couplings: Vec<Vec<f64>>,  // Dense coupling matrix J_ij
}

impl PBitNetwork {
    pub fn step(&mut self) {
        // Asynchronous Gibbs sampling
        for i in 0..self.pbits.len() {
            let input = self.compute_local_field(i);
            self.pbits[i].update(input);
        }
    }
}
```

#### Ising Problem Encoding (`ising.rs`)

```rust
pub enum IsingProblem {
    MaxCut(Graph),
    QUBO { Q: Vec<Vec<f64>> },
    SK { J: Vec<Vec<f64>>, h: Vec<f64> },  // Sherrington-Kirkpatrick
}

pub fn encode_maxcut(graph: &Graph) -> PBitNetwork {
    // J_ij = -w_ij for edges, 0 otherwise
}

pub fn encode_qubo(Q: &[Vec<f64>]) -> PBitNetwork {
    // Map QUBO to Ising: s_i ∈ {-1, +1} ↔ x_i ∈ {0, 1}
}
```

#### Samplers (`sampler.rs`)

```rust
pub enum Sampler {
    Gibbs,                          // Standard Gibbs sampling
    SimulatedAnnealing { schedule: AnnealingSchedule },
    ParallelTempering { n_replicas: usize, temps: Vec<f64> },
}

pub fn sample(
    network: &mut PBitNetwork,
    sampler: &Sampler,
    n_steps: usize,
) -> Vec<SampleResult>
```

#### GPU Acceleration (`gpu.rs`)

```rust
pub struct GpuPBitNetwork {
    states: CudaBuffer<i8>,
    couplings: CudaBuffer<f32>,
    random_states: CudaBuffer<u32>,
}
```

**Performance**:
- **15-26× speedup** for n > 500 p-bits
- Optimal for MaxCut instances with dense connectivity

### 5. Tensor Networks (`tensor_network/`)

**Purpose**: Efficient contraction of tensor networks for quantum simulation

#### Core Tensor Operations (`tensor.rs`)

```rust
pub struct Tensor {
    data: Vec<Complex64>,
    indices: Vec<IndexLabel>,  // Named indices for contraction
    shape: Vec<usize>,
}

impl Tensor {
    pub fn contract(&self, other: &Tensor, idx: &IndexLabel) -> Tensor {
        // Sum over shared index
    }
}
```

#### Matrix-Based Contraction (`contraction.rs`)

```rust
pub fn contract_matrix(a: &Tensor, b: &Tensor, shared: &[IndexLabel]) -> Tensor {
    // Reshape to matrices, use BLAS GEMM
    // 10-100× faster than loop-based contraction
}
```

**Performance**:
- Small tensors (< 1K elements): Loop-based competitive
- Large tensors (> 10K elements): Matrix-based **10-100× faster**

#### Path Optimization (`path_optimizer.rs`)

```rust
pub enum PathOptimizer {
    Greedy,                    // Local cost minimization
    BeamSearch { width: usize },
    DynamicProgramming,        // Optimal but expensive
}

pub fn optimize_contraction_path(
    network: &[Tensor],
    optimizer: &PathOptimizer,
) -> ContractionPath
```

**Beam Search**:
- Width 5-10 typically finds near-optimal paths
- O(width × n²) complexity for n tensors

#### Memory-Bounded Slicing (`slicing.rs`)

```rust
pub enum SliceMode {
    Sum,    // Slice and sum results
    Stack,  // Slice and stack intermediate results
}

pub fn slice_contraction(
    network: &[Tensor],
    memory_limit: usize,
    mode: SliceMode,
) -> Tensor
```

**Use Case**: Contract large networks that don't fit in memory

### 6. Physics-Logic Coupling (`physics/logic_coupling.rs`)

**Purpose**: Bidirectional coupling between physics fields and tile computation

#### Coupling Mechanisms

**Heat → Errors**:
```rust
pub enum HeatDegradationMode {
    BitFlip,      // XOR specific bits based on heat
    BitClear,     // Clear bits when overheated
    FullDisable,  // Complete thermal shutdown
    StickyLatch,  // Tile freezes at current value
}

pub struct HeatCouplingConfig {
    pub threshold: u32,           // Heat level to trigger effects
    pub mode: HeatDegradationMode,
    pub affected_bits: u64,       // Bitmask of affected bits
}
```

**Charge → Bias**:
```rust
pub struct ChargeCouplingConfig {
    pub threshold: u32,
    pub bias_scale: f64,          // How charge shifts comparisons
}

// Affects: Lt, Gt, Eq, Neq comparisons, Mux selection, Zero detection
```

**Power → Enable**:
```rust
pub enum UnpoweredBehavior {
    Zero,     // Output 0
    Hold,     // Retain previous value
    HighZ,    // High impedance (undefined)
}

pub struct PowerCouplingConfig {
    pub minimum_power: u8,
    pub behavior: UnpoweredBehavior,
}
```

#### Main API

```rust
pub fn apply_physics_coupling(
    raw_output: u64,
    current: u64,
    tile_type: TileType,
    heat: u32,
    charge: u32,
    power: u8,
    config: &PhysicsCouplingConfig,
) -> u64

pub struct PhysicsCouplingConfig {
    pub enabled: bool,
    pub heat: HeatCouplingConfig,
    pub charge: ChargeCouplingConfig,
    pub power: PowerCouplingConfig,
}
```

**Simulation Integration**:
```rust
impl Simulation {
    pub fn enable_physics_coupling(&mut self)
    pub fn disable_physics_coupling(&mut self)
    pub fn set_physics_coupling_config(&mut self, config: PhysicsCouplingConfig)

    // For testing
    pub fn set_heat_field_for_test(&mut self, x: usize, y: usize, value: u32)
    pub fn set_power_field_for_test(&mut self, x: usize, y: usize, value: u8)
    pub fn set_charge_field_for_test(&mut self, x: usize, y: usize, value: u32)
}
```

**Design Principles**:
- **Deterministic**: No RNG, pure threshold-based logic
- **GPU Parity**: CPU and GPU produce identical results
- **Exempt Tiles**: ClockGlobal and Const always work regardless of physics

### 7. Quantum Search Engine (`search/`)

**Purpose**: Quantum-enhanced neural search with Grover amplification

#### Candidate Encoding (`candidate.rs`)

```rust
pub struct Candidate {
    genome: Vec<u64>,
    fitness: f64,
}

pub fn encode_xorshift64(seed: u64, genome_size: usize) -> Candidate {
    // Deterministic pseudo-random genome generation
}
```

#### Fitness Functions (`fitness.rs`)

```rust
pub enum FitnessFunction {
    OneMax,                          // Count 1-bits
    NKLandscape { n: usize, k: usize },
    MaxSAT { clauses: Vec<Clause> },
}

pub fn evaluate(candidate: &Candidate, fitness: &FitnessFunction) -> f64
```

#### Search Substrate (`substrate.rs`)

```rust
pub struct SearchSubstrate {
    population_size: usize,  // 1M to 268M candidates
    fitness_fn: FitnessFunction,
}

pub fn parallel_evaluate(&self, candidates: &[Candidate]) -> Vec<f64>
```

#### Selection Strategies (`selection.rs`)

```rust
pub enum SelectionStrategy {
    Tournament { size: usize },
    Roulette,
    Rank,
    Truncation { ratio: f64 },
    SUS,  // Stochastic Universal Sampling
}
```

#### Grover Amplification (`grover_amplify.rs`)

```rust
pub fn amplify_marked_states(
    population: &mut [Candidate],
    fitness_threshold: f64,
    iterations: usize,
) -> AmplificationResult {
    // O(√N) quantum speedup for finding high-fitness candidates
}
```

**Performance**:
- **3.65-3.94× amplification** on marked states
- **O(√N) speedup** over classical search

### 8. QRAM Extensions (Sprints 63, 70)

#### PackedPauliFrame64 (`packed_frame.rs`)

**Purpose**: Efficient ensemble Pauli tracking for 64 parallel noise trajectories

```rust
pub struct PackedPauliFrame64 {
    x_bits: Vec<u64>,  // Bit-sliced X components
    z_bits: Vec<u64>,  // Bit-sliced Z components
}

impl PackedPauliFrame64 {
    // Clifford propagation
    pub fn apply_cnot(&mut self, control: usize, target: usize)
    pub fn apply_h(&mut self, qubit: usize)
    pub fn apply_s(&mut self, qubit: usize)
    pub fn apply_cz(&mut self, a: usize, b: usize)
    pub fn apply_swap(&mut self, a: usize, b: usize)

    // Non-Clifford surrogates
    pub fn apply_ccz_surrogate(&mut self, a: usize, b: usize, c: usize)
    pub fn apply_toffoli_surrogate(&mut self, c1: usize, c2: usize, t: usize)
    pub fn apply_fredkin_surrogate(&mut self, c: usize, a: usize, b: usize)
}
```

**AoSoA Layout**: Array-of-Structs-of-Arrays for cache locality

#### Magic State Factory Scheduling (`factory_scheduler.rs`)

```rust
pub struct FactoryScheduler {
    factories: Vec<MagicStateFactory>,
    schedule: Schedule,
}

pub fn analyze_t_dependencies(circuit: &[QGate]) -> DependencyGraph {
    // Identify T-gate parallelism opportunities
}

pub fn schedule_factories(
    circuit: &[QGate],
    n_factories: usize,
    config: SchedulerConfig,
) -> FactorySchedule
```

**Scheduling Algorithms**:
- Greedy (fast, good approximation)
- List scheduling (better quality)
- Simulated annealing (near-optimal)

**Auto-Configuration**:
```rust
pub enum SchedulerObjective {
    MinimizeLatency,    // Fastest execution
    MinimizeQubits,     // Smallest footprint
    Balanced,           // Trade-off
}

pub fn auto_configure(
    circuit: &[QGate],
    objective: SchedulerObjective,
) -> FactoryScheduler
```

**Performance** (VQE H₂ with 84 T-gates):
- Single factory: baseline
- 4 factories: **4× speedup**

### 9. Simple CPU (`simulation.rs`)

**Purpose**: Minimal 8-bit CPU for educational and testing purposes

```rust
pub struct SimpleCPU {
    registers: u64,     // 8 registers packed into 64 bits
    memory: [u8; 256],
    pc: u8,
    flags: Flags,
}

pub struct Flags {
    zero: bool,
    carry: bool,
}
```

**ISA (7 Instructions)**:
| Opcode | Mnemonic | Operation |
|--------|----------|-----------|
| 0 | NOP | No operation |
| 1 | LOAD_IMM Rn, imm | Rn ← imm |
| 2 | ADD Rn, Rm | Rn ← Rn + Rm |
| 3 | SUB Rn, Rm | Rn ← Rn - Rm |
| 4 | JUMP addr | PC ← addr |
| 5 | JUMP_IF_ZERO addr | If Z: PC ← addr |
| 6 | HALT | Stop execution |

**Test Programs**:
- Arithmetic operations
- Loop execution
- Fibonacci sequence

---

## Hardware & Performance

### 1. Backend Comparison

| Backend | Precision | Performance (COPS) | Qubits | Requirements |
|---------|-----------|-------------------|---------|--------------|
| **Scalar** | Reference | 200M | 1-30 | None |
| **AVX2** | f32 | 1.2B | 1-28 | AVX2 CPU |
| **AVX512** | f32 | 2.1B | 1-28 | AVX512 CPU |
| **Cranelift JIT** | f32 | 800M-1.5B | 1-30 | JIT feature |
| **CUDA FP32** | f32 | 821M | 6-30 | NVIDIA GPU |
| **WMMA FP16** | f16→f32 | 12.4B | 6-30 | Tensor Cores |
| **WMMA FP8** | f8→f32 | 21.3B | 6-30 | Hopper+ GPU |

**COPS**: Computational Ops Per Second (1 COPS = 1 amplitude update)
**PCOPS**: Peta-COPS = 10¹⁵ ops/sec

### 2. GPU Scaling (RTX 4070)

**Qubit Scaling**:
| Qubits | Amplitudes | FP32 (ms) | WMMA FP16 (ms) | Speedup |
|--------|------------|-----------|----------------|---------|
| 10 | 1,024 | 0.05 | 0.04 | 1.25× |
| 15 | 32,768 | 0.8 | 0.12 | 6.7× |
| 20 | 1,048,576 | 25 | 3.2 | 7.8× |
| 25 | 33,554,432 | 800 | 52 | 15.4× |
| 28 | 268,435,456 | 6400 | 420 | 15.2× |

**Circuit Depth Scaling** (20 qubits):
| Depth | FP32 (ms) | WMMA FP16 (ms) | Speedup |
|-------|-----------|----------------|---------|
| 10 | 250 | 32 | 7.8× |
| 100 | 2500 | 165 | 15.2× |
| 1000 | 25000 | 1650 | 15.2× |

**Tensor Core Speedup** (constant beyond ~100 gates due to amortization)

### 3. FP8 Tensor Cores (EPIC 114)

**Architecture Requirements**:
- **Hopper**: H100, H200 (compute_90)
- **Blackwell**: B100, B200 (compute_100)
- **Ada Lovelace**: RTX 4090, 5090 (compute_89, FP16 only)

**FP8 Formats**:
- **E4M3**: 1 sign + 4 exponent + 3 mantissa (better precision)
- **E5M2**: 1 sign + 5 exponent + 2 mantissa (better range)

**Performance** (H100):
| Qubit | FP32 COPS | FP8 COPS | Speedup |
|-------|-----------|----------|---------|
| 20 | 42M | 1.1B | 26× |
| 25 | 42M | 1.1B | 26× |
| 28 | 42M | 1.1B | 26× |

**ILP Kernels** (Instruction-Level Parallelism):
- Unrolled loops (4-way)
- Memory latency hiding
- +15-20% performance over naive FP8

### 4. Cross-Backend Parity

**Validation** (`tests/compare3_parity.rs`):
```rust
#[test]
fn test_cross_backend_parity_h_gate() {
    let mut scalar_state = QState::new(3);
    let mut cuda_state = QState::new(3);

    apply_h_scalar(&mut scalar_state, 1);
    apply_h_cuda(&mut cuda_state, 1);

    assert_states_equal(&scalar_state, &cuda_state, 1e-6);
}
```

**Epsilon Tolerance**:
- **Standard**: ε = 1e-6 (6 decimal places)
- **FP8**: ε = 5e-3 (reduced precision, still useful)

**Parity Test Coverage**:
- All 23 gate types
- Circuit depths 1-1000
- Qubit counts 1-28
- Random circuits (1000+ test cases)

### 5. Memory Requirements

**Dense State**:
| Qubits | Amplitudes | Memory |
|--------|------------|--------|
| 10 | 1,024 | 8 KB |
| 20 | 1,048,576 | 8 MB |
| 25 | 33,554,432 | 256 MB |
| 28 | 268,435,456 | 2 GB |
| 30 | 1,073,741,824 | 8 GB |

**Block-Sparse State** (128 amplitudes per block):
| Qubits | Blocks | Memory (32 blocks) |
|--------|--------|-------------------|
| 20 | 8,192 | 32 KB (0.4% of dense) |
| 25 | 262,144 | 1 MB (0.4% of dense) |
| 30 | 8,388,608 | 32 MB (0.4% of dense) |

**Sparse State** (10% occupancy):
| Qubits | Non-zero | Memory |
|--------|----------|--------|
| 30 | 107M | 1.6 GB (20% of dense) |
| 40 | 110B | 1.6 TB (0.15% of dense) |
| 50 | 113T | (not practical) |

### 6. Benchmark Suite

**Example Benchmarks** (`examples/`):

**Performance Benchmarks**:
- `comprehensive_bench.rs`: Full backend comparison
- `cpu_vs_gpu_comparison.rs`: CPU vs GPU scaling analysis
- `epic103_fusion_scaling.rs`: Gate fusion optimization
- `epic114_fp8_bench.rs`: FP8 Tensor Core benchmark
- `sparse_vs_dense_benchmark.rs`: Sparse state performance

**Scaling Benchmarks**:
- `w_state_gpu_max.rs`: Billion-qubit W state
- `dicke_qubit_limit.rs`: Dicke state scaling limit (marked slow, #[ignore])
- `blackwell_tuning.rs`: RTX 5090 optimization

**Algorithm Benchmarks**:
- `vqe_h2_molecule.rs`: VQE molecular ground state
- `qaoa_maxcut.rs`: QAOA MaxCut optimization
- `grover_demo.rs`: Grover's search demonstration

**Running Benchmarks**:
```bash
# Single benchmark
cargo run --release --example comprehensive_bench --features cuda,perf-bench

# All examples
cargo run --release --examples --features cuda,perf-bench

# With timing
hyperfine 'cargo run --release --example epic114_fp8_bench --features cuda'
```

---

## Python Integration

### 1. Python Bindings Architecture

**Build System**: Maturin (PyO3-based)

**Package Structure**:
```
tileuniverse/
├── __init__.py           # Main API, exports all public classes
├── algorithms.py         # Pre-built quantum algorithm wrappers
└── rl/                   # Reinforcement Learning integration
    ├── __init__.py
    ├── quantum_agent.py  # QuantumSNNAgent for Gymnasium
    ├── vec_env.py        # Vectorized environment wrapper
    └── sb3_wrapper.py    # Stable-Baselines3 integration
```

**Rust FFI** (`python/src/lib.rs`):
```rust
use pyo3::prelude::*;

#[pyclass]
struct QuantumSimulation { ... }

#[pymethods]
impl QuantumSimulation {
    #[new]
    fn new() -> Self { ... }

    fn register_qdemo(&mut self, x: i32, y: i32, n_qubits: u32, gates: Vec<PyQGate>, seed: u64) { ... }

    fn tick_n(&mut self, n: u32) { ... }
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<QuantumSimulation>()?;
    m.add_class::<QuantumSNN>()?;
    m.add_function(wrap_pyfunction!(create_fast_ghz_state, m)?)?;
    Ok(())
}
```

### 2. Main API (`tileuniverse/__init__.py`)

**Classical Parallel Worlds**:
```python
import tileuniverse as tu
import numpy as np

# Create engine with 5 parallel worlds
engine = tu.Engine(worlds=5, size=(512, 512), ruleset="gol")

# Evolve for 100 ticks
engine.evolve(100)

# Get world state as numpy array
world = engine.get_world(0)  # Shape: (512, 512), dtype: uint8
```

**Quantum Simulation**:
```python
from tileuniverse import QuantumSimulation, QGate

sim = QuantumSimulation()

# Register quantum demo: 2 qubits at tile (50, 50)
gates = [
    QGate.h(0),           # Hadamard on qubit 0
    QGate.cnot(0, 1),     # CNOT(0 → 1)
    QGate.measure(0),     # Measure qubit 0
    QGate.measure(1),     # Measure qubit 1
]
sim.register_qdemo(50, 50, 2, gates, seed=42)

# Run 4 ticks (one gate per tick)
sim.tick_n(4)

# Get measurement results
result = sim.get_logic(50, 50)  # Returns boolean (for single-bit result)
```

**Fast Sparse States**:
```python
import tileuniverse as tu

# O(1) GHZ state - works for any qubit count
ghz = tu.create_fast_ghz_state(1_000_000_000)  # 1 billion qubits
verified = ghz.verify()  # GHZStateVerification
print(f"Qubits: {verified.n_qubits}, Valid: {verified.is_valid}")

# Vec-based W state - billion-qubit capable
w_state = tu.create_fast_w_state(1_000_000_000)
verified = w_state.verify()  # WStateVerification
print(f"Qubits: {verified.n_qubits}, Valid: {verified.is_valid}")
```

**Backend Selection**:
```python
import tileuniverse as tu

# Check CUDA availability
if tu.is_cuda_available():
    print("CUDA backend available")
    tu.set_backend("cuda")
else:
    tu.set_backend("scalar")

# Automatic backend selection (recommended)
tu.set_backend("auto")  # Uses heuristics from fusion.rs
```

### 3. Quantum-SNN Integration

**QuantumSNN Configuration**:
```python
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode

# Configure quantum-SNN hybrid
config = QuantumSNNConfig(
    n_inputs=4,               # Input spike sources
    hidden_layers=[32],       # SNN hidden layers
    n_outputs=3,              # Action space size
    mode=InterferenceMode.triggered(),  # Best performance mode
    ticks_per_decision=20,    # SNN ticks before quantum measurement
)

brain = QuantumSNN(config, seed=42)
```

**Interference Modes**:
```python
# Epsilon-greedy (classical baseline)
mode = InterferenceMode.epsilon_greedy(epsilon=0.1)

# Continuous interference (quantum every tick)
mode = InterferenceMode.continuous()

# Triggered interference (quantum at decision points) - BEST
mode = InterferenceMode.triggered()
```

**Decision & Learning**:
```python
# Encode observations as spike rates [0-255]
spike_rates = [128, 64, 200, 50]  # Example: 4 inputs
action = brain.decide(spike_rates)  # Returns action index [0, n_outputs)

# Learn from reward [-127, 127]
brain.learn(reward=50)  # Positive reward

# End of episode (reset eligibility traces, etc.)
brain.end_episode()
```

### 4. Gymnasium RL Integration (`rl/quantum_agent.py`)

**QuantumSNNAgent**:
```python
import gymnasium as gym
from tileuniverse.rl import QuantumSNNAgent

# Create environment
env = gym.make("MountainCar-v0")

# Create quantum-SNN agent (auto-configures for env)
agent = QuantumSNNAgent.for_env(
    env,
    mode="triggered",        # Interference mode
    hidden_layers=[32],      # SNN architecture
    ticks_per_decision=20,   # SNN simulation depth
)

# Training loop
total_reward = 0
for episode in range(1000):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Agent decides action
        action = agent.act(obs)

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Agent learns from reward
        agent.learn(reward)
        episode_reward += reward

    # End of episode
    agent.end_episode()
    total_reward += episode_reward

    if episode % 100 == 0:
        print(f"Episode {episode}, Avg Reward: {total_reward / 100:.2f}")
        total_reward = 0
```

**Supported Environments**:
- Classic Control: CartPole-v1, MountainCar-v0, Acrobot-v1
- Box2D: LunarLander-v2
- Atari: (with appropriate observation preprocessing)

**Observation Encoding**:
```python
# Automatic encoding (default)
agent = QuantumSNNAgent.for_env(env, obs_encoding="auto")

# Manual encoding
def custom_encoder(obs):
    # Map observation to spike rates [0-255]
    spike_rates = np.clip(obs * 128 + 128, 0, 255).astype(np.uint8)
    return spike_rates.tolist()

agent = QuantumSNNAgent.for_env(env, obs_encoder=custom_encoder)
```

### 5. Stable-Baselines3 Integration (`rl/sb3_wrapper.py`)

**TileUniverseSB3VecEnv**:
```python
from tileuniverse.rl import TileUniverseSB3VecEnv
from stable_baselines3 import PPO

# Vectorized environments (64 parallel worlds)
env = TileUniverseSB3VecEnv(worlds=64, size=(32, 32), ruleset="gol")

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    env.render()
```

### 6. Algorithm Wrappers (`algorithms.py`)

**Pre-built Quantum Algorithms**:
```python
from tileuniverse.algorithms import grover, deutsch_jozsa, bernstein_vazirani, bell_state

# Grover's search
result = grover(n_qubits=3, target=5, backend="cuda", seed=42)
print(f"Measured: {result.measured}, Success: {result.success}")

# Deutsch-Jozsa
result = deutsch_jozsa(n_qubits=4, oracle_type="balanced")
print(f"Oracle is: {result.oracle_type}")

# Bernstein-Vazirani
result = bernstein_vazirani(n_qubits=5, secret=0b10110)
print(f"Recovered secret: {result.measured:05b}")

# Bell state preparation
result = bell_state(bell_type="phi_plus")  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
print(f"Measurements: {result.measurements}")
```

### 7. Building Python Bindings

**Installation**:
```bash
cd python

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install maturin
pip install maturin numpy pytest gymnasium

# Build and install (development mode)
maturin develop --release

# Build wheel for distribution
maturin build --release
```

**Testing Python Bindings**:
```bash
cd python

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_quantum.py::test_grover -v

# With output
pytest tests/ -v -s
```

**Importing in Python**:
```python
import tileuniverse as tu

# Check version
print(tu.__version__)  # '0.4.0'

# List available items
print(dir(tu))
```

---

## Development Guide

### 1. Build System

**Rust Edition**: 2024 (bleeding edge)

**Cargo Workspace**:
```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    ".",
    "crates/logic-fabric-core",
    "python",
]
```

**Feature Flags**:

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cuda` | CUDA GPU backend | CUDA Toolkit 12.0+ |
| `quantum_jit` | JIT compilation infrastructure | - |
| `cranelift_jit` | Cranelift JIT backend | cranelift 0.104 |
| `perf-bench` | Benchmarking APIs | - |
| `cluster` | Distributed/cluster mode | tokio, serde |
| `serde` | Serialization for transport | serde, bincode |
| `config` | JSON/YAML config file support | serde_json, serde_yaml |
| `debug-signals` | Debug signal tracing | - |
| `burn-compute` | Burn/CubeCL integration | burn, cubecl |
| `visualizer` | Macroquad visualization | macroquad |
| `gpu-prototype` | GPU voxel engine | vulkano, wgpu |
| `proof-mode` | Deterministic proof mode | - |

**Build Commands**:
```bash
# Minimal build (scalar backend only)
cargo build

# With CUDA
cargo build --features cuda

# With JIT
cargo build --features quantum_jit,cranelift_jit

# Full feature set
cargo build --release --features cuda,quantum_jit,cranelift_jit,perf-bench,cluster,config

# Specific binary
cargo build --release --bin bench_engine --features perf-bench,cuda

# Check without building (faster)
cargo check --all-features
```

### 2. Testing

**Test Organization**:

**Unit Tests** (inline in modules):
```rust
// src/quantum.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_gate() {
        let mut state = QState::new(1);
        apply_h(&mut state, 0);
        // Assert state is (|0⟩ + |1⟩)/√2
    }
}
```

**Integration Tests** (`tests/` directory):
```rust
// tests/grover_integration.rs
#[test]
fn test_grover_2_qubits() {
    let result = run_grover(2, 3, BackendMode::Scalar, Some(42));
    assert_eq!(result.measured, 3);
    assert!(result.success);
}
```

**Running Tests**:
```bash
# All tests
cargo test

# Specific module
cargo test quantum::tests

# Specific test
cargo test test_hadamard_gate

# Integration tests only
cargo test --test grover_integration

# With output (see println!)
cargo test -- --nocapture

# Parallel execution (default)
cargo test -- --test-threads=8

# Single-threaded (for debugging)
cargo test -- --test-threads=1

# Ignored tests (slow benchmarks)
cargo test -- --ignored
```

**Test Coverage**:
- Unit tests: ~200+ tests across modules
- Integration tests: ~60+ tests in tests/
- Python tests: ~30+ tests in python/tests/

### 3. Code Style & Conventions

**Rustfmt** (auto-formatting):
```bash
# Format all code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check
```

**Clippy** (linting):
```bash
# Run linter
cargo clippy

# Fix auto-fixable lints
cargo clippy --fix

# Strict mode (deny warnings)
cargo clippy -- -D warnings
```

**Naming Conventions**:

| Item | Convention | Example |
|------|------------|---------|
| **Modules** | snake_case | `quantum_jit`, `qec` |
| **Structs** | PascalCase | `QState`, `QuantumSNN` |
| **Enums** | PascalCase | `QGate`, `BackendMode` |
| **Enum Variants** | PascalCase | `QGate::CNot`, `InterferenceMode::Triggered` |
| **Functions** | snake_case | `run_grover`, `apply_h` |
| **Constants** | SCREAMING_SNAKE_CASE | `DEFAULT_EPSILON`, `MAX_QUBITS` |
| **Type Parameters** | Single uppercase | `T`, `E`, `K`, `V` |

**Documentation Comments**:
```rust
/// Runs Grover's search algorithm.
///
/// # Arguments
/// * `n_qubits` - Number of qubits (search space 2^n)
/// * `target` - Target state to find [0, 2^n)
/// * `backend` - Execution backend (Scalar, CUDA, etc.)
/// * `seed` - RNG seed for measurement
///
/// # Returns
/// `GroverResult` containing measured state and success flag
///
/// # Example
/// ```
/// let result = run_grover(3, 5, BackendMode::Scalar, Some(42));
/// assert_eq!(result.measured, 5);
/// ```
pub fn run_grover(n_qubits: u32, target: u32, backend: BackendMode, seed: Option<u64>) -> GroverResult
```

### 4. Error Handling

**Result Types**:
```rust
pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug)]
pub enum EngineError {
    InvalidQubitIndex(u32),
    BackendNotAvailable(String),
    CudaError(String),
    QECError(String),
    ParseError(String),
}

impl std::error::Error for EngineError {}
impl std::fmt::Display for EngineError { ... }
```

**Usage**:
```rust
pub fn apply_gate(state: &mut QState, gate: &QGate) -> Result<()> {
    match gate {
        QGate::H(q) => {
            if *q >= state.n_qubits {
                return Err(EngineError::InvalidQubitIndex(*q));
            }
            apply_h(state, *q);
            Ok(())
        }
        _ => todo!(),
    }
}
```

### 5. Adding New Features

**Workflow**:

**1. Add New Quantum Gate**:
```rust
// Step 1: Add variant to QGate enum (quantum.rs)
pub enum QGate {
    // ... existing gates
    NewGate(u32, f32),  // Example: qubit index, parameter
}

// Step 2: Implement scalar kernel
pub fn apply_new_gate_scalar(state: &mut QState, q: u32, param: f32) {
    // Implementation
}

// Step 3: Add to fusion rules (fusion.rs)
impl Fusable for QGate {
    fn try_fuse(&self, other: &QGate) -> Option<QGate> {
        match (self, other) {
            (QGate::NewGate(q1, p1), QGate::NewGate(q2, p2)) if q1 == q2 => {
                Some(QGate::NewGate(*q1, p1 + p2))  // Example: parameter addition
            }
            _ => None,
        }
    }
}

// Step 4: Add CUDA kernel (cuda.rs)
const NEW_GATE_PTX: &str = r#"
    // CUDA PTX code
"#;

pub fn apply_new_gate_cuda(state: &CudaState, q: u32, param: f32) {
    // CUDA kernel launch
}

// Step 5: Add parity test (tests/compare3_parity.rs)
#[test]
fn test_new_gate_parity() {
    let mut scalar = QState::new(5);
    let mut cuda = QState::new(5);

    apply_new_gate_scalar(&mut scalar, 2, 0.5);
    apply_new_gate_cuda(&mut cuda, 2, 0.5);

    assert_states_equal(&scalar, &cuda, 1e-6);
}
```

**2. Add New Algorithm**:
```rust
// Create new file: src/algorithms/new_algorithm.rs
use crate::quantum::{QState, QGate};

pub struct NewAlgorithmResult {
    pub output: u32,
    pub iterations: usize,
}

pub fn run_new_algorithm(n_qubits: u32, params: NewAlgorithmParams) -> NewAlgorithmResult {
    // 1. Initialize state
    let mut state = QState::new(n_qubits);

    // 2. Build circuit
    let circuit = build_circuit(n_qubits, params);

    // 3. Execute
    for gate in circuit {
        state.apply_gate(&gate);
    }

    // 4. Measure
    let output = state.measure_all();

    NewAlgorithmResult { output, iterations: circuit.len() }
}

fn build_circuit(n_qubits: u32, params: NewAlgorithmParams) -> Vec<QGate> {
    // Circuit construction logic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_algorithm() {
        let result = run_new_algorithm(3, NewAlgorithmParams::default());
        // Assertions
    }
}
```

```rust
// Update src/algorithms/mod.rs
pub mod grover;
pub mod deutsch_jozsa;
pub mod new_algorithm;  // Add this line

pub use new_algorithm::{run_new_algorithm, NewAlgorithmResult};
```

**3. Add Python Binding**:
```rust
// python/src/lib.rs
#[pyfunction]
fn run_new_algorithm_py(n_qubits: u32, param: f64) -> PyResult<PyNewAlgorithmResult> {
    let result = engine::algorithms::new_algorithm::run_new_algorithm(
        n_qubits,
        NewAlgorithmParams { param },
    );

    Ok(PyNewAlgorithmResult {
        output: result.output,
        iterations: result.iterations,
    })
}

#[pyclass]
struct PyNewAlgorithmResult {
    #[pyo3(get)]
    output: u32,
    #[pyo3(get)]
    iterations: usize,
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    // ... existing bindings
    m.add_function(wrap_pyfunction!(run_new_algorithm_py, m)?)?;
    Ok(())
}
```

```python
# python/tileuniverse/algorithms.py
from tileuniverse._core import run_new_algorithm_py

def new_algorithm(n_qubits, param=1.0):
    """
    Run the new algorithm.

    Args:
        n_qubits (int): Number of qubits
        param (float): Algorithm parameter

    Returns:
        NewAlgorithmResult with output and iterations
    """
    return run_new_algorithm_py(n_qubits, param)
```

### 6. Debugging

**Logging**:
```rust
// Enable debug logging
env_logger::init();

log::debug!("Applying gate {:?} to qubit {}", gate, qubit);
log::info!("Circuit executed in {:.2} ms", elapsed_ms);
log::warn!("Backend {} not available, falling back to Scalar", backend);
log::error!("CUDA error: {}", cuda_error);
```

```bash
# Run with logging
RUST_LOG=debug cargo run --example grover_demo
RUST_LOG=engine::fusion=trace cargo test test_fusion
```

**Guard Words** (buffer overrun detection):
```rust
// Enable strict mode
std::env::set_var("JIT_DEBUG_STRICT", "1");

// Will panic if buffer overrun detected
state.check_guards();
```

**CUDA Debugging**:
```bash
# CUDA error checking
CUDA_LAUNCH_BLOCKING=1 cargo run --features cuda

# cuda-gdb
cuda-gdb --args target/release/bench_engine

# nvidia-smi (monitor GPU usage)
watch -n 1 nvidia-smi
```

### 7. Performance Profiling

**Cargo Bench**:
```rust
// benches/quantum_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_hadamard(c: &mut Criterion) {
    let mut state = QState::new(20);
    c.bench_function("hadamard_20_qubits", |b| {
        b.iter(|| {
            apply_h(black_box(&mut state), black_box(10));
        });
    });
}

criterion_group!(benches, bench_hadamard);
criterion_main!(benches);
```

```bash
cargo bench --features cuda
```

**Flamegraph**:
```bash
cargo install flamegraph
cargo flamegraph --example comprehensive_bench --features cuda
# Generates flamegraph.svg
```

**perf (Linux)**:
```bash
cargo build --release --features cuda
perf record --call-graph=dwarf ./target/release/bench_engine
perf report
```

---

## Known Issues & Limitations

### 1. Grover's Algorithm 4+ Qubit Limitation

**Status**: Known, not blocking
**Severity**: Medium (workaround available)

**Problem**:
- Works correctly for 2-3 qubits (90-95% success rate) ✅
- Reduced success for 4+ qubits (< 50% success rate) ⚠️

**Root Cause**:
- Multi-Controlled-Z (MCZ) gate decomposition
- n ≥ 4 controls use approximate decomposition (6 CNOT gates)
- Approximation introduces phase errors that accumulate

**Workaround**:
- Use 3-qubit Grover for demonstrations (sufficient for proof-of-concept)
- For larger search spaces: Use amplitude amplification with ancilla qubits

**Future Fix** (not planned):
- Implement ancilla-based Toffoli ladder (exact decomposition)
- Requires additional qubits (n-2 ancilla for n-controlled gate)
- Lower priority (Grover is primarily educational, not production)

**References**:
- `KNOWN_ISSUES.md` (detailed explanation)
- `tests/grover_integration.rs` (test cases demonstrating limitation)

### 2. VQE UCCSD Ansatz Simplified

**Status**: Known, not blocking
**Severity**: Low (hardware-efficient ansatz works correctly)

**Problem**:
- UCCSD ansatz is simplified (3 gates for single excitation vs 8 in full implementation)
- Missing Jordan-Wigner parity strings, full CNOT ladder
- Results in ~0.72 Ha error for H₂ molecule (VQE: -1.86 Ha, FCI: -1.14 Ha)

**Root Cause**:
- Simplified implementation prioritizes performance over chemical accuracy
- Full UCCSD requires deep circuits (> 100 gates for 4 qubits)

**Workaround**:
- **Use hardware-efficient ansatz** (default, recommended) ✅
- Hardware-efficient is widely used in NISQ literature and works correctly

**Status of Hardware-Efficient Ansatz**:
- ✅ Single-qubit Hamiltonians: Perfect -1.0 convergence
- ✅ H₂ molecule: Converges to correct energy with sufficient layers
- ✅ Tested and validated in `tests/vqe_integration.rs`

**Future Fix** (not planned):
- Implement full UCCSD with parity strings (lower priority)
- Hardware-efficient ansatz is production-ready for NISQ applications

### 3. Limited Native Hardware Targets

**Status**: Design decision
**Severity**: Low

**Current Support**:
- IBM Quantum: ✅ SX, Rz, CX (Falcon, Heron)
- IonQ: ✅ GPI, GPI2, MS (Aria)
- Rigetti: ❌ Not implemented
- Google Sycamore: ❌ Not implemented

**Rationale**:
- IBM and IonQ cover superconducting and trapped-ion architectures
- Adding more vendors is straightforward (copy ibm.rs structure)
- Lower priority until user demand

**Adding New Vendor**:
```rust
// src/transpile/new_vendor.rs
pub struct NewVendorTranspiler;

impl NewVendorTranspiler {
    pub fn transpile(&self, circuit: &[QGate]) -> NewVendorCircuit {
        // Decompose to native gates
    }
}
```

### 4. Maximum Qubit Count (Dense State)

**Limitation**: 30 qubits (8 GB memory)

**Reason**:
- Dense state: 2^n complex amplitudes × 8 bytes = 2^(n+3) bytes
- 30 qubits = 8 GB
- 31 qubits = 16 GB (exceeds typical RAM)

**Workarounds**:
- **Sparse states**: Up to 64 qubits (element-sparse) or 40+ qubits (block-sparse)
- **Fast sparse**: Billion-qubit GHZ/W states for specific use cases
- **Stabilizer**: Millions of qubits for Clifford circuits (QEC)

**Future Work**:
- Multi-GPU distributed state (EPIC planned but not implemented)
- Out-of-core state (swap to disk for > 32 qubits)

### 5. Python Binding Limitations

**GIL (Global Interpreter Lock)**:
- Python bindings hold GIL during Rust execution
- No parallelism from Python multi-threading
- **Workaround**: Use multiprocessing or async I/O

**No Direct State Access**:
- Cannot directly access quantum state amplitudes from Python
- **Rationale**: State is 8 GB for 30 qubits (too large for Python)
- **Workaround**: Use measurement API to extract probabilities

**Limited Error Messages**:
- Some Rust errors panic instead of returning to Python
- **Workaround**: Use debug builds for better error messages

---

## Roadmap & Future Work

### Completed (71+ Sprints)

✅ **Core Quantum Engine** (Sprints 0-21)
✅ **CUDA GPU Backend** (Sprints 16-18)
✅ **Gate Fusion & Optimization** (Sprints 19-20, EPICs 83-84)
✅ **Quantum Algorithms** (Grover, DJ, BV, Shor, VQE, QAOA)
✅ **QEC Module** (Stabilizer, Steane, Surface codes, Union-Find)
✅ **QRAM** (Bucket-brigade → Fault-tolerant → Magic state budgeting)
✅ **Quantum-SNN Hybrid** (Triggered interference, +69% performance)
✅ **Hardware Transpilation** (IBM/IonQ native gates, T-depth optimization)
✅ **OpenQASM 3.0** (Full import/export, interoperability)
✅ **Resource Estimation** (NISQ and fault-tolerant QC)
✅ **FP8 Tensor Cores** (26× speedup on Hopper GPUs)
✅ **Python Bindings** (PyO3, Gymnasium RL, Stable-Baselines3)

**New Completions (Sprints 58-71)**:
✅ **Fault-Tolerant Processor** (LogicalQubit, PauliFrame, MagicSupply, Scheduler)
✅ **Error Mitigation** (ZNE rescue, REM 94% error reduction, combined pipeline)
✅ **Error-Aware Compilation** (Error propagation, T-depth minimization, Pareto optimization)
✅ **Cross-Block GPU CNOT** (210M block-ops/sec for 8.4M qubits)
✅ **MinimalGhzState** (O(1) GHZ up to 2^64 qubits, 4KB fixed memory)
✅ **R-STDP Learning** (Action-specific credit, winner-takes-all, MountainCar)
✅ **GPU-Fused SNN** (Zero-PCIe, 2:4/4:8 sparsity, EPIC 97)
✅ **Stabilizer Neural Networks** (87× speedup, GPU 7.98× at 100K neurons)
✅ **P-Bit Module** (Probabilistic computing, 15-26× GPU speedup)
✅ **Tensor Networks** (Matrix contraction 10-100×, beam search, slicing)
✅ **Physics-Logic Coupling** (Heat/charge/power, deterministic, GPU parity)
✅ **Quantum Search Engine** (Grover amplification 3.65×, 268M candidates)
✅ **PackedPauliFrame64** (64 parallel trajectories, AoSoA layout)
✅ **Magic State Scheduling** (4× speedup with factory pools)
✅ **Hardware Topology** (Dijkstra/A*, factory placement, congestion analysis)
✅ **GPU Ising Mode** (29.2× speedup, 1.2B updates/sec)
✅ **1-Bit Packed Tiles** (27.5T evals/sec, 138× improvement, EPIC 120)

### In Progress

🚧 **Multi-GPU Distributed State** (EPIC planned)
🚧 **GPU Voxel Engine** (EPIC 120 prototype exists)
🚧 **Advanced SNN Features** (Curiosity, TD credit - EPICs 121-123)

### Future Work (Not Scheduled)

**Short Term** (1-3 Sprints):
- [ ] Ancilla-based Toffoli decomposition (fix Grover 4+ qubits)
- [ ] Full UCCSD ansatz for VQE (chemical accuracy)
- [ ] Rigetti/Google native transpilers (vendor expansion)
- [ ] OpenQASM 2.0 full compatibility (legacy support)
- [ ] Jupyter notebook integration (visualizations)

**Medium Term** (3-6 Sprints):
- [ ] Quantum chemistry integration (OpenFermion, PySCF)
- [ ] Quantum machine learning primitives (QSVM, QNN)
- [ ] Noise simulation beyond Pauli errors (amplitude damping, crosstalk)
- [ ] Advanced QEC codes (Color code, Bacon-Shor)
- [ ] MWPM decoder (Blossom algorithm, compare vs Union-Find)
- [ ] Clifford+T synthesis (exact T-count minimization)

**Long Term** (6+ Sprints):
- [ ] Multi-GPU distributed state (> 30 qubits dense simulation)
- [ ] Out-of-core state (swap to disk/SSD for > 32 qubits)
- [ ] Cloud integration (AWS Braket, Azure Quantum)
- [ ] Real quantum hardware execution (via OpenQASM export)
- [ ] Quantum compiler optimization (QUEKO, t|ket⟩ integration)
- [ ] Production deployment tools (Docker, Kubernetes)

### Research Directions

**Quantum Algorithms**:
- Quantum walk algorithms (search, element distinctness)
- Quantum annealing (D-Wave integration)
- Quantum sampling (BosonSampling, Gaussian processes)

**Quantum-Classical Hybrid**:
- Quantum kernel methods for classical ML
- Quantum feature maps (ZZFeatureMap, Pauli rotation)
- Hybrid neural-quantum architectures

**Fault-Tolerant QC**:
- Surface code layout optimization (Austin, planar, 3D)
- Magic state factory scheduling (minimize latency)
- Logical gate compilation (Clifford+T synthesis, gridsynth)
- Physical error rate extrapolation (from NISQ to FT regime)

**Performance Optimization**:
- Blackwell GPU tuning (FP4 when available?)
- Custom CUDA kernels for specific circuits (JIT-compiled)
- Sparse state auto-tuning (adaptive threshold)
- Memory hierarchy optimization (L1/L2 cache, HBM)

---

## Appendix

### A. Glossary

**Quantum Computing Terms**:
- **Qubit**: Quantum bit, superposition of |0⟩ and |1⟩
- **Amplitude**: Complex coefficient of basis state
- **Entanglement**: Non-local correlation between qubits
- **Superposition**: Linear combination of basis states
- **Measurement**: Collapse to classical bit (probabilistic)
- **Clifford Gates**: Efficient subset (H, S, CNOT, CZ)
- **T Gate**: Non-Clifford gate, enables universal quantum computing
- **Magic State**: Resource for fault-tolerant T gates
- **Stabilizer**: Pauli operator preserving a state
- **Syndrome**: Error pattern detected by stabilizer measurement

**Architecture Terms**:
- **SoA (Structure-of-Arrays)**: Memory layout separating real/imag
- **SIMD**: Single Instruction Multiple Data (AVX2, AVX512)
- **JIT**: Just-In-Time compilation (Cranelift)
- **WMMA**: Warp Matrix Multiply-Accumulate (Tensor Cores)
- **PTX**: Parallel Thread Execution (CUDA assembly)
- **ILP**: Instruction-Level Parallelism

**TileUniverse-Specific**:
- **COPS**: Computational Ops Per Second (1 amplitude update)
- **PCOPS**: Peta-COPS = 10¹⁵ ops/sec
- **TCOPS**: Tera-COPS = 10¹² ops/sec
- **EPIC**: Major feature milestone (135+ EPICs completed)
- **SPRINT**: Development iteration (71+ sprints completed)
- **TILE-8**: Minimal 8-bit CPU architecture
- **CHUNGUS**: Codename for SNN module (CHUNGUS 4)
- **P-Bit**: Probabilistic bit with sigmoid activation
- **ZNE**: Zero-Noise Extrapolation (error mitigation)
- **REM**: Readout Error Mitigation
- **R-STDP**: Reward-modulated Spike-Timing-Dependent Plasticity
- **FT**: Fault-Tolerant (quantum computing)
- **MinimalGhzState**: O(1) GHZ state representation

### B. References

**Papers & Algorithms**:
- Aaronson & Gottesman (2004): Improved Simulation of Stabilizer Circuits
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information
- Delfosse & Nickerson (2017): Almost-linear time decoding for Surface codes
- Preskill (2018): Quantum Computing in the NISQ era and beyond

**Hardware Documentation**:
- IBM Quantum: Qiskit documentation, OpenQASM 3.0 spec
- IonQ Aria: Native gate set documentation
- NVIDIA: CUDA C++ Programming Guide, WMMA API reference

**Software Projects**:
- Qiskit: IBM's quantum computing framework
- Cirq: Google's quantum computing framework
- ProjectQ: ETH Zurich quantum compiler
- Stim: Fast stabilizer circuit simulator
- PyMatching: MWPM decoder for QEC

### C. File Size Summary

**Large Files (> 100 KB)**:
- `src/fusion.rs`: 331 KB (gate fusion logic)
- `src/cuda.rs`: 315 KB (GPU kernels)
- `src/algebraic_fusion.rs`: 310 KB (algebraic optimization)
- `crates/logic-fabric-core/src/quantum.rs`: 96 KB (core quantum)
- `crates/logic-fabric-core/src/block_sparse_state.rs`: 63 KB
- `src/qec/ft_processor.rs`: ~50 KB (fault-tolerant processor)
- `src/snn/gpu_fused.rs`: ~40 KB (GPU-fused SNN)
- `src/transpile/error_aware_compiler.rs`: ~35 KB (error-aware compilation)

**Total Codebase**:
- Rust source: ~2.5 MB (220+ files)
- Python bindings: ~60 KB (source), 1.5 MB (compiled _core.pyd)
- Documentation: ~300 KB
- Examples: ~700 KB
- Tests: ~400 KB (70+ integration tests)

### D. Contact & Contributing

**Repository**: (Add your Git repository URL here)

**Bug Reports**: File issues on GitHub with:
- Minimal reproducible example
- Rust version (`rustc --version`)
- CUDA version (if applicable, `nvcc --version`)
- Error messages (full stack trace)

**Feature Requests**: Open discussion with:
- Use case description
- Expected API/behavior
- Performance requirements (if applicable)

**Contributing**:
1. Fork repository
2. Create feature branch (`git checkout -b feature/new-algorithm`)
3. Write tests (unit + integration)
4. Run `cargo test` and `cargo fmt`
5. Submit pull request with description

**License**: (Add license information here - likely MIT or Apache 2.0)

---

**End of Comprehensive Documentation**

*This documentation reflects TileUniverse as of Sprint 71.0 (January 2026).
For the latest updates, check the repository and SPRINTS/ directory.*
