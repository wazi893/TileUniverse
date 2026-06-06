README.md
Quantum Kernel Engine

A deterministic, multi-backend quantum micro-kernel simulator with scalar, AVX2/AVX512, and JIT acceleration.

Overview

The Quantum Kernel Engine is a high-performance, single-threaded quantum simulation runtime focused on micro-kernel throughput, deterministic semantics, and cross-backend reproducibility. It is not a general-purpose quantum simulator; instead, it is designed to benchmark and validate quantum gate kernels under different CPU execution backends.

The engine provides:

A low-level Structure-of-Arrays (SoA) state layout for N-qubit amplitude vectors

Multiple execution backends:

Scalar (reference implementation)

AVX2 / AVX512 vectorized kernels

Cranelift JIT backend (feature-gated, with IR scaffolding)

Deterministic classical logic and measurement semantics

A flexible benchmarking suite, including Kernel Farm, Combined, and Compare modes

A complete parity-testing pipeline ensuring all backends produce identical amplitudes within a strict epsilon tolerance

The engine acts as an experimental HPC environment for developing and comparing quantum gate kernels under controlled, repeatable conditions.

Core Principles

Determinism First

Classical logic runs in a stable order every tick.

Measurement and collapse always run scalar and never use SIMD or JIT.

No multithreading, no hidden nondeterministic behavior.

Cross-Backend Parity

Scalar backend is the authoritative reference.

AVX2, AVX512, and JIT must match scalar amplitudes within ~1e-6.

One Gate Per Tile Per Tick

The core evolution step applies a single quantum gate per tile each tick.

Ensures consistent performance measurement and easy backend comparison.

Separation of Classical and Quantum Work

Classical pipelines remain untouched by quantum backend changes.

Quantum execution runs after classical tick stabilization.

Backends
1. Scalar Backend (Reference Implementation)

Pure Rust, simple loops.

Provides the authoritative amplitude evolution.

Used for all parity tests and fallbacks.

2. AVX2 Backend

Vectorized implementations for:

H (Hadamard)

X (bit-flip / swap)

Phase (complex multiply on upper half)

CNot (masked swap using 8-lane blocks)

Features:

Tail-safe masked lanes

No inner-loop branching

Stable SoA alignment and pair ordering

Automatic CPU feature detection

3. AVX512 Backend

Detection and routing implemented.

Currently routes to AVX2 until real AVX512 kernels land.

Maintains future compatibility without affecting current behavior.

4. JIT Backend (Cranelift)

Feature-gated under:

[features]
quantum_jit = []
cranelift_jit = []


Capabilities:

JIT module (src/quantum_jit.rs) with:

JitBackend, JitKernel

Kernel caching by program hash

IR scaffolding for H, Phase, and CNot

Valid no-op proof path using Cranelift’s ObjectModule

Currently routes through a placeholder function until real compiled kernels are implemented.

State Layout (SoA)

Each N-qubit tile uses:

real[] and imag[] arrays (aligned)

Length: len = 2^N

Pair iteration produces deterministic ordering for all kernels

Memory alignment supports SIMD-friendly loads/stores

Bench Modes
1. FullTick

Runs classical logic + one quantum gate per tile per tick.

Outputs:

Logic ops/sec

Estimated q_ops/sec via FLOP model

Dirty/clean ratios

2. QuantumBackendOnly

Runs only quantum kernels for backend comparison.

3. CombinedLogicQuantum

Executes classical logic and quantum kernels in one loop and prints:

Measured logic ops/sec

Measured quantum ops/sec

Estimated total ops/sec

4. KernelFarmBench

A scheduler-driven benchmarking mode.

Runs many kernel types across multiple tiles

Measures per-kernel ops/sec, q_ops/sec, and totals

Allows scalar vs AVX2 vs AVX512 vs JIT comparisons

Integrates with CSV output tooling

5. Compare Modes

--backend compare

Scalar vs AVX2

--backend compare3

Scalar vs AVX2 vs JIT

Outputs include per-backend q_ops/sec and speedup ratios.

Testing & Parity

The engine has a comprehensive test suite (~139 tests), covering:

H, X, Phase, and CNot kernels

Parity across scalar vs AVX2 vs JIT

2–4 qubit sequences across all backends

Kernel Farm backend routing

Bench output invariants

Tolerance: epsilon ≈ 1e-6

All tests pass.

CSV & Reporting Tools

Located in:
User notes/Project Glossary/

Included scripts:

qscalar_matrix.ps1 — Scalar ops/sec tables

farm_matrix.ps1 — Kernel Farm matrix generation

Generates CSVs for scalar, AVX2, AVX512, and JIT backends

Building
Scalar-only:
cargo build

Enable JIT backend (no IR yet):
cargo build --features quantum_jit

Enable Cranelift scaffolding:
cargo build --features cranelift_jit

Enable full JIT compilation pipeline:
cargo build --features quantum_jit,cranelift_jit

Bench Examples
Kernel Farm:
cargo run --release --bin bench_engine --features perf-bench \
  --mode KernelFarmBench --ticks 64 --warmup 256 --backend scalar

Scalar vs AVX2 vs JIT (compare3):
cargo run --release --bin bench_engine --features perf-bench,quantum_jit,cranelift_jit \
  --mode farm --backend compare3

Next Steps (EPIC 57+)

Implement real Cranelift IR loops for H

Swap run_jit_kernel to call compiled functions

Add parity tests for JIT vs scalar

Add AVX512 kernel implementations

Expand farm compare mode to chart real JIT performance

License

(Insert license information here.)

Contributing

Coming soon — contribution guidelines, coding standards, and architecture diagrams.

End of README

Whenever you're ready, I can now produce the: