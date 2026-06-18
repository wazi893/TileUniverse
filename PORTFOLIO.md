# TileUniverse — Systems & GPU Engineering Portfolio

A from-scratch simulation engine in **Rust + CUDA** where a CPU, a C compiler, a learned placement AI, and spiking neurons all run on simulated logic gates — **verified bit-exact**. Built solo over 3+ years (360+ sprints, 3,000+ passing tests, a published crate).

**Through-line:** if it computes, it computes in real tiles — and it's verified.

Design priorities throughout: **deterministic, reproducible, cross-backend parity.** Every number below is verified in code/benchmarks, not just asserted.

**Live demos:** [python/website/demos/](python/website/demos/) — five self-contained HTML players (tile CPU, cellular GPU, QEC decoder, AlphaFabric placement, SNN spike cascade). Real engine output, zero dependencies.

---

## At a glance (verified on RTX 5090 unless noted)

| Result | Number | Where |
|---|---|---|
| TileCpuV2 physical authority | **~99.8%** of execution in physical tiles (MUL 0.2% software) | `src/tile_cpu/` |
| Differential verification | **3,175+** tests, software ISS ↔ physical tile fabric | `tests/` |
| V2 CPU throughput (JIT build) | **~32,500 cycles/sec**, ~31 µs/cycle | Sprint 384 dirty-residue |
| V2→CUDA megakernel | **K bit-exact** V2 CPUs in parallel; **3.8×** dead-op elimination | `src/tile_cpu/v2_device_gpu.rs` |
| SIMT lane-packed settle | **37.9G** op-evals/s (Gate 3 GO) | `src/tile_cpu/v2_simt_gpu.rs` |
| AlphaFabric timing-aware placement | **−32.8%** criticality-weighted wirelength on 77-gate LIF neuron | `examples/snn_neuro_alphafabric.rs` |
| Cellular tile evaluation (1-bit, packed) | **115 trillion updates/sec**, ~0.016 bytes/tile (≈ bandwidth roofline) | `src/cuda_tiles.rs` |
| Quantum state-vector evolution (FP16 Tensor Core) | **15.8 TCOPS** @ 24 qubits | `crates/logic-fabric-core/src/cuda.rs` |
| Cross-backend numerical parity | scalar / AVX / JIT / CUDA agree within **ε ≈ 1e-6** | `tests/compare3_parity.rs` |

---

## 1. TileCpuV2 — a CPU built from logic gates, with a compiler

**`src/tile_cpu/`** — fetch / decode / execute / ALU / write-back as **physical tile circuits**, not an interpreter shortcut.

- **32-bit ISA** (32 opcodes, 16 registers, 128 ROM / 128 RAM cells) on a 128×640×16 tile grid.
- **C-like compiler toolchain**: lexer → recursive-descent parser → register allocation → codegen → runs **bit-identical** on a software ISS and the physical fabric.
- **Brainfuck interpreter**, recursion, and arrays execute on the simulated CPU — enforced by golden-hash differential testing.
- **~99.8% physical authority**: only MUL still has a software fallback; everything else is real tile evaluation.

**Result:** A source program compiles to instruction words, runs on gates, and prints `HELLO, TILE CPU!` — verified against the reference on every change.

**Demonstrates:** ISA design, compiler construction, formal differential verification at scale, and the discipline to stop writing registers in software once the fabric can hold them.

**Try it:** `cargo run --release --example v2_compiled_hello` · [tile CPU live player](python/website/demos/tilecpu.html)

## 2. AlphaFabric + HLS — learned placement and behavioral synthesis to tiles

**`src/synth/`** — AIG → NPN4 → placement → routing, with a learned/optimized placer (AlphaChip-style).

- Policy trained on small circuits **generalizes one-shot to unseen widths** at ~60% of naive wirelength — zero per-instance search.
- Beats simulated-annealing baseline by **21–44%** on wirelength; timing-aware objective cuts **criticality-weighted WL by 32.8%** on a spiking neuron layout.
- **Correctness oracle**: a layout is legal only if the placed-and-routed circuit still computes the same function (golden-hash + reference-vs-tile differential) — the reward cannot be hacked.
- **HLS middle layer**: behavioral `Expr` AST → spatial tile datapath; Phase 3 adds sequential FSM acceleration for control-flow functions.

**Demonstrates:** EDA placement/routing, timing-driven optimization, synthesis pipeline automation, and hardware-oracle-gated ML rewards.

**Try it:** `cargo run --release --example eda_flow` · [AlphaFabric player](python/website/demos/alphafabric.html)

## 3. SNN-in-tiles — spiking neurons that compute on the fabric

**`src/snn/tile_lif.rs`** — leaky integrate-and-fire dynamics as **real tile circuits**, not a software neuron model.

- Full LIF: weighted synapses, Q0.8 arbitrary leak, absolute refractory — synthesized to AIG, placed, routed, ticked on tiles.
- **Tile == reference oracle**: exhaustive AIG truth tables + exported tile simulation; 19 `tile_lif` tests gate every change.
- **NeuroAlphaFabric**: the same timing-aware placer that optimizes adders lays out neuron datapaths — unroutable baselines become routable under SA while the oracle holds.

**Demonstrates:** neuromorphic hardware mapping, fixed-point datapath design, and the convergence of learned EDA with physical verification.

**Try it:** `cargo run --release --example snn_in_tiles` · [SNN player](python/website/demos/snn.html) · write-up: [SNN_IN_TILES_SUMMARY.md](SNN_IN_TILES_SUMMARY.md)

## 4. V2→CUDA megakernel — bit-exact CPU cycles on GPU

**`src/tile_cpu/v2_device_gpu.rs`** — K independent `TileCpuV2` instances advanced in parallel on CUDA, each cycle **bit-exact** vs the CPU `DeviceCpu` oracle.

Built incrementally with per-mechanism validation (ALU, flags, commit, run-to-halt on full charter). Recent perf work: dead-op elimination (**3.8×**), slot-pruning, K-saturation sweeps to ~65K lanes.

**Demonstrates:** SIMT correctness under exotic hardware constraints, oracle-driven GPU porting, and scaling verified execution paths rather than approximating them.

## 5. Register-resident packed-tile kernel — driving a kernel to the memory roofline

**`src/cuda_tiles.rs`** — `packed_tile_eval_register_kernel` and V2/V3 variants

A 1-bit cellular-automaton evaluator optimized in measured stages:

1. **Bit-packing** — 64 cells per 64-bit word; one `left | right | up | down` evaluates 64 tiles.
2. **Warp shuffles** for horizontal neighbors — zero-cost intra-warp communication.
3. **Register residency** — 32 rows per thread; vertical neighbors become register reads.

**Result:** **115 trillion cell-updates/sec** on a 1.07-billion-cell grid, **~0.016 bytes/tile** — within ~3% of the 1.8 TB/s bandwidth roofline. ~4.2× over the warp-shuffle baseline.

**Demonstrates:** roofline analysis, register-pressure/occupancy tuning, warp-level primitives, shared-memory halo exchange.

**Try it:** [cellular GPU player](python/website/demos/cellular.html) (~9T Game-of-Life updates/sec, bit-identical to CPU)

## 6. Quantum WMMA / Tensor Core backend — fast and correct

**`crates/logic-fabric-core/src/cuda.rs`** — FP16 **WMMA 16×16** gate application with pipelined accumulators. Scalar backend is authoritative; **ε ≈ 1e-6 parity** across CPU / AVX / JIT / GPU is a CI gate.

**Result:** **15.8 TCOPS** on RTX 5090; 2.5 TCOPS on RTX 4070.

**Demonstrates:** Tensor Core programming, mixed-precision numerics with bounded error, gate fusion, and accelerated paths kept provably correct.

## Honorable mentions

| Track | Highlight | Where |
|---|---|---|
| Sparse quantum GHZ | Unlimited-scale states with O(1) memory for GHZ; published crate | `crates/tileuniverse-quantum` |
| QEC surface-code decoder | Union-Find decoder visualization | [QEC player](python/website/demos/qec.html) |
| GPU-native sparse eval | Hierarchical dirty tracking + Blelloch prefix-sum worklist | `src/cuda_tiles.rs` *(architecture; throughput TBD)* |

---

## Reproducing the numbers

```bash
# Tile CPU — compile & run on physical gates
cargo run --release --example v2_compiled_hello

# EDA end-to-end (source → silicon → verified)
cargo run --release --example eda_flow

# SNN-in-tiles portfolio demo
cargo run --release --example snn_in_tiles

# Packed / register-resident cellular (115T/s)
cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3

# Export self-contained HTML players
cargo run --release --example v2_player_export
cargo run --release --example cellular_player_export

# Regenerate README demo GIF
python python/assets/make_demo_gif.py
```

Hardware: GPU results measured on RTX 5090 (and RTX 4070 where noted). See `docs/BENCHMARK_RTX5090.md`, `docs/EPIC_120_PACKED_TILES_REPORT.md`, and `docs/EPIC_122_REGISTER_RESIDENT_REPORT.md` for methodology.

---

## Skills demonstrated

**Systems / EDA:** ISA design · compiler construction · AIG synthesis · placement & routing · timing-driven optimization · differential hardware verification · property/oracle testing

**GPU / HPC:** CUDA C/C++ · Tensor Cores (WMMA) · warp-level primitives (shuffle, ballot) · register/occupancy tuning · roofline analysis · mixed-precision (FP16) · SIMT bit-exact porting · Rust FFI to CUDA · cross-backend numerical verification · SIMD (AVX2/AVX-512) · JIT (Cranelift) · benchmark-driven optimization