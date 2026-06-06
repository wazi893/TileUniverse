# TileUniverse — GPU Performance Engineering Portfolio

A high-performance quantum & cellular simulation engine in **Rust + CUDA**, built solo over 3+ years (360+ sprints, 3,000+ passing tests, a published crate). This page highlights the **GPU/accelerated-computing** work — the part most relevant to performance-engineering, HPC, and ML-infrastructure roles.

Design priorities throughout: **deterministic, reproducible, cross-backend parity.** Every number below is verified in code/benchmarks, not just asserted.

---

## At a glance (verified on RTX 5090)

| Result | Number | Where |
|---|---|---|
| Cellular tile evaluation (1-bit, packed) | **115 trillion updates/sec**, ~0.016 bytes/tile (≈ bandwidth roofline) | `src/cuda_tiles.rs` |
| Quantum state-vector evolution (FP16 Tensor Core) | **15.8 TCOPS** @ 24 qubits | `crates/logic-fabric-core/src/cuda.rs` |
| Neural-net inference on Tensor Cores (WMMA) | **24.3× over CPU** (hot path), ~600M evals/sec | `src/gpu_nn/wmma_nn_v2.rs` |
| Cross-backend numerical parity | scalar / AVX / JIT / CUDA agree within **ε ≈ 1e-6** | `tests/compare3_parity.rs` |

---

## 1. Register-resident packed-tile kernel — *driving a kernel to the memory roofline*

**`src/cuda_tiles.rs`** — `packed_tile_eval_register_kernel` and its V2/V3 variants (~lines 2505–2900)

A 1-bit cellular-automaton evaluator optimized in measured stages:

1. **Bit-packing** — 64 cells per 64-bit word, so a single `left | right | up | down` instruction evaluates 64 cells at once.
2. **Warp shuffles** (`__shfl_up_sync` / `__shfl_down_sync`) for horizontal neighbors — free intra-warp communication; only lanes 0/31 fall back to a memory load at warp boundaries, with explicit edge-bit injection across word boundaries.
3. **Register residency** — hold up to 32 rows per thread in registers across the entire depth loop, so *vertical* neighbors become register reads. Inter-warp halos are exchanged through shared memory; ping-pong buffers avoid races.

**Result:** **115 trillion cell-updates/sec** on a 1.07-billion-cell grid, with memory traffic driven down to **~0.016 bytes/tile** — within ~3% of the card's 1.8 TB/s bandwidth roofline. ~4.2× over the already warp-shuffle-optimized baseline.

**Demonstrates:** roofline analysis, register-pressure/occupancy tuning against the 255-register limit, warp-level primitives, shared-memory halo exchange, depth-batching to amortize launch overhead.

## 2. Quantum WMMA / Tensor Core backend — *fast and correct*

**`crates/logic-fabric-core/src/cuda.rs`** — `get_wmma_kernels`, `MULTISTATE_WMMA_KERNEL` (~lines 2000–2740)

Quantum gate application is matmul-shaped, so it maps onto FP16 **WMMA 16×16 tiles** with pipelined accumulators to hide latency, batching **1,024+ states** to saturate the GPU. Correctness is non-negotiable for a simulator, so the scalar backend is the authoritative reference and **ε ≈ 1e-6 parity** across CPU / AVX / JIT / GPU is enforced as a CI gate.

**Result:** **15.8 TCOPS** on RTX 5090; 2.5 TCOPS on RTX 4070.

**Demonstrates:** Tensor Core (WMMA) programming, mixed-precision (FP16) numerics with bounded error, gate fusion at the IR level to cut kernel launches, and the discipline to keep an accelerated path provably correct.

## 3. Warp-shuffle optimization journey — *measured, iterative perf work*

**`src/cuda_tiles.rs`** — `packed_tile_eval_shuffle_kernel` (line 2312) → shared-mem → register V1/V2/V3

The full optimization ladder, with throughput kept at each step: basic → warp-shuffle (27.5 TCOPS) → shared-memory → register-resident (53 → 69 → 115 TCOPS). A clean before/after narrative with receipts.

**Demonstrates:** profiling-driven optimization, knowing which technique to reach for and why, and abandoning approaches that don't pay off.

## Honorable mention — GPU-native sparse evaluation *(architecture)*

**`src/cuda_tiles.rs`** — `GpuSparseContext`, hierarchical L0/L1/L2 dirty-tracking bitsets, a Blelloch prefix-sum worklist, and `__ballot_sync` warp-aggregated atomic propagation — designed to eliminate per-tick CPU↔GPU round-trips for billion-cell grids. *(Presented as design; throughput not yet benchmarked.)*

---

## Reproducing the numbers

```bash
# Packed / register-resident cellular (115 TCOPS)
cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3

# Sparse evaluation demo
cargo run --release --features cuda --example sparse_perf_check
```

Hardware: results above measured on an RTX 5090 (and RTX 4070 where noted). See `docs/BENCHMARK_RTX5090.md`, `docs/EPIC_120_PACKED_TILES_REPORT.md`, and `docs/EPIC_122_REGISTER_RESIDENT_REPORT.md` for full methodology.

---

## Skills demonstrated

CUDA C/C++ · Tensor Cores (WMMA) · warp-level primitives (shuffle, ballot) · register/occupancy tuning · roofline analysis · mixed-precision (FP16) · shared-memory tiling & halo exchange · GPU sparse/irregular workloads (prefix sum, warp-aggregated atomics) · Rust FFI to CUDA · cross-backend numerical verification · SIMD (AVX2/AVX-512) · JIT (Cranelift) · benchmark-driven optimization
