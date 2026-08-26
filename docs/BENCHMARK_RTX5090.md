# RTX 5090 Benchmark Results

**Date:** January 2026; live confirmation refresh 2026-07-03
**GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM, Blackwell SM120)
**Current sweep:** `benchmarks/results/FELLOWSHIP_BENCH_CONFIRMATION_2026-07-03.md`

## Verified Benchmarks

### Quantum Substrate (PureMMA Tensor Cores)

Benchmark: `cargo run --example rtx5090_benchmark --features cuda,perf-bench --release`

| Qubits | VRAM Used | TCOPS (PureMMA) | Notes |
|--------|-----------|-----------------|-------|
| 12 | 32 MB | 12.27 | |
| 16 | 128 MB | 13.00 | |
| 20 | 512 MB | 13.00 | |
| 24 | 2.0 GB | 13.12 | Prior January peak was 15.80 |
| 26 | 4.0 GB | **13.13** | Current peak |
| 28 | 4.0 GB | 13.07 | |
| 30 | 8.0 GB | 13.07 | |
| 31 | 16.0 GB | 12.98 | |
| 32 | 32.0 GB | 12.84 | Maximum qubits |

**Peak Performance:** 13.13 TCOPS at 26 qubits in the 2026-07-03 sweep. The old 15.8 TCOPS
headline drifted down and should be treated as a January snapshot, not a current quote.

### Cellular Substrate (Depth-Batched CUDA)

Benchmark: `cargo run --bin bench_engine --features cuda,perf-bench --release -- --mode depth-batch --worlds N --depth 50`

| Worlds | Throughput |
|--------|------------|
| 1 | 141B evals/sec |
| 5 | 191B evals/sec |
| 10 | **199B evals/sec** |
| 16 | 199B evals/sec |
| 32 | 198B evals/sec |

**Peak Performance:** ~200B logic evals/sec at 10+ worlds

### Comparison: RTX 5090 vs RTX 4070

| Substrate | RTX 4070 | RTX 5090 | Improvement |
|-----------|----------|----------|-------------|
| Quantum | 2.5 TCOPS | 13.13 TCOPS | **5.3×** |
| Cellular | 40B evals/sec | 200B evals/sec | **5×** |

## Hardware Specifications

- **GPU:** NVIDIA GeForce RTX 5090
- **Architecture:** Blackwell (SM120, compute_120)
- **VRAM:** 32 GB GDDR7
- **Memory Bandwidth:** ~1.8 TB/s
- **CUDA Cores:** 21,760
- **Tensor Cores:** 680 (5th gen)

## TILE-8 Visualizer Note

The visualizer demo with 268M CPUs previously reported "2.25T evals/sec" but this was a **miscalculation** in the throughput display. The verified cellular substrate performance using the depth-batch benchmark kernel is ~200B evals/sec.

The visualizer configuration:
- 268M CPUs on 32,742 × 32,742 grid (1.07B tiles)
- Limited by Vulkan texture size (32768×32768)
- Uses 24.1 GB of 32 GB VRAM

## Utilization Analysis

### Tensor Core Utilization

Current 2026-07-03 sweep:
- PureMMA peak: 13.13 TCOPS at 26 qubits
- Each amplitude op is approximately 8 FP16 ops (complex multiply-accumulate)
- Effective FLOPS: approximately 105 TFLOPS
- Estimated Tensor Core utilization: approximately 12.1% of the rough 870 TFLOPS peak below

January snapshot retained for comparison:

RTX 5090 theoretical peak (FP16 Tensor):
- 680 Tensor Cores × 512 FP16 ops/cycle × ~2.5 GHz = ~870 TFLOPS

PureMMA kernel measured:
- 15.8 TCOPS = 15.8 trillion amplitude ops/second
- Each amplitude op ≈ 8 FP16 ops (complex multiply-accumulate)
- Effective FLOPS: 15.8 × 8 ≈ **126 TFLOPS**

**Estimated Tensor Core Utilization: ~14.5%**

### Memory Bandwidth Utilization

The 2026-07-03 sweep did not include an Nsight/NCU memory profile. Use the measured TCOPS table
above as the evidence; the arithmetic below is a January snapshot, not a fresh profiler result.

RTX 5090 memory bandwidth: ~1.8 TB/s

At 24 qubits (peak performance):
- State size: 2^24 × 8 bytes = 128 MB
- Load + store per circuit: 256 MB
- At 15.8 TCOPS with depth 1000: ~256 MB / 0.063s ≈ 4 GB/s

**Memory is NOT the bottleneck** - we're compute-bound at high depths.

### Why Not Higher Utilization?

1. **Quantum state layout**: SoA layout requires non-coalesced memory access patterns
2. **Register pressure**: Complex numbers require 2× storage
3. **WMMA fragment overhead**: Loading/storing fragments has fixed latency
4. **Kernel launch overhead**: Amortized but not zero

### Profiling Notes

NCU (Nsight Compute) profiling requires admin privileges on Windows:
```
ERR_NVGPUCTRPERM - GPU Performance Counters permission required
```

To run full profiling:
1. Run as Administrator, OR
2. Set `NVreg_RestrictProfilingToAdminUsers=0` in registry

## Reproduction

```bash
# Quantum benchmark
cargo run --example rtx5090_benchmark --features cuda,perf-bench --release

# Cellular benchmark
cargo run --bin bench_engine --features cuda,perf-bench --release -- --mode depth-batch --worlds 10 --depth 50

# Honest quantum benchmark (raw throughput, no fusion multipliers)
cargo run --example honest_quantum_bench --features cuda,perf-bench --release
```
