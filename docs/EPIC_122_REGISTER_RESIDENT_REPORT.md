# EPIC 122: Register-Resident Ultra-High-Throughput Tile Evaluation

**Date:** January 2026
**Status:** Complete
**Hardware:** NVIDIA RTX 5090 (32GB, Blackwell)
**Live confirmation refresh:** 2026-07-03, repo `4eaca9b`; see
`benchmarks/results/FELLOWSHIP_BENCH_CONFIRMATION_2026-07-03.md`

---

## Executive Summary

The original EPIC 122 run recorded **115 trillion tile evaluations per second** for spatial
boolean logic simulation. A live 2026-07-03 confirmation on the current repo/driver now prints
**96.51T tiles/sec** in the full variant ladder and **97.07T tiles/sec** through the single-variant
CLI path. Treat 115T as historical; use ~97T as the current quote.

| Metric | EPIC 120 (Before) | EPIC 122 (After) | Improvement |
|--------|-------------------|------------------|-------------|
| Throughput | 23.21T tiles/sec (2026-07-03 shuffle) | 96.51T ladder / 97.07T CLI | **4.2x vs live shuffle; 485.3x vs repo u64 baseline** |
| Grid size | 16K x 16K | 32K x 32K | 4x more tiles |
| Effective bytes/tile | 0.065 | ~0.016 | **4x better** |

---

## The Problem

EPIC 120's shuffle kernel achieved 27.5T tiles/sec, hitting 98% of L2-cached theoretical maximum. Each depth iteration still read/wrote to memory:

```cuda
for (step = 0; step < depth; step++) {
    current = src[idx];           // Memory read (L2 cached)
    // compute...
    dst[idx] = output;            // Memory write
    swap(src, dst);
    __syncthreads();
}
```

Even with L2 caching, memory bandwidth limited throughput.

---

## The Solution: Register-Resident Tiling

**Key insight**: Keep ALL data in registers for the entire depth iteration. Memory traffic only at kernel start/end.

### Design

- Each thread holds **32 rows** of its column in registers
- Each warp (32 threads) processes a **32x32 block** of words = **65,536 tiles**
- **Horizontal neighbors**: warp shuffle (zero cost)
- **Vertical neighbors**: register access (zero cost for interior rows)
- Only halo rows need memory access per step

### Register Budget

- RTX 5090: 255 registers per thread max
- 32 u64 words = 64 registers (32 bits per register)
- ~60 registers for temps/control = well within budget

### Memory Traffic Analysis

| Approach | Memory per tile per step | Notes |
|----------|--------------------------|-------|
| EPIC 120 Shuffle | ~0.065 bytes | L2 cached |
| EPIC 122 Register V3 | ~0.016 bytes | Register-resident |
| **Reduction** | **4x less** | |

---

## Implementation

Three register-resident kernel variants were implemented:

### V1: 16 Rows Per Thread
```cuda
#define REG_ROWS 16
unsigned long long r[REG_ROWS];  // 16 rows in registers
// ... compute with vertical neighbors from registers
```
**Result**: 53-58T tiles/sec (2x baseline)

### V2: 8 Rows + Cooperative Halo Exchange
```cuda
#define REG_ROWS_V2 8
__shared__ unsigned long long halo_shared[2][32];  // Inter-warp exchange
// ... use shared memory for halo between warps
```
**Result**: 69-71T tiles/sec (2.5x baseline)

### V3: 32 Rows Per Thread (Winner)
```cuda
#define REG_ROWS_V3 32
unsigned long long r[REG_ROWS_V3];  // Maximum register utilization
// ... 30 of 32 rows pure register-only vertical access
```
**Result**: originally 100-115T tiles/sec; current 2026-07-03 confirmation is 96.51-97.07T tiles/sec.

---

## Benchmark Results

### 16K x 16K Grid (268M tiles)
| Variant | Throughput | vs Baseline |
|---------|------------|-------------|
| Shuffle (baseline) | 27.5T | 1.0x |
| Register V1 | 53.3T | 1.9x |
| Register V2 | 69.2T | 2.5x |
| **Register V3** | **101.0T** | **3.7x** |

### 32K x 32K Grid (1B tiles) - Current Live Confirmation
| Variant | Throughput | vs Shuffle |
|---------|------------|-------------|
| Shuffle (baseline) | 23.21T | 1.0x |
| Register V1 | 47.21T | 2.0x |
| Register V2 | 60.68T | 2.6x |
| **Register V3** | **96.51T** | **4.2x** |

The repo CLI single-variant path also confirmed Register V3 at **97.07T tiles/sec** with
`improvement_vs_u64: 485.3`.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/cuda_tiles.rs:2031-2350` | Register-resident kernel source |
| `src/cuda_tiles.rs:6363-6375` | PackedKernelVariant enum |
| `src/cuda_tiles.rs:6555-6635` | Kernel launch code |
| `examples/packed_register_benchmark.rs` | Benchmark harness |

---

## Usage

### CLI
```bash
cargo run --bin bench_engine --features cuda,perf-bench --release -- \
  --mode packed-1bit --width 32768 --height 32768 --depth 2000 -r3

# Flags:
# -r    : Register V1 (16 rows)
# -r2   : Register V2 (8 rows + coop)
# -r3   : Register V3 (32 rows) [RECOMMENDED]
```

### Example
```bash
cargo run --example packed_register_benchmark --features cuda,perf-bench --release \
  -- --width 32768 --height 32768 --depth 2000
```

---

## Theoretical Analysis

### Why Register-Resident Works

1. **Memory hierarchy**: Registers are ~100x faster than L2 cache
2. **Amortization**: Load once, compute 2000+ iterations, store once
3. **Warp shuffles**: Free horizontal neighbor communication
4. **Interior optimization**: 30 of 32 rows need zero memory access per step

### Bandwidth Math

RTX 5090: 1.8 TB/s memory bandwidth

| Design | Effective bytes/tile | Max Throughput |
|--------|---------------------|----------------|
| u64-per-tile | ~48 | 37B tiles/sec |
| 1-bit packed (naive) | 0.75 | 2.4T tiles/sec |
| 1-bit packed (L2) | 0.065 | 28T tiles/sec |
| **Register-resident** | **0.016** | **112T tiles/sec** |

The original report recorded 115T, slightly exceeding this simple theoretical estimate. The
2026-07-03 confirmation records 96.51-97.07T, below the estimate but still in the same
register-resident regime.

---

## Limitations

1. **Halo propagation**: Interior rows don't see changes from adjacent blocks within a kernel launch. For fully correct wave propagation across the entire grid, shorter depth batches are needed.

2. **Non-uniform logic**: Current kernels only implement wire (OR) logic. Mixed tile types would need per-tile branching.

3. **Register pressure**: 32 rows is near the practical limit. Going higher would reduce occupancy.

---

## Future Work

1. **Persistent kernels**: Launch once, run forever, eliminate launch overhead entirely
2. **Multi-GPU NVLink**: Linear scaling with multiple GPUs
3. **Mixed tile types**: Uniform block regions with same logic type
4. **Tensor Core integration**: INT8 matrix ops for weighted signal mixing

---

## Conclusion

EPIC 122 demonstrates that **register-resident tiling** is the key to breaking the memory bandwidth barrier. By keeping data in registers across thousands of iterations, the current confirmed throughput is about **97T tiles/sec** - enough to simulate a **billion-tile world near 90 FPS** on this benchmark.

The 4.2x improvement over EPIC 120 shows that there's still significant optimization headroom even after hitting "theoretical maximum" on a previous design. The lesson: question whether memory access is truly necessary, and redesign to eliminate it.

---

*Report generated for EPIC 122 review. Contact: TileUniverse team*
