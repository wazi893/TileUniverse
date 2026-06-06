# EPIC 120: 1-Bit Packed Tile Evaluation

**Date:** January 2026
**Status:** Phase 1-2 Complete
**Hardware:** NVIDIA RTX 5090 (32GB, Blackwell)

---

## Executive Summary

We achieved **27.5 trillion tile evaluations per second** for a spatial boolean logic simulation — a **138× improvement** over the previous u64-per-tile baseline (200 billion/sec). This was accomplished by packing 64 spatial tiles into each 64-bit word, reducing memory traffic by 64×.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Throughput | 200B tiles/sec | 27.5T tiles/sec | **138×** |
| Memory per tile | 8 bytes | 0.125 bytes | **64×** |
| 1B tile grid memory | 8 GB | 128 MB | **64×** |

---

## Problem Statement

### Original System

TileUniverse simulates spatial logic circuits (inspired by Minecraft redstone). Each tile in a 2D grid can be a wire, AND gate, OR gate, etc. The simulation evaluates all tiles in parallel each tick.

**Previous architecture:**
- Each tile = 1 `u64` word (64 "parallel lanes")
- Memory per tile: 8 bytes for logic + 1 byte tile type + 16 bytes neighbors = ~48 bytes total
- Bottleneck: Memory bandwidth (1.8 TB/s on RTX 5090)
- Peak throughput: 200 billion tiles/sec

### The Insight

For boolean spatial logic (wire ON/OFF), 64 bits per tile is massive overkill. A single bit suffices. By packing 64 tiles into one u64 word:
- Memory traffic drops 64×
- Bitwise operations process 64 tiles simultaneously
- Cache efficiency dramatically improves

---

## Architecture

### Memory Layout

```
Row-major packed layout:
Row 0: [word0: tiles 0-63][word1: tiles 64-127][word2: tiles 128-191]...
Row 1: [word0: tiles W..W+63][word1: tiles W+64..W+127]...

Each u64 word = 64 consecutive horizontal tiles
Bit 0 = leftmost tile, Bit 63 = rightmost tile
```

### Grid Dimensions

| Grid | Tiles | Packed Size | Fits in L2? |
|------|-------|-------------|-------------|
| 8K×8K | 67M | 8 MB | Yes (96 MB L2) |
| 16K×16K | 268M | 33 MB | Yes |
| 32K×32K | 1.07B | 134 MB | Partially |

### Neighbor Access

**Horizontal neighbors (left/right):**
- Within word: bit shift (`<< 1` or `>> 1`)
- Across word boundary: inject edge bit from adjacent word
- Optimized with `__shfl_sync()` warp intrinsics (zero-cost communication)

**Vertical neighbors (up/down):**
- Adjacent rows in memory: `grid[idx - words_per_row]` and `grid[idx + words_per_row]`
- Relies on L2 cache for performance

### Wire Logic

```cuda
// Each thread processes 64 tiles (one u64 word)
output = left | right | up | down;  // OR all neighbors
```

For wire tiles, a tile turns ON if any neighbor is ON. This single bitwise OR processes 64 tiles in one instruction.

---

## Implementation

### CUDA Kernels

Three kernel variants were implemented:

| Variant | Description | Performance |
|---------|-------------|-------------|
| **Basic** | 2D thread grid, global memory for all neighbors | 25.4 T/s |
| **Shuffle** | Warp shuffles for horizontal neighbors | **27.3 T/s** |
| **SharedMem** | Shared memory caching for vertical neighbors | 20.8 T/s |

**Winner: Shuffle kernel** — simpler is faster because L2 cache handles vertical neighbors efficiently.

### Depth Batching

Multiple timesteps execute in a single kernel launch:
```cuda
for (int step = 0; step < depth; ++step) {
    // Compute all tiles
    // Ping-pong between buffers
    __syncthreads();
}
```

Benefits:
- Data stays hot in L2/registers
- Reduces kernel launch overhead
- Typical depth: 100 steps per launch

### Key Files

| File | Purpose |
|------|---------|
| `src/cuda_tiles.rs:1774-2025` | CUDA kernel source strings |
| `src/cuda_tiles.rs:2027-2165` | `PackedTileGrid`, `GpuPackedGrid` structs |
| `src/cuda_tiles.rs:2187-2343` | `run_packed_tile_eval_gpu()` launcher |
| `src/cuda_tiles.rs:2346-2386` | `benchmark_packed_tiles()` function |
| `src/bin/bench_engine.rs:3265-3450` | CLI benchmark mode |

---

## Results

### Throughput Scaling

| Grid Size | Tiles | Throughput | Notes |
|-----------|-------|------------|-------|
| 8K×8K | 67M | 25.6 T/s | Fits entirely in L2 |
| 16K×16K | 268M | **27.5 T/s** | Sweet spot |
| 32K×32K | 1.07B | 25.4 T/s | Exceeds L2, slight drop |

### Comparison to Baseline

```
Previous (u64-per-tile):    200,000,000,000 tiles/sec
Current (1-bit packed):  27,500,000,000,000 tiles/sec
                                            ─────────
Improvement:                                    138×
```

### Theoretical Analysis

RTX 5090 memory bandwidth: 1.8 TB/s

| Design | Bytes/tile | Theoretical Max |
|--------|-----------|-----------------|
| u64-per-tile | ~48 | 37B tiles/sec |
| 1-bit packed (naive) | 0.75 | 2.4T tiles/sec |
| 1-bit packed (L2 cached) | ~0.065 | 28T tiles/sec |

We're achieving **98% of the L2-cached theoretical maximum**.

---

## Benchmark Commands

```bash
# Basic kernel (8K×8K grid)
cargo run --bin bench_engine --features cuda,perf-bench --release -- \
  --mode packed-1bit --width 8192 --height 8192 --depth 100 --ticks 5000

# Shuffle kernel (recommended)
cargo run --bin bench_engine --features cuda,perf-bench --release -- \
  --mode packed-1bit --width 16384 --height 16384 --depth 100 --ticks 5000 --shuffle

# Shared memory kernel (for comparison)
cargo run --bin bench_engine --features cuda,perf-bench --release -- \
  --mode packed-1bit --width 8192 --height 8192 --depth 100 --ticks 5000 --smem
```

---

## Limitations & Trade-offs

### What We Lost

1. **Multi-bit signals**: No signal strength, only ON/OFF
2. **Per-tile types**: Current benchmark uses uniform wire tiles (no AND/OR/XOR variation)
3. **64-lane parallelism**: Can't evaluate 64 different circuits simultaneously anymore

### What We Gained

1. **64× memory efficiency**: 1B tiles in 128 MB vs 8 GB
2. **138× throughput**: 27.5T vs 200B tiles/sec
3. **Massive grids**: Can simulate billion-tile worlds in real-time
4. **Better mental model**: One tile = one spatial location (not 64 bundled wires)

### Open Questions

1. **Tile type support**: How to efficiently handle mixed tile types (AND, OR, XOR) without killing performance?
2. **Boundary conditions**: Current implementation uses zero-padding. Wrap-around? Reflective?
3. **Correctness validation**: No CPU reference implementation for packed tiles yet.
4. **Multi-GPU**: Linear scaling with NVLink?

---

## Future Directions

### Near-term: Multiple Tile Types

Add support for different logic gates while maintaining high throughput:

```cuda
// Option 1: Uniform blocks (all tiles in a region have same type)
// Option 2: Lookup table for tile types
// Option 3: Separate kernels per tile type, masked execution
```

### Medium-term: 8-bit Signal Strength

Extend from 1-bit to 8-bit per tile:
- Signal strength 0-255 (like redstone's 0-15 but finer)
- 8 tiles per u64 word (still 8× better than original)
- Enables weighted signal mixing
- Opens door to tensor core acceleration (INT8 matrix ops)

```
Memory: 1 byte per tile
Throughput estimate: 3-5T tiles/sec (8× less than 1-bit, but richer semantics)
```

### Long-term: Tensor Core Integration

Use NVIDIA tensor cores for signal processing:

| Approach | Description | Tensor Core Usage |
|----------|-------------|-------------------|
| INT8 mixing | Weighted neighbor combination | `mma.sync` for batch matmul |
| Learned routing | Neural network decides signal paths | INT8 inference |
| Hybrid layers | Wire tiles + compute tiles | Matrix ops at compute tiles |

---

## Conclusion

The 1-bit packed tile design achieves near-theoretical-maximum performance for boolean spatial logic simulation. At 27.5 trillion tiles/second, we can simulate a **billion-tile world at 25+ FPS** — opening possibilities for massive-scale virtual spatial computing that wasn't previously feasible.

The key insight was recognizing that for boolean logic, the previous 64-bit-per-tile design was 64× over-provisioned. By right-sizing the data representation, we achieved a 138× speedup with simpler code.

### Feedback Requested

1. **Architecture**: Is the packed bit layout optimal? Alternative representations?
2. **Tile types**: Best approach for supporting mixed logic gates?
3. **Tensor cores**: Is the 8-bit signal strength path worth pursuing?
4. **Applications**: What would you build with 27T tiles/sec?

---

## Appendix: Code Snippets

### Shuffle Kernel (Core Loop)

```cuda
for (int step = 0; step < depth; ++step) {
    unsigned long long current = src[idx];

    // Horizontal neighbors via warp shuffle (FREE!)
    unsigned long long left_neighbor = __shfl_up_sync(mask, current, 1);
    unsigned long long right_neighbor = __shfl_down_sync(mask, current, 1);

    // Handle warp boundaries
    if (lane == 0 && col > 0) left_neighbor = src[idx - 1];
    if (lane == 31 && col < words_per_row - 1) right_neighbor = src[idx + 1];

    // Compute neighbors with edge bit injection
    unsigned long long left = (current >> 1) | ((left_neighbor & 1ULL) << 63);
    unsigned long long right = (current << 1) | ((right_neighbor >> 63) & 1ULL);

    // Vertical neighbors (L2 cached)
    unsigned long long up = (row > 0) ? src[idx - words_per_row] : 0ULL;
    unsigned long long down = (row < height - 1) ? src[idx + words_per_row] : 0ULL;

    // Wire logic: OR all neighbors
    dst[idx] = left | right | up | down;

    // Ping-pong swap
    swap(src, dst);
    __syncthreads();
}
```

### Rust Benchmark Function

```rust
pub fn benchmark_packed_tiles(
    rt: &CudaRuntime,
    width: usize,
    height: usize,
    total_steps: u32,
    depth_per_launch: u32,
    warmup_steps: u32,
    variant: PackedKernelVariant,
) -> CudaResult<(u64, f64, f64, f64)>
```

---

*Report generated for EPIC 120 review. Contact: TileUniverse team*
