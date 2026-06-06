# RUTHLESS HPC ANALYSIS: RTX 4070 Laptop GPU

## Hardware Specs (RTX 4070 Laptop - AD106)
- **SMs:** 46
- **CUDA Cores:** 4608 (46 × 128 per SM, but actually 46 × 64 FP32 + 64 INT32 = mixed)
- **Tensor Cores:** 184 (4th gen, 4 per SM)
- **Memory Bandwidth:** 256 GB/s (GDDR6X, 128-bit bus)
- **L2 Cache:** 32 MB
- **Shared Memory:** 100 KB per SM (configurable)
- **FP16 Tensor TFLOPS:** ~233 TFLOPS (with sparsity), ~116 TFLOPS (dense)
- **Clock:** ~2.0 GHz boost

## Current Performance
- **Measured:** 1.28 TCOPS (amplitude ops/sec)
- **Claimed:** "Bandwidth saturated at 504 GB/s"

## PROBLEM #1: The 504 GB/s claim is WRONG

RTX 4070 Laptop has **256 GB/s** memory bandwidth, not 504 GB/s.
504 GB/s is the desktop RTX 4090!

Let's recalculate:
- 1.28 TCOPS = 1.28 × 10^12 amplitude ops/sec
- Each amplitude = 2 bytes (FP16)
- If we're reading AND writing each amplitude: 4 bytes per op
- Implied bandwidth: 1.28 × 10^12 × 4 = 5.12 TB/s ???

That's impossible. Something is wrong with the measurement.

## PROBLEM #2: WMMA Kernel is NOT Bandwidth-Bound

Looking at the kernel `wmma_multi_state_batched_ilp`:

```cuda
// LOAD from global → shared (once at start)
for (int i = lane; i < 256; i += 32) {
    buf0[i] = tile[i];
}

// Loop entirely in shared memory!
for (int d = 0; d < depth; d++) {
    wmma::load_matrix_sync(a_frag, bufs[X], 16);  // SHARED MEM LOAD
    wmma::mma_sync(...);                           // TENSOR CORE COMPUTE
    wmma::store_matrix_sync(bufs[Y], ...);        // SHARED MEM STORE
}

// STORE from shared → global (once at end)
for (int i = lane; i < 256; i += 32) {
    tile[i] = result[i];
}
```

The loop body does **ZERO global memory access**! All WMMA operations hit shared memory!

### Memory Traffic Analysis:
- **Global Load:** 256 × 2 bytes = 512 bytes per tile (once)
- **Global Store:** 256 × 2 bytes = 512 bytes per tile (once)
- **Total:** 1024 bytes per tile, regardless of depth

With 1024 states × 16 tiles × 1024 bytes = 16 MB per kernel launch.
At 256 GB/s, that's 16 MB / 256 GB/s = 62.5 microseconds minimum.

But we're running 100K depth, so:
- 100K × 16K tiles × 1 WMMA op = 1.6 billion WMMA ops
- Each WMMA is 16×16×16 = 4096 FMA ops = 8192 FLOPS
- Total: 1.6B × 8192 = 13.1 PFLOPS needed per kernel

At 116 TFLOPS, that's 13.1 PFLOPS / 116 TFLOPS = **113 seconds**!

But we measured ~5 seconds for 100K depth. That means we're only getting ~23 TFLOPS actual.

## THE REAL BOTTLENECK: SHARED MEMORY BANDWIDTH

WMMA fragments load from shared memory. Shared memory bandwidth is:
- 32 banks × 4 bytes × 2 GHz = 256 GB/s per SM
- But with bank conflicts, effective is much lower

Each WMMA load reads 256 elements × 2 bytes = 512 bytes.
At ~100 GB/s effective per SM, that's 5 ns per load.
With 46 SMs, total shared mem bandwidth = 4.6 TB/s theoretical.

The depth loop does:
- 1 wmma::load_matrix_sync (512 bytes from shmem)
- 1 wmma::mma_sync (pure compute)
- 1 wmma::store_matrix_sync (512 bytes to shmem)

Per iteration: 1024 bytes shmem traffic.
At 4.6 TB/s total: 1024 × 1.6B iters / 4.6 TB/s = 0.35 seconds theoretical.

But we're seeing 5 seconds. **~14× slower than shared memory bandwidth limit.**

## WHERE'S THE TIME GOING?

1. **WMMA Latency:** Each mma_sync has ~8 cycle latency even when pipelined
2. **Warp Scheduling:** Not enough concurrent warps to hide WMMA latency
3. **Shared Memory Bank Conflicts:** 16×16 layout may cause conflicts
4. **Register Pressure:** Fragment storage uses many registers, limits occupancy

## OPTIMIZATION OPPORTUNITIES

### 1. INCREASE OCCUPANCY
Current: 8 warps per block = 256 threads
SM can run up to 48 warps (1536 threads).
**We're at 17% occupancy!**

Try: 16 or 32 warps per block.

### 2. DOUBLE-BUFFERED GLOBAL LOADS
While processing tile N, prefetch tile N+1 from global memory.
Hides global memory latency completely.

### 3. PERSISTENT KERNEL
Launch once, process ALL tiles without returning to CPU.
Eliminates kernel launch overhead entirely.

### 4. WARP SPECIALIZATION
Some warps do loads, some do WMMA, some do stores.
Producer-consumer pipeline within the kernel.

### 5. LDMATRIX INSTEAD OF LOAD_MATRIX_SYNC
`ldmatrix` is faster than `wmma::load_matrix_sync` for some patterns.
Requires PTX-level programming.

### 6. LARGER TILES
16×16 is the minimum WMMA tile. Try 16×16×16 → accumulate with 32×8×16 or other shapes.
May get better tensor core utilization.

### 7. ASYNC COPY (cp.async)
Use cp.async for global→shared transfers to hide latency.
Available on SM 8.0+.

### 8. TENSOR MEMORY ACCELERATOR (TMA)
RTX 40 series has TMA for bulk async copies.
Can dramatically reduce load latency.

## VERDICT

**YOU ARE NOT HARDWARE-CAPPED. YOU'RE OCCUPANCY-CAPPED.**

The tensor cores can do ~116 TFLOPS.
You're getting ~23 TFLOPS.
That's **20% utilization**.

The path forward:
1. Increase occupancy (more warps per SM)
2. Hide latency (async operations, double buffering)
3. Use modern SM 8.9 features (TMA, cp.async)

## QUICK WIN: Test Higher Occupancy

Change this:
```cuda
let warps_per_block = 8u32;  // Current
```

To this:
```cuda
let warps_per_block = 16u32;  // 2× more warps
// or
let warps_per_block = 32u32;  // 4× more warps
```

And adjust shared memory accordingly. This alone could give 2-4× speedup.
