# TileAnneal Phase 1.5 Validation Report

**Date:** January 2026
**Status:** Complete - Reviewer Feedback Incorporated
**Hardware:** NVIDIA RTX 5090 (32GB, Blackwell)

---

## Executive Summary

TileAnneal is a GPU-resident Ising optimizer for **spatially embedded optimization problems**. It exploits the natural 2D locality of planar graphs to achieve extreme throughput on nearest-neighbor lattice problems.

> **Scope clarification:** TileAnneal v1 targets spatially embedded optimization (planar lattice graphs with uniform couplings). Arbitrary graphs require embedding and weighted edges are future work.

### Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Throughput | 10T spin-updates/sec | **12.25T/s** (122%) |
| Ferromagnetic ground state | 100% | **100%** |
| Planar MaxCut optimization | >95% of optimal | **96-98%** |
| Validation suite | All pass | **7/7 tests** |
| Run-to-run variance | Measured | **<0.5%** (see below) |

---

## 1. Scope and Applicability

### 1.1 What TileAnneal IS

- A **spatial optimization engine** for 2D lattice problems
- Absurdly fast on problems with **locality** (nearest-neighbor interactions)
- Ideal for **physics-derived**, **layout-constrained**, or **embedded** problems
- Examples: VLSI placement, protein folding lattice models, image segmentation

### 1.2 What TileAnneal is NOT

- A general MaxCut solver for arbitrary graphs
- A drop-in replacement for CPU simulated annealing
- A magic optimizer for non-planar or densely-connected graphs

### 1.3 Current Limitations

| Limitation | Status | Future Work |
|------------|--------|-------------|
| Uniform couplings only (J = ±1) | Phase 1 | Add J ∈ {-4...+4} in Phase 2 |
| 2D nearest-neighbor topology | By design | Graph embedding layer |
| No external field (h = 0) | Phase 1 | Add bias terms |

---

## 2. Variance Analysis

### 2.1 Quantitative Results (10 trials each)

| Benchmark | Mean | Std Dev | Min | Max | Trials |
|-----------|------|---------|-----|-----|--------|
| 64×64 MaxCut | 96.6% | 0.5% | 96.4% | 98.0% | 10 |
| 128×128 MaxCut | 98.2% | 0.1% | 98.2% | 98.4% | 10 |

### 2.2 Interpretation

With **best-solution tracking** (added per reviewer feedback), run-to-run variance dropped from ±15% to **<0.5%**. The remaining variance comes from:

1. GPU thread scheduling affecting RNG sequences
2. Multi-attempt selection (best of 3-5 runs)

This level of variance is acceptable for an optimization tool and is now a **measured property** rather than an unknown risk.

---

## 3. Best-Solution Tracking (Critical Fix)

### 3.1 The Problem

Previously, annealing returned the **final** configuration, not the **best** seen. This meant:
- Good solutions found at sweep 40,000 could be lost by sweep 100,000
- Reported quality depended on schedule luck
- Results were artificially worse than achievable

### 3.2 The Fix

Added `AnnealingResult` struct and `run_simulated_annealing_best()` function:

```rust
pub struct AnnealingResult {
    pub best_energy: i64,      // Best energy seen
    pub best_cut: usize,       // Best cut value seen
    pub best_sweep: u32,       // When best was found
    pub best_config: PackedTileGrid,  // Best configuration
    pub final_energy: i64,     // Final energy (may be worse)
    pub total_sweeps: u32,
}
```

The solver now:
1. Checks for improvement every 50-100 sweeps
2. Keeps a shadow copy of the best configuration
3. Returns best-seen, not final

### 3.3 Impact

| Metric | Before | After |
|--------|--------|-------|
| 128×128 MaxCut quality | 69-98% (high variance) | 98.2% ± 0.1% |
| Run-to-run consistency | Poor | Excellent |

---

## 4. Technical Implementation

### 4.1 Checkerboard Updates

> **Honesty note:** Checkerboard updates prioritize parallelism and optimization speed over exact equilibrium sampling. This is appropriate for optimization but not for statistical physics applications requiring detailed balance.

The 2D grid uses checkerboard decomposition:
- Phase 0: Update even sites (x+y even)
- Phase 1: Update odd sites (x+y odd)

Each phase updates half the spins simultaneously without conflicts.

### 4.2 Packed Bit Representation

Each spin = 1 bit, 64 spins per u64 word:
- **Memory:** 1B spins = 128 MB (vs 8 GB at 8 bytes/spin)
- **Throughput:** 12.25T spin-updates/sec

### 4.3 Cooling Schedule

Exponential cooling with configurable parameters:
```
β(t) = β_start × (β_end / β_start)^(t / T_total)
```

Typical parameters:
- β_start = 0.01-0.05 (hot start, accept most moves)
- β_end = 50.0 (cold finish, greedy descent)
- T_total = 100k-500k sweeps

---

## 5. Validation Suite

### 5.1 Test Summary

| Test | Description | Result |
|------|-------------|--------|
| 1 | Ground state stability (128×128) | ✅ E=-32512, \|M\|=1.0 |
| 2 | Energy monotonicity (256×256) | ✅ 60% of ground reached |
| 3 | Stochastic annealing ferro (128×128) | ✅ 100% of ground |
| 4 | Stochastic annealing ferro (256×256) | ✅ 100% of ground |
| 5 | Planted MaxCut single-temp (64×64) | ✅ 70.5% (baseline) |
| 6 | Planar MaxCut annealing (64×64) | ✅ 96.4% of optimal |
| 7 | Planar MaxCut annealing (128×128) | ✅ 98.2% of optimal |

### 5.2 Reproducibility

```bash
# Run full validation suite
cargo run --release --features cuda,perf-bench --bin bench_engine -- \
  --mode ising --validate

# Run throughput benchmark
cargo run --release --features cuda,perf-bench --bin bench_engine -- \
  --mode ising --width 16384 --height 16384
```

---

## 6. Performance Benchmarks

### 6.1 Throughput Scaling

| Grid Size | Spins | GPU Memory | Throughput | vs Target |
|-----------|-------|------------|------------|-----------|
| 8K × 8K | 67M | 16 MB | 12.03 T/s | 120% |
| 16K × 16K | 268M | 64 MB | 12.25 T/s | 122% |
| 32K × 32K | 1.07B | 256 MB | 12.15 T/s | 122% |

### 6.2 Practical Optimization Rates

| Problem Size | Sweeps/sec | Time for 1M sweeps |
|--------------|------------|-------------------|
| 100M spins | 122,489 | 8.2 seconds |
| 1B spins | 12,249 | 82 seconds |

---

## 7. Code Changes (This Report)

| File | Change | Lines |
|------|--------|-------|
| `src/cuda_tiles.rs` | Added `AnnealingResult` struct | +15 |
| `src/cuda_tiles.rs` | Added `run_simulated_annealing_best()` | +65 |
| `src/cuda_tiles.rs` | Updated validation to use best-tracking | +10 |

---

## 8. Remaining Work (Phase 2)

### 8.1 High Priority

1. **Adaptive annealing** - Adjust cooling rate based on acceptance ratio
2. **Weighted edges** - Support J ∈ {-4...+4} for richer problems

### 8.2 Medium Priority

3. **Parallel tempering** - Multiple replicas with temperature swaps
4. **External field** - Add bias terms (h ≠ 0)

### 8.3 Lower Priority

5. **Graph embedding** - Map arbitrary graphs to 2D lattice
6. **Multi-GPU** - Scale to 10B+ spins

---

## 9. Conclusion

Phase 1.5 validation is complete with all reviewer feedback addressed:

- ✅ **Best-solution tracking** implemented (correctness fix)
- ✅ **Variance measured** and reported (<0.5%)
- ✅ **Scope clarified** (planar/spatial MaxCut, not general)
- ✅ **Checkerboard disclaimer** added (optimization, not equilibrium)
- ✅ **7/7 tests pass** consistently

TileAnneal is ready for Phase 2: depth and robustness on real-world planar optimization problems.

---

*Report updated with reviewer feedback. January 2026.*
