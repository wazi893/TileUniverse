# SPRINT 81: Hierarchical Locality

**Status:** Design decisions locked
**Theme:** Factor problems into local + global, protect the fast path
**Duration:** Focused sprint (not feature-sprawl)

---

## Strategic Context

From technical review feedback:

> "Instead of asking 'How do we emulate all-to-all connectivity?', ask: 'How do we factor problems so their energy landscape is mostly local, with a small number of long-range corrections?'"

This sprint implements that insight. We're not "fixing" the lattice topology—we're embracing it as a feature and building structure on top.

---

## Objectives

### Primary: Hierarchical Hopfield
Build a two-level Hopfield network:
- **Level 0 (L0):** Dense local tiles, packed evaluation, fast
- **Level 1 (L1):** Boundary consistency layer, soft stitching between blocks

```
┌─────────────────────────────────────────────────────────────────┐
│  Hierarchical Hopfield                                          │
│                                                                 │
│  L1 (stitching):   ○───────○───────○───────○   (boundary only) │
│                    │       │       │       │                    │
│                    ▼       ▼       ▼       ▼                    │
│  L0 (local):     [███]   [███]   [███]   [███]  (packed, FAST) │
│                  8×8     8×8     8×8     8×8                    │
│                                                                 │
│  Total: 32×32 = 1024 neurons                                    │
│  L0 runs at packed speed (~97T tiles/sec, 2026-07-03)            │
│  L1 enforces boundary coherence between adjacent blocks         │
└─────────────────────────────────────────────────────────────────┘
```

### Secondary: Small-World Overlay (Stretch)
Add sparse long-range connections without breaking packed path.

### Tertiary: Multi-Resolution Grids (Stretch)
Coarse tiles that summarize fine tiles.

---

## Non-Goals (Protecting Focus)

- ❌ All-to-all connectivity emulation
- ❌ New tile types
- ❌ Changes to packed kernel inner loop
- ❌ Multi-GPU (save for Sprint 82)
- ❌ FPGA synthesis of hierarchical (save for later)
- ❌ Quantum integration

---

## Design Decisions (Locked)

### Decision 1: L0→L1 Projection — Majority Vote

**Chosen:** Majority per block.

**Rationale:** Parameter-free. Natural interpretation: "which attractor basin is
this block near?" Avoids tuning a threshold.

**Future extension:** Multi-neuron projection (magnetization + boundary coherence)
for richer L1 representation. Not in v1.

```rust
// Majority vote: count up-spins in block, compare to half
fn project_block(block: &PackedBlock) -> bool {
    let up_count = block.popcount();
    up_count > block.total / 2
}
```

### Decision 2: L1→L0 Correction — Boundary Bias with Distance Decay

**Chosen:** BoundaryBias with strength decaying from edge to center.

**Rationale:** Respects locality. Interior neurons find their own minimum;
only boundaries need global context. Analogous to domain decomposition in
PDE solvers—solve subproblems with boundary conditions from the global level.

UniformBias fights locality. Gating is too coarse (binary on/off).

```
Correction strength within an 8×8 block:

  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0    ← boundary row
  1.0  0.5  0.5  0.5  0.5  0.5  0.5  1.0
  1.0  0.5  0.0  0.0  0.0  0.0  0.5  1.0    ← center: no correction
  1.0  0.5  0.0  0.0  0.0  0.0  0.5  1.0
  1.0  0.5  0.0  0.0  0.0  0.0  0.5  1.0
  1.0  0.5  0.0  0.0  0.0  0.0  0.5  1.0
  1.0  0.5  0.5  0.5  0.5  0.5  0.5  1.0
  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0    ← boundary row
```

Corner neurons feel it strongest. Center feels nothing.

### Decision 3: Update Schedule — 4:1 L0:L1, L0-Heavy

**Chosen:** 4 L0 steps per 1 L1 step for recall. Configurable.

**Rationale:**
- L0 needs enough steps to approach local equilibrium before L1 reads
- Too few → L1 sees noise, makes bad corrections
- Too many → L0 gets stuck in local minima before L1 can guide

For annealing: hierarchical schedule (L1 settles at high temperature first,
then gradually couple L0).

```rust
pub struct HierarchicalConfig {
    /// L0 steps per L1 step (default: 4)
    pub l0_steps_per_cycle: u32,
    /// Base correction strength (decays with distance from boundary)
    pub correction_strength: f32,
    /// Beta (inverse temperature) for L0 blocks
    pub l0_beta: f32,
    /// Beta for L1 layer
    pub l1_beta: f32,
}
```

### Decision 4: L1 Architecture — Boundary Consistency Checker

**Chosen:** L1 does NOT store patterns. L1 enforces "adjacent blocks should
agree at their shared boundaries."

**Rationale:** This is simpler than full hierarchical Hopfield and gives ~80% of
the benefit. L1 becomes a soft stitching layer rather than a separate associative
memory. Less to go wrong, less to tune.

```
Block A boundary:  ...1 0 1    Block B boundary:  0 1 1...
                       │ │ │                       │ │ │
                       ▼ ▼ ▼                       ▼ ▼ ▼
L1 checks:         Do edges agree? → bias to align
```

This means:
- L1 coupling Jᵢⱼ is determined by boundary overlap, not stored patterns
- L1 has no Hebbian learning—it's purely structural
- The only "learning" is in L0 blocks (standard Hopfield within each block)

### Decision 5: Block Size — 8×8 primary, 16×16 benchmark

**Chosen:** Start with 8×8 blocks. Also benchmark 16×16.

**Rationale:** For 32×32 total grid, 8×8 blocks give 16 L1 neurons (4×4 grid
of blocks). Overhead is negligible. L1 cost scales as O(1/block_size⁴) for
all-to-all L1, so larger blocks are almost free.

### Decision 6: Pattern Encoding — Conditional Local Storage

**Chosen:** Two-level encoding.

```
To store pattern P:
  1. Coarse-grain: P_L1[b] = majority(P[block_b]) for each block b
  2. For each L0 block: store P[block_b] via standard Hebbian
  3. L1 stitching weights derived from boundary agreement of stored patterns
```

The key: L0 blocks memorize "local patch given boundary context from neighbors."
Not just the patch in isolation.

### Decision 7: Throughput Target — <1.5× slowdown

**Tightened from original 3×.** Since L1 is tiny (16 neurons for 32×32 grid),
overhead should be negligible if we batch enough L0 steps. If we're paying 3×
compute for 1.5× capacity, that's a bad trade.

### Decision 8: Benchmark Framing — By Pattern Type

Reframed from generic capacity to pattern-type-specific:

| Pattern Type | Flat Capacity | Hierarchical Capacity | Why |
|-------------|---------------|----------------------|-----|
| **Random** | ~0.14N | ≤ 0.14N (probably worse) | L1 bottleneck, no spatial structure to exploit |
| **Spatially correlated** | ? | Should beat flat | Blocks capture local structure, L1 provides coherence |
| **Compositional** (blocks reused) | ? | Should beat flat significantly | Same local patch in different global contexts |

If hierarchical doesn't beat flat on spatially correlated patterns, the
architecture is wrong and we should know that early.

---

## Phase 1: Implementation

Phase 1 design is done (above). Moving straight to implementation.

### 1.1 Data Structures

```rust
/// Hierarchical Hopfield network with boundary stitching
pub struct HierarchicalHopfield {
    /// Level 0: Local packed grids (one per block)
    l0_blocks: Vec<GpuIsingGrid>,

    /// L0 block dimensions (e.g., 8 for 8×8)
    block_size: usize,

    /// Grid of blocks
    blocks_x: usize,
    blocks_y: usize,

    /// Boundary correction mask (precomputed distance-decay)
    /// For each neuron in a block: correction weight [0.0, 1.0]
    correction_mask: Vec<f32>,

    /// L1 boundary coupling strengths
    /// Between adjacent block pairs (horizontal and vertical)
    l1_h_couplings: Vec<f32>,  // blocks_x-1 × blocks_y
    l1_v_couplings: Vec<f32>,  // blocks_x × blocks_y-1

    /// Configuration
    config: HierarchicalConfig,
}

pub struct HierarchicalConfig {
    pub l0_steps_per_cycle: u32,      // Default: 4
    pub correction_strength: f32,      // Default: 0.3
    pub l0_beta: f32,                  // L0 inverse temperature
    pub l1_beta: f32,                  // L1 inverse temperature
    pub block_size: usize,             // Default: 8
}
```

### 1.2 Kernels

**Kernel 1: L0 Block Update (EXISTING — no changes)**
- Standard packed Ising update
- Run on each block independently
- Fast path protected

**Kernel 2: Boundary Extraction**
```cuda
// Extract boundary spins from each L0 block
// For 8×8 block: 28 boundary neurons (perimeter)
__global__ void extract_boundaries(
    const uint64_t* l0_blocks,     // Packed block data
    uint32_t* boundary_spins,       // Output: boundary bits per block
    int block_size,
    int words_per_block_row
);
```

**Kernel 3: Boundary Agreement Check**
```cuda
// Compare boundaries between adjacent blocks
// Output: per-block bias direction (+1 or -1)
__global__ void check_boundary_agreement(
    const uint32_t* boundary_spins,  // All block boundaries
    float* correction_bias,           // Output: bias per block
    int blocks_x,
    int blocks_y,
    float correction_strength
);
```

**Kernel 4: Apply Boundary Correction**
```cuda
// Apply bias to L0 blocks with distance-decay mask
__global__ void apply_boundary_correction(
    uint64_t* l0_blocks,
    const float* correction_bias,
    const float* correction_mask,     // Precomputed decay weights
    int block_size,
    float beta                         // Temperature-modulated strength
);
```

### 1.3 Orchestration

```rust
impl HierarchicalHopfield {
    pub fn step(&mut self, rt: &CudaRuntime) -> CudaResult<StepStats> {
        // 1. Run L0 blocks (packed, fast) — bulk of compute
        for _ in 0..self.config.l0_steps_per_cycle {
            for block in &mut self.l0_blocks {
                run_ising_update(rt, block, 1,
                    IsingKernelVariant::Stochastic, -1.0)?;
            }
        }

        // 2. Extract boundaries from each block
        let boundaries = self.extract_boundaries(rt)?;

        // 3. Check agreement between adjacent blocks
        let corrections = self.check_agreement(rt, &boundaries)?;

        // 4. Apply boundary bias with distance decay
        self.apply_corrections(rt, &corrections)?;

        Ok(stats)
    }

    pub fn store_pattern(&mut self, pattern: &[i8]) -> Result<()> {
        // 1. Partition pattern into blocks
        // 2. Hebbian update per L0 block
        // 3. Compute boundary agreement structure
    }

    pub fn recall(&mut self, rt: &CudaRuntime,
                  corrupted: &[i8],
                  max_cycles: u32) -> CudaResult<Vec<i8>> {
        // 1. Load corrupted pattern into L0 blocks
        // 2. Run hierarchical steps until stable
        // 3. Read back converged state
    }
}
```

---

## Phase 2: Benchmarks

### 2.1 Throughput Comparison

| Configuration | Target | Fail Condition |
|--------------|--------|----------------|
| Flat 32×32 packed | Baseline | — |
| Hierarchical 16×(8×8), 4:1 ratio | <1.5× slowdown | >2× = redesign needed |
| Hierarchical 4×(16×16), 4:1 ratio | <1.3× slowdown | >1.5× = ratio too low |

### 2.2 Pattern Recovery by Type

**Test patterns:**

1. **Random** (control) — random ±1, no spatial structure
2. **Checkerboard variants** — lattice-natural patterns
3. **Stripes** (horizontal, vertical, diagonal)
4. **Block patterns** — each 8×8 block is uniform ±1 (tests L1 stitching)
5. **Compositional** — 4 different block types arranged in different global layouts
6. **Gradient** — smooth spatial transition (tests multi-scale)

For each: measure recall accuracy at 10%, 20%, 30%, 40% corruption.

### 2.3 Capacity Scaling

Store increasing numbers of spatially-correlated patterns until recall drops
below 90%. Compare flat vs hierarchical at each scale.

| Grid | Flat Patterns@90% | Hierarchical@90% | Improvement |
|------|-------------------|-------------------|-------------|
| 32×32 | ? | ? | ? |
| 64×64 | ? | ? | ? |
| 128×128 | ? | ? | ? |

### 2.4 Boundary Agreement Analysis

New diagnostic: track what percentage of inter-block boundaries agree over time.
This tells us whether L1 is actually doing useful work.

```
Cycle 0:  boundary agreement = 52% (near random)
Cycle 10: boundary agreement = 78%
Cycle 20: boundary agreement = 95%
Cycle 30: boundary agreement = 99% → converged
```

If boundaries converge quickly → L1 is working.
If boundaries don't improve → L1 architecture needs rethinking.

---

## Phase 3: Async L1 (If Phase 2 Succeeds)

Instead of strict 4:1 alternation, trigger L1 only when needed:

```rust
pub fn step_adaptive(&mut self, rt: &CudaRuntime) -> CudaResult<StepStats> {
    // 1. Run L0 steps
    for _ in 0..self.config.l0_steps_per_cycle {
        for block in &mut self.l0_blocks {
            run_ising_update(rt, block, 1, IsingKernelVariant::Stochastic, -1.0)?;
        }
    }

    // 2. Check if boundaries disagree
    let boundaries = self.extract_boundaries(rt)?;
    let disagreement = self.measure_disagreement(rt, &boundaries)?;

    // 3. Only correct if disagreement exceeds threshold
    if disagreement > self.config.disagreement_threshold {
        let corrections = self.check_agreement(rt, &boundaries)?;
        self.apply_corrections(rt, &corrections)?;
    }

    // When network is mostly converged, L1 stops firing → near-zero overhead
}
```

This makes L1 demand-driven. When the network is near convergence, L1 never
fires. Maximum efficiency.

---

## Phase 4: Small-World Overlay (Stretch)

If Phases 1-3 complete:

Add K long-range edges per neuron (K ∈ {1, 2, 4}):
- Stored as sparse adjacency list
- Evaluated in separate pass after local update
- Coupling strength weaker than local (e.g., 0.25×)
- Random but fixed topology (set at construction, not per-step)

---

## Success Criteria

### Must Have
- [ ] Hierarchical Hopfield runs on GPU
- [ ] L0 uses existing packed kernel (no modifications)
- [ ] Boundary stitching demonstrably improves block-pattern recovery
- [ ] Throughput within 1.5× of flat packed baseline
- [ ] Clean API: `HierarchicalHopfield::new()`, `.store_pattern()`, `.recall()`

### Should Have
- [ ] Benchmarks across all 6 pattern types
- [ ] Boundary agreement diagnostic
- [ ] Adaptive L1 (demand-driven)

### Nice to Have
- [ ] Small-world overlay
- [ ] Comparison at 64×64 and 128×128 scale
- [ ] Pattern visualization

---

## Technical Risks

| Risk | Mitigation |
|------|------------|
| L1 overhead exceeds 1.5× | L1 is tiny, batch more L0 steps |
| Boundary correction destabilizes L0 | Start with weak strength (0.1), anneal up |
| No capacity improvement on random patterns | Expected — reframe as "structure-dependent benefit" |
| Block boundaries create artifacts | Distance decay smooths transitions |
| Correction mask generation is complex | Precompute once at construction, store as GPU buffer |

---

## Implementation Order

1. `HierarchicalHopfield` struct + construction
2. Correction mask generation (CPU, precomputed)
3. Boundary extraction kernel
4. Boundary agreement kernel
5. Boundary correction kernel
6. `step()` orchestration
7. `store_pattern()` + `recall()` API
8. Benchmark harness
9. Pattern generators (6 types)
10. Run benchmarks, analyze, iterate

---

## References

- Ramsauer et al. (2020) "Hopfield Networks is All You Need"
- Hinton (2007) "Learning multiple layers of representation"
- Mezard & Montanari "Information, Physics, and Computation"
- Domain decomposition methods for PDEs (boundary condition analogy)
- Block spin renormalization group (projection analogy)

---

*Sprint 81: Boundary stitching. Protect the fast path. Let L0 do the heavy lifting.*
