# TileAnneal Phase 3: QUBO-Complete, Demonstrably Useful

**Date:** January 2026
**Status:** In Progress (Week 1-2 Complete)
**Prerequisite:** Phase 2 complete (weighted couplings, adaptive annealing, 11.4T/s)

---

## Executive Summary

Phase 3 transforms TileAnneal from "an optimizer kernel with guardrails" into "a QUBO solver that solves real problems."

**Theme:** QUBO-complete, spatially-native, demonstrably useful

**"Done" means:**
- You can represent arbitrary QUBO on a grid without hacks (bias terms)
- You have a visual demo that runs in seconds and clearly improves a real task
- The sentence "TileAnneal is a GPU-accelerated QUBO solver for spatial problems" is defensible

---

## Strategic Context

### What We Have (Phase 2 Complete)

| Capability | Status |
|------------|--------|
| Uniform fast path | 11.4T spin/s |
| Weighted couplings J∈{-4..+4} | ✅ |
| Adaptive annealing (ThreeStage) | 98.6% quality |
| Best-solution tracking | ✅ |
| ΔE-bucketed acceptance | ✅ |
| Acceptance tables (65 entries) | ✅ |

This is no longer research code. This is an optimizer kernel with:
- **Instrumentation** (acceptance tracking)
- **Control** (adaptive annealing)
- **Expressiveness** (weighted couplings)

That loop is what allows Phase 3 to exist.

### What We Must NOT Do

> "If you try to execute everything linearly, you'll dilute momentum."

**Deferred to Phase 4+:**
- Parallel tempering (powerful, but premature)
- Python bindings (only after the story is stable)
- Graph embedding (a trap if done too early)
- Multi-GPU (engineering heavy, not the bottleneck)

---

## Phase 3 Structure

### Weeks 1-2: Bias Terms (h_i) — Non-Negotiable

**Why mandatory:**
Without bias terms:
- You cannot represent general QUBOs cleanly
- You cannot encode constraints properly
- You cannot honestly say "QUBO solver"

**The Hamiltonian becomes complete:**
```
H = -Σ J_ij·s_i·s_j - Σ h_i·s_i
```

**Energy change with bias:**
```
ΔE = 2 * s_i * (Σ J_ij * s_j + h_i)
```

**Implementation:**

1. Add `h_bias` field to `GpuIsingGrid` (optional, like weighted couplings)
2. Extend weighted kernel to include bias term in ΔE
3. ΔE range expands: was ±32, now ±36 (4 neighbors × 4 + bias × 4)
4. Update acceptance table (73 entries for ΔE∈{-36..+36})
5. Add `compute_weighted_energy_with_bias()`

**Deliverables:**
- [x] `h_bias: Option<CudaSlice<i8>>` in GpuIsingGrid
- [x] Extended weighted kernel with bias term
- [x] Acceptance table for ΔE∈{-36..+36} (73 entries)
- [x] Validation: small QUBO test problems (`examples/bias_terms_test.rs`)
- [x] `set_bias()` and `new_with_qubo()` convenience methods

**Gate: PASSED**
TileAnneal can now represent arbitrary QUBO on a grid without hacks.

H = -Σ J_ij·s_i·s_j - Σ h_i·s_i (complete Hamiltonian)

**Infrastructure (Pre-Demo Hardening):**
- [x] `QuboBuilder` with normalization helpers (rescales arbitrary weights to i8 range)
- [x] Acceptance table monotonicity verification (prevents off-by-one bugs)
- [x] Frustrated QUBO validation (verifies best-energy tracking)
- [x] Bias overhead measurement (0% overhead confirmed)
- [x] Defensible claim scope tightened: "spatially embedded problems (2D lattice)"

See `examples/phase3_infrastructure_test.rs` for validation.

---

### Weeks 3-4: Image Segmentation Demo — Maximum Leverage

**Why this is strategic:**
- Uses only what we already have
- Visually obvious result
- Screams "spatial optimization"
- Avoids graph embedding complexity
- Survives skeptical audiences

**This is a conversion engine** — for collaborators, reviewers, and future users.

**The Problem:**
Given an image, partition pixels into K segments minimizing:
```
E = Σ_i data_cost(i, label_i) + λ Σ_{i,j∈neighbors} smoothness_cost(label_i, label_j)
```

For binary segmentation (K=2), this is exactly an Ising model:
- **Bias h_i** = data term (how much pixel i "wants" to be foreground)
- **Coupling J_ij** = smoothness term (penalty for adjacent pixels having different labels)

**Model Decision: Binary Potts/Ising**

This is the simplest, most standard choice and aligns perfectly with our architecture:

| Component | Ising Mapping | Physical Meaning |
|-----------|---------------|------------------|
| Spin s_i = +1 | Foreground | Pixel labeled as "object" |
| Spin s_i = -1 | Background | Pixel labeled as "background" |
| Coupling J > 0 | Ferromagnetic | Neighbors prefer **same** label (smoothness) |
| Bias h_i > 0 | External field | Pixel "wants" to be foreground |
| Bias h_i < 0 | External field | Pixel "wants" to be background |

**Why ferromagnetic (J > 0)?**
- Standard smoothness: adjacent pixels should have the same label
- Opposite of MaxCut (which is antiferromagnetic J < 0)
- This is the natural choice for image segmentation

**Critical: Don't Demo "Thresholding With Extra Steps"**

If bias is just `h_i = k*(intensity - t)` and couplings are constant, the output looks
like a smoothed threshold. Viewers will fairly think: "so… blur + threshold?"

**Must include at least one of:**

**A) User Scribbles/Seeds (recommended)**
```
Two sets of seed pixels: foreground scribble and background scribble.
Bias is strong at seeds and fades with distance (or stays strong only at seeds).
The solver then "fills in" the region smoothly.
```
Why it wins: shows constraint satisfaction and global consistency, not just filtering.

**B) Edge-Aware Smoothness**
```
J_ij = λ * exp(-α * (I_i - I_j)²)   // High coupling across similar pixels
                                     // Low coupling across edges (allow boundary)
```
Why it wins: boundaries align with edges → output looks intelligent.

**Implementation:**

1. **Bias from Seeds (primary) + Intensity (secondary):**
   ```rust
   fn compute_bias(image: &GrayImage, seeds: Option<&SeedMask>) -> Vec<f64> {
       // Strong positive bias at foreground seeds
       // Strong negative bias at background seeds
       // Mild bias from intensity at unseeded pixels
   }
   ```

2. **Edge-Aware Couplings:**
   ```rust
   fn compute_couplings(image: &GrayImage, lambda: f64, alpha: f64) -> (Vec<f64>, Vec<f64>) {
       // J_ij = lambda * exp(-alpha * (I_i - I_j)^2)
       // High coupling for similar neighbors, low for edges
   }
   ```

3. **Intent-Preserving Normalization:**
   ```rust
   // DON'T normalize J and h together blindly - it changes effective λ!
   // Normalize with intent:
   // 1. Decide λ:h ratio first (this IS the model)
   // 2. Scale both together only if needed to fit [-4,+4]
   // 3. Keep the λ:h ratio fixed
   ```

4. **Run TileAnneal with Segmentation Defaults:**
   - Explore: target 0.35 acceptance, shorter duration (don't overmix)
   - Freeze: ramp β until uphill acceptance ~0
   - Polish: T=0 until no improvement

5. **Flight Recorder (always log):**
   - Energy vs sweep
   - Acceptance buckets per phase (explore/freeze/polish)
   - Final β reached and time per stage
   - Runtime

**Demo Datasets (3 required):**

| Dataset | Purpose |
|---------|---------|
| Synthetic easy | Circle on noisy background - shows correctness |
| Real with edges | Object on background - shows edge-aware matters |
| Ambiguous | Low contrast - shows why seeds/optimization > thresholding |

If you only show "nice" images, people assume the method is fragile.

**Success Metrics:**
- Energy vs sweep curve (always)
- Runtime (always)
- IoU / pixel accuracy (for synthetic with ground truth)

**Week 3 Deliverables:**
- [ ] Image loading + mask output (PNG in/out)
- [ ] Bias from intensity baseline
- [ ] Seed support (optional but recommended)
- [ ] Constant λ coupling
- [ ] Energy curve + runtime logs

**Week 4 Deliverables:**
- [ ] Edge-aware couplings
- [ ] Intent-preserving normalization
- [ ] CLI demo: `--input image.png --output mask.png --lambda 1.0 --edge-aware --seeds seeds.png`
- [ ] 3 canonical runs with output examples + energy curves
- [ ] Short documentation

**Gate:**
A visual demo that runs in seconds and clearly improves a real task.

---

### Weeks 5-6: Quality Stress Test (Conditional)

**Only proceed if:**
- Image segmentation reveals hard cases
- Or weighted MaxCut quality plateaus on test instances

**If needed:**
- Analyze where ThreeStage fails
- Consider parallel tempering prototype
- Or tune existing adaptive parameters

**If not needed:**
- Document Phase 3 results
- Prepare for Phase 4 (parallel tempering, Python bindings)

---

## Success Criteria

At the end of Phase 3, this sentence must be truthful and defensible:

> "TileAnneal is a GPU-accelerated QUBO solver for spatially embedded problems
> (2D lattice / near-local couplings). It supports weighted couplings, bias terms,
> adaptive annealing, and solves real problems at trillion-update scale."

**Why this scope:**
- "Spatially embedded" prevents reviewers from assuming general QUBO (arbitrary graphs)
- Our architecture is optimized for 2D grids with local neighbor couplings
- Graph embedding for arbitrary QUBO is Phase 4+ (and may not be worth it)

**Concrete gates:**

| Week | Gate | Metric | Status |
|------|------|--------|--------|
| 2 | QUBO-complete | Can represent H = J + h | **PASSED** |
| 4 | Demo working | Visual segmentation in <10s | Pending |
| 4 | Quality | Segmentation visually correct | Pending |

---

## What Phase 3 Enables (Phase 4 Preview)

Once Phase 3 is complete, the natural next steps are:

1. **Parallel Tempering** — For glassy, frustrated landscapes
2. **Python Bindings** — After the API is stable and demo exists
3. **Benchmark Suite** — G-set MaxCut, comparison with baselines
4. **Multi-GPU** — Scale to 10B+ spins

But those come after we own the spatially-native domain completely.

---

## Anti-Patterns to Avoid

1. **Premature generalization** — Don't add graph embedding yet
2. **API freeze before demo** — Don't do Python bindings first
3. **Algorithmic escalation before value** — Parallel tempering is Phase 4
4. **Feature diffusion** — Lock the problem class, show value, then expand

---

## Technical Notes

### Bias Term Integration

The weighted kernel already computes:
```cuda
int neighbor_sum = j_left * s_left + j_right * s_right + j_up * s_up + j_down * s_down;
int delta_e = 2 * s * neighbor_sum;
```

With bias, this becomes:
```cuda
int neighbor_sum = j_left * s_left + j_right * s_right + j_up * s_up + j_down * s_down;
int bias = h_bias[idx];
int delta_e = 2 * s * (neighbor_sum + bias);
```

One additional global memory load per spin. Acceptable for weighted path.

### Image Segmentation Energy

For grayscale image with intensity I_i ∈ [0, 255]:

**Data term (bias):**
```
h_i = round((I_i - 128) * 4 / 128)  // Maps to [-4, +4]
```

**Smoothness term (coupling):**
```
J_ij = -λ  // Antiferromagnetic, penalizes different labels
         // λ typically 1-2, can use gradient magnitude for edge-aware
```

**Edge-aware variant:**
```
J_ij = -λ * exp(-|I_i - I_j| / σ)  // Weaker coupling at edges
```

---

*Phase 3 planning complete. January 2026.*
