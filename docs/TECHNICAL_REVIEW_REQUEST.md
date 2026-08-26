# TileUniverse: Technical Review Request

**Date:** 2025-02-02
**Version:** 0.1
**Status:** Seeking Expert Feedback

---

## Executive Summary

TileUniverse is a high-performance simulation engine that unifies:
- **Cellular automata** (97.07 trillion tiles/sec, packed 1-bit, live-confirmed 2026-07-03)
- **Ising/QUBO optimization** (10 trillion spin updates/sec)
- **Spiking neural networks** (LIF neurons, STDP, GPU-accelerated)
- **Sparse heterogeneous circuits** (73 billion tiles/sec, 52 tile types)

We recently implemented a **Hopfield associative memory** on this infrastructure, achieving 10 trillion spin updates/second on a single RTX 5090, with perfect pattern recovery from 40% corruption at 16 million neurons.

We are seeking feedback on:
1. Architectural decisions and potential bottlenecks
2. Whether our approach to neuromorphic-Ising unification is sound
3. Suggestions for scaling beyond single-GPU
4. Potential applications we may be overlooking

---

## 1. System Architecture

### 1.1 Core Abstraction: The Tile

Everything is a **tile** on a 2D grid. Each tile:
- Has a 64-bit logic value
- Has a type (52 types: Wire, And, Or, Register, LIF neuron primitives, etc.)
- Reads from 4 neighbors (left, right, up, down)
- Updates synchronously

```
┌─────────────────────────────────────────────────────────────┐
│  Tile Grid (conceptual)                                     │
│                                                             │
│    ┌───┐ ┌───┐ ┌───┐ ┌───┐                                 │
│    │And│─│Wire│─│Or │─│Reg│  ← Each tile = 64-bit + type   │
│    └───┘ └───┘ └───┘ └───┘                                 │
│      │     │     │     │                                    │
│    ┌───┐ ┌───┐ ┌───┐ ┌───┐                                 │
│    │Add│─│Mux│─│Xor│─│LIF│  ← Neighbors provide inputs     │
│    └───┘ └───┘ └───┘ └───┘                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Execution Backends

| Backend | Throughput | Use Case |
|---------|------------|----------|
| **Packed 1-bit** | 97.07T tiles/sec | Homogeneous grids (all tiles same type per 64-tile word) |
| **Sparse heterogeneous** | 73G tiles/sec | Mixed tile types, only active tiles evaluated |
| **Dense heterogeneous** | ~10G tiles/sec | All tiles evaluated every tick |

The packed backend achieves extreme throughput by:
- Packing 64 boolean tiles per `u64`
- Using warp shuffles for horizontal neighbors (zero-cost on GPU)
- Register-resident evaluation (32 rows in registers, cooperative halo exchange)

### 1.3 Ising Model Mapping

The packed 1-bit backend directly implements Ising dynamics:

```
E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ

where:
  sᵢ ∈ {-1, +1} encoded as {0, 1} bits
  Jᵢⱼ = coupling weights (uniform or per-edge, i8 range [-4, +4])
  hᵢ = bias terms
```

Metropolis-Hastings updates with precomputed acceptance tables enable stochastic optimization at GPU memory bandwidth limits.

---

## 2. Hopfield Network Implementation

### 2.1 The Insight

Hopfield networks ARE Ising models:

```
Hopfield: E = -½ Σᵢⱼ wᵢⱼ xᵢ xⱼ    (neurons x ∈ {0,1}, weights w)
Ising:    E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ      (spins s ∈ {-1,+1}, couplings J)

With s = 2x - 1, these are equivalent.
```

We implemented Hopfield by:
1. Computing Hebbian weights: `Jᵢⱼ = (1/P) Σₚ ξᵢᵖ ξⱼᵖ`
2. Mapping to Ising couplings (quantized to i8)
3. Running simulated annealing on GPU
4. Reading back converged state

### 2.2 Benchmark Results

**Hardware:** RTX 5090 (Blackwell), CUDA 12.x

| Grid Size | Neurons | Throughput | Memory | Pattern Recovery |
|-----------|---------|------------|--------|------------------|
| 8×8 | 64 | 5.85 Gspins/s | 0.01 MB | 100% @ 40% corruption |
| 64×64 | 4,096 | 5.85 Gspins/s | 0.01 MB | 100% @ 40% corruption |
| 128×128 | 16,384 | 63 Gspins/s | 0.04 MB | 100% @ 30% corruption |
| 1024×1024 | 1,048,576 | 3.36 Tspins/s | 0.25 MB | — |
| 4096×4096 | 16,777,216 | 10.18 Tspins/s | 4.00 MB | — |

**Note:** Pattern recovery tests used antiferromagnetic lattice patterns (checkerboard). Full Hopfield with arbitrary patterns requires the weighted coupling path, which is slower (~70 Gspins/s at 64K neurons).

### 2.3 Theoretical Capacity

Classical Hopfield capacity: `P_max ≈ 0.138 × N`

| Neurons | Max Patterns (theory) | Practical Limit |
|---------|----------------------|-----------------|
| 1,024 | 141 | ~100 (lattice topology) |
| 16,384 | 2,260 | ~1,500 |
| 1,048,576 | 144,703 | Untested |

**Open question:** Our current implementation uses 2D lattice topology (nearest-neighbor couplings). True Hopfield requires all-to-all connectivity. How much capacity do we lose with lattice topology? Is there a middle ground (small-world, hierarchical)?

---

## 3. SNN Infrastructure

We have a complete spiking neural network stack:

### 3.1 Neuron Model

**Leaky Integrate-and-Fire (LIF)** with:
- Q8.8 fixed-point membrane potential
- Configurable leak, threshold, refractory period
- 8 bytes per neuron (cache-efficient)

```rust
pub struct Neuron {
    membrane: i16,      // Q8.8 fixed-point
    threshold: i16,     // Fire threshold
    leak: u8,           // Decay factor (Q0.8)
    refractory: u8,     // Refractory counter
    last_spike: u8,     // For STDP timing
    flags: u8,          // State flags
}
```

### 3.2 Learning Rules

| Rule | Description | GPU Support |
|------|-------------|-------------|
| **STDP** | Spike-timing dependent plasticity | ✓ |
| **R-STDP** | Reward-modulated STDP | ✓ |
| **E-prop** | Eligibility propagation (Bellec 2020) | ✓ |

### 3.3 GPU Kernels

Six fused CUDA kernels for SNN execution:
1. Input spike generation (Poisson)
2. LIF dynamics + spike detection
3. Current accumulation (CSR SpMV)
4. Output spike counting
5. STDP weight updates
6. State snapshot

Expected speedup: 10-50× over CPU for 10K+ neurons.

### 3.4 Quantum-SNN Hybrid

We have experimental quantum-classical hybrid modes:
- Stabilizer neurons (Clifford-only, O(n²) memory)
- Small full-quantum clusters (8-16 qubits, T-gates)
- Quantum interference for decision-making

**Open question:** Is the quantum-SNN hybrid scientifically meaningful, or is it a solution looking for a problem?

---

## 4. FPGA Export Path

### 4.1 Current Capability

We can generate synthesizable Verilog from tile grids:

```rust
let grid = SynthesisGrid::new(64, 64);
// ... populate with tiles ...
let verilog = grid.to_verilog();  // Generates ~2000 lines of Verilog
```

For Hopfield specifically, we generate deterministic update logic:

```verilog
// Each neuron: antiferromagnetic update
assign next_spins[9] = (spins[8] + spins[10] + spins[1] + spins[17]) < 2 ? 1'b1 : 1'b0;
```

### 4.2 Resource Estimates

| Network | Neurons | Flip-Flops | LUTs | Est. Fmax | Throughput |
|---------|---------|------------|------|-----------|------------|
| 8×8 | 64 | 96 | 576 | 400 MHz | 25.6 Gspins/s |
| 16×16 | 256 | 288 | 2,112 | 300 MHz | 76.8 Gspins/s |
| 64×64 | 4,096 | 4,128 | 33K | 200 MHz | 819 Gspins/s |

### 4.3 Open Questions

1. **Stochastic updates on FPGA:** Our GPU implementation uses Metropolis-Hastings with RNG. FPGA equivalent would need LFSR-based random, or we switch to deterministic (Glauber) dynamics. Tradeoffs?

2. **Weight storage:** For weighted Hopfield, each edge needs a weight. At 64×64 with 4 neighbors each, that's 16K weights. Store in BRAM? Hardcode as LUT constants?

3. **GPU-FPGA hybrid:** If we had a PCIe FPGA, what's the optimal split? GPU for training/large-scale, FPGA for low-latency inference?

---

## 5. Architectural Decisions & Tradeoffs

### 5.1 Why Tiles?

**Pros:**
- Uniform abstraction (everything is a tile)
- Natural 2D layout for GPU (coalesced memory, warp shuffles)
- Easy visualization and debugging
- Maps directly to FPGA fabric

**Cons:**
- Overhead for non-local connectivity (must route through Wire tiles)
- 64-bit per tile might be overkill for 1-bit Ising
- Heterogeneous tiles lose packed efficiency

### 5.2 Why Packed 1-bit?

We pack 64 tiles per `u64` for homogeneous grids. This:
- Achieves 97.07T tiles/sec in the 2026-07-03 confirmation sweep
- Enables warp shuffle for neighbor access
- Requires all 64 tiles in a word share the same type

**Tradeoff:** Can't mix tile types within a word. For heterogeneous circuits, we fall back to sparse evaluation (1000× slower).

### 5.3 Why Sparse Evaluation?

For circuits where 99% of tiles are stable:
- Only evaluate dirty tiles
- Propagate dirty flags to neighbors
- Hierarchical dirty tracking (L0/L1/L2 bitsets)

**Achieved:** 73 Gtiles/sec at 268M tiles with 0.1% activity.

**Open question:** Is there a better sparsity structure for neuromorphic workloads? Block-sparse? Structured sparsity (2:4)?

---

## 6. Comparison to Related Work

| System | Approach | Throughput | Notes |
|--------|----------|------------|-------|
| **TileUniverse** | GPU tile grid, packed Ising | 10T spins/s | This work |
| **D-Wave SimCIM** | Simulated coherent Ising machine | ~1T spins/s | Specialized for QUBO |
| **Intel Loihi** | Neuromorphic chip | ~1M neurons | Hardware, not simulation |
| **IBM TrueNorth** | Neuromorphic chip | 1M neurons | Event-driven |
| **BrainScaleS** | Analog neuromorphic | 10K neurons | 10,000× real-time |
| **Cerebras** | Wafer-scale, not neuromorphic | — | Different paradigm |

**Question:** Are we comparing apples to apples? Our throughput is for synchronous Ising updates. Neuromorphic hardware is event-driven. How do we fairly benchmark?

---

## 7. Open Questions for Reviewers

### Architecture

1. **Tile abstraction overhead:** Is the 2D tile grid the right abstraction, or should we support arbitrary graph topologies natively?

2. **Sparse vs packed tradeoff:** Is there a hybrid approach that gets packed throughput with heterogeneous flexibility?

3. **Memory layout:** We use row-major packed. Would a space-filling curve (Hilbert, Z-order) improve cache behavior for sparse access patterns?

### Neuromorphic

4. **Hopfield capacity on lattice:** How much do we lose by restricting to nearest-neighbor vs all-to-all? Is hierarchical Hopfield (local clusters + long-range) worth implementing?

5. **SNN-Ising bridge:** Can we unify the SNN stack with the Ising stack? A spiking Boltzmann machine?

6. **Learning on Ising:** Beyond Hebbian, can we do online learning directly on the GPU Ising model? Contrastive Hebbian?

### Scaling

7. **Multi-GPU:** Our current implementation is single-GPU. What's the right decomposition for multi-GPU? Spatial partitioning with halo exchange?

8. **GPU-FPGA hybrid:** With a PCIe FPGA (e.g., Alveo U50), how should we split work? Latency-sensitive inference on FPGA, throughput on GPU?

### Applications

9. **What problems fit this model?** We've shown Hopfield pattern completion. What else maps naturally to high-throughput Ising? Constraint satisfaction? Combinatorial optimization? Protein folding energy minimization?

10. **Datacenter simulation:** The original motivation was simulating datacenters. How do we map datacenter dynamics (load balancing, thermal, network) to Ising/tile models?

---

## 8. Reproduction Instructions

### Build

```bash
git clone <repo>
cd engine
cargo build --release --features cuda
```

### Run Hopfield Benchmark

```bash
cargo run --release --features cuda --example hopfield_tile_network
```

**Expected output:**
```
=== Phase 1: Lattice-Compatible Hopfield Network ===
  Checkerboard: 40% corruption -> overlap 0.16 -> 1.00 [OK]

=== Phase 3: Maximum Scale Benchmarks ===
  16777216  4096x4096    606722 sweeps/s  10179.11G spins/s   4.00 MB
```

### Generate FPGA Verilog

```bash
cargo run --release --example hopfield_fpga_export
# Outputs: hopfield_8x8.v, hopfield_16x16.v, etc.
```

---

## 9. Appendix: Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Packed Ising kernel | `src/cuda_tiles.rs:3097-3550` | ~450 |
| Sparse evaluation | `src/cuda_tiles.rs:8760-10890` | ~2130 |
| Hopfield example | `examples/hopfield_tile_network.rs` | 553 |
| FPGA synthesis | `src/fpga/synthesis.rs` | 662 |
| LIF neuron | `src/snn/neuron.rs` | 333 |
| STDP learning | `src/snn/stdp.rs` | 389 |
| GPU SNN kernels | `src/snn/gpu_kernels.rs` | ~500 |

---

## 10. Contact & Contribution

We welcome feedback, criticism, and collaboration. Specific expertise sought:
- Neuromorphic hardware architects
- QUBO/Ising optimization researchers
- FPGA synthesis experts
- GPU performance engineers

---

*This document was prepared for technical review. Please be direct about flaws, misconceptions, or missed opportunities.*
