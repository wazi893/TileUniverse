# V3 Architecture Briefing: Computational Routing Fabric

**Audience**: External specialist with no codebase access.
**Purpose**: Solicit architectural feedback on a proposed evolution from tile-based CPU simulation toward neuromorphic-inspired computational interconnect.
**Status**: Pre-implementation. Thought experiment stage.

---

## What We Have Today (V2)

We have a working CPU built entirely from discrete logic tiles on a 2D grid.

Not a CPU *simulator* in the traditional sense — there is no `match opcode { ADD => a + b }` anywhere. Instead, thousands of tiny tiles (AND gates, OR gates, multiplexers, registers, wires) are placed on a 128x128 grid with 4 vertical layers. When the CPU executes an instruction, electrical signals physically propagate through these tiles until the circuit settles. The CPU *emerges* from the tile interactions, the same way a real chip emerges from transistors.

**Grid structure:**

```
L3  ─── Long-haul wiring (cross-chip routes)
L2  ─── Cross-bank wiring (medium-distance)
L1  ─── Local routing (short connections)
L0  ─── Components (gates, registers, muxes, ALU)
```

**CPU specs:**
- 32-bit instruction word, 64-bit data path
- 16 general-purpose registers (R0-R15)
- 64-entry ROM, 64-cell RAM
- Full ALU (add/sub/and/or/xor/shl/shr), branch logic, call/return
- 23-address MMIO bus with 8 peripheral device types (math coprocessor, display, quantum bridge, dataset server, etc.)
- 2,620 passing tests, 5 golden benchmark hashes, differential ISS oracle

**Tile primitives** (the only building blocks):

| Tile | Behavior |
|------|----------|
| Wire | output = left OR right OR up OR down |
| And | output = left AND right |
| Or | output = left OR right |
| Mux | output = select(up) ? right : left |
| Const | output = fixed value (set at build time) |
| Register64 | captures left on rising clock edge, holds value |
| ViaUp/ViaDown | passes value between layers at same (x,y) |
| CarryDetect | detects arithmetic carry across 64 bits |
| BitSelect | extracts single bit from a value |

From these ~10 types, we build everything. The simulation engine evaluates tiles in dependency order with wire-delay modeling until convergence (stable state).

**What works well:**
- Fully physical instruction fetch, decode, ALU, branching, halt, ROM bank selection
- Sparse evaluation via dirty-bitset propagation (only re-evaluate tiles whose inputs changed)
- Zone-scoped settling (pipeline, branch, commit, RAM each have pre-computed tile subsets)
- Differential testing: a software ISS (instruction set simulator) runs alongside the tile CPU; after every instruction, all 16 registers, 64 RAM cells, and PC are compared. 10,000+ random programs validated with zero divergence.

**The hybrid problem — what doesn't work yet:**

The CPU has 16 logical registers but only 8 physical Register64 tiles (organized as 2 banks of 8). When the CPU needs a register from the other bank, software intervenes: save the 8 physical tiles, overwrite them with the other bank's values, let the circuit propagate, then restore. Same pattern for RAM (64 logical cells, 8 physical).

This "hybrid assist" mechanism works correctly and is well-tested, but it's a cheat. A Redstone CPU doesn't pause to swap values in software. Neither should we.

**The 5 remaining hybrid assists:**

| Assist | Root cause |
|--------|-----------|
| Register bank switch | 8 of 16 registers are physical |
| Mixed-bank dual capture | Reading two registers from different banks |
| Mixed-bank ALU execution | ALU operands span banks |
| RAM high-bank read swap | 8 of 64 RAM cells are physical |
| Zero-shift carry override | ISA-defined (counter = 0, effectively free) |

3 of 5 are register-banking. 1 is RAM-banking. 1 is a non-issue.

**The routing bottleneck:**

We use A* routing on L1-L3 to connect components. Earlier sprints attempted to physically route the ROM bank selector and hit overflow — 20 nets had nowhere to go on the 128x128 grid. After 4 sprints of router improvements (congestion negotiation, bounded rip-up, topology experiments), we got ROM fully physical. But registers and RAM are bigger problems with more nets competing for space.

---

## The Insight That Started This

Vias (ViaUp/ViaDown) currently do nothing but pass a signal between layers. They exist purely for routing — when L1 is congested, you go up to L2, route around, come back down. The via is a dumb pipe.

**What if vias could compute?**

In a neural network, the connections between layers aren't passthrough — they carry weights, they modulate signals. The synapses do as much computation as the neurons. What if our inter-layer connections worked the same way?

This aligns with a 2025 paper ("Processing-in-Interconnect") proposing that routing hardware itself can perform neural computation: delays map to temporal coding, fan-out maps to broadcast, packet drop maps to thresholding. The router *is* the neuron.

---

## The Proposed V3 Path

Three phases, each building on the previous. The thesis:

1. **Interconnect is the primary compute substrate** — not just the component layer
2. **Memory locality is the governing constraint** — data should live near where it's used
3. **Sparsity/eventing is the execution model** — only compute what changed

### Phase 1: Programmable Via Behaviors

Add 3 new tile types that make the routing fabric computational:

**ThresholdVia** — conditional layer crossing.
Output = input if popcount(active neighbors) >= threshold, else 0.
Use case: natural fan-in aggregation. Instead of routing 4 signals to an AND gate, route them through a ThresholdVia with threshold=4. The routing *is* the logic.

**WeightedPassVia** — masking layer crossing.
Output = input AND weight_mask (stored 64-bit value).
Use case: operand selection. Instead of a mux tree that selects which register value reaches the ALU, route all register values through WeightedPassVias where only one has a non-zero mask. The routing *is* the mux.

**RefractoryVia** — temporal suppression.
Output = input normally, but after firing, suppresses output for N ticks.
Use case: timing control, oscillation prevention, pulse shaping. This is the most neuromorphic primitive — it introduces biological-style refractory periods into the digital fabric.

**Key design tension**: ThresholdVia and WeightedPassVia are combinational (stateless within a tick, like existing tiles). RefractoryVia has tick-spanning state. The simulation engine's convergence guarantee relies on tiles being pure functions of their inputs. RefractoryVia breaks this — its output depends on history.

**Proposed mitigation**: Restrict refractory tiles to via positions only (L1-L3, never L0). Vias are inherently unidirectional, so no feedback loops through refractory tiles. The engine pre-scans refractory tiles at each tick start and force-inserts active ones into the dirty set (analogous to how Register tiles are handled on clock edges).

### Phase 2: Layer Specialization by Traffic Class

Currently all 4 layers are functionally identical. Specialize them:

| Layer | Traffic class | Contents |
|-------|--------------|----------|
| L0 | Components | Gates, registers, ALU (unchanged) |
| L1 | Data plane | Register values, ALU results, RAM data, MMIO data |
| L2 | Control plane | Decoder outputs, enable signals, write-enables, branch flags |
| L3 | Long-range / broadcast | Cross-region signals, clock, fan-out trees |

**Why this helps:**
- Control signals change only during decode (once per instruction). If L2 is control-only, we can skip L2 propagation during ALU and writeback phases. Free speedup.
- Data signals are wide (64-bit) and high-bandwidth. Dedicating L1 to data prevents control wires from stealing data routing capacity.
- Smart vias (Phase 1) are allowed on L1 (data can tolerate masking) but forbidden on L2 (control must be bit-exact). This is a natural safety boundary.

**Routing impact**: The A* router gains layer-affinity soft costs. Data nets prefer L1, control nets prefer L2. Not a hard constraint (that would cause routing failures), but a 10x cost penalty for "wrong layer" routing.

**Key concern**: All existing wiring was placed without layer discipline over 60+ development sprints. Migration is incremental, not atomic.

### Phase 3: Distributed Register Islands

Instead of 8 Register64 tiles in one location (with software bank-swapping), place all 16 registers physically as 4 islands of 4:

```
Island 0 (R0-R3):   near ALU          — primary compute
Island 1 (R4-R7):   near RAM          — memory staging
Island 2 (R8-R11):  near branch logic — loop counters
Island 3 (R12-R15): near MMIO bus     — peripheral I/O
```

**Why this kills the hybrid assists**: The bank-swap exists because we have 16 logical registers but only 8 physical tiles. With 16 physical tiles (in 4 islands), there's nothing to swap. Three hybrid assists (bank switch, mixed-bank dual capture, mixed-bank ALU) drop to zero.

**Data flow**:
- Operand read: select island (2-bit) then register within island (2-bit). Intra-island selection is local L0 wiring. Inter-island selection routes on L1 (data plane).
- Writeback: ALU result broadcasts on L1 to all islands. Only the destination island's write-enable is active. Phase 1's ThresholdVia naturally gates this — each island entrance has a threshold via that only admits the writeback when the island's write-enable is high.

**Key concern**: The inter-island data bus may not fit on 128x128. ROM bank routing already required 4 sprints to resolve overflow. The register bus is bigger. Fallback: 2 islands of 8 instead of 4 of 4 (halves routing pressure, still eliminates all 3 register assists).

---

## Phase Dependencies

```
Phase 1 (Smart Vias)
    ↓ enables computational routing
Phase 2 (Layer Specialization)
    ↓ frees L1 data-plane capacity
Phase 3 (Distributed Registers)
    ↓ uses L1 capacity for inter-island bus
```

Phase 3 without Phase 2 will likely hit routing overflow. Phase 2 without Phase 1 works but misses the "routing is computation" opportunity.

---

## What We're Looking For Feedback On

1. **Smart via design**: Are ThresholdVia / WeightedPassVia / RefractoryVia the right primitives? Are we missing one? Is there a simpler set that achieves the same expressiveness? The refractory tile's convergence implications concern us — is there a cleaner way to get temporal behavior without breaking the pure-function propagation model?

2. **Layer specialization trade-offs**: Is data/control/long-range the right split? Should we separate read-path data from write-path data (NorthPole uses 4 NoCs)? Is 4 layers enough, or should we go to 6 or 8?

3. **Distributed register feasibility**: Is 4 islands of 4 the right granularity? Would 2 islands of 8 be more practical as a first step? How do real neuromorphic chips handle the fan-out problem of broadcasting results to distributed register files?

4. **Scaling**: At what point does 128x128 become fundamentally insufficient? If we need to go to 256x256, does that change any of the phase designs? The simulation cost scales with grid area — is there a way to keep the grid small while adding capacity?

5. **What are we not seeing?** Are there failure modes in the smart-via approach that we're underestimating? Are there design patterns from neuromorphic or FPGA architectures that would shortcut this entire approach?

---

## Reference Points

The design draws inspiration from several existing architectures:

- **Intel Loihi 2**: 2D mesh of 128 neuron cores with XY routing, GALS clocking, programmable neuron models
- **IBM NorthPole**: 256-core array with 4 specialized NoCs (partial sum / activation / model / instruction), 192MB distributed SRAM, zero off-chip memory
- **Darwin3**: 24x24 mesh with RISC-V management core + 575 neuron cores, CXY congestion-aware routing
- **SpiNNaker 2**: Packet-switched multicast spike routing, GALS, 152 processing elements per chip
- **"Processing-in-Interconnect" (arXiv 2025)**: Routing hardware as computational substrate — delays, broadcast, timeouts map to neural operations

The most direct parallel is NorthPole's multi-NoC architecture mapping onto our multi-layer grid, and the processing-in-interconnect paper's thesis that the routing fabric itself should compute.

---

## Appendix: Current System Metrics

- Grid: 128 x 128 x 4 layers = 65,536 tile positions
- Tile types: ~10 primitive types
- Physical tiles placed: ~2,000 (most of grid is unplaced Wire default)
- Instruction throughput: ~2 simulation ticks per instruction
- Sparse eval: only ~7,400 tiles evaluated per stage (of 65,536)
- Test count: 2,620 passing, 0 failing
- Benchmark golden hashes: 5, all locked
- Hybrid assists remaining: 5 (3 register, 1 RAM, 1 trivial)
- MMIO peripherals: 8 device types across 23 addresses
- Showcase programs: 11 (including MNIST inference via packed binary popcount)
