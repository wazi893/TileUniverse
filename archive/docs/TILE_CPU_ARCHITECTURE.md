# TileFabric CPU Architecture Specification

**Version:** 1.0
**Status:** Design Phase
**Author:** TileUniverse Architecture Team

---

## Executive Summary

This document specifies **TileFabric CPU** - a processor architecture where every gate, register, and wire is a discrete tile evaluated by the TileUniverse simulation engine. Unlike software emulators that mimic CPU behavior, TileFabric CPUs execute by propagating signals through physical tile connections, enabling:

- **True timing analysis** via propagation delay simulation
- **Automatic critical path detection** for frequency optimization
- **GPU acceleration** at 115 trillion tiles/second
- **Sparse evaluation** providing 1000x speedup for stable circuits
- **Visual debugging** - watch signals propagate through the datapath

The goal is not a toy. It's a platform for designing, simulating, and optimizing digital logic at scales and speeds previously requiring expensive EDA tools.

---

## 1. Design Philosophy

### 1.1 Why Tile-Based Execution Matters

Traditional CPU simulators fall into two categories:

1. **Functional simulators** (QEMU, gem5) - Execute instructions directly, ignore timing
2. **RTL simulators** (Verilator, VCS) - Simulate HDL, expensive, slow

TileFabric occupies a unique middle ground:

| Property | Functional Sim | RTL Sim | TileFabric |
|----------|---------------|---------|------------|
| Timing accuracy | None | Cycle-accurate | Gate-accurate |
| Speed | 100M+ inst/sec | 10K-1M cycles/sec | 1B+ tiles/sec |
| Visual debugging | Limited | Waveforms | Spatial signal flow |
| Optimization feedback | None | Synthesis reports | Real-time critical path |
| Learning curve | Low | Very high (HDL) | Medium (visual) |

### 1.2 Core Principle: Simulation IS Execution

The CPU doesn't "run" in a software loop. It **exists** as tiles on the grid. Execution happens when:

```rust
// This is how the CPU runs - the simulation IS the execution
for _ in 0..cycles {
    sim.tick_with_delays(1000);  // Propagate until stable
    sim.tick_clock();            // Clock edge - registers capture
}
```

There is no `cpu.step()` function that bypasses the tiles. The tiles ARE the CPU.

### 1.3 Signal Representation

Each tile carries a **64-bit value**. For an 8-bit CPU:
- Bits 0-7: Data value
- Bits 8-15: Secondary channel (carry, flags, etc.)
- Bits 16-63: Reserved for extensions (16-bit mode, vector ops)

Multi-bit buses use **parallel tiles**:
```
Bit 0: [Wire]----[And]----[Wire]
Bit 1: [Wire]----[And]----[Wire]
...
Bit 7: [Wire]----[And]----[Wire]
```

---

## 2. Architecture Overview

### 2.1 Harvard Architecture with Tile-Native Design

```
+------------------------------------------------------------------+
|                        TileFabric CPU                             |
|                                                                   |
|  +------------+     +-----------+     +------------+              |
|  |    ROM     |---->|  DECODE   |---->|  CONTROL   |              |
|  | (Const[])  |     | Mux8to1[] |     |  SIGNALS   |              |
|  +------------+     +-----------+     +------------+              |
|        ^                  |                 |                     |
|        |                  v                 v                     |
|  +------------+     +-----------+     +------------+              |
|  |     PC     |<----|  BRANCH   |     |    ALU     |              |
|  | ProgramCtr |     |   LOGIC   |     | Add/Sub/   |              |
|  +------------+     +-----------+     | And/Or/Xor |              |
|        |                  ^           +------------+              |
|        |                  |                 |                     |
|        v                  |                 v                     |
|  +------------+     +-----------+     +------------+              |
|  |   FETCH    |     |   FLAGS   |<----|  REG FILE  |              |
|  |  ADDRESS   |     | Zero/Carry|     | RegEnable[]|              |
|  +------------+     +-----------+     +------------+              |
|                                             |                     |
|                                             v                     |
|                                       +------------+              |
|                                       |    RAM     |              |
|                                       |  Ram[]     |              |
|                                       +------------+              |
+------------------------------------------------------------------+
```

### 2.2 Tile Inventory

| Component | Tile Types Used | Quantity (8-bit) | Critical Path Contribution |
|-----------|-----------------|------------------|---------------------------|
| ROM (16 words) | Const | 128 | 0 (synchronous) |
| PC | ProgramCounter | 1 | 0 (synchronous) |
| Instruction Decode | Mux8to1 | 2 | 3 deltas |
| Register File (4 regs) | RegEnable | 4 | 0 (synchronous) |
| ALU | Add, Sub, And, Or, Xor, Mux | 6 | 5 deltas (Add/Sub) |
| Flag Generation | Zero, Lt | 2 | 3-4 deltas |
| Branch Logic | And, Mux | 4 | 4 deltas |
| Routing | Wire, Cross, WireH, WireV | ~200 | 1 delta each |

**Total: ~350 tiles for minimal 8-bit CPU**

### 2.3 Instruction Set Architecture

16 instructions, 4-bit opcode, immediate-heavy design optimized for tile implementation:

```
| Opcode | Mnemonic | Format     | Operation                    | Cycles |
|--------|----------|------------|------------------------------|--------|
| 0x0    | NOP      | -          | No operation                 | 1      |
| 0x1    | LDI Rd,# | Rd[2] Im[4]| Rd = Imm (sign-extended)     | 1      |
| 0x2    | MOV Rd,Rs| Rd[2] Rs[2]| Rd = Rs                      | 1      |
| 0x3    | ADD Rd,Rs| Rd[2] Rs[2]| Rd = Rd + Rs                 | 1      |
| 0x4    | SUB Rd,Rs| Rd[2] Rs[2]| Rd = Rd - Rs                 | 1      |
| 0x5    | AND Rd,Rs| Rd[2] Rs[2]| Rd = Rd & Rs                 | 1      |
| 0x6    | OR  Rd,Rs| Rd[2] Rs[2]| Rd = Rd | Rs                 | 1      |
| 0x7    | XOR Rd,Rs| Rd[2] Rs[2]| Rd = Rd ^ Rs                 | 1      |
| 0x8    | SHL Rd   | Rd[2] -[2] | Rd = Rd << 1                 | 1      |
| 0x9    | SHR Rd   | Rd[2] -[2] | Rd = Rd >> 1                 | 1      |
| 0xA    | CMP Rd,Rs| Rd[2] Rs[2]| Flags = compare(Rd, Rs)      | 1      |
| 0xB    | JMP addr | Addr[6]    | PC = Addr                    | 1      |
| 0xC    | JZ  addr | Addr[6]    | if Z: PC = Addr              | 1      |
| 0xD    | JNZ addr | Addr[6]    | if !Z: PC = Addr             | 1      |
| 0xE    | LD  Rd   | Rd[2] -[2] | Rd = RAM[R3]                 | 2      |
| 0xF    | ST  Rs   | -[2] Rs[2] | RAM[R3] = Rs                 | 2      |
```

---

## 3. Datapath Design

### 3.1 Single-Cycle Execution

Each instruction completes in one clock cycle. The critical path determines maximum frequency:

```
Clock Edge
    |
    v
+-------+    +--------+    +--------+    +-------+    +--------+
|  PC   |--->|  ROM   |--->| DECODE |--->|  ALU  |--->|  REG   |
| (0Δ)  |    | (0Δ)   |    | (3Δ)   |    | (5Δ)  |    | (0Δ)   |
+-------+    +--------+    +--------+    +-------+    +--------+
                                              |
                              Total: 8Δ + routing (~15Δ)
```

With 1ns per delta: **Critical path = ~15ns → 66 MHz simulated frequency**

### 3.2 Register File Implementation

Four 8-bit registers using `RegEnable` tiles:

```
          Write Enable (from decode)
               |
               v
         +------------+
Rs1 ---->|            |----> Read Port 1
         | RegEnable  |
Rs2 ---->|            |----> Read Port 2
         +------------+
               ^
               |
          Write Data (from ALU)
```

**Tile Layout (per register):**
```
[Decoder3to8] --> [And] --> [RegEnable Bit 0]
                  [And] --> [RegEnable Bit 1]
                  [And] --> [RegEnable Bit 2]
                  ...
                  [And] --> [RegEnable Bit 7]
```

### 3.3 ALU Implementation

The ALU uses the actual arithmetic tiles - no software shortcuts:

```
        A Input (8 bits)          B Input (8 bits)
             |                          |
             v                          v
        +----+----+                +----+----+
        |  WireH  |                |  WireH  |
        +----+----+                +----+----+
             |                          |
    +--------+--------+--------+--------+
    |        |        |        |        |
    v        v        v        v        v
+------+ +------+ +------+ +------+ +------+
| Add  | | Sub  | | And  | | Or   | | Xor  |
+------+ +------+ +------+ +------+ +------+
    |        |        |        |        |
    +--------+--------+--------+--------+
                     |
                     v
              +------------+
              | Mux8to1    |<--- ALU Op Select (from decode)
              +------------+
                     |
                     v
                ALU Result
```

**Critical path through ALU: 5Δ (Add/Sub) + 3Δ (Mux) = 8Δ**

### 3.4 Branch Unit

Conditional branches use flag-gated muxes:

```
                     +--------+
        PC + 1 ----->|        |
                     |  Mux   |-----> Next PC
Branch Target ------>|        |
                     +--------+
                         ^
                         |
                    Branch Taken
                         |
              +----------+----------+
              |                     |
         +----+----+           +----+----+
         |   And   |<--Z Flag  |   And   |<--!Z Flag
         +----+----+           +----+----+
              ^                     ^
              |                     |
         JZ Opcode             JNZ Opcode
```

---

## 4. Memory Architecture

### 4.1 ROM: Instruction Storage

ROM is implemented as `Const` tiles with pre-set values:

```
Address (from PC)
       |
       v
+-------------+
| Decoder3to8 |  (3-bit address → 8 one-hot lines)
+-------------+
  | | | | | | | |
  v v v v v v v v
+--+--+--+--+--+--+--+--+
|C0|C1|C2|C3|C4|C5|C6|C7|  Const tiles (instruction bytes)
+--+--+--+--+--+--+--+--+
  | | | | | | | |
  v v v v v v v v
+---------------------+
|      Mux8to1        |  Select addressed word
+---------------------+
          |
          v
    Instruction Out
```

**Scaling:** 256-word ROM = 8 decoders + 256 Const tiles + 8 Mux8to1 = ~280 tiles

### 4.2 RAM: Data Storage

RAM uses `Ram` tiles (write-enable gated):

```
Address (from R3)          Write Enable        Write Data
       |                        |                   |
       v                        v                   v
+-------------+            +--------+          +--------+
| Decoder3to8 |----------->|  And   |--------->|  Ram   |---> Read Data
+-------------+            +--------+          +--------+
```

**Ram tile behavior:** `output = (write_enable != 0) ? write_data : current`

---

## 5. Control Unit

### 5.1 Instruction Decode

The 8-bit instruction word is decoded into control signals:

```
Instruction [7:0]
        |
        +--[7:4]---> Opcode
        |
        +--[3:2]---> Rd (destination register)
        |
        +--[1:0]---> Rs/Imm (source register or immediate)

Opcode Decode:
        |
        v
+-------------+
| Decoder3to8 |---> ALU_ADD, ALU_SUB, ALU_AND, ALU_OR, ALU_XOR
+-------------+
        |
        +---------> REG_WRITE_EN
        |
        +---------> MEM_READ, MEM_WRITE
        |
        +---------> BRANCH, BRANCH_COND
```

### 5.2 Control Signal Generation

Each control signal is a single bit generated by AND/OR logic on opcode bits:

```rust
// Example: REG_WRITE_EN is high for LDI, MOV, ADD, SUB, AND, OR, XOR, LD
// Opcodes 0x1-0x7, 0xE
REG_WRITE_EN = (opcode[3] == 0 && opcode != 0) || (opcode == 0xE)
```

**Tile implementation:**
```
Opcode[3] ---[Not]---+
                     |
Opcode[2] ---[Or]----+---[And]---> Part of REG_WRITE_EN
                     |
Opcode[1] ---[Or]----+
```

---

## 6. Critical Path Analysis

### 6.1 Path Enumeration

| Path | Description | Delay (Δ) |
|------|-------------|-----------|
| Fetch | PC → ROM Decode → Instruction | 0 + 2 + 3 = 5Δ |
| Decode | Instruction → Control Signals | 2 + 2 = 4Δ |
| RegRead | Reg Select → Reg Data Out | 2Δ |
| ALU | A,B → Result | 5Δ (Add/Sub) |
| RegWrite | ALU Result → Register | 0Δ (sync) |
| Branch | Flags → PC Mux → Next PC | 2 + 2 = 4Δ |

**Critical Path: Fetch → Decode → RegRead → ALU → RegWrite**
```
5Δ + 4Δ + 2Δ + 5Δ = 16Δ base
+ routing (est. 10 Wire tiles) = 10Δ
─────────────────────────────────
Total: ~26Δ per cycle
```

At 1ns/delta: **38 MHz maximum frequency**

### 6.2 Optimization Opportunities

1. **Pipelining**: Split into Fetch/Decode/Execute stages
   - Reduces critical path to ~10Δ per stage → **100 MHz**
   - Adds 2 cycle latency for branches

2. **Parallel Decode**: Pre-decode common patterns
   - Reduce decode from 4Δ to 2Δ

3. **Register Bypass**: Forward ALU result to next instruction
   - Eliminates RegRead for dependent instructions

---

## 7. Implementation Phases

### Phase 1: Proof of Concept (Target: Working Adder)
- [ ] 8-bit ripple-carry adder from Add tiles
- [ ] Verify propagation delays match expected
- [ ] Benchmark: tiles/sec, critical path detection

### Phase 2: Minimal Datapath
- [ ] PC (single ProgramCounter tile)
- [ ] 4-word ROM (Const + Mux)
- [ ] Single register (RegEnable)
- [ ] Pass-through ALU (just Add)
- [ ] Execute: LDI, ADD, NOP

### Phase 3: Full ALU + Registers
- [ ] Complete ALU (Add, Sub, And, Or, Xor)
- [ ] 4-register file with decode
- [ ] All register-register instructions

### Phase 4: Branches + Memory
- [ ] Flag generation (Zero, Carry)
- [ ] Conditional branch logic
- [ ] 16-byte RAM
- [ ] Load/Store instructions

### Phase 5: Optimization
- [ ] Critical path profiling
- [ ] Pipeline implementation
- [ ] Sparse evaluation integration
- [ ] GPU acceleration for multi-CPU

### Phase 6: Scale
- [ ] 16-bit datapath
- [ ] 256-word ROM, 256-byte RAM
- [ ] Multi-core (multiple CPUs on grid)
- [ ] Interconnect fabric

---

## 8. Integration with Simulation Engine

### 8.1 Execution Loop

```rust
pub struct TileCpu {
    origin: (usize, usize),
    pc_tile: usize,      // Index of ProgramCounter tile
    reg_tiles: [usize; 4], // Indices of register tiles
    // ... other tile indices
}

impl TileCpu {
    /// Execute one instruction via tile simulation
    pub fn tick(&self, sim: &mut Simulation) -> TimingStats {
        // 1. Propagate combinational logic until stable
        let stats = sim.tick_with_delays(MAX_DELTAS);

        // 2. Clock edge - sequential elements capture
        sim.tick_clock();

        stats
    }

    /// Get current PC value by reading tile state
    pub fn read_pc(&self, sim: &Simulation) -> u8 {
        sim.get_logic_at(self.pc_tile) as u8
    }

    /// Run program and collect metrics
    pub fn run(&self, sim: &mut Simulation, max_cycles: u64) -> CpuMetrics {
        let mut total_deltas = 0u64;
        let mut max_critical_path = 0u32;

        for cycle in 0..max_cycles {
            let stats = self.tick(sim);
            total_deltas += stats.total_deltas as u64;
            max_critical_path = max_critical_path.max(stats.critical_path_deltas);

            // Check for HALT (PC pointing to itself)
            if self.is_halted(sim) {
                break;
            }
        }

        CpuMetrics {
            cycles: cycle,
            total_deltas,
            max_critical_path,
            estimated_mhz: 1000.0 / (max_critical_path as f64),
        }
    }
}
```

### 8.2 Sparse Evaluation Benefit

For a typical instruction:
- ~50 tiles are in the active datapath
- ~300 tiles are stable (unused ROM words, inactive registers)

Sparse evaluation: **6x speedup** for single CPU, **100x+** for multi-CPU grids

### 8.3 GPU Acceleration Path

```rust
// Multi-CPU execution on GPU
let cpus: Vec<TileCpu> = instantiate_cpu_grid(8, 8); // 64 CPUs
let mut ctx = GpuSparseContext::new(sim.width(), sim.height())?;

for cycle in 0..1_000_000 {
    gpu_sparse_tick(&mut ctx, &sim)?;
    sim.tick_clock();
}
// 64 CPUs × 1M cycles in seconds, not hours
```

---

## 9. Comparison to Prior Art

| System | Gate Count | Max Freq | Simulation Speed | Visual Debug |
|--------|-----------|----------|------------------|--------------|
| Minecraft Redstone CPU | ~50K blocks | 0.05 Hz | 20 ticks/sec | In-game |
| Ben Eater 8-bit | ~500 ICs | 1 MHz | N/A (physical) | LEDs |
| RISC-V softcore (FPGA) | ~15K LUTs | 100 MHz | N/A (physical) | Limited |
| **TileFabric 8-bit** | ~350 tiles | 38 MHz sim | **1B+ tiles/sec** | Full spatial |

**Key differentiator:** TileFabric is the only system that combines:
1. Gate-level accuracy
2. Real-time simulation at billions of ops/sec
3. Automatic timing analysis
4. Spatial visualization of signal flow

---

## 10. Future Directions

### 10.1 AI-Guided Placement (AlphaChip-style)

The tile grid is a natural substrate for reinforcement learning:
- **State:** Current tile placement
- **Action:** Move/swap tiles
- **Reward:** -critical_path_deltas

```rust
// RL optimization loop
loop {
    let metrics = cpu.run(sim, 1000);
    let reward = -metrics.max_critical_path as f64;
    agent.update(state, action, reward);
    state = agent.choose_action(state);
    apply_placement(sim, state);
}
```

### 10.2 Hardware Export

Tile layouts can be compiled to:
- **Verilog/VHDL** for FPGA synthesis
- **GDS** for ASIC tapeout (via OpenROAD)
- **PCB netlist** for discrete logic

### 10.3 Quantum-Classical Hybrid

The existing quantum infrastructure enables:
- Quantum ALU tiles (superposition arithmetic)
- Probabilistic branching via measurement
- Grover-accelerated search in tile space

---

## Appendix A: Tile Type Quick Reference

| Tile | Inputs | Output | Delay | Notes |
|------|--------|--------|-------|-------|
| Wire | L,R,U,D | L\|R\|U\|D | 1Δ | 4-way OR |
| WireH | L,R | L\|R | 1Δ | Horizontal only |
| WireV | U,D | U\|D | 1Δ | Vertical only |
| Cross | L,R,U,D | H=L\|R, V=U\|D | 1Δ | Non-interfering cross |
| And | L,R | L&R | 2Δ | |
| Or | L,R | L\|R | 2Δ | |
| Xor | L,R | L^R | 3Δ | |
| Not | L | ~L | 2Δ | |
| Add | L,R | L+R | 5Δ | Wrapping |
| Sub | L,R | L-R | 5Δ | Wrapping |
| Mux | L,R,U | U?L:R | 2Δ | |
| Mux8to1 | L,R | L[R*8+7:R*8] | 3Δ | Byte select |
| RegEnable | L,U,R | U&R?L:cur | 0Δ | Sync capture |
| ProgramCounter | L,R | R&1?L:PC+1 | 0Δ | Jump or increment |
| Const | - | preset | 0Δ | Configured value |
| Ram | L,U | U?L:cur | 0Δ | Write-enable gated |

---

## Appendix B: Instruction Encoding

```
Byte format: [OOOO][RRSS]
             │     │  └── Source register / Immediate (2 bits)
             │     └───── Destination register (2 bits)
             └─────────── Opcode (4 bits)

Examples:
  LDI R2, #3  = 0001_1011 = 0x1B  (opcode=1, rd=2, imm=3)
  ADD R0, R1  = 0011_0001 = 0x31  (opcode=3, rd=0, rs=1)
  JMP 0x20    = 1011_0000 = 0xB0 + addr in next byte
```

---

*Document version 1.0 - Ready for implementation*
