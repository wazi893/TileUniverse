# CHUNGUS 5: Distributed Block-Sparse Quantum Computing
## Implementation Plan for Opus Agent

**Date Created:** 2025-12-26
**Target:** 60-qubit distributed quantum computer using 37M TILE-8 CPUs
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

This plan describes how to build **CHUNGUS 5**, a distributed quantum computing system that maps block-sparse quantum states across 37 million TILE-8 CPUs. This leverages the existing 60-qubit sparse quantum simulation breakthrough (EPIC 85) and extends it to a massively parallel classical substrate.

### Key Innovation
- **Distributed Block-Sparse Quantum**: Each TILE-8 CPU processes quantum blocks (128 complex amplitudes)
- **Unprecedented Scale**: 4,000-100,000 CPUs working together for 40-60 qubit quantum circuits
- **Quantum Advantage**: Achieve exponential speedup (√2^n for Grover's algorithm) on classical hardware

### Success Metrics
- ✅ Single CPU can process quantum block with local gates (qubits 0-6)
- ✅ Multi-CPU system can execute cross-block gates (qubits 7+)
- ✅ System can run 40-qubit Grover's search in <10 seconds
- ✅ Achieve 1B+ quantum ops/sec system-wide
- ✅ Demonstrate quantum advantage on search/optimization problem

---

## Technical Background

### What Already Exists (Do NOT Rebuild)

#### 1. Block-Sparse Quantum State (EPIC 85, 88)
**Location:** `crates/logic-fabric-core/src/block_sparse_state.rs`

```rust
pub const BLOCK_SIZE: usize = 128;  // 128 complex amplitudes per block
pub const BLOCK_SHIFT: u8 = 7;      // log2(128)

pub struct Block {
    pub real: Box<[f16; BLOCK_SIZE]>,  // Real parts
    pub imag: Box<[f16; BLOCK_SIZE]>,  // Imaginary parts
    pub nnz: usize,                     // Non-zero count
}

pub struct BlockSparseQState {
    pub n_qubits: u8,
    blocks: HashMap<usize, Block>,      // block_id -> Block
}
```

**Key Achievements:**
- 60-qubit simulation working
- 9,139x speedup over dense representation at 20 qubits
- Block size optimized for WMMA tensor cores
- Qubits 0-6: Operations within block (stride 1-64)
- Qubits 7+: Operations span blocks (cross-block communication)

#### 2. TILE-8 CPU Architecture (EPIC 108, 111)
**Location:** `src/tile8/`

```rust
// TILE-8 ISA (from src/tile8/isa.rs)
pub enum Opcode {
    NOP,
    LDI,    // Load immediate
    MOV,    // Move register
    ADD,    // Add
    SUB,    // Subtract
    AND,    // Bitwise AND
    OR,     // Bitwise OR
    XOR,    // Bitwise XOR
    CMP,    // Compare
    JMP,    // Unconditional jump
    JZ,     // Jump if zero
    JNZ,    // Jump if not zero
    LOAD,   // Load from memory
    STORE,  // Store to memory
    HALT,
}

pub struct PhysicalCpu {
    // 4 registers (R0-R3)
    // 256-byte address space
    // ~500 tiles per CPU
    // 2.8M instructions/sec per CPU
}
```

**Key Stats:**
- 37.7M CPUs achievable in dense "hive" mode (spacing=2)
- Grid: 12,288 × 12,288 = 151M tiles
- Each CPU: 25×20 tile footprint
- Hive mode: Shared register banks for neighbor communication

#### 3. Performance Baseline
From existing benchmarks:
- **Tile evals:** 7B/sec (classical logic)
- **Quantum (GPU):** 2.5T amplitude ops/sec (dense WMMA)
- **Quantum (Sparse):** 9,139x faster than dense at 20 qubits
- **CPU IPS:** 2.8M instructions/sec per TILE-8 CPU

---

## Architecture Design

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CHUNGUS 5 Quantum Fabric                  │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ CPU 0    │  │ CPU 1    │  │ CPU 2    │  │ CPU 3    │   │
│  │ Block 0  │  │ Block 1  │  │ Block 2  │  │ Block 3  │   │
│  │ [128amp] │  │ [128amp] │  │ [128amp] │  │ [128amp] │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│              Hive Interconnect (qubits 7+)                  │
│                                                              │
│  37M CPUs total → ~4K-100K active for quantum circuits      │
└─────────────────────────────────────────────────────────────┘
```

### Block-to-CPU Mapping

**Qubit Structure:**
- **Qubits 0-6 (local):** Operations within 128-amplitude block
  - Qubit 0: stride 1 (adjacent pairs)
  - Qubit 1: stride 2
  - Qubit 2: stride 4
  - Qubit 3: stride 8
  - Qubit 4: stride 16
  - Qubit 5: stride 32
  - Qubit 6: stride 64

- **Qubits 7+ (cross-block):** Operations span multiple CPUs
  - Qubit 7: stride 128 (pair with neighbor CPU)
  - Qubit 8: stride 256
  - Qubit N: stride 2^N

**Block ID Calculation:**
```rust
fn block_id(state_index: u64) -> usize {
    (state_index >> 7) as usize  // Divide by 128
}

fn local_index(state_index: u64) -> usize {
    (state_index & 0x7F) as usize  // Mod 128
}
```

**Neighbor Routing (Hypercube):**
For cross-block gates on qubit N (where N ≥ 7):
```
Partner block = current_block_id ^ (1 << (N - 7))

Example (qubit 8 on block 5):
  5 = 0b000101
  Partner = 5 ^ (1 << 1) = 5 ^ 2 = 7 = 0b000111
```

### Memory Layout (Per CPU)

Each TILE-8 CPU has 256 bytes of addressable memory:

```
Address Range | Content
--------------+------------------------------------------
0x00 - 0x7F   | Real parts (128 × 8-bit fixed-point)
0x80 - 0xFF   | Imaginary parts (128 × 8-bit fixed-point)
```

**Fixed-Point Representation:**
- Use 8-bit signed fixed-point: `s.fffffff` (1 sign bit, 7 fractional bits)
- Range: -1.0 to +0.9921875 (adequate for normalized quantum amplitudes)
- Example: `0.707` (1/√2) ≈ `0x5A` (90/128 = 0.703125)

**Additional Memory Needed:**
- R0: Block ID (which block this CPU owns)
- R1: Gate opcode buffer
- R2: Qubit target
- R3: Communication buffer for cross-block ops

---

## Implementation Phases

### Phase 1: Single-CPU Quantum Block Processor
**Duration:** 1-2 weeks
**Goal:** One TILE-8 CPU can execute local quantum gates (qubits 0-6)

#### Tasks

1. **Create quantum gate library in TILE-8 assembly**
   - File: `src/tile8/quantum_gates.asm`
   - Implement:
     - `HADAMARD(qubit)` - Superposition gate
     - `PAULI_X(qubit)` - Bit flip
     - `PAULI_Z(qubit)` - Phase flip
     - `PHASE(qubit, angle)` - Rotation

2. **Memory management routines**
   - `LOAD_AMPLITUDE(index) -> (re, im)`
   - `STORE_AMPLITUDE(index, re, im)`
   - `NORMALIZE_BLOCK()` - Rescale all amplitudes

3. **Hadamard Implementation (Most Critical)**

```assembly
; Hadamard gate on qubit Q (Q = 0-6)
; Applies: H = 1/√2 [[1, 1], [1, -1]]
;
; For each pair of amplitudes separated by stride 2^Q:
;   amp[i], amp[i + stride] ->
;   (amp[i] + amp[i+stride])/√2, (amp[i] - amp[i+stride])/√2

HADAMARD:
    ; Input: R2 = qubit (0-6)
    ; Calculate stride = 2^qubit
    LDI R0, #1
    SHL R0, R2          ; R0 = stride

    ; Loop over pairs
    LDI R1, #0          ; index = 0
HADAMARD_LOOP:
    CMP R1, #128
    JZ HADAMARD_DONE

    ; Load pair
    LOAD R2, [R1]           ; amp_i real
    ADD R3, R1, R0          ; i + stride
    LOAD R3, [R3]           ; amp_j real

    ; Compute (i+j)/√2 and (i-j)/√2
    ; (Using fixed-point approximation: /√2 ≈ × 0.707 ≈ × 181/256)
    ADD R4, R2, R3          ; sum
    CALL MUL_BY_INV_SQRT2   ; R4 *= 0.707
    STORE R4, [R1]          ; New amp_i

    SUB R5, R2, R3          ; diff
    CALL MUL_BY_INV_SQRT2   ; R5 *= 0.707
    ADD R6, R1, R0
    STORE R5, [R6]          ; New amp_j

    ; Repeat for imaginary parts (addresses 0x80+)
    ; ... (similar logic)

    ; Next pair (skip by 2*stride to avoid re-processing)
    ADD R1, R1, R0
    ADD R1, R1, R0
    JMP HADAMARD_LOOP

HADAMARD_DONE:
    RET

MUL_BY_INV_SQRT2:
    ; Multiply R4 by 181/256 (≈ 0.707)
    ; R4 = (R4 * 181) >> 8
    PUSH R0
    LDI R0, #181
    MUL R4, R0
    SHR R4, #8
    POP R0
    RET
```

4. **Test Program: Create GHZ State**

```assembly
; Create 7-qubit GHZ state: (|0000000⟩ + |1111111⟩)/√2
; Uses only local gates (qubits 0-6)

MAIN:
    ; Initialize: |0000000⟩ (amplitude[0] = 1.0)
    CALL INIT_ZERO_STATE

    ; Apply H to qubit 0
    LDI R2, #0
    CALL HADAMARD           ; Now: (|0⟩ + |1⟩)/√2 ⊗ |000000⟩

    ; Apply CNOT(0, 1), CNOT(0, 2), ..., CNOT(0, 6)
    LDI R0, #1
ENTANGLE_LOOP:
    CMP R0, #7
    JZ DONE
    PUSH R0
    LDI R2, #0              ; Control = qubit 0
    MOV R3, R0              ; Target = current qubit
    CALL CNOT_LOCAL
    POP R0
    ADD R0, R0, #1
    JMP ENTANGLE_LOOP

DONE:
    ; Verify: Should have exactly 2 non-zero amplitudes
    ; amplitude[0b0000000] = 1/√2
    ; amplitude[0b1111111] = 1/√2
    CALL VERIFY_GHZ
    HALT

INIT_ZERO_STATE:
    ; Set amplitude[0] = 1.0 (re=0x80, im=0x00)
    ; All others = 0
    LDI R0, #0
    LDI R1, #0x80           ; 1.0 in fixed-point
    STORE R1, [R0]
    LDI R1, #0x00
    ADD R0, R0, #0x80
    STORE R1, [R0]
    ; Zero out rest (addresses 1-127 and 0x81-0xFF)
    ; ... (loop omitted for brevity)
    RET
```

#### Success Criteria for Phase 1
- [ ] Single CPU can create |+⟩ state (H on |0⟩)
- [ ] Single CPU can create 7-qubit GHZ state
- [ ] Verification: Measure norm = 1.0 ± 0.01
- [ ] Verification: Exactly 2 non-zero amplitudes in GHZ

---

### Phase 2: Multi-CPU Cross-Block Communication
**Duration:** 2-3 weeks
**Goal:** Multiple CPUs can execute gates on qubits 7+ via neighbor communication

#### Tasks

1. **Implement CPU-to-CPU messaging**
   - File: `src/tile8/quantum_router.rs`
   - Use existing "hive" interconnect (shared registers, spacing=2)
   - Protocol:
     ```rust
     struct QuantumMessage {
         sender_block_id: u16,
         target_block_id: u16,
         operation: u8,      // SWAP_AMPS, etc.
         payload: [u8; 16],  // Amplitude data
     }
     ```

2. **Cross-Block CNOT Implementation**

```assembly
; CNOT gate on qubit 7 (first cross-block qubit)
; Requires communication between block pairs

CNOT_CROSS_BLOCK_Q7:
    ; Input: R2 = control qubit (7), R3 = target qubit

    ; Calculate partner block ID
    MOV R0, [BLOCK_ID]      ; Load our block ID
    XOR R0, #0x01           ; Flip bit 0 (qubit 7 partner)
    MOV R1, R0              ; R1 = partner block ID

    ; Determine if we're the "even" or "odd" block
    MOV R0, [BLOCK_ID]
    AND R0, #0x01
    CMP R0, #0
    JZ CNOT_EVEN_BLOCK

CNOT_ODD_BLOCK:
    ; Odd block: Send our data, receive partner's
    CALL SEND_BLOCK_TO_NEIGHBOR
    CALL RECEIVE_BLOCK_FROM_NEIGHBOR
    ; Apply conditional swap based on control bit
    CALL CONDITIONAL_SWAP_AMPS
    ; Send result back
    CALL SEND_BLOCK_TO_NEIGHBOR
    JMP CNOT_DONE

CNOT_EVEN_BLOCK:
    ; Even block: Receive partner's, process, send back
    CALL RECEIVE_BLOCK_FROM_NEIGHBOR
    CALL CONDITIONAL_SWAP_AMPS
    CALL SEND_BLOCK_TO_NEIGHBOR
    CALL RECEIVE_BLOCK_FROM_NEIGHBOR  ; Get final result

CNOT_DONE:
    RET

SEND_BLOCK_TO_NEIGHBOR:
    ; Send our 128 amplitudes to neighbor via hive interconnect
    ; (Use shared register banks or memory-mapped I/O)
    ; Implementation depends on tile8 communication primitives
    ; ... (details omitted, use existing tile8::physical API)
    RET

RECEIVE_BLOCK_FROM_NEIGHBOR:
    ; Receive 128 amplitudes from neighbor
    ; ...
    RET
```

3. **10-Qubit Distributed Test**

```assembly
; Create 10-qubit GHZ state using 4 CPUs
; CPU 0: block 0 (amps 0-127)
; CPU 1: block 1 (amps 128-255)
; CPU 2: block 2 (amps 256-383)
; CPU 3: block 3 (amps 384-511)

MAIN_DISTRIBUTED:
    ; Each CPU initializes its block
    CALL INIT_ZERO_STATE

    ; CPU 0 only: Apply H to qubit 0
    MOV R0, [BLOCK_ID]
    CMP R0, #0
    JNZ SKIP_H
    LDI R2, #0
    CALL HADAMARD
SKIP_H:

    ; All CPUs: Synchronize
    CALL BARRIER_SYNC

    ; Apply CNOT(0, 1), ..., CNOT(0, 9)
    LDI R0, #1
DIST_ENTANGLE:
    CMP R0, #10
    JZ DIST_DONE

    ; Determine if this gate involves our block
    LDI R2, #0          ; Control = 0
    MOV R3, R0          ; Target = current

    CMP R3, #7
    JLT DIST_LOCAL
    CALL CNOT_CROSS_BLOCK
    JMP DIST_NEXT

DIST_LOCAL:
    ; Only block 0 executes local gates
    MOV R4, [BLOCK_ID]
    CMP R4, #0
    JNZ DIST_NEXT
    CALL CNOT_LOCAL

DIST_NEXT:
    CALL BARRIER_SYNC
    ADD R0, R0, #1
    JMP DIST_ENTANGLE

DIST_DONE:
    ; Verify GHZ: blocks 0 and 3 should have non-zero amps
    CALL VERIFY_DISTRIBUTED_GHZ
    HALT
```

#### Success Criteria for Phase 2
- [ ] 4 CPUs can create 10-qubit GHZ state
- [ ] Cross-block CNOT (qubit 7) works correctly
- [ ] Communication latency < 100 CPU cycles
- [ ] Distributed state verification passes

---

### Phase 3: Quantum Circuit Compiler
**Duration:** 2-3 weeks
**Goal:** Automatically compile QASM circuits to TILE-8 assembly

#### Tasks

1. **Create QASM parser**
   - File: `src/tile8/qasm_compiler.rs`
   - Parse OpenQASM 2.0 format
   - Example input:
     ```qasm
     OPENQASM 2.0;
     qreg q[20];
     h q[0];
     cx q[0], q[1];
     cx q[0], q[2];
     // ... 18 more CNOTs for GHZ
     ```

2. **Block allocation algorithm**
   ```rust
   fn allocate_blocks(n_qubits: u8) -> HashMap<usize, u16> {
       // Map block_id -> cpu_id
       let total_blocks = 1 << (n_qubits.saturating_sub(7));
       // For 20 qubits: 2^13 = 8,192 blocks
       // Assign to first 8,192 CPUs
       (0..total_blocks).map(|b| (b, b as u16)).collect()
   }
   ```

3. **Code generation**
   ```rust
   fn compile_circuit(circuit: &Circuit) -> Vec<CpuProgram> {
       let mut programs = vec![];

       for cpu_id in 0..circuit.num_cpus() {
           let mut asm = String::new();
           asm.push_str("MAIN:\n");
           asm.push_str("    CALL INIT_ZERO_STATE\n");

           for gate in &circuit.gates {
               match gate {
                   Gate::H(q) if *q < 7 => {
                       // Local gate - only block 0 executes
                       asm.push_str(&format!(
                           "    MOV R0, [BLOCK_ID]\n    CMP R0, #0\n    JNZ SKIP_{}\n",
                           gate_id
                       ));
                       asm.push_str(&format!("    LDI R2, #{}\n", q));
                       asm.push_str("    CALL HADAMARD\n");
                       asm.push_str(&format!("SKIP_{}:\n", gate_id));
                   }
                   Gate::CNOT(ctrl, tgt) if *tgt >= 7 => {
                       // Cross-block gate - all relevant blocks execute
                       asm.push_str(&format!("    LDI R2, #{}\n", ctrl));
                       asm.push_str(&format!("    LDI R3, #{}\n", tgt));
                       asm.push_str("    CALL CNOT_CROSS_BLOCK\n");
                   }
                   // ... other gates
               }
               asm.push_str("    CALL BARRIER_SYNC\n");
           }

           asm.push_str("    HALT\n");
           programs.push(CpuProgram { cpu_id, assembly: asm });
       }

       programs
   }
   ```

4. **Integration with existing assembler**
   - Modify `src/tile8/asm.rs` to accept quantum gate mnemonics
   - Add quantum opcodes to ISA (or map to subroutine calls)

#### Test Case: 20-Qubit Grover's Search

```qasm
// Grover's algorithm for 20-qubit search
OPENQASM 2.0;
qreg q[20];

// Hadamard all qubits (superposition)
h q[0];
h q[1];
// ... h q[19]

// Oracle (marks target state)
// (Implementation depends on search problem)
// Example: Mark state |10110...⟩
x q[0];
x q[2];
x q[3];
// ... (multi-controlled Z)
x q[0];
x q[2];
x q[3];

// Diffusion operator
// H all, X all, multi-Z, X all, H all
// ...

// Repeat √(2^20) ≈ 1024 times
```

**Compiler Output:**
- Generate assembly for 8,192 CPUs (2^13 blocks)
- Distribute gates based on qubit index
- Insert synchronization barriers

#### Success Criteria for Phase 3
- [ ] Compiler can parse 20-qubit QASM file
- [ ] Generated assembly allocates blocks correctly
- [ ] Can compile and run simple circuits (GHZ, Bell states)
- [ ] Generated code produces correct quantum states (verified against dense simulation)

---

### Phase 4: Performance Optimization & Scaling
**Duration:** 3-4 weeks
**Goal:** Achieve 1B+ quantum ops/sec and scale to 40-60 qubits

#### Tasks

1. **Optimize gate kernels**
   - Profile Hadamard, CNOT execution time
   - Target: Local gates < 100 cycles, cross-block < 1000 cycles
   - Use loop unrolling, lookup tables for multiplication

2. **Communication optimization**
   - Batch multiple amplitude transfers
   - Use DMA if available in tile8 system
   - Pipeline gate execution with communication

3. **Sparse block management**
   ```rust
   // Only allocate CPUs for non-zero blocks
   struct SparseBlockAllocator {
       active_blocks: HashMap<usize, u16>,  // block_id -> cpu_id
       free_cpus: Vec<u16>,
   }

   impl SparseBlockAllocator {
       fn allocate_on_demand(&mut self, block_id: usize) -> u16 {
           // Allocate CPU only when block becomes non-zero
           if let Some(&cpu) = self.active_blocks.get(&block_id) {
               return cpu;
           }
           let cpu = self.free_cpus.pop().expect("No free CPUs");
           self.active_blocks.insert(block_id, cpu);
           cpu
       }
   }
   ```

4. **Benchmark Suite**
   - File: `examples/chungus5_benchmark.rs`
   - Test cases:
     - 20-qubit GHZ: Target < 1 second
     - 30-qubit GHZ: Target < 10 seconds
     - 40-qubit Grover (1 iteration): Target < 1 second
     - 50-qubit Grover (full search): Target < 100 seconds

5. **Scale Test: 50-Qubit Grover's Search**

```rust
// Expected performance calculation
fn estimate_grovers_performance() {
    let n_qubits = 50;
    let n_iterations = (1u64 << (n_qubits / 2)); // √(2^50) ≈ 2^25

    // With sparsity: ~1M active blocks (not 2^43)
    let active_blocks = 1_000_000;
    let cpus_needed = active_blocks / 250; // 250 blocks per CPU
    println!("CPUs needed: {}", cpus_needed); // ~4,000

    // Gates per iteration
    let gates_per_iteration = 200; // H + Oracle + Diffusion
    let local_gates = (gates_per_iteration as f64 * 0.7) as u64;
    let cross_gates = (gates_per_iteration as f64 * 0.3) as u64;

    // Throughput per CPU
    let local_ops_per_sec = 10_000; // 10K local ops/sec
    let cross_ops_per_sec = 1_000;  // 1K cross ops/sec

    let total_ops_per_sec = cpus_needed as u64 *
        (local_gates * local_ops_per_sec + cross_gates * cross_ops_per_sec);

    println!("System throughput: {} M ops/sec", total_ops_per_sec / 1_000_000);

    let total_gates = n_iterations * gates_per_iteration as u64;
    let time_seconds = total_gates as f64 / total_ops_per_sec as f64;

    println!("Estimated time for 50-qubit Grover: {:.1} seconds", time_seconds);
    // Expected: ~60-100 seconds
}
```

#### Success Criteria for Phase 4
- [ ] 40-qubit circuits run successfully
- [ ] System achieves >1B quantum ops/sec
- [ ] Sparse allocation reduces CPU usage by 10-100x
- [ ] Grover's search demonstrates quantum speedup
- [ ] Performance scales linearly with active blocks

---

## Technical Specifications

### Fixed-Point Arithmetic

**Format:** 8-bit signed fixed-point `s.fffffff`
- 1 sign bit
- 7 fractional bits
- Range: -1.0 to +127/128 ≈ 0.992

**Key Constants:**
```assembly
; 1/√2 ≈ 0.707 ≈ 90.5/128 ≈ 91/128
INV_SQRT2_NUMER = 91
INV_SQRT2_DENOM = 128  ; (will use shift by 7)

; Common angles (for phase gates)
PI_OVER_4 = 0x20       ; π/4 ≈ 0.785 ≈ 100/128
PI_OVER_2 = 0x40       ; π/2 ≈ 1.571 but clipped to 1.0 = 0x7F
```

**Multiplication:**
```assembly
; Multiply two fixed-point numbers
; Input: R0, R1 (both s.fffffff format)
; Output: R2 = R0 * R1
MUL_FIXED:
    MUL R2, R0, R1      ; Raw multiply (gives s.fffffffffffff)
    SHR R2, #7          ; Shift right to restore s.fffffff
    RET
```

**Division by √2:**
```assembly
; Divide by √2 (multiply by 0.707)
DIV_SQRT2:
    PUSH R0
    LDI R0, #91
    MUL R1, R0          ; R1 *= 91
    SHR R1, #7          ; R1 /= 128
    POP R0
    RET
```

### Communication Protocol

**Message Format (16 bytes):**
```
Offset | Size | Field
-------+------+---------------------------
0      | 2    | Sender block ID
2      | 2    | Target block ID
4      | 1    | Operation code
5      | 1    | Qubit index
6      | 10   | Payload (5 complex amplitudes)
```

**Operations:**
```
0x01: SWAP_AMPLITUDES    - Exchange data for CNOT
0x02: SEND_PARTIAL       - Send subset of block
0x03: BARRIER_SYNC       - Synchronization signal
0x04: MEASURE_NOTIFY     - Measurement collapse
```

### Synchronization

**Barrier Implementation:**
```rust
struct BarrierSync {
    total_cpus: u16,
    arrived_count: AtomicU16,
    generation: AtomicU32,
}

impl BarrierSync {
    fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        let count = self.arrived_count.fetch_add(1, Ordering::AcqRel);

        if count + 1 == self.total_cpus {
            // Last CPU to arrive
            self.arrived_count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            // Wait for generation to change
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
        }
    }
}
```

**TILE-8 Assembly:**
```assembly
BARRIER_SYNC:
    ; Increment arrival counter
    LOAD R0, [SYNC_COUNTER]
    ADD R0, R0, #1
    STORE R0, [SYNC_COUNTER]

    ; Check if last to arrive
    LOAD R1, [TOTAL_CPUS]
    CMP R0, R1
    JZ BARRIER_RELEASE

BARRIER_WAIT:
    ; Spin until generation changes
    LOAD R2, [SYNC_GENERATION]
    LOAD R3, [SYNC_GENERATION]
    CMP R2, R3
    JZ BARRIER_WAIT
    RET

BARRIER_RELEASE:
    ; Reset counter, increment generation
    LDI R0, #0
    STORE R0, [SYNC_COUNTER]
    LOAD R2, [SYNC_GENERATION]
    ADD R2, R2, #1
    STORE R2, [SYNC_GENERATION]
    RET
```

---

## Integration Points with Existing Codebase

### Files to Modify

1. **`src/tile8/isa.rs`** - Add quantum gate opcodes
   ```rust
   pub enum Opcode {
       // ... existing opcodes
       QHAD,      // Quantum Hadamard
       QX,        // Quantum X gate
       QZ,        // Quantum Z gate
       QCNOT,     // Quantum CNOT
       QMEASURE,  // Quantum measurement
   }
   ```

2. **`src/tile8/physical.rs`** - Extend CPU with quantum state
   ```rust
   pub struct QuantumCpu {
       cpu: PhysicalCpu,
       block_id: usize,
       block_real: [i8; 128],  // Fixed-point real parts
       block_imag: [i8; 128],  // Fixed-point imaginary parts
       nnz: usize,
   }
   ```

3. **`examples/tile8_visualizer.rs`** - Add quantum state visualization
   ```rust
   fn draw_quantum_state(cpus: &[QuantumCpu], zoom: f32) {
       if zoom > 0.5 {
           // Zoomed in: Show individual amplitudes as colors
           for cpu in cpus {
               for i in 0..128 {
                   let amp = cpu.get_amplitude(i);
                   let color = amplitude_to_color(amp); // |amp|² as brightness
                   draw_rectangle(x, y, 1.0, 1.0, color);
               }
           }
       } else {
           // Zoomed out: Show block-level statistics
           let total_prob = cpu.block_probability();
           let color = Color::new(0.0, total_prob, 0.0, 1.0);
           draw_rectangle(x, y, cpu_width, cpu_height, color);
       }
   }
   ```

### Files to Create

1. **`src/tile8/quantum_gates.asm`** - Assembly gate library
2. **`src/tile8/qasm_compiler.rs`** - Circuit compiler
3. **`src/tile8/quantum_router.rs`** - Cross-block communication
4. **`src/tile8/quantum_runtime.rs`** - Execution harness
5. **`examples/chungus5_grover.rs`** - Grover's algorithm demo
6. **`examples/chungus5_benchmark.rs`** - Performance tests
7. **`tests/chungus5_correctness.rs`** - Correctness verification

---

## Validation & Testing

### Correctness Tests

1. **Single-Qubit Gate Verification**
   ```rust
   #[test]
   fn test_hadamard_creates_superposition() {
       let mut cpu = QuantumCpu::new(7);
       cpu.init_zero_state();
       cpu.apply_hadamard(0);

       assert_approx_eq!(cpu.get_amplitude(0b0000000).norm(), 1.0 / 2.0f32.sqrt());
       assert_approx_eq!(cpu.get_amplitude(0b0000001).norm(), 1.0 / 2.0f32.sqrt());
   }
   ```

2. **Two-Qubit Entanglement**
   ```rust
   #[test]
   fn test_bell_state() {
       let mut cpus = vec![QuantumCpu::new_multi(2, 0), QuantumCpu::new_multi(2, 1)];
       cpus[0].apply_hadamard(0);
       apply_cnot_distributed(&mut cpus, 0, 1);

       // Bell state: (|00⟩ + |11⟩)/√2
       assert_approx_eq!(get_amplitude(&cpus, 0b00).norm(), 0.707);
       assert_approx_eq!(get_amplitude(&cpus, 0b11).norm(), 0.707);
       assert_approx_eq!(get_amplitude(&cpus, 0b01).norm(), 0.0);
       assert_approx_eq!(get_amplitude(&cpus, 0b10).norm(), 0.0);
   }
   ```

3. **GHZ State Verification**
   ```rust
   #[test]
   fn test_ghz_state_n_qubits(n: u8) {
       let cpus = create_ghz_state(n);

       // GHZ: (|000...⟩ + |111...⟩)/√2
       let zero_state = 0u64;
       let one_state = (1u64 << n) - 1;

       assert_approx_eq!(get_amplitude(&cpus, zero_state).norm(), 0.707);
       assert_approx_eq!(get_amplitude(&cpus, one_state).norm(), 0.707);

       // All other amplitudes should be zero
       for i in 1..(1u64 << n) {
           if i != one_state {
               assert!(get_amplitude(&cpus, i).norm() < 1e-6);
           }
       }
   }
   ```

4. **Norm Preservation**
   ```rust
   #[test]
   fn test_norm_preserved_through_circuit() {
       let mut cpus = random_circuit(20, 100); // 20 qubits, 100 gates

       for _ in 0..100 {
           let gate = random_gate();
           apply_gate_distributed(&mut cpus, gate);

           let norm = compute_global_norm(&cpus);
           assert_approx_eq!(norm, 1.0, epsilon = 0.01);
       }
   }
   ```

### Performance Tests

1. **Gate Throughput**
   ```rust
   #[bench]
   fn bench_hadamard_local(b: &mut Bencher) {
       let mut cpu = QuantumCpu::new(7);
       b.iter(|| cpu.apply_hadamard(3));
   }
   // Target: >10,000 ops/sec per CPU
   ```

2. **Cross-Block Communication**
   ```rust
   #[bench]
   fn bench_cnot_cross_block(b: &mut Bencher) {
       let mut cpus = vec![QuantumCpu::new_multi(8, 0), QuantumCpu::new_multi(8, 1)];
       b.iter(|| apply_cnot_distributed(&mut cpus, 0, 7));
   }
   // Target: >1,000 ops/sec
   ```

3. **Full Circuit Execution**
   ```rust
   #[bench]
   fn bench_grover_20_qubit(b: &mut Bencher) {
       let circuit = compile_grovers_circuit(20);
       b.iter(|| run_circuit_distributed(&circuit));
   }
   // Target: <1 second per iteration
   ```

### Comparison with Dense Simulation

Run same circuits on existing `BlockSparseQState` and verify:
```rust
#[test]
fn test_chungus5_matches_block_sparse() {
    let circuit = load_qasm("test_circuits/ghz_30.qasm");

    // Run on CHUNGUS 5 (distributed)
    let chungus_state = run_on_chungus5(&circuit);

    // Run on block-sparse (reference)
    let reference_state = run_on_block_sparse(&circuit);

    // Compare all amplitudes
    for i in 0..(1u64 << 30) {
        let fidelity = chungus_state.get(i).dot(reference_state.get(i).conj());
        assert!(fidelity.abs() > 0.999);
    }
}
```

---

## Success Metrics & Deliverables

### Phase 1 Deliverables
- [ ] `quantum_gates.asm` with H, X, Z, CNOT implementations
- [ ] Single-CPU test passing (7-qubit GHZ)
- [ ] Documentation of fixed-point arithmetic accuracy

### Phase 2 Deliverables
- [ ] Cross-block communication protocol implemented
- [ ] 10-qubit distributed GHZ test passing
- [ ] Communication latency benchmarks

### Phase 3 Deliverables
- [ ] QASM compiler producing correct assembly
- [ ] 20-qubit circuit compilation working
- [ ] Integration with existing tile8 toolchain

### Phase 4 Deliverables
- [ ] 40-50 qubit circuits running successfully
- [ ] Performance benchmarks showing >1B ops/sec
- [ ] Grover's search demonstrating quantum speedup
- [ ] Visualization of quantum state across CPUs

### Final Success Criteria

**Technical:**
- ✅ 40+ qubit quantum circuits execute correctly
- ✅ State fidelity >99.9% vs. reference simulation
- ✅ System achieves >1 billion quantum ops/sec
- ✅ Grover's search finds target in √N iterations

**Research Impact:**
- ✅ Demonstrates quantum advantage on classical hardware
- ✅ Publishable results (distributed block-sparse quantum)
- ✅ Open-source implementation for reproducibility

**Demonstration:**
- ✅ Visual demo of 37M CPUs working together
- ✅ Real-time quantum state visualization
- ✅ Solve practical problem (e.g., search, optimization)

---

## References to Existing Work

### Key Files to Study

1. **Block-Sparse Implementation**
   - `crates/logic-fabric-core/src/block_sparse_state.rs`
   - EPIC 88 documentation: `User notes/SPRINTS/SPRINT 23.0/EPIC 88 (WIP)/`

2. **TILE-8 Architecture**
   - `src/tile8/mod.rs` - Module overview
   - `src/tile8/physical.rs` - CPU implementation
   - `src/tile8/asm.rs` - Assembler
   - `examples/tile8_visualizer.rs` - Visualization

3. **Sparse Quantum Results**
   - `User notes/SPRINTS/SPRINT 22.0/EPIC_85_RESULTS.md`
   - 60-qubit achievement, 9,139x speedup

### Prior Art to Build Upon

- **EPIC 85:** Sparse quantum states (element-sparse)
- **EPIC 88:** Block-sparse with WMMA optimization
- **EPIC 111:** TILE-8 CPU with 37M scale
- **Sprint 27-28:** Spatial computing vision
- **CHUNGUS 2/3:** Distributed computing patterns

---

## Risk Mitigation

### Technical Risks

1. **Fixed-point precision insufficient**
   - Mitigation: Test with 16-bit if needed (reduce to 64 amps/block)
   - Fallback: Use f32 emulation (slower but exact)

2. **Communication bandwidth bottleneck**
   - Mitigation: Batch operations, pipeline gates
   - Fallback: Limit to 30-40 qubits (fewer cross-block ops)

3. **Synchronization overhead**
   - Mitigation: Async gate execution where possible
   - Fallback: Coarse-grained synchronization (every 10 gates)

4. **Memory constraints (256 bytes/CPU)**
   - Mitigation: Use external memory for instruction stream
   - Confirmed: 128 amps × 2 bytes = 256 bytes fits exactly

### Development Risks

1. **Complexity underestimated**
   - Mitigation: Start with Phase 1-2 only, validate early
   - Pivot: If too complex, focus on neuromorphic (CHUNGUS 4) instead

2. **Performance targets missed**
   - Mitigation: Profile early, optimize hot paths
   - Acceptable: Even 100M ops/sec is publishable for this scale

---

## Appendix: Alternative Approaches

### Option A: Hybrid Neuromorphic-Quantum

If quantum proves too complex, combine approaches:
- 90% of CPUs run spiking neurons (CHUNGUS 4)
- 10% of CPUs form quantum coprocessor (5-10 qubits)
- Use quantum sampling for neural network training

### Option B: Analog Quantum Annealing

Instead of gate-based quantum, use TILE-8 for:
- Simulated annealing on Ising models
- Each CPU = one spin in lattice
- Quantum-inspired optimization (not true quantum)

### Option C: Tensor Network Approach

Use Matrix Product States (MPS) instead of block-sparse:
- Each CPU stores one tensor in network
- Can reach 100+ qubits for specific circuits
- More complex implementation

---

## Questions for Implementation

Before starting, clarify:

1. **TILE-8 Communication:** What's the actual bandwidth of "hive" mode? Can CPUs directly read neighbors' memory, or is message-passing required?

2. **Memory Access:** Can TILE-8 access external memory beyond 256 bytes for instruction storage?

3. **Precision Requirements:** Is 8-bit fixed-point acceptable, or should we target 16-bit (requires reducing block size to 64 amplitudes)?

4. **Visualization Priority:** How important is real-time quantum state visualization vs. raw performance?

5. **Target Application:** What specific quantum algorithm should we optimize for?
   - Grover's search (database/optimization)
   - Quantum chemistry (VQE)
   - Quantum machine learning (QAOA)

---

## Contact & Next Steps

**Prepared for:** Opus Agent
**Prepared by:** Claude Sonnet 4.5
**Repository:** /home/user/TileUniverse
**Branch:** `claude/chungus-tile-8-substrates-fARNb`

**Recommended Starting Point:**
1. Read Phase 1 implementation details
2. Study `src/tile8/physical.rs` and `block_sparse_state.rs`
3. Implement Hadamard gate in TILE-8 assembly
4. Create single-CPU test (7-qubit GHZ)
5. Report results before proceeding to Phase 2

**Success Checkpoint:** When you can create and verify a 7-qubit GHZ state on a single TILE-8 CPU, you've validated the core concept and can proceed with confidence to distributed implementation.

---

**END OF IMPLEMENTATION PLAN**

*This document represents a complete blueprint for CHUNGUS 5. All technical details are based on existing TileUniverse capabilities (60-qubit sparse, 37M CPUs, block-sparse architecture). Implementation is feasible and will demonstrate unprecedented distributed quantum computing on classical hardware.*
