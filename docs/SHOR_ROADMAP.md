# Shor's Algorithm Scaling Roadmap

## The Goal
Build the most capable open-source Shor's algorithm implementation, pushing to the limits of classical simulation.

## Current State (Sprint 60)
- **Factors**: 15, 21 only (hardcoded circuits)
- **Qubits**: ~12
- **Dependency**: Python + Qiskit for circuit synthesis
- **Limitation**: No native modular arithmetic

---

## Phase 1: Native Modular Arithmetic (Sprint 61)

### Objective
Remove Qiskit dependency. Build pure-Rust modular exponentiation circuits.

### Components Needed

```
┌─────────────────────────────────────────────────────────────┐
│                  MODULAR ARITHMETIC STACK                    │
├─────────────────────────────────────────────────────────────┤
│  Modular Exponentiation: a^x mod N                          │
│    └── Repeated Squaring: a^(2^k) mod N                     │
│         └── Modular Multiplication: (a × b) mod N           │
│              └── Multiplication: a × b                       │
│                   └── Addition: a + b                        │
│                        └── Ripple Carry Adder               │
│                             └── Full Adder (Toffoli gates)  │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Order
1. **Quantum Full Adder** - 2 Toffoli gates + 1 CNOT
2. **Ripple Carry Adder** - n full adders for n-bit addition
3. **Controlled Addition** - Add only if control qubit is |1⟩
4. **Modular Addition** - (a + b) mod N with comparison and subtraction
5. **Modular Multiplication** - Shift-and-add algorithm
6. **Modular Exponentiation** - Square-and-multiply algorithm

### Deliverable
Factor any N < 256 (8-bit) with ~24 qubits, pure Rust, no Python.

---

## Phase 2: Memory-Optimized Simulation (Sprint 62)

### Objective
Push qubit limit from ~25 to ~35 using memory optimizations.

### Techniques
1. **Amplitude Compression**
   - Store amplitudes as f32 instead of f64 (2× memory savings)
   - Acceptable for Shor's since we only need period, not precise amplitudes

2. **Gate Fusion for QFT**
   - Fuse consecutive rotation gates
   - Reduce memory bandwidth bottleneck

3. **Lazy Evaluation**
   - Don't materialize full state until measurement
   - Stream amplitudes through gates

4. **Memory-Mapped State Vector**
   - Use mmap to extend beyond RAM
   - SSD-backed quantum state (~100× slower but enables larger states)

### Deliverable
Factor any N < 65,536 (16-bit) with ~35 qubits on 32GB RAM machine.

---

## Phase 3: Qubit Recycling (Sprint 63)

### Objective
Implement Politi et al. (2009) qubit recycling to reduce register size.

### Concept
Instead of 2n counting qubits, use 1 qubit recycled 2n times:
```
Standard:  |ψ⟩ = |q₀⟩|q₁⟩|q₂⟩...|q₂ₙ₋₁⟩|target⟩
Recycled:  |ψ⟩ = |q₀⟩|target⟩  (measure q₀, reset, reuse)
```

### Tradeoff
- Pro: Reduces qubit count by factor of 2n
- Con: Requires 2n sequential measurements (no parallelism)
- Con: Each measurement introduces decoherence in real hardware

### For Simulation
This is HUGE - we can trade time for space:
- 32-bit factoring: 67 qubits → ~35 qubits (16GB feasible)
- 64-bit factoring: 131 qubits → ~67 qubits (still impossible dense)

### Deliverable
Factor any N < 2^24 (24-bit) with ~30 qubits using recycling.

---

## Phase 4: Tensor Network Backend (Sprint 64-65)

### Objective
Implement Matrix Product State (MPS) representation for low-entanglement simulation.

### Why This Helps
Shor's algorithm has bounded entanglement entropy during most of the computation:
- QFT creates O(log n) entanglement
- Modular exponentiation creates O(n) entanglement at worst
- Can truncate small Schmidt coefficients with controlled error

### Implementation
```rust
pub struct MPSState {
    tensors: Vec<Array3<Complex64>>,  // Site tensors
    bond_dims: Vec<usize>,             // Bond dimensions (controls accuracy)
    max_bond: usize,                   // Truncation threshold
}

impl MPSState {
    fn apply_two_qubit_gate(&mut self, q1: usize, q2: usize, gate: &Gate) {
        // Contract, apply gate, SVD decompose, truncate
    }
}
```

### Expected Gains
- Low-entanglement states: 100+ qubits possible
- Shor's algorithm: Maybe 50-60 qubits before entanglement explodes
- Tradeoff: Approximate (controlled error), slower per gate

### Deliverable
Factor any N < 2^32 (32-bit) with MPS + truncation (approximate but verifiable).

---

## Phase 5: Distributed Simulation (Sprint 66-67)

### Objective
Distribute state vector across multiple machines.

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node 0    │     │   Node 1    │     │   Node 2    │
│ Amps 0-2³⁰  │────│ Amps 2³⁰-2³¹│────│ Amps 2³¹-2³²│
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      └───────────────────┴───────────────────┘
                    MPI / RDMA
```

### Communication Pattern
- Single-qubit gates: Local (no communication)
- Two-qubit gates on qubits in same partition: Local
- Two-qubit gates across partitions: All-to-all exchange

### Scaling
- 4 nodes × 64GB = 256GB → ~37 qubits
- 64 nodes × 64GB = 4TB → ~41 qubits
- 1024 nodes × 64GB = 64TB → ~45 qubits

### Deliverable
Factor any N < 2^40 (40-bit) on a cluster.

---

## The Hard Wall

Even with all optimizations, classical simulation hits a wall:

| Approach | Max Qubits | Max N Factorable |
|----------|------------|------------------|
| Dense (single machine) | ~35 | 2^16 = 65,536 |
| Qubit recycling | ~40 | 2^20 = 1,048,576 |
| Tensor networks | ~50-60 | 2^25 = 33,554,432 |
| Distributed (1000 nodes) | ~50 | 2^25 |

**RSA-2048 requires ~4,099 qubits. The gap is insurmountable.**

---

## What Would Be Genuinely Impressive

### Tier 1: Research Demo (Current + Sprint 61)
- Factor 255 = 3 × 5 × 17 (8-bit, ~24 qubits)
- First pure-Rust Shor's with native modular arithmetic
- **Story**: "Open-source quantum factoring without dependencies"

### Tier 2: Serious Benchmark (Sprint 62-63)
- Factor 65,521 (largest 16-bit prime × small prime)
- ~35 qubits with recycling
- **Story**: "Classical simulation factors 16-bit numbers"

### Tier 3: Research Frontier (Sprint 64-65)
- Factor ~1,000,000 (20-bit)
- Push MPS/tensor network limits
- **Story**: "Tensor network simulation of Shor's algorithm"

### Tier 4: Cluster Achievement (Sprint 66-67)
- Factor ~1 billion (30-bit)
- Multi-node distributed simulation
- **Story**: "Largest classical Shor's simulation to date"

---

## Comparison to Real Quantum Computers

| System | Qubits | Largest N Factored |
|--------|--------|-------------------|
| IBM (2001) | 7 | 15 |
| USTC (2021) | 10 | 21 |
| This project (goal) | 35-50 (simulated) | ~2^25 |
| Google Willow | 105 | Not optimized for Shor's |
| Future (2030?) | 10,000+ | RSA-2048 |

Our simulation could actually factor LARGER numbers than current quantum hardware because:
1. No decoherence
2. No gate errors
3. Perfect connectivity

**The irony**: Classical simulation of quantum computers can outperform actual quantum computers for specific algorithms, until we hit the exponential wall.

---

## Conclusion

RSA-2048 is impossible to break with classical simulation. Period.

But we can:
1. Build the most capable Shor's implementation in open source
2. Factor numbers up to ~30 bits (larger than any real quantum computer today)
3. Serve as educational/research tool for post-quantum cryptography
4. Demonstrate exactly WHY quantum computers are needed

The journey is more valuable than the destination here.
