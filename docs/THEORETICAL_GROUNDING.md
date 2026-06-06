# Theoretical Grounding: Sparse Quantum State Simulation

## 1. Characterizing Sparse-Preserving Circuits

### Definition: k-Sparse-Preserving Gates

A quantum gate G is **k-sparse-preserving** if:
```
For any state |ψ⟩ with s non-zero amplitudes:
  G|ψ⟩ has at most k×s non-zero amplitudes
```

### Gate Classification

| Gate Type | k-factor | Example |
|-----------|----------|---------|
| Diagonal gates | 1 | Z, S, T, Rz(θ), Phase |
| Permutation gates | 1 | X, Y, CNOT, SWAP, Toffoli |
| Hadamard | 2 | H |
| General 1-qubit | 2 | Ry(θ), U3 |
| General 2-qubit | 4 | Arbitrary unitary |

### Theorem: Sparsity Bound

For a circuit C with:
- Initial state with s₀ non-zero amplitudes
- h Hadamard (or equivalent) gates
- All other gates are permutation/diagonal

The final state has at most **2^h × s₀** non-zero amplitudes.

**Proof:** Each Hadamard can at most double the number of non-zero amplitudes.
Diagonal and permutation gates preserve the count exactly.

### Corollary: Polynomial Sparsity

If h = O(log n), then the state remains polynomially sparse: O(poly(n) × s₀).

**Implication:** Circuits with logarithmic Hadamard count are efficiently simulable
with our sparse representation.

---

## 2. Connection to Prior Art

### 2.1 Stabilizer Formalism (Gottesman-Knill Theorem)

**What it does:**
- Simulates Clifford circuits (H, S, CNOT) on |0⟩^n in O(n²) time and space
- Uses Pauli group representation instead of state vectors

**Comparison:**

| Aspect | Stabilizer | This Work |
|--------|------------|-----------|
| Gate set | Clifford only | Any sparse-preserving |
| Initial state | |0⟩^n only | Any sparse state |
| Memory | O(n²) | O(s) where s = #amplitudes |
| GHZ creation | O(n) | O(n) |
| W-state | Not native | O(n) |
| Non-Clifford | Requires tricks | Works if sparse |

**Our Position:** We extend beyond Clifford to any sparse state, but lose
the generality of Clifford circuits. Complementary, not competitive.

### 2.2 Tensor Networks (MPS/DMRG)

**What it does:**
- Exploits low entanglement structure
- Represents states as products of tensors with bounded bond dimension

**Comparison:**

| Aspect | Tensor Networks | This Work |
|--------|-----------------|-----------|
| Exploits | Low entanglement | Amplitude sparsity |
| Bond dim χ | O(2^(entanglement)) | N/A |
| Memory | O(n × χ²) | O(s) |
| GHZ (n qubits) | O(n) | O(1) memory |
| Random circuit | χ grows exponentially | s grows exponentially |

**Our Position:** Tensor networks exploit a different property (entanglement)
than we do (sparsity). GHZ is maximally entangled but has only 2 amplitudes.
For highly entangled but sparse states, we win. For low-entanglement dense
states, tensor networks win. Complementary approaches.

### 2.3 Decision Diagrams (QMDD/QBDD)

**What it does:**
- Represents state as directed acyclic graph
- Exploits repeated substructure in amplitudes

**Comparison:**

| Aspect | Decision Diagrams | This Work |
|--------|-------------------|-----------|
| Structure | DAG with sharing | HashMap of blocks |
| Memory | Varies with structure | O(s) |
| Lookup | O(n) path traversal | O(1) hash lookup |
| Implementation | Complex | Simple |

**Our Position:** Decision diagrams can be more efficient for states with
repeated subpatterns. Our approach is simpler and more predictable. For
GHZ/W states, performance is similar, but we scale to arbitrary precision
with BigInt addressing.

---

## 3. What We're NOT Claiming

### NOT: "Sparse states can be stored sparsely"
This is obvious. Everyone knows you can use a hashmap.

### NOT: "GHZ states are easy to simulate"
This is also known. The stabilizer formalism handles this.

### NOT: "We beat tensor networks in general"
For low-entanglement states, tensor networks are often better.

### NOT: "This enables quantum advantage"
Sparse states are classically simulable by definition.

---

## 4. What We ARE Claiming

### 1. Practical Implementation at Unprecedented Scale
We provide working code that actually runs 1M-qubit GHZ in 53 seconds.
This is not a theoretical bound - it's measured performance.

### 2. BigInt Addressing Removes Theoretical Limits
Using arbitrary-precision block IDs, there is no qubit count limit.
Previous implementations were bounded by 64/128-bit addressing.

### 3. Concrete Value for QEC/Benchmarking
Computing ideal logical states for error-corrected qubits has practical
value that other methods don't specifically target.

### 4. Simple, Reproducible Implementation
HashMap + block storage. No complex tensor contractions or tableau
manipulations. Easy to verify correctness.

---

## 5. Honest Framing of "1M Qubit" Claim

### The Misleading Version:
"We simulated one million qubits of quantum computation!"

### The Honest Version:
"We computed the ideal GHZ state for one million qubits - a state with
exactly 2 non-zero amplitudes in a state space of 2^1,000,000 dimensions.
This is useful for QEC verification but does NOT represent general
quantum simulation capability."

### What "1M Qubits" Actually Means:
1. We can ADDRESS any of 2^1,000,000 basis states
2. We can STORE the 2 that are non-zero
3. We can VERIFY correctness of the result
4. We CANNOT simulate arbitrary circuits on 1M qubits

---

## 6. The 100% Fidelity Question

### Criticism: "100% fidelity is meaningless"

**Response:**

We simulate IDEAL quantum mechanics with double-precision floating point.
Within numerical precision (~10^-15), our states are exactly correct.

**Why this matters:**
1. Hardware verification NEEDS the ideal reference
2. You can't measure "89% fidelity" without knowing what 100% looks like
3. Our role is computing |ψ_ideal⟩, not modeling noise

**What we don't do:**
- Depolarizing noise simulation
- Decoherence modeling
- Gate error simulation

**Future work could add:**
- Probabilistic Pauli errors (stays sparse for low error rates)
- Amplitude damping (increases sparsity!)
- Measurement errors (discrete, stays sparse)

---

## 7. Summary: Our Niche

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTUM STATE SPACE                          │
│                                                                 │
│  ┌──────────────────┐    ┌────────────────────┐                │
│  │   Low Entangle.  │    │  Clifford States   │                │
│  │   (Tensor Net)   │    │   (Stabilizer)     │                │
│  │                  │    │                    │                │
│  │    ┌────────────────────────┐             │                │
│  │    │    SPARSE STATES       │             │                │
│  │    │    (This Work)         │             │                │
│  │    │                        │             │                │
│  │    │   GHZ  W  Dicke        │             │                │
│  │    │   QEC Logical States   │             │                │
│  │    └────────────────────────┘             │                │
│  └──────────────────┘    └────────────────────┘                │
│                                                                 │
│            General Dense States                                 │
│            (Full statevector - exponential)                     │
└─────────────────────────────────────────────────────────────────┘
```

We occupy a specific niche: states that are sparse in the computational
basis but may have high entanglement or non-Clifford structure. This
includes important states for QEC and quantum networking, which is why
the work has practical value despite its specialization.
