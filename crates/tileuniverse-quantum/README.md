# tileuniverse-quantum

Structured sparse quantum-state representations with GHZ and W-state helpers.

This crate is not a general-purpose quantum circuit simulator. It stores compact representations for state families whose nonzero amplitudes are already known.

## Key Insight

GHZ state `|GHZ_n> = (|00...0> + |11...1>)/sqrt(2)` has exactly **2 non-zero amplitudes** regardless of `n`. Store only those two amplitudes; the qubit count is metadata.

```rust
use tileuniverse_quantum::*;

// Fixed endpoint-block memory for this concrete usize qubit count.
let ghz = MinimalGhzState::new(1_000_000_000);
let v = ghz.verify();
assert!((v.fidelity - 1.0).abs() < 1e-10);
```

## State Types

| Struct | Count Representation | Use Case |
|--------|-------------|----------|
| `MinimalGhzState` | `usize` | Fixed-size endpoint GHZ representation |
| `UnlimitedGhzState` | Materialized `BigUint` | Endpoint GHZ representation with large integer labels |
| `SymbolicGhzState` | Symbolic labels | Graham's number, TREE(3), and formal infinity labels |
| `SparseQuantumGridVec` | `usize` | Materialized W-state excitation amplitudes with O(n) memory |

## W-State Example

```rust
use tileuniverse_quantum::SparseQuantumGridVec;

// W-states materialize one amplitude per excitation position: O(n), not O(2^n).
let grid = SparseQuantumGridVec::new(1_000);
assert_eq!(grid.n_qubits(), 1_000);
```

## Symbolic Example

```rust
use tileuniverse_quantum::*;

// Graham's number as a symbolic qubit-count label.
let ghz = SymbolicGhzState::graham();
let v = ghz.verify();
assert!((v.fidelity - 1.0).abs() < 1e-10);
assert_eq!(v.size_class, "Graham-class");
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tileuniverse-quantum = "0.1"
```

## Features

- **GHZ endpoint states**: fixed endpoint-block storage for `usize`, `BigUint`, and symbolic labels
- **W-states**: O(n) memory for materialized n-qubit W-states via sparse block grid (BLOCK_SIZE=128)
- **Symbolic labels**: qubit counts as symbolic expressions (Graham's number, TREE(3), Knuth up-arrow, tetration, BigUint, and formal infinity labels)
- **Dicke states**: Generalized states with k excitations among n qubits
- **Symbolic amplitudes/fractions**: Algebraic verification of probabilities, entropy, binomial coefficients
- **Parallel verification**: Rayon-accelerated doctest examples

## Doctests

All public API includes runnable doctests:

```bash
cargo test --doc -p tileuniverse-quantum
```

## License

MIT
