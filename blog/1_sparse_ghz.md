# A Rust Crate for GHZ States at Graham's-Number Scale

**TL;DR.** I published [`tileuniverse-quantum`](https://crates.io/crates/tileuniverse-quantum), a small Rust crate for representing a few highly structured quantum states without materializing their full state vectors. The headline case is GHZ: because

```text
|GHZ_n> = (|00...0> + |11...1>) / sqrt(2)
```

has exactly two nonzero amplitudes for any positive `n`, the crate can construct and verify GHZ endpoint representations with ordinary, huge, or symbolic qubit counts without materializing the full state vector. The endpoint amplitudes are fixed-size; a `BigUint` label still takes storage proportional to the integer label. Symbolic labels include names like Graham's number, TREE(3), and a formal "infinite" label.

This is not a general-purpose quantum circuit simulator. It is a typed sparse representation for states whose structure is already known.

---

## The trick

A dense simulator for `n` qubits stores `2^n` complex amplitudes. That is the right default for arbitrary quantum states, because a generic state can have nonzero amplitude almost anywhere in the basis.

GHZ is different. It only needs:

- the amplitude for `|00...0>`
- the amplitude for `|11...1>`

Every other basis state has amplitude zero.

In `tileuniverse-quantum`, the fast path stores two fixed-size blocks and treats the qubit count as metadata:

```rust
use tileuniverse_quantum::MinimalGhzState;

let ghz = MinimalGhzState::new(1_000_000_000);
let v = ghz.verify();

assert!((v.fidelity - 1.0).abs() < 1e-10);
assert_eq!(v.max_spurious, 0.0);
```

The important part is not that a billion is large. It is that the memory layout is the same for 5 qubits and 1 billion qubits. On my machine the struct reports 4,112 bytes in both cases.

## Past `usize`

On a 64-bit target, `usize` is plenty for ordinary "large" labels, but it is not enough if you want to play with named-number scale. The crate has two larger representations.

For materializable large integers, use `UnlimitedGhzState` with `BigUint`:

```rust
use num_bigint::BigUint;
use tileuniverse_quantum::UnlimitedGhzState;

let googol = BigUint::from(10u32).pow(100);
let ghz = UnlimitedGhzState::new(googol);
let v = ghz.verify();

assert!((v.fidelity - 1.0).abs() < 1e-10);
```

For labels that are not meant to be expanded, use `SymbolicGhzState`:

```rust
use tileuniverse_quantum::SymbolicGhzState;

let ghz = SymbolicGhzState::graham();
let v = ghz.verify();

assert!((v.fidelity - 1.0).abs() < 1e-10);
assert_eq!(v.size_class, "Graham-class");
```

Here `Graham` is an enum variant, not a decimal integer. The verification checks the stored amplitudes and the symbolic label. It does not try to compute Graham's number, because that would miss the whole point.

You can also construct labels such as:

```rust
SymbolicGhzState::tree(3);
SymbolicGhzState::knuth_arrow(3, 4, 3);
SymbolicGhzState::infinite();
```

The last one should be read as a formal symbolic label, not a claim that this crate settles the analytic subtleties of infinite tensor-product Hilbert spaces.

## W-states are different

The crate also includes W-states:

```text
|W_n> = (|10...0> + |01...0> + ... + |00...1>) / sqrt(n)
```

A W-state has `n` nonzero amplitudes, not 2. The materialized W path stores one amplitude per excitation position, not a `2^n` basis-index vector, so it is O(n), not O(1):

```rust
use tileuniverse_quantum::SparseQuantumGridVec;

let mut grid = SparseQuantumGridVec::new(1_000_000);
grid.create_w_state_parallel();

let v = grid.verify_w_fidelity_parallel();
assert!((v.fidelity - 1.0).abs() < 1e-10);
assert_eq!(v.correct_amplitudes, 1_000_000);
assert_eq!(v.max_spurious_amplitude, 0.0);
```

For symbolic W-states, the crate represents the amplitude as `1/sqrt(n)` and checks normalization algebraically:

```rust
use tileuniverse_quantum::create_graham_w;

let w = create_graham_w();
let v = w.verify();

assert!(v.is_valid);
assert!(v.total_probability.is_one());
```

Again, this is a structural representation. It is useful for APIs, examples, and algebraic checks. It is not a substitute for a full simulator once arbitrary gates enter the picture.

## Reproducing the claims

The launch claims are now backed by code in the repo.

Run the doctests:

```bash
cargo test -p tileuniverse-quantum --doc
```

Run the claim checker:

```bash
cargo run --release -p tileuniverse-quantum --example launch_claims
```

That example asserts:

- GHZ fidelity is 1.0 for small, billion-qubit, googol-qubit, and Graham-labeled states
- the two concrete GHZ fast-path states report identical memory use
- the symbolic Graham W-state normalizes algebraically
- the default materialized W-state check has the expected `n` amplitudes, tight amplitude error, no spurious tail amplitudes, and fidelity near 1.0

If you only want the GHZ and symbolic checks, skip the materialized W-state allocation:

```bash
cargo run --release -p tileuniverse-quantum --example launch_claims -- --skip-w
```

To scale the W-state materialized check, pass a size explicitly:

```bash
cargo run --release -p tileuniverse-quantum --example launch_claims -- --w 100000000
```

The 1B W-state path is intentionally opt-in because it needs about 16 GB just for the amplitude blocks:

```bash
cargo run --release -p tileuniverse-quantum --example launch_claims -- --w 1000000000
```

To print the narrative Graham GHZ demo:

```bash
cargo run --release -p tileuniverse-quantum --example graham_ghz
```

## One run on my machine

These are rounded measurements from release builds on an AMD Ryzen 9 9950X3D. They are not universal benchmark promises; they are the kind of output the command prints on your own hardware. "Reported memory" is what the crate's `memory_bytes()` methods report: endpoint blocks plus label bytes for BigUint GHZ, struct size for symbolic objects, and allocated amplitude-block payload for materialized W-states. For materialized W-states, `Create` is the W-state fill after grid allocation; the checker also prints allocation time separately.

| Check | Reported memory | Create | Verify | Notes |
|---|---:|---:|---:|---|
| GHZ, 5 qubits | 4,112 B | <1 us | <1 us | checked by `launch_claims` |
| GHZ, 1B qubits | 4,112 B | <1 us | <1 us | same fast-path memory |
| GHZ, 10^100 qubits | 4,138 B | <1 us | <1 us | BigUint label |
| GHZ, Graham label | 4,152 B | ~2 us | single-digit us | symbolic label |
| W-state, 1M qubits | 16,001,024 B | ~1 ms | <1 ms | default materialized check |
| W-state, 100M qubits | 1,600,000,000 B | ~60 ms | ~34 ms | opt-in run |
| W-state, Graham label | 160 B | constant-size formula object | constant-size formula check | algebraic symbolic check |

The GHZ rows are flat because the representation stores two amplitudes. The materialized W-state rows scale with `n`. The symbolic W-state row is small because it stores a formula, not `n` amplitudes.

## What this is not

The honest limits matter more than the catchy title:

- It does not simulate arbitrary circuits at Graham's-number scale.
- It does not make exponential state vectors disappear for generic states.
- It does not let you apply arbitrary gates while staying in the GHZ fast path.
- "Verification" means "the stored representation has the expected endpoint/excitation amplitudes and labels"; symbolic W verification is an algebraic consistency check of the stored formula.
- Symbolic states are symbolic API objects. They are useful, but they are not exact symbolic quantum algebra engines.

The mathematical observation is also not novel. GHZ states having two nonzero amplitudes is textbook-level. What I wanted to package is the engineering version: a small Rust crate with doctested APIs for `usize`, `BigUint`, and symbolic qubit-count representations, plus explicit examples that make the boundaries visible.

## What it is useful for

I expect this to be useful as:

- a teaching example for sparse quantum representations
- a reference implementation for structured GHZ, W, and Dicke-state APIs
- a small crate to experiment with symbolic qubit labels in Rust
- a sanity check for discussions about what sparse simulation can and cannot buy you

## Get it

```toml
[dependencies]
tileuniverse-quantum = "0.1"
```

- **Crate:** https://crates.io/crates/tileuniverse-quantum
- **Docs:** https://docs.rs/tileuniverse-quantum
- **Source:** https://github.com/wazi893/TileUniverse

If you find a bug, a better API shape, or a structured-state family that belongs next to GHZ/W/Dicke, I would like to hear about it.
