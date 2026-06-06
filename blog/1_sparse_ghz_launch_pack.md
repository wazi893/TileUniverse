# Launch Asset Pack: tileuniverse-quantum GHZ Post

Use the blog post as the canonical link. Keep the framing narrow: this is a small Rust crate for structured sparse quantum states, not a general simulator that beats exponential scaling.

## Primary Links

- Blog post: https://github.com/wazi893/TileUniverse/blob/master/blog/1_sparse_ghz.md (source: `blog/1_sparse_ghz.md`)
- Crate: https://crates.io/crates/tileuniverse-quantum
- Docs: https://docs.rs/tileuniverse-quantum
- Source: https://github.com/wazi893/TileUniverse
- Repro command: `cargo run --release -p tileuniverse-quantum --example launch_claims`

## Hacker News

Suggested title:

```text
Show HN: A Rust crate for GHZ states with Graham's-number qubit labels
```

Alternate titles:

```text
Show HN: tileuniverse-quantum, sparse GHZ/W-state representations in Rust
```

```text
Show HN: Representing GHZ states at symbolic qubit counts in Rust
```

First comment draft:

```text
I published tileuniverse-quantum, a small Rust crate for representing a few highly structured quantum states without materializing a full 2^n state vector.

The catchy part is GHZ: since |GHZ_n> has only two nonzero amplitudes, the crate can represent the same endpoint state shape with ordinary usize counts, BigUint counts such as 10^100, or symbolic labels such as Graham's number. BigUint labels still occupy label storage; the full 2^n state vector is never materialized.

The non-catchy caveat is the important one: this is not a general quantum simulator and it does not make arbitrary circuit evolution cheap. It is a typed sparse representation for states whose structure is already known. Materialized W-states store one amplitude per excitation position, so they are O(n); symbolic W-states are algebraic API objects.

Repro hooks are in the repo:

  cargo test -p tileuniverse-quantum --doc
  cargo run --release -p tileuniverse-quantum --example launch_claims

I would especially welcome criticism on the API boundaries and on whether the public framing is honest enough.
```

## r/rust

Suggested title:

```text
tileuniverse-quantum: sparse GHZ/W-state representations with BigUint and symbolic qubit counts
```

Body draft:

```text
I published `tileuniverse-quantum` 0.1.1, a small Rust crate for sparse representations of a few structured quantum states:

- GHZ endpoint states with fixed amplitude storage: `MinimalGhzState`, `UnlimitedGhzState`, `SymbolicGhzState`
- W-states with materialized O(n) excitation-position storage or symbolic algebraic checks
- symbolic labels such as `Graham`, `Tree(3)`, Knuth up-arrow notation, and a formal infinity label
- doctested public examples

The honest framing: this is not a general-purpose quantum circuit simulator. GHZ is cheap here because the state has exactly two nonzero amplitudes. BigUint labels still have label-size storage. W-states are linear when materialized. Arbitrary gates generally break the structure and require a fallback representation.

Crate: https://crates.io/crates/tileuniverse-quantum
Docs: https://docs.rs/tileuniverse-quantum
Source: https://github.com/wazi893/TileUniverse

The launch post includes reproduction commands. The main in-repo check is:

    cargo run --release -p tileuniverse-quantum --example launch_claims

That example asserts the structural claims and prints hardware-local timings.

I would appreciate feedback on the API design, docs, and whether there are other structured state families worth adding.
```

## One-Line Copy

```text
tileuniverse-quantum is a Rust crate for sparse GHZ/W/Dicke-state representations with ordinary, BigUint, and symbolic qubit-count labels.
```

## Do Say

- GHZ endpoint amplitude storage is O(1) here because the representation stores two amplitudes.
- W-states are O(n) when materialized because they store one amplitude per excitation position.
- symbolic labels are API objects, not decimal expansions.
- symbolic W-state checks are algebraic formula checks, not numerical state-vector verification.
- the crate is useful for teaching, demos, and structured-state experiments.
- benchmark timings are hardware-local; correctness and memory invariants are asserted in repo examples/doctests.

## Do Not Say

- "general quantum simulator"
- "simulates any quantum computer at Graham's-number scale"
- "breaks the exponential barrier" without the GHZ-specific qualifier
- "infinite-qubit state" without "formal" or "symbolic"
- "proof" for runtime timing claims; use "measurement" or "example run"

## Pre-Launch Checklist

- Run `cargo test -p tileuniverse-quantum --doc`
- Run `cargo run --release -p tileuniverse-quantum --example launch_claims`
- Run `cargo run --release -p tileuniverse-quantum --example graham_ghz`
- Update the blog URL in the HN and Reddit drafts after publishing
- Recheck crate/docs/source links before posting; only use a crate-subdirectory GitHub link after that path exists on the public default branch
