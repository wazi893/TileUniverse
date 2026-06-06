# logic-fabric-core

A high-performance **quantum circuit simulation and gate-fusion optimization**
library in Rust. Its differentiator is an **algebraic gate-fusion engine** that
reduces circuit gate counts by ~36% on standard benchmarks — and, unusually,
ships with a verification harness that *proves* every reduction preserves the
circuit's unitary (not just that the gate count went down).

> **Status:** the single-qubit / algebraic / commutation fusion pipeline and the
> dense & sparse state simulators are production-grade and test-covered. The GPU
> (CUDA) backend is feature-gated and optional. Two-qubit KAK resynthesis is
> present but **classification-only** (synthesis is WIP — see [Scope](#scope)).

## Why it exists

Most circuit optimizers report a smaller gate count and ask you to trust it.
This one treats correctness as the headline: the bundled benchmark optimizes a
circuit and then **simulates the original and optimized circuits on several
random input states**, requiring `|⟨ψ_orig | ψ_opt⟩|² ≈ 1` on every one. A
single random input already exposes any relative-phase error with probability 1.

Optimization is also **deterministic**: the same circuit always produces the
same output, byte-for-byte, across runs and processes.

## Verified benchmark (QASMBench `small`, IBM `rz/sx/cx` basis)

Optimized with `ultimate_optimize`; each result checked for full unitary
equivalence (up to global phase) against the input.

| circuit | original | optimized | reduction |
|---|---:|---:|---:|
| bb84_n8 | 23 | 5 | 78% |
| quantumwalks_n2 | 38 | 11 | 71% |
| dnn_n8 | 1416 | 504 | 64% |
| error_correctiond3_n5 | 237 | 88 | 63% |
| vqe_n4 | 73 | 27 | 63% |
| qaoa_n6 | 378 | 142 | 62% |
| hhl_n7 | 990 | 409 | 59% |
| basis_trotter_n4 | 2353 | 1210 | 49% |

**Aggregate: 19,455 → 12,335 gates (36.6% reduction), 39/39 circuits verified
equivalent.** The result is **deterministic** — byte-identical across runs and
processes. (2 circuits use custom `gate` macros the parser does not yet expand;
see [Scope](#scope).)

Reproduce:

```bash
cargo run --release --example bench_fusion -- path/to/QASMBench/small
```

## Quick start

```rust
use logic_fabric_core::qasm::{parse_qasm, to_qasm};
use logic_fabric_core::algebraic_fusion::{ultimate_optimize, AlgebraicOp};
use logic_fabric_core::quantum::QGate;

let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
"#;

let (gates, n_qubits) = parse_qasm(src).unwrap();
let (ops, stats) = ultimate_optimize(gates, n_qubits as u8);

// Flatten the optimized ops back into a gate list.
let mut out = Vec::new();
for op in ops {
    match op {
        AlgebraicOp::Single(g) => out.push(g),
        AlgebraicOp::Power { gate, power } => (0..power).for_each(|_| out.push(gate.clone())),
        AlgebraicOp::Skip { .. } => {}
        _ => {}
    }
}
println!("{}", to_qasm(&out, n_qubits));
println!("effective speedup: {:.2}x", stats.throughput_multiplier);
```

Simulate a circuit directly:

```rust
use logic_fabric_core::quantum::{QState, QGate, QRng, apply_gate_scalar};

let mut state = QState::new_zero(2);
let mut rng = QRng::new(1);
for g in [QGate::H(0), QGate::CNot(0, 1)] {
    apply_gate_scalar(&mut state, &g, &mut rng);
}
// state is now the Bell pair (|00> + |11>)/sqrt(2)
```

## What's inside

| Module | Purpose |
|---|---|
| `quantum` | Dense statevector simulator, `QGate` set (H/X/Y/Z/Rx/Ry/Rz/Phase/U3/CNOT/CZ/SWAP/CCX/…), scalar backend |
| `algebraic_fusion` | Gate algebra: power reduction, rotation-chain merging, single-qubit mega-fusion (ZYZ), commutation reordering, `ultimate_optimize` pipeline |
| `fusion` | Higher-level fusion IR, cost model, CPU/GPU backend selection |
| `commutation` | Qiskit-style commutation rules and long-range cancellation |
| `sparse_state` / `block_sparse_state` | Sparse and GPU-tile-oriented state representations |
| `qasm` | OpenQASM 2.0 parse / serialize |
| `hardware` | Coupling maps + SWAP-insertion routing |
| `cuda` *(feature `cuda`)* | GPU backend: FP16 Tensor Core (WMMA), FP8, CUDA graphs |

## Optimization techniques

- **Single-qubit mega-fusion** — collapse a run of single-qubit gates into one
  `U3` via ZYZ Euler decomposition (global phase discarded).
- **Algebraic power reduction** — `G^n` reduced using each gate's order
  (self-inverse, periodic, rotation).
- **Rotation-chain merging** — `Rz(α)·Rz(β) → Rz(α+β)`, same for Rx/Ry.
- **Commutation reordering** — bring cancellable / fusible gates adjacent using
  commutation rules (e.g. `Rz` commutes through a CNOT control), then fuse.
- **Inverse cancellation** — adjacent and long-range mutual-inverse elimination.

## Features

| Feature | Effect |
|---|---|
| *(default)* | CPU scalar simulation + full fusion pipeline, no heavy deps |
| `cuda` | CUDA GPU backend (requires CUDA Toolkit; pulls in `cudarc`) |
| `quantum_jit` | JIT infrastructure hooks |

## Scope

To keep the benchmark and claims honest:

- **Two-qubit KAK resynthesis is classification-only.** CNOT-count
  classification (Makhlin invariants) is correct and tested, but local-unitary
  extraction is incomplete, so the pipeline does **not** currently reduce
  two-qubit CNOT counts. Single-qubit fusion is unaffected.
- **The QASM parser** handles the textbook gate set plus the IBM `rz/sx/cx`
  basis. It does **not** yet expand user-defined `gate` macros, so raw
  QASMBench files that define custom gates must be flattened first.
- **The CUDA backend** is optional and only built with `--features cuda`.

## Testing

```bash
cargo test --release            # unit + equivalence tests
cargo run --release --example bench_fusion -- <qasm dir>   # verified benchmark
```

## License

TBD.
