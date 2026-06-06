# Milestone 5 Agent Research Playbook

**Status:** Active execution guide  
**Audience:** Agents running independent experiments on M5 (E-prop, transfer, bindings, architecture)  
**Primary objective:** Turn M5 from "infrastructure complete" into "result validated"

---

## 1) Ground Truth (Do Not Debate)

1. R-STDP from random connectivity failed (near chance accuracy).
2. The real breakthrough was Fisher-initialized discriminative channels.
3. Frozen discriminative model is a real baseline (about 62.2% test accuracy on MNIST in prior runs).
4. E-prop infrastructure exists, compiles, and is unvalidated until executed.
5. Until E-prop beats frozen baseline, M5 is not a validated learning milestone.

This document defines what counts as validation and what does not.

---

## 2) Scope and Non-Goals

### In scope
1. Validate `examples/gpu_mnist_eprop.rs` on MNIST.
2. Quantify transfer behavior on Fashion-MNIST after MNIST signal is confirmed.
3. Smoke-test Python `BatchSNN` bindings with a minimal reproducible run.
4. Define clear promotion gates for architecture changes (recurrent or temporal coding).

### Out of scope (for now)
1. Claims of "learns from scratch from random init."
2. Major architecture rewrites before E-prop signal is established.
3. DVS/event-driven revival before M5 core validation.

---

## 3) Decision Order (Hard Priority)

1. **Gate A:** MNIST E-prop validation (`gpu_mnist_eprop`)  
2. **Gate B:** Fashion-MNIST transfer check (same pipeline, dataset swap)  
3. **Gate C:** Python binding end-to-end validation (`BatchSNN.from_discriminative`)  
4. **Gate D:** New architecture work (recurrent hidden pool, temporal/latency encoding)

No Gate D work should be treated as milestone progress if Gate A is still red.

---

## 4) Exact Success Criteria

## Gate A: MNIST E-prop
Run: `cargo run --release --features cuda --example gpu_mnist_eprop`

Required outcomes:
1. Example completes all configured epochs without runtime error.
2. `Best trained (e-prop)` is strictly greater than `Baseline (frozen, ALIF)`.
3. Epoch 10 test accuracy target: `>= 70.0%` minimum for "positive signal".
4. Promotion target: `>= 78.0%` at epoch 10 or better by final epoch.

Interpretation:
1. `>= 78.0%`: Gate A pass, proceed to B/C/D.
2. `70.0% to 77.9%`: partial pass, tune before architecture changes.
3. `< 70.0%` or no baseline improvement: Gate A fail, do tuning only.

## Gate B: Fashion-MNIST transfer
Run equivalent of Gate A with Fashion-MNIST IDX files.

Required outcomes:
1. Same code path executes without special-case hacks.
2. E-prop still improves over frozen baseline.
3. Report both absolute accuracy and delta over baseline.

Interpretation:
1. Positive delta confirms learning mechanism generalizes beyond MNIST.
2. Negative delta means MNIST improvements may be overfit to calibration.

## Gate C: Python bindings
Required outcomes:
1. `maturin develop --release` succeeds in `python/`.
2. `from tileuniverse import BatchSNN` works.
3. `BatchSNN.from_discriminative(...)` instance is created and one inference path runs.

Interpretation:
1. Pass gives a shareable validation artifact for external collaborators.
2. Fail blocks ecosystem claims.

---

## 5) Required Experimental Protocol

All agents must follow this protocol for comparable evidence.

1. Record commit SHA before each run.
2. Record GPU model and CUDA driver/runtime info.
3. Do not change more than one conceptual factor per run.
4. Save full console log per run.
5. Report:
   - baseline accuracy
   - epoch-10 test accuracy
   - best test accuracy
   - delta vs baseline (percentage points)
   - runtime per epoch
   - final weight stats (`min`, `max`, `mean`, `zero_frac`, `saturated_frac`)

If a run crashes or diverges, record it as a result (not discarded data).

---

## 6) Minimal Run Matrix (Exact Expectations)

### Stage 1: Confirm default behavior
1. Run `gpu_mnist_eprop` as committed, no code changes.
2. Require one complete log.

### Stage 2: Hyperparameter sweep only if Stage 1 misses promotion target
Sweep these parameters in `examples/gpu_mnist_eprop.rs`:
1. `LEARNING_RATE`: `[0.0002, 0.0005, 0.0010, 0.0020]`
2. `EPROP_GAMMA`: `[0.2, 0.3, 0.5]`
3. `ALIF_BETA`: `[0.05, 0.1, 0.2]`

Execution rule:
1. Change one parameter family at a time.
2. Keep others at default.
3. Stop sweep early if promotion target is reached.

### Stage 3: Transfer
1. Use best MNIST config.
2. Run Fashion-MNIST once for signal check.

---

## 7) Agent Work Split (Independent Research)

Use parallel, non-overlapping tracks:

1. **Agent A (Validation Owner):**
   - Gate A default run and summary.
   - Owns go/no-go call for M5 learner viability.

2. **Agent B (Tuning Owner):**
   - Hyperparameter sweeps only if Gate A is partial/fail.
   - Produces ranked config table by epoch-10 and best accuracy.

3. **Agent C (Transfer Owner):**
   - Fashion-MNIST replication on winning config.
   - Reports delta stability vs MNIST.

4. **Agent D (Bindings Owner):**
   - Python smoke test and tiny notebook/script artifact.
   - Confirms import, model construction, inference output.

Agents must not rewrite each other's scope during the same cycle.

---

## 8) Reporting Template (Required)

Copy this block in every report:

```text
Run ID:
Date:
Commit SHA:
Owner:
Scope: [Gate A | Gate B | Gate C | Tuning]
Dataset:
Hardware:

Config:
- K:
- H:
- TICKS:
- BATCH_SIZE:
- LEARNING_RATE:
- EPROP_ALPHA_PRE:
- EPROP_ALPHA_ELIG:
- EPROP_GAMMA:
- ALIF_ALPHA:
- ALIF_BETA:

Results:
- Baseline accuracy:
- Epoch 10 test accuracy:
- Best test accuracy:
- Delta vs baseline (pp):
- Runtime per epoch (avg):
- Final weight stats: min / max / mean / zero_frac / saturated_frac

Outcome:
- Gate status: [PASS | PARTIAL | FAIL]
- Confidence: [HIGH | MEDIUM | LOW]
- Next action:
```

---

## 9) Promotion Rules for New Architecture Work

Recurrent hidden connections or temporal/latency encoding can be promoted only when:
1. Gate A is PASS (`>= 78.0%` or clearly improved and stable with repeat evidence).
2. At least one transfer run (Gate B) shows non-negative delta over frozen baseline.

If these are not met, architecture work is exploratory only and must not be framed as milestone completion.

---

## 10) Command Quickstart

MNIST E-prop:
```bash
cargo run --release --features cuda --example gpu_mnist_eprop
```

Discriminative baseline/reference:
```bash
cargo run --release --features cuda --example gpu_mnist_discrim
```

Python bindings smoke (from `python/`):
```bash
maturin develop --release
python -c "from tileuniverse import BatchSNN; print(BatchSNN)"
```

---

## 11) Final Standard

M5 is considered validated only when we can state all of the following with logs:
1. E-prop improves over frozen discriminative baseline on MNIST.
2. That improvement is reproducible and not a single-run artifact.
3. Transfer signal is at least non-negative on Fashion-MNIST.
4. Python `BatchSNN` path runs end-to-end in a real environment.

Until then, the correct posture is "promising infrastructure, pending learning validation."
