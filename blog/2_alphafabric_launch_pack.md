# Launch Asset Pack: AlphaFabric (learned placement on a logic fabric)

Use the blog post as the canonical link. Keep the framing narrow: this is a learned
*placement* policy with a *physical correctness oracle*, not a general "RL beats EDA"
claim. The honest scope (generalization/speed on held-out circuits, not one-off wins)
is the credibility, so lead with it — don't bury it.

## Primary Links

- Blog post: <https://github.com/wazi893/TileUniverse/blob/master/blog/2_alphafabric.md> (source: `blog/2_alphafabric.md`)
- Source: <https://github.com/wazi893/TileUniverse>
- Repro commands (in repo):
  - `cargo run --release --example alphafabric_af2`        — SA baseline
  - `cargo run --release --example alphafabric_af4`        — learned placer (generalization)
  - `cargo run --release --example alphafabric_af5`        — route-aware learned policy
  - `cargo run --release --example alphafabric_af_timing`  — timing-driven SA + the oracle catch

> NOTE: public-mirror gate CLEARED 2026-07-09 — `src/synth/alphafabric/` and
> `blog/2_alphafabric.md` verified present on the public repo (blob hash-identical
> to private master); URLs above are live.

## Hacker News

Suggested title:

```text
Show HN: I taught a policy to place circuits on a logic fabric, with a correctness oracle
```

Alternate titles:

```text
Show HN: AlphaChip-style learned placement on a from-scratch tile logic fabric
```

```text
Show HN: Learned circuit placement where "legal" means the circuit still computes correctly
```

First comment draft:

```text
I built a tile-based logic fabric (gates are tiles; a CPU fetches/decodes/executes out of them), and then a learned placement policy on top of it in the spirit of AlphaChip.

The result I actually trust is generalization, not a one-off win: trained on a few small circuits, a linear policy places unseen circuits one-shot (no per-circuit search) at ~60% of the naive baseline's wirelength. That's the thing simulated annealing can't do — SA restarts from scratch every time. For a single circuit, SA is still the quality ceiling, and I say so everywhere.

The part I'm most proud of is the legality gate. Instead of "did it route?", a layout is only legal if the placed-and-routed circuit still computes the exact same function, checked by replaying it on the real simulated tiles against an ISA reference. That made the reward un-hackable — and it earned its keep: an over-aggressive timing weight once produced a layout that routed fine but computed the wrong answer. The oracle caught it, and it led me to a real router defect (same-net stacked cells treated as connected without a via) that I fixed — 244 failing cases to 0.

What I'm NOT claiming: not beating SA on one-off layouts; it's a linear policy, not a GNN or deep RL; and it's a faithful logic/routing simulation, not silicon.

Repro commands are in the post. I'd especially welcome criticism on whether the framing is honest enough and on the feature/policy design.
```

## r/rust

Suggested title:

```text
AlphaFabric: a learned circuit-placement policy over a Rust tile logic fabric, gated by a physical correctness oracle
```

Body draft:

```text
I wrote up AlphaFabric — a learned placement system built on a from-scratch logic-simulation fabric (all Rust). Gates are tiles on a 128x640x16 grid; there's an A* router with negotiated congestion and rip-up/reroute, static timing analysis, and — the key piece — a correctness oracle: a layout is only legal if the placed+routed circuit still computes the same function, verified by replaying it on the simulated tiles.

Arc: environment -> simulated-annealing baseline -> learned constructive policy -> route-aware -> timing-driven, every layout verified correct on real tiles.

Honest headline: the policy generalizes. Trained on widths 3/5/6, it places unseen widths 4/8 + a multiplier one-shot (zero search) at ~60% of naive-baseline wirelength. SA remains the single-circuit quality ceiling; the learned win is transfer/speed on held-out circuits.

A methodological finding I think is reusable: route-aware training only learns to spread if the training set actually contains congesting circuits — easy data never fires the penalty, so the policy stalls. A penalty term is worthless if your training distribution never triggers it.

Source + repro commands in the post. Feedback welcome on the policy/feature design and on the honesty of the framing.
```

## One-Line Copy

```text
AlphaFabric is a learned circuit-placement policy over a Rust tile logic fabric; it generalizes to unseen circuits one-shot at ~60% of baseline wirelength, with a physical correctness oracle as the hard legality gate.
```

## Resume Bullets (for shipped artifact)

```text
- Built a learned circuit-placement system (AlphaChip-style) over a from-scratch tile
  logic fabric in Rust: a policy trained on small circuits places unseen circuits
  one-shot (zero search) at ~60% of baseline wirelength, with every layout verified
  functionally correct by a physical replay oracle.
- Designed the legality gate as a physical correctness oracle (placed-and-routed circuit
  replayed on simulated tiles vs an ISA reference), making the optimization reward
  un-hackable; it caught a routable-but-wrong placement and led to root-causing and
  fixing a router via-defect (244 failing cases -> 0).
- Established a classical simulated-annealing baseline first (-21% to -45% routed
  wirelength across blocks; rescued an otherwise-unroutable 8-bit adder) and scoped the
  learned claim honestly to generalization/speed on held-out circuits, not one-off wins.
```

## Do Say

- The learned win is generalization/speed on held-out circuits, with zero per-circuit search.
- Simulated annealing is the single-circuit quality ceiling, and that's stated up front.
- Legality = the placed+routed circuit still computes the same function, proven on real tiles.
- It's a linear policy over connectivity features; a GNN is a documented future upgrade.
- The oracle caught a routable-but-wrong layout and surfaced a real router bug, since fixed.
- Numbers are wirelength/criticality ratios from in-repo, reproducible examples.

## Do Not Say

- "RL beats classical EDA" (it doesn't, for single circuits — and I say so)
- "GNN" / "deep RL" (it's a linear policy)
- "chip" / "silicon" / "tapeout" without the "tile-simulation fabric, not a PDK" qualifier
- "optimal" placement (it's better-than-baseline, oracle-legal, not provably optimal)
- bare throughput superlatives (EXAOPS/etc.) — this artifact's anchors are wirelength
  ratios + the correctness oracle, not op-rate

## Pre-Launch Checklist

- [x] Push `src/synth/alphafabric/` + `blog/2_alphafabric.md` to the public mirror
  (verified 2026-07-09: public blob hash-identical to private master).
- [x] Fill `<PUBLIC_REPO_URL>` in the post and this pack; verify every link resolves
  (post filled in the 2026-06-25 honesty fix; pack filled 2026-07-09).
- Re-run the four repro examples on a clean checkout and confirm the cited numbers
  (they were measured during development; verify before publishing as fact).
- Decide whether to publish the companion "one source, one SoC" co-design post first or
  link it as forthcoming.
```
