# AlphaFabric: I taught a policy to lay out circuits on a logic fabric — and a hardware oracle kept it honest

## TL;DR

I built a from-scratch logic-simulation fabric — a grid where logic gates are tiles and a CPU actually fetches, decodes, and executes out of them. Then I built a **learned placement system** on top of it, in the spirit of Google's AlphaChip: a policy that decides *where each gate goes*.

The headline result is the boring-sounding one that actually matters: the policy **generalizes**. Trained on a few small circuits, it places *unseen* circuits one-shot — no search — at **60% of the naive baseline's wirelength**. And the whole thing is fenced in by a **correctness oracle**: a layout is only "legal" if the placed-and-routed circuit still computes the exact same function, proven by replaying it on the real simulated tiles. That single constraint is what makes the reward un-hackable and the claim provable instead of hand-wavy.

This post is as much about what I'm *not* claiming as what I am. The honest scope is the whole point.

---

## The substrate already was the environment

Most "RL for X" projects die in the same place: you spend all your runway building the environment, the reward, and the simulator, and run out of road before the agent learns anything.

I got to skip that. My tile engine was already a fast, faithful, instrumented physical-design environment with a correctness oracle baked in:

- **A placement canvas** — a 128×640×16 tile grid with islands and banks.
- **A real router** — A* with negotiated congestion and bounded rip-up/reroute.
- **Quality signals** — total wirelength, via count, routing congestion/overflow, critical-path depth (static timing analysis), active-tile counts.
- **A correctness oracle** — golden output hashes plus a differential between an instruction-set reference and the physical tile execution.

That last one is the quiet killer feature. In most RL-for-placement work, "is this layout legal?" is a soft, fuzzy thing. Here it's a hard question with a true answer: *does the placed circuit still compute the right function?* If the layout changed the logic, the function changed, and the oracle says no. Reward −∞. You literally cannot reward-hack your way to a wrong circuit.

So the legal search space has a beautiful definition: **every layout whose function still matches the golden truth table.** The optimizer is free to do anything inside that set.

---

## Baseline before learning (this is a principle, not a step)

Here's the thing nobody tells you about "RL for chip placement": **for a single circuit, reinforcement learning usually loses to plain simulated annealing.** Real placers use SA and its cousins for a reason. AlphaChip doesn't beat SA by finding a better one-off layout — it beats it by *generalizing across many circuits*, amortizing the search.

So I refused to write a single learned parameter until I had a strong classical baseline to be honest against.

**The annealer.** A simulated-annealing placer over the grid: propose a swap or relocate, check legality, accept or reject by reward delta on a cooling schedule. The one engineering trick that made it tractable: anneal against a cheap **half-perimeter wirelength (HPWL)** proxy in the inner loop and only run the real router on the baseline and the best-so-far. Routing every proposed move took ~278 seconds across four test circuits; the HPWL proxy dropped that to ~1.6 seconds — and HPWL *is* the canonical placement objective anyway.

Results, all verified correct on real tiles:

| Circuit | Routed wirelength vs. naive |
|---|---|
| 4-bit adder | **−45%** |
| 8-bit equality | −26% |
| 8-bit parity | −21% |
| **4-bit multiplier** | **−28%** |
| 8-bit adder | naive layout was **unroutable** → SA made it routable |

That last row is my favorite. The naive row-major placement of the 8-bit adder *couldn't even be routed*. The annealer didn't just shrink it — it found a legal layout where the dumb approach found none.

**This was already a shippable artifact before any learning existed.** That's the principle: every stage stands on its own.

---

## Now the learning — and the honest claim

A linear policy over connectivity features — distance to the centroid of already-placed neighbors, how central a gate is, local density, boundary proximity — fit by deterministic black-box search to minimize wirelength. (A graph neural network is a documented *upgrade*, not something I'm claiming. Simple first.)

The claim I make is narrow and it's the only one RL legitimately earns here:

> Trained on circuit widths 3, 5, and 6, the policy places **unseen** widths 4 and 8 — and the multiplier — **one-shot, zero search**, at **60% of the naive baseline's wirelength.**

That's transfer. The model never saw those circuits and still placed them well immediately, with no per-circuit annealing budget. That is exactly the thing simulated annealing *cannot* do — SA starts from scratch every time.

Pure one-shot placement does over-cluster on the biggest blocks and can hurt routability, so the most useful mode is a **warm start**: the learned policy proposes an initial layout, then a *quarter-length* anneal cleans it up. That reaches 55% of the naive baseline — versus 51% for a *full-length* cold anneal. Most of the quality, a fraction of the search.

Then I made the training **route-aware**: fit against actual *routed* cost — which charges layouts that don't route — instead of the wirelength proxy. This started as a fix for held-out circuits that the wirelength-only policy placed un-routably. Then I found and fixed the underlying *router* defect (see the next section), and the gap mostly closed: both policies now place all five held-out circuits validly in one shot. So I'll make the narrow honest claim — route-aware training trades a little wirelength for more conservative spreading — rather than the inflated one I almost wrote down.

The more durable result is a methodological finding, and it cost me a wrong turn: **route-aware training only learns anything if the training set contains congesting circuits.** Train on easy data and the routability penalty never fires — there's no signal, and the policy just stalls. The lesson generalizes: a penalty term is worthless if your training distribution never triggers it.

---

## When the oracle earned its keep

I added a timing-driven objective too — weight each net's wirelength by how critical it is to the circuit's longest path, so the placer pulls timing-critical gates together. On most blocks this is a clean win (the 8-bit equality circuit improved **21%** on critical-path wirelength). On the 12-bit adder, turning timing-awareness *on* even fixed an otherwise-unroutable baseline.

But crank the timing weight too hard on the 4-bit multiplier — a congested block — and something interesting breaks. The layout *routes successfully*. The router is happy. Every net is connected.

**And the circuit computes the wrong answer.**

The oracle caught it. I'd been quietly assuming "routable ⟹ correct" — that if the router connected every net, the function was preserved. For dense, over-packed placements, that assumption is false: a too-tight solution can create an unintended physical adjacency that corrupts the logic while still settling to a stable, fully-routed state. I built a diagnostic (`Unroutable / NotConverged / WrongOutput / Correct`) to classify it, confirmed it was genuinely `WrongOutput`, and it pointed straight at a real router defect — same-net stacked cells being treated as connected without an actual via — which I then fixed (a regression that took 244 failing cases to 0). And here's the epilogue that makes the story complete: after the router fix, re-running the same over-aggressive-weight experiment shows the reclaimed multiplier layout now verifies **`Correct`** end-to-end. The "impossible" routable-but-wrong layout *was* that router defect, caught from the placement side by the physical gate. The `alphafabric_af_timing` example prints the live physical diagnosis on every run — so if any future change re-opens a gap between "routes" and "computes correctly," the demo will say so instead of assuming.

I want to be clear about what happened here, because it's the most important part of the whole project: **routability was not a sound proxy for correctness, and the only reason I know that is that the legality gate was physical, not heuristic.** A reward signal built on "did it route?" would have happily optimized toward a broken circuit. The oracle is what kept the optimizer — learned or classical — honest.

---

## What I'm *not* claiming

Because the credibility is in the scope:

- **I did not beat simulated annealing on one-off layouts.** SA is the quality ceiling for a single circuit and I say so everywhere. The learned win is *generalization and speed on unseen circuits*, full stop.
- **It's a linear policy, not a GNN, not deep RL.** A GNN is a future upgrade. I'm not dressing a small model up as a big one.
- **This is a tile-simulation fabric, not silicon.** The physics is a faithful logic-and-routing model, not a foundry PDK.

None of those caveats weaken the result. They're the reason you should believe it.

---

## Why I built this

I wanted one project that was unambiguously *machine learning for systems* — the AlphaChip / DSO.ai lane — rather than another image classifier. But I also wanted it to be the kind of thing I could be cross-examined on for twenty minutes and never have to bluff: concrete state, concrete action, concrete reward, a real baseline, an honest win, and a correctness oracle that makes the central claim *provable*.

The arc — environment → annealing baseline → learned policy → route-aware → timing-driven, every layout verified correct on real simulated tiles — is the whole pitch. The fabric was already the environment. The oracle was already the referee. All I added was a player that learned to generalize, and the discipline to only claim what the oracle could prove.

*Code: <https://github.com/wazi893/TileUniverse>. The fabric this runs on is a CPU built out of logic tiles with a C compiler that targets it — and that same compiler can now lower a marked function to a spatial tile datapath instead of instructions; the companion "one source, one SoC" post is forthcoming.*

---

### Appendix: the numbers, in one place
- Corpus: 76 parametric circuits (adders, equality, parity, AND-reduce), widths 2–20, split 56 train / 20 held-out.
- SA baseline: routed wirelength −21% to −45% across blocks; rescued an unroutable 8-bit adder.
- Learned one-shot (held-out, zero search): 60% of naive baseline wirelength.
- Warm start (learned + ¼ anneal): 55% of naive, vs 51% for full cold anneal.
- Both wirelength-only and route-aware policies place all 5 held-out circuits validly one-shot (after the router via-defect fix); route-aware trades a little wirelength for conservative spread.
- Timing-driven: −21% criticality-weighted wirelength on 8-bit equality; rescued an unroutable 12-bit adder; over-aggressive weighting on the multiplier once produced a routable-but-wrong layout — the correctness oracle caught it, and root-causing it surfaced a real router via-defect, since fixed (244 failing cases → 0). Post-fix the same experiment verifies `Correct`; the `alphafabric_af_timing` example prints the live physical diagnosis on every run.

### Reproduce
```
cargo run --release --example alphafabric_af2         # SA baseline
cargo run --release --example alphafabric_af4         # learned constructive placer (generalization)
cargo run --release --example alphafabric_af5         # route-aware learned policy
cargo run --release --example alphafabric_af_timing   # timing-driven annealing + the oracle catch
```
