# r/rust — AlphaFabric post (READY TO PASTE *after* the push lands)

> ✅ Public gate CLEARED 2026-07-09: blog/2_alphafabric.md + src/synth/alphafabric/
>    verified live on github.com/wazi893/TileUniverse (blob hash-identical to private
>    master). All links below resolve. NOTE: the 2026-07-09 blog number refresh
>    (−45%/−28% SA table, post-fix `Correct` epilogue) must be re-mirrored publicly
>    before posting.
>
> ✅ Numbers RE-VERIFIED 2026-07-09: all four examples re-run green on master.
>    60% one-shot exact; SA range now −21%..−45% (blog updated); post-router-fix the
>    af_timing recovery verifies `Correct` (demo + blog updated to match).
>
> ✅ Reddit account CONFIRMED able to post to r/rust (cellular launch landed and
>    stayed up 2026-07-09).
>
> ⚠️ Reddit can't be auto-filled by me — paste this in yourself at
>    https://www.reddit.com/r/rust/submit/?type=TEXT

## Title

```text
AlphaFabric: a learned circuit-placement policy over a Rust tile logic fabric, gated by a physical correctness oracle
```

## Body (Text post)

```text
I built AlphaFabric — a learned placement system on top of a from-scratch logic-simulation fabric, all in Rust. There's a live, no-install demo you can open and watch the placement relax in real time:

  Demo: https://wazi893.github.io/TileUniverse/demos/alphafabric.html

Gates are tiles on a slot grid; there's an A* router with negotiated congestion and rip-up/reroute, static timing analysis, and — the key piece — a correctness oracle: a layout is only legal if the placed+routed circuit still computes the same function, verified by replaying it on the simulated tiles against an ISA reference.

Arc: environment -> simulated-annealing baseline -> learned constructive policy -> route-aware -> timing-driven, every layout verified correct on real tiles.

Honest headline: the policy generalizes. Trained on a few small circuits, it places unseen circuits one-shot (zero search) at ~60% of naive-baseline wirelength. Simulated annealing remains the single-circuit quality ceiling; the learned win is transfer/speed on held-out circuits, not beating SA on a one-off layout.

A methodological finding I think is reusable: route-aware training only learns to spread if the training set actually contains congesting circuits — easy data never fires the penalty, so the policy stalls. A penalty term is worthless if your training distribution never triggers it.

What it's NOT: not "RL beats classical EDA" (it doesn't, for single circuits); it's a linear policy over connectivity features, not a GNN or deep RL; and it's a faithful logic/routing simulation, not silicon.

Source + repro commands:
  https://github.com/wazi893/TileUniverse

  cargo run --release --example alphafabric_af2        # SA baseline
  cargo run --release --example alphafabric_af4        # learned placer (generalization)
  cargo run --release --example alphafabric_af5        # route-aware learned policy
  cargo run --release --example alphafabric_af_timing  # timing-driven SA + the oracle catch

Feedback welcome on the policy/feature design and on the honesty of the framing.
```

## Push checklist (do these first, from your machine)

1. [x] Push `blog/2_alphafabric.md` to the public repo (master) — verified 2026-07-09.
2. [x] Push the `src/synth/alphafabric/` module to the public repo (master) — verified 2026-07-09.
3. [x] Verify both resolve:
   - https://github.com/wazi893/TileUniverse/blob/master/blog/2_alphafabric.md
   - https://github.com/wazi893/TileUniverse/tree/master/src/synth/alphafabric
4. [x] Re-run the four examples; confirm the cited numbers — done 2026-07-09 (blog table
   refreshed to −45%/−28%; af_timing prose fixed to the post-fix `Correct` diagnosis).
5. [x] Confirm your Reddit account can post to r/rust — cellular launch landed 2026-07-09.
6. [x] Re-mirror the refreshed `blog/2_alphafabric.md` (and the `alphafabric_af_timing.rs`
   prose fix) to the public repo — DONE 2026-07-09, public commit `a69b84e`.
   ALL GATES CLEAR — post whenever the account/timing is right.
```
