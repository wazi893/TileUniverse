# Launch Asset Pack: Cellular GPU (Game of Life on the packed tile fabric)

Use the **live interactive demo** as the canonical link — this is a visual launch, and the
demo is the artifact. Keep the framing narrow: this is a fast *packed-bit cellular kernel*
(64 cells per 64-bit word, Conway's Life rule) verified bit-identical to a CPU reference —
not a claim about general-purpose compute throughput.

Confirmed numbers (W, 2026-06-19): **6.7T** Game-of-Life cell-updates/sec and **115T**
packed tile-evals/sec on an RTX 5090. Canonical public repo: **github.com/wazi893/TileUniverse**.

> ⚠️ VERIFY BEFORE LAUNCH: the local landing page and demo gallery now use **6.7T**
> Game-of-Life cell-updates/sec and point GitHub traffic to `wazi893/TileUniverse`.
> Re-check the deployed pages before posting so HN/Reddit traffic sees the verified numbers.

## Primary Links

- **Live demo (canonical):** https://wazi893.github.io/TileUniverse/demos/cellular.html
- Demo gallery (all 5): https://wazi893.github.io/TileUniverse/demos/index.html
- Source: https://github.com/wazi893/TileUniverse
- Repro command (in repo):
  `cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3`

## Hacker News

Suggested title:

```text
Show HN: Conway's Game of Life at 6.7 trillion cell-updates/sec on one GPU
```

Alternate titles:

```text
Show HN: A packed-bit GPU cellular fabric running Game of Life, bit-exact vs CPU
```

```text
Show HN: Game of Life at trillions of updates/sec — interactive demo, no install
```

First comment draft:

```text
This is an interactive, single-page demo of Conway's Game of Life running on a GPU
cellular fabric I built from scratch in Rust. Open it and press space — every frame is
real engine output, no install, no dependencies.

The speed comes from bit-packing: 64 cells live in a single 64-bit word, so one machine
instruction advances 64 cells at once. Horizontal neighbors come from warp shuffles
(zero-cost), vertical neighbors from L2. On an RTX 5090 the packed kernel does 6.7 trillion
Game-of-Life cell-updates/sec, and ~115 trillion raw packed tile-evaluations/sec for the
simpler logic rule. That's about a 375x speedup over the naive one-cell-per-u64 layout.

The part I actually care about: the GPU output is verified bit-identical to a plain CPU
reference. It's not a "trust the pretty pixels" demo — same rule, same result, checked.

Honest scope: this is a fast kernel for packed local-neighborhood rules (Life, wire
propagation, logic gates), not a general-purpose compute-throughput claim. The number is
kernel-only on a single consumer GPU.

Demo: https://wazi893.github.io/TileUniverse/demos/cellular.html
All 5 demos: https://wazi893.github.io/TileUniverse/demos/index.html
Source + repro: https://github.com/wazi893/TileUniverse

Repro:
  cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3

Feedback welcome on the packing/kernel design and on whether the framing is honest enough.
```

## r/rust

Suggested title:

```text
A packed-bit GPU cellular fabric in Rust: Conway's Life at 6.7T cell-updates/sec, bit-exact vs CPU
```

Body draft:

```text
I built a GPU cellular-automaton fabric in Rust and put up a single-page interactive demo
of Conway's Game of Life on it: https://wazi893.github.io/TileUniverse/demos/cellular.html
(press space, no install).

How it's fast: cells are bit-packed 64-to-a-word, so one instruction evolves 64 cells.
Horizontal neighbors are warp shuffles, vertical neighbors come from L2, and the whole
inner loop is register-resident. On an RTX 5090 that's 6.7 trillion Life cell-updates/sec
and ~115T raw packed tile-evals/sec for the simpler logic rule — roughly 375x over a naive
u64-per-cell layout.

The bit I'm proudest of is correctness, not the headline number: GPU output is verified
bit-identical to a scalar CPU reference, so the demo is real engine output rather than a
shader that happens to look like Life.

Honest scope: it's a fast kernel for packed local-neighborhood rules, not a general compute
claim; the figure is kernel-only on one consumer GPU.

Source + repro commands: https://github.com/wazi893/TileUniverse
  cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3

I'd welcome criticism on the kernel/packing design and the benchmarking methodology.
```

## LinkedIn

Post draft (links go in the FIRST COMMENT, not the body — LinkedIn suppresses reach on
posts with outbound links in the body):

```text
I built a GPU cellular-automaton fabric in Rust, and the most fun way to show it is also
the simplest: Conway's Game of Life.

It runs at 6.7 trillion cell-updates per second on a single RTX 5090 — and the GPU output
is verified bit-identical to a plain CPU reference, so it's real engine output, not a
look-alike shader.

The trick is bit-packing: 64 cells share one 64-bit word, so a single instruction advances
64 cells at once. Neighbor reads come from warp shuffles and L2 cache. About 375x faster
than the obvious one-cell-per-word layout.

I put up a one-click interactive demo (no install — open it and press space), plus four
others: a CPU built entirely out of logic tiles, a surface-code decoder, a learned chip
placer, and a spiking neural network. Links in the comments.

Built in Rust + CUDA. Feedback and hard questions welcome.
```

First comment:

```text
Demo: https://wazi893.github.io/TileUniverse/demos/cellular.html
All 5: https://wazi893.github.io/TileUniverse/demos/index.html
Code: https://github.com/wazi893/TileUniverse
```

## One-Line Copy

```text
A GPU cellular fabric in Rust running Conway's Game of Life at 6.7T cell-updates/sec, with packed 64-cells-per-word evaluation verified bit-identical to a CPU reference.
```

## Do Say

- The speed comes from bit-packing (64 cells per 64-bit word) + warp shuffles for neighbors.
- GPU output is verified bit-identical to a CPU reference — correctness, not just visuals.
- ~375x over the naive one-cell-per-u64 layout.
- It's a fast kernel for packed local-neighborhood rules (Life, wire prop, logic gates).
- Numbers are kernel-only, single consumer GPU (RTX 5090), reproducible via bench_engine.

## Do Not Say

- bare "trillions of operations/sec" without the "cell-updates, packed Life rule" qualifier
- "fastest cellular automaton ever" / unqualified superlatives
- "general-purpose GPU compute" claims — this is a packed local-rule kernel
- treat the demo's frame rate as the benchmark (the kernel number is separate from rendering)

## Pre-Launch Checklist

- Verify the deployed landing page, demo gallery, and cellular demo show 6.7T
  Game-of-Life cell-updates/sec and point GitHub traffic to `wazi893/TileUniverse`.
- Open all three demo/gallery/repo links in a fresh browser and confirm they resolve.
- Post timing: HN on a weekday ~8–10am ET; sit on it 1–2 hrs for comments. r/rust a few
  hours later or next day. LinkedIn same day as HN, links in first comment.
```
