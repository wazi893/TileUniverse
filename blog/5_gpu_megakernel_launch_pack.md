# Launch Asset Pack: The Tile-CPU Megakernel (32k bit-exact CPUs on one GPU + a µop cache)

The artifact: a CUDA megakernel that runs **K independent, bit-exact instances of the
TileUniverse V2 tile CPU** — one GPU thread per instance, zero cross-lane sync — and the
optimization arc that took it from 0.2 M to **4.16 BILLION lane-instructions/sec** on one
RTX 5090, ending with the fake CPU growing a very real CPU feature: a **decoded-µop cache**.

Keep the framing narrow and honest: the payoff is **batch throughput** (parameter sweeps,
RL rollouts, fuzzing — K copies at once), NOT single-lane speed. One lane on the GPU is
slower than the CPU. Every number below is guarded by a bit-exactness assert against a
CPU oracle that is itself validated per-retired-instruction against the physical tile CPU.

Confirmed numbers (RTX 5090, sum_loop lockstep batch, all at the SAME batch size
K=65,536 so the generations are comparable; measured 2026-06-16 → 2026-07-02):

| Kernel generation | Mechanism | lane-instr/s @ K=65k |
|---|---|---|
| 1d megakernel (June) | per-lane full cycle | 51 M |
| + dead-op elimination | backward liveness from the ALU root: 7,282 → 2,054 ops | 180 M |
| + slot pruning | vals 5,694 → 2,071 slots; max K → 1.68 M lanes | 185 M |
| + lazy loads / packed metadata | 40 → 16.6 val B/op, 12.4 → 8.1 meta B/op | 382 M |
| + decode-once-per-warp (A) | leader decodes, warp broadcasts 22 slots | 279 M — **a loss at saturation** |
| **+ decode memoization (µop cache)** | decode is pure in pc → precompute per pc | **3.35 G (9.3×); peak 4.16 G @ K=32k** |

End-to-end at matched batch size: **51 M → 3.35 G = 66×**, of which the µop cache alone
is 9–18× depending on K.

## The story beats (use these in every version of the post)

1. **The machine**: a 32-bit/64-bit-data CPU whose fetch, decode, ALU, and writeback are
   *simulated logic tiles* (a 128×640×16 grid). A compiler (also in the repo) turns C-like
   source into its ISA. The GPU megakernel runs K copies of the tile datapath.
2. **The honest negative result**: the planned optimization (decode-once-per-warp: the
   warp leader evaluates the 1,848-op fetch/decode stage once, 31 lanes copy the 22-slot
   decode→trunk interface) was predicted ~4× from traffic math — it delivered +25–30%
   mid-band and was 20% SLOWER at saturation. Post-mortem: after the lazy-load work the
   kernel is **request-slot-bound, not byte-bound**; leader-only decode shrinks each
   memory request to one 32 B sector, so the ceiling drops. Real machines are measured,
   not estimated.
3. **The rescue**: the static analysis written to make design A safe *proves the decode is
   a pure function of pc* (no state-carrying ops, no lane-dependent reads). A program has
   at most 128 pcs. So: evaluate the decode ONCE per pc at build time, store the 22-slot
   interface in a `prog_len × 22 × 8 B` table, and at runtime no lane ever evaluates
   fetch/decode again — an ALU instruction is 22 table copies + the 206-op ALU trunk.
   **That is a decoded-µop cache** — the same trick x86 cores use to skip their decoders —
   except here it's provably exact, because the "silicon" is deterministic tiles.
4. **The metric discipline** (say this explicitly, it's the brand): with memoized decode,
   the eliminated 1,848 ops/instruction are reported as *work eliminated, bit-exact* —
   never multiplied back in as "op throughput". The headline is lane-instructions/sec,
   measured kernel-only, with the outputs asserted equal to the per-lane kernel and the
   CPU oracle at every K.

## Coverage (what "a CPU" means here — be precise if asked)

- 6 of the 8 compiled-benchmark corpus programs run on the GPU device end-to-end:
  fib_recursive, sum_loop, gcd_subtract, fact_iterative, sum_squares, array_sum_squares —
  including MUL (software authority, mirrored), conditional moves, shift carries, an MMIO
  console (compiled `putchar` programs print from all 32k lanes), and upper-bank fetch
  (programs up to 128 words; the two-sided super-mux bridge).
- Out of scope, honestly: programs past 128 words (`print_count`/`print_large`) — the
  *physical* CPU has no ROM there either (it falls back to gated software injection), so
  the device stops where the physical machine stops.
- Divergence: the µop-cache kernel is fully per-lane (each lane indexes the table with its
  own pc), so data-divergent batches keep the fast path. Verified with a
  divergence+reconvergence torture program (equal-length branch paths remerging onto a
  common ALU pc), every lane bit-exact vs its own oracle.

## Primary links

- Source (public mirror): https://github.com/wazi893/TileUniverse
- Demo gallery: https://wazi893.github.io/TileUniverse/demos/index.html
- Repro (in repo, needs CUDA):
  `cargo test --release --features cuda,slow-tests --lib device_run_warpdecode_bench -- --nocapture`

> ⚠️ LAUNCH GATE: this work currently lives on the PRIVATE dev repo
> (`TileUniverse-dev`, branch `feat/decode-once-warp`). The public mirror does NOT
> auto-update. Before posting anywhere: merge to master, push a public snapshot, and
> verify the repro command works from a fresh clone of the PUBLIC repo.

## Hacker News

Suggested title:

```text
Show HN: 32k bit-exact CPUs on one GPU – then I gave them a µop cache (17× speedup)
```

Alternate titles:

```text
Show HN: A CPU made of simulated logic tiles, 32,768 copies on one RTX 5090
```

```text
The optimization I planned lost 20%. The proof it needed won 17×.
```

## r/rust post draft

```text
Title: A CPU built from simulated logic tiles, running 32k bit-exact copies on one GPU (Rust + CUDA)

Body:
TileUniverse's V2 tile CPU is a full CPU — fetch, decode, ALU, writeback — built out of
simulated logic tiles, with a small C-like compiler targeting its ISA. I've been porting
the whole datapath to CUDA as a "megakernel": one GPU thread = one complete CPU instance,
K instances in parallel, zero cross-lane sync. Everything is oracle-gated: the GPU output
is asserted bit-exact against a CPU reference that's validated per-retired-instruction
against the tile machine itself.

The fun part is the optimization arc. The planned lever (decode-once-per-warp: the warp
leader evaluates the shared fetch/decode ops once, the other 31 lanes copy a 22-slot
interface) was predicted ~4× from traffic arithmetic — and measured SLOWER at saturation
(request-slot-bound, not byte-bound; leader loads shrink to lone 32B sectors). But the
static analysis written to make it safe proves decode is a pure function of pc — and a
program only has 128 pcs. So the decode is memoized per pc into a 22-slot table at build
time, and no lane ever runs fetch/decode again. The fake CPU grew a decoded-µop cache,
like real x86 cores — except provably exact, because the silicon is deterministic tiles.

Result: 382 M → 4.16 B lane-instructions/sec at K=32,768 on an RTX 5090 (10–18× across
batch sizes), bit-exact asserted at every K. Compiled programs (fib, gcd, factorial,
array kernels) run end-to-end; putchar programs print from all 32k lanes at once.

Honest scope: this accelerates BATCHES of independent CPU instances (sweeps / RL
rollouts / fuzzing) — a single lane is slower than the CPU sim. And the memoized decode
work is counted as eliminated, not as throughput.

Code: [public repo link] — the megakernel is src/tile_cpu/v2_device_gpu.rs.
```

## LinkedIn blurb

```text
I put 32,768 bit-exact copies of a CPU on one GPU — a CPU whose gates are simulated
logic tiles. Then a correctness proof I wrote for a failed optimization turned out to
license a decoded-µop cache (the same trick real x86 silicon uses), and throughput went
from 382 million to 4.16 billion instructions/sec. 10–18× — with every configuration
asserted bit-exact against the physical tile machine.

The lesson I keep re-learning: measure, don't estimate. The optimization I designed from
traffic arithmetic LOST 20% at saturation. The one the proof gave me for free won 17×.
```

## Anticipated objections (answer in comments, don't pre-empt in the post)

- *"Isn't the µop cache cheating — you're not simulating the tiles anymore?"* The decode
  values ARE tile-computed (evaluated once per pc from the same op arrays, on the same
  semantics), and the ALU trunk — the part that depends on runtime data — still evaluates
  every instruction. It's memoization of a proven-pure function, identical values,
  asserted bit-exact. Real CPUs make exactly this tradeoff in silicon.
- *"4 GHz-equivalent? So it's as fast as one real CPU?"* No single-lane claim: it's 4.16 G
  instructions/sec AGGREGATE across 32k instances (~127 K instr/s per instance). The
  product is batch capacity, not latency.
- *"What's it for?"* Architecture sweeps and RL-style rollout batches over a deterministic,
  fully-observable CPU (the repo uses the tile machine as an EDA/learning testbed), plus
  it's the most honest way I know to benchmark "how much verified computation fits on a
  GPU".

## Resume bullets (pick one)

```text
Built a CUDA "megakernel" running 32,768 bit-exact instances of a simulated-logic CPU on
one RTX 5090; designed a provably-exact decoded-µop cache (static purity analysis +
per-pc memoization) that took aggregate throughput from 382M to 4.16B instructions/sec
(10–18×), with every configuration asserted against a physically-validated oracle.
```

```text
Took a GPU logic-simulation kernel through five optimization generations (dead-op
elimination, slot compaction, lazy operand loads, packed metadata, decode memoization) —
66× end-to-end at matched batch size — using measured A/B benchmarks that overturned the
traffic-model prediction and localized the real bottleneck (request-slot-bound, not
bandwidth-bound).
```

## Launch checklist

- [ ] Merge `feat/decode-once-warp` to master (dev repo), full test pass.
- [ ] Push public mirror snapshot; verify repro command from a fresh public clone.
- [ ] Update the demo-gallery landing page if it mentions megakernel numbers.
- [ ] Per the channel plan: r/rust first (this is an engineering-with-a-hook piece);
      HN only after the cellular + AlphaFabric sequence per `project-launch-channels`.
