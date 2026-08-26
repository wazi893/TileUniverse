# Every Jump, Call, Return — and Now Every Multiply — in My CPU Is Computed by Cellular-Automaton Tiles

*Draft v2 — 2026-07-09 (v1 covered sprints 390-394; v2 adds the MUL island coda,
sprints 395-397). Companion: `blog/1_sparse_ghz.md` (the engine),
`blog/2_alphafabric.md` (learned placement on the same fabric).*

I build a 32-bit CPU out of cellular-automaton tiles — a 128×640×16 grid where every
tile computes one function of its neighbors per tick, and a Register64 tile captures
its input on a global clock edge like real sequential hardware. For most of this
project's life, the CPU had a dirty secret shared by every "CPU in a cellular
automaton" project I've seen: **when the machinery got hard, software quietly did the
work and wrote the answer into the grid.** The tiles were a visualization of a CPU,
not a CPU.

Five sprints ago the score was 99.8% "physical authority" — my metric for *the value
the CPU uses is the value the tiles computed* — with documented exceptions. Then I
audited the two newest subsystems and found the exceptions had quietly grown: in the
wide-PC mode that every compiled program larger than 256 words requires, **every jump,
call, and return target was written by software**. The tiles computed a target; the
executor overwrote it. And one instruction (MTLR, "move to link register") wasn't
implemented on the physical path at all — it silently executed as a register move,
a divergence no test caught because the compiler happened never to emit it.

This is the story of sprints 390–394: making every control transfer and every
link-register write physically authoritative, with a paper trail I can defend.

## What "physical" means here (the honest part, up front)

- The fabric is simulated (deterministic, reproducible — that's the point of the
  project). "Physical" means *computed by tile evaluation rules*, not by the Rust
  executor that orchestrates staging.
- I use two grades and I label them. **Physical**: the tiles compute the value from
  other tiles (the branch-taken LUT, the register read port, the LR capture mux).
  **Delivery-grade**: software copies *instruction-word data* — an immediate — into a
  physical route, and everything downstream (selection, masking, capture) is tiles.
  Copying an immediate is delivery; computing a result is not. The audit that draws
  this line is in the repo (`SPRINT_261_AUTHORITY_AUDIT.md`), and every flip in this
  arc kept a software fallback wired to a counter that must read **zero** in CI.
- 45 golden benchmark hashes are byte-identical across the entire arc. Every new
  structure is gated to wide-mode fabrics; the default fabric never changed.

## The arc, sprint by sprint

**S390 — the audit, a RET, and a bug that wasn't there.** Flipping RET to "trust the
physical target, verify, fall back on mismatch" cost zero new tiles — the LR capture
mux and target-selection mux had existed since sprint 125; software just never stopped
overwriting them. The same sprint fixed MTLR's silent divergence and, chasing a
suspected sequencing defect, proved the scary hypothesis *wrong* with an
instrumentation probe: the corruption was a HALT instruction self-targeting through an
8-bit immediate field. Root-caused, one-line fix, regression test. The suspected
"fragile wide-PC sequencing" was fine — and now there was evidence.

**S391 — immediates, and the probe that decided the design.** Wide immediate targets
(JMP.W and friends) ride a route that physically carries only 8 bits — but the route's
*wires* are 64-bit value carriers; only the source was narrow. One question decided
between a zero-tile fix and a week of routing: does an injected 16-bit target survive
to the clock edge, or does the instruction-delivery spine re-drive the route first?
The probe answered with a readout I still enjoy: the PC captured `300` while the route
visibly collapsed back to `44` *after* the edge — capture wins the race by
construction. Zero new tiles. All six wide branch kinds flipped to trust-verify,
taken and not-taken.

**S392 — the premise that died, and what it saved.** Register-indirect targets
(JMPR/CALLR) need `regs[rd]` delivered physically. The obvious tap — the operand
trunk, which a probe had shown carrying `regs[rd]` during execute — was the plan.
A pre-placement probe at the *capture moment* falsified it: the executor restores the
operand tree's selector constants mid-tick, and by the clock edge the trunk holds
garbage. The "positive recon" had been true at stage-X time and dead at capture time.
That probe cost an hour and killed a ~60-tile route that would have failed
mysteriously after a week. The sprint closed as a documented partial: the second
target mux and its select bit landed; the value path waited for a real design.

**S393 — the register read port.** The fix for S392's falsification is the sprint I'm
proudest of: a **16:1 read port built directly on the register file** — the one
structure in the machine whose values are stable through a whole tick, because
Register64 tiles only change on clock edges. Sixteen taps, per-column private routing
layers, a 15-mux selection ladder keyed by the physical rd field, ~1,090 hand-placed
tiles. Two process tools made it land on the first try: a **floorplan validator**
(every planned coordinate machine-checked against the live grid before placing
anything — it caught a mux orientation inversion, a layer collision, and 8 blocked
taps) and **per-phase capture-moment probes** (which caught the one bug class the
validator can't: a select rail reading a never-placed cell). With the port in place,
compiled recursion — `fact(5)` returning through `MFLR`/`JMPR` — runs with every
return target computed in tiles, zero fallbacks.

**S394 — the 10× cheaper twin.** MTLR needs `regs[rs]` — the *other* register field.
A second read port would be another thousand tiles. Instead: the S393 ladder is
select-agnostic, and the two consumers are disjoint by opcode (a jump is never MTLR).
Four rail muxes — `select = is_mtlr ? rs : rd` — plus a physical opcode-equality
decoder and a relocated LR capture cascade with a hardware mask, and every LR write in
the machine is physical. ~380 tiles. The assist counter that used to count MTLR's
software writes now counts fallbacks: zero.

## What I actually learned (the part worth stealing)

1. **Probe the premise at the moment that matters, before building.** Three times in
   five sprints, a cheap instrumentation probe at the clock-edge capture moment either
   falsified the design premise (S392), picked between designs (S391), or redirected a
   bug hunt (S390). Recon that reads state at the wrong point in the cycle is worse
   than no recon — it's confident and wrong.
2. **Trust-verify-fallback beats flag-day cutover.** Every authority flip kept the
   software path as a mismatch-triggered fallback with a counter asserted zero in
   tests. The counters are permanent regression tripwires: when S394 deliberately
   widened a predicate, a pinned counter in an S390 test fired exactly as designed.
3. **Machine-check hand layouts.** Hand P&R in a dense 3D grid is miserable exactly
   where it's silent (my L1 collisions don't error — they mis-evaluate). The
   validator-then-probe pipeline turned two ~1,000-tile builds into first-try lands.
4. **An honest DoD compounds.** S392 shipped as a "documented partial" instead of a
   rushed route. That partial — with its falsification evidence — is why S393 was
   designed right and landed clean.

## Coda: the last 0.2% (sprints 395-397)

One computation gap remained after the control-flow arc: **MUL** — the product was
still `wrapping_mul` injected into a physical writeback path. Three more sprints
closed it, and they're the cleanest demonstration of the method paying compound
interest.

The fix is a **multiplier built from adders and shifters**: a ~85-tile radix-2
shift-add island — three hold-muxed registers computing `P += (B&1 ? A : 0)`,
`A <<= 1`, `B >>= 1` — iterating **once per real clock edge**, because register
clocks in this fabric are global and there is no such thing as a private fast clock.
That constraint isn't a workaround; it's the architecture: in island mode, MUL is a
65-cycle instruction, exactly like early microprocessors, and the ISS parity contract
becomes end-state equality rather than cycle lockstep. The layout uses two annex
layers — computation and horizontal routing on one, verticals on the other, vias at
the turns — which makes wire crossings *impossible by construction* rather than
carefully avoided. The capstone test is compiled recursion: `fact(5)` — whose source
is literally `return n * fact(n - 1)` — runs with its four products computed in the
fabric and its return jumps resolved by the register read port from the control-flow
arc. The fallback counter reads zero.

And one more honest reveal, because this project runs on them: midway through, I
discovered the fabric's tile library already contained a **single-tile 64-bit
multiply primitive**, sitting unused since an early math-tiles experiment. I could
have closed the gap with one tile and a straight face — `Add` is a single-tick 64-bit
primitive too, and nobody objects to adders. I didn't use it. A claim like "the
machine computes its own multiplies" is worth more when the multiplier is built from
parts you've already justified, and the unused primitive now serves as an in-fabric
oracle. The gap is closed **by construction, not by definition.**

The remaining residual is a documented freeze behavior on HALT at wide addresses,
plus an optional polish ladder (the MUL sequencing itself is executor-orchestrated —
the *product bits* are tile-computed, and the counter says so every CI run). Same
rule as everything else here: it counts when the tiles compute it.

---

*Repo: github.com/wazi893/TileUniverse (engine + players). The sprint reports with
probe evidence, floorplans, and the authority audit live alongside the code.*

<!-- launch-pack notes (not part of the post):
Resume bullets on offer:
- Designed and hand-placed ~1,600 tiles of physical CPU structure (16:1 register read
  port, target-selection cascades, a 64-bit shift-add multiplier island) in a 3D
  cellular-automaton fabric, with machine-validated floorplans and capture-moment
  probes — zero regressions across 45 byte-identical golden benchmarks.
- Drove an 8-sprint verification campaign converting software-written control flow
  AND the last software-computed ALU operation to hardware-computed values
  (trust-verify-fallback with zero-expected counters asserted in CI); falsified two
  design premises with instrumentation before build, avoiding weeks of misdirected
  routing.
- Closed the final computation gap by construction, not definition: declined an
  existing single-tile multiply primitive in favor of a from-parts multiplier, with
  compiled recursive fact(5) as the end-to-end proof (recursion returns + products
  all tile-computed, fallback counter zero).
- HN/r/rust hook candidate: "My cellular-automaton CPU now computes its own
  multiplies — 65 cycles each, like it's 1978 (and why I refused the one-tile
  shortcut)". Honest reveal: simulated fabric, delivery-grade labeled, sequencing
  orchestration stated.
-->
