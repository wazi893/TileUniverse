# Tile CPU V2 Complete Analysis (Sprint 90 to Sprint 100)

## Document Metadata
- Date: 2026-02-12
- Audience: Senior engineers and architects with no codebase access
- Scope: Tile CPU V2 evolution from Sprint 90 foundation through Sprint 100 next-PC ingress physicalization
- Evidence baseline:
  - Git commits through `77c4553` (2026-02-12 01:20:11 -0500)
  - Sprint planning and execution documents under `User notes/SPRINTS/SPRINT 90.0` through `User notes/SPRINTS/SPRINT 100.0`

## 1. Executive Summary
Tile CPU V2 is a rapid, high-risk physicalization program that moved from a hybrid 16-bit bootstrap into a deeply routed, pipeline-structured architecture in roughly one day of concentrated implementation. The work succeeded on its own terms:

1. Functional scope expanded from initial hybrid execution to mostly tile-driven fetch/decode/select/commit/dataflow behavior.
2. Software write pressure on default execution path was reduced in planned stages (`17 -> 9 -> 7 -> 5 -> 4` depending on milestone and path class).
3. Regression discipline was consistently enforced via growing V2 and tile_cpu test suites.
4. The system is now architected for further physicalization (full branch network and/or physical inter-stage latch), with explicit route ownership and layered guard practices.

The cost profile is also clear:

1. Physical route depth and control fabrics increased critical path relative to earlier light-hybrid points.
2. A significant portion of control authority is still software-latched in Stage X for correctness and scope control.
3. Documentation and implementation cadence diverged at times (plan docs marked draft/planned while code already landed).

## 2. Method and Evidence Model
This analysis uses two source classes and treats them differently:

1. Ground truth for what shipped: git commits and committed source state.
2. Ground truth for intent, acceptance gates, and observed metrics: sprint docs and execution reports.

### 2.1 Commit Spine (Core V2)
The primary V2 code progression is:

1. `bab794c` - Sprint 90 foundation
2. `ee23d8b` - Sprint 91 fetch plus deferred ops
3. `f6b67bc` - Sprint 92 decoder LUT cutover
4. `ccf4dca` - Sprint 93 operand trees
5. `19a23c1` - Sprint 94 commit fabric
6. `0210ab0` - Sprint 96 two-stage software pipeline
7. `ffd3daf` - Sprint 98 dirty plus WE physicalization
8. `82cc0c7` - Sprint 99 ALU operand delivery physicalization
9. `77c4553` - Sprint 100 next-PC ingress physicalization

Supporting doc commits:

1. `8c6e670` - Sprint 95/96 plans published
2. `fd2ca8d` - Sprint 97 to 99 plans plus Sprint 98 execution report published

### 2.2 Documentation Integrity Note
Some sprint docs are intentionally planning artifacts and remain labeled `Planned` or `DRAFT FOR EXECUTION` even when implementation later landed. Therefore:

1. Status labels in plan docs are not treated as delivery truth.
2. Delivery truth comes from commit content plus explicit execution reports.

### 2.3 Workspace State Caveat
At analysis time, the repository had a large dirty working tree (including uncommitted Sprint 101 era edits). This report intentionally anchors to committed history through `77c4553` and does not rely on uncommitted behavior for conclusions.

## 3. Strategic Intent: Why V2 Exists
V2 was created as a parallel architecture path rather than an in-place refactor of V1.

Core intent from Sprint 90:

1. Move from 8-bit ISA pressure to a 16-bit design with 32 opcodes and 8 registers.
2. Preserve V1 stability by isolating V2 in new files.
3. Build a clean base where physicalization and later pipelining make architectural sense.

This decision is significant: it protected V1 throughput while allowing aggressive V2 experiments in routing, layering, and control decomposition.

## 4. Current System (Post Sprint 100)
The committed post-Sprint-100 V2 should be understood as a staged hybrid-physical system with explicit split of concerns.

### 4.1 Execution Model
1. Two-stage software pipeline with X-then-F ordering.
2. Stage X executes prior latch, commits, resolves control authority.
3. Stage F fetches/decode/selects for next latch.

Effect:

1. No RAW forwarding network required for current schedule.
2. Branch penalty avoided under current ordering model.
3. `critical_path_deltas` in metrics is a projected stage-max metric, not full-cycle total work.

### 4.2 Physical Data/Control Fabrics Present
1. Physical ROM fetch surfaces (including high-lane handling).
2. Physical extraction zone and decoder LUT evaluation path.
3. Physical operand trees with cross-bank data physicalization.
4. Physical ALU default operand delivery with conditional override muxing.
5. Physical commit fabric for register and flag capture through clocked tiles.
6. Physical default next-PC path plus physical ingress mux with software-latched override bridge.

### 4.3 Software Authority That Remains
1. Stage X branch decision logic using latched control and software LR handling.
2. Selected control bridges (for correctness and scope boundaries).
3. Pipeline control and metrics accounting remain software-managed.

## 5. Sprint-by-Sprint Deep Analysis

## 5.1 Sprint 90: 16-bit Hybrid Foundation (`bab794c`)
### What changed
1. Introduced dedicated V2 module stack (`v2_builder`, `v2_components`, `v2_execute`, `v2_wiring`, `v2_tests`).
2. Added 16-bit instruction helpers and decode model.
3. Integrated V2 benchmark path in `examples/tile_cpu_metrics.rs`.

### Why this mattered
1. Established clean architectural sandbox without destabilizing V1.
2. Locked major contracts: 16-word ROM, 8-cell RAM (for this phase), 8-register ISA surface.
3. Created independent test harness needed for later physicalization iterations.

### Scale
- 7 files changed, 1403 insertions, 1 deletion.

## 5.2 Sprint 91: Fetch Hardening and Deferred Opcodes (`ee23d8b`)
### What changed
1. Fetch path moved from software-program-vector reliance to tile-authoritative ROM outputs.
2. Guard sweep and routing fixes addressed high-lane fetch integrity issues.
3. Deferred opcodes completed (ANDI/ORI/XORI/LDB/STB), filling all 32 opcode slots.

### Verification outcome
Execution report states:
1. `v2_` tests: 31/31 pass.
2. Full lib tests: 2273 pass, 0 fail.
3. Metrics run completed.

### Technical significance
1. Closed the biggest semantic gap between intended tile execution and software fallback behavior.
2. Converted ISA completeness from planned to executed.

## 5.3 Sprint 92: Physical Decoder LUTs (`f6b67bc`)
### What changed
1. Decoder moved to tile LUT path (`ctrl_a`, `ctrl_b`) with software-fed selector bridge.
2. Tick flow moved from `V2ControlSignals::decode()` authority to tile control bundle authority.
3. Execute path refactored toward control-bit driven execution.

### Architectural tradeoff
1. Full opcode extraction from high ROM lane remained constrained by layout.
2. Chosen approach used software selector bridge while preserving physical LUT evaluation.

### Scale
- 3 files changed, 415 insertions, 305 deletions.

## 5.4 Sprint 93: Physical Operand Trees (`ccf4dca`)
### What changed
1. Added physical Tree A/Tree B mux networks.
2. Cut operand reads to tree roots.
3. Used hybrid ingress mapping where full collision-free routing was not yet feasible.

### Why this mattered
1. Operand selection moved materially into fabric.
2. Established parity testing model for selector correctness and cross-product coverage.

### Reported outcome
Verification section reports:
1. `v2_` tests: 37 pass.
2. Full lib: 2279 pass.
3. V2 benchmark: max critical path 20 deltas, avg 14.8, TEPS 22.5 M/s.

## 5.5 Sprint 94: Clocked Commit for Regs and Flags (`19a23c1`)
### What changed
1. Introduced commit fabrics and dirty sets for register/flag capture.
2. Reframed tick cycle around setup plus commit plus sync behavior.
3. Shifted branch checks toward committed tile flag state usage.

### Architectural significance
This sprint is where V2 stopped being "physical combinational islands with software writes" and became a clocked commit machine with explicit tile capture semantics.

### Scale
- 6 files changed, 861 insertions, 177 deletions.

## 5.6 Sprint 95: Unified Plan Published (`8c6e670`) and Deferred Integration
Sprint 95 is unusual in repository history:

1. A comprehensive "unified final" plan was committed as docs.
2. There is no dedicated same-day Sprint 95 code commit in the core chain.
3. Many Sprint 95 concepts (extraction unification, selector path simplification, chain collapse themes) appear later embedded into Sprint 96 to Sprint 98 code and docs.

Conclusion: Sprint 95 functioned as a design and sequencing consolidation milestone more than a single discrete code drop.

## 5.7 Sprint 96: Two-Stage Software Pipeline (`0210ab0`)
### What changed
1. Introduced `PipelineLatch` and stage decomposition (`run_stage_f`, `run_stage_x`).
2. Implemented X-then-F execution order.
3. Added projected critical-path model (stage max) while keeping total work accounting.

### Why this mattered
1. Created timing abstraction for future physical inter-stage latch work.
2. Clarified IPC accounting vs cycles under fill/drain semantics.
3. Reduced architectural ambiguity in branch and dependency behavior under staged execution.

### Scale
- 2 files changed, 479 insertions, 194 deletions.

## 5.8 Sprint 97: Cross-bank Data Physicalization (Docs + Delivered via 98-era chain)
Sprint 97 content is split across:

1. Plan docs (`SPRINT_97_V2_CROSS_BANK_DATA_PHYSICALIZATION.md`).
2. Phase notes (`SPRINT_97_PHASE_97_0_EXECUTION.md`).
3. Completion report (`SPRINT_97_EXECUTION_COMPLETE.md`).
4. Delivery effects visible in subsequent code state and Sprint 98 baseline.

### Reported key outcomes
1. Cross-bank operand data fully physicalized.
2. Stage-F software bridge writes removed.
3. Layering hardening with L2 guards and route ownership discipline.

### Reported validation snapshot
1. `v2_`: 48/48.
2. `tile_cpu`: 131/131.
3. Full lib: 2290 pass.
4. V2 max critical path: 28 deltas.

## 5.9 Sprint 98: Dirty Optimization and WE Path Physicalization (`ffd3daf`)
### What changed
1. Split pipeline dirty marking into backbone plus source-dependent register data sets.
2. Added conservative full-resync fallback to avoid stale-state bugs.
3. Replaced per-register WE toggling pattern with one-hot mask source and physical distribution.
4. Consolidated flag WE into packed mask source with physical split.

### Why this mattered
1. This is a major control-surface reduction sprint.
2. It materially reduced Stage X software writes (reported `9 -> 7`).
3. It improved maintainability by removing transient reset patterns.

### Reported metrics
Pre-98 vs post-98 (V2 Fibonacci):
1. Tile count: 13973 -> 14054.
2. Max critical path: 28 -> 29 deltas.
3. Simulated freq: 35.7 -> 34.5 MHz.
4. TEPS: 6.0 -> 5.6 M/s.
5. Avg eval/tick: 2700 -> 2643.
6. Avg switched/tick: 451 -> 509.

Interpretation: write reduction succeeded, but physical complexity continued to push timing and throughput guardrails.

## 5.10 Sprint 99: Physical ALU Operand Delivery (`82cc0c7`)
### What changed
1. Routed tree outputs physically to ALU inputs for default path.
2. Introduced per-ALU override mux pattern for immediate/special opcode cases.
3. Added stale-enable transition handling in execute path.

### Why this mattered
1. Default ALU ticks can avoid direct operand constant writes.
2. Explicitly separates default physical flow from override bridge behavior.
3. Prepared system for next-PC optimization by reducing remaining write classes.

### Scale
- 3 files changed, 227 insertions, 57 deletions.

## 5.11 Sprint 100: Next-PC Ingress and Override Bridge (`77c4553`)
### What changed
1. Physicalized default fallthrough next-PC computation path.
2. Added explicit 4-bit mask stage in layered route.
3. Added physical PC ingress mux.
4. Removed unconditional default-path next-PC write.
5. Added override tracking and clear-on-transition behavior.

### Reported validation snapshot
1. `v2_`: 56 pass.
2. `tile_cpu`: 139 pass.
3. Full lib: 2298 pass.
4. V2 max critical path: 31 deltas.
5. V2 avg critical path: 27.1 deltas.
6. V2 estimated max MHz: 32.3.
7. V2 tile count: 14215.

### Key effect
Default-path write profile reduced from 5 to 4 writes/tick class, with override-path writes intentionally variable by control-flow type.

## 6. Quantitative Evolution Summary

| Milestone | Writes/tick (default class) | Propagates/tick | Max critical path | Est max MHz | V2 tests | Notes |
|---|---:|---:|---:|---:|---:|---|
| Sprint 91 report | not standardized | not standardized | not listed in report block | n/a | 31 | Fetch tile authority and full opcode coverage complete |
| Sprint 93 report | n/a | n/a | 20 | n/a | 37 | Early strong timing before deep physical routing load |
| Sprint 96 baseline doc (post-95A) | 17 | 3 | 171 | 5.8 | 41 | Software-latch model and heavy staged overhead point |
| Sprint 97 execution report | 9 (reported baseline to 98) | 3 | 28 | 35.7 | 48 | Cross-bank physicalization phase considered complete |
| Sprint 98 execution report | 7 | 3 | 29 | 34.5 | 53 | WE physicalization and dirty optimization delivered |
| Sprint 99 baseline in Sprint 100 doc | 5 | 3 | 31 | 32.3 | 53 | Post-ALU operand physicalization state |
| Sprint 100 execution report | 4 (default path) | 3 | 31 | 32.3 | 56 | Next-PC ingress physicalized |

## 7. Code Churn and Concentration
From the 9 core V2 commits (90,91,92,93,94,96,98,99,100):

1. Total delta volume is very high relative to elapsed time.
2. `src/tile_cpu/v2_wiring.rs` and `src/tile_cpu/v2_execute.rs` are the primary concentration points.
3. `src/tile_cpu/v2_tests.rs` grew continuously as a safety scaffold and parity oracle.

Per-commit shortstat recap:

1. Sprint 90: 1403 insertions.
2. Sprint 91: 373 insertions, 48 deletions.
3. Sprint 92: 415 insertions, 305 deletions.
4. Sprint 93: 504 insertions, 16 deletions.
5. Sprint 94: 861 insertions, 177 deletions.
6. Sprint 96: 479 insertions, 194 deletions.
7. Sprint 98: 875 insertions, 247 deletions.
8. Sprint 99: 227 insertions, 57 deletions.
9. Sprint 100: 620 insertions, 19 deletions.

Engineering interpretation: the program favored correctness-preserving iterative replacement over compactness, then stabilized by adding tests before each high-risk cutover.

## 8. Verification and Test Maturity
The V2 testing strategy evolved from simple opcode checks to architecture-level invariants:

1. Fetch authority tests (ROM tile mutation and PC progression).
2. Decoder parity tests across opcode space.
3. Operand-tree parity and cross-product tests.
4. Pipeline fill/drain, branch no-penalty, dependency tests.
5. Route integrity assertions for critical coordinates and layers.
6. WE and flag mask parity tests.
7. Next-PC override transition and wrap behavior tests.

Reported suite growth trend:
1. V2 tests: 31 -> 37 -> 48 -> 53 -> 56.
2. tile_cpu tests: 120 -> 131 -> 136 -> 139.
3. Full lib tests: 2273 -> 2279 -> 2290 -> 2295 -> 2298.

## 9. Design Strengths
1. Strong isolation from V1 prevented collateral regressions during high-change periods.
2. Layer guard and reservation practices reduced silent routing corruption risk.
3. Cutover discipline was generally phase-gated and test-backed.
4. Explicit handling of stale-control transitions (ALU/PC override) shows mature hazard awareness.
5. Architecture now has clear insertion points for final physicalization steps.

## 10. Weaknesses and Risks
1. Throughput/timing narrative is non-monotonic and sometimes plan-driven rather than measured at each discrete step.
2. Documentation status fields are not consistently synchronized with implementation reality.
3. System remains hybrid in key control domains; full physical authority is not complete.
4. Large route and guard surfaces increase maintenance burden and onboarding cost.
5. Current codebase has significant concurrent local work, increasing integration and attribution risk if not tightly staged.

## 11. Deferred Work and Strategic Next Steps
Based on Sprint 100 outcomes and deferred notes, highest-leverage next work is:

1. Physical branch decision network from live decode/flags (remove remaining Stage X software authority where safe).
2. Physical inter-stage latch (clock-domain-aware), converting projected stage timing model into actual hardware-equivalent staging.
3. Continued route ownership rigor and focused dead-field cleanup as physical control paths replace bridges.

## 12. Key Takeaways for External Experts
1. V2 is no longer a prototype branch; it is a substantial architecture with disciplined physicalization scaffolding.
2. The main achievements are control/data path relocation into fabric and systematic write-pressure reduction under guarded correctness.
3. The main debt is remaining hybrid control authority plus complexity-driven timing pressure.
4. The project now sits at a meaningful decision point: pursue full physical control closure, or lock current architecture and optimize route/timing quality.

## Appendix A: Primary Artifacts Used

### A.1 Core code commits
- `bab794c` - Sprint 90 V2 foundation
- `ee23d8b` - Sprint 91 fetch plus deferred ops
- `f6b67bc` - Sprint 92 decoder LUTs
- `ccf4dca` - Sprint 93 operand trees
- `19a23c1` - Sprint 94 writeback plus flag commit
- `0210ab0` - Sprint 96 two-stage pipeline
- `ffd3daf` - Sprint 98 dirty plus WE path physicalization
- `82cc0c7` - Sprint 99 ALU operand delivery
- `77c4553` - Sprint 100 next-PC ingress and override bridge

### A.2 Supporting documentation commits
- `8c6e670` - Sprint 95/96 doc publication
- `fd2ca8d` - Sprint 97/98/99 doc publication and Sprint 98 execution report

### A.3 Sprint docs (selected)
- `User notes/SPRINTS/SPRINT 90.0/SPRINT_90_V2_16BIT_FOUNDATION.md`
- `User notes/SPRINTS/SPRINT 91.0/SPRINT_91_V2_PHYSICAL_FETCH_AND_DEFERRED_OPS.md`
- `User notes/SPRINTS/SPRINT 91.0/SPRINT_91_IMPLEMENTATION_REPORT.md`
- `User notes/SPRINTS/SPRINT 92.0/SPRINT_92_V2_PHYSICAL_DECODER.md`
- `User notes/SPRINTS/SPRINT 93.0/SPRINT_93_V2_FIELD_EXTRACTION_AND_OPERAND_TREES.md`
- `User notes/SPRINTS/SPRINT 94.0/SPRINT_94_V2_PHYSICAL_WRITEBACK_FLAGS.md`
- `User notes/SPRINTS/SPRINT 95.0/SPRINT_95_UNIFIED_FINAL.md`
- `User notes/SPRINTS/SPRINT 96.0/SPRINT_96_V2_TWO_STAGE_PIPELINE.md`
- `User notes/SPRINTS/SPRINT 97.0/SPRINT_97_V2_CROSS_BANK_DATA_PHYSICALIZATION.md`
- `User notes/SPRINTS/SPRINT 97.0/SPRINT_97_PHASE_97_0_EXECUTION.md`
- `User notes/SPRINTS/SPRINT 97.0/SPRINT_97_EXECUTION_COMPLETE.md`
- `User notes/SPRINTS/SPRINT 98.0/SPRINT_98_V2_STAGE_X_CLEANUP_AND_WE_PATH_PHYSICALIZATION.md`
- `User notes/SPRINTS/SPRINT 98.0/SPRINT_98_EXECUTION_REPORT.md`
- `User notes/SPRINTS/SPRINT 99.0/SPRINT_99_V2_PHYSICAL_ALU_OPERAND_DELIVERY.md`
- `User notes/SPRINTS/SPRINT 100.0/SPRINT_100_COORD_LOCK.md`
- `User notes/SPRINTS/SPRINT 100.0/SPRINT_100_V2_NEXT_PC_INGRESS_AND_OVERRIDE_PATH.md`
- `User notes/SPRINTS/SPRINT 100.0/SPRINT_100_EXECUTION_REPORT.md`

## Appendix B: Reproduction Commands
The following commands were used to construct this analysis:

```powershell
git log --date=short --pretty=format:"%h %ad %s" -- src/tile_cpu
git show --name-only --pretty=format:"%h %s" <commit>
git show --shortstat --pretty=format:"" <commit>
rg -n "tile cpu|v2|Sprint" "User notes/SPRINTS"
rg -n "struct TileCpuV2|run_stage_f|run_stage_x|tick" src/tile_cpu/v2_execute.rs
rg -n "V2CpuIndices|wire_v2_cpu" src/tile_cpu/v2_wiring.rs
rg -n "^fn test_v2_" src/tile_cpu/v2_tests.rs
```
