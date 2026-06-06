# Physical CPU Routing Report

## Status After Phase 1

The PhysicalCpu now has a fully tile-based decoder:
- IR → Shr(IR, 4) → WR fan-out → 4 spread Mux8to1 LUTs (row 6, cols 11/15/19/23)
- BitSelect(opcode, 3) for bank select (row 11)
- IR field extraction: BitSelect tiles for rd_bit0/1, rs_bit0/1 (rows 14-17)
- jump_target = And(IR, 0x3F) (row 18)

Bank merge stays in software (3 lines in `decode_physical()`). All 48 tests pass.

`PhysicalCpu::tick()` still has ~80 lines of software staging identical to `TileCpu::tick()`. The decoder portion went from ~10 lines (write opcode_lo Const + propagate + bank select) to ~3 lines (read Shr output + bank select). Net savings: ~7 lines. The remaining ~73 lines handle operand routing, ALU staging, register writeback, flags, branches, and memory.

## The Core Routing Problem

### Why phases 3-6 failed

The 2D tile grid has one fundamental constraint: **a tile can only be one type**. A cell cannot simultaneously be WireRight (carrying signal A horizontally) and WireDown (carrying signal B vertically). When two routing paths cross, one must yield.

This constraint makes multi-signal fan-out from stacked sources to spread destinations impossible without dedicated routing space.

### Concrete example: register-to-mux-tree routing

Registers are stacked at col 7, rows 21/23/25/27 (Reg0 through Reg3). The 4:1 mux tree expects data at cols 8/10/12/14 (D3/D2/D1/D0) on row 30.

Attempt 1 — WR fan-out per register row:
```
Row 21 (Reg0): Register8@7 → WR@8 → WR@9 → ... → WR@14  (Reg0 at col 14)
Row 23 (Reg1): Register8@7 → WR@8 → WR@9 → ... → WR@12  (Reg1 at col 12)
Row 25 (Reg2): Register8@7 → WR@8 → WR@9 → WR@10        (Reg2 at col 10)
Row 27 (Reg3): Register8@7 → WR@8                         (Reg3 at col 8)
```
Then WD from each target column down to row 30. But the WD chain for Reg0 (col 14) must pass through rows 22-29, including Reg1's row 23 where col 14 has a WR tile carrying Reg1's value. Placing WD at (14, 23) overwrites the WR, breaking Reg1's fan-out chain. Worse, the WD reads from above (Reg0's value from row 22), not from Reg1's WR chain.

Attempt 2 — dedicated routing columns: Same problem. Any vertical WD channel that passes through another register's fan-out row picks up that register's value instead of maintaining the value from above.

Attempt 3 — spread registers across columns: Placing Reg0 at col 14, Reg1 at col 12, etc. eliminates the stacking problem but requires 4 columns × 3 tiles per register = 12 columns for the register file alone, plus guard columns. The register writeback Mux feedback loop (Mux→Register8→Mux) constrains relative positions. Alignment with tree data positions is awkward.

### Same problem affects all data routing

| Source | Destination | Signals | Conflict |
|--------|------------|---------|----------|
| 4 registers (col 7, rows 21-27) | Tree A data (cols 8/10/12/14, row 30) | 4 | WD crosses WR |
| 4 registers (col 7, rows 21-27) | Tree B data (cols 16/18/20/22, row 30) | 4 | WD crosses WR |
| 8 ALU ops (col 11, rows 36-43) | Result tree data (spread cols, row 45+) | 8 | Same stacking issue |
| Decoder ctrl bits | Selector positions across trees | ~6 | Horizontal crosses vertical |
| Branch signals | PC area (row 1) | 1 | 50+ row WireUp chain |

### Bank merge — the simplest crossing fails

Even routing one signal (opcode bit 3) horizontally across four vertical LUT output WD chains (cols 11/15/19/23) is impossible. The WR chain from the BitSelect at col 8 must pass through cols 11, 15, 19, 23 — but those cells are WD tiles carrying LUT outputs downward. A tile cannot be both WR and WD.

## Alternative Approaches

### A. Crossbar tile type

Add a new `TileType::Crossbar` that passes two signals through the same cell:
- Horizontal: output = left (passes through to right neighbor)
- Vertical: output = up (passes through to down neighbor)

This directly solves every routing conflict. A Crossbar at the intersection of a WR chain and a WD chain allows both signals to pass through independently.

**Pros**: Minimal layout changes. Every routing conflict becomes trivially solvable. The existing spread decoder layout, register layout, and mux tree layout all remain.

**Cons**: New tile type in the simulation engine. Needs careful definition of "output" (two outputs per tile breaks the single-output-per-tile model). Evaluation semantics need thought — does the tile have one stored value or two?

**Implementation sketch**: The tile stores two values (horizontal and vertical). Neighbors read the appropriate one based on direction. `eval_tile` updates both from `left` and `up` inputs.

### B. Multi-row register layout with interleaved routing

Spread registers so each owns a unique set of columns, with no stacking:

```
Row 20-21: Reg0 (result@13, Mux@14, Reg8@15)  →  WD@15 goes straight down
Row 22-23: Reg1 (result@11, Mux@12, Reg8@13)  →  WD@13 goes straight down (col 13 free above)
                                                    But col 13 = result tile for Reg0. Conflict.
```

To avoid column conflicts, use wider spacing:

```
Reg0: cols 6-8,   Reg8 output at col 8   → feeds Tree A D0 at col 14 via WR@row21
Reg1: cols 10-12, Reg8 output at col 12  → feeds Tree A D1 at col 12 directly!
Reg2: cols 14-16, Reg8 output at col 16  → feeds Tree B D3 at col 16 directly!
Reg3: cols 18-20, Reg8 output at col 20  → feeds Tree B D1 at col 20 directly!
```

Each register's Reg8 is at a unique column. With careful column assignment, some tree data positions align directly (no routing needed). Others need short WR hops. Critical: no register occupies another's routing column.

**Pros**: No new tile types. Pure layout optimization.

**Cons**: Larger grid footprint. Complex alignment math. Register writeback routing (result bus to all 4 registers) becomes harder with spread positions. Mux feedback loops must still work.

### C. Time-multiplexed routing

Use the existing grid but execute routing in software-orchestrated phases within a single tick:

1. Phase 1: Write Reg0 value to a shared routing Const tile. Propagate. Read at destination.
2. Phase 2: Write Reg1 value. Propagate. Read at destination.
3. ... repeat for all signals.

This is essentially what the current software staging does, but formalized as a multi-phase tile evaluation.

**Pros**: No layout changes. No new tile types. Correct by construction.

**Cons**: This IS the current software approach with extra steps. Doesn't reduce software complexity — adds it. Multiple `propagate_combinational()` calls per tick is expensive.

### D. Wider routing channels with bridge tiles

Insert dedicated routing rows between components. Use `WireDown` tiles at routing columns and `guard` tiles everywhere else. Register values reach the routing channel via short WR hops on their own rows, then WD down through the clean channel.

The key insight: if the routing channel has NO WR chains (only WD tiles and guards), there are no horizontal/vertical conflicts.

```
Rows 20-27: Registers (stacked, col 7), with WR hops to unique columns on each reg row
Row 28:     Reg0 arrives via WD (started at row 22), col 14.  Guard everywhere else.
Row 29:     Reg0 WD continues. Reg1 arrives via WD (started at row 24), col 12.
Row 30:     Reg0 WD. Reg1 WD. Reg2 arrives (started at row 26), col 10.
Row 31:     Reg0 WD. Reg1 WD. Reg2 WD. Reg3 arrives (started at row 28), col 8.
Row 32:     All 4 values available at cols 8/10/12/14. Feed into tree data row.
```

The problem was that WD routing columns pass through OTHER registers' WR chains. The fix: ensure each register's WR chain is SHORT — only extends to its own routing column, not beyond.

```
Reg0 (row 21): WR from col 7 to col 14 only  (NOT to col 22)
Reg1 (row 23): WR from col 7 to col 12 only
Reg2 (row 25): WR from col 7 to col 10 only
Reg3 (row 27): WR from col 7 to col 8 only
```

Now Reg0's WD at col 14 passes through rows 22-31. On row 23 (Reg1's row), col 14 is NOT a WR tile — Reg1's WR chain only goes to col 12. So col 14 on row 23 is a guard (or WD), not WR. The WD chain continues cleanly.

But wait — col 14 on Reg1's reg row (23) was placed as a WR tile in the old code because the WR chain extended to col 20. With the short chains, col 14 on row 23 is free. However, the WD at (14, 23) reads up=(14, 22). What's at (14, 22)? That's Reg1's WE row. It was a guard. The WD chain from (14, 22) reads up=(14, 21)=WR@Reg0's row, which carries Reg0's value. Correct!

**This approach actually works for Tree A.** But for Tree B (cols 16/18/20/22), Reg0 needs to reach col 22. Its WR chain on row 21 would need to go from col 7 to col 22 — that's a long WR chain that passes through cols 10, 12, 14 which are WD routing columns for other registers.

Wait — on Reg0's OWN row (21), there are no WD tiles. The WD tiles for other registers start BELOW Reg0's row. On row 21, cols 8-22 can all be WR tiles carrying Reg0's value. WD tiles at cols 10/12/14 only start from row 22 downward (for Reg0's Tree A routing) and from row 24+ for Reg1, etc.

So the WR chains on each register's own row DON'T conflict with WD routing, because WD routing only starts on the row BELOW that register.

For Tree B routing:
```
Reg0 (row 21): WR from col 7 to col 22. WD starts at col 14 (row 22→32) AND col 22 (row 22→32).
Reg1 (row 23): WR from col 7 to col 20. WD at col 12 (row 24→32) AND col 20 (row 24→32).
Reg2 (row 25): WR from col 7 to col 18. WD at col 10 (row 26→32) AND col 18 (row 26→32).
Reg3 (row 27): WR from col 7 to col 16. WD at col 8 (row 28→32) AND col 16 (row 28→32).
```

**Conflict check**: Does Reg0's WD at col 22 (starting row 22) pass through any WR chains of lower registers?

- Row 23 (Reg1): WR chain goes from col 7 to col 20. Col 22 is beyond Reg1's WR chain. No conflict.
- Row 25 (Reg2): WR chain goes from col 7 to col 18. Col 22 is beyond. No conflict.
- Row 27 (Reg3): WR chain goes from col 7 to col 16. Col 22 is beyond. No conflict.

Does Reg0's WD at col 14 pass through lower WR chains?
- Row 23 (Reg1): WR to col 20. Col 14 IS within the WR chain. CONFLICT.

The WR chain at col 14 on row 23 carries Reg1's value. A WD tile at (14, 23) would overwrite this WR tile, breaking Reg1's chain to col 20.

**Fix**: On row 23, col 14 must be WR (for Reg1's chain). But we also need Reg0's value to pass through from (14, 22) to (14, 24). Can't do both.

**Potential fix**: Reg1 doesn't need its WR chain to pass through col 14 if Reg1's WD starts at col 12 (not col 14). But Reg1 also needs to reach col 20 (Tree B). Its WR chain must go from col 7 to col 20, passing through col 14.

### Variation D2: Staggered WR endpoints to avoid crossing

What if we change which register maps to which tree data position?

Currently: Reg0→D0, Reg1→D1, Reg2→D2, Reg3→D3. The mux tree selector picks based on rd/rs bits.

But the selector mapping is fixed by the tree hardware. D0 is selected when S0=0,S1=0; D3 when S0=MAX,S1=MAX. The ISA encodes rd as 2-bit field. rd=0 should select Reg0, which must be at D0. Can't reassign.

However, we CAN swap which physical tree position corresponds to which logical D index by rearranging the tree itself. If we build the tree so that D0 is at col 8 instead of col 14, the mapping changes. The `wire_mux_tree_4to1` function defines the physical layout — we could create a variant where data positions are ordered differently.

Alternatively, accept the crossing and add extra routing rows:

```
Row 21: Reg0 WR to col 22 (long chain)
Row 22: Reg0 WD at cols 14,22. Guards elsewhere.
Row 23: Reg1 WR to col 20 — but col 14 has Reg0 WD. CONFLICT.
```

Insert a transition row between each register where WR-to-WD handoff happens:

```
Row 21: Reg0 WR to col 22
Row 22: Reg0 WD at cols 14,22. Reg1 WE row. Col 14: WD (Reg0), not guard.
         But Reg1's WE row only uses col 6. Cols 14,22 are free for WD.
Row 23: Reg1 reg row. WR from col 7 to col 20.
         Col 14: Reg0 WD continues? No — Reg1's WR is at col 14.
```

Still conflicts on register reg rows. The WR chain from col 7 to col N MUST traverse all intermediate columns. Any intermediate column that has a WD from above will be overwritten.

**The only way out**: Each register's WR chain must NOT pass through any WD routing column of a higher register.

Assign routing columns by register priority:
- Reg3 (bottom): routes to cols 8, 16. Its WR chain is short (col 7→8 for A, col 7→16 for B).
- Reg2: routes to cols 10, 18. WR chain col 7→10 and 7→18. Doesn't pass through cols 8 or 16.

Wait — WR from col 7 to col 18 passes through col 8 (Reg3 A), col 10 (Reg2 A), col 16 (Reg3 B). If we order registers top-to-bottom as 0,1,2,3 and only route WD downward from each:

- Reg0 (row 21) WD at cols 14,22 starts at row 22, passes through rows 23-27 (other reg rows)
- On row 23 (Reg1), cols 14 and 22 are beyond Reg1's needed WR endpoint of col 20.
  - Col 14: Reg1's WR extends to col 20, so col 14 IS within the chain. CONFLICT.

Unless Reg1's WR chain SKIPS col 14. But WR chains are contiguous — WR reads left. If (14,23) is WD instead of WR, then (15,23) reads left=(14,23)=WD=Reg0's value, not Reg1's. The chain breaks.

### E. New tile type: WireCross

A simpler version of approach A. `WireCross` passes horizontal and vertical signals independently:
- Evaluates to: horizontal_out = left, vertical_out = up
- Right neighbor reads this tile's horizontal_out
- Down neighbor reads this tile's vertical_out

This requires the simulation engine to support tiles with two independent outputs. One implementation: store two values per tile (horizontal and vertical), and have the neighbor-reading logic check the source direction.

This is the most general solution and would unlock not just the CPU routing but any complex circuit in the grid.

### F. Software staging is fine — optimize what matters

The 80 lines of software staging in `tick()` execute in microseconds. The actual performance bottleneck is the `tick_with_delays()` calls which evaluate hundreds of tiles. Making the staging physical wouldn't improve wall-clock performance — it would actually increase it (more tiles to evaluate).

The value of physical staging is conceptual purity (the entire CPU is "just tiles") and educational (demonstrates how routing works in real hardware). If these goals don't justify the complexity, the current hybrid approach is optimal.

**Potential middle ground**: Keep software staging but refactor `tick()` to be cleaner. Extract the decode/execute/writeback phases into named methods. Add metrics tracking for software vs tile execution time. This makes the code more maintainable without the routing complexity.

## Recommendation

**Short term**: Keep the current hybrid PhysicalCpu (physical decoder + software staging). The routing constraints make full physical routing impractical without either new tile types or a fundamentally different layout.

**Medium term**: If full physical routing is desired, implement approach D (wider routing channels) with careful column assignment. This requires:
1. Each register's WR chain extends only to its furthest-needed column
2. WD routing columns are placed at positions that avoid crossing lower registers' WR chains
3. Extra rows between registers and mux trees serve as routing channels
4. Grid size increases from 128x128 to ~160x160

**Long term**: Implement approach A or E (Crossbar/WireCross tile). This is a one-time engine change that permanently solves all routing conflicts, not just for the CPU but for any circuit built on the grid.
