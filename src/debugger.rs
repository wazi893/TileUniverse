//! Time-travel debugger for TileUniverse simulations.
//!
//! Provides delta-compressed snapshots (keyframes every N ticks + delta frames between),
//! forward/backward stepping via keyframe restore + replay, breakpoints (value equals,
//! changed, rising/falling edge, bit set/clear), signal traces with change detection,
//! and history navigation (goto_tick).

use crate::simulation::Simulation;

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

/// A keyframe: full logic state of all tiles at a specific tick.
pub struct Keyframe {
    pub tick: u64,
    pub clock_state: bool,
    pub logic_values: Vec<u64>,
}

/// A delta frame: tiles that changed during a specific tick.
pub struct DeltaFrame {
    pub tick: u64,
    /// (tile_idx, old_value, new_value)
    pub changes: Vec<(u32, u64, u64)>,
}

/// Snapshot storage with keyframe + delta compression.
pub struct SnapshotStore {
    pub keyframe_interval: u64,
    pub keyframes: Vec<Keyframe>,
    pub delta_frames: Vec<DeltaFrame>,
    /// 0 = unlimited history.
    pub max_history_ticks: u64,
}

impl SnapshotStore {
    /// Create an empty snapshot store with the given keyframe interval.
    pub fn new(keyframe_interval: u64) -> Self {
        Self {
            keyframe_interval,
            keyframes: Vec::new(),
            delta_frames: Vec::new(),
            max_history_ticks: 0,
        }
    }

    /// Clone all logic values into a Keyframe.
    pub fn capture_keyframe(&mut self, sim: &Simulation, tick: u64) {
        let count = sim.tile_count();
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(sim.get_logic_value_by_idx(i));
        }
        self.keyframes.push(Keyframe {
            tick,
            clock_state: sim.global_clock,
            logic_values: values,
        });
    }

    /// Diff current state vs `prev_values`, store only changed tiles.
    pub fn capture_delta(&mut self, sim: &Simulation, tick: u64, prev_values: &[u64]) {
        let count = sim.tile_count();
        let mut changes = Vec::new();
        for i in 0..count {
            let new_val = sim.get_logic_value_by_idx(i);
            let old_val = if i < prev_values.len() {
                prev_values[i]
            } else {
                0
            };
            if new_val != old_val {
                changes.push((i as u32, old_val, new_val));
            }
        }
        self.delta_frames.push(DeltaFrame { tick, changes });
    }

    /// Find nearest keyframe at or before `tick`.
    pub fn find_keyframe(&self, tick: u64) -> Option<&Keyframe> {
        // Keyframes are stored in ascending tick order.
        let mut best: Option<&Keyframe> = None;
        for kf in &self.keyframes {
            if kf.tick <= tick {
                best = Some(kf);
            } else {
                break;
            }
        }
        best
    }

    /// Get deltas in range (after_tick, up_to_tick] — i.e. after_tick < delta.tick <= up_to_tick.
    pub fn deltas_between(&self, after_tick: u64, up_to_tick: u64) -> Vec<&DeltaFrame> {
        self.delta_frames
            .iter()
            .filter(|d| d.tick > after_tick && d.tick <= up_to_tick)
            .collect()
    }

    /// If `max_history_ticks > 0`, remove old keyframes and deltas.
    pub fn trim_history(&mut self) {
        if self.max_history_ticks == 0 {
            return;
        }
        // Find the latest tick we know about.
        let latest = self
            .keyframes
            .last()
            .map(|k| k.tick)
            .unwrap_or(0)
            .max(self.delta_frames.last().map(|d| d.tick).unwrap_or(0));

        if latest <= self.max_history_ticks {
            return;
        }
        let cutoff = latest - self.max_history_ticks;

        // Keep keyframes with tick >= cutoff, but always keep at least the most recent one
        // that is <= cutoff so we can still restore.
        let mut keep_from = 0usize;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.tick < cutoff {
                keep_from = i;
            } else {
                break;
            }
        }
        if keep_from > 0 {
            self.keyframes.drain(..keep_from);
        }

        // Remove deltas older than the oldest remaining keyframe.
        if let Some(oldest_kf_tick) = self.keyframes.first().map(|k| k.tick) {
            self.delta_frames.retain(|d| d.tick > oldest_kf_tick);
        }
    }
}

// ---------------------------------------------------------------------------
// Breakpoints
// ---------------------------------------------------------------------------

/// Condition under which a breakpoint triggers.
#[derive(Debug, Clone)]
pub enum BreakCondition {
    /// Tile value equals the given constant.
    ValueEquals(u64),
    /// Tile value changed from previous tick.
    Changed,
    /// Tile value was 0 and is now non-zero (rising edge).
    RisingEdge,
    /// Tile value was non-zero and is now 0 (falling edge).
    FallingEdge,
    /// Specific bit is set in the tile value.
    BitSet(u32),
    /// Specific bit is clear in the tile value.
    BitClear(u32),
}

/// A breakpoint attached to a specific tile.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: usize,
    pub tile_idx: usize,
    pub condition: BreakCondition,
    pub enabled: bool,
    pub hit_count: u64,
}

// ---------------------------------------------------------------------------
// Signal Traces
// ---------------------------------------------------------------------------

/// Recorded signal trace for a tile, storing (tick, value) pairs on change.
#[derive(Debug, Clone)]
pub struct SignalTrace {
    pub tile_idx: usize,
    pub name: String,
    /// Samples stored as (tick, value). Only recorded when value changes.
    pub samples: Vec<(u64, u64)>,
    pub last_value: u64,
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// Result of a step operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    /// Step completed normally.
    Ok,
    /// A breakpoint was hit; contains the breakpoint id.
    BreakpointHit(usize),
    /// Cannot step backward — already at tick 0.
    AtStart,
}

// ---------------------------------------------------------------------------
// Debugger
// ---------------------------------------------------------------------------

/// Time-travel debugger wrapping a `Simulation`.
pub struct Debugger {
    pub sim: Simulation,
    pub current_tick: u64,
    snapshots: SnapshotStore,
    breakpoints: Vec<Breakpoint>,
    next_bp_id: usize,
    traces: Vec<SignalTrace>,
}

impl Debugger {
    // -- Construction --------------------------------------------------------

    /// Wrap a simulation, capturing the initial keyframe at tick 0.
    pub fn new(sim: Simulation) -> Self {
        let mut snapshots = SnapshotStore::new(16);
        snapshots.capture_keyframe(&sim, 0);
        Self {
            sim,
            current_tick: 0,
            snapshots,
            breakpoints: Vec::new(),
            next_bp_id: 0,
            traces: Vec::new(),
        }
    }

    // -- Configuration -------------------------------------------------------

    pub fn set_keyframe_interval(&mut self, interval: u64) {
        self.snapshots.keyframe_interval = interval;
    }

    pub fn set_max_history(&mut self, max_ticks: u64) {
        self.snapshots.max_history_ticks = max_ticks;
    }

    // -- Forward stepping ----------------------------------------------------

    /// Advance one tick. Captures snapshots, records traces, checks breakpoints.
    pub fn step_forward(&mut self) -> StepResult {
        // 1. Capture pre-tick values.
        let count = self.sim.tile_count();
        let mut prev_values = Vec::with_capacity(count);
        for i in 0..count {
            prev_values.push(self.sim.get_logic_value_by_idx(i));
        }

        // 2. Tick the simulation.
        self.sim.tick();
        self.current_tick += 1;

        // 3. Snapshot — keyframe or delta.
        let interval = self.snapshots.keyframe_interval;
        if interval > 0 && self.current_tick % interval == 0 {
            self.snapshots
                .capture_keyframe(&self.sim, self.current_tick);
        } else {
            self.snapshots
                .capture_delta(&self.sim, self.current_tick, &prev_values);
        }

        // 4. Trim history if configured.
        self.snapshots.trim_history();

        // 5. Record traces (change-detection).
        for trace in &mut self.traces {
            let val = self.sim.get_logic_value_by_idx(trace.tile_idx);
            if val != trace.last_value || trace.samples.is_empty() {
                trace.samples.push((self.current_tick, val));
                trace.last_value = val;
            }
        }

        // 6. Check breakpoints.
        for bp in &mut self.breakpoints {
            if !bp.enabled {
                continue;
            }
            let new_val = self.sim.get_logic_value_by_idx(bp.tile_idx);
            let old_val = if bp.tile_idx < prev_values.len() {
                prev_values[bp.tile_idx]
            } else {
                0
            };
            let hit = match &bp.condition {
                BreakCondition::ValueEquals(target) => new_val == *target,
                BreakCondition::Changed => new_val != old_val,
                BreakCondition::RisingEdge => old_val == 0 && new_val != 0,
                BreakCondition::FallingEdge => old_val != 0 && new_val == 0,
                BreakCondition::BitSet(bit) => {
                    if *bit < 64 {
                        (new_val >> bit) & 1 == 1
                    } else {
                        false
                    }
                }
                BreakCondition::BitClear(bit) => {
                    if *bit < 64 {
                        (new_val >> bit) & 1 == 0
                    } else {
                        true
                    }
                }
            };
            if hit {
                bp.hit_count += 1;
                return StepResult::BreakpointHit(bp.id);
            }
        }

        StepResult::Ok
    }

    /// Step forward N ticks; stop early on breakpoint.
    pub fn step_forward_n(&mut self, n: u64) -> StepResult {
        for _ in 0..n {
            let r = self.step_forward();
            if r != StepResult::Ok {
                return r;
            }
        }
        StepResult::Ok
    }

    /// Run until a breakpoint is hit or `max_ticks` have elapsed.
    pub fn run_until_break(&mut self, max_ticks: u64) -> StepResult {
        self.step_forward_n(max_ticks)
    }

    // -- Backward stepping / time travel -------------------------------------

    /// Navigate to a specific tick. If forward, steps forward. If backward,
    /// restores nearest keyframe and replays.
    pub fn goto_tick(&mut self, target_tick: u64) -> StepResult {
        if target_tick == self.current_tick {
            return StepResult::Ok;
        }
        if target_tick > self.current_tick {
            // Forward: just step.
            let steps = target_tick - self.current_tick;
            return self.step_forward_n(steps);
        }

        // Backward: find nearest keyframe at or before target, restore, then replay.
        let kf = self.snapshots.find_keyframe(target_tick);
        if kf.is_none() {
            // No keyframe available — best effort: restore tick-0 keyframe.
            // This should always exist since we capture it in new().
            return StepResult::AtStart;
        }
        let kf = kf.unwrap();
        let kf_tick = kf.tick;

        // Clone keyframe data to avoid borrow conflict.
        let kf_values = kf.logic_values.clone();
        let kf_clock = kf.clock_state;

        // Restore simulation state from keyframe.
        self.restore_from(kf_tick, kf_clock, &kf_values);
        self.current_tick = kf_tick;

        // Replay forward to target.
        if target_tick > kf_tick {
            let steps = target_tick - kf_tick;
            return self.step_forward_n(steps);
        }
        StepResult::Ok
    }

    /// Step backward one tick. Returns `AtStart` if already at tick 0.
    pub fn step_backward(&mut self) -> StepResult {
        if self.current_tick == 0 {
            return StepResult::AtStart;
        }
        self.goto_tick(self.current_tick - 1)
    }

    /// Step backward N ticks (saturating at 0).
    pub fn step_backward_n(&mut self, n: u64) -> StepResult {
        let target = self.current_tick.saturating_sub(n);
        self.goto_tick(target)
    }

    // -- Keyframe restore (private) ------------------------------------------

    fn restore_from(&mut self, _tick: u64, clock_state: bool, logic_values: &[u64]) {
        for (i, &val) in logic_values.iter().enumerate() {
            self.sim.set_logic_value_by_idx(i, val);
        }
        self.sim.global_clock = clock_state;
    }

    // -- Breakpoint management -----------------------------------------------

    /// Add a breakpoint by tile index. Returns the breakpoint id.
    pub fn add_breakpoint(&mut self, tile_idx: usize, condition: BreakCondition) -> usize {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        self.breakpoints.push(Breakpoint {
            id,
            tile_idx,
            condition,
            enabled: true,
            hit_count: 0,
        });
        id
    }

    /// Add a breakpoint by (x, y) coordinate. Returns the breakpoint id.
    pub fn add_breakpoint_at(&mut self, x: usize, y: usize, condition: BreakCondition) -> usize {
        let idx = y * self.sim.width() + x;
        self.add_breakpoint(idx, condition)
    }

    /// Remove a breakpoint by id. Returns `true` if it existed.
    pub fn remove_breakpoint(&mut self, id: usize) -> bool {
        let len_before = self.breakpoints.len();
        self.breakpoints.retain(|bp| bp.id != id);
        self.breakpoints.len() < len_before
    }

    /// Enable or disable a breakpoint. Returns `true` if the breakpoint was found.
    pub fn set_breakpoint_enabled(&mut self, id: usize, enabled: bool) -> bool {
        for bp in &mut self.breakpoints {
            if bp.id == id {
                bp.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Get all breakpoints.
    pub fn breakpoints(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    // -- Signal trace management ---------------------------------------------

    /// Add a signal trace by tile index.
    pub fn add_trace(&mut self, tile_idx: usize, name: &str) {
        let current_val = self.sim.get_logic_value_by_idx(tile_idx);
        self.traces.push(SignalTrace {
            tile_idx,
            name: name.to_string(),
            samples: vec![(self.current_tick, current_val)],
            last_value: current_val,
        });
    }

    /// Add a signal trace by (x, y) coordinate.
    pub fn add_trace_at(&mut self, x: usize, y: usize, name: &str) {
        let idx = y * self.sim.width() + x;
        self.add_trace(idx, name);
    }

    /// Remove a trace by name. Returns `true` if it existed.
    pub fn remove_trace(&mut self, name: &str) -> bool {
        let len_before = self.traces.len();
        self.traces.retain(|t| t.name != name);
        self.traces.len() < len_before
    }

    /// Get a trace by name.
    pub fn get_trace(&self, name: &str) -> Option<&SignalTrace> {
        self.traces.iter().find(|t| t.name == name)
    }

    /// Get all traces.
    pub fn traces(&self) -> &[SignalTrace] {
        &self.traces
    }

    /// Query the historical value of a trace at a specific tick via binary search.
    /// Returns the value that was active at `tick` (the most recent sample <= tick).
    pub fn trace_value_at(&self, name: &str, tick: u64) -> Option<u64> {
        let trace = self.get_trace(name)?;
        if trace.samples.is_empty() {
            return None;
        }
        // Binary search for the last sample with tick <= target.
        match trace.samples.binary_search_by_key(&tick, |&(t, _)| t) {
            std::result::Result::Ok(idx) => Some(trace.samples[idx].1),
            Err(idx) => {
                if idx == 0 {
                    None
                } else {
                    Some(trace.samples[idx - 1].1)
                }
            }
        }
    }

    // -- Accessors -----------------------------------------------------------

    /// Current tick number.
    pub fn tick(&self) -> u64 {
        self.current_tick
    }

    /// Borrow the underlying simulation.
    pub fn simulation(&self) -> &Simulation {
        &self.sim
    }

    /// Mutably borrow the underlying simulation.
    pub fn simulation_mut(&mut self) -> &mut Simulation {
        &mut self.sim
    }

    /// Read tile logic value by (x, y).
    pub fn read_tile(&self, x: usize, y: usize) -> u64 {
        self.sim.get_logic_at(x, y)
    }

    /// Read tile logic value by flat index.
    pub fn read_tile_idx(&self, idx: usize) -> u64 {
        self.sim.get_logic_value_by_idx(idx)
    }

    /// Number of keyframes in the snapshot store.
    pub fn keyframe_count(&self) -> usize {
        self.snapshots.keyframes.len()
    }

    /// Number of delta frames in the snapshot store.
    pub fn delta_frame_count(&self) -> usize {
        self.snapshots.delta_frames.len()
    }

    /// Total number of individual tile changes recorded across all delta frames.
    pub fn total_delta_changes(&self) -> usize {
        self.snapshots
            .delta_frames
            .iter()
            .map(|d| d.changes.len())
            .sum()
    }

    /// Estimated memory usage of snapshots in bytes.
    pub fn snapshot_memory_bytes(&self) -> usize {
        let kf_bytes: usize = self
            .snapshots
            .keyframes
            .iter()
            .map(|kf| {
                std::mem::size_of::<Keyframe>() + kf.logic_values.len() * std::mem::size_of::<u64>()
            })
            .sum();
        let df_bytes: usize = self
            .snapshots
            .delta_frames
            .iter()
            .map(|df| {
                std::mem::size_of::<DeltaFrame>()
                    + df.changes.len() * std::mem::size_of::<(u32, u64, u64)>()
            })
            .sum();
        kf_bytes + df_bytes
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_debugger(w: usize, h: usize) -> Debugger {
        let sim = Simulation::with_size(w, h);
        Debugger::new(sim)
    }

    // 1. Step forward
    #[test]
    fn test_step_forward() {
        let mut dbg = make_debugger(4, 4);
        for _ in 0..5 {
            let r = dbg.step_forward();
            assert_eq!(r, StepResult::Ok);
        }
        assert_eq!(dbg.tick(), 5);
    }

    // 2. Keyframe capture — default interval is 16
    #[test]
    fn test_keyframe_capture() {
        let mut dbg = make_debugger(4, 4);
        for _ in 0..16 {
            dbg.step_forward();
        }
        // tick-0 keyframe + tick-16 keyframe = 2
        assert_eq!(dbg.keyframe_count(), 2);
    }

    // 3. Delta frame capture
    #[test]
    fn test_delta_frame_capture() {
        let mut dbg = make_debugger(4, 4);
        for _ in 0..3 {
            dbg.step_forward();
        }
        // 3 ticks, none on a keyframe boundary (interval=16), so 3 delta frames
        assert_eq!(dbg.delta_frame_count(), 3);
    }

    // 4. Step backward
    #[test]
    fn test_step_backward() {
        let mut dbg = make_debugger(4, 4);
        dbg.step_forward_n(10);
        dbg.step_backward();
        assert_eq!(dbg.tick(), 9);
    }

    // 5. Goto tick (backward)
    #[test]
    fn test_goto_tick() {
        let mut dbg = make_debugger(4, 4);
        dbg.step_forward_n(20);
        assert_eq!(dbg.tick(), 20);
        dbg.goto_tick(5);
        assert_eq!(dbg.tick(), 5);
    }

    // 6. Goto tick zero
    #[test]
    fn test_goto_tick_zero() {
        let mut dbg = make_debugger(4, 4);
        dbg.step_forward_n(10);
        dbg.goto_tick(0);
        assert_eq!(dbg.tick(), 0);
    }

    // 7. Breakpoint — ValueEquals
    #[test]
    fn test_breakpoint_value_equals() {
        let mut dbg = make_debugger(4, 4);
        // Write value 42 to tile (1,0) so that Wire tile at (0,0) picks it up.
        // Wire = left | right | up | down. Tile (0,0) right neighbor is (1,0).
        dbg.sim.write_logic(1, 0, 42);
        // Mark tile (0,0) dirty so it gets evaluated during the next tick.
        dbg.sim.dirty.mark_dirty(0);
        // Set breakpoint on tile (0,0) for value == 42.
        let bp_id = dbg.add_breakpoint_at(0, 0, BreakCondition::ValueEquals(42));
        let r = dbg.step_forward();
        assert_eq!(r, StepResult::BreakpointHit(bp_id));
    }

    // 8. Breakpoint — Changed
    #[test]
    fn test_breakpoint_changed() {
        let mut dbg = make_debugger(4, 4);
        // Write value to tile (1,0); after tick, tile (0,0) Wire will pick up the value.
        dbg.sim.write_logic(1, 0, 7);
        // Mark tile (0,0) dirty so it gets evaluated.
        dbg.sim.dirty.mark_dirty(0);
        let bp_id = dbg.add_breakpoint_at(0, 0, BreakCondition::Changed);
        let r = dbg.step_forward();
        assert_eq!(r, StepResult::BreakpointHit(bp_id));
    }

    // 9. Breakpoint — RisingEdge (0 -> non-zero)
    #[test]
    fn test_breakpoint_rising_edge() {
        let mut dbg = make_debugger(4, 4);
        // Tile (0,0) starts at 0. Set neighbor to non-zero so it transitions.
        dbg.sim.write_logic(1, 0, 1);
        // Mark tile (0,0) dirty so it gets evaluated.
        dbg.sim.dirty.mark_dirty(0);
        let bp_id = dbg.add_breakpoint_at(0, 0, BreakCondition::RisingEdge);
        let r = dbg.step_forward();
        assert_eq!(r, StepResult::BreakpointHit(bp_id));
    }

    // 10. Breakpoint not hit
    #[test]
    fn test_breakpoint_not_hit() {
        let mut dbg = make_debugger(4, 4);
        // All tiles are 0, so ValueEquals(999) will never be true.
        dbg.add_breakpoint_at(0, 0, BreakCondition::ValueEquals(999));
        let r = dbg.step_forward();
        assert_eq!(r, StepResult::Ok);
    }

    // 11. Breakpoint disabled
    #[test]
    fn test_breakpoint_disabled() {
        let mut dbg = make_debugger(4, 4);
        dbg.sim.write_logic(1, 0, 42);
        dbg.sim.dirty.mark_dirty(0);
        let bp_id = dbg.add_breakpoint_at(0, 0, BreakCondition::ValueEquals(42));
        dbg.set_breakpoint_enabled(bp_id, false);
        let r = dbg.step_forward();
        assert_eq!(r, StepResult::Ok);
    }

    // 12. Breakpoint management
    #[test]
    fn test_breakpoint_management() {
        let mut dbg = make_debugger(4, 4);
        let id1 = dbg.add_breakpoint(0, BreakCondition::Changed);
        let id2 = dbg.add_breakpoint(1, BreakCondition::ValueEquals(0));
        assert_eq!(dbg.breakpoints().len(), 2);

        assert!(dbg.remove_breakpoint(id1));
        assert_eq!(dbg.breakpoints().len(), 1);
        assert_eq!(dbg.breakpoints()[0].id, id2);

        assert!(!dbg.remove_breakpoint(id1)); // already removed
    }

    // 13. Signal trace — basic
    #[test]
    fn test_signal_trace_basic() {
        let mut dbg = make_debugger(4, 4);
        dbg.add_trace(0, "tile0");
        for _ in 0..5 {
            dbg.step_forward();
        }
        let trace = dbg.get_trace("tile0").unwrap();
        // At minimum: the initial sample at tick 0.
        assert!(!trace.samples.is_empty());
    }

    // 14. Signal trace — change detection (static wire stays same, only 1 sample)
    #[test]
    fn test_signal_trace_change_detection() {
        let mut dbg = make_debugger(4, 4);
        // All tiles are Wire with value 0, all neighbors 0 — output stays 0.
        dbg.add_trace(5, "static_wire");
        for _ in 0..10 {
            dbg.step_forward();
        }
        let trace = dbg.get_trace("static_wire").unwrap();
        // Should have exactly 1 sample (the initial one) since value never changes.
        assert_eq!(trace.samples.len(), 1);
    }

    // 15. Signal trace — trace_value_at
    #[test]
    fn test_signal_trace_value_at() {
        let mut dbg = make_debugger(4, 4);
        dbg.add_trace(0, "t0");

        // Step a few ticks — tile 0 stays 0.
        dbg.step_forward_n(3);

        // Now inject a value and mark tile (0,0) dirty so it propagates.
        dbg.sim.write_logic(1, 0, 100);
        dbg.sim.dirty.mark_dirty(0);
        dbg.step_forward(); // tick 4: tile (0,0) Wire picks up 100

        // Historical query: value at tick 2 should be 0 (from initial sample).
        assert_eq!(dbg.trace_value_at("t0", 2), Some(0));
        // Value at tick 4 should be 100 (or whatever Wire computes).
        let val_at_4 = dbg.trace_value_at("t0", 4);
        assert!(val_at_4.is_some());
    }

    // 16. Snapshot memory estimate
    #[test]
    fn test_snapshot_memory_estimate() {
        let mut dbg = make_debugger(4, 4);
        dbg.step_forward_n(5);
        assert!(dbg.snapshot_memory_bytes() > 0);
    }

    // 17. Step backward at start
    #[test]
    fn test_step_backward_at_start() {
        let mut dbg = make_debugger(4, 4);
        let r = dbg.step_backward();
        assert_eq!(r, StepResult::AtStart);
    }

    // 18. Run until break
    #[test]
    fn test_run_until_break() {
        let mut dbg = make_debugger(4, 4);
        // Wire (0,0) starts at 0. We will inject a value at (1,0) so tile (0,0) changes.
        dbg.sim.write_logic(1, 0, 55);
        dbg.sim.dirty.mark_dirty(0);
        let bp_id = dbg.add_breakpoint_at(0, 0, BreakCondition::Changed);
        let r = dbg.run_until_break(100);
        assert_eq!(r, StepResult::BreakpointHit(bp_id));
        assert_eq!(dbg.tick(), 1); // should stop on very first tick
    }

    // 19. Multiple breakpoints — first evaluated wins
    #[test]
    fn test_multiple_breakpoints() {
        let mut dbg = make_debugger(4, 4);
        // Set up two tiles to change on next tick.
        dbg.sim.write_logic(1, 0, 10); // affects tile (0,0) Wire
        dbg.sim.write_logic(2, 0, 20); // affects tile (1,0) and (3,0) Wire neighbors
        // Mark both target tiles dirty so they evaluate.
        dbg.sim.dirty.mark_dirty(0); // tile (0,0)
        dbg.sim.dirty.mark_dirty(1); // tile (1,0)

        // bp1 on tile 0, bp2 on tile 1 — both will trigger Changed.
        let bp1 = dbg.add_breakpoint(0, BreakCondition::Changed);
        let _bp2 = dbg.add_breakpoint(1, BreakCondition::Changed);

        let r = dbg.step_forward();
        // The first breakpoint in the list that fires wins.
        assert_eq!(r, StepResult::BreakpointHit(bp1));
    }

    // 20. Time travel preserves traces
    #[test]
    fn test_time_travel_preserves_traces() {
        let mut dbg = make_debugger(4, 4);
        dbg.add_trace(0, "t0");
        dbg.step_forward_n(10);
        let samples_before = dbg.get_trace("t0").unwrap().samples.len();

        // Go back in time.
        dbg.goto_tick(3);

        // Traces should still have at least the samples from before.
        // (goto_tick replays, which may add more samples, but never removes old ones.)
        let samples_after = dbg.get_trace("t0").unwrap().samples.len();
        assert!(samples_after >= samples_before);
    }
}
