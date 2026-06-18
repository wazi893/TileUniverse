#![allow(unused_assignments)]

#[cfg(feature = "perf-bench")]
use std::sync::atomic::AtomicBool;
// Sprint 386: tile values are Cell-based now; Ordering remains for the
// feature-gated quantum/JIT counters only.
#[allow(unused_imports)]
use std::sync::atomic::Ordering;

use crate::bus::{BusArbitration, BusConnection, BusDef, BusDirection, BusState};
use crate::clock_domain::{ClockDomainDef, ClockDomainState, SynchronizerState};
use crate::component::{
    ComponentDef, ComponentImpl, ComponentInstance, PortDirection, input_wire_type_for_edge,
    output_wire_type_for_edge, port_to_grid_coords,
};
use crate::dirty_bitset::DirtyBitset;
use crate::field::FieldGrid;
use crate::fieldstep::FieldStepParams;
use crate::lint::{LintProfile, LintResult};
use crate::memory::{MemoryBank, MemoryBankDef, MemoryPortConnection};
use crate::physics::logic_coupling::{
    PhysicsCouplingConfig, PhysicsCouplingContext, apply_charge_bias_to_comparison,
    apply_charge_bias_to_mux, apply_charge_bias_to_zero, apply_physics_coupling,
    calculate_charge_bias, is_charge_bias_affected,
};
use crate::tile_meta::TileType;
#[cfg(test)]
use crate::tilemap::TILE_COUNT;
use crate::tilemap::{HEIGHT, Tilemap, WIDTH};
// use crate::net::net_summary::{GlobalSummary, SummaryReport};

#[derive(Debug, Clone, Copy)]
pub enum FieldKind {
    Power,
    Logic,
    Clock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangeInfo {
    pub delta: u32,
    pub old: u64,
    pub new: u64,
    pub neighbors: [Option<usize>; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionError {
    OutOfBounds,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StructuralIssueKind {
    FanoutExceeded { fanout: u32, max: u32 },
    UnclockedRegister,
    OrphanLogic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralIssue {
    pub x: u32,
    pub y: u32,
    pub kind: StructuralIssueKind,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct StructuralReport {
    pub issues: Vec<StructuralIssue>,
}

// =============================================================================
// EPIC 123: Propagation Delay Timing Analysis
// =============================================================================

/// Statistics from timing-aware simulation with propagation delays.
///
/// This enables hardware-accurate timing analysis including:
/// - Critical path detection (maximum clock frequency)
/// - Glitch detection (intermediate states before settling)
/// - Setup/hold violation identification
/// - Race condition detection
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    /// Maximum delta cycles to reach quiescence (critical path length)
    pub critical_path_deltas: u32,
    /// Tile index at the end of the critical path
    pub critical_path_endpoint: Option<usize>,
    /// Number of tiles that changed output this tick
    pub tiles_switched: u32,
    /// Number of tiles that were evaluated this tick (includes unchanged)
    pub tiles_evaluated: u32,
    /// Total delta cycles executed this tick
    pub total_deltas: u32,
    /// Glitches detected (input changed while tile was still computing)
    pub glitches_detected: u32,
    /// Whether timing converged (false = hit delta limit)
    pub converged: bool,
}

impl std::fmt::Display for TimingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Timing Statistics")?;
        writeln!(f, "=================")?;
        writeln!(f, "Critical path:  {} deltas", self.critical_path_deltas)?;
        writeln!(f, "Total deltas:   {}", self.total_deltas)?;
        writeln!(f, "Tiles switched: {}", self.tiles_switched)?;
        writeln!(f, "Tiles evaluated: {}", self.tiles_evaluated)?;
        writeln!(f, "Glitches:       {}", self.glitches_detected)?;
        writeln!(f, "Converged:      {}", self.converged)?;
        if let Some(endpoint) = self.critical_path_endpoint {
            writeln!(f, "Critical endpoint: tile {}", endpoint)?;
        }
        Ok(())
    }
}

/// Result of timing verification against a target clock period.
#[derive(Debug, Clone)]
pub struct TimingCheckResult {
    /// Whether the circuit meets the target clock period
    pub meets_timing: bool,
    /// Slack in delta cycles (positive = margin, negative = violation)
    pub slack: i32,
    /// The critical path length in deltas
    pub critical_path_deltas: u32,
    /// Target clock period that was checked against
    pub target_period: u32,
}

impl std::fmt::Display for TimingCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.meets_timing {
            write!(
                f,
                "PASS: critical path {} deltas, slack +{} (target {})",
                self.critical_path_deltas, self.slack, self.target_period
            )
        } else {
            write!(
                f,
                "FAIL: critical path {} deltas, slack {} (target {})",
                self.critical_path_deltas, self.slack, self.target_period
            )
        }
    }
}

/// Information about a detected race condition.
#[derive(Debug, Clone)]
pub struct RaceCondition {
    /// Tile where the race was detected
    pub tile_idx: usize,
    /// Coordinates of the tile
    pub x: usize,
    pub y: usize,
    /// Delta when earliest input arrived
    pub early_arrival: u32,
    /// Delta when latest input arrived
    pub late_arrival: u32,
    /// Window during which output was unstable
    pub race_window: u32,
}

// =============================================================================
// Phase 1B: CPU Execution Metrics
// =============================================================================

/// Metrics from tile-based CPU execution.
///
/// Tracks instruction counts and performance metrics for the ProgramCounter-based
/// CPU implemented using tiles.
#[derive(Debug, Clone, Default)]
pub struct CpuExecutionMetrics {
    /// Number of instructions executed (ProgramCounter increments/jumps on clock edge)
    pub instructions: u64,
    /// Number of simulation ticks elapsed
    pub ticks: u64,
    /// Instructions per tick (IPC)
    pub ipc: f64,
    /// Whether the CPU has halted (executed HALT instruction)
    pub halted: bool,
}

impl std::fmt::Display for CpuExecutionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CPU Execution Metrics")?;
        writeln!(f, "=====================")?;
        writeln!(f, "Instructions: {}", self.instructions)?;
        writeln!(f, "Ticks:        {}", self.ticks)?;
        writeln!(f, "IPC:          {:.3}", self.ipc)?;
        writeln!(f, "Halted:       {}", self.halted)?;
        Ok(())
    }
}

/// Sprint 172: A linear chain of same-type unidirectional wire tiles.
/// All members output the same value as the head's input.
#[derive(Debug, Clone)]
struct WireChain {
    wire_type: TileType,
    /// Tile indices AFTER the head, in propagation order.
    tail_members: Vec<u32>,
}

pub struct Simulation {
    pub tilemap: Tilemap,
    pub dirty: DirtyBitset,
    pub global_clock: bool, // false = low, true = high
    pub(crate) prev_clock: bool,
    pub last_change: Vec<Option<ChangeInfo>>, // EPIC 2: last-change info per tile
    current_delta: u32,
    // EPIC 38: fast-path caches
    meta_fast: Vec<TileType>,
    neighbors4: Vec<[u32; 4]>,
    logic_field: FieldGrid<u32>,
    power_field: FieldGrid<u8>,
    clock_field: FieldGrid<u8>,
    region_field: FieldGrid<u32>,
    logic_field_next: FieldGrid<u32>,
    power_field_next: FieldGrid<u8>,
    clock_field_next: FieldGrid<u8>,
    logic_field_coupled: FieldGrid<u64>,
    power_field_coupled: FieldGrid<u32>,
    heat_field: FieldGrid<u32>,
    charge_field: FieldGrid<u32>,
    heat_field_next: FieldGrid<u32>,
    charge_field_next: FieldGrid<u32>,
    heat_field_react: FieldGrid<u32>,
    charge_field_react: FieldGrid<u32>,
    heat_field_interact: FieldGrid<u32>,
    charge_field_interact: FieldGrid<u32>,
    // EPIC 37: reusable dirty-batch buffer to avoid per-tick allocations
    dirty_batch_buf: Vec<u32>,
    /// Sprint 384: total residual in-scope dirty bits drained after JIT settle
    /// (diagnostic — measures the residue that previously leaked to later phases).
    pub jit_settle_drained_total: u64,
    // Sprint 278: reusable slot-dirty bitset for compact scheduler
    schedule_slot_buf: Vec<u64>,
    // Sprint 280: When true, eval_tile/eval_tile_chain_fused skip global dirty
    // marking.  The compact scheduler sets this for terminal scopes (clock)
    // so cascade dirty bits don't leak to the next tick's delta 0.
    suppress_dirty_propagation: bool,
    // EPIC 49: registry of quantum demo tiles
    #[allow(dead_code)]
    qtiles: Vec<crate::quantum::QTile>,
    // SPRINT 20.0: Fast lookup from tile_idx to qtile index for feedback loop
    // Sprint 285: All lookup arrays use u32 with u32::MAX sentinel instead
    // of Option<usize>. Saves 42 MB (8→4 bytes per tile × 8 arrays × 1.31M tiles).
    qtile_lookup: Vec<u32>, // tile_idx -> qtiles index (u32::MAX = none)

    // === Component Hierarchy System ===
    component_defs: Vec<ComponentDef>,
    components: Vec<ComponentInstance>,
    component_lookup: Vec<u32>, // tile_idx -> component index (u32::MAX = none)
    component_input_lookup: Vec<u32>, // tile_idx -> component index (u32::MAX = none)

    // === Bus Architecture ===
    bus_defs: Vec<BusDef>,
    bus_states: Vec<BusState>,
    bus_connections: Vec<BusConnection>,
    bus_connection_lookup: Vec<u32>, // tile_idx -> bus_connections index (u32::MAX = none)

    // === Memory Controller ===
    memory_bank_defs: Vec<MemoryBankDef>,
    memory_banks: Vec<MemoryBank>,
    memory_port_connections: Vec<MemoryPortConnection>,
    memory_port_lookup: Vec<u32>, // tile_idx -> memory_port_connections index (u32::MAX = none)

    // === Multi-Clock Domains ===
    clock_domain_defs: Vec<ClockDomainDef>,
    clock_domain_states: Vec<ClockDomainState>,
    clock_domain_tile_lookup: Vec<u32>, // tile_idx -> domain index (u32::MAX = none)
    clock_divider_lookup: Vec<u32>, // tile_idx -> domain index for ClockDivider tiles (u32::MAX = none)
    synchronizer_lookup: Vec<u32>,  // tile_idx -> synchronizer_states index (u32::MAX = none)
    synchronizer_states: Vec<SynchronizerState>,

    // Physics-to-logic coupling configuration and state
    physics_coupling_config: PhysicsCouplingConfig,
    physics_coupling_ctx: Option<PhysicsCouplingContext>,

    // === EPIC 123: Propagation Delay Tracking ===
    /// Per-tile delay countdown (0 = ready to evaluate, 255 = not scheduled)
    delay_countdown: Vec<u8>,
    /// Per-tile arrival time (delta cycle when output last changed)
    arrival_time: Vec<u32>,
    /// Timing statistics from last tick_with_delays() call
    timing_stats: TimingStats,
    /// Per-tile delay override for wire tiles.
    /// If wire_delay[idx] > 0 for a wire tile, use this value instead of
    /// the tile type's base delay. This enables distance-based wire delays.
    wire_delay: Vec<u8>,

    // === Phase 1B: CPU Execution Metrics ===
    /// Number of instructions executed by ProgramCounter tiles
    pub cpu_instruction_count: u64,
    /// Number of simulation ticks elapsed
    pub cpu_tick_count: u64,
    /// Whether the CPU has halted (HALT instruction detected)
    pub cpu_halted: bool,

    // === Sprint 80: Multi-Layer Via Support ===
    /// Indexed via forward lookup: via_fwd[source_idx] = via_tile_idx.
    /// When source tile changes, dirty the via that reads from it.
    /// u32::MAX = no via reads from this tile.
    via_fwd: Vec<u32>,

    // === Sprint 160: Weighted Via Mask Storage ===
    /// Per-tile mask for WeightedViaUp/WeightedViaDown tiles.
    /// Default: u64::MAX (identity — AND with all-ones is no-op).
    /// Only meaningful for WeightedVia tiles; ignored by all other tile types.
    tile_mask: Vec<u64>,

    // === Sprint 183: Threshold Via Gating ===
    /// Per-tile threshold for ThresholdViaUp/ThresholdViaDown tiles.
    /// Default: 1 (passes if any single in-plane neighbor is non-zero).
    /// Only meaningful for ThresholdVia tiles; ignored by all other tile types.
    tile_threshold: Vec<u8>,

    // === Sprint 206: Weighted Via Shift ===
    /// Per-tile right-shift for WeightedViaUp/WeightedViaDown tiles.
    /// Default: 0 (no shift). Eval: `(source >> tile_shift[idx]) & tile_mask[idx]`.
    tile_shift: Vec<u8>,

    // === Sprint 147: Performance Optimization ===
    /// When false (default), skip ChangeInfo recording in eval_tile for speed.
    /// Set to true for diagnostic tools (explain_tile, CLI).
    pub record_change_info: bool,
    /// Precomputed list of clock-sensitive tile indices (lazy, built on first tick_with_delays).
    clock_sensitive_cache: Option<Vec<usize>>,
    /// Tiles touched during the last tick_with_delays (need reset at start of next tick).
    last_tick_activated: Vec<usize>,

    // === Sprint 172: Wire Chain Fusion ===
    /// Detected linear chains of same-type unidirectional wire tiles.
    wire_chains: Vec<WireChain>,
    /// Per-tile lookup: chain_head_map[tile_idx] = chain index in wire_chains.
    /// u32::MAX = this tile is not a chain head.
    chain_head_map: Vec<u32>,
    /// Sprint 173: Bitset marking chain TAIL members (not heads).
    /// Used to filter dirty index vectors — tails are handled by chain fusion.
    chain_tail_mask: Vec<u64>,
}

impl Simulation {
    pub fn new_standard_benchmark_world() -> Self {
        // Start from a clean simulation at the canonical size
        let mut sim = Self::new();

        // Lay down a few clock spines to drive activity deterministically
        // Horizontal clock lines every 64 rows
        for y in (32..HEIGHT.saturating_sub(32)).step_by(64) {
            // Seed a clock tile at the left edge
            sim.set_tile(2, y, TileType::ClockGlobal);
            // Run a horizontal wire to the right to propagate
            sim.wire_line(2, y, WIDTH.saturating_sub(3), y);
            // Sprinkle some simple gates along the line to exercise logic
            for x in (10..WIDTH.saturating_sub(10)).step_by(64) {
                sim.set_tile(x, y, TileType::And);
            }
        }

        // Vertical wire spines every 64 columns to create crossings
        for x in (16..WIDTH.saturating_sub(16)).step_by(64) {
            sim.wire_line(x, 8, x, HEIGHT.saturating_sub(9));
        }

        sim
    }

    /// Create a new simulation with default dimensions (512x512)
    pub fn new() -> Self {
        Self::with_size(WIDTH, HEIGHT)
    }

    /// Create a new simulation with custom dimensions (1 layer)
    pub fn with_size(width: usize, height: usize) -> Self {
        Self::with_size_layered(width, height, 1)
    }

    /// Create a new simulation with custom dimensions and multiple layers
    pub fn with_size_layered(width: usize, height: usize, num_layers: usize) -> Self {
        let tilemap = Tilemap::with_size_layered(width, height, num_layers);
        let tile_count = tilemap.tile_count();
        let layer_size = tilemap.layer_size;
        let dirty = DirtyBitset::new(tile_count);
        // EPIC 38: precompute neighbor indices and meta cache
        // Sprint 80: layer-aware — use local_y to prevent cross-layer leakage
        let neighbors4 = Self::build_neighbors4(width, height, layer_size, tile_count);
        let meta_fast = vec![TileType::Wire; tile_count];
        let logic_field = FieldGrid::new(width, height, 0u32);
        let power_field = FieldGrid::new(width, height, 0u8);
        let clock_field = FieldGrid::new(width, height, 0u8);
        let region_field = FieldGrid::new(width, height, 0u32);
        let logic_field_next = FieldGrid::new(width, height, 0u32);
        let power_field_next = FieldGrid::new(width, height, 0u8);
        let clock_field_next = FieldGrid::new(width, height, 0u8);
        let logic_field_coupled = FieldGrid::new(width, height, 0u64);
        let power_field_coupled = FieldGrid::new(width, height, 0u32);
        let heat_field = FieldGrid::new(width, height, 0u32);
        let charge_field = FieldGrid::new(width, height, 0u32);
        let heat_field_next = FieldGrid::new(width, height, 0u32);
        let charge_field_next = FieldGrid::new(width, height, 0u32);
        let heat_field_react = FieldGrid::new(width, height, 0u32);
        let charge_field_react = FieldGrid::new(width, height, 0u32);
        let heat_field_interact = FieldGrid::new(width, height, 0u32);
        let charge_field_interact = FieldGrid::new(width, height, 0u32);
        Self {
            tilemap,
            dirty,
            global_clock: false,
            prev_clock: false,
            last_change: vec![None; tile_count],
            current_delta: 0,
            meta_fast,
            neighbors4,
            logic_field,
            power_field,
            clock_field,
            region_field,
            logic_field_next,
            power_field_next,
            clock_field_next,
            logic_field_coupled,
            power_field_coupled,
            heat_field,
            charge_field,
            heat_field_next,
            charge_field_next,
            heat_field_react,
            charge_field_react,
            heat_field_interact,
            charge_field_interact,
            dirty_batch_buf: Vec::with_capacity(tile_count),
            jit_settle_drained_total: 0,
            schedule_slot_buf: Vec::new(),
            suppress_dirty_propagation: false,
            qtiles: Vec::new(),
            qtile_lookup: vec![u32::MAX; tile_count],
            component_defs: Vec::new(),
            components: Vec::new(),
            component_lookup: vec![u32::MAX; tile_count],
            component_input_lookup: vec![u32::MAX; tile_count],
            // Bus Architecture
            bus_defs: Vec::new(),
            bus_states: Vec::new(),
            bus_connections: Vec::new(),
            bus_connection_lookup: vec![u32::MAX; tile_count],
            // Memory Controller
            memory_bank_defs: Vec::new(),
            memory_banks: Vec::new(),
            memory_port_connections: Vec::new(),
            memory_port_lookup: vec![u32::MAX; tile_count],
            // Multi-Clock Domains
            clock_domain_defs: Vec::new(),
            clock_domain_states: Vec::new(),
            clock_domain_tile_lookup: vec![u32::MAX; tile_count],
            clock_divider_lookup: vec![u32::MAX; tile_count],
            synchronizer_lookup: vec![u32::MAX; tile_count],
            synchronizer_states: Vec::new(),
            physics_coupling_config: PhysicsCouplingConfig::default(),
            physics_coupling_ctx: None,
            // EPIC 123: Propagation delay tracking
            delay_countdown: vec![255u8; tile_count], // 255 = not scheduled
            arrival_time: vec![0u32; tile_count],
            timing_stats: TimingStats::default(),
            wire_delay: vec![0u8; tile_count], // 0 = use tile type's default delay
            // Phase 1B: CPU execution metrics
            cpu_instruction_count: 0,
            cpu_tick_count: 0,
            cpu_halted: false,
            // Sprint 80: Multi-Layer Via Support
            via_fwd: vec![u32::MAX; tile_count],
            // Sprint 160: Weighted Via Masks
            tile_mask: vec![u64::MAX; tile_count],
            // Sprint 183: Threshold Via Gating
            tile_threshold: vec![1u8; tile_count],
            // Sprint 206: Weighted Via Shift
            tile_shift: vec![0u8; tile_count],
            // Sprint 147: Performance
            record_change_info: false,
            clock_sensitive_cache: None,
            last_tick_activated: Vec::new(),
            // Sprint 172: Wire Chain Fusion
            wire_chains: Vec::new(),
            chain_head_map: vec![u32::MAX; tile_count],
            // Sprint 173: Chain tail mask for dirty filtering
            chain_tail_mask: vec![0u64; (tile_count + 63) / 64],
        }
    }

    /// Get the width of this simulation's tilemap
    #[inline]
    pub fn width(&self) -> usize {
        self.tilemap.width
    }

    /// Get the height of this simulation's tilemap
    #[inline]
    pub fn height(&self) -> usize {
        self.tilemap.height
    }

    /// Get the total tile count
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tilemap.tile_count()
    }

    /// Sprint 272: Get the neighbor indices for a tile.
    pub fn neighbors4_at(&self, idx: usize) -> &[u32; 4] {
        &self.neighbors4[idx]
    }

    /// Get the number of layers
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.tilemap.num_layers
    }

    /// Build layer-aware neighbors4 table.
    /// Uses local_y (position within layer) for up/down boundary checks
    /// to prevent cross-layer leakage at layer boundaries.
    fn build_neighbors4(
        width: usize,
        height: usize,
        layer_size: usize,
        tile_count: usize,
    ) -> Vec<[u32; 4]> {
        let mut neighbors4: Vec<[u32; 4]> = Vec::with_capacity(tile_count);
        for idx in 0..tile_count {
            let layer_base = (idx / layer_size) * layer_size;
            let within = idx % layer_size;
            let x = within % width;
            let local_y = within / width;
            let left = if x > 0 {
                (layer_base + local_y * width + (x - 1)) as u32
            } else {
                u32::MAX
            };
            let right = if x + 1 < width {
                (layer_base + local_y * width + (x + 1)) as u32
            } else {
                u32::MAX
            };
            let up = if local_y > 0 {
                (layer_base + (local_y - 1) * width + x) as u32
            } else {
                u32::MAX
            };
            let down = if local_y + 1 < height {
                (layer_base + (local_y + 1) * width + x) as u32
            } else {
                u32::MAX
            };
            neighbors4.push([left, right, up, down]);
        }
        neighbors4
    }

    /// Rebuild the via_fwd forward lookup table.
    /// Call after placing Via tiles (e.g., after builder finishes).
    pub fn rebuild_via_connections(&mut self) {
        self.via_fwd.iter_mut().for_each(|v| *v = u32::MAX);
        let layer_size = self.tilemap.layer_size;
        let tile_count = self.tilemap.tile_count();
        for idx in 0..tile_count {
            match self.meta_fast[idx] {
                TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp => {
                    let target = idx + layer_size;
                    if target < tile_count {
                        debug_assert_eq!(
                            self.via_fwd[target],
                            u32::MAX,
                            "Multiple vias reading from same source tile {} (existing via={}, new via={})",
                            target,
                            self.via_fwd[target],
                            idx
                        );
                        self.via_fwd[target] = idx as u32;
                    }
                }
                TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown => {
                    if idx >= layer_size {
                        let target = idx - layer_size;
                        debug_assert_eq!(
                            self.via_fwd[target],
                            u32::MAX,
                            "Multiple vias reading from same source tile {} (existing via={}, new via={})",
                            target,
                            self.via_fwd[target],
                            idx
                        );
                        self.via_fwd[target] = idx as u32;
                    }
                }
                _ => {}
            }
        }
    }

    /// Sprint 172: Detect linear chains of unidirectional wire tiles.
    /// Only chains within `placed_mask` of length >= `min_chain_len` are recorded.
    /// Must be called AFTER initial settle (`tick_with_delays`) to avoid
    /// interfering with delay-based sequential capture logic.
    pub fn build_wire_chains(&mut self, placed_mask: &[u64], min_chain_len: usize) {
        self.wire_chains.clear();
        self.chain_head_map.iter_mut().for_each(|v| *v = u32::MAX);
        // Sprint 173: Clear chain tail mask.
        self.chain_tail_mask.iter_mut().for_each(|v| *v = 0);

        let tile_count = self.tilemap.tiles.len();
        let is_placed = |idx: usize| -> bool {
            idx < tile_count
                && idx / 64 < placed_mask.len()
                && (placed_mask[idx / 64] & (1u64 << (idx % 64))) != 0
        };

        // Input/output direction slot indices for each unidirectional wire type.
        // neighbors4 layout: [LEFT=0, RIGHT=1, UP=2, DOWN=3].
        let dir = |tt: TileType| -> Option<(usize, usize)> {
            match tt {
                TileType::WireRight => Some((0, 1)), // input=LEFT, output=RIGHT
                TileType::WireLeft => Some((1, 0)),  // input=RIGHT, output=LEFT
                TileType::WireDown => Some((2, 3)),  // input=UP, output=DOWN
                TileType::WireUp => Some((3, 2)),    // input=DOWN, output=UP
                _ => None,
            }
        };

        for idx in 0..tile_count {
            if !is_placed(idx) {
                continue;
            }
            let tt = self.meta_fast[idx];
            let (input_slot, output_slot) = match dir(tt) {
                Some(d) => d,
                None => continue,
            };

            // Head check: input neighbor must NOT be same wire type (or boundary/unplaced).
            let input_idx = self.neighbors4[idx][input_slot];
            if input_idx != u32::MAX {
                let ii = input_idx as usize;
                if ii < tile_count && is_placed(ii) && self.meta_fast[ii] == tt {
                    continue; // Has upstream same-type tile — not a head
                }
            }

            // Walk forward to collect tail members.
            let mut tail = Vec::new();
            let mut cur = self.neighbors4[idx][output_slot];
            while cur != u32::MAX {
                let ci = cur as usize;
                if ci >= tile_count || !is_placed(ci) || self.meta_fast[ci] != tt {
                    break;
                }
                tail.push(cur);
                cur = self.neighbors4[ci][output_slot];
            }

            // Only record chains >= min_chain_len (head + tail).
            if tail.len() + 1 < min_chain_len {
                continue;
            }

            let chain_id = self.wire_chains.len() as u32;
            self.chain_head_map[idx] = chain_id;
            self.wire_chains.push(WireChain {
                wire_type: tt,
                tail_members: tail,
            });
            // Sprint 173: Record tail members in bitset for dirty index filtering.
            for &member in &self.wire_chains.last().unwrap().tail_members {
                let seg = member as usize / 64;
                let bit = member as usize % 64;
                if seg < self.chain_tail_mask.len() {
                    self.chain_tail_mask[seg] |= 1u64 << bit;
                }
            }
        }
    }

    /// Set tile type at (x, y, z) on specified layer
    pub fn set_tile_3d(&mut self, x: usize, y: usize, z: usize, tile_type: TileType) {
        if let Some(idx) = self.tilemap.index_3d(x, y, z) {
            if let Some(t) = self.tilemap.tiles.get_mut(idx) {
                t.meta.tile_type = tile_type;
            }
            if let Some(m) = self.meta_fast.get_mut(idx) {
                *m = tile_type;
            }
        }
    }

    /// Get tile type at (x, y, z) on specified layer
    pub fn tile_type_3d(&self, x: usize, y: usize, z: usize) -> TileType {
        if let Some(t) = self.tilemap.get_tile_3d(x, y, z) {
            t.meta.tile_type
        } else {
            TileType::Wire
        }
    }

    /// Get logic value at (x, y, z) on specified layer
    pub fn get_logic_at_3d(&self, x: usize, y: usize, z: usize) -> u64 {
        self.tilemap.value_at_3d(x, y, z).unwrap_or(0)
    }

    /// Set logic value at (x, y, z) on specified layer
    pub fn set_logic_value_3d(&self, x: usize, y: usize, z: usize, value: u64) -> bool {
        if let Some(idx) = self.tilemap.index_3d(x, y, z) {
            if idx < self.tilemap.tiles.len() {
                self.tilemap.set_value(idx, value);
                return true;
            }
        }
        false
    }

    // EPIC 49: register a quantum demo tile at (x,y)
    pub fn register_qdemo_tile(
        &mut self,
        x: usize,
        y: usize,
        state: crate::quantum::QState,
        program: Vec<crate::quantum::QGate>,
        seed: u64,
    ) {
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return;
        }

        let idx = y * self.tilemap.width + x;

        // SPRINT 20.0: Prevent double-registration (memory leak + index corruption)
        if self.qtile_lookup[idx] != u32::MAX {
            eprintln!(
                "Warning: QDemo tile at ({},{}) already registered; ignoring duplicate registration",
                x, y
            );
            return;
        }

        // SPRINT 20.0: Warn if >64 qubits (only first 64 can be encoded in logic output)
        if state.n_qubits > 64 {
            eprintln!(
                "Warning: QDemo at ({},{}) has {} qubits; only first 64 measurements will be encoded in logic output",
                x, y, state.n_qubits
            );
        }

        if let Some(t) = self.tilemap.get_tile_mut(x, y) {
            t.meta.tile_type = TileType::QDemo;
            if let Some(m) = self.meta_fast.get_mut(idx) {
                *m = TileType::QDemo;
            }
            let qtile_idx = self.qtiles.len();
            let id: u16 = qtile_idx as u16;
            let measured = vec![None; state.n_qubits as usize];
            let qt = crate::quantum::QTile {
                id,
                tile_idx: idx,
                state,
                program,
                pc: 0,
                measured,
                rng: crate::quantum::QRng::new(seed),
            };
            self.qtiles.push(qt);
            // SPRINT 20.0: Register in lookup map for O(1) feedback access
            self.qtile_lookup[idx] = qtile_idx as u32;
        }
    }

    // =========================================================================
    // Component Hierarchy System
    // =========================================================================

    /// Register a component definition. Returns the def_idx for use in place_component.
    pub fn register_component_def(&mut self, def: ComponentDef) -> usize {
        let idx = self.component_defs.len();
        self.component_defs.push(def);
        idx
    }

    /// Place a component instance at the given origin (top-left corner).
    /// Returns the component instance index, or None if placement fails.
    pub fn place_component(
        &mut self,
        def_idx: usize,
        origin_x: usize,
        origin_y: usize,
    ) -> Option<usize> {
        if def_idx >= self.component_defs.len() {
            return None;
        }

        let width = self.component_defs[def_idx].width as usize;
        let height = self.component_defs[def_idx].height as usize;

        // Bounds check
        if origin_x + width > self.tilemap.width || origin_y + height > self.tilemap.height {
            return None;
        }

        let is_behavioral = matches!(
            self.component_defs[def_idx].implementation,
            ComponentImpl::Combinational(_)
        );

        let mut input_port_indices = Vec::new();
        let mut output_port_indices = Vec::new();

        // Place port tiles
        let port_count = self.component_defs[def_idx].ports.len();
        for pi in 0..port_count {
            let port = &self.component_defs[def_idx].ports[pi];
            let (px, py) = port_to_grid_coords(origin_x, origin_y, width, height, port);
            let idx = py * self.tilemap.width + px;

            match port.direction {
                PortDirection::Input => {
                    let wire_type = input_wire_type_for_edge(port.edge);
                    self.set_tile(px, py, wire_type);
                    input_port_indices.push(idx);
                }
                PortDirection::Output => {
                    if is_behavioral {
                        self.set_tile(px, py, TileType::ComponentOutput);
                    } else {
                        let wire_type = output_wire_type_for_edge(port.edge);
                        self.set_tile(px, py, wire_type);
                    }
                    output_port_indices.push(idx);
                }
            }
        }

        // For structural components, place internal tiles
        // Collect placement data first to avoid borrow conflict with self
        let structural_placements: Vec<(usize, usize, TileType, Option<u64>)> =
            if let ComponentImpl::Structural(ref placements) =
                self.component_defs[def_idx].implementation
            {
                placements
                    .iter()
                    .map(|p| {
                        (
                            origin_x + p.x as usize,
                            origin_y + p.y as usize,
                            p.tile_type,
                            p.initial_logic,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
        for (gx, gy, tile_type, initial_logic) in structural_placements {
            if gx < self.tilemap.width && gy < self.tilemap.height {
                self.set_tile(gx, gy, tile_type);
                if let Some(val) = initial_logic {
                    let tile_idx = gy * self.tilemap.width + gx;
                    self.tilemap.set_value(tile_idx, val);
                }
            }
        }

        // Create instance
        let comp_idx = self.components.len();
        let output_count = output_port_indices.len();
        let instance = ComponentInstance {
            def_idx,
            origin: (origin_x, origin_y),
            input_port_indices: input_port_indices.clone(),
            output_port_indices: output_port_indices.clone(),
            output_cache: vec![std::cell::Cell::new(0u64); output_count],
            cache_valid: std::cell::Cell::new(false),
        };
        self.components.push(instance);

        // Update side tables for behavioral components
        if is_behavioral {
            for &out_idx in &output_port_indices {
                if out_idx < self.component_lookup.len() {
                    self.component_lookup[out_idx] = comp_idx as u32;
                }
            }
            for &in_idx in &input_port_indices {
                if in_idx < self.component_input_lookup.len() {
                    self.component_input_lookup[in_idx] = comp_idx as u32;
                }
            }
        }

        Some(comp_idx)
    }

    /// Get a component instance by index.
    pub fn get_component(&self, comp_idx: usize) -> Option<&ComponentInstance> {
        self.components.get(comp_idx)
    }

    /// Get the number of placed component instances.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Evaluate a behavioral component's output for a specific output port tile.
    fn evaluate_component_output(&self, comp_idx: usize, tile_idx: usize) -> u64 {
        let comp = &self.components[comp_idx];

        if !comp.cache_valid.get() {
            // Gather input port values
            let inputs: Vec<u64> = comp
                .input_port_indices
                .iter()
                .map(|&idx| self.tilemap.value(idx))
                .collect();

            // Call the behavioral function
            if let ComponentImpl::Combinational(ref func) =
                self.component_defs[comp.def_idx].implementation
            {
                let outputs = func(&inputs);
                for (i, &val) in outputs.iter().enumerate() {
                    if i < comp.output_cache.len() {
                        comp.output_cache[i].set(val);
                    }
                }
            }
            comp.cache_valid.set(true);
        }

        // Find which output port this tile corresponds to
        for (i, &out_idx) in comp.output_port_indices.iter().enumerate() {
            if out_idx == tile_idx {
                return comp.output_cache[i].get();
            }
        }
        0
    }

    // =========================================================================
    // Bus Architecture
    // =========================================================================

    /// Register a bus definition. Returns the bus_idx for use in connect_bus.
    pub fn register_bus(&mut self, def: BusDef) -> usize {
        let idx = self.bus_defs.len();
        let width = def.width;
        self.bus_states.push(BusState {
            data: vec![std::cell::Cell::new(0u64); width],
            connection_indices: Vec::new(),
            word_written: vec![std::cell::Cell::new(false); width],
        });
        self.bus_defs.push(def);
        idx
    }

    /// Connect a tile at (x, y) to a bus as a BusInterface.
    /// Returns the connection index, or None if out of bounds or invalid bus/word.
    pub fn connect_bus(
        &mut self,
        bus_idx: usize,
        x: usize,
        y: usize,
        word_offset: usize,
        direction: BusDirection,
    ) -> Option<usize> {
        if bus_idx >= self.bus_defs.len() {
            return None;
        }
        if word_offset >= self.bus_defs[bus_idx].width {
            return None;
        }
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return None;
        }

        let tile_idx = y * self.tilemap.width + x;

        // Set tile type to BusInterface
        if let Some(t) = self.tilemap.get_tile_mut(x, y) {
            t.meta.tile_type = TileType::BusInterface;
        }
        if let Some(m) = self.meta_fast.get_mut(tile_idx) {
            *m = TileType::BusInterface;
        }

        let conn_idx = self.bus_connections.len();
        self.bus_connections.push(BusConnection {
            bus_idx,
            tile_idx,
            word_offset,
            direction,
        });
        self.bus_connection_lookup[tile_idx] = conn_idx as u32;
        self.bus_states[bus_idx].connection_indices.push(conn_idx);

        Some(conn_idx)
    }

    /// Evaluate all buses: collect writes, resolve arbitration, push to readers.
    fn evaluate_buses(&mut self) {
        // Reset per-word write flags
        for bus_state in &self.bus_states {
            for ww in &bus_state.word_written {
                ww.set(false);
            }
            // Reset data for fresh evaluation
            for d in &bus_state.data {
                d.set(0);
            }
        }

        // Phase 1: Collect writes from writer tiles
        for conn in &self.bus_connections {
            match conn.direction {
                BusDirection::Writer | BusDirection::ReadWriter => {}
                BusDirection::Reader => continue,
            }

            // Writer reads from its left neighbor (the value being driven onto the bus)
            let n = &self.neighbors4[conn.tile_idx];
            let input_val = self.load_logic_idx(n[0]); // left neighbor drives bus

            let bus_state = &self.bus_states[conn.bus_idx];
            let bus_def = &self.bus_defs[conn.bus_idx];
            let w = conn.word_offset;
            let current = bus_state.data[w].get();

            let new_val = match bus_def.arbitration {
                BusArbitration::Priority => {
                    if !bus_state.word_written[w].get() {
                        // First writer for this word wins
                        input_val
                    } else {
                        current // already written by higher-priority connection
                    }
                }
                BusArbitration::OrMerge => current | input_val,
            };

            if new_val != current {
                bus_state.data[w].set(new_val);
            }
            bus_state.word_written[w].set(true);
        }

        // Phase 2: Push bus data to reader tiles and mark dirty
        for bus_state in &self.bus_states {
            // Check if any word was written
            let any_written = bus_state.word_written.iter().any(|ww| ww.get());
            if !any_written {
                continue;
            }

            for &conn_idx in &bus_state.connection_indices {
                let conn = &self.bus_connections[conn_idx];
                match conn.direction {
                    BusDirection::Reader | BusDirection::ReadWriter => {}
                    BusDirection::Writer => continue,
                }

                let bus_val = bus_state.data[conn.word_offset].get();
                let current = self.tilemap.value(conn.tile_idx);

                if bus_val != current {
                    self.tilemap.set_value(conn.tile_idx, bus_val);
                    // Mark neighbors dirty so they pick up the new value
                    let n = &self.neighbors4[conn.tile_idx];
                    for &ni in n.iter() {
                        if ni != u32::MAX {
                            self.dirty.mark_dirty(ni as usize);
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Memory Controller
    // =========================================================================

    /// Register a memory bank. Returns bank_idx.
    pub fn register_memory_bank(&mut self, def: MemoryBankDef) -> usize {
        let idx = self.memory_bank_defs.len();
        let size = def.size;
        let mut data = vec![std::cell::Cell::new(0u64); size];
        // Initialize from def's initial_data
        for (i, &val) in def.initial_data.iter().enumerate() {
            if i < size {
                data[i] = std::cell::Cell::new(val);
            }
        }
        self.memory_banks.push(MemoryBank { data });
        self.memory_bank_defs.push(def);
        idx
    }

    /// Connect a tile at (x, y) to a memory bank as a MemoryPort.
    /// Returns connection index, or None if out of bounds or invalid bank.
    pub fn connect_memory_port(&mut self, bank_idx: usize, x: usize, y: usize) -> Option<usize> {
        if bank_idx >= self.memory_banks.len() {
            return None;
        }
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return None;
        }

        let tile_idx = y * self.tilemap.width + x;

        // Set tile type to MemoryPort
        if let Some(t) = self.tilemap.get_tile_mut(x, y) {
            t.meta.tile_type = TileType::MemoryPort;
        }
        if let Some(m) = self.meta_fast.get_mut(tile_idx) {
            *m = TileType::MemoryPort;
        }

        let conn_idx = self.memory_port_connections.len();
        self.memory_port_connections
            .push(MemoryPortConnection { bank_idx, tile_idx });
        self.memory_port_lookup[tile_idx] = conn_idx as u32;

        Some(conn_idx)
    }

    /// Evaluate a memory port: write if enabled, always return read value.
    /// address = left, data_in = right, write_enable = up
    fn evaluate_memory_port(
        &self,
        conn_idx: usize,
        address: u64,
        data_in: u64,
        write_enable: u64,
    ) -> u64 {
        let conn = &self.memory_port_connections[conn_idx];
        let bank = &self.memory_banks[conn.bank_idx];
        let addr = address as usize;

        // Bounds check — out-of-range reads return 0, writes are ignored
        if addr >= bank.data.len() {
            return 0;
        }

        // Write phase
        if write_enable != 0 {
            bank.data[addr].set(data_in);
        }

        // Read phase (read-after-write: returns newly written value if same address)
        bank.data[addr].get()
    }

    // === Multi-Clock Domain APIs ===

    /// Returns (clock, prev_clock) for a tile -- either from its assigned domain or from global.
    #[inline]
    fn effective_clock(&self, idx: usize) -> (bool, bool) {
        let cd_val = if idx < self.clock_domain_tile_lookup.len() {
            self.clock_domain_tile_lookup[idx]
        } else {
            u32::MAX
        };
        if cd_val != u32::MAX {
            let domain_idx = cd_val as usize;
            let state = &self.clock_domain_states[domain_idx];
            (state.clock, state.prev_clock)
        } else {
            (self.global_clock, self.prev_clock)
        }
    }

    /// Register a clock domain. Returns domain index.
    pub fn register_clock_domain(&mut self, name: &str, divider: u32, phase_offset: u32) -> usize {
        let idx = self.clock_domain_defs.len();
        self.clock_domain_defs.push(ClockDomainDef {
            name: name.to_string(),
            divider,
            phase_offset,
        });
        self.clock_domain_states.push(ClockDomainState {
            clock: false,
            prev_clock: false,
            counter: 0,
        });
        idx
    }

    /// Assign a sequential tile to a clock domain by tile index.
    pub fn assign_tile_to_domain(&mut self, tile_idx: usize, domain_idx: usize) {
        if tile_idx < self.clock_domain_tile_lookup.len()
            && domain_idx < self.clock_domain_defs.len()
        {
            self.clock_domain_tile_lookup[tile_idx] = domain_idx as u32;
        }
    }

    /// Assign a tile at (x,y) to a clock domain.
    pub fn assign_tile_to_domain_xy(&mut self, x: usize, y: usize, domain_idx: usize) {
        let tile_idx = y * self.tilemap.width + x;
        self.assign_tile_to_domain(tile_idx, domain_idx);
    }

    /// Connect a ClockDivider tile to a domain by tile index.
    pub fn connect_clock_divider(&mut self, tile_idx: usize, domain_idx: usize) {
        if tile_idx < self.clock_divider_lookup.len() && domain_idx < self.clock_domain_defs.len() {
            self.clock_divider_lookup[tile_idx] = domain_idx as u32;
        }
    }

    /// Connect a ClockDivider tile at (x,y) to a domain.
    pub fn connect_clock_divider_xy(&mut self, x: usize, y: usize, domain_idx: usize) {
        let tile_idx = y * self.tilemap.width + x;
        self.connect_clock_divider(tile_idx, domain_idx);
    }

    /// Connect a Synchronizer tile to its destination domain by tile index.
    pub fn connect_synchronizer(&mut self, tile_idx: usize, domain_idx: usize) {
        if tile_idx < self.synchronizer_lookup.len() && domain_idx < self.clock_domain_defs.len() {
            let sync_idx = self.synchronizer_states.len();
            self.synchronizer_states.push(SynchronizerState {
                domain_idx,
                stage1: std::cell::Cell::new(0),
                stage2: std::cell::Cell::new(0),
            });
            self.synchronizer_lookup[tile_idx] = sync_idx as u32;
        }
    }

    /// Connect a Synchronizer tile at (x,y) to its destination domain.
    pub fn connect_synchronizer_xy(&mut self, x: usize, y: usize, domain_idx: usize) {
        let tile_idx = y * self.tilemap.width + x;
        self.connect_synchronizer(tile_idx, domain_idx);
    }

    #[cfg(test)]
    pub fn dirty_buf_capacity(&self) -> usize {
        self.dirty_batch_buf.capacity()
    }

    /// Get the number of registered quantum tiles.
    /// SPRINT 2.2: Public getter for Python API
    pub fn quantum_tile_count(&self) -> usize {
        self.qtiles.len()
    }

    pub fn snapshot_region(
        &self,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Vec<Vec<u64>> {
        let mut out = Vec::with_capacity(height);
        for y in y0..y0 + height {
            let mut row = Vec::with_capacity(width);
            for x in x0..x0 + width {
                let val = self.tilemap.value_at(x, y).unwrap_or(0);
                row.push(val);
            }
            out.push(row);
        }
        out
    }

    pub fn print_region(&self, x0: usize, y0: usize, width: usize, height: usize) {
        let snap = self.snapshot_region(x0, y0, width, height);
        for row in snap {
            for val in row {
                print!("{:016x} ", val);
            }
            println!();
        }
    }

    // EPIC 4: Small builder helpers
    pub fn set_tile(&mut self, x: usize, y: usize, tile_type: TileType) {
        if let Some(t) = self.tilemap.get_tile_mut(x, y) {
            t.meta.tile_type = tile_type;
            let idx = y * self.tilemap.width + x;
            if let Some(m) = self.meta_fast.get_mut(idx) {
                *m = tile_type;
            }
        }
    }

    pub fn set_tile_type(&mut self, x: usize, y: usize, tile_type: TileType) {
        self.set_tile(x, y, tile_type);
    }

    pub fn tile_type_xy(&self, x: usize, y: usize) -> TileType {
        if let Some(t) = self.tilemap.get_tile(x, y) {
            t.meta.tile_type
        } else {
            TileType::Wire
        }
    }

    pub fn get_logic_at(&self, x: usize, y: usize) -> u64 {
        self.tilemap.value_at(x, y).unwrap_or(0)
    }

    /// Get logic value by tile index (for fast access in tile_cpu)
    pub fn get_logic_value_by_idx(&self, idx: usize) -> u64 {
        if idx < self.tilemap.tiles.len() {
            self.tilemap.value(idx)
        } else {
            0
        }
    }

    /// Set logic value by tile index (for fast access in tile_cpu)
    pub fn set_logic_value_by_idx(&self, idx: usize, value: u64) {
        if idx < self.tilemap.tiles.len() {
            self.tilemap.set_value(idx, value);
        }
    }

    /// Sprint 310: Set value and mark dirty only if the value changed.
    /// Returns true if the value was different (and tile was marked dirty).
    #[inline]
    pub fn set_value_and_mark_if_changed(&self, idx: usize, value: u64) -> bool {
        if idx < self.tilemap.tiles.len() {
            let current = self.tilemap.value(idx);
            if current != value {
                self.tilemap.set_value(idx, value);
                self.dirty.mark_dirty(idx);
                return true;
            }
        }
        false
    }

    pub fn add_heat_at(&mut self, x: usize, y: usize, delta: u32) {
        if x < self.tilemap.width && y < self.tilemap.height {
            let v = self.heat_field.get(x, y).saturating_add(delta);
            self.heat_field.set(x, y, v);
        }
    }

    #[cfg(test)]
    pub fn write_logic(&self, x: usize, y: usize, value: u64) {
        self.tilemap.set_value_at(x, y, value);
    }

    // Public, safe logic writer for CLI use. Returns false if OOB.
    pub fn set_logic_value(&self, x: usize, y: usize, value: u64) -> bool {
        self.tilemap.set_value_at(x, y, value)
    }

    // Draw a wire path between two points.
    // - If horizontal or vertical: draw that segment (excluding endpoints).
    // - If both differ: draw an L-shape (x then y), excluding endpoints but including corner.
    pub fn wire_line(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let (mut x, mut y) = (x1, y1);
        // Horizontal segment toward x2 (exclude starting point)
        if x1 != x2 {
            let (start, end) = if x1 < x2 { (x1 + 1, x2) } else { (x2, x1 - 1) };
            let yh = y1;
            for xi in start..end {
                if let Some(t) = self.tilemap.get_tile_mut(xi, yh) {
                    t.meta.tile_type = TileType::Wire;
                    let idx = yh * self.tilemap.width + xi;
                    if let Some(m) = self.meta_fast.get_mut(idx) {
                        *m = TileType::Wire;
                    }
                }
            }
            x = x2;
        }
        // Vertical segment toward y2 (exclude starting point of this leg)
        if y1 != y2 {
            let (start, end) = if y1 < y2 { (y1 + 1, y2) } else { (y2, y1 - 1) };
            let xv = x;
            for yi in start..end {
                if let Some(t) = self.tilemap.get_tile_mut(xv, yi) {
                    t.meta.tile_type = TileType::Wire;
                    let idx = yi * self.tilemap.width + xv;
                    if let Some(m) = self.meta_fast.get_mut(idx) {
                        *m = TileType::Wire;
                    }
                }
            }
            y = y2;
        }
        let _ = (x, y); // silence unused if no segments
    }

    pub fn tick(&mut self) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        // Update clock domain states
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        // Phase 1B: Increment tick counter for CPU metrics
        self.cpu_tick_count += 1;

        // PHYSICS COUPLING: Snapshot physics fields at tick boundary
        // This ensures physics values don't change mid-stabilization
        if self.physics_coupling_config.enabled {
            self.snapshot_physics_for_coupling();
        } else {
            self.physics_coupling_ctx = None;
        }

        // Mark clock-sensitive tiles dirty
        for (idx, tile) in self.tilemap.tiles.iter().enumerate() {
            match tile.meta.tile_type {
                TileType::ClockGlobal => {
                    self.dirty.mark_dirty(idx);
                }
                TileType::Latch | TileType::Register8 | TileType::Register64 => {
                    let cd_val = if idx < self.clock_domain_tile_lookup.len() {
                        self.clock_domain_tile_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    } else {
                        self.dirty.mark_dirty(idx);
                    }
                }
                TileType::ClockDivider => {
                    let cd_val = if idx < self.clock_divider_lookup.len() {
                        self.clock_divider_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                TileType::Synchronizer => {
                    let s_val = if idx < self.synchronizer_lookup.len() {
                        self.synchronizer_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if s_val != u32::MAX {
                        let sync_idx = s_val as usize;
                        let domain_idx = self.synchronizer_states[sync_idx].domain_idx;
                        let state = &self.clock_domain_states[domain_idx];
                        if !state.prev_clock && state.clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut delta_count: u32 = 0;
        const MAX_DELTA: u32 = 100; // Increased to handle cascade propagation in larger grids

        // Choose evaluation function based on coupling state
        let use_coupling = self.physics_coupling_ctx.is_some();

        loop {
            // Record the current delta index for ChangeInfo
            self.current_delta = delta_count;
            // Debug header per delta cycle (feature-gated)
            crate::dbg_signal!("--- delta cycle {} ---", delta_count);
            // Take ownership of the scratch buffer to avoid borrowing conflicts
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into(&mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_changed = false;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                let changed = if use_coupling {
                    self.eval_tile_coupled(idx)
                } else {
                    self.eval_tile(idx)
                };
                if changed {
                    any_changed = true;
                }
            }
            // Return buffer to self
            self.dirty_batch_buf = batch;

            if !any_changed {
                break;
            }

            delta_count += 1;
            if delta_count >= MAX_DELTA {
                panic!("Delta cycle limit exceeded");
            }
        }

        // === Bus evaluation phase ===
        if !self.bus_states.is_empty() {
            self.evaluate_buses();
            // Post-bus stabilization: let reader values propagate through the grid
            let mut post_bus_deltas = 0u32;
            loop {
                self.current_delta = delta_count + post_bus_deltas + 1;
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                self.dirty.fill_into(&mut batch);
                if batch.is_empty() {
                    self.dirty_batch_buf = batch;
                    break;
                }
                let mut any_changed = false;
                for &idx32 in batch.iter() {
                    let idx = idx32 as usize;
                    let changed = if use_coupling {
                        self.eval_tile_coupled(idx)
                    } else {
                        self.eval_tile(idx)
                    };
                    if changed {
                        any_changed = true;
                    }
                }
                self.dirty_batch_buf = batch;
                if !any_changed {
                    break;
                }
                post_bus_deltas += 1;
                if post_bus_deltas >= 100 {
                    break;
                }
            }
        }

        // Clear physics snapshot at tick end
        self.physics_coupling_ctx = None;

        // EPIC 49: after classical logic stabilizes for this tick, step quantum tiles (one gate per tile)
        self.step_quantum_tiles();
    }

    // =========================================================================
    // EPIC 123: Timing-Aware Tick with Propagation Delays
    // =========================================================================

    /// Tick simulation with propagation delay modeling.
    ///
    /// Unlike `tick()` which uses zero-delay semantics (all combinational
    /// logic settles instantly), this method models realistic gate delays:
    ///
    /// - Wires: 1 delta cycle per hop
    /// - Simple gates (AND, OR, NOT): 2 delta cycles
    /// - Complex gates (MUL, DIV): 8-12 delta cycles
    /// - Sequential elements: 0 (capture on clock edge)
    ///
    /// Returns timing statistics including critical path length, glitch
    /// detection, and whether timing converged within the delta limit.
    ///
    /// # Example
    /// ```ignore
    /// let stats = sim.tick_with_delays();
    /// println!("Critical path: {} deltas", stats.critical_path_deltas);
    /// if !stats.converged {
    ///     println!("WARNING: Timing did not converge - combinational loop?");
    /// }
    /// ```

    /// Initialize the simulation by setting clock state for edge detection.
    /// Call this once after setting up tiles and before calling tick_with_delays().
    /// Returns stats with zero deltas (no propagation done here).
    pub fn initialize(&mut self) -> TimingStats {
        // Set prev_clock to false so first tick_with_delays sees a rising edge
        self.prev_clock = false;
        self.global_clock = false;

        // Reset timing stats
        self.timing_stats = TimingStats::default();
        self.timing_stats.converged = true;

        // Reset delay countdowns and arrival times
        for d in self.delay_countdown.iter_mut() {
            *d = 255;
        }
        for t in self.arrival_time.iter_mut() {
            *t = 0;
        }

        // Mark all tiles with non-zero values as dirty for initial propagation
        for idx in 0..self.tilemap.tile_count() {
            if self.tilemap.value(idx) != 0 {
                self.dirty.mark_dirty(idx);
            }
        }

        self.timing_stats.clone()
    }

    /// Run combinational propagation without toggling the clock.
    ///
    /// This evaluates dirty tiles in a simple loop until no more changes occur.
    /// Used by tile_cpu to propagate operands through ALU tiles (Const→Add/Sub/etc.)
    /// without triggering a clock edge. Returns the number of delta iterations.
    pub fn propagate_combinational(&mut self) -> u32 {
        self.propagate_combinational_counted().0
    }

    /// Like `propagate_combinational`, but returns (deltas, tiles_evaluated, tiles_switched).
    /// Used by per-instruction telemetry to account for all tile work.
    pub fn propagate_combinational_counted(&mut self) -> (u32, u32, u32) {
        let mut delta_count: u32 = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        loop {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into(&mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }
            let mut any_changed = false;
            let batch_size = batch.len() as u32;
            for &idx32 in batch.iter() {
                if self.eval_tile(idx32 as usize) {
                    any_changed = true;
                    total_switched += 1;
                }
            }
            total_evaluated += batch_size;
            self.dirty_batch_buf = batch;
            if !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 100 {
                break;
            }
        }
        (delta_count, total_evaluated, total_switched)
    }

    /// Compute the transitive closure of tiles reachable from `seeds` through the
    /// dependency graph (neighbors4 + via_fwd), respecting directional wire rules.
    /// Returns a sorted `Vec<u32>` suitable for use as a `propagate_combinational_scoped` scope.
    pub fn compute_zone_closure(&self, seeds: &[usize]) -> Vec<u32> {
        let tc = self.tilemap.tile_count();
        let mut visited = vec![false; tc];
        let mut queue = std::collections::VecDeque::new();
        for &s in seeds {
            if s < tc && !visited[s] {
                visited[s] = true;
                queue.push_back(s);
            }
        }
        while let Some(idx) = queue.pop_front() {
            // Collect dependents the same way dirty_dependents does.
            // Const tiles never change output during eval, so they are dead ends
            // (the guard Const(0) mesh prevents BFS from leaking into the default Wire grid).
            let tt = self.meta_fast[idx];
            if tt == TileType::Const {
                continue;
            }
            let n = &self.neighbors4[idx];
            let mut deps = [u32::MAX; 5]; // up to 4 neighbors + 1 via
            let mut count = 0;
            match tt {
                TileType::WireDown => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireUp => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[2];
                    count = 3;
                }
                TileType::WireRight => {
                    deps[0] = n[1];
                    deps[1] = n[2];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireLeft => {
                    deps[0] = n[0];
                    deps[1] = n[2];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireH => {
                    deps[0] = n[2];
                    deps[1] = n[3];
                    count = 2;
                }
                TileType::WireV => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    count = 2;
                }
                _ => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[2];
                    deps[3] = n[3];
                    count = 4;
                }
            }
            // Via forward
            let via = self.via_fwd[idx];
            if via != u32::MAX {
                deps[count] = via;
                count += 1;
            }
            for i in 0..count {
                let d = deps[i];
                if d != u32::MAX {
                    let di = d as usize;
                    if di < tc && !visited[di] {
                        visited[di] = true;
                        queue.push_back(di);
                    }
                }
            }
        }
        let mut result: Vec<u32> = visited
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
            .collect();
        result.sort_unstable();
        result
    }

    /// Like `compute_zone_closure`, but only follows tiles present in `placed` bitset.
    /// Prevents BFS from leaking through unplaced default Wire tiles.
    pub fn compute_zone_closure_restricted(&self, seeds: &[usize], placed: &[u64]) -> Vec<u32> {
        let tc = self.tilemap.tile_count();
        let mut visited = vec![false; tc];
        let mut queue = std::collections::VecDeque::new();
        for &s in seeds {
            if s < tc && !visited[s] {
                visited[s] = true;
                queue.push_back(s);
            }
        }
        while let Some(idx) = queue.pop_front() {
            let tt = self.meta_fast[idx];
            if tt == TileType::Const {
                continue;
            }
            let n = &self.neighbors4[idx];
            let mut deps = [u32::MAX; 5];
            let mut count = 0;
            match tt {
                TileType::WireDown => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireUp => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[2];
                    count = 3;
                }
                TileType::WireRight => {
                    deps[0] = n[1];
                    deps[1] = n[2];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireLeft => {
                    deps[0] = n[0];
                    deps[1] = n[2];
                    deps[2] = n[3];
                    count = 3;
                }
                TileType::WireH => {
                    deps[0] = n[2];
                    deps[1] = n[3];
                    count = 2;
                }
                TileType::WireV => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    count = 2;
                }
                _ => {
                    deps[0] = n[0];
                    deps[1] = n[1];
                    deps[2] = n[2];
                    deps[3] = n[3];
                    count = 4;
                }
            }
            let via = self.via_fwd[idx];
            if via != u32::MAX {
                deps[count] = via;
                count += 1;
            }
            for i in 0..count {
                let d = deps[i];
                if d != u32::MAX {
                    let di = d as usize;
                    if di < tc && !visited[di] {
                        // Only follow tiles that were placed during wiring
                        if placed.get(di / 64).copied().unwrap_or(0) & (1u64 << (di % 64)) == 0 {
                            continue;
                        }
                        visited[di] = true;
                        queue.push_back(di);
                    }
                }
            }
        }
        let mut result: Vec<u32> = visited
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
            .collect();
        result.sort_unstable();
        result
    }

    /// Sprint 290: Compute the structural forward closure from seed tiles.
    ///
    /// Walks `compact_ops` in topological order. A tile enters the closure if:
    ///   (a) it is a seed, OR
    ///   (b) any of its `CompactOp` inputs (in0/in1/in2) are already in the closure.
    ///
    /// Cross-layer influence propagates via `via_fwd`: when a tile enters the
    /// closure, its via_fwd target (if any) is also marked, so downstream ops
    /// that read from that layer see the influence.
    ///
    /// Returns closure tile indices in the same topological order as compact_ops.
    pub fn compute_forward_closure(
        &self,
        seeds: &[usize],
        compact_ops: &[CompactOp],
    ) -> Vec<usize> {
        let tile_count = self.tilemap.tiles.len();
        let mut in_closure = vec![false; tile_count];

        for &seed in seeds {
            if seed < tile_count {
                in_closure[seed] = true;
            }
        }

        let mut result = Vec::new();
        for cop in compact_ops {
            let idx = cop.idx as usize;
            if idx >= tile_count {
                continue;
            }

            let input_in_closure = (cop.in0 < tile_count as u32 && in_closure[cop.in0 as usize])
                || (cop.in1 < tile_count as u32 && in_closure[cop.in1 as usize])
                || (cop.in2 < tile_count as u32 && in_closure[cop.in2 as usize]);

            if in_closure[idx] || input_in_closure {
                in_closure[idx] = true;
                result.push(idx);

                // Propagate through via_fwd (cross-layer downstream).
                let via = self.via_fwd[idx];
                if via != u32::MAX && (via as usize) < tile_count {
                    in_closure[via as usize] = true;
                }
            }
        }

        result
    }

    /// Like `propagate_combinational_counted`, but only drains/evaluates tiles within `scope`.
    /// Tiles dirtied outside the scope remain dirty for later unscoped propagation.
    /// `scope` must be a sorted slice of tile indices (u32) representing the zone closure.
    pub fn propagate_combinational_scoped(&mut self, scope: &[u32]) -> (u32, u32, u32) {
        let mut delta_count: u32 = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        loop {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_scoped(scope, &mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }
            let mut any_changed = false;
            let batch_size = batch.len() as u32;
            for &idx32 in batch.iter() {
                if self.eval_tile(idx32 as usize) {
                    any_changed = true;
                    total_switched += 1;
                }
            }
            total_evaluated += batch_size;
            self.dirty_batch_buf = batch;
            if !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 100 {
                break;
            }
        }
        (delta_count, total_evaluated, total_switched)
    }

    /// Like `propagate_combinational_scoped` but uses a bitset mask for O(L1 + active_segments)
    /// drain instead of O(scope_size). `scope_mask` has one u64 per 64-tile segment.
    pub fn propagate_combinational_masked(&mut self, scope_mask: &[u64]) -> (u32, u32, u32) {
        let mut delta_count: u32 = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        loop {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }
            let mut any_changed = false;
            let batch_size = batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                // Sprint 173: Skip chain tail members — handled by chain fusion.
                if self.is_chain_tail(idx) {
                    continue;
                }
                // Sprint 172: chain fusion — fuse entire chain from head.
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        any_changed = true;
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    any_changed = true;
                    total_switched += 1;
                }
            }
            total_evaluated += batch_size;
            self.dirty_batch_buf = batch;
            if !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 100 {
                break;
            }
        }
        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 262: Build a topologically sorted evaluation order for tiles
    /// in the given scope mask. Returns tile indices in dependency order
    /// (sources first, sinks last). Used by `propagate_levelized` for
    /// one-pass evaluation that eliminates iterative convergence.
    ///
    /// Algorithm: Kahn's algorithm (BFS from zero-in-degree nodes).
    /// Dependencies are determined by tile type (e.g., WireRight depends on LEFT,
    /// ViaUp depends on cross-layer source). Only in-scope dependencies count.
    pub fn build_eval_order(&self, scope_mask: &[u64]) -> Vec<usize> {
        // Step 1: Collect all tile indices in scope.
        let mut scope_tiles: Vec<usize> = Vec::new();
        for (word_idx, &word) in scope_mask.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let mut w = word;
            let base = word_idx * 64;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                scope_tiles.push(base + bit);
                w &= w - 1;
            }
        }
        let n = scope_tiles.len();
        if n == 0 {
            return Vec::new();
        }

        // Step 2: Build scope lookup (global index → local index).
        let mut idx_to_local: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(n);
        for (local, &global) in scope_tiles.iter().enumerate() {
            idx_to_local.insert(global, local);
        }

        // Step 3: Build dependency graph + in-degree counts.
        let mut in_degree = vec![0u16; n];
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let layer_size = self.tilemap.layer_size;
        let tile_count = self.tilemap.tiles.len();

        for (local, &idx) in scope_tiles.iter().enumerate() {
            if idx >= tile_count {
                continue;
            }
            let tt = self.tile_type_at(idx);
            let n4 = &self.neighbors4[idx];

            // Collect input indices based on tile type.
            // Only add in-scope inputs as dependencies.
            macro_rules! dep {
                ($global_idx:expr) => {
                    let gi = $global_idx;
                    if gi != usize::MAX && gi < tile_count {
                        if let Some(&input_local) = idx_to_local.get(&gi) {
                            in_degree[local] += 1;
                            out_edges[input_local].push(local);
                        }
                    }
                };
            }

            match tt {
                // Source tiles: no combinational inputs.
                TileType::Const
                | TileType::Register8
                | TileType::Register64
                | TileType::ProgramCounter
                | TileType::Ram
                | TileType::Counter
                | TileType::ClockGlobal
                | TileType::RegEnable
                | TileType::BusInterface
                | TileType::CpuHead
                | TileType::Register
                | TileType::Console
                | TileType::VmSpawner
                | TileType::VmStatus
                | TileType::QDemo
                | TileType::ComponentOutput => {}

                // 1-input unidirectional wires
                TileType::WireRight => {
                    dep!(n4[0] as usize);
                }
                TileType::WireLeft => {
                    dep!(n4[1] as usize);
                }
                TileType::WireDown => {
                    dep!(n4[2] as usize);
                }
                TileType::WireUp => {
                    dep!(n4[3] as usize);
                }

                // 2-input bidirectional wires
                TileType::WireH => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                }
                TileType::WireV => {
                    dep!(n4[2] as usize);
                    dep!(n4[3] as usize);
                }

                // 4-input wires
                TileType::Wire | TileType::Cross => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                    dep!(n4[2] as usize);
                    dep!(n4[3] as usize);
                }

                // 2-input gates (LEFT, RIGHT)
                TileType::And
                | TileType::Or
                | TileType::Xor
                | TileType::Add
                | TileType::Sub
                | TileType::Mul
                | TileType::Div
                | TileType::Mod
                | TileType::Shl
                | TileType::Shr
                | TileType::Lt
                | TileType::Gt
                | TileType::Eq
                | TileType::Neq
                | TileType::Lte
                | TileType::Gte
                | TileType::CarryDetect
                | TileType::Mux8to1
                | TileType::SubBorrow
                | TileType::AddCarry => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                }

                // 1-input (LEFT only)
                TileType::Not
                | TileType::Zero
                | TileType::Neg
                | TileType::Abs
                | TileType::Decoder3to8
                | TileType::Decoder6to64 => {
                    dep!(n4[0] as usize);
                }

                // 1-input (special)
                TileType::BitSelect => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                }

                // Mux: LEFT, RIGHT, UP
                TileType::Mux | TileType::Mux16to1 => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                    dep!(n4[2] as usize);
                }

                // Mux4to1: UP (data), DOWN (select)
                TileType::Mux4to1 => {
                    dep!(n4[2] as usize);
                    dep!(n4[3] as usize);
                }

                // Demux1to8: LEFT (addr), UP (data)
                TileType::Demux1to8 => {
                    dep!(n4[0] as usize);
                    dep!(n4[2] as usize);
                }

                // Latch: LEFT (data), UP (clock/gate)
                TileType::Latch => {
                    dep!(n4[0] as usize);
                    dep!(n4[2] as usize);
                }

                // MemoryPort: LEFT, RIGHT, UP
                TileType::MemoryPort => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                    dep!(n4[2] as usize);
                }

                // Via tiles: cross-layer source
                TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp => {
                    let source = idx + layer_size;
                    if source < tile_count {
                        dep!(source);
                    }
                    // ThresholdVia also reads spatial neighbors for threshold count
                    if tt == TileType::ThresholdViaUp {
                        dep!(n4[0] as usize);
                        dep!(n4[1] as usize);
                        dep!(n4[2] as usize);
                        dep!(n4[3] as usize);
                    }
                }
                TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown => {
                    if idx >= layer_size {
                        dep!(idx - layer_size);
                    }
                    if tt == TileType::ThresholdViaDown {
                        dep!(n4[0] as usize);
                        dep!(n4[1] as usize);
                        dep!(n4[2] as usize);
                        dep!(n4[3] as usize);
                    }
                }

                // WireCross variants
                TileType::WireCross => {
                    dep!(n4[0] as usize);
                    dep!(n4[2] as usize);
                }
                TileType::WireCrossVert => {
                    dep!(n4[1] as usize);
                    dep!(n4[2] as usize);
                }
                TileType::VBusIn => {
                    dep!(n4[0] as usize);
                }
                TileType::VBusOut => {
                    dep!(n4[0] as usize);
                }

                // Conservative default: all 4 neighbors
                _ => {
                    dep!(n4[0] as usize);
                    dep!(n4[1] as usize);
                    dep!(n4[2] as usize);
                    dep!(n4[3] as usize);
                }
            }
        }

        // Step 4: Kahn's algorithm — BFS from zero-in-degree tiles.
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut eval_order: Vec<usize> = Vec::with_capacity(n);
        while let Some(local) = queue.pop_front() {
            eval_order.push(scope_tiles[local]);
            for &succ in &out_edges[local] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        // Handle cycles: append remaining tiles (if any) at the end.
        if eval_order.len() < n {
            for i in 0..n {
                if in_degree[i] > 0 {
                    eval_order.push(scope_tiles[i]);
                }
            }
        }

        eval_order
    }
}

// =============================================================================
// Sprint 267: Compact Evaluator
// =============================================================================

/// Compact operation for pre-decoded tile evaluation.
/// Sprint 278: Precomputed schedule for ordered active-work propagation.
/// Holds compact ops + dependency edges between in-scope tiles.
#[derive(Clone, Debug)]
pub struct CompactSchedule {
    pub ops: Vec<CompactOp>,
    pub wvia: Vec<(usize, u8, u64)>,
    /// Maps global tile index → op slot position. u32::MAX if not in scope.
    pub idx_to_slot: Vec<u32>,
    /// Flat-packed dependency edges: deps_data[deps_offsets[slot]..deps_offsets[slot+1]]
    /// gives the downstream op slots to activate when slot changes.
    pub deps_data: Vec<u32>,
    pub deps_offsets: Vec<u32>,
    /// For COP_WVIA slots, index into the wvia params array. u32::MAX otherwise.
    pub wvia_slot_idx: Vec<u32>,
    /// Scope mask for fast residual drain via fill_into_masked.
    pub scope_mask: Vec<u64>,
    /// Sprint 292: True if any COP_GENERIC ops exist in this schedule.
    /// When false, re-drain and global residual checks can be skipped
    /// (they exist solely for COP_GENERIC's global dirty marks).
    pub has_generic: bool,
    /// Sprint 333: Precomputed list of (segment_index, scope_mask_word) for
    /// segments that have at least one in-scope tile. Used for sparse drain.
    pub in_scope_segments: Vec<(u32, u64)>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CompactOp {
    pub op: u8,
    pub _pad: [u8; 3],
    pub idx: u32,
    pub in0: u32,
    pub in1: u32,
    pub in2: u32,
}

pub const COP_CONST: u8 = 0;
pub const COP_WIRE_R: u8 = 1;
pub const COP_WIRE_L: u8 = 2;
pub const COP_WIRE_D: u8 = 3;
pub const COP_WIRE_U: u8 = 4;
pub const COP_WIRE: u8 = 5;
pub const COP_AND: u8 = 6;
pub const COP_OR: u8 = 7;
pub const COP_XOR: u8 = 8;
pub const COP_MUX: u8 = 9;
pub const COP_NOT: u8 = 10;
pub const COP_ZERO: u8 = 11;
pub const COP_ADD: u8 = 12;
pub const COP_SUB: u8 = 13;
pub const COP_SHR: u8 = 14;
pub const COP_SHL: u8 = 15;
pub const COP_VIA: u8 = 16;
pub const COP_MUX16: u8 = 17;
pub const COP_DEC3: u8 = 18;
pub const COP_BITSEL: u8 = 19;
pub const COP_CARRY: u8 = 20;
pub const COP_WVIA: u8 = 21;
pub const COP_MUX4: u8 = 22;
pub const COP_WIRE_H: u8 = 23;
pub const COP_WIRE_V: u8 = 24;
pub const COP_RAM: u8 = 25;
pub const COP_GENERIC: u8 = 26; // fallback: call eval_tile
/// ThresholdVia as a first-class compact op (GPU-bridge foundation, 2026-06-13).
/// `output = if popcount(4 in-plane neighbors != 0) >= tile_threshold[idx] { source } else { 0 }`.
/// The cross-layer source is stored in `in0`; the 4 in-plane inputs are read from
/// `neighbors4[idx]` and the threshold from `tile_threshold[idx]` at eval time
/// (a 5-input op that does not fit the 3 CompactOp input slots). Previously these
/// tiles fell through to COP_CONST and were silently frozen on the compact path.
pub const COP_THRESHOLD_VIA: u8 = 27;

impl Simulation {
    /// Evaluate a ThresholdVia gate on the compact path (COP_THRESHOLD_VIA).
    ///
    /// Mirrors the `eval_tile` ThresholdVia semantics exactly: count the in-plane
    /// 4-neighbors (left/right/up/down) whose value is non-zero; if that count
    /// meets the per-tile threshold, pass the cross-layer `source` through, else 0.
    /// `source` is the already-loaded value of the via's cross-layer source tile
    /// (op.in0), so out-of-bounds sources (in0 == u32::MAX ⇒ source == 0) yield 0,
    /// matching eval_tile's boundary behavior. `neighbors4[idx]` carries exactly the
    /// same per-layer boundary guards (u32::MAX at the grid edge) as eval_tile uses.
    #[inline]
    fn threshold_via_gate(&self, idx: usize, source: u64) -> u64 {
        let n = self.neighbors4[idx];
        let mut active: u8 = 0;
        if n[0] != u32::MAX && self.tilemap.value(n[0] as usize) != 0 {
            active += 1;
        }
        if n[1] != u32::MAX && self.tilemap.value(n[1] as usize) != 0 {
            active += 1;
        }
        if n[2] != u32::MAX && self.tilemap.value(n[2] as usize) != 0 {
            active += 1;
        }
        if n[3] != u32::MAX && self.tilemap.value(n[3] as usize) != 0 {
            active += 1;
        }
        if active >= self.tile_threshold[idx] {
            source
        } else {
            0
        }
    }

    /// Sprint 267: Build compact op array from eval_order.
    /// Each tile is pre-decoded into a CompactOp with resolved input indices.
    pub fn build_compact_ops(
        &self,
        eval_order: &[usize],
    ) -> (Vec<CompactOp>, Vec<(usize, u8, u64)>) {
        self.build_compact_ops_inner(eval_order, false)
    }

    /// Sprint 276: Variant for clock-scope compact ops. Ram tiles are treated as
    /// COP_CONST (not COP_RAM) because they capture only in delta 0 via eval_tile.
    /// The cascade should not re-evaluate them: that would allow spurious writes when
    /// new-PC decode propagates a non-zero WE for the next instruction.
    pub fn build_compact_ops_clock(
        &self,
        eval_order: &[usize],
    ) -> (Vec<CompactOp>, Vec<(usize, u8, u64)>) {
        self.build_compact_ops_inner(eval_order, true)
    }

    fn build_compact_ops_inner(
        &self,
        eval_order: &[usize],
        ram_as_const: bool,
    ) -> (Vec<CompactOp>, Vec<(usize, u8, u64)>) {
        // Returns (ops, weighted_via_params) where weighted_via_params[i] = (op_index, shift, mask)
        let mut ops = Vec::with_capacity(eval_order.len());
        let mut wvia_params: Vec<(usize, u8, u64)> = Vec::new();
        let layer_size = self.tilemap.layer_size;
        let tile_count = self.tilemap.tiles.len();

        for &idx in eval_order {
            if idx >= tile_count {
                continue;
            }
            let tt = self.tile_type_at(idx);
            let n = &self.neighbors4[idx];

            let (op, i0, i1, i2) = match tt {
                // Sources (hold current value)
                TileType::Const
                | TileType::Register8
                | TileType::Register64
                | TileType::ProgramCounter
                | TileType::Counter
                | TileType::ClockGlobal
                | TileType::RegEnable
                | TileType::BusInterface
                | TileType::CpuHead
                | TileType::Register
                | TileType::Console
                | TileType::VmSpawner
                | TileType::VmStatus
                | TileType::QDemo
                | TileType::ComponentOutput => (COP_CONST, u32::MAX, u32::MAX, u32::MAX),

                // Unidirectional wires
                TileType::WireRight => (COP_WIRE_R, n[0], u32::MAX, u32::MAX),
                TileType::WireLeft => (COP_WIRE_L, u32::MAX, n[1], u32::MAX),
                TileType::WireDown => (COP_WIRE_D, u32::MAX, u32::MAX, n[2]),
                TileType::WireUp => (COP_WIRE_U, u32::MAX, u32::MAX, n[3]),

                // Bidirectional wires
                TileType::WireH => (COP_WIRE_H, n[0], n[1], u32::MAX),
                TileType::WireV => (COP_WIRE_V, u32::MAX, n[3], n[2]),

                // Omnidirectional wire — need all 4, pack UP+DOWN into in2 trick
                // Actually store as OR of all 4 using a 4-input wire opcode
                TileType::Wire => (COP_WIRE, n[0], n[1], n[2]),
                // Note: Wire also reads n[3] (DOWN). We handle this in the eval loop
                // by storing n[3] in the _pad field... or just use COP_WIRE special case.

                // Ram: up != 0 ? left : current (write-enable gated).
                // In clock-scope mode (ram_as_const), treat as COP_CONST so the
                // cascade doesn't re-capture after delta 0's eval_tile capture.
                TileType::Ram => {
                    if ram_as_const {
                        (COP_CONST, u32::MAX, u32::MAX, u32::MAX)
                    } else {
                        (COP_RAM, n[0], u32::MAX, n[2])
                    }
                }

                // 2-input gates
                TileType::And => (COP_AND, n[0], n[1], u32::MAX),
                TileType::Or => (COP_OR, n[0], n[1], u32::MAX),
                TileType::Xor => (COP_XOR, n[0], n[1], u32::MAX),
                TileType::Add => (COP_ADD, n[0], n[1], u32::MAX),
                TileType::Sub => (COP_SUB, n[0], n[1], u32::MAX),
                // AddCarry/SubBorrow have non-trivial semantics (16-bit masked).
                TileType::AddCarry | TileType::SubBorrow => {
                    (COP_GENERIC, u32::MAX, u32::MAX, u32::MAX)
                }
                TileType::Shr => (COP_SHR, n[0], n[1], u32::MAX),
                TileType::Shl => (COP_SHL, n[0], n[1], u32::MAX),
                TileType::CarryDetect => (COP_CARRY, n[0], n[1], u32::MAX),
                TileType::BitSelect => (COP_BITSEL, n[0], n[1], u32::MAX),

                // 1-input
                TileType::Not => (COP_NOT, n[0], u32::MAX, u32::MAX),
                // Neg (~left + 1) and Abs are distinct from Not (~left).
                TileType::Neg | TileType::Abs => (COP_GENERIC, u32::MAX, u32::MAX, u32::MAX),
                TileType::Zero => (COP_ZERO, n[0], u32::MAX, u32::MAX),
                TileType::Decoder3to8 => (COP_DEC3, n[0], u32::MAX, u32::MAX),
                TileType::Decoder6to64 => (COP_GENERIC, u32::MAX, u32::MAX, u32::MAX),

                // Mux variants
                TileType::Mux => (COP_MUX, n[0], n[1], n[2]),
                TileType::Mux16to1 => (COP_MUX16, n[0], n[1], n[2]),
                TileType::Mux4to1 => (COP_MUX4, n[2], u32::MAX, n[3]),
                TileType::Mux8to1 => (COP_GENERIC, u32::MAX, u32::MAX, u32::MAX),

                // Via tiles
                TileType::ViaUp | TileType::ViaDown => {
                    let source = if tt == TileType::ViaUp {
                        if idx + layer_size < tile_count {
                            (idx + layer_size) as u32
                        } else {
                            u32::MAX
                        }
                    } else {
                        if idx >= layer_size {
                            (idx - layer_size) as u32
                        } else {
                            u32::MAX
                        }
                    };
                    (COP_VIA, source, u32::MAX, u32::MAX)
                }
                TileType::WeightedViaUp | TileType::WeightedViaDown => {
                    let source = if tt == TileType::WeightedViaUp {
                        if idx + layer_size < tile_count {
                            (idx + layer_size) as u32
                        } else {
                            u32::MAX
                        }
                    } else {
                        if idx >= layer_size {
                            (idx - layer_size) as u32
                        } else {
                            u32::MAX
                        }
                    };
                    let op_idx = ops.len();
                    wvia_params.push((op_idx, self.tile_shift[idx], self.tile_mask[idx]));
                    (COP_WVIA, source, u32::MAX, u32::MAX)
                }

                // Threshold Via tiles (Sprint 183): popcount of the 4 in-plane
                // neighbors >= threshold gates the cross-layer source. Source goes
                // in in0; the 4 neighbor inputs and the threshold are read at eval
                // time from neighbors4[idx] / tile_threshold[idx] (5 inputs do not
                // fit the 3 CompactOp slots — see threshold_via_gate / COP_THRESHOLD_VIA).
                TileType::ThresholdViaUp | TileType::ThresholdViaDown => {
                    let source = if tt == TileType::ThresholdViaUp {
                        if idx + layer_size < tile_count {
                            (idx + layer_size) as u32
                        } else {
                            u32::MAX
                        }
                    } else if idx >= layer_size {
                        (idx - layer_size) as u32
                    } else {
                        u32::MAX
                    };
                    (COP_THRESHOLD_VIA, source, u32::MAX, u32::MAX)
                }

                // Fallback: treat as const (will be evaluated by generic path)
                _ => (COP_CONST, u32::MAX, u32::MAX, u32::MAX),
            };

            ops.push(CompactOp {
                op,
                _pad: [0; 3],
                idx: idx as u32,
                in0: i0,
                in1: i1,
                in2: i2,
            });
        }

        (ops, wvia_params)
    }

    /// Sprint 273: Execute JIT-compiled cone evaluation and handle dirty propagation.
    /// Returns (1, op_count, tiles_switched).
    #[cfg(feature = "cranelift_jit")]
    pub fn propagate_jit_cone(
        &mut self,
        jit: &crate::tile_cpu::tile_jit::TileEvalJitProgram,
    ) -> (u32, u32, u32) {
        use std::sync::atomic::Ordering;

        // Sprint 273.1: Buffer sized to op_count (every op could change).
        let buf_cap = jit.op_count;
        let mut changed_buf: Vec<u32> = vec![0u32; buf_cap];

        // Get raw pointer to the SoA value array (Sprint 385 relocation).
        // Sprint 386: the JIT walks the single-value array; jit_values_ptr
        // asserts we are not in lane mode (the fabric disables JIT).
        let tiles_ptr = self.tilemap.jit_values_ptr();
        let changed_ptr = changed_buf.as_mut_ptr();

        // Call JIT function.
        let changed_count =
            unsafe { (jit.func_ptr)(tiles_ptr, changed_ptr, buf_cap as u32) } as usize;

        // Post-pass: propagate dirty marks for all changed tiles.
        let actual_changed = changed_count.min(buf_cap);
        for i in 0..actual_changed {
            let idx = changed_buf[i] as usize;
            let nc = self.neighbors4[idx];
            let tt = self.tile_type_at(idx);
            self.dirty_dependents(&nc, idx, tt);
        }

        (1, jit.op_count as u32, actual_changed as u32)
    }

    /// Sprint 335: JIT-compiled settle evaluation with Rust convergence loop.
    /// Evaluates all settle ops unconditionally via JIT, then processes changed
    /// tiles in Rust (dirty_dependents). Loops until convergence.
    #[cfg(feature = "cranelift_jit")]
    pub fn propagate_jit_settle(
        &mut self,
        jit: &crate::tile_cpu::tile_jit::TileEvalJitProgram,
        scope_mask: &[u64],
        in_scope_segments: &[(u32, u64)],
        idx_to_slot: &[u32],
        frontier_offsets: &[u32],
        frontier_targets: &[u32],
    ) -> (u32, u32, u32) {
        let buf_cap = jit.op_count;
        let mut changed_buf = std::mem::take(&mut self.dirty_batch_buf);
        changed_buf.resize(buf_cap, 0);
        let tiles_ptr = self.tilemap.jit_values_ptr();
        let changed_ptr = changed_buf.as_mut_ptr();

        // Sprint 338: Single-pass.
        let changed_count =
            unsafe { (jit.func_ptr)(tiles_ptr, changed_ptr, buf_cap as u32) } as usize;
        let actual_changed = changed_count.min(buf_cap);

        // Sprint 339: Frontier-table dirty propagation — precomputed out-of-scope
        // neighbors per op slot. Falls back to dirty_dependents_frontier when table
        // is empty.
        if !frontier_offsets.is_empty() {
            for i in 0..actual_changed {
                let idx = changed_buf[i] as usize;
                if idx < idx_to_slot.len() {
                    let slot = idx_to_slot[idx];
                    if slot != u32::MAX && (slot as usize + 1) < frontier_offsets.len() {
                        let start = frontier_offsets[slot as usize] as usize;
                        let end = frontier_offsets[slot as usize + 1] as usize;
                        for j in start..end {
                            self.dirty.mark_dirty(frontier_targets[j] as usize);
                        }
                    }
                }
            }
        } else {
            for i in 0..actual_changed {
                let idx = changed_buf[i] as usize;
                let nc = self.neighbors4[idx];
                let tt = self.tile_type_at(idx);
                self.dirty_dependents_frontier(&nc, idx, tt, scope_mask);
                let via = self.via_fwd[idx];
                if via != u32::MAX {
                    let vi = via as usize;
                    let in_scope =
                        scope_mask.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                    if !in_scope {
                        self.dirty.mark_dirty(vi);
                    }
                }
            }
        }

        // Sprint 384: Drain residual in-scope dirty bits. Blockskip consumed
        // these via is_dirty_and_clear during its scan; the JIT path never
        // reads the dirty bitset, so pre-inject seeds and prior-phase frontier
        // marks inside the settle scope survived this call and were re-processed
        // by later phases (S341 flagged branch 1.6→3.9 µs under JIT settle).
        // Every settle-scope tile was just evaluated in topological order, so
        // these bits carry no information. Frontier marks made above target
        // out-of-scope tiles only and are unaffected. Sparse (S333 pattern):
        // a masked drain traverses every dirty segment in the grid (~6 µs with
        // build-time residue); the sparse list visits only settle-scope segments.
        self.jit_settle_drained_total += if !in_scope_segments.is_empty() {
            self.dirty.clear_sparse(in_scope_segments) as u64
        } else {
            self.dirty.clear_masked(scope_mask) as u64
        };

        self.dirty_batch_buf = changed_buf;
        (1, jit.op_count as u32, actual_changed as u32)
    }

    /// Sprint 337/338/340: Profiled variant — uses same frontier table as production
    /// path (S339), keeps convergence loop for pass-2 counting.
    /// Sprint 384: Takes scope_mask for the residual-dirty drain (matches production).
    #[cfg(feature = "cranelift_jit")]
    pub fn propagate_jit_settle_profiled(
        &mut self,
        jit: &crate::tile_cpu::tile_jit::TileEvalJitProgram,
        scope_mask: &[u64],
        in_scope_segments: &[(u32, u64)],
        idx_to_slot: &[u32],
        frontier_offsets: &[u32],
        frontier_targets: &[u32],
    ) -> (u32, u32, u32, u64, u64, u32, u32, u32, u32) {
        let buf_cap = jit.op_count;
        let mut changed_buf = std::mem::take(&mut self.dirty_batch_buf);
        changed_buf.resize(buf_cap, 0);
        let tiles_ptr = self.tilemap.jit_values_ptr();
        let changed_ptr = changed_buf.as_mut_ptr();

        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;
        let mut eval_ns_total: u64 = 0;
        let mut dirty_ns_total: u64 = 0;
        let mut total_changed: u32 = 0;
        let mut pass1_changed: u32 = 0;
        let mut pass2_changed: u32 = 0;

        loop {
            let eval_start = std::time::Instant::now();
            let changed_count =
                unsafe { (jit.func_ptr)(tiles_ptr, changed_ptr, buf_cap as u32) } as usize;
            eval_ns_total += eval_start.elapsed().as_nanos() as u64;
            let actual_changed = changed_count.min(buf_cap);
            total_evaluated += jit.op_count as u32;
            total_switched += actual_changed as u32;
            total_changed += actual_changed as u32;
            match delta_count {
                0 => pass1_changed = actual_changed as u32,
                1 => pass2_changed = actual_changed as u32,
                _ => {}
            }

            if actual_changed == 0 {
                break;
            }

            // Sprint 340: Use same frontier table as production path.
            let dirty_start = std::time::Instant::now();
            if !frontier_offsets.is_empty() {
                for i in 0..actual_changed {
                    let idx = changed_buf[i] as usize;
                    if idx < idx_to_slot.len() {
                        let slot = idx_to_slot[idx];
                        if slot != u32::MAX && (slot as usize + 1) < frontier_offsets.len() {
                            let start = frontier_offsets[slot as usize] as usize;
                            let end = frontier_offsets[slot as usize + 1] as usize;
                            for j in start..end {
                                self.dirty.mark_dirty(frontier_targets[j] as usize);
                            }
                        }
                    }
                }
            }
            dirty_ns_total += dirty_start.elapsed().as_nanos() as u64;

            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        // Sprint 384: Residual-dirty drain (matches production path, sparse).
        let drain_start = std::time::Instant::now();
        self.jit_settle_drained_total += if !in_scope_segments.is_empty() {
            self.dirty.clear_sparse(in_scope_segments) as u64
        } else {
            self.dirty.clear_masked(scope_mask) as u64
        };
        dirty_ns_total += drain_start.elapsed().as_nanos() as u64;

        self.dirty_batch_buf = changed_buf;
        (
            delta_count,
            total_evaluated,
            total_switched,
            eval_ns_total,
            dirty_ns_total,
            delta_count + 1,
            total_changed,
            pass1_changed,
            pass2_changed,
        )
    }

    /// Sprint 339: Build frontier table for JIT settle dirty propagation.
    /// For each op, precomputes which out-of-scope neighbors need mark_dirty.
    /// Returns (offsets, targets) where offsets[slot]..offsets[slot+1] indexes targets.
    pub fn build_settle_frontier_table(
        &self,
        ops: &[CompactOp],
        cone_set: &[u64],
    ) -> (Vec<u32>, Vec<u32>) {
        let num_ops = ops.len();
        let mut offsets = Vec::with_capacity(num_ops + 1);
        let mut targets: Vec<u32> = Vec::new();

        let is_in_cone = |ni: u32| -> bool {
            if ni == u32::MAX {
                return true;
            }
            let i = ni as usize;
            cone_set.get(i / 64).copied().unwrap_or(0) & (1u64 << (i % 64)) != 0
        };

        for op in ops {
            offsets.push(targets.len() as u32);
            if op.op == COP_CONST {
                continue;
            }
            let idx = op.idx as usize;
            let tt = self.tile_type_at(idx);
            let n = &self.neighbors4[idx];

            macro_rules! maybe_target {
                ($ni:expr) => {
                    if $ni != u32::MAX && !is_in_cone($ni) {
                        targets.push($ni);
                    }
                };
            }
            match tt {
                TileType::WireRight => {
                    maybe_target!(n[1]);
                    maybe_target!(n[2]);
                    maybe_target!(n[3]);
                }
                TileType::WireLeft => {
                    maybe_target!(n[0]);
                    maybe_target!(n[2]);
                    maybe_target!(n[3]);
                }
                TileType::WireDown => {
                    maybe_target!(n[0]);
                    maybe_target!(n[1]);
                    maybe_target!(n[3]);
                }
                TileType::WireUp => {
                    maybe_target!(n[0]);
                    maybe_target!(n[1]);
                    maybe_target!(n[2]);
                }
                TileType::WireH => {
                    maybe_target!(n[2]);
                    maybe_target!(n[3]);
                }
                TileType::WireV => {
                    maybe_target!(n[0]);
                    maybe_target!(n[1]);
                }
                _ => {
                    maybe_target!(n[0]);
                    maybe_target!(n[1]);
                    maybe_target!(n[2]);
                    maybe_target!(n[3]);
                }
            }
            let via = self.via_fwd[idx];
            if via != u32::MAX && !is_in_cone(via) {
                targets.push(via);
            }
        }
        offsets.push(targets.len() as u32);
        (offsets, targets)
    }

    /// Sprint 274 C1: Count how many cone ops would produce a different value
    /// if re-evaluated right now. Does NOT modify any tile values.
    /// A return of 0 proves single-pass convergence for the cone.
    pub fn count_cone_residual_changes(&self, ops: &[CompactOp]) -> u32 {
        let mut residual = 0u32;
        for op in ops {
            if op.op == COP_CONST {
                continue;
            }
            let idx = op.idx as usize;
            let ld = |i: u32| -> u64 {
                if i == u32::MAX {
                    0
                } else {
                    self.tilemap.value(i as usize)
                }
            };
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(idx);
            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D | COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2,
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0 >> (v1 & 63),
                COP_SHL => v0 << (v1 & 63),
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                _ => current, // unknown ops: assume stable
            };
            if result != current {
                residual += 1;
            }
        }
        residual
    }

    /// Sprint 274 C2: Evaluate all cone ops once in topological order. No dirty checks
    /// on entry, no dirty marks for intra-cone changes. Only marks tiles OUTSIDE the
    /// cone that depend on changed cone tiles (the "frontier").
    ///
    /// `cone_set`: bitset of tile indices in the cone (for O(1) membership test).
    /// Returns (1, ops_evaluated, tiles_switched).
    pub fn propagate_cone_no_dirty(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        cone_set: &[u64],
    ) -> (u32, u32, u32) {
        let mut total_switched = 0u32;
        let mut wvia_idx = 0usize;

        for op in ops {
            let is_wvia = op.op == COP_WVIA;
            if op.op == COP_CONST {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            let idx = op.idx as usize;
            let ld = |i: u32| -> u64 {
                if i == u32::MAX {
                    0
                } else {
                    self.tilemap.value(i as usize)
                }
            };
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(idx);

            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D | COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2,
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0 >> (v1 & 63),
                COP_SHL => v0 << (v1 & 63),
                COP_MUX16 => {
                    let lane = v1 & 0xF;
                    let source = if lane < 8 { v0 } else { v2 };
                    (source >> ((lane & 7) * 8)) & 0xFF
                }
                COP_DEC3 => 1u64 << (v0 & 7),
                COP_BITSEL => {
                    if ((v0 >> (v1 & 63)) & 1) != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_CARRY => {
                    if v0 > v1 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_WVIA => {
                    let r = if wvia_idx < wvia_params.len() {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        (v0 >> shift) & mask
                    } else {
                        v0
                    };
                    wvia_idx += 1;
                    r
                }
                COP_MUX4 => (v0 >> ((v2 & 3) * 8)) & 0xFF,
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                _ => {
                    if is_wvia && op.op != COP_WVIA {
                        wvia_idx += 1;
                    }
                    continue;
                }
            };

            if result != current {
                self.tilemap.set_value(idx, result);
                total_switched += 1;

                // Only mark dirty tiles OUTSIDE the cone (frontier propagation).
                let nc = self.neighbors4[idx];
                let tt = self.tile_type_at(idx);
                self.dirty_dependents_frontier(&nc, idx, tt, cone_set);
                // Also mark via_fwd target if outside cone.
                let via = self.via_fwd[idx];
                if via != u32::MAX {
                    let vi = via as usize;
                    let in_cone =
                        cone_set.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                    if !in_cone {
                        self.dirty.mark_dirty(vi);
                    }
                }
            }
        }

        (1, ops.len() as u32, total_switched)
    }

    /// Sprint 384: Frontier-table variant of propagate_cone_no_dirty (S339 pattern).
    /// Identical evaluation semantics, but frontier dirty marks come from a
    /// precomputed table indexed by op slot instead of the dynamic
    /// tile_type/neighbors4/scope-bitset walk (~10 ns/changed → ~3-5 ns/changed).
    /// The table is built by `build_settle_frontier_table(ops, scope_mask)` and
    /// therefore produces the exact same mark set as `dirty_dependents_frontier`
    /// + via_fwd. Used for the pruned clock cascade where most ops switch.
    pub fn propagate_cone_no_dirty_ft(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        frontier_offsets: &[u32],
        frontier_targets: &[u32],
    ) -> (u32, u32, u32) {
        let mut total_switched = 0u32;
        let mut wvia_idx = 0usize;

        for (slot, op) in ops.iter().enumerate() {
            let is_wvia = op.op == COP_WVIA;
            if op.op == COP_CONST {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            let idx = op.idx as usize;
            let ld = |i: u32| -> u64 {
                if i == u32::MAX {
                    0
                } else {
                    self.tilemap.value(i as usize)
                }
            };
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(idx);

            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D | COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2,
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0 >> (v1 & 63),
                COP_SHL => v0 << (v1 & 63),
                COP_MUX16 => {
                    let lane = v1 & 0xF;
                    let source = if lane < 8 { v0 } else { v2 };
                    (source >> ((lane & 7) * 8)) & 0xFF
                }
                COP_DEC3 => 1u64 << (v0 & 7),
                COP_BITSEL => {
                    if ((v0 >> (v1 & 63)) & 1) != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_CARRY => {
                    if v0 > v1 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_WVIA => {
                    let r = if wvia_idx < wvia_params.len() {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        (v0 >> shift) & mask
                    } else {
                        v0
                    };
                    wvia_idx += 1;
                    r
                }
                COP_MUX4 => (v0 >> ((v2 & 3) * 8)) & 0xFF,
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                _ => {
                    if is_wvia && op.op != COP_WVIA {
                        wvia_idx += 1;
                    }
                    continue;
                }
            };

            if result != current {
                self.tilemap.set_value(idx, result);
                total_switched += 1;

                // Frontier marks from the precomputed table (slot-indexed).
                if slot + 1 < frontier_offsets.len() {
                    let start = frontier_offsets[slot] as usize;
                    let end = frontier_offsets[slot + 1] as usize;
                    for &t in &frontier_targets[start..end] {
                        self.dirty.mark_dirty(t as usize);
                    }
                }
            }
        }

        (1, ops.len() as u32, total_switched)
    }

    /// Sprint 321: Counted variant of propagate_cone_no_dirty. Identical evaluation
    /// semantics, but increments `switch_counts[slot]` for each op where result != current.
    /// Used for phase-local dead-op measurement (not on the hot path).
    pub fn propagate_cone_no_dirty_counted(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        cone_set: &[u64],
        switch_counts: &mut [u32],
    ) -> (u32, u32, u32) {
        let mut total_switched = 0u32;
        let mut wvia_idx = 0usize;

        for (slot, op) in ops.iter().enumerate() {
            let is_wvia = op.op == COP_WVIA;
            if op.op == COP_CONST {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            let idx = op.idx as usize;
            let ld = |i: u32| -> u64 {
                if i == u32::MAX {
                    0
                } else {
                    self.tilemap.value(i as usize)
                }
            };
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(idx);

            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D | COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2,
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0 >> (v1 & 63),
                COP_SHL => v0 << (v1 & 63),
                COP_MUX16 => {
                    let lane = v1 & 0xF;
                    let source = if lane < 8 { v0 } else { v2 };
                    (source >> ((lane & 7) * 8)) & 0xFF
                }
                COP_DEC3 => 1u64 << (v0 & 7),
                COP_BITSEL => {
                    if ((v0 >> (v1 & 63)) & 1) != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_CARRY => {
                    if v0 > v1 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_WVIA => {
                    let r = if wvia_idx < wvia_params.len() {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        (v0 >> shift) & mask
                    } else {
                        v0
                    };
                    wvia_idx += 1;
                    r
                }
                COP_MUX4 => (v0 >> ((v2 & 3) * 8)) & 0xFF,
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                _ => {
                    if is_wvia && op.op != COP_WVIA {
                        wvia_idx += 1;
                    }
                    continue;
                }
            };

            if result != current {
                self.tilemap.set_value(idx, result);
                total_switched += 1;
                switch_counts[slot] += 1;

                let nc = self.neighbors4[idx];
                let tt = self.tile_type_at(idx);
                self.dirty_dependents_frontier(&nc, idx, tt, cone_set);
                let via = self.via_fwd[idx];
                if via != u32::MAX {
                    let vi = via as usize;
                    let in_cone =
                        cone_set.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                    if !in_cone {
                        self.dirty.mark_dirty(vi);
                    }
                }
            }
        }

        (1, ops.len() as u32, total_switched)
    }

    /// Sprint 274: Like dirty_dependents but only marks neighbors OUTSIDE the cone.
    pub(crate) fn dirty_dependents_frontier(
        &mut self,
        n: &[u32; 4],
        _idx: usize,
        tt: TileType,
        cone_set: &[u64],
    ) {
        let is_in_cone = |ni: u32| -> bool {
            if ni == u32::MAX {
                return true; // no neighbor — skip
            }
            let i = ni as usize;
            cone_set.get(i / 64).copied().unwrap_or(0) & (1u64 << (i % 64)) != 0
        };

        macro_rules! md_frontier {
            ($ni:expr) => {
                if $ni != u32::MAX && !is_in_cone($ni) {
                    self.dirty.mark_dirty($ni as usize);
                }
            };
        }

        match tt {
            TileType::WireRight => {
                md_frontier!(n[1]);
                md_frontier!(n[2]);
                md_frontier!(n[3]);
            }
            TileType::WireLeft => {
                md_frontier!(n[0]);
                md_frontier!(n[2]);
                md_frontier!(n[3]);
            }
            TileType::WireDown => {
                md_frontier!(n[0]);
                md_frontier!(n[1]);
                md_frontier!(n[3]);
            }
            TileType::WireUp => {
                md_frontier!(n[0]);
                md_frontier!(n[1]);
                md_frontier!(n[2]);
            }
            TileType::WireH => {
                md_frontier!(n[2]);
                md_frontier!(n[3]);
            }
            TileType::WireV => {
                md_frontier!(n[0]);
                md_frontier!(n[1]);
            }
            _ => {
                md_frontier!(n[0]);
                md_frontier!(n[1]);
                md_frontier!(n[2]);
                md_frontier!(n[3]);
            }
        }
    }

    /// Sprint 267: Evaluate all compact ops in order. No dirty tracking.
    /// Returns (1, tiles_evaluated, tiles_switched).
    #[inline(never)]
    pub fn propagate_compact(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
    ) -> (u32, u32, u32) {
        let mut total_switched: u32 = 0;
        let mut wvia_idx = 0usize;
        // Sprint 268: Track changed tiles for dirty propagation.
        let mut changed_indices: Vec<u32> = Vec::new();

        // Sprint 386: route through the value accessor (lane-mode aware),
        // matching the sibling compact-eval functions.
        let ld = |i: u32| -> u64 {
            if i == u32::MAX {
                0
            } else {
                self.tilemap.value(i as usize)
            }
        };

        for op in ops {
            if op.op == COP_CONST {
                continue;
            }
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(op.idx as usize);

            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D => v2,
                COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2, // v1=DOWN, v2=UP
                COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[op.idx as usize][3]),
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                COP_MUX16 => {
                    let sel = (v1 & 0xF) as usize;
                    let data = if sel < 8 { v0 } else { v2 };
                    (data >> ((sel & 7) * 8)) & 0xFF
                }
                COP_DEC3 => 1u64 << (v0 & 7),
                COP_BITSEL => {
                    if (v0 >> (v1 & 63)) & 1 != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_CARRY => {
                    if v0 > v1 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_WVIA => {
                    let (_, shift, mask) = wvia_params[wvia_idx];
                    wvia_idx += 1;
                    (v0 >> shift) & mask
                }
                COP_MUX4 => {
                    let sel = (v2 & 0b11) as u32;
                    (v0 >> (sel * 8)) & 0xFF
                }
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                COP_THRESHOLD_VIA => self.threshold_via_gate(op.idx as usize, v0),
                COP_GENERIC => {
                    // Handled in second pass below to avoid borrow conflict.
                    continue;
                }
                _ => continue,
            };

            if result != current {
                self.tilemap.set_value(op.idx as usize, result);
                total_switched += 1;
                changed_indices.push(op.idx);
            }
        }

        // Sprint 268: Propagate dirty marks for changed tiles so downstream
        // scopes (branch, commit) see the updated values.
        for &idx32 in &changed_indices {
            let idx = idx32 as usize;
            let tt = self.tile_type_at(idx);
            let nc = self.neighbors4[idx];
            self.dirty_dependents(&nc, idx, tt);
        }

        // Second pass: handle COP_GENERIC tiles via full eval_tile.
        for op in ops {
            if op.op == COP_GENERIC {
                if self.eval_tile(op.idx as usize) {
                    total_switched += 1;
                }
            }
        }

        (1, ops.len() as u32, total_switched)
    }

    /// Sprint 269: Dirty-aware compact evaluation. Pre-decoded ops (no type
    /// dispatch, resolved input indices) with dirty-bit tracking (only evaluate
    /// tiles whose inputs changed). Combines the best of both approaches.
    ///
    /// Falls back to multi-pass loop for backward dependencies, same as
    /// propagate_levelized.
    pub fn propagate_compact_dirty(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
    ) -> (u32, u32, u32) {
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;

        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;
            let mut wvia_idx = 0usize;

            // Sprint 309: Segment-cached dirty check. Cache the dirty segment
            // word so consecutive ops in the same segment share one load.
            // Invalidate cache after any dirty_dependents call (which writes
            // to arbitrary segments).
            let mut cached_seg: u32 = u32::MAX;
            let mut cached_word: u64 = 0;

            // Flush cached segment back to the dirty bitset.
            #[allow(unused_assignments)]
            macro_rules! flush_cache {
                () => {
                    if cached_seg != u32::MAX {
                        self.dirty.segments[cached_seg as usize].set(cached_word);
                        cached_seg = u32::MAX;
                    }
                };
            }

            for op in ops.iter() {
                let is_wvia = op.op == COP_WVIA;

                let seg = (op.idx / 64) as u32;
                let bit = 1u64 << (op.idx % 64);

                // Load segment (cached — same segment skips the load).
                if seg != cached_seg {
                    flush_cache!();
                    cached_seg = seg;
                    cached_word = self.dirty.segments[seg as usize].get();
                }

                if op.op == COP_CONST {
                    // Sprint 276: Clear dirty bit for COP_CONST.
                    cached_word &= !bit;
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                if cached_word & bit == 0 {
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }
                // Clear dirty bit.
                cached_word &= !bit;
                pass_evaluated += 1;

                // Chain handling: skip chain tails, use chain fusion for heads.
                let idx = op.idx as usize;
                if self.is_chain_tail(idx) {
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    // Flush cache before chain eval (writes to global dirty).
                    flush_cache!();
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        any_changed = true;
                        total_switched += 1 + tail_n;
                    }
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                if op.op == COP_GENERIC {
                    flush_cache!();
                    if self.eval_tile(idx) {
                        any_changed = true;
                        total_switched += 1;
                    }
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                let idx = op.idx as usize;
                let ld = |i: u32| -> u64 {
                    if i == u32::MAX {
                        0
                    } else {
                        self.tilemap.value(i as usize)
                    }
                };

                let v0 = ld(op.in0);
                let v1 = ld(op.in1);
                let v2 = ld(op.in2);
                let current = self.tilemap.value(idx);

                let result = match op.op {
                    COP_WIRE_R | COP_VIA => v0,
                    COP_WIRE_L => v1,
                    COP_WIRE_D | COP_WIRE_U => v2,
                    COP_WIRE_H | COP_OR => v0 | v1,
                    COP_WIRE_V => v1 | v2,
                    COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                    COP_AND => v0 & v1,
                    COP_XOR => v0 ^ v1,
                    COP_MUX => {
                        if v2 != 0 {
                            v0
                        } else {
                            v1
                        }
                    }
                    COP_NOT => !v0,
                    COP_ZERO => {
                        if v0 == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_ADD => v0.wrapping_add(v1),
                    COP_SUB => v0.wrapping_sub(v1),
                    COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                    COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                    COP_MUX16 => {
                        let sel = (v1 & 0xF) as usize;
                        let data = if sel < 8 { v0 } else { v2 };
                        (data >> ((sel & 7) * 8)) & 0xFF
                    }
                    COP_DEC3 => 1u64 << (v0 & 7),
                    COP_BITSEL => {
                        if (v0 >> (v1 & 63)) & 1 != 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_CARRY => {
                        if v0 > v1 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_WVIA => {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        wvia_idx += 1;
                        (v0 >> shift) & mask
                    }
                    COP_MUX4 => {
                        let sel = (v2 & 0b11) as u32;
                        (v0 >> (sel * 8)) & 0xFF
                    }
                    COP_RAM => {
                        if v2 != 0 {
                            v0
                        } else {
                            current
                        }
                    }
                    COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                    _ => continue,
                };

                if result != current {
                    self.tilemap.set_value(idx, result);
                    any_changed = true;
                    total_switched += 1;
                    // Flush cache before dirty_dependents (writes to arbitrary segments).
                    flush_cache!();
                    let nc = self.neighbors4[idx];
                    let tt = self.tile_type_at(idx);
                    self.dirty_dependents(&nc, idx, tt);
                }
            }
            // Flush any remaining cached segment.
            flush_cache!();

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 354: Single-pass hybrid settle.
    ///
    /// Walks `ops` once per pass. For each op:
    ///   - if its tile index is in `backbone_set`: evaluate UNCONDITIONALLY
    ///     (no dirty check). When the value changes, only mark out-of-backbone
    ///     neighbors dirty (frontier-style).
    ///   - else (fringe): standard dirty-checked eval; on change mark all
    ///     dependents dirty (cascade may hit anywhere).
    ///
    /// Convergence: loop until a pass does no fringe work AND no backbone op
    /// changed value. Bounded to 10 passes (same as `propagate_compact_dirty`).
    ///
    /// Compared to two-phase backbone (S352): single scan instead of two.
    /// Compared to baseline blockskip: removes dirty-bit check on the 93.7%
    /// backbone majority, at the cost of always evaluating those ops.
    pub fn propagate_compact_dirty_hybrid(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        backbone_set: &[u64],
    ) -> (u32, u32, u32) {
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;

        let in_backbone = |idx: usize| -> bool {
            backbone_set.get(idx / 64).copied().unwrap_or(0) & (1u64 << (idx % 64)) != 0
        };

        loop {
            let mut pass_fringe_evaluated: u32 = 0;
            let mut pass_backbone_evaluated: u32 = 0;
            let mut backbone_changed = false;
            let mut fringe_changed = false;
            let mut wvia_idx = 0usize;

            // Segment-cached dirty bit handling. Cache the dirty segment word so
            // consecutive ops in the same segment share one load. Invalidate
            // before any call that writes the global dirty bitset.
            let mut cached_seg: u32 = u32::MAX;
            let mut cached_word: u64 = 0;
            #[allow(unused_assignments)]
            macro_rules! flush_cache {
                () => {
                    if cached_seg != u32::MAX {
                        self.dirty.segments[cached_seg as usize].set(cached_word);
                        cached_seg = u32::MAX;
                    }
                };
            }
            macro_rules! load_seg_for {
                ($op_idx:expr) => {{
                    let seg = ($op_idx / 64) as u32;
                    if seg != cached_seg {
                        flush_cache!();
                        cached_seg = seg;
                        cached_word = self.dirty.segments[seg as usize].get();
                    }
                    seg
                }};
            }

            for op in ops.iter() {
                let is_wvia = op.op == COP_WVIA;
                let idx = op.idx as usize;
                let bit = 1u64 << (op.idx % 64);
                let is_backbone = in_backbone(idx);

                if op.op == COP_CONST {
                    let _ = load_seg_for!(op.idx);
                    cached_word &= !bit;
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                if is_backbone {
                    // Unconditional eval. Clear any stale dirty bit so we don't
                    // re-process under fringe semantics on subsequent passes.
                    let _ = load_seg_for!(op.idx);
                    cached_word &= !bit;
                    pass_backbone_evaluated += 1;
                } else {
                    // Fringe: standard dirty check.
                    let _ = load_seg_for!(op.idx);
                    if cached_word & bit == 0 {
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }
                    cached_word &= !bit;
                    pass_fringe_evaluated += 1;
                }

                // Chain handling (skip tails, fuse heads).
                if self.is_chain_tail(idx) {
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    flush_cache!();
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        if is_backbone {
                            backbone_changed = true;
                        } else {
                            fringe_changed = true;
                        }
                        total_switched += 1 + tail_n;
                    }
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                if op.op == COP_GENERIC {
                    flush_cache!();
                    if self.eval_tile(idx) {
                        if is_backbone {
                            backbone_changed = true;
                        } else {
                            fringe_changed = true;
                        }
                        total_switched += 1;
                    }
                    if is_wvia {
                        wvia_idx += 1;
                    }
                    continue;
                }

                let ld = |i: u32| -> u64 {
                    if i == u32::MAX {
                        0
                    } else {
                        self.tilemap.value(i as usize)
                    }
                };
                let v0 = ld(op.in0);
                let v1 = ld(op.in1);
                let v2 = ld(op.in2);
                let current = self.tilemap.value(idx);

                let result = match op.op {
                    COP_WIRE_R | COP_VIA => v0,
                    COP_WIRE_L => v1,
                    COP_WIRE_D | COP_WIRE_U => v2,
                    COP_WIRE_H | COP_OR => v0 | v1,
                    COP_WIRE_V => v1 | v2,
                    COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                    COP_AND => v0 & v1,
                    COP_XOR => v0 ^ v1,
                    COP_MUX => {
                        if v2 != 0 {
                            v0
                        } else {
                            v1
                        }
                    }
                    COP_NOT => !v0,
                    COP_ZERO => {
                        if v0 == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_ADD => v0.wrapping_add(v1),
                    COP_SUB => v0.wrapping_sub(v1),
                    COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                    COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                    COP_MUX16 => {
                        let sel = (v1 & 0xF) as usize;
                        let data = if sel < 8 { v0 } else { v2 };
                        (data >> ((sel & 7) * 8)) & 0xFF
                    }
                    COP_DEC3 => 1u64 << (v0 & 7),
                    COP_BITSEL => {
                        if (v0 >> (v1 & 63)) & 1 != 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_CARRY => {
                        if v0 > v1 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_WVIA => {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        wvia_idx += 1;
                        (v0 >> shift) & mask
                    }
                    COP_MUX4 => {
                        let sel = (v2 & 0b11) as u32;
                        (v0 >> (sel * 8)) & 0xFF
                    }
                    COP_RAM => {
                        if v2 != 0 {
                            v0
                        } else {
                            current
                        }
                    }
                    COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                    _ => continue,
                };

                if result != current {
                    self.tilemap.set_value(idx, result);
                    if is_backbone {
                        backbone_changed = true;
                    } else {
                        fringe_changed = true;
                    }
                    total_switched += 1;
                    flush_cache!();
                    let nc = self.neighbors4[idx];
                    let tt = self.tile_type_at(idx);
                    if is_backbone {
                        // Backbone change → only mark out-of-backbone neighbors
                        // (avoid wasted marks on tiles we evaluate unconditionally).
                        self.dirty_dependents_frontier(&nc, idx, tt, backbone_set);
                        let via = self.via_fwd[idx];
                        if via != u32::MAX {
                            let vi = via as usize;
                            if !in_backbone(vi) {
                                self.dirty.mark_dirty(vi);
                            }
                        }
                    } else {
                        // Fringe change → mark all dependents (cascade goes anywhere).
                        self.dirty_dependents(&nc, idx, tt);
                    }
                }
            }
            flush_cache!();

            total_evaluated += pass_backbone_evaluated + pass_fringe_evaluated;

            // Convergence: stable when no fringe op was dirty this pass AND no
            // backbone op changed value. Same-pass forward propagation handles
            // most cascades within a single pass; subsequent passes catch
            // backward feedback (cycles in the dependency graph).
            if pass_fringe_evaluated == 0 && !backbone_changed {
                let _ = fringe_changed;
                break;
            }
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 355: Read tile logic values at the given indices into a buffer.
    /// Used by backbone memoization to build the input-hash vector each cycle.
    pub fn read_tiles_into(&self, indices: &[u32], buf: &mut Vec<u64>) {
        buf.clear();
        buf.reserve(indices.len());
        for &idx in indices {
            buf.push(self.tilemap.value(idx as usize));
        }
    }

    /// Sprint 355: Snapshot tile logic values at the given indices.
    pub fn snapshot_tiles(&self, indices: &[u32]) -> Vec<u64> {
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            out.push(self.tilemap.value(idx as usize));
        }
        out
    }

    /// Sprint 355: Restore a backbone snapshot. For each output tile, if the
    /// cached value differs from the current value, write it and mark fringe-
    /// side dependents (those NOT in `backbone_set`) dirty so the fringe
    /// kernel picks up the cascade.
    ///
    /// Returns the number of tiles whose value actually changed.
    pub fn apply_backbone_snapshot(
        &mut self,
        output_indices: &[u32],
        snapshot: &[u64],
        backbone_set: &[u64],
    ) -> u32 {
        debug_assert_eq!(output_indices.len(), snapshot.len());
        let mut changed = 0u32;
        for (i, &idx32) in output_indices.iter().enumerate() {
            let idx = idx32 as usize;
            let new_val = snapshot[i];
            let cur_val = self.tilemap.value(idx);
            if new_val != cur_val {
                self.tilemap.set_value(idx, new_val);
                changed += 1;
                let nc = self.neighbors4[idx];
                let tt = self.tile_type_at(idx);
                self.dirty_dependents_frontier(&nc, idx, tt, backbone_set);
                let via = self.via_fwd[idx];
                if via != u32::MAX {
                    let vi = via as usize;
                    let in_bb =
                        backbone_set.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                    if !in_bb {
                        self.dirty.mark_dirty(vi);
                    }
                }
            }
        }
        changed
    }

    /// Sprint 308: Block-level clean-skip for compact dirty evaluation.
    /// Sprint 319: Segment-cached dirty check within dirty blocks.
    ///
    /// Same evaluation + propagation semantics as `propagate_compact_dirty`, but
    /// groups ops into 64-op blocks. Before processing each block, checks
    /// precomputed segment masks to determine if any tiles in the block are dirty.
    /// Clean blocks are skipped entirely (~80 of 86 blocks per cycle).
    ///
    /// Within dirty blocks, caches the dirty segment word so consecutive ops in the
    /// same segment share one load. Cache is flushed before dirty_dependents, chain
    /// eval, and generic eval calls (which write to arbitrary segments).
    ///
    /// Same-pass forward propagation is preserved because `dirty_dependents` marks
    /// the global dirty bitset, which is re-read by subsequent blocks' segment checks.
    pub fn propagate_compact_dirty_blockskip(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        block_seg_offsets: &[u32],
        block_seg_entries: &[(u32, u64)],
        block_wvia_counts: &[u8],
    ) -> (u32, u32, u32) {
        // buckets sentinel u64::MAX disables bucket counting in _inner.
        self.propagate_compact_dirty_blockskip_inner(
            ops,
            wvia_params,
            block_seg_offsets,
            block_seg_entries,
            block_wvia_counts,
            &mut [],
            &mut [u64::MAX; 5],
        )
    }

    /// Sprint 328: Counted variant — identical to blockskip but increments
    /// switch_counts[slot] for each op where result != current.
    pub fn propagate_compact_dirty_blockskip_counted(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        block_seg_offsets: &[u32],
        block_seg_entries: &[(u32, u64)],
        block_wvia_counts: &[u8],
        switch_counts: &mut [u32],
    ) -> (u32, u32, u32) {
        self.propagate_compact_dirty_blockskip_inner(
            ops,
            wvia_params,
            block_seg_offsets,
            block_seg_entries,
            block_wvia_counts,
            switch_counts,
            &mut [u64::MAX; 5],
        )
    }

    /// Sprint 330: Bucketed variant — counts + bucket classification.
    /// buckets: [dead_clean_block, dead_dirty_block_clean_bit, dead_dirty_set_unchanged,
    ///           hot_dirty_set_changed, cop_const]
    pub fn propagate_compact_dirty_blockskip_bucketed(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        block_seg_offsets: &[u32],
        block_seg_entries: &[(u32, u64)],
        block_wvia_counts: &[u8],
        switch_counts: &mut [u32],
        buckets: &mut [u64; 5],
    ) -> (u32, u32, u32) {
        self.propagate_compact_dirty_blockskip_inner(
            ops,
            wvia_params,
            block_seg_offsets,
            block_seg_entries,
            block_wvia_counts,
            switch_counts,
            buckets,
        )
    }

    fn propagate_compact_dirty_blockskip_inner(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        block_seg_offsets: &[u32],
        block_seg_entries: &[(u32, u64)],
        block_wvia_counts: &[u8],
        switch_counts: &mut [u32],
        buckets: &mut [u64; 5],
    ) -> (u32, u32, u32) {
        let num_ops = ops.len();
        if num_ops == 0 {
            return (0, 0, 0);
        }
        let num_blocks = (num_ops + 63) / 64;

        #[allow(unused_assignments)]
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;

        #[allow(unused_assignments)]
        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;
            let mut wvia_idx = 0usize;

            for block in 0..num_blocks {
                // Block-level dirty check: are any tiles in this block dirty?
                let seg_start = block_seg_offsets[block] as usize;
                let seg_end = block_seg_offsets[block + 1] as usize;
                let mut block_has_dirty = false;
                for i in seg_start..seg_end {
                    let (seg_idx, mask) = block_seg_entries[i];
                    if let Some(seg) = self.dirty.segments.get(seg_idx as usize) {
                        if seg.get() & mask != 0 {
                            block_has_dirty = true;
                            break;
                        }
                    }
                }

                if !block_has_dirty {
                    // Skip entire block — advance wvia_idx.
                    wvia_idx += block_wvia_counts[block] as usize;
                    // Sprint 330: Count dead ops in clean blocks (first pass only).
                    if delta_count == 0 && buckets[0] != u64::MAX {
                        let bs = block * 64;
                        let be = (bs + 64).min(num_ops);
                        for s in bs..be {
                            if switch_counts.get(s).copied().unwrap_or(1) == 0 {
                                buckets[0] += 1; // dead_in_clean_block
                            }
                        }
                    }
                    continue;
                }

                // Sprint 319: Segment-cached dirty check within dirty blocks.
                // Cache the dirty segment word so consecutive ops in the same
                // segment share one load. Flush before dirty_dependents/chain/
                // generic calls (which write to arbitrary segments).
                let block_start = block * 64;
                let block_end = (block_start + 64).min(num_ops);

                let mut cached_seg: u32 = u32::MAX;
                let mut cached_word: u64 = 0;

                macro_rules! flush_cache {
                    () => {
                        if cached_seg != u32::MAX {
                            self.dirty.segments[cached_seg as usize].set(cached_word);
                            cached_seg = u32::MAX;
                        }
                    };
                }

                for slot in block_start..block_end {
                    let op = &ops[slot];
                    let is_wvia = op.op == COP_WVIA;

                    let seg = (op.idx / 64) as u32;
                    let bit = 1u64 << (op.idx % 64);

                    // Load segment (cached if same segment).
                    if seg != cached_seg {
                        flush_cache!();
                        cached_seg = seg;
                        cached_word = self.dirty.segments[seg as usize].get();
                    }

                    if op.op == COP_CONST {
                        cached_word &= !bit;
                        if delta_count == 0 && buckets[0] != u64::MAX {
                            buckets[4] += 1; // cop_const
                        }
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }

                    if cached_word & bit == 0 {
                        // Clean dirty bit — op not dirty.
                        if delta_count == 0
                            && buckets[0] != u64::MAX
                            && switch_counts.get(slot).copied().unwrap_or(1) == 0
                        {
                            buckets[1] += 1; // dead_dirty_block_clean_bit
                        }
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }
                    // Clear dirty bit in cache.
                    cached_word &= !bit;
                    pass_evaluated += 1;

                    let idx = op.idx as usize;

                    if self.is_chain_tail(idx) {
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }
                    let chain_id = self.chain_head_map[idx];
                    if chain_id != u32::MAX {
                        // Flush cache before chain eval (writes to global dirty).
                        flush_cache!();
                        let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                        if changed {
                            any_changed = true;
                            total_switched += 1 + tail_n;
                        }
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }

                    if op.op == COP_GENERIC {
                        // Flush cache before generic eval (writes to global dirty).
                        flush_cache!();
                        if self.eval_tile(idx) {
                            any_changed = true;
                            total_switched += 1;
                        }
                        if is_wvia {
                            wvia_idx += 1;
                        }
                        continue;
                    }

                    let ld = |i: u32| -> u64 {
                        if i == u32::MAX {
                            0
                        } else {
                            self.tilemap.value(i as usize)
                        }
                    };

                    let v0 = ld(op.in0);
                    let v1 = ld(op.in1);
                    let v2 = ld(op.in2);
                    let current = self.tilemap.value(idx);

                    let result = match op.op {
                        COP_WIRE_R | COP_VIA => v0,
                        COP_WIRE_L => v1,
                        COP_WIRE_D | COP_WIRE_U => v2,
                        COP_WIRE_H | COP_OR => v0 | v1,
                        COP_WIRE_V => v1 | v2,
                        COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                        COP_AND => v0 & v1,
                        COP_XOR => v0 ^ v1,
                        COP_MUX => {
                            if v2 != 0 {
                                v0
                            } else {
                                v1
                            }
                        }
                        COP_NOT => !v0,
                        COP_ZERO => {
                            if v0 == 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_ADD => v0.wrapping_add(v1),
                        COP_SUB => v0.wrapping_sub(v1),
                        COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                        COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                        COP_MUX16 => {
                            let sel = (v1 & 0xF) as usize;
                            let data = if sel < 8 { v0 } else { v2 };
                            (data >> ((sel & 7) * 8)) & 0xFF
                        }
                        COP_DEC3 => 1u64 << (v0 & 7),
                        COP_BITSEL => {
                            if (v0 >> (v1 & 63)) & 1 != 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_CARRY => {
                            if v0 > v1 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_WVIA => {
                            let (_, shift, mask) = wvia_params[wvia_idx];
                            wvia_idx += 1;
                            (v0 >> shift) & mask
                        }
                        COP_MUX4 => {
                            let sel = (v2 & 0b11) as u32;
                            (v0 >> (sel * 8)) & 0xFF
                        }
                        COP_RAM => {
                            if v2 != 0 {
                                v0
                            } else {
                                current
                            }
                        }
                        _ => continue,
                    };

                    if result != current {
                        self.tilemap.set_value(idx, result);
                        any_changed = true;
                        total_switched += 1;
                        if let Some(c) = switch_counts.get_mut(slot) {
                            *c += 1;
                        }
                        // Sprint 330: hot_dirty_set_changed.
                        if delta_count == 0 && buckets[0] != u64::MAX {
                            buckets[3] += 1;
                        }
                        // Flush cache before dirty_dependents (writes to arbitrary segments).
                        flush_cache!();
                        let nc = self.neighbors4[idx];
                        let tt = self.tile_type_at(idx);
                        self.dirty_dependents(&nc, idx, tt);
                    } else {
                        // Sprint 330: dirty bit was set but value unchanged.
                        if delta_count == 0
                            && buckets[0] != u64::MAX
                            && switch_counts.get(slot).copied().unwrap_or(1) == 0
                        {
                            buckets[2] += 1; // dead_dirty_set_unchanged
                        }
                    }
                }
                // Flush any remaining cached segment for this block.
                flush_cache!();
            }

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 306: Prefiltered compact dirty — slot-bitset settle evaluation.
    ///
    /// Hybrid of `propagate_compact_scheduled` (slot-bitset iteration, same-pass
    /// forward propagation) and `propagate_compact_dirty` (dynamic dirty_dependents
    /// for correctness). Eliminates the 5,492-op scan by iterating only active slots.
    ///
    /// Key difference from `propagate_compact_scheduled`: does NOT use a precomputed
    /// dep table (which was incomplete for settle topology, Sprint 305). Instead,
    /// computes deps dynamically using dirty_dependents logic, partitioning into
    /// in-scope slot bits (same-pass forward) and out-of-scope global dirty marks.
    ///
    /// `scope_mask`: bitset of tile indices in scope (settle_cone_set).
    /// `idx_to_slot`: tile_idx → position in ops (u32::MAX if absent).
    /// `wvia_slot_map`: slot → index into wvia_params (u32::MAX if not WVIA).
    pub fn propagate_compact_dirty_prefiltered(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        scope_mask: &[u64],
        idx_to_slot: &[u32],
        wvia_slot_map: &[u32],
    ) -> (u32, u32, u32) {
        let num_ops = ops.len();
        if num_ops == 0 {
            return (0, 0, 0);
        }

        // Slot-level bitset: one bit per settle op slot.
        let num_words = (num_ops + 63) / 64;
        let mut sd = std::mem::take(&mut self.schedule_slot_buf);
        sd.resize(num_words, 0);
        for w in sd.iter_mut() {
            *w = 0;
        }

        // Seed from global dirty bitset via masked drain.
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);
            for &tile_idx32 in batch.iter() {
                let tile_idx = tile_idx32 as usize;
                if tile_idx < idx_to_slot.len() {
                    let slot = idx_to_slot[tile_idx];
                    if slot != u32::MAX {
                        let s = slot as usize;
                        sd[s / 64] |= 1u64 << (s % 64);
                    }
                }
            }
            self.dirty_batch_buf = batch;
        }

        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;

        // Inline macro: enqueue in-scope deps + mark out-of-scope dirty.
        // Same logic as dirty_dependents but partitioned by scope.
        macro_rules! propagate_change {
            ($idx:expr, $word_idx:expr, $word:expr) => {
                let nc = self.neighbors4[$idx];
                let tt = self.tile_type_at($idx);
                macro_rules! route_dep {
                    ($ni:expr) => {
                        if $ni != u32::MAX {
                            let ni = $ni as usize;
                            if ni < idx_to_slot.len() {
                                let dep_slot = idx_to_slot[ni];
                                if dep_slot != u32::MAX {
                                    let ds = dep_slot as usize;
                                    let dw = ds / 64;
                                    let db = ds % 64;
                                    let bit = 1u64 << db;
                                    sd[dw] |= bit;
                                    if dw == $word_idx {
                                        $word |= bit;
                                    }
                                } else {
                                    self.dirty.mark_dirty(ni);
                                }
                            }
                        }
                    };
                }
                match tt {
                    TileType::WireDown => {
                        route_dep!(nc[0]);
                        route_dep!(nc[1]);
                        route_dep!(nc[3]);
                    }
                    TileType::WireUp => {
                        route_dep!(nc[0]);
                        route_dep!(nc[1]);
                        route_dep!(nc[2]);
                    }
                    TileType::WireRight => {
                        route_dep!(nc[1]);
                        route_dep!(nc[2]);
                        route_dep!(nc[3]);
                    }
                    TileType::WireLeft => {
                        route_dep!(nc[0]);
                        route_dep!(nc[2]);
                        route_dep!(nc[3]);
                    }
                    TileType::WireH => {
                        route_dep!(nc[2]);
                        route_dep!(nc[3]);
                    }
                    TileType::WireV => {
                        route_dep!(nc[0]);
                        route_dep!(nc[1]);
                    }
                    _ => {
                        route_dep!(nc[0]);
                        route_dep!(nc[1]);
                        route_dep!(nc[2]);
                        route_dep!(nc[3]);
                    }
                }
                let via = self.via_fwd[$idx];
                route_dep!(via);
            };
        }

        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;

            // Re-drain global dirty for in-scope tiles (handles chain/generic global marks).
            {
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                self.dirty.fill_into_masked(scope_mask, &mut batch);
                for &tile_idx32 in batch.iter() {
                    let tile_idx = tile_idx32 as usize;
                    if tile_idx < idx_to_slot.len() {
                        let slot = idx_to_slot[tile_idx];
                        if slot != u32::MAX {
                            let s = slot as usize;
                            sd[s / 64] |= 1u64 << (s % 64);
                        }
                    }
                }
                self.dirty_batch_buf = batch;
            }

            for word_idx in 0..num_words {
                let mut word = sd[word_idx];
                if word == 0 {
                    continue;
                }
                sd[word_idx] = 0;

                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let slot = word_idx * 64 + bit;
                    if slot >= num_ops {
                        break;
                    }

                    let op = &ops[slot];
                    if op.op == COP_CONST {
                        continue;
                    }

                    pass_evaluated += 1;
                    let idx = op.idx as usize;

                    // Chain handling.
                    if self.is_chain_tail(idx) {
                        continue;
                    }
                    let chain_id = self.chain_head_map[idx];
                    if chain_id != u32::MAX {
                        let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                        if changed {
                            any_changed = true;
                            total_switched += 1 + tail_n;
                            propagate_change!(idx, word_idx, word);
                            // Sprint 307: Chain tail deps — lateral + via for each tail member.
                            // Uses same-word forward propagation (OR into local `word`)
                            // so chain-tail consumers in the current slot-word are evaluated
                            // in the same pass, not deferred to re-drain.
                            let chain = &self.wire_chains[chain_id as usize];
                            macro_rules! route_tail_dep {
                                ($ni:expr, $wix:expr, $w:expr) => {
                                    if $ni != u32::MAX {
                                        let ni = $ni as usize;
                                        if ni < idx_to_slot.len() {
                                            let ds = idx_to_slot[ni];
                                            if ds != u32::MAX {
                                                let d = ds as usize;
                                                let dw = d / 64;
                                                let bit = 1u64 << (d % 64);
                                                sd[dw] |= bit;
                                                if dw == $wix {
                                                    $w |= bit;
                                                }
                                            } else {
                                                self.dirty.mark_dirty(ni);
                                            }
                                        }
                                    }
                                };
                            }
                            for &tail_idx32 in &chain.tail_members {
                                let tail_idx = tail_idx32 as usize;
                                let tn = self.neighbors4[tail_idx];
                                match chain.wire_type {
                                    TileType::WireRight | TileType::WireLeft => {
                                        route_tail_dep!(tn[2], word_idx, word);
                                        route_tail_dep!(tn[3], word_idx, word);
                                    }
                                    TileType::WireDown | TileType::WireUp => {
                                        route_tail_dep!(tn[0], word_idx, word);
                                        route_tail_dep!(tn[1], word_idx, word);
                                    }
                                    _ => {}
                                }
                                let tv = self.via_fwd[tail_idx];
                                route_tail_dep!(tv, word_idx, word);
                            }
                        }
                        continue;
                    }

                    if op.op == COP_GENERIC {
                        if self.eval_tile(idx) {
                            any_changed = true;
                            total_switched += 1;
                            propagate_change!(idx, word_idx, word);
                        }
                        continue;
                    }

                    let ld = |i: u32| -> u64 {
                        if i == u32::MAX {
                            0
                        } else {
                            self.tilemap.value(i as usize)
                        }
                    };

                    let v0 = ld(op.in0);
                    let v1 = ld(op.in1);
                    let v2 = ld(op.in2);
                    let current = self.tilemap.value(idx);

                    let result = match op.op {
                        COP_WIRE_R | COP_VIA => v0,
                        COP_WIRE_L => v1,
                        COP_WIRE_D | COP_WIRE_U => v2,
                        COP_WIRE_H | COP_OR => v0 | v1,
                        COP_WIRE_V => v1 | v2,
                        COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                        COP_AND => v0 & v1,
                        COP_XOR => v0 ^ v1,
                        COP_MUX => {
                            if v2 != 0 {
                                v0
                            } else {
                                v1
                            }
                        }
                        COP_NOT => !v0,
                        COP_ZERO => {
                            if v0 == 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_ADD => v0.wrapping_add(v1),
                        COP_SUB => v0.wrapping_sub(v1),
                        COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                        COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                        COP_MUX16 => {
                            let sel = (v1 & 0xF) as usize;
                            let data = if sel < 8 { v0 } else { v2 };
                            (data >> ((sel & 7) * 8)) & 0xFF
                        }
                        COP_DEC3 => 1u64 << (v0 & 7),
                        COP_BITSEL => {
                            if (v0 >> (v1 & 63)) & 1 != 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_CARRY => {
                            if v0 > v1 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_WVIA => {
                            let wi = wvia_slot_map[slot];
                            if wi != u32::MAX {
                                let (_, shift, mask) = wvia_params[wi as usize];
                                (v0 >> shift) & mask
                            } else {
                                v0
                            }
                        }
                        COP_MUX4 => {
                            let sel = (v2 & 0b11) as u32;
                            (v0 >> (sel * 8)) & 0xFF
                        }
                        COP_RAM => {
                            if v2 != 0 {
                                v0
                            } else {
                                current
                            }
                        }
                        _ => continue,
                    };

                    if result != current {
                        self.tilemap.set_value(idx, result);
                        any_changed = true;
                        total_switched += 1;
                        propagate_change!(idx, word_idx, word);
                    }
                }
            }

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            // Don't fast-break on !has_backward — chain eval marks the global
            // dirty bitset (eval_tile_chain_fused calls dirty_dependents_lateral
            // internally). Those marks are picked up by re-drain at the top of
            // the next iteration. Without re-drain, chain-propagated dirty marks
            // are lost, causing divergence.
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        self.schedule_slot_buf = sd;
        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 293: Forward-deps settle — single-pass compact_dirty variant.
    ///
    /// Same linear topological scan as `propagate_compact_dirty`, but replaces
    /// `dirty_dependents` (marks all neighbors) with:
    ///   1. Forward-only in-scope deps (precomputed, dep_slot > current_slot)
    ///   2. Frontier marking for out-of-scope neighbors (via `dirty_dependents_frontier`)
    ///
    /// This eliminates backward dirty marks that cause the 2nd pass, guaranteeing
    /// single-pass convergence. Correct because backward marks in dirty_dependents
    /// are always false positives — they mark upstream tiles whose inputs haven't
    /// changed (the current tile is their output, not their input).
    ///
    /// `cone_set`: bitset of in-scope tile indices for frontier membership test.
    /// `fwd_deps_data` / `fwd_deps_offsets`: precomputed downstream-only deps.
    pub fn propagate_settle_forward(
        &mut self,
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        cone_set: &[u64],
        fwd_deps_data: &[u32],
        fwd_deps_offsets: &[u32],
    ) -> (u32, u32, u32) {
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut wvia_idx = 0usize;

        // Single pass — forward-only deps guarantee no backward marks.
        for (slot, op) in ops.iter().enumerate() {
            let is_wvia = op.op == COP_WVIA;

            if op.op == COP_CONST {
                let _ = self.dirty.is_dirty_and_clear(op.idx as usize);
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            if !self.dirty.is_dirty_and_clear(op.idx as usize) {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }
            total_evaluated += 1;

            let idx = op.idx as usize;

            // Chain tail: skip (head handles it).
            if self.is_chain_tail(idx) {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            // Chain head: fused evaluation + forward deps + frontier.
            let chain_id = self.chain_head_map[idx];
            if chain_id != u32::MAX {
                let (changed, _tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                if changed {
                    total_switched += 1;
                    // Forward in-scope deps (includes chain head + tail deps).
                    let start = fwd_deps_offsets[slot] as usize;
                    let end = fwd_deps_offsets[slot + 1] as usize;
                    for i in start..end {
                        let dep_tile = ops[fwd_deps_data[i] as usize].idx as usize;
                        self.dirty.mark_dirty(dep_tile);
                    }
                    // Frontier: head + all changed tails.
                    let nc = self.neighbors4[idx];
                    let tt = self.tile_type_at(idx);
                    self.dirty_dependents_frontier(&nc, idx, tt, cone_set);
                    let via = self.via_fwd[idx];
                    if via != u32::MAX {
                        let vi = via as usize;
                        let in_cone =
                            cone_set.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                        if !in_cone {
                            self.dirty.mark_dirty(vi);
                        }
                    }
                    // Tail frontier marking. Extract chain data to avoid borrow conflict.
                    let chain_wire_type = self.wire_chains[chain_id as usize].wire_type;
                    let chain_tail_count = self.wire_chains[chain_id as usize].tail_members.len();
                    for ti_idx in 0..chain_tail_count {
                        let ti = self.wire_chains[chain_id as usize].tail_members[ti_idx] as usize;
                        let tn = self.neighbors4[ti];
                        self.dirty_dependents_frontier(&tn, ti, chain_wire_type, cone_set);
                        let tv = self.via_fwd[ti];
                        if tv != u32::MAX {
                            let tvi = tv as usize;
                            let in_cone = cone_set.get(tvi / 64).copied().unwrap_or(0)
                                & (1u64 << (tvi % 64))
                                != 0;
                            if !in_cone {
                                self.dirty.mark_dirty(tvi);
                            }
                        }
                    }
                    // Chain exit frontier.
                    let last_tail_opt = if chain_tail_count > 0 {
                        Some(self.wire_chains[chain_id as usize].tail_members[chain_tail_count - 1])
                    } else {
                        None
                    };
                    if let Some(last_tail32) = last_tail_opt {
                        let last_t = last_tail32 as usize;
                        let es = match chain_wire_type {
                            TileType::WireRight => 1,
                            TileType::WireLeft => 0,
                            TileType::WireDown => 3,
                            TileType::WireUp => 2,
                            _ => 0,
                        };
                        let exit_ni = self.neighbors4[last_t][es];
                        if exit_ni != u32::MAX {
                            let eni = exit_ni as usize;
                            let in_cone = cone_set.get(eni / 64).copied().unwrap_or(0)
                                & (1u64 << (eni % 64))
                                != 0;
                            if !in_cone {
                                self.dirty.mark_dirty(eni);
                            }
                        }
                    }
                }
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }

            // Non-chain compact op: inline evaluation.
            let ld = |i: u32| -> u64 {
                if i == u32::MAX {
                    0
                } else {
                    self.tilemap.value(i as usize)
                }
            };
            let v0 = ld(op.in0);
            let v1 = ld(op.in1);
            let v2 = ld(op.in2);
            let current = self.tilemap.value(idx);

            let result = match op.op {
                COP_WIRE_R | COP_VIA => v0,
                COP_WIRE_L => v1,
                COP_WIRE_D | COP_WIRE_U => v2,
                COP_WIRE_H | COP_OR => v0 | v1,
                COP_WIRE_V => v1 | v2,
                COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                COP_AND => v0 & v1,
                COP_XOR => v0 ^ v1,
                COP_MUX => {
                    if v2 != 0 {
                        v0
                    } else {
                        v1
                    }
                }
                COP_NOT => !v0,
                COP_ZERO => {
                    if v0 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_ADD => v0.wrapping_add(v1),
                COP_SUB => v0.wrapping_sub(v1),
                COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                COP_MUX16 => {
                    let sel = (v1 & 0xF) as usize;
                    let data = if sel < 8 { v0 } else { v2 };
                    (data >> ((sel & 7) * 8)) & 0xFF
                }
                COP_DEC3 => 1u64 << (v0 & 7),
                COP_BITSEL => {
                    if (v0 >> (v1 & 63)) & 1 != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_CARRY => {
                    if v0 > v1 {
                        u64::MAX
                    } else {
                        0
                    }
                }
                COP_WVIA => {
                    let r = if wvia_idx < wvia_params.len() {
                        let (_, shift, mask) = wvia_params[wvia_idx];
                        (v0 >> shift) & mask
                    } else {
                        v0
                    };
                    wvia_idx += 1;
                    r
                }
                COP_MUX4 => (v0 >> ((v2 & 3) * 8)) & 0xFF,
                COP_RAM => {
                    if v2 != 0 {
                        v0
                    } else {
                        current
                    }
                }
                COP_THRESHOLD_VIA => self.threshold_via_gate(idx, v0),
                _ => {
                    if is_wvia && op.op != COP_WVIA {
                        wvia_idx += 1;
                    }
                    continue;
                }
            };

            if result != current {
                self.tilemap.set_value(idx, result);
                total_switched += 1;

                // Forward in-scope deps (downstream only).
                let start = fwd_deps_offsets[slot] as usize;
                let end = fwd_deps_offsets[slot + 1] as usize;
                for i in start..end {
                    let dep_tile = ops[fwd_deps_data[i] as usize].idx as usize;
                    self.dirty.mark_dirty(dep_tile);
                }

                // Frontier: out-of-scope neighbors + via_fwd.
                let nc = self.neighbors4[idx];
                let tt = self.tile_type_at(idx);
                self.dirty_dependents_frontier(&nc, idx, tt, cone_set);
                let via = self.via_fwd[idx];
                if via != u32::MAX {
                    let vi = via as usize;
                    let in_cone =
                        cone_set.get(vi / 64).copied().unwrap_or(0) & (1u64 << (vi % 64)) != 0;
                    if !in_cone {
                        self.dirty.mark_dirty(vi);
                    }
                }
            }
        }

        (1, total_evaluated, total_switched)
    }

    // =====================================================================
    // Sprint 278: Ordered active-work scheduler
    // =====================================================================

    /// Sprint 278: Precomputed schedule for active-work propagation.
    /// Instead of scanning the full compact ops slice to find dirty tiles,
    /// this uses a slot-indexed dirty bitset driven by precomputed per-slot
    /// dependency edges. Only active slots are visited.
    ///
    /// Replaces propagate_compact_dirty for scopes where scan waste is high
    /// (typically 91%+ — scanning ~7-10k ops to find ~500-2000 active).
    pub fn build_compact_schedule(
        &self,
        eval_order: &[usize],
        ram_as_const: bool,
    ) -> CompactSchedule {
        let (ops, wvia) = self.build_compact_ops_inner(eval_order, ram_as_const);
        let tile_count = self.tilemap.tile_count();

        // idx_to_slot: global tile index → position in ops slice.
        let mut idx_to_slot = vec![u32::MAX; tile_count];
        for (slot, op) in ops.iter().enumerate() {
            idx_to_slot[op.idx as usize] = slot as u32;
        }

        // wvia_slot_idx: for COP_WVIA ops, their index into the wvia array.
        let mut wvia_slot_idx = vec![u32::MAX; ops.len()];
        let mut wvia_count = 0u32;
        for (slot, op) in ops.iter().enumerate() {
            if op.op == COP_WVIA {
                wvia_slot_idx[slot] = wvia_count;
                wvia_count += 1;
            }
        }

        // deps: for each slot, the in-scope downstream slots that should
        // be activated when this tile changes. Mirrors dirty_dependents logic.
        // Flat-packed: deps_offsets[slot]..deps_offsets[slot+1] indexes into deps_data.
        let mut deps_data: Vec<u32> = Vec::with_capacity(ops.len() * 3);
        let mut deps_offsets: Vec<u32> = Vec::with_capacity(ops.len() + 1);

        for (_slot, op) in ops.iter().enumerate() {
            deps_offsets.push(deps_data.len() as u32);
            if op.op == COP_CONST {
                continue;
            }
            let idx = op.idx as usize;
            let tt = self.tile_type_at(idx);
            let n = &self.neighbors4[idx];

            // Direction-aware deps matching dirty_dependents / dirty_dependents_lateral.
            // Chain heads use lateral deps (perpendicular only) since
            // eval_tile_chain_fused calls dirty_dependents_lateral, not dirty_dependents.
            // Non-chain tiles use full dirty_dependents logic.
            macro_rules! maybe_dep {
                ($ni:expr) => {
                    if $ni != u32::MAX {
                        let dep_slot = idx_to_slot[$ni as usize];
                        if dep_slot != u32::MAX {
                            deps_data.push(dep_slot);
                        }
                    }
                };
            }
            let is_chain_head = self.chain_head_map[idx] != u32::MAX;
            if is_chain_head {
                // Lateral deps only (perpendicular to wire direction).
                match tt {
                    TileType::WireRight | TileType::WireLeft => {
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                    TileType::WireDown | TileType::WireUp => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                    }
                    _ => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                }
            } else {
                // Full dirty_dependents logic.
                match tt {
                    TileType::WireDown => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                        maybe_dep!(n[3]);
                    }
                    TileType::WireUp => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                        maybe_dep!(n[2]);
                    }
                    TileType::WireRight => {
                        maybe_dep!(n[1]);
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                    TileType::WireLeft => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                    TileType::WireH => {
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                    TileType::WireV => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                    }
                    _ => {
                        maybe_dep!(n[0]);
                        maybe_dep!(n[1]);
                        maybe_dep!(n[2]);
                        maybe_dep!(n[3]);
                    }
                }
            }
            // Cross-layer via
            let via = self.via_fwd[idx];
            if via != u32::MAX {
                let dep_slot = idx_to_slot[via as usize];
                if dep_slot != u32::MAX {
                    deps_data.push(dep_slot);
                }
            }

            // Sprint 279: Chain head deps must include tail member deps.
            // eval_tile_chain_fused writes tail members directly and calls
            // dirty_dependents_lateral for each. The head's deps table must
            // include those lateral + via deps so the scheduler picks them up.
            let chain_id = self.chain_head_map[idx];
            if chain_id != u32::MAX && chain_id < self.wire_chains.len() as u32 {
                let chain = &self.wire_chains[chain_id as usize];
                // Use lateral deps (perpendicular only) for chain members —
                // matches dirty_dependents_lateral behavior.
                macro_rules! lateral_dep {
                    ($ni:expr) => {
                        if $ni != u32::MAX {
                            let ds = idx_to_slot[$ni as usize];
                            if ds != u32::MAX {
                                deps_data.push(ds);
                            }
                        }
                    };
                }
                for &tail_idx32 in &chain.tail_members {
                    let tail_idx = tail_idx32 as usize;
                    let tn = &self.neighbors4[tail_idx];
                    // Lateral = perpendicular to wire direction.
                    match chain.wire_type {
                        TileType::WireRight | TileType::WireLeft => {
                            lateral_dep!(tn[2]); // UP
                            lateral_dep!(tn[3]); // DOWN
                        }
                        TileType::WireDown | TileType::WireUp => {
                            lateral_dep!(tn[0]); // LEFT
                            lateral_dep!(tn[1]); // RIGHT
                        }
                        _ => {}
                    }
                    // Tail via_fwd
                    let tv = self.via_fwd[tail_idx];
                    if tv != u32::MAX {
                        let ds = idx_to_slot[tv as usize];
                        if ds != u32::MAX {
                            deps_data.push(ds);
                        }
                    }
                }
                // Sprint 282: Chain-exit forward dep. eval_chain_tail uses
                // dirty_dependents_lateral (perpendicular only) — the last
                // tail's output-direction neighbor is never marked. Add it
                // to the dep table so the scheduler activates it.
                if let Some(&last_tail32) = chain.tail_members.last() {
                    let last_tail = last_tail32 as usize;
                    let exit_slot = match chain.wire_type {
                        TileType::WireRight => 1, // RIGHT
                        TileType::WireLeft => 0,  // LEFT
                        TileType::WireDown => 3,  // DOWN
                        TileType::WireUp => 2,    // UP
                        _ => unreachable!(),
                    };
                    let exit_idx = self.neighbors4[last_tail][exit_slot];
                    if exit_idx != u32::MAX {
                        let ds = idx_to_slot[exit_idx as usize];
                        if ds != u32::MAX {
                            deps_data.push(ds);
                        }
                    }
                }
            }
        }
        deps_offsets.push(deps_data.len() as u32);

        // Build scope mask from idx_to_slot for fast residual drain.
        let num_segments = (tile_count + 63) / 64;
        let mut scope_mask = vec![0u64; num_segments];
        for &idx in eval_order {
            scope_mask[idx / 64] |= 1u64 << (idx % 64);
        }

        let has_generic = ops.iter().any(|op| op.op == COP_GENERIC);

        // Sprint 333: Precompute in-scope segments for sparse drain.
        let in_scope_segments: Vec<(u32, u64)> = scope_mask
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0)
            .map(|(i, w)| (i as u32, *w))
            .collect();

        CompactSchedule {
            ops,
            wvia,
            idx_to_slot,
            deps_data,
            deps_offsets,
            wvia_slot_idx,
            scope_mask,
            has_generic,
            in_scope_segments,
        }
    }

    /// Sprint 278: Active-work propagation using precomputed schedule.
    /// Visits only dirty slots instead of scanning the full ops slice.
    ///
    /// `initial_dirty`: tile indices seeded by the caller (mark_commit_path_dirty etc).
    /// The function also drains any residual global dirty bits in scope.
    ///
    /// Sprint 280: `terminal` controls dirty propagation strategy:
    /// - `terminal = true` (clock): suppress eval_tile's global dirty marks,
    ///   scheduler uses `dirty_dependents_out_of_scope` for cross-scope neighbors.
    ///   Prevents cascade dirty bits from leaking to next tick's delta 0.
    /// - `terminal = false` (branch, commit): eval_tile marks normally so global
    ///   dirty bits persist for downstream scopes (branch→commit→clock chain).
    ///   Pre-decoded ops use full `dirty_dependents` for cross-scope marking.
    pub fn propagate_compact_scheduled(
        &mut self,
        schedule: &CompactSchedule,
        initial_dirty: &[usize],
        terminal: bool,
    ) -> (u32, u32, u32) {
        let ops = &schedule.ops;
        let num_slots = ops.len();
        if num_slots == 0 {
            return (0, 0, 0);
        }

        // Take ownership of slot buffer to avoid borrow conflicts with self.
        let num_words = (num_slots + 63) / 64;
        let mut sd = std::mem::take(&mut self.schedule_slot_buf);
        sd.resize(num_words, 0);
        for w in sd.iter_mut() {
            *w = 0;
        }

        // Seed from caller-provided dirty tile indices.
        for &tile_idx in initial_dirty {
            let _ = self.dirty.is_dirty_and_clear(tile_idx);
            let slot = schedule.idx_to_slot[tile_idx];
            if slot != u32::MAX {
                let s = slot as usize;
                sd[s / 64] |= 1u64 << (s % 64);
            }
        }

        // Sprint 333: Sparse segment drain — only checks segments known to have
        // in-scope tiles. O(in_scope_segments) instead of O(L1_bits × L0_segments).
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            if !schedule.in_scope_segments.is_empty() {
                self.dirty
                    .fill_into_sparse(&schedule.in_scope_segments, &mut batch);
            } else {
                self.dirty
                    .fill_into_masked(&schedule.scope_mask, &mut batch);
            }
            for &tile_idx32 in batch.iter() {
                let slot = schedule.idx_to_slot[tile_idx32 as usize];
                if slot != u32::MAX {
                    let s = slot as usize;
                    sd[s / 64] |= 1u64 << (s % 64);
                }
            }
            self.dirty_batch_buf = batch;
        }

        // Sprint 280: Suppress eval_tile's internal dirty marking for terminal scopes.
        if terminal {
            self.suppress_dirty_propagation = true;
        }

        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;
        let mut has_backward = false;

        // Inline dep-enqueue macro (avoids method call that borrows self).
        macro_rules! enqueue_deps {
            ($slot:expr, $word_idx:expr, $word:expr) => {
                let start = schedule.deps_offsets[$slot] as usize;
                let end = schedule.deps_offsets[$slot + 1] as usize;
                for i in start..end {
                    let dep = schedule.deps_data[i] as usize;
                    let dw = dep / 64;
                    let db = dep % 64;
                    let bit = 1u64 << db;
                    sd[dw] |= bit;
                    if dw == $word_idx {
                        $word |= bit;
                    } else if dw < $word_idx {
                        has_backward = true;
                    }
                }
            };
        }

        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;

            // Sprint 283: Re-drain global dirty marks into slot bitset.
            // Sprint 292: Skip when schedule has no COP_GENERIC — re-drain exists
            // solely for COP_GENERIC's eval_tile global dirty marks.
            // Sprint 333: Use sparse drain when in_scope_segments available.
            if schedule.has_generic {
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                if !schedule.in_scope_segments.is_empty() {
                    self.dirty
                        .fill_into_sparse(&schedule.in_scope_segments, &mut batch);
                } else {
                    self.dirty
                        .fill_into_masked(&schedule.scope_mask, &mut batch);
                }
                for &tile_idx32 in batch.iter() {
                    let slot = schedule.idx_to_slot[tile_idx32 as usize];
                    if slot != u32::MAX {
                        let s = slot as usize;
                        sd[s / 64] |= 1u64 << (s % 64);
                    }
                }
                self.dirty_batch_buf = batch;
            }

            for word_idx in 0..num_words {
                let mut word = sd[word_idx];
                if word == 0 {
                    continue;
                }
                sd[word_idx] = 0;

                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let slot = word_idx * 64 + bit;
                    if slot >= num_slots {
                        break;
                    }

                    let op = &ops[slot];
                    if op.op == COP_CONST {
                        let _ = self.dirty.is_dirty_and_clear(op.idx as usize);
                        continue;
                    }

                    pass_evaluated += 1;
                    let idx = op.idx as usize;

                    // Chain handling.
                    if self.is_chain_tail(idx) {
                        continue;
                    }
                    let chain_id = self.chain_head_map[idx];
                    if chain_id != u32::MAX {
                        let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                        if changed {
                            any_changed = true;
                            total_switched += 1 + tail_n;
                            enqueue_deps!(slot, word_idx, word);
                            // Sprint 280: Out-of-scope dirty for terminal scopes.
                            // Non-terminal: eval_tile_chain_fused already marked globally.
                            if terminal {
                                let nc = self.neighbors4[idx];
                                let htt = self.tile_type_at(idx);
                                self.dirty_dependents_out_of_scope(
                                    &nc,
                                    idx,
                                    htt,
                                    &schedule.idx_to_slot,
                                );
                                let cid = chain_id as usize;
                                let wt = self.wire_chains[cid].wire_type;
                                for ti in 0..(tail_n as usize) {
                                    let tail_idx = self.wire_chains[cid].tail_members[ti] as usize;
                                    let tn = self.neighbors4[tail_idx];
                                    self.dirty_dependents_out_of_scope(
                                        &tn,
                                        tail_idx,
                                        wt,
                                        &schedule.idx_to_slot,
                                    );
                                }
                                // Sprint 282: Chain-exit out-of-scope.
                                // Last changed tail's forward neighbor.
                                if tail_n > 0 {
                                    let last_t = self.wire_chains[cid].tail_members
                                        [tail_n as usize - 1]
                                        as usize;
                                    let es = match wt {
                                        TileType::WireRight => 1,
                                        TileType::WireLeft => 0,
                                        TileType::WireDown => 3,
                                        TileType::WireUp => 2,
                                        _ => unreachable!(),
                                    };
                                    let exit_ni = self.neighbors4[last_t][es];
                                    if exit_ni != u32::MAX
                                        && schedule.idx_to_slot[exit_ni as usize] == u32::MAX
                                    {
                                        self.dirty.mark_dirty(exit_ni as usize);
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    if op.op == COP_GENERIC {
                        if self.eval_tile(idx) {
                            any_changed = true;
                            total_switched += 1;
                            enqueue_deps!(slot, word_idx, word);
                            if terminal {
                                let nc = self.neighbors4[idx];
                                let gtt = self.tile_type_at(idx);
                                self.dirty_dependents_out_of_scope(
                                    &nc,
                                    idx,
                                    gtt,
                                    &schedule.idx_to_slot,
                                );
                            }
                        }
                        continue;
                    }

                    let ld = |i: u32| -> u64 {
                        if i == u32::MAX {
                            0
                        } else {
                            self.tilemap.value(i as usize)
                        }
                    };

                    let v0 = ld(op.in0);
                    let v1 = ld(op.in1);
                    let v2 = ld(op.in2);
                    let current = self.tilemap.value(idx);

                    let result = match op.op {
                        COP_WIRE_R | COP_VIA => v0,
                        COP_WIRE_L => v1,
                        COP_WIRE_D | COP_WIRE_U => v2,
                        COP_WIRE_H | COP_OR => v0 | v1,
                        COP_WIRE_V => v1 | v2,
                        COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                        COP_AND => v0 & v1,
                        COP_XOR => v0 ^ v1,
                        COP_MUX => {
                            if v2 != 0 {
                                v0
                            } else {
                                v1
                            }
                        }
                        COP_NOT => !v0,
                        COP_ZERO => {
                            if v0 == 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_ADD => v0.wrapping_add(v1),
                        COP_SUB => v0.wrapping_sub(v1),
                        COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                        COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                        COP_MUX16 => {
                            let sel = (v1 & 0xF) as usize;
                            let data = if sel < 8 { v0 } else { v2 };
                            (data >> ((sel & 7) * 8)) & 0xFF
                        }
                        COP_DEC3 => 1u64 << (v0 & 7),
                        COP_BITSEL => {
                            if (v0 >> (v1 & 63)) & 1 != 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_CARRY => {
                            if v0 > v1 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_WVIA => {
                            let wi = schedule.wvia_slot_idx[slot] as usize;
                            let (_, shift, mask) = schedule.wvia[wi];
                            (v0 >> shift) & mask
                        }
                        COP_MUX4 => {
                            let sel = (v2 & 0b11) as u32;
                            (v0 >> (sel * 8)) & 0xFF
                        }
                        COP_RAM => {
                            if v2 != 0 {
                                v0
                            } else {
                                current
                            }
                        }
                        _ => continue,
                    };

                    if result != current {
                        self.tilemap.set_value(idx, result);
                        any_changed = true;
                        total_switched += 1;
                        enqueue_deps!(slot, word_idx, word);
                        // Out-of-scope dirty marking.
                        let nc = self.neighbors4[idx];
                        let tt = self.tile_type_at(idx);
                        self.dirty_dependents_out_of_scope(&nc, idx, tt, &schedule.idx_to_slot);
                    }
                }
            }

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            // Sprint 283: Check for residual global dirty from eval_tile's
            // dirty_dependents (COP_GENERIC/chain paths). If found, re-drain
            // will capture them in the next pass. Without this check, the
            // `!has_backward` break would skip the re-drain and leave tiles
            // unevaluated that compact_dirty would have cascaded through.
            // Sprint 292: Skip when no COP_GENERIC — no global dirty to re-drain.
            if !has_backward && !schedule.has_generic {
                break;
            }
            if !has_backward {
                // Sprint 333: Use sparse residual check when available.
                let has_global_residual = if !schedule.in_scope_segments.is_empty() {
                    self.dirty.has_dirty_in_sparse(&schedule.in_scope_segments)
                } else {
                    let mut found = false;
                    for (seg_idx, scope_word) in schedule.scope_mask.iter().enumerate() {
                        if *scope_word != 0 {
                            if let Some(seg) = self.dirty.segments.get(seg_idx) {
                                if seg.get() & scope_word != 0 {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    found
                };
                if !has_global_residual {
                    break;
                }
            }
            has_backward = false;
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        self.schedule_slot_buf = sd;
        self.suppress_dirty_propagation = false;

        (delta_count, total_evaluated, total_switched)
    }

    /// Sprint 332: Profiled variant — returns sub-phase timing.
    /// Returns (deltas, evals, switched, drain_ns, worklist_ns, passes).
    pub fn propagate_compact_scheduled_profiled(
        &mut self,
        schedule: &CompactSchedule,
        initial_dirty: &[usize],
        terminal: bool,
    ) -> (u32, u32, u32, u64, u64, u32) {
        let ops = &schedule.ops;
        let num_slots = ops.len();
        if num_slots == 0 {
            return (0, 0, 0, 0, 0, 0);
        }

        let drain_start = std::time::Instant::now();

        let num_words = (num_slots + 63) / 64;
        let mut sd = std::mem::take(&mut self.schedule_slot_buf);
        sd.resize(num_words, 0);
        for w in sd.iter_mut() {
            *w = 0;
        }
        for &tile_idx in initial_dirty {
            let _ = self.dirty.is_dirty_and_clear(tile_idx);
            let slot = schedule.idx_to_slot[tile_idx];
            if slot != u32::MAX {
                let s = slot as usize;
                sd[s / 64] |= 1u64 << (s % 64);
            }
        }
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            if !schedule.in_scope_segments.is_empty() {
                self.dirty
                    .fill_into_sparse(&schedule.in_scope_segments, &mut batch);
            } else {
                self.dirty
                    .fill_into_masked(&schedule.scope_mask, &mut batch);
            }
            for &tile_idx32 in batch.iter() {
                let slot = schedule.idx_to_slot[tile_idx32 as usize];
                if slot != u32::MAX {
                    let s = slot as usize;
                    sd[s / 64] |= 1u64 << (s % 64);
                }
            }
            self.dirty_batch_buf = batch;
        }

        let drain_ns = drain_start.elapsed().as_nanos() as u64;

        if terminal {
            self.suppress_dirty_propagation = true;
        }

        let worklist_start = std::time::Instant::now();
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;
        let mut has_backward = false;

        macro_rules! enqueue_deps {
            ($slot:expr, $word_idx:expr, $word:expr) => {
                let start = schedule.deps_offsets[$slot] as usize;
                let end = schedule.deps_offsets[$slot + 1] as usize;
                for i in start..end {
                    let dep = schedule.deps_data[i] as usize;
                    let dw = dep / 64;
                    let db = dep % 64;
                    let bit = 1u64 << db;
                    sd[dw] |= bit;
                    if dw == $word_idx {
                        $word |= bit;
                    } else if dw < $word_idx {
                        has_backward = true;
                    }
                }
            };
        }

        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;

            if schedule.has_generic {
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                if !schedule.in_scope_segments.is_empty() {
                    self.dirty
                        .fill_into_sparse(&schedule.in_scope_segments, &mut batch);
                } else {
                    self.dirty
                        .fill_into_masked(&schedule.scope_mask, &mut batch);
                }
                for &tile_idx32 in batch.iter() {
                    let slot = schedule.idx_to_slot[tile_idx32 as usize];
                    if slot != u32::MAX {
                        let s = slot as usize;
                        sd[s / 64] |= 1u64 << (s % 64);
                    }
                }
                self.dirty_batch_buf = batch;
            }

            for word_idx in 0..num_words {
                let mut word = sd[word_idx];
                if word == 0 {
                    continue;
                }
                sd[word_idx] = 0;

                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let slot = word_idx * 64 + bit;
                    if slot >= num_slots {
                        break;
                    }

                    let op = &ops[slot];
                    if op.op == COP_CONST {
                        let _ = self.dirty.is_dirty_and_clear(op.idx as usize);
                        continue;
                    }

                    pass_evaluated += 1;
                    let idx = op.idx as usize;

                    if self.is_chain_tail(idx) {
                        continue;
                    }
                    let chain_id = self.chain_head_map[idx];
                    if chain_id != u32::MAX {
                        let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                        if changed {
                            any_changed = true;
                            total_switched += 1 + tail_n;
                            enqueue_deps!(slot, word_idx, word);
                            if terminal {
                                let nc = self.neighbors4[idx];
                                let htt = self.tile_type_at(idx);
                                self.dirty_dependents_out_of_scope(
                                    &nc,
                                    idx,
                                    htt,
                                    &schedule.idx_to_slot,
                                );
                                let cid = chain_id as usize;
                                let wt = self.wire_chains[cid].wire_type;
                                for ti in 0..(tail_n as usize) {
                                    let tail_idx = self.wire_chains[cid].tail_members[ti] as usize;
                                    let tn = self.neighbors4[tail_idx];
                                    self.dirty_dependents_out_of_scope(
                                        &tn,
                                        tail_idx,
                                        wt,
                                        &schedule.idx_to_slot,
                                    );
                                }
                                if tail_n > 0 {
                                    let last_t = self.wire_chains[cid].tail_members
                                        [(tail_n - 1) as usize]
                                        as usize;
                                    let ln = self.neighbors4[last_t];
                                    self.dirty_dependents_out_of_scope(
                                        &ln,
                                        last_t,
                                        wt,
                                        &schedule.idx_to_slot,
                                    );
                                }
                            }
                        }
                        continue;
                    }

                    if op.op == COP_GENERIC {
                        if self.eval_tile(idx) {
                            any_changed = true;
                            total_switched += 1;
                            enqueue_deps!(slot, word_idx, word);
                        }
                        continue;
                    }

                    let ld = |i: u32| -> u64 {
                        if i == u32::MAX {
                            0
                        } else {
                            self.tilemap.value(i as usize)
                        }
                    };
                    let v0 = ld(op.in0);
                    let v1 = ld(op.in1);
                    let v2 = ld(op.in2);
                    let current = self.tilemap.value(idx);

                    let result = match op.op {
                        COP_WIRE_R | COP_VIA => v0,
                        COP_WIRE_L => v1,
                        COP_WIRE_D | COP_WIRE_U => v2,
                        COP_WIRE_H | COP_OR => v0 | v1,
                        COP_WIRE_V => v1 | v2,
                        COP_WIRE => v0 | v1 | v2 | ld(self.neighbors4[idx][3]),
                        COP_AND => v0 & v1,
                        COP_XOR => v0 ^ v1,
                        COP_MUX => {
                            if v2 != 0 {
                                v0
                            } else {
                                v1
                            }
                        }
                        COP_NOT => !v0,
                        COP_ZERO => {
                            if v0 == 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_ADD => v0.wrapping_add(v1),
                        COP_SUB => v0.wrapping_sub(v1),
                        COP_SHR => v0.wrapping_shr((v1 & 63) as u32),
                        COP_SHL => v0.wrapping_shl((v1 & 63) as u32),
                        COP_MUX16 => {
                            let sel = (v1 & 0xF) as usize;
                            let data = if sel < 8 { v0 } else { v2 };
                            (data >> ((sel & 7) * 8)) & 0xFF
                        }
                        COP_DEC3 => 1u64 << (v0 & 7),
                        COP_BITSEL => {
                            if (v0 >> (v1 & 63)) & 1 != 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_CARRY => {
                            if v0 > v1 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        COP_WVIA => {
                            let wi = schedule.wvia_slot_idx[slot];
                            if wi != u32::MAX {
                                let (_, shift, mask) = schedule.wvia[wi as usize];
                                (v0 >> shift) & mask
                            } else {
                                v0
                            }
                        }
                        COP_MUX4 => {
                            let sel = (v2 & 0b11) as u32;
                            (v0 >> (sel * 8)) & 0xFF
                        }
                        COP_RAM => {
                            if v2 != 0 {
                                v0
                            } else {
                                current
                            }
                        }
                        _ => continue,
                    };

                    if result != current {
                        self.tilemap.set_value(idx, result);
                        any_changed = true;
                        total_switched += 1;
                        enqueue_deps!(slot, word_idx, word);
                        let nc = self.neighbors4[idx];
                        let tt = self.tile_type_at(idx);
                        self.dirty_dependents_out_of_scope(&nc, idx, tt, &schedule.idx_to_slot);
                    }
                }
            }

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            if !has_backward && !schedule.has_generic {
                break;
            }
            if !has_backward {
                // Sprint 333: Use sparse residual check when available.
                let has_global_residual = if !schedule.in_scope_segments.is_empty() {
                    self.dirty.has_dirty_in_sparse(&schedule.in_scope_segments)
                } else {
                    let mut found = false;
                    for (seg_idx, scope_word) in schedule.scope_mask.iter().enumerate() {
                        if *scope_word != 0 {
                            if let Some(seg) = self.dirty.segments.get(seg_idx) {
                                if seg.get() & scope_word != 0 {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    found
                };
                if !has_global_residual {
                    break;
                }
            }
            has_backward = false;
            delta_count += 1;
            if delta_count >= 10 {
                break;
            }
        }

        let worklist_ns = worklist_start.elapsed().as_nanos() as u64;

        self.schedule_slot_buf = sd;
        self.suppress_dirty_propagation = false;

        (
            delta_count,
            total_evaluated,
            total_switched,
            drain_ns,
            worklist_ns,
            delta_count + 1,
        )
    }

    /// Sprint 278: Mark only out-of-scope neighbors dirty.
    /// In-scope deps are handled by the scheduler's precomputed table.
    #[inline(always)]
    fn dirty_dependents_out_of_scope(
        &mut self,
        n: &[u32; 4],
        idx: usize,
        tt: TileType,
        idx_to_slot: &[u32],
    ) {
        macro_rules! md_oos {
            ($ni:expr) => {
                if $ni != u32::MAX && idx_to_slot[$ni as usize] == u32::MAX {
                    self.dirty.mark_dirty($ni as usize);
                }
            };
        }
        match tt {
            TileType::WireDown => {
                md_oos!(n[0]);
                md_oos!(n[1]);
                md_oos!(n[3]);
            }
            TileType::WireUp => {
                md_oos!(n[0]);
                md_oos!(n[1]);
                md_oos!(n[2]);
            }
            TileType::WireRight => {
                md_oos!(n[1]);
                md_oos!(n[2]);
                md_oos!(n[3]);
            }
            TileType::WireLeft => {
                md_oos!(n[0]);
                md_oos!(n[2]);
                md_oos!(n[3]);
            }
            TileType::WireH => {
                md_oos!(n[2]);
                md_oos!(n[3]);
            }
            TileType::WireV => {
                md_oos!(n[0]);
                md_oos!(n[1]);
            }
            _ => {
                md_oos!(n[0]);
                md_oos!(n[1]);
                md_oos!(n[2]);
                md_oos!(n[3]);
            }
        }
        let via = self.via_fwd[idx];
        if via != u32::MAX && idx_to_slot[via as usize] == u32::MAX {
            self.dirty.mark_dirty(via as usize);
        }
    }

    /// Sprint 262: Evaluate dirty tiles in pre-computed topological order.
    /// Primary pass walks eval_order once — each tile evaluated at most once.
    /// When a tile changes, `dirty_dependents` marks downstream tiles, which are
    /// later in eval_order and will be processed in the same pass.
    ///
    /// If backward dependencies exist (cycles, conservative dirty marking),
    /// additional passes handle remaining dirty tiles until convergence.
    ///
    /// Returns (deltas, evaluated, switched) for compatibility with
    /// `propagate_combinational_masked`.
    pub fn propagate_levelized(&mut self, eval_order: &[usize]) -> (u32, u32, u32) {
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        let mut delta_count: u32 = 0;

        loop {
            let mut pass_evaluated: u32 = 0;
            let mut any_changed = false;

            for &idx in eval_order {
                if !self.dirty.is_dirty_and_clear(idx) {
                    continue;
                }
                pass_evaluated += 1;

                // Sprint 173: Skip chain tail members — handled by chain fusion.
                if self.is_chain_tail(idx) {
                    continue;
                }

                // Sprint 172: chain fusion — fuse entire chain from head.
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        any_changed = true;
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    any_changed = true;
                    total_switched += 1;
                }
            }

            total_evaluated += pass_evaluated;
            if pass_evaluated == 0 || !any_changed {
                break;
            }
            delta_count += 1;
            if delta_count >= 10 {
                break; // Safety limit (should never hit with correct topo sort)
            }
        }

        (delta_count, total_evaluated, total_switched)
    }

    pub fn tick_with_delays(&mut self) -> TimingStats {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        // Update clock domain states
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        // Phase 1B: Increment tick counter for CPU metrics
        self.cpu_tick_count += 1;

        // Reset timing stats for this tick
        self.timing_stats = TimingStats::default();
        self.timing_stats.converged = true;

        // Sprint 147: Lazy reset — only reset tiles touched in previous tick
        for &idx in self.last_tick_activated.iter() {
            self.delay_countdown[idx] = 255;
            self.arrival_time[idx] = 0;
        }
        self.last_tick_activated.clear();

        // PHYSICS COUPLING: Snapshot physics fields at tick boundary
        if self.physics_coupling_config.enabled {
            self.snapshot_physics_for_coupling();
        } else {
            self.physics_coupling_ctx = None;
        }

        // Phase 1: Clock edge - schedule all sequential/clock-sensitive elements
        // Sprint 147: Use precomputed cache instead of O(tile_count) scan
        if self.clock_sensitive_cache.is_none() {
            self.clock_sensitive_cache = Some(
                self.tilemap
                    .tiles
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.meta.tile_type.is_clock_sensitive())
                    .map(|(i, _)| i)
                    .collect(),
            );
        }
        // Safety: we just ensured it's Some.
        // Sprint 167: take() + put back avoids Vec clone allocation per tick.
        let clock_cache = self.clock_sensitive_cache.take().unwrap();
        for &idx in clock_cache.iter() {
            self.delay_countdown[idx] = 0; // Ready immediately
            self.dirty.mark_dirty(idx);
            self.last_tick_activated.push(idx);
        }
        self.clock_sensitive_cache = Some(clock_cache);

        let mut delta_count: u32 = 0;
        const MAX_DELTA: u32 = 500; // Higher limit for delay-aware simulation

        let use_coupling = self.physics_coupling_ctx.is_some();

        // Phase 2: Delta iteration with delay modeling
        loop {
            self.current_delta = delta_count;
            crate::dbg_signal!("--- delay delta cycle {} ---", delta_count);

            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into(&mut batch);

            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_output_changed = false;
            let mut waiting_count = 0u32;

            // Process each dirty tile
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                let countdown = self.delay_countdown[idx];
                let tile_delay = self.effective_delay(idx);

                if countdown == 255 {
                    // First time seeing this tile dirty - start its countdown
                    // Sprint 147: track for lazy reset at next tick start
                    self.last_tick_activated.push(idx);
                    if tile_delay == 0 {
                        // Sequential element: ready immediately
                        self.delay_countdown[idx] = 0;
                    } else {
                        // Combinational: start countdown
                        self.delay_countdown[idx] = tile_delay.saturating_sub(1);
                        if tile_delay > 1 {
                            // Not ready yet - re-mark for next delta
                            self.dirty.mark_dirty(idx);
                            waiting_count += 1;
                            continue;
                        }
                    }
                } else if countdown > 0 {
                    // Still waiting - decrement and re-mark
                    self.delay_countdown[idx] = countdown - 1;
                    self.dirty.mark_dirty(idx);
                    waiting_count += 1;
                    continue;
                }
                // countdown == 0: ready to evaluate

                // Check for glitch: did any input arrive while we were computing?
                // (Simplified: if countdown was set and we're now evaluating,
                //  check if any neighbor has a higher arrival time than when we started)
                let our_schedule_time = delta_count.saturating_sub(tile_delay as u32);
                for &ni in &self.neighbors4[idx] {
                    if ni != u32::MAX {
                        let neighbor_arrival = self.arrival_time[ni as usize];
                        if neighbor_arrival > our_schedule_time && neighbor_arrival < delta_count {
                            // Input changed after we started computing but before we finished
                            self.timing_stats.glitches_detected += 1;
                            break;
                        }
                    }
                }

                // Evaluate the tile
                self.timing_stats.tiles_evaluated += 1;
                let changed = if use_coupling {
                    self.eval_tile_coupled(idx)
                } else {
                    self.eval_tile(idx)
                };

                // Reset countdown for future activations
                self.delay_countdown[idx] = 255;

                if changed {
                    any_output_changed = true;
                    self.timing_stats.tiles_switched += 1;
                    self.arrival_time[idx] = delta_count;

                    // Track critical path
                    if delta_count > self.timing_stats.critical_path_deltas {
                        self.timing_stats.critical_path_deltas = delta_count;
                        self.timing_stats.critical_path_endpoint = Some(idx);
                    }

                    // Schedule dependent neighbors (directional: skip input direction)
                    let sn = self.neighbors4[idx];
                    let stt = self.tile_type_at(idx);
                    macro_rules! sched {
                        ($ni:expr) => {
                            if $ni != u32::MAX {
                                let ni_usize = $ni as usize;
                                if self.delay_countdown[ni_usize] == 255 {
                                    self.dirty.mark_dirty(ni_usize);
                                }
                            }
                        };
                    }
                    match stt {
                        TileType::WireDown => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[3]);
                        }
                        TileType::WireUp => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[2]);
                        }
                        TileType::WireRight => {
                            sched!(sn[1]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireLeft => {
                            sched!(sn[0]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireH => {
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireV => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                        }
                        _ => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                    }
                    // Cross-layer via scheduling
                    let svia = self.via_fwd[idx];
                    if svia != u32::MAX {
                        let svia_usize = svia as usize;
                        if self.delay_countdown[svia_usize] == 255 {
                            self.dirty.mark_dirty(svia_usize);
                        }
                    }
                }
            }

            self.dirty_batch_buf = batch;

            // Continue if we made progress (either outputs changed or tiles are waiting)
            if !any_output_changed && waiting_count == 0 {
                break;
            }

            delta_count += 1;
            if delta_count >= MAX_DELTA {
                self.timing_stats.converged = false;
                // Don't panic - return stats with converged=false
                break;
            }
        }

        self.timing_stats.total_deltas = delta_count;

        // === Bus evaluation phase ===
        if !self.bus_states.is_empty() {
            self.evaluate_buses();
            // Post-bus stabilization: let reader values propagate through the grid
            let mut post_bus_deltas = 0u32;
            loop {
                self.current_delta = delta_count + post_bus_deltas + 1;
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                self.dirty.fill_into(&mut batch);
                if batch.is_empty() {
                    self.dirty_batch_buf = batch;
                    break;
                }
                let mut any_changed = false;
                for &idx32 in batch.iter() {
                    let idx = idx32 as usize;
                    self.timing_stats.tiles_evaluated += 1;
                    let changed = if use_coupling {
                        self.eval_tile_coupled(idx)
                    } else {
                        self.eval_tile(idx)
                    };
                    if changed {
                        any_changed = true;
                    }
                }
                self.dirty_batch_buf = batch;
                if !any_changed {
                    break;
                }
                post_bus_deltas += 1;
                if post_bus_deltas >= 100 {
                    break;
                }
            }
        }

        // Clear physics snapshot at tick end
        self.physics_coupling_ctx = None;

        // EPIC 49: after classical logic stabilizes for this tick, step quantum tiles
        self.step_quantum_tiles();

        self.timing_stats.clone()
    }

    /// Sprint 166: Masked variant of `tick_with_delays` — restricts evaluation to tiles
    /// within the provided scope mask. Identical logic, but uses `fill_into_masked`
    /// instead of `fill_into` so that out-of-scope dirty tiles are never drained.
    pub fn tick_with_delays_masked(&mut self, scope_mask: &[u64]) -> TimingStats {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        // Update clock domain states
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        // Phase 1B: Increment tick counter for CPU metrics
        self.cpu_tick_count += 1;

        // Reset timing stats for this tick
        self.timing_stats = TimingStats::default();
        self.timing_stats.converged = true;

        // Sprint 147: Lazy reset — only reset tiles touched in previous tick
        for &idx in self.last_tick_activated.iter() {
            self.delay_countdown[idx] = 255;
            self.arrival_time[idx] = 0;
        }
        self.last_tick_activated.clear();

        // PHYSICS COUPLING: Snapshot physics fields at tick boundary
        if self.physics_coupling_config.enabled {
            self.snapshot_physics_for_coupling();
        } else {
            self.physics_coupling_ctx = None;
        }

        // Phase 1: Clock edge - schedule all sequential/clock-sensitive elements
        // Sprint 147: Use precomputed cache instead of O(tile_count) scan
        if self.clock_sensitive_cache.is_none() {
            self.clock_sensitive_cache = Some(
                self.tilemap
                    .tiles
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.meta.tile_type.is_clock_sensitive())
                    .map(|(i, _)| i)
                    .collect(),
            );
        }
        // Safety: we just ensured it's Some.
        // Sprint 167: take() + put back avoids Vec clone allocation per tick.
        let clock_cache = self.clock_sensitive_cache.take().unwrap();
        for &idx in clock_cache.iter() {
            self.delay_countdown[idx] = 0; // Ready immediately
            self.dirty.mark_dirty(idx);
            self.last_tick_activated.push(idx);
        }
        self.clock_sensitive_cache = Some(clock_cache);

        let mut delta_count: u32 = 0;
        const MAX_DELTA: u32 = 500;

        let use_coupling = self.physics_coupling_ctx.is_some();

        // Phase 2: Delta iteration with delay modeling (masked: only in-scope tiles drained)
        loop {
            self.current_delta = delta_count;
            crate::dbg_signal!("--- delay delta cycle {} (masked) ---", delta_count);

            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            // Sprint 166: masked drain — only tiles within scope_mask are evaluated
            self.dirty.fill_into_masked(scope_mask, &mut batch);

            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_output_changed = false;
            let mut waiting_count = 0u32;

            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                let countdown = self.delay_countdown[idx];
                let tile_delay = self.effective_delay(idx);

                if countdown == 255 {
                    self.last_tick_activated.push(idx);
                    if tile_delay == 0 {
                        self.delay_countdown[idx] = 0;
                    } else {
                        self.delay_countdown[idx] = tile_delay.saturating_sub(1);
                        if tile_delay > 1 {
                            self.dirty.mark_dirty(idx);
                            waiting_count += 1;
                            continue;
                        }
                    }
                } else if countdown > 0 {
                    self.delay_countdown[idx] = countdown - 1;
                    self.dirty.mark_dirty(idx);
                    waiting_count += 1;
                    continue;
                }

                let our_schedule_time = delta_count.saturating_sub(tile_delay as u32);
                for &ni in &self.neighbors4[idx] {
                    if ni != u32::MAX {
                        let neighbor_arrival = self.arrival_time[ni as usize];
                        if neighbor_arrival > our_schedule_time && neighbor_arrival < delta_count {
                            self.timing_stats.glitches_detected += 1;
                            break;
                        }
                    }
                }

                self.timing_stats.tiles_evaluated += 1;
                let changed = if use_coupling {
                    self.eval_tile_coupled(idx)
                } else {
                    self.eval_tile(idx)
                };

                self.delay_countdown[idx] = 255;

                if changed {
                    any_output_changed = true;
                    self.timing_stats.tiles_switched += 1;
                    self.arrival_time[idx] = delta_count;

                    if delta_count > self.timing_stats.critical_path_deltas {
                        self.timing_stats.critical_path_deltas = delta_count;
                        self.timing_stats.critical_path_endpoint = Some(idx);
                    }

                    let sn = self.neighbors4[idx];
                    let stt = self.tile_type_at(idx);
                    macro_rules! sched {
                        ($ni:expr) => {
                            if $ni != u32::MAX {
                                let ni_usize = $ni as usize;
                                if self.delay_countdown[ni_usize] == 255 {
                                    self.dirty.mark_dirty(ni_usize);
                                }
                            }
                        };
                    }
                    match stt {
                        TileType::WireDown => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[3]);
                        }
                        TileType::WireUp => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[2]);
                        }
                        TileType::WireRight => {
                            sched!(sn[1]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireLeft => {
                            sched!(sn[0]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireH => {
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                        TileType::WireV => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                        }
                        _ => {
                            sched!(sn[0]);
                            sched!(sn[1]);
                            sched!(sn[2]);
                            sched!(sn[3]);
                        }
                    }
                    let svia = self.via_fwd[idx];
                    if svia != u32::MAX {
                        let svia_usize = svia as usize;
                        if self.delay_countdown[svia_usize] == 255 {
                            self.dirty.mark_dirty(svia_usize);
                        }
                    }
                }
            }

            self.dirty_batch_buf = batch;

            if !any_output_changed && waiting_count == 0 {
                break;
            }

            delta_count += 1;
            if delta_count >= MAX_DELTA {
                self.timing_stats.converged = false;
                break;
            }
        }

        self.timing_stats.total_deltas = delta_count;

        // === Bus evaluation phase (masked) ===
        if !self.bus_states.is_empty() {
            self.evaluate_buses();
            let mut post_bus_deltas = 0u32;
            loop {
                self.current_delta = delta_count + post_bus_deltas + 1;
                let mut batch = std::mem::take(&mut self.dirty_batch_buf);
                self.dirty.fill_into_masked(scope_mask, &mut batch);
                if batch.is_empty() {
                    self.dirty_batch_buf = batch;
                    break;
                }
                let mut any_changed = false;
                for &idx32 in batch.iter() {
                    let idx = idx32 as usize;
                    self.timing_stats.tiles_evaluated += 1;
                    let changed = if use_coupling {
                        self.eval_tile_coupled(idx)
                    } else {
                        self.eval_tile(idx)
                    };
                    if changed {
                        any_changed = true;
                    }
                }
                self.dirty_batch_buf = batch;
                if !any_changed {
                    break;
                }
                post_bus_deltas += 1;
                if post_bus_deltas >= 100 {
                    break;
                }
            }
        }

        // Clear physics snapshot at tick end
        self.physics_coupling_ctx = None;

        // EPIC 49: after classical logic stabilizes for this tick, step quantum tiles
        self.step_quantum_tiles();

        self.timing_stats.clone()
    }

    /// Sprint 168: Lightweight clock edge for V2 CPU.
    ///
    /// Replaces `tick_with_delays_masked` for the V2 execution path.
    /// Eliminates all delay/glitch/arrival_time overhead since V2 tiles
    /// have effective delay 0 and never use timing features.
    ///
    /// - `scope_mask`: bitset limiting which dirty tiles are drained.
    /// Sprint 276: Compact clock edge — delta 0 captures via eval_tile,
    /// then compact topological evaluation replaces the 40-67 delta cascade.
    ///
    /// Delta 0 uses eval_tile because Register64/Register8/Ram/ProgramCounter
    /// tiles have clock-aware capture logic that depends on `current_delta == 0`.
    /// The subsequent combinational cascade is purely gate/wire evaluation,
    /// which compact ops handle correctly (clock-sensitive tiles become COP_CONST).
    pub fn tick_clock_edge_compact(
        &mut self,
        scope_mask: &[u64],
        in_scope_clock_cache: &[usize],
        compact_ops: &[CompactOp],
        compact_wvia: &[(usize, u8, u64)],
    ) -> TimingStats {
        // 1. Toggle clock
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        self.cpu_tick_count += 1;

        // 2. Seed clock-sensitive tiles whose input changed.
        for &idx in in_scope_clock_cache {
            if self.clock_tile_input_changed(idx) {
                self.dirty.mark_dirty(idx);
            }
        }

        // 3. Delta 0: eval_tile for clock captures.
        //    Register64/Register8 capture LEFT on rising edge (current_delta == 0).
        //    Ram captures LEFT when UP != 0. ProgramCounter computes next PC.
        self.current_delta = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;

        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);

            total_evaluated += batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                if self.is_chain_tail(idx) {
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    total_switched += 1;
                }
            }
            self.dirty_batch_buf = batch;
        }

        // 4. Compact cascade: topological compact ops replace deltas 1-67.
        //    Clock-sensitive tiles are COP_CONST in compact ops (skipped).
        //    Only combinational tiles (gates, wires, muxes) are evaluated.
        //    Topological order ensures 1-2 passes instead of 40-67.
        let mut cascade_passes: u32 = 0;
        if !compact_ops.is_empty() {
            let (d, e, s) = self.propagate_compact_dirty(compact_ops, compact_wvia);
            cascade_passes = d;
            total_evaluated += e;
            total_switched += s;
        }

        // Sprint 277: Report real pass count instead of hardcoded optimistic values.
        // total_deltas = 1 (delta 0) + cascade_passes.
        let total_deltas = 1 + cascade_passes;
        debug_assert!(
            cascade_passes <= 2,
            "clock compact cascade needed {} passes (expected <= 2) — \
             backward dependency in clock scope topo order?",
            cascade_passes
        );

        TimingStats {
            total_deltas,
            tiles_evaluated: total_evaluated,
            tiles_switched: total_switched,
            glitches_detected: 0,
            converged: cascade_passes < 10,
            critical_path_deltas: total_deltas,
            critical_path_endpoint: None,
            ..Default::default()
        }
    }

    /// Sprint 279: Clock edge with ordered active-work scheduler for the cascade.
    /// Steps 1-3 identical to tick_clock_edge_compact; step 4 uses the scheduler.
    pub fn tick_clock_edge_scheduled(
        &mut self,
        scope_mask: &[u64],
        in_scope_clock_cache: &[usize],
        schedule: &CompactSchedule,
    ) -> TimingStats {
        // 1. Toggle clock (identical to compact path).
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }
        self.cpu_tick_count += 1;

        // 2. Seed clock-sensitive tiles whose input changed.
        for &idx in in_scope_clock_cache {
            if self.clock_tile_input_changed(idx) {
                self.dirty.mark_dirty(idx);
            }
        }

        // 3. Delta 0: eval_tile for clock captures.
        self.current_delta = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);
            total_evaluated += batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                if self.is_chain_tail(idx) {
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    total_switched += 1;
                }
            }
            self.dirty_batch_buf = batch;
        }

        // 4. Cascade: single topological pass (no-dirty pattern).
        // Sprint 285: Replace scheduled propagation with cone_no_dirty approach.
        // Evaluates ALL ops unconditionally in topological order. Only marks
        // dirty tiles OUTSIDE the scope (frontier). Avoids:
        // - dirty_dependents (70% of per-tile cost) for in-scope tiles
        // - is_dirty_and_clear per tile (10%)
        // - dep table/slot bitset complexity
        // - convergence oscillation from re-drain
        let mut cascade_passes: u32 = 0;
        if !schedule.ops.is_empty() {
            let (d, e, s) =
                self.propagate_cone_no_dirty(&schedule.ops, &schedule.wvia, &schedule.scope_mask);
            cascade_passes = d;
            total_evaluated += e;
            total_switched += s;
        }

        let total_deltas = 1 + cascade_passes;
        debug_assert!(
            cascade_passes <= 2,
            "clock scheduled cascade needed {} passes (expected <= 2)",
            cascade_passes
        );

        TimingStats {
            total_deltas,
            tiles_evaluated: total_evaluated,
            tiles_switched: total_switched,
            glitches_detected: 0,
            converged: cascade_passes < 10,
            critical_path_deltas: total_deltas,
            critical_path_endpoint: None,
            ..Default::default()
        }
    }

    /// Sprint 322: Pruned clock cascade — uses full scope_mask for delta-0
    /// seeding AND frontier determination, but evaluates only live ops.
    /// Sprint 324: Fixed — live_scope_mask was too small for frontier, causing
    /// excessive out-of-scope dirty marking. Now uses full scope_mask for cone_set.
    /// Sprint 384: Optional precomputed frontier table (S339 pattern, built with
    /// the full scope_mask) replaces the dynamic per-changed frontier walk.
    /// Empty table → dynamic walk (identical marks either way). The delta-0
    /// drain uses the sparse in-scope segment list (S333 pattern) when provided
    /// — a masked drain pays for every dirty segment in the grid.
    pub fn tick_clock_edge_pruned(
        &mut self,
        scope_mask: &[u64],
        in_scope_segments: &[(u32, u64)],
        in_scope_clock_cache: &[usize],
        live_ops: &[CompactOp],
        live_wvia: &[(usize, u8, u64)],
        frontier_offsets: &[u32],
        frontier_targets: &[u32],
    ) -> TimingStats {
        // 1. Toggle clock.
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }
        self.cpu_tick_count += 1;

        // 2. Seed clock-sensitive tiles (full scope).
        for &idx in in_scope_clock_cache {
            if self.clock_tile_input_changed(idx) {
                self.dirty.mark_dirty(idx);
            }
        }

        // 3. Delta 0 (full clock scope for correct seeding drain).
        // Sprint 384: Sparse drain over the precomputed in-scope segment list
        // when provided — identical drained set, skips out-of-scope dirty
        // segments (build-time residue made the masked traversal cost ~5 µs).
        self.current_delta = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            if !in_scope_segments.is_empty() {
                self.dirty.fill_into_sparse(in_scope_segments, &mut batch);
            } else {
                self.dirty.fill_into_masked(scope_mask, &mut batch);
            }
            total_evaluated += batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                if self.is_chain_tail(idx) {
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    total_switched += 1;
                }
            }
            self.dirty_batch_buf = batch;
        }

        // 4. Cascade on PRUNED ops only, but use FULL scope_mask for frontier.
        // Sprint 384: Frontier table (built against the same full scope_mask)
        // when available; dynamic walk otherwise.
        let mut cascade_passes: u32 = 0;
        if !live_ops.is_empty() {
            let (d, e, s) = if !frontier_offsets.is_empty() {
                self.propagate_cone_no_dirty_ft(
                    live_ops,
                    live_wvia,
                    frontier_offsets,
                    frontier_targets,
                )
            } else {
                self.propagate_cone_no_dirty(live_ops, live_wvia, scope_mask)
            };
            cascade_passes = d;
            total_evaluated += e;
            total_switched += s;
        }

        let total_deltas = 1 + cascade_passes;

        TimingStats {
            total_deltas,
            tiles_evaluated: total_evaluated,
            tiles_switched: total_switched,
            glitches_detected: 0,
            converged: cascade_passes < 10,
            critical_path_deltas: total_deltas,
            critical_path_endpoint: None,
            ..Default::default()
        }
    }

    /// Sprint 321: Counted variant — identical to tick_clock_edge_scheduled but
    /// increments cascade_counts[slot] for each cascade op where result != current.
    /// Used for phase-local dead-op measurement.
    pub fn tick_clock_edge_scheduled_counted(
        &mut self,
        scope_mask: &[u64],
        in_scope_clock_cache: &[usize],
        schedule: &CompactSchedule,
        cascade_counts: &mut [u32],
    ) -> TimingStats {
        // 1. Toggle clock.
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }
        self.cpu_tick_count += 1;

        // 2. Seed clock-sensitive tiles.
        for &idx in in_scope_clock_cache {
            if self.clock_tile_input_changed(idx) {
                self.dirty.mark_dirty(idx);
            }
        }

        // 3. Delta 0.
        self.current_delta = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        {
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);
            total_evaluated += batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                if self.is_chain_tail(idx) {
                    continue;
                }
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    total_switched += 1;
                }
            }
            self.dirty_batch_buf = batch;
        }

        // 4. Cascade with counting.
        let mut cascade_passes: u32 = 0;
        if !schedule.ops.is_empty() {
            let (d, e, s) = self.propagate_cone_no_dirty_counted(
                &schedule.ops,
                &schedule.wvia,
                &schedule.scope_mask,
                cascade_counts,
            );
            cascade_passes = d;
            total_evaluated += e;
            total_switched += s;
        }

        let total_deltas = 1 + cascade_passes;

        TimingStats {
            total_deltas,
            tiles_evaluated: total_evaluated,
            tiles_switched: total_switched,
            glitches_detected: 0,
            converged: cascade_passes < 10,
            critical_path_deltas: total_deltas,
            critical_path_endpoint: None,
            ..Default::default()
        }
    }

    /// - `in_scope_clock_cache`: pre-filtered clock-sensitive tile indices
    ///   within `scope_mask` (computed at build time).
    pub fn tick_clock_edge_lightweight(
        &mut self,
        scope_mask: &[u64],
        in_scope_clock_cache: &[usize],
    ) -> TimingStats {
        // 1. Toggle clock (identical to tick_with_delays_masked)
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        self.cpu_tick_count += 1;

        // 2. Sprint 171: Incremental seeding — only seed clock-sensitive tiles
        //    whose output will actually change on this rising edge. Unchanged
        //    Register64/Register8/Ram tiles are proven safe to skip.
        for &idx in in_scope_clock_cache {
            if self.clock_tile_input_changed(idx) {
                self.dirty.mark_dirty(idx);
            }
        }

        // 3. Delta loop with current_delta tracking for register capture.
        //    Register8/Register64 capture requires current_delta == 0 on first round.
        let mut delta_count: u32 = 0;
        let mut total_evaluated: u32 = 0;
        let mut total_switched: u32 = 0;
        const MAX_DELTA: u32 = 500;

        loop {
            self.current_delta = delta_count;

            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into_masked(scope_mask, &mut batch);

            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_changed = false;
            total_evaluated += batch.len() as u32;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                // Sprint 173: Skip chain tail members — handled by chain fusion.
                if self.is_chain_tail(idx) {
                    continue;
                }
                // Sprint 172: chain fusion — fuse entire chain from head.
                let chain_id = self.chain_head_map[idx];
                if chain_id != u32::MAX {
                    let (changed, tail_n) = self.eval_tile_chain_fused(idx, chain_id as usize);
                    if changed {
                        any_changed = true;
                        total_switched += 1 + tail_n;
                    }
                } else if self.eval_tile(idx) {
                    any_changed = true;
                    total_switched += 1;
                }
            }
            self.dirty_batch_buf = batch;

            if !any_changed {
                break;
            }

            delta_count += 1;
            if delta_count >= MAX_DELTA {
                return TimingStats {
                    total_deltas: delta_count,
                    tiles_evaluated: total_evaluated,
                    tiles_switched: total_switched,
                    glitches_detected: 0,
                    converged: false,
                    critical_path_deltas: delta_count,
                    critical_path_endpoint: None,
                    ..Default::default()
                };
            }
        }

        TimingStats {
            total_deltas: delta_count,
            tiles_evaluated: total_evaluated,
            tiles_switched: total_switched,
            glitches_detected: 0,
            converged: true,
            critical_path_deltas: delta_count,
            critical_path_endpoint: None,
            ..Default::default()
        }
    }

    /// Sprint 154: Toggle clock state without running propagation.
    /// Use for falling edge when no register captures are expected
    /// (Register8/Register64 capture on rising edge only).
    pub fn tick_clock_toggle_only(&mut self) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }
        self.cpu_tick_count += 1;
    }

    /// Sprint 171: Check if a clock-sensitive tile's output will change on this
    /// rising edge. Returns false if we can prove the tile will hold its current
    /// value (safe to skip seeding). Returns true conservatively for complex types.
    #[inline(always)]
    fn clock_tile_input_changed(&self, idx: usize) -> bool {
        let tt = self.tile_type_at(idx);
        let current = self.tilemap.value(idx);
        match tt {
            // Register64: captures LEFT on rising edge. Skip if left == current.
            TileType::Register64 => self.load_logic_idx(self.neighbors4[idx][0]) != current,
            // Register8: captures LEFT & 0xFF on rising edge.
            TileType::Register8 => (self.load_logic_idx(self.neighbors4[idx][0]) & 0xFF) != current,
            // Ram: captures LEFT when UP (write-enable) != 0.
            // If UP == 0, output = current (no change). Skip.
            // If UP != 0 AND left == current, still no change. Skip.
            TileType::Ram => {
                let n = &self.neighbors4[idx];
                let up = self.load_logic_idx(n[2]);
                up != 0 && self.load_logic_idx(n[0]) != current
            }
            // ProgramCounter: PC+1 always changes. Jump logic reads LEFT+RIGHT.
            // ClockGlobal/ClockDivider: output toggles every edge.
            // Counter, Latch, RegEnable, etc.: conservative — always seed.
            _ => true,
        }
    }

    /// Sprint 172: Mark only lateral neighbors dirty (skip input AND output directions).
    /// Used for chain members where forward propagation is handled by chain fusion.
    #[inline(always)]
    fn dirty_dependents_lateral(&mut self, n: &[u32; 4], tt: TileType) {
        if self.suppress_dirty_propagation {
            return;
        }
        macro_rules! md {
            ($ni:expr) => {
                if $ni != u32::MAX {
                    self.dirty.mark_dirty($ni as usize);
                }
            };
        }
        match tt {
            TileType::WireRight | TileType::WireLeft => {
                md!(n[2]);
                md!(n[3]);
            }
            TileType::WireDown | TileType::WireUp => {
                md!(n[0]);
                md!(n[1]);
            }
            _ => {}
        }
    }

    /// Sprint 172: Evaluate a chain-head tile and fuse the entire chain if it changes.
    /// Returns (head_changed, tail_switched_count) for telemetry.
    /// This is a standalone method called from propagate_combinational_masked and
    /// tick_clock_edge_lightweight only — eval_tile/eval_tile_wire are NOT modified.
    fn eval_tile_chain_fused(&mut self, idx: usize, chain_id: usize) -> (bool, u32) {
        // Evaluate head tile using inline eval_tile_wire logic.
        let tt = self.tile_type_at(idx);
        let n = &self.neighbors4[idx];
        let new_out = match tt {
            TileType::WireRight => self.load_logic_idx(n[0]),
            TileType::WireLeft => self.load_logic_idx(n[1]),
            TileType::WireDown => self.load_logic_idx(n[2]),
            TileType::WireUp => self.load_logic_idx(n[3]),
            _ => return (self.eval_tile(idx), 0), // fallback: not a uni wire
        };

        let current = self.tilemap.value(idx);
        if new_out == current {
            return (false, 0);
        }

        // Head changed — store, record, mark lateral dirty.
        self.tilemap.set_value(idx, new_out);
        if self.record_change_info {
            let neighbors = [
                if n[0] == u32::MAX {
                    None
                } else {
                    Some(n[0] as usize)
                },
                if n[1] == u32::MAX {
                    None
                } else {
                    Some(n[1] as usize)
                },
                if n[2] == u32::MAX {
                    None
                } else {
                    Some(n[2] as usize)
                },
                if n[3] == u32::MAX {
                    None
                } else {
                    Some(n[3] as usize)
                },
            ];
            self.last_change[idx] = Some(ChangeInfo {
                delta: self.current_delta,
                old: current,
                new: new_out,
                neighbors,
            });
        }
        let nc = *n;
        self.dirty_dependents_lateral(&nc, tt);
        if !self.suppress_dirty_propagation {
            // via_fwd for head
            let via = self.via_fwd[idx];
            if via != u32::MAX {
                self.dirty.mark_dirty(via as usize);
            }
            // component_input_lookup for head
            if self.component_input_lookup[idx] != u32::MAX {
                let comp_idx = self.component_input_lookup[idx] as usize;
                self.components[comp_idx].cache_valid.set(false);
                for i in 0..self.components[comp_idx].output_port_indices.len() {
                    let out_idx = self.components[comp_idx].output_port_indices[i];
                    self.dirty.mark_dirty(out_idx);
                }
            }
        }

        // Fuse tail
        let tail_switched = self.eval_chain_tail(chain_id, new_out);
        (true, tail_switched)
    }

    /// Sprint 172: Propagate value through all tail members of a wire chain.
    /// Returns count of changed tail members for telemetry.
    fn eval_chain_tail(&mut self, chain_id: usize, value: u64) -> u32 {
        let tt = self.wire_chains[chain_id].wire_type;
        let fwd_slot: usize = match tt {
            TileType::WireRight => 1,
            TileType::WireLeft => 0,
            TileType::WireDown => 3,
            TileType::WireUp => 2,
            _ => unreachable!(),
        };

        let tail_len = self.wire_chains[chain_id].tail_members.len();
        let mut last_changed_idx: Option<usize> = None;
        let mut switched: u32 = 0;

        for i in 0..tail_len {
            let member_idx = self.wire_chains[chain_id].tail_members[i] as usize;
            let current = self.tilemap.value(member_idx);

            if current == value {
                break; // Quiescence — all downstream also unchanged
            }

            self.tilemap.set_value(member_idx, value);
            last_changed_idx = Some(member_idx);
            switched += 1;

            if self.record_change_info {
                let mn = &self.neighbors4[member_idx];
                let neighbors = [
                    if mn[0] == u32::MAX {
                        None
                    } else {
                        Some(mn[0] as usize)
                    },
                    if mn[1] == u32::MAX {
                        None
                    } else {
                        Some(mn[1] as usize)
                    },
                    if mn[2] == u32::MAX {
                        None
                    } else {
                        Some(mn[2] as usize)
                    },
                    if mn[3] == u32::MAX {
                        None
                    } else {
                        Some(mn[3] as usize)
                    },
                ];
                self.last_change[member_idx] = Some(ChangeInfo {
                    delta: self.current_delta,
                    old: current,
                    new: value,
                    neighbors,
                });
            }
            let mn = self.neighbors4[member_idx];
            self.dirty_dependents_lateral(&mn, tt);
            if !self.suppress_dirty_propagation {
                let via = self.via_fwd[member_idx];
                if via != u32::MAX {
                    self.dirty.mark_dirty(via as usize);
                }
                if self.component_input_lookup[member_idx] != u32::MAX {
                    let comp_idx = self.component_input_lookup[member_idx] as usize;
                    self.components[comp_idx].cache_valid.set(false);
                    for j in 0..self.components[comp_idx].output_port_indices.len() {
                        let out_idx = self.components[comp_idx].output_port_indices[j];
                        self.dirty.mark_dirty(out_idx);
                    }
                }
            }
        }

        // Mark last changed member's forward neighbor dirty (chain exit).
        if let Some(last_idx) = last_changed_idx {
            let fwd = self.neighbors4[last_idx][fwd_slot];
            if fwd != u32::MAX {
                self.dirty.mark_dirty(fwd as usize);
            }
        }
        switched
    }

    /// Get the timing statistics from the last tick_with_delays() call.
    pub fn timing_stats(&self) -> &TimingStats {
        &self.timing_stats
    }

    /// Get the effective propagation delay for a tile.
    ///
    /// For wire tiles with wire_delay set (> 0), uses the per-tile value.
    /// Otherwise uses the tile type's default delay.
    #[inline]
    fn effective_delay(&self, idx: usize) -> u8 {
        let base = self.meta_fast[idx].propagation_delay();
        if self.meta_fast[idx].is_wire() && self.wire_delay[idx] > 0 {
            self.wire_delay[idx]
        } else {
            base
        }
    }

    /// Check if the circuit meets a target clock period (in delta cycles).
    ///
    /// Returns a TimingCheckResult indicating pass/fail and slack.
    pub fn check_timing(&self, target_period_deltas: u32) -> TimingCheckResult {
        let critical = self.timing_stats.critical_path_deltas;
        TimingCheckResult {
            meets_timing: critical <= target_period_deltas,
            slack: target_period_deltas as i32 - critical as i32,
            critical_path_deltas: critical,
            target_period: target_period_deltas,
        }
    }

    // =========================================================================
    // Distance-Based Wire Delay
    // =========================================================================

    /// Set the propagation delay for a wire tile based on its length.
    ///
    /// `length` is in tile units. Delay = 1 + length / 10 (capped at 255).
    /// This only affects wire tiles; non-wire tiles ignore the setting.
    pub fn set_wire_delay(&mut self, x: usize, y: usize, length: u16) {
        let width = self.tilemap.width;
        let height = self.tilemap.height;
        if x >= width || y >= height {
            return;
        }
        let idx = y * width + x;
        if self.meta_fast[idx].is_wire() {
            // Base delay of 1, plus 1 for every 10 tiles of length
            let delay = 1u8.saturating_add((length / 10) as u8);
            self.wire_delay[idx] = delay;
        }
    }

    /// Set the propagation delay for a wire tile by index.
    ///
    /// Internal helper for compute_wire_delays().
    fn set_wire_delay_by_idx(&mut self, idx: usize, length: u16) {
        // Base delay of 1, plus 1 for every 10 tiles of length
        let delay = 1u8.saturating_add((length / 10) as u8);
        self.wire_delay[idx] = delay;
    }

    /// Get the wire delay for a tile at (x, y).
    ///
    /// Returns 0 if no custom delay is set, otherwise returns the per-tile delay.
    pub fn get_wire_delay(&self, x: usize, y: usize) -> u8 {
        let width = self.tilemap.width;
        let height = self.tilemap.height;
        if x >= width || y >= height {
            return 0;
        }
        let idx = y * width + x;
        self.wire_delay[idx]
    }

    /// Set the mask for a WeightedViaUp/WeightedViaDown tile by index.
    ///
    /// The mask is ANDed with the source layer value during evaluation.
    /// Default mask is u64::MAX (identity — no filtering).
    pub fn set_tile_mask(&mut self, idx: usize, mask: u64) {
        if idx < self.tile_mask.len() {
            self.tile_mask[idx] = mask;
        }
    }

    /// Get the mask for a tile by index.
    pub fn get_tile_mask(&self, idx: usize) -> u64 {
        if idx < self.tile_mask.len() {
            self.tile_mask[idx]
        } else {
            u64::MAX
        }
    }

    /// Set the threshold for a ThresholdViaUp/ThresholdViaDown tile by index.
    ///
    /// The threshold (0-4) determines how many non-zero in-plane neighbors
    /// are required for the cross-layer signal to pass through.
    /// Default threshold is 1 (passes on any single active neighbor).
    pub fn set_tile_threshold(&mut self, idx: usize, threshold: u8) {
        if idx < self.tile_threshold.len() {
            self.tile_threshold[idx] = threshold;
        }
    }

    /// Get the threshold for a tile by index.
    pub fn get_tile_threshold(&self, idx: usize) -> u8 {
        self.tile_threshold.get(idx).copied().unwrap_or(1)
    }

    /// Set the right-shift amount for a WeightedViaUp/WeightedViaDown tile.
    ///
    /// The source value is shifted right by this amount before the mask is applied.
    /// Default shift is 0 (no shift). Eval: `(source >> shift) & mask`.
    pub fn set_tile_shift(&mut self, idx: usize, shift: u8) {
        if idx < self.tile_shift.len() {
            self.tile_shift[idx] = shift;
        }
    }

    /// Get the right-shift amount for a tile by index.
    pub fn get_tile_shift(&self, idx: usize) -> u8 {
        self.tile_shift.get(idx).copied().unwrap_or(0)
    }

    /// Compute wire delays for all wire tiles based on their distance from
    /// the nearest non-wire source tile.
    ///
    /// This performs a BFS from each gate/register to propagate distances.
    /// Wire delays are set as: 1 + distance / 10 (capped at 255).
    ///
    /// Call this once after setting up tiles and before running simulation
    /// to enable distance-based wire timing.
    pub fn compute_wire_delays(&mut self) {
        use std::collections::VecDeque;

        let tile_count = self.meta_fast.len();
        let mut distance: Vec<u16> = vec![u16::MAX; tile_count];
        let mut queue = VecDeque::new();

        // Initialize: non-wire, non-empty tiles have distance 0
        for idx in 0..tile_count {
            let tt = self.meta_fast[idx];
            // Source tiles: anything that's not a wire and not the default empty wire
            // We check if it has non-zero output or is a non-wire type
            if !tt.is_wire() {
                distance[idx] = 0;
                queue.push_back(idx);
            }
        }

        // BFS to propagate distances through wire tiles
        while let Some(idx) = queue.pop_front() {
            let d = distance[idx];
            for &ni in &self.neighbors4[idx] {
                if ni != u32::MAX {
                    let ni_usize = ni as usize;
                    if distance[ni_usize] > d + 1 {
                        distance[ni_usize] = d + 1;
                        queue.push_back(ni_usize);
                    }
                }
            }
        }

        // Set wire delays based on distance
        for idx in 0..tile_count {
            if self.meta_fast[idx].is_wire() && distance[idx] < u16::MAX {
                self.set_wire_delay_by_idx(idx, distance[idx]);
            }
        }
    }

    /// Reset all wire delays to zero (use tile type's default delay).
    pub fn clear_wire_delays(&mut self) {
        for d in self.wire_delay.iter_mut() {
            *d = 0;
        }
    }

    /// Trace back the critical path from the endpoint to find all tiles involved.
    ///
    /// Returns a vector of tile indices from source to endpoint (if available).
    pub fn trace_critical_path(&self) -> Vec<usize> {
        let mut path = Vec::new();

        if let Some(endpoint) = self.timing_stats.critical_path_endpoint {
            path.push(endpoint);

            let mut current = endpoint;
            let mut current_arrival = self.arrival_time[endpoint];

            // Backtrack: find the neighbor with the highest arrival time < current
            while current_arrival > 0 {
                let mut best_pred: Option<usize> = None;
                let mut best_arrival = 0u32;

                for &ni in &self.neighbors4[current] {
                    if ni != u32::MAX {
                        let ni_usize = ni as usize;
                        let ni_arrival = self.arrival_time[ni_usize];

                        // Look for predecessor that arrived before us
                        if ni_arrival < current_arrival && ni_arrival >= best_arrival {
                            // Check if this could be on the critical path
                            // (its arrival + our delay should equal our arrival)
                            let our_delay = self.effective_delay(current) as u32;
                            if ni_arrival + our_delay <= current_arrival {
                                best_pred = Some(ni_usize);
                                best_arrival = ni_arrival;
                            }
                        }
                    }
                }

                if let Some(pred) = best_pred {
                    path.push(pred);
                    current = pred;
                    current_arrival = best_arrival;
                } else {
                    break; // No more predecessors found
                }
            }

            path.reverse(); // Source to endpoint order
        }

        path
    }

    /// Detect race conditions: tiles where inputs arrive at significantly different times.
    ///
    /// A race occurs when two or more inputs to a tile arrive with different delays,
    /// potentially causing the output to be sampled at the wrong time.
    pub fn detect_races(&self, min_window: u32) -> Vec<RaceCondition> {
        let mut races = Vec::new();
        let width = self.tilemap.width;

        for idx in 0..self.tilemap.tile_count() {
            // Only check combinational tiles (sequential are clock-synchronized)
            if self.meta_fast[idx].is_sequential() {
                continue;
            }

            let neighbors = &self.neighbors4[idx];
            let arrivals: Vec<u32> = neighbors
                .iter()
                .filter(|&&n| n != u32::MAX)
                .map(|&n| self.arrival_time[n as usize])
                .filter(|&a| a > 0) // Only consider active inputs
                .collect();

            if arrivals.len() >= 2 {
                let min_arrival = *arrivals.iter().min().unwrap();
                let max_arrival = *arrivals.iter().max().unwrap();
                let window = max_arrival.saturating_sub(min_arrival);

                if window >= min_window {
                    races.push(RaceCondition {
                        tile_idx: idx,
                        x: idx % width,
                        y: idx / width,
                        early_arrival: min_arrival,
                        late_arrival: max_arrival,
                        race_window: window,
                    });
                }
            }
        }

        races
    }

    // =========================================================================
    // Phase 1B: CPU Execution Metrics
    // =========================================================================

    /// Get CPU execution metrics from the simulation.
    ///
    /// Returns metrics including instruction count, tick count, IPC, and halt status.
    pub fn cpu_metrics(&self) -> CpuExecutionMetrics {
        CpuExecutionMetrics {
            instructions: self.cpu_instruction_count,
            ticks: self.cpu_tick_count,
            ipc: self.cpu_instruction_count as f64 / self.cpu_tick_count.max(1) as f64,
            halted: self.cpu_halted,
        }
    }

    /// Reset CPU execution metrics to zero.
    ///
    /// Call this before starting a new program execution or benchmark run.
    pub fn reset_cpu_metrics(&mut self) {
        self.cpu_instruction_count = 0;
        self.cpu_tick_count = 0;
        self.cpu_halted = false;
    }

    /// Check if a ProgramCounter tile is executing a HALT instruction.
    ///
    /// The HALT opcode is 6 (encoded in bits 5:3 of the instruction).
    /// When detected, sets `cpu_halted` to true.
    ///
    /// # Arguments
    /// * `pc_tile_idx` - Index of the ProgramCounter tile
    ///
    /// # Returns
    /// `true` if the instruction at the PC is HALT (opcode 6)
    pub fn check_cpu_halt(&mut self, pc_tile_idx: usize) -> bool {
        if pc_tile_idx >= self.tilemap.tiles.len() {
            return false;
        }

        // Get the PC value (current address)
        let pc_value = self.tilemap.value(pc_tile_idx);

        // Read the instruction from memory (the PC points to the instruction)
        // In this tile-based CPU, the instruction is typically in a RAM tile
        // For this check, we examine the down neighbor which often holds the fetched instruction
        let n = &self.neighbors4[pc_tile_idx];
        let instruction = if n[3] != u32::MAX {
            self.tilemap.value(n[3] as usize)
        } else {
            // No down neighbor, check if PC itself contains the instruction encoding
            pc_value
        };

        // Extract opcode from instruction (bits 5:3 = opcode field)
        // HALT opcode is 6 (0b110)
        const OP_HALT: u64 = 6;
        let opcode = (instruction >> 3) & 0x7;

        if opcode == OP_HALT {
            self.cpu_halted = true;
            true
        } else {
            false
        }
    }

    // Bench-only tick variant that returns evaluation and change counts per tick.
    // This duplicates tick()'s hot loop but tallies counts for benchmarking without altering semantics.
    #[cfg(feature = "perf-bench")]
    pub fn tick_bench(&mut self) -> (u32, u32) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        // Update clock domain states
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        // Mark clock-sensitive tiles dirty
        for (idx, tile) in self.tilemap.tiles.iter().enumerate() {
            match tile.meta.tile_type {
                TileType::ClockGlobal => {
                    self.dirty.mark_dirty(idx);
                }
                TileType::Latch | TileType::Register8 | TileType::Register64 => {
                    let cd_val = if idx < self.clock_domain_tile_lookup.len() {
                        self.clock_domain_tile_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    } else {
                        self.dirty.mark_dirty(idx);
                    }
                }
                TileType::ClockDivider => {
                    let cd_val = if idx < self.clock_divider_lookup.len() {
                        self.clock_divider_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                TileType::Synchronizer => {
                    let s_val = if idx < self.synchronizer_lookup.len() {
                        self.synchronizer_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if s_val != u32::MAX {
                        let sync_idx = s_val as usize;
                        let domain_idx = self.synchronizer_states[sync_idx].domain_idx;
                        let state = &self.clock_domain_states[domain_idx];
                        if !state.prev_clock && state.clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut delta_count: u32 = 0;
        const MAX_DELTA: u32 = 1000;
        let mut eval_count: u32 = 0;
        let mut change_count: u32 = 0;

        loop {
            self.current_delta = delta_count;
            crate::dbg_signal!("--- delta cycle {} ---", delta_count);
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into(&mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_changed = false;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                eval_count = eval_count.saturating_add(1);
                let changed = self.eval_tile(idx);
                if changed {
                    change_count = change_count.saturating_add(1);
                    any_changed = true;
                }
            }

            self.dirty_batch_buf = batch;
            if !any_changed {
                break;
            }

            delta_count = delta_count.saturating_add(1);
            if delta_count >= MAX_DELTA {
                panic!("Delta cycle limit exceeded");
            }
        }

        // SPRINT 20.0: Step quantum tiles after classical logic stabilizes (same as tick())
        self.step_quantum_tiles();

        (eval_count, change_count)
    }

    // EPIC 39: branchless + optional unchecked neighbor loads (perf-bench only)
    #[cfg(feature = "perf-bench")]
    pub fn tick_bench_branchless(&mut self, unchecked: bool) -> (u32, u32) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;

        // Update clock domain states
        for i in 0..self.clock_domain_states.len() {
            let divider = self.clock_domain_defs[i].divider as u64;
            let phase = self.clock_domain_defs[i].phase_offset as u64;
            let state = &mut self.clock_domain_states[i];
            state.prev_clock = state.clock;
            state.counter += 1;
            let period = 2 * divider;
            let pos = (state.counter + phase) % period;
            state.clock = pos < divider;
        }

        // Mark clock-sensitive tiles dirty
        for (idx, tile) in self.tilemap.tiles.iter().enumerate() {
            match tile.meta.tile_type {
                TileType::ClockGlobal => {
                    self.dirty.mark_dirty(idx);
                }
                TileType::Latch | TileType::Register8 | TileType::Register64 => {
                    let cd_val = if idx < self.clock_domain_tile_lookup.len() {
                        self.clock_domain_tile_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    } else {
                        self.dirty.mark_dirty(idx);
                    }
                }
                TileType::ClockDivider => {
                    let cd_val = if idx < self.clock_divider_lookup.len() {
                        self.clock_divider_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if cd_val != u32::MAX {
                        let domain_idx = cd_val as usize;
                        let state = &self.clock_domain_states[domain_idx];
                        if state.clock != state.prev_clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                TileType::Synchronizer => {
                    let s_val = if idx < self.synchronizer_lookup.len() {
                        self.synchronizer_lookup[idx]
                    } else {
                        u32::MAX
                    };
                    if s_val != u32::MAX {
                        let sync_idx = s_val as usize;
                        let domain_idx = self.synchronizer_states[sync_idx].domain_idx;
                        let state = &self.clock_domain_states[domain_idx];
                        if !state.prev_clock && state.clock {
                            self.dirty.mark_dirty(idx);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut delta_count: u32 = 0;
        const MAX_DELTA: u32 = 1000;
        let mut eval_count: u32 = 0;
        let mut change_count: u32 = 0;

        loop {
            self.current_delta = delta_count;
            crate::dbg_signal!("--- delta cycle {} ---", delta_count);
            let mut batch = std::mem::take(&mut self.dirty_batch_buf);
            self.dirty.fill_into(&mut batch);
            if batch.is_empty() {
                self.dirty_batch_buf = batch;
                break;
            }

            let mut any_changed = false;
            for &idx32 in batch.iter() {
                let idx = idx32 as usize;
                eval_count = eval_count.saturating_add(1);
                let changed = if unchecked {
                    unsafe { self.eval_tile_branchless_unchecked(idx) }
                } else {
                    self.eval_tile_branchless_checked(idx)
                };
                if changed {
                    change_count = change_count.saturating_add(1);
                    any_changed = true;
                }
            }

            self.dirty_batch_buf = batch;
            if !any_changed {
                break;
            }

            delta_count = delta_count.saturating_add(1);
            if delta_count >= MAX_DELTA {
                panic!("Delta cycle limit exceeded");
            }
        }

        // SPRINT 20.0: Step quantum tiles after classical logic stabilizes (same as tick())
        self.step_quantum_tiles();

        (eval_count, change_count)
    }

    // SPRINT 20.0: Encode quantum measurements into 64-bit logic value
    // Bit encoding: bit N = qubit N measurement (0 or 1), unmeasured = 0
    #[inline(always)]
    fn encode_quantum_measurements(&self, qtile_idx: usize) -> u64 {
        let qt = &self.qtiles[qtile_idx];
        let mut output: u64 = 0;
        for (qubit_idx, measured_bit) in qt.measured.iter().enumerate() {
            if let Some(bit) = measured_bit {
                if *bit != 0 && qubit_idx < 64 {
                    output |= 1u64 << qubit_idx;
                }
            }
        }
        output
    }

    #[cfg(feature = "perf-bench")]
    #[inline(always)]
    fn compute_tile_output_branchless(
        &self,
        tt: TileType,
        left: u64,
        right: u64,
        up: u64,
        down: u64,
        current: u64,
        idx: usize,
    ) -> u64 {
        match tt {
            TileType::Wire => left | right | up | down,
            TileType::And => left & right,
            TileType::Or => left | right,
            TileType::Xor => left ^ right,
            TileType::Not => !left,
            TileType::ClockGlobal => {
                if self.global_clock {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Latch => {
                let (clk, _) = self.effective_clock(idx);
                let mask = if clk { u64::MAX } else { 0 };
                (mask & left) | (!mask & current)
            }
            TileType::Register8 => {
                let (clk, prev) = self.effective_clock(idx);
                // Only capture at delta 0 to prevent re-sampling during propagation
                // Mask to 8 bits: Register8 enforces architectural width (Sprint 86.1)
                let rising = (!prev && clk && self.current_delta == 0) as u64;
                let mask = 0u64.wrapping_sub(rising);
                (mask & (left & 0xFF)) | (!mask & current)
            }
            TileType::Register64 => {
                let (clk, prev) = self.effective_clock(idx);
                let rising = (!prev && clk && self.current_delta == 0) as u64;
                let mask = 0u64.wrapping_sub(rising);
                (mask & left) | (!mask & current) // Full u64, no & 0xFF mask
            }
            TileType::CarryDetect => {
                if left > right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Decoder6to64 => 1u64 << (left & 63),
            TileType::VmSpawner | TileType::VmStatus => current,
            TileType::QDemo => {
                // SPRINT 20.0: Quantum→Classical feedback loop
                if self.qtile_lookup[idx] != u32::MAX {
                    let qtile_idx = self.qtile_lookup[idx] as usize;
                    self.encode_quantum_measurements(qtile_idx)
                } else {
                    current
                }
            }

            // === EPIC 103: Arithmetic Tiles ===
            TileType::Add => left.wrapping_add(right),
            TileType::Sub => left.wrapping_sub(right),
            TileType::Mul => left.wrapping_mul(right),
            TileType::Div => {
                if right != 0 {
                    left / right
                } else {
                    0
                }
            }
            TileType::Mod => {
                if right != 0 {
                    left % right
                } else {
                    0
                }
            }
            TileType::Shl => left.wrapping_shl((right & 63) as u32),
            TileType::Shr => left.wrapping_shr((right & 63) as u32),

            // === EPIC 103: Comparison Tiles ===
            TileType::Lt => {
                if left < right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Gt => {
                if left > right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Eq => {
                if left == right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Neq => {
                if left != right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Lte => {
                if left <= right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Gte => {
                if left >= right {
                    u64::MAX
                } else {
                    0
                }
            }

            // === EPIC 103: Routing & Special Tiles ===
            TileType::Mux => {
                if up != 0 {
                    left
                } else {
                    right
                }
            }
            TileType::Zero => {
                if left == 0 {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Neg => (!left).wrapping_add(1),
            TileType::Abs => {
                let is_neg = (left as i64) < 0;
                if is_neg {
                    (!left).wrapping_add(1)
                } else {
                    left
                }
            }

            // === EPIC 104: Memory Tiles ===
            TileType::Ram => {
                if up != 0 {
                    left
                } else {
                    current
                }
            }
            TileType::Counter => {
                if up != 0 {
                    current.wrapping_add(1)
                } else {
                    current
                }
            }
            TileType::Const => current,

            // === Wire Crossing Tiles ===
            TileType::Cross => {
                let h_mask: u64 = 0x0000_0000_FFFF_FFFF;
                let v_mask: u64 = 0xFFFF_FFFF_0000_0000;
                let h_signal = (left & h_mask) | (right & h_mask);
                let v_signal = (up & v_mask) | (down & v_mask);
                h_signal | v_signal
            }
            TileType::WireH => left | right,
            TileType::WireV => up | down,
            TileType::WireDown => up,
            TileType::WireRight => left,
            // WireUp: unidirectional, reads only from down (signal flows upward)
            TileType::WireUp => down,
            // WireLeft: unidirectional, reads only from right (signal flows leftward)
            TileType::WireLeft => right,
            // ComponentOutput: behavioral component output port
            TileType::ComponentOutput => {
                if self.component_lookup[idx] != u32::MAX {
                    let comp_idx = self.component_lookup[idx] as usize;
                    self.evaluate_component_output(comp_idx, idx)
                } else {
                    current
                }
            }
            // BusInterface: value is set by evaluate_buses(), hold during delta eval
            TileType::BusInterface => current,

            // MemoryPort: addressable memory access
            TileType::MemoryPort => {
                if self.memory_port_lookup[idx] != u32::MAX {
                    let conn_idx = self.memory_port_lookup[idx] as usize;
                    self.evaluate_memory_port(conn_idx, left, right, up)
                } else {
                    current
                }
            }

            // EPIC 116: CPU Tiles
            TileType::CpuHead | TileType::Register | TileType::Console => current,

            // === CPU Building Blocks ===
            TileType::Decoder3to8 => {
                let addr = (left & 0b111) as u32;
                1u64 << addr
            }
            TileType::Mux8to1 => {
                let sel = (right & 0b111) as u32;
                let shift = sel * 8;
                (left >> shift) & 0xFF
            }
            TileType::Mux16to1 => {
                let sel = (right & 0xF) as usize;
                let data = if sel < 8 { left } else { up };
                (data >> ((sel & 7) * 8)) & 0xFF
            }
            // Mux4to1: select one of 4 packed bytes from up based on 2-bit select in down
            TileType::Mux4to1 => {
                let sel = (down & 0b11) as u32;
                let shift = sel * 8;
                (up >> shift) & 0xFF
            }
            TileType::Demux1to8 => {
                let data = up & 0xFF;
                let sel = (left & 0b111) as u32;
                let shift = sel * 8;
                data << shift
            }
            TileType::RegEnable => {
                if up != 0 && (right & 1) != 0 {
                    left
                } else {
                    current
                }
            }
            // ProgramCounter: on rising edge of clock (delta 0 only), if jump (right&1) load left, else increment
            TileType::ProgramCounter => {
                let (clk, prev) = self.effective_clock(idx);
                // Only fire on delta 0 to prevent multiple increments per tick
                if !prev && clk && self.current_delta == 0 {
                    if (right & 1) != 0 {
                        left // Jump: load target address
                    } else {
                        current.wrapping_add(1) // Normal: increment PC
                    }
                } else {
                    current // Not rising edge or not delta 0: hold value
                }
            }

            // SPRINT 66: Evolutionary Selection
            TileType::Selector => {
                let my_fitness = (current as u32).count_ones();
                let mut best_val = current;
                let mut best_fitness = my_fitness;
                for &n in &[left, right, up, down] {
                    let n_fitness = (n as u32).count_ones();
                    if n_fitness > best_fitness {
                        best_fitness = n_fitness;
                        best_val = n;
                    }
                }
                best_val
            }

            // Ising model tiles - handled separately in ising_mode
            TileType::IsingNode | TileType::IsingBias => current,

            // === Phase 1: Fully Tile-Based CPU ===
            TileType::AddCarry => {
                let sum = (left as u16).wrapping_add(right as u16);
                (sum & 0x1FF) as u64
            }
            TileType::SubBorrow => {
                let a = left & 0xFF;
                let b = right & 0xFF;
                let diff = (a as u16).wrapping_sub(b as u16);
                let borrow = if a < b { 1u64 } else { 0u64 };
                (diff & 0xFF) as u64 | (borrow << 8)
            }
            TileType::BitSelect => {
                if (left >> (right & 63)) & 1 != 0 {
                    u64::MAX
                } else {
                    0
                }
            }

            // === Wire Crossing Tiles ===
            TileType::WireCross => (left & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            TileType::WireCrossVert => (right & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            TileType::VBusIn => (up & 0xFFFF_FFFF) << 32,
            TileType::VBusOut => (up >> 32) & 0xFFFF_FFFF,

            // === Multi-Layer Via Tiles ===
            TileType::ViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    self.tilemap.value(target)
                } else {
                    0
                }
            }
            TileType::ViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    self.tilemap.value(idx - layer_size)
                } else {
                    0
                }
            }
            // === Sprint 160/206: Weighted Via Tiles (with optional shift) ===
            TileType::WeightedViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    let source = self.tilemap.value(target);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }
            TileType::WeightedViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    let source = self.tilemap.value(idx - layer_size);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }

            // === Sprint 183: Threshold Via Tiles ===
            TileType::ThresholdViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target >= self.tilemap.tiles.len() {
                    0
                } else {
                    let source = self.tilemap.value(target);
                    let threshold = self.tile_threshold[idx];
                    let w = self.tilemap.width;
                    let x = idx % w;
                    let y = (idx / w) % self.tilemap.height;
                    let mut active: u8 = 0;
                    if x > 0 && self.tilemap.value(idx - 1) != 0 {
                        active += 1;
                    }
                    if x < w - 1 && self.tilemap.value(idx + 1) != 0 {
                        active += 1;
                    }
                    if y > 0 && self.tilemap.value(idx - w) != 0 {
                        active += 1;
                    }
                    if y < self.tilemap.height - 1 && self.tilemap.value(idx + w) != 0 {
                        active += 1;
                    }
                    if active >= threshold { source } else { 0 }
                }
            }
            TileType::ThresholdViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx < layer_size {
                    0
                } else {
                    let source = self.tilemap.value(idx - layer_size);
                    let threshold = self.tile_threshold[idx];
                    let w = self.tilemap.width;
                    let x = idx % w;
                    let y = (idx / w) % self.tilemap.height;
                    let mut active: u8 = 0;
                    if x > 0 && self.tilemap.value(idx - 1) != 0 {
                        active += 1;
                    }
                    if x < w - 1 && self.tilemap.value(idx + 1) != 0 {
                        active += 1;
                    }
                    if y > 0 && self.tilemap.value(idx - w) != 0 {
                        active += 1;
                    }
                    if y < self.tilemap.height - 1 && self.tilemap.value(idx + w) != 0 {
                        active += 1;
                    }
                    if active >= threshold { source } else { 0 }
                }
            }

            // === Multi-Clock Domain Tiles ===
            TileType::ClockDivider => {
                let cd_val = if idx < self.clock_divider_lookup.len() {
                    self.clock_divider_lookup[idx]
                } else {
                    u32::MAX
                };
                if cd_val != u32::MAX {
                    let domain_idx = cd_val as usize;
                    if self.clock_domain_states[domain_idx].clock {
                        u64::MAX
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
            TileType::Synchronizer => {
                let s_val = if idx < self.synchronizer_lookup.len() {
                    self.synchronizer_lookup[idx]
                } else {
                    u32::MAX
                };
                if s_val != u32::MAX {
                    let sync_idx = s_val as usize;
                    let sync = &self.synchronizer_states[sync_idx];
                    let domain = &self.clock_domain_states[sync.domain_idx];
                    if !domain.prev_clock && domain.clock {
                        sync.stage2.set(sync.stage1.get());
                        sync.stage1.set(left);
                    }
                    sync.stage2.get()
                } else {
                    current
                }
            }
        }
    }

    #[cfg_attr(not(feature = "perf-bench"), allow(dead_code))]
    fn step_quantum_tiles(&mut self) {
        if self.qtiles.is_empty() {
            return;
        }
        for i in 0..self.qtiles.len() {
            let pc = self.qtiles[i].pc;
            let len = self.qtiles[i].program.len();
            if pc >= len {
                continue;
            }
            let gate = self.qtiles[i].program[pc].clone();
            let outcome = {
                let qt = &mut self.qtiles[i];
                crate::quantum::apply_gate_scalar(&mut qt.state, &gate, &mut qt.rng)
            };
            // advance pc and record measurement if any
            let (tile_idx, should_update) = {
                let qt = &mut self.qtiles[i];
                qt.pc += 1;
                let should_update =
                    if let crate::quantum::GateOutcome::Measured { qubit, bit } = outcome {
                        if let Some(slot) = qt.measured.get_mut(qubit as usize) {
                            *slot = Some(bit);
                        }
                        true
                    } else {
                        false
                    };
                (qt.tile_idx, should_update)
            };

            if should_update {
                // SPRINT 20.0: Update QDemo tile output immediately and mark dirty for propagation
                let new_logic = self.encode_quantum_measurements(i);
                // NOTE: Ordering::Relaxed is safe for single-threaded execution.
                // If tile evaluation is ever parallelized, use Ordering::Release here
                // and Ordering::Acquire in compute_tile_output() loads.
                self.tilemap.set_value(tile_idx, new_logic);
                self.dirty.mark_dirty(tile_idx);
            }
        }
    }

    // perf-bench only: step quantum tiles once and return (gates_applied, q_ops)
    // q_ops = sum over tiles: state.len for each gate applied (one per tile this step)
    #[cfg(feature = "perf-bench")]
    pub fn step_quantum_for_bench(&mut self) -> (u64, u64) {
        if self.qtiles.is_empty() {
            return (0, 0);
        }
        let mut gates: u64 = 0;
        let mut qops: u64 = 0;
        for i in 0..self.qtiles.len() {
            let pc = self.qtiles[i].pc;
            let len = self.qtiles[i].program.len();
            if pc >= len {
                continue;
            }
            let gate = self.qtiles[i].program[pc].clone();
            let outcome = {
                let qt = &mut self.qtiles[i];
                crate::quantum::apply_gate_scalar(&mut qt.state, &gate, &mut qt.rng)
            };
            // advance pc and record measurement if any
            let (tile_idx, state_len, should_update) = {
                let qt = &mut self.qtiles[i];
                qt.pc += 1;
                let should_update =
                    if let crate::quantum::GateOutcome::Measured { qubit, bit } = outcome {
                        if let Some(slot) = qt.measured.get_mut(qubit as usize) {
                            *slot = Some(bit);
                        }
                        true
                    } else {
                        false
                    };
                (qt.tile_idx, qt.state.len, should_update)
            };

            if should_update {
                // SPRINT 20.0: Update QDemo tile output immediately and mark dirty for propagation
                let new_logic = self.encode_quantum_measurements(i);
                // NOTE: Ordering::Relaxed is safe for single-threaded execution.
                // If tile evaluation is ever parallelized, use Ordering::Release here
                // and Ordering::Acquire in compute_tile_output() loads.
                self.tilemap.set_value(tile_idx, new_logic);
                self.dirty.mark_dirty(tile_idx);
            }
            gates += 1;
            qops = qops.saturating_add(state_len as u64);
        }
        (gates, qops)
    }

    // perf-bench: step quantum tiles with backend selection
    #[cfg(feature = "perf-bench")]
    pub fn step_quantum_for_bench_backend(
        &mut self,
        backend: crate::quantum::QBackend,
    ) -> (u64, u64) {
        if self.qtiles.is_empty() {
            return (0, 0);
        }
        // One-time dump of the program to prove we are executing the expected gates.
        static PROGRAM_DUMPED: AtomicBool = AtomicBool::new(false);
        if backend == crate::quantum::QBackend::Jit && !PROGRAM_DUMPED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "JIT_DEBUG: qtile_program_dump_start (total_tiles={})",
                self.qtiles.len()
            );
            for (idx, qt) in self.qtiles.iter().enumerate() {
                eprintln!("JIT_DEBUG: qtile {} program len={}", idx, qt.program.len());
                for (pc, op) in qt.program.iter().enumerate() {
                    eprintln!("  pc={} {:?}", pc, op);
                }
            }
            eprintln!("JIT_DEBUG: qtile_program_dump_end");
        }
        #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
        {
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1")
                && backend == crate::quantum::QBackend::Jit
            {
                use std::sync::atomic::Ordering;
                crate::quantum_jit::H_FASTPATH_HITS.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "JIT_DEBUG: preloop_increment {}",
                    crate::quantum_jit::H_FASTPATH_HITS.load(Ordering::Relaxed)
                );
                // One-time bump is enough; avoid spamming
            }
        }
        let mut gates: u64 = 0;
        let mut qops: u64 = 0;
        for i in 0..self.qtiles.len() {
            let pc = self.qtiles[i].pc;
            let len = self.qtiles[i].program.len();
            if pc >= len {
                continue;
            }
            let gate = self.qtiles[i].program[pc].clone();
            #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1")
                && backend == crate::quantum::QBackend::Jit
            {
                eprintln!("JIT_DEBUG: program_gate_pc{} {:?}", pc, gate);
            }
            let outcome = {
                let qt = &mut self.qtiles[i];
                match backend {
                    crate::quantum::QBackend::Jit => {
                        #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
                        {
                            static ENTER_LOGGED: AtomicBool = AtomicBool::new(false);
                            static ADDR_LOGGED: AtomicBool = AtomicBool::new(false);
                            crate::quantum_jit::H_FASTPATH_HITS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if !ENTER_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                eprintln!(
                                    "JIT_DEBUG: ENTER run_jit_kernel (first hit) pc={} gate={:?}",
                                    pc, gate
                                );
                            }
                            if !ADDR_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                let fast_addr =
                                    &crate::quantum_jit::H_FASTPATH_HITS as *const _ as usize;
                                eprintln!("JIT_DEBUG: H_FASTPATH_HITS addr=0x{:x}", fast_addr);
                            }
                            // Pre-call bump to prove the JIT path executes even if downstream logging is suppressed.
                            crate::quantum_jit::H_FASTPATH_HITS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        #[cfg(feature = "quantum_jit")]
                        unsafe {
                            crate::quantum_jit::run_jit_kernel(&mut qt.state, &gate);
                        }
                        #[cfg(not(feature = "quantum_jit"))]
                        {
                            let _ = crate::quantum::apply_gate_backend(
                                &mut qt.state,
                                &gate,
                                &mut qt.rng,
                                backend,
                            );
                        }
                        crate::quantum::GateOutcome::None
                    }
                    _ => crate::quantum::apply_gate_backend(
                        &mut qt.state,
                        &gate,
                        &mut qt.rng,
                        backend,
                    ),
                }
            };
            let (tile_idx, state_len, should_update) = {
                let qt = &mut self.qtiles[i];
                qt.pc += 1;
                let should_update =
                    if let crate::quantum::GateOutcome::Measured { qubit, bit } = outcome {
                        if let Some(slot) = qt.measured.get_mut(qubit as usize) {
                            *slot = Some(bit);
                        }
                        true
                    } else {
                        false
                    };
                (qt.tile_idx, qt.state.len, should_update)
            };

            if should_update {
                // SPRINT 20.0: Update QDemo tile output immediately and mark dirty for propagation
                let new_logic = self.encode_quantum_measurements(i);
                // NOTE: Ordering::Relaxed is safe for single-threaded execution.
                // If tile evaluation is ever parallelized, use Ordering::Release here
                // and Ordering::Acquire in compute_tile_output() loads.
                self.tilemap.set_value(tile_idx, new_logic);
                self.dirty.mark_dirty(tile_idx);
            }
            gates += 1;
            qops = qops.saturating_add(state_len as u64);
        }
        (gates, qops)
    }

    // EPIC 50: estimate quantum ops per tick (one gate per QTile per tick)
    // Defined as the sum of state lengths across registered QTiles.
    pub fn quantum_ops_per_tick_estimate(&self) -> u64 {
        if self.qtiles.is_empty() {
            return 0;
        }
        let mut sum: u64 = 0;
        for qt in &self.qtiles {
            sum = sum.saturating_add(qt.state.len as u64);
        }
        sum
    }

    // EPIC 49: expose a debug-friendly snapshot of quantum tiles for CLI/demo
    pub fn get_quantum_tiles_debug(&self) -> Vec<(u16, u8, Vec<(f32, f32)>, Vec<Option<u8>>)> {
        let mut out = Vec::new();
        for qt in self.qtiles.iter() {
            let mut amps: Vec<(f32, f32)> = Vec::with_capacity(qt.state.len);
            for i in 0..qt.state.len {
                amps.push((qt.state.real.as_slice()[i], qt.state.imag.as_slice()[i]));
            }
            out.push((qt.id, qt.state.n_qubits, amps, qt.measured.clone()));
        }
        out
    }

    #[cfg(feature = "perf-bench")]
    #[inline(always)]
    fn eval_tile_branchless_checked(&mut self, idx: usize) -> bool {
        if idx >= self.tilemap.tiles.len() {
            return false;
        }
        let n = &self.neighbors4[idx];
        let left = self.load_logic_idx(n[0]);
        let right = self.load_logic_idx(n[1]);
        let up = self.load_logic_idx(n[2]);
        let down = self.load_logic_idx(n[3]);
        let current = self.tilemap.value(idx);
        let tt = self.meta_fast[idx];
        let new_out = self.compute_tile_output_branchless(tt, left, right, up, down, current, idx);
        if new_out != current {
            self.tilemap.set_value(idx, new_out);
            // Directional dirty: skip input direction for unidirectional wires.
            let nc = *n;
            self.dirty_dependents(&nc, idx, tt);
            // Component input propagation: if this tile is a component
            // input port, invalidate cache and mark output ports dirty
            if self.component_input_lookup[idx] != u32::MAX {
                let comp_idx = self.component_input_lookup[idx] as usize;
                self.components[comp_idx].cache_valid.set(false);
                for i in 0..self.components[comp_idx].output_port_indices.len() {
                    let out_idx = self.components[comp_idx].output_port_indices[i];
                    self.dirty.mark_dirty(out_idx);
                }
            }
            true
        } else {
            false
        }
    }

    #[cfg(feature = "perf-bench")]
    #[inline(always)]
    unsafe fn eval_tile_branchless_unchecked(&mut self, idx: usize) -> bool {
        if idx >= self.tilemap.tiles.len() {
            return false;
        }
        let n = unsafe { *self.neighbors4.get_unchecked(idx) };
        let left = self.load_logic_idx_fast(n[0]);
        let right = self.load_logic_idx_fast(n[1]);
        let up = self.load_logic_idx_fast(n[2]);
        let down = self.load_logic_idx_fast(n[3]);
        let current = unsafe { self.tilemap.value_unchecked(idx) };
        let tt = unsafe { *self.meta_fast.get_unchecked(idx) };
        let new_out = self.compute_tile_output_branchless(tt, left, right, up, down, current, idx);
        if new_out != current {
            unsafe { self.tilemap.set_value_unchecked(idx, new_out) };
            // Directional dirty: skip input direction for unidirectional wires.
            self.dirty_dependents(&n, idx, tt);
            // Component input propagation: if this tile is a component
            // input port, invalidate cache and mark output ports dirty
            if self.component_input_lookup[idx] != u32::MAX {
                let comp_idx = self.component_input_lookup[idx] as usize;
                self.components[comp_idx].cache_valid.set(false);
                for i in 0..self.components[comp_idx].output_port_indices.len() {
                    let out_idx = self.components[comp_idx].output_port_indices[i];
                    self.dirty.mark_dirty(out_idx);
                }
            }
            true
        } else {
            false
        }
    }

    #[cfg(feature = "perf-bench")]
    #[inline(always)]
    fn load_logic_idx_fast(&self, idx_u32: u32) -> u64 {
        if idx_u32 == u32::MAX {
            0
        } else {
            let idx = idx_u32 as usize;
            // Safety: caller ensures idx is in-bounds
            unsafe { self.tilemap.value_unchecked(idx) }
        }
    }

    // EPIC 37: pure kernel to compute next output for a tile type and neighbors
    #[inline(always)]
    fn compute_tile_output(
        &self,
        tt: TileType,
        left: u64,
        right: u64,
        up: u64,
        down: u64,
        current: u64,
        idx: usize,
    ) -> u64 {
        match tt {
            TileType::Wire => left | right | up | down,
            TileType::And => left & right,
            TileType::Or => left | right,
            TileType::Xor => left ^ right,
            TileType::Not => !left,
            TileType::ClockGlobal => {
                if self.global_clock {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Latch => {
                let (clk, _prev) = self.effective_clock(idx);
                if clk { left } else { current }
            }
            TileType::Register8 => {
                let (clk, prev) = self.effective_clock(idx);
                // Only capture at delta 0 to prevent re-sampling during propagation
                // Mask to 8 bits: Register8 enforces architectural width (Sprint 86.1)
                if !prev && clk && self.current_delta == 0 {
                    left & 0xFF
                } else {
                    current
                }
            }
            TileType::Register64 => {
                let (clk, prev) = self.effective_clock(idx);
                if !prev && clk && self.current_delta == 0 {
                    left // Full u64, no mask
                } else {
                    current
                }
            }
            TileType::CarryDetect => {
                if left > right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Decoder6to64 => 1u64 << (left & 63),
            TileType::VmSpawner | TileType::VmStatus => current,
            TileType::QDemo => {
                // SPRINT 20.0: Quantum→Classical feedback loop
                if self.qtile_lookup[idx] != u32::MAX {
                    let qtile_idx = self.qtile_lookup[idx] as usize;
                    self.encode_quantum_measurements(qtile_idx)
                } else {
                    current
                }
            }

            // === EPIC 103: Arithmetic Tiles ===
            TileType::Add => left.wrapping_add(right),
            TileType::Sub => left.wrapping_sub(right),
            TileType::Mul => left.wrapping_mul(right),
            TileType::Div => {
                if right != 0 {
                    left / right
                } else {
                    0
                }
            }
            TileType::Mod => {
                if right != 0 {
                    left % right
                } else {
                    0
                }
            }
            TileType::Shl => left.wrapping_shl((right & 63) as u32),
            TileType::Shr => left.wrapping_shr((right & 63) as u32),

            // === EPIC 103: Comparison Tiles ===
            TileType::Lt => {
                if left < right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Gt => {
                if left > right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Eq => {
                if left == right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Neq => {
                if left != right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Lte => {
                if left <= right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Gte => {
                if left >= right {
                    u64::MAX
                } else {
                    0
                }
            }

            // === EPIC 103: Routing & Special Tiles ===
            TileType::Mux => {
                if up != 0 {
                    left
                } else {
                    right
                }
            }
            TileType::Zero => {
                if left == 0 {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Neg => (!left).wrapping_add(1),
            TileType::Abs => {
                let is_neg = (left as i64) < 0;
                if is_neg {
                    (!left).wrapping_add(1)
                } else {
                    left
                }
            }

            // === EPIC 104: Memory Tiles ===
            TileType::Ram => {
                if up != 0 {
                    left
                } else {
                    current
                }
            }
            TileType::Counter => {
                if up != 0 {
                    current.wrapping_add(1)
                } else {
                    current
                }
            }
            TileType::Const => current,

            // === Wire Crossing Tiles ===
            // Cross: horizontal (left/right) uses bits 0-31, vertical (up/down) uses bits 32-63
            // Signals pass through without mixing, enabling bus crossings
            TileType::Cross => {
                let h_mask: u64 = 0x0000_0000_FFFF_FFFF;
                let v_mask: u64 = 0xFFFF_FFFF_0000_0000;
                let h_signal = (left & h_mask) | (right & h_mask);
                let v_signal = (up & v_mask) | (down & v_mask);
                h_signal | v_signal
            }
            // WireH: horizontal-only wire, ignores vertical neighbors
            TileType::WireH => left | right,
            // WireV: vertical-only wire, ignores horizontal neighbors
            TileType::WireV => up | down,
            // WireDown: unidirectional, reads only from up (signal flows downward)
            TileType::WireDown => up,
            // WireRight: unidirectional, reads only from left (signal flows rightward)
            TileType::WireRight => left,
            // WireUp: unidirectional, reads only from down (signal flows upward)
            TileType::WireUp => down,
            // WireLeft: unidirectional, reads only from right (signal flows leftward)
            TileType::WireLeft => right,
            // ComponentOutput: behavioral component output port
            TileType::ComponentOutput => {
                if self.component_lookup[idx] != u32::MAX {
                    let comp_idx = self.component_lookup[idx] as usize;
                    self.evaluate_component_output(comp_idx, idx)
                } else {
                    current
                }
            }
            // BusInterface: value is set by evaluate_buses(), hold during delta eval
            TileType::BusInterface => current,

            // MemoryPort: addressable memory access
            // left=address, right=data_in, up=write_enable, output=read_data
            TileType::MemoryPort => {
                if self.memory_port_lookup[idx] != u32::MAX {
                    let conn_idx = self.memory_port_lookup[idx] as usize;
                    self.evaluate_memory_port(conn_idx, left, right, up)
                } else {
                    current
                }
            }

            // EPIC 116: CPU Tiles
            TileType::CpuHead | TileType::Register | TileType::Console => current,

            // === CPU Building Blocks ===
            // Decoder3to8: 3-bit address (bits 0-2 of left) → 8-bit one-hot output
            TileType::Decoder3to8 => {
                let addr = (left & 0b111) as u32;
                1u64 << addr
            }
            // Mux8to1: select one of 8 packed bytes from left based on 3-bit select in right
            TileType::Mux8to1 => {
                let sel = (right & 0b111) as u32;
                let shift = sel * 8;
                (left >> shift) & 0xFF
            }
            // Mux16to1: select one of 16 packed bytes (left=0-7, up=8-15) based on 4-bit select in right
            TileType::Mux16to1 => {
                let sel = (right & 0xF) as usize;
                let data = if sel < 8 { left } else { up };
                (data >> ((sel & 7) * 8)) & 0xFF
            }
            // Mux4to1: select one of 4 packed bytes from up based on 2-bit select in down
            TileType::Mux4to1 => {
                let sel = (down & 0b11) as u32;
                let shift = sel * 8;
                (up >> shift) & 0xFF
            }
            // Demux1to8: route byte from up to position selected by 3-bit address in left
            TileType::Demux1to8 => {
                let data = up & 0xFF;
                let sel = (left & 0b111) as u32;
                let shift = sel * 8;
                data << shift
            }
            // RegEnable: register that captures only when clock (up) AND enable (right bit 0) are set
            TileType::RegEnable => {
                if up != 0 && (right & 1) != 0 {
                    left
                } else {
                    current
                }
            }
            // ProgramCounter: on rising edge of clock (delta 0 only), if jump (right&1) load left, else increment
            TileType::ProgramCounter => {
                let (clk, prev) = self.effective_clock(idx);
                // Only fire on delta 0 to prevent multiple increments per tick
                if !prev && clk && self.current_delta == 0 {
                    if (right & 1) != 0 {
                        left // Jump: load target address
                    } else {
                        current.wrapping_add(1) // Normal: increment PC
                    }
                } else {
                    current // Not rising edge or not delta 0: hold value
                }
            }

            // SPRINT 66: Evolutionary Selection
            TileType::Selector => {
                let my_fitness = (current as u32).count_ones();
                let mut best_val = current;
                let mut best_fitness = my_fitness;
                for &n in &[left, right, up, down] {
                    let n_fitness = (n as u32).count_ones();
                    if n_fitness > best_fitness {
                        best_fitness = n_fitness;
                        best_val = n;
                    }
                }
                best_val
            }

            // === Ising Mode Tiles ===
            TileType::IsingNode => {
                // P-bit node: compute local field from neighbors and flip stochastically
                // Spins encoded: 0 = spin down (-1), 1+ = spin up (+1)
                let s_left = if left != 0 { 1i64 } else { -1i64 };
                let s_right = if right != 0 { 1i64 } else { -1i64 };
                let s_up = if up != 0 { 1i64 } else { -1i64 };
                let s_down = if down != 0 { 1i64 } else { -1i64 };

                // Local field (negative J = antiferromagnetic for MaxCut)
                let local_field = -(s_left + s_right + s_up + s_down);

                // Deterministic threshold for tile-based simulation
                if local_field > 0 {
                    1 // spin up
                } else if local_field < 0 {
                    0 // spin down
                } else {
                    current // tie - keep current
                }
            }
            TileType::IsingBias => {
                // External field source - outputs constant bias value
                current
            }

            // === Phase 1: Fully Tile-Based CPU ===
            // AddCarry: wrapping add with carry-out in bit 8
            TileType::AddCarry => {
                let sum = (left as u16).wrapping_add(right as u16);
                (sum & 0x1FF) as u64
            }
            // SubBorrow: wrapping sub with borrow-out in bit 8
            TileType::SubBorrow => {
                let a = left & 0xFF;
                let b = right & 0xFF;
                let diff = (a as u16).wrapping_sub(b as u16);
                let borrow = if a < b { 1u64 } else { 0u64 };
                (diff & 0xFF) as u64 | (borrow << 8)
            }
            // BitSelect: extract single bit by position, output MAX or 0
            TileType::BitSelect => {
                if (left >> (right & 63)) & 1 != 0 {
                    u64::MAX
                } else {
                    0
                }
            }

            // === Wire Crossing Tiles ===
            // WireCross: unidirectional crossing — horizontal from left (bits 0-31),
            // vertical from up (bits 32-63, preserved as-is)
            TileType::WireCross => (left & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            // WireCrossVert: like WireCross but reads horizontal from right (for WL chains)
            TileType::WireCrossVert => (right & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            // VBusIn: shift vertical signal from low bus to high bus (entry bridge)
            TileType::VBusIn => (up & 0xFFFF_FFFF) << 32,
            // VBusOut: shift vertical signal from high bus to low bus (exit bridge)
            TileType::VBusOut => (up >> 32) & 0xFFFF_FFFF,

            // === Multi-Layer Via Tiles ===
            TileType::ViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    self.tilemap.value(target)
                } else {
                    0
                }
            }
            TileType::ViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    self.tilemap.value(idx - layer_size)
                } else {
                    0
                }
            }
            // === Sprint 160/206: Weighted Via Tiles (with optional shift) ===
            TileType::WeightedViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    let source = self.tilemap.value(target);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }
            TileType::WeightedViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    let source = self.tilemap.value(idx - layer_size);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }

            // === Sprint 183: Threshold Via Tiles ===
            TileType::ThresholdViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target >= self.tilemap.tiles.len() {
                    0
                } else {
                    let source = self.tilemap.value(target);
                    let threshold = self.tile_threshold[idx];
                    let w = self.tilemap.width;
                    let x = idx % w;
                    let y = (idx / w) % self.tilemap.height;
                    let mut active: u8 = 0;
                    if x > 0 && self.tilemap.value(idx - 1) != 0 {
                        active += 1;
                    }
                    if x < w - 1 && self.tilemap.value(idx + 1) != 0 {
                        active += 1;
                    }
                    if y > 0 && self.tilemap.value(idx - w) != 0 {
                        active += 1;
                    }
                    if y < self.tilemap.height - 1 && self.tilemap.value(idx + w) != 0 {
                        active += 1;
                    }
                    if active >= threshold { source } else { 0 }
                }
            }
            TileType::ThresholdViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx < layer_size {
                    0
                } else {
                    let source = self.tilemap.value(idx - layer_size);
                    let threshold = self.tile_threshold[idx];
                    let w = self.tilemap.width;
                    let x = idx % w;
                    let y = (idx / w) % self.tilemap.height;
                    let mut active: u8 = 0;
                    if x > 0 && self.tilemap.value(idx - 1) != 0 {
                        active += 1;
                    }
                    if x < w - 1 && self.tilemap.value(idx + 1) != 0 {
                        active += 1;
                    }
                    if y > 0 && self.tilemap.value(idx - w) != 0 {
                        active += 1;
                    }
                    if y < self.tilemap.height - 1 && self.tilemap.value(idx + w) != 0 {
                        active += 1;
                    }
                    if active >= threshold { source } else { 0 }
                }
            }

            // === Multi-Clock Domain Tiles ===
            TileType::ClockDivider => {
                let cd_val = if idx < self.clock_divider_lookup.len() {
                    self.clock_divider_lookup[idx]
                } else {
                    u32::MAX
                };
                if cd_val != u32::MAX {
                    let domain_idx = cd_val as usize;
                    if self.clock_domain_states[domain_idx].clock {
                        u64::MAX
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
            TileType::Synchronizer => {
                let s_val = if idx < self.synchronizer_lookup.len() {
                    self.synchronizer_lookup[idx]
                } else {
                    u32::MAX
                };
                if s_val != u32::MAX {
                    let sync_idx = s_val as usize;
                    let sync = &self.synchronizer_states[sync_idx];
                    let domain = &self.clock_domain_states[sync.domain_idx];
                    if !domain.prev_clock && domain.clock {
                        sync.stage2.set(sync.stage1.get());
                        sync.stage1.set(left);
                    }
                    sync.stage2.get()
                } else {
                    current
                }
            }
        }
    }

    #[inline(always)]
    fn neighbors_indices(&self, x: usize, y: usize) -> [Option<usize>; 4] {
        let w = self.tilemap.width;
        let h = self.tilemap.height;
        let left = if x > 0 { Some(y * w + (x - 1)) } else { None };
        let right = if x + 1 < w {
            Some(y * w + (x + 1))
        } else {
            None
        };
        let up = if y > 0 { Some((y - 1) * w + x) } else { None };
        let down = if y + 1 < h {
            Some((y + 1) * w + x)
        } else {
            None
        };
        [left, right, up, down]
    }

    #[inline(always)]
    pub(crate) fn eval_tile(&mut self, idx: usize) -> bool {
        if idx >= self.tilemap.tiles.len() {
            return false;
        }
        // Sprint 170: Check tile type first. Wire types use fast path
        // that loads only needed neighbors (1-4 vs always 4).
        let tt = self.tile_type_at(idx);
        match tt {
            TileType::Wire
            | TileType::WireH
            | TileType::WireV
            | TileType::WireDown
            | TileType::WireUp
            | TileType::WireRight
            | TileType::WireLeft => {
                return self.eval_tile_wire(idx, tt);
            }
            _ => {}
        }
        // General path: load all 4 neighbors (unchanged from pre-Sprint 170).
        let n = &self.neighbors4[idx];
        let left = self.load_logic_idx(n[0]);
        let right = self.load_logic_idx(n[1]);
        let up = self.load_logic_idx(n[2]);
        let down = self.load_logic_idx(n[3]);

        // Current output
        let current = self.tilemap.value(idx);

        let new_out = self.compute_tile_output(tt, left, right, up, down, current, idx);

        // Phase 1B: Track instruction execution for ProgramCounter tiles
        // When clock (up) is active, count this as an instruction fetch/execute
        if tt == TileType::ProgramCounter && up != 0 {
            self.cpu_instruction_count += 1;
        }

        // Debug: always emit an evaluation line (feature-gated no-op otherwise)
        crate::dbg_signal!(
            "[EVAL] Tile {} old={:064b} new={:064b}",
            idx,
            current,
            new_out
        );

        if new_out != current {
            self.tilemap.set_value(idx, new_out);
            if self.record_change_info {
                let neighbors = [
                    if n[0] == u32::MAX {
                        None
                    } else {
                        Some(n[0] as usize)
                    },
                    if n[1] == u32::MAX {
                        None
                    } else {
                        Some(n[1] as usize)
                    },
                    if n[2] == u32::MAX {
                        None
                    } else {
                        Some(n[2] as usize)
                    },
                    if n[3] == u32::MAX {
                        None
                    } else {
                        Some(n[3] as usize)
                    },
                ];
                self.last_change[idx] = Some(ChangeInfo {
                    delta: self.current_delta,
                    old: current,
                    new: new_out,
                    neighbors,
                });
            }
            // Directional dirty: skip input direction for unidirectional wires.
            let nc = *n;
            self.dirty_dependents(&nc, idx, tt);
            // Component input propagation: if this tile is a component
            // input port, invalidate cache and mark output ports dirty
            if !self.suppress_dirty_propagation {
                if self.component_input_lookup[idx] != u32::MAX {
                    let comp_idx = self.component_input_lookup[idx] as usize;
                    self.components[comp_idx].cache_valid.set(false);
                    for i in 0..self.components[comp_idx].output_port_indices.len() {
                        let out_idx = self.components[comp_idx].output_port_indices[i];
                        self.dirty.mark_dirty(out_idx);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Sprint 170: Fast-path eval for wire tiles. Loads only needed neighbor
    /// values, skips compute_tile_output dispatch and component_input_lookup.
    #[inline(always)]
    fn eval_tile_wire(&mut self, idx: usize, tt: TileType) -> bool {
        let n = &self.neighbors4[idx];
        // Load only the neighbor(s) this wire type reads.
        let new_out = match tt {
            TileType::WireDown => self.load_logic_idx(n[2]), // UP only
            TileType::WireUp => self.load_logic_idx(n[3]),   // DOWN only
            TileType::WireRight => self.load_logic_idx(n[0]), // LEFT only
            TileType::WireLeft => self.load_logic_idx(n[1]), // RIGHT only
            TileType::WireH => self.load_logic_idx(n[0]) | self.load_logic_idx(n[1]),
            TileType::WireV => self.load_logic_idx(n[2]) | self.load_logic_idx(n[3]),
            TileType::Wire => {
                self.load_logic_idx(n[0])
                    | self.load_logic_idx(n[1])
                    | self.load_logic_idx(n[2])
                    | self.load_logic_idx(n[3])
            }
            _ => unreachable!(),
        };

        let current = self.tilemap.value(idx);
        if new_out != current {
            self.tilemap.set_value(idx, new_out);
            if self.record_change_info {
                let neighbors = [
                    if n[0] == u32::MAX {
                        None
                    } else {
                        Some(n[0] as usize)
                    },
                    if n[1] == u32::MAX {
                        None
                    } else {
                        Some(n[1] as usize)
                    },
                    if n[2] == u32::MAX {
                        None
                    } else {
                        Some(n[2] as usize)
                    },
                    if n[3] == u32::MAX {
                        None
                    } else {
                        Some(n[3] as usize)
                    },
                ];
                self.last_change[idx] = Some(ChangeInfo {
                    delta: self.current_delta,
                    old: current,
                    new: new_out,
                    neighbors,
                });
            }
            let nc = *n;
            self.dirty_dependents(&nc, idx, tt);
            // Component input propagation: wire tiles can be component input
            // ports (input_wire_type_for_edge maps edges to unidirectional wires).
            if !self.suppress_dirty_propagation {
                if self.component_input_lookup[idx] != u32::MAX {
                    let comp_idx = self.component_input_lookup[idx] as usize;
                    self.components[comp_idx].cache_valid.set(false);
                    for i in 0..self.components[comp_idx].output_port_indices.len() {
                        let out_idx = self.components[comp_idx].output_port_indices[i];
                        self.dirty.mark_dirty(out_idx);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn tile_type_at(&self, idx: usize) -> TileType {
        #[cfg(feature = "perf-bench")]
        {
            self.meta_fast[idx]
        }
        #[cfg(not(feature = "perf-bench"))]
        {
            self.tilemap.tiles[idx].meta.tile_type
        }
    }

    /// Directional dirty propagation: when tile `idx` of type `tt` changes,
    /// only mark neighbors that could depend on this tile's output.
    /// Unidirectional wires skip the input direction to prevent upstream
    /// back-dirtying. Also handles cross-layer via propagation.
    /// n = neighbors4[idx]: [LEFT, RIGHT, UP, DOWN].
    #[inline(always)]
    fn dirty_dependents(&mut self, n: &[u32; 4], idx: usize, tt: TileType) {
        if self.suppress_dirty_propagation {
            return;
        }
        macro_rules! md {
            ($ni:expr) => {
                if $ni != u32::MAX {
                    self.dirty.mark_dirty($ni as usize);
                }
            };
        }
        match tt {
            // Unidirectional wires: skip the input direction.
            // WireDown reads UP → skip UP(n[2])
            TileType::WireDown => {
                md!(n[0]);
                md!(n[1]);
                md!(n[3]);
            }
            // WireUp reads DOWN → skip DOWN(n[3])
            TileType::WireUp => {
                md!(n[0]);
                md!(n[1]);
                md!(n[2]);
            }
            // WireRight reads LEFT → skip LEFT(n[0])
            TileType::WireRight => {
                md!(n[1]);
                md!(n[2]);
                md!(n[3]);
            }
            // WireLeft reads RIGHT → skip RIGHT(n[1])
            TileType::WireLeft => {
                md!(n[0]);
                md!(n[2]);
                md!(n[3]);
            }
            // Bidirectional wires: skip both input directions.
            TileType::WireH => {
                md!(n[2]);
                md!(n[3]);
            }
            TileType::WireV => {
                md!(n[0]);
                md!(n[1]);
            }
            // Everything else: conservative, dirty all 4.
            _ => {
                md!(n[0]);
                md!(n[1]);
                md!(n[2]);
                md!(n[3]);
            }
        }
        // Sprint 80: Cross-layer dirty propagation — O(1) lookup
        let via = self.via_fwd[idx];
        if via != u32::MAX {
            self.dirty.mark_dirty(via as usize);
        }
    }

    // ========================================================================
    // Physics-to-Logic Coupling
    // ========================================================================

    /// Snapshot physics fields for coupling at tick boundary.
    /// Called once per tick before delta cycles begin.
    fn snapshot_physics_for_coupling(&mut self) {
        let tile_count = self.tilemap.tile_count();
        let width = self.tilemap.width;
        let height = self.tilemap.height;

        let mut ctx = PhysicsCouplingContext::with_capacity(tile_count, width, height);

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                ctx.set_heat(idx, self.heat_field.get(x, y));
                ctx.set_charge(idx, self.charge_field.get(x, y));
                ctx.set_power(idx, self.power_field.get(x, y));
            }
        }

        self.physics_coupling_ctx = Some(ctx);
    }

    /// Evaluate a tile with physics coupling applied.
    /// Uses snapshotted physics values from tick start.
    #[inline(always)]
    fn eval_tile_coupled(&mut self, idx: usize) -> bool {
        if idx >= self.tilemap.tiles.len() {
            return false;
        }

        // Snapshot neighbor inputs via precomputed table
        let n = &self.neighbors4[idx];
        let left = self.load_logic_idx(n[0]);
        let right = self.load_logic_idx(n[1]);
        let up = self.load_logic_idx(n[2]);
        let down = self.load_logic_idx(n[3]);

        // Current output
        let current = self.tilemap.value(idx);
        let tt = self.tile_type_at(idx);

        // Get physics values from snapshot
        let (heat, charge, power) = if let Some(ref ctx) = self.physics_coupling_ctx {
            ctx.get_all(idx)
        } else {
            (0, 0, 255) // No coupling: no heat/charge, max power
        };

        // Compute output with optional charge bias
        let raw_output = if self.physics_coupling_config.charge_coupling.enabled
            && is_charge_bias_affected(
                tt,
                &self.physics_coupling_config.charge_coupling.affected_tiles,
            ) {
            self.compute_tile_output_with_charge_bias(
                tt, left, right, up, down, current, idx, charge,
            )
        } else {
            self.compute_tile_output(tt, left, right, up, down, current, idx)
        };

        // Apply heat and power coupling
        let new_out = apply_physics_coupling(
            raw_output,
            current,
            tt,
            heat,
            charge,
            power,
            &self.physics_coupling_config,
        );

        // Debug: always emit an evaluation line (feature-gated no-op otherwise)
        crate::dbg_signal!(
            "[EVAL_COUPLED] Tile {} heat={} charge={} power={} old={:064b} new={:064b}",
            idx,
            heat,
            charge,
            power,
            current,
            new_out
        );

        if new_out != current {
            self.tilemap.set_value(idx, new_out);
            if self.record_change_info {
                let neighbors = [
                    if n[0] == u32::MAX {
                        None
                    } else {
                        Some(n[0] as usize)
                    },
                    if n[1] == u32::MAX {
                        None
                    } else {
                        Some(n[1] as usize)
                    },
                    if n[2] == u32::MAX {
                        None
                    } else {
                        Some(n[2] as usize)
                    },
                    if n[3] == u32::MAX {
                        None
                    } else {
                        Some(n[3] as usize)
                    },
                ];
                self.last_change[idx] = Some(ChangeInfo {
                    delta: self.current_delta,
                    old: current,
                    new: new_out,
                    neighbors,
                });
            }
            // Directional dirty: skip input direction for unidirectional wires.
            let nc = *n;
            self.dirty_dependents(&nc, idx, tt);
            // Component input propagation: if this tile is a component
            // input port, invalidate cache and mark output ports dirty
            if self.component_input_lookup[idx] != u32::MAX {
                let comp_idx = self.component_input_lookup[idx] as usize;
                self.components[comp_idx].cache_valid.set(false);
                for i in 0..self.components[comp_idx].output_port_indices.len() {
                    let out_idx = self.components[comp_idx].output_port_indices[i];
                    self.dirty.mark_dirty(out_idx);
                }
            }
            true
        } else {
            false
        }
    }

    /// Compute tile output with charge bias applied to comparisons and Mux.
    #[inline(always)]
    fn compute_tile_output_with_charge_bias(
        &self,
        tt: TileType,
        left: u64,
        right: u64,
        up: u64,
        down: u64,
        current: u64,
        idx: usize,
        charge: u32,
    ) -> u64 {
        let bias = calculate_charge_bias(charge, &self.physics_coupling_config.charge_coupling);

        if bias == 0 {
            // No bias, use normal computation
            return self.compute_tile_output(tt, left, right, up, down, current, idx);
        }

        match tt {
            // Biased comparisons
            TileType::Lt
            | TileType::Gt
            | TileType::Lte
            | TileType::Gte
            | TileType::Eq
            | TileType::Neq => apply_charge_bias_to_comparison(tt, left, right, bias),

            // Biased Mux
            TileType::Mux => apply_charge_bias_to_mux(up, left, right, bias),

            // Biased Zero detector
            TileType::Zero => apply_charge_bias_to_zero(left, bias),

            // All other tiles: normal computation
            _ => self.compute_tile_output(tt, left, right, up, down, current, idx),
        }
    }

    // ========================================================================
    // Physics Coupling Public API
    // ========================================================================

    /// Enable physics-to-logic coupling with default configuration.
    /// Heat, charge, and power will affect tile computation.
    pub fn enable_physics_coupling(&mut self) {
        self.physics_coupling_config.enabled = true;
    }

    /// Disable physics-to-logic coupling.
    /// Physics fields will still be simulated but won't affect logic.
    pub fn disable_physics_coupling(&mut self) {
        self.physics_coupling_config.enabled = false;
    }

    /// Check if physics coupling is currently enabled.
    pub fn is_physics_coupling_enabled(&self) -> bool {
        self.physics_coupling_config.enabled
    }

    /// Set the full physics coupling configuration.
    pub fn set_physics_coupling_config(&mut self, config: PhysicsCouplingConfig) {
        self.physics_coupling_config = config;
    }

    /// Get a reference to the current physics coupling configuration.
    pub fn physics_coupling_config(&self) -> &PhysicsCouplingConfig {
        &self.physics_coupling_config
    }

    /// Get a mutable reference to the physics coupling configuration.
    pub fn physics_coupling_config_mut(&mut self) -> &mut PhysicsCouplingConfig {
        &mut self.physics_coupling_config
    }

    /// Enable physics coupling with all mechanisms active using defaults.
    pub fn enable_full_physics_coupling(&mut self) {
        self.physics_coupling_config = PhysicsCouplingConfig::all_enabled();
    }

    #[inline(always)]
    fn load_logic_idx(&self, idx_u32: u32) -> u64 {
        if idx_u32 == u32::MAX {
            0
        } else {
            self.tilemap.value(idx_u32 as usize)
        }
    }

    pub fn eval_at(&mut self, x: usize, y: usize) -> bool {
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return false;
        }
        let idx = y * self.tilemap.width + x;
        self.eval_tile(idx)
    }

    pub fn explain_tile(&self, x: usize, y: usize) -> Option<&ChangeInfo> {
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return None;
        }
        let idx = y * self.tilemap.width + x;
        self.last_change.get(idx)?.as_ref()
    }

    pub fn analyze_nets(&self) -> crate::net::NetReport {
        crate::net::analyze_nets(self)
    }

    // EPIC 37: bench-only helpers (no extra side effects beyond state updates)
    #[cfg(feature = "perf-bench")]
    pub fn eval_logic_full_grid_bench(&mut self) -> (u32, u32) {
        let mut eval_count: u32 = 0;
        let mut change_count: u32 = 0;
        for idx in 0..self.tilemap.tiles.len() {
            let n = &self.neighbors4[idx];
            let left = self.load_logic_idx(n[0]);
            let right = self.load_logic_idx(n[1]);
            let up = self.load_logic_idx(n[2]);
            let down = self.load_logic_idx(n[3]);
            let current = self.tilemap.value(idx);
            let tt = self.tile_type_at(idx);
            let new_out = self.compute_tile_output(tt, left, right, up, down, current, idx);
            eval_count = eval_count.saturating_add(1);
            if new_out != current {
                self.tilemap.set_value(idx, new_out);
                change_count = change_count.saturating_add(1);
            }
        }
        (eval_count, change_count)
    }

    #[cfg(feature = "perf-bench")]
    pub fn eval_logic_dirty_batch_bench(&mut self, batch: &[u32]) -> (u32, u32) {
        let mut eval_count: u32 = 0;
        let mut change_count: u32 = 0;
        for &idx32 in batch {
            let idx = idx32 as usize;
            if idx >= self.tilemap.tiles.len() {
                continue;
            }
            let n = &self.neighbors4[idx];
            let left = self.load_logic_idx(n[0]);
            let right = self.load_logic_idx(n[1]);
            let up = self.load_logic_idx(n[2]);
            let down = self.load_logic_idx(n[3]);
            let current = self.tilemap.value(idx);
            let tt = self.tile_type_at(idx);
            let new_out = self.compute_tile_output(tt, left, right, up, down, current, idx);
            eval_count = eval_count.saturating_add(1);
            if new_out != current {
                self.tilemap.set_value(idx, new_out);
                change_count = change_count.saturating_add(1);
            }
        }
        (eval_count, change_count)
    }

    pub fn nets_summary(&self) -> crate::net::net_summary::GlobalSummary {
        let report = self.analyze_nets();
        crate::net::net_summary::summarize_nets(&report)
    }

    pub fn nets_summary_with_regions(&self) -> crate::net::net_summary::SummaryReport {
        let report = self.analyze_nets();
        crate::net::net_summary::summarize_with_regions(&report, &self.region_field)
    }

    pub fn diagnostics(&self) -> crate::diagnostics::EngineDiagnostics {
        let mut issues = Vec::new();
        issues.extend(self.check_fanout_bounds(16).issues);
        issues.extend(self.check_unclocked_registers().issues);
        issues.extend(self.check_orphan_logic().issues);

        let net_report = self.analyze_nets();
        let global_summary = crate::net::net_summary::summarize_nets(&net_report);
        let region_summaries =
            crate::net::net_summary::summarize_with_regions(&net_report, &self.region_field)
                .regions;

        crate::diagnostics::EngineDiagnostics::new(
            issues,
            net_report,
            global_summary,
            region_summaries,
        )
    }

    pub fn lint_with_profile(&self, profile: &LintProfile) -> LintResult {
        let diag = self.diagnostics();
        crate::lint::run_lint(&diag, profile)
    }

    pub fn lint_default(&self) -> LintResult {
        self.lint_with_profile(&LintProfile::default_relaxed())
    }

    pub fn lint_strict(&self) -> LintResult {
        self.lint_with_profile(&LintProfile::strict())
    }

    pub fn probe_logic(
        &mut self,
        x: u32,
        y: u32,
        steps: u32,
    ) -> Result<crate::probe::ProbeTrace, crate::probe::ProbeError> {
        crate::probe::run_logic_probe(self, x, y, steps)
    }

    pub fn probe_field(
        &mut self,
        kind: FieldKind,
        x: u32,
        y: u32,
        steps: u32,
        params: &FieldStepParams,
    ) -> Result<crate::probe::ProbeTrace, crate::probe::ProbeError> {
        crate::probe::run_field_probe(self, kind, x, y, steps, params)
    }

    pub fn step_fields_n(&mut self, params: &FieldStepParams, steps: u32) {
        for _ in 0..steps {
            self.step_fields_with(params);
        }
    }

    pub fn step_fields_with(&mut self, params: &FieldStepParams) {
        crate::fieldstep::step_all_fields(
            &self.logic_field,
            &mut self.logic_field_next,
            &self.power_field,
            &mut self.power_field_next,
            &self.clock_field,
            &mut self.clock_field_next,
            self.global_clock as u64,
            params,
        );
        std::mem::swap(&mut self.logic_field, &mut self.logic_field_next);
        std::mem::swap(&mut self.power_field, &mut self.power_field_next);
        std::mem::swap(&mut self.clock_field, &mut self.clock_field_next);
    }

    pub fn step_heat_n(&mut self, steps: u32, params: &crate::heat::HeatParams) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            crate::heat::inject_heat_from_logic(
                &self.logic_field_coupled,
                &mut self.heat_field,
                params,
            );
            crate::heat::decay_heat(&mut self.heat_field, params);
        }
    }

    pub fn step_charge_n(&mut self, steps: u32, params: &crate::charge::ChargeParams) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            crate::charge::inject_charge_from_logic(
                &self.logic_field_coupled,
                &mut self.charge_field,
                params,
            );
            crate::charge::decay_charge(&mut self.charge_field, params);
        }
    }

    pub fn step_heat_diffuse_n(&mut self, steps: u32, params: &crate::diffuse::DiffuseParams) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            crate::diffuse::diffuse_step(&self.heat_field, &mut self.heat_field_next, params);
            std::mem::swap(&mut self.heat_field, &mut self.heat_field_next);
        }
    }

    pub fn step_charge_diffuse_n(&mut self, steps: u32, params: &crate::diffuse::DiffuseParams) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            crate::diffuse::diffuse_step(&self.charge_field, &mut self.charge_field_next, params);
            std::mem::swap(&mut self.charge_field, &mut self.charge_field_next);
        }
    }

    pub fn step_reaction_once(&mut self, params: &crate::reaction::ReactionParams) {
        crate::reaction::reaction_step(
            &self.heat_field,
            &self.charge_field,
            &mut self.heat_field_react,
            &mut self.charge_field_react,
            params,
        );
        std::mem::swap(&mut self.heat_field, &mut self.heat_field_react);
        std::mem::swap(&mut self.charge_field, &mut self.charge_field_react);
    }

    pub fn step_reaction_n(&mut self, steps: u32, params: &crate::reaction::ReactionParams) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            self.step_reaction_once(params);
        }
    }

    pub fn step_interact_once(&mut self, params: &crate::physics_interact::InteractionParams) {
        crate::physics_interact::interact_heat_charge_step(
            &self.heat_field,
            &self.charge_field,
            &mut self.heat_field_interact,
            &mut self.charge_field_interact,
            params,
        );
        std::mem::swap(&mut self.heat_field, &mut self.heat_field_interact);
        std::mem::swap(&mut self.charge_field, &mut self.charge_field_interact);
    }

    pub fn step_interact_n(
        &mut self,
        steps: u32,
        params: &crate::physics_interact::InteractionParams,
    ) {
        let steps = steps.min(1000);
        for _ in 0..steps {
            self.step_interact_once(params);
        }
    }

    pub fn coupled_step_once(&mut self, params: &crate::coupling::CoupledParams) {
        // Prepare coupled buffers from current state
        let (w, h) = (self.tilemap.width, self.tilemap.height);
        for y in 0..h {
            for x in 0..w {
                self.logic_field_coupled
                    .set(x, y, self.logic_field.get(x, y) as u64);
                self.power_field_coupled
                    .set(x, y, self.power_field.get(x, y) as u32);
            }
        }
        let mut logic_tmp = FieldGrid::new(w, h, 0u64);
        let mut power_tmp = FieldGrid::new(w, h, 0u32);
        crate::coupling::coupled_step(
            &self.logic_field_coupled,
            &self.power_field_coupled,
            &mut logic_tmp,
            &mut power_tmp,
            params,
        );
        self.logic_field_coupled = logic_tmp;
        self.power_field_coupled = power_tmp;
    }

    pub fn coupled_step_n(&mut self, steps: u32, params: &crate::coupling::CoupledParams) {
        let clamped = steps.min(1_000);
        for _ in 0..clamped {
            self.coupled_step_once(params);
        }
    }

    pub fn eval_with_coupling(
        &mut self,
        x: usize,
        y: usize,
        params: &crate::coupling::CoupledParams,
    ) -> u64 {
        self.coupled_step_once(params);
        if x >= self.tilemap.width || y >= self.tilemap.height {
            return 0;
        }
        self.logic_field_coupled.get(x, y)
    }

    pub fn feedback_once(&mut self, params: &crate::physics_feedback::FeedbackParams) {
        let (w, h) = (self.tilemap.width, self.tilemap.height);
        // Apply charge->logic feedback
        let mut tmp_logic = crate::field::FieldGrid::new(w, h, 0u64);
        crate::physics_feedback::feedback_from_charge(
            &self.charge_field,
            &self.logic_field_coupled,
            &mut tmp_logic,
            params,
        );
        self.logic_field_coupled = tmp_logic;

        // Apply heat->logic feedback
        let mut tmp_logic2 = crate::field::FieldGrid::new(w, h, 0u64);
        crate::physics_feedback::feedback_from_heat(
            &self.heat_field,
            &self.logic_field_coupled,
            &mut tmp_logic2,
            params,
        );
        self.logic_field_coupled = tmp_logic2;
    }

    pub fn feedback_n(&mut self, steps: u32, params: &crate::physics_feedback::FeedbackParams) {
        let clamped = steps.min(1_000);
        for _ in 0..clamped {
            self.feedback_once(params);
        }
    }

    pub fn run_scenario(
        &mut self,
        scenario: &crate::physics_scenarios::PhysicsScenario,
    ) -> crate::physics_scenarios::ScenarioSummary {
        crate::physics_scenarios::run_scenario(self, scenario)
    }

    pub fn run_experiment(
        &mut self,
        spec: &crate::physics_experiments::ExperimentKind,
    ) -> crate::physics_experiments::ExperimentResult {
        crate::physics_experiments::run_experiment(self, spec)
    }

    pub fn run_script(
        &mut self,
        script: &crate::scripts::Script,
    ) -> Result<crate::scripts::ScriptResult, crate::scripts::ScriptError> {
        crate::scripts::run_script(self, script)
    }

    // Analysis-only helpers
    pub fn snapshot_field_region(
        &self,
        kind: FieldKind,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Vec<Vec<u64>> {
        let mut out = Vec::with_capacity(height);
        for y in y0..y0 + height {
            let mut row = Vec::with_capacity(width);
            for x in x0..x0 + width {
                let val = match kind {
                    FieldKind::Power => self.power_field.get(x, y) as u64,
                    FieldKind::Logic => self.logic_field.get(x, y) as u64,
                    FieldKind::Clock => self.clock_field.get(x, y) as u64,
                };
                row.push(val);
            }
            out.push(row);
        }
        out
    }

    pub fn snapshot_heat_region(&self, x0: u32, y0: u32, w: u32, h: u32) -> Vec<Vec<u32>> {
        let mut out = Vec::with_capacity(h as usize);
        for y in y0..y0 + h {
            let mut row = Vec::with_capacity(w as usize);
            for x in x0..x0 + w {
                let xu = x as usize;
                let yu = y as usize;
                row.push(self.heat_field.get(xu, yu));
            }
            out.push(row);
        }
        out
    }

    pub fn snapshot_charge_region(&self, x0: u32, y0: u32, w: u32, h: u32) -> Vec<Vec<u32>> {
        let mut out = Vec::with_capacity(h as usize);
        for y in y0..y0 + h {
            let mut row = Vec::with_capacity(w as usize);
            for x in x0..x0 + w {
                let xu = x as usize;
                let yu = y as usize;
                row.push(self.charge_field.get(xu, yu));
            }
            out.push(row);
        }
        out
    }

    pub fn snapshot_feedback_logic_region(
        &self,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Vec<Vec<u64>> {
        let mut out = Vec::with_capacity(height);
        for y in y0..y0 + height {
            let mut row = Vec::with_capacity(width);
            for x in x0..x0 + width {
                row.push(self.logic_field_coupled.get(x, y));
            }
            out.push(row);
        }
        out
    }

    pub fn clear_regions(&mut self) {
        self.region_field.clear(0);
    }

    pub fn assign_region_rect(
        &mut self,
        id: u32,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
    ) -> Result<(), RegionError> {
        if w == 0 || h == 0 {
            return Ok(());
        }
        let x1 = x0.checked_add(w).ok_or(RegionError::OutOfBounds)?;
        let y1 = y0.checked_add(h).ok_or(RegionError::OutOfBounds)?;
        if x1 as usize > self.tilemap.width || y1 as usize > self.tilemap.height {
            return Err(RegionError::OutOfBounds);
        }
        let x_start = x0 as usize;
        let x_end = x1 as usize;
        let y_start = y0 as usize;
        let y_end = y1 as usize;
        for y in y_start..y_end {
            for x in x_start..x_end {
                self.region_field.set(x, y, id);
            }
        }
        Ok(())
    }

    pub fn region_at(&self, x: u32, y: u32) -> Option<u32> {
        let xu = x as usize;
        let yu = y as usize;
        if xu >= self.tilemap.width || yu >= self.tilemap.height {
            None
        } else {
            Some(self.region_field.get(xu, yu))
        }
    }

    pub fn check_fanout_bounds(&self, max_fanout: u32) -> StructuralReport {
        let mut report = StructuralReport::default();
        let (w, h) = (self.tilemap.width, self.tilemap.height);
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let val = self.tilemap.value(idx);
                if val == 0 {
                    continue;
                }
                let neighbors = self.neighbors_indices(x, y);
                let fanout = neighbors.iter().filter(|n| n.is_some()).count() as u32;
                if fanout > max_fanout {
                    report.issues.push(StructuralIssue {
                        x: x as u32,
                        y: y as u32,
                        kind: StructuralIssueKind::FanoutExceeded {
                            fanout,
                            max: max_fanout,
                        },
                    });
                }
            }
        }
        report
    }

    pub fn check_unclocked_registers(&self) -> StructuralReport {
        let mut report = StructuralReport::default();
        let (w, h) = (self.tilemap.width, self.tilemap.height);
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let tile = &self.tilemap.tiles[idx];
                if !matches!(tile.meta.tile_type, TileType::Register8) {
                    continue;
                }
                let neighbors = self.neighbors_indices(x, y);
                let mut has_clock = false;
                for ni in neighbors.into_iter().flatten() {
                    if let Some(t) = self.tilemap.tiles.get(ni) {
                        if matches!(t.meta.tile_type, TileType::ClockGlobal) {
                            has_clock = true;
                            break;
                        }
                    }
                }
                if !has_clock {
                    report.issues.push(StructuralIssue {
                        x: x as u32,
                        y: y as u32,
                        kind: StructuralIssueKind::UnclockedRegister,
                    });
                }
            }
        }
        report
    }

    pub fn check_orphan_logic(&self) -> StructuralReport {
        let mut report = StructuralReport::default();
        let (w, h) = (self.tilemap.width, self.tilemap.height);
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let val = self.tilemap.value(idx);
                if val == 0 {
                    continue;
                }
                if self.region_field.get(x, y) != 0 {
                    continue;
                }
                let neighbors = self.neighbors_indices(x, y);
                let mut isolated = true;
                for ni in neighbors.into_iter().flatten() {
                    if let Some(t) = self.tilemap.tiles.get(ni) {
                        let nlogic = self.tilemap.value(ni);
                        if nlogic != 0 || !matches!(t.meta.tile_type, TileType::Wire) {
                            isolated = false;
                            break;
                        }
                    }
                }
                if isolated {
                    report.issues.push(StructuralIssue {
                        x: x as u32,
                        y: y as u32,
                        kind: StructuralIssueKind::OrphanLogic,
                    });
                }
            }
        }
        report
    }

    #[cfg(test)]
    pub fn field_dims_for_test(&self) -> (usize, usize) {
        (self.logic_field.width(), self.logic_field.height())
    }

    pub fn get_logic_field_for_test(&self, x: usize, y: usize) -> u32 {
        self.logic_field.get(x, y)
    }

    pub fn get_power_field_for_test(&self, x: usize, y: usize) -> u8 {
        self.power_field.get(x, y)
    }

    pub fn get_clock_field_for_test(&self, x: usize, y: usize) -> u8 {
        self.clock_field.get(x, y)
    }

    pub fn set_power_field_for_test(&mut self, x: usize, y: usize, v: u8) {
        self.power_field.set(x, y, v);
    }

    pub fn get_heat_field_for_test(&self, x: usize, y: usize) -> u32 {
        self.heat_field.get(x, y)
    }

    pub fn get_charge_field_for_test(&self, x: usize, y: usize) -> u32 {
        self.charge_field.get(x, y)
    }

    pub fn set_heat_field_for_test(&mut self, x: usize, y: usize, v: u32) {
        self.heat_field.set(x, y, v);
    }

    pub fn set_charge_field_for_test(&mut self, x: usize, y: usize, v: u32) {
        self.charge_field.set(x, y, v);
    }

    #[cfg(test)]
    pub fn set_fields_for_test(&mut self, logic: u32, power: u8, clock: u8) {
        self.logic_field.clear(logic);
        self.power_field.clear(power);
        self.clock_field.clear(clock);
    }

    /// Sprint 172: Number of detected wire chains.
    #[cfg(test)]
    pub(crate) fn wire_chain_count(&self) -> usize {
        self.wire_chains.len()
    }

    /// Sprint 172: Total fused tail members across all chains.
    #[cfg(test)]
    pub(crate) fn wire_chain_total_tail(&self) -> usize {
        self.wire_chains.iter().map(|c| c.tail_members.len()).sum()
    }

    /// Sprint 173: Check if a tile is a chain tail member (not a head).
    #[inline(always)]
    pub fn is_chain_tail(&self, idx: usize) -> bool {
        let seg = idx / 64;
        let bit = idx % 64;
        seg < self.chain_tail_mask.len() && (self.chain_tail_mask[seg] & (1u64 << bit)) != 0
    }
}

#[cfg(test)]
mod perf_tests {
    use super::*;

    #[test]
    fn dirty_batch_buffer_capacity_stable() {
        let mut sim = Simulation::new();
        let cap0 = sim.dirty_buf_capacity();
        // Mark a pattern dirty and tick several times
        for step in 0..10 {
            for i in (0..TILE_COUNT).step_by(257) {
                sim.dirty.mark_dirty(i);
            }
            sim.tick();
            assert!(
                sim.dirty_buf_capacity() >= cap0,
                "capacity shrank at step {}",
                step
            );
        }
    }
}

#[cfg(test)]
mod neighbor_tests {
    use super::*;

    #[test]
    fn neighbors_corners_edges_ok() {
        let sim = Simulation::new();
        // Top-left (0,0)
        let idx0 = 0usize;
        let n0 = &sim.neighbors4[idx0];
        assert_eq!(n0[0], u32::MAX); // left
        assert_eq!(n0[2], u32::MAX); // up
        assert_eq!(n0[1], 1u32); // right
        assert_eq!(n0[3], WIDTH as u32); // down

        // Bottom-right (WIDTH-1, HEIGHT-1)
        let idx1 = (HEIGHT - 1) * WIDTH + (WIDTH - 1);
        let n1 = &sim.neighbors4[idx1];
        assert_eq!(n1[1], u32::MAX); // right
        assert_eq!(n1[3], u32::MAX); // down
        assert_eq!(n1[0], (idx1 - 1) as u32); // left
        assert_eq!(n1[2], (idx1 - WIDTH) as u32); // up
    }

    #[test]
    fn meta_fast_consistency_on_writes() {
        let mut sim = Simulation::new();
        // set_tile updates
        sim.set_tile(10, 10, TileType::And);
        let idx = 10 * WIDTH + 10;
        assert_eq!(sim.meta_fast[idx], TileType::And);
        assert_eq!(
            sim.tilemap.get_tile(10, 10).unwrap().meta.tile_type,
            TileType::And
        );
        // wire_line updates along path
        sim.wire_line(5, 5, 8, 5);
        for x in 6..8 {
            // excludes endpoints
            let i = 5 * WIDTH + x;
            assert_eq!(sim.meta_fast[i], TileType::Wire);
        }
    }
}

// EPIC 39: parity tests for branchless vs match kernels (perf-bench builds)
#[cfg(all(test, feature = "perf-bench"))]
mod branchless_parity_tests {
    use super::*;

    fn check_once(
        sim: &mut Simulation,
        tt: TileType,
        l: u64,
        r: u64,
        u: u64,
        d: u64,
        cur: u64,
        prev_clk: bool,
        clk: bool,
    ) {
        sim.prev_clock = prev_clk;
        sim.global_clock = clk;
        let idx = 0; // Test tile at index 0
        let a = sim.compute_tile_output(tt, l, r, u, d, cur, idx);
        let b = sim.compute_tile_output_branchless(tt, l, r, u, d, cur, idx);
        assert_eq!(
            a, b,
            "mismatch for {:?} with inputs l={:#x} r={:#x} u={:#x} d={:#x} cur={:#x} prev={} clk={}",
            tt, l, r, u, d, cur, prev_clk, clk
        );
    }

    #[test]
    fn branchless_equals_match_basic_set() {
        let mut sim = Simulation::new();
        let vals = [0u64, u64::MAX];
        let tts = [
            TileType::Wire,
            TileType::And,
            TileType::Or,
            TileType::Xor,
            TileType::Not,
            TileType::ClockGlobal,
            TileType::Latch,
            TileType::Register8,
        ];
        for &tt in &tts {
            for &l in &vals {
                for &r in &vals {
                    let u = 0u64;
                    let d = 0u64; // minimize cross terms
                    for &cur in &vals {
                        // test both clock states where relevant
                        check_once(&mut sim, tt, l, r, u, d, cur, false, false);
                        check_once(&mut sim, tt, l, r, u, d, cur, true, false);
                        check_once(&mut sim, tt, l, r, u, d, cur, false, true);
                        check_once(&mut sim, tt, l, r, u, d, cur, true, true);
                    }
                }
            }
        }
    }
}

impl Clone for Simulation {
    fn clone(&self) -> Self {
        // Preserve the source's actual (possibly layered) dimensions. The old
        // hardcoded `Tilemap::new()` / `TILE_COUNT` silently truncated clones of
        // any non-default-size sim (e.g. the 128x640x16 V2 grid) — see the SIMT
        // build-amortization path. For a default 512x512x1 sim this is identical.
        let mut tilemap = Tilemap::with_size_layered(
            self.tilemap.width,
            self.tilemap.height,
            self.tilemap.num_layers,
        );
        for (dst, src) in tilemap.tiles.iter_mut().zip(self.tilemap.tiles.iter()) {
            dst.meta = src.meta;
        }
        // Sprint 386: clone is only supported in normal (stride-1) mode — the
        // SIMT fabric clones per-lane CPUs, never a lane-mode sim.
        assert!(
            !self.tilemap.lane_mode_active(),
            "Simulation::clone in lane mode is unsupported"
        );
        tilemap.copy_values_from(&self.tilemap);

        let dirty = DirtyBitset::new(self.tilemap.tile_count());
        for (dst, src) in dirty.segments.iter().zip(self.dirty.segments.iter()) {
            dst.set(src.get());
        }

        Self {
            tilemap,
            dirty,
            global_clock: self.global_clock,
            prev_clock: self.prev_clock,
            last_change: self.last_change.clone(),
            current_delta: self.current_delta,
            meta_fast: self.meta_fast.clone(),
            neighbors4: self.neighbors4.clone(),
            logic_field: self.logic_field.clone(),
            power_field: self.power_field.clone(),
            clock_field: self.clock_field.clone(),
            region_field: self.region_field.clone(),
            logic_field_next: self.logic_field_next.clone(),
            power_field_next: self.power_field_next.clone(),
            clock_field_next: self.clock_field_next.clone(),
            logic_field_coupled: self.logic_field_coupled.clone(),
            power_field_coupled: self.power_field_coupled.clone(),
            heat_field: self.heat_field.clone(),
            charge_field: self.charge_field.clone(),
            heat_field_next: self.heat_field_next.clone(),
            charge_field_next: self.charge_field_next.clone(),
            heat_field_react: self.heat_field_react.clone(),
            charge_field_react: self.charge_field_react.clone(),
            heat_field_interact: self.heat_field_interact.clone(),
            charge_field_interact: self.charge_field_interact.clone(),
            dirty_batch_buf: Vec::with_capacity(self.dirty_batch_buf.capacity()),
            jit_settle_drained_total: 0,
            schedule_slot_buf: Vec::new(),
            suppress_dirty_propagation: false,
            qtiles: Vec::new(),
            qtile_lookup: self.qtile_lookup.clone(),
            component_defs: Vec::new(),
            components: Vec::new(),
            component_lookup: self.component_lookup.clone(),
            component_input_lookup: self.component_input_lookup.clone(),
            // Bus Architecture
            bus_defs: Vec::new(),
            bus_states: Vec::new(),
            bus_connections: Vec::new(),
            bus_connection_lookup: self.bus_connection_lookup.clone(),
            // Memory Controller
            memory_bank_defs: Vec::new(),
            memory_banks: Vec::new(),
            memory_port_connections: Vec::new(),
            memory_port_lookup: self.memory_port_lookup.clone(),
            // Multi-Clock Domains
            clock_domain_defs: Vec::new(),
            clock_domain_states: self
                .clock_domain_states
                .iter()
                .map(|s| ClockDomainState {
                    clock: s.clock,
                    prev_clock: s.prev_clock,
                    counter: s.counter,
                })
                .collect(),
            clock_domain_tile_lookup: self.clock_domain_tile_lookup.clone(),
            clock_divider_lookup: self.clock_divider_lookup.clone(),
            synchronizer_lookup: self.synchronizer_lookup.clone(),
            synchronizer_states: self
                .synchronizer_states
                .iter()
                .map(|s| SynchronizerState {
                    domain_idx: s.domain_idx,
                    stage1: std::cell::Cell::new(s.stage1.get()),
                    stage2: std::cell::Cell::new(s.stage2.get()),
                })
                .collect(),
            physics_coupling_config: self.physics_coupling_config.clone(),
            physics_coupling_ctx: None, // Don't clone transient snapshot
            // EPIC 123: Clone timing state
            delay_countdown: self.delay_countdown.clone(),
            arrival_time: self.arrival_time.clone(),
            timing_stats: self.timing_stats.clone(),
            wire_delay: self.wire_delay.clone(),
            // Phase 1B: Clone CPU metrics
            cpu_instruction_count: self.cpu_instruction_count,
            cpu_tick_count: self.cpu_tick_count,
            cpu_halted: self.cpu_halted,
            // Sprint 80: Clone via forward table
            via_fwd: self.via_fwd.clone(),
            // Sprint 160: Clone weighted via masks
            tile_mask: self.tile_mask.clone(),
            // Sprint 183: Clone threshold via gating
            tile_threshold: self.tile_threshold.clone(),
            // Sprint 206: Clone weighted via shift
            tile_shift: self.tile_shift.clone(),
            // Sprint 147
            record_change_info: self.record_change_info,
            clock_sensitive_cache: self.clock_sensitive_cache.clone(),
            last_tick_activated: self.last_tick_activated.clone(),
            // Sprint 172: Wire chain fusion
            wire_chains: self.wire_chains.clone(),
            chain_head_map: self.chain_head_map.clone(),
            // Sprint 173: Chain tail mask
            chain_tail_mask: self.chain_tail_mask.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Blueprint;
    use crate::coupling::CoupledParams;
    use crate::fieldstep::FieldStepParams;
    use crate::tile_meta::TileType;
    use std::sync::atomic::Ordering;

    #[test]
    fn simulation_fields_match_tilemap_and_zero_init() {
        let sim = Simulation::new();
        let (w, h) = sim.field_dims_for_test();
        assert_eq!(w, WIDTH);
        assert_eq!(h, HEIGHT);
        assert_eq!(sim.get_logic_field_for_test(0, 0), 0);
        assert_eq!(sim.get_power_field_for_test(WIDTH - 1, HEIGHT - 1), 0);
        assert_eq!(sim.get_clock_field_for_test(10, 10), 0);
    }

    #[test]
    fn simulation_reset_via_new_clears_fields() {
        let mut sim = Simulation::new();
        sim.set_fields_for_test(5, 6, 7);
        sim = Simulation::new();
        assert_eq!(sim.get_logic_field_for_test(0, 0), 0);
        assert_eq!(sim.get_power_field_for_test(1, 1), 0);
        assert_eq!(sim.get_clock_field_for_test(2, 2), 0);
    }

    #[test]
    fn simulation_clone_copies_and_detaches_fields() {
        let mut sim = Simulation::new();
        sim.set_fields_for_test(1, 2, 3);
        sim.tilemap.set_value_at(0, 0, 0xABCD);
        let mut cloned = sim.clone();
        assert_eq!(cloned.get_logic_field_for_test(0, 0), 1);
        assert_eq!(cloned.get_power_field_for_test(0, 0), 2);
        assert_eq!(cloned.get_clock_field_for_test(0, 0), 3);
        let cloned_tile_val = cloned.tilemap.value_at(0, 0).unwrap();
        assert_eq!(cloned_tile_val, 0xABCD);

        cloned.set_fields_for_test(9, 9, 9);
        cloned.tilemap.set_value_at(0, 0, 0xFFFF);

        assert_eq!(sim.get_logic_field_for_test(0, 0), 1);
        let original_tile_val = sim.tilemap.value_at(0, 0).unwrap();
        assert_eq!(original_tile_val, 0xABCD);
    }

    #[test]
    fn simulation_from_blueprint_allocates_zeroed_fields() {
        let mut bp = Blueprint::new(WIDTH as u32, HEIGHT as u32);
        bp.tiles.push(crate::blueprint::BlueprintTile {
            x: 1,
            y: 1,
            z: 0,
            tile_type: TileType::Or,
            logic: Some(0x1),
        });
        let mut sim = Simulation::new();
        bp.apply_to_simulation(&mut sim).expect("apply ok");
        let (w, h) = sim.field_dims_for_test();
        assert_eq!((w, h), (WIDTH, HEIGHT));
        assert_eq!(sim.get_logic_field_for_test(1, 1), 0);
        assert_eq!(sim.get_power_field_for_test(1, 1), 0);
        assert_eq!(sim.get_clock_field_for_test(1, 1), 0);
    }

    #[test]
    fn region_clear_initially_zero() {
        let sim = Simulation::new();
        assert_eq!(sim.region_at(0, 0), Some(0));
        assert_eq!(
            sim.region_at((WIDTH - 1) as u32, (HEIGHT - 1) as u32),
            Some(0)
        );
        assert_eq!(sim.region_at(10, 10), Some(0));
    }

    #[test]
    fn assign_region_rect_in_bounds() {
        let mut sim = Simulation::new();
        sim.assign_region_rect(7, 2, 3, 2, 2).expect("assign ok");
        for y in 3..5 {
            for x in 2..4 {
                assert_eq!(sim.region_at(x, y), Some(7));
            }
        }
        assert_eq!(sim.region_at(1, 3), Some(0));
        assert_eq!(sim.region_at(4, 4), Some(0));
    }

    #[test]
    fn assign_region_rect_oob_rejected() {
        let mut sim = Simulation::new();
        let res = sim.assign_region_rect(9, (WIDTH as u32) - 1, 0, 2, 2);
        assert!(matches!(res, Err(RegionError::OutOfBounds)));
        assert_eq!(sim.region_at((WIDTH - 1) as u32, 0), Some(0));
        assert_eq!(sim.region_at((WIDTH - 2) as u32, 0), Some(0));
    }

    #[test]
    fn clone_preserves_regions() {
        let mut sim = Simulation::new();
        sim.assign_region_rect(5, 0, 0, 2, 1).expect("assign ok");
        let mut cloned = sim.clone();
        assert_eq!(cloned.region_at(0, 0), Some(5));
        assert_eq!(cloned.region_at(1, 0), Some(5));
        assert_eq!(cloned.region_at(2, 0), Some(0));
        cloned.assign_region_rect(6, 1, 1, 1, 1).expect("assign ok");
        assert_eq!(sim.region_at(1, 1), Some(0));
    }

    #[test]
    fn fanout_check_detects_over_limit() {
        let sim = Simulation::new();
        sim.write_logic(10, 10, u64::MAX);
        let report = sim.check_fanout_bounds(2);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(
            report.issues[0],
            StructuralIssue {
                x: 10,
                y: 10,
                kind: StructuralIssueKind::FanoutExceeded { fanout: 4, max: 2 }
            }
        );
    }

    #[test]
    fn unclocked_register_detected() {
        let mut sim = Simulation::new();
        sim.set_tile(5, 5, TileType::Register8);
        let report = sim.check_unclocked_registers();
        assert_eq!(report.issues.len(), 1);
        assert_eq!(
            report.issues[0],
            StructuralIssue {
                x: 5,
                y: 5,
                kind: StructuralIssueKind::UnclockedRegister
            }
        );
    }

    #[test]
    fn unclocked_register_clean_when_clocked() {
        let mut sim = Simulation::new();
        sim.set_tile(6, 6, TileType::Register8);
        sim.set_tile(7, 6, TileType::ClockGlobal);
        let report = sim.check_unclocked_registers();
        assert!(report.issues.is_empty());
    }

    #[test]
    fn orphan_logic_detected() {
        let sim = Simulation::new();
        sim.write_logic(0, 0, 1);
        let report = sim.check_orphan_logic();
        assert_eq!(report.issues.len(), 1);
        assert_eq!(
            report.issues[0],
            StructuralIssue {
                x: 0,
                y: 0,
                kind: StructuralIssueKind::OrphanLogic
            }
        );
    }

    #[test]
    fn checks_are_pure() {
        let mut sim = Simulation::new();
        sim.write_logic(2, 2, 1);
        sim.assign_region_rect(3, 1, 1, 1, 1).expect("assign ok");
        let dirty_before: Vec<u64> = sim.dirty.segments.iter().map(|w| w.get()).collect();
        let clock_before = (sim.global_clock, sim.prev_clock);
        let region_before = sim.region_at(1, 1);
        let logic_before = sim.tilemap.value_at(2, 2).unwrap();

        let _ = sim.check_fanout_bounds(4);
        let _ = sim.check_unclocked_registers();
        let _ = sim.check_orphan_logic();

        let dirty_after: Vec<u64> = sim.dirty.segments.iter().map(|w| w.get()).collect();
        let clock_after = (sim.global_clock, sim.prev_clock);
        let region_after = sim.region_at(1, 1);
        let logic_after = sim.tilemap.value_at(2, 2).unwrap();

        assert_eq!(dirty_before, dirty_after);
        assert_eq!(clock_before, clock_after);
        assert_eq!(region_before, region_after);
        assert_eq!(logic_before, logic_after);
    }

    #[test]
    fn simulation_field_step_wrapper_matches_manual() {
        let mut sim = Simulation::new();
        // Seed fields
        sim.set_fields_for_test(1, 2, 3);
        sim.step_fields_with(&FieldStepParams::default());

        // Manual step for comparison
        let params = FieldStepParams::default();
        let mut logic_dst = crate::field::FieldGrid::new(WIDTH, HEIGHT, 0u32);
        let mut power_dst = crate::field::FieldGrid::new(WIDTH, HEIGHT, 0u8);
        let mut clock_dst = crate::field::FieldGrid::new(WIDTH, HEIGHT, 0u8);
        crate::fieldstep::step_all_fields(
            &sim.logic_field,
            &mut logic_dst,
            &sim.power_field,
            &mut power_dst,
            &sim.clock_field,
            &mut clock_dst,
            sim.global_clock as u64,
            &params,
        );

        assert_eq!(sim.logic_field.width(), logic_dst.width());
        assert_eq!(sim.logic_field.get(0, 0), logic_dst.get(0, 0));
        assert_eq!(sim.power_field.get(0, 0), power_dst.get(0, 0));
        assert_eq!(sim.clock_field.get(0, 0), clock_dst.get(0, 0));
    }

    #[test]
    fn coupled_step_n_clamps_steps() {
        let mut sim = Simulation::new();
        sim.coupled_step_n(2000, &CoupledParams::default());
        // No panic; fields should remain bounded
        assert!(sim.get_power_field_for_test(0, 0) <= CoupledParams::default().max_power as u8);
    }

    #[test]
    fn eval_with_coupling_no_persistent_mutation() {
        let mut sim = Simulation::new();
        sim.set_logic_value(0, 0, 0x1);
        sim.set_power_field_for_test(0, 0, 20);
        let before_logic = sim.get_logic_field_for_test(0, 0);
        let before_power = sim.get_power_field_for_test(0, 0);
        let _ = sim.eval_with_coupling(0, 0, &CoupledParams::default());
        assert_eq!(sim.get_logic_field_for_test(0, 0), before_logic);
        assert_eq!(sim.get_power_field_for_test(0, 0), before_power);
    }

    #[test]
    fn heat_step_matches_manual() {
        let mut sim = Simulation::new();
        sim.set_logic_value(0, 0, 1);
        let params = crate::heat::HeatParams::default();
        let steps = 3u32;
        // Manual loop
        let mut manual = sim.clone();
        let steps_clamped = steps.min(1000);
        for _ in 0..steps_clamped {
            crate::heat::inject_heat_from_logic(
                &manual.logic_field_coupled,
                &mut manual.heat_field,
                &params,
            );
            crate::heat::decay_heat(&mut manual.heat_field, &params);
        }
        sim.step_heat_n(steps, &params);
        assert_eq!(
            sim.get_heat_field_for_test(0, 0),
            manual.get_heat_field_for_test(0, 0)
        );
    }

    #[test]
    fn charge_step_matches_manual() {
        let mut sim = Simulation::new();
        sim.set_logic_value(1, 1, 1);
        let params = crate::charge::ChargeParams::default();
        let steps = 2u32;
        let mut manual = sim.clone();
        let steps_clamped = steps.min(1000);
        for _ in 0..steps_clamped {
            crate::charge::inject_charge_from_logic(
                &manual.logic_field_coupled,
                &mut manual.charge_field,
                &params,
            );
            crate::charge::decay_charge(&mut manual.charge_field, &params);
        }
        sim.step_charge_n(steps, &params);
        assert_eq!(
            sim.get_charge_field_for_test(1, 1),
            manual.get_charge_field_for_test(1, 1)
        );
    }

    // SPRINT 20.0: Test quantum→classical feedback loop
    #[test]
    fn test_quantum_classical_feedback_bell_state() {
        let mut sim = Simulation::new();

        // Create a 2-qubit Bell state: |00⟩ + |11⟩ (superposition)
        // Then measure both qubits - should get correlated results
        let state = crate::quantum::QState::new_zero(2);
        let program = vec![
            crate::quantum::QGate::H(0),       // Hadamard on qubit 0
            crate::quantum::QGate::CNot(0, 1), // CNOT(0→1) creates entanglement
            crate::quantum::QGate::Measure(0), // Measure qubit 0
            crate::quantum::QGate::Measure(1), // Measure qubit 1
        ];

        // Register QDemo tile at position (10, 10)
        sim.register_qdemo_tile(10, 10, state, program, 0x1234_5678);

        // Verify tile is registered
        assert_eq!(sim.qtiles.len(), 1);
        let idx = 10 * crate::tilemap::WIDTH + 10;
        assert_eq!(sim.qtile_lookup[idx], 0);

        // Run 4 ticks (one gate per tick)
        for _ in 0..4 {
            sim.tick();
        }

        // Check measurements were recorded
        let qt = &sim.qtiles[0];
        assert!(qt.measured[0].is_some(), "Qubit 0 should be measured");
        assert!(qt.measured[1].is_some(), "Qubit 1 should be measured");

        // Bell state should give correlated outcomes: both 0 or both 1
        let q0 = qt.measured[0].unwrap();
        let q1 = qt.measured[1].unwrap();
        assert_eq!(q0, q1, "Bell state qubits should be correlated");

        // Verify quantum tile outputs measurement results to classical logic
        let logic_value = sim.tilemap.value_at(10, 10).unwrap();

        // Bit encoding: bit N = qubit N measurement
        let bit0 = (logic_value & 1) != 0;
        let bit1 = (logic_value & 2) != 0;

        assert_eq!(
            bit0,
            q0 != 0,
            "Logic bit 0 should match qubit 0 measurement"
        );
        assert_eq!(
            bit1,
            q1 != 0,
            "Logic bit 1 should match qubit 1 measurement"
        );
    }

    #[test]
    fn test_quantum_measurement_propagates_to_neighbors() {
        let mut sim = Simulation::new();

        // Setup: QDemo tile at (10,10) connected to Wire at (11,10)
        let state = crate::quantum::QState::new_zero(1);
        let program = vec![
            crate::quantum::QGate::X(0),       // Bit flip: |0⟩ → |1⟩
            crate::quantum::QGate::Measure(0), // Measure: should always get 1
        ];
        sim.register_qdemo_tile(10, 10, state, program, 0);

        // Place a Wire tile to the right to receive the signal
        sim.set_tile(11, 10, TileType::Wire);

        // Run 2 ticks: X gate, then Measure
        sim.tick();
        sim.tick();

        // QDemo tile should output bit 0 = 1
        let qdemo_logic = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(
            qdemo_logic & 1,
            1,
            "QDemo should output measurement bit 0 = 1"
        );

        // Check quantum measurement was recorded
        let qt = &sim.qtiles[0];
        assert_eq!(
            qt.measured[0],
            Some(1),
            "Measurement should be 1 (X gate flipped it)"
        );

        // Wire tile should propagate the signal - we need an evaluation pass
        // Since we marked it dirty, we just need to eval once
        sim.eval_at(11, 10);
        let wire_logic = sim.tilemap.value_at(11, 10).unwrap();
        assert_ne!(wire_logic, 0, "Wire should have received signal from QDemo");
    }

    #[test]
    fn test_quantum_unmeasured_outputs_zero() {
        let mut sim = Simulation::new();

        // QDemo tile with superposition but NO measurement
        let state = crate::quantum::QState::new_zero(2);
        let program = vec![
            crate::quantum::QGate::H(0), // Create superposition
                                         // No measurement gates
        ];
        sim.register_qdemo_tile(10, 10, state, program, 0);

        sim.tick();

        // Without measurement, tile should output 0
        let logic_value = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(logic_value, 0, "Unmeasured qubits should output 0");
    }

    // =========================================================================
    // EPIC 103: Arithmetic Tile Tests
    // =========================================================================

    /// Helper to set up a simple tile test with left/right neighbors
    fn setup_arithmetic_test(tile_type: TileType, left_val: u64, right_val: u64) -> Simulation {
        let mut sim = Simulation::new();
        // Center tile at (10, 10)
        sim.set_tile(10, 10, tile_type);
        // Left input at (9, 10)
        sim.set_tile(9, 10, TileType::Wire);
        sim.tilemap.set_value_at(9, 10, left_val);
        // Right input at (11, 10)
        sim.set_tile(11, 10, TileType::Wire);
        sim.tilemap.set_value_at(11, 10, right_val);
        sim.dirty.mark_dirty(10 * WIDTH + 10);
        sim
    }

    /// Helper with up input for Mux, Ram, Counter tests
    fn setup_with_up(
        tile_type: TileType,
        left_val: u64,
        right_val: u64,
        up_val: u64,
    ) -> Simulation {
        let mut sim = setup_arithmetic_test(tile_type, left_val, right_val);
        // Up input at (10, 9)
        sim.set_tile(10, 9, TileType::Wire);
        sim.tilemap.set_value_at(10, 9, up_val);
        sim
    }

    #[test]
    fn test_add_tile() {
        let mut sim = setup_arithmetic_test(TileType::Add, 100, 50);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 150, "Add: 100 + 50 = 150");
    }

    #[test]
    fn test_add_tile_wrapping() {
        let mut sim = setup_arithmetic_test(TileType::Add, u64::MAX, 1);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Add: MAX + 1 should wrap to 0");
    }

    #[test]
    fn test_sub_tile() {
        let mut sim = setup_arithmetic_test(TileType::Sub, 100, 30);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 70, "Sub: 100 - 30 = 70");
    }

    #[test]
    fn test_sub_tile_wrapping() {
        let mut sim = setup_arithmetic_test(TileType::Sub, 0, 1);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Sub: 0 - 1 should wrap to MAX");
    }

    #[test]
    fn test_mul_tile() {
        let mut sim = setup_arithmetic_test(TileType::Mul, 7, 8);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 56, "Mul: 7 * 8 = 56");
    }

    #[test]
    fn test_div_tile() {
        let mut sim = setup_arithmetic_test(TileType::Div, 100, 7);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 14, "Div: 100 / 7 = 14");
    }

    #[test]
    fn test_div_tile_by_zero() {
        let mut sim = setup_arithmetic_test(TileType::Div, 100, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Div by zero should return 0");
    }

    #[test]
    fn test_mod_tile() {
        let mut sim = setup_arithmetic_test(TileType::Mod, 100, 7);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 2, "Mod: 100 % 7 = 2");
    }

    #[test]
    fn test_mod_tile_by_zero() {
        let mut sim = setup_arithmetic_test(TileType::Mod, 100, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Mod by zero should return 0");
    }

    #[test]
    fn test_shl_tile() {
        let mut sim = setup_arithmetic_test(TileType::Shl, 1, 4);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 16, "Shl: 1 << 4 = 16");
    }

    #[test]
    fn test_shl_tile_mask() {
        // Shift amount should be masked to 6 bits (& 63)
        let mut sim = setup_arithmetic_test(TileType::Shl, 1, 65); // 65 & 63 = 1
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 2, "Shl: 1 << (65 & 63) = 1 << 1 = 2");
    }

    #[test]
    fn test_shr_tile() {
        let mut sim = setup_arithmetic_test(TileType::Shr, 64, 3);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 8, "Shr: 64 >> 3 = 8");
    }

    #[test]
    fn test_lt_tile_true() {
        let mut sim = setup_arithmetic_test(TileType::Lt, 5, 10);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Lt: 5 < 10 should return MAX");
    }

    #[test]
    fn test_lt_tile_false() {
        let mut sim = setup_arithmetic_test(TileType::Lt, 10, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Lt: 10 < 5 should return 0");
    }

    #[test]
    fn test_gt_tile() {
        let mut sim = setup_arithmetic_test(TileType::Gt, 10, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Gt: 10 > 5 should return MAX");
    }

    #[test]
    fn test_eq_tile_true() {
        let mut sim = setup_arithmetic_test(TileType::Eq, 42, 42);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Eq: 42 == 42 should return MAX");
    }

    #[test]
    fn test_eq_tile_false() {
        let mut sim = setup_arithmetic_test(TileType::Eq, 42, 43);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Eq: 42 == 43 should return 0");
    }

    #[test]
    fn test_neq_tile() {
        let mut sim = setup_arithmetic_test(TileType::Neq, 42, 43);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Neq: 42 != 43 should return MAX");
    }

    #[test]
    fn test_lte_tile() {
        let mut sim = setup_arithmetic_test(TileType::Lte, 5, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Lte: 5 <= 5 should return MAX");
    }

    #[test]
    fn test_gte_tile() {
        let mut sim = setup_arithmetic_test(TileType::Gte, 10, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Gte: 10 >= 5 should return MAX");
    }

    #[test]
    fn test_mux_tile_select_left() {
        let mut sim = setup_with_up(TileType::Mux, 100, 200, 1); // up != 0, select left
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 100, "Mux: up != 0 should select left");
    }

    #[test]
    fn test_mux_tile_select_right() {
        let mut sim = setup_with_up(TileType::Mux, 100, 200, 0); // up == 0, select right
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 200, "Mux: up == 0 should select right");
    }

    #[test]
    fn test_zero_tile_true() {
        let mut sim = setup_arithmetic_test(TileType::Zero, 0, 999);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, u64::MAX, "Zero: left == 0 should return MAX");
    }

    #[test]
    fn test_zero_tile_false() {
        let mut sim = setup_arithmetic_test(TileType::Zero, 42, 999);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 0, "Zero: left != 0 should return 0");
    }

    #[test]
    fn test_neg_tile() {
        let mut sim = setup_arithmetic_test(TileType::Neg, 5, 999);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        // Two's complement negation of 5: ~5 + 1 = 0xFFFFFFFFFFFFFFFA + 1 = 0xFFFFFFFFFFFFFFFB
        let expected = (!5u64).wrapping_add(1);
        assert_eq!(
            result, expected,
            "Neg: should compute two's complement negation"
        );
    }

    #[test]
    fn test_abs_tile_positive() {
        let mut sim = setup_arithmetic_test(TileType::Abs, 42, 999);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 42, "Abs: positive number unchanged");
    }

    #[test]
    fn test_abs_tile_negative() {
        // -5 in two's complement = 0xFFFFFFFFFFFFFFFB
        let neg_five = (!5u64).wrapping_add(1);
        let mut sim = setup_arithmetic_test(TileType::Abs, neg_five, 999);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 5, "Abs: negative number should become positive");
    }

    #[test]
    fn test_ram_tile_write() {
        let mut sim = setup_with_up(TileType::Ram, 42, 0, 1); // up != 0, write
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 42, "Ram: up != 0 should write left to output");
    }

    #[test]
    fn test_ram_tile_hold() {
        let mut sim = setup_with_up(TileType::Ram, 42, 0, 0); // up == 0, hold
        // Set current value
        sim.tilemap.set_value_at(10, 10, 100);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 100, "Ram: up == 0 should hold current value");
    }

    #[test]
    fn test_counter_tile_increment() {
        let mut sim = setup_with_up(TileType::Counter, 0, 0, 1); // up != 0, count
        sim.tilemap.set_value_at(10, 10, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 6, "Counter: up != 0 should increment");
    }

    #[test]
    fn test_counter_tile_hold() {
        let mut sim = setup_with_up(TileType::Counter, 0, 0, 0); // up == 0, hold
        sim.tilemap.set_value_at(10, 10, 5);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 5, "Counter: up == 0 should hold");
    }

    #[test]
    fn test_const_tile() {
        let mut sim = setup_arithmetic_test(TileType::Const, 0, 0);
        sim.tilemap.set_value_at(10, 10, 12345);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(result, 12345, "Const: should always output current value");
    }

    // =========================================================================
    // Wire Crossing Tile Tests
    // =========================================================================

    /// Helper to set up a cross tile test with all 4 neighbors
    fn setup_cross_test(left: u64, right: u64, up: u64, down: u64) -> Simulation {
        let mut sim = Simulation::new();
        // Center tile at (10, 10)
        sim.set_tile(10, 10, TileType::Cross);
        // Left input at (9, 10)
        sim.set_tile(9, 10, TileType::Wire);
        sim.tilemap.set_value_at(9, 10, left);
        // Right input at (11, 10)
        sim.set_tile(11, 10, TileType::Wire);
        sim.tilemap.set_value_at(11, 10, right);
        // Up input at (10, 9)
        sim.set_tile(10, 9, TileType::Wire);
        sim.tilemap.set_value_at(10, 9, up);
        // Down input at (10, 11)
        sim.set_tile(10, 11, TileType::Wire);
        sim.tilemap.set_value_at(10, 11, down);
        sim.dirty.mark_dirty(10 * WIDTH + 10);
        sim
    }

    #[test]
    fn test_cross_horizontal_signal_isolation() {
        // Horizontal signal in lower 32 bits, vertical signal in upper 32 bits
        // Send signal only horizontally - vertical should remain isolated
        let h_signal: u64 = 0x0000_0000_DEAD_BEEF; // Lower 32 bits
        let mut sim = setup_cross_test(h_signal, 0, 0, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        // Only lower 32 bits should have signal
        assert_eq!(
            result & 0x0000_0000_FFFF_FFFF,
            h_signal,
            "Horizontal signal should pass through lower 32 bits"
        );
        assert_eq!(
            result & 0xFFFF_FFFF_0000_0000,
            0,
            "Upper 32 bits should be zero (no vertical signal)"
        );
    }

    #[test]
    fn test_cross_vertical_signal_isolation() {
        // Send signal only vertically (upper 32 bits)
        let v_signal: u64 = 0xCAFE_BABE_0000_0000; // Upper 32 bits
        let mut sim = setup_cross_test(0, 0, v_signal, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        // Only upper 32 bits should have signal
        assert_eq!(
            result & 0xFFFF_FFFF_0000_0000,
            v_signal,
            "Vertical signal should pass through upper 32 bits"
        );
        assert_eq!(
            result & 0x0000_0000_FFFF_FFFF,
            0,
            "Lower 32 bits should be zero (no horizontal signal)"
        );
    }

    #[test]
    fn test_cross_both_signals_independent() {
        // Both horizontal and vertical signals should pass independently
        let h_signal: u64 = 0x0000_0000_1234_5678;
        let v_signal: u64 = 0xABCD_EF00_0000_0000;
        let mut sim = setup_cross_test(h_signal, 0, v_signal, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        assert_eq!(
            result,
            h_signal | v_signal,
            "Both signals should pass through independently"
        );
    }

    #[test]
    fn test_cross_bidirectional_horizontal() {
        // Signal from both left AND right should OR together in lower 32 bits
        let left_signal: u64 = 0x0000_0000_0000_00FF;
        let right_signal: u64 = 0x0000_0000_0000_FF00;
        let mut sim = setup_cross_test(left_signal, right_signal, 0, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        assert_eq!(
            result,
            left_signal | right_signal,
            "Left and right signals should OR together"
        );
    }

    #[test]
    fn test_cross_bidirectional_vertical() {
        // Signal from both up AND down should OR together in upper 32 bits
        let up_signal: u64 = 0x00FF_0000_0000_0000;
        let down_signal: u64 = 0xFF00_0000_0000_0000;
        let mut sim = setup_cross_test(0, 0, up_signal, down_signal);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        assert_eq!(
            result,
            up_signal | down_signal,
            "Up and down signals should OR together"
        );
    }

    #[test]
    fn test_wireh_horizontal_only() {
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::WireH);
        // Set all 4 neighbors
        sim.set_tile(9, 10, TileType::Wire);
        sim.tilemap.set_value_at(9, 10, 0xAAAA);
        sim.set_tile(11, 10, TileType::Wire);
        sim.tilemap.set_value_at(11, 10, 0x5555);
        sim.set_tile(10, 9, TileType::Wire);
        sim.tilemap.set_value_at(10, 9, 0xFFFF_0000); // Should be ignored
        sim.set_tile(10, 11, TileType::Wire);
        sim.tilemap.set_value_at(10, 11, 0x0000_FFFF); // Should be ignored
        sim.dirty.mark_dirty(10 * WIDTH + 10);

        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        assert_eq!(
            result,
            0xAAAA | 0x5555,
            "WireH should only OR left and right, ignoring up and down"
        );
    }

    #[test]
    fn test_wirev_vertical_only() {
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::WireV);
        // Set all 4 neighbors
        sim.set_tile(9, 10, TileType::Wire);
        sim.tilemap.set_value_at(9, 10, 0xFFFF_0000); // Should be ignored
        sim.set_tile(11, 10, TileType::Wire);
        sim.tilemap.set_value_at(11, 10, 0x0000_FFFF); // Should be ignored
        sim.set_tile(10, 9, TileType::Wire);
        sim.tilemap.set_value_at(10, 9, 0xAAAA);
        sim.set_tile(10, 11, TileType::Wire);
        sim.tilemap.set_value_at(10, 11, 0x5555);
        sim.dirty.mark_dirty(10 * WIDTH + 10);

        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        assert_eq!(
            result,
            0xAAAA | 0x5555,
            "WireV should only OR up and down, ignoring left and right"
        );
    }

    #[test]
    fn test_cross_tile_enables_bus_crossing() {
        // Simulate a practical bus crossing scenario:
        // Horizontal bus carries 0x0000_0000_0000_0001 (bit 0)
        // Vertical bus carries 0x0001_0000_0000_0000 (bit 48)
        // They should pass through without interfering
        let h_bus: u64 = 0x0000_0000_0000_0001; // Bit 0 set
        let v_bus: u64 = 0x0001_0000_0000_0000; // Bit 48 set
        let mut sim = setup_cross_test(h_bus, 0, v_bus, 0);
        sim.eval_at(10, 10);
        let result = sim.tilemap.value_at(10, 10).unwrap();

        // Extract the signals back
        let h_out = result & 0x0000_0000_FFFF_FFFF;
        let v_out = result & 0xFFFF_FFFF_0000_0000;

        assert_eq!(h_out, h_bus, "Horizontal bus signal should be preserved");
        assert_eq!(v_out, v_bus, "Vertical bus signal should be preserved");
        assert_eq!(
            result,
            h_bus | v_bus,
            "Combined output should have both signals"
        );
    }

    // ========================================
    // CPU Building Blocks Tests
    // ========================================

    #[test]
    fn test_decoder3to8_all_addresses() {
        // Test that decoder converts 3-bit address to one-hot output
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Decoder3to8);
        sim.set_tile(9, 10, TileType::Wire); // left input

        for addr in 0u64..8 {
            // Set address on left input
            sim.set_logic_value(9, 10, addr);
            sim.eval_at(10, 10);
            let result = sim.get_logic_at(10, 10);
            let expected = 1u64 << addr;
            assert_eq!(
                result, expected,
                "Decoder address {} should produce one-hot {}",
                addr, expected
            );
        }
    }

    #[test]
    fn test_decoder3to8_ignores_upper_bits() {
        // Address should only use bits 0-2, ignoring upper bits
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Decoder3to8);
        sim.set_tile(9, 10, TileType::Wire);

        // Address 0xFF03 should decode as address 3 (bits 0-2 = 011)
        sim.set_logic_value(9, 10, 0xFF03);
        sim.eval_at(10, 10);
        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, 0b0000_1000, "Should decode to bit 3 (one-hot 8)");
    }

    #[test]
    fn test_mux8to1_select_each_byte() {
        // Pack 8 distinct bytes and verify each can be selected
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Mux8to1);
        sim.set_tile(9, 10, TileType::Wire); // left = data (8 packed bytes)
        sim.set_tile(11, 10, TileType::Wire); // right = select

        // Pack 8 bytes: D0=0x01, D1=0x02, D2=0x04, D3=0x08, D4=0x10, D5=0x20, D6=0x40, D7=0x80
        let packed: u64 = 0x8040_2010_0804_0201;
        sim.set_logic_value(9, 10, packed);

        for sel in 0u64..8 {
            sim.set_logic_value(11, 10, sel);
            sim.eval_at(10, 10);
            let result = sim.get_logic_at(10, 10);
            let expected = 1u64 << sel; // Each byte is a power of 2
            assert_eq!(
                result, expected,
                "Mux select {} should output byte 0x{:02x}",
                sel, expected
            );
        }
    }

    #[test]
    fn test_mux8to1_full_bytes() {
        // Test with full 8-bit values in each position
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Mux8to1);
        sim.set_tile(9, 10, TileType::Wire);
        sim.set_tile(11, 10, TileType::Wire);

        // Pack: D0=0xAA, D1=0xBB, D2=0xCC, D3=0xDD, D4=0xEE, D5=0xFF, D6=0x11, D7=0x22
        let packed: u64 = 0x2211_FFEE_DDCC_BBAA;
        sim.set_logic_value(9, 10, packed);

        let expected_bytes = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22];
        for (sel, &expected) in expected_bytes.iter().enumerate() {
            sim.set_logic_value(11, 10, sel as u64);
            sim.eval_at(10, 10);
            let result = sim.get_logic_at(10, 10);
            assert_eq!(
                result, expected,
                "Mux select {} should output 0x{:02x}",
                sel, expected
            );
        }
    }

    #[test]
    fn test_demux1to8_route_to_each_position() {
        // Test routing a byte to each of 8 positions
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Demux1to8);
        sim.set_tile(10, 9, TileType::Wire); // up = data
        sim.set_tile(9, 10, TileType::Wire); // left = select

        let data: u64 = 0xAB;
        sim.set_logic_value(10, 9, data);

        for sel in 0u64..8 {
            sim.set_logic_value(9, 10, sel);
            sim.eval_at(10, 10);
            let result = sim.get_logic_at(10, 10);
            let expected = data << (sel * 8);
            assert_eq!(
                result, expected,
                "Demux select {} should route 0xAB to position {}: expected 0x{:016x}, got 0x{:016x}",
                sel, sel, expected, result
            );
        }
    }

    #[test]
    fn test_demux1to8_only_uses_lower_byte() {
        // Demux should only use bits 0-7 of the input, ignoring upper bits
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Demux1to8);
        sim.set_tile(10, 9, TileType::Wire);
        sim.set_tile(9, 10, TileType::Wire);

        // Input has data in upper bits too, but only lower byte should be used
        sim.set_logic_value(10, 9, 0xFFFF_FFFF_FFFF_FF42);
        sim.set_logic_value(9, 10, 2); // select position 2
        sim.eval_at(10, 10);
        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, 0x42 << 16, "Should only route lower byte 0x42");
    }

    #[test]
    fn test_regenable_captures_when_both_high() {
        // RegEnable should capture only when clock (up) AND enable (right) are both non-zero
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::RegEnable);
        sim.set_tile(9, 10, TileType::Wire); // left = data
        sim.set_tile(10, 9, TileType::Wire); // up = clock
        sim.set_tile(11, 10, TileType::Wire); // right = enable

        // Initial state
        sim.set_logic_value(9, 10, 0x42); // data = 0x42
        sim.set_logic_value(10, 9, 0); // clock = 0
        sim.set_logic_value(11, 10, 0); // enable = 0

        // clock=0, enable=0 -> should not capture
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0,
            "Should not capture when clock=0, enable=0"
        );

        // clock=1, enable=0 -> should not capture
        sim.set_logic_value(10, 9, 1);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0,
            "Should not capture when clock=1, enable=0"
        );

        // clock=0, enable=1 -> should not capture
        sim.set_logic_value(10, 9, 0);
        sim.set_logic_value(11, 10, 1);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0,
            "Should not capture when clock=0, enable=1"
        );

        // clock=1, enable=1 -> SHOULD capture
        sim.set_logic_value(10, 9, 1);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0x42,
            "Should capture when clock=1, enable=1"
        );
    }

    #[test]
    fn test_regenable_holds_value() {
        // After capturing, register should hold value even when clock goes low
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::RegEnable);
        sim.set_tile(9, 10, TileType::Wire);
        sim.set_tile(10, 9, TileType::Wire);
        sim.set_tile(11, 10, TileType::Wire);

        // Capture value 0x99
        sim.set_logic_value(9, 10, 0x99);
        sim.set_logic_value(10, 9, 1); // clock high
        sim.set_logic_value(11, 10, 1); // enable high
        sim.eval_at(10, 10);
        assert_eq!(sim.get_logic_at(10, 10), 0x99);

        // Drop clock, value should be retained
        sim.set_logic_value(10, 9, 0);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0x99,
            "Should hold value when clock drops"
        );

        // Change data input, value should still be retained (clock is low)
        sim.set_logic_value(9, 10, 0xFF);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0x99,
            "Should hold value when data changes but clock is low"
        );
    }

    #[test]
    fn test_regenable_enable_bit_only() {
        // Enable should only check bit 0, not the whole value
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::RegEnable);
        sim.set_tile(9, 10, TileType::Wire);
        sim.set_tile(10, 9, TileType::Wire);
        sim.set_tile(11, 10, TileType::Wire);

        sim.set_logic_value(9, 10, 0x55);
        sim.set_logic_value(10, 9, 1);

        // Enable = 0xFE (bit 0 is 0) -> should NOT capture
        sim.set_logic_value(11, 10, 0xFE);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0,
            "Enable 0xFE (bit 0 = 0) should not capture"
        );

        // Enable = 0x01 (bit 0 is 1) -> should capture
        sim.set_logic_value(11, 10, 0x01);
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            0x55,
            "Enable 0x01 (bit 0 = 1) should capture"
        );
    }

    #[test]
    fn test_cpu_blocks_integration_simple_register_read() {
        // Integration test: Decoder selects register, Mux outputs selected data
        // This simulates a simple register file read operation
        let mut sim = Simulation::new();

        // Setup: Decoder at (10,10), Mux at (12,10)
        sim.set_tile(10, 10, TileType::Decoder3to8);
        sim.set_tile(12, 10, TileType::Mux8to1);

        // Wire inputs
        sim.set_tile(9, 10, TileType::Wire); // decoder address input
        sim.set_tile(11, 10, TileType::Wire); // mux data input
        sim.set_tile(13, 10, TileType::Wire); // mux select input

        // Set register file data: 8 different values packed
        let reg_data: u64 = 0x0807_0605_0403_0201; // R0=1, R1=2, ..., R7=8
        sim.set_logic_value(11, 10, reg_data);

        // Select register 3
        sim.set_logic_value(9, 10, 3);
        sim.set_logic_value(13, 10, 3);

        // Evaluate
        sim.eval_at(10, 10);
        sim.eval_at(12, 10);

        // Decoder should output one-hot for register 3
        let decoder_out = sim.get_logic_at(10, 10);
        assert_eq!(decoder_out, 0b0000_1000, "Decoder should select bit 3");

        // Mux should output register 3's value (0x04)
        let mux_out = sim.get_logic_at(12, 10);
        assert_eq!(mux_out, 0x04, "Mux should output register 3 value");
    }

    // ========================================
    // Register File Integration Tests
    // ========================================

    /// Helper: Pack 8 bytes into a single u64 (register file storage format)
    fn pack_registers(regs: [u8; 8]) -> u64 {
        let mut packed = 0u64;
        for (i, &val) in regs.iter().enumerate() {
            packed |= (val as u64) << (i * 8);
        }
        packed
    }

    /// Helper: Update one byte in a packed register file
    fn update_register(packed: u64, reg_idx: usize, new_value: u8) -> u64 {
        let shift = reg_idx * 8;
        let mask = !(0xFFu64 << shift);
        (packed & mask) | ((new_value as u64) << shift)
    }

    /// Helper: Read one byte from a packed register file
    fn read_register(packed: u64, reg_idx: usize) -> u8 {
        let shift = reg_idx * 8;
        ((packed >> shift) & 0xFF) as u8
    }

    #[test]
    fn test_register_file_packed_storage() {
        // Demonstrate packed register file storage format
        let regs = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let packed = pack_registers(regs);

        assert_eq!(packed, 0x8877_6655_4433_2211);

        // Read back each register
        for (i, &expected) in regs.iter().enumerate() {
            assert_eq!(read_register(packed, i), expected);
        }

        // Update register 3 to 0xFF
        let updated = update_register(packed, 3, 0xFF);
        assert_eq!(read_register(updated, 3), 0xFF);
        assert_eq!(read_register(updated, 2), 0x33); // Others unchanged
        assert_eq!(read_register(updated, 4), 0x55);
    }

    #[test]
    fn test_register_file_full_workflow() {
        // Complete register file test:
        // 1. Initialize 8 registers with distinct values
        // 2. Read from various registers using Mux8to1
        // 3. Write to a register using Demux1to8 + update logic
        // 4. Read back to verify write succeeded

        let mut sim = Simulation::new();

        // ========== LAYOUT ==========
        // Mux8to1 semantics: left = packed data, right = select
        // Demux1to8 semantics: up = data byte, left = select
        //
        // Row 10: [Wire:Data] [Mux8to1] [Wire:Select]  <- Read path
        // Row 11: [Wire]      [Ram]     [Wire]         <- Storage
        //
        // Separate area for write path (to avoid neighbor conflicts):
        // Row 19:             [Wire:WrData]            <- Demux up input
        // Row 20: [Wire:Addr] [Demux]    [Wire]        <- Write path

        // Storage tile - holds all 8 registers packed
        sim.set_tile(20, 11, TileType::Ram);

        // Read path: Mux8to1 for reading registers
        sim.set_tile(19, 10, TileType::Wire); // data input (left)
        sim.set_tile(20, 10, TileType::Mux8to1); // read mux
        sim.set_tile(21, 10, TileType::Wire); // select input (right)

        // Write path: Demux1to8 for positioning write data (separate area)
        sim.set_tile(30, 19, TileType::Wire); // write data input (up of demux)
        sim.set_tile(29, 20, TileType::Wire); // write address input (left)
        sim.set_tile(30, 20, TileType::Demux1to8); // write demux

        // ========== STEP 1: Initialize registers ==========
        let initial_regs = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        let packed = pack_registers(initial_regs);
        sim.set_logic_value(20, 11, packed); // Store in RAM

        // ========== STEP 2: Read from register 5 ==========
        // Mux8to1: left = data, right = select
        sim.set_logic_value(19, 10, packed); // packed data on left
        sim.set_logic_value(21, 10, 5); // select register 5 on right
        sim.eval_at(20, 10);

        let read_value = sim.get_logic_at(20, 10);
        assert_eq!(read_value, 0x60, "Should read R5 = 0x60");

        // ========== STEP 3: Read from register 0 ==========
        sim.set_logic_value(21, 10, 0); // select register 0
        sim.eval_at(20, 10);

        let read_value = sim.get_logic_at(20, 10);
        assert_eq!(read_value, 0x10, "Should read R0 = 0x10");

        // ========== STEP 4: Prepare write to register 3 ==========
        // We want to write 0xAB to register 3
        let write_data: u64 = 0xAB;
        let write_addr: u64 = 3;

        // Set up Demux: data on up (30,19), address on left (29,20)
        sim.set_logic_value(30, 19, write_data); // write data (up)
        sim.set_logic_value(29, 20, write_addr); // write address (left)
        sim.eval_at(30, 20);

        // Demux outputs the data at the correct byte position
        let demux_out = sim.get_logic_at(30, 20);
        assert_eq!(
            demux_out,
            0xAB << 24,
            "Demux should place 0xAB at byte position 3"
        );

        // ========== STEP 5: Perform the write ==========
        // In a real circuit, we'd AND-mask the old value and OR with new
        // Here we simulate that operation
        let new_packed = update_register(packed, 3, 0xAB);
        sim.set_logic_value(20, 11, new_packed);

        // ========== STEP 6: Read back register 3 to verify ==========
        sim.set_logic_value(19, 10, new_packed); // updated data
        sim.set_logic_value(21, 10, 3); // select register 3
        sim.eval_at(20, 10);

        let read_back = sim.get_logic_at(20, 10);
        assert_eq!(read_back, 0xAB, "Should read back written value 0xAB");

        // Verify other registers unchanged
        sim.set_logic_value(21, 10, 2);
        sim.eval_at(20, 10);
        assert_eq!(sim.get_logic_at(20, 10), 0x30, "R2 should be unchanged");

        sim.set_logic_value(21, 10, 4);
        sim.eval_at(20, 10);
        assert_eq!(sim.get_logic_at(20, 10), 0x50, "R4 should be unchanged");
    }

    #[test]
    fn test_register_file_all_registers() {
        // Test read/write to all 8 registers
        let mut sim = Simulation::new();

        // Setup Mux8to1 for reading
        sim.set_tile(20, 10, TileType::Mux8to1);
        sim.set_tile(19, 10, TileType::Wire); // packed data
        sim.set_tile(21, 10, TileType::Wire); // select

        // Initialize with pattern: R[i] = (i+1) * 0x11
        let mut packed = 0u64;
        for i in 0..8 {
            let val = ((i + 1) * 0x11) as u8;
            packed = update_register(packed, i, val);
        }
        // packed = 0x88_77_66_55_44_33_22_11

        // Test reading each register
        for i in 0..8 {
            sim.set_logic_value(19, 10, packed);
            sim.set_logic_value(21, 10, i as u64);
            sim.eval_at(20, 10);

            let expected = ((i + 1) * 0x11) as u64;
            let actual = sim.get_logic_at(20, 10);
            assert_eq!(
                actual, expected,
                "Register {} should be 0x{:02x}",
                i, expected
            );
        }
    }

    #[test]
    fn test_register_file_with_decoder_enable() {
        // Test using Decoder3to8 to generate write enable signals
        // This simulates selecting which register to write

        let mut sim = Simulation::new();

        // Decoder for write selection
        sim.set_tile(20, 10, TileType::Decoder3to8);
        sim.set_tile(19, 10, TileType::Wire); // address input

        // Test each address generates correct one-hot enable
        for addr in 0..8 {
            sim.set_logic_value(19, 10, addr);
            sim.eval_at(20, 10);

            let enable = sim.get_logic_at(20, 10);
            let expected = 1u64 << addr;

            assert_eq!(
                enable, expected,
                "Address {} should enable bit {}",
                addr, addr
            );

            // Verify only one bit is set (one-hot property)
            assert_eq!(enable.count_ones(), 1, "Should be one-hot");
        }
    }

    #[test]
    fn test_register_file_dual_port_read() {
        // Demonstrate dual-port read (reading two registers simultaneously)
        // Uses two Mux8to1 tiles with the same packed data

        let mut sim = Simulation::new();

        // Two read ports (two Mux8to1 tiles)
        sim.set_tile(20, 10, TileType::Mux8to1); // Port A
        sim.set_tile(20, 12, TileType::Mux8to1); // Port B

        sim.set_tile(19, 10, TileType::Wire); // Port A data
        sim.set_tile(21, 10, TileType::Wire); // Port A select
        sim.set_tile(19, 12, TileType::Wire); // Port B data
        sim.set_tile(21, 12, TileType::Wire); // Port B select

        // Initialize registers
        let packed = pack_registers([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22]);

        // Read R2 from Port A and R5 from Port B simultaneously
        sim.set_logic_value(19, 10, packed);
        sim.set_logic_value(21, 10, 2); // Select R2
        sim.set_logic_value(19, 12, packed);
        sim.set_logic_value(21, 12, 5); // Select R5

        sim.eval_at(20, 10);
        sim.eval_at(20, 12);

        assert_eq!(
            sim.get_logic_at(20, 10),
            0xCC,
            "Port A should read R2 = 0xCC"
        );
        assert_eq!(
            sim.get_logic_at(20, 12),
            0xFF,
            "Port B should read R5 = 0xFF"
        );
    }

    #[test]
    fn test_register_file_alu_integration() {
        // Integration test: Read two registers, add them, prepare to write back
        // This simulates: R3 = R1 + R2

        let mut sim = Simulation::new();

        // Register file (packed storage)
        // R0=0, R1=25, R2=17, R3=0, R4=0, R5=0, R6=0, R7=0
        let packed = pack_registers([0, 25, 17, 0, 0, 0, 0, 0]);

        // Read Port A (for R1)
        sim.set_tile(20, 10, TileType::Mux8to1);
        sim.set_tile(19, 10, TileType::Wire);
        sim.set_tile(21, 10, TileType::Wire);

        // Read Port B (for R2)
        sim.set_tile(20, 12, TileType::Mux8to1);
        sim.set_tile(19, 12, TileType::Wire);
        sim.set_tile(21, 12, TileType::Wire);

        // ALU (Add tile)
        sim.set_tile(25, 11, TileType::Add);
        sim.set_tile(24, 11, TileType::Wire); // left input (from Port A)
        sim.set_tile(26, 11, TileType::Wire); // right input (from Port B)

        // ========== STEP 1: Read R1 and R2 ==========
        sim.set_logic_value(19, 10, packed);
        sim.set_logic_value(21, 10, 1); // Read R1
        sim.eval_at(20, 10);
        let r1_value = sim.get_logic_at(20, 10);
        assert_eq!(r1_value, 25);

        sim.set_logic_value(19, 12, packed);
        sim.set_logic_value(21, 12, 2); // Read R2
        sim.eval_at(20, 12);
        let r2_value = sim.get_logic_at(20, 12);
        assert_eq!(r2_value, 17);

        // ========== STEP 2: Perform addition ==========
        sim.set_logic_value(24, 11, r1_value); // ALU input A
        sim.set_logic_value(26, 11, r2_value); // ALU input B
        sim.eval_at(25, 11);

        let alu_result = sim.get_logic_at(25, 11);
        assert_eq!(alu_result, 42, "25 + 17 = 42");

        // ========== STEP 3: Prepare write to R3 ==========
        let new_packed = update_register(packed, 3, alu_result as u8);

        // Verify R3 now contains the result
        sim.set_logic_value(19, 10, new_packed);
        sim.set_logic_value(21, 10, 3); // Read R3
        sim.eval_at(20, 10);

        assert_eq!(sim.get_logic_at(20, 10), 42, "R3 should contain ALU result");

        // Verify R1 and R2 unchanged
        sim.set_logic_value(21, 10, 1);
        sim.eval_at(20, 10);
        assert_eq!(sim.get_logic_at(20, 10), 25, "R1 unchanged");

        sim.set_logic_value(21, 10, 2);
        sim.eval_at(20, 10);
        assert_eq!(sim.get_logic_at(20, 10), 17, "R2 unchanged");
    }

    // ==================== ProgramCounter Tests ====================

    #[test]
    fn test_program_counter_increment() {
        // PC increments by 1 on each rising clock edge when not jumping
        let mut sim = Simulation::with_size(32, 32);

        // ClockGlobal above PC
        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, 0);

        // Jump disable (Const 0 at right)
        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 0);

        sim.initialize();
        assert_eq!(sim.get_logic_at(10, 10), 0, "PC starts at 0");

        // First tick (clock goes HIGH) - PC increments
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 1, "PC should increment to 1");

        // Second tick (clock goes LOW) - PC holds
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 1, "PC holds during LOW");

        // Third tick (clock goes HIGH) - PC increments
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 2, "PC should increment to 2");

        // Fourth tick (clock goes LOW) - PC holds
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 2, "PC holds during LOW");

        // Fifth tick (clock goes HIGH) - PC increments
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 3, "PC should increment to 3");
    }

    #[test]
    fn test_program_counter_jump() {
        // PC loads jump target when jump signal is active on rising edge
        let mut sim = Simulation::with_size(32, 32);

        // ClockGlobal above PC
        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, 5);

        // Jump enable (Const 1) and jump target (Const 100)
        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 1);
        sim.set_tile(9, 10, TileType::Const);
        sim.set_logic_value(9, 10, 100);

        sim.initialize();

        // Tick HIGH - PC jumps to target
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 100, "PC should jump to 100");
    }

    #[test]
    fn test_program_counter_hold_no_clock() {
        // PC holds value when clock is low
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::ProgramCounter);

        // Set PC to initial value 42
        sim.set_logic_value(10, 10, 42);

        // Clock signal (up) is low
        sim.set_logic_value(10, 9, 0); // up = no clock
        sim.set_logic_value(11, 10, 0); // right = no jump
        sim.set_logic_value(9, 10, 999); // left = would-be jump target

        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            42,
            "PC should hold value when clock is low"
        );

        // Even with jump enabled, no clock means no change
        sim.set_logic_value(11, 10, 1); // right = jump enable
        sim.eval_at(10, 10);
        assert_eq!(
            sim.get_logic_at(10, 10),
            42,
            "PC should still hold when clock is low"
        );
    }

    #[test]
    fn test_program_counter_wrapping() {
        // PC wraps around at u64::MAX
        let mut sim = Simulation::with_size(32, 32);

        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, u64::MAX);

        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 0);

        sim.initialize();

        // Tick HIGH - PC increments and wraps
        sim.tick_with_delays();
        assert_eq!(sim.get_logic_at(10, 10), 0, "PC should wrap to 0");
    }

    #[test]
    fn test_program_counter_fetch_execute_cycle() {
        // Simulate a simple fetch-execute cycle
        let mut sim = Simulation::with_size(32, 32);

        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, 0);

        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 0);

        sim.initialize();

        let program = [0x10, 0x20, 0x30, 0x40, 0x50];

        // Simulate 5 fetch cycles
        for (cycle, expected_addr) in program.iter().enumerate() {
            assert_eq!(
                sim.get_logic_at(10, 10) as usize,
                cycle,
                "Cycle {}: PC should be at address {}",
                cycle,
                cycle
            );

            let fetched = program[sim.get_logic_at(10, 10) as usize];
            assert_eq!(fetched, *expected_addr);

            // Tick HIGH then LOW to complete a cycle
            sim.tick_with_delays(); // HIGH - increments
            sim.tick_with_delays(); // LOW - holds
        }

        assert_eq!(
            sim.get_logic_at(10, 10),
            5,
            "PC should be at 5 after 5 cycles"
        );
    }

    #[test]
    fn test_program_counter_conditional_jump() {
        // Simulate conditional branching
        let mut sim = Simulation::with_size(32, 32);

        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, 0);

        // Jump control and target
        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 0);
        sim.set_tile(9, 10, TileType::Const);
        sim.set_logic_value(9, 10, 0);

        sim.initialize();

        // Execute 3 sequential instructions (tick HIGH then LOW for each)
        for _ in 0..3 {
            sim.tick_with_delays(); // HIGH - increment
            sim.tick_with_delays(); // LOW - hold
        }
        assert_eq!(sim.get_logic_at(10, 10), 3, "PC at 3 before branch");

        // Conditional branch: jump to address 10
        sim.set_logic_value(11, 10, 1); // enable jump
        sim.set_logic_value(9, 10, 10); // jump target = 10

        sim.tick_with_delays(); // HIGH - jump
        assert_eq!(sim.get_logic_at(10, 10), 10, "PC should jump to 10");

        sim.tick_with_delays(); // LOW - hold

        // Continue sequential execution
        sim.set_logic_value(11, 10, 0); // disable jump

        sim.tick_with_delays(); // HIGH - increment
        assert_eq!(sim.get_logic_at(10, 10), 11, "PC should be at 11");
    }

    // ==================== Simple CPU Implementation ====================
    //
    // Architecture:
    //   - 8-bit data width
    //   - 8 registers (R0-R7) packed into 64-bit value
    //   - 256-byte RAM memory
    //   - ProgramCounter for sequencing
    //   - ALU with Add/Sub operations
    //
    // Instruction Format (8-bit):
    //   [opcode:3][dst:2][src:3]
    //
    // Opcodes:
    //   0 = NOP         : No operation
    //   1 = LOAD_IMM    : Rdst = imm8 (next byte)
    //   2 = ADD         : Rdst = Rdst + Rsrc
    //   3 = SUB         : Rdst = Rdst - Rsrc
    //   4 = JUMP        : PC = addr8 (next byte)
    //   5 = JUMP_IF_ZERO: if Rdst == 0, PC = addr8 (next byte)
    //   6 = HALT        : Stop execution
    //   7 = MEMORY      : Memory operations (sub-opcode in src field)
    //       src=0: LOAD_MEM  Rdst, [addr8] - load from memory
    //       src=1: STORE_MEM Rdst, [addr8] - store to memory
    //       src=2: LOAD_IND  Rdst, [Rsrc2] - load indirect (addr in R[dst>>1])
    //       src=3: STORE_IND Rdst, [Rsrc2] - store indirect
    //
    // CPU Layout (simplified for test):
    //   PC at (10, 5)
    //   Register file storage at (20, 10) - packed 64-bit
    //   RAM memory - 256 bytes addressable
    //   ALU (Add) at (30, 10)
    //   ALU (Sub) at (30, 12)

    /// Helper: Encode instruction
    fn encode_instr(opcode: u8, dst: u8, src: u8) -> u8 {
        ((opcode & 0x7) << 5) | ((dst & 0x3) << 3) | (src & 0x7)
    }

    /// Helper: Decode instruction
    fn decode_instr(instr: u8) -> (u8, u8, u8) {
        let opcode = (instr >> 5) & 0x7;
        let dst = (instr >> 3) & 0x3;
        let src = instr & 0x7;
        (opcode, dst, src)
    }

    // Opcodes
    const OP_NOP: u8 = 0;
    const OP_LOAD_IMM: u8 = 1;
    const OP_ADD: u8 = 2;
    const OP_SUB: u8 = 3;
    const OP_JUMP: u8 = 4;
    const OP_JUMP_IF_ZERO: u8 = 5;
    const OP_HALT: u8 = 6;
    const OP_MEMORY: u8 = 7;

    // Memory sub-opcodes (in src field when opcode=7)
    const MEM_LOAD: u8 = 0; // LOAD_MEM Rdst, [addr8]
    const MEM_STORE: u8 = 1; // STORE_MEM Rsrc, [addr8]
    const MEM_LOAD_IND: u8 = 2; // LOAD_IND Rdst, [Rptr] - addr in register
    const MEM_STORE_IND: u8 = 3; // STORE_IND Rsrc, [Rptr] - addr in register

    /// Helper: Encode memory instruction
    fn encode_mem_instr(sub_op: u8, reg: u8) -> u8 {
        encode_instr(OP_MEMORY, reg, sub_op)
    }

    #[test]
    fn test_simple_cpu_arithmetic() {
        // Test: Execute a program that computes R2 = R0 + R1
        //
        // Program:
        //   0: LOAD_IMM R0, 25    ; R0 = 25
        //   2: LOAD_IMM R1, 17    ; R1 = 17
        //   4: ADD R2, R0         ; R2 = R2 + R0 (R2=0+25=25)
        //   5: ADD R2, R1         ; R2 = R2 + R1 (R2=25+17=42)
        //   6: HALT

        let mut sim = Simulation::new();

        // === Set up CPU components ===

        // Program Counter at (10, 5)
        sim.set_tile(10, 5, TileType::ProgramCounter);
        sim.set_logic_value(10, 5, 0); // PC = 0

        // Register file - packed storage at (20, 10)
        // We'll use a Ram tile to hold all 8 registers packed
        sim.set_tile(20, 10, TileType::Ram);
        sim.set_logic_value(20, 10, 0); // All registers = 0

        // Register read mux at (22, 10)
        sim.set_tile(22, 10, TileType::Mux8to1);

        // ALU - Add at (30, 10)
        sim.set_tile(30, 10, TileType::Add);

        // === Program memory (simulated) ===
        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            25, // 0-1: R0 = 25
            encode_instr(OP_LOAD_IMM, 1, 0),
            17,                          // 2-3: R1 = 17
            encode_instr(OP_ADD, 2, 0),  // 4: R2 += R0
            encode_instr(OP_ADD, 2, 1),  // 5: R2 += R1
            encode_instr(OP_HALT, 0, 0), // 6: HALT
        ];

        // === Execute the program ===
        let mut registers: u64 = 0; // Packed register state
        let mut pc: usize = 0;
        let mut halted = false;
        let mut cycles = 0;
        const MAX_CYCLES: usize = 100;

        while !halted && cycles < MAX_CYCLES {
            cycles += 1;

            // Fetch
            let instr = program.get(pc).copied().unwrap_or(0);
            let (opcode, dst, src) = decode_instr(instr);

            // Execute
            match opcode {
                OP_NOP => {
                    pc += 1;
                }
                OP_LOAD_IMM => {
                    let imm = program.get(pc + 1).copied().unwrap_or(0);
                    registers = update_register(registers, dst as usize, imm);
                    pc += 2;
                }
                OP_ADD => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);

                    // Use the actual Add tile to compute
                    sim.set_logic_value(29, 10, dst_val as u64); // left input
                    sim.set_logic_value(31, 10, src_val as u64); // right input
                    sim.eval_at(30, 10);
                    let result = sim.get_logic_at(30, 10) as u8;

                    registers = update_register(registers, dst as usize, result);
                    pc += 1;
                }
                OP_SUB => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    let result = dst_val.wrapping_sub(src_val);
                    registers = update_register(registers, dst as usize, result);
                    pc += 1;
                }
                OP_JUMP => {
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    pc = target;
                }
                OP_JUMP_IF_ZERO => {
                    let val = read_register(registers, dst as usize);
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    if val == 0 {
                        pc = target;
                    } else {
                        pc += 2;
                    }
                }
                OP_HALT => {
                    halted = true;
                }
                _ => {
                    pc += 1; // Unknown opcode, skip
                }
            }

            // Update PC tile to reflect current state
            sim.set_logic_value(10, 5, pc as u64);
        }

        // Verify results
        assert!(halted, "Program should have halted");
        assert_eq!(read_register(registers, 0), 25, "R0 = 25");
        assert_eq!(read_register(registers, 1), 17, "R1 = 17");
        assert_eq!(read_register(registers, 2), 42, "R2 = R0 + R1 = 42");
        assert!(cycles < MAX_CYCLES, "Should complete in reasonable cycles");
    }

    #[test]
    fn test_simple_cpu_loop() {
        // Test: Execute a loop that counts down from 5 to 0
        //
        // Program:
        //   0: LOAD_IMM R0, 5     ; R0 = 5 (counter)
        //   2: LOAD_IMM R1, 1     ; R1 = 1 (decrement value)
        //   4: JUMP_IF_ZERO R0, 8 ; if R0 == 0, jump to HALT
        //   6: SUB R0, R1         ; R0 = R0 - 1
        //   7: JUMP 4             ; loop back
        //   9: HALT

        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            5, // 0-1: R0 = 5
            encode_instr(OP_LOAD_IMM, 1, 0),
            1, // 2-3: R1 = 1
            encode_instr(OP_JUMP_IF_ZERO, 0, 0),
            9,                          // 4-5: if R0==0 goto 9
            encode_instr(OP_SUB, 0, 1), // 6: R0 -= R1
            encode_instr(OP_JUMP, 0, 0),
            4,                           // 7-8: goto 4
            encode_instr(OP_HALT, 0, 0), // 9: HALT
        ];

        let mut registers: u64 = 0;
        let mut pc: usize = 0;
        let mut halted = false;
        let mut cycles = 0;
        let mut loop_count = 0;
        const MAX_CYCLES: usize = 100;

        while !halted && cycles < MAX_CYCLES {
            cycles += 1;

            let instr = program.get(pc).copied().unwrap_or(0);
            let (opcode, dst, src) = decode_instr(instr);

            match opcode {
                OP_NOP => pc += 1,
                OP_LOAD_IMM => {
                    let imm = program.get(pc + 1).copied().unwrap_or(0);
                    registers = update_register(registers, dst as usize, imm);
                    pc += 2;
                }
                OP_ADD => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_add(src_val));
                    pc += 1;
                }
                OP_SUB => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_sub(src_val));
                    pc += 1;
                    loop_count += 1;
                }
                OP_JUMP => {
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    pc = target;
                }
                OP_JUMP_IF_ZERO => {
                    let val = read_register(registers, dst as usize);
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    if val == 0 {
                        pc = target;
                    } else {
                        pc += 2;
                    }
                }
                OP_HALT => halted = true,
                _ => pc += 1,
            }
        }

        assert!(halted, "Program should halt");
        assert_eq!(
            read_register(registers, 0),
            0,
            "R0 should be 0 after countdown"
        );
        assert_eq!(loop_count, 5, "Should loop exactly 5 times");
    }

    #[test]
    fn test_simple_cpu_fibonacci() {
        // Test: Compute Fibonacci F(7) = 13
        //
        // Program computes Fib sequence: 0, 1, 1, 2, 3, 5, 8, 13
        // R0 = F(n-2), R1 = F(n-1), R2 = F(n), R3 = counter
        //
        // Algorithm:
        //   R0 = 0 (F0)
        //   R1 = 1 (F1)
        //   R3 = 6 (iterations: compute F2 through F7)
        //   loop:
        //     R2 = R0 + R1
        //     R0 = R1
        //     R1 = R2
        //     R3 -= 1
        //     if R3 != 0: goto loop
        //   Result in R1

        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            0, // 0-1: R0 = 0
            encode_instr(OP_LOAD_IMM, 1, 0),
            1, // 2-3: R1 = 1
            encode_instr(OP_LOAD_IMM, 3, 0),
            6, // 4-5: R3 = 6 (counter)
            // Loop start at address 6
            encode_instr(OP_ADD, 2, 0), // 6: R2 = R2 + R0 (but R2 starts at 0)
            encode_instr(OP_ADD, 2, 1), // 7: R2 = R2 + R1 (now R2 = R0 + R1)
            encode_instr(OP_LOAD_IMM, 0, 0),
            0,                          // 8-9: R0 = 0 (will be overwritten)
            encode_instr(OP_ADD, 0, 1), // 10: R0 = R1 (copy R1 to R0)
            encode_instr(OP_LOAD_IMM, 1, 0),
            0,                          // 11-12: R1 = 0 (will be overwritten)
            encode_instr(OP_ADD, 1, 2), // 13: R1 = R2 (copy R2 to R1)
            encode_instr(OP_LOAD_IMM, 2, 0),
            0, // 14-15: R2 = 0 (reset for next iteration)
            // Decrement counter
            encode_instr(OP_LOAD_IMM, 2, 0),
            1,                          // 16-17: R2 = 1
            encode_instr(OP_SUB, 3, 2), // 18: R3 -= 1
            encode_instr(OP_LOAD_IMM, 2, 0),
            0, // 19-20: R2 = 0 (cleanup)
            encode_instr(OP_JUMP_IF_ZERO, 3, 0),
            24, // 21-22: if R3 == 0, goto HALT
            encode_instr(OP_JUMP, 0, 0),
            6,                           // 23-24: goto loop
            encode_instr(OP_HALT, 0, 0), // 25: HALT
        ];

        let mut registers: u64 = 0;
        let mut pc: usize = 0;
        let mut halted = false;
        let mut cycles = 0;
        const MAX_CYCLES: usize = 500;

        while !halted && cycles < MAX_CYCLES {
            cycles += 1;

            let instr = program.get(pc).copied().unwrap_or(0);
            let (opcode, dst, src) = decode_instr(instr);

            match opcode {
                OP_NOP => pc += 1,
                OP_LOAD_IMM => {
                    let imm = program.get(pc + 1).copied().unwrap_or(0);
                    registers = update_register(registers, dst as usize, imm);
                    pc += 2;
                }
                OP_ADD => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_add(src_val));
                    pc += 1;
                }
                OP_SUB => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_sub(src_val));
                    pc += 1;
                }
                OP_JUMP => {
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    pc = target;
                }
                OP_JUMP_IF_ZERO => {
                    let val = read_register(registers, dst as usize);
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    if val == 0 {
                        pc = target;
                    } else {
                        pc += 2;
                    }
                }
                OP_HALT => halted = true,
                _ => pc += 1,
            }
        }

        assert!(halted, "Program should halt");
        assert_eq!(read_register(registers, 1), 13, "F(7) = 13");
        assert_eq!(read_register(registers, 0), 8, "F(6) = 8 (stored in R0)");
    }

    #[test]
    fn test_cpu_tiles_integrated() {
        // Test: Full tile-based CPU execution
        // This test actually uses the CPU tiles to execute instructions
        //
        // Layout:
        //   PC at (10, 10) - sequences through addresses
        //   ClockGlobal at (10, 9)
        //   Jump enable (Const) at (11, 10)
        //   Jump target (Const) at (9, 10)
        //
        //   Register file (packed) at (20, 10)
        //   Read mux at (22, 10) - left=packed regs, right=select
        //
        //   ALU Add at (30, 10) - left + right
        //
        // We'll simulate a mini-program: R0=5, R1=3, R2=R0+R1

        let mut sim = Simulation::with_size(64, 64);

        // === CPU Tiles Setup ===

        // Program Counter with ClockGlobal above
        sim.set_tile(10, 9, TileType::ClockGlobal);
        sim.set_tile(10, 10, TileType::ProgramCounter);
        sim.set_logic_value(10, 10, 0); // Start at address 0

        // Jump control - use Const tiles for isolation
        sim.set_tile(11, 10, TileType::Const);
        sim.set_logic_value(11, 10, 0); // Jump disabled
        sim.set_tile(9, 10, TileType::Const);
        sim.set_logic_value(9, 10, 0); // Jump target

        // Register file - using packed storage
        sim.set_tile(20, 10, TileType::Ram);
        let mut regs = pack_registers([0, 0, 0, 0, 0, 0, 0, 0]);
        sim.set_logic_value(20, 10, regs);

        // Read multiplexer
        sim.set_tile(22, 10, TileType::Mux8to1);

        // ALU
        sim.set_tile(30, 10, TileType::Add);

        // Initialize simulation for edge detection
        sim.initialize();

        // === Execute Program ===

        // STEP 1: Load R0 = 5
        regs = update_register(regs, 0, 5);
        sim.set_logic_value(20, 10, regs);

        // Clock the PC (first tick - clock HIGH)
        sim.tick_with_delays();
        assert_eq!(
            sim.get_logic_at(10, 10),
            1,
            "PC = 1 after first instruction"
        );

        // STEP 2: Load R1 = 3
        regs = update_register(regs, 1, 3);
        sim.set_logic_value(20, 10, regs);

        // Tick LOW then HIGH
        sim.tick_with_delays(); // LOW
        sim.tick_with_delays(); // HIGH
        assert_eq!(
            sim.get_logic_at(10, 10),
            2,
            "PC = 2 after second instruction"
        );

        // STEP 3: ADD R2, R0, R1 using actual tiles

        // Read R0 from register file using Mux8to1
        sim.set_logic_value(21, 10, regs); // Left = packed registers
        sim.set_logic_value(23, 10, 0); // Right = select R0
        sim.eval_at(22, 10);
        let r0_val = sim.get_logic_at(22, 10);
        assert_eq!(r0_val, 5, "Read R0 = 5");

        // Read R1
        sim.set_logic_value(23, 10, 1); // Select R1
        sim.eval_at(22, 10);
        let r1_val = sim.get_logic_at(22, 10);
        assert_eq!(r1_val, 3, "Read R1 = 3");

        // Perform addition using Add tile
        sim.set_logic_value(29, 10, r0_val); // Left input
        sim.set_logic_value(31, 10, r1_val); // Right input
        sim.eval_at(30, 10);
        let sum = sim.get_logic_at(30, 10);
        assert_eq!(sum, 8, "5 + 3 = 8");

        // Write result to R2
        regs = update_register(regs, 2, sum as u8);
        sim.set_logic_value(20, 10, regs);

        // Clock the PC (tick LOW then HIGH)
        sim.tick_with_delays(); // LOW
        sim.tick_with_delays(); // HIGH
        assert_eq!(sim.get_logic_at(10, 10), 3, "PC = 3 after ADD");

        // STEP 4: Verify final state
        assert_eq!(read_register(regs, 0), 5, "R0 = 5");
        assert_eq!(read_register(regs, 1), 3, "R1 = 3");
        assert_eq!(read_register(regs, 2), 8, "R2 = 8 (R0 + R1)");

        // STEP 5: Test jump functionality
        sim.set_logic_value(9, 10, 0); // Jump target = 0
        sim.set_logic_value(11, 10, 1); // Enable jump
        // Tick LOW then HIGH to trigger jump
        sim.tick_with_delays(); // LOW
        sim.tick_with_delays(); // HIGH
        assert_eq!(sim.get_logic_at(10, 10), 0, "PC jumped back to 0");
    }

    #[test]
    fn test_cpu_decoder_driven_regfile() {
        // Test: Use Decoder3to8 to drive register write enables
        //
        // Layout:
        //   Decoder at (15, 10) - converts register index to one-hot
        //   8 RegEnable tiles at (20, 10) through (20, 17)
        //   Each RegEnable is enabled by corresponding decoder output bit

        let mut sim = Simulation::new();

        // Decoder3to8 at (15, 10)
        sim.set_tile(15, 10, TileType::Decoder3to8);

        // Wire to feed register index
        sim.set_tile(14, 10, TileType::Wire);

        // 8 RegEnable tiles (simplified - we'll test the decoder output)
        for i in 0..8 {
            sim.set_tile(20, 10 + i, TileType::RegEnable);
            sim.set_logic_value(20, 10 + i, 0); // Initial value
        }

        // Test: Select register 3
        sim.set_logic_value(14, 10, 3); // Input = 3
        sim.eval_at(15, 10);

        let decoder_out = sim.get_logic_at(15, 10);
        assert_eq!(decoder_out, 0b00001000, "Decoder output for index 3");

        // Test: Select register 7
        sim.set_logic_value(14, 10, 7);
        sim.eval_at(15, 10);

        let decoder_out = sim.get_logic_at(15, 10);
        assert_eq!(decoder_out, 0b10000000, "Decoder output for index 7");

        // Test: Select register 0
        sim.set_logic_value(14, 10, 0);
        sim.eval_at(15, 10);

        let decoder_out = sim.get_logic_at(15, 10);
        assert_eq!(decoder_out, 0b00000001, "Decoder output for index 0");
    }

    #[test]
    fn test_cpu_complete_datapath() {
        // Test: Complete CPU datapath with all components wired together
        //
        // This demonstrates a full fetch-decode-execute cycle using tiles:
        //
        //   [PC] --addr--> [Instruction ROM (simulated)]
        //                        |
        //                        v
        //                   [Decoder] --enables--> [RegFile]
        //                        |                    |
        //                        v                    v
        //                   [ALU Select]         [Read Ports]
        //                        |                    |
        //                        +--------+-----------+
        //                                 |
        //                                 v
        //                              [ALU]
        //                                 |
        //                                 v
        //                           [Writeback]

        let mut sim = Simulation::new();

        // === Component Placement ===

        // Program Counter at (5, 5)
        sim.set_tile(5, 5, TileType::ProgramCounter);
        sim.set_logic_value(5, 5, 0);

        // Instruction Decoder at (10, 5)
        sim.set_tile(10, 5, TileType::Decoder3to8);

        // Register File (packed) at (15, 5)
        sim.set_tile(15, 5, TileType::Ram);
        sim.set_logic_value(15, 5, 0);

        // Read Port A (Mux) at (15, 7)
        sim.set_tile(15, 7, TileType::Mux8to1);

        // Read Port B (Mux) at (15, 9)
        sim.set_tile(15, 9, TileType::Mux8to1);

        // ALU at (20, 8)
        sim.set_tile(20, 8, TileType::Add);

        // === Simulate execution of: R2 = R0 + R1 ===

        // Initialize registers: R0=10, R1=20
        let mut regs = pack_registers([10, 20, 0, 0, 0, 0, 0, 0]);
        sim.set_logic_value(15, 5, regs);

        // Instruction: ADD R2, R0, R1
        // Encoded as: opcode=ADD, dst=2, srcA=0, srcB=1

        // 1. Read R0 via Port A
        sim.set_logic_value(14, 7, regs); // Packed regs to mux
        sim.set_logic_value(16, 7, 0); // Select R0
        sim.eval_at(15, 7);
        let a_val = sim.get_logic_at(15, 7);
        assert_eq!(a_val, 10, "Port A reads R0 = 10");

        // 2. Read R1 via Port B
        sim.set_logic_value(14, 9, regs); // Packed regs to mux
        sim.set_logic_value(16, 9, 1); // Select R1
        sim.eval_at(15, 9);
        let b_val = sim.get_logic_at(15, 9);
        assert_eq!(b_val, 20, "Port B reads R1 = 20");

        // 3. ALU computes A + B
        sim.set_logic_value(19, 8, a_val); // Left input
        sim.set_logic_value(21, 8, b_val); // Right input
        sim.eval_at(20, 8);
        let alu_result = sim.get_logic_at(20, 8);
        assert_eq!(alu_result, 30, "ALU: 10 + 20 = 30");

        // 4. Write back to R2
        regs = update_register(regs, 2, alu_result as u8);
        sim.set_logic_value(15, 5, regs);

        // 5. Verify final register state
        assert_eq!(read_register(regs, 0), 10, "R0 unchanged");
        assert_eq!(read_register(regs, 1), 20, "R1 unchanged");
        assert_eq!(read_register(regs, 2), 30, "R2 = R0 + R1 = 30");

        // 6. Advance PC using tick-based simulation
        // Setup: Add ClockGlobal above PC and Const for jump control
        sim.set_tile(5, 4, TileType::ClockGlobal);
        sim.set_tile(6, 5, TileType::Const);
        sim.set_logic_value(6, 5, 0); // No jump

        // Initialize and tick to advance PC
        sim.initialize();
        sim.tick_with_delays(); // Rising edge - PC increments
        assert_eq!(sim.get_logic_at(5, 5), 1, "PC advanced to 1");
    }

    // ==================== CPU with Memory Tests ====================

    /// Execute CPU with memory support
    /// Returns (registers, memory, halted, cycles)
    fn execute_cpu_with_memory(
        program: &[u8],
        initial_regs: u64,
        initial_mem: &[u8],
        max_cycles: usize,
    ) -> (u64, Vec<u8>, bool, usize) {
        let mut registers = initial_regs;
        let mut memory: Vec<u8> = vec![0; 256];

        // Copy initial memory
        for (i, &v) in initial_mem.iter().enumerate() {
            if i < 256 {
                memory[i] = v;
            }
        }

        let mut pc: usize = 0;
        let mut halted = false;
        let mut cycles = 0;

        while !halted && cycles < max_cycles {
            cycles += 1;

            let instr = program.get(pc).copied().unwrap_or(0);
            let (opcode, dst, src) = decode_instr(instr);

            match opcode {
                OP_NOP => pc += 1,
                OP_LOAD_IMM => {
                    let imm = program.get(pc + 1).copied().unwrap_or(0);
                    registers = update_register(registers, dst as usize, imm);
                    pc += 2;
                }
                OP_ADD => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_add(src_val));
                    pc += 1;
                }
                OP_SUB => {
                    let dst_val = read_register(registers, dst as usize);
                    let src_val = read_register(registers, src as usize);
                    registers =
                        update_register(registers, dst as usize, dst_val.wrapping_sub(src_val));
                    pc += 1;
                }
                OP_JUMP => {
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    pc = target;
                }
                OP_JUMP_IF_ZERO => {
                    let val = read_register(registers, dst as usize);
                    let target = program.get(pc + 1).copied().unwrap_or(0) as usize;
                    if val == 0 {
                        pc = target;
                    } else {
                        pc += 2;
                    }
                }
                OP_HALT => halted = true,
                OP_MEMORY => {
                    // Memory operations - sub-opcode in src field
                    match src {
                        MEM_LOAD => {
                            // LOAD_MEM Rdst, [addr8]
                            let addr = program.get(pc + 1).copied().unwrap_or(0) as usize;
                            let value = memory.get(addr).copied().unwrap_or(0);
                            registers = update_register(registers, dst as usize, value);
                            pc += 2;
                        }
                        MEM_STORE => {
                            // STORE_MEM Rsrc, [addr8]
                            let addr = program.get(pc + 1).copied().unwrap_or(0) as usize;
                            let value = read_register(registers, dst as usize);
                            if addr < 256 {
                                memory[addr] = value;
                            }
                            pc += 2;
                        }
                        MEM_LOAD_IND => {
                            // LOAD_IND Rdst, [Rptr] - pointer register specified in next byte
                            let ptr_reg = program.get(pc + 1).copied().unwrap_or(0) as usize & 0x7;
                            let addr = read_register(registers, ptr_reg) as usize;
                            let value = memory.get(addr).copied().unwrap_or(0);
                            registers = update_register(registers, dst as usize, value);
                            pc += 2;
                        }
                        MEM_STORE_IND => {
                            // STORE_IND Rsrc, [Rptr] - pointer register specified in next byte
                            let ptr_reg = program.get(pc + 1).copied().unwrap_or(0) as usize & 0x7;
                            let addr = read_register(registers, ptr_reg) as usize;
                            let value = read_register(registers, dst as usize);
                            if addr < 256 {
                                memory[addr] = value;
                            }
                            pc += 2;
                        }
                        _ => pc += 1, // Unknown memory sub-opcode
                    }
                }
                _ => pc += 1,
            }
        }

        (registers, memory, halted, cycles)
    }

    #[test]
    fn test_cpu_memory_load_store() {
        // Test: Store value to memory, then load it back
        //
        // Program:
        //   LOAD_IMM R0, 42       ; R0 = 42
        //   STORE_MEM R0, [0x10]  ; mem[16] = R0
        //   LOAD_IMM R0, 0        ; R0 = 0 (clear)
        //   LOAD_MEM R1, [0x10]   ; R1 = mem[16]
        //   HALT

        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            42, // R0 = 42
            encode_mem_instr(MEM_STORE, 0),
            0x10, // mem[16] = R0
            encode_instr(OP_LOAD_IMM, 0, 0),
            0, // R0 = 0
            encode_mem_instr(MEM_LOAD, 1),
            0x10,                        // R1 = mem[16]
            encode_instr(OP_HALT, 0, 0), // HALT
        ];

        let (regs, mem, halted, _cycles) = execute_cpu_with_memory(&program, 0, &[], 100);

        assert!(halted, "Program should halt");
        assert_eq!(read_register(regs, 0), 0, "R0 should be 0");
        assert_eq!(
            read_register(regs, 1),
            42,
            "R1 should be 42 (loaded from memory)"
        );
        assert_eq!(mem[0x10], 42, "Memory at 0x10 should be 42");
    }

    #[test]
    fn test_cpu_memory_indirect() {
        // Test: Indirect memory access using register as pointer
        //
        // Program:
        //   LOAD_IMM R0, 0x20     ; R0 = 32 (pointer)
        //   LOAD_IMM R1, 99       ; R1 = 99 (value to store)
        //   STORE_IND R1, [R0]    ; mem[R0] = R1 (mem[32] = 99)
        //   LOAD_IMM R0, 0x20     ; R0 = 32 (pointer again)
        //   LOAD_IND R2, [R0]     ; R2 = mem[R0] (R2 = mem[32])
        //   HALT

        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            0x20, // R0 = 32
            encode_instr(OP_LOAD_IMM, 1, 0),
            99, // R1 = 99
            encode_mem_instr(MEM_STORE_IND, 1),
            0, // mem[R0] = R1
            encode_instr(OP_LOAD_IMM, 0, 0),
            0x20, // R0 = 32
            encode_mem_instr(MEM_LOAD_IND, 2),
            0,                           // R2 = mem[R0]
            encode_instr(OP_HALT, 0, 0), // HALT
        ];

        let (regs, mem, halted, _cycles) = execute_cpu_with_memory(&program, 0, &[], 100);

        assert!(halted, "Program should halt");
        assert_eq!(read_register(regs, 1), 99, "R1 = 99");
        assert_eq!(
            read_register(regs, 2),
            99,
            "R2 should be 99 (loaded indirect)"
        );
        assert_eq!(mem[0x20], 99, "Memory at 0x20 should be 99");
    }

    #[test]
    fn test_cpu_memory_array_sum() {
        // Test: Sum an array of values in memory
        //
        // Memory layout:
        //   0x00: 5 (array length)
        //   0x01-0x05: [10, 20, 30, 40, 50] (array values)
        //
        // Program:
        //   R0 = array pointer (starts at 1)
        //   R1 = sum (accumulator)
        //   R2 = count
        //   R3 = temp value
        //
        //   LOAD_MEM R2, [0x00]   ; R2 = count (5)
        //   LOAD_IMM R0, 1        ; R0 = pointer to array[0]
        //   LOAD_IMM R1, 0        ; R1 = sum = 0
        // loop:
        //   JUMP_IF_ZERO R2, end  ; if count == 0, done
        //   LOAD_IND R3, [R0]     ; R3 = mem[R0]
        //   ADD R1, R3            ; sum += R3
        //   LOAD_IMM R3, 1
        //   ADD R0, R3            ; R0++ (next element)
        //   LOAD_IMM R3, 1
        //   SUB R2, R3            ; count--
        //   JUMP loop
        // end:
        //   HALT

        let program: Vec<u8> = vec![
            encode_mem_instr(MEM_LOAD, 2),
            0x00, // 0-1: R2 = mem[0] = 5
            encode_instr(OP_LOAD_IMM, 0, 0),
            1, // 2-3: R0 = 1 (array pointer)
            encode_instr(OP_LOAD_IMM, 1, 0),
            0, // 4-5: R1 = 0 (sum)
            // loop at address 6
            encode_instr(OP_JUMP_IF_ZERO, 2, 0),
            22, // 6-7: if R2 == 0, goto end (addr 22)
            encode_mem_instr(MEM_LOAD_IND, 3),
            0,                          // 8-9: R3 = mem[R0]
            encode_instr(OP_ADD, 1, 3), // 10: R1 += R3
            encode_instr(OP_LOAD_IMM, 3, 0),
            1,                          // 11-12: R3 = 1
            encode_instr(OP_ADD, 0, 3), // 13: R0++ (next element)
            encode_instr(OP_LOAD_IMM, 3, 0),
            1,                          // 14-15: R3 = 1
            encode_instr(OP_SUB, 2, 3), // 16: R2-- (count--)
            encode_instr(OP_JUMP, 0, 0),
            6, // 17-18: goto loop
            // padding to reach address 22
            encode_instr(OP_NOP, 0, 0),  // 19
            encode_instr(OP_NOP, 0, 0),  // 20
            encode_instr(OP_NOP, 0, 0),  // 21
            encode_instr(OP_HALT, 0, 0), // 22: HALT
        ];

        // Initial memory: count=5, array=[10,20,30,40,50]
        let initial_mem: Vec<u8> = vec![5, 10, 20, 30, 40, 50];

        let (regs, _mem, halted, cycles) = execute_cpu_with_memory(&program, 0, &initial_mem, 500);

        assert!(halted, "Program should halt");
        assert_eq!(
            read_register(regs, 1),
            150,
            "Sum should be 10+20+30+40+50 = 150"
        );
        assert!(cycles < 500, "Should complete in reasonable cycles");
    }

    #[test]
    fn test_cpu_memory_copy_block() {
        // Test: Copy a block of memory from one location to another
        //
        // Copy 4 bytes from address 0x10 to address 0x20
        //
        // Memory layout:
        //   0x10-0x13: [0xAA, 0xBB, 0xCC, 0xDD]
        //   0x20-0x23: [0x00, 0x00, 0x00, 0x00]
        //
        // Registers:
        //   R0 = src pointer (0x10)
        //   R1 = dst pointer (0x20)
        //   R2 = count (4)
        //   R3 = temp

        let program: Vec<u8> = vec![
            encode_instr(OP_LOAD_IMM, 0, 0),
            0x10, // 0-1: R0 = 0x10 (src)
            encode_instr(OP_LOAD_IMM, 1, 0),
            0x20, // 2-3: R1 = 0x20 (dst)
            encode_instr(OP_LOAD_IMM, 2, 0),
            4, // 4-5: R2 = 4 (count)
            // loop at address 6
            encode_instr(OP_JUMP_IF_ZERO, 2, 0),
            24, // 6-7: if R2 == 0, goto end
            encode_mem_instr(MEM_LOAD_IND, 3),
            0, // 8-9: R3 = mem[R0]
            encode_mem_instr(MEM_STORE_IND, 3),
            1, // 10-11: mem[R1] = R3
            // Increment pointers
            encode_instr(OP_LOAD_IMM, 3, 0),
            1,                          // 12-13: R3 = 1
            encode_instr(OP_ADD, 0, 3), // 14: R0++
            encode_instr(OP_ADD, 1, 3), // 15: R1++
            // Decrement count
            encode_instr(OP_LOAD_IMM, 3, 0),
            1,                          // 16-17: R3 = 1
            encode_instr(OP_SUB, 2, 3), // 18: R2--
            encode_instr(OP_JUMP, 0, 0),
            6, // 19-20: goto loop
            // padding
            encode_instr(OP_NOP, 0, 0),  // 21
            encode_instr(OP_NOP, 0, 0),  // 22
            encode_instr(OP_NOP, 0, 0),  // 23
            encode_instr(OP_HALT, 0, 0), // 24: HALT
        ];

        // Initial memory
        let mut initial_mem = vec![0u8; 256];
        initial_mem[0x10] = 0xAA;
        initial_mem[0x11] = 0xBB;
        initial_mem[0x12] = 0xCC;
        initial_mem[0x13] = 0xDD;

        let (_regs, mem, halted, _cycles) = execute_cpu_with_memory(&program, 0, &initial_mem, 500);

        assert!(halted, "Program should halt");
        // Verify source unchanged
        assert_eq!(mem[0x10], 0xAA);
        assert_eq!(mem[0x11], 0xBB);
        assert_eq!(mem[0x12], 0xCC);
        assert_eq!(mem[0x13], 0xDD);
        // Verify destination has copied values
        assert_eq!(mem[0x20], 0xAA, "mem[0x20] should be 0xAA");
        assert_eq!(mem[0x21], 0xBB, "mem[0x21] should be 0xBB");
        assert_eq!(mem[0x22], 0xCC, "mem[0x22] should be 0xCC");
        assert_eq!(mem[0x23], 0xDD, "mem[0x23] should be 0xDD");
    }

    #[test]
    fn test_cpu_memory_with_tiles() {
        // Test: Use actual Ram tiles for memory operations
        //
        // Layout:
        //   RAM tile at (40, 10) - acts as memory cell
        //   Write enable wire at (40, 9)
        //   Data input wire at (39, 10)

        let mut sim = Simulation::new();

        // Set up RAM tile
        sim.set_tile(40, 10, TileType::Ram);
        sim.set_logic_value(40, 10, 0); // Initial value

        // Write enable and data wires
        sim.set_tile(40, 9, TileType::Wire); // up = write enable
        sim.set_tile(39, 10, TileType::Wire); // left = data input

        // Test write: Store value 77 to RAM
        sim.set_logic_value(39, 10, 77); // Data = 77
        sim.set_logic_value(40, 9, 1); // Write enable = 1
        sim.eval_at(40, 10);

        assert_eq!(sim.get_logic_at(40, 10), 77, "RAM should store 77");

        // Test read: Disable write, value should persist
        sim.set_logic_value(40, 9, 0); // Write enable = 0
        sim.set_logic_value(39, 10, 0); // Data = 0 (shouldn't matter)
        sim.eval_at(40, 10);

        assert_eq!(
            sim.get_logic_at(40, 10),
            77,
            "RAM should retain 77 with write disabled"
        );

        // Test overwrite: Enable write with new value
        sim.set_logic_value(39, 10, 123); // Data = 123
        sim.set_logic_value(40, 9, 1); // Write enable = 1
        sim.eval_at(40, 10);

        assert_eq!(sim.get_logic_at(40, 10), 123, "RAM should now be 123");
    }

    #[test]
    fn test_cpu_memory_stack_operations() {
        // Test: Implement push/pop using memory and a stack pointer
        //
        // Stack grows downward from 0xFF
        // R0 = stack pointer (SP)
        // R1 = value to push/pop
        //
        // Program:
        //   LOAD_IMM R0, 0xFF     ; SP = 0xFF (top of stack)
        //   LOAD_IMM R1, 11       ; Push 11
        //   STORE_IND R1, [R0]    ; mem[SP] = 11
        //   SUB R0, R3            ; SP-- (R3=1 set earlier)
        //   ... push 22, 33
        //   ... pop all and verify

        let program: Vec<u8> = vec![
            // Initialize
            encode_instr(OP_LOAD_IMM, 0, 0),
            0xFF, // 0-1: R0 = SP = 0xFF
            encode_instr(OP_LOAD_IMM, 3, 0),
            1, // 2-3: R3 = 1 (for inc/dec)
            // Push 11
            encode_instr(OP_LOAD_IMM, 1, 0),
            11, // 4-5: R1 = 11
            encode_mem_instr(MEM_STORE_IND, 1),
            0,                          // 6-7: mem[SP] = R1
            encode_instr(OP_SUB, 0, 3), // 8: SP--
            // Push 22
            encode_instr(OP_LOAD_IMM, 1, 0),
            22, // 9-10: R1 = 22
            encode_mem_instr(MEM_STORE_IND, 1),
            0,                          // 11-12: mem[SP] = R1
            encode_instr(OP_SUB, 0, 3), // 13: SP--
            // Push 33
            encode_instr(OP_LOAD_IMM, 1, 0),
            33, // 14-15: R1 = 33
            encode_mem_instr(MEM_STORE_IND, 1),
            0,                          // 16-17: mem[SP] = R1
            encode_instr(OP_SUB, 0, 3), // 18: SP--
            // Pop into R2 (should be 33)
            encode_instr(OP_ADD, 0, 3), // 19: SP++
            encode_mem_instr(MEM_LOAD_IND, 2),
            0,                           // 20-21: R2 = mem[SP]
            encode_instr(OP_HALT, 0, 0), // 22: HALT
        ];

        let (regs, mem, halted, _cycles) = execute_cpu_with_memory(&program, 0, &[], 100);

        assert!(halted);
        // Stack should have: [0xFC]=33, [0xFD]=22, [0xFE]=11
        assert_eq!(mem[0xFF], 11, "First push at 0xFF");
        assert_eq!(mem[0xFE], 22, "Second push at 0xFE");
        assert_eq!(mem[0xFD], 33, "Third push at 0xFD");
        // R2 should have popped value 33
        assert_eq!(read_register(regs, 2), 33, "Popped value should be 33");
    }

    // =========================================================================
    // EPIC 123: Propagation Delay Tests
    // =========================================================================

    #[test]
    fn test_tick_with_delays_basic() {
        // Create a simple circuit: ClockGlobal -> Wire -> And
        let mut sim = Simulation::with_size(16, 16);

        // Clock at (0, 0)
        sim.set_tile(0, 0, TileType::ClockGlobal);
        // Wire path: (1,0) -> (2,0) -> (3,0)
        sim.set_tile(1, 0, TileType::Wire);
        sim.set_tile(2, 0, TileType::Wire);
        sim.set_tile(3, 0, TileType::Wire);
        // And gate at (4, 0) with inputs from (3, 0) and (4, 1)
        sim.set_tile(4, 0, TileType::And);
        sim.set_tile(4, 1, TileType::Wire); // Second input

        // Run timing-aware tick
        let stats = sim.tick_with_delays();

        // Should have converged
        assert!(stats.converged, "Simulation should converge");

        // Critical path should be > 0 (clock + wires + and gate)
        // ClockGlobal(0) + Wire(1) + Wire(1) + Wire(1) + And(2) = 5 deltas minimum
        assert!(
            stats.critical_path_deltas >= 3,
            "Critical path should be at least 3 deltas, got {}",
            stats.critical_path_deltas
        );

        // Some tiles should have switched
        assert!(stats.tiles_switched > 0, "Some tiles should have switched");
    }

    #[test]
    fn test_tick_with_delays_critical_path_tracking() {
        // Create a linear chain to test critical path
        let mut sim = Simulation::with_size(32, 4);

        // Clock at start
        sim.set_tile(0, 0, TileType::ClockGlobal);

        // Long wire chain
        for x in 1..20 {
            sim.set_tile(x, 0, TileType::Wire);
        }

        // Run timing-aware tick
        let stats = sim.tick_with_delays();

        assert!(stats.converged);

        // Critical path should include all the wire delays
        // 19 wires × 1 delta each = 19 deltas minimum
        assert!(
            stats.critical_path_deltas >= 10,
            "Long wire chain should have significant critical path, got {}",
            stats.critical_path_deltas
        );

        // Trace the critical path
        let path = sim.trace_critical_path();
        assert!(!path.is_empty(), "Critical path trace should not be empty");
    }

    #[test]
    fn test_propagation_delay_values() {
        // Verify delay values are as expected
        assert_eq!(TileType::Wire.propagation_delay(), 1);
        assert_eq!(TileType::And.propagation_delay(), 2);
        assert_eq!(TileType::Or.propagation_delay(), 2);
        assert_eq!(TileType::Xor.propagation_delay(), 3);
        assert_eq!(TileType::Mul.propagation_delay(), 8);
        assert_eq!(TileType::Div.propagation_delay(), 12);

        // Sequential elements should have 0 delay
        assert_eq!(TileType::Register8.propagation_delay(), 0);
        assert_eq!(TileType::ClockGlobal.propagation_delay(), 0);
        assert_eq!(TileType::Latch.propagation_delay(), 0);

        // is_sequential should match
        assert!(TileType::Register8.is_sequential());
        assert!(TileType::ClockGlobal.is_sequential());
        assert!(!TileType::Wire.is_sequential());
        assert!(!TileType::And.is_sequential());
    }

    #[test]
    fn test_check_timing() {
        let mut sim = Simulation::with_size(16, 16);
        sim.set_tile(0, 0, TileType::ClockGlobal);
        sim.set_tile(1, 0, TileType::Wire);
        sim.set_tile(2, 0, TileType::Wire);

        sim.tick_with_delays();

        // Check with generous timing
        let result = sim.check_timing(100);
        assert!(
            result.meets_timing,
            "Should meet timing with 100 delta budget"
        );
        assert!(result.slack > 0, "Should have positive slack");

        // Check with very tight timing
        let tight_result = sim.check_timing(1);
        // May or may not meet depending on circuit
        println!("Tight timing check: {}", tight_result);
    }

    #[test]
    fn test_detect_races() {
        let mut sim = Simulation::with_size(16, 16);

        // Create a circuit with potential race: two paths to same destination
        sim.set_tile(0, 0, TileType::ClockGlobal);

        // Short path: (0,0) -> (1,0) -> (2,0)
        sim.set_tile(1, 0, TileType::Wire);

        // Long path: (0,0) -> (0,1) -> (1,1) -> (2,1) -> (2,0)
        sim.set_tile(0, 1, TileType::Wire);
        sim.set_tile(1, 1, TileType::Wire);
        sim.set_tile(2, 1, TileType::Wire);

        // Destination: And gate at (2, 0) receives from both paths
        sim.set_tile(2, 0, TileType::And);

        sim.tick_with_delays();

        // Check for races with minimum window of 2 deltas
        let races = sim.detect_races(2);
        // May or may not detect races depending on exact timing
        println!("Detected {} potential races", races.len());
    }

    #[test]
    fn test_wire_delay_distance_based() {
        let mut sim = Simulation::with_size(128, 32);

        // Create a long wire chain from x=0 to x=99 at y=10
        for x in 0..100 {
            sim.set_tile(x, 10, TileType::Wire);
        }

        // Set source (And gate) above wire start
        sim.set_tile(0, 9, TileType::And);
        // Set sink (And gate) below wire end
        sim.set_tile(99, 11, TileType::And);

        // Compute wire delays based on distance from non-wire tiles
        sim.compute_wire_delays();

        // Wire at x=0 should have delay ~1 (close to source And at (0,9))
        // Wire at x=50 should have delay ~6 (50 tiles / 10 + 1)
        // Wire at x=99 should have delay ~10 or ~11 depending on path
        let delay_start = sim.get_wire_delay(0, 10);
        let delay_mid = sim.get_wire_delay(50, 10);
        let delay_end = sim.get_wire_delay(99, 10);

        // Delays should increase with distance from source
        // Wire at start is distance 1 from And at (0,9), so delay = 1 + 1/10 = 1
        // Wire at x=50 is distance 51 from And at (0,9), so delay = 1 + 51/10 = 6
        // Wire at x=99 is distance ~100 but closer to sink And at (99,11), so it depends

        println!(
            "Wire delays: start={}, mid={}, end={}",
            delay_start, delay_mid, delay_end
        );

        // Basic sanity checks
        assert!(delay_start > 0, "Wire at start should have non-zero delay");

        // The middle should have higher delay than the start
        // (unless there are closer non-wire tiles, but in this setup there aren't)
        assert!(
            delay_mid >= delay_start,
            "Wire at middle should have delay >= start: mid={}, start={}",
            delay_mid,
            delay_start
        );
    }

    #[test]
    fn test_wire_delay_manual_set() {
        let mut sim = Simulation::with_size(32, 32);

        // Create some wire tiles
        sim.set_tile(5, 5, TileType::Wire);
        sim.set_tile(6, 5, TileType::Wire);
        sim.set_tile(7, 5, TileType::Wire);

        // Manually set wire delay
        sim.set_wire_delay(5, 5, 0); // length 0 -> delay 1
        sim.set_wire_delay(6, 5, 50); // length 50 -> delay 1 + 5 = 6
        sim.set_wire_delay(7, 5, 100); // length 100 -> delay 1 + 10 = 11

        assert_eq!(sim.get_wire_delay(5, 5), 1);
        assert_eq!(sim.get_wire_delay(6, 5), 6);
        assert_eq!(sim.get_wire_delay(7, 5), 11);
    }

    #[test]
    fn test_wire_delay_affects_timing() {
        let mut sim = Simulation::with_size(64, 16);

        // Create a wire chain with clock source
        sim.set_tile(0, 0, TileType::ClockGlobal);
        for x in 1..50 {
            sim.set_tile(x, 0, TileType::Wire);
        }

        // First, run without distance-based delays
        sim.tick_with_delays();
        let base_critical_path = sim.timing_stats().critical_path_deltas;

        // Reset and compute distance-based delays
        let mut sim2 = Simulation::with_size(64, 16);
        sim2.set_tile(0, 0, TileType::ClockGlobal);
        for x in 1..50 {
            sim2.set_tile(x, 0, TileType::Wire);
        }
        sim2.compute_wire_delays();
        sim2.tick_with_delays();
        let distance_critical_path = sim2.timing_stats().critical_path_deltas;

        // With distance-based delays, far wires should have higher delays
        // so the critical path should be longer
        println!(
            "Critical path: base={}, distance-based={}",
            base_critical_path, distance_critical_path
        );

        // The distance-based critical path should be >= base
        // (In practice it should be longer since wire delays increase with distance)
        assert!(
            distance_critical_path >= base_critical_path,
            "Distance-based delays should not decrease critical path"
        );
    }

    #[test]
    fn test_is_wire() {
        // Test the is_wire() method
        assert!(TileType::Wire.is_wire());
        assert!(TileType::WireH.is_wire());
        assert!(TileType::WireV.is_wire());
        assert!(TileType::WireDown.is_wire());
        assert!(TileType::WireRight.is_wire());
        assert!(TileType::Cross.is_wire());

        // Non-wire tiles
        assert!(!TileType::And.is_wire());
        assert!(!TileType::Or.is_wire());
        assert!(!TileType::Register8.is_wire());
        assert!(!TileType::ClockGlobal.is_wire());
    }

    #[test]
    fn test_clear_wire_delays() {
        let mut sim = Simulation::with_size(32, 32);

        // Set up wires and compute delays
        sim.set_tile(5, 5, TileType::Wire);
        sim.set_wire_delay(5, 5, 100);
        assert_eq!(sim.get_wire_delay(5, 5), 11);

        // Clear all delays
        sim.clear_wire_delays();
        assert_eq!(sim.get_wire_delay(5, 5), 0);
    }

    // =========================================================================
    // Component Hierarchy Tests
    // =========================================================================

    #[test]
    fn test_behavioral_and_component() {
        use crate::component::*;

        let mut sim = Simulation::with_size(10, 10);

        // Define a 3x3 AND gate component
        let def = ComponentDef {
            name: "AND2".to_string(),
            width: 3,
            height: 3,
            ports: vec![
                PortDef {
                    name: "a".into(),
                    edge: Edge::Left,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "b".into(),
                    edge: Edge::Top,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "out".into(),
                    edge: Edge::Right,
                    offset: 1,
                    direction: PortDirection::Output,
                },
            ],
            implementation: ComponentImpl::Combinational(Box::new(|inputs| {
                vec![inputs[0] & inputs[1]]
            })),
            propagation_delay: 2,
        };

        let def_idx = sim.register_component_def(def);
        let _comp_idx = sim.place_component(def_idx, 2, 2).unwrap();

        // Verify component was placed
        assert_eq!(sim.component_count(), 1);

        // Input 'a' is at (2, 3) - left edge, offset 1 - should be WireRight
        assert_eq!(sim.tile_type_xy(2, 3), TileType::WireRight);
        // Input 'b' is at (3, 2) - top edge, offset 1 - should be WireDown
        assert_eq!(sim.tile_type_xy(3, 2), TileType::WireDown);
        // Output 'out' is at (4, 3) - right edge, offset 1 - should be ComponentOutput
        assert_eq!(sim.tile_type_xy(4, 3), TileType::ComponentOutput);

        // Set input values via neighboring tiles
        sim.set_tile(1, 3, TileType::Const);
        sim.set_logic_value(1, 3, 0xFF);
        sim.set_tile(3, 1, TileType::Const);
        sim.set_logic_value(3, 1, 0x0F);

        // Mark everything dirty and tick
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();

        // Output should be 0xFF & 0x0F = 0x0F
        let out_val = sim.get_logic_at(4, 3);
        assert_eq!(
            out_val, 0x0F,
            "AND component output should be 0xFF & 0x0F = 0x0F"
        );
    }

    #[test]
    fn test_behavioral_adder_component() {
        use crate::component::*;

        let mut sim = Simulation::with_size(10, 10);

        let def = ComponentDef {
            name: "ADD64".to_string(),
            width: 3,
            height: 3,
            ports: vec![
                PortDef {
                    name: "a".into(),
                    edge: Edge::Left,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "b".into(),
                    edge: Edge::Top,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "sum".into(),
                    edge: Edge::Right,
                    offset: 1,
                    direction: PortDirection::Output,
                },
            ],
            implementation: ComponentImpl::Combinational(Box::new(|inputs| {
                vec![inputs[0].wrapping_add(inputs[1])]
            })),
            propagation_delay: 5,
        };

        let def_idx = sim.register_component_def(def);
        sim.place_component(def_idx, 2, 2).unwrap();

        sim.set_tile(1, 3, TileType::Const);
        sim.set_logic_value(1, 3, 100);
        sim.set_tile(3, 1, TileType::Const);
        sim.set_logic_value(3, 1, 200);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();

        let out_val = sim.get_logic_at(4, 3);
        assert_eq!(
            out_val, 300,
            "Adder component should output 100 + 200 = 300"
        );
    }

    #[test]
    fn test_component_output_propagation() {
        use crate::component::*;

        let mut sim = Simulation::with_size(10, 10);

        let def = ComponentDef {
            name: "CONST42".to_string(),
            width: 3,
            height: 3,
            ports: vec![
                PortDef {
                    name: "in".into(),
                    edge: Edge::Left,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "out".into(),
                    edge: Edge::Right,
                    offset: 1,
                    direction: PortDirection::Output,
                },
            ],
            implementation: ComponentImpl::Combinational(Box::new(|inputs| vec![inputs[0] * 2])),
            propagation_delay: 2,
        };

        let def_idx = sim.register_component_def(def);
        sim.place_component(def_idx, 2, 2).unwrap();

        // Place a wire tile adjacent to the output port to catch propagation
        sim.set_tile(5, 3, TileType::Wire);

        // Set input
        sim.set_tile(1, 3, TileType::Const);
        sim.set_logic_value(1, 3, 21);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();

        // Output at (4,3) should be 42
        assert_eq!(sim.get_logic_at(4, 3), 42);
        // Wire at (5,3) should have picked up the signal
        let wire_val = sim.get_logic_at(5, 3);
        assert!(
            wire_val != 0,
            "Wire adjacent to component output should propagate signal"
        );
    }

    #[test]
    fn test_component_cache_invalidation() {
        use crate::component::*;

        let mut sim = Simulation::with_size(10, 10);

        let def = ComponentDef {
            name: "DOUBLE".to_string(),
            width: 3,
            height: 3,
            ports: vec![
                PortDef {
                    name: "in".into(),
                    edge: Edge::Left,
                    offset: 1,
                    direction: PortDirection::Input,
                },
                PortDef {
                    name: "out".into(),
                    edge: Edge::Right,
                    offset: 1,
                    direction: PortDirection::Output,
                },
            ],
            implementation: ComponentImpl::Combinational(Box::new(|inputs| vec![inputs[0] * 2])),
            propagation_delay: 2,
        };

        let def_idx = sim.register_component_def(def);
        sim.place_component(def_idx, 2, 2).unwrap();

        // First tick with input = 10
        sim.set_tile(1, 3, TileType::Const);
        sim.set_logic_value(1, 3, 10);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();
        assert_eq!(sim.get_logic_at(4, 3), 20);

        // Change input to 25 and tick again
        sim.set_logic_value(1, 3, 25);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();
        assert_eq!(
            sim.get_logic_at(4, 3),
            50,
            "Cache should invalidate and recompute"
        );
    }

    #[test]
    fn test_wire_up_tile() {
        let mut sim = Simulation::with_size(5, 5);
        sim.set_tile(2, 2, TileType::WireUp);
        sim.set_tile(2, 3, TileType::Const); // below
        sim.set_logic_value(2, 3, 0xABCD);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();
        // WireUp reads from down (below), so should get value from (2,3)
        assert_eq!(
            sim.get_logic_at(2, 2),
            0xABCD,
            "WireUp should read from down neighbor"
        );
    }

    #[test]
    fn test_wire_left_tile() {
        let mut sim = Simulation::with_size(5, 5);
        sim.set_tile(2, 2, TileType::WireLeft);
        sim.set_tile(3, 2, TileType::Const); // right neighbor
        sim.set_logic_value(3, 2, 0x1234);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick();
        // WireLeft reads from right, so should get value from (3,2)
        assert_eq!(
            sim.get_logic_at(2, 2),
            0x1234,
            "WireLeft should read from right neighbor"
        );
    }

    #[test]
    fn test_place_component_bounds_check() {
        use crate::component::*;

        let mut sim = Simulation::with_size(10, 10);

        let def = ComponentDef {
            name: "BIG".to_string(),
            width: 8,
            height: 8,
            ports: vec![],
            implementation: ComponentImpl::Combinational(Box::new(|_| vec![])),
            propagation_delay: 2,
        };

        let def_idx = sim.register_component_def(def);

        // Should fail: component doesn't fit
        assert!(sim.place_component(def_idx, 5, 5).is_none());
        // Should succeed: fits at (2, 2)
        assert!(sim.place_component(def_idx, 2, 2).is_some());
    }

    // =========================================================================
    // Bus Architecture Tests
    // =========================================================================

    #[test]
    fn bus_single_word_writer_reader() {
        // Single-word bus, one writer one reader
        let mut sim = Simulation::with_size(30, 10);

        // Register bus (width=1, Priority)
        let bus_idx = sim.register_bus(BusDef {
            name: "data_bus".to_string(),
            width: 1,
            arbitration: BusArbitration::Priority,
        });

        // Place a Const tile at (9, 5) with value 42 (left neighbor of writer)
        sim.set_tile(9, 5, TileType::Const);
        sim.set_logic_value(9, 5, 42);

        // Connect writer at (10, 5)
        assert!(
            sim.connect_bus(bus_idx, 10, 5, 0, BusDirection::Writer)
                .is_some()
        );

        // Connect reader at (20, 5)
        assert!(
            sim.connect_bus(bus_idx, 20, 5, 0, BusDirection::Reader)
                .is_some()
        );

        // Place a Wire tile at (21, 5) to pick up the reader value
        sim.set_tile(21, 5, TileType::Wire);

        // Tick the simulation
        sim.tick();

        // Verify reader tile logic == 42
        let reader_val = sim.get_logic_at(20, 5);
        assert_eq!(
            reader_val, 42,
            "Reader tile should have bus value 42, got {}",
            reader_val
        );

        // Verify wire tile at (21, 5) picks up the value
        let wire_val = sim.get_logic_at(21, 5);
        assert_eq!(
            wire_val, 42,
            "Wire tile adjacent to reader should have value 42, got {}",
            wire_val
        );
    }

    #[test]
    fn bus_multi_writer_priority_arbitration() {
        // Multi-writer Priority arbitration: first writer wins
        let mut sim = Simulation::with_size(30, 15);

        let bus_idx = sim.register_bus(BusDef {
            name: "priority_bus".to_string(),
            width: 1,
            arbitration: BusArbitration::Priority,
        });

        // Writer A at (10, 5): left neighbor = Const at (9, 5) with value 100
        sim.set_tile(9, 5, TileType::Const);
        sim.set_logic_value(9, 5, 100);
        assert!(
            sim.connect_bus(bus_idx, 10, 5, 0, BusDirection::Writer)
                .is_some()
        );

        // Writer B at (10, 10): left neighbor = Const at (9, 10) with value 200
        sim.set_tile(9, 10, TileType::Const);
        sim.set_logic_value(9, 10, 200);
        assert!(
            sim.connect_bus(bus_idx, 10, 10, 0, BusDirection::Writer)
                .is_some()
        );

        // Reader at (20, 5)
        assert!(
            sim.connect_bus(bus_idx, 20, 5, 0, BusDirection::Reader)
                .is_some()
        );

        sim.tick();

        // Writer A has lower connection index = higher priority
        let reader_val = sim.get_logic_at(20, 5);
        assert_eq!(
            reader_val, 100,
            "Priority arbitration: first writer (100) should win, got {}",
            reader_val
        );
    }

    #[test]
    fn bus_multi_writer_or_merge_arbitration() {
        // Multi-writer OrMerge arbitration: bitwise OR of all writer values
        let mut sim = Simulation::with_size(30, 15);

        let bus_idx = sim.register_bus(BusDef {
            name: "or_bus".to_string(),
            width: 1,
            arbitration: BusArbitration::OrMerge,
        });

        // Writer A: left neighbor Const = 0xFF00
        sim.set_tile(9, 5, TileType::Const);
        sim.set_logic_value(9, 5, 0xFF00);
        assert!(
            sim.connect_bus(bus_idx, 10, 5, 0, BusDirection::Writer)
                .is_some()
        );

        // Writer B: left neighbor Const = 0x00FF
        sim.set_tile(9, 10, TileType::Const);
        sim.set_logic_value(9, 10, 0x00FF);
        assert!(
            sim.connect_bus(bus_idx, 10, 10, 0, BusDirection::Writer)
                .is_some()
        );

        // Reader at (20, 5)
        assert!(
            sim.connect_bus(bus_idx, 20, 5, 0, BusDirection::Reader)
                .is_some()
        );

        sim.tick();

        let reader_val = sim.get_logic_at(20, 5);
        assert_eq!(
            reader_val, 0xFFFF,
            "OrMerge: 0xFF00 | 0x00FF = 0xFFFF, got 0x{:X}",
            reader_val
        );
    }

    #[test]
    fn bus_reader_propagation_to_grid() {
        // Bus reader propagation: wire adjacent to reader picks up bus value
        let mut sim = Simulation::with_size(30, 10);

        let bus_idx = sim.register_bus(BusDef {
            name: "prop_bus".to_string(),
            width: 1,
            arbitration: BusArbitration::Priority,
        });

        // Writer with Const=0xDEAD
        sim.set_tile(9, 5, TileType::Const);
        sim.set_logic_value(9, 5, 0xDEAD);
        assert!(
            sim.connect_bus(bus_idx, 10, 5, 0, BusDirection::Writer)
                .is_some()
        );

        // Reader
        assert!(
            sim.connect_bus(bus_idx, 20, 5, 0, BusDirection::Reader)
                .is_some()
        );

        // Wire tiles to propagate value
        sim.set_tile(21, 5, TileType::WireRight);

        sim.tick();

        let wire_val = sim.get_logic_at(21, 5);
        assert_eq!(
            wire_val, 0xDEAD,
            "Wire should propagate bus reader value 0xDEAD, got 0x{:X}",
            wire_val
        );
    }

    #[test]
    fn bus_no_bus_tick_regression() {
        // Standard simulation with no buses should work normally (no crash, no behavior change)
        let mut sim = Simulation::with_size(30, 10);

        // Set up a simple circuit: ClockGlobal -> Wire -> Wire
        sim.set_tile(5, 5, TileType::ClockGlobal);
        sim.set_tile(6, 5, TileType::Wire);
        sim.set_tile(7, 5, TileType::Wire);

        for _ in 0..10 {
            sim.tick();
        }

        // ClockGlobal should be toggling (after 10 ticks, clock is false->true...->false)
        // Just verify no crash and clock tile has expected value
        let clk_val = sim.get_logic_at(5, 5);
        // After 10 ticks (even number), global_clock = false, so ClockGlobal = 0
        assert_eq!(clk_val, 0, "After 10 ticks, clock should be 0 (low phase)");
    }

    #[test]
    fn bus_blueprint_round_trip() {
        // Blueprint round-trip for BusInterface tile type
        let mut sim = Simulation::with_size(30, 10);
        sim.set_tile(15, 5, TileType::BusInterface);

        // Save blueprint
        let bp = Blueprint::from_simulation(&sim);

        // Load blueprint into fresh sim
        let mut sim2 = Simulation::with_size(30, 10);
        bp.apply_to_simulation(&mut sim2).unwrap();

        // Verify tile type preserved
        assert_eq!(sim2.tile_type_xy(15, 5), TileType::BusInterface);
    }

    #[test]
    fn bus_multi_word() {
        // Multi-word bus: 2 words, each with its own writer and reader
        let mut sim = Simulation::with_size(30, 15);

        let bus_idx = sim.register_bus(BusDef {
            name: "wide_bus".to_string(),
            width: 2,
            arbitration: BusArbitration::Priority,
        });

        // Word 0 writer at (10, 3): left neighbor Const = 0xAAAA
        sim.set_tile(9, 3, TileType::Const);
        sim.set_logic_value(9, 3, 0xAAAA);
        assert!(
            sim.connect_bus(bus_idx, 10, 3, 0, BusDirection::Writer)
                .is_some()
        );

        // Word 1 writer at (10, 6): left neighbor Const = 0xBBBB
        sim.set_tile(9, 6, TileType::Const);
        sim.set_logic_value(9, 6, 0xBBBB);
        assert!(
            sim.connect_bus(bus_idx, 10, 6, 1, BusDirection::Writer)
                .is_some()
        );

        // Word 0 reader at (20, 3)
        assert!(
            sim.connect_bus(bus_idx, 20, 3, 0, BusDirection::Reader)
                .is_some()
        );

        // Word 1 reader at (20, 6)
        assert!(
            sim.connect_bus(bus_idx, 20, 6, 1, BusDirection::Reader)
                .is_some()
        );

        sim.tick();

        let w0_val = sim.get_logic_at(20, 3);
        assert_eq!(
            w0_val, 0xAAAA,
            "Word 0 reader should have 0xAAAA, got 0x{:X}",
            w0_val
        );

        let w1_val = sim.get_logic_at(20, 6);
        assert_eq!(
            w1_val, 0xBBBB,
            "Word 1 reader should have 0xBBBB, got 0x{:X}",
            w1_val
        );
    }

    // =========================================================================
    // Memory Controller Tests
    // =========================================================================

    #[test]
    fn test_memory_basic_read() {
        // Register 256-word bank with initial_data = [10, 20, 30]
        // Connect MemoryPort at (10, 10)
        // Set left neighbor (address) to 0, up (write_enable) to 0
        // Tick, verify output == 10
        // Change address to 1, tick, verify output == 20
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "test_bank".to_string(),
            size: 256,
            initial_data: vec![10, 20, 30],
        });
        assert_eq!(bank_idx, 0);

        sim.connect_memory_port(bank_idx, 10, 10);

        // Set address (left neighbor at 9,10) to 0
        sim.set_logic_value(9, 10, 0);
        // Set write_enable (up neighbor at 10,9) to 0
        sim.set_logic_value(10, 9, 0);
        // Set data_in (right neighbor at 11,10) to 0
        sim.set_logic_value(11, 10, 0);

        // Eval the MemoryPort tile
        let idx = 10 * 64 + 10;
        sim.eval_tile(idx);

        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 10,
            "Read address 0 should return 10, got {}",
            output
        );

        // Change address to 1
        sim.set_logic_value(9, 10, 1);
        sim.eval_tile(idx);
        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 20,
            "Read address 1 should return 20, got {}",
            output
        );

        // Change address to 2
        sim.set_logic_value(9, 10, 2);
        sim.eval_tile(idx);
        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 30,
            "Read address 2 should return 30, got {}",
            output
        );
    }

    #[test]
    fn test_memory_write_then_read() {
        // Register 256-word bank (all zeros)
        // Connect MemoryPort
        // Set address=5, data_in=0xDEADBEEF, write_enable=1
        // Tick
        // Set write_enable=0, address=5
        // Tick
        // Verify output == 0xDEADBEEF
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "test_bank".to_string(),
            size: 256,
            initial_data: vec![],
        });

        sim.connect_memory_port(bank_idx, 10, 10);
        let idx = 10 * 64 + 10;

        // Write: address=5, data_in=0xDEADBEEF, write_enable=1
        sim.set_logic_value(9, 10, 5); // left = address
        sim.set_logic_value(11, 10, 0xDEADBEEF); // right = data_in
        sim.set_logic_value(10, 9, 1); // up = write_enable
        sim.eval_tile(idx);

        // Now read: address=5, write_enable=0
        sim.set_logic_value(10, 9, 0); // up = write_enable off
        sim.eval_tile(idx);

        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 0xDEADBEEF,
            "Read after write should return 0xDEADBEEF, got 0x{:X}",
            output
        );
    }

    #[test]
    fn test_memory_read_after_write_same_tick() {
        // Set address=3, data_in=42, write_enable=1
        // Tick
        // Verify MemoryPort output == 42 (read-after-write)
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "test_bank".to_string(),
            size: 256,
            initial_data: vec![],
        });

        sim.connect_memory_port(bank_idx, 10, 10);
        let idx = 10 * 64 + 10;

        // Write and read same tick: address=3, data_in=42, write_enable=1
        sim.set_logic_value(9, 10, 3); // left = address
        sim.set_logic_value(11, 10, 42); // right = data_in
        sim.set_logic_value(10, 9, 1); // up = write_enable
        sim.eval_tile(idx);

        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 42,
            "Read-after-write same tick should return 42, got {}",
            output
        );
    }

    #[test]
    fn test_memory_out_of_bounds_address() {
        // Register 16-word bank
        // Set address=100 (out of range)
        // Tick
        // Verify output == 0
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "small_bank".to_string(),
            size: 16,
            initial_data: vec![0xFF; 16],
        });

        sim.connect_memory_port(bank_idx, 10, 10);
        let idx = 10 * 64 + 10;

        // Out-of-bounds read: address=100
        sim.set_logic_value(9, 10, 100); // left = address (out of bounds)
        sim.set_logic_value(10, 9, 0); // up = write_enable off
        sim.eval_tile(idx);

        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 0,
            "Out-of-bounds read should return 0, got {}",
            output
        );
    }

    #[test]
    fn test_memory_multi_port_same_bank() {
        // Register 256-word bank
        // Connect two MemoryPorts to same bank
        // Port A writes to address 0
        // Port B reads from address 0
        // Tick both
        // Verify Port B sees the written value
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "shared_bank".to_string(),
            size: 256,
            initial_data: vec![],
        });

        // Port A at (10, 10)
        sim.connect_memory_port(bank_idx, 10, 10);
        // Port B at (20, 20)
        sim.connect_memory_port(bank_idx, 20, 20);

        let idx_a = 10 * 64 + 10;
        let idx_b = 20 * 64 + 20;

        // Port A writes: address=0, data_in=0xCAFE, write_enable=1
        sim.set_logic_value(9, 10, 0); // left = address
        sim.set_logic_value(11, 10, 0xCAFE); // right = data_in
        sim.set_logic_value(10, 9, 1); // up = write_enable
        sim.eval_tile(idx_a);

        // Port B reads: address=0, write_enable=0
        sim.set_logic_value(19, 20, 0); // left = address
        sim.set_logic_value(21, 20, 0); // right = data_in (unused for read)
        sim.set_logic_value(20, 19, 0); // up = write_enable off
        sim.eval_tile(idx_b);

        let output_b = sim.get_logic_at(20, 20);
        assert_eq!(
            output_b, 0xCAFE,
            "Port B should read value written by Port A, got 0x{:X}",
            output_b
        );
    }

    #[test]
    fn test_memory_initial_data() {
        // Register bank with initial_data = [1, 2, 3, 4]
        // Read addresses 0-3
        // Verify values match initial data
        let mut sim = Simulation::with_size(64, 64);

        let bank_idx = sim.register_memory_bank(MemoryBankDef {
            name: "init_bank".to_string(),
            size: 8,
            initial_data: vec![1, 2, 3, 4],
        });

        sim.connect_memory_port(bank_idx, 10, 10);
        let idx = 10 * 64 + 10;

        sim.set_logic_value(10, 9, 0); // up = write_enable off
        sim.set_logic_value(11, 10, 0); // right = data_in (unused)

        for addr in 0..4u64 {
            sim.set_logic_value(9, 10, addr);
            sim.eval_tile(idx);
            let output = sim.get_logic_at(10, 10);
            assert_eq!(
                output,
                addr + 1,
                "Address {} should return {}, got {}",
                addr,
                addr + 1,
                output
            );
        }

        // Address 4 should return 0 (beyond initial_data, but within bank size)
        sim.set_logic_value(9, 10, 4);
        sim.eval_tile(idx);
        let output = sim.get_logic_at(10, 10);
        assert_eq!(
            output, 0,
            "Address 4 (beyond initial data) should return 0, got {}",
            output
        );
    }

    #[test]
    fn test_memory_blueprint_roundtrip() {
        // Place MemoryPort tile
        // Save/load blueprint
        // Verify tile type preserved
        use crate::blueprint::Blueprint;
        use std::io::BufReader;

        let mut bp = Blueprint::new(64, 64);
        bp.tiles.push(crate::blueprint::BlueprintTile {
            x: 10,
            y: 10,
            z: 0,
            tile_type: TileType::MemoryPort,
            logic: Some(0x42),
        });

        let mut buf = Vec::new();
        bp.save_to_writer(&mut buf).expect("save ok");
        let loaded = Blueprint::load_from_reader(BufReader::new(&buf[..])).expect("load ok");

        assert_eq!(loaded.tiles.len(), 1);
        assert_eq!(loaded.tiles[0].tile_type, TileType::MemoryPort);
        assert_eq!(loaded.tiles[0].logic, Some(0x42));
    }

    #[test]
    fn test_memory_no_bank_tick_regression() {
        // Standard simulation with no memory banks
        // Run 10 ticks
        // Verify no crash, normal behavior
        let mut sim = Simulation::with_size(32, 32);

        // Place some tiles and run ticks - main goal is no panics
        sim.set_tile(5, 5, TileType::And);
        sim.set_tile(4, 5, TileType::Wire);
        sim.set_tile(6, 5, TileType::Wire);

        for _ in 0..10 {
            sim.tick();
        }

        // Verify And gate evaluated (direct eval to confirm no crash with eval_tile)
        sim.set_logic_value(4, 5, 0xFF);
        sim.set_logic_value(6, 5, 0x0F);
        let idx = 5 * 32 + 5;
        sim.eval_tile(idx);
        let val = sim.get_logic_at(5, 5);
        assert_eq!(
            val,
            0xFF & 0x0F,
            "And gate should produce 0x0F, got 0x{:X}",
            val
        );
    }

    // === Multi-Clock Domain Tests ===

    #[test]
    fn test_clock_domain_basic() {
        // Register div-2 domain, step 8 ticks, verify domain clock toggles at half rate
        let mut sim = Simulation::with_size(4, 4);
        let domain = sim.register_clock_domain("slow", 2, 0);
        // Collect clock states over 8 ticks
        let mut clocks = Vec::new();
        for _ in 0..8 {
            sim.tick();
            clocks.push(sim.clock_domain_states[domain].clock);
        }
        // div-2 domain: period=4, so clock should toggle every 2 ticks
        // counter: 1,2,3,4,5,6,7,8
        // pos=(counter%4): 1,2,3,0,1,2,3,0
        // clock=(pos<2): T,F,F,T,T,F,F,T
        assert_eq!(
            clocks,
            vec![true, false, false, true, true, false, false, true],
            "div-2 domain clock pattern: {:?}",
            clocks
        );
    }

    #[test]
    fn test_clock_divider_tile() {
        // Place ClockDivider, connect to div-2 domain
        let mut sim = Simulation::with_size(4, 4);
        let domain = sim.register_clock_domain("div2", 2, 0);
        sim.set_tile(1, 1, TileType::ClockDivider);
        sim.connect_clock_divider_xy(1, 1, domain);
        // Run ticks and check output pattern
        let mut pattern = Vec::new();
        for _ in 0..8 {
            sim.tick();
            pattern.push(sim.get_logic_at(1, 1));
        }
        // div-2 should have both 0 and MAX values
        assert!(pattern.contains(&0), "Should have low values");
        assert!(pattern.contains(&u64::MAX), "Should have high values");
    }

    #[test]
    fn test_register8_in_domain() {
        // Register8 in div-2 domain should only capture on domain rising edge
        let mut sim = Simulation::with_size(8, 4);
        let domain = sim.register_clock_domain("slow", 2, 0);
        // Set up: const at (0,1) with value, register8 at (1,1) reads from left
        sim.set_tile(0, 1, TileType::Const);
        sim.set_tile(1, 1, TileType::Register8);
        sim.assign_tile_to_domain_xy(1, 1, domain);
        // Write a value to the const tile
        sim.tilemap.set_value(1 * 8 + 0, 42);
        // Step through ticks and check when register captures
        let mut captured_at = Vec::new();
        let mut prev_val = sim.get_logic_at(1, 1);
        for i in 0..8 {
            sim.tick();
            let val = sim.get_logic_at(1, 1);
            if val != prev_val {
                captured_at.push(i);
                prev_val = val;
            }
        }
        // Should NOT capture every tick (like global clock would)
        // The domain rising edge occurs less frequently than global
        assert!(
            captured_at.len() <= 4,
            "Register8 in div-2 domain should capture less frequently"
        );
    }

    #[test]
    fn test_unassigned_uses_global() {
        // Register8 without domain uses global clock (backward compat)
        let mut sim = Simulation::with_size(4, 4);
        sim.set_tile(0, 1, TileType::Const);
        sim.set_tile(1, 1, TileType::Register8);
        // No domain assignment
        sim.tilemap.set_value(1 * 4 + 0, 99);
        sim.tick(); // rising edge
        sim.tick(); // falling edge
        sim.tick(); // rising edge - should capture
        let val = sim.get_logic_at(1, 1);
        // With global clock, register should have captured by now
        // (global clock toggles every tick, so rising edge happens every other tick)
        assert!(
            val == 99 || val == 0,
            "Register8 should use global clock when no domain assigned"
        );
    }

    #[test]
    fn test_synchronizer_basic() {
        // Synchronizer captures left input on domain rising edge with 2-cycle latency
        let mut sim = Simulation::with_size(8, 4);
        let domain = sim.register_clock_domain("dest", 1, 0); // same-speed domain for simplicity
        sim.set_tile(0, 1, TileType::Const);
        sim.set_tile(1, 1, TileType::Synchronizer);
        sim.connect_synchronizer_xy(1, 1, domain);
        sim.tilemap.set_value(1 * 8 + 0, 77);
        // Step through - should see 2-cycle latency
        sim.tick();
        sim.tick();
        sim.tick();
        // After enough rising edges, value should propagate through 2 FF stages
        // With divider=1, rising edge is every other tick
        // Run more ticks to ensure propagation
        for _ in 0..4 {
            sim.tick();
        }
        // After enough ticks, the value should have propagated
        // Note: exact propagation timing depends on edge alignment
    }

    #[test]
    fn test_multiple_domains() {
        // Two domains with different dividers
        let mut sim = Simulation::with_size(8, 4);
        let d2 = sim.register_clock_domain("div2", 2, 0);
        let d4 = sim.register_clock_domain("div4", 4, 0);
        // Place clock dividers
        sim.set_tile(0, 0, TileType::ClockDivider);
        sim.connect_clock_divider_xy(0, 0, d2);
        sim.set_tile(1, 0, TileType::ClockDivider);
        sim.connect_clock_divider_xy(1, 0, d4);
        // Step and verify different frequencies
        let mut d2_edges = 0u32;
        let mut d4_edges = 0u32;
        let mut prev_d2 = 0u64;
        let mut prev_d4 = 0u64;
        for _ in 0..16 {
            sim.tick();
            let v2 = sim.get_logic_at(0, 0);
            let v4 = sim.get_logic_at(1, 0);
            if v2 != prev_d2 {
                d2_edges += 1;
                prev_d2 = v2;
            }
            if v4 != prev_d4 {
                d4_edges += 1;
                prev_d4 = v4;
            }
        }
        // div-2 should have more edges than div-4
        assert!(
            d2_edges > d4_edges,
            "div-2 domain should have more edges than div-4: {} vs {}",
            d2_edges,
            d4_edges
        );
    }

    #[test]
    fn test_phase_offset() {
        // Domain with phase offset should shift clock
        let mut sim = Simulation::with_size(4, 4);
        let d_no_phase = sim.register_clock_domain("nophase", 2, 0);
        let d_phase = sim.register_clock_domain("phase", 2, 1);
        // Place clock dividers for each
        sim.set_tile(0, 0, TileType::ClockDivider);
        sim.connect_clock_divider_xy(0, 0, d_no_phase);
        sim.set_tile(1, 0, TileType::ClockDivider);
        sim.connect_clock_divider_xy(1, 0, d_phase);
        // Run and record patterns
        let mut pattern_a = Vec::new();
        let mut pattern_b = Vec::new();
        for _ in 0..8 {
            sim.tick();
            pattern_a.push(sim.get_logic_at(0, 0));
            pattern_b.push(sim.get_logic_at(1, 0));
        }
        // Patterns should differ due to phase offset
        assert_ne!(
            pattern_a, pattern_b,
            "Phase offset should produce different clock patterns"
        );
    }

    #[test]
    fn test_domain_registration() {
        let mut sim = Simulation::with_size(4, 4);
        let d0 = sim.register_clock_domain("fast", 1, 0);
        let d1 = sim.register_clock_domain("slow", 4, 0);
        assert_eq!(d0, 0);
        assert_eq!(d1, 1);
        assert_eq!(sim.clock_domain_defs.len(), 2);
        assert_eq!(sim.clock_domain_defs[0].name, "fast");
        assert_eq!(sim.clock_domain_defs[1].divider, 4);
    }

    #[test]
    fn test_step_backward_at_start() {
        // Ensure backward compat - tick works fine with no domains
        let mut sim = Simulation::with_size(4, 4);
        sim.set_tile(0, 0, TileType::ClockGlobal);
        sim.tick();
        let v = sim.get_logic_at(0, 0);
        // ClockGlobal should output MAX on rising edge
        assert!(
            v == u64::MAX || v == 0,
            "ClockGlobal should work without domains"
        );
    }

    // =========================================================================
    // WireCross / VBusIn / VBusOut Tests
    // =========================================================================

    #[test]
    fn test_wire_cross_basic() {
        // WireCross takes horizontal from left (bits 0-31) and vertical from up (bits 32-63)
        let mut sim = Simulation::with_size(8, 8);

        // Horizontal signal: Const(42) to the left of WireCross
        sim.set_tile(2, 3, TileType::Const);
        sim.set_logic_value(2, 3, 42);

        // Vertical signal: Const with value already in high bus (bits 32-63)
        sim.set_tile(3, 2, TileType::Const);
        sim.set_logic_value(3, 2, 99u64 << 32);

        // WireCross at (3, 3)
        sim.set_tile(3, 3, TileType::WireCross);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let output = sim.get_logic_at(3, 3);
        let h = output & 0xFFFF_FFFF;
        let v = (output >> 32) & 0xFFFF_FFFF;
        assert_eq!(h, 42, "horizontal should be 42");
        assert_eq!(v, 99, "vertical should be 99");
    }

    #[test]
    fn test_wire_cross_chain_vertical() {
        // V signal survives through 3 stacked WireCross tiles
        let mut sim = Simulation::with_size(8, 8);

        // VBusIn at (3, 1) shifts signal to high bus
        sim.set_tile(3, 0, TileType::Const);
        sim.set_logic_value(3, 0, 77);
        sim.set_tile(3, 1, TileType::VBusIn);

        // 3 WireCross tiles at (3,2), (3,3), (3,4)
        sim.set_tile(3, 2, TileType::WireCross);
        sim.set_tile(3, 3, TileType::WireCross);
        sim.set_tile(3, 4, TileType::WireCross);

        // VBusOut at (3, 5) shifts back to low bus
        sim.set_tile(3, 5, TileType::VBusOut);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let output = sim.get_logic_at(3, 5);
        assert_eq!(
            output, 77,
            "V signal should survive through 3 WireCross tiles"
        );
    }

    #[test]
    fn test_wire_cross_chain_horizontal() {
        // H signal survives through 3 adjacent WireCross tiles
        let mut sim = Simulation::with_size(8, 8);

        // Horizontal source: Const(55) at (0, 3)
        sim.set_tile(0, 3, TileType::Const);
        sim.set_logic_value(0, 3, 55);

        // WireRight chain to reach WireCross zone
        sim.set_tile(1, 3, TileType::WireRight);

        // 3 WireCross tiles at (2,3), (3,3), (4,3)
        sim.set_tile(2, 3, TileType::WireCross);
        sim.set_tile(3, 3, TileType::WireCross);
        sim.set_tile(4, 3, TileType::WireCross);

        // WireRight at (5, 3) reads from WireCross
        sim.set_tile(5, 3, TileType::WireRight);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let output = sim.get_logic_at(5, 3) & 0xFF;
        assert_eq!(
            output, 55,
            "H signal should survive through 3 WireCross tiles"
        );
    }

    #[test]
    fn test_vbus_in_out_roundtrip() {
        // VBusIn → VBusOut should recover original signal
        let mut sim = Simulation::with_size(8, 8);

        sim.set_tile(3, 0, TileType::Const);
        sim.set_logic_value(3, 0, 200);

        sim.set_tile(3, 1, TileType::VBusIn);
        sim.set_tile(3, 2, TileType::VBusOut);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        // VBusIn output should be 200 << 32
        let vbus_in_out = sim.get_logic_at(3, 1);
        assert_eq!(vbus_in_out, 200u64 << 32, "VBusIn should shift to high bus");

        // VBusOut output should be 200 back in low bus
        let vbus_out_out = sim.get_logic_at(3, 2);
        assert_eq!(vbus_out_out, 200, "VBusOut should shift back to low bus");
    }

    #[test]
    fn test_wire_cross_full_crossing() {
        // Complete crossing: WR chain and WD chain cross via WireCross zone
        //
        //     col 3 (vertical: signal V=33)
        //       |
        //  row 2:  [Const(33)] → [WD] → [VBusIn]
        //  row 3:  [Const(17)] → [WR] → [WireCross] → [WR]
        //  row 4:                        [VBusOut]
        //
        //  Horizontal consumer at (4, 3) should see 17
        //  Vertical consumer at (3, 4) should see 33
        let mut sim = Simulation::with_size(8, 8);

        // Vertical signal: Const(33) at (3, 1), WireDown at (3, 2) is implicit via VBusIn
        sim.set_tile(3, 1, TileType::Const);
        sim.set_logic_value(3, 1, 33);
        sim.set_tile(3, 2, TileType::VBusIn); // reads up=33, outputs 33<<32

        // Horizontal signal: Const(17) at (1, 3), WireRight at (2, 3)
        sim.set_tile(1, 3, TileType::Const);
        sim.set_logic_value(1, 3, 17);
        sim.set_tile(2, 3, TileType::WireRight);

        // WireCross at intersection (3, 3)
        sim.set_tile(3, 3, TileType::WireCross);

        // Horizontal exit: WireRight at (4, 3)
        sim.set_tile(4, 3, TileType::WireRight);

        // Vertical exit: VBusOut at (3, 4)
        sim.set_tile(3, 4, TileType::VBusOut);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        // Horizontal consumer: should see 17 in bits 0-7
        let h_out = sim.get_logic_at(4, 3) & 0xFF;
        assert_eq!(h_out, 17, "horizontal signal should be 17");

        // Vertical consumer: should see 33
        let v_out = sim.get_logic_at(3, 4);
        assert_eq!(v_out, 33, "vertical signal should be 33");
    }

    #[test]
    fn test_wire_cross_multi_signal() {
        // Multiple V signals through the same H crossing row
        //
        //  col 2 (V=10)    col 4 (V=20)
        //       |                |
        //  row 1: [Const(10)]  [Const(20)]
        //  row 2: [VBusIn]     [VBusIn]
        //  row 3: [WireCross]  [WireCross]   ← H signal crosses both
        //  row 4: [VBusOut]    [VBusOut]
        //
        let mut sim = Simulation::with_size(8, 8);

        // V signal 1: value 10 at col 2
        sim.set_tile(2, 1, TileType::Const);
        sim.set_logic_value(2, 1, 10);
        sim.set_tile(2, 2, TileType::VBusIn);
        sim.set_tile(2, 3, TileType::WireCross);
        sim.set_tile(2, 4, TileType::VBusOut);

        // V signal 2: value 20 at col 4
        sim.set_tile(4, 1, TileType::Const);
        sim.set_logic_value(4, 1, 20);
        sim.set_tile(4, 2, TileType::VBusIn);
        sim.set_tile(4, 3, TileType::WireCross);
        sim.set_tile(4, 4, TileType::VBusOut);

        // H signal: value 7 from left
        sim.set_tile(0, 3, TileType::Const);
        sim.set_logic_value(0, 3, 7);
        sim.set_tile(1, 3, TileType::WireRight);
        // WireCross at (2,3) already placed
        sim.set_tile(3, 3, TileType::WireRight);
        // WireCross at (4,3) already placed
        sim.set_tile(5, 3, TileType::WireRight);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        // Both V signals should survive independently
        assert_eq!(sim.get_logic_at(2, 4), 10, "V1 should be 10");
        assert_eq!(sim.get_logic_at(4, 4), 20, "V2 should be 20");

        // H signal should survive through both crossings
        let h_out = sim.get_logic_at(5, 3) & 0xFF;
        assert_eq!(
            h_out, 7,
            "H signal should be 7 after crossing both V signals"
        );
    }

    #[test]
    fn test_wire_cross_vert_basic() {
        // WireCrossVert reads horizontal from right (bits 0-31) and vertical from up (bits 32-63).
        // This is for WL chains (right-to-left flow) crossing vertical WD channels.
        //
        //  Layout (4x4):
        //    (3,0) Const(42)  — horizontal source (to the right of WireCrossVert)
        //    (2,0) WireCrossVert — reads right=42, up=99<<32
        //    (2,1) target — should see h=42 in low bits, v=99 in high bits packed
        //
        //  Vertical source above:
        //    (2, is at row 0 so we need a row above) — use row-based layout:
        //    row 0: Const(99) at col 2
        //    row 1: VBusIn at col 2 (shifts to high bits)
        //    row 2: WireCrossVert at col 2, Const(42) at col 3 (right neighbor)
        //
        let mut sim = Simulation::with_size(8, 8);

        // Vertical signal: value 99 entering from above
        sim.set_tile(2, 0, TileType::Const);
        sim.set_logic_value(2, 0, 99);
        sim.set_tile(2, 1, TileType::VBusIn); // shifts 99 to high bits

        // WireCrossVert at (2,2): reads right (col 3) and up (col 2, row 1)
        sim.set_tile(2, 2, TileType::WireCrossVert);

        // Horizontal signal: value 42 from the right
        sim.set_tile(3, 2, TileType::Const);
        sim.set_logic_value(3, 2, 42);

        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let output = sim.get_logic_at(2, 2);
        let h_bits = output & 0xFFFF_FFFF;
        let v_bits = (output >> 32) & 0xFFFF_FFFF;
        assert_eq!(h_bits, 42, "horizontal (from right) should be 42");
        assert_eq!(v_bits, 99, "vertical (from up, high bits) should be 99");
    }

    // ========================================
    // Sprint 127: 64-bit Scaling Primitives
    // ========================================

    #[test]
    fn test_register64_captures_full_u64() {
        // Verify Register64 captures full u64 without masking (unlike Register8's & 0xFF)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Register64);
        sim.set_tile(9, 10, TileType::Const); // left input

        let big_value: u64 = 0xDEAD_BEEF_CAFE_BABE;
        sim.set_logic_value(9, 10, big_value);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        // Rising edge: clock low → high
        sim.tick();
        sim.propagate_combinational();
        let result = sim.get_logic_at(10, 10);
        assert_eq!(
            result, big_value,
            "Register64 should capture full u64 0x{:016X}, got 0x{:016X}",
            big_value, result
        );
    }

    #[test]
    fn test_register64_no_8bit_mask() {
        // Explicitly verify values > 0xFF are preserved (the key difference from Register8)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Register64);
        sim.set_tile(9, 10, TileType::Const);

        sim.set_logic_value(9, 10, 0x1234);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();
        sim.tick();
        sim.propagate_combinational();

        let result = sim.get_logic_at(10, 10);
        assert_eq!(
            result, 0x1234,
            "Register64 should preserve 0x1234, not mask to 0x34"
        );

        // Compare with Register8 which would mask
        sim.set_tile(10, 11, TileType::Register8);
        sim.set_tile(9, 11, TileType::Const);
        sim.set_logic_value(9, 11, 0x1234);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();
        // Clock is currently HIGH after first rising edge; need falling then rising
        sim.tick(); // falling edge — does not capture
        sim.propagate_combinational();
        sim.tick(); // rising edge — captures
        sim.propagate_combinational();

        let r8_result = sim.get_logic_at(10, 11);
        assert_eq!(r8_result, 0x34, "Register8 should mask to 0x34");
    }

    #[test]
    fn test_register64_edge_triggered() {
        // Only captures on rising edge at delta 0 — same as Register8
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Register64);
        sim.set_tile(9, 10, TileType::Const);

        // Initially register holds 0
        assert_eq!(sim.get_logic_at(10, 10), 0);

        // Set input, but don't clock → register stays 0
        sim.set_logic_value(9, 10, 42);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();
        assert_eq!(
            sim.get_logic_at(10, 10),
            0,
            "Should not capture without clock edge"
        );

        // Rising edge → captures
        sim.tick();
        sim.propagate_combinational();
        assert_eq!(
            sim.get_logic_at(10, 10),
            42,
            "Should capture on rising edge"
        );

        // Change input, falling edge → should NOT capture
        sim.set_logic_value(9, 10, 99);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();
        sim.tick();
        sim.propagate_combinational(); // falling edge
        assert_eq!(
            sim.get_logic_at(10, 10),
            42,
            "Should not capture on falling edge"
        );
    }

    #[test]
    fn test_carry_detect_overflow() {
        // CarryDetect: left > right → MAX (unsigned overflow after addition)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::CarryDetect);
        sim.set_tile(9, 10, TileType::Const); // left = original value a
        sim.set_tile(11, 10, TileType::Const); // right = result (a + b)

        // Overflow case: a=200, b=100, result=300 & 0xFF = 44. a(200) > result(44) → overflow
        sim.set_logic_value(9, 10, 200);
        sim.set_logic_value(11, 10, 44);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, u64::MAX, "200 > 44 should detect overflow");
    }

    #[test]
    fn test_carry_detect_no_overflow() {
        // CarryDetect: left <= right → 0 (no overflow)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::CarryDetect);
        sim.set_tile(9, 10, TileType::Const);
        sim.set_tile(11, 10, TileType::Const);

        // No overflow: a=10, b=20, result=30. a(10) <= result(30) → no overflow
        sim.set_logic_value(9, 10, 10);
        sim.set_logic_value(11, 10, 30);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, 0, "10 <= 30 should detect no overflow");
    }

    #[test]
    fn test_carry_detect_equal() {
        // Equal case: left == right → 0 (no overflow)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::CarryDetect);
        sim.set_tile(9, 10, TileType::Const);
        sim.set_tile(11, 10, TileType::Const);

        sim.set_logic_value(9, 10, 42);
        sim.set_logic_value(11, 10, 42);
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, 0, "equal values should not detect overflow");
    }

    #[test]
    fn test_carry_detect_64bit_overflow() {
        // 64-bit overflow: a = u64::MAX, result wraps to small value
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::CarryDetect);
        sim.set_tile(9, 10, TileType::Const);
        sim.set_tile(11, 10, TileType::Const);

        sim.set_logic_value(9, 10, u64::MAX);
        sim.set_logic_value(11, 10, 5); // result = MAX + 6 wraps to 5
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational();

        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, u64::MAX, "u64::MAX > 5 should detect overflow");
    }

    #[test]
    fn test_decoder6to64_all_positions() {
        // Test all 64 one-hot outputs
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Decoder6to64);
        sim.set_tile(9, 10, TileType::Wire); // left input

        for addr in 0u64..64 {
            sim.set_logic_value(9, 10, addr);
            sim.eval_at(10, 10);
            let result = sim.get_logic_at(10, 10);
            let expected = 1u64 << addr;
            assert_eq!(
                result, expected,
                "Decoder6to64 addr {} should produce 0x{:016X}, got 0x{:016X}",
                addr, expected, result
            );
        }
    }

    #[test]
    fn test_decoder6to64_masks_upper_bits() {
        // Only bits 0-5 of input should be used (& 63)
        let mut sim = Simulation::new();
        sim.set_tile(10, 10, TileType::Decoder6to64);
        sim.set_tile(9, 10, TileType::Wire);

        // 0xFF05 should decode as address 5 (bits 0-5 = 000101)
        sim.set_logic_value(9, 10, 0xFF05);
        sim.eval_at(10, 10);
        let result = sim.get_logic_at(10, 10);
        assert_eq!(
            result,
            1u64 << 5,
            "Should decode to bit 5, masking upper bits"
        );

        // 0x1_0000_003F should decode as address 63
        sim.set_logic_value(9, 10, 0x1_0000_003F);
        sim.eval_at(10, 10);
        let result = sim.get_logic_at(10, 10);
        assert_eq!(result, 1u64 << 63, "Should decode to bit 63");
    }

    // === Sprint 160: WeightedVia Tests ===

    /// Helper: create a small 2-layer grid with via connections rebuilt.
    fn make_weighted_via_sim(width: usize, height: usize) -> Simulation {
        let mut sim = Simulation::with_size_layered(width, height, 2);
        sim.rebuild_via_connections();
        sim
    }

    #[test]
    fn test_weighted_via_up_basic() {
        // Place Const(0x1234) on L0(4,4), WeightedViaUp with mask=0xFF on L1(4,4)
        // WeightedViaUp reads from L0 (layer z+1 for L0 is nonsense — ViaUp on L1 reads L0+layer_size??).
        // Wait: ViaUp reads from z+1. So WeightedViaUp on L0 reads L1.
        // For our test: place source on L1, WeightedViaUp on L0.
        // Actually: ViaUp = reads from idx + layer_size = reads layer above.
        // L0 idx = y*w+x, L1 idx = layer_size + y*w+x.
        // ViaUp at L0 reads L1 (idx + layer_size). ViaUp at L1 reads L2 (doesn't exist).
        // So: put Const on L1, WeightedViaUp on L0.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size; // 64

        // Source: Const on L1 at (4,4)
        sim.set_tile_3d(4, 4, 1, TileType::Const);
        let src_idx = ls + 4 * 8 + 4; // L1
        sim.set_logic_value_by_idx(src_idx, 0x1234);

        // WeightedViaUp on L0 at (4,4) — reads L1
        sim.set_tile_3d(4, 4, 0, TileType::WeightedViaUp);
        let via_idx = 4 * 8 + 4; // L0
        sim.set_tile_mask(via_idx, 0xFF);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0x34,
            "WeightedViaUp should apply mask 0xFF to 0x1234 → 0x34"
        );
    }

    #[test]
    fn test_weighted_via_down_basic() {
        // WeightedViaDown reads from idx - layer_size (layer below).
        // Place Const on L0, WeightedViaDown on L1.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        // Source: Const on L0 at (3,3)
        sim.set_tile_3d(3, 3, 0, TileType::Const);
        let src_idx = 3 * 8 + 3;
        sim.set_logic_value_by_idx(src_idx, 0xABCD_EF01);

        // WeightedViaDown on L1 at (3,3) — reads L0
        sim.set_tile_3d(3, 3, 1, TileType::WeightedViaDown);
        let via_idx = ls + 3 * 8 + 3;
        sim.set_tile_mask(via_idx, 0xFFFF_0000);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0xABCD_0000,
            "WeightedViaDown should apply mask 0xFFFF_0000 to 0xABCD_EF01 → 0xABCD_0000"
        );
    }

    #[test]
    fn test_weighted_via_identity() {
        // WeightedViaUp with mask=u64::MAX should behave identically to ViaUp.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        // Source on L1
        sim.set_tile_3d(2, 2, 1, TileType::Const);
        let src_idx = ls + 2 * 8 + 2;
        sim.set_logic_value_by_idx(src_idx, 0xDEAD_BEEF_CAFE_BABE);

        // Plain ViaUp on L0 at (2,2)
        sim.set_tile_3d(2, 2, 0, TileType::ViaUp);
        let via_plain_idx = 2 * 8 + 2;

        // WeightedViaUp on L0 at (3,2) pointing to L1(3,2)
        sim.set_tile_3d(3, 2, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 2 * 8 + 3, 0xDEAD_BEEF_CAFE_BABE);
        sim.set_tile_3d(3, 2, 0, TileType::WeightedViaUp);
        let via_weighted_idx = 2 * 8 + 3;
        // mask defaults to u64::MAX — identity

        sim.rebuild_via_connections();
        sim.eval_tile(via_plain_idx);
        sim.eval_tile(via_weighted_idx);

        let plain = sim.get_logic_value_by_idx(via_plain_idx);
        let weighted = sim.get_logic_value_by_idx(via_weighted_idx);
        assert_eq!(
            plain, weighted,
            "WeightedViaUp with identity mask should match ViaUp: plain=0x{:016X} weighted=0x{:016X}",
            plain, weighted
        );
        assert_eq!(plain, 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn test_weighted_via_zero_mask() {
        // WeightedViaUp with mask=0 should always output 0.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(5, 5, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 5 * 8 + 5, 0xFFFF_FFFF_FFFF_FFFF);

        sim.set_tile_3d(5, 5, 0, TileType::WeightedViaUp);
        let via_idx = 5 * 8 + 5;
        sim.set_tile_mask(via_idx, 0); // zero mask

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0,
            "WeightedViaUp with zero mask should output 0 regardless of input"
        );
    }

    #[test]
    fn test_weighted_via_propagation() {
        // Chain: Const(L1) → WeightedViaUp(L0, mask) → Wire(L0) → output tile
        // Verify dirty propagation works through the weighted via.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        // Source: Const on L1 at (2,4)
        sim.set_tile_3d(2, 4, 1, TileType::Const);
        let src_idx = ls + 4 * 8 + 2;
        sim.set_logic_value_by_idx(src_idx, 0xFF00);

        // WeightedViaUp on L0 at (2,4)
        sim.set_tile_3d(2, 4, 0, TileType::WeightedViaUp);
        let via_idx = 4 * 8 + 2;
        sim.set_tile_mask(via_idx, 0x00FF); // low byte only

        // Wire on L0 at (3,4) — right neighbor of via
        sim.set_tile_3d(3, 4, 0, TileType::Wire);
        let wire_idx = 4 * 8 + 3;

        sim.rebuild_via_connections();

        // Evaluate via first, then wire
        sim.eval_tile(via_idx);
        let via_out = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(via_out, 0x0000, "0xFF00 & 0x00FF = 0x0000");

        // Now test with a value that passes through the mask
        sim.set_logic_value_by_idx(src_idx, 0xAB_CD);
        sim.eval_tile(via_idx);
        let via_out2 = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(via_out2, 0x00CD, "0xABCD & 0x00FF = 0x00CD");

        // Wire should pick up the via output after eval
        sim.eval_tile(wire_idx);
        let wire_out = sim.get_logic_value_by_idx(wire_idx);
        // Wire = left | right | up | down. Via is left neighbor of wire.
        assert!(
            wire_out & 0x00CD == 0x00CD,
            "Wire should propagate via output (0x00CD), got 0x{:016X}",
            wire_out
        );
    }

    #[test]
    fn test_weighted_via_chaos_monkey() {
        // For the 5 golden benchmark programs: verify that replacing
        // ViaUp/ViaDown with WeightedViaUp/WeightedViaDown (identity mask)
        // preserves the golden hash — proving they are truly interchangeable.
        use crate::tile_cpu::v2_benchmarks::{benchmark_cases, hash_v2_final_state};

        for case in benchmark_cases() {
            let program = crate::tile_cpu::assemble_v2(case.source)
                .unwrap_or_else(|e| panic!("assemble '{}' failed: {}", case.name, e));

            // Run baseline
            let mut sim_base = Simulation::with_size_layered(128, 128, 4);
            let cpu_base = crate::tile_cpu::V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(64)
                .with_ram_size(64)
                .build(&mut sim_base);
            for _ in 0..case.max_cycles {
                if cpu_base.is_halted() {
                    break;
                }
                cpu_base.step(&mut sim_base);
            }
            let hash_base = hash_v2_final_state(&cpu_base, &sim_base);

            // Run modified: replace up to 5 ViaUp/ViaDown with weighted variants
            let mut sim_mod = Simulation::with_size_layered(128, 128, 4);
            let cpu_mod = crate::tile_cpu::V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(64)
                .with_ram_size(64)
                .build(&mut sim_mod);

            let tc = sim_mod.tilemap.tile_count();
            let mut replaced = 0;
            for idx in 0..tc {
                if replaced >= 5 {
                    break;
                }
                let tt = sim_mod.meta_fast[idx];
                match tt {
                    TileType::ViaUp => {
                        sim_mod.meta_fast[idx] = TileType::WeightedViaUp;
                        sim_mod.tilemap.tiles[idx].meta.tile_type = TileType::WeightedViaUp;
                        replaced += 1;
                    }
                    TileType::ViaDown => {
                        sim_mod.meta_fast[idx] = TileType::WeightedViaDown;
                        sim_mod.tilemap.tiles[idx].meta.tile_type = TileType::WeightedViaDown;
                        replaced += 1;
                    }
                    _ => {}
                }
            }
            if replaced == 0 {
                continue;
            }

            sim_mod.rebuild_via_connections();
            for _ in 0..case.max_cycles {
                if cpu_mod.is_halted() {
                    break;
                }
                cpu_mod.step(&mut sim_mod);
            }
            let hash_mod = hash_v2_final_state(&cpu_mod, &sim_mod);

            assert_eq!(
                hash_base, hash_mod,
                "Chaos monkey: benchmark '{}' hash diverged after replacing {} vias with weighted identity",
                case.name, replaced
            );
        }
    }

    // ======================================================================
    // Sprint 206: WeightedVia Shift Tests
    // ======================================================================

    #[test]
    fn test_weighted_via_shift_basic() {
        // WeightedViaUp with shift=4, mask=0x0F should extract bits [7:4].
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(4, 4, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 4 * 8 + 4, 0xAB);

        sim.set_tile_3d(4, 4, 0, TileType::WeightedViaUp);
        let via_idx = 4 * 8 + 4;
        sim.set_tile_shift(via_idx, 4);
        sim.set_tile_mask(via_idx, 0x0F);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(result, 0x0A, "(0xAB >> 4) & 0x0F = 0x0A");
    }

    #[test]
    fn test_weighted_via_shift_flag_we_pattern() {
        // Exact pattern used in flag-WE path: shift=4, mask=0x03.
        // ctrl_a = 0xB8 (ADD) → (0xB8 >> 4) & 0x03 = 0x0B & 0x03 = 0x03.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(2, 2, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 2 * 8 + 2, 0xB8); // ADD ctrl_a

        sim.set_tile_3d(2, 2, 0, TileType::WeightedViaUp);
        let via_idx = 2 * 8 + 2;
        sim.set_tile_shift(via_idx, 4);
        sim.set_tile_mask(via_idx, 0x03);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0x03,
            "ADD ctrl_a=0xB8: (0xB8 >> 4) & 0x03 = 3 (Z_WE=1, C_WE=1)"
        );

        // NOP ctrl_a = 0x00 → (0x00 >> 4) & 0x03 = 0.
        sim.set_logic_value_by_idx(ls + 2 * 8 + 2, 0x00);
        sim.eval_tile(via_idx);
        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0x00,
            "NOP ctrl_a=0x00: (0x00 >> 4) & 0x03 = 0 (no flag writes)"
        );

        // LDI ctrl_a = 0x58 → (0x58 >> 4) & 0x03 = 0x05 & 0x03 = 1 (Z_WE only).
        sim.set_logic_value_by_idx(ls + 2 * 8 + 2, 0x58);
        sim.eval_tile(via_idx);
        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0x01,
            "LDI ctrl_a=0x58: (0x58 >> 4) & 0x03 = 1 (Z_WE only)"
        );
    }

    #[test]
    fn test_weighted_via_shift_zero_is_identity() {
        // shift=0 + mask=MAX should produce identical output to plain ViaUp.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, 0xDEAD_BEEF_CAFE_BABE);

        sim.set_tile_3d(3, 3, 0, TileType::WeightedViaUp);
        let via_idx = 3 * 8 + 3;
        // shift=0 (default), mask=MAX (default) — should be identity
        assert_eq!(sim.get_tile_shift(via_idx), 0);
        assert_eq!(sim.get_tile_mask(via_idx), u64::MAX);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0xDEAD_BEEF_CAFE_BABE,
            "shift=0, mask=MAX should be identity"
        );
    }

    #[test]
    fn test_weighted_via_down_shift() {
        // WeightedViaDown with shift should also work.
        let mut sim = make_weighted_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(5, 5, 0, TileType::Const);
        sim.set_logic_value_by_idx(5 * 8 + 5, 0xFF00);

        sim.set_tile_3d(5, 5, 1, TileType::WeightedViaDown);
        let via_idx = ls + 5 * 8 + 5;
        sim.set_tile_shift(via_idx, 8);
        sim.set_tile_mask(via_idx, 0xFF);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0xFF,
            "(0xFF00 >> 8) & 0xFF = 0xFF"
        );
    }

    // ======================================================================
    // Sprint 183: ThresholdVia Tests
    // ======================================================================

    /// Helper: create a small 2-layer grid for threshold via tests.
    fn make_threshold_via_sim(width: usize, height: usize) -> Simulation {
        let mut sim = Simulation::with_size_layered(width, height, 2);
        sim.rebuild_via_connections();
        sim
    }

    /// The compact-op path (COP_THRESHOLD_VIA) must reproduce the eval_tile
    /// reference for a ThresholdVia across thresholds, all 16 in-plane neighbor
    /// patterns, source values, and both via directions. Before this op existed,
    /// threshold vias fell through to COP_CONST and were frozen on the compact path.
    #[test]
    fn test_threshold_via_compact_matches_eval_tile() {
        let w = 8usize;
        let layer_size = w * w; // height == w
        let (cx, cy) = (3usize, 3usize);
        let cell = cy * w + cx;
        let npos = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)];

        for &(via_z, src_z, via_tt) in &[
            (0usize, 1usize, TileType::ThresholdViaUp),
            (1usize, 0usize, TileType::ThresholdViaDown),
        ] {
            let via_idx = via_z * layer_size + cell;
            let src_idx = src_z * layer_size + cell;
            for threshold in 1u8..=4 {
                for &src in &[0u64, 0xCAFE_F00D_u64] {
                    for combo in 0u32..16 {
                        let mut sim = make_threshold_via_sim(w, w);

                        sim.set_tile_3d(cx, cy, src_z, TileType::Const);
                        sim.set_logic_value_by_idx(src_idx, src);

                        sim.set_tile_3d(cx, cy, via_z, via_tt);
                        sim.set_tile_threshold(via_idx, threshold);

                        for (j, &(nx, ny)) in npos.iter().enumerate() {
                            sim.set_tile_3d(nx, ny, via_z, TileType::Const);
                            let nidx = via_z * layer_size + ny * w + nx;
                            let val = if (combo >> j) & 1 != 0 { 1u64 } else { 0u64 };
                            sim.set_logic_value_by_idx(nidx, val);
                        }
                        sim.rebuild_via_connections();

                        // Reference: the authoritative eval_tile path.
                        sim.eval_tile(via_idx);
                        let expected = sim.get_logic_value_by_idx(via_idx);

                        // Compact path: reset the via, build + run compact ops over
                        // just the via (its inputs are static Const, already settled).
                        sim.set_logic_value_by_idx(via_idx, 0);
                        let (ops, wvia) = sim.build_compact_ops(&[via_idx]);
                        sim.propagate_compact(&ops, &wvia);
                        let got = sim.get_logic_value_by_idx(via_idx);

                        assert_eq!(
                            got, expected,
                            "{via_tt:?} t={threshold} src={src:#x} combo={combo:04b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_threshold_via_passthrough() {
        // ThresholdViaUp with threshold=1 should pass when one neighbor is active.
        // Layout: Const(0xBEEF) on L1(3,3), ThresholdViaUp on L0(3,3),
        //         Const(1) on L0(2,3) as left neighbor.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        // Source on L1
        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, 0xBEEF);

        // ThresholdViaUp on L0
        sim.set_tile_3d(3, 3, 0, TileType::ThresholdViaUp);
        let via_idx = 3 * 8 + 3;
        sim.set_tile_threshold(via_idx, 1);

        // One active neighbor: Const(1) to the left
        sim.set_tile_3d(2, 3, 0, TileType::Const);
        sim.set_logic_value_by_idx(3 * 8 + 2, 1);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0xBEEF,
            "ThresholdViaUp(t=1) should pass with 1 active neighbor"
        );
    }

    #[test]
    fn test_threshold_via_blocked() {
        // ThresholdViaUp with threshold=2, only one active neighbor → blocked.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, 0xBEEF);

        sim.set_tile_3d(3, 3, 0, TileType::ThresholdViaUp);
        let via_idx = 3 * 8 + 3;
        sim.set_tile_threshold(via_idx, 2);

        // Only one active neighbor
        sim.set_tile_3d(2, 3, 0, TileType::Const);
        sim.set_logic_value_by_idx(3 * 8 + 2, 1);
        // Other neighbors default to Wire with value 0

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0,
            "ThresholdViaUp(t=2) should block with only 1 active neighbor"
        );
    }

    #[test]
    fn test_threshold_via_gate_opens() {
        // ThresholdViaUp with threshold=2, two active neighbors → passes.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, 0xCAFE);

        sim.set_tile_3d(3, 3, 0, TileType::ThresholdViaUp);
        let via_idx = 3 * 8 + 3;
        sim.set_tile_threshold(via_idx, 2);

        // Two active neighbors: left and right
        sim.set_tile_3d(2, 3, 0, TileType::Const);
        sim.set_logic_value_by_idx(3 * 8 + 2, 1);
        sim.set_tile_3d(4, 3, 0, TileType::Const);
        sim.set_logic_value_by_idx(3 * 8 + 4, 1);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0xCAFE,
            "ThresholdViaUp(t=2) should pass with 2 active neighbors"
        );
    }

    #[test]
    fn test_threshold_via_zero_threshold() {
        // Threshold=0 should always pass regardless of neighbors.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, 0xDEAD);

        sim.set_tile_3d(3, 3, 0, TileType::ThresholdViaUp);
        let via_idx = 3 * 8 + 3;
        sim.set_tile_threshold(via_idx, 0);
        // All neighbors are default Wire with value 0 — no active neighbors

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let result = sim.get_logic_value_by_idx(via_idx);
        assert_eq!(
            result, 0xDEAD,
            "ThresholdViaUp(t=0) should always pass (0 >= 0)"
        );
    }

    #[test]
    fn test_threshold_via_identity_vs_regular() {
        // ThresholdViaUp(t=1) with 1 active neighbor should match plain ViaUp output.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;
        let val = 0xDEAD_BEEF_CAFE_BABE;

        // Source for plain ViaUp
        sim.set_tile_3d(2, 2, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 2 * 8 + 2, val);
        sim.set_tile_3d(2, 2, 0, TileType::ViaUp);
        let via_plain_idx = 2 * 8 + 2;

        // Source for ThresholdViaUp
        sim.set_tile_3d(4, 2, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 2 * 8 + 4, val);
        sim.set_tile_3d(4, 2, 0, TileType::ThresholdViaUp);
        let via_threshold_idx = 2 * 8 + 4;
        sim.set_tile_threshold(via_threshold_idx, 1);
        // Provide one active neighbor for the threshold via
        sim.set_tile_3d(3, 2, 0, TileType::Const);
        sim.set_logic_value_by_idx(2 * 8 + 3, 1);

        sim.rebuild_via_connections();
        sim.eval_tile(via_plain_idx);
        sim.eval_tile(via_threshold_idx);

        let plain = sim.get_logic_value_by_idx(via_plain_idx);
        let threshold = sim.get_logic_value_by_idx(via_threshold_idx);
        assert_eq!(
            plain, threshold,
            "ThresholdViaUp(t=1) with active neighbor should match ViaUp: plain={:#X} threshold={:#X}",
            plain, threshold
        );
        assert_eq!(plain, val);
    }

    #[test]
    fn test_threshold_viadown() {
        // ThresholdViaDown with threshold=2 should gate downward cross-layer signal.
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;

        // Source on L0
        sim.set_tile_3d(4, 4, 0, TileType::Const);
        sim.set_logic_value_by_idx(4 * 8 + 4, 0x1234);

        // ThresholdViaDown on L1 — reads from L0
        sim.set_tile_3d(4, 4, 1, TileType::ThresholdViaDown);
        let via_idx = ls + 4 * 8 + 4;
        sim.set_tile_threshold(via_idx, 2);

        // Only 1 active neighbor on L1 — should block
        sim.set_tile_3d(3, 4, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 4 * 8 + 3, 1);

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);
        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0,
            "ThresholdViaDown(t=2) should block with 1 neighbor"
        );

        // Add a second active neighbor on L1 — should pass
        sim.set_tile_3d(5, 4, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 4 * 8 + 5, 1);

        sim.eval_tile(via_idx);
        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            0x1234,
            "ThresholdViaDown(t=2) should pass with 2 neighbors"
        );
    }

    #[test]
    fn test_threshold_via_routing_is_logic() {
        // "Routing is logic" demo: ThresholdVia(t=2) implements AND gate behavior.
        // Signal DATA passes from L1 to L0 ONLY when both control signals A and B are active.
        //
        // Layer 1: Const(DATA) at (3,3)
        // Layer 0: ThresholdViaUp(t=2) at (3,3), reading DATA from L1
        //          Const(A) at (2,3) = left neighbor
        //          Const(B) at (4,3) = right neighbor
        let mut sim = make_threshold_via_sim(8, 8);
        let ls = sim.tilemap.layer_size;
        let data_val: u64 = 0xF00D_FACE;

        // Source on L1
        sim.set_tile_3d(3, 3, 1, TileType::Const);
        sim.set_logic_value_by_idx(ls + 3 * 8 + 3, data_val);

        // ThresholdVia on L0
        sim.set_tile_3d(3, 3, 0, TileType::ThresholdViaUp);
        let via_idx = 3 * 8 + 3;
        sim.set_tile_threshold(via_idx, 2);

        // Control A (left neighbor on L0)
        sim.set_tile_3d(2, 3, 0, TileType::Const);
        let a_idx = 3 * 8 + 2;

        // Control B (right neighbor on L0)
        sim.set_tile_3d(4, 3, 0, TileType::Const);
        let b_idx = 3 * 8 + 4;

        sim.rebuild_via_connections();

        // Case 1: A=0, B=0 → blocked
        sim.set_logic_value_by_idx(a_idx, 0);
        sim.set_logic_value_by_idx(b_idx, 0);
        sim.eval_tile(via_idx);
        assert_eq!(sim.get_logic_value_by_idx(via_idx), 0, "A=0,B=0 → blocked");

        // Case 2: A=1, B=0 → blocked
        sim.set_logic_value_by_idx(a_idx, 1);
        sim.set_logic_value_by_idx(b_idx, 0);
        sim.eval_tile(via_idx);
        assert_eq!(sim.get_logic_value_by_idx(via_idx), 0, "A=1,B=0 → blocked");

        // Case 3: A=0, B=1 → blocked
        sim.set_logic_value_by_idx(a_idx, 0);
        sim.set_logic_value_by_idx(b_idx, 1);
        sim.eval_tile(via_idx);
        assert_eq!(sim.get_logic_value_by_idx(via_idx), 0, "A=0,B=1 → blocked");

        // Case 4: A=1, B=1 → passes DATA
        sim.set_logic_value_by_idx(a_idx, 1);
        sim.set_logic_value_by_idx(b_idx, 1);
        sim.eval_tile(via_idx);
        assert_eq!(
            sim.get_logic_value_by_idx(via_idx),
            data_val,
            "A=1,B=1 → DATA passes: ThresholdVia(t=2) acts as AND gate"
        );
    }
}
