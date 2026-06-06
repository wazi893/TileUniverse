//! V2 hybrid execution engine.
//!
//! Sprint 90 scope: software decode/control with tile-backed ALU evaluation.
use std::cell::{Cell, RefCell};
use std::time::Instant;

use crate::simulation::{Simulation, TimingStats};
use crate::tile_cpu::TileCpuMetrics;
use crate::tile_cpu::v2_components::{
    EXT_OFFSET, EXT_RD_HI, EXT_REG_INDIRECT, EXT_RS_HI, EXT_WIDE_IMM, effective_reg,
};
use crate::tile_cpu::v2_disassembler::disassemble_v2_word;
use crate::tile_cpu::v2_mmio::{V2MmioHandle, is_v2_mmio_addr};
use crate::tile_cpu::v2_trace::{V2TraceEntry, V2TraceLog, V2TraceMemEvent, V2TraceRegWrite};
use crate::tile_cpu::v2_wiring::{CTRL_A_LUT, V2CpuIndices};
use crate::tiles::tile_meta::TileType;

/// Sprint 355: Backbone memoization cache entry.
#[derive(Debug, Clone)]
///
/// `inputs` is the full input vector at the time the snapshot was taken.
/// On a hash hit, we still verify `inputs` matches the current input vector
/// before serving — eliminates the (vanishingly small) risk of a 64-bit
/// hash collision returning a wrong snapshot. Cost: ~50 u64 compares (~50ns).
pub struct CacheEntry {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

/// Sprint 355: Backbone memoization cache.
///
/// Maps an input-vector hash to (input vector, boundary output snapshot).
/// On a hit, restoring the snapshot + marking fringe-side dependents dirty
/// + running the fringe-only kernel reproduces the same per-cycle settle
/// output without re-evaluating the 5,152-op backbone closure.
///
/// Bounded by `capacity` with simple FIFO eviction.
#[derive(Debug, Clone)]
pub struct BackboneCache {
    pub map: rustc_hash::FxHashMap<u64, CacheEntry>,
    pub order: std::collections::VecDeque<u64>,
    pub capacity: usize,
}

impl BackboneCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: rustc_hash::FxHashMap::default(),
            order: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn get(&self, key: u64) -> Option<&CacheEntry> {
        self.map.get(&key)
    }

    pub fn insert(&mut self, key: u64, inputs: Vec<u64>, outputs: Vec<u64>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, CacheEntry { inputs, outputs });
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(old_key) = self.order.pop_front() {
                self.map.remove(&old_key);
            }
        }
        self.map.insert(key, CacheEntry { inputs, outputs });
        self.order.push_back(key);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// FNV-1a 64-bit hash of a slice of u64 values. Fast, deterministic, no
/// dependency on a hasher state.
#[inline]
fn hash_u64_slice(values: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in values {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Sprint 187: Software decoder LUT for ctrl_b (branch/mem/halt control).
/// Indexed by opcode (0-31). Mirrors the physical ctrl_b Mux16to1 LUT in v2_wiring.rs.
/// Used when PC >= 64 (upper bank group) where physical extraction tiles have stale data.
const CTRL_B_LUT: [u8; 32] = [
    0x00, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x87, 0x08, 0x10, 0x08, 0x10, 0x01, 0x02, 0x03, 0x04, 0x05, 0x46,
];

#[derive(Debug, Clone, Copy, Default)]
struct PipelineLatch {
    valid: bool,
    /// Sprint 370 (Gate B.3): widened to u32 to carry the wide PC through commit.
    pc: u32,
    opcode: u8,
    rd: u8,
    ctrl_b: u8,
    ir_low: u8,
    ir_ext: u16,
    a: u64,
    b: u64,
    /// Sprint 239: Raw physical Top Mux outputs, captured during Stage F while
    /// the high trees are settled (before Const restore). Used by Stage X trunk
    /// injection so the ALU depends on the physical read path, not software.
    phys_top_a: u64,
    phys_top_b: u64,
}

#[derive(Debug, Clone, Copy)]
struct StageStats {
    comb_deltas: u32,
    comb_eval: u32,
    comb_switched: u32,
    clock_deltas: u32,
    clock_eval: u32,
    clock_switched: u32,
    glitches: u32,
    converged: bool,
}

impl StageStats {
    fn empty() -> Self {
        Self {
            comb_deltas: 0,
            comb_eval: 0,
            comb_switched: 0,
            clock_deltas: 0,
            clock_eval: 0,
            clock_switched: 0,
            glitches: 0,
            converged: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2DebugBreakpoint {
    Pc(u8),
    RegEquals { reg: usize, value: u64 },
    Halted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2DebugStopReason {
    MaxCycles,
    Breakpoint(usize),
    Halted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2DebugRunResult {
    pub cycles: u64,
    pub reason: V2DebugStopReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2DebugSnapshot {
    /// Sprint 370 (Gate B.3): u32 to hold the wide PC.
    pub pc: u32,
    pub ir_low: u8,
    pub ir_ext: u16,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub regs: [u64; 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2HybridAssistCounters {
    pub stage_f_bank_switches: u64,
    pub stage_f_mixed_dual_capture: u64,
    pub stage_x_mixed_software: u64,
    pub ram_high_bank_read_swaps: u64,
    /// Sprint 187: ROM upper bank group selection (Banks 4-7) when PC >= 64.
    pub rom_upper_bank_group_select: u64,
}

/// Sprint 154: Per-stage wall-clock timing for profiling.
#[derive(Debug, Clone, Copy, Default)]
pub struct V2StageTiming {
    pub stage_f_ns: u64,
    pub stage_x_ns: u64,
    pub branch_ns: u64,
    pub commit_ns: u64,
    pub clock_ns: u64,
    pub ram_ns: u64,
}

/// Sprint 199: Shared state for a synth validation/replacement gate.
/// Holds enabled flag + dual-path check/mismatch counters.
#[derive(Debug, Clone)]
struct SynthGate {
    enabled: Cell<bool>,
    checks: Cell<u64>,
    mismatches: Cell<u64>,
}

impl SynthGate {
    fn new() -> Self {
        Self {
            enabled: Cell::new(false),
            checks: Cell::new(0),
            mismatches: Cell::new(0),
        }
    }

    fn enable(&self) {
        self.enabled.set(true);
        self.checks.set(0);
        self.mismatches.set(0);
    }

    fn disable(&self) {
        self.enabled.set(false);
    }

    fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    fn record_check(&self) {
        self.checks.set(self.checks.get() + 1);
    }

    fn record_mismatch(&self) {
        self.mismatches.set(self.mismatches.get() + 1);
    }

    fn checks(&self) -> u64 {
        self.checks.get()
    }

    fn mismatches(&self) -> u64 {
        self.mismatches.get()
    }
}

/// 16-bit V2 hybrid CPU.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TileCpuV2 {
    /// Origin position on the tile grid.
    pub origin: (usize, usize),

    pc_idx: usize,
    pc_next_mux_idx: usize,
    pc_ingress_idx: usize, // Sprint 115: L0 ViaUp between Mux and Register8
    // Sprint 152: rom_low_mux_idx / rom_high_mux_idx removed (overwrite eliminated).

    // Sprint 150: physical selector mux outputs (Final Mux per lane)
    rom_selected_low_idx: usize,
    rom_selected_high_idx: usize,
    // Sprint 185: physical byte2/byte3 selector outputs (lanes 2/3)
    rom_selected_byte2_idx: usize,
    rom_selected_byte3_idx: usize,

    // Sprint 150 Phase 2E: hybrid injection for Target Mux ir_low/ir_high delivery
    // Sprint 151: injection indices removed (physical westback routes).

    // Sprint 133: per-bank Mux16to1 indices for hybrid bank selection
    bank_low_mux_indices: [usize; 4],
    bank_high_mux_indices: [usize; 4],

    /// Sprint 187: Bank 4-7 Mux16to1 output indices [bank_within_group][lane]
    bank47_mux_indices: [[usize; 4]; 4],
    /// Sprint 211: Original Final Mux indices (Banks 0-3 physical selector output)
    final_mux_indices: [usize; 4],
    /// Sprint 211: Super Mux injection Const indices — LEFT (Banks 4-7)
    super_mux_inject_indices: [usize; 4],
    /// Sprint 211: Super Mux injection Const indices — RIGHT (Banks 0-3)
    super_mux_inject_right_indices: [usize; 4],
    // bank_byte2/3_mux_indices removed — Sprint 185 physical ir_ext
    extract_opcode_shr_idx: usize,
    extract_opcode_bit4_idx: usize,
    extract_rd_bit_indices: [usize; 3], // [rd0, rd1, rd2]
    extract_rs_field_idx: usize,
    ctrl_a_mux_idx: usize,
    ctrl_b_mux_idx: usize,
    branch_ctrl_b_l1_tap_idx: usize,
    branch_flag_z_l1_tap_idx: usize,
    branch_flag_c_l1_tap_idx: usize,
    branch_taken_core_idx: usize,
    pub(crate) branch_dirty_indices: Vec<usize>,
    branch_flag_dirty_indices: Vec<usize>,

    op_a_root_idx: usize,
    op_b_root_idx: usize,
    pipeline_dirty_indices: Vec<usize>,
    pipeline_backbone_dirty_indices: Vec<usize>,
    // Sprint 266: Targeted dirty set for upper-bank cycles.
    // Includes spine + R-override + branch-target + via-decode tiles.
    // Used instead of force_full_dirty to avoid marking all ~7,400 pipeline tiles.
    upper_bank_dirty_indices: Vec<usize>,
    pipeline_reg_data_dirty_indices: [Vec<usize>; 16],
    reg_tap_l1_indices: [usize; 16],

    rd_onehot_decode_idx: usize,
    rd_decode_l0_chain: Vec<usize>,
    we_mask_const_idx: usize,
    commit_dirty_indices: Vec<usize>,
    // Sprint 169: register writeback dirty indices (conditional).
    reg_wb_dirty_indices: Vec<usize>,

    flag_we_mask_const_idx: usize,
    flag_commit_dirty_indices: Vec<usize>,

    // Sprint 125: physical LR and Target Selection Mux
    _lr_idx: usize,
    _lr_dirty_indices: Vec<usize>,

    reg_indices: [usize; 16],
    flag_z_idx: usize,
    flag_c_idx: usize,
    ram_indices: [usize; 128],
    regs: [Cell<u64>; 16],
    ram: [Cell<u64>; 128],
    /// Sprint 362 (Gate A): extended main memory ("DRAM analog") at addresses
    /// 128..(128+main_mem.len()). Empty by default — allocated only by
    /// `configure_main_memory`. The CPU core stays in tiles; main memory is
    /// external software-backed storage reached over the (physical) bus.
    main_mem: Vec<Cell<u64>>,
    /// Sprint 362: data-address mask. Default 0x7F (7-bit, 128 locations);
    /// `configure_main_memory` widens it to a power of two covering main memory.
    /// The PC address space is unaffected (still 7-bit — that is Gate B).
    mem_addr_mask: Cell<usize>,
    /// Sprint 369 (Gate B.2): extended 8-bit instruction address space. When true,
    /// the physical PC mask is widened to 8 bits and instructions at PC>=128 are
    /// fetched from `program_ext` and injected through the Super-Mux Const path
    /// (the gated-Const-fallback authority tier — software selects the IR, real
    /// tiles decode/execute it). PC sequencing for the extended range is software-
    /// authoritative (the existing write_pc fallback). Default false → byte-
    /// identical to pre-369 (every PC read masks & 0x7F, ROM caps at 128).
    extended_pc: bool,
    /// Sprint 369: physical PC mask (0x7F default, 0xFF when extended_pc).
    /// Sprint 370 (Gate B.3): widened to u32 for the 16-bit wide-PC address space.
    pc_phys_mask: u32,
    /// Sprint 369: upper-half program store for instructions 128.. The physical
    /// ROM tiles only hold 0..127; this is the software-backed extension.
    program_ext: Vec<u32>,
    /// Sprint 370 (Gate B.3): instruction-address width in bits (7 default, 8 for
    /// extended_pc, 16 for wide_pc). >8 selects a Register64 PC/LR tile.
    pc_addr_bits: u8,
    /// Sprint 370: whether the PC/LR tiles are physically Register64 (wide PC).
    wide_pc: bool,
    /// Sprint 370 (Gate B.3): PC widened to u32 to hold the 16-bit address space.
    pc: Cell<u32>,
    flag_z: Cell<bool>,
    flag_c: Cell<bool>,
    halted: Cell<bool>,
    // Sprint 117: prev_ram_write_enabled removed ??? physical enable auto-deasserts.
    /// Sprint 370 (Gate B.3): LR widened to u32 for wide return addresses.
    lr: Cell<u32>,
    // Sprint 126: prev_software_pc_enable removed ??? all PC enables are physical.
    // Sprint 113: prev_neg_override removed ??? NEG L enable is now physical.
    // Sprint 123: prev_carry_shift_override removed ??? carry is now fully physical.
    pipeline_force_full_dirty: Cell<bool>,
    // Sprint 269: Set to true when a runtime Const-swap is active, disabling compact eval.
    // Sprint 304: Should never be set in max_authority with constswap settle ops.
    compact_eval_inhibit: Cell<bool>,
    // Sprint 304: Counter tracking how often compact_eval_inhibit was set to true.
    compact_eval_inhibit_count: Cell<u64>,
    // Sprint 270: Set to true when tile types change after compact ops were built
    // (e.g., external enable_synth_* calls). Forces levelized fallback.
    compact_ops_stale: Cell<bool>,
    changed_regs_mask: Cell<u16>,
    // Sprint 167: countdown for RAM writeback gating.
    // Set to 2 on ST/STB; decremented each cycle. When 0, skip 128-tile RAM sync.
    ram_writeback_countdown: Cell<u8>,
    // Sprint 169: countdown for reg-WB dirty gating.
    // Set to 2 on reg-write; decremented each cycle. When 0, skip ~305 mark_dirty calls.
    reg_wb_countdown: Cell<u8>,
    // Sprint 167: opt-in stage timing (default off, benchmarks enable).
    enable_stage_timing: bool,
    // Sprint 288: Separate flag for convergence probe (expensive shadow pass).
    // Previously coupled with enable_stage_timing, inflating Stage F measurements.
    enable_convergence_probe: bool,
    // Sprint 307: Toggle for A/B profiling of prefiltered settle.
    use_prefiltered_settle: Cell<bool>,
    // Sprint 312: Toggle for no-dirty settle (A/B profiling).
    use_no_dirty_settle: Cell<bool>,
    latch: Cell<PipelineLatch>,
    last_stage_x_valid: Cell<bool>,
    hybrid_stage_f_bank_switches: Cell<u64>,
    hybrid_stage_f_mixed_dual_capture: Cell<u64>,
    hybrid_stage_x_mixed_software: Cell<u64>,
    hybrid_ram_high_bank_read_swaps: Cell<u64>,
    /// Sprint 187: ROM upper bank group selection counter.
    hybrid_rom_upper_bank_group_select: Cell<u64>,
    last_stage_x_mmio_reads: RefCell<Vec<(u8, u64)>>,
    last_stage_x_mmio_writes: RefCell<Vec<(u8, u64)>>,
    retired_count: Cell<u64>,
    mmio: Option<V2MmioHandle>,
    cycle_count: Cell<u64>,
    last_stage_timing: Cell<V2StageTiming>,

    // Sprint 262: Per-cycle propagation counters (aggregate — all paths).
    propagate_calls_total: Cell<u64>,
    propagate_tiles_total: Cell<u64>,
    // Sprint 274: Stage F cone convergence instrumentation.
    cone_single_pass_checks: Cell<u64>,
    cone_residual_changes: Cell<u64>,
    // Sprint 277: Scan vs active instrumentation for scheduler design.
    // Tracks total ops scanned vs ops actually dirty across all compact_dirty calls.
    compact_scan_total: Cell<u64>,
    compact_active_total: Cell<u64>,
    // Sprint 311: Per-path propagation counters (calls, evals).
    prop_cone_calls: Cell<u64>,
    prop_cone_evals: Cell<u64>,
    prop_settle_calls: Cell<u64>,
    prop_settle_evals: Cell<u64>,
    prop_settle_scan: Cell<u64>,
    prop_constswap_calls: Cell<u64>,
    prop_constswap_evals: Cell<u64>,
    prop_branch_calls: Cell<u64>,
    prop_branch_evals: Cell<u64>,
    prop_commit_calls: Cell<u64>,
    prop_commit_evals: Cell<u64>,
    // Sprint 313: Settle call reason histogram.
    // [0]=combined, [1]=via_decode, [2]=mov_ldi, [3]=mov_ldi_wide,
    // [4]=alu_wide_imm, [5]=alu_sra, [6]=alu_trunk
    settle_reason_counts: [Cell<u64>; 7],

    pub grid_width: usize,
    pub tile_count: usize,

    // Sprint 146: zone closure scopes for scoped propagation
    pipeline_scope: Vec<u32>,
    branch_scope: Vec<u32>,
    commit_scope: Vec<u32>,

    // Sprint 147: bitset-form scope masks for O(L1+active) masked drain
    pipeline_scope_mask: Vec<u64>,
    branch_scope_mask: Vec<u64>,
    commit_scope_mask: Vec<u64>,

    // Sprint 262: topologically sorted evaluation order for levelized propagation
    pub(crate) pipeline_eval_order: Vec<usize>,
    // Sprint 263: branch + commit scope eval orders
    pub(crate) branch_eval_order: Vec<usize>,
    pub(crate) commit_eval_order: Vec<usize>,
    // Sprint 267: compact evaluator for pipeline scope
    pub(crate) pipeline_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) pipeline_compact_wvia: Vec<(usize, u8, u64)>,
    // Sprint 272: cone-pruned compact ops (only tiles feeding output seeds)
    pub(crate) pipeline_cone_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) pipeline_cone_wvia: Vec<(usize, u8, u64)>,
    // Sprint 274: cone membership bitset for frontier-only dirty propagation
    pub(crate) pipeline_cone_set: Vec<u64>,
    // Sprint 273: JIT-compiled cone evaluation function
    #[cfg(feature = "cranelift_jit")]
    pub(crate) pipeline_cone_jit:
        Option<std::sync::Arc<crate::tile_cpu::tile_jit::TileEvalJitProgram>>,
    // Sprint 335: JIT-compiled settle evaluation function
    #[cfg(feature = "cranelift_jit")]
    pub(crate) settle_jit: Option<std::sync::Arc<crate::tile_cpu::tile_jit::TileEvalJitProgram>>,
    // Sprint 352: Backbone JIT — compiled backbone-only ops for no-dirty evaluation.
    // Backbone (93.7% of settle) is structurally fixed every cycle. JIT-compiling
    // only backbone avoids COP_WIRE/COP_GENERIC in fringe that block full settle JIT.
    #[cfg(feature = "cranelift_jit")]
    pub(crate) backbone_jit: Option<std::sync::Arc<crate::tile_cpu::tile_jit::TileEvalJitProgram>>,
    // Sprint 352: Backbone JIT enable flag (separate from settle_jit_enabled).
    pub(crate) backbone_jit_enabled: Cell<bool>,
    // Sprint 270: compact evaluator for branch + commit scopes
    pub(crate) branch_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) branch_compact_wvia: Vec<(usize, u8, u64)>,
    pub(crate) commit_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) commit_compact_wvia: Vec<(usize, u8, u64)>,
    // Sprint 278: Ordered active-work schedule for commit scope.
    pub(crate) commit_schedule: Option<crate::simulation::CompactSchedule>,
    // Sprint 279: Schedules for branch and clock scopes.
    pub(crate) branch_schedule: Option<crate::simulation::CompactSchedule>,
    pub(crate) clock_schedule: Option<crate::simulation::CompactSchedule>,
    // Sprint 321: Phase-local clock cascade switch counts (per schedule.ops slot).
    // When non-empty, clock dispatch uses tick_clock_edge_scheduled_counted.
    pub(crate) clock_cascade_counts: std::cell::RefCell<Vec<u32>>,
    // Sprint 325: Dual-profile pruned clock cascade — separate for flag-writing
    // and non-flag cycles, matching the clock_cache bifurcation at line 3311.
    pub(crate) live_clock_ops_flags: std::cell::RefCell<Vec<crate::simulation::CompactOp>>,
    pub(crate) live_clock_wvia_flags: std::cell::RefCell<Vec<(usize, u8, u64)>>,
    pub(crate) live_clock_ops_noflags: std::cell::RefCell<Vec<crate::simulation::CompactOp>>,
    pub(crate) live_clock_wvia_noflags: std::cell::RefCell<Vec<(usize, u8, u64)>>,
    // Sprint 324/325: Warmup counting state — separate counters for each profile.
    clock_cascade_counts_flags: std::cell::RefCell<Vec<u32>>,
    clock_cascade_counts_noflags: std::cell::RefCell<Vec<u32>>,
    clock_warmup_flags_remaining: Cell<u32>,
    clock_warmup_noflags_remaining: Cell<u32>,
    // Sprint 324: Enable clock auto-warmup. Set by max_authority config.
    pub(crate) clock_auto_warmup_enabled: Cell<bool>,
    // Sprint 336: Explicit settle JIT gate (separate from clock auto-warmup).
    pub(crate) settle_jit_enabled: Cell<bool>,
    // Sprint 339: Precomputed out-of-scope frontier table for settle JIT.
    // Flat-packed: frontier_offsets[op_slot]..frontier_offsets[op_slot+1] indexes into frontier_targets.
    // Each target is a tile index that needs mark_dirty when the op changes value.
    pub(crate) settle_frontier_offsets: Vec<u32>,
    pub(crate) settle_frontier_targets: Vec<u32>,
    // Sprint 337: JIT settle sub-phase timing (accumulated ns, opt-in).
    pub(crate) jit_settle_eval_ns: Cell<u64>,
    pub(crate) jit_settle_dirty_ns: Cell<u64>,
    pub(crate) jit_settle_passes: Cell<u64>,
    pub(crate) jit_settle_changed: Cell<u64>,
    pub(crate) jit_settle_profiled: Cell<bool>,
    // Sprint 338: Per-pass changed counts (accumulated across cycles).
    pub(crate) jit_settle_pass1_changed: Cell<u64>,
    pub(crate) jit_settle_pass2_changed: Cell<u64>,
    // Sprint 328: Phase-local settle switch counts (per settle_compact_ops slot).
    pub(crate) settle_switch_counts: std::cell::RefCell<Vec<u32>>,
    // Sprint 330: Settle slot-level overlap buckets (accumulated across cycles).
    pub(crate) settle_overlap_buckets: [Cell<u64>; 5],
    // Sprint 332: Commit sub-phase timing (accumulated ns across cycles).
    pub(crate) commit_drain_ns: Cell<u64>,
    pub(crate) commit_worklist_ns: Cell<u64>,
    pub(crate) commit_profiled: Cell<bool>,
    // Sprint 334: Stage F sub-phase timing (accumulated ns, opt-in).
    pub(crate) stage_f_cone_ns: Cell<u64>,
    pub(crate) stage_f_settle_ns: Cell<u64>,
    pub(crate) stage_f_inject_ns: Cell<u64>,
    pub(crate) stage_f_profiled: Cell<bool>,
    // Sprint 289: Pipeline schedule for combined settle (use_cone=false path).
    pub(crate) pipeline_schedule: Option<crate::simulation::CompactSchedule>,
    // Sprint 290: Settle-scope compact ops — forward closure from all injection seeds.
    // Subset of pipeline_compact_ops containing only tiles reachable from Stage F injections.
    pub(crate) settle_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) settle_compact_wvia: Vec<(usize, u8, u64)>,
    // Sprint 291: Settle-scope cone set — bitset for frontier-only dirty propagation.
    // Sprint 306: Also used as scope_mask for prefiltered settle evaluation.
    pub(crate) settle_cone_set: Vec<u64>,
    // Sprint 318: Targeted trunk re-settle ops (forward closure from trunk terminals).
    // Used for alu_trunk re-settle instead of full 5,492-op settle scope.
    pub(crate) trunk_settle_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) trunk_settle_wvia: Vec<(usize, u8, u64)>,
    // Sprint 306: Prefiltered settle lookup tables.
    // idx_to_slot: tile_idx → position in settle_compact_ops (u32::MAX if absent).
    // wvia_slot_map: slot → index into settle_compact_wvia (u32::MAX if not WVIA).
    pub(crate) settle_idx_to_slot: Vec<u32>,
    pub(crate) settle_wvia_slot_map: Vec<u32>,
    pub(crate) settle_idx_to_slot_constswap: Vec<u32>,
    pub(crate) settle_wvia_slot_map_constswap: Vec<u32>,
    // Sprint 308: Block-level clean-skip for settle scan.
    // Per 64-op block: which dirty segments it touches + bitmask of tiles.
    // Flat-packed CSR: offsets[block]..offsets[block+1] → entries[(seg_idx, mask)].
    pub(crate) settle_block_seg_offsets: Vec<u32>,
    pub(crate) settle_block_seg_entries: Vec<(u32, u64)>,
    pub(crate) settle_block_wvia_counts: Vec<u8>,
    pub(crate) settle_block_seg_offsets_cs: Vec<u32>,
    pub(crate) settle_block_seg_entries_cs: Vec<(u32, u64)>,
    pub(crate) settle_block_wvia_counts_cs: Vec<u8>,
    // Sprint 292: Settle-scope schedule — active-work propagation on forward closure.
    // Zero COP_GENERIC in settle scope → no re-drain, single pass guaranteed.
    pub(crate) settle_schedule: Option<crate::simulation::CompactSchedule>,
    // Sprint 293: Forward-only deps for single-pass settle. Derived from schedule
    // deps but filtered to dep_slot > current_slot (downstream only). Eliminates
    // backward dirty marks that cause the 2nd pass in compact_dirty.
    pub(crate) settle_forward_deps_data: Vec<u32>,
    pub(crate) settle_forward_deps_offsets: Vec<u32>,
    // Sprint 304: Constswap variant of settle compact ops — R-Mux output tiles
    // pre-encoded as COP_CONST. Used for LDI.W/wide-imm/SRA instructions so they
    // can use compact settle instead of falling back to levelized propagation.
    pub(crate) settle_compact_ops_constswap: Vec<crate::simulation::CompactOp>,
    pub(crate) settle_compact_wvia_constswap: Vec<(usize, u8, u64)>,
    // Sprint 305: Backbone/fringe split for worklist-driven settle.
    // Backbone = structurally fixed (backbone_settle_seeds forward closure, ~93.7%).
    // Fringe = instruction-dependent (R0-R7 register paths, ~6.3%).
    // Backbone uses propagate_compact_scheduled (worklist), fringe uses compact_dirty.
    pub(crate) settle_backbone_schedule: Option<crate::simulation::CompactSchedule>,
    pub(crate) settle_fringe_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) settle_fringe_wvia: Vec<(usize, u8, u64)>,
    // Sprint 305: Constswap variants for backbone/fringe.
    pub(crate) settle_backbone_schedule_constswap: Option<crate::simulation::CompactSchedule>,
    pub(crate) settle_fringe_ops_constswap: Vec<crate::simulation::CompactOp>,
    pub(crate) settle_fringe_wvia_constswap: Vec<(usize, u8, u64)>,
    // Sprint 352: Backbone ops filtered for JIT compatibility (no COP_GENERIC/COP_WIRE).
    // These are the ops that get JIT-compiled into backbone_jit.
    pub(crate) backbone_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) backbone_wvia: Vec<(usize, u8, u64)>,
    // Sprint 352: Bitset of backbone tile indices — used as scope_mask for
    // dirty_dependents_frontier so only fringe/external tiles get marked dirty.
    pub(crate) backbone_cone_set: Vec<u64>,
    // Sprint 354: Single-pass hybrid settle gate.
    // When true, propagate_pipeline_until_settled walks settle_compact_ops once
    // and treats backbone ops as unconditional (no dirty check) while keeping
    // dirty-checked semantics for fringe ops. Opt-in for A/B profiling.
    pub(crate) hybrid_settle_enabled: Cell<bool>,

    // Sprint 355: Backbone memoization.
    // External input tile indices: tiles read by backbone ops that are NOT
    // themselves in backbone (i.e., the inputs to the backbone closure).
    // Their values fully determine backbone output state.
    pub(crate) backbone_input_indices: Vec<u32>,
    // Boundary output tile indices: backbone tiles read by fringe ops.
    // Snapshotted into the cache; restored on hit.
    pub(crate) backbone_output_indices: Vec<u32>,
    // Cache mapping input-vector hash to boundary output snapshot.
    pub(crate) backbone_cache: std::cell::RefCell<BackboneCache>,
    // Memoization enable gate (opt-in).
    pub(crate) memoization_enabled: Cell<bool>,
    // Per-cycle cache hit/miss counters.
    pub(crate) memo_hits: Cell<u64>,
    pub(crate) memo_misses: Cell<u64>,

    // Sprint 356: Decode-only memoization.
    // Decode externals = settle externals NOT carrying register state
    // (PC bits, IR bits, decoder Consts, ROM data injection points).
    pub(crate) decode_input_indices: Vec<u32>,
    // Decode tile outputs = settle op outputs whose value is a pure
    // function of decode externals (forward-closure, no register taint).
    // COP_CONST excluded — externally set, never re-evaluated.
    pub(crate) decode_output_indices: Vec<u32>,
    // Execute compact ops = settle ops NOT in the decode closure.
    // These read register-state-tainted values and must be re-evaluated each settle call.
    pub(crate) execute_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) execute_compact_wvia: Vec<(usize, u8, u64)>,
    // Bitset of execute op output indices — used as backbone_set when
    // running the hybrid kernel on the execute portion only.
    pub(crate) execute_cone_set: Vec<u64>,
    // Cache mapping decode-input-vector hash to decode tile snapshot.
    pub(crate) decode_cache: std::cell::RefCell<BackboneCache>,
    // Decode memoization gate (opt-in, takes priority over S355 / S354).
    pub(crate) decode_memoization_enabled: Cell<bool>,
    // Per-call cache hit/miss counters.
    pub(crate) decode_memo_hits: Cell<u64>,
    pub(crate) decode_memo_misses: Cell<u64>,

    // Sprint 357: Adaptive enable for decode memoization.
    //
    // Three-state machine: WARMUP (always use cache, accumulating hit/miss
    // counts), ENGAGED (always use cache, decision was made), or DISABLED
    // (skip cache entirely, fall through to S355/baseline). Transition out
    // of WARMUP happens once total cache calls reaches `adaptive_warmup_calls`,
    // based on lifetime hit rate vs `adaptive_decode_threshold`.
    //
    // Independent of `decode_memoization_enabled` — when adaptive is off,
    // decode-memo behaves as in S356 (always-on).
    pub(crate) adaptive_decode_enabled: Cell<bool>,
    pub(crate) adaptive_decode_threshold: Cell<f32>,
    /// Number of cache calls to observe before transitioning out of WARMUP
    /// into ENGAGED or DISABLED.
    pub(crate) adaptive_warmup_calls: Cell<u32>,
    /// 0 = warmup, 1 = engaged, 2 = disabled.
    pub(crate) adaptive_decode_mode: Cell<u8>,
    /// Metric: number of calls skipped due to adaptive being in DISABLED.
    pub(crate) adaptive_decode_skipped: Cell<u64>,
    // Sprint 357 (legacy, retained for potential future probe-based mode):
    // unused under the simple state-machine design.
    pub(crate) adaptive_decode_window: Cell<u32>,
    pub(crate) adaptive_probe_interval: Cell<u32>,
    pub(crate) adaptive_decode_history: std::cell::RefCell<std::collections::VecDeque<bool>>,
    pub(crate) adaptive_probe_counter: Cell<u32>,
    pub(crate) adaptive_decode_probes: Cell<u64>,

    // Sprint 166: unified clock scope mask for masked clock tick
    pub(crate) clock_scope_mask: Vec<u64>,
    // Sprint 276: compact ops for clock edge combinational cascade
    pub(crate) clock_compact_ops: Vec<crate::simulation::CompactOp>,
    pub(crate) clock_compact_wvia: Vec<(usize, u8, u64)>,

    // Sprint 168: Pre-filtered clock-sensitive tiles within clock_scope_mask.
    pub(crate) in_scope_clock_cache: Vec<usize>,
    /// Sprint 232: Clock cache WITHOUT flag Register8 tiles (flag_z, flag_c).
    /// Used on non-flag-writing instructions so flag tiles keep their value.
    in_scope_clock_cache_no_flags: Vec<usize>,

    // Sprint 196-198: synth replacement/validation gates (Sprint 199: unified struct)
    synth_branch: SynthGate,
    synth_branch_table: [bool; 32],
    branch_taken_saved_tile_type: Cell<TileType>,
    // Sprint 200: optional synth-generated block for live branch evaluation
    synth_branch_block: Option<crate::synth::integration::InjectedBlock>,

    synth_rd_decode: SynthGate,
    rd_decode_saved_tile_type: Cell<TileType>,
    // Sprint 200: optional synth-generated block for live rd_decode evaluation
    synth_rd_decode_block: Option<crate::synth::integration::InjectedBlock>,

    synth_ctrl_b: SynthGate,
    synth_ctrl_a: SynthGate,
    // Sprint 205: saved tile types for ctrl_a authority (Const swap on enable)
    we_mask_saved_tile_type: Cell<TileType>,
    flag_we_mask_saved_tile_type: Cell<TileType>,
    synth_operand: SynthGate,
    /// Sprint 231: physical operand authority — tree root reads are authoritative for R0-R7.
    physical_operand_authority: Cell<bool>,
    op_authority_checks: Cell<u64>,
    op_authority_mismatches: Cell<u64>,

    /// Sprint 233: Physical pre-commit decode delivery authority.
    physical_rd_decode: Cell<bool>,
    rd_decode_authority_checks: Cell<u64>,
    rd_decode_authority_mismatches: Cell<u64>,
    physical_we_mask: Cell<bool>,
    we_mask_authority_checks: Cell<u64>,
    we_mask_authority_mismatches: Cell<u64>,
    physical_flag_we_mask: Cell<bool>,
    flag_we_mask_authority_checks: Cell<u64>,
    flag_we_mask_authority_mismatches: Cell<u64>,

    /// Sprint 234: Physical Super Mux propagation. When true, LEFT/RIGHT Const tiles
    /// are injected with ROM lane values and the physical Mux selects via PC[6].
    physical_super_mux: Cell<bool>,
    super_mux_checks: Cell<u64>,
    super_mux_mismatches: Cell<u64>,

    /// Sprint 254: Dual-path verification for upper-bank IR delivery.
    upper_bank_ir_checks: Cell<u64>,
    upper_bank_ir_mismatches: Cell<u64>,

    /// Sprint 235: Physical writeback-data authority for non-ALU register-producing ops.
    /// LDI (narrow), MOV, conditional moves (when taken) — bank 0, R0-R7 only.
    physical_wb_data_authority: Cell<bool>,
    wb_data_checks: Cell<u64>,
    wb_data_mismatches: Cell<u64>,
    /// Sprint 235: Add tile L-operand enable Const index.
    alu_add_l_enable_idx: usize,
    /// Sprint 235: Add tile L-operand Mux index (dirty target after injection).
    alu_add_l_mux_idx: usize,

    /// Sprint 236: Physical high-register writeback authority.
    /// R8-R15 participate in clock-edge capture via merge mux fabric.
    physical_high_reg_writeback: Cell<bool>,
    high_reg_wb_checks: Cell<u64>,
    high_reg_wb_mismatches: Cell<u64>,

    /// Sprint 245: Sub-function ALU delivery authority.
    /// Software result injected into wb_data_mux for MUL/SRA/CLZ/CTZ/POPCNT.
    physical_sub_fn_delivery: Cell<bool>,
    sub_fn_delivery_checks: Cell<u64>,
    sub_fn_delivery_mismatches: Cell<u64>,
    /// Sprint 246: Sub-function flag authority (SRA/CLZ/CTZ/POPCNT/MUL with C_WE suppression).
    physical_sub_fn_flag_authority: Cell<bool>,
    /// Sprint 246: LD/LDB delivery + flag authority.
    physical_load_delivery: Cell<bool>,
    load_delivery_checks: Cell<u64>,
    load_delivery_mismatches: Cell<u64>,
    /// Sprint 236: Per-register Const injection indices for R8-R15.
    high_reg_wb_data_const_indices: [usize; 8],
    high_reg_we_const_indices: [usize; 8],
    high_reg_merge_mux_indices: [usize; 8],

    /// Sprint 237: High-register read authority — high tree + top mux Const indices.
    high_tree_a_sel_const_indices: [usize; 7],
    high_tree_b_sel_const_indices: [usize; 7],
    high_tree_a_data_const_indices: [usize; 8],
    high_tree_b_data_const_indices: [usize; 8],
    top_mux_a_sel_const_idx: usize,
    top_mux_b_sel_const_idx: usize,
    /// Sprint 237 Option 2-plus: dirty seed sets for Const-injection propagation.
    high_tree_a_dirty_indices: Vec<usize>,
    high_tree_b_dirty_indices: Vec<usize>,
    top_mux_a_dirty_indices: Vec<usize>,
    top_mux_b_dirty_indices: Vec<usize>,

    /// Sprint 239: Terminal L1 trunk tiles for ALU operand injection.
    alu_a_trunk_terminal_idx: usize,
    alu_b_trunk_terminal_idx: usize,
    alu_a_downstream_dirty: Vec<usize>,
    alu_b_downstream_dirty: Vec<usize>,

    /// Sprint 243: R Mux output WireDown tiles — injection targets for wide-immediate.
    alu_r_mux_output_indices: [usize; 8],
    /// Sprint 239: Verification counters for high-register ALU trunk re-source.
    upper_alu_trunk_checks: Cell<u64>,
    upper_alu_trunk_mismatches: Cell<u64>,

    synth_ram_decode: SynthGate,
    ram_write_decode_idx: usize,
    ram_write_gate_idx: usize,
    ram_decode_saved_tile_type: Cell<TileType>,
    // Sprint 201: optional synth-generated block for live ram_decode evaluation
    synth_ram_decode_block: Option<crate::synth::integration::InjectedBlock>,
    // Sprint 201: optional synth-generated block for live ctrl_b evaluation
    synth_ctrl_b_block: Option<crate::synth::integration::InjectedBlock>,
    // Sprint 205: optional synth-generated block for ctrl_a authority (fallback: CTRL_A_LUT)
    synth_ctrl_a_block: Option<crate::synth::integration::InjectedBlock>,
    /// Sprint 209: combined ctrl_a+ctrl_b LUT. Bits [7:0] = ctrl_a, [15:8] = ctrl_b.
    /// When set, feeds both ctrl_a injection and ctrl_b decode from a single lookup.
    combined_decode_lut: Option<[u16; 32]>,
    /// Sprint 209: number of times the combined decode LUT was consulted.
    combined_decode_checks: Cell<u64>,

    // Sprint 218: ALU readback dual-path verification
    synth_alu: SynthGate,
    /// Per-opcode ALU mismatch counters (indexed by opcode 0..31).
    alu_opcode_mismatches: [Cell<u64>; 32],
    /// Sprint 218: ALU result mux root tile index.
    wb_alu_root_idx: usize,
    /// Sprint 218: wb_data Mux tile index (final writeback source).
    wb_data_mux_idx: usize,
    /// Sprint 218: Register capture mismatch statistics.
    reg_capture_checks: Cell<u64>,
    reg_capture_mismatches: Cell<u64>,
    /// Sprint 218 D2: ALU mismatches where ctrl_a[2:0] (mux selector) also mismatched.
    alu_mux_select_mismatches: Cell<u64>,

    /// Sprint 219: Physical ALU authority — bitmask of opcodes using physical ALU result.
    physical_alu_opcodes: Cell<u32>,

    /// Sprint 220: Physical register writeback authority.
    /// When true, R0-R7 are in the clock cache and software injection is skipped
    /// for physically-authoritative ALU ops (clock capture writes the correct value).
    physical_reg_writeback: Cell<bool>,
    /// Sprint 220: Post-capture register writeback verification counters.
    reg_wb_checks: Cell<u64>,
    reg_wb_mismatches: Cell<u64>,

    /// Sprint 221: Physical flag writeback authority.
    /// When true, flag_z_idx and flag_c_idx are in the clock cache and software
    /// flag injection is skipped for flag-writing instructions.
    physical_flag_writeback: Cell<bool>,
    /// Sprint 221: Post-capture flag verification counters.
    flag_wb_checks: Cell<u64>,
    flag_wb_mismatches: Cell<u64>,
    flag_z_mismatches: Cell<u64>,
    flag_c_mismatches: Cell<u64>,

    /// Sprint 222: Physical ctrl_a/ctrl_b decode authority for bank0.
    /// Sprint 223: Extended to all banks (Const-swap injection at extraction
    /// ingress provides correct opcode bits for upper bank). LUT fallback on mismatch.
    physical_decode: Cell<bool>,
    /// Sprint 222: Decode authority verification counters.
    decode_ctrl_b_checks: Cell<u64>,
    decode_ctrl_b_mismatches: Cell<u64>,
    decode_ctrl_a_checks: Cell<u64>,
    decode_ctrl_a_mismatches: Cell<u64>,

    /// Sprint 223/224: Upper-bank PC override verification counters.
    /// Sprint 224: Non-RET branches (kind 0-6) use physical PC unconditionally.
    /// Only RET (kind 7) is mismatch-gated (LR delivery stale on upper bank).
    pc_override_checks: Cell<u64>,
    pc_override_mismatches: Cell<u64>,
    /// Sprint 224: Per-branch-kind PC mismatch breakdown.
    pc_mismatch_per_kind: [Cell<u64>; 8],

    /// Sprint 225: Physical branch direction authority.
    /// When true, the physical Mux16to1 branch LUT computes branch_taken — synth
    /// injection is skipped. Dual-path verification compares against synth truth table.
    physical_branch: Cell<bool>,
    /// Sprint 225: Branch direction verification counters.
    branch_dir_checks: Cell<u64>,
    branch_dir_mismatches: Cell<u64>,

    /// Sprint 229: Physical RAM writeback authority. When true, RAM tiles are
    /// un-elided from the clock cache and participate in clock edge captures.
    /// The Ram tile's built-in WE gate (output = UP!=0 ? LEFT : current) provides
    /// physical write-enable gating. Diagnostic counters verify correctness.
    physical_ram_writeback: Cell<bool>,
    /// Sprint 229: RAM readback verification counters.
    ram_wb_checks: Cell<u64>,
    ram_wb_mismatches: Cell<u64>,
    /// Sprint 229: Mismatches split by store vs non-store (WE gating proof).
    ram_wb_store_mismatches: Cell<u64>,
    ram_wb_nonstore_mismatches: Cell<u64>,
    /// Sprint 247: RAM store authority. When true, ST/STB Const-swaps the
    /// data bus ViaUp (Ram tile's LEFT neighbor) and injects WE into the UP
    /// neighbor. Physical clock-edge capture: UP!=0 → stored=LEFT.
    /// Bank 0 only (cells 0-7); cells 8-127 retain software write.
    physical_ram_store_authority: Cell<bool>,
    /// Sprint 247: Saved ViaUp tile (x,y,z,TileType) for post-clock restore.
    /// None = no restore pending.
    ram_store_saved_via: Cell<Option<(usize, usize, usize, TileType)>>,
    /// Sprint 247: UP neighbor index to restore to 0 after clock edge.
    /// 0 = no restore pending.
    ram_store_inject_up_idx: Cell<usize>,

    /// Sprint 230: Three-point RAM snapshots for store-cycle diagnosis.
    /// Captured for bank-0 cells (0..8) on every store cycle.
    /// [0] = post-commit (after commit_scope propagation, before re-inject)
    /// [1] = post-reinject (after 128-cell re-inject, before clock edge)
    /// [2] = post-clock (after clock edge, before countdown writeback)
    ram_snap_post_commit: [Cell<u64>; 8],
    ram_snap_post_reinject: [Cell<u64>; 8],
    ram_snap_post_clock: [Cell<u64>; 8],
    ram_snap_store_addr: Cell<usize>,

    /// Sprint 248: SRA sign-extension synth block. Computes sign extension mask
    /// from [sign, s0, s1, s2] inputs, producing 7 outputs (mask bits 57-63).
    synth_sra_block: Option<crate::synth::integration::InjectedBlock>,
    /// Sprint 248: Physical SRA computation authority. When true, SRA uses
    /// physical SHR from ALU + synth sign-extension mask instead of software.
    physical_sra_computation: Cell<bool>,
    /// Sprint 362: Physical 8x8 MUL synth block. 16 inputs (A[0:7], B[0:7]), 16 outputs (P[0:15]).
    synth_mul_block: Option<crate::synth::integration::InjectedBlock>,
    physical_mul_authority: Cell<bool>,
    /// Sprint 248: SRA dual-path verification counters.
    sra_computation_checks: Cell<u64>,
    sra_computation_mismatches: Cell<u64>,

    // Sprint 250: Hierarchical CLZ/CTZ/POPCNT synth blocks.
    // Shared bitscan family (CLZ + CTZ use same physical blocks, different input order).
    /// 8 byte-level BITSCAN8 blocks, one per byte. Shared between CLZ and CTZ.
    /// Each: 8 inputs (b0-b7), 4 outputs (has_nz, count[2:0]).
    synth_bitscan8_blocks: Option<[crate::synth::integration::InjectedBlock; 8]>,
    /// 2 half-group combine blocks. Shared between CLZ and CTZ.
    /// Each: 16 inputs (4 × has_nz + count[2:0]), 6 outputs (group_nz, idx[1:0], cnt[2:0]).
    synth_bitscan_half_blocks: Option<[crate::synth::integration::InjectedBlock; 2]>,
    /// 1 final combine block. Shared between CLZ and CTZ.
    /// 12 inputs (2 × group summary), 7 outputs (result[6:0]).
    synth_bitscan_final_block: Option<crate::synth::integration::InjectedBlock>,
    // Sprint 251: Hierarchical POPCNT blocks (pairwise adder tree).
    /// 8 byte-level POPCNT8 blocks. Each: 8 inputs (b0-b7), 4 outputs (pop[3:0]).
    synth_popcnt8_blocks: Option<[crate::synth::integration::InjectedBlock; 8]>,
    /// 4 × add(4) blocks: pairs of 4-bit byte popcounts → 5-bit sums.
    synth_popcnt_add4_blocks: Option<[crate::synth::integration::InjectedBlock; 4]>,
    /// 2 × add(5) blocks: pairs of 5-bit sums → 6-bit sums.
    synth_popcnt_add5_blocks: Option<[crate::synth::integration::InjectedBlock; 2]>,
    /// 1 × add(6) block: final 6-bit + 6-bit → 7-bit result.
    synth_popcnt_add6_block: Option<crate::synth::integration::InjectedBlock>,
    /// Sprint 250: Physical bitop computation authority.
    physical_bitop_computation: Cell<bool>,
    /// Sprint 250: Bitop dual-path verification counters.
    bitop_checks: Cell<u64>,
    bitop_mismatches: Cell<u64>,

    /// Sprint 255: Physical IR spine. When true, Super Mux outputs are physically
    /// routed through elevated z-planes to the extraction ingress. The Const-swap
    /// injection for upper-bank IR delivery is skipped; the spine delivers correct
    /// IR for ALL banks. R-override chain injection remains as safety net.
    physical_ir_spine: Cell<bool>,
    /// Sprint 255: Dual-path verification counters for spine delivery.
    ir_spine_checks: Cell<u64>,
    ir_spine_mismatches: Cell<u64>,

    /// Sprint 258: Computational via decode. When true, high-tree selector Const tiles
    /// have been replaced with WeightedViaUp tiles that extract rd/rs bits from IR
    /// via z-plane delivery. Software injection is skipped; via output is authoritative.
    physical_via_decode: Cell<bool>,
    /// Sprint 258: Tile indices for the Tree B sel2 inversion circuit (Zero + WireDown
    /// + WireLeft). These tiles need explicit dirty seeding since they're not in the
    /// original tree dirty indices.
    via_decode_inversion_dirty: Vec<usize>,
    /// Sprint 258: Dual-path verification counters for via decode.
    via_decode_checks: Cell<u64>,
    via_decode_mismatches: Cell<u64>,

    /// Sprint 260: Branch-target delivery verification.
    branch_target_checks: Cell<u64>,
    branch_target_mismatches: Cell<u64>,

    /// Sprint 259: Physical byte2 selector authority.
    physical_byte2_selector: Cell<bool>,
    byte2_selector_checks: Cell<u64>,
    byte2_selector_mismatches: Cell<u64>,
}

impl TileCpuV2 {
    pub(crate) fn from_wiring(
        origin: (usize, usize),
        idx: V2CpuIndices,
        mmio: Option<V2MmioHandle>,
    ) -> Self {
        Self {
            origin,
            pc_idx: idx.pc_idx,
            pc_next_mux_idx: idx.pc_next_mux_idx,
            pc_ingress_idx: idx.pc_ingress_idx,
            // Sprint 152: rom_low/high_mux_idx removed.
            rom_selected_low_idx: idx.rom_selected_low_idx,
            rom_selected_high_idx: idx.rom_selected_high_idx,
            rom_selected_byte2_idx: idx.rom_selected_byte2_idx,
            rom_selected_byte3_idx: idx.rom_selected_byte3_idx,
            // Sprint 151: injection indices removed.
            bank_low_mux_indices: idx.bank_low_mux_indices,
            bank_high_mux_indices: idx.bank_high_mux_indices,
            bank47_mux_indices: idx.bank47_mux_indices,
            final_mux_indices: idx.final_mux_indices,
            super_mux_inject_indices: idx.super_mux_inject_indices,
            super_mux_inject_right_indices: idx.super_mux_inject_right_indices,
            // bank_byte2/3_mux_indices removed — Sprint 185 physical ir_ext
            extract_opcode_shr_idx: idx.extract_opcode_shr_idx,
            extract_opcode_bit4_idx: idx.extract_opcode_bit4_idx,
            extract_rd_bit_indices: idx.extract_rd_bit_indices,
            extract_rs_field_idx: idx.extract_rs_field_idx,
            ctrl_a_mux_idx: idx.ctrl_a_mux_idx,
            ctrl_b_mux_idx: idx.ctrl_b_mux_idx,
            branch_ctrl_b_l1_tap_idx: idx.branch_ctrl_b_l1_tap_idx,
            branch_flag_z_l1_tap_idx: idx.branch_flag_z_l1_tap_idx,
            branch_flag_c_l1_tap_idx: idx.branch_flag_c_l1_tap_idx,
            branch_taken_core_idx: idx.branch_taken_core_idx,
            branch_dirty_indices: idx.branch_dirty_indices,
            branch_flag_dirty_indices: idx.branch_flag_dirty_indices,
            op_a_root_idx: idx.op_a_root_idx,
            op_b_root_idx: idx.op_b_root_idx,
            pipeline_dirty_indices: idx.pipeline_dirty_indices,
            pipeline_backbone_dirty_indices: idx.pipeline_backbone_dirty_indices,
            upper_bank_dirty_indices: Vec::new(), // populated by builder
            pipeline_reg_data_dirty_indices: idx.pipeline_reg_data_dirty_indices,
            reg_tap_l1_indices: idx.reg_tap_l1_indices,
            rd_onehot_decode_idx: idx.rd_onehot_decode_idx,
            rd_decode_l0_chain: idx.rd_decode_l0_chain,
            we_mask_const_idx: idx.we_mask_const_idx,
            commit_dirty_indices: idx.commit_dirty_indices,
            reg_wb_dirty_indices: idx.reg_wb_dirty_indices,
            flag_we_mask_const_idx: idx.flag_we_mask_const_idx,
            flag_commit_dirty_indices: idx.flag_commit_dirty_indices,
            _lr_idx: idx.lr_idx,
            _lr_dirty_indices: idx.lr_dirty_indices,
            reg_indices: idx.reg_indices,
            flag_z_idx: idx.flag_z_idx,
            flag_c_idx: idx.flag_c_idx,
            ram_indices: idx.ram_indices,
            regs: std::array::from_fn(|_| Cell::new(0)),
            ram: std::array::from_fn(|_| Cell::new(0)),
            pc: Cell::new(0),
            flag_z: Cell::new(false),
            flag_c: Cell::new(false),
            halted: Cell::new(false),
            lr: Cell::new(0),
            pipeline_force_full_dirty: Cell::new(true),
            compact_eval_inhibit: Cell::new(false),
            compact_eval_inhibit_count: Cell::new(0),
            compact_ops_stale: Cell::new(false),
            changed_regs_mask: Cell::new(0xFFFF),
            ram_writeback_countdown: Cell::new(2), // force writeback during initial settle
            reg_wb_countdown: Cell::new(2),        // force reg-WB during initial settle
            enable_stage_timing: false,
            enable_convergence_probe: false,
            use_prefiltered_settle: Cell::new(false),
            use_no_dirty_settle: Cell::new(false),
            latch: Cell::new(PipelineLatch::default()),
            last_stage_x_valid: Cell::new(false),
            hybrid_stage_f_bank_switches: Cell::new(0),
            hybrid_stage_f_mixed_dual_capture: Cell::new(0),
            hybrid_stage_x_mixed_software: Cell::new(0),
            hybrid_ram_high_bank_read_swaps: Cell::new(0),
            hybrid_rom_upper_bank_group_select: Cell::new(0),
            last_stage_x_mmio_reads: RefCell::new(Vec::new()),
            last_stage_x_mmio_writes: RefCell::new(Vec::new()),
            retired_count: Cell::new(0),
            mmio,
            cycle_count: Cell::new(0),
            last_stage_timing: Cell::new(V2StageTiming::default()),
            propagate_calls_total: Cell::new(0),
            propagate_tiles_total: Cell::new(0),
            cone_single_pass_checks: Cell::new(0),
            cone_residual_changes: Cell::new(0),
            compact_scan_total: Cell::new(0),
            compact_active_total: Cell::new(0),
            prop_cone_calls: Cell::new(0),
            prop_cone_evals: Cell::new(0),
            prop_settle_calls: Cell::new(0),
            prop_settle_evals: Cell::new(0),
            prop_settle_scan: Cell::new(0),
            prop_constswap_calls: Cell::new(0),
            prop_constswap_evals: Cell::new(0),
            prop_branch_calls: Cell::new(0),
            prop_branch_evals: Cell::new(0),
            prop_commit_calls: Cell::new(0),
            prop_commit_evals: Cell::new(0),
            settle_reason_counts: std::array::from_fn(|_| Cell::new(0)),
            grid_width: idx.grid_width,
            tile_count: idx.tile_count,
            pipeline_scope: idx.pipeline_scope,
            branch_scope: idx.branch_scope,
            commit_scope: idx.commit_scope,
            pipeline_scope_mask: idx.pipeline_scope_mask,
            pipeline_eval_order: idx.pipeline_eval_order,
            pipeline_compact_ops: Vec::new(), // built by builder
            pipeline_compact_wvia: Vec::new(),
            pipeline_cone_ops: Vec::new(),
            pipeline_cone_wvia: Vec::new(),
            pipeline_cone_set: Vec::new(),
            #[cfg(feature = "cranelift_jit")]
            pipeline_cone_jit: None,
            #[cfg(feature = "cranelift_jit")]
            settle_jit: None,
            #[cfg(feature = "cranelift_jit")]
            backbone_jit: None,
            backbone_jit_enabled: Cell::new(false),
            branch_compact_ops: Vec::new(),
            branch_compact_wvia: Vec::new(),
            commit_compact_ops: Vec::new(),
            commit_compact_wvia: Vec::new(),
            commit_schedule: None,
            branch_schedule: None,
            clock_schedule: None,
            clock_cascade_counts: std::cell::RefCell::new(Vec::new()),
            live_clock_ops_flags: std::cell::RefCell::new(Vec::new()),
            live_clock_wvia_flags: std::cell::RefCell::new(Vec::new()),
            live_clock_ops_noflags: std::cell::RefCell::new(Vec::new()),
            live_clock_wvia_noflags: std::cell::RefCell::new(Vec::new()),
            clock_cascade_counts_flags: std::cell::RefCell::new(Vec::new()),
            clock_cascade_counts_noflags: std::cell::RefCell::new(Vec::new()),
            clock_warmup_flags_remaining: Cell::new(0),
            clock_warmup_noflags_remaining: Cell::new(0),
            clock_auto_warmup_enabled: Cell::new(false),
            settle_jit_enabled: Cell::new(false),
            settle_frontier_offsets: Vec::new(),
            settle_frontier_targets: Vec::new(),
            jit_settle_eval_ns: Cell::new(0),
            jit_settle_dirty_ns: Cell::new(0),
            jit_settle_passes: Cell::new(0),
            jit_settle_changed: Cell::new(0),
            jit_settle_profiled: Cell::new(false),
            jit_settle_pass1_changed: Cell::new(0),
            jit_settle_pass2_changed: Cell::new(0),
            settle_switch_counts: std::cell::RefCell::new(Vec::new()),
            settle_overlap_buckets: std::array::from_fn(|_| Cell::new(0)),
            commit_drain_ns: Cell::new(0),
            commit_worklist_ns: Cell::new(0),
            commit_profiled: Cell::new(false),
            stage_f_cone_ns: Cell::new(0),
            stage_f_settle_ns: Cell::new(0),
            stage_f_inject_ns: Cell::new(0),
            stage_f_profiled: Cell::new(false),
            pipeline_schedule: None,
            settle_compact_ops: Vec::new(),
            settle_compact_wvia: Vec::new(),
            settle_cone_set: Vec::new(),
            trunk_settle_ops: Vec::new(),
            trunk_settle_wvia: Vec::new(),
            settle_idx_to_slot: Vec::new(),
            settle_wvia_slot_map: Vec::new(),
            settle_idx_to_slot_constswap: Vec::new(),
            settle_wvia_slot_map_constswap: Vec::new(),
            settle_block_seg_offsets: Vec::new(),
            settle_block_seg_entries: Vec::new(),
            settle_block_wvia_counts: Vec::new(),
            settle_block_seg_offsets_cs: Vec::new(),
            settle_block_seg_entries_cs: Vec::new(),
            settle_block_wvia_counts_cs: Vec::new(),
            settle_schedule: None,
            settle_forward_deps_data: Vec::new(),
            settle_forward_deps_offsets: Vec::new(),
            settle_compact_ops_constswap: Vec::new(),
            settle_compact_wvia_constswap: Vec::new(),
            settle_backbone_schedule: None,
            settle_fringe_ops: Vec::new(),
            settle_fringe_wvia: Vec::new(),
            settle_backbone_schedule_constswap: None,
            settle_fringe_ops_constswap: Vec::new(),
            settle_fringe_wvia_constswap: Vec::new(),
            backbone_ops: Vec::new(),
            backbone_wvia: Vec::new(),
            backbone_cone_set: Vec::new(),
            hybrid_settle_enabled: Cell::new(false),
            backbone_input_indices: Vec::new(),
            backbone_output_indices: Vec::new(),
            backbone_cache: RefCell::new(BackboneCache::new(256)),
            memoization_enabled: Cell::new(false),
            memo_hits: Cell::new(0),
            memo_misses: Cell::new(0),
            decode_input_indices: Vec::new(),
            decode_output_indices: Vec::new(),
            execute_compact_ops: Vec::new(),
            execute_compact_wvia: Vec::new(),
            execute_cone_set: Vec::new(),
            decode_cache: RefCell::new(BackboneCache::new(256)),
            decode_memoization_enabled: Cell::new(false),
            decode_memo_hits: Cell::new(0),
            decode_memo_misses: Cell::new(0),
            adaptive_decode_enabled: Cell::new(false),
            adaptive_decode_threshold: Cell::new(0.5),
            adaptive_warmup_calls: Cell::new(16),
            adaptive_decode_mode: Cell::new(0),
            adaptive_decode_skipped: Cell::new(0),
            adaptive_decode_window: Cell::new(0),
            adaptive_probe_interval: Cell::new(0),
            adaptive_decode_history: RefCell::new(std::collections::VecDeque::new()),
            adaptive_probe_counter: Cell::new(0),
            adaptive_decode_probes: Cell::new(0),
            branch_eval_order: idx.branch_eval_order,
            commit_eval_order: idx.commit_eval_order,
            branch_scope_mask: idx.branch_scope_mask,
            commit_scope_mask: idx.commit_scope_mask,
            clock_scope_mask: idx.clock_scope_mask,
            clock_compact_ops: Vec::new(),
            clock_compact_wvia: Vec::new(),
            in_scope_clock_cache: idx.in_scope_clock_cache.clone(),
            in_scope_clock_cache_no_flags: idx.in_scope_clock_cache,
            // Sprint 196: synth branch replacement
            synth_branch_table: crate::synth::integration::branch_taken_truth_table(),
            synth_branch: SynthGate::new(),
            branch_taken_saved_tile_type: Cell::new(TileType::Wire),
            synth_branch_block: None,
            synth_rd_decode: SynthGate::new(),
            rd_decode_saved_tile_type: Cell::new(TileType::Wire),
            synth_rd_decode_block: None,
            synth_ctrl_b: SynthGate::new(),
            synth_ctrl_a: SynthGate::new(),
            we_mask_saved_tile_type: Cell::new(TileType::Wire),
            flag_we_mask_saved_tile_type: Cell::new(TileType::Wire),
            synth_operand: SynthGate::new(),
            physical_operand_authority: Cell::new(false),
            op_authority_checks: Cell::new(0),
            op_authority_mismatches: Cell::new(0),
            physical_rd_decode: Cell::new(false),
            rd_decode_authority_checks: Cell::new(0),
            rd_decode_authority_mismatches: Cell::new(0),
            physical_we_mask: Cell::new(false),
            we_mask_authority_checks: Cell::new(0),
            we_mask_authority_mismatches: Cell::new(0),
            physical_flag_we_mask: Cell::new(false),
            flag_we_mask_authority_checks: Cell::new(0),
            flag_we_mask_authority_mismatches: Cell::new(0),
            physical_super_mux: Cell::new(false),
            super_mux_checks: Cell::new(0),
            super_mux_mismatches: Cell::new(0),
            upper_bank_ir_checks: Cell::new(0),
            upper_bank_ir_mismatches: Cell::new(0),
            physical_wb_data_authority: Cell::new(false),
            wb_data_checks: Cell::new(0),
            wb_data_mismatches: Cell::new(0),
            alu_add_l_enable_idx: idx.alu_add_l_enable_idx,
            alu_add_l_mux_idx: idx.alu_add_l_mux_idx,
            physical_high_reg_writeback: Cell::new(false),
            high_reg_wb_checks: Cell::new(0),
            high_reg_wb_mismatches: Cell::new(0),
            physical_sub_fn_delivery: Cell::new(false),
            sub_fn_delivery_checks: Cell::new(0),
            sub_fn_delivery_mismatches: Cell::new(0),
            physical_sub_fn_flag_authority: Cell::new(false),
            physical_load_delivery: Cell::new(false),
            load_delivery_checks: Cell::new(0),
            load_delivery_mismatches: Cell::new(0),
            high_reg_wb_data_const_indices: idx.high_reg_wb_data_const_indices,
            high_reg_we_const_indices: idx.high_reg_we_const_indices,
            high_reg_merge_mux_indices: idx.high_reg_merge_mux_indices,
            high_tree_a_sel_const_indices: idx.high_tree_a_sel_const_indices,
            high_tree_b_sel_const_indices: idx.high_tree_b_sel_const_indices,
            high_tree_a_data_const_indices: idx.high_tree_a_data_const_indices,
            high_tree_b_data_const_indices: idx.high_tree_b_data_const_indices,
            top_mux_a_sel_const_idx: idx.top_mux_a_sel_const_idx,
            top_mux_b_sel_const_idx: idx.top_mux_b_sel_const_idx,
            high_tree_a_dirty_indices: idx.high_tree_a_dirty_indices,
            high_tree_b_dirty_indices: idx.high_tree_b_dirty_indices,
            top_mux_a_dirty_indices: idx.top_mux_a_dirty_indices,
            top_mux_b_dirty_indices: idx.top_mux_b_dirty_indices,
            alu_a_trunk_terminal_idx: idx.alu_a_trunk_terminal_idx,
            alu_b_trunk_terminal_idx: idx.alu_b_trunk_terminal_idx,
            alu_a_downstream_dirty: idx.alu_a_downstream_dirty,
            alu_b_downstream_dirty: idx.alu_b_downstream_dirty,
            alu_r_mux_output_indices: idx.alu_r_mux_output_indices,
            upper_alu_trunk_checks: Cell::new(0),
            upper_alu_trunk_mismatches: Cell::new(0),
            synth_ram_decode: SynthGate::new(),
            ram_write_decode_idx: idx.ram_write_decode_idx,
            ram_write_gate_idx: idx.ram_write_gate_idx,
            ram_decode_saved_tile_type: Cell::new(TileType::Wire),
            synth_ram_decode_block: None,
            synth_ctrl_b_block: None,
            synth_ctrl_a_block: None,
            combined_decode_lut: None,
            combined_decode_checks: Cell::new(0),

            // Sprint 218: ALU readback dual-path verification
            synth_alu: SynthGate::new(),
            alu_opcode_mismatches: std::array::from_fn(|_| Cell::new(0)),
            wb_alu_root_idx: idx.wb_alu_root_idx,
            wb_data_mux_idx: idx.wb_data_mux_idx,
            reg_capture_checks: Cell::new(0),
            reg_capture_mismatches: Cell::new(0),
            alu_mux_select_mismatches: Cell::new(0),

            // Sprint 219: no opcodes physically authoritative by default
            physical_alu_opcodes: Cell::new(0),

            // Sprint 220: physical register writeback off by default
            physical_reg_writeback: Cell::new(false),
            reg_wb_checks: Cell::new(0),
            reg_wb_mismatches: Cell::new(0),

            // Sprint 221: physical flag writeback off by default
            physical_flag_writeback: Cell::new(false),
            flag_wb_checks: Cell::new(0),
            flag_wb_mismatches: Cell::new(0),
            flag_z_mismatches: Cell::new(0),
            flag_c_mismatches: Cell::new(0),

            // Sprint 222: physical decode off by default
            physical_decode: Cell::new(false),
            decode_ctrl_b_checks: Cell::new(0),
            decode_ctrl_b_mismatches: Cell::new(0),
            decode_ctrl_a_checks: Cell::new(0),
            decode_ctrl_a_mismatches: Cell::new(0),

            // Sprint 223/224: PC override counters
            pc_override_checks: Cell::new(0),
            pc_override_mismatches: Cell::new(0),
            pc_mismatch_per_kind: std::array::from_fn(|_| Cell::new(0)),

            // Sprint 225: Physical branch direction authority
            physical_branch: Cell::new(false),
            branch_dir_checks: Cell::new(0),
            branch_dir_mismatches: Cell::new(0),

            // Sprint 229: Physical RAM writeback authority
            physical_ram_writeback: Cell::new(false),
            ram_wb_checks: Cell::new(0),
            ram_wb_mismatches: Cell::new(0),
            ram_wb_store_mismatches: Cell::new(0),
            ram_wb_nonstore_mismatches: Cell::new(0),
            physical_ram_store_authority: Cell::new(false),
            main_mem: Vec::new(),
            mem_addr_mask: Cell::new(0x7F),
            extended_pc: false,
            pc_phys_mask: 0x7F,
            pc_addr_bits: 7,
            wide_pc: false,
            program_ext: Vec::new(),
            ram_store_saved_via: Cell::new(None),
            ram_store_inject_up_idx: Cell::new(0),
            ram_snap_post_commit: std::array::from_fn(|_| Cell::new(0)),
            ram_snap_post_reinject: std::array::from_fn(|_| Cell::new(0)),
            ram_snap_post_clock: std::array::from_fn(|_| Cell::new(0)),
            ram_snap_store_addr: Cell::new(0),

            // Sprint 248: SRA synth computation
            synth_sra_block: None,
            physical_sra_computation: Cell::new(false),
            synth_mul_block: None,
            physical_mul_authority: Cell::new(false),
            sra_computation_checks: Cell::new(0),
            sra_computation_mismatches: Cell::new(0),

            // Sprint 250: Hierarchical CLZ/CTZ synth blocks
            synth_bitscan8_blocks: None,
            synth_bitscan_half_blocks: None,
            synth_bitscan_final_block: None,

            // Sprint 251: Hierarchical POPCNT synth blocks
            synth_popcnt8_blocks: None,
            synth_popcnt_add4_blocks: None,
            synth_popcnt_add5_blocks: None,
            synth_popcnt_add6_block: None,

            physical_bitop_computation: Cell::new(false),
            bitop_checks: Cell::new(0),
            bitop_mismatches: Cell::new(0),

            physical_ir_spine: Cell::new(false),
            ir_spine_checks: Cell::new(0),
            ir_spine_mismatches: Cell::new(0),

            physical_via_decode: Cell::new(false),
            via_decode_inversion_dirty: Vec::new(),
            via_decode_checks: Cell::new(0),
            via_decode_mismatches: Cell::new(0),

            branch_target_checks: Cell::new(0),
            branch_target_mismatches: Cell::new(0),

            physical_byte2_selector: Cell::new(false),
            byte2_selector_checks: Cell::new(0),
            byte2_selector_mismatches: Cell::new(0),
        }
    }

    /// Resolve the active ROM fetch lanes for the committed PC.
    ///
    /// Lower-bank fetches read from the physical Final Mux outputs. Upper-bank fetches
    /// still source directly from the bank4-7 Mux16to1 tiles until the Super Mux path is
    /// fully routed back through the fetch/decode westback chain.
    fn read_active_rom_lanes(&self, sim: &Simulation) -> [u64; 4] {
        let pc = (sim.get_logic_value_by_idx(self.pc_idx) as u32) & self.pc_phys_mask;
        // Sprint 369 (Gate B.2): instructions at PC>=128 live in the software-backed
        // upper-half store, not physical ROM. Decompose the word into the 4 fetch
        // lanes (low/high/byte2/byte3) directly.
        if self.extended_pc && pc as usize >= 128 {
            let w = self
                .program_ext
                .get(pc as usize - 128)
                .copied()
                .unwrap_or(0);
            return [
                (w & 0xFF) as u64,
                ((w >> 8) & 0xFF) as u64,
                ((w >> 16) & 0xFF) as u64,
                ((w >> 24) & 0xFF) as u64,
            ];
        }
        let bank_group = (pc >> 6) & 1;
        let bank_within = ((pc >> 4) & 3) as usize;
        let mut lanes = [0u64; 4];
        for (lane, out) in lanes.iter_mut().enumerate() {
            *out = if bank_group == 0 {
                sim.get_logic_value_by_idx(self.final_mux_indices[lane])
            } else {
                sim.get_logic_value_by_idx(self.bank47_mux_indices[bank_within][lane])
            };
        }
        lanes
    }

    /// Sprint 132/143: Read the 16-bit extension word (bits 31-16 of the 32-bit instruction).
    /// This must work both after a live fetch and on a freshly built CPU before Stage F has
    /// populated the Sprint 211 Super Mux outputs.
    pub fn read_ir_ext(&self, sim: &Simulation) -> u16 {
        let lanes = self.read_active_rom_lanes(sim);
        let byte2 = lanes[2] as u8;
        let byte3 = lanes[3] as u8;
        (byte3 as u16) << 8 | byte2 as u16
    }

    fn opcode_uses_rs(opcode: u8) -> bool {
        matches!(
            opcode,
            0x02 | // MOV
            0x04 | 0x05 | 0x06 | 0x07 | 0x08 | // ADD/SUB/AND/OR/XOR
            0x0F | // CMP
            0x16 | // LD
            0x17 // ST
        )
    }

    /// Sprint 199: Unified memory address computation for LD/LDB/ST/STB.
    /// Covers offset addressing (EXT_OFFSET), byte-variant direct register reads,
    /// and immediate/register base addressing. Eliminates 4× duplication.
    fn compute_mem_addr(&self, latch: &PipelineLatch, opcode: u8, b: u64) -> usize {
        let has_offset = (latch.ir_ext & EXT_OFFSET) != 0;
        // LDB=0x18 and STB=0x19 read base register directly (opcode_uses_rs = false).
        // LD=0x16 and ST=0x17 use b = regs[rs] (opcode_uses_rs = true).
        let is_byte_variant = opcode == 0x18 || opcode == 0x19;
        if has_offset {
            let offset = ((latch.ir_ext >> 8) & 0xFF) as u64;
            let base_val = if is_byte_variant {
                let rs_lo = ((latch.ir_low >> 5) & 0x07) as u8;
                let rs_hi = (latch.ir_ext & EXT_RS_HI) != 0;
                self.regs[effective_reg(rs_lo, rs_hi) as usize].get()
            } else {
                b
            };
            (base_val.wrapping_add(offset) as usize) & self.mem_addr_mask.get()
        } else if is_byte_variant {
            // LDB/STB without offset: immediate address
            (latch.ir_low as usize) & self.mem_addr_mask.get()
        } else {
            // LD/ST without offset: register address
            (b as usize) & self.mem_addr_mask.get()
        }
    }

    // Sprint 151: inject_selector_outputs removed — physical westback routes
    // deliver ir_low/ir_high through tile propagation.

    fn propagate_pipeline_until_settled(&self, sim: &mut Simulation, stats: &mut StageStats) {
        // Sprint 356: Decode-only memoization. Hash the decode externals
        // (PC, IR, decoder Consts — not register state). On hit, restore
        // the cached decode tile values and run the hybrid kernel only on
        // the execute portion (operand muxes + register-tainted paths).
        // Hit-rate ceiling = #(distinct PC,IR states) ≈ one entry per
        // static instruction in any loop. On miss, run full hybrid +
        // snapshot decode tiles + insert. Drain dirty in settle_cone_set
        // on both paths.
        if self.decode_memoization_enabled.get()
            && !self.decode_input_indices.is_empty()
            && !self.execute_compact_ops.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
            && self.adaptive_decode_should_use_cache()
        {
            let mut input_buf: Vec<u64> = Vec::with_capacity(self.decode_input_indices.len());
            sim.read_tiles_into(&self.decode_input_indices, &mut input_buf);
            let key = hash_u64_slice(&input_buf);

            let mut hit_outputs: Option<Vec<u64>> = None;
            {
                let cache = self.decode_cache.borrow();
                if let Some(entry) = cache.get(key) {
                    if entry.inputs == input_buf {
                        hit_outputs = Some(entry.outputs.clone());
                    }
                }
            }

            if let Some(snapshot) = hit_outputs {
                // HIT: restore cached decode tile values directly. By the
                // closure invariant (decode tile = pure function of decode
                // externals), the snapshot equals what the hybrid kernel
                // would compute on the decode subset. Raw atomic store is
                // faster than compare-and-skip (apply_backbone_snapshot)
                // because most decode tiles change value across cycles,
                // making the per-tile load unfruitful.
                use std::sync::atomic::Ordering;
                debug_assert_eq!(snapshot.len(), self.decode_output_indices.len());
                for (i, &idx32) in self.decode_output_indices.iter().enumerate() {
                    sim.tilemap.tiles[idx32 as usize]
                        .logic
                        .store(snapshot[i], Ordering::Relaxed);
                }
                // Run hybrid kernel ONLY on execute portion. Treat all
                // execute ops as backbone (eval unconditionally) since
                // upstream register-state taint may have changed.
                let (d, e, s) = sim.propagate_compact_dirty_hybrid(
                    &self.execute_compact_ops,
                    &self.execute_compact_wvia,
                    &self.execute_cone_set,
                );
                {
                    let mut drain_buf = Vec::new();
                    sim.dirty
                        .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
                }
                stats.comb_deltas += d;
                stats.comb_eval += e;
                stats.comb_switched += s;
                self.propagate_calls_total
                    .set(self.propagate_calls_total.get() + 1);
                self.propagate_tiles_total
                    .set(self.propagate_tiles_total.get() + e as u64);
                let scan = self.execute_compact_ops.len() as u64;
                self.compact_scan_total
                    .set(self.compact_scan_total.get() + scan);
                self.compact_active_total
                    .set(self.compact_active_total.get() + e as u64);
                self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
                self.prop_settle_evals
                    .set(self.prop_settle_evals.get() + e as u64);
                self.prop_settle_scan
                    .set(self.prop_settle_scan.get() + scan);
                self.decode_memo_hits.set(self.decode_memo_hits.get() + 1);
                return;
            }

            // MISS (or hash collision / first-time key): full hybrid +
            // snapshot decode tile values + insert.
            let (d, e, s) = sim.propagate_compact_dirty_hybrid(
                &self.settle_compact_ops,
                &self.settle_compact_wvia,
                &self.backbone_cone_set,
            );
            {
                let mut drain_buf = Vec::new();
                sim.dirty
                    .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
            }
            let snap = sim.snapshot_tiles(&self.decode_output_indices);
            self.decode_cache.borrow_mut().insert(key, input_buf, snap);
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            let scan = self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
            self.decode_memo_misses
                .set(self.decode_memo_misses.get() + 1);
            return;
        }

        // Sprint 355: Settle memoization. Hash settle-seed input vector each
        // cycle; on cache hit, restore the full settle output snapshot and
        // skip kernel eval entirely. On miss, run the hybrid kernel for full
        // settle and snapshot all outputs into the cache. Drain dirty bits
        // in the settle scope on both paths.
        if self.memoization_enabled.get()
            && !self.backbone_input_indices.is_empty()
            && !self.backbone_output_indices.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            let mut input_buf: Vec<u64> = Vec::with_capacity(self.backbone_input_indices.len());
            sim.read_tiles_into(&self.backbone_input_indices, &mut input_buf);
            let key = hash_u64_slice(&input_buf);

            // Cache lookup with input verification (collision-safe).
            let mut hit_outputs: Option<Vec<u64>> = None;
            {
                let cache = self.backbone_cache.borrow();
                if let Some(entry) = cache.get(key) {
                    if entry.inputs == input_buf {
                        hit_outputs = Some(entry.outputs.clone());
                    }
                }
            }

            if let Some(snapshot) = hit_outputs {
                // HIT: directly write cached output values to all settle
                // tiles. No kernel eval needed — by determinism, the cached
                // outputs equal what hybrid_eval(seeds=input_buf) would
                // produce.
                use std::sync::atomic::Ordering;
                debug_assert_eq!(snapshot.len(), self.backbone_output_indices.len());
                for (i, &idx32) in self.backbone_output_indices.iter().enumerate() {
                    sim.tilemap.tiles[idx32 as usize]
                        .logic
                        .store(snapshot[i], Ordering::Relaxed);
                }
                {
                    let mut drain_buf = Vec::new();
                    sim.dirty
                        .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
                }
                self.propagate_calls_total
                    .set(self.propagate_calls_total.get() + 1);
                self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
                self.memo_hits.set(self.memo_hits.get() + 1);
                return;
            }

            // MISS (or hash collision): run hybrid kernel for full settle,
            // snapshot all outputs into the cache.
            let (d, e, s) = sim.propagate_compact_dirty_hybrid(
                &self.settle_compact_ops,
                &self.settle_compact_wvia,
                &self.backbone_cone_set,
            );
            {
                let mut drain_buf = Vec::new();
                sim.dirty
                    .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
            }
            let snap = sim.snapshot_tiles(&self.backbone_output_indices);
            self.backbone_cache
                .borrow_mut()
                .insert(key, input_buf, snap);
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            let scan = self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
            self.memo_misses.set(self.memo_misses.get() + 1);
            return;
        }

        // Sprint 354: Single-pass hybrid settle. One walk over settle_compact_ops;
        // backbone ops evaluate unconditionally, fringe ops are dirty-checked.
        // Same op count as the blockskip baseline but removes the dirty-check
        // overhead for the 93.7% backbone majority.
        if self.hybrid_settle_enabled.get()
            && !self.settle_compact_ops.is_empty()
            && !self.backbone_cone_set.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            let (d, e, s) = sim.propagate_compact_dirty_hybrid(
                &self.settle_compact_ops,
                &self.settle_compact_wvia,
                &self.backbone_cone_set,
            );
            // Drain residual dirty bits in settle scope (mirrors S312 path).
            {
                let mut drain_buf = Vec::new();
                sim.dirty
                    .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
            }
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            let scan = self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
            return;
        }
        // Sprint 352: Backbone + fringe split settle — evaluate backbone (~93.7%) via
        // no-dirty unconditional eval, then evaluate fringe (~6.3%) via dirty-driven
        // compact eval. With cranelift_jit, backbone uses JIT native code.
        if self.backbone_jit_enabled.get()
            && !self.backbone_ops.is_empty()
            && !self.settle_fringe_ops.is_empty()
            && !self.backbone_cone_set.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            // Phase 1: Backbone — evaluate all backbone ops unconditionally.
            // Uses backbone_cone_set as scope_mask so only fringe/external tiles
            // get marked dirty by changed backbone outputs.
            #[allow(unused_mut)]
            let (mut d1, mut e1, mut s1) = (0u32, 0u32, 0u32);
            #[cfg(feature = "cranelift_jit")]
            let used_jit = if let Some(ref jit) = self.backbone_jit {
                let r = sim.propagate_jit_settle(
                    jit,
                    &self.backbone_cone_set,
                    &[], // idx_to_slot unused with empty frontier
                    &[], // frontier_offsets empty → fallback to dirty_dependents_frontier
                    &[], // frontier_targets
                );
                d1 = r.0;
                e1 = r.1;
                s1 = r.2;
                true
            } else {
                false
            };
            #[cfg(not(feature = "cranelift_jit"))]
            let used_jit = false;
            if !used_jit {
                let r = sim.propagate_cone_no_dirty(
                    &self.backbone_ops,
                    &self.backbone_wvia,
                    &self.backbone_cone_set,
                );
                d1 = r.0;
                e1 = r.1;
                s1 = r.2;
            }

            // Phase 2: Dirty eval on full settle scope.
            // Backbone no-dirty already settled ~93.7% unconditionally. The remaining
            // dirty tiles (fringe inputs, backbone frontier) are caught by this pass.
            // Using full settle_compact_ops (not just fringe) ensures cross-boundary
            // cascades between backbone and fringe are handled correctly.
            let (d2, e2, s2) =
                sim.propagate_compact_dirty(&self.settle_compact_ops, &self.settle_compact_wvia);

            // Drain residual dirty bits in settle scope (same as S312 no-dirty path).
            {
                let mut drain_buf = Vec::new();
                sim.dirty
                    .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
            }

            let total_d = d1 + d2;
            let total_e = e1 + e2;
            let total_s = s1 + s2;
            stats.comb_deltas += total_d;
            stats.comb_eval += total_e;
            stats.comb_switched += total_s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + total_e as u64);
            // Scan accounting: Phase 1 scanned backbone_ops unconditionally,
            // Phase 2 scanned full settle_compact_ops for dirty check.
            let scan = self.backbone_ops.len() as u64 + self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + total_e as u64);
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + total_e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
            return;
        }
        // Sprint 312: No-dirty settle — evaluate all settle ops unconditionally.
        // At 35% activity (S311), no-dirty should beat compact_dirty's scan overhead.
        if self.use_no_dirty_settle.get()
            && !self.settle_compact_ops.is_empty()
            && !self.settle_cone_set.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            let (d, e, s) = sim.propagate_cone_no_dirty(
                &self.settle_compact_ops,
                &self.settle_compact_wvia,
                &self.settle_cone_set,
            );
            // Sprint 312: Drain residual dirty bits in settle scope.
            // propagate_cone_no_dirty doesn't clear dirty bits — they'd leak
            // into branch/commit/clock and corrupt state (S291 finding).
            {
                let mut drain_buf = Vec::new();
                sim.dirty
                    .fill_into_masked(&self.settle_cone_set, &mut drain_buf);
                // drain_buf is discarded — bits are cleared by fill_into_masked.
            }
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            let scan = self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
            return;
        }
        // Sprint 306: Prefiltered settle — uses fill_into_masked + slot bitset to
        // evaluate only dirty tiles with same-pass forward propagation.
        // Sprint 307: Gated by use_prefiltered_settle flag for A/B profiling.
        if self.use_prefiltered_settle.get()
            && !self.settle_idx_to_slot.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            let (d, e, s) = sim.propagate_compact_dirty_prefiltered(
                &self.settle_compact_ops,
                &self.settle_compact_wvia,
                &self.settle_cone_set,
                &self.settle_idx_to_slot,
                &self.settle_wvia_slot_map,
            );
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            // Sprint 312: record ops.len() for scan, not e (which is active count).
            let pf_scan = self.settle_compact_ops.len() as u64;
            self.compact_scan_total
                .set(self.compact_scan_total.get() + pf_scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            // Sprint 311: per-path settle (prefiltered).
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + pf_scan);
            return;
        }
        // Sprint 335: JIT-compiled settle evaluation with Rust convergence loop.
        // Sprint 336: Gated by explicit settle_jit_enabled (not clock auto-warmup).
        #[cfg(feature = "cranelift_jit")]
        if let Some(ref jit) = self.settle_jit {
            if !self.compact_eval_inhibit.get()
                && !self.compact_ops_stale.get()
                && self.settle_jit_enabled.get()
            {
                let (d, e, s) = if self.jit_settle_profiled.get() {
                    let (d, e, s, eval_ns, dirty_ns, passes, changed, p1c, p2c) = sim
                        .propagate_jit_settle_profiled(
                            jit,
                            &self.settle_idx_to_slot,
                            &self.settle_frontier_offsets,
                            &self.settle_frontier_targets,
                        );
                    self.jit_settle_eval_ns
                        .set(self.jit_settle_eval_ns.get() + eval_ns);
                    self.jit_settle_dirty_ns
                        .set(self.jit_settle_dirty_ns.get() + dirty_ns);
                    self.jit_settle_passes
                        .set(self.jit_settle_passes.get() + passes as u64);
                    self.jit_settle_changed
                        .set(self.jit_settle_changed.get() + changed as u64);
                    self.jit_settle_pass1_changed
                        .set(self.jit_settle_pass1_changed.get() + p1c as u64);
                    self.jit_settle_pass2_changed
                        .set(self.jit_settle_pass2_changed.get() + p2c as u64);
                    (d, e, s)
                } else {
                    sim.propagate_jit_settle(
                        jit,
                        &self.settle_cone_set,
                        &self.settle_idx_to_slot,
                        &self.settle_frontier_offsets,
                        &self.settle_frontier_targets,
                    )
                };
                stats.comb_deltas += d;
                stats.comb_eval += e;
                stats.comb_switched += s;
                self.propagate_calls_total
                    .set(self.propagate_calls_total.get() + 1);
                self.propagate_tiles_total
                    .set(self.propagate_tiles_total.get() + e as u64);
                self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
                self.prop_settle_evals
                    .set(self.prop_settle_evals.get() + e as u64);
                self.prop_settle_scan
                    .set(self.prop_settle_scan.get() + self.settle_compact_ops.len() as u64);
                return;
            }
        }
        // Sprint 308: Block-level clean-skip on sequential scan.
        if !self.settle_block_seg_offsets.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            // Sprint 328/330: Use counted/bucketed variant when counting is active.
            let counting = !self.settle_switch_counts.borrow().is_empty();
            let (d, e, s) = if counting {
                let mut buckets = [0u64; 5];
                let r = sim.propagate_compact_dirty_blockskip_bucketed(
                    &self.settle_compact_ops,
                    &self.settle_compact_wvia,
                    &self.settle_block_seg_offsets,
                    &self.settle_block_seg_entries,
                    &self.settle_block_wvia_counts,
                    &mut self.settle_switch_counts.borrow_mut(),
                    &mut buckets,
                );
                // Accumulate into persistent buckets.
                for i in 0..5 {
                    self.settle_overlap_buckets[i]
                        .set(self.settle_overlap_buckets[i].get() + buckets[i]);
                }
                r
            } else {
                sim.propagate_compact_dirty_blockskip(
                    &self.settle_compact_ops,
                    &self.settle_compact_wvia,
                    &self.settle_block_seg_offsets,
                    &self.settle_block_seg_entries,
                    &self.settle_block_wvia_counts,
                )
            };
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            self.compact_scan_total
                .set(self.compact_scan_total.get() + e as u64);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            // Sprint 311: per-path settle (blockskip).
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + self.settle_compact_ops.len() as u64);
            return;
        }
        self.propagate_pipeline_impl(sim, stats, false);
    }

    /// Sprint 304: Constswap settle — R-Mux output tiles as COP_CONST.
    /// Sprint 306: Prefiltered when lookup tables available.
    /// Sprint 307: Gated by use_prefiltered_settle (disabled by default — 3x slower).
    fn propagate_pipeline_constswap_settled(&self, sim: &mut Simulation, stats: &mut StageStats) {
        let (d, e, s) = if self.use_prefiltered_settle.get()
            && !self.settle_idx_to_slot_constswap.is_empty()
            && !self.compact_ops_stale.get()
        {
            sim.propagate_compact_dirty_prefiltered(
                &self.settle_compact_ops_constswap,
                &self.settle_compact_wvia_constswap,
                &self.settle_cone_set,
                &self.settle_idx_to_slot_constswap,
                &self.settle_wvia_slot_map_constswap,
            )
        } else if !self.settle_block_seg_offsets_cs.is_empty() && !self.compact_ops_stale.get() {
            sim.propagate_compact_dirty_blockskip(
                &self.settle_compact_ops_constswap,
                &self.settle_compact_wvia_constswap,
                &self.settle_block_seg_offsets_cs,
                &self.settle_block_seg_entries_cs,
                &self.settle_block_wvia_counts_cs,
            )
        } else if !self.settle_compact_ops_constswap.is_empty() && !self.compact_ops_stale.get() {
            sim.propagate_compact_dirty(
                &self.settle_compact_ops_constswap,
                &self.settle_compact_wvia_constswap,
            )
        } else {
            sim.propagate_levelized(&self.pipeline_eval_order)
        };
        stats.comb_deltas += d;
        stats.comb_eval += e;
        stats.comb_switched += s;
        self.propagate_calls_total
            .set(self.propagate_calls_total.get() + 1);
        self.propagate_tiles_total
            .set(self.propagate_tiles_total.get() + e as u64);
        let scan = self.settle_compact_ops_constswap.len() as u64;
        self.compact_scan_total
            .set(self.compact_scan_total.get() + scan);
        self.compact_active_total
            .set(self.compact_active_total.get() + e as u64);
        // Sprint 311: per-path constswap.
        self.prop_constswap_calls
            .set(self.prop_constswap_calls.get() + 1);
        self.prop_constswap_evals
            .set(self.prop_constswap_evals.get() + e as u64);
    }

    /// Sprint 272: Cone-pruned pipeline propagation for Stage F.
    /// Sprint 274: After cone eval, check if a second pass would change anything.
    /// Only evaluates tiles in the output cone (ROM lanes + operands + control).
    fn propagate_pipeline_cone(&self, sim: &mut Simulation, stats: &mut StageStats) {
        self.propagate_pipeline_impl(sim, stats, true);

        // Sprint 274 C1: Convergence probe — re-evaluate all cone ops and count
        // how many would produce a different value. If zero, single-pass is proven.
        // Sprint 288: Decoupled from enable_stage_timing (was inflating Stage F
        // measurements by ~1,223-op shadow pass). Now gated by separate flag.
        if self.enable_convergence_probe
            && !self.pipeline_cone_ops.is_empty()
            && !self.compact_ops_stale.get()
        {
            let residual = sim.count_cone_residual_changes(&self.pipeline_cone_ops);
            self.cone_single_pass_checks
                .set(self.cone_single_pass_checks.get() + 1);
            self.cone_residual_changes
                .set(self.cone_residual_changes.get() + residual as u64);
        }
    }

    fn propagate_pipeline_impl(
        &self,
        sim: &mut Simulation,
        stats: &mut StageStats,
        use_cone: bool,
    ) {
        // Sprint 304: With constswap settle ops, compact_eval_inhibit should never
        // be set during propagate_pipeline_impl in max_authority mode.
        // The legacy upper-bank path (only caller) is dead code when physical_ir_spine=true.
        debug_assert!(
            !self.compact_eval_inhibit.get() || !self.physical_ir_spine.get(),
            "Sprint 304: compact_eval_inhibit set while physical_ir_spine is active"
        );
        // Sprint 273: JIT-compiled cone evaluation when available.
        #[cfg(feature = "cranelift_jit")]
        let jit_result = if use_cone
            && self.pipeline_cone_jit.is_some()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            Some(sim.propagate_jit_cone(self.pipeline_cone_jit.as_ref().unwrap()))
        } else {
            None
        };
        #[cfg(not(feature = "cranelift_jit"))]
        let jit_result: Option<(u32, u32, u32)> = None;

        // Sprint 274: No-dirty cone eval — single pass, frontier-only dirty marks.
        let (d, e, s) = if let Some(r) = jit_result {
            r
        } else if use_cone
            && !self.pipeline_cone_ops.is_empty()
            && !self.pipeline_cone_set.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            sim.propagate_cone_no_dirty(
                &self.pipeline_cone_ops,
                &self.pipeline_cone_wvia,
                &self.pipeline_cone_set,
            )
        // Sprint 290: Settle-scope compact_dirty — forward closure from injection seeds.
        // Sprint 291: No-dirty tested — correct but 2% activity makes it slower.
        // Sprint 292: Scheduler tested — dep table incomplete for max_authority.
        // Sprint 293: Forward-deps tested — CompactOp-input-derived dep filter
        // doesn't capture via_fwd, chain exit, or tile-type write-side effects.
        // Current dep abstraction is incomplete, not that single-pass is impossible.
        // A semantic propagation graph with first-class write-side edges could work.
        } else if !use_cone
            && !self.settle_compact_ops.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            sim.propagate_compact_dirty(&self.settle_compact_ops, &self.settle_compact_wvia)
        } else if !self.pipeline_compact_ops.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            sim.propagate_compact_dirty(&self.pipeline_compact_ops, &self.pipeline_compact_wvia)
        } else {
            sim.propagate_levelized(&self.pipeline_eval_order)
        };
        stats.comb_deltas += d;
        stats.comb_eval += e;
        stats.comb_switched += s;
        self.propagate_calls_total
            .set(self.propagate_calls_total.get() + 1);
        self.propagate_tiles_total
            .set(self.propagate_tiles_total.get() + e as u64);
        // Sprint 290: Derive scan from dispatch path actually taken.
        let scan = if !use_cone
            && !self.settle_compact_ops.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            self.settle_compact_ops.len() as u64
        } else if !self.pipeline_compact_ops.is_empty()
            && !self.compact_eval_inhibit.get()
            && !self.compact_ops_stale.get()
        {
            self.pipeline_compact_ops.len() as u64
        } else {
            self.pipeline_eval_order.len() as u64
        };
        self.compact_scan_total
            .set(self.compact_scan_total.get() + scan);
        self.compact_active_total
            .set(self.compact_active_total.get() + e as u64);
        // Sprint 311: Per-path counters.
        if use_cone {
            self.prop_cone_calls.set(self.prop_cone_calls.get() + 1);
            self.prop_cone_evals
                .set(self.prop_cone_evals.get() + e as u64);
        } else {
            // Settle fallback (monolithic compact_dirty or levelized).
            self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
            self.prop_settle_evals
                .set(self.prop_settle_evals.get() + e as u64);
            self.prop_settle_scan
                .set(self.prop_settle_scan.get() + scan);
        }
    }

    /// Sprint 262: Read propagation counters.
    pub fn propagation_stats(&self) -> (u64, u64) {
        (
            self.propagate_calls_total.get(),
            self.propagate_tiles_total.get(),
        )
    }

    /// Sprint 277: Read scan/active ratio for scheduler design profiling.
    /// Returns (total_ops_scanned, total_ops_active) across all compact_dirty calls.
    pub fn scan_active_stats(&self) -> (u64, u64) {
        (
            self.compact_scan_total.get(),
            self.compact_active_total.get(),
        )
    }

    /// Sprint 311: Per-path propagation counters.
    /// Returns (calls, evals) for each path: (cone, settle, settle_scan, constswap, branch, commit).
    pub fn per_path_prop_stats(&self) -> [(u64, u64); 5] {
        [
            (self.prop_cone_calls.get(), self.prop_cone_evals.get()),
            (self.prop_settle_calls.get(), self.prop_settle_evals.get()),
            (
                self.prop_constswap_calls.get(),
                self.prop_constswap_evals.get(),
            ),
            (self.prop_branch_calls.get(), self.prop_branch_evals.get()),
            (self.prop_commit_calls.get(), self.prop_commit_evals.get()),
        ]
    }

    /// Sprint 311: Settle-only scan stat.
    pub fn settle_scan_total(&self) -> u64 {
        self.prop_settle_scan.get()
    }

    /// Sprint 313: Settle call reason histogram.
    /// Returns [combined, via_decode, mov_ldi, mov_ldi_wide, alu_wide_imm, alu_sra, alu_trunk].
    pub fn settle_reason_histogram(&self) -> [u64; 7] {
        std::array::from_fn(|i| self.settle_reason_counts[i].get())
    }

    /// Sprint 328: Enable phase-local settle switch counting.
    pub fn enable_settle_switch_counting(&self) {
        let n = self.settle_compact_ops.len();
        if n > 0 {
            *self.settle_switch_counts.borrow_mut() = vec![0u32; n];
        }
    }

    /// Sprint 328: Read and take settle switch counts.
    pub fn take_settle_switch_counts(&self) -> Vec<u32> {
        std::mem::take(&mut *self.settle_switch_counts.borrow_mut())
    }

    /// Sprint 330: Read settle overlap buckets.
    /// [dead_clean_block, dead_dirty_block_clean_bit, dead_dirty_set_unchanged,
    ///  hot_dirty_set_changed, cop_const]
    pub fn settle_overlap_buckets(&self) -> [u64; 5] {
        std::array::from_fn(|i| self.settle_overlap_buckets[i].get())
    }

    /// Sprint 324: Disable auto clock warmup (for A/B profiling baseline).
    pub fn disable_clock_auto_warmup(&self) {
        self.clock_auto_warmup_enabled.set(false);
    }

    /// Sprint 321: Enable phase-local clock cascade switch counting.
    /// Sprint 325: Uses flags profile for backward compatibility with S321 test.
    pub fn enable_clock_cascade_counting(&self) {
        if let Some(ref sched) = self.clock_schedule {
            *self.clock_cascade_counts_flags.borrow_mut() = vec![0u32; sched.ops.len()];
            // Disable auto-warmup so manual counting takes precedence.
            self.clock_warmup_flags_remaining.set(u32::MAX);
        }
    }

    /// Sprint 321: Read and take clock cascade switch counts (flags profile).
    pub fn take_clock_cascade_counts(&self) -> Vec<u32> {
        self.clock_warmup_flags_remaining.set(0);
        std::mem::take(&mut *self.clock_cascade_counts_flags.borrow_mut())
    }

    /// Sprint 325: Build pruned live-clock-ops from profiling counts into a
    /// specific profile (flags or noflags).
    fn build_live_clock_profile(
        &self,
        switch_counts: &[u32],
        target_ops: &std::cell::RefCell<Vec<crate::simulation::CompactOp>>,
        target_wvia: &std::cell::RefCell<Vec<(usize, u8, u64)>>,
    ) {
        let sched = match &self.clock_schedule {
            Some(s) => s,
            None => return,
        };
        if switch_counts.len() != sched.ops.len() {
            return;
        }

        let mut ops = Vec::new();
        let mut wvia = Vec::new();
        let mut wvia_idx = 0usize;

        for (slot, op) in sched.ops.iter().enumerate() {
            let is_wvia = op.op == crate::simulation::COP_WVIA;
            if switch_counts[slot] > 0 {
                ops.push(*op);
                if is_wvia {
                    wvia.push(sched.wvia[wvia_idx]);
                }
            }
            if is_wvia {
                wvia_idx += 1;
            }
        }

        *target_ops.borrow_mut() = ops;
        *target_wvia.borrow_mut() = wvia;
    }

    /// Sprint 322/325: Build pruned live-clock-ops (unified — builds both profiles
    /// from the same counts for backward compatibility with tests).
    pub fn build_live_clock_ops(&self, switch_counts: &[u32]) {
        self.build_live_clock_profile(
            switch_counts,
            &self.live_clock_ops_flags,
            &self.live_clock_wvia_flags,
        );
        self.build_live_clock_profile(
            switch_counts,
            &self.live_clock_ops_noflags,
            &self.live_clock_wvia_noflags,
        );
    }

    /// Sprint 304: How many times compact_eval_inhibit was set to true.
    /// Should be 0 for max_authority configs (constswap settle ops handle it).
    /// Sprint 307: Toggle prefiltered settle for A/B profiling.
    pub fn set_use_prefiltered_settle(&self, enable: bool) {
        self.use_prefiltered_settle.set(enable);
    }

    /// Sprint 312: Toggle no-dirty settle for A/B profiling.
    pub fn set_use_no_dirty_settle(&self, enable: bool) {
        self.use_no_dirty_settle.set(enable);
    }

    /// Sprint 352: Enable/disable backbone two-phase settle.
    pub fn set_backbone_jit_enabled(&self, enable: bool) {
        self.backbone_jit_enabled.set(enable);
    }

    /// Sprint 354: Enable/disable single-pass hybrid settle.
    /// When enabled, settle walks `settle_compact_ops` once per pass; backbone
    /// ops are evaluated unconditionally and fringe ops keep dirty-checked
    /// semantics. Takes priority over the S352 two-phase path when both are
    /// enabled.
    pub fn set_hybrid_settle_enabled(&self, enable: bool) {
        self.hybrid_settle_enabled.set(enable);
    }

    /// Sprint 355: Enable/disable backbone memoization.
    ///
    /// When enabled, the settle path hashes the backbone input vector each
    /// cycle, looks up a cache, and on hit restores the backbone boundary
    /// snapshot + runs the fringe-only kernel — skipping the 5,152-op
    /// backbone evaluation. Takes priority over hybrid (S354) and two-phase
    /// (S352). No-op if backbone partition is empty or contains irregular
    /// ops (COP_WIRE / COP_GENERIC) that would compromise input completeness.
    pub fn set_memoization_enabled(&self, enable: bool) {
        self.memoization_enabled.set(enable);
    }

    /// Sprint 355: Read cache hit/miss counters and current cache size.
    pub fn read_memoization_stats(&self) -> (u64, u64, usize) {
        (
            self.memo_hits.get(),
            self.memo_misses.get(),
            self.backbone_cache.borrow().len(),
        )
    }

    /// Sprint 355: Reset cache + counters. Called automatically by
    /// rebuild_eval_order; expose for tests.
    pub fn reset_memoization_cache(&self) {
        self.backbone_cache.borrow_mut().clear();
        self.memo_hits.set(0);
        self.memo_misses.set(0);
    }

    /// Sprint 356: Enable/disable decode-only memoization.
    ///
    /// Re-partitions settle scope into a *decode* subset (tiles whose value
    /// is purely a function of decode externals — PC, IR, decoder Consts,
    /// ROM data) and an *execute* subset (tiles that read register-state
    /// data through the operand tree). Caches the decode subset keyed on
    /// the decode-external hash; on hit, restores the cached decode tile
    /// values and runs the hybrid kernel only on the execute portion.
    /// On miss, runs the full hybrid kernel and snapshots decode outputs.
    ///
    /// Takes priority over S355 (full backbone memoization), S354 (single-pass
    /// hybrid), and S352 (two-phase). No-op if the partition was empty
    /// (e.g., rebuild has not yet run, or no settle ops exist).
    pub fn set_decode_memoization_enabled(&self, enable: bool) {
        self.decode_memoization_enabled.set(enable);
    }

    /// Sprint 356: Read decode-cache hit/miss counters and current cache size.
    pub fn read_decode_memoization_stats(&self) -> (u64, u64, usize) {
        (
            self.decode_memo_hits.get(),
            self.decode_memo_misses.get(),
            self.decode_cache.borrow().len(),
        )
    }

    /// Sprint 356: Inspect partition sizes (decode_inputs, decode_outputs,
    /// execute_ops). Useful for tests + per-benchmark reporting.
    pub fn read_decode_partition_sizes(&self) -> (usize, usize, usize) {
        (
            self.decode_input_indices.len(),
            self.decode_output_indices.len(),
            self.execute_compact_ops.len(),
        )
    }

    /// Sprint 356: Reset decode cache + counters.
    pub fn reset_decode_memoization_cache(&self) {
        self.decode_cache.borrow_mut().clear();
        self.decode_memo_hits.set(0);
        self.decode_memo_misses.set(0);
    }

    /// Sprint 357: Enable adaptive decode-memoization gating.
    ///
    /// Three-state machine: WARMUP → ENGAGED | DISABLED. The first
    /// `warmup_calls` cache calls always use the cache (warmup phase,
    /// accumulating decode_memo_hits/misses). After warmup, the lifetime
    /// hit rate is checked against `threshold`: if below, the cache is
    /// permanently disabled (subsequent calls fall through to S355/baseline);
    /// otherwise it stays engaged for the rest of the run. To re-attempt
    /// (e.g., after a known program-phase transition), call
    /// `reset_adaptive_decode_state` — it clears the mode back to WARMUP.
    ///
    /// No-op unless `set_decode_memoization_enabled(true)` is also called.
    /// Defaults: `threshold = 0.5`, `warmup_calls = 16`. Threshold sits well
    /// below the empirical S356 break-even (~0.8) on purpose: warmup misses
    /// dilute the lifetime rate, and a permissive threshold avoids penalising
    /// programs whose steady-state would clear the bar.
    pub fn set_adaptive_decode_memoization(&self, enable: bool, threshold: f32, warmup_calls: u32) {
        self.adaptive_decode_enabled.set(enable);
        self.adaptive_decode_threshold.set(threshold);
        self.adaptive_warmup_calls.set(warmup_calls.max(1));
    }

    /// Sprint 357: Read adaptive stats — (skipped calls, mode, lifetime
    /// hit rate, total cache calls observed).
    /// `mode`: 0 = warmup, 1 = engaged, 2 = disabled.
    pub fn read_adaptive_decode_stats(&self) -> (u64, u8, f32, u64) {
        let hits = self.decode_memo_hits.get();
        let misses = self.decode_memo_misses.get();
        let total = hits + misses;
        let rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };
        (
            self.adaptive_decode_skipped.get(),
            self.adaptive_decode_mode.get(),
            rate,
            total,
        )
    }

    /// Sprint 357: Reset adaptive state back to WARMUP and clear skip counter.
    /// Does NOT reset decode_memo_hits/misses — those are lifetime metrics.
    /// Combine with `reset_decode_memoization_cache()` for a full restart.
    pub fn reset_adaptive_decode_state(&self) {
        self.adaptive_decode_mode.set(0);
        self.adaptive_decode_skipped.set(0);
    }

    /// Sprint 357: Decide whether the next settle call should use the decode
    /// cache. Returns `true` when the cache path runs (lookup + hit-restore or
    /// miss-snapshot+insert), `false` when it's bypassed (fall through to S355
    /// or baseline). When `false`, also bumps the skip counter for metrics.
    ///
    /// State machine:
    ///   * WARMUP (mode == 0): always use cache. After `adaptive_warmup_calls`
    ///     observations (decode_memo_hits + decode_memo_misses), the lifetime
    ///     hit rate is checked. Above threshold → ENGAGED; below → DISABLED.
    ///   * ENGAGED (mode == 1): always use cache. Permanent until reset.
    ///   * DISABLED (mode == 2): bypass cache. Permanent until reset.
    ///
    /// When adaptive is off, always returns `true` — S356 always-on behavior.
    ///
    /// Note: programs whose total settle count never reaches `warmup_calls`
    /// stay in WARMUP forever and behave as S356. Use a small `warmup_calls`
    /// to engage adaptive earlier (faster cutoff for sequential programs);
    /// use a larger one to give loop-heavy programs more time to demonstrate
    /// hit-rate before locking in a decision.
    fn adaptive_decode_should_use_cache(&self) -> bool {
        if !self.adaptive_decode_enabled.get() {
            return true;
        }
        match self.adaptive_decode_mode.get() {
            1 => true,
            2 => {
                self.adaptive_decode_skipped
                    .set(self.adaptive_decode_skipped.get() + 1);
                false
            }
            _ => {
                // WARMUP — check whether enough samples to transition.
                let total = self.decode_memo_hits.get() + self.decode_memo_misses.get();
                let warmup = self.adaptive_warmup_calls.get() as u64;
                if total >= warmup {
                    let rate = self.decode_memo_hits.get() as f32 / total as f32;
                    if rate >= self.adaptive_decode_threshold.get() {
                        self.adaptive_decode_mode.set(1);
                        true
                    } else {
                        self.adaptive_decode_mode.set(2);
                        self.adaptive_decode_skipped
                            .set(self.adaptive_decode_skipped.get() + 1);
                        false
                    }
                } else {
                    true
                }
            }
        }
    }

    pub fn compact_eval_inhibit_count(&self) -> u64 {
        self.compact_eval_inhibit_count.get()
    }

    // Sprint 152: reapply_rom_bank_overwrite eliminated.
    // Extraction pipeline reads from westback ViaUp chain (L3→L2→L1→L0),
    // which delivers selector mux output (correct bank) to extraction tiles.
    // Directional wires (WireLeft/WireRight/WireDown) are unidirectional —
    // no UP/DOWN contamination between oy+3 selector path and oy+4 extraction chain.
    fn set_reg(&self, sim: &mut Simulation, reg: usize, value: u64) {
        if reg < 16 {
            self.regs[reg].set(value);
            sim.set_logic_value_by_idx(self.reg_indices[reg], value);
            if reg < 8 {
                // R0-R7 have operand tree data routes that need resync.
                // External writes (write_reg API) use full dirty because
                // pipeline_reg_data_dirty_indices may not cover all paths
                // on all grid configurations. The internal commit path
                // (clock edge capture) uses changed_regs_mask instead.
                self.pipeline_force_full_dirty.set(true);
            }
            // Sprint 164: R8-R15 read directly from register tiles, no pipeline resync needed
        }
    }

    fn mmio_read(&self, addr: usize) -> Option<u64> {
        if !is_v2_mmio_addr(addr) {
            return None;
        }
        if let Some(mmio) = &self.mmio {
            let value = mmio.device().read(addr as u8);
            self.last_stage_x_mmio_reads
                .borrow_mut()
                .push((addr as u8, value));
            return Some(value);
        }
        None
    }

    fn mmio_write(&self, addr: usize, value: u64) -> bool {
        if !is_v2_mmio_addr(addr) {
            return false;
        }
        if let Some(mmio) = &self.mmio {
            mmio.device().write(addr as u8, value);
            self.last_stage_x_mmio_writes
                .borrow_mut()
                .push((addr as u8, value));
            return true;
        }
        false
    }
    fn mark_pipeline_dirty(&self, sim: &mut Simulation) {
        if self.pipeline_force_full_dirty.replace(false) {
            for &idx in &self.pipeline_dirty_indices {
                sim.dirty.mark_dirty(idx);
            }
            self.changed_regs_mask.set(0);
            return;
        }

        for &idx in &self.pipeline_backbone_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }

        let changed_mask = self.changed_regs_mask.get();
        if changed_mask == 0 {
            return;
        }
        for reg in 0..16usize {
            if (changed_mask & (1 << reg)) != 0 {
                for &idx in &self.pipeline_reg_data_dirty_indices[reg] {
                    sim.dirty.mark_dirty(idx);
                }
            }
        }
        self.changed_regs_mask.set(0);
    }

    fn mark_commit_path_dirty(&self, sim: &mut Simulation) {
        for &idx in &self.commit_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
    }
    fn mark_flag_commit_dirty(&self, sim: &mut Simulation) {
        for &idx in &self.flag_commit_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
    }
    // Sprint 169: mark register-writeback tiles dirty (wb_data bus + WE mask + merge muxes).
    fn mark_reg_wb_dirty(&self, sim: &mut Simulation) {
        for &idx in &self.reg_wb_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
    }

    fn mark_branch_dirty(&self, sim: &mut Simulation) {
        // Sprint 154: Only mark flag-dependent tiles. Decode routes (ctrl_b,
        // selector_low, bank_sel) are stable from Stage-F — skip them.
        for &idx in &self.branch_flag_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
    }

    // Sprint 163: mark_ram_read_dirty and mark_ram_write_dirty removed —
    // Sprint 162 eliminated all RAM tree settle calls (direct software R/W).

    /// Sprint 173: Remove register/RAM/flag tiles from the clock cache.
    /// Software writeback unconditionally overwrites R0-R15 and RAM[0..n]
    /// AFTER the clock edge. Their clock-edge captures are wasted work.
    /// Sprint 208: Flag Z/C tiles remain elided — flag writeback is now driven
    /// by reading the physical Mux output after commit settle (see step()).
    pub(crate) fn elide_software_writeback_from_clock_cache(&mut self) {
        let mut elide_set = std::collections::HashSet::with_capacity(82);
        for &idx in &self.reg_indices {
            elide_set.insert(idx);
        }
        for &idx in &self.ram_indices {
            elide_set.insert(idx);
        }
        elide_set.insert(self.flag_z_idx);
        elide_set.insert(self.flag_c_idx);
        self.in_scope_clock_cache
            .retain(|idx| !elide_set.contains(idx));
    }

    /// Sprint 173: Remove chain tail members from dirty index vectors.
    /// Chain fusion handles tails when the chain head is evaluated —
    /// marking them dirty individually is wasted work.
    pub(crate) fn filter_dirty_indices_for_chains(&mut self, sim: &Simulation) {
        let filter = |v: &mut Vec<usize>| {
            v.retain(|&idx| !sim.is_chain_tail(idx));
        };
        filter(&mut self.pipeline_dirty_indices);
        filter(&mut self.pipeline_backbone_dirty_indices);
        for reg_vec in self.pipeline_reg_data_dirty_indices.iter_mut() {
            filter(reg_vec);
        }
        filter(&mut self.branch_dirty_indices);
        filter(&mut self.branch_flag_dirty_indices);
        filter(&mut self.commit_dirty_indices);
        filter(&mut self.reg_wb_dirty_indices);
        filter(&mut self.flag_commit_dirty_indices);
    }

    #[cfg(test)]
    pub(crate) fn settle_pipeline_only_counted(&self, sim: &mut Simulation) -> (u32, u32, u32) {
        self.mark_pipeline_dirty(sim);
        // Sprint 262: Levelized evaluation — one pass in topological order.
        sim.propagate_levelized(&self.pipeline_eval_order)
    }

    #[cfg(test)]
    pub(crate) fn settle_pipeline_only(&self, sim: &mut Simulation) {
        let _ = self.settle_pipeline_only_counted(sim);
    }

    fn run_stage_f(&self, sim: &mut Simulation) -> (PipelineLatch, StageStats) {
        let mut stats = StageStats::empty();

        // Read committed PC directly from the PC register tile.
        let pc = (sim.get_logic_value_by_idx(self.pc_idx) as u32) & self.pc_phys_mask;
        self.pc.set(pc);

        // Pipeline phase: fetch -> extraction -> decode -> tree selectors -> tree roots.
        // Sprint 272: First propagation uses cone-pruned eval (only output-feeding tiles).
        let cone_start = if self.stage_f_profiled.get() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.mark_pipeline_dirty(sim);
        self.propagate_pipeline_cone(sim, &mut stats);
        if let Some(t) = cone_start {
            self.stage_f_cone_ns
                .set(self.stage_f_cone_ns.get() + t.elapsed().as_nanos() as u64);
        }

        // Sprint 254: Combined Super Mux + upper-bank IR delivery.
        let early_bank_group = (pc >> 6) & 1;
        let rom_lanes = self.read_active_rom_lanes(sim);

        let super_mux_out = [
            self.rom_selected_low_idx,
            self.rom_selected_high_idx,
            self.rom_selected_byte2_idx,
            self.rom_selected_byte3_idx,
        ];

        // Sprint 369 (Gate B.2): instructions at PC>=128 have no physical ROM source.
        // Treat them as "upper bank" (Const-injected IR) regardless of the PC[6]
        // bank-group bit, which aliases past 128 ((128>>6)&1 == 0).
        let ext_upper = self.extended_pc && (pc as usize) >= 128;
        let is_upper = (early_bank_group == 1 && self.physical_decode.get()) || ext_upper;

        // Super Mux Const injection (all cycles when physical).
        if self.physical_super_mux.get() {
            let bank_within = ((pc >> 4) & 3) as usize;
            for lane in 0..4 {
                // Sprint 369: for the extended range, the physical bank muxes hold
                // stale data — inject the software-fetched lanes into BOTH mux inputs
                // so the physical PC[6] selector delivers the correct IR either way.
                let (bank03_val, bank47_val) = if ext_upper {
                    (rom_lanes[lane], rom_lanes[lane])
                } else {
                    (
                        sim.get_logic_value_by_idx(self.final_mux_indices[lane]),
                        sim.get_logic_value_by_idx(self.bank47_mux_indices[bank_within][lane]),
                    )
                };
                sim.set_logic_value_by_idx(self.super_mux_inject_right_indices[lane], bank03_val);
                sim.set_logic_value_by_idx(self.super_mux_inject_indices[lane], bank47_val);
                sim.dirty.mark_dirty(super_mux_out[lane]);
            }
        }

        // Sprint 254/255: Upper-bank extraction.
        // Sprint 304: The !physical_ir_spine path below is dead code in max_authority
        // (physical_ir_spine always true on 16-layer grids). Kept for auto() config.
        if is_upper {
            let (ox, oy) = self.origin;
            let gw = sim.width();

            // Sprint 369 (Gate B.2): the physical IR spine sources from physical ROM,
            // which has no entry for PC>=128. For the extended range, force the
            // Const-injection fallback (deliver the software-fetched IR to the
            // extraction ingress) even in max_authority. This is the documented
            // gated-Const-fallback — PC<128 still rides the physical spine.
            let force_const = !self.physical_ir_spine.get() || ext_upper;

            if force_const {
                let layer_size = gw * sim.height();
                sim.set_tile_3d(ox + 13, oy + 4, 0, TileType::Const);
                sim.set_logic_value_by_idx((oy + 4) * gw + (ox + 13), rom_lanes[1]);
                sim.set_tile_3d(ox + 1, oy + 4, 1, TileType::Const);
                sim.set_logic_value_by_idx(layer_size + (oy + 4) * gw + (ox + 1), rom_lanes[0]);
                // Sprint 269: Inhibit compact eval during the upper-bank Const-swap.
                // Sprint 304: dead code in max_authority for PC<128 (spine always true);
                // Sprint 369: live again only for the gated extended range (PC>=128).
                self.compact_eval_inhibit.set(true);
                self.compact_eval_inhibit_count
                    .set(self.compact_eval_inhibit_count.get() + 1);

                for i in 0..8usize {
                    let col = ox + 18 + i * 4;
                    sim.set_logic_value_by_idx((oy + 37) * gw + col, rom_lanes[0]);
                }
            }

            // Sprint 266: When spine is active, use targeted dirty set (spine +
            // R-override + branch-target + via-decode tiles) instead of marking
            // ALL ~7,400 pipeline tiles dirty. Legacy non-spine path keeps full dirty.
            // Sprint 369: the extended range injects via Const (no spine), so it also
            // needs the full dirty set to cascade the IR through decode.
            if self.physical_ir_spine.get()
                && !ext_upper
                && !self.upper_bank_dirty_indices.is_empty()
            {
                for &idx in &self.upper_bank_dirty_indices {
                    sim.dirty.mark_dirty(idx);
                }
            } else {
                self.pipeline_force_full_dirty.set(true);
            }
        }

        // Sprint 264: Batch IR decode + high-tree injection BEFORE the combined
        // propagation so Super Mux + spine + extraction + decode + high trees + top muxes
        // all settle in a single levelized pass instead of two separate passes.
        //
        // IR decode uses rom_lanes (available after Pass 0).
        // High-tree data uses register mirror (always available).
        // Selectors use physical via decode (settles during the combined pass) or software.
        // Top-mux selectors use software rd_hi/rs_hi (verified against physical byte2 after).

        let bank_group = early_bank_group;
        let low = rom_lanes[0] as u8;
        let high_val = rom_lanes[1];
        let ext = ((rom_lanes[3] as u16) << 8) | rom_lanes[2] as u16;
        let (ir_low, ir_high_val, ir_ext) = (low, high_val, ext);
        let rd_hi = (ir_ext & EXT_RD_HI) != 0;
        let rs_hi = (ir_ext & EXT_RS_HI) != 0;
        let ir_high = ir_high_val as u8;
        let opcode = (ir_high >> 3) & 0x1F;
        let rd = ir_high & 0x07;
        let rs_lo = (ir_low >> 5) & 0x07;

        // High-tree data injection (register values — no decode dependency).
        {
            for i in 0..8usize {
                sim.set_logic_value_by_idx(
                    self.high_tree_a_data_const_indices[i],
                    self.regs[8 + i].get(),
                );
            }
            const HTB_REG_MAP: [usize; 8] = [12, 13, 14, 15, 8, 9, 10, 11];
            for i in 0..8usize {
                sim.set_logic_value_by_idx(
                    self.high_tree_b_data_const_indices[i],
                    self.regs[HTB_REG_MAP[i]].get(),
                );
            }
        }

        // High-tree selector injection (needs decoded rd/rs from rom_lanes).
        let rd_s0 = if rd & 1 != 0 { u64::MAX } else { 0 };
        let rd_s1 = if rd & 2 != 0 { u64::MAX } else { 0 };
        let rd_s2 = if rd & 4 != 0 { u64::MAX } else { 0 };
        let rs_s0 = if rs_lo & 1 != 0 { u64::MAX } else { 0 };
        let rs_s1 = if rs_lo & 2 != 0 { u64::MAX } else { 0 };
        let rs_s2 = if rs_lo & 4 != 0 { 0 } else { u64::MAX }; // inverted

        if !self.physical_via_decode.get() {
            for i in 0..4usize {
                sim.set_logic_value_by_idx(self.high_tree_a_sel_const_indices[i], rd_s0);
            }
            for i in 4..6usize {
                sim.set_logic_value_by_idx(self.high_tree_a_sel_const_indices[i], rd_s1);
            }
            sim.set_logic_value_by_idx(self.high_tree_a_sel_const_indices[6], rd_s2);
            for i in 0..4usize {
                sim.set_logic_value_by_idx(self.high_tree_b_sel_const_indices[i], rs_s0);
            }
            for i in 4..6usize {
                sim.set_logic_value_by_idx(self.high_tree_b_sel_const_indices[i], rs_s1);
            }
            sim.set_logic_value_by_idx(self.high_tree_b_sel_const_indices[6], rs_s2);
        }

        // Sprint 264: Top-mux selectors from software rd_hi/rs_hi (available from
        // rom_lanes after Pass 0). Physical byte2 verification deferred to after
        // the combined propagation.
        sim.set_logic_value_by_idx(
            self.top_mux_a_sel_const_idx,
            if rd_hi { u64::MAX } else { 0 },
        );
        sim.set_logic_value_by_idx(
            self.top_mux_b_sel_const_idx,
            if rs_hi { u64::MAX } else { 0 },
        );

        // Seed high-tree + top-mux dirty sets (batched with Super Mux dirty marks).
        if self.physical_via_decode.get() {
            for &idx in &self.high_tree_a_sel_const_indices {
                sim.dirty.mark_dirty(idx);
            }
            for &idx in &self.high_tree_b_sel_const_indices {
                sim.dirty.mark_dirty(idx);
            }
            for &idx in &self.via_decode_inversion_dirty {
                sim.dirty.mark_dirty(idx);
            }
        }
        for &idx in &self.high_tree_a_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
        for &idx in &self.high_tree_b_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
        for &idx in &self.top_mux_a_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
        for &idx in &self.top_mux_b_dirty_indices {
            sim.dirty.mark_dirty(idx);
        }
        // === Sprint 264: Single combined propagation ===
        // Settles: Super Mux + spine + extraction + decode + high trees + top muxes.
        // Sprint 314: Pre-inject MOV/LDI L-enable into combined settle to eliminate
        // the separate MOV/LDI re-settle pass (was 78% of extra settle calls, S313).
        // LDI.W (wide) still needs a separate constswap settle for imm16 injection.
        let mov_ldi_pre_injected = self.physical_wb_data_authority.get()
            && (opcode == 0x02 || opcode == 0x03)
            && !(opcode == 0x03 && (ir_ext & EXT_WIDE_IMM) != 0); // exclude LDI.W
        if mov_ldi_pre_injected {
            sim.set_logic_value_by_idx(self.alu_add_l_enable_idx, u64::MAX);
            sim.dirty.mark_dirty(self.alu_add_l_mux_idx);
        }
        {
            // Sprint 334: Time injection work (everything between cone and settle).
            if let Some(_t) = cone_start {
                // cone_start was re-used as a marker that profiling is on.
                // inject_ns = time from cone end to settle start.
                // We capture it as: total_so_far - cone_ns at settle entry.
            }
            let settle_start = if self.stage_f_profiled.get() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            self.settle_reason_counts[0].set(self.settle_reason_counts[0].get() + 1); // combined
            if is_upper {
                self.mark_pipeline_dirty(sim);
                self.propagate_pipeline_until_settled(sim, &mut stats);
            } else {
                if self.physical_ir_spine.get() || self.physical_super_mux.get() {
                    self.propagate_pipeline_until_settled(sim, &mut stats);
                } else {
                    self.propagate_pipeline_until_settled(sim, &mut stats);
                }
            }
            if let Some(t) = settle_start {
                self.stage_f_settle_ns
                    .set(self.stage_f_settle_ns.get() + t.elapsed().as_nanos() as u64);
            }
        }
        // Sprint 314: Restore L-enable after combined settle.
        if mov_ldi_pre_injected {
            sim.set_logic_value_by_idx(self.alu_add_l_enable_idx, 0);
        }

        // Upper-bank restore + verification.
        if is_upper {
            let (ox, oy) = self.origin;
            let gw = sim.width();
            let layer_size = gw * sim.height();

            // Sprint 374 (Gate D.3 fix): mirror the `force_const` condition at the
            // fetch site (`!physical_ir_spine || ext_upper`). Previously this restore
            // was gated only on `!physical_ir_spine`, so in max_authority (spine on)
            // an extended-range (PC>=128) cycle set the IR-spine ingress tiles to
            // Const but never restored them to ViaUp. On a high->low control transfer
            // the next (low-bank) instruction then decoded against the stale Const
            // spine, corrupting its register writeback (B.2's test returned to HALT,
            // so it never exposed this). Restoring here makes the low instruction ride
            // the proper physical spine again.
            if !self.physical_ir_spine.get() || ext_upper {
                sim.set_tile_3d(ox + 13, oy + 4, 0, TileType::ViaUp);
                sim.set_tile_3d(ox + 1, oy + 4, 1, TileType::ViaUp);
                self.compact_eval_inhibit.set(false); // Sprint 269: restore after swap
            }

            let phys_ir_high = sim.get_logic_value_by_idx((oy + 4) * gw + (ox + 13));
            let phys_ir_low = sim.get_logic_value_by_idx(layer_size + (oy + 4) * gw + (ox + 1));

            if self.physical_ir_spine.get() {
                self.ir_spine_checks.set(self.ir_spine_checks.get() + 1);
                if phys_ir_high != rom_lanes[1] || phys_ir_low != rom_lanes[0] {
                    self.ir_spine_mismatches
                        .set(self.ir_spine_mismatches.get() + 1);
                }
                for i in 0..8usize {
                    let col = ox + 18 + i * 4;
                    let phys_r = sim.get_logic_value_by_idx((oy + 37) * gw + col);
                    if phys_r != rom_lanes[0] {
                        self.ir_spine_mismatches
                            .set(self.ir_spine_mismatches.get() + 1);
                    }
                }
            } else {
                self.upper_bank_ir_checks
                    .set(self.upper_bank_ir_checks.get() + 1);
                if phys_ir_high != rom_lanes[1] || phys_ir_low != rom_lanes[0] {
                    self.upper_bank_ir_mismatches
                        .set(self.upper_bank_ir_mismatches.get() + 1);
                }
            }
        }

        // Super Mux verification / legacy direct-set.
        if self.physical_super_mux.get() {
            for lane in 0..4 {
                let phys = sim.get_logic_value_by_idx(super_mux_out[lane]);
                self.super_mux_checks.set(self.super_mux_checks.get() + 1);
                if phys != rom_lanes[lane] {
                    self.super_mux_mismatches
                        .set(self.super_mux_mismatches.get() + 1);
                    sim.set_logic_value_by_idx(super_mux_out[lane], rom_lanes[lane]);
                }
            }
        } else {
            for (lane, &val) in rom_lanes.iter().enumerate() {
                sim.set_logic_value_by_idx(super_mux_out[lane], val);
            }
        }

        // Sprint 264: Deferred byte2 selector verification (after combined propagation).
        if self.physical_byte2_selector.get() {
            let byte2 = sim.get_logic_value_by_idx(self.rom_selected_byte2_idx);
            let phys_rd_hi = (byte2 & 1) != 0;
            let phys_rs_hi = (byte2 & 2) != 0;
            self.byte2_selector_checks
                .set(self.byte2_selector_checks.get() + 1);
            if phys_rd_hi != rd_hi || phys_rs_hi != rs_hi {
                self.byte2_selector_mismatches
                    .set(self.byte2_selector_mismatches.get() + 1);
            }
        }

        // Sprint 258: Dual-path verification for via decode selectors.
        if self.physical_via_decode.get() {
            self.via_decode_checks.set(self.via_decode_checks.get() + 1);
            let expected_a: [u64; 7] = [rd_s0, rd_s0, rd_s0, rd_s0, rd_s1, rd_s1, rd_s2];
            let raw_rs_s2 = if rs_lo & 4 != 0 { u64::MAX } else { 0 };
            let expected_b: [u64; 7] = [rs_s0, rs_s0, rs_s0, rs_s0, rs_s1, rs_s1, raw_rs_s2];
            let mut any_mismatch = false;
            for i in 0..7 {
                let phys_a = sim.get_logic_value_by_idx(self.high_tree_a_sel_const_indices[i]);
                if (phys_a != 0) != (expected_a[i] != 0) {
                    any_mismatch = true;
                    sim.set_logic_value_by_idx(
                        self.high_tree_a_sel_const_indices[i],
                        expected_a[i],
                    );
                    sim.dirty.mark_dirty(self.high_tree_a_sel_const_indices[i]);
                }
                let phys_b = sim.get_logic_value_by_idx(self.high_tree_b_sel_const_indices[i]);
                if (phys_b != 0) != (expected_b[i] != 0) {
                    any_mismatch = true;
                    sim.set_logic_value_by_idx(
                        self.high_tree_b_sel_const_indices[i],
                        expected_b[i],
                    );
                    sim.dirty.mark_dirty(self.high_tree_b_sel_const_indices[i]);
                }
            }
            if any_mismatch {
                self.via_decode_mismatches
                    .set(self.via_decode_mismatches.get() + 1);
                self.settle_reason_counts[1].set(self.settle_reason_counts[1].get() + 1); // via_decode
                self.propagate_pipeline_until_settled(sim, &mut stats);
            }
        }

        // Sprint 235: For MOV/LDI, zero out the Add tile's A operand so the
        // physical ALU produces the correct pass-through value (0 + B = B).
        // The Add tile's L Mux has LEFT=Const(0), RIGHT=operand_A, UP=L_enable.
        // Setting L_enable to MAX → L Mux selects LEFT=0 → Add computes 0+B=B.
        // Restored to 0 after re-propagation so ADD works correctly next cycle.
        // Sprint 241: bank_group guard removed — Super Mux + physical decode
        // provide correct opcode for all banks.
        // Sprint 314: Non-wide MOV/LDI L-enable was pre-injected into combined settle.
        // Only LDI.W still needs a separate pass (constswap for imm16).
        if self.physical_wb_data_authority.get() && (opcode == 0x02 || opcode == 0x03) {
            let ldi_wide = opcode == 0x03 && (ir_ext & EXT_WIDE_IMM) != 0;
            if ldi_wide {
                // LDI.W: inject imm16 + L-enable, constswap settle (separate pass).
                let hi = ((ir_ext >> 8) & 0xFF) as u64;
                let ldi_imm16 = (hi << 8) | (ir_low as u64);
                let r_out_idx = self.alu_r_mux_output_indices[0];
                sim.set_logic_value_by_idx(r_out_idx, ldi_imm16);
                sim.dirty.mark_dirty(r_out_idx - 1);
                sim.set_logic_value_by_idx(self.alu_add_l_enable_idx, u64::MAX);
                sim.dirty.mark_dirty(self.alu_add_l_mux_idx);
                self.settle_reason_counts[3].set(self.settle_reason_counts[3].get() + 1);
                self.propagate_pipeline_constswap_settled(sim, &mut stats);
                sim.set_logic_value_by_idx(self.alu_add_l_enable_idx, 0);
            }
            // else: non-wide MOV/LDI already handled by pre-injection (S314).
        }

        // Sprint 209: combined decode — extract ctrl_b (and cache ctrl_a) from single LUT.
        let combined_decode_val = self.combined_decode_lut.map(|lut| lut[opcode as usize]);

        // Sprint 204: ctrl_b from synth block (all banks), with LUT fallback.
        // Dual-path check against physical tile when bank_group == 0.
        // Sprint 222: physical_decode reads ctrl_b directly from physical Mux16to1,
        // with LUT fallback on mismatch.
        // Sprint 223: extended to all banks — Const-swap injection at extraction
        // ingress provides correct opcode bits for upper bank.
        let ctrl_b = if self.physical_decode.get() {
            // Sprint 222/223: Physical ctrl_b authority.
            let phys_cb = sim.get_logic_value_by_idx(self.ctrl_b_mux_idx) as u8;
            self.decode_ctrl_b_checks
                .set(self.decode_ctrl_b_checks.get() + 1);
            let expected_cb = if let Some(lut) = self.combined_decode_lut {
                ((lut[opcode as usize] >> 8) & 0xFF) as u8
            } else {
                CTRL_B_LUT[opcode as usize]
            };
            if phys_cb != expected_cb {
                self.decode_ctrl_b_mismatches
                    .set(self.decode_ctrl_b_mismatches.get() + 1);
                expected_cb
            } else {
                phys_cb
            }
        } else if let Some(combined) = combined_decode_val {
            self.combined_decode_checks
                .set(self.combined_decode_checks.get() + 1);
            ((combined >> 8) & 0xFF) as u8
        } else if self.synth_ctrl_b.is_enabled() {
            let synth_cb = if let Some(ref block) = self.synth_ctrl_b_block {
                let op_val = opcode as usize;
                let inputs: [u64; 5] = [
                    if (op_val) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 1) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 2) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 3) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 4) & 1 != 0 { u64::MAX } else { 0 },
                ];
                let outputs = crate::synth::integration::drive_synth_block(sim, block, &inputs);
                let mut cb_val = 0u8;
                for (i, &val) in outputs.iter().enumerate() {
                    if val != 0 {
                        cb_val |= 1 << i;
                    }
                }
                cb_val
            } else {
                CTRL_B_LUT[opcode as usize]
            };
            self.synth_ctrl_b.record_check();
            if bank_group == 0 {
                let physical_cb = sim.get_logic_value_by_idx(self.ctrl_b_mux_idx) as u8;
                if synth_cb != physical_cb {
                    self.synth_ctrl_b.record_mismatch();
                }
            }
            synth_cb
        } else {
            // No synth: use physical tile for bank 0, LUT for bank 1.
            if bank_group == 0 {
                sim.get_logic_value_by_idx(self.ctrl_b_mux_idx) as u8
            } else {
                CTRL_B_LUT[opcode as usize]
            }
        };

        // Sprint 205: ctrl_a validation/authority moved to inject_synth_pre_commit().

        let rd_eff = effective_reg(rd & 0x07, rd_hi) as usize;
        let rs_eff = if Self::opcode_uses_rs(opcode) {
            effective_reg(rs_lo, rs_hi) as usize
        } else {
            (rd_eff & 0x08) | rs_lo as usize
        };

        // Sprint 163/237: Read operands from top mux outputs.
        // Top mux selects between low tree root (rd_hi=0) and high tree root (rd_hi=1).
        // Sprint 198: Operand bypass authority — always read from reg_indices,
        // dual-path check against tree roots when applicable.
        let a_direct = sim.get_logic_value_by_idx(self.reg_indices[rd_eff]);
        let a = if self.synth_operand.is_enabled() {
            if bank_group == 0 {
                let tree_a = sim.get_logic_value_by_idx(self.op_a_root_idx);
                self.synth_operand.record_check();
                if tree_a != a_direct {
                    self.synth_operand.record_mismatch();
                }
                // Sprint 231: physical operand authority — tree root wins.
                if self.physical_operand_authority.get() {
                    self.op_authority_checks
                        .set(self.op_authority_checks.get() + 1);
                    if tree_a != a_direct {
                        self.op_authority_mismatches
                            .set(self.op_authority_mismatches.get() + 1);
                        a_direct // fallback on mismatch
                    } else {
                        tree_a
                    }
                } else {
                    a_direct
                }
            } else {
                a_direct
            }
        } else if bank_group == 1 {
            a_direct
        } else {
            sim.get_logic_value_by_idx(self.op_a_root_idx)
        };
        let b_direct = sim.get_logic_value_by_idx(self.reg_indices[rs_eff]);
        let b = if self.synth_operand.is_enabled() {
            // Only dual-path check op_b when opcode uses rs — otherwise rs_eff
            // is synthetic and the tree output is irrelevant.
            if bank_group == 0 && Self::opcode_uses_rs(opcode) {
                let tree_b = sim.get_logic_value_by_idx(self.op_b_root_idx);
                self.synth_operand.record_check();
                if tree_b != b_direct {
                    self.synth_operand.record_mismatch();
                }
                // Sprint 231: physical operand authority — tree root wins.
                if self.physical_operand_authority.get() {
                    self.op_authority_checks
                        .set(self.op_authority_checks.get() + 1);
                    if tree_b != b_direct {
                        self.op_authority_mismatches
                            .set(self.op_authority_mismatches.get() + 1);
                        b_direct // fallback on mismatch
                    } else {
                        tree_b
                    }
                } else {
                    b_direct
                }
            } else {
                b_direct
            }
        } else if bank_group == 1 {
            b_direct
        } else {
            sim.get_logic_value_by_idx(self.op_b_root_idx)
        };

        // Sprint 239: Capture raw physical Top Mux outputs before restore.
        // These are the settled values from the high trees + top muxes.
        let phys_top_a = sim.get_logic_value_by_idx(self.op_a_root_idx);
        let phys_top_b = sim.get_logic_value_by_idx(self.op_b_root_idx);

        // Sprint 237: Restore high tree + top mux Consts to 0 after readback.
        {
            for i in 0..8 {
                sim.set_logic_value_by_idx(self.high_tree_a_data_const_indices[i], 0);
                sim.set_logic_value_by_idx(self.high_tree_b_data_const_indices[i], 0);
            }
            if !self.physical_via_decode.get() {
                // Legacy: restore selector Consts to 0.
                for i in 0..7 {
                    sim.set_logic_value_by_idx(self.high_tree_a_sel_const_indices[i], 0);
                    sim.set_logic_value_by_idx(self.high_tree_b_sel_const_indices[i], 0);
                }
            }
            // else: WeightedViaUp tiles — no restore needed, they recompute each cycle.
            sim.set_logic_value_by_idx(self.top_mux_a_sel_const_idx, 0);
            sim.set_logic_value_by_idx(self.top_mux_b_sel_const_idx, 0);
        }

        let latch = PipelineLatch {
            valid: true,
            pc,
            opcode,
            rd,
            ctrl_b,
            ir_low,
            ir_ext,
            a,
            b,
            phys_top_a,
            phys_top_b,
        };

        (latch, stats)
    }

    fn run_stage_x(&self, sim: &mut Simulation, latch: PipelineLatch) -> StageStats {
        let mut stats = StageStats::empty();

        let opcode = latch.opcode;
        let ctrl_b = latch.ctrl_b;
        let imm8 = latch.ir_low;
        // Sprint 174: Wide immediate — 16-bit constant from extension word.
        let wide_imm = (latch.ir_ext & EXT_WIDE_IMM) != 0;
        let imm_val: u64 = if wide_imm {
            let hi = ((latch.ir_ext >> 8) & 0xFF) as u64;
            (hi << 8) | (imm8 as u64)
        } else {
            imm8 as u64
        };
        let rd_hi = (latch.ir_ext & EXT_RD_HI) != 0;
        let rd_eff = effective_reg(latch.rd & 0x07, rd_hi) as usize;
        let a = latch.a;
        let b = latch.b;
        let carry_before = self.flag_c.get();
        let flag_z_before = self.flag_z.get();
        let prev_regs: [u64; 16] = std::array::from_fn(|i| self.regs[i].get());

        let branch_kind = ctrl_b & 0x07;
        let is_halt = (ctrl_b & 0x20) != 0;
        if is_halt {
            self.halted.set(true);
        }

        let mut result = 0u64;
        let mut reg_write_value: Option<u64> = None;
        let mut next_flag_z = self.flag_z.get();
        let mut next_flag_c = self.flag_c.get();

        // Sprint 239: High-register ALU trunk re-source.
        // The physical ALU's L1 operand trunks start from the low tree roots (R0-R7
        // 8:1 Muxes). For high-register ops, the trunk carries the wrong operand.
        // Inject the physically-selected Top Mux outputs (latched during Stage F).
        // Uses save/restore pattern (not dirty-mark) to prevent stale values.
        //
        let uses_upper_regs = rd_hi || (latch.ir_ext & EXT_RS_HI) != 0;
        let is_sub_fn = match opcode {
            0x04 => (imm8 & 0x1F) == 1, // MUL
            0x09 => (imm8 & 0x1F) >= 1, // CLZ/CTZ/POPCNT
            0x0E => (imm8 & 0x08) != 0, // SRA
            _ => false,
        };
        let is_wide_imm = wide_imm && (0x10..=0x14).contains(&opcode);
        let mut saved_trunk_a = 0u64;
        let mut saved_trunk_b = 0u64;
        let mut trunk_injected = false;
        // Sprint 243: R Mux output injection for wide-immediate ALU ops.
        let mut r_mux_output_injected = false;
        let mut r_mux_output_col = 0usize;
        let mut r_mux_saved_tile_type = TileType::Wire;

        if self.synth_alu.is_enabled() {
            let phy_mask = self.physical_alu_opcodes.get();
            if is_wide_imm && (phy_mask & (1 << opcode)) != 0 && !is_sub_fn {
                // Sprint 243: Wide-immediate injection at R Mux output.
                // Sprint 304: Use constswap settle ops instead of physical tile-type swap.
                let col = (opcode - 0x10) as usize; // 0x10→0, 0x11→1, etc.
                let r_out_idx = self.alu_r_mux_output_indices[col];
                sim.set_logic_value_by_idx(r_out_idx, imm_val);
                sim.dirty.mark_dirty(r_out_idx - 1);
                r_mux_output_injected = true;
                r_mux_output_col = col;
                if uses_upper_regs {
                    saved_trunk_a = sim.get_logic_value_by_idx(self.alu_a_trunk_terminal_idx);
                    saved_trunk_b = sim.get_logic_value_by_idx(self.alu_b_trunk_terminal_idx);
                    trunk_injected = true;
                    if rd_hi {
                        sim.set_logic_value_by_idx(self.alu_a_trunk_terminal_idx, latch.phys_top_a);
                        for &idx in &self.alu_a_downstream_dirty {
                            sim.dirty.mark_dirty(idx);
                        }
                    }
                }
                self.settle_reason_counts[4].set(self.settle_reason_counts[4].get() + 1); // alu_wide_imm
                self.propagate_pipeline_constswap_settled(sim, &mut stats);
            } else if opcode == 0x0E
                && (imm8 & 0x08) != 0
                && self.physical_sra_computation.get()
                && (phy_mask & (1 << 0x0E)) != 0
            {
                // Sprint 249: SRA R Mux injection — corrected shift amount.
                // Sprint 304: Keep physical tile-type swap for SRA (deferred restore
                // through commit+clock for carry BitSelect), but use constswap settle
                // ops instead of compact_eval_inhibit.
                let r_out_idx = self.alu_r_mux_output_indices[7];
                let (rx, ry, rz) = Self::idx_to_xyz(sim, r_out_idx);
                r_mux_saved_tile_type = sim.tile_type_3d(rx, ry, rz);
                sim.set_tile_3d(rx, ry, rz, TileType::Const);
                sim.set_logic_value_by_idx(r_out_idx, (imm8 & 0x07) as u64);
                sim.dirty.mark_dirty(r_out_idx - 1);
                r_mux_output_injected = true;
                r_mux_output_col = 7;
                if uses_upper_regs {
                    saved_trunk_a = sim.get_logic_value_by_idx(self.alu_a_trunk_terminal_idx);
                    saved_trunk_b = sim.get_logic_value_by_idx(self.alu_b_trunk_terminal_idx);
                    trunk_injected = true;
                    if rd_hi {
                        sim.set_logic_value_by_idx(self.alu_a_trunk_terminal_idx, latch.phys_top_a);
                        for &idx in &self.alu_a_downstream_dirty {
                            sim.dirty.mark_dirty(idx);
                        }
                    }
                }
                self.settle_reason_counts[5].set(self.settle_reason_counts[5].get() + 1); // alu_sra
                self.propagate_pipeline_constswap_settled(sim, &mut stats);
            } else if uses_upper_regs
                && (phy_mask & (1 << opcode)) != 0
                && !is_sub_fn
                && !is_wide_imm
            {
                // Sprint 239: High-register trunk injection (non-wide-imm).
                saved_trunk_a = sim.get_logic_value_by_idx(self.alu_a_trunk_terminal_idx);
                saved_trunk_b = sim.get_logic_value_by_idx(self.alu_b_trunk_terminal_idx);
                trunk_injected = true;
                let rs_hi = (latch.ir_ext & EXT_RS_HI) != 0;
                if rd_hi {
                    sim.set_logic_value_by_idx(self.alu_a_trunk_terminal_idx, latch.phys_top_a);
                    for &idx in &self.alu_a_downstream_dirty {
                        sim.dirty.mark_dirty(idx);
                    }
                }
                if rs_hi && Self::opcode_uses_rs(opcode) {
                    sim.set_logic_value_by_idx(self.alu_b_trunk_terminal_idx, latch.phys_top_b);
                    for &idx in &self.alu_b_downstream_dirty {
                        sim.dirty.mark_dirty(idx);
                    }
                }
                self.settle_reason_counts[6].set(self.settle_reason_counts[6].get() + 1); // alu_trunk
                // Sprint 318: Use targeted trunk settle ops (678 of 5,492).
                if !self.trunk_settle_ops.is_empty() && !self.compact_ops_stale.get() {
                    let (d, e, s) = sim
                        .propagate_compact_dirty(&self.trunk_settle_ops, &self.trunk_settle_wvia);
                    stats.comb_deltas += d;
                    stats.comb_eval += e;
                    stats.comb_switched += s;
                    self.propagate_calls_total
                        .set(self.propagate_calls_total.get() + 1);
                    self.propagate_tiles_total
                        .set(self.propagate_tiles_total.get() + e as u64);
                    self.prop_settle_calls.set(self.prop_settle_calls.get() + 1);
                    self.prop_settle_evals
                        .set(self.prop_settle_evals.get() + e as u64);
                    self.prop_settle_scan
                        .set(self.prop_settle_scan.get() + self.trunk_settle_ops.len() as u64);
                } else {
                    self.propagate_pipeline_until_settled(sim, &mut stats);
                }
            }
        }

        // Sprint 219: Read physical ALU result at the START of Stage X.
        // The previous Stage F's pipeline settle already propagated:
        //   ctrl_a → Sprint 106 selector routes → ALU mux selectors
        //   register values → operand trees → ALU tile inputs → ALU compute → mux tree
        // Sprint 239: For high-reg ops, the re-source injection above updated the trunk.
        // So wb_alu_root_idx holds the correct ALU result for THIS instruction.
        let physical_alu_result = if self.synth_alu.is_enabled() {
            let mut val = sim.get_logic_value_by_idx(self.wb_alu_root_idx);
            // Sprint 249: Overlay synth sign extension on physical SHR for SRA.
            // After R Mux injection (above), wb_alu_root_idx holds the correct logical
            // SHR. The synth AIG block computes the 7-bit sign-extension mask. Combined:
            // physical_sra = physical_shr | mask.
            if opcode == 0x0E && (imm8 & 0x08) != 0 && self.physical_sra_computation.get() {
                if let Some(block) = &self.synth_sra_block {
                    let sign = if (a >> 63) & 1 != 0 { u64::MAX } else { 0 };
                    let s0 = if imm8 & 1 != 0 { u64::MAX } else { 0 };
                    let s1 = if imm8 & 2 != 0 { u64::MAX } else { 0 };
                    let s2 = if imm8 & 4 != 0 { u64::MAX } else { 0 };
                    let outputs = crate::synth::integration::drive_synth_block(
                        sim,
                        block,
                        &[sign, s0, s1, s2],
                    );
                    let mut mask = 0u64;
                    for (i, &v) in outputs.iter().enumerate() {
                        if v != 0 {
                            mask |= 1u64 << (57 + i);
                        }
                    }
                    let phys_sra = val | mask;
                    // Dual-path verification against software SRA.
                    let sw_sra = (a as i64).wrapping_shr(((imm8 & 0x07) as u64 & 63) as u32) as u64;
                    self.sra_computation_checks
                        .set(self.sra_computation_checks.get() + 1);
                    if phys_sra != sw_sra {
                        self.sra_computation_mismatches
                            .set(self.sra_computation_mismatches.get() + 1);
                    }
                    val = phys_sra;
                }
            }
            Some(val)
        } else {
            None
        };

        // Sprint 239/243: Restore injection tiles to pre-injection values.
        if trunk_injected {
            sim.set_logic_value_by_idx(self.alu_a_trunk_terminal_idx, saved_trunk_a);
            sim.set_logic_value_by_idx(self.alu_b_trunk_terminal_idx, saved_trunk_b);
        }
        // Sprint 249: Defer SRA R Mux restoration until after clock edge.
        // The carry BitSelect path depends on the corrected shift amount persisting
        // through commit propagation + clock edge capture.
        // Sprint 304: Only SRA still does physical tile-type swap (deferred restore).
        // Wide-imm no longer changes tile type — uses constswap settle ops instead.
        let sra_r_mux_deferred = r_mux_output_injected
            && opcode == 0x0E
            && (imm8 & 0x08) != 0
            && self.physical_sra_computation.get();

        // Sprint 154/167: per-subsection timing (opt-in).
        let ram_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };

        match opcode {
            0x02 => {
                // MOV/MOVZ/MOVNZ/MOVC/MOVNC rd, rs (Sprint 177: sub-function in imm5)
                // Sprint 364 (Gate C, physical): MFLR rd (sub_fn=5) sources LR from
                // its physical Register8 tile, not the software mirror. Result then
                // flows through the standard injection path (R0-R7: wb_data_mux Const
                // swap below; R8-R15: high_reg_wb_data_const path) — same authority
                // level as MUL: read from real tile, write to real tile, software only
                // chooses which mux input.
                let sub_fn = imm8 & 0x1F;
                result = if sub_fn == 5 {
                    // MFLR: read LR from its physical tile (8-bit value, zero-extended).
                    sim.get_logic_value_by_idx(self._lr_idx)
                } else {
                    b
                };
                let cond_met = match sub_fn {
                    1 => self.flag_z.get(),  // MOVZ: move if Z=1
                    2 => !self.flag_z.get(), // MOVNZ: move if Z=0
                    3 => self.flag_c.get(),  // MOVC: move if C=1
                    4 => !self.flag_c.get(), // MOVNC: move if C=0
                    _ => true,               // 0, 5-31: unconditional (incl. MFLR)
                };
                if cond_met {
                    reg_write_value = Some(result);
                }
            }
            0x03 => {
                // LDI rd, imm8 (Sprint 174: LDI.W rd, imm16 when wide)
                result = imm_val;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x04 => {
                // ADD rd, rs / MUL rd, rs (Sprint 188: imm5=1 → multiply)
                let sub_fn = imm8 & 0x1F;
                if sub_fn == 1 {
                    // MUL rd, rs
                    result = a.wrapping_mul(b);
                    reg_write_value = Some(result);
                    next_flag_z = result == 0;
                    // C unchanged for MUL
                } else {
                    // ADD rd, rs
                    result = a.wrapping_add(b);
                    reg_write_value = Some(result);
                    next_flag_z = result == 0;
                    next_flag_c = a > result;
                }
            }
            0x05 => {
                // SUB rd, rs
                result = a.wrapping_sub(b);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = result > a;
            }
            0x06 => {
                // AND rd, rs
                result = a & b;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x07 => {
                // OR rd, rs
                result = a | b;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x08 => {
                // XOR rd, rs
                result = a ^ b;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x09 => {
                // NOT/CLZ/CTZ/POPCNT rd (Sprint 176: sub-function in imm5)
                let sub_fn = imm8 & 0x1F;
                let sw_result = match sub_fn {
                    1 => a.leading_zeros() as u64,
                    2 => a.trailing_zeros() as u64,
                    3 => a.count_ones() as u64,
                    _ => !a, // 0 and 4-31: NOT
                };

                // Sprint 250/251: Hierarchical synth-backed CLZ/CTZ/POPCNT.
                if self.physical_bitop_computation.get()
                    && (sub_fn == 1 || sub_fn == 2)
                    && self.synth_bitscan8_blocks.is_some()
                {
                    let synth_result = self.evaluate_hierarchical_bitscan(
                        sim,
                        a,
                        sub_fn == 1, // is_clz
                    );
                    self.bitop_checks.set(self.bitop_checks.get() + 1);
                    if synth_result != sw_result {
                        self.bitop_mismatches.set(self.bitop_mismatches.get() + 1);
                    }
                    result = synth_result;
                } else if self.physical_bitop_computation.get()
                    && sub_fn == 3
                    && self.synth_popcnt8_blocks.is_some()
                {
                    let synth_result = self.evaluate_hierarchical_popcnt(sim, a);
                    self.bitop_checks.set(self.bitop_checks.get() + 1);
                    if synth_result != sw_result {
                        self.bitop_mismatches.set(self.bitop_mismatches.get() + 1);
                    }
                    result = synth_result;
                } else {
                    result = sw_result;
                }

                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x0A => {
                // NEG rd
                result = 0u64.wrapping_sub(b);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = result > a;
            }
            0x0B => {
                // INC rd
                let rhs = imm8 as u64;
                result = a.wrapping_add(rhs);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = a > result;
            }
            0x0C => {
                // DEC rd
                let rhs = imm8 as u64;
                result = a.wrapping_sub(rhs);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = result > a;
            }
            0x0D => {
                // SHL rd, imm
                // Sprint 186: skip carry when (rhs & 7) == 0 — physical BitSelect wraps.
                // next_flag_c stays as carry_before (initialized at line 607).
                let rhs = imm8 as u64;
                result = a.wrapping_shl((rhs & 63) as u32);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                if (rhs & 7) != 0 {
                    let bit_pos = (8u64.wrapping_sub(rhs)) & 63;
                    next_flag_c = ((a >> bit_pos) & 1) != 0;
                }
            }
            0x0E => {
                // SHR rd, imm / SRA rd, imm (Sprint 188: bit 3 → arithmetic)
                // Sprint 186: skip carry when (shift & 7) == 0 — physical BitSelect wraps.
                let is_arithmetic = (imm8 & 0x08) != 0;
                let shift_amount = (imm8 & 0x07) as u64;
                if is_arithmetic {
                    // Sprint 249: When physical_sra_computation is enabled, use physical
                    // SHR + synth sign extension computed at physical_alu_result read
                    // (R Mux injection → correct SHR → synth mask overlay).
                    if self.physical_sra_computation.get() {
                        if let Some(phys_sra) = physical_alu_result {
                            result = phys_sra;
                        } else {
                            result = (a as i64).wrapping_shr((shift_amount & 63) as u32) as u64;
                        }
                    } else {
                        // SRA: arithmetic (sign-extending) right shift
                        result = (a as i64).wrapping_shr((shift_amount & 63) as u32) as u64;
                    }
                } else {
                    // SHR: logical right shift
                    result = a.wrapping_shr((shift_amount & 63) as u32);
                }
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                if (shift_amount & 7) != 0 {
                    let bit_pos = shift_amount.wrapping_sub(1) & 63;
                    next_flag_c = ((a >> bit_pos) & 1) != 0;
                }
            }
            0x0F => {
                // CMP rd, rs
                result = a.wrapping_sub(b);
                next_flag_z = result == 0;
                next_flag_c = result > a;
            }
            0x10 => {
                // ADDI rd, imm8 (Sprint 174: ADDI.W rd, imm16 when wide)
                let rhs = imm_val;
                result = a.wrapping_add(rhs);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = a > result;
            }
            0x11 => {
                // SUBI rd, imm8 (Sprint 174: SUBI.W rd, imm16 when wide)
                let rhs = imm_val;
                result = a.wrapping_sub(rhs);
                reg_write_value = Some(result);
                next_flag_z = result == 0;
                next_flag_c = result > a;
            }
            0x12 => {
                // ANDI rd, imm8 (Sprint 174: ANDI.W rd, imm16 when wide)
                result = a & imm_val;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x13 => {
                // ORI rd, imm8 (Sprint 174: ORI.W rd, imm16 when wide)
                result = a | imm_val;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x14 => {
                // XORI rd, imm8 (Sprint 174: XORI.W rd, imm16 when wide)
                result = a ^ imm_val;
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x16 | 0x18 => {
                // LD/LDB — Sprint 162: direct read from software mirror.
                // All 128 RAM cells have permanent physical tiles; no bank swap needed.
                let addr = self.compute_mem_addr(&latch, opcode, b);
                if let Some(mmio_value) = self.mmio_read(addr) {
                    self.ram[addr].set(mmio_value);
                    sim.set_logic_value_by_idx(self.ram_indices[addr], mmio_value);
                }
                // Sprint 362: addr >= 128 reads extended main memory (software-backed).
                result = if addr < 128 {
                    self.ram[addr].get()
                } else {
                    self.main_mem[addr - 128].get()
                };
                reg_write_value = Some(result);
                next_flag_z = result == 0;
            }
            0x17 | 0x19 => {
                // ST/STB — Sprint 162: direct write to software mirror + physical tile.
                // Sprint 247: For bank 0 cells (0-7) with physical_ram_store_authority,
                // inject into the Ram tile's LEFT neighbor (data bus ViaUp) and UP
                // neighbor (WE extraction tile) instead of writing to the Ram tile.
                // The physical clock-edge capture evaluates: UP!=0 → stored=LEFT.
                let addr = self.compute_mem_addr(&latch, opcode, b);
                let write_data = latch.a;
                if self.mmio_write(addr, write_data) {
                    // MMIO writes bypass normal RAM semantics.
                } else if addr >= 128 {
                    // Sprint 362: extended main memory — software-backed, no tile.
                    self.main_mem[addr - 128].set(write_data);
                } else {
                    self.ram[addr].set(write_data);
                    if self.physical_ram_store_authority.get() && addr < 8 {
                        // Sprint 247: Const-swap the data bus ViaUp (LEFT neighbor)
                        // to prevent re-evaluation during clock edge propagation.
                        // The ViaUp is in the clock scope (reachable via L1 via_fwd),
                        // so without Const-swap it re-evaluates to stale L1 data.
                        let ram_idx = self.ram_indices[addr];
                        let gw = sim.width();
                        let left_idx = ram_idx - 1; // ViaUp at odd column
                        let up_idx = ram_idx - gw; // WE extraction tile
                        let (vx, vy, vz) = Self::idx_to_xyz(sim, left_idx);
                        let saved_via = sim.tile_type_3d(vx, vy, vz);
                        sim.set_tile_3d(vx, vy, vz, TileType::Const);
                        sim.set_logic_value_by_idx(left_idx, write_data);
                        sim.set_logic_value_by_idx(up_idx, u64::MAX);
                        self.ram_store_saved_via.set(Some((vx, vy, vz, saved_via)));
                        self.ram_store_inject_up_idx.set(up_idx);
                    } else {
                        sim.set_logic_value_by_idx(self.ram_indices[addr], write_data);
                    }
                    self.ram_snap_store_addr.set(addr); // Sprint 230: track for snapshots
                    // Sprint 167: trigger RAM writeback for 2 cycles (stale WE safety).
                    self.ram_writeback_countdown.set(2);
                }
            }
            _ => {
                // NOP/HALT/RET/JMP/Jcc/CALL and any reserved opcodes.
            }
        }
        let ram_elapsed = ram_start.map(|s| s.elapsed()).unwrap_or_default();

        // Sprint 219: Physical ALU authority override.
        // For opcodes with physical authority enabled, replace the software-computed
        // result with the physical ALU output (read at START of Stage X from
        // wb_alu_root_idx, which was settled by the previous Stage F).
        // Sub-function variants (MUL=0x04/imm5=1, SRA, CLZ, CTZ, POPCNT) are guarded
        // by the imm5 check — they don't have physical ALU tiles.
        // Sprint 220: Track whether physical ALU authority was applied for this
        // instruction, used to gate register writeback authority.
        let mut physical_alu_was_authoritative = false;
        if let Some(phys_alu) = physical_alu_result {
            let phy_mask = self.physical_alu_opcodes.get();
            let bank_group = (latch.pc >> 6) & 1;
            // Physical ALU is valid when:
            // - bank_group == 0, OR physical_decode enabled (Sprint 223).
            // - Opcode in phy_mask, register-writing, no sub-fn, no wide-imm.
            // Sprint 239: uses_upper_regs guard removed (trunk re-source injection).
            // CMP (flag-only, reg_write_value=None) is naturally excluded.
            // Note: wide-imm guard stays — the R Mux only carries imm8; ir_ext is
            // not physically routed to the ALU. Needs new wiring for authority.
            if phy_mask != 0
                && (phy_mask & (1 << opcode)) != 0
                && reg_write_value.is_some()
                && (bank_group == 0 || self.physical_decode.get())
            {
                // Guard sub-function variants (MUL, SRA, CLZ, CTZ, POPCNT).
                // Sprint 243: Wide-immediate guard removed — R Mux output injection
                // delivers imm16 to the physical ALU.
                if !is_sub_fn {
                    let uses_upper_regs = rd_hi || (latch.ir_ext & EXT_RS_HI) != 0;
                    if uses_upper_regs {
                        self.upper_alu_trunk_checks
                            .set(self.upper_alu_trunk_checks.get() + 1);
                    }
                    // Override: use physical ALU result instead of software computation.
                    result = phys_alu;
                    reg_write_value = Some(result);
                    physical_alu_was_authoritative = true;
                }
            }
        }

        // Sprint 235: Physical writeback-data authority for non-ALU register-producing ops.
        // For LDI (narrow) and MOV/conditional-moves (when taken), the physical ALU
        // mux tree output at wb_alu_root_idx carries the writeback value. Verify it
        // matches the software-expected result. If match, skip software register
        // injection.
        // Sprint 241: bank_group==0 guard removed — Super Mux (S211) + physical decode
        // (S223) provide correct IR/ctrl for all banks.
        // Scope: R0-R7 only (rd_hi handled by physical_high_reg_wb below).
        let mut physical_wb_path_was_authoritative = false;
        if self.physical_wb_data_authority.get()
            && !physical_alu_was_authoritative
            && reg_write_value.is_some()
        {
            let uses_upper_regs = rd_hi || (latch.ir_ext & EXT_RS_HI) != 0;
            if !uses_upper_regs && rd_eff < 8 {
                // Sprint 243: wide_imm guard removed for LDI — R Mux output
                // injection delivers imm16 during L-enable trick.
                let eligible = matches!(opcode, 0x02 | 0x03);
                if eligible {
                    if let Some(phys) = physical_alu_result {
                        self.wb_data_checks.set(self.wb_data_checks.get() + 1);
                        if phys == result {
                            physical_wb_path_was_authoritative = true;
                        } else {
                            self.wb_data_mismatches
                                .set(self.wb_data_mismatches.get() + 1);
                        }
                    }
                }
            }
        }

        // Sprint 236/238: Physical high-register writeback authority.
        // For LDI (narrow) and MOV/conditional-moves (when taken), the physical ALU
        // output is correct (L-enable injection zeroes A operand → 0+B=B).
        // Sprint 241: bank_group==0 guard removed (Super Mux + physical decode).
        let mut physical_high_reg_wb_was_authoritative = false;
        if self.physical_high_reg_writeback.get()
            && !physical_alu_was_authoritative
            && !physical_wb_path_was_authoritative
            && reg_write_value.is_some()
            && rd_hi
        {
            // Sprint 243: wide_imm guard removed for LDI — R Mux output injection.
            let eligible = matches!(opcode, 0x02 | 0x03);
            if eligible {
                if let Some(phys) = physical_alu_result {
                    if phys == result {
                        physical_high_reg_wb_was_authoritative = true;
                    }
                }
            }
        }

        // Sprint 240: CMP physical flag authority.
        // CMP (0x0F) writes Z/C flags but no register (reg_write_value=None), so
        // physical_alu_was_authoritative is never set. But the physical ALU does
        // produce the correct subtraction (Sprint 239 trunk injection feeds correct
        // operands for R0-R15). Verify the physical result matches software, then
        // trust the physical flag capture instead of software injection.
        let mut physical_cmp_flag_authoritative = false;
        if opcode == 0x0F && self.physical_flag_writeback.get() {
            if let Some(phys_alu) = physical_alu_result {
                let phy_mask = self.physical_alu_opcodes.get();
                let bank_group = (latch.pc >> 6) & 1;
                if phy_mask != 0
                    && (phy_mask & (1 << 0x0F)) != 0
                    && (bank_group == 0 || self.physical_decode.get())
                {
                    if phys_alu == result {
                        physical_cmp_flag_authoritative = true;
                    }
                }
            }
        }

        // Sprint 245: Sub-function ALU delivery authority flag.
        // The software result is injected into wb_data_mux (R0-R7) or high-reg merge
        // mux (R8-R15). Physical delivery path carries the value to the register.
        let physical_sub_fn_was_authoritative =
            is_sub_fn && self.physical_sub_fn_delivery.get() && reg_write_value.is_some();

        // Sprint 246: Sub-function flag authority. With wb_data_mux injection, the
        // flag Z zero-detect sees the correct value. MUL C_WE is suppressed in
        // inject_synth_pre_commit, so all sub-fn ops can be trusted for flags.
        // Sprint 248: SRA excluded when physical Shr tile uses raw imm8 (0x08|shift).
        // Sprint 249: R Mux injection fixes the shift amount → carry BitSelect correct.
        // SRA exclusion removed when physical_sra_computation is enabled.
        let physical_sub_fn_flag_authoritative = is_sub_fn
            && self.physical_sub_fn_flag_authority.get()
            && reg_write_value.is_some()
            && !(opcode == 0x0E && (imm8 & 0x08) != 0 && !self.physical_sra_computation.get());

        // Sprint 246: LD/LDB delivery + flag authority. The loaded value is injected
        // into wb_data_mux — flag Z zero-detect and register writeback are physical.
        // C_WE=0 for loads (ctrl_a[5]=0), so flag C unchanged.
        let physical_load_was_authoritative = matches!(opcode, 0x16 | 0x18)
            && self.physical_load_delivery.get()
            && reg_write_value.is_some();

        let pc_u8 = latch.pc & self.pc_phys_mask;

        // Sprint 107: branch LUT selector is now physical ??? ctrl_b[2:0], flag_z,
        // flag_c propagate through physical routes to the LUT during pipeline settle.
        // Ensure the branch assembly zone has settled with current flag values.
        // Sprint 110: branch_dirty_indices now includes the ~125-tile route from
        // branch_taken_core to the PC enable Or gate. This may need >100 deltas
        // to fully propagate, so we loop until stable.
        // Sprint 154: Mark only flag routes; propagation cascades through assembly/LUT/route.
        // Sprint 165: Skip branch settle for non-branch instructions.
        // When branch_kind == 0 (ctrl_b[2:0] = 0), the branch LUT output is always 0
        // regardless of flag values — no PC override. Flag propagation is unnecessary.
        // HALT has branch_kind=1, so is_halt guard is redundant but kept for clarity.
        let branch_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        // Sprint 196: Synth branch injection — drive Const tile with truth table result
        // and seed the FULL branch dirty set (37 tiles) to guarantee downstream propagation.
        if self.synth_branch.is_enabled() {
            // Sprint 200: use live synth block evaluation if available, else LUT fallback.
            let synth_taken = if let Some(ref block) = self.synth_branch_block {
                let inputs =
                    Self::compute_synth_branch_inputs(branch_kind, flag_z_before, carry_before);
                let outputs = crate::synth::integration::drive_synth_block(sim, block, &inputs);
                outputs[0] != 0
            } else {
                let sel = (branch_kind as usize)
                    | ((flag_z_before as usize) << 3)
                    | ((carry_before as usize) << 4);
                self.synth_branch_table[sel]
            };
            sim.set_logic_value_by_idx(
                self.branch_taken_core_idx,
                if synth_taken { u64::MAX } else { 0 },
            );
            // Seed FULL branch dirty set — branch_flag_dirty_indices (7 tiles) is
            // insufficient for synth mode because flag cascade may not reach Const tile.
            for &idx in &self.branch_dirty_indices {
                sim.dirty.mark_dirty(idx);
            }
        } else if branch_kind != 0 || is_halt {
            // Physical path: only flag routes needed (Sprint 154 optimization)
            self.mark_branch_dirty(sim);
        }
        // Settle branch scope (shared by both paths).
        // Sprint 270: Compact evaluation for branch scope (no Const-swaps here).
        // Sprint 279: Use scheduled propagation when available.
        if branch_kind != 0 || is_halt || self.synth_branch.is_enabled() {
            // Sprint 286: Branch uses compact_dirty (dirty-aware). No-dirty doesn't
            // work here because only ~7 flag tiles are seeded — evaluating all 7,300
            // branch ops unconditionally produces wrong results from stale inputs.
            let used_scheduled = false;
            let (d, e, s) = if !self.branch_compact_ops.is_empty() && !self.compact_ops_stale.get()
            {
                sim.propagate_compact_dirty(&self.branch_compact_ops, &self.branch_compact_wvia)
            } else {
                sim.propagate_levelized(&self.branch_eval_order)
            };
            stats.comb_deltas += d;
            stats.comb_eval += e;
            stats.comb_switched += s;
            self.propagate_calls_total
                .set(self.propagate_calls_total.get() + 1);
            self.propagate_tiles_total
                .set(self.propagate_tiles_total.get() + e as u64);
            // Sprint 281: Derive scan from dispatch path actually taken.
            let scan = if used_scheduled {
                e as u64 // scheduled: scan ≈ active (only dirty slots visited)
            } else {
                self.branch_compact_ops
                    .len()
                    .max(self.branch_eval_order.len()) as u64
            };
            self.compact_scan_total
                .set(self.compact_scan_total.get() + scan);
            self.compact_active_total
                .set(self.compact_active_total.get() + e as u64);
            // Sprint 311: per-path branch.
            self.prop_branch_calls.set(self.prop_branch_calls.get() + 1);
            self.prop_branch_evals
                .set(self.prop_branch_evals.get() + e as u64);
        }
        let branch_elapsed = branch_start.map(|s| s.elapsed()).unwrap_or_default();

        // Sprint 110: PC override enable is fully physical for all branch opcodes.
        // Sprint 125: RET target is fully physical (LR Register8 + Target Selection Mux).
        // Sprint 138: HALT target is also fully physical via ROM self-target encoding.
        let _branch_taken_core = sim.get_logic_value_by_idx(self.branch_taken_core_idx) != 0;

        // Sprint 225: Dual-path verification for physical branch direction.
        // Compare physical Mux LUT output against synth truth table.
        if self.physical_branch.get() && (branch_kind != 0 || is_halt) {
            let sel = (branch_kind as usize)
                | ((flag_z_before as usize) << 3)
                | ((carry_before as usize) << 4);
            let expected_taken = self.synth_branch_table[sel];
            self.branch_dir_checks.set(self.branch_dir_checks.get() + 1);
            if _branch_taken_core != expected_taken {
                self.branch_dir_mismatches
                    .set(self.branch_dir_mismatches.get() + 1);
            }
        }

        if is_halt {
            // HALT latch still sets halted; PC target/enable are physical.
        } else {
            match branch_kind {
                6 => {
                    // CALL: target = ir_low (physical). Enable is physical (kind=6).
                    // Sprint 125: LR capture is now physical (Register8 at ox+8,oy+8).
                    // Keep software mirror for diagnostic compatibility.
                    // Sprint 369: widen the return address to the physical PC mask.
                    self.lr.set(pc_u8.wrapping_add(1) & self.pc_phys_mask);
                }
                7 => {
                    // Sprint 125: RET is fully physical. Target Selection Mux on L2
                    // selects LR when is_ret=1 (ctrl_b bit 7). No software write needed.
                    // Keep software LR mirror in sync for read_lr() diagnostics.
                }
                _ => {}
            }
        }

        // Sprint 126: Software enable state machine removed ??? all PC enables are physical.

        // Keep pc_next_mux in the commit dirty set so PC target/value path settles.
        sim.dirty.mark_dirty(self.pc_next_mux_idx);

        // Sprint 101: register WE mask and packed flag WE mask are now physical:
        //   we_mask      = Decoder3to8(rd) & reg_write(ctrl_a[3])
        //   flag_we_mask = ctrl_a >> 4 (bit0=Z_WE, bit1=C_WE)
        let commit_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        self.inject_synth_pre_commit(sim, &latch, opcode);

        self.mark_commit_path_dirty(sim);
        // Sprint 169: Conditional reg-WB dirty gating (countdown pattern).
        // Set countdown=2 when a register is written. Cycle N executes the write
        // (countdown=2, WE=one-hot). Cycle N+1 (countdown=1) propagates WE=0 to
        // deassert the merge mux select lines. Cycle N+2 (countdown=0) skips.
        // Same pattern as Sprint 167 ram_writeback_countdown.
        if reg_write_value.is_some() {
            self.reg_wb_countdown.set(2);
        }
        if self.reg_wb_countdown.get() > 0 {
            self.mark_reg_wb_dirty(sim);
            self.reg_wb_countdown.set(self.reg_wb_countdown.get() - 1);
        }
        // Sprint 165: Skip flag commit settle for non-flag-writing instructions.
        // Physical flag_we_mask (ctrl_a[4:5]) is 0 for these opcodes, so flag
        // registers won't capture on the clock edge regardless.
        let writes_flags = matches!(
            opcode,
            0x03 | 0x04
                | 0x05
                | 0x06
                | 0x07
                | 0x08
                | 0x09
                | 0x0A
                | 0x0B
                | 0x0C
                | 0x0D
                | 0x0E
                | 0x0F
                | 0x10
                | 0x11
                | 0x12
                | 0x13
                | 0x14
                | 0x16
                | 0x18
        );
        if writes_flags {
            self.mark_flag_commit_dirty(sim);
        }

        // Sprint 236/238: Inject wb_data and WE into high-reg merge mux Const tiles.
        // This must happen before commit propagation so the merge mux settles with
        // the correct data before the clock edge captures.
        // Sprint 241: bank_group guard removed — merge mux Const injection is bank-
        // independent (we SET the values, they don't depend on physical decode state).
        let high_reg_injected =
            self.physical_high_reg_writeback.get() && rd_hi && reg_write_value.is_some();
        if high_reg_injected {
            let hi_slot = rd_eff - 8;
            sim.set_logic_value_by_idx(self.high_reg_wb_data_const_indices[hi_slot], result);
            sim.set_logic_value_by_idx(self.high_reg_we_const_indices[hi_slot], u64::MAX);
            sim.dirty
                .mark_dirty(self.high_reg_merge_mux_indices[hi_slot]);
        }

        // Sprint 243: For LDI.W, the wb_data_mux selects the L2 cascade path (RIGHT),
        // which only carries imm8 (ir_low). The full imm16 is in the ALU mux tree
        // (wb_alu_root_idx) but wb_data_mux doesn't read from it for LDI ops.
        // Fix: swap wb_data_mux to Const(imm16) before commit propagation so the
        // L1 delivery bus carries the full 16-bit immediate to the merge mux.
        // Restore to Mux after commit propagation.
        //
        // Sprint 245: Sub-function ALU delivery authority — same Const-swap mechanism
        // for MUL/SRA/CLZ/CTZ/POPCNT. The physical ALU computes the base operation
        // (ADD/SHR/NOT), not the sub-function. Inject the software result into
        // wb_data_mux so the physical delivery path carries the correct value.
        // R0-R7 register delivery only; R8-R15 handled by high_reg_injected above.
        //
        // Sprint 246: Extended wb_data_mux injection for flag authority.
        // - LD/LDB: inject loaded value for R0-R7 register delivery + flag Z path.
        // - R8-R15 sub-fn/load: inject for flag Z path only (register delivery via
        //   high-reg merge mux). WE mask is suppressed (Const(0)), so R0-R7 safe.
        // Sprint 362: Physical MUL check — must be before sub_fn_wb_inject (MUL excluded).
        let has_synth_mul =
            self.synth_mul_block.is_some() && self.physical_mul_authority.get() && is_sub_fn;
        let sub_fn_wb_inject = is_sub_fn
            && !((opcode == 0x04 && (latch.ir_ext & 0x1F) == 1) && has_synth_mul)
            && self.physical_sub_fn_delivery.get()
            && reg_write_value.is_some()
            && !rd_hi;
        let load_wb_inject = matches!(opcode, 0x16 | 0x18)
            && self.physical_load_delivery.get()
            && reg_write_value.is_some()
            && !rd_hi;
        let flag_only_wb_inject = rd_hi
            && reg_write_value.is_some()
            && ((is_sub_fn && self.physical_sub_fn_flag_authority.get())
                || (matches!(opcode, 0x16 | 0x18) && self.physical_load_delivery.get()));
        // Sprint 362: Physical MUL injection before software sub_fn_wb_inject.
        let wb_data_mux_saved_phys_mul: Option<(usize, usize, usize, TileType)> =
            if has_synth_mul && (matches!(opcode, 0x04) && (latch.ir_ext & 0x1F) == 1) {
                let block = self.synth_mul_block.as_ref().unwrap();
                let a_val = sim.get_logic_value_by_idx(self.alu_a_trunk_terminal_idx);
                let b_val = sim.get_logic_value_by_idx(self.alu_b_trunk_terminal_idx);
                let inputs: Vec<u64> = (0..16)
                    .map(|i| {
                        let val = if i < 8 { a_val } else { b_val };
                        if (val >> (i % 8)) & 1 != 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    })
                    .collect();
                let outputs =
                    crate::synth::integration::drive_injected_block_masked(sim, block, &inputs);
                let result: u64 = (0..16)
                    .filter(|i| outputs[*i] != 0)
                    .fold(0u64, |acc, i| acc | (1u64 << i));
                let (x, y, z) = Self::idx_to_xyz(sim, self.wb_data_mux_idx);
                let saved = sim.tile_type_3d(x, y, z);
                sim.set_tile_3d(x, y, z, TileType::Const);
                sim.set_logic_value_by_idx(self.wb_data_mux_idx, result);
                sim.dirty.mark_dirty(self.wb_data_mux_idx);
                // Drive trunk terminals to ensure ALU tree sees operands.
                if self.alu_a_trunk_terminal_idx < sim.tilemap.tiles.len()
                    && self.alu_b_trunk_terminal_idx < sim.tilemap.tiles.len()
                {
                    sim.dirty.mark_dirty(self.alu_a_trunk_terminal_idx);
                    sim.dirty.mark_dirty(self.alu_b_trunk_terminal_idx);
                }
                Some((x, y, z, saved))
            } else {
                None
            };
        let ldi_w_wb_data_injected = opcode == 0x03
            && (latch.ir_ext & EXT_WIDE_IMM) != 0
            && self.physical_wb_data_authority.get()
            && reg_write_value.is_some();
        // Sprint 364 (Gate C, physical): MFLR sources LR from its physical tile;
        // for R0-R7 we route through the same wb_data_mux Const-swap path as MUL.
        // R8-R15 are handled by the existing high_reg_wb_data_const injection above.
        let mflr_wb_inject =
            opcode == 0x02 && (imm8 & 0x1F) == 5 && reg_write_value.is_some() && !rd_hi;
        let wb_data_mux_saved = if ldi_w_wb_data_injected
            || sub_fn_wb_inject
            || load_wb_inject
            || flag_only_wb_inject
            || mflr_wb_inject
        {
            let (x, y, z) = Self::idx_to_xyz(sim, self.wb_data_mux_idx);
            let saved = sim.tile_type_3d(x, y, z);
            let inject_val = if ldi_w_wb_data_injected {
                let hi = ((latch.ir_ext >> 8) & 0xFF) as u64;
                (hi << 8) | (latch.ir_low as u64)
            } else {
                result
            };
            sim.set_tile_3d(x, y, z, TileType::Const);
            sim.set_logic_value_by_idx(self.wb_data_mux_idx, inject_val);
            sim.dirty.mark_dirty(self.wb_data_mux_idx);
            Some((x, y, z, saved))
        } else {
            None
        };

        // Sprint 242: When rd_hi=true and the physical WE And tile is live, temporarily
        // swap it to Const(0) so the zero WE survives commit propagation. The physical
        // And tile would otherwise re-evaluate to rd_onehot (the decoder only sees rd[0:2]).
        let we_mask_suppressed = rd_hi && reg_write_value.is_some() && self.physical_we_mask.get();
        let we_mask_saved_type = if we_mask_suppressed {
            let (x, y, z) = Self::idx_to_xyz(sim, self.we_mask_const_idx);
            let saved = sim.tile_type_3d(x, y, z);
            sim.set_tile_3d(x, y, z, TileType::Const);
            sim.set_logic_value_by_idx(self.we_mask_const_idx, 0);
            sim.dirty.mark_dirty(self.we_mask_const_idx);
            Some((x, y, z, saved))
        } else {
            None
        };

        // Sprint 244: RAM WE gate Const-swap — prevent transient WE pulses during
        // commit propagation AND clock edge.
        // Sprint 271: Narrowed to store instructions only. For non-stores, onehot is 0,
        // so the physical And gate computes 0 & enable = 0 — no swap needed.
        let ram_we_gate_suppressed =
            self.physical_ram_writeback.get() && self.synth_ram_decode.is_enabled();
        let is_store_op = opcode == 0x17 || opcode == 0x19;
        let ram_we_gate_saved = if ram_we_gate_suppressed && is_store_op {
            let onehot = sim.get_logic_value_by_idx(self.ram_write_decode_idx);
            let (x, y, z) = Self::idx_to_xyz(sim, self.ram_write_gate_idx);
            let saved = sim.tile_type_3d(x, y, z);
            sim.set_tile_3d(x, y, z, TileType::Const);
            sim.set_logic_value_by_idx(self.ram_write_gate_idx, onehot);
            sim.dirty.mark_dirty(self.ram_write_gate_idx);
            Some((x, y, z, saved))
        } else {
            None
        };

        // Sprint 246: MUL flag_we_mask Const-swap. The physical WeightedViaUp tile
        // computes flag_we_mask from ctrl_a, which gives 0x03 (Z+C WE) for opcode 0x04.
        // MUL needs C_WE=0 to preserve the previous carry. Temporarily swap to Const(0x01)
        // so the physical flag_c register doesn't capture during commit propagation.
        let flag_we_mask_saved = if is_sub_fn
            && opcode == 0x04
            && self.physical_sub_fn_flag_authority.get()
            && self.physical_flag_we_mask.get()
        {
            let (x, y, z) = Self::idx_to_xyz(sim, self.flag_we_mask_const_idx);
            let saved = sim.tile_type_3d(x, y, z);
            sim.set_tile_3d(x, y, z, TileType::Const);
            sim.set_logic_value_by_idx(self.flag_we_mask_const_idx, 0x01); // Z_WE only
            sim.dirty.mark_dirty(self.flag_we_mask_const_idx);
            Some((x, y, z, saved))
        } else {
            None
        };

        // Sprint 270→271: Compact evaluation for commit scope.
        // Fall back to levelized if any Const-swap is active (tile types mutated).
        // Sprint 271: Added wb_data_mux_saved — sub-fn/load/LDI.W delivery swaps
        // wb_data_mux to Const, but compact ops still encode it as COP_MUX.
        // Sprint 278: Use scheduled propagation when available (ordered active-work).
        let any_commit_swap = wb_data_mux_saved.is_some()
            || wb_data_mux_saved_phys_mul.is_some()
            || we_mask_saved_type.is_some()
            || ram_we_gate_saved.is_some()
            || flag_we_mask_saved.is_some();
        let mut commit_used_scheduled = false;
        let (d, e, s) = if self.commit_schedule.is_some()
            && !any_commit_swap
            && !self.compact_ops_stale.get()
        {
            commit_used_scheduled = true;
            if self.commit_profiled.get() {
                let (d, e, s, drain, wl, _) = sim.propagate_compact_scheduled_profiled(
                    self.commit_schedule.as_ref().unwrap(),
                    &[],
                    false,
                );
                self.commit_drain_ns.set(self.commit_drain_ns.get() + drain);
                self.commit_worklist_ns
                    .set(self.commit_worklist_ns.get() + wl);
                (d, e, s)
            } else {
                sim.propagate_compact_scheduled(self.commit_schedule.as_ref().unwrap(), &[], false)
            }
        } else if !self.commit_compact_ops.is_empty()
            && !any_commit_swap
            && !self.compact_ops_stale.get()
        {
            sim.propagate_compact_dirty(&self.commit_compact_ops, &self.commit_compact_wvia)
        } else {
            sim.propagate_levelized(&self.commit_eval_order)
        };
        stats.comb_deltas += d;
        stats.comb_eval += e;
        stats.comb_switched += s;
        self.propagate_calls_total
            .set(self.propagate_calls_total.get() + 1);
        self.propagate_tiles_total
            .set(self.propagate_tiles_total.get() + e as u64);
        // Sprint 281: Derive scan from dispatch path actually taken.
        let scan = if commit_used_scheduled {
            e as u64 // scheduled: only dirty slots visited
        } else {
            self.commit_compact_ops
                .len()
                .max(self.commit_eval_order.len()) as u64
        };
        self.compact_scan_total
            .set(self.compact_scan_total.get() + scan);
        self.compact_active_total
            .set(self.compact_active_total.get() + e as u64);
        // Sprint 311: per-path commit.
        self.prop_commit_calls.set(self.prop_commit_calls.get() + 1);
        self.prop_commit_evals
            .set(self.prop_commit_evals.get() + e as u64);

        // Sprint 242: Restore the WE And tile after commit propagation settled with WE=0.
        if let Some((x, y, z, saved)) = we_mask_saved_type {
            sim.set_tile_3d(x, y, z, saved);
        }
        // Sprint 246: Restore flag_we_mask tile after MUL commit.
        if let Some((x, y, z, saved)) = flag_we_mask_saved {
            sim.set_tile_3d(x, y, z, saved);
        }

        // Sprint 230: Snapshot #1 — bank-0 RAM state after commit propagation.
        if self.physical_ram_writeback.get() && matches!(opcode, 0x17 | 0x19) {
            for i in 0..8 {
                self.ram_snap_post_commit[i].set(sim.get_logic_value_by_idx(self.ram_indices[i]));
            }
        }

        // Sprint 224: Fix upper-bank branch target delivery to Target Selection Mux.
        // The physical westback ir_low at L2(ox+2, oy+4) is always bank 0 data.
        // Sprint 125 Phase 8 propagates this through L2 east bus → L3 hop → Target
        // Mux RIGHT. For upper bank, directly inject the correct ir_low value into
        // every tile in the delivery path, then re-propagate from the Target Mux.
        //
        // Sprint 260: When physical_ir_spine is active, L2(ox+2, oy+4) is ViaUp
        // reading L3(ox+2, oy+4) which reads the z=14 spine portal — ir_low for
        // ALL banks. The physical route already delivers correct ir_low. Skip injection.
        if (latch.pc >> 6) & 1 == 1 && self.physical_decode.get() && !self.physical_ir_spine.get() {
            let (ox, oy) = self.origin;
            let gw = sim.width();
            let layer_size = gw * sim.height();
            let ir_low_val = latch.ir_low as u64;
            // L2 east bus: (ox+2..7, oy+4) — source + WireRight chain
            for x in (ox + 2)..=(ox + 7) {
                sim.set_logic_value_by_idx(2 * layer_size + (oy + 4) * gw + x, ir_low_val);
            }
            // L3 ViaDown at (ox+7, oy+4)
            sim.set_logic_value_by_idx(3 * layer_size + (oy + 4) * gw + (ox + 7), ir_low_val);
            // L3 WireUp chain: (ox+7, oy+3..1)
            for y in (oy + 1)..=(oy + 3) {
                sim.set_logic_value_by_idx(3 * layer_size + y * gw + (ox + 7), ir_low_val);
            }
            // L2 ViaUp at (ox+7, oy+1) — feeds Target Mux RIGHT
            sim.set_logic_value_by_idx(2 * layer_size + (oy + 1) * gw + (ox + 7), ir_low_val);
            // Mark Target Mux dirty and re-propagate commit scope.
            let target_mux_idx = 2 * layer_size + (oy + 1) * gw + (ox + 6);
            sim.dirty.mark_dirty(target_mux_idx);
            // Sprint 271: re-propagation compact eval check. wb_data_mux swap
            // persists until after re-propagation (restored at line ~2655).
            // ram_we_gate swap only fires for stores (Sprint 271 narrowing).
            let (d2, e2, s2) = if !self.commit_compact_ops.is_empty()
                && wb_data_mux_saved.is_none()
                && ram_we_gate_saved.is_none()
                && !self.compact_ops_stale.get()
            {
                sim.propagate_compact_dirty(&self.commit_compact_ops, &self.commit_compact_wvia)
            } else {
                sim.propagate_levelized(&self.commit_eval_order)
            };
            stats.comb_deltas += d2;
            stats.comb_eval += e2;
            stats.comb_switched += s2;
        } else if (latch.pc >> 6) & 1 == 1
            && self.physical_decode.get()
            && self.physical_ir_spine.get()
        {
            // Sprint 260: Verify physical branch-target delivery.
            // The L2 ViaUp at (ox+7, oy+1) should carry correct ir_low from spine.
            let (ox, oy) = self.origin;
            let gw = sim.width();
            let layer_size = gw * sim.height();
            let phys = sim.get_logic_value_by_idx(2 * layer_size + (oy + 1) * gw + (ox + 7));
            let expected = latch.ir_low as u64;
            self.branch_target_checks
                .set(self.branch_target_checks.get() + 1);
            if phys != expected {
                self.branch_target_mismatches
                    .set(self.branch_target_mismatches.get() + 1);
            }
        }

        // Sprint 249: Restore wb_data_mux AFTER upper-bank re-propagation.
        // Previously restored right after the first commit propagation (Sprint 243).
        // But the upper-bank branch target re-propagation (Sprint 224) runs a second
        // commit-scope propagation. If wb_data_mux is restored to Mux before that,
        // it re-evaluates with natural ALU inputs (e.g., SHR instead of SRA),
        // corrupting the delivery value for sub-fn ops on upper bank.
        if let Some((x, y, z, saved)) = wb_data_mux_saved {
            sim.set_tile_3d(x, y, z, saved);
        }
        // Sprint 362: Restore wb_data_mux if physical MUL was used.
        if let Some((x, y, z, saved)) = wb_data_mux_saved_phys_mul {
            sim.set_tile_3d(x, y, z, saved);
        }

        // Sprint 244: Post-commit re-inject removed. The RAM WE gate Const-swap
        // prevents transient WE pulses during commit propagation, so Ram tiles
        // are no longer corrupted. Snapshot #2 retained for diagnostics.
        if self.physical_ram_writeback.get() && matches!(opcode, 0x17 | 0x19) {
            for i in 0..8 {
                self.ram_snap_post_reinject[i].set(sim.get_logic_value_by_idx(self.ram_indices[i]));
            }
        }

        let commit_elapsed = commit_start.map(|s| s.elapsed()).unwrap_or_default();

        // Clocked commit: rising edge captures, falling edge holds.
        // Sprint 166: Masked clock tick — evaluate only tiles within the CPU zone.
        let clock_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        // Sprint 168: Lightweight clock edge — no delay/glitch/arrival overhead.
        // Sprint 232: Use the no-flags clock cache on non-flag instructions so
        // flag Register8 tiles are excluded from the clock edge and keep their
        // current value. This eliminates the per-cycle flag re-inject.
        let clock_cache = if writes_flags || !self.physical_flag_writeback.get() {
            &self.in_scope_clock_cache
        } else {
            &self.in_scope_clock_cache_no_flags
        };
        // Sprint 276: Compact clock edge — delta 0 captures via eval_tile,
        // then topological compact ops replace the 40-67 delta cascade.
        // Sprint 279: Use scheduled cascade when available (ordered active-work).
        // Guard: fall back to lightweight when compact ops are stale (tile types
        // changed post-build) or when a runtime Const-swap is active (Sprint 269).
        // Sprint 282: Chain-exit dep fix added but clock schedule still
        // causes 35 authority test failures. Suppress is too broad for
        // non-max configs. Disabled pending propagation mode refinement.
        // Sprint 283: Clock schedule disabled. Non-terminal + re-drain gets
        // 97% coverage (569/587) but 74 mismatches remain and 11 passes needed
        // vs compact's 2. The dep table doesn't fully replicate dirty_dependents'
        // propagation — the scheduler needs either a complete dep table or a
        // different propagation strategy.
        // Sprint 286: Clock schedule uses no-dirty pattern (single topological
        // pass, frontier-only dirty marks). Replaces the scheduler's dep table
        // approach which had convergence issues (Sprint 283).
        let commit_rise = if self.clock_schedule.is_some()
            && !self.compact_ops_stale.get()
            && !self.compact_eval_inhibit.get()
        {
            // Sprint 325: Dual-profile auto-warmup — separate for flag-writing
            // and non-flag cycles. Each profile warmups independently (5 cycles).
            let is_flags = writes_flags || !self.physical_flag_writeback.get();
            let (live_ops_ref, live_wvia_ref, counts_ref, warmup_cell) = if is_flags {
                (
                    &self.live_clock_ops_flags,
                    &self.live_clock_wvia_flags,
                    &self.clock_cascade_counts_flags,
                    &self.clock_warmup_flags_remaining,
                )
            } else {
                (
                    &self.live_clock_ops_noflags,
                    &self.live_clock_wvia_noflags,
                    &self.clock_cascade_counts_noflags,
                    &self.clock_warmup_noflags_remaining,
                )
            };

            // Auto-start warmup counting for this profile if needed.
            let warmup = warmup_cell.get();
            if self.clock_auto_warmup_enabled.get()
                && live_ops_ref.borrow().is_empty()
                && warmup == 0
                && counts_ref.borrow().is_empty()
            {
                if let Some(ref sched) = self.clock_schedule {
                    *counts_ref.borrow_mut() = vec![0u32; sched.ops.len()];
                }
                warmup_cell.set(10);
            }

            let counting = !counts_ref.borrow().is_empty();

            if counting {
                // Warmup phase: run with counting.
                let r = sim.tick_clock_edge_scheduled_counted(
                    &self.clock_scope_mask,
                    clock_cache,
                    self.clock_schedule.as_ref().unwrap(),
                    &mut counts_ref.borrow_mut(),
                );
                let rem = warmup_cell.get();
                if rem > 0 {
                    warmup_cell.set(rem - 1);
                    if rem == 1 {
                        // Warmup complete: build pruned profile.
                        let counts = std::mem::take(&mut *counts_ref.borrow_mut());
                        self.build_live_clock_profile(&counts, live_ops_ref, live_wvia_ref);
                    }
                }
                r
            } else if !live_ops_ref.borrow().is_empty() {
                // Use pruned live-clock ops for this profile.
                let ops = live_ops_ref.borrow();
                let wvia = live_wvia_ref.borrow();
                sim.tick_clock_edge_pruned(
                    &self.clock_scope_mask,
                    clock_cache,
                    &ops,
                    &wvia,
                    &[], // unused — full scope_mask used for frontier
                )
            } else {
                sim.tick_clock_edge_scheduled(
                    &self.clock_scope_mask,
                    clock_cache,
                    self.clock_schedule.as_ref().unwrap(),
                )
            }
        } else if !self.clock_compact_ops.is_empty()
            && !self.compact_ops_stale.get()
            && !self.compact_eval_inhibit.get()
        {
            sim.tick_clock_edge_compact(
                &self.clock_scope_mask,
                clock_cache,
                &self.clock_compact_ops,
                &self.clock_compact_wvia,
            )
        } else {
            sim.tick_clock_edge_lightweight(&self.clock_scope_mask, clock_cache)
        };
        stats.clock_deltas += commit_rise.total_deltas;
        stats.clock_eval += commit_rise.tiles_evaluated;
        stats.clock_switched += commit_rise.tiles_switched;
        stats.glitches += commit_rise.glitches_detected;
        stats.converged &= commit_rise.converged;

        // Sprint 154: Falling edge — no register captures (Register8/Register64
        // capture on rising only). Skip propagation; just toggle clock state.
        sim.tick_clock_toggle_only();

        // Sprint 236: Restore high-reg Const tiles to 0 after clock capture.
        if high_reg_injected {
            let hi_slot = rd_eff - 8;
            sim.set_logic_value_by_idx(self.high_reg_wb_data_const_indices[hi_slot], 0);
            sim.set_logic_value_by_idx(self.high_reg_we_const_indices[hi_slot], 0);
        }
        // Sprint 244: Restore RAM WE And gate after clock edge captured with stable WE.
        if let Some((x, y, z, saved)) = ram_we_gate_saved {
            sim.set_tile_3d(x, y, z, saved);
        }
        // Sprint 247: Restore ViaUp tile type + clear UP after clock edge.
        if let Some((vx, vy, vz, saved_via)) = self.ram_store_saved_via.get() {
            sim.set_tile_3d(vx, vy, vz, saved_via);
            self.ram_store_saved_via.set(None);
        }
        // Sprint 249: Restore SRA R Mux output tile type after clock edge.
        // The carry BitSelect path reads (A >> (shift-1)) & 1 where shift comes from
        // the R Mux output → Sub tile. Keeping R Mux as Const(corrected_shift) through
        // commit propagation + clock edge ensures the flag C register captures correctly.
        if sra_r_mux_deferred {
            let r_out_idx = self.alu_r_mux_output_indices[r_mux_output_col];
            let (rx, ry, rz) = Self::idx_to_xyz(sim, r_out_idx);
            sim.set_tile_3d(rx, ry, rz, r_mux_saved_tile_type);
        }
        {
            let up_restore = self.ram_store_inject_up_idx.get();
            if up_restore != 0 {
                sim.set_logic_value_by_idx(up_restore, 0);
                self.ram_store_inject_up_idx.set(0);
            }
        }

        let clock_elapsed = clock_start.map(|s| s.elapsed()).unwrap_or_default();

        // Sprint 244: Post-clock re-inject removed. The RAM WE gate Const-swap
        // keeps WE stable through the clock edge, preventing transient corruption.
        // Snapshot #3 retained for diagnostics.
        if self.physical_ram_writeback.get() && matches!(opcode, 0x17 | 0x19) {
            for i in 0..8 {
                self.ram_snap_post_clock[i].set(sim.get_logic_value_by_idx(self.ram_indices[i]));
            }
        }

        // Sprint 219: ALU readback — compare physical ALU result (read at START of
        // Stage X, before commit disturbs the pipeline-settled values) against software.
        // The physical value was captured in `physical_alu_result` above.
        if let Some(phys_alu) = physical_alu_result {
            if reg_write_value.is_some() {
                let is_alu_opcode = matches!(
                    opcode,
                    0x04 | 0x05
                        | 0x06
                        | 0x07
                        | 0x08
                        | 0x09
                        | 0x0A
                        | 0x0B
                        | 0x0C
                        | 0x0D
                        | 0x0E
                        | 0x0F
                        // Sprint 226: immediate ALU ops (use_imm=1 → R Mux selects imm8)
                        | 0x10
                        | 0x11
                        | 0x12
                        | 0x13
                        | 0x14
                );
                // Sprint 243: skip_wide removed — R Mux output injection delivers
                // imm16 to the physical ALU for wide-immediate ops.
                // Sprint 245: skip sub-fn ops — physical ALU computes base operation
                // (ADD/NOT/SHR), not the sub-function (MUL/CLZ/CTZ/POPCNT/SRA).
                // Mismatch is expected and not diagnostic.
                if is_alu_opcode && !is_sub_fn {
                    self.synth_alu.record_check();
                    if phys_alu != result {
                        self.synth_alu.record_mismatch();
                        self.alu_opcode_mismatches[opcode as usize]
                            .set(self.alu_opcode_mismatches[opcode as usize].get() + 1);
                        // D2: Correlate — is the mux selector (ctrl_a[2:0]) also wrong?
                        // Sprint 223: check for all banks (Super Mux provides correct
                        // opcode bits for upper bank).
                        {
                            let phys_ca = sim.get_logic_value_by_idx(self.ctrl_a_mux_idx) as u8;
                            let expected_ca = CTRL_A_LUT[opcode as usize];
                            if (phys_ca & 0x07) != (expected_ca & 0x07) {
                                self.alu_mux_select_mismatches
                                    .set(self.alu_mux_select_mismatches.get() + 1);
                            }
                        }
                    }
                }
            }
        }

        let mut changed_regs_mask = 0u16;
        let mut final_regs = prev_regs;
        if let Some(value) = reg_write_value {
            final_regs[rd_eff] = value;
        }
        self.verify_synth_post_commit(sim, &latch, opcode);
        // Sprint 163/167: Detect physical-vs-software mismatch for R0-R7 and
        // conditionally write only registers that changed or have a mismatch.
        // Sprint 218: Also count register capture statistics for Phase 2 planning.
        // Sprint 220: When physical_reg_writeback is enabled AND the current instruction
        // had physical ALU authority, skip set_logic_value_by_idx for the target register
        // — the physical merge mux + clock edge capture already wrote the correct value.
        // Sprint 235: Also skip when physical writeback-data authority was established
        // for non-ALU register-producing ops (LDI, MOV, conditional moves).
        // Sprint 242: rd_hi guard removed. The WE suppression in inject_synth_pre_commit
        // now forces we_mask=0 for high-reg writes, preventing aliased low-register
        // corruption. Software R0-R7 writeback can now be safely skipped for all
        // authoritative ops regardless of rd_hi.
        // Sprint 245: Sub-fn delivery (R0-R7 only; R8-R15 handled by high_reg_injected).
        // Sprint 246: LD/LDB delivery (R0-R7 only; R8-R15 handled by high_reg_injected).
        let skip_software_reg_wb = self.physical_reg_writeback.get()
            && (physical_alu_was_authoritative
                || physical_wb_path_was_authoritative
                || (physical_sub_fn_was_authoritative && !rd_hi)
                || (physical_load_was_authoritative && !rd_hi));
        for i in 0..8usize {
            let phys_val = sim.get_logic_value_by_idx(self.reg_indices[i]);
            let mismatch = phys_val != final_regs[i];
            let changed = final_regs[i] != prev_regs[i];
            if self.synth_alu.is_enabled() {
                self.reg_capture_checks
                    .set(self.reg_capture_checks.get() + 1);
                if mismatch {
                    self.reg_capture_mismatches
                        .set(self.reg_capture_mismatches.get() + 1);
                }
            }
            if skip_software_reg_wb && i == (rd_eff & 0x07) {
                // Sprint 220: Physical clock capture is authoritative for target register.
                // Sprint 245: Also covers sub-fn delivery (wb_data_mux Const-swap).
                // Sprint 246: Also covers LD/LDB delivery.
                self.reg_wb_checks.set(self.reg_wb_checks.get() + 1);
                if physical_sub_fn_was_authoritative && !rd_hi {
                    self.sub_fn_delivery_checks
                        .set(self.sub_fn_delivery_checks.get() + 1);
                }
                if physical_load_was_authoritative && !rd_hi {
                    self.load_delivery_checks
                        .set(self.load_delivery_checks.get() + 1);
                }
                if mismatch {
                    // Physical capture didn't match expected — fall back to software.
                    self.reg_wb_mismatches.set(self.reg_wb_mismatches.get() + 1);
                    if physical_sub_fn_was_authoritative && !rd_hi {
                        self.sub_fn_delivery_mismatches
                            .set(self.sub_fn_delivery_mismatches.get() + 1);
                    }
                    if physical_load_was_authoritative && !rd_hi {
                        self.load_delivery_mismatches
                            .set(self.load_delivery_mismatches.get() + 1);
                    }
                    changed_regs_mask |= 1 << i;
                    self.regs[i].set(final_regs[i]);
                    sim.set_logic_value_by_idx(self.reg_indices[i], final_regs[i]);
                } else if changed {
                    // Physical capture correct — update software mirror only.
                    changed_regs_mask |= 1 << i;
                    self.regs[i].set(final_regs[i]);
                }
            } else if mismatch || changed {
                changed_regs_mask |= 1 << i;
                self.regs[i].set(final_regs[i]);
                sim.set_logic_value_by_idx(self.reg_indices[i], final_regs[i]);
            }
        }
        // Sprint 167/236/238: R8-R15 writeback with optional physical authority.
        // Sprint 238: Any merge-mux-injected high-reg write trusts clock capture.
        // high_reg_injected covers all register-writing bank-0 ops with rd_hi.
        let skip_high_reg_sw_wb = physical_high_reg_wb_was_authoritative || high_reg_injected;
        for i in 8..16usize {
            if skip_high_reg_sw_wb && i == rd_eff {
                // Sprint 236/238: Physical clock capture is authoritative for target high register.
                // Sprint 245: Also covers sub-fn high-reg delivery via high_reg_injected.
                // Sprint 246: Also covers LD/LDB high-reg delivery.
                self.high_reg_wb_checks
                    .set(self.high_reg_wb_checks.get() + 1);
                if physical_sub_fn_was_authoritative && rd_hi {
                    self.sub_fn_delivery_checks
                        .set(self.sub_fn_delivery_checks.get() + 1);
                }
                if physical_load_was_authoritative && rd_hi {
                    self.load_delivery_checks
                        .set(self.load_delivery_checks.get() + 1);
                }
                let phys_val = sim.get_logic_value_by_idx(self.reg_indices[i]);
                if phys_val != final_regs[i] {
                    // Physical capture didn't match expected — fall back to software.
                    self.high_reg_wb_mismatches
                        .set(self.high_reg_wb_mismatches.get() + 1);
                    if physical_sub_fn_was_authoritative && rd_hi {
                        self.sub_fn_delivery_mismatches
                            .set(self.sub_fn_delivery_mismatches.get() + 1);
                    }
                    if physical_load_was_authoritative && rd_hi {
                        self.load_delivery_mismatches
                            .set(self.load_delivery_mismatches.get() + 1);
                    }
                    changed_regs_mask |= 1 << i;
                    self.regs[i].set(final_regs[i]);
                    sim.set_logic_value_by_idx(self.reg_indices[i], final_regs[i]);
                } else if final_regs[i] != prev_regs[i] {
                    // Physical capture correct — update software mirror only.
                    changed_regs_mask |= 1 << i;
                    self.regs[i].set(final_regs[i]);
                }
            } else if final_regs[i] != prev_regs[i] {
                changed_regs_mask |= 1 << i;
                self.regs[i].set(final_regs[i]);
                sim.set_logic_value_by_idx(self.reg_indices[i], final_regs[i]);
            }
        }
        self.changed_regs_mask.set(changed_regs_mask);

        // Sprint 229: RAM readback diagnostic. When physical_ram_writeback is enabled,
        // RAM tiles participate in clock edge captures (un-elided from clock cache).
        // Compare physical RAM tiles against software mirror BEFORE countdown writeback
        // to measure whether the physical WE gating produces correct results.
        let _rw_count_before = self.ram_writeback_countdown.get();
        if self.physical_ram_writeback.get() {
            #[cfg(test)]
            let cycle = self.cycle_count.get();
            for i in 0..128usize {
                let phys = sim.get_logic_value_by_idx(self.ram_indices[i]);
                let expected = self.ram[i].get();
                self.ram_wb_checks.set(self.ram_wb_checks.get() + 1);
                if phys != expected {
                    self.ram_wb_mismatches.set(self.ram_wb_mismatches.get() + 1);
                    if matches!(opcode, 0x17 | 0x19) {
                        self.ram_wb_store_mismatches
                            .set(self.ram_wb_store_mismatches.get() + 1);
                        // Sprint 230: Probe data path for store-cycle mismatches.
                        #[cfg(test)]
                        if i < 8 {
                            let tree_a = sim.get_logic_value_by_idx(self.op_a_root_idx);
                            eprintln!(
                                "[S230] cycle={} cell={} phys=0x{:X} expected=0x{:X} tree_a=0x{:X}",
                                cycle, i, phys, expected, tree_a
                            );
                        }
                    } else {
                        self.ram_wb_nonstore_mismatches
                            .set(self.ram_wb_nonstore_mismatches.get() + 1);
                    }
                }
            }
        }

        // Sprint 244: Countdown writeback removed. The RAM WE gate Const-swap
        // eliminates transient WE corruption during both commit propagation and
        // clock edge, making the 128-cell re-inject unnecessary.
        // Countdown still decremented for diagnostic compatibility.
        let rw_count = self.ram_writeback_countdown.get();
        if rw_count > 0 {
            self.ram_writeback_countdown.set(rw_count - 1);
        }

        // Sprint 208: Conditional flag writeback.
        // Sprint 221: The L1 route from wb_data to the flag Zero tile at (ox+18, oy+50)
        // IS physically wired — route tiles overwrite the L1 guard Const(0) tiles placed
        // earlier. The physical path: wb_data → L1 south (ox+37, oy+11..49) → L1 west
        // (ox+18..36, oy+49) → L0 ViaUp (ox+17, oy+50) → Zero tile → flag_z_mux.
        // When physical_flag_writeback is enabled, the physical Mux + Register8 clock
        // capture computes and stores the correct flag values — software injection is
        // skipped for flag-writing instructions (with mismatch fallback).
        self.flag_z.set(next_flag_z);
        self.flag_c.set(next_flag_c);
        if writes_flags {
            if self.physical_flag_writeback.get()
                && (physical_alu_was_authoritative
                    || physical_wb_path_was_authoritative
                    || physical_high_reg_wb_was_authoritative
                    || physical_cmp_flag_authoritative
                    || physical_sub_fn_flag_authoritative
                    || physical_load_was_authoritative)
            {
                // Sprint 221: Physical flag authority — clock edge already captured
                // the correct flag values from the physical Mux outputs.
                // Sprint 240: Extended to LDI (wb_path/high_reg_wb verified wb_data
                // correct → flag Z capture correct) and CMP (physical ALU verified
                // correct → flag Z/C capture correct).
                // Sprint 246: Extended to sub-fn ops (wb_data_mux injection → flag Z
                // correct; MUL C_WE suppressed) and LD/LDB (wb_data_mux injection →
                // flag Z correct; C_WE=0 for loads).
                let phys_z = sim.get_logic_value_by_idx(self.flag_z_idx) != 0;
                let phys_c = sim.get_logic_value_by_idx(self.flag_c_idx) != 0;
                self.flag_wb_checks.set(self.flag_wb_checks.get() + 1);
                let z_ok = phys_z == next_flag_z;
                let c_ok = phys_c == next_flag_c;
                if !z_ok {
                    self.flag_z_mismatches.set(self.flag_z_mismatches.get() + 1);
                }
                if !c_ok {
                    self.flag_c_mismatches.set(self.flag_c_mismatches.get() + 1);
                }
                if !z_ok || !c_ok {
                    // Mismatch — fall back to software injection.
                    self.flag_wb_mismatches
                        .set(self.flag_wb_mismatches.get() + 1);
                    sim.set_logic_value_by_idx(
                        self.flag_z_idx,
                        if next_flag_z { u64::MAX } else { 0 },
                    );
                    sim.set_logic_value_by_idx(
                        self.flag_c_idx,
                        if next_flag_c { u64::MAX } else { 0 },
                    );
                }
            } else {
                sim.set_logic_value_by_idx(self.flag_z_idx, if next_flag_z { u64::MAX } else { 0 });
                sim.set_logic_value_by_idx(self.flag_c_idx, if next_flag_c { u64::MAX } else { 0 });
            }
        } else if self.physical_flag_writeback.get() {
            // Sprint 232: Non-flag instructions now use in_scope_clock_cache_no_flags,
            // so flag Register8 tiles are excluded from the clock edge entirely.
            // They keep their current value — no re-inject needed.
        }

        self.pc
            .set((sim.get_logic_value_by_idx(self.pc_idx) as u32) & self.pc_phys_mask);

        // Sprint 369 (Gate B.2): software PC authority for the extended address space.
        // The physical bank-group machinery below masks pc_u8 & 0x7F, which aliases
        // PC>=128 (e.g. 200 -> 72) and would corrupt the PC. For the extended range
        // (and, in extended mode, all PCs) compute the next PC in software with the
        // full 8-bit mask and write it — the documented gated fallback (PC sequencing
        // is software; decode/ALU/writeback remain physical via the injected IR).
        if self.extended_pc {
            let pcm = self.pc_phys_mask;
            let pc_full = latch.pc & pcm;
            let fall_through = pc_full.wrapping_add(1) & pcm;
            let reg_indirect = (latch.ir_ext & EXT_REG_INDIRECT) != 0;
            let expected_pc = if reg_indirect && (branch_kind == 1 || branch_kind == 6) {
                // JMPR (kind 1) / CALLR (kind 6): target = regs[rd] (full width).
                let rd_eff =
                    effective_reg(latch.rd & 0x07, (latch.ir_ext & EXT_RD_HI) != 0) as usize;
                (self.regs[rd_eff].get() as u32) & pcm
            } else if branch_kind == 7 {
                // RET: target = LR (software mirror, kept in sync above).
                self.lr.get() & pcm
            } else {
                // Sprint 371 (Gate B.3.1): JMP.W / CALL.W target the full 16-bit
                // address via the wide immediate (`imm_val`). For non-wide branches
                // `imm_val == imm8`, so existing extended-mode behavior is unchanged.
                // Control transfers in the wide range take the software target write
                // below (ir_low is 8-bit; physical wide *target* delivery is deferred).
                Self::compute_branch_pc(
                    branch_kind,
                    flag_z_before,
                    carry_before,
                    (imm_val as u32) & pcm,
                    fall_through,
                    self.lr.get(),
                )
            };
            // Sprint 370 (Gate B.3): physical fall-through authority for the wide PC.
            // On non-control-transfer cycles the physical Register64 PC + physical
            // pc_plus_one adder + wide mask-via carry PC+1 with no software write
            // (override_enable=0 selects the fall-through input). Trust the physical
            // PC, verify against software, fall back on mismatch (Sprint 224 model).
            // Control transfers (branches, JMPR/CALLR/RET) keep the software target
            // write — physical wide target delivery is deferred (ir_low is 8-bit).
            let is_control_transfer = branch_kind != 0 || reg_indirect;
            if self.wide_pc && !is_control_transfer {
                self.pc_override_checks
                    .set(self.pc_override_checks.get() + 1);
                if self.pc.get() != expected_pc {
                    self.pc_override_mismatches
                        .set(self.pc_override_mismatches.get() + 1);
                    self.write_pc(sim, expected_pc);
                }
                // else: physical PC is authoritative — no software write.
            } else {
                self.write_pc(sim, expected_pc);
            }
        } else if (latch.pc >> 6) & 1 == 1 {
            // Sprint 187: Override physical PC for upper bank group.
            // Sprint 224: Physical PC authority for upper bank. The Const-swap injection
            // at L2(ox+3, oy+4) ensures the Target Selection Mux receives correct ir_low
            // for branch targets. Physical PC is trusted — verification counters track
            // any residual mismatches for diagnostics only.
            let target = (imm8 as u32) & 0x7F;
            let fall_through = (pc_u8.wrapping_add(1)) & 0x7F;
            let expected_pc = Self::compute_branch_pc(
                branch_kind,
                flag_z_before,
                carry_before,
                target,
                fall_through,
                self.lr.get(),
            );
            if self.physical_decode.get() {
                // Sprint 224: Physical PC authority — trust physical PC, verify only.
                self.pc_override_checks
                    .set(self.pc_override_checks.get() + 1);
                if self.pc.get() != expected_pc {
                    self.pc_override_mismatches
                        .set(self.pc_override_mismatches.get() + 1);
                    self.pc_mismatch_per_kind[branch_kind as usize]
                        .set(self.pc_mismatch_per_kind[branch_kind as usize].get() + 1);
                    // Mismatch fallback: override with software PC to maintain correctness.
                    // Expected: zero mismatches after Sprint 224 ir_low Target Mux fix.
                    self.write_pc(sim, expected_pc);
                }
            } else {
                self.write_pc(sim, expected_pc);
            }
        }

        // Sprint 196: Independent PC verification for synth branch replacement.
        // Compare synth-predicted PC against the actual physical PC post-clock-edge.
        // Only for PC < 64 (upper bank group uses independent software override above).
        if self.synth_branch.is_enabled() && (latch.pc >> 6) & 1 == 0 {
            // Sprint 202: synth verification — validate synth-predicted PC against physical.
            let synth_taken = if let Some(ref block) = self.synth_branch_block {
                let inputs =
                    Self::compute_synth_branch_inputs(branch_kind, flag_z_before, carry_before);
                let outputs = crate::synth::integration::drive_synth_block(sim, block, &inputs);
                outputs[0] != 0
            } else {
                let sel = (branch_kind as usize)
                    | ((flag_z_before as usize) << 3)
                    | ((carry_before as usize) << 4);
                self.synth_branch_table[sel]
            };
            let target = (imm8 as u32) & 0x7F;
            let fall_through = (pc_u8.wrapping_add(1)) & 0x7F;
            let expected_pc = if synth_taken {
                Self::compute_branch_pc(
                    branch_kind,
                    flag_z_before,
                    carry_before,
                    target,
                    fall_through,
                    self.lr.get(),
                )
            } else {
                fall_through
            };
            self.synth_branch.record_check();
            let actual_pc = self.pc.get();
            if actual_pc != expected_pc {
                self.synth_branch.record_mismatch();
            }
        }

        // Sprint 154: store sub-stage timing (stage_f/x set in tick()).
        let prev = self.last_stage_timing.get();
        self.last_stage_timing.set(V2StageTiming {
            stage_f_ns: prev.stage_f_ns,
            stage_x_ns: prev.stage_x_ns,
            branch_ns: branch_elapsed.as_nanos() as u64,
            commit_ns: commit_elapsed.as_nanos() as u64,
            clock_ns: clock_elapsed.as_nanos() as u64,
            ram_ns: ram_elapsed.as_nanos() as u64,
        });

        stats
    }

    fn snapshot_regs(&self) -> [u64; 16] {
        std::array::from_fn(|i| self.regs[i].get())
    }

    fn snapshot_ram(&self) -> [u64; 128] {
        std::array::from_fn(|i| self.ram[i].get())
    }

    fn compose_word_from_latch(latch: PipelineLatch) -> u32 {
        let low16 = ((latch.opcode as u16 & 0x1F) << 11)
            | ((latch.rd as u16 & 0x07) << 8)
            | latch.ir_low as u16;
        ((latch.ir_ext as u32) << 16) | low16 as u32
    }

    fn append_trace_entry(
        &self,
        trace: &mut V2TraceLog,
        latch: PipelineLatch,
        regs_before: &[u64; 16],
        ram_before: &[u64; 128],
    ) {
        let regs_after = self.snapshot_regs();
        let ram_after = self.snapshot_ram();

        let mut reg_writes = Vec::new();
        for idx in 0..16usize {
            if regs_before[idx] != regs_after[idx] {
                reg_writes.push(V2TraceRegWrite {
                    reg: idx as u8,
                    value: regs_after[idx],
                });
            }
        }

        let mut ram_writes = Vec::new();
        for idx in 0..128usize {
            if ram_before[idx] != ram_after[idx] {
                ram_writes.push(V2TraceMemEvent {
                    addr: idx as u8,
                    value: ram_after[idx],
                });
            }
        }

        let mmio_reads = self
            .last_stage_x_mmio_reads
            .borrow()
            .iter()
            .map(|&(addr, value)| V2TraceMemEvent { addr, value })
            .collect();
        let mmio_writes = self
            .last_stage_x_mmio_writes
            .borrow()
            .iter()
            .map(|&(addr, value)| V2TraceMemEvent { addr, value })
            .collect();

        let word = Self::compose_word_from_latch(latch);
        let counters = self.read_hybrid_assist_counters();
        trace.push(V2TraceEntry {
            cycle: self.cycle_count.get(),
            retired: self.retired_count.get().saturating_sub(1),
            pc: latch.pc & self.pc_phys_mask,
            ir_low: latch.ir_low,
            ir_ext: latch.ir_ext,
            word,
            asm: disassemble_v2_word(word),
            flag_z: self.flag_z.get(),
            flag_c: self.flag_c.get(),
            reg_writes,
            ram_writes,
            mmio_reads,
            mmio_writes,
            stage_f_bank_switches: Some(counters.stage_f_bank_switches),
            stage_f_mixed_dual_capture: Some(counters.stage_f_mixed_dual_capture),
            stage_x_mixed_software: Some(counters.stage_x_mixed_software),
            ram_high_bank_read_swaps: Some(counters.ram_high_bank_read_swaps),
            rom_upper_bank_group_select: Some(counters.rom_upper_bank_group_select),
        });
    }

    /// Execute one V2 tick.
    pub fn tick(&self, sim: &mut Simulation) -> TimingStats {
        if self.halted.get() {
            self.last_stage_x_valid.set(false);
            return TimingStats::default();
        }

        self.last_stage_x_mmio_reads.borrow_mut().clear();
        self.last_stage_x_mmio_writes.borrow_mut().clear();

        let next_cycle = self.cycle_count.get() + 1;
        if let Some(mmio) = &self.mmio {
            mmio.device().tick(next_cycle);
        }
        self.cycle_count.set(next_cycle);

        let latch_in = self.latch.get();
        self.last_stage_x_valid.set(latch_in.valid);
        if latch_in.valid {
            self.retired_count.set(self.retired_count.get() + 1);
        }

        let x_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        let x_stats = if latch_in.valid {
            self.run_stage_x(sim, latch_in)
        } else {
            self.changed_regs_mask.set(0);
            StageStats::empty()
        };
        let x_elapsed = x_start.map(|s| s.elapsed()).unwrap_or_default();

        let f_start = if self.enable_stage_timing {
            Some(Instant::now())
        } else {
            None
        };
        let (next_latch, f_stats) = if self.halted.get() {
            (PipelineLatch::default(), StageStats::empty())
        } else {
            self.run_stage_f(sim)
        };
        let f_elapsed = f_start.map(|s| s.elapsed()).unwrap_or_default();
        self.latch.set(next_latch);

        // Sprint 154: store stage-level timing (sub-stage timing set in run_stage_x).
        let prev = self.last_stage_timing.get();
        self.last_stage_timing.set(V2StageTiming {
            stage_f_ns: f_elapsed.as_nanos() as u64,
            stage_x_ns: x_elapsed.as_nanos() as u64,
            branch_ns: prev.branch_ns,
            commit_ns: prev.commit_ns,
            clock_ns: prev.clock_ns,
            ram_ns: prev.ram_ns,
        });

        let stage_f_comb = f_stats.comb_deltas;
        let stage_x_comb = x_stats.comb_deltas;
        let clock_edge_deltas = x_stats.clock_deltas;
        let projected_deltas = stage_f_comb.max(stage_x_comb);
        let total_deltas = stage_f_comb + stage_x_comb + clock_edge_deltas;
        let total_eval = f_stats.comb_eval + x_stats.comb_eval + x_stats.clock_eval;
        let total_switched = f_stats.comb_switched + x_stats.comb_switched + x_stats.clock_switched;

        TimingStats {
            critical_path_deltas: projected_deltas,
            critical_path_endpoint: None,
            tiles_switched: total_switched,
            tiles_evaluated: total_eval,
            total_deltas,
            glitches_detected: x_stats.glitches,
            converged: f_stats.converged && x_stats.converged,
            ..Default::default()
        }
    }

    pub fn step(&self, sim: &mut Simulation) -> TimingStats {
        self.tick(sim)
    }

    pub fn step_with_trace(&self, sim: &mut Simulation, trace: &mut V2TraceLog) -> TimingStats {
        let latch_in = self.latch.get();
        let should_capture = latch_in.valid;
        let regs_before = if should_capture {
            Some(self.snapshot_regs())
        } else {
            None
        };
        let ram_before = if should_capture {
            Some(self.snapshot_ram())
        } else {
            None
        };

        let timing = self.tick(sim);
        if should_capture && self.last_stage_x_valid.get() {
            self.append_trace_entry(
                trace,
                latch_in,
                regs_before.as_ref().expect("present when should_capture"),
                ram_before.as_ref().expect("present when should_capture"),
            );
        }
        timing
    }

    /// Returns whether the most recent `tick` retired a valid Stage-X instruction.
    pub fn last_stage_x_valid(&self) -> bool {
        self.last_stage_x_valid.get()
    }

    pub fn read_cycle_count(&self) -> u64 {
        self.cycle_count.get()
    }

    pub fn read_retired_count(&self) -> u64 {
        self.retired_count.get()
    }

    fn breakpoint_matches(&self, sim: &Simulation, breakpoint: &V2DebugBreakpoint) -> bool {
        match *breakpoint {
            V2DebugBreakpoint::Pc(target) => self.read_pc(sim) == ((target as u32) & 0x7F),
            V2DebugBreakpoint::RegEquals { reg, value } => {
                reg < 16 && self.read_reg(sim, reg) == value
            }
            V2DebugBreakpoint::Halted => self.is_halted(),
        }
    }

    pub fn continue_until(
        &self,
        sim: &mut Simulation,
        max_cycles: u64,
        breakpoints: &[V2DebugBreakpoint],
    ) -> V2DebugRunResult {
        for cycle in 0..max_cycles {
            let _ = self.tick(sim);

            if let Some((idx, _)) = breakpoints
                .iter()
                .enumerate()
                .find(|(_, bp)| self.breakpoint_matches(sim, bp))
            {
                return V2DebugRunResult {
                    cycles: cycle + 1,
                    reason: V2DebugStopReason::Breakpoint(idx),
                };
            }

            if self.is_halted() {
                return V2DebugRunResult {
                    cycles: cycle + 1,
                    reason: V2DebugStopReason::Halted,
                };
            }
        }

        V2DebugRunResult {
            cycles: max_cycles,
            reason: V2DebugStopReason::MaxCycles,
        }
    }

    pub fn continue_until_with_trace(
        &self,
        sim: &mut Simulation,
        max_cycles: u64,
        breakpoints: &[V2DebugBreakpoint],
        trace: &mut V2TraceLog,
    ) -> V2DebugRunResult {
        for cycle in 0..max_cycles {
            let _ = self.step_with_trace(sim, trace);

            if let Some((idx, _)) = breakpoints
                .iter()
                .enumerate()
                .find(|(_, bp)| self.breakpoint_matches(sim, bp))
            {
                return V2DebugRunResult {
                    cycles: cycle + 1,
                    reason: V2DebugStopReason::Breakpoint(idx),
                };
            }

            if self.is_halted() {
                return V2DebugRunResult {
                    cycles: cycle + 1,
                    reason: V2DebugStopReason::Halted,
                };
            }
        }

        V2DebugRunResult {
            cycles: max_cycles,
            reason: V2DebugStopReason::MaxCycles,
        }
    }

    pub fn debug_snapshot(&self, sim: &Simulation) -> V2DebugSnapshot {
        let regs = std::array::from_fn(|i| self.read_reg(sim, i));
        let latch = self.latch.get();
        V2DebugSnapshot {
            pc: self.read_pc(sim),
            ir_low: latch.ir_low,
            ir_ext: latch.ir_ext,
            flag_z: self.read_flag_z(sim),
            flag_c: self.read_flag_c(sim),
            halted: self.is_halted(),
            regs,
        }
    }

    pub fn run(&self, sim: &mut Simulation, max_cycles: u64) -> TileCpuMetrics {
        let mut metrics = TileCpuMetrics::default();
        let mut total_critical = 0u64;

        for _ in 0..max_cycles {
            if self.halted.get() {
                break;
            }
            let stats = self.tick(sim);
            metrics.cycles += 1;
            if self.last_stage_x_valid.get() {
                metrics.instructions_executed += 1;
            }
            metrics.total_deltas += stats.total_deltas as u64;
            metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
            metrics.total_tiles_switched += stats.tiles_switched as u64;
            // In V2 pipeline mode this is projected combinational stage max.
            metrics.max_critical_path = metrics.max_critical_path.max(stats.critical_path_deltas);
            total_critical += stats.critical_path_deltas as u64;
            if !stats.converged {
                metrics.had_timing_violation = true;
            }
        }

        if metrics.cycles > 0 {
            metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
            metrics.ipc = metrics.instructions_executed as f64 / metrics.cycles as f64;
            if metrics.max_critical_path > 0 {
                metrics.estimated_max_mhz = 1000.0 / metrics.max_critical_path as f64;
            }
        }

        metrics
    }

    /// Return the number of tiles in each zone scope (for instrumentation).
    /// Sprint 163: ram_read and ram_write scopes removed (dead since Sprint 162).
    pub fn zone_scope_sizes(&self) -> [(&'static str, usize); 3] {
        [
            ("pipeline", self.pipeline_scope.len()),
            ("branch", self.branch_scope.len()),
            ("commit", self.commit_scope.len()),
        ]
    }

    pub fn read_hybrid_assist_counters(&self) -> V2HybridAssistCounters {
        V2HybridAssistCounters {
            stage_f_bank_switches: self.hybrid_stage_f_bank_switches.get(),
            stage_f_mixed_dual_capture: self.hybrid_stage_f_mixed_dual_capture.get(),
            stage_x_mixed_software: self.hybrid_stage_x_mixed_software.get(),
            ram_high_bank_read_swaps: self.hybrid_ram_high_bank_read_swaps.get(),
            rom_upper_bank_group_select: self.hybrid_rom_upper_bank_group_select.get(),
        }
    }

    /// Sprint 154: Read per-stage wall-clock timing from the last tick.
    pub fn read_last_stage_timing(&self) -> V2StageTiming {
        self.last_stage_timing.get()
    }

    /// Sprint 167: Enable/disable per-stage timing measurement.
    /// When disabled (default), all `Instant::now()` calls are skipped and
    /// `V2StageTiming` reports zeros. Benchmarks should enable this.
    pub fn set_stage_timing(&mut self, enabled: bool) {
        self.enable_stage_timing = enabled;
    }

    /// Sprint 288: Enable/disable the cone convergence probe independently
    /// of stage timing. The probe re-evaluates all cone ops as a shadow pass
    /// (~1,223 ops) to verify single-pass convergence.
    pub fn set_convergence_probe(&mut self, enabled: bool) {
        self.enable_convergence_probe = enabled;
    }

    pub fn read_reg(&self, _sim: &Simulation, reg: usize) -> u64 {
        if reg < 16 { self.regs[reg].get() } else { 0 }
    }

    pub fn read_operand_a_root(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.op_a_root_idx)
    }

    pub fn read_operand_b_root(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.op_b_root_idx)
    }

    pub fn read_extract_opcode_shr(&self, sim: &Simulation) -> u8 {
        (sim.get_logic_value_by_idx(self.extract_opcode_shr_idx) & 0x1F) as u8
    }

    pub fn read_extract_opcode_bit4(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.extract_opcode_bit4_idx) != 0
    }

    pub fn read_extract_rd(&self, sim: &Simulation) -> u8 {
        let b0 = u8::from(sim.get_logic_value_by_idx(self.extract_rd_bit_indices[0]) != 0);
        let b1 = u8::from(sim.get_logic_value_by_idx(self.extract_rd_bit_indices[1]) != 0);
        let b2 = u8::from(sim.get_logic_value_by_idx(self.extract_rd_bit_indices[2]) != 0);
        b0 | (b1 << 1) | (b2 << 2)
    }

    pub fn read_extract_rs_field(&self, sim: &Simulation) -> u8 {
        (sim.get_logic_value_by_idx(self.extract_rs_field_idx) & 0x07) as u8
    }

    pub fn read_we_mask_source(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.we_mask_const_idx)
    }

    pub fn read_flag_we_mask_source(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.flag_we_mask_const_idx)
    }

    pub fn read_branch_taken_core(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.branch_taken_core_idx) != 0
    }

    // Sprint 195: physical Decoder3to8 output (one-hot of rd field).
    // After step(), this tile reflects the current instruction's rd field
    // (part of pipeline_dirty_indices, propagated during step).
    pub fn read_physical_rd_decode(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.rd_onehot_decode_idx)
    }

    // Sprint 194: physical signal accessors for synth live shadow.
    // Unlike read_flag_z()/read_flag_c() which read from software cache,
    // these read from the physical tiles — what the Mux16to1 actually sees.

    pub fn read_physical_flag_z(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_z_idx) != 0
    }

    pub fn read_physical_flag_c(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_c_idx) != 0
    }

    pub fn read_physical_ctrl_b(&self, sim: &Simulation) -> u8 {
        (sim.get_logic_value_by_idx(self.ctrl_b_mux_idx) & 0x07) as u8
    }

    /// Sprint 218: Read physical ctrl_a value from tile.
    pub fn read_physical_ctrl_a(&self, sim: &Simulation) -> u8 {
        (sim.get_logic_value_by_idx(self.ctrl_a_mux_idx) & 0xFF) as u8
    }

    /// Sprint 218: Read physical ALU result mux output.
    pub fn read_physical_alu_result(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.wb_alu_root_idx)
    }

    /// Sprint 218: Read physical wb_data mux output (final writeback value).
    pub fn read_physical_wb_data(&self, sim: &Simulation) -> u64 {
        sim.get_logic_value_by_idx(self.wb_data_mux_idx)
    }

    /// Re-evaluate the physical branch-taken path with current tile values
    /// and return a consistent (ctrl_b, flag_z, flag_c, branch_taken) snapshot.
    ///
    /// After `step()`, branch_taken_core may reflect a stale evaluation because
    /// Stage-X commits new flags and Stage-F decodes new ctrl_b, but the branch
    /// scope is not re-propagated with those new values. This method forces
    /// re-propagation to get a consistent result.
    pub fn snapshot_branch_physical(&self, sim: &mut Simulation) -> (u8, bool, bool, bool) {
        sim.dirty.mark_dirty(self.branch_ctrl_b_l1_tap_idx);
        sim.dirty.mark_dirty(self.branch_flag_z_l1_tap_idx);
        sim.dirty.mark_dirty(self.branch_flag_c_l1_tap_idx);
        self.mark_branch_dirty(sim);
        // Sprint 263: Levelized evaluation for branch scope.
        let _ = sim.propagate_levelized(&self.branch_eval_order);

        let ctrl_b = (sim.get_logic_value_by_idx(self.ctrl_b_mux_idx) & 0x07) as u8;
        let flag_z = sim.get_logic_value_by_idx(self.flag_z_idx) != 0;
        let flag_c = sim.get_logic_value_by_idx(self.flag_c_idx) != 0;
        let taken = sim.get_logic_value_by_idx(self.branch_taken_core_idx) != 0;
        (ctrl_b, flag_z, flag_c, taken)
    }

    /// Sprint 196: Compute (x, y, z) from flat tile index using simulation grid dimensions.
    /// tile_count is placed-tile count (NOT layer_size), so we must use sim dimensions.
    fn idx_to_xyz(sim: &Simulation, idx: usize) -> (usize, usize, usize) {
        let w = sim.width();
        let layer_size = w * sim.height();
        let z = idx / layer_size;
        let rem = idx % layer_size;
        let y = rem / w;
        let x = rem % w;
        (x, y, z)
    }

    // ---- Sprint 202: Factored helpers for branch/store computation ----

    /// Pure computation: resolve branch_kind + flags to a next-PC value.
    /// Used by upper-bank override (which writes the result) and synth
    /// verification (which compares the result against physical PC).
    fn compute_branch_pc(
        branch_kind: u8,
        flag_z: bool,
        carry: bool,
        target: u32,
        fall_through: u32,
        lr: u32,
    ) -> u32 {
        match branch_kind {
            1 | 6 => target, // JMP, CALL
            2 => {
                if flag_z {
                    target
                } else {
                    fall_through
                }
            } // JZ
            3 => {
                if !flag_z {
                    target
                } else {
                    fall_through
                }
            } // JNZ
            4 => {
                if carry {
                    target
                } else {
                    fall_through
                }
            } // JC
            5 => {
                if !carry {
                    target
                } else {
                    fall_through
                }
            } // JNC
            7 => lr & 0x7F,  // RET
            _ => fall_through, // no branch
        }
    }

    /// Pack branch_kind bits + flags into the 5-element u64 input array
    /// expected by the synth branch-taken block.
    fn compute_synth_branch_inputs(branch_kind: u8, flag_z: bool, carry: bool) -> [u64; 5] {
        [
            if (branch_kind & 1) != 0 { u64::MAX } else { 0 },
            if (branch_kind & 2) != 0 { u64::MAX } else { 0 },
            if (branch_kind & 4) != 0 { u64::MAX } else { 0 },
            if flag_z { u64::MAX } else { 0 },
            if carry { u64::MAX } else { 0 },
        ]
    }

    /// Compute one-hot RAM decode for a store address (low 3 bits).
    /// Returns 0 for non-RAM addresses (MMIO or out-of-range).
    fn compute_store_onehot(store_addr: usize) -> u64 {
        if store_addr < 128 && !is_v2_mmio_addr(store_addr) {
            1u64 << (store_addr & 7)
        } else {
            0u64
        }
    }

    // ---- Sprint 199: Factored synth injection/verification helpers ----

    /// Inject rd_decode and ram_decode one-hot values before commit settle.
    fn inject_synth_pre_commit(&self, sim: &mut Simulation, latch: &PipelineLatch, opcode: u8) {
        // Sprint 197: rd_decode — inject one-hot WE into Const tile + L0 chain.
        // Sprint 200: use live synth block evaluation if available.
        // Sprint 233: physical_rd_decode — read physical Decoder3to8, inject only on mismatch.
        if self.synth_rd_decode.is_enabled() {
            let rd = latch.rd & 0x07;
            let expected_onehot = 1u64 << rd;

            let bank_group = (latch.pc >> 6) & 1;
            if self.physical_rd_decode.get() && (bank_group == 0 || self.physical_ir_spine.get()) {
                // Sprint 233/256: Physical Decoder3to8 is live — read its output.
                // Bank 0: always correct (physical IR extraction).
                // Upper banks: correct when physical_ir_spine delivers IR physically.
                let phys = sim.get_logic_value_by_idx(self.rd_onehot_decode_idx);
                self.rd_decode_authority_checks
                    .set(self.rd_decode_authority_checks.get() + 1);
                if phys != expected_onehot {
                    self.rd_decode_authority_mismatches
                        .set(self.rd_decode_authority_mismatches.get() + 1);
                    // Fallback: inject expected value.
                    sim.set_logic_value_by_idx(self.rd_onehot_decode_idx, expected_onehot);
                    for &idx in &self.rd_decode_l0_chain {
                        sim.set_logic_value_by_idx(idx, expected_onehot);
                    }
                }
                // Physical value (or fallback) is already in place; propagate downstream.
                sim.dirty.mark_dirty(self.we_mask_const_idx);
            } else if self.physical_rd_decode.get() && !self.physical_ir_spine.get() {
                // Upper bank without spine: inject expected (physical tile carries stale data).
                sim.set_logic_value_by_idx(self.rd_onehot_decode_idx, expected_onehot);
                for &idx in &self.rd_decode_l0_chain {
                    sim.set_logic_value_by_idx(idx, expected_onehot);
                }
                sim.dirty.mark_dirty(self.we_mask_const_idx);
            } else {
                // Legacy path: software-computed one-hot injected into Const tile.
                let onehot = if let Some(ref block) = self.synth_rd_decode_block {
                    let rd_val = rd as usize;
                    let inputs = [
                        if (rd_val & 1) != 0 { u64::MAX } else { 0 },
                        if (rd_val & 2) != 0 { u64::MAX } else { 0 },
                        if (rd_val & 4) != 0 { u64::MAX } else { 0 },
                    ];
                    let outputs = crate::synth::integration::drive_synth_block(sim, block, &inputs);
                    let mut oh = 0u64;
                    for (i, &val) in outputs.iter().enumerate() {
                        if val != 0 {
                            oh |= 1 << i;
                        }
                    }
                    oh
                } else {
                    expected_onehot
                };
                sim.set_logic_value_by_idx(self.rd_onehot_decode_idx, onehot);
                for &idx in &self.rd_decode_l0_chain {
                    sim.set_logic_value_by_idx(idx, onehot);
                }
                sim.dirty.mark_dirty(self.we_mask_const_idx);
            }
            self.synth_rd_decode.record_check();
        }
        // Sprint 198: ram_decode — inject one-hot WE into Const tile.
        // Sprint 201: use live synth block evaluation if available.
        if self.synth_ram_decode.is_enabled() {
            let is_store = opcode == 0x17 || opcode == 0x19;
            let onehot = if is_store {
                let store_addr = self.compute_mem_addr(latch, opcode, latch.b);
                if store_addr < 128 && !is_v2_mmio_addr(store_addr) {
                    if let Some(ref block) = self.synth_ram_decode_block {
                        let addr_val = (store_addr & 7) as usize;
                        let inputs = [
                            if (addr_val & 1) != 0 { u64::MAX } else { 0 },
                            if (addr_val & 2) != 0 { u64::MAX } else { 0 },
                            if (addr_val & 4) != 0 { u64::MAX } else { 0 },
                        ];
                        let outputs =
                            crate::synth::integration::drive_synth_block(sim, block, &inputs);
                        let mut oh = 0u64;
                        for (i, &val) in outputs.iter().enumerate() {
                            if val != 0 {
                                oh |= 1 << i;
                            }
                        }
                        oh
                    } else {
                        Self::compute_store_onehot(store_addr)
                    }
                } else {
                    0u64
                }
            } else {
                0u64
            };
            sim.set_logic_value_by_idx(self.ram_write_decode_idx, onehot);
            sim.dirty.mark_dirty(self.ram_write_gate_idx);
            self.synth_ram_decode.record_check();
        }

        // Sprint 205: ctrl_a authority — inject we_mask and flag_we_mask.
        // Must come AFTER rd_decode injection (we_mask uses rd_onehot).
        // Sprint 209: combined_decode_lut provides ctrl_a directly when available.
        // Sprint 222/223: physical_decode reads ctrl_a from physical Mux16to1 for
        // all banks (S223 extended from bank0-only), with LUT fallback on mismatch.
        // Injection into Const tiles unchanged.
        if self.synth_ctrl_a.is_enabled() {
            let bank_group = (latch.pc >> 6) & 1;
            let ctrl_a = if self.physical_decode.get() {
                // Sprint 222/223: Physical ctrl_a authority.
                let phys_ca = sim.get_logic_value_by_idx(self.ctrl_a_mux_idx) as u8;
                self.decode_ctrl_a_checks
                    .set(self.decode_ctrl_a_checks.get() + 1);
                let expected_ca = if let Some(lut) = self.combined_decode_lut {
                    (lut[opcode as usize] & 0xFF) as u8
                } else {
                    CTRL_A_LUT[opcode as usize]
                };
                if phys_ca != expected_ca {
                    self.decode_ctrl_a_mismatches
                        .set(self.decode_ctrl_a_mismatches.get() + 1);
                    expected_ca
                } else {
                    phys_ca
                }
            } else if let Some(lut) = self.combined_decode_lut {
                // Sprint 209: extract ctrl_a from combined LUT (bits [7:0]).
                (lut[opcode as usize] & 0xFF) as u8
            } else if let Some(ref block) = self.synth_ctrl_a_block {
                let op_val = opcode as usize;
                let inputs: [u64; 5] = [
                    if (op_val) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 1) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 2) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 3) & 1 != 0 { u64::MAX } else { 0 },
                    if (op_val >> 4) & 1 != 0 { u64::MAX } else { 0 },
                ];
                let outputs = crate::synth::integration::drive_synth_block(sim, block, &inputs);
                let mut ca_val = 0u8;
                for (i, &val) in outputs.iter().enumerate() {
                    if val != 0 {
                        ca_val |= 1 << i;
                    }
                }
                ca_val
            } else {
                CTRL_A_LUT[opcode as usize]
            };
            let reg_write = (ctrl_a >> 3) & 1 != 0;

            // we_mask: rd_onehot gated by reg_write bit.
            // Sprint 242: Suppress low-half WE when rd_hi=true. The physical WE
            // decoder only sees rd[0:2], so it fires for R(rd_eff & 7), corrupting
            // the aliased low register. Force WE=0 for high-reg writes; the high-reg
            // merge mux injection path (Sprint 236/238) handles R8-R15 writeback.
            let rd_hi = (latch.ir_ext & EXT_RD_HI) != 0;
            let rd_onehot = 1u64 << (latch.rd & 0x07);
            let expected_we_mask = if reg_write && !rd_hi { rd_onehot } else { 0 };
            // Sprint 242: When rd_hi=true, skip the physical authority comparison.
            // The physical And tile correctly computes rd_onehot & reg_write (it can't
            // see rd_hi — that's a known wiring limitation, not a tile bug). The actual
            // WE suppression is handled by the save/restore tile-type swap in run_stage_x.
            if self.physical_we_mask.get()
                && (bank_group == 0 || self.physical_ir_spine.get())
                && !rd_hi
            {
                // Sprint 233/256: Physical And tile is live — read its output.
                let phys = sim.get_logic_value_by_idx(self.we_mask_const_idx);
                self.we_mask_authority_checks
                    .set(self.we_mask_authority_checks.get() + 1);
                if phys != expected_we_mask {
                    self.we_mask_authority_mismatches
                        .set(self.we_mask_authority_mismatches.get() + 1);
                    sim.set_logic_value_by_idx(self.we_mask_const_idx, expected_we_mask);
                    sim.dirty.mark_dirty(self.we_mask_const_idx);
                }
            } else {
                sim.set_logic_value_by_idx(self.we_mask_const_idx, expected_we_mask);
                sim.dirty.mark_dirty(self.we_mask_const_idx);
            }

            // flag_we_mask: ctrl_a[4:5] (bit0=Z_WE, bit1=C_WE)
            // Sprint 246: MUL C_WE suppression. MUL shares opcode 0x04 (ADD) which
            // has C_WE=1, but MUL's carry should be unchanged. Override to Z_WE only
            // (0x01) so the physical flag_c register doesn't capture the ADD carry.
            let is_mul = opcode == 0x04 && (latch.ir_low & 0x1F) == 1;
            let expected_flag_we = if is_mul && self.physical_sub_fn_flag_authority.get() {
                ((ctrl_a >> 4) & 0x03 & 0x01) as u64 // Z_WE only, suppress C_WE
            } else {
                ((ctrl_a >> 4) & 0x03) as u64
            };
            if self.physical_flag_we_mask.get() && (bank_group == 0 || self.physical_ir_spine.get())
            {
                // Sprint 233/256: Physical WeightedViaUp tile is live — read its output.
                // Bank 0: always correct. Upper banks: correct when spine delivers IR.
                let phys = sim.get_logic_value_by_idx(self.flag_we_mask_const_idx);
                // Authority check against raw ctrl_a decode (what the physical Mux
                // correctly produces). MUL C_WE suppression is a sub-function override
                // applied after decode — not a decode failure.
                let raw_flag_we = ((ctrl_a >> 4) & 0x03) as u64;
                self.flag_we_mask_authority_checks
                    .set(self.flag_we_mask_authority_checks.get() + 1);
                if phys != raw_flag_we {
                    self.flag_we_mask_authority_mismatches
                        .set(self.flag_we_mask_authority_mismatches.get() + 1);
                }
                // Inject the (possibly MUL-suppressed) value for downstream flag capture.
                if phys != expected_flag_we {
                    sim.set_logic_value_by_idx(self.flag_we_mask_const_idx, expected_flag_we);
                    sim.dirty.mark_dirty(self.flag_we_mask_const_idx);
                }
            } else {
                sim.set_logic_value_by_idx(self.flag_we_mask_const_idx, expected_flag_we);
                sim.dirty.mark_dirty(self.flag_we_mask_const_idx);
            }

            self.synth_ctrl_a.record_check();
            // Dual-path verification when physical tiles are valid (bank_group==0).
            // Sprint 222: skip when physical_decode is active (comparison already done above).
            if bank_group == 0 && !self.physical_decode.get() {
                let physical_ca = sim.get_logic_value_by_idx(self.ctrl_a_mux_idx) as u8;
                if ctrl_a != physical_ca {
                    self.synth_ctrl_a.record_mismatch();
                }
            }
        }
    }

    /// Verify rd_decode and ram_decode Const tile readback after commit settle.
    fn verify_synth_post_commit(&self, sim: &Simulation, latch: &PipelineLatch, opcode: u8) {
        // Sprint 197: rd_decode — verify injected one-hot survived commit settle.
        if self.synth_rd_decode.is_enabled() {
            let expected = 1u64 << (latch.rd & 0x07);
            let readback = sim.get_logic_value_by_idx(self.rd_onehot_decode_idx);
            if readback != expected {
                self.synth_rd_decode.record_mismatch();
            }
        }
        // Sprint 198: ram_decode — verify Const tile readback.
        if self.synth_ram_decode.is_enabled() {
            let is_store = opcode == 0x17 || opcode == 0x19;
            let expected = if is_store {
                let store_addr = self.compute_mem_addr(latch, opcode, latch.b);
                Self::compute_store_onehot(store_addr)
            } else {
                0u64
            };
            let readback = sim.get_logic_value_by_idx(self.ram_write_decode_idx);
            if readback != expected {
                self.synth_ram_decode.record_mismatch();
            }
        }
    }

    /// Sprint 196: Enable synth-driven branch-taken replacement.
    /// Swaps the physical Mux tile at branch_taken_core to Const,
    /// so injected synth values survive propagation.
    pub fn enable_synth_branch(&self, sim: &mut Simulation) {
        if self.synth_branch.is_enabled() {
            return; // idempotent
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.branch_taken_core_idx);
        let saved = sim.tile_type_3d(x, y, z);
        debug_assert_ne!(
            saved,
            TileType::Wire,
            "branch_taken_core should be Mux, not default Wire — \
             idx_to_xyz may be wrong: ({x}, {y}, {z}) from idx {}",
            self.branch_taken_core_idx
        );
        self.branch_taken_saved_tile_type.set(saved);
        sim.set_tile_3d(x, y, z, TileType::Const);
        sim.set_logic_value_by_idx(self.branch_taken_core_idx, 0);
        self.synth_branch.enable();
        self.compact_ops_stale.set(true);
    }

    /// Sprint 196: Disable synth-driven branch-taken replacement.
    /// Restores the original tile type at branch_taken_core.
    pub fn disable_synth_branch(&self, sim: &mut Simulation) {
        if !self.synth_branch.is_enabled() {
            return; // idempotent
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.branch_taken_core_idx);
        let saved = self.branch_taken_saved_tile_type.get();
        sim.set_tile_3d(x, y, z, saved);
        self.synth_branch.disable();
        self.compact_ops_stale.set(true);
    }

    /// Sprint 196: Read synth branch mismatch count (PC mismatches).
    pub fn synth_branch_mismatches(&self) -> u64 {
        self.synth_branch.mismatches()
    }

    /// Sprint 196: Read synth branch check count (instructions verified).
    pub fn synth_branch_checks(&self) -> u64 {
        self.synth_branch.checks()
    }

    /// Sprint 200: Set the synth-generated branch-taken block for live evaluation.
    /// Must be called before the execution loop (requires `&mut self`).
    pub fn set_synth_branch_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_branch_block = Some(block);
    }

    /// Sprint 200: Set the synth-generated rd_decode block for live evaluation.
    pub fn set_synth_rd_decode_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_rd_decode_block = Some(block);
    }

    /// Sprint 201: Set the synth-generated ram_decode block for live evaluation.
    pub fn set_synth_ram_decode_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_ram_decode_block = Some(block);
    }

    /// Sprint 201: Set the synth-generated ctrl_b block for live evaluation.
    pub fn set_synth_ctrl_b_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_ctrl_b_block = Some(block);
    }

    /// Sprint 205: Set the synth-generated ctrl_a block for authority (fallback: CTRL_A_LUT).
    pub fn set_synth_ctrl_a_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_ctrl_a_block = Some(block);
    }

    /// Sprint 209: Set the combined ctrl_a+ctrl_b lookup table.
    pub fn set_combined_decode_lut(&mut self, lut: [u16; 32]) {
        self.combined_decode_lut = Some(lut);
    }

    /// Sprint 209: Number of times the combined decode LUT was consulted.
    pub fn combined_decode_checks(&self) -> u64 {
        self.combined_decode_checks.get()
    }

    /// Sprint 201: Public accessor for CTRL_B_LUT (for cross-module parity tests).
    pub fn ctrl_b_lut() -> &'static [u8; 32] {
        &CTRL_B_LUT
    }

    pub fn ctrl_a_lut() -> &'static [u8; 32] {
        &CTRL_A_LUT
    }

    /// Sprint 196: Read tile type at branch_taken_core for verification.
    pub fn branch_taken_core_tile_type(&self, sim: &Simulation) -> TileType {
        let (x, y, z) = Self::idx_to_xyz(sim, self.branch_taken_core_idx);
        sim.tile_type_3d(x, y, z)
    }

    /// Sprint 197: Enable synth-driven rd_decode replacement.
    /// Swaps the physical Decoder3to8 tile at rd_onehot_decode_idx to Const.
    pub fn enable_synth_rd_decode(&self, sim: &mut Simulation) {
        if self.synth_rd_decode.is_enabled() {
            return;
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.rd_onehot_decode_idx);
        let saved = sim.tile_type_3d(x, y, z);
        debug_assert_ne!(
            saved,
            TileType::Wire,
            "rd_onehot_decode should be Decoder3to8, not default Wire — \
             idx_to_xyz may be wrong: ({x}, {y}, {z}) from idx {}",
            self.rd_onehot_decode_idx
        );
        self.rd_decode_saved_tile_type.set(saved);
        sim.set_tile_3d(x, y, z, TileType::Const);
        sim.set_logic_value_by_idx(self.rd_onehot_decode_idx, 0);
        self.synth_rd_decode.enable();
        self.compact_ops_stale.set(true);
    }

    /// Sprint 197: Disable synth-driven rd_decode replacement.
    pub fn disable_synth_rd_decode(&self, sim: &mut Simulation) {
        if !self.synth_rd_decode.is_enabled() {
            return;
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.rd_onehot_decode_idx);
        let saved = self.rd_decode_saved_tile_type.get();
        sim.set_tile_3d(x, y, z, saved);
        self.synth_rd_decode.disable();
        self.compact_ops_stale.set(true);
    }

    /// Sprint 197: Read synth rd_decode mismatch count.
    pub fn synth_rd_decode_mismatches(&self) -> u64 {
        self.synth_rd_decode.mismatches()
    }

    /// Sprint 197: Read synth rd_decode check count.
    pub fn synth_rd_decode_checks(&self) -> u64 {
        self.synth_rd_decode.checks()
    }

    /// Sprint 197: Read physical register value from Register64 tile.
    pub fn read_physical_reg(&self, sim: &Simulation, reg: usize) -> u64 {
        if reg < 16 {
            sim.get_logic_value_by_idx(self.reg_indices[reg])
        } else {
            0
        }
    }

    /// Sprint 197: Read tile type at rd_onehot_decode for verification.
    pub fn rd_decode_tile_type(&self, sim: &Simulation) -> TileType {
        let (x, y, z) = Self::idx_to_xyz(sim, self.rd_onehot_decode_idx);
        sim.tile_type_3d(x, y, z)
    }

    /// Sprint 197: Enable synth ctrl_b software authority.
    /// No physical tile swap — overrides latch.ctrl_b with CTRL_B_LUT[opcode].
    pub fn enable_synth_ctrl_b(&self) {
        if self.synth_ctrl_b.is_enabled() {
            return;
        }
        self.synth_ctrl_b.enable();
    }

    /// Sprint 197: Disable synth ctrl_b software authority.
    pub fn disable_synth_ctrl_b(&self) {
        self.synth_ctrl_b.disable();
    }

    /// Sprint 197: Read synth ctrl_b mismatch count.
    pub fn synth_ctrl_b_mismatches(&self) -> u64 {
        self.synth_ctrl_b.mismatches()
    }

    /// Sprint 197: Read synth ctrl_b check count.
    pub fn synth_ctrl_b_checks(&self) -> u64 {
        self.synth_ctrl_b.checks()
    }

    // ---- Sprint 205: Ctrl_A authority ----

    /// Sprint 205: Enable synth ctrl_a authority.
    /// Swaps we_mask (And→Const) and flag_we_mask (WeightedViaUp→Const, Sprint 206) tiles so that
    /// injected synth values are not overwritten by physical eval during settle.
    /// Values are injected each cycle in `inject_synth_pre_commit()`.
    pub fn enable_synth_ctrl_a(&self, sim: &mut Simulation) {
        if self.synth_ctrl_a.is_enabled() {
            return;
        }
        // Swap we_mask And tile to Const
        let (x, y, z) = Self::idx_to_xyz(sim, self.we_mask_const_idx);
        self.we_mask_saved_tile_type.set(sim.tile_type_3d(x, y, z));
        sim.set_tile_3d(x, y, z, TileType::Const);
        sim.set_logic_value_by_idx(self.we_mask_const_idx, 0);

        // Swap flag_we_mask WeightedViaUp tile to Const (Sprint 206: was ViaUp)
        let (x, y, z) = Self::idx_to_xyz(sim, self.flag_we_mask_const_idx);
        self.flag_we_mask_saved_tile_type
            .set(sim.tile_type_3d(x, y, z));
        sim.set_tile_3d(x, y, z, TileType::Const);
        sim.set_logic_value_by_idx(self.flag_we_mask_const_idx, 0);

        self.synth_ctrl_a.enable();
        self.compact_ops_stale.set(true);
    }

    /// Sprint 205: Disable synth ctrl_a authority — restore original tile types.
    pub fn disable_synth_ctrl_a(&self, sim: &mut Simulation) {
        if !self.synth_ctrl_a.is_enabled() {
            return;
        }
        // Restore we_mask tile
        let (x, y, z) = Self::idx_to_xyz(sim, self.we_mask_const_idx);
        sim.set_tile_3d(x, y, z, self.we_mask_saved_tile_type.get());

        // Restore flag_we_mask tile
        let (x, y, z) = Self::idx_to_xyz(sim, self.flag_we_mask_const_idx);
        sim.set_tile_3d(x, y, z, self.flag_we_mask_saved_tile_type.get());

        self.synth_ctrl_a.disable();
        self.compact_ops_stale.set(true);
    }

    pub fn synth_ctrl_a_mismatches(&self) -> u64 {
        self.synth_ctrl_a.mismatches()
    }

    pub fn synth_ctrl_a_checks(&self) -> u64 {
        self.synth_ctrl_a.checks()
    }

    // ---- Sprint 198: Operand bypass authority ----

    /// Enable synth operand bypass — always read from reg_indices instead of tree roots.
    /// Dual-path check compares direct register read vs physical tree root output.
    pub fn enable_synth_operand_bypass(&self) {
        if self.synth_operand.is_enabled() {
            return;
        }
        self.synth_operand.enable();
    }

    /// Disable synth operand bypass.
    pub fn disable_synth_operand_bypass(&self) {
        self.synth_operand.disable();
    }

    pub fn synth_operand_mismatches(&self) -> u64 {
        self.synth_operand.mismatches()
    }

    pub fn synth_operand_checks(&self) -> u64 {
        self.synth_operand.checks()
    }

    // ---- Sprint 231: Physical operand authority ----

    /// Enable physical operand authority — tree root reads are authoritative for R0-R7.
    /// Requires synth operand bypass to be enabled (for dual-path comparison).
    pub fn set_physical_operand_authority(&self, enabled: bool) {
        self.physical_operand_authority.set(enabled);
        self.op_authority_checks.set(0);
        self.op_authority_mismatches.set(0);
    }

    pub fn physical_operand_authority(&self) -> bool {
        self.physical_operand_authority.get()
    }

    pub fn op_authority_checks(&self) -> u64 {
        self.op_authority_checks.get()
    }

    pub fn op_authority_mismatches(&self) -> u64 {
        self.op_authority_mismatches.get()
    }

    // ---- Sprint 198: RAM write address decoder replacement ----

    /// Enable synth RAM decoder replacement. Swaps Decoder3to8 tile to Const.
    pub fn enable_synth_ram_decode(&self, sim: &mut Simulation) {
        if self.synth_ram_decode.is_enabled() {
            return;
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.ram_write_decode_idx);
        let old_type = sim.tile_type_3d(x, y, z);
        debug_assert_eq!(
            old_type,
            TileType::Decoder3to8,
            "RAM decoder tile at ({x},{y},{z}) idx {} expected Decoder3to8, got {:?}",
            self.ram_write_decode_idx,
            old_type
        );
        self.ram_decode_saved_tile_type.set(old_type);
        sim.set_tile_3d(x, y, z, TileType::Const);
        self.synth_ram_decode.enable();
        self.compact_ops_stale.set(true);
    }

    /// Disable synth RAM decoder replacement. Restores original tile type.
    pub fn disable_synth_ram_decode(&self, sim: &mut Simulation) {
        if !self.synth_ram_decode.is_enabled() {
            return;
        }
        let (x, y, z) = Self::idx_to_xyz(sim, self.ram_write_decode_idx);
        sim.set_tile_3d(x, y, z, self.ram_decode_saved_tile_type.get());
        self.synth_ram_decode.disable();
        self.compact_ops_stale.set(true);
    }

    pub fn synth_ram_decode_mismatches(&self) -> u64 {
        self.synth_ram_decode.mismatches()
    }

    pub fn synth_ram_decode_checks(&self) -> u64 {
        self.synth_ram_decode.checks()
    }

    /// Return the And gate index downstream of the RAM decoder (for test observation).
    pub fn ram_write_gate_idx(&self) -> usize {
        self.ram_write_gate_idx
    }

    /// Read the current tile type of the RAM decoder tile (for test assertions).
    pub fn ram_decode_tile_type(&self, sim: &Simulation) -> TileType {
        let (x, y, z) = Self::idx_to_xyz(sim, self.ram_write_decode_idx);
        sim.tile_type_3d(x, y, z)
    }

    // ---- Sprint 218: ALU readback dual-path verification ----

    /// Enable ALU readback checking. Observational only — software remains authoritative.
    pub fn enable_alu_readback(&self) {
        self.synth_alu.enable();
        self.reg_capture_checks.set(0);
        self.reg_capture_mismatches.set(0);
        self.alu_mux_select_mismatches.set(0);
        for cell in &self.alu_opcode_mismatches {
            cell.set(0);
        }
    }

    pub fn alu_readback_checks(&self) -> u64 {
        self.synth_alu.checks()
    }

    pub fn alu_readback_mismatches(&self) -> u64 {
        self.synth_alu.mismatches()
    }

    /// Per-opcode mismatch counts (indexed by opcode 0..31).
    pub fn alu_opcode_mismatches(&self) -> [u64; 32] {
        std::array::from_fn(|i| self.alu_opcode_mismatches[i].get())
    }

    pub fn reg_capture_checks(&self) -> u64 {
        self.reg_capture_checks.get()
    }

    pub fn reg_capture_mismatches(&self) -> u64 {
        self.reg_capture_mismatches.get()
    }

    /// Sprint 239: Verification counters for high-register ALU trunk re-source.
    pub fn upper_alu_trunk_checks(&self) -> u64 {
        self.upper_alu_trunk_checks.get()
    }

    pub fn upper_alu_trunk_mismatches(&self) -> u64 {
        self.upper_alu_trunk_mismatches.get()
    }

    /// Sprint 218 D2: ALU mismatches where ctrl_a[2:0] (mux selector) also diverged.
    pub fn alu_mux_select_mismatches(&self) -> u64 {
        self.alu_mux_select_mismatches.get()
    }

    // ---- Sprint 219: Physical ALU authority ----

    /// Enable physical ALU authority for a set of opcodes (bitmask).
    /// When an opcode bit is set, the physical ALU result from `wb_alu_root_idx`
    /// is used instead of the software `match` computation.
    pub fn set_physical_alu_opcodes(&self, mask: u32) {
        self.physical_alu_opcodes.set(mask);
    }

    /// Returns the current physical ALU opcode bitmask.
    pub fn physical_alu_opcodes(&self) -> u32 {
        self.physical_alu_opcodes.get()
    }

    /// Convenience: enable physical ALU for all basic reg-reg ops (0x04-0x0F),
    /// excluding sub-function variants (MUL, SRA, CLZ, CTZ, POPCNT).
    /// Does NOT include immediate ALU ops (0x10-0x14) — use PHYSICAL_ALU_IMM8.
    pub fn enable_physical_alu_all_basic(&self) {
        // Bits 4-15: ADD, SUB, AND, OR, XOR, NOT, INC, DEC, SHL, SHR + reserved 0x0A/0x0B
        self.physical_alu_opcodes.set(0xFFF0);
    }

    // ---- Sprint 220: Physical register writeback authority ----

    /// Enable physical register writeback for R0-R7.
    /// Must be called AFTER `elide_software_writeback_from_clock_cache()`.
    /// Adds R0-R7 back to the clock cache so they capture on rising edge.
    pub(crate) fn restore_reg_clock_cache_for_physical_writeback(&mut self) {
        for i in 0..8 {
            let idx = self.reg_indices[i];
            if !self.in_scope_clock_cache.contains(&idx) {
                self.in_scope_clock_cache.push(idx);
            }
        }
        self.physical_reg_writeback.set(true);
        self.reg_wb_checks.set(0);
        self.reg_wb_mismatches.set(0);
    }

    pub fn physical_reg_writeback(&self) -> bool {
        self.physical_reg_writeback.get()
    }

    pub fn reg_wb_checks(&self) -> u64 {
        self.reg_wb_checks.get()
    }

    pub fn reg_wb_mismatches(&self) -> u64 {
        self.reg_wb_mismatches.get()
    }

    // ---- Sprint 221: Physical flag writeback authority ----

    /// Enable physical flag writeback for Z and C flags.
    /// Must be called AFTER `elide_software_writeback_from_clock_cache()`.
    /// Adds flag_z_idx and flag_c_idx back to the clock cache.
    pub(crate) fn restore_flag_clock_cache_for_physical_writeback(&mut self) {
        for &idx in &[self.flag_z_idx, self.flag_c_idx] {
            if !self.in_scope_clock_cache.contains(&idx) {
                self.in_scope_clock_cache.push(idx);
            }
        }
        // Sprint 232: Build the no-flags variant by excluding flag tiles.
        // This cache is used on non-flag instructions so flag Register8 tiles
        // keep their current value (no stale capture → no re-inject needed).
        self.in_scope_clock_cache_no_flags = self
            .in_scope_clock_cache
            .iter()
            .copied()
            .filter(|&idx| idx != self.flag_z_idx && idx != self.flag_c_idx)
            .collect();
        self.physical_flag_writeback.set(true);
        self.flag_wb_checks.set(0);
        self.flag_wb_mismatches.set(0);
        self.flag_z_mismatches.set(0);
        self.flag_c_mismatches.set(0);
    }

    pub fn physical_flag_writeback(&self) -> bool {
        self.physical_flag_writeback.get()
    }

    pub fn flag_wb_checks(&self) -> u64 {
        self.flag_wb_checks.get()
    }

    pub fn flag_wb_mismatches(&self) -> u64 {
        self.flag_wb_mismatches.get()
    }

    pub fn flag_z_mismatches(&self) -> u64 {
        self.flag_z_mismatches.get()
    }

    pub fn flag_c_mismatches(&self) -> u64 {
        self.flag_c_mismatches.get()
    }

    // ---- Sprint 222: Physical ctrl_a/ctrl_b decode authority ----

    /// Enable physical ctrl_a/ctrl_b authority for bank0.
    /// Stage F reads ctrl_b from the physical Mux16to1 decoder; inject_synth_pre_commit
    /// reads ctrl_a from the physical decoder. Both use LUT fallback on mismatch.
    pub fn set_physical_decode(&self, enabled: bool) {
        self.physical_decode.set(enabled);
        self.decode_ctrl_b_checks.set(0);
        self.decode_ctrl_b_mismatches.set(0);
        self.decode_ctrl_a_checks.set(0);
        self.decode_ctrl_a_mismatches.set(0);
    }

    pub fn physical_decode(&self) -> bool {
        self.physical_decode.get()
    }

    pub fn decode_ctrl_b_checks(&self) -> u64 {
        self.decode_ctrl_b_checks.get()
    }

    pub fn decode_ctrl_b_mismatches(&self) -> u64 {
        self.decode_ctrl_b_mismatches.get()
    }

    pub fn decode_ctrl_a_checks(&self) -> u64 {
        self.decode_ctrl_a_checks.get()
    }

    pub fn decode_ctrl_a_mismatches(&self) -> u64 {
        self.decode_ctrl_a_mismatches.get()
    }

    // ---- Sprint 223: Upper-bank authority extension ----

    pub fn pc_override_checks(&self) -> u64 {
        self.pc_override_checks.get()
    }

    pub fn pc_override_mismatches(&self) -> u64 {
        self.pc_override_mismatches.get()
    }

    /// Sprint 224: Per-branch-kind PC mismatch breakdown.
    pub fn pc_mismatch_per_kind(&self) -> [u64; 8] {
        std::array::from_fn(|i| self.pc_mismatch_per_kind[i].get())
    }

    // ---- Sprint 225: Physical branch direction authority ----

    pub fn set_physical_branch(&self, enabled: bool) {
        self.physical_branch.set(enabled);
        self.branch_dir_checks.set(0);
        self.branch_dir_mismatches.set(0);
    }

    pub fn physical_branch(&self) -> bool {
        self.physical_branch.get()
    }

    pub fn branch_dir_checks(&self) -> u64 {
        self.branch_dir_checks.get()
    }

    pub fn branch_dir_mismatches(&self) -> u64 {
        self.branch_dir_mismatches.get()
    }

    // ---- Sprint 229: Physical RAM writeback authority ----

    /// Enable physical RAM writeback. Un-elides all 128 RAM tiles from the clock
    /// cache so they participate in clock edge captures. The Ram tile's built-in
    /// WE gate (output = UP!=0 ? LEFT : current) provides physical write-enable.
    /// Must be called AFTER `elide_software_writeback_from_clock_cache()`.
    pub(crate) fn restore_ram_clock_cache_for_physical_writeback(&mut self) {
        for i in 0..128 {
            let idx = self.ram_indices[i];
            // Sprint 247: Add bank 0 Ram tiles (cells 0-7) to clock_scope_mask
            // so they are not filtered out by fill_into_masked() during the clock
            // edge delta loop. Without this, Ram tiles are seeded but never evaluated.
            // Upper-bank cells (8-127) are NOT added: their UP neighbor is the bank
            // below's Ram tile (not a WE extraction tile), so they would spuriously
            // capture on every clock edge.
            if i < 8 {
                let word = idx / 64;
                let bit = idx % 64;
                if word < self.clock_scope_mask.len() {
                    self.clock_scope_mask[word] |= 1u64 << bit;
                }
                // Sprint 277: Only add to clock cache if also in clock_scope_mask.
                // Cells 8-127 are intentionally excluded from both — seeding them
                // wastes work since fill_into_masked will never drain them.
                if !self.in_scope_clock_cache.contains(&idx) {
                    self.in_scope_clock_cache.push(idx);
                }
            }
        }
        self.physical_ram_writeback.set(true);
        self.ram_wb_checks.set(0);
        self.ram_wb_mismatches.set(0);
        self.ram_wb_store_mismatches.set(0);
        self.ram_wb_nonstore_mismatches.set(0);
    }

    /// Sprint 232: Rebuild the no-flags clock cache from the current in_scope_clock_cache.
    /// Must be called AFTER all clock cache modifications (reg, flag, RAM un-elision).
    pub(crate) fn rebuild_no_flags_clock_cache(&mut self) {
        self.in_scope_clock_cache_no_flags = self
            .in_scope_clock_cache
            .iter()
            .copied()
            .filter(|&idx| idx != self.flag_z_idx && idx != self.flag_c_idx)
            .collect();
    }

    pub fn physical_ram_writeback(&self) -> bool {
        self.physical_ram_writeback.get()
    }

    pub fn ram_wb_checks(&self) -> u64 {
        self.ram_wb_checks.get()
    }

    pub fn ram_wb_mismatches(&self) -> u64 {
        self.ram_wb_mismatches.get()
    }

    pub fn ram_wb_store_mismatches(&self) -> u64 {
        self.ram_wb_store_mismatches.get()
    }

    pub fn ram_wb_nonstore_mismatches(&self) -> u64 {
        self.ram_wb_nonstore_mismatches.get()
    }

    /// Sprint 230: Three-point RAM snapshots from the last store cycle.
    /// Returns (post_commit, post_reinject, post_clock) for bank-0 cells 0..8.
    pub fn ram_snapshots(&self) -> ([u64; 8], [u64; 8], [u64; 8]) {
        let mut pc = [0u64; 8];
        let mut pr = [0u64; 8];
        let mut pk = [0u64; 8];
        for i in 0..8 {
            pc[i] = self.ram_snap_post_commit[i].get();
            pr[i] = self.ram_snap_post_reinject[i].get();
            pk[i] = self.ram_snap_post_clock[i].get();
        }
        (pc, pr, pk)
    }

    /// Sprint 230: Store address from the last store cycle (for snapshot interpretation).
    pub fn ram_snap_store_addr(&self) -> usize {
        self.ram_snap_store_addr.get()
    }

    // ---- Sprint 233: Physical pre-commit decode delivery authority ----

    /// Sprint 233: Restore the rd_decode Decoder3to8 tile from Const back to its
    /// original type. Called by the builder when physical_rd_decode is enabled.
    pub fn restore_rd_decode_tile(&self, sim: &mut Simulation) {
        let (x, y, z) = Self::idx_to_xyz(sim, self.rd_onehot_decode_idx);
        sim.set_tile_3d(x, y, z, self.rd_decode_saved_tile_type.get());
        self.physical_rd_decode.set(true);
        self.compact_ops_stale.set(true);
    }

    /// Sprint 233: Restore the we_mask And tile from Const back to its original type.
    /// Called by the builder when physical_we_mask is enabled, after enable_synth_ctrl_a
    /// has swapped it to Const.
    pub fn restore_we_mask_tile(&self, sim: &mut Simulation) {
        let (x, y, z) = Self::idx_to_xyz(sim, self.we_mask_const_idx);
        sim.set_tile_3d(x, y, z, self.we_mask_saved_tile_type.get());
        self.physical_we_mask.set(true);
        self.compact_ops_stale.set(true);
    }

    /// Sprint 233: Restore the flag_we_mask WeightedViaUp tile from Const back to its
    /// original type. Called by the builder when physical_flag_we_mask is enabled.
    pub fn restore_flag_we_mask_tile(&self, sim: &mut Simulation) {
        let (x, y, z) = Self::idx_to_xyz(sim, self.flag_we_mask_const_idx);
        sim.set_tile_3d(x, y, z, self.flag_we_mask_saved_tile_type.get());
        self.physical_flag_we_mask.set(true);
        self.compact_ops_stale.set(true);
    }

    pub fn physical_rd_decode(&self) -> bool {
        self.physical_rd_decode.get()
    }

    pub fn physical_we_mask(&self) -> bool {
        self.physical_we_mask.get()
    }

    pub fn physical_flag_we_mask(&self) -> bool {
        self.physical_flag_we_mask.get()
    }

    pub fn rd_decode_authority_checks(&self) -> u64 {
        self.rd_decode_authority_checks.get()
    }

    pub fn rd_decode_authority_mismatches(&self) -> u64 {
        self.rd_decode_authority_mismatches.get()
    }

    pub fn we_mask_authority_checks(&self) -> u64 {
        self.we_mask_authority_checks.get()
    }

    pub fn we_mask_authority_mismatches(&self) -> u64 {
        self.we_mask_authority_mismatches.get()
    }

    pub fn flag_we_mask_authority_checks(&self) -> u64 {
        self.flag_we_mask_authority_checks.get()
    }

    pub fn flag_we_mask_authority_mismatches(&self) -> u64 {
        self.flag_we_mask_authority_mismatches.get()
    }

    // ---- Sprint 234: Physical Super Mux propagation ----

    pub fn set_physical_super_mux(&self, enabled: bool) {
        self.physical_super_mux.set(enabled);
    }

    pub fn physical_super_mux(&self) -> bool {
        self.physical_super_mux.get()
    }

    pub fn super_mux_checks(&self) -> u64 {
        self.super_mux_checks.get()
    }

    pub fn super_mux_mismatches(&self) -> u64 {
        self.super_mux_mismatches.get()
    }

    pub fn upper_bank_ir_checks(&self) -> u64 {
        self.upper_bank_ir_checks.get()
    }

    pub fn upper_bank_ir_mismatches(&self) -> u64 {
        self.upper_bank_ir_mismatches.get()
    }

    // ---- Sprint 235: Physical writeback-data authority ----

    pub fn set_physical_wb_data_authority(&self, enabled: bool) {
        self.physical_wb_data_authority.set(enabled);
    }

    pub fn physical_wb_data_authority(&self) -> bool {
        self.physical_wb_data_authority.get()
    }

    pub fn wb_data_checks(&self) -> u64 {
        self.wb_data_checks.get()
    }

    pub fn wb_data_mismatches(&self) -> u64 {
        self.wb_data_mismatches.get()
    }

    // ---- Sprint 236: Physical high-register writeback authority ----

    /// Enable physical writeback for R8-R15.
    /// Must be called AFTER `elide_software_writeback_from_clock_cache()`.
    /// Adds R8-R15 back to the clock cache so they capture on rising edge.
    pub(crate) fn restore_high_reg_clock_cache_for_physical_writeback(&mut self) {
        for i in 8..16 {
            let idx = self.reg_indices[i];
            if !self.in_scope_clock_cache.contains(&idx) {
                self.in_scope_clock_cache.push(idx);
            }
        }
        self.physical_high_reg_writeback.set(true);
        self.high_reg_wb_checks.set(0);
        self.high_reg_wb_mismatches.set(0);
    }

    pub fn physical_high_reg_writeback(&self) -> bool {
        self.physical_high_reg_writeback.get()
    }

    pub fn high_reg_wb_checks(&self) -> u64 {
        self.high_reg_wb_checks.get()
    }

    pub fn high_reg_wb_mismatches(&self) -> u64 {
        self.high_reg_wb_mismatches.get()
    }

    // ---- Sprint 245: Sub-function ALU delivery authority ----

    pub fn set_physical_sub_fn_delivery(&self, enabled: bool) {
        self.physical_sub_fn_delivery.set(enabled);
        self.sub_fn_delivery_checks.set(0);
        self.sub_fn_delivery_mismatches.set(0);
    }

    pub fn physical_sub_fn_delivery(&self) -> bool {
        self.physical_sub_fn_delivery.get()
    }

    pub fn sub_fn_delivery_checks(&self) -> u64 {
        self.sub_fn_delivery_checks.get()
    }

    pub fn sub_fn_delivery_mismatches(&self) -> u64 {
        self.sub_fn_delivery_mismatches.get()
    }

    // ---- Sprint 246: Sub-function flag authority + LD/LDB delivery ----

    pub fn set_physical_sub_fn_flag_authority(&self, enabled: bool) {
        self.physical_sub_fn_flag_authority.set(enabled);
    }

    pub fn physical_sub_fn_flag_authority(&self) -> bool {
        self.physical_sub_fn_flag_authority.get()
    }

    pub fn set_physical_load_delivery(&self, enabled: bool) {
        self.physical_load_delivery.set(enabled);
        self.load_delivery_checks.set(0);
        self.load_delivery_mismatches.set(0);
    }

    pub fn physical_load_delivery(&self) -> bool {
        self.physical_load_delivery.get()
    }

    pub fn load_delivery_checks(&self) -> u64 {
        self.load_delivery_checks.get()
    }

    pub fn load_delivery_mismatches(&self) -> u64 {
        self.load_delivery_mismatches.get()
    }

    pub fn set_physical_ram_store_authority(&self, enabled: bool) {
        self.physical_ram_store_authority.set(enabled);
    }

    pub fn physical_ram_store_authority(&self) -> bool {
        self.physical_ram_store_authority.get()
    }

    // ---- Sprint 250: Hierarchical bitscan evaluation ----

    /// Evaluate the 3-stage hierarchical CLZ or CTZ pipeline.
    ///
    /// Stage 1: 8 × BITSCAN8 blocks (one per byte of operand `a`).
    /// Stage 2: 2 × half-group combine blocks (4 byte results → 1 group summary).
    /// Stage 3: 1 × final combine block (2 group summaries → 7-bit result).
    ///
    /// For CLZ: bytes fed MSB-first (byte 7 has highest priority).
    ///   Bit order within each byte: normal (b0=LSB of byte, b7=MSB of byte).
    /// For CTZ: bytes fed LSB-first (byte 0 has highest priority).
    ///   Bit order within each byte: reversed (b0=MSB of byte, b7=LSB of byte)
    ///   so the "leading zeros" of the reversed byte = trailing zeros of original.
    fn evaluate_hierarchical_bitscan(&self, sim: &mut Simulation, a: u64, is_clz: bool) -> u64 {
        use crate::synth::integration::drive_synth_block;

        let byte_blocks = self.synth_bitscan8_blocks.as_ref().unwrap();
        let half_blocks = self.synth_bitscan_half_blocks.as_ref().unwrap();
        let final_block = self.synth_bitscan_final_block.as_ref().unwrap();

        // Stage 1: Evaluate 8 byte-level BITSCAN8 blocks.
        // Each block outputs: [has_nz, count0, count1, count2].
        let mut byte_results = [[0u64; 4]; 8];
        for byte_idx in 0..8usize {
            let byte_val = ((a >> (byte_idx * 8)) & 0xFF) as u8;

            // Build 8 boolean inputs for this byte's block.
            let inputs: [u64; 8] = if is_clz {
                // CLZ: normal bit order (b0=LSB, b7=MSB of byte).
                std::array::from_fn(|i| {
                    if (byte_val >> i) & 1 != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                })
            } else {
                // CTZ: reverse bit order so CLZ of reversed = CTZ of original.
                std::array::from_fn(|i| {
                    if (byte_val >> (7 - i)) & 1 != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                })
            };

            let outputs = drive_synth_block(sim, &byte_blocks[byte_idx], &inputs);
            for k in 0..4 {
                byte_results[byte_idx][k] = outputs[k];
            }
        }

        // Stage 2: Evaluate 2 half-group combine blocks.
        // For CLZ: upper half (bytes 7,6,5,4) and lower half (bytes 3,2,1,0).
        //   Each half-combine takes inputs in priority order (highest-priority first).
        // For CTZ: lower half (bytes 0,1,2,3) and upper half (bytes 4,5,6,7).
        let mut half_results = [[0u64; 6]; 2];

        // Half 0 = "priority half" (upper for CLZ, lower for CTZ).
        // Half 1 = "other half".
        let (h0_bytes, h1_bytes): ([usize; 4], [usize; 4]) = if is_clz {
            ([7, 6, 5, 4], [3, 2, 1, 0]) // MSB-first priority
        } else {
            ([0, 1, 2, 3], [4, 5, 6, 7]) // LSB-first priority
        };

        for (half_idx, byte_order) in [h0_bytes, h1_bytes].iter().enumerate() {
            // 16 inputs: 4 × (has_nz, count0, count1, count2)
            let mut inputs = [0u64; 16];
            for (slot, &bi) in byte_order.iter().enumerate() {
                inputs[slot * 4] = byte_results[bi][0]; // has_nz
                inputs[slot * 4 + 1] = byte_results[bi][1]; // count0
                inputs[slot * 4 + 2] = byte_results[bi][2]; // count1
                inputs[slot * 4 + 3] = byte_results[bi][3]; // count2
            }
            let outputs = drive_synth_block(sim, &half_blocks[half_idx], &inputs);
            for k in 0..6 {
                half_results[half_idx][k] = outputs[k];
            }
        }

        // Stage 3: Evaluate final combine block.
        // For CLZ: upper group (half 0) has priority, lower group (half 1) is fallback.
        // For CTZ: lower group (half 0) has priority, upper group (half 1) is fallback.
        // The final AIG is build_clz_final_combine_aig for CLZ:
        //   inputs: u_nz, u_idx0, u_idx1, u_cnt0, u_cnt1, u_cnt2,
        //           l_nz, l_idx0, l_idx1, l_cnt0, l_cnt1, l_cnt2
        // For CTZ (build_ctz_final_combine_aig):
        //   inputs: l_nz, l_idx0, l_idx1, l_cnt0, l_cnt1, l_cnt2,
        //           u_nz, u_idx0, u_idx1, u_cnt0, u_cnt1, u_cnt2
        // Since we share one final block, we always use the CLZ final block
        // but swap which half is "upper" vs "lower" based on is_clz.
        let (priority_half, fallback_half) = (0, 1);
        let inputs: [u64; 12] = [
            half_results[priority_half][0], // u_nz / l_nz (priority group)
            half_results[priority_half][1], // u_idx0 / l_idx0
            half_results[priority_half][2], // u_idx1 / l_idx1
            half_results[priority_half][3], // u_cnt0 / l_cnt0
            half_results[priority_half][4], // u_cnt1 / l_cnt1
            half_results[priority_half][5], // u_cnt2 / l_cnt2
            half_results[fallback_half][0], // l_nz / u_nz (fallback group)
            half_results[fallback_half][1], // l_idx0 / u_idx0
            half_results[fallback_half][2], // l_idx1 / u_idx1
            half_results[fallback_half][3], // l_cnt0 / u_cnt0
            half_results[fallback_half][4], // l_cnt1 / u_cnt1
            half_results[fallback_half][5], // l_cnt2 / u_cnt2
        ];
        let outputs = drive_synth_block(sim, final_block, &inputs);

        // Assemble 7-bit result from outputs.
        let mut val = 0u64;
        for i in 0..7 {
            if outputs[i] != 0 {
                val |= 1 << i;
            }
        }
        val
    }

    /// Sprint 251: Hierarchical POPCNT via pairwise adder tree.
    ///
    /// Stage 1: 8 × popcnt8 (8→4 bits each)
    /// Stage 2: 4 × add(4) — pairs of 4-bit popcounts → 5-bit sums
    /// Stage 3: 2 × add(5) — pairs of 5-bit sums → 6-bit sums
    /// Stage 4: 1 × add(6) — final 6+6 → 7-bit result (0..64)
    fn evaluate_hierarchical_popcnt(&self, sim: &mut Simulation, a: u64) -> u64 {
        use crate::synth::integration::drive_synth_block;

        let byte_blocks = self.synth_popcnt8_blocks.as_ref().unwrap();
        let add4_blocks = self.synth_popcnt_add4_blocks.as_ref().unwrap();
        let add5_blocks = self.synth_popcnt_add5_blocks.as_ref().unwrap();
        let add6_block = self.synth_popcnt_add6_block.as_ref().unwrap();

        // Stage 1: 8 byte-level POPCNT8 blocks.
        // Each outputs 4 bits: pop[3:0] (count of ones in this byte, 0..8).
        let mut byte_pops = [[0u64; 4]; 8];
        for byte_idx in 0..8usize {
            let byte_val = ((a >> (byte_idx * 8)) & 0xFF) as u8;
            let inputs: [u64; 8] = std::array::from_fn(|i| {
                if (byte_val >> i) & 1 != 0 {
                    u64::MAX
                } else {
                    0
                }
            });
            let outputs = drive_synth_block(sim, &byte_blocks[byte_idx], &inputs);
            for k in 0..4 {
                byte_pops[byte_idx][k] = outputs[k];
            }
        }

        // Stage 2: 4 × add(4) — pair adjacent byte popcounts.
        // add4 inputs: a0..a3, b0..b3 (8 total) → outputs: s0..s4 (5 bits).
        let mut add4_results = [[0u64; 5]; 4];
        for pair in 0..4usize {
            let a_idx = pair * 2;
            let b_idx = pair * 2 + 1;
            let mut inputs = [0u64; 8];
            for i in 0..4 {
                inputs[i] = byte_pops[a_idx][i];
                inputs[4 + i] = byte_pops[b_idx][i];
            }
            let outputs = drive_synth_block(sim, &add4_blocks[pair], &inputs);
            for k in 0..5 {
                add4_results[pair][k] = outputs[k];
            }
        }

        // Stage 3: 2 × add(5) — pair adjacent 5-bit sums.
        // add5 inputs: a0..a4, b0..b4 (10 total) → outputs: s0..s5 (6 bits).
        let mut add5_results = [[0u64; 6]; 2];
        for pair in 0..2usize {
            let a_idx = pair * 2;
            let b_idx = pair * 2 + 1;
            let mut inputs = [0u64; 10];
            for i in 0..5 {
                inputs[i] = add4_results[a_idx][i];
                inputs[5 + i] = add4_results[b_idx][i];
            }
            let outputs = drive_synth_block(sim, &add5_blocks[pair], &inputs);
            for k in 0..6 {
                add5_results[pair][k] = outputs[k];
            }
        }

        // Stage 4: 1 × add(6) — final sum.
        // add6 inputs: a0..a5, b0..b5 (12 total) → outputs: s0..s6 (7 bits).
        let mut inputs = [0u64; 12];
        for i in 0..6 {
            inputs[i] = add5_results[0][i];
            inputs[6 + i] = add5_results[1][i];
        }
        let outputs = drive_synth_block(sim, add6_block, &inputs);

        // Assemble 7-bit result.
        let mut val = 0u64;
        for i in 0..7 {
            if outputs[i] != 0 {
                val |= 1 << i;
            }
        }
        val
    }

    // ---- Sprint 248: SRA computation authority ----

    pub fn set_synth_sra_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_sra_block = Some(block);
    }

    pub fn set_physical_sra_computation(&self, enabled: bool) {
        self.physical_sra_computation.set(enabled);
        self.sra_computation_checks.set(0);
        self.sra_computation_mismatches.set(0);
    }

    pub fn physical_sra_computation(&self) -> bool {
        self.physical_sra_computation.get()
    }

    // ---- Sprint 362: MUL synth block ----

    pub fn set_synth_mul_block(&mut self, block: crate::synth::integration::InjectedBlock) {
        self.synth_mul_block = Some(block);
    }

    pub fn enable_synth_mul(&self) {
        self.synth_alu.enable();
        self.physical_mul_authority.set(true);
    }

    pub fn physical_mul_authority(&self) -> bool {
        self.physical_mul_authority.get()
    }

    pub fn sra_computation_checks(&self) -> u64 {
        self.sra_computation_checks.get()
    }

    pub fn sra_computation_mismatches(&self) -> u64 {
        self.sra_computation_mismatches.get()
    }

    // ---- Sprint 250: Hierarchical bitop computation authority ----

    pub fn set_bitscan_blocks(
        &mut self,
        byte_blocks: [crate::synth::integration::InjectedBlock; 8],
        half_blocks: [crate::synth::integration::InjectedBlock; 2],
        final_block: crate::synth::integration::InjectedBlock,
    ) {
        self.synth_bitscan8_blocks = Some(byte_blocks);
        self.synth_bitscan_half_blocks = Some(half_blocks);
        self.synth_bitscan_final_block = Some(final_block);
    }

    pub fn set_popcnt_blocks(
        &mut self,
        byte_blocks: [crate::synth::integration::InjectedBlock; 8],
        add4_blocks: [crate::synth::integration::InjectedBlock; 4],
        add5_blocks: [crate::synth::integration::InjectedBlock; 2],
        add6_block: crate::synth::integration::InjectedBlock,
    ) {
        self.synth_popcnt8_blocks = Some(byte_blocks);
        self.synth_popcnt_add4_blocks = Some(add4_blocks);
        self.synth_popcnt_add5_blocks = Some(add5_blocks);
        self.synth_popcnt_add6_block = Some(add6_block);
    }

    pub fn set_physical_bitop_computation(&self, enabled: bool) {
        self.physical_bitop_computation.set(enabled);
        self.bitop_checks.set(0);
        self.bitop_mismatches.set(0);
    }

    pub fn physical_bitop_computation(&self) -> bool {
        self.physical_bitop_computation.get()
    }

    pub fn bitop_checks(&self) -> u64 {
        self.bitop_checks.get()
    }

    pub fn bitop_mismatches(&self) -> u64 {
        self.bitop_mismatches.get()
    }

    pub fn set_physical_ir_spine(&self, enabled: bool) {
        self.physical_ir_spine.set(enabled);
        self.ir_spine_checks.set(0);
        self.ir_spine_mismatches.set(0);
    }

    pub fn physical_ir_spine(&self) -> bool {
        self.physical_ir_spine.get()
    }

    pub fn ir_spine_checks(&self) -> u64 {
        self.ir_spine_checks.get()
    }

    pub fn ir_spine_mismatches(&self) -> u64 {
        self.ir_spine_mismatches.get()
    }

    pub fn set_physical_via_decode(&mut self, enabled: bool, inversion_dirty: Vec<usize>) {
        self.physical_via_decode.set(enabled);
        self.via_decode_inversion_dirty = inversion_dirty;
        self.via_decode_checks.set(0);
        self.via_decode_mismatches.set(0);
    }

    pub fn physical_via_decode(&self) -> bool {
        self.physical_via_decode.get()
    }

    pub fn via_decode_checks(&self) -> u64 {
        self.via_decode_checks.get()
    }

    pub fn via_decode_mismatches(&self) -> u64 {
        self.via_decode_mismatches.get()
    }

    pub fn branch_target_checks(&self) -> u64 {
        self.branch_target_checks.get()
    }

    pub fn branch_target_mismatches(&self) -> u64 {
        self.branch_target_mismatches.get()
    }

    pub fn set_physical_byte2_selector(&self, enabled: bool) {
        self.physical_byte2_selector.set(enabled);
        self.byte2_selector_checks.set(0);
        self.byte2_selector_mismatches.set(0);
    }

    pub fn byte2_selector_checks(&self) -> u64 {
        self.byte2_selector_checks.get()
    }

    pub fn byte2_selector_mismatches(&self) -> u64 {
        self.byte2_selector_mismatches.get()
    }

    /// Sprint 274: Number of Stage F cone convergence checks performed.
    pub fn cone_convergence_checks(&self) -> u64 {
        self.cone_single_pass_checks.get()
    }

    /// Sprint 274: Total residual changes detected across all convergence checks.
    /// Zero means single-pass convergence holds.
    pub fn cone_convergence_residual(&self) -> u64 {
        self.cone_residual_changes.get()
    }

    /// Sprint 255: Extend pipeline and clock scope masks to include spine tile indices.
    /// Called after spine tiles are placed (which happens after BFS scope computation).
    pub fn extend_scope_masks_for_spine(&mut self, spine_indices: &[usize]) {
        for &idx in spine_indices {
            let seg = idx / 64;
            let bit = idx % 64;
            if seg < self.pipeline_scope_mask.len() {
                self.pipeline_scope_mask[seg] |= 1u64 << bit;
                self.clock_scope_mask[seg] |= 1u64 << bit;
            }
        }
        // Sprint 266: Also add to upper-bank dirty set.
        self.upper_bank_dirty_indices
            .extend_from_slice(spine_indices);
    }

    /// Sprint 272: Collect pass0 ROM cone seeds — only the tiles read between
    /// the first propagation (line 1163) and the next injection (line 1179).
    /// These are the 4 ROM lane super_mux outputs + the Final Mux / Bank47 Mux
    /// tiles that feed read_active_rom_lanes().
    /// Sprint 290: Collect the union of ALL dirty seeds that Stage F marks before
    /// the combined settle. This is the maximum possible dirty set across all
    /// cycle types (all registers, all injection paths, all upper-bank paths).
    /// Used to compute the structural forward closure for the settle-scope ops.
    pub(crate) fn all_settle_dirty_seeds(&self) -> Vec<usize> {
        let mut seeds = Vec::with_capacity(256);

        // Backbone dirty (always marked every cycle: ROM fetch, Super Mux, etc.)
        seeds.extend_from_slice(&self.pipeline_backbone_dirty_indices);

        // Per-register data dirty (include all 16 for worst case)
        for reg in 0..16 {
            seeds.extend_from_slice(&self.pipeline_reg_data_dirty_indices[reg]);
        }

        // Super Mux outputs (when physical_super_mux)
        seeds.push(self.rom_selected_low_idx);
        seeds.push(self.rom_selected_high_idx);
        seeds.push(self.rom_selected_byte2_idx);
        seeds.push(self.rom_selected_byte3_idx);

        // High-tree dirty indices
        seeds.extend_from_slice(&self.high_tree_a_dirty_indices);
        seeds.extend_from_slice(&self.high_tree_b_dirty_indices);

        // Top-mux dirty indices
        seeds.extend_from_slice(&self.top_mux_a_dirty_indices);
        seeds.extend_from_slice(&self.top_mux_b_dirty_indices);

        // Via decode inversion dirty (when physical_via_decode)
        seeds.extend_from_slice(&self.via_decode_inversion_dirty);

        // High-tree selector Consts (physical_via_decode marks these dirty)
        for &idx in &self.high_tree_a_sel_const_indices {
            seeds.push(idx);
        }
        for &idx in &self.high_tree_b_sel_const_indices {
            seeds.push(idx);
        }

        // Upper-bank dirty indices (physical_ir_spine targeted path)
        seeds.extend_from_slice(&self.upper_bank_dirty_indices);

        // High-tree data Const tiles (values set but downstream needs dirty)
        for &idx in &self.high_tree_a_data_const_indices {
            seeds.push(idx);
        }
        for &idx in &self.high_tree_b_data_const_indices {
            seeds.push(idx);
        }

        // Top-mux selector Const tiles
        seeds.push(self.top_mux_a_sel_const_idx);
        seeds.push(self.top_mux_b_sel_const_idx);

        // Super Mux inject points
        for &idx in &self.super_mux_inject_indices {
            seeds.push(idx);
        }
        for &idx in &self.super_mux_inject_right_indices {
            seeds.push(idx);
        }

        // Deduplicate
        seeds.sort_unstable();
        seeds.dedup();
        seeds.retain(|&idx| idx != 0);
        seeds
    }

    /// Sprint 292: Backbone-only settle seeds — everything EXCEPT per-register
    /// R0-R7 data dirty indices. Covers ROM fetch, Super Mux, high-tree (R8-R15
    /// data + selectors), top-mux, via decode, upper-bank. This is the fixed
    /// per-cycle overhead; per-register paths are variable by changed_regs_mask.
    pub(crate) fn backbone_settle_seeds(&self) -> Vec<usize> {
        let mut seeds = Vec::with_capacity(256);
        seeds.extend_from_slice(&self.pipeline_backbone_dirty_indices);
        // R8-R15 per-register dirty (always needed: high-tree data injection covers R8-R15)
        for reg in 8..16 {
            seeds.extend_from_slice(&self.pipeline_reg_data_dirty_indices[reg]);
        }
        seeds.push(self.rom_selected_low_idx);
        seeds.push(self.rom_selected_high_idx);
        seeds.push(self.rom_selected_byte2_idx);
        seeds.push(self.rom_selected_byte3_idx);
        seeds.extend_from_slice(&self.high_tree_a_dirty_indices);
        seeds.extend_from_slice(&self.high_tree_b_dirty_indices);
        seeds.extend_from_slice(&self.top_mux_a_dirty_indices);
        seeds.extend_from_slice(&self.top_mux_b_dirty_indices);
        seeds.extend_from_slice(&self.via_decode_inversion_dirty);
        for &idx in &self.high_tree_a_sel_const_indices {
            seeds.push(idx);
        }
        for &idx in &self.high_tree_b_sel_const_indices {
            seeds.push(idx);
        }
        seeds.extend_from_slice(&self.upper_bank_dirty_indices);
        for &idx in &self.high_tree_a_data_const_indices {
            seeds.push(idx);
        }
        for &idx in &self.high_tree_b_data_const_indices {
            seeds.push(idx);
        }
        seeds.push(self.top_mux_a_sel_const_idx);
        seeds.push(self.top_mux_b_sel_const_idx);
        for &idx in &self.super_mux_inject_indices {
            seeds.push(idx);
        }
        for &idx in &self.super_mux_inject_right_indices {
            seeds.push(idx);
        }
        seeds.sort_unstable();
        seeds.dedup();
        seeds.retain(|&idx| idx != 0);
        seeds
    }

    /// Sprint 292: Per-register R0-R7 settle seeds — just the low-tree operand
    /// data routes for a specific register.
    #[allow(dead_code)]
    pub(crate) fn register_settle_seeds(&self, reg: usize) -> &[usize] {
        &self.pipeline_reg_data_dirty_indices[reg]
    }

    pub(crate) fn collect_stage_f_output_seeds(&self) -> Vec<usize> {
        let mut seeds = Vec::with_capacity(24);
        // Super Mux outputs = ROM lane outputs (read at line 1167)
        seeds.push(self.rom_selected_low_idx);
        seeds.push(self.rom_selected_high_idx);
        seeds.push(self.rom_selected_byte2_idx);
        seeds.push(self.rom_selected_byte3_idx);
        // Final Mux + Bank47 Mux (read inside read_active_rom_lanes)
        for &idx in &self.final_mux_indices {
            seeds.push(idx);
        }
        for bank in &self.bank47_mux_indices {
            for &idx in bank {
                seeds.push(idx);
            }
        }
        seeds.retain(|&idx| idx != 0);
        seeds
    }

    /// Sprint 272: Compute output cone from compact ops via reverse topological pass.
    /// Returns (cone_ops, cone_wvia, cone_set_bitset).
    pub(crate) fn build_pipeline_cone(
        ops: &[crate::simulation::CompactOp],
        wvia: &[(usize, u8, u64)],
        seeds: &[usize],
    ) -> (
        Vec<crate::simulation::CompactOp>,
        Vec<(usize, u8, u64)>,
        Vec<u64>,
    ) {
        use std::collections::HashSet;

        // Mark seeds as needed.
        let mut needed: HashSet<usize> = seeds.iter().copied().collect();

        // Backward pass: reverse topological order.
        // If a tile is needed, its inputs are also needed.
        for op in ops.iter().rev() {
            let idx = op.idx as usize;
            if needed.contains(&idx) {
                if op.in0 != u32::MAX {
                    needed.insert(op.in0 as usize);
                }
                if op.in1 != u32::MAX {
                    needed.insert(op.in1 as usize);
                }
                if op.in2 != u32::MAX {
                    needed.insert(op.in2 as usize);
                }
            }
        }

        // Forward pass: filter ops to needed tiles, preserving topo order.
        let mut cone_ops = Vec::with_capacity(needed.len());
        let mut cone_wvia = Vec::new();
        let mut wvia_idx = 0usize;
        for op in ops {
            let is_wvia = op.op == crate::simulation::COP_WVIA;
            if needed.contains(&(op.idx as usize)) {
                cone_ops.push(*op);
                if is_wvia && wvia_idx < wvia.len() {
                    cone_wvia.push(wvia[wvia_idx]);
                }
            }
            if is_wvia {
                wvia_idx += 1;
            }
        }

        // Sprint 274: Build cone membership bitset for frontier-only dirty propagation.
        let max_idx = cone_ops.iter().map(|op| op.idx as usize).max().unwrap_or(0);
        let bitset_words = (max_idx / 64) + 1;
        let mut cone_set = vec![0u64; bitset_words];
        for op in &cone_ops {
            let i = op.idx as usize;
            cone_set[i / 64] |= 1u64 << (i % 64);
        }

        (cone_ops, cone_wvia, cone_set)
    }

    /// Sprint 262: Rebuild the topologically sorted evaluation order after
    /// scope mask modifications (e.g., spine extension, via decode tiles).
    pub fn rebuild_eval_order(&mut self, sim: &Simulation) {
        self.pipeline_eval_order = sim.build_eval_order(&self.pipeline_scope_mask);
        // Sprint 267: Rebuild compact ops from new eval order.
        let (ops, wvia) = sim.build_compact_ops(&self.pipeline_eval_order);
        self.pipeline_compact_ops = ops;
        self.pipeline_compact_wvia = wvia;
        // Sprint 289: Build pipeline schedule for combined settle.
        self.pipeline_schedule = Some(sim.build_compact_schedule(&self.pipeline_eval_order, false));
        // Sprint 270: Rebuild branch + commit compact ops.
        let (ops, wvia) = sim.build_compact_ops(&self.branch_eval_order);
        self.branch_compact_ops = ops;
        self.branch_compact_wvia = wvia;
        let (ops, wvia) = sim.build_compact_ops(&self.commit_eval_order);
        self.commit_compact_ops = ops;
        self.commit_compact_wvia = wvia;
        // Sprint 278: Rebuild commit schedule.
        self.commit_schedule = Some(sim.build_compact_schedule(&self.commit_eval_order, false));
        // Sprint 279: Rebuild branch + clock schedules.
        self.branch_schedule = Some(sim.build_compact_schedule(&self.branch_eval_order, false));
        // Sprint 276: Rebuild clock scope compact ops (tile types may have changed).
        // Use build_compact_ops_clock so RAM tiles are COP_CONST (captured in delta 0 only).
        let clock_eval_order = sim.build_eval_order(&self.clock_scope_mask);
        let (ops, wvia) = sim.build_compact_ops_clock(&clock_eval_order);
        self.clock_compact_ops = ops;
        self.clock_compact_wvia = wvia;
        self.clock_schedule = Some(sim.build_compact_schedule(&clock_eval_order, true));
        // Sprint 272: Rebuild cone-pruned pipeline ops.
        let seeds = self.collect_stage_f_output_seeds();
        let (cone_ops, cone_wvia, cone_set) = Self::build_pipeline_cone(
            &self.pipeline_compact_ops,
            &self.pipeline_compact_wvia,
            &seeds,
        );
        self.pipeline_cone_ops = cone_ops;
        self.pipeline_cone_wvia = cone_wvia;
        self.pipeline_cone_set = cone_set;
        // Sprint 273.1: Invalidate cached JIT — cone ops changed.
        #[cfg(feature = "cranelift_jit")]
        {
            self.pipeline_cone_jit = None;
        }
        // Sprint 290: Build settle-scope compact ops from forward closure of all
        // injection seeds. This is a subset of pipeline_compact_ops containing only
        // tiles structurally reachable from Stage F injection points.
        {
            let settle_seeds = self.all_settle_dirty_seeds();
            let closure_set: std::collections::HashSet<usize> = sim
                .compute_forward_closure(&settle_seeds, &self.pipeline_compact_ops)
                .into_iter()
                .collect();
            let mut s_ops = Vec::new();
            let mut s_wvia = Vec::new();
            let mut wvia_idx = 0usize;
            for op in &self.pipeline_compact_ops {
                let in_closure = closure_set.contains(&(op.idx as usize));
                if in_closure {
                    s_ops.push(*op);
                    if op.op == crate::simulation::COP_WVIA {
                        s_wvia.push(self.pipeline_compact_wvia[wvia_idx]);
                    }
                }
                if op.op == crate::simulation::COP_WVIA {
                    wvia_idx += 1;
                }
            }
            // Sprint 291: Build settle cone set bitset for no-dirty frontier propagation.
            let max_idx = s_ops.iter().map(|op| op.idx as usize).max().unwrap_or(0);
            let bitset_words = (max_idx / 64) + 1;
            let mut cone_set = vec![0u64; bitset_words];
            for op in &s_ops {
                let i = op.idx as usize;
                cone_set[i / 64] |= 1u64 << (i % 64);
            }
            self.settle_cone_set = cone_set;
            self.settle_compact_ops = s_ops;
            self.settle_compact_wvia = s_wvia;
            // Sprint 339: Build frontier table for JIT settle dirty propagation.
            {
                let (offsets, targets) = sim
                    .build_settle_frontier_table(&self.settle_compact_ops, &self.settle_cone_set);
                self.settle_frontier_offsets = offsets;
                self.settle_frontier_targets = targets;
            }
            // Sprint 306: Build prefiltered settle lookup tables.
            {
                let tile_count = sim.tilemap.tile_count();
                let num_ops = self.settle_compact_ops.len();

                let mut i2s = vec![u32::MAX; tile_count];
                let mut wvia_map = vec![u32::MAX; num_ops];
                let mut wi = 0u32;
                for (slot, op) in self.settle_compact_ops.iter().enumerate() {
                    i2s[op.idx as usize] = slot as u32;
                    if op.op == crate::simulation::COP_WVIA {
                        wvia_map[slot] = wi;
                        wi += 1;
                    }
                }
                self.settle_idx_to_slot = i2s;
                self.settle_wvia_slot_map = wvia_map;
            }
            // Sprint 318: Build trunk-only settle ops (forward closure from trunk terminals).
            {
                let trunk_seeds = self.trunk_inject_seeds_inner();
                let trunk_closure: std::collections::HashSet<usize> = sim
                    .compute_forward_closure(&trunk_seeds, &self.pipeline_compact_ops)
                    .into_iter()
                    .collect();
                let mut t_ops = Vec::new();
                let mut t_wvia = Vec::new();
                let mut wvia_idx = 0usize;
                for op in &self.settle_compact_ops {
                    if trunk_closure.contains(&(op.idx as usize)) {
                        t_ops.push(*op);
                        if op.op == crate::simulation::COP_WVIA {
                            t_wvia.push(self.settle_compact_wvia[wvia_idx]);
                        }
                    }
                    if op.op == crate::simulation::COP_WVIA {
                        wvia_idx += 1;
                    }
                }
                self.trunk_settle_ops = t_ops;
                self.trunk_settle_wvia = t_wvia;
            }
            // Sprint 308: Build block segment maps for clean-skip.
            {
                fn build_block_maps(
                    ops: &[crate::simulation::CompactOp],
                ) -> (Vec<u32>, Vec<(u32, u64)>, Vec<u8>) {
                    let num_ops = ops.len();
                    let num_blocks = (num_ops + 63) / 64;
                    let mut offsets = Vec::with_capacity(num_blocks + 1);
                    let mut entries: Vec<(u32, u64)> = Vec::new();
                    let mut wvia_counts = Vec::with_capacity(num_blocks);

                    for block in 0..num_blocks {
                        offsets.push(entries.len() as u32);
                        let start = block * 64;
                        let end = (start + 64).min(num_ops);
                        let mut wvia_count = 0u8;

                        // Collect unique (segment, mask) pairs for this block.
                        // Use a small inline buffer since blocks rarely touch >20 segments.
                        let mut seg_masks: Vec<(u32, u64)> = Vec::new();
                        for slot in start..end {
                            let op = &ops[slot];
                            if op.op == crate::simulation::COP_WVIA {
                                wvia_count += 1;
                            }
                            let seg = (op.idx / 64) as u32;
                            let bit = 1u64 << (op.idx % 64);
                            if let Some(entry) = seg_masks.iter_mut().find(|(s, _)| *s == seg) {
                                entry.1 |= bit;
                            } else {
                                seg_masks.push((seg, bit));
                            }
                        }
                        entries.extend_from_slice(&seg_masks);
                        wvia_counts.push(wvia_count);
                    }
                    offsets.push(entries.len() as u32);
                    (offsets, entries, wvia_counts)
                }

                let (o, e, w) = build_block_maps(&self.settle_compact_ops);
                self.settle_block_seg_offsets = o;
                self.settle_block_seg_entries = e;
                self.settle_block_wvia_counts = w;
            }
            // Sprint 292: Build settle schedule for active-work propagation.
            // Zero COP_GENERIC → no re-drain oscillation, single pass.
            let settle_eval_order: Vec<usize> = self
                .settle_compact_ops
                .iter()
                .map(|op| op.idx as usize)
                .collect();
            self.settle_schedule = Some(sim.build_compact_schedule(&settle_eval_order, false));
            // Sprint 293: Build forward-only deps from schedule. Keep a dep only
            // if the dep tile genuinely reads from the current tile (data-flow
            // filter, not position filter). Checks CompactOp inputs + COP_WIRE's
            // 4th neighbor. Eliminates false-positive backward marks from
            // dirty_dependents marking ALL neighbors.
            if let Some(ref sched) = self.settle_schedule {
                let num_slots = sched.ops.len();
                let mut fwd_data = Vec::with_capacity(sched.deps_data.len());
                let mut fwd_offsets = Vec::with_capacity(num_slots + 1);
                for slot in 0..num_slots {
                    fwd_offsets.push(fwd_data.len() as u32);
                    let current_idx = sched.ops[slot].idx;
                    let start = sched.deps_offsets[slot] as usize;
                    let end = sched.deps_offsets[slot + 1] as usize;
                    for i in start..end {
                        let dep_slot = sched.deps_data[i] as usize;
                        if dep_slot >= num_slots {
                            continue;
                        }
                        let dep_op = &sched.ops[dep_slot];
                        // Keep dep if the dep tile reads from the current tile.
                        let is_consumer = dep_op.in0 == current_idx
                            || dep_op.in1 == current_idx
                            || dep_op.in2 == current_idx
                            || (dep_op.op == crate::simulation::COP_WIRE
                                && sim.neighbors4_at(dep_op.idx as usize)[3] == current_idx);
                        if is_consumer {
                            fwd_data.push(dep_slot as u32);
                        }
                    }
                }
                fwd_offsets.push(fwd_data.len() as u32);
                self.settle_forward_deps_data = fwd_data;
                self.settle_forward_deps_offsets = fwd_offsets;
            }
            // Sprint 304: Build constswap variant — clone settle ops, patch R-Mux
            // output tiles to COP_CONST. Used for LDI.W/wide-imm/SRA so compact
            // settle works without compact_eval_inhibit.
            {
                let mut cs_ops = self.settle_compact_ops.clone();
                for col in 0..8 {
                    let target_idx = self.alu_r_mux_output_indices[col] as u32;
                    for op in cs_ops.iter_mut() {
                        if op.idx == target_idx {
                            op.op = crate::simulation::COP_CONST;
                            op.in0 = u32::MAX;
                            op.in1 = u32::MAX;
                            op.in2 = u32::MAX;
                            break;
                        }
                    }
                }
                self.settle_compact_ops_constswap = cs_ops;
                self.settle_compact_wvia_constswap = self.settle_compact_wvia.clone();
                // Sprint 306: Constswap prefilter lookup tables.
                {
                    let tile_count = sim.tilemap.tile_count();
                    let num_cs = self.settle_compact_ops_constswap.len();
                    let mut i2s = vec![u32::MAX; tile_count];
                    let mut wvia_map = vec![u32::MAX; num_cs];
                    let mut wi = 0u32;
                    for (slot, op) in self.settle_compact_ops_constswap.iter().enumerate() {
                        i2s[op.idx as usize] = slot as u32;
                        if op.op == crate::simulation::COP_WVIA {
                            wvia_map[slot] = wi;
                            wi += 1;
                        }
                    }
                    self.settle_idx_to_slot_constswap = i2s;
                    self.settle_wvia_slot_map_constswap = wvia_map;
                }
                // Sprint 308: Constswap block segment maps.
                {
                    fn build_block_maps_cs(
                        ops: &[crate::simulation::CompactOp],
                    ) -> (Vec<u32>, Vec<(u32, u64)>, Vec<u8>) {
                        let num_ops = ops.len();
                        let num_blocks = (num_ops + 63) / 64;
                        let mut offsets = Vec::with_capacity(num_blocks + 1);
                        let mut entries: Vec<(u32, u64)> = Vec::new();
                        let mut wvia_counts = Vec::with_capacity(num_blocks);
                        for block in 0..num_blocks {
                            offsets.push(entries.len() as u32);
                            let start = block * 64;
                            let end = (start + 64).min(num_ops);
                            let mut wvia_count = 0u8;
                            let mut seg_masks: Vec<(u32, u64)> = Vec::new();
                            for slot in start..end {
                                let op = &ops[slot];
                                if op.op == crate::simulation::COP_WVIA {
                                    wvia_count += 1;
                                }
                                let seg = (op.idx / 64) as u32;
                                let bit = 1u64 << (op.idx % 64);
                                if let Some(entry) = seg_masks.iter_mut().find(|(s, _)| *s == seg) {
                                    entry.1 |= bit;
                                } else {
                                    seg_masks.push((seg, bit));
                                }
                            }
                            entries.extend_from_slice(&seg_masks);
                            wvia_counts.push(wvia_count);
                        }
                        offsets.push(entries.len() as u32);
                        (offsets, entries, wvia_counts)
                    }
                    let (o, e, w) = build_block_maps_cs(&self.settle_compact_ops_constswap);
                    self.settle_block_seg_offsets_cs = o;
                    self.settle_block_seg_entries_cs = e;
                    self.settle_block_wvia_counts_cs = w;
                }
            }
            // Sprint 305: Backbone/fringe split for worklist-driven settle.
            {
                let backbone_seeds = self.backbone_settle_seeds();
                let backbone_closure: std::collections::HashSet<usize> = sim
                    .compute_forward_closure(&backbone_seeds, &self.pipeline_compact_ops)
                    .into_iter()
                    .collect();

                // Helper: partition ops+wvia into backbone/fringe preserving topo order.
                fn partition_ops(
                    ops: &[crate::simulation::CompactOp],
                    wvia: &[(usize, u8, u64)],
                    backbone_set: &std::collections::HashSet<usize>,
                ) -> (
                    Vec<crate::simulation::CompactOp>,
                    Vec<(usize, u8, u64)>,
                    Vec<crate::simulation::CompactOp>,
                    Vec<(usize, u8, u64)>,
                ) {
                    let mut bb_ops = Vec::new();
                    let mut bb_wvia = Vec::new();
                    let mut fr_ops = Vec::new();
                    let mut fr_wvia = Vec::new();
                    let mut wvia_idx = 0usize;
                    for op in ops {
                        let is_wvia = op.op == crate::simulation::COP_WVIA;
                        if backbone_set.contains(&(op.idx as usize)) {
                            bb_ops.push(*op);
                            if is_wvia {
                                bb_wvia.push(wvia[wvia_idx]);
                            }
                        } else {
                            fr_ops.push(*op);
                            if is_wvia {
                                fr_wvia.push(wvia[wvia_idx]);
                            }
                        }
                        if is_wvia {
                            wvia_idx += 1;
                        }
                    }
                    (bb_ops, bb_wvia, fr_ops, fr_wvia)
                }

                // Normal settle: partition + build backbone schedule.
                let (bb_ops, bb_wvia, fr_ops, fr_wvia) = partition_ops(
                    &self.settle_compact_ops,
                    &self.settle_compact_wvia,
                    &backbone_closure,
                );
                let bb_eval_order: Vec<usize> = bb_ops.iter().map(|op| op.idx as usize).collect();
                if !bb_eval_order.is_empty() {
                    self.settle_backbone_schedule =
                        Some(sim.build_compact_schedule(&bb_eval_order, false));
                }

                // Sprint 352: Store backbone ops for no-dirty evaluation.
                // Phase 2 uses full settle_compact_ops (not just fringe) for the
                // dirty pass, so fringe ops aren't needed separately here.
                self.backbone_ops = bb_ops;
                self.backbone_wvia = bb_wvia;
                self.settle_fringe_ops = fr_ops;
                self.settle_fringe_wvia = fr_wvia;

                // Build backbone cone set bitset for frontier marking.
                // Backbone no-dirty pass marks dirty only tiles outside this set.
                if !self.backbone_ops.is_empty() {
                    let max_idx = self
                        .backbone_ops
                        .iter()
                        .map(|op| op.idx as usize)
                        .max()
                        .unwrap_or(0);
                    let words = (max_idx / 64) + 1;
                    let mut cone_set = vec![0u64; words];
                    for op in &self.backbone_ops {
                        let idx = op.idx as usize;
                        cone_set[idx / 64] |= 1u64 << (idx % 64);
                    }
                    self.backbone_cone_set = cone_set;
                }

                // Constswap settle: partition + build backbone schedule.
                // TODO: re-enable after parity fix
                let (_, cs_bb_wvia, cs_fr_ops, cs_fr_wvia) = partition_ops(
                    &self.settle_compact_ops_constswap,
                    &self.settle_compact_wvia_constswap,
                    &backbone_closure,
                );
                self.settle_fringe_ops_constswap = cs_fr_ops;
                self.settle_fringe_wvia_constswap = cs_fr_wvia;

                let _ = cs_bb_wvia;

                // Sprint 355: Compute memoization key inputs and snapshot
                // outputs.
                //
                // Inputs = the externally-set settle seed tiles (Stage F
                // injects fresh values into these every cycle). Together
                // they fully determine settle-scope evaluation by
                // determinism — equal input vector implies equal final
                // tile state across the entire settle scope.
                //
                // Outputs = every settle-op output tile (backbone + fringe,
                // skipping COP_CONST since their values never change). On
                // cache hit we restore all of them so downstream phases
                // (Stage X, branch, commit, clock) see the correct
                // post-settle state without any kernel re-eval.
                if !self.backbone_ops.is_empty() {
                    // Defensive: backbone partition is supposed to exclude
                    // COP_WIRE / COP_GENERIC (S352 design). If either slips
                    // in, the hybrid kernel's miss path would still be
                    // correct (it handles them via flush_cache + eval_tile),
                    // but the partition-related invariants get murky. We
                    // keep the check informational — the seed-based input
                    // set we use is robust to it.
                    let _has_irregular = self.backbone_ops.iter().any(|op| {
                        op.op == crate::simulation::COP_WIRE
                            || op.op == crate::simulation::COP_GENERIC
                    });

                    // Inputs = all settle dirty seeds. The seed list is the
                    // complete entry-point set to the settle closure: their
                    // values (whether Const-injected or derived from prior
                    // phases like Stage X writeback) fully determine the
                    // settle scope's evaluation by determinism. Filtering
                    // to COP_CONST only is unsafe — Mux/Wire seeds whose
                    // inputs were updated by Stage X must contribute to
                    // the hash too, otherwise different program states can
                    // collide and serve stale snapshots.
                    let inputs: Vec<u32> = self
                        .all_settle_dirty_seeds()
                        .into_iter()
                        .map(|i| i as u32)
                        .collect();
                    self.backbone_input_indices = inputs;

                    let mut outputs: Vec<u32> = Vec::with_capacity(self.settle_compact_ops.len());
                    for op in &self.settle_compact_ops {
                        if op.op != crate::simulation::COP_CONST {
                            outputs.push(op.idx);
                        }
                    }
                    outputs.sort_unstable();
                    outputs.dedup();
                    self.backbone_output_indices = outputs;

                    // Cache contents are tied to the partition / index sets;
                    // rebuild invalidates them.
                    self.backbone_cache.borrow_mut().clear();
                }
            }

            // Sprint 356: Decode-only partition.
            //
            // Classify settle scope into:
            //   * decode = tiles whose value depends only on decode externals
            //     (PC, IR, decoder Consts, ROM data injection points), AND
            //   * execute = tiles tainted by register-state externals (R0-R15
            //     register tiles, high-tree data Consts holding R8-R15 values,
            //     flag Z/C tiles).
            //
            // Forward-closure: walk settle_compact_ops in topological order;
            // a tile is execute-tainted iff any of its inputs is execute-tainted
            // (input = in0/in1/in2; for COP_WIRE / COP_GENERIC also neighbors4).
            // COP_RAM is force-tainted (stateful: output depends on prior value).
            // COP_CONST is classified by membership in the explicit execute set.
            //
            // Cache key = hash of decode externals; cache value = snapshot of
            // decode tile values (excl. COP_CONST). On hit we restore those
            // values + run hybrid only on execute_compact_ops; on miss we run
            // full hybrid + snapshot decode + insert.
            //
            // Hit-rate ceiling = #(distinct decode-external states) which is
            // ~one per static instruction → near 100% in any loop.
            if !self.settle_compact_ops.is_empty() {
                use std::collections::HashSet;

                // Step 1: explicit execute-external set.
                // Tiles whose value carries register state across cycles.
                let mut execute_externals: HashSet<u32> = HashSet::new();
                for &idx in &self.high_tree_a_data_const_indices {
                    execute_externals.insert(idx as u32);
                }
                for &idx in &self.high_tree_b_data_const_indices {
                    execute_externals.insert(idx as u32);
                }
                // R0-R15 register tiles (Register8/Register64 → COP_CONST in
                // settle scope; their values are register state).
                for i in 0..16 {
                    execute_externals.insert(self.reg_indices[i] as u32);
                }
                // Flags (set by Stage X; commit/branch reads them — be safe).
                execute_externals.insert(self.flag_z_idx as u32);
                execute_externals.insert(self.flag_c_idx as u32);

                // Step 2: forward closure walk in topological order.
                // Track tainted tiles (execute) and their settle-internal
                // outputs. Untainted tiles whose all inputs are also untainted
                // become decode tiles.
                let mut tainted: HashSet<u32> = execute_externals.clone();
                let mut decode_set: HashSet<u32> = HashSet::new();

                for op in &self.settle_compact_ops {
                    let idx = op.idx;

                    // COP_RAM is stateful — force execute.
                    if op.op == crate::simulation::COP_RAM {
                        tainted.insert(idx);
                        continue;
                    }

                    // COP_CONST: classify by execute_externals membership.
                    if op.op == crate::simulation::COP_CONST {
                        if execute_externals.contains(&idx) {
                            tainted.insert(idx);
                        } else {
                            decode_set.insert(idx);
                        }
                        continue;
                    }

                    // Standard ops: tainted iff any input is tainted.
                    let mut input_tainted = false;
                    for &i in &[op.in0, op.in1, op.in2] {
                        if i != u32::MAX && tainted.contains(&i) {
                            input_tainted = true;
                            break;
                        }
                    }
                    // COP_WIRE reads n[3] (DOWN); COP_GENERIC may read all 4.
                    // Be conservative and check all 4 neighbors for both.
                    if !input_tainted
                        && (op.op == crate::simulation::COP_WIRE
                            || op.op == crate::simulation::COP_GENERIC)
                    {
                        let n = sim.neighbors4_at(idx as usize);
                        for &i in n {
                            if i != u32::MAX && tainted.contains(&i) {
                                input_tainted = true;
                                break;
                            }
                        }
                    }

                    if input_tainted {
                        tainted.insert(idx);
                    } else {
                        decode_set.insert(idx);
                    }
                }

                // Step 3: build decode_input_indices = decode externals.
                // External = (any tile read as input by a settle op AND not
                // produced by any settle op) PLUS (COP_CONST tiles in
                // decode_set, since their values are externally injected).
                let settle_outputs: HashSet<u32> =
                    self.settle_compact_ops.iter().map(|op| op.idx).collect();
                let mut decode_inputs_set: HashSet<u32> = HashSet::new();
                for op in &self.settle_compact_ops {
                    let consider = |i: u32, set: &mut HashSet<u32>| {
                        if i != u32::MAX && !settle_outputs.contains(&i) && !tainted.contains(&i) {
                            set.insert(i);
                        }
                    };
                    consider(op.in0, &mut decode_inputs_set);
                    consider(op.in1, &mut decode_inputs_set);
                    consider(op.in2, &mut decode_inputs_set);
                    if op.op == crate::simulation::COP_WIRE
                        || op.op == crate::simulation::COP_GENERIC
                    {
                        let n = sim.neighbors4_at(op.idx as usize);
                        for &i in n {
                            consider(i, &mut decode_inputs_set);
                        }
                    }
                    // COP_CONST in decode_set is itself a decode external.
                    if op.op == crate::simulation::COP_CONST && decode_set.contains(&op.idx) {
                        decode_inputs_set.insert(op.idx);
                    }
                }
                let mut decode_inputs: Vec<u32> = decode_inputs_set.into_iter().collect();
                decode_inputs.sort_unstable();
                self.decode_input_indices = decode_inputs;

                // Step 4: decode_output_indices = decode tiles that are
                // *read by something outside the decode interior*. We only
                // need to restore values that downstream code observes —
                // decode tiles read solely by other decode tiles can stay
                // stale on cache hit (their values are recomputed on the
                // next miss). The "boundary" set:
                //   * decode tiles read as inputs by execute_compact_ops
                //     (in0/in1/in2; for COP_WIRE/COP_GENERIC also n[3])
                //   * decode tiles read as inputs by clock/branch/commit
                //     compact ops (downstream-of-settle scopes)
                // Excludes COP_CONST (externally set, never re-evaluated)
                // and any decode tile read only by other decode tiles
                // (interior — value irrelevant to anyone after settle).
                let scopes: [&[crate::simulation::CompactOp]; 4] = [
                    &self.execute_compact_ops,
                    &self.clock_compact_ops,
                    &self.branch_compact_ops,
                    &self.commit_compact_ops,
                ];
                let mut needed: HashSet<u32> = HashSet::new();
                for ops in scopes {
                    for op in ops {
                        let consider_input = |i: u32, set: &mut HashSet<u32>| {
                            if i != u32::MAX && decode_set.contains(&i) {
                                set.insert(i);
                            }
                        };
                        consider_input(op.in0, &mut needed);
                        consider_input(op.in1, &mut needed);
                        consider_input(op.in2, &mut needed);
                        if op.op == crate::simulation::COP_WIRE
                            || op.op == crate::simulation::COP_GENERIC
                        {
                            let n = sim.neighbors4_at(op.idx as usize);
                            for &i in n {
                                consider_input(i, &mut needed);
                            }
                        }
                    }
                }
                // Always exclude COP_CONST (externally set; never written by
                // any kernel pass — restoring is a no-op).
                let const_set: HashSet<u32> = self
                    .settle_compact_ops
                    .iter()
                    .filter(|op| op.op == crate::simulation::COP_CONST)
                    .map(|op| op.idx)
                    .collect();
                let mut decode_outputs: Vec<u32> = needed
                    .into_iter()
                    .filter(|i| !const_set.contains(i))
                    .collect();
                decode_outputs.sort_unstable();
                self.decode_output_indices = decode_outputs;

                // Step 5: execute_compact_ops = settle ops NOT in decode_set,
                // preserving topological order. Walks parallel wvia counter.
                let mut exec_ops: Vec<crate::simulation::CompactOp> = Vec::new();
                let mut exec_wvia: Vec<(usize, u8, u64)> = Vec::new();
                let mut wvia_idx = 0usize;
                for op in &self.settle_compact_ops {
                    let is_wvia = op.op == crate::simulation::COP_WVIA;
                    if !decode_set.contains(&op.idx) {
                        exec_ops.push(*op);
                        if is_wvia {
                            exec_wvia.push(self.settle_compact_wvia[wvia_idx]);
                        }
                    }
                    if is_wvia {
                        wvia_idx += 1;
                    }
                }
                self.execute_compact_ops = exec_ops;
                self.execute_compact_wvia = exec_wvia;

                // Step 6: execute_cone_set bitset (all execute op output
                // indices). Used as backbone_set in the hybrid kernel: every
                // execute op evals unconditionally on cache hit since
                // upstream execute taint may have changed.
                if !self.execute_compact_ops.is_empty() {
                    let max_idx = self
                        .execute_compact_ops
                        .iter()
                        .map(|op| op.idx as usize)
                        .max()
                        .unwrap_or(0);
                    let words = (max_idx / 64) + 1;
                    let mut cone = vec![0u64; words];
                    for op in &self.execute_compact_ops {
                        let i = op.idx as usize;
                        cone[i / 64] |= 1u64 << (i % 64);
                    }
                    self.execute_cone_set = cone;
                } else {
                    self.execute_cone_set.clear();
                }

                // Cache invariants tied to partition/index sets — rebuild clears.
                self.decode_cache.borrow_mut().clear();
            }
        }
        // Clear stale flag — compact ops now match current tile types.
        self.compact_ops_stale.set(false);
    }

    /// Sprint 304: Measure forward closure sizes per instruction family.
    /// Returns (family_name, seed_count, closure_size) for each family.
    /// Sprint 317/318: Trunk injection seeds (terminals + downstream dirty).
    fn trunk_inject_seeds_inner(&self) -> Vec<usize> {
        let mut seeds = vec![self.alu_a_trunk_terminal_idx, self.alu_b_trunk_terminal_idx];
        seeds.extend_from_slice(&self.alu_a_downstream_dirty);
        seeds.extend_from_slice(&self.alu_b_downstream_dirty);
        seeds.sort_unstable();
        seeds.dedup();
        seeds
    }

    #[cfg(test)]
    pub(crate) fn trunk_inject_seeds(&self) -> Vec<usize> {
        self.trunk_inject_seeds_inner()
    }

    /// Used to evaluate potential scope reduction for per-family settle kernels (G4).
    #[cfg(test)]
    pub(crate) fn measure_per_family_closures(
        &self,
        sim: &Simulation,
    ) -> Vec<(&'static str, usize, usize)> {
        let mut results = Vec::new();

        // Combined (current settle scope — baseline for comparison)
        let all_seeds = self.all_settle_dirty_seeds();
        let all_closure = sim.compute_forward_closure(&all_seeds, &self.pipeline_compact_ops);
        results.push(("combined (all)", all_seeds.len(), all_closure.len()));

        // Backbone only (no per-register R0-R7 dirty)
        let backbone_seeds = self.backbone_settle_seeds();
        let backbone_closure =
            sim.compute_forward_closure(&backbone_seeds, &self.pipeline_compact_ops);
        results.push((
            "backbone-only",
            backbone_seeds.len(),
            backbone_closure.len(),
        ));

        // Per-register R0-R7 families: backbone + single register
        for reg in 0..8 {
            let mut seeds = backbone_seeds.clone();
            seeds.extend_from_slice(self.register_settle_seeds(reg));
            seeds.sort_unstable();
            seeds.dedup();
            let closure = sim.compute_forward_closure(&seeds, &self.pipeline_compact_ops);
            // Use static lifetime strings for register names
            let name = match reg {
                0 => "backbone+R0",
                1 => "backbone+R1",
                2 => "backbone+R2",
                3 => "backbone+R3",
                4 => "backbone+R4",
                5 => "backbone+R5",
                6 => "backbone+R6",
                7 => "backbone+R7",
                _ => unreachable!(),
            };
            results.push((name, seeds.len(), closure.len()));
        }

        // Branch family: backbone only (branch scope is separate)
        results.push((
            "branch (=backbone)",
            backbone_seeds.len(),
            backbone_closure.len(),
        ));

        results
    }

    #[cfg(test)]
    pub(crate) fn read_super_mux_lanes(&self, sim: &Simulation) -> [u8; 4] {
        [
            sim.get_logic_value_by_idx(self.rom_selected_low_idx) as u8,
            sim.get_logic_value_by_idx(self.rom_selected_high_idx) as u8,
            sim.get_logic_value_by_idx(self.rom_selected_byte2_idx) as u8,
            sim.get_logic_value_by_idx(self.rom_selected_byte3_idx) as u8,
        ]
    }

    #[cfg(test)]
    pub(crate) fn debug_set_branch_lut_selector(&self, sim: &mut Simulation, selector: u8) {
        let masked = selector & 0x1F;
        // Sprint 107: drive physical sources. Set ctrl_b Mux output and flag
        // Register8 outputs directly, then dirty the L1 taps to propagate
        // through the physical assembly path.
        let kind = (masked & 0x07) as u64;
        let z = if (masked & 0x08) != 0 { u64::MAX } else { 0 };
        let c = if (masked & 0x10) != 0 { u64::MAX } else { 0 };
        sim.set_logic_value_by_idx(self.ctrl_b_mux_idx, kind);
        sim.set_logic_value_by_idx(self.flag_z_idx, z);
        sim.set_logic_value_by_idx(self.flag_c_idx, c);
        sim.dirty.mark_dirty(self.branch_ctrl_b_l1_tap_idx);
        sim.dirty.mark_dirty(self.branch_flag_z_l1_tap_idx);
        sim.dirty.mark_dirty(self.branch_flag_c_l1_tap_idx);
        self.mark_branch_dirty(sim);
        // Sprint 263: Levelized evaluation for branch scope.
        let _ = sim.propagate_levelized(&self.branch_eval_order);
    }

    pub fn read_reg_tap(&self, sim: &Simulation, reg: usize) -> u64 {
        if reg < 8 {
            sim.get_logic_value_by_idx(self.reg_tap_l1_indices[reg])
        } else if reg < 16 {
            // High-register taps are not part of the low-lane dirty fanout map,
            // so read the committed Register64 tile directly.
            sim.get_logic_value_by_idx(self.reg_indices[reg])
        } else {
            0
        }
    }

    pub fn pipeline_dirty_len(&self) -> usize {
        self.pipeline_dirty_indices.len()
    }

    pub fn write_reg(&self, sim: &mut Simulation, reg: usize, value: u64) {
        self.set_reg(sim, reg, value);
    }

    pub fn read_pc(&self, _sim: &Simulation) -> u32 {
        self.pc.get()
    }

    pub fn write_pc(&self, sim: &mut Simulation, value: u32) {
        let v = value & self.pc_phys_mask;
        self.pc.set(v);
        sim.set_logic_value_by_idx(self.pc_idx, v as u64);
        sim.set_logic_value_by_idx(self.pc_next_mux_idx, v as u64);
        sim.set_logic_value_by_idx(self.pc_ingress_idx, v as u64);
        // Sprint 126: Software enable Const stays permanently at 0 (all enables physical).
    }

    pub fn read_flag_z(&self, _sim: &Simulation) -> bool {
        self.flag_z.get()
    }

    pub fn read_flag_c(&self, _sim: &Simulation) -> bool {
        self.flag_c.get()
    }

    pub fn read_lr(&self) -> u32 {
        self.lr.get()
    }

    pub fn write_lr(&self, value: u32) {
        self.lr.set(value & self.pc_phys_mask);
    }

    pub fn is_halted(&self) -> bool {
        self.halted.get()
    }

    pub fn read_ram(&self, _sim: &Simulation, addr: usize) -> u64 {
        if addr < 128 {
            self.ram[addr].get()
        } else if addr - 128 < self.main_mem.len() {
            self.main_mem[addr - 128].get()
        } else {
            0
        }
    }

    /// Sprint 362 (Gate A): allocate `words` of extended main memory and widen
    /// the data-address mask to a power of two covering [0, 128 + words). Passing
    /// 0 resets to the default 7-bit (128-location) address space. The scratchpad
    /// RAM (0..127) and MMIO window (41..63) are unchanged; main memory occupies
    /// addresses 128 and up.
    pub fn configure_main_memory(&mut self, words: usize) {
        if words == 0 {
            self.main_mem = Vec::new();
            self.mem_addr_mask.set(0x7F);
            return;
        }
        let total = (128 + words).next_power_of_two().max(128);
        self.mem_addr_mask.set(total - 1);
        self.main_mem = vec![Cell::new(0u64); total - 128];
    }

    /// Sprint 362: extended main-memory word count (0 if unconfigured).
    pub fn main_mem_len(&self) -> usize {
        self.main_mem.len()
    }

    /// Sprint 369 (Gate B.2): enable the extended 8-bit instruction address space.
    /// Widens the physical PC mask to 0xFF and stores instructions 128.. in
    /// `program_ext`. The PC-mask via tile must already be set to 0xFF at wiring
    /// time (`V2Builder::with_extended_pc`) — this configures the executor side
    /// (software PC reads, fetch source, upper-half program store).
    ///
    /// HALT words in the upper half are self-target-patched exactly as the wiring
    /// path patches physical ROM (`v2_wiring.rs` ~690), with an 8-bit mask.
    pub fn enable_extended_pc(&mut self, program: &[u32]) {
        self.extended_pc = true;
        self.pc_addr_bits = 8;
        self.pc_phys_mask = 0xFF;
        self.program_ext = Self::build_program_ext(program, self.pc_phys_mask);
    }

    /// Sprint 370 (Gate B.3): enable the wide 16-bit instruction address space
    /// (up to 65,536 instructions). Sets `pc_phys_mask = 0xFFFF` and `wide_pc`,
    /// stores instructions 128.. in `program_ext`. The PC/LR tiles must already be
    /// Register64 and the PC-mask via set to 0xFFFF at wiring time
    /// (`V2Builder::with_wide_pc`). Implies the extended-PC fetch path.
    pub fn enable_wide_pc(&mut self, program: &[u32]) {
        self.extended_pc = true;
        self.wide_pc = true;
        self.pc_addr_bits = 16;
        self.pc_phys_mask = 0xFFFF;
        self.program_ext = Self::build_program_ext(program, self.pc_phys_mask);
    }

    /// Build the upper-half program store (instructions 128..), mirroring the
    /// ROM-build HALT self-target patch (opcode 0x01) with the active PC mask.
    fn build_program_ext(program: &[u32], mask: u32) -> Vec<u32> {
        if program.len() > 128 {
            program[128..]
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let addr = 128 + i;
                    if ((w >> 11) & 0x1F) as u8 == 0x01 {
                        (w & !0xFF) | ((addr as u32) & mask)
                    } else {
                        w
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Sprint 369: whether the extended 8-bit instruction address space is active.
    pub fn extended_pc(&self) -> bool {
        self.extended_pc
    }

    /// Sprint 370: whether the wide 16-bit PC (Register64 tiles) is active.
    pub fn wide_pc(&self) -> bool {
        self.wide_pc
    }

    /// Sprint 369: the physical PC mask (0x7F default, 0xFF when extended_pc,
    /// 0xFFFF when wide_pc).
    pub fn pc_phys_mask(&self) -> u32 {
        self.pc_phys_mask
    }

    /// Sprint 362: current data-address mask (0x7F by default).
    pub fn mem_addr_mask(&self) -> usize {
        self.mem_addr_mask.get()
    }

    /// Sprint 162: physical tile index for a RAM address (0-127).
    pub fn ram_tile_idx(&self, addr: usize) -> usize {
        self.ram_indices[addr]
    }

    // Sprint 117: debug_set_ram_write_addr removed ??? addr/data/enable are physical.
    // Use write_ram() for direct software RAM writes (test setup), or ST/STB
    // instructions for physical end-to-end writes.

    pub fn write_ram(&self, sim: &mut Simulation, addr: usize, value: u64) {
        if addr < 128 {
            self.ram[addr].set(value);
            sim.set_logic_value_by_idx(self.ram_indices[addr], value);
        } else if addr - 128 < self.main_mem.len() {
            self.main_mem[addr - 128].set(value);
        }
    }

    /// Sprint 294: Write flag_z to Cell mirror and physical tile.
    pub fn write_flag_z(&self, sim: &mut Simulation, value: bool) {
        self.flag_z.set(value);
        sim.set_logic_value_by_idx(self.flag_z_idx, value as u64);
    }

    /// Sprint 294: Write flag_c to Cell mirror and physical tile.
    pub fn write_flag_c(&self, sim: &mut Simulation, value: bool) {
        self.flag_c.set(value);
        sim.set_logic_value_by_idx(self.flag_c_idx, value as u64);
    }

    /// Sprint 294: Set halted state (Cell only — no physical tile).
    pub fn set_halted(&self, value: bool) {
        self.halted.set(value);
    }

    // ── Visualization accessors (v2-viz crate) ──

    /// Topological eval order for the pipeline scope (tile indices).
    pub fn pipeline_eval_order(&self) -> &[usize] {
        &self.pipeline_eval_order
    }

    /// Topological eval order for the branch scope (tile indices).
    pub fn branch_eval_order(&self) -> &[usize] {
        &self.branch_eval_order
    }

    /// Topological eval order for the commit scope (tile indices).
    pub fn commit_eval_order(&self) -> &[usize] {
        &self.commit_eval_order
    }

    /// L1 bitset marking tiles in the pipeline scope.
    pub fn pipeline_scope_mask(&self) -> &[u64] {
        &self.pipeline_scope_mask
    }

    /// L1 bitset marking tiles in the branch scope.
    pub fn branch_scope_mask(&self) -> &[u64] {
        &self.branch_scope_mask
    }

    /// L1 bitset marking tiles in the commit scope.
    pub fn commit_scope_mask(&self) -> &[u64] {
        &self.commit_scope_mask
    }

    /// Sprint 166: Test-only accessor for clock scope mask coverage verification.
    #[cfg(test)]
    pub(crate) fn clock_scope_mask_test_info(&self) -> ClockScopeMaskTestInfo<'_> {
        ClockScopeMaskTestInfo {
            clock_scope_mask: &self.clock_scope_mask,
            pipeline_scope_mask: &self.pipeline_scope_mask,
            branch_scope_mask: &self.branch_scope_mask,
            commit_scope_mask: &self.commit_scope_mask,
            in_scope_clock_cache: &self.in_scope_clock_cache,
            commit_dirty_count: self.commit_dirty_indices.len(),
            reg_wb_dirty_count: self.reg_wb_dirty_indices.len(),
            reg_indices: &self.reg_indices,
            flag_z_idx: self.flag_z_idx,
            flag_c_idx: self.flag_c_idx,
            ram_indices: &self.ram_indices,
            pc_idx: self.pc_idx,
        }
    }
}

/// Sprint 166: Test-only struct for clock scope mask coverage verification.
#[cfg(test)]
pub(crate) struct ClockScopeMaskTestInfo<'a> {
    pub clock_scope_mask: &'a [u64],
    pub pipeline_scope_mask: &'a [u64],
    pub branch_scope_mask: &'a [u64],
    pub commit_scope_mask: &'a [u64],
    pub in_scope_clock_cache: &'a [usize],
    // Sprint 169: commit path partition sizes.
    pub commit_dirty_count: usize,
    pub reg_wb_dirty_count: usize,
    pub reg_indices: &'a [usize; 16],
    pub flag_z_idx: usize,
    pub flag_c_idx: usize,
    pub ram_indices: &'a [usize; 128],
    pub pc_idx: usize,
}
