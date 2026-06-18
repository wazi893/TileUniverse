//! Standalone builder for V2 hybrid CPU.

use std::cell::Cell;
use std::rc::Rc;

use crate::simulation::Simulation;
use crate::tile_cpu::v2_execute::TileCpuV2;
use crate::tile_cpu::v2_mmio::V2MmioHandle;
use crate::tile_cpu::v2_mmio_devices::V2LinkMailboxDevice;
use crate::tile_cpu::v2_wiring::{V2CpuIndices, wire_v2_cpu};
use crate::tiles::tile_meta::TileType;

use crate::synth::bridge::BridgeConfig;
use crate::synth::integration::{
    build_branch_taken_aig, build_clz_final_combine_aig, build_clz_half_combine_aig,
    build_clz8_aig, build_combined_decode_aig, build_ctrl_b_aig, build_decoder3to8_aig,
    build_mul_aig, build_popcnt_add_aig, build_popcnt8_aig, build_sra_sign_ext_aig,
    inject_synth_export_z, place_const_guard_region, precompute_block_cache,
};
use crate::synth::{CellLibrary, PlaceConfig, RouteConfig, SynthConfig, synthesize_to_simulation};

const V2_MAX_ROM_WORDS: usize = 128;
const V2_MAX_RAM_CELLS: usize = 128;

/// Configuration for synth-generated blocks.
///
/// Five decode gates: branch, rd_decode, ram_decode, ctrl_b (physically placed AIG),
/// ctrl_a (authority via CTRL_A_LUT injection into we_mask/flag_we_mask Const tiles).
/// Plus operand_bypass: always reads operands from reg_indices instead of physical trees.
///
/// Sprint 209: `enable_combined_decode` provides a unified ctrl_a+ctrl_b value source
/// from a single cached AIG evaluation. Mutually exclusive with `enable_ctrl_b`.
///
/// Sprint 222: `physical_decode` reads ctrl_a/ctrl_b from physical Mux16to1 decoders
/// for bank0, with LUT fallback on mismatch. Const injection delivery unchanged.
#[derive(Clone, Debug, Default)]
pub struct V2SynthConfig {
    pub enable_branch: bool,
    pub enable_rd_decode: bool,
    pub enable_ram_decode: bool,
    pub enable_ctrl_b: bool,
    /// Sprint 205: ctrl_a authority — injects we_mask and flag_we_mask via Const tile swap.
    pub enable_ctrl_a: bool,
    /// Sprint 205: operand bypass — always read from reg_indices, skip physical trees.
    pub enable_operand_bypass: bool,
    /// Sprint 209: combined ctrl_a+ctrl_b decode from a single AIG evaluation.
    /// Mutually exclusive with `enable_ctrl_b`. The ctrl_a authority injection path
    /// (`enable_ctrl_a`) is still required and operates independently.
    pub enable_combined_decode: bool,
    /// Sprint 218: ALU readback dual-path verification. When enabled, reads the
    /// physical ALU result mux output and compares against software every cycle.
    /// Observational only — does not change authority.
    pub enable_alu_readback: bool,
    /// Sprint 219: Physical ALU authority — bitmask of opcodes where the physical
    /// ALU result replaces software computation. 0 = all software (default).
    /// Use `PHYSICAL_ALU_ALL_BASIC` for reg-reg ops (0x04-0x0F),
    /// `PHYSICAL_ALU_IMM8` for 8-bit immediate ops (0x10-0x14), or combine both.
    pub physical_alu_opcodes: u32,
    /// Sprint 220: Physical register writeback authority. When true, R0-R7 are
    /// un-elided from the clock cache and software register injection is skipped
    /// for physically-authoritative ALU ops. Requires `physical_alu_opcodes != 0`.
    pub physical_reg_writeback: bool,
    /// Sprint 221: Physical flag writeback authority. When true, flag_z and flag_c
    /// Register8 tiles are un-elided from the clock cache and software flag injection
    /// is skipped for flag-writing instructions.
    pub physical_flag_writeback: bool,
    /// Sprint 222: Physical ctrl_a/ctrl_b authority for bank0. When true, Stage F
    /// reads ctrl_b from the physical Mux16to1 decoder (bank0, LUT fallback on
    /// mismatch), and inject_synth_pre_commit reads ctrl_a from the physical
    /// decoder instead of software LUT/synth. Delivery mechanism (Const injection
    /// for we_mask/flag_we_mask) unchanged. Requires physical_alu_opcodes != 0.
    pub physical_decode: bool,
    /// Sprint 225: Physical branch direction authority. When true, the physical
    /// Mux16to1 branch LUT computes branch_taken — synth injection is skipped.
    /// The Mux tile stays as Mux (not swapped to Const). Dual-path verification
    /// compares physical result against software truth table. Requires enable_branch.
    pub physical_branch: bool,
    /// Sprint 229: Physical RAM writeback authority. When true, RAM tiles are
    /// un-elided from the clock cache and participate in clock edge captures.
    /// The Ram tile's built-in WE gate (UP!=0 ? LEFT : current) provides physical
    /// write-enable gating. Diagnostic counters verify correctness; countdown
    /// writeback remains as safety net during Phase 1 (diagnostic).
    pub physical_ram_writeback: bool,
    /// Sprint 231: Physical operand authority. When true, Stage F reads operand
    /// values from physical tree roots (Tree A/B) instead of the software mirror
    /// for R0-R7 on bank 0. Requires `enable_operand_bypass` (for dual-path
    /// comparison). Falls back to software mirror on mismatch.
    pub physical_operand_authority: bool,
    /// Sprint 233: Physical rd_decode delivery authority. When true, the physical
    /// Decoder3to8 tile is NOT swapped to Const. inject_synth_pre_commit reads the
    /// physical one-hot output and injects only on mismatch. Requires enable_rd_decode.
    pub physical_rd_decode: bool,
    /// Sprint 233: Physical we_mask delivery authority. When true, the physical
    /// And tile (rd_onehot & reg_write) is NOT swapped to Const. inject_synth_pre_commit
    /// reads the physical output and injects only on mismatch. Requires enable_ctrl_a.
    pub physical_we_mask: bool,
    /// Sprint 233: Physical flag_we_mask delivery authority. When true, the physical
    /// WeightedViaUp tile is NOT swapped to Const. inject_synth_pre_commit reads the
    /// physical output and injects only on mismatch. Requires enable_ctrl_a.
    pub physical_flag_we_mask: bool,
    /// Sprint 234: Physical Super Mux propagation. When true, LEFT/RIGHT Const tiles
    /// are injected with ROM lane values and the Mux tile physically selects via PC[6].
    /// Output is verified against software-read rom_lanes; fallback on mismatch.
    pub physical_super_mux: bool,
    /// Sprint 235: Physical writeback-data authority for non-ALU register-producing ops.
    /// When true, LDI (narrow), MOV, and conditional moves (when taken) trust the
    /// physical writeback path — the register clock capture is verified instead of
    /// software-injected. Bank 0, R0-R7 only. Requires `physical_reg_writeback`.
    pub physical_wb_data_authority: bool,
    /// Sprint 236: Physical high-register writeback authority. When true, R8-R15
    /// participate in clock-edge capture via new merge mux fabric. Authority claimed
    /// for LDI narrow, MOV (low→high), conditional moves (low→high, taken) on bank 0.
    /// Requires `physical_wb_data_authority` (shares L-enable injection).
    pub physical_high_reg_writeback: bool,
    /// Sprint 245: Sub-function ALU delivery authority. When true, the software-computed
    /// result for sub-fn ops (MUL, SRA, CLZ, CTZ, POPCNT) is injected into wb_data_mux
    /// via Const-swap before commit propagation. The physical delivery path carries the
    /// value to the register — no set_logic_value_by_idx for register writeback.
    /// Flags remain software (MUL carry differs from physical ADD carry).
    /// Requires `physical_reg_writeback`.
    pub physical_sub_fn_delivery: bool,
    /// Sprint 246: Sub-function flag authority. When true, physical flag capture is
    /// trusted for sub-fn ops: SRA (Z+C correct), CLZ/CTZ/POPCNT (Z correct, C not
    /// written), MUL (Z correct, C_WE suppressed via flag_we_mask override).
    /// Requires `physical_sub_fn_delivery` + `physical_flag_writeback`.
    pub physical_sub_fn_flag_authority: bool,
    /// Sprint 246: LD/LDB delivery + flag authority. When true, the loaded value is
    /// injected into wb_data_mux before commit propagation. Physical delivery path
    /// handles register writeback (R0-R7; R8-R15 via high-reg merge mux). Physical
    /// flag Z capture is trusted (C_WE=0 for loads, so flag C unchanged).
    /// Requires `physical_reg_writeback`.
    pub physical_load_delivery: bool,
    /// Sprint 247: RAM store authority. When true, ST/STB skips the software write
    /// to the physical Ram tile (`set_logic_value_by_idx`). The physical clock-edge
    /// capture (Sprint 244 WE gate Const-swap) is authoritative: data bus (LEFT)
    /// carries store data from tree_a, WE gate (UP) selects target cell.
    /// Requires `physical_ram_writeback`.
    pub physical_ram_store_authority: bool,
    /// Sprint 248: SRA synth computation. When true, the SRA sign-extension
    /// overlay is computed by a 4-input AIG synth block instead of pure software.
    /// The physical ALU's SHR result is combined with the synth block's sign
    /// extension mask to produce the correct SRA result.
    /// Requires `physical_sub_fn_delivery`.
    pub physical_sra_computation: bool,
    /// Sprint 362: MUL synth computation. When true, the MUL sub-function
    /// (opcode 0x04, imm5=1) is computed by a 16-input AIG synth block instead
    /// of pure software. The physical ALU does not natively support MUL, so the
    /// result is injected into wb_data_mux via the synth block evaluation.
    /// Requires `physical_sub_fn_delivery`.
    pub physical_mul_computation: bool,
    /// Sprint 250: CLZ/CTZ/POPCNT AIG computation. When true, the bit-manipulation
    /// sub-functions (opcode 0x09, imm5 1-3) are computed by 64-input AIG blocks
    /// evaluated in software (blocks too large for physical placement at 128-wide).
    /// The AIG results replace software intrinsics and serve as authoritative.
    /// Requires `physical_sub_fn_delivery`.
    pub physical_bitop_computation: bool,
    /// Sprint 255: Physical IR spine. When true, Super Mux outputs are physically
    /// routed through elevated z-planes (z=14 for ir_low, z=15 for ir_high) to the
    /// extraction ingress. Eliminates Const-swap injection for upper-bank IR delivery.
    /// The spine delivers correct IR for ALL banks (Super Mux selects via PC[6]).
    /// Requires `physical_super_mux` and 16 layers.
    pub physical_ir_spine: bool,
    /// Sprint 258: Computational via decode for high-tree operand selectors.
    /// When true, the 14 selector Const tiles in High Trees A/B are replaced with
    /// WeightedViaUp tiles that extract rd[0:2] from ir_high and rs[0:2] from ir_low
    /// via z-plane delivery spines. Eliminates 14 selector injections per cycle.
    /// Tree B sel2 (rs[2]) uses a Zero inversion circuit for bank-swap polarity.
    /// Requires `physical_ir_spine` (the z-plane spines are the signal source).
    pub physical_via_decode: bool,
    /// Sprint 259: Physical byte2 selector authority. When true, the top-mux
    /// rd_hi/rs_hi selectors are extracted from the physical Super Mux byte 2
    /// tile output (rom_selected_byte2_idx) instead of software-computed ir_ext.
    /// No new tiles — the physical tile is read and bits injected into existing Consts.
    pub physical_byte2_selector: bool,
}

/// Bitmask for basic register-register ALU opcodes (0x04-0x0F): ADD, SUB, AND,
/// OR, XOR, NOT, INC, DEC, SHL, SHR + reserved 0x0A/0x0B.
/// Does NOT include immediate ALU ops (0x10-0x14) — use PHYSICAL_ALU_IMM8 for those.
pub const PHYSICAL_ALU_ALL_BASIC: u32 = 0xFFF0;

/// Sprint 226: Bitmask for 8-bit immediate ALU opcodes (0x10-0x14): ADDI, SUBI,
/// ANDI, ORI, XORI. Physical path uses use_imm=1 → R Mux selects ir_low (imm8).
/// Wide-immediate (.W) variants are excluded at runtime (physical path only carries imm8).
pub const PHYSICAL_ALU_IMM8: u32 = 0x1F_0000;

impl V2SynthConfig {
    /// Sprint 231: Enable all synth gates + combined decode + operand bypass +
    /// full physical authority (ALU, reg writeback, flag writeback, decode, branch,
    /// RAM writeback, operand authority).
    pub fn auto() -> Self {
        Self {
            enable_branch: true,
            enable_rd_decode: true,
            enable_ram_decode: true,
            enable_ctrl_b: false, // superseded by combined_decode
            enable_ctrl_a: true,
            enable_operand_bypass: true,
            enable_combined_decode: true,
            enable_alu_readback: false,
            physical_alu_opcodes: PHYSICAL_ALU_ALL_BASIC | PHYSICAL_ALU_IMM8,
            physical_reg_writeback: true,
            physical_flag_writeback: true,
            physical_decode: true,
            physical_branch: true,
            physical_ram_writeback: true,
            physical_operand_authority: true,
            physical_rd_decode: true,
            physical_we_mask: true,
            physical_flag_we_mask: true,
            physical_super_mux: true,
            physical_wb_data_authority: true,
            physical_high_reg_writeback: true,
            physical_sub_fn_delivery: true,
            physical_sub_fn_flag_authority: true,
            physical_load_delivery: true,
            physical_ram_store_authority: true,
            physical_sra_computation: true,
            physical_mul_computation: false, // requires synthesis; enable separately
            physical_bitop_computation: false, // requires 14-layer grid; use auto_bitop()
            physical_ir_spine: false,        // requires 16-layer grid; use auto_ir_spine()
            physical_via_decode: false,      // requires physical_ir_spine; use auto_via_decode()
            physical_byte2_selector: false,
        }
    }

    /// Sprint 250/251: Enable all auto() flags plus hierarchical CLZ/CTZ/POPCNT synth blocks.
    /// Requires a 14-layer grid (4 existing + 3 bitscan + 4 popcnt z-plane pairs).
    pub fn auto_bitop() -> Self {
        Self {
            physical_bitop_computation: true,
            ..Self::auto()
        }
    }

    /// Sprint 255: Enable all auto_bitop() flags plus physical IR spine routing.
    /// Requires a 16-layer grid (14 existing + 2 IR spine routing planes).
    pub fn auto_ir_spine() -> Self {
        Self {
            physical_ir_spine: true,
            ..Self::auto_bitop()
        }
    }

    /// Sprint 258: Enable all auto_ir_spine() flags plus computational via decode
    /// for high-tree operand selectors (rd[0:2] from ir_high, rs[0:2] from ir_low).
    /// Requires a 16-layer grid (same as auto_ir_spine).
    pub fn auto_via_decode() -> Self {
        Self {
            physical_via_decode: true,
            ..Self::auto_ir_spine()
        }
    }

    /// Sprint 259: Enable all auto_via_decode() + physical byte2 selector.
    pub fn auto_byte2_selector() -> Self {
        Self {
            physical_byte2_selector: true,
            ..Self::auto_via_decode()
        }
    }

    /// Sprint 259: Maximum physical authority preset.
    ///
    /// Enables every physical authority path including the 14-layer z-plane annex
    /// for hierarchical CLZ/CTZ/POPCNT and the 2-layer IR spine for physical
    /// upper-bank IR delivery. This is the target configuration for a fully
    /// physically-authentic CPU fabric (~100% computation authority).
    ///
    /// Requires: `Simulation::with_size_layered(128, 640, 16)`.
    pub fn max_authority() -> Self {
        Self::auto_byte2_selector()
    }

    /// Sprint 205 legacy: all synth gates without combined decode + physical branch.
    pub fn auto_separate() -> Self {
        Self {
            enable_branch: true,
            enable_rd_decode: true,
            enable_ram_decode: true,
            enable_ctrl_b: true,
            enable_ctrl_a: true,
            enable_operand_bypass: true,
            enable_combined_decode: false,
            enable_alu_readback: false,
            physical_alu_opcodes: 0,
            physical_reg_writeback: false,
            physical_flag_writeback: false,
            physical_decode: false,
            physical_branch: false,
            physical_ram_writeback: false,
            physical_operand_authority: false,
            physical_rd_decode: false,
            physical_we_mask: false,
            physical_flag_we_mask: false,
            physical_super_mux: false,
            physical_wb_data_authority: false,
            physical_high_reg_writeback: false,
            physical_sub_fn_delivery: false,
            physical_sub_fn_flag_authority: false,
            physical_load_delivery: false,
            physical_ram_store_authority: false,
            physical_sra_computation: false,
            physical_mul_computation: false,
            physical_bitop_computation: false,
            physical_ir_spine: false,
            physical_via_decode: false,
            physical_byte2_selector: false,
        }
    }
}

/// Builder for the 16-bit V2 hybrid CPU.
#[derive(Debug, Clone)]
pub struct V2Builder {
    origin: (usize, usize),
    program: Vec<u32>,
    rom_size: usize,
    ram_size: usize,
    main_mem_words: usize,
    extended_pc: bool,
    /// Sprint 370 (Gate B.3): wide 16-bit PC (Register64 tiles, up to 65,536 instrs).
    wide_pc: bool,
    mmio: Option<V2MmioHandle>,
    synth_config: Option<V2SynthConfig>,
}

impl Default for V2Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl V2Builder {
    /// Create a new V2 builder with defaults.
    pub fn new() -> Self {
        Self {
            origin: (0, 0),
            program: Vec::new(),
            rom_size: V2_MAX_ROM_WORDS,
            ram_size: V2_MAX_RAM_CELLS,
            main_mem_words: 0,
            extended_pc: false,
            wide_pc: false,
            mmio: None,
            synth_config: None,
        }
    }

    /// Set top-left origin on the simulation grid.
    pub fn with_origin(mut self, x: usize, y: usize) -> Self {
        self.origin = (x, y);
        self
    }

    /// Set V2 program words (32-bit: bits 15-0 = instruction, bits 31-16 = extension).
    pub fn with_program(mut self, program: &[u32]) -> Self {
        self.program = program.to_vec();
        self
    }

    /// Set ROM size in words (clamped to 16 for Sprint 90).
    pub fn with_rom_size(mut self, words: usize) -> Self {
        self.rom_size = words.min(V2_MAX_ROM_WORDS);
        self
    }

    /// Set RAM size in cells (clamped to 128 for Sprint 189).
    pub fn with_ram_size(mut self, cells: usize) -> Self {
        self.ram_size = cells.min(V2_MAX_RAM_CELLS);
        self
    }

    /// Sprint 362 (Gate A): allocate `words` of extended main memory (the "DRAM
    /// analog"), addressable at 128.. via flat LD/ST. The 128-cell scratchpad
    /// (tile-backed) and MMIO window are unchanged; this widens the data-address
    /// mask to a power of two covering the extra space. 0 = default 7-bit space.
    pub fn with_main_memory(mut self, words: usize) -> Self {
        self.main_mem_words = words;
        self
    }

    /// Sprint 369 (Gate B.2): enable the extended 8-bit instruction address space
    /// (PC mask 0x7F -> 0xFF, up to 256 instructions). Programs may then occupy ROM
    /// entries 0..255; entries 0..127 live in physical ROM tiles, 128..255 in a
    /// software-backed store fetched and injected through the Super-Mux Const path.
    /// Use JMPR/CALLR (register-indirect far-jump) to reach past the 8-bit branch
    /// immediate. PC sequencing for the extended range is software-authoritative.
    pub fn with_extended_pc(mut self) -> Self {
        self.extended_pc = true;
        self
    }

    /// Sprint 370 (Gate B.3): enable the wide 16-bit instruction address space (up
    /// to 65,536 instructions). The physical PC and LR registers become Register64
    /// and the PC-mask via widens to 0xFFFF, so the PC physically holds and
    /// increments a >8-bit value (true physical sequencing past the 256 line).
    /// Implies the extended-PC fetch path (instructions 128.. from the software
    /// store). Far jumps use JMPR/CALLR (register-indirect).
    pub fn with_wide_pc(mut self) -> Self {
        self.wide_pc = true;
        self.extended_pc = true;
        self
    }

    /// Sprint 370: instruction-address width in bits (7 default, 8 extended, 16 wide).
    fn pc_addr_bits(&self) -> u8 {
        if self.wide_pc {
            16
        } else if self.extended_pc {
            8
        } else {
            7
        }
    }

    /// Attach an MMIO device for reserved addresses.
    pub fn with_mmio(mut self, mmio: V2MmioHandle) -> Self {
        self.mmio = Some(mmio);
        self
    }

    /// Sprint 201: Enable synth-generated replacement blocks.
    /// Requires grid height >= 192 (CPU core 0..63, guard 64..69, synth zone 70+).
    pub fn with_synth_blocks(mut self, config: V2SynthConfig) -> Self {
        self.synth_config = Some(config);
        self
    }

    /// Re-derive the physical tile-index layout for this builder's config WITHOUT
    /// consuming the builder or mutating the real sim. The tile indices are layout
    /// positions (content-independent), so wiring a throwaway `scratch` sim of the
    /// same dimensions yields the same `V2CpuIndices` the real `build()` produces.
    ///
    /// This is the no-`v2_execute.rs`-edit hook the V2→CUDA device-cycle model uses to
    /// locate injection/readback tiles (the named index fields are private on
    /// `TileCpuV2`). Additive and golden-neutral — `build()` is unchanged.
    ///
    /// Returns a LAYOUT-only catalog: only tile-index fields are populated; the
    /// scope/eval-order/cache fields `build_core` computes are intentionally empty here.
    #[allow(dead_code)] // consumed by the V2→CUDA device-cycle driver (task #5).
    pub(crate) fn derive_indices(&self, scratch: &mut Simulation) -> V2CpuIndices {
        wire_v2_cpu(
            scratch,
            self.origin,
            &self.program,
            self.rom_size,
            self.ram_size,
            self.pc_addr_bits(),
        )
    }

    /// Internal: shared CPU construction (everything except wire chain fusion).
    /// Returns (cpu, placed_mask) for optional deferred chain finalization.
    fn build_core(self, sim: &mut Simulation) -> (TileCpuV2, Vec<u64>) {
        assert!(
            sim.num_layers() >= 4,
            "V2 Sprint 118 requires layered simulation (use Simulation::with_size_layered(..., ..., 4))"
        );

        let mut idx = wire_v2_cpu(
            sim,
            self.origin,
            &self.program,
            self.rom_size,
            self.ram_size,
            self.pc_addr_bits(),
        );

        // Sprint 147: compute zone closures with placed-tile restriction.
        // Only follows tiles that were explicitly placed during wiring, preventing
        // BFS leakage through unplaced default Wire tiles (especially on L1).
        idx.pipeline_scope =
            sim.compute_zone_closure_restricted(&idx.pipeline_dirty_indices, &idx.placed_mask);
        idx.branch_scope =
            sim.compute_zone_closure_restricted(&idx.branch_dirty_indices, &idx.placed_mask);
        // Sprint 169: commit scope = union of PC-commit + reg-WB + flag-commit.
        let mut commit_seeds = idx.commit_dirty_indices.clone();
        commit_seeds.extend_from_slice(&idx.reg_wb_dirty_indices);
        commit_seeds.extend_from_slice(&idx.flag_commit_dirty_indices);
        idx.commit_scope = sim.compute_zone_closure_restricted(&commit_seeds, &idx.placed_mask);

        // Sprint 147: convert Vec<u32> scopes to bitset masks for masked drain.
        let num_segments = (sim.tile_count() + 63) / 64;
        idx.pipeline_scope_mask = scope_to_mask(&idx.pipeline_scope, num_segments);
        idx.branch_scope_mask = scope_to_mask(&idx.branch_scope, num_segments);
        idx.commit_scope_mask = scope_to_mask(&idx.commit_scope, num_segments);

        // Sprint 166: clock scope = union of pipeline + branch + commit scopes.
        // Covers all tiles that could be affected by the rising clock edge.
        idx.clock_scope_mask = (0..num_segments)
            .map(|i| {
                idx.pipeline_scope_mask[i] | idx.branch_scope_mask[i] | idx.commit_scope_mask[i]
            })
            .collect();

        // Sprint 168: Pre-compute clock-sensitive tiles within the clock scope.
        // At runtime, the lightweight clock edge seeds ONLY these tiles (~20-50)
        // instead of the global clock_sensitive_cache (~500-1000).
        idx.in_scope_clock_cache = sim
            .tilemap
            .tiles
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                t.meta.tile_type.is_clock_sensitive() && {
                    let seg = i / 64;
                    let bit = i % 64;
                    seg < idx.clock_scope_mask.len()
                        && (idx.clock_scope_mask[seg] & (1u64 << bit)) != 0
                }
            })
            .map(|(i, _)| i)
            .collect();

        // Sprint 262: Build topologically sorted evaluation order for levelized propagation.
        idx.pipeline_eval_order = sim.build_eval_order(&idx.pipeline_scope_mask);
        // Sprint 263: branch + commit scope eval orders.
        idx.branch_eval_order = sim.build_eval_order(&idx.branch_scope_mask);
        idx.commit_eval_order = sim.build_eval_order(&idx.commit_scope_mask);
        // Sprint 276: Clock scope eval order for compact clock edge cascade.
        idx.clock_eval_order = sim.build_eval_order(&idx.clock_scope_mask);

        // Sprint 172: Save placed_mask before idx is consumed by from_wiring.
        let placed_mask_for_chains = idx.placed_mask.clone();
        // Sprint 276: Save clock eval order before idx is consumed.
        let clock_eval_order = std::mem::take(&mut idx.clock_eval_order);

        let mut cpu = TileCpuV2::from_wiring(self.origin, idx, self.mmio);

        // Sprint 362 (Gate A): allocate extended main memory if requested.
        if self.main_mem_words > 0 {
            cpu.configure_main_memory(self.main_mem_words);
        }

        // Sprint 369/370 (Gate B.2/B.3): configure the executor side of the extended
        // / wide PC address space (PC mask, upper-half program store). The PC-mask
        // via + PC/LR register types were already set during wiring.
        if self.wide_pc {
            cpu.enable_wide_pc(&self.program);
        } else if self.extended_pc {
            cpu.enable_extended_pc(&self.program);
        }

        // Sprint 267: Build compact ops from pipeline eval order.
        {
            let (ops, wvia) = sim.build_compact_ops(&cpu.pipeline_eval_order);
            cpu.pipeline_compact_ops = ops;
            cpu.pipeline_compact_wvia = wvia;
        }
        // Sprint 270: Build compact ops for branch + commit scopes.
        {
            let (ops, wvia) = sim.build_compact_ops(&cpu.branch_eval_order);
            cpu.branch_compact_ops = ops;
            cpu.branch_compact_wvia = wvia;
        }
        {
            let (ops, wvia) = sim.build_compact_ops(&cpu.commit_eval_order);
            cpu.commit_compact_ops = ops;
            cpu.commit_compact_wvia = wvia;
        }
        // Sprint 278: Build ordered active-work schedule for commit scope.
        cpu.commit_schedule = Some(sim.build_compact_schedule(&cpu.commit_eval_order, false));
        // Sprint 279: Build schedules for branch and clock scopes.
        cpu.branch_schedule = Some(sim.build_compact_schedule(&cpu.branch_eval_order, false));
        cpu.clock_schedule = Some(sim.build_compact_schedule(&clock_eval_order, true));
        // Sprint 276: Build compact ops for clock scope cascade.
        // Use build_compact_ops_clock so RAM tiles are COP_CONST (captured in delta 0 only).
        {
            let (ops, wvia) = sim.build_compact_ops_clock(&clock_eval_order);
            cpu.clock_compact_ops = ops;
            cpu.clock_compact_wvia = wvia;
        }
        // Sprint 271: Commit scope compact eval falls back to levelized when
        // wb_data_mux is Const-swapped (sub-fn/load/LDI.W delivery).

        // Sprint 272: Build cone-pruned pipeline compact ops.
        {
            let seeds = cpu.collect_stage_f_output_seeds();
            let (cone_ops, cone_wvia, cone_set) = TileCpuV2::build_pipeline_cone(
                &cpu.pipeline_compact_ops,
                &cpu.pipeline_compact_wvia,
                &seeds,
            );
            cpu.pipeline_cone_ops = cone_ops;
            cpu.pipeline_cone_wvia = cone_wvia;
            cpu.pipeline_cone_set = cone_set;

            // Sprint 284: JIT compilation moved to build() — after
            // rebuild_eval_order which clears pipeline_cone_jit.
        }

        // Sprint 173: Elide register/RAM/flag captures from clock cache.
        // Software writeback overwrites these tiles after every clock edge.
        cpu.elide_software_writeback_from_clock_cache();

        // Sprint 178: Use masked settle so only THIS CPU's placed tiles are evaluated.
        // Without masking, building a second CPU would re-evaluate the first CPU's
        // tiles during the global clock edge, re-capturing PC and corrupting state.
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.tick_with_delays_masked(&placed_mask_for_chains);
        sim.tick_with_delays_masked(&placed_mask_for_chains);

        (cpu, placed_mask_for_chains)
    }

    /// Build the V2 hybrid CPU and run settling ticks.
    pub fn build(self, sim: &mut Simulation) -> TileCpuV2 {
        let synth_config = self.synth_config.clone();
        let origin = self.origin;
        let (mut cpu, placed_mask) = self.build_core(sim);

        if let Some(config) = synth_config {
            setup_synth_blocks(&mut cpu, sim, &config, origin);
            // Sprint 262: Rebuild eval order after synth blocks extended the scope.
            cpu.rebuild_eval_order(sim);
        }

        // Sprint 172: Build wire chain fusion data AFTER initial settle.
        // This ensures tick_with_delays settle passes use the unmodified
        // eval_tile path (no chain fusion during delay-based sequential capture).
        sim.build_wire_chains(&placed_mask, 3);

        // Sprint 173: Remove chain tail members from dirty index vectors.
        // Tails are handled by chain fusion when the head is evaluated.
        cpu.filter_dirty_indices_for_chains(sim);

        // Sprint 284: JIT-compile cone ops AFTER rebuild_eval_order and chain
        // construction. rebuild_eval_order clears pipeline_cone_jit (Sprint 273.1),
        // so the JIT must be compiled as the final step.
        #[cfg(feature = "cranelift_jit")]
        {
            if let Err(e) = crate::tile_cpu::tile_jit::preflight_check(&cpu.pipeline_cone_ops) {
                eprintln!("JIT preflight rejected: {}", e);
            } else {
                match crate::tile_cpu::tile_jit::compile_tile_eval(
                    &cpu.pipeline_cone_ops,
                    &cpu.pipeline_cone_wvia,
                ) {
                    Ok(jit) => cpu.pipeline_cone_jit = Some(jit),
                    Err(e) => eprintln!("JIT cone compilation failed: {}", e),
                }
            }
            // Sprint 335: JIT-compile settle ops.
            if let Err(e) = crate::tile_cpu::tile_jit::preflight_check(&cpu.settle_compact_ops) {
                // Expected: settle may have COP_WIRE/COP_GENERIC/COP_THRESHOLD_VIA. Not fatal.
                let _ = e; // silently skip JIT for settle
            } else {
                match crate::tile_cpu::tile_jit::compile_tile_eval(
                    &cpu.settle_compact_ops,
                    &cpu.settle_compact_wvia,
                ) {
                    Ok(jit) => cpu.settle_jit = Some(jit),
                    Err(e) => eprintln!("JIT settle compilation failed: {}", e),
                }
            }
            // Sprint 352: JIT-compile backbone ops (filtered for JIT compatibility).
            // Backbone is structurally clean (no COP_WIRE/COP_GENERIC/COP_THRESHOLD_VIA) so preflight
            // should always pass, enabling JIT even when full settle JIT is blocked.
            if !cpu.backbone_ops.is_empty() {
                if let Err(e) = crate::tile_cpu::tile_jit::preflight_check(&cpu.backbone_ops) {
                    eprintln!("JIT backbone preflight rejected: {}", e);
                } else {
                    match crate::tile_cpu::tile_jit::compile_tile_eval(
                        &cpu.backbone_ops,
                        &cpu.backbone_wvia,
                    ) {
                        Ok(jit) => cpu.backbone_jit = Some(jit),
                        Err(e) => eprintln!("JIT backbone compilation failed: {}", e),
                    }
                }
            }
        }

        cpu
    }

    /// Sprint 178/203: Build the V2 CPU but skip wire chain construction.
    /// Returns (cpu, placed_mask) for deferred chain finalization via
    /// `finalize_multi_cpu_chains()`. Use this when building multiple CPUs
    /// in one grid — build all CPUs first, then finalize chains once.
    ///
    /// Sprint 203: Synth blocks are now supported. If `with_synth_blocks()`
    /// was called, synth blocks are installed after core wiring (mirrors
    /// the `build()` path). The caller must ensure the grid has sufficient
    /// height for synth zones (384 rows per CPU column).
    pub fn build_no_chains(self, sim: &mut Simulation) -> (TileCpuV2, Vec<u64>) {
        let synth_config = self.synth_config.clone();
        let origin = self.origin;
        let (mut cpu, placed_mask) = self.build_core(sim);
        if let Some(config) = synth_config {
            setup_synth_blocks(&mut cpu, sim, &config, origin);
            // Sprint 262: Rebuild eval order after synth blocks extended the scope.
            cpu.rebuild_eval_order(sim);
        }
        (cpu, placed_mask)
    }
}

/// Sprint 178: Merge placed masks and build wire chains for multi-CPU setups.
/// Must be called AFTER all CPUs are built and initially settled.
pub fn finalize_multi_cpu_chains(
    sim: &mut Simulation,
    cpus: &mut [&mut TileCpuV2],
    placed_masks: &[Vec<u64>],
) {
    assert!(
        !placed_masks.is_empty(),
        "finalize_multi_cpu_chains requires at least one placed_mask"
    );
    // 1. Merge all placed_masks (OR together)
    let num_segs = placed_masks[0].len();
    let mut merged = vec![0u64; num_segs];
    for mask in placed_masks {
        for (i, &word) in mask.iter().enumerate() {
            if i < num_segs {
                merged[i] |= word;
            }
        }
    }
    // 2. Build chains with merged mask
    sim.build_wire_chains(&merged, 3);
    // 3. Filter dirty indices for all CPUs
    for cpu in cpus.iter_mut() {
        cpu.filter_dirty_indices_for_chains(sim);
    }
}

/// Convert a sorted Vec<u32> scope to a bitset mask (one u64 per 64-tile segment).
fn scope_to_mask(scope: &[u32], num_segments: usize) -> Vec<u64> {
    let mut mask = vec![0u64; num_segments];
    for &idx in scope {
        let seg = idx as usize / 64;
        let bit = idx as usize % 64;
        if seg < mask.len() {
            mask[seg] |= 1u64 << bit;
        }
    }
    mask
}

/// Sprint 201: Build and inject synth blocks based on config.
///
/// All coordinates are origin-relative:
/// - Guard region: origin + (0..128, 64..70)
/// - Synth zone:   origin + (0..128, 70..)
///
/// Height requirement scales with block count (~50 rows per block).
fn setup_synth_blocks(
    cpu: &mut TileCpuV2,
    sim: &mut Simulation,
    config: &V2SynthConfig,
    origin: (usize, usize),
) {
    assert!(
        !(config.enable_combined_decode && config.enable_ctrl_b),
        "enable_combined_decode and enable_ctrl_b are mutually exclusive"
    );

    let (ox, oy) = origin;

    // Estimate height requirement: guard (6 rows) + ~50 rows per block + 2 gap each.
    // ctrl_a and combined_decode are software-only (no physical block).
    let block_count = [
        config.enable_branch,
        config.enable_rd_decode,
        config.enable_ram_decode,
        config.enable_ctrl_b,
        config.enable_ctrl_a,
        config.physical_sra_computation, // Sprint 248
    ]
    .iter()
    .filter(|&&b| b)
    .count();
    // ctrl_a is software-only (no physical block), so don't count it for height.
    let physical_blocks = block_count - if config.enable_ctrl_a { 1 } else { 0 };
    let min_height = oy + 70 + physical_blocks * 52;
    assert!(
        sim.tilemap.height >= min_height,
        "setup_synth_blocks: grid height {} too small for {block_count} synth block(s) \
         at origin ({ox},{oy}); need at least {min_height}",
        sim.tilemap.height
    );

    place_const_guard_region(sim, ox..ox + 128, oy + 64..oy + 70);

    let decode_route = RouteConfig {
        max_z: 1,
        no_crossings: true,
        ..RouteConfig::default()
    };

    let mut next_y = oy + 70;

    if config.enable_branch {
        let aig = build_branch_taken_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &RouteConfig::default(), 4);
        next_y += block.height + 2;
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_branch_block(block);
        // Sprint 225: When physical_branch is enabled, keep the Mux tile as Mux
        // (don't swap to Const). The physical LUT computes branch_taken; synth block
        // is retained for dual-path verification only.
        if !config.physical_branch {
            cpu.enable_synth_branch(sim);
        }
    }

    if config.enable_rd_decode {
        let aig = build_decoder3to8_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &decode_route, 4);
        next_y += block.height + 2;
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_rd_decode_block(block);
        cpu.enable_synth_rd_decode(sim);
        // Sprint 233: When physical_rd_decode is enabled, restore the Decoder3to8 tile
        // from Const back to its original type. The SynthGate stays enabled (injection
        // still runs), but the physical tile computes the one-hot; injection reads it
        // and injects only on mismatch.
        if config.physical_rd_decode {
            cpu.restore_rd_decode_tile(sim);
        }
    }

    if config.enable_ram_decode {
        let aig = build_decoder3to8_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &decode_route, 4);
        next_y += block.height + 2;
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_ram_decode_block(block);
        cpu.enable_synth_ram_decode(sim);
    }

    if config.enable_ctrl_b {
        let aig = build_ctrl_b_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &decode_route, 4);
        next_y += block.height + 2;
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_ctrl_b_block(block);
        cpu.enable_synth_ctrl_b();
    }

    if config.enable_ctrl_a {
        // Sprint 205: ctrl_a authority — swaps we_mask/flag_we_mask tiles to Const,
        // values injected each cycle in inject_synth_pre_commit().
        cpu.enable_synth_ctrl_a(sim);
        // Sprint 233: Restore physical tiles when delivery authority is enabled.
        // The And tile (we_mask) and WeightedViaUp tile (flag_we_mask) stay live;
        // inject_synth_pre_commit reads their output and injects only on mismatch.
        if config.physical_we_mask {
            cpu.restore_we_mask_tile(sim);
        }
        if config.physical_flag_we_mask {
            cpu.restore_flag_we_mask_tile(sim);
        }
    }

    // Sprint 248: SRA sign-extension synth block.
    if config.physical_sra_computation {
        let aig = build_sra_sign_ext_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &decode_route, 4);
        next_y += block.height + 2;
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_sra_block(block);
    }

    // Sprint 362: MUL synth block.
    if config.physical_mul_computation {
        let aig = build_mul_aig();
        let mut block = build_and_inject_block(sim, &aig, ox, next_y, &decode_route, 4);
        precompute_block_cache(sim, &mut block);
        cpu.set_synth_mul_block(block);
        cpu.enable_synth_mul();
    }

    // Sprint 250: Hierarchical CLZ/CTZ — byte-sliced synth blocks on z-plane annex.
    // Shared bitscan family: CLZ and CTZ use the same physical blocks with different
    // input ordering. 11 blocks packed across 3 z-planes:
    //   z=2..3: byte blocks 0-3
    //   z=4..5: byte blocks 4-7
    //   z=6..7: half-combine 0, half-combine 1, final combine
    // All annex planes are independent of the z=0..1 existing synth zone.
    if config.physical_bitop_computation {
        assert!(
            config.physical_sub_fn_delivery,
            "physical_bitop_computation requires physical_sub_fn_delivery"
        );
        assert!(
            sim.num_layers() >= 14,
            "physical_bitop_computation requires 14 layers (4 existing + 3 bitscan + 4 popcnt z-plane pairs)"
        );

        let bitscan_route = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };

        let lib = CellLibrary::tile_native();
        let synth_cfg = SynthConfig::default();
        let place_cfg = PlaceConfig {
            halo: 4,
            ..PlaceConfig::default()
        };

        // Synthesize each unique AIG ONCE, then inject the export multiple times.
        // Bitscan family (CLZ/CTZ): 3 unique AIGs.
        let bitscan8_export = synthesize_to_simulation(
            &build_clz8_aig(),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("BITSCAN8 synth failed");
        let half_export = synthesize_to_simulation(
            &build_clz_half_combine_aig(),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("HALF_COMBINE synth failed");
        let final_export = synthesize_to_simulation(
            &build_clz_final_combine_aig(),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("FINAL_COMBINE synth failed");

        // Sprint 251: POPCNT family: 4 unique AIGs.
        let popcnt8_export = synthesize_to_simulation(
            &build_popcnt8_aig(),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("POPCNT8 synth failed");
        let add4_export = synthesize_to_simulation(
            &build_popcnt_add_aig(4),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("POPCNT_ADD4 synth failed");
        let add5_export = synthesize_to_simulation(
            &build_popcnt_add_aig(5),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("POPCNT_ADD5 synth failed");
        let add6_export = synthesize_to_simulation(
            &build_popcnt_add_aig(6),
            &lib,
            &synth_cfg,
            &place_cfg,
            &bitscan_route,
        )
        .expect("POPCNT_ADD6 synth failed");

        let annex_y = oy + 70;

        // Guard regions for all annex z-planes (7 pairs: z=2..13).
        for base_z in (2..14).step_by(2) {
            for gy in annex_y.saturating_sub(6)..annex_y {
                for gx in ox..ox + 128 {
                    sim.set_tile_3d(gx, gy, base_z, TileType::Const);
                    sim.set_tile_3d(gx, gy, base_z + 1, TileType::Const);
                }
            }
        }

        let bh = bitscan8_export.sim.tilemap.height;
        let hh = half_export.sim.tilemap.height;
        let ph = popcnt8_export.sim.tilemap.height;

        // Sprint 265: Cache bitop synth blocks using pure AIG evaluation (no tile
        // propagation). evaluate_aig is ~1000x faster than propagate_combinational_masked.
        use crate::synth::mapping::evaluate_aig;

        fn aig_precompute_cache(
            aig: &crate::synth::aig::Aig,
            block: &mut crate::synth::integration::InjectedBlock,
        ) {
            use crate::synth::integration::SynthOutputs;
            let n_inputs = block.input_indices.len();
            let n_combos = 1usize << n_inputs;
            let mut cache = Vec::with_capacity(n_combos);
            for pattern in 0..n_combos {
                let input_bools: Vec<bool> =
                    (0..n_inputs).map(|i| (pattern >> i) & 1 != 0).collect();
                let out_bools = evaluate_aig(aig, &input_bools);
                let mut outputs = SynthOutputs::new();
                for &b in &out_bools {
                    outputs.push(if b { u64::MAX } else { 0 });
                }
                cache.push(outputs);
            }
            block.outputs_cache = Some(cache);
        }

        let bitscan8_aig = build_clz8_aig();
        let half_aig = build_clz_half_combine_aig();
        let final_aig = build_clz_final_combine_aig();
        let popcnt8_aig = build_popcnt8_aig();
        let add4_aig = build_popcnt_add_aig(4);
        let add5_aig = build_popcnt_add_aig(5);
        let add6_aig = build_popcnt_add_aig(6);

        // z=2..3: bitscan8 blocks 0-3.
        let mut bitscan_byte_blocks: Vec<crate::synth::integration::InjectedBlock> = Vec::new();
        let mut z2_y = annex_y;
        for _ in 0..4 {
            let mut block = inject_synth_export_z(sim, &bitscan8_export, ox, z2_y, 2);
            aig_precompute_cache(&bitscan8_aig, &mut block); // 8 inputs, 256 combos
            z2_y += bh + 2;
            bitscan_byte_blocks.push(block);
        }

        // z=4..5: bitscan8 blocks 4-7.
        let mut z4_y = annex_y;
        for _ in 0..4 {
            let mut block = inject_synth_export_z(sim, &bitscan8_export, ox, z4_y, 4);
            aig_precompute_cache(&bitscan8_aig, &mut block);
            z4_y += bh + 2;
            bitscan_byte_blocks.push(block);
        }

        // z=6..7: bitscan combines (half0, half1, final).
        let mut z6_y = annex_y;
        let mut half0 = inject_synth_export_z(sim, &half_export, ox, z6_y, 6);
        // half_combine: 16 inputs = 65K combos — still fast with AIG eval.
        aig_precompute_cache(&half_aig, &mut half0);
        z6_y += hh + 2;
        let mut half1 = inject_synth_export_z(sim, &half_export, ox, z6_y, 6);
        aig_precompute_cache(&half_aig, &mut half1);
        z6_y += hh + 2;
        let mut bitscan_final = inject_synth_export_z(sim, &final_export, ox, z6_y, 6);
        aig_precompute_cache(&final_aig, &mut bitscan_final);

        let bitscan_arr: [_; 8] = bitscan_byte_blocks.try_into().unwrap();
        cpu.set_bitscan_blocks(bitscan_arr, [half0, half1], bitscan_final);

        // Sprint 251: POPCNT blocks on z=8..13.
        // z=8..9: popcnt8 blocks 0-3.
        let mut popcnt_byte_blocks: Vec<crate::synth::integration::InjectedBlock> = Vec::new();
        let mut z8_y = annex_y;
        for _ in 0..4 {
            let mut block = inject_synth_export_z(sim, &popcnt8_export, ox, z8_y, 8);
            aig_precompute_cache(&popcnt8_aig, &mut block);
            z8_y += ph + 2;
            popcnt_byte_blocks.push(block);
        }

        // z=10..11: popcnt8 blocks 4-7.
        let mut z10_y = annex_y;
        for _ in 0..4 {
            let mut block = inject_synth_export_z(sim, &popcnt8_export, ox, z10_y, 10);
            aig_precompute_cache(&popcnt8_aig, &mut block);
            z10_y += ph + 2;
            popcnt_byte_blocks.push(block);
        }

        // z=12..13: adder tree (4 × add4, 2 × add5, 1 × add6).
        let a4h = add4_export.sim.tilemap.height;
        let a5h = add5_export.sim.tilemap.height;
        let mut z12_y = annex_y;

        let mut add4_blocks: Vec<crate::synth::integration::InjectedBlock> = Vec::new();
        for _ in 0..4 {
            let mut block = inject_synth_export_z(sim, &add4_export, ox, z12_y, 12);
            aig_precompute_cache(&add4_aig, &mut block);
            z12_y += a4h + 2;
            add4_blocks.push(block);
        }

        let mut add5_blocks: Vec<crate::synth::integration::InjectedBlock> = Vec::new();
        for _ in 0..2 {
            let mut block = inject_synth_export_z(sim, &add5_export, ox, z12_y, 12);
            aig_precompute_cache(&add5_aig, &mut block);
            z12_y += a5h + 2;
            add5_blocks.push(block);
        }

        let mut add6_block = inject_synth_export_z(sim, &add6_export, ox, z12_y, 12);
        aig_precompute_cache(&add6_aig, &mut add6_block);

        let popcnt_arr: [_; 8] = popcnt_byte_blocks.try_into().unwrap();
        let add4_arr: [_; 4] = add4_blocks.try_into().unwrap();
        let add5_arr: [_; 2] = add5_blocks.try_into().unwrap();
        cpu.set_popcnt_blocks(popcnt_arr, add4_arr, add5_arr, add6_block);

        cpu.set_physical_bitop_computation(true);
    }

    // Sprint 255: Physical IR spine — elevated routing from Super Mux to extraction ingress.
    // ir_low on z=14, ir_high on z=15. Replaces Const-swap injection for upper-bank IR.
    if config.physical_ir_spine {
        assert!(
            config.physical_super_mux,
            "physical_ir_spine requires physical_super_mux"
        );
        assert!(
            sim.num_layers() >= 16,
            "physical_ir_spine requires 16 layers (14 existing + 2 IR spine routing planes)"
        );

        let gw = sim.width();
        let gh = sim.height();
        let layer_size = gw * gh;
        let mut spine_indices: Vec<usize> = Vec::new();

        // Inline helper: place tile and record index for scope extension.
        macro_rules! place_spine {
            ($x:expr, $y:expr, $z:expr, $tt:expr) => {{
                let (sx, sy, sz) = ($x, $y, $z);
                sim.set_tile_3d(sx, sy, sz, $tt);
                let idx = sz * layer_size + sy * gw + sx;
                spine_indices.push(idx);
                idx
            }};
        }

        // --- ir_low spine (z=14) ---
        // Source: Super Mux lane 0 at L0(ox+83, oy+11).
        // Destination: L3(ox+2, oy+4) junction → extraction ingress.

        // 1. Lift column at (ox+83, oy+11): ViaDown z=1..14.
        //    ViaDown at z=N reads z=N-1, lifting the signal upward.
        for z in 1..=14 {
            place_spine!(ox + 83, oy + 11, z, TileType::ViaDown);
        }

        // 2. Horizontal WireLeft spine on z=14, y=11, from x=ox+82 down to x=ox+2.
        for x in (ox + 2)..=(ox + 82) {
            place_spine!(x, oy + 11, 14, TileType::WireLeft);
        }

        // 3. Vertical WireUp turn on z=14, x=ox+2, from y=10 down to y=4.
        //    The WireLeft at (ox+2, oy+11, z=14) feeds WireUp at (ox+2, oy+10, z=14).
        for y in (oy + 4)..=(oy + 10) {
            place_spine!(ox + 2, y, 14, TileType::WireUp);
        }

        // 4. Portal: ViaUp chain at (ox+2, oy+4), z=4..13.
        //    ViaUp at z=N reads z=N+1, bringing the signal down from z=14.
        //    z=13 reads z=14 (the WireUp tile). z=4 reads z=5. L3(z=3) reads z=4.
        for z in 4..=13 {
            place_spine!(ox + 2, oy + 4, z, TileType::ViaUp);
        }

        // 5. Change L3(ox+2, oy+4) from WireDown to ViaUp.
        //    Now reads z=4 (portal chain) instead of UP (westback pocket).
        sim.set_tile_3d(ox + 2, oy + 4, 3, TileType::ViaUp);
        // L3 junction tile is already in pipeline scope (placed during wiring),
        // so no need to add it to spine_indices.

        // --- ir_high spine (z=15) ---
        // Source: Super Mux lane 1 at L0(ox+86, oy+11).
        // Destination: L3(ox+13, oy+4) junction → extraction ingress.

        // 1. Lift column at (ox+86, oy+11): ViaDown z=1..15.
        for z in 1..=15 {
            place_spine!(ox + 86, oy + 11, z, TileType::ViaDown);
        }

        // 2. Horizontal WireLeft spine on z=15, y=11, from x=ox+85 down to x=ox+13.
        for x in (ox + 13)..=(ox + 85) {
            place_spine!(x, oy + 11, 15, TileType::WireLeft);
        }

        // 3. Vertical WireUp turn on z=15, x=ox+13, from y=10 down to y=4.
        for y in (oy + 4)..=(oy + 10) {
            place_spine!(ox + 13, y, 15, TileType::WireUp);
        }

        // 4. Portal: ViaUp chain at (ox+13, oy+4), z=4..14.
        //    z=14 reads z=15 (the WireUp tile). z=4 reads z=5. L3(z=3) reads z=4.
        for z in 4..=14 {
            place_spine!(ox + 13, oy + 4, z, TileType::ViaUp);
        }

        // 5. Change L3(ox+13, oy+4) from WireRight to ViaUp.
        //    Now reads z=4 (portal chain) instead of LEFT (westback chain).
        sim.set_tile_3d(ox + 13, oy + 4, 3, TileType::ViaUp);

        // --- R-override portal (z=14 spine → L3(ox+1, oy+33)) ---
        // Sprint 257: The L3 south column entry at L3(ox+1, oy+5) was overwritten by
        // Sprint 125 (WireLeft for branch-target route). The south column no longer
        // carries ir_low to the L2 ViaUp bridge at (ox+1, oy+33). Fix: deliver ir_low
        // directly from the z=14 spine to L3(ox+1, oy+33) via a second portal.
        //
        // Path: z=14 WireLeft at (ox+2, oy+11) → WireDown south to (ox+2, oy+33)
        //       → WireLeft to (ox+1, oy+33) → ViaUp portal z=13..4 → L3(ox+1, oy+33)

        // 1. Southward WireDown turn on z=14, x=ox+2, from y=oy+12 to y=oy+33.
        //    Reads from the WireLeft spine tile at (ox+2, oy+11, z=14).
        for y in (oy + 12)..=(oy + 33) {
            place_spine!(ox + 2, y, 14, TileType::WireDown);
        }

        // 2. Lateral WireLeft step on z=14, from (ox+2, oy+33) to (ox+1, oy+33).
        place_spine!(ox + 1, oy + 33, 14, TileType::WireLeft);

        // 3. Portal: ViaUp chain at (ox+1, oy+33), z=4..13.
        //    z=13 reads z=14 (WireLeft). z=4 reads z=5. L3(z=3) reads z=4.
        for z in 4..=13 {
            place_spine!(ox + 1, oy + 33, z, TileType::ViaUp);
        }

        // 4. Change L3(ox+1, oy+33) from WireDown to ViaUp.
        //    Was part of broken L3 south column. Now reads z=4 (R-override portal).
        //    L2 ViaUp bridge at (ox+1, oy+33) reads L3 → gets ir_low from portal.
        sim.set_tile_3d(ox + 1, oy + 33, 3, TileType::ViaUp);

        // Rebuild via connections so the new ViaDown/ViaUp tiles get linked.
        sim.rebuild_via_connections();

        // Extend pipeline + clock scope masks to include all spine tiles.
        // These tiles were placed after BFS scope computation in build_core.
        cpu.extend_scope_masks_for_spine(&spine_indices);

        // Sprint 256: R-override route scope extension.
        // The R-override chain (Sprint 114) delivers ir_low from L2(ox+1,oy+33)
        // ViaUp bridge → L2 east bus → L1 hop → L0 drops → WireDown chains to oy+37.
        // The bridge is in pipeline scope (Sprint 120 push_pipeline), but the
        // downstream route is NOT. Add those tiles to scope so propagation settles
        // the R-override values physically instead of software injection.
        let mut r_override_indices: Vec<usize> = Vec::new();
        // L2 east bus Phase A: ox+2..13 at oy+33
        for x in (ox + 2)..=(ox + 13) {
            r_override_indices.push(2 * layer_size + (oy + 33) * gw + x);
        }
        // L1 ViaUp (hop entry) at ox+13
        r_override_indices.push(layer_size + (oy + 33) * gw + (ox + 13));
        // L1 WireRight hop: ox+14..20
        for x in (ox + 14)..=(ox + 20) {
            r_override_indices.push(layer_size + (oy + 33) * gw + x);
        }
        // L2 ViaDown (hop exit) at ox+20
        r_override_indices.push(2 * layer_size + (oy + 33) * gw + (ox + 20));
        // L2 east bus Phase C: ox+21..46
        for x in (ox + 21)..=(ox + 46) {
            r_override_indices.push(2 * layer_size + (oy + 33) * gw + x);
        }
        // Drops and WireDown chains for all 8 R-override columns
        let alu_r_columns: [usize; 8] = std::array::from_fn(|i| ox + 18 + i * 4);
        for i in 0..8 {
            let col = alu_r_columns[i];
            if i == 0 {
                // L0 ViaUp reads L1 WireRight (part of hop chain)
                r_override_indices.push((oy + 33) * gw + col);
            } else if i == 4 {
                // Offset drop: L1 ViaUp + L0 ViaUp at ox+35, L0 WireLeft at ox+34
                r_override_indices.push(layer_size + (oy + 33) * gw + (col + 1));
                r_override_indices.push((oy + 33) * gw + (col + 1));
                r_override_indices.push((oy + 33) * gw + col);
            } else {
                // Standard: L1 ViaUp + L0 ViaUp
                r_override_indices.push(layer_size + (oy + 33) * gw + col);
                r_override_indices.push((oy + 33) * gw + col);
            }
            // L0 WireDown chains oy+34..36
            if i == 2 {
                // Sprint 237 detour via column ox+25
                r_override_indices.push((oy + 34) * gw + col); // WD(26,34)
                r_override_indices.push((oy + 34) * gw + (col - 1)); // WL(25,34)
                r_override_indices.push((oy + 35) * gw + (col - 1)); // WD(25,35)
                r_override_indices.push((oy + 36) * gw + (col - 1)); // WD(25,36)
                r_override_indices.push((oy + 36) * gw + col); // WR(26,36)
            } else {
                for y in (oy + 34)..=(oy + 36) {
                    r_override_indices.push(y * gw + col);
                }
            }
        }
        cpu.extend_scope_masks_for_spine(&r_override_indices);

        // Sprint 260: Branch-target route scope extension.
        // L2(ox+2, oy+4) ViaUp already reads L3 = spine ir_low (Sprint 151).
        // The east bus L2(ox+3..7, oy+4) + L3 hop (ox+7, oy+1..4) + L2 ViaUp (ox+7, oy+1)
        // need to be in pipeline scope so they settle during Stage F propagation.
        let mut branch_target_indices: Vec<usize> = Vec::new();
        // L2 east bus: ox+2..7 at oy+4 (ViaUp source + WireRight chain)
        for x in (ox + 2)..=(ox + 7) {
            branch_target_indices.push(2 * layer_size + (oy + 4) * gw + x);
        }
        // L3 ViaDown entry + WireUp chain + L2 ViaUp exit
        branch_target_indices.push(3 * layer_size + (oy + 4) * gw + (ox + 7));
        for y in (oy + 1)..=(oy + 3) {
            branch_target_indices.push(3 * layer_size + y * gw + (ox + 7));
        }
        branch_target_indices.push(2 * layer_size + (oy + 1) * gw + (ox + 7));
        // Target Mux at L2(ox+6, oy+1)
        branch_target_indices.push(2 * layer_size + (oy + 1) * gw + (ox + 6));
        cpu.extend_scope_masks_for_spine(&branch_target_indices);

        cpu.set_physical_ir_spine(true);
    }

    // Sprint 258: Computational via decode — replace high-tree selector Const tiles
    // with WeightedViaUp tiles that extract rd[0:2] from ir_high and rs[0:2] from
    // ir_low via z-plane delivery from the existing spines.
    if config.physical_via_decode {
        assert!(
            config.physical_ir_spine,
            "physical_via_decode requires physical_ir_spine"
        );

        let gw = sim.width();
        let gh = sim.height();
        let layer_size = gw * gh;
        let mut via_decode_indices: Vec<usize> = Vec::new();

        // Helper: place tile on z-plane, record for scope extension.
        macro_rules! place_zplane {
            ($x:expr, $y:expr, $z:expr, $tt:expr) => {{
                let (px, py, pz) = ($x, $y, $z);
                sim.set_tile_3d(px, py, pz, $tt);
                let idx = pz * layer_size + py * gw + px;
                via_decode_indices.push(idx);
                idx
            }};
        }

        // === Tree A: ir_high delivery (z=15 spine → L1 → WeightedViaUp selectors) ===
        //
        // z=15 spine at y=oy+11 carries ir_high, runs WireLeft x=13..85.
        // Tap at x=ox+85 (easternmost spine tile), extend east to x=87 via WireRight,
        // then drop south to y=24 via WireDown, ViaUp chain z=14→2, L1 ViaUp.
        // L1 horizontal bus fans out to all 7 selector positions.

        // 1a. z=15 WireRight extension: (ox+86..87, oy+11, z=15) — east of spine end.
        //     The ViaDown lift at (ox+86, oy+11, z=15) has ir_high.
        //     WireRight at ox+86 reads LEFT = ViaDown(ir_high). WireRight at ox+87 reads LEFT.
        //     Note: ox+86 on z=15 is already a ViaDown (lift top). We place WireRight
        //     at ox+87 only; ox+86 ViaDown output already carries ir_high.
        place_zplane!(ox + 87, oy + 11, 15, TileType::WireRight);

        // 1b. z=15 WireDown column: (ox+87, oy+12..24, z=15) — drops south.
        for y in (oy + 12)..=(oy + 24) {
            place_zplane!(ox + 87, y, 15, TileType::WireDown);
        }

        // 2. ViaUp chain at (ox+87, oy+24, z=2..14) — brings ir_high down from z=15.
        for z in 2..=14 {
            place_zplane!(ox + 87, oy + 24, z, TileType::ViaUp);
        }

        // 3. L1 ViaUp at (ox+87, oy+24) — reads z=2, delivers ir_high to L1.
        place_zplane!(ox + 87, oy + 24, 1, TileType::ViaUp);

        // 4. L1 horizontal bus at y=24: WireLeft west, WireRight east from drop point.
        for x in (ox + 81)..=(ox + 86) {
            place_zplane!(x, oy + 24, 1, TileType::WireLeft);
        }
        for x in (ox + 88)..=(ox + 93) {
            place_zplane!(x, oy + 24, 1, TileType::WireRight);
        }

        // 5. L1 WireDown columns for sel1 (y=27) and sel2 (y=30).
        //    sel1_left at (ox+82, oy+27): WireDown y=25..27
        for y in (oy + 25)..=(oy + 27) {
            place_zplane!(ox + 82, y, 1, TileType::WireDown);
        }
        //    sel1_right at (ox+90, oy+27): WireDown y=25..27
        for y in (oy + 25)..=(oy + 27) {
            place_zplane!(ox + 90, y, 1, TileType::WireDown);
        }
        //    sel2 at (ox+87, oy+30): WireDown y=25..30
        for y in (oy + 25)..=(oy + 30) {
            place_zplane!(ox + 87, y, 1, TileType::WireDown);
        }

        // 6. Swap Tree A selector Consts → WeightedViaUp.
        //    sel0 (4 tiles at y=24): shift=0, mask=1 → rd[0]
        for &x in &[ox + 81, ox + 85, ox + 89, ox + 93] {
            sim.set_tile(x, oy + 24, TileType::WeightedViaUp);
            let idx = (oy + 24) * gw + x;
            sim.set_tile_shift(idx, 0);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }
        //    sel1 (2 tiles at y=27): shift=1, mask=1 → rd[1]
        for &x in &[ox + 82, ox + 90] {
            sim.set_tile(x, oy + 27, TileType::WeightedViaUp);
            let idx = (oy + 27) * gw + x;
            sim.set_tile_shift(idx, 1);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }
        //    sel2 (1 tile at y=30): shift=2, mask=1 → rd[2]
        {
            let x = ox + 87;
            sim.set_tile(x, oy + 30, TileType::WeightedViaUp);
            let idx = (oy + 30) * gw + x;
            sim.set_tile_shift(idx, 2);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }

        // === Tree B: ir_low delivery (z=14 spine → east extension → L1 → selectors) ===
        //
        // z=14 spine carries ir_low. The ViaDown lift at (ox+83, oy+11, z=14) is the
        // source. Extend eastward to x=109 on z=14, then drop south at x=103 (Tree B
        // center) to y=24, ViaUp chain to L1, horizontal fan-out.

        // 1. z=14 eastward extension at y=12 (NOT y=11 — avoids ir_high lift at x=86).
        //    Drop ir_low from z=14 spine at (ox+82, oy+11) to y=12, then run east.
        //    WireDown at (ox+82, oy+12, z=14) reads UP = WireLeft spine = ir_low.
        place_zplane!(ox + 82, oy + 12, 14, TileType::WireDown);
        // WireRight east on z=14 at y=12: x=83..109. No ir_high lift collision.
        for x in (ox + 83)..=(ox + 109) {
            place_zplane!(x, oy + 12, 14, TileType::WireRight);
        }

        // 2. z=14 WireDown column at x=103: (ox+103, oy+13..24, z=14).
        //    Starts at y=13 (y=12 has WireRight from eastward extension).
        for y in (oy + 13)..=(oy + 24) {
            place_zplane!(ox + 103, y, 14, TileType::WireDown);
        }

        // 3. ViaUp chain at (ox+103, oy+24, z=2..13) — brings ir_low down from z=14.
        for z in 2..=13 {
            place_zplane!(ox + 103, oy + 24, z, TileType::ViaUp);
        }

        // 4. L1 ViaUp at (ox+103, oy+24) — reads z=2, delivers ir_low to L1.
        place_zplane!(ox + 103, oy + 24, 1, TileType::ViaUp);

        // 5. L1 horizontal bus at y=24: WireLeft west, WireRight east from drop point.
        for x in (ox + 97)..=(ox + 102) {
            place_zplane!(x, oy + 24, 1, TileType::WireLeft);
        }
        for x in (ox + 104)..=(ox + 109) {
            place_zplane!(x, oy + 24, 1, TileType::WireRight);
        }

        // 6. L1 WireDown columns for sel1 (y=27) and sel2 (y=30).
        for y in (oy + 25)..=(oy + 27) {
            place_zplane!(ox + 98, y, 1, TileType::WireDown);
        }
        for y in (oy + 25)..=(oy + 27) {
            place_zplane!(ox + 106, y, 1, TileType::WireDown);
        }
        for y in (oy + 25)..=(oy + 30) {
            place_zplane!(ox + 103, y, 1, TileType::WireDown);
        }

        // 7. Swap Tree B selector Consts → WeightedViaUp.
        //    sel0 (4 tiles at y=24): shift=5, mask=1 → rs[0]
        for &x in &[ox + 97, ox + 101, ox + 105, ox + 109] {
            sim.set_tile(x, oy + 24, TileType::WeightedViaUp);
            let idx = (oy + 24) * gw + x;
            sim.set_tile_shift(idx, 5);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }
        //    sel1 (2 tiles at y=27): shift=6, mask=1 → rs[1]
        for &x in &[ox + 98, ox + 106] {
            sim.set_tile(x, oy + 27, TileType::WeightedViaUp);
            let idx = (oy + 27) * gw + x;
            sim.set_tile_shift(idx, 6);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }
        //    sel2 (1 tile at y=30): shift=7, mask=1 → rs[2] raw (non-inverted).
        //    The inversion is handled by a Zero tile downstream.
        {
            let x = ox + 103;
            sim.set_tile(x, oy + 30, TileType::WeightedViaUp);
            let idx = (oy + 30) * gw + x;
            sim.set_tile_shift(idx, 7);
            sim.set_tile_mask(idx, 1);
            via_decode_indices.push(idx);
        }

        // 8. Tree B sel2 inversion circuit: rs[2] must be inverted for bank-swap.
        //    WeightedViaUp(shift=7,mask=1) at (ox+103,oy+30) outputs raw rs[2].
        //    Zero at (ox+104,oy+30) reads LEFT → inverts.
        //    WireDown at (ox+104,oy+31) carries inverted value south.
        //    WireLeft at (ox+103,oy+31) reads RIGHT → feeds root Mux UP input.
        let zero_idx = (oy + 30) * gw + (ox + 104);
        sim.set_tile(ox + 104, oy + 30, TileType::Zero);
        via_decode_indices.push(zero_idx);
        let inv_wd_idx = (oy + 31) * gw + (ox + 104);
        sim.set_tile(ox + 104, oy + 31, TileType::WireDown);
        via_decode_indices.push(inv_wd_idx);
        // Change existing WireDown at (ox+103, oy+31) to WireLeft.
        let inv_wl_idx = (oy + 31) * gw + (ox + 103);
        sim.set_tile(ox + 103, oy + 31, TileType::WireLeft);

        // Rebuild via connections for new ViaUp tiles.
        sim.rebuild_via_connections();

        // Extend scope masks to include all via decode infrastructure.
        cpu.extend_scope_masks_for_spine(&via_decode_indices);

        // Store inversion circuit indices for per-cycle dirty seeding.
        let inversion_dirty = vec![zero_idx, inv_wd_idx, inv_wl_idx];
        cpu.set_physical_via_decode(true, inversion_dirty);
    }

    // Sprint 259: Physical byte2 selector — no tile changes needed.
    // The execute code reads rom_selected_byte2_idx (physical Super Mux output)
    // and extracts rd_hi/rs_hi bits from the physical tile value.
    if config.physical_byte2_selector {
        cpu.set_physical_byte2_selector(true);
    }

    if config.enable_operand_bypass {
        cpu.enable_synth_operand_bypass();
    }

    // Sprint 209: combined decode — precompute [u16; 32] LUT from AIG evaluation.
    // This is a software-cached block (no physical placement) because the combined
    // AIG (~125 gates) exceeds current router capacity for physical synthesis.
    if config.enable_combined_decode {
        let combined_lut = build_combined_decode_lut();
        cpu.set_combined_decode_lut(combined_lut);
    }

    // Sprint 218: ALU readback dual-path verification (observational only).
    if config.enable_alu_readback {
        cpu.enable_alu_readback();
    }

    // Sprint 219: Physical ALU authority.
    if config.physical_alu_opcodes != 0 {
        // Physical ALU requires readback to be enabled (for diagnostics).
        cpu.enable_alu_readback();
        cpu.set_physical_alu_opcodes(config.physical_alu_opcodes);
    }

    // Sprint 220: Physical register writeback authority.
    if config.physical_reg_writeback {
        assert!(
            config.physical_alu_opcodes != 0,
            "physical_reg_writeback requires physical_alu_opcodes != 0"
        );
        cpu.restore_reg_clock_cache_for_physical_writeback();
    }

    // Sprint 221: Physical flag writeback authority.
    if config.physical_flag_writeback {
        assert!(
            config.physical_alu_opcodes != 0,
            "physical_flag_writeback requires physical_alu_opcodes != 0"
        );
        cpu.restore_flag_clock_cache_for_physical_writeback();
    }

    // Sprint 222: Physical ctrl_a/ctrl_b decode authority for bank0.
    if config.physical_decode {
        assert!(
            config.physical_alu_opcodes != 0,
            "physical_decode requires physical_alu_opcodes != 0"
        );
        cpu.set_physical_decode(true);
    }

    // Sprint 225: Physical branch direction authority.
    if config.physical_branch {
        assert!(
            config.enable_branch,
            "physical_branch requires enable_branch"
        );
        assert!(
            config.physical_decode,
            "physical_branch requires physical_decode (upper-bank selector inputs depend on Const-swap injection)"
        );
        cpu.set_physical_branch(true);
    }

    // Sprint 229: Physical RAM writeback authority.
    if config.physical_ram_writeback {
        cpu.restore_ram_clock_cache_for_physical_writeback();
    }

    // Sprint 231: Physical operand authority.
    if config.physical_operand_authority {
        assert!(
            config.enable_operand_bypass,
            "physical_operand_authority requires enable_operand_bypass"
        );
        cpu.set_physical_operand_authority(true);
    }

    // Sprint 233: Physical pre-commit decode delivery authority.
    if config.physical_rd_decode {
        assert!(
            config.enable_rd_decode,
            "physical_rd_decode requires enable_rd_decode"
        );
    }
    if config.physical_we_mask {
        assert!(
            config.enable_ctrl_a,
            "physical_we_mask requires enable_ctrl_a"
        );
    }
    if config.physical_flag_we_mask {
        assert!(
            config.enable_ctrl_a,
            "physical_flag_we_mask requires enable_ctrl_a"
        );
    }

    // Sprint 234: Physical Super Mux propagation.
    if config.physical_super_mux {
        cpu.set_physical_super_mux(true);
    }

    // Sprint 235: Physical writeback-data authority for non-ALU register-producing ops.
    if config.physical_wb_data_authority {
        assert!(
            config.physical_reg_writeback,
            "physical_wb_data_authority requires physical_reg_writeback"
        );
        cpu.set_physical_wb_data_authority(true);
    }

    // Sprint 236: Physical high-register writeback authority.
    if config.physical_high_reg_writeback {
        assert!(
            config.physical_wb_data_authority,
            "physical_high_reg_writeback requires physical_wb_data_authority"
        );
        cpu.restore_high_reg_clock_cache_for_physical_writeback();
    }

    // Sprint 245: Sub-function ALU delivery authority.
    if config.physical_sub_fn_delivery {
        assert!(
            config.physical_reg_writeback,
            "physical_sub_fn_delivery requires physical_reg_writeback"
        );
        cpu.set_physical_sub_fn_delivery(true);
    }

    // Sprint 246: Sub-function flag authority.
    if config.physical_sub_fn_flag_authority {
        assert!(
            config.physical_sub_fn_delivery,
            "physical_sub_fn_flag_authority requires physical_sub_fn_delivery"
        );
        assert!(
            config.physical_flag_writeback,
            "physical_sub_fn_flag_authority requires physical_flag_writeback"
        );
        cpu.set_physical_sub_fn_flag_authority(true);
    }

    // Sprint 246: LD/LDB delivery + flag authority.
    if config.physical_load_delivery {
        assert!(
            config.physical_reg_writeback,
            "physical_load_delivery requires physical_reg_writeback"
        );
        cpu.set_physical_load_delivery(true);
    }

    // Sprint 247: RAM store authority.
    if config.physical_ram_store_authority {
        assert!(
            config.physical_ram_writeback,
            "physical_ram_store_authority requires physical_ram_writeback"
        );
        cpu.set_physical_ram_store_authority(true);
    }

    // Sprint 248: SRA computation authority.
    if config.physical_sra_computation {
        assert!(
            config.physical_alu_opcodes & (1 << 0x0E) != 0,
            "physical_sra_computation requires SHR (0x0E) in physical_alu_opcodes"
        );
        cpu.set_physical_sra_computation(true);
    }

    // Sprint 232: Rebuild no-flags clock cache after all clock cache modifications.
    if config.physical_flag_writeback {
        cpu.rebuild_no_flags_clock_cache();
    }

    // Sprint 324: Enable clock auto-warmup for max_authority configs.
    // Only safe when all authority flags are set (stable clock behavior).
    if config.physical_wb_data_authority
        && config.physical_flag_writeback
        && config.physical_via_decode
    {
        cpu.clock_auto_warmup_enabled.set(true);
        // Sprint 336: Enable settle JIT for max_authority configs.
        cpu.settle_jit_enabled.set(true);
        // Sprint 352: Backbone JIT infrastructure built but opt-in until
        // A/B profiled against full-settle JIT baseline (S336).
        // cpu.backbone_jit_enabled.set(true);
    }
}

/// Sprint 209: Precompute combined ctrl_a+ctrl_b lookup table from AIG evaluation.
/// Returns [u16; 32] indexed by opcode: bits [7:0] = ctrl_a, bits [15:8] = ctrl_b.
fn build_combined_decode_lut() -> [u16; 32] {
    use crate::synth::mapping::evaluate_aig;
    let aig = build_combined_decode_aig();
    let mut lut = [0u16; 32];
    for opcode in 0..32u32 {
        let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
        let outputs = evaluate_aig(&aig, &inputs);
        let mut val = 0u16;
        for (i, &bit) in outputs.iter().enumerate() {
            if bit {
                val |= 1 << i;
            }
        }
        lut[opcode as usize] = val;
    }
    lut
}

/// Build a synth block and inject it into the simulation at the given offset.
fn build_and_inject_block(
    sim: &mut Simulation,
    aig: &crate::synth::aig::Aig,
    offset_x: usize,
    offset_y: usize,
    route_config: &RouteConfig,
    halo: usize,
) -> crate::synth::integration::InjectedBlock {
    build_and_inject_block_z(sim, aig, offset_x, offset_y, 0, route_config, halo)
}

/// Sprint 250: Build and inject a synth block at a z-plane offset.
fn build_and_inject_block_z(
    sim: &mut Simulation,
    aig: &crate::synth::aig::Aig,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
    route_config: &RouteConfig,
    halo: usize,
) -> crate::synth::integration::InjectedBlock {
    let lib = CellLibrary::tile_native();
    let bridge_cfg = BridgeConfig {
        halo,
        max_z: route_config.max_z,
        no_crossings: route_config.no_crossings,
        optimize: false,
    };
    let (block, _export) = crate::synth::bridge::compile_and_inject_with(
        aig,
        &lib,
        sim,
        offset_x,
        offset_y,
        offset_z,
        &bridge_cfg,
    )
    .expect("synth pipeline failed");
    block
}

/// Sprint 180: Topology for N-CPU arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2ArrayTopology {
    /// CPU[0]—CPU[1]—...—CPU[N-1]. N-1 links.
    Linear,
    /// CPU[0]—CPU[1]—...—CPU[N-1]—CPU[0]. N links (last wraps to first).
    Ring,
}

/// Sprint 180: Build an N-CPU array with bidirectional link mailboxes.
///
/// Each adjacent pair of CPUs shares a bidirectional link with two `Rc<Cell<u64>>`
/// cells (forward + backward). Each CPU's MMIO maps:
/// - Address 60 (`MMIO_MAILBOX_IN`): read = left neighbor sent, write = send to left
/// - Address 61 (`MMIO_MAILBOX_OUT`): read = right neighbor sent, write = send to right
///
/// CPUs are vertically stacked at origin (0, i*64) in a 128×(N*64)×4 grid.
pub fn build_v2_array(
    programs: &[&[u32]],
    topology: V2ArrayTopology,
) -> (Simulation, Vec<TileCpuV2>) {
    let n = programs.len();
    assert!(n >= 2, "build_v2_array requires at least 2 CPUs");

    let mut sim = Simulation::with_size_layered(128, n * 64, 4);

    // Create links: each link has (forward, backward) cells.
    let n_links = match topology {
        V2ArrayTopology::Linear => n - 1,
        V2ArrayTopology::Ring => n,
    };
    let links: Vec<(Rc<Cell<u64>>, Rc<Cell<u64>>)> = (0..n_links)
        .map(|_| (Rc::new(Cell::new(0u64)), Rc::new(Cell::new(0u64))))
        .collect();

    let mut cpus = Vec::with_capacity(n);
    let mut masks = Vec::with_capacity(n);

    for i in 0..n {
        // Left link: links[i-1] connects CPU[i-1] (lower) ↔ CPU[i] (higher).
        // left_recv = forward cell (what lower neighbor sent up)
        // left_send = backward cell (send down to lower neighbor)
        let (left_recv, left_send) = if i > 0 {
            (Some(links[i - 1].0.clone()), Some(links[i - 1].1.clone()))
        } else if topology == V2ArrayTopology::Ring {
            (Some(links[n - 1].0.clone()), Some(links[n - 1].1.clone()))
        } else {
            (None, None)
        };

        // Right link: links[i] connects CPU[i] (lower) ↔ CPU[i+1] (higher).
        // right_send = forward cell (send up to higher neighbor)
        // right_recv = backward cell (what higher neighbor sent down)
        let (right_send, right_recv) = if i < n - 1 {
            (Some(links[i].0.clone()), Some(links[i].1.clone()))
        } else if topology == V2ArrayTopology::Ring {
            (
                Some(links[i % n_links].0.clone()),
                Some(links[i % n_links].1.clone()),
            )
        } else {
            (None, None)
        };

        let device = V2LinkMailboxDevice::new(left_recv, left_send, right_send, right_recv);
        let handle = V2MmioHandle::new(device);

        let (cpu, mask) = V2Builder::new()
            .with_origin(0, i * 64)
            .with_program(programs[i])
            .with_rom_size(128)
            .with_ram_size(64)
            .with_mmio(handle)
            .build_no_chains(&mut sim);
        cpus.push(cpu);
        masks.push(mask);
    }

    // Finalize wire chains with merged placed masks.
    let mut cpu_refs: Vec<&mut TileCpuV2> = cpus.iter_mut().collect();
    finalize_multi_cpu_chains(&mut sim, &mut cpu_refs, &masks);

    (sim, cpus)
}

/// Sprint 203: Build an N-CPU array with synth blocks per CPU.
///
/// Uses **horizontal stacking** (each CPU in a 128-wide column) instead of
/// `build_v2_array()`'s vertical stacking. This gives each CPU its own synth
/// zone (y=70+) without layout conflicts.
///
/// Grid: `(n * 128) × 384 × 4`. CPU[i] at origin `(i * 128, 0)`.
/// Mailbox links use software `Rc<Cell<u64>>` — unaffected by layout.
pub fn build_v2_array_with_synth(
    programs: &[&[u32]],
    topology: V2ArrayTopology,
    synth: V2SynthConfig,
) -> (Simulation, Vec<TileCpuV2>) {
    let n = programs.len();
    assert!(n >= 2, "build_v2_array_with_synth requires at least 2 CPUs");

    let mut sim = Simulation::with_size_layered(n * 128, 384, 4);

    // Create links: each link has (forward, backward) cells.
    let n_links = match topology {
        V2ArrayTopology::Linear => n - 1,
        V2ArrayTopology::Ring => n,
    };
    let links: Vec<(Rc<Cell<u64>>, Rc<Cell<u64>>)> = (0..n_links)
        .map(|_| (Rc::new(Cell::new(0u64)), Rc::new(Cell::new(0u64))))
        .collect();

    let mut cpus = Vec::with_capacity(n);
    let mut masks = Vec::with_capacity(n);

    for i in 0..n {
        let (left_recv, left_send) = if i > 0 {
            (Some(links[i - 1].0.clone()), Some(links[i - 1].1.clone()))
        } else if topology == V2ArrayTopology::Ring {
            (Some(links[n - 1].0.clone()), Some(links[n - 1].1.clone()))
        } else {
            (None, None)
        };

        let (right_send, right_recv) = if i < n - 1 {
            (Some(links[i].0.clone()), Some(links[i].1.clone()))
        } else if topology == V2ArrayTopology::Ring {
            (
                Some(links[i % n_links].0.clone()),
                Some(links[i % n_links].1.clone()),
            )
        } else {
            (None, None)
        };

        let device = V2LinkMailboxDevice::new(left_recv, left_send, right_send, right_recv);
        let handle = V2MmioHandle::new(device);

        let (cpu, mask) = V2Builder::new()
            .with_origin(i * 128, 0)
            .with_program(programs[i])
            .with_rom_size(128)
            .with_ram_size(64)
            .with_mmio(handle)
            .with_synth_blocks(synth.clone())
            .build_no_chains(&mut sim);
        cpus.push(cpu);
        masks.push(mask);
    }

    // Finalize wire chains with merged placed masks.
    let mut cpu_refs: Vec<&mut TileCpuV2> = cpus.iter_mut().collect();
    finalize_multi_cpu_chains(&mut sim, &mut cpu_refs, &masks);

    (sim, cpus)
}
