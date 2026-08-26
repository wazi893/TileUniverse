//! Step 1a (V2 → CUDA device-shaped model): device-cycle **schema + static layout catalog**.
//!
//! Per the locked spec (`V2_CUDA_MEGAKERNEL_SPEC.md`) the megakernel cycle is modeled as
//! **inject sets → dense phase op arrays → readback taps**, with per-lane **hold-masks**
//! replacing the commit Const-swaps, plus a per-lane **RAM SoA** (both gate-0). This
//! module defines that schema + the static catalog the per-cycle driver (1b) and the CUDA
//! megakernel (1d) consume. **No edits to `v2_execute.rs`.**
//!
//! ### Phases (six — review fix P1)
//! `run_stage_x` runs a **trunk settle** (`trunk_settle_ops`, v2_execute ~3466) BEFORE the
//! branch/commit/clock arrays and before reading the ALU root — it is a distinct sixth
//! propagation, not one of the five. Without it, upper-reg ALU instructions (and the
//! charter set leans on high regs via P2 regalloc) would diverge. `DEVICE_PHASES` lists
//! all six in tick execution order: trunk_settle, branch, commit, clock (Stage X), then
//! cone, settle (Stage F). (When `trunk_settle_ops` is empty the real path uses the
//! constswap-settle fallback; 1b must detect that.)
//!
//! ### Catalog is LAYOUT-ONLY (review fix P1)
//! `derive_indices` calls raw `wire_v2_cpu`, so only the **tile-index (layout)** fields of
//! `V2CpuIndices` are populated — the post-build scope/eval-order/cache fields are EMPTY by
//! construction (those are computed later in `build_core`). 1b must take phase op arrays,
//! scopes, and eval orders from the **live cpu**, never from this catalog. Exposed via
//! `layout_indices()` and documented as such.
//!
//! ### Hold-mask classifier is ANALYSIS-ONLY (review fix P1)
//! [`CommitSwapIncidence`] classifies from the instruction WORD only. Real Const-swaps also
//! depend on runtime write-intent (`reg_write_value.is_some()`, physical-authority flags,
//! `is_sub_fn`, high-reg writeback branches in `run_stage_x` ~4168). It is an incidence
//! scanner; **1b must generate hold-masks from runtime write-intent, not from this.**
//!
//! The oracle is **architectural** (`hash_v2_final_state`: pc, lr, flags, halted, regs[16],
//! ram[128]) — so the device model needs architectural parity, not full-grid fidelity,
//! letting the device cycle drop the CPU's verification dual-paths and deferred restores.

use crate::simulation::Simulation;
use crate::tile_cpu::v2_components::EXT_REG_INDIRECT;
use crate::tile_cpu::v2_disassembler::V2DecodedInstruction;
use crate::tile_cpu::v2_full_phase_prover::UnifiedSlotUniverse;
use crate::tile_cpu::v2_simt_eval::{LaneKernel, LanePack};
use crate::tile_cpu::v2_wiring::V2CpuIndices;
use crate::tile_cpu::{TileCpuV2, decode_v2_word};

/// The six tile-eval propagations in `TileCpuV2::tick` execution order
/// (Stage X: trunk_settle, branch, commit, clock; Stage F: cone, settle).
pub const DEVICE_PHASES: [&str; 6] = [
    "trunk_settle",
    "branch",
    "commit",
    "clock",
    "cone",
    "settle",
];

/// Architectural state the megakernel carries per lane. Covers exactly the fields
/// `hash_v2_final_state` hashes, plus `cycle`/`retired` for the cycle-accurate 1b oracle
/// (review fix P2 — the hash omits them, but a single-cycle oracle should not).
/// (The pipeline latch needed to *drive* a cycle is added in 1b.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCycleState {
    pub pc: u32,
    pub lr: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub cycle: u64,
    pub retired: u64,
    pub regs: [u64; 16],
    /// Per-lane RAM SoA — **gate-0**. 0..128 is the scratchpad (charter config); main
    /// memory configs append beyond it.
    pub ram: Vec<u64>,
}

impl DeviceCycleState {
    /// Read the architectural state out of a live tile CPU + sim.
    pub fn capture(cpu: &TileCpuV2, sim: &Simulation, ram_words: usize) -> Self {
        Self {
            pc: cpu.read_pc(sim),
            lr: cpu.read_lr(),
            flag_z: cpu.read_flag_z(sim),
            flag_c: cpu.read_flag_c(sim),
            halted: cpu.is_halted(),
            cycle: cpu.read_cycle_count(),
            retired: cpu.read_retired_count(),
            regs: std::array::from_fn(|i| cpu.read_reg(sim, i)),
            ram: (0..ram_words).map(|a| cpu.read_ram(sim, a)).collect(),
        }
    }

    /// Field-level diff for the 1b single-cycle oracle (empty == match). Checks RAM length
    /// FIRST so a length mismatch can never masquerade as a match (review fix P2).
    pub fn diff(&self, other: &Self) -> Vec<String> {
        let mut d = Vec::new();
        if self.pc != other.pc {
            d.push(format!("pc {} != {}", self.pc, other.pc));
        }
        if self.lr != other.lr {
            d.push(format!("lr {} != {}", self.lr, other.lr));
        }
        if self.flag_z != other.flag_z {
            d.push(format!("flag_z {} != {}", self.flag_z, other.flag_z));
        }
        if self.flag_c != other.flag_c {
            d.push(format!("flag_c {} != {}", self.flag_c, other.flag_c));
        }
        if self.halted != other.halted {
            d.push(format!("halted {} != {}", self.halted, other.halted));
        }
        if self.cycle != other.cycle {
            d.push(format!("cycle {} != {}", self.cycle, other.cycle));
        }
        if self.retired != other.retired {
            d.push(format!("retired {} != {}", self.retired, other.retired));
        }
        for i in 0..16 {
            if self.regs[i] != other.regs[i] {
                d.push(format!("R{i} {} != {}", self.regs[i], other.regs[i]));
            }
        }
        if self.ram.len() != other.ram.len() {
            d.push(format!("ram len {} != {}", self.ram.len(), other.ram.len()));
        } else {
            for (a, (x, y)) in self.ram.iter().zip(other.ram.iter()).enumerate() {
                if x != y {
                    d.push(format!("RAM[{a}] {x} != {y}"));
                }
            }
        }
        d
    }
}

/// Tiles the device reads back into architectural state at end of cycle. (pc/lr/ram are
/// software-sequenced; regs/flags are physically committed and read from these taps.)
#[derive(Clone, Debug)]
pub struct ReadbackTaps {
    pub regs: [usize; 16],
    pub flag_z: usize,
    pub flag_c: usize,
    pub pc: usize,
    pub ram: [usize; 128],
}

/// **Analysis-only** classifier for the commit-phase Const-swap (hold-mask) incidence of a
/// decoded instruction WORD. Matches the spec's hold-mask set for scanning/reporting.
///
/// NOT a hold-mask generator: real Const-swaps also depend on runtime write-intent
/// (`reg_write_value.is_some()`, physical-authority flags, `is_sub_fn`, high-reg writeback).
/// 1b must compute hold-masks from runtime state, not from this (review fix P1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommitSwapIncidence {
    /// rd_hi → high-reg merge mux + WE-suppress Const-swap.
    pub high_reg_merge: bool,
    /// LDI.W / sub-fn (MUL, MFLR) / load → `wb_data_mux` Const-swap.
    pub wb_data_mux: bool,
    /// store → `ram_write_gate` (+ ViaUp for addr<8) Const-swap.
    pub ram_write_gate: bool,
    /// MUL → `flag_we` Z-only Const-swap.
    pub flag_we_mul: bool,
}

impl CommitSwapIncidence {
    /// Classify one instruction word (decode-only; word-shaped incidence, not runtime).
    pub fn classify(word: u32) -> Self {
        let d = decode_v2_word(word);
        let load = matches!(d.opcode, 0x16 | 0x18);
        let store = matches!(d.opcode, 0x17 | 0x19);
        let mul = d.opcode == 0x04 && d.imm5 == 1;
        let mflr = d.opcode == 0x02 && d.imm5 == 5;
        let ldi_w = d.opcode == 0x03 && d.wide_imm;
        Self {
            high_reg_merge: d.rd_hi,
            wb_data_mux: load || mul || mflr || ldi_w,
            ram_write_gate: store,
            flag_we_mul: mul,
        }
    }

    pub fn any(&self) -> bool {
        self.high_reg_merge || self.wb_data_mux || self.ram_write_gate || self.flag_we_mul
    }
}

/// The device-cycle plan: static inputs for the per-cycle driver (1b) and the CUDA
/// megakernel (1d) — the unified slot universe, the **six** dense phase op-array kernels
/// (tick order, incl. trunk-settle), the readback taps, and the LAYOUT-only index catalog.
pub struct DeviceCyclePlan {
    pub universe: UnifiedSlotUniverse,
    pub phases: Vec<(&'static str, LaneKernel)>,
    pub taps: ReadbackTaps,
    /// LAYOUT-ONLY catalog: only tile-index fields are valid; scope/eval/cache fields are
    /// EMPTY (derive_indices uses raw `wire_v2_cpu`). 1b reads scopes/eval-orders/op-arrays
    /// from the live cpu, never from here.
    layout: V2CpuIndices,
    /// Trunk-settle kernel with the trunk-terminal-writing ops removed, so injected
    /// operands into `alu_a/b_trunk_terminal_idx` stick under dense eval (1b.1b hold-mask).
    trunk_held: LaneKernel,
}

impl DeviceCyclePlan {
    /// Build the plan from a charter-config CPU + its recovered LAYOUT indices.
    /// (Caller derives indices via `V2Builder::derive_indices` before `build()` consumes
    /// the builder — see `build_charter_plan` in tests.)
    #[allow(dead_code)] // constructed by the 1b driver (task #5); used in tests now.
    pub(crate) fn new(
        cpu: &TileCpuV2,
        sim: &Simulation,
        layout: V2CpuIndices,
    ) -> Result<Self, String> {
        // Six phases in tick execution order. trunk_settle (Stage X) precedes branch.
        // (&Vec coerces to &[..] at the LaneKernel call below.)
        let arrays = [
            (
                "trunk_settle",
                &cpu.trunk_settle_ops,
                &cpu.trunk_settle_wvia,
            ),
            ("branch", &cpu.branch_compact_ops, &cpu.branch_compact_wvia),
            ("commit", &cpu.commit_compact_ops, &cpu.commit_compact_wvia),
            ("clock", &cpu.clock_compact_ops, &cpu.clock_compact_wvia),
            ("cone", &cpu.pipeline_cone_ops, &cpu.pipeline_cone_wvia),
            ("settle", &cpu.settle_compact_ops, &cpu.settle_compact_wvia),
        ];
        let mut phases = Vec::with_capacity(6);
        for (name, ops, wvia) in arrays {
            let kernel = LaneKernel::new_from_sim(ops, wvia, sim)
                .map_err(|e| format!("phase '{name}' kernel build failed: {e}"))?;
            phases.push((name, kernel));
        }
        let universe = UnifiedSlotUniverse::from_cpu(cpu, sim);
        let taps = ReadbackTaps {
            regs: layout.reg_indices,
            flag_z: layout.flag_z_idx,
            flag_c: layout.flag_c_idx,
            pc: layout.pc_idx,
            ram: layout.ram_indices,
        };
        // 1b.1b hold-mask: build a trunk-settle kernel that does NOT recompute the trunk
        // terminals (so injected operands stick under dense eval — the gap the 1b.1a oracle
        // found). Drop the terminal-writing ops + their wvia entries, filtered by tile idx
        // in lockstep so WVIA alignment is preserved.
        let term_a = layout.alu_a_trunk_terminal_idx as u32;
        let term_b = layout.alu_b_trunk_terminal_idx as u32;
        let held_ops: Vec<crate::simulation::CompactOp> = cpu
            .trunk_settle_ops
            .iter()
            .filter(|op| op.idx != term_a && op.idx != term_b)
            .cloned()
            .collect();
        let held_wvia: Vec<(usize, u8, u64)> = cpu
            .trunk_settle_wvia
            .iter()
            .filter(|w| w.0 as u32 != term_a && w.0 as u32 != term_b)
            .cloned()
            .collect();
        let trunk_held = LaneKernel::new_from_sim(&held_ops, &held_wvia, sim)
            .map_err(|e| format!("trunk_held kernel build failed: {e}"))?;
        Ok(Self {
            universe,
            phases,
            taps,
            layout,
            trunk_held,
        })
    }

    /// LAYOUT-ONLY index catalog (tile-index fields valid; scope/eval/cache EMPTY).
    #[allow(dead_code)] // consumed by the 1b injection-site resolver (task #5) + v2_device_gpu (1d).
    pub(crate) fn layout_indices(&self) -> &V2CpuIndices {
        &self.layout
    }

    /// 1d (CUDA megakernel): the three op-array kernels the device ALU runs, in tick
    /// order — `cone` (fetch+decode from ROM+PC), `settle` (decode → ALU op-select),
    /// `trunk_held` (ALU compute, with the trunk-terminal-writing ops removed so injected
    /// operands stick). `v2_device_gpu` unions these into the shared GPU slot universe.
    /// Pure layout/kernels — no cuda dependency, harmless in any build.
    #[allow(dead_code)] // consumed by v2_device_gpu (cuda-gated).
    pub(crate) fn alu_kernels(&self) -> (&LaneKernel, &LaneKernel, &LaneKernel) {
        (self.phase("cone"), self.phase("settle"), &self.trunk_held)
    }

    /// Whether the trunk-settle op array is populated (else the real path uses the
    /// constswap-settle fallback, which 1b must model separately).
    pub fn has_trunk_settle_ops(&self) -> bool {
        self.phases
            .iter()
            .find(|(n, _)| *n == "trunk_settle")
            .map(|(_, k)| k.op_count() > 0)
            .unwrap_or(false)
    }
}

// ===================== 1b.0: scalar control front-end =====================
//
// The megakernel's scalar shell must sequence PC, lr, and halt and emit the
// per-cycle RUNTIME hold-mask set, while the tiles compute the datapath. This
// section builds + validates that CONTROL path; the tile datapath injection +
// readback is 1b.1. Every quantity here is validated bit-exact against `V2Iss`
// (the proven architectural reference) per cycle on the charter set — no
// mid-cycle tile hooks needed, so no blind debugging.
//
// Mirrors `run_stage_x`'s extended-PC block (v2_execute ~4967) and
// `compute_branch_pc` (~5591) for the charter config (extended_pc, 8-bit PC).

/// Per-instruction commit hold-mask set, generated from RUNTIME write-intent
/// (review fix P1): `high_reg_merge` requires the instruction to actually write a
/// HIGH register, not just any rd_hi encoding. Unlike [`CommitSwapIncidence`]
/// (word-only incidence) this folds in `writes_rd`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HoldMaskSet {
    pub high_reg_merge: bool,
    pub wb_data_mux: bool,
    pub ram_write_gate: bool,
    pub flag_we_mul: bool,
}

impl HoldMaskSet {
    pub fn any(&self) -> bool {
        self.high_reg_merge || self.wb_data_mux || self.ram_write_gate || self.flag_we_mul
    }
}

/// Does this opcode write its `rd` register (architectural write-intent)?
/// NOTE: conditional moves (MOVZ/MOVNZ/MOVC/MOVNC, opcode 0x02 sub-fn 1..4) are
/// treated as writers here; their genuinely runtime-conditional write is a
/// non-charter nuance 1b.1 must refine (the charter set uses branches, not cmov).
pub fn writes_rd(opcode: u8) -> bool {
    matches!(opcode,
        0x02 | 0x03            // MOV/MOVx/LDI (and MFLR via sub-fn)
        | 0x04..=0x0E          // ADD..SHR (incl. MUL); CMP (0x0F) excluded
        | 0x10..=0x14          // ADDI..XORI
        | 0x16 | 0x18          // LD / LDB
    )
}

/// Opcode → branch_kind (== `ctrl_b & 0x07`): JMP=1, JZ=2, JNZ=3, JC=4, JNC=5,
/// CALL=6, RET=7, else 0. JMPR/CALLR share JMP/CALL kinds + the reg-indirect bit.
fn branch_kind(opcode: u8) -> u8 {
    match opcode {
        0x1A => 1,
        0x1B => 2,
        0x1C => 3,
        0x1D => 4,
        0x1E => 5,
        0x1F => 6,
        0x15 => 7,
        _ => 0,
    }
}

/// Pure next-PC resolution from branch_kind + flags (replica of
/// `TileCpuV2::compute_branch_pc`).
fn compute_branch_pc(bk: u8, z: bool, c: bool, target: u32, fall: u32, lr: u32) -> u32 {
    match bk {
        1 | 6 => target, // JMP, CALL
        2 => {
            if z {
                target
            } else {
                fall
            }
        } // JZ
        3 => {
            if !z {
                target
            } else {
                fall
            }
        } // JNZ
        4 => {
            if c {
                target
            } else {
                fall
            }
        } // JC
        5 => {
            if !c {
                target
            } else {
                fall
            }
        } // JNC
        7 => lr & 0x7F,  // RET
        _ => fall,       // no branch
    }
}

/// HALT decode (`ctrl_b & 0x20`, opcode 0x01).
pub fn halts(d: &V2DecodedInstruction) -> bool {
    d.opcode == 0x01
}

/// LR after this instruction (Some only for CALL/CALLR, branch_kind 6):
/// `(pc + 1) & 0xFF`. RET reads LR; nothing else writes it in the charter regime
/// (recursion uses MFLR/JMPR, not MTLR — see capability-ladder).
pub fn lr_after(d: &V2DecodedInstruction, cur_pc: u32) -> Option<u32> {
    if branch_kind(d.opcode) == 6 {
        Some((cur_pc & 0xFF).wrapping_add(1) & 0xFF)
    } else {
        None
    }
}

/// Next PC for the extended-PC (8-bit) charter config. Mirrors the `if
/// self.extended_pc` arm of `run_stage_x` (v2_execute ~4967): JMPR/CALLR →
/// regs[rd], RET → lr, else `compute_branch_pc`, all masked to 8 bits.
pub fn next_pc_extended(
    d: &V2DecodedInstruction,
    cur_pc: u32,
    flag_z: bool,
    flag_c: bool,
    lr: u32,
    regs: &[u64; 16],
) -> u32 {
    const PCM: u32 = 0xFF;
    let bk = branch_kind(d.opcode);
    let reg_indirect = (d.ir_ext & EXT_REG_INDIRECT) != 0;
    let pc_full = cur_pc & PCM;
    let fall = pc_full.wrapping_add(1) & PCM;
    if reg_indirect && (bk == 1 || bk == 6) {
        (regs[d.rd as usize] as u32) & PCM
    } else if bk == 7 {
        lr & PCM
    } else {
        compute_branch_pc(bk, flag_z, flag_c, (d.imm16 as u32) & PCM, fall, lr)
    }
}

/// Runtime commit hold-mask set for a decoded instruction (review fix P1).
pub fn runtime_hold_mask(d: &V2DecodedInstruction) -> HoldMaskSet {
    let writes = writes_rd(d.opcode);
    let load = matches!(d.opcode, 0x16 | 0x18);
    let store = matches!(d.opcode, 0x17 | 0x19);
    let mul = d.opcode == 0x04 && d.imm5 == 1;
    let mflr = d.opcode == 0x02 && d.imm5 == 5;
    let ldi_w = d.opcode == 0x03 && d.wide_imm;
    HoldMaskSet {
        high_reg_merge: writes && d.rd >= 8,
        wb_data_mux: load || mul || mflr || ldi_w,
        ram_write_gate: store,
        flag_we_mul: mul,
    }
}

// ===================== 1b.1: tile datapath (ALU slice) =====================
//
// First datapath element run through the tile op arrays device-side. Seed a lane buffer
// from the pre-cycle sim (which already carries this instruction's decode from the prior
// fetch), inject the operands into the ALU trunk terminals (the high-reg re-source applied
// uniformly), run the trunk-settle op array, and read the ALU root. Validated against the
// authoritative value via the golden-neutral `last_alu_result` capture hook (which records
// it before later phases clobber the tile).

impl DeviceCyclePlan {
    /// Device ALU result for operands `(a, b)` against the current `sim` decode state.
    /// Mirrors `run_stage_x`'s operand-trunk inject + trunk-settle (~3466).
    #[allow(dead_code)] // exercised by the WIP device-ALU oracle (1b.1, ignored test).
    pub(crate) fn device_alu_result(&self, sim: &Simulation, a: u64, b: u64) -> u64 {
        let mut lp = LanePack::from_sim(sim, 1);
        lp.set(self.layout.alu_a_trunk_terminal_idx, 0, a);
        lp.set(self.layout.alu_b_trunk_terminal_idx, 0, b);
        // Hold-aware trunk settle: trunk_held omits the terminal-writing ops, so the
        // injected operands above are not recomputed from the (stale) operand tree.
        self.trunk_held.eval(&mut lp);
        lp.get(self.layout.wb_alu_root_idx, 0)
    }

    fn phase(&self, name: &str) -> &LaneKernel {
        &self
            .phases
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("phase '{name}' missing"))
            .1
    }

    /// 1b.1d device commit for a HIGH-reg (rd>=8) ALU op: run the ALU (trunk inject +
    /// hold-aware trunk-settle), inject the high-reg merge Const tiles (COP_CONST, so the
    /// dense commit kernel holds them), run commit, and return the merge-mux output — the
    /// value the clock edge would capture into the register. Mirrors run_stage_x's
    /// high_reg_injected path (~4178).
    ///
    /// We read the merge-mux output rather than running the clock phase: the real clock is
    /// `tick_clock_edge_compact` (a global-clock TOGGLE + capture), not a pure op-array, so
    /// a dense clock_compact_ops eval captures nothing. The captured register value IS the
    /// settled merge-mux output, so on-device the clock reduces to `reg = merge_output`.
    #[allow(dead_code)] // exercised by device_commit_matches_real_on_charter.
    pub(crate) fn device_commit_reg_high(
        &self,
        sim: &Simulation,
        a: u64,
        b: u64,
        rd_eff: usize,
    ) -> u64 {
        debug_assert!(rd_eff >= 8, "high-reg path requires rd_eff >= 8");
        let mut lp = LanePack::from_sim(sim, 1);
        // Stage X ALU.
        lp.set(self.layout.alu_a_trunk_terminal_idx, 0, a);
        lp.set(self.layout.alu_b_trunk_terminal_idx, 0, b);
        self.trunk_held.eval(&mut lp);
        let alu = lp.get(self.layout.wb_alu_root_idx, 0);
        // High-reg merge inject: wb_data + WE into the per-slot Const tiles; suppress the
        // low-reg WE mask (rd_hi). These Const tiles are held by the dense commit kernel.
        let slot = rd_eff - 8;
        lp.set(self.layout.high_reg_wb_data_const_indices[slot], 0, alu);
        lp.set(self.layout.high_reg_we_const_indices[slot], 0, u64::MAX);
        lp.set(self.layout.we_mask_const_idx, 0, 0);
        // Commit settles the merge mux; its output is the value the clock would capture.
        self.phase("commit").eval(&mut lp);
        lp.get(self.layout.high_reg_merge_mux_indices[slot], 0)
    }

    /// 1c.1: STANDALONE fetch+decode+ALU. Regenerate the decode from the bare ROM+PC with
    /// NO real-cycle seeding: set the PC tile, eval cone, apply the super-mux bridge
    /// (`final_mux -> super_mux_inject_right`, low-bank pc<64), eval settle. The decode +
    /// ALU op-select settle from the ROM. Then inject operands into the trunk and run the
    /// hold-aware trunk settle. Returns the ALU result.
    pub(crate) fn standalone_alu(&self, sim: &Simulation, pc: u32, a: u64, b: u64) -> u64 {
        let mut lp = LanePack::from_sim(sim, 1);
        // Stage F: fetch + decode from ROM at PC.
        lp.set(self.layout.pc_idx, 0, pc as u64);
        self.phase("cone").eval(&mut lp);
        for lane in 0..4 {
            let fm = lp.get(self.layout.final_mux_indices[lane], 0);
            lp.set(self.layout.super_mux_inject_right_indices[lane], 0, fm);
        }
        self.phase("settle").eval(&mut lp);
        // Stage X: inject operands, hold-aware trunk settle.
        lp.set(self.layout.alu_a_trunk_terminal_idx, 0, a);
        lp.set(self.layout.alu_b_trunk_terminal_idx, 0, b);
        self.trunk_held.eval(&mut lp);
        lp.get(self.layout.wb_alu_root_idx, 0)
    }
}

/// 1c: standalone NON-PIPELINED device CPU. Architectural state lives in `DeviceCycleState`
/// (scalar); the ALU is computed via TILES (standalone fetch+decode+ALU over a persistent
/// lane buffer — decode regenerated from ROM+PC, operands trunk-injected); the rest of the
/// charter ISA is the megakernel's scalar shell. Validated run-to-halt vs `TileCpuV2`.
///
/// ## Coverage (extended past the charter set, 2026-07-02)
/// VALIDATED bit-exact (per-retired-instruction regs+flags+lr, final RAM/R0/halted/retired):
/// NOP, HALT, MOV, MFLR, LDI, ALU reg (ADD/SUB/AND/OR/XOR/NOT/NEG/INC/DEC + CMP), LD/LDB,
/// ST/STB, JMP/JZ/JNZ/CALL/RET/JMPR — plus the extended-corpus additions below.
/// SCALAR (mirrors the real CPU, where these are stage-X software authority / no tile
/// path): MUL (0x04 imm5==1: wrapping mul, Z only, C unchanged), conditional moves
/// (MOVZ/MOVNZ/MOVC/MOVNC on pre-instruction flags, no flag writes), MTLR,
/// SHL/SHR carry (ISS bit-position formulas from imm8).
/// MMIO console (addr 57/58): STB 57 appends to `console` (no RAM write); reads of 57/58
/// return last-byte/count. Other MMIO devices (41..=56, 59..=63) are NOT modeled —
/// programs touching them diverge loudly against the oracle.
/// UPPER BANK (pc 64..128): the super-mux bridge injects BOTH banks
/// (final_mux → right, bank47_mux[(pc>>4)&3] → left), mirroring `stage_f_inject`.
/// OUT OF SCOPE: pc >= 128 (no physical ROM on the real CPU either — it falls back to
/// gated software Const-injection there), SRA overlay (0x0E imm8&0x08, unvalidated).
pub struct DeviceCpu {
    pub state: DeviceCycleState,
    /// MMIO console bytes (STB to addr 57), mirroring the ISS/hardware console buffer.
    pub console: Vec<u8>,
    plan: DeviceCyclePlan,
    buf: LanePack,
    program: Vec<u32>,
    mem_mask: usize,
}

impl DeviceCpu {
    #[allow(dead_code)] // 1c driver (task #5), exercised by device_cpu_* tests.
    pub(crate) fn new(plan: DeviceCyclePlan, sim: &Simulation, program: Vec<u32>) -> Self {
        let buf = LanePack::from_sim(sim, 1);
        let state = DeviceCycleState {
            pc: 0,
            lr: 0,
            flag_z: false,
            flag_c: false,
            halted: false,
            cycle: 0,
            retired: 0,
            regs: [0; 16],
            ram: vec![0; 128],
        };
        assert!(
            program.len() <= 128,
            "DeviceCpu: physical decode covers pc < 128 (the real CPU has no physical \
             ROM past 128 either); program has {} words",
            program.len()
        );
        Self {
            state,
            console: Vec::new(),
            plan,
            buf,
            program,
            mem_mask: 0x7F,
        }
    }

    /// Tile ALU: regenerate decode from ROM at `pc`, inject operands, hold-aware trunk settle.
    /// The super-mux bridge injects BOTH banks (mirroring `stage_f_inject`): bank03 from
    /// `final_mux` into the right input, bank47 from `bank47_mux[(pc>>4)&3]` into the left;
    /// the physical PC[6] selector picks — so pc 0..128 decodes physically.
    fn alu(&mut self, pc: u32, a: u64, b: u64) -> u64 {
        let l = &self.plan.layout;
        let (pc_idx, fm, smir, smi, b47, ta, tb, wb) = (
            l.pc_idx,
            l.final_mux_indices,
            l.super_mux_inject_right_indices,
            l.super_mux_inject_indices,
            l.bank47_mux_indices,
            l.alu_a_trunk_terminal_idx,
            l.alu_b_trunk_terminal_idx,
            l.wb_alu_root_idx,
        );
        self.buf.set(pc_idx, 0, pc as u64);
        self.plan.phase("cone").eval(&mut self.buf);
        let bank_within = ((pc >> 4) & 3) as usize;
        for lane in 0..4 {
            let v = self.buf.get(fm[lane], 0);
            self.buf.set(smir[lane], 0, v);
            let v47 = self.buf.get(b47[bank_within][lane], 0);
            self.buf.set(smi[lane], 0, v47);
        }
        self.plan.phase("settle").eval(&mut self.buf);
        self.buf.set(ta, 0, a);
        self.buf.set(tb, 0, b);
        self.plan.trunk_held.eval(&mut self.buf);
        self.buf.get(wb, 0)
    }

    fn mem_addr(&self, d: &V2DecodedInstruction) -> usize {
        let byte_variant = matches!(d.opcode, 0x18 | 0x19);
        if d.has_offset {
            (self.state.regs[d.rs as usize].wrapping_add(d.offset as u64) as usize) & self.mem_mask
        } else if byte_variant {
            (d.imm8 as usize) & self.mem_mask
        } else {
            (self.state.regs[d.rs as usize] as usize) & self.mem_mask
        }
    }

    #[allow(dead_code)]
    pub(crate) fn step(&mut self) {
        if self.state.halted {
            return;
        }
        let pc = self.state.pc;
        let d = decode_v2_word(self.program[pc as usize]);
        let rd = d.rd as usize;
        let rs = d.rs as usize;
        let (fz0, fc0) = (self.state.flag_z, self.state.flag_c);
        let (a, b) = (self.state.regs[rd], self.state.regs[rs]);

        let mut wb_val: Option<(usize, u64)> = None;
        let mut new_z: Option<bool> = None;
        let mut new_c: Option<bool> = None;

        match d.opcode {
            0x00 => {}                        // NOP
            0x01 => self.state.halted = true, // HALT
            0x02 => match d.imm5 {
                // MOV family (sub-fn selects; mirrors the ISS 0x02 dispatch).
                5 => wb_val = Some((rd, self.state.lr as u64)), // MFLR
                // MTLR: lr = rs. This DeviceCycle oracle is 8-bit by construction (PCM =
                // 0xFF for every PC op — fall-through, RET, JMPR, branches; see next_pc),
                // so 0xFF is the correct in-domain mask, NOT a wide-mode bug (Sprint 390
                // re-verify). Wide programs use the authoritative ISS (v2_iss.rs) + the
                // physical executor, which mask LR by pc_mask / pc_phys_mask.
                6 => self.state.lr = (b as u32) & 0xFF, // MTLR (8-bit oracle PC width)
                sub => {
                    // MOV (0) / MOVZ/MOVNZ/MOVC/MOVNC (1..4) on PRE-instruction flags;
                    // no flag writes.
                    let cond = match sub {
                        1 => fz0,
                        2 => !fz0,
                        3 => fc0,
                        4 => !fc0,
                        _ => true,
                    };
                    if cond {
                        wb_val = Some((rd, b));
                    }
                }
            },
            0x03 => {
                // LDI (imm16==imm8 when not wide). LDI writes the Z flag (run_stage_x
                // writes_flags ~4147); C is unchanged.
                let v = d.imm16 as u64;
                wb_val = Some((rd, v));
                new_z = Some(v == 0);
            }
            0x04 if d.imm5 == 1 => {
                // MUL: stage-X SOFTWARE authority on the real CPU too (no tile path) —
                // scalar here is authority-faithful. Writes Z only; C unchanged.
                let r = a.wrapping_mul(b);
                wb_val = Some((rd, r));
                new_z = Some(r == 0);
            }
            0x04..=0x0E => {
                let r = self.alu(pc, a, b); // tiles
                wb_val = Some((rd, r));
                new_z = Some(r == 0);
                new_c = device_carry_ext(d.opcode, d.imm8, a, r, fc0);
            }
            0x0F => {
                let r = self.alu(pc, a, b); // CMP: flags only
                new_z = Some(r == 0);
                new_c = device_carry_ext(0x0F, d.imm8, a, r, fc0);
            }
            0x10..=0x14 => {
                let r = self.alu(pc, a, b); // ALU immediate
                wb_val = Some((rd, r));
                new_z = Some(r == 0);
                new_c = device_carry_ext(d.opcode, d.imm8, a, r, fc0);
            }
            0x16 | 0x18 => {
                let addr = self.mem_addr(&d); // LD / LDB (console taps at 57/58)
                let v = match addr {
                    57 => self.console.last().copied().unwrap_or(0) as u64,
                    58 => self.console.len() as u64,
                    _ => self.state.ram[addr],
                };
                wb_val = Some((rd, v));
                new_z = Some(v == 0);
            }
            0x17 | 0x19 => {
                let addr = self.mem_addr(&d); // ST / STB (store rd)
                match addr {
                    57 => self.console.push((a & 0xFF) as u8), // console: NOT a RAM write
                    58 => {}                                   // count is read-only
                    _ => self.state.ram[addr] = a,
                }
            }
            0x15 | 0x1A..=0x1F => {} // RET / JMP / JZ / JNZ / JC / JNC / CALL — control only
            other => panic!("DeviceCpu: unhandled opcode {other:#04x} at pc {pc}"),
        }

        if let Some((r, v)) = wb_val {
            self.state.regs[r] = v;
        }
        if let Some(z) = new_z {
            self.state.flag_z = z;
        }
        if let Some(c) = new_c {
            self.state.flag_c = c;
        }
        if !self.state.halted {
            self.state.pc = next_pc_extended(&d, pc, fz0, fc0, self.state.lr, &self.state.regs);
            if let Some(lr) = lr_after(&d, pc) {
                self.state.lr = lr;
            }
        }
        self.state.retired += 1;
        // Non-pipelined: one device cycle per instruction (cycle == retired). The pipelined
        // TileCpuV2's tick count is a different quantity and is never compared to this.
        self.state.cycle += 1;
    }

    #[allow(dead_code)]
    pub(crate) fn run_to_halt(&mut self, max: u64) {
        for _ in 0..max {
            if self.state.halted {
                break;
            }
            self.step();
        }
    }
}

/// ALU-class opcodes whose `wb_alu_root` is the logical ALU result: reg ALU (0x04..=0x0F,
/// incl. CMP) + immediate ALU (0x10..=0x14). MUL (0x04, imm5==1) and SRA overlay are
/// charter-absent; a stray one would surface as a validation mismatch, not silent error.
pub fn is_alu_class(opcode: u8) -> bool {
    matches!(opcode, 0x04..=0x14)
}

/// Device carry flag for an ALU op, from operand `a`, the ALU `result`, and the
/// pre-instruction carry. Mirrors `run_stage_x`'s per-opcode `next_flag_c` (~3612):
/// add-like `a > result`, sub-like `result > a`, bitwise unchanged. Returns None for
/// SHL/SHR (0x0D/0x0E — bit-position carry needs imm8; charter-absent) and non-ALU.
pub fn device_carry(opcode: u8, a: u64, result: u64, carry_before: bool) -> Option<bool> {
    match opcode {
        0x04 | 0x0B | 0x10 => Some(a > result), // ADD / INC / ADDI
        0x05 | 0x0A | 0x0C | 0x0F | 0x11 => Some(result > a), // SUB / NEG / DEC / CMP / SUBI
        0x06 | 0x07 | 0x08 | 0x09 | 0x12 | 0x13 | 0x14 => Some(carry_before), // bitwise: unchanged
        _ => None,                              // SHL/SHR + non-ALU
    }
}

/// `device_carry` extended with the shift carries, which need `imm8` (the shift amount
/// is an immediate). Mirrors the ISS exactly: SHL carry = bit `(8 - imm8) & 63` of `a`
/// (the ISA's legacy 8-bit-era formula, hardware parity); SHR carry = bit
/// `((imm8 & 7) - 1) & 63` of `a` (last bit shifted out; shift amount is 3 bits).
pub fn device_carry_ext(
    opcode: u8,
    imm8: u8,
    a: u64,
    result: u64,
    carry_before: bool,
) -> Option<bool> {
    match opcode {
        0x0D => {
            let bit = 8u64.wrapping_sub(imm8 as u64) & 63;
            Some((a >> bit) & 1 != 0)
        }
        0x0E => {
            let bit = ((imm8 & 0x07) as u64).wrapping_sub(1) & 63;
            Some((a >> bit) & 1 != 0)
        }
        _ => device_carry(opcode, a, result, carry_before),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_full_phase_prover::charter_cases;
    use crate::tile_cpu::{
        V2Builder, V2Iss, V2MmioCombinedDevice, V2MmioHandle, V2SynthConfig, compile_source,
    };
    use std::rc::Rc;

    /// Build a charter-config CPU + its plan. Derives indices on a throwaway sim BEFORE
    /// `build()` consumes the builder (so the same config drives both).
    fn build_charter_plan(source: &str) -> (TileCpuV2, Simulation, DeviceCyclePlan) {
        let program = compile_source(source).expect("compile");
        let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
        let builder = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(console))
            .with_synth_blocks(V2SynthConfig::max_authority());
        let mut scratch = Simulation::with_size_layered(128, 640, 16);
        let layout = builder.derive_indices(&mut scratch);
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = builder.build(&mut sim);
        let plan = DeviceCyclePlan::new(&cpu, &sim, layout).expect("plan");
        (cpu, sim, plan)
    }

    /// 1a core proof: `derive_indices` recovers the SAME tile indices the real CPU uses
    /// (validated against the test-only accessor), all SIX phase kernels build op-clean
    /// (incl. trunk-settle), the unified universe is non-empty, and `DeviceCycleState`
    /// captures with `lr`/`cycle`/`retired`.
    #[test]
    fn device_cycle_plan_indices_match_real_cpu() {
        let case = charter_cases()[0]; // fib_recursive
        let (cpu, sim, plan) = build_charter_plan(case.source);

        // The recovered readback taps must equal the real CPU's private indices.
        let info = cpu.clock_scope_mask_test_info();
        assert_eq!(plan.taps.regs, *info.reg_indices, "reg index mismatch");
        assert_eq!(plan.taps.flag_z, info.flag_z_idx, "flag_z index mismatch");
        assert_eq!(plan.taps.flag_c, info.flag_c_idx, "flag_c index mismatch");
        assert_eq!(plan.taps.pc, info.pc_idx, "pc index mismatch");
        assert_eq!(plan.taps.ram, *info.ram_indices, "ram index mismatch");

        // Six phases present in tick order (trunk_settle first), non-empty universe.
        assert_eq!(plan.phases.len(), 6);
        for (i, (name, _)) in plan.phases.iter().enumerate() {
            assert_eq!(*name, DEVICE_PHASES[i]);
        }
        assert!(!plan.universe.is_empty(), "unified universe empty");
        assert!(
            plan.layout_indices()
                .reg_indices
                .iter()
                .all(|&t| t < sim.tilemap.tile_count())
        );
        // Charter config uses the physical ALU → trunk-settle op array is populated.
        assert!(
            plan.has_trunk_settle_ops(),
            "charter config should expose a non-empty trunk-settle op array"
        );

        // State capture includes lr/cycle/retired (1b oracle) + RAM SoA.
        let state = DeviceCycleState::capture(&cpu, &sim, 128);
        assert_eq!(state.ram.len(), 128);
        assert_eq!(state.diff(&state).len(), 0, "self-diff must be empty");
    }

    /// The diff catches a RAM length mismatch instead of silently passing (review fix P2).
    #[test]
    fn diff_detects_ram_length_mismatch() {
        let a = DeviceCycleState {
            pc: 0,
            lr: 0,
            flag_z: false,
            flag_c: false,
            halted: false,
            cycle: 0,
            retired: 0,
            regs: [0; 16],
            ram: vec![0; 128],
        };
        let mut b = a.clone();
        b.ram = vec![0; 64];
        assert!(
            a.diff(&b).iter().any(|s| s.contains("ram len")),
            "length mismatch must surface in diff"
        );
    }

    /// The incidence classifier matches the spec's commit Const-swap set on known words.
    #[test]
    fn commit_swap_incidence_decodes_known_forms() {
        // Plain low-reg ADD (R0 = R1 + R2): no swap.
        let add = (0x04u32 << 11) | (1 << 5) | 2;
        let c = CommitSwapIncidence::classify(add);
        assert!(!c.any(), "plain low-reg ADD should need no swap: {c:?}");

        // MUL (opcode 0x04, imm5==1) → wb_data_mux + flag_we_mul.
        let mul = (0x04u32 << 11) | 1;
        let c = CommitSwapIncidence::classify(mul);
        assert!(
            c.wb_data_mux && c.flag_we_mul && c.any(),
            "MUL incidence: {c:?}"
        );
    }

    /// 1b.0 oracle: the scalar control front-end (next-PC across all branch kinds,
    /// lr on CALL/CALLR, halt) matches `V2Iss` bit-exact every cycle on the charter
    /// programs. Drives the ISS in lockstep and validates the PC-transition function
    /// against it. (Fast, CPU-only — the charter programs halt in <5k cycles.)
    #[test]
    fn device_control_matches_iss_on_charter() {
        for case in charter_cases() {
            let program = compile_source(case.source).expect("compile");
            let mut iss = V2Iss::from_program(&program);
            iss.enable_extended_pc(); // match the charter physical config (8-bit PC).

            let mut cycles = 0u64;
            while !iss.state().halted && cycles < case.max_cycles {
                // Snapshot pc/flags/lr/regs BEFORE the step (no Clone dependency).
                let (pc, fz, fc, lr) = {
                    let s = iss.state();
                    (s.pc, s.flag_z, s.flag_c, s.lr)
                };
                let regs = iss.state().regs; // [u64; 16] is Copy
                assert!(
                    (pc as usize) < program.len(),
                    "{}: pc {pc} out of program bounds ({} words)",
                    case.name,
                    program.len()
                );
                let d = decode_v2_word(program[pc as usize]);
                let my_next = next_pc_extended(&d, pc, fz, fc, lr, &regs);
                let my_lr = lr_after(&d, pc).unwrap_or(lr);
                let my_halt = halts(&d);

                iss.step();
                let after = iss.state();

                if my_halt {
                    assert!(after.halted, "{}: cycle {cycles} expected halt", case.name);
                    // pc/lr after halt are don't-care (execution stops).
                } else {
                    assert!(
                        !after.halted,
                        "{}: cycle {cycles} unexpected halt",
                        case.name
                    );
                    assert_eq!(
                        my_next, after.pc,
                        "{}: cycle {cycles} next-pc mine {my_next} != iss {}",
                        case.name, after.pc
                    );
                    assert_eq!(
                        my_lr, after.lr,
                        "{}: cycle {cycles} lr mine {my_lr} != iss {}",
                        case.name, after.lr
                    );
                }
                cycles += 1;
            }
            assert!(
                iss.state().halted,
                "{}: ISS did not halt within {} cycles",
                case.name,
                case.max_cycles
            );
        }
    }

    /// The runtime hold-mask generator folds in write-intent: a high-reg ADD needs the
    /// merge hold, but a CMP (no rd write) to a high reg does NOT.
    #[test]
    fn runtime_hold_mask_uses_write_intent() {
        use crate::tile_cpu::v2_components::EXT_RD_HI;
        let rd_hi = u32::from(EXT_RD_HI) << 16;

        // ADD writing R8 (high): rd_lo=0 + rd_hi -> rd=8; imm5=2 (plain ADD, not MUL).
        let add_r8 = (0x04u32 << 11) | 2 | rd_hi;
        let d = decode_v2_word(add_r8);
        assert!(d.rd >= 8 && writes_rd(d.opcode), "decoded {d:?}");
        assert!(
            runtime_hold_mask(&d).high_reg_merge,
            "high-reg ADD needs merge hold"
        );

        // CMP (0x0F) with rd_hi set: does NOT write rd -> no high_reg_merge.
        let cmp_hi = (0x0Fu32 << 11) | rd_hi;
        let d = decode_v2_word(cmp_hi);
        assert!(!writes_rd(d.opcode), "CMP must not write rd");
        assert!(
            !runtime_hold_mask(&d).high_reg_merge,
            "CMP (no rd write) must not request the high-reg merge hold"
        );
    }

    /// 1b.1b oracle: the device ALU (operand inject into trunk terminals + hold-aware
    /// trunk-settle) reproduces the authoritative `wb_alu_root` bit-exact on every
    /// ALU-class cycle of the charter programs, validated via the golden-neutral
    /// `last_alu_result` capture hook. The hold-aware kernel omits the terminal-writing ops
    /// so the injection is not clobbered (the gap the 1b.1a oracle localized).
    #[test]
    fn device_alu_matches_real_on_charter() {
        for case in charter_cases() {
            let (cpu, mut sim, plan) = build_charter_plan(case.source);
            let mut checked = 0u64;
            let mut cycles = 0u64;
            while !cpu.is_halted() && cycles < case.max_cycles {
                // Compute the device ALU from the PRE-cycle sim (decode already present),
                // for the instruction about to execute, on ALU-class cycles.
                let device = cpu.latch_operands().and_then(|(a, b, op)| {
                    is_alu_class(op).then(|| (plan.device_alu_result(&sim, a, b), op))
                });
                cpu.step(&mut sim);
                if let Some((dev, op)) = device {
                    let real = cpu.last_alu_result();
                    assert_eq!(
                        dev, real,
                        "{}: cycle {cycles} ALU op {op:#04x}: device {dev} != real {real}",
                        case.name
                    );
                    checked += 1;
                }
                cycles += 1;
            }
            assert!(
                cpu.is_halted(),
                "{}: did not halt in {} cycles",
                case.name,
                case.max_cycles
            );
            assert!(
                checked > 0,
                "{}: no ALU-class cycles were validated",
                case.name
            );
        }
    }

    /// 1b.1c (writeback target): the device ALU result lands in the destination register —
    /// `reg[rd_eff]` after the cycle equals the device ALU result on every ALU reg-write
    /// cycle of the charter set. Pins down the target+value the device commit must produce
    /// (the full hold-aware device commit is the next slice).
    #[test]
    fn device_alu_lands_in_dest_reg_on_charter() {
        for case in charter_cases() {
            let (cpu, mut sim, plan) = build_charter_plan(case.source);
            let mut checked = 0u64;
            let mut cycles = 0u64;
            while !cpu.is_halted() && cycles < case.max_cycles {
                // ALU reg-write cycle: compute the device ALU result from the pre-cycle sim.
                let device =
                    cpu.latch_operands()
                        .zip(cpu.latch_rd_eff())
                        .and_then(|((a, b, op), rd)| {
                            (is_alu_class(op) && writes_rd(op))
                                .then(|| (plan.device_alu_result(&sim, a, b), rd))
                        });
                cpu.step(&mut sim);
                if let Some((dev, rd)) = device {
                    let reg = cpu.read_reg(&sim, rd);
                    assert_eq!(
                        dev, reg,
                        "{}: cycle {cycles} R{rd} device ALU {dev} != committed reg {reg}",
                        case.name
                    );
                    checked += 1;
                }
                cycles += 1;
            }
            assert!(cpu.is_halted(), "{}: did not halt", case.name);
            assert!(
                checked > 0,
                "{}: no ALU reg-write cycles validated",
                case.name
            );
        }
    }

    /// 1b.1d oracle: the device commit (ALU -> high-reg merge inject -> commit + clock)
    /// captures the correct value into the destination register, matching TileCpuV2's
    /// `read_reg` post-step on HIGH-reg ALU writes. Sampled (commit+clock are ~19k ops
    /// each) — proves the device commit mechanism on real charter cycles.
    #[test]
    fn device_commit_matches_real_on_charter() {
        const SAMPLE: u64 = 6;
        for case in charter_cases() {
            let (cpu, mut sim, plan) = build_charter_plan(case.source);
            let mut checked = 0u64;
            let mut cycles = 0u64;
            while !cpu.is_halted() && cycles < case.max_cycles && checked < SAMPLE {
                // High-reg ALU reg-write cycle.
                let device =
                    cpu.latch_operands()
                        .zip(cpu.latch_rd_eff())
                        .and_then(|((a, b, op), rd)| {
                            (is_alu_class(op) && writes_rd(op) && rd >= 8)
                                .then(|| (plan.device_commit_reg_high(&sim, a, b, rd), rd))
                        });
                cpu.step(&mut sim);
                if let Some((dev_reg, rd)) = device {
                    assert_eq!(
                        dev_reg,
                        cpu.read_reg(&sim, rd),
                        "{}: cycle {cycles} R{rd} device commit {dev_reg} != real {}",
                        case.name,
                        cpu.read_reg(&sim, rd)
                    );
                    checked += 1;
                }
                cycles += 1;
            }
            assert!(
                checked > 0,
                "{}: no high-reg ALU reg-write cycles found",
                case.name
            );
        }
    }

    /// 1b.1e oracle: the device Z flag matches the real committed flag. Every ALU-class
    /// opcode (0x04..=0x14) writes flags, and the Z flag is `alu_result == 0`, so the device
    /// ALU result determines it. Validated vs `read_flag_z` post-step every ALU cycle of the
    /// charter set. (Carry is op-specific — a later slice.)
    #[test]
    fn device_z_flag_matches_real_on_charter() {
        for case in charter_cases() {
            let (cpu, mut sim, plan) = build_charter_plan(case.source);
            let mut checked = 0u64;
            let mut cycles = 0u64;
            while !cpu.is_halted() && cycles < case.max_cycles {
                let device_z = cpu.latch_operands().and_then(|(a, b, op)| {
                    is_alu_class(op).then(|| plan.device_alu_result(&sim, a, b) == 0)
                });
                cpu.step(&mut sim);
                if let Some(z) = device_z {
                    assert_eq!(
                        z,
                        cpu.read_flag_z(&sim),
                        "{}: cycle {cycles} Z flag device {z} != real {}",
                        case.name,
                        cpu.read_flag_z(&sim)
                    );
                    checked += 1;
                }
                cycles += 1;
            }
            assert!(cpu.is_halted(), "{}: did not halt", case.name);
            assert!(checked > 0, "{}: no ALU cycles validated", case.name);
        }
    }

    /// 1c.2/1c.3 oracle: the standalone `DeviceCpu` (register mirror -> tile ALU -> arch-state
    /// update, chained) matches `TileCpuV2` on the FULL charter set — sum_loop (loop),
    /// gcd_subtract (CALL/RET), fib_recursive (recursion: MFLR/JMPR + stack). Validated
    /// PER-RETIRED-INSTRUCTION (regs + flags after device instr K == TileCpuV2 after K
    /// retirements; pc is pipeline-offset so excluded), then final RAM + R0 at halt.
    fn validate_device_cpu(name: &str) {
        let case = charter_cases()
            .into_iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("missing charter program {name}"));
        validate_device_cpu_case(
            case.name,
            case.source,
            case.max_cycles,
            case.expected_r0,
            "",
        );
    }

    /// Per-retired-instruction oracle vs the REAL `TileCpuV2` (regs/flags/lr each retired
    /// instruction; final retired/R0/RAM), plus console vs the ISS MMIO mirror. Used for
    /// both the charter set and the extended corpus (MUL, upper-bank pc>=64, console).
    fn validate_device_cpu_case(
        name: &str,
        source: &str,
        max_cycles: u64,
        expected_r0: u64,
        expected_console: &str,
    ) {
        let program = compile_source(source).expect("compile");

        let (_c, sim, plan) = build_charter_plan(source);
        let mut dev = DeviceCpu::new(plan, &sim, program.clone());
        let (cpu, mut csim, _) = build_charter_plan(source);

        let mut instrs = 0u64;
        while !dev.state.halted && instrs < max_cycles {
            dev.step();
            instrs += 1;
            // Advance the pipelined reference until it has RETIRED `instrs` instructions.
            while cpu.read_retired_count() < instrs && !cpu.is_halted() {
                cpu.step(&mut csim);
            }
            for r in 0..16 {
                assert_eq!(
                    dev.state.regs[r],
                    cpu.read_reg(&csim, r),
                    "{name}: after instr {instrs} R{r}: device {} != tilecpu {}",
                    dev.state.regs[r],
                    cpu.read_reg(&csim, r)
                );
            }
            assert_eq!(
                dev.state.flag_z,
                cpu.read_flag_z(&csim),
                "{name}: after instr {instrs} flag_z"
            );
            assert_eq!(
                dev.state.flag_c,
                cpu.read_flag_c(&csim),
                "{name}: after instr {instrs} flag_c"
            );
            assert_eq!(
                dev.state.lr,
                cpu.read_lr(),
                "{name}: after instr {instrs} lr: device {} != tilecpu {}",
                dev.state.lr,
                cpu.read_lr()
            );
        }
        assert!(
            dev.state.halted,
            "{name}: device did not halt in {instrs} instrs"
        );
        assert!(cpu.is_halted(), "{name}: reference did not halt");
        assert_eq!(
            dev.state.retired,
            cpu.read_retired_count(),
            "{name}: final retired count"
        );
        assert_eq!(dev.state.regs[0], expected_r0, "{name}: final R0");
        for a in 0..128 {
            assert_eq!(
                dev.state.ram[a],
                cpu.read_ram(&csim, a),
                "{name}: final RAM[{a}]"
            );
        }

        // Console: the hardware buffer lives inside the MMIO device, so compare against
        // the ISS mirror (proven == hardware by the compiler-bench goldens) + the golden.
        let mut iss = V2Iss::from_program_with_mmio(&program);
        iss.enable_extended_pc();
        let mut n = 0u64;
        while !iss.state().halted && n < max_cycles {
            iss.step();
            n += 1;
        }
        assert_eq!(
            iss.console_string(),
            expected_console,
            "{name}: ISS console golden"
        );
        assert_eq!(
            String::from_utf8_lossy(&dev.console),
            expected_console,
            "{name}: device console"
        );
    }

    #[test]
    fn device_cpu_sum_loop_matches_tilecpu() {
        validate_device_cpu("sum_loop");
    }

    #[test]
    fn device_cpu_gcd_matches_tilecpu() {
        validate_device_cpu("gcd_subtract");
    }

    #[test]
    fn device_cpu_fib_matches_tilecpu() {
        validate_device_cpu("fib_recursive");
    }

    /// Extended-corpus case lookup (the compiled benchmark corpus, past the charter set).
    fn bench_case(name: &str) -> &'static crate::tile_cpu::V2CompiledBenchCase {
        crate::tile_cpu::V2_COMPILED_BENCHMARKS
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("missing bench corpus program {name}"))
    }

    fn validate_device_cpu_bench(name: &str) {
        let c = bench_case(name);
        validate_device_cpu_case(
            c.name,
            c.source,
            c.max_cycles,
            c.expected_r0,
            c.expected_console,
        );
    }

    /// MUL coverage (stage-X software authority; scalar on the device too): fact(10).
    #[test]
    fn device_cpu_fact_matches_tilecpu() {
        validate_device_cpu_bench("fact_iterative");
    }

    /// MUL-in-loop coverage: sum of squares.
    #[test]
    fn device_cpu_sum_squares_matches_tilecpu() {
        validate_device_cpu_bench("sum_squares");
    }

    /// UPPER-BANK coverage (80 program words, pc crosses 64): the two-sided super-mux
    /// bridge (final_mux -> right, bank47_mux[(pc>>4)&3] -> left) must decode pc 64..128
    /// physically. Also exercises MUL + indexed (base+offset) RAM addressing.
    #[test]
    fn device_cpu_array_matches_tilecpu() {
        validate_device_cpu_bench("array_sum_squares");
    }

    /// MMIO console coverage: a compiled-from-source putchar program prints through the
    /// device console (STB addr 57 appends a byte, does NOT write RAM).
    #[test]
    fn device_cpu_console_matches_iss() {
        validate_device_cpu_case(
            "console_hi",
            r#"
                int main() { putchar(72); putchar(73); putchar(33); return 0; }
            "#,
            10_000,
            0,
            "HI!",
        );
    }

    /// Conditional moves + shift carries + MTLR vs the ISS, per instruction (pc, regs,
    /// flags, lr). These are scalar paths (cmov/MTLR) or scalar-carry-over-tile-result
    /// (SHL/SHR) that no compiled corpus program emits — hand-assembled coverage.
    #[test]
    fn device_cpu_cmov_shift_mtlr_matches_iss() {
        let ldi = |rd: u32, imm: u32| (0x03u32 << 11) | (rd << 8) | imm;
        let cmp = |rd: u32, rs: u32| (0x0Fu32 << 11) | (rd << 8) | (rs << 5);
        let mov_sub = |rd: u32, rs: u32, sub: u32| (0x02u32 << 11) | (rd << 8) | (rs << 5) | sub;
        let shl = |rd: u32, imm: u32| (0x0Du32 << 11) | (rd << 8) | imm;
        let shr = |rd: u32, imm: u32| (0x0Eu32 << 11) | (rd << 8) | imm;
        let program = vec![
            ldi(1, 5),            //  0: R1 = 5
            ldi(2, 9),            //  1: R2 = 9
            cmp(1, 2),            //  2: CMP R1,R2 -> Z=0, C=1 (5-9 borrows)
            mov_sub(3, 1, 3),     //  3: MOVC  R3,R1 -> taken (C=1): R3=5
            mov_sub(4, 2, 4),     //  4: MOVNC R4,R2 -> not taken
            mov_sub(5, 2, 1),     //  5: MOVZ  R5,R2 -> not taken (Z=0)
            mov_sub(6, 2, 2),     //  6: MOVNZ R6,R2 -> taken: R6=9
            shl(1, 3),            //  7: SHL R1,3 -> R1=40, C=bit5(5)=0
            (0x1Eu32 << 11) | 10, // 8: JNC 10 -> taken (C=0)
            ldi(7, 99),           //  9: (skipped)
            shr(1, 2),            // 10: SHR R1,2 -> R1=10, C=bit1(40)=0
            mov_sub(0, 2, 6),     // 11: MTLR R2 -> lr=9
            mov_sub(7, 0, 5),     // 12: MFLR R7 -> R7=9
            (0x1Du32 << 11) | 15, // 13: JC 15 -> not taken (C=0)
            0x01u32 << 11,        // 14: HALT
        ];
        let d3 = decode_v2_word(program[3]);
        assert!(
            d3.opcode == 0x02 && d3.imm5 == 3 && d3.rd == 3 && d3.rs == 1,
            "{d3:?}"
        );

        // Device (full tile plan for the SHL/SHR tile results) vs ISS, per instruction.
        let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
        let builder = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(console))
            .with_synth_blocks(V2SynthConfig::max_authority());
        let mut scratch = Simulation::with_size_layered(128, 640, 16);
        let layout = builder.derive_indices(&mut scratch);
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = builder.build(&mut sim);
        let plan = DeviceCyclePlan::new(&cpu, &sim, layout).expect("plan");
        let mut dev = DeviceCpu::new(plan, &sim, program.clone());

        let mut iss = V2Iss::from_program(&program);
        iss.enable_extended_pc();
        let mut n = 0u64;
        while !dev.state.halted && n < 100 {
            dev.step();
            iss.step();
            n += 1;
            let s = iss.state();
            assert_eq!(dev.state.halted, s.halted, "instr {n}: halted");
            if !s.halted {
                assert_eq!(dev.state.pc, s.pc, "instr {n}: pc");
            }
            assert_eq!(dev.state.regs, s.regs, "instr {n}: regs");
            assert_eq!(dev.state.flag_z, s.flag_z, "instr {n}: Z");
            assert_eq!(dev.state.flag_c, s.flag_c, "instr {n}: C");
            assert_eq!(dev.state.lr, s.lr, "instr {n}: lr");
        }
        assert!(dev.state.halted, "did not halt");
        assert_eq!(dev.state.regs[3], 5, "MOVC");
        assert_eq!(dev.state.regs[4], 0, "MOVNC must not write");
        assert_eq!(dev.state.regs[6], 9, "MOVNZ");
        assert_eq!(dev.state.regs[1], 10, "SHL/SHR datapath");
        assert_eq!(dev.state.regs[7], 9, "MTLR->MFLR round trip");
    }

    /// 1c.1 oracle: the STANDALONE fetch+decode+ALU (decode regenerated from bare ROM+PC,
    /// no real-cycle seeding) computes the correct binary-register ALU op. Proves the
    /// standalone fetch+decode produces correct ALU control + datapath.
    #[test]
    fn standalone_alu_regenerates_and_computes() {
        let case = charter_cases()[1]; // sum_loop
        let program = compile_source(case.source).expect("compile");
        let (_cpu, sim, plan) = build_charter_plan(case.source);

        // First binary-register ALU instruction (ADD/SUB/AND/OR/XOR, not MUL) at pc<64.
        let (pc, op) = program
            .iter()
            .enumerate()
            .find_map(|(i, &w)| {
                let d = decode_v2_word(w);
                let binary = matches!(d.opcode, 0x05..=0x08) || (d.opcode == 0x04 && d.imm5 != 1);
                (i < 64 && binary).then_some((i as u32, d.opcode))
            })
            .expect("no binary-register ALU instruction in program");

        let (a, b) = (0x0000_0000_1234_5678u64, 0x0000_0000_0000_abcdu64);
        let device = plan.standalone_alu(&sim, pc, a, b);
        let expected = match op {
            0x04 => a.wrapping_add(b),
            0x05 => a.wrapping_sub(b),
            0x06 => a & b,
            0x07 => a | b,
            0x08 => a ^ b,
            _ => unreachable!(),
        };
        assert_eq!(
            device, expected,
            "standalone ALU op {op:#04x} at pc {pc}: device {device:#018x} != expected {expected:#018x}"
        );
    }

    /// 1b.1f oracle: the device carry flag matches the real committed flag every ALU cycle
    /// (excl. SHL/SHR) of the charter set, via the per-opcode derivation from the device ALU
    /// result + pre-instruction carry. Validated vs `read_flag_c` post-step.
    #[test]
    fn device_carry_flag_matches_real_on_charter() {
        for case in charter_cases() {
            let (cpu, mut sim, plan) = build_charter_plan(case.source);
            let mut checked = 0u64;
            let mut cycles = 0u64;
            while !cpu.is_halted() && cycles < case.max_cycles {
                let carry_before = cpu.read_flag_c(&sim);
                let device_c = cpu.latch_operands().and_then(|(a, b, op)| {
                    if !is_alu_class(op) {
                        return None;
                    }
                    let result = plan.device_alu_result(&sim, a, b);
                    device_carry(op, a, result, carry_before)
                });
                cpu.step(&mut sim);
                if let Some(c) = device_c {
                    assert_eq!(
                        c,
                        cpu.read_flag_c(&sim),
                        "{}: cycle {cycles} carry device {c} != real {}",
                        case.name,
                        cpu.read_flag_c(&sim)
                    );
                    checked += 1;
                }
                cycles += 1;
            }
            assert!(cpu.is_halted(), "{}: did not halt", case.name);
            assert!(checked > 0, "{}: no ALU cycles validated", case.name);
        }
    }
}
