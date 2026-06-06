//! TileCpu Execution Engine
//!
//! This module contains the core execution logic that runs the CPU by
//! combining tile simulation with software decode/execute/writeback.
//!
//! The fetch stage uses tiles (ROM + Mux8to1 + PC). Operand selection uses
//! 4:1 binary mux trees with individual Const data inputs. ALU operations
//! (Add/Sub/And/Or/Xor/Not/Shl/Shr) execute through tiles in a compressed
//! vertical stack. ALU result selection uses an 8:1 binary mux tree.
//! Decode and writeback are handled in software.

use std::cell::Cell;

use crate::simulation::{Simulation, TimingStats};
use crate::tile_cpu::NUM_REGISTERS;
use crate::tile_cpu::wiring::PhysicalCpuIndices;

/// Metrics collected during CPU execution
#[derive(Debug, Clone, Default)]
pub struct TileCpuMetrics {
    /// Number of clock cycles executed
    pub cycles: u64,
    /// Total propagation deltas across all cycles
    pub total_deltas: u64,
    /// Maximum critical path seen in any cycle
    pub max_critical_path: u32,
    /// Average critical path across cycles
    pub avg_critical_path: f64,
    /// Estimated maximum frequency in MHz (1000 / max_critical_path)
    pub estimated_max_mhz: f64,
    /// Instructions executed (may differ from cycles for multi-cycle ops)
    pub instructions_executed: u64,
    /// IPC (instructions per cycle)
    pub ipc: f64,
    /// Whether any cycle failed to converge
    pub had_timing_violation: bool,
    /// Total tiles evaluated across all cycles (both clock edges)
    pub total_tiles_evaluated: u64,
    /// Total tiles whose output changed across all cycles
    pub total_tiles_switched: u64,
}

impl std::fmt::Display for TileCpuMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TileCPU Execution Metrics")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Cycles executed:     {:>12}", self.cycles)?;
        writeln!(f, "Instructions:        {:>12}", self.instructions_executed)?;
        writeln!(f, "IPC:                 {:>12.3}", self.ipc)?;
        writeln!(
            f,
            "Max critical path:   {:>12} deltas",
            self.max_critical_path
        )?;
        writeln!(
            f,
            "Avg critical path:   {:>12.1} deltas",
            self.avg_critical_path
        )?;
        writeln!(
            f,
            "Est. max frequency:  {:>12.1} MHz",
            self.estimated_max_mhz
        )?;
        writeln!(f, "Tiles evaluated:     {:>12}", self.total_tiles_evaluated)?;
        writeln!(f, "Tiles switched:      {:>12}", self.total_tiles_switched)?;
        if self.had_timing_violation {
            writeln!(f, "WARNING: Timing violations detected!")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ExtendedExec {
    reg_write: bool,
    rd: usize,
    value: u64,
    update_z: bool,
    update_c: bool,
    carry: bool,
    halt: bool,
    is_call: bool,
    call_target: u8,
    is_ret: bool,
    set_lr: bool,
    lr_value: u64,
}

fn decode_extended_exec(registers: [u64; NUM_REGISTERS], lr: u64, instruction: u8) -> ExtendedExec {
    let ext_opcode = (instruction >> 4) & 0x0F;
    let rd = ((instruction >> 2) & 0x03) as usize;
    let rs = (instruction & 0x03) as usize;
    let imm = instruction & 0x03;
    let current = registers[rd] as u8;

    match ext_opcode {
        // EXT-NOP
        0x0 => ExtendedExec::default(),
        // HALT
        0x1 => ExtendedExec {
            halt: true,
            ..ExtendedExec::default()
        },
        // INC Rd
        0x2 => {
            let (value, carry) = current.overflowing_add(1);
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: true,
                carry,
                ..ExtendedExec::default()
            }
        }
        // DEC Rd
        0x3 => {
            let (value, carry) = current.overflowing_sub(1);
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: true,
                carry,
                ..ExtendedExec::default()
            }
        }
        // NOT Rd
        0x4 => {
            let value = !current;
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: false,
                carry: false,
                ..ExtendedExec::default()
            }
        }
        // NEG Rd (0 - Rd)
        0x5 => {
            let value = 0u8.wrapping_sub(current) as u64;
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: true,
                carry: current != 0,
                ..ExtendedExec::default()
            }
        }
        // ADDI Rd, #imm(2-bit)
        0x6 => {
            let (value, carry) = current.overflowing_add(imm);
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: true,
                carry,
                ..ExtendedExec::default()
            }
        }
        // SUBI Rd, #imm(2-bit)
        0x7 => {
            let (value, carry) = current.overflowing_sub(imm);
            ExtendedExec {
                reg_write: true,
                rd,
                value: value as u64,
                update_z: true,
                update_c: true,
                carry,
                ..ExtendedExec::default()
            }
        }
        // CALL addr6
        0x8 => ExtendedExec {
            is_call: true,
            call_target: instruction & 0x3F,
            ..ExtendedExec::default()
        },
        // RET
        0x9 => ExtendedExec {
            is_ret: true,
            ..ExtendedExec::default()
        },
        // MFLR Rd
        0xA => ExtendedExec {
            reg_write: true,
            rd,
            value: lr,
            ..ExtendedExec::default()
        },
        // MTLR Rs
        0xB => ExtendedExec {
            set_lr: true,
            lr_value: registers[rs],
            ..ExtendedExec::default()
        },
        // Reserved extended opcodes behave as EXT-NOP.
        _ => ExtendedExec::default(),
    }
}

/// A CPU that executes via tile simulation
///
/// This struct holds tile indices for key CPU components. Fetch happens
/// via tile propagation; register reads use Mux4to1 tiles; ALU ops execute
/// through tiles; decode/writeback happen in software.
#[derive(Debug, Clone)]
pub struct TileCpu {
    /// Origin position on the tile grid
    pub origin: (usize, usize),

    /// Tile index of the Program Counter
    pc_idx: usize,

    /// Tile index of the instruction register (Mux8to1 output)
    pub(crate) ir_idx: usize,

    /// Tile indices of the 4 general-purpose registers
    reg_indices: [usize; NUM_REGISTERS],

    /// Tile indices of the ROM words (Const tiles)
    rom_indices: Vec<usize>,

    /// Tile indices of the RAM bytes (Ram tiles)
    ram_indices: Vec<usize>,

    /// Tile index of the Zero flag (Register8)
    flag_z_idx: usize,

    /// Tile index of the Carry flag (Register8)
    flag_c_idx: usize,

    // Flag circuit tiles (Phase 6)
    /// Const tile: result value fed to Zero tile
    flag_z_result_idx: usize,
    /// Const tile: update_flags signal above Mux_Z
    flag_z_update_idx: usize,
    /// Zero tile that computes z_raw
    flag_z_zero_idx: usize,
    /// Mux tile for Z flag writeback
    flag_z_mux_idx: usize,
    /// Const tile: carry value fed to Mux_C
    flag_c_carry_idx: usize,
    /// Const tile: update_flags signal above Mux_C
    flag_c_update_idx: usize,
    /// Mux tile for C flag writeback
    flag_c_mux_idx: usize,

    /// Tile indices of ALU operation tiles [Add, Sub, And, Or, Xor, Not, Shl, Shr]
    alu_tile_indices: [usize; 8],

    // Operand A mux tree (4:1 binary tree)
    op_a_data_indices: [usize; 4],
    op_a_sel0_indices: [usize; 2],
    op_a_sel1_idx: usize,
    op_a_leaf_indices: [usize; 2],
    #[allow(dead_code)]
    op_a_root_idx: usize,

    // Operand B mux tree (4:1 binary tree)
    op_b_data_indices: [usize; 4],
    op_b_sel0_indices: [usize; 2],
    op_b_sel1_idx: usize,
    op_b_leaf_indices: [usize; 2],
    #[allow(dead_code)]
    op_b_root_idx: usize,

    // PC circuit (Phase 4)
    /// Const tile for next_pc — software writes next PC value here before tick
    next_pc_const_idx: usize,

    // Decoder tiles (Phase 2)
    /// Const tile for opcode_lo select
    decoder_opcode_lo_idx: usize,
    /// Mux8to1 LUTs for ctrl_a (lo=opcodes 0-7, hi=opcodes 8-F)
    decoder_ctrl_a_lo_idx: usize,
    decoder_ctrl_a_hi_idx: usize,
    /// Mux8to1 LUTs for ctrl_b (lo=opcodes 0-7, hi=opcodes 8-F)
    decoder_ctrl_b_lo_idx: usize,
    decoder_ctrl_b_hi_idx: usize,

    // ALU result mux tree (8:1 binary tree)
    alu_result_data_indices: [usize; 8],
    alu_result_sel0_indices: [usize; 4],
    alu_result_sel1_indices: [usize; 2],
    alu_result_sel2_idx: usize,
    alu_result_leaf_indices: [usize; 4],
    #[allow(dead_code)]
    alu_result_mid_indices: [usize; 2],
    alu_result_root_idx: usize,

    // Register writeback (Phase 5)
    /// Const tiles for write-enable per register (above each Mux)
    reg_we_indices: [usize; NUM_REGISTERS],
    /// Const tiles for result data per register (left of each Mux)
    reg_result_indices: [usize; NUM_REGISTERS],
    /// Mux tiles for writeback selection per register
    reg_mux_indices: [usize; NUM_REGISTERS],

    /// Grid width (for index calculations)
    #[allow(dead_code)]
    grid_width: usize,

    /// Total tiles used by this CPU
    pub tile_count: usize,
    /// EXT-prefix state: previous instruction was prefix marker (opcode 0x0)
    prev_prefix: Cell<bool>,
    /// Explicit halted state (EXT HALT)
    halted: Cell<bool>,
    /// Link register for EXT CALL/RET flow
    lr: Cell<u64>,
}

impl TileCpu {
    /// Create a TileCpu from pre-placed tile indices
    ///
    /// This is called by TileCpuBuilder after placing all tiles.
    pub(crate) fn new(
        origin: (usize, usize),
        pc_idx: usize,
        ir_idx: usize,
        reg_indices: [usize; NUM_REGISTERS],
        rom_indices: Vec<usize>,
        ram_indices: Vec<usize>,
        flag_z_idx: usize,
        flag_c_idx: usize,
        flag_z_result_idx: usize,
        flag_z_update_idx: usize,
        flag_z_zero_idx: usize,
        flag_z_mux_idx: usize,
        flag_c_carry_idx: usize,
        flag_c_update_idx: usize,
        flag_c_mux_idx: usize,
        alu_tile_indices: [usize; 8],
        op_a_data_indices: [usize; 4],
        op_a_sel0_indices: [usize; 2],
        op_a_sel1_idx: usize,
        op_a_leaf_indices: [usize; 2],
        op_a_root_idx: usize,
        op_b_data_indices: [usize; 4],
        op_b_sel0_indices: [usize; 2],
        op_b_sel1_idx: usize,
        op_b_leaf_indices: [usize; 2],
        op_b_root_idx: usize,
        next_pc_const_idx: usize,
        decoder_opcode_lo_idx: usize,
        decoder_ctrl_a_lo_idx: usize,
        decoder_ctrl_a_hi_idx: usize,
        decoder_ctrl_b_lo_idx: usize,
        decoder_ctrl_b_hi_idx: usize,
        alu_result_data_indices: [usize; 8],
        alu_result_sel0_indices: [usize; 4],
        alu_result_sel1_indices: [usize; 2],
        alu_result_sel2_idx: usize,
        alu_result_leaf_indices: [usize; 4],
        alu_result_mid_indices: [usize; 2],
        alu_result_root_idx: usize,
        reg_we_indices: [usize; NUM_REGISTERS],
        reg_result_indices: [usize; NUM_REGISTERS],
        reg_mux_indices: [usize; NUM_REGISTERS],
        grid_width: usize,
        tile_count: usize,
    ) -> Self {
        Self {
            origin,
            pc_idx,
            ir_idx,
            reg_indices,
            rom_indices,
            ram_indices,
            flag_z_idx,
            flag_c_idx,
            flag_z_result_idx,
            flag_z_update_idx,
            flag_z_zero_idx,
            flag_z_mux_idx,
            flag_c_carry_idx,
            flag_c_update_idx,
            flag_c_mux_idx,
            alu_tile_indices,
            op_a_data_indices,
            op_a_sel0_indices,
            op_a_sel1_idx,
            op_a_leaf_indices,
            op_a_root_idx,
            op_b_data_indices,
            op_b_sel0_indices,
            op_b_sel1_idx,
            op_b_leaf_indices,
            op_b_root_idx,
            next_pc_const_idx,
            decoder_opcode_lo_idx,
            decoder_ctrl_a_lo_idx,
            decoder_ctrl_a_hi_idx,
            decoder_ctrl_b_lo_idx,
            decoder_ctrl_b_hi_idx,
            alu_result_data_indices,
            alu_result_sel0_indices,
            alu_result_sel1_indices,
            alu_result_sel2_idx,
            alu_result_leaf_indices,
            alu_result_mid_indices,
            alu_result_root_idx,
            reg_we_indices,
            reg_result_indices,
            reg_mux_indices,
            grid_width,
            tile_count,
            prev_prefix: Cell::new(false),
            halted: Cell::new(false),
            lr: Cell::new(0),
        }
    }

    /// Run the tile-based decoder: write opcode_lo to select Const, propagate,
    /// then read ctrl_a/ctrl_b from the appropriate LUT bank.
    fn decode_via_tiles(&self, sim: &mut Simulation, instruction: u8) -> (u8, u8) {
        let opcode = (instruction >> 4) & 0x0F;
        let opcode_lo = opcode & 0x07;

        // Write opcode_lo to the decoder select Const
        sim.set_logic_value_by_idx(self.decoder_opcode_lo_idx, opcode_lo as u64);
        // Mark WireDown tiles dirty (they cascade opcode_lo from Const to Mux8to1s).
        // WireDown tiles are at decoder_opcode_lo_idx + grid_width * (1..=4).
        for i in 1..=4 {
            sim.dirty
                .mark_dirty(self.decoder_opcode_lo_idx + self.grid_width * i);
        }
        sim.propagate_combinational();

        // Read from the correct bank based on opcode bit 3
        let ctrl_a = if opcode >= 8 {
            sim.get_logic_value_by_idx(self.decoder_ctrl_a_hi_idx) as u8
        } else {
            sim.get_logic_value_by_idx(self.decoder_ctrl_a_lo_idx) as u8
        };
        let ctrl_b = if opcode >= 8 {
            sim.get_logic_value_by_idx(self.decoder_ctrl_b_hi_idx) as u8
        } else {
            sim.get_logic_value_by_idx(self.decoder_ctrl_b_lo_idx) as u8
        };

        (ctrl_a, ctrl_b)
    }

    /// Execute one clock cycle
    ///
    /// This method:
    /// 1. Reads the instruction from the Mux8to1 tile (settled from previous tick)
    /// 2. Decodes via tile-based LUT decoder → ctrl_a/ctrl_b control bytes
    /// 3. Executes ALU, writeback, flags, branches, memory using ctrl_a/ctrl_b
    /// 4. Advances the clock (Register8 captures, Mux8to1 fetches next instruction)
    ///
    /// Returns timing statistics for this cycle.
    pub fn tick(&self, sim: &mut Simulation) -> TimingStats {
        if self.halted.get() {
            return TimingStats::default();
        }

        // 1. Read the instruction that was settled during the previous tick (or build)
        let instruction = sim.get_logic_value_by_idx(self.ir_idx);
        let opcode = (instruction >> 4) & 0x0F;
        let mut rd = ((instruction >> 2) & 3) as usize;
        let rs = (instruction & 3) as usize;
        let immediate = (instruction & 3) as u64;
        let jump_target = instruction & 0x3F;

        // EXT prefix mode:
        // - opcode 0x0 acts as prefix marker (consumed tick, no side effects)
        // - next instruction executes through software-extended decode path
        let extended_mode = self.prev_prefix.replace(false);
        let prefix_tick = !extended_mode && opcode == 0x0;
        if prefix_tick {
            self.prev_prefix.set(true);
        }

        let mut ext = ExtendedExec::default();
        if extended_mode {
            let regs = [
                self.read_reg(sim, 0),
                self.read_reg(sim, 1),
                self.read_reg(sim, 2),
                self.read_reg(sim, 3),
            ];
            ext = decode_extended_exec(regs, self.lr.get(), instruction as u8);
            rd = ext.rd;
        }

        // 2. Decode via tile-based LUT decoder
        let (ctrl_a, ctrl_b) = self.decode_via_tiles(sim, instruction as u8);

        // Extract control signals from tile decoder output
        let mut alu_sel = (ctrl_a & 0x07) as usize;
        let mut reg_write_en = (ctrl_a >> 3) & 1 != 0;
        let mut update_flags = (ctrl_a >> 4) & 1 != 0;
        let mut use_immediate = (ctrl_a >> 5) & 1 != 0;
        let mut is_jmp = ctrl_b & 1 != 0;
        let mut is_jz = (ctrl_b >> 1) & 1 != 0;
        let mut is_jnz = (ctrl_b >> 2) & 1 != 0;
        let mut mem_read = (ctrl_b >> 3) & 1 != 0;
        let mut mem_write = (ctrl_b >> 4) & 1 != 0;

        if extended_mode {
            // Extended instructions are software-decoded. Suppress base decode controls.
            alu_sel = 0;
            reg_write_en = ext.reg_write;
            update_flags = false;
            use_immediate = false;
            is_jmp = false;
            is_jz = false;
            is_jnz = false;
            mem_read = false;
            mem_write = false;
        } else if prefix_tick {
            // Prefix marker is a no-op tick with PC advance.
            alu_sel = 0;
            reg_write_en = false;
            update_flags = false;
            use_immediate = false;
            is_jmp = false;
            is_jz = false;
            is_jnz = false;
            mem_read = false;
            mem_write = false;
        }

        // 3. Reset all register write-enables to 0 (hold mode)
        for i in 0..NUM_REGISTERS {
            sim.set_logic_value_by_idx(self.reg_we_indices[i], 0);
        }

        // 4. Reset flag update signals to 0 (hold mode)
        sim.set_logic_value_by_idx(self.flag_z_update_idx, 0);
        sim.set_logic_value_by_idx(self.flag_c_update_idx, 0);

        // 5. Compute ALU result and carry if needed (for writeback or flags)
        let (result, carry) = if extended_mode {
            (ext.value, ext.carry)
        } else {
            let needs_compute = (reg_write_en && !mem_read) || (update_flags && !reg_write_en);
            if needs_compute {
                if use_immediate {
                    (immediate, false)
                } else if opcode == 0x2 {
                    // MOV: pass through source register (no tile ALU equivalent)
                    (self.read_reg(sim, rs), false)
                } else {
                    // Tile-based ALU: set operands on mux trees, propagate through ALU
                    self.write_operand_mux_inputs(sim, rd as u8, rs as u8);
                    sim.propagate_combinational();

                    self.write_alu_result_tree_inputs(sim, alu_sel);
                    sim.propagate_combinational();

                    // Read carry/borrow from physical tile bit 8 after ALU settles
                    let carry = match alu_sel {
                        0 => {
                            let addcarry_out = sim.get_logic_value_by_idx(self.alu_tile_indices[0]);
                            (addcarry_out >> 8) & 1 != 0
                        }
                        1 => {
                            let subborrow_out =
                                sim.get_logic_value_by_idx(self.alu_tile_indices[1]);
                            (subborrow_out >> 8) & 1 != 0
                        }
                        _ => false,
                    };

                    let result = sim.get_logic_value_by_idx(self.alu_result_root_idx);
                    (result, carry)
                }
            } else {
                (0, false)
            }
        };

        // 6. Register writeback via Mux->Register8
        if reg_write_en && !mem_read {
            for i in 0..NUM_REGISTERS {
                sim.set_logic_value_by_idx(self.reg_result_indices[i], result as u64);
            }
            sim.set_logic_value_by_idx(self.reg_we_indices[rd], 1);
        }

        // 7. Update flags via tile circuit (Z: Zero tile, C: software carry)
        if extended_mode {
            if ext.update_z {
                sim.set_logic_value_by_idx(self.flag_z_result_idx, result as u64);
                sim.set_logic_value_by_idx(self.flag_z_update_idx, u64::MAX);
            }
            if ext.update_c {
                let carry_val = if carry { u64::MAX } else { 0 };
                sim.set_logic_value_by_idx(self.flag_c_carry_idx, carry_val);
                sim.set_logic_value_by_idx(self.flag_c_update_idx, u64::MAX);
            }
        } else if update_flags {
            sim.set_logic_value_by_idx(self.flag_z_result_idx, result as u64);
            sim.set_logic_value_by_idx(self.flag_z_update_idx, u64::MAX);
            let carry_val = if carry { u64::MAX } else { 0 };
            sim.set_logic_value_by_idx(self.flag_c_carry_idx, carry_val);
            sim.set_logic_value_by_idx(self.flag_c_update_idx, u64::MAX);
        }

        // 8. Memory operations
        if mem_read {
            // LD Rd - read RAM[R0] into Rd via Mux->Register8
            let addr = self.read_reg(sim, 0) as usize;
            let value = self.read_ram(sim, addr);
            for i in 0..NUM_REGISTERS {
                sim.set_logic_value_by_idx(self.reg_result_indices[i], value as u64);
            }
            sim.set_logic_value_by_idx(self.reg_we_indices[rd], 1);
        }

        if mem_write {
            // ST Rs - write Rs to RAM[R0]
            let addr = self.read_reg(sim, 0) as usize;
            let value = self.read_reg(sim, rs);
            self.write_ram(sim, addr, value);
        }

        // 9. Propagate register and flag writeback Mux tiles
        // This MUST run for ALL instructions (including EXT-prefix ticks) to settle
        // the Mux->Register8 feedback loops before tick_with_delays.
        for i in 0..NUM_REGISTERS {
            sim.dirty.mark_dirty(self.reg_mux_indices[i]);
        }
        sim.dirty.mark_dirty(self.flag_z_zero_idx);
        sim.dirty.mark_dirty(self.flag_z_mux_idx);
        sim.dirty.mark_dirty(self.flag_c_mux_idx);
        sim.propagate_combinational();

        // 10. Branch handling - compute next PC
        let current_pc = self.read_pc(sim);
        let branch_taken = if extended_mode || prefix_tick {
            false
        } else {
            is_jmp || (is_jz && self.read_flag_z(sim)) || (is_jnz && !self.read_flag_z(sim))
        };
        if branch_taken {
            self.prev_prefix.set(false);
        }

        if extended_mode && ext.set_lr {
            self.lr.set(ext.lr_value as u64);
        }

        let next_pc = if extended_mode && ext.is_call {
            self.lr.set(current_pc.wrapping_add(1) as u64);
            self.prev_prefix.set(false);
            ext.call_target as u64
        } else if extended_mode && ext.is_ret {
            self.prev_prefix.set(false);
            self.lr.get()
        } else if branch_taken {
            jump_target as u64
        } else {
            current_pc.wrapping_add(1) as u64
        };

        if extended_mode && ext.halt {
            self.halted.set(true);
            self.prev_prefix.set(false);
        }

        // Write next_pc to Const_NPC tile - Register8 captures this at rising edge
        sim.set_logic_value_by_idx(self.next_pc_const_idx, next_pc as u64);

        // 11. Advance clock: rising edge -> Register8 captures next_pc + register values,
        //     WireDown+WireLeft propagate -> Mux8to1 fetches next instruction
        let stats_rise = sim.tick_with_delays();

        // 12. Falling edge completes the clock cycle
        let stats_fall = sim.tick_with_delays();

        // Combine both edges: critical path from rising, evaluations from both
        TimingStats {
            tiles_evaluated: stats_rise.tiles_evaluated + stats_fall.tiles_evaluated,
            tiles_switched: stats_rise.tiles_switched + stats_fall.tiles_switched,
            total_deltas: stats_rise.total_deltas + stats_fall.total_deltas,
            ..stats_rise
        }
    }
    fn write_operand_mux_inputs(&self, sim: &mut Simulation, src_a: u8, src_b: u8) {
        // Write register values to A tree data inputs
        for i in 0..NUM_REGISTERS {
            let val = sim.get_logic_value_by_idx(self.reg_indices[i]) & 0xFF;
            sim.set_logic_value_by_idx(self.op_a_data_indices[i], val);
            sim.set_logic_value_by_idx(self.op_b_data_indices[i], val);
        }

        // Set A tree selectors (S0=bit0, S1=bit1 of src_a)
        let a_s0 = if src_a & 1 != 0 { u64::MAX } else { 0 };
        let a_s1 = if src_a & 2 != 0 { u64::MAX } else { 0 };
        sim.set_logic_value_by_idx(self.op_a_sel0_indices[0], a_s0);
        sim.set_logic_value_by_idx(self.op_a_sel0_indices[1], a_s0);
        sim.set_logic_value_by_idx(self.op_a_sel1_idx, a_s1);

        // Set B tree selectors (S0=bit0, S1=bit1 of src_b)
        let b_s0 = if src_b & 1 != 0 { u64::MAX } else { 0 };
        let b_s1 = if src_b & 2 != 0 { u64::MAX } else { 0 };
        sim.set_logic_value_by_idx(self.op_b_sel0_indices[0], b_s0);
        sim.set_logic_value_by_idx(self.op_b_sel0_indices[1], b_s0);
        sim.set_logic_value_by_idx(self.op_b_sel1_idx, b_s1);

        // Mark leaf muxes dirty (both trees)
        sim.dirty.mark_dirty(self.op_a_leaf_indices[0]);
        sim.dirty.mark_dirty(self.op_a_leaf_indices[1]);
        sim.dirty.mark_dirty(self.op_b_leaf_indices[0]);
        sim.dirty.mark_dirty(self.op_b_leaf_indices[1]);

        // Mark root muxes dirty — S1 is written via set_logic_value_by_idx (no dirty
        // propagation). If leaf mux outputs don't change (data coincidentally same),
        // root never re-evaluates with the new S1 selector. Explicitly dirtying root
        // ensures it reads the updated S1 value from its up neighbor.
        sim.dirty.mark_dirty(self.op_a_root_idx);
        sim.dirty.mark_dirty(self.op_b_root_idx);
    }

    /// Write ALU tile outputs to the result tree data inputs and set selectors.
    ///
    /// Reads each ALU tile output individually, writes to Const data inputs,
    /// sets S0/S1/S2 selector bits based on alu_sel, marks leaf muxes dirty.
    fn write_alu_result_tree_inputs(&self, sim: &mut Simulation, alu_sel: usize) {
        // Write each ALU output to its individual Const data input
        for i in 0..8 {
            let val = sim.get_logic_value_by_idx(self.alu_tile_indices[i]) & 0xFF;
            sim.set_logic_value_by_idx(self.alu_result_data_indices[i], val);
        }

        // Set selector bits (S0=bit0, S1=bit1, S2=bit2 of alu_sel, 0 or u64::MAX)
        let s0 = if alu_sel & 1 != 0 { u64::MAX } else { 0 };
        let s1 = if alu_sel & 2 != 0 { u64::MAX } else { 0 };
        let s2 = if alu_sel & 4 != 0 { u64::MAX } else { 0 };

        for idx in &self.alu_result_sel0_indices {
            sim.set_logic_value_by_idx(*idx, s0);
        }
        for idx in &self.alu_result_sel1_indices {
            sim.set_logic_value_by_idx(*idx, s1);
        }
        sim.set_logic_value_by_idx(self.alu_result_sel2_idx, s2);

        // Mark leaf muxes dirty
        for idx in &self.alu_result_leaf_indices {
            sim.dirty.mark_dirty(*idx);
        }
        // Mark mid muxes dirty — S1 is written via set_logic_value_by_idx (no dirty
        // propagation). If leaf outputs don't change, mid muxes never re-evaluate
        // with the new S1 selector.
        for idx in &self.alu_result_mid_indices {
            sim.dirty.mark_dirty(*idx);
        }
        // Mark the WireDown below S2 dirty — it routes S2 to the root's up neighbor.
        // Without this, the WD keeps its old S2 value and the root selects the wrong subtree.
        sim.dirty
            .mark_dirty(self.alu_result_sel2_idx + self.grid_width);
    }

    /// Run for N cycles, collecting metrics
    pub fn run(&self, sim: &mut Simulation, max_cycles: u64) -> TileCpuMetrics {
        let mut metrics = TileCpuMetrics::default();
        let mut total_critical = 0u64;

        for cycle in 0..max_cycles {
            let stats = self.tick(sim);

            metrics.cycles = cycle + 1;
            metrics.total_deltas += stats.total_deltas as u64;
            metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
            metrics.total_tiles_switched += stats.tiles_switched as u64;
            total_critical += stats.critical_path_deltas as u64;

            if stats.critical_path_deltas > metrics.max_critical_path {
                metrics.max_critical_path = stats.critical_path_deltas;
            }

            if !stats.converged {
                metrics.had_timing_violation = true;
            }

            // Check for halt condition (PC unchanged after cycle)
            // In a real implementation, we'd check for a HALT instruction
            if self.is_halted(sim) {
                break;
            }
        }

        // Calculate derived metrics
        if metrics.cycles > 0 {
            metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
            metrics.estimated_max_mhz = if metrics.max_critical_path > 0 {
                1000.0 / metrics.max_critical_path as f64
            } else {
                f64::INFINITY
            };
        }

        // For now, assume 1 instruction per cycle (single-cycle design)
        metrics.instructions_executed = metrics.cycles;
        metrics.ipc = 1.0;

        metrics
    }

    /// Run until halted or max cycles reached
    pub fn run_until_halt(&self, sim: &mut Simulation, max_cycles: u64) -> TileCpuMetrics {
        self.run(sim, max_cycles)
    }

    /// Check if CPU is halted
    ///
    /// Currently checks if PC is pointing to a JMP to self (infinite loop).
    pub fn is_halted(&self, sim: &Simulation) -> bool {
        if self.halted.get() {
            return true;
        }
        let pc = self.read_pc(sim);
        if (pc as usize) < self.rom_indices.len() {
            let instruction = self.read_rom(sim, pc as usize);
            // Halt on infinite loop: JMP to self
            let opcode = (instruction >> 4) & 0x0F;
            if opcode == 0x0B {
                // JMP
                let target = instruction & 0x3F;
                return target == pc;
            }
        }
        false
    }

    // =========================================================================
    // State Accessors - Read tile state directly
    // =========================================================================

    /// Read current Program Counter value
    pub fn read_pc(&self, sim: &Simulation) -> u8 {
        sim.get_logic_value_by_idx(self.pc_idx) as u8
    }

    /// Read a register value (0-3)
    pub fn read_reg(&self, sim: &Simulation, reg: usize) -> u64 {
        if reg < NUM_REGISTERS {
            sim.get_logic_value_by_idx(self.reg_indices[reg])
        } else {
            0
        }
    }

    /// Read ROM at address
    pub fn read_rom(&self, sim: &Simulation, addr: usize) -> u8 {
        if addr < self.rom_indices.len() {
            sim.get_logic_value_by_idx(self.rom_indices[addr]) as u8
        } else {
            0
        }
    }

    /// Read RAM at address
    pub fn read_ram(&self, sim: &Simulation, addr: usize) -> u64 {
        if addr < self.ram_indices.len() {
            sim.get_logic_value_by_idx(self.ram_indices[addr])
        } else {
            0
        }
    }

    /// Read Zero flag
    pub fn read_flag_z(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_z_idx) != 0
    }

    /// Read Carry flag
    pub fn read_flag_c(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_c_idx) != 0
    }

    /// Read ALU tile output for a specific operation.
    ///
    /// Index: 0=Add, 1=Sub, 2=And, 3=Or, 4=Xor
    pub fn read_alu_out(&self, sim: &Simulation) -> u8 {
        // Return the Add tile output by default (for backwards compatibility)
        sim.get_logic_value_by_idx(self.alu_tile_indices[0]) as u8
    }

    /// Dump CPU state for debugging
    pub fn dump_state(&self, sim: &Simulation) -> String {
        format!(
            "PC={:02X} R0={:02X} R1={:02X} R2={:02X} R3={:02X} Z={} C={} ALU={:02X}",
            self.read_pc(sim),
            self.read_reg(sim, 0),
            self.read_reg(sim, 1),
            self.read_reg(sim, 2),
            self.read_reg(sim, 3),
            if self.read_flag_z(sim) { '1' } else { '0' },
            if self.read_flag_c(sim) { '1' } else { '0' },
            self.read_alu_out(sim),
        )
    }

    // =========================================================================
    // State Mutators - For initialization and testing
    // =========================================================================

    /// Set a register value (for initialization)
    pub fn write_reg(&self, sim: &mut Simulation, reg: usize, value: u64) {
        if reg < NUM_REGISTERS {
            sim.set_logic_value_by_idx(self.reg_indices[reg], value);
        }
    }

    /// Set PC value (for initialization or jumps)
    pub fn write_pc(&self, sim: &mut Simulation, value: u8) {
        sim.set_logic_value_by_idx(self.pc_idx, value as u64);
    }

    pub fn read_lr(&self) -> u64 {
        self.lr.get()
    }

    pub fn write_lr(&self, value: u64) {
        self.lr.set(value);
    }

    /// Write RAM at address
    pub fn write_ram(&self, sim: &mut Simulation, addr: usize, value: u64) {
        if addr < self.ram_indices.len() {
            let ram_idx = self.ram_indices[addr];
            sim.set_logic_value_by_idx(ram_idx, value as u64);

            // write_ram() is used for direct initialization in tests/benchmarks.
            // Since this bypasses the physical write circuit, explicitly dirty the
            // RAM read path so L1 ViaDown/Mux tree reflects the new value immediately.
            sim.dirty.mark_dirty(ram_idx);
            let layer_size = sim.tilemap.layer_size;
            let via_idx = ram_idx + layer_size;
            if via_idx < sim.tilemap.tiles.len() {
                sim.dirty.mark_dirty(via_idx);
            }
            sim.propagate_combinational_counted();
        }
    }

    /// Get the tile index for a register (for external analysis)
    pub fn get_reg_tile_idx(&self, reg: usize) -> Option<usize> {
        if reg < NUM_REGISTERS {
            Some(self.reg_indices[reg])
        } else {
            None
        }
    }

    /// Get the PC tile index
    pub fn get_pc_tile_idx(&self) -> usize {
        self.pc_idx
    }
}

// =============================================================================
// Physical CPU — tile-based decoder, software operand/writeback staging
// =============================================================================

/// A CPU with a fully physical decoder circuit.
///
/// Unlike `TileCpu` which writes opcode_lo to a Const tile and calls
/// `propagate_combinational()`, the physical CPU's decoder is wired as:
///   IR → Shr(IR, 4) → WD chain → 4 Mux8to1 LUTs
///
/// After `tick_with_delays()`, the LUT outputs already contain the correct
/// control bytes for the fetched instruction. Software reads the Shr output
/// (bit 3 for bank select) and LUT outputs (ctrl_a/ctrl_b) — no writes,
/// no propagation needed for decode.
///
/// Operand staging, ALU result routing, register writeback, and branch logic
/// still use software (same as `TileCpu`). These will be replaced with
/// physical circuits incrementally.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PhysicalCpu {
    /// Origin position on the tile grid
    pub origin: (usize, usize),

    // Core tile indices (same as TileCpu)
    pc_idx: usize,
    pub(crate) ir_idx: usize,
    reg_indices: [usize; NUM_REGISTERS],
    rom_indices: Vec<usize>,
    ram_indices: Vec<usize>,
    flag_z_idx: usize,
    flag_c_idx: usize,

    // Flag circuit tiles (Phase 3b: Const-based flag inputs, software-written)
    flag_z_result_idx: usize,
    flag_z_update_idx: usize,
    flag_z_zero_idx: usize,
    flag_z_mux_idx: usize,
    flag_c_carry_idx: usize,
    flag_c_update_idx: usize,
    flag_c_mux_idx: usize,

    // ALU tiles
    alu_tile_indices: [usize; 8],

    // Operand mux trees (Tree A fully physical, Tree B data pre-written)
    #[allow(dead_code)]
    op_a_data_indices: [usize; 4],
    #[allow(dead_code)]
    op_a_sel0_indices: [usize; 2],
    #[allow(dead_code)]
    op_a_sel1_idx: usize,
    #[allow(dead_code)]
    op_a_leaf_indices: [usize; 2],

    // Sprint 86: Tree B data is now physical (ViaUp from L1), pre_write_tree_b_data eliminated
    #[allow(dead_code)]
    op_b_data_indices: [usize; 4],
    #[allow(dead_code)]
    op_b_sel0_indices: [usize; 2],
    #[allow(dead_code)]
    op_b_sel1_idx: usize,
    #[allow(dead_code)]
    op_b_leaf_indices: [usize; 2],

    // PC circuit — ViaUp tile (physical PC update, no software writes)
    #[allow(dead_code)]
    next_pc_const_idx: usize,

    // Branch gate network output — read physically by Mux on L0
    #[allow(dead_code)]
    branch_taken_idx: usize,

    // Physical decoder tiles (replaces Const opcode_lo + manual propagation)
    /// Shr tile that computes IR >> 4. Read its output for opcode.
    #[allow(dead_code)]
    pub(crate) shr_opcode_idx: usize,
    /// Mux8to1 LUT indices [a_lo, a_hi, b_lo, b_hi]
    #[allow(dead_code)]
    pub(crate) decoder_lut_indices: [usize; 4],
    /// Physical ctrl_a bit extraction tile indices:
    /// [alu_sel0, alu_sel1, alu_sel2, reg_write_en, update_flags, use_immediate]
    ctrl_a_bit_indices: [usize; 6],

    /// Physical ctrl_b bit extraction tile indices:
    /// [is_jmp, is_jz, is_jnz, mem_read, mem_write]
    ctrl_b_bit_indices: [usize; 5],

    // ALU result mux tree (physical data via WD, software selectors via Const)
    #[allow(dead_code)]
    alu_result_data_indices: [usize; 8],
    alu_result_sel0_indices: [usize; 4],
    alu_result_sel1_indices: [usize; 2],
    alu_result_sel2_idx: usize,
    alu_result_leaf_indices: [usize; 4],
    alu_result_mid_indices: [usize; 2],
    alu_result_root_idx: usize,

    // Register writeback (WE is now physical — Sprint 84)
    // reg_result_indices are ViaUp tiles (Sprint 85) — NOT software-writable
    reg_result_indices: [usize; NUM_REGISTERS],
    reg_mux_indices: [usize; NUM_REGISTERS],

    // L1 merge Mux infrastructure (Sprint 85)
    // Selects between physical ALU result (default) and software LD data
    wb_merge_mux_l1_indices: [usize; NUM_REGISTERS],
    ld_data_l1_indices: [usize; NUM_REGISTERS],
    mem_read_l1_indices: [usize; NUM_REGISTERS],

    // Physical RAM (Sprint 83)
    /// Const tile for mem_write control signal (software-written before tick)
    mem_write_const_idx: usize,
    /// Const tile for write data (software-written before tick)
    write_data_const_idx: usize,

    grid_width: usize,
    pub tile_count: usize,
    /// L1 RAM read Mux tree root — outputs addressed Ram cell value (Sprint 87)
    ram_read_mux_root_idx: usize,

    // Sprint 87: conditional merge Mux reset
    // Only reset mem_read selectors when previous tick modified them.
    prev_injected: Cell<bool>,
    /// EXT-prefix state: previous instruction was prefix marker (opcode 0x0)
    prev_prefix: Cell<bool>,
    /// Explicit halted state (EXT HALT)
    halted: Cell<bool>,
    /// Link register for EXT CALL/RET flow
    lr: Cell<u64>,
}

impl PhysicalCpu {
    /// Create from PhysicalCpuIndices returned by wire_physical_cpu()
    pub(crate) fn from_wiring(origin: (usize, usize), phys: &PhysicalCpuIndices) -> Self {
        Self {
            origin,
            pc_idx: phys.pc_idx,
            ir_idx: phys.ir_idx,
            reg_indices: phys.reg_indices,
            rom_indices: phys.rom_indices.clone(),
            ram_indices: phys.ram_indices.clone(),
            flag_z_idx: phys.flag_z_idx,
            flag_c_idx: phys.flag_c_idx,
            flag_z_result_idx: phys.flag_z_result_idx,
            flag_z_update_idx: phys.flag_z_update_idx,
            flag_z_zero_idx: phys.flag_z_zero_idx,
            flag_z_mux_idx: phys.flag_z_mux_idx,
            flag_c_carry_idx: phys.flag_c_carry_idx,
            flag_c_update_idx: phys.flag_c_update_idx,
            flag_c_mux_idx: phys.flag_c_mux_idx,
            alu_tile_indices: phys.alu_tile_indices,
            op_a_data_indices: phys.op_a_data_indices,
            op_a_sel0_indices: phys.op_a_sel0_indices,
            op_a_sel1_idx: phys.op_a_sel1_idx,
            op_a_leaf_indices: phys.op_a_leaf_indices,
            op_b_data_indices: phys.op_b_data_indices,
            op_b_sel0_indices: phys.op_b_sel0_indices,
            op_b_sel1_idx: phys.op_b_sel1_idx,
            op_b_leaf_indices: phys.op_b_leaf_indices,
            next_pc_const_idx: phys.next_pc_const_idx,
            branch_taken_idx: phys.branch_taken_idx,
            shr_opcode_idx: phys.shr_opcode_idx,
            decoder_lut_indices: phys.decoder_lut_indices,
            ctrl_a_bit_indices: phys.ctrl_a_bits,
            ctrl_b_bit_indices: phys.ctrl_b_bits,
            alu_result_data_indices: phys.alu_result_data_indices,
            alu_result_sel0_indices: phys.alu_result_sel0_indices,
            alu_result_sel1_indices: phys.alu_result_sel1_indices,
            alu_result_sel2_idx: phys.alu_result_sel2_idx,
            alu_result_leaf_indices: phys.alu_result_leaf_indices,
            alu_result_mid_indices: phys.alu_result_mid_indices,
            alu_result_root_idx: phys.alu_result_root_idx,
            reg_result_indices: phys.reg_result_indices,
            reg_mux_indices: phys.reg_mux_indices,
            wb_merge_mux_l1_indices: phys.wb_merge_mux_l1_indices,
            ld_data_l1_indices: phys.ld_data_l1_indices,
            mem_read_l1_indices: phys.mem_read_l1_indices,
            mem_write_const_idx: phys.mem_write_const_idx,
            write_data_const_idx: phys.write_data_const_idx,
            grid_width: phys.grid_width,
            tile_count: phys.tile_count,
            ram_read_mux_root_idx: phys.ram_read_mux_root_idx,
            prev_injected: Cell::new(false),
            prev_prefix: Cell::new(false),
            halted: Cell::new(false),
            lr: Cell::new(0),
        }
    }

    /// Execute one clock cycle using physical decoder + software staging.
    ///
    /// The decoder has already settled (from the previous tick's `tick_with_delays`
    /// or from the builder's settling ticks). We read the decoded control signals
    /// directly — no Const writes, no `propagate_combinational()` for decode.
    ///
    /// Operand staging, ALU result routing, register writeback, flags, branches,
    /// and memory still use software (same as `TileCpu::tick()`).
    pub fn tick(&self, sim: &mut Simulation) -> TimingStats {
        if self.halted.get() {
            return TimingStats::default();
        }

        // 1. Read IR (already settled) and decode from physical decoder
        let pc_before = self.read_pc(sim);
        let instruction = sim.get_logic_value_by_idx(self.ir_idx) as u8;
        let opcode = (instruction >> 4) & 0x0F;
        let rs = (instruction & 3) as usize;
        let immediate = instruction & 3;
        // jump_target computed physically: And(IR, 0x3F) at (ox+61, oy+19) -> L1 routing -> Mux

        // EXT prefix mode:
        // - opcode 0x0 acts as prefix marker (consumed tick, no side effects)
        // - next instruction executes through software-extended decode path
        let extended_mode = self.prev_prefix.replace(false);
        let prefix_tick = !extended_mode && opcode == 0x0;
        if prefix_tick {
            self.prev_prefix.set(true);
        }

        let mut ext = ExtendedExec::default();
        let mut ext_regs_before = [0u64; NUM_REGISTERS];
        if extended_mode {
            ext_regs_before = [
                self.read_reg(sim, 0),
                self.read_reg(sim, 1),
                self.read_reg(sim, 2),
                self.read_reg(sim, 3),
            ];
            ext = decode_extended_exec(ext_regs_before, self.lr.get(), instruction);
        }

        // 2. Read control signals from physical BitSelect tiles (no writes needed)
        let alu_sel_bit0 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[0]) != 0;
        let alu_sel_bit1 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[1]) != 0;
        let alu_sel_bit2 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[2]) != 0;
        let mut alu_sel = (alu_sel_bit0 as usize)
            | ((alu_sel_bit1 as usize) << 1)
            | ((alu_sel_bit2 as usize) << 2);
        let mut reg_write_en = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[3]) != 0;
        let mut update_flags = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[4]) != 0;
        let mut use_immediate = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[5]) != 0;

        // ctrl_b bits: is_jmp/is_jz/is_jnz are consumed by the physical gate network
        let mut mem_read = sim.get_logic_value_by_idx(self.ctrl_b_bit_indices[3]) != 0;
        let mut mem_write = sim.get_logic_value_by_idx(self.ctrl_b_bit_indices[4]) != 0;

        if extended_mode {
            // Extended instructions are software-decoded. Suppress base decode controls.
            alu_sel = 0;
            reg_write_en = ext.reg_write;
            update_flags = false;
            use_immediate = false;
            mem_read = false;
            mem_write = false;
        } else if prefix_tick {
            // Prefix marker is a no-op tick with PC advance.
            alu_sel = 0;
            reg_write_en = false;
            update_flags = false;
            use_immediate = false;
            mem_read = false;
            mem_write = false;
        }

        // 3. Reset flag update signals to 0 (hold mode)
        sim.set_logic_value_by_idx(self.flag_z_update_idx, 0);
        sim.set_logic_value_by_idx(self.flag_c_update_idx, 0);

        // 4. Compute ALU result
        let (result, carry) = if extended_mode {
            (ext.value, ext.carry)
        } else {
            let needs_compute = (reg_write_en && !mem_read) || (update_flags && !reg_write_en);
            if needs_compute {
                if use_immediate {
                    (immediate as u64, false)
                } else if opcode == 0x2 {
                    (self.read_reg(sim, rs), false)
                } else {
                    // Read ALU result directly for software use.
                    let carry = match alu_sel {
                        0 => {
                            let addcarry_out = sim.get_logic_value_by_idx(self.alu_tile_indices[0]);
                            (addcarry_out >> 8) & 1 != 0
                        }
                        1 => {
                            let subborrow_out =
                                sim.get_logic_value_by_idx(self.alu_tile_indices[1]);
                            (subborrow_out >> 8) & 1 != 0
                        }
                        _ => false,
                    };

                    let result = sim.get_logic_value_by_idx(self.alu_tile_indices[alu_sel]);

                    (result, carry)
                }
            } else {
                (0, false)
            }
        };

        // 5. Register writeback - physical ALU bus + L1 merge Mux (Sprint 85)
        let needs_software_inject = if extended_mode {
            ext.reg_write
        } else {
            use_immediate || opcode == 0x2 || mem_read
        };

        // Reset L1 merge Mux selectors to MAX (= select ALU result, the default).
        // Sprint 87: only reset when the previous tick actually modified selectors.
        if self.prev_injected.get() {
            for i in 0..NUM_REGISTERS {
                sim.set_logic_value_by_idx(self.mem_read_l1_indices[i], u64::MAX);
                sim.dirty.mark_dirty(self.wb_merge_mux_l1_indices[i]);
            }
        }

        // For non-ALU register writes, inject result via L1 merge Mux
        if reg_write_en && needs_software_inject && !mem_read {
            for i in 0..NUM_REGISTERS {
                sim.set_logic_value_by_idx(self.ld_data_l1_indices[i], result as u64);
                sim.set_logic_value_by_idx(self.mem_read_l1_indices[i], 0);
                sim.dirty.mark_dirty(self.wb_merge_mux_l1_indices[i]);
            }
        }

        // 6. Update flags (software-written Consts for Z and C circuits)
        if extended_mode {
            if ext.update_z {
                let result8 = result as u64 & 0xFF;
                sim.set_logic_value_by_idx(self.flag_z_result_idx, result8);
                sim.set_logic_value_by_idx(self.flag_z_update_idx, u64::MAX);
            }
            if ext.update_c {
                let carry_val = if carry { u64::MAX } else { 0 };
                sim.set_logic_value_by_idx(self.flag_c_carry_idx, carry_val);
                sim.set_logic_value_by_idx(self.flag_c_update_idx, u64::MAX);
            }
        } else if update_flags {
            let result8 = result as u64 & 0xFF;
            sim.set_logic_value_by_idx(self.flag_z_result_idx, result8);
            sim.set_logic_value_by_idx(self.flag_z_update_idx, u64::MAX);
            let carry_val = if carry { u64::MAX } else { 0 };
            sim.set_logic_value_by_idx(self.flag_c_carry_idx, carry_val);
            sim.set_logic_value_by_idx(self.flag_c_update_idx, u64::MAX);
        }

        // 7. Memory operations (Sprint 83: physical RAM subsystem)
        // Pre-write RAM control Consts BEFORE tick_with_delays().
        sim.set_logic_value_by_idx(
            self.mem_write_const_idx,
            if mem_write { u64::MAX } else { 0 },
        );
        if mem_write {
            let value = self.read_reg(sim, rs);
            sim.set_logic_value_by_idx(self.write_data_const_idx, value as u64);
        }
        // Dirty downstream consumers so they pick up new Const values.
        sim.dirty.mark_dirty(self.mem_write_const_idx - 1);
        if mem_write {
            sim.dirty.mark_dirty(self.write_data_const_idx + 1);
        }

        // LD: software-assisted read via L1 merge Mux
        if mem_read {
            let value = sim.get_logic_value_by_idx(self.ram_read_mux_root_idx) as u8;
            for i in 0..NUM_REGISTERS {
                sim.set_logic_value_by_idx(self.ld_data_l1_indices[i], value as u64);
                sim.set_logic_value_by_idx(self.mem_read_l1_indices[i], 0);
                sim.dirty.mark_dirty(self.wb_merge_mux_l1_indices[i]);
            }
        }

        // Track whether this tick modified merge Mux selectors.
        self.prev_injected
            .set((reg_write_en && needs_software_inject) || mem_read);

        // 8. Propagate flag Mux tiles (software-written Consts need BFS settling)
        sim.dirty.mark_dirty(self.flag_z_zero_idx);
        sim.dirty.mark_dirty(self.flag_z_mux_idx);
        sim.dirty.mark_dirty(self.flag_c_mux_idx);
        let (prop1_deltas, prop1_eval, prop1_switched) = sim.propagate_combinational_counted();

        // EXT LR side effects and deferred PC override target.
        let mut pc_override: Option<u64> = None;
        if extended_mode {
            if ext.set_lr {
                self.lr.set(ext.lr_value as u64);
            }
            if ext.is_call {
                self.lr.set(pc_before.wrapping_add(1) as u64);
                pc_override = Some(ext.call_target as u64);
                self.prev_prefix.set(false);
            } else if ext.is_ret {
                pc_override = Some(self.lr.get());
                self.prev_prefix.set(false);
            }
        }

        // 9. Rising + falling edges
        let stats_rise = sim.tick_with_delays();
        let stats_fall = sim.tick_with_delays();

        // Extended-op correction pass:
        // - CALL/RET: override PC after sequential hardware update.
        // - All extended ops: restore or apply software-authored register values,
        //   neutralizing base-op side effects from physical decoder control lines.
        let (pc_fix_deltas, pc_fix_eval, pc_fix_switched) = if extended_mode {
            if ext.reg_write {
                sim.set_logic_value_by_idx(self.reg_indices[ext.rd], ext.value as u64);
            } else {
                for i in 0..NUM_REGISTERS {
                    sim.set_logic_value_by_idx(self.reg_indices[i], ext_regs_before[i] as u64);
                }
            }
            if let Some(next_pc) = pc_override {
                sim.set_logic_value_by_idx(self.pc_idx, next_pc as u64);
            }
            sim.dirty.mark_all_dirty(sim.tile_count());
            sim.propagate_combinational_counted()
        } else {
            (0, 0, 0)
        };

        // 10. Write ALU result tree selectors (S0/S1/S2) for NEXT instruction
        let (prop2_deltas, prop2_eval, prop2_switched) = {
            let next_alu_sel_bit0 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[0]) != 0;
            let next_alu_sel_bit1 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[1]) != 0;
            let next_alu_sel_bit2 = sim.get_logic_value_by_idx(self.ctrl_a_bit_indices[2]) != 0;
            let next_alu_sel = (next_alu_sel_bit0 as usize)
                | ((next_alu_sel_bit1 as usize) << 1)
                | ((next_alu_sel_bit2 as usize) << 2);

            let s0 = if next_alu_sel & 1 != 0 { u64::MAX } else { 0 };
            let s1 = if next_alu_sel & 2 != 0 { u64::MAX } else { 0 };
            let s2 = if next_alu_sel & 4 != 0 { u64::MAX } else { 0 };
            for idx in &self.alu_result_sel0_indices {
                sim.set_logic_value_by_idx(*idx, s0);
            }
            for idx in &self.alu_result_sel1_indices {
                sim.set_logic_value_by_idx(*idx, s1);
            }
            sim.set_logic_value_by_idx(self.alu_result_sel2_idx, s2);
            for idx in &self.alu_result_leaf_indices {
                sim.dirty.mark_dirty(*idx);
            }
            for idx in &self.alu_result_mid_indices {
                sim.dirty.mark_dirty(*idx);
            }
            sim.dirty
                .mark_dirty(self.alu_result_sel2_idx + self.grid_width);
            sim.dirty.mark_dirty(self.alu_result_root_idx);
            sim.propagate_combinational_counted()
        };

        // Branch safety: if physical PC took a non-sequential path, clear prefix state.
        let pc_after = self.read_pc(sim);
        if pc_after != pc_before.wrapping_add(1) {
            self.prev_prefix.set(false);
        }

        if extended_mode && ext.halt {
            self.halted.set(true);
            self.prev_prefix.set(false);
        }

        // Combine all work: tick_with_delays (rise+fall) + both propagate passes
        let prop_eval = prop1_eval + pc_fix_eval + prop2_eval;
        let prop_switched = prop1_switched + pc_fix_switched + prop2_switched;
        let prop_deltas = prop1_deltas + pc_fix_deltas + prop2_deltas;
        TimingStats {
            tiles_evaluated: stats_rise.tiles_evaluated + stats_fall.tiles_evaluated + prop_eval,
            tiles_switched: stats_rise.tiles_switched + stats_fall.tiles_switched + prop_switched,
            total_deltas: stats_rise.total_deltas + stats_fall.total_deltas + prop_deltas,
            ..stats_rise
        }
    }
    /// Pre-write Tree B data Consts with post-writeback register values.
    /// Sprint 86: Now dead — Tree B data is physical (ViaUp from L1 register routing).
    #[allow(dead_code)]
    fn pre_write_tree_b_data(&self, sim: &mut Simulation) {
        for i in 0..NUM_REGISTERS {
            let val = sim.get_logic_value_by_idx(self.reg_mux_indices[i]) & 0xFF;
            sim.set_logic_value_by_idx(self.op_b_data_indices[i], val);
        }
        sim.dirty.mark_dirty(self.op_b_leaf_indices[0]);
        sim.dirty.mark_dirty(self.op_b_leaf_indices[1]);
    }

    /// Run for N cycles, collecting metrics
    pub fn run(&self, sim: &mut Simulation, max_cycles: u64) -> TileCpuMetrics {
        let mut metrics = TileCpuMetrics::default();
        let mut total_critical = 0u64;

        for cycle in 0..max_cycles {
            let stats = self.tick(sim);

            metrics.cycles = cycle + 1;
            metrics.total_deltas += stats.total_deltas as u64;
            metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
            metrics.total_tiles_switched += stats.tiles_switched as u64;
            total_critical += stats.critical_path_deltas as u64;

            if stats.critical_path_deltas > metrics.max_critical_path {
                metrics.max_critical_path = stats.critical_path_deltas;
            }

            if !stats.converged {
                metrics.had_timing_violation = true;
            }

            if self.is_halted(sim) {
                break;
            }
        }

        if metrics.cycles > 0 {
            metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
            metrics.estimated_max_mhz = if metrics.max_critical_path > 0 {
                1000.0 / metrics.max_critical_path as f64
            } else {
                f64::INFINITY
            };
        }

        metrics.instructions_executed = metrics.cycles;
        metrics.ipc = 1.0;

        metrics
    }

    // =========================================================================
    // State Accessors
    // =========================================================================

    pub fn read_pc(&self, sim: &Simulation) -> u8 {
        sim.get_logic_value_by_idx(self.pc_idx) as u8
    }

    pub fn read_reg(&self, sim: &Simulation, reg: usize) -> u64 {
        if reg < NUM_REGISTERS {
            sim.get_logic_value_by_idx(self.reg_indices[reg])
        } else {
            0
        }
    }

    pub fn read_rom(&self, sim: &Simulation, addr: usize) -> u8 {
        if addr < self.rom_indices.len() {
            sim.get_logic_value_by_idx(self.rom_indices[addr]) as u8
        } else {
            0
        }
    }

    pub fn read_ram(&self, sim: &Simulation, addr: usize) -> u64 {
        if addr < self.ram_indices.len() {
            sim.get_logic_value_by_idx(self.ram_indices[addr])
        } else {
            0
        }
    }

    pub fn read_flag_z(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_z_idx) != 0
    }

    pub fn read_flag_c(&self, sim: &Simulation) -> bool {
        sim.get_logic_value_by_idx(self.flag_c_idx) != 0
    }

    pub fn write_reg(&self, sim: &mut Simulation, reg: usize, value: u64) {
        if reg < NUM_REGISTERS {
            sim.set_logic_value_by_idx(self.reg_indices[reg], value);
        }
    }

    pub fn write_pc(&self, sim: &mut Simulation, value: u8) {
        sim.set_logic_value_by_idx(self.pc_idx, value as u64);
    }

    pub fn read_lr(&self) -> u64 {
        self.lr.get()
    }

    pub fn write_lr(&self, value: u64) {
        self.lr.set(value);
    }

    pub fn write_ram(&self, sim: &mut Simulation, addr: usize, value: u64) {
        if addr < self.ram_indices.len() {
            let ram_idx = self.ram_indices[addr];
            sim.set_logic_value_by_idx(ram_idx, value as u64);

            // Direct RAM writes bypass the physical write circuit. Dirty and settle
            // the read tree entry point so L1 RAM read Mux sees updated values.
            sim.dirty.mark_dirty(ram_idx);
            let layer_size = sim.tilemap.layer_size;
            let via_idx = ram_idx + layer_size;
            if via_idx < sim.tilemap.tiles.len() {
                sim.dirty.mark_dirty(via_idx);
            }
            sim.propagate_combinational_counted();
        }
    }

    pub fn is_halted(&self, sim: &Simulation) -> bool {
        if self.halted.get() {
            return true;
        }
        let pc = self.read_pc(sim);
        if (pc as usize) < self.rom_indices.len() {
            let instruction = self.read_rom(sim, pc as usize);
            let opcode = (instruction >> 4) & 0x0F;
            if opcode == 0x0B {
                let target = instruction & 0x3F;
                return target == pc;
            }
        }
        false
    }

    pub fn dump_state(&self, sim: &Simulation) -> String {
        format!(
            "PC={:02X} R0={:02X} R1={:02X} R2={:02X} R3={:02X} Z={} C={}",
            self.read_pc(sim),
            self.read_reg(sim, 0),
            self.read_reg(sim, 1),
            self.read_reg(sim, 2),
            self.read_reg(sim, 3),
            if self.read_flag_z(sim) { '1' } else { '0' },
            if self.read_flag_c(sim) { '1' } else { '0' },
        )
    }
}

// =============================================================================
// Critical Path Analysis Integration
// =============================================================================

impl TileCpu {
    /// Analyze critical path for current instruction
    ///
    /// Returns the sequence of tile indices on the critical path.
    pub fn analyze_critical_path(&self, sim: &Simulation) -> Vec<usize> {
        sim.trace_critical_path()
    }

    /// Check if current timing meets target frequency
    pub fn check_timing(&self, sim: &Simulation, target_mhz: f64) -> bool {
        let target_deltas = (1000.0 / target_mhz) as u32;
        let result = sim.check_timing(target_deltas);
        result.meets_timing
    }
}
