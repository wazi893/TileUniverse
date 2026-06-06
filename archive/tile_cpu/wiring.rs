//! Datapath Wiring - Complete signal routing for TileCpu
//!
//! This module handles the detailed wiring of all CPU components.
//! Every signal path is implemented as actual tile connections.
//!
//! # Layout Overview
//!
//! ```text
//! Y=0  : [Clock] [PC_____] [IR_____] [Decode________]
//! Y=1  : [Wire ] [Wire   ] [Wire   ] [Control_______]
//! Y=2  : ================================================ Instruction Bus
//! Y=3  : [Reg0 Read Mux_____] [Reg1 Read Mux_____]
//! Y=4  : [R0___] [R1___] [R2___] [R3___]  Register File
//! Y=5  : [Reg Write Demux___________________________]
//! Y=6  : ================================================ Register Bus A
//! Y=7  : ================================================ Register Bus B
//! Y=8  : [Add__] [Sub__] [And__] [Or___] [Xor__] [Shl_] [Shr_]
//! Y=9  : [ALU Result Mux____________________________]
//! Y=10 : ================================================ ALU Result Bus
//! Y=11 : [Zero_] [Carry] [Branch Logic______________]
//! Y=12 : ================================================ Writeback Bus
//! Y=16+: [ROM________________________________________________]
//! Y=32+: [RAM________________________________________________]
//! ```

use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tile_cpu::{DATAPATH_WIDTH, NUM_REGISTERS};

/// Complete wiring context for building a CPU
pub struct WiringContext<'a> {
    sim: &'a mut Simulation,
    origin: (usize, usize),
    pub grid_width: usize,
    tiles_placed: usize,

    // Component positions (tile indices)
    pub pc_idx: usize,
    pub ir_idx: usize,
    pub reg_indices: [usize; NUM_REGISTERS],
    pub alu_add_idx: usize,
    pub alu_sub_idx: usize,
    pub alu_and_idx: usize,
    pub alu_or_idx: usize,
    pub alu_xor_idx: usize,
    pub alu_mux_idx: usize,
    pub flag_z_idx: usize,
    pub flag_c_idx: usize,
    pub rom_indices: Vec<usize>,
    pub ram_indices: Vec<usize>,

    // Bus positions (for routing)
    pub instr_bus_y: usize,
    pub reg_bus_a_y: usize,
    pub reg_bus_b_y: usize,
    pub alu_result_y: usize,
    pub writeback_y: usize,
}

impl<'a> WiringContext<'a> {
    pub fn new(sim: &'a mut Simulation, origin: (usize, usize)) -> Self {
        let grid_width = sim.width();
        let (ox, oy) = origin;

        Self {
            sim,
            origin,
            grid_width,
            tiles_placed: 0,

            // Will be filled during wiring
            pc_idx: 0,
            ir_idx: 0,
            reg_indices: [0; NUM_REGISTERS],
            alu_add_idx: 0,
            alu_sub_idx: 0,
            alu_and_idx: 0,
            alu_or_idx: 0,
            alu_xor_idx: 0,
            alu_mux_idx: 0,
            flag_z_idx: 0,
            flag_c_idx: 0,
            rom_indices: Vec::new(),
            ram_indices: Vec::new(),

            // Bus Y positions relative to origin
            instr_bus_y: oy + 2,
            reg_bus_a_y: oy + 6,
            reg_bus_b_y: oy + 7,
            alu_result_y: oy + 10,
            writeback_y: oy + 12,
        }
    }

    /// Place a tile and return its index
    fn place(&mut self, x: usize, y: usize, tile_type: TileType) -> usize {
        self.sim.set_tile(x, y, tile_type);
        self.tiles_placed += 1;
        y * self.grid_width + x
    }

    /// Place a tile with initial value
    fn place_with_value(&mut self, x: usize, y: usize, tile_type: TileType, value: u64) -> usize {
        let idx = self.place(x, y, tile_type);
        self.sim.set_logic_value(x, y, value);
        idx
    }

    /// Draw a horizontal wire from x1 to x2 at given y
    fn wire_h(&mut self, y: usize, x1: usize, x2: usize) {
        let (start, end) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        for x in start..=end {
            self.place(x, y, TileType::WireH);
        }
    }

    /// Draw a vertical wire from y1 to y2 at given x
    fn wire_v(&mut self, x: usize, y1: usize, y2: usize) {
        let (start, end) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        for y in start..=end {
            self.place(x, y, TileType::WireV);
        }
    }

    /// Place a cross tile (for bus intersections)
    fn cross(&mut self, x: usize, y: usize) -> usize {
        self.place(x, y, TileType::Cross)
    }

    /// Get absolute position from relative
    fn abs(&self, rel_x: usize, rel_y: usize) -> (usize, usize) {
        (self.origin.0 + rel_x, self.origin.1 + rel_y)
    }

    pub fn total_tiles(&self) -> usize {
        self.tiles_placed
    }
}

// =============================================================================
// Phase 1: Instruction Fetch
// =============================================================================

/// Wire the instruction fetch path: PC → ROM → IR
pub fn wire_instruction_fetch(ctx: &mut WiringContext, program: &[u8], rom_size: usize) {
    let (ox, oy) = ctx.origin;

    // -------------------------------------------------------------------------
    // Clock Generator (provides clock signal to all sequential elements)
    // -------------------------------------------------------------------------
    let clock_x = ox;
    let clock_y = oy;
    ctx.place(clock_x, clock_y, TileType::ClockGlobal);

    // -------------------------------------------------------------------------
    // Program Counter
    // -------------------------------------------------------------------------
    // PC increments on clock, or loads from branch target
    // Input: left = branch target, right = branch taken flag
    let pc_x = ox + 2;
    let pc_y = oy;
    ctx.pc_idx = ctx.place(pc_x, pc_y, TileType::ProgramCounter);
    ctx.sim.set_logic_value(pc_x, pc_y, 0); // Start at address 0

    // Wire clock to PC (vertical from clock)
    ctx.wire_v(clock_x, clock_y, pc_y);

    // -------------------------------------------------------------------------
    // ROM with Address Decoder
    // -------------------------------------------------------------------------
    // ROM is laid out as 8 columns (one per address bit group) × N rows
    let rom_base_y = oy + 16;

    // Create ROM entries
    for addr in 0..rom_size {
        let rom_x = ox + (addr % 8) * 2; // Spread out for routing
        let rom_y = rom_base_y + (addr / 8);

        let value = if addr < program.len() {
            program[addr] as u64
        } else {
            0 // NOP
        };

        let idx = ctx.place_with_value(rom_x, rom_y, TileType::Const, value);
        ctx.rom_indices.push(idx);
    }

    // ROM address decoder: takes PC value, outputs one-hot select
    // For simplicity, we use Mux8to1 to select the right ROM word
    let rom_mux_x = ox + 4;
    let rom_mux_y = rom_base_y - 1;
    ctx.place(rom_mux_x, rom_mux_y, TileType::Mux8to1);

    // Wire PC to ROM mux select (vertical path)
    ctx.wire_v(pc_x, pc_y + 1, rom_mux_y - 1);
    ctx.place(pc_x, rom_mux_y, TileType::Wire); // Connect to mux

    // -------------------------------------------------------------------------
    // Instruction Register
    // -------------------------------------------------------------------------
    // Captures instruction on clock edge
    let ir_x = ox + 8;
    let ir_y = oy;
    ctx.ir_idx = ctx.place(ir_x, ir_y, TileType::Register8);

    // Wire ROM output to IR input
    ctx.wire_h(oy + 1, rom_mux_x, ir_x);
    ctx.wire_v(rom_mux_x, rom_mux_y, oy + 1);
    ctx.wire_v(ir_x, oy + 1, ir_y);

    // Wire clock to IR
    ctx.wire_h(clock_y, clock_x, ir_x - 1);
    ctx.place(ir_x - 1, ir_y, TileType::Wire);
}

// =============================================================================
// Phase 2: Instruction Decode
// =============================================================================

/// Wire the instruction decode logic
pub fn wire_instruction_decode(ctx: &mut WiringContext) {
    let (ox, oy) = ctx.origin;
    let ir_x = ox + 8;

    // -------------------------------------------------------------------------
    // Opcode Extraction (bits 7-4)
    // We use Shr to extract upper nibble
    // -------------------------------------------------------------------------
    let opcode_x = ox + 12;
    let opcode_y = oy;

    // Shift right by 4 to get opcode
    ctx.place_with_value(opcode_x + 1, opcode_y, TileType::Const, 4); // Shift amount
    ctx.place(opcode_x, opcode_y, TileType::Shr);

    // Wire IR to shifter
    ctx.wire_h(oy, ir_x + 1, opcode_x - 1);

    // -------------------------------------------------------------------------
    // Opcode Decoder (4-to-16, using two 3-to-8 decoders)
    // -------------------------------------------------------------------------
    let decode_y = oy + 1;

    // Decoder for lower 3 bits of opcode
    ctx.place(opcode_x, decode_y, TileType::Decoder3to8);

    // Wire opcode to decoder
    ctx.wire_v(opcode_x, opcode_y + 1, decode_y - 1);

    // -------------------------------------------------------------------------
    // Register Select Extraction
    // Rd = bits 3-2, Rs = bits 1-0
    // -------------------------------------------------------------------------
    let rd_x = ox + 16;
    let rs_x = ox + 20;

    // Rd extraction: (IR >> 2) & 0x03
    ctx.place_with_value(rd_x + 1, oy, TileType::Const, 2);
    ctx.place(rd_x, oy, TileType::Shr);
    ctx.place_with_value(rd_x + 1, oy + 1, TileType::Const, 0x03);
    ctx.place(rd_x, oy + 1, TileType::And);

    // Wire IR to Rd extraction
    ctx.wire_h(oy, ir_x + 1, rd_x - 1);

    // Rs extraction: IR & 0x03
    ctx.place_with_value(rs_x + 1, oy, TileType::Const, 0x03);
    ctx.place(rs_x, oy, TileType::And);

    // Wire IR to Rs extraction
    ctx.wire_h(oy, ir_x + 1, rs_x - 1);

    // -------------------------------------------------------------------------
    // Register Address Decoders
    // -------------------------------------------------------------------------
    // Rd decoder
    ctx.place(rd_x, oy + 2, TileType::Decoder3to8);
    ctx.wire_v(rd_x, oy + 1, oy + 2);

    // Rs decoder
    ctx.place(rs_x, oy + 2, TileType::Decoder3to8);
    ctx.wire_v(rs_x, oy, oy + 2);
}

// =============================================================================
// Phase 3: Register File
// =============================================================================

/// Wire the register file with read and write ports
pub fn wire_register_file(ctx: &mut WiringContext, initial_regs: &[u8; NUM_REGISTERS]) {
    let (ox, oy) = ctx.origin;
    let reg_y = oy + 4;

    // -------------------------------------------------------------------------
    // 4 Registers using RegEnable tiles
    // -------------------------------------------------------------------------
    for reg in 0..NUM_REGISTERS {
        let reg_x = ox + 4 + reg * 4;

        // Register tile
        ctx.reg_indices[reg] = ctx.place_with_value(
            reg_x, reg_y,
            TileType::RegEnable,
            initial_regs[reg] as u64
        );

        // Write enable routing (from Rd decoder, active when opcode writes)
        // This needs AND with reg_write_enable signal
        ctx.place(reg_x, reg_y - 1, TileType::And);

        // Read port A mux input (connects to reg_bus_a)
        ctx.place(reg_x, reg_y + 1, TileType::Wire);
        ctx.wire_v(reg_x, reg_y + 1, ctx.reg_bus_a_y);

        // Read port B mux input (connects to reg_bus_b)
        ctx.place(reg_x + 1, reg_y + 1, TileType::Wire);
        ctx.wire_v(reg_x + 1, reg_y + 2, ctx.reg_bus_b_y);
    }

    // -------------------------------------------------------------------------
    // Read Mux A: Selects which register goes to ALU input A
    // -------------------------------------------------------------------------
    let mux_a_x = ox + 2;
    let mux_a_y = oy + 3;
    ctx.place(mux_a_x, mux_a_y, TileType::Mux8to1);

    // Wire all registers to mux A data input (horizontal bus)
    ctx.wire_h(ctx.reg_bus_a_y, ox, ox + 24);

    // Wire Rd select to mux A
    let rd_x = ox + 16;
    ctx.wire_h(oy + 2, rd_x, mux_a_x);
    ctx.wire_v(mux_a_x, oy + 2, mux_a_y);

    // -------------------------------------------------------------------------
    // Read Mux B: Selects which register goes to ALU input B
    // -------------------------------------------------------------------------
    let mux_b_x = ox + 22;
    let mux_b_y = oy + 3;
    ctx.place(mux_b_x, mux_b_y, TileType::Mux8to1);

    // Wire all registers to mux B data input
    ctx.wire_h(ctx.reg_bus_b_y, ox, ox + 24);

    // Wire Rs select to mux B
    let rs_x = ox + 20;
    ctx.wire_h(oy + 2, rs_x, mux_b_x);
    ctx.wire_v(mux_b_x, oy + 2, mux_b_y);

    // -------------------------------------------------------------------------
    // Immediate Mux: Selects between Rs value or immediate from instruction
    // For LDI instruction, lower 2 bits of instruction are the immediate
    // -------------------------------------------------------------------------
    let imm_mux_x = ox + 24;
    let imm_mux_y = mux_b_y;
    ctx.place(imm_mux_x, imm_mux_y, TileType::Mux);

    // Wire mux B output to immediate mux left input
    ctx.wire_h(mux_b_y, mux_b_x + 1, imm_mux_x - 1);

    // Wire immediate value (lower 2 bits of IR) to right input
    let ir_x = ox + 8;
    // Extract lower 2 bits: IR & 0x03
    ctx.place_with_value(ir_x + 4, oy + 1, TileType::Const, 0x03);
    ctx.place(ir_x + 3, oy + 1, TileType::And);
    ctx.wire_v(ir_x + 3, oy + 1, imm_mux_y);
    ctx.wire_h(imm_mux_y, ir_x + 3, imm_mux_x);
}

// =============================================================================
// Phase 4: ALU
// =============================================================================

/// Wire the ALU with all operations
pub fn wire_alu(ctx: &mut WiringContext) {
    let (ox, oy) = ctx.origin;
    let alu_y = oy + 8;

    // -------------------------------------------------------------------------
    // ALU Input Routing
    // -------------------------------------------------------------------------
    // A input comes from read mux A (or forwarded from previous result)
    let a_bus_x = ox + 2;
    ctx.wire_v(a_bus_x, oy + 3, alu_y);

    // B input comes from immediate mux output
    let b_bus_x = ox + 24;
    ctx.wire_v(b_bus_x, oy + 3, alu_y);

    // Horizontal bus to distribute A and B to all ALU ops
    ctx.wire_h(alu_y - 1, a_bus_x, ox + 20);

    // -------------------------------------------------------------------------
    // ALU Operations
    // Each operation tile takes A (left) and B (right) inputs
    // -------------------------------------------------------------------------
    let ops = [
        (ox + 4, TileType::Add),
        (ox + 6, TileType::Sub),
        (ox + 8, TileType::And),
        (ox + 10, TileType::Or),
        (ox + 12, TileType::Xor),
        (ox + 14, TileType::Shl),
        (ox + 16, TileType::Shr),
    ];

    for (op_x, op_type) in ops.iter() {
        ctx.place(*op_x, alu_y, *op_type);

        // Wire A input (from vertical bus)
        ctx.cross(*op_x, alu_y - 1);

        // Wire B input (horizontal from B bus)
        ctx.place(*op_x + 1, alu_y, TileType::WireH);
    }

    // Store ALU tile indices
    ctx.alu_add_idx = alu_y * ctx.grid_width + ox + 4;
    ctx.alu_sub_idx = alu_y * ctx.grid_width + ox + 6;
    ctx.alu_and_idx = alu_y * ctx.grid_width + ox + 8;
    ctx.alu_or_idx = alu_y * ctx.grid_width + ox + 10;
    ctx.alu_xor_idx = alu_y * ctx.grid_width + ox + 12;

    // -------------------------------------------------------------------------
    // ALU Result Mux
    // Selects which operation result to use based on opcode
    // -------------------------------------------------------------------------
    let alu_mux_x = ox + 10;
    let alu_mux_y = alu_y + 1;
    ctx.alu_mux_idx = ctx.place(alu_mux_x, alu_mux_y, TileType::Mux8to1);

    // Wire all ALU outputs to mux data input
    for (op_x, _) in ops.iter() {
        ctx.wire_v(*op_x, alu_y + 1, alu_mux_y);
    }

    // Wire opcode (lower 3 bits) to mux select
    let opcode_x = ox + 12;
    ctx.wire_h(alu_mux_y + 1, opcode_x, alu_mux_x);

    // -------------------------------------------------------------------------
    // ALU Result Bus
    // -------------------------------------------------------------------------
    ctx.wire_h(ctx.alu_result_y, ox, ox + 24);
    ctx.wire_v(alu_mux_x, alu_mux_y + 1, ctx.alu_result_y);
}

// =============================================================================
// Phase 5: Flags
// =============================================================================

/// Wire the flag generation logic
pub fn wire_flags(ctx: &mut WiringContext) {
    let (ox, oy) = ctx.origin;
    let flag_y = oy + 11;

    // -------------------------------------------------------------------------
    // Zero Flag
    // Z = (ALU result == 0)
    // -------------------------------------------------------------------------
    let zero_x = ox + 4;
    ctx.flag_z_idx = ctx.place(zero_x, flag_y, TileType::Zero);

    // Wire ALU result to Zero detector
    ctx.wire_v(ox + 10, ctx.alu_result_y, flag_y);
    ctx.wire_h(flag_y, ox + 10, zero_x);

    // -------------------------------------------------------------------------
    // Carry Flag (captured in a latch)
    // For ADD: carry = (A + B) > 255
    // We use Lt tile: NOT(result >= A) when B > 0
    // Simplified: use upper bit of 9-bit addition
    // -------------------------------------------------------------------------
    let carry_x = ox + 8;
    ctx.flag_c_idx = ctx.place(carry_x, flag_y, TileType::Latch);

    // For now, carry detection is simplified - full implementation would
    // need a 9-bit adder or dedicated carry logic

    // -------------------------------------------------------------------------
    // Flag Register (captures flags on clock)
    // -------------------------------------------------------------------------
    let flag_reg_x = ox + 12;
    ctx.place(flag_reg_x, flag_y, TileType::Register8);

    // Wire flags to flag register
    ctx.wire_h(flag_y, zero_x, flag_reg_x);
}

// =============================================================================
// Phase 6: Branch Logic
// =============================================================================

/// Wire the branch logic and PC update
pub fn wire_branch_logic(ctx: &mut WiringContext) {
    let (ox, oy) = ctx.origin;
    let branch_y = oy + 11;

    // -------------------------------------------------------------------------
    // Branch Condition Logic
    // JMP: always take branch
    // JZ:  take if Z flag set
    // JNZ: take if Z flag clear
    // -------------------------------------------------------------------------
    let branch_x = ox + 16;

    // JZ condition: opcode==0xC AND Z
    ctx.place(branch_x, branch_y, TileType::And);

    // JNZ condition: opcode==0xD AND NOT(Z)
    ctx.place(branch_x + 2, branch_y, TileType::Not);
    ctx.place(branch_x + 4, branch_y, TileType::And);

    // Unconditional JMP: opcode==0xB
    ctx.place(branch_x + 6, branch_y, TileType::Wire);

    // OR all branch conditions together
    ctx.place(branch_x + 8, branch_y, TileType::Or);
    ctx.place(branch_x + 10, branch_y, TileType::Or);

    // -------------------------------------------------------------------------
    // Branch Target
    // Lower 6 bits of instruction for branch instructions
    // -------------------------------------------------------------------------
    let target_x = ox + 20;
    ctx.place_with_value(target_x + 1, branch_y, TileType::Const, 0x3F);
    ctx.place(target_x, branch_y, TileType::And);

    // Wire IR to target extraction
    let ir_x = ox + 8;
    ctx.wire_h(branch_y - 1, ir_x, target_x);
    ctx.wire_v(ir_x, oy, branch_y - 1);

    // -------------------------------------------------------------------------
    // PC Update Mux
    // Selects between PC+1 and branch target
    // -------------------------------------------------------------------------
    let pc_mux_x = ox + 1;
    let pc_mux_y = oy;
    ctx.place(pc_mux_x, pc_mux_y, TileType::Mux);

    // Wire branch taken signal to mux select (up input)
    ctx.wire_v(branch_x + 10, branch_y, pc_mux_y + 1);
    ctx.wire_h(pc_mux_y + 1, pc_mux_x, branch_x + 10);

    // Wire branch target to mux left input
    ctx.wire_h(branch_y, target_x, pc_mux_x);
    ctx.wire_v(pc_mux_x, branch_y, pc_mux_y);

    // PC (right input) gets PC+1 by default (internal to ProgramCounter tile)
}

// =============================================================================
// Phase 7: Writeback
// =============================================================================

/// Wire the writeback path from ALU to registers
pub fn wire_writeback(ctx: &mut WiringContext) {
    let (ox, oy) = ctx.origin;
    let reg_y = oy + 4;

    // -------------------------------------------------------------------------
    // Writeback Bus
    // Carries ALU result to all registers
    // -------------------------------------------------------------------------
    ctx.wire_h(ctx.writeback_y, ox, ox + 24);

    // Connect ALU result to writeback bus
    ctx.wire_v(ox + 10, ctx.alu_result_y, ctx.writeback_y);

    // -------------------------------------------------------------------------
    // Register Write Demux
    // Routes data to selected register based on Rd
    // -------------------------------------------------------------------------
    let demux_x = ox + 6;
    let demux_y = reg_y - 1;
    ctx.place(demux_x, demux_y, TileType::Demux1to8);

    // Wire writeback bus to demux data input
    ctx.wire_v(ox + 6, ctx.writeback_y, demux_y + 1);

    // Wire Rd to demux select
    let rd_x = ox + 16;
    ctx.wire_h(demux_y, rd_x, demux_x + 1);

    // -------------------------------------------------------------------------
    // Register Write Enable Generation
    // Write enable = opcode is register-writing instruction
    // -------------------------------------------------------------------------
    let we_gen_x = ox + 14;
    let we_gen_y = demux_y;

    // Write enable for: LDI(1), MOV(2), ADD(3), SUB(4), AND(5), OR(6), XOR(7), LD(E)
    // Simplified: write if opcode in {1,2,3,4,5,6,7,E}
    // = (opcode >= 1 AND opcode <= 7) OR (opcode == 0xE)

    ctx.place(we_gen_x, we_gen_y, TileType::Or);

    // Connect to all register write enables
    for reg in 0..NUM_REGISTERS {
        let reg_x = ox + 4 + reg * 4;
        ctx.wire_h(demux_y, demux_x, reg_x);
    }
}

// =============================================================================
// RAM Wiring
// =============================================================================

/// Wire the RAM for load/store instructions
pub fn wire_ram(ctx: &mut WiringContext, ram_size: usize) {
    let (ox, oy) = ctx.origin;
    let ram_base_y = oy + 32;

    // -------------------------------------------------------------------------
    // RAM Array
    // -------------------------------------------------------------------------
    for addr in 0..ram_size {
        let ram_x = ox + (addr % 8) * 2;
        let ram_y = ram_base_y + (addr / 8);

        let idx = ctx.place(ram_x, ram_y, TileType::Ram);
        ctx.ram_indices.push(idx);
    }

    // -------------------------------------------------------------------------
    // RAM Address Decoder
    // Uses R3 as address register
    // -------------------------------------------------------------------------
    let ram_mux_x = ox + 4;
    let ram_mux_y = ram_base_y - 1;
    ctx.place(ram_mux_x, ram_mux_y, TileType::Mux8to1);

    // Wire R3 to RAM address
    let r3_x = ox + 4 + 3 * 4;
    ctx.wire_v(r3_x, oy + 4, ram_mux_y);

    // -------------------------------------------------------------------------
    // RAM Write Enable
    // Active when opcode == 0xF (ST)
    // -------------------------------------------------------------------------
    let ram_we_x = ox + 16;
    let ram_we_y = ram_base_y - 2;
    ctx.place(ram_we_x, ram_we_y, TileType::And);

    // Wire to RAM tiles (up input is write enable)
    // Collect Y positions first to avoid borrow conflict
    let ram_y_positions: Vec<usize> = ctx.ram_indices.iter()
        .map(|idx| *idx / ctx.grid_width)
        .collect();
    for ram_y in ram_y_positions {
        ctx.wire_v(ram_we_x, ram_we_y, ram_y);
    }

    // -------------------------------------------------------------------------
    // RAM Read Path
    // Result goes to writeback bus for LD instruction
    // -------------------------------------------------------------------------
    ctx.wire_v(ram_mux_x, ram_mux_y, ctx.writeback_y);
}

// =============================================================================
// Complete Wiring
// =============================================================================

/// Wire the complete CPU datapath
pub fn wire_complete_cpu(
    ctx: &mut WiringContext,
    program: &[u8],
    rom_size: usize,
    ram_size: usize,
    initial_regs: &[u8; NUM_REGISTERS],
) {
    wire_instruction_fetch(ctx, program, rom_size);
    wire_instruction_decode(ctx);
    wire_register_file(ctx, initial_regs);
    wire_alu(ctx);
    wire_flags(ctx);
    wire_branch_logic(ctx);
    wire_writeback(ctx);
    wire_ram(ctx, ram_size);
}
