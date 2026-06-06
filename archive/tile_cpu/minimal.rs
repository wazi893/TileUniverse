//! Minimal CPU - Non-overlapping wiring for testing
//!
//! This module creates a minimal CPU with isolated signal paths.
//! No overlapping buses - each wire has a dedicated path.
//!
//! Layout (25x20 grid):
//! ```text
//!     0    5    10   15   20
//! 0   [CLK]     [PC]
//! 1        |    |
//! 2   [ROM0][ROM1][ROM2][ROM3]  // ROM row
//! 3        |
//! 4   [IR]-------.
//! 5   |    |     |
//! 6   [R0] |     |
//! 7   |    |     |
//! 8   [R1]-+     |
//! 9   |    |     |
//! 10  [ADD]------'
//! 11  |
//! 12  [WB]-------------------> (back to R0/R1 via demux)
//! ```

use crate::simulation::Simulation;
use crate::tile_meta::TileType;

/// Minimal CPU configuration
pub struct MinimalCpu {
    pub origin: (usize, usize),
    pub tile_count: usize,
    // Key positions
    pub pc_x: usize,
    pub pc_y: usize,
    pub ir_x: usize,
    pub ir_y: usize,
    pub r0_x: usize,
    pub r0_y: usize,
    pub r1_x: usize,
    pub r1_y: usize,
    pub add_x: usize,
    pub add_y: usize,
}

/// Helper struct for building
struct Builder<'a> {
    sim: &'a mut Simulation,
    tile_count: usize,
}

impl<'a> Builder<'a> {
    fn new(sim: &'a mut Simulation) -> Self {
        Self { sim, tile_count: 0 }
    }

    fn place(&mut self, x: usize, y: usize, tt: TileType) {
        self.sim.set_tile(x, y, tt);
        self.tile_count += 1;
    }

    fn place_val(&mut self, x: usize, y: usize, tt: TileType, val: u64) {
        self.sim.set_tile(x, y, tt);
        self.sim.set_logic_value(x, y, val);
        self.tile_count += 1;
    }
}

impl MinimalCpu {
    /// Build a minimal CPU with isolated wiring
    ///
    /// Supports instructions:
    /// - 0x12: LDI R0, #2  (opcode=1, rd=0, imm=2)
    /// - 0x17: LDI R1, #3  (opcode=1, rd=1, imm=3)
    /// - 0x31: ADD R0, R1  (opcode=3, rd=0, rs=1)
    /// - 0x00: NOP
    pub fn build(sim: &mut Simulation, origin: (usize, usize), program: &[u8]) -> Self {
        let (ox, oy) = origin;
        let mut b = Builder::new(sim);

        // =====================================================================
        // Clock (provides timing to sequential elements)
        // =====================================================================
        let clock_x = ox;
        let clock_y = oy;
        b.place(clock_x, clock_y, TileType::ClockGlobal);

        // =====================================================================
        // Program Counter
        // PC is at (ox+4, oy), clock connects from left
        // =====================================================================
        let pc_x = ox + 4;
        let pc_y = oy;
        b.place_val(pc_x, pc_y, TileType::ProgramCounter, 0);

        // Clock to PC: horizontal wire (no crossing needed)
        b.place(ox + 1, oy, TileType::WireH);
        b.place(ox + 2, oy, TileType::WireH);
        b.place(ox + 3, oy, TileType::WireH);

        // PC jump input (right) = 0 means no jump, always increment
        b.place_val(pc_x + 1, pc_y, TileType::Const, 0);

        // =====================================================================
        // ROM (4 entries, horizontal row)
        // PC output goes down, then selects ROM entry
        // =====================================================================
        let rom_y = oy + 3;

        // ROM entries
        for i in 0..4 {
            let rom_x = ox + i * 3;
            let val = if i < program.len() { program[i] as u64 } else { 0 };
            b.place_val(rom_x, rom_y, TileType::Const, val);
        }

        // ROM selector (Mux8to1) - selects which ROM entry based on PC
        let rom_mux_x = ox + 6;
        let rom_mux_y = oy + 2;
        b.place(rom_mux_x, rom_mux_y, TileType::Mux8to1);

        // PC to ROM mux select (vertical wire down, then horizontal)
        b.place(pc_x, oy + 1, TileType::WireV);
        b.place(pc_x, oy + 2, TileType::WireH);  // Turn right
        b.place(pc_x + 1, oy + 2, TileType::WireH);
        b.place(pc_x + 2, oy + 2, TileType::WireH);

        // ROM outputs connect to mux (below mux)
        // For simplicity, we wire only ROM[0] and ROM[1] as test
        b.place(rom_mux_x, rom_y, TileType::Wire); // Mux data input below

        // =====================================================================
        // Instruction Register
        // Captures ROM output on clock edge
        // =====================================================================
        let ir_x = ox + 2;
        let ir_y = oy + 5;
        b.place(ir_x, ir_y, TileType::Register8);

        // Clock to IR (from clock, go down then right)
        b.place(clock_x, oy + 1, TileType::WireV);
        b.place(clock_x, oy + 2, TileType::WireV);
        b.place(clock_x, oy + 3, TileType::WireV);
        b.place(clock_x, oy + 4, TileType::WireV);
        b.place(clock_x, oy + 5, TileType::WireH);
        b.place(ox + 1, oy + 5, TileType::WireH);

        // ROM mux output to IR input (down from mux, then left to IR)
        b.place(rom_mux_x, oy + 3, TileType::WireV);
        b.place(rom_mux_x, oy + 4, TileType::WireV);
        b.place(rom_mux_x, oy + 5, TileType::WireH);
        b.place(rom_mux_x - 1, oy + 5, TileType::WireH);
        b.place(rom_mux_x - 2, oy + 5, TileType::WireH);
        b.place(ir_x + 1, ir_y, TileType::WireH);

        // =====================================================================
        // Registers R0 and R1
        // =====================================================================
        let r0_x = ox + 12;
        let r0_y = oy + 7;
        b.place_val(r0_x, r0_y, TileType::RegEnable, 0);

        let r1_x = ox + 12;
        let r1_y = oy + 9;
        b.place_val(r1_x, r1_y, TileType::RegEnable, 0);

        // Clock to registers (separate vertical path on right side)
        let clock_reg_x = ox + 10;
        b.place(clock_reg_x, oy + 1, TileType::WireV);
        b.place(clock_reg_x, oy + 2, TileType::WireV);
        b.place(clock_reg_x, oy + 3, TileType::WireV);
        b.place(clock_reg_x, oy + 4, TileType::WireV);
        b.place(clock_reg_x, oy + 5, TileType::WireV);
        b.place(clock_reg_x, oy + 6, TileType::WireV);
        b.place(clock_reg_x, oy + 7, TileType::Cross); // Cross for R0 clock
        b.place(clock_reg_x, oy + 8, TileType::WireV);
        b.place(clock_reg_x, oy + 9, TileType::WireH); // R1 clock

        // Connect clock line to R0 (horizontal from cross)
        b.place(clock_reg_x + 1, r0_y, TileType::WireH);

        // Connect clock line to R1
        b.place(clock_reg_x + 1, r1_y, TileType::WireH);

        // Clock source to register clock line (horizontal at top)
        b.place(ox + 1, oy + 1, TileType::WireH);
        b.place(ox + 2, oy + 1, TileType::WireH);
        b.place(ox + 3, oy + 1, TileType::Cross); // Cross where PC vertical would be
        b.place(ox + 4, oy + 1, TileType::WireH);
        b.place(ox + 5, oy + 1, TileType::WireH);
        b.place(ox + 6, oy + 1, TileType::WireH);
        b.place(ox + 7, oy + 1, TileType::WireH);
        b.place(ox + 8, oy + 1, TileType::WireH);
        b.place(ox + 9, oy + 1, TileType::WireH);
        b.place(clock_reg_x, oy + 1, TileType::WireV); // Turn down

        // =====================================================================
        // ALU (just Add for now)
        // Takes R0 (left) and R1 (right), outputs sum
        // =====================================================================
        let add_x = ox + 14;
        let add_y = oy + 11;
        b.place(add_x, add_y, TileType::Add);

        // R0 output to Add (go right from R0, then down to Add left input)
        b.place(r0_x + 1, r0_y, TileType::WireH);
        b.place(r0_x + 2, r0_y, TileType::Cross); // Will cross with other signals
        b.place(r0_x + 2, oy + 8, TileType::WireV);
        b.place(r0_x + 2, oy + 9, TileType::Cross);
        b.place(r0_x + 2, oy + 10, TileType::WireV);
        b.place(r0_x + 2, oy + 11, TileType::WireH);
        b.place(add_x - 1, add_y, TileType::WireH);

        // R1 output to Add (go right from R1, then down)
        b.place(r1_x + 1, r1_y, TileType::WireH);
        b.place(r1_x + 2, r1_y, TileType::WireH); // Already cross from R0 path
        b.place(r1_x + 3, r1_y, TileType::WireH);
        b.place(r1_x + 3, oy + 10, TileType::WireV);
        b.place(r1_x + 3, oy + 11, TileType::WireH);
        b.place(add_x + 1, add_y, TileType::WireH);

        // =====================================================================
        // Writeback path
        // Add output goes back to R0/R1 data input
        // =====================================================================
        let wb_y = oy + 13;

        // Add output goes down
        b.place(add_x, oy + 12, TileType::WireV);
        b.place(add_x, wb_y, TileType::WireH);

        // Writeback mux (selects which register to write based on Rd)
        let wb_mux_x = ox + 18;
        b.place(wb_mux_x, wb_y, TileType::Demux1to8);

        // Connect Add output to demux
        b.place(add_x + 1, wb_y, TileType::WireH);
        b.place(add_x + 2, wb_y, TileType::WireH);
        b.place(add_x + 3, wb_y, TileType::WireH);

        // Demux outputs go up to registers
        // Output 0 -> R0 data input (left side of R0)
        b.place(wb_mux_x, oy + 12, TileType::WireV);
        b.place(wb_mux_x, oy + 11, TileType::WireV);
        b.place(wb_mux_x, oy + 10, TileType::WireV);
        b.place(wb_mux_x, oy + 9, TileType::WireV);
        b.place(wb_mux_x, oy + 8, TileType::WireV);
        b.place(wb_mux_x, oy + 7, TileType::WireH);
        b.place(wb_mux_x - 1, oy + 7, TileType::WireH);
        b.place(wb_mux_x - 2, oy + 7, TileType::WireH);
        b.place(wb_mux_x - 3, oy + 7, TileType::WireH);
        b.place(wb_mux_x - 4, oy + 7, TileType::WireH);
        b.place(wb_mux_x - 5, oy + 7, TileType::WireH);

        // =====================================================================
        // Immediate value path (for LDI instruction)
        // Lower bits of IR go to writeback
        // =====================================================================
        // Extract immediate from IR (bits 1-0)
        let imm_x = ox + 4;
        let imm_y = oy + 6;
        b.place_val(imm_x + 1, imm_y, TileType::Const, 0x07); // Mask for 3 bits
        b.place(imm_x, imm_y, TileType::And);

        // IR to immediate extractor
        b.place(ir_x, ir_y + 1, TileType::WireV);
        b.place(ir_x, imm_y, TileType::WireH);
        b.place(ir_x + 1, imm_y, TileType::WireH);

        // Immediate to writeback mux (LDI path)
        // For now, simplified - we'd need another mux to select ALU vs immediate

        Self {
            origin,
            tile_count: b.tile_count,
            pc_x,
            pc_y,
            ir_x,
            ir_y,
            r0_x,
            r0_y,
            r1_x,
            r1_y,
            add_x,
            add_y,
        }
    }

    /// Get PC value
    pub fn get_pc(&self, sim: &Simulation) -> u64 {
        sim.get_logic_at(self.pc_x, self.pc_y)
    }

    /// Get IR value
    pub fn get_ir(&self, sim: &Simulation) -> u64 {
        sim.get_logic_at(self.ir_x, self.ir_y)
    }

    /// Get R0 value
    pub fn get_r0(&self, sim: &Simulation) -> u64 {
        sim.get_logic_at(self.r0_x, self.r0_y)
    }

    /// Get R1 value
    pub fn get_r1(&self, sim: &Simulation) -> u64 {
        sim.get_logic_at(self.r1_x, self.r1_y)
    }

    /// Get Add output
    pub fn get_add_result(&self, sim: &Simulation) -> u64 {
        sim.get_logic_at(self.add_x, self.add_y)
    }

    /// Dump state
    pub fn dump_state(&self, sim: &Simulation) -> String {
        format!(
            "PC={} IR=0x{:02X} R0={} R1={} ADD={}",
            self.get_pc(sim),
            self.get_ir(sim),
            self.get_r0(sim),
            self.get_r1(sim),
            self.get_add_result(sim)
        )
    }
}

/// Build a super minimal test: just PC + Clock
pub fn build_pc_only(sim: &mut Simulation, origin: (usize, usize)) -> (usize, usize) {
    let (ox, oy) = origin;

    // Clock
    sim.set_tile(ox, oy, TileType::ClockGlobal);

    // PC with clock above
    sim.set_tile(ox, oy + 1, TileType::ProgramCounter);
    sim.set_logic_value(ox, oy + 1, 0);

    // Jump disable (right of PC) = 0
    sim.set_tile(ox + 1, oy + 1, TileType::Const);
    sim.set_logic_value(ox + 1, oy + 1, 0);

    (ox, oy + 1) // Return PC position
}

/// Build a minimal test: PC + Clock + IR
pub fn build_pc_ir(sim: &mut Simulation, origin: (usize, usize), rom_value: u64) -> (usize, usize, usize, usize) {
    let (ox, oy) = origin;

    // Clock at top
    sim.set_tile(ox + 2, oy, TileType::ClockGlobal);

    // PC below clock
    sim.set_tile(ox + 2, oy + 1, TileType::ProgramCounter);
    sim.set_logic_value(ox + 2, oy + 1, 0);

    // Jump disable
    sim.set_tile(ox + 3, oy + 1, TileType::Const);
    sim.set_logic_value(ox + 3, oy + 1, 0);

    // ROM constant (simulates single ROM entry)
    sim.set_tile(ox, oy + 2, TileType::Const);
    sim.set_logic_value(ox, oy + 2, rom_value);

    // IR with clock from top
    sim.set_tile(ox + 2, oy + 3, TileType::Register8);

    // Clock wire down to IR
    sim.set_tile(ox + 2, oy + 2, TileType::WireV);

    // ROM to IR input (horizontal wire)
    sim.set_tile(ox + 1, oy + 3, TileType::WireH);

    // PC position, IR position
    (ox + 2, oy + 1, ox + 2, oy + 3)
}

/// Build a test with two registers and add
/// Layout:
///   Const(10) at (ox, oy)    Add at (ox+2, oy)    Const(20) at (ox+4, oy)
///       left neighbor          center tile          right neighbor
pub fn build_add_test(sim: &mut Simulation, origin: (usize, usize)) -> (usize, usize, usize, usize, usize, usize) {
    let (ox, oy) = origin;

    // Const1(10) is left neighbor of Add
    sim.set_tile(ox, oy, TileType::Const);
    sim.set_logic_value(ox, oy, 10);

    // Add tile at center - takes left (ox, oy) and right (ox+2, oy)
    sim.set_tile(ox + 1, oy, TileType::Add);

    // Const2(20) is right neighbor of Add
    sim.set_tile(ox + 2, oy, TileType::Const);
    sim.set_logic_value(ox + 2, oy, 20);

    // (const1_x, const1_y, const2_x, const2_y, add_x, add_y)
    (ox, oy, ox + 2, oy, ox + 1, oy)
}
