//! TileCpu Builder - Construct CPUs from tiles
//!
//! This module handles the layout and placement of tiles to form a working CPU.
//! The builder uses the wiring module to create actual tile configurations.

use crate::simulation::Simulation;
use crate::tile_cpu::wiring::{WiringContext, wire_complete_cpu, wire_physical_cpu};
use crate::tile_cpu::{MAX_RAM_SIZE, MAX_ROM_SIZE, NUM_REGISTERS, PhysicalCpu, TileCpu};

/// Builder for constructing a TileCpu
///
/// # Example
///
/// ```rust,ignore
/// let cpu = TileCpuBuilder::new()
///     .with_origin(10, 10)
///     .with_program(&[0x12, 0x17, 0x31])
///     .with_rom_size(16)
///     .with_ram_size(16)
///     .build(&mut sim);
/// ```
#[derive(Debug, Clone)]
pub struct TileCpuBuilder {
    origin: (usize, usize),
    program: Vec<u8>,
    rom_size: usize,
    ram_size: usize,
    initial_regs: [u64; NUM_REGISTERS],
}

impl Default for TileCpuBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TileCpuBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            origin: (0, 0),
            program: Vec::new(),
            rom_size: 16,
            ram_size: 16,
            initial_regs: [0; NUM_REGISTERS],
        }
    }

    /// Set the origin (top-left corner) of the CPU on the grid
    pub fn with_origin(mut self, x: usize, y: usize) -> Self {
        self.origin = (x, y);
        self
    }

    /// Load a program into ROM
    pub fn with_program(mut self, program: &[u8]) -> Self {
        self.program = program.to_vec();
        self
    }

    /// Set ROM size (default: 16 bytes)
    pub fn with_rom_size(mut self, size: usize) -> Self {
        self.rom_size = size.min(MAX_ROM_SIZE);
        self
    }

    /// Set RAM size (default: 16 bytes)
    pub fn with_ram_size(mut self, size: usize) -> Self {
        self.ram_size = size.min(MAX_RAM_SIZE);
        self
    }

    /// Set initial register values
    pub fn with_initial_regs(mut self, regs: [u64; NUM_REGISTERS]) -> Self {
        self.initial_regs = regs;
        self
    }

    /// Build the CPU, placing tiles in the simulation
    ///
    /// This creates the complete wired datapath where instructions execute
    /// by propagating signals through actual tile connections.
    ///
    /// After building, two settling ticks are run to:
    /// 1. Advance PC from MAX to 0 (rising edge)
    /// 2. Settle Mux8to1 to fetch instruction[0]
    /// 3. Complete the clock cycle (falling edge)
    pub fn build(self, sim: &mut Simulation) -> TileCpu {
        // Create wiring context
        let mut ctx = WiringContext::new(sim, self.origin);

        // Wire the complete CPU datapath
        wire_complete_cpu(
            &mut ctx,
            &self.program,
            self.rom_size,
            self.ram_size,
            &self.initial_regs,
        );

        // Build TileCpu from wiring context
        let grid_width = ctx.grid_width;
        let tile_count = ctx.total_tiles();
        let ir_idx = ctx.ir_idx;

        let cpu = TileCpu::new(
            self.origin,
            ctx.pc_idx,
            ir_idx,
            ctx.reg_indices,
            ctx.rom_indices.clone(),
            ctx.ram_indices.clone(),
            ctx.flag_z_idx,
            ctx.flag_c_idx,
            ctx.flag_z_result_idx,
            ctx.flag_z_update_idx,
            ctx.flag_z_zero_idx,
            ctx.flag_z_mux_idx,
            ctx.flag_c_carry_idx,
            ctx.flag_c_update_idx,
            ctx.flag_c_mux_idx,
            ctx.alu_tile_indices,
            ctx.op_a_data_indices,
            ctx.op_a_sel0_indices,
            ctx.op_a_sel1_idx,
            ctx.op_a_leaf_indices,
            ctx.op_a_root_idx,
            ctx.op_b_data_indices,
            ctx.op_b_sel0_indices,
            ctx.op_b_sel1_idx,
            ctx.op_b_leaf_indices,
            ctx.op_b_root_idx,
            ctx.next_pc_const_idx,
            ctx.decoder_opcode_lo_idx,
            ctx.decoder_ctrl_a_lo_idx,
            ctx.decoder_ctrl_a_hi_idx,
            ctx.decoder_ctrl_b_lo_idx,
            ctx.decoder_ctrl_b_hi_idx,
            ctx.alu_result_data_indices,
            ctx.alu_result_sel0_indices,
            ctx.alu_result_sel1_indices,
            ctx.alu_result_sel2_idx,
            ctx.alu_result_leaf_indices,
            ctx.alu_result_mid_indices,
            ctx.alu_result_root_idx,
            ctx.reg_we_indices,
            ctx.reg_result_indices,
            ctx.reg_mux_indices,
            grid_width,
            tile_count,
        );

        // Mark all tiles dirty so they evaluate at least once during settling.
        // set_tile() does NOT mark tiles dirty, so placed tiles would otherwise
        // never enter the dirty set. Without this, combinational tiles like
        // Mux8to1, WireDown, WireLeft never evaluate during the settling ticks.
        sim.dirty.mark_all_dirty(sim.tile_count());

        // Run settling ticks to initialize state:
        // - Rising edge: Register8_PC captures Const_NPC (=0) → PC = 0
        // - All combinational tiles evaluate (ROM Const → Mux8to1 fetches instruction[0])
        // - Falling edge: completes the cycle
        // Note: Register8 registers capture 0 during settling (Mux not yet settled).
        // Initial register values are set below after settling.
        sim.tick_with_delays();
        sim.tick_with_delays();

        // Set initial register values after settling.
        // During settling, Register8 registers captured 0 (Mux tiles output 0).
        // We set the initial values directly, which the first tick() will propagate
        // through the Mux feedback loop correctly.
        for (i, &val) in self.initial_regs.iter().enumerate() {
            cpu.write_reg(sim, i, val);
        }

        cpu
    }

    /// Build a physical CPU with tile-based decoder circuit.
    ///
    /// The decoder is fully tile-based: IR → Shr(4) → WD chain → 4 Mux8to1 LUTs.
    /// After settling, the LUT outputs contain correct control signals for the
    /// fetched instruction. Software reads them directly — no Const writes,
    /// no `propagate_combinational()` for decode.
    ///
    /// Operand staging, ALU result routing, register writeback, flags, branches,
    /// and memory still use software staging (same as hybrid `build()`).
    pub fn build_physical(self, sim: &mut Simulation) -> PhysicalCpu {
        let mut ctx = WiringContext::new(sim, self.origin);

        let phys = wire_physical_cpu(
            &mut ctx,
            &self.program,
            self.rom_size,
            self.ram_size,
            &self.initial_regs,
        );

        let cpu = PhysicalCpu::from_wiring(self.origin, &phys);

        // Mark all tiles dirty so they evaluate during settling
        sim.dirty.mark_all_dirty(sim.tile_count());

        // Settling ticks: PC captures 0, Mux8to1 fetches instruction[0],
        // physical decoder settles (Shr computes opcode, LUTs produce ctrl bytes),
        // IR BitSelect tiles extract rd/rs/jump_target.
        // tick_with_delays() runs up to MAX_DELTA=500, settling the entire L1 PC
        // routing path (~200 deltas) within a single call.
        sim.tick_with_delays();
        sim.tick_with_delays();

        // Set initial register values
        for (i, &val) in self.initial_regs.iter().enumerate() {
            cpu.write_reg(sim, i, val);
        }

        cpu
    }

    /// Build a minimal CPU for testing (fewer wires, just core components)
    pub fn build_minimal(self, sim: &mut Simulation) -> TileCpu {
        use crate::tile_meta::TileType;

        let (ox, oy) = self.origin;
        let grid_width = sim.width();
        let mut tile_count = 0;

        // PC
        let pc_x = ox + 4;
        let pc_y = oy;
        sim.set_tile(pc_x, pc_y, TileType::ProgramCounter);
        sim.set_logic_value(pc_x, pc_y, 0);
        let pc_idx = pc_y * grid_width + pc_x;
        tile_count += 1;

        // Registers
        let mut reg_indices = [0usize; NUM_REGISTERS];
        for reg in 0..NUM_REGISTERS {
            let reg_x = ox + 4 + reg * 4;
            let reg_y = oy + 4;
            sim.set_tile(reg_x, reg_y, TileType::RegEnable);
            sim.set_logic_value(reg_x, reg_y, self.initial_regs[reg]);
            reg_indices[reg] = reg_y * grid_width + reg_x;
            tile_count += 1;
        }

        // ALU
        let alu_y = oy + 8;
        sim.set_tile(ox + 4, alu_y, TileType::Add);
        sim.set_tile(ox + 6, alu_y, TileType::Sub);
        sim.set_tile(ox + 8, alu_y, TileType::And);
        sim.set_tile(ox + 10, alu_y, TileType::Or);
        sim.set_tile(ox + 12, alu_y, TileType::Xor);
        tile_count += 5;

        sim.set_tile(ox + 8, alu_y + 1, TileType::Mux8to1);
        tile_count += 1;

        // Flags
        let flag_y = oy + 12;
        sim.set_tile(ox + 4, flag_y, TileType::Zero);
        let flag_z_idx = flag_y * grid_width + ox + 4;
        sim.set_tile(ox + 6, flag_y, TileType::Latch);
        let flag_c_idx = flag_y * grid_width + ox + 6;
        tile_count += 2;

        // ROM
        let rom_y = oy + 16;
        let mut rom_indices = Vec::with_capacity(self.rom_size);
        for addr in 0..self.rom_size {
            let rom_x = ox + (addr % 8);
            let rom_row = rom_y + (addr / 8);
            sim.set_tile(rom_x, rom_row, TileType::Const);
            let value = if addr < self.program.len() {
                self.program[addr] as u64
            } else {
                0
            };
            sim.set_logic_value(rom_x, rom_row, value);
            rom_indices.push(rom_row * grid_width + rom_x);
            tile_count += 1;
        }

        // RAM
        let ram_y = oy + 32;
        let mut ram_indices = Vec::with_capacity(self.ram_size);
        for addr in 0..self.ram_size {
            let ram_x = ox + (addr % 8);
            let ram_row = ram_y + (addr / 8);
            sim.set_tile(ram_x, ram_row, TileType::Ram);
            sim.set_logic_value(ram_x, ram_row, 0);
            ram_indices.push(ram_row * grid_width + ram_x);
            tile_count += 1;
        }

        TileCpu::new(
            self.origin,
            pc_idx,
            0, // ir_idx not used in minimal build
            reg_indices,
            rom_indices,
            ram_indices,
            flag_z_idx,
            flag_c_idx,
            0,                  // flag_z_result_idx
            0,                  // flag_z_update_idx
            0,                  // flag_z_zero_idx
            0,                  // flag_z_mux_idx
            0,                  // flag_c_carry_idx
            0,                  // flag_c_update_idx
            0,                  // flag_c_mux_idx
            [0; 8],             // alu_tile_indices
            [0; 4],             // op_a_data_indices
            [0; 2],             // op_a_sel0_indices
            0,                  // op_a_sel1_idx
            [0; 2],             // op_a_leaf_indices
            0,                  // op_a_root_idx
            [0; 4],             // op_b_data_indices
            [0; 2],             // op_b_sel0_indices
            0,                  // op_b_sel1_idx
            [0; 2],             // op_b_leaf_indices
            0,                  // op_b_root_idx
            0,                  // next_pc_const_idx
            0,                  // decoder_opcode_lo_idx
            0,                  // decoder_ctrl_a_lo_idx
            0,                  // decoder_ctrl_a_hi_idx
            0,                  // decoder_ctrl_b_lo_idx
            0,                  // decoder_ctrl_b_hi_idx
            [0; 8],             // alu_result_data_indices
            [0; 4],             // alu_result_sel0_indices
            [0; 2],             // alu_result_sel1_indices
            0,                  // alu_result_sel2_idx
            [0; 4],             // alu_result_leaf_indices
            [0; 2],             // alu_result_mid_indices
            0,                  // alu_result_root_idx
            [0; NUM_REGISTERS], // reg_we_indices
            [0; NUM_REGISTERS], // reg_result_indices
            [0; NUM_REGISTERS], // reg_mux_indices
            grid_width,
            tile_count,
        )
    }
}

// =============================================================================
// Layout Constants
// =============================================================================

/// CPU layout dimensions
pub struct CpuLayoutDimensions {
    /// Total width in tiles
    pub width: usize,
    /// Total height in tiles
    pub height: usize,
    /// Minimum grid size needed
    pub min_grid_size: (usize, usize),
}

impl TileCpuBuilder {
    /// Calculate layout dimensions before building
    pub fn layout_dimensions(&self) -> CpuLayoutDimensions {
        let rom_rows = self.rom_size.div_ceil(8);
        let ram_rows = self.ram_size.div_ceil(8);

        CpuLayoutDimensions {
            width: 64, // Wide for pure-tile datapath with registers at col 32+
            height: 48 + rom_rows + ram_rows,
            min_grid_size: (self.origin.0 + 64, self.origin.1 + 48 + rom_rows + ram_rows),
        }
    }
}
