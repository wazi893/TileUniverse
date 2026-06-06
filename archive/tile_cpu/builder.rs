//! TileCpu Builder - Construct CPUs from tiles
//!
//! This module handles the layout and placement of tiles to form a working CPU.
//! The builder uses the wiring module to create actual tile configurations.

use crate::simulation::Simulation;
use crate::tile_cpu::{
    TileCpu, DATAPATH_WIDTH, NUM_REGISTERS, MAX_ROM_SIZE, MAX_RAM_SIZE,
};
use crate::tile_cpu::wiring::{WiringContext, wire_complete_cpu};

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
    initial_regs: [u8; NUM_REGISTERS],
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
    pub fn with_initial_regs(mut self, regs: [u8; NUM_REGISTERS]) -> Self {
        self.initial_regs = regs;
        self
    }

    /// Build the CPU, placing tiles in the simulation
    ///
    /// This creates the complete wired datapath where instructions execute
    /// by propagating signals through actual tile connections.
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
        let alu_out_indices = [ctx.alu_mux_idx; DATAPATH_WIDTH]; // Simplified
        let grid_width = ctx.grid_width;
        let tile_count = ctx.total_tiles();

        TileCpu::new(
            self.origin,
            ctx.pc_idx,
            ctx.reg_indices,
            ctx.rom_indices.clone(),
            ctx.ram_indices.clone(),
            ctx.flag_z_idx,
            ctx.flag_c_idx,
            alu_out_indices,
            grid_width,
            tile_count,
        )
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
            sim.set_logic_value(reg_x, reg_y, self.initial_regs[reg] as u64);
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

        let alu_mux_idx = (alu_y + 1) * grid_width + ox + 8;
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

        let alu_out_indices = [alu_mux_idx; DATAPATH_WIDTH];

        TileCpu::new(
            self.origin,
            pc_idx,
            reg_indices,
            rom_indices,
            ram_indices,
            flag_z_idx,
            flag_c_idx,
            alu_out_indices,
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
        let rom_rows = (self.rom_size + 7) / 8;
        let ram_rows = (self.ram_size + 7) / 8;

        CpuLayoutDimensions {
            width: 32,  // Increased for full wiring
            height: 48 + rom_rows + ram_rows,
            min_grid_size: (
                self.origin.0 + 32,
                self.origin.1 + 48 + rom_rows + ram_rows,
            ),
        }
    }
}
