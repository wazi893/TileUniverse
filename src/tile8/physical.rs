//! EPIC 111: Physical TILE-8 CPU
//!
//! A minimal 8-bit CPU built from arithmetic tiles.
//! Uses Option E architecture: arithmetic tiles directly (Add, Sub, And, Or, Xor).
//!
//! Phase 1: Minimal CPU ✓
//! - PC (program counter)
//! - ROM (program storage)
//! - Registers R0-R3
//! - ALU (Add)
//! - LDI, MOV, ADD, SUB instructions
//!
//! Phase 2: Full ALU + Flags ✓
//! - Full ALU: Add, Sub, And, Or, Xor tiles
//! - Flags: Zero (Z), Carry (C)
//! - All 16 instructions
//!
//! Phase 3: Memory ✓
//! - 4-byte data RAM with LOAD/STORE
//!
//! Phase 4: Multi-CPU Instancing
//! - Grid-based CPU placement
//! - Parallel execution

use crate::slim_simulation::SlimSimulation;
use crate::tile_meta::TileType;
use rayon::prelude::*;

// =============================================================================
// Lightweight GPU CPU (minimal memory footprint)
// =============================================================================

/// Minimal CPU data for GPU execution (no heap allocations!)
#[derive(Debug, Clone, Copy)]
pub struct GpuCpuData {
    pub origin: (usize, usize),
    pub regs: [(usize, usize); 4], // Register tile positions [N,S,E,W]
}

/// Minimal CPU representation for GPU execution
///
/// Only stores what's needed for the inspector - the GPU shader
/// handles all execution logic internally.
///
/// Memory: ~80 bytes vs ~500+ bytes for CpuInstance
#[derive(Debug, Clone, Copy)]
pub struct GpuCpuInstance {
    pub id: usize,
    pub cpu: GpuCpuData, // Matches CpuInstance.cpu.* access pattern
}

impl GpuCpuInstance {
    pub fn new(id: usize, origin: (usize, usize)) -> Self {
        let (ox, oy) = origin;
        let cpu_x = ox + 1;
        let cpu_y = oy + 1;

        Self {
            id,
            cpu: GpuCpuData {
                origin,
                regs: [
                    (cpu_x, cpu_y.saturating_sub(1)), // Reg A = North
                    (cpu_x, cpu_y + 1),               // Reg B = South
                    (cpu_x + 1, cpu_y),               // Reg C = East
                    (cpu_x.saturating_sub(1), cpu_y), // Reg D = West
                ],
            },
        }
    }

    /// Place GPU-compatible CPU layout into simulation
    pub fn place(&self, sim: &mut SlimSimulation) {
        let (ox, oy) = self.cpu.origin;
        let cpu_x = ox + 1;
        let cpu_y = oy + 1;

        // Place CpuHead at center
        sim.set_tile(cpu_x, cpu_y, TileType::CpuHead);

        // Place Registers as immediate neighbors
        for &(rx, ry) in &self.cpu.regs {
            sim.set_tile(rx, ry, TileType::Register);
        }
    }

    /// Read register value from simulation (for inspector)
    pub fn read_register(&self, sim: &SlimSimulation, reg: usize) -> u8 {
        if reg < 4 {
            let (x, y) = self.cpu.regs[reg];
            sim.get_logic_at(x, y) as u8
        } else {
            0
        }
    }

    /// Place CPU tiles directly into a flat tile_types array (bypasses SlimSimulation)
    #[inline]
    pub fn place_direct(&self, tile_types: &mut [u8], grid_width: usize) {
        let (ox, oy) = self.cpu.origin;
        let cpu_x = ox + 1;
        let cpu_y = oy + 1;

        // Place CpuHead at center (type 40)
        let cpu_idx = cpu_y * grid_width + cpu_x;
        if cpu_idx < tile_types.len() {
            tile_types[cpu_idx] = 40; // CpuHead
        }

        // Place Registers as immediate neighbors (type 41)
        for &(rx, ry) in &self.cpu.regs {
            let reg_idx = ry * grid_width + rx;
            if reg_idx < tile_types.len() {
                tile_types[reg_idx] = 41; // Register
            }
        }
    }
}

/// Instantiate CPUs directly into tile_types array - NO SlimSimulation needed!
/// This saves ~6GB of RAM by avoiding the full simulation allocation.
pub fn instantiate_cpus_gpu_direct(
    tile_types: &mut [u8],
    grid_width: usize,
    grid_height: usize,
    count: usize,
    spacing: usize,
) -> Vec<GpuCpuInstance> {
    let cpus_per_row = (grid_width / spacing).max(1);
    let mut instances = Vec::with_capacity(count);

    for i in 0..count {
        let row = i / cpus_per_row;
        let col = i % cpus_per_row;
        let ox = col * spacing;
        let oy = row * spacing;

        // Bounds check
        if ox + 3 >= grid_width || oy + 3 >= grid_height {
            continue;
        }

        let cpu = GpuCpuInstance::new(i, (ox, oy));
        cpu.place_direct(tile_types, grid_width);
        instances.push(cpu);
    }

    instances
}

/// Instantiate CPUs with minimal memory footprint for GPU execution
pub fn instantiate_cpus_gpu_lightweight(
    sim: &mut SlimSimulation,
    count: usize,
    spacing: usize,
) -> Vec<GpuCpuInstance> {
    let grid_width = sim.width();
    let cpus_per_row = (grid_width / spacing).max(1);

    let mut instances = Vec::with_capacity(count);

    for i in 0..count {
        let row = i / cpus_per_row;
        let col = i % cpus_per_row;
        let ox = col * spacing;
        let oy = row * spacing;

        if ox + 3 >= grid_width {
            continue;
        }

        let cpu = GpuCpuInstance::new(i, (ox, oy));
        cpu.place(sim);
        instances.push(cpu);
    }

    instances
}

/// Physical TILE-8 CPU layout positions
///
/// All positions are relative to CPU origin.
/// Uses Const tiles for inputs (to avoid Wire feedback).
#[derive(Debug, Clone)]
pub struct PhysicalCpu {
    /// Origin position in the simulation grid
    pub origin: (usize, usize),

    // === Tile positions (relative to origin) ===
    /// Program Counter value storage
    pub pc: (usize, usize),

    /// Registers R0-R3
    pub regs: [(usize, usize); 4],

    /// ROM - program bytes (up to 16 for minimal demo)
    pub rom: Vec<(usize, usize)>,

    // === ALU Tiles ===
    /// ALU operand A input (shared)
    pub alu_a: (usize, usize),
    /// ALU operand B input (shared)
    pub alu_b: (usize, usize),
    /// ALU Add tile
    pub alu_add: (usize, usize),
    /// ALU Sub tile
    pub alu_sub: (usize, usize),
    /// ALU And tile
    pub alu_and: (usize, usize),
    /// ALU Or tile
    pub alu_or: (usize, usize),
    /// ALU Xor tile
    pub alu_xor: (usize, usize),

    // === Flags ===
    /// Zero flag (Z)
    pub flag_z: (usize, usize),
    /// Carry flag (C)
    pub flag_c: (usize, usize),

    // === Data Memory ===
    /// Data RAM base position for 256-byte memory space
    /// Memory is laid out in a 16x16 grid starting at this position
    /// Address calculation: (base_x + addr % 16, base_y + addr / 16)
    pub data_ram_base: (usize, usize),

    /// Current instruction storage
    pub instruction: (usize, usize),
}

impl PhysicalCpu {
    /// Create a new physical CPU at the given origin
    ///
    /// Layout (Extended for quantum operations):
    /// ```text
    ///      col 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    /// row 0: [PC]
    /// row 1: [ROM0][ROM1][ROM2][ROM3][ROM4][ROM5][ROM6][ROM7]...
    /// row 2: (empty - gap to avoid WE/ROM overlap)
    /// row 3: [WE0][WE1][WE2][WE3]  <- Write enable tiles
    /// row 4: [R0] [R1] [R2] [R3]   <- Register RAM tiles
    /// row 5: [A]  [ADD][B]         <- Each op needs own A/B inputs!
    /// row 6: [A]  [SUB][B]
    /// row 7: [A]  [AND][B]
    /// row 8: [A]  [OR ][B]
    /// row 9: [A]  [XOR][B]
    /// row 10: [Z_FLAG][C_FLAG]     <- Flags
    /// row 11: (reserved)
    /// row 12: (reserved)
    /// row 13: [INST]
    /// row 14+: [Memory Grid]       <- 256-byte memory (16x16 grid)
    ///          rows 14-29 (16 rows)
    ///          cols 0-15  (16 cols)
    ///          Address = row_offset * 16 + col_offset
    /// ```
    ///
    /// Note: Each ALU operation needs its own A/B input tiles because tiles
    /// only read from immediate neighbors.
    ///
    /// Memory Layout (256 bytes for quantum states):
    /// - Addresses 0-127: Real parts of complex amplitudes
    /// - Addresses 128-255: Imaginary parts of complex amplitudes
    pub fn new(origin: (usize, usize)) -> Self {
        let (ox, oy) = origin;

        Self {
            origin,
            pc: (ox, oy),
            rom: (0..16).map(|i| (ox + i, oy + 1)).collect(),
            regs: [
                (ox, oy + 4),     // R0 with WE at oy+3
                (ox + 1, oy + 4), // R1 with WE at oy+3
                (ox + 2, oy + 4), // R2 with WE at oy+3
                (ox + 3, oy + 4), // R3 with WE at oy+3
            ],
            // ALU - each operation has [A][OP][B] layout
            // We use alu_a/alu_b for ADD row, but set all rows in place()
            alu_a: (ox, oy + 5),
            alu_add: (ox + 1, oy + 5),
            alu_b: (ox + 2, oy + 5),
            alu_sub: (ox + 1, oy + 6),
            alu_and: (ox + 1, oy + 7),
            alu_or: (ox + 1, oy + 8),
            alu_xor: (ox + 1, oy + 9),
            // Flags
            flag_z: (ox, oy + 10),
            flag_c: (ox + 1, oy + 10),
            // Data RAM (256 bytes in 16x16 grid)
            // Base address at oy + 12, arranged as:
            // Row 12: addresses 0-15
            // Row 13: addresses 16-31
            // ...
            // Row 27: addresses 240-255
            data_ram_base: (ox, oy + 14), // Start after instruction row
            // Instruction
            instruction: (ox, oy + 13),
        }
    }

    /// Place all CPU tiles into the simulation
    pub fn place(&self, sim: &mut SlimSimulation) {
        // PC - use Const to hold value (we'll update it manually)
        sim.set_tile(self.pc.0, self.pc.1, TileType::Const);

        // ROM - Const tiles for each byte
        for &(x, y) in &self.rom {
            sim.set_tile(x, y, TileType::Const);
        }

        // Registers - RAM tiles
        // RAM: up != 0 ? left : current
        // We need WE (write enable) tiles above each register
        for &(x, y) in self.regs.iter() {
            sim.set_tile(x, y, TileType::Ram);
            // WE tile above (we'll control this to enable writes)
            sim.set_tile(x, y - 1, TileType::Const);
        }

        // ALU operation tiles - each row needs [A][OP][B] layout
        // Row 5: ADD
        sim.set_tile(self.alu_a.0, self.alu_a.1, TileType::Const); // A input
        sim.set_tile(self.alu_add.0, self.alu_add.1, TileType::Add);
        sim.set_tile(self.alu_b.0, self.alu_b.1, TileType::Const); // B input

        // Row 6: SUB (needs its own A/B inputs on same row)
        sim.set_tile(self.alu_sub.0 - 1, self.alu_sub.1, TileType::Const); // A input
        sim.set_tile(self.alu_sub.0, self.alu_sub.1, TileType::Sub);
        sim.set_tile(self.alu_sub.0 + 1, self.alu_sub.1, TileType::Const); // B input

        // Row 7: AND
        sim.set_tile(self.alu_and.0 - 1, self.alu_and.1, TileType::Const); // A input
        sim.set_tile(self.alu_and.0, self.alu_and.1, TileType::And);
        sim.set_tile(self.alu_and.0 + 1, self.alu_and.1, TileType::Const); // B input

        // Row 8: OR
        sim.set_tile(self.alu_or.0 - 1, self.alu_or.1, TileType::Const); // A input
        sim.set_tile(self.alu_or.0, self.alu_or.1, TileType::Or);
        sim.set_tile(self.alu_or.0 + 1, self.alu_or.1, TileType::Const); // B input

        // Row 9: XOR
        sim.set_tile(self.alu_xor.0 - 1, self.alu_xor.1, TileType::Const); // A input
        sim.set_tile(self.alu_xor.0, self.alu_xor.1, TileType::Xor);
        sim.set_tile(self.alu_xor.0 + 1, self.alu_xor.1, TileType::Const); // B input

        // Flags - Const tiles (we update them manually)
        sim.set_tile(self.flag_z.0, self.flag_z.1, TileType::Const);
        sim.set_tile(self.flag_c.0, self.flag_c.1, TileType::Const);

        // Data RAM (256 bytes) - Initialize memory grid with Const tiles
        // This creates a 16x16 grid starting at data_ram_base
        let (base_x, base_y) = self.data_ram_base;
        for row in 0..16 {
            for col in 0..16 {
                let x = base_x + col;
                let y = base_y + row;
                sim.set_tile(x, y, TileType::Const);
                // Initialize to 0
                sim.set_logic_value(x, y, 0);
            }
        }

        // Instruction register - Const tile
        sim.set_tile(self.instruction.0, self.instruction.1, TileType::Const);
    }

    /// Place GPU-compatible CPU layout (simplified for shader execution)
    ///
    /// The GPU shader expects this 3x3 layout:
    /// ```text
    ///       [RegA]
    /// [RegD] [CPU] [RegC]
    ///       [RegB]
    /// ```
    /// Where:
    /// - CPU (CpuHead, type 40) stores the IP and fetches instructions
    /// - RegA-D (Register, type 41) are N/S/E/W neighbors that store values
    pub fn place_gpu(&self, sim: &mut SlimSimulation) {
        let (ox, oy) = self.origin;

        // Ensure we have room for the layout (need at least 3x3 centered at ox+1, oy+1)
        let cpu_x = ox + 1;
        let cpu_y = oy + 1;

        // Place CpuHead at center
        sim.set_tile(cpu_x, cpu_y, TileType::CpuHead);

        // Place Registers as immediate neighbors (matches shader expectations)
        // Reg A (0) = North of CPU
        sim.set_tile(cpu_x, cpu_y - 1, TileType::Register);
        // Reg B (1) = South of CPU
        sim.set_tile(cpu_x, cpu_y + 1, TileType::Register);
        // Reg C (2) = East of CPU
        sim.set_tile(cpu_x + 1, cpu_y, TileType::Register);
        // Reg D (3) = West of CPU
        sim.set_tile(cpu_x - 1, cpu_y, TileType::Register);

        // Update the regs array to match GPU layout (for inspector compatibility)
        // Note: self is immutable here, so we can't update regs.
        // The visualizer should use the GPU positions directly.
    }

    /// Get GPU-compatible register positions (for use with place_gpu layout)
    pub fn gpu_reg_positions(&self) -> [(usize, usize); 4] {
        let (ox, oy) = self.origin;
        let cpu_x = ox + 1;
        let cpu_y = oy + 1;
        [
            (cpu_x, cpu_y - 1), // Reg A = North
            (cpu_x, cpu_y + 1), // Reg B = South
            (cpu_x + 1, cpu_y), // Reg C = East
            (cpu_x - 1, cpu_y), // Reg D = West
        ]
    }

    /// Get GPU CpuHead position
    pub fn gpu_cpu_position(&self) -> (usize, usize) {
        let (ox, oy) = self.origin;
        (ox + 1, oy + 1)
    }

    /// Load a program into ROM
    pub fn load_program(&self, sim: &mut SlimSimulation, program: &[u8]) {
        for (i, &byte) in program.iter().enumerate() {
            if i < self.rom.len() {
                let (x, y) = self.rom[i];
                sim.set_logic_value(x, y, byte as u64);
            }
        }
    }

    /// Get current PC value
    pub fn get_pc(&self, sim: &SlimSimulation) -> u8 {
        sim.get_logic_at(self.pc.0, self.pc.1) as u8
    }

    /// Set PC value
    pub fn set_pc(&self, sim: &mut SlimSimulation, value: u8) {
        sim.set_logic_value(self.pc.0, self.pc.1, value as u64);
    }

    /// Get register value
    pub fn get_reg(&self, sim: &SlimSimulation, reg: usize) -> u8 {
        if reg < 4 {
            let (x, y) = self.regs[reg];
            sim.get_logic_at(x, y) as u8
        } else {
            0
        }
    }

    /// Set register value (direct write, bypasses WE)
    pub fn set_reg(&self, sim: &mut SlimSimulation, reg: usize, value: u8) {
        if reg < 4 {
            let (x, y) = self.regs[reg];
            sim.set_logic_value(x, y, value as u64);
        }
    }

    /// Enable write for a register (set WE high)
    #[allow(dead_code)]
    fn enable_write(&self, sim: &mut SlimSimulation, reg: usize) {
        if reg < 4 {
            let (x, y) = self.regs[reg];
            sim.set_logic_value(x, y - 1, u64::MAX); // WE = 1
        }
    }

    /// Disable write for a register (set WE low)
    #[allow(dead_code)]
    fn disable_write(&self, sim: &mut SlimSimulation, reg: usize) {
        if reg < 4 {
            let (x, y) = self.regs[reg];
            sim.set_logic_value(x, y - 1, 0); // WE = 0
        }
    }

    /// Disable write for all registers
    #[allow(dead_code)]
    fn disable_all_writes(&self, sim: &mut SlimSimulation) {
        for i in 0..4 {
            self.disable_write(sim, i);
        }
    }

    // === Flag accessors ===

    /// Get Zero flag
    pub fn get_flag_z(&self, sim: &SlimSimulation) -> bool {
        sim.get_logic_at(self.flag_z.0, self.flag_z.1) != 0
    }

    /// Set Zero flag
    pub fn set_flag_z(&self, sim: &mut SlimSimulation, value: bool) {
        sim.set_logic_value(self.flag_z.0, self.flag_z.1, if value { 1 } else { 0 });
    }

    /// Get Carry flag
    pub fn get_flag_c(&self, sim: &SlimSimulation) -> bool {
        sim.get_logic_at(self.flag_c.0, self.flag_c.1) != 0
    }

    /// Set Carry flag
    pub fn set_flag_c(&self, sim: &mut SlimSimulation, value: bool) {
        sim.set_logic_value(self.flag_c.0, self.flag_c.1, if value { 1 } else { 0 });
    }

    /// Update flags based on result and operation
    fn update_flags(&self, sim: &mut SlimSimulation, result: u8, carry: bool) {
        self.set_flag_z(sim, result == 0);
        self.set_flag_c(sim, carry);
    }

    // === Memory accessors ===

    /// Get memory value at address (0-255)
    pub fn get_mem(&self, sim: &SlimSimulation, addr: usize) -> u8 {
        if addr < 256 {
            let (base_x, base_y) = self.data_ram_base;
            let x = base_x + (addr % 16);
            let y = base_y + (addr / 16);
            sim.get_logic_at(x, y) as u8
        } else {
            0
        }
    }

    /// Set memory value at address (0-255)
    pub fn set_mem(&self, sim: &mut SlimSimulation, addr: usize, value: u8) {
        if addr < 256 {
            let (base_x, base_y) = self.data_ram_base;
            let x = base_x + (addr % 16);
            let y = base_y + (addr / 16);
            // For quantum operations, directly set the value at the memory location
            // (RAM tiles not needed for every byte - using simulation memory directly)
            sim.set_logic_value(x, y, value as u64);
        }
    }

    /// Get tile locations for the entire quantum block (256 bytes)
    ///
    /// Returns a vector of (x, y) positions for all 256 memory tiles.
    /// Used by QuantumGrid for cross-block communication.
    pub fn get_block_tile_locations(&self) -> Vec<(usize, usize)> {
        let (base_x, base_y) = self.data_ram_base;
        let mut locs = Vec::with_capacity(256);

        for addr in 0..256 {
            let x = base_x + (addr % 16);
            let y = base_y + (addr / 16);
            locs.push((x, y));
        }

        locs
    }

    /// Apply cross-block CNOT using partner CPU's block
    ///
    /// For qubit >= 7, this performs the conditional swap with amplitudes
    /// from the partner CPU's block.
    ///
    /// # Arguments
    /// * `sim` - Simulation to write results to
    /// * `partner_block` - 256-byte block from partner CPU
    /// * `qubit` - Qubit index (must be >= 7)
    pub fn apply_cnot_cross_partner(
        &self,
        sim: &mut SlimSimulation,
        partner_block: &[u8; 256],
        qubit: u8,
    ) {
        use crate::tile8::quantum_ops::BLOCK_SIZE;

        assert!(qubit >= 7, "Cross-block CNOT only for qubits 7+");

        // Calculate which bit within the block this qubit represents
        // qubit 7 -> bit 0 of block_id
        // qubit 8 -> bit 1 of block_id, etc.
        let _block_bit = qubit - 7;

        // For cross-block CNOT, we need to swap amplitudes conditionally
        // based on the control qubit (assumed to be qubit 0)
        //
        // This is a simplified implementation for qubit 7 specifically.
        // TODO: Generalize for arbitrary control/target pairs

        // Read our current block
        let mut our_block = [0u8; 256];
        for addr in 0..256 {
            our_block[addr] = self.get_mem(sim, addr);
        }

        // Apply conditional swap for qubit 7
        // If control bit (qubit 0) is set, swap with partner
        for state in 0..BLOCK_SIZE {
            let control_set = (state & 1) != 0; // qubit 0 is control

            if control_set {
                // Swap real parts
                let _our_re = our_block[state];
                let partner_re = partner_block[state];
                self.set_mem(sim, state, partner_re);

                // Swap imaginary parts
                let _our_im = our_block[state + BLOCK_SIZE];
                let partner_im = partner_block[state + BLOCK_SIZE];
                self.set_mem(sim, state + BLOCK_SIZE, partner_im);
            }
        }
    }

    /// Fetch instruction at current PC
    pub fn fetch(&self, sim: &SlimSimulation) -> u8 {
        let pc = self.get_pc(sim) as usize;
        if pc < self.rom.len() {
            let (x, y) = self.rom[pc];
            sim.get_logic_at(x, y) as u8
        } else {
            0 // NOP
        }
    }

    /// Execute one instruction cycle (SOFTWARE-ONLY, fast)
    ///
    /// This version computes ALU results in software without using
    /// physical tiles or eval_full_grid(). Use this for scale testing.
    pub fn step_fast(&self, sim: &mut SlimSimulation) -> Option<u8> {
        let inst = self.fetch(sim);
        let opcode = (inst >> 4) & 0x0F;
        let rd = ((inst >> 2) & 0x03) as usize;
        let rs_or_imm = (inst & 0x03) as usize;

        match opcode {
            0x0 => {} // NOP
            0x1 => {
                // MOV Rd, Rs
                let val = self.get_reg(sim, rs_or_imm);
                self.set_reg(sim, rd, val);
            }
            0x2 => {
                // ADD Rd, Rs (software)
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);
                let result = a.wrapping_add(b);
                let carry = (a as u16 + b as u16) > 0xFF;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0x3 => {
                // SUB Rd, Rs (software)
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);
                let result = a.wrapping_sub(b);
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, a < b);
            }
            0x4 => {
                // AND Rd, Rs (software)
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);
                let result = a & b;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x5 => {
                // OR Rd, Rs (software)
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);
                let result = a | b;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x6 => {
                // XOR Rd, Rs (software)
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);
                let result = a ^ b;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x7 => {
                // NOT Rd
                let a = self.get_reg(sim, rd);
                let result = !a;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x8 => {
                // SHL Rd
                let a = self.get_reg(sim, rd);
                let carry = (a & 0x80) != 0;
                let result = a << 1;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0x9 => {
                // SHR Rd
                let a = self.get_reg(sim, rd);
                let carry = (a & 0x01) != 0;
                let result = a >> 1;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0xA => {
                // LDI Rd, #imm
                let result = rs_or_imm as u8;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0xB => {
                // JMP
                let addr = self.get_reg(sim, 3);
                self.set_pc(sim, addr);
                return Some(inst);
            }
            0xC => {
                // JZ
                if self.get_flag_z(sim) {
                    let addr = self.get_reg(sim, 3);
                    self.set_pc(sim, addr);
                    return Some(inst);
                }
            }
            0xD => {
                // JNZ
                if !self.get_flag_z(sim) {
                    let addr = self.get_reg(sim, 3);
                    self.set_pc(sim, addr);
                    return Some(inst);
                }
            }
            0xE => {
                // LOAD Rd from MEM[R3]
                let addr = (self.get_reg(sim, 3) & 0x03) as usize;
                let result = self.get_mem(sim, addr);
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0xF => {
                // STORE Rs to MEM[R3]
                let addr = (self.get_reg(sim, 3) & 0x03) as usize;
                let value = self.get_reg(sim, rs_or_imm);
                self.set_mem(sim, addr, value);
            }
            _ => {}
        }

        let pc = self.get_pc(sim);
        self.set_pc(sim, pc.wrapping_add(1));
        Some(inst)
    }

    /// Execute one instruction cycle
    ///
    /// Returns the instruction that was executed, or None if halted.
    pub fn step(&self, sim: &mut SlimSimulation) -> Option<u8> {
        // Fetch
        let inst = self.fetch(sim);
        let opcode = (inst >> 4) & 0x0F;
        let rd = ((inst >> 2) & 0x03) as usize;
        let rs_or_imm = (inst & 0x03) as usize;

        // Decode and execute
        match opcode {
            0x0 => {
                // NOP - do nothing
            }
            0x1 => {
                // MOV Rd, Rs
                let val = self.get_reg(sim, rs_or_imm);
                self.set_reg(sim, rd, val);
            }
            0x2 => {
                // ADD Rd, Rs - use physical Add tile
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);

                // Set ALU inputs
                sim.set_logic_value(self.alu_a.0, self.alu_a.1, a as u64);
                sim.set_logic_value(self.alu_b.0, self.alu_b.1, b as u64);

                // Evaluate ALU (all operations compute in parallel)
                sim.eval_full_grid();

                // Read result from Add tile
                let result = sim.get_logic_at(self.alu_add.0, self.alu_add.1) as u8;
                let carry = (a as u16 + b as u16) > 0xFF;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0x3 => {
                // SUB Rd, Rs - use physical Sub tile
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);

                // Set ALU inputs (Sub row has its own A/B)
                sim.set_logic_value(self.alu_sub.0 - 1, self.alu_sub.1, a as u64);
                sim.set_logic_value(self.alu_sub.0 + 1, self.alu_sub.1, b as u64);

                // Evaluate ALU
                sim.eval_full_grid();

                // Read result from Sub tile
                let result = sim.get_logic_at(self.alu_sub.0, self.alu_sub.1) as u8;
                let borrow = a < b;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, borrow);
            }
            0x4 => {
                // AND Rd, Rs - use physical And tile
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);

                // Set ALU inputs (And row has its own A/B)
                sim.set_logic_value(self.alu_and.0 - 1, self.alu_and.1, a as u64);
                sim.set_logic_value(self.alu_and.0 + 1, self.alu_and.1, b as u64);

                // Evaluate ALU
                sim.eval_full_grid();

                // Read result from And tile
                let result = sim.get_logic_at(self.alu_and.0, self.alu_and.1) as u8;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x5 => {
                // OR Rd, Rs - use physical Or tile
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);

                // Set ALU inputs (Or row has its own A/B)
                sim.set_logic_value(self.alu_or.0 - 1, self.alu_or.1, a as u64);
                sim.set_logic_value(self.alu_or.0 + 1, self.alu_or.1, b as u64);

                // Evaluate ALU
                sim.eval_full_grid();

                // Read result from Or tile
                let result = sim.get_logic_at(self.alu_or.0, self.alu_or.1) as u8;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x6 => {
                // XOR Rd, Rs - use physical Xor tile
                let a = self.get_reg(sim, rd);
                let b = self.get_reg(sim, rs_or_imm);

                // Set ALU inputs (Xor row has its own A/B)
                sim.set_logic_value(self.alu_xor.0 - 1, self.alu_xor.1, a as u64);
                sim.set_logic_value(self.alu_xor.0 + 1, self.alu_xor.1, b as u64);

                // Evaluate ALU
                sim.eval_full_grid();

                // Read result from Xor tile
                let result = sim.get_logic_at(self.alu_xor.0, self.alu_xor.1) as u8;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x7 => {
                // NOT Rd (no physical tile, compute directly)
                let a = self.get_reg(sim, rd);
                let result = !a;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0x8 => {
                // SHL Rd (shift left, carry = MSB)
                let a = self.get_reg(sim, rd);
                let carry = (a & 0x80) != 0;
                let result = a << 1;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0x9 => {
                // SHR Rd (shift right, carry = LSB)
                let a = self.get_reg(sim, rd);
                let carry = (a & 0x01) != 0;
                let result = a >> 1;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, carry);
            }
            0xA => {
                // LDI Rd, #imm
                let result = rs_or_imm as u8;
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0xB => {
                // JMP - jump to address in R3
                let addr = self.get_reg(sim, 3);
                self.set_pc(sim, addr);
                return Some(inst); // Don't increment PC
            }
            0xC => {
                // JZ - jump if Zero flag set
                if self.get_flag_z(sim) {
                    let addr = self.get_reg(sim, 3);
                    self.set_pc(sim, addr);
                    return Some(inst);
                }
            }
            0xD => {
                // JNZ - jump if Zero flag not set
                if !self.get_flag_z(sim) {
                    let addr = self.get_reg(sim, 3);
                    self.set_pc(sim, addr);
                    return Some(inst);
                }
            }
            0xE => {
                // LOAD Rd - load from MEM[R3]
                let addr = (self.get_reg(sim, 3) & 0x03) as usize;
                let result = self.get_mem(sim, addr);
                self.set_reg(sim, rd, result);
                self.update_flags(sim, result, false);
            }
            0xF => {
                // STORE Rs - store Rs to MEM[R3]
                let addr = (self.get_reg(sim, 3) & 0x03) as usize;
                let value = self.get_reg(sim, rs_or_imm);
                self.set_mem(sim, addr, value);
            }
            _ => {}
        }

        // Increment PC
        let pc = self.get_pc(sim);
        self.set_pc(sim, pc.wrapping_add(1));

        Some(inst)
    }

    /// Run for N instructions
    pub fn run(&self, sim: &mut SlimSimulation, n: usize) -> usize {
        let mut executed = 0;
        for _ in 0..n {
            if self.step(sim).is_some() {
                executed += 1;
            } else {
                break;
            }
        }
        executed
    }

    /// Dump CPU state for debugging
    pub fn dump_state(&self, sim: &SlimSimulation) -> String {
        format!(
            "PC={:02X} R0={:02X} R1={:02X} R2={:02X} R3={:02X} Z={} C={}",
            self.get_pc(sim),
            self.get_reg(sim, 0),
            self.get_reg(sim, 1),
            self.get_reg(sim, 2),
            self.get_reg(sim, 3),
            self.get_flag_z(sim) as u8,
            self.get_flag_c(sim) as u8,
        )
    }

    /// Get CPU dimensions (width x height in tiles)
    pub fn dimensions() -> (usize, usize) {
        // Width: max of ROM (16) or data RAM layout (8)
        // Height: 14 rows (0-13)
        (16, 14)
    }
}

// =============================================================================
// MULTI-CPU INSTANCING (Phase 4)
// =============================================================================

/// A handle to an instantiated CPU in the simulation grid
#[derive(Debug, Clone)]
pub struct CpuInstance {
    /// Unique ID for this CPU
    pub id: usize,
    /// The physical CPU layout
    pub cpu: PhysicalCpu,
}

impl CpuInstance {
    /// Read a register value
    pub fn read_register(&self, sim: &SlimSimulation, reg: usize) -> u8 {
        self.cpu.get_reg(sim, reg)
    }

    /// Read program counter
    pub fn read_pc(&self, sim: &SlimSimulation) -> u8 {
        self.cpu.get_pc(sim)
    }

    /// Read memory location
    pub fn read_mem(&self, sim: &SlimSimulation, addr: usize) -> u8 {
        self.cpu.get_mem(sim, addr)
    }

    /// Read flags
    pub fn read_flags(&self, sim: &SlimSimulation) -> (bool, bool) {
        (self.cpu.get_flag_z(sim), self.cpu.get_flag_c(sim))
    }
}

/// Instantiate multiple CPUs in a grid pattern
///
/// # Arguments
/// * `sim` - The simulation to place CPUs in
/// * `count` - Number of CPUs to create
/// * `spacing` - Distance between CPU origins (should be >= CPU width)
/// * `program` - Program to load into each CPU's ROM
///
/// # Returns
/// Vector of CpuInstance handles
///
/// # Example
/// ```ignore
/// let cpus = instantiate_cpus(&mut sim, 100, 20, &program);
/// // Run simulation
/// sim.eval_full_grid();
/// // Check results
/// for cpu in &cpus {
///     println!("CPU {} R0={}", cpu.id, cpu.read_register(&sim, 0));
/// }
/// ```
pub fn instantiate_cpus(
    sim: &mut SlimSimulation,
    count: usize,
    spacing: usize,
    program: &[u8],
) -> Vec<CpuInstance> {
    let (cpu_width, _cpu_height) = PhysicalCpu::dimensions();
    let grid_width = sim.width();

    // Calculate how many CPUs fit per row
    let cpus_per_row = (grid_width / spacing).max(1);

    let mut instances = Vec::with_capacity(count);

    for i in 0..count {
        let row = i / cpus_per_row;
        let col = i % cpus_per_row;

        // Calculate origin position
        let ox = col * spacing;
        let oy = row * spacing;

        // Ensure we stay within grid bounds
        if ox + cpu_width >= grid_width {
            continue;
        }

        let cpu = PhysicalCpu::new((ox, oy));
        cpu.place(sim);
        cpu.load_program(sim, program);

        instances.push(CpuInstance { id: i, cpu });
    }

    instances
}

/// Instantiate CPUs with GPU-compatible layout
///
/// Uses the simplified 3x3 layout (CpuHead + 4 Register neighbors)
/// that the GPU shader understands.
pub fn instantiate_cpus_gpu(
    sim: &mut SlimSimulation,
    count: usize,
    spacing: usize,
    _program: &[u8], // Program is loaded via GPU ROM, not tile values
) -> Vec<CpuInstance> {
    let grid_width = sim.width();

    // Calculate how many CPUs fit per row
    let cpus_per_row = (grid_width / spacing).max(1);

    let mut instances = Vec::with_capacity(count);

    for i in 0..count {
        let row = i / cpus_per_row;
        let col = i % cpus_per_row;

        // Calculate origin position
        let ox = col * spacing;
        let oy = row * spacing;

        // Ensure we stay within grid bounds (need 3x3 = spacing of at least 3)
        if ox + 3 >= grid_width {
            continue;
        }

        let mut cpu = PhysicalCpu::new((ox, oy));
        cpu.place_gpu(sim);

        // Update regs to match GPU layout for inspector compatibility
        cpu.regs = cpu.gpu_reg_positions();

        instances.push(CpuInstance { id: i, cpu });
    }

    instances
}

/// Run all CPUs for N instruction cycles
///
/// This executes the step() function for each CPU sequentially.
/// For true parallel execution, use the grid eval directly.
pub fn run_all_cpus(sim: &mut SlimSimulation, cpus: &[CpuInstance], cycles: usize) {
    for _ in 0..cycles {
        for instance in cpus {
            instance.cpu.step(sim);
        }
    }
}

/// Run all CPUs for N instruction cycles (SOFTWARE-ONLY, fast)
///
/// Uses step_fast() which computes ALU in software without eval_full_grid().
/// Use this for scale testing. ~100x faster than run_all_cpus with ALU ops.
pub fn run_all_cpus_fast(sim: &mut SlimSimulation, cpus: &[CpuInstance], cycles: usize) {
    for _ in 0..cycles {
        for instance in cpus {
            instance.cpu.step_fast(sim);
        }
    }
}

/// Calculate required grid size for N CPUs with given spacing
pub fn calculate_grid_size(count: usize, spacing: usize) -> (usize, usize) {
    let cpus_per_row = ((count as f64).sqrt().ceil() as usize).max(1);
    let rows = count.div_ceil(cpus_per_row);

    let width = cpus_per_row * spacing;
    let height = rows * spacing;

    (width, height)
}

// =============================================================================
// PARALLEL EXECUTION (Rayon)
// =============================================================================

/// Standalone CPU state for parallel execution (no simulation dependency)
#[derive(Clone)]
pub struct CpuState {
    pub pc: u8,
    pub regs: [u8; 4],
    pub flags_z: bool,
    pub flags_c: bool,
    pub rom: [u8; 16],
    pub ram: [u8; 4],
}

impl CpuState {
    /// Create a new CPU state with program
    pub fn new(program: &[u8]) -> Self {
        let mut rom = [0u8; 16];
        for (i, &byte) in program.iter().take(16).enumerate() {
            rom[i] = byte;
        }
        Self {
            pc: 0,
            regs: [0; 4],
            flags_z: false,
            flags_c: false,
            rom,
            ram: [0; 4],
        }
    }

    /// Execute one instruction (pure state, no simulation)
    #[inline]
    pub fn step(&mut self) {
        let inst = self.rom[self.pc as usize & 0x0F];
        let opcode = (inst >> 4) & 0x0F;
        let rd = ((inst >> 2) & 0x03) as usize;
        let rs_or_imm = (inst & 0x03) as usize;

        match opcode {
            0x0 => {} // NOP
            0x1 => {
                // MOV Rd, Rs
                self.regs[rd] = self.regs[rs_or_imm];
            }
            0x2 => {
                // ADD Rd, Rs
                let a = self.regs[rd];
                let b = self.regs[rs_or_imm];
                let result = a.wrapping_add(b);
                let carry = (a as u16 + b as u16) > 0xFF;
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = carry;
            }
            0x3 => {
                // SUB Rd, Rs
                let a = self.regs[rd];
                let b = self.regs[rs_or_imm];
                let result = a.wrapping_sub(b);
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = a < b;
            }
            0x4 => {
                // AND Rd, Rs
                let result = self.regs[rd] & self.regs[rs_or_imm];
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = false;
            }
            0x5 => {
                // OR Rd, Rs
                let result = self.regs[rd] | self.regs[rs_or_imm];
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = false;
            }
            0x6 => {
                // XOR Rd, Rs
                let result = self.regs[rd] ^ self.regs[rs_or_imm];
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = false;
            }
            0x7 => {
                // NOT Rd
                let result = !self.regs[rd];
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = false;
            }
            0x8 => {
                // SHL Rd
                let a = self.regs[rd];
                let carry = (a & 0x80) != 0;
                let result = a << 1;
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = carry;
            }
            0x9 => {
                // SHR Rd
                let a = self.regs[rd];
                let carry = (a & 0x01) != 0;
                let result = a >> 1;
                self.regs[rd] = result;
                self.flags_z = result == 0;
                self.flags_c = carry;
            }
            0xA => {
                // LDI Rd, #imm
                self.regs[rd] = rs_or_imm as u8;
            }
            0xB => {
                // JMP - PC = R3
                self.pc = self.regs[3];
                return; // Don't increment PC
            }
            0xC => {
                // JZ - if Z: PC = R3
                if self.flags_z {
                    self.pc = self.regs[3];
                    return;
                }
            }
            0xD => {
                // JNZ - if !Z: PC = R3
                if !self.flags_z {
                    self.pc = self.regs[3];
                    return;
                }
            }
            0xE => {
                // LOAD Rd = RAM[R3 & 0x03]
                let addr = (self.regs[3] & 0x03) as usize;
                self.regs[rd] = self.ram[addr];
            }
            0xF => {
                // STORE RAM[R3 & 0x03] = Rs
                let addr = (self.regs[3] & 0x03) as usize;
                self.ram[addr] = self.regs[rs_or_imm];
            }
            _ => {}
        }

        // Increment PC (wrap at 16)
        self.pc = (self.pc + 1) & 0x0F;
    }
}

/// Create CPU states from instances (for parallel execution)
pub fn create_cpu_states(count: usize, program: &[u8]) -> Vec<CpuState> {
    (0..count).map(|_| CpuState::new(program)).collect()
}

/// Run CPUs in parallel using Rayon
pub fn run_parallel(states: &mut [CpuState], cycles: usize) {
    for _ in 0..cycles {
        states.par_iter_mut().for_each(|state| {
            state.step();
        });
    }
}

/// Run CPUs in parallel - single cycle (useful for per-cycle analysis)
pub fn step_parallel(states: &mut [CpuState]) {
    states.par_iter_mut().for_each(|state| {
        state.step();
    });
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_cpu_creation() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Verify tiles were placed
        assert_eq!(cpu.origin, (4, 4));
        assert_eq!(cpu.pc, (4, 4));
    }

    #[test]
    fn test_ldi_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // LDI R0, #2  -> 0xA0 | (0 << 2) | 2 = 0xA2
        // LDI R1, #3  -> 0xA0 | (1 << 2) | 3 = 0xA7
        let program = [0xA2, 0xA7];
        cpu.load_program(&mut sim, &program);

        // Initial state
        assert_eq!(cpu.get_pc(&sim), 0);
        assert_eq!(cpu.get_reg(&sim, 0), 0);
        assert_eq!(cpu.get_reg(&sim, 1), 0);

        // Execute LDI R0, #2
        cpu.step(&mut sim);
        assert_eq!(cpu.get_pc(&sim), 1);
        assert_eq!(cpu.get_reg(&sim, 0), 2);

        // Execute LDI R1, #3
        cpu.step(&mut sim);
        assert_eq!(cpu.get_pc(&sim), 2);
        assert_eq!(cpu.get_reg(&sim, 1), 3);
    }

    #[test]
    fn test_add_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // LDI R0, #2  -> 0xA2
        // LDI R1, #3  -> 0xA7
        // ADD R0, R1  -> 0x20 | (0 << 2) | 1 = 0x21
        let program = [0xA2, 0xA7, 0x21];
        cpu.load_program(&mut sim, &program);

        // Run all 3 instructions
        cpu.run(&mut sim, 3);

        // R0 should be 2 + 3 = 5
        assert_eq!(cpu.get_reg(&sim, 0), 5);
        assert_eq!(cpu.get_reg(&sim, 1), 3);
        assert_eq!(cpu.get_pc(&sim), 3);
    }

    #[test]
    fn test_add_with_alu_tile() {
        // Verify the ALU Add tile is actually being used
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Set up registers manually
        cpu.set_reg(&mut sim, 0, 100);
        cpu.set_reg(&mut sim, 1, 55);

        // ADD R0, R1 -> 0x21
        let program = [0x21];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 should be 100 + 55 = 155
        assert_eq!(cpu.get_reg(&sim, 0), 155);
    }

    #[test]
    fn test_mov_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // LDI R0, #3  -> 0xA3
        // MOV R1, R0  -> 0x10 | (1 << 2) | 0 = 0x14
        let program = [0xA3, 0x14];
        cpu.load_program(&mut sim, &program);

        cpu.run(&mut sim, 2);

        assert_eq!(cpu.get_reg(&sim, 0), 3);
        assert_eq!(cpu.get_reg(&sim, 1), 3);
    }

    #[test]
    fn test_sub_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Set up: R0 = 10, R1 = 3
        cpu.set_reg(&mut sim, 0, 10);
        cpu.set_reg(&mut sim, 1, 3);

        // SUB R0, R1 -> 0x30 | (0 << 2) | 1 = 0x31
        let program = [0x31];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 should be 10 - 3 = 7
        assert_eq!(cpu.get_reg(&sim, 0), 7);
    }

    #[test]
    fn test_fibonacci_manual() {
        // Compute Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Program:
        // LDI R0, #0    ; fib(0) = 0   -> 0xA0
        // LDI R1, #1    ; fib(1) = 1   -> 0xA5
        // Loop:
        //   MOV R2, R1  ; temp = b     -> 0x19  (rd=2, rs=1)
        //   ADD R1, R0  ; b = a + b    -> 0x24  (rd=1, rs=0)
        //   MOV R0, R2  ; a = temp     -> 0x12  (rd=0, rs=2)
        //   (loop back manually by resetting PC)

        let program = [
            0xA0, // LDI R0, #0
            0xA5, // LDI R1, #1
            0x19, // MOV R2, R1: 0x10 | (2<<2) | 1 = 0x19
            0x24, // ADD R1, R0: 0x20 | (1<<2) | 0 = 0x24
            0x12, // MOV R0, R2: 0x10 | (0<<2) | 2 = 0x12
        ];
        cpu.load_program(&mut sim, &program);

        // Initial setup
        cpu.run(&mut sim, 2);
        assert_eq!(cpu.get_reg(&sim, 0), 0); // fib(0)
        assert_eq!(cpu.get_reg(&sim, 1), 1); // fib(1)

        // First iteration: fib(2) = 1
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 1); // was fib(1)
        assert_eq!(cpu.get_reg(&sim, 1), 1); // fib(2) = 0+1

        // Reset PC for next iteration
        cpu.set_pc(&mut sim, 2);
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 1); // was fib(2)
        assert_eq!(cpu.get_reg(&sim, 1), 2); // fib(3) = 1+1

        // Continue...
        cpu.set_pc(&mut sim, 2);
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 2); // was fib(3)
        assert_eq!(cpu.get_reg(&sim, 1), 3); // fib(4) = 1+2

        cpu.set_pc(&mut sim, 2);
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 3); // was fib(4)
        assert_eq!(cpu.get_reg(&sim, 1), 5); // fib(5) = 2+3

        cpu.set_pc(&mut sim, 2);
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 5); // was fib(5)
        assert_eq!(cpu.get_reg(&sim, 1), 8); // fib(6) = 3+5
    }

    #[test]
    fn test_all_registers() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // LDI R0, #0 -> 0xA0
        // LDI R1, #1 -> 0xA5
        // LDI R2, #2 -> 0xAA
        // LDI R3, #3 -> 0xAF
        let program = [0xA0, 0xA5, 0xAA, 0xAF];
        cpu.load_program(&mut sim, &program);

        cpu.run(&mut sim, 4);

        assert_eq!(cpu.get_reg(&sim, 0), 0);
        assert_eq!(cpu.get_reg(&sim, 1), 1);
        assert_eq!(cpu.get_reg(&sim, 2), 2);
        assert_eq!(cpu.get_reg(&sim, 3), 3);
    }

    #[test]
    fn test_dump_state() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        cpu.set_pc(&mut sim, 0x10);
        cpu.set_reg(&mut sim, 0, 0xAA);
        cpu.set_reg(&mut sim, 1, 0xBB);
        cpu.set_reg(&mut sim, 2, 0xCC);
        cpu.set_reg(&mut sim, 3, 0xDD);

        let state = cpu.dump_state(&sim);
        assert!(state.contains("PC=10"));
        assert!(state.contains("R0=AA"));
        assert!(state.contains("R1=BB"));
        assert!(state.contains("R2=CC"));
        assert!(state.contains("R3=DD"));
    }

    // ==========================================================================
    // Phase 2 Tests: Full ALU + Flags
    // ==========================================================================

    #[test]
    fn test_and_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 0b11110000, R1 = 0b10101010
        cpu.set_reg(&mut sim, 0, 0xF0);
        cpu.set_reg(&mut sim, 1, 0xAA);

        // AND R0, R1 -> 0x40 | (0 << 2) | 1 = 0x41
        let program = [0x41];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 0xF0 & 0xAA = 0xA0
        assert_eq!(cpu.get_reg(&sim, 0), 0xA0);
    }

    #[test]
    fn test_or_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 0b11110000, R1 = 0b00001111
        cpu.set_reg(&mut sim, 0, 0xF0);
        cpu.set_reg(&mut sim, 1, 0x0F);

        // OR R0, R1 -> 0x50 | (0 << 2) | 1 = 0x51
        let program = [0x51];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 0xF0 | 0x0F = 0xFF
        assert_eq!(cpu.get_reg(&sim, 0), 0xFF);
    }

    #[test]
    fn test_xor_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 0xFF, R1 = 0xAA
        cpu.set_reg(&mut sim, 0, 0xFF);
        cpu.set_reg(&mut sim, 1, 0xAA);

        // XOR R0, R1 -> 0x60 | (0 << 2) | 1 = 0x61
        let program = [0x61];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 0xFF ^ 0xAA = 0x55
        assert_eq!(cpu.get_reg(&sim, 0), 0x55);
    }

    #[test]
    fn test_not_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        cpu.set_reg(&mut sim, 0, 0xAA);

        // NOT R0 -> 0x70 | (0 << 2) = 0x70
        let program = [0x70];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = !0xAA = 0x55
        assert_eq!(cpu.get_reg(&sim, 0), 0x55);
    }

    #[test]
    fn test_shl_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        cpu.set_reg(&mut sim, 0, 0x55); // 0b01010101

        // SHL R0 -> 0x80 | (0 << 2) = 0x80
        let program = [0x80];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 0x55 << 1 = 0xAA
        assert_eq!(cpu.get_reg(&sim, 0), 0xAA);
        // MSB was 0, so no carry
        assert!(!cpu.get_flag_c(&sim));
    }

    #[test]
    fn test_shr_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        cpu.set_reg(&mut sim, 0, 0xAA); // 0b10101010

        // SHR R0 -> 0x90 | (0 << 2) = 0x90
        let program = [0x90];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 0xAA >> 1 = 0x55
        assert_eq!(cpu.get_reg(&sim, 0), 0x55);
        // LSB was 0, so no carry
        assert!(!cpu.get_flag_c(&sim));
    }

    #[test]
    fn test_zero_flag() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 5, R1 = 5
        cpu.set_reg(&mut sim, 0, 5);
        cpu.set_reg(&mut sim, 1, 5);

        // SUB R0, R1 -> 0x31
        let program = [0x31];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = 5 - 5 = 0
        assert_eq!(cpu.get_reg(&sim, 0), 0);
        // Zero flag should be set
        assert!(cpu.get_flag_z(&sim));
    }

    #[test]
    fn test_carry_flag_on_add() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 200, R1 = 100 (sum = 300 > 255)
        cpu.set_reg(&mut sim, 0, 200);
        cpu.set_reg(&mut sim, 1, 100);

        // ADD R0, R1 -> 0x21
        let program = [0x21];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = (200 + 100) & 0xFF = 44
        assert_eq!(cpu.get_reg(&sim, 0), 44);
        // Carry flag should be set (overflow)
        assert!(cpu.get_flag_c(&sim));
    }

    #[test]
    fn test_borrow_flag_on_sub() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 5, R1 = 10 (5 - 10 = -5, underflow)
        cpu.set_reg(&mut sim, 0, 5);
        cpu.set_reg(&mut sim, 1, 10);

        // SUB R0, R1 -> 0x31
        let program = [0x31];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 = (5 - 10) & 0xFF = 251
        assert_eq!(cpu.get_reg(&sim, 0), 251);
        // Carry (borrow) flag should be set
        assert!(cpu.get_flag_c(&sim));
    }

    #[test]
    fn test_jz_taken() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Set Z flag by doing 0 - 0
        cpu.set_reg(&mut sim, 0, 0);
        cpu.set_reg(&mut sim, 1, 0);
        cpu.set_reg(&mut sim, 3, 5); // Jump target

        // SUB R0, R1 -> sets Z flag
        // JZ -> should jump to R3 (address 5)
        let program = [0x31, 0xC0]; // SUB R0,R1; JZ
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim); // SUB
        assert!(cpu.get_flag_z(&sim));

        cpu.step(&mut sim); // JZ
        assert_eq!(cpu.get_pc(&sim), 5); // Jumped
    }

    #[test]
    fn test_jz_not_taken() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Set non-zero result
        cpu.set_reg(&mut sim, 0, 5);
        cpu.set_reg(&mut sim, 1, 1);
        cpu.set_reg(&mut sim, 3, 10); // Jump target

        // SUB R0, R1 -> R0 = 4, Z not set
        // JZ -> should NOT jump
        let program = [0x31, 0xC0]; // SUB R0,R1; JZ
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim); // SUB
        assert!(!cpu.get_flag_z(&sim));

        cpu.step(&mut sim); // JZ
        assert_eq!(cpu.get_pc(&sim), 2); // Continued to next instruction
    }

    #[test]
    fn test_jnz_taken() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Set non-zero result
        cpu.set_reg(&mut sim, 0, 5);
        cpu.set_reg(&mut sim, 1, 1);
        cpu.set_reg(&mut sim, 3, 8); // Jump target

        // SUB R0, R1 -> R0 = 4, Z not set
        // JNZ -> should jump
        let program = [0x31, 0xD0]; // SUB R0,R1; JNZ
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim); // SUB
        cpu.step(&mut sim); // JNZ
        assert_eq!(cpu.get_pc(&sim), 8); // Jumped
    }

    #[test]
    fn test_countdown_loop() {
        // A simple countdown loop: count from 3 to 0
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Program:
        // 0: LDI R0, #3     ; counter = 3   (0xA3)
        // 1: LDI R1, #1     ; decrement     (0xA5)
        // 2: LDI R3, #2     ; loop addr     (0xAE) - but R3=2
        // Wait, LDI can only load 0-3. Let me use: loop at addr 3
        // 0: LDI R0, #3     ; counter = 3   (0xA3)
        // 1: LDI R1, #1     ; decrement     (0xA5)
        // 2: LDI R3, #3     ; loop addr=3   (0xAF)
        // 3: SUB R0, R1     ; counter--     (0x31)
        // 4: JNZ            ; loop if != 0  (0xD0)
        // 5: NOP            ; done

        let program = [
            0xA3, // LDI R0, #3
            0xA5, // LDI R1, #1
            0xAF, // LDI R3, #3
            0x31, // SUB R0, R1
            0xD0, // JNZ
            0x00, // NOP (exit point)
        ];
        cpu.load_program(&mut sim, &program);

        // Run setup (3 instructions)
        cpu.run(&mut sim, 3);
        assert_eq!(cpu.get_reg(&sim, 0), 3);
        assert_eq!(cpu.get_reg(&sim, 3), 3);

        // First iteration: 3 -> 2
        cpu.run(&mut sim, 2); // SUB, JNZ
        assert_eq!(cpu.get_reg(&sim, 0), 2);
        assert_eq!(cpu.get_pc(&sim), 3); // Looped back

        // Second iteration: 2 -> 1
        cpu.run(&mut sim, 2);
        assert_eq!(cpu.get_reg(&sim, 0), 1);
        assert_eq!(cpu.get_pc(&sim), 3); // Looped back

        // Third iteration: 1 -> 0
        cpu.run(&mut sim, 2);
        assert_eq!(cpu.get_reg(&sim, 0), 0);
        assert_eq!(cpu.get_pc(&sim), 5); // Exited loop (JNZ not taken)
    }

    // ==========================================================================
    // Phase 3 Tests: Memory (LOAD/STORE)
    // ==========================================================================

    #[test]
    fn test_store_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // R0 = 42, R3 = 0 (address)
        cpu.set_reg(&mut sim, 0, 42);
        cpu.set_reg(&mut sim, 3, 0);

        // STORE R0 to MEM[R3] -> 0xF0 | (0 << 2) | 0 = 0xF0
        let program = [0xF0];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // Memory at address 0 should be 42
        assert_eq!(cpu.get_mem(&sim, 0), 42);
    }

    #[test]
    fn test_load_instruction() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Pre-load memory with value
        cpu.set_mem(&mut sim, 1, 99);

        // R3 = 1 (address), R0 will receive loaded value
        cpu.set_reg(&mut sim, 3, 1);

        // LOAD R0 from MEM[R3] -> 0xE0 | (0 << 2) = 0xE0
        let program = [0xE0];
        cpu.load_program(&mut sim, &program);

        cpu.step(&mut sim);

        // R0 should now contain 99
        assert_eq!(cpu.get_reg(&sim, 0), 99);
    }

    #[test]
    fn test_store_then_load() {
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Store value, then load it back to different register
        // R0 = 77, R3 = 2 (address)
        cpu.set_reg(&mut sim, 0, 77);
        cpu.set_reg(&mut sim, 3, 2);

        // STORE R0 to MEM[2], then LOAD R1 from MEM[2]
        // STORE R0: 0xF0
        // LOAD R1: 0xE4 (0xE0 | (1 << 2))
        let program = [0xF0, 0xE4];
        cpu.load_program(&mut sim, &program);

        cpu.run(&mut sim, 2);

        // R1 should now contain 77 (loaded from memory)
        assert_eq!(cpu.get_reg(&sim, 1), 77);
        assert_eq!(cpu.get_mem(&sim, 2), 77);
    }

    #[test]
    fn test_memory_array_operations() {
        // Test storing and loading from all 4 memory locations
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Store values 10, 20, 30, 40 to addresses 0, 1, 2, 3
        for i in 0..4u8 {
            cpu.set_mem(&mut sim, i as usize, (i + 1) * 10);
        }

        // Verify all values
        assert_eq!(cpu.get_mem(&sim, 0), 10);
        assert_eq!(cpu.get_mem(&sim, 1), 20);
        assert_eq!(cpu.get_mem(&sim, 2), 30);
        assert_eq!(cpu.get_mem(&sim, 3), 40);
    }

    #[test]
    fn test_memory_swap() {
        // Swap two memory locations using registers
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize: MEM[0] = 100, MEM[1] = 200
        cpu.set_mem(&mut sim, 0, 100);
        cpu.set_mem(&mut sim, 1, 200);

        // Program to swap MEM[0] and MEM[1]:
        // LDI R3, #0   ; addr = 0          (0xAC)
        // LOAD R0      ; R0 = MEM[0] = 100 (0xE0)
        // LDI R3, #1   ; addr = 1          (0xAD)
        // LOAD R1      ; R1 = MEM[1] = 200 (0xE4)
        // LDI R3, #0   ; addr = 0          (0xAC)
        // STORE R1     ; MEM[0] = 200      (0xF1)
        // LDI R3, #1   ; addr = 1          (0xAD)
        // STORE R0     ; MEM[1] = 100      (0xF0)
        let program = [
            0xAC, // LDI R3, #0
            0xE0, // LOAD R0
            0xAD, // LDI R3, #1
            0xE4, // LOAD R1
            0xAC, // LDI R3, #0
            0xF1, // STORE R1
            0xAD, // LDI R3, #1
            0xF0, // STORE R0
        ];
        cpu.load_program(&mut sim, &program);

        cpu.run(&mut sim, 8);

        // Verify swap
        assert_eq!(cpu.get_mem(&sim, 0), 200);
        assert_eq!(cpu.get_mem(&sim, 1), 100);
    }

    // ==========================================================================
    // Phase 4 Tests: Multi-CPU Instancing
    // ==========================================================================

    #[test]
    fn test_instantiate_4_cpus() {
        // Create 4 CPUs in a 2x2 grid
        let (width, height) = calculate_grid_size(4, 20);
        let mut sim = SlimSimulation::with_size(width, height);

        // Simple program: LDI R0, #3
        let program = [0xA3]; // LDI R0, #3

        let cpus = instantiate_cpus(&mut sim, 4, 20, &program);
        assert_eq!(cpus.len(), 4);

        // Run all CPUs for 1 cycle
        run_all_cpus(&mut sim, &cpus, 1);

        // All CPUs should have R0 = 3
        for cpu in &cpus {
            assert_eq!(cpu.read_register(&sim, 0), 3);
        }
    }

    #[test]
    fn test_multi_cpu_independent_state() {
        // Create 2 CPUs with different initial values
        let mut sim = SlimSimulation::with_size(64, 32);

        // Program: just NOP (do nothing)
        let program = [0x00];

        let cpus = instantiate_cpus(&mut sim, 2, 32, &program);
        assert_eq!(cpus.len(), 2);

        // Set different values in each CPU's registers
        cpus[0].cpu.set_reg(&mut sim, 0, 10);
        cpus[1].cpu.set_reg(&mut sim, 0, 20);

        // Verify independent state
        assert_eq!(cpus[0].read_register(&sim, 0), 10);
        assert_eq!(cpus[1].read_register(&sim, 0), 20);
    }

    #[test]
    fn test_multi_cpu_fibonacci() {
        // 4 CPUs all computing Fibonacci
        let (width, height) = calculate_grid_size(4, 20);
        let mut sim = SlimSimulation::with_size(width, height);

        // Fibonacci program (first 2 values)
        let program = [
            0xA0, // LDI R0, #0  (fib(0))
            0xA5, // LDI R1, #1  (fib(1))
        ];

        let cpus = instantiate_cpus(&mut sim, 4, 20, &program);

        // Run 2 cycles
        run_all_cpus(&mut sim, &cpus, 2);

        // All should have R0=0, R1=1
        for cpu in &cpus {
            assert_eq!(cpu.read_register(&sim, 0), 0);
            assert_eq!(cpu.read_register(&sim, 1), 1);
        }
    }

    #[test]
    fn test_instantiate_100_cpus() {
        // Scale test: 100 CPUs
        let (width, height) = calculate_grid_size(100, 20);
        let mut sim = SlimSimulation::with_size(width, height);

        // Program: ADD R0, R0 (doubles R0)
        // Start with R0 = 1, after 3 iterations: 1->2->4->8
        let program = [
            0xA1, // LDI R0, #1
            0x20, // ADD R0, R0
            0x20, // ADD R0, R0
            0x20, // ADD R0, R0
        ];

        let cpus = instantiate_cpus(&mut sim, 100, 20, &program);
        assert_eq!(cpus.len(), 100);

        // Run all 4 instructions
        run_all_cpus(&mut sim, &cpus, 4);

        // All should have R0 = 8 (1 * 2 * 2 * 2)
        for (i, cpu) in cpus.iter().enumerate() {
            assert_eq!(cpu.read_register(&sim, 0), 8, "CPU {} failed", i);
        }
    }

    #[test]
    fn test_calculate_grid_size() {
        // 4 CPUs with spacing 20 -> 2x2 grid = 40x40
        assert_eq!(calculate_grid_size(4, 20), (40, 40));

        // 9 CPUs with spacing 20 -> 3x3 grid = 60x60
        assert_eq!(calculate_grid_size(9, 20), (60, 60));

        // 10 CPUs with spacing 20 -> 4x3 grid = 80x60
        assert_eq!(calculate_grid_size(10, 20), (80, 60));

        // 100 CPUs with spacing 20 -> 10x10 grid = 200x200
        assert_eq!(calculate_grid_size(100, 20), (200, 200));
    }

    #[test]
    fn test_multi_cpu_countdown() {
        // 4 CPUs doing countdown from 3 to 0
        let (width, height) = calculate_grid_size(4, 20);
        let mut sim = SlimSimulation::with_size(width, height);

        let program = [
            0xA3, // LDI R0, #3  (counter)
            0xA5, // LDI R1, #1  (decrement)
            0x31, // SUB R0, R1
            0x31, // SUB R0, R1
            0x31, // SUB R0, R1
        ];

        let cpus = instantiate_cpus(&mut sim, 4, 20, &program);

        // Run 5 cycles
        run_all_cpus(&mut sim, &cpus, 5);

        // All should have R0 = 0
        for cpu in &cpus {
            assert_eq!(cpu.read_register(&sim, 0), 0);
        }
    }

    #[test]
    fn test_1000_cpus() {
        // Scale test: 1000 CPUs
        let (width, height) = calculate_grid_size(1000, 20);
        let mut sim = SlimSimulation::with_size(width, height);

        // Simple program
        let program = [0xA2]; // LDI R0, #2

        let cpus = instantiate_cpus(&mut sim, 1000, 20, &program);

        // Run 1 cycle
        run_all_cpus(&mut sim, &cpus, 1);

        // Sample check (not all 1000 to keep test fast)
        assert_eq!(cpus[0].read_register(&sim, 0), 2);
        assert_eq!(cpus[500].read_register(&sim, 0), 2);
        assert_eq!(cpus[999].read_register(&sim, 0), 2);
    }

    // ==========================================================================
    // Phase 5 Tests: Scale Testing
    // ==========================================================================

    #[test]
    fn test_10000_cpus() {
        // Scale test: 10,000 CPUs with ALU operations
        let (width, height) = calculate_grid_size(10_000, 20);
        println!(
            "Grid size for 10K CPUs: {}x{} = {} tiles",
            width,
            height,
            width * height
        );

        let mut sim = SlimSimulation::with_size(width, height);

        // Program with ALU: 1 + 1 = 2
        let program = [
            0xA1, // LDI R0, #1
            0xA5, // LDI R1, #1
            0x21, // ADD R0, R1  -> R0 = 2
        ];

        let start = std::time::Instant::now();
        let cpus = instantiate_cpus(&mut sim, 10_000, 20, &program);
        let place_time = start.elapsed();

        assert_eq!(cpus.len(), 10_000);
        println!("Placed 10K CPUs in {:?}", place_time);

        let start = std::time::Instant::now();
        run_all_cpus_fast(&mut sim, &cpus, 3); // Use fast (software) mode
        let run_time = start.elapsed();
        println!("Ran 3 cycles (with ADD) on 10K CPUs in {:?}", run_time);

        // Sample verification
        for i in [0, 1000, 5000, 9999] {
            assert_eq!(cpus[i].read_register(&sim, 0), 2, "CPU {} R0 failed", i);
        }
    }

    #[test]
    fn test_100000_cpus() {
        // Scale test: 100,000 CPUs
        let (width, height) = calculate_grid_size(100_000, 20);
        println!(
            "Grid size for 100K CPUs: {}x{} = {} tiles",
            width,
            height,
            width * height
        );

        let mut sim = SlimSimulation::with_size(width, height);

        // Fibonacci-like: 1+1=2, 2+1=3
        let program = [
            0xA1, // LDI R0, #1
            0xA5, // LDI R1, #1
            0x21, // ADD R0, R1  -> R0=2
            0x20, // ADD R0, R0  -> R0=4
        ];

        let start = std::time::Instant::now();
        let cpus = instantiate_cpus(&mut sim, 100_000, 20, &program);
        let place_time = start.elapsed();
        println!("Placed 100K CPUs in {:?}", place_time);

        let start = std::time::Instant::now();
        run_all_cpus_fast(&mut sim, &cpus, 4);
        let run_time = start.elapsed();
        println!("Ran 4 cycles on 100K CPUs in {:?}", run_time);

        // Sample verification
        assert_eq!(cpus[0].read_register(&sim, 0), 4);
        assert_eq!(cpus[50_000].read_register(&sim, 0), 4);
        assert_eq!(cpus[99_999].read_register(&sim, 0), 4);
    }

    #[test]
    fn test_300000_cpus() {
        // Architecture target: 302,000 CPUs @ 500 tiles each
        // From TILE8_REVISED_ARCHITECTURE.md: "Maximum CPUs: 302,000"
        let (width, height) = calculate_grid_size(300_000, 20);
        println!(
            "Grid size for 300K CPUs: {}x{} = {} tiles",
            width,
            height,
            width * height
        );

        let mut sim = SlimSimulation::with_size(width, height);

        // Simple program: LDI R0, #3; ADD R0, R0 -> R0=6
        let program = [
            0xA3, // LDI R0, #3
            0x20, // ADD R0, R0  -> R0=6
        ];

        let start = std::time::Instant::now();
        let cpus = instantiate_cpus(&mut sim, 300_000, 20, &program);
        let place_time = start.elapsed();
        println!("Placed 300K CPUs in {:?}", place_time);

        let start = std::time::Instant::now();
        run_all_cpus_fast(&mut sim, &cpus, 2);
        let run_time = start.elapsed();
        println!("Ran 2 cycles on 300K CPUs in {:?}", run_time);
        println!(
            "That's {} CPU-cycles/second!",
            600_000.0 / run_time.as_secs_f64()
        );

        // Sample verification
        assert_eq!(cpus[0].read_register(&sim, 0), 6);
        assert_eq!(cpus[150_000].read_register(&sim, 0), 6);
        assert_eq!(cpus[299_999].read_register(&sim, 0), 6);
    }

    #[test]
    #[ignore] // Very large test - run with: cargo test test_1000000_cpus -- --ignored --nocapture
    fn test_1000000_cpus() {
        // Scale test: 1,000,000 CPUs (1 MILLION!)
        let (width, height) = calculate_grid_size(1_000_000, 20);
        println!(
            "Grid size for 1M CPUs: {}x{} = {} tiles",
            width,
            height,
            width * height
        );

        let mut sim = SlimSimulation::with_size(width, height);

        let program = [
            0xA2, // LDI R0, #2
            0x20, // ADD R0, R0  -> R0=4
        ];

        let start = std::time::Instant::now();
        let cpus = instantiate_cpus(&mut sim, 1_000_000, 20, &program);
        let place_time = start.elapsed();
        println!("Placed 1M CPUs in {:?}", place_time);

        let start = std::time::Instant::now();
        run_all_cpus_fast(&mut sim, &cpus, 2);
        let run_time = start.elapsed();
        println!("Ran 2 cycles on 1M CPUs in {:?}", run_time);
        println!(
            "That's {} CPU-cycles/second!",
            2_000_000.0 / run_time.as_secs_f64()
        );

        // Sample verification
        assert_eq!(cpus[0].read_register(&sim, 0), 4);
        assert_eq!(cpus[500_000].read_register(&sim, 0), 4);
        assert_eq!(cpus[999_999].read_register(&sim, 0), 4);
    }

    // ==========================================================================
    // Parallel Execution Tests (Rayon)
    // ==========================================================================

    #[test]
    fn test_parallel_basic() {
        // Basic test: verify parallel execution produces correct results
        let program = [
            0xA2, // LDI R0, #2
            0x20, // ADD R0, R0  -> R0=4
        ];

        let mut states = create_cpu_states(1000, &program);
        run_parallel(&mut states, 2);

        // All CPUs should have R0 = 4
        for (i, state) in states.iter().enumerate() {
            assert_eq!(state.regs[0], 4, "CPU {} R0 failed", i);
        }
    }

    #[test]
    fn test_parallel_fibonacci() {
        // Fibonacci: 1, 1, 2, 3, 5, 8...
        let program = [
            0xA1, // LDI R0, #1     (fib_prev = 1)
            0xA5, // LDI R1, #1     (fib_curr = 1)
            0x18, // MOV R2, R0     (temp = fib_prev)
            0x10, // MOV R0, R1     (fib_prev = fib_curr)
            0x26, // ADD R1, R2     (fib_curr += temp)
        ];

        let mut states = create_cpu_states(100, &program);
        run_parallel(&mut states, 5); // 5 instructions

        // After 5 instructions: R0=1, R1=2
        for state in &states {
            assert_eq!(state.regs[0], 1);
            assert_eq!(state.regs[1], 2);
        }
    }

    #[test]
    fn test_parallel_1m_cpus() {
        // 1 MILLION CPUs in parallel!
        let program = [
            0xA3, // LDI R0, #3
            0x20, // ADD R0, R0  -> R0=6
        ];

        let start = std::time::Instant::now();
        let mut states = create_cpu_states(1_000_000, &program);
        let create_time = start.elapsed();
        println!("Created 1M CPU states in {:?}", create_time);

        let start = std::time::Instant::now();
        run_parallel(&mut states, 2);
        let run_time = start.elapsed();

        let total_cycles = 2_000_000u64;
        let ips = total_cycles as f64 / run_time.as_secs_f64();
        println!("Ran 2 cycles on 1M CPUs (parallel) in {:?}", run_time);
        println!("That's {:.2}M IPS ({:.2} GIPS)", ips / 1e6, ips / 1e9);

        // Sample verification
        assert_eq!(states[0].regs[0], 6);
        assert_eq!(states[500_000].regs[0], 6);
        assert_eq!(states[999_999].regs[0], 6);
    }

    #[test]
    fn test_parallel_10m_cpus() {
        // 10 MILLION CPUs - pushing for GIPS!
        let program = [
            0xA2, // LDI R0, #2
            0x20, // ADD R0, R0  -> R0=4
            0x20, // ADD R0, R0  -> R0=8
            0x20, // ADD R0, R0  -> R0=16
        ];

        let start = std::time::Instant::now();
        let mut states = create_cpu_states(10_000_000, &program);
        let create_time = start.elapsed();
        println!("Created 10M CPU states in {:?}", create_time);

        let start = std::time::Instant::now();
        run_parallel(&mut states, 4);
        let run_time = start.elapsed();

        let total_cycles = 40_000_000u64;
        let ips = total_cycles as f64 / run_time.as_secs_f64();
        println!("Ran 4 cycles on 10M CPUs (parallel) in {:?}", run_time);
        println!("That's {:.2}M IPS ({:.3} GIPS)", ips / 1e6, ips / 1e9);

        // Sample verification
        assert_eq!(states[0].regs[0], 16);
        assert_eq!(states[5_000_000].regs[0], 16);
        assert_eq!(states[9_999_999].regs[0], 16);
    }

    #[test]
    fn test_parallel_gips_attempt() {
        // Attempt to hit 1 GIPS with optimal settings
        let program = [
            0xA1, // LDI R0, #1
            0x20, // ADD R0, R0  -> R0=2
            0x20, // ADD R0, R0  -> R0=4
            0x20, // ADD R0, R0  -> R0=8
            0x20, // ADD R0, R0  -> R0=16
            0x20, // ADD R0, R0  -> R0=32
            0x20, // ADD R0, R0  -> R0=64
            0x20, // ADD R0, R0  -> R0=128
        ];

        let count = 1_000_000; // 1M CPUs
        let cycles = 100; // More cycles to amortize overhead

        let mut states = create_cpu_states(count, &program);

        // Warmup
        run_parallel(&mut states, 1);

        // Benchmark
        let start = std::time::Instant::now();
        run_parallel(&mut states, cycles);
        let run_time = start.elapsed();

        let total_cycles = (count * cycles) as u64;
        let ips = total_cycles as f64 / run_time.as_secs_f64();
        println!(
            "{}M CPUs × {} cycles = {} total instructions",
            count / 1_000_000,
            cycles,
            total_cycles
        );
        println!("Time: {:?}", run_time);
        println!("Throughput: {:.2}M IPS ({:.3} GIPS)", ips / 1e6, ips / 1e9);

        // Verify (wraps around due to 8-bit overflow and ROM wrap)
        // Just check all CPUs have same value
        let expected = states[0].regs[0];
        assert_eq!(states[count - 1].regs[0], expected);
    }

    #[test]
    fn test_parallel_vs_sequential() {
        // Compare parallel vs sequential performance
        let program = [
            0xA1, // LDI R0, #1
            0x20, // ADD R0, R0  -> R0=2
        ];

        let count = 100_000;
        let cycles = 10;

        // Sequential (single-threaded)
        let mut states_seq = create_cpu_states(count, &program);
        let start = std::time::Instant::now();
        for _ in 0..cycles {
            for state in states_seq.iter_mut() {
                state.step();
            }
        }
        let seq_time = start.elapsed();

        // Parallel (Rayon)
        let mut states_par = create_cpu_states(count, &program);
        let start = std::time::Instant::now();
        run_parallel(&mut states_par, cycles);
        let par_time = start.elapsed();

        // Verify both produce same result
        assert_eq!(states_seq[0].regs[0], states_par[0].regs[0]);

        let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();
        println!("100K CPUs × {} cycles:", cycles);
        println!("  Sequential: {:?}", seq_time);
        println!("  Parallel:   {:?}", par_time);
        println!("  Speedup:    {:.2}x", speedup);
    }
}
