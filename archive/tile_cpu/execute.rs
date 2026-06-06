//! TileCpu Execution Engine
//!
//! This module contains the core execution logic that runs the CPU by
//! calling the tile simulation engine. There is NO software emulation -
//! the CPU executes by propagating signals through tiles.

use crate::simulation::{Simulation, TimingStats};
use crate::tile_cpu::{MAX_PROPAGATION_DELTAS, NUM_REGISTERS, DATAPATH_WIDTH};

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
}

impl std::fmt::Display for TileCpuMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TileCPU Execution Metrics")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Cycles executed:     {:>12}", self.cycles)?;
        writeln!(f, "Instructions:        {:>12}", self.instructions_executed)?;
        writeln!(f, "IPC:                 {:>12.3}", self.ipc)?;
        writeln!(f, "Max critical path:   {:>12} deltas", self.max_critical_path)?;
        writeln!(f, "Avg critical path:   {:>12.1} deltas", self.avg_critical_path)?;
        writeln!(f, "Est. max frequency:  {:>12.1} MHz", self.estimated_max_mhz)?;
        if self.had_timing_violation {
            writeln!(f, "WARNING: Timing violations detected!")?;
        }
        Ok(())
    }
}

/// A CPU that executes via tile simulation
///
/// This struct holds tile indices for key CPU components. All execution
/// happens by reading/writing tile state through the Simulation.
#[derive(Debug, Clone)]
pub struct TileCpu {
    /// Origin position on the tile grid
    pub origin: (usize, usize),

    /// Tile index of the Program Counter
    pc_idx: usize,

    /// Tile indices of the 4 general-purpose registers
    reg_indices: [usize; NUM_REGISTERS],

    /// Tile indices of the ROM words (Const tiles)
    rom_indices: Vec<usize>,

    /// Tile indices of the RAM bytes (Ram tiles)
    ram_indices: Vec<usize>,

    /// Tile index of the Zero flag
    flag_z_idx: usize,

    /// Tile index of the Carry flag
    flag_c_idx: usize,

    /// Tile indices of ALU output (one per bit)
    alu_out_indices: [usize; DATAPATH_WIDTH],

    /// Grid width (for index calculations)
    grid_width: usize,

    /// Total tiles used by this CPU
    pub tile_count: usize,
}

impl TileCpu {
    /// Create a TileCpu from pre-placed tile indices
    ///
    /// This is called by TileCpuBuilder after placing all tiles.
    pub(crate) fn new(
        origin: (usize, usize),
        pc_idx: usize,
        reg_indices: [usize; NUM_REGISTERS],
        rom_indices: Vec<usize>,
        ram_indices: Vec<usize>,
        flag_z_idx: usize,
        flag_c_idx: usize,
        alu_out_indices: [usize; DATAPATH_WIDTH],
        grid_width: usize,
        tile_count: usize,
    ) -> Self {
        Self {
            origin,
            pc_idx,
            reg_indices,
            rom_indices,
            ram_indices,
            flag_z_idx,
            flag_c_idx,
            alu_out_indices,
            grid_width,
            tile_count,
        }
    }

    /// Execute one clock cycle via tile simulation
    ///
    /// This is the core execution method. It:
    /// 1. Toggles the clock (done inside tick_with_delays)
    /// 2. Propagates combinational logic until stable
    /// 3. Sequential elements capture on clock edge
    ///
    /// Returns timing statistics for this cycle.
    pub fn tick(&self, sim: &mut Simulation) -> TimingStats {
        // tick_with_delays toggles the clock and propagates until stable
        // Sequential elements (registers, PC, RAM) capture on the clock edge
        sim.tick_with_delays()
    }

    /// Run for N cycles, collecting metrics
    pub fn run(&self, sim: &mut Simulation, max_cycles: u64) -> TileCpuMetrics {
        let mut metrics = TileCpuMetrics::default();
        let mut total_critical = 0u64;

        for cycle in 0..max_cycles {
            let stats = self.tick(sim);

            metrics.cycles = cycle + 1;
            metrics.total_deltas += stats.total_deltas as u64;
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
    /// Currently checks if PC is pointing to a NOP (0x00) - placeholder
    /// for proper HALT detection.
    pub fn is_halted(&self, sim: &Simulation) -> bool {
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
    pub fn read_reg(&self, sim: &Simulation, reg: usize) -> u8 {
        if reg < NUM_REGISTERS {
            sim.get_logic_value_by_idx(self.reg_indices[reg]) as u8
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
    pub fn read_ram(&self, sim: &Simulation, addr: usize) -> u8 {
        if addr < self.ram_indices.len() {
            sim.get_logic_value_by_idx(self.ram_indices[addr]) as u8
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

    /// Read ALU output (8-bit value from individual bit tiles)
    pub fn read_alu_out(&self, sim: &Simulation) -> u8 {
        let mut result = 0u8;
        for (bit, &idx) in self.alu_out_indices.iter().enumerate() {
            if sim.get_logic_value_by_idx(idx) != 0 {
                result |= 1 << bit;
            }
        }
        result
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
    pub fn write_reg(&self, sim: &mut Simulation, reg: usize, value: u8) {
        if reg < NUM_REGISTERS {
            sim.set_logic_value_by_idx(self.reg_indices[reg], value as u64);
        }
    }

    /// Set PC value (for initialization or jumps)
    pub fn write_pc(&self, sim: &mut Simulation, value: u8) {
        sim.set_logic_value_by_idx(self.pc_idx, value as u64);
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
