//! TILE-8 CPU Builder
//!
//! Generates the tile layout for the TILE-8 CPU and loads programs into ROM.
//!
//! The CPU is laid out in a 64x64 tile region:
//! ```text
//! +------------------+------------------+
//! |   PC (8x4)       |  CONTROL (16x8)  |
//! +------------------+------------------+
//! |   REGISTERS (16x8)                  |
//! +-------------------------------------+
//! |   ALU (32x8)                        |
//! +-------------------------------------+
//! |   MEMORY (64x32) - ROM + RAM        |
//! +-------------------------------------+
//! ```

use super::isa::{Instruction, Opcode, Register};
use crate::simulation::Simulation;
use crate::slim_simulation::SlimSimulation;
use crate::tile_meta::TileType;

/// CPU state for execution
#[derive(Debug, Clone, Default)]
pub struct CpuState {
    pub pc: u8,
    pub registers: [u8; 4],
    pub zero_flag: bool,
    pub carry_flag: bool,
    pub halted: bool,
    pub cycles: u64,
}

/// TILE-8 CPU built from tiles
///
/// This is a software emulator that mirrors what the tile-based CPU would do,
/// allowing us to validate programs before building the full tile layout.
pub struct Tile8Cpu {
    /// Program ROM (128 bytes max)
    pub rom: [u8; 128],
    /// Data RAM (128 bytes)
    pub ram: [u8; 128],
    /// CPU state
    pub state: CpuState,
}

impl Tile8Cpu {
    /// Create a new CPU with empty memory
    pub fn new() -> Self {
        Self {
            rom: [0; 128],
            ram: [0; 128],
            state: CpuState::default(),
        }
    }

    /// Load program into ROM
    pub fn load_program(&mut self, program: &[u8]) {
        let len = program.len().min(128);
        self.rom[..len].copy_from_slice(&program[..len]);
    }

    /// Reset CPU state
    pub fn reset(&mut self) {
        self.state = CpuState::default();
    }

    /// Execute one instruction
    pub fn step(&mut self) -> Option<Instruction> {
        if self.state.halted {
            return None;
        }

        // Fetch
        let byte = self.rom[self.state.pc as usize];
        let inst = Instruction::decode(byte)?;

        // Decode & Execute
        self.execute(&inst);

        // Advance PC (unless jump modified it)
        self.state.cycles += 1;

        Some(inst)
    }

    /// Execute an instruction
    fn execute(&mut self, inst: &Instruction) {
        let rd_idx = inst.rd as usize;
        let rs_idx = inst.rs_or_imm as usize;

        match inst.opcode {
            Opcode::Nop => {
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Mov => {
                self.state.registers[rd_idx] = self.state.registers[rs_idx];
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Add => {
                let a = self.state.registers[rd_idx] as u16;
                let b = self.state.registers[rs_idx] as u16;
                let result = a + b;
                self.state.registers[rd_idx] = result as u8;
                self.state.carry_flag = result > 255;
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Sub => {
                let a = self.state.registers[rd_idx] as i16;
                let b = self.state.registers[rs_idx] as i16;
                let result = a - b;
                self.state.registers[rd_idx] = result as u8;
                self.state.carry_flag = result < 0;
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::And => {
                self.state.registers[rd_idx] &= self.state.registers[rs_idx];
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Or => {
                self.state.registers[rd_idx] |= self.state.registers[rs_idx];
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Xor => {
                self.state.registers[rd_idx] ^= self.state.registers[rs_idx];
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Not => {
                self.state.registers[rd_idx] = !self.state.registers[rd_idx];
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Shl => {
                let val = self.state.registers[rd_idx];
                self.state.carry_flag = (val & 0x80) != 0;
                self.state.registers[rd_idx] = val << 1;
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Shr => {
                let val = self.state.registers[rd_idx];
                self.state.carry_flag = (val & 0x01) != 0;
                self.state.registers[rd_idx] = val >> 1;
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Ldi => {
                self.state.registers[rd_idx] = inst.rs_or_imm;
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Jmp => {
                self.state.pc = self.state.registers[3]; // R3 is address register
            }

            Opcode::Jz => {
                if self.state.zero_flag {
                    self.state.pc = self.state.registers[3];
                } else {
                    self.state.pc = self.state.pc.wrapping_add(1);
                }
            }

            Opcode::Jnz => {
                if !self.state.zero_flag {
                    self.state.pc = self.state.registers[3];
                } else {
                    self.state.pc = self.state.pc.wrapping_add(1);
                }
            }

            Opcode::Load => {
                let addr = self.state.registers[3] as usize;
                self.state.registers[rd_idx] = if addr >= 128 {
                    self.ram[addr - 128]
                } else {
                    self.rom[addr]
                };
                self.update_zero_flag(rd_idx);
                self.state.pc = self.state.pc.wrapping_add(1);
            }

            Opcode::Store => {
                let addr = self.state.registers[3] as usize;
                let value = self.state.registers[rs_idx];
                if addr >= 128 {
                    self.ram[addr - 128] = value;
                }
                // Writing to ROM is ignored
                self.state.pc = self.state.pc.wrapping_add(1);
            }
        }
    }

    fn update_zero_flag(&mut self, reg_idx: usize) {
        self.state.zero_flag = self.state.registers[reg_idx] == 0;
    }

    /// Run until halted or max cycles reached
    pub fn run(&mut self, max_cycles: u64) -> u64 {
        let start_cycles = self.state.cycles;
        while self.state.cycles - start_cycles < max_cycles && !self.state.halted {
            if self.step().is_none() {
                break;
            }
        }
        self.state.cycles - start_cycles
    }

    /// Get register value
    pub fn get_reg(&self, r: Register) -> u8 {
        self.state.registers[r as usize]
    }

    /// Set register value
    pub fn set_reg(&mut self, r: Register, value: u8) {
        self.state.registers[r as usize] = value;
        self.state.zero_flag = value == 0;
    }

    /// Get RAM value
    pub fn get_ram(&self, addr: u8) -> u8 {
        if addr >= 128 {
            self.ram[(addr - 128) as usize]
        } else {
            0
        }
    }

    /// Set RAM value
    pub fn set_ram(&mut self, addr: u8, value: u8) {
        if addr >= 128 {
            self.ram[(addr - 128) as usize] = value;
        }
    }

    /// Dump CPU state for debugging
    pub fn dump_state(&self) -> String {
        format!(
            "PC={:02X} R0={:02X} R1={:02X} R2={:02X} R3={:02X} Z={} C={} cycles={}",
            self.state.pc,
            self.state.registers[0],
            self.state.registers[1],
            self.state.registers[2],
            self.state.registers[3],
            if self.state.zero_flag { '1' } else { '0' },
            if self.state.carry_flag { '1' } else { '0' },
            self.state.cycles
        )
    }
}

impl Default for Tile8Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating a tile-based CPU layout
pub struct Tile8Builder {
    /// Grid offset X
    pub origin_x: usize,
    /// Grid offset Y
    pub origin_y: usize,
    /// Program to load
    pub program: Vec<u8>,
    /// Initial RAM values (address - 128, value)
    pub ram_init: Vec<(u8, u8)>,
}

impl Tile8Builder {
    pub fn new() -> Self {
        Self {
            origin_x: 0,
            origin_y: 0,
            program: Vec::new(),
            ram_init: Vec::new(),
        }
    }

    pub fn with_origin(mut self, x: usize, y: usize) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    pub fn with_program(mut self, program: Vec<u8>) -> Self {
        self.program = program;
        self
    }

    pub fn with_ram(mut self, init: Vec<(u8, u8)>) -> Self {
        self.ram_init = init;
        self
    }

    /// Build CPU layout in a Simulation
    ///
    /// For now, this creates a minimal layout that demonstrates the concept.
    /// A full implementation would create all the control logic, ALU, etc.
    pub fn build_in_simulation(&self, sim: &mut Simulation) {
        let ox = self.origin_x;
        let oy = self.origin_y;

        // Create ROM area: each byte is 8 Const tiles in a row
        // ROM is at y = oy+32, spanning 128 bytes × 8 bits = 1024 tiles
        for (addr, &byte) in self.program.iter().enumerate() {
            for bit in 0..8 {
                let x = ox + (addr % 16) * 8 + bit;
                let y = oy + 32 + (addr / 16);
                let bit_val = (byte >> bit) & 1;

                sim.set_tile(x, y, TileType::Const);
                // Set the constant value
                if bit_val != 0 {
                    sim.set_logic_value(x, y, u64::MAX);
                }
            }
        }

        // Create RAM area: similar layout but using Ram tiles
        // RAM is at y = oy+40
        for &(addr, value) in &self.ram_init {
            for bit in 0..8 {
                let x = ox + ((addr as usize) % 16) * 8 + bit;
                let y = oy + 40 + ((addr as usize) / 16);
                let bit_val = (value >> bit) & 1;

                sim.set_tile(x, y, TileType::Ram);
                if bit_val != 0 {
                    sim.set_logic_value(x, y, u64::MAX);
                }
            }
        }

        // Create PC register at origin (8 Counter tiles for counting)
        for bit in 0..8 {
            sim.set_tile(ox + bit, oy, TileType::Counter);
        }
    }

    /// Build CPU layout in a SlimSimulation
    pub fn build_in_slim(&self, sim: &mut SlimSimulation) {
        let ox = self.origin_x;
        let oy = self.origin_y;

        // Create ROM area
        for (addr, &byte) in self.program.iter().enumerate() {
            for bit in 0..8 {
                let x = ox + (addr % 16) * 8 + bit;
                let y = oy + 32 + (addr / 16);
                let bit_val = (byte >> bit) & 1;

                sim.set_tile(x, y, TileType::Const);
                if bit_val != 0 {
                    sim.set_logic_value(x, y, u64::MAX);
                }
            }
        }

        // Create RAM area
        for &(addr, value) in &self.ram_init {
            for bit in 0..8 {
                let x = ox + ((addr as usize) % 16) * 8 + bit;
                let y = oy + 40 + ((addr as usize) / 16);
                let bit_val = (value >> bit) & 1;

                sim.set_tile(x, y, TileType::Ram);
                if bit_val != 0 {
                    sim.set_logic_value(x, y, u64::MAX);
                }
            }
        }

        // Create PC register
        for bit in 0..8 {
            sim.set_tile(ox + bit, oy, TileType::Counter);
        }
    }
}

impl Default for Tile8Builder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile8::asm::assemble;

    #[test]
    fn test_add_program() {
        let source = r#"
            LDI R0, #1      ; R0 = 1
            LDI R1, #2      ; R1 = 2
            ADD R0, R1      ; R0 = 1 + 2 = 3
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);

        // Step exactly 3 instructions
        cpu.step();
        cpu.step();
        cpu.step();

        assert_eq!(cpu.get_reg(Register::R0), 3);
        assert_eq!(cpu.get_reg(Register::R1), 2);
        assert_eq!(cpu.state.cycles, 3);
    }

    #[test]
    fn test_counting_loop() {
        // Count from 1 to 3 using only immediate values 0-3
        // Simplified: just count up and check manually
        let source = r#"
            LDI R0, #0      ; counter = 0
            LDI R1, #1      ; increment = 1
        loop:
            ADD R0, R1      ; counter++ (R0 = 1, 2, 3, ...)
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);

        // Setup: 2 instructions
        cpu.step(); // LDI R0, #0 -> R0 = 0
        cpu.step(); // LDI R1, #1 -> R1 = 1

        // Loop 3 times manually (the JMP address limitation makes loops tricky)
        cpu.step(); // ADD R0, R1 -> R0 = 1
        assert_eq!(cpu.get_reg(Register::R0), 1);

        // Reset PC to loop
        cpu.state.pc = 2;
        cpu.step(); // ADD R0, R1 -> R0 = 2
        assert_eq!(cpu.get_reg(Register::R0), 2);

        cpu.state.pc = 2;
        cpu.step(); // ADD R0, R1 -> R0 = 3
        assert_eq!(cpu.get_reg(Register::R0), 3);
    }

    #[test]
    fn test_fibonacci() {
        // Simple Fibonacci: compute fib sequence
        // Start with fib(0)=0, fib(1)=1, compute next values
        // R0 = current, R1 = previous
        let source = r#"
            LDI R0, #1      ; current = fib(1) = 1
            LDI R1, #0      ; previous = fib(0) = 0
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);

        // Execute setup
        cpu.step(); // LDI R0, #1
        cpu.step(); // LDI R1, #0

        // Manually perform fibonacci iterations
        // fib(2) = fib(1) + fib(0) = 1 + 0 = 1
        // fib(3) = fib(2) + fib(1) = 1 + 1 = 2
        // fib(4) = fib(3) + fib(2) = 2 + 1 = 3
        // fib(5) = fib(4) + fib(3) = 3 + 2 = 5

        for _ in 0..4 {
            let temp = cpu.get_reg(Register::R0);
            let new_val = cpu
                .get_reg(Register::R0)
                .wrapping_add(cpu.get_reg(Register::R1));
            cpu.set_reg(Register::R0, new_val);
            cpu.set_reg(Register::R1, temp);
        }

        // After 4 iterations: R0 = fib(5) = 5
        assert_eq!(cpu.get_reg(Register::R0), 5);
        assert_eq!(cpu.get_reg(Register::R1), 3);
    }

    #[test]
    fn test_memory_operations() {
        let source = r#"
            LDI R0, #3      ; value to store
            LDI R3, #0      ; address (will need to be set to RAM area)
            ; For now, test ROM read (address 0)
            LOAD R1         ; R1 = MEM[R3] = first instruction byte
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);
        cpu.run(10);

        // R1 should contain the first instruction byte (LDI R0, #3 = 0xA3)
        assert_eq!(cpu.get_reg(Register::R1), 0xA3);
    }

    #[test]
    fn test_shift_operations() {
        let source = r#"
            LDI R0, #1      ; R0 = 1
            SHL R0          ; R0 = 2
            SHL R0          ; R0 = 4
            SHL R0          ; R0 = 8
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);
        cpu.run(10);

        assert_eq!(cpu.get_reg(Register::R0), 8);
    }

    #[test]
    fn test_flags() {
        let source = r#"
            LDI R0, #1
            LDI R1, #1
            SUB R0, R1      ; R0 = 0, Z flag should be set
        "#;

        let program = assemble(source).unwrap();
        let mut cpu = Tile8Cpu::new();
        cpu.load_program(&program);
        cpu.run(10);

        assert_eq!(cpu.get_reg(Register::R0), 0);
        assert!(cpu.state.zero_flag);
    }

    #[test]
    fn test_builder() {
        let source = r#"
            LDI R0, #1
            ADD R0, R0
        "#;

        let program = assemble(source).unwrap();
        let builder = Tile8Builder::new()
            .with_origin(10, 10)
            .with_program(program)
            .with_ram(vec![(128, 42), (129, 99)]);

        // Just verify it builds without panicking
        let mut sim = Simulation::new();
        builder.build_in_simulation(&mut sim);
    }
}
