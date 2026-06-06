//! TILE-8 Instruction Set Architecture
//!
//! Encoding: 8-bit instructions
//! ```text
//! +---+---+---+---+---+---+---+---+
//! | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
//! +---+---+---+---+---+---+---+---+
//! |   OPCODE  |  Rd   |  Rs/Imm   |
//! | (4 bits)  |(2 bit)| (2 bits)  |
//! +---+---+---+---+---+---+---+---+
//! ```

use std::fmt;

/// 4 general-purpose registers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Register {
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
}

impl Register {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v & 0x03 {
            0 => Some(Register::R0),
            1 => Some(Register::R1),
            2 => Some(Register::R2),
            3 => Some(Register::R3),
            _ => None,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "R0" => Some(Register::R0),
            "R1" => Some(Register::R1),
            "R2" => Some(Register::R2),
            "R3" => Some(Register::R3),
            _ => None,
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::R0 => write!(f, "R0"),
            Register::R1 => write!(f, "R1"),
            Register::R2 => write!(f, "R2"),
            Register::R3 => write!(f, "R3"),
        }
    }
}

/// 16 opcodes (4 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0x0,   // No operation
    Mov = 0x1,   // Rd = Rs
    Add = 0x2,   // Rd = Rd + Rs
    Sub = 0x3,   // Rd = Rd - Rs
    And = 0x4,   // Rd = Rd & Rs
    Or = 0x5,    // Rd = Rd | Rs
    Xor = 0x6,   // Rd = Rd ^ Rs
    Not = 0x7,   // Rd = ~Rd
    Shl = 0x8,   // Rd = Rd << 1
    Shr = 0x9,   // Rd = Rd >> 1
    Ldi = 0xA,   // Rd = imm (0-3)
    Jmp = 0xB,   // PC = R3
    Jz = 0xC,    // if Z: PC = R3
    Jnz = 0xD,   // if !Z: PC = R3
    Load = 0xE,  // Rd = MEM[R3]
    Store = 0xF, // MEM[R3] = Rs
}

impl Opcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v & 0x0F {
            0x0 => Some(Opcode::Nop),
            0x1 => Some(Opcode::Mov),
            0x2 => Some(Opcode::Add),
            0x3 => Some(Opcode::Sub),
            0x4 => Some(Opcode::And),
            0x5 => Some(Opcode::Or),
            0x6 => Some(Opcode::Xor),
            0x7 => Some(Opcode::Not),
            0x8 => Some(Opcode::Shl),
            0x9 => Some(Opcode::Shr),
            0xA => Some(Opcode::Ldi),
            0xB => Some(Opcode::Jmp),
            0xC => Some(Opcode::Jz),
            0xD => Some(Opcode::Jnz),
            0xE => Some(Opcode::Load),
            0xF => Some(Opcode::Store),
            _ => None,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "NOP" => Some(Opcode::Nop),
            "MOV" => Some(Opcode::Mov),
            "ADD" => Some(Opcode::Add),
            "SUB" => Some(Opcode::Sub),
            "AND" => Some(Opcode::And),
            "OR" => Some(Opcode::Or),
            "XOR" => Some(Opcode::Xor),
            "NOT" => Some(Opcode::Not),
            "SHL" => Some(Opcode::Shl),
            "SHR" => Some(Opcode::Shr),
            "LDI" => Some(Opcode::Ldi),
            "JMP" => Some(Opcode::Jmp),
            "JZ" => Some(Opcode::Jz),
            "JNZ" => Some(Opcode::Jnz),
            "LOAD" => Some(Opcode::Load),
            "STORE" => Some(Opcode::Store),
            _ => None,
        }
    }

    /// Returns true if this opcode uses Rs (source register)
    pub fn uses_rs(&self) -> bool {
        matches!(
            self,
            Opcode::Mov
                | Opcode::Add
                | Opcode::Sub
                | Opcode::And
                | Opcode::Or
                | Opcode::Xor
                | Opcode::Store
        )
    }

    /// Returns true if this opcode uses Rd (destination register)
    pub fn uses_rd(&self) -> bool {
        matches!(
            self,
            Opcode::Mov
                | Opcode::Add
                | Opcode::Sub
                | Opcode::And
                | Opcode::Or
                | Opcode::Xor
                | Opcode::Not
                | Opcode::Shl
                | Opcode::Shr
                | Opcode::Ldi
                | Opcode::Load
        )
    }

    /// Returns true if this opcode uses an immediate value
    pub fn uses_imm(&self) -> bool {
        matches!(self, Opcode::Ldi)
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Opcode::Nop => "NOP",
            Opcode::Mov => "MOV",
            Opcode::Add => "ADD",
            Opcode::Sub => "SUB",
            Opcode::And => "AND",
            Opcode::Or => "OR",
            Opcode::Xor => "XOR",
            Opcode::Not => "NOT",
            Opcode::Shl => "SHL",
            Opcode::Shr => "SHR",
            Opcode::Ldi => "LDI",
            Opcode::Jmp => "JMP",
            Opcode::Jz => "JZ",
            Opcode::Jnz => "JNZ",
            Opcode::Load => "LOAD",
            Opcode::Store => "STORE",
        };
        write!(f, "{}", name)
    }
}

/// A decoded instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: Opcode,
    pub rd: Register,  // Destination register (bits 3-2)
    pub rs_or_imm: u8, // Source register or immediate (bits 1-0)
}

impl Instruction {
    /// Create a new instruction
    pub fn new(opcode: Opcode, rd: Register, rs_or_imm: u8) -> Self {
        Self {
            opcode,
            rd,
            rs_or_imm: rs_or_imm & 0x03,
        }
    }

    /// Encode instruction to single byte
    pub fn encode(&self) -> u8 {
        let opcode_bits = (self.opcode as u8) << 4;
        let rd_bits = (self.rd as u8) << 2;
        let rs_bits = self.rs_or_imm & 0x03;
        opcode_bits | rd_bits | rs_bits
    }

    /// Decode instruction from single byte
    pub fn decode(byte: u8) -> Option<Self> {
        let opcode = Opcode::from_u8(byte >> 4)?;
        let rd = Register::from_u8((byte >> 2) & 0x03)?;
        let rs_or_imm = byte & 0x03;
        Some(Self {
            opcode,
            rd,
            rs_or_imm,
        })
    }

    /// Get source register (if applicable)
    pub fn rs(&self) -> Option<Register> {
        if self.opcode.uses_rs() {
            Register::from_u8(self.rs_or_imm)
        } else {
            None
        }
    }

    /// Get immediate value (if applicable)
    pub fn imm(&self) -> Option<u8> {
        if self.opcode.uses_imm() {
            Some(self.rs_or_imm)
        } else {
            None
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode {
            Opcode::Nop => write!(f, "NOP"),
            Opcode::Jmp => write!(f, "JMP"),
            Opcode::Jz => write!(f, "JZ"),
            Opcode::Jnz => write!(f, "JNZ"),
            Opcode::Not | Opcode::Shl | Opcode::Shr => {
                write!(f, "{} {}", self.opcode, self.rd)
            }
            Opcode::Ldi => {
                write!(f, "LDI {}, #{}", self.rd, self.rs_or_imm)
            }
            Opcode::Load => {
                write!(f, "LOAD {}", self.rd)
            }
            Opcode::Store => {
                let rs = Register::from_u8(self.rs_or_imm).unwrap_or(Register::R0);
                write!(f, "STORE {}", rs)
            }
            _ => {
                let rs = Register::from_u8(self.rs_or_imm).unwrap_or(Register::R0);
                write!(f, "{} {}, {}", self.opcode, self.rd, rs)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        // NOP
        let nop = Instruction::new(Opcode::Nop, Register::R0, 0);
        assert_eq!(nop.encode(), 0x00);
        assert_eq!(Instruction::decode(0x00), Some(nop));

        // ADD R1, R2
        let add = Instruction::new(Opcode::Add, Register::R1, 2);
        assert_eq!(add.encode(), 0x26); // 0010_01_10
        assert_eq!(Instruction::decode(0x26), Some(add));

        // LDI R3, #3
        let ldi = Instruction::new(Opcode::Ldi, Register::R3, 3);
        assert_eq!(ldi.encode(), 0xAF); // 1010_11_11
        assert_eq!(Instruction::decode(0xAF), Some(ldi));

        // JMP (uses R3)
        let jmp = Instruction::new(Opcode::Jmp, Register::R0, 0);
        assert_eq!(jmp.encode(), 0xB0); // 1011_00_00
    }

    #[test]
    fn test_all_opcodes_encode() {
        for op in 0..16u8 {
            let opcode = Opcode::from_u8(op).unwrap();
            let inst = Instruction::new(opcode, Register::R0, 0);
            let encoded = inst.encode();
            let decoded = Instruction::decode(encoded).unwrap();
            assert_eq!(decoded.opcode, opcode);
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(
            format!("{}", Instruction::new(Opcode::Add, Register::R1, 2)),
            "ADD R1, R2"
        );
        assert_eq!(
            format!("{}", Instruction::new(Opcode::Ldi, Register::R0, 3)),
            "LDI R0, #3"
        );
        assert_eq!(
            format!("{}", Instruction::new(Opcode::Shl, Register::R2, 0)),
            "SHL R2"
        );
    }
}
