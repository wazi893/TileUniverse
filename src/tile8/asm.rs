//! TILE-8 Assembler
//!
//! Converts assembly text to binary machine code.
//!
//! Syntax:
//! ```asm
//! ; Comment
//! label:
//!     MOV R0, R1      ; Rd = Rs
//!     ADD R0, R1      ; Rd = Rd + Rs
//!     LDI R0, #3      ; Rd = immediate (0-3)
//!     LOAD R0         ; Rd = MEM[R3]
//!     STORE R1        ; MEM[R3] = Rs
//!     JMP             ; PC = R3
//!     JZ              ; if Zero: PC = R3
//!     JNZ             ; if !Zero: PC = R3
//! ```

use super::isa::{Instruction, Opcode, Register};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmError {
    UnknownOpcode(String),
    InvalidRegister(String),
    InvalidImmediate(String),
    MissingOperand(String),
    UnexpectedOperand(String),
    UndefinedLabel(String),
    DuplicateLabel(String),
    InvalidSyntax(String),
    ProgramTooLarge(usize),
}

impl std::fmt::Display for AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsmError::UnknownOpcode(s) => write!(f, "Unknown opcode: {}", s),
            AsmError::InvalidRegister(s) => write!(f, "Invalid register: {}", s),
            AsmError::InvalidImmediate(s) => write!(f, "Invalid immediate: {}", s),
            AsmError::MissingOperand(s) => write!(f, "Missing operand: {}", s),
            AsmError::UnexpectedOperand(s) => write!(f, "Unexpected operand: {}", s),
            AsmError::UndefinedLabel(s) => write!(f, "Undefined label: {}", s),
            AsmError::DuplicateLabel(s) => write!(f, "Duplicate label: {}", s),
            AsmError::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
            AsmError::ProgramTooLarge(n) => write!(f, "Program too large: {} bytes (max 128)", n),
        }
    }
}

impl std::error::Error for AsmError {}

/// Assembly result with binary and metadata
#[derive(Debug, Clone)]
pub struct AssemblyResult {
    /// Assembled binary code
    pub code: Vec<u8>,
    /// Label addresses
    pub labels: HashMap<String, u8>,
    /// Source line for each byte (for debugging)
    pub source_map: Vec<usize>,
}

/// TILE-8 Assembler
pub struct Assembler {
    /// Current address during assembly
    address: u8,
    /// Label definitions (name -> address)
    labels: HashMap<String, u8>,
    /// Forward references to resolve (address -> label name)
    forward_refs: Vec<(u8, String)>,
    /// Assembled code
    code: Vec<u8>,
    /// Source line mapping
    source_map: Vec<usize>,
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            address: 0,
            labels: HashMap::new(),
            forward_refs: Vec::new(),
            code: Vec::new(),
            source_map: Vec::new(),
        }
    }

    /// Assemble source text into binary
    pub fn assemble(&mut self, source: &str) -> Result<AssemblyResult, AsmError> {
        self.address = 0;
        self.labels.clear();
        self.forward_refs.clear();
        self.code.clear();
        self.source_map.clear();

        // First pass: collect labels and generate code
        for (line_num, line) in source.lines().enumerate() {
            self.process_line(line, line_num + 1)?;
        }

        // Check program size
        if self.code.len() > 128 {
            return Err(AsmError::ProgramTooLarge(self.code.len()));
        }

        Ok(AssemblyResult {
            code: self.code.clone(),
            labels: self.labels.clone(),
            source_map: self.source_map.clone(),
        })
    }

    fn process_line(&mut self, line: &str, line_num: usize) -> Result<(), AsmError> {
        // Remove comments
        let line = if let Some(pos) = line.find(';') {
            &line[..pos]
        } else {
            line
        };

        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }

        // Check for label
        if let Some(label) = line.strip_suffix(':') {
            let label = label.trim().to_uppercase();
            if self.labels.contains_key(&label) {
                return Err(AsmError::DuplicateLabel(label));
            }
            self.labels.insert(label, self.address);
            return Ok(());
        }

        // Parse instruction
        let instruction = self.parse_instruction(line)?;
        let encoded = instruction.encode();
        self.code.push(encoded);
        self.source_map.push(line_num);
        self.address = self.address.wrapping_add(1);

        Ok(())
    }

    fn parse_instruction(&self, line: &str) -> Result<Instruction, AsmError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(AsmError::InvalidSyntax(line.to_string()));
        }

        let opcode_str = parts[0].to_uppercase();
        let opcode = Opcode::from_str(&opcode_str)
            .ok_or_else(|| AsmError::UnknownOpcode(opcode_str.clone()))?;

        // Parse operands based on opcode
        match opcode {
            // No operands
            Opcode::Nop | Opcode::Jmp | Opcode::Jz | Opcode::Jnz => {
                Ok(Instruction::new(opcode, Register::R0, 0))
            }

            // Single register: NOT Rd, SHL Rd, SHR Rd
            Opcode::Not | Opcode::Shl | Opcode::Shr => {
                if parts.len() < 2 {
                    return Err(AsmError::MissingOperand(format!("{} requires Rd", opcode)));
                }
                let rd = self.parse_register(parts[1])?;
                Ok(Instruction::new(opcode, rd, 0))
            }

            // LOAD Rd (from MEM[R3])
            Opcode::Load => {
                if parts.len() < 2 {
                    return Err(AsmError::MissingOperand("LOAD requires Rd".to_string()));
                }
                let rd = self.parse_register(parts[1])?;
                Ok(Instruction::new(opcode, rd, 0))
            }

            // STORE Rs (to MEM[R3])
            Opcode::Store => {
                if parts.len() < 2 {
                    return Err(AsmError::MissingOperand("STORE requires Rs".to_string()));
                }
                let rs = self.parse_register(parts[1])?;
                Ok(Instruction::new(opcode, Register::R0, rs as u8))
            }

            // LDI Rd, #imm
            Opcode::Ldi => {
                if parts.len() < 2 {
                    return Err(AsmError::MissingOperand(
                        "LDI requires Rd, #imm".to_string(),
                    ));
                }
                let operands = self.parse_two_operands(&parts[1..])?;
                let rd = self.parse_register(&operands.0)?;
                let imm = self.parse_immediate(&operands.1)?;
                Ok(Instruction::new(opcode, rd, imm))
            }

            // Two registers: MOV Rd, Rs, ADD Rd, Rs, etc.
            _ => {
                if parts.len() < 2 {
                    return Err(AsmError::MissingOperand(format!(
                        "{} requires Rd, Rs",
                        opcode
                    )));
                }
                let operands = self.parse_two_operands(&parts[1..])?;
                let rd = self.parse_register(&operands.0)?;
                let rs = self.parse_register(&operands.1)?;
                Ok(Instruction::new(opcode, rd, rs as u8))
            }
        }
    }

    fn parse_two_operands(&self, parts: &[&str]) -> Result<(String, String), AsmError> {
        // Handle "Rd, Rs" or "Rd Rs" formats
        let joined = parts.join(" ");
        let operands: Vec<&str> = joined.split(',').map(|s| s.trim()).collect();

        if operands.len() >= 2 {
            Ok((operands[0].to_string(), operands[1].to_string()))
        } else if parts.len() >= 2 {
            Ok((
                parts[0].trim_end_matches(',').to_string(),
                parts[1].to_string(),
            ))
        } else {
            Err(AsmError::MissingOperand(
                "Expected two operands".to_string(),
            ))
        }
    }

    fn parse_register(&self, s: &str) -> Result<Register, AsmError> {
        let s = s.trim().trim_end_matches(',');
        Register::from_str(s).ok_or_else(|| AsmError::InvalidRegister(s.to_string()))
    }

    fn parse_immediate(&self, s: &str) -> Result<u8, AsmError> {
        let s = s.trim().trim_start_matches('#');

        // Try parsing as number
        let value = if s.starts_with("0x") || s.starts_with("0X") {
            u8::from_str_radix(&s[2..], 16)
        } else if s.starts_with("0b") || s.starts_with("0B") {
            u8::from_str_radix(&s[2..], 2)
        } else {
            s.parse::<u8>()
        };

        match value {
            Ok(v) if v <= 3 => Ok(v),
            Ok(v) => Err(AsmError::InvalidImmediate(format!(
                "{} is too large (max 3)",
                v
            ))),
            Err(_) => Err(AsmError::InvalidImmediate(s.to_string())),
        }
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to assemble source
pub fn assemble(source: &str) -> Result<Vec<u8>, AsmError> {
    let mut asm = Assembler::new();
    asm.assemble(source).map(|r| r.code)
}

/// Disassemble binary to instructions
pub fn disassemble(code: &[u8]) -> Vec<String> {
    code.iter()
        .enumerate()
        .map(|(addr, &byte)| {
            if let Some(inst) = Instruction::decode(byte) {
                format!("{:02X}: {:02X}  {}", addr, byte, inst)
            } else {
                format!("{:02X}: {:02X}  ???", addr, byte)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_program() {
        let source = r#"
            ; Add two numbers
            LDI R0, #1      ; R0 = 1
            LDI R1, #2      ; R1 = 2
            ADD R0, R1      ; R0 = R0 + R1 = 3
        "#;

        let result = assemble(source).unwrap();
        assert_eq!(result.len(), 3);

        // LDI R0, #1 = 1010_00_01 = 0xA1
        assert_eq!(result[0], 0xA1);
        // LDI R1, #2 = 1010_01_10 = 0xA6
        assert_eq!(result[1], 0xA6);
        // ADD R0, R1 = 0010_00_01 = 0x21
        assert_eq!(result[2], 0x21);
    }

    #[test]
    fn test_all_opcodes() {
        let source = r#"
            NOP
            MOV R0, R1
            ADD R0, R1
            SUB R0, R1
            AND R0, R1
            OR R0, R1
            XOR R0, R1
            NOT R0
            SHL R0
            SHR R0
            LDI R0, #0
            JMP
            JZ
            JNZ
            LOAD R0
            STORE R0
        "#;

        let result = assemble(source).unwrap();
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_labels() {
        let source = r#"
            start:
                LDI R0, #0
            loop:
                ADD R0, R1
                JNZ
        "#;

        let mut asm = Assembler::new();
        let result = asm.assemble(source).unwrap();

        assert_eq!(result.labels.get("START"), Some(&0));
        assert_eq!(result.labels.get("LOOP"), Some(&1));
        assert_eq!(result.code.len(), 3);
    }

    #[test]
    fn test_disassemble() {
        let code = vec![0xA1, 0xA6, 0x21];
        let dis = disassemble(&code);

        assert_eq!(dis[0], "00: A1  LDI R0, #1");
        assert_eq!(dis[1], "01: A6  LDI R1, #2");
        assert_eq!(dis[2], "02: 21  ADD R0, R1");
    }

    #[test]
    fn test_error_handling() {
        // Unknown opcode
        assert!(matches!(
            assemble("INVALID R0"),
            Err(AsmError::UnknownOpcode(_))
        ));

        // Invalid register
        assert!(matches!(
            assemble("MOV R5, R0"),
            Err(AsmError::InvalidRegister(_))
        ));

        // Invalid immediate
        assert!(matches!(
            assemble("LDI R0, #5"),
            Err(AsmError::InvalidImmediate(_))
        ));
    }

    #[test]
    fn test_fibonacci_program() {
        // A simple Fibonacci-style program
        let source = r#"
            ; Initialize fib(0) = 1, fib(1) = 1
            LDI R0, #1      ; R0 = fib(n)
            LDI R1, #1      ; R1 = fib(n-1)
            LDI R2, #0      ; R2 = temp
        loop:
            MOV R2, R0      ; temp = fib(n)
            ADD R0, R1      ; fib(n) = fib(n) + fib(n-1)
            MOV R1, R2      ; fib(n-1) = temp
            JMP             ; loop (R3 should point to loop)
        "#;

        let result = assemble(source).unwrap();
        assert_eq!(result.len(), 7);
    }
}
