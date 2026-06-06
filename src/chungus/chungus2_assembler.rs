use std::collections::HashMap;

/// CHUNGUS 2 Opcode Definitions (5 bits: 0-31)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0,
    Ldi = 1,  // Load Immediate: LDI Rd, Imm
    Mov = 2,  // Move: MOV Rd, Rs
    Add = 3,  // Add: ADD Rd, Rs1, Rs2
    Sub = 4,  // Sub: SUB Rd, Rs1, Rs2
    Mul = 5,  // Mul: MUL Rd, Rs1, Rs2
    Div = 6,  // Div: DIV Rd, Rs1, Rs2
    And = 7,  // Bitwise AND
    Or = 8,   // Bitwise OR
    Xor = 9,  // Bitwise XOR
    Not = 10, // Bitwise NOT: NOT Rd, Rs
    Jmp = 11, // Jump Absolute: JMP Addr
    Jz = 12,  // Jump if Zero: JZ Rs, Addr
    Jnz = 13, // Jump if Not Zero: JNZ Rs, Addr
    Eq = 14,  // Set Equal: EQ Rd, Rs1, Rs2
    Gt = 15,  // Set Greater Than: GT Rd, Rs1, Rs2
    Lt = 16,  // Set Less Than: LT Rd, Rs1, Rs2
    Out = 31, // Debug Output
}

/// 16-bit Instruction Format:
/// [ Opcode (5) ] [ Dest (3) ] [ Src1 (3) ] [ Src2 (3) ] [ Unused (2) ]
///
/// Registers: R0-R6, Target (R7 for implementation specifics)
pub fn assemble(source: &str) -> Result<Vec<u8>, String> {
    let mut binary = Vec::new();
    let mut labels = HashMap::new();

    let lines: Vec<&str> = source.lines().collect();

    // Pass 1: Symbol Table
    let mut pc = 0;
    for line in &lines {
        let clean = line.trim();
        if clean.is_empty() || clean.starts_with("//") || clean.starts_with(';') {
            continue;
        }

        if clean.ends_with(':') {
            let label = clean.trim_end_matches(':');
            labels.insert(label.to_string(), pc);
            continue;
        }
        // Each instruction is 2 bytes (16 bits)
        pc += 2;
    }

    // Pass 2: Assembly
    for (line_num, line) in lines.iter().enumerate() {
        let clean = line.trim();
        if clean.is_empty()
            || clean.starts_with("//")
            || clean.starts_with(';')
            || clean.ends_with(':')
        {
            continue;
        }

        // Split by whitespace but keep commas in mind? No, usually split by whitespace and trim commas.
        let parts: Vec<String> = clean.split_fields();
        if parts.is_empty() {
            continue;
        }

        let mnemonic = parts[0].to_uppercase();
        let op;
        let mut rd = 0u8;
        let mut rs1 = 0u8;
        let mut rs2 = 0u8;

        // Helper to parse operands
        // Opcodes have different signatures
        // Helper to parse operands
        match mnemonic.as_str() {
            "NOP" => op = Opcode::Nop as u8,
            "LDI" => {
                // LDI Rd, Imm8
                // Pack Imm8 into [Rs1:3][Rs2:3][Extra:2]
                op = Opcode::Ldi as u8;
                rd = parse_reg(&parts[1], line_num)?;
                let imm = parse_imm(&parts[2])?;
                rs1 = (imm >> 5) & 0x7;
                rs2 = (imm >> 2) & 0x7;
                // Bottom 2 bits handled in encoding stage (or unused/implicit)
            }
            "ADD" | "SUB" | "MUL" | "DIV" | "AND" | "OR" | "XOR" | "EQ" | "GT" | "LT" => {
                op = match mnemonic.as_str() {
                    "ADD" => Opcode::Add,
                    "SUB" => Opcode::Sub,
                    "MUL" => Opcode::Mul,
                    "DIV" => Opcode::Div,
                    "AND" => Opcode::And,
                    "OR" => Opcode::Or,
                    "XOR" => Opcode::Xor,
                    "EQ" => Opcode::Eq,
                    "GT" => Opcode::Gt,
                    "LT" => Opcode::Lt,
                    _ => unreachable!(),
                } as u8;
                if parts.len() < 4 {
                    return Err(format!("Line {}: {} requires 3 args", line_num, mnemonic));
                }
                rd = parse_reg(&parts[1], line_num)?;
                rs1 = parse_reg(&parts[2], line_num)?;
                rs2 = parse_reg(&parts[3], line_num)?;
            }
            "MOV" | "NOT" => {
                op = if mnemonic == "MOV" {
                    Opcode::Mov
                } else {
                    Opcode::Not
                } as u8;
                rd = parse_reg(&parts[1], line_num)?;
                rs1 = parse_reg(&parts[2], line_num)?;
            }
            "JMP" => {
                op = Opcode::Jmp as u8;
                let label = &parts[1];
                if let Some(_addr) = labels.get(label) {
                    // JMP uses special encoding later, just set op/addr placeholders if compatible
                    // But we don't have 'addr' field in struct.
                    // We use the labels lookup in the Encoding Phase (match op at bottom of loop).
                    // So here we just set Op.
                } else {
                    return Err(format!("Line {}: Unknown label '{}'", line_num, label));
                }
            }
            "JZ" | "JNZ" => {
                op = if mnemonic == "JZ" {
                    Opcode::Jz
                } else {
                    Opcode::Jnz
                } as u8;
                // JZ Rs, Label
                rs1 = parse_reg(&parts[1], line_num)?;
                let label = &parts[2];
                if !labels.contains_key(label) {
                    return Err(format!(
                        "Line {}: Unknown label to jump '{}'",
                        line_num, label
                    ));
                }
            }
            "OUT" => {
                op = Opcode::Out as u8;
            }
            _ => {
                return Err(format!(
                    "Line {}: Unknown mnemonic '{}'",
                    line_num, mnemonic
                ));
            }
        }

        // Encode
        let mut word: u16 = 0;
        word |= (op as u16) << 11;

        // Custom Encoding per Opcode Type if needed
        match op {
            1 => {
                // LDI
                // Standard Rd
                word |= (rd as u16) << 8;
                // Reconstruction of Imm8
                let imm = parse_imm(&parts[2]).unwrap_or(0); // parse again essentially
                // Imm is bits 0..7.
                // LDI format in shader: [Rs1:3][Rs2:3][Extra:2] maps to bits 7..0
                // My generic encoder does: Rs1<<5, Rs2<<2.
                // This effectively places Rs1 at bits 7..5? No.
                // Generic: Rs1 is at bit 5 (length 3 -> 5,6,7).
                // Generic: Rs2 is at bit 2 (length 3 -> 2,3,4).
                // We need bits 0,1.
                // So generic encoder maps [7..0] exactly if we treat [Rs1, Rs2, Extra] as Imm.
                // Imm8 = (Rs1<<5) | (Rs2<<2) | Extra.
                // So we just need to set the word directly for lower 8 bits.
                word |= imm as u16;
            }
            11 => {
                // JMP
                let label = &parts[1];
                let addr = *labels.get(label).unwrap_or(&0) as u16;
                word |= (addr & 0x7FF) << 0; // JMP uses 11 bits?
                // Wait, JMP Op is 5 bits (15..11).
                // Addr is 11 bits (10..0).
                // Previous code: word |= (addr as u16) << 2; (dropped 2 bits? or shifted up?)
                // If I want 11 bits of address, I should use bits 0..10.
                // So: word |= addr & 0x7FF;
            }
            12 | 13 => {
                // JZ, JNZ
                // JZ Rs, Label.
                // Op(5) [15..11]
                // Rs1(3) [used for condition reg] -> Where does generic put Rs1? Bit 5.
                // Generic puts Rs1 at 5..7.
                // We need 8 bits for Address (Target).
                // Address needs to be in other slots?
                // Slots available: Rd(3) [10..8], Rs2(3) [4..2], Unused(2) [1..0].
                // Total 8 bits: 10,9,8, 4,3,2, 1,0.
                // This is fragmented.
                // Let's use:
                // Rs1 (Condition) -> Bit 5..7 (Standard Position)
                // Address bits -> Rd (10..8) | Rs2 (4..2) | Extra (1..0)?
                // Let's map Address High (7..5) to Rd.
                // Address Mid (4..2) to Rs2.
                // Address Low (1..0) to Extra.
                let label = &parts[2];
                let addr = *labels.get(label).unwrap_or(&0) as u16;

                // Rs1 is already parsed and set.
                word |= (rs1 as u16) << 5;

                // Pack Address
                let addr_h = (addr >> 5) & 0x7;
                let addr_m = (addr >> 2) & 0x7;
                let addr_l = addr & 0x3;

                word |= (addr_h) << 8; // In Rd slot
                word |= (addr_m) << 2; // In Rs2 slot
                word |= addr_l; // In Extra slot
            }
            _ => {
                // Standard Encoding
                word |= (rd as u16) << 8;
                word |= (rs1 as u16) << 5;
                word |= (rs2 as u16) << 2;
            }
        }

        binary.push((word & 0xFF) as u8);
        binary.push((word >> 8) as u8);
    }

    Ok(binary)
}

fn parse_reg(s: &str, line: usize) -> Result<u8, String> {
    let clean = s.trim_matches(',');
    if !clean.to_uppercase().starts_with('R') {
        return Err(format!(
            "Line {}: Invalid register '{}', expected R0-R7",
            line, clean
        ));
    }
    let num_str = &clean[1..];
    let num = num_str
        .parse::<u8>()
        .map_err(|_| format!("Line {}: Invalid reg number", line))?;
    if num > 7 {
        return Err(format!(
            "Line {}: Register R{} out of range (0-7)",
            line, num
        ));
    }
    Ok(num)
}

fn parse_imm(s: &str) -> Result<u8, String> {
    let s = s.trim_matches(',');
    let val = if s.starts_with("0x") {
        u16::from_str_radix(&s[2..], 16).map_err(|_| "Invalid hex".to_string())?
    } else {
        s.parse::<u16>().map_err(|_| "Invalid number".to_string())?
    };
    if val > 255 {
        return Err("Immediate too large for 8-bit".to_string());
    }
    Ok(val as u8)
}

// Extension trait to split string but handle commas nicely
trait SplitFields {
    fn split_fields(&self) -> Vec<String>;
}

impl SplitFields for str {
    fn split_fields(&self) -> Vec<String> {
        self.split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}
