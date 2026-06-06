use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Nop = 0,
    Mov = 1,
    Add = 2,
    Sub = 3,
    Jmp = 4,
    Jz = 5,
    Jnz = 6,
    // Directions / Math
    Shr = 7,
    // Debug/IO
    Out = 255,
}

#[derive(Debug)]
pub enum Operand {
    Register(u8),
    Immediate(u16),
    Label(String),
}

pub fn assemble(source: &str) -> Result<Vec<u8>, String> {
    let mut binary = Vec::new();
    let mut labels = HashMap::new();

    let lines: Vec<&str> = source.lines().collect();

    // Pass 1: Offset calculation & Label gathering
    let mut pc = 0;
    for line in &lines {
        // Strip inline comments (everything after ;)
        let clean = line.split(';').next().unwrap_or("").trim();
        if clean.is_empty() {
            continue;
        }

        if clean.ends_with(':') {
            let label_name = clean.trim_end_matches(':');
            labels.insert(label_name.to_string(), pc);
            continue;
        }

        // Assume 4 bytes per instruction for now
        pc += 4;
    }

    // Pass 2: Generation
    for (line_num, line) in lines.iter().enumerate() {
        // Strip inline comments (everything after ;)
        let clean = line.split(';').next().unwrap_or("").trim();
        if clean.is_empty() || clean.ends_with(':') {
            continue;
        }

        let parts: Vec<&str> = clean.split_whitespace().collect();
        let mnemonic = parts[0].to_uppercase();

        let op;
        let mut r_dest = 0u8;
        let mut r_src = 0u8;
        let mut imm = 0u16;

        match mnemonic.as_str() {
            "NOP" => op = Opcode::Nop as u8,
            "MOV" => {
                op = Opcode::Mov as u8;
                // MOV Dest, Src
                if parts.len() < 3 {
                    return Err(format!("Line {}: MOV requires 2 args", line_num));
                }
                r_dest = parse_reg(parts[1])?;

                if let Ok(reg) = parse_reg(parts[2]) {
                    r_src = reg;
                } else if let Ok(val) = parse_imm(parts[2]) {
                    imm = val;
                    r_src = 255; // Flag for Immediate
                } else {
                    return Err(format!("Line {}: Invalid source '{}'", line_num, parts[2]));
                }
            }
            "ADD" => {
                op = Opcode::Add as u8;
                r_dest = parse_reg(parts[1])?;
                if let Ok(reg) = parse_reg(parts[2]) {
                    r_src = reg;
                } else {
                    imm = parse_imm(parts[2])?;
                    r_src = 255;
                }
            }
            "SUB" => {
                op = Opcode::Sub as u8;
                r_dest = parse_reg(parts[1])?;
                if let Ok(reg) = parse_reg(parts[2]) {
                    r_src = reg;
                } else {
                    imm = parse_imm(parts[2])?;
                    r_src = 255;
                }
            }
            "JMP" => {
                op = Opcode::Jmp as u8;
                // JMP Label
                let target = parts[1];
                if let Some(addr) = labels.get(target) {
                    imm = *addr as u16;
                } else {
                    return Err(format!("Line {}: Unknown label '{}'", line_num, target));
                }
            }
            "JZ" => {
                op = Opcode::Jz as u8;
                let target = parts[1];
                if let Some(addr) = labels.get(target) {
                    imm = *addr as u16;
                } else {
                    return Err(format!("Line {}: Unknown label '{}'", line_num, target));
                }
            }
            "JNZ" => {
                op = Opcode::Jnz as u8;
                let target = parts[1];
                if let Some(addr) = labels.get(target) {
                    imm = *addr as u16;
                } else {
                    return Err(format!("Line {}: Unknown label '{}'", line_num, target));
                }
            }
            "SHR" => {
                op = Opcode::Shr as u8;
                r_dest = parse_reg(parts[1])?;
                if let Ok(reg) = parse_reg(parts[2]) {
                    r_src = reg;
                } else {
                    imm = parse_imm(parts[2])?;
                    r_src = 255;
                }
            }
            "OUT" => {
                op = Opcode::Out as u8;
                r_src = parse_reg(parts[1])?;
            }
            _ => {
                return Err(format!(
                    "Line {}: Unknown mnemonic '{}'",
                    line_num, mnemonic
                ));
            }
        }

        // Revised 4-Byte Struct:
        // 0: Opcode
        // 1: Flags (Bit 0: Src is Imm?) | (Dest Reg << 4) | (Src Reg)
        // 2: Imm High
        // 3: Imm Low

        let mut flags = 0u8;

        // If Src is Imm (255), set bit 7 (matching shader logic: flags & 0x80)
        if r_src == 255 {
            flags |= 0x80; // Src is Immediate
            r_src = 0;
        }

        flags |= (r_dest & 0x0F) << 4;
        flags |= r_src & 0x0F;

        binary.push(op);
        binary.push(flags);
        binary.push((imm >> 8) as u8);
        binary.push((imm & 0xFF) as u8);
    }

    Ok(binary)
}

fn parse_reg(s: &str) -> Result<u8, String> {
    let s = s.trim_end_matches(',');
    match s.to_uppercase().as_str() {
        "A" => Ok(0),
        "B" => Ok(1),
        "C" => Ok(2),
        "D" => Ok(3),
        "IP" => Ok(4),
        "SP" => Ok(5),
        _ => Err(format!("Invalid register: {}", s)),
    }
}

fn parse_imm(s: &str) -> Result<u16, String> {
    let s = s.trim_end_matches(',');
    if s.starts_with("0x") {
        u16::from_str_radix(&s[2..], 16).map_err(|_| "Invalid hex".to_string())
    } else {
        s.parse::<u16>().map_err(|_| "Invalid number".to_string())
    }
}
