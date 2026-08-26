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

#[cfg(test)]
mod tests {
    use super::*;

    // The 4-byte instruction encoding this module emits (pinned by these tests):
    //   byte 0: opcode
    //   byte 1: flags = (src_is_imm ? 0x80 : 0) | ((r_dest & 0x0F) << 4) | (r_src & 0x0F)
    //   byte 2: imm >> 8   (high)
    //   byte 3: imm & 0xFF (low)

    #[test]
    fn nop_encodes_to_all_zero() {
        assert_eq!(assemble("NOP").unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn mov_register_to_register() {
        // MOV A, B -> dest=A(0) src=B(1), reg mode (no imm flag)
        assert_eq!(assemble("MOV A, B").unwrap(), vec![1, 0x01, 0, 0]);
    }

    #[test]
    fn mov_immediate_sets_high_flag_bit() {
        // MOV A, 5 -> immediate source: flag bit 0x80 set, src nibble cleared to 0
        assert_eq!(assemble("MOV A, 5").unwrap(), vec![1, 0x80, 0, 5]);
    }

    #[test]
    fn mov_hex_immediate_splits_across_two_bytes() {
        assert_eq!(
            assemble("MOV A, 0x1234").unwrap(),
            vec![1, 0x80, 0x12, 0x34]
        );
    }

    #[test]
    fn dest_and_src_nibbles_pack_higher_registers() {
        // MOV IP, SP -> dest=IP(4) in high nibble, src=SP(5) in low nibble
        assert_eq!(assemble("MOV IP, SP").unwrap(), vec![1, 0x45, 0, 0]);
    }

    #[test]
    fn add_register_and_immediate_forms() {
        assert_eq!(assemble("ADD B, C").unwrap(), vec![2, 0x12, 0, 0]);
        // ADD B, 3 -> imm flag 0x80 | dest B(1)<<4
        assert_eq!(assemble("ADD B, 3").unwrap(), vec![2, 0x90, 0, 3]);
    }

    #[test]
    fn sub_register_form() {
        assert_eq!(assemble("SUB D, A").unwrap(), vec![3, 0x30, 0, 0]);
    }

    #[test]
    fn shr_immediate_form() {
        assert_eq!(assemble("SHR A, 1").unwrap(), vec![7, 0x80, 0, 1]);
    }

    #[test]
    fn out_uses_source_register_only() {
        assert_eq!(assemble("OUT A").unwrap(), vec![255, 0x00, 0, 0]);
        assert_eq!(assemble("OUT C").unwrap(), vec![255, 0x02, 0, 0]);
    }

    #[test]
    fn mnemonics_and_registers_are_case_insensitive() {
        assert_eq!(assemble("mov a, b").unwrap(), assemble("MOV A, B").unwrap());
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let src = "\
; leading comment
   MOV A, 5   ; inline comment

ADD A, B
";
        let expected = assemble("MOV A, 5\nADD A, B").unwrap();
        assert_eq!(assemble(src).unwrap(), expected);
        assert_eq!(expected, vec![1, 0x80, 0, 5, 2, 0x01, 0, 0]);
    }

    #[test]
    fn backward_label_resolves_to_earlier_address() {
        // start: is at pc 0; the NOP occupies bytes 0..4; JMP start targets 0.
        let src = "\
start:
    NOP
    JMP start
";
        assert_eq!(
            assemble(src).unwrap(),
            vec![0, 0, 0, 0, /* JMP */ 4, 0, 0, 0]
        );
    }

    #[test]
    fn forward_label_resolves_to_later_address() {
        // JMP end (pc 0), NOP (pc 4), end: at pc 8, OUT A (pc 8).
        let src = "\
    JMP end
    NOP
end:
    OUT A
";
        assert_eq!(
            assemble(src).unwrap(),
            vec![
                /* JMP end */ 4, 0, 0, 8, /* NOP */ 0, 0, 0, 0, /* OUT A */ 255, 0,
                0, 0
            ]
        );
    }

    #[test]
    fn conditional_jumps_encode_target_address() {
        let src = "\
    NOP
loop:
    JZ loop
    JNZ loop
";
        // loop: is at pc 4 (after the leading NOP).
        assert_eq!(
            assemble(src).unwrap(),
            vec![
                0, 0, 0, 0, /* JZ loop */ 5, 0, 0, 4, /* JNZ loop */ 6, 0, 0, 4
            ]
        );
    }

    #[test]
    fn unknown_mnemonic_is_rejected() {
        let err = assemble("FOO A, B").unwrap_err();
        assert!(err.contains("Unknown mnemonic"), "got: {err}");
        assert!(err.contains("FOO"), "got: {err}");
    }

    #[test]
    fn mov_with_too_few_args_is_rejected() {
        let err = assemble("MOV A").unwrap_err();
        assert!(err.contains("MOV requires 2 args"), "got: {err}");
    }

    #[test]
    fn unknown_label_is_rejected() {
        let err = assemble("JMP nowhere").unwrap_err();
        assert!(err.contains("Unknown label"), "got: {err}");
        assert!(err.contains("nowhere"), "got: {err}");
    }

    #[test]
    fn invalid_register_is_rejected() {
        let err = assemble("MOV Z, A").unwrap_err();
        assert!(err.contains("Invalid register"), "got: {err}");
    }

    #[test]
    fn mov_source_that_is_neither_register_nor_immediate_is_rejected() {
        let err = assemble("MOV A, xyz").unwrap_err();
        assert!(err.contains("Invalid source"), "got: {err}");
    }

    #[test]
    fn add_with_unparseable_immediate_is_rejected() {
        let err = assemble("ADD A, zzz").unwrap_err();
        assert!(err.contains("Invalid number"), "got: {err}");
    }
}
