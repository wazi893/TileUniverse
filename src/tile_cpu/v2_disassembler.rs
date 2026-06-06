//! V2 disassembler and structured decode helpers.
//!
//! Phase 1 (Sprint 140): stable decode/disasm surface for trace and tooling.

use crate::tile_cpu::v2_components::{
    EXT_OFFSET, EXT_RD_HI, EXT_RS_HI, EXT_WIDE_IMM, effective_reg,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2DecodedInstruction {
    pub word: u32,
    pub opcode: u8,
    pub mnemonic: &'static str,
    pub rd_lo: u8,
    pub rs_lo: u8,
    pub rd: u8,
    pub rs: u8,
    pub imm5: u8,
    pub imm8: u8,
    pub jump_target: u8,
    pub ir_ext: u16,
    pub rd_hi: bool,
    pub rs_hi: bool,
    pub wide_imm: bool,
    pub has_offset: bool,
    pub offset: u8,
    pub imm16: u16,
    pub ext_reserved: u16,
}

impl V2DecodedInstruction {
    pub fn has_reserved_ext_bits(&self) -> bool {
        self.ext_reserved != 0
    }
}

fn mnemonic_for_opcode(opcode: u8) -> &'static str {
    match opcode {
        0x00 => "NOP",
        0x01 => "HALT",
        0x02 => "MOV",
        0x03 => "LDI",
        0x04 => "ADD",
        0x05 => "SUB",
        0x06 => "AND",
        0x07 => "OR",
        0x08 => "XOR",
        0x09 => "NOT",
        0x0A => "NEG",
        0x0B => "INC",
        0x0C => "DEC",
        0x0D => "SHL",
        0x0E => "SHR",
        0x0F => "CMP",
        0x10 => "ADDI",
        0x11 => "SUBI",
        0x12 => "ANDI",
        0x13 => "ORI",
        0x14 => "XORI",
        0x15 => "RET",
        0x16 => "LD",
        0x17 => "ST",
        0x18 => "LDB",
        0x19 => "STB",
        0x1A => "JMP",
        0x1B => "JZ",
        0x1C => "JNZ",
        0x1D => "JC",
        0x1E => "JNC",
        0x1F => "CALL",
        _ => "UNK",
    }
}

fn reg_name(reg: u8) -> String {
    format!("R{reg}")
}

/// Decode one 32-bit V2 instruction word into structured fields.
pub fn decode_v2_word(word: u32) -> V2DecodedInstruction {
    let opcode = ((word >> 11) & 0x1F) as u8;
    let rd_lo = ((word >> 8) & 0x07) as u8;
    let rs_lo = ((word >> 5) & 0x07) as u8;
    let imm5 = (word & 0x1F) as u8;
    let imm8 = (word & 0xFF) as u8;
    let ir_ext = ((word >> 16) & 0xFFFF) as u16;

    let rd_hi = (ir_ext & EXT_RD_HI) != 0;
    let rs_hi = (ir_ext & EXT_RS_HI) != 0;
    let wide_imm = (ir_ext & EXT_WIDE_IMM) != 0;
    let has_offset = (ir_ext & EXT_OFFSET) != 0;
    let offset = if has_offset {
        ((ir_ext >> 8) & 0xFF) as u8
    } else {
        0
    };
    let imm16 = if wide_imm {
        (((ir_ext >> 8) & 0xFF) << 8) | (imm8 as u16)
    } else {
        imm8 as u16
    };
    // Sprint 174/175: bits 8-15 are payload when wide_imm or has_offset is set (not reserved).
    // Only unused flag bits in the low byte are truly reserved.
    let known_flags = EXT_RD_HI | EXT_RS_HI | EXT_WIDE_IMM | EXT_OFFSET;
    let ext_reserved = if wide_imm || has_offset {
        ir_ext & !known_flags & 0x00FF
    } else {
        ir_ext & !known_flags
    };

    // Sprint 176/177: sub-function-aware mnemonics for opcodes 0x09 and 0x02
    let mnemonic = if opcode == 0x09 {
        match imm5 {
            1 => "CLZ",
            2 => "CTZ",
            3 => "POPCNT",
            _ => "NOT",
        }
    } else if opcode == 0x02 {
        match imm5 {
            1 => "MOVZ",
            2 => "MOVNZ",
            3 => "MOVC",
            4 => "MOVNC",
            _ => "MOV",
        }
    } else {
        mnemonic_for_opcode(opcode)
    };

    V2DecodedInstruction {
        word,
        opcode,
        mnemonic,
        rd_lo,
        rs_lo,
        rd: effective_reg(rd_lo, rd_hi),
        rs: effective_reg(rs_lo, rs_hi),
        imm5,
        imm8,
        // Sprint 187: 7-bit jump target (was 0x3F).
        jump_target: imm8 & 0x7F,
        ir_ext,
        rd_hi,
        rs_hi,
        wide_imm,
        has_offset,
        offset,
        imm16,
        ext_reserved,
    }
}

/// Disassemble one V2 32-bit instruction word into stable assembler text.
pub fn disassemble_v2_word(word: u32) -> String {
    let d = decode_v2_word(word);

    // Current V2 toolchain only defines extension bits 0..1.
    if d.has_reserved_ext_bits() {
        return format!(
            ".word32 0x{word:08X} ; RESERVED_EXT=0x{:04X}",
            d.ext_reserved
        );
    }

    match d.opcode {
        0x00 => "NOP".to_string(),
        0x01 => "HALT".to_string(),
        0x02 => format!("{} {}, {}", d.mnemonic, reg_name(d.rd), reg_name(d.rs)),
        0x03 => {
            if d.wide_imm {
                format!("LDI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("LDI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x04 => {
            // Sprint 188: imm5=1 → MUL
            if d.imm5 == 1 {
                format!("MUL {}, {}", reg_name(d.rd), reg_name(d.rs))
            } else {
                format!("ADD {}, {}", reg_name(d.rd), reg_name(d.rs))
            }
        }
        0x05 => format!("SUB {}, {}", reg_name(d.rd), reg_name(d.rs)),
        0x06 => format!("AND {}, {}", reg_name(d.rd), reg_name(d.rs)),
        0x07 => format!("OR {}, {}", reg_name(d.rd), reg_name(d.rs)),
        0x08 => format!("XOR {}, {}", reg_name(d.rd), reg_name(d.rs)),
        0x09 => {
            // Sprint 176: sub-function in imm5
            let mnemonic = match d.imm5 {
                1 => "CLZ",
                2 => "CTZ",
                3 => "POPCNT",
                _ => "NOT",
            };
            format!("{} {}", mnemonic, reg_name(d.rd))
        }
        0x0A => format!("NEG {}", reg_name(d.rd)),
        0x0B => format!("INC {}", reg_name(d.rd)),
        0x0C => format!("DEC {}", reg_name(d.rd)),
        0x0D => format!("SHL {}, {}", reg_name(d.rd), d.imm5 & 0x07),
        0x0E => {
            // Sprint 188: bit 3 of imm5 → SRA
            if d.imm5 & 0x08 != 0 {
                format!("SRA {}, {}", reg_name(d.rd), d.imm5 & 0x07)
            } else {
                format!("SHR {}, {}", reg_name(d.rd), d.imm5 & 0x07)
            }
        }
        0x0F => format!("CMP {}, {}", reg_name(d.rd), reg_name(d.rs)),
        0x10 => {
            if d.wide_imm {
                format!("ADDI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("ADDI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x11 => {
            if d.wide_imm {
                format!("SUBI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("SUBI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x12 => {
            if d.wide_imm {
                format!("ANDI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("ANDI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x13 => {
            if d.wide_imm {
                format!("ORI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("ORI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x14 => {
            if d.wide_imm {
                format!("XORI.W {}, {}", reg_name(d.rd), d.imm16)
            } else {
                format!("XORI {}, {}", reg_name(d.rd), d.imm8)
            }
        }
        0x15 => "RET".to_string(),
        0x16 => {
            if d.has_offset {
                format!("LD {}, [{}+{}]", reg_name(d.rd), reg_name(d.rs), d.offset)
            } else {
                format!("LD {}, [{}]", reg_name(d.rd), reg_name(d.rs))
            }
        }
        0x17 => {
            if d.has_offset {
                format!("ST [{}+{}], {}", reg_name(d.rs), d.offset, reg_name(d.rd))
            } else {
                format!("ST [{}], {}", reg_name(d.rs), reg_name(d.rd))
            }
        }
        0x18 => {
            if d.has_offset {
                format!("LDB {}, [{}+{}]", reg_name(d.rd), reg_name(d.rs), d.offset)
            } else {
                format!("LDB {}, [{}]", reg_name(d.rd), d.imm8)
            }
        }
        0x19 => {
            if d.has_offset {
                format!("STB [{}+{}], {}", reg_name(d.rs), d.offset, reg_name(d.rd))
            } else {
                format!("STB [{}], {}", d.imm8, reg_name(d.rd))
            }
        }
        0x1A => format!("JMP {}", d.jump_target),
        0x1B => format!("JZ {}", d.jump_target),
        0x1C => format!("JNZ {}", d.jump_target),
        0x1D => format!("JC {}", d.jump_target),
        0x1E => format!("JNC {}", d.jump_target),
        0x1F => format!("CALL {}", d.jump_target),
        _ => format!(".word32 0x{word:08X} ; UNKNOWN_OPCODE=0x{:02X}", d.opcode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_assembler::assemble_v2;
    use crate::tile_cpu::v2_components::{
        v2_add, v2_addi, v2_and, v2_andi, v2_call, v2_cmp, v2_dec, v2_halt, v2_inc, v2_jc, v2_jmp,
        v2_jnc, v2_jnz, v2_jz, v2_ld, v2_ldb, v2_ldi, v2_mov, v2_neg, v2_nop, v2_not, v2_or,
        v2_ori, v2_ret, v2_shl, v2_shr, v2_st, v2_stb, v2_sub, v2_subi, v2_with_ext,
        v2_with_reg_ext, v2_xor, v2_xori,
    };

    fn assert_round_trip(word: u32) {
        let asm = disassemble_v2_word(word);
        let program = assemble_v2(&asm).expect("disassembled line should reassemble");
        assert_eq!(program, vec![word], "round-trip mismatch for: {asm}");
    }

    #[test]
    fn test_decode_v2_word_fields_and_ext() {
        let word = v2_with_reg_ext(v2_mov(1, 2), true, false);
        let d = decode_v2_word(word);
        assert_eq!(d.opcode, 0x02);
        assert_eq!(d.rd_lo, 1);
        assert_eq!(d.rs_lo, 2);
        assert!(d.rd_hi);
        assert!(!d.rs_hi);
        assert_eq!(d.rd, 9);
        assert_eq!(d.rs, 2);
        assert_eq!(d.mnemonic, "MOV");
    }

    #[test]
    fn test_disassemble_v2_fixtures_mixed_bank_and_extension() {
        let mixed = v2_with_reg_ext(v2_add(0, 1), true, true);
        assert_eq!(disassemble_v2_word(mixed), "ADD R8, R9");

        let ext_reserved = v2_with_ext(v2_ldi(0, 42), 0x00F0);
        assert_eq!(
            disassemble_v2_word(ext_reserved),
            ".word32 0x00F0182A ; RESERVED_EXT=0x00F0"
        );
    }

    #[test]
    fn test_disassemble_v2_round_trip_all_opcodes_with_helpers() {
        let words = [
            v2_nop(),
            v2_halt(),
            v2_mov(2, 1),
            v2_ldi(3, 200),
            v2_add(1, 7),
            v2_sub(6, 4),
            v2_and(0, 5),
            v2_or(4, 3),
            v2_xor(7, 2),
            v2_not(2),
            v2_neg(6),
            v2_inc(1),
            v2_dec(0),
            v2_shl(3, 5),
            v2_shr(3, 2),
            v2_cmp(5, 4),
            v2_addi(4, 123),
            v2_subi(4, 77),
            v2_andi(1, 0xAA),
            v2_ori(2, 0x55),
            v2_xori(7, 0x0F),
            v2_ret(),
            v2_ld(3, 1),
            v2_st(2, 6),
            v2_ldb(5, 63),
            v2_stb(4, 9),
            v2_jmp(63),
            v2_jz(12),
            v2_jnz(31),
            v2_jc(7),
            v2_jnc(40),
            v2_call(11),
            v2_with_reg_ext(v2_mov(1, 0), true, true),
            v2_with_reg_ext(v2_ld(2, 3), true, false),
            v2_with_reg_ext(v2_st(3, 2), false, true),
            v2_with_reg_ext(v2_ldb(0, 8), true, false),
        ];

        for &word in &words {
            assert_round_trip(word);
        }
    }
}
