//! V2 assembler.
//!
//! Text assembly input -> V2 32-bit instruction words (`Vec<u32>`).
//!
//! Sprint 155 enhancements:
//! - `.equ NAME VALUE` — constant definitions
//! - `.org ADDR` — set output address (NOP padding)
//! - `PUSH Rn` / `POP Rn` — pseudo-instructions (R15 as stack pointer)
//! - `SP` register alias for R15
//! - Built-in symbolic MMIO address names

use std::collections::HashMap;

use crate::tile_cpu::v2_components::{
    EXT_OFFSET, EXT_RD_HI, EXT_RS_HI, EXT_WIDE_IMM, v2_add, v2_addi, v2_and, v2_andi, v2_call,
    v2_call_w, v2_callr, v2_clz, v2_cmp, v2_ctz, v2_dec, v2_halt, v2_inc, v2_jc, v2_jc_w, v2_jmp,
    v2_jmp_w, v2_jmpr, v2_jnc, v2_jnc_w, v2_jnz, v2_jnz_w, v2_jz, v2_jz_w, v2_ld, v2_ldb, v2_ldi,
    v2_mflr, v2_mov, v2_movc, v2_movnc, v2_movnz, v2_movz, v2_mtlr, v2_mul, v2_neg, v2_nop, v2_not,
    v2_or, v2_ori, v2_popcnt, v2_ret, v2_shl, v2_shr, v2_sra, v2_st, v2_stb, v2_sub, v2_subi,
    v2_with_ext, v2_with_reg_ext, v2_xor, v2_xori,
};
use crate::tile_cpu::v2_mmio_devices::{
    MMIO_CONSOLE_COUNT, MMIO_CONSOLE_DATA, MMIO_DATASET_CMD, MMIO_DATASET_DATA,
    MMIO_DATASET_STATUS, MMIO_DISPLAY_CMD, MMIO_DISPLAY_STATUS, MMIO_MAILBOX_IN, MMIO_MAILBOX_OUT,
    MMIO_MATH_A, MMIO_MATH_B, MMIO_MATH_CMD, MMIO_MATH_RESULT, MMIO_PBIT_CTRL, MMIO_PBIT_RESULT,
    MMIO_QUANTUM_CMD, MMIO_QUANTUM_DATA, MMIO_QUANTUM_PARAM, MMIO_QUANTUM_QUBIT, MMIO_RNG_DATA,
    MMIO_SNN_CMD, MMIO_SNN_DATA, MMIO_TIMER_CYCLE,
};

/// Sprint 374 (Gate D.3): the maximum program size, matching the wide PC's 16-bit
/// instruction address space (B.3). Programs above 128 words require a wide-PC run
/// config (`with_extended_pc` / `with_wide_pc`) and `.W` branch/call forms to reach
/// targets past the 7-bit window.
pub const V2_MAX_PROGRAM_WORDS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2AsmError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for V2AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for V2AsmError {}

fn asm_err(line: usize, message: impl Into<String>) -> V2AsmError {
    V2AsmError {
        line,
        message: message.into(),
    }
}

fn strip_comments(line: &str) -> &str {
    let mut end = line.len();
    if let Some(i) = line.find(';') {
        end = end.min(i);
    }
    if let Some(i) = line.find('#') {
        end = end.min(i);
    }
    &line[..end]
}

fn split_leading_label(line: &str) -> Option<(&str, &str)> {
    let colon = line.find(':')?;
    let first_ws = line.find(char::is_whitespace).unwrap_or(line.len());
    if colon >= first_ws {
        return None;
    }
    let (label, rest) = line.split_at(colon);
    Some((label.trim(), rest[1..].trim()))
}

fn valid_label(label: &str) -> bool {
    let mut chars = label.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn parse_reg(token: &str, line: usize) -> Result<u8, V2AsmError> {
    let t = token.trim().to_ascii_uppercase();
    if t == "SP" {
        return Ok(15);
    }
    if !t.starts_with('R') {
        return Err(asm_err(line, format!("expected register, got '{token}'")));
    }
    let idx = t[1..]
        .parse::<u8>()
        .map_err(|_| asm_err(line, format!("invalid register '{token}'")))?;
    if idx > 15 {
        return Err(asm_err(
            line,
            format!("register out of range '{token}', expected R0..R15"),
        ));
    }
    Ok(idx)
}

/// Parse a numeric literal (decimal or 0x hex) without symbol resolution.
fn parse_u16_literal(token: &str, line: usize) -> Result<u16, V2AsmError> {
    let t = token.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u16::from_str_radix(hex, 16)
            .map_err(|_| asm_err(line, format!("invalid hex immediate '{token}'")));
    }
    t.parse::<u16>()
        .map_err(|_| asm_err(line, format!("invalid immediate '{token}'")))
}

/// Parse a u16 value: tries numeric literal first, then symbol table lookup.
fn parse_u16(token: &str, symbols: &HashMap<String, u16>, line: usize) -> Result<u16, V2AsmError> {
    let t = token.trim();
    if let Ok(v) = parse_u16_literal(t, line) {
        return Ok(v);
    }
    let key = t.to_string();
    if let Some(&v) = symbols.get(&key) {
        return Ok(v);
    }
    Err(asm_err(
        line,
        format!("unknown symbol or invalid immediate '{token}'"),
    ))
}

fn parse_u8(token: &str, symbols: &HashMap<String, u16>, line: usize) -> Result<u8, V2AsmError> {
    let value = parse_u16(token, symbols, line)?;
    if value > 0xFF {
        return Err(asm_err(
            line,
            format!("immediate out of range '{token}', expected 0..255"),
        ));
    }
    Ok(value as u8)
}

/// Parse a memory immediate address (0-63, RAM range).
fn parse_target(
    token: &str,
    symbols: &HashMap<String, u16>,
    line: usize,
) -> Result<u8, V2AsmError> {
    let value = parse_u16(token, symbols, line)?;
    if value > 127 {
        return Err(asm_err(
            line,
            format!("memory address out of range '{token}', expected 0..127"),
        ));
    }
    Ok(value as u8)
}

/// Sprint 187: Parse a branch target (0-127, 7-bit PC range).
fn parse_branch_target(
    token: &str,
    symbols: &HashMap<String, u16>,
    line: usize,
) -> Result<u8, V2AsmError> {
    let value = parse_u16(token, symbols, line)?;
    if value > 127 {
        return Err(asm_err(
            line,
            format!("branch target out of range '{token}', expected 0..127"),
        ));
    }
    Ok(value as u8)
}

/// Sprint 371 (Gate B.3.1): parse a wide branch target (0-65535, 16-bit PC range)
/// for `JMP.W` / `CALL.W`. Labels resolving above the 7-bit limit are legal here.
fn parse_branch_target_wide(
    token: &str,
    symbols: &HashMap<String, u16>,
    line: usize,
) -> Result<u16, V2AsmError> {
    parse_u16(token, symbols, line)
}

fn split_mnemonic_and_args(line: &str) -> (&str, Vec<&str>) {
    let mut parts = line.splitn(2, char::is_whitespace);
    let mnemonic = parts.next().unwrap_or("").trim();
    let args_str = parts.next().unwrap_or("").trim();
    let args = if args_str.is_empty() {
        Vec::new()
    } else {
        args_str.split(',').map(|a| a.trim()).collect()
    };
    (mnemonic, args)
}

enum MemOp {
    Reg(u8),
    Imm(u8),
    /// Sprint 175: Register+offset addressing, e.g. [R1+5].
    RegOffset(u8, u8),
}

fn parse_mem_operand(
    token: &str,
    symbols: &HashMap<String, u16>,
    line: usize,
) -> Result<MemOp, V2AsmError> {
    let trimmed = token.trim();
    if !(trimmed.starts_with('[') && trimmed.ends_with(']')) {
        return Err(asm_err(
            line,
            format!("expected memory operand like [R0] or [42], got '{token}'"),
        ));
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    // Sprint 175: detect [R+N] base+offset syntax.
    if let Some(plus_pos) = inner.find('+') {
        let reg_part = inner[..plus_pos].trim();
        let off_part = inner[plus_pos + 1..].trim();
        let reg = parse_reg(reg_part, line)?;
        let off = parse_u8(off_part, symbols, line)?;
        return Ok(MemOp::RegOffset(reg, off));
    }
    if let Ok(reg) = parse_reg(inner, line) {
        return Ok(MemOp::Reg(reg));
    }
    let imm = parse_target(inner, symbols, line)?;
    Ok(MemOp::Imm(imm))
}

fn encode_rr(base: fn(u8, u8) -> u32, rd: u8, rs: u8) -> u32 {
    v2_with_reg_ext(base(rd & 0x07, rs & 0x07), rd >= 8, rs >= 8)
}

fn encode_r(base: fn(u8) -> u32, rd: u8) -> u32 {
    v2_with_reg_ext(base(rd & 0x07), rd >= 8, false)
}

fn encode_ri(base: fn(u8, u8) -> u32, rd: u8, imm8: u8) -> u32 {
    v2_with_reg_ext(base(rd & 0x07, imm8), rd >= 8, false)
}

/// Sprint 174: Encode a wide-immediate instruction with 16-bit constant.
fn encode_ri_wide(base: fn(u8, u8) -> u32, rd: u8, imm16: u16) -> u32 {
    let lo = (imm16 & 0xFF) as u8;
    let hi = ((imm16 >> 8) & 0xFF) as u16;
    let mut ext = EXT_WIDE_IMM | (hi << 8);
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    v2_with_ext(base(rd & 0x07, lo), ext)
}

/// Sprint 175: Encode a base+offset memory instruction.
/// `opcode` is the raw opcode (0x16..0x19), `rd` is the data register, `rs` is the base register,
/// `offset` is the 8-bit offset. Uses enc() format with EXT_OFFSET extension.
fn encode_rr_offset(opcode: u8, rd: u8, rs: u8, offset: u8) -> u32 {
    let base =
        ((opcode as u32 & 0x1F) << 11) | (((rd & 0x07) as u32) << 8) | (((rs & 0x07) as u32) << 5);
    let mut ext = EXT_OFFSET | ((offset as u16) << 8);
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    if rs >= 8 {
        ext |= EXT_RS_HI;
    }
    v2_with_ext(base, ext)
}

fn expect_args_count(
    args: &[&str],
    expected: usize,
    line: usize,
    mnemonic: &str,
) -> Result<(), V2AsmError> {
    if args.len() != expected {
        return Err(asm_err(
            line,
            format!(
                "{mnemonic} expects {expected} operand(s), got {}",
                args.len()
            ),
        ));
    }
    Ok(())
}

/// Build the default symbol table with built-in MMIO address names.
fn builtin_symbols() -> HashMap<String, u16> {
    let mut m = HashMap::new();
    m.insert("DISPLAY_CMD".to_string(), MMIO_DISPLAY_CMD as u16);
    m.insert("DISPLAY_STATUS".to_string(), MMIO_DISPLAY_STATUS as u16);
    m.insert("TIMER".to_string(), MMIO_TIMER_CYCLE as u16);
    m.insert("CONSOLE_DATA".to_string(), MMIO_CONSOLE_DATA as u16);
    m.insert("CONSOLE_COUNT".to_string(), MMIO_CONSOLE_COUNT as u16);
    m.insert("RNG".to_string(), MMIO_RNG_DATA as u16);
    m.insert("MAILBOX_IN".to_string(), MMIO_MAILBOX_IN as u16);
    m.insert("MAILBOX_OUT".to_string(), MMIO_MAILBOX_OUT as u16);
    m.insert("PBIT_CTRL".to_string(), MMIO_PBIT_CTRL as u16);
    m.insert("PBIT_RESULT".to_string(), MMIO_PBIT_RESULT as u16);
    m.insert("MATH_A".to_string(), MMIO_MATH_A as u16);
    m.insert("MATH_B".to_string(), MMIO_MATH_B as u16);
    m.insert("MATH_CMD".to_string(), MMIO_MATH_CMD as u16);
    m.insert("MATH_RESULT".to_string(), MMIO_MATH_RESULT as u16);
    m.insert("SNN_DATA".to_string(), MMIO_SNN_DATA as u16);
    m.insert("SNN_CMD".to_string(), MMIO_SNN_CMD as u16);
    m.insert("QUANTUM_CMD".to_string(), MMIO_QUANTUM_CMD as u16);
    m.insert("QUANTUM_QUBIT".to_string(), MMIO_QUANTUM_QUBIT as u16);
    m.insert("QUANTUM_DATA".to_string(), MMIO_QUANTUM_DATA as u16);
    m.insert("QUANTUM_PARAM".to_string(), MMIO_QUANTUM_PARAM as u16);
    m.insert("DATASET_CMD".to_string(), MMIO_DATASET_CMD as u16);
    m.insert("DATASET_DATA".to_string(), MMIO_DATASET_DATA as u16);
    m.insert("DATASET_STATUS".to_string(), MMIO_DATASET_STATUS as u16);
    m
}

fn is_directive(line: &str) -> bool {
    let upper = line.to_ascii_uppercase();
    upper.starts_with(".EQU ") || upper.starts_with(".ORG ")
}

/// Assemble V2 source code into 32-bit instruction words.
pub fn assemble_v2(source: &str) -> Result<Vec<u32>, V2AsmError> {
    let mut symbols = builtin_symbols();
    let mut pc = 0usize;

    // Pass 1: collect .equ constants and labels, accounting for .org and PUSH/POP sizes.
    for (line_idx, raw) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let mut line = strip_comments(raw).trim();
        if line.is_empty() {
            continue;
        }

        // .equ directive
        let upper = line.to_ascii_uppercase();
        if upper.starts_with(".EQU ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 3 {
                return Err(asm_err(line_no, ".equ requires NAME VALUE"));
            }
            let name = parts[1].to_string();
            if !valid_label(&name) {
                return Err(asm_err(line_no, format!("invalid .equ name '{name}'")));
            }
            let value = parse_u16_literal(parts[2], line_no)?;
            if symbols.contains_key(&name) {
                return Err(asm_err(line_no, format!("duplicate symbol '{name}'")));
            }
            symbols.insert(name, value);
            continue;
        }

        // .org directive
        if upper.starts_with(".ORG ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(asm_err(line_no, ".org requires ADDRESS"));
            }
            let target = parse_u16_literal(parts[1], line_no)? as usize;
            if target < pc {
                return Err(asm_err(
                    line_no,
                    format!(".org {target} would go backward (PC={pc})"),
                ));
            }
            // Sprint 374 (Gate D.3): the wide PC (B.3) addresses up to 65,536
            // instructions; programs above 128 words require a wide-PC run config.
            if (target as usize) > V2_MAX_PROGRAM_WORDS {
                return Err(asm_err(
                    line_no,
                    format!(".org {target} exceeds the {V2_MAX_PROGRAM_WORDS}-word program limit"),
                ));
            }
            pc = target;
            continue;
        }

        // Labels
        if let Some((label, rest)) = split_leading_label(line) {
            if !valid_label(label) {
                return Err(asm_err(line_no, format!("invalid label '{label}'")));
            }
            if symbols.contains_key(label) {
                return Err(asm_err(line_no, format!("duplicate symbol '{label}'")));
            }
            symbols.insert(label.to_string(), pc as u16);
            line = rest;
            if line.is_empty() {
                continue;
            }
        }

        // Count instruction words (PUSH/POP expand to 2)
        let (mnemonic_raw, _) = split_mnemonic_and_args(line);
        let mnemonic = mnemonic_raw.to_ascii_uppercase();
        let count = match mnemonic.as_str() {
            "PUSH" | "POP" => 2,
            _ => 1,
        };
        pc += count;
        if pc > V2_MAX_PROGRAM_WORDS {
            return Err(asm_err(
                line_no,
                format!("program exceeds {V2_MAX_PROGRAM_WORDS} instruction words"),
            ));
        }
    }

    // Pass 2: emission.
    let mut program = Vec::new();
    for (line_idx, raw) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let mut line = strip_comments(raw).trim();
        if line.is_empty() {
            continue;
        }

        // Skip .equ
        if is_directive(line) {
            let upper = line.to_ascii_uppercase();
            if upper.starts_with(".EQU ") {
                continue;
            }
            // .org — pad with NOPs
            if upper.starts_with(".ORG ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let target = parse_u16_literal(parts[1], line_no)? as usize;
                while program.len() < target {
                    program.push(v2_nop());
                }
                continue;
            }
        }

        // Skip labels
        if let Some((_, rest)) = split_leading_label(line) {
            line = rest;
            if line.is_empty() {
                continue;
            }
        }

        let (mnemonic_raw, args) = split_mnemonic_and_args(line);
        let mnemonic = mnemonic_raw.to_ascii_uppercase();
        match mnemonic.as_str() {
            "PUSH" => {
                expect_args_count(&args, 1, line_no, "PUSH")?;
                let rd = parse_reg(args[0], line_no)?;
                // ST [R15], Rd   (R15 = SP, high-bank)
                program.push(v2_with_reg_ext(v2_st(15 & 0x07, rd & 0x07), rd >= 8, true));
                // DEC R15
                program.push(v2_with_reg_ext(v2_dec(15 & 0x07), true, false));
            }
            "POP" => {
                expect_args_count(&args, 1, line_no, "POP")?;
                let rd = parse_reg(args[0], line_no)?;
                // INC R15
                program.push(v2_with_reg_ext(v2_inc(15 & 0x07), true, false));
                // LD Rd, [R15]
                program.push(v2_with_reg_ext(v2_ld(rd & 0x07, 15 & 0x07), rd >= 8, true));
            }
            "NOP" => {
                expect_args_count(&args, 0, line_no, "NOP")?;
                program.push(v2_nop());
            }
            "HALT" => {
                expect_args_count(&args, 0, line_no, "HALT")?;
                program.push(v2_halt());
            }
            "RET" => {
                expect_args_count(&args, 0, line_no, "RET")?;
                program.push(v2_ret());
            }
            // Sprint 364 (Gate C): LR↔reg bridge for nested-call save/restore.
            // MFLR Rd  -> Rd = LR    (callee captures return address at entry)
            // MTLR Rs  -> LR = Rs    (callee restores LR before RET)
            // Typical use: MFLR R8 | PUSH R8 | ... CALL ... | POP R8 | MTLR R8 | RET
            "MFLR" => {
                expect_args_count(&args, 1, line_no, "MFLR")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_mflr, rd));
            }
            "MTLR" => {
                expect_args_count(&args, 1, line_no, "MTLR")?;
                let rs = parse_reg(args[0], line_no)?;
                // MTLR uses the rs field (source). encode_r sets rd_hi only — so
                // build the bank-extension manually for rs.
                program.push(v2_with_reg_ext(v2_mtlr(rs & 0x07), false, rs >= 8));
            }
            "MOV" => {
                expect_args_count(&args, 2, line_no, "MOV")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_mov, rd, rs));
            }
            // Sprint 177: Conditional move mnemonics
            "MOVZ" => {
                expect_args_count(&args, 2, line_no, "MOVZ")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_movz, rd, rs));
            }
            "MOVNZ" => {
                expect_args_count(&args, 2, line_no, "MOVNZ")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_movnz, rd, rs));
            }
            "MOVC" => {
                expect_args_count(&args, 2, line_no, "MOVC")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_movc, rd, rs));
            }
            "MOVNC" => {
                expect_args_count(&args, 2, line_no, "MOVNC")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_movnc, rd, rs));
            }
            "LDI" => {
                expect_args_count(&args, 2, line_no, "LDI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                if imm > 0xFF {
                    // Sprint 174: Auto-widen to LDI.W when value > 255.
                    program.push(encode_ri_wide(v2_ldi, rd, imm));
                } else {
                    program.push(encode_ri(v2_ldi, rd, imm as u8));
                }
            }
            "ADD" => {
                expect_args_count(&args, 2, line_no, "ADD")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_add, rd, rs));
            }
            "MUL" => {
                expect_args_count(&args, 2, line_no, "MUL")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_mul, rd, rs));
            }
            "SUB" => {
                expect_args_count(&args, 2, line_no, "SUB")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_sub, rd, rs));
            }
            "AND" => {
                expect_args_count(&args, 2, line_no, "AND")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_and, rd, rs));
            }
            "OR" => {
                expect_args_count(&args, 2, line_no, "OR")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_or, rd, rs));
            }
            "XOR" => {
                expect_args_count(&args, 2, line_no, "XOR")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_xor, rd, rs));
            }
            "CMP" => {
                expect_args_count(&args, 2, line_no, "CMP")?;
                let rd = parse_reg(args[0], line_no)?;
                let rs = parse_reg(args[1], line_no)?;
                program.push(encode_rr(v2_cmp, rd, rs));
            }
            "NOT" => {
                expect_args_count(&args, 1, line_no, "NOT")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_not, rd));
            }
            // Sprint 176: Bit manipulation sub-functions
            "CLZ" => {
                expect_args_count(&args, 1, line_no, "CLZ")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_clz, rd));
            }
            "CTZ" => {
                expect_args_count(&args, 1, line_no, "CTZ")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_ctz, rd));
            }
            "POPCNT" => {
                expect_args_count(&args, 1, line_no, "POPCNT")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_popcnt, rd));
            }
            "NEG" => {
                expect_args_count(&args, 1, line_no, "NEG")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_neg, rd));
            }
            "INC" => {
                expect_args_count(&args, 1, line_no, "INC")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_inc, rd));
            }
            "DEC" => {
                expect_args_count(&args, 1, line_no, "DEC")?;
                let rd = parse_reg(args[0], line_no)?;
                program.push(encode_r(v2_dec, rd));
            }
            "SHL" => {
                expect_args_count(&args, 2, line_no, "SHL")?;
                let rd = parse_reg(args[0], line_no)?;
                let amount = parse_u8(args[1], &symbols, line_no)?;
                if amount > 7 {
                    return Err(asm_err(
                        line_no,
                        format!("SHL amount out of range '{}', expected 0..7", args[1]),
                    ));
                }
                program.push(encode_ri(v2_shl, rd, amount));
            }
            "SHR" => {
                expect_args_count(&args, 2, line_no, "SHR")?;
                let rd = parse_reg(args[0], line_no)?;
                let amount = parse_u8(args[1], &symbols, line_no)?;
                if amount > 7 {
                    return Err(asm_err(
                        line_no,
                        format!("SHR amount out of range '{}', expected 0..7", args[1]),
                    ));
                }
                program.push(encode_ri(v2_shr, rd, amount));
            }
            "SRA" => {
                expect_args_count(&args, 2, line_no, "SRA")?;
                let rd = parse_reg(args[0], line_no)?;
                let amount = parse_u8(args[1], &symbols, line_no)?;
                if amount > 7 {
                    return Err(asm_err(
                        line_no,
                        format!("SRA amount out of range '{}', expected 0..7", args[1]),
                    ));
                }
                program.push(encode_ri(v2_sra, rd, amount));
            }
            "ADDI" => {
                expect_args_count(&args, 2, line_no, "ADDI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u8(args[1], &symbols, line_no)?;
                program.push(encode_ri(v2_addi, rd, imm));
            }
            "SUBI" => {
                expect_args_count(&args, 2, line_no, "SUBI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u8(args[1], &symbols, line_no)?;
                program.push(encode_ri(v2_subi, rd, imm));
            }
            "ANDI" => {
                expect_args_count(&args, 2, line_no, "ANDI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u8(args[1], &symbols, line_no)?;
                program.push(encode_ri(v2_andi, rd, imm));
            }
            "ORI" => {
                expect_args_count(&args, 2, line_no, "ORI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u8(args[1], &symbols, line_no)?;
                program.push(encode_ri(v2_ori, rd, imm));
            }
            "XORI" => {
                expect_args_count(&args, 2, line_no, "XORI")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u8(args[1], &symbols, line_no)?;
                program.push(encode_ri(v2_xori, rd, imm));
            }
            // Sprint 174: Explicit wide-immediate variants.
            "LDI.W" => {
                expect_args_count(&args, 2, line_no, "LDI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_ldi, rd, imm));
            }
            "ADDI.W" => {
                expect_args_count(&args, 2, line_no, "ADDI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_addi, rd, imm));
            }
            "SUBI.W" => {
                expect_args_count(&args, 2, line_no, "SUBI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_subi, rd, imm));
            }
            "ANDI.W" => {
                expect_args_count(&args, 2, line_no, "ANDI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_andi, rd, imm));
            }
            "ORI.W" => {
                expect_args_count(&args, 2, line_no, "ORI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_ori, rd, imm));
            }
            "XORI.W" => {
                expect_args_count(&args, 2, line_no, "XORI.W")?;
                let rd = parse_reg(args[0], line_no)?;
                let imm = parse_u16(args[1], &symbols, line_no)?;
                program.push(encode_ri_wide(v2_xori, rd, imm));
            }
            "LD" => {
                expect_args_count(&args, 2, line_no, "LD")?;
                let rd = parse_reg(args[0], line_no)?;
                let mem = parse_mem_operand(args[1], &symbols, line_no)?;
                match mem {
                    MemOp::Reg(rs) => program.push(encode_rr(v2_ld, rd, rs)),
                    MemOp::RegOffset(rs, offset) => {
                        program.push(encode_rr_offset(0x16, rd, rs, offset));
                    }
                    MemOp::Imm(_) => {
                        return Err(asm_err(
                            line_no,
                            "LD expects register-indirect source like [R0] or [R0+5]",
                        ));
                    }
                }
            }
            "ST" => {
                expect_args_count(&args, 2, line_no, "ST")?;
                let mem = parse_mem_operand(args[0], &symbols, line_no)?;
                let rd = parse_reg(args[1], line_no)?;
                match mem {
                    MemOp::Reg(rs) => {
                        program.push(v2_with_reg_ext(
                            v2_st(rs & 0x07, rd & 0x07),
                            rd >= 8,
                            rs >= 8,
                        ));
                    }
                    MemOp::RegOffset(rs, offset) => {
                        program.push(encode_rr_offset(0x17, rd, rs, offset));
                    }
                    MemOp::Imm(_) => {
                        return Err(asm_err(
                            line_no,
                            "ST expects register-indirect destination like [R0] or [R0+5]",
                        ));
                    }
                }
            }
            "LDB" => {
                expect_args_count(&args, 2, line_no, "LDB")?;
                let rd = parse_reg(args[0], line_no)?;
                let mem = parse_mem_operand(args[1], &symbols, line_no)?;
                match mem {
                    MemOp::Imm(addr) => program.push(encode_ri(v2_ldb, rd, addr)),
                    MemOp::RegOffset(rs, offset) => {
                        program.push(encode_rr_offset(0x18, rd, rs, offset));
                    }
                    MemOp::Reg(_) => {
                        return Err(asm_err(
                            line_no,
                            "LDB expects direct [42] or offset [R0+5] source",
                        ));
                    }
                }
            }
            "STB" => {
                expect_args_count(&args, 2, line_no, "STB")?;
                let mem = parse_mem_operand(args[0], &symbols, line_no)?;
                let rd = parse_reg(args[1], line_no)?;
                match mem {
                    MemOp::Imm(addr) => program.push(encode_ri(v2_stb, rd, addr)),
                    MemOp::RegOffset(rs, offset) => {
                        program.push(encode_rr_offset(0x19, rd, rs, offset));
                    }
                    MemOp::Reg(_) => {
                        return Err(asm_err(
                            line_no,
                            "STB expects direct [42] or offset [R0+5] destination",
                        ));
                    }
                }
            }
            "JMP" => {
                expect_args_count(&args, 1, line_no, "JMP")?;
                program.push(v2_jmp(parse_branch_target(args[0], &symbols, line_no)?));
            }
            "JZ" => {
                expect_args_count(&args, 1, line_no, "JZ")?;
                program.push(v2_jz(parse_branch_target(args[0], &symbols, line_no)?));
            }
            "JNZ" => {
                expect_args_count(&args, 1, line_no, "JNZ")?;
                program.push(v2_jnz(parse_branch_target(args[0], &symbols, line_no)?));
            }
            "JC" => {
                expect_args_count(&args, 1, line_no, "JC")?;
                program.push(v2_jc(parse_branch_target(args[0], &symbols, line_no)?));
            }
            "JNC" => {
                expect_args_count(&args, 1, line_no, "JNC")?;
                program.push(v2_jnc(parse_branch_target(args[0], &symbols, line_no)?));
            }
            "CALL" => {
                expect_args_count(&args, 1, line_no, "CALL")?;
                program.push(v2_call(parse_branch_target(args[0], &symbols, line_no)?));
            }
            // Sprint 371 (Gate B.3.1): wide-immediate far-jump / far-call. Targets
            // any instruction 0..65535 directly from the instruction (no register
            // load). Companion to the physical wide PC (Gate B.3).
            "JMP.W" => {
                expect_args_count(&args, 1, line_no, "JMP.W")?;
                program.push(v2_jmp_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            "CALL.W" => {
                expect_args_count(&args, 1, line_no, "CALL.W")?;
                program.push(v2_call_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            // Sprint 374 (Gate D.3): wide conditional branches — reach any instruction
            // 0..65535 when the flag test is taken (the `.W` companions to JZ/JNZ/JC/JNC).
            "JZ.W" => {
                expect_args_count(&args, 1, line_no, "JZ.W")?;
                program.push(v2_jz_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            "JNZ.W" => {
                expect_args_count(&args, 1, line_no, "JNZ.W")?;
                program.push(v2_jnz_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            "JC.W" => {
                expect_args_count(&args, 1, line_no, "JC.W")?;
                program.push(v2_jc_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            "JNC.W" => {
                expect_args_count(&args, 1, line_no, "JNC.W")?;
                program.push(v2_jnc_w(parse_branch_target_wide(
                    args[0], &symbols, line_no,
                )?));
            }
            // Sprint 369 (Gate B.2): register-indirect far-jump / far-call.
            // Target is regs[rd] — reaches any instruction address past the 8-bit
            // branch immediate. JMPR Rd / CALLR Rd.
            "JMPR" => {
                expect_args_count(&args, 1, line_no, "JMPR")?;
                program.push(v2_jmpr(parse_reg(args[0], line_no)?));
            }
            "CALLR" => {
                expect_args_count(&args, 1, line_no, "CALLR")?;
                program.push(v2_callr(parse_reg(args[0], line_no)?));
            }
            _ => {
                return Err(asm_err(
                    line_no,
                    format!("unknown mnemonic '{mnemonic_raw}'"),
                ));
            }
        }
    }

    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_asm_basic_program() {
        let src = r#"
            LDI R0, 10
            LDI R1, 3
            ADD R0, R1
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(
            program,
            vec![v2_ldi(0, 10), v2_ldi(1, 3), v2_add(0, 1), v2_halt()]
        );
    }

    #[test]
    fn test_v2_asm_labels_branch() {
        let src = r#"
            LDI R0, 2
        loop:
            DEC R0
            JNZ loop
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program, vec![v2_ldi(0, 2), v2_dec(0), v2_jnz(1), v2_halt()]);
    }

    #[test]
    fn test_v2_asm_high_register_extension_bits() {
        let src = r#"
            LDI R8, 7
            MOV R9, R8
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(
            program,
            vec![
                v2_with_reg_ext(v2_ldi(0, 7), true, false),
                v2_with_reg_ext(v2_mov(1, 0), true, true),
                v2_halt()
            ]
        );
    }

    #[test]
    fn test_v2_asm_memory_forms() {
        let src = r#"
            LDI R0, 12
            LDI R1, 99
            ST [R0], R1
            LD R2, [R0]
            STB [56], R2
            LDB R3, [56]
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 7);
    }

    #[test]
    fn test_v2_asm_invalid_register_error() {
        let src = "LDI R16, 1";
        let err = assemble_v2(src).expect_err("expected failure");
        assert!(err.to_string().contains("R0..R15"));
    }

    #[test]
    fn test_v2_asm_duplicate_label_error() {
        let src = r#"
        loop: NOP
        loop: HALT
        "#;
        let err = assemble_v2(src).expect_err("expected failure");
        assert!(err.to_string().contains("duplicate symbol"));
    }

    // --- Sprint 155 new features ---

    #[test]
    fn test_v2_asm_sp_alias() {
        let src = r#"
            LDI SP, 47
            MOV R0, SP
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(
            program,
            vec![
                v2_with_reg_ext(v2_ldi(7, 47), true, false), // R15 = SP
                v2_with_reg_ext(v2_mov(0, 7), false, true),  // MOV R0, R15
                v2_halt()
            ]
        );
    }

    #[test]
    fn test_v2_asm_equ_constants() {
        let src = r#"
            .equ MY_VAL 42
            .equ OTHER 0x0A
            LDI R0, MY_VAL
            LDI R1, OTHER
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program, vec![v2_ldi(0, 42), v2_ldi(1, 10), v2_halt()]);
    }

    #[test]
    fn test_v2_asm_equ_in_memory_operand() {
        let src = r#"
            .equ ADDR 56
            LDI R0, 7
            STB [ADDR], R0
            LDB R1, [ADDR]
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(
            program,
            vec![v2_ldi(0, 7), v2_stb(0, 56), v2_ldb(1, 56), v2_halt()]
        );
    }

    #[test]
    fn test_v2_asm_builtin_mmio_names() {
        let src = r#"
            LDI R0, 0x55
            STB [TIMER], R0
            LDB R1, [RNG]
            STB [DISPLAY_CMD], R0
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 5);
        // Verify TIMER resolves to 56, RNG to 59, DISPLAY_CMD to 48
        assert_eq!(program[1], v2_stb(0, 56));
        assert_eq!(program[2], v2_ldb(1, 59));
        assert_eq!(program[3], v2_stb(0, 48));
    }

    #[test]
    fn test_v2_asm_org_directive() {
        let src = r#"
            LDI R0, 1
            .org 4
            LDI R1, 2
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        // [0] LDI R0,1  [1..3] NOP NOP NOP  [4] LDI R1,2  [5] HALT
        assert_eq!(program.len(), 6);
        assert_eq!(program[0], v2_ldi(0, 1));
        assert_eq!(program[1], v2_nop());
        assert_eq!(program[2], v2_nop());
        assert_eq!(program[3], v2_nop());
        assert_eq!(program[4], v2_ldi(1, 2));
        assert_eq!(program[5], v2_halt());
    }

    #[test]
    fn test_v2_asm_org_with_label() {
        let src = r#"
            JMP start
            .org 4
        start:
            LDI R0, 99
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 6);
        assert_eq!(program[0], v2_jmp(4)); // start is at address 4
        assert_eq!(program[4], v2_ldi(0, 99));
    }

    /// Sprint 369 (Gate B.2): JMPR / CALLR text mnemonics assemble to the
    /// register-indirect far-jump encoders, including the R8-R15 high-register
    /// extension bit.
    #[test]
    fn test_v2_asm_jmpr_callr() {
        let src = r#"
            JMPR R3
            CALLR R1
            JMPR R10
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 4);
        assert_eq!(program[0], v2_jmpr(3));
        assert_eq!(program[1], v2_callr(1));
        assert_eq!(program[2], v2_jmpr(10)); // high register sets EXT_RD_HI
    }

    /// Sprint 371 (Gate B.3.1): JMP.W / CALL.W assemble to the wide-immediate
    /// encoders, and a label resolving above the 7-bit branch limit (300) is legal
    /// for the `.W` forms (it would be rejected by plain JMP/CALL).
    #[test]
    fn test_v2_asm_jmp_w_call_w() {
        let src = r#"
            .equ FAR 300
            JMP.W FAR
            CALL.W 65535
            CALL.W 5
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 4);
        assert_eq!(program[0], v2_jmp_w(300));
        assert_eq!(program[1], v2_call_w(65535));
        assert_eq!(program[2], v2_call_w(5));
        // A wide CALL to a small target still round-trips to the same value a plain
        // CALL would *decode* to (imm_val == imm8), but carries EXT_WIDE_IMM.
        assert_ne!(program[2], v2_call(5));
    }

    /// Sprint 374 (Gate D.3): wide conditional branches assemble to the `.W` encoders,
    /// including a far label (> 7-bit window).
    #[test]
    fn test_v2_asm_wide_conditional_branches() {
        let src = r#"
            .equ FAR 300
            JZ.W FAR
            JNZ.W 400
            JC.W 500
            JNC.W 600
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 5);
        assert_eq!(program[0], v2_jz_w(300));
        assert_eq!(program[1], v2_jnz_w(400));
        assert_eq!(program[2], v2_jc_w(500));
        assert_eq!(program[3], v2_jnc_w(600));
    }

    /// Sprint 374 (Gate D.3): a program larger than 128 words now assembles (the old
    /// 128-word ROM cap is lifted to the wide-PC limit).
    #[test]
    fn test_v2_asm_program_above_128_words() {
        let mut src = String::new();
        for _ in 0..200 {
            src.push_str("NOP\n");
        }
        src.push_str("HALT\n");
        let program = assemble_v2(&src).expect("a >128-word program must assemble");
        assert_eq!(program.len(), 201);
    }

    /// Sprint 371: plain JMP/CALL still reject targets above the 7-bit limit — the
    /// wide range is opt-in via the `.W` forms.
    #[test]
    fn test_v2_asm_plain_jmp_rejects_wide_target() {
        let err = assemble_v2("JMP 300\nHALT\n").expect_err("expected out-of-range failure");
        assert!(format!("{err:?}").contains("range"), "got: {err:?}");
    }

    #[test]
    fn test_v2_asm_push_pop() {
        let src = r#"
            LDI SP, 47     ; init stack pointer
            LDI R0, 100
            PUSH R0
            LDI R0, 0
            POP R1
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        // LDI SP,47 | LDI R0,100 | ST [R15],R0 | DEC R15 | LDI R0,0 | INC R15 | LD R1,[R15] | HALT
        assert_eq!(program.len(), 8);
        // PUSH R0 = ST [R15], R0  +  DEC R15
        assert_eq!(program[2], v2_with_reg_ext(v2_st(7, 0), false, true)); // ST [R15], R0
        assert_eq!(program[3], v2_with_reg_ext(v2_dec(7), true, false)); // DEC R15
        // POP R1 = INC R15  +  LD R1, [R15]
        assert_eq!(program[5], v2_with_reg_ext(v2_inc(7), true, false)); // INC R15
        assert_eq!(program[6], v2_with_reg_ext(v2_ld(1, 7), false, true)); // LD R1, [R15]
    }

    /// Sprint 364 (Gate C): MFLR / MTLR assembler mnemonics produce the same
    /// encoding as the v2_mflr / v2_mtlr builders, including bank extension.
    #[test]
    fn test_v2_asm_mflr_mtlr() {
        let src = r#"
            MFLR R8       ; capture return address into a callee-saved reg
            MTLR R8       ; restore LR before RET
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        assert_eq!(program.len(), 3);
        // MFLR R8 -> v2_mflr(0) base, rd_hi=true
        assert_eq!(program[0], v2_with_reg_ext(v2_mflr(0), true, false));
        // MTLR R8 -> v2_mtlr(0) base, rs_hi=true (source uses rs field)
        assert_eq!(program[1], v2_with_reg_ext(v2_mtlr(0), false, true));
    }

    #[test]
    fn test_v2_asm_push_pop_label_offsets() {
        // PUSH/POP expand to 2 instructions each, labels after them must account for that.
        let src = r#"
            PUSH R0
            PUSH R1
        after_push:
            NOP
            JMP after_push
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble failed");
        // PUSH R0 (2) + PUSH R1 (2) + NOP (1) + JMP (1) + HALT (1) = 7
        assert_eq!(program.len(), 7);
        // after_push is at PC=4 (after 2+2 instructions)
        assert_eq!(program[5], v2_jmp(4));
    }

    #[test]
    fn test_v2_asm_equ_duplicate_errors() {
        // .equ clashing with built-in
        let src = ".equ TIMER 99\nHALT";
        let err = assemble_v2(src).expect_err("expected failure");
        assert!(err.to_string().contains("duplicate symbol"));
    }

    #[test]
    fn test_v2_asm_org_backward_error() {
        let src = r#"
            NOP
            NOP
            NOP
            .org 2
        "#;
        let err = assemble_v2(src).expect_err("expected failure");
        assert!(err.to_string().contains("backward"));
    }
}
