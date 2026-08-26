//! V2 ISA components: decode and assembler helpers.
//!
//! Sprint 91 contract:
//! - 16-bit instruction words
//! - 8 base registers in instruction bits (`rd`/`rs` are 3-bit fields)
//! - Sprint 135 extension: `ext[0]=rd_hi`, `ext[1]=rs_hi` for banked R8-R15 access
//! - 32 opcode slots, all assigned

/// V2 ALU operation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V2AluOp {
    #[default]
    PassA = 0,
    PassB = 1,
    Add = 2,
    Sub = 3,
    And = 4,
    Or = 5,
    Xor = 6,
    Not = 7,
    Shl = 8,
    Shr = 9,
}

/// Software control bundle for V2 hybrid execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2ControlSignals {
    pub opcode: u8,
    pub rd: u8,
    pub rs: u8,
    pub imm5: u8,
    pub imm8: u8,
    pub jump_target: u8,

    pub alu_op: V2AluOp,
    pub reg_write_en: bool,
    pub update_flags_z: bool,
    pub update_flags_c: bool,
    pub use_immediate: bool,
    pub mem_read: bool,
    pub mem_write: bool,
    pub is_jmp: bool,
    pub is_jz: bool,
    pub is_jnz: bool,
    pub is_jc: bool,
    pub is_jnc: bool,
    pub is_call: bool,
    pub is_ret: bool,
    pub is_halt: bool,
}

impl V2ControlSignals {
    /// Decode one V2 instruction (only bits 15-0 are used for decode).
    pub fn decode(instruction: u32) -> Self {
        let opcode = ((instruction >> 11) & 0x1F) as u8;
        let rd = ((instruction >> 8) & 0x07) as u8;
        let rs = ((instruction >> 5) & 0x07) as u8;
        let imm5 = (instruction & 0x1F) as u8;
        let imm8 = (instruction & 0xFF) as u8;

        let mut s = Self {
            opcode,
            rd,
            rs,
            imm5,
            imm8,
            jump_target: imm8,
            ..Self::default()
        };

        match opcode {
            0x00 => {
                // NOP
            }
            0x01 => {
                // HALT
                s.is_halt = true;
            }
            0x02 => {
                // MOV rd, rs
                s.alu_op = V2AluOp::PassB;
                s.reg_write_en = true;
            }
            0x03 => {
                // LDI rd, imm8
                s.alu_op = V2AluOp::PassB;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
            }
            0x04 => {
                // ADD rd, rs
                s.alu_op = V2AluOp::Add;
                s.reg_write_en = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x05 => {
                // SUB rd, rs
                s.alu_op = V2AluOp::Sub;
                s.reg_write_en = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x06 => {
                // AND rd, rs
                s.alu_op = V2AluOp::And;
                s.reg_write_en = true;
                s.update_flags_z = true;
            }
            0x07 => {
                // OR rd, rs
                s.alu_op = V2AluOp::Or;
                s.reg_write_en = true;
                s.update_flags_z = true;
            }
            0x08 => {
                // XOR rd, rs
                s.alu_op = V2AluOp::Xor;
                s.reg_write_en = true;
                s.update_flags_z = true;
            }
            0x09 => {
                // NOT rd
                s.alu_op = V2AluOp::Not;
                s.reg_write_en = true;
                s.update_flags_z = true;
            }
            0x0A => {
                // NEG rd (0 - rd): Sprint 111 cleared use_imm — rs=rd encoding
                // means Tree B physically delivers reg[rd] as R operand.
                s.alu_op = V2AluOp::Sub;
                s.reg_write_en = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x0B => {
                // INC rd
                s.alu_op = V2AluOp::Add;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x0C => {
                // DEC rd
                s.alu_op = V2AluOp::Sub;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x0D => {
                // SHL rd, imm3
                s.alu_op = V2AluOp::Shl;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x0E => {
                // SHR rd, imm3
                s.alu_op = V2AluOp::Shr;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x0F => {
                // CMP rd, rs
                s.alu_op = V2AluOp::Sub;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x10 => {
                // ADDI rd, imm8
                s.alu_op = V2AluOp::Add;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x11 => {
                // SUBI rd, imm8
                s.alu_op = V2AluOp::Sub;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
                s.update_flags_c = true;
            }
            0x12 => {
                // ANDI rd, imm8
                s.alu_op = V2AluOp::And;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
            }
            0x13 => {
                // ORI rd, imm8
                s.alu_op = V2AluOp::Or;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
            }
            0x14 => {
                // XORI rd, imm8
                s.alu_op = V2AluOp::Xor;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
            }
            0x15 => {
                // RET
                s.is_ret = true;
            }
            0x16 => {
                // LD rd, [rs]
                s.mem_read = true;
                s.reg_write_en = true;
                s.update_flags_z = true;
            }
            0x17 => {
                // ST [rs], rd
                s.mem_write = true;
            }
            0x18 => {
                // LDB rd, [imm8]
                s.mem_read = true;
                s.reg_write_en = true;
                s.use_immediate = true;
                s.update_flags_z = true;
            }
            0x19 => {
                // STB [imm8], rd
                s.mem_write = true;
                s.use_immediate = true;
            }
            0x1A => {
                // JMP imm8
                s.is_jmp = true;
            }
            0x1B => {
                // JZ imm8
                s.is_jz = true;
            }
            0x1C => {
                // JNZ imm8
                s.is_jnz = true;
            }
            0x1D => {
                // JC imm8
                s.is_jc = true;
            }
            0x1E => {
                // JNC imm8
                s.is_jnc = true;
            }
            0x1F => {
                // CALL imm8
                s.is_call = true;
            }
            _ => {}
        }

        s
    }
}

const fn enc(op: u8, rd: u8, rs: u8, imm5: u8) -> u32 {
    ((op as u32 & 0x1F) << 11)
        | ((rd as u32 & 0x07) << 8)
        | ((rs as u32 & 0x07) << 5)
        | (imm5 as u32 & 0x1F)
}

const fn enc_imm8(op: u8, rd: u8, imm8: u8) -> u32 {
    ((op as u32 & 0x1F) << 11) | ((rd as u32 & 0x07) << 8) | imm8 as u32
}

const fn enc_branch(op: u8, target: u8) -> u32 {
    ((op as u32 & 0x1F) << 11) | target as u32
}

pub const fn v2_nop() -> u32 {
    enc(0x00, 0, 0, 0)
}
pub const fn v2_halt() -> u32 {
    enc(0x01, 0, 0, 0)
}
pub const fn v2_mov(rd: u8, rs: u8) -> u32 {
    enc(0x02, rd, rs, 0)
}
// Sprint 177: Conditional move sub-functions (opcode 0x02, imm5 selects condition)
pub const fn v2_movz(rd: u8, rs: u8) -> u32 {
    enc(0x02, rd, rs, 1)
}
pub const fn v2_movnz(rd: u8, rs: u8) -> u32 {
    enc(0x02, rd, rs, 2)
}
pub const fn v2_movc(rd: u8, rs: u8) -> u32 {
    enc(0x02, rd, rs, 3)
}
pub const fn v2_movnc(rd: u8, rs: u8) -> u32 {
    enc(0x02, rd, rs, 4)
}
// Sprint 364 (Gate C): LR↔reg bridge — sub-functions on opcode 0x02.
// MFLR Rd → regs[rd] = lr      (sub_fn = 5)
// MTLR Rs → lr = regs[rs]      (sub_fn = 6)
// These are the only way to read/write LR, which is what makes nested calls
// possible: a callee must save LR (PUSH LR pseudo = MFLR R0; PUSH R0) before
// issuing its own CALL, and restore (POP R0; MTLR R0) before RET.
// ISS-only today (physical executor still has LR as write-only from CALL).
pub const fn v2_mflr(rd: u8) -> u32 {
    enc(0x02, rd, 0, 5)
}
pub const fn v2_mtlr(rs: u8) -> u32 {
    enc(0x02, 0, rs, 6)
}
pub const fn v2_ldi(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x03, rd, imm8)
}
pub const fn v2_add(rd: u8, rs: u8) -> u32 {
    enc(0x04, rd, rs, 0)
}
/// Sprint 188: MUL rd, rs — unsigned multiply (lower 64 bits).
/// Sub-function on opcode 0x04: imm5 = 1 selects multiply.
pub const fn v2_mul(rd: u8, rs: u8) -> u32 {
    enc(0x04, rd, rs, 1)
}
pub const fn v2_sub(rd: u8, rs: u8) -> u32 {
    enc(0x05, rd, rs, 0)
}
pub const fn v2_and(rd: u8, rs: u8) -> u32 {
    enc(0x06, rd, rs, 0)
}
pub const fn v2_or(rd: u8, rs: u8) -> u32 {
    enc(0x07, rd, rs, 0)
}
pub const fn v2_xor(rd: u8, rs: u8) -> u32 {
    enc(0x08, rd, rs, 0)
}
pub const fn v2_not(rd: u8) -> u32 {
    enc(0x09, rd, 0, 0)
}
// Sprint 176: Bit manipulation sub-functions (opcode 0x09, imm5 selects operation)
pub const fn v2_clz(rd: u8) -> u32 {
    enc(0x09, rd, 0, 1)
}
pub const fn v2_ctz(rd: u8) -> u32 {
    enc(0x09, rd, 0, 2)
}
pub const fn v2_popcnt(rd: u8) -> u32 {
    enc(0x09, rd, 0, 3)
}
pub const fn v2_neg(rd: u8) -> u32 {
    enc(0x0A, rd, rd, 0) // rs=rd so Tree B carries reg[rd]
}
pub const fn v2_inc(rd: u8) -> u32 {
    enc(0x0B, rd, 0, 1) // imm5=1 for physical ir_low delivery (Phase B)
}
pub const fn v2_dec(rd: u8) -> u32 {
    enc(0x0C, rd, 0, 1) // imm5=1 for physical ir_low delivery (Phase B)
}
pub const fn v2_shl(rd: u8, amount: u8) -> u32 {
    enc(0x0D, rd, 0, amount & 0x07)
}
pub const fn v2_shr(rd: u8, amount: u8) -> u32 {
    enc(0x0E, rd, 0, amount & 0x07)
}
/// Sprint 188: SRA rd, amount — arithmetic right shift (sign-extending).
/// Sub-function on opcode 0x0E: bit 3 of imm5 = 1 selects arithmetic mode.
pub const fn v2_sra(rd: u8, amount: u8) -> u32 {
    enc(0x0E, rd, 0, 0x08 | (amount & 0x07))
}
pub const fn v2_cmp(rd: u8, rs: u8) -> u32 {
    enc(0x0F, rd, rs, 0)
}
pub const fn v2_addi(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x10, rd, imm8)
}
pub const fn v2_subi(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x11, rd, imm8)
}
pub const fn v2_andi(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x12, rd, imm8)
}
pub const fn v2_ori(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x13, rd, imm8)
}
pub const fn v2_xori(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x14, rd, imm8)
}
pub const fn v2_ret() -> u32 {
    enc(0x15, 0, 0, 0)
}
pub const fn v2_ld(rd: u8, rs: u8) -> u32 {
    enc(0x16, rd, rs, 0)
}
pub const fn v2_st(rs: u8, rd: u8) -> u32 {
    enc(0x17, rd, rs, 0)
}
pub const fn v2_ldb(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x18, rd, imm8)
}
pub const fn v2_stb(rd: u8, imm8: u8) -> u32 {
    enc_imm8(0x19, rd, imm8)
}
pub const fn v2_jmp(target: u8) -> u32 {
    enc_branch(0x1A, target)
}
pub const fn v2_jz(target: u8) -> u32 {
    enc_branch(0x1B, target)
}
pub const fn v2_jnz(target: u8) -> u32 {
    enc_branch(0x1C, target)
}
pub const fn v2_jc(target: u8) -> u32 {
    enc_branch(0x1D, target)
}
pub const fn v2_jnc(target: u8) -> u32 {
    enc_branch(0x1E, target)
}
pub const fn v2_call(target: u8) -> u32 {
    enc_branch(0x1F, target)
}

/// Sprint 363 (Gate B): register-indirect jump — `PC = regs[rd]`. The target
/// register index goes in the `rd` field; `EXT_REG_INDIRECT` selects the mode.
/// Reaches any instruction address (the far jump beyond the 8-bit imm limit).
pub const fn v2_jmpr(rd: u8) -> u32 {
    let mut ext = EXT_REG_INDIRECT;
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    v2_with_ext(enc(0x1A, rd & 0x07, 0, 0), ext)
}

/// Sprint 363 (Gate B): register-indirect call — `LR = PC+1; PC = regs[rd]`.
pub const fn v2_callr(rd: u8) -> u32 {
    let mut ext = EXT_REG_INDIRECT;
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    v2_with_ext(enc(0x1F, rd & 0x07, 0, 0), ext)
}

/// Sprint 371 (Gate B.3.1): wide-immediate jump — `PC = target16` for targets
/// 0..65,535 in a single instruction, with no register load. Reuses the
/// `EXT_WIDE_IMM` machinery: the low byte rides the branch immediate field, the
/// high byte the extension word [15:8]. The static far-jump companion to B.3's
/// physical wide PC — `JMPR` needs the target in a register first; `JMP.W` does not.
pub const fn v2_jmp_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1A, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}

/// Sprint 371 (Gate B.3.1): wide-immediate call — `LR = PC+1; PC = target16`.
/// Far-call companion to `JMP.W`; LR is captured physically as for any CALL.
pub const fn v2_call_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1F, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}

/// Sprint 374 (Gate D.3): wide-immediate conditional branches — the `.W` companions
/// to `JZ`/`JNZ`/`JC`/`JNC`, reaching any instruction 0..65,535 when the branch is
/// taken. Same `EXT_WIDE_IMM` encoding as `JMP.W`; the condition (flag test) is
/// unchanged. These let a program above the 7-bit branch window keep its loops/`if`s.
pub const fn v2_jz_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1B, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}
pub const fn v2_jnz_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1C, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}
pub const fn v2_jc_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1D, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}
pub const fn v2_jnc_w(target16: u16) -> u32 {
    v2_with_ext(
        enc_branch(0x1E, (target16 & 0xFF) as u8),
        EXT_WIDE_IMM | (((target16 >> 8) & 0xFF) << 8),
    )
}

// =====================================================================
// V2 CALLING CONVENTION (Sprint 364, Gate C)
// =====================================================================
// The ABI atop the CALL/RET + PUSH/POP primitives. The single LR register
// means a callee that itself issues a CALL MUST save LR first (otherwise
// the inner CALL clobbers it and the outer RET goes to the wrong place).
//
//   Role                            | Registers
//   --------------------------------|----------------------------------
//   Arguments (caller → callee)     | R0, R1, R2, R3
//   Return value                    | R0
//   Caller-saved (volatile + LR)    | R0..R7, LR
//   Callee-saved (preserved)        | R8..R14
//   Stack pointer                   | R15 (alias: SP)
//   Stack discipline                | Empty-descending: ST [SP], R; DEC SP
//
// Nested-call prologue (callee that calls another function):
//     MFLR R0          ; or use the PUSH LR pseudo
//     PUSH R0          ; save return address
//     ; ...body, which may issue CALL/CALLR...
//     POP  R0
//     MTLR R0          ; or use the POP LR pseudo
//     RET
//
// Leaf functions (no CALL inside) can skip the save entirely and just RET.
// =====================================================================

/// Sprint 174: Encode a wide-immediate instruction with 16-bit constant.
/// Low byte in base imm8; high byte in extension bits [15:8]; EXT_WIDE_IMM flag at bit 2.
const fn wide_imm_ext(rd: u8, imm16: u16) -> u16 {
    let hi = (imm16 >> 8) & 0xFF;
    let mut ext = EXT_WIDE_IMM | (hi << 8);
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    ext
}
pub const fn v2_ldi_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_ldi(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}
pub const fn v2_addi_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_addi(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}
pub const fn v2_subi_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_subi(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}
pub const fn v2_andi_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_andi(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}
pub const fn v2_ori_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_ori(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}
pub const fn v2_xori_w(rd: u8, imm16: u16) -> u32 {
    v2_with_ext(
        v2_xori(rd & 0x07, (imm16 & 0xFF) as u8),
        wide_imm_ext(rd, imm16),
    )
}

/// Sprint 175: Compose extension word for base+offset memory addressing.
/// Offset in bits [15:8], EXT_OFFSET at bit 3, plus rd_hi/rs_hi bank bits.
const fn offset_ext(rd: u8, rs: u8, offset: u8) -> u16 {
    let mut ext = EXT_OFFSET | ((offset as u16) << 8);
    if rd >= 8 {
        ext |= EXT_RD_HI;
    }
    if rs >= 8 {
        ext |= EXT_RS_HI;
    }
    ext
}
/// Sprint 175: LD rd, [rs+offset]
pub const fn v2_ld_o(rd: u8, rs: u8, offset: u8) -> u32 {
    v2_with_ext(
        enc(0x16, rd & 0x07, rs & 0x07, 0),
        offset_ext(rd, rs, offset),
    )
}
/// Sprint 175: ST [rs+offset], rd  (rs = address base, rd = data source)
pub const fn v2_st_o(rs: u8, rd: u8, offset: u8) -> u32 {
    v2_with_ext(
        enc(0x17, rd & 0x07, rs & 0x07, 0),
        offset_ext(rd, rs, offset),
    )
}
/// Sprint 175: LDB rd, [rs+offset]
pub const fn v2_ldb_o(rd: u8, rs: u8, offset: u8) -> u32 {
    v2_with_ext(
        enc(0x18, rd & 0x07, rs & 0x07, 0),
        offset_ext(rd, rs, offset),
    )
}
/// Sprint 175: STB [rs+offset], rd  (rs = address base, rd = data source)
pub const fn v2_stb_o(rs: u8, rd: u8, offset: u8) -> u32 {
    v2_with_ext(
        enc(0x19, rd & 0x07, rs & 0x07, 0),
        offset_ext(rd, rs, offset),
    )
}

/// Sprint 132: Compose a 32-bit instruction from a 16-bit base and 16-bit extension.
pub const fn v2_with_ext(base: u32, ext: u16) -> u32 {
    base | ((ext as u32) << 16)
}

pub const EXT_RD_HI: u16 = 0x0001;
pub const EXT_RS_HI: u16 = 0x0002;
/// Sprint 174: 16-bit immediate mode. When set, imm16 = (ir_ext >> 8) << 8 | imm8.
pub const EXT_WIDE_IMM: u16 = 0x0004;
/// Sprint 175: Base+offset memory addressing. When set, ext[15:8] = offset.
/// Effective address = (regs[rs] + offset) & 0x3F.
pub const EXT_OFFSET: u16 = 0x0008;
/// Sprint 363 (Gate B): register-indirect branch. When set on JMP (0x1A) or
/// CALL (0x1F), the branch target is `regs[rd]` (full width, masked by the PC
/// mask) instead of the 8-bit `imm8`. This is the "far jump" that reaches any
/// instruction address regardless of the 8-bit immediate limit.
pub const EXT_REG_INDIRECT: u16 = 0x0010;

pub const fn effective_reg(lo: u8, hi: bool) -> u8 {
    (lo & 0x07) | ((hi as u8) << 3)
}

/// Sprint 135: Compose register-bank extension bits.
/// `rd_hi` and `rs_hi` select R8-R15 banks while preserving base opcode format.
pub const fn v2_with_reg_ext(base: u32, rd_hi: bool, rs_hi: bool) -> u32 {
    let mut ext = 0u16;
    if rd_hi {
        ext |= EXT_RD_HI;
    }
    if rs_hi {
        ext |= EXT_RS_HI;
    }
    v2_with_ext(base, ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_decode_call_ret() {
        let call = V2ControlSignals::decode(v2_call(7));
        assert_eq!(call.opcode, 0x1F);
        assert!(call.is_call);
        assert_eq!(call.jump_target, 7);

        let ret = V2ControlSignals::decode(v2_ret());
        assert_eq!(ret.opcode, 0x15);
        assert!(ret.is_ret);
    }

    #[test]
    fn test_v2_decode_addi() {
        let s = V2ControlSignals::decode(v2_addi(3, 200));
        assert_eq!(s.opcode, 0x10);
        assert_eq!(s.rd, 3);
        assert_eq!(s.imm8, 200);
        assert_eq!(s.alu_op, V2AluOp::Add);
        assert!(s.reg_write_en);
        assert!(s.use_immediate);
        assert!(s.update_flags_z);
        assert!(s.update_flags_c);
    }

    #[test]
    fn test_v2_decode_andi() {
        let s = V2ControlSignals::decode(v2_andi(1, 0xAA));
        assert_eq!(s.opcode, 0x12);
        assert_eq!(s.rd, 1);
        assert_eq!(s.imm8, 0xAA);
        assert_eq!(s.alu_op, V2AluOp::And);
        assert!(s.reg_write_en);
        assert!(s.use_immediate);
        assert!(s.update_flags_z);
    }

    #[test]
    fn test_v2_decode_direct_page_ops() {
        let ldb = V2ControlSignals::decode(v2_ldb(3, 7));
        assert_eq!(ldb.opcode, 0x18);
        assert_eq!(ldb.rd, 3);
        assert_eq!(ldb.imm8, 7);
        assert!(ldb.mem_read);
        assert!(ldb.reg_write_en);
        assert!(ldb.use_immediate);
        assert!(ldb.update_flags_z);

        let stb = V2ControlSignals::decode(v2_stb(2, 5));
        assert_eq!(stb.opcode, 0x19);
        assert_eq!(stb.rd, 2);
        assert_eq!(stb.imm8, 5);
        assert!(stb.mem_write);
        assert!(stb.use_immediate);
    }

    #[test]
    fn test_v2_with_reg_ext_bits() {
        let base = v2_mov(3, 1);
        let instr = v2_with_reg_ext(base, true, true);
        let ext = (instr >> 16) as u16;
        assert_eq!(ext & EXT_RD_HI, EXT_RD_HI);
        assert_eq!(ext & EXT_RS_HI, EXT_RS_HI);

        let instr2 = v2_with_reg_ext(base, false, true);
        let ext2 = (instr2 >> 16) as u16;
        assert_eq!(ext2 & EXT_RD_HI, 0x0000);
        assert_eq!(ext2 & EXT_RS_HI, EXT_RS_HI);
    }

    #[test]
    fn test_effective_reg_helper() {
        assert_eq!(effective_reg(0, false), 0);
        assert_eq!(effective_reg(7, false), 7);
        assert_eq!(effective_reg(0, true), 8);
        assert_eq!(effective_reg(7, true), 15);
    }
}
