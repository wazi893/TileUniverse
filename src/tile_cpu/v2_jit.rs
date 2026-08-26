//! V2 ISS JIT compiler via Cranelift (Sprint 182.2).
//!
//! Compiles a V2 program (up to 128 instruction words) into native x86-64 code
//! using Cranelift. The JIT function operates on a `V2JitState` struct that
//! mirrors `V2IssState` with a C-compatible layout.
//!
//! Feature-gated behind `cranelift_jit`.

use cranelift_codegen::ir::{
    AbiParam, Block, Function, InstBuilder, MemFlags, Signature, Value, types,
};
use cranelift_codegen::{isa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use target_lexicon::Triple;

use crate::tile_cpu::v2_components::{
    EXT_OFFSET, EXT_RD_HI, EXT_RS_HI, EXT_WIDE_IMM, effective_reg,
};

// ---------------------------------------------------------------------------
// V2JitState — C-compatible state for JIT function
// ---------------------------------------------------------------------------

/// C-compatible state struct passed to JIT-compiled functions.
///
/// Layout (offsets in bytes):
/// - regs[16]: offset 0, 128 bytes
/// - ram[128]: offset 128, 1024 bytes
/// - pc:       offset 1152
/// - lr:       offset 1156
/// - flag_z:   offset 1160
/// - flag_c:   offset 1161
/// - halted:   offset 1162
#[repr(C)]
#[derive(Debug, Clone)]
pub struct V2JitState {
    pub regs: [u64; 16],
    pub ram: [u64; 128],
    pub pc: u32,
    pub lr: u32,
    pub flag_z: u8,
    pub flag_c: u8,
    pub halted: u8,
    _pad: u8,
}

impl Default for V2JitState {
    fn default() -> Self {
        Self {
            regs: [0u64; 16],
            ram: [0u64; 128],
            pc: 0,
            lr: 0,
            flag_z: 0,
            flag_c: 0,
            halted: 0,
            _pad: 0,
        }
    }
}

impl V2JitState {
    /// Convert from V2IssState for parity testing.
    pub fn from_iss_state(s: &crate::tile_cpu::v2_iss::V2IssState) -> Self {
        let mut js = Self::default();
        js.regs = s.regs;
        js.ram = s.ram;
        js.pc = s.pc as u32;
        js.lr = s.lr as u32;
        js.flag_z = s.flag_z as u8;
        js.flag_c = s.flag_c as u8;
        js.halted = s.halted as u8;
        js
    }

    /// Merge JIT state back into an IssState.
    pub fn to_iss_state(&self) -> crate::tile_cpu::v2_iss::V2IssState {
        crate::tile_cpu::v2_iss::V2IssState {
            // Sprint 370 widened V2IssState.pc/lr to u32; the JIT still models
            // the 7-bit PC era, so the 0x7F mask is preserved.
            pc: self.pc & 0x7F,
            lr: self.lr & 0x7F,
            flag_z: self.flag_z != 0,
            flag_c: self.flag_c != 0,
            halted: self.halted != 0,
            regs: self.regs,
            ram: self.ram,
            main_mem: Vec::new(),
            mem_addr_mask: 0x7F,
            iss_math_a: 0,
            iss_math_b: 0,
            iss_math_status: 0,
            iss_math_result: 0,
        }
    }
}

// Offsets into V2JitState
const OFF_REGS: i32 = 0; // regs[0]
const OFF_RAM: i32 = 128; // ram[0]
const OFF_PC: i32 = 1152; // pc (u32)
const OFF_LR: i32 = 1156; // lr (u32)
const OFF_FLAG_Z: i32 = 1160; // flag_z (u8)
const OFF_FLAG_C: i32 = 1161; // flag_c (u8)
const OFF_HALTED: i32 = 1162; // halted (u8)

// ---------------------------------------------------------------------------
// JIT program wrapper
// ---------------------------------------------------------------------------

/// Thread-safe wrapper for JITModule (raw pointers inside).
struct SendJITModule(JITModule);
unsafe impl Send for SendJITModule {}
unsafe impl Sync for SendJITModule {}

/// JIT function signature: (state_ptr, max_cycles) -> cycles_executed.
pub type V2JitFn = unsafe extern "C" fn(*mut V2JitState, u32) -> u32;

/// A compiled V2 program ready to execute.
pub struct V2JitProgram {
    _module: SendJITModule,
    pub func_ptr: V2JitFn,
}

// V2JitProgram contains raw pointers (from JITModule) but the compiled code
// is immutable after finalization. Safe to send across threads.
unsafe impl Send for V2JitProgram {}
unsafe impl Sync for V2JitProgram {}

/// Global cache: program hash -> compiled program.
static JIT_CACHE: once_cell::sync::Lazy<Mutex<HashMap<u64, Arc<V2JitProgram>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

fn hash_program(program: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64; // FNV-1a offset basis
    for &w in program {
        h ^= w as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ---------------------------------------------------------------------------
// MMIO detection
// ---------------------------------------------------------------------------

/// Sprint 390 (Phase D): check if a program uses MFLR/MTLR (opcode 0x02, sub_fn 5/6).
/// This Cranelift JIT models a charter compute datapath and has NO LR-store codegen —
/// the MOV catch-all in the 0x02 arm collapses sub_fn 5/6 into `MOV rd, rs`, which would
/// silently corrupt (write rd, ignore LR). Rather than emit LR codegen into a JIT that
/// does not model the LR register, decline such programs so they run on the authoritative
/// ISS. No compute benchmark uses MFLR/MTLR (the compiler returns via JMPR), so this is
/// inert for the perf corpus.
pub fn program_uses_lr_ops(program: &[u32]) -> bool {
    for &word in program {
        let opcode = ((word >> 11) & 0x1F) as u8;
        let imm8 = (word & 0xFF) as u8;
        let sub_fn = imm8 & 0x1F;
        if opcode == 0x02 && (sub_fn == 5 || sub_fn == 6) {
            return true;
        }
    }
    false
}

/// Check if a program uses MMIO addresses (>= 41).
/// If so, JIT cannot handle it — fall back to ISS.
pub fn program_uses_mmio(program: &[u32]) -> bool {
    for &word in program {
        let opcode = ((word >> 11) & 0x1F) as u8;
        let imm8 = (word & 0xFF) as u8;
        let ir_ext = ((word >> 16) & 0xFFFF) as u16;
        let has_offset = (ir_ext & EXT_OFFSET) != 0;

        match opcode {
            // LDB rd, [imm8] — direct addressing to MMIO
            0x18 if !has_offset && imm8 >= 41 => return true,
            // STB [imm8], rd — direct addressing to MMIO
            0x19 if !has_offset && imm8 >= 41 => return true,
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// Compile a V2 program to native code via Cranelift.
pub fn compile_v2_program(program: &[u32]) -> Result<Arc<V2JitProgram>, String> {
    // Sprint 189: JIT supports up to 128-word ROM.
    if program.len() > 128 {
        return Err("JIT does not support programs > 128 instructions".into());
    }
    // Sprint 390 (Phase D): the JIT has no LR-store codegen; decline MFLR/MTLR programs
    // so they run on the authoritative ISS instead of the MOV-collapse fast path.
    if program_uses_lr_ops(program) {
        return Err("JIT does not model LR (MFLR/MTLR) — fall back to ISS".into());
    }
    let key = hash_program(program);

    // Check cache
    {
        let cache = JIT_CACHE.lock().unwrap();
        if let Some(prog) = cache.get(&key) {
            return Ok(prog.clone());
        }
    }

    // Compile
    let compiled = compile_v2_program_inner(program)?;
    let arc = Arc::new(compiled);

    // Store in cache
    {
        let mut cache = JIT_CACHE.lock().unwrap();
        cache.insert(key, arc.clone());
    }

    Ok(arc)
}

fn compile_v2_program_inner(program: &[u32]) -> Result<V2JitProgram, String> {
    let triple = Triple::host();
    let bld = settings::builder();
    let flags = settings::Flags::new(bld);
    let target_isa = isa::lookup(triple.clone())
        .map_err(|e| format!("ISA lookup: {e}"))?
        .finish(flags)
        .map_err(|e| format!("ISA finish: {e}"))?;

    let jitb = JITBuilder::with_isa(target_isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jitb);

    // Signature: (state_ptr: i64, max_cycles: i32) -> i32
    let mut ctx = module.make_context();
    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(&triple));
    sig.params.push(AbiParam::new(types::I64)); // state_ptr
    sig.params.push(AbiParam::new(types::I32)); // max_cycles
    sig.returns.push(AbiParam::new(types::I32)); // cycles_executed

    let mut func = Function::new();
    func.signature = sig.clone();

    let mut fbctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);

    // Variables for cycle counter and PC
    let var_cycles = Variable::from_u32(0);
    let var_pc = Variable::from_u32(1);
    let var_flag_z = Variable::from_u32(2);
    let var_flag_c = Variable::from_u32(3);
    fb.declare_var(var_cycles, types::I32);
    fb.declare_var(var_pc, types::I32);
    fb.declare_var(var_flag_z, types::I8);
    fb.declare_var(var_flag_c, types::I8);

    // Create blocks: entry, cycle_header, per-PC blocks, exit, halt
    let entry_block = fb.create_block();
    let cycle_header = fb.create_block();
    let exit_block = fb.create_block();
    let halt_block = fb.create_block();

    // One block per ROM slot
    let rom_size = program.len().min(128);
    let pc_blocks: Vec<Block> = (0..rom_size).map(|_| fb.create_block()).collect();
    // For out-of-range PCs, jump to halt
    let oob_block = fb.create_block();

    // ---- Entry block ----
    fb.append_block_params_for_function_params(entry_block);
    fb.switch_to_block(entry_block);
    fb.seal_block(entry_block);

    let state_ptr = fb.block_params(entry_block)[0];
    let max_cycles = fb.block_params(entry_block)[1];

    // Initialize variables
    let zero_i32 = fb.ins().iconst(types::I32, 0);
    fb.def_var(var_cycles, zero_i32);

    // Load initial PC from state
    let mflags = MemFlags::new();
    let init_pc = fb.ins().load(types::I32, mflags, state_ptr, OFF_PC);
    let mask_63_i32 = fb.ins().iconst(types::I32, 63);
    let init_pc_masked = fb.ins().band(init_pc, mask_63_i32);
    fb.def_var(var_pc, init_pc_masked);

    // Load initial flags from state
    let init_z = fb.ins().load(types::I8, mflags, state_ptr, OFF_FLAG_Z);
    let init_c = fb.ins().load(types::I8, mflags, state_ptr, OFF_FLAG_C);
    fb.def_var(var_flag_z, init_z);
    fb.def_var(var_flag_c, init_c);

    // Check if already halted
    let init_halted = fb.ins().load(types::I8, mflags, state_ptr, OFF_HALTED);
    let zero_i8 = fb.ins().iconst(types::I8, 0);
    let already_halted = fb.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::NotEqual,
        init_halted,
        zero_i8,
    );
    fb.ins()
        .brif(already_halted, exit_block, &[], cycle_header, &[]);

    // ---- Cycle header: check max_cycles, dispatch on PC ----
    fb.switch_to_block(cycle_header);

    let cur_cycles = fb.use_var(var_cycles);
    let over_limit = fb.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual,
        cur_cycles,
        max_cycles,
    );
    fb.ins().brif(over_limit, exit_block, &[], oob_block, &[]);

    // Dispatch block: cascading brif chain on PC value
    fb.switch_to_block(oob_block);
    let cur_pc = fb.use_var(var_pc);

    for i in 0..rom_size {
        let pc_val = fb.ins().iconst(types::I32, i as i64);
        let matches = fb.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::Equal,
            cur_pc,
            pc_val,
        );
        let next_dispatch = fb.create_block();
        fb.ins()
            .brif(matches, pc_blocks[i], &[], next_dispatch, &[]);
        fb.switch_to_block(next_dispatch);
        fb.seal_block(next_dispatch);
    }
    // Out-of-range PC: halt
    fb.ins().jump(halt_block, &[]);

    // ---- Per-PC instruction blocks ----
    for pc_idx in 0..rom_size {
        fb.switch_to_block(pc_blocks[pc_idx]);

        let word = program[pc_idx];
        emit_instruction(
            &mut fb,
            word,
            pc_idx as u32,
            state_ptr,
            var_cycles,
            var_pc,
            var_flag_z,
            var_flag_c,
            cycle_header,
            halt_block,
        );
    }

    // ---- Halt block: set halted=1, store state, jump to exit ----
    fb.switch_to_block(halt_block);
    let one_i8 = fb.ins().iconst(types::I8, 1);
    fb.ins().store(mflags, one_i8, state_ptr, OFF_HALTED);
    fb.ins().jump(exit_block, &[]);

    // ---- Exit block: store final PC and flags, return cycles ----
    fb.switch_to_block(exit_block);
    let final_pc = fb.use_var(var_pc);
    let final_z = fb.use_var(var_flag_z);
    let final_c = fb.use_var(var_flag_c);
    let final_cycles = fb.use_var(var_cycles);
    fb.ins().store(mflags, final_pc, state_ptr, OFF_PC);
    fb.ins().store(mflags, final_z, state_ptr, OFF_FLAG_Z);
    fb.ins().store(mflags, final_c, state_ptr, OFF_FLAG_C);
    fb.ins().return_(&[final_cycles]);

    // Seal all blocks
    fb.seal_block(cycle_header);
    fb.seal_block(oob_block);
    for blk in &pc_blocks {
        fb.seal_block(*blk);
    }
    fb.seal_block(halt_block);
    fb.seal_block(exit_block);

    fb.finalize();

    // Compile
    ctx.func = func;
    static FN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let suffix = FN_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let fname = format!("v2_jit_{suffix}");

    let func_id = module
        .declare_function(&fname, Linkage::Local, &sig)
        .map_err(|e| format!("declare: {e}"))?;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define: {e}"))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {e}"))?;

    let code = module.get_finalized_function(func_id);
    let fnptr = code as *const ();
    let jit_fn: V2JitFn = unsafe { std::mem::transmute(fnptr) };

    Ok(V2JitProgram {
        _module: SendJITModule(module),
        func_ptr: jit_fn,
    })
}

// ---------------------------------------------------------------------------
// Per-instruction IR emission
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_instruction(
    fb: &mut FunctionBuilder,
    word: u32,
    pc_idx: u32,
    state_ptr: Value,
    var_cycles: Variable,
    var_pc: Variable,
    var_flag_z: Variable,
    var_flag_c: Variable,
    cycle_header: Block,
    halt_block: Block,
) {
    let mflags = MemFlags::new();

    // Decode
    let opcode = ((word >> 11) & 0x1F) as u8;
    let rd_lo = ((word >> 8) & 0x07) as u8;
    let rs_lo = ((word >> 5) & 0x07) as u8;
    let imm8 = (word & 0xFF) as u8;
    let ir_ext = ((word >> 16) & 0xFFFF) as u16;

    let wide_imm = (ir_ext & EXT_WIDE_IMM) != 0;
    let imm_val: u64 = if wide_imm {
        let hi = ((ir_ext >> 8) & 0xFF) as u64;
        (hi << 8) | (imm8 as u64)
    } else {
        imm8 as u64
    };
    let rd_hi = (ir_ext & EXT_RD_HI) != 0;
    let rs_hi = (ir_ext & EXT_RS_HI) != 0;
    let rd = effective_reg(rd_lo, rd_hi) as usize;
    let rs = effective_reg(rs_lo, rs_hi) as usize;

    let has_offset = (ir_ext & EXT_OFFSET) != 0;
    let offset_val = if has_offset {
        ((ir_ext >> 8) & 0xFF) as u64
    } else {
        0
    };

    let next_pc = (pc_idx.wrapping_add(1)) & 0x7F;

    // Helper: load reg[idx]
    let load_reg = |fb: &mut FunctionBuilder, idx: usize| -> Value {
        let off = OFF_REGS + (idx as i32) * 8;
        fb.ins().load(types::I64, mflags, state_ptr, off)
    };

    // Helper: store reg[idx]
    let store_reg = |fb: &mut FunctionBuilder, idx: usize, val: Value| {
        let off = OFF_REGS + (idx as i32) * 8;
        fb.ins().store(mflags, val, state_ptr, off);
    };

    // Helper: load ram[addr_val & 0x7F]
    let load_ram = |fb: &mut FunctionBuilder, addr_val: Value| -> Value {
        let mask = fb.ins().iconst(types::I64, 0x7F);
        let masked = fb.ins().band(addr_val, mask);
        let offset = fb.ins().ishl_imm(masked, 3); // * 8 bytes
        let ram_base = fb.ins().iconst(types::I64, OFF_RAM as i64);
        let ptr_off = fb.ins().iadd(ram_base, offset);
        let addr = fb.ins().iadd(state_ptr, ptr_off);
        fb.ins().load(types::I64, mflags, addr, 0)
    };

    // Helper: store ram[addr_val & 0x7F]
    let store_ram = |fb: &mut FunctionBuilder, addr_val: Value, data: Value| {
        let mask = fb.ins().iconst(types::I64, 0x7F);
        let masked = fb.ins().band(addr_val, mask);
        let offset = fb.ins().ishl_imm(masked, 3); // * 8 bytes
        let ram_base = fb.ins().iconst(types::I64, OFF_RAM as i64);
        let ptr_off = fb.ins().iadd(ram_base, offset);
        let addr = fb.ins().iadd(state_ptr, ptr_off);
        fb.ins().store(mflags, data, addr, 0);
    };

    // Helper: increment cycle counter
    let inc_cycles = |fb: &mut FunctionBuilder| {
        let c = fb.use_var(var_cycles);
        let one = fb.ins().iconst(types::I32, 1);
        let c1 = fb.ins().iadd(c, one);
        fb.def_var(var_cycles, c1);
    };

    // Helper: set next PC and jump to cycle header
    let advance = |fb: &mut FunctionBuilder, next: u32| {
        let pc_val = fb.ins().iconst(types::I32, next as i64);
        fb.def_var(var_pc, pc_val);
        fb.ins().jump(cycle_header, &[]);
    };

    // Helper: update Z flag from result (icmp returns I8: 0 or 1)
    let update_z = |fb: &mut FunctionBuilder, result: Value| {
        let zero = fb.ins().iconst(types::I64, 0);
        let is_zero = fb
            .ins()
            .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, result, zero);
        fb.def_var(var_flag_z, is_zero);
    };

    // Helper: update C flag (carry = a > result for ADD, result > a for SUB)
    let _set_c = |fb: &mut FunctionBuilder, val: bool| {
        let c = fb.ins().iconst(types::I8, val as i64);
        fb.def_var(var_flag_c, c);
    };

    // Helper: C = (a > result) unsigned — ADD carry (icmp returns I8)
    let update_c_add = |fb: &mut FunctionBuilder, a: Value, result: Value| {
        let carry = fb.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThan,
            a,
            result,
        );
        fb.def_var(var_flag_c, carry);
    };

    // Helper: C = (result > a) unsigned — SUB borrow (icmp returns I8)
    let update_c_sub = |fb: &mut FunctionBuilder, a: Value, result: Value| {
        let borrow = fb.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThan,
            result,
            a,
        );
        fb.def_var(var_flag_c, borrow);
    };

    inc_cycles(fb);

    match opcode {
        0x00 => {
            // NOP
            advance(fb, next_pc);
        }
        0x01 => {
            // HALT
            fb.ins().jump(halt_block, &[]);
        }
        0x02 => {
            // MOV/MOVZ/MOVNZ/MOVC/MOVNC
            let sub_fn = imm8 & 0x1F;
            let b = load_reg(fb, rs);

            match sub_fn {
                0 | 5..=31 => {
                    // Unconditional MOV. Note: sub_fn 5 (MFLR) and 6 (MTLR) never reach
                    // here — `compile_v2_program` declines LR-op programs to the ISS
                    // (Sprint 390), since this JIT has no LR-register model. 7..=31 are
                    // reserved MOV encodings.
                    store_reg(fb, rd, b);
                    advance(fb, next_pc);
                }
                1 => {
                    // MOVZ: fire if flag_z != 0
                    let z = fb.use_var(var_flag_z);
                    let zero = fb.ins().iconst(types::I8, 0);
                    let cond =
                        fb.ins()
                            .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, z, zero);
                    let then_blk = fb.create_block();
                    let merge_blk = fb.create_block();
                    fb.ins().brif(cond, then_blk, &[], merge_blk, &[]);

                    fb.switch_to_block(then_blk);
                    fb.seal_block(then_blk);
                    store_reg(fb, rd, b);
                    fb.ins().jump(merge_blk, &[]);

                    fb.switch_to_block(merge_blk);
                    fb.seal_block(merge_blk);
                    advance(fb, next_pc);
                }
                2 => {
                    // MOVNZ: fire if flag_z == 0
                    let z = fb.use_var(var_flag_z);
                    let zero = fb.ins().iconst(types::I8, 0);
                    let cond =
                        fb.ins()
                            .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, z, zero);
                    let then_blk = fb.create_block();
                    let merge_blk = fb.create_block();
                    fb.ins().brif(cond, then_blk, &[], merge_blk, &[]);

                    fb.switch_to_block(then_blk);
                    fb.seal_block(then_blk);
                    store_reg(fb, rd, b);
                    fb.ins().jump(merge_blk, &[]);

                    fb.switch_to_block(merge_blk);
                    fb.seal_block(merge_blk);
                    advance(fb, next_pc);
                }
                3 => {
                    // MOVC: fire if flag_c != 0
                    let c = fb.use_var(var_flag_c);
                    let zero = fb.ins().iconst(types::I8, 0);
                    let cond =
                        fb.ins()
                            .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, c, zero);
                    let then_blk = fb.create_block();
                    let merge_blk = fb.create_block();
                    fb.ins().brif(cond, then_blk, &[], merge_blk, &[]);

                    fb.switch_to_block(then_blk);
                    fb.seal_block(then_blk);
                    store_reg(fb, rd, b);
                    fb.ins().jump(merge_blk, &[]);

                    fb.switch_to_block(merge_blk);
                    fb.seal_block(merge_blk);
                    advance(fb, next_pc);
                }
                4 => {
                    // MOVNC: fire if flag_c == 0
                    let c = fb.use_var(var_flag_c);
                    let zero = fb.ins().iconst(types::I8, 0);
                    let cond =
                        fb.ins()
                            .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, c, zero);
                    let then_blk = fb.create_block();
                    let merge_blk = fb.create_block();
                    fb.ins().brif(cond, then_blk, &[], merge_blk, &[]);

                    fb.switch_to_block(then_blk);
                    fb.seal_block(then_blk);
                    store_reg(fb, rd, b);
                    fb.ins().jump(merge_blk, &[]);

                    fb.switch_to_block(merge_blk);
                    fb.seal_block(merge_blk);
                    advance(fb, next_pc);
                }
                _ => unreachable!(),
            }
        }
        0x03 => {
            // LDI rd, imm / LDI.W rd, imm16
            let val = fb.ins().iconst(types::I64, imm_val as i64);
            store_reg(fb, rd, val);
            update_z(fb, val);
            advance(fb, next_pc);
        }
        0x04 => {
            // ADD rd, rs / MUL rd, rs (Sprint 188: imm5=1 → multiply)
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let sub_fn = imm8 & 0x1F;
            if sub_fn == 1 {
                // MUL
                let result = fb.ins().imul(a, b);
                store_reg(fb, rd, result);
                update_z(fb, result);
                // C unchanged for MUL
            } else {
                // ADD
                let result = fb.ins().iadd(a, b);
                store_reg(fb, rd, result);
                update_z(fb, result);
                update_c_add(fb, a, result);
            }
            advance(fb, next_pc);
        }
        0x05 => {
            // SUB rd, rs
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let result = fb.ins().isub(a, b);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_sub(fb, a, result);
            advance(fb, next_pc);
        }
        0x06 => {
            // AND rd, rs
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let result = fb.ins().band(a, b);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x07 => {
            // OR rd, rs
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let result = fb.ins().bor(a, b);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x08 => {
            // XOR rd, rs
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let result = fb.ins().bxor(a, b);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x09 => {
            // NOT/CLZ/CTZ/POPCNT
            let sub_fn = imm8 & 0x1F;
            let a = load_reg(fb, rd);
            let result = match sub_fn {
                1 => fb.ins().clz(a),
                2 => fb.ins().ctz(a),
                3 => fb.ins().popcnt(a),
                _ => fb.ins().bnot(a), // NOT (default)
            };
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x0A => {
            // NEG rd: 0 - rd (using b from same bank)
            // In ISS, NEG does: result = 0 - b (b = rd via Tree B)
            let a = load_reg(fb, rd);
            let zero = fb.ins().iconst(types::I64, 0);
            let b = load_reg(fb, rd); // rd is read via Tree B in hardware
            let result = fb.ins().isub(zero, b);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_sub(fb, a, result);
            advance(fb, next_pc);
        }
        0x0B => {
            // INC rd, imm8
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm8 as i64);
            let result = fb.ins().iadd(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_add(fb, a, result);
            advance(fb, next_pc);
        }
        0x0C => {
            // DEC rd, imm8
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm8 as i64);
            let result = fb.ins().isub(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_sub(fb, a, result);
            advance(fb, next_pc);
        }
        0x0D => {
            // SHL rd, imm
            let a = load_reg(fb, rd);
            let shift = (imm8 as u32) & 63;
            let result = fb.ins().ishl_imm(a, shift as i64);
            store_reg(fb, rd, result);
            update_z(fb, result);
            // Carry: bit at position (8 - shift) before shift
            let bit_pos = (8u64.wrapping_sub(imm8 as u64)) & 63;
            let shifted = fb.ins().ushr_imm(a, bit_pos as i64);
            let one_i64 = fb.ins().iconst(types::I64, 1);
            let bit = fb.ins().band(shifted, one_i64);
            let zero_i64 = fb.ins().iconst(types::I64, 0);
            let c_cond = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                bit,
                zero_i64,
            );
            fb.def_var(var_flag_c, c_cond);
            advance(fb, next_pc);
        }
        0x0E => {
            // SHR rd, imm / SRA rd, imm (Sprint 188: bit 3 → arithmetic)
            let a = load_reg(fb, rd);
            let is_arithmetic = (imm8 & 0x08) != 0;
            let shift_amount = (imm8 & 0x07) as u32;
            let result = if is_arithmetic {
                fb.ins().sshr_imm(a, shift_amount as i64)
            } else {
                fb.ins().ushr_imm(a, shift_amount as i64)
            };
            store_reg(fb, rd, result);
            update_z(fb, result);
            // Carry: bit at position (shift_amount - 1) before shift
            let bit_pos = (shift_amount as u64).wrapping_sub(1) & 63;
            let shifted = fb.ins().ushr_imm(a, bit_pos as i64);
            let one_i64 = fb.ins().iconst(types::I64, 1);
            let bit = fb.ins().band(shifted, one_i64);
            let zero_i64 = fb.ins().iconst(types::I64, 0);
            let c_cond = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                bit,
                zero_i64,
            );
            fb.def_var(var_flag_c, c_cond);
            advance(fb, next_pc);
        }
        0x0F => {
            // CMP rd, rs (SUB without writeback)
            let a = load_reg(fb, rd);
            let b = load_reg(fb, rs);
            let result = fb.ins().isub(a, b);
            update_z(fb, result);
            update_c_sub(fb, a, result);
            advance(fb, next_pc);
        }
        0x10 => {
            // ADDI rd, imm / ADDI.W rd, imm16
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm_val as i64);
            let result = fb.ins().iadd(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_add(fb, a, result);
            advance(fb, next_pc);
        }
        0x11 => {
            // SUBI rd, imm / SUBI.W rd, imm16
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm_val as i64);
            let result = fb.ins().isub(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            update_c_sub(fb, a, result);
            advance(fb, next_pc);
        }
        0x12 => {
            // ANDI rd, imm / ANDI.W
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm_val as i64);
            let result = fb.ins().band(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x13 => {
            // ORI rd, imm / ORI.W
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm_val as i64);
            let result = fb.ins().bor(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x14 => {
            // XORI rd, imm / XORI.W
            let a = load_reg(fb, rd);
            let rhs = fb.ins().iconst(types::I64, imm_val as i64);
            let result = fb.ins().bxor(a, rhs);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x15 => {
            // RET: PC = LR
            let lr = fb.ins().load(types::I32, mflags, state_ptr, OFF_LR);
            let mask = fb.ins().iconst(types::I32, 0x7F);
            let lr_masked = fb.ins().band(lr, mask);
            fb.def_var(var_pc, lr_masked);
            fb.ins().jump(cycle_header, &[]);
        }
        0x16 => {
            // LD rd, [rs] / LD rd, [rs+offset]
            let b = load_reg(fb, rs);
            let addr = if has_offset {
                let off = fb.ins().iconst(types::I64, offset_val as i64);
                fb.ins().iadd(b, off)
            } else {
                b
            };
            let result = load_ram(fb, addr);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x17 => {
            // ST [rs], rd / ST [rs+offset], rd
            let b = load_reg(fb, rs);
            let addr = if has_offset {
                let off = fb.ins().iconst(types::I64, offset_val as i64);
                fb.ins().iadd(b, off)
            } else {
                b
            };
            let a = load_reg(fb, rd);
            store_ram(fb, addr, a);
            advance(fb, next_pc);
        }
        0x18 => {
            // LDB rd, [imm8] / LDB rd, [rs+offset]
            let addr = if has_offset {
                let base = load_reg(fb, rs);
                let off = fb.ins().iconst(types::I64, offset_val as i64);
                fb.ins().iadd(base, off)
            } else {
                fb.ins().iconst(types::I64, imm8 as i64)
            };
            let result = load_ram(fb, addr);
            store_reg(fb, rd, result);
            update_z(fb, result);
            advance(fb, next_pc);
        }
        0x19 => {
            // STB [imm8], rd / STB [rs+offset], rd
            let addr = if has_offset {
                let base = load_reg(fb, rs);
                let off = fb.ins().iconst(types::I64, offset_val as i64);
                fb.ins().iadd(base, off)
            } else {
                fb.ins().iconst(types::I64, imm8 as i64)
            };
            let a = load_reg(fb, rd);
            store_ram(fb, addr, a);
            advance(fb, next_pc);
        }
        0x1A => {
            // JMP imm8
            let target = (imm8 & 0x7F) as u32;
            let pc_val = fb.ins().iconst(types::I32, target as i64);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);
        }
        0x1B => {
            // JZ imm8
            let z = fb.use_var(var_flag_z);
            let zero = fb.ins().iconst(types::I8, 0);
            let cond = fb
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, z, zero);
            let taken_blk = fb.create_block();
            let fallthru_blk = fb.create_block();
            fb.ins().brif(cond, taken_blk, &[], fallthru_blk, &[]);

            fb.switch_to_block(taken_blk);
            fb.seal_block(taken_blk);
            let target = (imm8 & 0x7F) as i64;
            let pc_val = fb.ins().iconst(types::I32, target);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);

            fb.switch_to_block(fallthru_blk);
            fb.seal_block(fallthru_blk);
            advance(fb, next_pc);
        }
        0x1C => {
            // JNZ imm8
            let z = fb.use_var(var_flag_z);
            let zero = fb.ins().iconst(types::I8, 0);
            let cond = fb
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, z, zero);
            let taken_blk = fb.create_block();
            let fallthru_blk = fb.create_block();
            fb.ins().brif(cond, taken_blk, &[], fallthru_blk, &[]);

            fb.switch_to_block(taken_blk);
            fb.seal_block(taken_blk);
            let target = (imm8 & 0x7F) as i64;
            let pc_val = fb.ins().iconst(types::I32, target);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);

            fb.switch_to_block(fallthru_blk);
            fb.seal_block(fallthru_blk);
            advance(fb, next_pc);
        }
        0x1D => {
            // JC imm8
            let c = fb.use_var(var_flag_c);
            let zero = fb.ins().iconst(types::I8, 0);
            let cond = fb
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, c, zero);
            let taken_blk = fb.create_block();
            let fallthru_blk = fb.create_block();
            fb.ins().brif(cond, taken_blk, &[], fallthru_blk, &[]);

            fb.switch_to_block(taken_blk);
            fb.seal_block(taken_blk);
            let target = (imm8 & 0x7F) as i64;
            let pc_val = fb.ins().iconst(types::I32, target);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);

            fb.switch_to_block(fallthru_blk);
            fb.seal_block(fallthru_blk);
            advance(fb, next_pc);
        }
        0x1E => {
            // JNC imm8
            let c = fb.use_var(var_flag_c);
            let zero = fb.ins().iconst(types::I8, 0);
            let cond = fb
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, c, zero);
            let taken_blk = fb.create_block();
            let fallthru_blk = fb.create_block();
            fb.ins().brif(cond, taken_blk, &[], fallthru_blk, &[]);

            fb.switch_to_block(taken_blk);
            fb.seal_block(taken_blk);
            let target = (imm8 & 0x7F) as i64;
            let pc_val = fb.ins().iconst(types::I32, target);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);

            fb.switch_to_block(fallthru_blk);
            fb.seal_block(fallthru_blk);
            advance(fb, next_pc);
        }
        0x1F => {
            // CALL imm8
            let lr_val = fb.ins().iconst(types::I32, next_pc as i64);
            fb.ins().store(mflags, lr_val, state_ptr, OFF_LR);
            let target = (imm8 & 0x7F) as i64;
            let pc_val = fb.ins().iconst(types::I32, target);
            fb.def_var(var_pc, pc_val);
            fb.ins().jump(cycle_header, &[]);
        }
        _ => {
            // Unknown opcode: treat as NOP
            advance(fb, next_pc);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_assembler::assemble_v2;

    #[test]
    fn test_v2_jit_basic_arithmetic() {
        let prog = assemble_v2("LDI R0, 5\n ADDI R0, 3\n HALT").unwrap();
        let jit = compile_v2_program(&prog).expect("JIT compile failed");
        let mut state = V2JitState::default();
        let cycles = unsafe { (jit.func_ptr)(&mut state, 100) };
        assert_eq!(state.regs[0], 8, "5+3=8");
        assert_eq!(state.halted, 1);
        assert!(cycles > 0);
    }

    #[test]
    fn test_v2_jit_memory_ops() {
        let prog = assemble_v2(
            r#"
            LDI R0, 42
            LDI R1, 5
            ST [R1], R0
            LD R2, [R1]
            HALT
        "#,
        )
        .unwrap();
        let jit = compile_v2_program(&prog).expect("JIT compile failed");
        let mut state = V2JitState::default();
        unsafe { (jit.func_ptr)(&mut state, 100) };
        assert_eq!(state.ram[5], 42, "RAM[5]=42");
        assert_eq!(state.regs[2], 42, "R2=42 from LD");
        assert_eq!(state.halted, 1);
    }

    #[test]
    fn test_v2_jit_sprint174_177() {
        let prog = assemble_v2(
            r#"
            LDI.W R0, 500
            CLZ R0
            POPCNT R0
            LDI R1, 10
            CMP R0, R1
            MOVNC R0, R1
            ADDI.W R0, 1000
            HALT
        "#,
        )
        .unwrap();
        let jit = compile_v2_program(&prog).expect("JIT compile failed");
        let mut state = V2JitState::default();
        unsafe { (jit.func_ptr)(&mut state, 100) };
        // LDI.W 500 → CLZ(500)=55 → POPCNT(55)=5 → CMP(5,10): C=1 →
        // MOVNC skip → ADDI.W(5,1000)=1005
        // Actually: 500 = 0x1F4, bit 8 is highest → CLZ = 64-9 = 55
        // POPCNT(55) = POPCNT(0b110111) = 5
        // CMP 5, 10: 5 < 10 → C=1. MOVNC fires if C=0 → skip. R0 stays 5.
        // ADDI.W 5, 1000 = 1005
        assert_eq!(state.regs[0], 1005, "expected 1005");
        assert_eq!(state.halted, 1);
    }

    #[test]
    fn test_v2_jit_parity_with_iss() {
        use crate::tile_cpu::v2_iss::V2Iss;
        let prog = assemble_v2(
            r#"
            LDI R0, 30
            LDI R1, 50
            CMP R0, R1
            MOVC R0, R1
            LDI R2, 0
            ST [R2], R0
            LDI.W R3, 1000
            ADDI R3, 55
            INC R2
            ST [R2], R3
            HALT
        "#,
        )
        .unwrap();

        // Run ISS
        let mut iss = V2Iss::from_program(&prog);
        for _ in 0..200 {
            if iss.step().is_none() {
                break;
            }
        }

        // Run JIT
        let jit = compile_v2_program(&prog).expect("JIT compile failed");
        let mut jit_state = V2JitState::default();
        unsafe { (jit.func_ptr)(&mut jit_state, 200) };

        // Compare
        for i in 0..16 {
            assert_eq!(
                jit_state.regs[i],
                iss.state().regs[i],
                "R{i}: JIT={} ISS={}",
                jit_state.regs[i],
                iss.state().regs[i]
            );
        }
        for i in 0..64 {
            assert_eq!(
                jit_state.ram[i],
                iss.state().ram[i],
                "RAM[{i}]: JIT={} ISS={}",
                jit_state.ram[i],
                iss.state().ram[i]
            );
        }
        assert_eq!(jit_state.pc, iss.state().pc as u32, "PC mismatch");
        assert_eq!(
            jit_state.flag_z,
            iss.state().flag_z as u8,
            "Z flag mismatch"
        );
        assert_eq!(
            jit_state.flag_c,
            iss.state().flag_c as u8,
            "C flag mismatch"
        );
        assert_eq!(
            jit_state.halted,
            iss.state().halted as u8,
            "halted mismatch"
        );
    }
}
