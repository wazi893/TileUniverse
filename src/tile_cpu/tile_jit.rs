//! Sprint 273: Tile evaluation JIT compiler via Cranelift.
//!
//! Compiles a CompactOp array (typically the cone-pruned pipeline ops) into
//! native x86-64 code. The compiled function evaluates all ops in one pass
//! without dirty-bit checks, writing changed tile indices to an output buffer.
//!
//! Feature-gated behind `cranelift_jit`.

use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_codegen::{isa, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::sync::Arc;
use target_lexicon::Triple;

use crate::simulation::{
    COP_ADD, COP_AND, COP_BITSEL, COP_CARRY, COP_CONST, COP_DEC3, COP_GENERIC, COP_MUX, COP_MUX4,
    COP_MUX16, COP_NOT, COP_OR, COP_RAM, COP_SHL, COP_SHR, COP_SUB, COP_THRESHOLD_VIA, COP_VIA,
    COP_WIRE, COP_WIRE_D, COP_WIRE_H, COP_WIRE_L, COP_WIRE_R, COP_WIRE_U, COP_WIRE_V, COP_WVIA,
    COP_XOR, COP_ZERO, CompactOp,
};

// ---------------------------------------------------------------------------
// JIT program wrapper
// ---------------------------------------------------------------------------

struct SendJITModule(JITModule);
unsafe impl Send for SendJITModule {}
unsafe impl Sync for SendJITModule {}

/// JIT function signature:
///   (tiles_ptr: *mut u8, changed_buf: *mut u32, buf_cap: u32) -> u32
///
/// - tiles_ptr: pointer to tilemap.values[0] (stride = TILE_STRIDE bytes per value)
/// - changed_buf: output buffer for indices of tiles that changed value
/// - buf_cap: capacity of changed_buf
/// - returns: number of changed tiles written to changed_buf
pub type TileEvalJitFn = unsafe extern "C" fn(*mut u8, *mut u32, u32) -> u32;

pub struct TileEvalJitProgram {
    _module: SendJITModule,
    pub func_ptr: TileEvalJitFn,
    pub op_count: usize,
}

impl std::fmt::Debug for TileEvalJitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileEvalJitProgram")
            .field("op_count", &self.op_count)
            .finish()
    }
}

unsafe impl Send for TileEvalJitProgram {}
unsafe impl Sync for TileEvalJitProgram {}

/// Value stride in bytes. Sprint 385 relocated tile values into the SoA
/// `Tilemap::values: Vec<AtomicU64>`, so the JIT walks a dense u64 array
/// (8 bytes per tile) instead of the old 16-byte `Tile` struct.
const TILE_STRIDE: i64 = 8;

// Sprint 273.1: Compile-time layout verification.
const _: () = {
    // AtomicU64 is repr(transparent) over u64; the JIT loads/stores raw u64s
    // at values_ptr + idx * 8.
    assert!(
        std::mem::size_of::<std::sync::atomic::AtomicU64>() == TILE_STRIDE as usize,
        "AtomicU64 size mismatch — JIT assumes 8-byte value stride"
    );
};

// ---------------------------------------------------------------------------
// Preflight verification
// ---------------------------------------------------------------------------

/// Sprint 273.1: Check if a CompactOp array is safe to JIT-compile.
/// Returns Err with reason if any unsupported ops are present.
pub fn preflight_check(ops: &[CompactOp]) -> Result<(), String> {
    for (i, op) in ops.iter().enumerate() {
        if op.op == COP_GENERIC {
            return Err(format!(
                "cone op {} (tile {}) is COP_GENERIC — JIT cannot handle eval_tile fallback",
                i, op.idx
            ));
        }
        if op.op == COP_WIRE {
            return Err(format!(
                "cone op {} (tile {}) is COP_WIRE — JIT drops 4th input (omnidirectional wire)",
                i, op.idx
            ));
        }
        if op.op == COP_THRESHOLD_VIA {
            return Err(format!(
                "cone op {} (tile {}) is COP_THRESHOLD_VIA — JIT signature has 3 inputs, \
                 threshold via needs 4 in-plane neighbors + cross-layer source",
                i, op.idx
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// Compile a CompactOp array into a native tile evaluation function.
///
/// Performs preflight verification before compilation. Returns Err if the
/// cone contains unsupported ops (COP_GENERIC, COP_WIRE, COP_THRESHOLD_VIA).
pub fn compile_tile_eval(
    ops: &[CompactOp],
    wvia_params: &[(usize, u8, u64)],
) -> Result<Arc<TileEvalJitProgram>, String> {
    // Sprint 273.1: Preflight — refuse compilation if cone has unsupported ops.
    preflight_check(ops)?;
    let triple = Triple::host();
    let bld = settings::builder();
    let flags = settings::Flags::new(bld);
    let target_isa = isa::lookup(triple.clone())
        .map_err(|e| format!("ISA lookup: {e}"))?
        .finish(flags)
        .map_err(|e| format!("ISA finish: {e}"))?;

    let jitb = JITBuilder::with_isa(target_isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jitb);

    // Signature: (tiles_ptr: i64, changed_buf: i64, buf_cap: i32) -> i32
    let mut ctx = module.make_context();
    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(&triple));
    sig.params.push(AbiParam::new(types::I64)); // tiles_ptr
    sig.params.push(AbiParam::new(types::I64)); // changed_buf
    sig.params.push(AbiParam::new(types::I32)); // buf_cap
    sig.returns.push(AbiParam::new(types::I32)); // changed_count

    let mut func = cranelift_codegen::ir::Function::new();
    func.signature = sig.clone();

    let mut fbctx = FunctionBuilderContext::new();
    let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);

    // Variable for changed count
    let var_changed = Variable::from_u32(0);
    fb.declare_var(var_changed, types::I32);

    // Entry block
    let entry_block = fb.create_block();
    fb.append_block_params_for_function_params(entry_block);
    fb.switch_to_block(entry_block);
    fb.seal_block(entry_block);

    let tiles_ptr = fb.block_params(entry_block)[0];
    let changed_buf = fb.block_params(entry_block)[1];
    let buf_cap = fb.block_params(entry_block)[2];

    let zero_i32 = fb.ins().iconst(types::I32, 0);
    fb.def_var(var_changed, zero_i32);

    let mflags = MemFlags::new();

    // Helper: load tile logic value at index
    let tile_load = |fb: &mut FunctionBuilder, idx: u32| -> Value {
        if idx == u32::MAX {
            return fb.ins().iconst(types::I64, 0);
        }
        let offset = (idx as i64) * TILE_STRIDE;
        fb.ins().load(types::I64, mflags, tiles_ptr, offset as i32)
    };

    // Precompute wvia params mapping: for each op that is COP_WVIA, what's its shift/mask?
    let mut wvia_idx = 0usize;

    // Emit one block per op (straight-line, no branching except change detection)
    for op in ops {
        let is_wvia = op.op == COP_WVIA;

        // Skip constants — they never change.
        if op.op == COP_CONST {
            if is_wvia {
                wvia_idx += 1;
            }
            continue;
        }

        // COP_GENERIC and COP_WIRE are rejected by preflight_check.
        // If we somehow get here, skip safely.
        if op.op == COP_GENERIC || op.op == COP_WIRE {
            if is_wvia {
                wvia_idx += 1;
            }
            continue;
        }

        let idx = op.idx;
        let v0 = tile_load(&mut fb, op.in0);
        let v1 = tile_load(&mut fb, op.in1);
        let v2 = tile_load(&mut fb, op.in2);
        let current = tile_load(&mut fb, idx);

        let result = match op.op {
            COP_WIRE_R | COP_VIA => v0,
            COP_WIRE_L => v1,
            COP_WIRE_D | COP_WIRE_U => v2,
            COP_WIRE_H | COP_OR => fb.ins().bor(v0, v1),
            COP_WIRE_V => fb.ins().bor(v1, v2),
            // COP_WIRE rejected by preflight — unreachable.
            COP_WIRE => unreachable!("COP_WIRE should be caught by preflight_check"),
            COP_AND => fb.ins().band(v0, v1),
            COP_XOR => fb.ins().bxor(v0, v1),
            COP_MUX => {
                // if v2 != 0 { v0 } else { v1 }
                let zero = fb.ins().iconst(types::I64, 0);
                let cond =
                    fb.ins()
                        .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, v2, zero);
                fb.ins().select(cond, v0, v1)
            }
            COP_NOT => fb.ins().bnot(v0),
            COP_ZERO => {
                // if v0 == 0 { MAX } else { 0 }
                let zero = fb.ins().iconst(types::I64, 0);
                let max_val = fb.ins().iconst(types::I64, u64::MAX as i64);
                let cond = fb
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, v0, zero);
                fb.ins().select(cond, max_val, zero)
            }
            COP_ADD => fb.ins().iadd(v0, v1),
            COP_SUB => fb.ins().isub(v0, v1),
            COP_SHR => {
                let mask63 = fb.ins().iconst(types::I64, 63);
                let shift = fb.ins().band(v1, mask63);
                fb.ins().ushr(v0, shift)
            }
            COP_SHL => {
                let mask63 = fb.ins().iconst(types::I64, 63);
                let shift = fb.ins().band(v1, mask63);
                fb.ins().ishl(v0, shift)
            }
            COP_MUX16 => {
                // lane = v1 & 0xF; if lane < 8 { v0 } else { v2 } >> (lane&7)*8 & 0xFF
                let mask_f = fb.ins().iconst(types::I64, 0xF);
                let lane = fb.ins().band(v1, mask_f);
                let eight = fb.ins().iconst(types::I64, 8);
                let cond = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                    lane,
                    eight,
                );
                let source = fb.ins().select(cond, v0, v2);
                let mask7 = fb.ins().iconst(types::I64, 7);
                let lane_within = fb.ins().band(lane, mask7);
                let three = fb.ins().iconst(types::I64, 3);
                let shift_amt = fb.ins().ishl(lane_within, three);
                let shifted = fb.ins().ushr(source, shift_amt);
                let mask_ff = fb.ins().iconst(types::I64, 0xFF);
                fb.ins().band(shifted, mask_ff)
            }
            COP_DEC3 => {
                let mask7 = fb.ins().iconst(types::I64, 7);
                let bits = fb.ins().band(v0, mask7);
                let one = fb.ins().iconst(types::I64, 1);
                fb.ins().ishl(one, bits)
            }
            COP_BITSEL => {
                let mask63 = fb.ins().iconst(types::I64, 63);
                let shift = fb.ins().band(v1, mask63);
                let shifted = fb.ins().ushr(v0, shift);
                let one = fb.ins().iconst(types::I64, 1);
                let bit = fb.ins().band(shifted, one);
                let zero = fb.ins().iconst(types::I64, 0);
                let max_val = fb.ins().iconst(types::I64, u64::MAX as i64);
                let cond =
                    fb.ins()
                        .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, bit, zero);
                fb.ins().select(cond, max_val, zero)
            }
            COP_CARRY => {
                // if v0 > v1 { MAX } else { 0 }
                let zero = fb.ins().iconst(types::I64, 0);
                let max_val = fb.ins().iconst(types::I64, u64::MAX as i64);
                let cond = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThan,
                    v0,
                    v1,
                );
                fb.ins().select(cond, max_val, zero)
            }
            COP_WVIA => {
                if wvia_idx < wvia_params.len() {
                    let (_, shift, mask) = wvia_params[wvia_idx];
                    let shift_val = fb.ins().iconst(types::I64, shift as i64);
                    let mask_val = fb.ins().iconst(types::I64, mask as i64);
                    let shifted = fb.ins().ushr(v0, shift_val);
                    fb.ins().band(shifted, mask_val)
                } else {
                    v0
                }
            }
            COP_MUX4 => {
                // (v0 >> ((v2 & 0b11) * 8)) & 0xFF
                let mask3 = fb.ins().iconst(types::I64, 3);
                let sel = fb.ins().band(v2, mask3);
                let three = fb.ins().iconst(types::I64, 3);
                let shift_amt = fb.ins().ishl(sel, three);
                let shifted = fb.ins().ushr(v0, shift_amt);
                let mask_ff = fb.ins().iconst(types::I64, 0xFF);
                fb.ins().band(shifted, mask_ff)
            }
            COP_RAM => {
                // if v2 != 0 { v0 } else { current }
                let zero = fb.ins().iconst(types::I64, 0);
                let cond =
                    fb.ins()
                        .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, v2, zero);
                fb.ins().select(cond, v0, current)
            }
            _ => {
                if is_wvia {
                    wvia_idx += 1;
                }
                continue;
            }
        };

        if is_wvia {
            wvia_idx += 1;
        }

        // Compare result with current: if different, store and record in changed_buf
        let changed_block = fb.create_block();
        let merge_block = fb.create_block();

        let cond = fb.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::NotEqual,
            result,
            current,
        );
        fb.ins().brif(cond, changed_block, &[], merge_block, &[]);

        // Changed block: store new value, append index to changed_buf
        fb.switch_to_block(changed_block);
        fb.seal_block(changed_block);

        let store_offset = (idx as i64) * TILE_STRIDE;
        fb.ins()
            .store(mflags, result, tiles_ptr, store_offset as i32);

        // Append to changed_buf if space available
        let cur_changed = fb.use_var(var_changed);
        let has_space = fb.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
            cur_changed,
            buf_cap,
        );
        let write_block = fb.create_block();
        let skip_write = fb.create_block();
        fb.ins().brif(has_space, write_block, &[], skip_write, &[]);

        fb.switch_to_block(write_block);
        fb.seal_block(write_block);
        // changed_buf[cur_changed] = idx
        let four = fb.ins().iconst(types::I32, 4); // u32 = 4 bytes
        let byte_off = fb.ins().imul(cur_changed, four);
        let byte_off_64 = fb.ins().sextend(types::I64, byte_off);
        let write_addr = fb.ins().iadd(changed_buf, byte_off_64);
        let idx_val = fb.ins().iconst(types::I32, idx as i64);
        fb.ins().store(mflags, idx_val, write_addr, 0);
        fb.ins().jump(skip_write, &[]);

        fb.switch_to_block(skip_write);
        fb.seal_block(skip_write);
        let new_changed = fb.ins().iadd_imm(cur_changed, 1);
        fb.def_var(var_changed, new_changed);
        fb.ins().jump(merge_block, &[]);

        fb.switch_to_block(merge_block);
        fb.seal_block(merge_block);
    }

    // Return changed count
    let final_changed = fb.use_var(var_changed);
    fb.ins().return_(&[final_changed]);

    fb.finalize();

    // Compile
    ctx.func = func;
    static FN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let suffix = FN_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let fname = format!("tile_eval_{suffix}");

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
    let jit_fn: TileEvalJitFn = unsafe { std::mem::transmute(fnptr) };

    Ok(Arc::new(TileEvalJitProgram {
        _module: SendJITModule(module),
        func_ptr: jit_fn,
        op_count: ops.len(),
    }))
}
