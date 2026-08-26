//! Sprint 372 (Gate D.1): Compiler backend foundation — embedded AST → V2 assembly.
//!
//! This is the first rung of **Gate D**: a minimal compiler that lowers a structured
//! high-level program (expressed as a Rust AST) to V2 assembly *text*, which the
//! existing [`assemble_v2`](crate::tile_cpu::assemble_v2) two-pass assembler turns
//! into machine words. The generated assembly is human-readable — it *is* the
//! "port software vs hand-asm" artifact.
//!
//! The compiler does the genuinely hard part of a backend: register allocation,
//! stack-frame construction, the calling convention, and control-flow lowering. A
//! text source parser (a C-like front end) is a later sprint that produces this same
//! AST.
//!
//! # Calling convention / register usage (Gate C ABI)
//! - Args in `R0..R3`, return value in `R0`.
//! - `R14 = FP` (frame pointer), `R15 = SP` (stack pointer, empty-descending).
//! - `R4` = transient binary-op left-operand holder, `R5` = transient address scratch.
//! - Locals live in the stack frame at `FP - slot` (RAM-backed); the expression
//!   evaluation stack uses `PUSH`/`POP` below `SP`.
//!
//! # Returns without MTLR
//! `MTLR` has no physical implementation (ISS-only), so we never restore `LR`.
//! Instead the prologue saves the return address (`MFLR R5; PUSH R5`) and the
//! epilogue returns with `POP R5; JMPR R5` — both `MFLR` and `JMPR` are physical
//! (Sprints 368 / 369). This makes arbitrary recursion work on the tile CPU.
//! Because `JMPR`'s physical target path lives in the extended-PC commit block,
//! compiled programs run under `with_extended_pc()` / `enable_extended_pc()`.
//!
//! # Scope (D.1)
//! One 64-bit integer type (unsigned comparisons); `+ - * & | ^`; the six
//! comparisons in `if`/`while` conditions; `let`/assign/`if`/`while`/`return`;
//! functions with ≤4 params and recursion. Programs must fit the assembler's
//! 128-word ROM and the RAM stack. Deferred: >4 params, materialized booleans,
//! `&&`/`||`, globals, arrays, multiple types, larger programs.

use crate::tile_cpu::assemble_v2;
use crate::tile_cpu::v2_mmio::{V2_MMIO_BASE, V2_MMIO_END};
use crate::tile_cpu::v2_mmio_devices::{
    MMIO_ACCEL_ARG_DATA, MMIO_ACCEL_ARG_SELECT, MMIO_ACCEL_RESULT,
};

// ===================================================================
// AST
// ===================================================================

pub use tileuniverse_v2_lang::*;

// ===================================================================
// Codegen
// ===================================================================

// Fixed register assignments.
const FP: &str = "R14";
const SP: &str = "R15";
const RESULT: &str = "R0"; // primary expression result / return value / arg0
const LEFT: &str = "R4"; // transient binary-op left-operand holder
const ADDR: &str = "R5"; // transient address-computation scratch

// Sprint 378 (Perf P2): register allocation. Locals live in the callee-saved bank
// R8..R13 (6 registers) instead of RAM frame slots — a local access becomes a single
// register move instead of MOV+SUBI+LD/ST. Functions with more locals than this fall
// back to the RAM-frame scheme. R14 = FP (frame mode only), R15 = SP.
const FIRST_LOCAL_REG: u8 = 8;
const MAX_REG_LOCALS: usize = 6;

/// Where a local variable lives.
#[derive(Debug, Clone, Copy)]
enum LocalLoc {
    /// In a callee-saved register R8..R13 (register mode).
    Reg(u8),
    /// In a RAM frame slot at `FP - slot` (frame mode, > 6 locals).
    Slot(usize),
}

struct FnCtx {
    /// name → location (register or frame slot).
    locals: Vec<(String, LocalLoc)>,
    /// Number of RAM frame slots (0 in register mode).
    frame_slots: usize,
    /// Callee-saved registers used for locals, to save/restore (register mode).
    reg_locals: Vec<u8>,
    epilogue: String,
}

impl FnCtx {
    fn loc_of(&self, name: &str) -> Option<LocalLoc> {
        self.locals.iter().find(|(n, _)| n == name).map(|(_, l)| *l)
    }
}

/// A resolved global array: its fixed RAM base address (Sprint 382, Gate F).
#[derive(Debug, Clone, Copy)]
struct GlobalLoc {
    base: u8,
}

struct Codegen {
    out: Vec<String>,
    label_counter: usize,
    /// name → fixed RAM base/len for global arrays.
    globals: Vec<(String, GlobalLoc)>,
    /// Sprint 388: the accel-marked function this program may call — `(name, arity)`.
    /// Calls to it lower to the MMIO accelerator protocol instead of `CALL.W`.
    accel: Option<(String, usize)>,
}

impl Codegen {
    fn new() -> Self {
        Codegen {
            out: Vec::new(),
            label_counter: 0,
            globals: Vec::new(),
            accel: None,
        }
    }

    /// Resolve a global array by name.
    fn global_of(&self, name: &str) -> Option<GlobalLoc> {
        self.globals
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, g)| *g)
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.out.push(format!("    {}", line.into()));
    }

    fn label(&mut self, name: &str) {
        self.out.push(format!("{name}:"));
    }

    fn comment(&mut self, text: &str) {
        self.out.push(format!("    ; {text}"));
    }

    fn fresh_label(&mut self) -> String {
        let l = format!("_cc_{}", self.label_counter);
        self.label_counter += 1;
        l
    }

    // ---- frame helpers ----

    /// Emit address of local `slot` into `ADDR` and return the register token to use
    /// as the `[reg]` base. Slot 0 lives at FP directly, so no scratch needed.
    fn local_addr_reg(&mut self, slot: usize) -> String {
        if slot == 0 {
            FP.to_string()
        } else {
            self.emit(format!("MOV {ADDR}, {FP}"));
            self.emit(format!("SUBI {ADDR}, {slot}"));
            ADDR.to_string()
        }
    }

    /// Load local `name` into RESULT (R0).
    fn load_local(&mut self, ctx: &FnCtx, name: &str) -> Result<(), CompileError> {
        match ctx
            .loc_of(name)
            .ok_or_else(|| CompileError(format!("use of undeclared variable '{name}'")))?
        {
            LocalLoc::Reg(r) => self.emit(format!("MOV {RESULT}, R{r}")),
            LocalLoc::Slot(slot) => {
                let base = self.local_addr_reg(slot);
                self.emit(format!("LD {RESULT}, [{base}]"));
            }
        }
        Ok(())
    }

    /// Store RESULT (R0) into local `name`.
    fn store_local(&mut self, ctx: &FnCtx, name: &str) -> Result<(), CompileError> {
        match ctx
            .loc_of(name)
            .ok_or_else(|| CompileError(format!("assignment to undeclared variable '{name}'")))?
        {
            LocalLoc::Reg(r) => self.emit(format!("MOV R{r}, {RESULT}")),
            LocalLoc::Slot(slot) => {
                let base = self.local_addr_reg(slot);
                self.emit(format!("ST [{base}], {RESULT}"));
            }
        }
        Ok(())
    }

    // ---- expressions (result left in RESULT/R0) ----

    fn gen_expr(&mut self, ctx: &FnCtx, e: &Expr) -> Result<(), CompileError> {
        match e {
            Expr::Int(n) => {
                if *n < 0 || *n > 0xFFFF {
                    return Err(CompileError(format!(
                        "integer literal {n} out of supported range (0..=65535)"
                    )));
                }
                // LDI auto-widens to LDI.W in the assembler for values > 255.
                self.emit(format!("LDI {RESULT}, {n}"));
            }
            Expr::Var(name) => self.load_local(ctx, name)?,
            Expr::Bin(op, a, b) => {
                self.gen_expr(ctx, a)?; // a -> R0
                // Sprint 378 (Perf P2): when the right operand is trivial (a register
                // local or a literal) it can't clobber R0, so we combine directly with
                // no PUSH/POP spill. `SUB dst, src` computes dst - src = a - b directly.
                match trivial_operand(ctx, b) {
                    Some(Trivial::Reg(r)) => {
                        let src = format!("R{r}");
                        self.emit(format!("{} {RESULT}, {src}", binop_mnem(*op)));
                    }
                    Some(Trivial::Imm(n)) => {
                        self.emit(format!("LDI {LEFT}, {n}")); // b -> R4
                        self.emit(format!("{} {RESULT}, {LEFT}", binop_mnem(*op)));
                    }
                    None => {
                        self.emit(format!("PUSH {RESULT}")); // save a
                        self.gen_expr(ctx, b)?; // b -> R0
                        self.emit(format!("POP {LEFT}")); // R4 = a, R0 = b
                        match op {
                            // Commutative: R0 = b op a = a op b.
                            ArithOp::Add
                            | ArithOp::Mul
                            | ArithOp::And
                            | ArithOp::Or
                            | ArithOp::Xor => {
                                self.emit(format!("{} {RESULT}, {LEFT}", binop_mnem(*op)));
                            }
                            // Non-commutative: R0 = a - b = R4 - R0.
                            ArithOp::Sub => {
                                self.emit(format!("SUB {LEFT}, {RESULT}"));
                                self.emit(format!("MOV {RESULT}, {LEFT}"));
                            }
                        }
                    }
                }
            }
            Expr::Call(name, args) => self.gen_call(ctx, name, args)?,
            Expr::Index(name, idx) => {
                let g = self
                    .global_of(name)
                    .ok_or_else(|| CompileError(format!("use of undeclared array '{name}'")))?;
                self.gen_expr(ctx, idx)?; // index -> R0
                // address = R5(index) + base; load the cell into R0. (Bounds are the
                // program's responsibility — no runtime check, like C.)
                self.emit(format!("MOV {ADDR}, {RESULT}"));
                self.emit(format!("LD {RESULT}, [{ADDR}+{}]", g.base));
            }
        }
        Ok(())
    }

    fn gen_call(&mut self, ctx: &FnCtx, name: &str, args: &[Expr]) -> Result<(), CompileError> {
        // Sprint 375 (Gate E): `putchar(c)` is a builtin — write the low byte of `c` to
        // the MMIO console device (STB [CONSOLE_DATA], R0) instead of a function call.
        if name == "putchar" {
            if args.len() != 1 {
                return Err(CompileError(format!(
                    "putchar expects exactly 1 argument, got {}",
                    args.len()
                )));
            }
            self.gen_expr(ctx, &args[0])?; // char value -> R0
            self.emit("STB [CONSOLE_DATA], R0");
            return Ok(());
        }
        // Sprint 388 (HW/SW co-design, Phase 2): a call to the `accel`-marked function
        // lowers to the MMIO accelerator's indexed-operand protocol instead of CALL.W:
        // for each argument, write its index to ARG_SELECT and its value to ARG_DATA
        // (full 64-bit ST — the device latches operand[index]); then a single LD from
        // RESULT triggers evaluation of the synthesized tile datapath and returns the
        // answer in R0. No LR/stack-frame involvement — the "call" is three stores and
        // a load per argument, like the putchar builtin but with a result.
        if let Some((accel_name, accel_arity)) = self.accel.clone()
            && name == accel_name
        {
            if args.len() != accel_arity {
                return Err(CompileError(format!(
                    "accel function '{accel_name}' expects {accel_arity} argument(s), got {}",
                    args.len()
                )));
            }
            // Stage every argument on the eval stack first: an argument expression
            // may itself contain calls (including nested accel calls) that would
            // clobber the device's operand latches mid-sequence.
            for arg in args {
                self.gen_expr(ctx, arg)?;
                self.emit(format!("PUSH {RESULT}"));
            }
            self.emit(format!("LDI {ADDR}, 0")); // base 0 for [R5+addr] MMIO access
            for i in (0..args.len()).rev() {
                self.emit(format!("POP {RESULT}")); // arg i value
                self.emit(format!("LDI {LEFT}, {i}"));
                self.emit(format!("ST [{ADDR}+{MMIO_ACCEL_ARG_SELECT}], {LEFT}"));
                self.emit(format!("ST [{ADDR}+{MMIO_ACCEL_ARG_DATA}], {RESULT}"));
            }
            // Read RESULT: the device evaluates the tile datapath on this access.
            self.emit(format!("LD {RESULT}, [{ADDR}+{MMIO_ACCEL_RESULT}]"));
            return Ok(());
        }
        if args.len() > 4 {
            return Err(CompileError(format!(
                "call to '{name}' with {} args; D.1 supports at most 4",
                args.len()
            )));
        }
        // Sprint 380 (Perf P4): a single-argument call needs no stack staging — the arg
        // evaluates into R0, which IS arg-register 0, so the old `PUSH R0; POP R0`
        // round-trip was a no-op. (Most calls in practice — fib, print_int, div10.)
        if args.len() == 1 {
            self.gen_expr(ctx, &args[0])?; // arg -> R0 = arg0
            self.emit(format!("CALL.W {name}"));
            return Ok(());
        }
        // Multi-arg: evaluate each arg fully and stage it on the stack; only then move
        // into the arg registers. This survives nested calls clobbering R0..R3.
        for arg in args {
            self.gen_expr(ctx, arg)?;
            self.emit(format!("PUSH {RESULT}"));
        }
        for i in (0..args.len()).rev() {
            self.emit(format!("POP R{i}"));
        }
        // Wide call: the callee may live anywhere in the (extended) instruction space.
        self.emit(format!("CALL.W {name}"));
        Ok(())
    }

    // ---- conditions: emit CMP + a jump to `skip` taken when the condition is FALSE ----

    fn gen_cond_skip(
        &mut self,
        ctx: &FnCtx,
        cond: &Cond,
        skip_label: &str,
    ) -> Result<(), CompileError> {
        let Cond::Cmp(op, a, b) = cond;
        // Evaluate so LEFT(R4) = a, RESULT(R0) = b. All flag-setting work (incl. any
        // load) happens before CMP; nothing touches flags before the jump.
        self.gen_expr(ctx, a)?; // a -> R0
        // Sprint 378 (Perf P2): a trivial right operand avoids the PUSH/POP spill —
        // move a to R4, then load b into R0.
        match trivial_operand(ctx, b) {
            Some(Trivial::Reg(r)) => {
                self.emit(format!("MOV {LEFT}, {RESULT}")); // R4 = a
                self.emit(format!("MOV {RESULT}, R{r}")); // R0 = b
            }
            Some(Trivial::Imm(n)) => {
                self.emit(format!("MOV {LEFT}, {RESULT}")); // R4 = a
                self.emit(format!("LDI {RESULT}, {n}")); // R0 = b
            }
            None => {
                self.emit(format!("PUSH {RESULT}"));
                self.gen_expr(ctx, b)?;
                self.emit(format!("POP {LEFT}")); // R4 = a, R0 = b
            }
        }
        // CMP a,b sets Z=(a==b), C=(a<b) unsigned. Choose operands + the negated jump.
        // Wide conditional branches (`.W`) so the skip target can live anywhere in
        // the extended instruction space (programs > 128 instructions).
        match op {
            CmpOp::Eq => {
                self.emit(format!("CMP {LEFT}, {RESULT}"));
                self.emit(format!("JNZ.W {skip_label}"));
            }
            CmpOp::Ne => {
                self.emit(format!("CMP {LEFT}, {RESULT}"));
                self.emit(format!("JZ.W {skip_label}"));
            }
            CmpOp::Lt => {
                self.emit(format!("CMP {LEFT}, {RESULT}"));
                self.emit(format!("JNC.W {skip_label}"));
            }
            CmpOp::Ge => {
                self.emit(format!("CMP {LEFT}, {RESULT}"));
                self.emit(format!("JC.W {skip_label}"));
            }
            CmpOp::Gt => {
                // a>b <=> b<a: CMP b,a, true when C; skip when NC.
                self.emit(format!("CMP {RESULT}, {LEFT}"));
                self.emit(format!("JNC.W {skip_label}"));
            }
            CmpOp::Le => {
                // a<=b <=> !(b<a): CMP b,a, true when NC; skip when C.
                self.emit(format!("CMP {RESULT}, {LEFT}"));
                self.emit(format!("JC.W {skip_label}"));
            }
        }
        Ok(())
    }

    // ---- statements ----

    fn gen_stmt(&mut self, ctx: &FnCtx, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::Let(name, e) | Stmt::Assign(name, e) => {
                self.gen_expr(ctx, e)?;
                self.store_local(ctx, name)?;
            }
            Stmt::Expr(e) => {
                self.gen_expr(ctx, e)?; // evaluated for effect; R0 discarded
            }
            Stmt::StoreIndex(name, idx, val) => {
                let g = self
                    .global_of(name)
                    .ok_or_else(|| CompileError(format!("store to undeclared array '{name}'")))?;
                // Evaluate the index first and stage it (value eval may clobber everything).
                self.gen_expr(ctx, idx)?; // index -> R0
                self.emit(format!("PUSH {RESULT}"));
                self.gen_expr(ctx, val)?; // value -> R0
                self.emit(format!("POP {ADDR}")); // R5 = index
                self.emit(format!("ST [{ADDR}+{}], {RESULT}", g.base));
            }
            Stmt::Return(e) => {
                self.gen_expr(ctx, e)?; // result in R0
                let epi = ctx.epilogue.clone();
                self.emit(format!("JMP.W {epi}"));
            }
            Stmt::If(cond, then_body, else_body) => {
                let l_else = self.fresh_label();
                let l_end = self.fresh_label();
                self.gen_cond_skip(ctx, cond, &l_else)?;
                self.gen_block(ctx, then_body)?;
                self.emit(format!("JMP.W {l_end}"));
                self.label(&l_else);
                self.gen_block(ctx, else_body)?;
                self.label(&l_end);
            }
            Stmt::While(cond, body) => {
                let l_top = self.fresh_label();
                let l_end = self.fresh_label();
                self.label(&l_top);
                self.gen_cond_skip(ctx, cond, &l_end)?;
                self.gen_block(ctx, body)?;
                self.emit(format!("JMP.W {l_top}"));
                self.label(&l_end);
            }
        }
        Ok(())
    }

    fn gen_block(&mut self, ctx: &FnCtx, body: &[Stmt]) -> Result<(), CompileError> {
        for s in body {
            self.gen_stmt(ctx, s)?;
        }
        Ok(())
    }

    // ---- function ----

    fn gen_func(&mut self, func: &Func) -> Result<(), CompileError> {
        if func.params.len() > 4 {
            return Err(CompileError(format!(
                "function '{}' has {} params; D.1 supports at most 4",
                func.name,
                func.params.len()
            )));
        }
        let locals = collect_locals(func);
        if locals.len() > 255 {
            return Err(CompileError(format!(
                "function '{}' has too many locals ({})",
                func.name,
                locals.len()
            )));
        }

        // Sprint 378 (Perf P2): allocate locals to registers when they fit; otherwise
        // fall back to the RAM frame. Register mode needs no FP/frame.
        let register_mode = locals.len() <= MAX_REG_LOCALS;
        let ctx = if register_mode {
            let reg_locals: Vec<u8> = (0..locals.len())
                .map(|i| FIRST_LOCAL_REG + i as u8)
                .collect();
            FnCtx {
                locals: locals
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), LocalLoc::Reg(FIRST_LOCAL_REG + i as u8)))
                    .collect(),
                frame_slots: 0,
                reg_locals,
                epilogue: format!("_cc_epi_{}", func.name),
            }
        } else {
            FnCtx {
                locals: locals
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), LocalLoc::Slot(i)))
                    .collect(),
                frame_slots: locals.len(),
                reg_locals: Vec::new(),
                epilogue: format!("_cc_epi_{}", func.name),
            }
        };

        self.out.push(String::new());
        self.label(&func.name);
        self.comment(&format!(
            "fn {}({}) -> R0  [{}]",
            func.name,
            func.params.join(", "),
            if register_mode {
                format!("{} reg locals", locals.len())
            } else {
                format!("{} frame slots", ctx.frame_slots)
            }
        ));

        // Sprint 380 (Perf P4): a leaf function (no real calls) never clobbers LR, so it
        // skips saving the return address and returns via RET (physical LR) instead of
        // the MFLR/PUSH … POP/JMPR idiom.
        let leaf = is_leaf(func);

        // Prologue: non-leaf functions save the return address (return via JMPR — MTLR
        // is ISS-only); leaves leave LR untouched and RET at the end.
        if !leaf {
            self.emit(format!("MFLR {ADDR}")); // R5 = return address
            self.emit(format!("PUSH {ADDR}")); // save it
        }
        if register_mode {
            // Save the callee-saved registers we use for locals, then move params in.
            for &r in &ctx.reg_locals {
                self.emit(format!("PUSH R{r}"));
            }
            for (i, _param) in func.params.iter().enumerate() {
                // Param i arrived in R{i}; local i is R{FIRST_LOCAL_REG + i}.
                self.emit(format!("MOV R{}, R{i}", FIRST_LOCAL_REG + i as u8));
            }
        } else {
            self.emit(format!("PUSH {FP}")); // save caller FP
            self.emit(format!("MOV {FP}, {SP}")); // FP = SP
            if ctx.frame_slots > 0 {
                self.emit(format!("SUBI {SP}, {}", ctx.frame_slots)); // allocate locals
            }
            for (i, _param) in func.params.iter().enumerate() {
                let base = self.local_addr_reg(i);
                self.emit(format!("ST [{base}], R{i}"));
            }
        }

        // Body.
        self.gen_block(&ctx, &func.body)?;

        // Epilogue (also the fall-through target if a function lacks an explicit
        // trailing return — R0 is then whatever the last expression left). The eval
        // stack is balanced at every statement boundary, so SP is at the post-prologue
        // level here in register mode (no FP needed to restore it).
        let epi = ctx.epilogue.clone();
        self.label(&epi);
        if register_mode {
            for &r in ctx.reg_locals.iter().rev() {
                self.emit(format!("POP R{r}")); // restore callee-saved locals
            }
        } else {
            self.emit(format!("MOV {SP}, {FP}")); // deallocate locals
            self.emit(format!("POP {FP}")); // restore caller FP
        }
        if leaf {
            self.emit("RET"); // LR is intact (no calls clobbered it)
        } else {
            self.emit(format!("POP {ADDR}")); // R5 = saved return address
            self.emit(format!("JMPR {ADDR}")); // return (physical; avoids MTLR)
        }
        Ok(())
    }
}

/// Sprint 378 (Perf P2): a right-hand operand that can be combined without spilling R0
/// — it cannot itself clobber R0 during evaluation.
enum Trivial {
    /// A register-backed local (its value is already in `R{reg}`).
    Reg(u8),
    /// An integer literal (0..=65535).
    Imm(i64),
}

/// Classify `e` as a trivial right operand, or `None` if it must be spilled (a call,
/// a nested expression, or a frame-slot local whose load clobbers R0).
fn trivial_operand(ctx: &FnCtx, e: &Expr) -> Option<Trivial> {
    match e {
        Expr::Int(n) if *n >= 0 && *n <= 0xFFFF => Some(Trivial::Imm(*n)),
        Expr::Var(name) => match ctx.loc_of(name) {
            Some(LocalLoc::Reg(r)) => Some(Trivial::Reg(r)),
            _ => None,
        },
        _ => None,
    }
}

/// The ALU mnemonic for a binary op. `OP dst, src` computes `dst = dst OP src`, so for
/// `dst = a`, `src = b` this is `a OP b` (incl. `SUB` = `a - b`).
fn binop_mnem(op: ArithOp) -> &'static str {
    match op {
        ArithOp::Add => "ADD",
        ArithOp::Sub => "SUB",
        ArithOp::Mul => "MUL",
        ArithOp::And => "AND",
        ArithOp::Or => "OR",
        ArithOp::Xor => "XOR",
    }
}

/// Sprint 380 (Perf P4): a function is a "leaf" if it makes no real function calls.
/// `putchar` compiles to a store (not `CALL.W`) and doesn't clobber LR, so it doesn't
/// count. A leaf's LR is never overwritten, so it can return with `RET` (physical LR)
/// without the MFLR/save/restore idiom.
fn is_leaf(func: &Func) -> bool {
    fn expr_has_call(e: &Expr) -> bool {
        match e {
            Expr::Call(name, args) => name != "putchar" || args.iter().any(expr_has_call),
            Expr::Bin(_, a, b) => expr_has_call(a) || expr_has_call(b),
            Expr::Index(_, idx) => expr_has_call(idx),
            Expr::Int(_) | Expr::Var(_) => false,
        }
    }
    fn cond_has_call(c: &Cond) -> bool {
        let Cond::Cmp(_, a, b) = c;
        expr_has_call(a) || expr_has_call(b)
    }
    fn stmts_have_call(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::Let(_, e) | Stmt::Assign(_, e) | Stmt::Return(e) | Stmt::Expr(e) => {
                expr_has_call(e)
            }
            Stmt::StoreIndex(_, idx, val) => expr_has_call(idx) || expr_has_call(val),
            Stmt::If(c, t, e) => cond_has_call(c) || stmts_have_call(t) || stmts_have_call(e),
            Stmt::While(c, b) => cond_has_call(c) || stmts_have_call(b),
        })
    }
    !stmts_have_call(&func.body)
}

/// Collect the ordered, de-duplicated set of local names: params first (in order),
/// then `let`-declared names in first-appearance order (recursing into blocks).
fn collect_locals(func: &Func) -> Vec<String> {
    let mut names: Vec<String> = func.params.clone();
    fn walk(body: &[Stmt], names: &mut Vec<String>) {
        for s in body {
            match s {
                Stmt::Let(name, _) => {
                    if !names.iter().any(|n| n == name) {
                        names.push(name.clone());
                    }
                }
                Stmt::If(_, t, e) => {
                    walk(t, names);
                    walk(e, names);
                }
                Stmt::While(_, b) => walk(b, names),
                _ => {}
            }
        }
    }
    walk(&func.body, &mut names);
    names
}

/// Compile a [`Program`] to V2 assembly *text*.
pub fn compile_program(program: &Program) -> Result<String, CompileError> {
    // Sprint 388 (HW/SW co-design, Phase 2): register the accel-marked function so its
    // call sites lower to the MMIO accelerator protocol. The function body is NOT
    // compiled to instructions — it is synthesized to a tile datapath by the caller
    // (`compile_source_with_accel`), which owns the resulting MMIO device.
    if program.accel_funcs.len() > 1 {
        return Err(CompileError(format!(
            "{} accel functions declared; this sprint supports at most one accelerator \
             per program (one MMIO device window)",
            program.accel_funcs.len()
        )));
    }
    if let Some(af) = program.accel_funcs.first() {
        if program.funcs.iter().any(|f| f.name == af.name) {
            return Err(CompileError(format!(
                "function '{}' is defined both as a regular function and as accel",
                af.name
            )));
        }
        if af.name == program.entry {
            return Err(CompileError(format!(
                "entry function '{}' cannot be an accel function",
                af.name
            )));
        }
    }
    if !program.funcs.iter().any(|f| f.name == program.entry) {
        return Err(CompileError(format!(
            "entry function '{}' not found",
            program.entry
        )));
    }
    let mut cg = Codegen::new();
    if let Some(af) = program.accel_funcs.first() {
        cg.accel = Some((af.name.clone(), af.params.len()));
    }

    // Sprint 382 (Gate F): allocate global arrays into RAM, avoiding the reserved MMIO
    // window (`V2_MMIO_BASE..=V2_MMIO_END`, i.e. 41..=63). A load/store to an MMIO
    // address is intercepted by a device, not RAM — so a global must lie wholly within a
    // contiguous non-MMIO region. Two regions are available below the descending stack:
    //   low  = [0, V2_MMIO_BASE)                     (41 cells)
    //   high = [V2_MMIO_END+1, stack_top - headroom) (toward the stack)
    // Each array is placed in the first region with room; arrays never span the hole.
    const STACK_HEADROOM: usize = 16;
    let lo_end = V2_MMIO_BASE as usize; // 41
    let hi_start = V2_MMIO_END as usize + 1; // 64
    let hi_end = (program.stack_top as usize).saturating_sub(STACK_HEADROOM);
    let mut lo_cursor = 0usize;
    let mut hi_cursor = hi_start;
    let mut total_init: Vec<(u8, i64)> = Vec::new(); // (cell address, value) for init.
    for g in &program.globals {
        if cg.global_of(&g.name).is_some() {
            return Err(CompileError(format!("duplicate global array '{}'", g.name)));
        }
        if g.len == 0 {
            return Err(CompileError(format!(
                "global array '{}' has length 0",
                g.name
            )));
        }
        // Pick the first region (low, then high) that fits the whole array contiguously.
        let base = if lo_cursor + g.len <= lo_end {
            let b = lo_cursor;
            lo_cursor += g.len;
            b
        } else if hi_cursor + g.len <= hi_end {
            let b = hi_cursor;
            hi_cursor += g.len;
            b
        } else {
            return Err(CompileError(format!(
                "no contiguous RAM region fits global '{}' ({} cells): low region has {} \
                 free, high region [{hi_start},{hi_end}) has {} free (MMIO window {}..={} \
                 is reserved)",
                g.name,
                g.len,
                lo_end.saturating_sub(lo_cursor),
                hi_end.saturating_sub(hi_cursor),
                V2_MMIO_BASE,
                V2_MMIO_END,
            )));
        };
        for (j, cell) in (0..g.len).map(|j| (j, g.init.get(j).copied().unwrap_or(0))) {
            if !(0..=0xFFFF).contains(&cell) {
                return Err(CompileError(format!(
                    "global '{}' initializer {cell} out of range (0..=65535)",
                    g.name
                )));
            }
            total_init.push(((base + j) as u8, cell));
        }
        cg.globals
            .push((g.name.clone(), GlobalLoc { base: base as u8 }));
    }

    // Entry stub: set up the stack, zero/literal-initialize globals, call the entry
    // function, then halt with the return value in R0.
    cg.comment(&format!(
        "Sprint 372 (Gate D.1) / 382 (Gate F) compiled program — entry: {}",
        program.entry
    ));
    cg.emit(format!("LDI {SP}, {}", program.stack_top));
    if !total_init.is_empty() {
        cg.comment(&format!(
            "init {} global cell(s) (deterministic; ISS==physical)",
            total_init.len()
        ));
        cg.emit(format!("LDI {ADDR}, 0")); // R5 = 0 base for [R5+addr] stores
        let mut last_val: Option<i64> = None;
        for (addr, val) in total_init {
            if last_val != Some(val) {
                cg.emit(format!("LDI {RESULT}, {val}")); // reload only when it changes
                last_val = Some(val);
            }
            cg.emit(format!("ST [{ADDR}+{addr}], {RESULT}"));
        }
    }
    cg.emit(format!("CALL.W {}", program.entry));
    cg.emit("HALT");
    for func in &program.funcs {
        cg.gen_func(func)?;
    }
    Ok(cg.out.join("\n"))
}

/// Compile a [`Program`] straight to machine words (assembles the generated text).
pub fn compile_to_words(program: &Program) -> Result<Vec<u32>, CompileError> {
    let asm = compile_program(program)?;
    assemble_v2(&asm).map_err(|e| {
        CompileError(format!(
            "internal: generated assembly failed to assemble at line {}: {} \n--- asm ---\n{}",
            e.line, e.message, asm
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_iss::V2Iss;

    /// Compile a program, run it on the ISS to halt, and return R0 (the entry's
    /// return value). Compiled programs target the extended-PC config (returns use
    /// JMPR, which is physical only under extended PC).
    fn run_compiled(program: &Program) -> u64 {
        let words = compile_to_words(program).expect("compile failed");
        // Sprint 374 (Gate D.3): programs may exceed 128 instructions; the wide PC
        // (0xFFFF mask) addresses up to 65,536, and the compiler emits `.W` branches.
        let mut iss = V2Iss::from_program(&words);
        iss.enable_wide_pc();
        for _ in 0..2_000_000 {
            if iss.state().halted {
                return iss.state().regs[0];
            }
            iss.step();
        }
        panic!("compiled program did not halt within the step budget");
    }

    fn func(name: &str, params: &[&str], body: Vec<Stmt>) -> Func {
        Func {
            name: name.to_string(),
            params: params.iter().map(|s| s.to_string()).collect(),
            body,
        }
    }

    /// `fact(n) = n < 2 ? 1 : n * fact(n-1)`; `main = fact(5) = 120`. Recursion.
    fn factorial_program() -> Program {
        let fact = func(
            "fact",
            &["n"],
            vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(int(1))],
                vec![Stmt::Return(mul(
                    var("n"),
                    call("fact", vec![sub(var("n"), int(1))]),
                ))],
            )],
        );
        let main = func("main", &[], vec![Stmt::Return(call("fact", vec![int(5)]))]);
        Program::new("main", vec![main, fact])
    }

    #[test]
    fn test_v2_compiler_factorial_recursive_iss() {
        assert_eq!(run_compiled(&factorial_program()), 120);
    }

    /// `fib(n) = n < 2 ? n : fib(n-1) + fib(n-2)`; `main = fib(10) = 55`. Double
    /// recursion stresses argument staging / the spill discipline.
    #[test]
    fn test_v2_compiler_fib_recursive_iss() {
        let fib = func(
            "fib",
            &["n"],
            vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(var("n"))],
                vec![Stmt::Return(add(
                    call("fib", vec![sub(var("n"), int(1))]),
                    call("fib", vec![sub(var("n"), int(2))]),
                ))],
            )],
        );
        let main = func("main", &[], vec![Stmt::Return(call("fib", vec![int(10)]))]);
        assert_eq!(run_compiled(&Program::new("main", vec![main, fib])), 55);
    }

    /// Iterative `sum(1..=10) = 55` — exercises `let`, `while`, `assign`, `<=`.
    #[test]
    fn test_v2_compiler_while_sum_iss() {
        let main = func(
            "main",
            &[],
            vec![
                Stmt::Let("i".into(), int(1)),
                Stmt::Let("s".into(), int(0)),
                Stmt::While(
                    le(var("i"), int(10)),
                    vec![
                        Stmt::Assign("s".into(), add(var("s"), var("i"))),
                        Stmt::Assign("i".into(), add(var("i"), int(1))),
                    ],
                ),
                Stmt::Return(var("s")),
            ],
        );
        assert_eq!(run_compiled(&Program::new("main", vec![main])), 55);
    }

    /// Subtraction-based `gcd(48, 36) = 12` — `while`, `if/else`, `!=`, `>`, `-`,
    /// and reassigning parameters.
    #[test]
    fn test_v2_compiler_gcd_iss() {
        let gcd = func(
            "gcd",
            &["a", "b"],
            vec![
                Stmt::While(
                    ne(var("a"), var("b")),
                    vec![Stmt::If(
                        gt(var("a"), var("b")),
                        vec![Stmt::Assign("a".into(), sub(var("a"), var("b")))],
                        vec![Stmt::Assign("b".into(), sub(var("b"), var("a")))],
                    )],
                ),
                Stmt::Return(var("a")),
            ],
        );
        let main = func(
            "main",
            &[],
            vec![Stmt::Return(call("gcd", vec![int(48), int(36)]))],
        );
        assert_eq!(run_compiled(&Program::new("main", vec![main, gcd])), 12);
    }

    /// Multi-arg call + arithmetic: `add3(a,b,c) = a+b+c`, `main = add3(10,20,12) = 42`.
    #[test]
    fn test_v2_compiler_multi_arg_call_iss() {
        let add3 = func(
            "add3",
            &["a", "b", "c"],
            vec![Stmt::Return(add(add(var("a"), var("b")), var("c")))],
        );
        let main = func(
            "main",
            &[],
            vec![Stmt::Return(call("add3", vec![int(10), int(20), int(12)]))],
        );
        assert_eq!(run_compiled(&Program::new("main", vec![main, add3])), 42);
    }

    /// The compiler emits human-readable assembly that round-trips through the
    /// assembler — the "port software vs hand-asm" artifact.
    #[test]
    fn test_v2_compiler_emits_readable_asm() {
        let add2 = func(
            "add2",
            &["a", "b"],
            vec![Stmt::Return(add(var("a"), var("b")))],
        );
        let main = func(
            "main",
            &[],
            vec![Stmt::Return(call("add2", vec![int(3), int(4)]))],
        );
        let program = Program::new("main", vec![main, add2]);
        let asm = compile_program(&program).expect("compile");
        // Spot-check the shape: entry stub, the JMPR-based return idiom, a label.
        assert!(asm.contains("CALL.W main"), "asm:\n{asm}");
        assert!(asm.contains("HALT"), "asm:\n{asm}");
        assert!(asm.contains("MFLR R5"), "asm:\n{asm}");
        assert!(asm.contains("JMPR R5"), "asm:\n{asm}");
        assert!(asm.contains("add2:"), "asm:\n{asm}");
        // And it actually computes: add2(3,4) = 7.
        assert_eq!(run_compiled(&program), 7);
    }

    /// A program that exceeds the 128-instruction window: `fact(5) + fib(10) +
    /// gcd(48,36) = 187`. The later functions sit above address 127, reached by wide
    /// calls; their internal loops/ifs use wide conditional branches. Runs on the ISS
    /// under the wide-PC config.
    #[test]
    fn test_v2_compiler_large_program_iss() {
        let fact = func(
            "fact",
            &["n"],
            vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(int(1))],
                vec![Stmt::Return(mul(
                    var("n"),
                    call("fact", vec![sub(var("n"), int(1))]),
                ))],
            )],
        );
        let fib = func(
            "fib",
            &["n"],
            vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(var("n"))],
                vec![Stmt::Return(add(
                    call("fib", vec![sub(var("n"), int(1))]),
                    call("fib", vec![sub(var("n"), int(2))]),
                ))],
            )],
        );
        let gcd = func(
            "gcd",
            &["a", "b"],
            vec![
                Stmt::While(
                    ne(var("a"), var("b")),
                    vec![Stmt::If(
                        gt(var("a"), var("b")),
                        vec![Stmt::Assign("a".into(), sub(var("a"), var("b")))],
                        vec![Stmt::Assign("b".into(), sub(var("b"), var("a")))],
                    )],
                ),
                Stmt::Return(var("a")),
            ],
        );
        let main = func(
            "main",
            &[],
            vec![Stmt::Return(add(
                add(call("fact", vec![int(5)]), call("fib", vec![int(10)])),
                call("gcd", vec![int(48), int(36)]),
            ))],
        );
        let program = Program::new("main", vec![main, fact, fib, gcd]);
        // The program must actually exceed the old 128-word ceiling.
        let words = compile_to_words(&program).expect("compile failed");
        assert!(
            words.len() > 128,
            "scale test must exceed 128 words (got {})",
            words.len()
        );
        assert_eq!(run_compiled(&program), 187);
    }

    /// Sprint 382 (Gate F): global array fill + read. `a[i] = i*i` for i in 0..5, then
    /// sum `a[0..5]` = 0+1+4+9+16 = 30. Exercises `StoreIndex`, `Index`, the entry-stub
    /// zero-init, and base+offset addressing.
    #[test]
    fn test_v2_compiler_array_fill_sum_iss() {
        let idx = |e: Expr| Expr::Index("a".into(), Box::new(e));
        let main = func(
            "main",
            &[],
            vec![
                Stmt::Let("i".into(), int(0)),
                Stmt::Let("s".into(), int(0)),
                Stmt::While(
                    lt(var("i"), int(5)),
                    vec![
                        Stmt::StoreIndex("a".into(), var("i"), mul(var("i"), var("i"))),
                        Stmt::Assign("i".into(), add(var("i"), int(1))),
                    ],
                ),
                Stmt::Let("k".into(), int(0)),
                Stmt::While(
                    lt(var("k"), int(5)),
                    vec![
                        Stmt::Assign("s".into(), add(var("s"), idx(var("k")))),
                        Stmt::Assign("k".into(), add(var("k"), int(1))),
                    ],
                ),
                Stmt::Return(var("s")),
            ],
        );
        let prog = Program::with_globals(
            "main",
            vec![GlobalArray {
                name: "a".into(),
                len: 5,
                init: vec![],
            }],
            vec![main],
        );
        assert_eq!(run_compiled(&prog), 30);
    }

    /// Initializer values survive into RAM and read back. `data = {11, 22, 33}`; return
    /// `data[0] + data[2]` = 44.
    #[test]
    fn test_v2_compiler_array_initializer_iss() {
        let idx = |i: i64| Expr::Index("data".into(), Box::new(int(i)));
        let main = func("main", &[], vec![Stmt::Return(add(idx(0), idx(2)))]);
        let prog = Program::with_globals(
            "main",
            vec![GlobalArray {
                name: "data".into(),
                len: 3,
                init: vec![11, 22, 33],
            }],
            vec![main],
        );
        assert_eq!(run_compiled(&prog), 44);
    }

    /// Error surfaces: undeclared variable, missing entry, too many params/args.
    #[test]
    fn test_v2_compiler_error_reporting() {
        let bad_var = func("main", &[], vec![Stmt::Return(var("nope"))]);
        assert!(compile_program(&Program::new("main", vec![bad_var])).is_err());

        let no_entry = func("main", &[], vec![Stmt::Return(int(0))]);
        assert!(compile_program(&Program::new("start", vec![no_entry])).is_err());

        let five_params = func("f", &["a", "b", "c", "d", "e"], vec![Stmt::Return(int(0))]);
        let m = func("main", &[], vec![Stmt::Return(int(0))]);
        assert!(compile_program(&Program::new("main", vec![m, five_params])).is_err());
    }
}
