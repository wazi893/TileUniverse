//! Sequential HLS (Sprint 383, Phase 3): scheduling + binding + FSM generation.
//!
//! The combinational layer (`synth::hls`) stops at straight-line expressions. This
//! module lifts HLS to functions with **state and control flow** — `if`, `while`,
//! reassignment — the part that makes it real high-level synthesis. The classical
//! pipeline, on our IR:
//!
//! 1. **CDFG** — the `Func` body is lowered to a control/data-flow graph of basic
//!    blocks ([`Cdfg`]); each block is a list of assignments plus a terminator
//!    (jump / conditional branch / return).
//! 2. **Scheduling** — one basic block per control step (an ASAP schedule with
//!    operator chaining inside the step: every op in a block executes in the same
//!    cycle, combinationally). Resource-constrained list scheduling is the planned
//!    refinement.
//! 3. **Binding** — every source variable (parameter or local) binds to one
//!    `width`-bit register; the FSM *reuses the same physical operators across
//!    cycles* (the loop body's adder is one piece of fabric, exercised every
//!    iteration — temporal hardware reuse, the thing a one-shot combinational
//!    unroll cannot do). Cross-state functional-unit sharing is the planned
//!    refinement.
//! 4. **Datapath + FSM generation** — one [`Aig`] holds the datapath *and* the
//!    controller: a state register (FSM) selects, via mux chains, each register's
//!    next value and the next state. Conditions become comparator circuits feeding
//!    the next-state mux.
//!
//! The generated logic is pure combinational next-state/next-register logic; state
//! lives in software between ticks, exactly the feedback pattern of
//! [`synth::sequential`](crate::synth::sequential). The combinational cone is
//! compiled through the **bridge** (multi-layer place & route) into a real tile
//! grid, and every tick is a masked propagation of real tiles — physical authority
//! holds for the sequential layer too.
//!
//! # Semantics
//!
//! Matches the CPU compiler so the two targets agree: `width`-bit wraparound
//! arithmetic (the CPU wraps at 64 bits, so values that fit the datapath width
//! agree exactly) and **unsigned** comparisons (the CPU compiles `<` to `CMP` +
//! carry, i.e. unsigned borrow).
//!
//! Not supported here: `Call` (needs an inter-procedural FSM), `Index` /
//! `StoreIndex` (need memory ports). They return [`HlsError::Unsupported`].
//!
//! # Known capacity boundary (Sprint 383 finding)
//!
//! Place & route is only *validated* up to roughly 200 mapped gates. FSM
//! netlists in the ~150–300 gate range can produce an export that mis-evaluates
//! a single output bit for some input patterns (reproducer: the `sum_to` FSM at
//! width 5, unoptimized — the mapped netlist is correct, the placed grid is
//! not). The defense is exact rather than statistical: every run-time FSM
//! transition is **audited against the AIG spec**; on a divergence the datapath
//! recompiles at the next capacity config and restarts, and only an exhausted
//! config ladder is an error. Silent corruption is therefore impossible — a
//! completed run is bit-for-bit the AIG's behavior, executed by tiles.
//!
//! The router defect behind that finding was fixed (layer-transition
//! exclusivity + post-route validation in `synth::routing`; see
//! `examples/router_bug_hunt.rs` for the diagnostic). Post-fix envelope
//! (`examples/seq_width_probe.rs`): width 6 loop kernels fully verified, width 8
//! verified for moderate cones (`sum_to`); the largest width-8 cones (`gcd`) hit
//! a loud routing-capacity failure — a quality-of-results limit, not a
//! correctness one. The run-time audit remains as defense in depth.

use std::collections::HashMap;

use crate::synth::aig::{Aig, AigLit};
use crate::synth::bridge::{BridgeConfig, compile_aig_to_export};
use crate::synth::cell_lib::CellLibrary;
use crate::synth::export::{SynthExport, evaluate_exported};
use crate::synth::hls::{HlsConfig, HlsError, Lowering, check_width};
use tileuniverse_v2_lang::{CmpOp, Cond, Expr, Func, Stmt};

// ===========================================================================
// 1. CDFG — control/data-flow graph
// ===========================================================================

/// A basic block: straight-line assignments plus one terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// `(variable, expression)` in program order. Later assignments in the same
    /// block see earlier ones (operator chaining within the control step).
    pub assigns: Vec<(String, Expr)>,
    /// What happens at the end of the cycle.
    pub term: Terminator,
}

/// Block terminator: where control goes on the next cycle.
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Unconditional transfer to a block.
    Jump(usize),
    /// `if cond` (evaluated on this block's post-assignment values) `then .0 else .1`.
    Branch(Cond, usize, usize),
    /// Function result; control transfers to the DONE state.
    Return(Expr),
}

/// The control/data-flow graph of a function body. Block 0 is the entry.
#[derive(Clone, Debug)]
pub struct Cdfg {
    pub blocks: Vec<BasicBlock>,
}

/// Internal: a block being built (terminator not yet decided).
struct DraftBlock {
    assigns: Vec<(String, Expr)>,
    term: Option<Terminator>,
}

struct CdfgBuilder {
    blocks: Vec<DraftBlock>,
}

impl CdfgBuilder {
    fn new_block(&mut self) -> usize {
        self.blocks.push(DraftBlock {
            assigns: Vec::new(),
            term: None,
        });
        self.blocks.len() - 1
    }

    /// Lower a statement list starting in block `cur`. Returns the open block in
    /// which control continues, or `None` if every path ended in `return`.
    fn lower(&mut self, stmts: &[Stmt], mut cur: usize) -> Result<Option<usize>, HlsError> {
        for stmt in stmts {
            match stmt {
                Stmt::Let(name, e) | Stmt::Assign(name, e) => {
                    self.blocks[cur].assigns.push((name.clone(), e.clone()));
                }
                Stmt::Return(e) => {
                    self.blocks[cur].term = Some(Terminator::Return(e.clone()));
                    return Ok(None);
                }
                Stmt::While(cond, body) => {
                    // cur -> header; header branches into the body or past the loop.
                    // The header carries no assignments, so the condition reads the
                    // registers as committed by the previous cycle.
                    let header = self.new_block();
                    self.blocks[cur].term = Some(Terminator::Jump(header));
                    let body_entry = self.new_block();
                    let exit = self.new_block();
                    self.blocks[header].term =
                        Some(Terminator::Branch(cond.clone(), body_entry, exit));
                    if let Some(body_end) = self.lower(body, body_entry)? {
                        self.blocks[body_end].term = Some(Terminator::Jump(header));
                    }
                    cur = exit;
                }
                Stmt::If(cond, then_stmts, else_stmts) => {
                    // The branch is cur's terminator: the condition chains onto this
                    // block's assignments (post-assignment values).
                    let then_entry = self.new_block();
                    let else_entry = self.new_block();
                    self.blocks[cur].term =
                        Some(Terminator::Branch(cond.clone(), then_entry, else_entry));
                    let t_end = self.lower(then_stmts, then_entry)?;
                    let e_end = self.lower(else_stmts, else_entry)?;
                    match (t_end, e_end) {
                        (None, None) => return Ok(None),
                        _ => {
                            let merge = self.new_block();
                            if let Some(t) = t_end {
                                self.blocks[t].term = Some(Terminator::Jump(merge));
                            }
                            if let Some(e) = e_end {
                                self.blocks[e].term = Some(Terminator::Jump(merge));
                            }
                            cur = merge;
                        }
                    }
                }
                Stmt::StoreIndex(..) => {
                    return Err(HlsError::Unsupported("array store (needs memory ports)"));
                }
                Stmt::Expr(_) => {
                    return Err(HlsError::Unsupported(
                        "statement expression (calls need an inter-procedural FSM)",
                    ));
                }
            }
        }
        Ok(Some(cur))
    }
}

/// Build the CDFG for a function body.
///
/// Every control path must end in `return` (a fall-off-the-end path is
/// [`HlsError::NoReturn`]).
pub fn build_cdfg(body: &[Stmt]) -> Result<Cdfg, HlsError> {
    let mut b = CdfgBuilder { blocks: Vec::new() };
    let entry = b.new_block();
    if b.lower(body, entry)?.is_some() {
        return Err(HlsError::NoReturn);
    }
    let blocks = b
        .blocks
        .into_iter()
        .map(|d| {
            d.term
                .map(|term| BasicBlock {
                    assigns: d.assigns,
                    term,
                })
                .ok_or(HlsError::NoReturn)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Cdfg { blocks })
}

// ===========================================================================
// 2+3+4. Schedule + bind + generate the FSM/datapath AIG
// ===========================================================================

/// Unsigned comparator over two LSB-first buses. Matches the CPU target's
/// `CMP`-and-carry semantics for values that fit the datapath width.
fn cmp_bus(aig: &mut Aig, op: CmpOp, a: &[AigLit], b: &[AigLit]) -> AigLit {
    // a < b  (unsigned)  =  NOT carry-out of (a + ~b + 1).
    fn ltu(aig: &mut Aig, a: &[AigLit], b: &[AigLit]) -> AigLit {
        let mut carry = AigLit::TRUE;
        for k in 0..a.len() {
            let nb = aig.not(b[k]);
            let axb = aig.xor(a[k], nb);
            let ab = aig.and(a[k], nb);
            let caxb = aig.and(carry, axb);
            carry = aig.or(ab, caxb);
        }
        aig.not(carry)
    }
    fn equ(aig: &mut Aig, a: &[AigLit], b: &[AigLit]) -> AigLit {
        let mut ne = AigLit::FALSE;
        for k in 0..a.len() {
            let d = aig.xor(a[k], b[k]);
            ne = aig.or(ne, d);
        }
        aig.not(ne)
    }
    match op {
        CmpOp::Eq => equ(aig, a, b),
        CmpOp::Ne => {
            let e = equ(aig, a, b);
            aig.not(e)
        }
        CmpOp::Lt => ltu(aig, a, b),
        CmpOp::Gt => ltu(aig, b, a),
        CmpOp::Ge => {
            let lt = ltu(aig, a, b);
            aig.not(lt)
        }
        CmpOp::Le => {
            let gt = ltu(aig, b, a);
            aig.not(gt)
        }
    }
}

/// Collect every register-bound variable: parameters first (in declaration
/// order), then each `let`-introduced local at first appearance.
fn collect_vars(func: &Func) -> Vec<String> {
    fn walk(stmts: &[Stmt], seen: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Let(name, _) => {
                    if !seen.iter().any(|n| n == name) {
                        seen.push(name.clone());
                    }
                }
                Stmt::If(_, t, e) => {
                    walk(t, seen);
                    walk(e, seen);
                }
                Stmt::While(_, b) => walk(b, seen),
                _ => {}
            }
        }
    }
    let mut vars = func.params.clone();
    walk(&func.body, &mut vars);
    vars
}

/// Summary of the synthesized controller + datapath (the scheduling/binding
/// read-out, alongside the AIG's own `stats()`).
#[derive(Clone, Copy, Debug)]
pub struct SeqStats {
    /// FSM states = basic blocks + the DONE state. One control step each.
    pub num_states: usize,
    /// Bound registers: one per source variable, plus the return register.
    pub num_registers: usize,
    /// Total state bits (FSM bits + registers × width).
    pub num_state_bits: usize,
    /// FSM state-register width in bits.
    pub fsm_bits: usize,
}

/// A sequential datapath: FSM + registers, with the combinational next-state
/// logic compiled into a live tile grid.
///
/// State (FSM + registers) lives in software between ticks; every tick drives the
/// current state into the export's tile grid, propagates real tiles, and reads
/// back the next state — the same feedback pattern as
/// [`SequentialCircuit`](crate::synth::sequential::SequentialCircuit), built on
/// the bridge's multi-layer place & route for capacity.
pub struct SeqDatapath {
    export: SynthExport,
    /// Current state bits, LSB-first: `[fsm][ret][vars...]`.
    state: Vec<bool>,
    width: usize,
    fsm_bits: usize,
    num_params: usize,
    stats: SeqStats,
    aig: Aig,
    /// Index into [`SEQ_CONFIGS`] of the next capacity config to try if the
    /// current placement diverges from the AIG at run time.
    next_config: usize,
}

/// Outcome of running a [`SeqDatapath`] to completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqRunOutcome {
    /// The function result (masked to the datapath width).
    pub result: u64,
    /// Control steps executed (FSM transitions until DONE was observed).
    pub cycles: u64,
}

impl SeqDatapath {
    /// Scheduling/binding summary.
    pub fn stats(&self) -> SeqStats {
        self.stats
    }

    /// The generated controller+datapath AIG (for reporting, e.g. `aig.stats()`).
    pub fn aig(&self) -> &Aig {
        &self.aig
    }

    /// Placed grid footprint `(width, height, layers)` of the live tile export.
    pub fn grid_info(&self) -> (usize, usize, usize) {
        (
            self.export.sim.width(),
            self.export.sim.height(),
            self.export.sim.num_layers(),
        )
    }

    /// Run the FSM from the initial state with the given parameter values until
    /// the DONE state is reached (or `max_cycles` ticks pass).
    ///
    /// Every tick evaluates the next-state logic **as real tiles**, and every
    /// transition is audited against the AIG spec (the hard backstop for the
    /// place&route capacity boundary — see the module docs). On a divergence the
    /// datapath recompiles at the next capacity config and restarts the run from
    /// the initial state, so a completed run always executed entirely on one
    /// fabric. Only when every config has diverged does it return `Err`.
    pub fn run(&mut self, args: &[u64], max_cycles: u64) -> Result<SeqRunOutcome, HlsError> {
        if args.len() != self.num_params {
            return Err(HlsError::Backend(format!(
                "expected {} arguments, got {}",
                self.num_params,
                args.len()
            )));
        }
        loop {
            match self.run_once(args, max_cycles) {
                Ok(out) => return Ok(out),
                Err(RunFail::Timeout) => {
                    return Err(HlsError::Backend(format!(
                        "FSM did not reach DONE within {max_cycles} cycles"
                    )));
                }
                Err(RunFail::Diverged(cycle)) => {
                    if !self.escalate()? {
                        return Err(HlsError::Backend(format!(
                            "placed datapath diverged from its AIG at cycle {cycle} and \
                             every capacity config is exhausted — place&route defect \
                             past the validated envelope"
                        )));
                    }
                }
            }
        }
    }

    /// One complete tile-fabric run attempt on the current placement.
    fn run_once(&mut self, args: &[u64], max_cycles: u64) -> Result<SeqRunOutcome, RunFail> {
        let w = self.width;
        let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };

        // Initial state: FSM = entry block (0), ret = 0, params = args, locals = 0.
        for b in self.state.iter_mut() {
            *b = false;
        }
        for (p, &val) in args.iter().enumerate() {
            let base = self.fsm_bits + w + p * w;
            for k in 0..w {
                self.state[base + k] = (val & mask) >> k & 1 == 1;
            }
        }

        for cycle in 0..max_cycles {
            let outs = evaluate_exported(&mut self.export, &self.state);

            // Audit the tile transition against the AIG spec: a divergence is a
            // loud, recoverable event, never a silently wrong answer.
            let spec = crate::synth::mapping::evaluate_aig(&self.aig, &self.state);
            if spec != outs {
                return Err(RunFail::Diverged(cycle));
            }

            // Output order: [next_state (num_state_bits)] [done] [result (width)].
            let n = self.state.len();
            if outs[n] {
                let result = outs[n + 1..n + 1 + w]
                    .iter()
                    .enumerate()
                    .fold(0u64, |acc, (k, &b)| acc | ((b as u64) << k));
                return Ok(SeqRunOutcome {
                    result,
                    cycles: cycle,
                });
            }
            self.state.copy_from_slice(&outs[..n]);
        }
        Err(RunFail::Timeout)
    }

    /// Recompile the datapath at the next capacity config. Returns `Ok(false)`
    /// when the ladder is exhausted.
    fn escalate(&mut self) -> Result<bool, HlsError> {
        let lib = CellLibrary::tile_native();
        while self.next_config < SEQ_CONFIGS.len() {
            let c = &SEQ_CONFIGS[self.next_config];
            self.next_config += 1;
            if let Ok(export) = compile_aig_to_export(&self.aig, &lib, c) {
                self.export = export;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Internal outcome of one run attempt on a single placement.
enum RunFail {
    /// The placed grid disagreed with the AIG at this cycle.
    Diverged(u64),
    /// DONE was never reached within the cycle budget.
    Timeout,
}

/// Generate the combinational controller+datapath AIG for a function.
///
/// Inputs (state feedback, LSB-first): FSM bits, the return register, then one
/// `width`-bit register per variable. Outputs: the next value of every state bit,
/// a `done` flag (FSM == DONE), and the `result` bus (the return register).
pub fn build_seq_aig(func: &Func, cfg: &HlsConfig) -> Result<(Aig, SeqStats), HlsError> {
    let width = check_width(cfg.width)?;
    let cdfg = build_cdfg(&func.body)?;
    let vars = collect_vars(func);

    let num_states = cdfg.blocks.len() + 1; // + DONE
    let done_state = cdfg.blocks.len();
    let fsm_bits = usize::max(1, num_states.next_power_of_two().trailing_zeros() as usize);

    let mut aig = Aig::new();
    let fsm: Vec<AigLit> = aig.add_input_bus("fsm", fsm_bits as u32);
    let ret: Vec<AigLit> = aig.add_input_bus("ret", width as u32);
    let mut reg_of_var: HashMap<String, Vec<AigLit>> = HashMap::new();
    let mut var_order: Vec<Vec<AigLit>> = Vec::with_capacity(vars.len());
    for v in &vars {
        let bus = aig.add_input_bus(&format!("reg_{v}"), width as u32);
        reg_of_var.insert(v.clone(), bus.clone());
        var_order.push(bus);
    }

    // One-hot state selectors.
    let select: Vec<AigLit> = (0..num_states)
        .map(|s| {
            let mut sel = AigLit::TRUE;
            for (k, &fbit) in fsm.iter().enumerate() {
                let want = (s >> k) & 1 == 1;
                let term = if want { fbit } else { aig.not(fbit) };
                sel = aig.and(sel, term);
            }
            sel
        })
        .collect();

    // Per-state contributions: next-register values and next-FSM encodings.
    // Defaults (hold register / stay in state) are the mux-chain initial values.
    let mut next_fsm: Vec<AigLit> = fsm.clone();
    let mut next_ret: Vec<AigLit> = ret.clone();
    let mut next_vars: Vec<Vec<AigLit>> = var_order.clone();

    // Constant bit `k` of state encoding `s`.
    fn enc(s: usize, k: usize) -> AigLit {
        if (s >> k) & 1 == 1 {
            AigLit::TRUE
        } else {
            AigLit::FALSE
        }
    }

    for (b, block) in cdfg.blocks.iter().enumerate() {
        let sel = select[b];

        // Thread the block's assignments through the combinational lowering: each
        // assignment sees the chained values of earlier ones (same control step).
        let mut lo = Lowering {
            aig: &mut aig,
            env: reg_of_var
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            width,
        };
        let mut assigned: Vec<(String, Vec<AigLit>)> = Vec::new();
        for (name, e) in &block.assigns {
            let bus = lo.lower(e)?;
            lo.env.insert(name.clone(), bus.clone());
            // Last assignment to a name within the block wins.
            if let Some(slot) = assigned.iter_mut().find(|(n, _)| n == name) {
                slot.1 = bus;
            } else {
                assigned.push((name.clone(), bus));
            }
        }

        // Terminator: next FSM state (and possibly the return value), evaluated in
        // the post-assignment environment.
        let (target_bits, ret_bus): (Vec<AigLit>, Option<Vec<AigLit>>) = match &block.term {
            Terminator::Jump(t) => ((0..fsm_bits).map(|k| enc(*t, k)).collect(), None),
            Terminator::Branch(Cond::Cmp(op, lhs, rhs), t, f) => {
                let a = lo.lower(lhs)?;
                let bbus = lo.lower(rhs)?;
                let c = cmp_bus(lo.aig, *op, &a, &bbus);
                let bits = (0..fsm_bits)
                    .map(|k| lo.aig.mux(c, enc(*t, k), enc(*f, k)))
                    .collect();
                (bits, None)
            }
            Terminator::Return(e) => {
                let bus = lo.lower(e)?;
                (
                    (0..fsm_bits).map(|k| enc(done_state, k)).collect(),
                    Some(bus),
                )
            }
        };

        // Merge this state's contributions into the global mux chains.
        for k in 0..fsm_bits {
            next_fsm[k] = aig.mux(sel, target_bits[k], next_fsm[k]);
        }
        if let Some(bus) = ret_bus {
            for k in 0..width {
                next_ret[k] = aig.mux(sel, bus[k], next_ret[k]);
            }
        }
        for (name, bus) in assigned {
            let vi = vars
                .iter()
                .position(|v| *v == name)
                .ok_or_else(|| HlsError::UnknownVar(name.clone()))?;
            for k in 0..width {
                next_vars[vi][k] = aig.mux(sel, bus[k], next_vars[vi][k]);
            }
        }
    }

    // Outputs: next state bits first (mirroring the input layout), then done + result.
    aig.add_output_bus("next_fsm", &next_fsm);
    aig.add_output_bus("next_ret", &next_ret);
    for (i, bus) in next_vars.iter().enumerate() {
        aig.add_output_bus(&format!("next_reg_{}", vars[i]), bus);
    }
    aig.add_output("done", select[done_state]);
    aig.add_output_bus("result", &ret);

    let stats = SeqStats {
        num_states,
        num_registers: vars.len() + 1,
        num_state_bits: fsm_bits + (vars.len() + 1) * width,
        fsm_bits,
    };
    Ok((aig, stats))
}

/// The escalating place & route capacity ladder for FSM cones. They are dense
/// (mux chains + comparators), so two deviations from the combinational
/// defaults: start at the capacity 8-bit datapaths already need, and ALWAYS
/// optimize the AIG before mapping (fewer gates and shorter routes keep the
/// cone nearer the validated envelope).
const SEQ_CONFIGS: [BridgeConfig; 3] = [
    BridgeConfig {
        halo: 6,
        max_z: 3,
        no_crossings: true,
        optimize: true,
    },
    BridgeConfig {
        halo: 8,
        max_z: 3,
        no_crossings: true,
        optimize: true,
    },
    BridgeConfig {
        halo: 10,
        max_z: 3,
        no_crossings: true,
        optimize: true,
    },
];

/// Synthesize a function with control flow into a running FSM + datapath on a
/// fresh tile grid.
///
/// Pipeline: CDFG → schedule (block per control step) → bind (register per
/// variable) → AIG → place & route (multi-layer, escalating capacity) → live
/// export grid. Every run-time transition is audited against the AIG; on a
/// divergence the datapath transparently recompiles at the next capacity config
/// and restarts (see [`SeqDatapath::run`]).
pub fn synthesize_seq_func(func: &Func, cfg: &HlsConfig) -> Result<SeqDatapath, HlsError> {
    let width = check_width(cfg.width)?;
    let (aig, stats) = build_seq_aig(func, cfg)?;

    let lib = CellLibrary::tile_native();
    let mut last_err = String::from("no placement attempted");
    let mut compiled = None;
    let mut next_config = SEQ_CONFIGS.len();
    for (i, c) in SEQ_CONFIGS.iter().enumerate() {
        match compile_aig_to_export(&aig, &lib, c) {
            Ok(export) => {
                compiled = Some(export);
                next_config = i + 1;
                break;
            }
            Err(e) => last_err = e.to_string(),
        }
    }
    let export = compiled.ok_or(HlsError::Backend(last_err))?;

    Ok(SeqDatapath {
        export,
        state: vec![false; stats.num_state_bits],
        width,
        fsm_bits: stats.fsm_bits,
        num_params: func.params.len(),
        stats,
        aig,
        next_config,
    })
}

// ===========================================================================
// Reference evaluator (the spec the FSM must satisfy)
// ===========================================================================

/// Reference interpreter for the sequential subset: `width`-bit wraparound
/// arithmetic, unsigned comparisons. The spec [`SeqDatapath`] must match.
pub fn eval_func_ref(func: &Func, args: &[u64], width: u32) -> Result<u64, HlsError> {
    check_width(width)?;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if args.len() != func.params.len() {
        return Err(HlsError::Backend(format!(
            "expected {} arguments, got {}",
            func.params.len(),
            args.len()
        )));
    }
    let mut env: HashMap<String, u64> = func
        .params
        .iter()
        .cloned()
        .zip(args.iter().map(|&a| a & mask))
        .collect();

    fn cond(c: &Cond, env: &HashMap<String, u64>, mask: u64) -> Result<bool, HlsError> {
        let Cond::Cmp(op, a, b) = c;
        let x = crate::synth::hls::eval_expr_ref(a, env, mask.count_ones())?;
        let y = crate::synth::hls::eval_expr_ref(b, env, mask.count_ones())?;
        Ok(match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
        })
    }

    // Iteration budget so a non-terminating loop errors instead of hanging.
    let mut fuel: u64 = 1_000_000;

    fn exec(
        stmts: &[Stmt],
        env: &mut HashMap<String, u64>,
        mask: u64,
        fuel: &mut u64,
    ) -> Result<Option<u64>, HlsError> {
        let width = mask.count_ones();
        for s in stmts {
            if *fuel == 0 {
                return Err(HlsError::Backend(
                    "reference evaluation exceeded its iteration budget".into(),
                ));
            }
            *fuel -= 1;
            match s {
                Stmt::Let(n, e) | Stmt::Assign(n, e) => {
                    let v = crate::synth::hls::eval_expr_ref(e, env, width)?;
                    env.insert(n.clone(), v);
                }
                Stmt::Return(e) => {
                    return Ok(Some(crate::synth::hls::eval_expr_ref(e, env, width)?));
                }
                Stmt::If(c, t, els) => {
                    let taken = cond(c, env, mask)?;
                    let branch = if taken { t } else { els };
                    if let Some(r) = exec(branch, env, mask, fuel)? {
                        return Ok(Some(r));
                    }
                }
                Stmt::While(c, body) => {
                    while cond(c, env, mask)? {
                        if *fuel == 0 {
                            return Err(HlsError::Backend(
                                "reference evaluation exceeded its iteration budget".into(),
                            ));
                        }
                        *fuel -= 1;
                        if let Some(r) = exec(body, env, mask, fuel)? {
                            return Ok(Some(r));
                        }
                    }
                }
                Stmt::StoreIndex(..) => {
                    return Err(HlsError::Unsupported("array store (needs memory ports)"));
                }
                Stmt::Expr(_) => {
                    return Err(HlsError::Unsupported(
                        "statement expression (calls need an inter-procedural FSM)",
                    ));
                }
            }
        }
        Ok(None)
    }

    exec(&func.body, &mut env, mask, &mut fuel)?.ok_or(HlsError::NoReturn)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_iss::V2Iss;
    use crate::tile_cpu::v2_parser::{compile_source, parse_program};

    /// Parse a source string and pull out one function.
    fn func_of(src: &str, name: &str) -> Func {
        let program = parse_program(src).expect("parse");
        program
            .funcs
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("function `{name}` not found"))
            .clone()
    }

    const SUM_TO: &str = r#"
        int sum_to(int n) {
            int s = 0;
            int i = 1;
            while (i <= n) {
                s = s + i;
                i = i + 1;
            }
            return s;
        }
    "#;

    #[test]
    fn cdfg_sum_loop_shape() {
        let func = func_of(SUM_TO, "sum_to");
        let cdfg = build_cdfg(&func.body).expect("cdfg");
        // entry (s=0, i=1) -> header (branch) -> body (s,i; jump header) -> exit (return).
        assert_eq!(cdfg.blocks.len(), 4, "entry/header/body/exit");
        assert!(matches!(cdfg.blocks[0].term, Terminator::Jump(1)));
        assert!(matches!(cdfg.blocks[1].term, Terminator::Branch(_, 2, 3)));
        assert!(matches!(cdfg.blocks[2].term, Terminator::Jump(1)));
        assert!(matches!(cdfg.blocks[3].term, Terminator::Return(_)));
        assert_eq!(cdfg.blocks[0].assigns.len(), 2);
        assert_eq!(cdfg.blocks[2].assigns.len(), 2);
    }

    #[test]
    fn cdfg_missing_return_rejected() {
        let func = func_of("int f(int a) { int b = a + 1; return b; }", "f");
        assert!(build_cdfg(&func.body).is_ok());
        // A body with no return on the fall-through path.
        let no_ret = vec![Stmt::Let("x".into(), Expr::Int(1))];
        assert_eq!(build_cdfg(&no_ret).unwrap_err(), HlsError::NoReturn);
    }

    /// The FSM next-state logic, evaluated purely at the AIG level, must agree
    /// with the reference interpreter — software differential before the tiles.
    #[test]
    fn seq_aig_sum_loop_matches_reference_in_software() {
        use crate::synth::mapping::evaluate_aig;
        let func = func_of(SUM_TO, "sum_to");
        let width = 6u32;
        let w = width as usize;
        let (aig, stats) = build_seq_aig(&func, &HlsConfig { width }).expect("seq aig");

        for n in 0..=8u64 {
            // Software FSM run: state -> evaluate AIG -> next state.
            let mut state = vec![false; stats.num_state_bits];
            for k in 0..w {
                state[stats.fsm_bits + w + k] = (n >> k) & 1 == 1; // param n
            }
            let mut result = None;
            for _ in 0..200 {
                let outs = evaluate_aig(&aig, &state);
                let nbits = stats.num_state_bits;
                if outs[nbits] {
                    let r = outs[nbits + 1..nbits + 1 + w]
                        .iter()
                        .enumerate()
                        .fold(0u64, |acc, (k, &b)| acc | ((b as u64) << k));
                    result = Some(r);
                    break;
                }
                state.copy_from_slice(&outs[..nbits]);
            }
            let expected = eval_func_ref(&func, &[n], width).expect("ref");
            assert_eq!(result, Some(expected), "sum_to({n})");
        }
    }

    /// Physical authority for the sequential layer: the while-loop FSM runs as
    /// real tiles — the loop body's adder is ONE piece of fabric reused every
    /// iteration — and matches the reference for every n. (Width 5 keeps the
    /// optimized FSM cone inside the place&route correctness envelope; sum_to(7)
    /// = 28 still fits.)
    #[test]
    #[ignore = "slow (>100s tile sim); run via: cargo nextest run --profile nightly --run-ignored all"]
    fn seq_sum_loop_runs_as_real_tiles() {
        let func = func_of(SUM_TO, "sum_to");
        let width = 5u32;
        let mut dp = synthesize_seq_func(&func, &HlsConfig { width }).expect("seq synth");

        for n in [0u64, 1, 3, 5, 7] {
            let expected = eval_func_ref(&func, &[n], width).expect("ref");
            let out = dp.run(&[n], 200).expect("fsm run");
            assert_eq!(out.result, expected, "sum_to({n}) on tiles");
            // Multi-cycle reality check: 2 cycles per iteration + prologue.
            assert!(
                out.cycles >= 2 * n + 2,
                "sum_to({n}) finished suspiciously fast ({} cycles)",
                out.cycles
            );
        }
    }

    /// if/else datapath: |a - b| via a branch, on tiles.
    #[test]
    #[ignore = "slow (>100s tile sim); run via: cargo nextest run --profile nightly --run-ignored all"]
    fn seq_if_else_abs_diff_runs_as_real_tiles() {
        let src = r#"
            int absdiff(int a, int b) {
                int d = 0;
                if (a > b) { d = a - b; } else { d = b - a; }
                return d;
            }
        "#;
        let func = func_of(src, "absdiff");
        let width = 5u32;
        let mut dp = synthesize_seq_func(&func, &HlsConfig { width }).expect("seq synth");

        for (a, b) in [(0u64, 0u64), (9, 4), (4, 9), (31, 1), (7, 7)] {
            let expected = eval_func_ref(&func, &[a, b], width).expect("ref");
            let out = dp.run(&[a, b], 50).expect("fsm run");
            assert_eq!(out.result, expected, "absdiff({a},{b}) on tiles");
        }
    }

    /// THE gate for Phase 3: two targets, one source, identical answers.
    /// The same source string is (1) compiled to instructions and run on the V2
    /// ISS (the CPU reference) and (2) synthesized to an FSM+datapath and run as
    /// tiles. Results must be bit-identical for every input.
    #[test]
    #[ignore = "slow (>100s tile sim); run via: cargo nextest run --profile nightly --run-ignored all"]
    fn seq_two_targets_agree_differential() {
        // (function source, name, arg sets) — values chosen to fit the 5-bit
        // datapath so 64-bit CPU arithmetic and 5-bit FSM arithmetic coincide.
        // (Width 5 keeps the optimized FSM cones inside the place&route
        // correctness envelope; wider sum/gcd cones exceed it today.)
        let sum_args: &[&[u64]] = &[&[0], &[1], &[5], &[7]];
        let gcd_args: &[&[u64]] = &[&[12, 8], &[21, 14], &[13, 7], &[30, 12]];
        let cases: &[(&str, &str, &[&[u64]])] = &[
            (SUM_TO, "sum_to", sum_args),
            (
                r#"
                int gcd(int a, int b) {
                    while (a != b) {
                        if (a > b) { a = a - b; } else { b = b - a; }
                    }
                    return a;
                }
                "#,
                "gcd",
                gcd_args,
            ),
        ];

        let width = 5u32;
        for (src, name, arg_sets) in cases {
            let func = func_of(src, name);
            let mut dp = synthesize_seq_func(&func, &HlsConfig { width })
                .unwrap_or_else(|e| panic!("{name}: seq synth failed: {e}"));

            for args in *arg_sets {
                // Target 1: the CPU. Wrap the function in a main() with literal args.
                let call_args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                let full_src = format!(
                    "{src}\nint main() {{ return {name}({}); }}",
                    call_args.join(", ")
                );
                let words = compile_source(&full_src).expect("compile_source");
                let mut iss = V2Iss::from_program_with_mmio(&words);
                iss.enable_wide_pc();
                let mut cpu_r0 = None;
                for _ in 0..1_000_000 {
                    if iss.state().halted {
                        cpu_r0 = Some(iss.state().regs[0]);
                        break;
                    }
                    iss.step();
                }
                let cpu_r0 = cpu_r0.expect("ISS halted");

                // Target 2: the FSM datapath, on real tiles.
                let out = dp.run(args, 5_000).expect("fsm run");

                assert_eq!(
                    out.result, cpu_r0,
                    "{name}({args:?}): CPU target ({cpu_r0}) != HLS target ({})",
                    out.result
                );
            }
        }
    }

    #[test]
    fn seq_rejects_unsupported() {
        let func = func_of("int f(int n) { int s = g(n); return s; }", "f");
        assert!(matches!(
            build_seq_aig(&func, &HlsConfig { width: 4 }),
            Err(HlsError::Unsupported(_))
        ));
    }
}
