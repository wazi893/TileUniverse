//! High-Level Synthesis (HLS) — the behavioral "middle layer".
//!
//! This module is the bridge between the two big tracks of the engine:
//!
//! - the **front-end** (`tile_cpu::v2_compiler`) parses a C-like source into an
//!   [`Expr`]/[`Func`] AST and lowers it to *sequential CPU instructions*;
//! - the **back-end** (`synth::{aig, mapping, placement, routing, bridge}`) maps an
//!   [`Aig`] to *spatial tiles*, places, routes, and runs them as real logic.
//!
//! Classical HLS is the step in between: turning an algorithmic description into a
//! datapath rather than into a program. This module does exactly that for the
//! **combinational** subset of the language — it consumes the *same `Expr` AST the
//! CPU compiler uses* and bit-blasts it into a multi-bit [`Aig`] (one structural
//! gate per bit of arithmetic), which the existing back-end then synthesizes into
//! a tile circuit. The result is the unification: **one source AST, two targets** —
//! a soft-CPU program *or* spatially-synthesized logic.
//!
//! # Scope (v1: combinational datapath)
//!
//! Supported expressions over fixed-width integer inputs:
//! - constants, variable references, and `let`-bound intermediate signals;
//! - bitwise `&` `|` `^`;
//! - `+` (ripple-carry adder), `-` (two's-complement), `*` (shift-add array multiplier).
//!
//! Arithmetic is truncated to the configured datapath width (`u64`-style wraparound),
//! matching the CPU back-end's semantics so the two targets agree bit-for-bit.
//!
//! Not yet supported (these need scheduling / state / memory — the *sequential* HLS
//! extension): function `Call`, array `Index`, and control flow (`if`/`while`).
//! They return [`HlsError::Unsupported`].
//!
//! # Verification
//!
//! Every synthesized AIG is checked two ways in the tests: against a Rust reference
//! evaluator ([`eval_expr_ref`]) via the software AIG simulator (`mapping::evaluate_aig`),
//! and — for the physical-authority standard — by compiling the AIG into a real
//! [`Simulation`](crate::simulation::Simulation) grid and running it as tiles.

use std::collections::HashMap;

use crate::synth::aig::{Aig, AigLit};
use crate::synth::bridge::{BridgeConfig, compile_aig_to_export};
use crate::synth::cell_lib::CellLibrary;
use crate::synth::timing::{TimingConstraint, analyze_timing};
use crate::synth::{SynthConfig, materialize_inversions, synthesize};
use crate::tile_cpu::v2_compiler::{ArithOp, Expr, Func, Stmt};

/// Configuration for behavioral synthesis.
#[derive(Clone, Copy, Debug)]
pub struct HlsConfig {
    /// Datapath bit width. Every variable becomes a `width`-bit input bus and all
    /// arithmetic is truncated to `width` bits. Must be `1..=64`.
    pub width: u32,
}

impl Default for HlsConfig {
    fn default() -> Self {
        HlsConfig { width: 8 }
    }
}

/// Errors from behavioral synthesis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HlsError {
    /// The datapath width was 0 or greater than 64.
    BadWidth(u32),
    /// A variable was referenced that is neither an input nor a `let`-bound signal.
    UnknownVar(String),
    /// An expression or statement that the combinational v1 layer cannot synthesize
    /// (function call, array index, control flow). Carries a short reason.
    Unsupported(&'static str),
    /// The function body had no `return` statement.
    NoReturn,
    /// The synthesis back-end (map/place/route) failed while building a report.
    Backend(String),
}

impl std::fmt::Display for HlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HlsError::BadWidth(w) => write!(f, "datapath width {w} out of range (must be 1..=64)"),
            HlsError::UnknownVar(n) => write!(f, "unknown variable `{n}`"),
            HlsError::Unsupported(why) => write!(f, "unsupported by combinational HLS: {why}"),
            HlsError::NoReturn => write!(f, "function body has no `return`"),
            HlsError::Backend(e) => write!(f, "synthesis back-end failed: {e}"),
        }
    }
}

impl std::error::Error for HlsError {}

/// Synthesize a single combinational [`Expr`] into a one-output-bus [`Aig`].
///
/// `input_order` names the input buses in a deterministic order; each becomes a
/// `cfg.width`-bit primary input bus (LSB first). The result has a single output
/// bus named `"out"` of `cfg.width` bits.
///
/// # Example
/// ```ignore
/// let cfg = HlsConfig { width: 8 };
/// // a * b + c
/// let expr = Expr::Bin(ArithOp::Add,
///     Box::new(Expr::Bin(ArithOp::Mul, var("a"), var("b"))),
///     var("c"));
/// let aig = synthesize_expr(&expr, &["a", "b", "c"], &cfg)?;
/// ```
pub fn synthesize_expr(
    expr: &Expr,
    input_order: &[&str],
    cfg: &HlsConfig,
) -> Result<Aig, HlsError> {
    let width = check_width(cfg.width)?;
    let mut aig = Aig::new();
    let mut env: HashMap<String, Vec<AigLit>> = HashMap::new();
    for name in input_order {
        let bus = aig.add_input_bus(name, width as u32);
        env.insert((*name).to_string(), bus);
    }
    let out = {
        let mut lo = Lowering {
            aig: &mut aig,
            env,
            width,
        };
        lo.lower(expr)?
    };
    aig.add_output_bus("out", &out);
    Ok(aig)
}

/// Synthesize a straight-line combinational [`Func`] into an [`Aig`].
///
/// The parameters become input buses (in declaration order). The body may contain
/// any number of `let name = expr;` / `name = expr;` statements (each defining or
/// updating a combinational intermediate signal) followed by a single
/// `return expr;`. The returned expression drives a `"out"` output bus.
///
/// This is the same `Func` the CPU compiler consumes — so a straight-line function
/// can be compiled to instructions *or* synthesized to a tile circuit from one source.
pub fn synthesize_func(func: &Func, cfg: &HlsConfig) -> Result<Aig, HlsError> {
    let width = check_width(cfg.width)?;
    let mut aig = Aig::new();
    let mut env: HashMap<String, Vec<AigLit>> = HashMap::new();
    for p in &func.params {
        let bus = aig.add_input_bus(p, width as u32);
        env.insert(p.clone(), bus);
    }

    let mut out_bus: Option<Vec<AigLit>> = None;
    {
        let mut lo = Lowering {
            aig: &mut aig,
            env,
            width,
        };
        for stmt in &func.body {
            match stmt {
                Stmt::Let(name, e) | Stmt::Assign(name, e) => {
                    let bus = lo.lower(e)?;
                    lo.env.insert(name.clone(), bus);
                }
                Stmt::Return(e) => {
                    out_bus = Some(lo.lower(e)?);
                    break;
                }
                Stmt::If(..) | Stmt::While(..) => {
                    return Err(HlsError::Unsupported("control flow (needs an FSM)"));
                }
                Stmt::StoreIndex(..) => {
                    return Err(HlsError::Unsupported("array store (needs memory)"));
                }
                Stmt::Expr(_) => {
                    return Err(HlsError::Unsupported("statement expression (no effect)"));
                }
            }
        }
    }

    let out = out_bus.ok_or(HlsError::NoReturn)?;
    aig.add_output_bus("out", &out);
    Ok(aig)
}

pub(crate) fn check_width(width: u32) -> Result<usize, HlsError> {
    if width == 0 || width > 64 {
        Err(HlsError::BadWidth(width))
    } else {
        Ok(width as usize)
    }
}

/// Bit-blasting datapath lowering: `Expr` -> `Vec<AigLit>` (a `width`-bit bus, LSB first).
///
/// `pub(crate)` so the sequential HLS layer (`hls_seq`) reuses the same operator
/// lowering — both targets' datapath bits must come from one implementation.
pub(crate) struct Lowering<'a> {
    pub(crate) aig: &'a mut Aig,
    pub(crate) env: HashMap<String, Vec<AigLit>>,
    pub(crate) width: usize,
}

impl Lowering<'_> {
    /// A constant bus: bit `k` is TRUE iff bit `k` of `c` (two's-complement) is set.
    pub(crate) fn const_bus(&self, c: i64) -> Vec<AigLit> {
        let bits = c as u64;
        (0..self.width)
            .map(|k| {
                if (bits >> k) & 1 == 1 {
                    AigLit::TRUE
                } else {
                    AigLit::FALSE
                }
            })
            .collect()
    }

    /// One-bit full adder. Returns `(sum, carry_out)`.
    pub(crate) fn full_adder(&mut self, a: AigLit, b: AigLit, cin: AigLit) -> (AigLit, AigLit) {
        let a_xor_b = self.aig.xor(a, b);
        let sum = self.aig.xor(a_xor_b, cin);
        let a_and_b = self.aig.and(a, b);
        let cin_and_axb = self.aig.and(cin, a_xor_b);
        let cout = self.aig.or(a_and_b, cin_and_axb);
        (sum, cout)
    }

    /// Ripple-carry add of two buses, truncated to `width` (final carry discarded).
    pub(crate) fn add_bus(&mut self, a: &[AigLit], b: &[AigLit], cin: AigLit) -> Vec<AigLit> {
        let mut carry = cin;
        let mut out = Vec::with_capacity(self.width);
        for k in 0..self.width {
            let (sum, cout) = self.full_adder(a[k], b[k], carry);
            out.push(sum);
            carry = cout;
        }
        out
    }

    /// Two's-complement subtract: `a - b = a + ~b + 1`.
    pub(crate) fn sub_bus(&mut self, a: &[AigLit], b: &[AigLit]) -> Vec<AigLit> {
        let nb: Vec<AigLit> = b.iter().map(|&bit| self.aig.not(bit)).collect();
        self.add_bus(a, &nb, AigLit::TRUE)
    }

    /// Shift-add array multiplier, truncated to `width`.
    pub(crate) fn mul_bus(&mut self, a: &[AigLit], b: &[AigLit]) -> Vec<AigLit> {
        let mut acc = self.const_bus(0);
        for i in 0..self.width {
            // Partial product of `a` gated by `b[i]`, shifted left by `i`.
            let mut pp = vec![AigLit::FALSE; self.width];
            for j in 0..(self.width - i) {
                pp[i + j] = self.aig.and(a[j], b[i]);
            }
            acc = self.add_bus(&acc, &pp, AigLit::FALSE);
        }
        acc
    }

    /// Per-bit binary op for the bitwise family.
    pub(crate) fn bitwise(
        &mut self,
        a: &[AigLit],
        b: &[AigLit],
        op: fn(&mut Aig, AigLit, AigLit) -> AigLit,
    ) -> Vec<AigLit> {
        (0..self.width).map(|k| op(self.aig, a[k], b[k])).collect()
    }

    /// Lower an expression into a `width`-bit bus.
    pub(crate) fn lower(&mut self, e: &Expr) -> Result<Vec<AigLit>, HlsError> {
        match e {
            Expr::Int(c) => Ok(self.const_bus(*c)),
            Expr::Var(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| HlsError::UnknownVar(name.clone())),
            Expr::Bin(op, a, b) => {
                let x = self.lower(a)?;
                let y = self.lower(b)?;
                Ok(match op {
                    ArithOp::And => self.bitwise(&x, &y, Aig::and),
                    ArithOp::Or => self.bitwise(&x, &y, Aig::or),
                    ArithOp::Xor => self.bitwise(&x, &y, Aig::xor),
                    ArithOp::Add => self.add_bus(&x, &y, AigLit::FALSE),
                    ArithOp::Sub => self.sub_bus(&x, &y),
                    ArithOp::Mul => self.mul_bus(&x, &y),
                })
            }
            Expr::Call(..) => Err(HlsError::Unsupported("function call (needs sequencing)")),
            Expr::Index(..) => Err(HlsError::Unsupported("array index (needs memory)")),
        }
    }
}

/// Reference evaluator: the *spec* the synthesized circuit must match.
///
/// Evaluates a combinational `Expr` over the given integer environment with
/// `width`-bit wraparound arithmetic. Returns the same value the synthesized AIG
/// (and the resulting tiles) must produce.
pub fn eval_expr_ref(expr: &Expr, env: &HashMap<String, u64>, width: u32) -> Result<u64, HlsError> {
    if width == 0 || width > 64 {
        return Err(HlsError::BadWidth(width));
    }
    let mask: u64 = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    fn go(e: &Expr, env: &HashMap<String, u64>, mask: u64) -> Result<u64, HlsError> {
        Ok(match e {
            Expr::Int(c) => (*c as u64) & mask,
            Expr::Var(name) => {
                *env.get(name)
                    .ok_or_else(|| HlsError::UnknownVar(name.clone()))?
                    & mask
            }
            Expr::Bin(op, a, b) => {
                let x = go(a, env, mask)?;
                let y = go(b, env, mask)?;
                let r = match op {
                    ArithOp::Add => x.wrapping_add(y),
                    ArithOp::Sub => x.wrapping_sub(y),
                    ArithOp::Mul => x.wrapping_mul(y),
                    ArithOp::And => x & y,
                    ArithOp::Or => x | y,
                    ArithOp::Xor => x ^ y,
                };
                r & mask
            }
            Expr::Call(..) => {
                return Err(HlsError::Unsupported("function call (needs sequencing)"));
            }
            Expr::Index(..) => return Err(HlsError::Unsupported("array index (needs memory)")),
        })
    }
    go(expr, env, mask)
}

// ===========================================================================
// Phase 2: dual-target metrics (Sprint 383)
// ===========================================================================

/// Cost read-out for the **HLS target** of a source function — the numbers that sit
/// opposite the CPU target's `program_words` / `instructions_executed` / `cycles`
/// (from `tile_cpu::v2_compiler_bench`) in the one-source-two-targets comparison.
///
/// Everything here is derived from already-tested paths: AIG statistics, the
/// technology mapper, the gate-level STA ([`analyze_timing`]), and the placed
/// `SynthExport` grid. The latency–area trade-off in one struct: the CPU reuses one
/// ALU over many cycles; the datapath spends [`HlsReport::placed_area_tiles`] of
/// fabric to finish in [`HlsReport::critical_path_delay`] tile-delays.
#[derive(Clone, Debug)]
pub struct HlsReport {
    /// Datapath bit width the function was synthesized at.
    pub width: u32,
    /// Primary input bits of the AIG (`num_params * width`).
    pub aig_inputs: u32,
    /// Structural AND gates in the bit-blasted AIG.
    pub aig_and_nodes: usize,
    /// AIG logic depth (levels of AND gates).
    pub aig_depth: u32,
    /// Physical cell count after technology mapping + inversion materialization —
    /// each tile-native cell is one logic tile.
    pub mapped_cells: usize,
    /// STA critical-path delay through the mapped netlist, in tile propagation
    /// delays. This is the combinational latency of the datapath: one answer
    /// every `critical_path_delay` tile-ticks, vs. the CPU's `cycles`.
    pub critical_path_delay: f64,
    /// Placed + routed footprint, when place & route succeeded. `None` means the
    /// circuit exceeded routing capacity even at the escalated halo/layer config —
    /// the netlist-level numbers above are still valid.
    pub placed: Option<PlacedFootprint>,
}

/// Footprint of the placed + routed export grid.
#[derive(Clone, Copy, Debug)]
pub struct PlacedFootprint {
    /// Export grid width in tiles.
    pub width: usize,
    /// Export grid height in tiles.
    pub height: usize,
    /// Layers used (routing may use z > 0).
    pub layers: usize,
}

impl PlacedFootprint {
    /// Total placed footprint in tiles (width × height of the export grid).
    pub fn area_tiles(&self) -> usize {
        self.width * self.height
    }
}

impl std::fmt::Display for HlsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HLS {}-bit: {} ANDs depth {} -> {} cells, critical path {:.0}",
            self.width,
            self.aig_and_nodes,
            self.aig_depth,
            self.mapped_cells,
            self.critical_path_delay,
        )?;
        match &self.placed {
            Some(p) => write!(
                f,
                ", placed {}x{} ({} tiles)",
                p.width,
                p.height,
                p.area_tiles()
            ),
            None => write!(f, ", not placeable at current routing capacity"),
        }
    }
}

/// Synthesize a straight-line [`Func`] and measure its HLS-target cost.
///
/// Convenience wrapper: [`synthesize_func`] then [`report_aig`].
pub fn report_func(func: &Func, cfg: &HlsConfig) -> Result<HlsReport, HlsError> {
    let aig = synthesize_func(func, cfg)?;
    report_aig(&aig, cfg.width)
}

/// Measure the HLS-target cost of an already-synthesized datapath AIG.
///
/// Runs the existing back-end paths read-only: map → materialize inversions →
/// STA (intrinsic constraint, so `critical_path_delay` is the achieved delay) →
/// place + route + export (for the placed footprint). No simulation is driven.
///
/// Place & route escalates halo / routing layers for dense datapaths, mirroring
/// `validate_synth_pipeline_adaptive`; if every config fails, `placed` is `None`
/// (the netlist-level metrics are unaffected).
pub fn report_aig(aig: &Aig, width: u32) -> Result<HlsReport, HlsError> {
    let lib = CellLibrary::tile_native();
    let stats = aig.stats();

    let synth_result = synthesize(aig, &lib, &SynthConfig::default());
    let materialized = materialize_inversions(&synth_result.netlist, &lib);
    let timing = analyze_timing(&materialized, &lib, &TimingConstraint::intrinsic());

    // Escalating capacity, same spirit as validate_synth_pipeline_adaptive.
    let configs = [
        BridgeConfig::default(),
        BridgeConfig {
            halo: 6,
            max_z: 3,
            ..BridgeConfig::default()
        },
        BridgeConfig {
            halo: 8,
            max_z: 3,
            ..BridgeConfig::default()
        },
    ];
    let placed = configs.iter().find_map(|cfg| {
        compile_aig_to_export(aig, &lib, cfg)
            .ok()
            .map(|export| PlacedFootprint {
                width: export.sim.width(),
                height: export.sim.height(),
                layers: export.sim.num_layers(),
            })
    });

    Ok(HlsReport {
        width,
        aig_inputs: stats.num_inputs,
        aig_and_nodes: stats.num_and_nodes,
        aig_depth: stats.depth,
        mapped_cells: materialized.num_gates(),
        critical_path_delay: timing.circuit_delay,
        placed,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;
    use crate::synth::bridge::{compile_and_inject, evaluate_block};
    use crate::synth::cell_lib::CellLibrary;
    use crate::synth::mapping::evaluate_aig;

    fn var(name: &str) -> Box<Expr> {
        Box::new(Expr::Var(name.to_string()))
    }
    fn bin(op: ArithOp, a: Box<Expr>, b: Box<Expr>) -> Box<Expr> {
        Box::new(Expr::Bin(op, a, b))
    }

    /// Pack a per-variable value into the flat LSB-first bool input vector that
    /// `evaluate_aig` expects, in the order the input buses were added.
    fn aig_inputs(order: &[(&str, u64)], width: usize) -> Vec<bool> {
        let mut bits = Vec::with_capacity(order.len() * width);
        for &(_, val) in order {
            for k in 0..width {
                bits.push((val >> k) & 1 == 1);
            }
        }
        bits
    }

    /// Read a width-bit LSB-first bool slice back into a u64.
    fn bits_to_u64(bits: &[bool]) -> u64 {
        bits.iter()
            .enumerate()
            .fold(0u64, |acc, (k, &b)| acc | ((b as u64) << k))
    }

    /// Exhaustively check `synthesize_expr` against the reference evaluator using
    /// the software AIG simulator. `order` lists the inputs; each ranges over
    /// `0..2^width`.
    fn check_exhaustive(expr: &Expr, order: &[&str], width: u32) {
        let cfg = HlsConfig { width };
        let aig = synthesize_expr(expr, order, &cfg).expect("synthesis");
        let w = width as usize;
        let span = 1u64 << width;
        // Iterate the full cartesian product of inputs.
        let total = span.pow(order.len() as u32);
        for combo in 0..total {
            let mut vals: Vec<(&str, u64)> = Vec::with_capacity(order.len());
            let mut rem = combo;
            for &name in order {
                vals.push((name, rem % span));
                rem /= span;
            }
            let env: HashMap<String, u64> = vals.iter().map(|&(n, v)| (n.to_string(), v)).collect();
            let expected = eval_expr_ref(expr, &env, width).expect("ref eval");

            let inputs = aig_inputs(&vals, w);
            let out_bits = evaluate_aig(&aig, &inputs);
            assert_eq!(out_bits.len(), w, "output width");
            let actual = bits_to_u64(&out_bits);
            assert_eq!(
                actual, expected,
                "mismatch for {:?}: expected {} got {}",
                vals, expected, actual
            );
        }
    }

    #[test]
    fn hls_bitwise_matches_reference() {
        // (a & b) | (c ^ a)
        let expr = bin(
            ArithOp::Or,
            bin(ArithOp::And, var("a"), var("b")),
            bin(ArithOp::Xor, var("c"), var("a")),
        );
        check_exhaustive(&expr, &["a", "b", "c"], 4);
    }

    #[test]
    fn hls_adder_matches_reference() {
        let expr = bin(ArithOp::Add, var("a"), var("b"));
        check_exhaustive(&expr, &["a", "b"], 4);
    }

    #[test]
    fn hls_subtract_matches_reference() {
        let expr = bin(ArithOp::Sub, var("a"), var("b"));
        check_exhaustive(&expr, &["a", "b"], 4);
    }

    #[test]
    fn hls_multiplier_matches_reference() {
        let expr = bin(ArithOp::Mul, var("a"), var("b"));
        check_exhaustive(&expr, &["a", "b"], 4);
    }

    #[test]
    fn hls_compound_datapath_matches_reference() {
        // (a + b) * c - a   — exercises adder, multiplier, and subtract together.
        let expr = bin(
            ArithOp::Sub,
            bin(
                ArithOp::Mul,
                bin(ArithOp::Add, var("a"), var("b")),
                var("c"),
            ),
            var("a"),
        );
        check_exhaustive(&expr, &["a", "b", "c"], 3);
    }

    #[test]
    fn hls_constant_folding_in_datapath() {
        // a * 2 + 1
        let expr = bin(
            ArithOp::Add,
            bin(ArithOp::Mul, var("a"), Box::new(Expr::Int(2))),
            Box::new(Expr::Int(1)),
        );
        check_exhaustive(&expr, &["a"], 5);
    }

    #[test]
    fn hls_func_with_let_bindings() {
        // int f(a, b, c) { let t = a + b; return t * c; }
        let func = Func {
            name: "f".to_string(),
            params: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            body: vec![
                Stmt::Let("t".to_string(), *bin(ArithOp::Add, var("a"), var("b"))),
                Stmt::Return(*bin(ArithOp::Mul, var("t"), var("c"))),
            ],
        };
        let cfg = HlsConfig { width: 3 };
        let aig = synthesize_func(&func, &cfg).expect("func synthesis");

        let w = 3usize;
        let span = 1u64 << w;
        for a in 0..span {
            for b in 0..span {
                for c in 0..span {
                    let t = (a + b) & (span - 1);
                    let expected = (t * c) & (span - 1);
                    let inputs = aig_inputs(&[("a", a), ("b", b), ("c", c)], w);
                    let actual = bits_to_u64(&evaluate_aig(&aig, &inputs));
                    assert_eq!(actual, expected, "f({a},{b},{c})");
                }
            }
        }
    }

    #[test]
    fn hls_rejects_call_and_index() {
        let cfg = HlsConfig::default();
        let call = Expr::Call("g".to_string(), vec![]);
        assert!(matches!(
            synthesize_expr(&call, &[], &cfg),
            Err(HlsError::Unsupported("function call (needs sequencing)"))
        ));
        let idx = Expr::Index("arr".to_string(), var("i"));
        assert!(matches!(
            synthesize_expr(&idx, &["i"], &cfg),
            Err(HlsError::Unsupported("array index (needs memory)"))
        ));
    }

    #[test]
    fn hls_bad_width_rejected() {
        let expr = Expr::Var("a".to_string());
        assert!(matches!(
            synthesize_expr(&expr, &["a"], &HlsConfig { width: 0 }),
            Err(HlsError::BadWidth(0))
        ));
        assert!(matches!(
            synthesize_expr(&expr, &["a"], &HlsConfig { width: 65 }),
            Err(HlsError::BadWidth(65))
        ));
    }

    /// Physical-authority test: synthesize an adder, compile it into a real tile
    /// grid, and run it as tiles — proving the HLS output is genuine spatial logic,
    /// not just a netlist. (Width 3 keeps the exhaustive 64-combo sweep light.)
    #[test]
    fn hls_adder_runs_as_real_tiles() {
        let width = 3u32;
        let w = width as usize;
        let expr = bin(ArithOp::Add, var("a"), var("b"));
        let aig = synthesize_expr(&expr, &["a", "b"], &HlsConfig { width }).expect("synthesis");

        let lib = CellLibrary::tile_native();
        let mut sim = Simulation::with_size_layered(256, 1280, 16);
        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 200, 0).expect("compile+inject");

        assert_eq!(block.input_indices.len(), 2 * w, "two w-bit input buses");
        assert_eq!(block.output_indices.len(), w, "one w-bit output bus");

        let span = 1u64 << width;
        for a in 0..span {
            for b in 0..span {
                // Drive inputs LSB-first, bus order [a, b], matching AIG input order.
                let mut inputs: Vec<u64> = Vec::with_capacity(2 * w);
                for k in 0..w {
                    inputs.push(if (a >> k) & 1 == 1 { u64::MAX } else { 0 });
                }
                for k in 0..w {
                    inputs.push(if (b >> k) & 1 == 1 { u64::MAX } else { 0 });
                }
                let outs = evaluate_block(&mut sim, &block, &inputs).expect("evaluate");
                let actual = outs
                    .iter()
                    .enumerate()
                    .fold(0u64, |acc, (k, &v)| acc | (((v != 0) as u64) << k));
                let expected = (a + b) & (span - 1);
                assert_eq!(actual, expected, "tile adder {a}+{b}");
            }
        }
    }

    /// Phase 2 (Sprint 383): the dual-target report is a pure read-out over
    /// already-tested paths — verify the numbers are sane and, critically, that
    /// generating a report does not perturb correctness (the AIG still matches
    /// the reference spec).
    #[test]
    fn hls_report_madd_metrics_sane() {
        // madd(a, b, c) = a * b + c — the showcase function.
        let expr = bin(
            ArithOp::Add,
            bin(ArithOp::Mul, var("a"), var("b")),
            var("c"),
        );
        let width = 4u32;
        let aig =
            synthesize_expr(&expr, &["a", "b", "c"], &HlsConfig { width }).expect("synthesis");
        let report = report_aig(&aig, width).expect("report");

        assert_eq!(report.width, width);
        assert_eq!(report.aig_inputs, 3 * width, "three width-bit input buses");
        assert!(report.aig_and_nodes > 0, "multiplier needs gates");
        assert!(report.aig_depth > 0);
        assert!(report.mapped_cells > 0, "mapped to physical cells");
        assert!(
            report.critical_path_delay > 0.0,
            "STA must see a non-zero critical path"
        );
        let placed = report.placed.expect("4-bit madd must place and route");
        assert!(placed.area_tiles() > 0, "placed footprint exists");

        // The report path must not perturb the function: spot-check vs reference.
        for (a, b, c) in [(0u64, 0, 0), (3, 5, 7), (15, 15, 15), (9, 2, 4)] {
            let env: HashMap<String, u64> = [("a", a), ("b", b), ("c", c)]
                .iter()
                .map(|&(n, v)| (n.to_string(), v))
                .collect();
            let expected = eval_expr_ref(&expr, &env, width).expect("ref");
            let inputs = aig_inputs(&[("a", a), ("b", b), ("c", c)], width as usize);
            let actual = bits_to_u64(&evaluate_aig(&aig, &inputs));
            assert_eq!(actual, expected, "madd({a},{b},{c}) after report");
        }
    }

    /// The latency–area trade-off must point the right way as width scales:
    /// a wider datapath costs more area and more combinational delay.
    #[test]
    fn hls_report_scales_with_width() {
        let expr = bin(ArithOp::Add, var("a"), var("b"));
        let r4 = {
            let aig = synthesize_expr(&expr, &["a", "b"], &HlsConfig { width: 4 }).unwrap();
            report_aig(&aig, 4).expect("4-bit report")
        };
        let r8 = {
            let aig = synthesize_expr(&expr, &["a", "b"], &HlsConfig { width: 8 }).unwrap();
            report_aig(&aig, 8).expect("8-bit report")
        };
        assert!(
            r8.aig_and_nodes > r4.aig_and_nodes,
            "wider adder has more gates"
        );
        assert!(
            r8.mapped_cells > r4.mapped_cells,
            "wider adder maps to more cells"
        );
        assert!(
            r8.critical_path_delay > r4.critical_path_delay,
            "ripple carry: wider adder has a longer critical path ({} vs {})",
            r8.critical_path_delay,
            r4.critical_path_delay
        );
    }
}
