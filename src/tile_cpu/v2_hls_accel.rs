//! Sprint 388 (HW/SW co-design, Phases 1-3): the HLS accelerator MMIO device.
//!
//! "One source, one SoC": a function marked `accel` in the C-like source is NOT
//! compiled to CPU instructions — it is synthesized by the HLS middle layer
//! ([`synthesize_func`]) into a tile datapath, placed + routed into a real
//! [`SynthExport`] grid, and attached to the V2 CPU as a memory-mapped accelerator.
//! Call sites lower to the indexed MMIO protocol (see `v2_compiler::gen_call`), so
//! the same source file yields a heterogeneous SoC: CPU instructions for the cold
//! code, spatial tiles for the marked hot function.
//!
//! # Device model (Phase 1)
//!
//! The device OWNS its tile grid: it wraps the `SynthExport` produced by
//! [`compile_aig_to_export`] and evaluates it per access via [`evaluate_exported`]
//! — the answers are computed by real tiles (physical authority holds; this is the
//! `synth::sequential` precedent). Co-residency in the CPU's host grid (annex
//! region, guard rows) is explicitly deferred to a later sprint.
//!
//! # MMIO protocol (addresses 41-43, see `v2_mmio_devices`)
//!
//! Indexed operands to stay frugal with the fully-assigned 41..=63 window:
//! - write `ARG_SELECT` (41): latch an operand index;
//! - write `ARG_DATA` (42): store the (full 64-bit) value into `operand[index]`
//!   (ignored when the index is out of range);
//! - read `RESULT` (43): mask the latched operands to the datapath width, evaluate
//!   the exported tile circuit, and return the result.
//!
//! Reads of `ARG_SELECT` / `ARG_DATA` return the latched index / selected operand.
//! Writes to `RESULT` are ignored. Straight-line functions use the combinational
//! tile datapath. Functions with `if` / `while` fall through to the sequential
//! FSM datapath (`synth::hls_seq`) and run to completion on `RESULT` reads; an
//! explicit async start/busy/done protocol remains a later refinement.

use std::cell::{Cell, RefCell};
use std::fmt;

use crate::synth::aig::Aig;
use crate::synth::bridge::{BridgeConfig, compile_aig_to_export};
use crate::synth::cell_lib::CellLibrary;
use crate::synth::export::{SynthExport, evaluate_exported};
use crate::synth::hls::{HlsConfig, HlsError, synthesize_func};
use crate::synth::hls_seq::{SeqDatapath, eval_func_ref as eval_seq_func_ref, synthesize_seq_func};
use crate::tile_cpu::v2_compiler::{CompileError, Func, compile_to_words};
use crate::tile_cpu::v2_mmio::V2MmioDevice;
use crate::tile_cpu::v2_mmio_devices::{
    MMIO_ACCEL_ARG_DATA, MMIO_ACCEL_ARG_SELECT, MMIO_ACCEL_RESULT,
};
use crate::tile_cpu::v2_parser::{inject_prelude, parse_program};

const DEFAULT_SEQ_MAX_CYCLES: u64 = 100_000;

/// Reference evaluator for an accel [`Func`]: the software *spec* the synthesized
/// tile datapath must match. It uses the sequential HLS interpreter so the ISS
/// mirror covers both straight-line and `if` / `while` accelerator functions.
/// The ISS-vs-physical differential is literally "software spec vs real tiles".
pub fn eval_accel_func_ref(func: &Func, args: &[u64], width: u32) -> Result<u64, HlsError> {
    eval_seq_func_ref(func, args, width)
}

enum HlsAccelBackend {
    Combinational {
        export: RefCell<SynthExport>,
    },
    Sequential {
        datapath: RefCell<SeqDatapath>,
        max_cycles: u64,
    },
}

impl HlsAccelBackend {
    fn kind(&self) -> &'static str {
        match self {
            HlsAccelBackend::Combinational { .. } => "combinational",
            HlsAccelBackend::Sequential { .. } => "sequential",
        }
    }
}

/// Sprint 388: an HLS-synthesized tile datapath behind MMIO.
///
/// Owns the placed + routed export grid and evaluates it (real tiles) on every
/// `RESULT` read. Register it via [`V2MmioCombinedDevice::with_accel`]
/// (crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice::with_accel).
pub struct V2MmioHlsAccelDevice {
    /// The source-level function this datapath implements (kept for the ISS mirror
    /// and diagnostics).
    func: Func,
    /// Datapath bit width (operands and result are masked to this width).
    width: u32,
    /// The placed + routed tile backend. `RefCell` because the MMIO trait reads
    /// with `&self` but evaluation drives the grid/FSM.
    backend: HlsAccelBackend,
    /// Latched operand index (raw, as written to ARG_SELECT).
    select: Cell<u64>,
    /// Latched operand values, one per function parameter (full 64-bit latches;
    /// masking happens at evaluation).
    operands: RefCell<Vec<u64>>,
    /// Number of accelerator evaluations performed (RESULT reads).
    evals: Cell<u64>,
    /// Last sequential-FSM cycle count, if this device is backed by `hls_seq`.
    last_seq_cycles: Cell<Option<u64>>,
    /// Last backend execution error. MMIO reads cannot return `Result`, so a failed
    /// backend run records the error and returns zero.
    last_error: RefCell<Option<String>>,
}

impl V2MmioHlsAccelDevice {
    /// Synthesize `func` into a tile datapath and wrap it as an MMIO device.
    ///
    /// Straight-line functions use the combinational HLS layer. If that layer
    /// rejects control flow, the function is retried through sequential HLS.
    /// Place & route escalates halo / routing layers for dense datapaths (the
    /// `report_aig` ladder).
    pub fn from_func(func: &Func, cfg: &HlsConfig) -> Result<Self, CompileError> {
        let num_args = func.params.len();
        let backend = match synthesize_func(func, cfg) {
            Ok(aig) => HlsAccelBackend::Combinational {
                export: RefCell::new(Self::place_combinational_export(func, &aig)?),
            },
            Err(comb_err) => match synthesize_seq_func(func, cfg) {
                Ok(datapath) => HlsAccelBackend::Sequential {
                    datapath: RefCell::new(datapath),
                    max_cycles: DEFAULT_SEQ_MAX_CYCLES,
                },
                Err(seq_err) => {
                    return Err(CompileError(format!(
                        "accel function '{}' cannot be synthesized as combinational or \
                         sequential HLS: combinational path: {comb_err}; sequential \
                         path: {seq_err}",
                        func.name
                    )));
                }
            },
        };
        Ok(Self {
            func: func.clone(),
            width: cfg.width,
            backend,
            select: Cell::new(0),
            operands: RefCell::new(vec![0u64; num_args]),
            evals: Cell::new(0),
            last_seq_cycles: Cell::new(None),
            last_error: RefCell::new(None),
        })
    }

    fn place_combinational_export(func: &Func, aig: &Aig) -> Result<SynthExport, CompileError> {
        let lib = CellLibrary::tile_native();
        // Escalating capacity ladder (the `report_aig` spirit) - but skip the default
        // config for large datapaths: measured on madd, `BridgeConfig::default()`
        // FAILS routing for every width >= 5 after burning 4-48 s, while halo 6 /
        // 3 layers succeeds. Starting big AIGs at the higher-capacity config keeps
        // device synthesis inside test budgets.
        let mut configs = Vec::new();
        if aig.stats().num_and_nodes <= 128 {
            configs.push(BridgeConfig::default());
        }
        configs.push(BridgeConfig {
            halo: 6,
            max_z: 3,
            ..BridgeConfig::default()
        });
        configs.push(BridgeConfig {
            halo: 8,
            max_z: 3,
            ..BridgeConfig::default()
        });
        let mut export = None;
        let mut last_err = String::new();
        for bridge_cfg in &configs {
            match compile_aig_to_export(aig, &lib, bridge_cfg) {
                Ok(e) => {
                    export = Some(e);
                    break;
                }
                Err(e) => last_err = format!("{e:?}"),
            }
        }
        export.ok_or_else(|| {
            CompileError(format!(
                "accel function '{}': place & route failed at every capacity config: {last_err}",
                func.name
            ))
        })
    }

    /// The source-level function this device implements.
    pub fn func(&self) -> &Func {
        &self.func
    }

    /// Datapath bit width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Number of tile-datapath evaluations performed (RESULT reads) — proof that
    /// the answers were computed in tiles, not software.
    pub fn evals(&self) -> u64 {
        self.evals.get()
    }

    /// Last sequential-FSM run length, when this device uses the sequential HLS
    /// backend. Combinational accelerator reads leave this as `None`.
    pub fn last_seq_cycles(&self) -> Option<u64> {
        self.last_seq_cycles.get()
    }

    /// Last backend execution error observed during a `RESULT` read.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.borrow().clone()
    }

    fn mask(&self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }

    /// Evaluate the exported tile circuit on the latched operands (masked to the
    /// datapath width). Inputs are packed bus-major, LSB-first — the same order
    /// `synthesize_func` added the parameter input buses.
    fn evaluate_tiles(&self) -> u64 {
        *self.last_error.borrow_mut() = None;
        let result = match &self.backend {
            HlsAccelBackend::Combinational { export } => {
                self.last_seq_cycles.set(None);
                let w = self.width as usize;
                let operands = self.operands.borrow();
                let mut bits = Vec::with_capacity(operands.len() * w);
                for &v in operands.iter() {
                    for k in 0..w {
                        bits.push((v >> k) & 1 == 1);
                    }
                }
                let mut export = export.borrow_mut();
                let out = evaluate_exported(&mut export, &bits);
                out.iter()
                    .enumerate()
                    .fold(0u64, |acc, (k, &b)| acc | ((b as u64) << k))
            }
            HlsAccelBackend::Sequential {
                datapath,
                max_cycles,
            } => {
                let mask = self.mask();
                let args = self
                    .operands
                    .borrow()
                    .iter()
                    .map(|&v| v & mask)
                    .collect::<Vec<_>>();
                match datapath.borrow_mut().run(&args, *max_cycles) {
                    Ok(out) => {
                        self.last_seq_cycles.set(Some(out.cycles));
                        out.result
                    }
                    Err(e) => {
                        self.last_seq_cycles.set(None);
                        *self.last_error.borrow_mut() = Some(e.to_string());
                        0
                    }
                }
            }
        };
        self.evals.set(self.evals.get() + 1);
        result
    }
}

impl V2MmioDevice for V2MmioHlsAccelDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_ACCEL_ARG_SELECT => self.select.get(),
            MMIO_ACCEL_ARG_DATA => {
                let sel = self.select.get() as usize;
                self.operands.borrow().get(sel).copied().unwrap_or(0)
            }
            MMIO_ACCEL_RESULT => self.evaluate_tiles(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_ACCEL_ARG_SELECT => self.select.set(value),
            MMIO_ACCEL_ARG_DATA => {
                let sel = self.select.get() as usize;
                let mut operands = self.operands.borrow_mut();
                if sel < operands.len() {
                    operands[sel] = value;
                }
            }
            // RESULT is read-only; other addresses are not ours.
            _ => {}
        }
    }
}

// Manual Debug: the export grid (a full Simulation) is intentionally not printed.
impl fmt::Debug for V2MmioHlsAccelDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("V2MmioHlsAccelDevice")
            .field("func", &self.func.name)
            .field("width", &self.width)
            .field("backend", &self.backend.kind())
            .field("num_args", &self.operands.borrow().len())
            .field("evals", &self.evals.get())
            .field("last_seq_cycles", &self.last_seq_cycles.get())
            .field("last_error", &self.last_error.borrow())
            .finish()
    }
}

/// A compiled "SoC" program: CPU machine words plus the accelerator device (when the
/// source marked a function `accel`), ready to register on the combined MMIO device.
pub struct V2SocProgram {
    /// The CPU program (accel call sites lowered to the MMIO protocol).
    pub words: Vec<u32>,
    /// The synthesized accelerator, if the source had an `accel` function.
    pub accel: Option<V2MmioHlsAccelDevice>,
}

/// Sprint 388 Phase 2 entry point: parse + compile C-like source where functions may
/// be marked `accel`. Returns the CPU program AND the synthesized accelerator device.
///
/// The sibling of [`compile_source`](crate::tile_cpu::v2_parser::compile_source),
/// which keeps its `Vec<u32>` signature and rejects accel-marked source.
pub fn compile_source_with_accel(src: &str, cfg: &HlsConfig) -> Result<V2SocProgram, CompileError> {
    let mut program = parse_program(src).map_err(|e| CompileError(e.to_string()))?;
    inject_prelude(&mut program).map_err(|e| CompileError(e.to_string()))?;
    if program.accel_funcs.len() > 1 {
        return Err(CompileError(format!(
            "{} accel functions declared; this sprint supports at most one accelerator \
             per program (one MMIO device window)",
            program.accel_funcs.len()
        )));
    }
    let accel = match program.accel_funcs.first() {
        Some(func) => Some(V2MmioHlsAccelDevice::from_func(func, cfg)?),
        None => None,
    };
    let words = compile_to_words(&program)?;
    Ok(V2SocProgram { words, accel })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_compiler::{ArithOp, Expr, Stmt};

    fn accel_func_of(src: &str, name: &str) -> Func {
        let program = parse_program(src).expect("parse accel source");
        program
            .accel_funcs
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("accel function `{name}` not found"))
            .clone()
    }

    fn madd_func() -> Func {
        // madd(a, b, c) = a * b + c — the showcase function.
        Func {
            name: "madd".to_string(),
            params: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            body: vec![Stmt::Return(Expr::Bin(
                ArithOp::Add,
                Box::new(Expr::Bin(
                    ArithOp::Mul,
                    Box::new(Expr::Var("a".to_string())),
                    Box::new(Expr::Var("b".to_string())),
                )),
                Box::new(Expr::Var("c".to_string())),
            ))],
        }
    }

    const SUM_TO_ACCEL_SRC: &str = r#"
        accel int sum_to(int n) {
            int s = 0;
            int i = 1;
            while (i <= n) {
                s = s + i;
                i = i + 1;
            }
            return s;
        }
        int main() { return sum_to(5); }
    "#;

    /// Phase 1 unit test: drive the device through the raw MMIO protocol and compare
    /// every result against the software reference evaluator. Width 4 keeps the
    /// synthesis fast; a handful of arg combos, not an exhaustive sweep.
    #[test]
    fn test_v2_hls_accel_device_matches_reference() {
        let func = madd_func();
        let cfg = HlsConfig { width: 4 };
        let dev = V2MmioHlsAccelDevice::from_func(&func, &cfg).expect("synthesize madd");

        for &(a, b, c) in &[
            (0u64, 0u64, 0u64),
            (3, 5, 7),
            (15, 15, 15), // wraps at width 4
            (9, 2, 4),
            (1, 11, 0),
            (7, 7, 13),
        ] {
            // The indexed MMIO protocol: select index, write value, per operand.
            for (i, v) in [(0u64, a), (1, b), (2, c)] {
                dev.write(MMIO_ACCEL_ARG_SELECT, i);
                dev.write(MMIO_ACCEL_ARG_DATA, v);
            }
            let actual = dev.read(MMIO_ACCEL_RESULT);
            let expected = eval_accel_func_ref(&func, &[a, b, c], cfg.width).expect("ref");
            assert_eq!(actual, expected, "madd({a},{b},{c}) tiles vs reference");
        }
        assert_eq!(dev.evals(), 6, "each RESULT read evaluates the tiles once");

        // Protocol edges: ARG_SELECT reads back; out-of-range writes are ignored;
        // RESULT writes are ignored.
        dev.write(MMIO_ACCEL_ARG_SELECT, 1);
        assert_eq!(dev.read(MMIO_ACCEL_ARG_SELECT), 1);
        assert_eq!(dev.read(MMIO_ACCEL_ARG_DATA), 7, "operand[1] latch");
        dev.write(MMIO_ACCEL_ARG_SELECT, 99);
        dev.write(MMIO_ACCEL_ARG_DATA, 42); // out of range: ignored
        assert_eq!(dev.read(MMIO_ACCEL_ARG_DATA), 0, "OOB select reads 0");
        dev.write(MMIO_ACCEL_RESULT, 123); // read-only: ignored
    }

    /// Phase 3 unit test: control flow accelerators are implemented by the
    /// sequential FSM backend and still compute through tiles.
    #[test]
    #[ignore = "slow (>100s tile sim); run via: cargo nextest run --profile nightly --run-ignored all"]
    fn test_v2_hls_seq_accel_device_matches_reference() {
        let func = accel_func_of(SUM_TO_ACCEL_SRC, "sum_to");
        let cfg = HlsConfig { width: 5 };
        let dev = V2MmioHlsAccelDevice::from_func(&func, &cfg).expect("synthesize sum_to");

        for n in [0u64, 1, 3, 5, 7] {
            dev.write(MMIO_ACCEL_ARG_SELECT, 0);
            dev.write(MMIO_ACCEL_ARG_DATA, n);
            let actual = dev.read(MMIO_ACCEL_RESULT);
            let expected = eval_accel_func_ref(&func, &[n], cfg.width).expect("ref");
            assert_eq!(actual, expected, "sum_to({n}) seq tiles vs reference");
            assert!(
                dev.last_seq_cycles().unwrap_or(0) >= 2 * n + 2,
                "sum_to({n}) finished suspiciously fast: {:?}",
                dev.last_seq_cycles()
            );
            assert_eq!(dev.last_error(), None);
        }
        assert_eq!(dev.evals(), 5, "each RESULT read runs the FSM once");
    }

    /// Unsupported side effects remain clear compile errors, even though control
    /// flow is now accepted by the sequential backend.
    #[test]
    fn test_v2_hls_accel_rejects_statement_call() {
        let src = r#"
            accel int noisy(int n) {
                putchar(n);
                return n;
            }
            int main() { return noisy(3); }
        "#;
        let err = compile_source_with_accel(src, &HlsConfig { width: 5 })
            .err()
            .expect("statement call in an accel function must be rejected");
        assert!(
            err.0.contains("unsupported") || err.0.contains("statement expression"),
            "error should name the unsupported construct: {err}"
        );
    }

    /// `compile_source_with_accel` on accel-free source behaves like the plain
    /// compiler: words come back, no device.
    #[test]
    fn test_v2_hls_accel_source_without_marker_has_no_device() {
        let src = "int main() { return 41 + 1; }";
        let soc = compile_source_with_accel(src, &HlsConfig::default()).expect("compile");
        assert!(soc.accel.is_none());
        assert!(!soc.words.is_empty());
    }
}
