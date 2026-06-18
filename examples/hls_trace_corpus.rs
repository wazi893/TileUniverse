//! HLS → interpretability trace corpus (bridge experiment #3, Sprint 383 follow-up).
//!
//! The fellowship-track testbed needs computation with causal structure known *by
//! construction*. Sequential HLS provides exactly that, with a TEMPORAL axis: a
//! C-like source becomes an FSM+datapath whose per-tick state trajectory is fully
//! determined by the source — and editing the source gives **on-manifold,
//! generator-level counterfactuals** (re-synthesize, run, diff trajectories).
//!
//! Emits JSONL records to `target/hls_traces.jsonl`:
//!   { program, width, args, result, ticks: [{fsm, regs:{name:val}}, ...] }
//! plus paired counterfactual runs (sum_to with `i+1` vs `i+2` step) including the
//! first tick where the trajectories diverge.
//!
//! Every trace is gated: the tile-fabric run's result must equal the reference
//! interpreter (`eval_func_ref`) — corrupted traces are refused, not emitted.
//!
//! Run: `cargo run --release --example hls_trace_corpus`

use std::fmt::Write as _;
use std::fs;

use engine::synth::CellLibrary;
use engine::synth::bridge::{BridgeConfig, compile_aig_to_export};
use engine::synth::export::{SynthExport, evaluate_exported};
use engine::synth::hls::HlsConfig;
use engine::synth::hls_seq::{SeqStats, build_seq_aig, eval_func_ref};
use engine::tile_cpu::v2_compiler::Func;
use engine::tile_cpu::v2_parser::parse_program;

/// One tick of the architectural state: FSM value + every register value.
struct Tick {
    fsm: u64,
    regs: Vec<u64>,
}

struct Trace {
    ticks: Vec<Tick>,
    result: u64,
}

/// Compile a function's FSM+datapath to a live tile export (always optimized,
/// escalating capacity — mirrors synthesize_seq_func's ladder).
fn compile(func: &Func, width: u32) -> (SynthExport, SeqStats, Vec<String>) {
    let (aig, stats) = build_seq_aig(func, &HlsConfig { width }).expect("seq aig");
    let lib = CellLibrary::tile_native();
    let mut export = None;
    for halo in [6usize, 8, 10] {
        let cfg = BridgeConfig {
            halo,
            max_z: 3,
            optimize: true,
            ..BridgeConfig::default()
        };
        if let Ok(e) = compile_aig_to_export(&aig, &lib, &cfg) {
            export = Some(e);
            break;
        }
    }
    // Register names: ret first, then vars in collect order = params then lets.
    // (Mirrors the [fsm][ret][vars] state layout of build_seq_aig.)
    let mut names = vec!["ret".to_string()];
    names.extend(func.params.iter().cloned());
    let n_regs = (stats.num_state_bits - stats.fsm_bits) / width as usize;
    let mut i = 0;
    while names.len() < n_regs {
        names.push(format!("local{i}"));
        i += 1;
    }
    (export.expect("FSM cone routed"), stats, names)
}

/// Run the FSM on real tiles, recording the architectural state every tick.
fn run_traced(
    export: &mut SynthExport,
    stats: &SeqStats,
    width: u32,
    args: &[u64],
    max_cycles: usize,
) -> Option<Trace> {
    let w = width as usize;
    let n = stats.num_state_bits;
    let n_regs = (n - stats.fsm_bits) / w;
    let mut state = vec![false; n];
    for (p, &val) in args.iter().enumerate() {
        let base = stats.fsm_bits + w + p * w; // skip fsm + ret
        for k in 0..w {
            state[base + k] = (val >> k) & 1 == 1;
        }
    }
    let decode = |state: &[bool]| -> Tick {
        let fsm = state[..stats.fsm_bits]
            .iter()
            .enumerate()
            .fold(0u64, |a, (k, &b)| a | ((b as u64) << k));
        let regs = (0..n_regs)
            .map(|r| {
                let base = stats.fsm_bits + r * w;
                (0..w).fold(0u64, |a, k| a | ((state[base + k] as u64) << k))
            })
            .collect();
        Tick { fsm, regs }
    };

    let mut ticks = vec![decode(&state)];
    for _ in 0..max_cycles {
        let outs = evaluate_exported(export, &state);
        if outs[n] {
            let result = outs[n + 1..n + 1 + w]
                .iter()
                .enumerate()
                .fold(0u64, |a, (k, &b)| a | ((b as u64) << k));
            return Some(Trace { ticks, result });
        }
        state.copy_from_slice(&outs[..n]);
        ticks.push(decode(&state));
    }
    None
}

fn json_trace(program: &str, width: u32, args: &[u64], names: &[String], t: &Trace) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        r#"{{"program":"{program}","width":{width},"args":{args:?},"result":{},"ticks":["#,
        t.result
    );
    for (i, tick) in t.ticks.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, r#"{{"fsm":{},"regs":{{"#, tick.fsm);
        for (r, v) in tick.regs.iter().enumerate() {
            if r > 0 {
                s.push(',');
            }
            let _ = write!(s, r#""{}":{v}"#, names[r]);
        }
        s.push_str("}}");
    }
    s.push_str("]}");
    s
}

fn main() {
    const SUM_TO: &str = "int sum_to(int n) { int s = 0; int i = 1; while (i <= n) { s = s + i; i = i + 1; } return s; }";
    // The counterfactual: ONE source token changed (step i+1 -> i+2).
    const SUM_TO_CF: &str = "int sum_to(int n) { int s = 0; int i = 1; while (i <= n) { s = s + i; i = i + 2; } return s; }";
    const GCD: &str = "int gcd(int a, int b) { while (a != b) { if (a > b) { a = a - b; } else { b = b - a; } } return a; }";

    let width = 5u32;
    let mut out = String::new();
    let mut emitted = 0usize;
    let mut refused = 0usize;

    // --- Base corpus: factual traces, gated against the reference interpreter ---
    let runs: &[(&str, &str, &[&[u64]])] = &[
        ("sum_to", SUM_TO, &[&[3], &[5], &[7]]),
        ("gcd", GCD, &[&[12, 8], &[21, 14], &[13, 7]]),
    ];
    for (name, src, arg_sets) in runs {
        let prog = parse_program(src).expect("parse");
        let func = prog.funcs.iter().find(|f| f.name == *name).unwrap();
        let (mut export, stats, names) = compile(func, width);
        for args in *arg_sets {
            let Some(trace) = run_traced(&mut export, &stats, width, args, 300) else {
                refused += 1;
                continue;
            };
            let expect = eval_func_ref(func, args, width).unwrap();
            if trace.result != expect {
                refused += 1;
                println!(
                    "REFUSED {name}{args:?}: tiles {} != ref {expect}",
                    trace.result
                );
                continue;
            }
            out.push_str(&json_trace(name, width, args, &names, &trace));
            out.push('\n');
            emitted += 1;
            println!(
                "{name}{args:?} = {} in {} ticks (fsm trajectory: {:?}...)",
                trace.result,
                trace.ticks.len(),
                trace
                    .ticks
                    .iter()
                    .take(8)
                    .map(|t| t.fsm)
                    .collect::<Vec<_>>()
            );
        }
    }

    // --- Counterfactual pair: same args, one source token differs ---
    println!("\ncounterfactual pair (sum_to step 1 vs step 2), args [5]:");
    let mut pair = Vec::new();
    for (tag, src) in [("factual", SUM_TO), ("counterfactual", SUM_TO_CF)] {
        let prog = parse_program(src).expect("parse");
        let func = prog.funcs.iter().find(|f| f.name == "sum_to").unwrap();
        let (mut export, stats, names) = compile(func, width);
        let trace = run_traced(&mut export, &stats, width, &[5], 300).expect("trace");
        let expect = eval_func_ref(func, &[5], width).unwrap();
        assert_eq!(trace.result, expect, "{tag}: gate");
        println!(
            "  {tag}: result={} ticks={}",
            trace.result,
            trace.ticks.len()
        );
        out.push_str(&json_trace(
            &format!("sum_to/{tag}"),
            width,
            &[5],
            &names,
            &trace,
        ));
        out.push('\n');
        emitted += 1;
        pair.push(trace);
    }
    let diverge = pair[0]
        .ticks
        .iter()
        .zip(pair[1].ticks.iter())
        .position(|(a, b)| a.fsm != b.fsm || a.regs != b.regs);
    println!(
        "  trajectories diverge at tick {:?} (the edit's causal horizon)",
        diverge
    );

    fs::create_dir_all("target").ok();
    fs::write("target/hls_traces.jsonl", &out).expect("write corpus");
    println!("\n{emitted} traces emitted ({refused} refused) -> target/hls_traces.jsonl");
}
