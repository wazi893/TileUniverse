//! Via-aware HLS (bridge experiment #4): map HLS-generated datapaths against the
//! active-interconnect cell library (ThresholdVia cells, Convergence track S2-S3.5)
//! and compare STA critical path + cell count vs the plain tile-native library.
//!
//! Datapaths come FROM source code via the HLS middle layer (Sprint 383) — this is
//! "HLS onto active interconnect": carry chains and majority logic compress into
//! single delay-1 via cells. Read-out only; equivalence is enforced by the same
//! mapper checks the Convergence track already validated.
//!
//! Run: `cargo run --release --example hls_via_sta`

use engine::synth::hls::{HlsConfig, synthesize_func};
use engine::synth::hls_seq::build_seq_aig;
use engine::synth::{
    Aig, CellLibrary, MapConfig, SynthConfig, TimingConstraint, analyze_timing,
    cell_library_with_vias, materialize_inversions, synthesize,
};
use engine::tile_cpu::v2_parser::parse_program;

fn report(aig: &Aig, lib: &CellLibrary, map_config: MapConfig) -> (usize, f64) {
    let config = SynthConfig {
        map_config,
        ..SynthConfig::default()
    };
    let result = synthesize(aig, lib, &config);
    let materialized = materialize_inversions(&result.netlist, lib);
    let timing = analyze_timing(&materialized, lib, &TimingConstraint::intrinsic());
    (materialized.num_gates(), timing.circuit_delay)
}

fn main() {
    let combinational_src = r#"
        int madd(int a, int b, int c) { return a * b + c; }
        int acc(int a, int b) { return a + b; }
    "#;
    let seq_src = r#"
        int sum_to(int n) {
            int s = 0;
            int i = 1;
            while (i <= n) { s = s + i; i = i + 1; }
            return s;
        }
    "#;

    let prog = parse_program(combinational_src).expect("parse");
    let seq_prog = parse_program(seq_src).expect("parse");

    // (label, AIG)
    let mut cases: Vec<(String, Aig)> = Vec::new();
    for (name, widths) in [("acc", vec![4u32, 8]), ("madd", vec![4, 8])] {
        let func = prog.funcs.iter().find(|f| f.name == name).unwrap();
        for w in widths {
            let aig = synthesize_func(func, &HlsConfig { width: w }).expect("hls");
            cases.push((format!("{name}{w}"), aig));
        }
    }
    let sum_func = seq_prog.funcs.iter().find(|f| f.name == "sum_to").unwrap();
    let (seq_aig, _stats) = build_seq_aig(sum_func, &HlsConfig { width: 5 }).expect("seq hls");
    cases.push(("sum_to5_fsm".to_string(), seq_aig));

    let plain_lib = CellLibrary::tile_native();
    let via_lib = cell_library_with_vias();

    println!("HLS datapaths mapped onto plain tiles vs active-interconnect (via) cells\n");
    println!(
        "{:<12} {:>11} {:>11} | {:>11} {:>11} | {:>8} {:>8}",
        "datapath", "cells", "crit.path", "via cells", "via c.path", "Δcells", "Δdelay"
    );
    println!("{}", "-".repeat(82));
    for (name, aig) in &cases {
        let (pc, pd) = report(aig, &plain_lib, MapConfig::default());
        let (vc, vd) = report(aig, &via_lib, MapConfig::via_aware());
        println!(
            "{:<12} {:>11} {:>11.0} | {:>11} {:>11.0} | {:>+7.0}% {:>+7.0}%",
            name,
            pc,
            pd,
            vc,
            vd,
            (vc as f64 - pc as f64) / pc as f64 * 100.0,
            (vd - pd) / pd * 100.0,
        );
    }
    println!(
        "\nVia cells are ThresholdVia primitives (Convergence S2-S3.5): a 3-input\n\
         majority or OR3/AND3 is ONE delay-1 cell, so adder carry chains and FSM\n\
         comparators compress. Same source code, third target flavor."
    );
}
