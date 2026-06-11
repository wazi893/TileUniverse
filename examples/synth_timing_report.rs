//! Static Timing Analysis demo.
//!
//! Runs the synthesis flow on the standard benchmark circuits and prints a
//! `report_timing`-style critical-path report for each — the EDA-tool view of
//! "how fast is this circuit and where is the bottleneck."
//!
//! Run: `cargo run --release --example synth_timing_report`

use engine::synth::benchmark::{
    build_8bit_adder, build_8bit_eq_comparator, build_16bit_adder, build_decoder4to16,
    build_priority_encoder8,
};
use engine::synth::timing::format_path;
use engine::synth::{
    Aig, CellLibrary, SynthConfig, TimingConstraint, analyze_timing, critical_path, synthesize,
};

fn run(name: &str, aig: &Aig) {
    let lib = CellLibrary::tile_native();
    let result = synthesize(aig, &lib, &SynthConfig::default());
    let netlist = &result.netlist;

    let report = analyze_timing(netlist, &lib, &TimingConstraint::intrinsic());
    println!("\n##### {name} #####");
    println!(
        "  inputs={} outputs={} cells={} circuit_delay={:.2}",
        netlist.primary_inputs.len(),
        netlist.primary_outputs.len(),
        report.area_cells,
        report.circuit_delay
    );
    let path = critical_path(netlist, &lib, &report);
    print!("{}", format_path(&path, &report));
}

fn main() {
    println!("TileUniverse — Static Timing Analysis (gate-level, pre-layout)");

    run("adder8", &build_8bit_adder());
    run("adder16", &build_16bit_adder());
    run("eq8", &build_8bit_eq_comparator());
    run("decoder4to16", &build_decoder4to16());
    run("prienc8", &build_priority_encoder8());

    // Demonstrate a clock constraint that the adder8 cannot meet.
    let lib = CellLibrary::tile_native();
    let adder = build_8bit_adder();
    let result = synthesize(&adder, &lib, &SynthConfig::default());
    let intrinsic = analyze_timing(&result.netlist, &lib, &TimingConstraint::intrinsic());
    let tight = intrinsic.circuit_delay * 0.6;
    let report = analyze_timing(&result.netlist, &lib, &TimingConstraint::with_period(tight));
    let path = critical_path(&result.netlist, &lib, &report);
    println!("\n##### adder8 @ tight period {tight:.2} (expect VIOLATED) #####");
    print!("{}", format_path(&path, &report));
}
