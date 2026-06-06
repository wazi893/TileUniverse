//! Via-aware technology mapping: compare conventional tile-native mapping vs
//! active-via mapping, measured by the S1 static timing analyzer.
//!
//! Run: `cargo run --release --example synth_via_mapping`

use engine::synth::aig::Aig;
use engine::synth::benchmark::{
    build_16bit_adder, build_4bit_adder, build_8bit_adder, build_8bit_eq_comparator,
    build_decoder4to16, build_priority_encoder8,
};
use engine::synth::cuts::enumerate_cuts_with_limit;
use engine::synth::mapping::{map_to_library, verify_equivalence, MapConfig};
use engine::synth::timing::{analyze_timing, TimingConstraint};
use engine::synth::{cell_library_with_vias, CellLibrary};
use std::collections::BTreeMap;

fn maj3_aig() -> Aig {
    // maj(a,b,c) = (a&b) | (a&c) | (b&c)
    let mut aig = Aig::new();
    let a = aig.add_input("a");
    let b = aig.add_input("b");
    let c = aig.add_input("c");
    let ab = aig.and(a, b);
    let ac = aig.and(a, c);
    let bc = aig.and(b, c);
    let t = aig.or(ab, ac);
    let m = aig.or(t, bc);
    aig.add_output("m", m);
    aig
}

fn probe_cuts(name: &str, aig: &Aig) {
    let count = |limit: usize| {
        enumerate_cuts_with_limit(aig, limit)
            .iter()
            .flat_map(|nc| nc.cuts.iter())
            .filter(|c| c.num_leaves == 3 && (c.truth_table == 0xE8 || c.truth_table == 0x17))
            .count()
    };
    println!(
        "  [{}] majority 3-cuts found: budget 8 -> {}, budget 32 -> {}",
        name,
        count(8),
        count(32)
    );
}

fn compare(name: &str, aig: &Aig) {
    let base_lib = CellLibrary::tile_native();
    let via_lib = cell_library_with_vias();

    let base = map_to_library(aig, &base_lib, &MapConfig::default());
    let via = map_to_library(aig, &via_lib, &MapConfig::via_aware());

    let ok = verify_equivalence(aig, &via, &via_lib);

    let bt = analyze_timing(&base, &base_lib, &TimingConstraint::intrinsic());
    let vt = analyze_timing(&via, &via_lib, &TimingConstraint::intrinsic());

    // Via cell usage histogram.
    let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
    for g in &via.gates {
        let n = via_lib.cell(g.cell_idx).name;
        if n.starts_with("VIA_") {
            *hist.entry(n).or_default() += 1;
        }
    }
    let hist_s: Vec<String> = hist.iter().map(|(k, v)| format!("{}×{}", k, v)).collect();

    let dd = 100.0 * (bt.circuit_delay - vt.circuit_delay) / bt.circuit_delay;
    let da = 100.0 * (bt.area_cells as f64 - vt.area_cells as f64) / bt.area_cells as f64;

    println!(
        "  {:<14} delay {:>5.1} -> {:>5.1} ({:>5.1}%)   cells {:>3} -> {:>3} ({:>5.1}%)   eq={}",
        name, bt.circuit_delay, vt.circuit_delay, dd, bt.area_cells, vt.area_cells, da, ok
    );
    if !hist_s.is_empty() {
        println!("       via cells: {}", hist_s.join(", "));
    }
}

fn main() {
    println!("TileUniverse — Via-Aware Technology Mapping (vs conventional)\n");

    println!("Cut-enumeration probe (does the majority cut survive?):");
    probe_cuts("maj3", &maj3_aig());
    probe_cuts("adder4", &build_4bit_adder());

    println!("\nMapping comparison (delay & area via S1 STA):");
    compare("maj3", &maj3_aig());
    compare("adder4", &build_4bit_adder());
    compare("adder8", &build_8bit_adder());
    compare("adder16", &build_16bit_adder());
    compare("eq8", &build_8bit_eq_comparator());
    compare("decoder4to16", &build_decoder4to16());
    compare("prienc8", &build_priority_encoder8());
}
