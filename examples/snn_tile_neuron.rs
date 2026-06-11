//! A neuron that computes in tiles.
//!
//! Demonstrates the convergence of the two tracks: a McCulloch-Pitts threshold
//! neuron's fire decision is the native operation of a ThresholdVia tile, so a
//! neuron *is* a single delay-1 tile — and the active-via synthesizer compiles
//! a neuron's fire function straight back into that tile.
//!
//! Run: `cargo run --release --example snn_tile_neuron`

use engine::snn::tile_neuron::TileNeuron;
use engine::synth::cell_library_with_vias;
use engine::synth::mapping::{MapConfig, map_to_library};
use engine::synth::timing::{TimingConstraint, analyze_timing};

fn truth_row(neuron: &TileNeuron, inputs: &[bool]) -> (bool, bool) {
    (neuron.fire_reference(inputs), neuron.fire_physical(inputs))
}

fn show(name: &str, neuron: &TileNeuron) {
    let k = neuron.num_inputs as usize;
    println!(
        "\n{} neuron  (k={}, threshold={}):",
        name, neuron.num_inputs, neuron.threshold
    );
    println!("   inputs        ref  tile");
    let mut all_match = true;
    for m in 0u32..(1u32 << k) {
        let inp: Vec<bool> = (0..k).map(|i| (m >> i) & 1 != 0).collect();
        let (r, p) = truth_row(neuron, &inp);
        all_match &= r == p;
        let bits: String = inp.iter().map(|&b| if b { '1' } else { '0' }).collect();
        println!(
            "   {:<12}  {:>3}  {:>4}{}",
            bits,
            r as u8,
            p as u8,
            if r == p { "" } else { "  <-- MISMATCH" }
        );
    }
    println!(
        "   physical tile matches reference: {}",
        if all_match { "yes" } else { "NO" }
    );

    // Closing the loop: synthesize the fire function with the via library.
    let aig = neuron.fire_aig();
    let lib = cell_library_with_vias();
    let netlist = map_to_library(&aig, &lib, &MapConfig::via_aware());
    let cells: Vec<&str> = netlist
        .gates
        .iter()
        .map(|g| lib.cell(g.cell_idx).name)
        .collect();
    let t = analyze_timing(&netlist, &lib, &TimingConstraint::intrinsic());
    println!(
        "   synthesizes to {} tile(s) {:?}, delay {:.0}",
        netlist.num_gates(),
        cells,
        t.circuit_delay
    );
}

fn main() {
    println!("TileUniverse — A Neuron That Computes In Tiles");
    println!("(McCulloch-Pitts threshold neuron == one ThresholdVia tile)");

    show("AND", &TileNeuron::and(2));
    show("OR", &TileNeuron::or(3));
    show("MAJORITY-3", &TileNeuron::majority(3));
    show("MAJORITY-5-of...4", &TileNeuron::new(4, 3)); // "at least 3 of 4"

    println!("\nA 3-input majority neuron — the canonical 'vote' unit — is a single");
    println!("delay-1 tile. Its fire decision is taken by a real ThresholdVia, and the");
    println!("synthesizer compiles the fire function straight back into that tile.");
    println!("The neuron does not run *on* the fabric; it *is* the fabric.");
}
