//! AlphaFabric AF1 demo.
//!
//! Builds the placement-optimization environment for a few circuits and prints
//! the baseline (row-major) placement quality plus the two-tier legality gate:
//! `map_ok` (mapped netlist == AIG) and `phys_ok` (placed+routed tiles compute
//! the right truth table). This is the measurable design problem that the AF2
//! simulated-annealing placer will optimize against.
//!
//! Run: `cargo run --release --example alphafabric_af1`

use engine::synth::alphafabric::{
    Circuit, PlacementEnv, build_eq_comparator, build_parity, build_ripple_adder,
};

fn main() {
    let circuits = [
        Circuit::from_aig("adder4", build_ripple_adder(4)),
        Circuit::from_aig("eq8", build_eq_comparator(8)),
        Circuit::from_aig("parity8", build_parity(8)),
    ];

    println!("AlphaFabric AF1 — baseline placement quality (row-major)\n");
    println!(
        "{:<9} {:>5} {:>6} {:>5} {:>5} {:>8} {:>7} {:>8}",
        "circuit", "gates", "wire", "vias", "cross", "cost", "map_ok", "phys_ok"
    );
    for c in &circuits {
        let env = PlacementEnv::new(c).expect("env builds");
        let s = env.score();
        let map_ok = c.mapping_is_correct();
        let phys_ok = env.verify_physical();
        println!(
            "{:<9} {:>5} {:>6} {:>5} {:>5} {:>8.0} {:>7} {:>8}",
            c.name,
            env.num_gates(),
            s.metrics.wire_tiles,
            s.metrics.vias,
            s.metrics.crossings,
            s.cost,
            map_ok,
            phys_ok,
        );
    }

    println!(
        "\nLegality invariant: a routable layout must be physically correct.\n\
         Next (AF2): perturb placement via swap/relocate under simulated annealing\n\
         to minimize `cost`, keeping `phys_ok` true."
    );
}
