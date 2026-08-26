//! Active-via cell catalog: which boolean functions a single computational
//! via (delay 1) realizes, each verified against the real tile evaluator.
//!
//! This is the "active interconnect" thesis made concrete — a ThresholdVia is
//! a threshold logic gate, so logic can be folded into the routing layer.
//!
//! Run: `cargo run --release --example synth_via_cells`

use engine::synth::via_cells::{verify_via_cell, via_cell_catalog, via_npn4_coverage};

fn main() {
    println!("TileUniverse — Active-Via Cell Catalog (delay-1 computational vias)");
    println!("Compare: AND/OR cost 2, XOR costs 3. Every via below costs 1.\n");

    println!(
        "  {:<14} {:>6} {:>4} {:>4} {:>7} {:>8}  physical",
        "cell", "inputs", "k", "T", "enable", "tt"
    );
    println!("  {}", "-".repeat(70));

    for cell in via_cell_catalog() {
        let ok = verify_via_cell(&cell);
        println!(
            "  {:<14} {:>6} {:>4} {:>4} {:>7} {:>#08X}  {}",
            cell.name,
            cell.num_inputs,
            cell.num_neighbors,
            cell.threshold,
            cell.uses_enable,
            cell.truth_table,
            if ok { "verified" } else { "MISMATCH" },
        );
    }

    let (distinct, _) = via_npn4_coverage();
    println!(
        "\n  Distinct NPN4 classes covered by the via catalog: {}",
        distinct
    );

    println!("\nHeadline: VIA_MAJ3 (majority-of-3, tt=0xE8) == full-adder carry-out.");
    println!("A ripple-carry adder's critical path is a chain of carries; each carry");
    println!("becomes ONE delay-1 via instead of an AND/OR cone. That is the win S3 will");
    println!("map and S1's STA will measure.");
}
