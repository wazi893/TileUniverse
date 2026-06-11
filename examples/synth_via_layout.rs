//! Physical via circuits: computational vias placed and run as real tiles.
//!
//! Closes the "is it just a netlist transform?" gap — these circuits are placed
//! on an actual layered tile grid and evaluated by the real simulator. Each
//! output is taken by a ThresholdVia tile; multi-level functions are realized by
//! stacking vias through the cross-layer source term (AND of threshold clauses).
//!
//! Run: `cargo run --release --example synth_via_layout`

use engine::synth::via_layout::{ViaCircuit, ViaClause, ViaStack, realize};

fn dump(name: &str, circuit: &ViaCircuit, labels: &[&str]) {
    let mut phys = realize(circuit);
    let n = circuit.num_pis;
    println!(
        "\n{} — {} layer(s), {} tiles placed",
        name,
        circuit.max_depth(),
        phys.tile_count()
    );
    let header: String = (0..n).map(|i| ((b'a' + i as u8) as char)).collect();
    println!("   {}   {:<24} (ref vs tile)", header, labels.join(" "));
    let mut all_ok = true;
    for combo in 0u32..(1u32 << n) {
        let pis: Vec<bool> = (0..n).map(|i| (combo >> i) & 1 != 0).collect();
        let tile = phys.eval(&pis);
        let refr = circuit.eval_reference(&pis);
        all_ok &= tile == refr;
        let inbits: String = pis.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let outbits: String = tile.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let mark = if tile == refr { "" } else { "  <-- MISMATCH" };
        println!("   {}   {}{}", inbits, outbits, mark);
    }
    println!(
        "   physical tiles match reference for all {} inputs: {}",
        1 << n,
        if all_ok { "yes" } else { "NO" }
    );
}

fn main() {
    println!("TileUniverse — Physical Via Circuits (placed + run on real tiles)");

    // Multi-output threshold layer: 3 neurons sharing 3 inputs.
    let layer = ViaCircuit::new(
        3,
        vec![
            ViaStack::new(vec![ViaClause::and(vec![0, 1, 2])]),
            ViaStack::new(vec![ViaClause::or(vec![0, 1, 2])]),
            ViaStack::new(vec![ViaClause::majority(vec![0, 1, 2])]),
        ],
    );
    dump(
        "Threshold layer [AND, OR, MAJ]",
        &layer,
        &["AND", "OR", "MAJ"],
    );

    // Depth-3 monotone CNF in one vertical via stack:
    //   (a|b) & (c|d) & maj(a,c,d)
    let cnf = ViaCircuit::new(
        4,
        vec![ViaStack::new(vec![
            ViaClause::or(vec![0, 1]),
            ViaClause::or(vec![2, 3]),
            ViaClause::majority(vec![0, 2, 3]),
        ])],
    );
    dump("Monotone CNF stack (a|b)&(c|d)&maj(a,c,d)", &cnf, &["f"]);

    println!("\nEach output bit above is produced by a ThresholdVia tile. Depth comes");
    println!("from stacking vias through the cross-layer source (out[z] = out[z+1] AND");
    println!("threshold_z(neighbors)); width comes from independent stacks. No lateral");
    println!("routing, no feedback — the logic lives in the interconnect itself.");
}
