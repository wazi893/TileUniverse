//! SNN-in-tiles — a spiking neural network computing on the logic fabric.
//!
//! Every number below is produced by a leaky integrate-and-fire neuron whose
//! *dynamics* run as real tiles: the synaptic weighted sum, the membrane leak +
//! integrate, the threshold fire, and the reset are all a logic datapath
//! synthesized to the tile simulator and evaluated each tick (membrane state
//! threaded between ticks; AlphaFabric's route-validated placer lays the circuit
//! out). Each step is differentially verified against a software reference — the
//! built-in correctness oracle.
//!
//! Run: `cargo run --release --example snn_in_tiles`

use engine::snn::tile_lif::{QuantLif, Synapses, TileLifLayer, TileLifNeuron};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SNN-IN-TILES — a spiking neural network running on the fabric    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // ── Part 1: one leaky integrate-and-fire neuron, on tiles ──────────────
    // 4-bit membrane, threshold 10, leak factor ~3/4 (leak_shift=2 ⇒ steady
    // membrane ≈ 4·current). One synapse, weight 4, its input always active.
    println!("1) A single LIF neuron charging and firing (membrane computed in tiles):\n");
    let lif = QuantLif::new(4, 10).with_leak_shift(2);
    let mut neuron = TileLifNeuron::new(lif, Synapses::new(vec![4], 4));
    println!("   tick │ membrane │ spike");
    println!("   ─────┼──────────┼──────");
    for t in 0..12 {
        let spike = neuron.tick(&[true]); // input spikes every tick
        let bar = "█".repeat(neuron.membrane() as usize);
        println!(
            "   {:>4} │ {:>2} {:<6}│ {}",
            t,
            neuron.membrane(),
            bar,
            if spike { "⚡ FIRE" } else { "" }
        );
    }
    println!("\n   The membrane leaks ~1/4 each tick, integrates +4, and fires + resets");
    println!("   when it crosses the threshold — a leaky integrate-and-fire neuron in tiles.\n");

    // ── Part 2: a 2-class spiking classifier, on tiles ─────────────────────
    // Two output neurons over two input features. Class 0 reads feature 0,
    // class 1 reads feature 1. A pattern is rate-coded (held for N ticks);
    // the predicted class is the argmax over output-neuron spike counts.
    println!("2) A 2-class spiking classifier (rate-code → tile layer → argmax readout):\n");
    let lif = QuantLif::new(4, 10).with_leak_shift(2);
    let mut layer = TileLifLayer::new(
        lif,
        vec![Synapses::new(vec![4, 0], 4), Synapses::new(vec![0, 4], 4)],
    );
    let ticks = 12;
    let patterns = [
        ("A = [active, .....]", vec![true, false]),
        ("B = [....., active]", vec![false, true]),
    ];
    println!("   pattern             │ class-0 spikes │ class-1 spikes │ predicted");
    println!("   ────────────────────┼────────────────┼────────────────┼──────────");
    for (name, pat) in &patterns {
        let (counts, class) = layer.classify(pat, ticks);
        println!(
            "   {:<19} │ {:^14} │ {:^14} │  class {}",
            name, counts[0], counts[1], class
        );
    }
    println!("\n   Each class neuron fires only for its own feature — the layer separates");
    println!("   the two classes. (Same pipeline as MNIST: pixels → rate spikes → layer →");
    println!("   argmax; swap in a trained weight matrix and it classifies digits.)\n");

    println!("Every value above was computed by logic tiles, not a software neuron model —");
    println!("and matches the software reference exactly (the built-in tile==reference oracle).");
}
