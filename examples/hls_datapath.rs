//! High-Level Synthesis demo: compile a C-like function into a *spatial tile circuit*.
//!
//! The engine has two back-ends for the same source language:
//!
//! 1. the **CPU back-end** (`v2_compiler`) lowers a function to instructions that run
//!    sequentially on the tile CPU (see `v2_compiled_hello.rs`);
//! 2. the **HLS back-end** (`synth::hls`, this demo) lowers the *same AST* into a
//!    combinational datapath — a multi-bit AIG — which the synthesis stack maps,
//!    places, and routes into real logic tiles.
//!
//! So one source function becomes either a program or a circuit. Here we parse
//! `madd(a, b, c) = a * b + c`, synthesize it to a datapath at two widths to show how
//! it scales, then compile the 4-bit version into a tile grid and *run it as tiles*.
//!
//! Run: `cargo run --release --example hls_datapath`

use std::error::Error;

use engine::simulation::Simulation;
use engine::synth::bridge::{compile_and_inject, evaluate_block};
use engine::synth::{CellLibrary, HlsConfig, synthesize_func};
use engine::tile_cpu::v2_parser::parse_program;

fn main() -> Result<(), Box<dyn Error>> {
    // The same C-like source the CPU compiler accepts. A straight-line combinational
    // function: no loops, no calls, no memory — just a datapath.
    let source = r#"
        int madd(int a, int b, int c) {
            return a * b + c;
        }
    "#;

    let program = parse_program(source).map_err(|e| format!("parse error: {e:?}"))?;
    let func = program
        .funcs
        .iter()
        .find(|f| f.name == "madd")
        .ok_or("function `madd` not found")?;

    println!("Source function: madd(a, b, c) = a * b + c");
    println!("Parameters: {:?}\n", func.params);

    // --- Datapath scaling: synthesize the same AST at two widths ---
    println!("HLS datapath synthesis (one AST, structural circuit per width):");
    for width in [4u32, 8u32] {
        let aig = synthesize_func(func, &HlsConfig { width })?;
        let stats = aig.stats();
        println!(
            "  {:>2}-bit: {} inputs, {} outputs, {} AND gates, depth {}",
            width, stats.num_inputs, stats.num_outputs, stats.num_and_nodes, stats.depth
        );
    }
    println!();

    // --- Run the 4-bit datapath as real tiles ---
    let width = 4u32;
    let w = width as usize;
    let mask = (1u64 << width) - 1;
    let aig = synthesize_func(func, &HlsConfig { width })?;

    let lib = CellLibrary::tile_native();
    let mut sim = Simulation::with_size_layered(256, 1280, 16);
    let (block, _export) = compile_and_inject(&aig, &lib, &mut sim, 64, 100, 0)?;

    println!(
        "Compiled 4-bit madd into {}x{} tiles ({} input pins, {} output pins).",
        block.width,
        block.height,
        block.input_indices.len(),
        block.output_indices.len()
    );
    println!("Running on the tile fabric (result truncated to 4 bits):\n");
    println!("   a  b  c | tiles | expected");
    println!("  ---------+-------+---------");

    let samples = [(2u64, 3, 1), (3, 3, 0), (1, 5, 2), (7, 2, 1), (5, 5, 5)];
    let mut all_match = true;
    for (a, b, c) in samples {
        // Drive inputs LSB-first in bus order [a, b, c], matching the AIG input order.
        let mut inputs: Vec<u64> = Vec::with_capacity(3 * w);
        for &val in &[a, b, c] {
            for k in 0..w {
                inputs.push(if (val >> k) & 1 == 1 { u64::MAX } else { 0 });
            }
        }
        let outs = evaluate_block(&mut sim, &block, &inputs)?;
        let tiles = outs
            .iter()
            .enumerate()
            .fold(0u64, |acc, (k, &v)| acc | (((v != 0) as u64) << k));
        let expected = (a.wrapping_mul(b).wrapping_add(c)) & mask;
        let ok = tiles == expected;
        all_match &= ok;
        println!(
            "  {:>2} {:>2} {:>2} | {:>5} | {:>8} {}",
            a,
            b,
            c,
            tiles,
            expected,
            if ok { "" } else { "<-- MISMATCH" }
        );
    }

    println!();
    if all_match {
        println!("All samples match: the parsed source now runs as spatial logic tiles.");
    } else {
        return Err("tile datapath disagreed with the reference".into());
    }

    Ok(())
}
