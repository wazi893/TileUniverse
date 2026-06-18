//! End-to-end digital-design flow demo: **source code → silicon → verified**.
//!
//! This one binary drives the entire flow a commercial EDA toolchain performs, on
//! TileUniverse's own engine, and proves the result correct:
//!
//! ```text
//!   C-like source                      (front end / HLS input)
//!     -> AST                           (parser, shared with the soft-CPU compiler)
//!     -> bit-blasted datapath AIG      HIGH-LEVEL SYNTHESIS   (Stratus HLS / Synopsys HLS)
//!     -> technology-mapped netlist     LOGIC SYNTHESIS        (Genus / Design Compiler)
//!     -> critical-path / slack         STATIC TIMING ANALYSIS (Tempus / PrimeTime)
//!     -> placed + routed tile grid     PLACE & ROUTE          (Innovus / IC Compiler II)
//!     -> run as real tiles, compared   EQUIVALENCE CHECK      (Conformal LEC / Formality)
//!        bit-for-bit to an independent
//!        reference across ALL inputs
//! ```
//!
//! The point is not the toy datapaths — it is that the *whole flow is complete and
//! correctness-gated*: every circuit is physically simulated as tiles and checked
//! exhaustively against an independent Rust reference. "Routable" is not assumed to
//! imply "correct"; it is verified.
//!
//! Run:  `cargo run --release --example eda_flow`

use engine::synth::bridge::{BridgeConfig, compile_aig_to_export};
use engine::synth::{CellLibrary, HlsConfig, evaluate_exported, report_aig, synthesize_func};
use engine::tile_cpu::v2_compiler::Func;
use engine::tile_cpu::v2_parser::parse_program;

/// One design under test: a name, its C-like source, the function to synthesize, the
/// datapath bit width, and an independent reference model of what it should compute.
struct Design {
    name: &'static str,
    source: &'static str,
    func: &'static str,
    width: u32,
    /// Golden reference: (a, b, c) -> expected output, truncated to `width` bits.
    spec: fn(u64, u64, u64, u64) -> u64,
}

fn designs() -> Vec<Design> {
    vec![
        Design {
            name: "alu  — fused bitwise/arith datapath",
            source: "\
int alu(int a, int b, int c) {
    int s = a + b;
    return (s ^ c) & (a | b);
}
int main() { return alu(1, 2, 3); }
",
            func: "alu",
            width: 4,
            spec: |a, b, c, mask| {
                let s = a.wrapping_add(b) & mask;
                ((s ^ c) & (a | b)) & mask
            },
        },
        Design {
            name: "mac  — multiply-accumulate (array multiplier)",
            source: "\
int mac(int a, int b, int c) {
    return a * b + c;
}
int main() { return mac(2, 3, 1); }
",
            func: "mac",
            width: 3,
            spec: |a, b, c, mask| a.wrapping_mul(b).wrapping_add(c) & mask,
        },
    ]
}

fn main() {
    println!("\n========================================================================");
    println!("  TileUniverse — end-to-end EDA flow:  source -> tiles -> verified");
    println!("========================================================================");

    let mut all_ok = true;
    for d in designs() {
        all_ok &= run_flow(&d);
    }

    println!("\n========================================================================");
    if all_ok {
        println!("  RESULT: every design synthesized, placed, routed, and verified");
        println!("          bit-for-bit on the physical tile fabric. Flow is sound.");
    } else {
        println!("  RESULT: a design FAILED equivalence checking — see above.");
        std::process::exit(1);
    }
    println!("========================================================================\n");
}

/// Drive the full flow for one design and return whether it passed equivalence checking.
fn run_flow(d: &Design) -> bool {
    let width = d.width;
    let mask: u64 = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };

    println!("\n------------------------------------------------------------------------");
    println!("  DESIGN: {}   ({}-bit datapath)", d.name, width);
    println!("------------------------------------------------------------------------");

    // --- Stage 0: front end. Parse C-like source into the shared AST. ----------------
    println!("\n[0] SOURCE  (C-like front end)");
    for line in d.source.trim_end().lines() {
        println!("      | {line}");
    }
    let program = parse_program(d.source).expect("source should parse");
    let func: Func = program
        .funcs
        .into_iter()
        .find(|f| f.name == d.func)
        .unwrap_or_else(|| panic!("function `{}` not found in source", d.func));

    // --- Stage 1: high-level synthesis. AST -> bit-blasted datapath AIG. --------------
    let aig = synthesize_func(&func, &HlsConfig { width }).expect("HLS should succeed");
    let report = report_aig(&aig, width).expect("report");
    println!("\n[1] HIGH-LEVEL SYNTHESIS   (Stratus HLS / Synopsys HLS)");
    println!(
        "      {} primary-input bits  ->  {} AND nodes, logic depth {}",
        report.aig_inputs, report.aig_and_nodes, report.aig_depth
    );

    // --- Stage 2: logic synthesis / technology mapping. ------------------------------
    println!("\n[2] LOGIC SYNTHESIS / TECH MAPPING   (Genus / Design Compiler)");
    println!(
        "      cut-based mapping to the tile cell library  ->  {} physical cells",
        report.mapped_cells
    );

    // --- Stage 3: static timing analysis. --------------------------------------------
    println!("\n[3] STATIC TIMING ANALYSIS   (Tempus / PrimeTime)");
    println!(
        "      combinational critical path = {:.0} tile-delays  (one result every {:.0} ticks)",
        report.critical_path_delay, report.critical_path_delay
    );

    // --- Stage 4: place & route, then export to a live tile grid. --------------------
    // compile_aig_to_export runs map -> materialize inversions -> place -> route ->
    // export, escalating halo/routing layers as needed, and hands back a Simulation
    // whose tiles we can drive and read.
    let lib = CellLibrary::tile_native();
    let mut export = compile_aig_to_export(&aig, &lib, &BridgeConfig::default())
        .expect("place & route should succeed for these datapaths");
    println!("\n[4] PLACE & ROUTE   (Innovus / IC Compiler II)");
    match report.placed {
        Some(p) => println!(
            "      placed {}x{} = {} tiles across {} layer(s)",
            p.width,
            p.height,
            p.area_tiles(),
            p.layers
        ),
        None => println!("      (footprint unavailable; live grid below)"),
    }
    println!(
        "      live tile grid: {}x{}x{}",
        export.sim.width(),
        export.sim.height(),
        export.sim.num_layers()
    );

    // --- Stage 5: equivalence check. Run as real tiles; compare to the spec. ---------
    // Exhaustive when the input space is small enough; otherwise a deterministic sweep.
    let n_params = func.params.len() as u32;
    let total: u64 = 1u64 << (n_params * width);
    let exhaustive = total <= 8192;
    let checks = if exhaustive { total } else { 4096 };

    println!("\n[5] EQUIVALENCE CHECK   (Conformal LEC / Formality)");
    println!(
        "      running on physical tiles, comparing to an independent reference{}",
        if exhaustive {
            format!("\n      over ALL {total} input combinations")
        } else {
            format!("\n      over {checks} sampled input combinations (space = {total})")
        }
    );

    let mut failures = 0u64;
    let mut first_fail: Option<(u64, u64, u64, u64, u64)> = None;
    for i in 0..checks {
        // Pick (a,b,c). Exhaustive: i is the packed index. Sampled: spread deterministically.
        let idx = if exhaustive {
            i
        } else {
            i.wrapping_mul(2_654_435_761) % total
        };
        let a = (idx) & mask;
        let b = (idx >> width) & mask;
        let c = (idx >> (2 * width)) & mask;
        let vals = [a, b, c];

        // Pack inputs in AIG primary-input order: each param is a width-bit bus, LSB first.
        let mut bits = Vec::with_capacity((n_params * width) as usize);
        for &v in vals.iter().take(n_params as usize) {
            for bit in 0..width {
                bits.push((v >> bit) & 1 == 1);
            }
        }

        // Run the tiles and reassemble the output bus (also LSB first).
        let out_bits = evaluate_exported(&mut export, &bits);
        let mut got = 0u64;
        for (k, &bit) in out_bits.iter().enumerate() {
            if bit {
                got |= 1u64 << k;
            }
        }

        let want = (d.spec)(a, b, c, mask) & mask;
        if got != want {
            failures += 1;
            if first_fail.is_none() {
                first_fail = Some((a, b, c, want, got));
            }
        }
    }

    if failures == 0 {
        println!(
            "      PASS — tile output matched the reference on all {checks} cases (0 mismatches)"
        );
        true
    } else {
        let (a, b, c, want, got) = first_fail.unwrap();
        println!("      FAIL — {failures}/{checks} mismatches");
        println!("             first: a={a} b={b} c={c}  expected {want}, tiles gave {got}");
        false
    }
}
