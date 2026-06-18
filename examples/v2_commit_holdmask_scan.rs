//! Step-1.5 evidence: quantify the **commit Const-swap (hold-mask) incidence**.
//!
//! The cycle-spec analysis (V2_CUDA_MEGAKERNEL_SPEC.md) found that the commit phase
//! injects values via tile-TYPE Const-swaps for a large class of instructions, forcing
//! a levelized eval. On-device (no per-lane type mutation) each such swap becomes a
//! per-lane hold-mask + value injection. This scan shows how broad that class is on the
//! charter programs — i.e. that hold-masks are gate-0, not an expansion item.
//!
//! Pure static decode (no CPU build, no run). Run:
//!   cargo run --release --example v2_commit_holdmask_scan

use engine::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS;
use engine::tile_cpu::{CHARTER_PROGRAMS, compile_source, decode_v2_word};

#[derive(Default)]
struct SwapCounts {
    total: usize,
    load: usize,           // LD/LDB  -> wb_data_mux Const-swap
    store: usize,          // ST/STB  -> ram_write_gate (+ ViaUp for addr<8) Const-swap
    mul: usize,            // MUL     -> wb_data_mux + flag_we MUL swap
    mflr: usize,           // MFLR    -> wb_data_mux Const-swap
    ldi_w: usize,          // LDI.W   -> wb_data_mux imm16 Const-swap
    rd_hi: usize,          // rd>=8   -> high-reg merge mux + WE-suppress Const-swap
    needs_holdmask: usize, // any of the above
}

fn classify(program: &[u32]) -> SwapCounts {
    let mut c = SwapCounts {
        total: program.len(),
        ..Default::default()
    };
    for &word in program {
        let d = decode_v2_word(word);
        let load = matches!(d.opcode, 0x16 | 0x18);
        let store = matches!(d.opcode, 0x17 | 0x19);
        let mul = d.opcode == 0x04 && d.imm5 == 1;
        let mflr = d.opcode == 0x02 && d.imm5 == 5;
        let ldi_w = d.opcode == 0x03 && d.wide_imm;
        let rd_hi = d.rd_hi;
        c.load += load as usize;
        c.store += store as usize;
        c.mul += mul as usize;
        c.mflr += mflr as usize;
        c.ldi_w += ldi_w as usize;
        c.rd_hi += rd_hi as usize;
        if load || store || mul || mflr || ldi_w || rd_hi {
            c.needs_holdmask += 1;
        }
    }
    c
}

fn main() {
    println!("V2 -> CUDA: commit Const-swap (hold-mask) incidence — charter set\n");
    println!(
        "{:<16}{:>7}{:>6}{:>6}{:>5}{:>6}{:>7}{:>7}{:>12}",
        "program", "instrs", "LD", "ST", "MUL", "MFLR", "LDI.W", "rd_hi", "needs_hold"
    );
    println!("{}", "-".repeat(72));
    for name in CHARTER_PROGRAMS {
        let case = V2_COMPILED_BENCHMARKS
            .iter()
            .find(|c| c.name == name)
            .expect("charter program in corpus");
        let program = compile_source(case.source).expect("compile");
        let c = classify(&program);
        let pct = if c.total > 0 {
            c.needs_holdmask as f64 / c.total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "{:<16}{:>7}{:>6}{:>6}{:>5}{:>6}{:>7}{:>7}{:>9} ({:.0}%)",
            name, c.total, c.load, c.store, c.mul, c.mflr, c.ldi_w, c.rd_hi, c.needs_holdmask, pct
        );
    }
    println!(
        "\nStatic instruction-stream incidence (dynamic is higher — these sit in hot loops).\n\
         Takeaway: commit hold-masks are required for a large fraction of instructions,\n\
         so per-lane hold/inject is gate-0 (with the RAM SoA), not an expansion item."
    );
}
