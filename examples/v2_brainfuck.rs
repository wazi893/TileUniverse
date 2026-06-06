//! Sprint 382 (Gate F): a **Brainfuck interpreter**, written in the C-like language,
//! compiled by our own compiler, running on the **physical tile CPU**.
//!
//! This is the self-hosting-class milestone: the thing executing on the logic-gate
//! fabric is itself a universal language processor. The BF program is embedded as a
//! global array (`prog`); the BF tape is another global array (`tape`); bracket matching
//! is a depth-scan that needs no extra memory. Every byte of output is produced by the
//! interpreter running in tiles — and is bit-for-bit identical to the ISS reference.
//!
//! The program runs under the 16-bit **wide PC** (`with_wide_pc`), so the physical PC
//! sequences past the 256-instruction line (the 8-bit `extended_pc` path wraps there).
//!
//! Run: `cargo run --release --example v2_brainfuck`

use std::error::Error;
use std::rc::Rc;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{V2Builder, V2SynthConfig};

/// The Brainfuck interpreter, in the C-like language. `PLEN` is substituted with the BF
/// program length. Locals (plen, pc, ptr, op, depth) fit register-mode allocation.
const BF_INTERPRETER: &str = r#"
    int run(int plen) {
        int pc = 0;
        int ptr = 0;
        while (pc < plen) {
            int op = prog[pc];
            if (op == 62) { ptr = ptr + 1; }            // >
            if (op == 60) { ptr = ptr - 1; }            // <
            if (op == 43) { tape[ptr] = tape[ptr] + 1; } // +
            if (op == 45) { tape[ptr] = tape[ptr] - 1; } // -
            if (op == 46) { putchar(tape[ptr]); }        // .
            if (op == 91) {                              // [
                if (tape[ptr] == 0) {
                    int depth = 1;
                    while (depth > 0) {
                        pc = pc + 1;
                        if (prog[pc] == 91) { depth = depth + 1; }
                        if (prog[pc] == 93) { depth = depth - 1; }
                    }
                }
            }
            if (op == 93) {                              // ]
                if (tape[ptr] != 0) {
                    int depth = 1;
                    while (depth > 0) {
                        pc = pc - 1;
                        if (prog[pc] == 91) { depth = depth - 1; }
                        if (prog[pc] == 93) { depth = depth + 1; }
                    }
                }
            }
            pc = pc + 1;
        }
        return 0;
    }
    int main() { return run(PLEN); }
"#;

/// Build a compilable program from a Brainfuck source string: each BF byte becomes an
/// `int` initializer for `prog`; `tape` is a zeroed data array.
fn bf_program(bf: &str, tape_len: usize) -> String {
    let codes: Vec<String> = bf.bytes().map(|b| (b as u32).to_string()).collect();
    let body = BF_INTERPRETER.replace("PLEN", &bf.len().to_string());
    format!(
        "int prog[] = {{ {} }};\nint tape[{}];\n{}",
        codes.join(", "),
        tape_len,
        body
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    // A multiply loop builds '1' in a tape cell; increments print '2'..'5'. Program +
    // tape stay under 41 cells so they fit one contiguous non-MMIO RAM region.
    let bf = "+++++++[>+++++++<-]>.+.+.+.+.";
    let source = bf_program(bf, 4);
    let program = compile_source(&source).map_err(|e| e.to_string())?;

    println!("Brainfuck program : {bf}");
    println!("compiled to        : {} instruction words", program.len());

    let console = Rc::new(V2MmioCombinedDevice::new(0x1234));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_wide_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);

    let mut ticks = 0u64;
    while !cpu.is_halted() && ticks < 5_000_000 {
        cpu.step(&mut sim);
        ticks += 1;
    }

    println!("halted after       : {ticks} tile-CPU ticks");
    println!("----------------------------------------");
    println!(
        "Brainfuck output (produced by an interpreter running in logic tiles):\n  {}",
        console.ref_pack.console_string()
    );
    println!("----------------------------------------");
    Ok(())
}
