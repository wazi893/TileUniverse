//! Sprint 376 (Gate E.2): compile a C-like program that COMPUTES and PRINTS — the
//! Fibonacci sequence — and run it on the physical tile CPU.
//!
//! `print_int` is auto-included by the compiler (written in the language itself:
//! decimal printing via `div10`/`mod10` repeated subtraction, since the ISA has no
//! divide/modulo). The numbers you see are computed by recursive `fib` calls and
//! formatted by `print_int`, all executing in the logic-gate-simulated tile fabric.
//!
//! Run: `cargo run --release --example v2_compiled_fib`

use std::error::Error;
use std::rc::Rc;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{V2Builder, V2SynthConfig};

fn main() -> Result<(), Box<dyn Error>> {
    // Compute and print fib(0)..fib(10), space-separated, then a newline.
    let source = r#"
        int fib(int n) {
            if (n < 2) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        int main() {
            int i = 0;
            while (i <= 10) {
                print_int(fib(i));
                putchar(32);       // space
                i = i + 1;
            }
            putchar(10);           // newline
            return 0;
        }
    "#;

    let program = compile_source(source).map_err(|e| e.to_string())?;
    println!(
        "compiled {} instruction words (incl. the print_int prelude)",
        program.len()
    );

    let console = Rc::new(V2MmioCombinedDevice::new(0x1234));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);

    let mut ticks = 0u64;
    while !cpu.is_halted() && ticks < 5_000_000 {
        cpu.step(&mut sim);
        ticks += 1;
    }

    println!("halted after {ticks} tile-CPU ticks");
    println!("Fibonacci sequence (computed AND formatted in tiles):");
    println!("----------------------------------------");
    print!("{}", console.ref_pack.console_string());
    println!("----------------------------------------");
    Ok(())
}
