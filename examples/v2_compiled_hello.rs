//! Sprint 375 (Gate E): compile a C-like source program and run it on the physical
//! tile CPU, printing whatever it writes to the MMIO console.
//!
//! This is the end-to-end showcase: source text -> lexer/parser -> AST -> codegen ->
//! assembly -> machine code -> logic-gate-simulated CPU -> visible output. The string
//! you see below is produced by `putchar` calls executing in the tile fabric.
//!
//! Run: `cargo run --release --example v2_compiled_hello`

use std::error::Error;
use std::rc::Rc;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{V2Builder, V2SynthConfig};

fn main() -> Result<(), Box<dyn Error>> {
    // A C-like program. `putchar(c)` writes one byte to the console device; the rest
    // is ordinary compiled code (a function, a loop, arithmetic, the calling
    // convention) running on the from-scratch tile CPU.
    let source = r#"
        int print_char(int c) {
            putchar(c);
            return 0;
        }
        int main() {
            // Print "HELLO, TILE CPU!" followed by a newline.
            print_char(72);  print_char(69);  print_char(76);  print_char(76);
            print_char(79);  print_char(44);  print_char(32);  print_char(84);
            print_char(73);  print_char(76);  print_char(69);  print_char(32);
            print_char(67);  print_char(80);  print_char(85);  print_char(33);
            print_char(10);
            return 0;
        }
    "#;

    let program = compile_source(source).map_err(|e| e.to_string())?;
    println!("compiled {} instruction words from source", program.len());

    // Build the physical tile CPU: extended PC (the compiler returns via JMPR) +
    // maximum physical authority + an attached console device we can read back.
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

    // Run the program on the tile fabric until it halts.
    let mut ticks = 0u64;
    while !cpu.is_halted() && ticks < 200_000 {
        cpu.step(&mut sim);
        ticks += 1;
    }

    println!("halted after {ticks} tile-CPU ticks");
    println!("console output (computed in tiles):");
    println!("----------------------------------------");
    print!("{}", console.ref_pack.console_string());
    println!("----------------------------------------");
    Ok(())
}
