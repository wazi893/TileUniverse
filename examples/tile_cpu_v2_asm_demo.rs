use std::error::Error;

use engine::simulation::Simulation;
use engine::tile_cpu::{V2Builder, assemble_v2};

fn main() -> Result<(), Box<dyn Error>> {
    let source = r#"
        ; Sum 3 + 2 + 1 into R8, then roundtrip through RAM[40].
        LDI R8, 0
        LDI R9, 3
    loop:
        ADD R8, R9
        DEC R9
        JNZ loop
        STB [40], R8
        LDB R10, [40]
        HALT
    "#;

    let program = assemble_v2(source)?;
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(64)
        .with_ram_size(64)
        .build(&mut sim);

    let metrics = cpu.run(&mut sim, 128);

    println!("assembled_words={}", program.len());
    println!("halted={}", cpu.is_halted());
    println!("cycles={}", metrics.cycles);
    println!("instructions_executed={}", metrics.instructions_executed);
    println!("R8={}", cpu.read_reg(&sim, 8));
    println!("R10={}", cpu.read_reg(&sim, 10));
    println!("RAM40={}", cpu.read_ram(&sim, 40));

    Ok(())
}
