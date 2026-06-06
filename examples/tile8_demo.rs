//! EPIC 108: TILE-8 CPU Demo
//!
//! Demonstrates the TILE-8 CPU emulator with several programs:
//! 1. Simple addition
//! 2. Fibonacci sequence
//! 3. Memory operations
//!
//! Run with: cargo run --example tile8_demo

use engine::tile8::asm::disassemble;
use engine::tile8::{Assembler, Register, Tile8Cpu};

fn main() {
    println!("=== TILE-8 CPU Demo ===\n");

    demo_add();
    demo_fibonacci();
    demo_memory();
    demo_shift_multiply();

    println!("\n=== All demos complete! ===");
}

fn demo_add() {
    println!("--- Demo 1: Add Two Numbers ---");

    let source = r#"
        LDI R0, #2      ; R0 = 2
        LDI R1, #3      ; R1 = 3
        ADD R0, R1      ; R0 = 2 + 3 = 5
    "#;

    let mut asm = Assembler::new();
    let result = asm.assemble(source).unwrap();

    println!("Source:");
    for line in source.lines() {
        let line = line.trim();
        if !line.is_empty() {
            println!("  {}", line);
        }
    }

    println!("\nDisassembly:");
    for line in disassemble(&result.code) {
        println!("  {}", line);
    }

    let mut cpu = Tile8Cpu::new();
    cpu.load_program(&result.code);

    println!("\nExecution trace:");
    for i in 0..3 {
        let inst = cpu.step().unwrap();
        println!("  Step {}: {} -> {}", i + 1, inst, cpu.dump_state());
    }

    println!(
        "\nResult: R0 = {} (expected 5)\n",
        cpu.get_reg(Register::R0)
    );
}

fn demo_fibonacci() {
    println!("--- Demo 2: Fibonacci Sequence ---");
    println!("Computing first 10 Fibonacci numbers...\n");

    // We'll use the emulator to compute fibonacci manually
    // since our ISA's limited immediate values make loops tricky
    let source = r#"
        LDI R0, #1      ; fib(n) = 1
        LDI R1, #0      ; fib(n-1) = 0
    "#;

    let mut asm = Assembler::new();
    let result = asm.assemble(source).unwrap();

    let mut cpu = Tile8Cpu::new();
    cpu.load_program(&result.code);

    // Initialize
    cpu.step(); // R0 = 1
    cpu.step(); // R1 = 0

    let mut fibs = vec![cpu.get_reg(Register::R1), cpu.get_reg(Register::R0)]; // [0, 1]

    println!("  fib(0) = 0");
    println!("  fib(1) = 1");

    // Compute next 8 numbers
    for i in 2..10 {
        let temp = cpu.get_reg(Register::R0);
        let sum = cpu
            .get_reg(Register::R0)
            .wrapping_add(cpu.get_reg(Register::R1));
        cpu.set_reg(Register::R0, sum);
        cpu.set_reg(Register::R1, temp);
        fibs.push(sum);
        println!("  fib({}) = {}", i, sum);
    }

    println!("\nSequence: {:?}\n", fibs);
}

fn demo_memory() {
    println!("--- Demo 3: Memory Operations ---");

    let source = r#"
        LDI R0, #0      ; address offset
        LOAD R1         ; R1 = ROM[R3] (reads instruction at address 0)
    "#;

    let mut asm = Assembler::new();
    let result = asm.assemble(source).unwrap();

    let mut cpu = Tile8Cpu::new();
    cpu.load_program(&result.code);

    // Set some RAM values
    cpu.set_ram(128, 42); // RAM[0] = 42
    cpu.set_ram(129, 99); // RAM[1] = 99

    println!("Initial RAM:");
    println!("  RAM[128] = {}", cpu.get_ram(128));
    println!("  RAM[129] = {}", cpu.get_ram(129));

    // Execute
    cpu.step(); // LDI R0, #0
    cpu.step(); // LOAD R1 from ROM[0]

    println!("\nAfter LOAD R1 from ROM[0]:");
    println!(
        "  R1 = 0x{:02X} (first instruction byte)",
        cpu.get_reg(Register::R1)
    );

    // Now read from RAM
    cpu.set_reg(Register::R3, 128); // Point to RAM
    cpu.state.pc = 1; // Go back to LOAD instruction
    cpu.step();

    println!("\nAfter LOAD R1 from RAM[128]:");
    println!(
        "  R1 = {} (we stored 42 there)\n",
        cpu.get_reg(Register::R1)
    );
}

fn demo_shift_multiply() {
    println!("--- Demo 4: Multiply by Shifting ---");
    println!("Computing 3 * 4 = 12 using shifts and adds...\n");

    // 3 * 4 = 3 * 2^2 = (3 << 2)
    // We can't do 3 * 4 directly, but we can shift
    let source = r#"
        LDI R0, #3      ; R0 = 3
        SHL R0          ; R0 = 6 (3 * 2)
        SHL R0          ; R0 = 12 (3 * 4)
    "#;

    let mut asm = Assembler::new();
    let result = asm.assemble(source).unwrap();

    let mut cpu = Tile8Cpu::new();
    cpu.load_program(&result.code);

    println!("Execution trace:");
    cpu.step();
    println!("  LDI R0, #3  -> R0 = {}", cpu.get_reg(Register::R0));
    cpu.step();
    println!(
        "  SHL R0      -> R0 = {} (3 * 2)",
        cpu.get_reg(Register::R0)
    );
    cpu.step();
    println!(
        "  SHL R0      -> R0 = {} (3 * 4)",
        cpu.get_reg(Register::R0)
    );

    println!(
        "\nResult: 3 * 4 = {} (expected 12)\n",
        cpu.get_reg(Register::R0)
    );
}
