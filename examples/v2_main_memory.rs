//! Gate A capability demo — Main Memory (extended data address space).
//!
//! Before this sprint the V2 CPU had a 7-bit data address space: 128 cells,
//! period. You could not hold an array of more than ~100 elements, let alone a
//! buffer or a table. This demo fills and sums a **1000-element array** living
//! in extended main memory (addresses 128..1127) — work the 128-cell scratchpad
//! could never express — entirely via flat `LD`/`ST` machine code.
//!
//! What changed: `with_main_memory(words)` widens the data-address mask and
//! attaches a software-backed main memory above address 127 (the "DRAM analog").
//! The CPU core stays physical (tiles); main memory is external storage reached
//! over the bus — exactly the von Neumann CPU/DRAM separation.
//!
//! Run:
//! ```bash
//! cargo run --release --example v2_main_memory
//! ```

fn main() {
    use engine::simulation::Simulation;
    use engine::tile_cpu::{V2Builder, assemble_v2};

    // N = 1000, built as 250 * 4 (LDI immediates are 8-bit, so we scale up).
    const N: u64 = 1000;

    println!("===============================================================");
    println!("  Gate A: V2 CPU Main Memory — extended data address space");
    println!("  Task: fill mem[128..128+{N}) with mem[128+i]=i, then sum it.");
    println!("  (The 128-cell scratchpad cannot hold a {N}-element array.)");
    println!("===============================================================\n");

    let src = r#"
        LDI R1, 128        ; base address of main memory
        LDI R2, 0          ; i
        LDI R3, 250
        ADD R3, R3         ; 500
        ADD R3, R3         ; N = 1000
        LDI R5, 0          ; sum
    fill:
        CMP R2, R3
        JZ sum_start
        MOV R4, R1
        ADD R4, R2         ; addr = 128 + i
        ST [R4], R2        ; mem[128+i] = i
        INC R2
        JMP fill
    sum_start:
        LDI R2, 0
    sum:
        CMP R2, R3
        JZ done
        MOV R4, R1
        ADD R4, R2
        LD R6, [R4]        ; R6 = mem[128+i]
        ADD R5, R6         ; sum += R6
        INC R2
        JMP sum
    done:
        HALT
    "#;

    let program = assemble_v2(src).expect("assemble main-memory program");
    println!("  Assembled program: {} instruction words", program.len());

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_main_memory(1024) // covers addresses 128..1151
        .build(&mut sim);

    println!(
        "  Configured main memory: {} words, address mask 0x{:X} (max addr {})",
        cpu.main_mem_len(),
        cpu.mem_addr_mask(),
        cpu.mem_addr_mask()
    );

    // Run to halt.
    let max_cycles = 100_000u64;
    let t0 = std::time::Instant::now();
    let mut cycles = 0u64;
    while cycles < max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
        cycles += 1;
    }
    let secs = t0.elapsed().as_secs_f32();

    if !cpu.is_halted() {
        eprintln!("  WARN: program did not halt within {max_cycles} cycles");
    }

    let sum = cpu.read_reg(&sim, 5);
    let expected = N * (N - 1) / 2;

    println!("\n  Ran {cycles} cycles in {secs:.3}s");
    println!("  Spot-checks across the scratchpad/main-memory boundary:");
    for k in [0usize, 127, 128, 500, 999] {
        println!("    mem[{:>4}] = {}", 128 + k, cpu.read_ram(&sim, 128 + k));
    }
    println!("\n  Computed sum of 0..{}: {sum}", N - 1);
    println!("  Expected (N*(N-1)/2):      {expected}");

    println!("\n===============================================================");
    if sum == expected {
        println!("  PASS — the V2 CPU processed a {N}-element array in main memory.");
        println!(
            "  Old ceiling: 128 data cells. New ceiling: {} words.",
            cpu.main_mem_len()
        );
    } else {
        println!("  FAIL — sum {sum} != expected {expected}");
        std::process::exit(1);
    }
    println!("===============================================================");
}
