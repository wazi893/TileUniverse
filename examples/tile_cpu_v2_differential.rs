use std::error::Error;

use engine::tile_cpu::v2_components::{v2_add, v2_cmp, v2_halt, v2_jnz, v2_ldi, v2_with_reg_ext};
use engine::tile_cpu::{V2DiffConfig, run_v2_differential};

fn main() -> Result<(), Box<dyn Error>> {
    let program = [
        v2_ldi(0, 0),                               // R0 = 0
        v2_ldi(1, 1),                               // R1 = 1
        v2_ldi(2, 10),                              // R2 = 10 (loop bound)
        v2_with_reg_ext(v2_ldi(0, 3), true, false), // R8 = 3 (mixed-bank coverage)
        v2_add(0, 1),                               // R0 += 1
        v2_with_reg_ext(v2_add(0, 1), true, false), // R8 += R1
        v2_cmp(0, 2),                               // compare R0 vs 10
        v2_jnz(4),                                  // loop until R0 == 10
        v2_halt(),
    ];

    let stats = match run_v2_differential(&program, V2DiffConfig { max_ticks: 1024 }) {
        Ok(stats) => stats,
        Err(err) => return Err(format!("differential mismatch: {:?}", err).into()),
    };
    println!(
        "differential parity passed: ticks={} retired={}",
        stats.ticks, stats.retired
    );
    Ok(())
}
