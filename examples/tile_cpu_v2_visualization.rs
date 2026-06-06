use std::error::Error;
use std::fs::{File, create_dir_all};
use std::path::Path;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_components::{v2_add, v2_halt, v2_ldb, v2_ldi, v2_mov, v2_stb};
use engine::tile_cpu::{
    V2Builder, V2VizLayerLayout, V2VizRenderParams, V2VizSession, write_v2_viz_frame_ppm,
};

fn main() -> Result<(), Box<dyn Error>> {
    let program = [
        v2_ldi(0, 7), // R0 = 7
        v2_ldi(1, 5), // R1 = 5
        v2_add(0, 1), // R0 = 12
        v2_stb(0, 9), // RAM[9] = R0
        v2_ldb(2, 9), // R2 = RAM[9]
        v2_mov(3, 2), // R3 = R2
        v2_halt(),
    ];

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(64)
        .with_ram_size(64)
        .build(&mut sim);

    let mut viz = V2VizSession::new(&sim);
    let params = V2VizRenderParams {
        scale: 1,
        layout: V2VizLayerLayout::Grid2x2,
        highlight_active: true,
        show_flow: true,
        max_flow_edges: 2048,
    };

    let out_dir = Path::new("target/test_output/v2_viz_demo");
    create_dir_all(out_dir)?;

    let mut frame_count = 0usize;
    for step in 0..24usize {
        let _ = cpu.step(&mut sim);
        let frame = viz.capture_frame(&sim, &params);
        let frame_path = out_dir.join(format!("frame_{:03}.ppm", step));
        let mut f = File::create(&frame_path)?;
        write_v2_viz_frame_ppm(&frame, &mut f)?;
        frame_count += 1;

        if cpu.is_halted() && frame.metrics.active_count == 0 {
            break;
        }
    }

    println!("generated {} frames in {}", frame_count, out_dir.display());
    println!(
        "final regs: R0={} R1={} R2={} R3={}",
        cpu.read_reg(&sim, 0),
        cpu.read_reg(&sim, 1),
        cpu.read_reg(&sim, 2),
        cpu.read_reg(&sim, 3)
    );
    Ok(())
}
