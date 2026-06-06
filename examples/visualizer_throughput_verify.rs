//! Verify visualizer throughput calculation
//!
//! This test checks if the throughput displayed matches actual GPU work.
//! Run: cargo run --release --example visualizer_throughput_verify --features "gpu-prototype"

#[cfg(feature = "gpu-prototype")]
fn main() {
    use std::time::Instant;

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  VISUALIZER THROUGHPUT VERIFICATION                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Simulate what the visualizer calculates
    let cpu_count = 268_000_000usize;
    let spacing = 2usize;

    // Calculate grid size (same as visualizer)
    let cpus_per_row = ((cpu_count as f64).sqrt().ceil() as usize).max(1);
    let rows = (cpu_count + cpus_per_row - 1) / cpus_per_row;
    let width = cpus_per_row * spacing;
    let height = rows * spacing;
    let total_tiles = width * height;

    println!("Configuration (RTX 5090 benchmark mode):");
    println!("  CPUs:         {}", format_thousands(cpu_count));
    println!(
        "  Grid:         {} × {}",
        format_thousands(width),
        format_thousands(height)
    );
    println!("  Total tiles:  {}", format_thousands(total_tiles));
    println!(
        "  Tiles/CPU:    {:.2}",
        total_tiles as f64 / cpu_count as f64
    );
    println!();

    // What the UI displays vs what actually runs
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" THROUGHPUT CALCULATION ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let fps = 70.0;

    println!("The visualizer UI code:");
    println!("  let steps_per_frame = if grid_width > 10000 {{ 30.0 }} else {{ 1.0 }};");
    println!("  let evals_per_sec = tiles * fps * steps_per_frame;");
    println!();

    let ui_steps_per_frame = if width > 10000 { 30.0 } else { 1.0 };
    let ui_throughput = total_tiles as f64 * fps * ui_steps_per_frame;

    println!("UI DISPLAYED throughput (assumes 30 steps/frame):");
    println!(
        "  {} tiles × {} fps × {} steps = {:.2}T evals/sec",
        format_thousands(total_tiles),
        fps,
        ui_steps_per_frame,
        ui_throughput / 1e12
    );
    println!();

    println!("ACTUAL execution code:");
    println!("  let steps_per_frame = if viz.running {{ viz.speed_multiplier }} else {{ 1 }};");
    println!("  speed_multiplier defaults to: 1");
    println!();

    let actual_steps_default = 1.0;
    let actual_throughput_default = total_tiles as f64 * fps * actual_steps_default;

    println!("ACTUAL throughput (default slider=1):");
    println!(
        "  {} tiles × {} fps × {} steps = {:.2}B evals/sec",
        format_thousands(total_tiles),
        fps,
        actual_steps_default,
        actual_throughput_default / 1e9
    );
    println!();

    let discrepancy = ui_throughput / actual_throughput_default;
    println!(
        "DISCREPANCY: UI shows {:.0}x more than default execution!",
        discrepancy
    );

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" WHAT DOES 'THROUGHPUT' ACTUALLY MEASURE?");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("Each 'eval' = one tile processed by GPU compute shader");
    println!();
    println!("Per tile, the shader performs:");
    println!("  - 1 meta texture read (tile type)");
    println!("  - 1 state texture read (current value)");
    println!("  - 4 neighbor texture reads");
    println!("  - Logic computation (switch/case based on type)");
    println!("  - 1 state texture write");
    println!("  - 1 color texture write");
    println!("  = ~8-20 memory operations + ALU per tile");
    println!();

    // Estimate actual GPU ops
    let ops_per_tile = 15.0; // Conservative average
    let actual_gpu_ops = actual_throughput_default * ops_per_tile;
    let claimed_gpu_ops = ui_throughput * ops_per_tile;

    println!("Estimated actual GPU memory ops (slider=1):");
    println!(
        "  {:.2}B evals × ~15 ops/eval = {:.2}T GPU ops/sec",
        actual_throughput_default / 1e9,
        actual_gpu_ops / 1e12
    );
    println!();
    println!("Claimed GPU memory ops (UI display, slider assumed=30):");
    println!(
        "  {:.2}T evals × ~15 ops/eval = {:.2}T GPU ops/sec",
        ui_throughput / 1e12,
        claimed_gpu_ops / 1e12
    );

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" SANITY CHECK: RTX 5090 CAPABILITIES");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let rtx5090_bandwidth = 1792.0; // GB/s
    let rtx5090_fp32_tflops = 104.0; // TFLOPS

    // Each tile read/write is ~4 bytes (u32)
    // 8 reads + 2 writes = 10 × 4 bytes = 40 bytes per tile
    let bytes_per_tile = 40.0;

    let max_tiles_by_bandwidth = (rtx5090_bandwidth * 1e9) / bytes_per_tile;

    println!("RTX 5090 specs:");
    println!("  Memory bandwidth: {} GB/s", rtx5090_bandwidth);
    println!("  FP32 compute:     {} TFLOPS", rtx5090_fp32_tflops);
    println!();
    println!(
        "Maximum tile evals by bandwidth ({} bytes/tile):",
        bytes_per_tile
    );
    println!(
        "  {} GB/s ÷ {} bytes = {:.2}B tiles/sec",
        rtx5090_bandwidth,
        bytes_per_tile,
        max_tiles_by_bandwidth / 1e9
    );
    println!();

    if ui_throughput > max_tiles_by_bandwidth {
        println!(
            "⚠️  CLAIMED throughput ({:.2}T) EXCEEDS bandwidth limit ({:.2}B)!",
            ui_throughput / 1e12,
            max_tiles_by_bandwidth / 1e9
        );
        println!("    This is physically impossible unless:");
        println!("    1. Cache hits reduce effective bandwidth needs");
        println!("    2. The measurement is incorrect");
    } else {
        println!("✓  Claimed throughput is within bandwidth limits");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!(" PER-CPU ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Each CPU executes: steps_per_frame instructions per frame
    // At 70 FPS with steps=30: 70 × 30 = 2100 instructions/sec per CPU

    let cpu_instructions_per_sec_claimed = fps * 30.0;
    let cpu_instructions_per_sec_actual = fps * 1.0;

    println!("CPU instruction throughput:");
    println!(
        "  Claimed (30 steps/frame): {:.0} instructions/CPU/sec",
        cpu_instructions_per_sec_claimed
    );
    println!(
        "  Actual default (1 step):  {:.0} instructions/CPU/sec",
        cpu_instructions_per_sec_actual
    );
    println!();
    println!(
        "Total CPU instructions across all {} CPUs:",
        format_thousands(cpu_count)
    );
    println!(
        "  Claimed: {:.2}B instructions/sec",
        (cpu_count as f64 * cpu_instructions_per_sec_claimed) / 1e9
    );
    println!(
        "  Actual:  {:.2}B instructions/sec",
        (cpu_count as f64 * cpu_instructions_per_sec_actual) / 1e9
    );

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CONCLUSION                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("The 2.25T evals/sec figure is the COLLECTIVE throughput of ALL tiles:");
    println!(
        "  - {} tiles evaluated in parallel per GPU step",
        format_thousands(total_tiles)
    );
    println!("  - Running at {} FPS", fps);
    println!("  - WITH {} steps per frame (IF slider is set to 30)", 30);
    println!();
    println!(
        "Each individual CPU executes ~{:.0} instructions/second (at 30 steps/frame)",
        cpu_instructions_per_sec_claimed
    );
    println!();
    println!("The throughput is REAL if:");
    println!("  1. The speed_multiplier slider was manually set to 30");
    println!(
        "  2. The GPU can sustain {:.2}B tile evals/sec bandwidth",
        max_tiles_by_bandwidth / 1e9
    );
    println!();
    println!("To verify: Run the visualizer with --size 268000000 and check:");
    println!("  - What does the 'Speed (Cycles/Frame)' slider show?");
    println!("  - Does changing it affect displayed throughput? (It shouldn't - bug!)");
}

#[cfg(not(feature = "gpu-prototype"))]
fn main() {
    eprintln!("Requires --features gpu-prototype");
}

fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
