use std::error::Error;
use std::fs::create_dir_all;
use std::path::PathBuf;

use engine::tile_cpu::{
    assemble_v2, capture_v2_replay_bundle, read_v2_replay_bundle_dir, verify_v2_replay_bundle,
    write_v2_replay_bundle_dir,
};

fn main() -> Result<(), Box<dyn Error>> {
    let source = r#"
        LDI R8, 0
        LDI R9, 5
    loop:
        ADD R8, R9
        DEC R9
        JNZ loop
        HALT
    "#;
    let words = assemble_v2(source)?;
    let bundle = capture_v2_replay_bundle(&words, 256)?;

    let out_dir = PathBuf::from("User notes/SPRINTS/SPRINT 142.0/replay_demo_v2");
    create_dir_all(&out_dir)?;
    write_v2_replay_bundle_dir(&bundle, &out_dir)?;

    let loaded = read_v2_replay_bundle_dir(&out_dir)?;
    verify_v2_replay_bundle(&loaded).map_err(|e| format!("replay verification failed: {e}"))?;

    println!("replay_bundle_dir={}", out_dir.display());
    println!("schema={}", loaded.schema_version);
    println!("mmio_mode={}", loaded.mmio_mode);
    println!(
        "mmio_snapshot_kind={}",
        loaded.mmio_snapshot_kind.as_deref().unwrap_or("none")
    );
    println!(
        "mmio_initial_bytes={}",
        loaded
            .mmio_initial_state
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    );
    println!(
        "mmio_final_bytes={}",
        loaded
            .mmio_final_state
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    );
    println!("cycles={}", loaded.cycle_count);
    println!("retired={}", loaded.retired_count);
    println!("trace_entries={}", loaded.trace_entries);
    println!("trace_hash=0x{:016X}", loaded.trace_hash);
    println!("final_state_hash=0x{:016X}", loaded.final_state_hash);
    println!(
        "hybrid=f_switch:{} f_mixed:{} x_mixed:{} ram_hi_rd:{} rom_upper:{}",
        loaded.hybrid.stage_f_bank_switches,
        loaded.hybrid.stage_f_mixed_dual_capture,
        loaded.hybrid.stage_x_mixed_software,
        loaded.hybrid.ram_high_bank_read_swaps,
        loaded.hybrid.rom_upper_bank_group_select,
    );
    println!("replay verification passed.");
    Ok(())
}
