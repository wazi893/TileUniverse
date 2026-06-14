//! Generate the self-contained "Cellular at Scale" player.
//!
//! Captures real Game-of-Life runs from the GPU packed kernel, measures device
//! throughput, and writes one interactive `cellular_player.html` with all data
//! embedded — no runtime dependencies.
//!
//!     cargo run --release --features cuda --example cellular_player_export [-- <out.html>]

use engine::cellular_player::export_default_cellular_player;
use logic_fabric_core::cuda::CudaRuntime;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cellular_player.html"));

    let rt = CudaRuntime::new()?;
    println!("device: {}", rt.device_name().unwrap_or_else(|_| "?".into()));
    println!("capturing Game-of-Life demos on the GPU…");

    export_default_cellular_player(&rt, &out)?;

    let kb = std::fs::metadata(&out)?.len() as f64 / 1024.0;
    println!("wrote {} ({:.1} KB)", out.display(), kb);
    Ok(())
}
