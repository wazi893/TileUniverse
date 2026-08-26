//! Generate the single-file "AI Control" hub (both eval players in one HTML).
//!
//!     cargo run --release --example aicontrol_hub_export [-- <out.html>]

use engine::aicontrol_hub::export_default_hub;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("aicontrol_hub.html"));
    export_default_hub(&out)?;
    let kb = std::fs::metadata(&out)?.len() as f64 / 1024.0;
    println!("wrote {} ({:.1} KB)", out.display(), kb);
    Ok(())
}
