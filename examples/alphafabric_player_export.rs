//! Generate the self-contained "AlphaFabric · learned placement" player.
//!
//!     cargo run --release --example alphafabric_player_export [-- <out.html>]

use engine::alphafabric_player::export_default_af_player;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("alphafabric_player.html"));
    export_default_af_player(&out)?;
    let kb = std::fs::metadata(&out)?.len() as f64 / 1024.0;
    println!("wrote {} ({:.1} KB)", out.display(), kb);
    Ok(())
}
