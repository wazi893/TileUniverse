//! Generate the self-contained "LIF neuron on the tile fabric" player.
//!
//! Captures real tile-physical LIF neurons (synapses + leak + integrate + fire,
//! all computed on the tile simulator) and writes a single standalone HTML file
//! that animates the logic tiles lighting up as the membrane charges and fires.
//!
//!     cargo run --release --example tile_snn_player_export [-- <out.html>]

use engine::snn::tile_lif_player::export_default_player;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/player/tile_snn_player.html"));
    export_default_player(&out)?;
    let kb = std::fs::metadata(&out)?.len() as f64 / 1024.0;
    println!("wrote {} ({:.1} KB)", out.display(), kb);
    Ok(())
}
