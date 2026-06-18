//! Generate the self-contained "Surface Code · Decoder" player.
//!
//! Captures real surface-code decoding shots (errors → syndrome → minimum-weight
//! matching correction → logical-error check) and writes one interactive
//! `qec_player.html` with all data embedded — no runtime dependencies.
//!
//!     cargo run --release --example qec_player_export [-- <out.html>]

use engine::qec_player::export_default_qec_player;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("qec_player.html"));

    export_default_qec_player(&out)?;

    let kb = std::fs::metadata(&out)?.len() as f64 / 1024.0;
    println!("wrote {} ({:.1} KB)", out.display(), kb);
    Ok(())
}
