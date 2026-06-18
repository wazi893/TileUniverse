//! Generate the self-contained TileCpuV2 live-execution player.
//!
//! Runs the default demo programs through the real capture path and writes one
//! interactive `tile_cpu_v2_player.html` with all data embedded — no runtime
//! dependencies. Open it in any browser and press space.
//!
//!     cargo run --release --example v2_player_export [-- <out.html>]

use std::error::Error;
use std::path::PathBuf;

use engine::tile_cpu::{build_player_html, default_player_programs, write_player_html};

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tile_cpu_v2_player.html"));

    let programs = default_player_programs();
    let html = build_player_html(&programs, 0)?;
    write_player_html(&out, &html)?;

    println!(
        "wrote {} ({:.1} KB, {} programs)",
        out.display(),
        html.len() as f64 / 1024.0,
        programs.len()
    );
    for p in &programs {
        println!("  · {}", p.name);
    }
    Ok(())
}
