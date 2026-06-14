//! Self-contained "Cellular at scale" HTML player.
//!
//! Captures real Conway's Game of Life runs from the GPU packed kernel
//! ([`crate::cuda_life`]) — every animation frame is the kernel's own output —
//! and embeds them, plus a measured device-throughput headline and a
//! CPU-oracle-verified badge, into one interactive HTML file with no runtime
//! dependencies. The same 64-cells-per-`u64` packing the engine's 115T tiles/sec
//! kernels use drives Conway's rule here at multiple trillion cell-updates/sec.
//!
//! Frames are embedded as base64 of the packed bit-grid (fixed `words_per_row *
//! height * 8` bytes per frame, little-endian `u64`s); the template decodes a
//! cell's bit directly by byte index.

use crate::cuda_life::{
    LifeFrameRun, capture_life_gpu, measure_life_throughput_gpu, pattern_colonies,
    pattern_multi_gun, pattern_soup, verify_gpu_matches_cpu,
};
use crate::cuda_tiles::PackedTileGrid;
use logic_fabric_core::cuda::{CudaResult, CudaRuntime};
use std::fs::{create_dir_all, write};
use std::path::Path;

const TEMPLATE: &str = include_str!("cellular_player_template.html");
const MARKER: &str = "/*__CELLULAR_BUNDLE__*/null";

/// Animation grid edge (multiple of 64 for clean horizontal wrap; <=256 so a
/// cell index fits the renderer's expectations).
const DIM: usize = 256;
/// Headline-throughput grid (big enough to be device-bound, not launch-bound).
const PERF_DIM: usize = 8192;
const PERF_GENS: u32 = 100;

/// A named demo: a seed pattern plus how long to evolve it and how often to
/// sample a frame.
pub struct CellDemo {
    pub name: String,
    pub seed: PackedTileGrid,
    pub generations: u32,
    pub sample_every: u32,
}

/// The default switchable demo set, all on a `DIM`×`DIM` toroidal grid.
pub fn default_demos() -> Vec<CellDemo> {
    vec![
        CellDemo {
            name: "Primordial soup".into(),
            seed: pattern_soup(DIM, DIM, 0x00C0_FFEE, 90),
            generations: 210,
            sample_every: 3,
        },
        CellDemo {
            name: "Glider guns".into(),
            seed: pattern_multi_gun(DIM, DIM),
            generations: 280,
            sample_every: 4,
        },
        CellDemo {
            name: "Colony grid (4×4)".into(),
            seed: pattern_colonies(DIM, DIM, 4, 4, 6, 0xBEEF),
            generations: 180,
            sample_every: 3,
        },
    ]
}

/// Build the populated player HTML, capturing each demo on the GPU.
pub fn build_cellular_html(rt: &CudaRuntime, demos: Vec<CellDemo>) -> CudaResult<String> {
    let device = rt.device_name().unwrap_or_else(|_| "GPU".into());
    let verified = verify_gpu_matches_cpu(rt, DIM, DIM, 60, 0xA5A5)?;
    let perf = measure_life_throughput_gpu(rt, PERF_DIM, PERF_DIM, PERF_GENS, 0x1234_5678)?;

    let mut demo_json = String::from("[");
    for (i, demo) in demos.iter().enumerate() {
        if i > 0 {
            demo_json.push(',');
        }
        let run = capture_life_gpu(rt, demo.seed.clone(), demo.generations, demo.sample_every)?;
        demo_json.push_str(&demo_to_json(&demo.name, &run));
    }
    demo_json.push(']');

    let cups = perf.cell_updates_per_sec;
    let bundle = format!(
        "{{\"device\":\"{}\",\"verified\":{},\"throughput\":{{\"w\":{},\"h\":{},\"gens\":{},\"cellUpdates\":{},\"cups\":{:.0},\"ms\":{:.4}}},\"demos\":{},\"default\":0}}",
        json_escape(&device),
        verified,
        perf.width,
        perf.height,
        perf.generations,
        perf.cell_updates,
        cups,
        perf.elapsed_secs * 1e3,
        demo_json
    );

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(logic_fabric_core::cuda::CudaError::InvalidConfig(format!(
            "template marker count = {n}, expected 1"
        )));
    }
    Ok(TEMPLATE.replace(MARKER, &bundle))
}

/// Capture the default demos and write the player to `path`.
pub fn export_default_cellular_player(rt: &CudaRuntime, path: impl AsRef<Path>) -> CudaResult<()> {
    let html = build_cellular_html(rt, default_demos())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)
                .map_err(|e| logic_fabric_core::cuda::CudaError::InvalidConfig(e.to_string()))?;
        }
    }
    write(path, html).map_err(|e| logic_fabric_core::cuda::CudaError::InvalidConfig(e.to_string()))?;
    Ok(())
}

fn demo_to_json(name: &str, run: &LifeFrameRun) -> String {
    let frame_bytes = run.words_per_row * run.height * 8;
    let mut bytes = Vec::with_capacity(frame_bytes * run.frames.len());
    for frame in &run.frames {
        for w in frame {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
    }
    let data = base64(&bytes);

    let pops = run
        .populations
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let gens = run
        .frame_gens
        .iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"name\":\"{}\",\"w\":{},\"h\":{},\"wpr\":{},\"frames\":{},\"sampleEvery\":{},\"gens\":{},\"pops\":[{}],\"frameGens\":[{}],\"data\":\"{}\"}}",
        json_escape(name),
        run.width,
        run.height,
        run.words_per_row,
        run.frames.len(),
        run.sample_every,
        run.generations,
        pops,
        gens,
        data
    )
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_known_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
    }
}
