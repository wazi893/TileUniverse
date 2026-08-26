//! Self-contained "LIF neuron on the tile fabric" HTML player.
//!
//! Unlike [`crate::snn_player`] (which animates the *software* [`SNNNetwork`]),
//! this captures the **tile-physical** neuron from [`crate::snn::tile_lif`]: a
//! leaky integrate-and-fire neuron whose synaptic sum + membrane datapath is
//! synthesized to an AIG, placed + routed, and **exported to the tile
//! simulator**. The neuron is stepped tick-by-tick with the membrane threaded
//! back as state, and at every tick we record the on/off state of *every real
//! tile* — so the player shows logic tiles physically lighting up as the
//! membrane integrates, crosses threshold, fires, and resets.
//!
//! Each captured tick is differentially checked against the software reference
//! (the same `tile == reference` oracle the `tile_lif` tests use); the embedded
//! data carries a `verified` flag.
//!
//! Deterministic and CPU-only.

use std::collections::HashMap;
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

use crate::snn::tile_lif::{QuantLif, Synapses};
use crate::synth::alphafabric::{AnnealConfig, Circuit, PlacementEnv, anneal};
use crate::synth::export::{SynthExport, evaluate_exported, export_to_simulation};
use crate::synth::routing::{RouteConfig, route_placed_netlist};
use crate::tiles::tile_meta::TileType;

const TEMPLATE: &str = include_str!("tile_lif_player_template.html");
const MARKER: &str = "/*__TILESNN_BUNDLE__*/null";

/// One renderable tile of the exported neuron circuit.
struct Cell {
    /// Flat tile index into `sim.tilemap` (`z*w*h + y*w + x`).
    idx: usize,
    x: usize,
    y: usize,
    z: usize,
    /// 0 = wire, 1 = gate, 2 = via, 3 = input pad, 4 = output pad.
    kind: u8,
    /// Short glyph/label (gate symbol or pad name like `v0`, `s1`, `spk`).
    label: String,
}

/// A captured tile-physical neuron run.
struct PlayerDataset {
    name: String,
    blurb: String,
    bits: u32,
    threshold: u64,
    leak: String,
    weights: Vec<u64>,
    width: usize,
    height: usize,
    layers: usize,
    cells: Vec<Cell>,
    ticks: usize,
    /// Per-tick on/off bit over `cells` (frame-major, LSB-first packed).
    frames: Vec<u8>,
    /// Membrane value entering / leaving each tick, and the synaptic current.
    v_in: Vec<u64>,
    v_out: Vec<u64>,
    current: Vec<u64>,
    spike: Vec<bool>,
    /// Input spike vector per tick (one entry per synapse).
    stim: Vec<Vec<bool>>,
    gate_count: usize,
    verified: bool,
}

/// A capture configuration: a LIF neuron plus a stimulus.
pub struct PlayerConfig {
    pub name: String,
    pub blurb: String,
    pub lif: QuantLif,
    pub synapses: Synapses,
    /// Input spike vector per tick (length = number of synapses).
    pub stimulus: Vec<Vec<bool>>,
}

/// The default switchable demos: three tile-physical LIF neurons.
pub fn default_configs() -> Vec<PlayerConfig> {
    // Helper stimuli.
    let always_on = |k: usize, ticks: usize| vec![vec![true; k]; ticks];

    // Demo A: classic integrate-and-fire sawtooth. One strong synapse always
    // spiking; leak ~3/4 so the membrane climbs ~4/tick, crosses 10, fires, and
    // resets — over and over.
    let demo_a = PlayerConfig {
        name: "Integrate · fire · reset".into(),
        blurb: "One synapse (w=4) spiking every tick. The 4-bit membrane leaks ~1/4, \
                integrates +4, and fires + resets when it crosses threshold 10 — \
                the sawtooth of a leaky integrate-and-fire neuron, in tiles."
            .into(),
        lif: QuantLif::new(4, 10).with_leak_shift(2),
        synapses: Synapses::new(vec![4], 4),
        stimulus: always_on(1, 18),
    };

    // Demo B: two synapses, a bursty input. Shows synaptic summation and the
    // membrane leaking back down during the quiet windows.
    let mut stim_b = Vec::new();
    for blk in 0..4 {
        // burst: both synapses fire for 3 ticks, then 2 quiet ticks.
        for _ in 0..3 {
            stim_b.push(vec![blk % 2 == 0, true]);
        }
        for _ in 0..2 {
            stim_b.push(vec![false, false]);
        }
    }
    let demo_b = PlayerConfig {
        name: "Two synapses · bursts".into(),
        blurb: "Two weighted synapses (w=3, w=4) driven in bursts. The weighted sum \
                charges the membrane; in the quiet gaps it leaks back down — \
                synaptic integration and leak, computed on the fabric."
            .into(),
        lif: QuantLif::new(4, 10).with_leak_shift(1),
        synapses: Synapses::new(vec![3, 4], 4),
        stimulus: stim_b,
    };

    // Demo C: full Q0.8 leak, a steady drive that produces a slower, smoother
    // climb to threshold (a "real" fractional leak factor, not a power of two).
    let demo_c = PlayerConfig {
        name: "Fractional leak (Q0.8)".into(),
        blurb: "A 5-bit membrane with an arbitrary fixed-point leak factor 0.75 (Q0.8), \
                steady input current 6. The membrane settles toward current/(1-leak) \
                and fires periodically — the full LIF leak, as a tile multiply."
            .into(),
        lif: QuantLif::new(5, 20).with_leak_q8(192),
        synapses: Synapses::new(vec![6], 5),
        stimulus: always_on(1, 20),
    };

    vec![demo_a, demo_b, demo_c]
}

/// Build + place + route + export a LIF neuron's next-state circuit to tiles.
/// Mirrors `TileLifNeuron::new`: cheap row-major baseline first, falling back to
/// AlphaFabric's route-validated SA placer only when the neuron doesn't route.
fn build_neuron_export(lif: &QuantLif, synapses: &Synapses) -> (SynthExport, usize) {
    let circuit = Circuit::from_aig("tile_lif_player_neuron", lif.neuron_step_aig(synapses));
    let gate_count = circuit.netlist.num_gates();
    let mut env = PlacementEnv::new(&circuit).expect("neuron placement env builds");
    let rc = RouteConfig {
        prefer_horizontal_first: true,
        no_crossings: true,
        max_z: 3,
    };
    let routed = match route_placed_netlist(&circuit.netlist, &env.placed(), &rc) {
        Ok(r) => r,
        Err(_) => {
            let cfg = AnnealConfig {
                route_validated_best: true,
                iterations: 1200,
                ..AnnealConfig::default()
            };
            anneal(&mut env, &cfg);
            route_placed_netlist(&circuit.netlist, &env.placed(), &rc)
                .expect("neuron routes after route-validated SA placement")
        }
    };
    (export_to_simulation(&routed, &circuit.netlist), gate_count)
}

/// Categorise a tile type for rendering: (kind, glyph).
fn classify_tile(tt: TileType) -> (u8, &'static str) {
    match tt {
        TileType::And => (1, "&"),
        TileType::Or => (1, "≥1"),
        TileType::Xor => (1, "^"),
        TileType::Not => (1, "!"),
        TileType::ViaUp | TileType::ViaDown => (2, ""),
        _ => (0, ""), // wires (Wire/WireH/WireV/WireUp/WireDown/WireLeft/WireRight/Cross)
    }
}

/// Build the static cell list for the exported circuit.
fn build_cells(export: &SynthExport, bits: u32, n_syn: usize) -> Vec<Cell> {
    let tm = &export.sim.tilemap;
    let w = tm.width;
    let h = tm.height;
    let layer_size = w * h;
    let bits = bits as usize;

    // Output anchors (x,y) -> label.
    let out_labels: HashMap<(usize, usize), String> = export
        .output_coords
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| {
            let label = if i < bits {
                format!("v{i}'")
            } else {
                "spk".to_string()
            };
            ((x, y), label)
        })
        .collect();

    let mut cells = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Input pads first (these sit on Const anchor tiles, so the circuit scan
    // below skips them — add them explicitly so the input bits can glow).
    for (i, &(x, y)) in export.input_coords.iter().enumerate() {
        let idx = y * w + x;
        let label = if i < bits {
            format!("v{i}")
        } else {
            format!("s{}", i - bits)
        };
        let _ = n_syn; // labelled by position; kept for clarity
        cells.push(Cell {
            idx,
            x,
            y,
            z: 0,
            kind: 3,
            label,
        });
        seen.insert(idx);
    }

    // All non-Const circuit tiles (wires, gates, vias) across every layer.
    for idx in 0..tm.tiles.len() {
        let tt = tm.tiles[idx].meta.tile_type;
        if tt == TileType::Const || seen.contains(&idx) {
            continue;
        }
        let z = idx / layer_size;
        let rem = idx % layer_size;
        let y = rem / w;
        let x = rem % w;
        let (mut kind, glyph) = classify_tile(tt);
        let mut label = glyph.to_string();
        if z == 0
            && let Some(l) = out_labels.get(&(x, y))
        {
            kind = 4;
            label = l.clone();
        }
        cells.push(Cell {
            idx,
            x,
            y,
            z,
            kind,
            label,
        });
        seen.insert(idx);
    }

    // Any output anchor whose tile happened to be Const (no wire there) — add it
    // so the output bit is still visible.
    for (i, &(x, y)) in export.output_coords.iter().enumerate() {
        let idx = y * w + x;
        if seen.contains(&idx) {
            continue;
        }
        let label = if i < bits {
            format!("v{i}'")
        } else {
            "spk".to_string()
        };
        cells.push(Cell {
            idx,
            x,
            y,
            z: 0,
            kind: 4,
            label,
        });
        seen.insert(idx);
    }

    cells
}

/// Capture one configuration into a dataset (runs the neuron on tiles).
fn capture(cfg: &PlayerConfig) -> PlayerDataset {
    let bits = cfg.lif.bits;
    let n_syn = cfg.synapses.weights.len();
    let (mut export, gate_count) = build_neuron_export(&cfg.lif, &cfg.synapses);

    let cells = build_cells(&export, bits, n_syn);
    let n_cells = cells.len();
    let bytes_per_frame = n_cells.div_ceil(8);

    let tm_w = export.sim.tilemap.width;
    let tm_h = export.sim.tilemap.height;
    let layers = export.sim.tilemap.tiles.len() / (tm_w * tm_h);

    let ticks = cfg.stimulus.len();
    let mut frames = vec![0u8; ticks * bytes_per_frame];
    let mut v_in = Vec::with_capacity(ticks);
    let mut v_out = Vec::with_capacity(ticks);
    let mut current = Vec::with_capacity(ticks);
    let mut spike = Vec::with_capacity(ticks);

    // Software reference, threaded in lockstep, for the oracle check.
    let mut ref_v = 0u64;
    let mut verified = true;

    let mut state = 0u64;
    let bits_u = bits as usize;
    for (t, spikes) in cfg.stimulus.iter().enumerate() {
        // Inputs: membrane bus v (state) then the synapse spikes.
        let mut ins: Vec<bool> = (0..bits_u).map(|i| (state >> i) & 1 == 1).collect();
        ins.extend_from_slice(spikes);
        let out = evaluate_exported(&mut export, &ins);

        // Capture the on/off state of every rendered tile AFTER convergence.
        let base = t * bytes_per_frame;
        for (c, cell) in cells.iter().enumerate() {
            if export.sim.tilemap.value(cell.idx) != 0 {
                frames[base + (c >> 3)] |= 1 << (c & 7);
            }
        }

        let v_next = (0..bits_u).fold(0u64, |acc, i| acc | ((out[i] as u64) << i));
        let fired = out[bits_u];

        // Reference: synaptic current then the membrane step.
        let cur = cfg.synapses.current_reference(spikes);
        let (ref_next, ref_spike) = cfg.lif.step_reference(ref_v, cur);
        if (v_next, fired) != (ref_next, ref_spike) || !export.last_converged {
            verified = false;
        }
        ref_v = ref_next;

        v_in.push(state);
        v_out.push(v_next);
        current.push(cur);
        spike.push(fired);
        state = v_next;
    }

    PlayerDataset {
        name: cfg.name.clone(),
        blurb: cfg.blurb.clone(),
        bits,
        threshold: cfg.lif.threshold,
        leak: leak_label(&cfg.lif),
        weights: cfg.synapses.weights.clone(),
        width: tm_w,
        height: tm_h,
        layers,
        cells,
        ticks,
        frames,
        v_in,
        v_out,
        current,
        spike,
        stim: cfg.stimulus.clone(),
        gate_count,
        verified,
    }
}

/// A human-readable leak description for the info panel.
fn leak_label(lif: &QuantLif) -> String {
    if let Some(q) = lif.leak_q8 {
        format!("Q0.8 × {:.3}", q as f64 / 256.0)
    } else if lif.leak_shift == 0 {
        "none (pure I&F)".to_string()
    } else {
        format!(
            "1 − 2^−{} ≈ {:.3}",
            lif.leak_shift,
            1.0 - 2f64.powi(-(lif.leak_shift as i32))
        )
    }
}

/// Build the populated player HTML for the given configs.
pub fn build_player_html(configs: &[PlayerConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let datasets: Vec<PlayerDataset> = configs.iter().map(capture).collect();
    let mut json = String::from("{\"demos\":[");
    for (i, d) in datasets.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&dataset_json(d));
    }
    json.push_str("],\"default\":0}");

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(format!("template marker count = {n}, expected 1"));
    }
    Ok(TEMPLATE.replace(MARKER, &json))
}

/// Capture the default demos and write the player to `path`.
pub fn export_default_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_player_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn dataset_json(d: &PlayerDataset) -> String {
    let cells = d
        .cells
        .iter()
        .map(|c| {
            format!(
                "[{},{},{},{},\"{}\"]",
                c.x,
                c.y,
                c.z,
                c.kind,
                json_escape(&c.label)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let v_in = join_u64(&d.v_in);
    let v_out = join_u64(&d.v_out);
    let cur = join_u64(&d.current);
    let spk = d
        .spike
        .iter()
        .map(|&s| if s { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",");
    let stim = d
        .stim
        .iter()
        .map(|s| {
            format!(
                "[{}]",
                s.iter()
                    .map(|&b| if b { "1" } else { "0" })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let weights = join_u64(&d.weights);
    format!(
        "{{\"name\":\"{}\",\"blurb\":\"{}\",\"bits\":{},\"threshold\":{},\"leak\":\"{}\",\
         \"weights\":[{}],\"gates\":{},\"w\":{},\"h\":{},\"layers\":{},\"verified\":{},\
         \"cells\":[{}],\"ticks\":{},\"frames\":\"{}\",\
         \"vIn\":[{}],\"vOut\":[{}],\"current\":[{}],\"spike\":[{}],\"stim\":[{}]}}",
        json_escape(&d.name),
        json_escape(&d.blurb),
        d.bits,
        d.threshold,
        json_escape(&d.leak),
        weights,
        d.gate_count,
        d.width,
        d.height,
        d.layers,
        d.verified,
        cells,
        d.ticks,
        base64(&d.frames),
        v_in,
        v_out,
        cur,
        spk,
        stim,
    )
}

fn join_u64(xs: &[u64]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let nn = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((nn >> 18) & 63) as usize] as char);
        out.push(T[((nn >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((nn >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(nn & 63) as usize] as char
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
    fn capture_is_verified_and_lights_tiles() {
        let cfg = &default_configs()[0];
        let d = capture(cfg);
        // The tile run matches the software reference, every tick.
        assert!(d.verified, "tile capture must match the LIF reference");
        // The neuron actually fires at least once over the run.
        assert!(d.spike.iter().any(|&s| s), "expected at least one spike");
        // There are real circuit tiles, and they light up (some frame bit set).
        assert!(
            d.cells.len() > 8,
            "too few rendered tiles: {}",
            d.cells.len()
        );
        assert!(d.frames.iter().any(|&b| b != 0), "no tiles ever lit");
        // Geometry is consistent.
        assert_eq!(d.v_out.len(), d.ticks);
        assert!(d.width * d.height * d.layers >= d.cells.len());
    }

    #[test]
    fn build_html_embeds_bundle() {
        // One small demo keeps the test fast.
        let cfg = PlayerConfig {
            name: "t".into(),
            blurb: "t".into(),
            lif: QuantLif::new(4, 10).with_leak_shift(2),
            synapses: Synapses::new(vec![4], 4),
            stimulus: vec![vec![true]; 6],
        };
        let html = build_player_html(std::slice::from_ref(&cfg)).expect("build");
        assert!(!html.contains(MARKER), "marker must be replaced");
        assert!(html.contains("\"demos\":["));
        assert!(html.contains("\"frames\":\""));
        assert!(html.contains("\"cells\":["));
    }
}
