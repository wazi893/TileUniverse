//! Self-contained HTML execution player for [`TileCpuV2`](crate::tile_cpu::TileCpuV2).
//!
//! Bundles a *real* captured execution timeline — the static tile floorplan plus
//! per-instruction active-tile / signal-flow / architectural-state data — into a
//! single interactive HTML file with no runtime dependencies. Open the output in
//! any browser to watch a CPU built entirely from cellular-automaton logic tiles
//! execute a program: every glowing tile is a real signal that changed in the
//! fabric that instruction.
//!
//! # Pipeline
//!
//! [`build_player_html`] runs each program through the same capture path as the
//! trace-viz bundle ([`capture_v2_trace_viz_timeline`]) — so the playback is a
//! truthful replay, not a hand-drawn animation — serialises it to compact JSON,
//! and injects that JSON into the bundled HTML template
//! (`v2_player_template.html`). The template renders the chip on a `<canvas>`:
//! the dim floorplan, the glowing trail of active tiles, the signal-flow edges,
//! and a synchronised inspector (disassembly, register file, RAM/MMIO deltas,
//! and an MMIO console).
//!
//! The coordinate packing here (`(z*H + y)*W + x`) is mirrored verbatim by the
//! template's `unpack()` — the two must stay in lock-step.

use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

use crate::simulation::Simulation;
use crate::tile_cpu::{
    V2Builder, V2MmioHandle, V2MmioRefDevicePack, V2TraceEntry, V2VizFlowEdge, V2VizFrame,
    V2VizLayerLayout, V2VizRenderParams, V2VizSession, assemble_v2, capture_v2_trace_viz_timeline,
    disassemble_v2_word,
};
use crate::tile_meta::TileType;

/// The HTML shell (CSS + player JS). Data is injected at [`PLAYER_DATA_MARKER`].
const PLAYER_TEMPLATE: &str = include_str!("v2_player_template.html");
/// Sentinel replaced with the embedded bundle JSON. It is valid JS (`null`) so
/// the unpopulated template still opens standalone as an empty shell.
const PLAYER_DATA_MARKER: &str = "/*__TILECPU_PLAYER_BUNDLE__*/null";

/// Fixed grid + memory geometry for the player CPUs. All demos share this shape
/// so the static floorplan is identical across programs and embedded once.
const GRID_W: usize = 128;
const GRID_H: usize = 128;
const GRID_LAYERS: usize = 4;
const ROM_SIZE: usize = 64;
const RAM_SIZE: usize = 64;
/// Upper bound on captured cycles per program (demos retire well under this).
const MAX_CYCLES: u64 = 400;

/// A named program to feature in the player's demo selector.
#[derive(Debug, Clone)]
pub struct V2PlayerProgram {
    pub name: String,
    pub source: String,
}

impl V2PlayerProgram {
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
        }
    }
}

/// The default switchable demo set:
/// 1. **Accumulate loop** (canonical "watch the CPU compute" hero): sums 3+2+1
///    in a register through a `DEC`/`JNZ` loop, then round-trips through RAM.
/// 2. **Fibonacci**: iterative `t=a+b; a=b; b=t` — pure ALU + branch.
/// 3. **Print "HELLO"**: stores ASCII bytes to the MMIO console (addr 57),
///    which the player surfaces as live console text.
pub fn default_player_programs() -> Vec<V2PlayerProgram> {
    vec![
        V2PlayerProgram::new(
            "Accumulate loop  (Σ = 6)",
            "        LDI R8, 0\n        LDI R9, 3\n    loop:\n        ADD R8, R9\n        DEC R9\n        JNZ loop\n        STB [40], R8\n        LDB R10, [40]\n        HALT\n",
        ),
        V2PlayerProgram::new(
            "Fibonacci  (8 steps)",
            "        LDI R8, 0\n        LDI R9, 1\n        LDI R12, 8\n    loop:\n        MOV R10, R8\n        ADD R10, R9\n        MOV R8, R9\n        MOV R9, R10\n        DEC R12\n        JNZ loop\n        STB [40], R9\n        HALT\n",
        ),
        V2PlayerProgram::new(
            "Print \"HELLO\"  (MMIO console)",
            "        LDI R0, 72\n        STB [57], R0\n        LDI R0, 69\n        STB [57], R0\n        LDI R0, 76\n        STB [57], R0\n        STB [57], R0\n        LDI R0, 79\n        STB [57], R0\n        HALT\n",
        ),
    ]
}

/// Coordinate packing shared with the player JS (`unpack()` in the template).
#[inline]
fn pack(x: usize, y: usize, z: usize) -> u32 {
    ((z * GRID_H + y) * GRID_W + x) as u32
}

/// Bucket a tile type into a floorplan colour category (mirrors the colour
/// groups in [`crate::tile_cpu::v2_visualization`]): 0=route/via, 1=register/RAM,
/// 2=ALU, 3=control/mux, 4=const, 5=other.
fn tile_category(tt: TileType) -> u8 {
    match tt {
        TileType::Wire
        | TileType::WireH
        | TileType::WireV
        | TileType::WireDown
        | TileType::WireRight
        | TileType::WireUp
        | TileType::WireLeft
        | TileType::Cross
        | TileType::WireCross
        | TileType::WireCrossVert
        | TileType::VBusIn
        | TileType::VBusOut
        | TileType::ViaUp
        | TileType::ViaDown => 0,
        TileType::Register8
        | TileType::Register64
        | TileType::Register
        | TileType::RegEnable
        | TileType::Latch
        | TileType::Counter
        | TileType::MemoryPort
        | TileType::Ram => 1,
        TileType::Add
        | TileType::Sub
        | TileType::Mul
        | TileType::Div
        | TileType::Mod
        | TileType::Shl
        | TileType::Shr
        | TileType::AddCarry
        | TileType::SubBorrow
        | TileType::CarryDetect
        | TileType::And
        | TileType::Or
        | TileType::Xor
        | TileType::Not
        | TileType::Neg
        | TileType::Abs
        | TileType::Lt
        | TileType::Gt
        | TileType::Eq
        | TileType::Neq
        | TileType::Lte
        | TileType::Gte => 2,
        TileType::Mux
        | TileType::Mux4to1
        | TileType::Mux8to1
        | TileType::Mux16to1
        | TileType::Decoder3to8
        | TileType::Decoder6to64
        | TileType::ProgramCounter
        | TileType::ClockGlobal
        | TileType::CpuHead
        | TileType::BitSelect
        | TileType::Zero
        | TileType::Selector => 3,
        TileType::Const => 4,
        _ => 5,
    }
}

/// A program captured into player-ready form.
struct CapturedProgram {
    name: String,
    disasm: Vec<String>,
    /// Pre-rendered JSON array of per-step objects.
    steps_json: String,
}

/// Run one program through the real capture path and produce its player data,
/// plus the static floorplan tiles (shared across programs, taken from the first).
fn capture_program(program: &V2PlayerProgram) -> Result<(CapturedProgram, Vec<(u32, u8)>), String> {
    let words =
        assemble_v2(&program.source).map_err(|e| format!("assemble '{}': {e:?}", program.name))?;

    let mut sim = Simulation::with_size_layered(GRID_W, GRID_H, GRID_LAYERS);
    // Attach the reference MMIO device so stores into the MMIO window (e.g. the
    // console at addr 57) are routed and recorded in the trace. Programs that
    // never touch MMIO are unaffected. Fixed seed keeps capture deterministic.
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&words)
        .with_rom_size(ROM_SIZE)
        .with_ram_size(RAM_SIZE)
        .with_mmio(V2MmioHandle::new(V2MmioRefDevicePack::new(0)))
        .build(&mut sim);

    let params = V2VizRenderParams {
        scale: 1,
        layout: V2VizLayerLayout::Grid2x2,
        highlight_active: true,
        show_flow: true,
        max_flow_edges: 512,
    };
    let mut viz = V2VizSession::new(&sim);
    let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, MAX_CYCLES);
    if timeline.is_empty() {
        return Err(format!(
            "program '{}' produced no retired steps",
            program.name
        ));
    }
    if timeline.trace.len() != timeline.frames.len() {
        return Err(format!(
            "program '{}' trace/frame mismatch: {} vs {}",
            program.name,
            timeline.trace.len(),
            timeline.frames.len()
        ));
    }

    // Static floorplan: every non-default tile (the grid is default-filled with
    // plain `Wire`, which we omit so only placed components/routing show).
    let mut structure: Vec<(u32, u8)> = Vec::new();
    let tile_count = sim.tile_count();
    for idx in 0..tile_count {
        let tt = sim.tilemap.tiles[idx].meta.tile_type;
        if tt == TileType::Wire {
            continue;
        }
        let (x, y, z) = sim.tilemap.coords_from_idx(idx);
        if z >= GRID_LAYERS {
            continue;
        }
        structure.push((pack(x, y, z), tile_category(tt)));
    }

    let disasm: Vec<String> = words.iter().map(|&w| disassemble_v2_word(w)).collect();

    let entries = timeline.trace.entries();
    let frames = &timeline.frames;
    let mut steps_json = String::from("[");
    for i in 0..timeline.len() {
        if i > 0 {
            steps_json.push(',');
        }
        steps_json.push_str(&step_json(&entries[i], &frames[i]));
    }
    steps_json.push(']');

    Ok((
        CapturedProgram {
            name: program.name.clone(),
            disasm,
            steps_json,
        },
        structure,
    ))
}

/// One per-step JSON object: instruction context + state deltas + the rendered
/// active-tile and signal-flow geometry for that retired instruction.
fn step_json(e: &V2TraceEntry, f: &V2VizFrame) -> String {
    let regs = pair_list(e.reg_writes.iter().map(|w| (w.reg as u64, w.value)));
    let ram = pair_list(e.ram_writes.iter().map(|m| (m.addr as u64, m.value)));
    let mmio_w = pair_list(e.mmio_writes.iter().map(|m| (m.addr as u64, m.value)));
    let mmio_r = pair_list(e.mmio_reads.iter().map(|m| (m.addr as u64, m.value)));
    let act = packed_list(f.active_tiles.iter().map(|c| pack(c.x, c.y, c.z)));
    let flow = flow_list(&f.flow_edges);
    format!(
        "{{\"pc\":{},\"cy\":{},\"asm\":\"{}\",\"z\":{},\"c\":{},\"regs\":{},\"ram\":{},\"mmioW\":{},\"mmioR\":{},\"act\":{},\"flow\":{}}}",
        e.pc,
        e.cycle,
        json_escape(&e.asm),
        e.flag_z as u8,
        e.flag_c as u8,
        regs,
        ram,
        mmio_w,
        mmio_r,
        act,
        flow
    )
}

/// Assemble the embedded bundle JSON and inject it into the template.
pub fn build_player_html(
    programs: &[V2PlayerProgram],
    default_idx: usize,
) -> Result<String, String> {
    if programs.is_empty() {
        return Err("no programs provided".to_string());
    }

    let mut captured = Vec::with_capacity(programs.len());
    let mut structure: Option<Vec<(u32, u8)>> = None;
    for program in programs {
        let (cap, st) = capture_program(program)?;
        if structure.is_none() {
            structure = Some(st);
        }
        captured.push(cap);
    }
    let structure = structure.expect("at least one program captured");

    let mut json = String::new();
    json.push_str(&format!(
        "{{\"meta\":{{\"w\":{GRID_W},\"h\":{GRID_H},\"layers\":{GRID_LAYERS}}},\"structure\":["
    ));
    for (i, (p, c)) in structure.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("[{p},{c}]"));
    }
    json.push_str("],\"programs\":[");
    for (i, cap) in captured.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"name\":\"{}\",\"disasm\":[",
            json_escape(&cap.name)
        ));
        for (j, d) in cap.disasm.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            json.push('"');
            json.push_str(&json_escape(d));
            json.push('"');
        }
        json.push_str("],\"steps\":");
        json.push_str(&cap.steps_json);
        json.push('}');
    }
    json.push_str(&format!(
        "],\"default\":{}}}",
        default_idx.min(programs.len() - 1)
    ));

    let marker_count = PLAYER_TEMPLATE.matches(PLAYER_DATA_MARKER).count();
    if marker_count != 1 {
        return Err(format!(
            "player template marker count = {marker_count}, expected exactly 1"
        ));
    }
    Ok(PLAYER_TEMPLATE.replace(PLAYER_DATA_MARKER, &json))
}

/// Write a populated player HTML to disk (creating parent dirs as needed).
pub fn write_player_html(path: impl AsRef<Path>, html: &str) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }
    write(path, html)
}

/// Convenience: build the default demo set and write the player to `path`.
pub fn export_default_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_player_html(&default_player_programs(), 0)?;
    write_player_html(path, &html).map_err(|e| e.to_string())
}

// ── JSON helpers (hand-rolled to avoid a serde feature dependency) ─────────────

fn pair_list<I: IntoIterator<Item = (u64, u64)>>(items: I) -> String {
    let mut out = String::from("[");
    let mut first = true;
    for (a, b) in items {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!("[{a},{b}]"));
    }
    out.push(']');
    out
}

fn packed_list<I: IntoIterator<Item = u32>>(items: I) -> String {
    let mut out = String::from("[");
    let mut first = true;
    for p in items {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&p.to_string());
    }
    out.push(']');
    out
}

fn flow_list(edges: &[V2VizFlowEdge]) -> String {
    let mut out = String::from("[");
    for (i, e) in edges.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let from = pack(e.from.x, e.from.y, e.from.z);
        let to = pack(e.to.x, e.to.y, e.to.z);
        out.push_str(&format!("[{from},{to}]"));
    }
    out.push(']');
    out
}

/// JSON string-body escaping (no surrounding quotes).
fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
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
    fn test_pack_roundtrip_matches_template_formula() {
        // Mirror of the template's unpack(): x = p%W; y = (p/W)%H; z = p/(W*H).
        for &(x, y, z) in &[(0usize, 0usize, 0usize), (5, 9, 2), (127, 99, 3)] {
            let p = pack(x, y, z) as usize;
            assert_eq!(p % GRID_W, x);
            assert_eq!((p / GRID_W) % GRID_H, y);
            assert_eq!(p / (GRID_W * GRID_H), z);
        }
    }

    #[test]
    fn test_json_helpers_format() {
        assert_eq!(pair_list([(8u64, 6u64), (9, 0)]), "[[8,6],[9,0]]");
        assert_eq!(pair_list(std::iter::empty::<(u64, u64)>()), "[]");
        assert_eq!(packed_list([1u32, 2, 3]), "[1,2,3]");
        assert_eq!(json_escape("a\"b\\c"), "a\\\"b\\\\c");
    }

    #[test]
    fn test_build_player_html_embeds_real_bundle() {
        // A single tiny program keeps the test well under the time budget.
        let programs = vec![V2PlayerProgram::new(
            "tiny",
            "LDI R8, 7\nLDI R9, 5\nADD R8, R9\nHALT\n",
        )];
        let html = build_player_html(&programs, 0).expect("build should pass");

        // Marker replaced; template shell present.
        assert!(!html.contains(PLAYER_DATA_MARKER));
        assert!(html.contains("Tile<span class=\"accent\">CPU</span>"));
        // Real captured content embedded.
        assert!(html.contains("\"structure\":["));
        assert!(html.contains("\"programs\":["));
        assert!(html.contains("\"name\":\"tiny\""));
        assert!(html.contains("ADD R8, R9"));
        assert!(html.contains("\"act\":["));
    }

    #[test]
    fn test_build_player_html_deterministic() {
        let programs = vec![V2PlayerProgram::new(
            "det",
            "LDI R0, 1\nLDI R1, 2\nADD R0, R1\nHALT\n",
        )];
        let a = build_player_html(&programs, 0).expect("a");
        let b = build_player_html(&programs, 0).expect("b");
        assert_eq!(a, b);
    }
}
