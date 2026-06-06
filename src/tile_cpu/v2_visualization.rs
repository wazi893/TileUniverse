use std::collections::BTreeSet;
use std::fs::{File, create_dir_all, read_to_string, write};
use std::io::{self, Write};
use std::path::Path;

use crate::gui::render::{ImageBuffer, RgbPixel, write_ppm};
use crate::simulation::Simulation;
use crate::tile_cpu::{TileCpuV2, V2TraceLog};
use crate::tile_meta::TileType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct V2VizCoord {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct V2VizFlowEdge {
    pub from: V2VizCoord,
    pub to: V2VizCoord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2VizLayerLayout {
    Grid2x2,
    HorizontalStrip,
    VerticalStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2VizRenderParams {
    pub scale: u32,
    pub layout: V2VizLayerLayout,
    pub highlight_active: bool,
    pub show_flow: bool,
    pub max_flow_edges: usize,
}

impl Default for V2VizRenderParams {
    fn default() -> Self {
        Self {
            scale: 2,
            layout: V2VizLayerLayout::Grid2x2,
            highlight_active: true,
            show_flow: true,
            max_flow_edges: 2048,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2VizFrameMetrics {
    pub active_count: usize,
    pub flow_edge_count: usize,
    pub active_per_layer: Vec<usize>,
    pub nonzero_logic_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2VizFrame {
    pub tile_width: usize,
    pub tile_height: usize,
    pub layers: usize,
    pub active_tiles: Vec<V2VizCoord>,
    pub flow_edges: Vec<V2VizFlowEdge>,
    pub image: ImageBuffer,
    pub metrics: V2VizFrameMetrics,
}

impl V2VizFrame {
    pub fn checksum(&self) -> u32 {
        let mut hash: u32 = 0x811C9DC5;
        for p in &self.image.pixels {
            hash ^= p.r as u32;
            hash = hash.wrapping_mul(0x01000193);
            hash ^= p.g as u32;
            hash = hash.wrapping_mul(0x01000193);
            hash ^= p.b as u32;
            hash = hash.wrapping_mul(0x01000193);
        }
        hash
    }
}

#[derive(Debug, Clone, Default)]
pub struct V2VizSession {
    prev_logic: Vec<u64>,
}

impl V2VizSession {
    pub fn new(sim: &Simulation) -> Self {
        Self {
            prev_logic: snapshot_logic(sim),
        }
    }

    pub fn reset(&mut self, sim: &Simulation) {
        self.prev_logic = snapshot_logic(sim);
    }

    pub fn capture_frame(&mut self, sim: &Simulation, params: &V2VizRenderParams) -> V2VizFrame {
        let current_logic = snapshot_logic(sim);
        let tile_count = current_logic.len();
        let tile_width = sim.tilemap.width;
        let tile_height = sim.tilemap.height;
        let layers = sim.tilemap.num_layers;

        let mut active_indices = Vec::new();
        for idx in 0..tile_count {
            let prev = self.prev_logic.get(idx).copied().unwrap_or(0);
            if current_logic[idx] != prev {
                active_indices.push(idx);
            }
        }

        let mut active_mask = vec![false; tile_count];
        for &idx in &active_indices {
            active_mask[idx] = true;
        }

        let flow_idx_edges =
            build_flow_edges(sim, &active_mask, params.show_flow, params.max_flow_edges);
        let flow_edges = flow_idx_edges
            .iter()
            .map(|&(from_idx, to_idx)| V2VizFlowEdge {
                from: idx_to_coord(sim, from_idx),
                to: idx_to_coord(sim, to_idx),
            })
            .collect::<Vec<_>>();

        let active_tiles = active_indices
            .iter()
            .map(|&idx| idx_to_coord(sim, idx))
            .collect::<Vec<_>>();

        let mut active_per_layer = vec![0usize; layers];
        for coord in &active_tiles {
            if coord.z < active_per_layer.len() {
                active_per_layer[coord.z] += 1;
            }
        }

        let nonzero_logic_count = current_logic.iter().filter(|&&v| v != 0).count();

        let image = render_layered_image(sim, params, &current_logic, &active_mask, &flow_edges);

        self.prev_logic = current_logic;

        V2VizFrame {
            tile_width,
            tile_height,
            layers,
            active_tiles,
            flow_edges,
            image,
            metrics: V2VizFrameMetrics {
                active_count: active_indices.len(),
                flow_edge_count: flow_idx_edges.len(),
                active_per_layer,
                nonzero_logic_count,
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct V2TraceVizTimeline {
    pub trace: V2TraceLog,
    pub frames: Vec<V2VizFrame>,
}

impl V2TraceVizTimeline {
    pub fn len(&self) -> usize {
        self.trace.len().min(self.frames.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn checksums(&self) -> Vec<u32> {
        self.frames.iter().map(V2VizFrame::checksum).collect()
    }

    pub fn step_summary(&self, idx: usize) -> Option<V2TraceVizStepSummary> {
        let entry = self.trace.entries().get(idx)?;
        let frame = self.frames.get(idx)?;
        Some(V2TraceVizStepSummary {
            retired: idx,
            cycle: entry.cycle,
            pc: entry.pc,
            asm: entry.asm.clone(),
            frame_checksum: frame.checksum(),
            active_tiles: frame.metrics.active_count,
            flow_edges: frame.metrics.flow_edge_count,
            reg_deltas: entry.reg_writes.len(),
            ram_deltas: entry.ram_writes.len(),
            mmio_reads: entry.mmio_reads.len(),
            mmio_writes: entry.mmio_writes.len(),
            frame_file: frame_file_name(idx, entry.cycle, entry.pc),
        })
    }

    pub fn step_summaries(&self) -> Vec<V2TraceVizStepSummary> {
        (0..self.len())
            .filter_map(|idx| self.step_summary(idx))
            .collect()
    }
}

pub const V2_TRACE_VIZ_INDEX_FILE: &str = "index.tsv";
pub const V2_TRACE_VIZ_TRACE_FILE: &str = "trace.log";
pub const V2_TRACE_VIZ_SUMMARY_FILE: &str = "summary.tsv";
const V2_TRACE_VIZ_INDEX_HEADER: &str = "retired\tcycle\tpc\tframe_checksum\tactive_tiles\tflow_edges\treg_deltas\tram_deltas\tmmio_reads\tmmio_writes\tframe_file\tasm";
const V2_TRACE_VIZ_SUMMARY_HEADER: &str = "retired\tcycle\tpc\tasm\tactive_tiles\tflow_edges\treg_deltas\tram_deltas\tmmio_reads\tmmio_writes";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2TraceVizDeltaKind {
    Regs,
    Ram,
    Mmio,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2TraceVizStepSummary {
    pub retired: usize,
    pub cycle: u64,
    pub pc: u32,
    pub asm: String,
    pub frame_checksum: u32,
    pub active_tiles: usize,
    pub flow_edges: usize,
    pub reg_deltas: usize,
    pub ram_deltas: usize,
    pub mmio_reads: usize,
    pub mmio_writes: usize,
    pub frame_file: String,
}

impl V2TraceVizStepSummary {
    pub fn to_tsv_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.retired,
            self.cycle,
            self.pc,
            self.frame_checksum,
            self.active_tiles,
            self.flow_edges,
            self.reg_deltas,
            self.ram_deltas,
            self.mmio_reads,
            self.mmio_writes,
            escape_tsv_field(&self.frame_file),
            escape_tsv_field(&self.asm)
        )
    }

    pub fn from_tsv_line(line: &str) -> Result<Self, String> {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 12 {
            return Err(format!("invalid index line field count: {}", parts.len()));
        }
        Ok(Self {
            retired: parse_usize(parts[0], "retired")?,
            cycle: parse_u64(parts[1], "cycle")?,
            pc: parse_u32(parts[2], "pc")?,
            frame_checksum: parse_u32(parts[3], "frame_checksum")?,
            active_tiles: parse_usize(parts[4], "active_tiles")?,
            flow_edges: parse_usize(parts[5], "flow_edges")?,
            reg_deltas: parse_usize(parts[6], "reg_deltas")?,
            ram_deltas: parse_usize(parts[7], "ram_deltas")?,
            mmio_reads: parse_usize(parts[8], "mmio_reads")?,
            mmio_writes: parse_usize(parts[9], "mmio_writes")?,
            frame_file: unescape_tsv_field(parts[10])?,
            asm: unescape_tsv_field(parts[11])?,
        })
    }

    pub fn step_report_line(&self) -> String {
        format!(
            "step={} cycle={} pc={} asm=\"{}\" checksum=0x{:08X} active={} flow={} regs={} ram={} mmio_r={} mmio_w={}",
            self.retired,
            self.cycle,
            self.pc,
            self.asm,
            self.frame_checksum,
            self.active_tiles,
            self.flow_edges,
            self.reg_deltas,
            self.ram_deltas,
            self.mmio_reads,
            self.mmio_writes
        )
    }
}

/// Capture a trace-synchronized visualization timeline.
///
/// One frame is collected for each retired instruction entry. Frame index `i`
/// aligns with `trace.entries()[i]`.
pub fn capture_v2_trace_viz_timeline(
    cpu: &TileCpuV2,
    sim: &mut Simulation,
    viz: &mut V2VizSession,
    params: &V2VizRenderParams,
    max_cycles: u64,
) -> V2TraceVizTimeline {
    let mut trace = V2TraceLog::new();
    let mut frames = Vec::new();

    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }

        let before = trace.len();
        let _ = cpu.step_with_trace(sim, &mut trace);
        let frame = viz.capture_frame(sim, params);
        if trace.len() > before {
            frames.push(frame);
        }
    }

    V2TraceVizTimeline { trace, frames }
}

pub fn export_v2_trace_viz_bundle(
    timeline: &V2TraceVizTimeline,
    out_dir: impl AsRef<Path>,
) -> io::Result<Vec<V2TraceVizStepSummary>> {
    let out_dir = out_dir.as_ref();
    create_dir_all(out_dir)?;

    write(
        out_dir.join(V2_TRACE_VIZ_TRACE_FILE),
        timeline.trace.to_lines(),
    )?;
    let summaries = timeline.step_summaries();

    let mut index = String::new();
    index.push_str(V2_TRACE_VIZ_INDEX_HEADER);
    index.push('\n');
    for summary in &summaries {
        let frame = timeline.frames.get(summary.retired).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "frame index out of bounds")
        })?;
        let mut file = File::create(out_dir.join(&summary.frame_file))?;
        write_v2_viz_frame_ppm(frame, &mut file)?;
        index.push_str(&summary.to_tsv_line());
        index.push('\n');
    }
    write(out_dir.join(V2_TRACE_VIZ_INDEX_FILE), index)?;
    Ok(summaries)
}

pub fn load_v2_trace_viz_bundle_summaries(
    out_dir: impl AsRef<Path>,
) -> io::Result<Vec<V2TraceVizStepSummary>> {
    let out_dir = out_dir.as_ref();
    let index_text = read_to_string(out_dir.join(V2_TRACE_VIZ_INDEX_FILE))?;
    let mut lines = index_text.lines();
    let Some(header) = lines.next() else {
        return Ok(Vec::new());
    };
    if header != V2_TRACE_VIZ_INDEX_HEADER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected index header: '{header}'"),
        ));
    }

    let mut out = Vec::new();
    for (line_idx, line) in lines.enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = V2TraceVizStepSummary::from_tsv_line(trimmed).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid index row {}: {e}", line_idx + 2),
            )
        })?;
        out.push(parsed);
    }
    Ok(out)
}

pub fn load_v2_trace_viz_trace_text(out_dir: impl AsRef<Path>) -> io::Result<String> {
    read_to_string(out_dir.as_ref().join(V2_TRACE_VIZ_TRACE_FILE))
}

pub fn write_v2_trace_viz_summary_tsv(
    summaries: &[V2TraceVizStepSummary],
    out_path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut out = String::new();
    out.push_str(V2_TRACE_VIZ_SUMMARY_HEADER);
    out.push('\n');
    for summary in summaries {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            summary.retired,
            summary.cycle,
            summary.pc,
            escape_tsv_field(summary.asm.as_str()),
            summary.active_tiles,
            summary.flow_edges,
            summary.reg_deltas,
            summary.ram_deltas,
            summary.mmio_reads,
            summary.mmio_writes
        ));
    }
    write(out_path, out)
}

pub fn v2_viz_find_next_by_pc(
    summaries: &[V2TraceVizStepSummary],
    start_idx: usize,
    pc: u32,
) -> Option<usize> {
    if summaries.is_empty() {
        return None;
    }
    for (idx, summary) in summaries
        .iter()
        .enumerate()
        .skip(start_idx.saturating_add(1))
    {
        if summary.pc == pc {
            return Some(idx);
        }
    }
    None
}

pub fn v2_viz_find_next_by_asm(
    summaries: &[V2TraceVizStepSummary],
    start_idx: usize,
    needle: &str,
) -> Option<usize> {
    if summaries.is_empty() {
        return None;
    }
    let query = needle.trim().to_ascii_lowercase();
    if query.is_empty() {
        return None;
    }

    for (idx, summary) in summaries
        .iter()
        .enumerate()
        .skip(start_idx.saturating_add(1))
    {
        if summary.asm.to_ascii_lowercase().contains(query.as_str()) {
            return Some(idx);
        }
    }
    None
}

pub fn v2_viz_find_next_by_delta(
    summaries: &[V2TraceVizStepSummary],
    start_idx: usize,
    kind: V2TraceVizDeltaKind,
) -> Option<usize> {
    if summaries.is_empty() {
        return None;
    }

    for (idx, summary) in summaries
        .iter()
        .enumerate()
        .skip(start_idx.saturating_add(1))
    {
        if summary_has_delta(summary, kind) {
            return Some(idx);
        }
    }
    None
}

pub fn v2_viz_find_prev_by_delta(
    summaries: &[V2TraceVizStepSummary],
    start_idx: usize,
    kind: V2TraceVizDeltaKind,
) -> Option<usize> {
    if summaries.is_empty() {
        return None;
    }
    let mut idx = start_idx.min(summaries.len().saturating_sub(1));
    while idx > 0 {
        idx -= 1;
        if summary_has_delta(&summaries[idx], kind) {
            return Some(idx);
        }
    }
    None
}

pub fn write_v2_viz_frame_ppm(frame: &V2VizFrame, mut w: impl Write) -> io::Result<()> {
    write_ppm(&frame.image, &mut w)
}

pub fn v2_viz_frame_to_ppm_bytes(frame: &V2VizFrame) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_v2_viz_frame_ppm(frame, &mut buf)?;
    Ok(buf)
}

fn frame_file_name(retired: usize, cycle: u64, pc: u32) -> String {
    format!("retired_{retired:03}_cycle_{cycle:04}_pc_{pc:04X}.ppm")
}

fn summary_has_delta(summary: &V2TraceVizStepSummary, kind: V2TraceVizDeltaKind) -> bool {
    match kind {
        V2TraceVizDeltaKind::Regs => summary.reg_deltas > 0,
        V2TraceVizDeltaKind::Ram => summary.ram_deltas > 0,
        V2TraceVizDeltaKind::Mmio => summary.mmio_reads > 0 || summary.mmio_writes > 0,
    }
}

fn escape_tsv_field(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(ch),
        }
    }
    out
}

fn unescape_tsv_field(input: &str) -> Result<String, String> {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        let Some(next) = chars.next() else {
            return Err("dangling escape sequence".to_string());
        };
        match next {
            '\\' => out.push('\\'),
            't' => out.push('\t'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            other => {
                return Err(format!("unsupported escape sequence: \\{other}"));
            }
        }
    }
    Ok(out)
}

fn parse_usize(raw: &str, field: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|e| format!("invalid {field} '{raw}': {e}"))
}

fn parse_u64(raw: &str, field: &str) -> Result<u64, String> {
    raw.parse::<u64>()
        .map_err(|e| format!("invalid {field} '{raw}': {e}"))
}

fn parse_u32(raw: &str, field: &str) -> Result<u32, String> {
    raw.parse::<u32>()
        .map_err(|e| format!("invalid {field} '{raw}': {e}"))
}

fn snapshot_logic(sim: &Simulation) -> Vec<u64> {
    let tile_count = sim.tile_count();
    let mut out = Vec::with_capacity(tile_count);
    for idx in 0..tile_count {
        out.push(sim.get_logic_value_by_idx(idx));
    }
    out
}

fn idx_to_coord(sim: &Simulation, idx: usize) -> V2VizCoord {
    let (x, y, z) = sim.tilemap.coords_from_idx(idx);
    V2VizCoord { x, y, z }
}

fn build_flow_edges(
    sim: &Simulation,
    active_mask: &[bool],
    show_flow: bool,
    max_flow_edges: usize,
) -> Vec<(usize, usize)> {
    if !show_flow || max_flow_edges == 0 {
        return Vec::new();
    }

    let tile_count = sim.tile_count();
    let mut edges = BTreeSet::new();

    for idx in 0..tile_count {
        if idx >= active_mask.len() || !active_mask[idx] {
            continue;
        }
        if edges.len() >= max_flow_edges {
            break;
        }

        let coord = idx_to_coord(sim, idx);
        let tt = sim.tilemap.tiles[idx].meta.tile_type;

        if let Some(change) = sim.last_change.get(idx).and_then(|entry| entry.as_ref()) {
            for neighbor in change.neighbors.iter().flatten() {
                let n_idx = *neighbor;
                if n_idx >= tile_count || n_idx >= active_mask.len() || !active_mask[n_idx] {
                    continue;
                }
                if !is_input_candidate(sim, tt, coord, n_idx) {
                    continue;
                }
                edges.insert((n_idx, idx));
                if edges.len() >= max_flow_edges {
                    break;
                }
            }
        }

        if edges.len() >= max_flow_edges {
            break;
        }

        if tt == TileType::ViaUp && coord.z + 1 < sim.tilemap.num_layers {
            let src_idx = idx + sim.tilemap.layer_size;
            if src_idx < tile_count && src_idx < active_mask.len() && active_mask[src_idx] {
                edges.insert((src_idx, idx));
            }
        } else if tt == TileType::ViaDown && coord.z > 0 {
            let src_idx = idx - sim.tilemap.layer_size;
            if src_idx < tile_count && src_idx < active_mask.len() && active_mask[src_idx] {
                edges.insert((src_idx, idx));
            }
        }
    }

    edges.into_iter().take(max_flow_edges).collect()
}

fn is_input_candidate(sim: &Simulation, tt: TileType, to: V2VizCoord, from_idx: usize) -> bool {
    let from = idx_to_coord(sim, from_idx);
    if from.z != to.z {
        return false;
    }

    let is_left = from.y == to.y && from.x + 1 == to.x;
    let is_right = from.y == to.y && to.x + 1 == from.x;
    let is_up = from.x == to.x && from.y + 1 == to.y;
    let is_down = from.x == to.x && to.y + 1 == from.y;

    match tt {
        TileType::WireDown => is_up,
        TileType::WireUp => is_down,
        TileType::WireRight => is_left,
        TileType::WireLeft => is_right,
        TileType::WireH => is_left || is_right,
        TileType::WireV => is_up || is_down,
        _ => is_left || is_right || is_up || is_down,
    }
}

fn render_layered_image(
    sim: &Simulation,
    params: &V2VizRenderParams,
    logic: &[u64],
    active_mask: &[bool],
    flow_edges: &[V2VizFlowEdge],
) -> ImageBuffer {
    let scale = params.scale.max(1);
    let tile_w = sim.tilemap.width;
    let tile_h = sim.tilemap.height;
    let layers = sim.tilemap.num_layers.max(1);

    let (layer_cols, layer_rows) = layer_grid(layers, params.layout);
    let canvas_w_tiles = layer_cols * tile_w;
    let canvas_h_tiles = layer_rows * tile_h;
    let out_w = (canvas_w_tiles as u32).saturating_mul(scale);
    let out_h = (canvas_h_tiles as u32).saturating_mul(scale);

    let mut img = ImageBuffer {
        width: out_w,
        height: out_h,
        pixels: vec![RgbPixel { r: 0, g: 0, b: 0 }; (out_w as usize) * (out_h as usize)],
    };

    for idx in 0..logic.len() {
        let coord = idx_to_coord(sim, idx);
        let tile = sim.tilemap.tiles[idx].meta.tile_type;
        let mut color = tile_color(tile, logic[idx] != 0);
        if params.highlight_active && idx < active_mask.len() && active_mask[idx] {
            color = blend_add(
                color,
                RgbPixel {
                    r: 120,
                    g: 90,
                    b: 0,
                },
            );
        }
        fill_scaled_tile(
            &mut img, coord, color, scale, tile_w, tile_h, layer_cols, layer_rows,
        );
    }

    if params.show_flow {
        for edge in flow_edges {
            draw_flow_edge(
                &mut img,
                edge,
                scale,
                tile_w,
                tile_h,
                layer_cols,
                layer_rows,
                RgbPixel {
                    r: 20,
                    g: 220,
                    b: 220,
                },
            );
        }
    }

    img
}

fn layer_grid(layers: usize, layout: V2VizLayerLayout) -> (usize, usize) {
    match layout {
        V2VizLayerLayout::Grid2x2 => {
            let cols = 2usize;
            let rows = layers.div_ceil(cols);
            (cols, rows.max(1))
        }
        V2VizLayerLayout::HorizontalStrip => (layers, 1),
        V2VizLayerLayout::VerticalStrip => (1, layers),
    }
}

fn tile_color(tt: TileType, logic_nonzero: bool) -> RgbPixel {
    let mut c = match tt {
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
        | TileType::ViaDown => RgbPixel {
            r: 40,
            g: 90,
            b: 185,
        },
        TileType::Register8
        | TileType::Register64
        | TileType::Register
        | TileType::RegEnable
        | TileType::Latch
        | TileType::Counter
        | TileType::MemoryPort
        | TileType::Ram => RgbPixel {
            r: 45,
            g: 165,
            b: 90,
        },
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
        | TileType::Gte => RgbPixel {
            r: 190,
            g: 70,
            b: 55,
        },
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
        | TileType::Selector => RgbPixel {
            r: 200,
            g: 145,
            b: 45,
        },
        TileType::Const => RgbPixel {
            r: 95,
            g: 95,
            b: 95,
        },
        _ => RgbPixel {
            r: 120,
            g: 120,
            b: 120,
        },
    };

    if logic_nonzero {
        c = brighten(c, 40);
    } else {
        c = dim(c, 20);
    }
    c
}

fn brighten(c: RgbPixel, amt: u8) -> RgbPixel {
    RgbPixel {
        r: c.r.saturating_add(amt),
        g: c.g.saturating_add(amt),
        b: c.b.saturating_add(amt),
    }
}

fn dim(c: RgbPixel, amt: u8) -> RgbPixel {
    RgbPixel {
        r: c.r.saturating_sub(amt),
        g: c.g.saturating_sub(amt),
        b: c.b.saturating_sub(amt),
    }
}

fn blend_add(a: RgbPixel, b: RgbPixel) -> RgbPixel {
    RgbPixel {
        r: a.r.saturating_add(b.r),
        g: a.g.saturating_add(b.g),
        b: a.b.saturating_add(b.b),
    }
}

fn fill_scaled_tile(
    img: &mut ImageBuffer,
    coord: V2VizCoord,
    color: RgbPixel,
    scale: u32,
    tile_w: usize,
    tile_h: usize,
    layer_cols: usize,
    layer_rows: usize,
) {
    if coord.z >= layer_cols * layer_rows {
        return;
    }

    let (tile_px, tile_py) = tile_origin(coord, tile_w, tile_h, layer_cols, layer_rows);
    let base_x = (tile_px as u32).saturating_mul(scale) as usize;
    let base_y = (tile_py as u32).saturating_mul(scale) as usize;
    let sw = scale as usize;
    let stride = img.width as usize;

    for dy in 0..sw {
        let y = base_y + dy;
        if y >= img.height as usize {
            continue;
        }
        let row = y * stride;
        for dx in 0..sw {
            let x = base_x + dx;
            if x >= img.width as usize {
                continue;
            }
            img.pixels[row + x] = color;
        }
    }
}

fn tile_origin(
    coord: V2VizCoord,
    tile_w: usize,
    tile_h: usize,
    layer_cols: usize,
    _layer_rows: usize,
) -> (usize, usize) {
    let layer_col = coord.z % layer_cols;
    let layer_row = coord.z / layer_cols;
    (layer_col * tile_w + coord.x, layer_row * tile_h + coord.y)
}

fn draw_flow_edge(
    img: &mut ImageBuffer,
    edge: &V2VizFlowEdge,
    scale: u32,
    tile_w: usize,
    tile_h: usize,
    layer_cols: usize,
    layer_rows: usize,
    color: RgbPixel,
) {
    let (sx, sy) = tile_center(edge.from, scale, tile_w, tile_h, layer_cols, layer_rows);
    let (tx, ty) = tile_center(edge.to, scale, tile_w, tile_h, layer_cols, layer_rows);
    draw_line(img, sx, sy, tx, ty, color);
}

fn tile_center(
    coord: V2VizCoord,
    scale: u32,
    tile_w: usize,
    tile_h: usize,
    layer_cols: usize,
    layer_rows: usize,
) -> (i32, i32) {
    let (tile_px, tile_py) = tile_origin(coord, tile_w, tile_h, layer_cols, layer_rows);
    let px = (tile_px as u32)
        .saturating_mul(scale)
        .saturating_add(scale / 2) as i32;
    let py = (tile_py as u32)
        .saturating_mul(scale)
        .saturating_add(scale / 2) as i32;
    (px, py)
}

fn draw_line(img: &mut ImageBuffer, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: RgbPixel) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        put_pixel(img, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err.saturating_mul(2);
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put_pixel(img: &mut ImageBuffer, x: i32, y: i32, color: RgbPixel) {
    if x < 0 || y < 0 {
        return;
    }
    let ux = x as usize;
    let uy = y as usize;
    if ux >= img.width as usize || uy >= img.height as usize {
        return;
    }
    let idx = uy * img.width as usize + ux;
    img.pixels[idx] = blend_add(img.pixels[idx], color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;
    use crate::tile_cpu::V2Builder;
    use crate::tile_cpu::v2_components::{v2_add, v2_halt, v2_ldi};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn build_v2(program: &[u32]) -> (Simulation, crate::tile_cpu::TileCpuV2) {
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(program)
            .with_rom_size(64)
            .with_ram_size(64)
            .build(&mut sim);
        (sim, cpu)
    }

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "engine_v2_trace_viz_{label}_{}_{}",
            std::process::id(),
            nanos
        ));
        p
    }

    #[test]
    fn test_v2_viz_initial_capture_has_no_activity() {
        let (sim, _cpu) = build_v2(&[v2_halt()]);
        let mut viz = V2VizSession::new(&sim);
        let frame = viz.capture_frame(&sim, &V2VizRenderParams::default());
        assert_eq!(frame.metrics.active_count, 0);
        assert_eq!(frame.layers, 4);
        assert_eq!(frame.image.width, 512);
        assert_eq!(frame.image.height, 512);
    }

    #[test]
    fn test_v2_viz_step_activity_bounds() {
        let (mut sim, cpu) = build_v2(&[v2_ldi(0, 7), v2_halt()]);
        let mut viz = V2VizSession::new(&sim);

        let _ = cpu.step(&mut sim);
        let _ = viz.capture_frame(&sim, &V2VizRenderParams::default());
        let _ = cpu.step(&mut sim);
        let frame = viz.capture_frame(&sim, &V2VizRenderParams::default());
        assert!(frame.metrics.active_count > 0);
        for c in &frame.active_tiles {
            assert!(c.x < 128);
            assert!(c.y < 128);
            assert!(c.z < 4);
        }
    }

    #[test]
    fn test_v2_viz_deterministic_checksums() {
        fn run_once() -> Vec<u32> {
            let (mut sim, cpu) = build_v2(&[v2_ldi(0, 1), v2_ldi(1, 2), v2_add(0, 1), v2_halt()]);
            let mut viz = V2VizSession::new(&sim);
            let params = V2VizRenderParams {
                scale: 1,
                ..V2VizRenderParams::default()
            };

            let mut sums = Vec::new();
            for _ in 0..8 {
                let _ = cpu.step(&mut sim);
                let frame = viz.capture_frame(&sim, &params);
                sums.push(frame.checksum());
                if cpu.is_halted() {
                    break;
                }
            }
            sums
        }

        let a = run_once();
        let b = run_once();
        assert_eq!(a, b);
    }

    #[test]
    fn test_v2_viz_flow_edges_capped() {
        let (mut sim, cpu) = build_v2(&[v2_ldi(0, 3), v2_ldi(1, 4), v2_add(0, 1), v2_halt()]);
        let mut viz = V2VizSession::new(&sim);
        let params = V2VizRenderParams {
            max_flow_edges: 8,
            ..V2VizRenderParams::default()
        };

        let _ = cpu.step(&mut sim);
        let frame = viz.capture_frame(&sim, &params);
        assert!(frame.flow_edges.len() <= 8);
    }

    #[test]
    fn test_v2_trace_viz_timeline_alignment() {
        let (mut sim, cpu) = build_v2(&[v2_ldi(0, 1), v2_ldi(1, 2), v2_add(0, 1), v2_halt()]);
        let mut viz = V2VizSession::new(&sim);
        let params = V2VizRenderParams {
            scale: 1,
            ..V2VizRenderParams::default()
        };

        let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, 32);
        assert!(!timeline.is_empty());
        assert_eq!(timeline.trace.len(), timeline.frames.len());
        assert_eq!(timeline.trace.entries()[0].asm, "LDI R0, 1");
        assert_eq!(timeline.trace.entries()[1].asm, "LDI R1, 2");
        assert_eq!(timeline.trace.entries()[2].asm, "ADD R0, R1");
        assert_eq!(timeline.trace.entries()[3].asm, "HALT");
    }

    #[test]
    fn test_v2_trace_viz_timeline_deterministic_checksums() {
        fn run_once() -> (String, Vec<u32>) {
            let (mut sim, cpu) = build_v2(&[v2_ldi(0, 1), v2_ldi(1, 2), v2_add(0, 1), v2_halt()]);
            let mut viz = V2VizSession::new(&sim);
            let params = V2VizRenderParams {
                scale: 1,
                ..V2VizRenderParams::default()
            };
            let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, 32);
            (timeline.trace.to_lines(), timeline.checksums())
        }

        let a = run_once();
        let b = run_once();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn test_v2_trace_viz_bundle_export_load_roundtrip() {
        let (mut sim, cpu) = build_v2(&[v2_ldi(0, 1), v2_ldi(1, 2), v2_add(0, 1), v2_halt()]);
        let mut viz = V2VizSession::new(&sim);
        let params = V2VizRenderParams {
            scale: 1,
            ..V2VizRenderParams::default()
        };
        let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, 32);
        let out_dir = unique_temp_dir("bundle");
        let summaries =
            export_v2_trace_viz_bundle(&timeline, &out_dir).expect("export should pass");
        assert_eq!(summaries.len(), timeline.len());

        let loaded = load_v2_trace_viz_bundle_summaries(&out_dir).expect("load should pass");
        assert_eq!(loaded, summaries);
        let trace_text = load_v2_trace_viz_trace_text(&out_dir).expect("trace text should load");
        assert_eq!(trace_text, timeline.trace.to_lines());

        for summary in &loaded {
            assert!(out_dir.join(&summary.frame_file).exists());
        }

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_v2_trace_viz_step_summary_reports_fields() {
        let (mut sim, cpu) = build_v2(&[v2_ldi(0, 1), v2_halt()]);
        let mut viz = V2VizSession::new(&sim);
        let params = V2VizRenderParams {
            scale: 1,
            ..V2VizRenderParams::default()
        };
        let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, 16);
        let summary = timeline
            .step_summary(0)
            .expect("timeline must have at least one step");
        let report = summary.step_report_line();
        assert!(report.contains("step=0"));
        assert!(report.contains("asm=\"LDI R0, 1\""));
        assert!(report.contains("checksum=0x"));
    }

    #[test]
    fn test_v2_trace_viz_search_helpers_stable_index_mapping() {
        let summaries = vec![
            V2TraceVizStepSummary {
                retired: 0,
                cycle: 1,
                pc: 0,
                asm: "LDI R0, 1".to_string(),
                frame_checksum: 1,
                active_tiles: 10,
                flow_edges: 5,
                reg_deltas: 1,
                ram_deltas: 0,
                mmio_reads: 0,
                mmio_writes: 0,
                frame_file: "a.ppm".to_string(),
            },
            V2TraceVizStepSummary {
                retired: 1,
                cycle: 2,
                pc: 2,
                asm: "STB [40], R0".to_string(),
                frame_checksum: 2,
                active_tiles: 10,
                flow_edges: 5,
                reg_deltas: 0,
                ram_deltas: 1,
                mmio_reads: 0,
                mmio_writes: 0,
                frame_file: "b.ppm".to_string(),
            },
            V2TraceVizStepSummary {
                retired: 2,
                cycle: 3,
                pc: 4,
                asm: "LDB R1, [56]".to_string(),
                frame_checksum: 3,
                active_tiles: 10,
                flow_edges: 5,
                reg_deltas: 1,
                ram_deltas: 0,
                mmio_reads: 1,
                mmio_writes: 0,
                frame_file: "c.ppm".to_string(),
            },
        ];

        assert_eq!(v2_viz_find_next_by_pc(&summaries, 0, 4), Some(2));
        assert_eq!(v2_viz_find_next_by_pc(&summaries, 2, 4), None);

        assert_eq!(v2_viz_find_next_by_asm(&summaries, 0, "ldb"), Some(2));
        assert_eq!(v2_viz_find_next_by_asm(&summaries, 0, "missing"), None);

        assert_eq!(
            v2_viz_find_next_by_delta(&summaries, 0, V2TraceVizDeltaKind::Ram),
            Some(1)
        );
        assert_eq!(
            v2_viz_find_next_by_delta(&summaries, 0, V2TraceVizDeltaKind::Mmio),
            Some(2)
        );
        assert_eq!(
            v2_viz_find_prev_by_delta(&summaries, 2, V2TraceVizDeltaKind::Regs),
            Some(0)
        );
    }

    #[test]
    fn test_v2_trace_viz_summary_export_file() {
        let summaries = vec![
            V2TraceVizStepSummary {
                retired: 0,
                cycle: 1,
                pc: 0,
                asm: "LDI R0, 1".to_string(),
                frame_checksum: 1,
                active_tiles: 4,
                flow_edges: 2,
                reg_deltas: 1,
                ram_deltas: 0,
                mmio_reads: 0,
                mmio_writes: 0,
                frame_file: "f0.ppm".to_string(),
            },
            V2TraceVizStepSummary {
                retired: 1,
                cycle: 2,
                pc: 1,
                asm: "HALT".to_string(),
                frame_checksum: 2,
                active_tiles: 3,
                flow_edges: 1,
                reg_deltas: 0,
                ram_deltas: 0,
                mmio_reads: 0,
                mmio_writes: 0,
                frame_file: "f1.ppm".to_string(),
            },
        ];
        let out_dir = unique_temp_dir("summary");
        std::fs::create_dir_all(&out_dir).expect("create out_dir");
        let out_path = out_dir.join(V2_TRACE_VIZ_SUMMARY_FILE);
        write_v2_trace_viz_summary_tsv(&summaries, &out_path).expect("summary write should pass");

        let text = std::fs::read_to_string(&out_path).expect("summary read should pass");
        assert!(text.starts_with(V2_TRACE_VIZ_SUMMARY_HEADER));
        assert!(text.contains("LDI R0, 1"));
        assert!(text.contains("\n1\t2\t1\tHALT\t"));

        let _ = std::fs::remove_dir_all(&out_dir);
    }
}
