use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tilemap::{HEIGHT, WIDTH};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PatternParams {
    pub blob_threshold: u32,
    pub edge_threshold: u32,
    pub min_blob_area: u32,
    pub max_results: u32,
}

impl Default for PatternParams {
    fn default() -> Self {
        Self {
            blob_threshold: 1,
            edge_threshold: 1,
            min_blob_area: 1,
            max_results: 10,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FieldSelect {
    Power,
    LogicField,
    Clock,
    Heat,
    Charge,
}

#[derive(Debug, Clone)]
pub struct Blob {
    pub id: u32,
    pub area: u32,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub peak: u32,
    pub mean: u32,
}

#[derive(Debug, Clone, Default)]
pub struct EdgeSummary {
    pub count: u32,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

#[derive(Debug, Clone)]
pub struct PatternSummary {
    pub field: String,
    pub blobs: Vec<Blob>,
    pub edges: EdgeSummary,
}

#[derive(Debug, Clone)]
pub struct OscillatorCandidate {
    pub x: u32,
    pub y: u32,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub struct DetectionReport {
    pub field_summaries: Vec<PatternSummary>,
    pub oscillators: Vec<OscillatorCandidate>,
}

fn sample_field(sim: &Simulation, sel: FieldSelect, x: usize, y: usize) -> u32 {
    match sel {
        FieldSelect::Power => sim.get_power_field_for_test(x, y) as u32,
        FieldSelect::LogicField => sim.get_logic_field_for_test(x, y),
        FieldSelect::Clock => sim.get_clock_field_for_test(x, y) as u32,
        FieldSelect::Heat => sim.get_heat_field_for_test(x, y),
        FieldSelect::Charge => sim.get_charge_field_for_test(x, y),
    }
}

pub fn detect_blobs_u32(sim: &Simulation, sel: FieldSelect, params: &PatternParams) -> Vec<Blob> {
    let w = WIDTH;
    let h = HEIGHT;
    let mut visited = vec![false; w * h];
    let mut blobs: Vec<Blob> = Vec::new();
    let mut id_counter: u32 = 1;
    let thr = params.blob_threshold;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] {
                continue;
            }
            let val = sample_field(sim, sel, x, y);
            if val < thr {
                continue;
            }

            // BFS flood fill
            let mut q = VecDeque::new();
            q.push_back((x, y));
            visited[idx] = true;

            let mut area: u32 = 0;
            let mut x0 = x as u32;
            let mut y0 = y as u32;
            let mut x1 = x as u32;
            let mut y1 = y as u32;
            let mut peak: u32 = 0;
            let mut sum: u64 = 0;

            while let Some((cx, cy)) = q.pop_front() {
                area = area.saturating_add(1);
                let v = sample_field(sim, sel, cx, cy);
                if v > peak {
                    peak = v;
                }
                sum = sum.saturating_add(v as u64);
                if (cx as u32) < x0 {
                    x0 = cx as u32;
                }
                if (cy as u32) < y0 {
                    y0 = cy as u32;
                }
                if (cx as u32) > x1 {
                    x1 = cx as u32;
                }
                if (cy as u32) > y1 {
                    y1 = cy as u32;
                }

                let neighbors = [
                    (cx as isize - 1, cy as isize),
                    (cx as isize + 1, cy as isize),
                    (cx as isize, cy as isize - 1),
                    (cx as isize, cy as isize + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx < 0 || ny < 0 {
                        continue;
                    }
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if ux >= w || uy >= h {
                        continue;
                    }
                    let nidx = uy * w + ux;
                    if visited[nidx] {
                        continue;
                    }
                    let nv = sample_field(sim, sel, ux, uy);
                    if nv >= thr {
                        visited[nidx] = true;
                        q.push_back((ux, uy));
                    }
                }
            }

            if area >= params.min_blob_area {
                let mean = if area > 0 {
                    (sum / area as u64) as u32
                } else {
                    0
                };
                blobs.push(Blob {
                    id: id_counter,
                    area,
                    x0,
                    y0,
                    x1,
                    y1,
                    peak,
                    mean,
                });
                id_counter = id_counter.saturating_add(1);
            }
        }
    }

    // Rank deterministically by area desc, then by id asc (first seen)
    blobs.sort_by(|a, b| b.area.cmp(&a.area).then(a.id.cmp(&b.id)));
    if blobs.len() as u32 > params.max_results {
        blobs.truncate(params.max_results as usize);
    }
    blobs
}

pub fn detect_edges_u32(sim: &Simulation, sel: FieldSelect, params: &PatternParams) -> EdgeSummary {
    let w = WIDTH;
    let h = HEIGHT;
    let threshold = params.edge_threshold;
    let mut count: u32 = 0;
    let mut bbox = EdgeSummary {
        count: 0,
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    for y in 0..h {
        for x in 0..w {
            let v = sample_field(sim, sel, x, y);
            let mut max_diff = 0u32;
            let neighbors = [
                (x as isize - 1, y as isize),
                (x as isize + 1, y as isize),
                (x as isize, y as isize - 1),
                (x as isize, y as isize + 1),
            ];
            for (nx, ny) in neighbors {
                if nx < 0 || ny < 0 {
                    continue;
                }
                let ux = nx as usize;
                let uy = ny as usize;
                if ux >= w || uy >= h {
                    continue;
                }
                let nv = sample_field(sim, sel, ux, uy);
                let diff = v.abs_diff(nv);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            if max_diff >= threshold {
                if count == 0 {
                    bbox.x0 = x as u32;
                    bbox.y0 = y as u32;
                    bbox.x1 = x as u32;
                    bbox.y1 = y as u32;
                } else {
                    if (x as u32) < bbox.x0 {
                        bbox.x0 = x as u32;
                    }
                    if (y as u32) < bbox.y0 {
                        bbox.y0 = y as u32;
                    }
                    if (x as u32) > bbox.x1 {
                        bbox.x1 = x as u32;
                    }
                    if (y as u32) > bbox.y1 {
                        bbox.y1 = y as u32;
                    }
                }
                count = count.saturating_add(1);
            }
        }
    }
    bbox.count = count;
    bbox
}

pub fn summarize_field(
    sim: &Simulation,
    sel: FieldSelect,
    params: &PatternParams,
) -> PatternSummary {
    let field_name = match sel {
        FieldSelect::Power => "power",
        FieldSelect::LogicField => "logic_field",
        FieldSelect::Clock => "clock",
        FieldSelect::Heat => "heat",
        FieldSelect::Charge => "charge",
    }
    .to_string();

    let blobs = detect_blobs_u32(sim, sel, params);
    let edges = detect_edges_u32(sim, sel, params);
    PatternSummary {
        field: field_name,
        blobs,
        edges,
    }
}

pub fn detect_oscillators(sim: &Simulation) -> Vec<OscillatorCandidate> {
    let mut out = Vec::new();
    // Conservative heuristic: flag NOT gate with wires left and right as a simple motif.
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if let Some(t) = sim.tilemap.get_tile(x, y)
                && matches!(t.meta.tile_type, TileType::Not)
            {
                let left_ok = x > 0
                    && sim
                        .tilemap
                        .get_tile(x - 1, y)
                        .map(|lt| matches!(lt.meta.tile_type, TileType::Wire))
                        .unwrap_or(false);
                let right_ok = x + 1 < WIDTH
                    && sim
                        .tilemap
                        .get_tile(x + 1, y)
                        .map(|rt| matches!(rt.meta.tile_type, TileType::Wire))
                        .unwrap_or(false);
                if left_ok && right_ok {
                    out.push(OscillatorCandidate {
                        x: x as u32,
                        y: y as u32,
                        reason: "local_not_motif",
                    });
                }
            }
        }
    }
    out
}

pub fn detect_patterns(
    sim: &Simulation,
    fields: &[FieldSelect],
    params: &PatternParams,
) -> DetectionReport {
    let mut summaries = Vec::new();
    for &f in fields {
        summaries.push(summarize_field(sim, f, params));
    }
    let oscillators = detect_oscillators(sim);
    DetectionReport {
        field_summaries: summaries,
        oscillators,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_single_cell() {
        let mut sim = Simulation::new();
        sim.set_heat_field_for_test(1, 1, 5);
        let p = PatternParams {
            blob_threshold: 1,
            edge_threshold: 1,
            min_blob_area: 1,
            max_results: 10,
        };
        let blobs = detect_blobs_u32(&sim, FieldSelect::Heat, &p);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 1);
        assert_eq!((blobs[0].x0, blobs[0].y0), (1, 1));
    }

    #[test]
    fn edges_detect_simple_ridge() {
        let mut sim = Simulation::new();
        for x in 0..3 {
            sim.set_charge_field_for_test(x, 1, 10);
        }
        let p = PatternParams::default();
        let edges = detect_edges_u32(&sim, FieldSelect::Charge, &p);
        assert!(edges.count > 0);
    }

    #[test]
    fn oscillator_candidate_local_not() {
        let mut sim = Simulation::new();
        sim.set_tile(1, 1, TileType::Not);
        sim.set_tile(0, 1, TileType::Wire);
        sim.set_tile(2, 1, TileType::Wire);
        let cands = detect_oscillators(&sim);
        assert!(cands.iter().any(|c| c.x == 1 && c.y == 1));
    }
}
