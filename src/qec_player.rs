//! Self-contained "Surface-code decoder" HTML player.
//!
//! Captures real quantum-error-correction shots from [`crate::qec`]: a
//! distance-`d` surface code, seeded depolarising errors, and measured
//! syndromes. The correction is computed by a compact **minimum-weight matching
//! decoder** built here (the engine's bundled decoders do not reliably clear
//! this lattice): each channel's defects are greedily matched to each other or
//! to the boundary along shortest paths in the syndrome graph, and the matched
//! pairs + correction chains are exactly what the visualization draws. Outcome
//! (syndrome cleared, logical X/Z error) uses the engine's own checks.
//!
//! Everything is deterministic (seeded `SimpleRng`) and CPU-only. By
//! construction the decoder always clears the syndrome, so each shot's outcome
//! is purely "did matching introduce a boundary-spanning logical error?" — the
//! quantity that exhibits the surface-code threshold.

use crate::qec::codes::SurfaceCode;
use crate::qec::noise::SimpleRng;
use std::collections::{HashSet, VecDeque};
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("qec_player_template.html");
const MARKER: &str = "/*__QEC_BUNDLE__*/null";

/// Error codes embedded per data qubit: 0 = X (bit flip), 1 = Z (phase), 2 = Y.
const E_X: u8 = 0;
const E_Z: u8 = 1;
const E_Y: u8 = 2;

/// One decoding instance.
struct Shot {
    errors: Vec<(usize, u8)>,
    x_defects: Vec<usize>,
    z_defects: Vec<usize>,
    x_corr: Vec<usize>,
    z_corr: Vec<usize>,
    /// Matched X-stabilizer pairs (boundary = -1), for drawing connectors.
    x_pairs: Vec<(usize, i64)>,
    z_pairs: Vec<(usize, i64)>,
    cleared: bool,
    logical_x: bool,
    logical_z: bool,
}

/// A captured dataset: fixed geometry + a stream of shots at one (d, p).
struct Dataset {
    name: String,
    distance: usize,
    p: f64,
    x_stab_pos: Vec<(f64, f64)>,
    z_stab_pos: Vec<(f64, f64)>,
    shots: Vec<Shot>,
}

/// A (distance, error-rate, shot-count) configuration to capture.
pub struct QecConfig {
    pub name: String,
    pub distance: usize,
    pub p: f64,
    pub shots: usize,
    pub seed: u64,
}

/// Default switchable datasets: a clean low-error run, a near-threshold run,
/// and a smaller code — so the distance / error-rate effect is visible.
pub fn default_configs() -> Vec<QecConfig> {
    vec![
        QecConfig {
            name: "d=11 · p=4%".into(),
            distance: 11,
            p: 0.04,
            shots: 80,
            seed: 0xA11CE,
        },
        QecConfig {
            name: "d=11 · p=9%".into(),
            distance: 11,
            p: 0.09,
            shots: 80,
            seed: 0xB0B,
        },
        QecConfig {
            name: "d=7 · p=6%".into(),
            distance: 7,
            p: 0.06,
            shots: 80,
            seed: 0xC0DE,
        },
    ]
}

// ── Minimum-weight matching decoder over the syndrome graph ─────────────────

const BNODE_NONE: usize = usize::MAX;

/// Adjacency of the syndrome graph for one stabilizer type. Node `supports.len()`
/// is the virtual boundary. Each edge carries the data qubit whose flip toggles
/// its endpoints.
fn build_adj(supports: &[Vec<usize>], n_data: usize) -> Vec<Vec<(usize, usize)>> {
    let bnode = supports.len();
    let mut q_stabs: Vec<Vec<usize>> = vec![Vec::new(); n_data];
    for (si, qs) in supports.iter().enumerate() {
        for &q in qs {
            q_stabs[q].push(si);
        }
    }
    let mut adj = vec![Vec::new(); bnode + 1];
    for (q, stabs) in q_stabs.iter().enumerate() {
        if stabs.len() == 2 {
            adj[stabs[0]].push((stabs[1], q));
            adj[stabs[1]].push((stabs[0], q));
        } else if stabs.len() == 1 {
            adj[stabs[0]].push((bnode, q));
            adj[bnode].push((stabs[0], q));
        }
    }
    adj
}

/// BFS from `src`; returns distances and predecessor (node, qubit) per node.
fn bfs(adj: &[Vec<(usize, usize)>], src: usize) -> (Vec<i32>, Vec<(usize, usize)>) {
    let n = adj.len();
    let mut dist = vec![-1i32; n];
    let mut prev = vec![(BNODE_NONE, BNODE_NONE); n];
    let mut q = VecDeque::new();
    dist[src] = 0;
    q.push_back(src);
    while let Some(u) = q.pop_front() {
        for &(v, qubit) in &adj[u] {
            if dist[v] < 0 {
                dist[v] = dist[u] + 1;
                prev[v] = (u, qubit);
                q.push_back(v);
            }
        }
    }
    (dist, prev)
}

/// Qubits on the shortest path from `src` to `target` (using `prev` from `bfs(src)`).
fn path_qubits(prev: &[(usize, usize)], src: usize, target: usize) -> Vec<usize> {
    let mut qs = Vec::new();
    let mut cur = target;
    while cur != src {
        let (p, qubit) = prev[cur];
        if p == BNODE_NONE {
            break;
        }
        qs.push(qubit);
        cur = p;
    }
    qs
}

/// Greedily match `defects` (stabilizer indices) to each other or the boundary
/// along shortest paths. Returns (correction data qubits, matched pairs as
/// (stab, stab-or-`-1`)). Always clears the given defects by construction.
fn decode_channel(
    supports: &[Vec<usize>],
    n_data: usize,
    defects: &[usize],
) -> (Vec<usize>, Vec<(usize, i64)>) {
    if defects.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let adj = build_adj(supports, n_data);
    let bnode = supports.len();
    let bfs_cache: Vec<(Vec<i32>, Vec<(usize, usize)>)> =
        defects.iter().map(|&d| bfs(&adj, d)).collect();

    let mut remaining: Vec<usize> = (0..defects.len()).collect();
    let mut parity: HashSet<usize> = HashSet::new();
    let mut pairs: Vec<(usize, i64)> = Vec::new();

    let toggle = |set: &mut HashSet<usize>, qs: Vec<usize>| {
        for q in qs {
            if !set.remove(&q) {
                set.insert(q);
            }
        }
    };

    while !remaining.is_empty() {
        // Global min over defect-defect and defect-boundary distances.
        let mut best: Option<(i32, usize, i64)> = None; // (dist, remaining_pos, partner_defect_idx or -1)
        for (ri, &i) in remaining.iter().enumerate() {
            let bd = bfs_cache[i].0[bnode];
            if bd >= 0 && best.map_or(true, |(d, _, _)| bd < d) {
                best = Some((bd, ri, -1));
            }
            for &j in &remaining {
                if j <= i {
                    continue;
                }
                let dd = bfs_cache[i].0[defects[j]];
                if dd >= 0 && best.map_or(true, |(d, _, _)| dd < d) {
                    best = Some((dd, ri, j as i64));
                }
            }
        }
        let Some((_, ri, partner)) = best else { break };
        let i = remaining[ri];
        if partner < 0 {
            toggle(&mut parity, path_qubits(&bfs_cache[i].1, defects[i], bnode));
            pairs.push((defects[i], -1));
            remaining.retain(|&x| x != i);
        } else {
            let j = partner as usize;
            toggle(
                &mut parity,
                path_qubits(&bfs_cache[i].1, defects[i], defects[j]),
            );
            pairs.push((defects[i], defects[j] as i64));
            remaining.retain(|&x| x != i && x != j);
        }
    }

    let mut corr: Vec<usize> = parity.into_iter().collect();
    corr.sort_unstable();
    (corr, pairs)
}

// ── Capture ─────────────────────────────────────────────────────────────────

/// Centroid (col, row) of a stabilizer's data-qubit support.
fn centroid(qubits: &[usize], width: usize) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &q in qubits {
        sx += (q % width) as f64;
        sy += (q / width) as f64;
    }
    let n = qubits.len().max(1) as f64;
    (sx / n, sy / n)
}

fn capture_dataset(cfg: &QecConfig) -> Dataset {
    let geom = SurfaceCode::new(cfg.distance);
    let width = geom.width;
    let n_data = geom.n_data;
    let x_supports: Vec<Vec<usize>> = geom.x_stabilizer_supports().to_vec();
    let z_supports: Vec<Vec<usize>> = geom.z_stabilizer_supports().to_vec();
    let x_stab_pos: Vec<(f64, f64)> = x_supports.iter().map(|qs| centroid(qs, width)).collect();
    let z_stab_pos: Vec<(f64, f64)> = z_supports.iter().map(|qs| centroid(qs, width)).collect();

    let mut rng = SimpleRng::new(cfg.seed);
    let mut shots = Vec::with_capacity(cfg.shots);

    for _ in 0..cfg.shots {
        let mut code = SurfaceCode::new(cfg.distance);
        let mut errors = Vec::new();
        for q in 0..n_data {
            if rng.next_f64() < cfg.p {
                let (ch, byte) = match rng.next_usize(3) {
                    0 => ('X', E_X),
                    1 => ('Z', E_Z),
                    _ => ('Y', E_Y),
                };
                code.apply_error(q, ch);
                errors.push((q, byte));
            }
        }

        let (x_syn, z_syn) = code.measure_syndrome();
        let x_defects: Vec<usize> = (0..x_syn.len()).filter(|&i| x_syn[i] == 1).collect();
        let z_defects: Vec<usize> = (0..z_syn.len()).filter(|&i| z_syn[i] == 1).collect();

        // X-stabilizer defects (detect Z errors) -> Z corrections; Z-stab -> X corrections.
        let (z_corr, x_pairs) = decode_channel(&x_supports, n_data, &x_defects);
        let (x_corr, z_pairs) = decode_channel(&z_supports, n_data, &z_defects);

        code.correct(&x_corr, &z_corr);
        let (x_after, z_after) = code.measure_syndrome();
        let cleared = x_after.iter().all(|&b| b == 0) && z_after.iter().all(|&b| b == 0);
        let logical_x = code.has_logical_x_error();
        let logical_z = code.has_logical_z_error();

        shots.push(Shot {
            errors,
            x_defects,
            z_defects,
            x_corr,
            z_corr,
            x_pairs,
            z_pairs,
            cleared,
            logical_x,
            logical_z,
        });
    }

    Dataset {
        name: cfg.name.clone(),
        distance: cfg.distance,
        p: cfg.p,
        x_stab_pos,
        z_stab_pos,
        shots,
    }
}

// ── Serialization ───────────────────────────────────────────────────────────

/// Build the populated player HTML for the given configs.
pub fn build_qec_html(configs: &[QecConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let datasets: Vec<Dataset> = configs.iter().map(capture_dataset).collect();

    let mut json = String::from("{\"datasets\":[");
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

/// Capture the default datasets and write the player to `path`.
pub fn export_default_qec_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_qec_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
        }
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn dataset_json(d: &Dataset) -> String {
    let mut s = format!(
        "{{\"name\":\"{}\",\"d\":{},\"p\":{},\"xStab\":{},\"zStab\":{},\"shots\":[",
        json_escape(&d.name),
        d.distance,
        d.p,
        pos_list(&d.x_stab_pos),
        pos_list(&d.z_stab_pos)
    );
    for (i, sh) in d.shots.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&shot_json(sh));
    }
    s.push_str("]}");
    s
}

fn shot_json(sh: &Shot) -> String {
    let errs = sh
        .errors
        .iter()
        .map(|(q, t)| format!("[{q},{t}]"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"e\":[{}],\"xd\":{},\"zd\":{},\"xc\":{},\"zc\":{},\"xm\":{},\"zm\":{},\"ok\":{},\"lx\":{},\"lz\":{}}}",
        errs,
        usize_list(&sh.x_defects),
        usize_list(&sh.z_defects),
        usize_list(&sh.x_corr),
        usize_list(&sh.z_corr),
        pair_list(&sh.x_pairs),
        pair_list(&sh.z_pairs),
        sh.cleared,
        sh.logical_x,
        sh.logical_z
    )
}

fn pos_list(v: &[(f64, f64)]) -> String {
    let inner = v
        .iter()
        .map(|(x, y)| format!("[{x},{y}]"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn usize_list(v: &[usize]) -> String {
    let inner = v
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn pair_list(v: &[(usize, i64)]) -> String {
    let inner = v
        .iter()
        .map(|(a, b)| format!("[{a},{b}]"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
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
    fn test_decoder_always_clears() {
        // A correct matching decoder must clear the syndrome on every shot.
        for &(d, p) in &[(7usize, 0.06f64), (11, 0.04), (11, 0.12)] {
            let cfg = QecConfig {
                name: "t".into(),
                distance: d,
                p,
                shots: 120,
                seed: 99,
            };
            let ds = capture_dataset(&cfg);
            let cleared = ds.shots.iter().filter(|s| s.cleared).count();
            assert_eq!(
                cleared,
                ds.shots.len(),
                "d={d} p={p}: {cleared}/{} cleared",
                ds.shots.len()
            );
        }
    }

    #[test]
    fn test_threshold_trend() {
        // Logical-error rate should rise with p and fall with distance.
        let rate = |d, p| {
            let ds = capture_dataset(&QecConfig {
                name: "t".into(),
                distance: d,
                p,
                shots: 300,
                seed: 5,
            });
            ds.shots
                .iter()
                .filter(|s| s.logical_x || s.logical_z)
                .count() as f64
                / ds.shots.len() as f64
        };
        assert!(rate(11, 0.12) > rate(11, 0.03), "rate should rise with p");
        assert!(
            rate(11, 0.05) <= rate(7, 0.05) + 0.05,
            "larger d should not be worse at low p"
        );
    }

    #[test]
    fn test_build_html_embeds_bundle() {
        let cfgs = vec![QecConfig {
            name: "mini".into(),
            distance: 5,
            p: 0.06,
            shots: 8,
            seed: 3,
        }];
        let html = build_qec_html(&cfgs).expect("build");
        assert!(!html.contains(MARKER));
        assert!(html.contains("\"datasets\":["));
        assert!(html.contains("\"name\":\"mini\""));
        assert!(html.contains("\"xStab\":["));
        assert!(html.contains("\"xm\":["));
    }
}
