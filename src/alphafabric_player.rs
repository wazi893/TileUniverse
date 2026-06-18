//! Self-contained "AlphaFabric · learned placement" HTML player.
//!
//! Captures a real placement-optimization trajectory from
//! [`crate::synth::alphafabric`]: a mapped tile circuit is placed on the slot
//! grid and improved by simulated annealing against the engine's own
//! half-perimeter-wirelength objective, recording the best layout found over
//! time. The learned policy's deterministic one-shot placement is captured as a
//! reference. Everything (gate positions, nets, HPWL, the learned result) is
//! embedded into one interactive HTML file — watch the chip floorplan relax and
//! the wirelength drop as it optimizes.
//!
//! Deterministic (seeded) and CPU-only. The circuit construction, slot grid,
//! HPWL, routing/score and the learned placer are all the engine's own.

use crate::synth::alphafabric::{
    Circuit, PlacementEnv, PolicyWeights, build_eq_comparator, build_parity, build_ripple_adder,
    construct_placement,
};
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("alphafabric_player_template.html");
const MARKER: &str = "/*__AF_BUNDLE__*/null";

/// One sampled best-so-far layout.
struct Frame {
    pos: Vec<(usize, usize)>,
    hpwl: usize,
}

struct AfDataset {
    name: String,
    min_x: usize,
    min_y: usize,
    width: usize,
    height: usize,
    gates: usize,
    nets: Vec<Vec<usize>>,
    frames: Vec<Frame>,
    baseline_hpwl: usize,
    best_hpwl: usize,
    learned_hpwl: usize,
    final_wire_tiles: usize,
    routable: bool,
}

/// A circuit to feature: a name and an AIG builder.
pub struct AfConfig {
    pub name: String,
    pub build: fn() -> crate::synth::alphafabric::Circuit,
    pub iters: usize,
    pub seed: u64,
}

/// Default switchable demos across different circuit structures.
pub fn default_configs() -> Vec<AfConfig> {
    vec![
        AfConfig {
            name: "Ripple adder · 6-bit".into(),
            build: || Circuit::from_aig("adder6", build_ripple_adder(6)),
            iters: 6000,
            seed: 0xA1,
        },
        AfConfig {
            name: "Equality comparator · 10-bit".into(),
            build: || Circuit::from_aig("eq10", build_eq_comparator(10)),
            iters: 6000,
            seed: 0xB2,
        },
        AfConfig {
            name: "Parity tree · 16-bit".into(),
            build: || Circuit::from_aig("parity16", build_parity(16)),
            iters: 6000,
            seed: 0xC3,
        },
    ]
}

#[inline]
fn sm_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
#[inline]
fn sm_f64(s: &mut u64) -> f64 {
    (sm_next(s) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn positions(env: &PlacementEnv) -> Vec<(usize, usize)> {
    let cv = env.canvas();
    let asg = env.assignment();
    (0..env.num_gates())
        .map(|g| cv.slot_coord(asg[g]))
        .collect()
}

fn capture(cfg: &AfConfig) -> AfDataset {
    let circuit = (cfg.build)();
    let mut env = PlacementEnv::new(&circuit).expect("placement env");
    let cv = env.canvas();
    let n = env.num_gates();
    let nets = env.hyperedges().to_vec();

    env.reset();
    let baseline_hpwl = env.hpwl();

    // Learned policy, one-shot constructive placement (deterministic).
    let weights = PolicyWeights {
        theta: [1.0, 0.15, 0.0, 0.0],
    };
    let learned_assign = construct_placement(&env, &weights);
    env.restore_assignment(&learned_assign);
    let learned_hpwl = env.hpwl();

    // Simulated annealing from the row-major baseline; record best-so-far.
    env.reset();
    let mut rng = cfg.seed | 1;
    let mut cur = baseline_hpwl;
    let mut best = baseline_hpwl;
    let mut best_assign = env.assignment().to_vec();

    let sample_every = (cfg.iters / 80).max(1);
    let mut frames = vec![Frame {
        pos: positions(&env),
        hpwl: best,
    }];
    let t0 = (baseline_hpwl as f64 / 12.0).max(1.0);

    for it in 0..cfg.iters {
        let temp = t0 * (0.4f64 / t0).powf(it as f64 / cfg.iters as f64);

        let accept = |cur: usize, new: usize, rng: &mut u64| -> bool {
            new <= cur || sm_f64(rng) < ((cur as f64 - new as f64) / temp).exp()
        };

        if sm_next(&mut rng) & 1 == 0 {
            let a = (sm_next(&mut rng) as usize) % n;
            let b = (sm_next(&mut rng) as usize) % n;
            if a == b {
                continue;
            }
            let _ = env.swap(a, b);
            let nh = env.hpwl();
            if accept(cur, nh, &mut rng) {
                cur = nh;
            } else {
                let _ = env.swap(a, b);
            }
        } else {
            let empties = env.empty_slots();
            if empties.is_empty() {
                continue;
            }
            let g = (sm_next(&mut rng) as usize) % n;
            let old = env.assignment()[g];
            let s = empties[(sm_next(&mut rng) as usize) % empties.len()];
            let _ = env.relocate(g, s);
            let nh = env.hpwl();
            if accept(cur, nh, &mut rng) {
                cur = nh;
            } else {
                let _ = env.relocate(g, old);
            }
        }

        if cur < best {
            best = cur;
            best_assign = env.assignment().to_vec();
        }
        if (it + 1) % sample_every == 0 {
            let save = env.assignment().to_vec();
            env.restore_assignment(&best_assign);
            frames.push(Frame {
                pos: positions(&env),
                hpwl: best,
            });
            env.restore_assignment(&save);
        }
    }

    env.restore_assignment(&best_assign);
    let score = env.score();

    AfDataset {
        name: cfg.name.clone(),
        min_x: cv.min_x,
        min_y: cv.min_y,
        width: cv.cols * cv.pitch + 2 * cv.halo,
        height: cv.rows * cv.pitch + 2 * cv.halo,
        gates: n,
        nets,
        frames,
        baseline_hpwl,
        best_hpwl: best,
        learned_hpwl,
        final_wire_tiles: score.metrics.wire_tiles,
        routable: score.metrics.routable,
    }
}

/// Build the populated player HTML for the given configs.
pub fn build_af_html(configs: &[AfConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let datasets: Vec<AfDataset> = configs.iter().map(capture).collect();

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

/// Capture the default demos and write the player to `path`.
pub fn export_default_af_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_af_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
        }
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn dataset_json(d: &AfDataset) -> String {
    let nets = d
        .nets
        .iter()
        .map(|e| {
            format!(
                "[{}]",
                e.iter()
                    .map(|g| g.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let frames = d
        .frames
        .iter()
        .map(|f| {
            let pos = f
                .pos
                .iter()
                .map(|(x, y)| format!("[{x},{y}]"))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"p\":[{pos}],\"h\":{}}}", f.hpwl)
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"name\":\"{}\",\"minX\":{},\"minY\":{},\"w\":{},\"h\":{},\"gates\":{},\"nets\":[{}],\"frames\":[{}],\"baseHpwl\":{},\"bestHpwl\":{},\"learnedHpwl\":{},\"wireTiles\":{},\"routable\":{}}}",
        json_escape(&d.name),
        d.min_x,
        d.min_y,
        d.width,
        d.height,
        d.gates,
        nets,
        frames,
        d.baseline_hpwl,
        d.best_hpwl,
        d.learned_hpwl,
        d.final_wire_tiles,
        d.routable
    )
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
    fn test_capture_improves_hpwl() {
        let cfg = AfConfig {
            name: "t".into(),
            build: || Circuit::from_aig("adder4", build_ripple_adder(4)),
            iters: 3000,
            seed: 1,
        };
        let d = capture(&cfg);
        assert!(d.frames.len() > 10);
        assert!(d.gates > 0 && !d.nets.is_empty());
        // Annealing must not make wirelength worse than the baseline.
        assert!(d.best_hpwl <= d.baseline_hpwl);
        // On a structured adder it should find a real improvement.
        assert!(
            d.best_hpwl < d.baseline_hpwl,
            "no improvement: {} -> {}",
            d.baseline_hpwl,
            d.best_hpwl
        );
        // Every frame's positions cover all gates.
        for f in &d.frames {
            assert_eq!(f.pos.len(), d.gates);
        }
    }

    #[test]
    fn test_build_html_embeds_bundle() {
        let cfgs = vec![AfConfig {
            name: "mini".into(),
            build: || Circuit::from_aig("eq4", build_eq_comparator(4)),
            iters: 1500,
            seed: 2,
        }];
        let html = build_af_html(&cfgs).expect("build");
        assert!(!html.contains(MARKER));
        assert!(html.contains("\"datasets\":["));
        assert!(html.contains("\"name\":\"mini\""));
        assert!(html.contains("\"frames\":["));
        assert!(html.contains("\"learnedHpwl\":"));
    }
}
