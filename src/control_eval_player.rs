//! Self-contained "ControlEval · thermal covert channel" HTML player.
//!
//! Replays the AI-control thermal side-channel evaluation from [`crate::control_eval`]:
//! an untrusted co-tenant exfiltrates a secret byte by modulating the heat of a shared
//! cooling zone; a receiver decodes the carrier; a trusted monitor watches for periodic
//! structure and contains the channel when it alerts. This player captures the *real
//! painted heat field* tick-by-tick (the glowing zone bus, the adversary/benign cores,
//! the monitor and receiver tiles) together with the receiver's decoded bit stream and
//! the final detection report, and embeds everything into one interactive HTML file —
//! watch the carrier pulse, the bits land, and the monitor slam the channel shut.
//!
//! Two runs are captured: the attack (adversary active) and the false-positive baseline
//! (adversary silent). Deterministic and CPU-only; the simulation, the receiver and the
//! monitor are all the engine's own. Nothing here mutates `control_eval`.

use crate::control_eval::{ControlEvalConfig, ControlEvalState, REGION, build_scenario, step};
use crate::simulation::Simulation;
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("control_eval_player_template.html");
const MARKER: &str = "/*__CTRL_BUNDLE__*/null";

/// Capture one heat frame every `STRIDE` ticks (keeps the embedded animation compact while
/// staying smooth — `BIT_PERIOD` is 16, so this is ~4 frames per transmitted bit).
const STRIDE: u32 = 4;

/// One decoded receiver bit, tagged with the tick it landed on (for timeline placement).
struct BitEvent {
    tick: u32,
    truth: u32,
    value: u32,
    correct: bool,
    before: bool,
}

/// One captured control-eval run.
struct RunData {
    name: &'static str,
    label: &'static str,
    adversary: bool,
    secret: u8,
    ticks: u32,
    /// Quantized heat frames: one string per sampled tick, `w*h` chars, 1 hex digit (0-15) per cell.
    frames: Vec<String>,
    frame_ticks: Vec<u32>,
    bits: Vec<BitEvent>,
    detection_tick: Option<u32>,
    detection_latency: Option<u32>,
    detected_period: Option<u32>,
    bits_leaked: u32,
    bits_transmitted: u32,
    ber_pre: u32,
    ber_post: u32,
    monitor_score: u32,
}

/// A run to feature: a name, a label and a config builder.
pub struct CtrlConfig {
    pub name: &'static str,
    pub label: &'static str,
    pub cfg: ControlEvalConfig,
}

/// The two canonical runs from the `control_eval` dashboard.
pub fn default_configs() -> Vec<CtrlConfig> {
    vec![
        CtrlConfig {
            name: "attack",
            label: "Adversary active — real exfiltration",
            cfg: ControlEvalConfig {
                secret: 0xB4,
                adversary_enabled: true,
                ..Default::default()
            },
        },
        CtrlConfig {
            name: "baseline",
            label: "Adversary silent — false-positive control",
            cfg: ControlEvalConfig {
                adversary_enabled: false,
                ..Default::default()
            },
        },
    ]
}

fn capture(c: &CtrlConfig) -> RunData {
    let mut sim = Simulation::new();
    build_scenario(&mut sim);
    let ticks = c.cfg.ticks;
    let mut state = ControlEvalState::new(c.cfg.clone());

    let mut frames = Vec::new();
    let mut frame_ticks = Vec::new();
    let mut bits = Vec::new();
    let mut prev_decoded = 0usize;

    for t in 0..ticks {
        sim.tick();
        step(&mut sim, &mut state, t);

        // A new decoded bit lands at this tick whenever the receiver's slot closed.
        if state.decoded.len() > prev_decoded {
            for d in &state.decoded[prev_decoded..] {
                bits.push(BitEvent {
                    tick: t,
                    truth: d.truth,
                    value: d.value,
                    correct: d.correct,
                    before: d.before_containment,
                });
            }
            prev_decoded = state.decoded.len();
        }

        if t % STRIDE == 0 || t == ticks - 1 {
            frames.push(quantize_frame(&sim));
            frame_ticks.push(t);
        }
    }

    let r = state.report();
    RunData {
        name: c.name,
        label: c.label,
        adversary: c.cfg.adversary_enabled,
        secret: c.cfg.secret,
        ticks,
        frames,
        frame_ticks,
        bits,
        detection_tick: r.detection_tick,
        detection_latency: r.detection_latency,
        detected_period: r.detected_period,
        bits_leaked: r.bits_leaked_before_detection,
        bits_transmitted: r.bits_transmitted_before_detection,
        ber_pre: r.pre_containment_ber_pct,
        ber_post: r.post_containment_ber_pct,
        monitor_score: r.monitor_score,
    }
}

/// Snapshot the painted heat region, quantize each cell to 0-15 and pack as a hex string.
fn quantize_frame(sim: &Simulation) -> String {
    let grid = sim.snapshot_heat_region(REGION.x0, REGION.y0, REGION.w, REGION.h);
    let mut s = String::with_capacity((REGION.w * REGION.h) as usize);
    for row in &grid {
        for &v in row {
            let level = (v >> 4).min(15); // 0..=255 -> 0..=15
            s.push(std::char::from_digit(level, 16).unwrap());
        }
    }
    s
}

/// Build the populated player HTML for the given runs.
pub fn build_ctrl_html(configs: &[CtrlConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let runs: Vec<RunData> = configs.iter().map(capture).collect();

    let mut json = String::from("{\"runs\":[");
    for (i, r) in runs.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&run_json(r));
    }
    json.push_str(&format!(
        "],\"default\":0,\"w\":{},\"h\":{}}}",
        REGION.w, REGION.h
    ));

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(format!("template marker count = {n}, expected 1"));
    }
    Ok(TEMPLATE.replace(MARKER, &json))
}

/// Capture the default runs and write the player to `path`.
pub fn export_default_ctrl_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_ctrl_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn opt(v: Option<u32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

fn run_json(r: &RunData) -> String {
    let frames = r
        .frames
        .iter()
        .map(|f| format!("\"{f}\""))
        .collect::<Vec<_>>()
        .join(",");
    let ftk = r
        .frame_ticks
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let bits = r
        .bits
        .iter()
        .map(|b| {
            format!(
                "{{\"tick\":{},\"truth\":{},\"value\":{},\"correct\":{},\"before\":{}}}",
                b.tick, b.truth, b.value, b.correct, b.before
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"name\":\"{}\",\"label\":\"{}\",\"adversary\":{},\"secret\":{},\"secretHex\":\"{:02X}\",\
         \"ticks\":{},\"frames\":[{}],\"frameTicks\":[{}],\"bits\":[{}],\
         \"detectionTick\":{},\"detectionLatency\":{},\"detectedPeriod\":{},\
         \"bitsLeaked\":{},\"bitsTransmitted\":{},\"berPre\":{},\"berPost\":{},\"monitorScore\":{}}}",
        r.name,
        r.label,
        r.adversary,
        r.secret,
        r.secret,
        r.ticks,
        frames,
        ftk,
        bits,
        opt(r.detection_tick),
        opt(r.detection_latency),
        opt(r.detected_period),
        r.bits_leaked,
        r.bits_transmitted,
        r.ber_pre,
        r.ber_post,
        r.monitor_score,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attack_is_detected_and_leaks_then_baseline_is_quiet() {
        let cfgs = default_configs();
        let attack = capture(&cfgs[0]);
        let baseline = capture(&cfgs[1]);

        // The attack run must be caught by the monitor and must have leaked some bits first.
        assert!(attack.adversary);
        assert!(
            attack.detection_tick.is_some(),
            "monitor must detect the real exfiltration"
        );
        assert!(attack.bits_transmitted > 0, "attack should transmit bits");
        assert!(!attack.bits.is_empty(), "receiver should decode bits");

        // The baseline must not false-alarm.
        assert!(!baseline.adversary);
        assert!(
            baseline.detection_tick.is_none(),
            "benign baseline must not false-positive"
        );
    }

    #[test]
    fn test_frames_are_well_formed() {
        let d = capture(&default_configs()[0]);
        let cells = (REGION.w * REGION.h) as usize;
        assert!(d.frames.len() > 20, "expected a smooth animation");
        assert_eq!(d.frames.len(), d.frame_ticks.len());
        for f in &d.frames {
            assert_eq!(f.len(), cells, "each frame packs one hex char per cell");
            assert!(f.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn test_build_html_embeds_bundle() {
        let html = build_ctrl_html(&default_configs()).expect("build");
        assert!(!html.contains(MARKER), "marker must be consumed");
        assert!(html.contains("\"runs\":["));
        assert!(html.contains("\"name\":\"attack\""));
        assert!(html.contains("\"name\":\"baseline\""));
        assert!(html.contains("\"frames\":["));
        assert!(html.contains("\"detectionTick\":"));
    }
}
