//! Self-contained "AI Control" hub: ONE HTML file containing both control evals.
//!
//! Wraps the two AI-control players — the [`crate::control_eval_player`] thermal
//! covert-channel replay and the [`crate::sched_eval_player`] throughput↔safety
//! scheduler — into a single tabbed page with a narrative overview and live
//! headline stat cards computed from the real eval APIs. Each player is hosted in
//! an isolated `<iframe srcdoc>` (set from a JSON-encoded string at runtime, so
//! the two players' CSS/JS never collide and no HTML-attribute escaping is needed).
//!
//! The result is the portfolio "front door": a single file you can hand someone
//! that opens to the story and lets them scrub both interactive evals. Deterministic
//! and CPU-only; reuses the players verbatim — no new visualization logic.

use crate::control_eval::{ControlEvalConfig, run};
use crate::control_eval_player::{build_ctrl_html, default_configs as ctrl_configs};
use crate::sched_eval::{
    Cluster, DEFAULT_LAMBDAS, demo_jobs, optimize_assignment, pareto_frontier,
    safe_operating_point, schedule_greedy,
};
use crate::sched_eval_player::{build_sched_html, default_configs as sched_configs};
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("aicontrol_hub_template.html");
const MARKER: &str = "/*__HUB_BUNDLE__*/null";

/// Headline numbers for the overview cards, pulled live from the eval APIs.
struct HubStats {
    // thermal channel
    detection_latency: u32,
    bits_leaked: u32,
    bits_transmitted: u32,
    baseline_alerts: u32,
    // scheduler
    greedy_leaked: u32,
    greedy_makespan: u64,
    safe_makespan: u64,
    safe_overhead_pct: i64,
}

fn capture_stats() -> HubStats {
    // Thermal covert channel: the default attack + the benign baseline.
    let attack = run(ControlEvalConfig {
        secret: 0xB4,
        adversary_enabled: true,
        ..Default::default()
    });
    let baseline = run(ControlEvalConfig {
        adversary_enabled: false,
        ..Default::default()
    });

    // Scheduler: greedy (leaky) vs the derived safe operating point on the frontier.
    let jobs = demo_jobs();
    let cluster = Cluster::new(4, 2);
    let greedy = schedule_greedy(&jobs, cluster);
    let frontier = pareto_frontier(&jobs, cluster, &DEFAULT_LAMBDAS, 4_000, 0xC0FFEE);
    let knee = safe_operating_point(&frontier);
    let safe_lambda = knee
        .map(|p| p.lambda)
        .unwrap_or(*DEFAULT_LAMBDAS.last().unwrap());
    let optimized = optimize_assignment(&jobs, cluster, safe_lambda, 4_000, 0xC0FFEE);

    let safe_overhead_pct = if greedy.makespan > 0 {
        ((optimized.makespan as i64 - greedy.makespan as i64) * 100) / greedy.makespan as i64
    } else {
        0
    };

    HubStats {
        detection_latency: attack.detection_latency.unwrap_or(0),
        bits_leaked: attack.bits_leaked_before_detection,
        bits_transmitted: attack.bits_transmitted_before_detection,
        baseline_alerts: u32::from(baseline.detection_tick.is_some()),
        greedy_leaked: greedy.total_bits_leaked,
        greedy_makespan: greedy.makespan,
        safe_makespan: optimized.makespan,
        safe_overhead_pct,
    }
}

/// Build the populated hub HTML embedding both players.
pub fn build_hub_html() -> Result<String, String> {
    let thermal = build_ctrl_html(&ctrl_configs())?;
    let scheduler = build_sched_html(&sched_configs())?;
    let stats = capture_stats();

    let bundle = format!(
        "{{\"thermal\":\"{}\",\"scheduler\":\"{}\",\"stats\":{}}}",
        json_escape_script(&thermal),
        json_escape_script(&scheduler),
        stats_json(&stats),
    );

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(format!("template marker count = {n}, expected 1"));
    }
    Ok(TEMPLATE.replace(MARKER, &bundle))
}

/// Build the hub and write it to `path`.
pub fn export_default_hub(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_hub_html()?;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn stats_json(s: &HubStats) -> String {
    format!(
        "{{\"detectionLatency\":{},\"bitsLeaked\":{},\"bitsTransmitted\":{},\"baselineAlerts\":{},\
         \"greedyLeaked\":{},\"greedyMakespan\":{},\"safeMakespan\":{},\"safeOverheadPct\":{}}}",
        s.detection_latency,
        s.bits_leaked,
        s.bits_transmitted,
        s.baseline_alerts,
        s.greedy_leaked,
        s.greedy_makespan,
        s.safe_makespan,
        s.safe_overhead_pct,
    )
}

/// JSON-string-escape that is also safe to embed inside a `<script>` block: escapes
/// `\`, `"`, control chars, AND `<` (as `<`) so an inner `</script>` cannot
/// terminate the hub's script. When JS reads the string back it is exact again.
fn json_escape_script(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 16);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '<' => out.push_str("\\u003c"),
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
    fn test_stats_are_consistent_with_evals() {
        let s = capture_stats();
        // The attack leaks something and is detected; the baseline does not alert.
        assert!(s.bits_transmitted > 0);
        assert!(s.bits_leaked <= s.bits_transmitted);
        assert_eq!(s.baseline_alerts, 0, "benign baseline must not false-alarm");
        // Greedy leaks; the safe point closes the channel at a non-negative overhead.
        assert!(s.greedy_leaked > 0, "greedy should leak");
        assert!(s.safe_makespan >= s.greedy_makespan || s.safe_overhead_pct >= 0);
    }

    #[test]
    fn test_build_hub_embeds_both_players() {
        let html = build_hub_html().expect("build");
        assert!(!html.contains(MARKER), "marker must be consumed");
        assert!(html.contains("\"thermal\":\""));
        assert!(html.contains("\"scheduler\":\""));
        // No raw inner </script> survived the script-safe escaping.
        let inner = "</script>";
        // The hub itself ends with exactly one real </script>; embedded ones are <-escaped.
        assert_eq!(
            html.matches(inner).count(),
            1,
            "embedded players must not introduce extra </script> tags"
        );
        assert!(html.contains("\\u003c"), "inner markup is script-escaped");
    }
}
