//! Self-contained "SchedEval · throughput↔safety frontier" HTML player.
//!
//! Captures the workload-scheduling control evaluation from [`crate::sched_eval`]:
//! a workload of real compiled tile-CPU programs (golden-measured cycle costs) is
//! placed onto a small cluster under four policies — throughput-greedy,
//! naive isolation-aware, constraint-aware secure-packing, and a simulated-annealing
//! optimizer run at the safe operating weight. Each schedule is rendered as a Gantt
//! chart (cores × time, bars colored by tenant trust class) with the exploitable
//! adversary∩victim co-locations flagged in red, alongside the throughput↔safety
//! Pareto frontier (makespan vs bits leaked across a λ sweep) with the derived safe
//! operating knee marked.
//!
//! Everything (jobs, placements, exploits, frontier) is embedded into one
//! interactive HTML file. Deterministic (seeded) and CPU-only. The schedulers,
//! the leak counts and the frontier are all the engine's own; the safety verdict
//! comes from the `control_eval` thermal covert-channel oracle.

use crate::sched_eval::{
    Cluster, DEFAULT_LAMBDAS, FrontierPoint, Job, Schedule, Tenant, contention_jobs, demo_jobs,
    optimize_assignment, pareto_frontier, safe_operating_point, schedule_greedy,
    schedule_isolation_aware, schedule_secure_packing,
};
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("sched_eval_player_template.html");
const MARKER: &str = "/*__SCHED_BUNDLE__*/null";

/// The optimizer budget used for the embedded frontier (matches `sched_eval` defaults).
const OPT_ITERS: u32 = 4_000;
const OPT_SEED: u64 = 0xC0FFEE;

/// One scheduling scenario plus everything needed to render it.
struct ScenarioData {
    name: &'static str,
    blurb: &'static str,
    jobs: Vec<Job>,
    cluster: Cluster,
    policies: Vec<Schedule>,
    frontier: Vec<FrontierPoint>,
    knee: Option<FrontierPoint>,
}

/// A scenario to feature: a name, a blurb and a job-set builder.
pub struct SchedConfig {
    pub name: &'static str,
    pub blurb: &'static str,
    pub jobs: fn() -> Vec<Job>,
}

/// The two canonical scenarios from the `sched_eval` dashboard.
pub fn default_configs() -> Vec<SchedConfig> {
    vec![
        SchedConfig {
            name: "balanced_backfill",
            blurb: "Trusted jobs can backfill the cluster safely, so the optimizer removes \
                    the covert channel for free — zero leakage at the greedy makespan.",
            jobs: demo_jobs,
        },
        SchedConfig {
            name: "victim_contention",
            blurb: "Little trusted slack: mostly victim and adversary work, so closing the \
                    channel cannot hide inside backfill and safety carries a visible price.",
            jobs: contention_jobs,
        },
    ]
}

fn capture(cfg: &SchedConfig) -> ScenarioData {
    let jobs = (cfg.jobs)();
    let cluster = Cluster::new(4, 2);

    let frontier = pareto_frontier(&jobs, cluster, &DEFAULT_LAMBDAS, OPT_ITERS, OPT_SEED);
    let knee = safe_operating_point(&frontier);

    // The four policies we render side by side. The optimizer runs at the safe knee weight when
    // one exists (the cheapest λ that already closed the channel), else the strongest weight.
    let opt_lambda = knee
        .map(|p| p.lambda)
        .unwrap_or_else(|| *DEFAULT_LAMBDAS.last().unwrap());
    let policies = vec![
        schedule_greedy(&jobs, cluster),
        schedule_isolation_aware(&jobs, cluster),
        schedule_secure_packing(&jobs, cluster),
        optimize_assignment(&jobs, cluster, opt_lambda, OPT_ITERS, OPT_SEED),
    ];

    ScenarioData {
        name: cfg.name,
        blurb: cfg.blurb,
        jobs,
        cluster,
        policies,
        frontier,
        knee,
    }
}

/// Build the populated player HTML for the given scenarios.
pub fn build_sched_html(configs: &[SchedConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let scenarios: Vec<ScenarioData> = configs.iter().map(capture).collect();

    let mut json = String::from("{\"scenarios\":[");
    for (i, s) in scenarios.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&scenario_json(s));
    }
    json.push_str("],\"default\":0}");

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(format!("template marker count = {n}, expected 1"));
    }
    Ok(TEMPLATE.replace(MARKER, &json))
}

/// Capture the default scenarios and write the player to `path`.
pub fn export_default_sched_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_sched_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn tenant_str(t: Tenant) -> &'static str {
    match t {
        Tenant::Trusted => "trusted",
        Tenant::Victim => "victim",
        Tenant::Adversary => "adversary",
    }
}

fn scenario_json(s: &ScenarioData) -> String {
    let jobs = s
        .jobs
        .iter()
        .map(|j| {
            format!(
                "{{\"name\":\"{}\",\"cost\":{},\"tenant\":\"{}\"}}",
                json_escape(j.name),
                j.cost_cycles,
                tenant_str(j.tenant)
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let policies = s
        .policies
        .iter()
        .map(policy_json)
        .collect::<Vec<_>>()
        .join(",");

    let frontier = s
        .frontier
        .iter()
        .map(|p| {
            format!(
                "{{\"lambda\":{},\"makespan\":{},\"util\":{},\"bits\":{},\"windows\":{}}}",
                p.lambda, p.makespan, p.utilization_pct, p.bits_leaked, p.exploitable_windows
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let knee = match &s.knee {
        Some(p) => format!(
            "{{\"lambda\":{},\"makespan\":{},\"bits\":{}}}",
            p.lambda, p.makespan, p.bits_leaked
        ),
        None => "null".to_string(),
    };

    format!(
        "{{\"name\":\"{}\",\"blurb\":\"{}\",\"cluster\":{{\"cores\":{},\"coresPerRack\":{}}},\
         \"jobs\":[{}],\"policies\":[{}],\"frontier\":[{}],\"knee\":{}}}",
        json_escape(s.name),
        json_escape(s.blurb),
        s.cluster.n_cores,
        s.cluster.cores_per_rack,
        jobs,
        policies,
        frontier,
        knee
    )
}

fn policy_json(s: &Schedule) -> String {
    let placements = s
        .placements
        .iter()
        .map(|p| {
            format!(
                "{{\"j\":{},\"core\":{},\"start\":{},\"end\":{}}}",
                p.job_index, p.core, p.start, p.end
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let exploits = s
        .exploitable
        .iter()
        .map(|e| {
            format!(
                "{{\"adv\":{},\"vic\":{},\"rack\":{},\"overlap\":{},\"bits\":{},\"detected\":{}}}",
                e.adversary_job, e.victim_job, e.rack, e.overlap_cycles, e.bits_leaked, e.detected
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"name\":\"{}\",\"makespan\":{},\"util\":{},\"bits\":{},\"windows\":{},\"detected\":{},\
         \"placements\":[{}],\"exploits\":[{}]}}",
        json_escape(s.policy),
        s.makespan,
        s.utilization_pct,
        s.total_bits_leaked,
        s.exploitable.len(),
        s.windows_detected,
        placements,
        exploits
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
    fn test_capture_has_four_policies_and_a_frontier() {
        let cfg = &default_configs()[0];
        let d = capture(cfg);
        assert_eq!(d.policies.len(), 4, "greedy/iso/secure/optimized");
        assert_eq!(d.frontier.len(), DEFAULT_LAMBDAS.len());
        // Every placement covers a real job and a real core.
        for pol in &d.policies {
            assert_eq!(pol.placements.len(), d.jobs.len());
            for p in &pol.placements {
                assert!(p.job_index < d.jobs.len() && p.core < d.cluster.n_cores);
                assert!(p.end >= p.start);
            }
        }
        // The greedy policy is the throughput baseline; the safe policies must not leak.
        let greedy = &d.policies[0];
        let iso = &d.policies[1];
        let secure = &d.policies[2];
        assert!(greedy.total_bits_leaked > 0, "greedy should leak");
        assert_eq!(iso.total_bits_leaked, 0, "isolation must be leak-free");
        assert_eq!(
            secure.total_bits_leaked, 0,
            "secure-packing must be leak-free"
        );
    }

    #[test]
    fn test_knee_is_zero_leak_when_present() {
        for cfg in default_configs() {
            let d = capture(&cfg);
            if let Some(k) = d.knee {
                assert_eq!(k.bits_leaked, 0, "the safe knee must be a zero-leak point");
            }
        }
    }

    #[test]
    fn test_build_html_embeds_bundle() {
        let html = build_sched_html(&default_configs()).expect("build");
        assert!(!html.contains(MARKER), "marker must be consumed");
        assert!(html.contains("\"scenarios\":["));
        assert!(html.contains("\"name\":\"balanced_backfill\""));
        assert!(html.contains("\"name\":\"victim_contention\""));
        assert!(html.contains("\"policies\":["));
        assert!(html.contains("\"frontier\":["));
        assert!(html.contains("\"tenant\":\"adversary\""));
    }
}
