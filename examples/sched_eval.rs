//! Workload-scheduling control-evaluation dashboard.
//!
//! Schedules a workload of real compiled tile-CPU programs onto a small cluster under two
//! policies — throughput-greedy and isolation-aware — and prints the throughput <-> safety
//! frontier between them.
//!
//! Run: `cargo run --release --example sched_eval`

use engine::sched_eval::{
    Cluster, Job, Tenant, audit_costs, contention_jobs, demo_jobs, frontier_csv, frontier_report,
    frontier_sweep_report, schedule_greedy,
};

fn main() {
    let cluster = Cluster::new(4, 2);

    println!("Workload-scheduling control evaluation");
    println!("======================================\n");
    println!(
        "Two deterministic scenarios, both built from real compiled tile-CPU programs:\n\
         1. balanced_backfill: trusted jobs can backfill safely, so the optimizer removes leakage for free.\n\
         2. victim_contention: little trusted slack, so zero leakage has a visible throughput price.\n"
    );

    run_scenario("balanced_backfill", &demo_jobs(), cluster);
    println!();
    run_scenario("victim_contention", &contention_jobs(), cluster);

    println!(
        "\nReading the two frontiers: the balanced workload shows why naive isolation can be too\n\
         pessimistic -- the search optimizer finds a 0-leak placement at the greedy makespan.\n\
         The contention workload is the counterexample: with mostly victim/adversary work,\n\
         safety cannot be hidden inside trusted backfill, so the makespan rises.\n"
    );
    println!(
        "Every number is deterministic and reproducible. The scheduler reasons over REAL measured\n\
         program costs; the safety verdict comes from the control_eval thermal covert-channel oracle."
    );
}

fn run_scenario(name: &str, jobs: &[Job], cluster: Cluster) {
    println!("Scenario: {name}");
    println!("Workload (real tile-CPU programs, golden-measured costs):");
    for (i, j) in jobs.iter().enumerate() {
        let role = match j.tenant {
            Tenant::Trusted => "trusted",
            Tenant::Victim => "victim ",
            Tenant::Adversary => "ADVERSARY",
        };
        println!(
            "  #{:<2} {:<18} {:>6} cyc   [{}]",
            i, j.name, j.cost_cycles, role
        );
    }
    println!();

    print!("{}", frontier_report(jobs, cluster));
    println!();
    print!("{}", frontier_sweep_report(jobs, cluster));

    // Show the offending placement greedy produced.
    let greedy = schedule_greedy(jobs, cluster);
    if let Some(x) = greedy.exploitable.first() {
        println!(
            "\n  example exploit (greedy): #{} '{}' (adversary) co-located with #{} '{}' (victim) in rack {}",
            x.adversary_job,
            jobs[x.adversary_job].name,
            x.victim_job,
            jobs[x.victim_job].name,
            x.rack
        );
        println!(
            "    overlap {} cycles -> {} secret bits leaked, monitor caught it: {}",
            x.overlap_cycles, x.bits_leaked, x.detected
        );
    }

    // Prove the cost table is real by re-measuring each job live on the V2 ISS.
    println!("\n  cost-table audit (scheduler's assumed cost vs live ISS execution):");
    println!(
        "    {:<18} {:>8} {:>10} {:>11} {:>6}",
        "program", "assumed", "live_instr", "modeled_cyc", "match"
    );
    let mut all_match = true;
    for a in audit_costs(jobs) {
        all_match &= a.matches;
        let status = match &a.error {
            Some(e) => e.clone(),
            None => {
                if a.matches {
                    "ok".into()
                } else {
                    "DIFF".into()
                }
            }
        };
        println!(
            "    {:<18} {:>8} {:>10} {:>11} {:>6}",
            a.name, a.assumed_cost, a.live_retired, a.modeled_cost, status
        );
    }
    println!(
        "    => cost table {} live execution",
        if all_match {
            "MATCHES"
        } else {
            "DIVERGES from"
        }
    );

    // Reproducible data behind the frontier plot (deterministic from the demo seed).
    println!("\n  frontier CSV (lambda,makespan,utilization_pct,bits_leaked,exploitable_windows):");
    for line in frontier_csv(jobs, cluster).lines() {
        println!("    {line}");
    }
}
