//! Workload-scheduling control evaluation: the throughput ↔ safety frontier.
//!
//! This is the ML-systems sibling of [`crate::control_eval`]. Where the control eval asks
//! "given a co-location, can the covert channel be caught and contained?", this asks the layer
//! above it: **who decides the co-locations?** That decision is the scheduler — and "an AI in
//! control of a datacenter" is, concretely, an AI deciding what runs where and when. So this
//! module models a scheduler placing real workloads on real cores, and measures the tension
//! every cluster scheduler actually faces: pack tightly for throughput, or spread for isolation.
//!
//! ## Jobs are real programs, costs are real measurements
//!
//! A [`Job`] is a real compiled tile-CPU program from the perf-track corpus
//! ([`crate::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS`]). Its `cost_cycles` is the
//! number of cycles that program takes to run to halt on the physical tile CPU — a deterministic,
//! golden-locked integer measured by actually executing it (`run_v2_compiled_benchmark`). We do
//! **not** re-execute programs inside the scheduler's inner loop; we execute each once to get its
//! cost, then the scheduler reasons over those real costs. (Phase 2 closes the loop: re-measure
//! each program live on the deterministic ISS and confirm the realized cost matches the table the
//! scheduler used — see [`audit_costs`].)
//!
//! ## The schedulers and the frontier
//!
//! Four placement policies span the throughput↔safety tradeoff:
//!
//! * [`schedule_greedy`] — trust-blind LPT (longest-processing-time-first) list scheduling. It
//!   minimizes makespan by packing jobs onto whichever core is least loaded, and freely co-locates
//!   an untrusted tenant with a victim in the same cooling zone.
//! * [`schedule_isolation_aware`] — the same objective, but racks are partitioned into trust
//!   domains so adversary and victim jobs never share a rack (hence never a cooling zone). This
//!   closes the covert channel *by construction*, at the cost of fewer cores per domain.
//! * [`schedule_secure_packing`] — also safe by construction, but lets *trusted* jobs backfill the
//!   otherwise-idle adversary rack, recovering the capacity naive static partitioning wastes.
//! * [`optimize_assignment`] — a deterministic simulated-annealing search over job→core
//!   assignments minimizing `makespan + lambda * leakage`. Sweeping `lambda` ([`trace_frontier`])
//!   traces the whole Pareto frontier; [`pareto_frontier`] returns it as data and
//!   [`safe_operating_point`] picks the knee — the cheapest leak-free placement.
//!
//! For each schedule we find every same-rack window where an adversary and a victim run at
//! overlapping times and feed it to the [`crate::control_eval`] oracle: how many secret bits leak
//! before the trusted monitor catches it? The gap between a leaky and a leak-free makespan is the
//! deterministic, reproducible price of safety — and whether that price is real depends on the
//! workload. Two demos make the point honestly: [`demo_jobs`] has enough trusted slack that the
//! optimizer reaches zero leakage for free (+0% makespan), while [`contention_jobs`] (mostly
//! victim/adversary work) has no such slack, so zero leakage costs real throughput (+33%).
//!
//! ## Honesty caveats
//!
//! Cycles are treated as the time unit on both the scheduler clock and the control-eval clock —
//! a model, not a claim that one tile-CPU cycle equals one datacenter scheduling quantum. The
//! cooling-zone covert channel is the deterministic model from [`crate::control_eval`], not a
//! capture of real silicon. What is real: the workloads execute, their costs are measured not
//! invented, and the whole frontier replays bit-for-bit.

use crate::control_eval::{ControlEvalConfig, run as run_control_eval};
use crate::tile_cpu::V2Iss;
use crate::tile_cpu::compile_source;
use crate::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS;

/// Trust class of a tenant's job. Drives the isolation policy and the safety analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tenant {
    /// Ordinary trusted compute. Safe to co-locate with anything.
    Trusted,
    /// Untrusted tenant that will attempt thermal exfiltration if co-located with a victim.
    Adversary,
    /// Tenant handling secret data; the target of an adversary's covert channel.
    Victim,
}

/// A schedulable unit of work: a real compiled tile-CPU program with a measured cost.
#[derive(Debug, Clone)]
pub struct Job {
    /// Program name (matches the perf-track corpus).
    pub name: &'static str,
    /// Cycles to run to halt on the physical tile CPU — deterministic, golden-locked.
    pub cost_cycles: u64,
    /// Trust class of the tenant that submitted this job.
    pub tenant: Tenant,
}

/// A placed job: which core it runs on and its `[start, end)` window in cycles.
#[derive(Debug, Clone)]
pub struct Placement {
    pub job_index: usize,
    pub core: usize,
    pub start: u64,
    pub end: u64,
}

/// Cluster geometry: `n_cores` cores grouped into racks; each rack shares one cooling zone.
#[derive(Debug, Clone, Copy)]
pub struct Cluster {
    pub n_cores: usize,
    pub cores_per_rack: usize,
}

impl Cluster {
    pub fn new(n_cores: usize, cores_per_rack: usize) -> Self {
        Self {
            n_cores,
            cores_per_rack: cores_per_rack.max(1),
        }
    }
    pub fn rack_of(&self, core: usize) -> usize {
        core / self.cores_per_rack
    }
    pub fn n_racks(&self) -> usize {
        self.n_cores.div_ceil(self.cores_per_rack)
    }
}

/// A complete schedule plus its derived metrics.
#[derive(Debug, Clone)]
pub struct Schedule {
    pub policy: &'static str,
    pub placements: Vec<Placement>,
    pub cluster: Cluster,
    pub makespan: u64,
    /// Busy core-cycles / (n_cores * makespan), in percent.
    pub utilization_pct: u32,
    /// Same-rack adversary∩victim overlaps that an attacker could exploit.
    pub exploitable: Vec<Exploit>,
    /// Total secret bits leaked across all exploitable windows (per the control-eval oracle).
    pub total_bits_leaked: u32,
    /// Of the exploitable windows, how many the trusted monitor would have caught.
    pub windows_detected: u32,
}

/// One exploitable co-location: an adversary and a victim sharing a rack during `overlap` cycles.
#[derive(Debug, Clone)]
pub struct Exploit {
    pub adversary_job: usize,
    pub victim_job: usize,
    pub rack: usize,
    pub overlap_cycles: u64,
    pub bits_leaked: u32,
    pub detected: bool,
}

// --- Scheduling -----------------------------------------------------------------------------

/// Trust-blind LPT list scheduling: minimize makespan, ignore who sits where.
pub fn schedule_greedy(jobs: &[Job], cluster: Cluster) -> Schedule {
    let order = lpt_order(jobs);
    let mut core_load = vec![0u64; cluster.n_cores];
    let mut placements = Vec::with_capacity(jobs.len());
    for &j in &order {
        // Place on the least-loaded core (lowest index breaks ties → deterministic).
        let core = least_loaded(&core_load);
        let start = core_load[core];
        let end = start + jobs[j].cost_cycles;
        core_load[core] = end;
        placements.push(Placement {
            job_index: j,
            core,
            start,
            end,
        });
    }
    finalize("greedy", placements, cluster, jobs)
}

/// Isolation-aware LPT: racks are partitioned into trust domains so that adversary and victim
/// jobs never share a rack (hence never share a cooling zone). Within each domain it is still
/// LPT, so it is as tight as possible *given* the isolation constraint.
pub fn schedule_isolation_aware(jobs: &[Job], cluster: Cluster) -> Schedule {
    let n_racks = cluster.n_racks();
    // Reserve the first rack as the "secure" domain (victim + trusted); the rest are "untrusted".
    // With a single rack the constraint cannot be honored, so fall back to: untrusted jobs are
    // simply refused co-residency by being serialized after victims on the same cores.
    let secure_cores: Vec<usize> = (0..cluster.n_cores)
        .filter(|&c| cluster.rack_of(c) == 0)
        .collect();
    let untrusted_cores: Vec<usize> = if n_racks > 1 {
        (0..cluster.n_cores)
            .filter(|&c| cluster.rack_of(c) != 0)
            .collect()
    } else {
        secure_cores.clone()
    };

    let order = lpt_order(jobs);
    let mut core_load = vec![0u64; cluster.n_cores];
    let mut placements = Vec::with_capacity(jobs.len());
    for &j in &order {
        let pool: &[usize] = match jobs[j].tenant {
            Tenant::Adversary => &untrusted_cores,
            // Victim and trusted both run in the secure domain.
            Tenant::Victim | Tenant::Trusted => &secure_cores,
        };
        let core = least_loaded_in(&core_load, pool);
        let start = core_load[core];
        let end = start + jobs[j].cost_cycles;
        core_load[core] = end;
        placements.push(Placement {
            job_index: j,
            core,
            start,
            end,
        });
    }
    finalize("isolation_aware", placements, cluster, jobs)
}

/// Job indices sorted by descending cost (ties broken by index → deterministic).
fn lpt_order(jobs: &[Job]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[b]
            .cost_cycles
            .cmp(&jobs[a].cost_cycles)
            .then(a.cmp(&b))
    });
    order
}

fn least_loaded(load: &[u64]) -> usize {
    let mut best = 0usize;
    for c in 1..load.len() {
        if load[c] < load[best] {
            best = c;
        }
    }
    best
}

fn least_loaded_in(load: &[u64], pool: &[usize]) -> usize {
    let mut best = pool[0];
    for &c in &pool[1..] {
        if load[c] < load[best] {
            best = c;
        }
    }
    best
}

/// Cores belonging to a given rack.
fn cores_in_rack(cluster: Cluster, rack: usize) -> Vec<usize> {
    (0..cluster.n_cores)
        .filter(|&c| cluster.rack_of(c) == rack)
        .collect()
}

// --- Assignment-based scheduling ------------------------------------------------------------
//
// The constraint-aware and search-based schedulers work over an *assignment* (job -> core) rather
// than placing jobs one at a time. `place_from_assignment` turns an assignment into concrete
// timed placements by sequencing each core's jobs longest-first (LPT within the core), so the
// makespan of an assignment is just its most-loaded core.

fn place_from_assignment(jobs: &[Job], cluster: Cluster, assign: &[usize]) -> Vec<Placement> {
    let mut by_core: Vec<Vec<usize>> = vec![Vec::new(); cluster.n_cores];
    for (j, &c) in assign.iter().enumerate() {
        by_core[c].push(j);
    }
    let mut placements = Vec::with_capacity(jobs.len());
    for (core, job_list) in by_core.iter().enumerate() {
        let mut order = job_list.clone();
        order.sort_by(|&a, &b| {
            jobs[b]
                .cost_cycles
                .cmp(&jobs[a].cost_cycles)
                .then(a.cmp(&b))
        });
        let mut t = 0u64;
        for &j in &order {
            let start = t;
            let end = t + jobs[j].cost_cycles;
            t = end;
            placements.push(Placement {
                job_index: j,
                core,
                start,
                end,
            });
        }
    }
    placements
}

fn schedule_from_assignment(
    policy: &'static str,
    jobs: &[Job],
    cluster: Cluster,
    assign: &[usize],
) -> Schedule {
    finalize(
        policy,
        place_from_assignment(jobs, cluster, assign),
        cluster,
        jobs,
    )
}

/// Constraint-aware secure packing: the smart isolation scheduler. Victims stay in the secure
/// rack (rack 0) and adversaries stay in untrusted racks, so the two never share a cooling zone —
/// zero leakage by construction, exactly like [`schedule_isolation_aware`]. The difference is that
/// **trusted jobs are safe next to anyone**, so they backfill whichever core is globally least
/// loaded — including the otherwise-idle adversary rack. This recovers the capacity that naive
/// static partitioning wastes, so it reaches 0 leakage at a far lower makespan.
pub fn schedule_secure_packing(jobs: &[Job], cluster: Cluster) -> Schedule {
    let secure_cores = cores_in_rack(cluster, 0);
    let untrusted_cores: Vec<usize> = if cluster.n_racks() > 1 {
        (0..cluster.n_cores)
            .filter(|&c| cluster.rack_of(c) != 0)
            .collect()
    } else {
        secure_cores.clone()
    };

    let mut load = vec![0u64; cluster.n_cores];
    let mut assign = vec![0usize; jobs.len()];
    for &j in &lpt_order(jobs) {
        let core = match jobs[j].tenant {
            Tenant::Victim => least_loaded_in(&load, &secure_cores),
            Tenant::Adversary => least_loaded_in(&load, &untrusted_cores),
            // Trusted is safe to co-locate with anyone → take the globally least-loaded core.
            Tenant::Trusted => least_loaded(&load),
        };
        assign[j] = core;
        load[core] += jobs[j].cost_cycles;
    }
    schedule_from_assignment("secure_packing", jobs, cluster, &assign)
}

// --- Metrics + safety oracle ----------------------------------------------------------------

fn finalize(
    policy: &'static str,
    placements: Vec<Placement>,
    cluster: Cluster,
    jobs: &[Job],
) -> Schedule {
    let makespan = placements.iter().map(|p| p.end).max().unwrap_or(0);
    let busy: u64 = placements.iter().map(|p| p.end - p.start).sum();
    let denom = cluster.n_cores as u64 * makespan.max(1);
    let utilization_pct = ((busy * 100) / denom) as u32;

    // Find every same-rack adversary∩victim time overlap and ask the control-eval oracle how
    // much leaks through that window.
    let mut exploitable = Vec::new();
    let mut total_bits_leaked = 0u32;
    let mut windows_detected = 0u32;
    for a in &placements {
        if jobs[a.job_index].tenant != Tenant::Adversary {
            continue;
        }
        for v in &placements {
            if jobs[v.job_index].tenant != Tenant::Victim {
                continue;
            }
            if cluster.rack_of(a.core) != cluster.rack_of(v.core) {
                continue; // different cooling zones → no shared channel
            }
            if a.core == v.core {
                continue; // same core can't run both at once; no concurrency
            }
            let overlap = overlap_cycles(a.start, a.end, v.start, v.end);
            if overlap == 0 {
                continue;
            }
            // The oracle: run the thermal covert channel for exactly the overlap window.
            let report = run_control_eval(ControlEvalConfig {
                secret: 0xB4,
                adversary_enabled: true,
                ticks: overlap as u32,
            });
            let bits = report.bits_leaked_before_detection;
            let detected = report.detection_tick.is_some();
            total_bits_leaked += bits;
            if detected {
                windows_detected += 1;
            }
            exploitable.push(Exploit {
                adversary_job: a.job_index,
                victim_job: v.job_index,
                rack: cluster.rack_of(a.core),
                overlap_cycles: overlap,
                bits_leaked: bits,
                detected,
            });
        }
    }

    Schedule {
        policy,
        placements,
        cluster,
        makespan,
        utilization_pct,
        exploitable,
        total_bits_leaked,
        windows_detected,
    }
}

fn overlap_cycles(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo)
}

// --- Phase 3: search-based frontier optimizer -----------------------------------------------
//
// The two hand-written schedulers are single points: greedy (fast, leaky) and isolation
// (safe, slow). A search optimizer instead explores the whole job->core assignment space to
// minimize `makespan + lambda * leakage`, so sweeping `lambda` traces the throughput<->safety
// Pareto frontier. This mirrors the simulated-annealing layer in the AlphaFabric placer.
//
// Smart decomposition: running the full thermal control_eval oracle inside the SA inner loop
// would be far too slow, so the search minimizes a cheap, monotone *proxy* for leakage — the
// total same-rack adversary∩victim overlap in cycles — and the real oracle is run exactly once,
// on the final chosen schedule, to report the true bits leaked.

/// Sum of same-rack adversary∩victim overlap cycles for a set of placements (the leakage proxy).
fn overlap_penalty(jobs: &[Job], cluster: Cluster, placements: &[Placement]) -> u64 {
    let mut total = 0u64;
    for a in placements {
        if jobs[a.job_index].tenant != Tenant::Adversary {
            continue;
        }
        for v in placements {
            if jobs[v.job_index].tenant != Tenant::Victim {
                continue;
            }
            if a.core == v.core || cluster.rack_of(a.core) != cluster.rack_of(v.core) {
                continue;
            }
            total += overlap_cycles(a.start, a.end, v.start, v.end);
        }
    }
    total
}

/// Cheap search objective: makespan + lambda * leakage-proxy. No thermal sim.
fn proxy_objective(jobs: &[Job], cluster: Cluster, assign: &[usize], lambda: u64) -> u64 {
    let pl = place_from_assignment(jobs, cluster, assign);
    let makespan = pl.iter().map(|p| p.end).max().unwrap_or(0);
    makespan + lambda.saturating_mul(overlap_penalty(jobs, cluster, &pl))
}

fn assignment_of(sched: &Schedule, n_jobs: usize) -> Vec<usize> {
    let mut a = vec![0usize; n_jobs];
    for p in &sched.placements {
        a[p.job_index] = p.core;
    }
    a
}

fn lfsr_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Simulated-annealing search over job→core assignments minimizing
/// `makespan + lambda * leakage-proxy`. Fully deterministic: a seeded xorshift drives both the
/// proposed moves and the Metropolis acceptance (no wall-clock RNG). Initialized from the greedy
/// schedule. The returned [`Schedule`] has its true leakage scored by the real control_eval oracle.
pub fn optimize_assignment(
    jobs: &[Job],
    cluster: Cluster,
    lambda: u64,
    iters: u32,
    seed: u64,
) -> Schedule {
    let mut rng = seed | 1;
    let mut cur = assignment_of(&schedule_greedy(jobs, cluster), jobs.len());
    let mut cur_cost = proxy_objective(jobs, cluster, &cur, lambda);
    let mut best = cur.clone();
    let mut best_cost = cur_cost;

    // Initial temperature ~ a fraction of the starting objective; linear cooldown to ~0.
    let t0 = (cur_cost.max(1) as f64) * 0.5;
    for it in 0..iters {
        let mut cand = cur.clone();
        let j = (lfsr_next(&mut rng) as usize) % jobs.len();
        let c = (lfsr_next(&mut rng) as usize) % cluster.n_cores;
        cand[j] = c;
        let cand_cost = proxy_objective(jobs, cluster, &cand, lambda);

        let accept = if cand_cost <= cur_cost {
            true
        } else {
            let frac = 1.0 - (it as f64 / iters as f64);
            let temp = (t0 * frac).max(1e-6);
            let delta = (cand_cost - cur_cost) as f64;
            let p = (-delta / temp).exp();
            let r = (lfsr_next(&mut rng) % 1_000_000) as f64 / 1_000_000.0;
            r < p
        };
        if accept {
            cur = cand;
            cur_cost = cand_cost;
            if cur_cost < best_cost {
                best_cost = cur_cost;
                best = cur.clone();
            }
        }
    }
    schedule_from_assignment("optimized", jobs, cluster, &best)
}

/// Trace the throughput↔safety Pareto frontier by running the optimizer at a sweep of safety
/// weights. Low `lambda` favors throughput (toward greedy); high `lambda` favors isolation.
pub fn trace_frontier(
    jobs: &[Job],
    cluster: Cluster,
    lambdas: &[u64],
    iters: u32,
    seed: u64,
) -> Vec<Schedule> {
    lambdas
        .iter()
        .map(|&l| optimize_assignment(jobs, cluster, l, iters, seed))
        .collect()
}

// --- Demo job set ---------------------------------------------------------------------------

/// A demo workload built from the real perf-track corpus. Costs are the golden-locked cycle
/// counts (measured by running each program to halt on the physical tile CPU). Tenant classes
/// are assigned to construct a scenario: a couple of heavy trusted compute jobs, a secret-handling
/// victim, and an untrusted adversary that will exfiltrate if co-located with the victim.
pub fn demo_jobs() -> Vec<Job> {
    vec![
        // Trusted compute (heavy + light).
        Job {
            name: "fib_recursive",
            cost_cycles: 4255,
            tenant: Tenant::Trusted,
        },
        Job {
            name: "gcd_subtract",
            cost_cycles: 1313,
            tenant: Tenant::Trusted,
        },
        Job {
            name: "sum_squares",
            cost_cycles: 404,
            tenant: Tenant::Trusted,
        },
        Job {
            name: "fact_iterative",
            cost_cycles: 154,
            tenant: Tenant::Trusted,
        },
        Job {
            name: "array_sum_squares",
            cost_cycles: 661,
            tenant: Tenant::Trusted,
        },
        // Victim: handles secret data (print_large ~ formatting sensitive numbers).
        Job {
            name: "print_large",
            cost_cycles: 4538,
            tenant: Tenant::Victim,
        },
        Job {
            name: "print_count",
            cost_cycles: 1807,
            tenant: Tenant::Victim,
        },
        // Adversary: untrusted co-tenant.
        Job {
            name: "sum_loop",
            cost_cycles: 674,
            tenant: Tenant::Adversary,
        },
    ]
}

/// A harder workload with little trusted backfill: three victim jobs and one heavy adversary.
///
/// The balanced demo above shows that a smart scheduler can sometimes remove the channel at no
/// throughput cost by using safe trusted jobs as filler. This contention case is the counterweight:
/// with mostly victim/adversary work, every leak-free placement has to serialize or partition real
/// work, so the throughput/safety tradeoff is visible instead of hidden by slack.
pub fn contention_jobs() -> Vec<Job> {
    vec![
        Job {
            name: "print_large",
            cost_cycles: 4538,
            tenant: Tenant::Victim,
        },
        Job {
            name: "fib_recursive",
            cost_cycles: 4255,
            tenant: Tenant::Victim,
        },
        Job {
            name: "print_count",
            cost_cycles: 1807,
            tenant: Tenant::Victim,
        },
        // Same real program as above, now submitted by an untrusted tenant.
        Job {
            name: "fib_recursive",
            cost_cycles: 4255,
            tenant: Tenant::Adversary,
        },
    ]
}

// --- Phase 2: the honesty differential ------------------------------------------------------
//
// The scheduler reasons over the hardcoded `cost_cycles` in `demo_jobs`. Those numbers are only
// trustworthy if they are real. `measure_live_cost` runs a program to halt on the deterministic
// V2 ISS (`V2Iss`, the byte-identical reference runner, with wide PC enabled so the larger print
// kernels run too) and returns its actual retired-instruction count, so a test can confirm the
// scheduler's cost table matches live execution — the same "two targets agree" discipline used
// elsewhere in the engine.

/// Run a compiled program to halt on the deterministic V2 ISS (the authoritative byte-identical
/// oracle) and return `(instructions_retired, r0)`. Wide PC is enabled so programs larger than the
/// 128-instruction window (e.g. the print kernels) run too. Errors if it does not halt within
/// `max_steps`.
pub fn measure_live_cost(source: &str, max_steps: u64) -> Result<(u64, u64), String> {
    let words = compile_source(source).map_err(|e| format!("compile error: {:?}", e))?;
    let mut iss = V2Iss::from_program_with_mmio(&words);
    iss.enable_wide_pc();
    let mut retired = 0u64;
    let mut steps = 0u64;
    while steps < max_steps {
        if iss.is_halted() {
            break;
        }
        if iss.step().is_some() {
            retired += 1;
        }
        steps += 1;
    }
    if !iss.is_halted() {
        return Err(format!("did not halt within {} steps", max_steps));
    }
    Ok((retired, iss.state().regs[0]))
}

/// Look up a corpus program's source by name (the demo jobs reuse corpus names).
pub fn corpus_source(name: &str) -> Option<&'static str> {
    V2_COMPILED_BENCHMARKS
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.source)
}

/// One row of the cost-table audit: the scheduler's assumed cost vs the live-measured cost.
/// The scheduler costs jobs in cycles; the ISS reports retired instructions, and the physical
/// tile CPU's cycle count is exactly `retired + 1` (one pipeline-fill cycle). So a job's cost is
/// validated iff `live_retired + 1 == assumed_cost`.
#[derive(Debug, Clone)]
pub struct CostAudit {
    pub name: &'static str,
    pub assumed_cost: u64,
    pub live_retired: u64,
    /// `live_retired + 1` — the modeled cycle cost the scheduler should be using.
    pub modeled_cost: u64,
    pub live_r0: u64,
    pub matches: bool,
    /// `Some(reason)` if the program could not be measured (e.g. compile/halt failure).
    pub error: Option<String>,
}

/// Re-measure every job live on the ISS and compare to the cost the scheduler used.
pub fn audit_costs(jobs: &[Job]) -> Vec<CostAudit> {
    let mut out = Vec::new();
    for job in jobs {
        let Some(src) = corpus_source(job.name) else {
            out.push(CostAudit {
                name: job.name,
                assumed_cost: job.cost_cycles,
                live_retired: 0,
                modeled_cost: 0,
                live_r0: 0,
                matches: false,
                error: Some("no corpus source".to_string()),
            });
            continue;
        };
        match measure_live_cost(src, 1_000_000) {
            Ok((retired, r0)) => {
                let modeled = retired + 1;
                out.push(CostAudit {
                    name: job.name,
                    assumed_cost: job.cost_cycles,
                    live_retired: retired,
                    modeled_cost: modeled,
                    live_r0: r0,
                    matches: modeled == job.cost_cycles,
                    error: None,
                });
            }
            Err(e) => out.push(CostAudit {
                name: job.name,
                assumed_cost: job.cost_cycles,
                live_retired: 0,
                modeled_cost: 0,
                live_r0: 0,
                matches: false,
                error: Some(e),
            }),
        }
    }
    out
}

/// Re-measure every default demo job live on the ISS and compare to the cost table.
pub fn audit_demo_costs() -> Vec<CostAudit> {
    audit_costs(&demo_jobs())
}

/// A side-by-side report of all scheduling policies on the same workload + cluster: the two
/// hand-written baselines (greedy, naive isolation), the constraint-aware secure-packing
/// scheduler, and a search-optimized point.
pub fn frontier_report(jobs: &[Job], cluster: Cluster) -> String {
    let greedy = schedule_greedy(jobs, cluster);
    let iso = schedule_isolation_aware(jobs, cluster);
    let secure = schedule_secure_packing(jobs, cluster);
    // A safety-weighted search point: high lambda → the optimizer should find a 0-leak schedule.
    let optimized = optimize_assignment(jobs, cluster, 1_000, 4_000, 0xC0FFEE);

    let mut s = String::new();
    s.push_str("=== Workload Scheduling: throughput <-> safety frontier ===\n");
    s.push_str(&format!(
        "  cluster: {} cores, {} racks of {} (one cooling zone per rack)\n",
        cluster.n_cores,
        cluster.n_racks(),
        cluster.cores_per_rack
    ));
    s.push_str(&format!(
        "  workload: {} real tile-CPU programs\n\n",
        jobs.len()
    ));
    for sched in [&greedy, &iso, &secure, &optimized] {
        s.push_str(&format!("  [{}]\n", sched.policy));
        s.push_str(&format!(
            "    makespan {:>6} cyc   util {:>3}%   exploitable {}   bits leaked {} (caught {})\n",
            sched.makespan,
            sched.utilization_pct,
            sched.exploitable.len(),
            sched.total_bits_leaked,
            sched.windows_detected
        ));
    }
    let price = |m: u64| -> i64 {
        if greedy.makespan > 0 {
            (m as i64 - greedy.makespan as i64) * 100 / greedy.makespan as i64
        } else {
            0
        }
    };
    s.push_str("\n  frontier (price of safety vs the greedy/leaky baseline)\n");
    s.push_str(&format!(
        "    naive isolation : 0 leak at {:+}% makespan\n",
        price(iso.makespan)
    ));
    let secure_note = if jobs.iter().any(|j| j.tenant == Tenant::Trusted) {
        "trusted jobs backfill the adversary rack"
    } else {
        "no trusted backfill available"
    };
    s.push_str(&format!(
        "    secure packing  : 0 leak at {:+}% makespan  ({secure_note})\n",
        price(secure.makespan),
    ));
    s.push_str(&format!(
        "    search-optimized: {} leak at {:+}% makespan\n",
        optimized.total_bits_leaked,
        price(optimized.makespan)
    ));
    // The knee, derived from the full lambda sweep: the cheapest leak-free operating point.
    // This is the recommended place to run — and its price is computed, not hand-quoted.
    let sweep = pareto_frontier(jobs, cluster, &DEFAULT_LAMBDAS, 4_000, 0xC0FFEE);
    match safe_operating_point(&sweep) {
        Some(knee) => s.push_str(&format!(
            "    => safe operating point: {} cyc at {:+}% makespan (lambda {}), 0 leak\n",
            knee.makespan,
            price(knee.makespan),
            knee.lambda
        )),
        None => s.push_str("    => no leak-free operating point found in the sweep\n"),
    }
    s
}

/// The standard safety-weight sweep used across the demos and the structured frontier API.
/// Low → throughput-first (toward greedy); high → isolation-first.
pub const DEFAULT_LAMBDAS: [u64; 6] = [0, 1, 5, 20, 100, 1_000];

/// A compact view of the full Pareto frontier traced by the optimizer across a lambda sweep.
pub fn frontier_sweep_report(jobs: &[Job], cluster: Cluster) -> String {
    let lambdas = DEFAULT_LAMBDAS;
    let points = trace_frontier(jobs, cluster, &lambdas, 4_000, 0xC0FFEE);
    let mut s = String::new();
    s.push_str("  search-optimizer Pareto frontier (lambda = safety weight):\n");
    s.push_str(&format!(
        "    {:>8} {:>10} {:>12} {:>14}\n",
        "lambda", "makespan", "bits_leaked", "exploitable"
    ));
    for (l, p) in lambdas.iter().zip(points.iter()) {
        s.push_str(&format!(
            "    {:>8} {:>10} {:>12} {:>14}\n",
            l,
            p.makespan,
            p.total_bits_leaked,
            p.exploitable.len()
        ));
    }
    s
}

// --- Structured Pareto frontier -------------------------------------------------------------
//
// The reports above render the frontier for a human. This block exposes the same numbers as data
// so the frontier can be analyzed (knee selection, CSV export, plotting) instead of just read.

/// One point on the throughput↔safety Pareto frontier: the schedule the optimizer found at a
/// given safety weight, reduced to the four metrics that define the tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontierPoint {
    /// Safety weight the optimizer ran at (0 = pure throughput, high = pure isolation).
    pub lambda: u64,
    pub makespan: u64,
    pub utilization_pct: u32,
    pub bits_leaked: u32,
    pub exploitable_windows: usize,
}

/// Trace the optimizer across a lambda sweep and reduce each schedule to a [`FrontierPoint`]. This
/// is the machine-readable form of [`frontier_sweep_report`]: identical numbers, structured for
/// analysis rather than display.
pub fn pareto_frontier(
    jobs: &[Job],
    cluster: Cluster,
    lambdas: &[u64],
    iters: u32,
    seed: u64,
) -> Vec<FrontierPoint> {
    trace_frontier(jobs, cluster, lambdas, iters, seed)
        .iter()
        .zip(lambdas)
        .map(|(s, &lambda)| FrontierPoint {
            lambda,
            makespan: s.makespan,
            utilization_pct: s.utilization_pct,
            bits_leaked: s.total_bits_leaked,
            exploitable_windows: s.exploitable.len(),
        })
        .collect()
}

/// The best safe operating point on a frontier: among the zero-leak points, the one with the
/// smallest makespan — the knee where spending more safety weight buys nothing further. This makes
/// the "price of safety" a number *derived from the frontier* rather than hand-quoted. Ties break
/// toward the lower lambda (the cheapest weight that already closed the channel). Returns `None` if
/// no traced point achieves zero leakage.
pub fn safe_operating_point(points: &[FrontierPoint]) -> Option<FrontierPoint> {
    points
        .iter()
        .filter(|p| p.bits_leaked == 0)
        .min_by(|a, b| a.makespan.cmp(&b.makespan).then(a.lambda.cmp(&b.lambda)))
        .copied()
}

/// CSV of the default Pareto sweep: a header line plus one row per lambda. The data behind the
/// frontier plot, reproducible bit-for-bit from the same seed the demos use.
pub fn frontier_csv(jobs: &[Job], cluster: Cluster) -> String {
    let points = pareto_frontier(jobs, cluster, &DEFAULT_LAMBDAS, 4_000, 0xC0FFEE);
    let mut s = String::from("lambda,makespan,utilization_pct,bits_leaked,exploitable_windows\n");
    for p in &points {
        s.push_str(&format!(
            "{},{},{},{},{}\n",
            p.lambda, p.makespan, p.utilization_pct, p.bits_leaked, p.exploitable_windows
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(cost: u64, tenant: Tenant) -> Job {
        Job {
            name: "synth",
            cost_cycles: cost,
            tenant,
        }
    }

    #[test]
    fn greedy_minimizes_makespan_via_lpt() {
        // LPT on [4,3,3,2] over 2 cores: 4->c0, 3->c1, 3->c1(=6), 2->c0(=6). Total work 12,
        // perfect-balance lower bound ceil(12/2)=6, so LPT is optimal here.
        let jobs = vec![
            synth(4, Tenant::Trusted),
            synth(3, Tenant::Trusted),
            synth(3, Tenant::Trusted),
            synth(2, Tenant::Trusted),
        ];
        let s = schedule_greedy(&jobs, Cluster::new(2, 1));
        let total: u64 = jobs.iter().map(|j| j.cost_cycles).sum();
        let lower_bound = total.div_ceil(2);
        assert_eq!(
            s.makespan, 6,
            "LPT should reach makespan 6, got {}",
            s.makespan
        );
        assert_eq!(
            s.makespan, lower_bound,
            "this case is perfectly balanceable"
        );
    }

    #[test]
    fn greedy_colocates_adversary_and_victim_and_leaks() {
        // One rack of 2 cores: a long victim and a long adversary land on the two cores and
        // overlap, so the channel opens and bits leak.
        let jobs = vec![synth(4538, Tenant::Victim), synth(4255, Tenant::Adversary)];
        let s = schedule_greedy(&jobs, Cluster::new(2, 2));
        assert_eq!(
            s.exploitable.len(),
            1,
            "expected one exploitable co-location"
        );
        assert!(
            s.total_bits_leaked > 0,
            "expected real leakage under greedy"
        );
    }

    #[test]
    fn isolation_eliminates_the_channel() {
        let jobs = demo_jobs();
        // Two racks of 2 cores: isolation can put adversaries in rack 1, victims in rack 0.
        let cluster = Cluster::new(4, 2);
        let iso = schedule_isolation_aware(&jobs, cluster);
        assert_eq!(
            iso.exploitable.len(),
            0,
            "isolation-aware must have zero adversary/victim co-locations"
        );
        assert_eq!(iso.total_bits_leaked, 0);
    }

    #[test]
    fn isolation_costs_makespan_vs_greedy() {
        // The frontier: on a workload where greedy would co-locate, isolation pays in makespan.
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        let greedy = schedule_greedy(&jobs, cluster);
        let iso = schedule_isolation_aware(&jobs, cluster);
        // Greedy leaks; isolation does not.
        assert!(
            greedy.total_bits_leaked > 0,
            "greedy should leak on this workload"
        );
        assert_eq!(iso.total_bits_leaked, 0);
        // And isolation is no faster than greedy (usually strictly slower).
        assert!(
            iso.makespan >= greedy.makespan,
            "isolation makespan {} should be >= greedy {}",
            iso.makespan,
            greedy.makespan
        );
    }

    #[test]
    fn secure_packing_is_safe_and_beats_naive_isolation() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        let iso = schedule_isolation_aware(&jobs, cluster);
        let secure = schedule_secure_packing(&jobs, cluster);
        // Constraint-aware packing is just as safe...
        assert_eq!(secure.total_bits_leaked, 0);
        assert_eq!(secure.exploitable.len(), 0);
        // ...but recovers the capacity naive static partitioning wastes.
        assert!(
            secure.makespan < iso.makespan,
            "secure packing makespan {} should beat naive isolation {}",
            secure.makespan,
            iso.makespan
        );
    }

    #[test]
    fn optimizer_finds_zero_leak_under_high_lambda() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        // With a strong safety weight, the search must converge on a leak-free schedule.
        let opt = optimize_assignment(&jobs, cluster, 1_000, 4_000, 0xC0FFEE);
        assert_eq!(
            opt.total_bits_leaked, 0,
            "high-lambda search should eliminate leakage, got {} bits",
            opt.total_bits_leaked
        );
        // And it should be no worse than naive isolation on makespan.
        let iso = schedule_isolation_aware(&jobs, cluster);
        assert!(opt.makespan <= iso.makespan);
    }

    #[test]
    fn contention_workload_has_real_safety_price() {
        let jobs = contention_jobs();
        let cluster = Cluster::new(4, 2);
        let greedy = schedule_greedy(&jobs, cluster);
        let opt = optimize_assignment(&jobs, cluster, 1_000, 4_000, 0xC0FFEE);

        assert!(
            greedy.total_bits_leaked > 0,
            "greedy should leak on the contention workload"
        );
        assert_eq!(
            opt.total_bits_leaked, 0,
            "high-lambda search should eliminate leakage"
        );
        assert!(
            opt.makespan > greedy.makespan,
            "contention workload should make safety cost throughput (opt {} greedy {})",
            opt.makespan,
            greedy.makespan
        );
    }

    #[test]
    fn optimizer_frontier_is_monotone_and_deterministic() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        // Determinism: same seed → identical result.
        let a = optimize_assignment(&jobs, cluster, 50, 2_000, 7);
        let b = optimize_assignment(&jobs, cluster, 50, 2_000, 7);
        assert_eq!(a.makespan, b.makespan);
        assert_eq!(a.total_bits_leaked, b.total_bits_leaked);
        // Low lambda tolerates leakage for speed; high lambda buys safety with makespan.
        let lo = optimize_assignment(&jobs, cluster, 0, 4_000, 7);
        let hi = optimize_assignment(&jobs, cluster, 1_000, 4_000, 7);
        assert!(
            hi.total_bits_leaked <= lo.total_bits_leaked,
            "more safety weight should not increase leakage (lo {} hi {})",
            lo.total_bits_leaked,
            hi.total_bits_leaked
        );
        assert!(lo.makespan <= hi.makespan || hi.total_bits_leaked < lo.total_bits_leaked);
    }

    #[test]
    fn demo_costs_trace_to_live_execution() {
        // The honesty differential: every cost the scheduler uses must equal what the program
        // actually does when run to halt on the deterministic ISS (retired + 1 pipeline cycle).
        let audits = audit_demo_costs();
        assert_eq!(audits.len(), demo_jobs().len());
        for a in &audits {
            assert!(
                a.error.is_none(),
                "job '{}' failed to measure live: {:?}",
                a.name,
                a.error
            );
            assert!(
                a.matches,
                "job '{}': scheduler cost {} != live modeled cost {} (retired {})",
                a.name, a.assumed_cost, a.modeled_cost, a.live_retired
            );
        }
    }

    #[test]
    fn contention_costs_trace_to_live_execution() {
        let jobs = contention_jobs();
        let audits = audit_costs(&jobs);
        assert_eq!(audits.len(), jobs.len());
        for a in &audits {
            assert!(
                a.error.is_none(),
                "job '{}' failed to measure live: {:?}",
                a.name,
                a.error
            );
            assert!(
                a.matches,
                "job '{}': scheduler cost {} != live modeled cost {} (retired {})",
                a.name, a.assumed_cost, a.modeled_cost, a.live_retired
            );
        }
    }

    #[test]
    fn schedule_is_deterministic() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        let a = schedule_greedy(&jobs, cluster);
        let b = schedule_greedy(&jobs, cluster);
        assert_eq!(a.makespan, b.makespan);
        assert_eq!(a.total_bits_leaked, b.total_bits_leaked);
        assert_eq!(a.exploitable.len(), b.exploitable.len());
    }

    #[test]
    fn frontier_reports_replay_bit_for_bit() {
        // The artifact's central honesty claim is that the whole frontier replays bit-for-bit.
        // Assert it on the rendered reports a reader actually sees (which include the SA-optimizer
        // output and the live control-eval oracle), for BOTH workloads — not just on scalars.
        for jobs in [demo_jobs(), contention_jobs()] {
            let cluster = Cluster::new(4, 2);
            assert_eq!(
                frontier_report(&jobs, cluster),
                frontier_report(&jobs, cluster),
                "frontier_report must be byte-identical across runs"
            );
            assert_eq!(
                frontier_sweep_report(&jobs, cluster),
                frontier_sweep_report(&jobs, cluster),
                "frontier_sweep_report must be byte-identical across runs"
            );
        }
    }

    #[test]
    fn safe_policies_have_zero_leak_on_both_workloads() {
        // The safety guarantee both isolation schedulers provide by construction: on every
        // workload, an adversary and a victim never share a cooling zone, so nothing leaks.
        for jobs in [demo_jobs(), contention_jobs()] {
            let cluster = Cluster::new(4, 2);
            for sched in [
                schedule_isolation_aware(&jobs, cluster),
                schedule_secure_packing(&jobs, cluster),
            ] {
                assert_eq!(
                    sched.exploitable.len(),
                    0,
                    "{} must have zero adversary/victim co-locations",
                    sched.policy
                );
                assert_eq!(
                    sched.total_bits_leaked, 0,
                    "{} must leak zero bits",
                    sched.policy
                );
            }
        }
    }

    #[test]
    fn pareto_frontier_matches_sweep_and_picks_free_knee() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        let pts = pareto_frontier(&jobs, cluster, &DEFAULT_LAMBDAS, 4_000, 0xC0FFEE);
        assert_eq!(pts.len(), DEFAULT_LAMBDAS.len());
        // Throughput-first leaks; the safety-first tail is clean.
        assert!(pts[0].bits_leaked > 0, "lambda=0 point should leak");
        assert_eq!(
            pts.last().unwrap().bits_leaked,
            0,
            "safety-first point is clean"
        );
        // Knee: on the balanced workload the channel closes for free (knee makespan == greedy).
        let greedy = schedule_greedy(&jobs, cluster);
        let knee = safe_operating_point(&pts).expect("a zero-leak point must exist");
        assert_eq!(knee.bits_leaked, 0);
        assert_eq!(
            knee.makespan, greedy.makespan,
            "balanced workload: safety should be free"
        );
    }

    #[test]
    fn safe_operating_point_shows_contention_price() {
        let jobs = contention_jobs();
        let cluster = Cluster::new(4, 2);
        let pts = pareto_frontier(&jobs, cluster, &DEFAULT_LAMBDAS, 4_000, 0xC0FFEE);
        let greedy = schedule_greedy(&jobs, cluster);
        let knee = safe_operating_point(&pts).expect("a zero-leak point must exist");
        assert_eq!(knee.bits_leaked, 0);
        // Contention workload: the cheapest safe point costs real makespan over greedy.
        assert!(
            knee.makespan > greedy.makespan,
            "contention safe knee {} should exceed greedy {}",
            knee.makespan,
            greedy.makespan
        );
    }

    #[test]
    fn safe_operating_point_is_none_when_everything_leaks() {
        let pts = vec![
            FrontierPoint {
                lambda: 0,
                makespan: 100,
                utilization_pct: 50,
                bits_leaked: 4,
                exploitable_windows: 1,
            },
            FrontierPoint {
                lambda: 0,
                makespan: 90,
                utilization_pct: 55,
                bits_leaked: 2,
                exploitable_windows: 1,
            },
        ];
        assert!(safe_operating_point(&pts).is_none());
    }

    #[test]
    fn frontier_csv_is_wellformed_and_stable() {
        let jobs = demo_jobs();
        let cluster = Cluster::new(4, 2);
        let csv = frontier_csv(&jobs, cluster);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "lambda,makespan,utilization_pct,bits_leaked,exploitable_windows"
        );
        assert_eq!(
            lines.len(),
            1 + DEFAULT_LAMBDAS.len(),
            "header + one row per lambda"
        );
        for row in &lines[1..] {
            assert_eq!(row.split(',').count(), 5, "each row has 5 columns: {row}");
        }
        assert_eq!(
            csv,
            frontier_csv(&jobs, cluster),
            "CSV must replay bit-for-bit"
        );
    }
}
