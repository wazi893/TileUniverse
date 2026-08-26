//! AF2 — a simulated-annealing placer over [`PlacementEnv`].
//!
//! This is the *classical baseline* the AlphaFabric roadmap insists on before
//! any learning: for a single placement instance, well-tuned simulated annealing
//! is the bar a learned policy must clear.
//!
//! ## Why HPWL in the inner loop
//!
//! Routing the full netlist on every move is far too slow for thousands of SA
//! steps. So — exactly like a production placer — annealing optimizes the cheap
//! [half-perimeter wirelength](PlacementEnv::hpwl) proxy (O(pins), no routing),
//! and the router runs only twice: once on the baseline and once on the best
//! layout found, to report true routed metrics and the legality gate. HPWL
//! correlates strongly with routed wirelength, so minimizing it minimizes the
//! real cost while keeping the search interactive.
//!
//! Determinism is a project invariant, so randomness comes from a fixed-seed
//! SplitMix64 generator — same seed and config gives the same result every run.
//! The environment is left holding the best layout found.

use super::env::PlacementEnv;
use super::reward::PlacementScore;

/// Simulated-annealing configuration.
#[derive(Clone, Copy, Debug)]
pub struct AnnealConfig {
    pub iterations: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub seed: u64,
    /// Probability of proposing a `relocate` (to an empty slot) rather than a
    /// `swap`. Relocate moves explore geometry beyond pure permutations.
    pub relocate_prob: f64,
    /// If true, start from the env's current assignment (e.g. a learned
    /// warm-start) instead of resetting to the row-major baseline.
    pub start_from_current: bool,
    /// Weight on the criticality-weighted-wirelength timing term
    /// ([`PlacementEnv::timing_cost`]) in the SA objective. 0.0 (default) anneals
    /// pure HPWL, reproducing the legacy trajectory exactly. Positive values make
    /// the search timing-driven: it spends wirelength to pull low-slack nets
    /// short. Both terms are O(pins) proxies, so the inner loop stays routing-free.
    pub timing_weight: f64,
    /// If true, a candidate may only become the committed *best* after a bounded
    /// route check on a proxy-improvement event (not every step). The search
    /// trajectory is unchanged — only best-tracking is gated — so the run favors
    /// low-objective layouts that still route without putting the full router in
    /// the annealing inner loop. This recovers routability when an aggressive
    /// `timing_weight` over-packs a congested block. Default false (no extra
    /// routing, legacy behavior).
    pub route_validated_best: bool,
    /// AF7: weight on the congestion proxy ([`PlacementEnv::congestion_cost`],
    /// Σ occupied-8-neighbor² over gates) in the SA objective. 0.0 (default)
    /// reproduces the legacy trajectory exactly. Positive values make the search
    /// resist over-packing — the *preventive* counterpart to the
    /// `route_validated_best` band-aid for the aggressive-timing-weight failure
    /// mode. Still routing-free in the inner loop.
    pub congestion_weight: f64,
}

impl Default for AnnealConfig {
    fn default() -> Self {
        Self {
            iterations: 6000,
            t_start: 25.0,
            t_end: 0.1,
            seed: 0x5EED_A1F0,
            relocate_prob: 0.5,
            start_from_current: false,
            timing_weight: 0.0,
            route_validated_best: false,
            congestion_weight: 0.0,
        }
    }
}

/// Outcome of an annealing run.
#[derive(Clone, Debug)]
pub struct AnnealResult {
    /// Routed score of the row-major baseline the run started from.
    pub baseline: PlacementScore,
    /// Routed score of the best layout found (the env is left holding this).
    pub best: PlacementScore,
    /// HPWL proxy of the baseline.
    pub baseline_hpwl: usize,
    /// HPWL proxy of the best layout (what SA actually minimized).
    pub best_hpwl: usize,
    /// Criticality-weighted wirelength of the baseline layout.
    pub baseline_timing: f64,
    /// Criticality-weighted wirelength of the best layout. With a positive
    /// `timing_weight` this is what the search drove down; with weight 0 it is
    /// just reported (whatever the HPWL-optimal layout happened to score).
    pub best_timing: f64,
    /// `slot_of_gate` assignment for the best layout.
    pub best_assignment: Vec<usize>,
    /// Number of proposed moves accepted.
    pub accepted: usize,
    /// Number of moves proposed.
    pub iterations: usize,
}

impl AnnealResult {
    /// Fractional routed-cost reduction vs the baseline (0.0 if none).
    pub fn improvement(&self) -> f64 {
        if self.baseline.cost <= 0.0 || !self.baseline.metrics.routable {
            return 0.0;
        }
        ((self.baseline.cost - self.best.cost) / self.baseline.cost).max(0.0)
    }

    /// Fractional HPWL reduction vs the baseline (what SA directly optimized).
    pub fn hpwl_improvement(&self) -> f64 {
        if self.baseline_hpwl == 0 {
            return 0.0;
        }
        let saved = self.baseline_hpwl.saturating_sub(self.best_hpwl) as f64;
        saved / self.baseline_hpwl as f64
    }

    /// Fractional reduction in criticality-weighted wirelength vs the baseline
    /// (the timing-driven win; 0.0 if none or no critical nets).
    pub fn timing_improvement(&self) -> f64 {
        if self.baseline_timing <= 0.0 {
            return 0.0;
        }
        ((self.baseline_timing - self.best_timing) / self.baseline_timing).max(0.0)
    }
}

/// Deterministic SplitMix64 PRNG (no external dependency; reproducible).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (n must be > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform f64 in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A proposed move plus the information needed to undo it.
enum Move {
    Swap(usize, usize),
    Relocate { gate: usize, from: usize, to: usize },
}

/// Run simulated annealing on `env`, leaving it holding the best layout found.
pub fn anneal(env: &mut PlacementEnv, config: &AnnealConfig) -> AnnealResult {
    if !config.start_from_current {
        env.reset();
    }
    let baseline = env.score();
    let baseline_hpwl = env.hpwl();
    let baseline_timing = env.timing_cost();
    let n = env.num_gates();

    let mut best_assignment = env.assignment().to_vec();

    // Nothing to optimize with fewer than two gates.
    if n < 2 || config.iterations == 0 {
        return AnnealResult {
            baseline,
            best: baseline,
            baseline_hpwl,
            best_hpwl: baseline_hpwl,
            baseline_timing,
            best_timing: baseline_timing,
            best_assignment,
            accepted: 0,
            iterations: 0,
        };
    }

    let mut rng = SplitMix64::new(config.seed);
    // SA minimizes this objective (HPWL plus the optional timing and congestion
    // terms). With both weights 0 it is exactly HPWL — the legacy trajectory.
    let mut current = objective(env, config.timing_weight, config.congestion_weight);
    // Track the lowest-objective layout. When route-validating, the baseline is
    // only an eligible best if it routes; otherwise start from +inf so the first
    // routable improvement seeds the best.
    let mut best_obj = if !config.route_validated_best || baseline.metrics.routable {
        current
    } else {
        f64::INFINITY
    };
    let cooling = (config.t_end / config.t_start).powf(1.0 / config.iterations as f64);
    let mut t = config.t_start;
    let mut accepted = 0usize;
    // Route-validation rate limit: at most ~64 routes per run, spread evenly
    // across the schedule so late (lower-objective) layouts are still checked.
    // Only consulted when route_validated_best is set.
    let route_stride = (config.iterations / 64).max(1);
    let mut last_route_check = 0usize;

    for iter in 0..config.iterations {
        let mv = propose(env, &mut rng, config.relocate_prob);
        apply(env, &mv);

        let cand = objective(env, config.timing_weight, config.congestion_weight);
        let delta = cand - current;
        let accept = delta <= 0.0 || rng.unit() < (-delta / t).exp();

        if accept {
            current = cand;
            accepted += 1;
            if cand < best_obj {
                if !config.route_validated_best {
                    // Default path: commit the proxy improvement directly (no
                    // routing — the legacy trajectory is preserved exactly).
                    best_obj = cand;
                    best_assignment.copy_from_slice(env.assignment());
                } else if iter - last_route_check >= route_stride {
                    // Rate-limited validation: commit only if this layout routes.
                    // A within-cooldown improvement is skipped; a later one past
                    // the stride captures the gain (the objective keeps falling).
                    last_route_check = iter;
                    if env.score().metrics.routable {
                        best_obj = cand;
                        best_assignment.copy_from_slice(env.assignment());
                    }
                }
            }
        } else {
            undo(env, &mv);
        }

        t *= cooling;
    }

    // Leave the environment holding the best layout, and route it once for truth.
    env.restore_assignment(&best_assignment);
    let best_hpwl = env.hpwl();
    let best_timing = env.timing_cost();
    let best = env.score();

    AnnealResult {
        baseline,
        best,
        baseline_hpwl,
        best_hpwl,
        baseline_timing,
        best_timing,
        best_assignment,
        accepted,
        iterations: config.iterations,
    }
}

/// SA search objective: HPWL plus the optional criticality-weighted timing and
/// congestion terms. With both weights `0.0` this is exactly the HPWL proxy
/// (the extra terms are skipped, not just zero-weighted, so the legacy path
/// pays no extra cost).
fn objective(env: &PlacementEnv, timing_weight: f64, congestion_weight: f64) -> f64 {
    let mut obj = env.hpwl() as f64;
    if timing_weight != 0.0 {
        obj += timing_weight * env.timing_cost();
    }
    if congestion_weight != 0.0 {
        obj += congestion_weight * env.congestion_cost();
    }
    obj
}

fn propose(env: &PlacementEnv, rng: &mut SplitMix64, relocate_prob: f64) -> Move {
    let n = env.num_gates();
    let empty = env.empty_slots();
    if !empty.is_empty() && rng.unit() < relocate_prob {
        let gate = rng.below(n);
        let from = env.assignment()[gate];
        let to = empty[rng.below(empty.len())];
        Move::Relocate { gate, from, to }
    } else {
        let a = rng.below(n);
        let mut b = rng.below(n);
        if a == b {
            b = (b + 1) % n;
        }
        Move::Swap(a, b)
    }
}

fn apply(env: &mut PlacementEnv, mv: &Move) {
    match *mv {
        Move::Swap(a, b) => {
            env.swap(a, b).expect("swap of valid gates");
        }
        Move::Relocate { gate, to, .. } => {
            env.relocate(gate, to).expect("relocate to empty slot");
        }
    }
}

fn undo(env: &mut PlacementEnv, mv: &Move) {
    match *mv {
        Move::Swap(a, b) => {
            env.swap(a, b).expect("swap is its own inverse");
        }
        Move::Relocate { gate, from, .. } => {
            env.relocate(gate, from).expect("origin slot is now empty");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::alphafabric::Circuit;
    use crate::synth::alphafabric::corpus::{build_eq_comparator, build_ripple_adder};

    fn cfg(iterations: usize) -> AnnealConfig {
        AnnealConfig {
            iterations,
            ..AnnealConfig::default()
        }
    }

    #[test]
    fn anneal_reduces_hpwl_and_does_not_regress_routed_cost() {
        let c = Circuit::from_aig("adder4", build_ripple_adder(4));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let r = anneal(&mut env, &cfg(4000));
        assert!(r.best.metrics.routable, "best layout must route");
        assert!(
            r.best_hpwl <= r.baseline_hpwl,
            "SA must not increase HPWL: best {} > baseline {}",
            r.best_hpwl,
            r.baseline_hpwl
        );
        assert!(
            r.best.cost <= r.baseline.cost,
            "routed cost should not regress: best {} > baseline {}",
            r.best.cost,
            r.baseline.cost
        );
    }

    #[test]
    fn anneal_improves_a_real_circuit() {
        // Deterministic seed -> a stable, reproducible HPWL improvement.
        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let r = anneal(&mut env, &cfg(5000));
        assert!(
            r.best_hpwl < r.baseline_hpwl,
            "expected SA to beat row-major HPWL (baseline {}, best {})",
            r.baseline_hpwl,
            r.best_hpwl
        );
    }

    #[test]
    fn anneal_is_deterministic() {
        let c = Circuit::from_aig("adder4", build_ripple_adder(4));
        let mut e1 = PlacementEnv::new(&c).expect("env builds");
        let mut e2 = PlacementEnv::new(&c).expect("env builds");
        let r1 = anneal(&mut e1, &cfg(3000));
        let r2 = anneal(&mut e2, &cfg(3000));
        assert_eq!(r1.best_assignment, r2.best_assignment);
        assert_eq!(r1.best_hpwl, r2.best_hpwl);
        assert_eq!(r1.best.cost, r2.best.cost);
        assert_eq!(r1.accepted, r2.accepted);
    }

    #[test]
    fn timing_weight_drives_down_critical_wirelength() {
        // On a deep-critical-path circuit (ripple carry), a positive timing
        // weight should yield a layout with lower criticality-weighted wirelength
        // than pure-HPWL SA — the timing-driven-placement win. Both runs use the
        // cheap proxy objective (no per-step routing); routability is asserted
        // off the final routed score, and physical correctness is covered by the
        // adder4 test, so no exhaustive verify is needed here.
        let c = Circuit::from_aig("adder8", build_ripple_adder(8));

        let mut env_hpwl = PlacementEnv::new(&c).expect("env builds");
        let hpwl_only = anneal(&mut env_hpwl, &cfg(3000));

        let timed_cfg = AnnealConfig {
            timing_weight: 6.0,
            ..cfg(3000)
        };
        let mut env_timed = PlacementEnv::new(&c).expect("env builds");
        let timed = anneal(&mut env_timed, &timed_cfg);

        assert!(timed.best.metrics.routable, "timing-aware best must route");
        assert!(
            timed.best_timing < hpwl_only.best_timing,
            "timing-aware SA should shorten critical nets: timed {} vs hpwl-only {}",
            timed.best_timing,
            hpwl_only.best_timing
        );
        assert!(
            timed.timing_improvement() > 0.0,
            "should beat its own baseline"
        );
    }

    #[test]
    fn timing_weight_zero_is_byte_identical_to_legacy() {
        // The default (timing_weight 0) must reproduce pure-HPWL annealing
        // exactly — same best layout, HPWL, and acceptance count.
        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut e1 = PlacementEnv::new(&c).expect("env builds");
        let mut e2 = PlacementEnv::new(&c).expect("env builds");
        let legacy = anneal(&mut e1, &cfg(4000));
        let explicit_zero = anneal(
            &mut e2,
            &AnnealConfig {
                timing_weight: 0.0,
                ..cfg(4000)
            },
        );
        assert_eq!(legacy.best_assignment, explicit_zero.best_assignment);
        assert_eq!(legacy.best_hpwl, explicit_zero.best_hpwl);
        assert_eq!(legacy.accepted, explicit_zero.accepted);
        assert_eq!(legacy.best.cost, explicit_zero.best.cost);
    }

    #[test]
    fn route_validated_best_returns_routable_layout() {
        // route_validated_best gates best-tracking on routability, so the run
        // returns a routable best by construction. Keep default coverage tiny:
        // larger congested cases belong in examples/slow-tests, not every local
        // `cargo test`.
        let c = Circuit::from_aig("adder2", build_ripple_adder(2));
        let cfg_safe = AnnealConfig {
            iterations: 32,
            route_validated_best: true,
            ..AnnealConfig::default()
        };
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let r = anneal(&mut env, &cfg_safe);
        assert!(
            r.best.metrics.routable,
            "route-validated best must always route"
        );
    }

    #[test]
    fn congestion_weight_zero_is_byte_identical_to_legacy() {
        // The default (congestion_weight 0) must reproduce the existing
        // trajectory exactly — the term is skipped, not zero-weighted.
        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut e1 = PlacementEnv::new(&c).expect("env builds");
        let mut e2 = PlacementEnv::new(&c).expect("env builds");
        let legacy = anneal(&mut e1, &cfg(4000));
        let explicit_zero = anneal(
            &mut e2,
            &AnnealConfig {
                congestion_weight: 0.0,
                ..cfg(4000)
            },
        );
        assert_eq!(legacy.best_assignment, explicit_zero.best_assignment);
        assert_eq!(legacy.best_hpwl, explicit_zero.best_hpwl);
        assert_eq!(legacy.accepted, explicit_zero.accepted);
    }

    #[test]
    fn congestion_weight_restores_routability_on_overpacked_mul4() {
        // AF7: at timing_weight 6 the mul4 layout over-packs and fails to route
        // (the timing-track step-5 failure mode). A moderate congestion weight
        // PREVENTS the over-packing in the objective itself — routing-free in
        // the inner loop, unlike the route_validated_best band-aid — and the
        // timing win is kept (measured at this seed: c=0 timing 1105 unroutable
        // vs c=0.5 timing 1065 routable; baseline row-major ~1290).
        //
        // Multi-seed context (measured with EVAL_SEEDS, recorded in the AF6
        // loop log): pure-HPWL SA on mul4 routes for only 2/5 seeds and t=6/c=0
        // for 3/5 — SA-compacted mul4 routability is fragile regardless of
        // objective; c=0.5 was the only swept config routable on ALL 5 seeds
        // (suggestive at n=5, stated as such, not asserted).
        use crate::synth::benchmark::build_4bit_multiplier;
        let c = Circuit::from_aig("mul4", build_4bit_multiplier());
        let seed = 0xA11C_E5ED;

        let mut env0 = PlacementEnv::new(&c).expect("env builds");
        let over = anneal(
            &mut env0,
            &AnnealConfig {
                seed,
                timing_weight: 6.0,
                ..AnnealConfig::default()
            },
        );
        assert!(
            !over.best.metrics.routable,
            "the failure mode must exist at this seed for the test to be meaningful"
        );

        let mut env = PlacementEnv::new(&c).expect("env builds");
        let fixed = anneal(
            &mut env,
            &AnnealConfig {
                seed,
                timing_weight: 6.0,
                congestion_weight: 0.5,
                ..AnnealConfig::default()
            },
        );
        assert!(
            fixed.best.metrics.routable,
            "congestion weight 0.5 should restore routability"
        );
        assert!(
            fixed.best_timing <= over.best_timing,
            "the congestion term should not cost the timing win here: \
             c=0.5 {:.0} vs c=0 {:.0}",
            fixed.best_timing,
            over.best_timing
        );
        assert!(
            fixed.timing_improvement() > 0.0,
            "still a timing win over the row-major baseline"
        );
    }

    #[test]
    fn annealed_layout_is_physically_correct() {
        // The optimized placement must still compute correctly on real tiles
        // (the env is left holding the best layout).
        let c = Circuit::from_aig("adder4", build_ripple_adder(4));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let r = anneal(&mut env, &cfg(3000));
        assert_eq!(env.score().cost, r.best.cost, "env holds the best layout");
        assert!(
            env.verify_physical(),
            "annealed layout must be physically correct"
        );
    }
}
