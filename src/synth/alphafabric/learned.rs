//! AF4/AF5 — a learned constructive placer.
//!
//! The roadmap's Phase 2 calls for a *learned* placer whose honest win is
//! **generalization**, with "simple featurization first; a GNN is an upgrade".
//! This module is that: a greedy constructive placer driven by a small linear
//! policy, with the policy fit from a circuit family by black-box optimization.
//!
//! ## The policy (two learned parts)
//!
//! - **Slot policy** ([`PolicyWeights`]): when placing a gate, every empty slot
//!   is scored by `theta . phi(gate, slot)` and the lowest-scoring slot is taken.
//!   Features `phi` are connectivity-aware so position matters without a GNN:
//!   0. distance to the centroid of the gate's already-placed neighbors,
//!   1. distance to the canvas center,
//!   2. local occupancy density (congestion),
//!   3. distance to the canvas boundary.
//! - **Order policy** ([`OrderWeights`]): the order gates are placed in, scored
//!   by `psi . [degree, fanin, fanout]`, highest first. The AF4 default is pure
//!   degree-descending (`psi = [1, 0, 0]`); AF5 learns it.
//!
//! All slot features are normalized by canvas span so one policy transfers
//! across circuit sizes — which is what makes generalization possible.
//!
//! ## AF4 vs AF5
//!
//! - **AF4** ([`train`]) fits the slot policy to minimize **HPWL** — fast, but it
//!   only minimizes wirelength, so it can over-cluster and hurt routability; its
//!   one-shot output is best used as an SA warm-start.
//! - **AF5** ([`train_route_aware`]) fits the full [`Policy`] (slot + order)
//!   against the **routed cost** (which charges unroutable layouts), so the
//!   one-shot placement is itself routable and physically correct — no SA repair.
//!
//! Simulated annealing (AF2) remains the gold-quality reference; neither path
//! claims to beat it on final quality.

use super::circuit::Circuit;
use super::env::PlacementEnv;

/// Number of slot features in the constructive policy.
pub const NUM_FEATURES: usize = 4;
/// Number of gate-ordering features.
pub const NUM_ORDER_FEATURES: usize = 3;

/// Linear slot-scoring weights (lower score = better slot).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyWeights {
    pub theta: [f64; NUM_FEATURES],
}

impl PolicyWeights {
    /// Sensible start: neighbor-centroid distance dominates, light center pull.
    pub fn hand_tuned() -> Self {
        Self {
            theta: [1.0, 0.15, 0.0, 0.0],
        }
    }
}

impl std::fmt::Display for PolicyWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[neigh={:.2} center={:.2} density={:.2} boundary={:.2}]",
            self.theta[0], self.theta[1], self.theta[2], self.theta[3]
        )
    }
}

/// Linear gate-ordering weights over `[degree, fanin, fanout]` (higher = placed
/// earlier).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderWeights {
    pub psi: [f64; NUM_ORDER_FEATURES],
}

impl OrderWeights {
    /// The AF4 default: pure degree-descending order.
    pub fn degree_descending() -> Self {
        Self { psi: [1.0, 0.0, 0.0] }
    }
}

impl std::fmt::Display for OrderWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[deg={:.2} fanin={:.2} fanout={:.2}]",
            self.psi[0], self.psi[1], self.psi[2]
        )
    }
}

/// The full learned policy: slot scoring + gate ordering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    pub slot: PolicyWeights,
    pub order: OrderWeights,
}

impl Policy {
    pub fn hand_tuned() -> Self {
        Self {
            slot: PolicyWeights::hand_tuned(),
            order: OrderWeights::degree_descending(),
        }
    }
}

impl std::fmt::Display for Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "slot {} order {}", self.slot, self.order)
    }
}

/// Deterministic SplitMix64 PRNG (local; keeps AF4/AF5 dependency-free).
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
    /// Uniform f64 in `[-1, 1)`.
    fn signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Undirected gate adjacency from the env's net hyperedges.
fn adjacency(env: &PlacementEnv) -> Vec<Vec<usize>> {
    let n = env.num_gates();
    let mut sets: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
    for edge in env.hyperedges() {
        for (i, &a) in edge.iter().enumerate() {
            for &b in &edge[i + 1..] {
                sets[a].insert(b);
                sets[b].insert(a);
            }
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Per-gate directed fanin / fanout counts (for the ordering policy).
fn fanin_fanout(env: &PlacementEnv) -> (Vec<usize>, Vec<usize>) {
    let netlist = &env.circuit().netlist;
    let n = env.num_gates();
    let num_pi = netlist.primary_inputs.len();
    let fanin: Vec<usize> = netlist.gates.iter().map(|g| g.fanins.len()).collect();
    let mut fanout = vec![0usize; n];
    for gate in &netlist.gates {
        for &f in &gate.fanins {
            let fi = f as usize;
            if fi >= num_pi && fi - num_pi < n {
                fanout[fi - num_pi] += 1;
            }
        }
    }
    (fanin, fanout)
}

/// Core greedy constructive placement under explicit slot + order weights.
fn construct_core(env: &PlacementEnv, slot: &PolicyWeights, order: &OrderWeights) -> Vec<usize> {
    let n = env.num_gates();
    let canvas = env.canvas();
    let num_slots = canvas.num_slots();
    let adj = adjacency(env);
    let (fanin, fanout) = fanin_fanout(env);

    // Placement order: learned linear key over [degree, fanin, fanout], desc.
    let order_key = |g: usize| -> f64 {
        order.psi[0] * adj[g].len() as f64
            + order.psi[1] * fanin[g] as f64
            + order.psi[2] * fanout[g] as f64
    };
    let mut sequence: Vec<usize> = (0..n).collect();
    sequence.sort_by(|&a, &b| {
        order_key(b)
            .partial_cmp(&order_key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let span = ((canvas.cols + canvas.rows) * canvas.pitch).max(1) as f64;
    let center = (
        canvas.min_x as f64
            + canvas.halo as f64
            + (canvas.cols as f64 - 1.0) * canvas.pitch as f64 / 2.0,
        canvas.min_y as f64
            + canvas.halo as f64
            + (canvas.rows as f64 - 1.0) * canvas.pitch as f64 / 2.0,
    );
    let grid_max = canvas.cols.max(canvas.rows).max(1) as f64;

    let mut slot_of_gate = vec![usize::MAX; n];
    let mut gate_of_slot = vec![usize::MAX; num_slots];

    for &g in &sequence {
        // Centroid of already-placed neighbors (None if none placed yet).
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut cnt = 0u32;
        for &nb in &adj[g] {
            if slot_of_gate[nb] != usize::MAX {
                let (x, y) = canvas.slot_coord(slot_of_gate[nb]);
                sx += x as f64;
                sy += y as f64;
                cnt += 1;
            }
        }
        let centroid = (cnt > 0).then(|| (sx / cnt as f64, sy / cnt as f64));

        let mut best_slot = usize::MAX;
        let mut best_score = f64::INFINITY;
        for s in 0..num_slots {
            if gate_of_slot[s] != usize::MAX {
                continue;
            }
            let (xs, ys) = canvas.slot_coord(s);
            let (xf, yf) = (xs as f64, ys as f64);

            let f_neigh = match centroid {
                Some((cx, cy)) => ((xf - cx).abs() + (yf - cy).abs()) / span,
                None => 0.0,
            };
            let f_center = ((xf - center.0).abs() + (yf - center.1).abs()) / span;
            let f_density = local_density(s, &gate_of_slot, &canvas) / 8.0;
            let f_boundary = boundary_distance(s, &canvas) / grid_max;

            let score = slot.theta[0] * f_neigh
                + slot.theta[1] * f_center
                + slot.theta[2] * f_density
                + slot.theta[3] * f_boundary;
            if score < best_score {
                best_score = score;
                best_slot = s;
            }
        }

        slot_of_gate[g] = best_slot;
        gate_of_slot[best_slot] = g;
    }

    slot_of_gate
}

/// Construct a `slot_of_gate` assignment under slot weights with the default
/// degree-descending order (AF4 path). Pure; apply with
/// [`PlacementEnv::restore_assignment`].
pub fn construct_placement(env: &PlacementEnv, weights: &PolicyWeights) -> Vec<usize> {
    construct_core(env, weights, &OrderWeights::degree_descending())
}

/// Construct under a full learned [`Policy`] (slot + order; AF5 path).
pub fn construct_with_policy(env: &PlacementEnv, policy: &Policy) -> Vec<usize> {
    construct_core(env, &policy.slot, &policy.order)
}

/// Number of occupied slots among the 8 grid-neighbors of `slot`.
fn local_density(slot: usize, gate_of_slot: &[usize], canvas: &super::env::Canvas) -> f64 {
    let (cols, rows) = (canvas.cols, canvas.rows);
    let (col, row) = (slot % cols, slot / cols);
    let mut count = 0u32;
    for dr in -1i64..=1 {
        for dc in -1i64..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = row as i64 + dr;
            let nc = col as i64 + dc;
            if nr < 0 || nc < 0 || nr >= rows as i64 || nc >= cols as i64 {
                continue;
            }
            let ns = nr as usize * cols + nc as usize;
            if gate_of_slot[ns] != usize::MAX {
                count += 1;
            }
        }
    }
    count as f64
}

/// Grid-cell distance from `slot` to the nearest canvas boundary.
fn boundary_distance(slot: usize, canvas: &super::env::Canvas) -> f64 {
    let (cols, rows) = (canvas.cols, canvas.rows);
    let (col, row) = (slot % cols, slot / cols);
    col.min(cols - 1 - col).min(row).min(rows - 1 - row) as f64
}

/// Apply slot weights (degree order) to `env`, leaving it holding the layout.
pub fn place(env: &mut PlacementEnv, weights: &PolicyWeights) {
    let assignment = construct_placement(env, weights);
    env.restore_assignment(&assignment);
}

/// Apply a full [`Policy`] to `env`, leaving it holding the layout.
pub fn place_policy(env: &mut PlacementEnv, policy: &Policy) {
    let assignment = construct_with_policy(env, policy);
    env.restore_assignment(&assignment);
}

// ---------------------------------------------------------------------------
// AF4: HPWL-only training (slot weights)
// ---------------------------------------------------------------------------

/// Mean HPWL ratio (constructed / row-major) over pre-built envs.
fn mean_hpwl_ratio_envs(
    envs: &mut [PlacementEnv],
    baseline_hpwl: &[f64],
    weights: &PolicyWeights,
) -> f64 {
    if envs.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for (env, &base) in envs.iter_mut().zip(baseline_hpwl.iter()) {
        let assignment = construct_placement(env, weights);
        env.restore_assignment(&assignment);
        sum += env.hpwl() as f64 / base.max(1.0);
    }
    sum / envs.len() as f64
}

/// Training configuration for the policy fit.
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub iterations: usize,
    pub seed: u64,
    pub init_step: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iterations: 250,
            seed: 0xA11C_E5ED,
            init_step: 0.5,
        }
    }
}

/// Result of the AF4 HPWL fit.
#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub weights: PolicyWeights,
    /// Mean HPWL ratio (learned / row-major) on the training set (< 1.0 better).
    pub train_ratio: f64,
    /// Mean HPWL ratio of the hand-tuned starting policy, for reference.
    pub hand_tuned_ratio: f64,
}

/// AF4: fit slot weights to minimize mean HPWL ratio via deterministic pattern
/// search. Fast (no routing); the one-shot output is best used as a warm-start.
pub fn train(circuits: &[Circuit], config: &TrainConfig) -> TrainOutcome {
    let mut envs: Vec<PlacementEnv> = circuits
        .iter()
        .map(|c| PlacementEnv::new(c).expect("env builds"))
        .collect();
    let baseline: Vec<f64> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            e.hpwl() as f64
        })
        .collect();

    let mut theta = PolicyWeights::hand_tuned();
    let hand_tuned_ratio = mean_hpwl_ratio_envs(&mut envs, &baseline, &theta);
    let mut best = hand_tuned_ratio;
    let mut step = config.init_step;
    let mut rng = SplitMix64::new(config.seed);

    for _ in 0..config.iterations {
        let mut cand = theta;
        for k in 0..NUM_FEATURES {
            cand.theta[k] += rng.signed_unit() * step;
        }
        let ratio = mean_hpwl_ratio_envs(&mut envs, &baseline, &cand);
        if ratio < best {
            best = ratio;
            theta = cand;
        } else {
            step *= 0.99;
        }
    }

    TrainOutcome {
        weights: theta,
        train_ratio: best,
        hand_tuned_ratio,
    }
}

// ---------------------------------------------------------------------------
// AF5: route-aware training (slot + order weights)
// ---------------------------------------------------------------------------

/// Mean routed-cost ratio (routed cost / row-major HPWL) over pre-built envs.
/// Unroutable layouts carry the env's flat penalty, so minimizing this jointly
/// rewards routability and short wiring — the key difference from HPWL-only.
fn mean_routed_ratio(envs: &mut [PlacementEnv], baseline_hpwl: &[f64], policy: &Policy) -> f64 {
    if envs.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for (env, &base) in envs.iter_mut().zip(baseline_hpwl.iter()) {
        let assignment = construct_with_policy(env, policy);
        env.restore_assignment(&assignment);
        sum += env.score().cost / base.max(1.0);
    }
    sum / envs.len() as f64
}

/// Result of the AF5 route-aware fit.
#[derive(Clone, Debug)]
pub struct RouteAwareOutcome {
    pub policy: Policy,
    /// Mean routed-cost ratio on the training set (lower is better).
    pub train_cost_ratio: f64,
    /// Same metric for the hand-tuned starting policy, for reference.
    pub hand_tuned_cost_ratio: f64,
}

/// AF5: fit the full policy (slot + order) against routed cost, so the one-shot
/// placement is routable and physically correct without an SA repair. Routes
/// per evaluation — keep the training set small.
pub fn train_route_aware(circuits: &[Circuit], config: &TrainConfig) -> RouteAwareOutcome {
    let mut envs: Vec<PlacementEnv> = circuits
        .iter()
        .map(|c| PlacementEnv::new(c).expect("env builds"))
        .collect();
    let baseline: Vec<f64> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            e.hpwl() as f64
        })
        .collect();

    let mut policy = Policy::hand_tuned();
    let hand_tuned_cost_ratio = mean_routed_ratio(&mut envs, &baseline, &policy);
    let mut best = hand_tuned_cost_ratio;
    let mut step = config.init_step;
    let mut rng = SplitMix64::new(config.seed);

    for _ in 0..config.iterations {
        let mut cand = policy;
        for k in 0..NUM_FEATURES {
            cand.slot.theta[k] += rng.signed_unit() * step;
        }
        for k in 0..NUM_ORDER_FEATURES {
            cand.order.psi[k] += rng.signed_unit() * step;
        }
        let ratio = mean_routed_ratio(&mut envs, &baseline, &cand);
        if ratio < best {
            best = ratio;
            policy = cand;
        } else {
            step *= 0.99;
        }
    }

    RouteAwareOutcome {
        policy,
        train_cost_ratio: best,
        hand_tuned_cost_ratio,
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Per-circuit evaluation of a fixed policy.
#[derive(Clone, Debug)]
pub struct CircuitEval {
    pub name: String,
    pub gates: usize,
    pub rowmajor_hpwl: usize,
    pub learned_hpwl: usize,
    /// learned / row-major HPWL (< 1.0 = improvement).
    pub ratio: f64,
    pub learned_cost: f64,
    pub routable: bool,
    pub phys_ok: bool,
}

/// Evaluate a full policy on each circuit: construct, measure HPWL vs row-major,
/// route, and physically verify.
pub fn evaluate(circuits: &[Circuit], policy: &Policy) -> Vec<CircuitEval> {
    circuits
        .iter()
        .map(|c| {
            let mut env = PlacementEnv::new(c).expect("env builds");
            env.reset();
            let rowmajor_hpwl = env.hpwl();
            place_policy(&mut env, policy);
            let learned_hpwl = env.hpwl();
            let score = env.score();
            let phys_ok = score.metrics.routable && env.verify_physical();
            CircuitEval {
                name: c.name.clone(),
                gates: env.num_gates(),
                rowmajor_hpwl,
                learned_hpwl,
                ratio: learned_hpwl as f64 / (rowmajor_hpwl.max(1)) as f64,
                learned_cost: score.cost,
                routable: score.metrics.routable,
                phys_ok,
            }
        })
        .collect()
}

/// Mean HPWL ratio of a policy over a set of circuits (< 1.0 = better than
/// row-major). Convenience for reporting generalization.
pub fn mean_hpwl_ratio(circuits: &[Circuit], policy: &Policy) -> f64 {
    let evals = evaluate(circuits, policy);
    if evals.is_empty() {
        return 1.0;
    }
    evals.iter().map(|e| e.ratio).sum::<f64>() / evals.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::alphafabric::corpus::{build_eq_comparator, build_ripple_adder};

    fn slot_policy(w: PolicyWeights) -> Policy {
        Policy {
            slot: w,
            order: OrderWeights::degree_descending(),
        }
    }

    #[test]
    fn construct_beats_row_major() {
        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        env.reset();
        let row_major = env.hpwl();
        place(&mut env, &PolicyWeights::hand_tuned());
        assert!(
            env.hpwl() < row_major,
            "constructive HPWL {} should beat row-major {}",
            env.hpwl(),
            row_major
        );
    }

    #[test]
    fn construct_is_deterministic() {
        let c = Circuit::from_aig("adder4", build_ripple_adder(4));
        let env = PlacementEnv::new(&c).expect("env builds");
        let p = Policy::hand_tuned();
        assert_eq!(
            construct_with_policy(&env, &p),
            construct_with_policy(&env, &p)
        );
    }

    #[test]
    fn constructed_layout_is_physically_correct() {
        let c = Circuit::from_aig("adder4", build_ripple_adder(4));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        place(&mut env, &PolicyWeights::hand_tuned());
        assert!(env.score().metrics.routable, "constructed layout must route");
        assert!(env.verify_physical(), "constructed layout must be correct");
    }

    #[test]
    fn training_is_deterministic() {
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
        ];
        let cfg = TrainConfig {
            iterations: 60,
            ..TrainConfig::default()
        };
        let a = train(&train_set, &cfg);
        let b = train(&train_set, &cfg);
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.train_ratio, b.train_ratio);
    }

    #[test]
    fn warm_start_does_not_worsen_hpwl_and_stays_correct() {
        use crate::synth::alphafabric::{anneal, AnnealConfig};
        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        place(&mut env, &PolicyWeights::hand_tuned());
        let one_shot = env.hpwl();
        let cfg = AnnealConfig {
            iterations: 1500,
            start_from_current: true,
            ..AnnealConfig::default()
        };
        let r = anneal(&mut env, &cfg);
        assert!(r.best_hpwl <= one_shot, "warm-start should not worsen HPWL");
        assert!(r.best.metrics.routable, "warm-start layout should route");
        assert!(env.verify_physical(), "warm-start layout must be correct");
    }

    #[test]
    fn learned_policy_generalizes_to_held_out_circuit() {
        // Train (HPWL) on small circuits, apply to a wider unseen one.
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("adder5", build_ripple_adder(5)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
        ];
        let cfg = TrainConfig {
            iterations: 120,
            ..TrainConfig::default()
        };
        let outcome = train(&train_set, &cfg);

        let held_out = [Circuit::from_aig("eq8", build_eq_comparator(8))];
        let ratio = mean_hpwl_ratio(&held_out, &slot_policy(outcome.weights));
        assert!(ratio < 1.0, "learned policy should generalize (ratio {ratio})");
    }

    #[test]
    fn route_aware_training_is_deterministic() {
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
        ];
        let cfg = TrainConfig {
            iterations: 12,
            ..TrainConfig::default()
        };
        let a = train_route_aware(&train_set, &cfg);
        let b = train_route_aware(&train_set, &cfg);
        assert_eq!(a.policy, b.policy);
        assert_eq!(a.train_cost_ratio, b.train_cost_ratio);
    }

    #[test]
    fn route_aware_one_shot_is_valid_on_held_out() {
        // The AF5 win: a route-aware policy trained on small circuits yields a
        // one-shot layout on a held-out circuit that is routable and correct,
        // with no SA repair.
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
        ];
        let cfg = TrainConfig {
            iterations: 15,
            ..TrainConfig::default()
        };
        let outcome = train_route_aware(&train_set, &cfg);

        let c = Circuit::from_aig("eq8", build_eq_comparator(8));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        place_policy(&mut env, &outcome.policy);
        assert!(env.score().metrics.routable, "one-shot must route");
        assert!(env.verify_physical(), "one-shot must be physically correct");
    }
}
