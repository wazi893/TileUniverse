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
//!
//! ## AF6: timing-aware training
//!
//! [`TrainConfig::timing_weight`] (default `0.0` = the AF4/AF5 objectives,
//! bit-for-bit) adds the placement-dependent timing term
//! ([`PlacementEnv::timing_cost`], criticality-weighted wirelength) to either
//! objective, so a policy can be fit to pull timing-critical nets short — the
//! learned counterpart of the timing-driven annealing track.
//!
//! Honest scope (measured): the reduction holds **in-distribution** (−14% on
//! the train adders at w=2); one-shot timing quality does **not** reliably
//! transfer to held-out wider circuits, because net-criticality structure
//! changes with depth while the slot features are only span-normalized. See
//! `timing_aware_training_reduces_in_distribution_timing` for the numbers.

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
        Self {
            psi: [1.0, 0.0, 0.0],
        }
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

/// Two-hop gate adjacency: neighbors-of-neighbors, excluding self and 1-hop.
/// The AF6b "GNN-style" lookahead feature's support set.
fn adjacency_2hop(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    adj.iter()
        .enumerate()
        .map(|(g, near)| {
            let mut set: std::collections::BTreeSet<usize> = Default::default();
            for &nb in near {
                for &nb2 in &adj[nb] {
                    set.insert(nb2);
                }
            }
            set.remove(&g);
            for &nb in near {
                set.remove(&nb);
            }
            set.into_iter().collect()
        })
        .collect()
}

/// Per-gate timing criticality: max STA criticality over the gate's incident
/// nets (0.0..=1.0). The AF6c criticality-aware feature's per-gate scalar.
fn gate_criticality(env: &PlacementEnv) -> Vec<f64> {
    let mut crit = vec![0.0f64; env.num_gates()];
    for (edge, &c) in env.hyperedges().iter().zip(env.net_criticalities()) {
        for &g in edge {
            if c > crit[g] {
                crit[g] = c;
            }
        }
    }
    crit
}

/// Core greedy constructive placement under explicit slot + order weights.
/// `two_hop_weight` scores slots by distance to the centroid of the gate's
/// *placed two-hop neighborhood* (AF6b); `crit_pull` adds extra neighbor-pull
/// proportional to the gate's timing criticality (AF6c). `0.0` skips each
/// computation entirely, so all pre-AF6 call paths are bit-for-bit unchanged.
fn construct_core(
    env: &PlacementEnv,
    slot: &PolicyWeights,
    order: &OrderWeights,
    two_hop_weight: f64,
    crit_pull: f64,
) -> Vec<usize> {
    let n = env.num_gates();
    let canvas = env.canvas();
    let num_slots = canvas.num_slots();
    let adj = adjacency(env);
    let adj2 = if two_hop_weight != 0.0 {
        adjacency_2hop(&adj)
    } else {
        Vec::new()
    };
    let gate_crit = if crit_pull != 0.0 {
        gate_criticality(env)
    } else {
        Vec::new()
    };
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

        // Centroid of the placed TWO-HOP neighborhood (AF6b lookahead).
        let centroid2 = if two_hop_weight != 0.0 {
            let mut sx2 = 0.0;
            let mut sy2 = 0.0;
            let mut cnt2 = 0u32;
            for &nb in &adj2[g] {
                if slot_of_gate[nb] != usize::MAX {
                    let (x, y) = canvas.slot_coord(slot_of_gate[nb]);
                    sx2 += x as f64;
                    sy2 += y as f64;
                    cnt2 += 1;
                }
            }
            (cnt2 > 0).then(|| (sx2 / cnt2 as f64, sy2 / cnt2 as f64))
        } else {
            None
        };

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

            let mut score = slot.theta[0] * f_neigh
                + slot.theta[1] * f_center
                + slot.theta[2] * f_density
                + slot.theta[3] * f_boundary;
            if let Some((cx, cy)) = centroid2 {
                score += two_hop_weight * (((xf - cx).abs() + (yf - cy).abs()) / span);
            }
            if crit_pull != 0.0 {
                // Critical gates get extra pull toward their placed neighbors —
                // the transferable "place critical gates tighter" lever.
                score += crit_pull * gate_crit[g] * f_neigh;
            }
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
    construct_core(env, weights, &OrderWeights::degree_descending(), 0.0, 0.0)
}

/// Construct under a full learned [`Policy`] (slot + order; AF5 path).
pub fn construct_with_policy(env: &PlacementEnv, policy: &Policy) -> Vec<usize> {
    construct_core(env, &policy.slot, &policy.order, 0.0, 0.0)
}

/// AF6b: a [`Policy`] extended with the two-hop-centroid lookahead weight.
/// `two_hop == 0.0` is bit-for-bit the base policy (the feature is skipped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoHopPolicy {
    pub policy: Policy,
    pub two_hop: f64,
}

impl TwoHopPolicy {
    pub fn hand_tuned() -> Self {
        Self {
            policy: Policy::hand_tuned(),
            two_hop: 0.0,
        }
    }
}

impl std::fmt::Display for TwoHopPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} two_hop={:.2}", self.policy, self.two_hop)
    }
}

/// Construct under an AF6b two-hop-extended policy.
pub fn construct_with_two_hop(env: &PlacementEnv, p: &TwoHopPolicy) -> Vec<usize> {
    construct_core(env, &p.policy.slot, &p.policy.order, p.two_hop, 0.0)
}

/// Apply an AF6b two-hop policy to `env`, leaving it holding the layout.
pub fn place_two_hop(env: &mut PlacementEnv, p: &TwoHopPolicy) {
    let assignment = construct_with_two_hop(env, p);
    env.restore_assignment(&assignment);
}

/// AF6c: slot weights extended with the criticality-pull weight — the
/// *transferable* timing lever. `crit_pull == 0.0` is bit-for-bit the base
/// 4-feature policy (the feature is skipped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CritPolicy {
    pub slot: PolicyWeights,
    pub crit_pull: f64,
}

impl CritPolicy {
    pub fn hand_tuned() -> Self {
        Self {
            slot: PolicyWeights::hand_tuned(),
            crit_pull: 0.0,
        }
    }
}

impl std::fmt::Display for CritPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} crit_pull={:.2}", self.slot, self.crit_pull)
    }
}

/// Construct under an AF6c criticality-aware policy (degree-descending order).
pub fn construct_with_crit(env: &PlacementEnv, p: &CritPolicy) -> Vec<usize> {
    construct_core(
        env,
        &p.slot,
        &OrderWeights::degree_descending(),
        0.0,
        p.crit_pull,
    )
}

/// Apply an AF6c criticality-aware policy to `env`.
pub fn place_crit(env: &mut PlacementEnv, p: &CritPolicy) {
    let assignment = construct_with_crit(env, p);
    env.restore_assignment(&assignment);
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

/// The training proxy for the current layout: HPWL plus the criticality-weighted
/// wirelength term (AF6). Both are O(pins), route-free. `timing_weight == 0.0`
/// reduces *exactly* to HPWL (`x + 0.0 * t == x`), so the AF4/AF5 defaults are
/// numerically unchanged.
fn proxy_cost(env: &PlacementEnv, timing_weight: f64) -> f64 {
    if timing_weight == 0.0 {
        env.hpwl() as f64
    } else {
        env.hpwl() as f64 + timing_weight * env.timing_cost()
    }
}

/// Mean proxy ratio (constructed / row-major) over pre-built envs.
fn mean_hpwl_ratio_envs(
    envs: &mut [PlacementEnv],
    baseline_proxy: &[f64],
    weights: &PolicyWeights,
    timing_weight: f64,
) -> f64 {
    if envs.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for (env, &base) in envs.iter_mut().zip(baseline_proxy.iter()) {
        let assignment = construct_placement(env, weights);
        env.restore_assignment(&assignment);
        sum += proxy_cost(env, timing_weight) / base.max(1.0);
    }
    sum / envs.len() as f64
}

/// Training configuration for the policy fit.
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub iterations: usize,
    pub seed: u64,
    pub init_step: f64,
    /// AF6: weight of the criticality-weighted-wirelength term in the training
    /// objective (`hpwl + timing_weight * timing_cost`, and
    /// `routed_cost + timing_weight * timing_cost` for the route-aware fit).
    /// Default `0.0` — the AF4/AF5 objectives, bit-for-bit.
    pub timing_weight: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iterations: 250,
            seed: 0xA11C_E5ED,
            init_step: 0.5,
            timing_weight: 0.0,
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

/// AF4: fit slot weights to minimize the mean proxy ratio (HPWL, plus the AF6
/// timing term when `config.timing_weight > 0`) via deterministic pattern
/// search. Fast (no routing); the one-shot output is best used as a warm-start.
pub fn train(circuits: &[Circuit], config: &TrainConfig) -> TrainOutcome {
    let w = config.timing_weight;
    let mut envs: Vec<PlacementEnv> = circuits
        .iter()
        .map(|c| PlacementEnv::new(c).expect("env builds"))
        .collect();
    let baseline: Vec<f64> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            proxy_cost(e, w)
        })
        .collect();

    let mut theta = PolicyWeights::hand_tuned();
    let hand_tuned_ratio = mean_hpwl_ratio_envs(&mut envs, &baseline, &theta, w);
    let mut best = hand_tuned_ratio;
    let mut step = config.init_step;
    let mut rng = SplitMix64::new(config.seed);

    for _ in 0..config.iterations {
        let mut cand = theta;
        for k in 0..NUM_FEATURES {
            cand.theta[k] += rng.signed_unit() * step;
        }
        let ratio = mean_hpwl_ratio_envs(&mut envs, &baseline, &cand, w);
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
// AF6b: two-hop-extended training (slot weights + two-hop lookahead)
// ---------------------------------------------------------------------------

/// Result of the AF6b two-hop fit.
#[derive(Clone, Debug)]
pub struct TwoHopOutcome {
    pub policy: TwoHopPolicy,
    /// Mean proxy ratio on the training set (< 1.0 better than row-major).
    pub train_ratio: f64,
    /// Same metric for the hand-tuned 4-feature start, for reference.
    pub hand_tuned_ratio: f64,
}

/// AF6b: fit slot weights PLUS the two-hop-centroid lookahead weight against the
/// training proxy (HPWL, plus the AF6 timing term if configured). Same
/// deterministic pattern search as [`train`], one extra dimension; the base
/// 4-feature paths are untouched (their RNG stream never sees this function).
pub fn train_two_hop(circuits: &[Circuit], config: &TrainConfig) -> TwoHopOutcome {
    let w = config.timing_weight;
    let mut envs: Vec<PlacementEnv> = circuits
        .iter()
        .map(|c| PlacementEnv::new(c).expect("env builds"))
        .collect();
    let baseline: Vec<f64> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            proxy_cost(e, w)
        })
        .collect();

    let ratio_of = |envs: &mut [PlacementEnv], baseline: &[f64], p: &TwoHopPolicy| -> f64 {
        let mut sum = 0.0;
        for (env, &base) in envs.iter_mut().zip(baseline.iter()) {
            let assignment = construct_with_two_hop(env, p);
            env.restore_assignment(&assignment);
            sum += proxy_cost(env, w) / base.max(1.0);
        }
        sum / envs.len().max(1) as f64
    };

    let mut policy = TwoHopPolicy::hand_tuned();
    let hand_tuned_ratio = ratio_of(&mut envs, &baseline, &policy);
    let mut best = hand_tuned_ratio;
    let mut step = config.init_step;
    let mut rng = SplitMix64::new(config.seed);

    for _ in 0..config.iterations {
        let mut cand = policy;
        for k in 0..NUM_FEATURES {
            cand.policy.slot.theta[k] += rng.signed_unit() * step;
        }
        cand.two_hop += rng.signed_unit() * step;
        let ratio = ratio_of(&mut envs, &baseline, &cand);
        if ratio < best {
            best = ratio;
            policy = cand;
        } else {
            step *= 0.99;
        }
    }

    TwoHopOutcome {
        policy,
        train_ratio: best,
        hand_tuned_ratio,
    }
}

// ---------------------------------------------------------------------------
// AF6c: criticality-aware training (slot weights + crit-pull)
// ---------------------------------------------------------------------------

/// Result of the AF6c criticality-aware fit.
#[derive(Clone, Debug)]
pub struct CritOutcome {
    pub policy: CritPolicy,
    /// Mean proxy ratio on the training set (< 1.0 better than row-major).
    pub train_ratio: f64,
    /// Same metric for the hand-tuned start, for reference.
    pub hand_tuned_ratio: f64,
}

/// AF6c: fit slot weights PLUS the criticality-pull weight. Meant to be trained
/// with `config.timing_weight > 0` so the objective rewards short critical nets
/// and the policy has a *criticality-aware feature* to respond with — the
/// missing lever behind AF6a's transfer negative (the 4 generic features cannot
/// distinguish critical gates at inference time).
pub fn train_crit(circuits: &[Circuit], config: &TrainConfig) -> CritOutcome {
    let w = config.timing_weight;
    let mut envs: Vec<PlacementEnv> = circuits
        .iter()
        .map(|c| PlacementEnv::new(c).expect("env builds"))
        .collect();
    let baseline: Vec<f64> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            proxy_cost(e, w)
        })
        .collect();

    let ratio_of = |envs: &mut [PlacementEnv], baseline: &[f64], p: &CritPolicy| -> f64 {
        let mut sum = 0.0;
        for (env, &base) in envs.iter_mut().zip(baseline.iter()) {
            let assignment = construct_with_crit(env, p);
            env.restore_assignment(&assignment);
            sum += proxy_cost(env, w) / base.max(1.0);
        }
        sum / envs.len().max(1) as f64
    };

    let mut policy = CritPolicy::hand_tuned();
    let hand_tuned_ratio = ratio_of(&mut envs, &baseline, &policy);
    let mut best = hand_tuned_ratio;
    let mut step = config.init_step;
    let mut rng = SplitMix64::new(config.seed);

    for _ in 0..config.iterations {
        let mut cand = policy;
        for k in 0..NUM_FEATURES {
            cand.slot.theta[k] += rng.signed_unit() * step;
        }
        cand.crit_pull += rng.signed_unit() * step;
        let ratio = ratio_of(&mut envs, &baseline, &cand);
        if ratio < best {
            best = ratio;
            policy = cand;
        } else {
            step *= 0.99;
        }
    }

    CritOutcome {
        policy,
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
/// AF6: `timing_weight > 0` adds the criticality-weighted-wirelength term
/// (`0.0` is bit-for-bit the AF5 objective).
fn mean_routed_ratio(
    envs: &mut [PlacementEnv],
    baseline_hpwl: &[f64],
    policy: &Policy,
    timing_weight: f64,
) -> f64 {
    if envs.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    for (env, &base) in envs.iter_mut().zip(baseline_hpwl.iter()) {
        let assignment = construct_with_policy(env, policy);
        env.restore_assignment(&assignment);
        let timing_term = if timing_weight == 0.0 {
            0.0
        } else {
            timing_weight * env.timing_cost()
        };
        sum += (env.score().cost + timing_term) / base.max(1.0);
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

    let w = config.timing_weight;
    let mut policy = Policy::hand_tuned();
    let hand_tuned_cost_ratio = mean_routed_ratio(&mut envs, &baseline, &policy, w);
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
        let ratio = mean_routed_ratio(&mut envs, &baseline, &cand, w);
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
// AF6d: multi-seed evaluation (error bars for transfer claims)
// ---------------------------------------------------------------------------

/// Mean/std of a metric across training seeds. Session-3 lesson: pattern-search
/// results are local optima, so any held-out comparison from a single seed is
/// noise until shown otherwise — transfer claims need error bars.
#[derive(Clone, Debug)]
pub struct MultiSeedEval {
    /// The metric per seed, in `seeds` order.
    pub per_seed: Vec<f64>,
    pub mean: f64,
    /// Sample standard deviation (n-1); 0.0 for fewer than 2 seeds.
    pub std: f64,
}

/// Run `run(seed)` for each seed and aggregate the returned metric. The closure
/// owns training + evaluation, so any train/eval combination gets error bars
/// without dedicated API per experiment.
pub fn multi_seed(seeds: &[u64], mut run: impl FnMut(u64) -> f64) -> MultiSeedEval {
    let per_seed: Vec<f64> = seeds.iter().map(|&s| run(s)).collect();
    let n = per_seed.len();
    let mean = if n == 0 {
        0.0
    } else {
        per_seed.iter().sum::<f64>() / n as f64
    };
    let std = if n < 2 {
        0.0
    } else {
        (per_seed
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt()
    };
    MultiSeedEval {
        per_seed,
        mean,
        std,
    }
}

/// Default seed set for multi-seed evals (arbitrary fixed constants — keep
/// stable so results are reproducible across sessions).
pub const EVAL_SEEDS: [u64; 5] = [
    0xA11C_E5ED,
    0xB0B5_1DE5,
    0xC0FF_EE11,
    0xD00D_FEED,
    0x5EED_0005,
];

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
    /// Criticality-weighted wirelength of the row-major layout (AF6 timing metric).
    pub rowmajor_timing: f64,
    /// Criticality-weighted wirelength of the learned one-shot layout.
    pub learned_timing: f64,
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
            let rowmajor_timing = env.timing_cost();
            place_policy(&mut env, policy);
            let learned_hpwl = env.hpwl();
            let learned_timing = env.timing_cost();
            let score = env.score();
            let phys_ok = score.metrics.routable && env.verify_physical();
            CircuitEval {
                name: c.name.clone(),
                gates: env.num_gates(),
                rowmajor_hpwl,
                learned_hpwl,
                ratio: learned_hpwl as f64 / (rowmajor_hpwl.max(1)) as f64,
                learned_cost: score.cost,
                rowmajor_timing,
                learned_timing,
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
        assert!(
            env.score().metrics.routable,
            "constructed layout must route"
        );
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
        use crate::synth::alphafabric::{AnnealConfig, anneal};
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
        assert!(
            ratio < 1.0,
            "learned policy should generalize (ratio {ratio})"
        );
    }

    #[test]
    fn timing_zero_objective_is_exactly_hpwl() {
        // The AF6 guard: with timing_weight 0 the proxy is bit-for-bit HPWL, so
        // AF4/AF5 default training is numerically unchanged.
        let c = Circuit::from_aig("adder5", build_ripple_adder(5));
        let mut env = PlacementEnv::new(&c).expect("env builds");
        env.reset();
        assert_eq!(proxy_cost(&env, 0.0), env.hpwl() as f64);
        place(&mut env, &PolicyWeights::hand_tuned());
        assert_eq!(proxy_cost(&env, 0.0), env.hpwl() as f64);
    }

    #[test]
    fn timing_aware_training_reduces_in_distribution_timing() {
        // AF6 mechanism proof: adding the timing term to the training objective
        // reduces one-shot criticality-weighted wirelength ON THE TRAINING
        // DISTRIBUTION. Measured (deterministic, seed-pinned): w=0 -> 603.6,
        // w=2 -> 518.1 (-14%).
        //
        // HONEST SCOPE — what this does NOT claim: one-shot timing quality does
        // not reliably transfer to held-out wider adders (measured: adder8
        // 723.8 hpwl-trained vs 785-794 timing-trained on this train set).
        // Unlike the HPWL slot features, which are canvas-span-normalized, net
        // criticality structure changes with circuit depth, so a small-circuit
        // timing fit overfits its own critical paths. A near-scale curriculum
        // (adder3/5/7, w=1) improved both held-out adder8 and adder10 vs its
        // own w=0 baseline, but non-monotonically in w — a stable transfer
        // recipe (likely: depth-normalized criticality features) is future work.
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("adder5", build_ripple_adder(5)),
        ];
        let base_cfg = TrainConfig {
            iterations: 80,
            ..TrainConfig::default()
        };
        let timing_of = |weights: &PolicyWeights, c: &Circuit| -> f64 {
            let mut env = PlacementEnv::new(c).expect("env builds");
            place(&mut env, weights);
            env.timing_cost()
        };
        let in_dist = |weights: &PolicyWeights| -> f64 {
            train_set.iter().map(|c| timing_of(weights, c)).sum()
        };

        let hpwl_policy = train(&train_set, &base_cfg);
        let timing_policy = train(
            &train_set,
            &TrainConfig {
                timing_weight: 2.0,
                ..base_cfg
            },
        );
        let hpwl_timing = in_dist(&hpwl_policy.weights);
        let timing_timing = in_dist(&timing_policy.weights);
        assert!(
            timing_timing < hpwl_timing,
            "timing-aware training should shorten critical nets in-distribution: \
             timing-trained {timing_timing:.1} vs hpwl-trained {hpwl_timing:.1}"
        );
    }

    #[test]
    fn two_hop_feature_generalizes_on_held_out_adders() {
        // AF6b: the two-hop-centroid lookahead earns nonzero weight in training
        // and improves ONE-SHOT held-out generalization on the deep-adder family
        // where the 1-hop policy plateaus. Measured (deterministic, seed-pinned):
        // train ratio 0.560 -> 0.515; held-out adder8 0.759 -> 0.659, adder10
        // 0.796 -> 0.643, eq8 tie (0.444). The adder8 layout is verified
        // physically correct, so the win is not bought with over-clustering.
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("adder5", build_ripple_adder(5)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
            Circuit::from_aig("adder7", build_ripple_adder(7)),
            Circuit::from_aig("eq7", build_eq_comparator(7)),
        ];
        let cfg = TrainConfig {
            iterations: 100,
            ..TrainConfig::default()
        };
        let base = train(&train_set, &cfg);
        let two = train_two_hop(&train_set, &cfg);
        assert!(
            two.train_ratio < base.train_ratio,
            "two-hop should improve the training fit: {:.3} vs {:.3}",
            two.train_ratio,
            base.train_ratio
        );

        let ratio_pair = |c: &Circuit| -> (f64, f64) {
            let mut env = PlacementEnv::new(c).expect("env builds");
            env.reset();
            let row = env.hpwl() as f64;
            place(&mut env, &base.weights);
            let b = env.hpwl() as f64 / row;
            place_two_hop(&mut env, &two.policy);
            let t = env.hpwl() as f64 / row;
            (b, t)
        };

        let adder8 = Circuit::from_aig("adder8", build_ripple_adder(8));
        let (b8, t8) = ratio_pair(&adder8);
        assert!(t8 < b8, "held-out adder8: 2hop {t8:.3} vs base {b8:.3}");
        let (b10, t10) = ratio_pair(&Circuit::from_aig("adder10", build_ripple_adder(10)));
        assert!(
            t10 < b10,
            "held-out adder10: 2hop {t10:.3} vs base {b10:.3}"
        );
        let (be, te) = ratio_pair(&Circuit::from_aig("eq8", build_eq_comparator(8)));
        assert!(
            te <= be + 1e-9,
            "held-out eq8 must not regress: {te:.3} vs {be:.3}"
        );

        // The generalization win must not be bought with an illegal layout.
        let mut env = PlacementEnv::new(&adder8).expect("env builds");
        place_two_hop(&mut env, &two.policy);
        assert!(
            env.score().metrics.routable,
            "2hop adder8 one-shot must route"
        );
        assert!(
            env.verify_physical(),
            "2hop adder8 one-shot must be correct"
        );
    }

    #[test]
    fn crit_pull_zero_is_bit_identical_to_base_policy() {
        // AF6c guard: crit_pull == 0.0 must produce exactly the 4-feature
        // placement (the feature is skipped, not merely weighted to zero).
        let c = Circuit::from_aig("adder5", build_ripple_adder(5));
        let env = PlacementEnv::new(&c).expect("env builds");
        let base = construct_placement(&env, &PolicyWeights::hand_tuned());
        let crit = construct_with_crit(&env, &CritPolicy::hand_tuned());
        assert_eq!(base, crit);
    }

    #[test]
    fn crit_training_is_deterministic_and_earns_weight_under_timing_objective() {
        // AF6c: with a timing-weighted objective the search assigns the
        // criticality-pull feature nonzero weight (measured: 0.45 on the small
        // adder set at w=2, deterministic).
        //
        // HONEST SCOPE — the hypothesis this feature was built to test ("a
        // criticality-aware feature fixes AF6a's timing-transfer negative") was
        // NOT confirmed: on the curriculum set the feature earns ~0 weight and
        // matches objective-only training; cross-session comparisons show
        // held-out timing at this scale is dominated by pattern-search seed
        // noise (same set+objective, different search dims: adder10 1310 vs
        // 1114). Conclusion recorded in the AF6 loop log: timing-transfer
        // claims need a multi-seed eval harness (next session) before any
        // further feature work.
        let train_set = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("adder5", build_ripple_adder(5)),
        ];
        let cfg = TrainConfig {
            iterations: 80,
            timing_weight: 2.0,
            ..TrainConfig::default()
        };
        let a = train_crit(&train_set, &cfg);
        let b = train_crit(&train_set, &cfg);
        assert_eq!(a.policy, b.policy);
        assert!(
            a.policy.crit_pull != 0.0,
            "crit feature should earn nonzero weight under a timing objective \
             (got {})",
            a.policy.crit_pull
        );
        assert!(
            a.train_ratio < a.hand_tuned_ratio,
            "training should improve on the hand-tuned start"
        );
    }

    #[test]
    fn two_hop_win_is_seed_robust() {
        // AF6d: the AF6b two-hop held-out win, re-run across all EVAL_SEEDS
        // with a PAIRED per-seed comparison. Measured: 2-hop beats the base
        // 4-feature policy on all 5 seeds (base 0.656±0.030 vs 2hop
        // 0.607±0.048 mean held-out HPWL ratio) — the claim is seed-robust,
        // not a single-seed artifact.
        //
        // Contrast (measured with the same harness, recorded in the AF6 loop
        // log, no assertion possible for a null): the AF6a timing-objective
        // held-out effect is statistically indistinguishable from noise
        // (w=0: 786±96 vs w=2: 829±46 on adder8 timing, 3 of 5 seeds worse) —
        // confirming sessions 1 and 3 with error bars.
        let curriculum = [
            Circuit::from_aig("adder3", build_ripple_adder(3)),
            Circuit::from_aig("adder5", build_ripple_adder(5)),
            Circuit::from_aig("eq5", build_eq_comparator(5)),
            Circuit::from_aig("adder7", build_ripple_adder(7)),
            Circuit::from_aig("eq7", build_eq_comparator(7)),
        ];
        let held_out = [
            Circuit::from_aig("adder8", build_ripple_adder(8)),
            Circuit::from_aig("eq8", build_eq_comparator(8)),
            Circuit::from_aig("adder10", build_ripple_adder(10)),
        ];
        let cfg = TrainConfig {
            iterations: 100,
            ..TrainConfig::default()
        };
        let held_ratio = |place_fn: &dyn Fn(&mut PlacementEnv)| -> f64 {
            held_out
                .iter()
                .map(|c| {
                    let mut env = PlacementEnv::new(c).expect("env builds");
                    env.reset();
                    let row = env.hpwl() as f64;
                    place_fn(&mut env);
                    env.hpwl() as f64 / row
                })
                .sum::<f64>()
                / held_out.len() as f64
        };

        let base = multi_seed(&EVAL_SEEDS, |s| {
            let o = train(&curriculum, &TrainConfig { seed: s, ..cfg });
            held_ratio(&|env| place(env, &o.weights))
        });
        let twoh = multi_seed(&EVAL_SEEDS, |s| {
            let o = train_two_hop(&curriculum, &TrainConfig { seed: s, ..cfg });
            held_ratio(&|env| place_two_hop(env, &o.policy))
        });

        for (i, (b, t)) in base.per_seed.iter().zip(&twoh.per_seed).enumerate() {
            assert!(
                t < b,
                "2-hop should win on every seed (paired): seed[{i}] 2hop {t:.3} \
                 vs base {b:.3}"
            );
        }
        assert!(
            twoh.mean < base.mean,
            "2-hop mean {:.3} should beat base mean {:.3}",
            twoh.mean,
            base.mean
        );
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
