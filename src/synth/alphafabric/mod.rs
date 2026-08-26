//! AlphaFabric AF1 — a placement-optimization environment over the synth stack.
//!
//! This is Phase 0 of the AlphaFabric roadmap (`User notes/ROADMAP/ALPHAFABRIC_ROADMAP.md`):
//! turn physical design into a clean, scriptable optimization problem so later
//! phases (a simulated-annealing baseline, then a learned placer) have something
//! to optimize against. It adds **no new physics** — it orchestrates the
//! existing `synth` pipeline:
//!
//! - [`Circuit`] — an immutable problem instance (AIG + mapped netlist + library).
//! - [`PlacementEnv`] — a slot-grid canvas with `reset`/`relocate`/`swap` actions
//!   and a route-based [`score`](PlacementEnv::score). Non-overlap is guaranteed
//!   by construction (gates occupy distinct slots).
//! - [`reward`] — per-layout quality metrics ([`PlacementMetrics`]) and the scalar
//!   cost the optimizer minimizes ([`RewardWeights`]).
//! - [`corpus`] — parametric circuit families with a deterministic train/test split.
//! - [`anneal`] — AF2 simulated-annealing placer (the classical baseline a learned
//!   placer must beat); leaves the env holding the best layout it found.
//! - [`learned`] — AF4 learned constructive placer (a linear policy over
//!   connectivity features, fit to a circuit family; the honest win is
//!   generalization to unseen circuits, not beating SA on final quality).
//! - [`route_frontier`] — reproducible SA route-frontier benchmark for the
//!   citation `madd` circuit (`examples/sa_route_probe.rs` is a thin CLI).
//!
//! ## Legality gate (two tiers)
//!
//! Placement never changes a circuit's logic function, so correctness splits into:
//! 1. **Mapping correctness** ([`Circuit::mapping_is_correct`]) — the mapped netlist
//!    equals the AIG. Placement-invariant; checked once per circuit.
//! 2. **Physical correctness** ([`verify_physical`]) — export the placed + routed
//!    layout to real tiles and check its truth table against the AIG. This is the
//!    strongest gate and honors the project's physical-authority standard: a
//!    *re-placed* circuit must still compute correctly on the fabric.
//!
//! A layout's per-step legality is then just: routable (router connected every net).
//! The honest invariant the tests assert is **routable ⟹ correct**.

pub mod anneal;
pub mod circuit;
pub mod corpus;
pub mod env;
pub mod hls_corpus;
pub mod learned;
pub mod reward;
pub mod route_frontier;
pub mod sa_export;

pub use anneal::{AnnealConfig, AnnealResult, anneal};
pub use circuit::Circuit;
pub use corpus::{
    CorpusEntry, Split, build_and_reduce, build_eq_comparator, build_parity, build_ripple_adder,
    corpus, split,
};
pub use env::{Canvas, EnvError, PlacementEnv, verify_physical};
pub use hls_corpus::{HlsInstance, hls_instances};
pub use learned::{
    CircuitEval, CritOutcome, CritPolicy, EVAL_SEEDS, MultiSeedEval, OrderWeights, Policy,
    PolicyWeights, RouteAwareOutcome, TrainConfig, TrainOutcome, TwoHopOutcome, TwoHopPolicy,
    construct_placement, construct_with_crit, construct_with_policy, construct_with_two_hop,
    evaluate, mean_hpwl_ratio, multi_seed, place, place_crit, place_policy, place_two_hop, train,
    train_crit, train_route_aware, train_two_hop,
};
pub use reward::{PlacementMetrics, PlacementScore, RewardWeights};
pub use route_frontier::{
    RouteFrontierCase, RouteFrontierConfig, RouteFrontierRow, RouteOutcome, SaOutcome,
    check_default_claims, default_claim_cases, frontier, madd_circuit, madd_func, place_config,
    route_config, run_route_frontier, run_route_frontier_row,
};
pub use sa_export::{SaExport, sa_place_escalating, sa_place_to_export};
