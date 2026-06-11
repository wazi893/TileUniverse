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

pub use anneal::{anneal, AnnealConfig, AnnealResult};
pub use circuit::Circuit;
pub use learned::{
    construct_placement, construct_with_policy, evaluate, mean_hpwl_ratio, place, place_policy,
    train, train_route_aware, CircuitEval, OrderWeights, Policy, PolicyWeights, RouteAwareOutcome,
    TrainConfig, TrainOutcome,
};
pub use corpus::{
    build_and_reduce, build_eq_comparator, build_parity, build_ripple_adder, corpus, split,
    CorpusEntry, Split,
};
pub use env::{verify_physical, Canvas, EnvError, PlacementEnv};
pub use hls_corpus::{hls_instances, HlsInstance};
pub use reward::{PlacementMetrics, PlacementScore, RewardWeights};
