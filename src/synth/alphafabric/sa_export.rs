//! Connectivity-aware (SA) placement for HLS / co-design datapaths.
//!
//! The deterministic row placer ([`place_mapped_netlist`](crate::synth::placement::place_mapped_netlist))
//! places gates in pure topological order, ignoring connectivity — so heavily
//! connected gates land far apart, producing long cross-grid nets that the
//! heuristic router cannot resolve as the datapath grows. AlphaFabric's
//! simulated-annealing placer minimizes HPWL (it co-locates connected gates),
//! which both lowers the routing halo a datapath needs and extends the routable
//! width frontier.
//!
//! Measured (madd, AlphaFabric route config): a width-6 datapath that is
//! row-major-*unroutable* at halo 3 routes and verifies on tiles after SA; a
//! width-8 datapath routes+verifies at halo 4. Routability across halo is
//! **non-monotonic** (the router is a heuristic, not complete), so
//! [`sa_place_escalating`] tries increasing halos and gates each on a physical
//! truth-table check — "routable" is not a sound legality proxy on its own.

use crate::synth::export::{SynthExport, export_to_simulation};
use crate::synth::placement::PlaceConfig;
use crate::synth::routing::{RouteConfig, route_placed_netlist};

use super::anneal::{AnnealConfig, anneal};
use super::circuit::Circuit;
use super::env::PlacementEnv;
use super::reward::RewardWeights;

/// An SA-placed, routed, exported datapath plus the metadata of how it was placed.
pub struct SaExport {
    /// The placed + routed tile grid, ready for `evaluate_exported` / injection.
    pub export: SynthExport,
    /// Routing halo the layout succeeded at.
    pub halo: usize,
    /// Fractional HPWL reduction the anneal achieved vs the row-major baseline.
    pub hpwl_improvement: f64,
    /// Whether the layout was physically verified (truth table vs the AIG).
    pub verified: bool,
}

/// The AlphaFabric export route config (multi-layer, no crossings) — the config
/// under which placed layouts export and evaluate correctly on tiles.
fn export_route_config() -> RouteConfig {
    RouteConfig {
        prefer_horizontal_first: true,
        no_crossings: true,
        max_z: 3,
    }
}

/// SA-place `circuit` at a fixed routing `halo`, then route + export it.
///
/// Returns `None` if the SA-placed layout fails to route at this halo, or (when
/// `verify` is set) if the routed layout fails the physical truth-table check.
/// `verify` is the sound legality gate — routability alone does not guarantee the
/// exported circuit computes the right function.
///
/// Takes a `&Circuit` (technology-mapped once) so an escalation can try many halos
/// without re-mapping; the AIG is not cloned (`Aig` is not `Clone`).
pub fn sa_place_to_export(
    circuit: &Circuit,
    halo: usize,
    anneal_config: &AnnealConfig,
    verify: bool,
) -> Option<SaExport> {
    let place_config = PlaceConfig {
        min_x: 3,
        min_y: 3,
        z: 0,
        max_columns: None,
        halo,
        max_width: None,
    };
    let mut env = PlacementEnv::with_config(
        circuit,
        place_config,
        export_route_config(),
        RewardWeights::default(),
    )
    .ok()?;

    let result = anneal(&mut env, anneal_config);
    if !result.best.metrics.routable {
        return None;
    }
    // The env holds the best layout; verify it on real tiles before committing.
    if verify && !env.verify_physical() {
        return None;
    }

    let placed = env.placed();
    let routed = route_placed_netlist(&circuit.netlist, &placed, &export_route_config()).ok()?;
    let export = export_to_simulation(&routed, &circuit.netlist);
    Some(SaExport {
        export,
        halo,
        hpwl_improvement: result.hpwl_improvement(),
        verified: verify,
    })
}

/// Verify-gated escalation: try SA placement at each halo in `halos`, returning
/// the first layout that routes and physically verifies. This is the response to
/// the router's non-monotonic routability — a higher halo is not strictly easier,
/// so the search probes several and trusts only physical verification.
pub fn sa_place_escalating(
    circuit: &Circuit,
    halos: &[usize],
    anneal_config: &AnnealConfig,
) -> Option<SaExport> {
    halos
        .iter()
        .find_map(|&halo| sa_place_to_export(circuit, halo, anneal_config, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::alphafabric::corpus::build_ripple_adder;

    /// A small datapath SA-places, routes, exports, and physically verifies — the
    /// helper's contract on a fast circuit (adder4 = 16 inputs, exhaustive verify).
    #[test]
    fn sa_place_to_export_routes_and_verifies() {
        let circuit = Circuit::from_aig("adder4", build_ripple_adder(4));
        let cfg = AnnealConfig {
            iterations: 4000,
            ..AnnealConfig::default()
        };
        let sa =
            sa_place_to_export(&circuit, 3, &cfg, true).expect("adder4 SA-places and verifies");
        assert!(sa.verified);
        assert!(sa.export.sim.width() > 0 && sa.export.sim.height() > 0);
    }
}
