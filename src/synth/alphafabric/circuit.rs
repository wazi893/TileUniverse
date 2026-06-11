//! Immutable problem instance for the AlphaFabric placement environment.

use crate::synth::aig::Aig;
use crate::synth::cell_lib::CellLibrary;
use crate::synth::mapping::{map_to_library, verify_equivalence, MapConfig, MappedNetlist};
use crate::synth::materialize_inversions;

/// A combinational circuit ready to be placed.
///
/// Bundles the AIG (the functional source of truth), its technology-mapped
/// netlist (what actually gets placed), and the cell library they were mapped
/// against. Built once; the [`PlacementEnv`](super::PlacementEnv) borrows it.
///
/// The stored `netlist` is **inversion-materialized**: `output_inverted` /
/// `fanin_inverted` flags are rewritten into explicit `Not` cells so the layout
/// can be exported to real tiles (the fabric has no implicit inversion). This
/// keeps placement, routing, scoring, and physical export all consistent.
pub struct Circuit {
    pub name: String,
    pub aig: Aig,
    pub netlist: MappedNetlist,
    pub lib: CellLibrary,
}

impl Circuit {
    /// Build a circuit from an AIG, technology-mapping it with the default
    /// (delay-oriented) configuration against the tile-native cell library.
    pub fn from_aig(name: impl Into<String>, aig: Aig) -> Self {
        let lib = CellLibrary::tile_native();
        let mapped = map_to_library(&aig, &lib, &MapConfig::default());
        // Make inversions explicit so the netlist can be placed and exported to
        // real tiles. Semantics-preserving, so equivalence vs the AIG holds.
        let netlist = materialize_inversions(&mapped, &lib);
        Circuit {
            name: name.into(),
            aig,
            netlist,
            lib,
        }
    }

    /// Build from a zero-argument AIG builder (e.g. `build_4bit_adder`).
    pub fn from_builder(name: impl Into<String>, build: fn() -> Aig) -> Self {
        Self::from_aig(name, build())
    }

    /// Number of placeable gates (mapped cells).
    pub fn num_gates(&self) -> usize {
        self.netlist.num_gates()
    }

    /// Number of primary inputs.
    pub fn num_inputs(&self) -> usize {
        self.aig.num_inputs() as usize
    }

    /// Number of primary outputs.
    pub fn num_outputs(&self) -> usize {
        self.netlist.primary_outputs.len()
    }

    /// True iff the mapped netlist computes the same function as the AIG.
    ///
    /// This is the placement-invariant correctness check (tier 1 of the legality
    /// gate). Exhaustive for <= 20 inputs, deterministically sampled above that.
    pub fn mapping_is_correct(&self) -> bool {
        verify_equivalence(&self.aig, &self.netlist, &self.lib)
    }
}
