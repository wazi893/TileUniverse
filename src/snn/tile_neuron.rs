//! Physical threshold neuron — a neuron that computes in tiles.
//!
//! The fire decision of a spiking neuron is a threshold: it spikes when its
//! integrated input crosses a level. The simplest such neuron — the
//! McCulloch-Pitts neuron — fires when *at least `T` of its `k` binary inputs*
//! are active:
//!
//! ```text
//! fire = (count of active inputs >= threshold)
//! ```
//!
//! That is *exactly* the native operation of a `ThresholdVia` tile (see
//! `simulation.rs`): `output = source if popcount(active neighbors) >= T else 0`.
//! So a threshold neuron is not *emulated* on the tile fabric — it **is** a
//! single delay-1 tile. This is the convergence of the two tracks: the
//! active-via synthesizer (S1–S3.5) and the SNN substrate meet at one primitive.
//!
//! This module provides:
//! - [`TileNeuron`]: a McCulloch-Pitts neuron (`k <= 4` inputs, threshold `T`).
//! - A reference fire function and a **physical** fire that runs the real tile
//!   evaluator, verified to agree.
//! - [`TileNeuron::fire_aig`]: the neuron's fire function as an AIG, which the
//!   via-aware synthesizer maps back to a single ThresholdVia cell — closing
//!   the loop (a neuron compiles into the interconnect).
//!
//! Per the physical-authority standard: if it computes, it computes in tiles.
//! Here the neuron's decision is taken by an actual tile, not in software.

use crate::synth::aig::Aig;
use crate::synth::via_cells::eval_threshold_via_physical;

/// A McCulloch-Pitts threshold neuron realizable as one ThresholdVia tile.
#[derive(Clone, Copy, Debug)]
pub struct TileNeuron {
    /// Number of binary synaptic inputs (1..=4 — a via's in-plane neighbors).
    pub num_inputs: u8,
    /// Firing threshold: spikes when `>= threshold` inputs are active (1..=k).
    pub threshold: u8,
}

impl TileNeuron {
    /// Construct a neuron. Panics if `num_inputs` is not in `1..=4` or
    /// `threshold` is not in `1..=num_inputs`.
    pub fn new(num_inputs: u8, threshold: u8) -> Self {
        assert!(
            (1..=4).contains(&num_inputs),
            "TileNeuron supports 1..=4 inputs (a ThresholdVia has 4 neighbors)"
        );
        assert!(
            (1..=num_inputs).contains(&threshold),
            "threshold must be in 1..=num_inputs"
        );
        TileNeuron {
            num_inputs,
            threshold,
        }
    }

    /// A 2-input AND neuron (fires only when both inputs spike).
    pub fn and(num_inputs: u8) -> Self {
        TileNeuron::new(num_inputs, num_inputs)
    }

    /// An OR neuron (fires when any input spikes).
    pub fn or(num_inputs: u8) -> Self {
        TileNeuron::new(num_inputs, 1)
    }

    /// A majority neuron (fires when more than half of inputs spike).
    pub fn majority(num_inputs: u8) -> Self {
        TileNeuron::new(num_inputs, num_inputs / 2 + 1)
    }

    /// Reference McCulloch-Pitts fire decision (pure software).
    pub fn fire_reference(&self, inputs: &[bool]) -> bool {
        assert_eq!(inputs.len(), self.num_inputs as usize);
        let active = inputs.iter().filter(|&&b| b).count() as u8;
        active >= self.threshold
    }

    /// Fire decision computed by the **real tile evaluator** — the neuron's
    /// decision is taken by an actual ThresholdVia tile.
    pub fn fire_physical(&self, inputs: &[bool]) -> bool {
        assert_eq!(inputs.len(), self.num_inputs as usize);
        eval_threshold_via_physical(self.threshold, inputs)
    }

    /// The neuron's fire function as an AIG, for feeding the synthesizer.
    ///
    /// Built as the threshold function "at least `T` of `k`" =
    /// OR over all size-`T` input subsets of (AND of that subset). The
    /// via-aware mapper folds this back into a single ThresholdVia cell.
    pub fn fire_aig(&self) -> Aig {
        let k = self.num_inputs as usize;
        let t = self.threshold as u32;
        let mut aig = Aig::new();
        let inputs: Vec<_> = (0..k).map(|i| aig.add_input(&format!("x{}", i))).collect();

        let mut terms = Vec::new();
        for mask in 0u32..(1u32 << k) {
            if mask.count_ones() != t {
                continue;
            }
            // AND together the inputs selected by this size-T subset.
            let mut acc = None;
            for (i, &lit) in inputs.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    acc = Some(match acc {
                        None => lit,
                        Some(a) => aig.and(a, lit),
                    });
                }
            }
            if let Some(a) = acc {
                terms.push(a);
            }
        }

        // OR all the subset-terms together.
        let mut out = terms[0];
        for &term in &terms[1..] {
            out = aig.or(out, term);
        }
        aig.add_output("fire", out);
        aig
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::cell_library_with_vias;
    use crate::synth::mapping::{map_to_library, verify_equivalence, MapConfig};

    fn all_inputs(k: u8) -> impl Iterator<Item = Vec<bool>> {
        (0u32..(1u32 << k)).map(move |m| (0..k).map(|i| (m >> i) & 1 != 0).collect())
    }

    #[test]
    fn and_or_majority_neurons_match_reference() {
        // AND2
        let and2 = TileNeuron::and(2);
        for inp in all_inputs(2) {
            let expect = inp[0] && inp[1];
            assert_eq!(and2.fire_reference(&inp), expect);
        }
        // OR3
        let or3 = TileNeuron::or(3);
        for inp in all_inputs(3) {
            let expect = inp.iter().any(|&b| b);
            assert_eq!(or3.fire_reference(&inp), expect);
        }
        // MAJ3
        let maj3 = TileNeuron::majority(3);
        for inp in all_inputs(3) {
            let active = inp.iter().filter(|&&b| b).count();
            assert_eq!(maj3.fire_reference(&inp), active >= 2);
        }
    }

    /// The physical-authority test: the tile-evaluated neuron must agree with
    /// the reference for every neuron shape and every input pattern.
    #[test]
    fn physical_neuron_matches_reference_exhaustive() {
        for k in 1u8..=4 {
            for t in 1u8..=k {
                let neuron = TileNeuron::new(k, t);
                for inp in all_inputs(k) {
                    assert_eq!(
                        neuron.fire_physical(&inp),
                        neuron.fire_reference(&inp),
                        "neuron k={} t={} disagrees on {:?}",
                        k,
                        t,
                        inp
                    );
                }
            }
        }
    }

    /// Closing the loop: a majority neuron's fire function, fed to the
    /// via-aware synthesizer, compiles to a single VIA_MAJ3 cell.
    #[test]
    fn majority_neuron_compiles_to_single_via() {
        let neuron = TileNeuron::majority(3);
        let aig = neuron.fire_aig();
        let lib = cell_library_with_vias();
        let netlist = map_to_library(&aig, &lib, &MapConfig::via_aware());

        assert!(
            verify_equivalence(&aig, &netlist, &lib),
            "neuron fire AIG and its via mapping must be equivalent"
        );
        assert_eq!(
            netlist.num_gates(),
            1,
            "majority neuron should compile to ONE tile, got {} (cells: {:?})",
            netlist.num_gates(),
            netlist
                .gates
                .iter()
                .map(|g| lib.cell(g.cell_idx).name)
                .collect::<Vec<_>>()
        );
        assert_eq!(lib.cell(netlist.gates[0].cell_idx).name, "VIA_MAJ3");
    }

    /// The neuron's fire_aig must be functionally correct vs the reference.
    #[test]
    fn fire_aig_matches_reference() {
        use crate::synth::mapping::evaluate_aig;
        for k in 1u8..=4 {
            for t in 1u8..=k {
                let neuron = TileNeuron::new(k, t);
                let aig = neuron.fire_aig();
                for inp in all_inputs(k) {
                    let got = evaluate_aig(&aig, &inp);
                    assert_eq!(
                        got[0],
                        neuron.fire_reference(&inp),
                        "fire_aig k={} t={} wrong on {:?}",
                        k,
                        t,
                        inp
                    );
                }
            }
        }
    }
}
