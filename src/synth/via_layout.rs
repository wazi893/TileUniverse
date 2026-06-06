//! Physical layout of computational-via circuits — vias placed and run as real
//! tiles, not just a netlist.
//!
//! # Why a dedicated layout (not the gate router)
//!
//! A `ThresholdVia` does not behave like a side-pin gate: it counts **all four**
//! in-plane neighbors and gates a **cross-layer** source
//! (`output = source AND (popcount(neighbors) >= T)`). So a wire routed past a
//! via would be miscounted, a producer placed next to a consumer would feed back
//! into its own popcount, and a 4-input via has no free side to emit from. The
//! conventional side-pin placer/router cannot express this.
//!
//! # The correct-by-construction layout
//!
//! The cross-layer `source` term is the key. Stack vias **vertically**: a
//! `ThresholdViaUp` at layer `z` reads its source from layer `z+1`, so if the
//! via above it is another stage, then
//!
//! ```text
//! out[z] = out[z+1] AND threshold_z(neighbors on layer z)
//! ```
//!
//! Chaining the whole stack yields `out[0] = AND over levels of
//! threshold_level(inputs_level)` — an **AND of threshold clauses**. With each
//! clause free to be an OR (T=1), AND (T=k), or majority, a stack computes a
//! monotone CNF / AND-of-thresholds. Each level's inputs live on its own
//! z-layer, so levels never interfere and there is no lateral routing and no
//! feedback. Multiple independent stacks placed side by side give multiple
//! outputs.
//!
//! Every output is taken by a real tile (physical-authority standard), and the
//! result is verified against a software reference.

use crate::simulation::Simulation;
use crate::tiles::tile_meta::TileType;

/// One threshold clause: fires iff at least `threshold` of `inputs` (primary
/// input indices) are active. `inputs.len()` must be in `1..=4`.
#[derive(Clone, Debug)]
pub struct ViaClause {
    pub threshold: u8,
    pub inputs: Vec<usize>,
}

impl ViaClause {
    pub fn new(threshold: u8, inputs: Vec<usize>) -> Self {
        assert!(
            (1..=4).contains(&inputs.len()),
            "a via clause has 1..=4 inputs (a ThresholdVia has 4 neighbors)"
        );
        assert!(
            (1..=inputs.len() as u8).contains(&threshold),
            "threshold must be in 1..=inputs.len()"
        );
        ViaClause { threshold, inputs }
    }

    /// Convenience: an OR clause over the given inputs.
    pub fn or(inputs: Vec<usize>) -> Self {
        ViaClause::new(1, inputs)
    }

    /// Convenience: an AND clause over the given inputs.
    pub fn and(inputs: Vec<usize>) -> Self {
        let t = inputs.len() as u8;
        ViaClause::new(t, inputs)
    }

    /// Convenience: a majority clause over the given inputs.
    pub fn majority(inputs: Vec<usize>) -> Self {
        let t = inputs.len() as u8 / 2 + 1;
        ViaClause::new(t, inputs)
    }

    fn eval(&self, pis: &[bool]) -> bool {
        let active = self.inputs.iter().filter(|&&i| pis[i]).count() as u8;
        active >= self.threshold
    }
}

/// A vertical via stack: output = AND of all clauses (bottom clause is the base
/// of the source chain, clause[0] produces the stack output at layer 0).
#[derive(Clone, Debug)]
pub struct ViaStack {
    pub clauses: Vec<ViaClause>,
}

impl ViaStack {
    pub fn new(clauses: Vec<ViaClause>) -> Self {
        assert!(!clauses.is_empty(), "a via stack needs at least one clause");
        ViaStack { clauses }
    }

    pub fn depth(&self) -> usize {
        self.clauses.len()
    }

    fn eval(&self, pis: &[bool]) -> bool {
        self.clauses.iter().all(|c| c.eval(pis))
    }
}

/// A via circuit: several independent stacks (multiple outputs) over a shared
/// set of `num_pis` primary inputs.
#[derive(Clone, Debug)]
pub struct ViaCircuit {
    pub num_pis: usize,
    pub stacks: Vec<ViaStack>,
}

impl ViaCircuit {
    pub fn new(num_pis: usize, stacks: Vec<ViaStack>) -> Self {
        ViaCircuit { num_pis, stacks }
    }

    /// Software reference evaluation (one bool per stack/output).
    pub fn eval_reference(&self, pis: &[bool]) -> Vec<bool> {
        assert_eq!(pis.len(), self.num_pis);
        self.stacks.iter().map(|s| s.eval(pis)).collect()
    }

    /// Maximum stack depth.
    pub fn max_depth(&self) -> usize {
        self.stacks.iter().map(|s| s.depth()).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Physical realization
// ---------------------------------------------------------------------------

/// In-plane neighbor offsets used as clause input slots, in order.
const NEIGHBORS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Horizontal pitch between stacks (each stack uses a 3-wide footprint + gap).
const STACK_PITCH: usize = 4;

/// A placed, runnable via circuit on a real `Simulation` grid.
pub struct PhysicalViaCircuit {
    sim: Simulation,
    width: usize,
    layer_size: usize,
    num_layers: usize,
    /// Per-stack base x coordinate (vias sit at column `sx`, row `sy`).
    stack_x: Vec<usize>,
    sy: usize,
    /// The circuit being realized (for input wiring + output read).
    circuit: ViaCircuit,
}

#[inline]
fn idx_at(width: usize, x: usize, y: usize, z: usize, layer_size: usize) -> usize {
    z * layer_size + y * width + x
}

/// Build the physical tile grid for a via circuit.
pub fn realize(circuit: &ViaCircuit) -> PhysicalViaCircuit {
    let num_stacks = circuit.stacks.len();
    let max_depth = circuit.max_depth().max(1);

    // Layout: each stack at column sx = 2 + s*PITCH, row sy = 2.
    // Neighbors span sx-1..sx+1 and sy-1..sy+1; PITCH=4 leaves a Const-0 gap.
    let sy = 2usize;
    let stack_x: Vec<usize> = (0..num_stacks).map(|s| 2 + s * STACK_PITCH).collect();
    let width = 2 + num_stacks * STACK_PITCH + 2;
    let height = sy + 3; // rows 0..sy+2, with guard rows top/bottom
    // Layers 0..max_depth-1 hold via stages; layer max_depth holds the tied-high
    // Const source for the topmost stage.
    let num_layers = max_depth + 1;
    let layer_size = width * height;

    let mut sim = Simulation::with_size_layered(width, height, num_layers);
    // Const(0) guard fill everywhere (prevents default-Wire leakage and keeps
    // unused via neighbors at 0).
    for z in 0..num_layers {
        for y in 0..height {
            for x in 0..width {
                sim.set_tile_3d(x, y, z, TileType::Const);
            }
        }
    }

    // Place each stack's via tiles and its tied-high source.
    for (s, stack) in circuit.stacks.iter().enumerate() {
        let sx = stack_x[s];
        let depth = stack.depth();
        for (level, clause) in stack.clauses.iter().enumerate() {
            // clause[level] lives at layer z=level; reads source from z=level+1.
            sim.set_tile_3d(sx, sy, level, TileType::ThresholdViaUp);
            let via_idx = idx_at(width, sx, sy, level, layer_size);
            sim.set_tile_threshold(via_idx, clause.threshold);
        }
        // Tied-high source above the topmost stage (z = depth).
        let src_idx = idx_at(width, sx, sy, depth, layer_size);
        sim.set_logic_value_by_idx(src_idx, 1);
    }

    sim.rebuild_via_connections();

    PhysicalViaCircuit {
        sim,
        width,
        layer_size,
        num_layers,
        stack_x,
        sy,
        circuit: circuit.clone(),
    }
}

impl PhysicalViaCircuit {
    /// Drive primary inputs into the grid and evaluate every stack on real
    /// tiles. Returns one bool per stack (output high?).
    pub fn eval(&mut self, pis: &[bool]) -> Vec<bool> {
        assert_eq!(pis.len(), self.circuit.num_pis);
        let width = self.width;
        let ls = self.layer_size;

        // 1. Drive each via stage's neighbor cells from its clause inputs;
        //    keep unused neighbors at 0; keep tied-high sources at 1.
        for (s, stack) in self.circuit.stacks.iter().enumerate() {
            let sx = self.stack_x[s];
            let depth = stack.depth();
            for (level, clause) in stack.clauses.iter().enumerate() {
                for (slot, &(dx, dy)) in NEIGHBORS.iter().enumerate() {
                    let nx = (sx as isize + dx) as usize;
                    let ny = (self.sy as isize + dy) as usize;
                    let val = clause
                        .inputs
                        .get(slot)
                        .map(|&pi| pis[pi])
                        .unwrap_or(false);
                    let nidx = idx_at(width, nx, ny, level, ls);
                    self.sim.set_logic_value_by_idx(nidx, if val { 1 } else { 0 });
                }
            }
            // Re-assert the tied-high source for the top stage.
            let src_idx = idx_at(width, sx, self.sy, depth, ls);
            self.sim.set_logic_value_by_idx(src_idx, 1);
        }

        // 2. Evaluate vias from the top of each stack downward, because a lower
        //    via reads the via above it as its source.
        for z in (0..self.num_layers).rev() {
            for (s, stack) in self.circuit.stacks.iter().enumerate() {
                if z < stack.depth() {
                    let via_idx = idx_at(width, self.stack_x[s], self.sy, z, ls);
                    self.sim.eval_tile(via_idx);
                }
            }
        }

        // 3. Read each stack's output at layer 0.
        self.circuit
            .stacks
            .iter()
            .enumerate()
            .map(|(s, _)| {
                let out_idx = idx_at(width, self.stack_x[s], self.sy, 0, ls);
                self.sim.get_logic_value_by_idx(out_idx) != 0
            })
            .collect()
    }

    /// Total tile count of the physical layout (an area figure).
    pub fn tile_count(&self) -> usize {
        self.layer_size * self.num_layers
    }
}

/// Build and verify a via circuit against its reference for every input
/// combination (only feasible for small `num_pis`). Returns true iff all match.
pub fn verify_circuit(circuit: &ViaCircuit) -> bool {
    let n = circuit.num_pis;
    assert!(n <= 16, "exhaustive verify only for num_pis <= 16");
    let mut phys = realize(circuit);
    for combo in 0u32..(1u32 << n) {
        let pis: Vec<bool> = (0..n).map(|i| (combo >> i) & 1 != 0).collect();
        let got = phys.eval(&pis);
        let expect = circuit.eval_reference(&pis);
        if got != expect {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_clause_majority_runs_in_tiles() {
        // One stack, one clause = MAJ3. Equivalent to a single physical neuron.
        let circuit = ViaCircuit::new(3, vec![ViaStack::new(vec![ViaClause::majority(
            vec![0, 1, 2],
        )])]);
        assert!(verify_circuit(&circuit));
    }

    #[test]
    fn and_of_two_clauses() {
        // (a OR b) AND (c AND d) over 4 PIs — a depth-2 stack.
        let circuit = ViaCircuit::new(
            4,
            vec![ViaStack::new(vec![
                ViaClause::or(vec![0, 1]),
                ViaClause::and(vec![2, 3]),
            ])],
        );
        // Spot-check the reference itself.
        assert_eq!(circuit.eval_reference(&[true, false, true, true]), vec![true]);
        assert_eq!(
            circuit.eval_reference(&[false, false, true, true]),
            vec![false]
        );
        assert!(verify_circuit(&circuit));
    }

    #[test]
    fn three_level_monotone_cnf() {
        // (a|b) & (c|d) & maj(a,c,d) over 4 PIs — depth-3 stack.
        let circuit = ViaCircuit::new(
            4,
            vec![ViaStack::new(vec![
                ViaClause::or(vec![0, 1]),
                ViaClause::or(vec![2, 3]),
                ViaClause::majority(vec![0, 2, 3]),
            ])],
        );
        assert!(verify_circuit(&circuit));
    }

    #[test]
    fn multi_output_layer() {
        // Three independent outputs over 3 shared PIs: AND, OR, MAJ.
        let circuit = ViaCircuit::new(
            3,
            vec![
                ViaStack::new(vec![ViaClause::and(vec![0, 1, 2])]),
                ViaStack::new(vec![ViaClause::or(vec![0, 1, 2])]),
                ViaStack::new(vec![ViaClause::majority(vec![0, 1, 2])]),
            ],
        );
        // Reference sanity: inputs 110 -> AND=0, OR=1, MAJ=1.
        assert_eq!(
            circuit.eval_reference(&[true, true, false]),
            vec![false, true, true]
        );
        assert!(verify_circuit(&circuit));
    }

    #[test]
    fn stacks_of_different_depths_coexist() {
        // A depth-1 stack and a depth-3 stack share the grid.
        let circuit = ViaCircuit::new(
            4,
            vec![
                ViaStack::new(vec![ViaClause::or(vec![0, 1])]),
                ViaStack::new(vec![
                    ViaClause::and(vec![0, 1]),
                    ViaClause::or(vec![2, 3]),
                    ViaClause::majority(vec![1, 2, 3]),
                ]),
            ],
        );
        assert!(verify_circuit(&circuit));
    }
}
