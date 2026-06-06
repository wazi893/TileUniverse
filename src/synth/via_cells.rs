//! Active-via cell characterization for technology mapping.
//!
//! A `ThresholdVia` is not passive interconnect — it *computes* as the signal
//! crosses layers. Its physical evaluation (see `simulation.rs`) is:
//!
//! ```text
//! active = popcount(in-plane 4-neighbors N,S,E,W that are non-zero)
//! output = if active >= threshold { source } else { 0 }
//! ```
//!
//! With boolean signals (`0` / non-zero) this is a **threshold logic gate**:
//! the four in-plane neighbors are "count" inputs and the cross-layer `source`
//! is an optional enable. Crucially every via has **delay 1**, while AND/OR
//! cost 2 and XOR costs 3 — so folding logic into a via is both a delay and an
//! area win (one tile, no routing hop).
//!
//! This module enumerates exactly which boolean functions a single
//! ThresholdVia realizes, and **verifies each against the real tile evaluator**
//! (not just an abstract formula) so the characterization carries physical
//! authority. The headline cell is `VIA_MAJ3` — majority-of-3 — because that
//! is precisely a full-adder carry, the gate on the critical path of every
//! ripple-carry adder.
//!
//! Truth-table convention matches [`crate::synth::cell_lib`]: bit `i` =
//! `f(i_0, .., i_{n-1})` with `i_j = (i >> j) & 1`. For an "enabled" cell,
//! input 0 is the enable (cross-layer source) and inputs `1..=k` are the
//! neighbors.

use crate::simulation::Simulation;
use crate::synth::cell_lib::{Cell, CellLibrary, ViaInfo};
use crate::tiles::tile_meta::TileType;

/// A boolean function realized by a single active via (delay 1).
#[derive(Clone, Debug)]
pub struct ViaCell {
    /// Human-readable name (e.g. `VIA_MAJ3`, `VIA_AND2`, `VIA_EN_OR2`).
    pub name: String,
    /// The underlying tile type (always a ThresholdVia variant for now).
    pub tile_type: TileType,
    /// Total boolean inputs: `num_neighbors` (+ 1 when `uses_enable`).
    pub num_inputs: u8,
    /// Number of in-plane neighbor inputs `k` (1..=4).
    pub num_neighbors: u8,
    /// Threshold `T` (1..=k) the neighbor popcount is compared against.
    pub threshold: u8,
    /// Whether the cross-layer source is a logic input (enable) vs tied high.
    pub uses_enable: bool,
    /// Truth table over `num_inputs` inputs.
    pub truth_table: u16,
    /// Propagation delay (1 for all vias).
    pub delay: u8,
}

/// Truth table for "at least `t` of `k` neighbors are 1" (source tied high).
///
/// `k` inputs, `2^k` meaningful bits.
pub fn threshold_tt(k: u8, t: u8) -> u16 {
    let mut tt = 0u16;
    for i in 0..(1u16 << k) {
        if (i.count_ones() as u8) >= t {
            tt |= 1 << i;
        }
    }
    tt
}

/// Truth table for `enable AND (at least t of k neighbors)`.
///
/// `k + 1` inputs; input 0 is the enable (cross-layer source).
pub fn threshold_tt_enabled(k: u8, t: u8) -> u16 {
    let n = k + 1;
    let mut tt = 0u16;
    for i in 0..(1u16 << n) {
        let enable = i & 1;
        let neighbors = i >> 1;
        if enable != 0 && (neighbors.count_ones() as u8) >= t {
            tt |= 1 << i;
        }
    }
    tt
}

/// Semantic name for a pure (source-tied-high) threshold function.
fn pure_name(k: u8, t: u8) -> String {
    match (k, t) {
        (1, 1) => "VIA_BUF".to_string(),
        (2, 1) => "VIA_OR2".to_string(),
        (2, 2) => "VIA_AND2".to_string(),
        (3, 1) => "VIA_OR3".to_string(),
        (3, 2) => "VIA_MAJ3".to_string(),
        (3, 3) => "VIA_AND3".to_string(),
        (4, 1) => "VIA_OR4".to_string(),
        (4, 4) => "VIA_AND4".to_string(),
        (k, t) => format!("VIA_GE{}OF{}", t, k),
    }
}

/// The full catalog of boolean functions a single ThresholdVia realizes.
///
/// - Pure threshold cells: `k = 1..=4`, `t = 1..=k` (source tied high).
/// - Enabled threshold cells: `k = 1..=3`, `t = 1..=k` (source = enable input,
///   keeping total inputs `k+1 <= 4` within the K=4 mapping budget).
pub fn via_cell_catalog() -> Vec<ViaCell> {
    let mut cells = Vec::new();

    // Pure threshold functions.
    for k in 1u8..=4 {
        for t in 1u8..=k {
            cells.push(ViaCell {
                name: pure_name(k, t),
                tile_type: TileType::ThresholdViaUp,
                num_inputs: k,
                num_neighbors: k,
                threshold: t,
                uses_enable: false,
                truth_table: threshold_tt(k, t),
                delay: 1,
            });
        }
    }

    // Enabled threshold functions: out = source AND (>=t of k).
    for k in 1u8..=3 {
        for t in 1u8..=k {
            let base = pure_name(k, t);
            let stem = base.strip_prefix("VIA_").unwrap_or(&base);
            cells.push(ViaCell {
                name: format!("VIA_EN_{}", stem),
                tile_type: TileType::ThresholdViaUp,
                num_inputs: k + 1,
                num_neighbors: k,
                threshold: t,
                uses_enable: true,
                truth_table: threshold_tt_enabled(k, t),
                delay: 1,
            });
        }
    }

    cells
}

/// Static name for a pure (source-tied-high) threshold via cell.
///
/// Mirrors [`pure_name`] but returns a `&'static str` so the value can live in
/// a [`Cell`] (whose `name` is `&'static str`).
fn pure_via_static_name(k: u8, t: u8) -> &'static str {
    match (k, t) {
        (1, 1) => "VIA_BUF",
        (2, 1) => "VIA_OR2",
        (2, 2) => "VIA_AND2",
        (3, 1) => "VIA_OR3",
        (3, 2) => "VIA_MAJ3",
        (3, 3) => "VIA_AND3",
        (4, 1) => "VIA_OR4",
        (4, 2) => "VIA_GE2OF4",
        (4, 3) => "VIA_GE3OF4",
        (4, 4) => "VIA_AND4",
        _ => "VIA_TH",
    }
}

/// Build a cell library = tile-native base cells **plus** the pure active-via
/// cells (delay 1). This is the opt-in library for via-aware mapping.
///
/// The base library [`CellLibrary::tile_native`] is left untouched on purpose:
/// adding delay-1 vias there would silently outrank delay-2 gates and rewrite
/// every existing mapping golden. Only the pure (symmetric, source-tied-high)
/// threshold cells are included — they are placement-simple (every input is a
/// neighbor; the source is a constant enable) and cover the high-value
/// functions: wide AND/OR and MAJ3 (the full-adder carry).
pub fn cell_library_with_vias() -> CellLibrary {
    let mut cells: Vec<Cell> = CellLibrary::tile_native().cells().to_vec();
    for vc in via_cell_catalog() {
        // Pure cells only, and skip the degenerate 1-input buffer (routing/
        // buffer insertion is handled elsewhere).
        if vc.uses_enable || vc.num_neighbors < 2 {
            continue;
        }
        cells.push(Cell {
            name: pure_via_static_name(vc.num_neighbors, vc.threshold),
            tile_type: vc.tile_type,
            num_inputs: vc.num_inputs,
            truth_table: vc.truth_table,
            delay: vc.delay,
            via: Some(ViaInfo {
                num_neighbors: vc.num_neighbors,
                threshold: vc.threshold,
                uses_enable: false,
            }),
        });
    }
    CellLibrary::from_cells(cells)
}

#[inline]
fn idx_at(w: usize, x: usize, y: usize, z: usize, layer_size: usize) -> usize {
    z * layer_size + y * w + x
}

/// Evaluate a pure threshold via on real tiles: the cross-layer source is tied
/// high (constant enable) and `inputs` drive the first `inputs.len()` in-plane
/// neighbors. Returns whether the via fired (output != 0).
///
/// This is the McCulloch-Pitts neuron primitive realized in a single delay-1
/// tile: `fire = (count of active inputs >= threshold)`. Used by both the via
/// cell verifier and `snn::tile_neuron`.
///
/// Panics if `inputs.len() > 4` (a ThresholdVia has four in-plane neighbors).
pub fn eval_threshold_via_physical(threshold: u8, inputs: &[bool]) -> bool {
    assert!(
        inputs.len() <= 4,
        "a ThresholdVia has at most 4 in-plane neighbor inputs"
    );
    let w = 8usize;
    let h = 8usize;
    let layer_size = w * h;
    let (cx, cy) = (3usize, 3usize);
    let npos = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)];

    let mut sim = Simulation::with_size_layered(w, h, 2);
    sim.set_tile_3d(cx, cy, 0, TileType::ThresholdViaUp);
    let via_idx = idx_at(w, cx, cy, 0, layer_size);
    sim.set_tile_threshold(via_idx, threshold);

    // Cross-layer source tied high.
    sim.set_tile_3d(cx, cy, 1, TileType::Const);
    sim.set_logic_value_by_idx(idx_at(w, cx, cy, 1, layer_size), 1);

    for (j, &(nx, ny)) in npos.iter().enumerate() {
        sim.set_tile_3d(nx, ny, 0, TileType::Const);
        let val = if j < inputs.len() && inputs[j] { 1u64 } else { 0u64 };
        sim.set_logic_value_by_idx(idx_at(w, nx, ny, 0, layer_size), val);
    }

    sim.rebuild_via_connections();
    sim.eval_tile(via_idx);
    sim.get_logic_value_by_idx(via_idx) != 0
}

/// Verify a via cell's truth table against the **real tile evaluator**.
///
/// Builds a 2-layer grid, places the ThresholdVia with the cell's threshold,
/// a Const source above it, and Const-driven neighbors, then sweeps every input
/// combination and confirms the physical via output matches `cell.truth_table`.
///
/// Returns `true` iff every combination matches.
pub fn verify_via_cell(cell: &ViaCell) -> bool {
    let w = 8usize;
    let h = 8usize;
    let layer_size = w * h;
    let (cx, cy) = (3usize, 3usize);
    // Neighbor positions in input order: left, right, up, down.
    let npos = [
        (cx - 1, cy),
        (cx + 1, cy),
        (cx, cy - 1),
        (cx, cy + 1),
    ];
    let k = cell.num_neighbors as usize;

    for combo in 0u32..(1u32 << cell.num_inputs) {
        let mut sim = Simulation::with_size_layered(w, h, 2);

        // The via on L0, reading its source from L1 (ThresholdViaUp).
        sim.set_tile_3d(cx, cy, 0, cell.tile_type);
        let via_idx = idx_at(w, cx, cy, 0, layer_size);
        sim.set_tile_threshold(via_idx, cell.threshold);

        // Cross-layer source on L1.
        sim.set_tile_3d(cx, cy, 1, TileType::Const);
        let src_idx = idx_at(w, cx, cy, 1, layer_size);

        let (enable_bit, neighbor_bits) = if cell.uses_enable {
            (combo & 1, combo >> 1)
        } else {
            (1u32, combo) // source tied high
        };
        sim.set_logic_value_by_idx(src_idx, if enable_bit != 0 { 1 } else { 0 });

        // Drive k neighbors; hold the rest at Const 0 so they aren't counted.
        for (j, &(nx, ny)) in npos.iter().enumerate() {
            sim.set_tile_3d(nx, ny, 0, TileType::Const);
            let ni = idx_at(w, nx, ny, 0, layer_size);
            let val = if j < k && (neighbor_bits >> j) & 1 != 0 {
                1u64
            } else {
                0u64
            };
            sim.set_logic_value_by_idx(ni, val);
        }

        sim.rebuild_via_connections();
        sim.eval_tile(via_idx);

        let got = sim.get_logic_value_by_idx(via_idx) != 0;
        let expected = (cell.truth_table >> combo) & 1 == 1;
        if got != expected {
            return false;
        }
    }
    true
}

/// Extend a `num_inputs`-input truth table to a 16-bit (4-input) table by
/// treating the high inputs as don't-cares (the function ignores them).
pub fn extend_to_tt4(tt: u16, num_inputs: u8) -> u16 {
    if num_inputs >= 4 {
        return tt;
    }
    let mask = (1u16 << num_inputs) - 1;
    let mut out = 0u16;
    for i in 0u16..16 {
        let lo = i & mask;
        if (tt >> lo) & 1 == 1 {
            out |= 1 << i;
        }
    }
    out
}

/// How many distinct NPN4 equivalence classes the via catalog covers.
///
/// Returns `(distinct_classes, per_cell)` where `per_cell` pairs each cell
/// name with its NPN4 `class_id`.
pub fn via_npn4_coverage() -> (usize, Vec<(String, u8)>) {
    use crate::synth::npn::npn4_classify;
    let mut seen = std::collections::BTreeSet::new();
    let mut per_cell = Vec::new();
    for cell in via_cell_catalog() {
        let tt4 = extend_to_tt4(cell.truth_table, cell.num_inputs);
        let class_id = npn4_classify(tt4).class_id;
        seen.insert(class_id);
        per_cell.push((cell.name, class_id));
    }
    (seen.len(), per_cell)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_truth_tables_are_correct() {
        // 2-input
        assert_eq!(threshold_tt(2, 1), 0xE, "OR2");
        assert_eq!(threshold_tt(2, 2), 0x8, "AND2");
        // 3-input
        assert_eq!(threshold_tt(3, 1), 0xFE, "OR3");
        assert_eq!(threshold_tt(3, 2), 0xE8, "MAJ3");
        assert_eq!(threshold_tt(3, 3), 0x80, "AND3");
        // 4-input
        assert_eq!(threshold_tt(4, 1), 0xFFFE, "OR4");
        assert_eq!(threshold_tt(4, 4), 0x8000, "AND4");
    }

    #[test]
    fn maj3_equals_full_adder_carry() {
        // Full-adder carry-out: cout = 1 when >= 2 of {a, b, cin} are 1.
        // Compute the carry truth table directly and compare to VIA_MAJ3.
        let mut carry_tt = 0u16;
        for i in 0u16..8 {
            let a = i & 1;
            let b = (i >> 1) & 1;
            let c = (i >> 2) & 1;
            if a + b + c >= 2 {
                carry_tt |= 1 << i;
            }
        }
        assert_eq!(carry_tt, 0xE8);
        assert_eq!(threshold_tt(3, 2), carry_tt, "MAJ3 == full-adder carry");
    }

    #[test]
    fn enabled_and2_truth_table() {
        // enable AND (>=1 of 1 neighbor) = enable AND n0 = AND2 over (en, n0).
        assert_eq!(threshold_tt_enabled(1, 1), 0x8);
    }

    #[test]
    fn catalog_is_nonempty_and_named() {
        let catalog = via_cell_catalog();
        assert!(catalog.len() >= 10, "catalog too small: {}", catalog.len());
        let names: Vec<&str> = catalog.iter().map(|c| c.name.as_str()).collect();
        for must in ["VIA_AND2", "VIA_OR2", "VIA_MAJ3", "VIA_AND3", "VIA_AND4"] {
            assert!(names.contains(&must), "missing {} in catalog", must);
        }
        // Every via has delay 1.
        assert!(catalog.iter().all(|c| c.delay == 1));
    }

    /// The physical-authority test: every catalog cell's truth table must match
    /// the real ThresholdVia tile evaluation across all input combinations.
    #[test]
    fn catalog_matches_physical_tile_evaluation() {
        for cell in via_cell_catalog() {
            assert!(
                verify_via_cell(&cell),
                "via cell {} (k={} t={} enable={}) tt={:#06X} does not match physical tile eval",
                cell.name,
                cell.num_neighbors,
                cell.threshold,
                cell.uses_enable,
                cell.truth_table
            );
        }
    }

    #[test]
    fn maj3_physically_realized_in_one_tile() {
        // Spotlight the headline: majority-3 (the adder carry) in a single
        // delay-1 via, verified against the real evaluator.
        let maj3 = via_cell_catalog()
            .into_iter()
            .find(|c| c.name == "VIA_MAJ3")
            .expect("VIA_MAJ3 present");
        assert_eq!(maj3.delay, 1);
        assert_eq!(maj3.truth_table, 0xE8);
        assert!(verify_via_cell(&maj3));
    }

    #[test]
    fn npn4_coverage_reported() {
        let (distinct, per_cell) = via_npn4_coverage();
        assert!(distinct >= 1);
        assert_eq!(per_cell.len(), via_cell_catalog().len());
    }

    // -----------------------------------------------------------------------
    // S3: via-aware mapping
    // -----------------------------------------------------------------------

    #[test]
    fn via_library_contains_pure_via_cells() {
        let lib = cell_library_with_vias();
        let names: Vec<&str> = lib.cells().iter().map(|c| c.name).collect();
        assert!(names.contains(&"VIA_MAJ3"));
        assert!(names.contains(&"VIA_AND3"));
        // Base cells still present.
        assert!(names.contains(&"AND2"));
        assert!(names.contains(&"XOR2"));
        // VIA_AND2 (delay 1) must outrank base AND2 (delay 2) for tt=0x8.
        let best = lib.best_cell_for(0x8, 2).expect("AND2-family cell");
        assert_eq!(lib.cell(best).delay, 1, "via AND2 should win on delay");
    }

    /// Via-aware mapping must remain functionally correct (both cut budgets).
    #[test]
    fn via_mapping_is_equivalent_on_benchmarks() {
        use crate::synth::benchmark::{
            build_4bit_adder, build_8bit_adder, build_8bit_eq_comparator, build_priority_encoder8,
        };
        use crate::synth::mapping::{map_to_library, verify_equivalence, MapConfig};

        let lib = cell_library_with_vias();
        for (name, aig) in [
            ("adder4", build_4bit_adder()),
            ("adder8", build_8bit_adder()),
            ("eq8", build_8bit_eq_comparator()),
            ("prienc8", build_priority_encoder8()),
        ] {
            let netlist = map_to_library(&aig, &lib, &MapConfig::via_aware());
            assert!(
                verify_equivalence(&aig, &netlist, &lib),
                "via-mapped {} is not equivalent to its AIG",
                name
            );
        }
    }

    /// The headline: via-aware mapping must reduce the adder's critical-path
    /// delay (the ripple carry chain folds into MAJ3 vias) and must use the
    /// VIA_MAJ3 cell.
    #[test]
    fn via_mapping_reduces_adder_delay() {
        use crate::synth::benchmark::build_8bit_adder;
        use crate::synth::mapping::{map_to_library, MapConfig};
        use crate::synth::timing::{analyze_timing, TimingConstraint};

        let base_lib = CellLibrary::tile_native();
        let via_lib = cell_library_with_vias();
        let aig = build_8bit_adder();

        let base = map_to_library(&aig, &base_lib, &MapConfig::default());
        let via = map_to_library(&aig, &via_lib, &MapConfig::via_aware());

        let base_t = analyze_timing(&base, &base_lib, &TimingConstraint::intrinsic());
        let via_t = analyze_timing(&via, &via_lib, &TimingConstraint::intrinsic());

        assert!(
            via_t.circuit_delay < base_t.circuit_delay,
            "via mapping should reduce adder delay: base={} via={}",
            base_t.circuit_delay,
            via_t.circuit_delay
        );
    }

    /// S3.5: with the via-aware cut budget, a full-adder carry folds into a
    /// single VIA_MAJ3 cell (majority-of-3 in one delay-1 tile).
    #[test]
    fn full_adder_carry_folds_to_maj3() {
        use crate::synth::aig::Aig;
        use crate::synth::mapping::{map_to_library, verify_equivalence, MapConfig};

        // cout = (a & b) | (cin & (a ^ b)) = majority(a, b, cin).
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let cin = aig.add_input("cin");
        let axb = aig.xor(a, b);
        let ab = aig.and(a, b);
        let caxb = aig.and(cin, axb);
        let cout = aig.or(ab, caxb);
        aig.add_output("cout", cout);

        let via_lib = cell_library_with_vias();
        let netlist = map_to_library(&aig, &via_lib, &MapConfig::via_aware());
        assert!(verify_equivalence(&aig, &netlist, &via_lib));
        let used_maj3 = netlist
            .gates
            .iter()
            .any(|g| via_lib.cell(g.cell_idx).name == "VIA_MAJ3");
        assert!(
            used_maj3,
            "full-adder carry should fold to VIA_MAJ3 with via-aware budget (cells: {:?})",
            netlist
                .gates
                .iter()
                .map(|g| via_lib.cell(g.cell_idx).name)
                .collect::<Vec<_>>()
        );
    }

    /// The MAJ3 cell (full-adder carry) is available to the mapper and wins on
    /// delay. (Whether the default cut enumeration *surfaces* the `{a,b,cin}`
    /// cut is a separate budget question — at MAX_CUTS=8 the carry collapses to
    /// VIA_OR3 + VIA_AND2 instead, still halving the carry-stage delay.)
    #[test]
    fn maj3_cell_is_mapper_ready() {
        let lib = cell_library_with_vias();
        // best_cell_for(majority tt, 3 inputs) must return VIA_MAJ3 at delay 1.
        let idx = lib
            .best_cell_for(0xE8, 3)
            .expect("majority-3 cell must exist in via library");
        let cell = lib.cell(idx);
        assert_eq!(cell.name, "VIA_MAJ3");
        assert_eq!(cell.delay, 1);
        assert_eq!(cell.num_inputs, 3);
        assert!(cell.via.is_some());
    }
}
