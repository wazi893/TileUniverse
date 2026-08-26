//! QEC Decoders
//!
//! Algorithms for decoding syndromes to identify and correct errors.
//!
//! - [`UnionFindDecoder`]: Delfosse–Nickerson cluster growth + peeling
//! - [`GreedyMatchingDecoder`]: nearest-neighbor matching (player / baseline)
//! - [`LookupDecoder`]: small-code lookup
//! - [`MWPMDecoder`]: deprecated greedy placeholder; use `MWPMDecoderFB`

use super::codes::SurfaceCode;
use super::union_find::{SyndromeLattice, decode_lattice};

/// Syndrome: bit vector indicating which stabilizers have -1 eigenvalue
pub type Syndrome = Vec<u8>;

/// Correction: which Pauli operations to apply
#[derive(Clone, Debug)]
pub struct Correction {
    /// X corrections: apply X to these qubits
    pub x_corrections: Vec<usize>,
    /// Z corrections: apply Z to these qubits
    pub z_corrections: Vec<usize>,
}

impl Correction {
    pub fn none() -> Self {
        Self {
            x_corrections: vec![],
            z_corrections: vec![],
        }
    }

    pub fn x(qubit: usize) -> Self {
        Self {
            x_corrections: vec![qubit],
            z_corrections: vec![],
        }
    }

    pub fn z(qubit: usize) -> Self {
        Self {
            x_corrections: vec![],
            z_corrections: vec![qubit],
        }
    }
}

/// Lookup table decoder for small codes
///
/// Pre-computes syndrome → correction mapping.
/// Fast but memory-intensive for large codes.
pub struct LookupDecoder {
    /// Map from syndrome to correction
    table: std::collections::HashMap<Vec<u8>, Correction>,
}

impl Default for LookupDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupDecoder {
    pub fn new() -> Self {
        Self {
            table: std::collections::HashMap::new(),
        }
    }

    /// Add an entry to the lookup table
    pub fn add_entry(&mut self, syndrome: Vec<u8>, correction: Correction) {
        self.table.insert(syndrome, correction);
    }

    /// Decode syndrome using lookup table
    pub fn decode(&self, syndrome: &[u8]) -> Correction {
        self.table
            .get(syndrome)
            .cloned()
            .unwrap_or_else(Correction::none)
    }

    /// Build lookup table for repetition code
    pub fn for_repetition_code(n: usize) -> Self {
        let mut decoder = Self::new();

        // No error: all zeros
        decoder.add_entry(vec![0; n - 1], Correction::none());

        // Single X error at position i: syndromes at i-1 and i are flipped
        for error_pos in 0..n {
            let mut syndrome = vec![0u8; n - 1];

            if error_pos > 0 {
                syndrome[error_pos - 1] = 1;
            }
            if error_pos < n - 1 {
                syndrome[error_pos] = 1;
            }

            decoder.add_entry(syndrome, Correction::x(error_pos));
        }

        decoder
    }
}

/// Minimum Weight Perfect Matching (MWPM) decoder
///
/// Standard decoder for surface codes. Matches syndrome defects
/// to minimize total error weight.
///
/// **DEPRECATED**: This is a placeholder using greedy matching.
/// For true MWPM decoding, use `MWPMDecoderFB` with the `mwpm` feature enabled.
/// The fusion-blossom based decoder provides optimal matching with better
/// threshold performance.
///
/// # Migration
///
/// ```ignore
/// // Old (greedy):
/// let decoder = MWPMDecoder::new(5);
///
/// // New (true MWPM):
/// use engine::qec::MWPMDecoderFB;
/// let decoder = MWPMDecoderFB::for_surface_code(5, 0.01);
/// ```
#[deprecated(
    since = "0.78.1",
    note = "Use MWPMDecoderFB with the `mwpm` feature for true MWPM decoding"
)]
pub struct MWPMDecoder {
    /// Code distance
    #[allow(dead_code)]
    distance: usize,
}

#[allow(deprecated)]
impl MWPMDecoder {
    pub fn new(distance: usize) -> Self {
        Self { distance }
    }

    /// Decode syndrome (simplified version)
    ///
    /// Full MWPM would use Blossom algorithm or similar.
    /// This simplified version uses greedy matching.
    pub fn decode(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> Correction {
        let mut correction = Correction::none();

        // Find syndrome defects (positions where syndrome = 1)
        let x_defects: Vec<usize> = x_syndrome
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == 1)
            .map(|(i, _)| i)
            .collect();

        let z_defects: Vec<usize> = z_syndrome
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == 1)
            .map(|(i, _)| i)
            .collect();

        // Greedy matching: pair adjacent defects
        // This is a simplified heuristic, not true MWPM
        for defects in x_defects.chunks(2) {
            if defects.len() == 2 {
                // Connect defects with Z chain
                for q in defects[0]..=defects[1] {
                    correction.z_corrections.push(q);
                }
            } else if defects.len() == 1 {
                // Unpaired defect - connect to boundary
                correction.z_corrections.push(defects[0]);
            }
        }

        for defects in z_defects.chunks(2) {
            if defects.len() == 2 {
                // Connect defects with X chain
                for q in defects[0]..=defects[1] {
                    correction.x_corrections.push(q);
                }
            } else if defects.len() == 1 {
                correction.x_corrections.push(defects[0]);
            }
        }

        correction
    }
}

/// Union-Find decoder for surface codes (Delfosse–Nickerson).
///
/// Synchronized cluster growth on the qubit-edge syndrome graph, then peeling
/// on the grown forest. Nearly-linear in the number of checks. This is an
/// approximation to MWPM, not blossom matching — for true MWPM use
/// [`crate::qec::MWPMDecoderFB`].
pub struct UnionFindDecoder {
    /// Code distance
    distance: usize,
    /// Number of X stabilizers
    num_x_stabs: usize,
    /// Number of Z stabilizers
    num_z_stabs: usize,
    x_lattice: SyndromeLattice,
    z_lattice: SyndromeLattice,
}

impl UnionFindDecoder {
    /// Create a decoder whose lattices match [`SurfaceCode::new`].
    pub fn new(distance: usize) -> Self {
        Self::from_surface_code(&SurfaceCode::new(distance))
    }

    /// Build from an existing surface-code lattice (syndrome order = supports order).
    pub fn from_surface_code(code: &SurfaceCode) -> Self {
        Self::from_stabilizers(
            code.distance,
            code.x_stabilizer_supports(),
            code.z_stabilizer_supports(),
        )
    }

    /// Build from explicit stabilizer supports in syndrome-vector order.
    pub fn from_stabilizers(
        distance: usize,
        x_stabilizers: &[Vec<usize>],
        z_stabilizers: &[Vec<usize>],
    ) -> Self {
        let n_data = distance * distance;
        Self {
            distance,
            num_x_stabs: x_stabilizers.len(),
            num_z_stabs: z_stabilizers.len(),
            x_lattice: SyndromeLattice::from_supports(x_stabilizers, n_data),
            z_lattice: SyndromeLattice::from_supports(z_stabilizers, n_data),
        }
    }

    /// Get the code distance this decoder was created for.
    pub fn distance(&self) -> usize {
        self.distance
    }

    /// Decode X syndrome (detects Z errors) → Z corrections.
    fn decode_x_syndrome(&self, x_syndrome: &[u8]) -> Vec<usize> {
        decode_lattice(&self.x_lattice, x_syndrome)
    }

    /// Decode Z syndrome (detects X errors) → X corrections.
    fn decode_z_syndrome(&self, z_syndrome: &[u8]) -> Vec<usize> {
        decode_lattice(&self.z_lattice, z_syndrome)
    }

    /// Decode both X and Z syndromes for a surface code.
    ///
    /// Returns `(x_corrections, z_corrections)`:
    /// - `x_corrections`: apply X (fixes Z syndrome)
    /// - `z_corrections`: apply Z (fixes X syndrome)
    pub fn decode(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> (Vec<usize>, Vec<usize>) {
        let z_corrections = self.decode_x_syndrome(x_syndrome);
        let x_corrections = self.decode_z_syndrome(z_syndrome);
        (x_corrections, z_corrections)
    }

    /// Decode and return a Correction struct
    pub fn decode_to_correction(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> Correction {
        let (x_corr, z_corr) = self.decode(x_syndrome, z_syndrome);
        Correction {
            x_corrections: x_corr,
            z_corrections: z_corr,
        }
    }
}

/// Greedy nearest-neighbor matching over the syndrome graph.
///
/// Repeatedly pairs the closest remaining defects (defect–defect preferred on
/// ties, else the boundary) along a shortest path. Always clears the syndrome
/// by construction; may apply a logical when a shorter bulk edge ties a
/// boundary edge of equal length. Baseline / visualization decoder — not
/// Union-Find and not MWPM.
pub struct GreedyMatchingDecoder {
    n_data: usize,
    x_supports: Vec<Vec<usize>>,
    z_supports: Vec<Vec<usize>>,
}

const NO_NODE: usize = usize::MAX;

/// Adjacency for one stabilizer type: node `supports.len()` is the virtual
/// boundary. Each edge is a data qubit.
fn syndrome_adj(supports: &[Vec<usize>], n_data: usize) -> Vec<Vec<(usize, usize)>> {
    let bnode = supports.len();
    let mut q_stabs: Vec<Vec<usize>> = vec![Vec::new(); n_data];
    for (si, qs) in supports.iter().enumerate() {
        for &q in qs {
            if q < n_data {
                q_stabs[q].push(si);
            }
        }
    }
    let mut adj = vec![Vec::new(); bnode + 1];
    for (q, stabs) in q_stabs.iter().enumerate() {
        match stabs.len() {
            2 => {
                adj[stabs[0]].push((stabs[1], q));
                adj[stabs[1]].push((stabs[0], q));
            }
            1 => {
                adj[stabs[0]].push((bnode, q));
                adj[bnode].push((stabs[0], q));
            }
            _ => {}
        }
    }
    adj
}

fn syndrome_bfs(adj: &[Vec<(usize, usize)>], src: usize) -> (Vec<i32>, Vec<(usize, usize)>) {
    let n = adj.len();
    let mut dist = vec![-1i32; n];
    let mut prev = vec![(NO_NODE, NO_NODE); n];
    let mut queue = std::collections::VecDeque::new();
    dist[src] = 0;
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        for &(v, qubit) in &adj[u] {
            if dist[v] < 0 {
                dist[v] = dist[u] + 1;
                prev[v] = (u, qubit);
                queue.push_back(v);
            }
        }
    }
    (dist, prev)
}

fn syndrome_path_qubits(prev: &[(usize, usize)], src: usize, target: usize) -> Vec<usize> {
    let mut qs = Vec::new();
    let mut cur = target;
    while cur != src {
        let (p, qubit) = prev[cur];
        if p == NO_NODE {
            break;
        }
        qs.push(qubit);
        cur = p;
    }
    qs
}

/// Greedy matching of `defects`. Returns `(correction qubits, pairs)` where a
/// pair is `(stab, partner_or_-1)`. On equal distance, defect–defect wins over
/// the boundary.
pub fn greedy_match_channel(
    supports: &[Vec<usize>],
    n_data: usize,
    defects: &[usize],
) -> (Vec<usize>, Vec<(usize, i64)>) {
    if defects.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let adj = syndrome_adj(supports, n_data);
    let bnode = supports.len();
    let bfs: Vec<(Vec<i32>, Vec<(usize, usize)>)> =
        defects.iter().map(|&d| syndrome_bfs(&adj, d)).collect();

    let mut remaining: Vec<usize> = (0..defects.len()).collect();
    let mut parity: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut pairs: Vec<(usize, i64)> = Vec::new();

    let toggle = |set: &mut std::collections::HashSet<usize>, qs: &[usize]| {
        for &q in qs {
            if !set.remove(&q) {
                set.insert(q);
            }
        }
    };

    while !remaining.is_empty() {
        // (distance, remaining_pos, partner_defect_idx or -1)
        let mut best: Option<(i32, usize, i64)> = None;
        for (ri, &i) in remaining.iter().enumerate() {
            for &j in &remaining {
                if j <= i {
                    continue;
                }
                let dd = bfs[i].0[defects[j]];
                if dd >= 0 && best.is_none_or(|(d, _, _)| dd < d) {
                    best = Some((dd, ri, j as i64));
                }
            }
            let bd = bfs[i].0[bnode];
            if bd >= 0 && best.is_none_or(|(d, _, _)| bd < d) {
                best = Some((bd, ri, -1));
            }
        }
        let Some((_, ri, partner)) = best else {
            break;
        };
        let i = remaining[ri];
        if partner < 0 {
            toggle(
                &mut parity,
                &syndrome_path_qubits(&bfs[i].1, defects[i], bnode),
            );
            pairs.push((defects[i], -1));
            remaining.retain(|&x| x != i);
        } else {
            let j = partner as usize;
            toggle(
                &mut parity,
                &syndrome_path_qubits(&bfs[i].1, defects[i], defects[j]),
            );
            pairs.push((defects[i], defects[j] as i64));
            remaining.retain(|&x| x != i && x != j);
        }
    }

    let mut corr: Vec<usize> = parity.into_iter().collect();
    corr.sort_unstable();
    (corr, pairs)
}

impl GreedyMatchingDecoder {
    pub fn new(distance: usize) -> Self {
        Self::from_surface_code(&SurfaceCode::new(distance))
    }

    pub fn from_surface_code(code: &SurfaceCode) -> Self {
        Self {
            n_data: code.n_data,
            x_supports: code.x_stabilizer_supports().to_vec(),
            z_supports: code.z_stabilizer_supports().to_vec(),
        }
    }

    /// Returns `(x_corrections, z_corrections)`.
    pub fn decode(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> (Vec<usize>, Vec<usize>) {
        let x_defects: Vec<usize> = (0..x_syndrome.len())
            .filter(|&i| x_syndrome[i] == 1)
            .collect();
        let z_defects: Vec<usize> = (0..z_syndrome.len())
            .filter(|&i| z_syndrome[i] == 1)
            .collect();
        let (z_corr, _) = greedy_match_channel(&self.x_supports, self.n_data, &x_defects);
        let (x_corr, _) = greedy_match_channel(&self.z_supports, self.n_data, &z_defects);
        (x_corr, z_corr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_decoder() {
        let decoder = LookupDecoder::for_repetition_code(5);

        // No error syndrome
        let correction = decoder.decode(&[0, 0, 0, 0]);
        assert!(correction.x_corrections.is_empty());

        // Single error at position 2: syndrome [0,1,1,0]
        let correction = decoder.decode(&[0, 1, 1, 0]);
        assert_eq!(correction.x_corrections, vec![2]);
    }

    #[test]
    fn test_union_find_decoder_creation() {
        let decoder = UnionFindDecoder::new(3);
        assert_eq!(decoder.distance, 3);
        assert_eq!(decoder.num_x_stabs, 4); // 2 interior + 2 boundary
        assert_eq!(decoder.num_z_stabs, 4); // 2 interior + 2 boundary
        println!(
            "Distance 3: {} X stabs, {} Z stabs",
            decoder.num_x_stabs, decoder.num_z_stabs
        );

        let decoder5 = UnionFindDecoder::new(5);
        println!(
            "Distance 5: {} X stabs, {} Z stabs",
            decoder5.num_x_stabs, decoder5.num_z_stabs
        );
    }

    #[test]
    fn test_union_find_no_errors() {
        let decoder = UnionFindDecoder::new(3);

        // No errors: all zeros
        let x_syn = vec![0u8; decoder.num_x_stabs];
        let z_syn = vec![0u8; decoder.num_z_stabs];

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert!(x_corr.is_empty(), "No X corrections for trivial syndrome");
        assert!(z_corr.is_empty(), "No Z corrections for trivial syndrome");
    }

    #[test]
    fn test_union_find_single_defect() {
        let decoder = UnionFindDecoder::new(3);

        // Single X syndrome defect (detecting a Z error)
        let mut x_syn = vec![0u8; decoder.num_x_stabs];
        x_syn[0] = 1; // First X stabilizer triggered

        let z_syn = vec![0u8; decoder.num_z_stabs];

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);

        println!("Single X defect: x_corr={:?}, z_corr={:?}", x_corr, z_corr);

        // Should have a Z correction (to fix X syndrome)
        assert!(
            !z_corr.is_empty(),
            "Should have Z correction for X syndrome defect"
        );
    }

    #[test]
    fn test_union_find_paired_defects() {
        let decoder = UnionFindDecoder::new(5);

        // Two adjacent X syndrome defects (should pair and correct)
        let mut x_syn = vec![0u8; decoder.num_x_stabs];
        if decoder.num_x_stabs >= 2 {
            x_syn[0] = 1;
            x_syn[1] = 1;
        }

        let z_syn = vec![0u8; decoder.num_z_stabs];

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);

        println!("Paired X defects: x_corr={:?}, z_corr={:?}", x_corr, z_corr);
    }

    fn shot_ok(code: &SurfaceCode) -> bool {
        let (x_after, z_after) = code.measure_syndrome();
        x_after.iter().all(|&s| s == 0)
            && z_after.iter().all(|&s| s == 0)
            && !code.has_logical_x_error()
            && !code.has_logical_z_error()
    }

    /// Weight-1 errors on a distance-d rotated lattice: detectable, min-weight
    /// (≤1 qubit per channel), syndrome-clear, no logical.
    fn decode_weight1(decoder: &UnionFindDecoder, code: &mut SurfaceCode, _q: usize) -> bool {
        let (x_syn, z_syn) = code.measure_syndrome();
        let already_trivial = x_syn.iter().all(|&s| s == 0) && z_syn.iter().all(|&s| s == 0);
        if already_trivial {
            return false;
        }
        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        if x_corr.len() > 1 || z_corr.len() > 1 {
            return false;
        }
        code.correct(&x_corr, &z_corr);
        shot_ok(code)
    }

    #[test]
    fn test_union_find_with_surface_code() {
        let distance = 3;
        let decoder = UnionFindDecoder::new(distance);

        let mut code = SurfaceCode::new(distance);
        let center_qubit = 4;
        code.apply_x_error_at(center_qubit);

        let (x_syn, z_syn) = code.measure_syndrome();
        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert_eq!(
            x_corr,
            vec![4],
            "center X must peel to qubit 4, not a logical"
        );
        assert!(z_corr.is_empty());
        code.correct(&x_corr, &z_corr);
        assert!(shot_ok(&code), "center X must clear with no logical");
    }

    #[test]
    fn test_union_find_single_error_all_positions() {
        for distance in [3, 5, 7] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoder::new(distance);

            let mut x_success = 0;
            let mut z_success = 0;
            let mut y_success = 0;

            for q in 0..n_data {
                let mut code = SurfaceCode::new(distance);
                code.apply_x_error_at(q);
                if decode_weight1(&decoder, &mut code, q) {
                    x_success += 1;
                }

                let mut code = SurfaceCode::new(distance);
                code.apply_z_error_at(q);
                if decode_weight1(&decoder, &mut code, q) {
                    z_success += 1;
                }

                let mut code = SurfaceCode::new(distance);
                code.apply_y_error_at(q);
                if decode_weight1(&decoder, &mut code, q) {
                    y_success += 1;
                }
            }

            assert_eq!(x_success, n_data, "d={distance}: X single errors");
            assert_eq!(z_success, n_data, "d={distance}: Z single errors");
            assert_eq!(y_success, n_data, "d={distance}: Y single errors");
        }
    }

    #[test]
    fn test_union_find_noise_simulation() {
        use crate::qec::codes::SurfaceCode;
        use crate::qec::noise::SimpleRng;

        println!("\n=== Union-Find Decoder Noise Simulation ===");

        let n_trials = 500;
        let mut rng = SimpleRng::new(42424);
        let error_types = ['X', 'Y', 'Z'];

        for distance in [3, 5, 7] {
            println!("\nDistance {} surface code:", distance);
            let n_data = distance * distance;
            let decoder = UnionFindDecoder::new(distance);

            for &error_prob in &[0.01, 0.02, 0.05, 0.08, 0.10] {
                let mut success_count = 0;
                let mut logical_errors = 0;

                for _ in 0..n_trials {
                    let mut code = SurfaceCode::new(distance);

                    // Apply random errors
                    for q in 0..n_data {
                        if rng.next_f64() < error_prob {
                            let error_type = error_types[rng.next_usize(3)];
                            code.apply_error(q, error_type);
                        }
                    }

                    // Decode and correct
                    let (x_syn, z_syn) = code.measure_syndrome();
                    let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                    code.correct(&x_corr, &z_corr);

                    // Check syndrome
                    let (x_after, z_after) = code.measure_syndrome();
                    if x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0) {
                        success_count += 1;
                    }

                    // Check logical errors
                    if code.has_logical_x_error() || code.has_logical_z_error() {
                        logical_errors += 1;
                    }
                }

                let success_rate = success_count as f64 / n_trials as f64 * 100.0;
                let logical_rate = logical_errors as f64 / n_trials as f64 * 100.0;

                println!(
                    "  p={:.2}: syndrome_ok={:5.1}%, logical_err={:5.1}%",
                    error_prob, success_rate, logical_rate
                );
                assert_eq!(
                    success_count, n_trials,
                    "d={distance} p={error_prob}: peeling must always clear"
                );
            }
        }
    }

    #[test]
    fn test_union_find_threshold_behavior() {
        use crate::qec::codes::SurfaceCode;
        use crate::qec::noise::SimpleRng;

        println!("\n=== Union-Find Threshold Behavior ===");
        println!("Comparing logical error rate across distances\n");

        let n_trials = 300;
        let mut rng = SimpleRng::new(98765);
        let error_types = ['X', 'Y', 'Z'];

        // Test at ~10% error rate (near threshold)
        let error_prob = 0.05;

        println!(
            "{:>10} {:>10} {:>15} {:>15}",
            "Distance", "Qubits", "Syndrome OK", "Logical Err"
        );
        println!("{:-<10} {:-<10} {:-<15} {:-<15}", "", "", "", "");

        let mut rates: Vec<(usize, f64)> = Vec::new();
        for distance in [3, 5, 7, 9] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoder::new(distance);

            let mut syndrome_ok = 0;
            let mut logical_errors = 0;

            for _ in 0..n_trials {
                let mut code = SurfaceCode::new(distance);

                // Apply random errors
                for q in 0..n_data {
                    if rng.next_f64() < error_prob {
                        let error_type = error_types[rng.next_usize(3)];
                        code.apply_error(q, error_type);
                    }
                }

                // Decode
                let (x_syn, z_syn) = code.measure_syndrome();
                let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                code.correct(&x_corr, &z_corr);

                let (x_after, z_after) = code.measure_syndrome();
                if x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0) {
                    syndrome_ok += 1;
                }

                if code.has_logical_x_error() || code.has_logical_z_error() {
                    logical_errors += 1;
                }
            }

            let syndrome_rate = syndrome_ok as f64 / n_trials as f64 * 100.0;
            let logical_rate = logical_errors as f64 / n_trials as f64 * 100.0;

            println!(
                "{:>10} {:>10} {:>14.1}% {:>14.1}%",
                distance, n_data, syndrome_rate, logical_rate
            );
            assert_eq!(
                syndrome_ok, n_trials,
                "d={distance}: peeling must always clear"
            );
            rates.push((distance, logical_rate));
        }

        let r3 = rates.iter().find(|(d, _)| *d == 3).unwrap().1;
        let r9 = rates.iter().find(|(d, _)| *d == 9).unwrap().1;
        assert!(
            r9 <= r3,
            "logical rate should not rise with distance below threshold: d=3 {r3}% vs d=9 {r9}%"
        );
    }

    #[test]
    fn test_greedy_center_prefers_bulk_on_tie() {
        let decoder = GreedyMatchingDecoder::new(3);
        let mut code = SurfaceCode::new(3);
        code.apply_x_error_at(4);
        let (x_syn, z_syn) = code.measure_syndrome();
        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert!(z_corr.is_empty());
        assert_eq!(x_corr, vec![4]);
        code.correct(&x_corr, &z_corr);
        assert!(shot_ok(&code));
    }
}
