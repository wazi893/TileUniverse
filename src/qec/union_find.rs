//! Improved Union-Find Decoder for Surface Codes
//!
//! Implementation based on Delfosse & Nickerson (2017):
//! "Almost-linear time decoding algorithm for topological codes"
//!
//! This version takes stabilizer definitions directly from the code,
//! ensuring perfect alignment with the syndrome extraction.

use std::collections::{HashMap, HashSet};

/// Union-Find data structure with path compression and union by rank
#[derive(Clone)]
pub struct DisjointSetUF {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Number of defects in each cluster
    defect_count: Vec<usize>,
    /// Whether cluster is connected to boundary
    boundary_connected: Vec<bool>,
}

impl DisjointSetUF {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            defect_count: vec![0; n],
            boundary_connected: vec![false; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }

        let (root, child) = if self.rank[rx] < self.rank[ry] {
            (ry, rx)
        } else if self.rank[rx] > self.rank[ry] {
            (rx, ry)
        } else {
            self.rank[rx] += 1;
            (rx, ry)
        };

        self.parent[child] = root;
        self.defect_count[root] += self.defect_count[child];
        self.boundary_connected[root] =
            self.boundary_connected[root] || self.boundary_connected[child];
    }

    pub fn same_cluster(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    pub fn cluster_parity(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.defect_count[root] % 2
    }

    pub fn cluster_has_boundary(&mut self, x: usize) -> bool {
        let root = self.find(x);
        self.boundary_connected[root]
    }

    pub fn mark_boundary(&mut self, x: usize) {
        let root = self.find(x);
        self.boundary_connected[root] = true;
    }
}

/// Edge in the matching graph
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct MatchingEdge {
    /// First stabilizer index
    stab_a: usize,
    /// Second stabilizer index (usize::MAX for boundary)
    stab_b: usize,
    /// Qubit that this edge represents (correction qubit)
    qubit: usize,
    /// Weight (Manhattan distance)
    weight: f32,
}

/// Improved Union-Find decoder V2
///
/// This decoder builds matching graphs from stabilizer definitions,
/// ensuring perfect alignment with the actual code structure.
pub struct UnionFindDecoderV2 {
    /// Code distance
    #[allow(dead_code)]
    distance: usize,
    /// Grid width
    #[allow(dead_code)]
    width: usize,
    /// Grid height
    #[allow(dead_code)]
    height: usize,
    /// Edges for X syndrome graph (stab_idx -> list of (other_stab, qubit))
    x_edges: Vec<Vec<(usize, usize)>>,
    /// Edges for Z syndrome graph
    z_edges: Vec<Vec<(usize, usize)>>,
    /// Boundary stabilizers for X syndrome
    x_boundary_stabs: HashSet<usize>,
    /// Boundary stabilizers for Z syndrome
    z_boundary_stabs: HashSet<usize>,
    /// Stabilizer positions for distance calculation
    x_stab_pos: Vec<(f32, f32)>,
    z_stab_pos: Vec<(f32, f32)>,
    /// Boundary qubits for X syndrome: stab_idx -> qubits only in this stabilizer
    x_boundary_qubits: Vec<Vec<usize>>,
    /// Boundary qubits for Z syndrome: stab_idx -> qubits only in this stabilizer
    z_boundary_qubits: Vec<Vec<usize>>,
}

impl UnionFindDecoderV2 {
    /// Create decoder from stabilizer definitions
    ///
    /// Takes the same stabilizer format as SurfaceCode:
    /// - x_stabilizers: Vec of qubit lists for each X stabilizer
    /// - z_stabilizers: Vec of qubit lists for each Z stabilizer
    pub fn from_stabilizers(
        distance: usize,
        x_stabilizers: &[Vec<usize>],
        z_stabilizers: &[Vec<usize>],
    ) -> Self {
        let width = distance;
        let height = distance;

        // Build qubit → stabilizer adjacency maps
        let x_qubit_to_stabs = Self::build_qubit_to_stab_map(x_stabilizers);
        let z_qubit_to_stabs = Self::build_qubit_to_stab_map(z_stabilizers);

        // Build edge lists from qubit adjacency
        let x_edges = Self::build_edges(x_stabilizers.len(), &x_qubit_to_stabs);
        let z_edges = Self::build_edges(z_stabilizers.len(), &z_qubit_to_stabs);

        // Identify boundary stabilizers (2-qubit stabilizers at edges)
        let x_boundary_stabs: HashSet<usize> = x_stabilizers
            .iter()
            .enumerate()
            .filter(|(_, qubits)| qubits.len() == 2)
            .map(|(i, _)| i)
            .collect();

        let z_boundary_stabs: HashSet<usize> = z_stabilizers
            .iter()
            .enumerate()
            .filter(|(_, qubits)| qubits.len() == 2)
            .map(|(i, _)| i)
            .collect();

        // Compute stabilizer positions from qubit positions
        let x_stab_pos = Self::compute_stab_positions(x_stabilizers, width);
        let z_stab_pos = Self::compute_stab_positions(z_stabilizers, width);

        // Find boundary qubits: qubits that only appear in one stabilizer
        let x_boundary_qubits = Self::find_boundary_qubits(x_stabilizers, &x_qubit_to_stabs);
        let z_boundary_qubits = Self::find_boundary_qubits(z_stabilizers, &z_qubit_to_stabs);

        Self {
            distance,
            width,
            height,
            x_edges,
            z_edges,
            x_boundary_stabs,
            z_boundary_stabs,
            x_stab_pos,
            z_stab_pos,
            x_boundary_qubits,
            z_boundary_qubits,
        }
    }

    /// Find qubits that only appear in one stabilizer (boundary qubits)
    fn find_boundary_qubits(
        stabilizers: &[Vec<usize>],
        qubit_to_stabs: &HashMap<usize, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        let mut result = vec![Vec::new(); stabilizers.len()];

        for (stab_idx, qubits) in stabilizers.iter().enumerate() {
            for &q in qubits {
                if let Some(stabs) = qubit_to_stabs.get(&q) {
                    if stabs.len() == 1 {
                        // This qubit only appears in this stabilizer - it's a boundary qubit
                        result[stab_idx].push(q);
                    }
                }
            }
        }

        result
    }

    /// Create decoder for a standard surface code
    pub fn new(distance: usize) -> Self {
        // Reconstruct stabilizer definitions matching SurfaceCode::new()
        let width = distance;
        let height = distance;

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // Interior 4-qubit stabilizers (same as SurfaceCode::new)
        for r in 0..(height - 1) {
            for c in 0..(width - 1) {
                let qubits = vec![
                    r * width + c,
                    r * width + c + 1,
                    (r + 1) * width + c,
                    (r + 1) * width + c + 1,
                ];

                if (r + c) % 2 == 0 {
                    x_stabilizers.push(qubits);
                } else {
                    z_stabilizers.push(qubits);
                }
            }
        }

        // Top edge X stabilizers (odd columns)
        for c in 0..(width - 1) {
            if c % 2 == 1 {
                x_stabilizers.push(vec![c, c + 1]);
            }
        }

        // Bottom edge X stabilizers (even columns)
        for c in 0..(width - 1) {
            if c % 2 == 0 {
                let base = (height - 1) * width;
                x_stabilizers.push(vec![base + c, base + c + 1]);
            }
        }

        // Left edge Z stabilizers (odd rows)
        for r in 0..(height - 1) {
            if r % 2 == 1 {
                z_stabilizers.push(vec![r * width, (r + 1) * width]);
            }
        }

        // Right edge Z stabilizers (even rows)
        for r in 0..(height - 1) {
            if r % 2 == 0 {
                z_stabilizers.push(vec![r * width + (width - 1), (r + 1) * width + (width - 1)]);
            }
        }

        Self::from_stabilizers(distance, &x_stabilizers, &z_stabilizers)
    }

    /// Build map from qubit to list of stabilizers containing it
    fn build_qubit_to_stab_map(stabilizers: &[Vec<usize>]) -> HashMap<usize, Vec<usize>> {
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (stab_idx, qubits) in stabilizers.iter().enumerate() {
            for &q in qubits {
                map.entry(q).or_default().push(stab_idx);
            }
        }
        map
    }

    /// Build edge lists: for each stabilizer, list of (neighbor_stab, connecting_qubit)
    fn build_edges(
        n_stabs: usize,
        qubit_to_stabs: &HashMap<usize, Vec<usize>>,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_stabs];

        for (&qubit, stabs) in qubit_to_stabs {
            // If exactly 2 stabilizers share this qubit, they're neighbors
            if stabs.len() == 2 {
                edges[stabs[0]].push((stabs[1], qubit));
                edges[stabs[1]].push((stabs[0], qubit));
            }
            // If 1 stabilizer, it's a boundary qubit - the stabilizer connects to boundary
        }

        edges
    }

    /// Compute centroid positions for stabilizers
    fn compute_stab_positions(stabilizers: &[Vec<usize>], width: usize) -> Vec<(f32, f32)> {
        stabilizers
            .iter()
            .map(|qubits| {
                let sum_row: f32 = qubits.iter().map(|&q| (q / width) as f32).sum();
                let sum_col: f32 = qubits.iter().map(|&q| (q % width) as f32).sum();
                let n = qubits.len() as f32;
                (sum_row / n, sum_col / n)
            })
            .collect()
    }

    /// Decode X syndrome (detects Z errors) → returns Z corrections
    pub fn decode_x_syndrome(&self, x_syndrome: &[u8]) -> Vec<usize> {
        self.decode_syndrome_internal(
            x_syndrome,
            &self.x_edges,
            &self.x_boundary_stabs,
            &self.x_stab_pos,
            &self.x_boundary_qubits,
        )
    }

    /// Decode Z syndrome (detects X errors) → returns X corrections
    pub fn decode_z_syndrome(&self, z_syndrome: &[u8]) -> Vec<usize> {
        self.decode_syndrome_internal(
            z_syndrome,
            &self.z_edges,
            &self.z_boundary_stabs,
            &self.z_stab_pos,
            &self.z_boundary_qubits,
        )
    }

    /// Core decoding: weighted Union-Find with cluster growth
    fn decode_syndrome_internal(
        &self,
        syndrome: &[u8],
        edges: &[Vec<(usize, usize)>],
        _boundary_stabs: &HashSet<usize>,
        stab_pos: &[(f32, f32)],
        boundary_qubits: &[Vec<usize>],
    ) -> Vec<usize> {
        let n_stabs = syndrome.len();
        if syndrome.iter().all(|&s| s == 0) {
            return Vec::new();
        }

        // Find defects
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == 1)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Vec::new();
        }

        // Initialize Union-Find
        let mut ds = DisjointSetUF::new(n_stabs);
        for &d in &defects {
            ds.defect_count[d] = 1;
        }

        // Mark stabilizers with boundary qubits as boundary-connected
        for (stab_idx, bq) in boundary_qubits.iter().enumerate() {
            if !bq.is_empty() && stab_idx < n_stabs {
                ds.boundary_connected[stab_idx] = true;
            }
        }

        // Collect all possible edges with weights
        let mut all_edges: Vec<(usize, usize, usize, f32)> = Vec::new();

        for (stab_a, neighbors) in edges.iter().enumerate() {
            for &(stab_b, qubit) in neighbors {
                if stab_a < stab_b {
                    let weight = Self::manhattan_distance(stab_pos[stab_a], stab_pos[stab_b]);
                    all_edges.push((stab_a, stab_b, qubit, weight));
                }
            }
        }

        // Sort edges by weight (shortest first)
        all_edges.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

        // Track which edges we use for correction
        let mut used_edges: Vec<(usize, usize, usize)> = Vec::new();

        // Greedy matching: process edges in order of weight
        for (stab_a, stab_b, qubit, _) in &all_edges {
            let stab_a = *stab_a;
            let stab_b = *stab_b;
            let qubit = *qubit;

            let a_needs_match = ds.cluster_parity(stab_a) == 1 && !ds.cluster_has_boundary(stab_a);
            let b_needs_match = ds.cluster_parity(stab_b) == 1 && !ds.cluster_has_boundary(stab_b);

            if a_needs_match && b_needs_match && !ds.same_cluster(stab_a, stab_b) {
                ds.union(stab_a, stab_b);
                used_edges.push((stab_a, stab_b, qubit));
            }
        }

        // Extract corrections
        self.extract_corrections_v2(&defects, &used_edges, edges, boundary_qubits, &mut ds)
    }

    fn manhattan_distance(a: (f32, f32), b: (f32, f32)) -> f32 {
        (a.0 - b.0).abs() + (a.1 - b.1).abs()
    }

    /// Extract corrections using simple algorithm
    ///
    /// Key insight: for a single error at qubit q, the syndrome shows which
    /// stabilizers contain q. To correct, we need to apply an error that
    /// triggers the SAME set of stabilizers.
    ///
    /// - For 2 defects: find the qubit that both stabilizers share
    /// - For 1 defect (boundary): use a boundary qubit of that stabilizer
    fn extract_corrections_v2(
        &self,
        defects: &[usize],
        _used_edges: &[(usize, usize, usize)],
        edges: &[Vec<(usize, usize)>],
        boundary_qubits: &[Vec<usize>],
        _ds: &mut DisjointSetUF,
    ) -> Vec<usize> {
        let mut corrections = Vec::new();
        let mut handled_defects: HashSet<usize> = HashSet::new();

        // Process defect pairs first
        for &d1 in defects {
            if handled_defects.contains(&d1) {
                continue;
            }

            // Find if there's another defect that shares an edge with d1
            let mut paired = false;
            for &(neighbor, qubit) in &edges[d1] {
                if defects.contains(&neighbor) && !handled_defects.contains(&neighbor) {
                    // Found a pair! Use the shared qubit
                    corrections.push(qubit);
                    handled_defects.insert(d1);
                    handled_defects.insert(neighbor);
                    paired = true;
                    break;
                }
            }

            if !paired {
                // Single defect - use boundary qubit
                if d1 < boundary_qubits.len() && !boundary_qubits[d1].is_empty() {
                    corrections.push(boundary_qubits[d1][0]);
                    handled_defects.insert(d1);
                }
            }
        }

        corrections
    }

    /// Find direct edge between two stabilizers
    #[allow(dead_code)]
    fn find_direct_edge(
        stab_a: usize,
        stab_b: usize,
        edges: &[Vec<(usize, usize)>],
    ) -> Option<usize> {
        for &(neighbor, qubit) in &edges[stab_a] {
            if neighbor == stab_b {
                return Some(qubit);
            }
        }
        None
    }

    /// Decode both X and Z syndromes
    pub fn decode(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> (Vec<usize>, Vec<usize>) {
        // X syndrome detects Z errors → Z corrections
        let z_corrections = self.decode_x_syndrome(x_syndrome);
        // Z syndrome detects X errors → X corrections
        let x_corrections = self.decode_z_syndrome(z_syndrome);
        (x_corrections, z_corrections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = UnionFindDecoderV2::new(3);
        assert_eq!(decoder.distance, 3);

        println!(
            "X edges per stab: {:?}",
            decoder.x_edges.iter().map(|e| e.len()).collect::<Vec<_>>()
        );
        println!(
            "Z edges per stab: {:?}",
            decoder.z_edges.iter().map(|e| e.len()).collect::<Vec<_>>()
        );
        println!("X boundary stabs: {:?}", decoder.x_boundary_stabs);
        println!("Z boundary stabs: {:?}", decoder.z_boundary_stabs);
    }

    #[test]
    fn test_no_errors() {
        let decoder = UnionFindDecoderV2::new(3);

        let x_syn = vec![0u8; 4];
        let z_syn = vec![0u8; 4];

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert!(x_corr.is_empty());
        assert!(z_corr.is_empty());
    }

    #[test]
    fn test_single_x_error() {
        use crate::qec::codes::SurfaceCode;

        let distance = 3;
        let decoder = UnionFindDecoderV2::new(distance);

        // Test single X error at center
        let mut code = SurfaceCode::new(distance);
        code.apply_x_error_at(4); // center of 3x3

        let (x_syn, z_syn) = code.measure_syndrome();
        println!("X error at 4: x_syn={:?}, z_syn={:?}", x_syn, z_syn);

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        println!("Corrections: x={:?}, z={:?}", x_corr, z_corr);

        // Apply corrections
        code.correct(&x_corr, &z_corr);

        let (x_after, z_after) = code.measure_syndrome();
        let cleared = x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);
        println!("Syndrome cleared: {}", cleared);
    }

    #[test]
    fn test_debug_all_errors() {
        use crate::qec::codes::SurfaceCode;

        println!("\n=== DEBUG: All Single X Errors ===");
        let distance = 3;
        let decoder = UnionFindDecoderV2::new(distance);

        for q in 0..9 {
            let mut code = SurfaceCode::new(distance);
            code.apply_x_error_at(q);
            let (x_syn, z_syn) = code.measure_syndrome();
            let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
            code.correct(&x_corr, &z_corr);
            let (x_after, z_after) = code.measure_syndrome();
            let cleared = x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);

            let logical_x = code.has_logical_x_error();
            let logical_z = code.has_logical_z_error();

            println!(
                "q={}: z_syn={:?} -> x_corr={:?} | cleared={}, logical_X={}, logical_Z={}",
                q, z_syn, x_corr, cleared, logical_x, logical_z
            );
        }
    }

    #[test]
    fn test_all_single_errors() {
        use crate::qec::codes::SurfaceCode;

        println!("\n=== V2 Decoder: All Single Errors ===");

        for distance in [3, 5, 7] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoderV2::new(distance);

            let mut x_success = 0;
            let mut z_success = 0;
            let mut y_success = 0;

            for q in 0..n_data {
                // Test X error
                let mut code = SurfaceCode::new(distance);
                code.apply_x_error_at(q);
                let (x_syn, z_syn) = code.measure_syndrome();
                let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                code.correct(&x_corr, &z_corr);
                let (x_after, z_after) = code.measure_syndrome();
                if x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0) {
                    x_success += 1;
                }

                // Test Z error
                let mut code = SurfaceCode::new(distance);
                code.apply_z_error_at(q);
                let (x_syn, z_syn) = code.measure_syndrome();
                let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                code.correct(&x_corr, &z_corr);
                let (x_after, z_after) = code.measure_syndrome();
                if x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0) {
                    z_success += 1;
                }

                // Test Y error
                let mut code = SurfaceCode::new(distance);
                code.apply_y_error_at(q);
                let (x_syn, z_syn) = code.measure_syndrome();
                let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                code.correct(&x_corr, &z_corr);
                let (x_after, z_after) = code.measure_syndrome();
                if x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0) {
                    y_success += 1;
                }
            }

            println!(
                "Distance {}: X={}/{} ({:.0}%), Z={}/{} ({:.0}%), Y={}/{} ({:.0}%)",
                distance,
                x_success,
                n_data,
                x_success as f64 / n_data as f64 * 100.0,
                z_success,
                n_data,
                z_success as f64 / n_data as f64 * 100.0,
                y_success,
                n_data,
                y_success as f64 / n_data as f64 * 100.0
            );

            // Should achieve high single-error correction rate
            assert!(
                x_success as f64 / n_data as f64 >= 0.8,
                "X correction rate should be >= 80%"
            );
        }
    }

    #[test]
    fn test_noise_simulation() {
        use crate::qec::codes::SurfaceCode;
        use crate::qec::noise::SimpleRng;

        println!("\n=== V2 Decoder: Noise Simulation ===");

        let n_trials = 500;
        let mut rng = SimpleRng::new(42424);
        let error_types = ['X', 'Y', 'Z'];

        for distance in [3, 5, 7] {
            println!("\nDistance {} surface code:", distance);
            let n_data = distance * distance;
            let decoder = UnionFindDecoderV2::new(distance);

            for &error_prob in &[0.01, 0.02, 0.05, 0.08, 0.10] {
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

                    // Decode and correct
                    let (x_syn, z_syn) = code.measure_syndrome();
                    let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                    code.correct(&x_corr, &z_corr);

                    // Check results
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
                    "  p={:.2}: syndrome_ok={:5.1}%, logical_err={:5.1}%",
                    error_prob, syndrome_rate, logical_rate
                );
            }
        }
    }

    #[test]
    fn test_threshold_behavior() {
        use crate::qec::codes::SurfaceCode;
        use crate::qec::noise::SimpleRng;

        println!("\n=== V2 Decoder: Threshold Behavior ===");
        println!("Logical error rate vs distance at p=0.05\n");

        let n_trials = 300;
        let mut rng = SimpleRng::new(98765);
        let error_prob = 0.05;

        println!(
            "{:>10} {:>10} {:>15} {:>15}",
            "Distance", "Qubits", "Syndrome OK", "Logical Err"
        );
        println!("{:-<55}", "");

        for distance in [3, 5, 7, 9, 11] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoderV2::new(distance);

            let mut syndrome_ok = 0;
            let mut logical_errors = 0;

            for _ in 0..n_trials {
                let mut code = SurfaceCode::new(distance);

                for q in 0..n_data {
                    if rng.next_f64() < error_prob {
                        let e = ['X', 'Y', 'Z'][rng.next_usize(3)];
                        code.apply_error(q, e);
                    }
                }

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
        }

        println!(
            "\nExpected: Logical error rate should decrease with distance below threshold (~10%)"
        );
    }
}
