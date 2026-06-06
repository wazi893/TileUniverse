//! QEC Decoders
//!
//! Algorithms for decoding syndromes to identify and correct errors.

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

/// Union-Find decoder for Surface Codes
///
/// Fast decoder with nearly-linear complexity O(n α(n)) where α is
/// the inverse Ackermann function. Based on the algorithm by
/// Delfosse & Nickerson (2017).
///
/// The decoder works by:
/// 1. Creating nodes for each syndrome defect and virtual boundaries
/// 2. Growing clusters from defects until they merge or hit boundaries
/// 3. Using cluster parity to determine corrections
///
/// For a distance-d surface code:
/// - Syndrome nodes: one per stabilizer
/// - Boundary nodes: virtual nodes representing code boundaries
/// - Edges: connections between adjacent syndrome locations
pub struct UnionFindDecoder {
    /// Code distance
    distance: usize,
    /// Grid width
    width: usize,
    /// Grid height
    height: usize,
    /// Number of X stabilizers
    num_x_stabs: usize,
    /// Number of Z stabilizers
    num_z_stabs: usize,
    /// X stabilizer positions: (row, col, qubits)
    x_stab_info: Vec<StabilizerInfo>,
    /// Z stabilizer positions: (row, col, qubits)
    z_stab_info: Vec<StabilizerInfo>,
}

/// Information about a stabilizer for the decoder
#[derive(Clone, Debug)]
struct StabilizerInfo {
    /// Row position (for distance calculation)
    row: f32,
    /// Column position
    col: f32,
    /// Qubits this stabilizer acts on
    qubits: Vec<usize>,
    /// Is this a boundary stabilizer?
    is_boundary: bool,
}

/// Union-Find data structure with path compression and union by rank
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Size of each cluster (for parity tracking)
    size: Vec<usize>,
    /// Whether cluster is connected to boundary
    boundary_connected: Vec<bool>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            boundary_connected: vec![false; n],
        }
    }

    /// Find with path compression
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union by rank, returns the new root
    fn union(&mut self, x: usize, y: usize) -> usize {
        let rx = self.find(x);
        let ry = self.find(y);

        if rx == ry {
            return rx;
        }

        // Union by rank
        let (root, child) = if self.rank[rx] < self.rank[ry] {
            (ry, rx)
        } else if self.rank[rx] > self.rank[ry] {
            (rx, ry)
        } else {
            self.rank[rx] += 1;
            (rx, ry)
        };

        self.parent[child] = root;
        self.size[root] += self.size[child];
        self.boundary_connected[root] =
            self.boundary_connected[root] || self.boundary_connected[child];

        root
    }

    fn mark_boundary(&mut self, x: usize) {
        let root = self.find(x);
        self.boundary_connected[root] = true;
    }

    fn is_boundary_connected(&mut self, x: usize) -> bool {
        let root = self.find(x);
        self.boundary_connected[root]
    }
}

impl UnionFindDecoder {
    /// Create a Union-Find decoder for a surface code of given distance
    pub fn new(distance: usize) -> Self {
        let width = distance;
        let height = distance;

        // Build stabilizer info matching the surface code structure
        // This must match the stabilizer enumeration in SurfaceCode::new()
        let mut x_stab_info = Vec::new();
        let mut z_stab_info = Vec::new();

        // Interior 4-qubit stabilizers (checkerboard pattern)
        for r in 0..(height - 1) {
            for c in 0..(width - 1) {
                let qubits = vec![
                    r * width + c,
                    r * width + c + 1,
                    (r + 1) * width + c,
                    (r + 1) * width + c + 1,
                ];
                let info = StabilizerInfo {
                    row: r as f32 + 0.5,
                    col: c as f32 + 0.5,
                    qubits,
                    is_boundary: false,
                };

                if (r + c) % 2 == 0 {
                    x_stab_info.push(info);
                } else {
                    z_stab_info.push(info);
                }
            }
        }

        // Top edge X stabilizers (2-qubit, at odd columns)
        for c in 0..(width - 1) {
            if c % 2 == 1 {
                x_stab_info.push(StabilizerInfo {
                    row: -0.5, // Above the grid (boundary)
                    col: c as f32 + 0.5,
                    qubits: vec![c, c + 1],
                    is_boundary: true,
                });
            }
        }

        // Bottom edge X stabilizers (2-qubit, at even columns)
        for c in 0..(width - 1) {
            if c % 2 == 0 {
                let base = (height - 1) * width;
                x_stab_info.push(StabilizerInfo {
                    row: height as f32 - 0.5, // At the bottom (boundary)
                    col: c as f32 + 0.5,
                    qubits: vec![base + c, base + c + 1],
                    is_boundary: true,
                });
            }
        }

        // Left edge Z stabilizers (2-qubit, at odd rows)
        for r in 0..(height - 1) {
            if r % 2 == 1 {
                z_stab_info.push(StabilizerInfo {
                    row: r as f32 + 0.5,
                    col: -0.5, // Left of grid (boundary)
                    qubits: vec![r * width, (r + 1) * width],
                    is_boundary: true,
                });
            }
        }

        // Right edge Z stabilizers (2-qubit, at even rows)
        for r in 0..(height - 1) {
            if r % 2 == 0 {
                z_stab_info.push(StabilizerInfo {
                    row: r as f32 + 0.5,
                    col: width as f32 - 0.5, // Right of grid (boundary)
                    qubits: vec![r * width + (width - 1), (r + 1) * width + (width - 1)],
                    is_boundary: true,
                });
            }
        }

        Self {
            distance,
            width,
            height,
            num_x_stabs: x_stab_info.len(),
            num_z_stabs: z_stab_info.len(),
            x_stab_info,
            z_stab_info,
        }
    }

    /// Get the code distance this decoder was created for.
    pub fn distance(&self) -> usize {
        self.distance
    }

    /// Calculate Manhattan distance between two stabilizers
    fn stab_distance(a: &StabilizerInfo, b: &StabilizerInfo) -> f32 {
        (a.row - b.row).abs() + (a.col - b.col).abs()
    }

    /// Find edges between stabilizers that share a qubit
    #[allow(dead_code)]
    fn find_edges(stab_info: &[StabilizerInfo]) -> Vec<(usize, usize, usize)> {
        let mut edges = Vec::new();

        for i in 0..stab_info.len() {
            for j in (i + 1)..stab_info.len() {
                // Check if stabilizers share a qubit
                for &qi in &stab_info[i].qubits {
                    for &qj in &stab_info[j].qubits {
                        if qi == qj {
                            // They share qubit qi
                            edges.push((i, j, qi));
                            break;
                        }
                    }
                }
            }
        }

        edges
    }

    /// Decode X syndrome (detects Z errors) → returns Z corrections
    fn decode_x_syndrome(&self, x_syndrome: &[u8]) -> Vec<usize> {
        if x_syndrome.iter().all(|&s| s == 0) {
            return Vec::new();
        }

        let n = self.num_x_stabs;
        let mut ds = DisjointSet::new(n);

        // Find defects (triggered stabilizers)
        let defects: Vec<usize> = x_syndrome
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == 1)
            .map(|(i, _)| i)
            .collect();

        // Mark boundary stabilizers
        for (i, info) in self.x_stab_info.iter().enumerate() {
            if info.is_boundary {
                ds.mark_boundary(i);
            }
        }

        // Grow clusters using weighted growth
        // Strategy: connect defects that are close together
        let _max_growth_radius = self.distance as f32;

        for radius in 1..=(self.distance + 1) {
            let r = radius as f32;

            // Try to connect defects within current radius
            for i in 0..defects.len() {
                for j in (i + 1)..defects.len() {
                    let d1 = defects[i];
                    let d2 = defects[j];

                    if ds.find(d1) != ds.find(d2) {
                        let dist =
                            Self::stab_distance(&self.x_stab_info[d1], &self.x_stab_info[d2]);
                        if dist <= r * 2.0 {
                            ds.union(d1, d2);
                        }
                    }
                }

                // Connect to boundary if within radius
                let d = defects[i];
                if !ds.is_boundary_connected(d) {
                    let info = &self.x_stab_info[d];
                    // Distance to top/bottom boundary
                    let to_top = info.row + 0.5;
                    let to_bottom = self.height as f32 - info.row - 0.5;
                    let min_boundary_dist = to_top.min(to_bottom);

                    if min_boundary_dist <= r {
                        ds.mark_boundary(d);
                    }
                }
            }
        }

        // Generate corrections
        // Strategy: for each pair of defects, find the qubit that connects them
        // (i.e., the qubit that both stabilizers share)
        let mut corrections = Vec::new();
        let mut processed_roots = std::collections::HashSet::new();

        for &d in &defects {
            let root = ds.find(d);

            if processed_roots.contains(&root) {
                continue;
            }
            processed_roots.insert(root);

            // Get all defects in this cluster
            let cluster_defects: Vec<usize> = defects
                .iter()
                .filter(|&&x| ds.find(x) == root)
                .copied()
                .collect();

            // For any cluster with defects, we need to generate corrections
            // The key insight: a single error creates 2 syndrome defects
            // We correct by finding the shared qubit between paired defects

            if cluster_defects.len() == 1 {
                // Single unpaired defect: connect to boundary
                // Correct on any qubit of this stabilizer
                if let Some(q) = self.x_stab_info[cluster_defects[0]].qubits.first() {
                    corrections.push(*q);
                }
            } else {
                // Multiple defects: pair them up and find connecting qubits
                for chunk in cluster_defects.chunks(2) {
                    if chunk.len() == 2 {
                        // Find the shared qubit between these two stabilizers
                        if let Some(q) = self.find_connecting_qubit(
                            &self.x_stab_info[chunk[0]],
                            &self.x_stab_info[chunk[1]],
                        ) {
                            corrections.push(q);
                        }
                    } else if chunk.len() == 1 {
                        // Odd one out: connect to boundary
                        if let Some(q) = self.x_stab_info[chunk[0]].qubits.first() {
                            corrections.push(*q);
                        }
                    }
                }
            }
        }

        corrections
    }

    /// Decode Z syndrome (detects X errors) → returns X corrections
    fn decode_z_syndrome(&self, z_syndrome: &[u8]) -> Vec<usize> {
        if z_syndrome.iter().all(|&s| s == 0) {
            return Vec::new();
        }

        let n = self.num_z_stabs;
        let mut ds = DisjointSet::new(n);

        // Find defects
        let defects: Vec<usize> = z_syndrome
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == 1)
            .map(|(i, _)| i)
            .collect();

        // Mark boundary stabilizers
        for (i, info) in self.z_stab_info.iter().enumerate() {
            if info.is_boundary {
                ds.mark_boundary(i);
            }
        }

        // Grow clusters
        for radius in 1..=(self.distance + 1) {
            let r = radius as f32;

            for i in 0..defects.len() {
                for j in (i + 1)..defects.len() {
                    let d1 = defects[i];
                    let d2 = defects[j];

                    if ds.find(d1) != ds.find(d2) {
                        let dist =
                            Self::stab_distance(&self.z_stab_info[d1], &self.z_stab_info[d2]);
                        if dist <= r * 2.0 {
                            ds.union(d1, d2);
                        }
                    }
                }

                // Connect to boundary
                let d = defects[i];
                if !ds.is_boundary_connected(d) {
                    let info = &self.z_stab_info[d];
                    // Distance to left/right boundary
                    let to_left = info.col + 0.5;
                    let to_right = self.width as f32 - info.col - 0.5;
                    let min_boundary_dist = to_left.min(to_right);

                    if min_boundary_dist <= r {
                        ds.mark_boundary(d);
                    }
                }
            }
        }

        // Generate corrections
        let mut corrections = Vec::new();
        let mut processed_roots = std::collections::HashSet::new();

        for &d in &defects {
            let root = ds.find(d);

            if processed_roots.contains(&root) {
                continue;
            }
            processed_roots.insert(root);

            let cluster_defects: Vec<usize> = defects
                .iter()
                .filter(|&&x| ds.find(x) == root)
                .copied()
                .collect();

            if cluster_defects.len() == 1 {
                // Single unpaired defect: connect to boundary
                if let Some(q) = self.z_stab_info[cluster_defects[0]].qubits.first() {
                    corrections.push(*q);
                }
            } else {
                // Multiple defects: pair them up
                for chunk in cluster_defects.chunks(2) {
                    if chunk.len() == 2 {
                        if let Some(q) = self.find_connecting_qubit(
                            &self.z_stab_info[chunk[0]],
                            &self.z_stab_info[chunk[1]],
                        ) {
                            corrections.push(q);
                        }
                    } else if chunk.len() == 1 {
                        if let Some(q) = self.z_stab_info[chunk[0]].qubits.first() {
                            corrections.push(*q);
                        }
                    }
                }
            }
        }

        corrections
    }

    /// Find a qubit that connects two stabilizers (shared or on path between them)
    fn find_connecting_qubit(&self, a: &StabilizerInfo, b: &StabilizerInfo) -> Option<usize> {
        // First check for shared qubit
        for &qa in &a.qubits {
            for &qb in &b.qubits {
                if qa == qb {
                    return Some(qa);
                }
            }
        }

        // No shared qubit, return first qubit of a
        a.qubits.first().copied()
    }

    /// Decode both X and Z syndromes for a surface code
    ///
    /// Returns (x_corrections, z_corrections) where:
    /// - x_corrections: qubits to apply X (fixes Z syndrome)
    /// - z_corrections: qubits to apply Z (fixes X syndrome)
    pub fn decode(&self, x_syndrome: &[u8], z_syndrome: &[u8]) -> (Vec<usize>, Vec<usize>) {
        // X syndrome detects Z errors → apply Z corrections
        let z_corrections = self.decode_x_syndrome(x_syndrome);

        // Z syndrome detects X errors → apply X corrections
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

    #[test]
    fn test_union_find_with_surface_code() {
        use crate::qec::codes::SurfaceCode;

        println!("\n=== Union-Find Decoder with Surface Code ===");

        let distance = 3;
        let decoder = UnionFindDecoder::new(distance);

        // Test single X error
        let mut code = SurfaceCode::new(distance);
        let center_qubit = 4; // Center of 3x3 grid
        code.apply_x_error_at(center_qubit);

        let (x_syn, z_syn) = code.measure_syndrome();
        println!(
            "X error at qubit {}: x_syn={:?}, z_syn={:?}",
            center_qubit, x_syn, z_syn
        );

        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        println!("Decoder output: x_corr={:?}, z_corr={:?}", x_corr, z_corr);

        // Apply corrections
        code.correct(&x_corr, &z_corr);

        // Check if syndrome cleared
        let (x_syn_after, z_syn_after) = code.measure_syndrome();
        let success = x_syn_after.iter().all(|&s| s == 0) && z_syn_after.iter().all(|&s| s == 0);
        println!("After correction: success={}", success);
    }

    #[test]
    fn test_union_find_single_error_all_positions() {
        use crate::qec::codes::SurfaceCode;

        println!("\n=== Union-Find: Single Error at All Positions ===");

        for distance in [3, 5] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoder::new(distance);

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

            // Union-Find should achieve high single-error correction
            assert!(
                x_success as f64 / n_data as f64 > 0.7,
                "X error correction should be >70%"
            );
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
        }

        println!("\nExpected: Logical error rate should decrease with distance below threshold");
    }
}
