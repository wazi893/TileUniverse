//! Delfosse–Nickerson Union-Find decoder for surface codes.
//!
//! Cluster growth on the qubit-edge syndrome graph, then leaf-peeling on the
//! grown forest. Complexity is nearly linear in the number of checks.
//!
//! "Almost-linear time decoding algorithm for topological codes"
//! (Delfosse & Nickerson, 2017).

use std::collections::HashSet;

/// Syndrome graph for one CSS channel.
///
/// Vertices are the channel's stabilizers plus one virtual boundary. Each data
/// qubit is an edge: two stabilizers that share it, or a single stabilizer to
/// the boundary.
#[derive(Clone, Debug)]
pub struct SyndromeLattice {
    /// Number of real stabilizer vertices.
    pub n_stabs: usize,
    /// Data-qubit count (edge labels are in `0..n_data`).
    pub n_data: usize,
    /// Virtual boundary vertex index (`n_stabs`).
    pub bnode: usize,
    /// `n_stabs + 1`.
    pub n_nodes: usize,
    /// `(u, v, qubit)` undirected edges. `v` may be `bnode`.
    pub edges: Vec<(usize, usize, usize)>,
}

impl SyndromeLattice {
    /// Build from stabilizer supports in syndrome-vector order.
    pub fn from_supports(supports: &[Vec<usize>], n_data: usize) -> Self {
        let n_stabs = supports.len();
        let bnode = n_stabs;
        let mut q_stabs: Vec<Vec<usize>> = vec![Vec::new(); n_data];
        for (si, qs) in supports.iter().enumerate() {
            for &q in qs {
                if q < n_data {
                    q_stabs[q].push(si);
                }
            }
        }
        let mut edges = Vec::new();
        for (q, stabs) in q_stabs.iter().enumerate() {
            match stabs.len() {
                2 => edges.push((stabs[0], stabs[1], q)),
                1 => edges.push((stabs[0], bnode, q)),
                _ => {}
            }
        }
        Self {
            n_stabs,
            n_data,
            bnode,
            n_nodes: n_stabs + 1,
            edges,
        }
    }
}

/// Union-Find with cluster odd-parity and boundary-sink flags.
struct ClusterUf {
    parent: Vec<usize>,
    rank: Vec<u8>,
    odd: Vec<bool>,
    boundary: Vec<bool>,
}

impl ClusterUf {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            odd: vec![false; n],
            boundary: vec![false; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let mut rx = self.find(x);
        let mut ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            std::mem::swap(&mut rx, &mut ry);
        }
        self.parent[ry] = rx;
        if self.rank[rx] == self.rank[ry] {
            self.rank[rx] += 1;
        }
        self.odd[rx] ^= self.odd[ry];
        self.boundary[rx] |= self.boundary[ry];
    }
}

/// Decode one CSS channel by synchronized half-edge growth + peeling.
///
/// The virtual boundary never grows. A cluster is active iff it is odd and
/// does not touch the boundary. Grown edges that connect distinct clusters
/// form a forest; peeling that forest always reproduces the input syndrome
/// if growth ran to completion.
pub fn decode_lattice(lattice: &SyndromeLattice, syndrome: &[u8]) -> Vec<usize> {
    if syndrome.iter().all(|&s| s == 0) {
        return Vec::new();
    }
    debug_assert_eq!(syndrome.len(), lattice.n_stabs);

    let n = lattice.n_nodes;
    let mut ds = ClusterUf::new(n);
    ds.boundary[lattice.bnode] = true;
    for (i, &s) in syndrome.iter().enumerate() {
        if s == 1 {
            ds.odd[i] = true;
        }
    }

    let mut growth = vec![0u8; lattice.edges.len()];
    let mut fused = vec![false; lattice.edges.len()];
    let max_steps = n * 2 + 2;

    for _ in 0..max_steps {
        let mut active_root = vec![false; n];
        let mut any_active = false;
        for v in 0..lattice.n_stabs {
            let r = ds.find(v);
            if ds.odd[r] && !ds.boundary[r] {
                active_root[r] = true;
                any_active = true;
            }
        }
        if !any_active {
            break;
        }

        for (ei, &(u, v, _)) in lattice.edges.iter().enumerate() {
            if fused[ei] {
                continue;
            }
            let ru = ds.find(u);
            let rv = ds.find(v);
            if ru == rv {
                continue;
            }
            if active_root[ru] {
                growth[ei] = growth[ei].saturating_add(1);
            }
            if active_root[rv] {
                growth[ei] = growth[ei].saturating_add(1);
            }
        }

        for (ei, &(u, v, _)) in lattice.edges.iter().enumerate() {
            if fused[ei] || growth[ei] < 2 {
                continue;
            }
            if ds.find(u) != ds.find(v) {
                fused[ei] = true;
                ds.union(u, v);
            }
        }
    }

    peel(lattice, &fused, syndrome)
}

/// Leaf-peel the grown forest. Never peel the boundary sink: odd parity on a
/// component that touches the boundary is absorbed by the boundary edge.
fn peel(lattice: &SyndromeLattice, fused: &[bool], syndrome: &[u8]) -> Vec<usize> {
    let n = lattice.n_nodes;
    let mut adj: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n];
    let mut deg = vec![0i32; n];
    for (ei, &(u, v, q)) in lattice.edges.iter().enumerate() {
        if fused[ei] {
            adj[u].push((v, ei, q));
            adj[v].push((u, ei, q));
            deg[u] += 1;
            deg[v] += 1;
        }
    }

    let mut syn = vec![0u8; n];
    for (i, &s) in syndrome.iter().enumerate() {
        syn[i] = s;
    }

    let mut used_edge = vec![false; lattice.edges.len()];
    let mut parity: HashSet<usize> = HashSet::new();

    loop {
        let mut did = false;
        for v in 0..lattice.n_stabs {
            if deg[v] != 1 {
                continue;
            }
            let mut found = None;
            for &(u, ei, q) in &adj[v] {
                if !used_edge[ei] {
                    found = Some((u, ei, q));
                    break;
                }
            }
            let Some((u, ei, q)) = found else {
                continue;
            };
            used_edge[ei] = true;
            deg[v] -= 1;
            deg[u] -= 1;
            if syn[v] == 1 {
                if !parity.remove(&q) {
                    parity.insert(q);
                }
                syn[u] ^= 1;
                syn[v] = 0;
            }
            did = true;
        }
        if !did {
            break;
        }
    }

    let mut corr: Vec<usize> = parity.into_iter().collect();
    corr.sort_unstable();
    corr
}

/// Improved Union-Find decoder (alias of the canonical decoder).
///
/// Use [`crate::qec::decoder::UnionFindDecoder`].
#[deprecated(
    since = "0.79.0",
    note = "Use engine::qec::UnionFindDecoder; V2 is now the same growth+peel decoder"
)]
pub type UnionFindDecoderV2 = crate::qec::decoder::UnionFindDecoder;

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::qec::codes::SurfaceCode;
    use crate::qec::decoder::UnionFindDecoder;

    fn cleared_and_logical(code: &SurfaceCode) -> (bool, bool, bool) {
        let (x_after, z_after) = code.measure_syndrome();
        let cleared = x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);
        (
            cleared,
            code.has_logical_x_error(),
            code.has_logical_z_error(),
        )
    }

    #[test]
    fn lattice_d3_z_channel_center_edge() {
        let code = SurfaceCode::new(3);
        let lat = SyndromeLattice::from_supports(code.z_stabilizer_supports(), code.n_data);
        // Qubit 4 is shared by two Z-stabs → one interior edge.
        assert!(
            lat.edges.iter().any(|&(_, _, q)| q == 4),
            "center qubit must be a Z-channel edge"
        );
    }

    #[test]
    fn v2_alias_single_errors_have_no_logical() {
        for distance in [3, 5, 7] {
            let n_data = distance * distance;
            let decoder = UnionFindDecoderV2::new(distance);
            for q in 0..n_data {
                for apply in [
                    |c: &mut SurfaceCode, q| c.apply_x_error_at(q),
                    |c: &mut SurfaceCode, q| c.apply_z_error_at(q),
                    |c: &mut SurfaceCode, q| c.apply_y_error_at(q),
                ] {
                    let mut code = SurfaceCode::new(distance);
                    apply(&mut code, q);
                    let (x_syn, z_syn) = code.measure_syndrome();
                    assert!(
                        !x_syn.iter().all(|&s| s == 0) || !z_syn.iter().all(|&s| s == 0),
                        "d={distance} q={q}: weight-1 must be detectable"
                    );
                    let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
                    assert!(
                        x_corr.len() <= 1 && z_corr.len() <= 1,
                        "d={distance} q={q}: min-weight, got x={x_corr:?} z={z_corr:?}"
                    );
                    code.correct(&x_corr, &z_corr);
                    let (cleared, lx, lz) = cleared_and_logical(&code);
                    assert!(cleared, "d={distance} q={q}: syndrome must clear");
                    assert!(
                        !lx && !lz,
                        "d={distance} q={q}: weight-1 must not be a logical"
                    );
                }
            }
        }
    }

    #[test]
    fn decode_lattice_center_x_is_qubit_4() {
        let mut code = SurfaceCode::new(3);
        code.apply_x_error_at(4);
        let decoder = UnionFindDecoder::new(3);
        let (x_syn, z_syn) = code.measure_syndrome();
        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert!(z_corr.is_empty());
        assert_eq!(x_corr, vec![4], "center X must peel to the shared qubit");
    }
}
