use std::collections::HashSet;

/// Represents the physical connectivity of a quantum processor.
#[derive(Clone, Debug)]
pub struct CouplingMap {
    pub num_qubits: u8,
    pub adj: Vec<HashSet<u8>>,
    pub distances: Vec<Vec<u8>>,
}

impl CouplingMap {
    pub fn new(num_qubits: u8) -> Self {
        Self {
            num_qubits,
            adj: vec![HashSet::new(); num_qubits as usize],
            distances: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, u: u8, v: u8) {
        if u < self.num_qubits && v < self.num_qubits {
            self.adj[u as usize].insert(v);
            self.adj[v as usize].insert(u); // Undirected by default for most couplings
        }
    }

    pub fn new_linear(num_qubits: u8) -> Self {
        let mut map = Self::new(num_qubits);
        for i in 0..num_qubits.saturating_sub(1) {
            map.add_edge(i, i + 1);
        }
        map.compute_distances();
        map
    }

    pub fn new_grid(width: u8, height: u8) -> Self {
        let num_qubits = width * height;
        let mut map = Self::new(num_qubits);
        for y in 0..height {
            for x in 0..width {
                let u = y * width + x;
                // Right neighbor
                if x + 1 < width {
                    map.add_edge(u, u + 1);
                }
                // Down neighbor
                if y + 1 < height {
                    map.add_edge(u, u + width);
                }
            }
        }
        map.compute_distances();
        map
    }

    pub fn new_fully_connected(num_qubits: u8) -> Self {
        let mut map = Self::new(num_qubits);
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                map.add_edge(i, j);
            }
        }
        map.compute_distances();
        map
    }

    /// Precompute All-Pairs Shortest Paths (Floyd-Warshall or BFS)
    pub fn compute_distances(&mut self) {
        let n = self.num_qubits as usize;
        let mut dist = vec![vec![u8::MAX; n]; n];

        for i in 0..n {
            dist[i][i] = 0;
            for &neighbor in &self.adj[i] {
                dist[i][neighbor as usize] = 1;
            }
        }

        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[i][k] != u8::MAX && dist[k][j] != u8::MAX {
                        dist[i][j] = std::cmp::min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        self.distances = dist;
    }

    pub fn distance(&self, u: u8, v: u8) -> u8 {
        if self.distances.is_empty() {
            // Fallback if not computed? Or panic?
            // Given it's a library, maybe compute on demand or assert.
            // For now, return heuristic or MAX.
            u8::MAX
        } else {
            self.distances[u as usize][v as usize]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology() {
        let map = CouplingMap::new_linear(4);
        assert_eq!(map.distance(0, 3), 3);
    }
}

use crate::quantum::QGate;
use std::collections::VecDeque;

pub struct SimpleRouter {
    pub map: CouplingMap,
    pub logical_to_physical: Vec<u8>,
    pub physical_to_logical: Vec<u8>,
}

impl SimpleRouter {
    pub fn new(map: CouplingMap) -> Self {
        let n = map.num_qubits;
        let identity: Vec<u8> = (0..n).collect();
        Self {
            map,
            logical_to_physical: identity.clone(),
            physical_to_logical: identity,
        }
    }

    pub fn route(&mut self, circuit: &[QGate]) -> Vec<QGate> {
        let mut routed_circuit = Vec::with_capacity(circuit.len() * 2);

        for gate in circuit {
            match gate {
                QGate::CZ(c, t) => {
                    self.route_two_qubit_gate(&mut routed_circuit, *c, *t, |c, t| QGate::CZ(c, t))
                }
                QGate::CNot(c, t) => {
                    self.route_two_qubit_gate(&mut routed_circuit, *c, *t, |c, t| QGate::CNot(c, t))
                }
                QGate::Swap(a, b) => {
                    self.route_two_qubit_gate(&mut routed_circuit, *a, *b, |a, b| QGate::Swap(a, b))
                }
                // Single qubit gates just apply to current physical location
                g => {
                    if let Some(targets) = g.targets() {
                        let mapped_targets: Vec<u8> = targets
                            .iter()
                            .map(|&t| self.logical_to_physical[t as usize])
                            .collect();
                        // Reconstruct gate with mapped target(s)
                        // This is tedious without a helper, but for 1-qubit gates:
                        match g {
                            QGate::H(_) => routed_circuit.push(QGate::H(mapped_targets[0])),
                            QGate::X(_) => routed_circuit.push(QGate::X(mapped_targets[0])),
                            QGate::Y(_) => routed_circuit.push(QGate::Y(mapped_targets[0])),
                            QGate::Z(_) => routed_circuit.push(QGate::Z(mapped_targets[0])),
                            QGate::Phase(_, phi) => {
                                routed_circuit.push(QGate::Phase(mapped_targets[0], *phi))
                            }
                            QGate::Rx(_, theta) => {
                                routed_circuit.push(QGate::Rx(mapped_targets[0], *theta))
                            }
                            QGate::Ry(_, theta) => {
                                routed_circuit.push(QGate::Ry(mapped_targets[0], *theta))
                            }
                            QGate::Rz(_, theta) => {
                                routed_circuit.push(QGate::Rz(mapped_targets[0], *theta))
                            }
                            QGate::Measure(_) => {
                                routed_circuit.push(QGate::Measure(mapped_targets[0]))
                            }
                            _ => routed_circuit.push(g.clone()), // Fallback (e.g. Toffoli implementation TODO)
                        }
                    } else {
                        routed_circuit.push(g.clone());
                    }
                }
            }
        }
        routed_circuit
    }

    fn route_two_qubit_gate<F>(&mut self, out: &mut Vec<QGate>, l1: u8, l2: u8, gate_ctor: F)
    where
        F: Fn(u8, u8) -> QGate,
    {
        let p1 = self.logical_to_physical[l1 as usize];
        let p2 = self.logical_to_physical[l2 as usize];

        // Find path p1 -> p2
        let path = self.find_path(p1, p2);

        // If adjacent or same (distance 0 error?), just emit
        if path.len() <= 2 {
            out.push(gate_ctor(p1, p2));
            return;
        }

        // Swap p1 along path until it neighbors p2
        // Path is [p1, n1, n2, ..., p2]
        // We swap p1 with n1, then n1 with n2...
        // Actually, we update the mapping as we physically swap.

        let mut curr = p1;
        // Swap along path excluding the last step (which makes them adjacent)
        for &next_node in path.iter().skip(1).take(path.len() - 2) {
            // Emit physical Swap
            out.push(QGate::Swap(curr, next_node));

            // Update mapping
            self.swap_map(curr, next_node);
            curr = next_node;
        }

        // Now curr is adjacent to p2 (which hasn't moved yet, assuming we moved only p1 towards p2)
        // Check adjacency
        // assert!(self.map.distance(curr, p2) == 1);

        out.push(gate_ctor(curr, p2));
    }

    fn swap_map(&mut self, p1: u8, p2: u8) {
        // Swap physical locations in the mapping
        let l1 = self.physical_to_logical[p1 as usize];
        let l2 = self.physical_to_logical[p2 as usize];

        self.logical_to_physical[l1 as usize] = p2;
        self.logical_to_physical[l2 as usize] = p1;

        self.physical_to_logical[p1 as usize] = l2;
        self.physical_to_logical[p2 as usize] = l1;
    }

    fn find_path(&self, start: u8, end: u8) -> Vec<u8> {
        // BFS
        let mut queue = VecDeque::new();
        queue.push_back(vec![start]);
        let mut visited = HashSet::new();
        visited.insert(start);

        while let Some(path) = queue.pop_front() {
            let last = *path.last().unwrap();
            if last == end {
                return path;
            }

            for &neighbor in &self.map.adj[last as usize] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    queue.push_back(new_path);
                }
            }
        }
        panic!("Disconnected topology or invalid qubits");
    }
}

// Helper trait to extract targets
impl QGate {
    pub fn targets(&self) -> Option<Vec<u8>> {
        match self {
            QGate::H(t)
            | QGate::X(t)
            | QGate::Y(t)
            | QGate::Z(t)
            | QGate::Phase(t, _)
            | QGate::Rx(t, _)
            | QGate::Ry(t, _)
            | QGate::Rz(t, _)
            | QGate::Measure(t) => Some(vec![*t]),
            _ => None, // 2+ qubit gates handled separately or defaults
        }
    }
}
