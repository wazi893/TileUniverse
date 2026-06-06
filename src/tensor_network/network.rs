//! Tensor network graph representation.

use super::tensor::{Tensor, TensorIndex};
use std::collections::{HashMap, HashSet};

/// A tensor network represented as a collection of tensors with shared indices.
#[derive(Clone, Debug)]
pub struct TensorNetwork {
    /// All tensors in the network.
    pub tensors: Vec<Tensor>,
    /// Open indices (not contracted) - these are the "physical" qubit wires.
    /// First n_qubits are inputs, next n_qubits are outputs.
    pub open_indices: Vec<TensorIndex>,
    /// Number of qubits in the circuit.
    pub n_qubits: u8,
    /// Index counter for generating unique IDs.
    next_index_id: u32,
}

impl TensorNetwork {
    /// Create a new empty tensor network.
    pub fn new(n_qubits: u8) -> Self {
        Self {
            tensors: Vec::new(),
            open_indices: Vec::new(),
            n_qubits,
            next_index_id: 0,
        }
    }

    /// Generate a new unique index with dimension 2 (qubit).
    pub fn new_qubit_index(&mut self) -> TensorIndex {
        let id = self.next_index_id;
        self.next_index_id += 1;
        TensorIndex::qubit(id)
    }

    /// Generate a new unique index with arbitrary dimension.
    pub fn new_index(&mut self, dim: u8) -> TensorIndex {
        let id = self.next_index_id;
        self.next_index_id += 1;
        TensorIndex::new(id, dim)
    }

    /// Add a tensor to the network.
    pub fn add_tensor(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    /// Number of tensors in the network.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Total memory footprint (number of complex elements).
    pub fn total_elements(&self) -> usize {
        self.tensors.iter().map(|t| t.numel()).sum()
    }

    /// Get the set of all index IDs in the network.
    pub fn all_index_ids(&self) -> HashSet<u32> {
        let mut ids = HashSet::new();
        for tensor in &self.tensors {
            for idx in &tensor.indices {
                ids.insert(idx.id);
            }
        }
        ids
    }

    /// Find which tensors contain a given index.
    pub fn tensors_with_index(&self, index_id: u32) -> Vec<usize> {
        self.tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.has_index(index_id))
            .map(|(i, _)| i)
            .collect()
    }

    /// Build a map from index ID to the tensors that contain it.
    pub fn index_to_tensors_map(&self) -> HashMap<u32, Vec<usize>> {
        let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
        for (tid, tensor) in self.tensors.iter().enumerate() {
            for idx in &tensor.indices {
                map.entry(idx.id).or_default().push(tid);
            }
        }
        map
    }

    /// Find all pairs of tensors that share a contractable (non-open) index.
    pub fn find_contractable_pairs(&self) -> Vec<(usize, usize, TensorIndex)> {
        let open_ids: HashSet<u32> = self.open_indices.iter().map(|i| i.id).collect();
        let index_map = self.index_to_tensors_map();

        let mut pairs = Vec::new();
        for (idx_id, tensor_ids) in index_map {
            // Skip open indices - they shouldn't be contracted
            if open_ids.contains(&idx_id) {
                continue;
            }
            // An index shared by exactly 2 tensors can be contracted
            if tensor_ids.len() == 2 {
                let idx = self.tensors[tensor_ids[0]]
                    .indices
                    .iter()
                    .find(|i| i.id == idx_id)
                    .copied()
                    .unwrap();
                pairs.push((tensor_ids[0], tensor_ids[1], idx));
            }
        }
        pairs
    }

    /// Get all indices that are shared between two specific tensors.
    pub fn shared_indices(&self, t1: usize, t2: usize) -> Vec<TensorIndex> {
        let open_ids: HashSet<u32> = self.open_indices.iter().map(|i| i.id).collect();

        self.tensors[t1]
            .indices
            .iter()
            .filter(|idx| !open_ids.contains(&idx.id) && self.tensors[t2].has_index(idx.id))
            .copied()
            .collect()
    }

    /// Check if the network is fully contracted (single scalar tensor).
    pub fn is_fully_contracted(&self) -> bool {
        self.tensors.len() == 1 && self.tensors[0].is_scalar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let mut network = TensorNetwork::new(2);
        assert_eq!(network.tensor_count(), 0);

        let idx1 = network.new_qubit_index();
        let idx2 = network.new_qubit_index();
        assert_ne!(idx1.id, idx2.id);
    }

    #[test]
    fn test_index_sharing() {
        let mut network = TensorNetwork::new(2);

        let shared_idx = network.new_qubit_index();
        let idx1 = network.new_qubit_index();
        let idx2 = network.new_qubit_index();

        // Tensor 1 has [idx1, shared_idx]
        let t1 = Tensor::new(vec![idx1, shared_idx]);
        network.add_tensor(t1);

        // Tensor 2 has [shared_idx, idx2]
        let t2 = Tensor::new(vec![shared_idx, idx2]);
        network.add_tensor(t2);

        let pairs = network.find_contractable_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].2.id, shared_idx.id);
    }
}
