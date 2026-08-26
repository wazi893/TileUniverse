//! Contraction path representation.

use super::network::TensorNetwork;
use super::tensor::TensorIndex;
use std::collections::HashSet;

/// A single step in a contraction path.
#[derive(Clone, Debug)]
pub struct ContractionStep {
    /// First tensor index in the current tensor list.
    pub tensor_a: usize,
    /// Second tensor index in the current tensor list.
    pub tensor_b: usize,
    /// Indices being contracted (summed over).
    pub contracted_indices: Vec<TensorIndex>,
    /// Estimated cost (number of multiplications).
    pub cost: u64,
    /// Estimated result size (number of elements).
    pub result_size: usize,
}

/// A complete contraction order for a tensor network.
#[derive(Clone, Debug)]
pub struct ContractionPath {
    /// Ordered sequence of contraction steps.
    pub steps: Vec<ContractionStep>,
    /// Total estimated cost (sum of all step costs).
    pub total_cost: u64,
    /// Peak memory usage (maximum intermediate tensor size).
    pub peak_memory: usize,
}

impl ContractionPath {
    /// Create an empty contraction path.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_cost: 0,
            peak_memory: 0,
        }
    }

    /// Add a step and update statistics.
    pub fn add_step(&mut self, step: ContractionStep) {
        self.total_cost += step.cost;
        self.peak_memory = self.peak_memory.max(step.result_size);
        self.steps.push(step);
    }

    /// Number of steps in the path.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if path is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for ContractionPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute peak memory by replaying a contraction path from first principles.
///
/// This is the **single source of truth** for peak memory estimation.
/// All optimizers must produce paths whose recorded `peak_memory` matches
/// the value returned by this function.
///
/// The function simulates the contraction sequence without actually performing
/// the contractions - it only tracks tensor index sets and computes result sizes.
///
/// # Arguments
/// * `network` - The tensor network to contract
/// * `path` - The contraction path to replay
///
/// # Returns
/// Peak memory in number of elements (multiply by 8 for bytes with Complex32)
pub fn compute_peak_memory(network: &TensorNetwork, path: &ContractionPath) -> usize {
    if path.steps.is_empty() {
        // No contractions - peak is the largest input tensor
        return network.tensors.iter().map(|t| t.numel()).max().unwrap_or(0);
    }

    // Track tensor index sets (lightweight - no actual data)
    let mut tensor_indices: Vec<Vec<TensorIndex>> =
        network.tensors.iter().map(|t| t.indices.clone()).collect();

    let mut peak = 0usize;

    for step in &path.steps {
        // Validate step indices
        if step.tensor_a >= tensor_indices.len() || step.tensor_b >= tensor_indices.len() {
            // Invalid path - return max possible to signal error
            return usize::MAX;
        }

        let indices_a = &tensor_indices[step.tensor_a];
        let indices_b = &tensor_indices[step.tensor_b];

        // Compute contracted index IDs
        let contracted_ids: HashSet<u32> = step.contracted_indices.iter().map(|i| i.id).collect();

        // Result indices = (A ∪ B) - contracted
        // Keep indices from A that aren't contracted
        let mut result_indices: Vec<TensorIndex> = indices_a
            .iter()
            .filter(|idx| !contracted_ids.contains(&idx.id))
            .copied()
            .collect();

        // Add indices from B that aren't contracted and aren't already in result
        for idx in indices_b {
            if !contracted_ids.contains(&idx.id) && !result_indices.iter().any(|i| i.id == idx.id) {
                result_indices.push(*idx);
            }
        }

        // Compute result size (product of dimensions, minimum 1 for scalar)
        let result_size: usize = if result_indices.is_empty() {
            1
        } else {
            result_indices.iter().map(|i| i.dim as usize).product()
        };

        peak = peak.max(result_size);

        // Update tensor list using same removal logic as execute_path
        let (first, second) = if step.tensor_a < step.tensor_b {
            (step.tensor_a, step.tensor_b)
        } else {
            (step.tensor_b, step.tensor_a)
        };
        tensor_indices.swap_remove(second);
        tensor_indices.swap_remove(first);
        tensor_indices.push(result_indices);
    }

    peak
}

/// Detailed result from peak memory computation, for debugging.
#[derive(Clone, Debug)]
pub struct PeakMemoryProfile {
    /// Peak memory (max result size across all steps)
    pub peak: usize,
    /// Result size at each step
    pub step_sizes: Vec<usize>,
    /// Index of the step that achieves the peak
    pub peak_step: usize,
    /// Indices of the tensor at the peak step
    pub peak_indices: Vec<TensorIndex>,
}

/// Compute peak memory with detailed profiling information.
///
/// Use this for debugging path optimizers and slicing planners.
pub fn compute_peak_memory_profile(
    network: &TensorNetwork,
    path: &ContractionPath,
) -> PeakMemoryProfile {
    if path.steps.is_empty() {
        let max_input = network.tensors.iter().map(|t| t.numel()).max().unwrap_or(0);
        return PeakMemoryProfile {
            peak: max_input,
            step_sizes: vec![],
            peak_step: 0,
            peak_indices: vec![],
        };
    }

    let mut tensor_indices: Vec<Vec<TensorIndex>> =
        network.tensors.iter().map(|t| t.indices.clone()).collect();

    let mut peak = 0usize;
    let mut peak_step = 0usize;
    let mut peak_indices: Vec<TensorIndex> = vec![];
    let mut step_sizes: Vec<usize> = Vec::with_capacity(path.steps.len());

    for (step_idx, step) in path.steps.iter().enumerate() {
        if step.tensor_a >= tensor_indices.len() || step.tensor_b >= tensor_indices.len() {
            step_sizes.push(usize::MAX);
            continue;
        }

        let indices_a = &tensor_indices[step.tensor_a];
        let indices_b = &tensor_indices[step.tensor_b];

        let contracted_ids: HashSet<u32> = step.contracted_indices.iter().map(|i| i.id).collect();

        let mut result_indices: Vec<TensorIndex> = indices_a
            .iter()
            .filter(|idx| !contracted_ids.contains(&idx.id))
            .copied()
            .collect();

        for idx in indices_b {
            if !contracted_ids.contains(&idx.id) && !result_indices.iter().any(|i| i.id == idx.id) {
                result_indices.push(*idx);
            }
        }

        let result_size: usize = if result_indices.is_empty() {
            1
        } else {
            result_indices.iter().map(|i| i.dim as usize).product()
        };

        step_sizes.push(result_size);

        if result_size > peak {
            peak = result_size;
            peak_step = step_idx;
            peak_indices = result_indices.clone();
        }

        let (first, second) = if step.tensor_a < step.tensor_b {
            (step.tensor_a, step.tensor_b)
        } else {
            (step.tensor_b, step.tensor_a)
        };
        tensor_indices.swap_remove(second);
        tensor_indices.swap_remove(first);
        tensor_indices.push(result_indices);
    }

    PeakMemoryProfile {
        peak,
        step_sizes,
        peak_step,
        peak_indices,
    }
}

/// Verify that a path's recorded peak_memory matches the replayed computation.
///
/// Returns Ok(()) if consistent, Err with details if mismatch.
pub fn verify_path_peak_memory(
    network: &TensorNetwork,
    path: &ContractionPath,
) -> Result<(), String> {
    let computed = compute_peak_memory(network, path);
    let recorded = path.peak_memory;

    if computed == recorded {
        Ok(())
    } else {
        let profile = compute_peak_memory_profile(network, path);
        Err(format!(
            "Peak memory mismatch: recorded={}, computed={}. \
             Peak occurs at step {} with indices {:?}. \
             Step sizes: {:?}",
            recorded, computed, profile.peak_step, profile.peak_indices, profile.step_sizes
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::circuit_convert::circuit_to_network;
    use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
    use crate::tensor_network::tensor::Tensor;
    use logic_fabric_core::quantum::QGate;

    #[test]
    fn test_path_creation() {
        let mut path = ContractionPath::new();
        assert!(path.is_empty());

        path.add_step(ContractionStep {
            tensor_a: 0,
            tensor_b: 1,
            contracted_indices: vec![],
            cost: 100,
            result_size: 16,
        });

        assert_eq!(path.len(), 1);
        assert_eq!(path.total_cost, 100);
        assert_eq!(path.peak_memory, 16);
    }

    #[test]
    fn test_compute_peak_memory_simple_circuit() {
        // Simple 2-qubit circuit
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        let computed = compute_peak_memory(&network, &path);
        let recorded = path.peak_memory;

        assert_eq!(
            computed, recorded,
            "Peak memory mismatch: computed={}, recorded={}",
            computed, recorded
        );
    }

    #[test]
    fn test_compute_peak_memory_larger_circuit() {
        // Larger circuit to stress test
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::T(0),
            QGate::T(1),
            QGate::CNot(0, 2),
        ];
        let network = circuit_to_network(&circuit, 3);
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        let result = verify_path_peak_memory(&network, &path);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_compute_peak_memory_profile() {
        let circuit = vec![QGate::H(0), QGate::H(1), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        let profile = compute_peak_memory_profile(&network, &path);

        // Profile should have one entry per step
        assert_eq!(profile.step_sizes.len(), path.steps.len());

        // Peak should match computed
        assert_eq!(profile.peak, compute_peak_memory(&network, &path));

        // Peak step should be valid
        assert!(profile.peak_step < path.steps.len() || path.steps.is_empty());
    }

    #[test]
    fn test_replay_peak_equivalence_random_circuits() {
        // Test multiple random-ish circuits to catch edge cases
        let test_cases = vec![
            // (n_qubits, gates)
            (2, vec![QGate::H(0), QGate::X(1)]),
            (3, vec![QGate::H(0), QGate::CNot(0, 1), QGate::CNot(1, 2)]),
            (
                4,
                vec![
                    QGate::H(0),
                    QGate::H(1),
                    QGate::H(2),
                    QGate::H(3),
                    QGate::CNot(0, 1),
                    QGate::CNot(2, 3),
                    QGate::CZ(1, 2),
                ],
            ),
            (
                3,
                vec![
                    QGate::X(0),
                    QGate::Y(1),
                    QGate::Z(2),
                    QGate::CNot(0, 1),
                    QGate::CNot(1, 2),
                    QGate::CNot(2, 0),
                ],
            ),
        ];

        for (n_qubits, circuit) in test_cases {
            let network = circuit_to_network(&circuit, n_qubits);
            let path = find_greedy_order(&network, GreedyMetric::MinMemory);

            let result = verify_path_peak_memory(&network, &path);
            assert!(
                result.is_ok(),
                "Failed for {}-qubit circuit with {} gates: {:?}",
                n_qubits,
                circuit.len(),
                result.err()
            );
        }
    }

    #[test]
    fn test_empty_path() {
        let network = TensorNetwork::new(1);
        let path = ContractionPath::new();

        // Empty path should return 0 (no tensors to measure)
        let computed = compute_peak_memory(&network, &path);
        assert_eq!(computed, 0);
    }

    #[test]
    fn test_single_tensor_network() {
        let mut network = TensorNetwork::new(1);
        let idx = TensorIndex::new(1, 4);
        network.add_tensor(Tensor::new(vec![idx]));

        let path = ContractionPath::new(); // No contractions needed

        // Peak should be the single tensor's size
        let computed = compute_peak_memory(&network, &path);
        assert_eq!(computed, 4);
    }
}
