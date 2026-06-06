//! Greedy contraction order finder.

use super::contraction::contract_tensors_smart;
use super::network::TensorNetwork;
use super::path::{ContractionPath, ContractionStep};
use super::tensor::{Tensor, TensorIndex};
use std::collections::HashSet;

/// Cost metric for greedy selection.
#[derive(Clone, Copy, Debug, Default)]
pub enum GreedyMetric {
    /// Minimize intermediate tensor size (memory-optimal).
    #[default]
    MinMemory,
    /// Minimize FLOPs for each contraction.
    MinFlops,
    /// Minimize number of indices in result.
    MinIndices,
}

/// Find a greedy contraction order.
///
/// At each step, selects the contraction that minimizes the chosen metric.
/// Not globally optimal but O(n^2) per step and works well for shallow circuits.
pub fn find_greedy_order(network: &TensorNetwork, metric: GreedyMetric) -> ContractionPath {
    let mut path = ContractionPath::new();
    let mut tensors = network.tensors.clone();
    let open_ids: HashSet<u32> = network.open_indices.iter().map(|i| i.id).collect();

    while tensors.len() > 1 {
        let best = find_best_contraction(&tensors, &open_ids, metric);

        match best {
            Some((i, j, contracted, cost, result_size)) => {
                let step = ContractionStep {
                    tensor_a: i,
                    tensor_b: j,
                    contracted_indices: contracted.clone(),
                    cost,
                    result_size,
                };
                path.add_step(step);

                // Contract and update tensor list
                let result = contract_tensors_smart(&tensors[i], &tensors[j], &contracted);

                // Remove contracted tensors (larger index first to avoid shifting)
                let (first, second) = if i < j { (i, j) } else { (j, i) };
                tensors.swap_remove(second);
                tensors.swap_remove(first);
                tensors.push(result);
            }
            None => {
                // No more contractions possible - remaining tensors share no indices
                // This can happen if the network is disconnected
                break;
            }
        }
    }

    path
}

/// Find the best contraction pair according to the metric.
///
/// Returns None only when there's 0 or 1 tensor remaining.
/// When tensors have no shared indices (disconnected components),
/// returns an outer product contraction (empty shared indices).
fn find_best_contraction(
    tensors: &[Tensor],
    open_ids: &HashSet<u32>,
    metric: GreedyMetric,
) -> Option<(usize, usize, Vec<TensorIndex>, u64, usize)> {
    if tensors.len() < 2 {
        return None;
    }

    let mut best_score = f64::MAX;
    let mut best: Option<(usize, usize, Vec<TensorIndex>, u64, usize)> = None;

    for i in 0..tensors.len() {
        for j in (i + 1)..tensors.len() {
            // Find shared non-open indices
            let shared: Vec<TensorIndex> = tensors[i]
                .indices
                .iter()
                .filter(|idx| !open_ids.contains(&idx.id) && tensors[j].has_index(idx.id))
                .copied()
                .collect();

            if shared.is_empty() {
                continue; // Prefer contractions with shared indices first
            }

            let (score, cost, result_size) =
                compute_contraction_cost(&tensors[i], &tensors[j], &shared, metric);

            if score < best_score {
                best_score = score;
                best = Some((i, j, shared, cost, result_size));
            }
        }
    }

    // If no shared-index contractions found but multiple tensors remain,
    // we have disconnected components - use outer product to combine them
    if best.is_none() && tensors.len() >= 2 {
        // Pick first two tensors for outer product (empty shared indices)
        let result_size = tensors[0].numel() * tensors[1].numel();
        best = Some((0, 1, vec![], result_size as u64, result_size));
    }

    best
}

/// Compute the cost/score of a contraction.
fn compute_contraction_cost(
    t1: &Tensor,
    t2: &Tensor,
    contracted: &[TensorIndex],
    metric: GreedyMetric,
) -> (f64, u64, usize) {
    let contracted_dim: usize = contracted.iter().map(|i| i.dim as usize).product();

    // Count uncontracted indices
    let contracted_ids: HashSet<u32> = contracted.iter().map(|i| i.id).collect();

    let t1_kept: usize = t1
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .map(|i| i.dim as usize)
        .product::<usize>()
        .max(1);

    let t2_kept: usize = t2
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .map(|i| i.dim as usize)
        .product::<usize>()
        .max(1);

    let result_size = t1_kept * t2_kept;
    let flops = contracted_dim * result_size;
    let cost = flops as u64;

    let score = match metric {
        GreedyMetric::MinMemory => result_size as f64,
        GreedyMetric::MinFlops => flops as f64,
        GreedyMetric::MinIndices => {
            let t1_uncontracted = t1
                .indices
                .iter()
                .filter(|i| !contracted_ids.contains(&i.id))
                .count();
            let t2_uncontracted = t2
                .indices
                .iter()
                .filter(|i| !contracted_ids.contains(&i.id))
                .count();
            (t1_uncontracted + t2_uncontracted) as f64
        }
    };

    (score, cost, result_size)
}

/// Execute a contraction path on a tensor network, returning the final tensor.
pub fn execute_path(network: &TensorNetwork, path: &ContractionPath) -> Tensor {
    let mut tensors = network.tensors.clone();

    for step in &path.steps {
        let result = contract_tensors_smart(
            &tensors[step.tensor_a],
            &tensors[step.tensor_b],
            &step.contracted_indices,
        );

        // Remove contracted tensors (larger index first)
        let (first, second) = if step.tensor_a < step.tensor_b {
            (step.tensor_a, step.tensor_b)
        } else {
            (step.tensor_b, step.tensor_a)
        };
        tensors.swap_remove(second);
        tensors.swap_remove(first);
        tensors.push(result);
    }

    tensors
        .pop()
        .unwrap_or_else(|| Tensor::scalar(logic_fabric_core::quantum::Complex32::new(0.0, 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::tensor::TensorIndex;

    #[test]
    fn test_greedy_simple_chain() {
        // Create a simple chain: A - B - C
        let mut network = TensorNetwork::new(0);

        let idx_ab = TensorIndex::qubit(0);
        let idx_bc = TensorIndex::qubit(1);
        let idx_a = TensorIndex::qubit(2);
        let idx_c = TensorIndex::qubit(3);

        // A has [idx_a, idx_ab]
        let a = Tensor::new(vec![idx_a, idx_ab]);
        network.add_tensor(a);

        // B has [idx_ab, idx_bc]
        let b = Tensor::new(vec![idx_ab, idx_bc]);
        network.add_tensor(b);

        // C has [idx_bc, idx_c]
        let c = Tensor::new(vec![idx_bc, idx_c]);
        network.add_tensor(c);

        // Mark idx_a and idx_c as open
        network.open_indices = vec![idx_a, idx_c];

        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        // Should have 2 steps for 3 tensors
        assert_eq!(path.len(), 2);
    }
}
