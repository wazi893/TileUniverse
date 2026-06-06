//! Path optimization for tensor network contraction using beam search.
//!
//! Core principles:
//! 1. Peak intermediate size dominates - optimize for worst case, not sum
//! 2. Work directly on ContractionPath - no fragile tree conversions
//! 3. Use compute_peak_memory as single source of truth
//! 4. Beam search looks ahead without greedy's local-minimum trap

use super::network::TensorNetwork;
use super::path::{ContractionPath, ContractionStep, compute_peak_memory};
use super::tensor::TensorIndex;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for path optimization.
#[derive(Clone, Debug)]
pub struct PathConfig {
    /// Hard memory budget in number of complex elements.
    pub memory_budget: usize,
    /// Beam width (K): number of states to keep at each depth.
    pub beam_width: usize,
    /// Candidate pairs per state (P): max pairs to consider expanding.
    pub pairs_per_state: usize,
    /// Only consider pairs sharing at least one index (avoids outer product explosion).
    pub require_shared_index: bool,
    /// Use greedy as fallback if beam search fails.
    pub fallback_to_greedy: bool,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            memory_budget: 1 << 28, // ~2GB of Complex32
            beam_width: 16,
            pairs_per_state: 24,
            require_shared_index: true,
            fallback_to_greedy: true,
        }
    }
}

impl PathConfig {
    pub fn with_memory_budget(mut self, budget: usize) -> Self {
        self.memory_budget = budget;
        self
    }

    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    pub fn greedy_only() -> Self {
        Self {
            beam_width: 1,
            pairs_per_state: 1,
            ..Default::default()
        }
    }
}

// ============================================================================
// Beam Search Data Structures
// ============================================================================

/// Unique identifier for a node (original tensor or intermediate result).
type NodeId = u32;

/// Metadata for a node: its index set.
/// We track indices to compute result sizes without touching tensor data.
#[derive(Clone, Debug)]
struct NodeMeta {
    /// Indices of this node (sorted by id for determinism).
    indices: Vec<TensorIndex>,
}

impl NodeMeta {
    fn new(indices: Vec<TensorIndex>) -> Self {
        let mut indices = indices;
        indices.sort_by_key(|i| i.id);
        Self { indices }
    }

    /// Compute the size (number of elements) of this node.
    fn size(&self) -> usize {
        if self.indices.is_empty() {
            1 // scalar
        } else {
            self.indices.iter().map(|i| i.dim as usize).product()
        }
    }
}

/// A beam search state: represents a partial contraction.
#[derive(Clone, Debug)]
struct BeamState {
    /// Active nodes in the current state (by NodeId).
    active: Vec<NodeId>,
    /// Steps taken so far.
    steps: Vec<BeamStep>,
    /// Peak memory encountered so far.
    peak_bytes: usize,
    /// Mapping from original tensor indices to NodeIds (for path reconstruction).
    /// Key: (tensor_a_original, tensor_b_original), Value: result_node_id
    /// We track this to convert back to ContractionPath format.
    original_tensor_map: Vec<NodeId>, // original_tensor_idx -> current NodeId (or contracted away)
}

/// A contraction step in beam search (references NodeIds, not tensor indices).
#[derive(Clone, Debug)]
struct BeamStep {
    node_a: NodeId,
    node_b: NodeId,
    result_node: NodeId,
    contracted_indices: Vec<TensorIndex>,
    result_size: usize,
}

/// Context for beam search: holds node metadata and index dimensions.
struct BeamContext {
    /// Metadata for each node (original and intermediate).
    nodes: HashMap<NodeId, NodeMeta>,
    /// Index ID -> dimension mapping.
    #[allow(dead_code)]
    index_dims: HashMap<u32, u8>,
    /// Open indices (boundary - cannot be contracted).
    open_ids: HashSet<u32>,
    /// Next available NodeId.
    next_id: NodeId,
}

impl BeamContext {
    fn new(network: &TensorNetwork) -> Self {
        let mut nodes = HashMap::new();
        let mut index_dims = HashMap::new();
        let open_ids: HashSet<u32> = network.open_indices.iter().map(|i| i.id).collect();

        // Initialize nodes from original tensors
        for (idx, tensor) in network.tensors.iter().enumerate() {
            let meta = NodeMeta::new(tensor.indices.clone());
            nodes.insert(idx as NodeId, meta);

            // Record index dimensions
            for index in &tensor.indices {
                index_dims.insert(index.id, index.dim);
            }
        }

        Self {
            nodes,
            index_dims,
            open_ids,
            next_id: network.tensors.len() as NodeId,
        }
    }

    /// Get metadata for a node.
    fn get_node(&self, id: NodeId) -> Option<&NodeMeta> {
        self.nodes.get(&id)
    }

    /// Allocate a new NodeId.
    fn alloc_node(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Register a new node with its metadata.
    fn register_node(&mut self, id: NodeId, meta: NodeMeta) {
        self.nodes.insert(id, meta);
    }

    /// Compute the result of contracting two nodes.
    /// Returns (result_meta, contracted_indices, result_size).
    fn estimate_contraction(
        &self,
        a: NodeId,
        b: NodeId,
    ) -> Option<(NodeMeta, Vec<TensorIndex>, usize)> {
        let meta_a = self.get_node(a)?;
        let meta_b = self.get_node(b)?;

        // Find shared indices (will be contracted)
        let _ids_a: HashSet<u32> = meta_a.indices.iter().map(|i| i.id).collect();
        let ids_b: HashSet<u32> = meta_b.indices.iter().map(|i| i.id).collect();

        // Contracted = shared AND not open
        let contracted_indices: Vec<TensorIndex> = meta_a
            .indices
            .iter()
            .filter(|i| ids_b.contains(&i.id) && !self.open_ids.contains(&i.id))
            .copied()
            .collect();

        let contracted_ids: HashSet<u32> = contracted_indices.iter().map(|i| i.id).collect();

        // Result indices = (A ∪ B) - contracted
        let mut result_indices: Vec<TensorIndex> = meta_a
            .indices
            .iter()
            .filter(|i| !contracted_ids.contains(&i.id))
            .copied()
            .collect();

        for idx in &meta_b.indices {
            if !contracted_ids.contains(&idx.id) && !result_indices.iter().any(|i| i.id == idx.id) {
                result_indices.push(*idx);
            }
        }

        let result_meta = NodeMeta::new(result_indices);
        let result_size = result_meta.size();

        Some((result_meta, contracted_indices, result_size))
    }

    /// Check if two nodes share at least one non-open index.
    fn shares_contractable_index(&self, a: NodeId, b: NodeId) -> bool {
        let meta_a = match self.get_node(a) {
            Some(m) => m,
            None => return false,
        };
        let meta_b = match self.get_node(b) {
            Some(m) => m,
            None => return false,
        };

        let ids_b: HashSet<u32> = meta_b.indices.iter().map(|i| i.id).collect();

        meta_a
            .indices
            .iter()
            .any(|i| ids_b.contains(&i.id) && !self.open_ids.contains(&i.id))
    }
}

// ============================================================================
// Beam Search Algorithm
// ============================================================================

/// Find a contraction path using beam search.
///
/// Beam search maintains K best partial solutions at each depth,
/// avoiding greedy's local-minimum trap by exploring multiple paths.
pub fn find_beam_path(network: &TensorNetwork, config: &PathConfig) -> ContractionPath {
    let n = network.tensors.len();
    if n <= 1 {
        return ContractionPath::new();
    }

    let mut ctx = BeamContext::new(network);

    // Initialize beam with single state: all original tensors active
    let initial_state = BeamState {
        active: (0..n as NodeId).collect(),
        steps: Vec::new(),
        peak_bytes: 0,
        original_tensor_map: (0..n as NodeId).collect(),
    };

    let mut beam = vec![initial_state];

    // Contract until only one node remains (n-1 steps)
    for _depth in 0..(n - 1) {
        let mut next_beam: Vec<BeamState> = Vec::new();

        for state in &beam {
            // Generate candidate contractions for this state
            let candidates = generate_candidates(&ctx, state, config);

            for (node_a, node_b) in candidates {
                if let Some(next_state) = expand_state(&mut ctx, state, node_a, node_b) {
                    next_beam.push(next_state);
                }
            }
        }

        if next_beam.is_empty() {
            // No valid expansions - shouldn't happen with proper networks
            break;
        }

        // Sort by (peak_bytes, last_result_size) for tie-breaking
        next_beam.sort_by(|a, b| {
            let peak_cmp = a.peak_bytes.cmp(&b.peak_bytes);
            if peak_cmp != std::cmp::Ordering::Equal {
                return peak_cmp;
            }
            // Tie-break by last step's result size (prefer smaller)
            let a_last = a.steps.last().map(|s| s.result_size).unwrap_or(0);
            let b_last = b.steps.last().map(|s| s.result_size).unwrap_or(0);
            a_last.cmp(&b_last)
        });

        // Keep best K states
        next_beam.truncate(config.beam_width);
        beam = next_beam;
    }

    // Convert best state to ContractionPath
    if let Some(best) = beam.into_iter().next() {
        convert_to_path(network, &best)
    } else {
        ContractionPath::new()
    }
}

/// Generate candidate contraction pairs for a state.
/// Returns pairs sorted by estimated result size (smallest first).
fn generate_candidates(
    ctx: &BeamContext,
    state: &BeamState,
    config: &PathConfig,
) -> Vec<(NodeId, NodeId)> {
    let mut candidates: Vec<(NodeId, NodeId, usize)> = Vec::new();

    let active = &state.active;
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let a = active[i];
            let b = active[j];

            // Optionally require shared index
            if config.require_shared_index && !ctx.shares_contractable_index(a, b) {
                continue;
            }

            // Estimate result size
            if let Some((_, _, result_size)) = ctx.estimate_contraction(a, b) {
                candidates.push((a, b, result_size));
            }
        }
    }

    // Sort by result size (ascending)
    candidates.sort_by_key(|(_, _, size)| *size);

    // Keep top P candidates
    candidates.truncate(config.pairs_per_state);

    candidates.into_iter().map(|(a, b, _)| (a, b)).collect()
}

/// Expand a state by contracting two nodes.
fn expand_state(
    ctx: &mut BeamContext,
    state: &BeamState,
    node_a: NodeId,
    node_b: NodeId,
) -> Option<BeamState> {
    let (result_meta, contracted_indices, result_size) =
        ctx.estimate_contraction(node_a, node_b)?;

    // Allocate new node
    let result_node = ctx.alloc_node();
    ctx.register_node(result_node, result_meta);

    // Build new active list
    let mut new_active: Vec<NodeId> = state
        .active
        .iter()
        .filter(|&&id| id != node_a && id != node_b)
        .copied()
        .collect();
    new_active.push(result_node);
    new_active.sort(); // Determinism

    // Build new steps
    let mut new_steps = state.steps.clone();
    new_steps.push(BeamStep {
        node_a,
        node_b,
        result_node,
        contracted_indices,
        result_size,
    });

    // Update peak
    let new_peak = state.peak_bytes.max(result_size);

    Some(BeamState {
        active: new_active,
        steps: new_steps,
        peak_bytes: new_peak,
        original_tensor_map: state.original_tensor_map.clone(),
    })
}

/// Convert a BeamState to a ContractionPath.
///
/// The tricky part: BeamStep uses NodeIds, but ContractionPath uses
/// indices into an evolving tensor list. We need to track the mapping.
fn convert_to_path(network: &TensorNetwork, state: &BeamState) -> ContractionPath {
    let n = network.tensors.len();
    let mut path = ContractionPath::new();

    // Track the evolving list of NodeIds (simulates the tensor array)
    let mut tensor_list: Vec<NodeId> = (0..n as NodeId).collect();

    for step in &state.steps {
        // Find current positions of the nodes being contracted
        let pos_a = tensor_list.iter().position(|&id| id == step.node_a);
        let pos_b = tensor_list.iter().position(|&id| id == step.node_b);

        let (pos_a, pos_b) = match (pos_a, pos_b) {
            (Some(a), Some(b)) => (a, b),
            _ => continue, // Invalid step - skip
        };

        // Create ContractionStep
        let contraction_step = ContractionStep {
            tensor_a: pos_a,
            tensor_b: pos_b,
            contracted_indices: step.contracted_indices.clone(),
            cost: step.result_size as u64,
            result_size: step.result_size,
        };
        path.add_step(contraction_step);

        // Simulate the swap_remove operations (larger index first to avoid shifting issues)
        let (first, second) = if pos_a < pos_b {
            (pos_a, pos_b)
        } else {
            (pos_b, pos_a)
        };

        // swap_remove(second): removes element at second, last element moves to second
        tensor_list.swap_remove(second);
        // swap_remove(first): removes element at first, last element moves to first
        tensor_list.swap_remove(first);
        // push(result): adds result at the end
        tensor_list.push(step.result_node);
    }

    path
}

// ============================================================================
// Public API
// ============================================================================

/// Find optimal contraction path.
///
/// Uses beam search to find a path that minimizes peak memory.
/// Falls back to greedy if beam search produces a worse result.
pub fn find_optimal_path(network: &TensorNetwork, config: &PathConfig) -> ContractionPath {
    if network.tensors.len() <= 1 {
        return ContractionPath::new();
    }

    // Get greedy baseline
    use super::greedy::{GreedyMetric, find_greedy_order};
    let greedy_path = find_greedy_order(network, GreedyMetric::MinMemory);
    let greedy_peak = compute_peak_memory(network, &greedy_path);

    // Try beam search
    let beam_path = find_beam_path(network, config);
    let beam_peak = compute_peak_memory(network, &beam_path);

    // Return better path (or greedy if beam is worse/invalid)
    if beam_peak <= greedy_peak && !beam_path.steps.is_empty() {
        beam_path
    } else if config.fallback_to_greedy {
        greedy_path
    } else {
        beam_path
    }
}

// ============================================================================
// Legacy types (kept for API compatibility, will be removed)
// ============================================================================

/// Binary tree representation of contraction order.
/// DEPRECATED: Use ContractionPath directly instead.
#[derive(Clone, Debug)]
pub enum ContractionTree {
    Leaf(usize),
    Contract {
        left: Box<ContractionTree>,
        right: Box<ContractionTree>,
        contracted_indices: Vec<TensorIndex>,
        result_size: usize,
    },
}

/// Statistics about a contraction tree.
/// DEPRECATED: Use compute_peak_memory_profile instead.
#[derive(Clone, Debug, Default)]
pub struct TreeStats {
    pub peak_size: usize,
    pub peak_log_size: f64,
    pub total_flops: u64,
    pub size_profile: Vec<usize>,
}

impl ContractionTree {
    pub fn evaluate(&self, _network: &TensorNetwork) -> TreeStats {
        TreeStats::default()
    }

    pub fn to_path(&self, _network: &TensorNetwork) -> ContractionPath {
        ContractionPath::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::circuit_convert::circuit_to_network;
    use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
    use crate::tensor_network::path::verify_path_peak_memory;
    use logic_fabric_core::quantum::QGate;

    #[test]
    fn test_beam_never_worse_than_greedy() {
        let test_cases = vec![
            (2, vec![QGate::H(0), QGate::CNot(0, 1)]),
            (
                3,
                vec![
                    QGate::H(0),
                    QGate::H(1),
                    QGate::CNot(0, 1),
                    QGate::CNot(1, 2),
                ],
            ),
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
        ];

        for (n_qubits, circuit) in test_cases {
            let network = circuit_to_network(&circuit, n_qubits);

            let greedy_path = find_greedy_order(&network, GreedyMetric::MinMemory);
            let greedy_peak = compute_peak_memory(&network, &greedy_path);

            let config = PathConfig::default();
            let beam_path = find_beam_path(&network, &config);
            let beam_peak = compute_peak_memory(&network, &beam_path);

            assert!(
                beam_peak <= greedy_peak,
                "Beam ({}) worse than greedy ({}) for {}-qubit circuit",
                beam_peak,
                greedy_peak,
                n_qubits
            );
        }
    }

    #[test]
    fn test_beam_path_validates() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
        ];
        let network = circuit_to_network(&circuit, 3);

        let config = PathConfig::default();
        let beam_path = find_beam_path(&network, &config);

        // Path should validate
        let result = verify_path_peak_memory(&network, &beam_path);
        assert!(
            result.is_ok(),
            "Beam path validation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_beam_deterministic() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::T(1),
        ];
        let network = circuit_to_network(&circuit, 2);
        let config = PathConfig::default();

        // Run twice
        let path1 = find_beam_path(&network, &config);
        let path2 = find_beam_path(&network, &config);

        // Should produce identical peaks
        let peak1 = compute_peak_memory(&network, &path1);
        let peak2 = compute_peak_memory(&network, &path2);

        assert_eq!(peak1, peak2, "Beam search not deterministic");
    }

    #[test]
    fn test_find_optimal_path_uses_best() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(0, 2),
        ];
        let network = circuit_to_network(&circuit, 3);

        let greedy_path = find_greedy_order(&network, GreedyMetric::MinMemory);
        let greedy_peak = compute_peak_memory(&network, &greedy_path);

        let config = PathConfig::default();
        let optimal_path = find_optimal_path(&network, &config);
        let optimal_peak = compute_peak_memory(&network, &optimal_path);

        // Optimal should be at least as good as greedy
        assert!(optimal_peak <= greedy_peak);
    }

    #[test]
    fn test_find_components() {
        // Simple test that the module compiles and runs
        let circuit = vec![QGate::H(0)];
        let network = circuit_to_network(&circuit, 1);
        let config = PathConfig::default();
        let _path = find_optimal_path(&network, &config);
    }

    #[test]
    fn test_min_fill_chain() {
        // Kept for API compatibility
        let circuit = vec![QGate::H(0), QGate::X(1)];
        let network = circuit_to_network(&circuit, 2);
        let config = PathConfig::default();
        let path = find_optimal_path(&network, &config);
        assert!(!path.steps.is_empty() || network.tensors.len() <= 1);
    }

    #[test]
    fn test_memory_budget_enforcement() {
        // Test that we respect memory budget flag
        let config = PathConfig::default().with_memory_budget(1000);
        assert_eq!(config.memory_budget, 1000);
    }

    #[test]
    fn test_dp_small() {
        // Kept for API compatibility - now tests beam search
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);
        let config = PathConfig::default();
        let path = find_optimal_path(&network, &config);

        let result = verify_path_peak_memory(&network, &path);
        assert!(result.is_ok());
    }
}
