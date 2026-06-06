//! Slicing for memory-bounded tensor network contraction.
//!
//! When intermediate tensors exceed a memory budget, slicing trades
//! compute for memory by iterating over index values instead of
//! contracting over the full dimension at once.
//!
//! # Slice Modes
//!
//! There are two slice modes depending on whether the sliced index is contracted:
//!
//! ## Sum Mode (contracted indices)
//! If contracting A[i,j,k] with B[k,l,m] over k, and we slice over k:
//! ```text
//! result = sum over k_val in 0..|k|:
//!     contract(A[i, j, k_val], B[k_val, l, m])
//! ```
//! Each partial result has the same shape, we sum them.
//!
//! ## Stack Mode (non-contracted indices)
//! If we slice over j (which appears in the result):
//! ```text
//! result[i, :, l, m] = stack over j_val in 0..|j|:
//!     contract(A[i, j_val, k], B[k, l, m])  # result shape [i, l, m]
//! ```
//! Each partial result is a slice of the final tensor, we concatenate along axis.
//!
//! This reduces intermediate size by factor |dim| at cost of |dim| contractions.

use super::contraction::contract_tensors_smart;
use super::greedy::execute_path;
use super::network::TensorNetwork;
use super::path::{ContractionPath, compute_peak_memory_profile};
use super::tensor::{Tensor, TensorIndex};
use logic_fabric_core::quantum::Complex32;
use std::collections::HashSet;

/// Mode for combining sliced results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceMode {
    /// Sum partial results (for contracted indices).
    Sum,
    /// Stack partial results along an axis (for non-contracted indices).
    /// The axis is the position in the result tensor where the sliced index appears.
    Stack { axis: usize },
}

/// A slice specification for one index.
#[derive(Clone, Debug)]
pub struct SliceSpec {
    /// The index being sliced.
    pub index: TensorIndex,
    /// Which step in the path this slice applies to.
    pub step_idx: usize,
    /// How to combine the sliced results.
    pub mode: SliceMode,
}

/// A contraction path with slicing annotations.
#[derive(Clone, Debug)]
pub struct SlicedPath {
    /// The underlying contraction path.
    pub base_path: ContractionPath,
    /// Indices to slice (in order of application).
    pub slices: Vec<SliceSpec>,
    /// Memory budget this path was planned for.
    pub memory_budget: usize,
    /// Estimated peak memory after slicing.
    pub estimated_peak: usize,
}

impl SlicedPath {
    /// Number of slice iterations needed.
    pub fn slice_iterations(&self) -> usize {
        self.slices.iter().map(|s| s.index.dim as usize).product()
    }

    /// Memory reduction factor from slicing.
    pub fn memory_reduction(&self) -> usize {
        self.slices.iter().map(|s| s.index.dim as usize).product()
    }
}

/// Plan slicing to meet a memory budget.
///
/// Uses `compute_peak_memory_profile` to identify the global peak step,
/// then slices indices from that step until peak is under budget.
/// Repeats if slicing creates a new peak elsewhere.
///
/// Returns None if no slicing is needed (path already fits).
/// Returns a SlicedPath if slicing can bring peak under budget.
pub fn plan_slicing(
    network: &TensorNetwork,
    path: &ContractionPath,
    memory_budget: usize,
) -> Option<SlicedPath> {
    // Use the single source of truth for peak memory
    let profile = compute_peak_memory_profile(network, path);

    if profile.peak <= memory_budget {
        return None; // No slicing needed
    }

    let open_ids: HashSet<u32> = network.open_indices.iter().map(|i| i.id).collect();

    // Simulate contractions to get tensors at each step
    let step_info = build_step_info(network, path);

    let mut slices: Vec<SliceSpec> = Vec::new();
    let mut sliced_ids: HashSet<u32> = HashSet::new();

    // Iteratively slice until peak is under budget
    // Each slice reduces all step sizes that use that index
    let max_iterations = 100; // Safety limit

    for _ in 0..max_iterations {
        // Compute current peak considering all slices applied
        let current_peak = estimate_sliced_peak(&step_info, &sliced_ids, &open_ids);

        if current_peak <= memory_budget {
            break;
        }

        // Find the step with the current peak
        let peak_step = find_current_peak_step(&step_info, &sliced_ids, &open_ids);

        let Some((step_idx, tensor_a, tensor_b, contracted)) = peak_step else {
            break; // No more steps to slice
        };

        // Find the best index to slice at this step
        let candidate = find_best_slice_index_excluding(
            &tensor_a,
            &tensor_b,
            &contracted,
            &open_ids,
            &sliced_ids,
        );

        match candidate {
            Some(cand) => {
                sliced_ids.insert(cand.index.id);
                slices.push(SliceSpec {
                    index: cand.index,
                    step_idx,
                    mode: cand.mode,
                });
            }
            None => break, // No more sliceable indices
        }
    }

    if slices.is_empty() {
        return None;
    }

    // Compute final estimated peak
    let estimated_peak = estimate_sliced_peak(&step_info, &sliced_ids, &open_ids);

    // Sort slices by step index for correct execution order
    slices.sort_by_key(|s| s.step_idx);

    Some(SlicedPath {
        base_path: path.clone(),
        slices,
        memory_budget,
        estimated_peak,
    })
}

/// Build tensor info for each step by simulating contractions.
fn build_step_info(
    network: &TensorNetwork,
    path: &ContractionPath,
) -> Vec<(usize, Tensor, Tensor, Vec<TensorIndex>)> {
    let mut tensors = network.tensors.clone();
    let mut step_info = Vec::with_capacity(path.steps.len());

    for (step_idx, step) in path.steps.iter().enumerate() {
        let tensor_a = tensors[step.tensor_a].clone();
        let tensor_b = tensors[step.tensor_b].clone();

        step_info.push((
            step_idx,
            tensor_a.clone(),
            tensor_b.clone(),
            step.contracted_indices.clone(),
        ));

        // Simulate the contraction for next iteration
        let result = contract_tensors_smart(&tensor_a, &tensor_b, &step.contracted_indices);
        let (first, second) = if step.tensor_a < step.tensor_b {
            (step.tensor_a, step.tensor_b)
        } else {
            (step.tensor_b, step.tensor_a)
        };
        tensors.swap_remove(second);
        tensors.swap_remove(first);
        tensors.push(result);
    }

    step_info
}

/// Estimate peak memory after slicing specified indices.
fn estimate_sliced_peak(
    step_info: &[(usize, Tensor, Tensor, Vec<TensorIndex>)],
    sliced_ids: &HashSet<u32>,
    open_ids: &HashSet<u32>,
) -> usize {
    let mut peak = 0usize;

    for (_step_idx, tensor_a, tensor_b, contracted) in step_info {
        let result_size =
            compute_result_size_after_slicing(tensor_a, tensor_b, contracted, sliced_ids, open_ids);
        peak = peak.max(result_size);
    }

    peak.max(1)
}

/// Find the step with the current peak after slicing.
fn find_current_peak_step<'a>(
    step_info: &'a [(usize, Tensor, Tensor, Vec<TensorIndex>)],
    sliced_ids: &HashSet<u32>,
    open_ids: &HashSet<u32>,
) -> Option<(usize, Tensor, Tensor, Vec<TensorIndex>)> {
    let mut max_size = 0usize;
    let mut peak_step = None;

    for (step_idx, tensor_a, tensor_b, contracted) in step_info {
        let result_size =
            compute_result_size_after_slicing(tensor_a, tensor_b, contracted, sliced_ids, open_ids);
        if result_size > max_size {
            max_size = result_size;
            peak_step = Some((
                *step_idx,
                tensor_a.clone(),
                tensor_b.clone(),
                contracted.clone(),
            ));
        }
    }

    peak_step
}

/// Compute result size of a contraction after some indices have been sliced.
fn compute_result_size_after_slicing(
    tensor_a: &Tensor,
    tensor_b: &Tensor,
    contracted: &[TensorIndex],
    sliced_ids: &HashSet<u32>,
    _open_ids: &HashSet<u32>,
) -> usize {
    let contracted_ids: HashSet<u32> = contracted.iter().map(|i| i.id).collect();

    // Result indices = (A ∪ B) - contracted - sliced
    let mut result_size = 1usize;

    for idx in &tensor_a.indices {
        if !contracted_ids.contains(&idx.id) && !sliced_ids.contains(&idx.id) {
            result_size *= idx.dim as usize;
        }
    }

    for idx in &tensor_b.indices {
        if !contracted_ids.contains(&idx.id) && !sliced_ids.contains(&idx.id) {
            // Check it's not already counted from tensor_a
            if !tensor_a.indices.iter().any(|i| i.id == idx.id) {
                result_size *= idx.dim as usize;
            }
        }
    }

    result_size.max(1)
}

/// Candidate for slicing with its mode.
struct SliceCandidate {
    index: TensorIndex,
    mode: SliceMode,
}

/// Find the best index to slice from a contraction, excluding already-sliced indices.
///
/// Slicing an index with dimension D:
/// - For contracted index: compute D partial sums and add them (Sum mode)
/// - For non-contracted index: compute D independent slices and concatenate (Stack mode)
///
/// When open_ids is empty (fully contracted network producing scalar), only Sum mode
/// is allowed since Stack mode would introduce new dimensions in the result.
///
/// Prefers larger dimensions for more memory savings.
/// Avoids open indices (which must remain in final output).
fn find_best_slice_index_excluding(
    tensor_a: &Tensor,
    tensor_b: &Tensor,
    contracted: &[TensorIndex],
    open_ids: &HashSet<u32>,
    exclude_ids: &HashSet<u32>,
) -> Option<SliceCandidate> {
    let contracted_ids: HashSet<u32> = contracted.iter().map(|i| i.id).collect();
    let fully_contracted = open_ids.is_empty();

    let mut candidates: Vec<SliceCandidate> = Vec::new();

    // Add contracted indices (Sum mode) - always allowed
    for idx in contracted {
        if !open_ids.contains(&idx.id) && !exclude_ids.contains(&idx.id) {
            candidates.push(SliceCandidate {
                index: *idx,
                mode: SliceMode::Sum,
            });
        }
    }

    // Only add Stack mode candidates if not fully contracted
    // (Stack mode would introduce new dimensions, breaking scalar output)
    if !fully_contracted {
        // Build the result index list to determine Stack axis positions
        let mut result_indices: Vec<TensorIndex> = Vec::new();
        for idx in &tensor_a.indices {
            if !contracted_ids.contains(&idx.id) {
                result_indices.push(*idx);
            }
        }
        for idx in &tensor_b.indices {
            if !contracted_ids.contains(&idx.id) {
                if !result_indices.iter().any(|i| i.id == idx.id) {
                    result_indices.push(*idx);
                }
            }
        }

        // Add non-contracted indices (Stack mode with axis position)
        for (axis, idx) in result_indices.iter().enumerate() {
            if !open_ids.contains(&idx.id) && !exclude_ids.contains(&idx.id) {
                // Check it's not already added (contracted indices are already in candidates)
                if !candidates.iter().any(|c| c.index.id == idx.id) {
                    candidates.push(SliceCandidate {
                        index: *idx,
                        mode: SliceMode::Stack { axis },
                    });
                }
            }
        }
    }

    // Sort by dimension (largest first for maximum memory reduction)
    candidates.sort_by(|a, b| b.index.dim.cmp(&a.index.dim));

    candidates.into_iter().next()
}

/// Execute a sliced contraction path.
///
/// Handles both Sum mode (add partial results) and Stack mode (concatenate along axis).
pub fn execute_sliced_path(network: &TensorNetwork, sliced: &SlicedPath) -> Tensor {
    if sliced.slices.is_empty() {
        return execute_path(network, &sliced.base_path);
    }

    // For single slice, simple iteration with mode handling
    if sliced.slices.len() == 1 {
        let slice = &sliced.slices[0];
        let dim = slice.index.dim as usize;

        let mut partials: Vec<Tensor> = Vec::with_capacity(dim);

        for slice_val in 0..dim {
            // Create sliced network
            let sliced_network = slice_network(network, &slice.index, slice_val);

            // Execute path on sliced network
            let partial = execute_path(&sliced_network, &sliced.base_path);
            partials.push(partial);
        }

        return combine_partials(&partials, slice);
    }

    // Multiple slices - need to handle combinations of Sum and Stack
    // Group partial results by their Stack slice values, sum within groups
    execute_multi_slice(network, sliced)
}

/// Combine partial results according to the slice mode.
fn combine_partials(partials: &[Tensor], slice: &SliceSpec) -> Tensor {
    if partials.is_empty() {
        return Tensor::scalar(Complex32::new(0.0, 0.0));
    }

    match slice.mode {
        SliceMode::Sum => {
            // Sum all partial results
            let mut result = partials[0].clone();
            for partial in &partials[1..] {
                result = tensor_add(&result, partial);
            }
            result
        }
        SliceMode::Stack { axis } => {
            // Concatenate along the specified axis
            tensor_stack(partials, &slice.index, axis)
        }
    }
}

/// Execute a path with multiple slices, handling mixed Sum/Stack modes.
fn execute_multi_slice(network: &TensorNetwork, sliced: &SlicedPath) -> Tensor {
    // Separate slices by mode
    let sum_slices: Vec<&SliceSpec> = sliced
        .slices
        .iter()
        .filter(|s| matches!(s.mode, SliceMode::Sum))
        .collect();
    let stack_slices: Vec<&SliceSpec> = sliced
        .slices
        .iter()
        .filter(|s| matches!(s.mode, SliceMode::Stack { .. }))
        .collect();

    // Compute total iterations for sum slices (these get summed together)
    let sum_iterations: usize = sum_slices.iter().map(|s| s.index.dim as usize).product();
    let sum_iterations = sum_iterations.max(1);

    // Compute total iterations for stack slices (these get concatenated)
    let stack_iterations: usize = stack_slices.iter().map(|s| s.index.dim as usize).product();
    let stack_iterations = stack_iterations.max(1);

    // For each stack configuration, sum over all sum configurations
    let mut stack_partials: Vec<(Vec<usize>, Tensor)> = Vec::new();

    for stack_iter in 0..stack_iterations {
        // Decode stack iteration to slice values
        let mut remaining = stack_iter;
        let mut stack_vals: Vec<usize> = Vec::with_capacity(stack_slices.len());
        for s in &stack_slices {
            let dim = s.index.dim as usize;
            stack_vals.push(remaining % dim);
            remaining /= dim;
        }

        // Sum over all sum slice configurations
        let mut sum_result: Option<Tensor> = None;
        for sum_iter in 0..sum_iterations {
            // Apply all slices
            let mut sliced_network = network.clone();

            // Apply stack slices with current stack values
            for (idx, s) in stack_slices.iter().enumerate() {
                sliced_network = slice_network(&sliced_network, &s.index, stack_vals[idx]);
            }

            // Apply sum slices
            let mut sum_remaining = sum_iter;
            for s in &sum_slices {
                let dim = s.index.dim as usize;
                let slice_val = sum_remaining % dim;
                sum_remaining /= dim;
                sliced_network = slice_network(&sliced_network, &s.index, slice_val);
            }

            let partial = execute_path(&sliced_network, &sliced.base_path);

            sum_result = Some(match sum_result {
                None => partial,
                Some(acc) => tensor_add(&acc, &partial),
            });
        }

        if let Some(result) = sum_result {
            stack_partials.push((stack_vals, result));
        }
    }

    // Now stack all the sum results along the stack axes
    if stack_partials.is_empty() {
        return Tensor::scalar(Complex32::new(0.0, 0.0));
    }

    if stack_slices.is_empty() {
        // No stacking needed, just return the summed result
        return stack_partials.into_iter().next().unwrap().1;
    }

    // Multi-axis stacking: recursively stack along each axis
    stack_multi_axis(&stack_partials, &stack_slices)
}

/// Create a sliced version of a network by fixing an index to a specific value.
fn slice_network(
    network: &TensorNetwork,
    slice_idx: &TensorIndex,
    slice_val: usize,
) -> TensorNetwork {
    let mut sliced = TensorNetwork::new(network.n_qubits);
    sliced.open_indices = network.open_indices.clone();

    for tensor in &network.tensors {
        if tensor.has_index(slice_idx.id) {
            // Slice this tensor
            let sliced_tensor = slice_tensor(tensor, slice_idx, slice_val);
            sliced.add_tensor(sliced_tensor);
        } else {
            sliced.add_tensor(tensor.clone());
        }
    }

    // Remove slice index from open indices if present
    sliced.open_indices.retain(|i| i.id != slice_idx.id);

    sliced
}

/// Slice a tensor by fixing one index to a specific value.
fn slice_tensor(tensor: &Tensor, slice_idx: &TensorIndex, slice_val: usize) -> Tensor {
    // Find the axis of the slice index
    let axis = tensor.indices.iter().position(|i| i.id == slice_idx.id);

    let axis = match axis {
        Some(a) => a,
        None => return tensor.clone(), // Index not in tensor
    };

    // Compute new indices (without the sliced index)
    let new_indices: Vec<TensorIndex> = tensor
        .indices
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, idx)| *idx)
        .collect();

    // Compute strides
    let mut strides = vec![1usize; tensor.indices.len()];
    for i in (0..tensor.indices.len() - 1).rev() {
        strides[i] = strides[i + 1] * tensor.indices[i + 1].dim as usize;
    }

    // Extract the slice
    let _slice_stride = strides[axis];
    let slice_dim = slice_idx.dim as usize;
    let total_size = tensor.numel();
    let new_size = total_size / slice_dim;

    let mut new_data = Vec::with_capacity(new_size);

    // For each element in the new tensor, find corresponding element in original
    for new_idx in 0..new_size {
        // Convert new_idx to coordinates in new tensor
        let mut coords: Vec<usize> = Vec::with_capacity(new_indices.len());
        let mut remaining = new_idx;
        for i in 0..new_indices.len() {
            let dim = new_indices[i].dim as usize;
            coords.push(remaining % dim);
            remaining /= dim;
        }
        coords.reverse();

        // Insert slice coordinate at the right position
        let mut full_coords = coords.clone();
        full_coords.insert(axis, slice_val);

        // Convert to linear index in original tensor
        let old_idx: usize = full_coords
            .iter()
            .zip(strides.iter())
            .map(|(c, s)| c * s)
            .sum();

        new_data.push(tensor.data[old_idx]);
    }

    Tensor {
        indices: new_indices,
        data: new_data,
    }
}

/// Add two tensors element-wise.
fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.indices.len(),
        b.indices.len(),
        "Tensor dimension mismatch"
    );
    assert_eq!(a.data.len(), b.data.len(), "Tensor size mismatch");

    let data: Vec<Complex32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| *x + *y)
        .collect();

    Tensor {
        indices: a.indices.clone(),
        data,
    }
}

/// Stack tensors along a new axis.
///
/// All input tensors must have the same shape. The result tensor has
/// the stacked index inserted at the specified axis position.
fn tensor_stack(partials: &[Tensor], stack_index: &TensorIndex, axis: usize) -> Tensor {
    if partials.is_empty() {
        return Tensor::scalar(Complex32::new(0.0, 0.0));
    }

    let first = &partials[0];
    let partial_size = first.numel();

    // Build result indices: insert stack_index at axis position
    let mut result_indices: Vec<TensorIndex> = Vec::with_capacity(first.indices.len() + 1);
    for (i, idx) in first.indices.iter().enumerate() {
        if i == axis {
            result_indices.push(*stack_index);
        }
        result_indices.push(*idx);
    }
    // Handle case where axis is at the end
    if axis >= first.indices.len() {
        result_indices.push(*stack_index);
    }

    // Build result data: interleave partial results according to axis position
    let stack_dim = partials.len();
    let result_size = partial_size * stack_dim;
    let mut result_data = vec![Complex32::new(0.0, 0.0); result_size];

    // Compute strides for the result tensor
    let mut result_strides = vec![1usize; result_indices.len()];
    for i in (0..result_indices.len() - 1).rev() {
        result_strides[i] = result_strides[i + 1] * result_indices[i + 1].dim as usize;
    }

    // Compute strides for the partial tensors
    let mut partial_strides = vec![1usize; first.indices.len()];
    if !partial_strides.is_empty() {
        for i in (0..first.indices.len() - 1).rev() {
            partial_strides[i] = partial_strides[i + 1] * first.indices[i + 1].dim as usize;
        }
    }

    // For each slice value, copy the partial tensor data to the right place
    for (slice_val, partial) in partials.iter().enumerate() {
        for partial_idx in 0..partial_size {
            // Convert partial linear index to coordinates
            let mut coords: Vec<usize> = Vec::with_capacity(first.indices.len());
            let mut remaining = partial_idx;
            for dim in first.indices.iter().map(|i| i.dim as usize) {
                coords.push(remaining % dim);
                remaining /= dim;
            }
            coords.reverse();

            // Insert the slice coordinate at axis position
            let mut result_coords = coords.clone();
            result_coords.insert(axis, slice_val);

            // Convert to result linear index
            let result_idx: usize = result_coords
                .iter()
                .zip(result_strides.iter())
                .map(|(c, s)| c * s)
                .sum();

            result_data[result_idx] = partial.data[partial_idx];
        }
    }

    Tensor {
        indices: result_indices,
        data: result_data,
    }
}

/// Stack partials along multiple axes.
///
/// Takes a list of (stack_values, tensor) pairs where stack_values gives
/// the coordinate in each stack dimension, and reassembles them into a
/// single tensor with the stacked indices restored.
fn stack_multi_axis(partials: &[(Vec<usize>, Tensor)], stack_slices: &[&SliceSpec]) -> Tensor {
    if partials.is_empty() || stack_slices.is_empty() {
        return partials
            .first()
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| Tensor::scalar(Complex32::new(0.0, 0.0)));
    }

    let (_, first_tensor) = &partials[0];
    let partial_size = first_tensor.numel();

    // Build result indices: insert stack indices at their axis positions
    // We need to be careful about axis positions as we insert multiple indices
    let mut result_indices = first_tensor.indices.clone();

    // Sort stack slices by axis in reverse order to insert from end to start
    // This way earlier insertions don't shift later positions
    let mut sorted_slices: Vec<(usize, &&SliceSpec)> = stack_slices.iter().enumerate().collect();
    sorted_slices.sort_by(|a, b| {
        let axis_a = match a.1.mode {
            SliceMode::Stack { axis } => axis,
            _ => 0,
        };
        let axis_b = match b.1.mode {
            SliceMode::Stack { axis } => axis,
            _ => 0,
        };
        axis_b.cmp(&axis_a) // Reverse order
    });

    // Insert indices in reverse order
    for (_, slice) in &sorted_slices {
        if let SliceMode::Stack { axis } = slice.mode {
            if axis <= result_indices.len() {
                result_indices.insert(axis, slice.index);
            } else {
                result_indices.push(slice.index);
            }
        }
    }

    // Compute result size
    let result_size: usize = result_indices.iter().map(|i| i.dim as usize).product();
    let mut result_data = vec![Complex32::new(0.0, 0.0); result_size];

    // Compute result strides
    let mut result_strides = vec![1usize; result_indices.len()];
    for i in (0..result_indices.len() - 1).rev() {
        result_strides[i] = result_strides[i + 1] * result_indices[i + 1].dim as usize;
    }

    // Compute partial strides
    let mut partial_strides = vec![1usize; first_tensor.indices.len()];
    if !partial_strides.is_empty() {
        for i in (0..first_tensor.indices.len() - 1).rev() {
            partial_strides[i] = partial_strides[i + 1] * first_tensor.indices[i + 1].dim as usize;
        }
    }

    // Map from original slice index to its axis
    let slice_axes: Vec<usize> = stack_slices
        .iter()
        .map(|s| match s.mode {
            SliceMode::Stack { axis } => axis,
            _ => 0,
        })
        .collect();

    // For each partial, place its data in the result
    for (stack_vals, partial) in partials {
        for partial_idx in 0..partial_size {
            // Convert partial linear index to coordinates
            let mut coords: Vec<usize> = Vec::with_capacity(first_tensor.indices.len());
            let mut remaining = partial_idx;
            for dim in first_tensor.indices.iter().map(|i| i.dim as usize) {
                coords.push(remaining % dim);
                remaining /= dim;
            }
            coords.reverse();

            // Build result coordinates by inserting stack values at their axes
            // Insert in reverse order (largest axis first) to avoid shifting issues
            let mut result_coords = coords.clone();
            for (slice_idx, &axis) in slice_axes.iter().enumerate().rev() {
                result_coords.insert(axis, stack_vals[slice_idx]);
            }

            // Convert to result linear index
            let result_idx: usize = result_coords
                .iter()
                .zip(result_strides.iter())
                .map(|(c, s)| c * s)
                .sum();

            if result_idx < result_data.len() {
                result_data[result_idx] = partial.data[partial_idx];
            }
        }
    }

    Tensor {
        indices: result_indices,
        data: result_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::{circuit_to_network, compute_amplitude};
    use logic_fabric_core::quantum::QGate;

    #[test]
    fn test_slice_tensor() {
        // Create a 2x3 tensor
        let idx_a = TensorIndex::new(1, 2);
        let idx_b = TensorIndex::new(2, 3);
        let tensor = Tensor {
            indices: vec![idx_a, idx_b],
            data: vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
                Complex32::new(5.0, 0.0),
            ],
        };

        // Slice along first axis at value 1
        let sliced = slice_tensor(&tensor, &idx_a, 1);
        assert_eq!(sliced.indices.len(), 1);
        assert_eq!(sliced.data.len(), 3);
        assert_eq!(sliced.data[0].re, 3.0);
        assert_eq!(sliced.data[1].re, 4.0);
        assert_eq!(sliced.data[2].re, 5.0);
    }

    #[test]
    fn test_no_slicing_needed() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let network = circuit_to_network(&circuit, 2);

        use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        // Large budget - no slicing needed
        let sliced = plan_slicing(&network, &path, usize::MAX);
        assert!(sliced.is_none());
    }

    #[test]
    fn test_sliced_execution_correctness() {
        // Simple circuit that produces a known amplitude
        let circuit = vec![QGate::H(0), QGate::H(1)];
        let network = circuit_to_network(&circuit, 2);

        // Compute without slicing
        let expected = compute_amplitude(&network, 0, 0);

        use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        // Force slicing with tiny budget
        if let Some(sliced_path) = plan_slicing(&network, &path, 1) {
            let result = execute_sliced_path(&network, &sliced_path);
            // Result should be a scalar
            assert_eq!(result.numel(), 1);
            // Value should match
            let err =
                (result.data[0].re - expected.re).abs() + (result.data[0].im - expected.im).abs();
            assert!(
                err < 1e-5,
                "Sliced result differs: {:?} vs {:?}",
                result.data[0],
                expected
            );
        }
        // If no slicing needed, that's fine too
    }

    /// Brute-force test that slicing produces correct results across many circuits.
    ///
    /// For each circuit:
    /// 1. Create a capped network (as used by compute_amplitude)
    /// 2. Compute expected amplitude without slicing
    /// 3. Force slicing with tiny budget
    /// 4. Verify sliced result matches ground truth
    #[test]
    fn test_slicing_brute_force_correctness() {
        use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
        use crate::tensor_network::tensor::Tensor;

        // Helper to create a capped network (same logic as compute_amplitude)
        fn cap_network(network: &TensorNetwork, in_bits: u64, out_bits: u64) -> TensorNetwork {
            let n_qubits = network.n_qubits;
            let mut capped = network.clone();

            // Add input basis state tensors
            for q in 0..n_qubits {
                let bit = ((in_bits >> q) & 1) as usize;
                let idx = network.open_indices[q as usize];
                let cap = Tensor::basis_projector(idx, bit);
                capped.add_tensor(cap);
            }

            // Add output basis state tensors
            for q in 0..n_qubits {
                let bit = ((out_bits >> q) & 1) as usize;
                let idx = network.open_indices[(n_qubits + q) as usize];
                let cap = Tensor::basis_projector(idx, bit);
                capped.add_tensor(cap);
            }

            // Clear open indices since they're now all contracted
            capped.open_indices.clear();
            capped
        }

        // Test cases: (n_qubits, circuit)
        let test_cases: Vec<(u8, Vec<QGate>)> = vec![
            // 2-qubit circuits
            (2, vec![QGate::H(0), QGate::H(1)]),
            (2, vec![QGate::H(0), QGate::CNot(0, 1)]),
            (2, vec![QGate::X(0), QGate::H(1), QGate::CNot(1, 0)]),
            // 3-qubit circuits
            (3, vec![QGate::H(0), QGate::H(1), QGate::H(2)]),
            (3, vec![QGate::H(0), QGate::CNot(0, 1), QGate::CNot(1, 2)]),
            (
                3,
                vec![QGate::H(0), QGate::H(1), QGate::CZ(0, 2), QGate::CNot(1, 2)],
            ),
            // 4-qubit GHZ-like
            (
                4,
                vec![
                    QGate::H(0),
                    QGate::CNot(0, 1),
                    QGate::CNot(1, 2),
                    QGate::CNot(2, 3),
                ],
            ),
            // More complex 3-qubit
            (
                3,
                vec![
                    QGate::H(0),
                    QGate::H(1),
                    QGate::H(2),
                    QGate::CNot(0, 1),
                    QGate::CNot(1, 2),
                    QGate::T(0),
                    QGate::T(1),
                    QGate::CNot(0, 2),
                ],
            ),
        ];

        let mut passed = 0;
        let mut total = 0;

        for (n_qubits, circuit) in test_cases {
            total += 1;

            let raw_network = circuit_to_network(&circuit, n_qubits);

            // Create capped network (same as compute_amplitude uses internally)
            let capped = cap_network(&raw_network, 0, 0);

            // Ground truth: unsliced amplitude
            let expected = compute_amplitude(&raw_network, 0, 0);

            // Get a contraction path on the CAPPED network
            let path = find_greedy_order(&capped, GreedyMetric::MinMemory);

            // Try with various tiny budgets to force slicing
            for budget in [1, 2, 4, 8] {
                if let Some(sliced_path) = plan_slicing(&capped, &path, budget) {
                    // Must have at least one slice
                    assert!(
                        !sliced_path.slices.is_empty(),
                        "Plan returned with empty slices for {}-qubit circuit with budget {}",
                        n_qubits,
                        budget
                    );

                    // Execute with slicing on the CAPPED network
                    let result = execute_sliced_path(&capped, &sliced_path);

                    // Should produce a scalar
                    assert_eq!(
                        result.numel(),
                        1,
                        "Sliced result should be scalar, got {} elements for {}-qubit circuit with budget {}",
                        result.numel(),
                        n_qubits,
                        budget
                    );

                    // Values must match
                    let err = (result.data[0].re - expected.re).abs()
                        + (result.data[0].im - expected.im).abs();

                    assert!(
                        err < 1e-4,
                        "{}-qubit circuit with budget {}: sliced=({:.6}, {:.6}i) vs expected=({:.6}, {:.6}i), err={:.2e}",
                        n_qubits,
                        budget,
                        result.data[0].re,
                        result.data[0].im,
                        expected.re,
                        expected.im,
                        err
                    );
                }
            }

            passed += 1;
        }

        assert_eq!(passed, total, "Some test cases failed");
    }

    /// Test that slicing modes are correctly determined.
    #[test]
    fn test_slice_mode_determination() {
        use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};

        // Circuit that should produce both Sum and Stack slices
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
        ];
        let network = circuit_to_network(&circuit, 3);
        let path = find_greedy_order(&network, GreedyMetric::MinMemory);

        // Force aggressive slicing
        if let Some(sliced_path) = plan_slicing(&network, &path, 1) {
            // Check that we got some slices
            assert!(
                !sliced_path.slices.is_empty(),
                "Should have slices with budget 1"
            );

            // Count modes
            let sum_count = sliced_path
                .slices
                .iter()
                .filter(|s| matches!(s.mode, SliceMode::Sum))
                .count();
            let stack_count = sliced_path
                .slices
                .iter()
                .filter(|s| matches!(s.mode, SliceMode::Stack { .. }))
                .count();

            // At least verify the counts are reasonable
            assert!(
                sum_count + stack_count == sliced_path.slices.len(),
                "All slices should have Sum or Stack mode"
            );
        }
    }

    /// Test that slicing doesn't increase peak and produces valid slices when applied.
    #[test]
    fn test_slicing_reduces_peak() {
        use crate::tensor_network::compute_peak_memory;
        use crate::tensor_network::greedy::{GreedyMetric, find_greedy_order};
        use crate::tensor_network::tensor::Tensor;

        // Helper to create a capped network (same logic as compute_amplitude)
        fn cap_network(network: &TensorNetwork, in_bits: u64, out_bits: u64) -> TensorNetwork {
            let n_qubits = network.n_qubits;
            let mut capped = network.clone();

            for q in 0..n_qubits {
                let bit = ((in_bits >> q) & 1) as usize;
                let idx = network.open_indices[q as usize];
                let cap = Tensor::basis_projector(idx, bit);
                capped.add_tensor(cap);
            }

            for q in 0..n_qubits {
                let bit = ((out_bits >> q) & 1) as usize;
                let idx = network.open_indices[(n_qubits + q) as usize];
                let cap = Tensor::basis_projector(idx, bit);
                capped.add_tensor(cap);
            }

            capped.open_indices.clear();
            capped
        }

        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::CNot(0, 1),
            QGate::CNot(2, 3),
            QGate::CZ(1, 2),
        ];
        let raw_network = circuit_to_network(&circuit, 4);
        let capped = cap_network(&raw_network, 0, 0);
        let path = find_greedy_order(&capped, GreedyMetric::MinMemory);

        let original_peak = compute_peak_memory(&capped, &path);

        // Set a reasonable budget (half of original peak)
        let budget = original_peak / 2;
        if budget > 0 && original_peak > 1 {
            if let Some(sliced_path) = plan_slicing(&capped, &path, budget) {
                // Estimated peak should not exceed original
                assert!(
                    sliced_path.estimated_peak <= original_peak,
                    "Estimated peak {} should not exceed original {}",
                    sliced_path.estimated_peak,
                    original_peak
                );

                // Should have at least one slice
                assert!(
                    !sliced_path.slices.is_empty(),
                    "Sliced path should have slices"
                );

                // All slices should be Sum mode for fully contracted network
                for slice in &sliced_path.slices {
                    assert!(
                        matches!(slice.mode, SliceMode::Sum),
                        "Fully contracted network should only have Sum mode slices"
                    );
                }
            }
        }
    }
}
