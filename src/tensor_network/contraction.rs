//! Tensor contraction implementation.
//!
//! Provides both explicit-loop and matrix-multiplication based contraction.
//! The matmul approach is 10-100x faster for medium-to-large tensors.

use super::tensor::{Tensor, TensorIndex};
use logic_fabric_core::quantum::Complex32;
use std::collections::HashSet;

/// Block size for cache-efficient matrix multiplication.
const BLOCK_SIZE: usize = 32;

/// Filter contracted indices to only those that exist in both tensors.
///
/// This is needed for sliced execution where some indices may have been
/// removed by slicing before the contraction is performed.
fn filter_valid_contracted(a: &Tensor, b: &Tensor, contracted: &[TensorIndex]) -> Vec<TensorIndex> {
    contracted
        .iter()
        .filter(|idx| a.has_index(idx.id) && b.has_index(idx.id))
        .copied()
        .collect()
}

/// Contract two tensors over shared indices.
///
/// This is an explicit-loop implementation. For each configuration of
/// uncontracted indices, sum over contracted indices.
pub fn contract_tensors(a: &Tensor, b: &Tensor, contracted: &[TensorIndex]) -> Tensor {
    // Filter to only indices that exist in both tensors (handles sliced networks)
    let contracted = filter_valid_contracted(a, b, contracted);

    // Handle edge cases
    if contracted.is_empty() {
        return outer_product(a, b);
    }

    // Identify which indices are contracted vs kept
    let contracted_ids: HashSet<u32> = contracted.iter().map(|i| i.id).collect();

    let a_kept: Vec<TensorIndex> = a
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .copied()
        .collect();

    let b_kept: Vec<TensorIndex> = b
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .copied()
        .collect();

    // Result indices: a_kept then b_kept
    let result_indices: Vec<TensorIndex> = a_kept.iter().chain(b_kept.iter()).copied().collect();

    // Handle scalar result
    if result_indices.is_empty() {
        let sum = contract_to_scalar(a, b, &contracted);
        return Tensor::scalar(sum);
    }

    let mut result = Tensor::new(result_indices.clone());

    // Precompute index mappings for efficiency
    let a_contracted_positions: Vec<usize> = contracted
        .iter()
        .map(|idx| a.index_position(idx.id).unwrap())
        .collect();

    let b_contracted_positions: Vec<usize> = contracted
        .iter()
        .map(|idx| b.index_position(idx.id).unwrap())
        .collect();

    let a_kept_positions: Vec<usize> = a_kept
        .iter()
        .map(|idx| a.index_position(idx.id).unwrap())
        .collect();

    let b_kept_positions: Vec<usize> = b_kept
        .iter()
        .map(|idx| b.index_position(idx.id).unwrap())
        .collect();

    // Contracted dimension
    let contracted_size: usize = contracted.iter().map(|i| i.dim as usize).product();

    // Iterate over all result configurations
    let result_size = result.numel();

    for result_flat in 0..result_size {
        let result_multi = result.unflatten_index(result_flat);

        // Split result_multi into a_kept and b_kept parts
        let a_kept_values: Vec<usize> = result_multi[..a_kept.len()].to_vec();
        let b_kept_values: Vec<usize> = result_multi[a_kept.len()..].to_vec();

        // Sum over contracted indices
        let mut sum = Complex32::new(0.0, 0.0);

        for contracted_flat in 0..contracted_size {
            let contracted_values = unflatten_contracted(contracted_flat, &contracted);

            // Build full multi-indices for a and b
            let a_multi = build_full_index(
                &a.indices,
                &a_kept_positions,
                &a_kept_values,
                &a_contracted_positions,
                &contracted_values,
            );

            let b_multi = build_full_index(
                &b.indices,
                &b_kept_positions,
                &b_kept_values,
                &b_contracted_positions,
                &contracted_values,
            );

            let a_val = a.get(&a_multi);
            let b_val = b.get(&b_multi);
            sum = sum.add(a_val.mul(b_val));
        }

        result.data[result_flat] = sum;
    }

    result
}

/// Contract two tensors to a scalar (all indices contracted).
fn contract_to_scalar(a: &Tensor, b: &Tensor, contracted: &[TensorIndex]) -> Complex32 {
    let contracted_size: usize = contracted.iter().map(|i| i.dim as usize).product();

    let a_positions: Vec<usize> = contracted
        .iter()
        .map(|idx| a.index_position(idx.id).unwrap())
        .collect();

    let b_positions: Vec<usize> = contracted
        .iter()
        .map(|idx| b.index_position(idx.id).unwrap())
        .collect();

    let mut sum = Complex32::new(0.0, 0.0);

    for flat in 0..contracted_size {
        let values = unflatten_contracted(flat, contracted);

        let a_multi = build_contracted_index(&a.indices, &a_positions, &values);
        let b_multi = build_contracted_index(&b.indices, &b_positions, &values);

        let a_val = a.get(&a_multi);
        let b_val = b.get(&b_multi);
        sum = sum.add(a_val.mul(b_val));
    }

    sum
}

/// Compute outer product of two tensors (no shared indices).
fn outer_product(a: &Tensor, b: &Tensor) -> Tensor {
    let result_indices: Vec<TensorIndex> =
        a.indices.iter().chain(b.indices.iter()).copied().collect();

    let mut result = Tensor::new(result_indices);

    for ai in 0..a.numel() {
        for bi in 0..b.numel() {
            let ri = ai * b.numel() + bi;
            result.data[ri] = a.data[ai].mul(b.data[bi]);
        }
    }

    result
}

/// Unflatten a contracted index configuration.
fn unflatten_contracted(flat: usize, indices: &[TensorIndex]) -> Vec<usize> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut result = vec![0; indices.len()];
    let mut remaining = flat;

    for i in (0..indices.len()).rev() {
        let dim = indices[i].dim as usize;
        result[i] = remaining % dim;
        remaining /= dim;
    }

    result
}

/// Build a full multi-index for a tensor given kept and contracted values.
fn build_full_index(
    indices: &[TensorIndex],
    kept_positions: &[usize],
    kept_values: &[usize],
    contracted_positions: &[usize],
    contracted_values: &[usize],
) -> Vec<usize> {
    let mut result = vec![0; indices.len()];

    for (i, &pos) in kept_positions.iter().enumerate() {
        result[pos] = kept_values[i];
    }

    for (i, &pos) in contracted_positions.iter().enumerate() {
        result[pos] = contracted_values[i];
    }

    result
}

/// Build a multi-index with only contracted values.
fn build_contracted_index(
    indices: &[TensorIndex],
    positions: &[usize],
    values: &[usize],
) -> Vec<usize> {
    let mut result = vec![0; indices.len()];
    for (i, &pos) in positions.iter().enumerate() {
        result[pos] = values[i];
    }
    result
}

// ============================================================================
// Matrix multiplication based contraction
// ============================================================================

/// Contract two tensors using matrix multiplication.
///
/// This is significantly faster than explicit loops for medium-to-large tensors.
/// The approach:
/// 1. Permute A so contracted indices are at the end
/// 2. Permute B so contracted indices are at the beginning
/// 3. Reshape both to 2D matrices
/// 4. Perform matrix multiplication
/// 5. Reshape result back to tensor
pub fn contract_as_matmul(a: &Tensor, b: &Tensor, contracted: &[TensorIndex]) -> Tensor {
    // Filter to only indices that exist in both tensors (handles sliced networks)
    let contracted = filter_valid_contracted(a, b, contracted);

    // Handle edge cases
    if contracted.is_empty() {
        return outer_product(a, b);
    }

    let contracted_ids: HashSet<u32> = contracted.iter().map(|i| i.id).collect();

    // Identify kept indices
    let a_kept: Vec<TensorIndex> = a
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .copied()
        .collect();

    let b_kept: Vec<TensorIndex> = b
        .indices
        .iter()
        .filter(|i| !contracted_ids.contains(&i.id))
        .copied()
        .collect();

    // Handle scalar result
    if a_kept.is_empty() && b_kept.is_empty() {
        let sum = contract_to_scalar(a, b, &contracted);
        return Tensor::scalar(sum);
    }

    // Compute dimensions
    let a_kept_dim: usize = a_kept
        .iter()
        .map(|i| i.dim as usize)
        .product::<usize>()
        .max(1);
    let b_kept_dim: usize = b_kept
        .iter()
        .map(|i| i.dim as usize)
        .product::<usize>()
        .max(1);
    let contracted_dim: usize = contracted.iter().map(|i| i.dim as usize).product();

    // Build permutation for A: [kept..., contracted...]
    let mut a_perm: Vec<usize> = Vec::with_capacity(a.indices.len());
    for idx in &a_kept {
        a_perm.push(a.index_position(idx.id).unwrap());
    }
    for idx in &contracted {
        a_perm.push(a.index_position(idx.id).unwrap());
    }

    // Build permutation for B: [contracted..., kept...]
    let mut b_perm: Vec<usize> = Vec::with_capacity(b.indices.len());
    for idx in &contracted {
        b_perm.push(b.index_position(idx.id).unwrap());
    }
    for idx in &b_kept {
        b_perm.push(b.index_position(idx.id).unwrap());
    }

    // Permute tensors
    let a_permuted = if is_identity_perm(&a_perm) {
        a.clone()
    } else {
        a.permute(&a_perm)
    };

    let b_permuted = if is_identity_perm(&b_perm) {
        b.clone()
    } else {
        b.permute(&b_perm)
    };

    // Matrix multiply: A is (a_kept_dim, contracted_dim), B is (contracted_dim, b_kept_dim)
    let c_data = complex_matmul_blocked(
        &a_permuted.data,
        &b_permuted.data,
        a_kept_dim,
        contracted_dim,
        b_kept_dim,
    );

    // Build result tensor
    let result_indices: Vec<TensorIndex> = a_kept.iter().chain(b_kept.iter()).copied().collect();

    // Handle case where result is scalar
    if result_indices.is_empty() {
        return Tensor::scalar(c_data[0]);
    }

    Tensor {
        indices: result_indices,
        data: c_data,
    }
}

/// Check if permutation is identity (no-op).
fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

/// Cache-blocked complex matrix multiplication.
///
/// Computes C[m,n] = sum_k A[m,k] * B[k,n]
/// Uses blocking for better cache utilization.
fn complex_matmul_blocked(
    a: &[Complex32],
    b: &[Complex32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); m * n];

    // For small matrices, use simple algorithm
    if m * k * n < 1024 {
        return complex_matmul_simple(a, b, m, k, n);
    }

    // Blocked matrix multiplication
    for i0 in (0..m).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(m);

        for j0 in (0..n).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(n);

            for p0 in (0..k).step_by(BLOCK_SIZE) {
                let p_end = (p0 + BLOCK_SIZE).min(k);

                // Process block
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_ip = a[i * k + p];

                        for j in j0..j_end {
                            let b_pj = b[p * n + j];
                            let idx = i * n + j;
                            c[idx] = c[idx].add(a_ip.mul(b_pj));
                        }
                    }
                }
            }
        }
    }

    c
}

/// Simple matrix multiplication (no blocking).
fn complex_matmul_simple(
    a: &[Complex32],
    b: &[Complex32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); m * n];

    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                let b_pj = b[p * n + j];
                let idx = i * n + j;
                c[idx] = c[idx].add(a_ip.mul(b_pj));
            }
        }
    }

    c
}

/// Smart contraction dispatcher: chooses between matmul and explicit loops.
///
/// Uses matmul for larger tensors where it provides significant speedup.
pub fn contract_tensors_smart(a: &Tensor, b: &Tensor, contracted: &[TensorIndex]) -> Tensor {
    // For small tensors, explicit loops have less overhead
    // For larger tensors, matmul's cache efficiency wins
    let total_size = a.numel() * b.numel();

    if total_size < 64 {
        contract_tensors(a, b, contracted)
    } else {
        contract_as_matmul(a, b, contracted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outer_product() {
        let idx_a = TensorIndex::qubit(0);
        let idx_b = TensorIndex::qubit(1);

        let mut a = Tensor::new(vec![idx_a]);
        a.data[0] = Complex32::new(1.0, 0.0);
        a.data[1] = Complex32::new(2.0, 0.0);

        let mut b = Tensor::new(vec![idx_b]);
        b.data[0] = Complex32::new(3.0, 0.0);
        b.data[1] = Complex32::new(4.0, 0.0);

        let result = outer_product(&a, &b);
        assert_eq!(result.rank(), 2);
        assert_eq!(result.numel(), 4);
        assert_eq!(result.data[0].re, 3.0); // 1 * 3
        assert_eq!(result.data[1].re, 4.0); // 1 * 4
        assert_eq!(result.data[2].re, 6.0); // 2 * 3
        assert_eq!(result.data[3].re, 8.0); // 2 * 4
    }

    #[test]
    fn test_contract_to_scalar() {
        // Create two vectors and contract them (dot product)
        let idx = TensorIndex::qubit(0);

        let mut a = Tensor::new(vec![idx]);
        a.data[0] = Complex32::new(1.0, 0.0);
        a.data[1] = Complex32::new(2.0, 0.0);

        let mut b = Tensor::new(vec![idx]);
        b.data[0] = Complex32::new(3.0, 0.0);
        b.data[1] = Complex32::new(4.0, 0.0);

        let result = contract_tensors(&a, &b, &[idx]);
        assert!(result.is_scalar());
        assert_eq!(result.data[0].re, 11.0); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_matrix_multiplication() {
        // Contract two 2x2 matrices: C[i,k] = sum_j A[i,j] * B[j,k]
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);
        let k = TensorIndex::qubit(2);

        // A = [[1, 2], [3, 4]]
        let mut a = Tensor::new(vec![i, j]);
        a.data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];

        // B = [[5, 6], [7, 8]]
        let mut b = Tensor::new(vec![j, k]);
        b.data = vec![
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ];

        let result = contract_tensors(&a, &b, &[j]);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(result.rank(), 2);
        assert_eq!(result.data[0].re, 19.0);
        assert_eq!(result.data[1].re, 22.0);
        assert_eq!(result.data[2].re, 43.0);
        assert_eq!(result.data[3].re, 50.0);
    }

    #[test]
    fn test_matmul_contraction_simple() {
        // Same as test_matrix_multiplication but using matmul approach
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);
        let k = TensorIndex::qubit(2);

        // A = [[1, 2], [3, 4]]
        let mut a = Tensor::new(vec![i, j]);
        a.data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];

        // B = [[5, 6], [7, 8]]
        let mut b = Tensor::new(vec![j, k]);
        b.data = vec![
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ];

        let result = contract_as_matmul(&a, &b, &[j]);

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result.rank(), 2);
        assert!((result.data[0].re - 19.0).abs() < 1e-6);
        assert!((result.data[1].re - 22.0).abs() < 1e-6);
        assert!((result.data[2].re - 43.0).abs() < 1e-6);
        assert!((result.data[3].re - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_parity_with_explicit() {
        // Verify matmul gives same result as explicit loops
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);
        let k = TensorIndex::qubit(2);
        let l = TensorIndex::qubit(3);

        // Create tensors with some complex values
        let mut a = Tensor::new(vec![i, j, k]);
        for idx in 0..8 {
            a.data[idx] = Complex32::new(idx as f32, (idx as f32) * 0.5);
        }

        let mut b = Tensor::new(vec![j, k, l]);
        for idx in 0..8 {
            b.data[idx] = Complex32::new((7 - idx) as f32, (idx as f32) * 0.3);
        }

        // Contract over j and k
        let explicit_result = contract_tensors(&a, &b, &[j, k]);
        let matmul_result = contract_as_matmul(&a, &b, &[j, k]);

        // Results should match
        assert_eq!(explicit_result.rank(), matmul_result.rank());
        assert_eq!(explicit_result.numel(), matmul_result.numel());

        for idx in 0..explicit_result.numel() {
            let err_re = (explicit_result.data[idx].re - matmul_result.data[idx].re).abs();
            let err_im = (explicit_result.data[idx].im - matmul_result.data[idx].im).abs();
            assert!(
                err_re < 1e-5 && err_im < 1e-5,
                "Mismatch at {}: explicit=({}, {}i) matmul=({}, {}i)",
                idx,
                explicit_result.data[idx].re,
                explicit_result.data[idx].im,
                matmul_result.data[idx].re,
                matmul_result.data[idx].im
            );
        }
    }

    #[test]
    fn test_matmul_outer_product() {
        // Test outer product via matmul
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);

        let mut a = Tensor::new(vec![i]);
        a.data = vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)];

        let mut b = Tensor::new(vec![j]);
        b.data = vec![Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)];

        let result = contract_as_matmul(&a, &b, &[]);

        assert_eq!(result.rank(), 2);
        assert_eq!(result.data[0].re, 3.0);
        assert_eq!(result.data[1].re, 4.0);
        assert_eq!(result.data[2].re, 6.0);
        assert_eq!(result.data[3].re, 8.0);
    }

    #[test]
    fn test_matmul_to_scalar() {
        // Contract to scalar
        let idx = TensorIndex::qubit(0);

        let mut a = Tensor::new(vec![idx]);
        a.data = vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, 0.0)];

        let mut b = Tensor::new(vec![idx]);
        b.data = vec![Complex32::new(3.0, 0.0), Complex32::new(4.0, -1.0)];

        let result = contract_as_matmul(&a, &b, &[idx]);

        assert!(result.is_scalar());
        // (1+i)*3 + 2*(4-i) = 3+3i + 8-2i = 11+i
        assert!((result.data[0].re - 11.0).abs() < 1e-6);
        assert!((result.data[0].im - 1.0).abs() < 1e-6);
    }
}
