//! Core tensor types for tensor network simulation.
//!
//! Tensors are multi-dimensional arrays with labeled indices. When two tensors
//! share an index, they can be contracted (summed over that index).

use logic_fabric_core::quantum::Complex32;
use std::f32::consts::FRAC_1_SQRT_2;

/// A unique identifier for a tensor index (edge in the network graph).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorIndex {
    /// Unique ID for this index
    pub id: u32,
    /// Dimension of this index (typically 2 for qubits)
    pub dim: u8,
}

impl TensorIndex {
    /// Create a qubit index (dimension 2).
    pub fn qubit(id: u32) -> Self {
        Self { id, dim: 2 }
    }

    /// Create an index with arbitrary dimension.
    pub fn new(id: u32, dim: u8) -> Self {
        Self { id, dim }
    }
}

/// A tensor with explicit index labels.
///
/// Storage is row-major with indices ordered by their position in `indices`.
/// For a 3-index tensor [i, j, k] with dims [2, 2, 2]:
///   data[i*4 + j*2 + k] = T_{ijk}
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Index labels in order (determines data layout)
    pub indices: Vec<TensorIndex>,
    /// Complex tensor data (row-major)
    pub data: Vec<Complex32>,
}

impl Tensor {
    /// Create a new tensor with given indices (initialized to zero).
    pub fn new(indices: Vec<TensorIndex>) -> Self {
        let size: usize = indices.iter().map(|idx| idx.dim as usize).product();
        let size = if size == 0 { 1 } else { size }; // Scalar tensor
        Self {
            indices,
            data: vec![Complex32::new(0.0, 0.0); size],
        }
    }

    /// Create a scalar tensor (rank 0) with given value.
    pub fn scalar(value: Complex32) -> Self {
        Self {
            indices: Vec::new(),
            data: vec![value],
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Rank (number of indices).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Check if this is a scalar (rank 0).
    pub fn is_scalar(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get element by multi-index.
    pub fn get(&self, multi_idx: &[usize]) -> Complex32 {
        let flat_idx = self.flatten_index(multi_idx);
        self.data[flat_idx]
    }

    /// Set element by multi-index.
    pub fn set(&mut self, multi_idx: &[usize], value: Complex32) {
        let flat_idx = self.flatten_index(multi_idx);
        self.data[flat_idx] = value;
    }

    /// Convert multi-index to flat index (row-major).
    pub fn flatten_index(&self, multi_idx: &[usize]) -> usize {
        if self.indices.is_empty() {
            return 0;
        }
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..self.indices.len()).rev() {
            flat += multi_idx[i] * stride;
            stride *= self.indices[i].dim as usize;
        }
        flat
    }

    /// Convert flat index to multi-index.
    pub fn unflatten_index(&self, flat: usize) -> Vec<usize> {
        if self.indices.is_empty() {
            return Vec::new();
        }
        let mut result = vec![0; self.indices.len()];
        let mut remaining = flat;
        for i in (0..self.indices.len()).rev() {
            let dim = self.indices[i].dim as usize;
            result[i] = remaining % dim;
            remaining /= dim;
        }
        result
    }

    /// Compute strides for each index.
    pub fn strides(&self) -> Vec<usize> {
        if self.indices.is_empty() {
            return Vec::new();
        }
        let mut strides = vec![1; self.indices.len()];
        for i in (0..self.indices.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.indices[i + 1].dim as usize;
        }
        strides
    }

    /// Find position of an index by ID.
    pub fn index_position(&self, id: u32) -> Option<usize> {
        self.indices.iter().position(|idx| idx.id == id)
    }

    /// Check if tensor contains a given index.
    pub fn has_index(&self, id: u32) -> bool {
        self.indices.iter().any(|idx| idx.id == id)
    }

    // ========================================================================
    // Gate tensor constructors
    // ========================================================================

    /// Create identity tensor (delta function) for a single qubit.
    pub fn identity_1q(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[0] = Complex32::new(1.0, 0.0); // |0><0|
        tensor.data[3] = Complex32::new(1.0, 0.0); // |1><1|
        tensor
    }

    /// Create Hadamard gate tensor.
    pub fn hadamard(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        let h = FRAC_1_SQRT_2;
        tensor.data[0] = Complex32::new(h, 0.0); // <0|H|0>
        tensor.data[1] = Complex32::new(h, 0.0); // <0|H|1>
        tensor.data[2] = Complex32::new(h, 0.0); // <1|H|0>
        tensor.data[3] = Complex32::new(-h, 0.0); // <1|H|1>
        tensor
    }

    /// Create Pauli-X gate tensor.
    pub fn pauli_x(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[1] = Complex32::new(1.0, 0.0); // <0|X|1>
        tensor.data[2] = Complex32::new(1.0, 0.0); // <1|X|0>
        tensor
    }

    /// Create Pauli-Y gate tensor.
    pub fn pauli_y(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[1] = Complex32::new(0.0, -1.0); // <0|Y|1> = -i
        tensor.data[2] = Complex32::new(0.0, 1.0); // <1|Y|0> = i
        tensor
    }

    /// Create Pauli-Z gate tensor.
    pub fn pauli_z(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[0] = Complex32::new(1.0, 0.0); // <0|Z|0>
        tensor.data[3] = Complex32::new(-1.0, 0.0); // <1|Z|1>
        tensor
    }

    /// Create T gate tensor (pi/8 rotation).
    pub fn t_gate(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[0] = Complex32::new(1.0, 0.0);
        // e^(i*pi/4) = cos(pi/4) + i*sin(pi/4) = (1+i)/sqrt(2)
        let phase = FRAC_1_SQRT_2;
        tensor.data[3] = Complex32::new(phase, phase);
        tensor
    }

    /// Create T-dagger gate tensor.
    pub fn tdg_gate(in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        tensor.data[0] = Complex32::new(1.0, 0.0);
        // e^(-i*pi/4) = (1-i)/sqrt(2)
        let phase = FRAC_1_SQRT_2;
        tensor.data[3] = Complex32::new(phase, -phase);
        tensor
    }

    /// Create Rx gate tensor.
    pub fn rx_gate(theta: f32, in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        let half = theta / 2.0;
        let c = half.cos();
        let s = half.sin();
        tensor.data[0] = Complex32::new(c, 0.0);
        tensor.data[1] = Complex32::new(0.0, -s);
        tensor.data[2] = Complex32::new(0.0, -s);
        tensor.data[3] = Complex32::new(c, 0.0);
        tensor
    }

    /// Create Ry gate tensor.
    pub fn ry_gate(theta: f32, in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        let half = theta / 2.0;
        let c = half.cos();
        let s = half.sin();
        tensor.data[0] = Complex32::new(c, 0.0);
        tensor.data[1] = Complex32::new(-s, 0.0);
        tensor.data[2] = Complex32::new(s, 0.0);
        tensor.data[3] = Complex32::new(c, 0.0);
        tensor
    }

    /// Create Rz gate tensor.
    pub fn rz_gate(theta: f32, in_idx: TensorIndex, out_idx: TensorIndex) -> Self {
        let mut tensor = Tensor::new(vec![out_idx, in_idx]);
        let half = theta / 2.0;
        // e^(-i*theta/2) and e^(i*theta/2)
        tensor.data[0] = Complex32::new(half.cos(), -half.sin());
        tensor.data[3] = Complex32::new(half.cos(), half.sin());
        tensor
    }

    /// Create CNOT gate tensor (rank 4).
    /// Convention: T[out_ctrl, out_tgt, in_ctrl, in_tgt]
    pub fn cnot(
        in_ctrl: TensorIndex,
        in_tgt: TensorIndex,
        out_ctrl: TensorIndex,
        out_tgt: TensorIndex,
    ) -> Self {
        let mut tensor = Tensor::new(vec![out_ctrl, out_tgt, in_ctrl, in_tgt]);
        // CNOT: |ab> -> |a, b XOR a>
        for in_c in 0..2 {
            for in_t in 0..2 {
                let out_c = in_c;
                let out_t = in_t ^ in_c;
                let idx = out_c * 8 + out_t * 4 + in_c * 2 + in_t;
                tensor.data[idx] = Complex32::new(1.0, 0.0);
            }
        }
        tensor
    }

    /// Create CZ gate tensor (rank 4).
    pub fn cz(
        in_ctrl: TensorIndex,
        in_tgt: TensorIndex,
        out_ctrl: TensorIndex,
        out_tgt: TensorIndex,
    ) -> Self {
        let mut tensor = Tensor::new(vec![out_ctrl, out_tgt, in_ctrl, in_tgt]);
        // CZ: diagonal, -1 when both qubits are |1>
        for in_c in 0..2 {
            for in_t in 0..2 {
                let idx = in_c * 8 + in_t * 4 + in_c * 2 + in_t;
                let phase = if in_c == 1 && in_t == 1 { -1.0 } else { 1.0 };
                tensor.data[idx] = Complex32::new(phase, 0.0);
            }
        }
        tensor
    }

    /// Create a basis state projector tensor (for capping open indices).
    /// Projects onto |bit> state.
    pub fn basis_projector(idx: TensorIndex, bit: usize) -> Self {
        let mut tensor = Tensor::new(vec![idx]);
        tensor.data[bit] = Complex32::new(1.0, 0.0);
        tensor
    }

    // ========================================================================
    // Tensor manipulation for matrix multiplication
    // ========================================================================

    /// Permute tensor indices according to the given order.
    ///
    /// `perm` specifies the new position of each original index.
    /// For example, `perm = [2, 0, 1]` moves index 0 to position 2,
    /// index 1 to position 0, and index 2 to position 1.
    pub fn permute(&self, perm: &[usize]) -> Self {
        if self.indices.is_empty() {
            return self.clone();
        }

        assert_eq!(
            perm.len(),
            self.indices.len(),
            "Permutation length mismatch"
        );

        // Compute inverse permutation for index reordering
        let mut inv_perm = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // New indices in permuted order
        let new_indices: Vec<TensorIndex> = perm.iter().map(|&p| self.indices[p]).collect();

        // Compute strides for old and new layouts
        let _old_strides = self.strides();
        let new_tensor = Tensor::new(new_indices.clone());
        let new_strides = new_tensor.strides();

        let mut result = Tensor::new(new_indices);

        // Iterate over all elements and copy with permuted indices
        for old_flat in 0..self.data.len() {
            let old_multi = self.unflatten_index(old_flat);

            // Permute multi-index
            let new_multi: Vec<usize> = perm.iter().map(|&p| old_multi[p]).collect();

            // Compute new flat index
            let new_flat = if new_strides.is_empty() {
                0
            } else {
                new_multi
                    .iter()
                    .zip(new_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum()
            };

            result.data[new_flat] = self.data[old_flat];
        }

        result
    }

    /// Reshape tensor to new index configuration.
    ///
    /// The total size must match. This is a zero-copy operation when possible.
    pub fn reshape(&self, new_indices: Vec<TensorIndex>) -> Self {
        let new_size: usize = new_indices.iter().map(|idx| idx.dim as usize).product();
        let new_size = if new_size == 0 { 1 } else { new_size };

        assert_eq!(
            self.data.len(),
            new_size,
            "Reshape size mismatch: {} vs {}",
            self.data.len(),
            new_size
        );

        Self {
            indices: new_indices,
            data: self.data.clone(),
        }
    }

    /// Create a tensor filled with zeros with given indices.
    pub fn zeros(indices: Vec<TensorIndex>) -> Self {
        Self::new(indices)
    }

    /// Get the shape (dimensions) of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.indices.iter().map(|idx| idx.dim as usize).collect()
    }

    /// Get the dimension at a specific axis.
    pub fn dim(&self, axis: usize) -> usize {
        self.indices[axis].dim as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let idx = TensorIndex::qubit(0);
        let tensor = Tensor::new(vec![idx, idx]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.rank(), 2);
    }

    #[test]
    fn test_scalar_tensor() {
        let tensor = Tensor::scalar(Complex32::new(1.0, 2.0));
        assert!(tensor.is_scalar());
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.rank(), 0);
    }

    #[test]
    fn test_flatten_unflatten() {
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);
        let k = TensorIndex::qubit(2);
        let tensor = Tensor::new(vec![i, j, k]);

        for flat in 0..8 {
            let multi = tensor.unflatten_index(flat);
            let back = tensor.flatten_index(&multi);
            assert_eq!(flat, back);
        }
    }

    #[test]
    fn test_hadamard_tensor() {
        let in_idx = TensorIndex::qubit(0);
        let out_idx = TensorIndex::qubit(1);
        let h = Tensor::hadamard(in_idx, out_idx);

        let expected = FRAC_1_SQRT_2;
        assert!((h.get(&[0, 0]).re - expected).abs() < 1e-6);
        assert!((h.get(&[0, 1]).re - expected).abs() < 1e-6);
        assert!((h.get(&[1, 0]).re - expected).abs() < 1e-6);
        assert!((h.get(&[1, 1]).re - (-expected)).abs() < 1e-6);
    }

    #[test]
    fn test_cnot_tensor() {
        let in_c = TensorIndex::qubit(0);
        let in_t = TensorIndex::qubit(1);
        let out_c = TensorIndex::qubit(2);
        let out_t = TensorIndex::qubit(3);
        let cnot = Tensor::cnot(in_c, in_t, out_c, out_t);

        // |00> -> |00>
        assert_eq!(cnot.get(&[0, 0, 0, 0]).re, 1.0);
        // |01> -> |01>
        assert_eq!(cnot.get(&[0, 1, 0, 1]).re, 1.0);
        // |10> -> |11>
        assert_eq!(cnot.get(&[1, 1, 1, 0]).re, 1.0);
        // |11> -> |10>
        assert_eq!(cnot.get(&[1, 0, 1, 1]).re, 1.0);
    }

    #[test]
    fn test_permute_2d() {
        // 2x2 matrix transposition
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);

        let mut a = Tensor::new(vec![i, j]);
        // A = [[1, 2], [3, 4]]
        a.data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];

        // Transpose: swap i and j
        let transposed = a.permute(&[1, 0]);

        // A^T = [[1, 3], [2, 4]]
        assert_eq!(transposed.get(&[0, 0]).re, 1.0);
        assert_eq!(transposed.get(&[0, 1]).re, 3.0);
        assert_eq!(transposed.get(&[1, 0]).re, 2.0);
        assert_eq!(transposed.get(&[1, 1]).re, 4.0);
    }

    #[test]
    fn test_permute_3d() {
        // 3D tensor permutation
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);
        let k = TensorIndex::qubit(2);

        let mut t = Tensor::new(vec![i, j, k]);
        // Fill with unique values for verification
        for idx in 0..8 {
            t.data[idx] = Complex32::new(idx as f32, 0.0);
        }

        // Permute [0,1,2] -> [2,0,1]
        let permuted = t.permute(&[2, 0, 1]);

        // Verify a few elements
        // Original T[0,0,0] = 0 -> Permuted T[0,0,0] = 0
        assert_eq!(permuted.get(&[0, 0, 0]).re, 0.0);
        // Original T[0,0,1] = 1 -> Permuted T[1,0,0] (since k moves to position 0)
        assert_eq!(permuted.get(&[1, 0, 0]).re, 1.0);
        // Original T[0,1,0] = 2 -> Permuted T[0,0,1] (since j moves to position 2)
        assert_eq!(permuted.get(&[0, 0, 1]).re, 2.0);
    }

    #[test]
    fn test_reshape() {
        let i = TensorIndex::qubit(0);
        let j = TensorIndex::qubit(1);

        let mut a = Tensor::new(vec![i, j]);
        a.data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];

        // Reshape to single index with dim 4
        let flat_idx = TensorIndex::new(100, 4);
        let reshaped = a.reshape(vec![flat_idx]);

        assert_eq!(reshaped.rank(), 1);
        assert_eq!(reshaped.numel(), 4);
        assert_eq!(reshaped.data[0].re, 1.0);
        assert_eq!(reshaped.data[1].re, 2.0);
        assert_eq!(reshaped.data[2].re, 3.0);
        assert_eq!(reshaped.data[3].re, 4.0);
    }
}
