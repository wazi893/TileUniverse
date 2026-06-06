// EPIC 85: Sparse Quantum State Representation
//
// For large qubit counts (20+), dense state vectors become impractical.
// Many quantum algorithms produce sparse states where most amplitudes are ~0.
//
// This module provides a sparse representation that:
// - Uses hash maps for O(1) amplitude lookup
// - Only stores non-zero amplitudes (above threshold)
// - Converts to/from dense QState for compatibility
// - Supports gate operations directly on sparse representation

use std::collections::HashMap;

/// Threshold below which amplitudes are considered zero
pub const SPARSE_THRESHOLD: f32 = 1e-10;

/// A complex amplitude (real, imaginary)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn norm_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn is_negligible(&self) -> bool {
        self.norm_sq() < SPARSE_THRESHOLD * SPARSE_THRESHOLD
    }

    #[inline]
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    pub fn scale(&self, s: f32) -> Complex {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline]
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Sparse quantum state representation
///
/// Only stores non-zero amplitudes in a hash map.
/// Basis state index (u64) -> Complex amplitude
#[derive(Clone, Debug)]
pub struct SparseQState {
    /// Number of qubits
    pub n_qubits: u8,
    /// Non-zero amplitudes: basis_index -> amplitude
    amplitudes: HashMap<u64, Complex>,
    /// Cached norm (invalidated on modification)
    cached_norm: Option<f64>,
}

impl SparseQState {
    /// Create a new sparse state initialized to |0...0⟩
    pub fn new_zero(n_qubits: u8) -> Self {
        assert!(n_qubits > 0 && n_qubits <= 64, "n_qubits must be 1-64");
        let mut amplitudes = HashMap::new();
        amplitudes.insert(0, Complex::ONE);
        Self {
            n_qubits,
            amplitudes,
            cached_norm: Some(1.0),
        }
    }

    /// Create from a single basis state |index⟩
    pub fn from_basis(n_qubits: u8, index: u64) -> Self {
        assert!(n_qubits > 0 && n_qubits <= 64);
        // For 64 qubits, all u64 indices are valid (0 to 2^64-1)
        // For <64 qubits, check that index fits within the state space
        if n_qubits < 64 {
            assert!(index < (1u64 << n_qubits), "Basis index out of range");
        }
        let mut amplitudes = HashMap::new();
        amplitudes.insert(index, Complex::ONE);
        Self {
            n_qubits,
            amplitudes,
            cached_norm: Some(1.0),
        }
    }

    /// Number of non-zero amplitudes (sparsity measure)
    pub fn nnz(&self) -> usize {
        self.amplitudes.len()
    }

    /// Total possible amplitudes (2^n_qubits)
    /// Returns u128 to handle 64 qubits (2^64 = 18.4 quintillion)
    pub fn dim(&self) -> u128 {
        1u128 << self.n_qubits
    }

    /// Sparsity ratio (nnz / dim)
    pub fn sparsity(&self) -> f64 {
        self.nnz() as f64 / self.dim() as f64
    }

    /// Get amplitude for a basis state (returns 0 if not stored)
    #[inline]
    pub fn get(&self, index: u64) -> Complex {
        self.amplitudes
            .get(&index)
            .copied()
            .unwrap_or(Complex::ZERO)
    }

    /// Set amplitude for a basis state
    #[inline]
    pub fn set(&mut self, index: u64, amp: Complex) {
        self.cached_norm = None;
        if amp.is_negligible() {
            self.amplitudes.remove(&index);
        } else {
            self.amplitudes.insert(index, amp);
        }
    }

    /// Compute the norm (sqrt of sum of |amplitude|^2)
    pub fn norm(&mut self) -> f64 {
        if let Some(n) = self.cached_norm {
            return n;
        }
        let norm_sq: f64 = self.amplitudes.values().map(|c| c.norm_sq() as f64).sum();
        let n = norm_sq.sqrt();
        self.cached_norm = Some(n);
        n
    }

    /// Normalize the state
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 0.0 && (n - 1.0).abs() > 1e-10 {
            let scale = 1.0 / n as f32;
            for amp in self.amplitudes.values_mut() {
                amp.re *= scale;
                amp.im *= scale;
            }
            self.cached_norm = Some(1.0);
        }
    }

    /// Prune negligible amplitudes
    pub fn prune(&mut self) {
        self.amplitudes.retain(|_, amp| !amp.is_negligible());
        self.cached_norm = None;
    }

    /// Iterator over non-zero amplitudes
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Complex)> {
        self.amplitudes.iter().map(|(&k, v)| (k, v))
    }

    // =========================================================================
    // Gate Operations (sparse-aware)
    // =========================================================================

    /// Apply X gate (bit flip) to qubit q
    pub fn apply_x(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");
        let mask = 1u64 << q;

        // X gate flips bit q: |...0...⟩ ↔ |...1...⟩
        // We need to swap amplitudes for pairs that differ only in bit q
        let old_amps: Vec<(u64, Complex)> = self.amplitudes.drain().collect();
        for (idx, amp) in old_amps {
            let new_idx = idx ^ mask;
            self.amplitudes.insert(new_idx, amp);
        }
        self.cached_norm = None;
    }

    /// Apply Z gate (phase flip) to qubit q
    pub fn apply_z(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");
        let mask = 1u64 << q;

        // Z gate: |0⟩ -> |0⟩, |1⟩ -> -|1⟩
        // Negate amplitudes where bit q is 1
        for (idx, amp) in self.amplitudes.iter_mut() {
            if (idx & mask) != 0 {
                amp.re = -amp.re;
                amp.im = -amp.im;
            }
        }
        // Norm unchanged
    }

    /// Apply Hadamard gate to qubit q
    ///
    /// H|0⟩ = (|0⟩ + |1⟩)/√2
    /// H|1⟩ = (|0⟩ - |1⟩)/√2
    ///
    /// This can increase sparsity (from 1 to 2 amplitudes per input)
    pub fn apply_h(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");
        let mask = 1u64 << q;
        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;

        // Collect all current amplitudes
        let old_amps: Vec<(u64, Complex)> = self.amplitudes.drain().collect();

        // For each amplitude, compute the two outputs
        for (idx, amp) in old_amps {
            let idx_0 = idx & !mask; // bit q = 0
            let idx_1 = idx | mask; // bit q = 1

            let is_one = (idx & mask) != 0;

            if !is_one {
                // Input had bit q = 0: H|0⟩ = (|0⟩ + |1⟩)/√2
                let scaled = amp.scale(sqrt2_inv);
                self.add_amplitude(idx_0, scaled);
                self.add_amplitude(idx_1, scaled);
            } else {
                // Input had bit q = 1: H|1⟩ = (|0⟩ - |1⟩)/√2
                let scaled_pos = amp.scale(sqrt2_inv);
                let scaled_neg = amp.scale(-sqrt2_inv);
                self.add_amplitude(idx_0, scaled_pos);
                self.add_amplitude(idx_1, scaled_neg);
            }
        }

        self.prune();
        self.cached_norm = None;
    }

    /// Apply CNOT gate (control, target)
    pub fn apply_cnot(&mut self, control: u8, target: u8) {
        assert!(control < self.n_qubits && target < self.n_qubits);
        assert!(control != target);

        let ctrl_mask = 1u64 << control;
        let tgt_mask = 1u64 << target;

        // CNOT flips target bit when control bit is 1
        let old_amps: Vec<(u64, Complex)> = self.amplitudes.drain().collect();
        for (idx, amp) in old_amps {
            let new_idx = if (idx & ctrl_mask) != 0 {
                idx ^ tgt_mask // Flip target
            } else {
                idx
            };
            self.amplitudes.insert(new_idx, amp);
        }
        self.cached_norm = None;
    }

    /// Helper: add amplitude to index (accumulating if exists)
    #[inline]
    fn add_amplitude(&mut self, idx: u64, amp: Complex) {
        if amp.is_negligible() {
            return;
        }
        self.amplitudes
            .entry(idx)
            .and_modify(|existing| *existing = existing.add(&amp))
            .or_insert(amp);
    }

    // =========================================================================
    // Conversion to/from dense QState
    // =========================================================================

    /// Convert to dense QState (for small qubit counts or verification)
    pub fn to_dense(&self) -> Option<crate::quantum::QState> {
        if self.n_qubits > 30 {
            return None; // Too large for dense
        }
        let mut state = crate::quantum::QState::new_zero(self.n_qubits);
        // Clear the default |0⟩ amplitude
        state.real.as_mut_slice()[0] = 0.0;

        for (&idx, amp) in &self.amplitudes {
            state.real.as_mut_slice()[idx as usize] = amp.re;
            state.imag.as_mut_slice()[idx as usize] = amp.im;
        }
        Some(state)
    }

    /// Create from dense QState (keeping only non-negligible amplitudes)
    pub fn from_dense(state: &crate::quantum::QState) -> Self {
        let mut sparse = Self {
            n_qubits: state.n_qubits,
            amplitudes: HashMap::new(),
            cached_norm: None,
        };

        for i in 0..state.len {
            let re = state.real.as_slice()[i];
            let im = state.imag.as_slice()[i];
            let amp = Complex::new(re, im);
            if !amp.is_negligible() {
                sparse.amplitudes.insert(i as u64, amp);
            }
        }

        sparse
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_zero_state() {
        let state = SparseQState::new_zero(10);
        assert_eq!(state.nnz(), 1);
        assert_eq!(state.get(0), Complex::ONE);
        assert_eq!(state.get(1), Complex::ZERO);
    }

    #[test]
    fn test_sparse_x_gate() {
        let mut state = SparseQState::new_zero(3);
        assert_eq!(state.get(0), Complex::ONE);

        state.apply_x(0);
        assert_eq!(state.get(0), Complex::ZERO);
        assert_eq!(state.get(1), Complex::ONE);

        state.apply_x(1);
        assert_eq!(state.get(1), Complex::ZERO);
        assert_eq!(state.get(3), Complex::ONE); // |011⟩
    }

    #[test]
    fn test_sparse_h_gate() {
        let mut state = SparseQState::new_zero(2);
        state.apply_h(0);

        // Should have |00⟩ and |01⟩ with equal amplitude
        assert_eq!(state.nnz(), 2);
        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
        assert!((state.get(0).re - sqrt2_inv).abs() < 1e-6);
        assert!((state.get(1).re - sqrt2_inv).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_h_preserves_norm() {
        let mut state = SparseQState::new_zero(5);
        for q in 0..5 {
            state.apply_h(q);
        }
        let norm = state.norm();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Norm should be 1.0, got {}",
            norm
        );
        assert_eq!(state.nnz(), 32); // 2^5 = 32 equal superposition
    }

    #[test]
    fn test_sparse_large_qubit_count() {
        // Test that we can create sparse states for large qubit counts
        let mut state = SparseQState::new_zero(50);
        assert_eq!(state.nnz(), 1);

        // Apply X to create |1...0⟩
        state.apply_x(49);
        assert_eq!(state.nnz(), 1);
        assert_eq!(state.get(1u64 << 49), Complex::ONE);
    }

    #[test]
    fn test_sparse_cnot() {
        // Create |10⟩
        let mut state = SparseQState::new_zero(2);
        state.apply_x(1);

        // CNOT(1, 0): |10⟩ -> |11⟩
        state.apply_cnot(1, 0);
        assert_eq!(state.get(0b11), Complex::ONE);
        assert_eq!(state.nnz(), 1);
    }
}
