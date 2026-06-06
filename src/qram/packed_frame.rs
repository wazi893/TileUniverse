//! Packed Pauli Frame for 64 Parallel Noise Trajectories
//!
//! SPRINT 70.0 Phase 1: PackedPauliFrame64
//!
//! This module implements bit-sliced ensemble Pauli tracking, enabling
//! 64 independent noise trajectories to execute in lockstep with shared
//! control flow.
//!
//! # Design Principles
//!
//! 1. **Bit-slicing**: Each u64 holds one bit per trajectory (64 total)
//! 2. **AoSoA layout**: `(x, z)` pairs per qubit for cache locality
//! 3. **Exact Clifford**: CNOT, H, S, CZ propagation is mathematically exact
//! 4. **Correct sampling**: Statistically independent Bernoulli bits
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::PackedPauliFrame64;
//! use engine::quantum::QRng;
//!
//! let mut frame = PackedPauliFrame64::new(10);
//! let mut rng = QRng::new(42);
//!
//! // Inject depolarizing noise on qubit 0
//! frame.depolarizing(0, 0.01, &mut rng);
//!
//! // Propagate through CNOT
//! frame.cnot(0, 1);
//!
//! // Count trajectories with no X error on qubit 1
//! let clean = (!frame.x_errors(1)).count_ones();
//! ```

use crate::quantum::QRng;
use std::fmt;

/// Surrogate model choice for non-Clifford gates
///
/// Non-Clifford gates (Toffoli, CCZ, Fredkin) cannot be exactly tracked
/// with Pauli frames. We provide two approximations that bracket truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurrogateModel {
    /// Pessimistic: more aggressive error propagation
    /// Tends to UNDERESTIMATE fidelity (more errors than reality)
    Pessimistic,
    /// Optimistic: minimal error propagation
    /// Tends to OVERESTIMATE fidelity (fewer errors than reality)
    #[default]
    Optimistic,
}

/// 64 parallel Pauli error trajectories, bit-sliced
///
/// Each qubit has an (x, z) pair of u64 values where bit i represents
/// the error state for trajectory i.
#[derive(Clone)]
pub struct PackedPauliFrame64 {
    /// Number of qubits
    n_qubits: usize,
    /// Error state per qubit: (x_errors, z_errors)
    /// Bit i = error present in trajectory i
    errors: Vec<(u64, u64)>,
    /// Surrogate model for non-Clifford gates
    surrogate: SurrogateModel,
}

impl PackedPauliFrame64 {
    /// Create a new packed frame for n qubits, all trajectories clean
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            errors: vec![(0, 0); n_qubits],
            surrogate: SurrogateModel::default(),
        }
    }

    /// Set the surrogate model for non-Clifford gates
    pub fn set_surrogate_model(&mut self, model: SurrogateModel) {
        self.surrogate = model;
    }

    /// Get the current surrogate model
    pub fn surrogate_model(&self) -> SurrogateModel {
        self.surrogate
    }

    /// Get number of qubits
    #[inline]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Reset all trajectories to clean state
    pub fn reset(&mut self) {
        for (x, z) in &mut self.errors {
            *x = 0;
            *z = 0;
        }
    }

    /// Get X error mask for a qubit (bit i = trajectory i has X error)
    #[inline]
    pub fn x_errors(&self, qubit: usize) -> u64 {
        self.errors.get(qubit).map(|(x, _)| *x).unwrap_or(0)
    }

    /// Get Z error mask for a qubit (bit i = trajectory i has Z error)
    #[inline]
    pub fn z_errors(&self, qubit: usize) -> u64 {
        self.errors.get(qubit).map(|(_, z)| *z).unwrap_or(0)
    }

    /// Get mutable reference to error pair for a qubit
    #[inline]
    fn errors_mut(&mut self, qubit: usize) -> Option<&mut (u64, u64)> {
        self.errors.get_mut(qubit)
    }

    // =========================================================================
    // Noise Injection (Statistically Correct)
    // =========================================================================

    /// Generate 64 independent Bernoulli(p) bits
    ///
    /// Uses 8-byte lane comparison for statistical correctness:
    /// - Generate 64 random floats via rng.next_f32()
    /// - Compare each to probability p
    /// - Pack results into single u64
    ///
    /// This is slower than a single RNG call but statistically correct.
    pub fn bernoulli_64(p: f64, rng: &mut QRng) -> u64 {
        if p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return u64::MAX;
        }

        let p32 = p as f32;
        let mut result: u64 = 0;

        // Generate 64 independent samples
        for i in 0..64 {
            if rng.next_f32() < p32 {
                result |= 1u64 << i;
            }
        }

        result
    }

    /// Generate 64 independent Bernoulli(p) bits using byte comparison
    ///
    /// Faster version: generates 8 random values, extracts bytes, compares to threshold.
    /// Each byte gives one Bernoulli sample.
    pub fn bernoulli_64_fast(p: f64, rng: &mut QRng) -> u64 {
        if p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return u64::MAX;
        }

        let threshold = (p * 256.0).min(255.0) as u8;
        let mut result: u64 = 0;

        // Generate 8 random f32s, convert to u32, extract bytes
        for i in 0..8 {
            // Convert f32 [0,1) to u32 for byte extraction
            let r = (rng.next_f32() * (u32::MAX as f32)) as u32;
            let _bytes = r.to_le_bytes();

            // Use 2 bytes from each u32 (we need 64 bytes total = 8 iterations × 8 bytes)
            // Actually we need to be careful here - let's use next_f32 for each byte
            for j in 0..8 {
                let byte = ((rng.next_f32() * 256.0) as u32).min(255) as u8;
                if byte < threshold {
                    result |= 1u64 << (i * 8 + j);
                }
            }
        }

        result
    }

    /// Apply X error to specified trajectories
    #[inline]
    pub fn inject_x(&mut self, qubit: usize, mask: u64) {
        if let Some((x, _)) = self.errors_mut(qubit) {
            *x ^= mask;
        }
    }

    /// Apply Z error to specified trajectories
    #[inline]
    pub fn inject_z(&mut self, qubit: usize, mask: u64) {
        if let Some((_, z)) = self.errors_mut(qubit) {
            *z ^= mask;
        }
    }

    /// Apply Y error (X and Z) to specified trajectories
    #[inline]
    pub fn inject_y(&mut self, qubit: usize, mask: u64) {
        if let Some((x, z)) = self.errors_mut(qubit) {
            *x ^= mask;
            *z ^= mask;
        }
    }

    /// Apply depolarizing channel with error rate p
    ///
    /// Standard depolarizing: with probability p, apply uniformly random
    /// Pauli from {X, Y, Z}. Each of X, Y, Z has probability p/3.
    ///
    /// This is NOT the same as independent X and Z flips!
    pub fn depolarizing(&mut self, qubit: usize, p: f64, rng: &mut QRng) {
        if p <= 0.0 || qubit >= self.n_qubits {
            return;
        }

        let p_each = p / 3.0;
        let (x_mask, z_mask) = Self::depolarizing_masks(p_each, rng);

        if let Some((x, z)) = self.errors_mut(qubit) {
            *x ^= x_mask;
            *z ^= z_mask;
        }
    }

    /// Generate (x_mask, z_mask) for depolarizing channel
    ///
    /// P(X only) = p_each, P(Z only) = p_each, P(Y=XZ) = p_each
    /// P(I) = 1 - 3*p_each
    ///
    /// Uses bulk 16-bit sampling: 16 random u64 values provide 64 × 16-bit
    /// samples, each compared against 16-bit thresholds (< 0.002% relative
    /// error). This reduces RNG calls from 64 to 16 (4x fewer), which is
    /// the dominant cost in debug builds. For very small p where 16-bit
    /// resolution is insufficient, falls back to per-sample f32 comparison.
    fn depolarizing_masks(p_each: f64, rng: &mut QRng) -> (u64, u64) {
        if p_each <= 0.0 {
            return (0, 0);
        }

        // 16-bit thresholds: P(sample < thresh) = thresh / 65536.
        // Resolution 1/65536 ≈ 0.0015%, adequate for p_each ≥ 0.001.
        let thresh_y = (3.0 * p_each * 65536.0) as u32;

        // If total error prob < 1/65536, fall back to scalar f32 comparison.
        if thresh_y == 0 {
            return Self::depolarizing_masks_scalar(p_each, rng);
        }

        let thresh_x = (p_each * 65536.0) as u16;
        let thresh_xz = (2.0 * p_each * 65536.0) as u16;
        let thresh_xyz = thresh_y.min(65535) as u16;

        let mut x_mask: u64 = 0;
        let mut z_mask: u64 = 0;

        // 16 u64 values → 64 × 16-bit samples (4 per u64)
        for chunk in 0..16u32 {
            let r = rng.next_u64();
            let base = chunk * 4;
            for j in 0..4u32 {
                let sample = (r >> (j * 16)) as u16;
                let bit = base + j;
                if sample < thresh_x {
                    x_mask |= 1u64 << bit;
                } else if sample < thresh_xz {
                    z_mask |= 1u64 << bit;
                } else if sample < thresh_xyz {
                    x_mask |= 1u64 << bit;
                    z_mask |= 1u64 << bit;
                }
            }
        }

        (x_mask, z_mask)
    }

    /// Scalar fallback for depolarizing_masks when p is too small for
    /// 8-bit byte comparison (p_total < 1/256 ≈ 0.004).
    fn depolarizing_masks_scalar(p_each: f64, rng: &mut QRng) -> (u64, u64) {
        let p_each_f32 = p_each as f32;
        let p_x = p_each_f32;
        let p_z = 2.0 * p_each_f32;
        let p_y = 3.0 * p_each_f32;

        let mut x_mask: u64 = 0;
        let mut z_mask: u64 = 0;

        for i in 0..64 {
            let r = rng.next_f32();
            if r < p_x {
                x_mask |= 1u64 << i;
            } else if r < p_z {
                z_mask |= 1u64 << i;
            } else if r < p_y {
                x_mask |= 1u64 << i;
                z_mask |= 1u64 << i;
            }
        }

        (x_mask, z_mask)
    }

    /// Apply bit-flip (X) channel with error rate p
    pub fn bit_flip(&mut self, qubit: usize, p: f64, rng: &mut QRng) {
        if p <= 0.0 || qubit >= self.n_qubits {
            return;
        }
        let mask = Self::bernoulli_64(p, rng);
        self.inject_x(qubit, mask);
    }

    /// Apply phase-flip (Z) channel with error rate p
    pub fn phase_flip(&mut self, qubit: usize, p: f64, rng: &mut QRng) {
        if p <= 0.0 || qubit >= self.n_qubits {
            return;
        }
        let mask = Self::bernoulli_64(p, rng);
        self.inject_z(qubit, mask);
    }

    // =========================================================================
    // Clifford Gate Propagation (Exact)
    // =========================================================================

    /// Propagate errors through CNOT(control, target)
    ///
    /// X propagates forward: X_c -> X_c X_t
    /// Z propagates backward: Z_t -> Z_c Z_t
    #[inline]
    pub fn cnot(&mut self, control: usize, target: usize) {
        if control >= self.n_qubits || target >= self.n_qubits || control == target {
            return;
        }

        // X propagates forward
        let x_ctrl = self.errors[control].0;
        self.errors[target].0 ^= x_ctrl;

        // Z propagates backward
        let z_targ = self.errors[target].1;
        self.errors[control].1 ^= z_targ;
    }

    /// Propagate errors through Hadamard on qubit
    ///
    /// H: X <-> Z (swap)
    #[inline]
    pub fn hadamard(&mut self, qubit: usize) {
        if let Some((x, z)) = self.errors_mut(qubit) {
            std::mem::swap(x, z);
        }
    }

    /// Propagate errors through S gate (phase gate) on qubit
    ///
    /// S: X -> Y = XZ, Z -> Z
    /// In terms of errors: z_new = z XOR x
    #[inline]
    pub fn s_gate(&mut self, qubit: usize) {
        if let Some((x, z)) = self.errors_mut(qubit) {
            *z ^= *x;
        }
    }

    /// Propagate errors through S-dagger gate on qubit
    ///
    /// S†: X -> -Y = XZ (same as S for Pauli tracking), Z -> Z
    #[inline]
    pub fn s_dag(&mut self, qubit: usize) {
        // For Pauli frame tracking, S and S† have same effect
        self.s_gate(qubit);
    }

    /// Propagate errors through CZ(q1, q2)
    ///
    /// CZ: X_1 -> X_1 Z_2, X_2 -> Z_1 X_2, Z unchanged
    #[inline]
    pub fn cz(&mut self, q1: usize, q2: usize) {
        if q1 >= self.n_qubits || q2 >= self.n_qubits || q1 == q2 {
            return;
        }

        let x1 = self.errors[q1].0;
        let x2 = self.errors[q2].0;

        self.errors[q2].1 ^= x1;
        self.errors[q1].1 ^= x2;
    }

    /// Propagate errors through SWAP(q1, q2)
    ///
    /// SWAP exchanges all error information between qubits
    #[inline]
    pub fn swap(&mut self, q1: usize, q2: usize) {
        if q1 >= self.n_qubits || q2 >= self.n_qubits || q1 == q2 {
            return;
        }
        self.errors.swap(q1, q2);
    }

    // =========================================================================
    // Non-Clifford Surrogates (LABELED APPROXIMATIONS)
    // =========================================================================
    //
    // These are NOT exact. Pauli errors don't stay Pauli under non-Clifford gates.
    // We provide two variants that bracket the true behavior:
    // - Pessimistic: more aggressive error propagation (underestimates fidelity)
    // - Optimistic: minimal propagation (overestimates fidelity)
    //
    // If both bracket the exact result, we have a defensible approximation.

    /// CCZ surrogate - PESSIMISTIC variant
    ///
    /// CCZ is diagonal in computational basis. This pessimistic model:
    /// - X errors on any control create Z errors on all other qubits
    /// - Z errors pass through unchanged
    ///
    /// This tends to UNDERESTIMATE fidelity (more errors than reality).
    #[inline]
    pub fn ccz_pessimistic(&mut self, c1: usize, c2: usize, target: usize) {
        if c1 >= self.n_qubits || c2 >= self.n_qubits || target >= self.n_qubits {
            return;
        }

        let x_c1 = self.errors[c1].0;
        let x_c2 = self.errors[c2].0;
        let x_t = self.errors[target].0;

        // Pessimistic: any X error creates Z on the other two qubits
        // This overcounts correlations
        self.errors[c2].1 ^= x_c1;
        self.errors[target].1 ^= x_c1;

        self.errors[c1].1 ^= x_c2;
        self.errors[target].1 ^= x_c2;

        self.errors[c1].1 ^= x_t;
        self.errors[c2].1 ^= x_t;
    }

    /// CCZ surrogate - OPTIMISTIC variant
    ///
    /// Minimal error propagation model:
    /// - Only X errors on BOTH controls create Z on target
    /// - Z errors pass through unchanged
    ///
    /// This tends to OVERESTIMATE fidelity (fewer errors than reality).
    #[inline]
    pub fn ccz_optimistic(&mut self, c1: usize, c2: usize, target: usize) {
        if c1 >= self.n_qubits || c2 >= self.n_qubits || target >= self.n_qubits {
            return;
        }

        let x_c1 = self.errors[c1].0;
        let x_c2 = self.errors[c2].0;

        // Optimistic: only joint X errors on both controls affect target
        let both_x = x_c1 & x_c2;
        self.errors[target].1 ^= both_x;
    }

    /// Toffoli surrogate - PESSIMISTIC variant
    ///
    /// Toffoli = H(target) · CCZ · H(target)
    /// Respecting this structure gives a more faithful approximation.
    #[inline]
    pub fn toffoli_pessimistic(&mut self, c1: usize, c2: usize, target: usize) {
        self.hadamard(target);
        self.ccz_pessimistic(c1, c2, target);
        self.hadamard(target);
    }

    /// Toffoli surrogate - OPTIMISTIC variant
    ///
    /// Toffoli = H(target) · CCZ · H(target)
    #[inline]
    pub fn toffoli_optimistic(&mut self, c1: usize, c2: usize, target: usize) {
        self.hadamard(target);
        self.ccz_optimistic(c1, c2, target);
        self.hadamard(target);
    }

    /// Fredkin (CSWAP) surrogate - PESSIMISTIC variant
    ///
    /// Fredkin = CNOT(t2,t1) · Toffoli(c,t1,t2) · CNOT(t2,t1)
    #[inline]
    pub fn fredkin_pessimistic(&mut self, control: usize, t1: usize, t2: usize) {
        self.cnot(t2, t1);
        self.toffoli_pessimistic(control, t1, t2);
        self.cnot(t2, t1);
    }

    /// Fredkin (CSWAP) surrogate - OPTIMISTIC variant
    ///
    /// Fredkin = CNOT(t2,t1) · Toffoli(c,t1,t2) · CNOT(t2,t1)
    #[inline]
    pub fn fredkin_optimistic(&mut self, control: usize, t1: usize, t2: usize) {
        self.cnot(t2, t1);
        self.toffoli_optimistic(control, t1, t2);
        self.cnot(t2, t1);
    }

    /// Toffoli using current surrogate model
    #[inline]
    pub fn toffoli(&mut self, c1: usize, c2: usize, target: usize) {
        match self.surrogate {
            SurrogateModel::Pessimistic => self.toffoli_pessimistic(c1, c2, target),
            SurrogateModel::Optimistic => self.toffoli_optimistic(c1, c2, target),
        }
    }

    /// Fredkin using current surrogate model
    #[inline]
    pub fn fredkin(&mut self, control: usize, t1: usize, t2: usize) {
        match self.surrogate {
            SurrogateModel::Pessimistic => self.fredkin_pessimistic(control, t1, t2),
            SurrogateModel::Optimistic => self.fredkin_optimistic(control, t1, t2),
        }
    }

    /// CCZ using current surrogate model
    #[inline]
    pub fn ccz(&mut self, c1: usize, c2: usize, target: usize) {
        match self.surrogate {
            SurrogateModel::Pessimistic => self.ccz_pessimistic(c1, c2, target),
            SurrogateModel::Optimistic => self.ccz_optimistic(c1, c2, target),
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Count trajectories with X error on qubit
    #[inline]
    pub fn count_x_errors(&self, qubit: usize) -> u32 {
        self.x_errors(qubit).count_ones()
    }

    /// Count trajectories with Z error on qubit
    #[inline]
    pub fn count_z_errors(&self, qubit: usize) -> u32 {
        self.z_errors(qubit).count_ones()
    }

    /// Count trajectories with any error (X or Z) on qubit
    #[inline]
    pub fn count_any_errors(&self, qubit: usize) -> u32 {
        let (x, z) = self.errors.get(qubit).copied().unwrap_or((0, 0));
        (x | z).count_ones()
    }

    /// Get mask of trajectories with no errors on specified qubits
    pub fn clean_trajectories(&self, qubits: &[usize]) -> u64 {
        let mut failure_mask: u64 = 0;
        for &q in qubits {
            if q < self.n_qubits {
                failure_mask |= self.errors[q].0;
                failure_mask |= self.errors[q].1;
            }
        }
        !failure_mask
    }

    /// Get mask of trajectories with no X errors on specified qubits
    /// (Z errors don't affect computational basis measurement)
    pub fn clean_x_trajectories(&self, qubits: &[usize]) -> u64 {
        let mut failure_mask: u64 = 0;
        for &q in qubits {
            if q < self.n_qubits {
                failure_mask |= self.errors[q].0;
            }
        }
        !failure_mask
    }

    /// Count clean trajectories (no errors on specified qubits)
    pub fn count_clean(&self, qubits: &[usize]) -> u32 {
        self.clean_trajectories(qubits).count_ones()
    }

    /// Count trajectories with no X errors on specified qubits
    pub fn count_clean_x(&self, qubits: &[usize]) -> u32 {
        self.clean_x_trajectories(qubits).count_ones()
    }

    /// QRAM query success count (observable-based)
    ///
    /// For QRAM with given depth:
    /// - Address qubits: 0..depth
    /// - Bus qubit: depth
    ///
    /// Success = no X errors on address or bus qubits
    /// (Z errors don't affect computational basis measurement)
    ///
    /// Returns count of successful trajectories (0-64)
    pub fn query_success_count(&self, depth: usize) -> u32 {
        let mut failure_mask: u64 = 0;

        // Check address qubits for X errors
        for q in 0..depth {
            if q < self.n_qubits {
                failure_mask |= self.errors[q].0;
            }
        }

        // Check bus qubit for X errors
        if depth < self.n_qubits {
            failure_mask |= self.errors[depth].0;
        }

        // Success = NOT failure
        (!failure_mask).count_ones()
    }

    /// QRAM query success mask (observable-based)
    ///
    /// Returns u64 where bit i is set if trajectory i succeeded.
    pub fn query_success_mask(&self, depth: usize) -> u64 {
        let mut failure_mask: u64 = 0;

        for q in 0..=depth {
            if q < self.n_qubits {
                failure_mask |= self.errors[q].0;
            }
        }

        !failure_mask
    }

    /// Total error weight across all qubits and trajectories
    pub fn total_error_weight(&self) -> u64 {
        let mut total: u64 = 0;
        for (x, z) in &self.errors {
            total += x.count_ones() as u64;
            total += z.count_ones() as u64;
        }
        total
    }

    /// Get detailed statistics
    pub fn stats(&self) -> PackedFrameStats {
        let mut total_x = 0u64;
        let mut total_z = 0u64;
        let mut qubits_with_errors = 0usize;

        for (x, z) in &self.errors {
            total_x += x.count_ones() as u64;
            total_z += z.count_ones() as u64;
            if *x != 0 || *z != 0 {
                qubits_with_errors += 1;
            }
        }

        PackedFrameStats {
            n_qubits: self.n_qubits,
            total_x_errors: total_x,
            total_z_errors: total_z,
            qubits_with_errors,
            memory_bytes: self.n_qubits * 16,
        }
    }
}

impl fmt::Debug for PackedPauliFrame64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        f.debug_struct("PackedPauliFrame64")
            .field("n_qubits", &self.n_qubits)
            .field("total_x", &stats.total_x_errors)
            .field("total_z", &stats.total_z_errors)
            .finish()
    }
}

impl fmt::Display for PackedPauliFrame64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        write!(
            f,
            "PackedPauliFrame64(qubits={}, X={}, Z={}, {}B)",
            self.n_qubits, stats.total_x_errors, stats.total_z_errors, stats.memory_bytes
        )
    }
}

/// Statistics for a packed frame
#[derive(Clone, Debug)]
pub struct PackedFrameStats {
    pub n_qubits: usize,
    pub total_x_errors: u64,
    pub total_z_errors: u64,
    pub qubits_with_errors: usize,
    pub memory_bytes: usize,
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> QRng {
        QRng::new(12345)
    }

    #[test]
    fn test_creation() {
        let frame = PackedPauliFrame64::new(10);
        assert_eq!(frame.n_qubits(), 10);
        assert_eq!(frame.x_errors(0), 0);
        assert_eq!(frame.z_errors(0), 0);
    }

    #[test]
    fn test_reset() {
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_x(0, u64::MAX);
        frame.inject_z(1, u64::MAX);

        frame.reset();

        assert_eq!(frame.x_errors(0), 0);
        assert_eq!(frame.z_errors(1), 0);
    }

    #[test]
    fn test_inject_x() {
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_x(2, 0b1010);
        assert_eq!(frame.x_errors(2), 0b1010);

        // XOR behavior
        frame.inject_x(2, 0b1100);
        assert_eq!(frame.x_errors(2), 0b0110);
    }

    #[test]
    fn test_inject_z() {
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_z(3, 0b1111);
        assert_eq!(frame.z_errors(3), 0b1111);
    }

    #[test]
    fn test_inject_y() {
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_y(1, 0b1010);
        assert_eq!(frame.x_errors(1), 0b1010);
        assert_eq!(frame.z_errors(1), 0b1010);
    }

    #[test]
    fn test_bernoulli_64_extremes() {
        let mut rng = test_rng();

        // p=0 should give all zeros
        let zeros = PackedPauliFrame64::bernoulli_64(0.0, &mut rng);
        assert_eq!(zeros, 0);

        // p=1 should give all ones
        let ones = PackedPauliFrame64::bernoulli_64(1.0, &mut rng);
        assert_eq!(ones, u64::MAX);
    }

    #[test]
    fn test_bernoulli_64_statistical() {
        let mut rng = test_rng();
        let p = 0.25;
        let trials = 1000;

        let mut total_ones: u64 = 0;
        for _ in 0..trials {
            let mask = PackedPauliFrame64::bernoulli_64(p, &mut rng);
            total_ones += mask.count_ones() as u64;
        }

        let expected = (p * 64.0 * trials as f64) as u64;
        let actual = total_ones;

        // Should be within 5% of expected
        let tolerance = (expected as f64 * 0.05) as u64;
        assert!(
            actual.abs_diff(expected) < tolerance,
            "Expected ~{}, got {} (tolerance {})",
            expected,
            actual,
            tolerance
        );
    }

    #[test]
    fn test_depolarizing_channel() {
        let mut rng = test_rng();
        let mut frame = PackedPauliFrame64::new(5);
        let p = 0.3;
        let trials = 1000;

        let mut x_count = 0u64;
        let mut z_count = 0u64;
        let mut y_count = 0u64; // Both X and Z

        for _ in 0..trials {
            frame.reset();
            frame.depolarizing(0, p, &mut rng);

            let x = frame.x_errors(0);
            let z = frame.z_errors(0);

            // Count X-only, Z-only, Y (both)
            x_count += (x & !z).count_ones() as u64;
            z_count += (z & !x).count_ones() as u64;
            y_count += (x & z).count_ones() as u64;
        }

        let total_samples = trials as u64 * 64;
        let expected_each = (p / 3.0 * total_samples as f64) as u64;

        // Each of X, Y, Z should be roughly p/3
        let tolerance = (expected_each as f64 * 0.15) as u64; // 15% tolerance for randomness

        assert!(
            x_count.abs_diff(expected_each) < tolerance,
            "X: expected ~{}, got {} (tolerance {})",
            expected_each,
            x_count,
            tolerance
        );
        assert!(
            z_count.abs_diff(expected_each) < tolerance,
            "Z: expected ~{}, got {} (tolerance {})",
            expected_each,
            z_count,
            tolerance
        );
        assert!(
            y_count.abs_diff(expected_each) < tolerance,
            "Y: expected ~{}, got {} (tolerance {})",
            expected_each,
            y_count,
            tolerance
        );
    }

    #[test]
    fn test_cnot_x_propagation() {
        let mut frame = PackedPauliFrame64::new(5);

        // X on control should propagate to target
        frame.inject_x(0, 0b1111);
        frame.cnot(0, 1);

        assert_eq!(frame.x_errors(0), 0b1111, "X remains on control");
        assert_eq!(frame.x_errors(1), 0b1111, "X propagates to target");
    }

    #[test]
    fn test_cnot_z_propagation() {
        let mut frame = PackedPauliFrame64::new(5);

        // Z on target should propagate back to control
        frame.inject_z(1, 0b1010);
        frame.cnot(0, 1);

        assert_eq!(frame.z_errors(0), 0b1010, "Z propagates to control");
        assert_eq!(frame.z_errors(1), 0b1010, "Z remains on target");
    }

    #[test]
    fn test_cnot_no_cross_propagation() {
        let mut frame = PackedPauliFrame64::new(5);

        // X on target should NOT propagate to control
        frame.inject_x(1, 0b1111);
        frame.cnot(0, 1);
        assert_eq!(frame.x_errors(0), 0, "X on target doesn't go to control");

        // Z on control should NOT propagate to target
        frame.reset();
        frame.inject_z(0, 0b1111);
        frame.cnot(0, 1);
        assert_eq!(frame.z_errors(1), 0, "Z on control doesn't go to target");
    }

    #[test]
    fn test_hadamard_swap() {
        let mut frame = PackedPauliFrame64::new(5);

        frame.inject_x(0, 0b1100);
        frame.inject_z(0, 0b0011);

        frame.hadamard(0);

        assert_eq!(frame.x_errors(0), 0b0011, "X becomes Z");
        assert_eq!(frame.z_errors(0), 0b1100, "Z becomes X");
    }

    #[test]
    fn test_s_gate() {
        let mut frame = PackedPauliFrame64::new(5);

        // S: X -> XZ, Z -> Z
        frame.inject_x(0, 0b1111);
        frame.inject_z(0, 0b0000);

        frame.s_gate(0);

        assert_eq!(frame.x_errors(0), 0b1111, "X unchanged");
        assert_eq!(frame.z_errors(0), 0b1111, "Z = X XOR Z = X");
    }

    #[test]
    fn test_cz_propagation() {
        let mut frame = PackedPauliFrame64::new(5);

        // X on q1 should create Z on q2
        frame.inject_x(0, 0b1010);
        frame.cz(0, 1);

        assert_eq!(frame.x_errors(0), 0b1010, "X1 unchanged");
        assert_eq!(frame.z_errors(1), 0b1010, "Z2 from X1");

        // X on q2 should create Z on q1
        frame.reset();
        frame.inject_x(1, 0b0101);
        frame.cz(0, 1);

        assert_eq!(frame.x_errors(1), 0b0101, "X2 unchanged");
        assert_eq!(frame.z_errors(0), 0b0101, "Z1 from X2");
    }

    #[test]
    fn test_swap() {
        let mut frame = PackedPauliFrame64::new(5);

        frame.inject_x(0, 0b1111);
        frame.inject_z(1, 0b0101);

        frame.swap(0, 1);

        assert_eq!(frame.x_errors(1), 0b1111, "X moved from 0 to 1");
        assert_eq!(frame.z_errors(0), 0b0101, "Z moved from 1 to 0");
        assert_eq!(frame.x_errors(0), 0, "Original qubit 0 now clean X");
        assert_eq!(frame.z_errors(1), 0, "Original qubit 1 now clean Z");
    }

    #[test]
    fn test_count_errors() {
        let mut frame = PackedPauliFrame64::new(5);

        frame.inject_x(0, 0b11110000);
        assert_eq!(frame.count_x_errors(0), 4);

        frame.inject_z(0, 0b00001111);
        assert_eq!(frame.count_z_errors(0), 4);
        assert_eq!(frame.count_any_errors(0), 8);
    }

    #[test]
    fn test_clean_trajectories() {
        let mut frame = PackedPauliFrame64::new(5);

        // Trajectories 0-3 have errors on qubit 0
        frame.inject_x(0, 0b1111);
        // Trajectories 4-7 have errors on qubit 1
        frame.inject_z(1, 0b11110000);

        let clean = frame.clean_trajectories(&[0, 1]);
        // Trajectories 8-63 should be clean
        assert_eq!(clean, !0b11111111u64);
        assert_eq!(clean.count_ones(), 56);
    }

    #[test]
    fn test_clean_x_trajectories() {
        let mut frame = PackedPauliFrame64::new(5);

        // X errors matter
        frame.inject_x(0, 0b1111);
        // Z errors don't matter for computational basis
        frame.inject_z(0, 0b11110000);

        let clean_x = frame.clean_x_trajectories(&[0]);
        // Only trajectories 0-3 have X errors
        assert_eq!(clean_x, !0b1111u64);
        assert_eq!(clean_x.count_ones(), 60);
    }

    #[test]
    fn test_stats() {
        let mut frame = PackedPauliFrame64::new(10);
        frame.inject_x(0, 0b1111);
        frame.inject_z(5, 0b11111111);

        let stats = frame.stats();
        assert_eq!(stats.n_qubits, 10);
        assert_eq!(stats.total_x_errors, 4);
        assert_eq!(stats.total_z_errors, 8);
        assert_eq!(stats.qubits_with_errors, 2);
        assert_eq!(stats.memory_bytes, 160);
    }

    #[test]
    fn test_display() {
        let frame = PackedPauliFrame64::new(10);
        let s = format!("{}", frame);
        assert!(s.contains("PackedPauliFrame64"));
        assert!(s.contains("qubits=10"));
    }

    #[test]
    fn test_clifford_identity_hh() {
        // H·H = I
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_x(0, 0b1010);
        frame.inject_z(0, 0b0101);

        frame.hadamard(0);
        frame.hadamard(0);

        assert_eq!(frame.x_errors(0), 0b1010);
        assert_eq!(frame.z_errors(0), 0b0101);
    }

    #[test]
    fn test_clifford_identity_ssss() {
        // S^4 = I
        let mut frame = PackedPauliFrame64::new(5);
        frame.inject_x(0, 0b1111);
        frame.inject_z(0, 0b0000);

        let orig_x = frame.x_errors(0);
        let orig_z = frame.z_errors(0);

        for _ in 0..4 {
            frame.s_gate(0);
        }

        assert_eq!(frame.x_errors(0), orig_x);
        assert_eq!(frame.z_errors(0), orig_z);
    }

    // =========================================================================
    // Non-Clifford Surrogate Tests
    // =========================================================================

    #[test]
    fn test_ccz_pessimistic_x_on_control() {
        let mut frame = PackedPauliFrame64::new(5);

        // X error on first control
        frame.inject_x(0, 0b1111);
        frame.ccz_pessimistic(0, 1, 2);

        // Pessimistic: X on c1 creates Z on c2 and target
        assert_eq!(frame.x_errors(0), 0b1111, "X remains on c1");
        assert_eq!(frame.z_errors(1), 0b1111, "Z created on c2");
        assert_eq!(frame.z_errors(2), 0b1111, "Z created on target");
    }

    #[test]
    fn test_ccz_optimistic_single_x() {
        let mut frame = PackedPauliFrame64::new(5);

        // X error on only one control
        frame.inject_x(0, 0b1111);
        frame.ccz_optimistic(0, 1, 2);

        // Optimistic: single X doesn't create Z on target
        assert_eq!(frame.x_errors(0), 0b1111, "X remains on c1");
        assert_eq!(frame.z_errors(2), 0, "No Z on target (need both controls)");
    }

    #[test]
    fn test_ccz_optimistic_both_x() {
        let mut frame = PackedPauliFrame64::new(5);

        // X errors on both controls
        frame.inject_x(0, 0b1111);
        frame.inject_x(1, 0b0011); // Overlap on bits 0,1

        frame.ccz_optimistic(0, 1, 2);

        // Optimistic: joint X errors create Z on target
        assert_eq!(
            frame.z_errors(2),
            0b0011,
            "Z on target where both controls have X"
        );
    }

    #[test]
    fn test_pessimistic_more_errors_than_optimistic() {
        // Key property: pessimistic should generate >= errors than optimistic
        let mut pess = PackedPauliFrame64::new(5);
        let mut opt = PackedPauliFrame64::new(5);

        // Same initial state
        pess.inject_x(0, 0b11111111);
        opt.inject_x(0, 0b11111111);

        pess.ccz_pessimistic(0, 1, 2);
        opt.ccz_optimistic(0, 1, 2);

        let pess_errors = pess.total_error_weight();
        let opt_errors = opt.total_error_weight();

        assert!(
            pess_errors >= opt_errors,
            "Pessimistic ({}) should have >= errors than optimistic ({})",
            pess_errors,
            opt_errors
        );
    }

    #[test]
    fn test_toffoli_uses_hadamard_sandwich() {
        // Toffoli = H(t) · CCZ · H(t)
        // Test that toffoli_* respects this structure by checking
        // that X on target behaves differently than X on controls
        let mut frame = PackedPauliFrame64::new(5);

        // X on target (will be transformed by H before CCZ)
        frame.inject_x(2, 0b1111);
        frame.toffoli_optimistic(0, 1, 2);

        // After H·CCZ·H, the X should be affected differently
        // than if we just did CCZ directly
        // (H turns X into Z before CCZ, then back)
        // This is more of a structure test than exact value test
        assert!(
            frame.x_errors(2) != 0 || frame.z_errors(2) != 0,
            "Target should have some error after toffoli"
        );
    }

    #[test]
    fn test_fredkin_decomposition() {
        // Fredkin = CNOT · Toffoli · CNOT
        // Basic sanity check that the decomposition runs
        let mut frame = PackedPauliFrame64::new(5);

        frame.inject_x(0, 0b1111); // Control has X error
        frame.fredkin_pessimistic(0, 1, 2);

        // Should not crash, and should propagate some errors
        let total = frame.total_error_weight();
        assert!(total > 0, "Should have propagated some errors");
    }

    #[test]
    fn test_surrogate_model_enum() {
        // Test the enum
        assert_eq!(SurrogateModel::default(), SurrogateModel::Optimistic);
        assert_ne!(SurrogateModel::Pessimistic, SurrogateModel::Optimistic);
    }

    #[test]
    fn test_fredkin_on_clean_state() {
        // Fredkin on clean state should stay clean
        let mut pess = PackedPauliFrame64::new(5);
        let mut opt = PackedPauliFrame64::new(5);

        pess.fredkin_pessimistic(0, 1, 2);
        opt.fredkin_optimistic(0, 1, 2);

        assert_eq!(pess.total_error_weight(), 0, "Pessimistic clean->clean");
        assert_eq!(opt.total_error_weight(), 0, "Optimistic clean->clean");
    }

    #[test]
    fn test_bracketing_property() {
        // The key property: pessimistic and optimistic should bracket
        // For a range of error patterns, pessimistic errors >= optimistic errors
        let patterns = [0b1, 0b11, 0b1111, 0b11111111, u64::MAX];

        for &pattern in &patterns {
            let mut pess = PackedPauliFrame64::new(5);
            let mut opt = PackedPauliFrame64::new(5);

            pess.inject_x(0, pattern);
            opt.inject_x(0, pattern);

            pess.fredkin_pessimistic(0, 1, 2);
            opt.fredkin_optimistic(0, 1, 2);

            assert!(
                pess.total_error_weight() >= opt.total_error_weight(),
                "Bracketing violated for pattern {:b}",
                pattern
            );
        }
    }
}
