//! QEC Error Model for Surface Code
//!
//! Provides logical error rate calculations based on surface code parameters.
//! This is a phenomenological model - it doesn't simulate individual physical
//! qubits, but uses validated scaling formulas.
//!
//! # Surface Code Scaling
//!
//! For a distance-d surface code with physical error rate p:
//!
//! ```text
//! p_logical ≈ 0.1 × (p / p_threshold)^((d+1)/2)
//! ```
//!
//! where p_threshold ≈ 1% for depolarizing noise.
//!
//! # References
//!
//! - Fowler et al., "Surface codes: Towards practical large-scale quantum computation"
//! - Gidney & Ekerå, "How to factor 2048 bit RSA integers in 8 hours"

use std::fmt;

/// Surface code error model.
///
/// Computes logical error rates based on code distance, physical error rate,
/// and number of code cycles. This is a phenomenological model suitable for
/// resource estimation.
#[derive(Clone, Debug)]
pub struct QecModel {
    /// Code distance (must be odd for surface code)
    pub distance: usize,
    /// Physical error rate per gate/measurement
    pub p_phys: f64,
    /// Error threshold (typically ~1% for surface code)
    pub p_threshold: f64,
    /// Logical error rate per code cycle
    pub p_logical_per_cycle: f64,
    /// Physical qubits per logical qubit (approximately 2d²)
    pub physical_qubits_per_logical: usize,
}

impl QecModel {
    /// Standard surface code threshold for depolarizing noise
    pub const DEPOLARIZING_THRESHOLD: f64 = 0.01;

    /// Standard surface code threshold for circuit-level noise
    pub const CIRCUIT_THRESHOLD: f64 = 0.0057;

    /// Create a new QEC model with default threshold
    ///
    /// Uses the standard depolarizing threshold of ~1%.
    pub fn new(distance: usize, p_phys: f64) -> Self {
        Self::with_threshold(distance, p_phys, Self::DEPOLARIZING_THRESHOLD)
    }

    /// Create QEC model with custom threshold
    pub fn with_threshold(distance: usize, p_phys: f64, p_threshold: f64) -> Self {
        // Standard surface code scaling formula
        // p_L ≈ 0.1 × (p/p_th)^((d+1)/2)
        let ratio = p_phys / p_threshold;
        let exponent = (distance as f64 + 1.0) / 2.0;
        let p_logical = 0.1 * ratio.powf(exponent);

        // Physical qubits: approximately 2d² for rotated surface code
        // (d² data qubits + d² ancilla qubits, roughly)
        let physical_qubits = 2 * distance * distance;

        Self {
            distance,
            p_phys,
            p_threshold,
            p_logical_per_cycle: p_logical.min(1.0), // Cap at 1
            physical_qubits_per_logical: physical_qubits,
        }
    }

    /// Create QEC model for circuit-level noise model
    ///
    /// Uses the more realistic circuit-level threshold of ~0.57%.
    pub fn circuit_level(distance: usize, p_phys: f64) -> Self {
        Self::with_threshold(distance, p_phys, Self::CIRCUIT_THRESHOLD)
    }

    /// Logical error probability after n code cycles
    ///
    /// Assumes independent errors per cycle (pessimistic for correlated errors).
    pub fn error_after_cycles(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        1.0 - (1.0 - self.p_logical_per_cycle).powi(n as i32)
    }

    /// Number of cycles to reach target logical error rate
    ///
    /// Returns None if target is unachievable (p_logical_per_cycle >= target).
    pub fn cycles_for_target_error(&self, target: f64) -> Option<usize> {
        if self.p_logical_per_cycle >= target {
            return None; // Can never achieve target
        }
        if target >= 1.0 {
            return Some(usize::MAX); // Any number of cycles is fine
        }

        // Solve: 1 - (1 - p)^n = target
        // (1 - p)^n = 1 - target
        // n = log(1 - target) / log(1 - p)
        let n = (1.0 - target).ln() / (1.0 - self.p_logical_per_cycle).ln();
        Some(n.ceil() as usize)
    }

    /// Required distance to achieve target error rate with given physical rate
    ///
    /// Inverts the scaling formula to find minimum distance.
    pub fn required_distance(p_phys: f64, target_per_cycle: f64, p_threshold: f64) -> usize {
        if p_phys >= p_threshold {
            return usize::MAX; // Below threshold, can't achieve any target
        }

        // p_L = 0.1 × (p/p_th)^((d+1)/2)
        // target = 0.1 × ratio^((d+1)/2)
        // ratio^((d+1)/2) = target / 0.1 = 10 × target
        // ((d+1)/2) × ln(ratio) = ln(10 × target)
        // d+1 = 2 × ln(10 × target) / ln(ratio)
        // d = 2 × ln(10 × target) / ln(ratio) - 1

        let ratio = p_phys / p_threshold;
        let d = 2.0 * (10.0 * target_per_cycle).ln() / ratio.ln() - 1.0;

        // Round up to next odd number (surface code requires odd distance)
        let d_ceil = d.ceil() as usize;
        if d_ceil % 2 == 0 {
            d_ceil + 1
        } else {
            d_ceil.max(3) // Minimum distance is 3
        }
    }

    /// Physical qubits needed for n logical qubits
    pub fn physical_qubits_for(&self, n_logical: usize) -> usize {
        n_logical * self.physical_qubits_per_logical
    }

    /// Check if operating below threshold (error correction is beneficial)
    pub fn is_below_threshold(&self) -> bool {
        self.p_phys < self.p_threshold
    }

    /// Suppression factor: how much better is logical vs physical
    ///
    /// Returns ratio p_phys / p_logical_per_cycle.
    /// Values > 1 mean error correction is helping.
    pub fn suppression_factor(&self) -> f64 {
        if self.p_logical_per_cycle > 0.0 {
            self.p_phys / self.p_logical_per_cycle
        } else {
            f64::INFINITY
        }
    }

    /// Lambda parameter (used in some scaling formulas)
    ///
    /// λ = p / p_threshold
    pub fn lambda(&self) -> f64 {
        self.p_phys / self.p_threshold
    }

    /// Expected number of logical errors over n cycles across m qubits
    pub fn expected_logical_errors(&self, n_qubits: usize, n_cycles: usize) -> f64 {
        // Each qubit has independent error probability per cycle
        // Expected errors = n_qubits × n_cycles × p_logical_per_cycle
        // (for small p, this is approximately correct)
        n_qubits as f64 * n_cycles as f64 * self.p_logical_per_cycle
    }

    /// Probability that at least one logical error occurs
    pub fn any_error_probability(&self, n_qubits: usize, n_cycles: usize) -> f64 {
        // P(at least one error) = 1 - P(no errors)
        // P(no errors) = (1 - p_per_cycle)^(n_qubits × n_cycles)
        let total_opportunities = n_qubits * n_cycles;
        1.0 - (1.0 - self.p_logical_per_cycle).powi(total_opportunities as i32)
    }
}

impl fmt::Display for QecModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Surface Code QEC Model:")?;
        writeln!(f, "  Distance:          {}", self.distance)?;
        writeln!(f, "  Physical p:        {:.2e}", self.p_phys)?;
        writeln!(f, "  Threshold:         {:.2e}", self.p_threshold)?;
        writeln!(f, "  Logical p/cycle:   {:.2e}", self.p_logical_per_cycle)?;
        writeln!(
            f,
            "  Phys qubits/log:   {}",
            self.physical_qubits_per_logical
        )?;
        writeln!(f, "  Suppression:       {:.1}×", self.suppression_factor())?;
        writeln!(
            f,
            "  Below threshold:   {}",
            if self.is_below_threshold() {
                "yes"
            } else {
                "NO"
            }
        )?;
        Ok(())
    }
}

// ============================================================================
// Phenomenological Noise Model
// ============================================================================

/// Phenomenological noise model for decoder testing.
///
/// This applies noise directly to logical error tracking rather than
/// simulating physical qubits. Useful for integration testing.
#[derive(Clone, Debug)]
pub struct PhenomenologicalNoise {
    /// X error probability per qubit per round
    pub p_x: f64,
    /// Z error probability per qubit per round
    pub p_z: f64,
    /// Measurement error probability
    pub p_meas: f64,
}

impl PhenomenologicalNoise {
    /// Create symmetric depolarizing-like noise
    pub fn depolarizing(p: f64) -> Self {
        // Depolarizing: each of X, Y, Z with probability p/3
        // Y = XZ, so effective X and Z are each ~2p/3
        Self {
            p_x: 2.0 * p / 3.0,
            p_z: 2.0 * p / 3.0,
            p_meas: p,
        }
    }

    /// Create bit-flip only noise
    pub fn bit_flip(p: f64) -> Self {
        Self {
            p_x: p,
            p_z: 0.0,
            p_meas: 0.0,
        }
    }

    /// Create phase-flip only noise
    pub fn phase_flip(p: f64) -> Self {
        Self {
            p_x: 0.0,
            p_z: p,
            p_meas: 0.0,
        }
    }

    /// Create from separate probabilities
    pub fn new(p_x: f64, p_z: f64, p_meas: f64) -> Self {
        Self { p_x, p_z, p_meas }
    }

    /// Total error probability (approximate, for estimation)
    pub fn total_error_probability(&self) -> f64 {
        // Pessimistic: any error type counts
        1.0 - (1.0 - self.p_x) * (1.0 - self.p_z) * (1.0 - self.p_meas)
    }
}

// ============================================================================
// Decoder Performance Model
// ============================================================================

/// Decoder performance characteristics.
///
/// Different decoders have different accuracy/speed tradeoffs.
/// This allows modeling decoder choice in resource estimation.
#[derive(Clone, Debug)]
pub enum DecoderModel {
    /// Minimum Weight Perfect Matching (optimal but slow)
    MWPM {
        /// Threshold achieved (typically ~10.3% for surface code)
        threshold: f64,
    },
    /// Union-Find decoder (fast but sub-optimal)
    UnionFind {
        /// Threshold achieved (typically ~5-8%)
        threshold: f64,
    },
    /// Custom decoder with specified threshold
    Custom { name: String, threshold: f64 },
}

impl DecoderModel {
    /// Standard MWPM decoder
    pub fn mwpm() -> Self {
        Self::MWPM { threshold: 0.103 }
    }

    /// Standard Union-Find decoder
    pub fn union_find() -> Self {
        Self::UnionFind { threshold: 0.07 }
    }

    /// Get the threshold for this decoder
    pub fn threshold(&self) -> f64 {
        match self {
            Self::MWPM { threshold } => *threshold,
            Self::UnionFind { threshold } => *threshold,
            Self::Custom { threshold, .. } => *threshold,
        }
    }

    /// Create QEC model using this decoder's threshold
    pub fn create_model(&self, distance: usize, p_phys: f64) -> QecModel {
        QecModel::with_threshold(distance, p_phys, self.threshold())
    }
}

impl fmt::Display for DecoderModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MWPM { threshold } => write!(f, "MWPM (threshold: {:.1}%)", threshold * 100.0),
            Self::UnionFind { threshold } => {
                write!(f, "Union-Find (threshold: {:.1}%)", threshold * 100.0)
            }
            Self::Custom { name, threshold } => {
                write!(f, "{} (threshold: {:.1}%)", name, threshold * 100.0)
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qec_model_scaling() {
        // At 0.1% physical error, distance 7:
        // p_L = 0.1 × (0.001/0.01)^((7+1)/2) = 0.1 × 0.1^4 = 1e-5
        let model = QecModel::new(7, 1e-3);
        assert!(model.p_logical_per_cycle <= 1e-4);
        assert!(model.p_logical_per_cycle >= 1e-6);
        println!("d=7, p=1e-3: p_L = {:.2e}", model.p_logical_per_cycle);
    }

    #[test]
    fn test_qec_model_below_threshold() {
        let model = QecModel::new(5, 0.001);
        assert!(model.is_below_threshold());

        let model_above = QecModel::new(5, 0.02);
        assert!(!model_above.is_below_threshold());
    }

    #[test]
    fn test_error_accumulation() {
        let model = QecModel::new(7, 1e-3);

        let p_1 = model.error_after_cycles(1);
        let p_100 = model.error_after_cycles(100);
        let p_1000 = model.error_after_cycles(1000);

        // Error should increase with cycles
        assert!(p_1 < p_100);
        assert!(p_100 < p_1000);

        // But should still be small for good parameters
        assert!(p_1000 < 0.01);

        println!("Error after 1 cycle: {:.2e}", p_1);
        println!("Error after 100 cycles: {:.2e}", p_100);
        println!("Error after 1000 cycles: {:.2e}", p_1000);
    }

    #[test]
    fn test_required_distance() {
        // For 0.1% physical error, to get 1e-10 logical error per cycle
        let d = QecModel::required_distance(1e-3, 1e-10, 0.01);
        assert!(d >= 13); // Should need fairly high distance
        assert!(d <= 25); // But not unreasonably high
        assert!(d % 2 == 1); // Should be odd
        println!("Required distance for 1e-10: {}", d);
    }

    #[test]
    fn test_suppression_factor() {
        let model = QecModel::new(7, 1e-3);
        let suppression = model.suppression_factor();

        // For d=7, p=1e-3: suppression = 1e-3 / 1e-5 = 100×
        // Should be significant suppression (>= 10×)
        assert!(suppression >= 10.0);
        println!("Suppression factor: {:.1}×", suppression);
    }

    #[test]
    fn test_cycles_for_target() {
        let model = QecModel::new(7, 1e-3);

        // How many cycles before we hit 1% logical error?
        let cycles = model.cycles_for_target_error(0.01);
        assert!(cycles.is_some());
        let cycles = cycles.unwrap();
        assert!(cycles > 1000); // Should be many cycles
        println!("Cycles for 1% error: {}", cycles);
    }

    #[test]
    fn test_decoder_thresholds() {
        let mwpm = DecoderModel::mwpm();
        let uf = DecoderModel::union_find();

        // MWPM should have higher threshold
        assert!(mwpm.threshold() > uf.threshold());

        // Create models with each decoder
        let model_mwpm = mwpm.create_model(5, 0.005);
        let model_uf = uf.create_model(5, 0.005);

        // MWPM should give lower logical error rate
        assert!(model_mwpm.p_logical_per_cycle < model_uf.p_logical_per_cycle);
    }
}
