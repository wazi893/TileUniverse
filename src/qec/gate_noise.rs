//! Circuit-Level Noise Model (Phase 78.2)
//!
//! Provides per-gate error injection for realistic QEC simulation.
//! Implements explicit Pauli channels for depolarizing noise.
//!
//! # Noise Channels
//!
//! ## Single-Qubit Depolarizing Channel
//! With probability p_single, applies X, Y, or Z (each with probability p_single/3).
//! The channel is: E(rho) = (1-p) * rho + (p/3) * (X*rho*X + Y*rho*Y + Z*rho*Z)
//!
//! ## Two-Qubit Depolarizing Channel
//! With probability p_two, applies one of 15 non-identity Pauli pairs (each with p_two/15).
//! The 15 Paulis are: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
//!
//! ## Measurement Error
//! With probability p_meas, the classical measurement outcome is flipped.
//!
//! ## State Preparation Error
//! With probability p_prep, the initial state preparation has a bit-flip.

use super::noise::SimpleRng;

/// Pauli operator for error injection.
///
/// Represents single-qubit Pauli operators used in depolarizing channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Identity (no error)
    I,
    /// Pauli X (bit-flip)
    X,
    /// Pauli Y (bit-flip + phase-flip)
    Y,
    /// Pauli Z (phase-flip)
    Z,
}

impl Pauli {
    /// Convert to index (0=I, 1=X, 2=Y, 3=Z)
    #[inline]
    pub fn as_index(self) -> usize {
        match self {
            Pauli::I => 0,
            Pauli::X => 1,
            Pauli::Y => 2,
            Pauli::Z => 3,
        }
    }

    /// Create from index (0=I, 1=X, 2=Y, 3=Z)
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        match idx % 4 {
            0 => Pauli::I,
            1 => Pauli::X,
            2 => Pauli::Y,
            _ => Pauli::Z,
        }
    }
}

/// Per-gate error model with explicit Pauli channel documentation.
///
/// This model supports depolarizing noise on single-qubit and two-qubit gates,
/// as well as measurement and state preparation errors.
///
/// # Error Rates
///
/// - `p_single`: Single-qubit depolarizing rate. With probability p_single,
///   X, Y, or Z is applied (each with p_single/3).
/// - `p_two`: Two-qubit depolarizing rate. With probability p_two,
///   one of 15 non-II Pauli pairs is applied (each with p_two/15).
/// - `p_meas`: Measurement bit-flip probability on classical outcome.
/// - `p_prep`: State preparation bit-flip probability.
/// - `p_idle`: Idle error rate per cycle.
///
/// # Presets
///
/// The module provides ballpark error models for common hardware:
/// - `superconducting_ballpark()`: ~0.1% single, ~1% two-qubit
/// - `ion_trap_ballpark()`: ~0.01% single, ~0.1% two-qubit
///
/// Note: These are illustrative values, not canonical hardware specifications.
#[derive(Clone, Debug)]
pub struct GateErrorModel {
    /// Single-qubit depolarizing: X/Y/Z each with p_single/3
    pub p_single: f64,
    /// Two-qubit depolarizing: 15 non-identity Paulis each with p_two/15
    pub p_two: f64,
    /// Measurement: classical bit-flip on outcome
    pub p_meas: f64,
    /// State prep: classical bit-flip on initial state
    pub p_prep: f64,
    /// Idle error per "cycle" (units defined by cycle_duration_ns)
    pub p_idle: f64,
    /// What is a "cycle"? None = arbitrary units
    pub cycle_duration_ns: Option<f64>,
}

impl GateErrorModel {
    /// No errors (for baseline comparison).
    ///
    /// Use this model to verify circuit behavior without noise.
    pub fn ideal() -> Self {
        Self {
            p_single: 0.0,
            p_two: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
            p_idle: 0.0,
            cycle_duration_ns: None,
        }
    }

    /// Ballpark superconducting qubit error rates.
    ///
    /// These are illustrative values based on typical superconducting hardware:
    /// - Single-qubit gates: ~0.1% error rate
    /// - Two-qubit gates: ~1% error rate
    /// - Measurement: ~1% error rate
    /// - State preparation: ~0.1% error rate
    /// - Idle: ~0.01% per 20ns cycle
    ///
    /// NOTE: These are approximate values, not canonical hardware specifications.
    pub fn superconducting_ballpark() -> Self {
        Self {
            p_single: 0.001,
            p_two: 0.01,
            p_meas: 0.01,
            p_prep: 0.001,
            p_idle: 0.0001,
            cycle_duration_ns: Some(20.0),
        }
    }

    /// Ballpark ion trap qubit error rates.
    ///
    /// These are illustrative values based on typical trapped-ion hardware:
    /// - Single-qubit gates: ~0.01% error rate
    /// - Two-qubit gates: ~0.1% error rate
    /// - Measurement: ~0.1% error rate
    /// - State preparation: ~0.01% error rate
    /// - Idle: ~0.001% per 100us cycle
    ///
    /// NOTE: These are approximate values, not canonical hardware specifications.
    pub fn ion_trap_ballpark() -> Self {
        Self {
            p_single: 0.0001,
            p_two: 0.001,
            p_meas: 0.001,
            p_prep: 0.0001,
            p_idle: 0.00001,
            cycle_duration_ns: Some(100_000.0),
        }
    }

    /// Create a custom error model with specified rates.
    pub fn custom(p_single: f64, p_two: f64, p_meas: f64, p_prep: f64, p_idle: f64) -> Self {
        Self {
            p_single,
            p_two,
            p_meas,
            p_prep,
            p_idle,
            cycle_duration_ns: None,
        }
    }

    /// Scale all error rates by a factor.
    ///
    /// Useful for threshold studies where you want to sweep error rates.
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            p_single: self.p_single * factor,
            p_two: self.p_two * factor,
            p_meas: self.p_meas * factor,
            p_prep: self.p_prep * factor,
            p_idle: self.p_idle * factor,
            cycle_duration_ns: self.cycle_duration_ns,
        }
    }

    /// Set the cycle duration for idle error calculations.
    pub fn with_cycle_duration(mut self, ns: f64) -> Self {
        self.cycle_duration_ns = Some(ns);
        self
    }
}

impl Default for GateErrorModel {
    fn default() -> Self {
        Self::ideal()
    }
}

/// Error result from gate execution.
///
/// Describes what error (if any) occurred during a noisy gate operation.
#[derive(Clone, Debug, PartialEq)]
pub enum GateError {
    /// No error occurred
    None,
    /// Single-qubit error: Pauli applied to one qubit
    Single {
        /// The qubit that experienced the error
        qubit: usize,
        /// The Pauli error that was applied
        pauli: Pauli,
    },
    /// Two-qubit error: Paulis applied to both qubits
    TwoQubit {
        /// First qubit
        q1: usize,
        /// Pauli on first qubit
        p1: Pauli,
        /// Second qubit
        q2: usize,
        /// Pauli on second qubit
        p2: Pauli,
    },
    /// Measurement error: classical bit was flipped
    Measurement {
        /// The qubit that was measured
        qubit: usize,
    },
    /// State preparation error: initial state had bit-flip
    Preparation {
        /// The qubit that was prepared
        qubit: usize,
    },
}

impl GateError {
    /// Check if this is a "no error" result
    pub fn is_none(&self) -> bool {
        matches!(self, GateError::None)
    }

    /// Check if this is any kind of error
    pub fn is_error(&self) -> bool {
        !self.is_none()
    }
}

/// Sample single-qubit depolarizing error.
///
/// With probability p: returns X, Y, or Z (each with p/3).
/// With probability 1-p: returns None (identity).
///
/// # Arguments
/// * `p` - Total depolarizing probability
/// * `rng` - Random number generator
///
/// # Returns
/// `Some(Pauli)` if an error occurred, `None` if no error.
pub fn sample_single_qubit_error(p: f64, rng: &mut SimpleRng) -> Option<Pauli> {
    let r = rng.next_f64();
    if r >= p {
        return None;
    }
    // Which Pauli? Each has probability p/3
    let choice = (rng.next_f64() * 3.0) as usize;
    Some(match choice {
        0 => Pauli::X,
        1 => Pauli::Y,
        _ => Pauli::Z,
    })
}

/// Sample two-qubit depolarizing error.
///
/// With probability p: returns one of 15 non-II Pauli pairs (each with p/15).
/// With probability 1-p: returns None (identity on both qubits).
///
/// The 15 non-identity Pauli pairs are:
/// IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
///
/// # Arguments
/// * `p` - Total depolarizing probability
/// * `rng` - Random number generator
///
/// # Returns
/// `Some((Pauli, Pauli))` if an error occurred, `None` if no error.
pub fn sample_two_qubit_error(p: f64, rng: &mut SimpleRng) -> Option<(Pauli, Pauli)> {
    let r = rng.next_f64();
    if r >= p {
        return None;
    }
    // 15 non-identity two-qubit Paulis: all (P1, P2) except (I, I)
    // Index 0-14 maps to these
    let choice = (rng.next_f64() * 15.0) as usize;
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

    // Enumerate: skip II, then IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
    let mut idx = 0;
    for &p1 in &paulis {
        for &p2 in &paulis {
            if p1 == Pauli::I && p2 == Pauli::I {
                continue;
            }
            if idx == choice {
                return Some((p1, p2));
            }
            idx += 1;
        }
    }
    // Fallback (shouldn't reach)
    Some((Pauli::X, Pauli::X))
}

/// Sample measurement error (bit flip).
///
/// With probability p: returns true (flip the measurement outcome).
/// With probability 1-p: returns false (no flip).
///
/// # Arguments
/// * `p` - Bit-flip probability
/// * `rng` - Random number generator
///
/// # Returns
/// `true` if the measurement outcome should be flipped.
pub fn sample_measurement_error(p: f64, rng: &mut SimpleRng) -> bool {
    rng.next_f64() < p
}

/// Sample state preparation error.
///
/// With probability p: returns true (preparation had bit-flip).
/// With probability 1-p: returns false (no error).
///
/// # Arguments
/// * `p` - Bit-flip probability
/// * `rng` - Random number generator
///
/// # Returns
/// `true` if the state preparation had an error.
pub fn sample_preparation_error(p: f64, rng: &mut SimpleRng) -> bool {
    rng.next_f64() < p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_ideal_model_produces_no_errors() {
        let model = GateErrorModel::ideal();
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            assert!(sample_single_qubit_error(model.p_single, &mut rng).is_none());
            assert!(sample_two_qubit_error(model.p_two, &mut rng).is_none());
            assert!(!sample_measurement_error(model.p_meas, &mut rng));
            assert!(!sample_preparation_error(model.p_prep, &mut rng));
        }
    }

    #[test]
    fn test_single_qubit_channel_distribution() {
        let mut rng = SimpleRng::new(42);
        let p = 0.3;
        let n = 10000;
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;
        let mut none_count = 0;

        for _ in 0..n {
            match sample_single_qubit_error(p, &mut rng) {
                Some(Pauli::X) => x_count += 1,
                Some(Pauli::Y) => y_count += 1,
                Some(Pauli::Z) => z_count += 1,
                Some(Pauli::I) => panic!("Identity should not be returned as error"),
                None => none_count += 1,
            }
        }

        // Should have ~70% None, ~10% each X/Y/Z
        let none_frac = none_count as f64 / n as f64;
        assert!(
            (none_frac - 0.7).abs() < 0.05,
            "None fraction: {} (expected ~0.7)",
            none_frac
        );

        let x_frac = x_count as f64 / n as f64;
        let y_frac = y_count as f64 / n as f64;
        let z_frac = z_count as f64 / n as f64;
        // Each should be ~p/3 = 0.1
        assert!(
            (x_frac - 0.1).abs() < 0.03,
            "X fraction: {} (expected ~0.1)",
            x_frac
        );
        assert!(
            (y_frac - 0.1).abs() < 0.03,
            "Y fraction: {} (expected ~0.1)",
            y_frac
        );
        assert!(
            (z_frac - 0.1).abs() < 0.03,
            "Z fraction: {} (expected ~0.1)",
            z_frac
        );
    }

    #[test]
    fn test_two_qubit_channel_has_15_outcomes() {
        let mut rng = SimpleRng::new(42);
        let p = 1.0; // Always error for testing distribution
        let mut outcomes = HashSet::new();

        for _ in 0..10000 {
            if let Some(pair) = sample_two_qubit_error(p, &mut rng) {
                outcomes.insert((pair.0 as u8, pair.1 as u8));
            }
        }

        // Should see all 15 non-II combinations
        assert_eq!(
            outcomes.len(),
            15,
            "Should have 15 distinct outcomes, got {}",
            outcomes.len()
        );
        // II should not be in outcomes
        assert!(
            !outcomes.contains(&(Pauli::I as u8, Pauli::I as u8)),
            "II should not be in outcomes"
        );
    }

    #[test]
    fn test_two_qubit_channel_distribution() {
        let mut rng = SimpleRng::new(42);
        let p = 0.45; // 45% error rate
        let n = 15000;
        let mut none_count = 0;

        for _ in 0..n {
            if sample_two_qubit_error(p, &mut rng).is_none() {
                none_count += 1;
            }
        }

        // Should have ~55% None
        let none_frac = none_count as f64 / n as f64;
        assert!(
            (none_frac - 0.55).abs() < 0.05,
            "None fraction: {} (expected ~0.55)",
            none_frac
        );
    }

    #[test]
    fn test_measurement_error_rate() {
        let mut rng = SimpleRng::new(42);
        let p = 0.25;
        let n = 10000;
        let mut flip_count = 0;

        for _ in 0..n {
            if sample_measurement_error(p, &mut rng) {
                flip_count += 1;
            }
        }

        let flip_frac = flip_count as f64 / n as f64;
        assert!(
            (flip_frac - 0.25).abs() < 0.03,
            "Flip fraction: {} (expected ~0.25)",
            flip_frac
        );
    }

    #[test]
    fn test_scaled_model() {
        let base = GateErrorModel::superconducting_ballpark();
        let scaled = base.scaled(2.0);

        assert!((scaled.p_single - 2.0 * base.p_single).abs() < 1e-10);
        assert!((scaled.p_two - 2.0 * base.p_two).abs() < 1e-10);
        assert!((scaled.p_meas - 2.0 * base.p_meas).abs() < 1e-10);
        assert!((scaled.p_prep - 2.0 * base.p_prep).abs() < 1e-10);
        assert!((scaled.p_idle - 2.0 * base.p_idle).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_index_conversion() {
        assert_eq!(Pauli::I.as_index(), 0);
        assert_eq!(Pauli::X.as_index(), 1);
        assert_eq!(Pauli::Y.as_index(), 2);
        assert_eq!(Pauli::Z.as_index(), 3);

        assert_eq!(Pauli::from_index(0), Pauli::I);
        assert_eq!(Pauli::from_index(1), Pauli::X);
        assert_eq!(Pauli::from_index(2), Pauli::Y);
        assert_eq!(Pauli::from_index(3), Pauli::Z);
    }

    #[test]
    fn test_gate_error_is_none() {
        assert!(GateError::None.is_none());
        assert!(!GateError::None.is_error());

        let single_err = GateError::Single {
            qubit: 0,
            pauli: Pauli::X,
        };
        assert!(!single_err.is_none());
        assert!(single_err.is_error());

        let two_err = GateError::TwoQubit {
            q1: 0,
            p1: Pauli::X,
            q2: 1,
            p2: Pauli::Z,
        };
        assert!(!two_err.is_none());
        assert!(two_err.is_error());
    }

    #[test]
    fn test_custom_model() {
        let model = GateErrorModel::custom(0.01, 0.05, 0.02, 0.005, 0.001);
        assert!((model.p_single - 0.01).abs() < 1e-10);
        assert!((model.p_two - 0.05).abs() < 1e-10);
        assert!((model.p_meas - 0.02).abs() < 1e-10);
        assert!((model.p_prep - 0.005).abs() < 1e-10);
        assert!((model.p_idle - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_superconducting_preset() {
        let model = GateErrorModel::superconducting_ballpark();
        assert!((model.p_single - 0.001).abs() < 1e-10);
        assert!((model.p_two - 0.01).abs() < 1e-10);
        assert_eq!(model.cycle_duration_ns, Some(20.0));
    }

    #[test]
    fn test_ion_trap_preset() {
        let model = GateErrorModel::ion_trap_ballpark();
        assert!((model.p_single - 0.0001).abs() < 1e-10);
        assert!((model.p_two - 0.001).abs() < 1e-10);
        assert_eq!(model.cycle_duration_ns, Some(100_000.0));
    }
}
