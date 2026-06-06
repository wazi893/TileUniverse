//! Noise Models for QEC Simulation
//!
//! Provides common noise channels for testing error correction.
//! Uses a simple LCG-based RNG to avoid external dependencies.

use super::stabilizer::StabilizerTableau;

/// Simple linear congruential generator for reproducible noise
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next u64
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate f64 in [0, 1)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate usize in [0, max)
    #[inline]
    pub fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Depolarizing noise channel
///
/// With probability p, applies a random Pauli (X, Y, or Z) to each qubit.
pub struct DepolarizingNoise {
    /// Error probability per qubit
    pub p: f64,
}

impl DepolarizingNoise {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        Self { p }
    }

    /// Apply noise to all qubits in the tableau
    pub fn apply(&self, tableau: &mut StabilizerTableau, rng: &mut SimpleRng) {
        for q in 0..tableau.n_qubits() {
            if rng.next_f64() < self.p {
                // Random Pauli: 0=X, 1=Y, 2=Z
                match rng.next_usize(3) {
                    0 => tableau.pauli_x(q),
                    1 => tableau.pauli_y(q),
                    _ => tableau.pauli_z(q),
                }
            }
        }
    }

    /// Apply noise to specific qubit
    pub fn apply_to_qubit(
        &self,
        tableau: &mut StabilizerTableau,
        qubit: usize,
        rng: &mut SimpleRng,
    ) {
        if rng.next_f64() < self.p {
            match rng.next_usize(3) {
                0 => tableau.pauli_x(qubit),
                1 => tableau.pauli_y(qubit),
                _ => tableau.pauli_z(qubit),
            }
        }
    }
}

/// Bit-flip noise (X errors only)
pub struct BitFlipNoise {
    pub p: f64,
}

impl BitFlipNoise {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        Self { p }
    }

    pub fn apply(&self, tableau: &mut StabilizerTableau, rng: &mut SimpleRng) {
        for q in 0..tableau.n_qubits() {
            if rng.next_f64() < self.p {
                tableau.pauli_x(q);
            }
        }
    }
}

/// Phase-flip noise (Z errors only)
pub struct PhaseFlipNoise {
    pub p: f64,
}

impl PhaseFlipNoise {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        Self { p }
    }

    pub fn apply(&self, tableau: &mut StabilizerTableau, rng: &mut SimpleRng) {
        for q in 0..tableau.n_qubits() {
            if rng.next_f64() < self.p {
                tableau.pauli_z(q);
            }
        }
    }
}

/// Measurement error model
///
/// With probability p, flips the measurement outcome.
pub struct MeasurementError {
    pub p: f64,
}

impl MeasurementError {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        Self { p }
    }

    /// Apply error to measurement outcome
    pub fn apply(&self, outcome: u8, rng: &mut SimpleRng) -> u8 {
        if rng.next_f64() < self.p {
            1 - outcome
        } else {
            outcome
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing() {
        let mut rng = SimpleRng::new(42);
        let mut tab = StabilizerTableau::new(10);
        let noise = DepolarizingNoise::new(0.1);

        // Just verify it runs without panic
        noise.apply(&mut tab, &mut rng);
    }

    #[test]
    fn test_bit_flip() {
        let mut rng = SimpleRng::new(42);
        let mut tab = StabilizerTableau::new(10);
        let noise = BitFlipNoise::new(0.5);

        noise.apply(&mut tab, &mut rng);
    }
}
