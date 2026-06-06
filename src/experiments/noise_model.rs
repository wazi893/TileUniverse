// =============================================================================
// src/noise_model.rs
// =============================================================================
//
// SPRINT 3.3: Quantum Noise / Decoherence Model
//
// Real quantum computers have noise. Qubits decohere. Gates aren't perfect.
// This module lets us explore: What happens to quantum advantage under noise?
//
// Hypotheses to test:
// - Quantum organisms lose advantage in high-noise scenarios
// - Optimal advantage occurs at "medium coherence" levels
// - Evolution might discover noise-resistant circuit structures
//
// Noise types modeled:
// - Phase damping (T2 decoherence)
// - Amplitude damping (T1 decay)
// - Gate errors (random rotations)
// - Measurement errors (bit flips)
//
// =============================================================================

use crate::quantum::QGate;
use std::f32::consts::PI;

// =============================================================================
// Noise Configuration
// =============================================================================

/// Configuration for quantum noise simulation.
#[derive(Clone, Debug)]
pub struct NoiseConfig {
    /// Probability of phase error per gate (0.0 to 1.0)
    /// Simulates T2 decoherence - phase information leaks to environment
    pub phase_error_rate: f32,

    /// Probability of bit flip per gate (0.0 to 1.0)
    /// Simulates T1 decay - energy leaks to environment
    pub bit_flip_rate: f32,

    /// Probability of depolarizing error per gate (0.0 to 1.0)
    /// Random Pauli error (X, Y, or Z with equal probability)
    pub depolarizing_rate: f32,

    /// Gate fidelity (0.0 to 1.0)
    /// 1.0 = perfect gates, 0.9 = 10% error per gate
    pub gate_fidelity: f32,

    /// Measurement error rate (0.0 to 1.0)
    /// Probability of reading wrong bit value
    pub measurement_error_rate: f32,

    /// Whether to apply noise to single-qubit gates
    pub noise_single_qubit: bool,

    /// Whether to apply noise to two-qubit gates (usually noisier)
    pub noise_two_qubit: bool,

    /// Multiplier for two-qubit gate noise (typically 2-10x worse)
    pub two_qubit_noise_multiplier: f32,
}

impl NoiseConfig {
    /// Perfect quantum computer (no noise).
    pub fn perfect() -> Self {
        Self {
            phase_error_rate: 0.0,
            bit_flip_rate: 0.0,
            depolarizing_rate: 0.0,
            gate_fidelity: 1.0,
            measurement_error_rate: 0.0,
            noise_single_qubit: false,
            noise_two_qubit: false,
            two_qubit_noise_multiplier: 1.0,
        }
    }

    /// Low noise (good quantum computer, ~99.9% fidelity).
    pub fn low() -> Self {
        Self {
            phase_error_rate: 0.001,
            bit_flip_rate: 0.0005,
            depolarizing_rate: 0.001,
            gate_fidelity: 0.999,
            measurement_error_rate: 0.005,
            noise_single_qubit: true,
            noise_two_qubit: true,
            two_qubit_noise_multiplier: 3.0,
        }
    }

    /// Medium noise (typical current hardware, ~99% fidelity).
    pub fn medium() -> Self {
        Self {
            phase_error_rate: 0.01,
            bit_flip_rate: 0.005,
            depolarizing_rate: 0.01,
            gate_fidelity: 0.99,
            measurement_error_rate: 0.02,
            noise_single_qubit: true,
            noise_two_qubit: true,
            two_qubit_noise_multiplier: 5.0,
        }
    }

    /// High noise (noisy hardware, ~95% fidelity).
    pub fn high() -> Self {
        Self {
            phase_error_rate: 0.05,
            bit_flip_rate: 0.02,
            depolarizing_rate: 0.05,
            gate_fidelity: 0.95,
            measurement_error_rate: 0.05,
            noise_single_qubit: true,
            noise_two_qubit: true,
            two_qubit_noise_multiplier: 5.0,
        }
    }

    /// Very high noise (experimental/educational).
    pub fn very_high() -> Self {
        Self {
            phase_error_rate: 0.1,
            bit_flip_rate: 0.05,
            depolarizing_rate: 0.1,
            gate_fidelity: 0.9,
            measurement_error_rate: 0.1,
            noise_single_qubit: true,
            noise_two_qubit: true,
            two_qubit_noise_multiplier: 3.0,
        }
    }

    /// Custom noise level (0.0 = perfect, 1.0 = maximum noise).
    pub fn scaled(noise_level: f32) -> Self {
        let level = noise_level.clamp(0.0, 1.0);

        Self {
            phase_error_rate: 0.1 * level,
            bit_flip_rate: 0.05 * level,
            depolarizing_rate: 0.1 * level,
            gate_fidelity: 1.0 - 0.1 * level,
            measurement_error_rate: 0.1 * level,
            noise_single_qubit: level > 0.0,
            noise_two_qubit: level > 0.0,
            two_qubit_noise_multiplier: 3.0 + 2.0 * level,
        }
    }

    /// Calculate effective gate fidelity for a circuit.
    pub fn effective_fidelity(&self, n_single_gates: usize, n_two_qubit_gates: usize) -> f32 {
        let single_fidelity = if self.noise_single_qubit {
            self.gate_fidelity
        } else {
            1.0
        };

        let two_q_fidelity = if self.noise_two_qubit {
            self.gate_fidelity.powf(self.two_qubit_noise_multiplier)
        } else {
            1.0
        };

        single_fidelity.powi(n_single_gates as i32) * two_q_fidelity.powi(n_two_qubit_gates as i32)
    }

    /// Amplify noise by a factor λ (for Zero-Noise Extrapolation).
    ///
    /// **SPRINT 71.0: Gate-noise-only scaling**
    ///
    /// Scales ONLY gate error rates by λ, NOT measurement errors:
    /// - ✓ Phase error rate scaled by λ
    /// - ✓ Bit flip rate scaled by λ
    /// - ✓ Depolarizing rate scaled by λ
    /// - ✓ Gate fidelity degraded by λ
    /// - ✗ Measurement error rate **unchanged**
    ///
    /// **Rationale**: This allows the same REM calibration to be used at all
    /// noise levels. REM calibration is built at λ=1.0, and since readout
    /// errors don't change with λ, the calibration remains valid for all
    /// amplified circuits. This enables the REM→ZNE pre-mitigation architecture.
    ///
    /// Error rates are clamped to [0.0, 1.0]. Gate fidelity is clamped to
    /// minimum 0.01 (99% error rate is maximum).
    ///
    /// # Example
    /// ```ignore
    /// let base = NoiseConfig::medium();
    /// let amplified = base.amplify_by(2.0);  // 2x gate noise
    ///
    /// // Gate errors amplified
    /// assert!(amplified.depolarizing_rate >= base.depolarizing_rate * 2.0);
    /// assert!(amplified.phase_error_rate >= base.phase_error_rate * 2.0);
    ///
    /// // Readout errors UNCHANGED (gate-noise-only scaling)
    /// assert_eq!(amplified.measurement_error_rate, base.measurement_error_rate);
    /// ```
    pub fn amplify_by(&self, lambda: f64) -> Self {
        Self {
            // Amplify gate errors
            phase_error_rate: (self.phase_error_rate * lambda as f32).min(1.0),
            bit_flip_rate: (self.bit_flip_rate * lambda as f32).min(1.0),
            depolarizing_rate: (self.depolarizing_rate * lambda as f32).min(1.0),
            gate_fidelity: 1.0 - ((1.0 - self.gate_fidelity) * lambda as f32).min(0.99),

            // SPRINT 71.0: Do NOT amplify measurement errors (gate-noise-only scaling)
            // This allows the same REM calibration to work at all noise levels
            measurement_error_rate: self.measurement_error_rate,

            ..*self
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self::perfect()
    }
}

// =============================================================================
// Noisy Circuit Execution
// =============================================================================

/// A simple RNG for noise application.
#[derive(Clone, Debug)]
pub struct NoiseRng {
    state: u64,
}

impl NoiseRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns true with probability p.
    pub fn chance(&mut self, p: f32) -> bool {
        (self.next_u64() as f32 / u64::MAX as f32) < p
    }

    /// Returns a random float in [0, 1).
    pub fn random(&mut self) -> f32 {
        self.next_u64() as f32 / u64::MAX as f32
    }

    /// Returns a random value in [0, max).
    pub fn gen_range(&mut self, max: u32) -> u32 {
        (self.next_u64() % max as u64) as u32
    }
}

/// Apply noise to a circuit, returning a potentially modified version.
///
/// This simulates the effect of running on noisy hardware by:
/// 1. Potentially adding error gates after each gate
/// 2. Potentially modifying gate parameters
/// 3. Potentially flipping measurement results
pub fn apply_noise_to_circuit(
    circuit: &[QGate],
    config: &NoiseConfig,
    _n_qubits: u8,
    seed: u64,
) -> Vec<QGate> {
    let mut rng = NoiseRng::new(seed);
    let mut noisy_circuit = Vec::with_capacity(circuit.len() * 2);

    for gate in circuit {
        let is_multi_qubit = matches!(gate, QGate::CNot(_, _) | QGate::CZ(_, _))
            || matches!(gate, QGate::Toffoli(_, _, _) | QGate::CCZ(_, _, _));

        let should_apply_noise = if is_multi_qubit {
            config.noise_two_qubit
        } else {
            config.noise_single_qubit
        };

        if !should_apply_noise {
            noisy_circuit.push(gate.clone());
            continue;
        }

        let noise_multiplier = if is_multi_qubit {
            config.two_qubit_noise_multiplier
        } else {
            1.0
        };

        // Apply gate (possibly with modified parameters)
        let noisy_gate = apply_gate_error(gate, config.gate_fidelity, &mut rng);
        noisy_circuit.push(noisy_gate);

        // Add potential phase error
        if rng.chance(config.phase_error_rate * noise_multiplier) {
            let qubits = gate_qubits(gate);
            if !qubits.is_empty() {
                let q = qubits[rng.gen_range(qubits.len() as u32) as usize];
                let random_phase = rng.random() * 2.0 * PI;
                noisy_circuit.push(QGate::Phase(q, random_phase));
            }
        }

        // Add potential bit flip
        if rng.chance(config.bit_flip_rate * noise_multiplier) {
            let qubits = gate_qubits(gate);
            if !qubits.is_empty() {
                let q = qubits[rng.gen_range(qubits.len() as u32) as usize];
                noisy_circuit.push(QGate::X(q));
            }
        }

        // Add potential depolarizing error (random Pauli)
        if rng.chance(config.depolarizing_rate * noise_multiplier) {
            let qubits = gate_qubits(gate);
            if !qubits.is_empty() {
                let q = qubits[rng.gen_range(qubits.len() as u32) as usize];
                match rng.gen_range(3) {
                    0 => noisy_circuit.push(QGate::X(q)),
                    1 => noisy_circuit.push(QGate::Z(q)),
                    _ => {
                        // Y = iXZ, approximate with X then Z
                        noisy_circuit.push(QGate::X(q));
                        noisy_circuit.push(QGate::Z(q));
                    }
                }
            }
        }
    }

    noisy_circuit
}

/// Apply error to a single gate (parameter noise).
fn apply_gate_error(gate: &QGate, fidelity: f32, rng: &mut NoiseRng) -> QGate {
    if fidelity >= 1.0 || !rng.chance(1.0 - fidelity) {
        return gate.clone();
    }

    // Add small random rotation to parameterized gates
    match gate {
        QGate::Phase(q, angle) => {
            let error = (rng.random() - 0.5) * PI * 0.1 * (1.0 - fidelity);
            QGate::Phase(*q, angle + error)
        }
        // For non-parameterized gates, we might add a small phase error
        // but this is handled by the phase_error_rate instead
        _ => gate.clone(),
    }
}

/// Get qubits involved in a gate.
fn gate_qubits(gate: &QGate) -> Vec<u8> {
    match gate {
        QGate::H(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::Phase(q, _)
        | QGate::Measure(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::U3(q, _, _, _)
        | QGate::T(q)
        | QGate::Tdg(q) => {
            vec![*q]
        }
        QGate::CNot(a, b) | QGate::CZ(a, b) | QGate::Swap(a, b) => vec![*a, *b],
        QGate::Toffoli(a, b, c) => vec![*a, *b, *c],
        QGate::CCZ(a, b, c) => vec![*a, *b, *c],
        // Shor's algorithm: Controlled rotation gates
        QGate::CPhase(a, b, _) | QGate::CRz(a, b, _) => vec![*a, *b],
    }
}

/// Apply measurement error to a result bitstring.
pub fn apply_measurement_error(result: u64, n_qubits: u8, config: &NoiseConfig, seed: u64) -> u64 {
    if config.measurement_error_rate <= 0.0 {
        return result;
    }

    let mut rng = NoiseRng::new(seed);
    let mut noisy_result = result;

    for q in 0..n_qubits {
        if rng.chance(config.measurement_error_rate) {
            // Flip this bit
            noisy_result ^= 1 << q;
        }
    }

    noisy_result
}

// =============================================================================
// Noise Analysis
// =============================================================================

/// Calculate the expected success probability of a circuit under noise.
pub fn expected_success_probability(circuit: &[QGate], config: &NoiseConfig) -> f32 {
    let mut n_single = 0;
    let mut n_two = 0;
    let mut n_three = 0;

    for gate in circuit {
        match gate {
            QGate::H(_)
            | QGate::X(_)
            | QGate::Y(_)
            | QGate::Z(_)
            | QGate::Phase(_, _)
            | QGate::Measure(_)
            | QGate::Rx(_, _)
            | QGate::Ry(_, _)
            | QGate::Rz(_, _)
            | QGate::U3(_, _, _, _)
            | QGate::T(_)
            | QGate::Tdg(_) => {
                n_single += 1;
            }
            QGate::CNot(_, _) | QGate::CZ(_, _) | QGate::Swap(_, _) => {
                n_two += 1;
            }
            QGate::Toffoli(_, _, _) => {
                n_three += 1;
            }
            QGate::CCZ(_, _, _) => {
                n_three += 1;
            }
            // Shor's algorithm: Controlled rotation gates
            QGate::CPhase(_, _, _) | QGate::CRz(_, _, _) => {
                n_two += 1;
            }
        }
    }

    // Compute overall fidelity
    let single_fidelity = if config.noise_single_qubit {
        (1.0 - config.phase_error_rate)
            * (1.0 - config.bit_flip_rate)
            * (1.0 - config.depolarizing_rate)
            * config.gate_fidelity
    } else {
        1.0
    };

    let multi_fidelity = if config.noise_two_qubit {
        let base = (1.0 - config.phase_error_rate * config.two_qubit_noise_multiplier)
            * (1.0 - config.bit_flip_rate * config.two_qubit_noise_multiplier)
            * (1.0 - config.depolarizing_rate * config.two_qubit_noise_multiplier)
            * config.gate_fidelity.powf(config.two_qubit_noise_multiplier);
        base
    } else {
        1.0
    };

    let circuit_fidelity =
        single_fidelity.powi(n_single as i32) * multi_fidelity.powi((n_two + n_three) as i32);

    let measurement_fidelity = (1.0 - config.measurement_error_rate).powi(4); // Assuming 4 qubits

    circuit_fidelity * measurement_fidelity
}

/// Categorize noise resilience of a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseResilience {
    /// Highly resilient (short, few entangling gates)
    High,
    /// Moderately resilient
    Medium,
    /// Low resilience (long, many entangling gates)
    Low,
    /// Very fragile (will likely fail under any noise)
    Fragile,
}

pub fn categorize_resilience(circuit: &[QGate], config: &NoiseConfig) -> NoiseResilience {
    let success_prob = expected_success_probability(circuit, config);

    if success_prob > 0.9 {
        NoiseResilience::High
    } else if success_prob > 0.7 {
        NoiseResilience::Medium
    } else if success_prob > 0.5 {
        NoiseResilience::Low
    } else {
        NoiseResilience::Fragile
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_config_perfect() {
        let config = NoiseConfig::perfect();
        assert_eq!(config.gate_fidelity, 1.0);
        assert_eq!(config.phase_error_rate, 0.0);
    }

    #[test]
    fn test_noise_config_scaled() {
        let zero = NoiseConfig::scaled(0.0);
        assert_eq!(zero.gate_fidelity, 1.0);

        let full = NoiseConfig::scaled(1.0);
        assert!(full.gate_fidelity < 1.0);
        assert!(full.phase_error_rate > 0.0);
    }

    #[test]
    fn test_apply_noise_perfect() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let config = NoiseConfig::perfect();

        let noisy = apply_noise_to_circuit(&circuit, &config, 4, 12345);

        // Perfect noise = no change
        assert_eq!(noisy.len(), circuit.len());
    }

    #[test]
    fn test_apply_noise_high() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let config = NoiseConfig::high();

        // Run many times, should sometimes add error gates
        let mut had_errors = false;
        for seed in 0..100 {
            let noisy = apply_noise_to_circuit(&circuit, &config, 4, seed);
            if noisy.len() > circuit.len() {
                had_errors = true;
                break;
            }
        }

        assert!(had_errors, "High noise should sometimes add error gates");
    }

    #[test]
    fn test_measurement_error() {
        let config = NoiseConfig {
            measurement_error_rate: 1.0, // Always flip
            ..NoiseConfig::perfect()
        };

        let original = 0b0000;
        let noisy = apply_measurement_error(original, 4, &config, 12345);

        // With 100% error rate, at least some bits should flip
        assert_ne!(original, noisy);
    }

    #[test]
    fn test_expected_success_probability() {
        let short_circuit = vec![QGate::H(0)];
        let long_circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
        ];

        let config = NoiseConfig::medium();

        let short_prob = expected_success_probability(&short_circuit, &config);
        let long_prob = expected_success_probability(&long_circuit, &config);

        assert!(
            short_prob > long_prob,
            "Shorter circuits should have higher success probability"
        );
    }

    #[test]
    fn test_categorize_resilience() {
        let short = vec![QGate::H(0)];
        let long = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::H(1),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
        ];

        let config = NoiseConfig::medium();

        let short_resilience = categorize_resilience(&short, &config);
        let long_resilience = categorize_resilience(&long, &config);

        // Short circuits should be at least as resilient as long ones
        // High=0, Medium=1, Low=2, Fragile=3 (lower is better)
        assert!(
            short_resilience as u8 <= long_resilience as u8,
            "Short circuit resilience {:?} should be >= long circuit {:?}",
            short_resilience,
            long_resilience
        );
    }

    #[test]
    fn test_effective_fidelity() {
        let config = NoiseConfig::medium();

        let fidelity_0 = config.effective_fidelity(0, 0);
        let fidelity_10 = config.effective_fidelity(10, 0);
        let fidelity_10_5 = config.effective_fidelity(10, 5);

        assert!(fidelity_0 > fidelity_10);
        assert!(fidelity_10 > fidelity_10_5);
    }

    #[test]
    fn test_amplify_by_gate_noise_only() {
        // SPRINT 71.0: Verify gate-noise-only scaling
        let base = NoiseConfig::medium();
        let amplified = base.amplify_by(2.0);

        // Gate errors should be amplified
        assert!(
            amplified.phase_error_rate >= base.phase_error_rate * 1.99,
            "Phase error should be amplified: {} vs {}",
            amplified.phase_error_rate,
            base.phase_error_rate
        );
        assert!(
            amplified.bit_flip_rate >= base.bit_flip_rate * 1.99,
            "Bit flip should be amplified: {} vs {}",
            amplified.bit_flip_rate,
            base.bit_flip_rate
        );
        assert!(
            amplified.depolarizing_rate >= base.depolarizing_rate * 1.99,
            "Depolarizing should be amplified: {} vs {}",
            amplified.depolarizing_rate,
            base.depolarizing_rate
        );
        assert!(
            amplified.gate_fidelity < base.gate_fidelity,
            "Gate fidelity should decrease: {} vs {}",
            amplified.gate_fidelity,
            base.gate_fidelity
        );

        // Measurement error should NOT be amplified (gate-noise-only)
        assert_eq!(
            amplified.measurement_error_rate, base.measurement_error_rate,
            "Measurement error should be UNCHANGED for gate-noise-only scaling"
        );
    }

    #[test]
    fn test_amplify_by_preserves_rem_calibration() {
        // SPRINT 71.0: Verify that REM calibration can be shared across noise levels
        let base = NoiseConfig::medium();

        // Multiple amplification levels
        let lambda_1 = base.amplify_by(1.0);
        let lambda_2 = base.amplify_by(2.0);
        let lambda_3 = base.amplify_by(3.0);

        // All should have the same measurement error rate
        assert_eq!(lambda_1.measurement_error_rate, base.measurement_error_rate);
        assert_eq!(lambda_2.measurement_error_rate, base.measurement_error_rate);
        assert_eq!(lambda_3.measurement_error_rate, base.measurement_error_rate);

        // But gate errors should differ
        assert!(lambda_2.depolarizing_rate > lambda_1.depolarizing_rate);
        assert!(lambda_3.depolarizing_rate > lambda_2.depolarizing_rate);
    }
}
