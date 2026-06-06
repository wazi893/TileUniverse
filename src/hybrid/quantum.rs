//! Quantum substrate implementation.
//!
//! Wraps quantum state simulation for circuit execution within
//! the hybrid orchestration framework.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tile8::sparse_quantum_bigint::Complex64;

use super::job::{Gate, JobSpec, MeasurementBasis, MeasurementSpec, QuantumCircuit, QuantumSpec};
use super::result::{JobStage, StageMetrics, StageOutputs, StageResult};
use super::substrate::{
    Substrate, SubstrateCapabilities, SubstrateError, SubstrateId, SubstrateType,
};

/// Quantum simulation substrate.
///
/// This substrate handles quantum circuit execution including:
/// - State vector simulation
/// - Gate application
/// - Measurement and sampling
pub struct QuantumSubstrate {
    /// Unique identifier
    id: SubstrateId,

    /// Human-readable name
    name: String,

    /// Maximum qubits supported
    max_qubits: usize,

    /// Current capacity (0.0 = busy, 1.0 = idle)
    capacity: f64,

    /// Whether GPU acceleration is available
    gpu_available: bool,

    /// Supported gate types
    supported_gates: Vec<String>,

    /// Current state vector (for persistent state across stages)
    state: Option<Vec<Complex64>>,

    /// RNG seed for measurements
    rng_state: u64,
}

impl QuantumSubstrate {
    /// Create a new quantum substrate.
    pub fn new(id: u64, name: &str, max_qubits: usize) -> Self {
        Self {
            id: SubstrateId(id),
            name: name.to_string(),
            max_qubits,
            capacity: 1.0,
            gpu_available: false,
            supported_gates: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "S".to_string(),
                "T".to_string(),
                "Rx".to_string(),
                "Ry".to_string(),
                "Rz".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
                "SWAP".to_string(),
            ],
            state: None,
            rng_state: 12345,
        }
    }

    /// Create a substrate with GPU acceleration.
    pub fn with_gpu(mut self) -> Self {
        self.gpu_available = true;
        self
    }

    /// Set the RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Initialize state vector to |0...0⟩.
    fn init_state(&mut self, n_qubits: usize) {
        let size = 1 << n_qubits;
        let mut state = vec![Complex64 { re: 0.0, im: 0.0 }; size];
        state[0] = Complex64 { re: 1.0, im: 0.0 };
        self.state = Some(state);
    }

    /// Apply a single gate to the state.
    fn apply_gate(&mut self, gate: &Gate) -> Result<(), SubstrateError> {
        let state = self
            .state
            .as_mut()
            .ok_or(SubstrateError::Internal("No state initialized".to_string()))?;

        match gate {
            Gate::H(q) => apply_h(state, *q),
            Gate::X(q) => apply_x(state, *q),
            Gate::Y(q) => apply_y(state, *q),
            Gate::Z(q) => apply_z(state, *q),
            Gate::S(q) => apply_s(state, *q),
            Gate::T(q) => apply_t(state, *q),
            Gate::Rx(q, theta) => apply_rx(state, *q, *theta),
            Gate::Ry(q, theta) => apply_ry(state, *q, *theta),
            Gate::Rz(q, theta) => apply_rz(state, *q, *theta),
            Gate::CNOT(ctrl, tgt) => apply_cnot(state, *ctrl, *tgt),
            Gate::CZ(ctrl, tgt) => apply_cz(state, *ctrl, *tgt),
            Gate::SWAP(q1, q2) => apply_swap(state, *q1, *q2),
            Gate::CRz(ctrl, tgt, theta) => apply_crz(state, *ctrl, *tgt, *theta),
            Gate::Measure(q) => {
                // Mid-circuit measurement - collapse qubit
                self.measure_qubit(*q)?;
            }
            Gate::RxParam(_, _) | Gate::RyParam(_, _) | Gate::RzParam(_, _) => {
                return Err(SubstrateError::InvalidSpec(
                    "Parameterized gates must be bound before execution".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Execute a circuit on the state.
    fn execute_circuit(&mut self, circuit: &QuantumCircuit) -> Result<(), SubstrateError> {
        if circuit.n_qubits > self.max_qubits {
            return Err(SubstrateError::InvalidSpec(format!(
                "Circuit requires {} qubits, substrate supports {}",
                circuit.n_qubits, self.max_qubits
            )));
        }

        self.init_state(circuit.n_qubits);

        for gate in &circuit.gates {
            self.apply_gate(gate)?;
        }

        Ok(())
    }

    /// Measure a single qubit (collapse).
    fn measure_qubit(&mut self, qubit: usize) -> Result<u8, SubstrateError> {
        // Get random number first to avoid borrow conflict
        let random = self.next_random();

        let state = self
            .state
            .as_mut()
            .ok_or(SubstrateError::Internal("No state initialized".to_string()))?;

        let n_qubits = (state.len() as f64).log2() as usize;
        if qubit >= n_qubits {
            return Err(SubstrateError::InvalidSpec(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        // Calculate probability of measuring |0⟩
        let mut prob_0 = 0.0;
        for (i, amp) in state.iter().enumerate() {
            if (i >> qubit) & 1 == 0 {
                prob_0 += amp.re * amp.re + amp.im * amp.im;
            }
        }

        // Sample outcome using pre-generated random number
        let outcome = if random < prob_0 { 0 } else { 1 };

        // Collapse state
        let norm = if outcome == 0 {
            prob_0.sqrt()
        } else {
            (1.0 - prob_0).sqrt()
        };

        for (i, amp) in state.iter_mut().enumerate() {
            if ((i >> qubit) & 1) as u8 == outcome {
                amp.re /= norm;
                amp.im /= norm;
            } else {
                amp.re = 0.0;
                amp.im = 0.0;
            }
        }

        Ok(outcome)
    }

    /// Measure all qubits and return bitstring.
    #[allow(dead_code)]
    fn measure_all(&mut self, n_qubits: usize) -> Result<String, SubstrateError> {
        let mut result = String::with_capacity(n_qubits);
        for q in (0..n_qubits).rev() {
            result.push(if self.measure_qubit(q)? == 0 {
                '0'
            } else {
                '1'
            });
        }
        Ok(result)
    }

    /// Sample from the state distribution.
    fn sample(
        &mut self,
        n_qubits: usize,
        shots: usize,
    ) -> Result<HashMap<String, usize>, SubstrateError> {
        let state = self
            .state
            .as_ref()
            .ok_or(SubstrateError::Internal("No state initialized".to_string()))?;

        // Build probability distribution
        let probs: Vec<f64> = state.iter().map(|a| a.re * a.re + a.im * a.im).collect();

        let mut counts = HashMap::new();

        for _ in 0..shots {
            // Sample from distribution
            let mut cumulative = 0.0;
            let random = self.next_random();
            let mut outcome = probs.len() - 1;

            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if random < cumulative {
                    outcome = i;
                    break;
                }
            }

            // Convert to bitstring
            let bitstring = format!("{:0width$b}", outcome, width = n_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(counts)
    }

    /// Generate next random number (0.0 to 1.0).
    fn next_random(&mut self) -> f64 {
        // Xorshift64
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Execute a quantum job specification.
    fn execute_quantum(&mut self, spec: &QuantumSpec) -> Result<StageOutputs, SubstrateError> {
        // Set seed if provided
        if let Some(seed) = spec.seed {
            self.rng_state = seed;
        }

        // Execute circuit
        self.execute_circuit(&spec.circuit)?;

        // Handle measurement
        match &spec.measurement {
            MeasurementSpec::Computational => {
                // Sample from the distribution
                let counts = self.sample(spec.n_qubits, spec.shots)?;
                Ok(StageOutputs::MeasurementCounts(counts))
            }
            MeasurementSpec::Partial(qubits) => {
                // Sample only specified qubits
                // For simplicity, we measure specified qubits and return counts
                let mut counts = HashMap::new();
                for _ in 0..spec.shots {
                    let mut bitstring = String::new();
                    for &q in qubits.iter().rev() {
                        bitstring.push(if self.measure_qubit(q)? == 0 {
                            '0'
                        } else {
                            '1'
                        });
                    }
                    *counts.entry(bitstring).or_insert(0) += 1;
                    // Reinitialize for next shot (since measurement collapsed state)
                    self.execute_circuit(&spec.circuit)?;
                }
                Ok(StageOutputs::MeasurementCounts(counts))
            }
            MeasurementSpec::None => {
                // Return state vector
                let state = self.state.take().unwrap_or_default();
                Ok(StageOutputs::StateVector(state))
            }
            MeasurementSpec::Custom(bases) => {
                // Rotate to appropriate basis before measuring
                // For each qubit, apply appropriate rotation
                for (q, basis) in bases.iter().enumerate() {
                    match basis {
                        MeasurementBasis::Z => {} // No rotation needed
                        MeasurementBasis::X => {
                            self.apply_gate(&Gate::H(q))?;
                        }
                        MeasurementBasis::Y => {
                            // Rotate Y -> Z
                            self.apply_gate(&Gate::S(q))?;
                            self.apply_gate(&Gate::H(q))?;
                        }
                    }
                }
                let counts = self.sample(spec.n_qubits, spec.shots)?;
                Ok(StageOutputs::MeasurementCounts(counts))
            }
        }
    }
}

impl Substrate for QuantumSubstrate {
    fn id(&self) -> SubstrateId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn substrate_type(&self) -> SubstrateType {
        SubstrateType::Quantum
    }

    fn capacity(&self) -> f64 {
        self.capacity
    }

    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            max_qubits: Some(self.max_qubits),
            max_depth: None,
            gpu_available: self.gpu_available,
            max_parallel_jobs: 1, // State vector simulation is single-threaded
            supported_gates: self.supported_gates.clone(),
            supports_variational: false, // Classical handles optimization
            supports_observer: false,
        }
    }

    fn can_handle(&self, spec: &JobSpec) -> bool {
        match spec {
            JobSpec::Quantum(q) => q.n_qubits <= self.max_qubits,
            _ => false,
        }
    }

    fn estimate_time(&self, spec: &JobSpec) -> Duration {
        match spec {
            JobSpec::Quantum(quantum) => {
                // Rough estimate based on gate count and qubit count
                let state_size = 1u64 << quantum.n_qubits;
                let gate_count = quantum.circuit.gate_count();
                let shots = quantum.shots;

                // Time scales with state_size * gates + shots
                let ops = state_size * gate_count as u64 + shots as u64 * state_size;
                Duration::from_nanos(ops / 1000) // ~1000 ops per nanosecond
            }
            _ => Duration::ZERO,
        }
    }

    fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError> {
        self.capacity = 0.0; // Mark as busy

        let start = Instant::now();

        let result = match &stage.spec {
            JobSpec::Quantum(spec) => self.execute_quantum(spec),
            _ => Err(SubstrateError::UnsupportedJob(
                "QuantumSubstrate only handles Quantum jobs".to_string(),
            )),
        };

        let duration = start.elapsed();
        self.capacity = 1.0; // Mark as idle

        match result {
            Ok(outputs) => {
                let metrics = if let JobSpec::Quantum(spec) = &stage.spec {
                    StageMetrics::quantum(
                        spec.circuit.gate_count(),
                        estimate_depth(&spec.circuit),
                        spec.shots,
                        spec.n_qubits,
                    )
                } else {
                    StageMetrics::default()
                };
                Ok(StageResult::success(stage.index, duration, outputs).with_metrics(metrics))
            }
            Err(e) => Ok(StageResult::failure(stage.index, duration, e.to_string())),
        }
    }

    fn reset(&mut self) {
        self.state = None;
        self.capacity = 1.0;
    }
}

// Gate implementations

fn apply_h(state: &mut [Complex64], qubit: usize) {
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) == 0 {
            let j = i | mask;
            let a = state[i];
            let b = state[j];
            state[i] = Complex64 {
                re: sqrt2_inv * (a.re + b.re),
                im: sqrt2_inv * (a.im + b.im),
            };
            state[j] = Complex64 {
                re: sqrt2_inv * (a.re - b.re),
                im: sqrt2_inv * (a.im - b.im),
            };
        }
    }
}

fn apply_x(state: &mut [Complex64], qubit: usize) {
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) == 0 {
            let j = i | mask;
            state.swap(i, j);
        }
    }
}

fn apply_y(state: &mut [Complex64], qubit: usize) {
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) == 0 {
            let j = i | mask;
            let a = state[i];
            let b = state[j];
            // Y = [[0, -i], [i, 0]]
            state[i] = Complex64 {
                re: b.im,
                im: -b.re,
            };
            state[j] = Complex64 {
                re: -a.im,
                im: a.re,
            };
        }
    }
}

fn apply_z(state: &mut [Complex64], qubit: usize) {
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) != 0 {
            state[i].re = -state[i].re;
            state[i].im = -state[i].im;
        }
    }
}

fn apply_s(state: &mut [Complex64], qubit: usize) {
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) != 0 {
            // S = [[1, 0], [0, i]]
            let a = state[i];
            state[i] = Complex64 {
                re: -a.im,
                im: a.re,
            };
        }
    }
}

fn apply_t(state: &mut [Complex64], qubit: usize) {
    let n = state.len();
    let mask = 1 << qubit;
    let t_re = std::f64::consts::FRAC_1_SQRT_2;
    let t_im = std::f64::consts::FRAC_1_SQRT_2;

    for i in 0..n {
        if (i & mask) != 0 {
            // T = [[1, 0], [0, exp(i*pi/4)]]
            let a = state[i];
            state[i] = Complex64 {
                re: a.re * t_re - a.im * t_im,
                im: a.re * t_im + a.im * t_re,
            };
        }
    }
}

fn apply_rx(state: &mut [Complex64], qubit: usize, theta: f64) {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) == 0 {
            let j = i | mask;
            let a = state[i];
            let b = state[j];
            // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
            state[i] = Complex64 {
                re: cos * a.re + sin * b.im,
                im: cos * a.im - sin * b.re,
            };
            state[j] = Complex64 {
                re: cos * b.re + sin * a.im,
                im: cos * b.im - sin * a.re,
            };
        }
    }
}

fn apply_ry(state: &mut [Complex64], qubit: usize, theta: f64) {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        if (i & mask) == 0 {
            let j = i | mask;
            let a = state[i];
            let b = state[j];
            // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            state[i] = Complex64 {
                re: cos * a.re - sin * b.re,
                im: cos * a.im - sin * b.im,
            };
            state[j] = Complex64 {
                re: sin * a.re + cos * b.re,
                im: sin * a.im + cos * b.im,
            };
        }
    }
}

fn apply_rz(state: &mut [Complex64], qubit: usize, theta: f64) {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let n = state.len();
    let mask = 1 << qubit;

    for i in 0..n {
        let a = state[i];
        if (i & mask) == 0 {
            // exp(-i*θ/2)
            state[i] = Complex64 {
                re: cos * a.re + sin * a.im,
                im: cos * a.im - sin * a.re,
            };
        } else {
            // exp(i*θ/2)
            state[i] = Complex64 {
                re: cos * a.re - sin * a.im,
                im: cos * a.im + sin * a.re,
            };
        }
    }
}

fn apply_cnot(state: &mut [Complex64], control: usize, target: usize) {
    let n = state.len();
    let ctrl_mask = 1 << control;
    let tgt_mask = 1 << target;

    for i in 0..n {
        if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
            let j = i | tgt_mask;
            state.swap(i, j);
        }
    }
}

fn apply_cz(state: &mut [Complex64], control: usize, target: usize) {
    let n = state.len();
    let ctrl_mask = 1 << control;
    let tgt_mask = 1 << target;

    for i in 0..n {
        if (i & ctrl_mask) != 0 && (i & tgt_mask) != 0 {
            state[i].re = -state[i].re;
            state[i].im = -state[i].im;
        }
    }
}

fn apply_swap(state: &mut [Complex64], q1: usize, q2: usize) {
    let n = state.len();
    let mask1 = 1 << q1;
    let mask2 = 1 << q2;

    for i in 0..n {
        let b1 = (i >> q1) & 1;
        let b2 = (i >> q2) & 1;
        if b1 != b2 && b1 == 0 {
            let j = (i ^ mask1) ^ mask2;
            state.swap(i, j);
        }
    }
}

fn apply_crz(state: &mut [Complex64], control: usize, target: usize, theta: f64) {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let n = state.len();
    let ctrl_mask = 1 << control;
    let tgt_mask = 1 << target;

    for i in 0..n {
        if (i & ctrl_mask) != 0 {
            let a = state[i];
            if (i & tgt_mask) == 0 {
                state[i] = Complex64 {
                    re: cos * a.re + sin * a.im,
                    im: cos * a.im - sin * a.re,
                };
            } else {
                state[i] = Complex64 {
                    re: cos * a.re - sin * a.im,
                    im: cos * a.im + sin * a.re,
                };
            }
        }
    }
}

/// Estimate circuit depth (rough heuristic).
fn estimate_depth(circuit: &QuantumCircuit) -> usize {
    // Simple estimate: number of gates / number of qubits
    // Real implementation would do proper scheduling
    if circuit.n_qubits == 0 {
        0
    } else {
        (circuit.gate_count() + circuit.n_qubits - 1) / circuit.n_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_substrate_creation() {
        let substrate = QuantumSubstrate::new(1, "TestQuantum", 20);

        assert_eq!(substrate.name(), "TestQuantum");
        assert_eq!(substrate.substrate_type(), SubstrateType::Quantum);
        assert!(substrate.capacity() > 0.99);
    }

    #[test]
    fn test_can_handle() {
        let substrate = QuantumSubstrate::new(1, "TestQuantum", 10);

        // Should handle circuits within qubit limit
        let small_circuit = QuantumSpec::new(5, QuantumCircuit::new(5));
        assert!(substrate.can_handle(&JobSpec::Quantum(small_circuit)));

        // Should not handle circuits exceeding qubit limit
        let large_circuit = QuantumSpec::new(15, QuantumCircuit::new(15));
        assert!(!substrate.can_handle(&JobSpec::Quantum(large_circuit)));
    }

    #[test]
    fn test_execute_bell_state() {
        let mut substrate = QuantumSubstrate::new(1, "TestQuantum", 10).with_seed(42);

        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);

        let spec = QuantumSpec::new(2, circuit).with_shots(1000);
        let stage = JobStage::new(0, JobSpec::Quantum(spec));

        let result = substrate.execute(&stage).unwrap();
        assert!(result.success);

        if let StageOutputs::MeasurementCounts(counts) = &result.outputs {
            // Bell state should give ~50% |00⟩ and ~50% |11⟩
            let count_00 = counts.get("00").copied().unwrap_or(0);
            let count_11 = counts.get("11").copied().unwrap_or(0);
            assert!(count_00 > 400 && count_00 < 600);
            assert!(count_11 > 400 && count_11 < 600);
            // No other outcomes
            assert_eq!(counts.len(), 2);
        } else {
            panic!("Expected measurement counts");
        }
    }

    #[test]
    fn test_execute_ghz_state() {
        let mut substrate = QuantumSubstrate::new(1, "TestQuantum", 10).with_seed(123);

        let spec = QuantumSpec::ghz_state(3).with_shots(1000);
        let stage = JobStage::new(0, JobSpec::Quantum(spec));

        let result = substrate.execute(&stage).unwrap();
        assert!(result.success);

        if let StageOutputs::MeasurementCounts(counts) = &result.outputs {
            // GHZ should give ~50% |000⟩ and ~50% |111⟩
            assert!(counts.contains_key("000") || counts.contains_key("111"));
        }
    }

    #[test]
    fn test_state_vector_output() {
        let mut substrate = QuantumSubstrate::new(1, "TestQuantum", 10);

        let mut circuit = QuantumCircuit::new(1);
        circuit.h(0);

        let spec = QuantumSpec::new(1, circuit).with_measurement(MeasurementSpec::None);
        let stage = JobStage::new(0, JobSpec::Quantum(spec));

        let result = substrate.execute(&stage).unwrap();

        if let StageOutputs::StateVector(state) = &result.outputs {
            // |+⟩ state
            assert_eq!(state.len(), 2);
            let expected = 1.0 / 2.0_f64.sqrt();
            assert!((state[0].re - expected).abs() < 1e-10);
            assert!((state[1].re - expected).abs() < 1e-10);
        } else {
            panic!("Expected state vector");
        }
    }

    #[test]
    fn test_rotation_gates() {
        let mut substrate = QuantumSubstrate::new(1, "TestQuantum", 10);

        // Test Rx(π) = X (up to global phase)
        let mut circuit = QuantumCircuit::new(1);
        circuit.rx(0, std::f64::consts::PI);

        let spec = QuantumSpec::new(1, circuit).with_measurement(MeasurementSpec::None);
        let stage = JobStage::new(0, JobSpec::Quantum(spec));

        let result = substrate.execute(&stage).unwrap();

        if let StageOutputs::StateVector(state) = &result.outputs {
            // Should be in |1⟩ state (up to global phase)
            assert!(state[0].re.abs() < 1e-10 && state[0].im.abs() < 1e-10);
            assert!(
                (state[1].re.abs() - 1.0).abs() < 1e-10 || (state[1].im.abs() - 1.0).abs() < 1e-10
            );
        }
    }
}
