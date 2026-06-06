//! Sprint 62.0: Circuit Batching for Parameter Sweeps
//!
//! Provides efficient execution of multiple circuits with different parameters.
//! Ideal for VQE optimization where the same circuit is evaluated many times
//! with different parameter values.
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::{CircuitBatch, BatchResult, HybridScheduler};
//!
//! let batch = CircuitBatch::new(ansatz_circuit)
//!     .with_parameter_sets(parameter_grid)
//!     .with_shots(1000);
//!
//! let results = scheduler.execute_batch(batch)?;
//! for (params, counts) in results.iter() {
//!     let energy = compute_energy(&counts);
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::job::{JobSpec, MeasurementSpec, QuantumCircuit, QuantumSpec};
use super::result::{JobStage, StageOutputs};
use super::substrate::{Substrate, SubstrateError};

/// A batch of circuits to execute with different parameters.
///
/// All circuits share the same structure but have different parameter
/// values bound to parameterized gates.
#[derive(Clone, Debug)]
pub struct CircuitBatch {
    /// Base circuit with parameterized gates
    pub base_circuit: QuantumCircuit,

    /// Parameter sets to evaluate (each inner Vec has same length as circuit parameters)
    pub parameter_sets: Vec<Vec<f64>>,

    /// Number of measurement shots per circuit
    pub shots: usize,

    /// Measurement specification
    pub measurement: MeasurementSpec,

    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

impl CircuitBatch {
    /// Create a new circuit batch from a base circuit.
    pub fn new(base_circuit: QuantumCircuit) -> Self {
        Self {
            base_circuit,
            parameter_sets: Vec::new(),
            shots: 1000,
            measurement: MeasurementSpec::Computational,
            seed: None,
        }
    }

    /// Add parameter sets to evaluate.
    pub fn with_parameter_sets(mut self, params: Vec<Vec<f64>>) -> Self {
        self.parameter_sets = params;
        self
    }

    /// Add a single parameter set.
    pub fn add_parameters(&mut self, params: Vec<f64>) {
        self.parameter_sets.push(params);
    }

    /// Set number of measurement shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set measurement specification.
    pub fn with_measurement(mut self, measurement: MeasurementSpec) -> Self {
        self.measurement = measurement;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get number of circuits in the batch.
    pub fn len(&self) -> usize {
        self.parameter_sets.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.parameter_sets.is_empty()
    }

    /// Get expected parameter count per circuit.
    pub fn expected_param_count(&self) -> usize {
        self.base_circuit.parameter_count()
    }

    /// Validate that all parameter sets have correct length.
    pub fn validate(&self) -> Result<(), String> {
        let expected = self.expected_param_count();
        for (i, params) in self.parameter_sets.iter().enumerate() {
            if params.len() != expected {
                return Err(format!(
                    "Parameter set {} has {} params, expected {}",
                    i,
                    params.len(),
                    expected
                ));
            }
        }
        Ok(())
    }
}

/// Result of batch circuit execution.
#[derive(Clone, Debug)]
pub struct BatchResult {
    /// Individual results for each parameter set
    pub results: Vec<BatchEntry>,

    /// Total execution time
    pub total_duration: Duration,

    /// Number of circuits executed
    pub circuit_count: usize,
}

impl BatchResult {
    /// Create a new batch result.
    pub fn new(results: Vec<BatchEntry>, total_duration: Duration) -> Self {
        let circuit_count = results.len();
        Self {
            results,
            total_duration,
            circuit_count,
        }
    }

    /// Get result for a specific parameter set index.
    pub fn get(&self, index: usize) -> Option<&BatchEntry> {
        self.results.get(index)
    }

    /// Iterate over (parameters, output) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &BatchEntry> {
        self.results.iter()
    }

    /// Get all measurement counts (for computational basis measurement).
    pub fn all_counts(&self) -> Vec<Option<&HashMap<String, usize>>> {
        self.results
            .iter()
            .map(|entry| match &entry.output {
                StageOutputs::MeasurementCounts(counts) => Some(counts),
                _ => None,
            })
            .collect()
    }

    /// Get average execution time per circuit.
    pub fn avg_duration(&self) -> Duration {
        if self.circuit_count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.circuit_count as u32
        }
    }

    /// Check if all circuits succeeded.
    pub fn all_success(&self) -> bool {
        self.results.iter().all(|r| r.success)
    }

    /// Count successful circuits.
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }
}

/// Single entry in a batch result.
#[derive(Clone, Debug)]
pub struct BatchEntry {
    /// Index in the batch
    pub index: usize,

    /// Parameters used for this circuit
    pub parameters: Vec<f64>,

    /// Execution output
    pub output: StageOutputs,

    /// Whether execution succeeded
    pub success: bool,

    /// Execution duration for this circuit
    pub duration: Duration,

    /// Error message if failed
    pub error: Option<String>,
}

impl BatchEntry {
    /// Create a successful batch entry.
    pub fn success(
        index: usize,
        parameters: Vec<f64>,
        output: StageOutputs,
        duration: Duration,
    ) -> Self {
        Self {
            index,
            parameters,
            output,
            success: true,
            duration,
            error: None,
        }
    }

    /// Create a failed batch entry.
    pub fn failure(index: usize, parameters: Vec<f64>, error: String, duration: Duration) -> Self {
        Self {
            index,
            parameters,
            output: StageOutputs::Empty,
            success: false,
            duration,
            error: Some(error),
        }
    }
}

/// Execute a batch of circuits on a substrate.
///
/// This function handles the iteration and parameter binding,
/// executing each circuit on the provided substrate.
pub fn execute_batch<S: Substrate>(
    substrate: &mut S,
    batch: &CircuitBatch,
) -> Result<BatchResult, SubstrateError> {
    // Validate batch
    batch
        .validate()
        .map_err(|e| SubstrateError::InvalidSpec(e))?;

    if batch.is_empty() {
        return Ok(BatchResult::new(Vec::new(), Duration::ZERO));
    }

    let start = Instant::now();
    let mut results = Vec::with_capacity(batch.len());

    let n_qubits = batch.base_circuit.n_qubits;

    for (i, params) in batch.parameter_sets.iter().enumerate() {
        let circuit_start = Instant::now();

        // Bind parameters to create concrete circuit
        let bound_circuit = batch.base_circuit.bind_parameters(params);

        // Create quantum spec
        let mut spec = QuantumSpec::new(n_qubits, bound_circuit)
            .with_shots(batch.shots)
            .with_measurement(batch.measurement.clone());

        // Set seed with offset for reproducibility
        if let Some(seed) = batch.seed {
            spec = spec.with_seed(seed.wrapping_add(i as u64));
        }

        // Create job stage
        let stage = JobStage::new(i, JobSpec::Quantum(spec));

        // Execute
        match substrate.execute(&stage) {
            Ok(stage_result) => {
                results.push(BatchEntry::success(
                    i,
                    params.clone(),
                    stage_result.outputs,
                    circuit_start.elapsed(),
                ));
            }
            Err(e) => {
                results.push(BatchEntry::failure(
                    i,
                    params.clone(),
                    e.to_string(),
                    circuit_start.elapsed(),
                ));
            }
        }
    }

    Ok(BatchResult::new(results, start.elapsed()))
}

/// Generate a parameter grid for sweeping.
///
/// Creates all combinations of parameter values.
pub fn parameter_grid(ranges: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if ranges.is_empty() {
        return vec![];
    }

    let mut result = vec![vec![]];

    for range in ranges {
        let mut new_result = Vec::new();
        for existing in &result {
            for &value in range {
                let mut new_vec = existing.clone();
                new_vec.push(value);
                new_result.push(new_vec);
            }
        }
        result = new_result;
    }

    result
}

/// Generate a linear range of parameter values.
pub fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return vec![];
    }
    if count == 1 {
        return vec![start];
    }

    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_batch_creation() {
        let circuit = QuantumCircuit::hardware_efficient(2, 1);
        let batch = CircuitBatch::new(circuit.clone())
            .with_shots(2000)
            .with_seed(42);

        assert_eq!(batch.shots, 2000);
        assert_eq!(batch.seed, Some(42));
        assert!(batch.is_empty());
    }

    #[test]
    fn test_add_parameters() {
        let circuit = QuantumCircuit::hardware_efficient(2, 1);
        let mut batch = CircuitBatch::new(circuit);

        batch.add_parameters(vec![0.1, 0.2, 0.3, 0.4]);
        batch.add_parameters(vec![0.5, 0.6, 0.7, 0.8]);

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_parameter_grid() {
        let ranges = vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0]];

        let grid = parameter_grid(&ranges);

        assert_eq!(grid.len(), 6); // 2 * 3
        assert!(grid.contains(&vec![0.0, 0.0]));
        assert!(grid.contains(&vec![0.0, 1.0]));
        assert!(grid.contains(&vec![0.0, 2.0]));
        assert!(grid.contains(&vec![1.0, 0.0]));
        assert!(grid.contains(&vec![1.0, 1.0]));
        assert!(grid.contains(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_linspace() {
        let values = linspace(0.0, 1.0, 5);
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[2] - 0.5).abs() < 1e-10);
        assert!((values[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_single() {
        let values = linspace(0.5, 1.5, 1);
        assert_eq!(values.len(), 1);
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_empty() {
        let values = linspace(0.0, 1.0, 0);
        assert!(values.is_empty());
    }

    #[test]
    fn test_batch_result_operations() {
        let entries = vec![
            BatchEntry::success(0, vec![0.1], StageOutputs::Empty, Duration::from_millis(10)),
            BatchEntry::success(1, vec![0.2], StageOutputs::Empty, Duration::from_millis(15)),
            BatchEntry::failure(2, vec![0.3], "error".to_string(), Duration::from_millis(5)),
        ];

        let result = BatchResult::new(entries, Duration::from_millis(30));

        assert_eq!(result.circuit_count, 3);
        assert_eq!(result.success_count(), 2);
        assert!(!result.all_success());
    }

    #[test]
    fn test_batch_validation() {
        let circuit = QuantumCircuit::hardware_efficient(2, 1);
        let param_count = circuit.parameter_count();

        let mut batch = CircuitBatch::new(circuit);
        batch.add_parameters(vec![0.0; param_count]);
        batch.add_parameters(vec![1.0; param_count]);

        assert!(batch.validate().is_ok());

        // Add wrong-length parameters
        batch.add_parameters(vec![0.0; param_count + 1]);
        assert!(batch.validate().is_err());
    }
}
