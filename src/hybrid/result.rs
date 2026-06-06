//! Result types for the hybrid orchestration layer.
//!
//! This module defines the output types for hybrid jobs, including
//! measurement outcomes, optimization results, and execution metadata.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use crate::tile8::sparse_quantum_bigint::Complex64;

use super::job::JobSpec;

/// Unique identifier for a submitted job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Job({})", self.0)
    }
}

/// Current state of a job in the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    /// Job is waiting in the queue
    Queued,
    /// Job is currently executing
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed with an error
    Failed,
    /// Job was cancelled
    Cancelled,
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobState::Queued => write!(f, "Queued"),
            JobState::Running => write!(f, "Running"),
            JobState::Completed => write!(f, "Completed"),
            JobState::Failed => write!(f, "Failed"),
            JobState::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// A stage of execution within a hybrid job.
///
/// Jobs can be broken into multiple stages that execute on different
/// substrates. For example, a VQE iteration has:
/// 1. Quantum stage: Execute ansatz and measure
/// 2. Classical stage: Compute cost from measurements
/// 3. Classical stage: Optimize parameters
#[derive(Clone, Debug)]
pub struct JobStage {
    /// Stage index within the job
    pub index: usize,

    /// Job specification for this stage
    pub spec: JobSpec,

    /// Input data from previous stages
    pub inputs: HashMap<String, DataRef>,

    /// Stage metadata
    pub metadata: StageMetadata,
}

impl JobStage {
    /// Create a new job stage.
    pub fn new(index: usize, spec: JobSpec) -> Self {
        Self {
            index,
            spec,
            inputs: HashMap::new(),
            metadata: StageMetadata::default(),
        }
    }

    /// Add an input reference.
    pub fn with_input(mut self, name: &str, data_ref: DataRef) -> Self {
        self.inputs.insert(name.to_string(), data_ref);
        self
    }
}

/// Metadata for a job stage.
#[derive(Clone, Debug, Default)]
pub struct StageMetadata {
    /// Human-readable description
    pub description: Option<String>,

    /// Expected duration (hint for scheduler)
    pub expected_duration: Option<Duration>,

    /// Priority within the job (higher = more urgent)
    pub priority: u32,

    /// Tags for filtering/routing
    pub tags: Vec<String>,
}

/// Reference to data produced by a previous stage.
#[derive(Clone, Debug)]
pub enum DataRef {
    /// Reference to stage output by index
    StageOutput(usize),

    /// Reference to a named output from a stage
    NamedOutput { stage: usize, name: String },

    /// Literal value embedded in the reference
    Literal(LiteralData),

    /// Reference to external data (bridge)
    External { bridge_id: u64, key: String },
}

/// Literal data that can be embedded in a DataRef.
#[derive(Clone, Debug)]
pub enum LiteralData {
    /// Floating point values
    Floats(Vec<f64>),

    /// Integer values
    Ints(Vec<i64>),

    /// Complex amplitudes
    Amplitudes(Vec<Complex64>),

    /// Binary data
    Bytes(Vec<u8>),

    /// String data
    Text(String),
}

/// Result of executing a single stage.
#[derive(Clone, Debug)]
pub struct StageResult {
    /// Stage index
    pub stage_index: usize,

    /// Whether execution succeeded
    pub success: bool,

    /// Execution duration
    pub duration: Duration,

    /// Output data
    pub outputs: StageOutputs,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Execution metrics
    pub metrics: StageMetrics,
}

impl StageResult {
    /// Create a successful stage result.
    pub fn success(stage_index: usize, duration: Duration, outputs: StageOutputs) -> Self {
        Self {
            stage_index,
            success: true,
            duration,
            outputs,
            error: None,
            metrics: StageMetrics::default(),
        }
    }

    /// Create a failed stage result.
    pub fn failure(stage_index: usize, duration: Duration, error: String) -> Self {
        Self {
            stage_index,
            success: false,
            duration,
            outputs: StageOutputs::Empty,
            error: Some(error),
            metrics: StageMetrics::default(),
        }
    }

    /// Add metrics to the result.
    pub fn with_metrics(mut self, metrics: StageMetrics) -> Self {
        self.metrics = metrics;
        self
    }
}

/// Output data from a stage execution.
#[derive(Clone, Debug)]
pub enum StageOutputs {
    /// No output
    Empty,

    /// Measurement counts from quantum execution
    MeasurementCounts(HashMap<String, usize>),

    /// State vector (for simulation without measurement)
    StateVector(Vec<Complex64>),

    /// Optimization result
    Optimization {
        /// Optimal parameters found
        optimal_params: Vec<f64>,
        /// Optimal cost value
        optimal_cost: f64,
        /// Number of iterations performed
        iterations: usize,
        /// Convergence history
        history: Vec<f64>,
    },

    /// Classical computation result
    Classical {
        /// Output values
        values: Vec<f64>,
        /// Named outputs
        named: HashMap<String, Vec<f64>>,
    },

    /// Observer grid result
    Observer {
        /// Final collapse ratio
        collapse_ratio: f64,
        /// Final anomaly score
        anomaly_score: f64,
        /// Total ticks executed
        ticks: u64,
        /// Measurement outcomes
        measurements: Vec<u8>,
    },

    /// Raw bytes output
    Raw(Vec<u8>),
}

impl StageOutputs {
    /// Get measurement counts if available.
    pub fn as_counts(&self) -> Option<&HashMap<String, usize>> {
        match self {
            StageOutputs::MeasurementCounts(counts) => Some(counts),
            _ => None,
        }
    }

    /// Get optimization result if available.
    pub fn as_optimization(&self) -> Option<(&Vec<f64>, f64)> {
        match self {
            StageOutputs::Optimization {
                optimal_params,
                optimal_cost,
                ..
            } => Some((optimal_params, *optimal_cost)),
            _ => None,
        }
    }

    /// Get state vector if available.
    pub fn as_state_vector(&self) -> Option<&Vec<Complex64>> {
        match self {
            StageOutputs::StateVector(sv) => Some(sv),
            _ => None,
        }
    }
}

/// Execution metrics for a stage.
#[derive(Clone, Debug, Default)]
pub struct StageMetrics {
    /// Number of quantum gates executed
    pub gate_count: usize,

    /// Circuit depth
    pub circuit_depth: usize,

    /// Number of shots executed
    pub shots: usize,

    /// Number of qubits used
    pub qubits_used: usize,

    /// CPU time used (if applicable)
    pub cpu_time: Option<Duration>,

    /// GPU time used (if applicable)
    pub gpu_time: Option<Duration>,

    /// Memory high water mark (bytes)
    pub peak_memory: Option<usize>,

    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

impl StageMetrics {
    /// Create metrics for quantum execution.
    pub fn quantum(gate_count: usize, circuit_depth: usize, shots: usize, qubits: usize) -> Self {
        Self {
            gate_count,
            circuit_depth,
            shots,
            qubits_used: qubits,
            ..Default::default()
        }
    }

    /// Add a custom metric.
    pub fn with_custom(mut self, name: &str, value: f64) -> Self {
        self.custom.insert(name.to_string(), value);
        self
    }
}

/// Complete result of a hybrid job.
#[derive(Clone, Debug)]
pub struct JobResult {
    /// Job identifier
    pub job_id: JobId,

    /// Final job state
    pub state: JobState,

    /// Total execution time
    pub total_duration: Duration,

    /// Results from each stage
    pub stage_results: Vec<StageResult>,

    /// Final output (from last stage)
    pub final_output: StageOutputs,

    /// Aggregated metrics
    pub metrics: JobMetrics,
}

impl JobResult {
    /// Create a new job result.
    pub fn new(job_id: JobId, state: JobState, total_duration: Duration) -> Self {
        Self {
            job_id,
            state,
            total_duration,
            stage_results: Vec::new(),
            final_output: StageOutputs::Empty,
            metrics: JobMetrics::default(),
        }
    }

    /// Check if the job completed successfully.
    pub fn is_success(&self) -> bool {
        self.state == JobState::Completed
    }

    /// Get the error message if the job failed.
    pub fn error(&self) -> Option<&str> {
        if self.state == JobState::Failed {
            self.stage_results
                .iter()
                .rev()
                .find_map(|r| r.error.as_deref())
        } else {
            None
        }
    }

    /// Get the optimal parameters if this was a variational job.
    pub fn optimal_params(&self) -> Option<&Vec<f64>> {
        match &self.final_output {
            StageOutputs::Optimization { optimal_params, .. } => Some(optimal_params),
            _ => None,
        }
    }

    /// Get the optimal cost if this was a variational job.
    pub fn optimal_cost(&self) -> Option<f64> {
        match &self.final_output {
            StageOutputs::Optimization { optimal_cost, .. } => Some(*optimal_cost),
            _ => None,
        }
    }
}

/// Aggregated metrics for a complete job.
#[derive(Clone, Debug, Default)]
pub struct JobMetrics {
    /// Total quantum gates executed
    pub total_gates: usize,

    /// Total shots across all stages
    pub total_shots: usize,

    /// Maximum qubits used in any stage
    pub max_qubits: usize,

    /// Number of stages executed
    pub stages_executed: usize,

    /// Number of stages that failed
    pub stages_failed: usize,

    /// Time spent on quantum substrates
    pub quantum_time: Duration,

    /// Time spent on classical substrates
    pub classical_time: Duration,

    /// Time spent waiting/scheduling
    pub overhead_time: Duration,
}

impl JobMetrics {
    /// Aggregate metrics from stage results.
    pub fn from_stages(stage_results: &[StageResult]) -> Self {
        let mut metrics = Self::default();
        metrics.stages_executed = stage_results.len();

        for result in stage_results {
            if !result.success {
                metrics.stages_failed += 1;
            }

            metrics.total_gates += result.metrics.gate_count;
            metrics.total_shots += result.metrics.shots;
            metrics.max_qubits = metrics.max_qubits.max(result.metrics.qubits_used);
        }

        metrics
    }
}

/// A tracked hybrid job with full execution state.
#[derive(Debug)]
pub struct HybridJob {
    /// Job identifier
    pub id: JobId,

    /// Original job specification
    pub spec: JobSpec,

    /// Current state
    pub state: JobState,

    /// Submission time
    pub submitted_at: Instant,

    /// Start time (when execution began)
    pub started_at: Option<Instant>,

    /// Completion time
    pub completed_at: Option<Instant>,

    /// Current stage index
    pub current_stage: usize,

    /// Stage results collected so far
    pub stage_results: Vec<StageResult>,

    /// Priority (higher = more urgent)
    pub priority: u32,

    /// Tags for filtering
    pub tags: Vec<String>,
}

impl HybridJob {
    /// Create a new hybrid job.
    pub fn new(id: JobId, spec: JobSpec) -> Self {
        Self {
            id,
            spec,
            state: JobState::Queued,
            submitted_at: Instant::now(),
            started_at: None,
            completed_at: None,
            current_stage: 0,
            stage_results: Vec::new(),
            priority: 0,
            tags: Vec::new(),
        }
    }

    /// Set the job priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a tag to the job.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Mark the job as started.
    pub fn start(&mut self) {
        self.state = JobState::Running;
        self.started_at = Some(Instant::now());
    }

    /// Mark the job as completed.
    pub fn complete(&mut self, final_result: StageResult) {
        self.stage_results.push(final_result);
        self.state = JobState::Completed;
        self.completed_at = Some(Instant::now());
    }

    /// Mark the job as failed.
    pub fn fail(&mut self, error_result: StageResult) {
        self.stage_results.push(error_result);
        self.state = JobState::Failed;
        self.completed_at = Some(Instant::now());
    }

    /// Cancel the job.
    pub fn cancel(&mut self) {
        self.state = JobState::Cancelled;
        self.completed_at = Some(Instant::now());
    }

    /// Get the total duration so far.
    pub fn elapsed(&self) -> Duration {
        match self.started_at {
            Some(start) => match self.completed_at {
                Some(end) => end.duration_since(start),
                None => start.elapsed(),
            },
            None => Duration::ZERO,
        }
    }

    /// Convert to a JobResult.
    pub fn into_result(self) -> JobResult {
        let total_duration = self.elapsed();
        let final_output = self
            .stage_results
            .last()
            .map(|r| r.outputs.clone())
            .unwrap_or(StageOutputs::Empty);

        let metrics = JobMetrics::from_stages(&self.stage_results);

        JobResult {
            job_id: self.id,
            state: self.state,
            total_duration,
            stage_results: self.stage_results,
            final_output,
            metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id_display() {
        let id = JobId(42);
        assert_eq!(format!("{}", id), "Job(42)");
    }

    #[test]
    fn test_job_state_display() {
        assert_eq!(format!("{}", JobState::Queued), "Queued");
        assert_eq!(format!("{}", JobState::Running), "Running");
        assert_eq!(format!("{}", JobState::Completed), "Completed");
        assert_eq!(format!("{}", JobState::Failed), "Failed");
    }

    #[test]
    fn test_stage_result_success() {
        let result = StageResult::success(
            0,
            Duration::from_millis(100),
            StageOutputs::MeasurementCounts(HashMap::new()),
        );

        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_stage_result_failure() {
        let result = StageResult::failure(0, Duration::from_millis(50), "Test error".to_string());

        assert!(!result.success);
        assert_eq!(result.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_stage_outputs_as_counts() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 512);
        counts.insert("11".to_string(), 512);

        let outputs = StageOutputs::MeasurementCounts(counts);
        let retrieved = outputs.as_counts().unwrap();

        assert_eq!(retrieved.get("00"), Some(&512));
        assert_eq!(retrieved.get("11"), Some(&512));
    }

    #[test]
    fn test_job_metrics_from_stages() {
        let results = vec![
            StageResult::success(0, Duration::from_millis(100), StageOutputs::Empty)
                .with_metrics(StageMetrics::quantum(50, 10, 1024, 5)),
            StageResult::success(1, Duration::from_millis(50), StageOutputs::Empty)
                .with_metrics(StageMetrics::quantum(30, 5, 512, 3)),
        ];

        let metrics = JobMetrics::from_stages(&results);

        assert_eq!(metrics.stages_executed, 2);
        assert_eq!(metrics.stages_failed, 0);
        assert_eq!(metrics.total_gates, 80);
        assert_eq!(metrics.total_shots, 1536);
        assert_eq!(metrics.max_qubits, 5);
    }

    #[test]
    fn test_hybrid_job_lifecycle() {
        let job = HybridJob::new(
            JobId(1),
            JobSpec::Classical(crate::hybrid::job::ClassicalSpec::new(
                crate::hybrid::job::ClassicalOp::Custom {
                    name: "test".into(),
                    inputs: vec![],
                },
            )),
        );

        assert_eq!(job.state, JobState::Queued);
        assert!(job.started_at.is_none());

        let mut job = job;
        job.start();
        assert_eq!(job.state, JobState::Running);
        assert!(job.started_at.is_some());

        job.complete(StageResult::success(
            0,
            Duration::from_millis(100),
            StageOutputs::Empty,
        ));
        assert_eq!(job.state, JobState::Completed);
        assert!(job.completed_at.is_some());
    }
}
