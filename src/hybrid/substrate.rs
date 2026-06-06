//! Substrate trait and types for the hybrid orchestration layer.
//!
//! A substrate is a compute backend that can execute certain types of jobs.
//! The orchestration layer routes jobs to appropriate substrates based on
//! their capabilities.

use std::fmt;
use std::time::Duration;

use super::job::JobSpec;
use super::result::{JobStage, StageResult};

/// Unique identifier for a substrate instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubstrateId(pub u64);

impl fmt::Display for SubstrateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Substrate({})", self.0)
    }
}

/// Type of compute substrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubstrateType {
    /// Classical computation (CPUs, GPUs)
    Classical,
    /// Quantum simulation
    Quantum,
    /// Observer grid (hybrid decision layer)
    Observer,
    /// Combined/custom hybrid
    Hybrid,
}

impl fmt::Display for SubstrateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubstrateType::Classical => write!(f, "Classical"),
            SubstrateType::Quantum => write!(f, "Quantum"),
            SubstrateType::Observer => write!(f, "Observer"),
            SubstrateType::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Errors that can occur during substrate execution.
#[derive(Debug, Clone)]
pub enum SubstrateError {
    /// Job type not supported by this substrate
    UnsupportedJob(String),
    /// Substrate resources exhausted
    ResourceExhausted,
    /// Execution failed with message
    ExecutionFailed(String),
    /// Execution timed out
    Timeout,
    /// Invalid job specification
    InvalidSpec(String),
    /// Missing required input data
    MissingInput(String),
    /// Internal substrate error
    Internal(String),
}

impl fmt::Display for SubstrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubstrateError::UnsupportedJob(msg) => write!(f, "Unsupported job: {}", msg),
            SubstrateError::ResourceExhausted => write!(f, "Resource exhausted"),
            SubstrateError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            SubstrateError::Timeout => write!(f, "Execution timed out"),
            SubstrateError::InvalidSpec(msg) => write!(f, "Invalid specification: {}", msg),
            SubstrateError::MissingInput(msg) => write!(f, "Missing input: {}", msg),
            SubstrateError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for SubstrateError {}

/// Capabilities of a substrate.
#[derive(Clone, Debug, Default)]
pub struct SubstrateCapabilities {
    /// Maximum number of qubits (for quantum substrates)
    pub max_qubits: Option<usize>,

    /// Maximum circuit depth (for quantum substrates)
    pub max_depth: Option<usize>,

    /// Whether GPU acceleration is available
    pub gpu_available: bool,

    /// Maximum number of parallel jobs
    pub max_parallel_jobs: usize,

    /// Supported gate types (for quantum substrates)
    pub supported_gates: Vec<String>,

    /// Whether variational loops are natively supported
    pub supports_variational: bool,

    /// Whether observer dynamics are supported
    pub supports_observer: bool,
}

/// Common interface for all compute substrates.
///
/// Substrates are the execution backends for hybrid computation. Each substrate
/// type (classical, quantum, observer) implements this trait to provide a
/// uniform interface for the scheduler.
///
/// # Example Implementation
///
/// ```ignore
/// struct MySubstrate {
///     id: SubstrateId,
/// }
///
/// impl Substrate for MySubstrate {
///     fn id(&self) -> SubstrateId { self.id }
///     fn name(&self) -> &str { "MySubstrate" }
///     fn substrate_type(&self) -> SubstrateType { SubstrateType::Classical }
///     // ...
/// }
/// ```
pub trait Substrate: Send {
    /// Returns the unique identifier for this substrate.
    fn id(&self) -> SubstrateId;

    /// Returns a human-readable name for this substrate.
    fn name(&self) -> &str;

    /// Returns the type of this substrate.
    fn substrate_type(&self) -> SubstrateType;

    /// Returns the current capacity of this substrate.
    ///
    /// - 0.0 means the substrate is fully busy
    /// - 1.0 means the substrate is completely idle
    /// - Values in between indicate partial availability
    fn capacity(&self) -> f64;

    /// Returns the capabilities of this substrate.
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities::default()
    }

    /// Checks if this substrate can handle the given job specification.
    ///
    /// Returns `true` if the substrate can execute this type of job,
    /// `false` otherwise.
    fn can_handle(&self, spec: &JobSpec) -> bool;

    /// Estimates the execution time for a job.
    ///
    /// This is used by the scheduler for planning and prioritization.
    /// The estimate should be conservative (overestimate rather than underestimate).
    fn estimate_time(&self, spec: &JobSpec) -> Duration;

    /// Executes a job stage on this substrate.
    ///
    /// This is the main execution method. It blocks until the stage completes
    /// or an error occurs.
    ///
    /// # Arguments
    ///
    /// * `stage` - The job stage to execute, including inputs
    ///
    /// # Returns
    ///
    /// * `Ok(StageResult)` - The result of the execution
    /// * `Err(SubstrateError)` - If execution fails
    fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError>;

    /// Resets the substrate to its initial state.
    ///
    /// This clears any cached state, temporary data, or ongoing computations.
    fn reset(&mut self);

    /// Returns whether the substrate is currently idle.
    fn is_idle(&self) -> bool {
        self.capacity() >= 0.99
    }

    /// Returns whether the substrate is currently busy.
    fn is_busy(&self) -> bool {
        self.capacity() < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_id_display() {
        let id = SubstrateId(42);
        assert_eq!(format!("{}", id), "Substrate(42)");
    }

    #[test]
    fn test_substrate_type_display() {
        assert_eq!(format!("{}", SubstrateType::Classical), "Classical");
        assert_eq!(format!("{}", SubstrateType::Quantum), "Quantum");
        assert_eq!(format!("{}", SubstrateType::Observer), "Observer");
        assert_eq!(format!("{}", SubstrateType::Hybrid), "Hybrid");
    }

    #[test]
    fn test_substrate_error_display() {
        let err = SubstrateError::UnsupportedJob("test".into());
        assert!(format!("{}", err).contains("test"));
    }
}
