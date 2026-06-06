//! Hybrid Scheduler for orchestrating multi-substrate computation.
//!
//! The HybridScheduler is the central coordinator that routes jobs to
//! appropriate substrates based on their type and requirements.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      HybridScheduler                            │
//! │  - Job queue management                                         │
//! │  - Resource allocation                                          │
//! │  - Substrate routing                                            │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!               ┌───────────────┼───────────────┐
//!               ▼               ▼               ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │ ClassicalEngine │ │  QuantumEngine  │ │ ObserverEngine  │
//! │ impl Substrate  │ │ impl Substrate  │ │ impl Substrate  │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use super::bridge::DataBridge;
use super::job::JobSpec;
use super::result::{
    HybridJob, JobId, JobMetrics, JobResult, JobStage, JobState, StageOutputs, StageResult,
};
use super::substrate::{Substrate, SubstrateError, SubstrateId, SubstrateType};

/// Configuration for the hybrid scheduler.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum jobs in the queue
    pub max_queue_size: usize,

    /// Maximum concurrent jobs
    pub max_concurrent: usize,

    /// Default timeout for jobs
    pub default_timeout: Duration,

    /// Enable automatic substrate selection
    pub auto_route: bool,

    /// Enable job preemption for high-priority jobs
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            max_concurrent: 4,
            default_timeout: Duration::from_secs(300),
            auto_route: true,
            enable_preemption: false,
        }
    }
}

/// Handle for non-blocking job execution (Sprint 62.0).
///
/// Returned by `submit_nonblocking()`, this handle allows checking
/// job status and retrieving results without blocking.
///
/// # Example
///
/// ```ignore
/// let handle = scheduler.submit_nonblocking(spec);
///
/// // Do other work while job runs...
///
/// // Check if done (non-blocking)
/// if let Some(result) = handle.try_get() {
///     println!("Done: {:?}", result);
/// }
///
/// // Or wait for completion (blocking)
/// let result = handle.wait();
/// ```
pub struct JobHandle {
    /// Job ID
    job_id: JobId,
    /// Channel receiver for result
    receiver: mpsc::Receiver<JobResult>,
}

impl JobHandle {
    /// Create a new job handle.
    fn new(job_id: JobId, receiver: mpsc::Receiver<JobResult>) -> Self {
        Self { job_id, receiver }
    }

    /// Get the job ID.
    pub fn id(&self) -> JobId {
        self.job_id
    }

    /// Try to get the result (non-blocking).
    ///
    /// Returns `Some(result)` if the job has completed, `None` if still running.
    pub fn try_get(&self) -> Option<JobResult> {
        self.receiver.try_recv().ok()
    }

    /// Wait for the result (blocking).
    ///
    /// This consumes the handle and blocks until the job completes.
    pub fn wait(self) -> Option<JobResult> {
        self.receiver.recv().ok()
    }

    /// Wait for the result with a timeout.
    ///
    /// Returns `None` if the timeout expires before the job completes.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<JobResult> {
        self.receiver.recv_timeout(timeout).ok()
    }

    /// Check if the result is ready without consuming it.
    pub fn is_ready(&self) -> bool {
        // Unfortunately mpsc doesn't have a peek, so we can't do this efficiently
        // For now, this always returns false until try_get is called
        false
    }
}

/// The hybrid scheduler coordinates execution across substrates.
///
/// # Example
///
/// ```ignore
/// use engine::hybrid::{HybridScheduler, JobSpec, QuantumSpec};
///
/// let mut scheduler = HybridScheduler::new();
///
/// // Register substrates
/// scheduler.register_substrate(Box::new(classical_substrate));
/// scheduler.register_substrate(Box::new(quantum_substrate));
///
/// // Submit a job
/// let job_id = scheduler.submit(JobSpec::Quantum(quantum_spec));
///
/// // Wait for completion
/// let result = scheduler.wait(job_id);
/// ```
pub struct HybridScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Registered substrates
    substrates: Vec<Box<dyn Substrate>>,

    /// Job queue
    queue: VecDeque<HybridJob>,

    /// Currently running jobs
    running: HashMap<JobId, HybridJob>,

    /// Completed jobs (kept for result retrieval)
    completed: HashMap<JobId, JobResult>,

    /// Data bridge for inter-substrate communication
    bridge: Arc<DataBridge>,

    /// Next job ID
    next_job_id: u64,

    /// Total jobs submitted
    total_submitted: u64,

    /// Total jobs completed
    total_completed: u64,

    /// Scheduler start time
    started_at: Instant,
}

impl HybridScheduler {
    /// Create a new hybrid scheduler with default configuration.
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Create a scheduler with custom configuration.
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            config,
            substrates: Vec::new(),
            queue: VecDeque::new(),
            running: HashMap::new(),
            completed: HashMap::new(),
            bridge: Arc::new(DataBridge::new()),
            next_job_id: 1,
            total_submitted: 0,
            total_completed: 0,
            started_at: Instant::now(),
        }
    }

    /// Register a substrate with the scheduler.
    pub fn register_substrate(&mut self, substrate: Box<dyn Substrate>) {
        self.substrates.push(substrate);
    }

    /// Get the data bridge for inter-substrate communication.
    pub fn bridge(&self) -> Arc<DataBridge> {
        Arc::clone(&self.bridge)
    }

    /// Submit a job to the scheduler.
    ///
    /// Returns a job ID that can be used to check status or retrieve results.
    pub fn submit(&mut self, spec: JobSpec) -> JobId {
        let job_id = JobId(self.next_job_id);
        self.next_job_id += 1;

        let job = HybridJob::new(job_id, spec);
        self.queue.push_back(job);
        self.total_submitted += 1;

        job_id
    }

    /// Submit a high-priority job.
    pub fn submit_priority(&mut self, spec: JobSpec, priority: u32) -> JobId {
        let job_id = JobId(self.next_job_id);
        self.next_job_id += 1;

        let job = HybridJob::new(job_id, spec).with_priority(priority);

        // Insert at front for high priority
        if priority > 0 {
            self.queue.push_front(job);
        } else {
            self.queue.push_back(job);
        }

        self.total_submitted += 1;
        job_id
    }

    /// Get the current state of a job.
    pub fn job_state(&self, job_id: JobId) -> Option<JobState> {
        // Check completed jobs first
        if let Some(result) = self.completed.get(&job_id) {
            return Some(result.state);
        }

        // Check running jobs
        if let Some(job) = self.running.get(&job_id) {
            return Some(job.state);
        }

        // Check queue
        for job in &self.queue {
            if job.id == job_id {
                return Some(job.state);
            }
        }

        None
    }

    /// Get the result of a completed job.
    pub fn get_result(&self, job_id: JobId) -> Option<&JobResult> {
        self.completed.get(&job_id)
    }

    /// Wait for a job to complete and return its result.
    ///
    /// This is a blocking operation that runs the scheduler until the
    /// specified job completes.
    pub fn wait(&mut self, job_id: JobId) -> Option<JobResult> {
        // If already completed, return immediately
        if let Some(result) = self.completed.remove(&job_id) {
            return Some(result);
        }

        // Run scheduler until job completes
        loop {
            self.tick();

            if let Some(result) = self.completed.remove(&job_id) {
                return Some(result);
            }

            // Check if job still exists
            let job_exists =
                self.running.contains_key(&job_id) || self.queue.iter().any(|j| j.id == job_id);

            if !job_exists {
                return None;
            }
        }
    }

    /// Submit a job for non-blocking execution (Sprint 62.0).
    ///
    /// This creates a dedicated thread to execute the job and returns
    /// a handle for retrieving the result later. The job runs independently
    /// of the main scheduler queue.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handle = scheduler.submit_nonblocking(spec);
    ///
    /// // Do other work...
    ///
    /// // Get result when needed
    /// let result = handle.wait();
    /// ```
    ///
    /// # Note
    ///
    /// Non-blocking jobs use their own substrate instances and don't
    /// share state with the main scheduler. This is ideal for independent
    /// quantum circuits that don't need inter-substrate communication.
    pub fn submit_nonblocking(&mut self, spec: JobSpec) -> JobHandle {
        let job_id = JobId(self.next_job_id);
        self.next_job_id += 1;
        self.total_submitted += 1;

        let (sender, receiver) = mpsc::channel();

        // Spawn execution thread
        thread::spawn(move || {
            let result = execute_job_standalone(job_id, spec);
            let _ = sender.send(result);
        });

        JobHandle::new(job_id, receiver)
    }

    /// Submit multiple jobs for parallel non-blocking execution.
    ///
    /// Returns handles for all jobs. Jobs execute independently and
    /// can be collected as they complete.
    pub fn submit_batch_nonblocking(&mut self, specs: Vec<JobSpec>) -> Vec<JobHandle> {
        specs
            .into_iter()
            .map(|spec| self.submit_nonblocking(spec))
            .collect()
    }

    /// Run a single scheduling tick.
    ///
    /// This checks for completed jobs, starts new jobs from the queue,
    /// and advances execution.
    pub fn tick(&mut self) {
        // Update bridge tick
        self.bridge.tick();

        // Move queued jobs to running if capacity available
        while self.running.len() < self.config.max_concurrent && !self.queue.is_empty() {
            if let Some(mut job) = self.queue.pop_front() {
                job.start();
                self.running.insert(job.id, job);
            }
        }

        // Execute running jobs
        // Collect job info first to avoid borrow conflicts
        let jobs_to_execute: Vec<(JobId, JobSpec, usize)> = self
            .running
            .iter()
            .map(|(id, job)| (*id, job.spec.clone(), job.current_stage))
            .collect();

        for (job_id, spec, current_stage) in jobs_to_execute {
            // Execute the job step
            if let Some(result) = self.execute_job_step_internal(&spec, current_stage) {
                // Job completed or failed
                if let Some(mut completed_job) = self.running.remove(&job_id) {
                    if result.success {
                        completed_job.complete(result);
                    } else {
                        completed_job.fail(result);
                    }
                    let job_result = completed_job.into_result();
                    self.completed.insert(job_id, job_result);
                    self.total_completed += 1;
                }
            }
        }
    }

    /// Execute one step of a job (internal implementation).
    ///
    /// Returns Some(StageResult) if the job completed, None if still running.
    fn execute_job_step_internal(
        &mut self,
        spec: &JobSpec,
        current_stage: usize,
    ) -> Option<StageResult> {
        // Find a substrate that can handle this job
        let substrate_idx = self.find_substrate_for(spec)?;

        // Create a job stage
        let stage = JobStage::new(current_stage, spec.clone());

        // Execute on the substrate
        let substrate = &mut self.substrates[substrate_idx];
        let start = Instant::now();

        match substrate.execute(&stage) {
            Ok(result) => Some(result),
            Err(e) => Some(StageResult::failure(
                current_stage,
                start.elapsed(),
                e.to_string(),
            )),
        }
    }

    /// Find a substrate that can handle the given job spec.
    fn find_substrate_for(&self, spec: &JobSpec) -> Option<usize> {
        // Find substrates that can handle this job
        let mut candidates: Vec<(usize, f64)> = self
            .substrates
            .iter()
            .enumerate()
            .filter(|(_, s)| s.can_handle(spec))
            .map(|(i, s)| (i, s.capacity()))
            .collect();

        // Sort by capacity (prefer substrates with more available resources)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        candidates.first().map(|(idx, _)| *idx)
    }

    /// Cancel a job.
    pub fn cancel(&mut self, job_id: JobId) -> bool {
        // Try to remove from queue
        if let Some(pos) = self.queue.iter().position(|j| j.id == job_id) {
            let mut job = self.queue.remove(pos).unwrap();
            job.cancel();
            let result = job.into_result();
            self.completed.insert(job_id, result);
            return true;
        }

        // Try to cancel running job
        if let Some(mut job) = self.running.remove(&job_id) {
            job.cancel();
            let result = job.into_result();
            self.completed.insert(job_id, result);
            return true;
        }

        false
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_submitted: self.total_submitted,
            total_completed: self.total_completed,
            queue_size: self.queue.len(),
            running_count: self.running.len(),
            substrate_count: self.substrates.len(),
            uptime: self.started_at.elapsed(),
        }
    }

    /// List registered substrates.
    pub fn list_substrates(&self) -> Vec<SubstrateInfo> {
        self.substrates
            .iter()
            .map(|s| SubstrateInfo {
                id: s.id(),
                name: s.name().to_string(),
                substrate_type: s.substrate_type(),
                capacity: s.capacity(),
            })
            .collect()
    }

    /// Get the number of queued jobs.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get the number of running jobs.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Check if the scheduler is idle.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.running.is_empty()
    }

    /// Reset all substrates to their initial state.
    pub fn reset_substrates(&mut self) {
        for substrate in &mut self.substrates {
            substrate.reset();
        }
    }

    /// Clear completed job results.
    pub fn clear_completed(&mut self) {
        self.completed.clear();
    }
}

impl Default for HybridScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler statistics.
#[derive(Clone, Debug)]
pub struct SchedulerStats {
    /// Total jobs submitted
    pub total_submitted: u64,

    /// Total jobs completed
    pub total_completed: u64,

    /// Current queue size
    pub queue_size: usize,

    /// Current running count
    pub running_count: usize,

    /// Number of registered substrates
    pub substrate_count: usize,

    /// Scheduler uptime
    pub uptime: Duration,
}

/// Information about a registered substrate.
#[derive(Clone, Debug)]
pub struct SubstrateInfo {
    /// Substrate ID
    pub id: SubstrateId,

    /// Human-readable name
    pub name: String,

    /// Substrate type
    pub substrate_type: SubstrateType,

    /// Current capacity (0.0 to 1.0)
    pub capacity: f64,
}

// ============================================================================
// Phase 3: Enhanced Execution Methods
// ============================================================================

impl HybridScheduler {
    /// Execute a variational hybrid loop (VQE, QAOA pattern).
    ///
    /// This implements the core quantum-classical hybrid loop:
    /// 1. Execute parameterized quantum circuit
    /// 2. Measure and compute cost function
    /// 3. Classical optimization step
    /// 4. Repeat until convergence
    pub fn execute_variational(
        &mut self,
        spec: &crate::hybrid::job::VariationalSpec,
    ) -> Result<StageOutputs, SubstrateError> {
        use crate::hybrid::job::{JobSpec, MeasurementSpec, QuantumSpec};

        let _start = Instant::now();

        // Initialize parameters
        let mut params = spec
            .initial_params
            .clone()
            .unwrap_or_else(|| vec![0.1; spec.ansatz.parameter_count()]);

        let mut history = Vec::new();
        let mut iterations = 0;

        // Variational loop
        for i in 0..spec.max_iterations {
            iterations = i + 1;

            // 1. Bind parameters to circuit
            let bound_circuit = spec.ansatz.bind_parameters(&params);

            // 2. Execute quantum circuit
            let mut quantum_spec = QuantumSpec::new(spec.ansatz.n_qubits, bound_circuit)
                .with_shots(spec.shots)
                .with_measurement(MeasurementSpec::Computational);
            if let Some(s) = spec.seed {
                quantum_spec = quantum_spec.with_seed(s.wrapping_add(i as u64));
            }

            let stage = JobStage::new(i, JobSpec::Quantum(quantum_spec));

            // Find quantum substrate
            let quantum_idx =
                self.find_substrate_for(&stage.spec)
                    .ok_or(SubstrateError::UnsupportedJob(
                        "No quantum substrate available".to_string(),
                    ))?;

            let result = self.substrates[quantum_idx].execute(&stage)?;

            // 3. Compute cost from measurements
            let cost = if let StageOutputs::MeasurementCounts(ref counts) = result.outputs {
                self.evaluate_cost(&spec.cost_function, counts, &params)
            } else {
                0.0
            };

            history.push(cost);

            // 4. Check convergence
            if history.len() > 1 {
                let delta = (history[history.len() - 1] - history[history.len() - 2]).abs();
                if delta < spec.tolerance {
                    break;
                }
            }

            // 5. Classical optimization step
            params = self.optimization_step(&spec.optimizer, &params, cost, i);
        }

        Ok(StageOutputs::Optimization {
            optimal_params: params,
            optimal_cost: *history.last().unwrap_or(&0.0),
            iterations,
            history,
        })
    }

    /// Evaluate cost function from measurement results.
    fn evaluate_cost(
        &self,
        cost_fn: &crate::hybrid::job::CostFunction,
        counts: &std::collections::HashMap<String, usize>,
        params: &[f64],
    ) -> f64 {
        use crate::hybrid::job::CostFunction;

        match cost_fn {
            CostFunction::ExpectationValue(terms) => {
                // Compute expectation value from Pauli terms
                let total_shots: usize = counts.values().sum();
                if total_shots == 0 {
                    return 0.0;
                }

                let mut expectation = 0.0;
                for term in terms {
                    // For each Pauli term, compute its expectation value
                    let mut term_exp = 0.0;
                    for (bitstring, &count) in counts {
                        // Compute parity of measured qubits in this term
                        let mut parity = 1i32;
                        for &(qubit, pauli) in &term.paulis {
                            if pauli == 'Z' {
                                let bit = bitstring
                                    .chars()
                                    .rev()
                                    .nth(qubit)
                                    .map(|c| c == '1')
                                    .unwrap_or(false);
                                if bit {
                                    parity *= -1;
                                }
                            }
                            // X and Y measurement would require different basis
                        }
                        term_exp += parity as f64 * count as f64 / total_shots as f64;
                    }
                    expectation += term.coeff * term_exp;
                }
                expectation
            }
            CostFunction::MaxCut { edges } => {
                // MaxCut: count cuts weighted by probability
                let total_shots: usize = counts.values().sum();
                if total_shots == 0 {
                    return 0.0;
                }

                let mut expected_cut = 0.0;
                for (bitstring, &count) in counts {
                    let prob = count as f64 / total_shots as f64;
                    let mut cut_value = 0.0;
                    for &(i, j, weight) in edges {
                        let bit_i = bitstring
                            .chars()
                            .rev()
                            .nth(i)
                            .map(|c| c == '1')
                            .unwrap_or(false);
                        let bit_j = bitstring
                            .chars()
                            .rev()
                            .nth(j)
                            .map(|c| c == '1')
                            .unwrap_or(false);
                        if bit_i != bit_j {
                            cut_value += weight;
                        }
                    }
                    expected_cut += prob * cut_value;
                }
                // Return negative for minimization
                -expected_cut
            }
            CostFunction::Custom { name: _ } => {
                // Custom cost functions need external evaluation
                params.iter().sum::<f64>().abs() // Placeholder
            }
            CostFunction::Zero => 0.0,
        }
    }

    /// Perform one classical optimization step.
    fn optimization_step(
        &self,
        optimizer: &crate::hybrid::job::Optimizer,
        params: &[f64],
        _cost: f64,
        iteration: usize,
    ) -> Vec<f64> {
        use crate::hybrid::job::Optimizer;

        match optimizer {
            Optimizer::COBYLA { maxiter: _ } | Optimizer::NelderMead { maxiter: _ } => {
                // Simplified: small random perturbation
                // Real implementation would use proper optimization library
                params
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        let noise = 0.01 * ((iteration * 7 + i * 13) % 100) as f64 / 100.0 - 0.005;
                        p + noise
                    })
                    .collect()
            }
            Optimizer::SPSA { a, c, maxiter: _ } => {
                // SPSA: Simultaneous Perturbation Stochastic Approximation
                let ak = a / (iteration + 1) as f64;
                let ck = c / ((iteration + 1) as f64).powf(0.166);

                params
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        let delta = if (iteration + i) % 2 == 0 { 1.0 } else { -1.0 };
                        p - ak * delta * ck
                    })
                    .collect()
            }
            Optimizer::GradientDescent {
                learning_rate,
                maxiter: _,
            } => {
                // Gradient descent with numerical gradient (simplified)
                params
                    .iter()
                    .map(|&p| p - learning_rate * 0.01) // Simplified gradient
                    .collect()
            }
            Optimizer::Adam {
                learning_rate,
                beta1: _,
                beta2: _,
                maxiter: _,
            } => {
                // Simplified Adam
                params.iter().map(|&p| p - learning_rate * 0.01).collect()
            }
        }
    }

    /// Execute a pipeline of jobs sequentially.
    ///
    /// Each stage can reference outputs from previous stages through the data bridge.
    pub fn execute_pipeline(
        &mut self,
        stages: &[JobSpec],
    ) -> Result<Vec<StageResult>, SubstrateError> {
        let mut results = Vec::with_capacity(stages.len());

        for (i, spec) in stages.iter().enumerate() {
            // Store previous result in bridge if available
            if let Some(prev_result) = results.last() {
                self.store_result_in_bridge(i - 1, prev_result);
            }

            // Execute this stage
            let stage = JobStage::new(i, spec.clone());
            let substrate_idx =
                self.find_substrate_for(spec)
                    .ok_or(SubstrateError::UnsupportedJob(format!(
                        "No substrate for pipeline stage {}",
                        i
                    )))?;

            let result = self.substrates[substrate_idx].execute(&stage)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute multiple jobs in parallel.
    ///
    /// Note: In this single-threaded implementation, jobs are executed
    /// round-robin style. True parallelism would require async execution.
    pub fn execute_parallel(
        &mut self,
        jobs: &[JobSpec],
    ) -> Result<Vec<StageResult>, SubstrateError> {
        let mut results = Vec::with_capacity(jobs.len());

        // In a real implementation, this would use async/threads
        // For now, execute sequentially but track as parallel
        for (i, spec) in jobs.iter().enumerate() {
            let stage = JobStage::new(i, spec.clone());
            let substrate_idx =
                self.find_substrate_for(spec)
                    .ok_or(SubstrateError::UnsupportedJob(format!(
                        "No substrate for parallel job {}",
                        i
                    )))?;

            let result = self.substrates[substrate_idx].execute(&stage)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Store a stage result in the data bridge for later retrieval.
    fn store_result_in_bridge(&self, stage_idx: usize, result: &StageResult) {
        let key = format!("stage_{}", stage_idx);

        match &result.outputs {
            StageOutputs::MeasurementCounts(counts) => {
                self.bridge.put_counts(&key, counts.clone(), "scheduler");
            }
            StageOutputs::Optimization { optimal_params, .. } => {
                self.bridge
                    .put_parameters(&key, optimal_params.clone(), "scheduler");
            }
            StageOutputs::Classical { values, .. } => {
                self.bridge.put_floats(&key, values.clone(), "scheduler");
            }
            _ => {}
        }
    }

    /// Submit and execute a variational job, returning the result directly.
    pub fn run_variational(
        &mut self,
        spec: crate::hybrid::job::VariationalSpec,
    ) -> Result<JobResult, SubstrateError> {
        let start = Instant::now();

        let outputs = self.execute_variational(&spec)?;

        let mut result = JobResult::new(
            JobId(self.next_job_id),
            JobState::Completed,
            start.elapsed(),
        );
        result.final_output = outputs;
        self.next_job_id += 1;

        Ok(result)
    }

    /// Submit and execute a pipeline job, returning the result directly.
    pub fn run_pipeline(&mut self, stages: Vec<JobSpec>) -> Result<JobResult, SubstrateError> {
        let start = Instant::now();

        let stage_results = self.execute_pipeline(&stages)?;

        let final_output = stage_results
            .last()
            .map(|r| r.outputs.clone())
            .unwrap_or(StageOutputs::Empty);

        let metrics = JobMetrics::from_stages(&stage_results);

        Ok(JobResult {
            job_id: JobId({
                self.next_job_id += 1;
                self.next_job_id - 1
            }),
            state: JobState::Completed,
            total_duration: start.elapsed(),
            stage_results,
            final_output,
            metrics,
        })
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

// Builder pattern for HybridScheduler
impl HybridScheduler {
    /// Add a classical substrate using builder pattern.
    pub fn with_substrate(mut self, substrate: Box<dyn Substrate>) -> Self {
        self.substrates.push(substrate);
        self
    }

    /// Set configuration using builder pattern.
    pub fn with_config_builder(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum concurrent jobs.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.config.max_concurrent = max;
        self
    }

    /// Set default timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.default_timeout = timeout;
        self
    }
}

// ============================================================================
// Standalone Execution (Sprint 62.0)
// ============================================================================

/// Execute a job standalone (outside the main scheduler).
///
/// This creates fresh substrate instances and executes the job directly.
/// Used by `submit_nonblocking` for independent thread execution.
fn execute_job_standalone(job_id: JobId, spec: JobSpec) -> JobResult {
    use super::classical::ClassicalSubstrate;
    use super::observer::ObserverSubstrate;
    use super::quantum::QuantumSubstrate;

    let start = Instant::now();

    let result = match &spec {
        JobSpec::Quantum(quantum_spec) => {
            // Create a fresh quantum substrate for this job
            let mut substrate = QuantumSubstrate::new(0, "async-quantum", 30);
            if let Some(seed) = quantum_spec.seed {
                substrate = substrate.with_seed(seed);
            }

            let stage = JobStage::new(0, spec.clone());
            match substrate.execute(&stage) {
                Ok(stage_result) => {
                    let mut result = JobResult::new(job_id, JobState::Completed, start.elapsed());
                    result.final_output = stage_result.outputs.clone();
                    result.stage_results.push(stage_result);
                    result
                }
                Err(e) => {
                    let mut result = JobResult::new(job_id, JobState::Failed, start.elapsed());
                    result.stage_results.push(StageResult::failure(
                        0,
                        start.elapsed(),
                        e.to_string(),
                    ));
                    result
                }
            }
        }
        JobSpec::Classical(_classical_spec) => {
            let mut substrate = ClassicalSubstrate::new(0, "async-classical");
            let stage = JobStage::new(0, spec.clone());
            match substrate.execute(&stage) {
                Ok(stage_result) => {
                    let mut result = JobResult::new(job_id, JobState::Completed, start.elapsed());
                    result.final_output = stage_result.outputs.clone();
                    result.stage_results.push(stage_result);
                    result
                }
                Err(e) => {
                    let mut result = JobResult::new(job_id, JobState::Failed, start.elapsed());
                    result.stage_results.push(StageResult::failure(
                        0,
                        start.elapsed(),
                        e.to_string(),
                    ));
                    result
                }
            }
        }
        JobSpec::Observer(_observer_spec) => {
            let mut substrate = ObserverSubstrate::new(0, "async-observer");
            let stage = JobStage::new(0, spec.clone());
            match substrate.execute(&stage) {
                Ok(stage_result) => {
                    let mut result = JobResult::new(job_id, JobState::Completed, start.elapsed());
                    result.final_output = stage_result.outputs.clone();
                    result.stage_results.push(stage_result);
                    result
                }
                Err(e) => {
                    let mut result = JobResult::new(job_id, JobState::Failed, start.elapsed());
                    result.stage_results.push(StageResult::failure(
                        0,
                        start.elapsed(),
                        e.to_string(),
                    ));
                    result
                }
            }
        }
        JobSpec::Variational(_) | JobSpec::Pipeline(_) | JobSpec::Parallel(_) => {
            // Complex job types require scheduler coordination
            // Return failure for now
            let mut result = JobResult::new(job_id, JobState::Failed, start.elapsed());
            result.stage_results.push(StageResult::failure(
                0,
                start.elapsed(),
                "Complex job types not supported in non-blocking mode".to_string(),
            ));
            result
        }
    };

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::job::{ClassicalOp, ClassicalSpec};
    use crate::hybrid::substrate::SubstrateError;

    // Mock substrate for testing
    struct MockSubstrate {
        id: SubstrateId,
        name: String,
        substrate_type: SubstrateType,
        capacity: f64,
    }

    impl MockSubstrate {
        fn new(id: u64, name: &str, st: SubstrateType) -> Self {
            Self {
                id: SubstrateId(id),
                name: name.to_string(),
                substrate_type: st,
                capacity: 1.0,
            }
        }
    }

    impl Substrate for MockSubstrate {
        fn id(&self) -> SubstrateId {
            self.id
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn substrate_type(&self) -> SubstrateType {
            self.substrate_type
        }

        fn capacity(&self) -> f64 {
            self.capacity
        }

        fn can_handle(&self, _spec: &JobSpec) -> bool {
            true // Accept everything for testing
        }

        fn estimate_time(&self, _spec: &JobSpec) -> Duration {
            Duration::from_millis(100)
        }

        fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError> {
            Ok(StageResult::success(
                stage.index,
                Duration::from_millis(50),
                StageOutputs::Empty,
            ))
        }

        fn reset(&mut self) {
            self.capacity = 1.0;
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = HybridScheduler::new();
        assert_eq!(scheduler.substrates.len(), 0);
        assert_eq!(scheduler.queue.len(), 0);
    }

    #[test]
    fn test_substrate_registration() {
        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(MockSubstrate::new(
            1,
            "Test",
            SubstrateType::Classical,
        )));

        assert_eq!(scheduler.substrates.len(), 1);
        let substrates = scheduler.list_substrates();
        assert_eq!(substrates[0].name, "Test");
    }

    #[test]
    fn test_job_submission() {
        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(MockSubstrate::new(
            1,
            "Test",
            SubstrateType::Classical,
        )));

        let spec = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));

        let job_id = scheduler.submit(spec);
        assert_eq!(job_id, JobId(1));
        assert_eq!(scheduler.queue_len(), 1);

        let state = scheduler.job_state(job_id).unwrap();
        assert_eq!(state, JobState::Queued);
    }

    #[test]
    fn test_job_execution() {
        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(MockSubstrate::new(
            1,
            "Test",
            SubstrateType::Classical,
        )));

        let spec = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));

        let job_id = scheduler.submit(spec);

        // Execute until complete
        let result = scheduler.wait(job_id).unwrap();

        assert!(result.is_success());
        assert_eq!(result.state, JobState::Completed);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(MockSubstrate::new(
            1,
            "Test",
            SubstrateType::Classical,
        )));

        let spec = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));

        let job_id = scheduler.submit(spec);
        let _ = scheduler.wait(job_id);

        let stats = scheduler.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.substrate_count, 1);
    }

    #[test]
    fn test_job_cancellation() {
        let mut scheduler = HybridScheduler::new();

        let spec = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));

        let job_id = scheduler.submit(spec);

        assert!(scheduler.cancel(job_id));

        let state = scheduler.job_state(job_id).unwrap();
        assert_eq!(state, JobState::Cancelled);
    }

    #[test]
    fn test_priority_submission() {
        let mut scheduler = HybridScheduler::new();

        let spec1 = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "low".into(),
            inputs: vec![],
        }));

        let spec2 = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "high".into(),
            inputs: vec![],
        }));

        let _job1 = scheduler.submit(spec1);
        let job2 = scheduler.submit_priority(spec2, 10);

        // High priority job should be at front
        assert_eq!(scheduler.queue.front().unwrap().id, job2);
    }

    #[test]
    fn test_tile8_execution_through_scheduler() {
        use crate::hybrid::classical::ClassicalSubstrate;

        let mut scheduler = HybridScheduler::new();
        scheduler.register_substrate(Box::new(ClassicalSubstrate::new(1, "Tile8")));

        // Tile8 ISA: bits 7-4 = opcode, bits 3-2 = Rd, bits 1-0 = Rs/Imm
        // LDI R0, #1 = 0xA << 4 | 0 << 2 | 1 = 0xA1
        // ADD R0, R0 = 0x2 << 4 | 0 << 2 | 0 = 0x20 (R0 = R0 + R0)
        let program = vec![
            0xA1, // LDI R0, #1 -> R0 = 1
            0x20, // ADD R0, R0 -> R0 = 2
            0x20, // ADD R0, R0 -> R0 = 4
        ];
        let spec = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Tile8Program {
            binary: program,
            ticks: 3,
        }));

        let job_id = scheduler.submit(spec);
        let result = scheduler.wait(job_id).unwrap();

        assert!(result.is_success());
        assert_eq!(result.state, JobState::Completed);

        if let StageOutputs::Classical { values, named } = &result.final_output {
            assert_eq!(values.len(), 4);
            assert_eq!(values[0], 4.0); // R0 = 4 after LDI #1 then two doubles
            assert!(named.contains_key("pc"));
        } else {
            panic!("Expected Classical output");
        }
    }
}
