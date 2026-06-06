//! Classical substrate implementation.
//!
//! Wraps Tile8 CPU simulation for classical computation within
//! the hybrid orchestration framework.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::tile8::{create_cpu_states, run_parallel};

use super::job::{ClassicalOp, ClassicalSpec, CostFunction, JobSpec, Optimizer, PauliTerm};
use super::result::{JobStage, StageMetrics, StageOutputs, StageResult};
use super::substrate::{
    Substrate, SubstrateCapabilities, SubstrateError, SubstrateId, SubstrateType,
};

/// Classical compute substrate using Tile8 CPUs.
///
/// This substrate handles classical computation including:
/// - Tile8 CPU program execution
/// - Classical optimization steps (COBYLA, NelderMead, etc.)
/// - Cost function evaluation
/// - Post-processing operations
pub struct ClassicalSubstrate {
    /// Unique identifier
    id: SubstrateId,

    /// Human-readable name
    name: String,

    /// Current capacity (0.0 = busy, 1.0 = idle)
    capacity: f64,

    /// Number of parallel workers
    num_workers: usize,

    /// Whether GPU acceleration is available
    gpu_available: bool,

    /// Current jobs in flight
    jobs_in_flight: usize,
}

impl ClassicalSubstrate {
    /// Create a new classical substrate.
    pub fn new(id: u64, name: &str) -> Self {
        Self {
            id: SubstrateId(id),
            name: name.to_string(),
            capacity: 1.0,
            num_workers: num_cpus(),
            gpu_available: false,
            jobs_in_flight: 0,
        }
    }

    /// Create a substrate with GPU acceleration.
    pub fn with_gpu(mut self) -> Self {
        self.gpu_available = true;
        self
    }

    /// Set the number of workers.
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers;
        self
    }

    /// Execute a classical operation.
    fn execute_classical(&mut self, spec: &ClassicalSpec) -> Result<StageOutputs, SubstrateError> {
        match &spec.operation {
            ClassicalOp::Tile8Program { binary, ticks } => {
                self.execute_tile8_program(binary, *ticks)
            }
            ClassicalOp::Optimize { method, params } => self.execute_optimize(method, params),
            ClassicalOp::PostProcess { algorithm, inputs } => {
                self.execute_post_process(algorithm, inputs)
            }
            ClassicalOp::EvaluateCost { cost_fn, params } => {
                self.execute_evaluate_cost(cost_fn, params)
            }
            ClassicalOp::Custom { name, inputs } => self.execute_custom(name, inputs),
        }
    }

    /// Execute a Tile8 program.
    ///
    /// Runs the Tile8 CPU simulator with the given binary program for the
    /// specified number of ticks. Uses parallel execution via Rayon when
    /// multiple CPU instances are configured.
    ///
    /// # Output Format
    /// Returns `StageOutputs::Classical` with:
    /// - `values`: Final register values [R0, R1, R2, R3] as f64
    /// - `named`: Contains "pc", "flags", "ram" for detailed state access
    fn execute_tile8_program(
        &mut self,
        binary: &[u8],
        ticks: usize,
    ) -> Result<StageOutputs, SubstrateError> {
        if binary.is_empty() {
            return Err(SubstrateError::InvalidSpec(
                "Empty Tile8 program binary".to_string(),
            ));
        }

        // Create CPU state from the binary program
        // For now, run a single CPU; could extend to parallel instances
        let mut states = create_cpu_states(1, binary);

        // Execute for the specified number of ticks
        run_parallel(&mut states, ticks);

        // Extract final state from the first (and only) CPU
        let cpu = &states[0];

        // Pack register values into output
        let values: Vec<f64> = cpu.regs.iter().map(|&r| r as f64).collect();

        // Pack detailed state into named outputs
        let mut named = HashMap::new();
        named.insert("pc".to_string(), vec![cpu.pc as f64]);
        named.insert(
            "flags".to_string(),
            vec![cpu.flags_z as u8 as f64, cpu.flags_c as u8 as f64],
        );
        named.insert(
            "ram".to_string(),
            cpu.ram.iter().map(|&b| b as f64).collect(),
        );

        Ok(StageOutputs::Classical { values, named })
    }

    /// Execute an optimization step.
    fn execute_optimize(
        &mut self,
        method: &Optimizer,
        params: &[f64],
    ) -> Result<StageOutputs, SubstrateError> {
        // Placeholder optimization - just return the params unchanged
        // Real implementation would perform actual optimization step
        match method {
            Optimizer::COBYLA { maxiter: _ } => {
                // COBYLA step
                Ok(StageOutputs::Optimization {
                    optimal_params: params.to_vec(),
                    optimal_cost: 0.0,
                    iterations: 1,
                    history: vec![0.0],
                })
            }
            Optimizer::NelderMead { maxiter: _ } => {
                // Nelder-Mead step
                Ok(StageOutputs::Optimization {
                    optimal_params: params.to_vec(),
                    optimal_cost: 0.0,
                    iterations: 1,
                    history: vec![0.0],
                })
            }
            Optimizer::SPSA {
                a: _,
                c: _,
                maxiter: _,
            } => {
                // SPSA step
                Ok(StageOutputs::Optimization {
                    optimal_params: params.to_vec(),
                    optimal_cost: 0.0,
                    iterations: 1,
                    history: vec![0.0],
                })
            }
            Optimizer::GradientDescent {
                learning_rate: _,
                maxiter: _,
            } => {
                // Gradient descent step
                Ok(StageOutputs::Optimization {
                    optimal_params: params.to_vec(),
                    optimal_cost: 0.0,
                    iterations: 1,
                    history: vec![0.0],
                })
            }
            Optimizer::Adam {
                learning_rate: _,
                beta1: _,
                beta2: _,
                maxiter: _,
            } => {
                // Adam step
                Ok(StageOutputs::Optimization {
                    optimal_params: params.to_vec(),
                    optimal_cost: 0.0,
                    iterations: 1,
                    history: vec![0.0],
                })
            }
        }
    }

    /// Execute post-processing.
    fn execute_post_process(
        &mut self,
        algorithm: &str,
        inputs: &[f64],
    ) -> Result<StageOutputs, SubstrateError> {
        match algorithm {
            "continued_fractions" => {
                // Post-processing for Shor's algorithm
                // Real implementation would do continued fraction expansion
                Ok(StageOutputs::Classical {
                    values: inputs.to_vec(),
                    named: std::collections::HashMap::new(),
                })
            }
            _ => Err(SubstrateError::UnsupportedJob(format!(
                "Unknown post-processing algorithm: {}",
                algorithm
            ))),
        }
    }

    /// Evaluate a cost function.
    fn execute_evaluate_cost(
        &mut self,
        cost_fn: &CostFunction,
        params: &[f64],
    ) -> Result<StageOutputs, SubstrateError> {
        let cost = match cost_fn {
            CostFunction::ExpectationValue(terms) => self.evaluate_expectation(terms, params),
            CostFunction::MaxCut { edges } => self.evaluate_maxcut(edges, params),
            CostFunction::Custom { name: _ } => 0.0, // Custom functions need external evaluation
            CostFunction::Zero => 0.0,
        };

        Ok(StageOutputs::Classical {
            values: vec![cost],
            named: std::collections::HashMap::new(),
        })
    }

    /// Evaluate expectation value of Pauli Hamiltonian.
    fn evaluate_expectation(&self, terms: &[PauliTerm], _params: &[f64]) -> f64 {
        // Placeholder - real implementation would compute expectation values
        // from measurement outcomes
        terms.iter().map(|t| t.coeff).sum()
    }

    /// Evaluate MaxCut cost.
    fn evaluate_maxcut(&self, edges: &[(usize, usize, f64)], params: &[f64]) -> f64 {
        // Simple MaxCut evaluation - sum of edges where endpoints differ
        let mut cost = 0.0;
        for (i, j, weight) in edges {
            if params.get(*i).map(|&v| v > 0.5).unwrap_or(false)
                != params.get(*j).map(|&v| v > 0.5).unwrap_or(false)
            {
                cost += weight;
            }
        }
        cost
    }

    /// Execute custom operation.
    fn execute_custom(
        &mut self,
        _name: &str,
        inputs: &[f64],
    ) -> Result<StageOutputs, SubstrateError> {
        // Custom operations return inputs unchanged
        Ok(StageOutputs::Classical {
            values: inputs.to_vec(),
            named: std::collections::HashMap::new(),
        })
    }

    /// Update capacity based on jobs in flight.
    fn update_capacity(&mut self) {
        self.capacity = if self.num_workers > 0 {
            1.0 - (self.jobs_in_flight as f64 / self.num_workers as f64)
        } else {
            0.0
        };
    }
}

impl Substrate for ClassicalSubstrate {
    fn id(&self) -> SubstrateId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn substrate_type(&self) -> SubstrateType {
        SubstrateType::Classical
    }

    fn capacity(&self) -> f64 {
        self.capacity
    }

    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            max_qubits: None,
            max_depth: None,
            gpu_available: self.gpu_available,
            max_parallel_jobs: self.num_workers,
            supported_gates: vec![],
            supports_variational: true, // Classical optimizer support
            supports_observer: false,
        }
    }

    fn can_handle(&self, spec: &JobSpec) -> bool {
        matches!(spec, JobSpec::Classical(_))
    }

    fn estimate_time(&self, spec: &JobSpec) -> Duration {
        match spec {
            JobSpec::Classical(classical) => match &classical.operation {
                ClassicalOp::Tile8Program { ticks, .. } => {
                    // Rough estimate: 1ms per 1000 ticks
                    Duration::from_millis((*ticks as u64) / 1000 + 1)
                }
                ClassicalOp::Optimize { method, .. } => match method {
                    Optimizer::COBYLA { maxiter } => Duration::from_millis(*maxiter as u64 * 10),
                    Optimizer::NelderMead { maxiter } => Duration::from_millis(*maxiter as u64 * 5),
                    Optimizer::SPSA { maxiter, .. } => Duration::from_millis(*maxiter as u64 * 2),
                    Optimizer::GradientDescent { maxiter, .. } => {
                        Duration::from_millis(*maxiter as u64)
                    }
                    Optimizer::Adam { maxiter, .. } => Duration::from_millis(*maxiter as u64),
                },
                ClassicalOp::PostProcess { .. } => Duration::from_millis(10),
                ClassicalOp::EvaluateCost { .. } => Duration::from_millis(1),
                ClassicalOp::Custom { .. } => Duration::from_millis(100),
            },
            _ => Duration::from_millis(0),
        }
    }

    fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError> {
        self.jobs_in_flight += 1;
        self.update_capacity();

        let start = Instant::now();

        let result = match &stage.spec {
            JobSpec::Classical(spec) => self.execute_classical(spec),
            _ => Err(SubstrateError::UnsupportedJob(
                "ClassicalSubstrate only handles Classical jobs".to_string(),
            )),
        };

        let duration = start.elapsed();

        self.jobs_in_flight = self.jobs_in_flight.saturating_sub(1);
        self.update_capacity();

        match result {
            Ok(outputs) => Ok(StageResult::success(stage.index, duration, outputs)
                .with_metrics(StageMetrics::default())),
            Err(e) => Ok(StageResult::failure(stage.index, duration, e.to_string())),
        }
    }

    fn reset(&mut self) {
        self.jobs_in_flight = 0;
        self.capacity = 1.0;
    }
}

/// Get number of available CPUs.
fn num_cpus() -> usize {
    // Simple heuristic - could use actual CPU detection
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_substrate_creation() {
        let substrate = ClassicalSubstrate::new(1, "TestClassical");

        assert_eq!(substrate.name(), "TestClassical");
        assert_eq!(substrate.substrate_type(), SubstrateType::Classical);
        assert!(substrate.capacity() > 0.99);
    }

    #[test]
    fn test_can_handle() {
        let substrate = ClassicalSubstrate::new(1, "TestClassical");

        let classical_job = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));

        assert!(substrate.can_handle(&classical_job));
    }

    #[test]
    fn test_execute_custom() {
        let mut substrate = ClassicalSubstrate::new(1, "TestClassical");

        let spec = ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![1.0, 2.0, 3.0],
        });

        let stage = JobStage::new(0, JobSpec::Classical(spec));
        let result = substrate.execute(&stage).unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_estimate_time() {
        let substrate = ClassicalSubstrate::new(1, "TestClassical");

        let spec = ClassicalSpec::new(ClassicalOp::Tile8Program {
            binary: vec![],
            ticks: 10000,
        });

        let estimate = substrate.estimate_time(&JobSpec::Classical(spec));
        assert!(estimate > Duration::ZERO);
    }

    #[test]
    fn test_capacity_update() {
        let mut substrate = ClassicalSubstrate::new(1, "TestClassical").with_workers(2);

        assert!(substrate.capacity() > 0.99);

        substrate.jobs_in_flight = 1;
        substrate.update_capacity();
        assert!(substrate.capacity() < 1.0);
        assert!(substrate.capacity() > 0.0);

        substrate.jobs_in_flight = 2;
        substrate.update_capacity();
        assert!(substrate.capacity() < 0.01);
    }

    #[test]
    fn test_execute_tile8_program() {
        let mut substrate = ClassicalSubstrate::new(1, "TestClassical");

        // Simple program: LDI R0, #1; ADD R0, R0 (x3)
        // After 4 ticks: R0 = 1 -> 2 -> 4 -> 8
        let program = vec![
            0xA1, // LDI R0, #1
            0x20, // ADD R0, R0 -> 2
            0x20, // ADD R0, R0 -> 4
            0x20, // ADD R0, R0 -> 8
        ];

        let spec = ClassicalSpec::new(ClassicalOp::Tile8Program {
            binary: program,
            ticks: 4,
        });

        let stage = JobStage::new(0, JobSpec::Classical(spec));
        let result = substrate.execute(&stage).unwrap();

        assert!(result.success);

        if let StageOutputs::Classical { values, named } = &result.outputs {
            // R0 should be 8 after 4 iterations of doubling from 1
            assert_eq!(values.len(), 4); // 4 registers
            assert_eq!(values[0], 8.0); // R0 = 8

            // Check PC advanced
            assert!(named.contains_key("pc"));
            assert!(named.contains_key("flags"));
            assert!(named.contains_key("ram"));
        } else {
            panic!("Expected Classical output");
        }
    }

    #[test]
    fn test_execute_tile8_empty_program_fails() {
        let mut substrate = ClassicalSubstrate::new(1, "TestClassical");

        let spec = ClassicalSpec::new(ClassicalOp::Tile8Program {
            binary: vec![],
            ticks: 10,
        });

        let stage = JobStage::new(0, JobSpec::Classical(spec));
        let result = substrate.execute(&stage).unwrap();

        // Should fail with empty program
        assert!(!result.success);
    }
}
