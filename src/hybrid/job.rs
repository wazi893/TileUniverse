//! Job specifications for the hybrid orchestration layer.
//!
//! Jobs describe what computation to perform. The scheduler routes jobs
//! to appropriate substrates based on their type and requirements.

use crate::math_accelerator::CircuitCharacteristics;
use crate::tile8::ObserverGridConfig;
use logic_fabric_core::quantum::QGate;

/// Specification for a hybrid computation job.
#[derive(Clone, Debug)]
pub enum JobSpec {
    /// Pure classical computation
    Classical(ClassicalSpec),

    /// Pure quantum circuit execution
    Quantum(QuantumSpec),

    /// Variational hybrid algorithm (VQE, QAOA)
    Variational(VariationalSpec),

    /// Observer-controlled hybrid computation
    Observer(ObserverSpec),

    /// Multi-stage pipeline (sequential execution)
    Pipeline(Vec<JobSpec>),

    /// Parallel execution of independent jobs
    Parallel(Vec<JobSpec>),
}

impl JobSpec {
    /// Returns whether this is a classical job.
    pub fn is_classical(&self) -> bool {
        matches!(self, JobSpec::Classical(_))
    }

    /// Returns whether this is a quantum job.
    pub fn is_quantum(&self) -> bool {
        matches!(self, JobSpec::Quantum(_))
    }

    /// Returns whether this is a variational job.
    pub fn is_variational(&self) -> bool {
        matches!(self, JobSpec::Variational(_))
    }

    /// Returns whether this is an observer job.
    pub fn is_observer(&self) -> bool {
        matches!(self, JobSpec::Observer(_))
    }

    /// Returns whether this is a composite job (pipeline or parallel).
    pub fn is_composite(&self) -> bool {
        matches!(self, JobSpec::Pipeline(_) | JobSpec::Parallel(_))
    }

    /// Estimates the number of qubits needed for this job.
    pub fn qubit_requirement(&self) -> Option<usize> {
        match self {
            JobSpec::Quantum(q) => Some(q.n_qubits),
            JobSpec::Variational(v) => Some(v.ansatz.n_qubits),
            JobSpec::Observer(o) => o.quantum_state.as_ref().map(|q| q.n_qubits),
            JobSpec::Pipeline(stages) => stages.iter().filter_map(|s| s.qubit_requirement()).max(),
            JobSpec::Parallel(jobs) => jobs.iter().filter_map(|j| j.qubit_requirement()).max(),
            _ => None,
        }
    }
}

// ============================================================================
// Classical Job Specification
// ============================================================================

/// Specification for classical computation.
#[derive(Clone, Debug)]
pub struct ClassicalSpec {
    /// The classical operation to perform
    pub operation: ClassicalOp,
}

impl ClassicalSpec {
    /// Create a new classical spec with the given operation.
    pub fn new(operation: ClassicalOp) -> Self {
        Self { operation }
    }

    /// Create a Tile8 program execution spec.
    pub fn tile8_program(binary: Vec<u8>, ticks: usize) -> Self {
        Self {
            operation: ClassicalOp::Tile8Program { binary, ticks },
        }
    }

    /// Create an optimization step spec.
    pub fn optimize(method: Optimizer, params: Vec<f64>) -> Self {
        Self {
            operation: ClassicalOp::Optimize { method, params },
        }
    }
}

/// Classical operations.
#[derive(Clone, Debug)]
pub enum ClassicalOp {
    /// Run a Tile8 CPU program
    Tile8Program {
        /// Compiled binary
        binary: Vec<u8>,
        /// Number of ticks to execute
        ticks: usize,
    },

    /// Classical optimization step
    Optimize {
        /// Optimization method
        method: Optimizer,
        /// Current parameters
        params: Vec<f64>,
    },

    /// Post-processing operation (e.g., continued fractions for Shor's)
    PostProcess {
        /// Algorithm identifier
        algorithm: String,
        /// Input data
        inputs: Vec<f64>,
    },

    /// Evaluate a cost function
    EvaluateCost {
        /// Cost function to evaluate
        cost_fn: CostFunction,
        /// Parameters
        params: Vec<f64>,
    },

    /// Custom classical operation
    Custom {
        /// Operation name
        name: String,
        /// Input values
        inputs: Vec<f64>,
    },
}

// ============================================================================
// Quantum Job Specification
// ============================================================================

/// Specification for quantum circuit execution.
#[derive(Clone, Debug)]
pub struct QuantumSpec {
    /// Number of qubits
    pub n_qubits: usize,

    /// Circuit to execute
    pub circuit: QuantumCircuit,

    /// Measurement specification
    pub measurement: MeasurementSpec,

    /// Number of shots (repetitions)
    pub shots: usize,

    /// Random seed for measurement (None = random)
    pub seed: Option<u64>,

    /// Cached circuit characteristics for routing decisions (Sprint 76.0).
    /// Computed lazily on first access via `get_or_compute_characteristics()`.
    cached_characteristics: Option<CircuitCharacteristics>,
}

impl QuantumSpec {
    /// Create a new quantum spec.
    pub fn new(n_qubits: usize, circuit: QuantumCircuit) -> Self {
        Self {
            n_qubits,
            circuit,
            measurement: MeasurementSpec::Computational,
            shots: 1024,
            seed: None,
            cached_characteristics: None,
        }
    }

    /// Get or compute circuit characteristics for routing decisions.
    ///
    /// This method caches the result after first computation, so subsequent
    /// calls are O(1). The initial computation is O(n) where n is gate count.
    ///
    /// # Sprint 76.0
    ///
    /// Used by the router to make intelligent backend decisions based on
    /// mathematical analysis of the circuit structure.
    pub fn get_or_compute_characteristics(&mut self) -> &CircuitCharacteristics {
        if self.cached_characteristics.is_none() {
            let qgates = self.circuit.to_qgates();
            let chars = crate::math_accelerator::analyze_circuit(&qgates, self.n_qubits as u8);
            self.cached_characteristics = Some(chars);
        }
        self.cached_characteristics.as_ref().unwrap()
    }

    /// Returns a reference to cached characteristics if already computed.
    ///
    /// Returns `None` if characteristics have not been computed yet.
    /// Use `get_or_compute_characteristics()` to ensure computation.
    pub fn cached_characteristics(&self) -> Option<&CircuitCharacteristics> {
        self.cached_characteristics.as_ref()
    }

    /// Clear cached characteristics.
    ///
    /// Call this if the circuit is modified after characteristics were computed.
    pub fn invalidate_characteristics(&mut self) {
        self.cached_characteristics = None;
    }

    /// Create a uniform superposition state spec.
    pub fn uniform_superposition(n_qubits: usize) -> Self {
        let mut circuit = QuantumCircuit::new(n_qubits);
        for i in 0..n_qubits {
            circuit.h(i);
        }
        Self::new(n_qubits, circuit)
    }

    /// Create a GHZ state spec.
    pub fn ghz_state(n_qubits: usize) -> Self {
        let mut circuit = QuantumCircuit::new(n_qubits);
        circuit.h(0);
        for i in 1..n_qubits {
            circuit.cnot(0, i);
        }
        Self::new(n_qubits, circuit)
    }

    /// Set the number of shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the measurement specification.
    pub fn with_measurement(mut self, measurement: MeasurementSpec) -> Self {
        self.measurement = measurement;
        self
    }
}

/// A quantum circuit representation.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub n_qubits: usize,

    /// Gate sequence
    pub gates: Vec<Gate>,

    /// Parameter names (for variational circuits)
    pub parameters: Vec<String>,
}

impl QuantumCircuit {
    /// Create a new empty circuit.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            parameters: Vec::new(),
        }
    }

    /// Add a Hadamard gate.
    pub fn h(&mut self, qubit: usize) {
        self.gates.push(Gate::H(qubit));
    }

    /// Add a Pauli-X gate.
    pub fn x(&mut self, qubit: usize) {
        self.gates.push(Gate::X(qubit));
    }

    /// Add a Pauli-Y gate.
    pub fn y(&mut self, qubit: usize) {
        self.gates.push(Gate::Y(qubit));
    }

    /// Add a Pauli-Z gate.
    pub fn z(&mut self, qubit: usize) {
        self.gates.push(Gate::Z(qubit));
    }

    /// Add a CNOT gate.
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.gates.push(Gate::CNOT(control, target));
    }

    /// Add an Rz rotation gate.
    pub fn rz(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::Rz(qubit, angle));
    }

    /// Add an Ry rotation gate.
    pub fn ry(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::Ry(qubit, angle));
    }

    /// Add an Rx rotation gate.
    pub fn rx(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::Rx(qubit, angle));
    }

    /// Add a parameterized Rz gate.
    pub fn rz_param(&mut self, qubit: usize, param_name: &str) {
        let param_idx = self.parameters.len();
        self.parameters.push(param_name.to_string());
        self.gates.push(Gate::RzParam(qubit, param_idx));
    }

    /// Add a parameterized Ry gate.
    pub fn ry_param(&mut self, qubit: usize, param_name: &str) {
        let param_idx = self.parameters.len();
        self.parameters.push(param_name.to_string());
        self.gates.push(Gate::RyParam(qubit, param_idx));
    }

    /// Returns the number of gates in the circuit.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Returns the number of parameters.
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Convert circuit gates to `QGate` representation for math_accelerator analysis.
    ///
    /// # Sprint 76.0
    ///
    /// This enables the mathematical analysis module to analyze circuits
    /// defined using the hybrid job specification format.
    pub fn to_qgates(&self) -> Vec<QGate> {
        self.gates.iter().filter_map(|g| g.to_qgate()).collect()
    }

    /// Bind parameter values to create a concrete circuit.
    pub fn bind_parameters(&self, values: &[f64]) -> QuantumCircuit {
        let mut bound = self.clone();
        bound.gates = self
            .gates
            .iter()
            .map(|g| match g {
                Gate::RzParam(q, idx) => Gate::Rz(*q, values.get(*idx).copied().unwrap_or(0.0)),
                Gate::RyParam(q, idx) => Gate::Ry(*q, values.get(*idx).copied().unwrap_or(0.0)),
                Gate::RxParam(q, idx) => Gate::Rx(*q, values.get(*idx).copied().unwrap_or(0.0)),
                other => other.clone(),
            })
            .collect();
        bound.parameters.clear();
        bound
    }

    /// Create a hardware-efficient ansatz.
    pub fn hardware_efficient(n_qubits: usize, depth: usize) -> Self {
        let mut circuit = Self::new(n_qubits);

        for d in 0..depth {
            // Single-qubit rotation layer
            for q in 0..n_qubits {
                circuit.ry_param(q, &format!("ry_{}_{}", d, q));
                circuit.rz_param(q, &format!("rz_{}_{}", d, q));
            }

            // Entangling layer (linear connectivity)
            for q in 0..n_qubits.saturating_sub(1) {
                circuit.cnot(q, q + 1);
            }
        }

        // Final rotation layer
        for q in 0..n_qubits {
            circuit.ry_param(q, &format!("ry_final_{}", q));
        }

        circuit
    }
}

/// Quantum gate types.
#[derive(Clone, Debug)]
pub enum Gate {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),

    // Rotation gates (fixed angle)
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),

    // Parameterized rotation gates (parameter index)
    RxParam(usize, usize),
    RyParam(usize, usize),
    RzParam(usize, usize),

    // Two-qubit gates
    CNOT(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),

    // Controlled rotation
    CRz(usize, usize, f64),

    // Measurement (mid-circuit)
    Measure(usize),
}

impl Gate {
    /// Convert to the core `QGate` representation.
    ///
    /// Returns `None` for gates that don't have a direct `QGate` equivalent
    /// (e.g., parameterized gates that haven't been bound yet).
    ///
    /// # Sprint 76.0
    ///
    /// This conversion enables mathematical analysis of circuits for
    /// intelligent routing decisions.
    pub fn to_qgate(&self) -> Option<QGate> {
        match self {
            Gate::H(q) => Some(QGate::H(*q as u8)),
            Gate::X(q) => Some(QGate::X(*q as u8)),
            Gate::Y(q) => Some(QGate::Y(*q as u8)),
            Gate::Z(q) => Some(QGate::Z(*q as u8)),
            Gate::S(q) => Some(QGate::Phase(*q as u8, std::f32::consts::FRAC_PI_2)),
            Gate::T(q) => Some(QGate::Phase(*q as u8, std::f32::consts::FRAC_PI_4)),
            Gate::Rx(q, angle) => Some(QGate::Rx(*q as u8, *angle as f32)),
            Gate::Ry(q, angle) => Some(QGate::Ry(*q as u8, *angle as f32)),
            Gate::Rz(q, angle) => Some(QGate::Rz(*q as u8, *angle as f32)),
            // Parameterized gates cannot be converted without binding
            Gate::RxParam(_, _) | Gate::RyParam(_, _) | Gate::RzParam(_, _) => None,
            Gate::CNOT(ctrl, tgt) => Some(QGate::CNot(*ctrl as u8, *tgt as u8)),
            Gate::CZ(ctrl, tgt) => Some(QGate::CZ(*ctrl as u8, *tgt as u8)),
            Gate::SWAP(a, b) => Some(QGate::Swap(*a as u8, *b as u8)),
            Gate::CRz(ctrl, tgt, angle) => Some(QGate::CRz(*ctrl as u8, *tgt as u8, *angle as f32)),
            Gate::Measure(q) => Some(QGate::Measure(*q as u8)),
        }
    }
}

/// Measurement specification.
#[derive(Clone, Debug)]
pub enum MeasurementSpec {
    /// Measure all qubits in computational basis
    Computational,

    /// Measure specific qubits
    Partial(Vec<usize>),

    /// No measurement (return state vector)
    None,

    /// Measure in specified basis per qubit
    Custom(Vec<MeasurementBasis>),
}

/// Measurement basis for a single qubit.
#[derive(Clone, Copy, Debug)]
pub enum MeasurementBasis {
    /// Computational basis (Z)
    Z,
    /// X basis
    X,
    /// Y basis
    Y,
}

// ============================================================================
// Variational Job Specification
// ============================================================================

/// Specification for variational hybrid algorithms (VQE, QAOA).
#[derive(Clone, Debug)]
pub struct VariationalSpec {
    /// Parameterized quantum circuit (ansatz)
    pub ansatz: QuantumCircuit,

    /// Classical optimizer
    pub optimizer: Optimizer,

    /// Cost function to minimize
    pub cost_function: CostFunction,

    /// Maximum iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Initial parameters (None = random initialization)
    pub initial_params: Option<Vec<f64>>,

    /// Number of shots per evaluation
    pub shots: usize,

    /// Random seed
    pub seed: Option<u64>,
}

impl Default for VariationalSpec {
    fn default() -> Self {
        Self {
            ansatz: QuantumCircuit::new(2),
            optimizer: Optimizer::COBYLA { maxiter: 100 },
            cost_function: CostFunction::Zero,
            max_iterations: 100,
            tolerance: 1e-6,
            initial_params: None,
            shots: 1024,
            seed: None,
        }
    }
}

impl VariationalSpec {
    /// Create a new variational spec.
    pub fn new(ansatz: QuantumCircuit, cost_function: CostFunction) -> Self {
        Self {
            ansatz,
            cost_function,
            ..Default::default()
        }
    }

    /// Set the optimizer.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the initial parameters.
    pub fn with_initial_params(mut self, params: Vec<f64>) -> Self {
        self.initial_params = Some(params);
        self
    }

    /// Set the number of shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }
}

/// Classical optimizers for variational algorithms.
#[derive(Clone, Debug)]
pub enum Optimizer {
    /// Constrained Optimization BY Linear Approximation
    COBYLA { maxiter: usize },

    /// Nelder-Mead simplex method
    NelderMead { maxiter: usize },

    /// Simultaneous Perturbation Stochastic Approximation
    SPSA { a: f64, c: f64, maxiter: usize },

    /// Gradient descent
    GradientDescent { learning_rate: f64, maxiter: usize },

    /// Adam optimizer
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        maxiter: usize,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::COBYLA { maxiter: 100 }
    }
}

/// Cost functions for variational algorithms.
#[derive(Clone, Debug)]
pub enum CostFunction {
    /// Expectation value of a Hamiltonian (represented as Pauli strings)
    ExpectationValue(Vec<PauliTerm>),

    /// MaxCut objective for a graph
    MaxCut {
        /// Edges as (vertex1, vertex2, weight)
        edges: Vec<(usize, usize, f64)>,
    },

    /// Custom cost function (returns cost given measurement counts)
    Custom {
        /// Name for identification
        name: String,
    },

    /// Zero cost function (for testing)
    Zero,
}

/// A term in a Pauli Hamiltonian: coefficient * P1 ⊗ P2 ⊗ ... ⊗ Pn
#[derive(Clone, Debug)]
pub struct PauliTerm {
    /// Coefficient
    pub coeff: f64,

    /// Pauli operators indexed by qubit (only non-identity terms)
    /// (qubit_index, pauli_type) where pauli_type is 'X', 'Y', or 'Z'
    pub paulis: Vec<(usize, char)>,
}

impl PauliTerm {
    /// Create a new Pauli term.
    pub fn new(coeff: f64, paulis: Vec<(usize, char)>) -> Self {
        Self { coeff, paulis }
    }

    /// Create an identity term (scalar).
    pub fn identity(coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![],
        }
    }

    /// Create a single Z term.
    pub fn z(coeff: f64, qubit: usize) -> Self {
        Self {
            coeff,
            paulis: vec![(qubit, 'Z')],
        }
    }

    /// Create a ZZ term.
    pub fn zz(coeff: f64, q1: usize, q2: usize) -> Self {
        Self {
            coeff,
            paulis: vec![(q1, 'Z'), (q2, 'Z')],
        }
    }
}

// ============================================================================
// Observer Job Specification
// ============================================================================

/// Specification for observer-controlled hybrid computation.
#[derive(Clone, Debug)]
pub struct ObserverSpec {
    /// Initial quantum state (optional - can start from scratch)
    pub quantum_state: Option<QuantumSpec>,

    /// Observer grid configuration
    pub observer_config: ObserverGridConfig,

    /// Trigger condition for collapse/termination
    pub trigger: ObserverTrigger,

    /// Maximum ticks to run
    pub max_ticks: u64,
}

impl ObserverSpec {
    /// Create a new observer spec with default configuration.
    pub fn new(observer_config: ObserverGridConfig) -> Self {
        Self {
            quantum_state: None,
            observer_config,
            trigger: ObserverTrigger::AfterTicks(100),
            max_ticks: 10000,
        }
    }

    /// Set the initial quantum state.
    pub fn with_quantum_state(mut self, spec: QuantumSpec) -> Self {
        self.quantum_state = Some(spec);
        self
    }

    /// Set the trigger condition.
    pub fn with_trigger(mut self, trigger: ObserverTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Set the maximum ticks.
    pub fn with_max_ticks(mut self, max_ticks: u64) -> Self {
        self.max_ticks = max_ticks;
        self
    }
}

/// Trigger conditions for observer-controlled computation.
#[derive(Clone, Debug)]
pub enum ObserverTrigger {
    /// Terminate after N ticks
    AfterTicks(u64),

    /// Terminate when anomaly score exceeds threshold
    OnAnomaly { threshold: f64 },

    /// Terminate when collapse ratio exceeds threshold
    OnCollapseRatio { threshold: f64 },

    /// Terminate when all observers have collapsed
    OnFullCollapse,

    /// External/manual trigger (never auto-terminate)
    Manual,

    /// Compound trigger: any of the conditions
    Any(Vec<ObserverTrigger>),

    /// Compound trigger: all of the conditions
    All(Vec<ObserverTrigger>),
}

impl Default for ObserverTrigger {
    fn default() -> Self {
        ObserverTrigger::AfterTicks(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_circuit_basic() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);

        assert_eq!(circuit.n_qubits, 2);
        assert_eq!(circuit.gate_count(), 2);
    }

    #[test]
    fn test_quantum_circuit_hardware_efficient() {
        let circuit = QuantumCircuit::hardware_efficient(4, 2);

        assert_eq!(circuit.n_qubits, 4);
        assert!(circuit.parameter_count() > 0);
    }

    #[test]
    fn test_quantum_circuit_bind_parameters() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.ry_param(0, "theta");
        circuit.rz_param(1, "phi");

        assert_eq!(circuit.parameter_count(), 2);

        let bound = circuit.bind_parameters(&[0.5, 1.0]);
        assert_eq!(bound.parameter_count(), 0);
    }

    #[test]
    fn test_variational_spec_default() {
        let spec = VariationalSpec::default();
        assert_eq!(spec.max_iterations, 100);
        assert_eq!(spec.shots, 1024);
    }

    #[test]
    fn test_job_spec_qubit_requirement() {
        let quantum_job = JobSpec::Quantum(QuantumSpec::new(10, QuantumCircuit::new(10)));
        assert_eq!(quantum_job.qubit_requirement(), Some(10));

        let classical_job = JobSpec::Classical(ClassicalSpec::new(ClassicalOp::Custom {
            name: "test".into(),
            inputs: vec![],
        }));
        assert_eq!(classical_job.qubit_requirement(), None);
    }
}
