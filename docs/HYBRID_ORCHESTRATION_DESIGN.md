# Quantum-HPC Hybrid Orchestration Layer

## Inspired by RIKEN's JHPC-Quantum Architecture

RIKEN's approach to quantum-classical hybrid computing provides a blueprint for integrating our existing substrates into a unified system.

---

## Current State: Disconnected Substrates

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Tile CPUs      │  │   Quantum Grid   │  │   Observer Grid  │
│   268M cells     │  │   Sparse states  │  │   Hybrid layer   │
│   Classical sim  │  │   GHZ/W/general  │  │   Collapse ctrl  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                        No orchestration
                        No unified API
                        Manual wiring only
```

---

## Proposed: Hybrid Orchestration Layer (Sprint 61+)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HybridOrchestrator                           │
│  - Job queue management                                         │
│  - Resource allocation                                          │
│  - Substrate routing                                            │
│  - Result aggregation                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ClassicalEngine │ │  QuantumEngine  │ │ ObserverEngine  │
│                 │ │                 │ │                 │
│ - Tile8 CPUs    │ │ - SparseQuantum │ │ - ObserverGrid  │
│ - SlimSim       │ │ - QuantumGrid   │ │ - AnomalyDetect │
│ - GPU parallel  │ │ - GPU kernels   │ │ - Measurement   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SubstrateInterface                           │
│  - Unified memory model                                         │
│  - Cross-substrate data transfer                                │
│  - Synchronization primitives                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. HybridJob

A unit of work that can span multiple substrates.

```rust
/// A hybrid computation job that may use classical, quantum, or both
pub struct HybridJob {
    /// Unique job identifier
    pub id: JobId,

    /// Job specification
    pub spec: JobSpec,

    /// Current execution state
    pub state: JobState,

    /// Results from completed stages
    pub results: Vec<StageResult>,
}

pub enum JobSpec {
    /// Pure classical computation (Tile8 CPUs)
    Classical(ClassicalSpec),

    /// Pure quantum circuit execution
    Quantum(QuantumCircuit),

    /// Variational hybrid (VQE, QAOA)
    Variational {
        ansatz: QuantumCircuit,
        optimizer: ClassicalOptimizer,
        cost_function: CostFunction,
        max_iterations: usize,
    },

    /// Observer-controlled hybrid
    ObserverHybrid {
        quantum_state: QuantumSpec,
        observer_config: ObserverGridConfig,
        measurement_strategy: MeasurementStrategy,
    },

    /// Multi-stage pipeline
    Pipeline(Vec<JobSpec>),
}

pub enum JobState {
    Queued,
    Preparing,
    ExecutingClassical { progress: f64 },
    ExecutingQuantum { circuit_depth: usize, current_gate: usize },
    ExecutingObserver { tick: u64, collapsed_ratio: f64 },
    Aggregating,
    Completed,
    Failed(String),
}
```

### 2. SubstrateInterface

Unified interface for all compute substrates.

```rust
/// Common interface for all compute substrates
pub trait Substrate {
    /// Substrate identifier
    fn id(&self) -> SubstrateId;

    /// Available capacity (0.0 to 1.0)
    fn capacity(&self) -> f64;

    /// Execute a compatible job stage
    fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError>;

    /// Check if this substrate can handle a job type
    fn can_handle(&self, spec: &JobSpec) -> bool;

    /// Estimated execution time
    fn estimate_time(&self, spec: &JobSpec) -> Duration;
}

/// Classical substrate (Tile8 CPUs, SlimSim)
pub struct ClassicalSubstrate {
    pub cpus: Vec<Tile8Cpu>,
    pub sim: SlimSimulation,
    pub gpu_available: bool,
}

/// Quantum substrate (sparse simulation)
pub struct QuantumSubstrate {
    pub grid: SparseQuantumGridVec,
    pub max_qubits: usize,
    pub backend: QuantumBackend,
}

/// Observer substrate (hybrid decision layer)
pub struct ObserverSubstrate {
    pub grid: ObserverGrid,
    pub quantum_binding: Option<QuantumSubstrate>,
}
```

### 3. DataBridge

Cross-substrate data transfer.

```rust
/// Handles data transfer between substrates
pub struct DataBridge {
    /// Shared memory pool for zero-copy transfers
    memory_pool: SharedMemoryPool,

    /// Pending transfers
    transfers: VecDeque<Transfer>,
}

pub struct Transfer {
    pub source: SubstrateId,
    pub destination: SubstrateId,
    pub data: TransferData,
    pub priority: Priority,
}

pub enum TransferData {
    /// Classical state (CPU registers, memory)
    ClassicalState(Vec<u8>),

    /// Quantum amplitudes (for simulation handoff)
    QuantumAmplitudes(Vec<Complex64>),

    /// Measurement results
    MeasurementResults(Vec<u8>),

    /// Observer decisions
    ObserverDecisions(Vec<bool>),

    /// Probability distributions
    Probabilities(Vec<f64>),
}
```

### 4. HybridScheduler

Orchestrates job execution across substrates.

```rust
/// Main orchestrator for hybrid computation
pub struct HybridScheduler {
    /// Available substrates
    substrates: Vec<Box<dyn Substrate>>,

    /// Job queue
    queue: PriorityQueue<HybridJob>,

    /// Data bridge
    bridge: DataBridge,

    /// Execution history (for optimization)
    history: ExecutionHistory,
}

impl HybridScheduler {
    /// Submit a new job
    pub fn submit(&mut self, spec: JobSpec) -> JobId;

    /// Poll for job completion
    pub fn poll(&mut self, id: JobId) -> Option<JobResult>;

    /// Run scheduler loop (blocking)
    pub fn run(&mut self);

    /// Route job to optimal substrate
    fn route(&self, spec: &JobSpec) -> SubstrateId;

    /// Execute variational loop
    fn run_variational(&mut self, job: &mut HybridJob) -> Result<f64, SchedulerError>;
}
```

---

## Example: VQE with Hybrid Orchestration

```rust
use engine::hybrid::{HybridScheduler, JobSpec, ClassicalOptimizer};
use engine::quantum::QuantumCircuit;
use engine::hamiltonians::Hamiltonian;

fn run_vqe_hybrid() {
    // Create scheduler with all substrates
    let mut scheduler = HybridScheduler::new()
        .with_classical(ClassicalSubstrate::default())
        .with_quantum(QuantumSubstrate::new(30)) // 30 qubits
        .build();

    // Define VQE job
    let h2_hamiltonian = Hamiltonian::h2_molecule(0.735);
    let ansatz = QuantumCircuit::hardware_efficient(4, 2);

    let job = JobSpec::Variational {
        ansatz,
        optimizer: ClassicalOptimizer::COBYLA { maxiter: 100 },
        cost_function: CostFunction::ExpectationValue(h2_hamiltonian),
        max_iterations: 100,
    };

    // Submit and wait
    let job_id = scheduler.submit(job);

    // The scheduler handles:
    // 1. Parameter initialization (classical)
    // 2. Circuit execution (quantum substrate)
    // 3. Measurement and expectation value (quantum → classical)
    // 4. Gradient estimation (classical)
    // 5. Parameter update (classical)
    // 6. Repeat until convergence

    let result = scheduler.wait(job_id);
    println!("VQE converged to energy: {}", result.final_value);
}
```

---

## Example: Observer-Controlled Quantum Search

```rust
use engine::hybrid::{HybridScheduler, JobSpec, MeasurementStrategy};
use engine::observer_grid::{ObserverGridConfig, ObservationRule};

fn run_observer_search() {
    let mut scheduler = HybridScheduler::new()
        .with_quantum(QuantumSubstrate::new(20))
        .with_observer(ObserverSubstrate::new())
        .build();

    // Create quantum superposition
    let quantum_spec = QuantumSpec::UniformSuperposition { n_qubits: 20 };

    // Observer grid decides when to collapse
    let observer_config = ObserverGridConfig {
        num_observers: 1_000_000,
        default_rule: ObservationRule::ProbabilityThreshold(0.01),
        ..Default::default()
    };

    let job = JobSpec::ObserverHybrid {
        quantum_state: quantum_spec,
        observer_config,
        measurement_strategy: MeasurementStrategy::AnomalyTriggered {
            threshold: 0.1,
            max_ticks: 1000,
        },
    };

    let job_id = scheduler.submit(job);

    // The scheduler handles:
    // 1. Initialize quantum state (quantum substrate)
    // 2. Bind probabilities to observer grid (quantum → observer)
    // 3. Run observation dynamics (observer substrate)
    // 4. Detect anomalies (observer substrate)
    // 5. Collapse on anomaly detection (observer → quantum)
    // 6. Return measurement result

    let result = scheduler.wait(job_id);
    match result {
        JobResult::ObserverHybrid { measurement, anomaly_score, ticks } => {
            println!("Collapsed after {} ticks with anomaly score {}", ticks, anomaly_score);
            println!("Measurement: {:?}", measurement);
        }
        _ => unreachable!(),
    }
}
```

---

## Example: Shor's Algorithm (Hybrid Pipeline)

```rust
use engine::hybrid::{HybridScheduler, JobSpec};
use engine::algorithms::shor::{ShorSpec, ModularExponentiationMethod};

fn factor_number(n: u64) {
    let mut scheduler = HybridScheduler::new()
        .with_classical(ClassicalSubstrate::with_gpu())
        .with_quantum(QuantumSubstrate::new(40))
        .build();

    let job = JobSpec::Pipeline(vec![
        // Stage 1: Classical preprocessing (find suitable base)
        JobSpec::Classical(ClassicalSpec::ShorPreprocess { n }),

        // Stage 2: Quantum period finding
        JobSpec::Quantum(QuantumCircuit::shor_period_finding(n)),

        // Stage 3: Classical postprocessing (continued fractions)
        JobSpec::Classical(ClassicalSpec::ShorPostprocess),
    ]);

    let job_id = scheduler.submit(job);
    let result = scheduler.wait(job_id);

    match result {
        JobResult::Pipeline(stages) => {
            let factors = stages.last().unwrap().as_factors();
            println!("{} = {} × {}", n, factors.0, factors.1);
        }
        _ => unreachable!(),
    }
}
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Sprint 61)
- [ ] Define `Substrate` trait
- [ ] Implement `ClassicalSubstrate` wrapper
- [ ] Implement `QuantumSubstrate` wrapper
- [ ] Basic `DataBridge` with copy semantics

### Phase 2: Scheduler (Sprint 62)
- [ ] `HybridScheduler` with job queue
- [ ] Simple routing (match job type to substrate)
- [ ] Sequential pipeline execution

### Phase 3: Variational Support (Sprint 63)
- [ ] Variational job type
- [ ] Classical optimizer integration
- [ ] Quantum-classical data loop
- [ ] VQE and QAOA via scheduler

### Phase 4: Observer Integration (Sprint 64)
- [ ] `ObserverSubstrate` wrapper
- [ ] Quantum-observer binding
- [ ] Anomaly-triggered measurement
- [ ] Observer-hybrid job type

### Phase 5: Advanced Features (Sprint 65+)
- [ ] Parallel job execution
- [ ] Resource-aware scheduling
- [ ] Checkpoint/restart
- [ ] Distributed execution (multiple machines)

---

## Comparison to RIKEN

| RIKEN Component | Our Implementation | Notes |
|-----------------|-------------------|-------|
| QHScheduler | `HybridScheduler` | Simpler (single machine) |
| h3-Open-BDEC/QH | `DataBridge` | In-memory, not network |
| SQC Interface | `Substrate` trait | Same concept, Rust traits |
| Fugaku | `ClassicalSubstrate` | Our Tile8 CPUs |
| IBM/Quantinuum | `QuantumSubstrate` | Simulation, not hardware |
| - | `ObserverSubstrate` | **Our unique addition** |

The Observer Grid is something RIKEN doesn't have - a hybrid layer where classical decisions control quantum collapse dynamics. This could be our differentiator.

---

## Why This Matters

With orchestration, we can:

1. **Run VQE/QAOA properly** - Currently manual, would be automatic
2. **Benchmark hybrid algorithms** - Compare strategies systematically
3. **Test quantum-classical boundaries** - Where does quantum advantage appear?
4. **Develop observer-controlled algorithms** - New hybrid paradigm
5. **Match RIKEN's architecture** - Same patterns, open source

The orchestration layer turns disconnected pieces into a coherent quantum-classical computer.
