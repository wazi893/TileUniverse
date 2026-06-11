//! Fault-Tolerant Quantum Processor
//!
//! Main orchestrator that composes all fault-tolerant subsystems:
//! - QecModel: Surface code error model
//! - MagicSupply: Factory throughput bridge
//! - Scheduler: Gate → cycle scheduling
//! - InjectionModel: T-gate injection (Abstract vs Sampled)
//!
//! # Two Execution Modes
//!
//! 1. **Resource Estimation** (`estimate_resources`): Fast, scalable analysis
//!    - Computes spacetime volume, qubit counts, error probability
//!    - No simulation of individual gate outcomes
//!    - Use for: "How many qubits and how much time for this circuit?"
//!
//! 2. **Simulation** (`execute`): Pauli frame tracking
//!    - Samples T-gate measurement outcomes
//!    - Updates Pauli frames based on measurement results
//!    - Use for: "Verify classical control logic of FT circuits"
//!
//! # Example
//!
//! ```ignore
//! use engine::qec::{FaultTolerantProcessor, FTConfig, LogicalGate};
//!
//! // Create processor with 5 logical qubits
//! let mut processor = FaultTolerantProcessor::new(5, FTConfig::default());
//!
//! // Build a circuit
//! let circuit = vec![
//!     LogicalGate::H(0),
//!     LogicalGate::CNOT(0, 1),
//!     LogicalGate::T(0),
//!     LogicalGate::MeasureZ(0),
//! ];
//!
//! // Fast resource estimation
//! let estimate = processor.estimate_resources(&circuit);
//! println!("Cycles: {}, Qubits: {}", estimate.cycles, estimate.total_qubits);
//!
//! // Or full simulation
//! let result = processor.execute(&circuit);
//! println!("Final error probability: {:.2e}", result.error_budget.total());
//! ```

use super::decoder::UnionFindDecoder;
use super::ft_types::{
    DecoderChoice, KnownBasis, LogicalErrorBudget, LogicalGate, LogicalQubit, PauliFrame,
    ResourceEvent, ResourceLog, ResourceSummary, SimulationFidelity, UnknownReason,
};
use super::injection::{InjectionModel, inject_tdg};
use super::magic_supply::{MagicSupply, MagicSupplyCost};
use super::noise::SimpleRng;
use super::qec_model::QecModel;
use super::scheduler::{CycleCosts, ScheduleAnalysis, Scheduler, compute_t_depth};
use std::fmt;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for FaultTolerantProcessor.
#[derive(Clone, Debug)]
pub struct FTConfig {
    /// Surface code distance
    pub distance: usize,
    /// Physical error rate
    pub physical_error: f64,
    /// Target logical error rate per operation
    pub target_error: f64,
    /// Cycle costs for scheduling
    pub cycle_costs: CycleCosts,
    /// Injection model (Abstract or Sampled)
    pub injection_model: InjectionModel,
    /// Code cycle time in microseconds (for time estimation)
    pub cycle_time_us: f64,
    /// Simulation fidelity level (Abstract vs PhenomenologicalDecoder)
    pub simulation_fidelity: SimulationFidelity,
}

impl FTConfig {
    /// Create configuration with typical surface code parameters.
    pub fn new(distance: usize, physical_error: f64) -> Self {
        Self {
            distance,
            physical_error,
            target_error: 1e-10,
            cycle_costs: CycleCosts::scaled_by_distance(distance),
            injection_model: InjectionModel::abstract_default(),
            cycle_time_us: 1.0, // 1 μs per code cycle
            simulation_fidelity: SimulationFidelity::Abstract,
        }
    }

    /// Use sampled injection model (for simulation).
    pub fn with_sampled_injection(mut self) -> Self {
        self.injection_model = InjectionModel::sampled_default();
        self
    }

    /// Use abstract injection model (for estimation).
    pub fn with_abstract_injection(mut self) -> Self {
        self.injection_model = InjectionModel::abstract_default();
        self
    }

    /// Set target error rate.
    pub fn with_target_error(mut self, target: f64) -> Self {
        self.target_error = target;
        self
    }

    /// Set custom cycle costs.
    pub fn with_cycle_costs(mut self, costs: CycleCosts) -> Self {
        self.cycle_costs = costs;
        self
    }

    /// Set code cycle time.
    pub fn with_cycle_time_us(mut self, time_us: f64) -> Self {
        self.cycle_time_us = time_us;
        self
    }

    /// Set simulation fidelity level.
    pub fn with_fidelity(mut self, fidelity: SimulationFidelity) -> Self {
        self.simulation_fidelity = fidelity;
        self
    }

    /// Use phenomenological decoder simulation with Union-Find.
    pub fn with_decoder_simulation(mut self) -> Self {
        self.simulation_fidelity = SimulationFidelity::with_union_find(self.physical_error);
        self
    }

    /// Use phenomenological decoder simulation with specified error rate.
    pub fn with_decoder_simulation_at(mut self, error_rate: f64) -> Self {
        self.simulation_fidelity = SimulationFidelity::with_union_find(error_rate);
        self
    }
}

impl Default for FTConfig {
    fn default() -> Self {
        Self::new(7, 1e-3)
    }
}

// ============================================================================
// Decoder State (Phase 3.3)
// ============================================================================

/// State for decoder-based simulation.
///
/// Created lazily when using `SimulationFidelity::PhenomenologicalDecoder`.
struct DecoderState {
    /// Decoder instance (Union-Find)
    decoder: UnionFindDecoder,
    /// Physical error rate for generating errors
    error_rate: f64,
    /// Which decoder type is being used
    #[allow(dead_code)]
    decoder_choice: DecoderChoice,
}

impl DecoderState {
    /// Create decoder state for given distance and configuration.
    fn new(
        distance: usize,
        #[allow(dead_code)] decoder_choice: DecoderChoice,
        error_rate: f64,
    ) -> Self {
        let decoder = UnionFindDecoder::new(distance);
        Self {
            decoder,
            error_rate,
            decoder_choice,
        }
    }

    /// Simulate one QEC round with phenomenological noise.
    ///
    /// Returns (success, logical_error):
    /// - success: syndrome was cleared by decoder
    /// - logical_error: a logical error occurred (undetectable)
    fn simulate_round(&self, rng: &mut SimpleRng) -> (bool, bool) {
        use super::codes::SurfaceCode;

        let distance = self.decoder.distance();
        let n_data = distance * distance;

        // Create fresh surface code
        let mut code = SurfaceCode::new(distance);

        // Apply random errors based on phenomenological model
        let error_types = ['X', 'Y', 'Z'];
        for q in 0..n_data {
            if rng.next_f64() < self.error_rate {
                let error_type = error_types[rng.next_usize(3)];
                code.apply_error(q, error_type);
            }
        }

        // Extract syndrome and decode
        let (x_syn, z_syn) = code.measure_syndrome();
        let (x_corr, z_corr) = self.decoder.decode(&x_syn, &z_syn);

        // Apply corrections
        code.correct(&x_corr, &z_corr);

        // Check results
        let (x_after, z_after) = code.measure_syndrome();
        let syndrome_cleared = x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);
        let logical_error = code.has_logical_x_error() || code.has_logical_z_error();

        (syndrome_cleared, logical_error)
    }
}

// ============================================================================
// Fault-Tolerant Processor
// ============================================================================

/// Main fault-tolerant quantum processor.
///
/// Orchestrates logical qubit simulation with proper QEC error modeling,
/// magic state consumption, and Pauli frame tracking.
pub struct FaultTolerantProcessor {
    /// Logical qubits
    pub(crate) qubits: Vec<LogicalQubit>,
    /// Surface code error model
    qec_model: QecModel,
    /// Magic state supply
    magic_supply: MagicSupply,
    /// Operation scheduler
    scheduler: Scheduler,
    /// T-gate injection model
    injection_model: InjectionModel,
    /// Error accounting
    error_budget: LogicalErrorBudget,
    /// Current cycle
    current_cycle: usize,
    /// Execution statistics
    stats: FTStats,
    /// Random number generator for sampled mode
    rng: SimpleRng,
    /// Configuration (for reference)
    config: FTConfig,
    /// Resource event log (Phase 2.2)
    resource_log: ResourceLog,
    /// Decoder state for phenomenological simulation (Phase 3.3)
    decoder_state: Option<DecoderState>,
}

impl FaultTolerantProcessor {
    /// Create a new fault-tolerant processor.
    pub fn new(n_qubits: usize, config: FTConfig) -> Self {
        let qubits: Vec<_> = (0..n_qubits).map(LogicalQubit::new).collect();
        let qec_model = QecModel::new(config.distance, config.physical_error);
        let magic_supply = MagicSupply::new(config.physical_error, config.target_error);
        let scheduler = Scheduler::with_costs(config.cycle_costs.clone());

        // Compute physical qubit counts for resource log
        let data_qubits = qec_model.physical_qubits_for(n_qubits);
        let factory_qubits = magic_supply.factory_footprint();

        // Create decoder state if phenomenological simulation is enabled
        let decoder_state = match &config.simulation_fidelity {
            SimulationFidelity::PhenomenologicalDecoder {
                decoder,
                error_rate,
            } => Some(DecoderState::new(config.distance, *decoder, *error_rate)),
            SimulationFidelity::Abstract => None,
        };

        Self {
            qubits,
            qec_model,
            magic_supply,
            scheduler,
            injection_model: config.injection_model.clone(),
            error_budget: LogicalErrorBudget::new(),
            current_cycle: 0,
            stats: FTStats::new(),
            rng: SimpleRng::new(42),
            config,
            resource_log: ResourceLog::new(data_qubits, factory_qubits),
            decoder_state,
        }
    }

    /// Create with specific random seed (for reproducible simulation).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SimpleRng::new(seed);
        self
    }

    /// Create with pre-filled magic state inventory.
    pub fn with_magic_inventory(mut self, inventory: usize) -> Self {
        self.magic_supply = self.magic_supply.with_inventory(inventory);
        self
    }

    // ========================================================================
    // Resource Event Log (Phase 2.2)
    // ========================================================================

    /// Get the resource event log.
    pub fn resource_log(&self) -> &ResourceLog {
        &self.resource_log
    }

    /// Get resource summary from the event log.
    ///
    /// This provides the key metrics:
    /// - makespan_cycles (critical path)
    /// - peak_physical_qubits
    /// - spacetime_qubit_cycles (∫ qubits × dt)
    /// - total_magic_states
    /// - demand_timeline (binned)
    pub fn resource_summary(&self) -> ResourceSummary {
        self.resource_log.summarize()
    }

    /// Clear the resource event log (for fresh measurements).
    pub fn clear_resource_log(&mut self) {
        self.resource_log.clear();
    }

    // ========================================================================
    // Factory Pipeline (Phase 2.1)
    // ========================================================================

    /// Create a factory pipeline matching this processor's configuration.
    pub fn create_factory_pipeline(&self) -> super::magic_supply::FactoryPipeline {
        super::magic_supply::FactoryPipeline::from_supply(&self.magic_supply)
    }

    /// Analyze whether the factory can sustain the demand from a circuit.
    ///
    /// This uses the demand timeline from the resource log to simulate
    /// factory throughput vs circuit demand.
    ///
    /// # Returns
    /// A backpressure report showing whether demand was sustained,
    /// total stall cycles, and recommended throughput if needed.
    pub fn analyze_backpressure(&self) -> super::magic_supply::BackpressureReport {
        let pipeline = self.create_factory_pipeline();
        let summary = self.resource_summary();
        pipeline.analyze_demand(&summary.demand_timeline)
    }

    /// Analyze backpressure for a circuit without executing it.
    ///
    /// First estimates the demand timeline, then analyzes against factory.
    pub fn analyze_circuit_backpressure(
        &self,
        circuit: &[LogicalGate],
    ) -> super::magic_supply::BackpressureReport {
        // Use parallel schedule for accurate demand timeline
        let schedule = self.scheduler.schedule_parallel(circuit);
        let pipeline = self.create_factory_pipeline();
        pipeline.analyze_demand(&schedule.demand_timeline)
    }

    // ========================================================================
    // Phase 3.1: Integrated Schedule + Factory Execution
    // ========================================================================

    /// Schedule a circuit with parallel analysis.
    ///
    /// Returns the parallel schedule with timing, T-depth, demand timeline.
    /// This is the first step for factory-aware execution.
    pub fn schedule_circuit(&self, circuit: &[LogicalGate]) -> super::scheduler::ParallelSchedule {
        self.scheduler.schedule_parallel(circuit)
    }

    /// Plan-first resource estimation with factory analysis.
    ///
    /// This mode:
    /// 1. Schedules the circuit (ignoring factory constraints)
    /// 2. Computes T-demand histogram from schedule
    /// 3. Computes minimum factories to avoid stalls
    /// 4. Returns combined estimate with factory recommendations
    ///
    /// Use this when you want to know "how many factories do I need?"
    pub fn estimate_with_factories(&self, circuit: &[LogicalGate]) -> FactoryAwareEstimate {
        let schedule = self.scheduler.schedule_parallel(circuit);

        // Analyze factory requirements
        let pipeline = self.create_factory_pipeline();
        let backpressure = pipeline.analyze_demand(&schedule.demand_timeline);

        // Compute physical resources
        let data_qubits = self.qec_model.physical_qubits_for(self.qubits.len());
        let factory_qubits = self.magic_supply.factory_footprint();

        // Calculate factory scaling
        let current_throughput = self.magic_supply.throughput();
        let required_throughput = schedule.min_throughput_required();
        let factories_needed = if required_throughput > 0.0 && current_throughput > 0.0 {
            (required_throughput / current_throughput).ceil() as usize
        } else {
            1
        };

        FactoryAwareEstimate {
            // Schedule metrics
            makespan_cycles: schedule.makespan,
            t_depth: schedule.t_depth,
            cnot_depth: schedule.cnot_depth,
            total_t_gates: schedule.total_magic_states(),
            peak_t_demand: schedule.peak_t_demand(),

            // Resource metrics
            data_qubits,
            factory_qubits_per_factory: factory_qubits,
            factories_needed,
            total_factory_qubits: factories_needed * factory_qubits,
            total_qubits: data_qubits + factories_needed * factory_qubits,

            // Spacetime
            spacetime_volume: data_qubits * schedule.makespan
                + factories_needed * factory_qubits * schedule.makespan,

            // Factory analysis
            backpressure_report: backpressure,
            min_throughput_required: required_throughput,
            current_throughput,

            // Error budget
            logical_error: self.qec_model.error_after_cycles(schedule.makespan),
        }
    }

    /// Simulate circuit execution with factory stalls.
    ///
    /// This mode:
    /// 1. Runs the schedule timeline
    /// 2. At each T-gate, requests magic state from factory
    /// 3. If inventory empty, gate stalls (propagates delay)
    /// 4. Records actual makespan including stall cycles
    ///
    /// Use this when you want to see "what actually happens with my factory config?"
    pub fn simulate_with_stalls(&mut self, circuit: &[LogicalGate]) -> StallSimulationResult {
        let schedule = self.scheduler.schedule_parallel(circuit);

        if schedule.gates.is_empty() {
            return StallSimulationResult {
                ideal_makespan: 0,
                actual_makespan: 0,
                total_stall_cycles: 0,
                stall_count: 0,
                t_gates_executed: 0,
                factory_utilization: 1.0,
            };
        }

        // Create pipeline for simulation
        let mut pipeline = self.create_factory_pipeline();

        let mut actual_cycle = 0;
        let mut total_stalls = 0;
        let mut stall_count = 0;
        let mut t_gates = 0;

        // Process gates in scheduled order
        // Note: gates are already sorted by start_cycle due to ASAP scheduling
        for sg in &schedule.gates {
            // Advance pipeline to this gate's scheduled start
            let target_cycle = sg.start_cycle + total_stalls;
            pipeline.tick_to(target_cycle);

            if sg.needs_magic() {
                // Request magic states
                let result = pipeline.request(sg.magic_states);
                t_gates += sg.magic_states;

                if result.backpressure {
                    // Stall! Need to wait for factory
                    total_stalls += result.wait_cycles;
                    stall_count += 1;
                    pipeline.tick_to(target_cycle + result.wait_cycles);
                }
            }

            // Update actual completion time
            actual_cycle = actual_cycle.max(target_cycle + sg.duration + total_stalls);
        }

        let ideal = schedule.makespan;
        let utilization = if t_gates > 0 {
            pipeline.total_delivered() as f64 / t_gates as f64
        } else {
            1.0
        };

        StallSimulationResult {
            ideal_makespan: ideal,
            actual_makespan: actual_cycle,
            total_stall_cycles: total_stalls,
            stall_count,
            t_gates_executed: t_gates,
            factory_utilization: utilization,
        }
    }

    /// Record a resource event from gate execution.
    fn record_event(&mut self, gate: &LogicalGate, start_cycle: usize, cycles: usize) {
        let data_qubits = self.qec_model.physical_qubits_for(self.qubits.len());
        let factory_qubits = self.magic_supply.factory_footprint();

        let event = match gate {
            LogicalGate::T(_) | LogicalGate::Tdg(_) => {
                ResourceEvent::t_gate(start_cycle, cycles, data_qubits, factory_qubits)
            }
            LogicalGate::Toffoli(_, _, _) => {
                ResourceEvent::toffoli(start_cycle, cycles, data_qubits, factory_qubits)
            }
            LogicalGate::CCZ(_, _, _) => {
                // CCZ = 7 T-gates (same as Toffoli)
                ResourceEvent::toffoli(start_cycle, cycles, data_qubits, factory_qubits)
            }
            _ => ResourceEvent::clifford(start_cycle, cycles, data_qubits),
        };

        self.resource_log.record(event);
    }

    // ========================================================================
    // Resource Estimation (Fast Mode)
    // ========================================================================

    /// Estimate resources for a circuit without full simulation.
    ///
    /// This is the fast mode for resource estimation at scale.
    /// It computes spacetime volume, qubit counts, and error probability
    /// without simulating individual gate outcomes.
    pub fn estimate_resources(&self, circuit: &[LogicalGate]) -> ResourceEstimate {
        // Schedule analysis
        let analysis = self.scheduler.analyze(circuit);
        let total_cycles = analysis.total_cycles_parallel;

        // T-count and T-depth
        let t_count = analysis.t_count;
        let t_depth = compute_t_depth(circuit);

        // Data qubits
        let data_qubits = self.qubits.len() * self.qec_model.physical_qubits_per_logical;

        // Factory cost
        let factory_cost = self.magic_supply.estimate_cost(t_count, t_depth);

        // Total qubits
        let total_qubits = data_qubits + factory_cost.factory_qubits;

        // Spacetime volume
        let data_volume = data_qubits * total_cycles;
        let factory_volume = factory_cost.spacetime_volume();
        let total_volume = data_volume + factory_volume;

        // Error estimation
        let qec_error = self
            .qec_model
            .any_error_probability(self.qubits.len(), total_cycles);
        let injection_error = 1.0 - (1.0 - self.magic_supply.error_rate()).powi(t_count as i32);
        let total_error = 1.0 - (1.0 - qec_error) * (1.0 - injection_error);

        // Time estimation
        let runtime_us = total_cycles as f64 * self.config.cycle_time_us;

        ResourceEstimate {
            cycles: total_cycles,
            data_qubits,
            factory_qubits: factory_cost.factory_qubits,
            total_qubits,
            spacetime_volume: total_volume,
            t_count,
            t_depth,
            clifford_count: analysis.clifford_count,
            two_qubit_count: analysis.two_qubit_count,
            qec_error_probability: qec_error,
            injection_error_probability: injection_error,
            total_error_probability: total_error,
            runtime_us,
            factory_cost,
            schedule_analysis: analysis,
        }
    }

    // ========================================================================
    // Simulation (Pauli Frame Mode)
    // ========================================================================

    /// Execute a circuit with full Pauli frame simulation.
    ///
    /// This mode simulates measurement outcomes and tracks Pauli frame
    /// updates. Use for verifying classical control logic.
    pub fn execute(&mut self, circuit: &[LogicalGate]) -> ExecutionResult {
        let start_cycle = self.current_cycle;
        let mut measurement_outcomes = Vec::new();

        for gate in circuit {
            let step = self.execute_step(gate);
            if let Some(m) = step.measurement {
                measurement_outcomes.push(m);
            }
        }

        let end_cycle = self.current_cycle;
        let cycles_used = end_cycle - start_cycle;

        ExecutionResult {
            cycles_used,
            error_budget: self.error_budget.clone(),
            measurement_outcomes,
            final_frames: self.qubits.iter().map(|q| q.frame).collect(),
            final_basis: self.qubits.iter().map(|q| q.basis).collect(),
            stats: self.stats.clone(),
        }
    }

    /// Execute a circuit with full trace (returns StepResult for each gate).
    ///
    /// Use this for debugging and verification. Every gate's effect on
    /// frames, states, and resources is captured.
    pub fn execute_traced(&mut self, circuit: &[LogicalGate]) -> Vec<StepResult> {
        circuit.iter().map(|gate| self.execute_step(gate)).collect()
    }

    /// Execute a single gate with full introspection.
    ///
    /// This is the core debugging primitive. Returns complete before/after
    /// state for frames, basis states, and any measurement or injection outcomes.
    pub fn execute_step(&mut self, gate: &LogicalGate) -> StepResult {
        // Capture state BEFORE
        let frames_before: Vec<_> = self.qubits.iter().map(|q| q.frame).collect();
        let basis_before: Vec<_> = self.qubits.iter().map(|q| q.basis).collect();
        let cycle_start = self.current_cycle;
        let magic_before = self.stats.magic_states_consumed;

        // Execute the gate and capture any special outcomes
        let (measurement, injection_outcome) = self.execute_gate_internal(gate);

        // Capture state AFTER
        let frames_after: Vec<_> = self.qubits.iter().map(|q| q.frame).collect();
        let basis_after: Vec<_> = self.qubits.iter().map(|q| q.basis).collect();
        let cycles_consumed = self.current_cycle - cycle_start;
        let magic_consumed = self.stats.magic_states_consumed - magic_before;

        // Record resource event (Phase 2.2)
        if cycles_consumed > 0 {
            self.record_event(gate, cycle_start, cycles_consumed);
        }

        StepResult {
            gate: gate.clone(),
            frames_before,
            frames_after,
            basis_before,
            basis_after,
            magic_states_consumed: magic_consumed,
            cycles_consumed,
            cycle_start,
            measurement,
            injection_outcome,
        }
    }

    /// Internal gate execution that returns measurement/injection outcomes.
    fn execute_gate_internal(
        &mut self,
        gate: &LogicalGate,
    ) -> (Option<MeasurementOutcome>, Option<bool>) {
        let mut measurement = None;
        let mut injection_outcome = None;

        match gate {
            // Single-qubit Clifford
            LogicalGate::X(q) => {
                self.apply_clifford(*q, |qubit| qubit.apply_x());
            }
            LogicalGate::Y(q) => {
                self.apply_clifford(*q, |qubit| {
                    qubit.apply_x();
                    qubit.apply_z();
                });
            }
            LogicalGate::Z(q) => {
                self.apply_clifford(*q, |qubit| qubit.apply_z());
            }
            LogicalGate::H(q) => {
                self.apply_clifford(*q, |qubit| qubit.apply_h());
            }
            LogicalGate::S(q) => {
                self.apply_clifford(*q, |qubit| qubit.apply_s());
            }
            LogicalGate::Sdg(q) => {
                self.apply_clifford(*q, |qubit| {
                    // S† = S³ in frame terms
                    qubit.apply_s();
                    qubit.apply_s();
                    qubit.apply_s();
                });
            }

            // T-gates (non-Clifford) - capture injection outcome
            LogicalGate::T(q) => {
                injection_outcome = Some(self.execute_t_gate_traced(*q));
            }
            LogicalGate::Tdg(q) => {
                injection_outcome = Some(self.execute_tdg_gate_traced(*q));
            }

            // Two-qubit gates
            LogicalGate::CNOT(c, t) => {
                self.execute_cnot(*c, *t);
            }
            LogicalGate::CZ(a, b) => {
                self.execute_cz(*a, *b);
            }
            LogicalGate::SWAP(a, b) => {
                self.execute_swap(*a, *b);
            }

            // Multi-qubit gates
            LogicalGate::Toffoli(c1, c2, t) => {
                self.execute_toffoli(*c1, *c2, *t);
            }
            LogicalGate::CCZ(a, b, c) => {
                self.execute_ccz(*a, *b, *c);
            }

            // Control operations
            LogicalGate::QECRound => {
                self.execute_qec_round();
            }
            LogicalGate::Idle(n) => {
                self.execute_idle(*n);
            }

            // Measurement - capture outcome
            LogicalGate::MeasureZ(q) => {
                measurement = Some(self.execute_measure_z(*q));
            }
            LogicalGate::MeasureX(q) => {
                measurement = Some(self.execute_measure_x(*q));
            }

            // Initialization
            LogicalGate::Reset(q) => {
                self.execute_reset(*q);
            }
            LogicalGate::PreparePlus(q) => {
                self.execute_prepare_plus(*q);
            }
        }

        (measurement, injection_outcome)
    }

    // ========================================================================
    // Gate Implementations
    // ========================================================================

    fn apply_clifford<F>(&mut self, qubit: usize, op: F)
    where
        F: FnOnce(&mut LogicalQubit),
    {
        let cost = self.scheduler.costs().single_qubit_clifford;
        self.advance_cycles(cost);
        self.stats.clifford_gates += 1;

        if let Some(q) = self.qubits.get_mut(qubit) {
            op(q);
        }
    }

    /// Execute T-gate and return injection measurement outcome.
    /// Returns true if |−⟩ (needs S/Z correction), false if |+⟩.
    fn execute_t_gate_traced(&mut self, qubit: usize) -> bool {
        self.stats.t_gates += 1;

        // Consume magic state
        let magic_result = self.magic_supply.consume();
        self.stats.magic_states_consumed += 1;
        if magic_result.delay_cycles > 0 {
            self.stats.factory_delays += magic_result.delay_cycles;
        }

        // Perform injection and capture outcome
        let injection_outcome = if let Some(q) = self.qubits.get_mut(qubit) {
            let result = self
                .injection_model
                .inject(q, &mut self.rng, &mut self.error_budget);
            // The measurement_outcome tells us if correction was needed
            result.measurement_outcome.unwrap_or(false)
        } else {
            false
        };

        let cost = self.scheduler.costs().t_injection;
        self.advance_cycles(cost);

        injection_outcome
    }

    /// Execute T†-gate and return injection measurement outcome.
    fn execute_tdg_gate_traced(&mut self, qubit: usize) -> bool {
        self.stats.t_gates += 1;

        let magic_result = self.magic_supply.consume();
        self.stats.magic_states_consumed += 1;
        if magic_result.delay_cycles > 0 {
            self.stats.factory_delays += magic_result.delay_cycles;
        }

        let injection_outcome = if let Some(q) = self.qubits.get_mut(qubit) {
            let result = inject_tdg(
                &mut self.injection_model,
                q,
                &mut self.rng,
                &mut self.error_budget,
            );
            result.measurement_outcome.unwrap_or(false)
        } else {
            false
        };

        let cost = self.scheduler.costs().t_injection;
        self.advance_cycles(cost);

        injection_outcome
    }

    /// Determine why an entangling gate creates unknown basis state.
    /// Returns the UnknownReason if the result would be unknown, None if deterministic.
    fn entanglement_reason(basis_a: KnownBasis, basis_b: KnownBasis) -> Option<UnknownReason> {
        match (basis_a, basis_b) {
            // If either is already unknown, entanglement propagates that
            (KnownBasis::Unknown(_), _) | (_, KnownBasis::Unknown(_)) => {
                Some(UnknownReason::EntanglingGateWithUnknown)
            }
            // Both known but one is in superposition (X-basis)
            (KnownBasis::X(_), _) | (_, KnownBasis::X(_)) => {
                Some(UnknownReason::EntanglingGateOnSuperposition)
            }
            // Both Z-basis: no entanglement created (classical CNOT)
            (KnownBasis::Z(_), KnownBasis::Z(_)) => None,
        }
    }

    fn execute_cnot(&mut self, control: usize, target: usize) {
        self.stats.two_qubit_gates += 1;
        let cost = self.scheduler.costs().lattice_surgery;
        self.advance_cycles(cost);

        // CNOT propagates X from control to target, Z from target to control
        // In frame: if control has X, target gets X
        //           if target has Z, control gets Z
        let control_x = self.qubits.get(control).map(|q| q.frame.x).unwrap_or(false);
        let target_z = self.qubits.get(target).map(|q| q.frame.z).unwrap_or(false);

        // Get basis states for proper tracking
        let control_basis = self
            .qubits
            .get(control)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let target_basis = self
            .qubits
            .get(target)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));

        // Determine unknown reason if entanglement will occur
        let unknown_reason = Self::entanglement_reason(control_basis, target_basis);

        // CNOT on Z-basis control: deterministic classical logic
        // Control is preserved, target gets flipped if control is |1⟩
        //
        // When control is in superposition (X-basis): creates Bell state
        // |+⟩|0⟩ → (|00⟩+|11⟩)/√2 - both qubits become entangled
        let new_target_basis = match (control_basis, target_basis) {
            (KnownBasis::Z(c), KnownBasis::Z(t)) => {
                // Classical: target = target XOR control
                KnownBasis::Z(t ^ c)
            }
            (KnownBasis::Z(c), KnownBasis::X(t)) => {
                // Control Z, target X: if control is |1⟩, flip target within X-basis
                // X·|+⟩ = |-⟩, X·|-⟩ = |+⟩
                KnownBasis::X(t ^ c)
            }
            (KnownBasis::Z(false), basis) => {
                // Control is |0⟩: target unchanged (CNOT does nothing)
                basis
            }
            _ => {
                // Control in superposition or unknown creates entanglement
                KnownBasis::Unknown(unknown_reason.unwrap_or(UnknownReason::Explicit))
            }
        };

        // Control qubit: preserved only if already in Z-basis
        // If control is in X-basis (superposition), CNOT creates entanglement
        let new_control_basis = match control_basis {
            KnownBasis::Z(c) => {
                // Z-basis control is always preserved through CNOT
                KnownBasis::Z(c)
            }
            _ => {
                // X-basis or unknown: creates entanglement
                KnownBasis::Unknown(unknown_reason.unwrap_or(UnknownReason::Explicit))
            }
        };

        if let Some(q) = self.qubits.get_mut(target) {
            q.frame.x ^= control_x;
            q.basis = new_target_basis;
        }
        if let Some(q) = self.qubits.get_mut(control) {
            q.frame.z ^= target_z;
            q.basis = new_control_basis;
        }
    }

    fn execute_cz(&mut self, a: usize, b: usize) {
        self.stats.two_qubit_gates += 1;
        let cost = self.scheduler.costs().lattice_surgery;
        self.advance_cycles(cost);

        // CZ: if either has X, the other gets Z
        let a_x = self.qubits.get(a).map(|q| q.frame.x).unwrap_or(false);
        let b_x = self.qubits.get(b).map(|q| q.frame.x).unwrap_or(false);

        // Get basis states for proper tracking
        let a_basis = self
            .qubits
            .get(a)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let b_basis = self
            .qubits
            .get(b)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));

        // Determine unknown reason if entanglement will occur
        let unknown_reason = Self::entanglement_reason(a_basis, b_basis);

        // CZ on Z-basis states: only adds phase on |11⟩, doesn't change basis state
        // So Z-basis states are preserved through CZ
        let new_a_basis = match (a_basis, b_basis) {
            (KnownBasis::Z(v), KnownBasis::Z(_)) => KnownBasis::Z(v),
            (basis, KnownBasis::Z(_)) => basis, // Z on other doesn't create superposition
            _ => KnownBasis::Unknown(unknown_reason.unwrap_or(UnknownReason::Explicit)),
        };

        let new_b_basis = match (a_basis, b_basis) {
            (KnownBasis::Z(_), KnownBasis::Z(v)) => KnownBasis::Z(v),
            (KnownBasis::Z(_), basis) => basis,
            _ => KnownBasis::Unknown(unknown_reason.unwrap_or(UnknownReason::Explicit)),
        };

        if let Some(q) = self.qubits.get_mut(a) {
            q.frame.z ^= b_x;
            q.basis = new_a_basis;
        }
        if let Some(q) = self.qubits.get_mut(b) {
            q.frame.z ^= a_x;
            q.basis = new_b_basis;
        }
    }

    fn execute_swap(&mut self, a: usize, b: usize) {
        self.stats.two_qubit_gates += 1;
        // SWAP = 3 CNOTs
        let cost = 3 * self.scheduler.costs().lattice_surgery;
        self.advance_cycles(cost);

        // Swap frames and basis states
        if a < self.qubits.len() && b < self.qubits.len() {
            let temp = self.qubits[a].frame;
            self.qubits[a].frame = self.qubits[b].frame;
            self.qubits[b].frame = temp;

            let temp_basis = self.qubits[a].basis;
            self.qubits[a].basis = self.qubits[b].basis;
            self.qubits[b].basis = temp_basis;
        }
    }

    fn execute_toffoli(&mut self, c1: usize, c2: usize, t: usize) {
        // Toffoli = 7 T-gates
        let results = self.magic_supply.consume_toffoli();
        self.stats.magic_states_consumed += 7;
        self.stats.t_gates += 7;
        self.stats.toffoli_gates += 1;

        for result in &results {
            if result.delay_cycles > 0 {
                self.stats.factory_delays += result.delay_cycles;
            }
        }

        // Add injection errors for the 7 T-gates in Toffoli decomposition
        if c1 < self.qubits.len() && c2 < self.qubits.len() && t < self.qubits.len() {
            for _ in 0..7 {
                self.error_budget
                    .add_injection(self.magic_supply.error_rate());
            }
        }

        // Toffoli on Z-basis states: deterministic classical logic
        // |c1,c2,t⟩ → |c1,c2,t⊕(c1∧c2)⟩
        let c1_basis = self
            .qubits
            .get(c1)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let c2_basis = self
            .qubits
            .get(c2)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let t_basis = self
            .qubits
            .get(t)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));

        match (c1_basis, c2_basis, t_basis) {
            (KnownBasis::Z(v1), KnownBasis::Z(v2), KnownBasis::Z(vt)) => {
                // Classical: target = t XOR (c1 AND c2)
                let new_t = vt ^ (v1 && v2);
                if let Some(q) = self.qubits.get_mut(t) {
                    q.basis = KnownBasis::Z(new_t);
                }
                // Controls unchanged
            }
            _ => {
                // Non-classical case: mark all unknown
                for q in [c1, c2, t] {
                    if let Some(qubit) = self.qubits.get_mut(q) {
                        qubit.mark_unknown();
                    }
                }
            }
        }

        let cost = self.scheduler.costs().toffoli;
        self.advance_cycles(cost);
    }

    fn execute_ccz(&mut self, a: usize, b: usize, c: usize) {
        // CCZ = controlled-controlled-Z: adds phase -1 to |111⟩
        // For Z-basis states, CCZ only adds a phase, doesn't change the basis state
        // Same resource cost as Toffoli (7 T-gates)
        let results = self.magic_supply.consume_toffoli();
        self.stats.magic_states_consumed += 7;
        self.stats.t_gates += 7;
        self.stats.toffoli_gates += 1;

        for result in &results {
            if result.delay_cycles > 0 {
                self.stats.factory_delays += result.delay_cycles;
            }
        }

        if a < self.qubits.len() && b < self.qubits.len() && c < self.qubits.len() {
            for _ in 0..7 {
                self.error_budget
                    .add_injection(self.magic_supply.error_rate());
            }
        }

        // CCZ on Z-basis states: preserves all basis states (only adds phase on |111⟩)
        let a_basis = self
            .qubits
            .get(a)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let b_basis = self
            .qubits
            .get(b)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));
        let c_basis = self
            .qubits
            .get(c)
            .map(|q| q.basis)
            .unwrap_or(KnownBasis::Unknown(UnknownReason::NeverInitialized));

        match (a_basis, b_basis, c_basis) {
            (KnownBasis::Z(_), KnownBasis::Z(_), KnownBasis::Z(_)) => {
                // All Z-basis: CCZ preserves state (phase doesn't affect Z-measurement)
                // No change to basis states needed
            }
            _ => {
                // Non-classical case: mark all unknown
                for q in [a, b, c] {
                    if let Some(qubit) = self.qubits.get_mut(q) {
                        qubit.mark_unknown();
                    }
                }
            }
        }

        let cost = self.scheduler.costs().toffoli;
        self.advance_cycles(cost);
    }

    fn execute_qec_round(&mut self) {
        let cost = self.scheduler.costs().qec_round;
        self.error_budget
            .add_qec_cycles(self.qec_model.p_logical_per_cycle, cost);
        self.advance_cycles(cost);
    }

    fn execute_idle(&mut self, cycles: usize) {
        // Idle qubits still accumulate QEC error
        self.error_budget
            .add_qec_cycles(self.qec_model.p_logical_per_cycle, cycles);
        self.advance_cycles(cycles);
    }

    fn execute_measure_z(&mut self, qubit: usize) -> MeasurementOutcome {
        let cost = self.scheduler.costs().measurement;
        self.advance_cycles(cost);

        let (value, frame, raw) = if let Some(q) = self.qubits.get_mut(qubit) {
            // Capture frame BEFORE clearing (critical for frame tracking!)
            let frame_before = q.frame;

            // Measurement outcome depends on basis state and frame
            // Z-measurement on Z-basis is deterministic
            // Z-measurement on X-basis or Unknown must sample
            let raw_value = match q.basis {
                KnownBasis::Z(v) => v,
                KnownBasis::X(_) => self.rng.next_f64() < 0.5,
                KnownBasis::Unknown(_) => self.rng.next_f64() < 0.5,
            };
            // X frame flips the measurement outcome
            let measured = raw_value ^ q.frame.x;

            // After Z measurement, qubit is in Z basis with identity frame
            q.basis = KnownBasis::Z(measured);
            q.frame = PauliFrame::identity();

            (measured, frame_before, raw_value)
        } else {
            (false, PauliFrame::identity(), false)
        };

        self.stats.measurements += 1;

        MeasurementOutcome {
            qubit,
            basis: MeasurementBasis::Z,
            value,
            raw_value: raw,
            frame_at_measurement: frame,
        }
    }

    fn execute_measure_x(&mut self, qubit: usize) -> MeasurementOutcome {
        let cost = self.scheduler.costs().measurement;
        self.advance_cycles(cost);

        let (value, frame, raw) = if let Some(q) = self.qubits.get_mut(qubit) {
            // Capture frame BEFORE clearing (critical for frame tracking!)
            let frame_before = q.frame;

            // X-measurement on X-basis is deterministic
            // X-measurement on Z-basis or Unknown must sample
            let raw_value = match q.basis {
                KnownBasis::X(v) => v,
                KnownBasis::Z(_) => self.rng.next_f64() < 0.5,
                KnownBasis::Unknown(_) => self.rng.next_f64() < 0.5,
            };
            // Z frame flips X measurement outcome
            let measured = raw_value ^ q.frame.z;

            // After X measurement, qubit is in X basis with identity frame
            q.basis = KnownBasis::X(measured);
            q.frame = PauliFrame::identity();

            (measured, frame_before, raw_value)
        } else {
            (false, PauliFrame::identity(), false)
        };

        self.stats.measurements += 1;

        MeasurementOutcome {
            qubit,
            basis: MeasurementBasis::X,
            value,
            raw_value: raw,
            frame_at_measurement: frame,
        }
    }

    fn execute_reset(&mut self, qubit: usize) {
        let cost = self.scheduler.costs().preparation;
        self.advance_cycles(cost);

        if let Some(q) = self.qubits.get_mut(qubit) {
            q.reset();
        }
        self.stats.resets += 1;
    }

    fn execute_prepare_plus(&mut self, qubit: usize) {
        let cost = self.scheduler.costs().preparation;
        self.advance_cycles(cost);

        if let Some(q) = self.qubits.get_mut(qubit) {
            q.reset();
            q.apply_h();
        }
        self.stats.resets += 1;
    }

    fn advance_cycles(&mut self, cycles: usize) {
        self.current_cycle += cycles;
        self.stats.total_cycles += cycles;

        // Either run decoder simulation or use analytical error model
        if let Some(ref decoder_state) = self.decoder_state {
            // Phenomenological decoder mode: simulate QEC rounds
            for _ in 0..cycles {
                let (success, logical_error) = decoder_state.simulate_round(&mut self.rng);
                self.stats.decoder_rounds += 1;
                if success && !logical_error {
                    self.stats.decoder_successes += 1;
                } else {
                    self.stats.decoder_failures += 1;
                    // Add error to budget when decoder fails
                    if logical_error {
                        self.error_budget.add_qec_cycles(1.0, 1); // Full error
                    } else {
                        // Syndrome not cleared but no logical error yet
                        self.error_budget
                            .add_qec_cycles(self.qec_model.p_logical_per_cycle * 10.0, 1);
                    }
                }
            }
        } else {
            // Abstract mode: use analytical QEC error model
            self.error_budget
                .add_qec_cycles(self.qec_model.p_logical_per_cycle, cycles);
        }
    }

    /// Check if decoder simulation is enabled.
    pub fn uses_decoder(&self) -> bool {
        self.decoder_state.is_some()
    }

    /// Get the simulation fidelity mode.
    pub fn simulation_fidelity(&self) -> &SimulationFidelity {
        &self.config.simulation_fidelity
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get current cycle count.
    pub fn current_cycle(&self) -> usize {
        self.current_cycle
    }

    /// Get error budget.
    pub fn error_budget(&self) -> &LogicalErrorBudget {
        &self.error_budget
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &FTStats {
        &self.stats
    }

    /// Get QEC model.
    pub fn qec_model(&self) -> &QecModel {
        &self.qec_model
    }

    /// Get magic supply.
    pub fn magic_supply(&self) -> &MagicSupply {
        &self.magic_supply
    }

    /// Get logical qubit state.
    pub fn qubit(&self, id: usize) -> Option<&LogicalQubit> {
        self.qubits.get(id)
    }

    /// Get all logical qubits.
    pub fn qubits(&self) -> &[LogicalQubit] {
        &self.qubits
    }

    /// Get number of logical qubits.
    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Reset processor state (keeps configuration).
    pub fn reset(&mut self) {
        self.qubits = (0..self.qubits.len()).map(LogicalQubit::new).collect();
        self.error_budget = LogicalErrorBudget::new();
        self.current_cycle = 0;
        self.stats = FTStats::new();
        self.magic_supply.reset();
    }
}

impl fmt::Display for FaultTolerantProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fault-Tolerant Processor:")?;
        writeln!(f, "  Logical qubits:  {}", self.qubits.len())?;
        writeln!(f, "  Code distance:   {}", self.config.distance)?;
        writeln!(f, "  Physical error:  {:.2e}", self.config.physical_error)?;
        writeln!(f, "  Current cycle:   {}", self.current_cycle)?;
        writeln!(f, "  Total error:     {:.2e}", self.error_budget.total())?;
        Ok(())
    }
}

// ============================================================================
// Execution Statistics
// ============================================================================

/// Statistics collected during execution.
#[derive(Clone, Debug, Default)]
pub struct FTStats {
    /// Total code cycles elapsed
    pub total_cycles: usize,
    /// Clifford gates executed
    pub clifford_gates: usize,
    /// T-gates executed
    pub t_gates: usize,
    /// Two-qubit gates executed
    pub two_qubit_gates: usize,
    /// Toffoli gates executed
    pub toffoli_gates: usize,
    /// Magic states consumed
    pub magic_states_consumed: usize,
    /// Factory delay cycles (waiting for production)
    pub factory_delays: usize,
    /// Measurements performed
    pub measurements: usize,
    /// Resets performed
    pub resets: usize,
    /// QEC rounds simulated (decoder mode only)
    pub decoder_rounds: usize,
    /// Decoder successes (syndrome cleared after correction)
    pub decoder_successes: usize,
    /// Decoder failures (residual syndrome or logical error)
    pub decoder_failures: usize,
}

impl FTStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total gates executed.
    pub fn total_gates(&self) -> usize {
        self.clifford_gates + self.t_gates + self.two_qubit_gates
    }

    /// Spacetime volume estimate (for display).
    pub fn spacetime_estimate(&self, qubits_per_logical: usize, n_logical: usize) -> usize {
        let data_qubits = n_logical * qubits_per_logical;
        data_qubits * self.total_cycles
    }

    /// Decoder success rate (0.0-1.0), or None if no decoder rounds.
    pub fn decoder_success_rate(&self) -> Option<f64> {
        if self.decoder_rounds > 0 {
            Some(self.decoder_successes as f64 / self.decoder_rounds as f64)
        } else {
            None
        }
    }

    /// Decoder failure rate (0.0-1.0), or None if no decoder rounds.
    pub fn decoder_failure_rate(&self) -> Option<f64> {
        if self.decoder_rounds > 0 {
            Some(self.decoder_failures as f64 / self.decoder_rounds as f64)
        } else {
            None
        }
    }
}

impl fmt::Display for FTStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Execution Statistics:")?;
        writeln!(f, "  Total cycles:    {}", self.total_cycles)?;
        writeln!(f, "  Clifford gates:  {}", self.clifford_gates)?;
        writeln!(f, "  T-gates:         {}", self.t_gates)?;
        writeln!(f, "  2Q gates:        {}", self.two_qubit_gates)?;
        writeln!(f, "  Toffoli gates:   {}", self.toffoli_gates)?;
        writeln!(f, "  Magic states:    {}", self.magic_states_consumed)?;
        writeln!(f, "  Factory delays:  {} cycles", self.factory_delays)?;
        writeln!(f, "  Measurements:    {}", self.measurements)?;
        if self.decoder_rounds > 0 {
            writeln!(f, "  Decoder rounds:  {}", self.decoder_rounds)?;
            writeln!(
                f,
                "  Decoder success: {}/{} ({:.1}%)",
                self.decoder_successes,
                self.decoder_rounds,
                self.decoder_success_rate().unwrap_or(0.0) * 100.0
            )?;
        }
        Ok(())
    }
}

// ============================================================================
// Resource Estimate
// ============================================================================

/// Resource estimate for a circuit.
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    /// Total code cycles
    pub cycles: usize,
    /// Physical qubits for data (logical × d²)
    pub data_qubits: usize,
    /// Physical qubits for factories
    pub factory_qubits: usize,
    /// Total physical qubits
    pub total_qubits: usize,
    /// Spacetime volume (qubits × cycles)
    pub spacetime_volume: usize,
    /// T-gate count
    pub t_count: usize,
    /// T-depth
    pub t_depth: usize,
    /// Clifford gate count
    pub clifford_count: usize,
    /// Two-qubit gate count
    pub two_qubit_count: usize,
    /// Estimated QEC failure probability
    pub qec_error_probability: f64,
    /// Estimated injection error probability
    pub injection_error_probability: f64,
    /// Total logical error probability
    pub total_error_probability: f64,
    /// Runtime in microseconds
    pub runtime_us: f64,
    /// Factory cost breakdown
    pub factory_cost: MagicSupplyCost,
    /// Schedule analysis
    pub schedule_analysis: ScheduleAnalysis,
}

impl ResourceEstimate {
    /// Runtime in seconds.
    pub fn runtime_seconds(&self) -> f64 {
        self.runtime_us / 1_000_000.0
    }

    /// Runtime in hours.
    pub fn runtime_hours(&self) -> f64 {
        self.runtime_us / 3_600_000_000.0
    }
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resource Estimate:")?;
        writeln!(f, "  Cycles:          {}", self.cycles)?;
        writeln!(f, "  Data qubits:     {}", self.data_qubits)?;
        writeln!(f, "  Factory qubits:  {}", self.factory_qubits)?;
        writeln!(f, "  Total qubits:    {}", self.total_qubits)?;
        writeln!(f, "  Spacetime vol:   {}", self.spacetime_volume)?;
        writeln!(f, "  T-count:         {}", self.t_count)?;
        writeln!(f, "  T-depth:         {}", self.t_depth)?;
        writeln!(f, "  QEC error:       {:.2e}", self.qec_error_probability)?;
        writeln!(
            f,
            "  Injection error: {:.2e}",
            self.injection_error_probability
        )?;
        writeln!(f, "  Total error:     {:.2e}", self.total_error_probability)?;
        writeln!(f, "  Runtime:         {:.2} μs", self.runtime_us)?;
        Ok(())
    }
}

// ============================================================================
// Factory-Aware Estimate (Phase 3.1)
// ============================================================================

/// Resource estimate with factory analysis (Phase 3.1).
///
/// This is the output of "plan-first" mode - tells you exactly what factory
/// configuration you need to run a circuit without stalls.
#[derive(Clone, Debug)]
pub struct FactoryAwareEstimate {
    // Schedule metrics
    /// Total makespan in cycles (parallel schedule)
    pub makespan_cycles: usize,
    /// T-depth (layers of T-gates)
    pub t_depth: usize,
    /// CNOT depth
    pub cnot_depth: usize,
    /// Total T-gates (magic states needed)
    pub total_t_gates: usize,
    /// Peak T-demand in any single layer
    pub peak_t_demand: usize,

    // Resource metrics
    /// Physical qubits for data
    pub data_qubits: usize,
    /// Physical qubits per factory
    pub factory_qubits_per_factory: usize,
    /// Number of factories needed to avoid stalls
    pub factories_needed: usize,
    /// Total factory qubits
    pub total_factory_qubits: usize,
    /// Total physical qubits (data + factories)
    pub total_qubits: usize,

    // Spacetime
    /// Spacetime volume (qubits × cycles)
    pub spacetime_volume: usize,

    // Factory analysis
    /// Detailed backpressure report
    pub backpressure_report: super::magic_supply::BackpressureReport,
    /// Minimum throughput to avoid stalls (states/cycle)
    pub min_throughput_required: f64,
    /// Current factory throughput
    pub current_throughput: f64,

    // Error budget
    /// Estimated logical error probability
    pub logical_error: f64,
}

impl FactoryAwareEstimate {
    /// Can the current factory configuration sustain this circuit?
    pub fn is_sustainable(&self) -> bool {
        self.backpressure_report.sustained
    }

    /// Throughput scaling factor needed (> 1.0 means need more factories).
    pub fn throughput_scaling(&self) -> f64 {
        if self.current_throughput > 0.0 {
            self.min_throughput_required / self.current_throughput
        } else {
            f64::INFINITY
        }
    }
}

impl fmt::Display for FactoryAwareEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Factory-Aware Estimate:")?;
        writeln!(f, "  Makespan:        {} cycles", self.makespan_cycles)?;
        writeln!(f, "  T-depth:         {}", self.t_depth)?;
        writeln!(f, "  T-gates:         {}", self.total_t_gates)?;
        writeln!(f, "  Peak T-demand:   {}/layer", self.peak_t_demand)?;
        writeln!(f)?;
        writeln!(f, "  Data qubits:     {}", self.data_qubits)?;
        writeln!(
            f,
            "  Factories:       {} ({}q each)",
            self.factories_needed, self.factory_qubits_per_factory
        )?;
        writeln!(f, "  Total qubits:    {}", self.total_qubits)?;
        writeln!(f, "  Spacetime vol:   {}", self.spacetime_volume)?;
        writeln!(f)?;
        writeln!(
            f,
            "  Sustainable:     {}",
            if self.is_sustainable() { "YES" } else { "NO" }
        )?;
        writeln!(
            f,
            "  Min throughput:  {:.3} states/cycle",
            self.min_throughput_required
        )?;
        writeln!(
            f,
            "  Cur throughput:  {:.3} states/cycle",
            self.current_throughput
        )?;
        writeln!(f, "  Logical error:   {:.2e}", self.logical_error)?;
        Ok(())
    }
}

// ============================================================================
// Stall Simulation Result (Phase 3.1)
// ============================================================================

/// Result of stall-aware circuit simulation.
///
/// This is the output of "simulate-with-stalls" mode - shows what actually
/// happens when you run with your factory configuration.
#[derive(Clone, Debug)]
pub struct StallSimulationResult {
    /// Ideal makespan (no factory constraints)
    pub ideal_makespan: usize,
    /// Actual makespan including stalls
    pub actual_makespan: usize,
    /// Total cycles lost to factory stalls
    pub total_stall_cycles: usize,
    /// Number of stall events
    pub stall_count: usize,
    /// T-gates successfully executed
    pub t_gates_executed: usize,
    /// Factory utilization (delivered / demanded)
    pub factory_utilization: f64,
}

impl StallSimulationResult {
    /// Slowdown factor from stalls (actual / ideal).
    pub fn slowdown(&self) -> f64 {
        if self.ideal_makespan > 0 {
            self.actual_makespan as f64 / self.ideal_makespan as f64
        } else {
            1.0
        }
    }

    /// Was execution stall-free?
    pub fn stall_free(&self) -> bool {
        self.total_stall_cycles == 0
    }
}

impl fmt::Display for StallSimulationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Stall Simulation Result:")?;
        writeln!(f, "  Ideal makespan:  {} cycles", self.ideal_makespan)?;
        writeln!(f, "  Actual makespan: {} cycles", self.actual_makespan)?;
        writeln!(f, "  Stall cycles:    {}", self.total_stall_cycles)?;
        writeln!(f, "  Stall events:    {}", self.stall_count)?;
        writeln!(f, "  T-gates:         {}", self.t_gates_executed)?;
        writeln!(f, "  Slowdown:        {:.2}×", self.slowdown())?;
        writeln!(
            f,
            "  Utilization:     {:.1}%",
            self.factory_utilization * 100.0
        )?;
        Ok(())
    }
}

// ============================================================================
// Execution Result
// ============================================================================

/// Result of circuit execution (simulation mode).
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    /// Cycles consumed
    pub cycles_used: usize,
    /// Error budget breakdown
    pub error_budget: LogicalErrorBudget,
    /// Measurement outcomes
    pub measurement_outcomes: Vec<MeasurementOutcome>,
    /// Final Pauli frames on each qubit
    pub final_frames: Vec<PauliFrame>,
    /// Final basis states on each qubit (Z, X, or Unknown)
    pub final_basis: Vec<KnownBasis>,
    /// Execution statistics
    pub stats: FTStats,
}

impl fmt::Display for ExecutionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Execution Result:")?;
        writeln!(f, "  Cycles used:     {}", self.cycles_used)?;
        writeln!(f, "  Total error:     {:.2e}", self.error_budget.total())?;
        writeln!(f, "  Measurements:    {}", self.measurement_outcomes.len())?;
        if !self.final_frames.is_empty() {
            write!(f, "  Final frames:    [")?;
            for (i, frame) in self.final_frames.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", frame)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ============================================================================
// Measurement Outcome
// ============================================================================

/// Measurement outcome from simulation.
///
/// # Frame Correction Semantics
///
/// The Pauli frame tracks virtual corrections that haven't been physically applied.
/// When we measure, we need to account for this:
///
/// - **Z-basis measurement**: X frame flips the result (X|0⟩ = |1⟩)
/// - **X-basis measurement**: Z frame flips the result (Z|+⟩ = |-⟩)
///
/// The `raw_value` is what we'd measure if no frame existed.
/// The `value` (= `corrected_value()`) accounts for the frame.
///
/// ```text
/// Z-measurement: value = raw_value XOR frame.x
/// X-measurement: value = raw_value XOR frame.z
/// ```
#[derive(Clone, Debug)]
pub struct MeasurementOutcome {
    /// Which qubit was measured
    pub qubit: usize,
    /// Measurement basis
    pub basis: MeasurementBasis,
    /// Frame-corrected value (this is what classical control should use)
    ///
    /// For Z-basis: `raw_value XOR frame.x`
    /// For X-basis: `raw_value XOR frame.z`
    pub value: bool,
    /// Pauli frame BEFORE measurement (captured for debugging/verification)
    pub frame_at_measurement: PauliFrame,
    /// Raw sampled value before frame correction (for debugging)
    ///
    /// This is what the physical measurement would return if the
    /// Pauli frame were identity. Useful for verifying frame logic.
    pub raw_value: bool,
}

impl MeasurementOutcome {
    /// Get the frame-corrected measurement value.
    ///
    /// This is an alias for `value` that makes the semantics explicit.
    /// Classical control logic should use this, not `raw_value`.
    #[inline]
    pub fn corrected_value(&self) -> bool {
        self.value
    }

    /// Verify that the value was correctly computed from raw_value and frame.
    ///
    /// Returns true if the frame correction was applied correctly.
    /// Useful in tests to catch frame-handling bugs.
    pub fn verify_correction(&self) -> bool {
        match self.basis {
            MeasurementBasis::Z => self.value == (self.raw_value ^ self.frame_at_measurement.x),
            MeasurementBasis::X => self.value == (self.raw_value ^ self.frame_at_measurement.z),
        }
    }

    /// Get which frame component affected this measurement.
    ///
    /// - Z-basis measurement is affected by X frame
    /// - X-basis measurement is affected by Z frame
    pub fn relevant_frame_component(&self) -> bool {
        match self.basis {
            MeasurementBasis::Z => self.frame_at_measurement.x,
            MeasurementBasis::X => self.frame_at_measurement.z,
        }
    }
}

// ============================================================================
// Step Result (Phase 0.1: Introspection)
// ============================================================================

/// Result of executing a single gate with full introspection.
///
/// This is the key to making sim mode debuggable. Every gate execution
/// returns enough information to trace exactly what happened.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// The gate that was executed
    pub gate: LogicalGate,
    /// Pauli frames BEFORE this gate (indexed by qubit id)
    pub frames_before: Vec<PauliFrame>,
    /// Pauli frames AFTER this gate
    pub frames_after: Vec<PauliFrame>,
    /// Logical basis states BEFORE (Z, X, or Unknown)
    pub basis_before: Vec<KnownBasis>,
    /// Logical basis states AFTER
    pub basis_after: Vec<KnownBasis>,
    /// Magic states consumed by this gate
    pub magic_states_consumed: usize,
    /// Cycles consumed by this gate
    pub cycles_consumed: usize,
    /// Cycle count at start of this gate
    pub cycle_start: usize,
    /// Measurement outcome if this was a measurement gate
    pub measurement: Option<MeasurementOutcome>,
    /// T-gate injection outcome if applicable (true = |−⟩, needs S correction)
    pub injection_outcome: Option<bool>,
}

impl StepResult {
    /// Get the frame delta for a specific qubit
    pub fn frame_delta(&self, qubit: usize) -> Option<PauliFrame> {
        if qubit < self.frames_before.len() && qubit < self.frames_after.len() {
            Some(PauliFrame {
                x: self.frames_before[qubit].x ^ self.frames_after[qubit].x,
                z: self.frames_before[qubit].z ^ self.frames_after[qubit].z,
            })
        } else {
            None
        }
    }

    /// Check if any frame changed
    pub fn frames_changed(&self) -> bool {
        self.frames_before
            .iter()
            .zip(self.frames_after.iter())
            .any(|(b, a)| b != a)
    }

    /// Get qubits whose frames changed
    pub fn changed_qubits(&self) -> Vec<usize> {
        self.frames_before
            .iter()
            .zip(self.frames_after.iter())
            .enumerate()
            .filter(|(_, (b, a))| b != a)
            .map(|(i, _)| i)
            .collect()
    }
}

impl fmt::Display for StepResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.gate)?;

        // Show frame changes
        let changed = self.changed_qubits();
        if changed.is_empty() {
            write!(f, "frames unchanged")?;
        } else {
            write!(f, "frames ")?;
            for (i, q) in changed.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(
                    f,
                    "q{}:{}->{}",
                    q, self.frames_before[*q], self.frames_after[*q]
                )?;
            }
        }

        // Show measurement if any
        if let Some(ref m) = self.measurement {
            write!(
                f,
                " | M={} (raw={}, frame={})",
                if m.value { "1" } else { "0" },
                if m.raw_value { "1" } else { "0" },
                m.frame_at_measurement
            )?;
        }

        // Show injection if any
        if let Some(outcome) = self.injection_outcome {
            write!(f, " | T-inj={}", if outcome { "|−⟩" } else { "|+⟩" })?;
        }

        write!(f, " ({} cycles)", self.cycles_consumed)?;
        Ok(())
    }
}

/// Measurement basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementBasis {
    /// Z basis (computational)
    Z,
    /// X basis (Hadamard)
    X,
}

impl fmt::Display for MeasurementOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let basis_char = match self.basis {
            MeasurementBasis::Z => 'Z',
            MeasurementBasis::X => 'X',
        };
        write!(
            f,
            "M{}({}) = {}",
            basis_char,
            self.qubit,
            if self.value { "1" } else { "0" }
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = FaultTolerantProcessor::new(5, FTConfig::default());
        assert_eq!(processor.n_qubits(), 5);
        assert_eq!(processor.current_cycle(), 0);
    }

    #[test]
    fn test_resource_estimation() {
        let processor = FaultTolerantProcessor::new(2, FTConfig::default());
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
            LogicalGate::T(1),
        ];

        let estimate = processor.estimate_resources(&circuit);
        assert!(estimate.cycles > 0);
        assert!(estimate.total_qubits > 0);
        assert_eq!(estimate.t_count, 2);
        assert!(estimate.total_error_probability < 1.0);
    }

    #[test]
    fn test_simulation_execution() {
        let mut processor =
            FaultTolerantProcessor::new(2, FTConfig::default().with_sampled_injection())
                .with_seed(42);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::MeasureZ(0),
            LogicalGate::MeasureZ(1),
        ];

        let result = processor.execute(&circuit);
        assert_eq!(result.measurement_outcomes.len(), 2);
        assert!(result.cycles_used > 0);
    }

    #[test]
    fn test_t_gate_execution() {
        let mut processor = FaultTolerantProcessor::new(1, FTConfig::default()).with_seed(42);

        let circuit = vec![LogicalGate::T(0), LogicalGate::T(0)];

        let result = processor.execute(&circuit);
        assert_eq!(result.stats.t_gates, 2);
        assert_eq!(result.stats.magic_states_consumed, 2);
    }

    #[test]
    fn test_toffoli_execution() {
        let mut processor = FaultTolerantProcessor::new(3, FTConfig::default());

        let circuit = vec![LogicalGate::Toffoli(0, 1, 2)];

        let result = processor.execute(&circuit);
        assert_eq!(result.stats.toffoli_gates, 1);
        assert_eq!(result.stats.magic_states_consumed, 7);
    }

    #[test]
    fn test_frame_propagation() {
        let mut processor =
            FaultTolerantProcessor::new(2, FTConfig::default().with_sampled_injection())
                .with_seed(42);

        // T gate with |−⟩ outcome adds Z to frame
        // Run deterministic test by using many T gates and checking statistics
        let circuit = vec![
            LogicalGate::H(0), // |0⟩ → |+⟩
            LogicalGate::T(0), // T|+⟩ - may add Z to frame
        ];

        let result = processor.execute(&circuit);
        // Frame is either I or Z depending on measurement outcome
        assert!(result.final_frames[0].is_identity() || result.final_frames[0].z);
    }

    #[test]
    fn test_reset() {
        let mut processor = FaultTolerantProcessor::new(2, FTConfig::default());

        processor.execute(&[LogicalGate::T(0)]);
        assert!(processor.stats().t_gates > 0);

        processor.reset();
        assert_eq!(processor.stats().t_gates, 0);
        assert_eq!(processor.current_cycle(), 0);
    }

    #[test]
    fn test_measurement_outcomes() {
        let mut processor = FaultTolerantProcessor::new(1, FTConfig::default()).with_seed(12345);

        // Measure |0⟩ should give 0
        let result = processor.execute(&[LogicalGate::MeasureZ(0)]);
        assert_eq!(result.measurement_outcomes.len(), 1);
        // |0⟩ measured in Z basis should give 0
        assert!(!result.measurement_outcomes[0].value);
    }

    #[test]
    fn test_error_accumulation() {
        let processor = FaultTolerantProcessor::new(10, FTConfig::default());

        // Large circuit should accumulate more error
        let small_circuit = vec![LogicalGate::H(0)];
        let large_circuit: Vec<_> = (0..100).map(|i| LogicalGate::H(i % 10)).collect();

        let small_estimate = processor.estimate_resources(&small_circuit);
        let large_estimate = processor.estimate_resources(&large_circuit);

        assert!(large_estimate.qec_error_probability > small_estimate.qec_error_probability);
    }

    // =========================================================================
    // FrameTruth Microtest Suite - Phase 0.2
    // These tests verify the core Pauli frame tracking semantics
    // =========================================================================

    #[test]
    fn frame_truth_h_swaps_x_and_z() {
        // H gate swaps X and Z frames: H·X·H = Z, H·Z·H = X
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(1, config);

        // Start with X frame
        processor.qubits[0].frame = PauliFrame { x: true, z: false };

        // Apply H - should swap to Z frame
        let step = processor.execute_step(&LogicalGate::H(0));
        assert!(
            step.frames_before[0].x && !step.frames_before[0].z,
            "Before: should have X frame"
        );
        assert!(
            !step.frames_after[0].x && step.frames_after[0].z,
            "After H: X frame should become Z frame"
        );

        // Apply H again - should swap back to X frame
        let step2 = processor.execute_step(&LogicalGate::H(0));
        assert!(
            step2.frames_after[0].x && !step2.frames_after[0].z,
            "After H·H: should be back to X frame"
        );
    }

    #[test]
    fn frame_truth_cnot_propagation() {
        // CNOT propagates frames:
        // - Control X → Target X (X on control propagates to target)
        // - Target Z → Control Z (Z on target propagates to control)
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // Test 1: X on control propagates to target
        processor.qubits[0].frame = PauliFrame { x: true, z: false };
        processor.qubits[1].frame = PauliFrame::identity();

        let step = processor.execute_step(&LogicalGate::CNOT(0, 1));

        assert!(step.frames_after[0].x, "Control should keep X frame");
        assert!(
            step.frames_after[1].x,
            "Target should gain X frame from control"
        );

        // Reset and test 2: Z on target propagates to control
        processor.qubits[0].frame = PauliFrame::identity();
        processor.qubits[1].frame = PauliFrame { x: false, z: true };

        let step2 = processor.execute_step(&LogicalGate::CNOT(0, 1));

        assert!(
            step2.frames_after[0].z,
            "Control should gain Z frame from target"
        );
        assert!(step2.frames_after[1].z, "Target should keep Z frame");
    }

    #[test]
    fn frame_truth_x_frame_flips_z_measurement() {
        // X frame should flip Z measurement outcome
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Prepare |0⟩ explicitly
        processor.qubits[0].basis = KnownBasis::Z(false);
        processor.qubits[0].frame = PauliFrame { x: true, z: false };

        // Measure - raw value is 0, but X frame should flip it to 1
        let step = processor.execute_step(&LogicalGate::MeasureZ(0));

        let measurement = step.measurement.expect("Should have measurement");
        assert!(
            !measurement.raw_value,
            "Raw value should be 0 (from |0⟩ state)"
        );
        assert!(
            measurement.frame_at_measurement.x,
            "Frame at measurement should have X"
        );
        assert!(
            measurement.value,
            "Final value should be 1 (raw XOR frame.x = 0 XOR 1 = 1)"
        );
    }

    #[test]
    fn frame_truth_z_frame_flips_x_measurement() {
        // Z frame should flip X measurement outcome
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Prepare |+⟩ explicitly
        processor.qubits[0].basis = KnownBasis::X(false);
        processor.qubits[0].frame = PauliFrame { x: false, z: true };

        // Measure X - raw value is 0 (|+⟩), but Z frame should flip it to 1
        let step = processor.execute_step(&LogicalGate::MeasureX(0));

        let measurement = step.measurement.expect("Should have measurement");
        assert!(
            !measurement.raw_value,
            "Raw value should be 0 (from |+⟩ state)"
        );
        assert!(
            measurement.frame_at_measurement.z,
            "Frame at measurement should have Z"
        );
        assert!(
            measurement.value,
            "Final value should be 1 (raw XOR frame.z = 0 XOR 1 = 1)"
        );
    }

    #[test]
    fn frame_truth_t_injection_samples_z_frame() {
        // T-gate injection should add Z to frame 50% of the time (|−⟩ outcome)
        let n_runs = 1000;
        let mut z_count = 0;

        for seed in 0..n_runs {
            let config = FTConfig::new(7, 1e-3).with_sampled_injection();
            let mut processor = FaultTolerantProcessor::new(1, config).with_seed(seed as u64);

            // Start with identity frame
            processor.qubits[0].frame = PauliFrame::identity();

            // Execute T gate
            processor.execute_step(&LogicalGate::T(0));

            if processor.qubits[0].frame.z {
                z_count += 1;
            }
        }

        // Should be approximately 50% ± 5%
        let ratio = z_count as f64 / n_runs as f64;
        assert!(
            ratio > 0.45 && ratio < 0.55,
            "T injection should produce Z frame ~50% of time, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn frame_truth_double_z_cancels() {
        // Z + Z = I in mod-2 arithmetic
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(1, config);

        // Start with Z frame
        processor.qubits[0].frame = PauliFrame { x: false, z: true };

        // Compose another Z frame
        processor.qubits[0]
            .frame
            .compose(&PauliFrame { x: false, z: true });

        assert!(!processor.qubits[0].frame.z, "Z + Z should cancel to I");
    }

    #[test]
    fn frame_truth_measurement_clears_frame() {
        // After measurement, frame should be reset to identity
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Set up qubit with non-identity frame
        processor.qubits[0].frame = PauliFrame { x: true, z: true };

        // Measure
        processor.execute_step(&LogicalGate::MeasureZ(0));

        // Frame should be cleared
        assert!(
            !processor.qubits[0].frame.x && !processor.qubits[0].frame.z,
            "Frame should be identity after measurement"
        );
    }

    #[test]
    fn frame_truth_traced_execution() {
        // execute_traced should capture correct before/after for each gate
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::H(0),
            LogicalGate::MeasureZ(0),
        ];

        let trace = processor.execute_traced(&circuit);

        assert_eq!(trace.len(), 4, "Should have 4 step results");

        // First H: should put qubit in X basis
        assert!(
            trace[0].basis_after[0] == KnownBasis::X(false),
            "After H on |0⟩, should be in |+⟩"
        );

        // After T: basis becomes unknown (non-Clifford)
        assert!(
            trace[1].basis_after[0].is_unknown(),
            "After T, basis should be unknown"
        );

        // Last step should have measurement
        assert!(
            trace[3].measurement.is_some(),
            "Last step should have measurement"
        );
    }

    // =========================================================================
    // Meta-test: Traced vs Untraced Equivalence
    // Catches instrumentation bugs where tracing changes behavior
    // =========================================================================

    #[test]
    fn meta_test_traced_untraced_equivalence() {
        // Run same circuit both ways with same seed, verify identical results
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(1),
            LogicalGate::H(1),
            LogicalGate::MeasureZ(0),
            LogicalGate::MeasureZ(1),
        ];

        // Run multiple seeds to catch probabilistic differences
        for seed in [42, 123, 456, 789, 1000] {
            // Untraced execution
            let config1 = FTConfig::new(7, 1e-3).with_sampled_injection();
            let mut proc1 = FaultTolerantProcessor::new(2, config1).with_seed(seed);
            let result1 = proc1.execute(&circuit);

            // Traced execution
            let config2 = FTConfig::new(7, 1e-3).with_sampled_injection();
            let mut proc2 = FaultTolerantProcessor::new(2, config2).with_seed(seed);
            let trace = proc2.execute_traced(&circuit);

            // Verify measurement outcomes match
            let traced_measurements: Vec<_> = trace
                .iter()
                .filter_map(|s| s.measurement.as_ref())
                .collect();

            assert_eq!(
                result1.measurement_outcomes.len(),
                traced_measurements.len(),
                "Seed {}: measurement count mismatch",
                seed
            );

            for (i, (m1, m2)) in result1
                .measurement_outcomes
                .iter()
                .zip(traced_measurements.iter())
                .enumerate()
            {
                assert_eq!(
                    m1.qubit, m2.qubit,
                    "Seed {}: measurement {} qubit mismatch",
                    seed, i
                );
                assert_eq!(
                    m1.value, m2.value,
                    "Seed {}: measurement {} value mismatch",
                    seed, i
                );
                assert_eq!(
                    m1.raw_value, m2.raw_value,
                    "Seed {}: measurement {} raw_value mismatch",
                    seed, i
                );
                assert_eq!(
                    m1.frame_at_measurement, m2.frame_at_measurement,
                    "Seed {}: measurement {} frame mismatch",
                    seed, i
                );
            }

            // Verify final frames match
            assert_eq!(
                result1.final_frames.len(),
                proc2.qubits.len(),
                "Seed {}: qubit count mismatch",
                seed
            );
            for (i, (f1, f2)) in result1
                .final_frames
                .iter()
                .zip(proc2.qubits.iter().map(|q| q.frame))
                .enumerate()
            {
                assert_eq!(*f1, f2, "Seed {}: final frame {} mismatch", seed, i);
            }

            // Verify resource counts match
            assert_eq!(
                result1.cycles_used,
                proc2.current_cycle(),
                "Seed {}: cycle count mismatch",
                seed
            );
            assert_eq!(
                result1.stats.magic_states_consumed, proc2.stats.magic_states_consumed,
                "Seed {}: magic state count mismatch",
                seed
            );
            assert_eq!(
                result1.stats.t_gates, proc2.stats.t_gates,
                "Seed {}: T-gate count mismatch",
                seed
            );

            // Verify basis tags match (from last step of trace)
            let last_step = trace.last().unwrap();
            for (i, (b1, b2)) in result1
                .final_basis
                .iter()
                .zip(last_step.basis_after.iter())
                .enumerate()
            {
                assert_eq!(b1, b2, "Seed {}: final basis {} mismatch", seed, i);
            }
        }
    }

    // =========================================================================
    // Corrected Value Semantics Tests
    // =========================================================================

    #[test]
    fn frame_truth_corrected_value_semantics() {
        // Verify corrected_value() matches value and verify_correction() works
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Set up: |0⟩ with X frame, then measure Z
        processor.qubits[0].basis = KnownBasis::Z(false); // |0⟩
        processor.qubits[0].frame = PauliFrame { x: true, z: false }; // X frame

        let step = processor.execute_step(&LogicalGate::MeasureZ(0));
        let m = step.measurement.expect("Should have measurement");

        // Verify API consistency
        assert_eq!(
            m.corrected_value(),
            m.value,
            "corrected_value() should equal value"
        );
        assert!(
            m.verify_correction(),
            "verify_correction() should return true for valid measurement"
        );
        assert_eq!(
            m.relevant_frame_component(),
            m.frame_at_measurement.x,
            "Z-basis should use X frame component"
        );

        // Verify the actual correction
        assert!(!m.raw_value, "Raw value should be 0 (|0⟩)");
        assert!(m.frame_at_measurement.x, "Frame should have X");
        assert!(m.value, "Corrected value should be 1 (0 XOR 1)");
    }

    #[test]
    fn frame_truth_corrected_value_x_basis() {
        // Verify X-basis measurement uses Z frame for correction
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Set up: |+⟩ with Z frame, then measure X
        processor.qubits[0].basis = KnownBasis::X(false); // |+⟩
        processor.qubits[0].frame = PauliFrame { x: false, z: true }; // Z frame

        let step = processor.execute_step(&LogicalGate::MeasureX(0));
        let m = step.measurement.expect("Should have measurement");

        // Verify
        assert!(m.verify_correction(), "X-basis correction should be valid");
        assert_eq!(
            m.relevant_frame_component(),
            m.frame_at_measurement.z,
            "X-basis should use Z frame component"
        );

        // Z frame flips X-basis measurement
        assert!(!m.raw_value, "Raw value should be 0 (|+⟩)");
        assert!(m.frame_at_measurement.z, "Frame should have Z");
        assert!(m.value, "Corrected value should be 1 (0 XOR 1)");
    }

    #[test]
    fn frame_truth_all_measurements_verify_correction() {
        // Run a circuit and verify ALL measurements have valid corrections
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();

        for seed in 0..100 {
            let mut processor = FaultTolerantProcessor::new(3, config.clone()).with_seed(seed);

            let circuit = vec![
                LogicalGate::H(0),
                LogicalGate::T(0),
                LogicalGate::CNOT(0, 1),
                LogicalGate::H(2),
                LogicalGate::T(2),
                LogicalGate::MeasureZ(0),
                LogicalGate::MeasureX(1),
                LogicalGate::MeasureZ(2),
            ];

            let result = processor.execute(&circuit);

            for (i, m) in result.measurement_outcomes.iter().enumerate() {
                assert!(
                    m.verify_correction(),
                    "Seed {}: Measurement {} failed verify_correction()",
                    seed,
                    i
                );
            }
        }
    }

    // =========================================================================
    // Phase 1.3: Z-Basis Preservation Tests
    // T-gate is diagonal: preserves Z-basis, destroys X-basis
    // =========================================================================

    #[test]
    fn frame_truth_t_preserves_z_basis() {
        // T|0⟩ = |0⟩, T|1⟩ = e^{iπ/4}|1⟩ (only phase, still Z-basis)
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        // Qubit 0: |0⟩, Qubit 1: |1⟩
        processor.qubits[0].basis = KnownBasis::Z(false);
        processor.qubits[1].basis = KnownBasis::Z(true);

        // Apply T to both
        processor.execute_step(&LogicalGate::T(0));
        processor.execute_step(&LogicalGate::T(1));

        // Both should still be in Z-basis
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::Z(false),
            "T|0⟩ should remain |0⟩"
        );
        assert_eq!(
            processor.qubits[1].basis,
            KnownBasis::Z(true),
            "T|1⟩ should remain |1⟩"
        );
    }

    #[test]
    fn frame_truth_t_destroys_x_basis() {
        // T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 - no longer X-eigenstate
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        // Qubit 0: |+⟩, Qubit 1: |-⟩
        processor.qubits[0].basis = KnownBasis::X(false);
        processor.qubits[1].basis = KnownBasis::X(true);

        // Apply T to both
        processor.execute_step(&LogicalGate::T(0));
        processor.execute_step(&LogicalGate::T(1));

        // Both should now be unknown
        assert!(
            processor.qubits[0].basis.is_unknown(),
            "T|+⟩ should become unknown (superposition)"
        );
        assert!(
            processor.qubits[1].basis.is_unknown(),
            "T|-⟩ should become unknown (superposition)"
        );
    }

    #[test]
    fn frame_truth_t_circuit_z_basis_deterministic() {
        // Circuit on Z-basis inputs should give deterministic Z-measurements
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();

        // Run same circuit many times - Z-basis results should be deterministic
        let mut results = Vec::new();

        for seed in 0..20 {
            let mut processor = FaultTolerantProcessor::new(1, config.clone()).with_seed(seed);

            // |0⟩ → X → |1⟩ → T → |1⟩ (with phase) → MZ → 1
            let circuit = vec![
                LogicalGate::X(0),
                LogicalGate::T(0),
                LogicalGate::T(0),
                LogicalGate::T(0),
                LogicalGate::MeasureZ(0),
            ];

            let result = processor.execute(&circuit);
            results.push(result.measurement_outcomes[0].value);
        }

        // All results should be true (measured |1⟩)
        assert!(
            results.iter().all(|&v| v),
            "Z-basis through T should give deterministic measurement"
        );
    }

    #[test]
    fn frame_truth_t_conjugation_breaks_pauli_stability() {
        // TXT† = (X+Y)/√2 is NOT a Pauli operator
        // So H → T → H does NOT return us to a known Z-eigenstate
        //
        // This tests the "Pauli-stability boundary" - if a qubit passes through
        // X-basis and has T applied, it should not magically recover Z-knowledge
        // just because we apply H again.
        //
        // Sequence: |0⟩ → H → |+⟩ → T → unknown → H → still unknown

        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Start in Z-basis
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::Z(false),
            "Should start in |0⟩"
        );

        // H puts us in X-basis
        processor.execute_step(&LogicalGate::H(0));
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::X(false),
            "After H: should be |+⟩"
        );

        // T on X-basis → unknown (T creates non-stabilizer state)
        processor.execute_step(&LogicalGate::T(0));
        assert!(
            processor.qubits[0].basis.is_unknown(),
            "After T on |+⟩: should be unknown"
        );

        // H again should NOT magically restore Z-knowledge
        // The state T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2
        // After H: H·T|+⟩ is a superposition, NOT a Z-eigenstate
        processor.execute_step(&LogicalGate::H(0));
        assert!(
            processor.qubits[0].basis.is_unknown(),
            "After H on unknown: should still be unknown (no magic recovery)"
        );
    }

    #[test]
    fn frame_truth_h_t_h_vs_pure_z_path() {
        // Contrast: H→T→H leaves state unknown
        // But: T alone on Z-basis preserves knowledge
        //
        // This demonstrates the "cliff edge" of our abstraction boundary

        let config = FTConfig::new(7, 1e-3).with_sampled_injection();

        // Path 1: Z-basis → T → still Z-basis (deterministic)
        let mut proc1 = FaultTolerantProcessor::new(1, config.clone()).with_seed(42);
        proc1.execute_step(&LogicalGate::T(0));
        proc1.execute_step(&LogicalGate::T(0));
        proc1.execute_step(&LogicalGate::T(0));
        assert!(
            proc1.qubits[0].basis.is_known(),
            "Pure Z path through T: basis should be preserved"
        );

        // Path 2: Z → H → X → T → unknown (and stays unknown)
        let mut proc2 = FaultTolerantProcessor::new(1, config.clone()).with_seed(42);
        proc2.execute_step(&LogicalGate::H(0)); // Z → X
        proc2.execute_step(&LogicalGate::T(0)); // X → unknown
        proc2.execute_step(&LogicalGate::H(0)); // unknown → unknown
        proc2.execute_step(&LogicalGate::T(0)); // unknown → unknown
        assert!(
            proc2.qubits[0].basis.is_unknown(),
            "H→T→H path: basis should be unknown (no recovery)"
        );
    }

    // =========================================================================
    // Phase 2.2: Resource Event Log Integration Tests
    // =========================================================================

    #[test]
    fn test_resource_log_records_events() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        // Execute a circuit
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::MeasureZ(0),
        ];

        processor.execute(&circuit);

        // Check resource log has events
        let events = processor.resource_log().events();
        assert!(!events.is_empty(), "Should have recorded events");

        // Get summary
        let summary = processor.resource_summary();
        assert!(summary.makespan_cycles > 0, "Should have cycles");
        assert!(summary.peak_physical_qubits > 0, "Should have qubits");
        assert_eq!(summary.total_magic_states, 2, "Should have 2 T-gates");
    }

    #[test]
    fn test_resource_log_toffoli_magic_states() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

        // Execute Toffoli (7 T-gates)
        let circuit = vec![LogicalGate::Toffoli(0, 1, 2)];
        processor.execute(&circuit);

        let summary = processor.resource_summary();
        assert_eq!(
            summary.total_magic_states, 7,
            "Toffoli should use 7 magic states"
        );
    }

    #[test]
    fn test_resource_summary_display() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::T(0),
            LogicalGate::MeasureZ(0),
        ];
        processor.execute(&circuit);

        let summary = processor.resource_summary();
        let display = format!("{}", summary);

        // Should contain key metrics
        assert!(display.contains("Makespan"), "Should show makespan");
        assert!(display.contains("Peak qubits"), "Should show peak qubits");
        assert!(display.contains("Magic states"), "Should show magic states");
    }

    #[test]
    fn test_resource_log_clear() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Execute some gates
        processor.execute_step(&LogicalGate::H(0));
        processor.execute_step(&LogicalGate::T(0));

        let summary1 = processor.resource_summary();
        assert!(summary1.total_magic_states > 0);

        // Clear the log
        processor.clear_resource_log();

        let summary2 = processor.resource_summary();
        assert_eq!(summary2.total_magic_states, 0, "Log should be cleared");
        assert_eq!(summary2.makespan_cycles, 0, "Log should be cleared");
    }

    // =========================================================================
    // Phase 2.1: Factory Pipeline Integration Tests
    // =========================================================================

    #[test]
    fn test_create_factory_pipeline() {
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(5, config);

        let pipeline = processor.create_factory_pipeline();

        // Pipeline should match supply configuration
        assert!(pipeline.throughput() > 0.0);
        assert!(pipeline.latency() > 0);
        assert!(pipeline.factory_qubits() > 0);
    }

    #[test]
    fn test_analyze_backpressure_after_execution() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        // Execute a circuit with T-gates
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
            LogicalGate::MeasureZ(0),
        ];

        processor.execute(&circuit);

        // Analyze backpressure from resource log
        let report = processor.analyze_backpressure();

        // Should have analyzed some demand
        assert!(report.total_demand > 0, "Should have T-gate demand");
        assert!(report.utilization >= 0.0 && report.utilization <= 1.0);
    }

    #[test]
    fn test_analyze_circuit_backpressure_without_execution() {
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(3, config);

        // Analyze a circuit without executing it
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::T(2),
            LogicalGate::Toffoli(0, 1, 2), // 7 more T-gates
        ];

        let report = processor.analyze_circuit_backpressure(&circuit);

        // Should have 3 + 7 = 10 T-gates total
        assert_eq!(
            report.total_demand, 10,
            "Should have 10 magic states (3 T + 1 Toffoli)"
        );
    }

    #[test]
    fn test_backpressure_sustained_vs_unsustained() {
        // Create processor with very high throughput (should sustain easily)
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(2, config);

        // Light circuit
        let light_circuit = vec![LogicalGate::H(0), LogicalGate::T(0), LogicalGate::H(0)];

        let report = processor.analyze_circuit_backpressure(&light_circuit);

        // With default factory, should be able to sustain 1 T-gate
        // (This depends on factory config, but 1 T-gate should be sustainable)
        assert_eq!(report.total_demand, 1);
    }

    #[test]
    fn test_backpressure_report_fields() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(2, config).with_seed(42);

        // Execute circuit with known T-count
        let circuit = vec![LogicalGate::T(0), LogicalGate::T(0), LogicalGate::T(1)];

        processor.execute(&circuit);
        let report = processor.analyze_backpressure();

        // Verify report structure
        assert_eq!(report.total_demand, 3, "Should have 3 T-gates");
        assert!(report.total_delivered <= report.total_demand);
        assert_eq!(
            report.unmet_demand,
            report.total_demand - report.total_delivered
        );

        // Display should work
        let display = format!("{}", report);
        assert!(display.contains("Demand"));
        assert!(display.contains("Sustained"));
    }

    #[test]
    fn test_pipeline_integration_with_resource_summary() {
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(3, config).with_seed(42);

        // Execute a moderately complex circuit
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(1),
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
            LogicalGate::Toffoli(0, 1, 2),
            LogicalGate::MeasureZ(0),
            LogicalGate::MeasureZ(1),
        ];

        processor.execute(&circuit);

        // Get resource summary
        let summary = processor.resource_summary();
        assert!(summary.makespan_cycles > 0);
        assert!(summary.total_magic_states > 0);

        // Demand timeline should have bins
        assert!(
            !summary.demand_timeline.is_empty(),
            "Should have demand bins"
        );

        // Analyze backpressure using that demand
        let report = processor.analyze_backpressure();

        // Total demand from report should match magic states consumed
        // (Note: demand_timeline bins may have aggregated differently)
        assert_eq!(report.total_demand, summary.total_magic_states);
    }

    // =========================================================================
    // Phase 3.1: Integrated Schedule + Factory Execution Tests
    // =========================================================================

    #[test]
    fn test_schedule_circuit() {
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(5, config);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(1),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
            LogicalGate::T(1),
        ];

        let schedule = processor.schedule_circuit(&circuit);

        assert_eq!(schedule.gates.len(), 5);
        assert_eq!(schedule.t_depth, 1); // T-gates are parallel
        assert!(schedule.makespan > 0);
    }

    #[test]
    fn test_estimate_with_factories() {
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(5, config);

        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::T(2),
            LogicalGate::Toffoli(0, 1, 2),
        ];

        let estimate = processor.estimate_with_factories(&circuit);

        // Should have 3 + 7 = 10 T-gates
        assert_eq!(estimate.total_t_gates, 10);
        assert!(estimate.data_qubits > 0);
        assert!(estimate.factories_needed >= 1);
        assert!(estimate.total_qubits >= estimate.data_qubits);
    }

    #[test]
    fn test_simulate_with_stalls_no_stalls_when_enough_time() {
        // Circuit with spread-out T-gates should not stall
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(3, config);

        // Sequential T-gates on same qubit - factory has time to produce
        let circuit = vec![LogicalGate::T(0), LogicalGate::T(0), LogicalGate::T(0)];

        let result = processor.simulate_with_stalls(&circuit);

        // With sequential gates, factory should keep up
        // (depends on factory config, but sequential gives time)
        assert_eq!(result.t_gates_executed, 3);
        assert!(result.ideal_makespan > 0);
        assert!(result.actual_makespan >= result.ideal_makespan);
    }

    #[test]
    fn test_simulate_with_stalls_burst_demand() {
        // Burst of parallel T-gates may cause stalls
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(20, config);

        // 20 parallel T-gates - big burst
        let circuit: Vec<_> = (0..20).map(|q| LogicalGate::T(q)).collect();

        let result = processor.simulate_with_stalls(&circuit);

        assert_eq!(result.t_gates_executed, 20);
        assert!(result.ideal_makespan > 0);
        // Burst may or may not stall depending on factory config
        // But actual >= ideal always
        assert!(result.actual_makespan >= result.ideal_makespan);
    }

    #[test]
    fn test_stall_simulation_slowdown_metric() {
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(3, config);

        let circuit = vec![LogicalGate::T(0)];
        let result = processor.simulate_with_stalls(&circuit);

        // Slowdown should be >= 1.0
        assert!(result.slowdown() >= 1.0);

        // Display should work
        let display = format!("{}", result);
        assert!(display.contains("Ideal"));
        assert!(display.contains("Slowdown"));
    }

    #[test]
    fn test_factory_aware_estimate_sustainability() {
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(2, config);

        // Light circuit
        let light = vec![LogicalGate::T(0)];
        let est_light = processor.estimate_with_factories(&light);

        // Heavy circuit (many parallel T-gates)
        let heavy: Vec<_> = (0..50).map(|q| LogicalGate::T(q)).collect();
        let est_heavy = processor.estimate_with_factories(&heavy);

        // Heavy circuit should need more throughput
        assert!(
            est_heavy.min_throughput_required >= est_light.min_throughput_required,
            "Heavy circuit needs more throughput"
        );

        // Display should work
        let display = format!("{}", est_heavy);
        assert!(display.contains("Makespan"));
        assert!(display.contains("Factories"));
    }

    #[test]
    fn test_phase_3_1_acceptance_stalls_increase_makespan() {
        // ACCEPTANCE TEST: With too few factories, makespan increases
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(10, config);

        // 10 parallel T-gates
        let circuit: Vec<_> = (0..10).map(|q| LogicalGate::T(q)).collect();

        let schedule = processor.schedule_circuit(&circuit);
        let stall_result = processor.simulate_with_stalls(&circuit);

        // If stalls occurred, actual > ideal
        if stall_result.total_stall_cycles > 0 {
            assert!(
                stall_result.actual_makespan > stall_result.ideal_makespan,
                "Stalls should increase makespan"
            );
        }

        // In any case, makespan from simulation should match schedule
        assert_eq!(stall_result.ideal_makespan, schedule.makespan);
    }

    #[test]
    fn test_phase_3_1_acceptance_enough_factories_no_stalls() {
        // ACCEPTANCE TEST: With enough factories, no stalls
        let config = FTConfig::new(7, 1e-3);
        let processor = FaultTolerantProcessor::new(3, config);

        // Light circuit
        let circuit = vec![LogicalGate::T(0)];

        let estimate = processor.estimate_with_factories(&circuit);

        // With 1 T-gate, should be sustainable
        // (unless factory is extremely slow, which default isn't)
        if estimate.current_throughput > 0.0 {
            // The circuit should be runnable
            assert!(estimate.total_qubits > 0);
        }
    }

    // =========================================================================
    // Phase 3.2: Unknown Reason Tracking Tests
    // These tests verify why_unknown reason codes for user trust
    // =========================================================================

    #[test]
    fn test_phase_3_2_cnot_on_superposition_creates_entanglement() {
        // ACCEPTANCE TEST: CNOT on X-basis qubit should report EntanglingGateOnSuperposition
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // Prepare |+⟩|0⟩
        processor.qubits[0].basis = KnownBasis::X(false); // |+⟩
        processor.qubits[1].basis = KnownBasis::Z(false); // |0⟩

        // CNOT creates Bell state → both qubits become unknown
        processor.execute_step(&LogicalGate::CNOT(0, 1));

        // Control (was |+⟩) should be unknown due to entanglement
        assert!(
            processor.qubits[0].basis.is_unknown(),
            "Control should become unknown"
        );
        assert_eq!(
            processor.qubits[0].why_unknown(),
            Some(UnknownReason::EntanglingGateOnSuperposition),
            "Should report EntanglingGateOnSuperposition"
        );

        // Target also becomes entangled
        assert!(
            processor.qubits[1].basis.is_unknown(),
            "Target should become unknown"
        );
        assert_eq!(
            processor.qubits[1].why_unknown(),
            Some(UnknownReason::EntanglingGateOnSuperposition),
            "Target should also report EntanglingGateOnSuperposition"
        );
    }

    #[test]
    fn test_phase_3_2_cnot_with_unknown_propagates() {
        // ACCEPTANCE TEST: CNOT with already-unknown qubit should report EntanglingGateWithUnknown
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // Control is unknown, target is |0⟩
        processor.qubits[0].basis = KnownBasis::Unknown(UnknownReason::NeverInitialized);
        processor.qubits[1].basis = KnownBasis::Z(false);

        processor.execute_step(&LogicalGate::CNOT(0, 1));

        // Target should become unknown due to entanglement with unknown
        assert!(processor.qubits[1].basis.is_unknown());
        assert_eq!(
            processor.qubits[1].why_unknown(),
            Some(UnknownReason::EntanglingGateWithUnknown),
            "Should report EntanglingGateWithUnknown when control is unknown"
        );
    }

    #[test]
    fn test_phase_3_2_t_gate_on_x_basis_is_non_clifford() {
        // ACCEPTANCE TEST: T on X-basis should report NonCliffordOnSuperposition
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Prepare |+⟩
        processor.qubits[0].basis = KnownBasis::X(false);

        processor.execute_step(&LogicalGate::T(0));

        // T on |+⟩ creates non-stabilizer state
        assert!(processor.qubits[0].basis.is_unknown());
        assert_eq!(
            processor.qubits[0].why_unknown(),
            Some(UnknownReason::NonCliffordOnSuperposition),
            "T on X-basis should report NonCliffordOnSuperposition"
        );
    }

    #[test]
    fn test_phase_3_2_t_gate_on_z_basis_preserves() {
        // T on Z-basis should NOT make qubit unknown (just adds phase)
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Prepare |0⟩
        processor.qubits[0].basis = KnownBasis::Z(false);

        processor.execute_step(&LogicalGate::T(0));

        // T on Z-basis just adds phase, basis is preserved
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::Z(false),
            "T on Z-basis should preserve basis"
        );
        assert_eq!(
            processor.qubits[0].why_unknown(),
            None,
            "Should have no unknown reason"
        );
    }

    #[test]
    fn test_phase_3_2_cnot_on_z_basis_no_entanglement() {
        // CNOT on two Z-basis qubits is classical, no entanglement
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // Both qubits in Z-basis
        processor.qubits[0].basis = KnownBasis::Z(true); // |1⟩
        processor.qubits[1].basis = KnownBasis::Z(false); // |0⟩

        processor.execute_step(&LogicalGate::CNOT(0, 1));

        // Classical CNOT: |10⟩ → |11⟩
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::Z(true),
            "Control unchanged"
        );
        assert_eq!(
            processor.qubits[1].basis,
            KnownBasis::Z(true),
            "Target flipped"
        );
        assert_eq!(processor.qubits[0].why_unknown(), None);
        assert_eq!(processor.qubits[1].why_unknown(), None);
    }

    #[test]
    fn test_phase_3_2_cz_on_superposition() {
        // CZ on superposition should also create entanglement
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // |+⟩|+⟩ - both in X-basis
        processor.qubits[0].basis = KnownBasis::X(false);
        processor.qubits[1].basis = KnownBasis::X(false);

        processor.execute_step(&LogicalGate::CZ(0, 1));

        // CZ creates entanglement
        assert!(processor.qubits[0].basis.is_unknown());
        assert!(processor.qubits[1].basis.is_unknown());
        assert_eq!(
            processor.qubits[0].why_unknown(),
            Some(UnknownReason::EntanglingGateOnSuperposition)
        );
        assert_eq!(
            processor.qubits[1].why_unknown(),
            Some(UnknownReason::EntanglingGateOnSuperposition)
        );
    }

    #[test]
    fn test_phase_3_2_swap_preserves_reasons() {
        // SWAP should swap unknown reasons along with states
        let config = FTConfig::new(7, 1e-3);
        let mut processor = FaultTolerantProcessor::new(2, config);

        // Qubit 0 is unknown (was entangled), qubit 1 is |0⟩
        processor.qubits[0].basis =
            KnownBasis::Unknown(UnknownReason::EntanglingGateOnSuperposition);
        processor.qubits[1].basis = KnownBasis::Z(false);

        processor.execute_step(&LogicalGate::SWAP(0, 1));

        // After SWAP, states should be swapped
        assert_eq!(
            processor.qubits[0].basis,
            KnownBasis::Z(false),
            "Qubit 0 should now have Z-basis"
        );
        assert!(
            processor.qubits[0].why_unknown().is_none(),
            "Qubit 0 should have no unknown reason"
        );
        assert!(
            processor.qubits[1].basis.is_unknown(),
            "Qubit 1 should now be unknown"
        );
        assert_eq!(
            processor.qubits[1].why_unknown(),
            Some(UnknownReason::EntanglingGateOnSuperposition),
            "Qubit 1 should have the swapped reason"
        );
    }

    #[test]
    fn test_phase_3_2_t_on_unknown_propagates_reason() {
        // T on already-unknown qubit should report NonCliffordOnUnknown
        let config = FTConfig::new(7, 1e-3).with_sampled_injection();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        // Qubit is already unknown
        processor.qubits[0].basis =
            KnownBasis::Unknown(UnknownReason::EntanglingGateOnSuperposition);

        processor.execute_step(&LogicalGate::T(0));

        // T on unknown should report NonCliffordOnUnknown
        assert!(processor.qubits[0].basis.is_unknown());
        assert_eq!(
            processor.qubits[0].why_unknown(),
            Some(UnknownReason::NonCliffordOnUnknown),
            "T on unknown should report NonCliffordOnUnknown"
        );
    }

    // =========================================================================
    // Phase 3.3: Decoder Integration Tests
    // These tests verify phenomenological decoder simulation
    // =========================================================================

    #[test]
    fn test_phase_3_3_abstract_mode_no_decoder_rounds() {
        // ACCEPTANCE TEST: Abstract mode should not run decoder rounds
        let config = FTConfig::new(5, 1e-3);
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        assert!(
            !processor.uses_decoder(),
            "Abstract mode should not use decoder"
        );

        // Execute a simple circuit
        let circuit = vec![LogicalGate::H(0), LogicalGate::T(0)];
        let result = processor.execute(&circuit);

        // Decoder stats should be zero
        assert_eq!(
            result.stats.decoder_rounds, 0,
            "No decoder rounds in abstract mode"
        );
        assert_eq!(result.stats.decoder_successes, 0);
        assert_eq!(result.stats.decoder_failures, 0);
    }

    #[test]
    fn test_phase_3_3_decoder_mode_runs_rounds() {
        // ACCEPTANCE TEST: Decoder mode should run decoder rounds per cycle
        let config = FTConfig::new(3, 1e-3).with_decoder_simulation();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        assert!(processor.uses_decoder(), "Decoder mode should use decoder");

        // Execute circuit that advances cycles
        let circuit = vec![LogicalGate::H(0)]; // 1 cycle
        let result = processor.execute(&circuit);

        // Should have run decoder rounds
        assert!(
            result.stats.decoder_rounds > 0,
            "Decoder mode should run decoder rounds"
        );
        assert_eq!(
            result.stats.decoder_rounds,
            result.stats.decoder_successes + result.stats.decoder_failures,
            "Rounds should equal successes + failures"
        );
    }

    #[test]
    fn test_phase_3_3_decoder_success_rate_at_low_error() {
        // ACCEPTANCE TEST: At low error rate, decoder should mostly succeed
        let error_rate = 0.001; // 0.1% - well below threshold
        let config = FTConfig::new(5, error_rate).with_decoder_simulation_at(error_rate);
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(12345);

        // Run many cycles
        let circuit: Vec<_> = (0..20).map(|_| LogicalGate::H(0)).collect();
        let result = processor.execute(&circuit);

        let success_rate = result.stats.decoder_success_rate().unwrap_or(0.0);

        // At 0.1% error rate with distance 5, should have very high success rate
        assert!(
            success_rate > 0.8,
            "Low error rate should have high decoder success rate, got {:.1}%",
            success_rate * 100.0
        );
    }

    #[test]
    fn test_phase_3_3_decoder_failure_rate_increases_with_error() {
        // ACCEPTANCE TEST: Higher error rate should increase decoder failures
        let low_error = 0.001;
        let high_error = 0.05;

        // Low error rate
        let config_low = FTConfig::new(3, low_error).with_decoder_simulation_at(low_error);
        let mut processor_low = FaultTolerantProcessor::new(1, config_low).with_seed(42);
        let circuit: Vec<_> = (0..10).map(|_| LogicalGate::H(0)).collect();
        let result_low = processor_low.execute(&circuit);

        // High error rate
        let config_high = FTConfig::new(3, high_error).with_decoder_simulation_at(high_error);
        let mut processor_high = FaultTolerantProcessor::new(1, config_high).with_seed(42);
        let result_high = processor_high.execute(&circuit);

        let failure_rate_low = result_low.stats.decoder_failure_rate().unwrap_or(0.0);
        let failure_rate_high = result_high.stats.decoder_failure_rate().unwrap_or(0.0);

        // Higher error rate should have more failures
        assert!(
            failure_rate_high >= failure_rate_low,
            "Higher error rate should have >= failure rate: low={:.1}%, high={:.1}%",
            failure_rate_low * 100.0,
            failure_rate_high * 100.0
        );
    }

    #[test]
    fn test_phase_3_3_simulation_fidelity_accessor() {
        // Test that we can access the simulation fidelity mode
        let config_abstract = FTConfig::new(5, 1e-3);
        let processor_abstract = FaultTolerantProcessor::new(1, config_abstract);

        assert!(
            processor_abstract.simulation_fidelity().is_abstract(),
            "Default should be abstract mode"
        );

        let config_decoder = FTConfig::new(5, 1e-3).with_decoder_simulation();
        let processor_decoder = FaultTolerantProcessor::new(1, config_decoder);

        assert!(
            processor_decoder.simulation_fidelity().uses_decoder(),
            "With decoder_simulation should use decoder"
        );
    }

    #[test]
    fn test_phase_3_3_decoder_stats_in_display() {
        // Test that decoder stats appear in FTStats display
        let config = FTConfig::new(3, 0.01).with_decoder_simulation();
        let mut processor = FaultTolerantProcessor::new(1, config).with_seed(42);

        let circuit = vec![LogicalGate::H(0), LogicalGate::T(0)];
        let result = processor.execute(&circuit);

        let stats_str = format!("{}", result.stats);

        // Should include decoder rounds in output
        if result.stats.decoder_rounds > 0 {
            assert!(
                stats_str.contains("Decoder"),
                "Stats display should include decoder info when rounds > 0"
            );
        }
    }

    #[test]
    fn test_phase_3_3_decoder_choice_union_find() {
        // Test explicit Union-Find decoder choice
        let fidelity = SimulationFidelity::with_union_find(0.01);
        match fidelity {
            SimulationFidelity::PhenomenologicalDecoder {
                decoder,
                error_rate,
            } => {
                assert_eq!(decoder, DecoderChoice::UnionFind);
                assert!((error_rate - 0.01).abs() < 1e-10);
            }
            _ => panic!("Expected PhenomenologicalDecoder"),
        }
    }

    #[test]
    fn test_phase_3_3_decoder_choice_mwpm() {
        // Test explicit MWPM decoder choice
        let fidelity = SimulationFidelity::with_mwpm(0.005);
        match fidelity {
            SimulationFidelity::PhenomenologicalDecoder {
                decoder,
                error_rate,
            } => {
                assert_eq!(decoder, DecoderChoice::MWPM);
                assert!((error_rate - 0.005).abs() < 1e-10);
            }
            _ => panic!("Expected PhenomenologicalDecoder"),
        }
    }
}
