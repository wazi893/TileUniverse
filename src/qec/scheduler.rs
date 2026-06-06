//! Logical Circuit Scheduler
//!
//! Compiles logical gates into timed operations measured in surface code cycles.
//! "Surface code is spacetime, not just gates."
//!
//! # Cycle Costs
//!
//! Different operations have different spacetime costs:
//! - Single-qubit Clifford: ~1 cycle
//! - Transversal two-qubit: ~2 cycles
//! - Lattice surgery CNOT: ~d cycles (distance-dependent)
//! - T-gate injection: ~d cycles (includes measurement and correction)
//!
//! # Scheduling Strategy
//!
//! The scheduler:
//! 1. Assigns cycle costs to each gate
//! 2. Identifies parallelism opportunities
//! 3. Handles dependencies (gates on same qubit must be sequential)
//! 4. Inserts implicit QEC rounds as needed

use super::ft_types::{DemandBin, LogicalGate, TimedOp};
use std::fmt;

// ============================================================================
// Cycle Costs Configuration
// ============================================================================

/// Cycle costs for logical operations.
///
/// These are parameterized because costs depend on:
/// - Implementation choice (transversal vs lattice surgery)
/// - Code distance
/// - Hardware constraints
#[derive(Clone, Debug)]
pub struct CycleCosts {
    /// Single-qubit Clifford gate (X, Y, Z, H, S)
    pub single_qubit_clifford: usize,

    /// Two-qubit Clifford gate using transversal implementation
    pub two_qubit_transversal: usize,

    /// Two-qubit gate using lattice surgery (typically CNOT)
    pub lattice_surgery: usize,

    /// T-gate injection (includes ancilla prep, measurement, correction)
    pub t_injection: usize,

    /// Toffoli gate (may use different decomposition)
    pub toffoli: usize,

    /// Single QEC round (syndrome extraction + decode)
    pub qec_round: usize,

    /// Measurement operation
    pub measurement: usize,

    /// State preparation (reset + initialize)
    pub preparation: usize,

    /// Idle cycle (memory, no operation)
    pub idle: usize,
}

impl CycleCosts {
    /// Default costs for typical surface code implementation
    ///
    /// These are rough estimates suitable for resource estimation.
    /// Actual costs depend on specific hardware and layout.
    pub fn default_surface_code() -> Self {
        Self {
            single_qubit_clifford: 1,
            two_qubit_transversal: 2,
            lattice_surgery: 10, // Typically ~d cycles
            t_injection: 5,      // Injection + measurement + correction
            toffoli: 35,         // ~7 T-gates × 5 cycles each
            qec_round: 1,
            measurement: 2,
            preparation: 2,
            idle: 1,
        }
    }

    /// Costs scaled by code distance
    ///
    /// Some operations (like lattice surgery) scale with distance.
    pub fn scaled_by_distance(distance: usize) -> Self {
        Self {
            single_qubit_clifford: 1,
            two_qubit_transversal: 2,
            lattice_surgery: distance, // Scales with d
            t_injection: distance,     // Scales with d
            toffoli: 7 * distance,     // 7 T-gates
            qec_round: 1,
            measurement: distance / 2 + 1,
            preparation: distance / 2 + 1,
            idle: 1,
        }
    }

    /// Optimistic costs (lower bound for estimation)
    pub fn optimistic() -> Self {
        Self {
            single_qubit_clifford: 1,
            two_qubit_transversal: 1,
            lattice_surgery: 5,
            t_injection: 3,
            toffoli: 21, // 7 × 3
            qec_round: 1,
            measurement: 1,
            preparation: 1,
            idle: 1,
        }
    }

    /// Pessimistic costs (upper bound for estimation)
    pub fn pessimistic() -> Self {
        Self {
            single_qubit_clifford: 2,
            two_qubit_transversal: 4,
            lattice_surgery: 20,
            t_injection: 10,
            toffoli: 70, // 7 × 10
            qec_round: 2,
            measurement: 4,
            preparation: 4,
            idle: 1,
        }
    }

    /// Get cycle cost for a specific gate
    pub fn cost_of(&self, gate: &LogicalGate) -> usize {
        match gate {
            // Single-qubit Clifford
            LogicalGate::X(_)
            | LogicalGate::Y(_)
            | LogicalGate::Z(_)
            | LogicalGate::H(_)
            | LogicalGate::S(_)
            | LogicalGate::Sdg(_) => self.single_qubit_clifford,

            // T-gates
            LogicalGate::T(_) | LogicalGate::Tdg(_) => self.t_injection,

            // Two-qubit gates (assume lattice surgery for CNOT)
            LogicalGate::CNOT(_, _) => self.lattice_surgery,
            LogicalGate::CZ(_, _) => self.lattice_surgery,
            LogicalGate::SWAP(_, _) => 3 * self.lattice_surgery, // 3 CNOTs

            // Multi-qubit
            LogicalGate::Toffoli(_, _, _) | LogicalGate::CCZ(_, _, _) => self.toffoli,

            // Control
            LogicalGate::QECRound => self.qec_round,
            LogicalGate::Idle(n) => n * self.idle,

            // Measurement and preparation
            LogicalGate::MeasureZ(_) | LogicalGate::MeasureX(_) => self.measurement,
            LogicalGate::Reset(_) | LogicalGate::PreparePlus(_) => self.preparation,
        }
    }
}

impl Default for CycleCosts {
    fn default() -> Self {
        Self::default_surface_code()
    }
}

impl fmt::Display for CycleCosts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cycle Costs:")?;
        writeln!(f, "  1Q Clifford:    {} cycles", self.single_qubit_clifford)?;
        writeln!(f, "  2Q Transversal: {} cycles", self.two_qubit_transversal)?;
        writeln!(f, "  Lattice Surgery:{} cycles", self.lattice_surgery)?;
        writeln!(f, "  T Injection:    {} cycles", self.t_injection)?;
        writeln!(f, "  Toffoli:        {} cycles", self.toffoli)?;
        writeln!(f, "  QEC Round:      {} cycles", self.qec_round)?;
        Ok(())
    }
}

// ============================================================================
// Scheduler
// ============================================================================

/// Logical circuit scheduler.
///
/// Compiles a sequence of logical gates into timed operations,
/// respecting dependencies and enabling parallelism analysis.
#[derive(Clone, Debug)]
pub struct Scheduler {
    /// Cycle costs configuration
    costs: CycleCosts,
    /// Whether to insert QEC rounds automatically
    auto_qec: bool,
    /// Maximum gates between QEC rounds (if auto_qec is true)
    qec_interval: usize,
}

impl Scheduler {
    /// Create scheduler with default costs
    pub fn new() -> Self {
        Self {
            costs: CycleCosts::default(),
            auto_qec: false,
            qec_interval: 10,
        }
    }

    /// Create scheduler with custom costs
    pub fn with_costs(costs: CycleCosts) -> Self {
        Self {
            costs,
            auto_qec: false,
            qec_interval: 10,
        }
    }

    /// Enable automatic QEC round insertion
    pub fn with_auto_qec(mut self, interval: usize) -> Self {
        self.auto_qec = true;
        self.qec_interval = interval;
        self
    }

    /// Compile circuit into timed operations (sequential scheduling)
    ///
    /// This is the simplest scheduling: each gate starts after the previous ends.
    /// No parallelism is exploited.
    pub fn compile_sequential(&self, circuit: &[LogicalGate]) -> Vec<TimedOp> {
        let mut ops = Vec::with_capacity(circuit.len());
        let mut current_cycle = 0;
        let mut gates_since_qec = 0;

        for gate in circuit {
            let duration = self.costs.cost_of(gate);
            ops.push(TimedOp::new(gate.clone(), current_cycle, duration));
            current_cycle += duration;
            gates_since_qec += 1;

            // Insert QEC round if needed
            if self.auto_qec && gates_since_qec >= self.qec_interval {
                let qec_duration = self.costs.qec_round;
                ops.push(TimedOp::new(
                    LogicalGate::QECRound,
                    current_cycle,
                    qec_duration,
                ));
                current_cycle += qec_duration;
                gates_since_qec = 0;
            }
        }

        ops
    }

    /// Compile circuit with parallelism (ASAP scheduling)
    ///
    /// Gates are scheduled as-soon-as-possible while respecting
    /// qubit dependencies. Independent gates can run in parallel.
    pub fn compile(&self, circuit: &[LogicalGate]) -> Vec<TimedOp> {
        if circuit.is_empty() {
            return Vec::new();
        }

        let mut ops = Vec::with_capacity(circuit.len());

        // Track when each qubit becomes available
        let max_qubit = circuit.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);
        let mut qubit_available_at = vec![0usize; max_qubit + 1];

        let mut gates_since_qec = 0;

        for gate in circuit {
            let qubits = gate.qubits();
            let duration = self.costs.cost_of(gate);

            // Find earliest start time (when all involved qubits are free)
            let start_cycle = if qubits.is_empty() {
                // Global operations (QECRound, Idle) wait for everything
                *qubit_available_at.iter().max().unwrap_or(&0)
            } else {
                qubits.iter().map(|&q| qubit_available_at[q]).max().unwrap()
            };

            ops.push(TimedOp::new(gate.clone(), start_cycle, duration));

            // Update availability
            let end_cycle = start_cycle + duration;
            if qubits.is_empty() {
                // Global operation blocks all qubits
                for t in &mut qubit_available_at {
                    *t = end_cycle;
                }
            } else {
                for &q in &qubits {
                    qubit_available_at[q] = end_cycle;
                }
            }

            // Track QEC intervals
            if matches!(gate, LogicalGate::QECRound) {
                gates_since_qec = 0;
            } else {
                gates_since_qec += 1;
            }

            // Auto-insert QEC if needed
            if self.auto_qec && gates_since_qec >= self.qec_interval {
                let qec_start = *qubit_available_at.iter().max().unwrap_or(&0);
                let qec_duration = self.costs.qec_round;
                ops.push(TimedOp::new(LogicalGate::QECRound, qec_start, qec_duration));
                let qec_end = qec_start + qec_duration;
                for t in &mut qubit_available_at {
                    *t = qec_end;
                }
                gates_since_qec = 0;
            }
        }

        ops
    }

    /// Get total cycles for a circuit (with parallelism)
    pub fn total_cycles(&self, circuit: &[LogicalGate]) -> usize {
        let ops = self.compile(circuit);
        ops.iter().map(|op| op.end_cycle()).max().unwrap_or(0)
    }

    /// Get total cycles for sequential execution (no parallelism)
    pub fn total_cycles_sequential(&self, circuit: &[LogicalGate]) -> usize {
        circuit.iter().map(|g| self.costs.cost_of(g)).sum()
    }

    /// Analyze circuit parallelism
    pub fn analyze(&self, circuit: &[LogicalGate]) -> ScheduleAnalysis {
        let ops = self.compile(circuit);
        let seq_cycles = self.total_cycles_sequential(circuit);
        let par_cycles = ops.iter().map(|op| op.end_cycle()).max().unwrap_or(0);

        // Count concurrent operations at each cycle
        let mut max_concurrent = 0;
        if !ops.is_empty() {
            for cycle in 0..par_cycles {
                let concurrent = ops
                    .iter()
                    .filter(|op| op.start_cycle <= cycle && op.end_cycle() > cycle)
                    .count();
                max_concurrent = max_concurrent.max(concurrent);
            }
        }

        // Count gate types
        let t_count = circuit.iter().map(|g| g.t_count()).sum();
        let clifford_count = circuit.iter().filter(|g| g.is_clifford()).count();
        let two_qubit_count = circuit
            .iter()
            .filter(|g| matches!(g, LogicalGate::CNOT(_, _) | LogicalGate::CZ(_, _)))
            .count();

        ScheduleAnalysis {
            total_gates: circuit.len(),
            total_cycles_sequential: seq_cycles,
            total_cycles_parallel: par_cycles,
            parallelism_factor: if par_cycles > 0 {
                seq_cycles as f64 / par_cycles as f64
            } else {
                1.0
            },
            max_concurrent_ops: max_concurrent,
            t_count,
            clifford_count,
            two_qubit_count,
        }
    }

    /// Get reference to cycle costs
    pub fn costs(&self) -> &CycleCosts {
        &self.costs
    }

    // ========================================================================
    // Phase 3.0: Parallel Schedule with Demand Timeline
    // ========================================================================

    /// Schedule circuit with full parallel analysis and demand timeline.
    ///
    /// This is the key Phase 3.0 output: transforms "counts stuff" into
    /// "predicts execution" by computing:
    /// - Makespan (critical path under resource constraints)
    /// - Per-gate start times
    /// - Magic state demand timeline for factory analysis
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scheduler = Scheduler::new();
    /// let schedule = scheduler.schedule_parallel(&circuit);
    /// println!("Makespan: {} cycles", schedule.makespan);
    /// println!("Min throughput: {:.3} states/cycle", schedule.min_throughput_required());
    /// ```
    pub fn schedule_parallel(&self, circuit: &[LogicalGate]) -> ParallelSchedule {
        if circuit.is_empty() {
            return ParallelSchedule {
                gates: Vec::new(),
                makespan: 0,
                t_depth: 0,
                cnot_depth: 0,
                peak_width: 0,
                demand_timeline: Vec::new(),
            };
        }

        // Track per-qubit availability
        let max_qubit = circuit.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);
        let mut qubit_available_at = vec![0usize; max_qubit + 1];

        // Track T-depth and CNOT-depth per qubit
        let mut qubit_t_depth = vec![0usize; max_qubit + 1];
        let mut qubit_cnot_depth = vec![0usize; max_qubit + 1];

        // Schedule each gate
        let mut scheduled_gates = Vec::with_capacity(circuit.len());
        let mut makespan = 0;

        for (gate_index, gate) in circuit.iter().enumerate() {
            let qubits = gate.qubits();
            let duration = self.costs.cost_of(gate);
            let magic_states = gate.t_count();

            // Find earliest start time
            let start_cycle = if qubits.is_empty() {
                // Global operations wait for everything
                *qubit_available_at.iter().max().unwrap_or(&0)
            } else {
                qubits.iter().map(|&q| qubit_available_at[q]).max().unwrap()
            };

            let end_cycle = start_cycle + duration;
            makespan = makespan.max(end_cycle);

            // Update qubit availability
            if qubits.is_empty() {
                for t in &mut qubit_available_at {
                    *t = end_cycle;
                }
            } else {
                for &q in &qubits {
                    qubit_available_at[q] = end_cycle;
                }
            }

            // Track T-depth
            if gate.requires_magic_state() && !qubits.is_empty() {
                let max_t = qubits.iter().map(|&q| qubit_t_depth[q]).max().unwrap();
                for &q in &qubits {
                    qubit_t_depth[q] = max_t + 1;
                }
            } else if qubits.len() >= 2 {
                // Two-qubit gates propagate T-depth
                let max_t = qubits.iter().map(|&q| qubit_t_depth[q]).max().unwrap();
                for &q in &qubits {
                    qubit_t_depth[q] = max_t;
                }
            }

            // Track CNOT-depth
            if matches!(gate, LogicalGate::CNOT(_, _) | LogicalGate::CZ(_, _)) && !qubits.is_empty()
            {
                let max_c = qubits.iter().map(|&q| qubit_cnot_depth[q]).max().unwrap();
                for &q in &qubits {
                    qubit_cnot_depth[q] = max_c + 1;
                }
            } else if qubits.len() >= 2 {
                let max_c = qubits.iter().map(|&q| qubit_cnot_depth[q]).max().unwrap();
                for &q in &qubits {
                    qubit_cnot_depth[q] = max_c;
                }
            }

            scheduled_gates.push(ScheduledGate {
                gate_index,
                gate: gate.clone(),
                start_cycle,
                duration,
                qubits,
                magic_states,
            });
        }

        let t_depth = qubit_t_depth.into_iter().max().unwrap_or(0);
        let cnot_depth = qubit_cnot_depth.into_iter().max().unwrap_or(0);

        // Compute peak width (max concurrent qubits)
        let peak_width = self.compute_peak_width(&scheduled_gates, makespan);

        // Build demand timeline
        let demand_timeline = self.build_demand_timeline(&scheduled_gates, makespan);

        ParallelSchedule {
            gates: scheduled_gates,
            makespan,
            t_depth,
            cnot_depth,
            peak_width,
            demand_timeline,
        }
    }

    /// Compute peak width (maximum qubits active at any cycle).
    fn compute_peak_width(&self, gates: &[ScheduledGate], makespan: usize) -> usize {
        if gates.is_empty() || makespan == 0 {
            return 0;
        }

        let mut peak = 0;
        // Sample at regular intervals for efficiency
        let step = (makespan / 100).max(1);
        for cycle in (0..makespan).step_by(step) {
            let active: std::collections::HashSet<usize> = gates
                .iter()
                .filter(|g| g.start_cycle <= cycle && g.end_cycle() > cycle)
                .flat_map(|g| g.qubits.iter().copied())
                .collect();
            peak = peak.max(active.len());
        }
        peak
    }

    /// Build demand timeline binned by T-depth layers.
    ///
    /// Each bin represents a T-layer and contains the magic states needed.
    fn build_demand_timeline(&self, gates: &[ScheduledGate], makespan: usize) -> Vec<DemandBin> {
        if gates.is_empty() || makespan == 0 {
            return Vec::new();
        }

        // Find all T-gate start times
        let t_gates: Vec<_> = gates.iter().filter(|g| g.needs_magic()).collect();
        if t_gates.is_empty() {
            return Vec::new();
        }

        // Group by start cycle into bins
        // Use natural T-depth layers: gates with same start cycle are in same bin
        let mut bins_map: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new();

        for gate in &t_gates {
            let entry = bins_map.entry(gate.start_cycle).or_insert((0, 0));
            entry.0 += gate.magic_states; // magic_states
            entry.1 += 1; // t_gates count
        }

        // Convert to DemandBin, merging adjacent cycles into layers
        let mut bins = Vec::new();
        let mut current_bin: Option<DemandBin> = None;
        let t_cost = self.costs.t_injection;

        for (&start, &(magic, t_count)) in &bins_map {
            let end = start + t_cost;

            if let Some(ref mut bin) = current_bin {
                // Merge if overlapping or adjacent
                if start <= bin.end_cycle + 1 {
                    bin.end_cycle = bin.end_cycle.max(end);
                    bin.magic_states += magic;
                    bin.t_gates += t_count;
                } else {
                    // Gap - push current and start new
                    bins.push(bin.clone());
                    current_bin = Some(DemandBin {
                        start_cycle: start,
                        end_cycle: end,
                        magic_states: magic,
                        t_gates: t_count,
                    });
                }
            } else {
                current_bin = Some(DemandBin {
                    start_cycle: start,
                    end_cycle: end,
                    magic_states: magic,
                    t_gates: t_count,
                });
            }
        }

        if let Some(bin) = current_bin {
            bins.push(bin);
        }

        bins
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Schedule Analysis
// ============================================================================

/// Analysis of a scheduled circuit.
#[derive(Clone, Debug)]
pub struct ScheduleAnalysis {
    /// Total number of gates
    pub total_gates: usize,
    /// Cycles if executed sequentially
    pub total_cycles_sequential: usize,
    /// Cycles with parallelism
    pub total_cycles_parallel: usize,
    /// Speedup from parallelism
    pub parallelism_factor: f64,
    /// Maximum operations running concurrently
    pub max_concurrent_ops: usize,
    /// Total T-gate count
    pub t_count: usize,
    /// Total Clifford gate count
    pub clifford_count: usize,
    /// Total two-qubit gate count
    pub two_qubit_count: usize,
}

impl fmt::Display for ScheduleAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Schedule Analysis:")?;
        writeln!(f, "  Total gates:        {}", self.total_gates)?;
        writeln!(f, "  Sequential cycles:  {}", self.total_cycles_sequential)?;
        writeln!(f, "  Parallel cycles:    {}", self.total_cycles_parallel)?;
        writeln!(f, "  Parallelism factor: {:.2}×", self.parallelism_factor)?;
        writeln!(f, "  Max concurrent:     {}", self.max_concurrent_ops)?;
        writeln!(f, "  T-count:            {}", self.t_count)?;
        writeln!(f, "  Clifford gates:     {}", self.clifford_count)?;
        writeln!(f, "  Two-qubit gates:    {}", self.two_qubit_count)?;
        Ok(())
    }
}

// ============================================================================
// Parallel Schedule (Phase 3.0)
// ============================================================================

/// A scheduled gate with timing and resource requirements.
///
/// This is the output unit of parallel scheduling - each gate has an assigned
/// start cycle and tracks whether it requires magic states.
#[derive(Clone, Debug)]
pub struct ScheduledGate {
    /// Index in original circuit
    pub gate_index: usize,
    /// The logical gate
    pub gate: LogicalGate,
    /// Start cycle (determined by dependencies)
    pub start_cycle: usize,
    /// Duration in cycles
    pub duration: usize,
    /// Qubits involved (cached for convenience)
    pub qubits: Vec<usize>,
    /// Number of magic states required (0 for Clifford)
    pub magic_states: usize,
}

impl ScheduledGate {
    /// End cycle (exclusive)
    pub fn end_cycle(&self) -> usize {
        self.start_cycle + self.duration
    }

    /// Whether this gate requires magic states
    pub fn needs_magic(&self) -> bool {
        self.magic_states > 0
    }
}

/// Complete parallel schedule with demand analysis.
///
/// This is the main output of Phase 3.0 - transforms "counts stuff" into
/// "predicts execution" by providing:
/// - Makespan under resource constraints
/// - Gate start times
/// - Magic state demand timeline
#[derive(Clone, Debug)]
pub struct ParallelSchedule {
    /// All scheduled gates with timing
    pub gates: Vec<ScheduledGate>,
    /// Total makespan (critical path)
    pub makespan: usize,
    /// T-depth (layers of T-gates)
    pub t_depth: usize,
    /// CNOT depth (layers of CNOT gates)
    pub cnot_depth: usize,
    /// Maximum qubits active concurrently
    pub peak_width: usize,
    /// Demand bins for factory analysis
    pub demand_timeline: Vec<DemandBin>,
}

impl ParallelSchedule {
    /// Get all T-gates scheduled in a specific cycle range.
    pub fn t_gates_in_range(&self, start: usize, end: usize) -> Vec<&ScheduledGate> {
        self.gates
            .iter()
            .filter(|g| g.needs_magic() && g.start_cycle >= start && g.start_cycle < end)
            .collect()
    }

    /// Get total magic states consumed.
    pub fn total_magic_states(&self) -> usize {
        self.gates.iter().map(|g| g.magic_states).sum()
    }

    /// Get peak T-demand per cycle.
    pub fn peak_t_demand(&self) -> usize {
        self.demand_timeline
            .iter()
            .map(|b| b.magic_states)
            .max()
            .unwrap_or(0)
    }

    /// Minimum factory throughput to avoid stalls (states/cycle).
    pub fn min_throughput_required(&self) -> f64 {
        if self.makespan == 0 {
            return 0.0;
        }
        // Need to deliver total_magic_states over makespan cycles
        // But also handle bursts - peak demand over any window
        let total = self.total_magic_states() as f64;
        let makespan = self.makespan as f64;

        // Average throughput
        let avg = total / makespan;

        // Check burst windows - find maximum demand density
        let mut max_density = avg;
        for bin in &self.demand_timeline {
            let duration = bin.duration().max(1) as f64;
            let density = bin.magic_states as f64 / duration;
            max_density = max_density.max(density);
        }

        max_density
    }
}

impl fmt::Display for ParallelSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Parallel Schedule:")?;
        writeln!(f, "  Gates:           {}", self.gates.len())?;
        writeln!(f, "  Makespan:        {} cycles", self.makespan)?;
        writeln!(f, "  T-depth:         {}", self.t_depth)?;
        writeln!(f, "  CNOT-depth:      {}", self.cnot_depth)?;
        writeln!(f, "  Peak width:      {} qubits", self.peak_width)?;
        writeln!(f, "  Magic states:    {}", self.total_magic_states())?;
        writeln!(f, "  Demand bins:     {}", self.demand_timeline.len())?;
        writeln!(
            f,
            "  Min throughput:  {:.3} states/cycle",
            self.min_throughput_required()
        )?;
        Ok(())
    }
}

// ============================================================================
// T-Depth Analysis
// ============================================================================

/// Compute T-depth of a circuit.
///
/// T-depth is the number of sequential T-gate layers (T-gates that cannot
/// be parallelized). This is often the bottleneck for fault-tolerant circuits.
pub fn compute_t_depth(circuit: &[LogicalGate]) -> usize {
    if circuit.is_empty() {
        return 0;
    }

    // Track T-depth for each qubit
    let max_qubit = circuit.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);
    let mut qubit_t_depth = vec![0usize; max_qubit + 1];

    for gate in circuit {
        let qubits = gate.qubits();
        if qubits.is_empty() {
            continue;
        }

        if gate.requires_magic_state() {
            // T-gate: increment depth on all involved qubits
            let max_depth = qubits.iter().map(|&q| qubit_t_depth[q]).max().unwrap();
            let new_depth = max_depth + 1;
            for &q in &qubits {
                qubit_t_depth[q] = new_depth;
            }
        } else if matches!(
            gate,
            LogicalGate::CNOT(_, _) | LogicalGate::CZ(_, _) | LogicalGate::SWAP(_, _)
        ) {
            // Two-qubit Clifford: propagate max depth to both qubits
            let max_depth = qubits.iter().map(|&q| qubit_t_depth[q]).max().unwrap();
            for &q in &qubits {
                qubit_t_depth[q] = max_depth;
            }
        }
        // Single-qubit Cliffords don't affect T-depth
    }

    qubit_t_depth.into_iter().max().unwrap_or(0)
}

/// Compute detailed T-gate layer structure.
///
/// Returns a list of layers, where each layer contains T-gates that can
/// be executed in parallel.
pub fn compute_t_layers(circuit: &[LogicalGate]) -> Vec<Vec<usize>> {
    if circuit.is_empty() {
        return Vec::new();
    }

    let max_qubit = circuit.iter().flat_map(|g| g.qubits()).max().unwrap_or(0);

    // Track which layer each qubit is at
    let mut qubit_layer = vec![0usize; max_qubit + 1];
    let mut layers: Vec<Vec<usize>> = Vec::new();

    for (gate_idx, gate) in circuit.iter().enumerate() {
        if !gate.requires_magic_state() {
            // Non-T gates: propagate max layer for two-qubit gates
            let qubits = gate.qubits();
            if qubits.len() >= 2 {
                let max_layer = qubits.iter().map(|&q| qubit_layer[q]).max().unwrap();
                for &q in &qubits {
                    qubit_layer[q] = max_layer;
                }
            }
            continue;
        }

        // T-gate: assign to next layer
        let qubits = gate.qubits();
        let layer = qubits.iter().map(|&q| qubit_layer[q]).max().unwrap();

        // Ensure layer exists
        while layers.len() <= layer {
            layers.push(Vec::new());
        }

        layers[layer].push(gate_idx);

        // Update qubit layers
        for &q in &qubits {
            qubit_layer[q] = layer + 1;
        }
    }

    layers
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_costs() {
        let costs = CycleCosts::default_surface_code();
        assert!(costs.t_injection > costs.single_qubit_clifford);
        assert!(costs.toffoli > costs.t_injection);
    }

    #[test]
    fn test_sequential_scheduling() {
        let scheduler = Scheduler::new();
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::T(0),
            LogicalGate::CNOT(0, 1),
        ];

        let ops = scheduler.compile_sequential(&circuit);
        assert_eq!(ops.len(), 3);

        // Check sequential timing
        assert_eq!(ops[0].start_cycle, 0);
        assert_eq!(ops[1].start_cycle, ops[0].end_cycle());
        assert_eq!(ops[2].start_cycle, ops[1].end_cycle());
    }

    #[test]
    fn test_parallel_scheduling() {
        let scheduler = Scheduler::new();

        // Two independent single-qubit gates can run in parallel
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(1),
            LogicalGate::CNOT(0, 1),
        ];

        let ops = scheduler.compile(&circuit);
        assert_eq!(ops.len(), 3);

        // H(0) and H(1) should start at same time
        assert_eq!(ops[0].start_cycle, 0);
        assert_eq!(ops[1].start_cycle, 0);

        // CNOT should start after both H gates
        assert!(ops[2].start_cycle >= ops[0].end_cycle());
        assert!(ops[2].start_cycle >= ops[1].end_cycle());
    }

    #[test]
    fn test_parallelism_analysis() {
        let scheduler = Scheduler::new();
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(1),
            LogicalGate::H(2),
            LogicalGate::CNOT(0, 1),
        ];

        let analysis = scheduler.analyze(&circuit);
        assert_eq!(analysis.total_gates, 4);
        assert!(analysis.parallelism_factor > 1.0); // Some parallelism
        assert!(analysis.max_concurrent_ops >= 2); // At least 2 H gates concurrent
    }

    #[test]
    fn test_t_depth() {
        // Sequential T gates on same qubit
        let circuit1 = vec![LogicalGate::T(0), LogicalGate::T(0), LogicalGate::T(0)];
        assert_eq!(compute_t_depth(&circuit1), 3);

        // Parallel T gates on different qubits
        let circuit2 = vec![LogicalGate::T(0), LogicalGate::T(1), LogicalGate::T(2)];
        assert_eq!(compute_t_depth(&circuit2), 1);

        // Mixed
        let circuit3 = vec![
            LogicalGate::T(0),
            LogicalGate::T(1),
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0),
        ];
        assert_eq!(compute_t_depth(&circuit3), 2);
    }

    #[test]
    fn test_t_layers() {
        let circuit = vec![
            LogicalGate::T(0), // Layer 0
            LogicalGate::T(1), // Layer 0
            LogicalGate::CNOT(0, 1),
            LogicalGate::T(0), // Layer 1
            LogicalGate::T(1), // Layer 1
        ];

        let layers = compute_t_layers(&circuit);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].len(), 2); // Two T gates in layer 0
        assert_eq!(layers[1].len(), 2); // Two T gates in layer 1
    }

    #[test]
    fn test_auto_qec() {
        let scheduler = Scheduler::new().with_auto_qec(3);
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(1),
            LogicalGate::H(2),
            LogicalGate::H(3), // Should trigger QEC after this
            LogicalGate::H(4),
        ];

        let ops = scheduler.compile_sequential(&circuit);

        // Should have 5 H gates + at least 1 QEC round
        let qec_count = ops
            .iter()
            .filter(|op| matches!(op.op, LogicalGate::QECRound))
            .count();
        assert!(qec_count >= 1);
    }

    // =========================================================================
    // Phase 3.0: Parallel Schedule Acceptance Tests
    // =========================================================================

    #[test]
    fn test_schedule_parallel_wide_circuit() {
        // ACCEPTANCE TEST 1: 100 independent gates on 100 qubits
        // Should schedule in ~1 gate duration, NOT 100×
        let scheduler = Scheduler::new();
        let circuit: Vec<_> = (0..100).map(|q| LogicalGate::H(q)).collect();

        let schedule = scheduler.schedule_parallel(&circuit);

        // All 100 H gates should run in parallel
        // Makespan should be 1 cycle (single-qubit Clifford cost)
        let single_clifford = scheduler.costs().single_qubit_clifford;
        assert_eq!(
            schedule.makespan, single_clifford,
            "100 independent H gates should complete in {} cycle(s), got {}",
            single_clifford, schedule.makespan
        );

        // Sequential would be 100 × single_clifford
        let sequential = 100 * single_clifford;
        assert!(
            schedule.makespan < sequential,
            "Parallel ({}) should be much less than sequential ({})",
            schedule.makespan,
            sequential
        );
    }

    #[test]
    fn test_schedule_parallel_t_layer_demand_spike() {
        // ACCEPTANCE TEST 2: Wide T-layer shows sharp spike in demand bins
        let scheduler = Scheduler::new();

        // 10 parallel T-gates on different qubits
        let circuit: Vec<_> = (0..10).map(|q| LogicalGate::T(q)).collect();

        let schedule = scheduler.schedule_parallel(&circuit);

        // Should have T-depth = 1 (all parallel)
        assert_eq!(schedule.t_depth, 1, "All T-gates should be in layer 0");

        // Should have exactly one demand bin with all 10 T-gates
        assert_eq!(
            schedule.demand_timeline.len(),
            1,
            "Should have single demand bin for parallel T-layer"
        );
        assert_eq!(
            schedule.demand_timeline[0].magic_states, 10,
            "Demand bin should have all 10 magic states"
        );

        // Peak demand should be 10
        assert_eq!(schedule.peak_t_demand(), 10);
    }

    #[test]
    fn test_schedule_parallel_sequential_t_gates() {
        // T-gates on same qubit must be sequential
        let scheduler = Scheduler::new();
        let circuit = vec![LogicalGate::T(0), LogicalGate::T(0), LogicalGate::T(0)];

        let schedule = scheduler.schedule_parallel(&circuit);

        // T-depth should be 3 (all sequential)
        assert_eq!(schedule.t_depth, 3);

        // Makespan should be 3 × T-injection cost
        let t_cost = scheduler.costs().t_injection;
        assert_eq!(schedule.makespan, 3 * t_cost);

        // Demand timeline should have 3 separate bins (or merged if adjacent)
        // At minimum, total magic states = 3
        let total_magic: usize = schedule
            .demand_timeline
            .iter()
            .map(|b| b.magic_states)
            .sum();
        assert_eq!(total_magic, 3);
    }

    #[test]
    fn test_schedule_parallel_mixed_circuit() {
        // Mix of parallel and sequential gates
        let scheduler = Scheduler::new();
        let circuit = vec![
            // Layer 0: Two parallel H gates
            LogicalGate::H(0),
            LogicalGate::H(1),
            // Layer 1: CNOT depends on both
            LogicalGate::CNOT(0, 1),
            // Layer 2: Parallel T-gates
            LogicalGate::T(0),
            LogicalGate::T(1),
        ];

        let schedule = scheduler.schedule_parallel(&circuit);

        // Check gate timings
        assert_eq!(schedule.gates[0].start_cycle, 0, "H(0) starts at 0");
        assert_eq!(schedule.gates[1].start_cycle, 0, "H(1) starts at 0");

        // CNOT starts after both H gates
        let h_end = scheduler.costs().single_qubit_clifford;
        assert_eq!(
            schedule.gates[2].start_cycle, h_end,
            "CNOT starts after H gates"
        );

        // T-gates start after CNOT
        let cnot_end = h_end + scheduler.costs().lattice_surgery;
        assert_eq!(
            schedule.gates[3].start_cycle, cnot_end,
            "T(0) starts after CNOT"
        );
        assert_eq!(
            schedule.gates[4].start_cycle, cnot_end,
            "T(1) starts after CNOT"
        );

        // T-depth should be 1 (both T-gates are parallel)
        assert_eq!(schedule.t_depth, 1);
    }

    #[test]
    fn test_schedule_parallel_toffoli_demand() {
        // Toffoli gate requires 7 magic states
        let scheduler = Scheduler::new();
        let circuit = vec![LogicalGate::Toffoli(0, 1, 2)];

        let schedule = scheduler.schedule_parallel(&circuit);

        assert_eq!(
            schedule.total_magic_states(),
            7,
            "Toffoli needs 7 magic states"
        );
        assert_eq!(schedule.t_depth, 1, "Toffoli counts as 1 T-layer");
    }

    #[test]
    fn test_schedule_parallel_min_throughput() {
        // Test minimum throughput calculation
        let scheduler = Scheduler::new();

        // 10 T-gates all parallel (instant burst)
        let burst_circuit: Vec<_> = (0..10).map(|q| LogicalGate::T(q)).collect();
        let burst_schedule = scheduler.schedule_parallel(&burst_circuit);

        // 10 T-gates sequential on one qubit (spread over time)
        let seq_circuit: Vec<_> = (0..10).map(|_| LogicalGate::T(0)).collect();
        let seq_schedule = scheduler.schedule_parallel(&seq_circuit);

        // Burst should require higher throughput
        assert!(
            burst_schedule.min_throughput_required() > seq_schedule.min_throughput_required(),
            "Burst ({:.3}) should need more throughput than sequential ({:.3})",
            burst_schedule.min_throughput_required(),
            seq_schedule.min_throughput_required()
        );
    }

    #[test]
    fn test_schedule_parallel_empty_circuit() {
        let scheduler = Scheduler::new();
        let schedule = scheduler.schedule_parallel(&[]);

        assert_eq!(schedule.makespan, 0);
        assert_eq!(schedule.t_depth, 0);
        assert!(schedule.demand_timeline.is_empty());
    }

    #[test]
    fn test_schedule_parallel_cnot_depth() {
        let scheduler = Scheduler::new();
        let circuit = vec![
            LogicalGate::CNOT(0, 1), // Layer 0
            LogicalGate::CNOT(2, 3), // Layer 0 (parallel)
            LogicalGate::CNOT(0, 2), // Layer 1 (depends on both)
        ];

        let schedule = scheduler.schedule_parallel(&circuit);

        // First two CNOTs parallel, third sequential
        assert_eq!(schedule.cnot_depth, 2);
    }
}
