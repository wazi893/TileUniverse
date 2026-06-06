// =============================================================================
// Factory Scheduler - Dynamic Magic State Production Scheduling
// =============================================================================
// Sprint 63.0: Magic State Factory Scheduling
//
// Core scheduling infrastructure for parallel magic state distillation factories.
// Analyzes T-gate dependencies, assigns factories, and generates execution schedules.
//
// Key Components:
// - DependencyGraph: T-gate layering and parallelism analysis
// - Factory: Individual distillation factory abstraction
// - FactoryScheduler: Main scheduling engine
// - Schedule: Complete mapping of T-gates → factories with timeline

use crate::qram::{DistillationProtocol, ToffoliContext};
use crate::quantum::QGate;
use std::collections::{BTreeMap, HashMap, VecDeque};

// =============================================================================
// Dependency Graph
// =============================================================================

/// T-gate dependency graph for circuit analysis
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    /// T-gates organized by layer (layer i must complete before layer i+1)
    pub layers: Vec<Vec<TGateNode>>,

    /// Total T-count
    pub t_count: usize,

    /// T-depth (number of layers)
    pub t_depth: usize,

    /// Average parallelism per layer
    pub avg_parallelism: f64,
}

/// A T-gate node in dependency graph
#[derive(Clone, Debug)]
pub struct TGateNode {
    /// Unique ID within circuit
    pub id: usize,

    /// Qubit this T-gate acts on
    pub qubit: usize,

    /// Type of T-gate
    pub gate_type: TGateType,

    /// T-gate IDs that must complete first
    pub dependencies: Vec<usize>,
}

/// Type of T-gate
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TGateType {
    /// T gate
    T,
    /// T-dagger gate
    Tdg,
    /// Toffoli gate (decomposed into 7 T-gates)
    Toffoli { c1: usize, c2: usize },
}

impl DependencyGraph {
    /// Build dependency graph from circuit
    pub fn from_circuit(circuit: &[QGate]) -> Self {
        let mut layers = vec![];
        let mut current_layer = vec![];
        let mut last_t_on_qubit: HashMap<usize, usize> = HashMap::new();
        let mut t_gate_id = 0;

        for (_gate_idx, gate) in circuit.iter().enumerate() {
            if let Some((qubit, gate_type)) = Self::extract_t_gate(gate) {
                // Check if this T-gate has dependencies
                let mut has_dependency = false;

                if let Some(&_prev_id) = last_t_on_qubit.get(&qubit) {
                    // T gates on same qubit are sequential
                    has_dependency = true;
                }

                // For Toffoli, check both control qubits
                if let TGateType::Toffoli { c1, c2 } = gate_type {
                    if last_t_on_qubit.contains_key(&c1) || last_t_on_qubit.contains_key(&c2) {
                        has_dependency = true;
                    }
                }

                // If has dependency, start new layer
                if has_dependency && !current_layer.is_empty() {
                    layers.push(current_layer);
                    current_layer = vec![];
                }

                let node = TGateNode {
                    id: t_gate_id,
                    qubit,
                    gate_type,
                    dependencies: vec![],
                };

                current_layer.push(node);
                last_t_on_qubit.insert(qubit, t_gate_id);

                // For Toffoli, mark both controls as occupied
                if let TGateType::Toffoli { c1, c2 } = gate_type {
                    last_t_on_qubit.insert(c1, t_gate_id);
                    last_t_on_qubit.insert(c2, t_gate_id);
                }

                t_gate_id += 1;
            }
        }

        // Push final layer
        if !current_layer.is_empty() {
            layers.push(current_layer);
        }

        let t_count = layers.iter().map(|l| l.len()).sum();
        let t_depth = layers.len();
        let avg_parallelism = if t_depth > 0 {
            t_count as f64 / t_depth as f64
        } else {
            0.0
        };

        DependencyGraph {
            layers,
            t_count,
            t_depth,
            avg_parallelism,
        }
    }

    /// Extract T-gate information from QGate
    fn extract_t_gate(gate: &QGate) -> Option<(usize, TGateType)> {
        match gate {
            QGate::T(q) => Some((*q as usize, TGateType::T)),
            QGate::Tdg(q) => Some((*q as usize, TGateType::Tdg)),
            QGate::Toffoli(c1, c2, t) => Some((
                *t as usize,
                TGateType::Toffoli {
                    c1: *c1 as usize,
                    c2: *c2 as usize,
                },
            )),
            _ => None,
        }
    }

    /// Compute critical path length (minimum cycles with unlimited factories)
    pub fn critical_path_length(&self, factory_latency: usize) -> usize {
        self.t_depth * factory_latency
    }

    /// Maximum parallelism (largest layer size)
    pub fn max_parallelism(&self) -> usize {
        self.layers.iter().map(|l| l.len()).max().unwrap_or(1)
    }

    /// Get statistics
    pub fn stats(&self) -> DependencyGraphStats {
        DependencyGraphStats {
            t_count: self.t_count,
            t_depth: self.t_depth,
            avg_parallelism: self.avg_parallelism,
            max_parallelism: self.max_parallelism(),
            layer_distribution: self.layers.iter().map(|l| l.len()).collect(),
        }
    }
}

/// Statistics about dependency graph
#[derive(Clone, Debug)]
pub struct DependencyGraphStats {
    pub t_count: usize,
    pub t_depth: usize,
    pub avg_parallelism: f64,
    pub max_parallelism: usize,
    pub layer_distribution: Vec<usize>,
}

// =============================================================================
// Factory
// =============================================================================

/// A single magic state distillation factory
#[derive(Clone, Debug)]
pub struct Factory {
    /// Factory ID
    pub id: usize,

    /// Distillation protocol used
    pub protocol: DistillationProtocol,

    /// Physical location on device (if hardware-mapped)
    pub location: Option<(usize, usize)>,

    /// Current production queue
    pub queue: VecDeque<MagicStateRequest>,

    /// Cycles until next magic state ready
    pub cycles_until_ready: usize,

    /// Total magic states produced (for tracking)
    pub states_produced: usize,
}

/// Request for magic state production
#[derive(Clone, Debug)]
pub struct MagicStateRequest {
    /// T-gate ID that will consume this state
    pub t_gate_id: usize,

    /// Circuit layer where T-gate appears
    pub layer: usize,

    /// Target error rate for this magic state
    pub target_error: f64,

    /// Context (routing, correction, data)
    pub context: ToffoliContext,
}

impl Factory {
    /// Create new factory with protocol
    pub fn new(id: usize, protocol: DistillationProtocol) -> Self {
        Factory {
            id,
            protocol,
            location: None,
            queue: VecDeque::new(),
            cycles_until_ready: 0,
            states_produced: 0,
        }
    }

    /// Create factory at specific location
    pub fn new_at_location(
        id: usize,
        protocol: DistillationProtocol,
        location: (usize, usize),
    ) -> Self {
        let mut factory = Self::new(id, protocol);
        factory.location = Some(location);
        factory
    }

    /// Add request to queue
    pub fn enqueue(&mut self, request: MagicStateRequest) {
        self.queue.push_back(request);
    }

    /// Check if factory is idle
    pub fn is_idle(&self) -> bool {
        self.cycles_until_ready == 0 && self.queue.is_empty()
    }

    /// Get qubit footprint
    pub fn qubit_count(&self) -> usize {
        self.protocol.qubit_overhead
    }

    /// Reset factory state
    pub fn reset(&mut self) {
        self.queue.clear();
        self.cycles_until_ready = 0;
        self.states_produced = 0;
    }
}

// =============================================================================
// Schedule
// =============================================================================

/// Complete schedule mapping factories to T-gates
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Schedule {
    /// Assignment: T-gate ID → Factory ID
    pub assignments: Vec<usize>,

    /// Production timeline: Cycle → [(Factory, T-gate)]
    pub timeline: BTreeMap<usize, Vec<(usize, usize)>>,

    /// Total cycles to complete all magic state production
    pub total_cycles: usize,

    /// Factory utilization (% time active)
    pub factory_utilization: Vec<f64>,
}

impl Schedule {
    /// Create empty schedule
    pub fn new(n_t_gates: usize, n_factories: usize) -> Self {
        Schedule {
            assignments: vec![0; n_t_gates],
            timeline: BTreeMap::new(),
            total_cycles: 0,
            factory_utilization: vec![0.0; n_factories],
        }
    }

    /// Get factory utilization statistics
    pub fn utilization_stats(&self) -> UtilizationStats {
        let avg =
            self.factory_utilization.iter().sum::<f64>() / self.factory_utilization.len() as f64;
        let min = self
            .factory_utilization
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .factory_utilization
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        UtilizationStats {
            average: avg,
            minimum: min,
            maximum: max,
            per_factory: self.factory_utilization.clone(),
        }
    }

    /// Find bottleneck factory (highest load or lowest utilization)
    pub fn bottleneck_factory(&self) -> Option<usize> {
        self.factory_utilization
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }
}

/// Factory utilization statistics
#[derive(Clone, Debug)]
pub struct UtilizationStats {
    pub average: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub per_factory: Vec<f64>,
}

// =============================================================================
// Factory Scheduler
// =============================================================================

/// Dynamic magic state factory scheduler
#[derive(Clone)]
pub struct FactoryScheduler {
    /// Pool of available factories
    factories: Vec<Factory>,

    /// T-gate dependency graph from circuit
    pub(crate) dependency_graph: Option<DependencyGraph>,

    /// Current schedule
    schedule: Option<Schedule>,
}

impl FactoryScheduler {
    /// Create scheduler with N factories using given protocol
    pub fn new(num_factories: usize, protocol: DistillationProtocol) -> Self {
        let factories = (0..num_factories)
            .map(|id| Factory::new(id, protocol.clone()))
            .collect();

        FactoryScheduler {
            factories,
            dependency_graph: None,
            schedule: None,
        }
    }

    /// Create scheduler with existing factories
    pub fn with_factories(factories: Vec<Factory>) -> Self {
        FactoryScheduler {
            factories,
            dependency_graph: None,
            schedule: None,
        }
    }

    /// Build dependency graph from circuit
    pub fn analyze_circuit(&mut self, circuit: &[QGate]) -> &DependencyGraph {
        let dep_graph = DependencyGraph::from_circuit(circuit);
        self.dependency_graph = Some(dep_graph);
        self.dependency_graph.as_ref().unwrap()
    }

    /// Get dependency graph (must call analyze_circuit first)
    pub fn dependency_graph(&self) -> Option<&DependencyGraph> {
        self.dependency_graph.as_ref()
    }

    /// Set the schedule (used by optimizers that generate schedules externally)
    pub fn set_schedule(&mut self, schedule: Schedule) {
        self.schedule = Some(schedule);
    }

    /// Generate schedule using greedy algorithm
    pub fn schedule_greedy(&mut self) -> &Schedule {
        let dep_graph = self
            .dependency_graph
            .as_ref()
            .expect("Must call analyze_circuit first");

        let n_factories = self.factories.len();
        let mut schedule = Schedule::new(dep_graph.t_count, n_factories);

        // Reset all factories
        for factory in &mut self.factories {
            factory.reset();
        }

        let mut current_cycle = 0;
        let mut factory_active_cycles = vec![0usize; n_factories];

        // Process layers in order (respects dependencies)
        for (_layer_idx, layer) in dep_graph.layers.iter().enumerate() {
            // Find earliest cycle when a factory will be ready
            let min_ready_cycle = self
                .factories
                .iter()
                .map(|f| current_cycle + f.cycles_until_ready)
                .min()
                .unwrap_or(current_cycle);

            // Advance time
            let cycles_elapsed = min_ready_cycle.saturating_sub(current_cycle);
            current_cycle = min_ready_cycle;

            // Update factory states
            for factory in &mut self.factories {
                factory.cycles_until_ready =
                    factory.cycles_until_ready.saturating_sub(cycles_elapsed);
            }

            // Assign T-gates in this layer to available factories
            for t_gate in layer {
                // Find factory with minimum cycles_until_ready
                let (factory_id, _) = self
                    .factories
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, f)| f.cycles_until_ready)
                    .unwrap();

                // Assign this T-gate to this factory
                schedule.assignments[t_gate.id] = factory_id;

                let factory = &mut self.factories[factory_id];
                let production_start = current_cycle + factory.cycles_until_ready;
                let production_end = production_start + factory.protocol.cycle_overhead;

                // Add to timeline
                schedule
                    .timeline
                    .entry(production_end)
                    .or_insert_with(Vec::new)
                    .push((factory_id, t_gate.id));

                // Update factory state
                factory.cycles_until_ready = production_end - current_cycle;
                factory.states_produced += 1;

                // Track active cycles
                factory_active_cycles[factory_id] += factory.protocol.cycle_overhead;
            }
        }

        // Calculate total cycles
        schedule.total_cycles = *schedule.timeline.keys().max().unwrap_or(&0);

        // Calculate utilization
        schedule.factory_utilization = factory_active_cycles
            .iter()
            .map(|&active| {
                if schedule.total_cycles > 0 {
                    active as f64 / schedule.total_cycles as f64
                } else {
                    0.0
                }
            })
            .collect();

        self.schedule = Some(schedule);
        self.schedule.as_ref().unwrap()
    }

    /// Simulate schedule execution, return timing metrics
    pub fn simulate(&self) -> ScheduleMetrics {
        let schedule = self
            .schedule
            .as_ref()
            .expect("Must generate schedule first");
        let dep_graph = self
            .dependency_graph
            .as_ref()
            .expect("Must analyze circuit first");

        let total_cycles = schedule.total_cycles;
        let utilization_stats = schedule.utilization_stats();

        // Calculate idle cycles (circuit waiting for magic states)
        // Simplified: assume no idle cycles if schedule exists
        let idle_cycles = 0;

        // Critical path is T-depth × factory latency
        let factory_latency = self.factories[0].protocol.cycle_overhead;
        let critical_path_cycles = dep_graph.t_depth * factory_latency;

        ScheduleMetrics {
            total_cycles,
            idle_cycles,
            factory_utilization: utilization_stats.per_factory,
            avg_utilization: utilization_stats.average,
            bottleneck_factory: schedule.bottleneck_factory(),
            critical_path_cycles,
        }
    }

    /// Find minimum factory count for target latency
    pub fn find_min_factories(
        &self,
        circuit: &[QGate],
        target_cycles: usize,
        protocol: DistillationProtocol,
    ) -> usize {
        let dep_graph = DependencyGraph::from_circuit(circuit);
        let factory_latency = protocol.cycle_overhead;

        // Binary search
        let mut low = 1;
        let mut high = dep_graph.max_parallelism();
        let mut best = high;

        while low <= high {
            let mid = (low + high) / 2;

            // Estimate cycles with 'mid' factories
            let estimated_cycles = Self::estimate_schedule_length(&dep_graph, mid, factory_latency);

            if estimated_cycles <= target_cycles {
                best = mid;
                high = mid.saturating_sub(1);
            } else {
                low = mid + 1;
            }
        }

        best
    }

    /// Estimate schedule length for given factory count
    fn estimate_schedule_length(
        dep_graph: &DependencyGraph,
        n_factories: usize,
        factory_latency: usize,
    ) -> usize {
        // Conservative estimate
        let layers_per_batch = (dep_graph.avg_parallelism / n_factories as f64).ceil() as usize;
        let batches = (dep_graph.t_depth as f64 / layers_per_batch.max(1) as f64).ceil() as usize;
        batches.max(dep_graph.t_depth) * factory_latency
    }

    /// Get factories
    pub fn factories(&self) -> &[Factory] {
        &self.factories
    }

    /// Get mutable factories
    pub fn factories_mut(&mut self) -> &mut [Factory] {
        &mut self.factories
    }

    /// Get current schedule
    pub fn schedule(&self) -> Option<&Schedule> {
        self.schedule.as_ref()
    }
}

// =============================================================================
// Schedule Metrics
// =============================================================================

/// Metrics from schedule simulation
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ScheduleMetrics {
    /// Total execution cycles
    pub total_cycles: usize,

    /// Cycles spent waiting for magic states (circuit idle time)
    pub idle_cycles: usize,

    /// Factory utilization per factory
    pub factory_utilization: Vec<f64>,

    /// Average utilization across all factories
    pub avg_utilization: f64,

    /// Bottleneck factory (highest utilization)
    pub bottleneck_factory: Option<usize>,

    /// Critical path through schedule (T-depth × latency)
    pub critical_path_cycles: usize,
}

impl ScheduleMetrics {
    /// Check if schedule is T-depth limited
    pub fn is_t_depth_limited(&self) -> bool {
        // If total cycles ≈ critical path, we're T-depth limited
        let ratio = self.total_cycles as f64 / self.critical_path_cycles.max(1) as f64;
        ratio < 1.2 // Within 20% of critical path
    }

    /// Check if schedule is throughput limited
    pub fn is_throughput_limited(&self) -> bool {
        !self.is_t_depth_limited()
    }

    /// Generate report
    pub fn report(&self) -> String {
        format!(
            "Schedule Metrics:\n\
             Total Cycles: {}\n\
             Idle Cycles: {} ({:.1}%)\n\
             Avg Factory Utilization: {:.1}%\n\
             Critical Path: {} cycles\n\
             Bottleneck: {:?}\n\
             Limitation: {}",
            self.total_cycles,
            self.idle_cycles,
            100.0 * self.idle_cycles as f64 / self.total_cycles.max(1) as f64,
            100.0 * self.avg_utilization,
            self.critical_path_cycles,
            self.bottleneck_factory,
            if self.is_t_depth_limited() {
                "T-depth limited"
            } else {
                "Throughput limited"
            }
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_graph_simple() {
        let circuit = vec![
            QGate::H(0),
            QGate::T(0),
            QGate::T(1),
            QGate::H(0),
            QGate::T(0),
        ];

        let dep_graph = DependencyGraph::from_circuit(&circuit);

        assert_eq!(dep_graph.t_count, 3);
        assert_eq!(dep_graph.t_depth, 2); // T(0), T(1) parallel; then T(0) sequential
        assert!(dep_graph.avg_parallelism > 1.0);
    }

    #[test]
    fn test_factory_creation() {
        let protocol = DistillationProtocol::fifteen_to_one();
        let factory = Factory::new(0, protocol);

        assert_eq!(factory.id, 0);
        assert_eq!(factory.states_produced, 0);
        assert!(factory.is_idle());
    }

    #[test]
    fn test_scheduler_greedy() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2), QGate::T(0)];

        let protocol = DistillationProtocol::fifteen_to_one();
        let mut scheduler = FactoryScheduler::new(2, protocol);

        scheduler.analyze_circuit(&circuit);
        let schedule = scheduler.schedule_greedy();

        assert_eq!(schedule.assignments.len(), 4);
        assert!(schedule.total_cycles > 0);
    }

    #[test]
    fn test_schedule_metrics() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(0)];

        let protocol = DistillationProtocol::fifteen_to_one();
        let mut scheduler = FactoryScheduler::new(2, protocol);

        scheduler.analyze_circuit(&circuit);
        scheduler.schedule_greedy();

        let metrics = scheduler.simulate();

        assert!(metrics.total_cycles > 0);
        assert!(metrics.avg_utilization > 0.0);
        assert!(metrics.avg_utilization <= 1.0);
    }
}
