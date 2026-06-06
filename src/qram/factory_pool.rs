// =============================================================================
// Factory Pool - Multi-Protocol Factory Management
// =============================================================================
// Sprint 63.0: Magic State Factory Scheduling
//
// Manages pools of heterogeneous distillation factories with auto-configuration.
// Supports multiple protocols, topology-aware placement, and optimization strategies.

use crate::qram::{DependencyGraph, DistillationProtocol, Factory};
use crate::quantum::QGate;
use std::collections::HashMap;

// =============================================================================
// Factory Pool
// =============================================================================

/// Pool of heterogeneous factories
#[derive(Clone, Debug)]
pub struct FactoryPool {
    /// Factories organized by protocol type
    factories_by_protocol: HashMap<String, Vec<usize>>, // protocol name → factory IDs

    /// All factories (flat list)
    all_factories: Vec<Factory>,

    /// Total qubit footprint
    total_qubits: usize,

    /// Next factory ID
    next_id: usize,
}

impl FactoryPool {
    /// Create empty pool
    pub fn new() -> Self {
        FactoryPool {
            factories_by_protocol: HashMap::new(),
            all_factories: Vec::new(),
            total_qubits: 0,
            next_id: 0,
        }
    }

    /// Create pool with N homogeneous factories
    pub fn homogeneous(n: usize, protocol: DistillationProtocol) -> Self {
        let mut pool = Self::new();

        for _ in 0..n {
            pool.add_factory(protocol.clone());
        }

        pool
    }

    /// Create pool with mixed protocols
    ///
    /// Example: vec![(10, fifteen_to_one), (5, reed_muller)]
    pub fn heterogeneous(configs: Vec<(usize, DistillationProtocol)>) -> Self {
        let mut pool = Self::new();

        for (count, protocol) in configs {
            for _ in 0..count {
                pool.add_factory(protocol.clone());
            }
        }

        pool
    }

    /// Auto-configure pool for circuit and target error rate
    pub fn auto_configure(circuit: &[QGate], target_error: f64, max_qubits: usize) -> Self {
        Self::auto_configure_with_strategy(
            circuit,
            target_error,
            max_qubits,
            AutoConfigStrategy::Balanced,
        )
    }

    /// Auto-configure with specific strategy
    pub fn auto_configure_with_strategy(
        circuit: &[QGate],
        target_error: f64,
        max_qubits: usize,
        strategy: AutoConfigStrategy,
    ) -> Self {
        let dep_graph = DependencyGraph::from_circuit(circuit);

        // Select protocol based on target error
        let protocol = select_protocol_for_error(target_error);

        // Determine factory count based on strategy
        let n_factories = match strategy {
            AutoConfigStrategy::MinimizeLatency => {
                // Use enough factories to saturate parallelism
                dep_graph.max_parallelism()
            }

            AutoConfigStrategy::MinimizeQubits => {
                // Use single factory (accept T-depth latency)
                1
            }

            AutoConfigStrategy::Balanced => {
                // Use sqrt(parallelism) factories
                (dep_graph.max_parallelism() as f64).sqrt().ceil() as usize
            }

            AutoConfigStrategy::TargetLatency { max_cycles } => {
                // Binary search for minimum factories meeting latency target
                binary_search_factory_count(&dep_graph, target_error, max_cycles)
            }
        };

        // Ensure we don't exceed qubit budget
        let qubit_per_factory = protocol.qubit_overhead;
        let max_factories = max_qubits / qubit_per_factory.max(1);
        let n_factories = n_factories.min(max_factories).max(1);

        Self::homogeneous(n_factories, protocol)
    }

    /// Add factory to pool
    pub fn add_factory(&mut self, protocol: DistillationProtocol) -> usize {
        let factory_id = self.next_id;
        self.next_id += 1;

        let factory = Factory::new(factory_id, protocol.clone());

        self.total_qubits += factory.qubit_count();
        self.all_factories.push(factory);

        // Add to protocol map
        self.factories_by_protocol
            .entry(protocol.name.to_string())
            .or_insert_with(Vec::new)
            .push(factory_id);

        factory_id
    }

    /// Remove factory from pool
    pub fn remove_factory(&mut self, factory_id: usize) -> Option<Factory> {
        if let Some(idx) = self.all_factories.iter().position(|f| f.id == factory_id) {
            let factory = self.all_factories.remove(idx);
            self.total_qubits = self.total_qubits.saturating_sub(factory.qubit_count());

            // Remove from protocol map
            if let Some(ids) = self.factories_by_protocol.get_mut(factory.protocol.name) {
                ids.retain(|&id| id != factory_id);
            }

            Some(factory)
        } else {
            None
        }
    }

    /// Get factory by ID
    pub fn get_factory(&self, id: usize) -> Option<&Factory> {
        self.all_factories.iter().find(|f| f.id == id)
    }

    /// Get mutable factory by ID
    pub fn get_factory_mut(&mut self, id: usize) -> Option<&mut Factory> {
        self.all_factories.iter_mut().find(|f| f.id == id)
    }

    /// Get all factories using a specific protocol
    pub fn factories_with_protocol(&self, protocol_name: &str) -> Vec<&Factory> {
        if let Some(ids) = self.factories_by_protocol.get(protocol_name) {
            ids.iter().filter_map(|&id| self.get_factory(id)).collect()
        } else {
            vec![]
        }
    }

    /// Total qubit footprint of pool
    pub fn total_qubits(&self) -> usize {
        self.total_qubits
    }

    /// Number of factories in pool
    pub fn factory_count(&self) -> usize {
        self.all_factories.len()
    }

    /// Get all factories
    pub fn all_factories(&self) -> &[Factory] {
        &self.all_factories
    }

    /// Get mutable all factories
    pub fn all_factories_mut(&mut self) -> &mut Vec<Factory> {
        &mut self.all_factories
    }

    /// Estimate cost (qubits × time) for N magic states
    pub fn estimate_cost(&self, n_states: usize) -> (usize, usize) {
        if self.all_factories.is_empty() {
            return (0, 0);
        }

        // Calculate throughput
        let total_throughput: f64 = self
            .all_factories
            .iter()
            .map(|f| 1.0 / f.protocol.cycle_overhead as f64)
            .sum();

        // Time = states / throughput
        let cycles = (n_states as f64 / total_throughput).ceil() as usize;

        (self.total_qubits, cycles)
    }

    /// Get protocol distribution
    pub fn protocol_distribution(&self) -> HashMap<String, usize> {
        let mut dist = HashMap::new();

        for factory in &self.all_factories {
            *dist.entry(factory.protocol.name.to_string()).or_insert(0) += 1;
        }

        dist
    }

    /// Get statistics
    pub fn stats(&self) -> FactoryPoolStats {
        let protocol_dist = self.protocol_distribution();

        let avg_qubit_per_factory = if !self.all_factories.is_empty() {
            self.total_qubits as f64 / self.all_factories.len() as f64
        } else {
            0.0
        };

        let avg_latency = if !self.all_factories.is_empty() {
            self.all_factories
                .iter()
                .map(|f| f.protocol.cycle_overhead)
                .sum::<usize>() as f64
                / self.all_factories.len() as f64
        } else {
            0.0
        };

        FactoryPoolStats {
            total_factories: self.all_factories.len(),
            total_qubits: self.total_qubits,
            protocol_distribution: protocol_dist,
            avg_qubit_per_factory,
            avg_latency,
        }
    }
}

impl Default for FactoryPool {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Auto-Configuration
// =============================================================================

/// Auto-configuration strategy
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutoConfigStrategy {
    /// Minimize latency (use many factories)
    MinimizeLatency,

    /// Minimize qubit footprint (use few factories, accept higher latency)
    MinimizeQubits,

    /// Balance latency and qubits
    Balanced,

    /// Custom: target latency with minimum qubits
    TargetLatency { max_cycles: usize },
}

/// Select appropriate distillation protocol for target error rate
pub fn select_protocol_for_error(target_error: f64) -> DistillationProtocol {
    // Starting from physical error ~1e-3
    if target_error > 1e-6 {
        // Fast, low suppression (7-to-1)
        DistillationProtocol::seven_to_one()
    } else if target_error > 1e-10 {
        // Standard (15-to-1, single level)
        DistillationProtocol::fifteen_to_one()
    } else if target_error > 1e-15 {
        // 15-to-1 with 2 levels (cascaded)
        DistillationProtocol::fifteen_to_one()
    } else {
        // Very low error (Reed-Muller)
        DistillationProtocol::reed_muller()
    }
}

/// Binary search for minimum factory count meeting latency target
pub fn binary_search_factory_count(
    dep_graph: &DependencyGraph,
    target_error: f64,
    max_cycles: usize,
) -> usize {
    let protocol = select_protocol_for_error(target_error);
    let factory_latency = protocol.cycle_overhead;

    // Minimum: 1 factory, Maximum: max parallelism
    let mut low = 1;
    let mut high = dep_graph.max_parallelism();
    let mut best = high;

    while low <= high {
        let mid = (low + high) / 2;

        // Estimate cycles with 'mid' factories
        let estimated_cycles = estimate_schedule_length(dep_graph, mid, factory_latency);

        if estimated_cycles <= max_cycles {
            best = mid;
            high = mid.saturating_sub(1); // Try fewer factories
        } else {
            low = mid + 1; // Need more factories
        }
    }

    best
}

/// Estimate schedule length for given factory count
pub fn estimate_schedule_length(
    dep_graph: &DependencyGraph,
    n_factories: usize,
    factory_latency: usize,
) -> usize {
    // Conservative estimate: average layer parallelism / factories
    let avg_per_layer = dep_graph.avg_parallelism;
    let layers_per_factory_batch = (avg_per_layer / n_factories as f64).ceil() as usize;
    let batches =
        (dep_graph.t_depth as f64 / layers_per_factory_batch.max(1) as f64).ceil() as usize;

    batches.max(dep_graph.t_depth) * factory_latency
}

// =============================================================================
// Statistics
// =============================================================================

/// Factory pool statistics
#[derive(Clone, Debug)]
pub struct FactoryPoolStats {
    /// Total number of factories
    pub total_factories: usize,

    /// Total qubit footprint
    pub total_qubits: usize,

    /// Protocol distribution (protocol name → count)
    pub protocol_distribution: HashMap<String, usize>,

    /// Average qubits per factory
    pub avg_qubit_per_factory: f64,

    /// Average factory latency (cycles)
    pub avg_latency: f64,
}

impl FactoryPoolStats {
    /// Generate report
    pub fn report(&self) -> String {
        let mut lines = vec![
            format!("=== Factory Pool Statistics ==="),
            format!("Total Factories: {}", self.total_factories),
            format!("Total Qubits: {}", self.total_qubits),
            format!("Avg Qubits/Factory: {:.1}", self.avg_qubit_per_factory),
            format!("Avg Latency: {:.1} cycles", self.avg_latency),
            format!("Protocol Distribution:"),
        ];

        for (protocol, count) in &self.protocol_distribution {
            lines.push(format!("  {}: {} factories", protocol, count));
        }

        lines.join("\n")
    }
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Factory pool builder for fluent configuration
pub struct FactoryPoolBuilder {
    pool: FactoryPool,
}

impl FactoryPoolBuilder {
    /// Create new builder
    pub fn new() -> Self {
        FactoryPoolBuilder {
            pool: FactoryPool::new(),
        }
    }

    /// Add N factories with protocol
    pub fn add(mut self, n: usize, protocol: DistillationProtocol) -> Self {
        for _ in 0..n {
            self.pool.add_factory(protocol.clone());
        }
        self
    }

    /// Add single factory
    pub fn add_one(mut self, protocol: DistillationProtocol) -> Self {
        self.pool.add_factory(protocol);
        self
    }

    /// Build pool
    pub fn build(self) -> FactoryPool {
        self.pool
    }
}

impl Default for FactoryPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homogeneous_pool() {
        let protocol = DistillationProtocol::fifteen_to_one();
        let pool = FactoryPool::homogeneous(5, protocol);

        assert_eq!(pool.factory_count(), 5);
        assert_eq!(pool.total_qubits(), 5 * 20); // 15-to-1 uses ~20 qubits
    }

    #[test]
    fn test_heterogeneous_pool() {
        let configs = vec![
            (3, DistillationProtocol::fifteen_to_one()),
            (2, DistillationProtocol::reed_muller()),
        ];

        let pool = FactoryPool::heterogeneous(configs);

        assert_eq!(pool.factory_count(), 5);
        assert_eq!(pool.factories_with_protocol("15-to-1").len(), 3);
        assert_eq!(pool.factories_with_protocol("Reed-Muller").len(), 2);
    }

    #[test]
    fn test_add_remove_factory() {
        let mut pool = FactoryPool::new();
        let protocol = DistillationProtocol::fifteen_to_one();

        let id = pool.add_factory(protocol);
        assert_eq!(pool.factory_count(), 1);

        let removed = pool.remove_factory(id);
        assert!(removed.is_some());
        assert_eq!(pool.factory_count(), 0);
    }

    #[test]
    fn test_protocol_selection() {
        let p1 = select_protocol_for_error(1e-5);
        assert_eq!(p1.name, "7-to-1");

        let p2 = select_protocol_for_error(1e-7);
        assert_eq!(p2.name, "15-to-1");

        let p3 = select_protocol_for_error(1e-16);
        assert_eq!(p3.name, "Reed-Muller");
    }

    #[test]
    fn test_auto_configure() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2), QGate::T(0)];

        let pool = FactoryPool::auto_configure(&circuit, 1e-8, 1000);

        assert!(pool.factory_count() > 0);
        assert!(pool.total_qubits() <= 1000);
    }

    #[test]
    fn test_builder() {
        let pool = FactoryPoolBuilder::new()
            .add(3, DistillationProtocol::fifteen_to_one())
            .add(2, DistillationProtocol::seven_to_one())
            .build();

        assert_eq!(pool.factory_count(), 5);
    }

    #[test]
    fn test_estimate_cost() {
        let pool = FactoryPool::homogeneous(2, DistillationProtocol::fifteen_to_one());

        let (qubits, cycles) = pool.estimate_cost(100);

        assert_eq!(qubits, 40); // 2 factories × 20 qubits
        assert!(cycles > 0);
    }
}
