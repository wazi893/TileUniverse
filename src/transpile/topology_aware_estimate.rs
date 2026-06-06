// =============================================================================
// Topology-Aware Resource Estimation - Integration with Factory Scheduling
// =============================================================================
// Sprint 64.0: Hardware-Topology Factory Placement
//
// Extends Sprint 63's DynamicResourceEstimator with hardware topology awareness,
// adding realistic routing costs for magic state delivery.

use crate::hardware::{
    DeviceTopology, PhysicalFactoryLayout, PlacementOptimizer, PlacementStrategy,
};
use crate::qram::{
    BottleneckType, DependencyGraph, FactoryPool, FactoryScheduler, OptimizationStrategy,
    ScheduleOptimizer, ThroughputModel,
};
use crate::quantum::QGate;
use crate::transpile::dynamic_estimate::{DynamicResourceEstimate, FaultTolerantConfig};

// =============================================================================
// Topology Type Enum
// =============================================================================

/// Topology type for resource estimation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyType {
    HeavyHex,
    Grid2D,
    AllToAll,
    Linear,
    Custom,
}

impl From<&DeviceTopology> for TopologyType {
    fn from(topology: &DeviceTopology) -> Self {
        match topology {
            DeviceTopology::HeavyHex { .. } => TopologyType::HeavyHex,
            DeviceTopology::Grid2D { .. } => TopologyType::Grid2D,
            DeviceTopology::FullyConnected { .. } | DeviceTopology::LinearChain { .. } => {
                TopologyType::AllToAll
            }
            DeviceTopology::Ring { .. } => TopologyType::Linear,
            _ => TopologyType::Custom,
        }
    }
}

// =============================================================================
// Topology-Aware Estimator
// =============================================================================

/// Resource estimator with hardware topology awareness
pub struct TopologyAwareEstimator {
    /// Circuit to estimate
    circuit: Vec<QGate>,

    /// Fault-tolerant configuration
    ft_config: FaultTolerantConfig,

    /// Device topology
    topology: DeviceTopology,

    /// Factory placement strategy
    placement_strategy: PlacementStrategy,

    /// Schedule optimization strategy
    schedule_strategy: OptimizationStrategy,

    /// Factory pool (optional, auto-configured if not provided)
    factory_pool: Option<FactoryPool>,
}

impl TopologyAwareEstimator {
    /// Create estimator
    pub fn new(
        circuit: Vec<QGate>,
        ft_config: FaultTolerantConfig,
        topology: DeviceTopology,
        placement_strategy: PlacementStrategy,
    ) -> Self {
        TopologyAwareEstimator {
            circuit,
            ft_config,
            topology,
            placement_strategy,
            schedule_strategy: OptimizationStrategy::Greedy,
            factory_pool: None,
        }
    }

    /// Set factory pool manually
    pub fn with_factory_pool(mut self, pool: FactoryPool) -> Self {
        self.factory_pool = Some(pool);
        self
    }

    /// Set schedule optimization strategy
    pub fn with_schedule_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.schedule_strategy = strategy;
        self
    }

    /// Perform topology-aware estimation
    pub fn estimate(&mut self) -> TopologyAwareEstimate {
        // 1. Analyze circuit T-gates
        let dep_graph = DependencyGraph::from_circuit(&self.circuit);

        // 2. Configure factory pool if not provided
        let pool = if let Some(ref pool) = self.factory_pool {
            pool.clone()
        } else {
            use crate::qram::AutoConfigStrategy;
            FactoryPool::auto_configure_with_strategy(
                &self.circuit,
                self.ft_config.target_logical_error_rate,
                self.ft_config.max_factory_qubits.unwrap_or(100_000),
                AutoConfigStrategy::Balanced,
            )
        };

        // 3. Create scheduler and generate schedule
        let mut scheduler = FactoryScheduler::with_factories(pool.all_factories().to_vec());
        scheduler.analyze_circuit(&self.circuit);

        let mut optimizer = ScheduleOptimizer::new(scheduler, self.schedule_strategy.clone());
        let schedule = optimizer.optimize();

        // 4. Optimize factory placement on topology
        let coupling_map = self.topology.to_coupling_map();
        let mut placement_opt =
            PlacementOptimizer::new(coupling_map, self.placement_strategy.clone());

        let layout = placement_opt.optimize(&self.circuit, pool.all_factories(), &schedule);

        // 5. Simulate schedule
        let metrics = optimizer.scheduler().simulate();

        // 6. Calculate throughput model (without routing)
        let throughput_model = ThroughputModel::new(pool.clone(), &self.circuit);
        let exec_estimate = throughput_model.estimate_execution_time();

        // 7. Calculate costs
        let data_qubits = self.circuit_data_qubits();
        let factory_qubits = pool.total_qubits();
        let total_qubits = data_qubits + factory_qubits;

        let base_cycles = schedule.total_cycles;
        let swap_overhead = layout.total_swap_cost;
        let total_cycles = base_cycles + swap_overhead;

        let qubit_cycles = total_qubits as u64 * total_cycles as u64;
        let qubit_hours = qubit_cycles as f64 / 1e6 / 3600.0;

        // 8. Calculate routing quality
        let routing_quality = self.calculate_routing_quality(&layout, &dep_graph);

        TopologyAwareEstimate {
            // Circuit properties
            t_count: dep_graph.t_count,
            t_depth: dep_graph.t_depth,
            avg_parallelism: dep_graph.avg_parallelism,

            // Factory properties
            num_factories: pool.factory_count(),
            factory_qubits,

            // Data qubit properties
            data_qubits,
            logical_qubits: self.circuit_logical_qubits(),
            code_distance: self.ft_config.code_distance,

            // Execution properties (without routing)
            base_schedule_cycles: base_cycles,
            factory_utilization: metrics.avg_utilization,

            // Topology-specific properties
            topology_type: TopologyType::from(&self.topology),
            swap_overhead,
            peak_congestion: layout.peak_congestion,
            routing_quality,

            // Total execution (with routing)
            total_cycles,
            bottleneck_type: exec_estimate.bottleneck_type,

            // Cost metrics
            total_qubits,
            qubit_cycles,
            qubit_hours,

            // Throughput metrics
            steady_state_throughput: exec_estimate.steady_state_throughput,
            average_latency: exec_estimate.average_latency,

            // Full details
            layout,
            dynamic_estimate: DynamicResourceEstimate {
                t_count: dep_graph.t_count,
                t_depth: dep_graph.t_depth,
                avg_parallelism: dep_graph.avg_parallelism,
                num_factories: pool.factory_count(),
                factory_qubits,
                data_qubits,
                logical_qubits: self.circuit_logical_qubits(),
                code_distance: self.ft_config.code_distance,
                schedule_cycles: base_cycles,
                bottleneck_type: exec_estimate.bottleneck_type,
                factory_utilization: metrics.avg_utilization,
                total_qubits,
                qubit_cycles: total_qubits as u64 * base_cycles as u64,
                qubit_hours: (total_qubits as u64 * base_cycles as u64) as f64 / 1e6 / 3600.0,
                steady_state_throughput: exec_estimate.steady_state_throughput,
                average_latency: exec_estimate.average_latency,
                schedule,
                metrics,
                exec_estimate,
            },
        }
    }

    fn circuit_logical_qubits(&self) -> usize {
        let max_qubit = self
            .circuit
            .iter()
            .filter_map(|g| Self::gate_max_qubit(g))
            .max()
            .unwrap_or(0);
        max_qubit + 1
    }

    fn circuit_data_qubits(&self) -> usize {
        let d = self.ft_config.code_distance;
        self.circuit_logical_qubits() * d * d
    }

    fn gate_max_qubit(gate: &QGate) -> Option<usize> {
        match gate {
            QGate::H(q) | QGate::X(q) | QGate::Y(q) | QGate::Z(q) | QGate::T(q) | QGate::Tdg(q) => {
                Some(*q as usize)
            }
            QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => Some((*c).max(*t) as usize),
            QGate::Toffoli(c1, c2, t) | QGate::CCZ(c1, c2, t) => {
                Some((*c1).max(*c2).max(*t) as usize)
            }
            _ => None,
        }
    }

    fn calculate_routing_quality(
        &self,
        layout: &PhysicalFactoryLayout,
        dep_graph: &DependencyGraph,
    ) -> f64 {
        // Quality metric: 1.0 (optimal) to 0.0 (poor)
        // Based on avg SWAPs per T-gate relative to theoretical minimum

        if dep_graph.t_count == 0 {
            return 1.0;
        }

        let avg_swaps = layout.metrics.avg_swaps_per_t;

        // Theoretical minimum: avg distance in topology
        let coupling = self.topology.to_coupling_map();
        let avg_distance = coupling.average_distance();

        if avg_distance == 0.0 {
            return 1.0;
        }

        // Quality = theoretical / actual (clamped to [0, 1])
        (avg_distance / avg_swaps.max(1.0)).min(1.0)
    }
}

// =============================================================================
// Topology-Aware Estimate
// =============================================================================

/// Topology-aware resource estimate
#[derive(Clone, Debug)]
pub struct TopologyAwareEstimate {
    // === Circuit Properties ===
    pub t_count: usize,
    pub t_depth: usize,
    pub avg_parallelism: f64,

    // === Factory Properties ===
    pub num_factories: usize,
    pub factory_qubits: usize,

    // === Data Qubit Properties ===
    pub data_qubits: usize,
    pub logical_qubits: usize,
    pub code_distance: usize,

    // === Base Execution (without routing) ===
    pub base_schedule_cycles: usize,
    pub factory_utilization: f64,

    // === Topology-Specific Properties ===
    pub topology_type: TopologyType,
    pub swap_overhead: usize,
    pub peak_congestion: usize,
    pub routing_quality: f64,

    // === Total Execution (with routing) ===
    pub total_cycles: usize,
    pub bottleneck_type: BottleneckType,

    // === Cost Metrics ===
    pub total_qubits: usize,
    pub qubit_cycles: u64,
    pub qubit_hours: f64,

    // === Throughput Metrics ===
    pub steady_state_throughput: f64,
    pub average_latency: f64,

    // === Full Details ===
    pub layout: PhysicalFactoryLayout,
    pub dynamic_estimate: DynamicResourceEstimate,
}

impl TopologyAwareEstimate {
    /// Generate detailed report
    pub fn report(&self) -> String {
        let overhead_pct = if self.base_schedule_cycles > 0 {
            (self.swap_overhead as f64 / self.base_schedule_cycles as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "=== Topology-Aware Resource Estimate ===\n\
             \n\
             Circuit:\n\
             - T-gates: {} (T-depth: {})\n\
             - Avg Parallelism: {:.2}\n\
             - Logical Qubits: {}\n\
             \n\
             Hardware:\n\
             - Topology: {:?}\n\
             - Device Qubits: {} (available for placement)\n\
             - Code Distance: {}\n\
             - Physical Qubits (data): {}\n\
             - Physical Qubits (factories): {}\n\
             - Total Physical Qubits: {}\n\
             \n\
             Factory Scheduling:\n\
             - Factories: {}\n\
             - Base Schedule: {} cycles ({:.2} ms at 1MHz)\n\
             - Factory Utilization: {:.1}%\n\
             \n\
             Routing Overhead:\n\
             - SWAP Operations: {} cycles\n\
             - Overhead: {:.1}% of base schedule\n\
             - Peak Congestion: {} SWAPs/edge\n\
             - Routing Quality: {:.1}% (1.0 = optimal)\n\
             \n\
             Total Execution:\n\
             - Total Cycles: {} ({:.2} ms at 1MHz)\n\
             - Bottleneck: {:?}\n\
             - Throughput: {:.3} states/cycle\n\
             - Avg Latency: {:.1} cycles/state\n\
             \n\
             Cost:\n\
             - Total Qubit-Cycles: {}\n\
             - Qubit-Hours: {:.2} (at 1MHz)\n\
             \n\
             {}",
            self.t_count,
            self.t_depth,
            self.avg_parallelism,
            self.logical_qubits,
            self.topology_type,
            self.total_qubits,
            self.code_distance,
            self.data_qubits,
            self.factory_qubits,
            self.total_qubits,
            self.num_factories,
            self.base_schedule_cycles,
            self.base_schedule_cycles as f64 / 1e3,
            self.factory_utilization * 100.0,
            self.swap_overhead,
            overhead_pct,
            self.peak_congestion,
            self.routing_quality * 100.0,
            self.total_cycles,
            self.total_cycles as f64 / 1e3,
            self.bottleneck_type,
            self.steady_state_throughput,
            self.average_latency,
            self.qubit_cycles,
            self.qubit_hours,
            self.recommendation()
        )
    }

    fn recommendation(&self) -> &str {
        let overhead_pct = if self.base_schedule_cycles > 0 {
            (self.swap_overhead as f64 / self.base_schedule_cycles as f64) * 100.0
        } else {
            0.0
        };

        if overhead_pct > 100.0 {
            "Recommendation:\n\
             SWAP overhead exceeds base execution time. Consider:\n\
             - Using all-to-all topology (IonQ, trapped ions)\n\
             - Reducing code distance to free more qubits for factories\n\
             - Co-optimizing circuit mapping with factory placement"
        } else if overhead_pct > 50.0 {
            "Recommendation:\n\
             SWAP overhead is significant. Consider:\n\
             - Optimizing factory placement strategy\n\
             - Using device with better connectivity\n\
             - Accepting overhead if within time budget"
        } else if overhead_pct > 20.0 {
            "Recommendation:\n\
             SWAP overhead is moderate. Current placement is reasonable.\n\
             Minor improvements possible through better placement strategies."
        } else {
            "Recommendation:\n\
             SWAP overhead is low. Current topology and placement are well-suited.\n\
             Focus on other optimization opportunities."
        }
    }

    /// Compare with dynamic estimate (without topology)
    pub fn compare_with_dynamic(&self) -> TopologyComparison {
        let dynamic_cycles = self.base_schedule_cycles;
        let topology_cycles = self.total_cycles;
        let overhead_ratio = if dynamic_cycles > 0 {
            (topology_cycles as f64 / dynamic_cycles as f64) - 1.0
        } else {
            0.0
        };

        TopologyComparison {
            dynamic_cycles,
            topology_cycles,
            swap_overhead: self.swap_overhead,
            overhead_ratio,
            dynamic_qubit_hours: self.dynamic_estimate.qubit_hours,
            topology_qubit_hours: self.qubit_hours,
        }
    }
}

// =============================================================================
// Topology Comparison
// =============================================================================

/// Comparison between dynamic (Sprint 63) and topology-aware (Sprint 64)
#[derive(Clone, Debug)]
pub struct TopologyComparison {
    pub dynamic_cycles: usize,
    pub topology_cycles: usize,
    pub swap_overhead: usize,
    pub overhead_ratio: f64,
    pub dynamic_qubit_hours: f64,
    pub topology_qubit_hours: f64,
}

impl TopologyComparison {
    pub fn report(&self) -> String {
        format!(
            "=== Dynamic vs Topology-Aware Comparison ===\n\
             Dynamic Estimate (Sprint 63): {} cycles ({:.2} qubit-hours)\n\
             Topology-Aware (Sprint 64): {} cycles ({:.2} qubit-hours)\n\
             SWAP Overhead: {} cycles ({:.1}% increase)\n\
             \n\
             Explanation:\n\
             Dynamic assumes instant magic state delivery (Sprint 63).\n\
             Topology-aware includes realistic routing cost (Sprint 64).",
            self.dynamic_cycles,
            self.dynamic_qubit_hours,
            self.topology_cycles,
            self.topology_qubit_hours,
            self.swap_overhead,
            self.overhead_ratio * 100.0
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::DeviceTopology;

    #[test]
    #[allow(unused_comparisons)]
    fn test_topology_aware_estimator() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2), QGate::T(0)];

        let topology = DeviceTopology::linear_chain(50);
        let ft_config = FaultTolerantConfig::nisq_era();

        let mut estimator = TopologyAwareEstimator::new(
            circuit,
            ft_config,
            topology,
            PlacementStrategy::MinimizeTotalSwapCost,
        );

        let estimate = estimator.estimate();

        assert_eq!(estimate.t_count, 4);
        assert!(estimate.total_cycles > 0);
        assert!(estimate.swap_overhead >= 0);
        assert!(estimate.total_qubits > 0);
    }

    #[test]
    fn test_all_to_all_minimal_overhead() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2)];

        let topology = DeviceTopology::fully_connected(32);
        let ft_config = FaultTolerantConfig::nisq_era();

        let mut estimator = TopologyAwareEstimator::new(
            circuit,
            ft_config,
            topology,
            PlacementStrategy::MinimizeAverageDistance,
        );

        let estimate = estimator.estimate();

        // All-to-all should have low overhead
        let overhead_ratio = estimate.swap_overhead as f64 / estimate.base_schedule_cycles as f64;
        assert!(overhead_ratio < 0.5); // < 50%
    }

    #[test]
    fn test_comparison() {
        let circuit = vec![QGate::T(0), QGate::T(1)];

        let topology = DeviceTopology::linear_chain(20);
        let ft_config = FaultTolerantConfig::nisq_era();

        let mut estimator = TopologyAwareEstimator::new(
            circuit,
            ft_config,
            topology,
            PlacementStrategy::MinimizeTotalSwapCost,
        );

        let estimate = estimator.estimate();
        let comparison = estimate.compare_with_dynamic();

        assert!(comparison.topology_cycles >= comparison.dynamic_cycles);
        assert!(comparison.overhead_ratio >= 0.0);
    }
}
