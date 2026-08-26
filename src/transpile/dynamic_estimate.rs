// =============================================================================
// Dynamic Resource Estimation - Factory-Aware FTQC Resource Estimation
// =============================================================================
// Sprint 63.0: Magic State Factory Scheduling
//
// Integrates factory scheduling with fault-tolerant resource estimation
// for realistic execution time and qubit-hour projections.

use crate::qram::{
    AutoConfigStrategy, BottleneckType, DependencyGraph, FactoryPool, FactoryScheduler,
    OptimizationStrategy, ScheduleOptimizer, ThroughputModel,
};
use crate::quantum::QGate;

#[cfg(feature = "config")]
use serde_json;

// =============================================================================
// Dynamic Resource Estimator
// =============================================================================

/// Dynamic resource estimation with factory scheduling
pub struct DynamicResourceEstimator {
    /// Circuit to estimate
    circuit: Vec<QGate>,

    /// Fault-tolerant configuration
    ft_config: FaultTolerantConfig,

    /// Factory pool (auto-configured or user-specified)
    factory_pool: Option<FactoryPool>,

    /// Schedule optimization strategy
    optimization_strategy: OptimizationStrategy,
}

impl DynamicResourceEstimator {
    /// Create estimator with circuit and FT config
    pub fn new(circuit: Vec<QGate>, ft_config: FaultTolerantConfig) -> Self {
        DynamicResourceEstimator {
            circuit,
            ft_config,
            factory_pool: None,
            optimization_strategy: OptimizationStrategy::ListScheduling {
                priority: crate::qram::PriorityRule::EarliestDeadline,
            },
        }
    }

    /// Set factory pool manually
    pub fn with_factory_pool(mut self, pool: FactoryPool) -> Self {
        self.factory_pool = Some(pool);
        self
    }

    /// Set optimization strategy
    pub fn with_optimization_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.optimization_strategy = strategy;
        self
    }

    /// Perform dynamic estimation with factory scheduling
    pub fn estimate(&mut self) -> DynamicResourceEstimate {
        // 1. Analyze circuit T-gates
        let dep_graph = DependencyGraph::from_circuit(&self.circuit);

        // 2. Configure factory pool if not provided
        let pool = if let Some(ref pool) = self.factory_pool {
            pool.clone()
        } else {
            // Auto-configure based on circuit and target error rate
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

        let mut optimizer = ScheduleOptimizer::new(scheduler, self.optimization_strategy.clone());
        let schedule = optimizer.optimize();

        // 4. Simulate schedule
        let metrics = optimizer.scheduler().simulate();

        // 5. Calculate throughput model
        let throughput_model = ThroughputModel::new(pool.clone(), &self.circuit);
        let exec_estimate = throughput_model.estimate_execution_time();

        // 6. Calculate costs
        let data_qubits = self.circuit_data_qubits();
        let factory_qubits = pool.total_qubits();
        let total_qubits = data_qubits + factory_qubits;

        let execution_cycles = schedule.total_cycles;
        let qubit_cycles = total_qubits as u64 * execution_cycles as u64;

        // At 1MHz clock rate
        let qubit_hours = qubit_cycles as f64 / 1e6 / 3600.0;

        DynamicResourceEstimate {
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

            // Schedule properties
            schedule_cycles: execution_cycles,
            bottleneck_type: exec_estimate.bottleneck_type,
            factory_utilization: metrics.avg_utilization,

            // Cost metrics
            total_qubits,
            qubit_cycles,
            qubit_hours,

            // Throughput metrics
            steady_state_throughput: exec_estimate.steady_state_throughput,
            average_latency: exec_estimate.average_latency,

            // Full details
            schedule,
            metrics,
            exec_estimate,
        }
    }

    fn circuit_logical_qubits(&self) -> usize {
        // Count maximum qubit index in circuit
        let max_qubit = self
            .circuit
            .iter()
            .filter_map(Self::gate_max_qubit)
            .max()
            .unwrap_or(0);

        max_qubit + 1
    }

    fn circuit_data_qubits(&self) -> usize {
        // Multiply logical qubits by code overhead (d^2 for surface code)
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
}

// =============================================================================
// Fault-Tolerant Configuration
// =============================================================================

/// Fault-tolerant configuration parameters
#[derive(Clone, Debug)]
pub struct FaultTolerantConfig {
    /// Surface code distance
    pub code_distance: usize,

    /// Physical error rate (per gate)
    pub physical_error_rate: f64,

    /// Target logical error rate
    pub target_logical_error_rate: f64,

    /// Maximum qubits for factories (budget constraint)
    pub max_factory_qubits: Option<usize>,
}

impl FaultTolerantConfig {
    /// Standard config for NISQ-era emulation
    pub fn nisq_era() -> Self {
        FaultTolerantConfig {
            code_distance: 3,
            physical_error_rate: 1e-3,
            target_logical_error_rate: 1e-6,
            max_factory_qubits: Some(10_000),
        }
    }

    /// Config for fault-tolerant regime
    pub fn fault_tolerant() -> Self {
        FaultTolerantConfig {
            code_distance: 17,
            physical_error_rate: 1e-3,
            target_logical_error_rate: 1e-12,
            max_factory_qubits: Some(100_000),
        }
    }

    /// Custom config
    pub fn custom(
        code_distance: usize,
        physical_error_rate: f64,
        target_logical_error_rate: f64,
    ) -> Self {
        FaultTolerantConfig {
            code_distance,
            physical_error_rate,
            target_logical_error_rate,
            max_factory_qubits: None,
        }
    }
}

// =============================================================================
// Dynamic Resource Estimate
// =============================================================================

/// Dynamic resource estimate with factory scheduling
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DynamicResourceEstimate {
    // === Circuit Properties ===
    /// Total T-count
    pub t_count: usize,

    /// T-depth
    pub t_depth: usize,

    /// Average parallelism
    pub avg_parallelism: f64,

    // === Factory Properties ===
    /// Number of factories
    pub num_factories: usize,

    /// Qubits used by factories
    pub factory_qubits: usize,

    // === Data Qubit Properties ===
    /// Physical qubits for circuit data
    pub data_qubits: usize,

    /// Logical qubits in circuit
    pub logical_qubits: usize,

    /// Surface code distance
    pub code_distance: usize,

    // === Execution Properties ===
    /// Total execution cycles (with factory scheduling)
    pub schedule_cycles: usize,

    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Factory utilization (0.0 to 1.0)
    pub factory_utilization: f64,

    // === Cost Metrics ===
    /// Total qubits (data + factories)
    pub total_qubits: usize,

    /// Qubit-cycles (space × time)
    pub qubit_cycles: u64,

    /// Qubit-hours (at 1MHz clock rate)
    pub qubit_hours: f64,

    // === Throughput Metrics ===
    /// Steady-state magic state throughput
    pub steady_state_throughput: f64,

    /// Average latency per magic state
    pub average_latency: f64,

    // === Full Details ===
    /// Complete schedule
    pub schedule: crate::qram::Schedule,

    /// Schedule metrics
    pub metrics: crate::qram::ScheduleMetrics,

    /// Execution estimate
    pub exec_estimate: crate::qram::ExecutionEstimate,
}

impl DynamicResourceEstimate {
    /// Generate detailed report
    pub fn report(&self) -> String {
        format!(
            "=== Dynamic Resource Estimate ===\n\
             \n\
             Circuit:\n\
             - T-gates: {} (T-depth: {})\n\
             - Avg Parallelism: {:.2}\n\
             - Logical Qubits: {}\n\
             \n\
             Fault-Tolerant Configuration:\n\
             - Code Distance: {}\n\
             - Physical Qubits (data): {}\n\
             - Physical Qubits (factories): {}\n\
             - Total Physical Qubits: {}\n\
             \n\
             Factory Scheduling:\n\
             - Factories: {}\n\
             - Execution Cycles: {} ({:.2} ms at 1MHz)\n\
             - Bottleneck: {:?}\n\
             - Factory Utilization: {:.1}%\n\
             - Throughput: {:.3} states/cycle\n\
             - Avg Latency: {:.1} cycles/state\n\
             \n\
             Cost:\n\
             - Total Qubit-Cycles: {}\n\
             - Qubit-Hours: {:.2} (at 1MHz)\n\
             \n\
             Recommendation:\n\
             {}",
            self.t_count,
            self.t_depth,
            self.avg_parallelism,
            self.logical_qubits,
            self.code_distance,
            self.data_qubits,
            self.factory_qubits,
            self.total_qubits,
            self.num_factories,
            self.schedule_cycles,
            self.schedule_cycles as f64 / 1e3,
            self.bottleneck_type,
            self.factory_utilization * 100.0,
            self.steady_state_throughput,
            self.average_latency,
            self.qubit_cycles,
            self.qubit_hours,
            self.recommendation()
        )
    }

    fn recommendation(&self) -> &str {
        match self.bottleneck_type {
            BottleneckType::TDepthLimited => {
                "T-depth limited. Adding more factories won't help significantly.\n\
                 Consider:\n\
                 - T-depth optimization via gate reordering\n\
                 - Circuit restructuring to reduce sequential T-gates\n\
                 - Accept current latency if within budget"
            }
            BottleneckType::ThroughputLimited => {
                "Throughput limited. Adding more factories will reduce latency.\n\
                 Consider:\n\
                 - Increase factory count to match avg parallelism\n\
                 - Current utilization suggests room for more factories\n\
                 - Balance cost vs latency based on application requirements"
            }
        }
    }

    /// Compare with static estimate (Sprint 51 style)
    pub fn static_comparison(&self) -> StaticComparison {
        // Static estimate: assumes serial magic state production
        let static_cycles = self.t_count * self.average_latency as usize;

        let speedup = static_cycles as f64 / self.schedule_cycles as f64;

        StaticComparison {
            static_cycles,
            dynamic_cycles: self.schedule_cycles,
            speedup,
            static_qubit_hours: (static_cycles as u64 * self.total_qubits as u64) as f64
                / 1e6
                / 3600.0,
            dynamic_qubit_hours: self.qubit_hours,
        }
    }

    /// Convert to JSON (for reporting)
    #[cfg(feature = "config")]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Comparison with static estimate
#[derive(Clone, Debug)]
pub struct StaticComparison {
    pub static_cycles: usize,
    pub dynamic_cycles: usize,
    pub speedup: f64,
    pub static_qubit_hours: f64,
    pub dynamic_qubit_hours: f64,
}

impl StaticComparison {
    pub fn report(&self) -> String {
        format!(
            "=== Static vs Dynamic Comparison ===\n\
             Static Estimate: {} cycles ({:.2} qubit-hours)\n\
             Dynamic Estimate: {} cycles ({:.2} qubit-hours)\n\
             Speedup: {:.2}x\n\
             \n\
             Explanation:\n\
             Static assumes serial magic state production (Sprint 51 style).\n\
             Dynamic uses parallel factories with realistic scheduling (Sprint 63).",
            self.static_cycles,
            self.static_qubit_hours,
            self.dynamic_cycles,
            self.dynamic_qubit_hours,
            self.speedup
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
    fn test_dynamic_estimator() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2), QGate::T(0)];

        let ft_config = FaultTolerantConfig::nisq_era();
        let mut estimator = DynamicResourceEstimator::new(circuit, ft_config);

        let estimate = estimator.estimate();

        assert_eq!(estimate.t_count, 4);
        assert!(estimate.schedule_cycles > 0);
        assert!(estimate.total_qubits > 0);
        assert!(estimate.qubit_hours > 0.0);
    }

    #[test]
    fn test_ft_config() {
        let config = FaultTolerantConfig::nisq_era();
        assert_eq!(config.code_distance, 3);

        let config = FaultTolerantConfig::fault_tolerant();
        assert_eq!(config.code_distance, 17);
    }

    #[test]
    fn test_static_comparison() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(0)];

        let ft_config = FaultTolerantConfig::nisq_era();
        let mut estimator = DynamicResourceEstimator::new(circuit, ft_config);

        let estimate = estimator.estimate();
        let comparison = estimate.static_comparison();

        assert!(comparison.static_cycles > 0);
        assert!(comparison.dynamic_cycles > 0);
        assert!(comparison.speedup >= 1.0); // Dynamic should be at least as good
    }
}
