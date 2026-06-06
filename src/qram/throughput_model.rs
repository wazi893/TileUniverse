// =============================================================================
// Throughput Model - Latency and Throughput Analysis
// =============================================================================
// Sprint 63.0: Magic State Factory Scheduling
//
// Models factory throughput, latency, and bottleneck identification for
// realistic execution time estimates.

use crate::qram::{DependencyGraph, FactoryPool};
use crate::quantum::QGate;

// =============================================================================
// Throughput Model
// =============================================================================

/// Throughput model for factory pools
pub struct ThroughputModel {
    /// Factory pool
    pool: FactoryPool,

    /// Circuit T-gate distribution
    t_distribution: TDistribution,

    /// Communication latency (cycles to route magic state to circuit qubit)
    communication_latency: usize,

    /// Factory startup overhead (cycles to initialize factory)
    startup_overhead: usize,
}

impl ThroughputModel {
    /// Create throughput model from pool and circuit
    pub fn new(pool: FactoryPool, circuit: &[QGate]) -> Self {
        let t_distribution = TDistribution::from_circuit(circuit);

        ThroughputModel {
            pool,
            t_distribution,
            communication_latency: 10, // Default: 10 cycles
            startup_overhead: 5,       // Default: 5 cycles
        }
    }

    /// Set communication latency
    pub fn with_communication_latency(mut self, cycles: usize) -> Self {
        self.communication_latency = cycles;
        self
    }

    /// Set startup overhead
    pub fn with_startup_overhead(mut self, cycles: usize) -> Self {
        self.startup_overhead = cycles;
        self
    }

    /// Calculate steady-state throughput (magic states per cycle)
    pub fn steady_state_throughput(&self) -> f64 {
        // Sum throughput of all factories
        self.pool
            .all_factories()
            .iter()
            .map(|f| 1.0 / f.protocol.cycle_overhead as f64)
            .sum()
    }

    /// Calculate average latency (cycles from request to magic state ready)
    pub fn average_latency(&self) -> f64 {
        if self.pool.all_factories().is_empty() {
            return 0.0;
        }

        // Weighted average of protocol latencies + communication
        let total_protocols = self.pool.all_factories().len();
        let sum_latency: usize = self
            .pool
            .all_factories()
            .iter()
            .map(|f| f.protocol.cycle_overhead + self.communication_latency)
            .sum();

        sum_latency as f64 / total_protocols as f64
    }

    /// Estimate total execution time for circuit
    pub fn estimate_execution_time(&self) -> ExecutionEstimate {
        let n_factories = self.pool.all_factories().len();
        if n_factories == 0 {
            return ExecutionEstimate::default();
        }

        let _avg_parallelism = self.t_distribution.avg_per_layer;
        let factory_latency = self.average_latency();

        // Calculate bottleneck: max(T-depth limited, throughput limited)
        let time_depth_limited = self.t_distribution.t_depth as f64 * factory_latency;
        let time_throughput_limited =
            self.t_distribution.t_count as f64 / self.steady_state_throughput();

        let bottleneck_cycles = time_depth_limited.max(time_throughput_limited) as usize;

        // Add startup and communication overhead
        let total_cycles = bottleneck_cycles + self.startup_overhead;

        // Determine bottleneck type
        let bottleneck_type = if time_depth_limited > time_throughput_limited {
            BottleneckType::TDepthLimited
        } else {
            BottleneckType::ThroughputLimited
        };

        // Calculate utilization
        let factory_utilization = self.estimate_utilization(total_cycles);

        // Recommend factory count
        let recommended_factories = self.recommend_factory_count();

        ExecutionEstimate {
            total_cycles,
            bottleneck_cycles,
            bottleneck_type,
            factory_utilization,
            recommended_factories,
            steady_state_throughput: self.steady_state_throughput(),
            average_latency: factory_latency,
        }
    }

    fn estimate_utilization(&self, total_cycles: usize) -> f64 {
        if total_cycles == 0 {
            return 0.0;
        }

        let total_work = self.t_distribution.t_count * self.average_latency() as usize;
        let total_capacity = self.pool.all_factories().len() * total_cycles;

        if total_capacity == 0 {
            0.0
        } else {
            (total_work as f64 / total_capacity as f64).min(1.0)
        }
    }

    fn recommend_factory_count(&self) -> usize {
        // Recommend: ceil(avg_parallelism)
        self.t_distribution.avg_per_layer.ceil() as usize
    }

    /// Get T-distribution
    pub fn t_distribution(&self) -> &TDistribution {
        &self.t_distribution
    }

    /// Get factory pool
    pub fn pool(&self) -> &FactoryPool {
        &self.pool
    }
}

// =============================================================================
// T-Gate Distribution
// =============================================================================

/// T-gate distribution in circuit
#[derive(Clone, Debug)]
pub struct TDistribution {
    /// Total T-count
    pub t_count: usize,

    /// T-depth
    pub t_depth: usize,

    /// T-gates per layer (histogram)
    pub layer_histogram: Vec<usize>,

    /// Average T-gates per layer
    pub avg_per_layer: f64,

    /// Max T-gates in any layer
    pub max_per_layer: usize,
}

impl TDistribution {
    /// Build from circuit
    pub fn from_circuit(circuit: &[QGate]) -> Self {
        let dep_graph = DependencyGraph::from_circuit(circuit);

        let layer_histogram: Vec<usize> =
            dep_graph.layers.iter().map(|layer| layer.len()).collect();

        let t_count = layer_histogram.iter().sum();
        let t_depth = layer_histogram.len();
        let avg_per_layer = if t_depth > 0 {
            t_count as f64 / t_depth as f64
        } else {
            0.0
        };
        let max_per_layer = *layer_histogram.iter().max().unwrap_or(&0);

        TDistribution {
            t_count,
            t_depth,
            layer_histogram,
            avg_per_layer,
            max_per_layer,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> String {
        format!(
            "T-Count: {}, T-Depth: {}, Avg/Layer: {:.2}, Max/Layer: {}",
            self.t_count, self.t_depth, self.avg_per_layer, self.max_per_layer
        )
    }
}

// =============================================================================
// Execution Estimate
// =============================================================================

/// Execution time estimate
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ExecutionEstimate {
    /// Total execution cycles
    pub total_cycles: usize,

    /// Bottleneck cycles (critical path)
    pub bottleneck_cycles: usize,

    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,

    /// Factory utilization (0.0 to 1.0)
    pub factory_utilization: f64,

    /// Recommended factory count
    pub recommended_factories: usize,

    /// Steady-state throughput (magic states/cycle)
    pub steady_state_throughput: f64,

    /// Average latency per magic state (cycles)
    pub average_latency: f64,
}

impl ExecutionEstimate {
    /// Generate report
    pub fn report(&self) -> String {
        format!(
            "=== Execution Estimate ===\n\
             Total Cycles: {}\n\
             Bottleneck Cycles: {}\n\
             Bottleneck Type: {:?}\n\
             Factory Utilization: {:.1}%\n\
             Steady-State Throughput: {:.3} states/cycle\n\
             Average Latency: {:.1} cycles/state\n\
             Recommended Factories: {}\n\
             \n\
             Recommendation: {}",
            self.total_cycles,
            self.bottleneck_cycles,
            self.bottleneck_type,
            self.factory_utilization * 100.0,
            self.steady_state_throughput,
            self.average_latency,
            self.recommended_factories,
            self.recommendation()
        )
    }

    fn recommendation(&self) -> &str {
        match self.bottleneck_type {
            BottleneckType::TDepthLimited => {
                "T-depth limited. Adding more factories won't help significantly. \
                 Consider: T-depth optimization, gate reordering, or accept current latency."
            }
            BottleneckType::ThroughputLimited => {
                "Throughput limited. Adding more factories will reduce latency. \
                 Consider: Increase factory count to match avg parallelism."
            }
        }
    }

    /// Convert to milliseconds at given clock rate (MHz)
    pub fn to_milliseconds(&self, clock_mhz: f64) -> f64 {
        self.total_cycles as f64 / (clock_mhz * 1e3)
    }

    /// Convert to seconds at given clock rate (MHz)
    pub fn to_seconds(&self, clock_mhz: f64) -> f64 {
        self.total_cycles as f64 / (clock_mhz * 1e6)
    }
}

impl Default for ExecutionEstimate {
    fn default() -> Self {
        ExecutionEstimate {
            total_cycles: 0,
            bottleneck_cycles: 0,
            bottleneck_type: BottleneckType::TDepthLimited,
            factory_utilization: 0.0,
            recommended_factories: 1,
            steady_state_throughput: 0.0,
            average_latency: 0.0,
        }
    }
}

/// Type of bottleneck
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum BottleneckType {
    /// Limited by T-depth (sequential dependencies)
    TDepthLimited,

    /// Limited by factory throughput (not enough factories)
    ThroughputLimited,
}

// =============================================================================
// Scaling Analysis
// =============================================================================

/// Analyze scaling with different factory counts
pub struct ScalingAnalysis {
    /// Results for each factory count
    pub results: Vec<(usize, ExecutionEstimate)>,
}

impl ScalingAnalysis {
    /// Create new analysis
    pub fn new() -> Self {
        ScalingAnalysis {
            results: Vec::new(),
        }
    }

    /// Add result
    pub fn add_result(&mut self, n_factories: usize, estimate: ExecutionEstimate) {
        self.results.push((n_factories, estimate));
    }

    /// Find optimal factory count (best utilization with minimal cycles)
    pub fn optimal_factory_count(&self) -> Option<usize> {
        self.results
            .iter()
            .filter(|(_, est)| est.factory_utilization > 0.7) // At least 70% utilization
            .min_by_key(|(_, est)| est.total_cycles)
            .map(|(n, _)| *n)
    }

    /// Generate report
    pub fn report(&self) -> String {
        let mut lines = vec!["=== Scaling Analysis ===".to_string()];

        lines.push(format!(
            "{:<12} {:<12} {:<15} {:<12}",
            "Factories", "Cycles", "Utilization", "Bottleneck"
        ));
        lines.push("-".repeat(60));

        for (n_factories, estimate) in &self.results {
            lines.push(format!(
                "{:<12} {:<12} {:<15} {:<12}",
                n_factories,
                estimate.total_cycles,
                format!("{:.1}%", estimate.factory_utilization * 100.0),
                format!("{:?}", estimate.bottleneck_type)
            ));
        }

        if let Some(optimal) = self.optimal_factory_count() {
            lines.push(format!("\nOptimal Factory Count: {}", optimal));
        }

        lines.join("\n")
    }
}

impl Default for ScalingAnalysis {
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
    use crate::qram::DistillationProtocol;

    #[test]
    fn test_t_distribution() {
        let circuit = vec![
            QGate::T(0),
            QGate::T(1),
            QGate::T(2),
            QGate::T(0),
            QGate::T(1),
        ];

        let dist = TDistribution::from_circuit(&circuit);

        assert_eq!(dist.t_count, 5);
        assert!(dist.t_depth > 0);
        assert!(dist.avg_per_layer > 0.0);
    }

    #[test]
    fn test_throughput_model() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(0)];

        let protocol = DistillationProtocol::fifteen_to_one();
        let pool = FactoryPool::homogeneous(2, protocol);

        let model = ThroughputModel::new(pool, &circuit);

        let throughput = model.steady_state_throughput();
        assert!(throughput > 0.0);

        let latency = model.average_latency();
        assert!(latency > 0.0);
    }

    #[test]
    fn test_execution_estimate() {
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2), QGate::T(0)];

        let protocol = DistillationProtocol::fifteen_to_one();
        let pool = FactoryPool::homogeneous(2, protocol);

        let model = ThroughputModel::new(pool, &circuit);
        let estimate = model.estimate_execution_time();

        assert!(estimate.total_cycles > 0);
        assert!(estimate.factory_utilization >= 0.0);
        assert!(estimate.factory_utilization <= 1.0);
    }

    #[test]
    fn test_scaling_analysis() {
        let mut analysis = ScalingAnalysis::new();

        for n in 1..=5 {
            let estimate = ExecutionEstimate {
                total_cycles: 100 / n,
                factory_utilization: 0.8,
                ..Default::default()
            };
            analysis.add_result(n, estimate);
        }

        assert!(analysis.optimal_factory_count().is_some());
    }
}
