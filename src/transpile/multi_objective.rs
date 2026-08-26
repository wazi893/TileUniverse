// =============================================================================
// Multi-Objective Optimization - Pareto Frontier Exploration
// =============================================================================
// Sprint 69.0: Error-Aware Circuit Compilation
//
// Explores the trade-off space between error, latency, qubits, and cost
// to find Pareto-optimal compilation strategies.

use crate::qram::DistillationProtocol;
use crate::quantum::QGate;
use crate::transpile::{
    circuit_rewrite::RewriteStrategy,
    error_analysis::ErrorModel,
    error_aware_compiler::{
        CompilationResult, CompilerConfig, ErrorAwareCircuitCompiler, OptimizationObjective,
    },
};

// =============================================================================
// Pareto Point
// =============================================================================

/// A single point in the multi-objective solution space
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Compilation configuration that produced this point
    pub config: CompilerConfig,

    /// Compilation result
    pub result: CompilationResult,

    /// Error metric (lower is better)
    pub error: f64,

    /// Latency metric in cycles (lower is better)
    pub latency: usize,

    /// Qubit count (lower is better)
    pub qubits: usize,

    /// Cost metric in qubit-cycles (lower is better)
    pub cost: f64,
}

impl ParetoPoint {
    /// Create Pareto point from compilation result
    pub fn from_result(config: CompilerConfig, result: CompilationResult) -> Self {
        let error = result.error_propagation.final_error;
        let latency = result.protocols.total_cost;
        let qubits = result.error_propagation.num_qubits;
        let cost = (qubits as f64) * (latency as f64);

        ParetoPoint {
            config,
            result,
            error,
            latency,
            qubits,
            cost,
        }
    }

    /// Check if this point dominates another
    /// A dominates B if A is better or equal on all objectives and strictly better on at least one
    pub fn dominates(&self, other: &ParetoPoint) -> bool {
        let better_or_equal = self.error <= other.error
            && self.latency <= other.latency
            && self.qubits <= other.qubits
            && self.cost <= other.cost;

        let strictly_better = self.error < other.error
            || self.latency < other.latency
            || self.qubits < other.qubits
            || self.cost < other.cost;

        better_or_equal && strictly_better
    }

    /// Check if point is dominated by any in a set
    pub fn is_dominated_by_any(&self, points: &[ParetoPoint]) -> bool {
        points.iter().any(|p| p.dominates(self))
    }

    /// Get normalized score for single objective
    pub fn objective_score(&self, objective: &OptimizationObjective) -> f64 {
        match objective {
            OptimizationObjective::MinimizeError => self.error,
            OptimizationObjective::MinimizeLatency => self.latency as f64,
            OptimizationObjective::MinimizeQubits => self.qubits as f64,
            OptimizationObjective::MinimizeCost => self.cost,
            OptimizationObjective::Balanced => {
                // Balanced: geometric mean of all normalized objectives
                // (requires normalization context)
                self.error
            }
        }
    }
}

// =============================================================================
// Pareto Frontier
// =============================================================================

/// Collection of non-dominated solutions
#[derive(Clone, Debug)]
pub struct ParetoFrontier {
    /// Non-dominated points
    pub points: Vec<ParetoPoint>,

    /// Total points evaluated
    pub total_evaluated: usize,

    /// Normalization factors (for visualization)
    pub normalization: NormalizationFactors,
}

#[derive(Clone, Debug)]
pub struct NormalizationFactors {
    pub min_error: f64,
    pub max_error: f64,
    pub min_latency: usize,
    pub max_latency: usize,
    pub min_qubits: usize,
    pub max_qubits: usize,
    pub min_cost: f64,
    pub max_cost: f64,
}

impl ParetoFrontier {
    /// Create empty frontier
    pub fn new() -> Self {
        ParetoFrontier {
            points: Vec::new(),
            total_evaluated: 0,
            normalization: NormalizationFactors {
                min_error: f64::MAX,
                max_error: 0.0,
                min_latency: usize::MAX,
                max_latency: 0,
                min_qubits: usize::MAX,
                max_qubits: 0,
                min_cost: f64::MAX,
                max_cost: 0.0,
            },
        }
    }

    /// Add point to frontier (if not dominated)
    pub fn add_point(&mut self, point: ParetoPoint) {
        self.total_evaluated += 1;

        // Update normalization factors
        self.normalization.min_error = self.normalization.min_error.min(point.error);
        self.normalization.max_error = self.normalization.max_error.max(point.error);
        self.normalization.min_latency = self.normalization.min_latency.min(point.latency);
        self.normalization.max_latency = self.normalization.max_latency.max(point.latency);
        self.normalization.min_qubits = self.normalization.min_qubits.min(point.qubits);
        self.normalization.max_qubits = self.normalization.max_qubits.max(point.qubits);
        self.normalization.min_cost = self.normalization.min_cost.min(point.cost);
        self.normalization.max_cost = self.normalization.max_cost.max(point.cost);

        // Check if dominated by existing points
        if point.is_dominated_by_any(&self.points) {
            return;
        }

        // Remove points dominated by new point
        self.points.retain(|p| !point.dominates(p));

        // Add new point
        self.points.push(point);
    }

    /// Get point with best score for objective
    pub fn best_for_objective(&self, objective: &OptimizationObjective) -> Option<&ParetoPoint> {
        self.points.iter().min_by(|a, b| {
            a.objective_score(objective)
                .partial_cmp(&b.objective_score(objective))
                .unwrap()
        })
    }

    /// Filter points by constraints
    pub fn filter_by_constraints(&self, constraints: &Constraints) -> Vec<&ParetoPoint> {
        self.points
            .iter()
            .filter(|p| constraints.satisfied_by(p))
            .collect()
    }

    /// Get normalized metrics for point
    pub fn normalize(&self, point: &ParetoPoint) -> NormalizedMetrics {
        let error = self.normalize_value(
            point.error,
            self.normalization.min_error,
            self.normalization.max_error,
        );

        let latency = self.normalize_value(
            point.latency as f64,
            self.normalization.min_latency as f64,
            self.normalization.max_latency as f64,
        );

        let qubits = self.normalize_value(
            point.qubits as f64,
            self.normalization.min_qubits as f64,
            self.normalization.max_qubits as f64,
        );

        let cost = self.normalize_value(
            point.cost,
            self.normalization.min_cost,
            self.normalization.max_cost,
        );

        NormalizedMetrics {
            error,
            latency,
            qubits,
            cost,
        }
    }

    fn normalize_value(&self, value: f64, min: f64, max: f64) -> f64 {
        if (max - min).abs() < 1e-10 {
            0.5
        } else {
            (value - min) / (max - min)
        }
    }

    /// Generate comprehensive report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=== Pareto Frontier Report ===\n".to_string());
        lines.push(format!("Total points evaluated: {}", self.total_evaluated));
        lines.push(format!("Non-dominated solutions: {}", self.points.len()));
        lines.push(String::new());

        if self.points.is_empty() {
            lines.push("No solutions found.".to_string());
            return lines.join("\n");
        }

        // Range summary
        lines.push("Objective Ranges:".to_string());
        lines.push(format!(
            "  Error: {:.2e} - {:.2e}",
            self.normalization.min_error, self.normalization.max_error
        ));
        lines.push(format!(
            "  Latency: {} - {} cycles",
            self.normalization.min_latency, self.normalization.max_latency
        ));
        lines.push(format!(
            "  Qubits: {} - {}",
            self.normalization.min_qubits, self.normalization.max_qubits
        ));
        lines.push(format!(
            "  Cost: {:.0} - {:.0} qubit-cycles",
            self.normalization.min_cost, self.normalization.max_cost
        ));
        lines.push(String::new());

        // Best for each objective
        lines.push("Best Solutions:".to_string());

        if let Some(best_error) = self.best_for_objective(&OptimizationObjective::MinimizeError) {
            lines.push(format!(
                "  Lowest Error: {:.2e} (latency: {}, qubits: {})",
                best_error.error, best_error.latency, best_error.qubits
            ));
        }

        if let Some(best_latency) = self.best_for_objective(&OptimizationObjective::MinimizeLatency)
        {
            lines.push(format!(
                "  Lowest Latency: {} cycles (error: {:.2e}, qubits: {})",
                best_latency.latency, best_latency.error, best_latency.qubits
            ));
        }

        if let Some(best_qubits) = self.best_for_objective(&OptimizationObjective::MinimizeQubits) {
            lines.push(format!(
                "  Fewest Qubits: {} (error: {:.2e}, latency: {})",
                best_qubits.qubits, best_qubits.error, best_qubits.latency
            ));
        }

        if let Some(best_cost) = self.best_for_objective(&OptimizationObjective::MinimizeCost) {
            lines.push(format!(
                "  Lowest Cost: {:.0} qubit-cycles (error: {:.2e})",
                best_cost.cost, best_cost.error
            ));
        }

        lines.push(String::new());

        // All points
        lines.push("All Pareto-Optimal Solutions:".to_string());
        for (i, point) in self.points.iter().enumerate() {
            lines.push(format!(
                "  [{}] Error: {:.2e}, Latency: {}, Qubits: {}, Cost: {:.0}",
                i + 1,
                point.error,
                point.latency,
                point.qubits,
                point.cost
            ));
        }

        lines.join("\n")
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct NormalizedMetrics {
    pub error: f64,
    pub latency: f64,
    pub qubits: f64,
    pub cost: f64,
}

// =============================================================================
// Constraints
// =============================================================================

/// User-defined constraints for solution filtering
#[derive(Clone, Debug)]
pub struct Constraints {
    pub max_error: Option<f64>,
    pub max_latency: Option<usize>,
    pub max_qubits: Option<usize>,
    pub max_cost: Option<f64>,
}

impl Constraints {
    /// Create no constraints
    pub fn none() -> Self {
        Constraints {
            max_error: None,
            max_latency: None,
            max_qubits: None,
            max_cost: None,
        }
    }

    /// Check if point satisfies all constraints
    pub fn satisfied_by(&self, point: &ParetoPoint) -> bool {
        if let Some(max_error) = self.max_error
            && point.error > max_error
        {
            return false;
        }

        if let Some(max_latency) = self.max_latency
            && point.latency > max_latency
        {
            return false;
        }

        if let Some(max_qubits) = self.max_qubits
            && point.qubits > max_qubits
        {
            return false;
        }

        if let Some(max_cost) = self.max_cost
            && point.cost > max_cost
        {
            return false;
        }

        true
    }

    /// Set error constraint
    pub fn with_max_error(mut self, max_error: f64) -> Self {
        self.max_error = Some(max_error);
        self
    }

    /// Set latency constraint
    pub fn with_max_latency(mut self, max_latency: usize) -> Self {
        self.max_latency = Some(max_latency);
        self
    }

    /// Set qubit constraint
    pub fn with_max_qubits(mut self, max_qubits: usize) -> Self {
        self.max_qubits = Some(max_qubits);
        self
    }

    /// Set cost constraint
    pub fn with_max_cost(mut self, max_cost: f64) -> Self {
        self.max_cost = Some(max_cost);
        self
    }
}

// =============================================================================
// Multi-Objective Optimizer
// =============================================================================

/// Configuration space to explore
#[derive(Clone, Debug)]
pub struct ConfigurationSpace {
    /// Error budgets to try
    pub error_budgets: Vec<f64>,

    /// Rewrite strategy combinations to try
    pub strategy_combinations: Vec<Vec<RewriteStrategy>>,

    /// Protocol sets to try
    pub protocol_sets: Vec<Vec<DistillationProtocol>>,

    /// Input error rates to try
    pub input_errors: Vec<f64>,
}

impl ConfigurationSpace {
    /// Create default exploration space
    pub fn default_exploration() -> Self {
        ConfigurationSpace {
            error_budgets: vec![1e-2, 1e-3, 1e-4, 1e-5], // 4 budgets
            strategy_combinations: vec![
                vec![RewriteStrategy::TDepthMinimization],
                vec![
                    RewriteStrategy::GateCancellation,
                    RewriteStrategy::CliffordReduction,
                ],
                vec![
                    RewriteStrategy::TDepthMinimization,
                    RewriteStrategy::GateCancellation,
                    RewriteStrategy::CliffordReduction,
                ],
            ], // 3 combinations
            protocol_sets: vec![
                vec![DistillationProtocol::ten_to_two()],
                vec![
                    DistillationProtocol::ten_to_two(),
                    DistillationProtocol::fifteen_to_one(),
                ],
            ], // 2 protocol sets
            input_errors: vec![1e-3],                    // 1 input error - 4×3×2×1 = 24
        }
    }

    /// Create minimal exploration (fast)
    pub fn minimal() -> Self {
        ConfigurationSpace {
            error_budgets: vec![1e-3, 1e-4], // Sufficient budgets for test circuit
            strategy_combinations: vec![vec![RewriteStrategy::TDepthMinimization]],
            protocol_sets: vec![vec![
                DistillationProtocol::ten_to_two(),
                DistillationProtocol::fifteen_to_one(),
            ]],
            input_errors: vec![1e-3, 5e-4], // Two input error levels
        }
    }

    /// Create comprehensive exploration (slow, thorough)
    pub fn comprehensive() -> Self {
        ConfigurationSpace {
            error_budgets: vec![1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            strategy_combinations: vec![
                vec![RewriteStrategy::TDepthMinimization],
                vec![RewriteStrategy::GateCancellation],
                vec![RewriteStrategy::CliffordReduction],
                vec![
                    RewriteStrategy::GateCancellation,
                    RewriteStrategy::CliffordReduction,
                ],
                vec![
                    RewriteStrategy::TDepthMinimization,
                    RewriteStrategy::GateCancellation,
                ],
                vec![
                    RewriteStrategy::TDepthMinimization,
                    RewriteStrategy::GateCancellation,
                    RewriteStrategy::CliffordReduction,
                ],
            ],
            protocol_sets: vec![
                vec![DistillationProtocol::ten_to_two()],
                vec![DistillationProtocol::fifteen_to_one()],
                vec![
                    DistillationProtocol::ten_to_two(),
                    DistillationProtocol::fifteen_to_one(),
                ],
            ],
            input_errors: vec![1e-2, 5e-3, 1e-3],
        }
    }

    /// Count total configurations
    pub fn total_configs(&self) -> usize {
        self.error_budgets.len()
            * self.strategy_combinations.len()
            * self.protocol_sets.len()
            * self.input_errors.len()
    }
}

/// Multi-objective optimizer exploring Pareto frontier
pub struct MultiObjectiveOptimizer {
    /// Circuit to optimize
    circuit: Vec<QGate>,

    /// Configuration space to explore
    space: ConfigurationSpace,

    /// Base error model
    error_model: ErrorModel,
}

impl MultiObjectiveOptimizer {
    /// Create optimizer
    pub fn new(circuit: Vec<QGate>, space: ConfigurationSpace) -> Self {
        MultiObjectiveOptimizer {
            circuit,
            space,
            error_model: ErrorModel::depolarizing(1e-4),
        }
    }

    /// Set error model
    pub fn with_error_model(mut self, error_model: ErrorModel) -> Self {
        self.error_model = error_model;
        self
    }

    /// Explore Pareto frontier
    pub fn explore(&mut self) -> ParetoFrontier {
        let mut frontier = ParetoFrontier::new();

        // Iterate through configuration space
        for error_budget in &self.space.error_budgets {
            for strategies in &self.space.strategy_combinations {
                for protocols in &self.space.protocol_sets {
                    for input_error in &self.space.input_errors {
                        // Build configuration
                        let config = CompilerConfig {
                            error_budget: *error_budget,
                            resource_budget: None,
                            objective: OptimizationObjective::Balanced,
                            error_model: self.error_model.clone(),
                            rewrite_strategies: strategies.clone(),
                            available_protocols: protocols.clone(),
                            input_error: *input_error,
                        };

                        // Compile with this configuration
                        let mut compiler =
                            ErrorAwareCircuitCompiler::new(self.circuit.clone(), config.clone());
                        let result = compiler.compile();

                        // Add to frontier
                        let point = ParetoPoint::from_result(config, result);
                        frontier.add_point(point);
                    }
                }
            }
        }

        frontier
    }

    /// Explore with constraints
    pub fn explore_constrained(&mut self, constraints: Constraints) -> ParetoFrontier {
        let mut frontier = self.explore();

        // Filter to only constrained points
        frontier.points.retain(|p| constraints.satisfied_by(p));

        frontier
    }

    /// Find single best solution for objective
    pub fn optimize_single_objective(
        &mut self,
        objective: OptimizationObjective,
    ) -> Option<ParetoPoint> {
        let frontier = self.explore();
        frontier.best_for_objective(&objective).cloned()
    }
}

// =============================================================================
// Trade-off Analysis
// =============================================================================

/// Trade-off analysis between two objectives
#[derive(Clone, Debug)]
pub struct TradeoffAnalysis {
    /// Objective A (x-axis)
    pub objective_a: OptimizationObjective,

    /// Objective B (y-axis)
    pub objective_b: OptimizationObjective,

    /// Points in trade-off space
    pub points: Vec<(f64, f64)>,
}

impl TradeoffAnalysis {
    /// Create trade-off analysis from frontier
    pub fn from_frontier(
        frontier: &ParetoFrontier,
        objective_a: OptimizationObjective,
        objective_b: OptimizationObjective,
    ) -> Self {
        let points = frontier
            .points
            .iter()
            .map(|p| {
                let a = p.objective_score(&objective_a);
                let b = p.objective_score(&objective_b);
                (a, b)
            })
            .collect();

        TradeoffAnalysis {
            objective_a,
            objective_b,
            points,
        }
    }

    /// Generate simple text plot
    pub fn plot_text(&self, width: usize, height: usize) -> String {
        if self.points.is_empty() {
            return "No data points to plot.".to_string();
        }

        // Find ranges
        let min_x = self
            .points
            .iter()
            .map(|(x, _)| x)
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_x = self
            .points
            .iter()
            .map(|(x, _)| x)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_y = self
            .points
            .iter()
            .map(|(_, y)| y)
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_y = self
            .points
            .iter()
            .map(|(_, y)| y)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Create grid
        let mut grid = vec![vec![' '; width]; height];

        // Plot points
        for (x, y) in &self.points {
            let ix = ((x - min_x) / (max_x - min_x) * (width - 1) as f64) as usize;
            let iy = height - 1 - ((y - min_y) / (max_y - min_y) * (height - 1) as f64) as usize;
            grid[iy.min(height - 1)][ix.min(width - 1)] = '*';
        }

        // Convert to string
        let mut lines = Vec::new();
        lines.push(format!("{:?} vs {:?}", self.objective_b, self.objective_a));
        for row in grid {
            lines.push(row.iter().collect());
        }
        lines.push(format!("X: {:.2e} to {:.2e}", min_x, max_x));
        lines.push(format!("Y: {:.2e} to {:.2e}", min_y, max_y));

        lines.join("\n")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_circuit() -> Vec<QGate> {
        vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::T(0),
            QGate::T(1),
            QGate::CNot(0, 1),
            QGate::T(0),
        ]
    }

    #[test]
    fn test_pareto_dominance() {
        let config = CompilerConfig::new();

        // Point A: low error, high latency
        let result_a = CompilationResult {
            circuit: vec![],
            protocols: crate::transpile::adaptive_protocols::ProtocolAssignment {
                assignments: Default::default(),
                total_cost: 1000,
                expected_error: 1e-6,
                protocol_counts: Default::default(),
            },
            error_propagation: crate::transpile::error_analysis::ErrorPropagation {
                num_qubits: 2,
                num_layers: 1,
                error_state: vec![vec![1e-6, 1e-6]],
                final_error: 1e-6,
                total_error: 1e-5,
                error_by_gate_type: Default::default(),
            },
            rewrite_result: crate::transpile::circuit_rewrite::RewrittenCircuit {
                circuit: vec![],
                original_t_depth: 3,
                optimized_t_depth: 2,
                original_gate_count: 6,
                optimized_gate_count: 5,
                improvement_pct: 33.3,
                strategies_applied: vec![],
            },
            criticality: vec![],
            metrics: crate::transpile::error_aware_compiler::OptimizationMetrics {
                compilation_time_ms: 10,
                error_budget_utilization: 60.0,
                speedup_vs_baseline: None,
            },
        };

        // Point B: high error, low latency
        let mut result_b = result_a.clone();
        result_b.protocols.total_cost = 500;
        result_b.error_propagation.final_error = 1e-5;

        let point_a = ParetoPoint::from_result(config.clone(), result_a);
        let point_b = ParetoPoint::from_result(config, result_b);

        // Neither should dominate (trade-off)
        assert!(!point_a.dominates(&point_b));
        assert!(!point_b.dominates(&point_a));
    }

    #[test]
    fn test_pareto_frontier_basic() {
        let mut frontier = ParetoFrontier::new();

        let config = CompilerConfig::new();
        let base_result = CompilationResult {
            circuit: vec![],
            protocols: crate::transpile::adaptive_protocols::ProtocolAssignment {
                assignments: Default::default(),
                total_cost: 1000,
                expected_error: 1e-6,
                protocol_counts: Default::default(),
            },
            error_propagation: crate::transpile::error_analysis::ErrorPropagation {
                num_qubits: 2,
                num_layers: 1,
                error_state: vec![vec![1e-6, 1e-6]],
                final_error: 1e-6,
                total_error: 1e-5,
                error_by_gate_type: Default::default(),
            },
            rewrite_result: crate::transpile::circuit_rewrite::RewrittenCircuit {
                circuit: vec![],
                original_t_depth: 3,
                optimized_t_depth: 2,
                original_gate_count: 6,
                optimized_gate_count: 5,
                improvement_pct: 33.3,
                strategies_applied: vec![],
            },
            criticality: vec![],
            metrics: crate::transpile::error_aware_compiler::OptimizationMetrics {
                compilation_time_ms: 10,
                error_budget_utilization: 60.0,
                speedup_vs_baseline: None,
            },
        };

        // Add three non-dominated points
        let point1 = ParetoPoint::from_result(config.clone(), base_result.clone());
        frontier.add_point(point1);

        let mut result2 = base_result.clone();
        result2.protocols.total_cost = 500;
        result2.error_propagation.final_error = 1e-5;
        let point2 = ParetoPoint::from_result(config.clone(), result2);
        frontier.add_point(point2);

        let mut result3 = base_result;
        result3.protocols.total_cost = 2000;
        result3.error_propagation.final_error = 1e-7;
        let point3 = ParetoPoint::from_result(config, result3);
        frontier.add_point(point3);

        assert_eq!(frontier.points.len(), 3);
        assert_eq!(frontier.total_evaluated, 3);
    }

    #[test]
    fn test_constraints() {
        let constraints = Constraints::none()
            .with_max_error(1e-5)
            .with_max_latency(1000);

        let config = CompilerConfig::new();
        let result = CompilationResult {
            circuit: vec![],
            protocols: crate::transpile::adaptive_protocols::ProtocolAssignment {
                assignments: Default::default(),
                total_cost: 500,
                expected_error: 1e-6,
                protocol_counts: Default::default(),
            },
            error_propagation: crate::transpile::error_analysis::ErrorPropagation {
                num_qubits: 2,
                num_layers: 1,
                error_state: vec![vec![1e-6, 1e-6]],
                final_error: 1e-6,
                total_error: 1e-5,
                error_by_gate_type: Default::default(),
            },
            rewrite_result: crate::transpile::circuit_rewrite::RewrittenCircuit {
                circuit: vec![],
                original_t_depth: 3,
                optimized_t_depth: 2,
                original_gate_count: 6,
                optimized_gate_count: 5,
                improvement_pct: 33.3,
                strategies_applied: vec![],
            },
            criticality: vec![],
            metrics: crate::transpile::error_aware_compiler::OptimizationMetrics {
                compilation_time_ms: 10,
                error_budget_utilization: 60.0,
                speedup_vs_baseline: None,
            },
        };

        let point = ParetoPoint::from_result(config, result);
        assert!(constraints.satisfied_by(&point));
    }

    #[test]
    fn test_configuration_space() {
        let space = ConfigurationSpace::minimal();
        assert_eq!(space.total_configs(), 4); // 2 × 1 × 1 × 2

        let space = ConfigurationSpace::default_exploration();
        assert_eq!(space.total_configs(), 24); // 4 × 3 × 2 × 1
    }

    #[test]
    fn test_multi_objective_optimizer() {
        let circuit = test_circuit();
        let space = ConfigurationSpace::minimal();

        let mut optimizer = MultiObjectiveOptimizer::new(circuit, space);
        let frontier = optimizer.explore();

        assert!(!frontier.points.is_empty());
        assert_eq!(frontier.total_evaluated, 4);
    }

    #[test]
    fn test_best_for_objective() {
        let circuit = test_circuit();
        let space = ConfigurationSpace::minimal();

        let mut optimizer = MultiObjectiveOptimizer::new(circuit, space);
        let frontier = optimizer.explore();

        let best_error = frontier.best_for_objective(&OptimizationObjective::MinimizeError);
        let best_latency = frontier.best_for_objective(&OptimizationObjective::MinimizeLatency);

        assert!(best_error.is_some());
        assert!(best_latency.is_some());

        // Best error point should have lowest error
        if let Some(point) = best_error {
            assert!(frontier.points.iter().all(|p| point.error <= p.error));
        }
    }

    #[test]
    fn test_tradeoff_analysis() {
        let circuit = test_circuit();
        let space = ConfigurationSpace::minimal();

        let mut optimizer = MultiObjectiveOptimizer::new(circuit, space);
        let frontier = optimizer.explore();

        let tradeoff = TradeoffAnalysis::from_frontier(
            &frontier,
            OptimizationObjective::MinimizeError,
            OptimizationObjective::MinimizeLatency,
        );

        assert_eq!(tradeoff.points.len(), frontier.points.len());
    }
}
