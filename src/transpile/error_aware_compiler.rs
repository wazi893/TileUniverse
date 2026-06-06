// =============================================================================
// Error-Aware Circuit Compiler - Main Integration API
// =============================================================================
// Sprint 69.0: Error-Aware Circuit Compilation
//
// Coordinates error analysis, circuit rewriting, and adaptive protocol
// selection to produce optimized, error-budget-constrained circuits.

use crate::qram::DistillationProtocol;
use crate::quantum::QGate;
use crate::transpile::{
    adaptive_protocols::{AdaptiveProtocolSelector, ProtocolAssignment},
    circuit_rewrite::{CircuitRewriter, RewriteStrategy, RewrittenCircuit},
    error_analysis::{ErrorAnalyzer, ErrorModel, ErrorPropagation, GateCriticality},
};

// =============================================================================
// Optimization Objective
// =============================================================================

/// Optimization objective for compilation
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize total execution time (latency)
    MinimizeLatency,

    /// Minimize total qubits used
    MinimizeQubits,

    /// Minimize final error
    MinimizeError,

    /// Minimize cost (qubit-hours)
    MinimizeCost,

    /// Balance all objectives
    Balanced,
}

// =============================================================================
// Compiler Configuration
// =============================================================================

/// Configuration for error-aware compiler
#[derive(Clone, Debug)]
pub struct CompilerConfig {
    /// Error budget (total algorithm error)
    pub error_budget: f64,

    /// Resource budget (max qubits)
    pub resource_budget: Option<usize>,

    /// Optimization objective
    pub objective: OptimizationObjective,

    /// Error model
    pub error_model: ErrorModel,

    /// Rewrite strategies
    pub rewrite_strategies: Vec<RewriteStrategy>,

    /// Available distillation protocols
    pub available_protocols: Vec<DistillationProtocol>,

    /// Input magic state error rate
    pub input_error: f64,
}

impl CompilerConfig {
    /// Create default config
    pub fn new() -> Self {
        CompilerConfig {
            error_budget: 1e-6,
            resource_budget: None,
            objective: OptimizationObjective::Balanced,
            error_model: ErrorModel::depolarizing(1e-4),
            rewrite_strategies: vec![
                RewriteStrategy::GateCancellation,
                RewriteStrategy::CliffordReduction,
                RewriteStrategy::TDepthMinimization,
            ],
            available_protocols: vec![
                DistillationProtocol::ten_to_two(),
                DistillationProtocol::fifteen_to_one(),
            ],
            input_error: 1e-2,
        }
    }

    /// Set error budget
    pub fn with_error_budget(mut self, budget: f64) -> Self {
        self.error_budget = budget;
        self
    }

    /// Set resource budget
    pub fn with_resource_budget(mut self, budget: usize) -> Self {
        self.resource_budget = Some(budget);
        self
    }

    /// Set optimization objective
    pub fn with_objective(mut self, objective: OptimizationObjective) -> Self {
        self.objective = objective;
        self
    }

    /// Add rewrite strategy
    pub fn with_strategy(mut self, strategy: RewriteStrategy) -> Self {
        self.rewrite_strategies.push(strategy);
        self
    }

    /// Set available protocols
    pub fn with_protocols(mut self, protocols: Vec<DistillationProtocol>) -> Self {
        self.available_protocols = protocols;
        self
    }
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Compilation Result
// =============================================================================

/// Result of error-aware compilation
#[derive(Clone, Debug)]
pub struct CompilationResult {
    /// Optimized circuit
    pub circuit: Vec<QGate>,

    /// Protocol assignment
    pub protocols: ProtocolAssignment,

    /// Error propagation analysis
    pub error_propagation: ErrorPropagation,

    /// Rewrite result
    pub rewrite_result: RewrittenCircuit,

    /// Gate criticality scores
    pub criticality: Vec<GateCriticality>,

    /// Optimization metrics
    pub metrics: OptimizationMetrics,
}

impl CompilationResult {
    /// Generate comprehensive report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=== Error-Aware Compilation Report ===\n".to_string());

        // Circuit summary
        lines.push("Circuit:".to_string());
        lines.push(format!(
            "  Original: {} gates",
            self.rewrite_result.original_gate_count
        ));
        lines.push(format!(
            "  Optimized: {} gates",
            self.rewrite_result.optimized_gate_count
        ));
        lines.push(format!(
            "  T-depth: {} → {} ({:.1}% improvement)",
            self.rewrite_result.original_t_depth,
            self.rewrite_result.optimized_t_depth,
            self.rewrite_result.improvement_pct
        ));
        lines.push(String::new());

        // Error analysis
        lines.push("Error Analysis:".to_string());
        lines.push(format!(
            "  Final error: {:.2e}",
            self.error_propagation.final_error
        ));
        lines.push(format!(
            "  Total accumulated: {:.2e}",
            self.error_propagation.total_error
        ));
        lines.push(String::new());

        // Criticality
        let critical_count = self.criticality.iter().filter(|g| g.is_critical()).count();
        let moderate_count = self.criticality.iter().filter(|g| g.is_moderate()).count();
        let non_critical_count = self
            .criticality
            .iter()
            .filter(|g| g.is_non_critical())
            .count();

        lines.push("Gate Criticality:".to_string());
        lines.push(format!("  Critical: {} gates", critical_count));
        lines.push(format!("  Moderate: {} gates", moderate_count));
        lines.push(format!("  Non-critical: {} gates", non_critical_count));
        lines.push(String::new());

        // Protocol assignment
        lines.push("Protocol Assignment:".to_string());
        for (protocol, count) in self.protocols.protocol_distribution() {
            lines.push(format!("  {}: {} T-gates", protocol, count));
        }
        lines.push(format!(
            "  Total cost: {} cycles",
            self.protocols.total_cost
        ));
        lines.push(String::new());

        // Metrics
        lines.push("Performance Metrics:".to_string());
        lines.push(format!(
            "  Success probability: {:.2}%",
            self.success_probability() * 100.0
        ));
        lines.push(format!(
            "  Error budget utilization: {:.1}%",
            self.metrics.error_budget_utilization
        ));
        lines.push(String::new());

        // Recommendations
        lines.push(self.recommendation());

        lines.join("\n")
    }

    /// Calculate success probability
    pub fn success_probability(&self) -> f64 {
        1.0 - self.error_propagation.final_error
    }

    /// Generate recommendation
    fn recommendation(&self) -> String {
        let mut rec = vec!["Recommendations:".to_string()];

        if self.metrics.error_budget_utilization > 95.0 {
            rec.push("  ⚠ Error budget nearly exhausted. Consider:".to_string());
            rec.push("    - Increasing error budget".to_string());
            rec.push("    - Using higher-accuracy protocols".to_string());
        } else if self.metrics.error_budget_utilization < 60.0 {
            rec.push("  ✓ Good error margin. Can potentially:".to_string());
            rec.push("    - Use faster protocols for non-critical gates".to_string());
            rec.push("    - Reduce compilation time".to_string());
        } else {
            rec.push("  ✓ Error budget well-utilized".to_string());
        }

        if self.rewrite_result.improvement_pct > 30.0 {
            rec.push("  ✓ Excellent T-depth reduction".to_string());
        } else if self.rewrite_result.improvement_pct < 10.0 {
            rec.push(
                "  ⚠ Limited T-depth improvement. Circuit may be already optimized.".to_string(),
            );
        }

        rec.join("\n")
    }
}

// =============================================================================
// Optimization Metrics
// =============================================================================

/// Metrics from optimization
#[derive(Clone, Debug)]
pub struct OptimizationMetrics {
    /// Compilation time (milliseconds)
    pub compilation_time_ms: u128,

    /// Error budget utilization (percentage)
    pub error_budget_utilization: f64,

    /// Speedup vs baseline (if available)
    pub speedup_vs_baseline: Option<f64>,
}

// =============================================================================
// Error-Aware Circuit Compiler
// =============================================================================

/// Main error-aware circuit compiler
pub struct ErrorAwareCircuitCompiler {
    /// Original circuit
    circuit: Vec<QGate>,

    /// Configuration
    config: CompilerConfig,
}

impl ErrorAwareCircuitCompiler {
    /// Create compiler
    pub fn new(circuit: Vec<QGate>, config: CompilerConfig) -> Self {
        ErrorAwareCircuitCompiler { circuit, config }
    }

    /// Compile circuit with error awareness
    pub fn compile(&mut self) -> CompilationResult {
        let start = std::time::Instant::now();

        // Phase 1: Error Analysis
        let mut analyzer =
            ErrorAnalyzer::new(self.circuit.clone(), self.config.error_model.clone());
        let error_propagation = analyzer.propagate_errors(1e-10);
        let criticality = analyzer.identify_critical_gates();

        // Phase 2: Circuit Rewriting
        let mut rewriter = CircuitRewriter::new(self.circuit.clone());
        for strategy in &self.config.rewrite_strategies {
            rewriter.add_strategy(strategy.clone());
        }
        let rewrite_result = rewriter.rewrite();
        let optimized_circuit = rewrite_result.circuit.clone();

        // Phase 3: Adaptive Protocol Selection
        let mut protocol_selector = AdaptiveProtocolSelector::new(
            self.config.available_protocols.clone(),
            criticality.clone(),
        );
        protocol_selector.set_error_budget(self.config.error_budget);
        protocol_selector.set_input_error(self.config.input_error);

        let protocols = protocol_selector.select_protocols();

        // Calculate metrics
        let compilation_time_ms = start.elapsed().as_millis();
        let error_budget_utilization = if self.config.error_budget > 0.0 {
            (error_propagation.final_error / self.config.error_budget) * 100.0
        } else {
            0.0
        };

        let metrics = OptimizationMetrics {
            compilation_time_ms,
            error_budget_utilization,
            speedup_vs_baseline: None,
        };

        CompilationResult {
            circuit: optimized_circuit,
            protocols,
            error_propagation,
            rewrite_result,
            criticality,
            metrics,
        }
    }

    /// Analyze errors only (no optimization)
    pub fn analyze_errors(&self) -> ErrorPropagation {
        let mut analyzer =
            ErrorAnalyzer::new(self.circuit.clone(), self.config.error_model.clone());
        analyzer.propagate_errors(1e-10)
    }

    /// Rewrite circuit only (no protocol selection)
    pub fn rewrite_circuit(&mut self) -> RewrittenCircuit {
        let mut rewriter = CircuitRewriter::new(self.circuit.clone());
        for strategy in &self.config.rewrite_strategies {
            rewriter.add_strategy(strategy.clone());
        }
        rewriter.rewrite()
    }

    /// Select protocols only (no rewriting)
    pub fn select_protocols(&mut self) -> ProtocolAssignment {
        let mut analyzer =
            ErrorAnalyzer::new(self.circuit.clone(), self.config.error_model.clone());
        analyzer.propagate_errors(1e-10);
        let criticality = analyzer.identify_critical_gates();

        let mut selector =
            AdaptiveProtocolSelector::new(self.config.available_protocols.clone(), criticality);
        selector.set_error_budget(self.config.error_budget);
        selector.set_input_error(self.config.input_error);

        selector.select_protocols()
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
    fn test_compiler_basic() {
        let circuit = test_circuit();
        let mut config = CompilerConfig::new();
        config.error_budget = 1e-3; // Sufficient for test circuit
        config.input_error = 1e-3; // Lower input error

        let mut compiler = ErrorAwareCircuitCompiler::new(circuit, config);
        let result = compiler.compile();

        assert!(!result.circuit.is_empty());
        // Protocol expected error should be within budget (not raw propagation error)
        assert!(result.protocols.expected_error <= 1e-3);
        assert!(!result.protocols.assignments.is_empty());
    }

    #[test]
    fn test_error_analysis_only() {
        let circuit = test_circuit();
        let config = CompilerConfig::new();

        let compiler = ErrorAwareCircuitCompiler::new(circuit, config);
        let propagation = compiler.analyze_errors();

        assert!(propagation.final_error > 0.0);
        assert_eq!(propagation.num_qubits, 2);
    }

    #[test]
    fn test_circuit_rewrite_only() {
        let circuit = test_circuit();
        let config = CompilerConfig::new();

        let mut compiler = ErrorAwareCircuitCompiler::new(circuit, config);
        let rewrite = compiler.rewrite_circuit();

        assert!(rewrite.optimized_gate_count <= rewrite.original_gate_count);
    }

    #[test]
    fn test_protocol_selection_only() {
        let circuit = test_circuit();
        let mut config = CompilerConfig::new();
        config.error_budget = 1e-3; // Sufficient budget
        config.input_error = 1e-3; // Lower input error

        let mut compiler = ErrorAwareCircuitCompiler::new(circuit, config);
        let protocols = compiler.select_protocols();

        // Should assign protocols to T-gates
        assert!(protocols.total_gates() > 0);
        assert!(protocols.total_cost > 0);
    }

    #[test]
    fn test_success_probability() {
        let circuit = test_circuit();
        let config = CompilerConfig::new().with_error_budget(1e-6);

        let mut compiler = ErrorAwareCircuitCompiler::new(circuit, config);
        let result = compiler.compile();

        let prob = result.success_probability();
        assert!(prob > 0.99); // Should be high for small circuit
        assert!(prob <= 1.0);
    }

    #[test]
    fn test_different_objectives() {
        let circuit = test_circuit();

        let objectives = vec![
            OptimizationObjective::MinimizeLatency,
            OptimizationObjective::MinimizeError,
            OptimizationObjective::Balanced,
        ];

        for objective in objectives {
            let config = CompilerConfig::new()
                .with_error_budget(1e-6)
                .with_objective(objective);

            let mut compiler = ErrorAwareCircuitCompiler::new(circuit.clone(), config);
            let result = compiler.compile();

            assert!(!result.circuit.is_empty());
        }
    }
}
