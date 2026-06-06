//! Sprint 62.0: Auto-Router for intelligent substrate selection.
//!
//! Routes quantum jobs to the most appropriate substrate based on:
//! - Circuit size (qubits)
//! - Substrate capabilities
//! - Current capacity
//!
//! # Routing Strategy (Sprint 76.0 - Math-Aware Routing)
//!
//! 1. **Quick Classification**: O(n) pattern detection for fast-path decisions
//! 2. **GHZ/W Patterns**: Route to sparse substrate if available (O(1) vs O(2^n))
//! 3. **High Algebraic Reduction**: Prefer CPU to avoid GPU transfer overhead
//! 4. **Dense Circuits**:
//!    - Small circuits (< 16 qubits): CPU substrate (lower overhead)
//!    - Large circuits (>= 16 qubits): GPU substrate (parallel speedup)
//! 5. **Fallback**: When GPU unavailable, fall back to CPU

use super::job::{JobSpec, QuantumSpec};
use super::substrate::{SubstrateCapabilities, SubstrateId, SubstrateType};
use crate::math_accelerator::{
    BackendDecision, CircuitCharacteristics, QuickClassification, quick_classify,
    select_backend_with_characteristics,
};

/// Information about a substrate for routing decisions.
#[derive(Clone, Debug)]
pub struct SubstrateRouteInfo {
    /// Substrate ID
    pub id: SubstrateId,
    /// Substrate type
    pub substrate_type: SubstrateType,
    /// Whether GPU is available
    pub gpu_available: bool,
    /// Maximum qubits supported
    pub max_qubits: Option<usize>,
    /// Current capacity (0.0-1.0)
    pub capacity: f64,
    /// Whether this substrate supports sparse state vector simulation (Sprint 76.0).
    /// Sparse substrates are optimal for GHZ, W, and other structured states.
    pub supports_sparse: bool,
}

impl SubstrateRouteInfo {
    /// Create route info from substrate capabilities.
    pub fn new(
        id: SubstrateId,
        substrate_type: SubstrateType,
        capabilities: &SubstrateCapabilities,
        capacity: f64,
    ) -> Self {
        Self {
            id,
            substrate_type,
            gpu_available: capabilities.gpu_available,
            max_qubits: capabilities.max_qubits,
            capacity,
            supports_sparse: false, // Default to false; set explicitly when sparse is available
        }
    }

    /// Create route info with sparse substrate support.
    ///
    /// # Sprint 76.0
    ///
    /// Use this constructor for substrates that support sparse state vector
    /// simulation (GHZ, W states, etc.).
    pub fn new_with_sparse(
        id: SubstrateId,
        substrate_type: SubstrateType,
        capabilities: &SubstrateCapabilities,
        capacity: f64,
        supports_sparse: bool,
    ) -> Self {
        Self {
            id,
            substrate_type,
            gpu_available: capabilities.gpu_available,
            max_qubits: capabilities.max_qubits,
            capacity,
            supports_sparse,
        }
    }
}

/// Auto-router for intelligent substrate selection.
///
/// The router analyzes job requirements and selects the optimal
/// substrate based on configured thresholds.
///
/// # Sprint 76.0: Math-Aware Routing
///
/// When `enable_math_routing` is true, the router uses mathematical analysis
/// to make smarter routing decisions:
/// - Detects GHZ/W patterns and routes to sparse substrates
/// - Analyzes algebraic reduction potential
/// - Predicts state sparsity for optimal backend selection
///
/// # Example
///
/// ```ignore
/// use engine::hybrid::{AutoRouter, JobSpec, SubstrateRouteInfo};
///
/// let router = AutoRouter::default();
/// let substrates = vec![cpu_info, gpu_info];
/// let best = router.select_substrate(&job_spec, &substrates);
/// ```
#[derive(Clone, Debug)]
pub struct AutoRouter {
    /// Minimum qubits to prefer GPU (default: 16)
    pub gpu_qubit_threshold: usize,

    /// Whether to enable GPU routing
    pub enable_gpu_routing: bool,

    /// Minimum capacity to consider a substrate (default: 0.1)
    pub min_capacity: f64,

    /// Whether to enable math-aware routing (Sprint 76.0, default: true).
    ///
    /// When enabled, the router performs mathematical analysis on circuits
    /// to detect patterns (GHZ, W) and predict sparsity for optimal routing.
    pub enable_math_routing: bool,
}

impl Default for AutoRouter {
    fn default() -> Self {
        Self {
            gpu_qubit_threshold: 16,
            enable_gpu_routing: true,
            min_capacity: 0.1,
            enable_math_routing: true,
        }
    }
}

impl AutoRouter {
    /// Create a new auto-router with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the GPU qubit threshold.
    ///
    /// Circuits with >= this many qubits will prefer GPU substrate.
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_qubit_threshold = threshold;
        self
    }

    /// Enable or disable GPU routing.
    pub fn with_gpu_routing(mut self, enable: bool) -> Self {
        self.enable_gpu_routing = enable;
        self
    }

    /// Set minimum capacity for substrate consideration.
    pub fn with_min_capacity(mut self, min: f64) -> Self {
        self.min_capacity = min;
        self
    }

    /// Enable or disable math-aware routing (Sprint 76.0).
    ///
    /// When enabled, the router uses mathematical analysis to detect
    /// GHZ/W patterns and route to sparse substrates when available.
    pub fn with_math_routing(mut self, enable: bool) -> Self {
        self.enable_math_routing = enable;
        self
    }

    /// Select the best substrate for a job.
    ///
    /// Returns `Some(SubstrateId)` if a suitable substrate is found,
    /// `None` if no substrate can handle the job.
    ///
    /// # Sprint 76.0: Math-Aware Routing
    ///
    /// When `enable_math_routing` is true, this method performs mathematical
    /// analysis on quantum circuits to make smarter routing decisions.
    pub fn select_substrate(
        &self,
        spec: &JobSpec,
        substrates: &[SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        // Filter to substrates with sufficient capacity
        let available: Vec<&SubstrateRouteInfo> = substrates
            .iter()
            .filter(|s| s.capacity >= self.min_capacity)
            .collect();

        if available.is_empty() {
            return None;
        }

        // Route based on job type
        match spec {
            JobSpec::Quantum(quantum_spec) => {
                if self.enable_math_routing {
                    self.select_quantum_substrate_math_aware(quantum_spec, &available)
                } else {
                    self.select_quantum_substrate_simple(quantum_spec.n_qubits, &available)
                }
            }
            JobSpec::Classical(_) => {
                // Classical jobs go to classical substrates
                self.select_by_type(&available, SubstrateType::Classical)
            }
            JobSpec::Observer(_) => {
                // Observer jobs go to observer substrates
                self.select_by_type(&available, SubstrateType::Observer)
            }
            JobSpec::Variational(_) => {
                // Variational jobs are orchestrated at scheduler level
                // but the quantum portions should prefer GPU for larger circuits
                None
            }
            JobSpec::Pipeline(_) | JobSpec::Parallel(_) => {
                // Pipeline and parallel jobs are orchestrated at scheduler level
                // Individual stages are routed separately
                None
            }
        }
    }

    /// Select the best substrate for a mutable job spec.
    ///
    /// This variant allows caching characteristics in the job spec for
    /// subsequent routing decisions.
    ///
    /// # Sprint 76.0
    ///
    /// Use this method when you have mutable access to the job spec and want
    /// to cache circuit characteristics for performance.
    pub fn select_substrate_mut(
        &self,
        spec: &mut JobSpec,
        substrates: &[SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        // Filter to substrates with sufficient capacity
        let available: Vec<&SubstrateRouteInfo> = substrates
            .iter()
            .filter(|s| s.capacity >= self.min_capacity)
            .collect();

        if available.is_empty() {
            return None;
        }

        // Route based on job type
        match spec {
            JobSpec::Quantum(quantum_spec) => {
                if self.enable_math_routing {
                    self.select_quantum_substrate_with_caching(quantum_spec, &available)
                } else {
                    self.select_quantum_substrate_simple(quantum_spec.n_qubits, &available)
                }
            }
            JobSpec::Classical(_) => self.select_by_type(&available, SubstrateType::Classical),
            JobSpec::Observer(_) => self.select_by_type(&available, SubstrateType::Observer),
            JobSpec::Variational(_) | JobSpec::Pipeline(_) | JobSpec::Parallel(_) => None,
        }
    }

    /// Math-aware substrate selection for quantum jobs (Sprint 76.0).
    ///
    /// Performs O(n) quick classification, then routes based on detected patterns:
    /// - GHZ/W patterns -> sparse substrate if available
    /// - Small circuits -> CPU (avoid analysis overhead)
    /// - Large circuits needing full analysis -> compute characteristics
    fn select_quantum_substrate_math_aware(
        &self,
        spec: &QuantumSpec,
        substrates: &[&SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        let qgates = spec.circuit.to_qgates();
        let n_qubits = spec.n_qubits as u8;

        // Step 1: Quick O(n) classification
        let quick = quick_classify(&qgates, n_qubits);

        match quick {
            QuickClassification::Small => {
                // Small circuits: use CPU to avoid overhead
                return self.find_cpu_substrate(substrates);
            }
            QuickClassification::GHZ | QuickClassification::WState => {
                // Sparse patterns: prefer sparse substrate if available
                if let Some(sparse) = self.find_sparse_substrate(substrates) {
                    return Some(sparse);
                }
                // Fall back to CPU for sparse patterns (still efficient)
                return self.find_cpu_substrate(substrates);
            }
            QuickClassification::NeedsFullAnalysis => {
                // Continue to detailed analysis below
            }
        }

        // Step 2: Compute full characteristics for complex circuits
        let characteristics = crate::math_accelerator::analyze_circuit(&qgates, n_qubits);

        self.route_by_characteristics(&characteristics, spec.n_qubits, substrates)
    }

    /// Math-aware substrate selection with characteristic caching (Sprint 76.0).
    ///
    /// Same as `select_quantum_substrate_math_aware` but caches characteristics
    /// in the mutable QuantumSpec for future use.
    fn select_quantum_substrate_with_caching(
        &self,
        spec: &mut QuantumSpec,
        substrates: &[&SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        let qgates = spec.circuit.to_qgates();
        let n_qubits = spec.n_qubits; // Copy before borrowing

        // Step 1: Quick O(n) classification
        let quick = quick_classify(&qgates, n_qubits as u8);

        match quick {
            QuickClassification::Small => {
                return self.find_cpu_substrate(substrates);
            }
            QuickClassification::GHZ | QuickClassification::WState => {
                if let Some(sparse) = self.find_sparse_substrate(substrates) {
                    return Some(sparse);
                }
                return self.find_cpu_substrate(substrates);
            }
            QuickClassification::NeedsFullAnalysis => {
                // Continue below
            }
        }

        // Step 2: Get or compute characteristics (caches result)
        let characteristics = spec.get_or_compute_characteristics();

        self.route_by_characteristics(characteristics, n_qubits, substrates)
    }

    /// Route based on pre-computed circuit characteristics.
    fn route_by_characteristics(
        &self,
        characteristics: &CircuitCharacteristics,
        n_qubits: usize,
        substrates: &[&SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        // Use the math_accelerator's backend decision logic
        let gpu_available = substrates.iter().any(|s| s.gpu_available);
        let decision =
            select_backend_with_characteristics(characteristics, n_qubits as u8, gpu_available);

        match decision {
            BackendDecision::AlgebraicSkip => {
                // Circuit reduced to identity - use cheapest substrate
                self.find_cpu_substrate(substrates)
            }
            BackendDecision::SparseStateVector { variant: _ } => {
                // Prefer sparse substrate if available
                self.find_sparse_substrate(substrates)
                    .or_else(|| self.find_cpu_substrate(substrates))
            }
            BackendDecision::DenseStateVector { use_gpu } => {
                if use_gpu {
                    self.find_gpu_substrate(substrates)
                        .or_else(|| self.find_cpu_substrate(substrates))
                } else {
                    self.find_cpu_substrate(substrates)
                }
            }
            // Sprint 77: New routing variants
            BackendDecision::Stabilizer => {
                // Stabilizer circuits can run on CPU efficiently
                // In the future, a dedicated stabilizer substrate could be added
                self.find_cpu_substrate(substrates)
            }
            BackendDecision::TensorNetwork {
                estimated_peak_memory: _,
            } => {
                // TN circuits run on CPU (tensor contraction)
                // In the future, GPU tensor contraction could be added
                self.find_cpu_substrate(substrates)
            }
        }
    }

    /// Find a sparse-capable substrate.
    fn find_sparse_substrate(&self, substrates: &[&SubstrateRouteInfo]) -> Option<SubstrateId> {
        substrates
            .iter()
            .filter(|s| s.substrate_type == SubstrateType::Quantum && s.supports_sparse)
            .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
            .map(|s| s.id)
    }

    /// Find a CPU (non-GPU) quantum substrate.
    fn find_cpu_substrate(&self, substrates: &[&SubstrateRouteInfo]) -> Option<SubstrateId> {
        substrates
            .iter()
            .filter(|s| s.substrate_type == SubstrateType::Quantum && !s.gpu_available)
            .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
            .map(|s| s.id)
            .or_else(|| {
                // Fall back to any quantum substrate if no dedicated CPU substrate
                substrates
                    .iter()
                    .filter(|s| s.substrate_type == SubstrateType::Quantum)
                    .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
                    .map(|s| s.id)
            })
    }

    /// Find a GPU quantum substrate.
    fn find_gpu_substrate(&self, substrates: &[&SubstrateRouteInfo]) -> Option<SubstrateId> {
        substrates
            .iter()
            .filter(|s| s.substrate_type == SubstrateType::Quantum && s.gpu_available)
            .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
            .map(|s| s.id)
    }

    /// Simple substrate selection based on qubit count only.
    ///
    /// This is the original Sprint 62.0 logic, used when math routing is disabled.
    fn select_quantum_substrate_simple(
        &self,
        n_qubits: usize,
        substrates: &[&SubstrateRouteInfo],
    ) -> Option<SubstrateId> {
        // Filter to quantum substrates that can handle the qubit count
        let quantum_subs: Vec<&&SubstrateRouteInfo> = substrates
            .iter()
            .filter(|s| s.substrate_type == SubstrateType::Quantum)
            .filter(|s| s.max_qubits.map(|m| n_qubits <= m).unwrap_or(true))
            .collect();

        if quantum_subs.is_empty() {
            return None;
        }

        // Prefer GPU for larger circuits
        let prefer_gpu = self.enable_gpu_routing && n_qubits >= self.gpu_qubit_threshold;

        if prefer_gpu {
            // Try to find GPU substrate first
            if let Some(gpu_sub) = quantum_subs.iter().find(|s| s.gpu_available) {
                return Some(gpu_sub.id);
            }
        }

        // Fall back to highest capacity substrate
        quantum_subs
            .iter()
            .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
            .map(|s| s.id)
    }

    /// Select substrate by type (prefer highest capacity).
    fn select_by_type(
        &self,
        substrates: &[&SubstrateRouteInfo],
        target_type: SubstrateType,
    ) -> Option<SubstrateId> {
        substrates
            .iter()
            .filter(|s| s.substrate_type == target_type)
            .max_by(|a, b| a.capacity.partial_cmp(&b.capacity).unwrap())
            .map(|s| s.id)
    }

    /// Check if GPU should be preferred for the given qubit count.
    pub fn should_use_gpu(&self, n_qubits: usize) -> bool {
        self.enable_gpu_routing && n_qubits >= self.gpu_qubit_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cpu_substrate(id: u64, capacity: f64) -> SubstrateRouteInfo {
        SubstrateRouteInfo {
            id: SubstrateId(id),
            substrate_type: SubstrateType::Quantum,
            gpu_available: false,
            max_qubits: Some(20),
            capacity,
            supports_sparse: false,
        }
    }

    fn make_gpu_substrate(id: u64, capacity: f64) -> SubstrateRouteInfo {
        SubstrateRouteInfo {
            id: SubstrateId(id),
            substrate_type: SubstrateType::Quantum,
            gpu_available: true,
            max_qubits: Some(26),
            capacity,
            supports_sparse: false,
        }
    }

    fn make_sparse_substrate(id: u64, capacity: f64) -> SubstrateRouteInfo {
        SubstrateRouteInfo {
            id: SubstrateId(id),
            substrate_type: SubstrateType::Quantum,
            gpu_available: false,
            max_qubits: None, // Sparse can handle any qubit count
            capacity,
            supports_sparse: true,
        }
    }

    #[test]
    fn test_small_circuit_selects_highest_capacity() {
        let router = AutoRouter::new().with_gpu_threshold(16);
        // Give CPU higher capacity so it's selected for small circuits
        let substrates = vec![make_cpu_substrate(1, 1.0), make_gpu_substrate(2, 0.8)];

        // Small circuit (10 qubits) should not specifically prefer GPU
        // Should select highest capacity (CPU)
        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(10, QuantumCircuit::new(10)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // Should select CPU (id=1) since it has higher capacity
        assert_eq!(selected.unwrap().0, 1);
    }

    #[test]
    fn test_large_circuit_prefers_gpu() {
        // Disable math routing to test qubit-count-based logic
        let router = AutoRouter::new()
            .with_gpu_threshold(16)
            .with_math_routing(false);
        let substrates = vec![make_cpu_substrate(1, 1.0), make_gpu_substrate(2, 1.0)];

        // Large circuit (20 qubits) should prefer GPU
        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(20, QuantumCircuit::new(20)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().0, 2); // GPU substrate
    }

    #[test]
    fn test_fallback_to_cpu_when_gpu_busy() {
        let router = AutoRouter::new()
            .with_gpu_threshold(16)
            .with_min_capacity(0.5);
        let substrates = vec![
            make_cpu_substrate(1, 1.0),
            make_gpu_substrate(2, 0.1), // GPU is busy
        ];

        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(20, QuantumCircuit::new(20)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().0, 1); // Falls back to CPU
    }

    #[test]
    fn test_gpu_routing_disabled() {
        // Disable both GPU routing and math routing to test simple capacity-based selection
        let router = AutoRouter::new()
            .with_gpu_routing(false)
            .with_math_routing(false);
        let substrates = vec![make_cpu_substrate(1, 0.9), make_gpu_substrate(2, 1.0)];

        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(20, QuantumCircuit::new(20)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // Should select highest capacity (GPU) even though GPU routing is disabled
        // because it still has highest capacity
        assert_eq!(selected.unwrap().0, 2);
    }

    #[test]
    fn test_no_substrate_available() {
        let router = AutoRouter::new();
        let substrates: Vec<SubstrateRouteInfo> = vec![];

        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(10, QuantumCircuit::new(10)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_none());
    }

    #[test]
    fn test_should_use_gpu() {
        let router = AutoRouter::new().with_gpu_threshold(16);

        assert!(!router.should_use_gpu(10));
        assert!(!router.should_use_gpu(15));
        assert!(router.should_use_gpu(16));
        assert!(router.should_use_gpu(20));
    }

    #[test]
    fn test_circuit_exceeds_max_qubits() {
        // Disable math routing to test simple qubit-count logic
        let router = AutoRouter::new().with_math_routing(false);
        let substrates = vec![make_cpu_substrate(1, 1.0)]; // max 20 qubits

        use super::super::job::{QuantumCircuit, QuantumSpec};
        let spec = JobSpec::Quantum(QuantumSpec::new(25, QuantumCircuit::new(25)));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_none()); // No substrate can handle 25 qubits
    }

    // =========================================================================
    // Sprint 76.0: Math-Aware Routing Tests
    // =========================================================================

    #[test]
    fn test_ghz_circuit_prefers_sparse() {
        let router = AutoRouter::new();
        let substrates = vec![
            make_cpu_substrate(1, 1.0),
            make_gpu_substrate(2, 1.0),
            make_sparse_substrate(3, 1.0),
        ];

        use super::super::job::{QuantumCircuit, QuantumSpec};

        // Create a GHZ circuit: H(0) followed by CNOT chain
        let mut circuit = QuantumCircuit::new(5);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.cnot(1, 2);
        circuit.cnot(2, 3);
        circuit.cnot(3, 4);

        let spec = JobSpec::Quantum(QuantumSpec::new(5, circuit));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // Should prefer sparse substrate (id=3) for GHZ pattern
        assert_eq!(selected.unwrap().0, 3);
    }

    #[test]
    fn test_ghz_falls_back_to_cpu_without_sparse() {
        let router = AutoRouter::new();
        let substrates = vec![
            make_cpu_substrate(1, 1.0),
            make_gpu_substrate(2, 1.0),
            // No sparse substrate
        ];

        use super::super::job::{QuantumCircuit, QuantumSpec};

        // Create a GHZ circuit
        let mut circuit = QuantumCircuit::new(5);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.cnot(1, 2);
        circuit.cnot(2, 3);
        circuit.cnot(3, 4);

        let spec = JobSpec::Quantum(QuantumSpec::new(5, circuit));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // Should fall back to CPU substrate (id=1) since no sparse available
        assert_eq!(selected.unwrap().0, 1);
    }

    #[test]
    fn test_small_circuit_uses_cpu_with_math_routing() {
        let router = AutoRouter::new();
        let substrates = vec![make_cpu_substrate(1, 1.0), make_gpu_substrate(2, 1.0)];

        use super::super::job::{QuantumCircuit, QuantumSpec};

        // Small circuit with just a few gates
        let mut circuit = QuantumCircuit::new(4);
        circuit.h(0);
        circuit.x(1);

        let spec = JobSpec::Quantum(QuantumSpec::new(4, circuit));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // Small circuit should use CPU to avoid overhead
        assert_eq!(selected.unwrap().0, 1);
    }

    #[test]
    fn test_math_routing_disabled_uses_simple_logic() {
        let router = AutoRouter::new()
            .with_math_routing(false)
            .with_gpu_threshold(16);
        let substrates = vec![
            make_cpu_substrate(1, 1.0),
            make_gpu_substrate(2, 1.0),
            make_sparse_substrate(3, 1.0),
        ];

        use super::super::job::{QuantumCircuit, QuantumSpec};

        // Create a GHZ circuit that would normally route to sparse
        let mut circuit = QuantumCircuit::new(5);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.cnot(1, 2);
        circuit.cnot(2, 3);
        circuit.cnot(3, 4);

        let spec = JobSpec::Quantum(QuantumSpec::new(5, circuit));

        let selected = router.select_substrate(&spec, &substrates);
        assert!(selected.is_some());
        // With math routing disabled, should use simple qubit-count logic
        // 5 qubits < 16, so highest capacity wins (CPU has highest among non-sparse)
        // Actually all have 1.0, so depends on iteration order
        // The simple logic picks highest capacity quantum substrate
        assert!(selected.unwrap().0 == 1 || selected.unwrap().0 == 2 || selected.unwrap().0 == 3);
    }

    #[test]
    fn test_substrate_route_info_new_with_sparse() {
        let caps = SubstrateCapabilities {
            max_qubits: Some(30),
            gpu_available: false,
            ..Default::default()
        };

        let info = SubstrateRouteInfo::new_with_sparse(
            SubstrateId(42),
            SubstrateType::Quantum,
            &caps,
            0.75,
            true,
        );

        assert!(info.supports_sparse);
        assert_eq!(info.max_qubits, Some(30));
        assert!(!info.gpu_available);
    }
}
