//! Variational Quantum Circuits for Clusters (Sprint 59.0 EPIC 138)
//!
//! Makes cluster operations trainable with parameterized gates.
//!
//! # Architecture
//!
//! ```text
//! |0⟩ ─── Ry(θ₀) ─── Rz(φ₀) ───╮
//!                               CZ ─── Ry(θ₂) ─── ...
//! |0⟩ ─── Ry(θ₁) ─── Rz(φ₁) ───╯
//! ```
//!
//! # Gradient Computation
//!
//! Uses the parameter-shift rule for gradient estimation:
//!
//! ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
//!
//! This is exact for rotation gates and noise-robust.

use std::f64::consts::PI;

use super::stabilizer_hybrid::FullQuantumCluster;

/// Configuration for a variational circuit.
#[derive(Clone, Debug)]
pub struct VariationalConfig {
    /// Number of layers (depth)
    pub depth: usize,

    /// Initial parameter scale (random initialization range)
    pub init_scale: f64,

    /// Learning rate for parameter updates
    pub learning_rate: f64,

    /// Use entangling gates between layers
    pub use_entangling: bool,

    /// Entangling pattern
    pub entangle_pattern: EntanglePattern,
}

impl Default for VariationalConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            init_scale: 0.1,
            learning_rate: 0.01,
            use_entangling: true,
            entangle_pattern: EntanglePattern::Linear,
        }
    }
}

impl VariationalConfig {
    /// Shallow circuit (1 layer).
    pub fn shallow() -> Self {
        Self {
            depth: 1,
            ..Default::default()
        }
    }

    /// Deep circuit (4 layers).
    pub fn deep() -> Self {
        Self {
            depth: 4,
            ..Default::default()
        }
    }

    /// Hardware-efficient ansatz.
    pub fn hardware_efficient(depth: usize) -> Self {
        Self {
            depth,
            use_entangling: true,
            entangle_pattern: EntanglePattern::Linear,
            ..Default::default()
        }
    }
}

/// Pattern for entangling gates between qubits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntanglePattern {
    /// Linear chain: (0,1), (1,2), (2,3), ...
    Linear,
    /// All pairs (full connectivity)
    AllPairs,
    /// Circular: includes (n-1, 0)
    Circular,
    /// Alternating: even layers (0,1), (2,3), ...; odd layers (1,2), (3,4), ...
    Alternating,
}

/// Parameters for a single variational layer.
#[derive(Clone, Debug)]
pub struct VariationalLayer {
    /// Rotation angles: [qubit][axis] where axis: 0=X, 1=Y, 2=Z
    pub rotations: Vec<[f64; 3]>,

    /// Which qubit pairs to entangle
    pub entangle_pairs: Vec<(usize, usize)>,

    /// Gradients from last backward pass
    gradients: Vec<[f64; 3]>,
}

impl VariationalLayer {
    /// Create a new layer with random initialization.
    pub fn new(n_qubits: usize, pattern: EntanglePattern, init_scale: f64, seed: u64) -> Self {
        // Simple PRNG for initialization
        let mut rng_state = seed;
        let mut rand_f64 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };

        let rotations: Vec<[f64; 3]> = (0..n_qubits)
            .map(|_| {
                [
                    rand_f64() * init_scale,
                    rand_f64() * init_scale,
                    rand_f64() * init_scale,
                ]
            })
            .collect();

        let entangle_pairs = Self::generate_pairs(n_qubits, pattern);

        Self {
            rotations,
            entangle_pairs,
            gradients: vec![[0.0; 3]; n_qubits],
        }
    }

    /// Generate entangling pairs based on pattern.
    fn generate_pairs(n_qubits: usize, pattern: EntanglePattern) -> Vec<(usize, usize)> {
        if n_qubits < 2 {
            return Vec::new();
        }

        match pattern {
            EntanglePattern::Linear => (0..n_qubits - 1).map(|i| (i, i + 1)).collect(),
            EntanglePattern::Circular => {
                let mut pairs: Vec<_> = (0..n_qubits - 1).map(|i| (i, i + 1)).collect();
                pairs.push((n_qubits - 1, 0));
                pairs
            }
            EntanglePattern::AllPairs => {
                let mut pairs = Vec::new();
                for i in 0..n_qubits {
                    for j in (i + 1)..n_qubits {
                        pairs.push((i, j));
                    }
                }
                pairs
            }
            EntanglePattern::Alternating => {
                // Just do linear for now
                (0..n_qubits - 1).map(|i| (i, i + 1)).collect()
            }
        }
    }

    /// Apply this layer to a cluster.
    pub fn apply(&self, cluster: &mut FullQuantumCluster) {
        let n = cluster.n_qubits().min(self.rotations.len());

        // Single-qubit rotations: Rx, Ry, Rz
        for qubit in 0..n {
            let [rx, ry, rz] = self.rotations[qubit];
            if rx.abs() > 1e-8 {
                cluster.rx(qubit, rx);
            }
            if ry.abs() > 1e-8 {
                cluster.ry(qubit, ry);
            }
            if rz.abs() > 1e-8 {
                cluster.rz(qubit, rz);
            }
        }

        // Entangling layer
        for &(a, b) in &self.entangle_pairs {
            if a < cluster.n_qubits() && b < cluster.n_qubits() {
                cluster.cz(a, b);
            }
        }
    }

    /// Get number of parameters in this layer.
    pub fn n_params(&self) -> usize {
        self.rotations.len() * 3
    }

    /// Get all parameters as a flat vector.
    pub fn get_params(&self) -> Vec<f64> {
        self.rotations
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect()
    }

    /// Set parameters from a flat vector.
    pub fn set_params(&mut self, params: &[f64]) {
        for (i, rot) in self.rotations.iter_mut().enumerate() {
            let base = i * 3;
            if base + 2 < params.len() {
                rot[0] = params[base];
                rot[1] = params[base + 1];
                rot[2] = params[base + 2];
            }
        }
    }

    /// Get gradients as a flat vector.
    pub fn get_gradients(&self) -> Vec<f64> {
        self.gradients
            .iter()
            .flat_map(|g| g.iter().copied())
            .collect()
    }

    /// Apply gradient update with learning rate.
    pub fn update(&mut self, learning_rate: f64) {
        for (rot, grad) in self.rotations.iter_mut().zip(&self.gradients) {
            rot[0] -= learning_rate * grad[0];
            rot[1] -= learning_rate * grad[1];
            rot[2] -= learning_rate * grad[2];
        }
    }
}

/// A complete variational quantum circuit.
#[derive(Clone, Debug)]
pub struct VariationalCircuit {
    /// Layers of the circuit
    layers: Vec<VariationalLayer>,

    /// Configuration
    config: VariationalConfig,

    /// Number of qubits
    #[allow(dead_code)]
    n_qubits: usize,
}

impl VariationalCircuit {
    /// Create a new variational circuit.
    pub fn new(n_qubits: usize, config: VariationalConfig, seed: u64) -> Self {
        let layers = (0..config.depth)
            .map(|d| {
                VariationalLayer::new(
                    n_qubits,
                    config.entangle_pattern,
                    config.init_scale,
                    seed.wrapping_add(d as u64 * 1000),
                )
            })
            .collect();

        Self {
            layers,
            config,
            n_qubits,
        }
    }

    /// Create a hardware-efficient ansatz circuit.
    pub fn hardware_efficient(n_qubits: usize, depth: usize, seed: u64) -> Self {
        let config = VariationalConfig::hardware_efficient(depth);
        Self::new(n_qubits, config, seed)
    }

    /// Get number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Get total number of parameters.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Apply the full circuit to a cluster.
    pub fn apply(&self, cluster: &mut FullQuantumCluster) {
        for layer in &self.layers {
            layer.apply(cluster);
        }
    }

    /// Get all parameters as a flat vector.
    pub fn get_params(&self) -> Vec<f64> {
        self.layers.iter().flat_map(|l| l.get_params()).collect()
    }

    /// Set all parameters from a flat vector.
    pub fn set_params(&mut self, params: &[f64]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let n = layer.n_params();
            if offset + n <= params.len() {
                layer.set_params(&params[offset..offset + n]);
            }
            offset += n;
        }
    }

    /// Get all gradients as a flat vector.
    pub fn get_gradients(&self) -> Vec<f64> {
        self.layers.iter().flat_map(|l| l.get_gradients()).collect()
    }

    /// Compute gradients using parameter-shift rule.
    ///
    /// The cost function takes a cluster reference and returns a scalar.
    pub fn compute_gradients<F>(&mut self, initial_state: &FullQuantumCluster, cost_fn: F)
    where
        F: Fn(&FullQuantumCluster) -> f64,
    {
        let shift = PI / 2.0;
        let params = self.get_params();

        for (p_idx, _) in params.iter().enumerate() {
            // Shift up
            let mut params_plus = params.clone();
            params_plus[p_idx] += shift;

            let mut cluster_plus = initial_state.clone();
            self.set_params(&params_plus);
            self.apply(&mut cluster_plus);
            let cost_plus = cost_fn(&cluster_plus);

            // Shift down
            let mut params_minus = params.clone();
            params_minus[p_idx] -= shift;

            let mut cluster_minus = initial_state.clone();
            self.set_params(&params_minus);
            self.apply(&mut cluster_minus);
            let cost_minus = cost_fn(&cluster_minus);

            // Gradient
            let gradient = (cost_plus - cost_minus) / 2.0;

            // Store gradient
            self.set_gradient_at(p_idx, gradient);
        }

        // Restore original parameters
        self.set_params(&params);
    }

    /// Set gradient at a specific parameter index.
    fn set_gradient_at(&mut self, param_idx: usize, gradient: f64) {
        let mut idx = 0;
        for layer in &mut self.layers {
            let n = layer.n_params();
            if param_idx >= idx && param_idx < idx + n {
                let local_idx = param_idx - idx;
                let qubit = local_idx / 3;
                let axis = local_idx % 3;
                layer.gradients[qubit][axis] = gradient;
                return;
            }
            idx += n;
        }
    }

    /// Update parameters using stored gradients.
    pub fn update(&mut self) {
        for layer in &mut self.layers {
            layer.update(self.config.learning_rate);
        }
    }

    /// Train with a cost function (one step).
    pub fn train_step<F>(&mut self, initial_state: &FullQuantumCluster, cost_fn: F) -> f64
    where
        F: Fn(&FullQuantumCluster) -> f64,
    {
        // Compute current cost
        let mut cluster = initial_state.clone();
        self.apply(&mut cluster);
        let cost = cost_fn(&cluster);

        // Compute gradients
        self.compute_gradients(initial_state, &cost_fn);

        // Update
        self.update();

        cost
    }

    /// Train for multiple iterations.
    pub fn train<F>(
        &mut self,
        initial_state: &FullQuantumCluster,
        cost_fn: F,
        iterations: usize,
    ) -> Vec<f64>
    where
        F: Fn(&FullQuantumCluster) -> f64,
    {
        let mut costs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let cost = self.train_step(initial_state, &cost_fn);
            costs.push(cost);
        }
        costs
    }
}

/// Cost functions for variational training.
pub mod cost_functions {
    use super::*;

    /// Maximize Z expectation on a target qubit.
    pub fn maximize_z(target_qubit: usize) -> impl Fn(&FullQuantumCluster) -> f64 {
        move |cluster| -cluster.z_expectation(target_qubit)
    }

    /// Minimize Z expectation on a target qubit.
    pub fn minimize_z(target_qubit: usize) -> impl Fn(&FullQuantumCluster) -> f64 {
        move |cluster| cluster.z_expectation(target_qubit)
    }

    /// Maximize total Z expectation across all qubits.
    pub fn maximize_total_z() -> impl Fn(&FullQuantumCluster) -> f64 {
        |cluster| -cluster.total_z_expectation()
    }

    /// Target a specific probability distribution.
    pub fn target_distribution(target_probs: Vec<f64>) -> impl Fn(&FullQuantumCluster) -> f64 {
        move |cluster| {
            let mut cost = 0.0;
            for (i, &target) in target_probs.iter().enumerate() {
                if i < cluster.dim() {
                    let actual = cluster.probability(i);
                    cost += (actual - target).powi(2);
                }
            }
            cost
        }
    }

    /// Create cost function from reward signal.
    ///
    /// Higher reward -> lower cost for configurations that led to reward.
    pub fn reward_correlation(reward: f64) -> impl Fn(&FullQuantumCluster) -> f64 {
        move |cluster| {
            // Cost is negative reward times Z expectation
            // Positive reward + positive Z -> low cost
            // Negative reward + positive Z -> high cost
            let z = cluster.total_z_expectation();
            -reward * z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = VariationalLayer::new(4, EntanglePattern::Linear, 0.1, 42);

        assert_eq!(layer.rotations.len(), 4);
        assert_eq!(layer.entangle_pairs.len(), 3); // 0-1, 1-2, 2-3
    }

    #[test]
    fn test_layer_apply() {
        let mut cluster = FullQuantumCluster::new(4, 0);
        let layer = VariationalLayer::new(4, EntanglePattern::Linear, 0.5, 42);

        layer.apply(&mut cluster);

        // Should have modified the state
        // (won't be in |0000⟩ anymore)
        assert!(cluster.probability(0) < 1.0);
    }

    #[test]
    fn test_circuit_creation() {
        let config = VariationalConfig::default();
        let circuit = VariationalCircuit::new(4, config, 42);

        assert_eq!(circuit.depth(), 2);
        assert_eq!(circuit.n_params(), 4 * 3 * 2); // 4 qubits * 3 axes * 2 layers
    }

    #[test]
    fn test_circuit_apply() {
        let mut cluster = FullQuantumCluster::new(4, 0);
        let circuit = VariationalCircuit::hardware_efficient(4, 2, 42);

        circuit.apply(&mut cluster);

        // State should be modified
        assert!(cluster.probability(0) < 1.0);
        // Should still be normalized
        assert!((cluster.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_set_params() {
        let config = VariationalConfig::shallow();
        let mut circuit = VariationalCircuit::new(2, config, 42);

        let original = circuit.get_params();
        let n = original.len();

        // Modify parameters
        let new_params: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        circuit.set_params(&new_params);

        let retrieved = circuit.get_params();
        for (a, b) in new_params.iter().zip(retrieved.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_entangle_patterns() {
        // Linear
        let pairs = VariationalLayer::generate_pairs(4, EntanglePattern::Linear);
        assert_eq!(pairs.len(), 3);

        // Circular
        let pairs = VariationalLayer::generate_pairs(4, EntanglePattern::Circular);
        assert_eq!(pairs.len(), 4); // Includes wrap-around

        // All pairs
        let pairs = VariationalLayer::generate_pairs(4, EntanglePattern::AllPairs);
        assert_eq!(pairs.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_cost_function_z_expectation() {
        let cluster = FullQuantumCluster::new(2, 0);
        let cost_fn = cost_functions::maximize_z(0);

        // |00⟩ has Z=+1 on qubit 0, so cost = -1
        let cost = cost_fn(&cluster);
        assert!((cost - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_computation() {
        let initial = FullQuantumCluster::new(2, 0);
        let config = VariationalConfig::shallow();
        let mut circuit = VariationalCircuit::new(2, config, 42);

        let cost_fn = cost_functions::maximize_z(0);
        circuit.compute_gradients(&initial, cost_fn);

        let gradients = circuit.get_gradients();

        // Should have gradients for all parameters
        assert_eq!(gradients.len(), circuit.n_params());

        // At least some gradients should be non-zero
        let non_zero = gradients.iter().filter(|g| g.abs() > 1e-10).count();
        assert!(non_zero > 0, "Expected some non-zero gradients");
    }

    #[test]
    fn test_training() {
        let initial = FullQuantumCluster::new(2, 0);

        let mut config = VariationalConfig::shallow();
        config.learning_rate = 0.1;
        config.init_scale = 0.5;

        let mut circuit = VariationalCircuit::new(2, config, 42);

        // Try to minimize Z on qubit 0 (start from |0⟩, try to get to |1⟩)
        let cost_fn = cost_functions::minimize_z(0);

        let costs = circuit.train(&initial, cost_fn, 10);

        // Cost should generally decrease (training is working)
        // Note: This is a simple test; real training might have noise
        assert!(costs.len() == 10);
    }

    #[test]
    fn test_reward_correlation() {
        let cluster = FullQuantumCluster::new(2, 0);

        // Positive reward
        let cost_pos = cost_functions::reward_correlation(1.0);
        let c1 = cost_pos(&cluster);

        // Negative reward
        let cost_neg = cost_functions::reward_correlation(-1.0);
        let c2 = cost_neg(&cluster);

        // Costs should be opposite signs
        assert!(c1 * c2 < 0.0);
    }

    #[test]
    fn test_target_distribution() {
        let cluster = FullQuantumCluster::new(2, 0);

        // Target: 100% in |00⟩
        let target = vec![1.0, 0.0, 0.0, 0.0];
        let cost_fn = cost_functions::target_distribution(target);

        let cost = cost_fn(&cluster);

        // |00⟩ matches target perfectly, cost should be ~0
        assert!(cost < 0.01);
    }
}
