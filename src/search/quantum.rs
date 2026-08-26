//! Quantum Grover Amplification for search (Phase 4).
//!
//! Implements quantum-inspired amplitude amplification to boost
//! selection probability of high-fitness candidates.
//!
//! # Theory
//!
//! Grover's algorithm provides quadratic speedup for unstructured search.
//! For a population of N candidates with M "good" solutions (above threshold),
//! classical random sampling needs O(N/M) attempts, while Grover needs O(√(N/M)).
//!
//! We simulate this by:
//! 1. Encoding fitness as amplitudes: |ψ⟩ = Σ αᵢ|i⟩
//! 2. Applying oracle: marks candidates above fitness threshold
//! 3. Applying diffusion: amplifies marked amplitudes
//! 4. Measuring: sample with boosted probabilities

use std::f64::consts::PI;

use super::candidate::{Rng, Xorshift64};
use super::substrate::SearchSubstrate;

/// Configuration for quantum-enhanced selection
#[derive(Clone, Debug)]
pub struct QuantumConfig {
    /// Number of Grover iterations (optimal ≈ π/4 * √(N/M))
    pub iterations: usize,
    /// Fitness threshold for oracle (candidates above this are "good")
    pub fitness_threshold: u8,
    /// Whether to auto-compute optimal iterations
    pub auto_iterations: bool,
    /// Minimum amplification factor (prevents over-rotation)
    pub min_amplification: f64,
    /// Maximum amplification factor
    pub max_amplification: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            fitness_threshold: 128, // Above median
            auto_iterations: true,
            min_amplification: 1.0,
            max_amplification: 10.0,
        }
    }
}

impl QuantumConfig {
    /// Create config with specific iteration count
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self.auto_iterations = false;
        self
    }

    /// Set fitness threshold
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.fitness_threshold = threshold;
        self
    }

    /// Enable auto iteration computation
    pub fn with_auto_iterations(mut self) -> Self {
        self.auto_iterations = true;
        self
    }

    /// Set amplification bounds
    pub fn with_amplification_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_amplification = min;
        self.max_amplification = max;
        self
    }

    /// Compute optimal iterations for given N (total) and M (marked)
    pub fn optimal_iterations(n: usize, m: usize) -> usize {
        if m == 0 || m >= n {
            return 1;
        }
        // Optimal iterations ≈ π/4 * √(N/M)
        let ratio = n as f64 / m as f64;
        let optimal = (PI / 4.0 * ratio.sqrt()).round() as usize;
        optimal.max(1).min(100) // Clamp to reasonable range
    }
}

/// Quantum amplitude state for the population
///
/// Represents |ψ⟩ = Σ αᵢ|i⟩ where αᵢ is the amplitude for candidate i
#[derive(Clone, Debug)]
pub struct QuantumState {
    /// Complex amplitudes (stored as real part only for efficiency)
    /// In full quantum: α = re^(iθ), we track |α|²
    amplitudes: Vec<f64>,
    /// Phase information (for interference)
    phases: Vec<f64>,
    /// Number of qubits (log₂ of population size)
    #[allow(dead_code)]
    num_qubits: usize,
}

impl QuantumState {
    /// Create uniform superposition over N states
    pub fn uniform(n: usize) -> Self {
        let amplitude = 1.0 / (n as f64).sqrt();
        Self {
            amplitudes: vec![amplitude; n],
            phases: vec![0.0; n],
            num_qubits: (n as f64).log2().ceil() as usize,
        }
    }

    /// Create from fitness values (fitness-weighted superposition)
    pub fn from_fitness(fitnesses: &[u8]) -> Self {
        let n = fitnesses.len();
        let total: f64 = fitnesses.iter().map(|&f| f as f64).sum();

        let amplitudes = if total > 0.0 {
            fitnesses
                .iter()
                .map(|&f| (f as f64 / total).sqrt())
                .collect()
        } else {
            vec![1.0 / (n as f64).sqrt(); n]
        };

        Self {
            amplitudes,
            phases: vec![0.0; n],
            num_qubits: (n as f64).log2().ceil() as usize,
        }
    }

    /// Get the number of states
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Get probability for state i: |αᵢ|²
    #[inline]
    pub fn probability(&self, i: usize) -> f64 {
        self.amplitudes[i] * self.amplitudes[i]
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a * a).collect()
    }

    /// Normalize amplitudes to ensure Σ|αᵢ|² = 1
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a * a).sum();
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for a in &mut self.amplitudes {
                *a /= norm;
            }
        }
    }

    /// Apply phase flip to marked states (oracle)
    pub fn apply_oracle(&mut self, marked: &[bool]) {
        for (i, &is_marked) in marked.iter().enumerate() {
            if is_marked {
                self.amplitudes[i] = -self.amplitudes[i];
                self.phases[i] += PI;
            }
        }
    }

    /// Apply Grover diffusion operator
    /// D = 2|ψ⟩⟨ψ| - I (reflection about mean)
    pub fn apply_diffusion(&mut self) {
        let n = self.amplitudes.len();
        let mean: f64 = self.amplitudes.iter().sum::<f64>() / n as f64;

        for a in &mut self.amplitudes {
            *a = 2.0 * mean - *a;
        }
    }

    /// Perform one Grover iteration (oracle + diffusion)
    pub fn grover_iteration(&mut self, marked: &[bool]) {
        self.apply_oracle(marked);
        self.apply_diffusion();
    }

    /// Get amplitude statistics
    pub fn statistics(&self) -> QuantumStateStats {
        let n = self.amplitudes.len();
        let probs = self.probabilities();

        let max_prob = probs.iter().cloned().fold(0.0f64, f64::max);
        let min_prob = probs.iter().cloned().fold(1.0f64, f64::min);
        let mean_prob = 1.0 / n as f64;

        // Entropy: -Σ p log p
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // Max entropy for uniform distribution
        let max_entropy = (n as f64).ln();

        QuantumStateStats {
            num_states: n,
            max_probability: max_prob,
            min_probability: min_prob,
            mean_probability: mean_prob,
            entropy,
            uniformity: entropy / max_entropy,
        }
    }

    /// Sample a state index according to amplitude probabilities
    pub fn sample(&self, rng: &mut Xorshift64) -> usize {
        let target = rng.next_f64();
        let mut cumulative = 0.0;

        for (i, &a) in self.amplitudes.iter().enumerate() {
            cumulative += a * a;
            if cumulative >= target {
                return i;
            }
        }

        self.amplitudes.len() - 1
    }

    /// Sample multiple states
    pub fn sample_multiple(&self, count: usize, rng: &mut Xorshift64) -> Vec<usize> {
        (0..count).map(|_| self.sample(rng)).collect()
    }
}

/// Statistics about quantum state
#[derive(Clone, Debug, Default)]
pub struct QuantumStateStats {
    pub num_states: usize,
    pub max_probability: f64,
    pub min_probability: f64,
    pub mean_probability: f64,
    pub entropy: f64,
    pub uniformity: f64, // 1.0 = uniform, 0.0 = concentrated
}

/// Grover-enhanced selector for search
pub struct GroverSelector {
    config: QuantumConfig,
    /// Cached quantum state
    state: Option<QuantumState>,
    /// Statistics from last selection
    last_stats: Option<GroverStats>,
}

impl GroverSelector {
    /// Create a new Grover selector
    pub fn new(config: QuantumConfig) -> Self {
        Self {
            config,
            state: None,
            last_stats: None,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(QuantumConfig::default())
    }

    /// Create with specific iteration count
    pub fn with_iterations(iterations: usize) -> Self {
        Self::new(QuantumConfig::default().with_iterations(iterations))
    }

    /// Select candidates using Grover amplification
    pub fn select(
        &mut self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        if n == 0 {
            return vec![];
        }

        // Determine fitness threshold (adaptive based on population)
        let stats = substrate.statistics();
        let threshold = if self.config.fitness_threshold == 0 {
            // Auto threshold: use mean + 0.5*std as "good"
            (stats.mean_fitness + 0.5 * stats.std_fitness) as u8
        } else {
            self.config.fitness_threshold
        };

        // Mark candidates above threshold
        let marked: Vec<bool> = substrate
            .candidates()
            .iter()
            .map(|c| c.fitness >= threshold)
            .collect();

        let num_marked = marked.iter().filter(|&&m| m).count();

        // Compute optimal iterations if auto
        let iterations = if self.config.auto_iterations {
            QuantumConfig::optimal_iterations(n, num_marked.max(1))
        } else {
            self.config.iterations
        };

        // Initialize quantum state (uniform superposition)
        let mut state = QuantumState::uniform(n);
        let initial_stats = state.statistics();

        // Apply Grover iterations
        for _ in 0..iterations {
            state.grover_iteration(&marked);
        }

        // Normalize (numerical stability)
        state.normalize();

        let final_stats = state.statistics();

        // Compute amplification achieved
        let marked_prob_before: f64 = marked
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(_, _)| 1.0 / n as f64)
            .sum();

        let marked_prob_after: f64 = marked
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(i, _)| state.probability(i))
            .sum();

        let amplification = if marked_prob_before > 0.0 {
            marked_prob_after / marked_prob_before
        } else {
            1.0
        };

        // Record statistics
        self.last_stats = Some(GroverStats {
            population_size: n,
            num_marked,
            threshold,
            iterations,
            amplification,
            marked_prob_before,
            marked_prob_after,
            initial_entropy: initial_stats.entropy,
            final_entropy: final_stats.entropy,
        });

        // Cache state for inspection
        self.state = Some(state.clone());

        // Sample from amplified distribution
        state.sample_multiple(count, rng)
    }

    /// Get statistics from last selection
    pub fn last_stats(&self) -> Option<&GroverStats> {
        self.last_stats.as_ref()
    }

    /// Get cached quantum state
    pub fn state(&self) -> Option<&QuantumState> {
        self.state.as_ref()
    }

    /// Get config
    pub fn config(&self) -> &QuantumConfig {
        &self.config
    }
}

/// Statistics from Grover selection
#[derive(Clone, Debug)]
pub struct GroverStats {
    pub population_size: usize,
    pub num_marked: usize,
    pub threshold: u8,
    pub iterations: usize,
    pub amplification: f64,
    pub marked_prob_before: f64,
    pub marked_prob_after: f64,
    pub initial_entropy: f64,
    pub final_entropy: f64,
}

impl GroverStats {
    /// Effective speedup from amplification
    pub fn effective_speedup(&self) -> f64 {
        self.amplification.sqrt()
    }
}

/// Quantum-classical hybrid selector
/// Uses Grover for exploration, classical for exploitation
pub struct HybridQuantumSelector {
    /// Grover selector for quantum-enhanced selection
    grover: GroverSelector,
    /// Quantum selection probability (0.0-1.0)
    quantum_prob: f64,
    /// Generation counter
    generation: u64,
}

impl HybridQuantumSelector {
    /// Create a new hybrid selector
    pub fn new(grover_config: QuantumConfig, quantum_prob: f64) -> Self {
        Self {
            grover: GroverSelector::new(grover_config),
            quantum_prob: quantum_prob.clamp(0.0, 1.0),
            generation: 0,
        }
    }

    /// Create with default settings (50% quantum)
    pub fn balanced() -> Self {
        Self::new(QuantumConfig::default(), 0.5)
    }

    /// Create quantum-heavy selector (80% quantum)
    pub fn quantum_heavy() -> Self {
        Self::new(QuantumConfig::default(), 0.8)
    }

    /// Create classical-heavy selector (20% quantum)
    pub fn classical_heavy() -> Self {
        Self::new(QuantumConfig::default(), 0.2)
    }

    /// Select candidates using hybrid approach
    pub fn select(
        &mut self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        if n == 0 {
            return vec![];
        }

        // Determine how many to select via quantum vs classical
        let quantum_count = (count as f64 * self.quantum_prob).round() as usize;
        let classical_count = count - quantum_count;

        let mut selected = Vec::with_capacity(count);

        // Quantum selection (Grover-amplified)
        if quantum_count > 0 {
            let quantum_selected = self.grover.select(substrate, quantum_count, rng);
            selected.extend(quantum_selected);
        }

        // Classical selection (fitness-proportionate)
        if classical_count > 0 {
            let total_fitness: f64 = substrate
                .candidates()
                .iter()
                .map(|c| c.fitness as f64)
                .sum();

            for _ in 0..classical_count {
                if total_fitness > 0.0 {
                    let target = rng.next_f64() * total_fitness;
                    let mut cumulative = 0.0;
                    let mut found = false;

                    for (i, c) in substrate.candidates().iter().enumerate() {
                        cumulative += c.fitness as f64;
                        if cumulative >= target {
                            selected.push(i);
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        selected.push(n - 1);
                    }
                } else {
                    // Uniform random if all zero fitness
                    selected.push((rng.next_u64() as usize) % n);
                }
            }
        }

        self.generation += 1;
        selected
    }

    /// Get Grover statistics from last selection
    pub fn grover_stats(&self) -> Option<&GroverStats> {
        self.grover.last_stats()
    }

    /// Adjust quantum probability based on search progress
    pub fn adapt_quantum_prob(&mut self, improvement_rate: f64) {
        // More quantum when stuck (low improvement), more classical when improving
        if improvement_rate < 0.1 {
            self.quantum_prob = (self.quantum_prob + 0.1).min(0.9);
        } else if improvement_rate > 0.3 {
            self.quantum_prob = (self.quantum_prob - 0.1).max(0.1);
        }
    }
}

/// Compute theoretical Grover speedup
pub fn theoretical_speedup(n: usize, m: usize) -> f64 {
    if m == 0 || m >= n {
        return 1.0;
    }
    // Classical: O(N/M), Quantum: O(√(N/M))
    // Speedup = √(N/M)
    (n as f64 / m as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::OneMax;

    #[test]
    fn test_quantum_state_uniform() {
        let state = QuantumState::uniform(100);
        assert_eq!(state.len(), 100);

        // All probabilities should be equal
        let prob = state.probability(0);
        assert!((prob - 0.01).abs() < 0.0001);

        // Sum of probabilities should be 1
        let total: f64 = state.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_quantum_state_from_fitness() {
        let fitnesses = vec![100, 50, 25, 25];
        let state = QuantumState::from_fitness(&fitnesses);

        // Higher fitness should have higher probability
        assert!(state.probability(0) > state.probability(1));
        assert!(state.probability(1) > state.probability(2));

        // Sum should be 1
        let total: f64 = state.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_grover_iteration() {
        let mut state = QuantumState::uniform(8);
        let marked = vec![true, false, false, false, false, false, false, false];

        // Before Grover: uniform probabilities
        let prob_before = state.probability(0);
        assert!((prob_before - 0.125).abs() < 0.001);

        // After one iteration: marked state should have higher probability
        state.grover_iteration(&marked);
        let prob_after = state.probability(0);

        assert!(
            prob_after > prob_before,
            "Marked state probability should increase: {} > {}",
            prob_after,
            prob_before
        );
    }

    #[test]
    fn test_optimal_iterations() {
        // For N=100, M=1: optimal ≈ π/4 * √100 ≈ 7.85
        let opt = QuantumConfig::optimal_iterations(100, 1);
        assert!((7..=9).contains(&opt), "Got {}", opt);

        // For N=100, M=25: optimal ≈ π/4 * √4 ≈ 1.57
        let opt = QuantumConfig::optimal_iterations(100, 25);
        assert!((1..=3).contains(&opt), "Got {}", opt);

        // Edge cases
        assert_eq!(QuantumConfig::optimal_iterations(100, 0), 1);
        assert_eq!(QuantumConfig::optimal_iterations(100, 100), 1);
    }

    #[test]
    fn test_grover_selector() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut selector = GroverSelector::with_iterations(3);
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 20, &mut rng);
        assert_eq!(selected.len(), 20);

        // Check that statistics were recorded
        let stats = selector.last_stats().unwrap();
        assert!(stats.amplification >= 1.0, "Should have amplification");
        println!("Grover amplification: {:.2}x", stats.amplification);
    }

    #[test]
    fn test_grover_amplifies_good_candidates() {
        let mut substrate = SearchSubstrate::new(1000, OneMax::new(16));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let pop_mean = substrate.statistics().mean_fitness;

        // Use a lower threshold to mark more candidates (better for stable testing)
        let mut selector = GroverSelector::new(
            QuantumConfig::default()
                .with_threshold((pop_mean * 0.9) as u8) // Mark top ~60%
                .with_iterations(2), // Fixed iterations for reproducibility
        );
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 200, &mut rng);

        // Selected candidates should have higher average fitness
        let selected_mean: f64 = selected
            .iter()
            .map(|&i| substrate.get(i).fitness as f64)
            .sum::<f64>()
            / selected.len() as f64;

        println!("Population mean: {:.2}", pop_mean);
        println!("Selected mean: {:.2}", selected_mean);

        let stats = selector.last_stats().unwrap();
        println!("Amplification: {:.2}x", stats.amplification);
        println!(
            "Marked prob: {:.4} -> {:.4}",
            stats.marked_prob_before, stats.marked_prob_after
        );
        println!("Num marked: {}", stats.num_marked);

        // Primary test: amplification should be > 1.0
        assert!(
            stats.amplification >= 1.0,
            "Grover should provide amplification: got {:.2}x",
            stats.amplification
        );

        // Secondary: selected mean should be close to or above pop mean
        // (with amplification, the selected distribution is biased towards marked)
        assert!(
            selected_mean >= pop_mean as f64 * 0.95,
            "Selected mean {:.2} too low vs pop mean {:.2}",
            selected_mean,
            pop_mean
        );
    }

    #[test]
    fn test_hybrid_selector() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut selector = HybridQuantumSelector::balanced();
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 20, &mut rng);
        assert_eq!(selected.len(), 20);
    }

    #[test]
    fn test_theoretical_speedup() {
        // N=100, M=1 should give √100 = 10x speedup
        let speedup = theoretical_speedup(100, 1);
        assert!((speedup - 10.0).abs() < 0.1);

        // N=100, M=25 should give √4 = 2x speedup
        let speedup = theoretical_speedup(100, 25);
        assert!((speedup - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantum_state_statistics() {
        let state = QuantumState::uniform(100);
        let stats = state.statistics();

        assert_eq!(stats.num_states, 100);
        assert!((stats.uniformity - 1.0).abs() < 0.001, "Should be uniform");
    }

    #[test]
    fn test_state_sampling() {
        // Use 8 states for more stable dynamics
        let mut state = QuantumState::uniform(8);
        // Mark 2 states
        let marked = vec![true, true, false, false, false, false, false, false];

        // For N=8, M=2: optimal iterations ≈ π/4 * √4 ≈ 1.57, so use 1-2
        state.grover_iteration(&marked);

        let mut rng = Xorshift64::new(42);
        let samples = state.sample_multiple(1000, &mut rng);

        // Count selections of marked states (indices 0 and 1)
        let marked_count = samples.iter().filter(|&&i| i == 0 || i == 1).count();

        // Should be more than uniform (uniform would be 250 for 2/8)
        // After 1 Grover iteration, expect ~40-50% probability for marked states
        assert!(
            marked_count > 300,
            "Marked states should be sampled more often: got {} (uniform ~250)",
            marked_count
        );
    }
}
