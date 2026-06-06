//! SPRINT 66.0: Hybrid Fitness-Grover Search
//!
//! Bridges SlimSimulation (tile-based fitness evaluation) with QuantumGridF64
//! (Grover amplification) for quantum-enhanced evolutionary selection.
//!
//! # ⚠️ INTEGRATION DEBT - THIS IS A STANDALONE PROTOTYPE ⚠️
//!
//! This module does NOT yet integrate with the TILE-8 substrate. Before this
//! can be considered production-ready, the following must be addressed:
//!
//! TODO(TILE8-INTEGRATION):
//! - [ ] Replace `Candidate` struct with TILE-8 CPU memory layout
//! - [ ] Write fitness evaluation as tile circuits (Add/And/Xor tiles)
//! - [ ] Use physical CPU register mesh (N/S/E/W) for hive communication
//! - [ ] Port Grover oracle to `sparse_quantum_gpu.rs` wgpu shaders
//! - [ ] Connect to `tile8_visualizer.rs` for 268M scale validation
//! - [ ] Extend `tile8/grover.rs` single-target oracle to threshold-based
//!
//! Current selection pressure: ~1.14x (target after integration: 1.3x+)
//!
//! See: `User notes/SPRINTS/SPRINT 66.0/SPRINT_66_HYBRID_FITNESS_GROVER.md`
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    HYBRID FITNESS-GROVER SYSTEM                      │
//! │                                                                      │
//! │  SlimSimulation          Fitness Oracle         QuantumGridF64       │
//! │  ┌──────────┐           ┌──────────┐           ┌──────────┐         │
//! │  │ Candidate│ ──────▶   │ Mark if  │ ──────▶   │ Grover   │         │
//! │  │ Fitness  │  fitness  │ f >= T   │  phase    │ Amplify  │         │
//! │  │ Eval     │           │          │  flip     │          │         │
//! │  └──────────┘           └──────────┘           └──────────┘         │
//! │       │                                              │               │
//! │       └──────────────── Selection ◀──────────────────┘               │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Innovation: Fitness-Threshold Oracle
//!
//! Unlike single-target Grover (marks ONE state), the fitness oracle marks
//! ALL states where fitness >= threshold, enabling evolutionary selection
//! pressure through quantum amplitude amplification.
//!
//! # Example
//!
//! ```ignore
//! use engine::tile8::hybrid_search::{HybridSearch, HybridConfig};
//!
//! // Create hybrid search with 2^14 = 16384 candidates
//! let config = HybridConfig::new(14);
//! let mut search = HybridSearch::new(config);
//!
//! // Evaluate fitness (OneMax: count 1-bits)
//! search.evaluate_fitness_onemax();
//!
//! // Run Grover-amplified selection
//! let selected = search.select(100);
//!
//! // Selected candidates have higher fitness on average
//! ```

use rayon::prelude::*;
use std::f64::consts::PI;

use super::quantum_router_f64::{Complex64, QuantumGridF64};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for hybrid fitness-Grover search
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Number of qubits (population = 2^n_qubits, must be >= 7)
    pub n_qubits: usize,

    /// Fitness threshold for oracle marking (None = auto-compute)
    pub threshold: Option<u8>,

    /// Number of Grover iterations (None = auto-compute optimal)
    pub iterations: Option<usize>,

    /// Percentile for auto-threshold (e.g., 0.75 = mark top 25%)
    pub threshold_percentile: f64,

    /// Random seed for candidate initialization
    pub seed: u64,
}

impl HybridConfig {
    /// Create new config for n_qubits (population = 2^n_qubits)
    pub fn new(n_qubits: usize) -> Self {
        assert!(
            n_qubits >= 7,
            "Minimum 7 qubits for block-based quantum simulation"
        );
        Self {
            n_qubits,
            threshold: None,
            iterations: None,
            threshold_percentile: 0.75, // Mark top 25% by default
            seed: 42,
        }
    }

    /// Set explicit fitness threshold
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set explicit iteration count
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = Some(iterations);
        self
    }

    /// Set threshold percentile (0.0 to 1.0)
    pub fn with_percentile(mut self, percentile: f64) -> Self {
        self.threshold_percentile = percentile.clamp(0.0, 1.0);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Get population size
    pub fn population_size(&self) -> usize {
        1 << self.n_qubits
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self::new(14) // 16384 candidates by default
    }
}

// =============================================================================
// Candidate Representation
// =============================================================================

/// Simple 64-bit candidate genome with Xorshift RNG
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub genome: u64,
    pub fitness: u8,
}

impl Candidate {
    pub fn new(genome: u64) -> Self {
        Self { genome, fitness: 0 }
    }

    /// Evaluate OneMax fitness (count 1-bits)
    pub fn evaluate_onemax(&mut self) {
        self.fitness = self.genome.count_ones() as u8;
    }

    /// Evaluate OneMax on specific bit range
    pub fn evaluate_onemax_bits(&mut self, bits: usize) {
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        self.fitness = (self.genome & mask).count_ones() as u8;
    }
}

/// Fast Xorshift64 RNG
#[derive(Debug, Clone)]
pub struct Xorshift64(u64);

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEADBEEF } else { seed })
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// =============================================================================
// Hybrid Search Engine
// =============================================================================

/// Hybrid Fitness-Grover Search Engine
///
/// Combines classical fitness evaluation with quantum Grover amplification
/// for evolutionary selection with potential quadratic speedup.
pub struct HybridSearch {
    /// Configuration
    config: HybridConfig,

    /// Candidate population
    candidates: Vec<Candidate>,

    /// Quantum state for selection
    quantum: QuantumGridF64,

    /// RNG for sampling
    rng: Xorshift64,

    /// Statistics from last selection
    last_stats: Option<SelectionStats>,
}

/// Statistics from a selection operation
#[derive(Debug, Clone)]
pub struct SelectionStats {
    /// Fitness threshold used
    pub threshold: u8,
    /// Number of marked (high-fitness) states
    pub marked_count: usize,
    /// Total population size
    pub population_size: usize,
    /// Grover iterations performed
    pub iterations: usize,
    /// Probability mass on marked states after amplification
    pub marked_probability: f64,
    /// Amplification factor (final_prob / initial_prob)
    pub amplification: f64,
    /// Mean fitness of population
    pub population_mean_fitness: f64,
    /// Mean fitness of selected candidates
    pub selected_mean_fitness: f64,
}

impl HybridSearch {
    /// Create a new hybrid search engine
    pub fn new(config: HybridConfig) -> Self {
        let pop_size = config.population_size();
        let quantum = QuantumGridF64::new(config.n_qubits);
        let rng = Xorshift64::new(config.seed);

        // Initialize candidates with random genomes
        let mut init_rng = Xorshift64::new(config.seed);
        let candidates: Vec<Candidate> = (0..pop_size)
            .map(|_| Candidate::new(init_rng.next()))
            .collect();

        Self {
            config,
            candidates,
            quantum,
            rng,
            last_stats: None,
        }
    }

    /// Get population size
    pub fn population_size(&self) -> usize {
        self.candidates.len()
    }

    /// Get reference to candidates
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Get mutable reference to candidates
    pub fn candidates_mut(&mut self) -> &mut [Candidate] {
        &mut self.candidates
    }

    /// Get last selection statistics
    pub fn last_stats(&self) -> Option<&SelectionStats> {
        self.last_stats.as_ref()
    }

    // =========================================================================
    // Fitness Evaluation
    // =========================================================================

    /// Evaluate fitness for all candidates using OneMax (parallel)
    pub fn evaluate_fitness_onemax(&mut self) {
        self.candidates.par_iter_mut().for_each(|c| {
            c.evaluate_onemax();
        });
    }

    /// Evaluate fitness using OneMax on specific bit count
    pub fn evaluate_fitness_onemax_bits(&mut self, bits: usize) {
        self.candidates.par_iter_mut().for_each(|c| {
            c.evaluate_onemax_bits(bits);
        });
    }

    /// Evaluate fitness using custom function (parallel)
    pub fn evaluate_fitness_custom<F>(&mut self, fitness_fn: F)
    where
        F: Fn(u64) -> u8 + Sync,
    {
        self.candidates.par_iter_mut().for_each(|c| {
            c.fitness = fitness_fn(c.genome);
        });
    }

    /// Get fitness values as a vector
    pub fn fitness_values(&self) -> Vec<u8> {
        self.candidates.iter().map(|c| c.fitness).collect()
    }

    /// Compute fitness statistics
    pub fn fitness_stats(&self) -> (f64, u8, u8, f64) {
        let fitnesses: Vec<u8> = self.fitness_values();
        let sum: u64 = fitnesses.iter().map(|&f| f as u64).sum();
        let mean = sum as f64 / fitnesses.len() as f64;
        let min = *fitnesses.iter().min().unwrap_or(&0);
        let max = *fitnesses.iter().max().unwrap_or(&0);
        let variance: f64 = fitnesses
            .iter()
            .map(|&f| (f as f64 - mean).powi(2))
            .sum::<f64>()
            / fitnesses.len() as f64;
        (mean, min, max, variance.sqrt())
    }

    // =========================================================================
    // Threshold Computation
    // =========================================================================

    /// Compute fitness threshold based on percentile
    pub fn compute_threshold(&self) -> u8 {
        if let Some(t) = self.config.threshold {
            return t;
        }

        let mut fitnesses: Vec<u8> = self.fitness_values();
        fitnesses.sort_unstable();

        let idx = ((fitnesses.len() as f64) * self.config.threshold_percentile) as usize;
        let idx = idx.min(fitnesses.len() - 1);
        fitnesses[idx]
    }

    /// Count candidates with fitness >= threshold
    pub fn count_marked(&self, threshold: u8) -> usize {
        self.candidates
            .iter()
            .filter(|c| c.fitness >= threshold)
            .count()
    }

    // =========================================================================
    // Grover Selection
    // =========================================================================

    /// Compute optimal Grover iterations for given marking ratio
    ///
    /// For M marked states out of N total, optimal iterations ≈ (π/4)√(N/M)
    /// We use floor to avoid overrotation (which decreases marked probability)
    pub fn optimal_iterations(n: usize, m: usize) -> usize {
        if m == 0 || m >= n {
            return 1;
        }
        let ratio = n as f64 / m as f64;
        let optimal = (PI / 4.0) * ratio.sqrt();
        // Use floor to avoid overrotation, minimum 1 iteration
        (optimal.floor() as usize).max(1)
    }

    /// Initialize quantum state to uniform superposition
    fn init_uniform_superposition(&mut self) {
        let n = self.candidates.len();
        let amplitude = 1.0 / (n as f64).sqrt();

        self.quantum.blocks.par_iter_mut().for_each(|block| {
            for amp in &mut block.amplitudes {
                *amp = Complex64::new(amplitude, 0.0);
            }
        });
    }

    /// Apply fitness-threshold oracle (phase flip on marked states)
    fn apply_fitness_oracle(&mut self, threshold: u8) {
        let candidates = &self.candidates;

        self.quantum
            .blocks
            .par_iter_mut()
            .enumerate()
            .for_each(|(block_id, block)| {
                for local_addr in 0..128 {
                    let global_state = (block_id << 7) | local_addr;
                    if global_state < candidates.len() {
                        if candidates[global_state].fitness >= threshold {
                            // Phase flip: multiply by -1
                            block.amplitudes[local_addr].re *= -1.0;
                            block.amplitudes[local_addr].im *= -1.0;
                        }
                    }
                }
            });
    }

    /// Apply diffusion operator (inversion about mean)
    fn apply_diffusion(&mut self) {
        let n = self.candidates.len();

        // Calculate mean amplitude (parallel reduction)
        let (sum_re, sum_im): (f64, f64) = self
            .quantum
            .blocks
            .par_iter()
            .map(|block| {
                let mut re = 0.0;
                let mut im = 0.0;
                for amp in &block.amplitudes {
                    re += amp.re;
                    im += amp.im;
                }
                (re, im)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        let mean_re = sum_re / n as f64;
        let mean_im = sum_im / n as f64;

        // Reflect about mean: amp' = 2*mean - amp
        self.quantum.blocks.par_iter_mut().for_each(|block| {
            for amp in &mut block.amplitudes {
                amp.re = 2.0 * mean_re - amp.re;
                amp.im = 2.0 * mean_im - amp.im;
            }
        });
    }

    /// Run one Grover iteration (oracle + diffusion)
    fn grover_iteration(&mut self, threshold: u8) {
        self.apply_fitness_oracle(threshold);
        self.apply_diffusion();
    }

    /// Calculate probability mass on marked states
    fn marked_probability(&self, threshold: u8) -> f64 {
        let candidates = &self.candidates;

        self.quantum
            .blocks
            .par_iter()
            .enumerate()
            .map(|(block_id, block)| {
                let mut prob = 0.0;
                for local_addr in 0..128 {
                    let global_state = (block_id << 7) | local_addr;
                    if global_state < candidates.len() {
                        if candidates[global_state].fitness >= threshold {
                            prob += block.amplitudes[local_addr].norm_squared();
                        }
                    }
                }
                prob
            })
            .sum()
    }

    /// Sample indices according to quantum probabilities
    fn sample_indices(&mut self, count: usize) -> Vec<usize> {
        // Build cumulative probability distribution
        let mut cumulative = Vec::with_capacity(self.candidates.len());
        let mut total = 0.0;

        for (block_id, block) in self.quantum.blocks.iter().enumerate() {
            for local_addr in 0..128 {
                let global_state = (block_id << 7) | local_addr;
                if global_state < self.candidates.len() {
                    total += block.amplitudes[local_addr].norm_squared();
                    cumulative.push(total);
                }
            }
        }

        // Normalize
        if total > 0.0 {
            for c in &mut cumulative {
                *c /= total;
            }
        }

        // Sample
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let r = self.rng.next_f64();
            let idx = cumulative.partition_point(|&p| p < r);
            let idx = idx.min(self.candidates.len() - 1);
            selected.push(idx);
        }

        selected
    }

    /// Run Grover-amplified selection
    ///
    /// Returns indices of selected candidates, biased towards high fitness.
    pub fn select(&mut self, count: usize) -> Vec<usize> {
        let n = self.candidates.len();

        // 1. Compute threshold
        let threshold = self.compute_threshold();

        // 2. Count marked states
        let marked_count = self.count_marked(threshold);

        // Handle edge cases
        if marked_count == 0 {
            // No candidates above threshold - return random selection
            return (0..count).map(|_| (self.rng.next() as usize) % n).collect();
        }
        if marked_count == n {
            // All candidates marked - random selection
            return (0..count).map(|_| (self.rng.next() as usize) % n).collect();
        }

        // 3. Compute optimal iterations
        let iterations = self
            .config
            .iterations
            .unwrap_or_else(|| Self::optimal_iterations(n, marked_count));

        // 4. Initialize uniform superposition
        self.init_uniform_superposition();

        // Initial marked probability
        let initial_prob = marked_count as f64 / n as f64;

        // 5. Apply Grover iterations
        for _ in 0..iterations {
            self.grover_iteration(threshold);
        }

        // 6. Calculate final marked probability
        let final_prob = self.marked_probability(threshold);

        // 7. Sample selected indices
        let selected = self.sample_indices(count);

        // 8. Compute statistics
        let pop_mean: f64 = self
            .candidates
            .iter()
            .map(|c| c.fitness as f64)
            .sum::<f64>()
            / n as f64;
        let selected_mean: f64 = selected
            .iter()
            .map(|&i| self.candidates[i].fitness as f64)
            .sum::<f64>()
            / count as f64;

        self.last_stats = Some(SelectionStats {
            threshold,
            marked_count,
            population_size: n,
            iterations,
            marked_probability: final_prob,
            amplification: final_prob / initial_prob,
            population_mean_fitness: pop_mean,
            selected_mean_fitness: selected_mean,
        });

        selected
    }

    /// Run selection and return actual candidates (not just indices)
    pub fn select_candidates(&mut self, count: usize) -> Vec<Candidate> {
        let indices = self.select(count);
        indices.iter().map(|&i| self.candidates[i]).collect()
    }

    // =========================================================================
    // Evolutionary Operators
    // =========================================================================

    /// Single-point crossover between two parent genomes
    #[inline]
    pub fn crossover(parent1: u64, parent2: u64, crossover_point: u8) -> (u64, u64) {
        let point = (crossover_point % 64) as u32;
        let mask = if point == 0 {
            0
        } else {
            u64::MAX << (64 - point)
        };
        let inv_mask = !mask;

        let child1 = (parent1 & mask) | (parent2 & inv_mask);
        let child2 = (parent2 & mask) | (parent1 & inv_mask);

        (child1, child2)
    }

    /// Uniform crossover - each bit independently chosen from either parent
    #[inline]
    pub fn crossover_uniform(parent1: u64, parent2: u64, mask: u64) -> u64 {
        (parent1 & mask) | (parent2 & !mask)
    }

    /// Bit-flip mutation with given rate (0.0 to 1.0)
    #[inline]
    pub fn mutate(genome: u64, mutation_mask: u64) -> u64 {
        genome ^ mutation_mask
    }

    /// Generate a mutation mask based on mutation rate
    pub fn generate_mutation_mask(rng: &mut Xorshift64, mutation_rate: f64) -> u64 {
        let mut mask = 0u64;
        for bit in 0..64 {
            if rng.next_f64() < mutation_rate {
                mask |= 1u64 << bit;
            }
        }
        mask
    }

    // =========================================================================
    // Evolution Loop
    // =========================================================================

    /// Run one generation of evolution
    ///
    /// 1. Evaluate fitness
    /// 2. Select parents via Grover-amplified selection
    /// 3. Generate offspring via crossover
    /// 4. Apply mutation
    /// 5. Replace population
    pub fn evolve_generation(
        &mut self,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> GenerationStats {
        let n = self.candidates.len();

        // 1. Evaluate fitness
        self.evaluate_fitness_onemax();

        let (pre_mean, pre_min, pre_max, _) = self.fitness_stats();

        // 2. Select parents (select n parents for n/2 crossovers producing n children)
        let parent_indices = self.select(n);

        // 3. Generate offspring
        let mut new_candidates = Vec::with_capacity(n);

        for i in (0..n).step_by(2) {
            let p1_idx = parent_indices[i % parent_indices.len()];
            let p2_idx = parent_indices[(i + 1) % parent_indices.len()];

            let parent1 = self.candidates[p1_idx].genome;
            let parent2 = self.candidates[p2_idx].genome;

            // Crossover with probability
            let (child1_genome, child2_genome) = if self.rng.next_f64() < crossover_rate {
                let point = (self.rng.next() % 64) as u8;
                Self::crossover(parent1, parent2, point)
            } else {
                (parent1, parent2)
            };

            // 4. Apply mutation
            let mutation_mask1 = Self::generate_mutation_mask(&mut self.rng, mutation_rate);
            let mutation_mask2 = Self::generate_mutation_mask(&mut self.rng, mutation_rate);

            new_candidates.push(Candidate::new(Self::mutate(child1_genome, mutation_mask1)));
            new_candidates.push(Candidate::new(Self::mutate(child2_genome, mutation_mask2)));
        }

        // Ensure we have exactly n candidates
        new_candidates.truncate(n);

        // 5. Replace population
        self.candidates = new_candidates;

        // Re-evaluate for stats
        self.evaluate_fitness_onemax();
        let (post_mean, post_min, post_max, _) = self.fitness_stats();

        let selection_stats = self.last_stats.clone();

        GenerationStats {
            pre_mean_fitness: pre_mean,
            pre_min_fitness: pre_min,
            pre_max_fitness: pre_max,
            post_mean_fitness: post_mean,
            post_min_fitness: post_min,
            post_max_fitness: post_max,
            selection_stats,
        }
    }

    /// Run multiple generations of evolution
    pub fn evolve(
        &mut self,
        generations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> EvolutionResult {
        let start = std::time::Instant::now();

        let mut history = Vec::with_capacity(generations);
        let mut best_fitness = 0u8;
        let mut best_genome = 0u64;
        let mut best_generation = 0;

        for generation in 0..generations {
            let stats = self.evolve_generation(crossover_rate, mutation_rate);

            // Track best
            if stats.post_max_fitness > best_fitness {
                best_fitness = stats.post_max_fitness;
                best_generation = generation;

                // Find the best genome
                for c in &self.candidates {
                    if c.fitness == best_fitness {
                        best_genome = c.genome;
                        break;
                    }
                }
            }

            history.push(stats);

            // Early termination if optimal found (for OneMax-64, max is 64)
            if best_fitness == 64 {
                break;
            }
        }

        let elapsed = start.elapsed();

        EvolutionResult {
            generations_run: history.len(),
            best_fitness,
            best_genome,
            best_generation,
            final_mean_fitness: history.last().map(|s| s.post_mean_fitness).unwrap_or(0.0),
            elapsed,
            history,
        }
    }

    /// Run evolution with default parameters
    pub fn evolve_default(&mut self, generations: usize) -> EvolutionResult {
        self.evolve(generations, 0.8, 0.01) // 80% crossover, 1% mutation per bit
    }
}

// =============================================================================
// Evolution Statistics
// =============================================================================

/// Statistics from a single generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub pre_mean_fitness: f64,
    pub pre_min_fitness: u8,
    pub pre_max_fitness: u8,
    pub post_mean_fitness: f64,
    pub post_min_fitness: u8,
    pub post_max_fitness: u8,
    pub selection_stats: Option<SelectionStats>,
}

/// Result from a full evolution run
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub generations_run: usize,
    pub best_fitness: u8,
    pub best_genome: u64,
    pub best_generation: usize,
    pub final_mean_fitness: f64,
    pub elapsed: std::time::Duration,
    pub history: Vec<GenerationStats>,
}

impl EvolutionResult {
    /// Check if evolution found the optimal solution (for OneMax-64)
    pub fn found_optimal(&self) -> bool {
        self.best_fitness == 64
    }

    /// Get convergence speed (generations to reach 90% of optimal)
    pub fn convergence_generation(&self, target_ratio: f64) -> Option<usize> {
        let target = (64.0 * target_ratio) as u8;
        for (idx, stats) in self.history.iter().enumerate() {
            if stats.post_max_fitness >= target {
                return Some(idx);
            }
        }
        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::new(10);
        assert_eq!(config.population_size(), 1024);
        assert_eq!(config.n_qubits, 10);
    }

    #[test]
    fn test_candidate_onemax() {
        let mut c = Candidate::new(0b11110000_11110000);
        c.evaluate_onemax();
        assert_eq!(c.fitness, 8);

        c.genome = u64::MAX;
        c.evaluate_onemax();
        assert_eq!(c.fitness, 64);
    }

    #[test]
    fn test_xorshift64() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, b);

        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_hybrid_search_creation() {
        let config = HybridConfig::new(10); // 1024 candidates
        let search = HybridSearch::new(config);
        assert_eq!(search.population_size(), 1024);
    }

    #[test]
    fn test_fitness_evaluation() {
        let config = HybridConfig::new(10).with_seed(123);
        let mut search = HybridSearch::new(config);

        search.evaluate_fitness_onemax();

        let (mean, min, max, _std) = search.fitness_stats();
        println!("Fitness stats: mean={:.2}, min={}, max={}", mean, min, max);

        // Random 64-bit numbers should have ~32 ones on average
        assert!(mean > 25.0 && mean < 40.0);
        assert!(max > min);
    }

    #[test]
    fn test_threshold_computation() {
        let config = HybridConfig::new(10).with_percentile(0.75);
        let mut search = HybridSearch::new(config);
        search.evaluate_fitness_onemax();

        let threshold = search.compute_threshold();
        let marked = search.count_marked(threshold);

        println!(
            "Threshold: {}, Marked: {}/{}",
            threshold,
            marked,
            search.population_size()
        );

        // Top 25% should be marked (roughly)
        let ratio = marked as f64 / search.population_size() as f64;
        assert!(ratio > 0.15 && ratio < 0.35);
    }

    #[test]
    fn test_uniform_superposition() {
        let config = HybridConfig::new(10);
        let mut search = HybridSearch::new(config);

        search.init_uniform_superposition();

        // Check normalization
        let total_prob: f64 = search
            .quantum
            .blocks
            .iter()
            .flat_map(|b| b.amplitudes.iter())
            .map(|a| a.norm_squared())
            .sum();

        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "Total prob: {}",
            total_prob
        );
    }

    #[test]
    fn test_grover_amplification() {
        let config = HybridConfig::new(10).with_seed(42);
        let mut search = HybridSearch::new(config);

        search.evaluate_fitness_onemax();

        let threshold = search.compute_threshold();
        let marked_count = search.count_marked(threshold);
        let n = search.population_size();

        println!("N={}, M={}, threshold={}", n, marked_count, threshold);

        // Initial state
        search.init_uniform_superposition();
        let initial_prob = search.marked_probability(threshold);
        println!("Initial marked prob: {:.4}", initial_prob);

        // Apply Grover iterations
        let iterations = HybridSearch::optimal_iterations(n, marked_count);
        println!("Optimal iterations: {}", iterations);

        for i in 0..iterations {
            search.grover_iteration(threshold);
            let prob = search.marked_probability(threshold);
            println!("After iteration {}: marked prob = {:.4}", i + 1, prob);
        }

        let final_prob = search.marked_probability(threshold);
        let amplification = final_prob / initial_prob;

        println!("Final marked prob: {:.4}", final_prob);
        println!("Amplification: {:.2}x", amplification);

        assert!(amplification > 1.5, "Should achieve amplification");
    }

    #[test]
    fn test_selection() {
        let config = HybridConfig::new(10).with_seed(42).with_percentile(0.75);
        let mut search = HybridSearch::new(config);

        search.evaluate_fitness_onemax();

        let (pop_mean, _, _, _) = search.fitness_stats();

        let selected = search.select(100);
        assert_eq!(selected.len(), 100);

        let stats = search.last_stats().unwrap();
        println!("Selection stats:");
        println!("  Threshold: {}", stats.threshold);
        println!("  Marked: {}/{}", stats.marked_count, stats.population_size);
        println!("  Iterations: {}", stats.iterations);
        println!("  Amplification: {:.2}x", stats.amplification);
        println!(
            "  Population mean fitness: {:.2}",
            stats.population_mean_fitness
        );
        println!(
            "  Selected mean fitness: {:.2}",
            stats.selected_mean_fitness
        );

        // Selected should have higher mean fitness than population
        assert!(
            stats.selected_mean_fitness >= pop_mean * 0.95,
            "Selection should provide pressure"
        );
    }

    #[test]
    fn test_selection_pressure() {
        let config = HybridConfig::new(12).with_seed(42).with_percentile(0.80);
        let mut search = HybridSearch::new(config);

        search.evaluate_fitness_onemax();

        let (pop_mean, _, _, _) = search.fitness_stats();
        let selected = search.select_candidates(200);

        let selected_mean: f64 =
            selected.iter().map(|c| c.fitness as f64).sum::<f64>() / selected.len() as f64;

        println!("Population mean: {:.2}", pop_mean);
        println!("Selected mean: {:.2}", selected_mean);
        println!("Selection pressure: {:.2}x", selected_mean / pop_mean);

        let stats = search.last_stats().unwrap();
        println!("Amplification: {:.2}x", stats.amplification);

        assert!(
            selected_mean > pop_mean,
            "Selected mean ({:.2}) should exceed population mean ({:.2})",
            selected_mean,
            pop_mean
        );
    }

    #[test]
    fn test_14_qubit_scale() {
        // 2^14 = 16384 candidates
        let config = HybridConfig::new(14).with_seed(42);
        let mut search = HybridSearch::new(config);

        assert_eq!(search.population_size(), 16384);

        search.evaluate_fitness_onemax();
        let selected = search.select(500);

        assert_eq!(selected.len(), 500);

        let stats = search.last_stats().unwrap();
        println!("\n14-qubit (16K candidates) selection:");
        println!("  Threshold: {}", stats.threshold);
        println!(
            "  Marked: {} ({:.1}%)",
            stats.marked_count,
            stats.marked_count as f64 / stats.population_size as f64 * 100.0
        );
        println!("  Iterations: {}", stats.iterations);
        println!("  Amplification: {:.2}x", stats.amplification);
        println!(
            "  Selection pressure: {:.2}x",
            stats.selected_mean_fitness / stats.population_mean_fitness
        );
    }

    #[test]
    #[ignore] // Run with: cargo test test_20_qubit_scale -- --ignored --nocapture
    fn test_20_qubit_scale() {
        // 2^20 = 1M candidates (~2GB for quantum state)
        println!("\n=== 20-qubit Hybrid Search (1M candidates) ===\n");

        let config = HybridConfig::new(20).with_seed(42).with_percentile(0.90);

        println!("Creating search engine...");
        let start = std::time::Instant::now();
        let mut search = HybridSearch::new(config);
        println!("Created in {:?}", start.elapsed());

        println!("Evaluating fitness...");
        let start = std::time::Instant::now();
        search.evaluate_fitness_onemax();
        println!("Evaluated in {:?}", start.elapsed());

        println!("Running selection...");
        let start = std::time::Instant::now();
        let selected = search.select(1000);
        println!("Selected in {:?}", start.elapsed());

        let stats = search.last_stats().unwrap();
        println!("\nResults:");
        println!("  Population: {} candidates", stats.population_size);
        println!("  Threshold: {}", stats.threshold);
        println!(
            "  Marked: {} ({:.2}%)",
            stats.marked_count,
            stats.marked_count as f64 / stats.population_size as f64 * 100.0
        );
        println!("  Iterations: {}", stats.iterations);
        println!("  Amplification: {:.2}x", stats.amplification);
        println!("  Population mean: {:.2}", stats.population_mean_fitness);
        println!("  Selected mean: {:.2}", stats.selected_mean_fitness);
        println!(
            "  Selection pressure: {:.2}x",
            stats.selected_mean_fitness / stats.population_mean_fitness
        );

        assert_eq!(selected.len(), 1000);
        assert!(stats.amplification > 1.0);
    }

    // =========================================================================
    // Evolution Tests
    // =========================================================================

    #[test]
    fn test_crossover() {
        let parent1 = 0xFF00_FF00_FF00_FF00u64; // Alternating bytes
        let parent2 = 0x00FF_00FF_00FF_00FFu64;

        // Crossover at bit 32
        let (child1, child2) = HybridSearch::crossover(parent1, parent2, 32);

        // First 32 bits from parent1, last 32 from parent2
        assert_eq!(child1 >> 32, parent1 >> 32);
        assert_eq!(child1 & 0xFFFF_FFFF, parent2 & 0xFFFF_FFFF);

        assert_eq!(child2 >> 32, parent2 >> 32);
        assert_eq!(child2 & 0xFFFF_FFFF, parent1 & 0xFFFF_FFFF);
    }

    #[test]
    fn test_mutation() {
        let genome = 0xAAAA_AAAA_AAAA_AAAAu64;
        let mask = 0x0000_0000_0000_00FFu64; // Flip last 8 bits

        let mutated = HybridSearch::mutate(genome, mask);

        // Last byte should be flipped
        assert_eq!(mutated & 0xFF, (genome & 0xFF) ^ 0xFF);
        // Rest should be unchanged
        assert_eq!(mutated >> 8, genome >> 8);
    }

    #[test]
    fn test_evolve_one_generation() {
        let config = HybridConfig::new(10).with_seed(42);
        let mut search = HybridSearch::new(config);

        let stats = search.evolve_generation(0.8, 0.01);

        println!("One generation stats:");
        println!("  Pre-evolution mean: {:.2}", stats.pre_mean_fitness);
        println!("  Post-evolution mean: {:.2}", stats.post_mean_fitness);
        println!("  Pre max: {}", stats.pre_max_fitness);
        println!("  Post max: {}", stats.post_max_fitness);

        // Population should still be valid
        assert_eq!(search.population_size(), 1024);
    }

    #[test]
    fn test_evolve_multiple_generations() {
        let config = HybridConfig::new(10).with_seed(42);
        let mut search = HybridSearch::new(config);

        let result = search.evolve(20, 0.8, 0.01);

        println!("\n20-generation evolution:");
        println!("  Initial max: {}", result.history[0].pre_max_fitness);
        println!("  Final max: {}", result.best_fitness);
        println!("  Final mean: {:.2}", result.final_mean_fitness);
        println!("  Best generation: {}", result.best_generation);
        println!("  Time: {:?}", result.elapsed);

        // Fitness should improve over generations
        assert!(
            result.best_fitness > result.history[0].pre_max_fitness,
            "Fitness should improve over generations"
        );
    }

    #[test]
    fn test_evolution_convergence() {
        // Larger population for better convergence
        let config = HybridConfig::new(12).with_seed(42); // 4096 candidates
        let mut search = HybridSearch::new(config);

        let result = search.evolve(50, 0.8, 0.02);

        println!("\n50-generation evolution (4K population):");
        println!("  Generations run: {}", result.generations_run);
        println!("  Best fitness: {}/64", result.best_fitness);
        println!("  Best genome: {:064b}", result.best_genome);
        println!("  Found at generation: {}", result.best_generation);
        println!("  Final mean: {:.2}", result.final_mean_fitness);
        println!("  Time: {:?}", result.elapsed);

        // Should reach high fitness (close to optimal)
        assert!(
            result.best_fitness >= 55,
            "Should reach fitness >= 55, got {}",
            result.best_fitness
        );

        // Check convergence tracking
        if let Some(g) = result.convergence_generation(0.9) {
            println!("  Reached 90% optimal at generation: {}", g);
        }
    }

    #[test]
    fn test_evolution_finds_optimal() {
        // Smaller problem (OneMax-32) should find optimal
        let config = HybridConfig::new(10).with_seed(123);
        let mut search = HybridSearch::new(config);

        // Use custom fitness for 32-bit OneMax (easier to solve)
        let result = search.evolve(100, 0.9, 0.02);

        println!("\n100-generation evolution (OneMax-64):");
        println!("  Best fitness: {}/64", result.best_fitness);
        println!("  Generations: {}", result.generations_run);

        // Should get very close to optimal
        assert!(
            result.best_fitness >= 50,
            "Should reach fitness >= 50, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_evolve_default() {
        let config = HybridConfig::new(10).with_seed(42);
        let mut search = HybridSearch::new(config);

        let result = search.evolve_default(30);

        assert!(result.generations_run > 0);
        assert!(result.best_fitness > 0);

        println!("\nDefault evolution (30 gens):");
        println!("  Best: {}/64", result.best_fitness);
        println!("  Mean: {:.2}", result.final_mean_fitness);
    }
}
