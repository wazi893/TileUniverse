//! Search substrate: manages the population of candidates.
//!
//! Provides parallel fitness evaluation and population statistics.

use rayon::prelude::*;
use std::sync::Arc;

use super::candidate::{Candidate, Rng, Xorshift64};
use super::fitness::FitnessFunction;

/// Statistics about the current population
#[derive(Clone, Debug)]
pub struct SubstrateStats {
    /// Current generation number
    pub generation: u64,
    /// Population size
    pub population_size: usize,
    /// Best fitness in population
    pub best_fitness: u8,
    /// Worst fitness in population
    pub worst_fitness: u8,
    /// Mean fitness
    pub mean_fitness: f32,
    /// Standard deviation of fitness
    pub std_fitness: f32,
    /// Number of unique fitness values
    pub unique_fitness_count: usize,
    /// Fitness histogram (256 bins)
    pub fitness_histogram: [u32; 256],
    /// Best candidate index
    pub best_index: usize,
    /// Mean Hamming distance (diversity measure)
    pub diversity: f32,
}

impl Default for SubstrateStats {
    fn default() -> Self {
        Self {
            generation: 0,
            population_size: 0,
            best_fitness: 0,
            worst_fitness: 0,
            mean_fitness: 0.0,
            std_fitness: 0.0,
            unique_fitness_count: 0,
            fitness_histogram: [0u32; 256],
            best_index: 0,
            diversity: 0.0,
        }
    }
}

impl SubstrateStats {
    /// Selection pressure: ratio of best to mean fitness
    pub fn selection_pressure(&self) -> f32 {
        if self.mean_fitness > 0.0 {
            self.best_fitness as f32 / self.mean_fitness
        } else {
            1.0
        }
    }
}

/// The search substrate - manages the population of candidates.
///
/// Designed for massive parallelism (1M-268M candidates).
pub struct SearchSubstrate {
    /// All candidates in the population
    candidates: Vec<Candidate>,
    /// The fitness function (Arc for thread-safe sharing)
    fitness_fn: Arc<dyn FitnessFunction>,
    /// Current generation number
    generation: u64,
    /// Random seed for reproducibility
    seed: u64,
}

impl SearchSubstrate {
    /// Create a new substrate with uninitialized candidates.
    ///
    /// Call `initialize_random` or `initialize_with` before using.
    pub fn new(size: usize, fitness_fn: impl FitnessFunction + 'static) -> Self {
        Self {
            candidates: vec![Candidate::default(); size],
            fitness_fn: Arc::new(fitness_fn),
            generation: 0,
            seed: 0,
        }
    }

    /// Create with a specific seed for reproducibility
    pub fn with_seed(size: usize, fitness_fn: impl FitnessFunction + 'static, seed: u64) -> Self {
        let mut substrate = Self::new(size, fitness_fn);
        substrate.seed = seed;
        substrate
    }

    /// Initialize population with random candidates
    pub fn initialize_random(&mut self, seed: u64) {
        self.seed = seed;
        self.generation = 0;

        // Parallel initialization with per-chunk RNG
        let chunk_size = (self.candidates.len() / rayon::current_num_threads()).max(1000);

        self.candidates
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                // Each chunk gets a different seed derived from the main seed
                let chunk_seed =
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let mut rng = Xorshift64::new(chunk_seed);

                for candidate in chunk.iter_mut() {
                    *candidate = Candidate::random(&mut rng);
                    candidate.generation = 0;
                }
            });
    }

    /// Initialize with a custom initialization function
    pub fn initialize_with<F>(&mut self, init_fn: F)
    where
        F: Fn(usize) -> Candidate + Send + Sync,
    {
        self.generation = 0;
        self.candidates
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, c)| {
                *c = init_fn(i);
                c.generation = 0;
            });
    }

    /// Evaluate fitness for all candidates (parallel)
    pub fn evaluate_all(&mut self) {
        let fitness_fn = Arc::clone(&self.fitness_fn);

        self.candidates.par_iter_mut().for_each(|candidate| {
            candidate.fitness = fitness_fn.evaluate(candidate);
        });
    }

    /// Evaluate fitness and return time taken
    pub fn evaluate_all_timed(&mut self) -> std::time::Duration {
        let start = std::time::Instant::now();
        self.evaluate_all();
        start.elapsed()
    }

    /// Get the best candidate in the population
    pub fn best_candidate(&self) -> &Candidate {
        self.candidates
            .iter()
            .max_by_key(|c| c.fitness)
            .expect("Population is empty")
    }

    /// Get the best candidate's index
    pub fn best_index(&self) -> usize {
        self.candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.fitness)
            .map(|(i, _)| i)
            .expect("Population is empty")
    }

    /// Get the worst candidate in the population
    pub fn worst_candidate(&self) -> &Candidate {
        self.candidates
            .iter()
            .min_by_key(|c| c.fitness)
            .expect("Population is empty")
    }

    /// Get a candidate by index
    #[inline]
    pub fn get(&self, index: usize) -> &Candidate {
        &self.candidates[index]
    }

    /// Get a mutable candidate by index
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut Candidate {
        &mut self.candidates[index]
    }

    /// Get all candidates
    #[inline]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Get mutable access to all candidates
    #[inline]
    pub fn candidates_mut(&mut self) -> &mut [Candidate] {
        &mut self.candidates
    }

    /// Population size
    #[inline]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Current generation
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Increment generation counter
    pub fn next_generation(&mut self) {
        self.generation += 1;
        for c in &mut self.candidates {
            c.generation = self.generation as u32;
        }
    }

    /// Get the fitness function
    pub fn fitness_fn(&self) -> &dyn FitnessFunction {
        self.fitness_fn.as_ref()
    }

    /// Check if the optimal solution has been found
    pub fn is_solved(&self) -> bool {
        self.fitness_fn.is_optimal(self.best_candidate())
    }

    /// Compute population statistics
    pub fn statistics(&self) -> SubstrateStats {
        if self.candidates.is_empty() {
            return SubstrateStats::default();
        }

        // Compute histogram and basic stats in parallel
        let (histogram, sum, sum_sq, best_idx, best_fit, worst_fit) = self
            .candidates
            .par_iter()
            .enumerate()
            .fold(
                || ([0u32; 256], 0u64, 0u64, 0usize, 0u8, 255u8),
                |(mut hist, sum, sum_sq, best_idx, best_fit, worst_fit), (i, c)| {
                    hist[c.fitness as usize] += 1;
                    let new_sum = sum + c.fitness as u64;
                    let new_sum_sq = sum_sq + (c.fitness as u64).pow(2);
                    let (new_best_idx, new_best_fit) = if c.fitness > best_fit {
                        (i, c.fitness)
                    } else {
                        (best_idx, best_fit)
                    };
                    let new_worst_fit = worst_fit.min(c.fitness);
                    (
                        hist,
                        new_sum,
                        new_sum_sq,
                        new_best_idx,
                        new_best_fit,
                        new_worst_fit,
                    )
                },
            )
            .reduce(
                || ([0u32; 256], 0u64, 0u64, 0usize, 0u8, 255u8),
                |(mut h1, s1, sq1, bi1, bf1, wf1), (h2, s2, sq2, bi2, bf2, wf2)| {
                    for i in 0..256 {
                        h1[i] += h2[i];
                    }
                    let (best_idx, best_fit) = if bf2 > bf1 { (bi2, bf2) } else { (bi1, bf1) };
                    (h1, s1 + s2, sq1 + sq2, best_idx, best_fit, wf1.min(wf2))
                },
            );

        let n = self.candidates.len() as f32;
        let mean = sum as f32 / n;
        let variance = (sum_sq as f32 / n) - (mean * mean);
        let std = variance.sqrt();

        let unique_count = histogram.iter().filter(|&&c| c > 0).count();

        SubstrateStats {
            generation: self.generation,
            population_size: self.candidates.len(),
            best_fitness: best_fit,
            worst_fitness: worst_fit,
            mean_fitness: mean,
            std_fitness: std,
            unique_fitness_count: unique_count,
            fitness_histogram: histogram,
            best_index: best_idx,
            diversity: 0.0, // Computed separately if needed
        }
    }

    /// Compute diversity (mean pairwise Hamming distance)
    ///
    /// This is O(n²) so only samples if population is large
    pub fn compute_diversity(&self, sample_size: usize) -> f32 {
        let n = self.candidates.len();
        if n < 2 {
            return 0.0;
        }

        let actual_sample = sample_size.min(n);

        if actual_sample >= n {
            // Full computation for small populations
            let mut total_distance = 0u64;
            let mut pairs = 0u64;
            for i in 0..n {
                for j in (i + 1)..n {
                    total_distance +=
                        self.candidates[i].hamming_distance(&self.candidates[j]) as u64;
                    pairs += 1;
                }
            }
            if pairs > 0 {
                total_distance as f32 / pairs as f32
            } else {
                0.0
            }
        } else {
            // Sample-based estimation
            let mut rng = Xorshift64::new(self.generation);
            let mut total_distance = 0u64;
            let pairs = actual_sample.saturating_mul(actual_sample - 1) / 2;

            for i in 0..actual_sample {
                let idx_i = (rng.next_u64() as usize) % n;
                for _j in (i + 1)..actual_sample {
                    let idx_j = (rng.next_u64() as usize) % n;
                    total_distance +=
                        self.candidates[idx_i].hamming_distance(&self.candidates[idx_j]) as u64;
                }
            }

            if pairs > 0 {
                total_distance as f32 / pairs as f32
            } else {
                0.0
            }
        }
    }

    /// Replace population with new candidates
    pub fn replace_all(&mut self, new_candidates: Vec<Candidate>) {
        assert_eq!(
            new_candidates.len(),
            self.candidates.len(),
            "New population must be same size"
        );
        self.candidates = new_candidates;
    }

    /// Replace candidate at index
    #[inline]
    pub fn replace(&mut self, index: usize, candidate: Candidate) {
        self.candidates[index] = candidate;
    }

    /// Swap candidates at two indices
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.candidates.swap(i, j);
    }

    /// Sort population by fitness (descending)
    pub fn sort_by_fitness(&mut self) {
        self.candidates
            .par_sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));
    }

    /// Get top k candidates by fitness
    pub fn top_k(&self, k: usize) -> Vec<&Candidate> {
        let mut indices: Vec<usize> = (0..self.candidates.len()).collect();
        indices.par_sort_unstable_by(|&a, &b| {
            self.candidates[b].fitness.cmp(&self.candidates[a].fitness)
        });
        indices
            .into_iter()
            .take(k)
            .map(|i| &self.candidates[i])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::OneMax;

    #[test]
    fn test_substrate_creation() {
        let substrate = SearchSubstrate::new(1000, OneMax::standard());
        assert_eq!(substrate.len(), 1000);
        assert_eq!(substrate.generation(), 0);
    }

    #[test]
    fn test_random_initialization() {
        let mut substrate = SearchSubstrate::new(10000, OneMax::standard());
        substrate.initialize_random(42);

        // Check that candidates are different
        let first = substrate.get(0).bits;
        let different = substrate.candidates().iter().any(|c| c.bits != first);
        assert!(different, "All candidates should not be identical");
    }

    #[test]
    fn test_parallel_evaluation() {
        let mut substrate = SearchSubstrate::new(100_000, OneMax::standard());
        substrate.initialize_random(42);

        let duration = substrate.evaluate_all_timed();
        println!("Evaluated 100K candidates in {:?}", duration);

        // Check that fitness values were set
        let has_nonzero_fitness = substrate.candidates().iter().any(|c| c.fitness > 0);
        assert!(has_nonzero_fitness, "Should have non-zero fitness values");
    }

    #[test]
    fn test_best_candidate() {
        let mut substrate = SearchSubstrate::new(1000, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let best = substrate.best_candidate();
        assert!(best.fitness > 0);

        // Best should have fitness >= all others
        for c in substrate.candidates() {
            assert!(best.fitness >= c.fitness);
        }
    }

    #[test]
    fn test_statistics() {
        let mut substrate = SearchSubstrate::new(10000, OneMax::standard());
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let stats = substrate.statistics();
        assert_eq!(stats.population_size, 10000);
        assert!(stats.mean_fitness > 0.0);
        assert!(stats.best_fitness >= stats.worst_fitness);
        assert!(stats.std_fitness >= 0.0);
    }

    #[test]
    fn test_diversity() {
        let mut substrate = SearchSubstrate::new(100, OneMax::standard());
        substrate.initialize_random(42);

        let diversity = substrate.compute_diversity(100);
        // Random 64-bit candidates should have ~32 bit diversity
        assert!(diversity > 20.0 && diversity < 44.0, "Got {}", diversity);
    }

    #[test]
    fn test_sort_by_fitness() {
        let mut substrate = SearchSubstrate::new(1000, OneMax::standard());
        substrate.initialize_random(42);
        substrate.evaluate_all();
        substrate.sort_by_fitness();

        // Check sorted order
        for i in 1..substrate.len() {
            assert!(substrate.get(i - 1).fitness >= substrate.get(i).fitness);
        }
    }

    #[test]
    fn test_top_k() {
        let mut substrate = SearchSubstrate::new(1000, OneMax::standard());
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let top = substrate.top_k(10);
        assert_eq!(top.len(), 10);

        // All top-10 should have fitness >= the 10th one
        let min_top = top.iter().map(|c| c.fitness).min().unwrap();
        for c in substrate.candidates() {
            if c.fitness > min_top {
                assert!(top.iter().any(|t| t.bits == c.bits));
            }
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // Skip in debug mode - stack overflow due to larger stack frames
    fn test_1m_scale() {
        // Test at 1M scale (Phase 1 gate check target)
        let mut substrate = SearchSubstrate::new(1_000_000, OneMax::standard());
        substrate.initialize_random(42);

        let start = std::time::Instant::now();
        substrate.evaluate_all();
        let duration = start.elapsed();

        println!("1M candidates evaluated in {:?}", duration);
        assert!(
            duration.as_secs_f32() < 1.0,
            "Should evaluate 1M in <1 second, took {:?}",
            duration
        );

        let stats = substrate.statistics();
        println!("Best fitness: {}", stats.best_fitness);
        println!("Mean fitness: {:.2}", stats.mean_fitness);
    }
}
