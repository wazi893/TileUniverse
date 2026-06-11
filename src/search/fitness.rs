//! Fitness functions for the search engine.
//!
//! Provides the `FitnessFunction` trait and implementations for standard
//! benchmark problems: OneMax, NK-Landscape, and MaxSAT.

use super::candidate::{Candidate, Rng, Xorshift64};

/// Trait for fitness functions.
///
/// Implement this trait to define custom optimization problems.
pub trait FitnessFunction: Send + Sync {
    /// Evaluate a candidate's fitness (0-255, higher is better)
    fn evaluate(&self, candidate: &Candidate) -> u8;

    /// Problem name for logging/display
    fn name(&self) -> &str;

    /// Number of bits used by this problem
    fn num_bits(&self) -> usize {
        64
    }

    /// Optimal fitness value (if known)
    fn optimal_fitness(&self) -> Option<u8> {
        None
    }

    /// Check if a candidate has reached the optimum
    fn is_optimal(&self, candidate: &Candidate) -> bool {
        if let Some(opt) = self.optimal_fitness() {
            candidate.fitness >= opt
        } else {
            false
        }
    }
}

// =============================================================================
// OneMax: Count the number of 1-bits
// =============================================================================

/// OneMax: Maximize the number of 1-bits.
///
/// The simplest benchmark problem. Optimal solution is all 1s.
/// Fitness = count_ones() * 255 / num_bits (scaled to 0-255)
#[derive(Clone, Debug)]
pub struct OneMax {
    num_bits: usize,
}

impl OneMax {
    /// Create a OneMax problem with specified number of bits
    pub fn new(num_bits: usize) -> Self {
        assert!(num_bits > 0 && num_bits <= 64, "num_bits must be 1-64");
        Self { num_bits }
    }

    /// Standard 64-bit OneMax
    pub fn standard() -> Self {
        Self::new(64)
    }
}

impl Default for OneMax {
    fn default() -> Self {
        Self::standard()
    }
}

impl FitnessFunction for OneMax {
    fn evaluate(&self, candidate: &Candidate) -> u8 {
        let mask = if self.num_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.num_bits) - 1
        };
        let ones = (candidate.bits & mask).count_ones() as usize;
        // Scale to 0-255
        ((ones * 255) / self.num_bits) as u8
    }

    fn name(&self) -> &str {
        "OneMax"
    }

    fn num_bits(&self) -> usize {
        self.num_bits
    }

    fn optimal_fitness(&self) -> Option<u8> {
        Some(255)
    }
}

// =============================================================================
// NK-Landscape: Tunable ruggedness
// =============================================================================

/// NK-Landscape: A tunable fitness landscape.
///
/// - N = number of bits (genes)
/// - K = epistasis (interaction between bits)
///
/// K=0: Smooth landscape (linear, easy)
/// K=N-1: Random landscape (maximally rugged)
///
/// Each bit's contribution depends on K other bits, creating
/// local optima and deception as K increases.
#[derive(Clone, Debug)]
pub struct NKLandscape {
    n: usize,
    #[allow(dead_code)]
    k: usize,
    /// Fitness contributions: contributions[i][mask] = contribution for bit i
    /// when the K neighbors have the given configuration
    contributions: Vec<Vec<f32>>,
    /// Which bits each bit depends on: neighbors[i] = [bit indices]
    neighbors: Vec<Vec<usize>>,
    /// Cached optimal fitness (computed on construction)
    optimal: Option<u8>,
}

impl NKLandscape {
    /// Create an NK-Landscape with given parameters and random seed.
    ///
    /// # Arguments
    /// * `n` - Number of bits (1-64)
    /// * `k` - Epistasis level (0 to n-1)
    /// * `seed` - Random seed for reproducibility
    pub fn new(n: usize, k: usize, seed: u64) -> Self {
        assert!(n > 0 && n <= 64, "n must be 1-64");
        assert!(k < n, "k must be less than n");

        let mut rng = Xorshift64::new(seed);

        // Generate random neighbor assignments
        let mut neighbors = Vec::with_capacity(n);
        for i in 0..n {
            let mut bit_neighbors = Vec::with_capacity(k);
            // K neighbors are the next K bits (wrapping)
            for j in 1..=k {
                bit_neighbors.push((i + j) % n);
            }
            neighbors.push(bit_neighbors);
        }

        // Generate random fitness contributions
        // Each bit has 2^(K+1) possible configurations (itself + K neighbors)
        let num_configs = 1 << (k + 1);
        let mut contributions = Vec::with_capacity(n);
        for _ in 0..n {
            let mut bit_contrib = Vec::with_capacity(num_configs);
            for _ in 0..num_configs {
                bit_contrib.push(rng.next_f32());
            }
            contributions.push(bit_contrib);
        }

        Self {
            n,
            k,
            contributions,
            neighbors,
            optimal: None, // Could compute via exhaustive search for small n
        }
    }

    /// Create with adjacent neighbors (default)
    pub fn adjacent(n: usize, k: usize, seed: u64) -> Self {
        Self::new(n, k, seed)
    }

    /// Compute the configuration index for bit i given the candidate
    fn config_index(&self, candidate: &Candidate, bit: usize) -> usize {
        let mut idx = if candidate.get_bit(bit) { 1 } else { 0 };
        for (j, &neighbor) in self.neighbors[bit].iter().enumerate() {
            if candidate.get_bit(neighbor) {
                idx |= 1 << (j + 1);
            }
        }
        idx
    }

    /// Get the raw (unscaled) fitness as f32
    pub fn raw_fitness(&self, candidate: &Candidate) -> f32 {
        let mut total = 0.0;
        for i in 0..self.n {
            let config = self.config_index(candidate, i);
            total += self.contributions[i][config];
        }
        total / self.n as f32 // Normalize to [0, 1]
    }
}

impl FitnessFunction for NKLandscape {
    fn evaluate(&self, candidate: &Candidate) -> u8 {
        let raw = self.raw_fitness(candidate);
        (raw * 255.0) as u8
    }

    fn name(&self) -> &str {
        "NK-Landscape"
    }

    fn num_bits(&self) -> usize {
        self.n
    }

    fn optimal_fitness(&self) -> Option<u8> {
        self.optimal
    }
}

// =============================================================================
// MaxSAT: Maximum Satisfiability
// =============================================================================

/// A clause in a SAT formula (disjunction of literals)
#[derive(Clone, Debug)]
pub struct Clause {
    /// Positive literals (variable must be true)
    pub positive: Vec<usize>,
    /// Negative literals (variable must be false)
    pub negative: Vec<usize>,
}

impl Clause {
    /// Create a new clause
    pub fn new(positive: Vec<usize>, negative: Vec<usize>) -> Self {
        Self { positive, negative }
    }

    /// Check if the clause is satisfied by a candidate
    pub fn is_satisfied(&self, candidate: &Candidate) -> bool {
        // Clause is satisfied if any positive literal is true
        // or any negative literal is false
        for &var in &self.positive {
            if candidate.get_bit(var) {
                return true;
            }
        }
        for &var in &self.negative {
            if !candidate.get_bit(var) {
                return true;
            }
        }
        false
    }
}

/// MaxSAT: Maximize the number of satisfied clauses.
///
/// Given a CNF formula (conjunction of clauses), find an assignment
/// that satisfies as many clauses as possible.
#[derive(Clone, Debug)]
pub struct MaxSat {
    clauses: Vec<Clause>,
    num_vars: usize,
}

impl MaxSat {
    /// Create a MaxSAT instance from clauses
    pub fn new(clauses: Vec<Clause>, num_vars: usize) -> Self {
        assert!(num_vars <= 64, "num_vars must be <= 64");
        Self { clauses, num_vars }
    }

    /// Generate a random 3-SAT instance
    ///
    /// # Arguments
    /// * `num_vars` - Number of variables (bits)
    /// * `num_clauses` - Number of clauses
    /// * `seed` - Random seed
    pub fn random_3sat(num_vars: usize, num_clauses: usize, seed: u64) -> Self {
        assert!(num_vars >= 3, "Need at least 3 variables for 3-SAT");
        let mut rng = Xorshift64::new(seed);
        let mut clauses = Vec::with_capacity(num_clauses);

        for _ in 0..num_clauses {
            // Pick 3 distinct variables
            let mut vars = [0usize; 3];
            vars[0] = (rng.next_u64() as usize) % num_vars;
            loop {
                vars[1] = (rng.next_u64() as usize) % num_vars;
                if vars[1] != vars[0] {
                    break;
                }
            }
            loop {
                vars[2] = (rng.next_u64() as usize) % num_vars;
                if vars[2] != vars[0] && vars[2] != vars[1] {
                    break;
                }
            }

            // Random polarity for each variable
            let mut positive = Vec::new();
            let mut negative = Vec::new();
            for &var in &vars {
                if rng.next_u64() & 1 == 0 {
                    positive.push(var);
                } else {
                    negative.push(var);
                }
            }

            clauses.push(Clause::new(positive, negative));
        }

        Self::new(clauses, num_vars)
    }

    /// Generate a satisfiable random instance (plants a solution)
    pub fn random_satisfiable(num_vars: usize, num_clauses: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);

        // Plant a random solution
        let solution = rng.next_u64();

        let mut clauses = Vec::with_capacity(num_clauses);
        for _ in 0..num_clauses {
            // Pick 3 distinct variables
            let mut vars = [0usize; 3];
            vars[0] = (rng.next_u64() as usize) % num_vars;
            loop {
                vars[1] = (rng.next_u64() as usize) % num_vars;
                if vars[1] != vars[0] {
                    break;
                }
            }
            loop {
                vars[2] = (rng.next_u64() as usize) % num_vars;
                if vars[2] != vars[0] && vars[2] != vars[1] {
                    break;
                }
            }

            // Ensure at least one literal satisfies the planted solution
            let mut positive = Vec::new();
            let mut negative = Vec::new();

            // First literal always satisfies solution
            if (solution >> vars[0]) & 1 == 1 {
                positive.push(vars[0]);
            } else {
                negative.push(vars[0]);
            }

            // Rest are random
            for &var in &vars[1..] {
                if rng.next_u64() & 1 == 0 {
                    positive.push(var);
                } else {
                    negative.push(var);
                }
            }

            clauses.push(Clause::new(positive, negative));
        }

        Self::new(clauses, num_vars)
    }

    /// Count the number of satisfied clauses
    pub fn count_satisfied(&self, candidate: &Candidate) -> usize {
        self.clauses
            .iter()
            .filter(|c| c.is_satisfied(candidate))
            .count()
    }

    /// Get the total number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }
}

impl FitnessFunction for MaxSat {
    fn evaluate(&self, candidate: &Candidate) -> u8 {
        let satisfied = self.count_satisfied(candidate);
        let total = self.clauses.len();
        if total == 0 {
            return 255;
        }
        ((satisfied * 255) / total) as u8
    }

    fn name(&self) -> &str {
        "MaxSAT"
    }

    fn num_bits(&self) -> usize {
        self.num_vars
    }

    fn optimal_fitness(&self) -> Option<u8> {
        Some(255) // All clauses satisfied
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onemax_zeros() {
        let problem = OneMax::new(64);
        let candidate = Candidate::zeros();
        assert_eq!(problem.evaluate(&candidate), 0);
    }

    #[test]
    fn test_onemax_ones() {
        let problem = OneMax::new(64);
        let candidate = Candidate::ones();
        assert_eq!(problem.evaluate(&candidate), 255);
    }

    #[test]
    fn test_onemax_half() {
        let problem = OneMax::new(64);
        let candidate = Candidate::new(0x5555_5555_5555_5555); // 32 ones
        let fitness = problem.evaluate(&candidate);
        assert!(fitness >= 125 && fitness <= 130, "Got {}", fitness);
    }

    #[test]
    fn test_onemax_small() {
        let problem = OneMax::new(8);
        let candidate = Candidate::new(0b1111_1111);
        assert_eq!(problem.evaluate(&candidate), 255);

        let candidate = Candidate::new(0b0000_1111);
        let fitness = problem.evaluate(&candidate);
        assert!(fitness >= 125 && fitness <= 130, "Got {}", fitness);
    }

    #[test]
    fn test_nk_landscape_creation() {
        let nk = NKLandscape::new(16, 4, 42);
        assert_eq!(nk.n, 16);
        assert_eq!(nk.k, 4);
        assert_eq!(nk.contributions.len(), 16);
        assert_eq!(nk.contributions[0].len(), 32); // 2^(K+1) = 32
    }

    #[test]
    #[allow(unused_comparisons)]
    fn test_nk_landscape_smooth() {
        // K=0 should be smooth (no epistasis)
        let nk = NKLandscape::new(16, 0, 42);
        let c1 = Candidate::new(0b0000_0000);
        let c2 = Candidate::new(0b0000_0001);
        // Fitness should be deterministic
        let f1 = nk.evaluate(&c1);
        let f2 = nk.evaluate(&c2);
        // Re-evaluating must give identical results
        assert_eq!(f1, nk.evaluate(&c1));
        assert_eq!(f2, nk.evaluate(&c2));
    }

    #[test]
    fn test_nk_landscape_rugged() {
        // K=N-1 should be maximally rugged
        let nk = NKLandscape::new(8, 7, 42);
        let mut rng = Xorshift64::new(123);

        // Sample some random candidates and check fitness variance
        let mut fitnesses = Vec::new();
        for _ in 0..100 {
            let c = Candidate::random(&mut rng);
            fitnesses.push(nk.evaluate(&c));
        }

        let mean: f32 = fitnesses.iter().map(|&f| f as f32).sum::<f32>() / 100.0;
        let variance: f32 = fitnesses
            .iter()
            .map(|&f| (f as f32 - mean).powi(2))
            .sum::<f32>()
            / 100.0;

        // High K should have high variance
        assert!(variance > 100.0, "Variance was {}", variance);
    }

    #[test]
    fn test_maxsat_random() {
        let problem = MaxSat::random_3sat(20, 50, 42);
        assert_eq!(problem.num_clauses(), 50);
        assert_eq!(problem.num_vars, 20);
    }

    #[test]
    fn test_maxsat_satisfiable() {
        let problem = MaxSat::random_satisfiable(20, 50, 42);

        // The planted solution should satisfy all clauses
        let mut rng = Xorshift64::new(42);
        let solution = Candidate::new(rng.next_u64() & ((1 << 20) - 1));

        let satisfied = problem.count_satisfied(&solution);
        assert_eq!(satisfied, 50, "Planted solution should satisfy all clauses");
    }

    #[test]
    fn test_maxsat_evaluation() {
        let problem = MaxSat::random_3sat(10, 20, 42);
        let candidate = Candidate::new(0b11_1111_1111);
        let fitness = problem.evaluate(&candidate);
        assert!(fitness > 0, "Should satisfy some clauses");
    }

    #[test]
    fn test_clause_satisfaction() {
        // Clause: (x0 OR NOT x1)
        let clause = Clause::new(vec![0], vec![1]);

        let c1 = Candidate::new(0b01); // x0=1, x1=0 -> satisfied (x0 is true)
        assert!(clause.is_satisfied(&c1));

        let c2 = Candidate::new(0b00); // x0=0, x1=0 -> satisfied (x1 is false)
        assert!(clause.is_satisfied(&c2));

        let c3 = Candidate::new(0b10); // x0=0, x1=1 -> not satisfied
        assert!(!clause.is_satisfied(&c3));

        let c4 = Candidate::new(0b11); // x0=1, x1=1 -> satisfied (x0 is true)
        assert!(clause.is_satisfied(&c4));
    }
}
