//! Selection strategies for the search engine.
//!
//! Provides both classical (tournament, roulette, truncation) and
//! quantum-enhanced (Grover) selection strategies.

use super::candidate::{Rng, Xorshift64};
use super::substrate::SearchSubstrate;

/// Selection strategy enumeration.
///
/// Supports both classical and quantum-enhanced selection methods.
#[derive(Clone, Debug)]
pub enum SelectionStrategy {
    /// Tournament selection: pick best from random subset
    Tournament {
        /// Number of candidates in each tournament
        tournament_size: usize,
    },

    /// Roulette wheel / fitness proportionate selection
    Roulette,

    /// Truncation selection: keep top fraction
    Truncation {
        /// Fraction of population to keep (0.0-1.0)
        survival_rate: f32,
    },

    /// Rank-based selection: probability proportional to rank, not fitness
    Rank {
        /// Selection pressure (1.0 = uniform, 2.0 = linear bias to best)
        pressure: f32,
    },

    /// Stochastic Universal Sampling (SUS)
    /// Like roulette but with evenly spaced pointers
    SUS,

    /// Quantum Grover-amplified selection (Phase 4+)
    /// Placeholder for quantum-enhanced selection
    QuantumGrover {
        /// Number of Grover iterations
        iterations: usize,
    },
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        // Tournament selection is a good default: O(n) time, adjustable pressure
        Self::Tournament { tournament_size: 7 }
    }
}

impl SelectionStrategy {
    /// Create tournament selection with given size
    pub fn tournament(size: usize) -> Self {
        Self::Tournament {
            tournament_size: size,
        }
    }

    /// Create truncation selection with given survival rate
    pub fn truncation(rate: f32) -> Self {
        assert!(rate > 0.0 && rate <= 1.0, "Survival rate must be (0, 1]");
        Self::Truncation {
            survival_rate: rate,
        }
    }

    /// Create rank selection with given pressure
    pub fn rank(pressure: f32) -> Self {
        assert!(pressure >= 1.0, "Pressure must be >= 1.0");
        Self::Rank { pressure }
    }

    /// Create quantum Grover selection with given iterations
    pub fn quantum_grover(iterations: usize) -> Self {
        Self::QuantumGrover { iterations }
    }

    /// Get a human-readable name for this strategy
    pub fn name(&self) -> &str {
        match self {
            Self::Tournament { .. } => "Tournament",
            Self::Roulette => "Roulette",
            Self::Truncation { .. } => "Truncation",
            Self::Rank { .. } => "Rank",
            Self::SUS => "SUS",
            Self::QuantumGrover { .. } => "QuantumGrover",
        }
    }

    /// Check if this is a quantum strategy
    pub fn is_quantum(&self) -> bool {
        matches!(self, Self::QuantumGrover { .. })
    }
}

/// Trait for selection operators
///
/// Note: Uses concrete Xorshift64 type for dyn-compatibility
pub trait Selector: Send + Sync {
    /// Select `count` parent indices from the substrate
    fn select(&self, substrate: &SearchSubstrate, count: usize, rng: &mut Xorshift64)
    -> Vec<usize>;

    /// Name of this selector
    fn name(&self) -> &str;
}

/// Tournament selection implementation
pub struct TournamentSelector {
    pub tournament_size: usize,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize) -> Self {
        assert!(tournament_size >= 2, "Tournament size must be >= 2");
        Self { tournament_size }
    }
}

impl Selector for TournamentSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            // Pick tournament_size random candidates
            let mut best_idx = (rng.next_u64() as usize) % n;
            let mut best_fitness = substrate.get(best_idx).fitness;

            for _ in 1..self.tournament_size {
                let idx = (rng.next_u64() as usize) % n;
                let fitness = substrate.get(idx).fitness;
                if fitness > best_fitness {
                    best_idx = idx;
                    best_fitness = fitness;
                }
            }

            selected.push(best_idx);
        }

        selected
    }

    fn name(&self) -> &str {
        "Tournament"
    }
}

/// Roulette wheel (fitness proportionate) selection
pub struct RouletteSelector;

impl RouletteSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RouletteSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl Selector for RouletteSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        if n == 0 {
            return vec![];
        }

        // Compute cumulative fitness
        let total_fitness: u64 = substrate
            .candidates()
            .iter()
            .map(|c| c.fitness as u64)
            .sum();

        if total_fitness == 0 {
            // All zero fitness: select randomly
            return (0..count).map(|_| (rng.next_u64() as usize) % n).collect();
        }

        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let target = rng.next_u64() % total_fitness;
            let mut cumulative = 0u64;
            let mut found = false;

            for (i, c) in substrate.candidates().iter().enumerate() {
                cumulative += c.fitness as u64;
                if cumulative > target {
                    selected.push(i);
                    found = true;
                    break;
                }
            }

            // Edge case: if we didn't select anyone (target == total_fitness - 1), select last
            if !found {
                selected.push(n - 1);
            }
        }

        selected
    }

    fn name(&self) -> &str {
        "Roulette"
    }
}

/// Truncation selection
pub struct TruncationSelector {
    pub survival_rate: f32,
}

impl TruncationSelector {
    pub fn new(survival_rate: f32) -> Self {
        assert!(
            survival_rate > 0.0 && survival_rate <= 1.0,
            "Survival rate must be (0, 1]"
        );
        Self { survival_rate }
    }
}

impl Selector for TruncationSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        let survivors = ((n as f32) * self.survival_rate) as usize;
        let survivors = survivors.max(1);

        // Get indices sorted by fitness (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| substrate.get(b).fitness.cmp(&substrate.get(a).fitness));

        // Select randomly from the survivors
        (0..count)
            .map(|_| indices[(rng.next_u64() as usize) % survivors])
            .collect()
    }

    fn name(&self) -> &str {
        "Truncation"
    }
}

/// Rank-based selection
pub struct RankSelector {
    pub pressure: f32,
}

impl RankSelector {
    pub fn new(pressure: f32) -> Self {
        assert!(pressure >= 1.0, "Pressure must be >= 1.0");
        Self { pressure }
    }
}

impl Selector for RankSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        if n == 0 {
            return vec![];
        }

        // Get indices sorted by fitness (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| substrate.get(b).fitness.cmp(&substrate.get(a).fitness));

        // Compute rank-based probabilities
        // p(rank) = (2 - s + 2(s-1)(n-rank)/(n-1)) / n
        // where s = pressure, rank is 0-indexed from best
        let s = self.pressure;
        let n_f = n as f32;

        let mut cumulative = Vec::with_capacity(n);
        let mut total = 0.0;

        for rank in 0..n {
            let prob = (2.0 - s + 2.0 * (s - 1.0) * (n_f - 1.0 - rank as f32) / (n_f - 1.0)) / n_f;
            total += prob;
            cumulative.push(total);
        }

        // Select based on rank probabilities
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let target = rng.next_f32() * total;
            let rank = cumulative
                .iter()
                .position(|&c| c >= target)
                .unwrap_or(n - 1);
            selected.push(indices[rank]);
        }

        selected
    }

    fn name(&self) -> &str {
        "Rank"
    }
}

/// Stochastic Universal Sampling (SUS)
pub struct SUSSelector;

impl SUSSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SUSSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl Selector for SUSSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let n = substrate.len();
        if n == 0 || count == 0 {
            return vec![];
        }

        // Compute cumulative fitness
        let total_fitness: f64 = substrate
            .candidates()
            .iter()
            .map(|c| c.fitness as f64)
            .sum();

        if total_fitness == 0.0 {
            return (0..count).map(|_| (rng.next_u64() as usize) % n).collect();
        }

        // Pointer spacing
        let spacing = total_fitness / count as f64;
        let start = rng.next_f64() * spacing;

        let mut selected = Vec::with_capacity(count);
        let mut cumulative = 0.0;
        let mut i = 0;

        for j in 0..count {
            let pointer = start + j as f64 * spacing;

            while i < n && cumulative + substrate.get(i).fitness as f64 <= pointer {
                cumulative += substrate.get(i).fitness as f64;
                i += 1;
            }

            selected.push(i.min(n - 1));
        }

        selected
    }

    fn name(&self) -> &str {
        "SUS"
    }
}

/// Quantum Grover selector wrapper (implements Selector trait)
pub struct QuantumGroverSelector {
    iterations: usize,
    inner: super::quantum::GroverSelector,
}

impl QuantumGroverSelector {
    pub fn new(iterations: usize) -> Self {
        use super::quantum::{GroverSelector, QuantumConfig};
        Self {
            iterations,
            inner: GroverSelector::new(
                QuantumConfig::default()
                    .with_iterations(iterations)
                    .with_auto_iterations(),
            ),
        }
    }

    /// Get statistics from last selection
    pub fn last_stats(&self) -> Option<&super::quantum::GroverStats> {
        self.inner.last_stats()
    }
}

impl Selector for QuantumGroverSelector {
    fn select(
        &self,
        substrate: &SearchSubstrate,
        count: usize,
        rng: &mut Xorshift64,
    ) -> Vec<usize> {
        // Create a mutable copy for selection (GroverSelector needs &mut self)
        use super::quantum::{GroverSelector, QuantumConfig};
        let mut selector = GroverSelector::new(
            QuantumConfig::default()
                .with_iterations(self.iterations)
                .with_auto_iterations(),
        );
        selector.select(substrate, count, rng)
    }

    fn name(&self) -> &str {
        "QuantumGrover"
    }
}

/// Create a selector from a strategy
pub fn create_selector(strategy: &SelectionStrategy) -> Box<dyn Selector> {
    match strategy {
        SelectionStrategy::Tournament { tournament_size } => {
            Box::new(TournamentSelector::new(*tournament_size))
        }
        SelectionStrategy::Roulette => Box::new(RouletteSelector::new()),
        SelectionStrategy::Truncation { survival_rate } => {
            Box::new(TruncationSelector::new(*survival_rate))
        }
        SelectionStrategy::Rank { pressure } => Box::new(RankSelector::new(*pressure)),
        SelectionStrategy::SUS => Box::new(SUSSelector::new()),
        SelectionStrategy::QuantumGrover { iterations } => {
            Box::new(QuantumGroverSelector::new(*iterations))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::OneMax;

    fn create_test_substrate() -> SearchSubstrate {
        let mut substrate = SearchSubstrate::new(1000, OneMax::standard());
        substrate.initialize_random(42);
        substrate.evaluate_all();
        substrate
    }

    #[test]
    fn test_tournament_selection() {
        let substrate = create_test_substrate();
        let selector = TournamentSelector::new(7);
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 100, &mut rng);
        assert_eq!(selected.len(), 100);

        // All indices should be valid
        for &idx in &selected {
            assert!(idx < substrate.len());
        }

        // With tournament selection, higher fitness should be more common
        let mean_fitness: f32 = selected
            .iter()
            .map(|&i| substrate.get(i).fitness as f32)
            .sum::<f32>()
            / 100.0;
        let pop_mean = substrate.statistics().mean_fitness;

        // Selected mean should be higher than population mean
        assert!(
            mean_fitness > pop_mean,
            "Selected mean {} should be > pop mean {}",
            mean_fitness,
            pop_mean
        );
    }

    #[test]
    fn test_roulette_selection() {
        let substrate = create_test_substrate();
        let selector = RouletteSelector::new();
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 100, &mut rng);
        assert_eq!(selected.len(), 100);

        // Should bias towards higher fitness
        let mean_fitness: f32 = selected
            .iter()
            .map(|&i| substrate.get(i).fitness as f32)
            .sum::<f32>()
            / 100.0;
        let pop_mean = substrate.statistics().mean_fitness;

        assert!(mean_fitness >= pop_mean * 0.9); // Allow some variance
    }

    #[test]
    fn test_truncation_selection() {
        let substrate = create_test_substrate();
        let selector = TruncationSelector::new(0.1); // Top 10%
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 100, &mut rng);
        assert_eq!(selected.len(), 100);

        // All selected should be from top 10%
        let mut sorted_fitness: Vec<u8> =
            substrate.candidates().iter().map(|c| c.fitness).collect();
        sorted_fitness.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = sorted_fitness[99]; // Top 10% threshold

        for &idx in &selected {
            assert!(
                substrate.get(idx).fitness >= threshold,
                "Selected fitness {} below threshold {}",
                substrate.get(idx).fitness,
                threshold
            );
        }
    }

    #[test]
    fn test_rank_selection() {
        let substrate = create_test_substrate();
        let selector = RankSelector::new(2.0); // Linear ranking
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 100, &mut rng);
        assert_eq!(selected.len(), 100);

        // Should have selection pressure
        let mean_fitness: f32 = selected
            .iter()
            .map(|&i| substrate.get(i).fitness as f32)
            .sum::<f32>()
            / 100.0;
        let pop_mean = substrate.statistics().mean_fitness;

        assert!(mean_fitness > pop_mean);
    }

    #[test]
    fn test_sus_selection() {
        let substrate = create_test_substrate();
        let selector = SUSSelector::new();
        let mut rng = Xorshift64::new(123);

        let selected = selector.select(&substrate, 100, &mut rng);
        assert_eq!(selected.len(), 100);

        // SUS should have lower variance than roulette
        // Just verify it works for now
        for &idx in &selected {
            assert!(idx < substrate.len());
        }
    }

    #[test]
    fn test_create_selector() {
        let strategy = SelectionStrategy::Tournament { tournament_size: 5 };
        let selector = create_selector(&strategy);
        assert_eq!(selector.name(), "Tournament");

        let strategy = SelectionStrategy::Roulette;
        let selector = create_selector(&strategy);
        assert_eq!(selector.name(), "Roulette");
    }

    #[test]
    fn test_selection_strategy_names() {
        assert_eq!(
            SelectionStrategy::Tournament { tournament_size: 7 }.name(),
            "Tournament"
        );
        assert_eq!(SelectionStrategy::Roulette.name(), "Roulette");
        assert_eq!(
            SelectionStrategy::Truncation { survival_rate: 0.5 }.name(),
            "Truncation"
        );
        assert_eq!(SelectionStrategy::Rank { pressure: 2.0 }.name(), "Rank");
        assert_eq!(SelectionStrategy::SUS.name(), "SUS");
        assert_eq!(
            SelectionStrategy::QuantumGrover { iterations: 1 }.name(),
            "QuantumGrover"
        );
    }
}
