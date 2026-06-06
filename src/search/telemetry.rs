//! Per-generation telemetry and instrumentation.
//!
//! Provides comprehensive metrics collection for analyzing
//! search algorithm behavior and performance.

use std::time::Duration;

/// Telemetry data for a single generation
#[derive(Clone, Debug, Default)]
pub struct GenerationTelemetry {
    /// Generation number
    pub generation: u64,
    /// Best fitness in population
    pub best_fitness: u8,
    /// Mean fitness
    pub mean_fitness: f32,
    /// Standard deviation of fitness
    pub std_fitness: f32,
    /// Diversity measure (mean Hamming distance)
    pub diversity: f32,
    /// Time spent on fitness evaluation
    pub eval_time: Duration,
    /// Total time for generation
    pub generation_time: Duration,
    /// Selection pressure (best/mean ratio)
    pub selection_pressure: f32,
    /// Number of improvements this generation
    pub improvements: u32,
}

impl GenerationTelemetry {
    /// Check if this generation improved over the previous
    pub fn improved_over(&self, previous: &GenerationTelemetry) -> bool {
        self.best_fitness > previous.best_fitness
    }

    /// Evaluations per second for this generation
    pub fn evals_per_second(&self, population_size: usize) -> f64 {
        if self.eval_time.as_secs_f64() > 0.0 {
            population_size as f64 / self.eval_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Percentage of evaluation time vs total generation time
    pub fn eval_fraction(&self) -> f32 {
        if self.generation_time.as_secs_f32() > 0.0 {
            self.eval_time.as_secs_f32() / self.generation_time.as_secs_f32()
        } else {
            0.0
        }
    }
}

/// Collects telemetry across multiple generations
#[derive(Clone, Debug, Default)]
pub struct TelemetryCollector {
    /// All recorded telemetry
    records: Vec<GenerationTelemetry>,
    /// Running best fitness
    running_best: u8,
    /// Generation where running best was achieved
    best_generation: u64,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Record telemetry for a generation
    pub fn record(&mut self, telemetry: GenerationTelemetry) {
        if telemetry.best_fitness > self.running_best {
            self.running_best = telemetry.best_fitness;
            self.best_generation = telemetry.generation;
        }
        self.records.push(telemetry);
    }

    /// Get all recorded telemetry
    pub fn all(&self) -> &[GenerationTelemetry] {
        &self.records
    }

    /// Number of generations recorded
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get the running best fitness
    pub fn best_fitness(&self) -> u8 {
        self.running_best
    }

    /// Get the generation where best was achieved
    pub fn best_generation(&self) -> u64 {
        self.best_generation
    }

    /// Get the last recorded telemetry
    pub fn last(&self) -> Option<&GenerationTelemetry> {
        self.records.last()
    }

    /// Get telemetry for a specific generation
    pub fn get(&self, generation: u64) -> Option<&GenerationTelemetry> {
        self.records.get(generation as usize)
    }

    /// Calculate summary statistics
    pub fn summary(&self) -> TelemetrySummary {
        if self.records.is_empty() {
            return TelemetrySummary::default();
        }

        let total_time: Duration = self.records.iter().map(|t| t.generation_time).sum();
        let total_eval_time: Duration = self.records.iter().map(|t| t.eval_time).sum();

        let mean_diversity: f32 =
            self.records.iter().map(|t| t.diversity).sum::<f32>() / self.records.len() as f32;

        let improvements: u32 = self.records.iter().map(|t| t.improvements).sum();

        let fitness_improvement =
            if let (Some(first), Some(last)) = (self.records.first(), self.records.last()) {
                last.best_fitness as i16 - first.best_fitness as i16
            } else {
                0
            };

        // Count stagnant generations (no improvement)
        let stagnant_generations = self
            .records
            .windows(2)
            .filter(|w| w[1].best_fitness <= w[0].best_fitness)
            .count();

        TelemetrySummary {
            total_generations: self.records.len() as u64,
            total_time,
            total_eval_time,
            mean_diversity,
            total_improvements: improvements,
            fitness_improvement,
            stagnant_generations: stagnant_generations as u64,
            best_fitness: self.running_best,
            best_generation: self.best_generation,
        }
    }

    /// Get fitness history (for plotting)
    pub fn fitness_history(&self) -> Vec<(u64, u8, f32)> {
        self.records
            .iter()
            .map(|t| (t.generation, t.best_fitness, t.mean_fitness))
            .collect()
    }

    /// Get diversity history (for plotting)
    pub fn diversity_history(&self) -> Vec<(u64, f32)> {
        self.records
            .iter()
            .map(|t| (t.generation, t.diversity))
            .collect()
    }

    /// Clear all recorded telemetry
    pub fn clear(&mut self) {
        self.records.clear();
        self.running_best = 0;
        self.best_generation = 0;
    }
}

/// Summary statistics across all generations
#[derive(Clone, Debug, Default)]
pub struct TelemetrySummary {
    /// Total generations recorded
    pub total_generations: u64,
    /// Total runtime
    pub total_time: Duration,
    /// Total evaluation time
    pub total_eval_time: Duration,
    /// Mean diversity across generations
    pub mean_diversity: f32,
    /// Total number of improvements
    pub total_improvements: u32,
    /// Net fitness improvement (final - initial)
    pub fitness_improvement: i16,
    /// Number of generations without improvement
    pub stagnant_generations: u64,
    /// Best fitness achieved
    pub best_fitness: u8,
    /// Generation where best was achieved
    pub best_generation: u64,
}

impl TelemetrySummary {
    /// Average time per generation
    pub fn avg_generation_time(&self) -> Duration {
        if self.total_generations > 0 {
            self.total_time / self.total_generations as u32
        } else {
            Duration::ZERO
        }
    }

    /// Fraction of time spent on evaluation
    pub fn eval_fraction(&self) -> f32 {
        if self.total_time.as_secs_f32() > 0.0 {
            self.total_eval_time.as_secs_f32() / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }

    /// Improvement rate (improvements per generation)
    pub fn improvement_rate(&self) -> f32 {
        if self.total_generations > 0 {
            self.total_improvements as f32 / self.total_generations as f32
        } else {
            0.0
        }
    }

    /// Stagnation rate (stagnant generations / total)
    pub fn stagnation_rate(&self) -> f32 {
        if self.total_generations > 0 {
            self.stagnant_generations as f32 / self.total_generations as f32
        } else {
            0.0
        }
    }
}

/// Comparison between two runs (e.g., quantum vs classical)
#[derive(Clone, Debug)]
pub struct TelemetryComparison {
    /// Name of first run
    pub name_a: String,
    /// Name of second run
    pub name_b: String,
    /// Summary of first run
    pub summary_a: TelemetrySummary,
    /// Summary of second run
    pub summary_b: TelemetrySummary,
}

impl TelemetryComparison {
    /// Create a comparison between two runs
    pub fn new(
        name_a: impl Into<String>,
        collector_a: &TelemetryCollector,
        name_b: impl Into<String>,
        collector_b: &TelemetryCollector,
    ) -> Self {
        Self {
            name_a: name_a.into(),
            name_b: name_b.into(),
            summary_a: collector_a.summary(),
            summary_b: collector_b.summary(),
        }
    }

    /// Speedup of B over A (generations to reach same fitness)
    pub fn generation_speedup(&self) -> Option<f32> {
        // Compare generations to reach same best fitness
        if self.summary_b.best_generation > 0 && self.summary_a.best_generation > 0 {
            Some(self.summary_a.best_generation as f32 / self.summary_b.best_generation as f32)
        } else {
            None
        }
    }

    /// Runtime speedup of B over A
    pub fn runtime_speedup(&self) -> f32 {
        if self.summary_b.total_time.as_secs_f32() > 0.0 {
            self.summary_a.total_time.as_secs_f32() / self.summary_b.total_time.as_secs_f32()
        } else {
            1.0
        }
    }

    /// Fitness advantage of B over A
    pub fn fitness_advantage(&self) -> i16 {
        self.summary_b.best_fitness as i16 - self.summary_a.best_fitness as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_telemetry() {
        let t1 = GenerationTelemetry {
            generation: 0,
            best_fitness: 100,
            mean_fitness: 50.0,
            ..Default::default()
        };

        let t2 = GenerationTelemetry {
            generation: 1,
            best_fitness: 110,
            mean_fitness: 55.0,
            ..Default::default()
        };

        assert!(t2.improved_over(&t1));
        assert!(!t1.improved_over(&t2));
    }

    #[test]
    fn test_telemetry_collector() {
        let mut collector = TelemetryCollector::new();

        for i in 0..10 {
            collector.record(GenerationTelemetry {
                generation: i,
                best_fitness: (50 + i * 5) as u8,
                mean_fitness: (30 + i * 3) as f32,
                diversity: 20.0 - i as f32,
                generation_time: Duration::from_millis(100),
                eval_time: Duration::from_millis(80),
                improvements: 10,
                ..Default::default()
            });
        }

        assert_eq!(collector.len(), 10);
        assert_eq!(collector.best_fitness(), 95);
        assert_eq!(collector.best_generation(), 9);
    }

    #[test]
    fn test_telemetry_summary() {
        let mut collector = TelemetryCollector::new();

        // Simulate convergence: fitness improves then plateaus
        for i in 0..20 {
            let fitness = if i < 10 { 50 + i * 10 } else { 140 };
            collector.record(GenerationTelemetry {
                generation: i as u64,
                best_fitness: fitness as u8,
                mean_fitness: fitness as f32 * 0.7,
                diversity: 30.0 - i as f32,
                generation_time: Duration::from_millis(100),
                eval_time: Duration::from_millis(80),
                improvements: if i < 10 { 5 } else { 0 },
                ..Default::default()
            });
        }

        let summary = collector.summary();
        assert_eq!(summary.total_generations, 20);
        assert_eq!(summary.best_fitness, 140);
        assert_eq!(summary.fitness_improvement, 90); // 140 - 50
        assert!(summary.stagnant_generations >= 9); // Last 10 generations stagnant
    }

    #[test]
    fn test_fitness_history() {
        let mut collector = TelemetryCollector::new();

        for i in 0..5 {
            collector.record(GenerationTelemetry {
                generation: i,
                best_fitness: (i * 10) as u8,
                mean_fitness: (i * 5) as f32,
                ..Default::default()
            });
        }

        let history = collector.fitness_history();
        assert_eq!(history.len(), 5);
        assert_eq!(history[0], (0, 0, 0.0));
        assert_eq!(history[4], (4, 40, 20.0));
    }

    #[test]
    fn test_comparison() {
        let mut collector_a = TelemetryCollector::new();
        let mut collector_b = TelemetryCollector::new();

        // A takes 100 generations to reach fitness 200
        for i in 0..100 {
            collector_a.record(GenerationTelemetry {
                generation: i,
                best_fitness: (i * 2) as u8,
                generation_time: Duration::from_millis(100),
                ..Default::default()
            });
        }

        // B takes 50 generations to reach fitness 200
        for i in 0..50 {
            collector_b.record(GenerationTelemetry {
                generation: i,
                best_fitness: (i * 4) as u8,
                generation_time: Duration::from_millis(100),
                ..Default::default()
            });
        }

        let comparison =
            TelemetryComparison::new("Classical", &collector_a, "Quantum", &collector_b);

        // B should be ~2x faster in generations
        let speedup = comparison.generation_speedup().unwrap();
        assert!(speedup > 1.5, "Expected speedup > 1.5, got {}", speedup);
    }

    #[test]
    fn test_empty_collector() {
        let collector = TelemetryCollector::new();
        assert!(collector.is_empty());
        assert_eq!(collector.best_fitness(), 0);

        let summary = collector.summary();
        assert_eq!(summary.total_generations, 0);
        assert_eq!(summary.best_fitness, 0);
    }

    #[test]
    fn test_eval_fraction() {
        let t = GenerationTelemetry {
            generation_time: Duration::from_millis(100),
            eval_time: Duration::from_millis(80),
            ..Default::default()
        };

        let frac = t.eval_fraction();
        assert!((frac - 0.8).abs() < 0.01);
    }
}
