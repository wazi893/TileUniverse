//! Observer substrate implementation.
//!
//! Wraps the Observer Grid (Sprint 60.0) for observer-controlled
//! hybrid quantum-classical computation within the orchestration framework.

use std::time::{Duration, Instant};

use crate::tile8::{ObserverGrid, ObserverGridConfig};

use super::job::{JobSpec, ObserverSpec, ObserverTrigger};
use super::result::{JobStage, StageMetrics, StageOutputs, StageResult};
use super::substrate::{
    Substrate, SubstrateCapabilities, SubstrateError, SubstrateId, SubstrateType,
};

/// Observer Grid substrate for observer-controlled computation.
///
/// This substrate manages the Observer Grid from Sprint 60.0, enabling:
/// - Observer-controlled quantum measurement
/// - Wavefront collapse dynamics
/// - Anomaly detection
/// - Protected sanctuary regions
pub struct ObserverSubstrate {
    /// Unique identifier
    id: SubstrateId,

    /// Human-readable name
    name: String,

    /// Current capacity (0.0 = busy, 1.0 = idle)
    capacity: f64,

    /// Observer grid instance
    grid: Option<ObserverGrid>,

    /// Default configuration
    default_config: ObserverGridConfig,

    /// Maximum observers supported
    max_observers: usize,
}

impl ObserverSubstrate {
    /// Create a new observer substrate.
    pub fn new(id: u64, name: &str) -> Self {
        Self {
            id: SubstrateId(id),
            name: name.to_string(),
            capacity: 1.0,
            grid: None,
            default_config: ObserverGridConfig::default(),
            max_observers: 1_000_000, // Default 1M observers
        }
    }

    /// Set maximum observers.
    pub fn with_max_observers(mut self, max: usize) -> Self {
        self.max_observers = max;
        self
    }

    /// Set default configuration.
    pub fn with_config(mut self, config: ObserverGridConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Initialize the observer grid.
    fn init_grid(&mut self, config: &ObserverGridConfig) -> Result<(), SubstrateError> {
        if config.num_observers > self.max_observers {
            return Err(SubstrateError::InvalidSpec(format!(
                "Observer count {} exceeds maximum {}",
                config.num_observers, self.max_observers
            )));
        }

        self.grid = Some(ObserverGrid::new(config.clone()));
        Ok(())
    }

    /// Execute observer-controlled computation.
    fn execute_observer(&mut self, spec: &ObserverSpec) -> Result<StageOutputs, SubstrateError> {
        // Initialize grid with configuration
        self.init_grid(&spec.observer_config)?;

        // Run ticks until trigger condition
        let mut ticks_executed = 0u64;

        loop {
            // Get mutable access, run tick, then release borrow
            {
                let grid = self
                    .grid
                    .as_mut()
                    .ok_or(SubstrateError::Internal("Grid not initialized".to_string()))?;
                grid.tick();
            }
            ticks_executed += 1;

            // Check termination conditions with immutable access
            let should_terminate = {
                let grid = self
                    .grid
                    .as_ref()
                    .ok_or(SubstrateError::Internal("Grid not initialized".to_string()))?;
                Self::check_trigger_static(&spec.trigger, grid, ticks_executed)
            };

            if should_terminate {
                break;
            }

            if ticks_executed >= spec.max_ticks {
                break;
            }
        }

        // Collect results
        let grid = self
            .grid
            .as_ref()
            .ok_or(SubstrateError::Internal("Grid not initialized".to_string()))?;

        let stats = &grid.stats;
        let collapse_ratio = if stats.total > 0 {
            stats.collapsed_count as f64 / stats.total as f64
        } else {
            0.0
        };

        // Get collapsed measurements from observers
        let measurements: Vec<u8> = grid
            .observers
            .iter()
            .filter_map(|obs| obs.measurement())
            .collect();

        Ok(StageOutputs::Observer {
            collapse_ratio,
            anomaly_score: stats.anomaly_score,
            ticks: ticks_executed,
            measurements,
        })
    }

    /// Calculate collapse ratio from stats.
    fn collapse_ratio(stats: &crate::tile8::ObservationStats) -> f64 {
        if stats.total > 0 {
            stats.collapsed_count as f64 / stats.total as f64
        } else {
            0.0
        }
    }

    /// Check if trigger condition is met (static version to avoid borrow issues).
    fn check_trigger_static(trigger: &ObserverTrigger, grid: &ObserverGrid, ticks: u64) -> bool {
        match trigger {
            ObserverTrigger::AfterTicks(n) => ticks >= *n,
            ObserverTrigger::OnAnomaly { threshold } => grid.stats.anomaly_score >= *threshold,
            ObserverTrigger::OnCollapseRatio { threshold } => {
                Self::collapse_ratio(&grid.stats) >= *threshold
            }
            ObserverTrigger::OnFullCollapse => Self::collapse_ratio(&grid.stats) >= 0.999,
            ObserverTrigger::Manual => false, // Never auto-terminate
            ObserverTrigger::Any(conditions) => conditions
                .iter()
                .any(|c| Self::check_trigger_static(c, grid, ticks)),
            ObserverTrigger::All(conditions) => conditions
                .iter()
                .all(|c| Self::check_trigger_static(c, grid, ticks)),
        }
    }
}

impl Substrate for ObserverSubstrate {
    fn id(&self) -> SubstrateId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn substrate_type(&self) -> SubstrateType {
        SubstrateType::Observer
    }

    fn capacity(&self) -> f64 {
        self.capacity
    }

    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            max_qubits: None,
            max_depth: None,
            gpu_available: false,
            max_parallel_jobs: 1,
            supported_gates: vec![],
            supports_variational: false,
            supports_observer: true,
        }
    }

    fn can_handle(&self, spec: &JobSpec) -> bool {
        match spec {
            JobSpec::Observer(obs) => obs.observer_config.num_observers <= self.max_observers,
            _ => false,
        }
    }

    fn estimate_time(&self, spec: &JobSpec) -> Duration {
        match spec {
            JobSpec::Observer(obs) => {
                // Rough estimate: 1ms per 10000 observers per tick
                let observers = obs.observer_config.num_observers;
                let ticks = match &obs.trigger {
                    ObserverTrigger::AfterTicks(n) => *n,
                    _ => obs.max_ticks.min(1000), // Assume worst case for unknown triggers
                };
                Duration::from_micros((observers as u64 * ticks) / 10000 + 1)
            }
            _ => Duration::ZERO,
        }
    }

    fn execute(&mut self, stage: &JobStage) -> Result<StageResult, SubstrateError> {
        self.capacity = 0.0; // Mark as busy

        let start = Instant::now();

        let result = match &stage.spec {
            JobSpec::Observer(spec) => self.execute_observer(spec),
            _ => Err(SubstrateError::UnsupportedJob(
                "ObserverSubstrate only handles Observer jobs".to_string(),
            )),
        };

        let duration = start.elapsed();
        self.capacity = 1.0; // Mark as idle

        match result {
            Ok(outputs) => {
                let mut metrics = StageMetrics::default();
                if let StageOutputs::Observer { ticks, .. } = &outputs {
                    metrics.custom.insert("ticks".to_string(), *ticks as f64);
                }
                Ok(StageResult::success(stage.index, duration, outputs).with_metrics(metrics))
            }
            Err(e) => Ok(StageResult::failure(stage.index, duration, e.to_string())),
        }
    }

    fn reset(&mut self) {
        self.grid = None;
        self.capacity = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_substrate_creation() {
        let substrate = ObserverSubstrate::new(1, "TestObserver");

        assert_eq!(substrate.name(), "TestObserver");
        assert_eq!(substrate.substrate_type(), SubstrateType::Observer);
        assert!(substrate.capacity() > 0.99);
    }

    #[test]
    fn test_can_handle() {
        let substrate = ObserverSubstrate::new(1, "TestObserver").with_max_observers(1000);

        // Should handle small observer grids
        let small_config = ObserverGridConfig {
            num_observers: 100,
            ..Default::default()
        };
        let small_spec = ObserverSpec::new(small_config);
        assert!(substrate.can_handle(&JobSpec::Observer(small_spec)));

        // Should not handle large observer grids exceeding limit
        let large_config = ObserverGridConfig {
            num_observers: 10000,
            ..Default::default()
        };
        let large_spec = ObserverSpec::new(large_config);
        assert!(!substrate.can_handle(&JobSpec::Observer(large_spec)));
    }

    #[test]
    fn test_execute_basic() {
        let mut substrate = ObserverSubstrate::new(1, "TestObserver");

        let config = ObserverGridConfig {
            num_observers: 64,
            ..Default::default()
        };

        let spec = ObserverSpec::new(config)
            .with_trigger(ObserverTrigger::AfterTicks(10))
            .with_max_ticks(100);

        let stage = JobStage::new(0, JobSpec::Observer(spec));
        let result = substrate.execute(&stage).unwrap();

        assert!(result.success);

        if let StageOutputs::Observer {
            collapse_ratio,
            ticks,
            ..
        } = &result.outputs
        {
            assert!(*collapse_ratio >= 0.0 && *collapse_ratio <= 1.0);
            assert!(*ticks >= 10);
        } else {
            panic!("Expected observer outputs");
        }
    }

    #[test]
    fn test_trigger_conditions() {
        // Test AfterTicks
        let trigger = ObserverTrigger::AfterTicks(100);
        let config = ObserverGridConfig::default();
        let grid = ObserverGrid::new(config);
        assert!(!ObserverSubstrate::check_trigger_static(
            &trigger, &grid, 50
        ));
        assert!(ObserverSubstrate::check_trigger_static(
            &trigger, &grid, 100
        ));
        assert!(ObserverSubstrate::check_trigger_static(
            &trigger, &grid, 150
        ));
    }

    #[test]
    fn test_compound_triggers() {
        let config = ObserverGridConfig::default();
        let grid = ObserverGrid::new(config);

        // Test Any
        let any_trigger = ObserverTrigger::Any(vec![
            ObserverTrigger::AfterTicks(10),
            ObserverTrigger::AfterTicks(20),
        ]);
        assert!(ObserverSubstrate::check_trigger_static(
            &any_trigger,
            &grid,
            15
        ));
        assert!(!ObserverSubstrate::check_trigger_static(
            &any_trigger,
            &grid,
            5
        ));

        // Test All
        let all_trigger = ObserverTrigger::All(vec![
            ObserverTrigger::AfterTicks(10),
            ObserverTrigger::AfterTicks(20),
        ]);
        assert!(ObserverSubstrate::check_trigger_static(
            &all_trigger,
            &grid,
            25
        ));
        assert!(!ObserverSubstrate::check_trigger_static(
            &all_trigger,
            &grid,
            15
        ));
    }

    #[test]
    fn test_estimate_time() {
        let substrate = ObserverSubstrate::new(1, "TestObserver");

        let config = ObserverGridConfig {
            num_observers: 10000,
            ..Default::default()
        };

        let spec = ObserverSpec::new(config).with_trigger(ObserverTrigger::AfterTicks(100));

        let estimate = substrate.estimate_time(&JobSpec::Observer(spec));
        assert!(estimate > Duration::ZERO);
    }
}
