//! Sprint 62.0: Checkpointing for Variational Algorithms
//!
//! Provides save/restore functionality for variational jobs (VQE, QAOA),
//! allowing interrupted optimizations to be resumed from the last checkpoint.
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::{VariationalCheckpoint, CheckpointManager};
//!
//! // Create checkpoint during optimization
//! let checkpoint = VariationalCheckpoint::new(
//!     params.clone(),
//!     current_cost,
//!     iteration,
//!     history.clone(),
//! );
//!
//! // Save to file
//! checkpoint.save("vqe_checkpoint.json")?;
//!
//! // Later, restore and resume
//! let checkpoint = VariationalCheckpoint::load("vqe_checkpoint.json")?;
//! let (params, iteration) = (checkpoint.params, checkpoint.iteration);
//! ```

use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Checkpoint data for variational optimization.
///
/// Contains all state needed to resume an interrupted optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariationalCheckpoint {
    /// Current parameter values
    pub params: Vec<f64>,

    /// Current cost/energy value
    pub cost: f64,

    /// Current iteration number
    pub iteration: usize,

    /// Cost history for convergence analysis
    pub history: Vec<f64>,

    /// Timestamp when checkpoint was created (Unix epoch seconds)
    pub timestamp: u64,

    /// Optional metadata (algorithm name, ansatz type, etc.)
    #[serde(default)]
    pub metadata: CheckpointMetadata,
}

/// Metadata for checkpoint identification and tracking.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Algorithm name (e.g., "VQE", "QAOA")
    pub algorithm: Option<String>,

    /// Ansatz type (e.g., "hardware_efficient", "uccsd")
    pub ansatz_type: Option<String>,

    /// Number of qubits
    pub n_qubits: Option<usize>,

    /// Optimizer name
    pub optimizer: Option<String>,

    /// Problem description
    pub problem: Option<String>,

    /// Custom key-value pairs
    #[serde(default)]
    pub extra: std::collections::HashMap<String, String>,
}

impl VariationalCheckpoint {
    /// Create a new checkpoint.
    pub fn new(params: Vec<f64>, cost: f64, iteration: usize, history: Vec<f64>) -> Self {
        Self {
            params,
            cost,
            iteration,
            history,
            timestamp: current_timestamp(),
            metadata: CheckpointMetadata::default(),
        }
    }

    /// Add metadata to the checkpoint.
    pub fn with_metadata(mut self, metadata: CheckpointMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set algorithm name.
    pub fn with_algorithm(mut self, algorithm: &str) -> Self {
        self.metadata.algorithm = Some(algorithm.to_string());
        self
    }

    /// Set ansatz type.
    pub fn with_ansatz(mut self, ansatz: &str) -> Self {
        self.metadata.ansatz_type = Some(ansatz.to_string());
        self
    }

    /// Set number of qubits.
    pub fn with_qubits(mut self, n_qubits: usize) -> Self {
        self.metadata.n_qubits = Some(n_qubits);
        self
    }

    /// Save checkpoint to a JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load checkpoint from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the age of the checkpoint in seconds.
    pub fn age_seconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.timestamp)
    }

    /// Check if the checkpoint is stale (older than max_age_seconds).
    pub fn is_stale(&self, max_age_seconds: u64) -> bool {
        self.age_seconds() > max_age_seconds
    }

    /// Get best cost from history.
    pub fn best_cost(&self) -> f64 {
        self.history.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Check if optimization has converged based on recent history.
    pub fn has_converged(&self, tolerance: f64, window: usize) -> bool {
        if self.history.len() < window {
            return false;
        }

        let recent: Vec<f64> = self.history.iter().rev().take(window).cloned().collect();
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);

        (max - min).abs() < tolerance
    }
}

/// Manager for automatic checkpointing during optimization.
#[derive(Debug)]
pub struct CheckpointManager {
    /// Base path for checkpoint files
    base_path: String,

    /// Checkpoint interval (iterations between saves)
    interval: usize,

    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,

    /// Counter for checkpoint numbering
    counter: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            interval: 10, // Save every 10 iterations by default
            max_checkpoints: 5,
            counter: 0,
        }
    }

    /// Set checkpoint interval.
    pub fn with_interval(mut self, interval: usize) -> Self {
        self.interval = interval;
        self
    }

    /// Set maximum checkpoints to keep.
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Check if a checkpoint should be saved at this iteration.
    pub fn should_save(&self, iteration: usize) -> bool {
        self.interval > 0 && iteration % self.interval == 0
    }

    /// Save a checkpoint and manage old checkpoints.
    pub fn save(&mut self, checkpoint: &VariationalCheckpoint) -> io::Result<String> {
        let path = format!("{}_{}.json", self.base_path, self.counter);
        checkpoint.save(&path)?;

        // Clean up old checkpoints
        if self.counter >= self.max_checkpoints {
            let old_path = format!(
                "{}_{}.json",
                self.base_path,
                self.counter - self.max_checkpoints
            );
            let _ = fs::remove_file(old_path); // Ignore errors
        }

        self.counter += 1;
        Ok(path)
    }

    /// Load the latest checkpoint.
    pub fn load_latest(&self) -> Option<VariationalCheckpoint> {
        if self.counter == 0 {
            return None;
        }

        let path = format!("{}_{}.json", self.base_path, self.counter - 1);
        VariationalCheckpoint::load(&path).ok()
    }

    /// Find and load the most recent valid checkpoint.
    pub fn find_checkpoint(&self) -> Option<VariationalCheckpoint> {
        // Try loading checkpoints from most recent to oldest
        for i in (0..self.counter).rev() {
            let path = format!("{}_{}.json", self.base_path, i);
            if let Ok(checkpoint) = VariationalCheckpoint::load(&path) {
                return Some(checkpoint);
            }
        }

        // Also try base path without counter
        let direct_path = format!("{}.json", self.base_path);
        VariationalCheckpoint::load(&direct_path).ok()
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new("checkpoint")
    }
}

/// Get current Unix timestamp.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint =
            VariationalCheckpoint::new(vec![0.1, 0.2, 0.3], -1.5, 10, vec![-1.0, -1.2, -1.4, -1.5]);

        assert_eq!(checkpoint.params.len(), 3);
        assert_eq!(checkpoint.cost, -1.5);
        assert_eq!(checkpoint.iteration, 10);
        assert_eq!(checkpoint.history.len(), 4);
    }

    #[test]
    fn test_checkpoint_metadata() {
        let checkpoint = VariationalCheckpoint::new(vec![0.1], -1.0, 5, vec![-1.0])
            .with_algorithm("VQE")
            .with_ansatz("hardware_efficient")
            .with_qubits(4);

        assert_eq!(checkpoint.metadata.algorithm, Some("VQE".to_string()));
        assert_eq!(
            checkpoint.metadata.ansatz_type,
            Some("hardware_efficient".to_string())
        );
        assert_eq!(checkpoint.metadata.n_qubits, Some(4));
    }

    #[test]
    fn test_checkpoint_json_roundtrip() {
        let original = VariationalCheckpoint::new(
            vec![0.1, 0.2, 0.3, 0.4],
            -1.234,
            42,
            vec![-0.5, -0.8, -1.0, -1.234],
        )
        .with_algorithm("VQE");

        let json = original.to_json().unwrap();
        let restored = VariationalCheckpoint::from_json(&json).unwrap();

        assert_eq!(original.params, restored.params);
        assert_eq!(original.cost, restored.cost);
        assert_eq!(original.iteration, restored.iteration);
        assert_eq!(original.metadata.algorithm, restored.metadata.algorithm);
    }

    #[test]
    fn test_best_cost() {
        let checkpoint =
            VariationalCheckpoint::new(vec![0.1], -1.5, 5, vec![-1.0, -1.2, -1.5, -1.3, -1.4]);

        assert_eq!(checkpoint.best_cost(), -1.5);
    }

    #[test]
    fn test_convergence_check() {
        // Not converged - large variation
        let checkpoint1 =
            VariationalCheckpoint::new(vec![0.1], -1.0, 10, vec![-0.5, -0.8, -1.0, -0.9, -1.1]);
        assert!(!checkpoint1.has_converged(0.01, 3));

        // Converged - small variation
        let checkpoint2 = VariationalCheckpoint::new(
            vec![0.1],
            -1.0001,
            20,
            vec![-1.0, -1.0001, -0.9999, -1.00005],
        );
        assert!(checkpoint2.has_converged(0.001, 3));
    }

    #[test]
    fn test_checkpoint_file_io() {
        let checkpoint =
            VariationalCheckpoint::new(vec![0.1, 0.2], -2.0, 15, vec![-1.5, -1.8, -2.0]);

        let path = "test_checkpoint_temp.json";

        // Save
        checkpoint.save(path).unwrap();
        assert!(Path::new(path).exists());

        // Load
        let loaded = VariationalCheckpoint::load(path).unwrap();
        assert_eq!(checkpoint.params, loaded.params);
        assert_eq!(checkpoint.cost, loaded.cost);

        // Cleanup
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new("test_mgr")
            .with_interval(5)
            .with_max_checkpoints(3);

        assert!(manager.should_save(0));
        assert!(!manager.should_save(1));
        assert!(manager.should_save(5));
        assert!(manager.should_save(10));
    }

    #[test]
    fn test_checkpoint_age() {
        let checkpoint = VariationalCheckpoint::new(vec![0.1], -1.0, 1, vec![-1.0]);

        // Checkpoint was just created, so age should be small
        assert!(checkpoint.age_seconds() < 2);

        // Not stale with 1 hour limit
        assert!(!checkpoint.is_stale(3600));
    }
}
