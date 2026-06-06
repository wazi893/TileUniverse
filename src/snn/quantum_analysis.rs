//! Quantum-Neural Analysis Tools (Sprint 59.0 EPIC 136)
//!
//! Provides snapshot and recording capabilities for analyzing
//! hybrid quantum-neural network dynamics.
//!
//! # Key Types
//!
//! - [`QuantumNeuralSnapshot`]: Point-in-time network state capture
//! - [`ClusterSnapshot`]: Per-cluster quantum state metrics
//! - [`QuantumNeuralRecorder`]: Time series recording with ring buffer

use super::stabilizer_hybrid::HybridQuantumNetwork;

/// Snapshot of a single cluster's quantum state.
#[derive(Clone, Debug)]
pub struct ClusterSnapshot {
    /// Cluster ID
    pub id: usize,

    /// Whether cluster is currently active
    pub active: bool,

    /// Entanglement entropy (bipartite, in bits)
    pub entropy: f64,

    /// Z expectation values for each qubit
    pub z_expectations: Vec<f64>,

    /// X expectation values for each qubit
    pub x_expectations: Vec<f64>,

    /// Total Z expectation (average)
    pub total_z: f64,

    /// Total probability (should be ~1.0)
    pub total_probability: f64,
}

impl ClusterSnapshot {
    /// Check if cluster state is valid (normalized).
    pub fn is_normalized(&self) -> bool {
        (self.total_probability - 1.0).abs() < 0.01
    }

    /// Get coherence estimate based on X expectations.
    ///
    /// High coherence = more quantum superposition.
    pub fn coherence(&self) -> f64 {
        if self.x_expectations.is_empty() {
            return 0.0;
        }
        self.x_expectations.iter().map(|x| x.abs()).sum::<f64>() / self.x_expectations.len() as f64
    }

    /// Get polarization (bias toward |0⟩ or |1⟩).
    ///
    /// Returns value in [-1, 1]: +1 = all |0⟩, -1 = all |1⟩, 0 = balanced.
    pub fn polarization(&self) -> f64 {
        self.total_z
    }
}

/// Complete snapshot of the quantum-neural network state.
#[derive(Clone, Debug)]
pub struct QuantumNeuralSnapshot {
    /// Tick number when snapshot was taken
    pub tick: u64,

    /// Spike pattern from backbone (neuron indices that fired)
    pub backbone_spikes: Vec<usize>,

    /// Number of active clusters
    pub active_clusters: usize,

    /// Per-cluster snapshots
    pub cluster_states: Vec<ClusterSnapshot>,

    /// Synapse correlations: (source, target, correlation)
    pub correlations: Vec<(usize, usize, f64)>,

    /// Output spike counts
    pub output_counts: Vec<u32>,

    /// Total memory usage in bytes
    pub memory_bytes: usize,
}

impl QuantumNeuralSnapshot {
    /// Capture current state of a hybrid network.
    pub fn capture(network: &HybridQuantumNetwork) -> Self {
        let mut cluster_states = Vec::new();

        for cluster_id in 0..network.n_clusters() {
            if let Some(cluster) = network.cluster(cluster_id) {
                let n_qubits = cluster.n_qubits();

                let z_expectations: Vec<f64> =
                    (0..n_qubits).map(|q| cluster.z_expectation(q)).collect();

                let x_expectations: Vec<f64> =
                    (0..n_qubits).map(|q| cluster.x_expectation(q)).collect();

                let split = n_qubits / 2;
                let entropy = if split > 0 {
                    cluster.entanglement_entropy(split)
                } else {
                    0.0
                };

                cluster_states.push(ClusterSnapshot {
                    id: cluster.id(),
                    active: cluster.active,
                    entropy,
                    z_expectations,
                    x_expectations,
                    total_z: cluster.total_z_expectation(),
                    total_probability: cluster.total_probability(),
                });
            }
        }

        // Get synapse correlations from backbone
        let mut correlations = Vec::new();
        let backbone = network.backbone();
        for i in 0..backbone.num_synapses() {
            if let Some(syn) = backbone.synapse(i) {
                let corr = syn.correlation as f64 / 1000.0; // Normalize from i16
                correlations.push((syn.source, syn.target, corr));
            }
        }

        Self {
            tick: network.tick(),
            backbone_spikes: Vec::new(), // Filled by caller if needed
            active_clusters: network.active_cluster_count(),
            cluster_states,
            correlations,
            output_counts: network.get_output_spike_counts(),
            memory_bytes: network.memory_bytes(),
        }
    }

    /// Get average entropy across all active clusters.
    pub fn average_entropy(&self) -> f64 {
        let active: Vec<_> = self.cluster_states.iter().filter(|c| c.active).collect();
        if active.is_empty() {
            return 0.0;
        }
        active.iter().map(|c| c.entropy).sum::<f64>() / active.len() as f64
    }

    /// Get average correlation magnitude across all synapses.
    pub fn average_correlation(&self) -> f64 {
        if self.correlations.is_empty() {
            return 0.0;
        }
        self.correlations
            .iter()
            .map(|(_, _, c)| c.abs())
            .sum::<f64>()
            / self.correlations.len() as f64
    }

    /// Get strongest positive correlation.
    pub fn max_correlation(&self) -> f64 {
        self.correlations
            .iter()
            .map(|(_, _, c)| *c)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Get strongest negative correlation.
    pub fn min_correlation(&self) -> f64 {
        self.correlations
            .iter()
            .map(|(_, _, c)| *c)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
}

/// Recorder for quantum-neural network time series.
///
/// Uses a ring buffer to maintain a fixed-size history.
#[derive(Clone, Debug)]
pub struct QuantumNeuralRecorder {
    /// Ring buffer of snapshots
    snapshots: Vec<QuantumNeuralSnapshot>,

    /// Maximum number of snapshots to retain
    max_snapshots: usize,

    /// Total snapshots recorded (may exceed max_snapshots)
    total_recorded: u64,
}

impl QuantumNeuralRecorder {
    /// Create a new recorder with specified capacity.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::with_capacity(max_snapshots.min(10000)),
            max_snapshots: max_snapshots.max(1),
            total_recorded: 0,
        }
    }

    /// Record a snapshot from the network.
    pub fn record(&mut self, network: &HybridQuantumNetwork) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0); // Ring buffer behavior
        }
        self.snapshots.push(QuantumNeuralSnapshot::capture(network));
        self.total_recorded += 1;
    }

    /// Record a snapshot with spike information.
    pub fn record_with_spikes(&mut self, network: &HybridQuantumNetwork, spikes: &[usize]) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0);
        }
        let mut snapshot = QuantumNeuralSnapshot::capture(network);
        snapshot.backbone_spikes = spikes.to_vec();
        self.snapshots.push(snapshot);
        self.total_recorded += 1;
    }

    /// Get number of snapshots currently stored.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get total number of snapshots recorded (including dropped).
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Clear all recorded snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Get snapshot by index (0 = oldest).
    pub fn get(&self, index: usize) -> Option<&QuantumNeuralSnapshot> {
        self.snapshots.get(index)
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&QuantumNeuralSnapshot> {
        self.snapshots.last()
    }

    /// Get all snapshots as a slice.
    pub fn snapshots(&self) -> &[QuantumNeuralSnapshot] {
        &self.snapshots
    }

    // =========================================================================
    // TIME SERIES EXTRACTION
    // =========================================================================

    /// Get entropy time series for a specific cluster.
    pub fn entropy_series(&self, cluster_id: usize) -> Vec<f64> {
        self.snapshots
            .iter()
            .filter_map(|s| s.cluster_states.get(cluster_id).map(|c| c.entropy))
            .collect()
    }

    /// Get average entropy time series (across all active clusters).
    pub fn average_entropy_series(&self) -> Vec<f64> {
        self.snapshots.iter().map(|s| s.average_entropy()).collect()
    }

    /// Get Z expectation time series for a cluster qubit.
    pub fn z_expectation_series(&self, cluster_id: usize, qubit: usize) -> Vec<f64> {
        self.snapshots
            .iter()
            .filter_map(|s| {
                s.cluster_states
                    .get(cluster_id)
                    .and_then(|c| c.z_expectations.get(qubit).copied())
            })
            .collect()
    }

    /// Get active cluster count time series.
    pub fn active_clusters_series(&self) -> Vec<usize> {
        self.snapshots.iter().map(|s| s.active_clusters).collect()
    }

    /// Get output spike count time series for an output neuron.
    pub fn output_count_series(&self, output_idx: usize) -> Vec<u32> {
        self.snapshots
            .iter()
            .filter_map(|s| s.output_counts.get(output_idx).copied())
            .collect()
    }

    /// Get correlation time series for a specific synapse.
    pub fn correlation_series(&self, source: usize, target: usize) -> Vec<f64> {
        self.snapshots
            .iter()
            .filter_map(|s| {
                s.correlations
                    .iter()
                    .find(|(src, tgt, _)| *src == source && *tgt == target)
                    .map(|(_, _, c)| *c)
            })
            .collect()
    }

    /// Get average correlation time series.
    pub fn average_correlation_series(&self) -> Vec<f64> {
        self.snapshots
            .iter()
            .map(|s| s.average_correlation())
            .collect()
    }

    /// Get spike rate for a neuron over a sliding window.
    pub fn spike_rate(&self, neuron: usize, window: usize) -> Vec<f64> {
        if self.snapshots.len() < window || window == 0 {
            return Vec::new();
        }

        self.snapshots
            .windows(window)
            .map(|w| {
                let count = w
                    .iter()
                    .filter(|s| s.backbone_spikes.contains(&neuron))
                    .count();
                count as f64 / window as f64
            })
            .collect()
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get statistics summary for the recording period.
    pub fn statistics(&self) -> RecordingStatistics {
        if self.snapshots.is_empty() {
            return RecordingStatistics::default();
        }

        let entropy_series = self.average_entropy_series();
        let corr_series = self.average_correlation_series();
        let active_series = self.active_clusters_series();

        RecordingStatistics {
            tick_count: self.snapshots.len() as u64,
            entropy_mean: mean(&entropy_series),
            entropy_std: std_dev(&entropy_series),
            entropy_max: max_val(&entropy_series),
            correlation_mean: mean(&corr_series),
            correlation_std: std_dev(&corr_series),
            active_clusters_mean: mean(
                &active_series.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
        }
    }
}

/// Summary statistics for a recording.
#[derive(Clone, Debug, Default)]
pub struct RecordingStatistics {
    /// Number of ticks recorded
    pub tick_count: u64,

    /// Mean entropy across time
    pub entropy_mean: f64,

    /// Standard deviation of entropy
    pub entropy_std: f64,

    /// Maximum entropy observed
    pub entropy_max: f64,

    /// Mean correlation magnitude
    pub correlation_mean: f64,

    /// Standard deviation of correlation
    pub correlation_std: f64,

    /// Mean number of active clusters
    pub active_clusters_mean: f64,
}

// Helper functions for statistics
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn max_val(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::stabilizer_hybrid::HybridNetworkConfig;

    #[test]
    fn test_snapshot_capture() {
        let config = HybridNetworkConfig::small();
        let network = HybridQuantumNetwork::new(config, 42);

        let snapshot = QuantumNeuralSnapshot::capture(&network);

        assert_eq!(snapshot.tick, 0);
        assert!(!snapshot.cluster_states.is_empty());
        assert!(snapshot.memory_bytes > 0);
    }

    #[test]
    fn test_cluster_snapshot_metrics() {
        let config = HybridNetworkConfig::small();
        let network = HybridQuantumNetwork::new(config, 42);

        let snapshot = QuantumNeuralSnapshot::capture(&network);

        if let Some(cluster) = snapshot.cluster_states.first() {
            assert!(cluster.is_normalized());
            assert!(cluster.entropy >= 0.0);
        }
    }

    #[test]
    fn test_recorder_creation() {
        let recorder = QuantumNeuralRecorder::new(100);
        assert_eq!(recorder.len(), 0);
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_recorder_record() {
        let mut recorder = QuantumNeuralRecorder::new(100);
        let config = HybridNetworkConfig::small();
        let network = HybridQuantumNetwork::new(config, 42);

        recorder.record(&network);

        assert_eq!(recorder.len(), 1);
        assert!(!recorder.is_empty());
        assert!(recorder.latest().is_some());
    }

    #[test]
    fn test_recorder_ring_buffer() {
        let mut recorder = QuantumNeuralRecorder::new(5);
        let config = HybridNetworkConfig::small();
        let mut network = HybridQuantumNetwork::new(config, 42);

        // Record more than capacity
        for _ in 0..10 {
            recorder.record(&network);
            network.step(&[200, 150]);
        }

        // Should only have 5 snapshots
        assert_eq!(recorder.len(), 5);
        assert_eq!(recorder.total_recorded(), 10);
    }

    #[test]
    fn test_entropy_series() {
        let mut recorder = QuantumNeuralRecorder::new(100);
        let config = HybridNetworkConfig::small();
        let mut network = HybridQuantumNetwork::new(config, 42);

        for _ in 0..10 {
            recorder.record(&network);
            network.step(&[200, 150]);
        }

        let series = recorder.entropy_series(0);
        assert_eq!(series.len(), 10);
    }

    #[test]
    fn test_statistics() {
        let mut recorder = QuantumNeuralRecorder::new(100);
        let config = HybridNetworkConfig::small();
        let mut network = HybridQuantumNetwork::new(config, 42);

        for _ in 0..20 {
            recorder.record(&network);
            network.step(&[200, 150]);
        }

        let stats = recorder.statistics();
        assert_eq!(stats.tick_count, 20);
        assert!(stats.entropy_mean >= 0.0);
    }

    #[test]
    fn test_average_metrics() {
        let config = HybridNetworkConfig::small();
        let network = HybridQuantumNetwork::new(config, 42);

        let snapshot = QuantumNeuralSnapshot::capture(&network);

        let avg_entropy = snapshot.average_entropy();
        let avg_corr = snapshot.average_correlation();

        // Should be valid numbers
        assert!(!avg_entropy.is_nan());
        assert!(!avg_corr.is_nan());
    }
}
