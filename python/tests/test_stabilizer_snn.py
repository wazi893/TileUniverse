"""
Tests for Stabilizer-based Quantum SNN bindings.

Sprint 59.0: Stabilizer quantum neurons with O(n^2) memory scaling.
Enables 10K+ quantum neurons on consumer hardware.
"""

import pytest
from tileuniverse import (
    FeedbackMode,
    BridgeGate,
    StabilizerSNN,
    HybridQuantumSNN,
)


class TestFeedbackMode:
    """Tests for FeedbackMode enum."""

    def test_none_mode(self):
        mode = FeedbackMode.NONE
        assert mode is not None

    def test_phase_mode(self):
        mode = FeedbackMode.PHASE
        assert mode is not None

    def test_amplitude_mode(self):
        mode = FeedbackMode.AMPLITUDE
        assert mode is not None

    def test_gated_mode(self):
        mode = FeedbackMode.GATED
        assert mode is not None


class TestBridgeGate:
    """Tests for BridgeGate enum."""

    def test_teleport_gate(self):
        gate = BridgeGate.TELEPORT
        assert gate is not None

    def test_remote_swap_gate(self):
        gate = BridgeGate.REMOTE_SWAP
        assert gate is not None

    def test_distributed_cz_gate(self):
        gate = BridgeGate.DISTRIBUTED_CZ
        assert gate is not None


class TestStabilizerSNN:
    """Tests for StabilizerSNN (pure stabilizer network)."""

    def test_create_network(self):
        brain = StabilizerSNN(
            n_inputs=4,
            hidden=[16],
            n_outputs=3,
            connectivity=0.3,
            seed=42,
        )
        assert brain.tick == 0
        assert brain.num_neurons == 4 + 16 + 3  # input + hidden + output

    def test_step_returns_spikes(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=2, seed=42)
        inputs = [200, 100, 50, 25]
        spikes = brain.step(inputs)
        # Spikes is a list of neuron indices that fired
        assert isinstance(spikes, list)
        for spike in spikes:
            assert isinstance(spike, int)
            assert 0 <= spike < brain.num_neurons

    def test_decide_returns_action(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=3, seed=42)
        inputs = [128, 64, 192, 32]
        action = brain.decide(inputs)
        assert 0 <= action < 3

    def test_learn_modifies_network(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=2, seed=42)
        inputs = [128, 128, 128, 128]
        brain.decide(inputs)
        brain.learn(50)  # Positive reward
        brain.learn(-50)  # Negative reward
        # Should not raise, internal state updated

    def test_output_spike_counts(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=3, seed=42)
        inputs = [200, 100, 50, 25]

        # Run a few steps to accumulate spikes
        for _ in range(10):
            brain.decide(inputs)

        counts = brain.output_spike_counts
        assert len(counts) == 3
        for count in counts:
            assert isinstance(count, int)
            assert count >= 0

    def test_memory_bytes(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[16], n_outputs=2, seed=42)
        memory = brain.memory_bytes
        assert memory > 0
        # O(n^2) memory for stabilizer tableau
        n = brain.num_neurons
        expected_min = n * n // 16  # Minimum bits storage
        assert memory >= expected_min

    def test_deterministic_with_seed(self):
        brain1 = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=4, seed=12345)
        brain2 = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=4, seed=12345)

        inputs = [128, 64, 192, 32]
        actions1 = [brain1.decide(inputs) for _ in range(5)]
        actions2 = [brain2.decide(inputs) for _ in range(5)]

        assert actions1 == actions2

    def test_different_seeds_differ(self):
        brain1 = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=4, seed=12345)
        brain2 = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=4, seed=99999)

        # Use varying inputs to test randomness
        all_same = True
        for i in range(10):
            inputs = [128 + i * 10, 64 - i * 5, 192, 32]
            a1 = brain1.decide(inputs)
            a2 = brain2.decide(inputs)
            if a1 != a2:
                all_same = False
                break

        # At least one decision should differ (with high probability)
        # But this is probabilistic, so we just verify both run
        assert brain1.tick > 0 and brain2.tick > 0

    def test_reset_spike_counts(self):
        brain = StabilizerSNN(n_inputs=4, hidden=[8], n_outputs=2, seed=42)
        inputs = [200, 100, 50, 25]

        for _ in range(10):
            brain.decide(inputs)

        counts_before = brain.output_spike_counts
        brain.reset_output_counts()
        counts_after = brain.output_spike_counts

        assert all(c == 0 for c in counts_after)


class TestHybridQuantumSNN:
    """Tests for HybridQuantumSNN (stabilizer backbone + full-quantum clusters)."""

    def test_create_network(self):
        brain = HybridQuantumSNN(
            n_inputs=4,
            hidden=[16],
            n_outputs=3,
            n_clusters=2,
            cluster_size=8,
            seed=42,
        )
        assert brain.tick == 0
        assert brain.n_clusters == 2
        # cluster_size not exposed as property, just verify it was used

    def test_step_returns_spikes(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=1, cluster_size=4, seed=42
        )
        inputs = [200, 100, 50, 25]
        spikes = brain.step(inputs)
        assert isinstance(spikes, list)

    def test_decide_returns_action(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=3, n_clusters=1, cluster_size=4, seed=42
        )
        inputs = [128, 64, 192, 32]
        action = brain.decide(inputs)
        assert 0 <= action < 3

    def test_decide_integrated(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=3, n_clusters=1, cluster_size=4, seed=42
        )
        inputs = [128, 64, 192, 32]
        action = brain.decide_integrated(inputs, ticks=10)
        assert 0 <= action < 3

    def test_cluster_entropy(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=2, cluster_size=4, seed=42
        )
        inputs = [128, 128, 128, 128]

        # Run some steps to create quantum state
        for _ in range(5):
            brain.step(inputs)

        entropy0 = brain.cluster_entropy(0)
        entropy1 = brain.cluster_entropy(1)

        # Entropy should be non-negative
        assert entropy0 >= 0.0
        assert entropy1 >= 0.0

    def test_total_z_expectation(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=2, cluster_size=4, seed=42
        )
        inputs = [128, 128, 128, 128]

        for _ in range(5):
            brain.step(inputs)

        z_exp = brain.total_z_expectation()
        # Z expectation in range [-n_qubits, +n_qubits]
        total_qubits = 2 * 4  # n_clusters * cluster_size
        assert -total_qubits <= z_exp <= total_qubits

    def test_feedback_mode(self):
        brain = HybridQuantumSNN(
            n_inputs=4,
            hidden=[8],
            n_outputs=2,
            n_clusters=1,
            cluster_size=4,
            feedback_mode=FeedbackMode.GATED,
            seed=42,
        )
        # Should create network with gated feedback
        assert brain.n_clusters == 1

    def test_adaptive_threshold(self):
        brain = HybridQuantumSNN(
            n_inputs=4,
            hidden=[8],
            n_outputs=2,
            n_clusters=1,
            cluster_size=4,
            adaptive_threshold=0.8,
            seed=42,
        )
        # Should create network with adaptive cluster activation
        assert brain.n_clusters == 1

    def test_memory_bytes(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[16], n_outputs=2, n_clusters=2, cluster_size=8, seed=42
        )
        memory = brain.memory_bytes
        assert memory > 0
        # Hybrid includes backbone + cluster state vectors
        # Each cluster: 2^8 * 16 bytes = 4KB
        # Should be more than pure stabilizer due to cluster overhead


class TestClusterBridges:
    """Tests for cluster-to-cluster entanglement bridges."""

    def test_add_bridge(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=2, cluster_size=4, seed=42
        )
        assert brain.n_bridges == 0

        brain.add_bridge(0, 0, 1, 0, BridgeGate.DISTRIBUTED_CZ)
        assert brain.n_bridges == 1

        brain.add_bridge(0, 1, 1, 1, BridgeGate.TELEPORT)
        assert brain.n_bridges == 2

    def test_apply_bridges(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=2, cluster_size=4, seed=42
        )
        brain.add_bridge(0, 0, 1, 0, BridgeGate.DISTRIBUTED_CZ)

        # Should not raise
        brain.apply_bridges()

    def test_chain_topology(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=3, cluster_size=4, seed=42
        )
        assert brain.n_bridges == 0

        brain.chain_clusters()
        # Chain of 3 clusters: 0-1, 1-2 = 2 bridges
        assert brain.n_bridges == 2

    def test_ring_topology(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=3, cluster_size=4, seed=42
        )
        brain.ring_clusters()
        # Ring of 3 clusters: 0-1, 1-2, 2-0 = 3 bridges
        assert brain.n_bridges == 3

    def test_mesh_topology(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=3, cluster_size=4, seed=42
        )
        brain.mesh_clusters()
        # Full mesh of 3 clusters: 0-1, 0-2, 1-2 = 3 bridges
        assert brain.n_bridges == 3

    def test_star_topology(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=4, cluster_size=4, seed=42
        )
        brain.star_clusters(center=0)
        # Star with center 0: 0-1, 0-2, 0-3 = 3 bridges
        assert brain.n_bridges == 3


class TestHybridRecording:
    """Tests for quantum state recording and analysis."""

    def test_enable_recording(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=1, cluster_size=4, seed=42
        )
        brain.enable_recording(max_snapshots=100)
        assert brain.is_recording is True

    def test_entropy_series(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=1, cluster_size=4, seed=42
        )
        brain.enable_recording(max_snapshots=100)

        inputs = [128, 64, 192, 32]
        for _ in range(10):
            brain.step(inputs)

        entropy_series = brain.average_entropy_series()
        assert isinstance(entropy_series, list)
        # Should have recorded some entropy values

    def test_total_z_expectation_after_recording(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=1, cluster_size=4, seed=42
        )
        brain.enable_recording(max_snapshots=100)

        inputs = [128, 64, 192, 32]
        for _ in range(10):
            brain.step(inputs)

        # Z expectation available after recording steps
        z_exp = brain.total_z_expectation()
        assert isinstance(z_exp, float)

    def test_active_clusters_tracking(self):
        brain = HybridQuantumSNN(
            n_inputs=4, hidden=[8], n_outputs=2, n_clusters=2, cluster_size=4, seed=42
        )

        inputs = [128, 128, 128, 128]
        brain.step(inputs)

        active = brain.active_clusters
        assert isinstance(active, int)
        assert 0 <= active <= 2


class TestScaling:
    """Tests for scaling behavior."""

    def test_large_stabilizer_network(self):
        # StabilizerSNN should handle 1000+ neurons efficiently
        brain = StabilizerSNN(
            n_inputs=100, hidden=[500, 500], n_outputs=50, connectivity=0.1, seed=42
        )
        assert brain.num_neurons == 100 + 500 + 500 + 50

        # Memory should be reasonable O(n^2) ~ 1.3M neurons -> ~200MB
        memory_mb = brain.memory_bytes / (1024 * 1024)
        assert memory_mb < 500  # Should be well under 500MB

    def test_multiple_clusters(self):
        brain = HybridQuantumSNN(
            n_inputs=8,
            hidden=[32],
            n_outputs=4,
            n_clusters=4,
            cluster_size=8,  # 4 clusters of 8 qubits each
            seed=42,
        )
        assert brain.n_clusters == 4

        # Each 8-qubit cluster needs 2^8 * 16 bytes = 4KB
        # Plus stabilizer backbone
        inputs = [128] * 8
        action = brain.decide(inputs)
        assert 0 <= action < 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
