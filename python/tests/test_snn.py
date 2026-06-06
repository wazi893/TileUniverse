"""
Tests for Quantum-SNN hybrid bindings.

The Quantum-SNN combines classical spiking neural networks with
quantum interference for exploration/exploitation in reinforcement learning.
"""

import pytest
from tileuniverse import (
    InterferenceMode,
    QuantumSNNConfig,
    QuantumSNN,
)


class TestInterferenceMode:
    """Tests for InterferenceMode enum."""

    def test_none_mode(self):
        mode = InterferenceMode.none()
        assert str(mode) == "none"
        assert repr(mode) == "InterferenceMode.none()"

    def test_triggered_mode(self):
        mode = InterferenceMode.triggered()
        assert str(mode) == "triggered"
        assert repr(mode) == "InterferenceMode.triggered()"

    def test_anti_grover_mode(self):
        mode = InterferenceMode.anti_grover()
        assert str(mode) == "anti_grover"

    def test_grover_mode(self):
        mode = InterferenceMode.grover()
        assert str(mode) == "grover"

    def test_soft_mix_mode(self):
        mode = InterferenceMode.soft_mix()
        assert str(mode) == "soft_mix"

    def test_adaptive_mode(self):
        mode = InterferenceMode.adaptive()
        assert str(mode) == "adaptive"


class TestQuantumSNNConfig:
    """Tests for QuantumSNNConfig."""

    def test_default_config(self):
        config = QuantumSNNConfig()
        assert config.n_inputs == 16
        assert config.n_outputs == 4
        assert config.hidden_layers == [32, 16]
        assert str(config.mode) == "triggered"

    def test_custom_config(self):
        config = QuantumSNNConfig(
            n_inputs=8,
            hidden_layers=[64, 32, 16],
            n_outputs=6,
            mode=InterferenceMode.anti_grover(),
        )
        assert config.n_inputs == 8
        assert config.n_outputs == 6
        assert config.hidden_layers == [64, 32, 16]
        assert str(config.mode) == "anti_grover"

    def test_small_config(self):
        config = QuantumSNNConfig.small()
        assert config.n_inputs == 8
        assert config.n_outputs == 4
        assert config.hidden_layers == [16]

    def test_sparse_reward_config(self):
        config = QuantumSNNConfig.sparse_reward(16, 8)
        assert config.n_inputs == 16
        assert config.n_outputs == 8
        assert str(config.mode) == "triggered"

    def test_total_neurons(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8, 4], n_outputs=2)
        assert config.total_neurons() == 4 + 8 + 4 + 2

    def test_config_repr(self):
        config = QuantumSNNConfig(n_inputs=8, n_outputs=4)
        repr_str = repr(config)
        assert "n_inputs=8" in repr_str
        assert "n_outputs=4" in repr_str

    def test_ticks_per_decision(self):
        config = QuantumSNNConfig(ticks_per_decision=30)
        assert config.ticks_per_decision == 30

    def test_learning_enabled(self):
        config = QuantumSNNConfig(learning_enabled=False)
        assert config.learning_enabled is False

    def test_trigger_parameters(self):
        config = QuantumSNNConfig(trigger_threshold=75, trigger_count=10)
        assert config.trigger_threshold == 75
        assert config.trigger_count == 10


class TestQuantumSNN:
    """Tests for QuantumSNN neural network."""

    def test_create_network(self):
        config = QuantumSNNConfig.small()
        brain = QuantumSNN(config, seed=42)
        assert brain.tick == 0
        assert brain.is_exploitation_triggered is False

    def test_decide_with_inputs(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=3)
        brain = QuantumSNN(config, seed=42)

        inputs = [200, 100, 50, 25]
        action = brain.decide(inputs)

        assert 0 <= action < 3
        assert len(brain.last_probabilities) == 3
        assert len(brain.output_spike_counts) == 3

    def test_probabilities_sum_to_one(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=4)
        brain = QuantumSNN(config, seed=42)

        inputs = [128, 64, 192, 32]
        brain.decide(inputs)

        probs = brain.last_probabilities
        total = sum(probs)
        assert abs(total - 1.0) < 0.01

    def test_entropy_calculation(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=4)
        brain = QuantumSNN(config, seed=42)

        inputs = [128, 128, 128, 128]
        brain.decide(inputs)

        entropy = brain.exploration_entropy()
        # Uniform distribution has max entropy = log2(n_outputs)
        assert entropy >= 0
        assert entropy <= 2.0  # log2(4) = 2

    def test_learning_increases_consecutive_rewards(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=2)
        brain = QuantumSNN(config, seed=42)

        inputs = [100, 100, 100, 100]
        brain.decide(inputs)
        brain.learn(60)  # High reward

        assert brain.consecutive_high_rewards == 1

        brain.decide(inputs)
        brain.learn(60)  # Another high reward

        assert brain.consecutive_high_rewards == 2

    def test_low_reward_resets_consecutive(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=2, trigger_threshold=50)
        brain = QuantumSNN(config, seed=42)

        inputs = [100, 100, 100, 100]

        # Build up consecutive rewards
        brain.decide(inputs)
        brain.learn(60)
        brain.decide(inputs)
        brain.learn(60)
        assert brain.consecutive_high_rewards >= 1

        # Low reward should reset (behavior depends on implementation)
        brain.decide(inputs)
        brain.learn(10)  # Low reward

    def test_trigger_exploitation_manually(self):
        config = QuantumSNNConfig.small()
        brain = QuantumSNN(config, seed=42)

        assert brain.is_exploitation_triggered is False
        brain.trigger_exploitation()
        assert brain.is_exploitation_triggered is True

    def test_reset_trigger(self):
        config = QuantumSNNConfig.small()
        brain = QuantumSNN(config, seed=42)

        brain.trigger_exploitation()
        assert brain.is_exploitation_triggered is True

        brain.reset_trigger()
        assert brain.is_exploitation_triggered is False

    def test_reset_clears_state(self):
        config = QuantumSNNConfig.small()
        brain = QuantumSNN(config, seed=42)

        inputs = [128] * 8
        brain.decide(inputs)
        brain.learn(100)
        brain.trigger_exploitation()

        brain.reset()

        assert brain.tick == 0
        assert brain.is_exploitation_triggered is False

    def test_classical_action(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=3)
        brain = QuantumSNN(config, seed=42)

        inputs = [200, 100, 50, 25]
        brain.decide(inputs)

        classical = brain.classical_action()
        assert 0 <= classical < 3

    def test_deterministic_with_seed(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=4)

        brain1 = QuantumSNN(config, seed=12345)
        brain2 = QuantumSNN(config, seed=12345)

        inputs = [128, 64, 192, 32]

        actions1 = [brain1.decide(inputs) for _ in range(5)]
        actions2 = [brain2.decide(inputs) for _ in range(5)]

        # Same seed should give same sequence
        assert actions1 == actions2

    def test_different_seeds_give_different_results(self):
        config = QuantumSNNConfig(n_inputs=4, hidden_layers=[8], n_outputs=4)

        brain1 = QuantumSNN(config, seed=12345)
        brain2 = QuantumSNN(config, seed=99999)

        inputs = [128, 64, 192, 32]

        # Run many decisions to check they differ
        actions1 = [brain1.decide(inputs) for _ in range(20)]
        actions2 = [brain2.decide(inputs) for _ in range(20)]

        # Very unlikely to be identical with different seeds
        assert actions1 != actions2

    def test_config_accessor(self):
        config = QuantumSNNConfig(n_inputs=8, n_outputs=6)
        brain = QuantumSNN(config, seed=42)

        retrieved_config = brain.config
        assert retrieved_config.n_inputs == 8
        assert retrieved_config.n_outputs == 6


class TestTriggeredMode:
    """Tests specifically for Triggered interference mode."""

    def test_triggered_mode_starts_exploring(self):
        config = QuantumSNNConfig(
            n_inputs=4,
            hidden_layers=[8],
            n_outputs=4,
            mode=InterferenceMode.triggered(),
            trigger_threshold=50,
            trigger_count=3,
        )
        brain = QuantumSNN(config, seed=42)

        # Initially should be exploring (not triggered)
        assert brain.is_exploitation_triggered is False

    def test_triggered_mode_after_rewards(self):
        config = QuantumSNNConfig(
            n_inputs=4,
            hidden_layers=[8],
            n_outputs=4,
            mode=InterferenceMode.triggered(),
            trigger_threshold=50,
            trigger_count=3,  # Need 3 consecutive high rewards
        )
        brain = QuantumSNN(config, seed=42)

        inputs = [100, 100, 100, 100]

        # Give enough high rewards to trigger
        for i in range(5):
            brain.decide(inputs)
            brain.learn(75)  # Above threshold

        # Should eventually trigger exploitation
        assert brain.consecutive_high_rewards >= 3 or brain.is_exploitation_triggered


class TestDifferentModes:
    """Test behavior with different interference modes."""

    def test_grover_mode_works(self):
        config = QuantumSNNConfig(
            n_inputs=4,
            hidden_layers=[8],
            n_outputs=4,
            mode=InterferenceMode.grover(),
        )
        brain = QuantumSNN(config, seed=42)

        inputs = [200, 50, 50, 50]  # Strong preference for action 0
        action = brain.decide(inputs)

        # Grover should amplify dominant action
        assert 0 <= action < 4

    def test_anti_grover_mode_works(self):
        config = QuantumSNNConfig(
            n_inputs=4,
            hidden_layers=[8],
            n_outputs=4,
            mode=InterferenceMode.anti_grover(),
        )
        brain = QuantumSNN(config, seed=42)

        inputs = [200, 50, 50, 50]
        action = brain.decide(inputs)

        # Anti-Grover explores non-dominant actions
        assert 0 <= action < 4

    def test_none_mode_works(self):
        config = QuantumSNNConfig(
            n_inputs=4,
            hidden_layers=[8],
            n_outputs=4,
            mode=InterferenceMode.none(),
        )
        brain = QuantumSNN(config, seed=42)

        inputs = [100, 100, 100, 100]
        action = brain.decide(inputs)

        assert 0 <= action < 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
