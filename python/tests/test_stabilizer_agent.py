"""
Tests for Stabilizer-based Gymnasium agents.

Sprint 59.0: Reinforcement learning agents using stabilizer quantum neurons.
"""

import pytest
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces

        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False


# Skip all tests if gymnasium not available
pytestmark = pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")

from tileuniverse.rl import (
    StabilizerSNNAgent,
    HybridQuantumSNNAgent,
    train_stabilizer_agent,
)
from tileuniverse import FeedbackMode


class TestStabilizerSNNAgent:
    """Tests for StabilizerSNNAgent."""

    def test_create_agent_direct(self):
        agent = StabilizerSNNAgent(
            n_inputs=4, n_actions=2, hidden_layers=[16], seed=42
        )
        assert agent.n_inputs == 4
        assert agent.n_actions == 2
        assert agent.episodes == 0
        assert agent.total_steps == 0

    def test_for_env_discrete_observation(self):
        env = gym.make("CartPole-v1")
        agent = StabilizerSNNAgent.for_env(env)

        # CartPole has 4 continuous observations
        assert agent.n_inputs == 4
        # CartPole has 2 discrete actions
        assert agent.n_actions == 2
        env.close()

    def test_act_returns_valid_action(self):
        env = gym.make("CartPole-v1")
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)

        assert isinstance(action, int)
        assert 0 <= action < 2
        env.close()

    def test_learn_accepts_reward(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        agent.act(obs)
        agent.learn(1.0)  # Positive reward
        agent.learn(-1.0)  # Negative reward
        # Should not raise

    def test_end_episode_returns_stats(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for _ in range(10):
            agent.act(obs)
            agent.learn(1.0)

        stats = agent.end_episode()

        assert "episode" in stats
        assert "reward" in stats
        assert "length" in stats
        assert stats["episode"] == 0
        assert stats["reward"] == 10.0
        assert stats["length"] == 10

    def test_episode_tracking(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for ep in range(3):
            for _ in range(5):
                agent.act(obs)
                agent.learn(1.0)
            stats = agent.end_episode()
            assert stats["episode"] == ep

        assert agent.episodes == 3

    def test_reset_clears_state(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for _ in range(5):
            agent.act(obs)
            agent.learn(1.0)
        agent.end_episode()

        agent.reset()

        assert agent.episodes == 0
        assert agent.total_steps == 0

    def test_get_stats(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, hidden_layers=[16], seed=42)
        stats = agent.get_stats()

        assert "episodes" in stats
        assert "total_steps" in stats
        assert "num_neurons" in stats
        assert "memory_kb" in stats
        assert stats["num_neurons"] == 4 + 16 + 2

    def test_deterministic_with_seed(self):
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        agent1 = StabilizerSNNAgent(n_inputs=4, n_actions=4, seed=12345)
        agent2 = StabilizerSNNAgent(n_inputs=4, n_actions=4, seed=12345)

        actions1 = [agent1.act(obs) for _ in range(10)]
        actions2 = [agent2.act(obs) for _ in range(10)]

        assert actions1 == actions2

    def test_learning_disabled(self):
        agent = StabilizerSNNAgent(
            n_inputs=4, n_actions=2, learning_enabled=False, seed=42
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        agent.act(obs)
        agent.learn(100.0)  # Should not affect network (learning disabled)

        stats = agent.end_episode()
        assert stats["reward"] == 100.0  # Reward still tracked


class TestHybridQuantumSNNAgent:
    """Tests for HybridQuantumSNNAgent."""

    def test_create_agent_direct(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4,
            n_actions=2,
            hidden_layers=[16],
            n_clusters=1,
            cluster_size=4,
            seed=42,
        )
        assert agent.n_inputs == 4
        assert agent.n_actions == 2

    def test_for_env(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(env, n_clusters=2, cluster_size=4)

        assert agent.n_inputs == 4
        assert agent.n_actions == 2
        env.close()

    def test_act_returns_valid_action(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(env, n_clusters=1, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)

        assert isinstance(action, int)
        assert 0 <= action < 2
        env.close()

    def test_cluster_entropy_property(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4, n_actions=2, n_clusters=2, cluster_size=4, seed=42
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for _ in range(5):
            agent.act(obs)

        entropy = agent.cluster_entropy
        assert len(entropy) == 2
        for e in entropy:
            assert e >= 0.0

    def test_total_z_expectation_property(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4, n_actions=2, n_clusters=2, cluster_size=4, seed=42
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for _ in range(5):
            agent.act(obs)

        z_exp = agent.total_z_expectation
        total_qubits = 2 * 4
        assert -total_qubits <= z_exp <= total_qubits

    def test_end_episode_with_entropy(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4, n_actions=2, n_clusters=1, cluster_size=4, seed=42
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])

        for _ in range(10):
            agent.act(obs)
            agent.learn(1.0)

        stats = agent.end_episode()

        assert "episode" in stats
        assert "avg_entropy" in stats
        assert "active_clusters" in stats

    def test_cluster_topology_chain(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(
            env, n_clusters=3, cluster_topology="chain", seed=42
        )

        obs, _ = env.reset()
        action = agent.act(obs)
        assert 0 <= action < 2
        env.close()

    def test_cluster_topology_ring(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(
            env, n_clusters=3, cluster_topology="ring", seed=42
        )

        obs, _ = env.reset()
        action = agent.act(obs)
        assert 0 <= action < 2
        env.close()

    def test_cluster_topology_mesh(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(
            env, n_clusters=3, cluster_topology="mesh", seed=42
        )

        obs, _ = env.reset()
        action = agent.act(obs)
        assert 0 <= action < 2
        env.close()

    def test_cluster_topology_star(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(
            env, n_clusters=4, cluster_topology="star", seed=42
        )

        obs, _ = env.reset()
        action = agent.act(obs)
        assert 0 <= action < 2
        env.close()

    def test_feedback_mode(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4,
            n_actions=2,
            n_clusters=1,
            cluster_size=4,
            feedback_mode=FeedbackMode.GATED,
            seed=42,
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        action = agent.act(obs)
        assert 0 <= action < 2

    def test_adaptive_threshold(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4,
            n_actions=2,
            n_clusters=2,
            cluster_size=4,
            adaptive_threshold=0.8,
            seed=42,
        )
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        action = agent.act(obs)
        assert 0 <= action < 2

    def test_add_bridge(self):
        from tileuniverse import BridgeGate

        agent = HybridQuantumSNNAgent(
            n_inputs=4, n_actions=2, n_clusters=2, cluster_size=4, seed=42
        )
        agent.add_bridge(0, 0, 1, 0, BridgeGate.DISTRIBUTED_CZ)
        # Should not raise

    def test_get_stats(self):
        agent = HybridQuantumSNNAgent(
            n_inputs=4,
            n_actions=2,
            hidden_layers=[16],
            n_clusters=2,
            cluster_size=4,
            seed=42,
        )
        stats = agent.get_stats()

        assert "n_clusters" in stats
        assert "active_clusters" in stats
        assert "n_bridges" in stats
        assert "cluster_entropy" in stats


class TestTrainStabilizerAgent:
    """Tests for the train_stabilizer_agent helper function."""

    def test_train_stabilizer_agent_basic(self):
        env = gym.make("CartPole-v1")
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        stats = train_stabilizer_agent(env, agent, n_episodes=3, verbose=0)

        assert len(stats) == 3
        assert all("reward" in s for s in stats)
        assert all("length" in s for s in stats)
        env.close()

    def test_train_hybrid_agent(self):
        env = gym.make("CartPole-v1")
        agent = HybridQuantumSNNAgent.for_env(env, n_clusters=1, seed=42)

        stats = train_stabilizer_agent(env, agent, n_episodes=3, verbose=0)

        assert len(stats) == 3
        env.close()

    def test_train_with_max_steps(self):
        env = gym.make("CartPole-v1")
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        stats = train_stabilizer_agent(
            env, agent, n_episodes=2, max_steps=10, verbose=0
        )

        # Episodes should be truncated at max_steps
        assert all(s["length"] <= 10 for s in stats)
        env.close()

    def test_train_with_callback(self):
        env = gym.make("CartPole-v1")
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        callback_calls = []

        def callback(episode, stats):
            callback_calls.append((episode, stats["reward"]))

        train_stabilizer_agent(
            env, agent, n_episodes=3, verbose=0, callback=callback
        )

        assert len(callback_calls) == 3
        assert callback_calls[0][0] == 0
        assert callback_calls[1][0] == 1
        assert callback_calls[2][0] == 2
        env.close()


class TestBoxObservationSpace:
    """Tests for Box observation spaces."""

    def test_continuous_observations(self):
        env = gym.make("CartPole-v1")  # Box observation space
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)

        assert 0 <= action < 2
        env.close()

    def test_mountaincar(self):
        """Test with MountainCar-v0 (2D continuous observation)."""
        env = gym.make("MountainCar-v0")
        agent = StabilizerSNNAgent.for_env(env, seed=42)

        assert agent.n_inputs == 2
        assert agent.n_actions == 3

        obs, _ = env.reset()
        action = agent.act(obs)

        assert 0 <= action < 3
        env.close()

    def test_acrobot(self):
        """Test with Acrobot-v1 (6D continuous observation)."""
        env = gym.make("Acrobot-v1")
        agent = HybridQuantumSNNAgent.for_env(env, n_clusters=1, seed=42)

        assert agent.n_inputs == 6
        assert agent.n_actions == 3

        obs, _ = env.reset()
        action = agent.act(obs)

        assert 0 <= action < 3
        env.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_hidden_layers(self):
        # No hidden layers - direct input-output
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=2, hidden_layers=[], seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        action = agent.act(obs)
        assert 0 <= action < 2

    def test_large_hidden_layers(self):
        agent = StabilizerSNNAgent(
            n_inputs=8, n_actions=4, hidden_layers=[128, 64, 32], seed=42
        )
        stats = agent.get_stats()
        assert stats["num_neurons"] == 8 + 128 + 64 + 32 + 4

    def test_single_action(self):
        # Degenerate case - only 1 action
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=1, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        action = agent.act(obs)
        assert action == 0

    def test_many_actions(self):
        agent = StabilizerSNNAgent(n_inputs=4, n_actions=10, seed=42)
        obs = np.array([0.5, -0.5, 0.1, -0.1])
        action = agent.act(obs)
        assert 0 <= action < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
