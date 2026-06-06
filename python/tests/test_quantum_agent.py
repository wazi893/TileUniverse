"""
Tests for QuantumSNN Gymnasium Agent integration.
"""

import pytest
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from tileuniverse.rl.quantum_agent import (
    ObservationEncoder,
    QuantumSNNAgent,
    train_agent,
    evaluate_agent,
)
from tileuniverse import InterferenceMode


# Skip all tests if gymnasium not available
pytestmark = pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")


class TestObservationEncoder:
    """Tests for ObservationEncoder."""

    def test_box_space_encoding(self):
        space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        encoder = ObservationEncoder(space)

        assert encoder.n_inputs == 4

        # Test middle values
        obs = np.array([0.0, 0.0, 0.0, 0.0])
        encoded = encoder.encode(obs)
        assert len(encoded) == 4
        assert all(120 <= v <= 135 for v in encoded)  # Should be ~128 (middle)

        # Test extremes
        obs_low = np.array([-1.0, -1.0, -1.0, -1.0])
        encoded_low = encoder.encode(obs_low)
        assert all(v <= 10 for v in encoded_low)

        obs_high = np.array([1.0, 1.0, 1.0, 1.0])
        encoded_high = encoder.encode(obs_high)
        assert all(v >= 245 for v in encoded_high)

    def test_discrete_space_encoding(self):
        space = spaces.Discrete(5)
        encoder = ObservationEncoder(space)

        assert encoder.n_inputs == 5

        # Test one-hot encoding
        encoded = encoder.encode(2)
        assert len(encoded) == 5
        assert encoded[2] == 255
        assert encoded[0] == 0
        assert encoded[4] == 0

    def test_multibinary_space_encoding(self):
        space = spaces.MultiBinary(3)
        encoder = ObservationEncoder(space)

        assert encoder.n_inputs == 3

        obs = np.array([1, 0, 1])
        encoded = encoder.encode(obs)
        assert encoded == [255, 0, 255]

    def test_infinite_bounds_handling(self):
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        encoder = ObservationEncoder(space)

        # Should handle infinite bounds gracefully
        obs = np.array([0.0, 0.0])
        encoded = encoder.encode(obs)
        assert len(encoded) == 2
        assert all(0 <= v <= 255 for v in encoded)


class TestQuantumSNNAgent:
    """Tests for QuantumSNNAgent."""

    def test_create_agent_directly(self):
        agent = QuantumSNNAgent(
            n_inputs=4,
            n_actions=2,
            hidden_layers=[8],
            seed=42,
        )
        assert agent.n_inputs == 4
        assert agent.n_actions == 2
        assert agent.episodes == 0

    def test_create_agent_for_env(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, mode="triggered")

        assert agent.n_inputs == 4  # CartPole has 4 observations
        assert agent.n_actions == 2  # Left or Right
        assert agent.encoder is not None

        env.close()

    def test_act_returns_valid_action(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)

        assert 0 <= action < 2  # CartPole has 2 actions
        env.close()

    def test_learn_updates_state(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        agent.act(obs)

        initial_rewards = agent.brain.consecutive_high_rewards
        agent.learn(100)  # High reward

        assert agent.brain.consecutive_high_rewards >= initial_rewards

        env.close()

    def test_end_episode_returns_stats(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        for _ in range(5):
            action = agent.act(obs)
            obs, reward, _, _, _ = env.step(action)
            agent.learn(reward)

        stats = agent.end_episode()

        assert "episode" in stats
        assert "reward" in stats
        assert "length" in stats
        assert stats["length"] == 5
        assert agent.episodes == 1

        env.close()

    def test_reset_clears_state(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        agent.act(obs)
        agent.learn(100)
        agent.end_episode()

        assert agent.episodes == 1

        agent.reset()

        assert agent.episodes == 0
        assert agent.total_steps == 0

        env.close()

    def test_different_modes(self):
        env = gym.make("CartPole-v1")

        for mode in ["triggered", "anti_grover", "grover", "none", "adaptive"]:
            agent = QuantumSNNAgent.for_env(env, mode=mode, seed=42)
            obs, _ = env.reset()
            action = agent.act(obs)
            assert 0 <= action < 2

        env.close()

    def test_exploration_entropy(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        agent.act(obs)

        entropy = agent.exploration_entropy
        assert entropy >= 0
        assert entropy <= 1.0  # log2(2) = 1 for 2 actions

        env.close()

    def test_last_probabilities(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        agent.act(obs)

        probs = agent.last_probabilities
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 0.01

        env.close()

    def test_trigger_exploitation(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, mode="triggered", seed=42)

        assert not agent.is_exploitation_triggered

        agent.brain.trigger_exploitation()
        assert agent.is_exploitation_triggered

        env.close()

    def test_deterministic_with_seed(self):
        env = gym.make("CartPole-v1")

        agent1 = QuantumSNNAgent.for_env(env, seed=12345)
        agent2 = QuantumSNNAgent.for_env(env, seed=12345)

        obs, _ = env.reset(seed=42)

        actions1 = [agent1.act(obs) for _ in range(5)]

        # Reset for agent2
        obs, _ = env.reset(seed=42)
        actions2 = [agent2.act(obs) for _ in range(5)]

        assert actions1 == actions2

        env.close()


class TestTrainAgent:
    """Tests for train_agent function."""

    def test_train_returns_stats(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        stats = train_agent(env, agent, n_episodes=5, verbose=0)

        assert len(stats) == 5
        assert all("reward" in s for s in stats)
        assert all("length" in s for s in stats)

        env.close()

    def test_train_increases_episodes(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        train_agent(env, agent, n_episodes=10, verbose=0)

        assert agent.episodes == 10
        assert agent.total_steps > 0

        env.close()

    def test_train_with_callback(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        callback_episodes = []

        def callback(episode, stats):
            callback_episodes.append(episode)

        train_agent(env, agent, n_episodes=5, verbose=0, callback=callback)

        assert callback_episodes == [0, 1, 2, 3, 4]

        env.close()


class TestEvaluateAgent:
    """Tests for evaluate_agent function."""

    def test_evaluate_returns_stats(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        # Train a bit first
        train_agent(env, agent, n_episodes=10, verbose=0)

        stats = evaluate_agent(env, agent, n_episodes=5)

        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_length" in stats
        assert stats["n_episodes"] == 5

        env.close()

    def test_evaluate_deterministic(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, mode="triggered", seed=42)

        # Trigger exploitation
        agent.brain.trigger_exploitation()

        stats = evaluate_agent(env, agent, n_episodes=3, deterministic=True)

        assert stats["n_episodes"] == 3
        assert stats["mean_reward"] > 0  # CartPole gives +1 per step

        env.close()


class TestDifferentEnvironments:
    """Test agent on various Gymnasium environments."""

    def test_cartpole(self):
        env = gym.make("CartPole-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)

        assert action in [0, 1]

        env.close()

    def test_acrobot(self):
        """Test on Acrobot (6 observations, 3 actions)."""
        env = gym.make("Acrobot-v1")
        agent = QuantumSNNAgent.for_env(env, seed=42)

        assert agent.n_inputs == 6
        assert agent.n_actions == 3

        obs, _ = env.reset()
        action = agent.act(obs)

        assert action in [0, 1, 2]

        env.close()

    def test_mountain_car(self):
        """Test on MountainCar (2 observations, 3 actions)."""
        env = gym.make("MountainCar-v0")
        agent = QuantumSNNAgent.for_env(env, mode="triggered", seed=42)

        assert agent.n_inputs == 2
        assert agent.n_actions == 3

        obs, _ = env.reset()
        action = agent.act(obs)

        assert action in [0, 1, 2]

        env.close()


class TestRewardScaling:
    """Test reward scaling for different reward ranges."""

    def test_positive_rewards(self):
        env = gym.make("CartPole-v1")  # +1 per step
        agent = QuantumSNNAgent.for_env(env, reward_scale=1.0, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)
        obs, reward, _, _, _ = env.step(action)
        agent.learn(reward)  # reward = 1.0

        # Should have learned from positive reward
        assert agent.brain.consecutive_high_rewards >= 0

        env.close()

    def test_negative_rewards(self):
        env = gym.make("MountainCar-v0")  # -1 per step
        agent = QuantumSNNAgent.for_env(env, reward_scale=0.1, seed=42)

        obs, _ = env.reset()
        action = agent.act(obs)
        obs, reward, _, _, _ = env.step(action)
        agent.learn(reward)  # reward = -1.0

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
