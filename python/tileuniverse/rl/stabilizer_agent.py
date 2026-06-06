"""
Stabilizer and Hybrid Quantum SNN Agents for Gymnasium environments.

Sprint 59.0: Provides agents using stabilizer-based quantum neural networks
for efficient large-scale quantum simulation.

Key Features:
- StabilizerSNNAgent: Pure stabilizer network (10K+ neurons, O(n^2) memory)
- HybridQuantumSNNAgent: Stabilizer backbone + full-quantum clusters

Example:
    >>> import gymnasium as gym
    >>> from tileuniverse.rl import StabilizerSNNAgent, HybridQuantumSNNAgent
    >>>
    >>> env = gym.make("CartPole-v1")
    >>> agent = HybridQuantumSNNAgent.for_env(env, n_clusters=2)
    >>>
    >>> # Training loop
    >>> for episode in range(1000):
    ...     obs, _ = env.reset()
    ...     total_reward = 0
    ...     done = False
    ...     while not done:
    ...         action = agent.act(obs)
    ...         obs, reward, terminated, truncated, _ = env.step(action)
    ...         agent.learn(reward)
    ...         total_reward += reward
    ...         done = terminated or truncated
    ...     agent.end_episode()
"""

import numpy as np
from typing import Optional, List, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from tileuniverse import StabilizerSNN, HybridQuantumSNN, FeedbackMode, BridgeGate

# Re-use observation encoder from existing quantum_agent
from .quantum_agent import ObservationEncoder


class StabilizerSNNAgent:
    """
    Gymnasium agent using stabilizer-based quantum neural network.

    Uses the Aaronson-Gottesman stabilizer formalism for efficient simulation
    of quantum neurons. Supports O(n^2) memory scaling, enabling networks with
    10K+ quantum neurons on consumer hardware.

    Attributes:
        brain: The underlying StabilizerSNN network
        encoder: Observation encoder
        episodes: Number of completed episodes
        total_steps: Total environment steps taken
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        hidden_layers: Optional[List[int]] = None,
        connectivity: float = 0.3,
        learning_enabled: bool = True,
        reward_scale: float = 1.0,
        seed: int = 42,
        encoder: Optional[ObservationEncoder] = None,
    ):
        """
        Create a StabilizerSNN agent.

        Args:
            n_inputs: Number of input features (observation size)
            n_actions: Number of discrete actions
            hidden_layers: Hidden layer sizes (default: adaptive based on inputs)
            connectivity: Connection probability between layers (default: 0.3)
            learning_enabled: Whether to learn from rewards
            reward_scale: Scale factor for rewards
            seed: Random seed
            encoder: Optional pre-configured observation encoder
        """
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        self.encoder = encoder
        self.learning_enabled = learning_enabled

        # Auto-size hidden layers if not specified
        if hidden_layers is None:
            hidden_layers = self._auto_hidden_layers(n_inputs, n_actions)

        # Create network
        self.brain = StabilizerSNN(
            n_inputs=n_inputs,
            hidden=hidden_layers,
            n_outputs=n_actions,
            connectivity=connectivity,
            seed=seed,
        )

        # Statistics
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history: List[Dict[str, Any]] = []

    @classmethod
    def for_env(
        cls,
        env: gym.Env,
        hidden_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> "StabilizerSNNAgent":
        """
        Create an agent configured for a specific Gymnasium environment.

        Args:
            env: Gymnasium environment
            hidden_layers: Optional hidden layer sizes
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured StabilizerSNNAgent
        """
        # Create encoder for observation space
        encoder = ObservationEncoder(env.observation_space)
        n_inputs = encoder.n_inputs

        # Get number of actions
        if isinstance(env.action_space, spaces.Discrete):
            n_actions = env.action_space.n
        else:
            raise ValueError(
                f"StabilizerSNNAgent requires Discrete action space, "
                f"got {type(env.action_space)}"
            )

        return cls(
            n_inputs=n_inputs,
            n_actions=n_actions,
            hidden_layers=hidden_layers,
            encoder=encoder,
            **kwargs,
        )

    def _auto_hidden_layers(self, n_inputs: int, n_actions: int) -> List[int]:
        """Automatically size hidden layers."""
        # Stabilizer networks can scale larger than classical SNNs
        hidden_size = max(32, min(128, n_inputs * 2 + n_actions * 4))
        return [hidden_size]

    def act(self, observation: np.ndarray) -> int:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation

        Returns:
            Selected action index
        """
        # Encode observation to spike rates
        if self.encoder is not None:
            inputs = self.encoder.encode(observation)
        else:
            inputs = np.asarray(observation).flatten().tolist()
            inputs = [max(0, min(255, int(x))) for x in inputs]

        # Make decision
        action = self.brain.decide(inputs)

        self.total_steps += 1
        self._episode_length += 1

        return action

    def learn(self, reward: float):
        """
        Provide reward signal for learning.

        Args:
            reward: Reward from environment
        """
        if not self.learning_enabled:
            self._episode_reward += reward
            return

        # Scale reward to [-127, 127] range
        scaled_reward = int(np.clip(reward * self.reward_scale * 100, -127, 127))
        self.brain.learn(scaled_reward)

        self._episode_reward += reward

    def end_episode(self) -> Dict[str, Any]:
        """
        Signal end of episode and get statistics.

        Returns:
            Dict with episode statistics
        """
        stats = {
            "episode": self.episodes,
            "reward": self._episode_reward,
            "length": self._episode_length,
            "tick": self.brain.tick,
            "memory_kb": self.brain.memory_bytes // 1024,
        }

        self._episode_history.append(stats)
        self.episodes += 1

        # Reset episode counters
        self._episode_reward = 0.0
        self._episode_length = 0

        return stats

    def reset(self):
        """Reset agent state."""
        self.brain.reset_output_counts()
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "num_neurons": self.brain.num_neurons,
            "memory_kb": self.brain.memory_bytes // 1024,
            "episode_history": self._episode_history[-100:],
        }


class HybridQuantumSNNAgent:
    """
    Gymnasium agent using hybrid quantum-stabilizer neural network.

    Combines a stabilizer backbone (efficient, scalable) with small full-quantum
    clusters (8-16 qubits) for true quantum advantage via non-Clifford gates.

    Features:
    - Adaptive cluster activation based on input entropy
    - Bidirectional feedback between backbone and clusters
    - Cluster-to-cluster bridges for distributed quantum processing
    - Built-in entropy recording for analysis

    Attributes:
        brain: The underlying HybridQuantumSNN network
        encoder: Observation encoder
        episodes: Number of completed episodes
        total_steps: Total environment steps taken
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        hidden_layers: Optional[List[int]] = None,
        n_clusters: int = 1,
        cluster_size: int = 8,
        feedback_mode: Optional[FeedbackMode] = None,
        adaptive_threshold: Optional[float] = None,
        cluster_topology: str = "none",
        learning_enabled: bool = True,
        reward_scale: float = 1.0,
        ticks_per_decision: int = 10,
        seed: int = 42,
        encoder: Optional[ObservationEncoder] = None,
    ):
        """
        Create a HybridQuantumSNN agent.

        Args:
            n_inputs: Number of input features (observation size)
            n_actions: Number of discrete actions
            hidden_layers: Hidden layer sizes (default: adaptive)
            n_clusters: Number of full-quantum clusters (default: 1)
            cluster_size: Qubits per cluster, 4-16 (default: 8)
            feedback_mode: Feedback mode (default: None)
            adaptive_threshold: Entropy threshold for adaptive activation
            cluster_topology: Cluster connection topology ("none", "chain", "ring", "mesh", "star")
            learning_enabled: Whether to learn from rewards
            reward_scale: Scale factor for rewards
            ticks_per_decision: Number of ticks per decision
            seed: Random seed
            encoder: Optional pre-configured observation encoder
        """
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        self.encoder = encoder
        self.learning_enabled = learning_enabled
        self.ticks_per_decision = ticks_per_decision

        # Auto-size hidden layers if not specified
        if hidden_layers is None:
            hidden_layers = self._auto_hidden_layers(n_inputs, n_actions)

        # Create network
        self.brain = HybridQuantumSNN(
            n_inputs=n_inputs,
            hidden=hidden_layers,
            n_outputs=n_actions,
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            feedback_mode=feedback_mode,
            adaptive_threshold=adaptive_threshold,
            seed=seed,
        )

        # Set up cluster topology
        if cluster_topology == "chain":
            self.brain.chain_clusters()
        elif cluster_topology == "ring":
            self.brain.ring_clusters()
        elif cluster_topology == "mesh":
            self.brain.mesh_clusters()
        elif cluster_topology == "star":
            self.brain.star_clusters(0)

        # Enable recording for analysis
        self.brain.enable_recording(1000)

        # Statistics
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history: List[Dict[str, Any]] = []

    @classmethod
    def for_env(
        cls,
        env: gym.Env,
        hidden_layers: Optional[List[int]] = None,
        n_clusters: int = 1,
        cluster_size: int = 8,
        feedback_mode: Optional[FeedbackMode] = None,
        cluster_topology: str = "none",
        **kwargs,
    ) -> "HybridQuantumSNNAgent":
        """
        Create an agent configured for a specific Gymnasium environment.

        Args:
            env: Gymnasium environment
            hidden_layers: Optional hidden layer sizes
            n_clusters: Number of quantum clusters (default: 1)
            cluster_size: Qubits per cluster (default: 8)
            feedback_mode: Optional feedback mode
            cluster_topology: Topology ("none", "chain", "ring", "mesh", "star")
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured HybridQuantumSNNAgent
        """
        # Create encoder for observation space
        encoder = ObservationEncoder(env.observation_space)
        n_inputs = encoder.n_inputs

        # Get number of actions
        if isinstance(env.action_space, spaces.Discrete):
            n_actions = env.action_space.n
        else:
            raise ValueError(
                f"HybridQuantumSNNAgent requires Discrete action space, "
                f"got {type(env.action_space)}"
            )

        return cls(
            n_inputs=n_inputs,
            n_actions=n_actions,
            hidden_layers=hidden_layers,
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            feedback_mode=feedback_mode,
            cluster_topology=cluster_topology,
            encoder=encoder,
            **kwargs,
        )

    def _auto_hidden_layers(self, n_inputs: int, n_actions: int) -> List[int]:
        """Automatically size hidden layers."""
        hidden_size = max(32, min(64, n_inputs * 2 + n_actions * 4))
        return [hidden_size]

    def act(self, observation: np.ndarray) -> int:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation

        Returns:
            Selected action index
        """
        # Encode observation to spike rates
        if self.encoder is not None:
            inputs = self.encoder.encode(observation)
        else:
            inputs = np.asarray(observation).flatten().tolist()
            inputs = [max(0, min(255, int(x))) for x in inputs]

        # Apply bridges if any
        if self.brain.n_bridges > 0:
            self.brain.apply_bridges()

        # Make decision with integration
        action = self.brain.decide_integrated(inputs, self.ticks_per_decision)

        self.total_steps += 1
        self._episode_length += 1

        return action

    def learn(self, reward: float):
        """
        Provide reward signal for learning.

        Args:
            reward: Reward from environment
        """
        if not self.learning_enabled:
            self._episode_reward += reward
            return

        # Scale reward to [-127, 127] range
        scaled_reward = int(np.clip(reward * self.reward_scale * 100, -127, 127))
        self.brain.learn(scaled_reward)

        self._episode_reward += reward

    def end_episode(self) -> Dict[str, Any]:
        """
        Signal end of episode and get statistics.

        Returns:
            Dict with episode statistics
        """
        # Get quantum metrics
        avg_entropy = 0.0
        if self.brain.is_recording:
            entropy_series = self.brain.average_entropy_series()
            if entropy_series:
                avg_entropy = np.mean(entropy_series)

        stats = {
            "episode": self.episodes,
            "reward": self._episode_reward,
            "length": self._episode_length,
            "tick": self.brain.tick,
            "active_clusters": self.brain.active_clusters,
            "avg_entropy": avg_entropy,
            "memory_kb": self.brain.memory_bytes // 1024,
        }

        self._episode_history.append(stats)
        self.episodes += 1

        # Reset episode counters
        self._episode_reward = 0.0
        self._episode_length = 0

        return stats

    def reset(self):
        """Reset agent state."""
        self.brain.reset_output_counts()
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history = []

    @property
    def cluster_entropy(self) -> List[float]:
        """Get current entropy for each cluster."""
        return [self.brain.cluster_entropy(i) for i in range(self.brain.n_clusters)]

    @property
    def total_z_expectation(self) -> float:
        """Get total Z expectation across clusters."""
        return self.brain.total_z_expectation()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "n_clusters": self.brain.n_clusters,
            "active_clusters": self.brain.active_clusters,
            "n_bridges": self.brain.n_bridges,
            "memory_kb": self.brain.memory_bytes // 1024,
            "cluster_entropy": self.cluster_entropy,
            "episode_history": self._episode_history[-100:],
        }

    def add_bridge(
        self,
        source_cluster: int,
        source_qubit: int,
        target_cluster: int,
        target_qubit: int,
        gate_type: Optional[BridgeGate] = None,
    ):
        """
        Add a bridge between two clusters.

        Args:
            source_cluster: Source cluster ID
            source_qubit: Source qubit index
            target_cluster: Target cluster ID
            target_qubit: Target qubit index
            gate_type: Bridge gate type (default: DISTRIBUTED_CZ)
        """
        self.brain.add_bridge(
            source_cluster, source_qubit, target_cluster, target_qubit, gate_type
        )


def train_stabilizer_agent(
    env: gym.Env,
    agent: "StabilizerSNNAgent | HybridQuantumSNNAgent",
    n_episodes: int = 1000,
    max_steps: Optional[int] = None,
    verbose: int = 1,
    callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Train a stabilizer-based agent on a Gymnasium environment.

    Args:
        env: Gymnasium environment
        agent: StabilizerSNNAgent or HybridQuantumSNNAgent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Verbosity level (0=silent, 1=episode summaries)
        callback: Optional callback(episode, stats) after each episode

    Returns:
        List of episode statistics
    """
    all_stats = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(reward)

            step += 1
            if max_steps is not None and step >= max_steps:
                done = True

        stats = agent.end_episode()
        all_stats.append(stats)

        if verbose >= 1:
            entropy_str = ""
            if "avg_entropy" in stats:
                entropy_str = f", entropy={stats['avg_entropy']:.3f}"
            print(
                f"Episode {episode + 1}/{n_episodes}: "
                f"reward={stats['reward']:.2f}, "
                f"length={stats['length']}{entropy_str}"
            )

        if callback is not None:
            callback(episode, stats)

    return all_stats
