"""
TileUniverse RL Module

GPU-accelerated parallel environments for reinforcement learning.
Train agents across 100+ parallel universes at 40B+ evals/sec.

Example (Vectorized):
    >>> from tileuniverse.rl import TileUniverseVecEnv
    >>> env = TileUniverseVecEnv(worlds=64, size=(32, 32))
    >>> obs, info = env.reset()
    >>> obs, rewards, terminated, truncated, info = env.step(actions)

Example (Single):
    >>> from tileuniverse.rl import TileUniverseEnv
    >>> env = TileUniverseEnv(size=(32, 32))
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(action)

Example (QuantumSNN Agent - Triggered mode for sparse rewards):
    >>> import gymnasium as gym
    >>> from tileuniverse.rl import QuantumSNNAgent, train_agent
    >>> env = gym.make("MountainCar-v0")
    >>> agent = QuantumSNNAgent.for_env(env, mode="triggered")
    >>> stats = train_agent(env, agent, n_episodes=500)

Example (StabilizerSNN Agent - 10K+ neuron scale):
    >>> import gymnasium as gym
    >>> from tileuniverse.rl import StabilizerSNNAgent
    >>> env = gym.make("CartPole-v1")
    >>> agent = StabilizerSNNAgent.for_env(env)
    >>> for episode in range(100):
    ...     obs, _ = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = agent.act(obs)
    ...         obs, reward, term, trunc, _ = env.step(action)
    ...         agent.learn(reward)
    ...         done = term or trunc
    ...     agent.end_episode()

Example (HybridQuantumSNN Agent - quantum clusters in stabilizer backbone):
    >>> import gymnasium as gym
    >>> from tileuniverse.rl import HybridQuantumSNNAgent
    >>> env = gym.make("CartPole-v1")
    >>> agent = HybridQuantumSNNAgent.for_env(env, n_clusters=2, cluster_topology="ring")
    >>> # Training loop same as above
"""

from .gridworld import ParallelGridworld, EMPTY, WALL, GOAL, AGENT
from .vec_env import TileUniverseVecEnv, TileUniverseEnv
from .trajectory import (
    Trajectory,
    TrajectoryRecorder,
    TrajectoryPlayer,
    record_episode,
    compare_trajectories,
)
from .quantum_agent import (
    QuantumSNNAgent,
    ObservationEncoder,
    train_agent,
    evaluate_agent,
    compare_modes,
)
from .stabilizer_agent import (
    StabilizerSNNAgent,
    HybridQuantumSNNAgent,
    train_stabilizer_agent,
)

# Atari benchmark (optional import - requires gymnasium[atari])
try:
    from .atari_benchmark import (
        BenchmarkConfig,
        AtariRAMEncoder,
        EpsilonGreedyAgent,
        RandomAgent,
        run_benchmark,
        make_atari_env,
        plot_learning_curves,
    )
    _HAS_ATARI = True
except ImportError:
    _HAS_ATARI = False
    BenchmarkConfig = None  # type: ignore
    AtariRAMEncoder = None  # type: ignore
    run_benchmark = None  # type: ignore

# SB3 wrapper (optional import - requires stable-baselines3)
try:
    from .sb3_wrapper import TileUniverseSB3VecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    TileUniverseSB3VecEnv = None  # type: ignore

__all__ = [
    # Environments
    "ParallelGridworld",
    "TileUniverseVecEnv",
    "TileUniverseEnv",
    "TileUniverseSB3VecEnv",
    # Trajectory recording
    "Trajectory",
    "TrajectoryRecorder",
    "TrajectoryPlayer",
    "record_episode",
    "compare_trajectories",
    # Grid constants
    "EMPTY",
    "WALL",
    "GOAL",
    "AGENT",
    # QuantumSNN Agent
    "QuantumSNNAgent",
    "ObservationEncoder",
    "train_agent",
    "evaluate_agent",
    "compare_modes",
    # StabilizerSNN Agents (Sprint 59.0)
    "StabilizerSNNAgent",
    "HybridQuantumSNNAgent",
    "train_stabilizer_agent",
    # Atari Benchmark (Sprint 72.0)
    "BenchmarkConfig",
    "AtariRAMEncoder",
    "EpsilonGreedyAgent",
    "RandomAgent",
    "run_benchmark",
    "make_atari_env",
    "plot_learning_curves",
]
