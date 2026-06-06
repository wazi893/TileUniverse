"""
Atari Benchmark Harness for Quantum-SNN Agents

This module provides tools for benchmarking Quantum-SNN agents on Atari games,
comparing triggered interference mode against classical baselines.

Usage:
    python -m tileuniverse.rl.atari_benchmark --game Breakout --episodes 1000 --seeds 5

Requirements:
    pip install gymnasium[atari] ale-py
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing
    # Register ALE environments
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    raise ImportError("Please install gymnasium: pip install gymnasium[atari] ale-py")

# Import our agent
from tileuniverse.rl import QuantumSNNAgent


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for Atari benchmark runs."""

    # Environment
    game: str = "Breakout"
    use_ram: bool = True  # Use RAM (128 bytes) vs pixels
    frameskip: int = 4
    repeat_action_probability: float = 0.0

    # Training
    n_episodes: int = 1000
    max_steps_per_episode: int = 10000

    # Agent
    hidden_layers: List[int] = field(default_factory=lambda: [64])
    ticks_per_decision: int = 20
    trigger_threshold: int = 30
    trigger_count: int = 3

    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    modes: List[str] = field(default_factory=lambda: [
        "triggered",
        "anti_grover",
        "soft_mix",
        "none",  # Pure classical WTA
    ])

    # Baselines
    run_epsilon_greedy: bool = True
    epsilon_values: List[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])

    # Curiosity (optional)
    enable_curiosity: bool = False
    curiosity_type: str = "prediction"  # "novelty", "prediction", "combined"
    curiosity_scale: float = 0.5

    # TD Credit (optional)
    enable_td_credit: bool = False
    td_alpha: float = 0.05
    td_gamma: float = 0.99
    td_scale: float = 50.0

    # Output
    output_dir: str = "benchmark_results"
    save_interval: int = 100  # Save stats every N episodes
    verbose: int = 1


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode: int
    reward: float
    length: int
    time_seconds: float
    exploitation_triggered: bool
    trigger_episode: Optional[int]
    exploration_entropy: float
    mean_q_value: Optional[float] = None


@dataclass
class RunStats:
    """Statistics for a complete training run."""
    config: Dict
    mode: str
    seed: int
    episodes: List[EpisodeStats]
    total_time: float
    final_mean_reward: float
    final_std_reward: float
    episodes_to_threshold: Optional[int]  # Episodes to reach reward threshold


# =============================================================================
# Observation Encoding
# =============================================================================

class AtariRAMEncoder:
    """
    Encodes Atari RAM observations (128 bytes) to spike rates [0-255].

    RAM observations are already in [0, 255], so this is mostly pass-through
    with optional preprocessing.
    """

    def __init__(self, normalize: bool = True, add_deltas: bool = False):
        """
        Args:
            normalize: If True, apply contrast enhancement
            add_deltas: If True, append frame-to-frame deltas (doubles input size)
        """
        self.normalize = normalize
        self.add_deltas = add_deltas
        self.prev_obs = None

    def reset(self):
        """Reset state between episodes."""
        self.prev_obs = None

    def encode(self, obs: np.ndarray) -> List[int]:
        """
        Encode RAM observation to spike rates.

        Args:
            obs: RAM observation, shape (128,), dtype uint8

        Returns:
            List of spike rates [0-255]
        """
        # Ensure correct shape
        obs = np.asarray(obs).flatten()
        assert len(obs) == 128, f"Expected 128 bytes, got {len(obs)}"

        # Base encoding: RAM bytes are already [0, 255]
        encoded = obs.astype(np.float32)

        if self.normalize:
            # Contrast enhancement: stretch around mean
            mean = encoded.mean()
            std = encoded.std() + 1e-8
            encoded = (encoded - mean) / std
            encoded = np.tanh(encoded * 0.5) * 127.5 + 127.5

        # Clip to valid range
        encoded = np.clip(encoded, 0, 255).astype(np.uint8)

        if self.add_deltas:
            if self.prev_obs is not None:
                # Compute deltas, scale to [0, 255]
                deltas = (obs.astype(np.int16) - self.prev_obs.astype(np.int16))
                deltas = ((deltas + 255) / 2).clip(0, 255).astype(np.uint8)
            else:
                deltas = np.full(128, 128, dtype=np.uint8)  # Neutral
            self.prev_obs = obs.copy()
            encoded = np.concatenate([encoded, deltas])

        return encoded.tolist()

    @property
    def n_inputs(self) -> int:
        return 256 if self.add_deltas else 128


# =============================================================================
# Baseline Agents
# =============================================================================

class EpsilonGreedyAgent:
    """
    Simple epsilon-greedy baseline for comparison.
    Uses the same SNN backbone but with classical exploration.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        epsilon: float = 0.1,
        hidden_layers: List[int] = None,
        seed: int = 42,
    ):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)

        # Use QuantumSNN in "none" mode (pure classical)
        self.agent = QuantumSNNAgent(
            n_inputs=n_inputs,
            n_actions=n_actions,
            hidden_layers=hidden_layers or [64],
            mode="none",
            seed=seed,
        )

    def act(self, observation: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_actions)
        return self.agent.act(observation)

    def learn(self, reward: float, observation: np.ndarray = None, **kwargs):
        self.agent.learn(reward, observation=observation, **kwargs)

    def end_episode(self) -> Dict:
        return self.agent.end_episode()

    def reset(self):
        self.agent.reset()


class RandomAgent:
    """Uniform random baseline."""

    def __init__(self, n_actions: int, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self.episode_reward = 0
        self.episode_length = 0

    def act(self, observation: np.ndarray) -> int:
        return self.rng.randint(self.n_actions)

    def learn(self, reward: float, **kwargs):
        self.episode_reward += reward
        self.episode_length += 1

    def end_episode(self) -> Dict:
        stats = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
        }
        self.episode_reward = 0
        self.episode_length = 0
        return stats

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0


# =============================================================================
# Environment Setup
# =============================================================================

def make_atari_env(game: str, use_ram: bool = True, seed: int = 42) -> gym.Env:
    """
    Create an Atari environment with appropriate settings.

    Args:
        game: Game name (e.g., "Breakout", "Pong", "SpaceInvaders")
        use_ram: If True, use RAM observations (128 bytes)
        seed: Random seed

    Returns:
        Gymnasium environment
    """
    # Construct environment ID
    env_id = f"ALE/{game}-v5"

    # Create environment with appropriate observation type
    env = gym.make(
        env_id,
        obs_type="ram" if use_ram else "rgb",
        frameskip=4,
        repeat_action_probability=0.0,
        render_mode=None,
    )

    # Set seed
    env.reset(seed=seed)

    return env


def get_reward_threshold(game: str) -> float:
    """
    Get the 'solved' reward threshold for common Atari games.
    Used to measure sample efficiency.
    """
    thresholds = {
        "Breakout": 30.0,
        "Pong": 18.0,
        "SpaceInvaders": 500.0,
        "Qbert": 5000.0,
        "Seaquest": 1000.0,
        "BeamRider": 2000.0,
        "Enduro": 300.0,
        "Asteroids": 1000.0,
        "MsPacman": 2000.0,
        "Frostbite": 300.0,
        "MontezumaRevenge": 100.0,  # Very hard
        "Venture": 200.0,
        "PrivateEye": 100.0,
    }
    return thresholds.get(game, 100.0)


# =============================================================================
# Training Loop
# =============================================================================

def train_agent_atari(
    env: gym.Env,
    agent,
    encoder: AtariRAMEncoder,
    config: BenchmarkConfig,
    seed: int,
    mode: str,
    progress_callback: Optional[Callable] = None,
) -> RunStats:
    """
    Train an agent on Atari and collect statistics.

    Args:
        env: Atari environment
        agent: Agent to train (QuantumSNNAgent or baseline)
        encoder: RAM encoder
        config: Benchmark configuration
        seed: Random seed for this run
        mode: Interference mode name (for logging)
        progress_callback: Optional callback(episode, stats) for progress updates

    Returns:
        RunStats with full training statistics
    """
    episodes: List[EpisodeStats] = []
    reward_threshold = get_reward_threshold(config.game)
    episodes_to_threshold = None
    trigger_episode = None

    # Running statistics
    recent_rewards = deque(maxlen=100)

    start_time = time.time()

    for episode in range(config.n_episodes):
        episode_start = time.time()

        # Reset
        obs, info = env.reset(seed=seed + episode)
        encoder.reset()
        if hasattr(agent, 'reset'):
            agent.reset()

        episode_reward = 0.0
        episode_length = 0

        for step in range(config.max_steps_per_episode):
            # Encode observation
            spike_rates = encoder.encode(obs)

            # Select action
            action = agent.act(spike_rates)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Scale reward for agent (Atari rewards vary widely)
            scaled_reward = np.clip(reward, -1.0, 1.0)  # Clip to [-1, 1]

            # Learn
            if hasattr(agent, 'learn_with_td') and config.enable_td_credit:
                next_spike_rates = encoder.encode(next_obs) if not done else spike_rates
                agent.learn_with_td(scaled_reward, next_spike_rates, done)
            elif hasattr(agent, 'learn_with_curiosity') and config.enable_curiosity:
                agent.learn_with_curiosity(scaled_reward, spike_rates)
            else:
                agent.learn(scaled_reward, terminated=done, observation=obs)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        # End episode
        episode_stats_dict = agent.end_episode() if hasattr(agent, 'end_episode') else {}
        episode_time = time.time() - episode_start

        # Check if exploitation triggered (for QuantumSNN)
        exploitation_triggered = False
        if hasattr(agent, 'is_exploitation_triggered'):
            exploitation_triggered = agent.is_exploitation_triggered
            if exploitation_triggered and trigger_episode is None:
                trigger_episode = episode
        elif hasattr(agent, 'agent') and hasattr(agent.agent, 'is_exploitation_triggered'):
            exploitation_triggered = agent.agent.is_exploitation_triggered
            if exploitation_triggered and trigger_episode is None:
                trigger_episode = episode

        # Get exploration entropy
        entropy = 0.0
        if hasattr(agent, 'exploration_entropy'):
            try:
                # Could be property or method
                ent = agent.exploration_entropy
                entropy = ent() if callable(ent) else ent
            except:
                entropy = 0.0
        elif hasattr(agent, 'agent') and hasattr(agent.agent, 'exploration_entropy'):
            try:
                ent = agent.agent.exploration_entropy
                entropy = ent() if callable(ent) else ent
            except:
                entropy = 0.0

        # Record episode
        ep_stats = EpisodeStats(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            time_seconds=episode_time,
            exploitation_triggered=exploitation_triggered,
            trigger_episode=trigger_episode,
            exploration_entropy=entropy,
        )
        episodes.append(ep_stats)
        recent_rewards.append(episode_reward)

        # Check threshold
        if episodes_to_threshold is None and len(recent_rewards) >= 100:
            mean_100 = np.mean(recent_rewards)
            if mean_100 >= reward_threshold:
                episodes_to_threshold = episode

        # Progress callback
        if progress_callback:
            progress_callback(episode, ep_stats)

        # Verbose output
        if config.verbose >= 1 and (episode + 1) % 10 == 0:
            mean_recent = np.mean(recent_rewards) if recent_rewards else 0
            print(f"  Episode {episode+1}/{config.n_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Mean(100): {mean_recent:.1f} | "
                  f"Triggered: {exploitation_triggered}")

    total_time = time.time() - start_time

    # Compute final statistics
    final_rewards = [e.reward for e in episodes[-100:]]

    return RunStats(
        config=asdict(config),
        mode=mode,
        seed=seed,
        episodes=episodes,
        total_time=total_time,
        final_mean_reward=np.mean(final_rewards),
        final_std_reward=np.std(final_rewards),
        episodes_to_threshold=episodes_to_threshold,
    )


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> Dict[str, List[RunStats]]:
    """
    Run the complete Atari benchmark comparing different modes and baselines.

    Args:
        config: Benchmark configuration

    Returns:
        Dictionary mapping mode/baseline name to list of RunStats (one per seed)
    """
    results: Dict[str, List[RunStats]] = {}

    # Create output directory
    output_path = Path(config.output_dir) / config.game
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment to get action space
    test_env = make_atari_env(config.game, use_ram=config.use_ram, seed=0)
    n_actions = test_env.action_space.n
    test_env.close()

    # Create encoder
    encoder = AtariRAMEncoder(normalize=True, add_deltas=False)
    n_inputs = encoder.n_inputs

    print(f"\n{'='*60}")
    print(f"Atari Benchmark: {config.game}")
    print(f"{'='*60}")
    print(f"Actions: {n_actions} | Inputs: {n_inputs}")
    print(f"Episodes: {config.n_episodes} | Seeds: {len(config.seeds)}")
    print(f"Modes: {config.modes}")
    print(f"{'='*60}\n")

    # Run Quantum-SNN modes
    for mode in config.modes:
        print(f"\n--- Mode: {mode} ---")
        mode_results = []

        for seed in config.seeds:
            print(f"\n  Seed {seed}:")

            # Create environment
            env = make_atari_env(config.game, use_ram=config.use_ram, seed=seed)

            # Create agent
            agent = QuantumSNNAgent(
                n_inputs=n_inputs,
                n_actions=n_actions,
                hidden_layers=config.hidden_layers,
                mode=mode,
                ticks_per_decision=config.ticks_per_decision,
                trigger_threshold=config.trigger_threshold,
                trigger_count=config.trigger_count,
                seed=seed,
            )

            # Enable optional features
            if config.enable_curiosity:
                if config.curiosity_type == "novelty":
                    agent.enable_novelty_curiosity(config.curiosity_scale, 0.001)
                elif config.curiosity_type == "prediction":
                    agent.enable_prediction_curiosity(0.01, config.curiosity_scale)
                elif config.curiosity_type == "combined":
                    agent.enable_combined_curiosity(config.curiosity_scale)

            if config.enable_td_credit:
                agent.enable_td_credit(config.td_alpha, config.td_gamma, config.td_scale)

            # Train
            stats = train_agent_atari(env, agent, encoder, config, seed, mode)
            mode_results.append(stats)

            env.close()

            print(f"    Final Mean: {stats.final_mean_reward:.1f} +/- {stats.final_std_reward:.1f}")
            if stats.episodes_to_threshold:
                print(f"    Solved at episode: {stats.episodes_to_threshold}")

        results[mode] = mode_results

        # Save intermediate results
        save_results(results, output_path / "results_intermediate.json")

    # Run epsilon-greedy baselines
    if config.run_epsilon_greedy:
        for epsilon in config.epsilon_values:
            mode_name = f"epsilon_{epsilon}"
            print(f"\n--- Baseline: Epsilon-Greedy (eps={epsilon}) ---")
            mode_results = []

            for seed in config.seeds:
                print(f"\n  Seed {seed}:")

                env = make_atari_env(config.game, use_ram=config.use_ram, seed=seed)

                agent = EpsilonGreedyAgent(
                    n_inputs=n_inputs,
                    n_actions=n_actions,
                    epsilon=epsilon,
                    hidden_layers=config.hidden_layers,
                    seed=seed,
                )

                stats = train_agent_atari(env, agent, encoder, config, seed, mode_name)
                mode_results.append(stats)

                env.close()

                print(f"    Final Mean: {stats.final_mean_reward:.1f} +/- {stats.final_std_reward:.1f}")

            results[mode_name] = mode_results

    # Run random baseline
    print(f"\n--- Baseline: Random ---")
    random_results = []
    for seed in config.seeds:
        env = make_atari_env(config.game, use_ram=config.use_ram, seed=seed)
        agent = RandomAgent(n_actions=n_actions, seed=seed)
        stats = train_agent_atari(env, agent, encoder, config, seed, "random")
        random_results.append(stats)
        env.close()
    results["random"] = random_results

    # Save final results
    save_results(results, output_path / "results_final.json")

    # Generate summary
    print_summary(results, config)

    return results


# =============================================================================
# Results Handling
# =============================================================================

def save_results(results: Dict[str, List[RunStats]], path: Path):
    """Save results to JSON file."""

    def stats_to_dict(stats: RunStats) -> Dict:
        return {
            "mode": stats.mode,
            "seed": stats.seed,
            "total_time": stats.total_time,
            "final_mean_reward": stats.final_mean_reward,
            "final_std_reward": stats.final_std_reward,
            "episodes_to_threshold": stats.episodes_to_threshold,
            "episodes": [
                {
                    "episode": e.episode,
                    "reward": e.reward,
                    "length": e.length,
                    "time_seconds": e.time_seconds,
                    "exploitation_triggered": e.exploitation_triggered,
                    "exploration_entropy": e.exploration_entropy,
                }
                for e in stats.episodes
            ]
        }

    data = {
        mode: [stats_to_dict(s) for s in stats_list]
        for mode, stats_list in results.items()
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {path}")


def print_summary(results: Dict[str, List[RunStats]], config: BenchmarkConfig):
    """Print summary table comparing all modes."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY: {config.game}")
    print(f"{'='*80}")
    print(f"{'Mode':<20} {'Mean Reward':<15} {'Std':<10} {'Solved@':<12} {'Time (min)':<10}")
    print("-" * 80)

    # Sort by mean reward
    mode_scores = []
    for mode, stats_list in results.items():
        mean_rewards = [s.final_mean_reward for s in stats_list]
        mean = np.mean(mean_rewards)
        std = np.std(mean_rewards)

        solved_episodes = [s.episodes_to_threshold for s in stats_list if s.episodes_to_threshold]
        solved_str = f"{np.mean(solved_episodes):.0f}" if solved_episodes else "N/A"

        total_time = sum(s.total_time for s in stats_list) / 60  # minutes

        mode_scores.append((mode, mean, std, solved_str, total_time))

    # Sort by mean reward descending
    mode_scores.sort(key=lambda x: x[1], reverse=True)

    for mode, mean, std, solved, time_min in mode_scores:
        print(f"{mode:<20} {mean:>10.1f}     {std:>6.1f}    {solved:>8}     {time_min:>6.1f}")

    print("=" * 80)

    # Highlight best
    best_mode = mode_scores[0][0]
    best_reward = mode_scores[0][1]

    # Compare triggered vs epsilon-greedy
    triggered_score = next((s for s in mode_scores if s[0] == "triggered"), None)
    eps_scores = [s for s in mode_scores if s[0].startswith("epsilon_")]

    if triggered_score and eps_scores:
        best_eps = max(eps_scores, key=lambda x: x[1])
        improvement = ((triggered_score[1] - best_eps[1]) / best_eps[1]) * 100
        print(f"\nTriggered vs Best Epsilon-Greedy: {improvement:+.1f}%")

    print(f"\nBest Mode: {best_mode} (Mean Reward: {best_reward:.1f})")


# =============================================================================
# Visualization
# =============================================================================

def plot_learning_curves(results: Dict[str, List[RunStats]], output_path: Path):
    """
    Generate learning curve plots.

    Requires matplotlib: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color scheme
    colors = {
        "triggered": "#2ecc71",      # Green
        "anti_grover": "#3498db",    # Blue
        "soft_mix": "#9b59b6",       # Purple
        "none": "#95a5a6",           # Gray
        "epsilon_0.1": "#e74c3c",    # Red
        "epsilon_0.05": "#e67e22",   # Orange
        "epsilon_0.01": "#f1c40f",   # Yellow
        "random": "#7f8c8d",         # Dark gray
    }

    # Plot 1: Raw learning curves
    ax1 = axes[0]
    for mode, stats_list in results.items():
        # Average across seeds
        all_rewards = []
        for stats in stats_list:
            rewards = [e.reward for e in stats.episodes]
            all_rewards.append(rewards)

        # Pad to same length
        max_len = max(len(r) for r in all_rewards)
        padded = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]

        mean_rewards = np.mean(padded, axis=0)
        std_rewards = np.std(padded, axis=0)

        # Smooth with rolling mean
        window = 50
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')

        color = colors.get(mode, "#000000")
        ax1.plot(smoothed, label=mode, color=color, linewidth=2)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Learning Curves (Smoothed)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    ax2 = axes[1]
    modes = list(results.keys())
    means = []
    stds = []

    for mode in modes:
        stats_list = results[mode]
        final_rewards = [s.final_mean_reward for s in stats_list]
        means.append(np.mean(final_rewards))
        stds.append(np.std(final_rewards))

    # Sort by mean
    sorted_idx = np.argsort(means)[::-1]
    modes = [modes[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    bar_colors = [colors.get(m, "#000000") for m in modes]

    bars = ax2.barh(modes, means, xerr=stds, color=bar_colors, capsize=5)
    ax2.set_xlabel("Mean Reward (Last 100 Episodes)")
    ax2.set_title("Final Performance Comparison")
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path / "learning_curves.png", dpi=150)
    plt.close()

    print(f"Learning curves saved to: {output_path / 'learning_curves.png'}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Atari Benchmark for Quantum-SNN")

    parser.add_argument("--game", type=str, default="Breakout",
                        help="Atari game name (e.g., Breakout, Pong, SpaceInvaders)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for multiple runs")
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["triggered", "anti_grover", "soft_mix", "none"],
                        help="Quantum-SNN interference modes to test")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                        help="Hidden layer sizes")
    parser.add_argument("--ticks", type=int, default=20,
                        help="SNN ticks per decision")
    parser.add_argument("--curiosity", action="store_true",
                        help="Enable curiosity-driven exploration")
    parser.add_argument("--td-credit", action="store_true",
                        help="Enable TD-style credit assignment")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip epsilon-greedy baselines")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=normal, 2=debug)")

    args = parser.parse_args()

    config = BenchmarkConfig(
        game=args.game,
        n_episodes=args.episodes,
        seeds=args.seeds,
        modes=args.modes,
        hidden_layers=args.hidden,
        ticks_per_decision=args.ticks,
        enable_curiosity=args.curiosity,
        enable_td_credit=args.td_credit,
        run_epsilon_greedy=not args.no_baselines,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Run benchmark
    results = run_benchmark(config)

    # Generate plots
    output_path = Path(config.output_dir) / config.game
    plot_learning_curves(results, output_path)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
