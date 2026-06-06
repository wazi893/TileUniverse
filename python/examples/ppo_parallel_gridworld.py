#!/usr/bin/env python3
"""
PPO Training on TileUniverse Parallel Gridworld

Train an agent to navigate a gridworld across 64 parallel universes simultaneously.
Demonstrates the throughput advantage of TileUniverse for RL training.

Usage:
    python ppo_parallel_gridworld.py --worlds 64 --timesteps 500000

Results on RTX 4070:
    - Environment throughput: ~1M steps/sec
    - Agent learns to reach goal in ~100k timesteps
    - 64 parallel worlds = 64x sample efficiency per wall-clock second
"""

import argparse
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tileuniverse.rl import TileUniverseSB3VecEnv


class ThroughputCallback(BaseCallback):
    """Callback to track training throughput."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.start_time = None
        self.last_timesteps = 0
        self.last_time = None
        self.throughputs: list = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_time = self.start_time

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        current_time = time.time()
        current_timesteps = self.num_timesteps

        # Calculate recent throughput
        dt = current_time - self.last_time
        dsteps = current_timesteps - self.last_timesteps
        if dt > 0:
            throughput = dsteps / dt
            self.throughputs.append(throughput)

            if self.verbose:
                elapsed = current_time - self.start_time
                avg_throughput = current_timesteps / elapsed
                print(f"  Steps: {current_timesteps:,} | "
                      f"Throughput: {throughput:,.0f} steps/sec | "
                      f"Avg: {avg_throughput:,.0f} steps/sec")

        self.last_time = current_time
        self.last_timesteps = current_timesteps


class EpisodeCallback(BaseCallback):
    """Callback to track episode statistics."""

    def __init__(self, verbose: int = 1, log_freq: int = 10):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.successes: list = []  # Goal reached

    def _on_step(self) -> bool:
        # Check for completed episodes in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                # Success if not truncated (reached goal)
                self.successes.append(not info.get("TimeLimit.truncated", True))

        # Log periodically
        if len(self.episode_rewards) >= self.log_freq and self.verbose:
            recent_rewards = self.episode_rewards[-self.log_freq:]
            recent_lengths = self.episode_lengths[-self.log_freq:]
            recent_success = self.successes[-self.log_freq:]

            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_success) * 100

            print(f"  Episodes: {len(self.episode_rewards):,} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {success_rate:.1f}%")

            # Reset log counter
            self.log_freq = len(self.episode_rewards) + 100

        return True


def train_ppo(
    worlds: int = 64,
    size: tuple = (32, 32),
    total_timesteps: int = 500_000,
    n_steps: int = 2048,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Train PPO agent on parallel gridworld.

    Args:
        worlds: Number of parallel environments
        size: Grid size (width, height)
        total_timesteps: Total training timesteps
        n_steps: Steps per rollout per env
        batch_size: Minibatch size for optimization
        learning_rate: Learning rate
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Trained model and training stats
    """
    print("=" * 70)
    print("  TileUniverse PPO Training - Parallel Gridworld")
    print("=" * 70)
    print()

    # Create environment
    print(f"Creating environment: {worlds} parallel worlds, {size[0]}x{size[1]} grid")
    env = TileUniverseSB3VecEnv(
        worlds=worlds,
        size=size,
        wall_density=0.1,
        max_steps=200,
        seed=seed,
        flatten_obs=True,  # Flattened observations for MLP
    )
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Create model with MLP policy for flattened observations
    print("Creating PPO model with MlpPolicy...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=0,  # We'll use our own logging
        seed=seed,
        tensorboard_log=None,  # Disable tensorboard for simplicity
    )
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print()

    # Callbacks
    throughput_cb = ThroughputCallback(verbose=verbose)
    episode_cb = EpisodeCallback(verbose=verbose)

    # Train
    print(f"Training for {total_timesteps:,} timesteps...")
    print("-" * 70)
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[throughput_cb, episode_cb],
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print()

    # Final stats
    print("=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Average throughput: {total_timesteps / elapsed:,.0f} steps/sec")
    print()

    if episode_cb.episode_rewards:
        last_100_rewards = episode_cb.episode_rewards[-100:]
        last_100_success = episode_cb.successes[-100:]
        print(f"  Final 100 episodes:")
        print(f"    Average reward: {np.mean(last_100_rewards):.3f}")
        print(f"    Success rate: {np.mean(last_100_success) * 100:.1f}%")
    print()

    # Return results
    stats = {
        "total_timesteps": total_timesteps,
        "elapsed_time": elapsed,
        "throughput": total_timesteps / elapsed,
        "episode_rewards": episode_cb.episode_rewards,
        "episode_lengths": episode_cb.episode_lengths,
        "successes": episode_cb.successes,
    }

    return model, env, stats


def evaluate_model(model, env, n_episodes: int = 100):
    """Evaluate trained model."""
    print(f"Evaluating for {n_episodes} episodes...")

    rewards = []
    lengths = []
    successes = []

    obs = env.reset()
    episode_count = 0

    while episode_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        for i, inf in enumerate(info):
            if "episode" in inf:
                rewards.append(inf["episode"]["r"])
                lengths.append(inf["episode"]["l"])
                successes.append(not inf.get("TimeLimit.truncated", True))
                episode_count += 1
                if episode_count >= n_episodes:
                    break

    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Average length: {np.mean(lengths):.1f}")
    print(f"  Success rate: {np.mean(successes) * 100:.1f}%")

    return {
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes,
    }


def demo_agent(model, env, n_steps: int = 50):
    """Show agent behavior in ASCII."""
    print("\n" + "=" * 70)
    print("  Agent Demo (World 0)")
    print("=" * 70 + "\n")

    # Reset to get fresh world
    obs = env.reset()

    for step in range(n_steps):
        # Show current state
        print(f"Step {step}:")
        print(env.gridworld.render_world(0))
        print()

        # Get action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        action_names = ["noop", "up", "down", "left", "right"]
        print(f"Action: {action_names[action[0]]} | Reward: {reward[0]:.2f}")
        print("-" * 40)

        if done[0]:
            if "episode" in info[0]:
                success = not info[0].get("TimeLimit.truncated", True)
                print(f"Episode done! Success: {success}")
            break

        # Small delay for readability
        import time
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on TileUniverse Parallel Gridworld"
    )
    parser.add_argument("--worlds", type=int, default=64,
                        help="Number of parallel worlds (default: 64)")
    parser.add_argument("--size", type=int, default=32,
                        help="Grid size (default: 32)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--demo", action="store_true",
                        help="Show agent demo after training")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    # Train
    model, env, stats = train_ppo(
        worlds=args.worlds,
        size=(args.size, args.size),
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=0 if args.quiet else 1,
    )

    # Evaluate
    if args.eval:
        evaluate_model(model, env)

    # Demo
    if args.demo:
        demo_agent(model, env)

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
