#!/usr/bin/env python3
"""
Monte Carlo Simulation with TileUniverse

Demonstrates running 100 parallel worlds with parameter variations
to gather statistics - a preview of future EDA/design exploration features.

This is the "Synopsys moment" - showing that TileUniverse can do
speculative multi-world exploration that traditional tools cannot.
"""

import numpy as np
import time
from tileuniverse.rl import ParallelGridworld


def monte_carlo_navigation(
    n_worlds: int = 100,
    n_trials: int = 10,
    grid_size: int = 32,
    wall_densities: list = None,
    max_steps: int = 100,
):
    """
    Run Monte Carlo simulation of agent navigation across many worlds.

    Vary wall density and measure:
    - Success rate (reaching goal)
    - Average steps to goal
    - Average reward

    Args:
        n_worlds: Number of parallel worlds per trial
        n_trials: Number of Monte Carlo trials
        grid_size: Grid dimension (grid_size x grid_size)
        wall_densities: List of wall densities to test
        max_steps: Max steps per episode
    """
    if wall_densities is None:
        wall_densities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 70)
    print("  TileUniverse Monte Carlo Navigation Simulation")
    print("=" * 70)
    print()
    print(f"  Worlds per trial: {n_worlds}")
    print(f"  Trials per density: {n_trials}")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Wall densities: {wall_densities}")
    print()

    results = {}

    for density in wall_densities:
        print(f"Testing wall_density={density:.2f}...")

        density_results = {
            "success_rate": [],
            "avg_steps": [],
            "avg_reward": [],
        }

        for trial in range(n_trials):
            # Create environment with this wall density
            env = ParallelGridworld(
                worlds=n_worlds,
                size=(grid_size, grid_size),
                wall_density=density,
                max_steps=max_steps,
                seed=42 + trial,
            )
            env.reset()

            # Run episodes with simple heuristic policy
            # (move toward goal when possible)
            successes = 0
            total_steps = 0
            total_reward = 0

            for _ in range(max_steps):
                # Heuristic: move toward goal
                actions = compute_heuristic_actions(env)
                obs, rewards, dones, info = env.step(actions)

                total_reward += rewards.sum()
                total_steps += (~dones).sum()

                # Count successes
                successes += info["goal_reached"].sum()

                # All done?
                if dones.all():
                    break

            density_results["success_rate"].append(successes / n_worlds)
            density_results["avg_steps"].append(total_steps / n_worlds)
            density_results["avg_reward"].append(total_reward / n_worlds)

        # Aggregate results
        results[density] = {
            "success_rate": np.mean(density_results["success_rate"]),
            "success_std": np.std(density_results["success_rate"]),
            "avg_steps": np.mean(density_results["avg_steps"]),
            "avg_reward": np.mean(density_results["avg_reward"]),
        }

    return results


def compute_heuristic_actions(env):
    """
    Simple heuristic: move toward goal.

    Actions: 0=noop, 1=up, 2=down, 3=left, 4=right
    """
    actions = np.zeros(env.worlds, dtype=np.int32)

    for w in range(env.worlds):
        ax, ay = env.agent_positions[w]
        gx, gy = env.goal_positions[w]

        dx = gx - ax
        dy = gy - ay

        # Prioritize larger delta
        if abs(dx) > abs(dy):
            if dx > 0:
                actions[w] = 4  # right
            elif dx < 0:
                actions[w] = 3  # left
        else:
            if dy > 0:
                actions[w] = 2  # down
            elif dy < 0:
                actions[w] = 1  # up

        # Add some randomness to avoid getting stuck
        if np.random.random() < 0.1:
            actions[w] = np.random.randint(1, 5)

    return actions


def print_results(results):
    """Pretty-print Monte Carlo results."""
    print()
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print()
    print(f"{'Density':>10} {'Success%':>12} {'Avg Steps':>12} {'Avg Reward':>12}")
    print("-" * 50)

    for density in sorted(results.keys()):
        r = results[density]
        print(f"{density:>10.2f} "
              f"{r['success_rate']*100:>11.1f}% "
              f"{r['avg_steps']:>12.1f} "
              f"{r['avg_reward']:>12.2f}")

    print()


def run_timing_benchmark(n_worlds: int = 100, grid_size: int = 32, steps: int = 1000):
    """
    Benchmark Monte Carlo throughput.
    """
    print("=" * 70)
    print("  Monte Carlo Throughput Benchmark")
    print("=" * 70)
    print()

    env = ParallelGridworld(
        worlds=n_worlds,
        size=(grid_size, grid_size),
        seed=42,
    )
    env.reset()

    start = time.time()
    for _ in range(steps):
        actions = np.random.randint(0, 5, size=n_worlds)
        obs, rewards, dones, info = env.step(actions)

        # Auto-reset done environments
        if dones.any():
            env.reset(world_mask=dones)

    elapsed = time.time() - start
    total_steps = steps * n_worlds
    throughput = total_steps / elapsed

    print(f"  Worlds: {n_worlds}")
    print(f"  Steps per world: {steps}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {throughput/1e6:.2f}M steps/sec")
    print()

    return throughput


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo Navigation Demo")
    parser.add_argument("--worlds", type=int, default=100,
                        help="Parallel worlds (default: 100)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Monte Carlo trials per density (default: 10)")
    parser.add_argument("--size", type=int, default=32,
                        help="Grid size (default: 32)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark only")

    args = parser.parse_args()

    if args.benchmark:
        run_timing_benchmark(args.worlds, args.size, steps=5000)
    else:
        # Run Monte Carlo simulation
        results = monte_carlo_navigation(
            n_worlds=args.worlds,
            n_trials=args.trials,
            grid_size=args.size,
        )
        print_results(results)

        # Also run benchmark
        print()
        run_timing_benchmark(args.worlds, args.size, steps=1000)

    print("Done!")
