"""
Benchmark: Compare ALL Interference Modes on Sparse Reward Tasks

Tests all available quantum interference modes:
- none: Pure classical SNN (no quantum interference)
- triggered: Anti-Grover exploration until high rewards, then Grover exploitation
- anti_grover: Always amplify diversity (exploration)
- grover: Always amplify dominant action (exploitation)
- soft_mix: Blend of exploration/exploitation
- adaptive: Dynamically adjust based on reward history
"""

import numpy as np
import time
from typing import List, Dict
import gymnasium as gym
from gymnasium import spaces

from tileuniverse.rl import QuantumSNNAgent, train_agent


# ============================================================
# SPARSE REWARD ENVIRONMENTS
# ============================================================

class HiddenGoalMaze:
    """Maze where goal position is NOT in observation."""

    def __init__(self, size=6, max_steps=100, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = [0, 0]
        self.goal_pos = [
            self.rng.integers(self.size // 2, self.size),
            self.rng.integers(self.size // 2, self.size),
        ]
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {"success": True}
        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}
        return self._get_obs(), 0.0, False, False, {}


class DeceptiveMaze:
    """Maze with trap (easy small reward) vs optimal (hard big reward)."""

    def __init__(self, size=10, max_steps=200, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.start_pos = [size // 2, size // 2]
        self.trap_pos = [size // 2 + 1, size // 2]
        self.goal_pos = [size - 1, size - 1]
        self.agent_pos = None
        self.steps = 0
        self.visited_trap = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = list(self.start_pos)
        self.steps = 0
        self.visited_trap = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        if self.agent_pos == self.trap_pos and not self.visited_trap:
            self.visited_trap = True
            return self._get_obs(), 20.0, True, False, {"success": True, "optimal": False}
        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {"success": True, "optimal": True}
        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False, "optimal": False}
        return self._get_obs(), 0.0, False, False, {}


class LongHorizonSearch:
    """Must visit N checkpoints in sequence."""

    def __init__(self, size=8, n_checkpoints=3, max_steps=200, seed=None):
        self.size = size
        self.n_checkpoints = n_checkpoints
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0, high=max(size - 1, n_checkpoints), shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.agent_pos = None
        self.checkpoints = None
        self.checkpoints_hit = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = [0, 0]
        self.checkpoints = []
        used = {(0, 0)}
        for _ in range(self.n_checkpoints):
            while True:
                pos = (self.rng.integers(0, self.size), self.rng.integers(0, self.size))
                if pos not in used:
                    self.checkpoints.append(list(pos))
                    used.add(pos)
                    break
        self.checkpoints_hit = 0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], self.checkpoints_hit], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        if self.checkpoints_hit < len(self.checkpoints):
            if self.agent_pos == self.checkpoints[self.checkpoints_hit]:
                self.checkpoints_hit += 1
                if self.checkpoints_hit == len(self.checkpoints):
                    return self._get_obs(), 100.0, True, False, {"success": True}

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}
        return self._get_obs(), 0.0, False, False, {}


class MultiGoalExploration:
    """Multiple goals with different values - must explore to find best."""

    def __init__(self, size=8, max_steps=150, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.agent_pos = None
        self.goals = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = [self.size // 2, self.size // 2]
        self.goals = []
        used = {tuple(self.agent_pos)}

        # High value goal far away
        self.goals.append(([self.size - 1, self.size - 1], 100.0))
        used.add((self.size - 1, self.size - 1))

        # Lower value goals closer
        for reward in [30.0, 20.0, 10.0]:
            while True:
                pos = [self.rng.integers(0, self.size), self.rng.integers(0, self.size)]
                if tuple(pos) not in used:
                    self.goals.append((pos, reward))
                    used.add(tuple(pos))
                    break

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        self.agent_pos[0] = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        self.agent_pos[1] = max(0, min(self.size - 1, self.agent_pos[1] + dy))

        for goal_pos, reward in self.goals:
            if self.agent_pos == goal_pos:
                return self._get_obs(), reward, True, False, {
                    "success": True,
                    "optimal": reward == 100.0,
                    "reward_found": reward
                }

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False, "optimal": False}
        return self._get_obs(), 0.0, False, False, {}


# ============================================================
# BENCHMARK RUNNER
# ============================================================

ALL_MODES = ["triggered", "anti_grover", "grover", "adaptive", "soft_mix", "none"]


def train_mode(env, mode, n_episodes, seed):
    """Train agent with specific interference mode."""
    agent = QuantumSNNAgent.for_env(
        env,
        mode=mode,
        seed=seed,
        ticks_per_decision=20,
        hidden_layers=[16, 8],
        trigger_threshold=60,
        trigger_count=3,
    )

    rewards = []
    successes = []
    optimals = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(reward)
            obs = next_obs
            ep_reward += reward

        agent.end_episode()
        rewards.append(ep_reward)
        successes.append(info.get("success", ep_reward > 0))
        optimals.append(info.get("optimal", False))

    return rewards, successes, optimals


def run_comparison(env_name, env_factory, n_episodes=300, n_seeds=5):
    """Compare all modes on an environment."""
    print(f"\n{'='*70}")
    print(f"COMPARING ALL MODES: {env_name}")
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}")
    print(f"{'='*70}")

    results = {mode: {"rewards": [], "successes": [], "optimals": [], "times": []}
               for mode in ALL_MODES}

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")

        for mode in ALL_MODES:
            env = env_factory()
            start = time.time()
            rewards, successes, optimals = train_mode(env, mode, n_episodes, seed * 1000)
            elapsed = time.time() - start

            results[mode]["rewards"].append(rewards)
            results[mode]["successes"].append(successes)
            results[mode]["optimals"].append(optimals)
            results[mode]["times"].append(elapsed)

            final_success = np.mean(successes[-50:]) * 100
            final_reward = np.mean(rewards[-50:])
            print(f"  {mode:<12}: success={final_success:5.1f}%  reward={final_reward:6.1f}  time={elapsed:.1f}s")

    return results


def print_comparison(results, env_name):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {env_name}")
    print(f"{'='*70}")

    # Header
    print(f"\n{'Mode':<14} {'Success%':>10} {'Optimal%':>10} {'Reward':>10} {'Time':>8}")
    print("-" * 56)

    mode_stats = {}
    for mode in ALL_MODES:
        final_success = np.mean([np.mean(s[-50:]) for s in results[mode]["successes"]]) * 100
        final_optimal = np.mean([np.mean(o[-50:]) for o in results[mode]["optimals"]]) * 100
        final_reward = np.mean([np.mean(r[-50:]) for r in results[mode]["rewards"]])
        avg_time = np.mean(results[mode]["times"])

        mode_stats[mode] = {
            "success": final_success,
            "optimal": final_optimal,
            "reward": final_reward,
            "time": avg_time,
        }

        print(f"{mode:<14} {final_success:>9.1f}% {final_optimal:>9.1f}% {final_reward:>10.1f} {avg_time:>7.1f}s")

    # Find winner
    best_mode = max(mode_stats.keys(), key=lambda m: mode_stats[m]["success"])
    best_success = mode_stats[best_mode]["success"]

    print(f"\n{'='*70}")
    print(f"WINNER: {best_mode.upper()} ({best_success:.1f}% success)")
    print(f"{'='*70}")

    return mode_stats


def run_all_benchmarks():
    """Run comparison on all sparse reward environments."""
    print("=" * 70)
    print("QUANTUM INTERFERENCE MODE COMPARISON")
    print("Testing all modes on sparse reward environments")
    print("=" * 70)
    print("\nModes tested:")
    print("  - none:        Pure classical SNN")
    print("  - triggered:   Anti-Grover -> Grover on high rewards")
    print("  - anti_grover: Always amplify diversity (exploration)")
    print("  - grover:      Always amplify dominant (exploitation)")
    print("  - soft_mix:    Blend exploration/exploitation")
    print("  - adaptive:    Dynamic adjustment")

    all_stats = {}

    # Test 1: Hidden Goal Maze
    results1 = run_comparison(
        "HiddenGoalMaze-6x6",
        lambda: HiddenGoalMaze(size=6, max_steps=100),
        n_episodes=300,
        n_seeds=5,
    )
    all_stats["HiddenGoal"] = print_comparison(results1, "HiddenGoalMaze-6x6")

    # Test 2: Deceptive Maze
    results2 = run_comparison(
        "DeceptiveMaze-10x10",
        lambda: DeceptiveMaze(size=10, max_steps=200),
        n_episodes=300,
        n_seeds=5,
    )
    all_stats["Deceptive"] = print_comparison(results2, "DeceptiveMaze-10x10")

    # Test 3: Long Horizon Search
    results3 = run_comparison(
        "LongHorizonSearch-8x8",
        lambda: LongHorizonSearch(size=8, n_checkpoints=3, max_steps=200),
        n_episodes=300,
        n_seeds=5,
    )
    all_stats["LongHorizon"] = print_comparison(results3, "LongHorizonSearch-8x8")

    # Test 4: Multi-Goal Exploration
    results4 = run_comparison(
        "MultiGoalExploration-8x8",
        lambda: MultiGoalExploration(size=8, max_steps=150),
        n_episodes=300,
        n_seeds=5,
    )
    all_stats["MultiGoal"] = print_comparison(results4, "MultiGoalExploration-8x8")

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ALL MODES ACROSS ALL ENVIRONMENTS")
    print("=" * 70)

    # Compute average success across all environments
    avg_by_mode = {}
    for mode in ALL_MODES:
        successes = [all_stats[env][mode]["success"] for env in all_stats]
        avg_by_mode[mode] = np.mean(successes)

    print(f"\n{'Mode':<14} {'HiddenGoal':>12} {'Deceptive':>12} {'LongHorizon':>12} {'MultiGoal':>12} {'AVERAGE':>10}")
    print("-" * 76)

    for mode in ALL_MODES:
        vals = [all_stats[env][mode]["success"] for env in all_stats]
        avg = avg_by_mode[mode]
        print(f"{mode:<14} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}% {vals[3]:>11.1f}% {avg:>9.1f}%")

    # Ranking
    print("\n" + "-" * 76)
    print("RANKING (by average success rate):")
    ranked = sorted(avg_by_mode.items(), key=lambda x: -x[1])
    for i, (mode, avg) in enumerate(ranked, 1):
        diff_from_best = ranked[0][1] - avg
        if i == 1:
            print(f"  {i}. {mode:<14} {avg:5.1f}%  (BEST)")
        else:
            print(f"  {i}. {mode:<14} {avg:5.1f}%  (-{diff_from_best:.1f}%)")

    print("\n" + "=" * 70)
    print(f"OVERALL WINNER: {ranked[0][0].upper()}")
    print("=" * 70)

    return all_stats, avg_by_mode


if __name__ == "__main__":
    all_stats, avg_by_mode = run_all_benchmarks()
    print("\n[Benchmark complete]")
