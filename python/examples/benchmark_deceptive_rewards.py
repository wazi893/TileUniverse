"""
Benchmark: Deceptive Reward Environments

Tests environments where greedy exploitation leads to local optima.
Triggered mode's exploration should help escape these traps.
"""

import numpy as np
import time
from collections import defaultdict
from typing import List, Dict

import gymnasium as gym
from gymnasium import spaces

from tileuniverse.rl import QuantumSNNAgent, train_agent


class EpsilonGreedyAgent:
    """Tabular Q-learning baseline."""

    def __init__(self, n_states_bins, n_actions, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.bins = n_states_bins
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.last_state = None
        self.last_action = None
        self._episode_reward = 0
        self._episode_length = 0
        self.episodes = 0

    def discretize(self, obs):
        if isinstance(obs, (int, np.integer)):
            return (int(obs),)
        bins_per_dim = 10
        discretized = []
        for i, val in enumerate(obs):
            low, high = self.bins[i]
            if high == low:
                discretized.append(0)
            else:
                bin_idx = int((val - low) / (high - low) * bins_per_dim)
                bin_idx = max(0, min(bins_per_dim - 1, bin_idx))
                discretized.append(bin_idx)
        return tuple(discretized)

    def act(self, obs):
        state = self.discretize(obs)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_table[state])
        self.last_state = state
        self.last_action = action
        self._episode_length += 1
        return action

    def learn(self, reward, next_obs, done):
        if self.last_state is None:
            return
        next_state = self.discretize(next_obs)
        best_next = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] += self.alpha * td_error
        self._episode_reward += reward

    def end_episode(self):
        stats = {
            "episode": self.episodes,
            "reward": self._episode_reward,
            "length": self._episode_length,
        }
        self.episodes += 1
        self._episode_reward = 0
        self._episode_length = 0
        self.last_state = None
        self.last_action = None
        return stats


class DeceptiveMaze:
    """
    Maze with a "trap" - local optimum that's easy to find but suboptimal.

    Layout:
    - Agent starts at center
    - Small reward (20) is close to start
    - Large reward (100) requires going through penalty zone first

    This tests whether the agent explores past early rewards.
    """

    def __init__(self, size=10, max_steps=200, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Fixed positions
        self.start_pos = [size // 2, size // 2]
        self.trap_pos = [size // 2 + 1, size // 2]  # Easy to find (adjacent)
        self.goal_pos = [size - 1, size - 1]  # Far corner (optimal)
        self.penalty_zone = set()  # Zone before optimal goal

        # Create penalty zone (must pass through to reach goal)
        for x in range(size - 3, size):
            for y in range(size - 3, size):
                if [x, y] != self.goal_pos:
                    self.penalty_zone.add((x, y))

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
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Check trap (suboptimal but easy reward)
        if self.agent_pos == self.trap_pos and not self.visited_trap:
            self.visited_trap = True
            return self._get_obs(), 20.0, True, False, {
                "success": True,
                "optimal": False,
                "found": "trap",
            }

        # Check optimal goal
        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {
                "success": True,
                "optimal": True,
                "found": "goal",
            }

        # Penalty zone (discourages going toward optimal)
        reward = 0.0
        if tuple(self.agent_pos) in self.penalty_zone:
            reward = -1.0

        if self.steps >= self.max_steps:
            return self._get_obs(), reward, False, True, {
                "success": False,
                "optimal": False,
            }

        return self._get_obs(), reward, False, False, {}


class RiskyExploration:
    """
    Environment where safe actions give small rewards,
    but risky exploration gives large rewards.

    Models real-world scenarios where novel approaches
    might fail but have high upside.
    """

    def __init__(self, size=8, max_steps=100, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Safe zone: left half, small consistent rewards
        # Risky zone: right half, possible failure but big rewards
        self.safe_boundary = size // 2

        self.agent_pos = None
        self.steps = 0
        self.collected_safe = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = [0, self.size // 2]  # Start on left edge
        self.steps = 0
        self.collected_safe = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Safe zone: small consistent rewards
        if self.agent_pos[0] < self.safe_boundary:
            if self.collected_safe < 5:
                self.collected_safe += 1
                return self._get_obs(), 5.0, False, False, {"zone": "safe"}

        # Risky zone: right side
        if self.agent_pos[0] >= self.safe_boundary:
            # Big reward at far right
            if self.agent_pos[0] == self.size - 1:
                return self._get_obs(), 100.0, True, False, {
                    "success": True,
                    "optimal": True,
                    "zone": "risky",
                }

            # Random failure in risky zone (30% chance)
            if self.rng.random() < 0.3:
                return self._get_obs(), -10.0, True, False, {
                    "success": False,
                    "zone": "risky_fail",
                }

        if self.steps >= self.max_steps:
            # End with safe rewards collected
            return self._get_obs(), 0.0, False, True, {
                "success": self.collected_safe > 0,
                "optimal": False,
            }

        return self._get_obs(), 0.0, False, False, {}


class CliffWalkSparse:
    """
    Cliff walk with sparse rewards.

    The agent must walk along a cliff edge. Falling off gives -100.
    The safe path is longer but avoids the cliff.
    The risky path is shorter but one wrong step = disaster.

    Only rewards at the goal (no intermediate rewards).
    """

    def __init__(self, width=12, height=4, max_steps=100, seed=None):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0, high=max(width, height) - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.start_pos = [0, 0]
        self.goal_pos = [width - 1, 0]

        # Cliff: bottom row except start and goal
        self.cliff = set()
        for x in range(1, width - 1):
            self.cliff.add((x, 0))

        self.agent_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = list(self.start_pos)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.width - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.height - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Check cliff
        if tuple(self.agent_pos) in self.cliff:
            return self._get_obs(), -100.0, True, False, {
                "success": False,
                "fell": True,
            }

        # Check goal
        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {
                "success": True,
                "optimal": True,
            }

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}

        return self._get_obs(), 0.0, False, False, {}


class LongHorizonSearch:
    """
    Environment requiring long-horizon planning.

    The reward is only available after visiting multiple checkpoints
    in sequence. Random exploration rarely succeeds.
    """

    def __init__(self, size=8, n_checkpoints=3, max_steps=200, seed=None):
        self.size = size
        self.n_checkpoints = n_checkpoints
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(3,), dtype=np.float32  # x, y, checkpoints_hit
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

        # Generate random checkpoint positions
        self.checkpoints = []
        used = {(0, 0)}
        for _ in range(self.n_checkpoints):
            while True:
                pos = (
                    self.rng.integers(0, self.size),
                    self.rng.integers(0, self.size),
                )
                if pos not in used:
                    self.checkpoints.append(list(pos))
                    used.add(pos)
                    break

        self.checkpoints_hit = 0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1], self.checkpoints_hit],
            dtype=np.float32,
        )

    def step(self, action):
        self.steps += 1

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Check if hit next checkpoint in sequence
        if self.checkpoints_hit < len(self.checkpoints):
            next_checkpoint = self.checkpoints[self.checkpoints_hit]
            if self.agent_pos == next_checkpoint:
                self.checkpoints_hit += 1

                # All checkpoints hit!
                if self.checkpoints_hit == len(self.checkpoints):
                    return self._get_obs(), 100.0, True, False, {
                        "success": True,
                        "optimal": True,
                    }

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {
                "success": False,
                "checkpoints_hit": self.checkpoints_hit,
            }

        return self._get_obs(), 0.0, False, False, {}


def get_obs_bounds(env):
    """Get observation bounds for discretization."""
    space = env.observation_space
    if isinstance(space, spaces.Discrete):
        return [(0, space.n - 1)]
    bounds = []
    for i in range(space.shape[0]):
        low = space.low[i] if not np.isinf(space.low[i]) else -10
        high = space.high[i] if not np.isinf(space.high[i]) else 10
        bounds.append((low, high))
    return bounds


def train_epsilon_greedy(env, n_episodes, epsilon=0.2, seed=0):
    """Train epsilon-greedy agent."""
    np.random.seed(seed)
    bounds = get_obs_bounds(env)
    agent = EpsilonGreedyAgent(
        n_states_bins=bounds,
        n_actions=env.action_space.n,
        epsilon=epsilon,
        alpha=0.1,
        gamma=0.99,
    )

    rewards = []
    optimal_found = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

        agent.end_episode()
        rewards.append(ep_reward)
        optimal_found.append(info.get("optimal", False))

    return rewards, optimal_found


def train_triggered(env, n_episodes, seed=0):
    """Train Triggered mode agent."""
    agent = QuantumSNNAgent.for_env(
        env,
        mode="triggered",
        seed=seed,
        ticks_per_decision=20,
        hidden_layers=[16, 8],
        trigger_threshold=60,
        trigger_count=3,
    )

    rewards = []
    optimal_found = []

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
        optimal_found.append(info.get("optimal", False))

    return rewards, optimal_found


def run_benchmark(env_name, env_factory, n_episodes=300, n_seeds=5):
    """Run benchmark on environment."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {env_name}")
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}")
    print(f"{'='*60}")

    results = {
        "triggered": {"rewards": [], "optimal": [], "times": []},
        "epsilon_greedy": {"rewards": [], "optimal": [], "times": []},
    }

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")

        # Triggered
        env = env_factory()
        start = time.time()
        rewards, optimal = train_triggered(env, n_episodes, seed=seed * 1000)
        elapsed = time.time() - start

        results["triggered"]["rewards"].append(rewards)
        results["triggered"]["optimal"].append(optimal)
        results["triggered"]["times"].append(elapsed)

        final_reward = np.mean(rewards[-50:])
        optimal_rate = np.mean(optimal[-50:]) * 100
        print(f"  Triggered:       reward={final_reward:6.1f}  optimal={optimal_rate:5.1f}%  time={elapsed:.1f}s")

        # Epsilon-greedy
        env = env_factory()
        start = time.time()
        rewards, optimal = train_epsilon_greedy(env, n_episodes, epsilon=0.2, seed=seed * 1000)
        elapsed = time.time() - start

        results["epsilon_greedy"]["rewards"].append(rewards)
        results["epsilon_greedy"]["optimal"].append(optimal)
        results["epsilon_greedy"]["times"].append(elapsed)

        final_reward = np.mean(rewards[-50:])
        optimal_rate = np.mean(optimal[-50:]) * 100
        print(f"  Epsilon-Greedy:  reward={final_reward:6.1f}  optimal={optimal_rate:5.1f}%  time={elapsed:.1f}s")

    return results


def print_results(results, env_name):
    """Print detailed results."""
    print(f"\n{'-'*60}")
    print(f"RESULTS: {env_name}")
    print(f"{'-'*60}")

    for method in ["triggered", "epsilon_greedy"]:
        all_rewards = results[method]["rewards"]
        all_optimal = results[method]["optimal"]

        final_rewards = [np.mean(r[-50:]) for r in all_rewards]
        final_optimal = [np.mean(o[-50:]) * 100 for o in all_optimal]

        print(f"\n{method.upper().replace('_', '-')}:")
        print(f"  Final avg reward:   {np.mean(final_rewards):6.1f} +/- {np.std(final_rewards):.1f}")
        print(f"  Optimal found rate: {np.mean(final_optimal):5.1f}% +/- {np.std(final_optimal):.1f}%")

    trig_optimal = np.mean([np.mean(o[-50:]) for o in results["triggered"]["optimal"]]) * 100
    eg_optimal = np.mean([np.mean(o[-50:]) for o in results["epsilon_greedy"]["optimal"]]) * 100

    diff = trig_optimal - eg_optimal

    print(f"\n{'='*60}")
    if diff > 0:
        print(f"WINNER: TRIGGERED (+{diff:.1f}% optimal solutions found)")
    elif diff < 0:
        print(f"WINNER: EPSILON-GREEDY (+{-diff:.1f}% optimal solutions found)")
    else:
        print("TIE")
    print(f"{'='*60}")

    return trig_optimal, eg_optimal


if __name__ == "__main__":
    print("=" * 60)
    print("DECEPTIVE REWARD BENCHMARK")
    print("Testing exploration in environments with local optima")
    print("=" * 60)

    all_results = {}

    # Test 1: Deceptive Maze (trap vs optimal)
    print("\n" + "=" * 60)
    print("TEST 1: Deceptive Maze")
    print("Easy trap (20 reward) vs hard optimal (100 reward)")
    print("=" * 60)

    results1 = run_benchmark(
        "DeceptiveMaze-10x10",
        lambda: DeceptiveMaze(size=10, max_steps=200),
        n_episodes=400,
        n_seeds=5,
    )
    t1, e1 = print_results(results1, "DeceptiveMaze-10x10")
    all_results["DeceptiveMaze"] = (t1, e1)

    # Test 2: Risky Exploration
    print("\n" + "=" * 60)
    print("TEST 2: Risky Exploration")
    print("Safe small rewards vs risky big reward")
    print("=" * 60)

    results2 = run_benchmark(
        "RiskyExploration-8x8",
        lambda: RiskyExploration(size=8, max_steps=100),
        n_episodes=400,
        n_seeds=5,
    )
    t2, e2 = print_results(results2, "RiskyExploration-8x8")
    all_results["RiskyExploration"] = (t2, e2)

    # Test 3: Cliff Walk Sparse
    print("\n" + "=" * 60)
    print("TEST 3: Cliff Walk (Sparse)")
    print("Must navigate around cliff with no intermediate rewards")
    print("=" * 60)

    results3 = run_benchmark(
        "CliffWalkSparse-12x4",
        lambda: CliffWalkSparse(width=12, height=4, max_steps=100),
        n_episodes=400,
        n_seeds=5,
    )
    t3, e3 = print_results(results3, "CliffWalkSparse-12x4")
    all_results["CliffWalkSparse"] = (t3, e3)

    # Test 4: Long Horizon Search
    print("\n" + "=" * 60)
    print("TEST 4: Long Horizon Search")
    print("Must visit 3 checkpoints in sequence")
    print("=" * 60)

    results4 = run_benchmark(
        "LongHorizonSearch-8x8",
        lambda: LongHorizonSearch(size=8, n_checkpoints=3, max_steps=200),
        n_episodes=400,
        n_seeds=5,
    )
    t4, e4 = print_results(results4, "LongHorizonSearch-8x8")
    all_results["LongHorizonSearch"] = (t4, e4)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: DECEPTIVE REWARD BENCHMARKS")
    print("=" * 60)
    print(f"\n{'Environment':<25} {'Triggered':>12} {'Eps-Greedy':>12} {'Diff':>10}")
    print("-" * 60)

    total_trig = 0
    total_eg = 0

    for env_name, (trig, eg) in all_results.items():
        diff = trig - eg
        sign = "+" if diff >= 0 else ""
        print(f"{env_name:<25} {trig:>11.1f}% {eg:>11.1f}% {sign}{diff:>9.1f}%")
        total_trig += trig
        total_eg += eg

    avg_trig = total_trig / len(all_results)
    avg_eg = total_eg / len(all_results)
    avg_diff = avg_trig - avg_eg

    print("-" * 60)
    sign = "+" if avg_diff >= 0 else ""
    print(f"{'AVERAGE':<25} {avg_trig:>11.1f}% {avg_eg:>11.1f}% {sign}{avg_diff:>9.1f}%")

    print("\n" + "=" * 60)
    if avg_diff > 0:
        print(f"OVERALL WINNER: TRIGGERED MODE (+{avg_diff:.1f}% optimal solutions)")
    else:
        print(f"OVERALL WINNER: EPSILON-GREEDY (+{-avg_diff:.1f}% optimal solutions)")
    print("=" * 60)

    print("\n[Benchmark complete]")
