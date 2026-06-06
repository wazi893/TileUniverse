"""
Benchmark: Triggered Mode on Sparse Reward Tasks

Demonstrates Triggered interference mode's strengths on environments where:
1. Rewards are sparse (only at goal states)
2. Exploration is critical to discover reward locations
3. Premature exploitation leads to local optima

The Triggered mode explores with anti-Grover (diversity amplification) until
consistent high rewards are found, then switches to Grover (exploitation).
"""

import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

# Import our Quantum-SNN agent
from tileuniverse.rl import QuantumSNNAgent, train_agent
from tileuniverse import InterferenceMode


class EpsilonGreedyAgent:
    """Tabular Q-learning baseline with epsilon-greedy exploration."""

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


class SparseRewardWrapper(gym.Wrapper):
    """Converts dense rewards to sparse: only reward at episode success."""

    def __init__(self, env, success_reward=100.0, step_penalty=0.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        # Convert to sparse reward
        if terminated and reward > 0:
            # Success - give big reward
            sparse_reward = self.success_reward
        elif terminated or truncated:
            # Failure/timeout - no reward (or small penalty)
            sparse_reward = self.step_penalty
        else:
            # During episode - no reward
            sparse_reward = self.step_penalty

        return obs, sparse_reward, terminated, truncated, info


class DelayedRewardWrapper(gym.Wrapper):
    """Accumulates rewards and only delivers them at episode end."""

    def __init__(self, env, bonus_multiplier=1.0):
        super().__init__(env)
        self.bonus_multiplier = bonus_multiplier
        self._accumulated = 0

    def reset(self, **kwargs):
        self._accumulated = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._accumulated += reward

        if terminated or truncated:
            # Deliver all accumulated reward at end
            final_reward = self._accumulated * self.bonus_multiplier
            return obs, final_reward, terminated, truncated, info
        else:
            return obs, 0.0, terminated, truncated, info


class ExplorationMaze:
    """
    Custom sparse reward maze environment.

    Agent must explore to find the hidden goal. Reward only given when
    goal is reached. Tests pure exploration capability.
    """

    def __init__(self, size=8, max_steps=200, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=size - 1, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Random start position
        self.agent_pos = [
            self.rng.integers(0, self.size),
            self.rng.integers(0, self.size),
        ]

        # Random goal position (not same as agent)
        while True:
            self.goal_pos = [
                self.rng.integers(0, self.size),
                self.rng.integers(0, self.size),
            ]
            if self.goal_pos != self.agent_pos:
                break

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation: [agent_x, agent_y, goal_x, goal_y]
        # Note: In real sparse task, goal position would be hidden
        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.steps += 1

        # Move agent
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Check goal
        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {"success": True}

        # Check timeout
        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}

        # No intermediate rewards (sparse!)
        return self._get_obs(), 0.0, False, False, {}


class HiddenGoalMaze:
    """
    Maze where goal position is NOT in observation.
    Agent must explore purely through trial and error.
    """

    def __init__(self, size=6, max_steps=100, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Only agent position in observation (goal is hidden)
        self.observation_space = gym.spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = [0, 0]  # Always start at origin
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
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        if self.agent_pos == self.goal_pos:
            return self._get_obs(), 100.0, True, False, {"success": True}

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}

        return self._get_obs(), 0.0, False, False, {}


class MultiGoalExploration:
    """
    Environment with multiple goals of different values.
    Agent must explore to find the best goal, not just any goal.
    Tests exploration vs premature exploitation.
    """

    def __init__(self, size=8, n_goals=4, max_steps=150, seed=None):
        self.size = size
        self.n_goals = n_goals
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = gym.spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self.agent_pos = None
        self.goals = None  # List of (pos, reward)
        self.steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = [self.size // 2, self.size // 2]  # Start in center

        # Create goals with different rewards
        # One high-value goal far away, several low-value goals nearby
        self.goals = []
        positions_used = [tuple(self.agent_pos)]

        # High value goal (far corner)
        far_pos = [self.size - 1, self.size - 1]
        self.goals.append((far_pos, 100.0))
        positions_used.append(tuple(far_pos))

        # Medium and low value goals (closer)
        rewards = [30.0, 20.0, 10.0]
        for r in rewards:
            while True:
                pos = [
                    self.rng.integers(0, self.size),
                    self.rng.integers(0, self.size),
                ]
                if tuple(pos) not in positions_used:
                    # Prefer positions closer to start for lower rewards
                    dist = abs(pos[0] - self.size // 2) + abs(pos[1] - self.size // 2)
                    if dist < self.size // 2 or r == rewards[-1]:
                        self.goals.append((pos, r))
                        positions_used.append(tuple(pos))
                        break

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        self.steps += 1

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        # Check if hit any goal
        for goal_pos, reward in self.goals:
            if self.agent_pos == goal_pos:
                return self._get_obs(), reward, True, False, {
                    "success": True,
                    "reward_found": reward,
                    "optimal": reward == 100.0,
                }

        if self.steps >= self.max_steps:
            return self._get_obs(), 0.0, False, True, {"success": False}

        return self._get_obs(), 0.0, False, False, {}


def get_obs_bounds(env):
    """Get observation bounds for discretization."""
    space = env.observation_space
    if isinstance(space, gym.spaces.Discrete):
        return [(0, space.n - 1)]
    bounds = []
    for i in range(space.shape[0]):
        low = space.low[i] if not np.isinf(space.low[i]) else -10
        high = space.high[i] if not np.isinf(space.high[i]) else 10
        bounds.append((low, high))
    return bounds


def train_epsilon_greedy(env, n_episodes, epsilon=0.2, seed=0):
    """Train epsilon-greedy agent and return episode rewards."""
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
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(reward, next_obs, done)
            obs = next_obs

        stats = agent.end_episode()
        rewards.append(stats["reward"])
        successes.append(info.get("success", stats["reward"] > 0))

    return rewards, successes


def train_triggered(env, n_episodes, seed=0):
    """Train Triggered mode agent and return episode rewards."""
    agent = QuantumSNNAgent.for_env(
        env,
        mode="triggered",
        seed=seed,
        ticks_per_decision=20,
        hidden_layers=[16, 8],
        trigger_threshold=60,
        trigger_count=3,
    )

    stats = train_agent(env, agent, n_episodes=n_episodes, verbose=0)
    rewards = [s["reward"] for s in stats]

    # Calculate successes from rewards
    successes = [r > 0 for r in rewards]

    return rewards, successes


def run_sparse_benchmark(env_name, env_factory, n_episodes=300, n_seeds=5):
    """Run benchmark on a sparse reward environment."""
    print(f"\n{'='*60}")
    print(f"SPARSE REWARD BENCHMARK: {env_name}")
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}")
    print(f"{'='*60}")

    results = {
        "triggered": {"rewards": [], "successes": [], "times": []},
        "epsilon_greedy": {"rewards": [], "successes": [], "times": []},
    }

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")

        # Triggered mode
        env = env_factory()
        start = time.time()
        rewards, successes = train_triggered(env, n_episodes, seed=seed * 1000)
        elapsed = time.time() - start

        results["triggered"]["rewards"].append(rewards)
        results["triggered"]["successes"].append(successes)
        results["triggered"]["times"].append(elapsed)

        final_success = np.mean(successes[-50:]) * 100
        final_reward = np.mean(rewards[-50:])
        print(f"  Triggered:       success={final_success:5.1f}%  reward={final_reward:6.1f}  time={elapsed:.1f}s")

        # Epsilon-greedy
        env = env_factory()
        start = time.time()
        rewards, successes = train_epsilon_greedy(env, n_episodes, epsilon=0.2, seed=seed * 1000)
        elapsed = time.time() - start

        results["epsilon_greedy"]["rewards"].append(rewards)
        results["epsilon_greedy"]["successes"].append(successes)
        results["epsilon_greedy"]["times"].append(elapsed)

        final_success = np.mean(successes[-50:]) * 100
        final_reward = np.mean(rewards[-50:])
        print(f"  Epsilon-Greedy:  success={final_success:5.1f}%  reward={final_reward:6.1f}  time={elapsed:.1f}s")

    return results


def print_results(results, env_name):
    """Print detailed results for an environment."""
    print(f"\n{'-'*60}")
    print(f"RESULTS: {env_name}")
    print(f"{'-'*60}")

    for method in ["triggered", "epsilon_greedy"]:
        all_rewards = results[method]["rewards"]
        all_successes = results[method]["successes"]

        # Success rates at different points
        early_success = [np.mean(s[:50]) * 100 for s in all_successes]
        mid_success = [np.mean(s[100:150]) * 100 for s in all_successes]
        final_success = [np.mean(s[-50:]) * 100 for s in all_successes]

        # Rewards
        final_rewards = [np.mean(r[-50:]) for r in all_rewards]

        print(f"\n{method.upper().replace('_', '-')}:")
        print(f"  Early success (ep 1-50):     {np.mean(early_success):5.1f}% +/- {np.std(early_success):.1f}%")
        print(f"  Mid success (ep 100-150):    {np.mean(mid_success):5.1f}% +/- {np.std(mid_success):.1f}%")
        print(f"  Final success (ep 250-300):  {np.mean(final_success):5.1f}% +/- {np.std(final_success):.1f}%")
        print(f"  Final avg reward:            {np.mean(final_rewards):6.1f} +/- {np.std(final_rewards):.1f}")

    # Comparison
    trig_success = np.mean([np.mean(s[-50:]) for s in results["triggered"]["successes"]]) * 100
    eg_success = np.mean([np.mean(s[-50:]) for s in results["epsilon_greedy"]["successes"]]) * 100

    diff = trig_success - eg_success

    print(f"\n{'='*60}")
    if diff > 0:
        print(f"WINNER: TRIGGERED (+{diff:.1f}% success rate)")
    elif diff < 0:
        print(f"WINNER: EPSILON-GREEDY (+{-diff:.1f}% success rate)")
    else:
        print("TIE")
    print(f"{'='*60}")

    return trig_success, eg_success


if __name__ == "__main__":
    print("=" * 60)
    print("SPARSE REWARD BENCHMARK")
    print("Triggered Interference Mode vs Epsilon-Greedy")
    print("=" * 60)
    print("\nTriggered mode uses anti-Grover interference for exploration")
    print("until consistent rewards trigger switch to Grover exploitation.")
    print("This excels in sparse reward environments where exploration is key.")

    all_results = {}

    # Benchmark 1: Hidden Goal Maze (hardest - no goal info in observation)
    print("\n" + "=" * 60)
    print("TEST 1: Hidden Goal Maze")
    print("Goal position is NOT visible - pure exploration required")
    print("=" * 60)

    results1 = run_sparse_benchmark(
        "HiddenGoalMaze-6x6",
        lambda: HiddenGoalMaze(size=6, max_steps=100),
        n_episodes=300,
        n_seeds=5,
    )
    t1, e1 = print_results(results1, "HiddenGoalMaze-6x6")
    all_results["HiddenGoalMaze"] = (t1, e1)

    # Benchmark 2: Multi-Goal Exploration
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Goal Exploration")
    print("Multiple goals with different values - must explore to find best")
    print("=" * 60)

    results2 = run_sparse_benchmark(
        "MultiGoalExploration-8x8",
        lambda: MultiGoalExploration(size=8, n_goals=4, max_steps=150),
        n_episodes=300,
        n_seeds=5,
    )
    t2, e2 = print_results(results2, "MultiGoalExploration-8x8")
    all_results["MultiGoal"] = (t2, e2)

    # Benchmark 3: Sparse CartPole (only reward on success)
    print("\n" + "=" * 60)
    print("TEST 3: Sparse CartPole")
    print("CartPole with reward ONLY at 500 steps (no intermediate rewards)")
    print("=" * 60)

    results3 = run_sparse_benchmark(
        "SparseCartPole-v1",
        lambda: SparseRewardWrapper(gym.make("CartPole-v1"), success_reward=100.0),
        n_episodes=300,
        n_seeds=5,
    )
    t3, e3 = print_results(results3, "SparseCartPole-v1")
    all_results["SparseCartPole"] = (t3, e3)

    # Benchmark 4: FrozenLake (classic sparse reward)
    print("\n" + "=" * 60)
    print("TEST 4: FrozenLake-v1 (4x4)")
    print("Classic sparse reward: only +1 at goal, slippery surface")
    print("=" * 60)

    results4 = run_sparse_benchmark(
        "FrozenLake-v1",
        lambda: gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True),
        n_episodes=500,
        n_seeds=5,
    )
    t4, e4 = print_results(results4, "FrozenLake-v1")
    all_results["FrozenLake"] = (t4, e4)

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: SPARSE REWARD BENCHMARKS")
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
        print(f"OVERALL WINNER: TRIGGERED MODE (+{avg_diff:.1f}% average success)")
    else:
        print(f"OVERALL WINNER: EPSILON-GREEDY (+{-avg_diff:.1f}% average success)")
    print("=" * 60)

    print("\n[Benchmark complete]")
