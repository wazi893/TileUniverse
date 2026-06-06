"""
Benchmark: Triggered vs Epsilon-Greedy on Gymnasium environments

Compares the Quantum-SNN Triggered interference mode against a standard
epsilon-greedy Q-learning baseline on multiple environments.
"""

import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
from tileuniverse.rl import QuantumSNNAgent, train_agent


class EpsilonGreedyAgent:
    """Simple epsilon-greedy agent with Q-learning."""

    def __init__(self, n_states_bins, n_actions, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.bins = n_states_bins

        # Q-table: discretized state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        self.last_state = None
        self.last_action = None
        self._episode_reward = 0
        self._episode_length = 0
        self.episodes = 0

    def discretize(self, obs):
        """Discretize continuous observation."""
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

        # Q-learning update
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


def get_obs_bounds(env):
    """Get observation bounds for discretization."""
    space = env.observation_space
    bounds = []
    for i in range(space.shape[0]):
        low = space.low[i] if not np.isinf(space.low[i]) else -10
        high = space.high[i] if not np.isinf(space.high[i]) else 10
        bounds.append((low, high))
    return bounds


def run_benchmark(env_name, n_episodes=500, n_seeds=5, epsilon=0.1):
    """Run benchmark comparing Triggered vs Epsilon-Greedy."""

    print(f"\n{'='*60}")
    print(f"Benchmark: {env_name}")
    print(f"Episodes: {n_episodes}, Seeds: {n_seeds}, Epsilon: {epsilon}")
    print(f"{'='*60}")

    results = {
        "triggered": {"rewards": [], "times": []},
        "epsilon_greedy": {"rewards": [], "times": []},
    }

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")

        # --- Triggered Mode ---
        env = gym.make(env_name)
        agent = QuantumSNNAgent.for_env(
            env,
            mode="triggered",
            seed=seed * 1000,
            ticks_per_decision=15,
        )

        start = time.time()
        stats = train_agent(env, agent, n_episodes=n_episodes, verbose=0)
        elapsed = time.time() - start

        rewards = [s["reward"] for s in stats]
        final_avg = np.mean(rewards[-50:])
        results["triggered"]["rewards"].append(rewards)
        results["triggered"]["times"].append(elapsed)

        print(f"  Triggered:       final_avg={final_avg:7.1f}  time={elapsed:.1f}s")
        env.close()

        # --- Epsilon-Greedy ---
        env = gym.make(env_name)
        np.random.seed(seed * 1000)

        bounds = get_obs_bounds(env)
        eg_agent = EpsilonGreedyAgent(
            n_states_bins=bounds,
            n_actions=env.action_space.n,
            epsilon=epsilon,
            alpha=0.1,
            gamma=0.99,
        )

        start = time.time()
        eg_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = eg_agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                eg_agent.learn(reward, next_obs, done)
                obs = next_obs
            ep_stats = eg_agent.end_episode()
            eg_rewards.append(ep_stats["reward"])

        elapsed = time.time() - start
        final_avg = np.mean(eg_rewards[-50:])
        results["epsilon_greedy"]["rewards"].append(eg_rewards)
        results["epsilon_greedy"]["times"].append(elapsed)

        print(f"  Epsilon-Greedy:  final_avg={final_avg:7.1f}  time={elapsed:.1f}s")
        env.close()

    return results


def print_summary(results, env_name):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {env_name}")
    print(f"{'='*60}")

    for method in ["triggered", "epsilon_greedy"]:
        all_rewards = results[method]["rewards"]

        # Final performance (last 50 episodes average across seeds)
        final_avgs = [np.mean(r[-50:]) for r in all_rewards]

        # Learning progress at different points
        early_avgs = [np.mean(r[:50]) for r in all_rewards]
        mid_avgs = [np.mean(r[200:250]) for r in all_rewards]

        mean_final = np.mean(final_avgs)
        std_final = np.std(final_avgs)
        mean_time = np.mean(results[method]["times"])

        print(f"\n{method.upper().replace('_', '-')}:")
        print(f"  Early (ep 1-50):    {np.mean(early_avgs):7.1f} +/- {np.std(early_avgs):.1f}")
        print(f"  Mid (ep 200-250):   {np.mean(mid_avgs):7.1f} +/- {np.std(mid_avgs):.1f}")
        print(f"  Final (ep 450-500): {mean_final:7.1f} +/- {std_final:.1f}")
        print(f"  Avg time:           {mean_time:.1f}s")

    # Comparison
    trig_final = np.mean([np.mean(r[-50:]) for r in results["triggered"]["rewards"]])
    eg_final = np.mean([np.mean(r[-50:]) for r in results["epsilon_greedy"]["rewards"]])

    diff = trig_final - eg_final
    pct = (diff / abs(eg_final)) * 100 if eg_final != 0 else 0

    print(f"\n{'='*60}")
    winner = "TRIGGERED" if diff > 0 else "EPSILON-GREEDY"
    print(f"WINNER: {winner} ({abs(diff):.1f} better, {abs(pct):.1f}%)")
    print(f"{'='*60}")

    return trig_final, eg_final


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM-SNN TRIGGERED vs EPSILON-GREEDY BENCHMARK")
    print("=" * 60)

    all_results = {}

    # CartPole - Dense rewards (baseline comparison)
    results_cartpole = run_benchmark(
        "CartPole-v1",
        n_episodes=500,
        n_seeds=5,
        epsilon=0.1
    )
    trig_cp, eg_cp = print_summary(results_cartpole, "CartPole-v1")
    all_results["CartPole-v1"] = (trig_cp, eg_cp)

    # Acrobot - Sparse rewards (Triggered should excel here)
    results_acrobot = run_benchmark(
        "Acrobot-v1",
        n_episodes=500,
        n_seeds=5,
        epsilon=0.15
    )
    trig_ac, eg_ac = print_summary(results_acrobot, "Acrobot-v1")
    all_results["Acrobot-v1"] = (trig_ac, eg_ac)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n{'Environment':<20} {'Triggered':>12} {'Eps-Greedy':>12} {'Diff':>10}")
    print("-" * 60)
    for env_name, (trig, eg) in all_results.items():
        diff = trig - eg
        print(f"{env_name:<20} {trig:>12.1f} {eg:>12.1f} {diff:>+10.1f}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
