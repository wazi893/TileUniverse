"""
Hyperparameter Sweep v2: Focus on Single-Layer Architectures

Discovery from v1: Single-layer networks outperform multi-layer by ~90%!
This sweep optimizes the single-layer configuration.
"""

import numpy as np
import time
from typing import List, Dict
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from tileuniverse.rl import QuantumSNNAgent


# Environments
class HiddenGoalMaze:
    def __init__(self, size=6, max_steps=100):
        self.size = size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.agent_pos = [0, 0]
        self.goal_pos = [self.rng.integers(self.size//2, self.size),
                         self.rng.integers(self.size//2, self.size)]
        self.steps = 0
        return np.array(self.agent_pos, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        self.agent_pos[0] = max(0, min(self.size-1, self.agent_pos[0]+dx))
        self.agent_pos[1] = max(0, min(self.size-1, self.agent_pos[1]+dy))
        if self.agent_pos == self.goal_pos:
            return np.array(self.agent_pos, dtype=np.float32), 100.0, True, False, {"success": True}
        if self.steps >= self.max_steps:
            return np.array(self.agent_pos, dtype=np.float32), 0.0, False, True, {"success": False}
        return np.array(self.agent_pos, dtype=np.float32), 0.0, False, False, {}


class DeceptiveMaze:
    def __init__(self, size=10, max_steps=200):
        self.size = size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.start = [size//2, size//2]
        self.trap = [size//2+1, size//2]
        self.goal = [size-1, size-1]

    def reset(self, seed=None, options=None):
        self.agent_pos = list(self.start)
        self.steps = 0
        self.trapped = False
        return np.array(self.agent_pos, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        self.agent_pos[0] = max(0, min(self.size-1, self.agent_pos[0]+dx))
        self.agent_pos[1] = max(0, min(self.size-1, self.agent_pos[1]+dy))
        if self.agent_pos == self.trap and not self.trapped:
            self.trapped = True
            return np.array(self.agent_pos, dtype=np.float32), 20.0, True, False, {"success": True, "optimal": False}
        if self.agent_pos == self.goal:
            return np.array(self.agent_pos, dtype=np.float32), 100.0, True, False, {"success": True, "optimal": True}
        if self.steps >= self.max_steps:
            return np.array(self.agent_pos, dtype=np.float32), 0.0, False, True, {"success": False}
        return np.array(self.agent_pos, dtype=np.float32), 0.0, False, False, {}


class LongHorizonSearch:
    def __init__(self, size=8, n_checkpoints=3, max_steps=200):
        self.size = size
        self.n_checkpoints = n_checkpoints
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=max(size-1, n_checkpoints), shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.agent_pos = [0, 0]
        self.checkpoints = []
        used = {(0,0)}
        for _ in range(self.n_checkpoints):
            while True:
                pos = (self.rng.integers(0, self.size), self.rng.integers(0, self.size))
                if pos not in used:
                    self.checkpoints.append(list(pos))
                    used.add(pos)
                    break
        self.hits = 0
        self.steps = 0
        return np.array([self.agent_pos[0], self.agent_pos[1], self.hits], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        self.agent_pos[0] = max(0, min(self.size-1, self.agent_pos[0]+dx))
        self.agent_pos[1] = max(0, min(self.size-1, self.agent_pos[1]+dy))
        if self.hits < len(self.checkpoints) and self.agent_pos == self.checkpoints[self.hits]:
            self.hits += 1
            if self.hits == len(self.checkpoints):
                return np.array([self.agent_pos[0], self.agent_pos[1], self.hits], dtype=np.float32), 100.0, True, False, {"success": True}
        if self.steps >= self.max_steps:
            return np.array([self.agent_pos[0], self.agent_pos[1], self.hits], dtype=np.float32), 0.0, False, True, {"success": False}
        return np.array([self.agent_pos[0], self.agent_pos[1], self.hits], dtype=np.float32), 0.0, False, False, {}


@dataclass
class Config:
    trigger_threshold: int
    trigger_count: int
    ticks_per_decision: int
    hidden_size: int  # Single layer size

    def __str__(self):
        return f"th={self.trigger_threshold}, tc={self.trigger_count}, tpd={self.ticks_per_decision}, hs={self.hidden_size}"


def evaluate(config: Config, envs, n_episodes=300, n_seeds=5):
    """Evaluate configuration with more episodes and seeds for accuracy."""
    results = {"config": config, "envs": {}}

    for env_name, env_factory in envs:
        env_results = []
        for seed in range(n_seeds):
            env = env_factory()
            agent = QuantumSNNAgent.for_env(
                env,
                mode="triggered",
                seed=seed * 1000,
                ticks_per_decision=config.ticks_per_decision,
                hidden_layers=[config.hidden_size],  # Single layer!
                trigger_threshold=config.trigger_threshold,
                trigger_count=config.trigger_count,
            )

            successes = []
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    agent.learn(reward)
                agent.end_episode()
                successes.append(info.get("success", False))

            final = np.mean(successes[-50:]) * 100
            env_results.append(final)

        results["envs"][env_name] = np.mean(env_results)

    results["score"] = np.mean(list(results["envs"].values()))
    return results


def main():
    envs = [
        ("HiddenGoal", lambda: HiddenGoalMaze(size=6, max_steps=100)),
        ("Deceptive", lambda: DeceptiveMaze(size=10, max_steps=200)),
        ("LongHorizon", lambda: LongHorizonSearch(size=8, n_checkpoints=3, max_steps=200)),
    ]

    print("=" * 70)
    print("SINGLE-LAYER ARCHITECTURE OPTIMIZATION")
    print("=" * 70)

    # Test different single-layer sizes with optimal-ish threshold/count
    configs = []

    # Grid search single layer sizes
    for hs in [8, 12, 16, 20, 24, 28, 32, 40, 48]:
        configs.append(Config(50, 3, 20, hs))

    # Fine-tune threshold/count with best sizes
    for th in [30, 40, 50, 60, 70]:
        for tc in [2, 3, 4, 5]:
            configs.append(Config(th, tc, 20, 24))

    # Remove duplicates
    seen = set()
    unique = []
    for c in configs:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    configs = unique

    print(f"Testing {len(configs)} configurations")
    print("=" * 70)

    all_results = []
    best_score = 0
    best_config = None

    start_total = time.time()

    for i, config in enumerate(configs):
        start = time.time()
        results = evaluate(config, envs, n_episodes=300, n_seeds=5)
        elapsed = time.time() - start

        all_results.append(results)

        if results["score"] > best_score:
            best_score = results["score"]
            best_config = config
            marker = "*** BEST ***"
        else:
            marker = ""

        print(f"[{i+1:2}/{len(configs)}] {config}")
        print(f"       HG={results['envs']['HiddenGoal']:5.1f}%  "
              f"DM={results['envs']['Deceptive']:5.1f}%  "
              f"LH={results['envs']['LongHorizon']:5.1f}%  "
              f"AVG={results['score']:5.1f}%  ({elapsed:.1f}s) {marker}")

    total_time = time.time() - start_total

    # Sort by score
    all_results.sort(key=lambda x: -x["score"])

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Rank':<5} {'Score':>7} {'HiddenGoal':>11} {'Deceptive':>10} {'LongHorizon':>12}  Config")
    print("-" * 80)

    for i, r in enumerate(all_results[:10]):
        c = r["config"]
        print(f"{i+1:<5} {r['score']:>6.1f}% {r['envs']['HiddenGoal']:>10.1f}% "
              f"{r['envs']['Deceptive']:>9.1f}% {r['envs']['LongHorizon']:>11.1f}%  {c}")

    # Best config
    best = all_results[0]
    c = best["config"]

    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"  hidden_layers:      [{c.hidden_size}]  (SINGLE LAYER)")
    print(f"  trigger_threshold:  {c.trigger_threshold}")
    print(f"  trigger_count:      {c.trigger_count}")
    print(f"  ticks_per_decision: {c.ticks_per_decision}")
    print(f"\n  Overall Score:      {best['score']:.1f}%")
    print(f"    HiddenGoal:  {best['envs']['HiddenGoal']:.1f}%")
    print(f"    Deceptive:   {best['envs']['Deceptive']:.1f}%")
    print(f"    LongHorizon: {best['envs']['LongHorizon']:.1f}%")

    # Compare to old baseline
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)
    print(f"  Old baseline ([16,8], th=60, tc=3):  ~29%")
    print(f"  New optimized:                        {best['score']:.1f}%")
    print(f"  Improvement:                         +{best['score'] - 29:.1f}%  ({(best['score']/29 - 1)*100:.0f}% relative)")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\n" + "=" * 70)
    print("RECOMMENDED DEFAULTS FOR QuantumSNNAgent")
    print("=" * 70)
    print(f"""
    agent = QuantumSNNAgent.for_env(
        env,
        mode="triggered",
        hidden_layers=[{c.hidden_size}],      # Single layer is key!
        trigger_threshold={c.trigger_threshold},
        trigger_count={c.trigger_count},
        ticks_per_decision={c.ticks_per_decision},
    )
    """)

    return all_results


if __name__ == "__main__":
    main()
