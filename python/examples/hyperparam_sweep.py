"""
Hyperparameter Sweep for Triggered Interference Mode

Optimizes:
- trigger_threshold: Reward threshold for "high reward" detection
- trigger_count: Consecutive high rewards needed to trigger exploitation
- ticks_per_decision: SNN simulation ticks per action
- hidden_layers: Network architecture
- reward_scale: Scaling factor for rewards
"""

import numpy as np
import time
import itertools
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

import gymnasium as gym
from gymnasium import spaces

from tileuniverse.rl import QuantumSNNAgent, train_agent


# ============================================================
# SPARSE REWARD ENVIRONMENTS (compact versions)
# ============================================================

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


# ============================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================

@dataclass
class HyperConfig:
    trigger_threshold: int
    trigger_count: int
    ticks_per_decision: int
    hidden_layers: List[int]
    reward_scale: float

    def to_dict(self):
        return {
            "trigger_threshold": self.trigger_threshold,
            "trigger_count": self.trigger_count,
            "ticks_per_decision": self.ticks_per_decision,
            "hidden_layers": self.hidden_layers,
            "reward_scale": self.reward_scale,
        }

    def __str__(self):
        return f"th={self.trigger_threshold},tc={self.trigger_count},tpd={self.ticks_per_decision},hl={self.hidden_layers},rs={self.reward_scale}"


# Define search space
TRIGGER_THRESHOLDS = [30, 50, 70, 90]
TRIGGER_COUNTS = [2, 3, 5, 8]
TICKS_PER_DECISION = [10, 20, 30]
HIDDEN_LAYERS_OPTIONS = [
    [8],
    [16],
    [16, 8],
    [32, 16],
    [32, 16, 8],
]
REWARD_SCALES = [0.5, 1.0, 2.0]


def generate_configs(quick=False) -> List[HyperConfig]:
    """Generate hyperparameter configurations to test."""
    if quick:
        # Quick sweep: fewer combinations
        configs = []
        for th in [30, 60, 90]:
            for tc in [2, 5]:
                for tpd in [15, 25]:
                    for hl in [[16], [16, 8], [32, 16]]:
                        configs.append(HyperConfig(th, tc, tpd, hl, 1.0))
        return configs

    # Full grid search (warning: many combinations!)
    configs = []
    for th, tc, tpd, hl, rs in itertools.product(
        TRIGGER_THRESHOLDS, TRIGGER_COUNTS, TICKS_PER_DECISION,
        HIDDEN_LAYERS_OPTIONS, REWARD_SCALES
    ):
        configs.append(HyperConfig(th, tc, tpd, hl, rs))
    return configs


# ============================================================
# EVALUATION
# ============================================================

def evaluate_config(config: HyperConfig, envs: List, n_episodes=200, n_seeds=3) -> Dict:
    """Evaluate a single hyperparameter configuration."""
    results = {"config": config.to_dict(), "envs": {}}

    for env_name, env_factory in envs:
        env_results = []

        for seed in range(n_seeds):
            env = env_factory()
            agent = QuantumSNNAgent.for_env(
                env,
                mode="triggered",
                seed=seed * 1000,
                ticks_per_decision=config.ticks_per_decision,
                hidden_layers=config.hidden_layers,
                trigger_threshold=config.trigger_threshold,
                trigger_count=config.trigger_count,
                reward_scale=config.reward_scale,
            )

            successes = []
            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    agent.learn(reward)
                agent.end_episode()
                successes.append(info.get("success", False))

            # Final success rate (last 50 episodes)
            final_success = np.mean(successes[-50:]) * 100
            env_results.append(final_success)

        results["envs"][env_name] = {
            "mean": np.mean(env_results),
            "std": np.std(env_results),
            "raw": env_results,
        }

    # Compute overall score (average across environments)
    results["overall_mean"] = np.mean([r["mean"] for r in results["envs"].values()])
    results["overall_std"] = np.mean([r["std"] for r in results["envs"].values()])

    return results


def run_sweep(quick=True, n_episodes=200, n_seeds=3):
    """Run hyperparameter sweep."""

    # Environments to test
    envs = [
        ("HiddenGoal", lambda: HiddenGoalMaze(size=6, max_steps=100)),
        ("Deceptive", lambda: DeceptiveMaze(size=10, max_steps=200)),
        ("LongHorizon", lambda: LongHorizonSearch(size=8, n_checkpoints=3, max_steps=200)),
    ]

    configs = generate_configs(quick=quick)

    print("=" * 70)
    print("HYPERPARAMETER SWEEP FOR TRIGGERED MODE")
    print("=" * 70)
    print(f"Configurations to test: {len(configs)}")
    print(f"Episodes per config: {n_episodes}")
    print(f"Seeds per config: {n_seeds}")
    print(f"Environments: {[e[0] for e in envs]}")
    print("=" * 70)

    all_results = []
    best_score = 0
    best_config = None

    start_total = time.time()

    for i, config in enumerate(configs):
        start = time.time()

        results = evaluate_config(config, envs, n_episodes, n_seeds)
        all_results.append(results)

        elapsed = time.time() - start

        # Check if best
        if results["overall_mean"] > best_score:
            best_score = results["overall_mean"]
            best_config = config

        # Progress output
        print(f"\n[{i+1}/{len(configs)}] {config}")
        print(f"  HiddenGoal: {results['envs']['HiddenGoal']['mean']:5.1f}% +/- {results['envs']['HiddenGoal']['std']:.1f}")
        print(f"  Deceptive:  {results['envs']['Deceptive']['mean']:5.1f}% +/- {results['envs']['Deceptive']['std']:.1f}")
        print(f"  LongHorizon:{results['envs']['LongHorizon']['mean']:5.1f}% +/- {results['envs']['LongHorizon']['std']:.1f}")
        print(f"  OVERALL:    {results['overall_mean']:5.1f}%  (time: {elapsed:.1f}s)")

        if results["overall_mean"] == best_score:
            print(f"  *** NEW BEST ***")

    total_time = time.time() - start_total

    # Sort by overall score
    all_results.sort(key=lambda x: -x["overall_mean"])

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Rank':<5} {'Score':>7} {'Threshold':>10} {'Count':>6} {'Ticks':>6} {'Layers':>15} {'Scale':>6}")
    print("-" * 70)

    for i, r in enumerate(all_results[:10]):
        c = r["config"]
        print(f"{i+1:<5} {r['overall_mean']:>6.1f}% {c['trigger_threshold']:>10} {c['trigger_count']:>6} "
              f"{c['ticks_per_decision']:>6} {str(c['hidden_layers']):>15} {c['reward_scale']:>6}")

    # Best config analysis
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    best_result = all_results[0]
    print(f"  trigger_threshold:  {best_result['config']['trigger_threshold']}")
    print(f"  trigger_count:      {best_result['config']['trigger_count']}")
    print(f"  ticks_per_decision: {best_result['config']['ticks_per_decision']}")
    print(f"  hidden_layers:      {best_result['config']['hidden_layers']}")
    print(f"  reward_scale:       {best_result['config']['reward_scale']}")
    print(f"\n  Overall Score:      {best_result['overall_mean']:.1f}%")
    print(f"\n  Per-environment:")
    for env_name, env_result in best_result["envs"].items():
        print(f"    {env_name}: {env_result['mean']:.1f}% +/- {env_result['std']:.1f}")

    # Compare to baseline (our previous default)
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)

    # Find baseline config
    baseline = None
    for r in all_results:
        c = r["config"]
        if (c["trigger_threshold"] == 60 and c["trigger_count"] == 3 and
            c["ticks_per_decision"] == 20 and c["hidden_layers"] == [16, 8]):
            baseline = r
            break

    if baseline:
        improvement = best_result["overall_mean"] - baseline["overall_mean"]
        print(f"  Baseline (th=60, tc=3, tpd=20, hl=[16,8]): {baseline['overall_mean']:.1f}%")
        print(f"  Best found:                                {best_result['overall_mean']:.1f}%")
        print(f"  Improvement:                               +{improvement:.1f}%")
    else:
        print("  Baseline config not in sweep (run full sweep to compare)")

    print(f"\n  Total sweep time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results
    results_file = "hyperparam_sweep_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "best": best_result,
            "top10": all_results[:10],
            "all": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)

    return all_results, best_result


def run_focused_sweep():
    """Run a more focused sweep around promising regions."""

    envs = [
        ("HiddenGoal", lambda: HiddenGoalMaze(size=6, max_steps=100)),
        ("Deceptive", lambda: DeceptiveMaze(size=10, max_steps=200)),
        ("LongHorizon", lambda: LongHorizonSearch(size=8, n_checkpoints=3, max_steps=200)),
    ]

    # Focused search based on intuition:
    # - Lower threshold = trigger sooner (more exploitation)
    # - Higher threshold = explore longer
    # - Fewer trigger_count = switch faster
    # - More ticks = more accurate but slower

    configs = []

    # Explore the threshold/count tradeoff
    for th in [20, 40, 60, 80]:
        for tc in [2, 3, 4, 5, 7]:
            configs.append(HyperConfig(th, tc, 20, [16, 8], 1.0))

    # Explore network size with best threshold/count
    for hl in [[8], [16], [24], [16, 8], [24, 12], [32, 16], [32, 16, 8]]:
        configs.append(HyperConfig(50, 3, 20, hl, 1.0))

    # Explore ticks
    for tpd in [10, 15, 20, 25, 30, 40]:
        configs.append(HyperConfig(50, 3, tpd, [16, 8], 1.0))

    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    configs = unique_configs

    print("=" * 70)
    print("FOCUSED HYPERPARAMETER SWEEP")
    print("=" * 70)
    print(f"Configurations to test: {len(configs)}")
    print("=" * 70)

    all_results = []
    best_score = 0

    start_total = time.time()

    for i, config in enumerate(configs):
        start = time.time()
        results = evaluate_config(config, envs, n_episodes=250, n_seeds=3)
        all_results.append(results)
        elapsed = time.time() - start

        if results["overall_mean"] > best_score:
            best_score = results["overall_mean"]
            marker = "*** BEST ***"
        else:
            marker = ""

        print(f"[{i+1:2}/{len(configs)}] {config}")
        print(f"       Score: {results['overall_mean']:5.1f}%  ({elapsed:.1f}s) {marker}")

    total_time = time.time() - start_total

    # Sort and display results
    all_results.sort(key=lambda x: -x["overall_mean"])

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)

    for i, r in enumerate(all_results[:10]):
        c = r["config"]
        print(f"{i+1:2}. {r['overall_mean']:5.1f}%  th={c['trigger_threshold']:2}, tc={c['trigger_count']}, "
              f"tpd={c['ticks_per_decision']:2}, hl={str(c['hidden_layers']):<12}")

    # Detailed best config
    best = all_results[0]
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION FOUND")
    print("=" * 70)
    print(f"  trigger_threshold:  {best['config']['trigger_threshold']}")
    print(f"  trigger_count:      {best['config']['trigger_count']}")
    print(f"  ticks_per_decision: {best['config']['ticks_per_decision']}")
    print(f"  hidden_layers:      {best['config']['hidden_layers']}")
    print(f"  reward_scale:       {best['config']['reward_scale']}")
    print(f"\n  Overall Score:      {best['overall_mean']:.1f}%")
    for env_name, env_result in best["envs"].items():
        print(f"    {env_name}: {env_result['mean']:.1f}%")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return all_results, best


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("Running FULL sweep (this will take a while)...")
        all_results, best = run_sweep(quick=False, n_episodes=200, n_seeds=3)
    elif len(sys.argv) > 1 and sys.argv[1] == "focused":
        print("Running FOCUSED sweep...")
        all_results, best = run_focused_sweep()
    else:
        print("Running QUICK sweep (use 'full' or 'focused' for more thorough search)...")
        all_results, best = run_sweep(quick=True, n_episodes=150, n_seeds=2)
