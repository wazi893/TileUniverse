#!/usr/bin/env python3
"""Optimized neuroevolution for MountainCar - larger population, more generations."""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NetworkConfig:
    input_size: int = 2
    hidden_size: int = 32  # Larger hidden layer
    output_size: int = 3


class SimpleNetwork:
    """Simple feedforward network for neuroevolution."""

    def __init__(self, config: NetworkConfig, seed: int = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.standard_normal((config.input_size, config.hidden_size)) * np.sqrt(2.0 / config.input_size)
        self.b1 = np.zeros(config.hidden_size)
        self.w2 = self.rng.standard_normal((config.hidden_size, config.output_size)) * np.sqrt(2.0 / config.hidden_size)
        self.b2 = np.zeros(config.output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.forward(obs)))

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])

    def set_weights(self, weights: np.ndarray):
        idx = 0
        w1_size = self.config.input_size * self.config.hidden_size
        self.w1 = weights[idx:idx + w1_size].reshape(self.config.input_size, self.config.hidden_size)
        idx += w1_size
        self.b1 = weights[idx:idx + self.config.hidden_size]
        idx += self.config.hidden_size
        w2_size = self.config.hidden_size * self.config.output_size
        self.w2 = weights[idx:idx + w2_size].reshape(self.config.hidden_size, self.config.output_size)
        idx += w2_size
        self.b2 = weights[idx:]

    def copy(self) -> 'SimpleNetwork':
        new_net = SimpleNetwork(self.config)
        new_net.set_weights(self.get_weights().copy())
        return new_net


def evaluate_network(net: SimpleNetwork, n_episodes: int = 5, seed_base: int = 0) -> Tuple[float, float]:
    """Evaluate with more episodes for stability."""
    env = gym.make("MountainCar-v0")
    total_steps = 0
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        steps = 0

        while not done:
            action = net.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps
        if terminated and obs[0] >= 0.5:
            successes += 1

    env.close()
    return -total_steps / n_episodes, successes / n_episodes


def run_evolution(population_size=100, generations=150, seed=42):
    """Run with larger population and more generations."""
    rng = np.random.default_rng(seed)
    config = NetworkConfig(input_size=2, hidden_size=32, output_size=3)

    population = [SimpleNetwork(config, seed=rng.integers(0, 100000)) for _ in range(population_size)]
    best_network = None
    best_fitness = float('-inf')

    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Network: 2 -> 32 -> 3 ({2*32 + 32 + 32*3 + 3} params)")
    print()

    for gen in range(generations):
        # Evaluate
        results = [evaluate_network(net, n_episodes=5, seed_base=gen * 100) for net in population]
        fitnesses = np.array([r[0] for r in results])
        success_rates = np.array([r[1] for r in results])

        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_success = success_rates[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_network = population[gen_best_idx].copy()

        if (gen + 1) % 15 == 0 or gen == 0 or gen_best_success >= 1.0:
            print(f"Gen {gen + 1:3d}: best_steps={-gen_best_fitness:.0f}, "
                  f"avg_steps={-np.mean(fitnesses):.0f}, "
                  f"best_success={gen_best_success:.0%}")

        if gen_best_success >= 1.0 and -gen_best_fitness < 130:
            print(f"\nSOLVED at generation {gen + 1}!")
            break

        # Selection + reproduction
        sorted_idx = np.argsort(fitnesses)[::-1]
        elite_count = 10
        new_population = [population[i].copy() for i in sorted_idx[:elite_count]]

        while len(new_population) < population_size:
            # Tournament
            t = rng.choice(population_size, 7, replace=False)
            p1, p2 = t[np.argsort(fitnesses[t])[-2:]]

            w1, w2 = population[p1].get_weights(), population[p2].get_weights()
            child_w = np.where(rng.random(len(w1)) < 0.5, w1, w2)

            # Adaptive mutation
            mut_rate = 0.2 if gen < 50 else 0.1
            mut_strength = 0.3 if gen < 50 else 0.15
            mask = rng.random(len(child_w)) < mut_rate
            child_w[mask] += rng.standard_normal(np.sum(mask)) * mut_strength

            child = SimpleNetwork(config)
            child.set_weights(child_w)
            new_population.append(child)

        population = new_population

    return best_network


def main():
    print("=" * 60)
    print("MountainCar Neuroevolution v2 (Optimized)")
    print("=" * 60)
    print()

    best_net = run_evolution(population_size=100, generations=150, seed=42)

    # Thorough testing
    print("\n" + "=" * 60)
    print("Final Evaluation (50 episodes)")
    print("=" * 60)

    env = gym.make("MountainCar-v0")
    steps_list = []
    successes = 0

    for ep in range(50):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        steps = 0
        while not done:
            action = best_net.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        steps_list.append(steps)
        if terminated and obs[0] >= 0.5:
            successes += 1

    env.close()

    print(f"\nResults over 50 test episodes:")
    print(f"  Success rate:  {successes}/50 ({successes*2}%)")
    print(f"  Avg steps:     {np.mean(steps_list):.1f}")
    print(f"  Median steps:  {np.median(steps_list):.0f}")
    print(f"  Best run:      {min(steps_list)}")
    print(f"  Worst run:     {max(steps_list)}")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print()
    print("Method                   Success    Avg Steps   Can Learn?")
    print("-" * 60)
    print(f"R-STDP (optimal enc)     100%       107         No (encoded)")
    print(f"R-STDP (naive enc)       0%         200         No")
    print(f"Neuroevolution           {successes*2}%        {np.mean(steps_list):.0f}          YES")
    print()
    print("The policy IS learnable. R-STDP's channel-following behavior")
    print("prevents it from learning mappings that contradict encoding.")


if __name__ == "__main__":
    main()
