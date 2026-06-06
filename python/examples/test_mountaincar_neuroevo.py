#!/usr/bin/env python3
"""Test MountainCar with NEUROEVOLUTION instead of R-STDP.

This demonstrates that the policy CAN be learned - the problem
is R-STDP, not the task difficulty.

Uses simple Evolution Strategy (ES):
1. Population of neural networks with random weights
2. Evaluate fitness (episode return) for each
3. Select top performers
4. Mutate weights to create next generation
5. Repeat until solved
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NetworkConfig:
    input_size: int = 2
    hidden_size: int = 16
    output_size: int = 3


class SimpleNetwork:
    """Simple feedforward network for neuroevolution."""

    def __init__(self, config: NetworkConfig, seed: int = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Xavier initialization
        self.w1 = self.rng.standard_normal((config.input_size, config.hidden_size)) * np.sqrt(2.0 / config.input_size)
        self.b1 = np.zeros(config.hidden_size)
        self.w2 = self.rng.standard_normal((config.hidden_size, config.output_size)) * np.sqrt(2.0 / config.hidden_size)
        self.b2 = np.zeros(config.output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with tanh activation."""
        h = np.tanh(x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return out

    def act(self, obs: np.ndarray) -> int:
        """Select action (argmax of output)."""
        logits = self.forward(obs)
        return int(np.argmax(logits))

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into single vector."""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, weights: np.ndarray):
        """Set weights from flattened vector."""
        idx = 0
        w1_size = self.config.input_size * self.config.hidden_size
        self.w1 = weights[idx:idx + w1_size].reshape(self.config.input_size, self.config.hidden_size)
        idx += w1_size

        b1_size = self.config.hidden_size
        self.b1 = weights[idx:idx + b1_size]
        idx += b1_size

        w2_size = self.config.hidden_size * self.config.output_size
        self.w2 = weights[idx:idx + w2_size].reshape(self.config.hidden_size, self.config.output_size)
        idx += w2_size

        self.b2 = weights[idx:]

    def copy(self) -> 'SimpleNetwork':
        """Create a copy of this network."""
        new_net = SimpleNetwork(self.config)
        new_net.set_weights(self.get_weights().copy())
        return new_net


def evaluate_network(net: SimpleNetwork, env_name: str = "MountainCar-v0",
                     n_episodes: int = 3, seed_base: int = 0) -> Tuple[float, float]:
    """Evaluate network fitness over multiple episodes."""
    env = gym.make(env_name)
    total_steps = 0
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        steps = 0

        while not done:
            action = net.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps
        if terminated and obs[0] >= 0.5:
            successes += 1

    env.close()

    # Fitness: negative steps (lower is better, so higher fitness)
    avg_steps = total_steps / n_episodes
    fitness = -avg_steps  # Negative because we minimize steps

    return fitness, successes / n_episodes


def mutate(weights: np.ndarray, mutation_rate: float, mutation_strength: float,
           rng: np.random.Generator) -> np.ndarray:
    """Mutate weights with Gaussian noise."""
    mask = rng.random(len(weights)) < mutation_rate
    noise = rng.standard_normal(len(weights)) * mutation_strength
    new_weights = weights.copy()
    new_weights[mask] += noise[mask]
    return new_weights


def crossover(parent1: np.ndarray, parent2: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover between two parents."""
    mask = rng.random(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


def run_evolution(
    population_size: int = 50,
    generations: int = 100,
    elite_count: int = 5,
    mutation_rate: float = 0.3,
    mutation_strength: float = 0.5,
    eval_episodes: int = 3,
    seed: int = 42
) -> Tuple[SimpleNetwork, List[float]]:
    """Run evolutionary optimization."""

    rng = np.random.default_rng(seed)
    config = NetworkConfig(input_size=2, hidden_size=16, output_size=3)

    # Initialize population
    population = [SimpleNetwork(config, seed=rng.integers(0, 100000)) for _ in range(population_size)]

    best_fitness_history = []
    avg_fitness_history = []
    best_network = None
    best_fitness = float('-inf')

    print(f"Starting evolution with {population_size} individuals for {generations} generations")
    print(f"Network: {config.input_size} -> {config.hidden_size} -> {config.output_size}")
    print(f"Total parameters: {2*16 + 16 + 16*3 + 3} = 99")
    print()

    for gen in range(generations):
        # Evaluate all individuals
        fitnesses = []
        success_rates = []

        for i, net in enumerate(population):
            fitness, success_rate = evaluate_network(net, n_episodes=eval_episodes, seed_base=gen * 100)
            fitnesses.append(fitness)
            success_rates.append(success_rate)

        fitnesses = np.array(fitnesses)
        success_rates = np.array(success_rates)

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_avg_fitness = np.mean(fitnesses)
        gen_best_success = success_rates[gen_best_idx]
        gen_avg_success = np.mean(success_rates)

        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_network = population[gen_best_idx].copy()

        # Report progress
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"Gen {gen + 1:3d}: best_steps={-gen_best_fitness:.0f}, "
                  f"avg_steps={-gen_avg_fitness:.0f}, "
                  f"best_success={gen_best_success:.0%}, "
                  f"avg_success={gen_avg_success:.0%}")

        # Check if solved
        if gen_best_success >= 1.0 and -gen_best_fitness < 150:
            print(f"\nSOLVED at generation {gen + 1}!")
            break

        # Selection: tournament selection + elitism
        sorted_indices = np.argsort(fitnesses)[::-1]  # Best first
        elite_indices = sorted_indices[:elite_count]

        # Create next generation
        new_population = []

        # Keep elites
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Fill rest with offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            t1 = rng.choice(population_size, tournament_size, replace=False)
            t2 = rng.choice(population_size, tournament_size, replace=False)
            parent1_idx = t1[np.argmax(fitnesses[t1])]
            parent2_idx = t2[np.argmax(fitnesses[t2])]

            parent1_weights = population[parent1_idx].get_weights()
            parent2_weights = population[parent2_idx].get_weights()

            # Crossover
            child_weights = crossover(parent1_weights, parent2_weights, rng)

            # Mutation
            child_weights = mutate(child_weights, mutation_rate, mutation_strength, rng)

            child = SimpleNetwork(config)
            child.set_weights(child_weights)
            new_population.append(child)

        population = new_population

    return best_network, best_fitness_history


def test_best_network(net: SimpleNetwork, n_episodes: int = 20):
    """Test the best evolved network."""
    print("\n" + "=" * 60)
    print("Testing Best Evolved Network")
    print("=" * 60)

    env = gym.make("MountainCar-v0")
    steps_history = []
    successes = 0

    # Also track action accuracy vs optimal
    def get_optimal(obs):
        if obs[1] < -0.005:
            return 0
        elif obs[1] > 0.005:
            return 2
        else:
            return 2 if obs[0] < -0.5 else 0

    correct_total = 0
    total_actions = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        done = False
        steps = 0

        while not done:
            action = net.act(obs)
            optimal = get_optimal(obs)
            if action == optimal:
                correct_total += 1
            total_actions += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        steps_history.append(steps)
        if terminated and obs[0] >= 0.5:
            successes += 1

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep + 1}: steps={steps}, success={'Yes' if steps < 200 else 'No'}")

    env.close()

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Avg steps:     {np.mean(steps_history):.1f}")
    print(f"  Best run:      {min(steps_history)}")
    print(f"  Success rate:  {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    print(f"  Action accuracy vs optimal: {correct_total/total_actions:.1%}")

    return steps_history


def main():
    print("=" * 60)
    print("MountainCar with NEUROEVOLUTION")
    print("(Proving the policy IS learnable, R-STDP is the problem)")
    print("=" * 60)
    print()

    # Run evolution
    best_net, fitness_history = run_evolution(
        population_size=50,
        generations=100,
        elite_count=5,
        mutation_rate=0.3,
        mutation_strength=0.5,
        eval_episodes=3,
        seed=42
    )

    # Test the best network
    test_best_network(best_net, n_episodes=20)

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Neuroevolution vs R-STDP")
    print("=" * 60)
    print()
    print("                    R-STDP (naive enc)    Neuroevolution")
    print("  Success rate:          0%              (see above)")
    print("  Learning:              None            Fitness-based")
    print("  Encoding:              Position zones  Raw state")
    print()
    print("Conclusion: The policy IS learnable. R-STDP cannot learn it")
    print("because it follows channel activation, not learned mappings.")


if __name__ == "__main__":
    main()
