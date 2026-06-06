#!/usr/bin/env python3
"""Test MountainCar with LEARNABLE encoding.

This encoding provides:
- Position and velocity to ALL channels (no isolation)
- But different MAGNITUDES per channel based on state
- The network must learn which channel to strengthen

This tests if R-STDP can learn when given the same information
but with different "biases" per channel.
"""

import gymnasium as gym
import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode


def encode_observation_learnable(obs: np.ndarray) -> list:
    """Learnable encoding: all channels get info, but with different emphasis.

    Each channel receives position and velocity, but:
    - ch0 (left action): emphasizes negative velocity
    - ch1 (no action): emphasizes near-zero velocity
    - ch2 (right action): emphasizes positive velocity

    The emphasis is SUBTLE - not complete isolation.
    Network must learn to amplify the correct channel.
    """
    position, velocity = obs

    # Base rates from normalized state
    pos_norm = (position + 1.2) / 1.8  # [0, 1]
    vel_norm = (velocity + 0.07) / 0.14  # [0, 1]

    base_pos = int(100 + pos_norm * 100)  # [100, 200]
    base_vel = int(100 + vel_norm * 100)  # [100, 200]

    # Velocity-based bias (subtle, not complete isolation)
    # When velocity is negative, ch0 gets +30 boost
    # When velocity is positive, ch2 gets +30 boost
    # When velocity is near zero, ch1 gets +30 boost

    vel_bias = 40  # Subtle bias, not complete isolation

    if velocity < -0.01:
        ch0_boost = vel_bias
        ch1_boost = 0
        ch2_boost = 0
    elif velocity > 0.01:
        ch0_boost = 0
        ch1_boost = 0
        ch2_boost = vel_bias
    else:
        ch0_boost = 0
        ch1_boost = vel_bias
        ch2_boost = 0

    # All channels get state info, but with different boosts
    ch0_rates = [base_pos + ch0_boost, base_vel + ch0_boost]
    ch1_rates = [base_pos + ch1_boost, base_vel + ch1_boost]
    ch2_rates = [base_pos + ch2_boost, base_vel + ch2_boost]

    return ch0_rates + ch1_rates + ch2_rates


def get_optimal_action(obs: np.ndarray) -> int:
    """Get the momentum-building optimal action."""
    position, velocity = obs
    if velocity < -0.005:
        return 0
    elif velocity > 0.005:
        return 2
    else:
        return 2 if position < -0.5 else 0


def test_mountaincar_learnable():
    """Test MountainCar with learnable encoding."""
    print("=" * 60)
    print("MountainCar with LEARNABLE Encoding")
    print("(All channels get info, subtle velocity bias)")
    print("=" * 60)

    # Test encoding
    print("\nTesting learnable encoding:")
    test_cases = [
        (np.array([-0.5, -0.04]), "Moving left (v=-0.04)"),
        (np.array([-0.5, 0.0]),   "Stationary (v=0)"),
        (np.array([-0.5, 0.04]),  "Moving right (v=0.04)"),
    ]

    print("\n  All channels active, but with velocity-based bias:")
    for obs, desc in test_cases:
        enc = encode_observation_learnable(obs)
        optimal = get_optimal_action(obs)
        print(f"  {desc:25s} -> ch0={enc[:2]}, ch1={enc[2:4]}, ch2={enc[4:]} | optimal={optimal}")

    print("\n  Note: Channels differ by ~40 in boosted channel")
    print("  Network must learn to follow the velocity hint")

    n_inputs = 2
    n_outputs = 3

    config = QuantumSNNConfig(
        n_inputs=n_inputs,
        hidden_layers=[32],
        n_outputs=n_outputs,
        mode=InterferenceMode.soft_mix(),
        ticks_per_decision=20,
        learning_enabled=True,
        duplicate_inputs=True,
    )

    brain = QuantumSNN(config, seed=42)
    env = gym.make("MountainCar-v0")

    n_episodes = 300
    steps_history = []
    accuracy_history = []

    print(f"\nRunning {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        steps = 0
        done = False
        correct_count = 0
        total_decisions = 0

        while not done:
            inputs = encode_observation_learnable(obs)

            _ = brain.decide(inputs)
            action = brain.classical_action()

            optimal = get_optimal_action(obs)
            if action == optimal:
                correct_count += 1
            total_decisions += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Learning signal
            if action == optimal:
                learning_signal = 70
            else:
                learning_signal = 0

            if terminated and next_obs[0] >= 0.5:
                learning_signal = 127

            brain.learn_channeled(learning_signal)
            obs = next_obs

        steps_history.append(steps)
        accuracy = correct_count / total_decisions if total_decisions > 0 else 0
        accuracy_history.append(accuracy)

        if (episode + 1) % 30 == 0:
            recent_avg_steps = np.mean(steps_history[-30:])
            recent_accuracy = np.mean(accuracy_history[-30:])
            recent_successes = sum(1 for s in steps_history[-30:] if s < 200)
            print(f"  Episode {episode + 1:3d}: steps={steps:3d}, "
                  f"last_30_avg={recent_avg_steps:.1f}, "
                  f"accuracy={recent_accuracy:.1%}, "
                  f"successes={recent_successes}/30")

    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    first_30_avg = np.mean(steps_history[:30])
    last_30_avg = np.mean(steps_history[-30:])
    first_30_acc = np.mean(accuracy_history[:30])
    last_30_acc = np.mean(accuracy_history[-30:])
    total_successes = sum(1 for s in steps_history if s < 200)

    print(f"\nPerformance:")
    print(f"  First 30 episodes avg steps: {first_30_avg:.1f}")
    print(f"  Last 30 episodes avg steps:  {last_30_avg:.1f}")
    print(f"  Total successes:             {total_successes}/{n_episodes}")

    print(f"\nAction Accuracy:")
    print(f"  First 30 episodes: {first_30_acc:.1%}")
    print(f"  Last 30 episodes:  {last_30_acc:.1%}")
    print(f"  Random baseline:   ~33%")

    acc_improvement = last_30_acc - first_30_acc
    print(f"\nAccuracy improvement: {acc_improvement:+.1%}")

    if last_30_acc > 0.6:
        print("\nSUCCESS: Network learned to follow velocity hints")
    elif last_30_acc > first_30_acc + 0.1:
        print("\nPARTIAL: Some learning detected")
    else:
        print("\nNO LEARNING: R-STDP cannot learn from subtle hints")

    return steps_history, accuracy_history


if __name__ == "__main__":
    test_mountaincar_learnable()
