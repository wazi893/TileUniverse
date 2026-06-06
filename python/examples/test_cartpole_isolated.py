#!/usr/bin/env python3
"""Test CartPole with TRULY ISOLATED encoding (like bandit test).

Key changes from test_cartpole_duplicated.py:
1. Complete isolation: 0 vs 200 (not 30 vs 200)
2. Include velocity in decision
3. No ambiguous center zone
"""

import gymnasium as gym
import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode


def encode_observation_isolated(obs: np.ndarray) -> list:
    """Encode with COMPLETE isolation like the bandit test.

    Uses angle + velocity to determine which channel activates.
    Only ONE channel gets input, the other gets ZERO.
    """
    cart_pos, cart_vel, pole_angle, pole_vel = obs

    # Combine angle and velocity for better control
    # If angle and velocity have same sign: falling faster -> stronger signal
    # If angle and velocity have opposite sign: recovering -> weaker signal
    urgency = pole_angle + 0.5 * pole_vel

    # COMPLETE ISOLATION: one channel ON (200), other OFF (0)
    if urgency > 0:
        # Need to push RIGHT (action 1)
        # Channel 1 gets ALL input, channel 0 gets NOTHING
        ch0_rates = [0, 0, 0, 0]
        magnitude = min(250, int(150 + abs(urgency) * 200))
        ch1_rates = [magnitude, magnitude, magnitude, magnitude]
    else:
        # Need to push LEFT (action 0)
        # Channel 0 gets ALL input, channel 1 gets NOTHING
        magnitude = min(250, int(150 + abs(urgency) * 200))
        ch0_rates = [magnitude, magnitude, magnitude, magnitude]
        ch1_rates = [0, 0, 0, 0]

    return ch0_rates + ch1_rates


def test_cartpole_isolated():
    """Test CartPole with truly isolated encoding."""
    print("=" * 60)
    print("CartPole with TRULY ISOLATED Encoding")
    print("(Like the bandit test that achieved 100%)")
    print("=" * 60)

    # Test encoding
    print("\nTesting isolated encoding:")
    test_right = np.array([0.0, 0.0, 0.1, 0.2])  # tilting right, falling
    test_left = np.array([0.0, 0.0, -0.1, -0.2])  # tilting left, falling
    test_recovering = np.array([0.0, 0.0, 0.1, -0.3])  # tilting right but recovering

    enc_right = encode_observation_isolated(test_right)
    enc_left = encode_observation_isolated(test_left)
    enc_recover = encode_observation_isolated(test_recovering)

    print(f"  Falling RIGHT:     ch0={enc_right[:4]}, ch1={enc_right[4:]}")
    print(f"  Falling LEFT:      ch0={enc_left[:4]}, ch1={enc_left[4:]}")
    print(f"  Recovering (right): ch0={enc_recover[:4]}, ch1={enc_recover[4:]}")
    print(f"  Expected: Complete isolation (one channel 0, other ~200)")

    n_inputs = 4
    n_outputs = 2

    config = QuantumSNNConfig(
        n_inputs=n_inputs,
        hidden_layers=[32],
        n_outputs=n_outputs,
        mode=InterferenceMode.soft_mix(),  # Mode irrelevant when using classical_action()
        ticks_per_decision=20,
        learning_enabled=True,
        duplicate_inputs=True,
    )

    print(f"\nConfiguration:")
    print(f"  duplicate_inputs: {config.duplicate_inputs}")
    print(f"  Using classical WTA (best performance)")

    brain = QuantumSNN(config, seed=42)
    env = gym.make("CartPole-v1")

    n_episodes = 200
    rewards_history = []

    print(f"\nRunning {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        done = False
        correct_count = 0
        wrong_count = 0

        while not done:
            inputs = encode_observation_isolated(obs)

            # Use classical WTA for action selection (best performance)
            _ = brain.decide(inputs)
            action = brain.classical_action()

            # Track correctness using urgency metric
            urgency = obs[2] + 0.5 * obs[3]
            correct_action = 1 if urgency > 0 else 0
            if action == correct_action:
                correct_count += 1
            else:
                wrong_count += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Selective reinforcement
            if action == correct_action:
                learning_signal = int(50 + abs(urgency) * 100)
                learning_signal = min(100, learning_signal)
            else:
                learning_signal = 0

            brain.learn_channeled(learning_signal)

            obs = next_obs
            total_reward += reward

        rewards_history.append(total_reward)
        accuracy = correct_count / (correct_count + wrong_count) if (correct_count + wrong_count) > 0 else 0

        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(rewards_history[-20:])
            print(f"  Episode {episode + 1:3d}: reward={total_reward:3.0f}, "
                  f"last_20_avg={recent_avg:.1f}, accuracy={accuracy:.1%}")

    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    first_20_avg = np.mean(rewards_history[:20])
    last_20_avg = np.mean(rewards_history[-20:])
    overall_avg = np.mean(rewards_history)
    best_run = max(rewards_history)

    print(f"\nPerformance:")
    print(f"  First 20 episodes avg: {first_20_avg:.1f}")
    print(f"  Last 20 episodes avg:  {last_20_avg:.1f}")
    print(f"  Overall avg:           {overall_avg:.1f}")
    print(f"  Best run:              {best_run:.0f}")
    print(f"  Random baseline:       ~20-25")

    improvement = last_20_avg - first_20_avg
    print(f"\nImprovement: {improvement:+.1f}")

    if last_20_avg > 100:
        print("\nEXCELLENT: Solving CartPole (avg > 100)")
    elif last_20_avg > 50:
        print("\nGOOD: Significant learning (avg > 50)")
    elif last_20_avg > first_20_avg + 10:
        print("\nPARTIAL: Some improvement detected")
    else:
        print("\nNOT LEARNING: No significant improvement")

    return rewards_history


if __name__ == "__main__":
    test_cartpole_isolated()
