#!/usr/bin/env python3
"""Test MountainCar with NAIVE isolated encoding.

This encoding provides channel isolation based on POSITION ZONES,
not optimal action. The network must LEARN which action to take.

Position zones:
- Zone 0 (left):   pos < -0.7  -> ch0 active
- Zone 1 (middle): -0.7 <= pos < -0.3 -> ch1 active
- Zone 2 (right):  pos >= -0.3 -> ch2 active

The velocity is encoded as magnitude in the active channel.
The network must learn: zone + velocity direction -> correct action
"""

import gymnasium as gym
import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode


def encode_observation_naive(obs: np.ndarray) -> list:
    """Naive encoding: isolate by position zone, encode velocity magnitude.

    Does NOT encode the optimal action - network must learn the mapping.
    """
    position, velocity = obs

    # Normalize position to [0, 1] range: -1.2 to 0.6 -> 0 to 1
    pos_norm = (position + 1.2) / 1.8

    # Normalize velocity to [0, 1] range: -0.07 to 0.07 -> 0 to 1
    vel_norm = (velocity + 0.07) / 0.14

    # Encode as spike rates [100-250]
    pos_rate = int(100 + pos_norm * 150)
    vel_rate = int(100 + vel_norm * 150)

    # Determine zone based on position (not action!)
    if position < -0.7:
        zone = 0  # Left zone
    elif position < -0.3:
        zone = 1  # Middle zone (valley)
    else:
        zone = 2  # Right zone (near goal)

    # COMPLETE ISOLATION: only active zone gets input
    ch0_rates = [0, 0]
    ch1_rates = [0, 0]
    ch2_rates = [0, 0]

    if zone == 0:
        ch0_rates = [pos_rate, vel_rate]
    elif zone == 1:
        ch1_rates = [pos_rate, vel_rate]
    else:
        ch2_rates = [pos_rate, vel_rate]

    return ch0_rates + ch1_rates + ch2_rates


def get_optimal_action(obs: np.ndarray) -> int:
    """Get the momentum-building optimal action."""
    position, velocity = obs
    vel_threshold = 0.001

    if velocity < -vel_threshold:
        return 0  # Moving left, push left
    elif velocity > vel_threshold:
        return 2  # Moving right, push right
    else:
        return 2 if position < -0.5 else 0


def test_mountaincar_naive():
    """Test MountainCar with naive isolated encoding."""
    print("=" * 60)
    print("MountainCar with NAIVE Isolated Encoding")
    print("(Channel isolation by POSITION ZONE, not optimal action)")
    print("=" * 60)

    # Test encoding - show it doesn't encode the answer
    print("\nTesting naive encoding:")
    test_cases = [
        (np.array([-0.9, -0.03]), "Left zone, moving left"),
        (np.array([-0.9, 0.03]),  "Left zone, moving right"),
        (np.array([-0.5, -0.03]), "Middle zone, moving left"),
        (np.array([-0.5, 0.03]),  "Middle zone, moving right"),
        (np.array([-0.1, -0.03]), "Right zone, moving left"),
        (np.array([-0.1, 0.03]),  "Right zone, moving right"),
    ]

    print("\n  Position zones determine channel, velocity encoded as magnitude:")
    for obs, desc in test_cases:
        enc = encode_observation_naive(obs)
        optimal = get_optimal_action(obs)
        print(f"  {desc:30s} -> ch0={enc[:2]}, ch1={enc[2:4]}, ch2={enc[4:]} | optimal={optimal}")

    print("\n  Note: Same zone activates regardless of optimal action!")
    print("  The network must LEARN the zone+velocity -> action mapping.")

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

    print(f"\nConfiguration:")
    print(f"  n_inputs: {n_inputs} (duplicated to {n_inputs * n_outputs})")
    print(f"  n_outputs: {n_outputs}")
    print(f"  duplicate_inputs: {config.duplicate_inputs}")

    brain = QuantumSNN(config, seed=42)
    env = gym.make("MountainCar-v0")

    n_episodes = 300  # More episodes since learning is harder
    steps_history = []
    accuracy_history = []

    print(f"\nRunning {n_episodes} episodes...")
    print("(This is harder - network must learn, not just follow encoding)")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        steps = 0
        done = False
        correct_count = 0
        total_decisions = 0

        while not done:
            inputs = encode_observation_naive(obs)

            _ = brain.decide(inputs)
            action = brain.classical_action()

            optimal = get_optimal_action(obs)
            if action == optimal:
                correct_count += 1
            total_decisions += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Learning signal based on outcome
            if action == optimal:
                learning_signal = 70
            else:
                learning_signal = 0

            # Big bonus for reaching goal
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
    overall_avg = np.mean(steps_history)
    best_run = min(steps_history)
    total_successes = sum(1 for s in steps_history if s < 200)

    print(f"\nPerformance (lower steps = better):")
    print(f"  First 30 episodes avg steps: {first_30_avg:.1f}")
    print(f"  Last 30 episodes avg steps:  {last_30_avg:.1f}")
    print(f"  Overall avg steps:           {overall_avg:.1f}")
    print(f"  Best run:                    {best_run}")
    print(f"  Total successes:             {total_successes}/{n_episodes}")
    print(f"  Random baseline:             ~200 (timeout)")

    print(f"\nAction Accuracy (did network learn the mapping?):")
    print(f"  First 30 episodes: {first_30_acc:.1%}")
    print(f"  Last 30 episodes:  {last_30_acc:.1%}")
    print(f"  Random baseline:   ~33%")

    improvement = first_30_avg - last_30_avg
    acc_improvement = last_30_acc - first_30_acc
    print(f"\nImprovement:")
    print(f"  Steps: {improvement:+.1f} (positive = faster)")
    print(f"  Accuracy: {acc_improvement:+.1%} (positive = better)")

    # Verdict
    if last_30_acc > first_30_acc + 0.1:
        print("\nLEARNING DETECTED: Accuracy improved significantly")
    elif last_30_avg < first_30_avg - 10:
        print("\nPARTIAL LEARNING: Steps improved but accuracy flat")
    else:
        print("\nNO LEARNING: Network did not learn the mapping")

    return steps_history, accuracy_history


if __name__ == "__main__":
    test_mountaincar_naive()
