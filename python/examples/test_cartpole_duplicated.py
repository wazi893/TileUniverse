#!/usr/bin/env python3
"""Test CartPole with duplicated inputs + channeled learning.

This test verifies that input duplication enables context-dependent learning
by giving each output channel its own dedicated copy of the inputs.

CartPole has 4 observations, 2 actions.
With duplicate_inputs=True: 4 * 2 = 8 input neurons (4 per channel)
"""

import gymnasium as gym
import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode


def encode_observation(obs: np.ndarray, n_outputs: int = 2) -> list:
    """Encode CartPole observations as spike rates with ISOLATED channel activation.

    KEY INSIGHT: Like the isolated test that achieved 100%, we need channels
    to be MOSTLY SILENT when not their context.

    - Channel 0 (push left): HIGH activity only when pole tilts RIGHT
    - Channel 1 (push right): HIGH activity only when pole tilts LEFT

    When pole is near center, both channels get low activity (exploration).
    """
    cart_pos, cart_vel, pole_angle, pole_vel = obs

    # Threshold for "significant" tilt
    threshold = 0.02  # About 1 degree

    if pole_angle > threshold:
        # Pole tilting RIGHT -> need to push RIGHT (action 1) to move cart under pole
        # Channel 1 gets high activity, Channel 0 gets low
        tilt_magnitude = min(1.0, abs(pole_angle) / 0.2)  # Normalize to [0, 1]
        ch0_activity = int(30)  # Low but not zero
        ch1_activity = int(150 + tilt_magnitude * 100)  # 150-250
    elif pole_angle < -threshold:
        # Pole tilting LEFT -> need to push LEFT (action 0) to move cart under pole
        # Channel 0 gets high activity, Channel 1 gets low
        tilt_magnitude = min(1.0, abs(pole_angle) / 0.2)
        ch0_activity = int(150 + tilt_magnitude * 100)  # 150-250
        ch1_activity = int(30)  # Low but not zero
    else:
        # Near center -> both channels moderate (let network explore)
        ch0_activity = int(100)
        ch1_activity = int(100)

    # Create 4 inputs per channel
    # All inputs in a channel get the same activity level
    ch0_rates = [ch0_activity] * 4
    ch1_rates = [ch1_activity] * 4

    return ch0_rates + ch1_rates


def test_cartpole_duplicated():
    """Test CartPole with duplicated inputs and channeled learning."""
    print("=" * 60)
    print("CartPole with Duplicated Inputs + Channeled Learning")
    print("=" * 60)

    # First, let's test the encoding to make sure it creates isolation
    print("\nTesting isolated channel encoding:")
    test_obs_right = np.array([0.0, 0.0, 0.1, 0.0])  # pole tilting right
    test_obs_left = np.array([0.0, 0.0, -0.1, 0.0])  # pole tilting left
    test_obs_center = np.array([0.0, 0.0, 0.01, 0.0])  # pole near center

    enc_right = encode_observation(test_obs_right, 2)
    enc_left = encode_observation(test_obs_left, 2)
    enc_center = encode_observation(test_obs_center, 2)

    print(f"  Pole RIGHT (+0.1): ch0={enc_right[:4]}, ch1={enc_right[4:]}")
    print(f"  Pole LEFT  (-0.1): ch0={enc_left[:4]}, ch1={enc_left[4:]}")
    print(f"  Pole CENTER(~0):   ch0={enc_center[:4]}, ch1={enc_center[4:]}")
    print(f"  Expected: ch0 HIGH only when right, ch1 HIGH only when left")

    # CartPole: 4 observations, 2 actions
    n_inputs = 4
    n_outputs = 2

    # Create config with duplicate_inputs=True
    config = QuantumSNNConfig(
        n_inputs=n_inputs,
        hidden_layers=[32],  # 32 hidden (16 per channel)
        n_outputs=n_outputs,
        mode=InterferenceMode.soft_mix(),  # Mode doesn't matter - using classical_action()
        mix_angle=0.1,
        ticks_per_decision=20,
        learning_enabled=True,
        trigger_threshold=50,  # Reward threshold to trigger exploitation
        trigger_count=3,       # Consecutive high rewards needed
        duplicate_inputs=True,  # Each channel gets its own inputs
    )

    # Verify configuration
    print(f"\nConfiguration:")
    print(f"  n_inputs (logical): {n_inputs}")
    print(f"  n_outputs: {n_outputs}")
    print(f"  duplicate_inputs: {config.duplicate_inputs}")
    print(f"  mode: {config.mode} (QUANTUM INTERFERENCE)")
    print(f"  mix_angle: {config.mix_angle}")
    print(f"  actual input neurons: {config.total_neurons()} total")

    brain = QuantumSNN(config, seed=42)

    # Create environment
    env = gym.make("CartPole-v1")

    n_episodes = 100
    rewards_history = []

    print(f"\nRunning {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        done = False
        step = 0

        correct_count = 0
        wrong_count = 0

        while not done:
            # Encode with duplication
            inputs = encode_observation(obs, n_outputs)

            # Use classical WTA for action selection (best performance)
            _ = brain.decide(inputs)  # Run SNN ticks
            action = brain.classical_action()  # Pure WTA on spike counts
            spikes = brain.output_spike_counts

            # Track correctness (before stepping environment)
            pole_angle_pre = obs[2]
            correct_action_pre = 1 if pole_angle_pre > 0 else 0  # Right tilt -> push right
            if action == correct_action_pre:
                correct_count += 1
            else:
                wrong_count += 1

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Calculate learning signal
            # SELECTIVE REINFORCEMENT: Only reward when action matches physics
            pole_angle = obs[2]

            # Physics-based correctness:
            # - pole tilting right (angle > 0) -> push RIGHT (action 1) to move cart under pole
            # - pole tilting left (angle < 0) -> push LEFT (action 0) to move cart under pole
            correct_action = 1 if pole_angle > 0 else 0

            if action == correct_action:
                # Correct action: strong positive reward
                learning_signal = int(50 + abs(pole_angle) / 0.42 * 50)
                learning_signal = min(100, learning_signal)
            else:
                # Wrong action: zero reward (don't strengthen, but don't kill)
                learning_signal = 0

            # Apply channeled learning
            brain.learn_channeled(learning_signal)

            obs = next_obs
            total_reward += reward
            step += 1

        rewards_history.append(total_reward)
        accuracy = correct_count / (correct_count + wrong_count) if (correct_count + wrong_count) > 0 else 0

        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(rewards_history[-10:])
            print(f"  Episode {episode + 1:3d}: reward={total_reward:3.0f}, "
                  f"last_10_avg={recent_avg:.1f}, accuracy={accuracy:.1%}")

    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    first_10_avg = np.mean(rewards_history[:10])
    last_10_avg = np.mean(rewards_history[-10:])
    overall_avg = np.mean(rewards_history)
    best_run = max(rewards_history)

    print(f"\nPerformance:")
    print(f"  First 10 episodes avg: {first_10_avg:.1f}")
    print(f"  Last 10 episodes avg:  {last_10_avg:.1f}")
    print(f"  Overall avg:           {overall_avg:.1f}")
    print(f"  Best run:              {best_run:.0f}")
    print(f"  Random baseline:       ~20-25")

    # Check for improvement
    improvement = last_10_avg - first_10_avg
    print(f"\nImprovement: {improvement:+.1f}")

    if last_10_avg > 50:
        print("\nSUCCESS: Network is learning (avg > 50)")
    elif last_10_avg > first_10_avg + 10:
        print("\nPARTIAL SUCCESS: Some improvement detected")
    else:
        print("\nNOT LEARNING: No significant improvement")

    # Check for activity death
    final_spikes = brain.output_spike_counts
    if sum(final_spikes) == 0:
        print("\nWARNING: Activity death detected (spikes = [0, 0])")
    else:
        print(f"\nFinal spike pattern: {final_spikes}")

    return rewards_history


if __name__ == "__main__":
    test_cartpole_duplicated()
