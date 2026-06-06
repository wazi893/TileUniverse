#!/usr/bin/env python3
"""Test MountainCar with ISOLATED encoding (applying CartPole breakthrough).

MountainCar physics:
- Position: [-1.2, 0.6], Goal: >= 0.5
- Velocity: [-0.07, 0.07]
- Actions: 0=push left, 1=no push, 2=push right

Optimal policy: push in direction of velocity to build momentum
- velocity < 0: push left (action 0)
- velocity > 0: push right (action 2)
- The middle action (1) is rarely optimal
"""

import gymnasium as gym
import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode


def encode_observation_isolated(obs: np.ndarray) -> list:
    """Encode MountainCar with COMPLETE channel isolation.

    Key insight from CartPole: ONE channel must be completely silent (0),
    the active channel gets strong signal (200+).

    For MountainCar, we use velocity to determine push direction.
    """
    position, velocity = obs

    # Velocity threshold for "significant" movement
    vel_threshold = 0.001

    # Position-based urgency: closer to goal = less urgent to swing
    # Further from goal (left side) = more urgent
    position_urgency = max(0, (0.5 - position) / 1.7)  # 0 at goal, 1 at far left

    # Base magnitude scales with how far we are from goal
    base_magnitude = int(150 + position_urgency * 100)

    if velocity < -vel_threshold:
        # Moving LEFT: push LEFT to build momentum (action 0)
        magnitude = min(250, base_magnitude + int(abs(velocity) * 1500))
        ch0_rates = [magnitude, magnitude]  # LEFT active
        ch1_rates = [0, 0]                   # NONE silent
        ch2_rates = [0, 0]                   # RIGHT silent

    elif velocity > vel_threshold:
        # Moving RIGHT: push RIGHT to build momentum (action 2)
        magnitude = min(250, base_magnitude + int(abs(velocity) * 1500))
        ch0_rates = [0, 0]                   # LEFT silent
        ch1_rates = [0, 0]                   # NONE silent
        ch2_rates = [magnitude, magnitude]  # RIGHT active

    else:
        # Near zero velocity: push toward goal (right) to start moving
        # Or if on right side of valley, push left to swing back
        valley_bottom = -0.5
        if position < valley_bottom:
            # Left of valley: push right to start climbing
            ch0_rates = [0, 0]
            ch1_rates = [0, 0]
            ch2_rates = [180, 180]
        else:
            # Right of valley: push left to swing back and build momentum
            ch0_rates = [180, 180]
            ch1_rates = [0, 0]
            ch2_rates = [0, 0]

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
        # Near zero: push based on position relative to valley
        return 2 if position < -0.5 else 0


def test_mountaincar_isolated():
    """Test MountainCar with isolated encoding."""
    print("=" * 60)
    print("MountainCar with ISOLATED Encoding")
    print("(Applying CartPole breakthrough to 3-action environment)")
    print("=" * 60)

    # Test encoding
    print("\nTesting isolated encoding:")
    test_left = np.array([-0.5, -0.03])   # moving left
    test_right = np.array([-0.5, 0.03])   # moving right
    test_zero = np.array([-0.7, 0.0])     # stationary, left of valley

    enc_left = encode_observation_isolated(test_left)
    enc_right = encode_observation_isolated(test_right)
    enc_zero = encode_observation_isolated(test_zero)

    print(f"  Moving LEFT:  ch0={enc_left[:2]}, ch1={enc_left[2:4]}, ch2={enc_left[4:]}")
    print(f"  Moving RIGHT: ch0={enc_right[:2]}, ch1={enc_right[2:4]}, ch2={enc_right[4:]}")
    print(f"  Stationary:   ch0={enc_zero[:2]}, ch1={enc_zero[2:4]}, ch2={enc_zero[4:]}")
    print(f"  Expected: Complete isolation (one channel active, others 0)")

    n_inputs = 2  # position, velocity
    n_outputs = 3  # left, none, right

    config = QuantumSNNConfig(
        n_inputs=n_inputs,
        hidden_layers=[32],
        n_outputs=n_outputs,
        mode=InterferenceMode.soft_mix(),
        ticks_per_decision=20,
        learning_enabled=True,
        duplicate_inputs=True,  # Key: creates 2*3=6 input neurons
    )

    print(f"\nConfiguration:")
    print(f"  n_inputs: {n_inputs} (duplicated to {n_inputs * n_outputs})")
    print(f"  n_outputs: {n_outputs}")
    print(f"  duplicate_inputs: {config.duplicate_inputs}")

    brain = QuantumSNN(config, seed=42)
    env = gym.make("MountainCar-v0")

    n_episodes = 200
    rewards_history = []
    steps_history = []
    successes = 0

    print(f"\nRunning {n_episodes} episodes...")
    print("(MountainCar terminates at 200 steps or when goal reached)")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        steps = 0
        done = False
        correct_count = 0
        wrong_count = 0

        while not done:
            inputs = encode_observation_isolated(obs)

            # Use classical WTA for action selection
            _ = brain.decide(inputs)
            action = brain.classical_action()

            # Track correctness
            optimal = get_optimal_action(obs)
            if action == optimal:
                correct_count += 1
            else:
                wrong_count += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Reward shaping for learning signal
            # MountainCar gives -1 per step, +0 at goal
            # We use positive reinforcement for momentum-building actions
            if action == optimal:
                # Reward correct momentum-building
                learning_signal = 70
            else:
                learning_signal = 0

            # Bonus for reaching goal
            if terminated and next_obs[0] >= 0.5:
                learning_signal = 127  # Maximum positive signal
                successes += 1

            brain.learn_channeled(learning_signal)

            obs = next_obs
            total_reward += reward

        rewards_history.append(total_reward)
        steps_history.append(steps)
        accuracy = correct_count / (correct_count + wrong_count) if (correct_count + wrong_count) > 0 else 0

        if (episode + 1) % 20 == 0:
            recent_avg_steps = np.mean(steps_history[-20:])
            recent_successes = sum(1 for s in steps_history[-20:] if s < 200)
            print(f"  Episode {episode + 1:3d}: steps={steps:3d}, "
                  f"last_20_avg_steps={recent_avg_steps:.1f}, "
                  f"successes_last_20={recent_successes}, "
                  f"accuracy={accuracy:.1%}")

    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    first_20_avg = np.mean(steps_history[:20])
    last_20_avg = np.mean(steps_history[-20:])
    overall_avg = np.mean(steps_history)
    best_run = min(steps_history)
    total_successes = sum(1 for s in steps_history if s < 200)

    print(f"\nPerformance (lower steps = better):")
    print(f"  First 20 episodes avg steps: {first_20_avg:.1f}")
    print(f"  Last 20 episodes avg steps:  {last_20_avg:.1f}")
    print(f"  Overall avg steps:           {overall_avg:.1f}")
    print(f"  Best run (fewest steps):     {best_run}")
    print(f"  Total successes (<200 steps): {total_successes}/{n_episodes}")
    print(f"  Random baseline:             ~200 (timeout)")

    improvement = first_20_avg - last_20_avg  # Lower is better, so positive = improvement
    print(f"\nImprovement: {improvement:+.1f} steps (positive = faster)")

    # Success rate analysis
    first_20_success = sum(1 for s in steps_history[:20] if s < 200)
    last_20_success = sum(1 for s in steps_history[-20:] if s < 200)
    print(f"\nSuccess rate:")
    print(f"  First 20 episodes: {first_20_success}/20 ({first_20_success*5}%)")
    print(f"  Last 20 episodes:  {last_20_success}/20 ({last_20_success*5}%)")

    if last_20_avg < 150:
        print("\nEXCELLENT: Consistently solving MountainCar (avg < 150 steps)")
    elif last_20_success > first_20_success:
        print("\nGOOD: Learning detected (more successes over time)")
    elif last_20_avg < first_20_avg:
        print("\nPARTIAL: Some improvement in steps")
    else:
        print("\nNOT LEARNING: No significant improvement")

    return steps_history, rewards_history


if __name__ == "__main__":
    test_mountaincar_isolated()
