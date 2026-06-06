"""
MountainCar with Population Coding for R-STDP

Key insight: With a single input neuron, all outputs receive the same spike rate.
We need POPULATION coding: different neurons fire for different input ranges.

vel < 0 --> "left" neuron fires high
vel > 0 --> "right" neuron fires high

Each population neuron connects preferentially to its corresponding output.
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def population_encode(obs: np.ndarray) -> list:
    """
    Population coding: separate neurons for different velocity directions.

    For velocity:
    - left_vel: high when vel < 0, low when vel > 0
    - right_vel: high when vel > 0, low when vel < 0
    - neutral_vel: high when vel ~ 0

    This ensures different inputs fire for different velocity directions,
    which can then be routed to different outputs.
    """
    pos, vel = obs

    # Position encoding (not critical for heuristic)
    pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))

    # Velocity population: separate neurons for left/right
    # Left velocity neuron: fires when moving left
    left_vel = int(np.clip(-vel * 3500, 0, 255))  # High when vel negative

    # Right velocity neuron: fires when moving right
    right_vel = int(np.clip(vel * 3500, 0, 255))  # High when vel positive

    # Neutral velocity neuron: fires when nearly stationary
    neutral_vel = int(np.clip(255 - abs(vel) * 5000, 0, 255))

    # Absolute velocity (how fast, regardless of direction)
    vel_mag = int(np.clip(abs(vel) * 3500, 0, 255))

    return [pos_rate, left_vel, right_vel, neutral_vel, vel_mag]


def run_population_learning(
    n_episodes: int = 300,
    n_neurons: int = 100,  # Minimal network
    seed: int = 42,
):
    """
    Train with population-coded inputs and direct connectivity.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar with Population Coding + R-STDP")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("MountainCar-v0")
    n_inputs = 5  # pos, left_vel, right_vel, neutral_vel, vel_mag
    n_outputs = 3
    ticks_per_step = 15

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=50,  # Lower threshold
        leak=150,      # Less leak for integration
        connectivity="direct",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=30, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.9,
        ltp_rate=0.15,
        ltd_rate=0.05,
        learning_rate=0.02,
    )

    print(f"  Network: {n_neurons} neurons, {n_inputs} inputs, {n_outputs} outputs")
    print("  Input encoding: [pos, left_vel, right_vel, neutral_vel, vel_mag]")
    print("  Target mapping: left_vel -> action 0, right_vel -> action 2")
    print()

    correct = 0
    total = 0
    episode_lengths = []
    successes = 0

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = population_encode(obs)

            # Teacher: momentum heuristic
            teacher = 2 if obs[1] > 0 else 0

            # Student
            student = snn.step_learn_rates(rates, ticks_per_step)

            # Always follow teacher for now
            action = teacher

            total += 1
            if student == teacher:
                correct += 1

            # Imitation reward
            if student == teacher:
                reward = 1.0
            else:
                reward = -0.5

            snn.learn_action(teacher, reward, 0.3)

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated and next_obs[0] >= 0.5:
                successes += 1

            obs = next_obs

        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            match_rate = 100 * correct / total
            avg_len = np.mean(episode_lengths[-50:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: match={match_rate:5.1f}%, avg_len={avg_len:5.1f}, "
                  f"success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    print(f"  Final match rate: {100 * correct / total:.1f}%")
    print(f"  Successes: {successes} ({100*successes/n_episodes:.1f}%)")

    # Test specific cases
    print("\nTest cases:")
    snn.disable_learning()

    test_cases = [
        (-0.05, "moving left, should -> action 0"),
        (-0.02, "moving left, should -> action 0"),
        (0.0, "stationary"),
        (0.02, "moving right, should -> action 2"),
        (0.05, "moving right, should -> action 2"),
    ]

    for vel, desc in test_cases:
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)
        expected = 2 if vel > 0 else 0

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks_per_step)
        snn.sync()

        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        marker = "OK" if action == expected else "WRONG"

        print(f"  vel={vel:+.3f}: rates={rates[1:4]} -> counts={list(counts)} -> action={action} {marker}")

    env.close()
    return 100 * correct / total


def run_with_heuristic_handoff(
    n_episodes: int = 500,
    n_neurons: int = 200,
    seed: int = 42,
):
    """
    Start with heuristic, gradually hand off to SNN.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Population Coding with Gradual Handoff")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    n_inputs = 5
    n_outputs = 3
    ticks_per_step = 15

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=50,
        leak=150,
        connectivity="direct",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=30, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.9,
        ltp_rate=0.15,
        ltd_rate=0.05,
        learning_rate=0.02,
    )

    episode_lengths = []
    best_length = 200
    successes = 0
    correct = 0
    total = 0

    # Handoff schedule
    teacher_prob = 1.0
    teacher_decay = 0.995
    teacher_min = 0.1

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = population_encode(obs)
            teacher = 2 if obs[1] > 0 else 0
            student = snn.step_learn_rates(rates, ticks_per_step)

            # Probability-based handoff
            if np.random.random() < teacher_prob:
                action = teacher
            else:
                action = student

            total += 1
            if student == teacher:
                correct += 1

            # Reward for matching teacher
            if student == teacher:
                reward = 1.0
            else:
                reward = -0.5

            snn.learn_action(teacher, reward, 0.3)

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated and next_obs[0] >= 0.5:
                successes += 1

            obs = next_obs

        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 50 == 0:
            match_rate = 100 * correct / total
            avg_len = np.mean(episode_lengths[-50:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: match={match_rate:5.1f}%, avg={avg_len:5.1f}, "
                  f"best={best_length:3d}, teacher={teacher_prob:.2f}, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_lengths[:n_episodes//2])
    second_half = np.mean(episode_lengths[n_episodes//2:])
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))

    print(f"  First half avg: {first_half:.1f}")
    print(f"  Second half avg: {second_half:.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_length}")
    print(f"  Final match rate: {100 * correct / total:.1f}%")
    print(f"  Successes: {successes} ({100*successes/n_episodes:.1f}%)")

    if best_window < 200:
        print(f"  vs Random (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    env.close()
    return episode_lengths, successes


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TEST 1: Population Coding Learning")
    print("="*70 + "\n")

    match_rate = run_population_learning(n_episodes=300, n_neurons=100)

    print("\n" + "="*70)
    print("  TEST 2: Population Coding with Gradual Handoff")
    print("="*70 + "\n")

    lengths, successes = run_with_heuristic_handoff(n_episodes=500, n_neurons=200)
