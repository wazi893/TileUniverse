"""
Debug: Why isn't the SNN learning to imitate the momentum heuristic?

The heuristic is simple: if velocity > 0, push right (2); else push left (0).
This is essentially a single-feature binary classification.
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def simple_rates(obs: np.ndarray) -> list:
    """Minimal encoding: just velocity direction."""
    pos, vel = obs

    # Velocity direction only - this is ALL the heuristic needs
    # < 128 = moving left, > 128 = moving right
    vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))

    # Some context
    pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))

    return [vel_dir, pos_rate]


def run_simple_imitation(
    n_episodes: int = 300,
    n_neurons: int = 500,  # Small network
    seed: int = 42,
):
    """
    Simplest possible imitation: just learn vel > 0 -> action 2, else action 0.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Simple Imitation: Learn vel_dir -> action")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("MountainCar-v0")
    n_inputs = 2  # Just vel_dir and pos
    n_outputs = 3
    ticks_per_step = 15  # More ticks for clearer signal

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=60,  # Lower threshold
        leak=180,
        connectivity="direct",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=50, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.9,
        ltp_rate=0.2,
        ltd_rate=0.08,
        learning_rate=0.03,
    )

    print(f"  Network: {n_neurons} neurons, {n_inputs} inputs, {n_outputs} outputs")
    print()

    correct = 0
    total = 0
    episode_lengths = []

    # Track output spike distribution
    output_sums = [0, 0, 0]

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = simple_rates(obs)

            # Teacher action
            teacher = 2 if obs[1] > 0 else 0

            # Student action
            student = snn.step_learn_rates(rates, ticks_per_step)

            # Track output distribution
            counts = snn.get_output_counts()
            for i, c in enumerate(counts):
                output_sums[i] += c

            # Always follow teacher in this debug test
            action = teacher

            total += 1
            if student == teacher:
                correct += 1

            # Strong imitation signal
            if student == teacher:
                reward = 1.0
            else:
                reward = -1.0  # Strong negative for wrong answer

            snn.learn_action(teacher, reward, 0.3)

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            obs = next_obs

        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            match_rate = 100 * correct / total
            elapsed = time.time() - start_time
            avg_len = np.mean(episode_lengths[-50:])
            total_spikes = sum(output_sums)
            spike_pcts = [100 * s / total_spikes if total_spikes > 0 else 0 for s in output_sums]
            print(f"Episode {episode+1:3d}: match={match_rate:5.1f}%, avg_len={avg_len:5.1f}, "
                  f"spikes=[{spike_pcts[0]:.0f}%,{spike_pcts[1]:.0f}%,{spike_pcts[2]:.0f}%], "
                  f"time={elapsed:.1f}s")

    print()
    print(f"Final match rate: {100 * correct / total:.1f}%")
    print(f"Random baseline: ~50% (2-way classification)")

    env.close()
    return 100 * correct / total


def run_direct_training(
    n_steps: int = 10000,
    n_neurons: int = 500,
    seed: int = 42,
):
    """
    Direct training without environment loop.
    Just sample (vel_dir) -> (action) pairs and train.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Direct Training: vel_dir -> action (no env)")
    print("=" * 70)

    n_inputs = 2
    n_outputs = 3
    ticks_per_step = 20

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=60,
        leak=180,
        connectivity="direct",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=50, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.9,
        ltp_rate=0.2,
        ltd_rate=0.08,
        learning_rate=0.03,
    )

    np.random.seed(seed)
    correct = 0
    total = 0

    start_time = time.time()

    for step in range(n_steps):
        # Random velocity in (-0.07, 0.07)
        vel = np.random.uniform(-0.07, 0.07)
        pos = np.random.uniform(-1.2, 0.6)

        # Teacher
        teacher = 2 if vel > 0 else 0

        # Encode
        vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))
        pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))
        rates = [vel_dir, pos_rate]

        # Get student prediction
        student = snn.step_learn_rates(rates, ticks_per_step)

        total += 1
        if student == teacher:
            correct += 1
            reward = 1.0
        else:
            reward = -1.0

        snn.learn_action(teacher, reward, 0.3)

        # Reset eligibility every 100 steps to simulate episodes
        if (step + 1) % 100 == 0:
            snn.reset_eligibility()

        if (step + 1) % 1000 == 0:
            match_rate = 100 * correct / total
            elapsed = time.time() - start_time
            print(f"Step {step+1:5d}: match={match_rate:5.1f}%, time={elapsed:.1f}s")

    print()
    print(f"Final match rate: {100 * correct / total:.1f}%")

    # Test phase (no learning)
    print("\nTest phase (no learning):")
    snn.disable_learning()
    test_correct = 0
    test_total = 0

    for _ in range(1000):
        vel = np.random.uniform(-0.07, 0.07)
        pos = np.random.uniform(-1.2, 0.6)
        teacher = 2 if vel > 0 else 0

        vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))
        pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))
        rates = [vel_dir, pos_rate]

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks_per_step)
        snn.sync()

        counts = snn.get_output_counts()
        student = int(np.argmax(counts))

        test_total += 1
        if student == teacher:
            test_correct += 1

    print(f"Test match rate: {100 * test_correct / test_total:.1f}%")

    return 100 * correct / total


def analyze_snn_behavior(seed: int = 42):
    """
    Analyze what the SNN is actually doing.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  SNN Behavior Analysis")
    print("=" * 70)

    n_inputs = 2
    n_outputs = 3
    ticks = 20

    snn = tu.FusedSNN(
        n_neurons=500,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=60,
        leak=180,
        connectivity="direct",  # Use direct input->output connections
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=50, seed=seed)

    print("\nBEFORE TRAINING:")
    print("-" * 40)

    # Test with different vel_dir values
    test_vels = [-0.05, -0.02, 0.0, 0.02, 0.05]

    for vel in test_vels:
        vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))
        rates = [vel_dir, 128]  # neutral position

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        expected = 2 if vel > 0 else 0

        print(f"  vel={vel:+.3f} -> vel_dir={vel_dir:3d} -> counts={list(counts)} -> action={action} (expected={expected})")

    # Now train
    print("\n\nTRAINING (1000 samples)...")
    snn.enable_learning()
    # Lower learning rate to prevent collapse, lower anti-reward
    snn.set_learning_params(tau_decay=0.85, ltp_rate=0.3, ltd_rate=0.1, learning_rate=0.05)

    np.random.seed(seed)
    for i in range(1000):
        vel = np.random.uniform(-0.07, 0.07)
        teacher = 2 if vel > 0 else 0

        vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))
        rates = [vel_dir, 128]

        snn.reset_eligibility()
        student = snn.step_learn_rates(rates, ticks)

        reward = 1.0 if student == teacher else -1.0
        snn.learn_action(teacher, reward, 0.3)

    print("\nAFTER TRAINING:")
    print("-" * 40)

    snn.disable_learning()

    for vel in test_vels:
        vel_dir = int(np.clip(vel * 2000 + 128, 0, 255))
        rates = [vel_dir, 128]

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        expected = 2 if vel > 0 else 0
        marker = "OK" if action == expected else "WRONG"

        print(f"  vel={vel:+.3f} -> vel_dir={vel_dir:3d} -> counts={list(counts)} -> action={action} (expected={expected}) {marker}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ANALYSIS: What is the SNN learning?")
    print("="*70 + "\n")

    analyze_snn_behavior()

    print("\n" + "="*70)
    print("  TEST: Direct training")
    print("="*70 + "\n")

    run_direct_training(n_steps=5000, n_neurons=500)
