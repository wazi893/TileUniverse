"""
MountainCar with very low thresholds to ensure all outputs fire.

Key insight: R-STDP can only strengthen connections to outputs that fire.
If an output never fires, its eligibility is zero and learning can't help it.

Solution: Very low thresholds so all outputs fire, with differences in RATE.
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def population_encode(obs: np.ndarray) -> list:
    """Population coding for velocity direction."""
    pos, vel = obs

    pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))

    # Left/right velocity indicators
    left_vel = int(np.clip(-vel * 3500, 0, 255))
    right_vel = int(np.clip(vel * 3500, 0, 255))

    # Absolute velocity
    vel_mag = int(np.clip(abs(vel) * 3500, 0, 255))

    return [pos_rate, left_vel, right_vel, vel_mag]


def run_low_threshold_test(seed: int = 42):
    """Test with very low thresholds."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Low Threshold Test: Ensuring All Outputs Fire")
    print("=" * 70)

    n_inputs = 4
    n_outputs = 3
    n_neurons = 50  # Minimal
    ticks = 20

    # Moderate threshold for WTA dynamics - only strongest output should fire
    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=80,  # Higher threshold for WTA
        leak=180,      # Normal leak
        connectivity="direct",
        seed=seed,
    )

    # Check initial behavior
    print("\nBEFORE TRAINING (threshold=20):")
    test_vels = [-0.05, 0.0, 0.05]

    for vel in test_vels:
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        expected = 2 if vel > 0 else 0
        print(f"  vel={vel:+.3f}: rates={rates[1:3]} -> counts={list(counts)}")

    # Train
    print("\nTRAINING (1000 samples)...")
    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.05, learning_rate=0.03)

    np.random.seed(seed)
    correct = 0
    total = 0

    for i in range(1000):
        vel = np.random.uniform(-0.07, 0.07)
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)
        teacher = 2 if vel > 0 else 0

        snn.reset_eligibility()
        student = snn.step_learn_rates(rates, ticks)

        total += 1
        if student == teacher:
            correct += 1
            reward = 1.0
        else:
            reward = -0.3

        snn.learn_action(teacher, reward, 0.2)

        if (i + 1) % 200 == 0:
            print(f"  Step {i+1}: match={100*correct/total:.1f}%")

    print("\nAFTER TRAINING:")
    snn.disable_learning()

    for vel in test_vels:
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        expected = 2 if vel > 0 else 0
        marker = "OK" if action == expected else "WRONG"

        print(f"  vel={vel:+.3f}: rates={rates[1:3]} -> counts={list(counts)} -> action={action} {marker}")


def run_with_uniform_learning(seed: int = 42):
    """
    Test: Use UNIFORM learning instead of action-specific.

    Uniform R-STDP applies reward to ALL synapses equally.
    This might work better since it doesn't require the correct output to spike first.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("\n" + "=" * 70)
    print("  Uniform R-STDP (not action-specific)")
    print("=" * 70)

    n_inputs = 4
    n_outputs = 3
    n_neurons = 50
    ticks = 20

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=80,  # Higher threshold for WTA
        leak=180,
        connectivity="direct",
        seed=seed,
    )

    print("\nBEFORE TRAINING:")
    test_vels = [-0.05, 0.0, 0.05]

    for vel in test_vels:
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        print(f"  vel={vel:+.3f}: rates={rates[1:3]} -> counts={list(counts)}")

    print("\nTRAINING with UNIFORM R-STDP...")
    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.05, learning_rate=0.02)

    np.random.seed(seed)
    correct = 0
    total = 0

    for i in range(1000):
        vel = np.random.uniform(-0.07, 0.07)
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)
        teacher = 2 if vel > 0 else 0

        snn.reset_eligibility()
        student = snn.step_learn_rates(rates, ticks)

        total += 1
        if student == teacher:
            correct += 1
            reward = 1.0
        else:
            reward = -0.5

        # Use uniform learn() instead of learn_action()
        snn.learn(reward)

        if (i + 1) % 200 == 0:
            print(f"  Step {i+1}: match={100*correct/total:.1f}%")

    print("\nAFTER TRAINING:")
    snn.disable_learning()

    for vel in test_vels:
        obs = np.array([-0.5, vel])
        rates = population_encode(obs)

        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        expected = 2 if vel > 0 else 0
        marker = "OK" if action == expected else "WRONG"

        print(f"  vel={vel:+.3f}: rates={rates[1:3]} -> counts={list(counts)} -> action={action} {marker}")


if __name__ == "__main__":
    run_low_threshold_test()
    run_with_uniform_learning()
