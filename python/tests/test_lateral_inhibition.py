"""
Test Lateral Inhibition for Winner-Take-All Dynamics

With lateral inhibition, when one output fires it suppresses the others.
This should create clear differentiation that R-STDP can learn from.
"""

import numpy as np
import time
import tileuniverse as tu


def test_wta_dynamics(seed: int = 42):
    """Verify that lateral inhibition creates WTA behavior."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Testing Winner-Take-All Dynamics (Lateral Inhibition)")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    n_inputs = 4
    n_outputs = 3
    n_neurons = 50
    ticks = 50  # More ticks for inhibition to accumulate

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=30,  # Very low threshold - spike early
        leak=100,      # Less leak - maintain inhibition
        connectivity="direct",  # Now includes lateral inhibition
        seed=seed,
    )

    print("Testing with different input patterns:\n")

    # Test patterns: activate different inputs strongly
    test_cases = [
        ([200, 50, 10, 100], "Input 0 strong"),
        ([50, 200, 10, 100], "Input 1 strong"),
        ([50, 10, 200, 100], "Input 2 strong"),
        ([100, 100, 100, 100], "All equal"),
        ([200, 200, 10, 100], "Inputs 0+1 strong"),
    ]

    for rates, desc in test_cases:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()

        counts = snn.get_output_counts()
        winner = int(np.argmax(counts))
        total = sum(counts)

        # Check WTA: winner should have most spikes, others suppressed
        winner_pct = 100 * counts[winner] / total if total > 0 else 0

        print(f"  {desc:20s}: rates={rates[:3]} -> counts={list(counts)} -> winner={winner} ({winner_pct:.0f}%)")

    print()

    # Check if we have WTA behavior
    print("Expected WTA behavior: winner should have >50% of total spikes")
    print("If counts are uniform, lateral inhibition isn't working")


def test_learning_with_wta(seed: int = 42):
    """Test if R-STDP learning works better with WTA."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("\n" + "=" * 70)
    print("  R-STDP Learning with Lateral Inhibition")
    print("=" * 70)

    n_inputs = 4
    n_outputs = 3
    n_neurons = 50
    ticks = 50  # More ticks for inhibition to accumulate

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=30,  # Low threshold for early spiking
        leak=100,      # Less leak
        connectivity="direct",
        seed=seed,
    )

    # Simple task: input 1 high -> action 0, input 2 high -> action 2
    print("\nTask: left_vel (input 1) -> action 0, right_vel (input 2) -> action 2")
    print()

    print("BEFORE TRAINING:")
    test_vels = [
        ([100, 200, 10, 150], 0, "left_vel high"),
        ([100, 10, 200, 150], 2, "right_vel high"),
    ]

    for rates, expected, desc in test_vels:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        marker = "OK" if action == expected else "WRONG"
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected}) {marker}")

    # Train
    print("\nTRAINING (2000 samples)...")
    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.15, ltd_rate=0.05, learning_rate=0.02)

    np.random.seed(seed)
    correct = 0
    total = 0

    for i in range(2000):
        # Random velocity
        vel = np.random.uniform(-0.07, 0.07)
        teacher = 2 if vel > 0 else 0

        # Population encode
        left_vel = int(np.clip(-vel * 3500, 0, 255))
        right_vel = int(np.clip(vel * 3500, 0, 255))
        rates = [128, left_vel, right_vel, 128]

        snn.reset_eligibility()
        student = snn.step_learn_rates(rates, ticks)

        total += 1
        if student == teacher:
            correct += 1
            reward = 1.0
        else:
            reward = -0.5

        snn.learn_action(teacher, reward, 0.3)

        if (i + 1) % 500 == 0:
            print(f"  Step {i+1}: match={100*correct/total:.1f}%")

    print(f"\nFinal match rate: {100*correct/total:.1f}%")

    print("\nAFTER TRAINING:")
    snn.disable_learning()

    for rates, expected, desc in test_vels:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        marker = "OK" if action == expected else "WRONG"
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected}) {marker}")


if __name__ == "__main__":
    test_wta_dynamics()
    test_learning_with_wta()
