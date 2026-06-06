"""
Test timing-based three-factor STDP.

Key difference from our earlier R-STDP:
- Old approach: Checked if pre AND post spike in SAME tick (co-occurrence)
- New approach: Tracks WHEN neurons spike, computes timing-dependent eligibility
  - LTP: post fires AFTER pre (causal) -> strengthen synapse
  - LTD: post fires BEFORE pre (anti-causal) -> weaken synapse
"""

import numpy as np
import time
import tileuniverse as tu


def test_timing_stdp_classification(seed: int = 42):
    """
    Test timing-based STDP on simple velocity classification.

    This should break the cold start problem because:
    - Even if all outputs fire initially, they fire at DIFFERENT times
    - The timing difference creates different eligibility traces
    - Reward can then differentiate between outputs
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Timing-Based Three-Factor STDP Test")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    n_inputs = 4
    n_outputs = 3
    n_neurons = 100
    ticks = 50  # More ticks to capture timing differences

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

    # Enable timing-based STDP
    snn.enable_timing_stdp()
    snn.set_stdp_timing_params(tau_plus=20.0, tau_minus=20.0, window=30)

    print(f"  Timing STDP enabled: {snn.is_timing_stdp_enabled()}")
    print()

    # Test cases: left velocity -> action 0, right velocity -> action 2
    test_cases = [
        ([128, 200, 10, 150], 0, "left_vel high"),
        ([128, 10, 200, 150], 2, "right_vel high"),
        ([128, 100, 100, 100], 1, "neutral"),
    ]

    print("BEFORE TRAINING:")
    for rates, expected, desc in test_cases:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected})")

    # Training with timing-based STDP
    print("\nTRAINING with timing-based STDP (3000 samples)...")
    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.05)

    np.random.seed(seed)
    correct = 0
    total = 0

    start_time = time.time()

    for i in range(3000):
        # Generate random velocity
        vel = np.random.uniform(-0.07, 0.07)
        teacher = 2 if vel > 0 else 0

        # Encode velocity
        left_vel = int(np.clip(-vel * 3500, 0, 255))
        right_vel = int(np.clip(vel * 3500, 0, 255))
        rates = [128, left_vel, right_vel, 128]

        # Reset eligibility and spike times for new sample
        snn.reset_eligibility()
        snn.reset_spike_times()

        # Get student action (this internally tracks spike timing for eligibility)
        student = snn.step_learn_rates(rates, ticks)

        total += 1
        if student == teacher:
            correct += 1

        # Three-factor STDP: reward correct action, punish wrong one
        snn.learn_action(teacher, 1.0, 0.0)
        if student != teacher:
            snn.learn_action(student, -0.5, 0.0)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i+1}: match={100*correct/total:.1f}% ({elapsed:.1f}s)")

    print(f"\nFinal match rate: {100*correct/total:.1f}%")

    print("\nAFTER TRAINING:")
    snn.disable_learning()

    correct_after = 0
    for rates, expected, desc in test_cases:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        marker = "OK" if action == expected else "WRONG"
        if action == expected and expected != 1:  # neutral is unpredictable
            correct_after += 1
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected}) {marker}")

    print()
    print("=" * 70)
    success = correct_after >= 2 and (100 * correct / total) > 60
    if success:
        print("  RESULT: Timing-based STDP shows learning improvement!")
    else:
        print("  RESULT: Need further tuning")
    print("=" * 70)

    return 100 * correct / total


def compare_timing_vs_legacy(seed: int = 42):
    """
    Compare timing-based STDP vs legacy co-occurrence approach.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("\n" + "=" * 70)
    print("  Timing STDP vs Legacy Comparison")
    print("=" * 70)

    n_inputs = 4
    n_outputs = 3
    n_neurons = 100
    ticks = 50
    n_samples = 2000

    np.random.seed(seed)

    # Generate training data
    velocities = np.random.uniform(-0.07, 0.07, n_samples)
    teachers = (velocities > 0).astype(int) * 2  # 0 for left, 2 for right

    results = {}

    for mode in ["timing", "legacy"]:
        print(f"\nTraining with {mode.upper()} STDP...")

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

        if mode == "timing":
            snn.enable_timing_stdp()
            snn.set_stdp_timing_params(tau_plus=20.0, tau_minus=20.0, window=30)
        else:
            snn.disable_timing_stdp()

        snn.enable_learning()
        snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.05)

        correct = 0

        for i in range(n_samples):
            vel = velocities[i]
            teacher = teachers[i]

            left_vel = int(np.clip(-vel * 3500, 0, 255))
            right_vel = int(np.clip(vel * 3500, 0, 255))
            rates = [128, left_vel, right_vel, 128]

            snn.reset_eligibility()
            if mode == "timing":
                snn.reset_spike_times()

            student = snn.step_learn_rates(rates, ticks)

            if student == teacher:
                correct += 1

            snn.learn_action(teacher, 1.0, 0.0)
            if student != teacher:
                snn.learn_action(student, -0.5, 0.0)

            if (i + 1) % 500 == 0:
                print(f"  Step {i+1}: match={100*correct/(i+1):.1f}%")

        results[mode] = 100 * correct / n_samples
        print(f"  Final: {results[mode]:.1f}%")

    print("\n" + "=" * 70)
    print(f"  Timing STDP: {results['timing']:.1f}%")
    print(f"  Legacy STDP: {results['legacy']:.1f}%")
    diff = results['timing'] - results['legacy']
    if diff > 0:
        print(f"  Timing STDP is {diff:.1f}% better!")
    else:
        print(f"  Legacy STDP is {-diff:.1f}% better")
    print("=" * 70)

    return results


if __name__ == "__main__":
    match_rate = test_timing_stdp_classification()

    if match_rate > 50:
        compare_timing_vs_legacy()
