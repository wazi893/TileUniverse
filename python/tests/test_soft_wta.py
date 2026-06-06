"""
Soft-WTA: Apply winner-take-all in the learning signal, not the network.

The SNN outputs spike counts for all actions. We then:
1. Use argmax for action selection (hard WTA)
2. Apply reward only to the winning output's eligibility

This gives WTA behavior for learning without changing kernel architecture.
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
    vel_mag = int(np.clip(abs(vel) * 3500, 0, 255))

    return [pos_rate, left_vel, right_vel, vel_mag]


def test_soft_wta_learning(seed: int = 42):
    """
    Test learning with soft-WTA in the reward signal.

    Key insight: Instead of trying to create WTA in the network (which is hard),
    we apply WTA in the learning signal:
    - If student == teacher: big positive reward to teacher action
    - If student != teacher: negative reward to student, positive to teacher
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Soft-WTA Learning: WTA in reward signal, not network")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    n_inputs = 4
    n_outputs = 3
    n_neurons = 100
    ticks = 30

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

    print("BEFORE TRAINING:")
    test_cases = [
        ([128, 200, 10, 150], 0, "left_vel high"),
        ([128, 10, 200, 150], 2, "right_vel high"),
    ]

    for rates, expected, desc in test_cases:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected})")

    # Training with soft-WTA reward
    print("\nTRAINING with soft-WTA reward (3000 samples)...")
    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.03)

    np.random.seed(seed)
    correct = 0
    total = 0

    for i in range(3000):
        vel = np.random.uniform(-0.07, 0.07)
        teacher = 2 if vel > 0 else 0

        left_vel = int(np.clip(-vel * 3500, 0, 255))
        right_vel = int(np.clip(vel * 3500, 0, 255))
        rates = [128, left_vel, right_vel, 128]

        snn.reset_eligibility()
        student = snn.step_learn_rates(rates, ticks)

        total += 1
        if student == teacher:
            correct += 1

        # Soft-WTA reward strategy:
        # 1. Always reward the teacher action strongly
        # 2. If student was wrong, also punish the student action

        # Reward correct action
        snn.learn_action(teacher, 1.0, 0.0)  # Full reward to teacher, no anti-reward

        # Punish incorrect student action if different
        if student != teacher:
            snn.learn_action(student, -0.5, 0.0)  # Negative reward to wrong choice

        if (i + 1) % 500 == 0:
            print(f"  Step {i+1}: match={100*correct/total:.1f}%")

    print(f"\nFinal match rate: {100*correct/total:.1f}%")

    print("\nAFTER TRAINING:")
    snn.disable_learning()

    for rates, expected, desc in test_cases:
        snn.set_inputs(rates)
        snn.reset_outputs()
        snn.tick_many_with_counting(ticks)
        snn.sync()
        counts = snn.get_output_counts()
        action = int(np.argmax(counts))
        marker = "OK" if action == expected else "WRONG"
        print(f"  {desc}: counts={list(counts)} -> action={action} (expected={expected}) {marker}")

    return 100 * correct / total


def run_mountaincar_soft_wta(
    n_episodes: int = 500,
    n_neurons: int = 200,
    seed: int = 42,
):
    """Full MountainCar test with soft-WTA learning."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("\n" + "=" * 70)
    print("  MountainCar with Soft-WTA Learning")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    n_inputs = 4
    n_outputs = 3
    ticks = 20

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
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.03)

    episode_lengths = []
    best_length = 200
    successes = 0
    correct = 0
    total = 0

    # Teacher probability
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
            student = snn.step_learn_rates(rates, ticks)

            # Teacher/student selection
            if np.random.random() < teacher_prob:
                action = teacher
            else:
                action = student

            total += 1
            if student == teacher:
                correct += 1

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated and next_obs[0] >= 0.5:
                successes += 1

            # Soft-WTA: reward teacher, punish wrong student
            snn.learn_action(teacher, 1.0, 0.0)
            if student != teacher:
                snn.learn_action(student, -0.3, 0.0)

            obs = next_obs

        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_lengths[-50:])
            match = 100 * correct / total
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: avg={avg:6.1f}, best={best_length:3d}, "
                  f"match={match:.1f}%, teacher={teacher_prob:.2f}, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_lengths[:n_episodes//2])
    second_half = np.mean(episode_lengths[n_episodes//2:])
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))

    print(f"  First half avg: {first_half:.1f}")
    print(f"  Second half avg: {second_half:.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_length}")
    print(f"  Final match rate: {100*correct/total:.1f}%")
    print(f"  Successes: {successes} ({100*successes/n_episodes:.1f}%)")

    if best_window < 200:
        print(f"  vs Random (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    env.close()
    return episode_lengths, successes


if __name__ == "__main__":
    match_rate = test_soft_wta_learning()

    if match_rate > 60:  # If learning works, test on MountainCar
        run_mountaincar_soft_wta(n_episodes=500)
    else:
        print("\n[!] Basic learning didn't improve much, skipping MountainCar test")
