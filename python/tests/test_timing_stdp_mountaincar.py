"""
MountainCar with Timing-Based Three-Factor STDP.

This test verifies that proper spike timing in STDP enables learning
in a real RL environment, not just supervised classification.
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


def run_mountaincar_timing_stdp(
    n_episodes: int = 500,
    n_neurons: int = 200,
    seed: int = 42,
):
    """MountainCar with timing-based three-factor STDP."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar with Timing-Based Three-Factor STDP")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("MountainCar-v0")
    n_inputs = 4
    n_outputs = 3
    ticks = 30  # More ticks to capture timing differences

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

    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.03)

    print(f"  Timing STDP: {snn.is_timing_stdp_enabled()}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Ticks per decision: {ticks}")
    print()

    episode_lengths = []
    best_length = 200
    successes = 0
    correct = 0
    total = 0

    # Teacher probability decay
    teacher_prob = 1.0
    teacher_decay = 0.995
    teacher_min = 0.1

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        snn.reset_eligibility()
        snn.reset_spike_times()

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

            # Three-factor STDP: reward teacher, punish wrong student
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

    # Evaluate improvement
    improvement = first_half - second_half
    print()
    if improvement > 5:
        print(f"  RESULT: Training shows {improvement:.1f} step improvement!")
    elif second_half < 180:
        print(f"  RESULT: Good performance (avg {second_half:.1f} steps)")
    else:
        print(f"  RESULT: Minimal improvement")

    print("=" * 70)

    env.close()
    return episode_lengths, successes


def compare_with_random_baseline(n_episodes: int = 100, seed: int = 42):
    """Compare timing STDP performance with random agent."""

    print("\n" + "=" * 70)
    print("  Comparison: Timing STDP vs Random Agent")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    np.random.seed(seed)

    # Random baseline
    print("\nRandom Agent (100 episodes)...")
    random_lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 200:
            action = np.random.randint(3)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        random_lengths.append(steps)

    random_avg = np.mean(random_lengths)
    print(f"  Random avg: {random_avg:.1f} steps")

    # Teacher policy (optimal)
    print("\nTeacher Policy (momentum-based, 100 episodes)...")
    teacher_lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 200:
            action = 2 if obs[1] > 0 else 0
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        teacher_lengths.append(steps)

    teacher_avg = np.mean(teacher_lengths)
    print(f"  Teacher avg: {teacher_avg:.1f} steps")

    env.close()

    print()
    print(f"  Gap (random - teacher): {random_avg - teacher_avg:.1f} steps")
    print("=" * 70)

    return random_avg, teacher_avg


if __name__ == "__main__":
    # First show the baselines
    random_avg, teacher_avg = compare_with_random_baseline()

    # Then run timing STDP training
    print()
    episode_lengths, successes = run_mountaincar_timing_stdp(n_episodes=500)

    # Final comparison
    second_half_avg = np.mean(episode_lengths[250:])
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  Random baseline:     {random_avg:.1f} steps")
    print(f"  Teacher policy:      {teacher_avg:.1f} steps")
    print(f"  SNN (second half):   {second_half_avg:.1f} steps")

    # How much of the gap did we close?
    gap = random_avg - teacher_avg
    closed = random_avg - second_half_avg
    pct = (closed / gap * 100) if gap > 0 else 0
    print(f"  Gap closed: {closed:.1f} / {gap:.1f} = {pct:.1f}%")
    print("=" * 70)
