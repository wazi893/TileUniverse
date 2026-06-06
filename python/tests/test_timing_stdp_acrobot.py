"""
Acrobot with Timing-Based Three-Factor STDP.

Acrobot is similar to MountainCar:
- Build momentum by swinging
- Simple strategy: apply torque in direction of velocity
- Success = tip reaches target height

This should be a good fit for our STDP approach!
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def encode_acrobot(obs: np.ndarray) -> list:
    """
    Encode Acrobot observation to spike rates.

    obs[0]: cos(theta1) - angle of first link
    obs[1]: sin(theta1)
    obs[2]: cos(theta2) - angle of second link relative to first
    obs[3]: sin(theta2)
    obs[4]: angular velocity of first link
    obs[5]: angular velocity of second link
    """
    cos1, sin1, cos2, sin2, vel1, vel2 = obs

    # Angular velocity directions - the KEY signals
    # Dual-channel encoding like MountainCar
    vel1_neg = int(np.clip(-vel1 * 30, 0, 255))
    vel1_pos = int(np.clip(vel1 * 30, 0, 255))
    vel2_neg = int(np.clip(-vel2 * 15, 0, 255))
    vel2_pos = int(np.clip(vel2 * 15, 0, 255))

    return [vel1_neg, vel1_pos, vel2_neg, vel2_pos]


def teacher_policy(obs: np.ndarray) -> int:
    """
    Teacher: apply torque to pump energy into the system.

    Strategy: torque in the direction of link 2's velocity
    (like pushing a swing at the right time)

    Actions: 0 = negative torque, 1 = no torque, 2 = positive torque
    """
    vel2 = obs[5]  # Angular velocity of second link

    if vel2 > 0.1:
        return 2  # Positive torque
    elif vel2 < -0.1:
        return 0  # Negative torque
    else:
        return 1  # No torque (coast)


def run_acrobot_timing_stdp(
    n_episodes: int = 400,
    n_neurons: int = 100,
    seed: int = 42,
):
    """Acrobot with timing-based three-factor STDP."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  Acrobot with Timing-Based STDP")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("Acrobot-v1")  # Max 500 steps, -1 reward per step
    n_inputs = 4
    n_outputs = 3  # -1, 0, +1 torque
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

    # Enable timing-based STDP
    snn.enable_timing_stdp()
    snn.set_stdp_timing_params(tau_plus=20.0, tau_minus=20.0, window=30)

    snn.enable_learning()
    snn.set_learning_params(tau_decay=0.9, ltp_rate=0.2, ltd_rate=0.08, learning_rate=0.03)

    print(f"  Timing STDP: {snn.is_timing_stdp_enabled()}")
    print(f"  Neurons: {n_neurons}")
    print()

    episode_steps = []  # Lower is better (fewer steps to reach goal)
    best_steps = 500
    successes = 0
    correct = 0
    total = 0

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

        while not done and steps < 500:
            rates = encode_acrobot(obs)
            teacher = teacher_policy(obs)
            student = snn.step_learn_rates(rates, ticks)

            if np.random.random() < teacher_prob:
                action = teacher
            else:
                action = student

            total += 1
            if student == teacher:
                correct += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Success check
            if terminated and steps < 500:
                successes += 1

            # Three-factor STDP
            snn.learn_action(teacher, 1.0, 0.0)
            if student != teacher:
                snn.learn_action(student, -0.3, 0.0)

            obs = next_obs

        episode_steps.append(steps)
        if steps < best_steps:
            best_steps = steps

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_steps[-50:])
            match = 100 * correct / total
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: avg={avg:6.1f}, best={best_steps:3d}, "
                  f"match={match:.1f}%, teacher={teacher_prob:.2f}, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_steps[:n_episodes//2])
    second_half = np.mean(episode_steps[n_episodes//2:])
    best_window = min(np.mean(episode_steps[i:i+50]) for i in range(len(episode_steps) - 49))

    print(f"  First half avg: {first_half:.1f} steps")
    print(f"  Second half avg: {second_half:.1f} steps")
    print(f"  Best 50-ep window: {best_window:.1f} steps")
    print(f"  Best episode: {best_steps} steps")
    print(f"  Successes: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Final match rate: {100*correct/total:.1f}%")

    if best_window < 200:
        print(f"  vs Random (500): {(500 - best_window) / 500 * 100:.1f}% faster!")
    print("=" * 70)

    env.close()
    return episode_steps, successes


def quick_baseline(n_episodes: int = 50, seed: int = 42):
    """Quick baseline."""

    print("=" * 70)
    print("  Baselines")
    print("=" * 70)

    env = gym.make("Acrobot-v1")
    np.random.seed(seed)

    # Random
    random_steps = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 500:
            action = np.random.randint(3)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        random_steps.append(steps)

    print(f"  Random: {np.mean(random_steps):.1f} avg steps")

    # Teacher
    teacher_steps = []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 500:
            action = teacher_policy(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps < 500:
            successes += 1
        teacher_steps.append(steps)

    print(f"  Teacher: {np.mean(teacher_steps):.1f} avg steps ({successes}/{n_episodes} success)")
    print("=" * 70)

    env.close()
    return np.mean(random_steps), np.mean(teacher_steps)


if __name__ == "__main__":
    random_avg, teacher_avg = quick_baseline()
    print()
    episode_steps, successes = run_acrobot_timing_stdp(n_episodes=400)

    second_half = np.mean(episode_steps[200:])
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Random:   {random_avg:6.1f} steps")
    print(f"  Teacher:  {teacher_avg:6.1f} steps")
    print(f"  SNN:      {second_half:6.1f} steps")

    # Lower is better for Acrobot
    gap = random_avg - teacher_avg
    closed = random_avg - second_half
    pct = (closed / gap * 100) if gap > 0 else 0
    print(f"  Gap closed: {pct:.1f}%")
    print("=" * 70)
