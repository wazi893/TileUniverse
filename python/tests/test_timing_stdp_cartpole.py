"""
CartPole with Timing-Based Three-Factor STDP.

CartPole is simpler than MountainCar:
- Clear reward signal (survive = good)
- Intuitive optimal policy (move opposite to pole lean)
- Fast episodes during early training

Key insight: Use dual-channel encoding (like MountainCar's left_vel/right_vel)
for the pole angle - this makes the input-output mapping clearer.
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def encode_cartpole_dual(obs: np.ndarray) -> list:
    """
    Dual-channel encoding for CartPole.

    Key insight from MountainCar: separate left/right channels work better
    than a single centered channel. Apply same principle to pole angle.
    """
    pos, vel, angle, ang_vel = obs

    # Angle: separate left-lean and right-lean channels
    # This is the KEY signal for the decision
    lean_left = int(np.clip(-angle * 400, 0, 255))   # High when leaning left
    lean_right = int(np.clip(angle * 400, 0, 255))   # High when leaning right

    # Angular velocity: which way is it falling?
    falling_left = int(np.clip(-ang_vel * 80, 0, 255))
    falling_right = int(np.clip(ang_vel * 80, 0, 255))

    # The 4 key inputs: lean direction + falling direction
    return [lean_left, lean_right, falling_left, falling_right]


def teacher_policy(obs: np.ndarray) -> int:
    """Simple teacher: move opposite to pole lean + anticipate fall."""
    angle = obs[2]
    ang_vel = obs[3]

    # Combine angle and angular velocity
    combined = angle + 0.3 * ang_vel

    # 0 = push left, 1 = push right
    # Push right when leaning right to counteract
    return 1 if combined > 0 else 0


def run_cartpole_timing_stdp(
    n_episodes: int = 300,
    n_neurons: int = 100,
    seed: int = 42,
):
    """CartPole with timing-based three-factor STDP."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  CartPole with Timing-Based STDP (Dual-Channel Encoding)")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("CartPole-v1")
    n_inputs = 4   # lean_left, lean_right, falling_left, falling_right
    n_outputs = 2  # push left, push right
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
    print(f"  Neurons: {n_neurons}, Inputs: {n_inputs}, Outputs: {n_outputs}")
    print(f"  Ticks per decision: {ticks}")
    print()

    episode_rewards = []
    best_reward = 0
    correct = 0
    total = 0

    # Teacher probability decay
    teacher_prob = 1.0
    teacher_decay = 0.995  # Slower decay
    teacher_min = 0.1

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0

        snn.reset_eligibility()
        snn.reset_spike_times()

        while not done:
            rates = encode_cartpole_dual(obs)
            teacher = teacher_policy(obs)
            student = snn.step_learn_rates(rates, ticks)

            # Teacher/student selection
            if np.random.random() < teacher_prob:
                action = teacher
            else:
                action = student

            total += 1
            if student == teacher:
                correct += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Three-factor STDP
            snn.learn_action(teacher, 1.0, 0.0)
            if student != teacher:
                snn.learn_action(student, -0.5, 0.0)

            obs = next_obs

        episode_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 30 == 0:
            avg = np.mean(episode_rewards[-30:])
            match = 100 * correct / total
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: avg={avg:6.1f}, best={best_reward:3.0f}, "
                  f"match={match:.1f}%, teacher={teacher_prob:.2f}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_rewards[:n_episodes//2])
    second_half = np.mean(episode_rewards[n_episodes//2:])
    best_window = max(np.mean(episode_rewards[i:i+30]) for i in range(len(episode_rewards) - 29))

    print(f"  First half avg: {first_half:.1f}")
    print(f"  Second half avg: {second_half:.1f}")
    print(f"  Best 30-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_reward:.0f}")
    print(f"  Final match rate: {100*correct/total:.1f}%")

    if second_half >= 475:
        print(f"  STATUS: SOLVED!")
    elif second_half >= 300:
        print(f"  STATUS: Near-solved!")
    elif second_half >= 100:
        print(f"  STATUS: Good progress!")
    else:
        print(f"  STATUS: Learning...")

    print("=" * 70)

    env.close()
    return episode_rewards, best_reward


def quick_baseline(n_episodes: int = 50, seed: int = 42):
    """Quick baseline comparison."""

    print("=" * 70)
    print("  Baselines")
    print("=" * 70)

    env = gym.make("CartPole-v1")
    np.random.seed(seed)

    # Random
    random_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        reward = 0
        while not done:
            action = np.random.randint(2)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward += r
        random_rewards.append(reward)

    print(f"  Random: {np.mean(random_rewards):.1f} avg")

    # Teacher
    teacher_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        reward = 0
        while not done:
            action = teacher_policy(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward += r
        teacher_rewards.append(reward)

    print(f"  Teacher: {np.mean(teacher_rewards):.1f} avg")
    print("=" * 70)

    env.close()
    return np.mean(random_rewards), np.mean(teacher_rewards)


if __name__ == "__main__":
    random_avg, teacher_avg = quick_baseline()
    print()
    episode_rewards, best = run_cartpole_timing_stdp(n_episodes=300)

    second_half = np.mean(episode_rewards[150:])
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Random:   {random_avg:6.1f}")
    print(f"  Teacher:  {teacher_avg:6.1f}")
    print(f"  SNN:      {second_half:6.1f}")

    gap = teacher_avg - random_avg
    closed = second_half - random_avg
    pct = (closed / gap * 100) if gap > 0 else 0
    print(f"  Gap closed: {pct:.1f}%")
    print("=" * 70)
