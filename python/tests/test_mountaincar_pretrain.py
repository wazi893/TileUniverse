"""
MountainCar with Pre-trained Weights + R-STDP Fine-tuning

Since pure R-STDP can't learn mappings from scratch (all outputs fire equally,
so eligibility traces are uniform), we pre-train with the correct mapping
and then use R-STDP for online adaptation.

This is realistic for many applications: pre-train offline, deploy with R-STDP
for online adaptation to changing conditions.
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


def run_pretrained_test(seed: int = 42):
    """
    Create SNN with pre-trained weights that encode the momentum heuristic.

    Since we can't modify weights directly in the Python API, we'll use
    the standard SNN but with learning parameters tuned for stability.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar with Heuristic Handoff")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print()

    env = gym.make("MountainCar-v0")
    n_inputs = 4
    n_outputs = 3
    n_neurons = 100
    ticks = 15

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=60,
        leak=180,
        connectivity="direct",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=30, seed=seed)

    # Run for 500 episodes with gradual heuristic handoff
    episode_lengths = []
    best_length = 200
    successes = 0

    # Teacher probability decay
    teacher_prob = 1.0
    teacher_min = 0.1
    teacher_decay = 0.995

    # Enable learning with VERY low rates for stability
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.95,
        ltp_rate=0.05,   # Very low - just for adaptation
        ltd_rate=0.02,
        learning_rate=0.005,
    )

    start_time = time.time()

    for episode in range(500):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = population_encode(obs)

            # Teacher: momentum heuristic
            teacher = 2 if obs[1] > 0 else 0

            # Student
            student = snn.step_learn_rates(rates, ticks)

            # Probability-based handoff
            if np.random.random() < teacher_prob:
                action = teacher
            else:
                action = student

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated and next_obs[0] >= 0.5:
                successes += 1

            # Very gentle reinforcement
            if action == teacher:
                reward = 0.1
            else:
                reward = -0.05
            snn.learn_action(teacher, reward, 0.1)

            obs = next_obs

        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_lengths[-50:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:3d}: avg={avg:6.1f}, best={best_length:3d}, "
                  f"teacher={teacher_prob:.2f}, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_lengths[:250])
    second_half = np.mean(episode_lengths[250:])
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))

    print(f"  First half avg: {first_half:.1f}")
    print(f"  Second half avg: {second_half:.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_length}")
    print(f"  Successes: {successes} ({100*successes/500:.1f}%)")

    if best_window < 200:
        print(f"  vs Random (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    env.close()


def run_pure_heuristic_baseline(seed: int = 42):
    """Baseline: pure momentum heuristic without any SNN."""

    print("\n" + "=" * 70)
    print("  BASELINE: Pure Momentum Heuristic (no SNN)")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    episode_lengths = []
    successes = 0

    for episode in range(500):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        while not done and steps < 200:
            # Pure heuristic: push in direction of velocity
            action = 2 if obs[1] > 0 else 0

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated and obs[0] >= 0.5:
                successes += 1

        episode_lengths.append(steps)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_lengths[-100:])
            print(f"Episode {episode+1:3d}: avg={avg:.1f}, success={successes}")

    print()
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))
    print(f"  Average: {np.mean(episode_lengths):.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Successes: {successes} ({100*successes/500:.1f}%)")

    env.close()


if __name__ == "__main__":
    run_pure_heuristic_baseline()
    run_pretrained_test()
