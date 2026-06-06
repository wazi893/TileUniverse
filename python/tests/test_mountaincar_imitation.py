"""
MountainCar with Imitation Learning for R-STDP

Key insight: The momentum heuristic (push in direction of velocity) works.
Train the SNN to imitate this heuristic, then let it refine with RL.
"""

import gymnasium as gym
import numpy as np
import time
import tileuniverse as tu


def obs_to_rates(obs: np.ndarray) -> list:
    """Convert MountainCar observation to spike rates [0-255]."""
    pos, vel = obs

    # Position: -1.2 to 0.6 -> 0 to 255
    pos_rate = int(np.clip((pos + 1.2) / 1.8 * 255, 0, 255))

    # Velocity magnitude (amplified)
    vel_mag = int(np.clip(abs(vel) * 3000, 0, 255))

    # Velocity direction: < 128 = left, > 128 = right
    vel_dir = int(np.clip(vel * 1800 + 128, 0, 255))

    # Slope indicators
    left_slope = 255 if pos < -0.5 else 0
    right_slope = 255 if pos > -0.2 else 0

    # Energy feature
    height = np.sin(3 * pos)
    kinetic = 0.5 * vel * vel * 2500
    energy = (height + 1.0) / 2.0 + kinetic
    energy_rate = int(np.clip(energy * 200, 0, 255))

    return [pos_rate, vel_mag, vel_dir, left_slope, right_slope, energy_rate]


def momentum_heuristic(obs):
    """The optimal heuristic: push in direction of velocity."""
    _, vel = obs
    if vel > 0:
        return 2  # Push right
    else:
        return 0  # Push left


def run_imitation_learning(
    n_episodes: int = 500,
    n_neurons: int = 3000,
    seed: int = 42,
):
    """
    Phase 1: Pure imitation - train SNN to match heuristic
    Phase 2: Refinement - use RL to improve on heuristic
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar Imitation Learning + R-STDP Refinement")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    n_inputs = 6
    n_outputs = 3
    ticks_per_step = 10

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=80,
        leak=200,
        connectivity="layered",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=40, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.85,
        ltp_rate=0.25,
        ltd_rate=0.12,
        learning_rate=0.08,  # Higher learning rate for imitation
    )

    episode_lengths = []
    best_length = 200
    successes = 0

    # Track accuracy of imitating heuristic
    correct_matches = 0
    total_actions = 0

    # Phase parameters
    imitation_episodes = 200  # Pure imitation phase

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0
        episode_matches = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = obs_to_rates(obs)

            # Get heuristic action (teacher)
            teacher_action = momentum_heuristic(obs)

            # Get SNN action (student)
            student_action = snn.step_learn_rates(rates, ticks_per_step)

            # In imitation phase, always follow teacher
            # In refinement phase, follow student
            if episode < imitation_episodes:
                action = teacher_action
            else:
                action = student_action

            # Track match rate
            if student_action == teacher_action:
                episode_matches += 1
            total_actions += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Imitation reward: +1 if matches teacher, -1 if not
            if episode < imitation_episodes:
                if student_action == teacher_action:
                    imitation_reward = 1.0
                else:
                    imitation_reward = -0.5
                snn.learn_action(teacher_action, imitation_reward, 0.8)
            else:
                # Refinement phase: use environment reward shaping
                pos, vel = next_obs
                shaped_reward = 0.0

                # Reward momentum
                if abs(vel) > abs(obs[1]):
                    shaped_reward += 0.5

                # Reward height gain
                if pos > obs[0]:
                    shaped_reward += (pos - obs[0]) * 20

                # Goal bonus
                if terminated and pos >= 0.5:
                    shaped_reward += 10.0
                    successes += 1

                scaled = np.clip(shaped_reward / 5.0, -1.0, 1.0)
                snn.learn_action(action, scaled, 0.5)

            obs = next_obs

        correct_matches += episode_matches
        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        # Progress report
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_lengths[-50:])
            elapsed = time.time() - start_time
            match_rate = 100 * correct_matches / total_actions if total_actions > 0 else 0
            phase = "IMITATION" if episode < imitation_episodes else "REFINEMENT"
            print(f"Episode {episode+1:4d} [{phase:10}]: avg={recent_avg:6.1f}, best={best_length:3d}, "
                  f"match={match_rate:.1f}%, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)

    # Split by phase
    imitation_avg = np.mean(episode_lengths[:imitation_episodes])
    refinement_avg = np.mean(episode_lengths[imitation_episodes:])
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))

    print(f"  Imitation phase avg: {imitation_avg:.1f}")
    print(f"  Refinement phase avg: {refinement_avg:.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_length}")
    print(f"  Final match rate: {100 * correct_matches / total_actions:.1f}%")
    print(f"  Successes: {successes} ({100*successes/n_episodes:.1f}%)")

    if best_window < 200:
        print(f"  vs Random (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    env.close()
    return episode_lengths, successes


def run_curriculum_learning(
    n_episodes: int = 500,
    n_neurons: int = 3000,
    seed: int = 42,
):
    """
    Curriculum learning: gradually reduce heuristic assistance.

    Unlike sudden switch, this slowly mixes in SNN decisions.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar Curriculum Learning (Gradual Handoff)")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    n_inputs = 6
    n_outputs = 3
    ticks_per_step = 10

    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=80,
        leak=200,
        connectivity="layered",
        seed=seed,
    )

    snn.apply_threshold_asymmetry(spread=40, seed=seed)
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=0.85,
        ltp_rate=0.25,
        ltd_rate=0.12,
        learning_rate=0.06,
    )

    episode_lengths = []
    best_length = 200
    successes = 0

    # Curriculum: start with 100% teacher, decay to 10%
    teacher_prob = 1.0
    teacher_decay = 0.995
    teacher_min = 0.1

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0
        matches = 0
        total = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = obs_to_rates(obs)
            teacher_action = momentum_heuristic(obs)
            student_action = snn.step_learn_rates(rates, ticks_per_step)

            # Mix teacher and student
            if np.random.random() < teacher_prob:
                action = teacher_action
            else:
                action = student_action

            total += 1
            if student_action == teacher_action:
                matches += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            pos, vel = next_obs

            # Combined reward: imitation + task
            imitation_bonus = 0.5 if student_action == teacher_action else -0.3

            task_reward = 0.0
            if abs(vel) > abs(obs[1]):
                task_reward += 0.3
            if pos > obs[0]:
                task_reward += (pos - obs[0]) * 10
            if terminated and pos >= 0.5:
                task_reward += 10.0
                successes += 1

            # Weight imitation vs task by curriculum progress
            imitation_weight = teacher_prob
            task_weight = 1.0 - teacher_prob
            combined = imitation_weight * imitation_bonus + task_weight * task_reward

            scaled = np.clip(combined / 5.0, -1.0, 1.0)
            snn.learn_action(action, scaled, 0.5)

            obs = next_obs

        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        teacher_prob = max(teacher_min, teacher_prob * teacher_decay)

        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_lengths[-50:])
            match_rate = 100 * matches / total if total > 0 else 0
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d}: avg={recent_avg:6.1f}, best={best_length:3d}, "
                  f"teacher={teacher_prob:.2f}, match={match_rate:.0f}%, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_lengths[:n_episodes//2])
    second_half = np.mean(episode_lengths[n_episodes//2:])
    best_window = min(np.mean(episode_lengths[i:i+50]) for i in range(len(episode_lengths) - 49))

    print(f"  First half avg: {first_half:.1f}")
    print(f"  Second half avg: {second_half:.1f}")
    print(f"  Best 50-ep window: {best_window:.1f}")
    print(f"  Best episode: {best_length}")
    print(f"  Successes: {successes} ({100*successes/n_episodes:.1f}%)")

    if best_window < 200:
        print(f"  vs Random (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    env.close()
    return episode_lengths, successes


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TEST 1: Imitation Learning (Phase-based)")
    print("="*70 + "\n")

    lengths1, successes1 = run_imitation_learning(
        n_episodes=500,
        n_neurons=3000,
    )

    print("\n" + "="*70)
    print("  TEST 2: Curriculum Learning (Gradual Handoff)")
    print("="*70 + "\n")

    lengths2, successes2 = run_curriculum_learning(
        n_episodes=500,
        n_neurons=3000,
    )

    print("\n" + "="*70)
    print("  FINAL COMPARISON")
    print("="*70)

    window1 = min(np.mean(lengths1[i:i+50]) for i in range(len(lengths1)-49))
    window2 = min(np.mean(lengths2[i:i+50]) for i in range(len(lengths2)-49))

    print(f"  Imitation Learning:   best_window={window1:.1f}, successes={successes1}")
    print(f"  Curriculum Learning:  best_window={window2:.1f}, successes={successes2}")
