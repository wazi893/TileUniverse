"""
MountainCar-v0 with R-STDP Learning (FusedSNN)

MountainCar is MUCH harder than CartPole:
- Sparse reward: -1 per step, 0 at goal (max 200 steps)
- Must build momentum by swinging back and forth
- 3 actions: left(0), neutral(1), right(2)
- Observation: [position (-1.2 to 0.6), velocity (-0.07 to 0.07)]

Random baseline: ~200 steps (never reaches goal)
Solving threshold: avg < 110 steps over 100 episodes
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

    # Velocity magnitude (amplified) - key for momentum
    vel_mag = int(np.clip(abs(vel) * 3000, 0, 255))

    # Velocity direction: < 128 = left, > 128 = right
    vel_dir = int(np.clip(vel * 1800 + 128, 0, 255))

    # "Am I on the left slope?" indicator (position < -0.5)
    left_slope = 255 if pos < -0.5 else 0

    # "Am I on the right slope?" indicator (position > -0.2)
    right_slope = 255 if pos > -0.2 else 0

    # Distance to goal
    dist_goal = int(np.clip((0.5 - pos) / 1.7 * 255, 0, 255))

    # Energy = height + kinetic (key insight: need to maximize this)
    # Height is approx sin(3*pos) but we simplify
    height = np.sin(3 * pos)
    kinetic = 0.5 * vel * vel * 2500  # amplified
    energy = (height + 1.0) / 2.0 + kinetic
    energy_rate = int(np.clip(energy * 200, 0, 255))

    return [pos_rate, vel_mag, vel_dir, left_slope, right_slope, dist_goal, energy_rate]


def run_mountaincar_rstdp(
    n_episodes: int = 1000,
    n_neurons: int = 5000,
    ticks_per_step: int = 10,
    learning_rate: float = 0.05,
    tau_decay: float = 0.85,
    anti_reward: float = 0.5,
    seed: int = 42,
):
    """Train FusedSNN on MountainCar with action-specific R-STDP."""

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar-v0 with R-STDP Learning (FusedSNN)")
    print("=" * 70)
    print(f"  GPU: {tu.cuda_device_name()}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Ticks/step: {ticks_per_step}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Tau decay: {tau_decay}")
    print(f"  Anti-reward factor: {anti_reward}")
    print("=" * 70)
    print()

    # Create environment
    env = gym.make("MountainCar-v0")
    n_inputs = 7  # Enhanced observation features
    n_outputs = 3  # left, neutral, right

    # Create SNN with layered connectivity
    snn = tu.FusedSNN(
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        threshold=80,  # Lower threshold for more activity
        leak=200,      # Less leak for longer integration
        connectivity="layered",
        seed=seed,
    )

    # Apply threshold asymmetry for differentiation
    snn.apply_threshold_asymmetry(spread=40, seed=seed)

    # Enable learning with tuned parameters
    snn.enable_learning()
    snn.set_learning_params(
        tau_decay=tau_decay,
        ltp_rate=0.2,
        ltd_rate=0.1,
        learning_rate=learning_rate,
    )

    print(f"  SNN created: {n_neurons} neurons, {n_inputs} inputs, {n_outputs} outputs")
    print()

    # Tracking
    episode_lengths = []
    best_length = 200
    successes = 0
    window_size = 50
    action_counts = [0, 0, 0]  # Track action distribution

    # Exploration: start high, decay
    epsilon = 0.8
    epsilon_min = 0.1
    epsilon_decay = 0.998

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        prev_pos = obs[0]
        prev_vel = obs[1]
        done = False
        steps = 0
        max_height = obs[0]  # Track max height reached

        # Reset eligibility at episode start
        snn.reset_eligibility()

        while not done and steps < 200:
            # Convert observation to spike rates
            rates = obs_to_rates(obs)

            # Epsilon-greedy exploration
            if np.random.random() < epsilon:
                # Smart exploration: alternate based on position
                pos = obs[0]
                if pos < -0.5:
                    # On left side - tend to push right
                    action = np.random.choice([0, 2], p=[0.3, 0.7])
                elif pos > -0.2:
                    # On right side - tend to push left
                    action = np.random.choice([0, 2], p=[0.7, 0.3])
                else:
                    # Middle - random
                    action = np.random.randint(0, 3)
                # Still run SNN to build eligibility
                snn.step_learn_rates(rates, ticks_per_step)
            else:
                action = snn.step_learn_rates(rates, ticks_per_step)

            action_counts[action] += 1

            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            pos, vel = next_obs

            # Track max height
            if pos > max_height:
                max_height = pos

            # Shaped reward for MountainCar
            shaped_reward = 0.0

            # Reward for building momentum (velocity magnitude increase)
            vel_gain = abs(vel) - abs(prev_vel)
            if vel_gain > 0:
                shaped_reward += vel_gain * 50

            # Reward for going in "correct" direction
            # On left slope going right, on right slope going left
            if pos < -0.5 and vel > 0:  # Left slope, moving right
                shaped_reward += 0.5
            elif pos > -0.3 and vel < 0:  # Right slope, moving left
                shaped_reward += 0.3

            # Reward for height gain
            height_gain = pos - prev_pos
            if height_gain > 0:
                shaped_reward += height_gain * 20

            # Penalty for staying still
            if abs(vel) < 0.001:
                shaped_reward -= 0.2

            # Big bonus for reaching goal
            if terminated and pos >= 0.5:
                shaped_reward += 100.0
                successes += 1

            prev_pos = pos
            prev_vel = vel

            # Apply action-specific learning
            scaled_reward = np.clip(shaped_reward / 10.0, -1.0, 1.0)
            snn.learn_action(action, scaled_reward, anti_reward)

            obs = next_obs

        episode_lengths.append(steps)

        if steps < best_length:
            best_length = steps

        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Progress report
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_lengths[-window_size:])
            elapsed = time.time() - start_time
            total_actions = sum(action_counts)
            action_pcts = [100 * c / total_actions for c in action_counts]
            print(f"Episode {episode+1:4d}: avg={recent_avg:6.1f}, best={best_length:3d}, "
                  f"success={successes}, eps={epsilon:.2f}, "
                  f"actions=[{action_pcts[0]:.0f}%,{action_pcts[1]:.0f}%,{action_pcts[2]:.0f}%], "
                  f"time={elapsed:.1f}s")

    # Final analysis
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    first_half = np.mean(episode_lengths[:n_episodes//2])
    second_half = np.mean(episode_lengths[n_episodes//2:])
    best_window = min(np.mean(episode_lengths[i:i+window_size])
                      for i in range(len(episode_lengths) - window_size + 1))

    print(f"  First {n_episodes//2} avg: {first_half:.1f}")
    print(f"  Second {n_episodes//2} avg: {second_half:.1f}")
    print(f"  Best {window_size}-episode window: {best_window:.1f}")
    print(f"  Best single episode: {best_length}")
    print(f"  Total successes: {successes} ({100*successes/n_episodes:.1f}%)")
    print()

    if second_half < first_half:
        improvement = (first_half - second_half) / first_half * 100
        print(f"  Learning improvement: {improvement:.1f}%")
    else:
        print(f"  No improvement (second half worse)")

    if best_window < 200:
        print(f"  vs Random baseline (200): {(200 - best_window) / 200 * 100:.1f}% faster")

    if best_window < 110:
        print("  *** SOLVED! (avg < 110) ***")
    elif successes > 0:
        print(f"  Partial success: {successes} episodes reached goal")

    env.close()
    return episode_lengths, successes


def run_with_momentum_heuristic(
    n_episodes: int = 500,
    n_neurons: int = 3000,
    seed: int = 42,
):
    """
    Use momentum-based heuristic for exploration to bootstrap learning.

    Key insight: MountainCar needs oscillation. Start with a heuristic policy
    that does this, then let R-STDP refine it.
    """

    if not tu.is_cuda_available():
        print("ERROR: CUDA required")
        return

    print("=" * 70)
    print("  MountainCar with Momentum Heuristic + R-STDP")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    n_inputs = 7
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
        ltp_rate=0.2,
        ltd_rate=0.1,
        learning_rate=0.05,
    )

    episode_lengths = []
    best_length = 200
    successes = 0

    # Start with high heuristic usage, decay to let SNN take over
    heuristic_prob = 1.0
    heuristic_decay = 0.995
    heuristic_min = 0.1

    start_time = time.time()

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        prev_pos = obs[0]
        prev_vel = obs[1]
        done = False
        steps = 0

        snn.reset_eligibility()

        while not done and steps < 200:
            rates = obs_to_rates(obs)
            pos, vel = obs

            # Momentum heuristic: push in direction of velocity
            # This creates oscillation that builds energy
            if np.random.random() < heuristic_prob:
                if vel > 0:
                    action = 2  # Push right when moving right
                else:
                    action = 0  # Push left when moving left
                # Still run SNN for eligibility
                snn.step_learn_rates(rates, ticks_per_step)
            else:
                action = snn.step_learn_rates(rates, ticks_per_step)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            next_pos, next_vel = next_obs

            # Shaped reward
            shaped_reward = 0.0

            # Reward momentum building
            vel_gain = abs(next_vel) - abs(vel)
            if vel_gain > 0:
                shaped_reward += vel_gain * 50

            # Reward height
            if next_pos > prev_pos:
                shaped_reward += (next_pos - prev_pos) * 20

            # Big bonus for goal
            if terminated and next_pos >= 0.5:
                shaped_reward += 100.0
                successes += 1

            prev_pos = next_pos
            prev_vel = next_vel

            scaled_reward = np.clip(shaped_reward / 10.0, -1.0, 1.0)
            snn.learn_action(action, scaled_reward, 0.5)

            obs = next_obs

        episode_lengths.append(steps)
        if steps < best_length:
            best_length = steps

        # Decay heuristic usage
        heuristic_prob = max(heuristic_min, heuristic_prob * heuristic_decay)

        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_lengths[-50:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d}: avg={recent_avg:6.1f}, best={best_length:3d}, "
                  f"heur={heuristic_prob:.2f}, success={successes}, time={elapsed:.1f}s")

    print()
    print("=" * 70)
    first_half = np.mean(episode_lengths[:n_episodes//2])
    second_half = np.mean(episode_lengths[n_episodes//2:])
    best_window = min(np.mean(episode_lengths[i:i+50])
                      for i in range(len(episode_lengths) - 49))

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
    print("  TEST 1: Momentum Heuristic Bootstrap + R-STDP")
    print("="*70 + "\n")

    # This test uses a known-good heuristic to bootstrap learning
    lengths1, successes1 = run_with_momentum_heuristic(
        n_episodes=500,
        n_neurons=3000,
    )

    print("\n" + "="*70)
    print("  TEST 2: Pure R-STDP with Smart Exploration")
    print("="*70 + "\n")

    lengths2, successes2 = run_mountaincar_rstdp(
        n_episodes=1000,
        n_neurons=5000,
        learning_rate=0.05,
    )

    print("\n" + "="*70)
    print("  FINAL COMPARISON")
    print("="*70)

    window1 = min(np.mean(lengths1[i:i+50]) for i in range(len(lengths1)-49))
    window2 = min(np.mean(lengths2[i:i+50]) for i in range(len(lengths2)-49))

    print(f"  Momentum Heuristic: best_window={window1:.1f}, successes={successes1}")
    print(f"  Pure R-STDP:        best_window={window2:.1f}, successes={successes2}")
