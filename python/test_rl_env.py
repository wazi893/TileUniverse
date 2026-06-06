"""Test RL environment - parallel gridworld for training."""

from tileuniverse.rl import ParallelGridworld
import numpy as np

print("=== RL Environment Demo ===\n")

# Create 16 parallel training environments
env = ParallelGridworld(worlds=16, size=(16, 16), seed=42)
obs = env.reset()

print(f"Parallel environments: {env.worlds}")
print(f"Observation shape: {obs.shape}")
print(f"Actions: 5 (noop, up, down, left, right)")
print()

# Run 100 random steps
total_reward = 0
goals_reached = 0

for step in range(100):
    actions = np.random.randint(0, 5, size=env.worlds)
    obs, rewards, dones, info = env.step(actions)
    total_reward += rewards.sum()
    goals_reached += info["goal_reached"].sum()

    # Auto-reset done environments
    if dones.any():
        env.reset(world_mask=dones)

print(f"After 100 random steps:")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Goals reached: {goals_reached}")
print()

# Show one world
print("World 0 state:")
print(env.render_world(0))
