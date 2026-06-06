"""Test reversible mode - time travel through simulation."""

import tileuniverse as tu

print("=== Time Travel Demo ===\n")

# Create engine with reversible mode
engine = tu.Engine(
    worlds=1,
    size=(64, 64),
    ruleset="gol",
    reversible=True,
    max_history=500
)
engine.randomize(0, 0.3)

print(f"Reversible mode: {engine.is_reversible}")
print()

# Evolve forward
print("Evolving 100 steps...")
engine.evolve(100)
state_100 = engine.get_world(0).copy()
alive_100 = (state_100 != 0).sum()
print(f"  Step {engine.step}: {alive_100} alive cells")

# Rewind 50 steps
print("\nRewinding 50 steps...")
engine.rewind(50)
state_50 = engine.get_world(0)
alive_50 = (state_50 != 0).sum()
print(f"  Step {engine.step}: {alive_50} alive cells")

# Fast forward back
print("\nFast-forwarding 50 steps...")
engine.forward(50)
state_restored = engine.get_world(0)
print(f"  Step {engine.step}: {(state_restored != 0).sum()} alive cells")

# Verify state is identical
import numpy as np
match = np.array_equal(state_100, state_restored)
print(f"\nState restored perfectly: {match}")
print(f"History length: {engine.history_len} states saved")
