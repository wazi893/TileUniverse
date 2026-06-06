"""Test parallel worlds - simulate 100 universes at once."""

import tileuniverse as tu
import numpy as np

print("=== Parallel Worlds Demo ===\n")

# Create 100 parallel universes
engine = tu.Engine(worlds=100, size=(128, 128), ruleset="gol")

# Initialize each with different density
for w in range(engine.worlds):
    density = 0.1 + (w / 100) * 0.4  # 0.1 to 0.5
    engine.randomize(w, density)

print(f"Created {engine.worlds} parallel universes")
print(f"Grid size: {engine.size}")
print()

# Evolve all 100 worlds simultaneously
engine.evolve(100)

# Gather statistics across all universes
alive_counts = []
for w in range(engine.worlds):
    world = engine.get_world(w)
    alive_counts.append((world != 0).sum())

alive_counts = np.array(alive_counts)
print(f"After 100 steps across all universes:")
print(f"  Min alive:  {alive_counts.min():,}")
print(f"  Max alive:  {alive_counts.max():,}")
print(f"  Mean alive: {alive_counts.mean():,.0f}")
print(f"  Std dev:    {alive_counts.std():,.0f}")
print()
print("100 universes simulated in parallel!")
