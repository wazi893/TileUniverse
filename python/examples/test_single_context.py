"""
Test: Single Context Learning

Only train on context A to see if the network can learn ONE mapping.
"""

import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode

print("=== Single Context Learning ===\n")

config = QuantumSNNConfig(
    n_inputs=4,
    hidden_layers=[32, 16],
    n_outputs=2,
    mode=InterferenceMode.none(),
    ticks_per_decision=30,
    learning_enabled=True,
)

brain = QuantumSNN(config, seed=42)

context_a = [200, 50, 128, 128]
optimal_action = 0  # We want action 0 for context A

print("Training ONLY on context A: action 0 is correct")
print()

# Check initial behavior
actions_before = []
for _ in range(100):
    actions_before.append(brain.decide(context_a))
print(f"Before training: action 0 = {actions_before.count(0)}%, action 1 = {actions_before.count(1)}%")

# Train
for trial in range(500):
    action = brain.decide(context_a)
    if action == optimal_action:
        brain.learn_deep_gated(80, action)

    if trial % 100 == 99:
        # Check current behavior
        test_actions = []
        for _ in range(50):
            test_actions.append(brain.decide(context_a))
        pct_0 = test_actions.count(0) / len(test_actions) * 100
        spikes = list(brain.output_spike_counts)
        print(f"Trial {trial+1:>3}: action 0 = {pct_0:.0f}%, spikes={spikes}")

# Final check
actions_after = []
for _ in range(100):
    actions_after.append(brain.decide(context_a))
print()
print(f"After training: action 0 = {actions_after.count(0)}%, action 1 = {actions_after.count(1)}%")

improvement = actions_after.count(0) - actions_before.count(0)
if improvement > 10:
    print(f"\nLEARNING: Improved by {improvement} percentage points!")
else:
    print(f"\nNO SIGNIFICANT LEARNING: Change = {improvement} points")
