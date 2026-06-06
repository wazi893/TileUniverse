"""
Test: Channeled Learning with Fully Isolated Inputs

Use inputs that only activate one channel at a time.
"""

import numpy as np
from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode

print("=== Channeled Learning with Isolated Inputs ===\n")

config = QuantumSNNConfig(
    n_inputs=4,
    hidden_layers=[16],
    n_outputs=2,
    mode=InterferenceMode.none(),
    ticks_per_decision=30,
    learning_enabled=True,
)

brain = QuantumSNN(config, seed=42)

# Fully isolated inputs:
# Context A: Only ch0 active -> should learn action 0
# Context B: Only ch1 active -> should learn action 1
context_a = [200, 200, 0, 0]    # Only inputs 0-1
context_b = [0, 0, 200, 200]    # Only inputs 2-3

print("Isolated inputs:")
print(f"  Context A: {context_a} (ch0 only) -> action 0")
print(f"  Context B: {context_b} (ch1 only) -> action 1")
print()

# Check initial behavior
print("Initial behavior:")
for name, ctx, want in [("A", context_a, 0), ("B", context_b, 1)]:
    actions = []
    for _ in range(50):
        actions.append(brain.decide(ctx))
    correct = actions.count(want) / len(actions) * 100
    spikes = list(brain.output_spike_counts)
    print(f"  {name}: spikes={spikes}, correct={correct:.0f}%")

# Train
print("\nTraining...")
correct_a = 0
correct_b = 0
total_a = 0
total_b = 0

for trial in range(500):
    if trial % 2 == 0:
        inputs = context_a
        optimal = 0
        total_a += 1
    else:
        inputs = context_b
        optimal = 1
        total_b += 1

    action = brain.decide(inputs)

    if action == optimal:
        brain.learn_channeled(60, action)
        if trial % 2 == 0:
            correct_a += 1
        else:
            correct_b += 1
    # No punishment for wrong

    if trial % 100 == 99:
        acc_a = correct_a / total_a * 100
        acc_b = correct_b / total_b * 100
        overall = (correct_a + correct_b) / (total_a + total_b) * 100
        print(f"Trial {trial+1}: A={acc_a:.0f}%, B={acc_b:.0f}%, Overall={overall:.0f}%")

overall = (correct_a + correct_b) / (total_a + total_b) * 100
print()

# Final behavior
print("Final behavior:")
for name, ctx, want in [("A", context_a, 0), ("B", context_b, 1)]:
    actions = []
    for _ in range(50):
        actions.append(brain.decide(ctx))
    correct = actions.count(want) / len(actions) * 100
    spikes = list(brain.output_spike_counts)
    print(f"  {name}: spikes={spikes}, correct={correct:.0f}%")

print()
if overall > 75:
    print(f"*** SUCCESS: {overall:.1f}% > 75% target ***")
else:
    print(f"Result: {overall:.1f}%")
