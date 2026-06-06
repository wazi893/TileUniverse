"""Test different rulesets - GoL, Rule110, Wire, Logic."""

import tileuniverse as tu

print("=== Ruleset Comparison ===\n")

rulesets = ["gol", "rule110", "wire", "logic"]

for ruleset in rulesets:
    engine = tu.Engine(worlds=1, size=(128, 128), ruleset=ruleset)
    engine.randomize(0, 0.3)

    initial = (engine.get_world(0) != 0).sum()
    engine.evolve(50)
    final = (engine.get_world(0) != 0).sum()

    change = final - initial
    pct = (change / initial * 100) if initial > 0 else 0

    print(f"{ruleset:8s}: {initial:5d} -> {final:5d} ({change:+5d}, {pct:+.1f}%)")

print()
print("Each ruleset has different dynamics!")
