"""Explore several truth tables and print validation status."""
from tileuniverse import synthesize

functions = [
    ("2-input AND", 0x8, 2),
    ("2-input OR", 0xE, 2),
    ("2-input XOR", 0x6, 2),
    ("3-input MAJ", 0xE8, 3),
    ("3-input AND", 0x80, 3),
    ("3-input OR", 0xFE, 3),
    ("3-input XOR", 0x96, 3),
]

print(f"{'Function':<16} {'ANDs':>5} {'Mapped':>6} {'Grid':>11} {'Routes':>7} {'Status':>8}")
print("-" * 66)

for name, tt, n in functions:
    result = synthesize(truth_table=tt, num_inputs=n)
    status = "PASS" if result.validated else "FAIL"
    grid = f"{result.grid_width}x{result.grid_height}x{result.num_layers}"
    print(
        f"{name:<16} {result.num_and_nodes:>5} {result.mapped_gates:>6} "
        f"{grid:>11} {result.route_count:>7} {status:>8}"
    )
    if not result.validated:
        print(
            f"  first mismatch: out={result.first_mismatch_output} "
            f"combo={result.first_mismatch_input_combo} "
            f"expected={result.first_mismatch_expected} "
            f"actual={result.first_mismatch_actual}"
        )
