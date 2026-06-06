"""Quick test script for TileUniverse."""

import tileuniverse as tu

print(f"TileUniverse v{tu.__version__}")
print(f"GPU available: {tu.is_cuda_available()}")
if tu.is_cuda_available():
    print(f"GPU: {tu.cuda_device_name()}")
print()

# Basic engine test
e = tu.Engine(worlds=5, size=(256, 256), ruleset="gol")
e.randomize(0, 0.3)
e.evolve(50)
print(f"World shape: {e.get_world(0).shape}")
print(f"Step: {e.step}")
print(f"Alive cells: {(e.get_world(0) != 0).sum()}")
print()

# Quick benchmark
if tu.is_cuda_available():
    result = tu.benchmark(worlds=5, steps=100)
    print(f"Benchmark: {result.evals_per_sec / 1e9:.1f}B evals/sec")
else:
    print("Skipping GPU benchmark (no CUDA)")

print()
print("All tests passed!")
