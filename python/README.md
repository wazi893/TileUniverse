<p align="center">
  <img src="assets/logo.svg" alt="TileUniverse" width="400">
</p>

<h1 align="center">TileUniverse</h1>

<p align="center">
  <strong>Trillion-Scale GPU Simulation Engine</strong>
</p>

<p align="center">
  <a href="#performance">15.8 TCOPS</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Docs</a> •
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/cuda-12.0+-green.svg" alt="CUDA 12.0+">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

---

## Overview

TileUniverse is a high-performance simulation substrate for running massively parallel worlds on GPU hardware. It achieves **15.8 TCOPS** on the quantum substrate and **200 billion evaluations per second** on the cellular substrate (RTX 5090)—using consumer GPUs through optimized CUDA kernels, Tensor Cores, and depth-batched execution.

The engine provides:

- **Trillion-Scale Throughput** — 15.8 TCOPS quantum, 200B evals/sec cellular (RTX 5090)
- **Parallel World Simulation** — Run 100+ independent universes simultaneously
- **Tensor Core Acceleration** — WMMA intrinsics for 15× speedup over FP32
- **Reversible Physics** — Full state history with rewind/forward capabilities
- **RL Integration** — Gymnasium-compatible vectorized environments
- **V2 Tile CPU + Synth** — Assemble programs and compile truth tables from Python

TileUniverse is designed for researchers and engineers who need simulation throughput that traditional CPU-bound frameworks cannot provide.

---

## Performance

Verified benchmarks (January 2026):

### RTX 5090 (32GB, Blackwell)

| Substrate | Metric | Value |
|-----------|--------|-------|
| Quantum | Peak Throughput | **15.8 TCOPS** (PureMMA) |
| Cellular | Logic Evaluations | **200B evals/sec** |

### RTX 4070 (12GB, Ada Lovelace)

| Substrate | Metric | Value |
|-----------|--------|-------|
| Quantum | Throughput | **2.5 TCOPS** (WMMA) |
| Cellular | Logic Evaluations | **40B evals/sec** |

### Common Features

| Feature | Value |
|---------|-------|
| Parallel Worlds | 100+ concurrent |
| World Dimensions | Up to 4096×4096 |
| RL Environment Steps | ~1M steps/sec |

<details>
<summary>Benchmark Details (RTX 5090)</summary>

```
======================================================================
  TileUniverse Quantum Benchmark (January 2026)
======================================================================

  GPU: NVIDIA GeForce RTX 5090 (32GB, Blackwell)
  Backend: PureMMA Tensor Cores
  Qubit Range: 12-32 qubits

  PERFORMANCE: 15.8 TCOPS peak (24 qubits)

======================================================================
  TileUniverse Cellular Benchmark
======================================================================

  GPU: NVIDIA GeForce RTX 5090
  Worlds: 10
  Size: 512×512
  Depth: 50

  PERFORMANCE: 200B logic evals/sec
======================================================================
```

RTX 4070 achieves ~2.5 TCOPS quantum and ~40B cellular evals/sec.

</details>

---

## Features

### Parallel World Engine

Simulate hundreds of independent universes with a single API call. Each world maintains isolated state while sharing GPU resources efficiently.

```python
import tileuniverse as tu

# Initialize 100 parallel universes
engine = tu.Engine(worlds=100, size=(256, 256), ruleset="gol")

# Evolve all worlds simultaneously
engine.evolve(1000)

# Access individual world states
for w in range(engine.worlds):
    state = engine.get_world(w)  # Returns numpy array
```

### Reversible Simulation

Full state history tracking enables temporal navigation—rewind to any previous state, replay forward, or jump to specific points in the simulation timeline.

```python
engine = tu.Engine(worlds=1, size=(64, 64), ruleset="gol", reversible=True)
engine.evolve(100)

engine.rewind(50)      # Return to step 50
engine.forward(25)     # Advance to step 75
engine.goto_history(0) # Jump to initial state
```

### Reinforcement Learning

Native integration with Stable Baselines 3 and Gymnasium. Train agents across parallel environments at GPU speed.

```python
from tileuniverse.rl import TileUniverseSB3VecEnv
from stable_baselines3 import PPO

env = TileUniverseSB3VecEnv(worlds=64, size=(32, 32))
model = PPO("MlpPolicy", env, n_steps=2048)
model.learn(total_timesteps=500_000)
```

### Trajectory Analysis

Record, replay, and analyze agent trajectories with frame-by-frame inspection capabilities.

```python
from tileuniverse.rl import record_episode, TrajectoryPlayer

trajectory = record_episode(env, policy, max_steps=100)
player = TrajectoryPlayer(trajectory)

player.goto(50)    # Navigate to frame 50
player.rewind(10)  # Step backward
print(player.render_frame())  # ASCII visualization
```

---

## Quick Start

### Installation

```bash
pip install tileuniverse
```

**Requirements:**
- Python 3.8+
- NVIDIA GPU with CUDA support for the high-throughput simulation backends
- CUDA Toolkit 12.0+ for GPU-enabled builds
- No CUDA required for the V2 CPU and synth Python APIs

### Basic Usage

```python
import tileuniverse as tu

# Create engine
engine = tu.Engine(worlds=5, size=(512, 512), ruleset="gol")

# Initialize world with random pattern
engine.randomize(0, density=0.3)

# Run simulation
engine.evolve(100)

# Extract state as numpy array
world = engine.get_world(0)
print(f"Shape: {world.shape}")  # (512, 512)
```

### V2 CPU and Synth

```python
from tileuniverse import V2Cpu, synthesize

cpu = V2Cpu.from_asm("LDI R0, 42\nHALT")
cpu.run(100)
print(cpu.reg(0))
print(cpu.assist_counters())

result = synthesize(truth_table=0xE8, num_inputs=3)
print(result.summary())
```

### Verify Installation

```bash
python -c "import tileuniverse as tu; print(f'GPU: {tu.cuda_device_name()}')"
```

---

## Documentation

### Engine API

```python
engine = tu.Engine(
    worlds=5,              # Parallel universe count
    size=(512, 512),       # World dimensions (powers of 2)
    ruleset="gol",         # Simulation ruleset
    seed=42,               # Random seed
    reversible=False,      # Enable state history
    max_history=1000       # Maximum stored states
)
```

| Method | Description |
|--------|-------------|
| `evolve(steps)` | Advance simulation |
| `get_world(index)` | Extract world state as numpy array |
| `set_world(index, array)` | Set world state from numpy array |
| `randomize(index, density)` | Initialize with random pattern |
| `rewind(steps)` | Navigate backward in time |
| `forward(steps)` | Navigate forward in time |
| `reset()` | Clear all world states |

### Rulesets

| Ruleset | Description |
|---------|-------------|
| `gol` | Conway's Game of Life |
| `rule110` | Wolfram's Rule 110 |
| `wire` | Signal propagation |
| `logic` | Logic gate simulation |

### Configuration Files

```yaml
# scenario.yaml
name: experiment-001
worlds: 10
width: 512
height: 512
ruleset: gol
seed: 42
reversible: true
max_history: 500
```

```python
engine = tu.load_config("scenario.yaml")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                                │
│                   tileuniverse.Engine                            │
├─────────────────────────────────────────────────────────────────┤
│                     RL Integration                               │
│         TileUniverseSB3VecEnv • ParallelGridworld               │
├─────────────────────────────────────────────────────────────────┤
│                    Rust Core Engine                              │
│              PyO3 Bindings • State Management                    │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Kernels                                 │
│     Depth-Batched Execution • L2 Cache Optimization             │
├─────────────────────────────────────────────────────────────────┤
│                   Parallel Worlds                                │
│           World₀ │ World₁ │ World₂ │ ... │ Worldₙ              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Building from Source

```bash
git clone https://github.com/tileuniverse/tileuniverse
cd tileuniverse/engine/python

# Install build tools
pip install maturin

# Development build (CPU-only)
maturin develop --release

# Optional CUDA-enabled build
# maturin develop --release --features cuda

# Production wheel
maturin build --release
```

---

## Citation

If TileUniverse contributes to your research, please cite:

```bibtex
@software{tileuniverse2024,
  title     = {TileUniverse: GPU-Accelerated Parallel Universe Simulation Engine},
  author    = {TileUniverse Contributors},
  year      = {2024},
  url       = {https://github.com/tileuniverse/tileuniverse},
  note      = {40B logic evals/sec on consumer GPU hardware}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for researchers who need simulation throughput, not simulation overhead.</sub>
</p>
