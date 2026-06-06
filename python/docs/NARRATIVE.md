# TileUniverse Product Narrative

## Mission Statement

**TileUniverse exists to make high-performance parallel simulation accessible to individual researchers.**

We believe computational throughput should not be a barrier to scientific exploration. By co-designing simulation substrates around GPU hardware characteristics, we deliver performance previously requiring specialized clusters—on consumer hardware, through a clean Python API.

---

## The Problem

Modern research demands simulation at scale:

- **Reinforcement learning** requires millions of environment steps per training run
- **Artificial life** experiments need large populations evolving over billions of generations
- **Cellular automata** research requires exploration of vast parameter spaces

Traditional CPU-bound frameworks top out at tens of millions of evaluations per second—insufficient for serious research. Cloud compute is expensive. Specialized hardware is inaccessible.

**Researchers are bottlenecked by simulation throughput.**

---

## Our Solution

TileUniverse is a GPU-accelerated simulation substrate that achieves **40 billion logic evaluations per second** on consumer GPUs.

This isn't incremental improvement. It's three orders of magnitude faster than equivalent CPU implementations—enabling research workflows that were previously computationally prohibitive.

### How We Achieve This

1. **Depth-batched kernel execution**: Multiple simulation steps per kernel launch, amortizing overhead
2. **L2 cache-resident layouts**: World state fits in fast GPU cache, not slow DRAM
3. **Rust core with zero-copy Python bindings**: Native performance without serialization overhead
4. **Parallel universe architecture**: 100+ independent worlds share GPU resources efficiently

---

## Who We Serve

### Research Scientists
Computational biologists studying emergent behavior. Physicists exploring discrete spacetime models. Computer scientists investigating complexity theory.

**TileUniverse gives you the throughput to ask bigger questions.**

### RL Engineers
Training agents across parallel environments. Running hyperparameter sweeps. Collecting experience at scale.

**TileUniverse provides vectorized environments that match your GPU training speed.**

### Educators
Teaching cellular automata, complexity, emergence. Demonstrating computational universality. Visualizing parallel computation.

**TileUniverse makes GPU computing accessible through a clean Python API.**

---

## Design Principles

### Throughput Over Latency
We optimize for aggregate evaluation rate across many parallel worlds, not single-world step latency. This architectural decision enables our performance characteristics.

### Memory Hierarchy Awareness
Active world state stays in L2 cache. Memory transfers are minimized. Data layouts are coalesced. Every byte movement is intentional.

### Zero-Copy Interoperability
Python, Rust, and CUDA share memory without serialization. NumPy arrays map directly to GPU buffers. No intermediate copies.

### Minimal API Surface
```python
engine = tu.Engine(worlds=100, size=(256, 256), ruleset="gol")
engine.evolve(1000)
state = engine.get_world(0)
```

Three lines. That's it. Complexity lives in the implementation, not the interface.

---

## Technical Credibility

- **40.5 billion evals/sec** on RTX 4070 (verified benchmark)
- **800× faster** than multi-threaded CPU implementations
- **100+ parallel worlds** executing simultaneously
- **O(1) temporal navigation** with reversible state history
- **Native RL integration** with Gymnasium and Stable Baselines 3

Performance scales with GPU capability. RTX 5090 achieves ~200B evals/sec (verified January 2026).

---

## What We're Not

- **Not a game engine**: We don't render, animate, or handle input
- **Not a general-purpose simulator**: We focus on discrete cellular domains
- **Not a cloud service**: We run on your hardware, under your control
- **Not enterprise software**: No sales calls, no contracts, just open source

---

## The Vision

**A world where simulation throughput is unlimited and free.**

We're building the computational infrastructure for the next generation of complexity research. Today, it's cellular automata at 40 billion evals/sec. Tomorrow, it's the substrate for artificial life experiments that would take years on traditional hardware.

TileUniverse is the beginning of that infrastructure.

---

## Voice and Tone

### We Are
- **Technical**: Precise language, verifiable claims, reproducible benchmarks
- **Confident**: We built something genuinely fast; we can state that directly
- **Accessible**: Clean APIs, clear documentation, helpful defaults
- **Honest**: We acknowledge limitations alongside capabilities

### We Are Not
- **Hyperbolic**: No "revolutionary," "game-changing," or "world's best"
- **Condescending**: Technical users deserve technical depth
- **Vague**: Concrete numbers, not hand-wavy promises
- **Salesy**: Open source speaks for itself

### Example Copy

**Good:**
> TileUniverse achieves 40 billion logic evaluations per second on consumer GPUs—approximately 800× faster than optimized multi-threaded CPU implementations.

**Bad:**
> TileUniverse is an incredibly fast, revolutionary simulation engine that will change how you think about computing!

---

## Taglines

**Primary:**
> GPU-Accelerated Parallel Universe Simulation Engine

**Secondary options:**
> Simulation throughput, not simulation overhead.

> 40 billion reasons to rethink your simulator.

> The substrate for parallel worlds.

> Research-grade simulation. Consumer-grade hardware.

---

## Elevator Pitch

**30-second version:**

"TileUniverse is a GPU-accelerated simulation engine for parallel discrete worlds. It achieves 40 billion logic evaluations per second on consumer hardware—about 800 times faster than CPU implementations. If you're training RL agents, studying cellular automata, or running artificial life experiments, TileUniverse eliminates the simulation bottleneck."

**10-second version:**

"TileUniverse runs 100 parallel universe simulations at 40 billion evaluations per second on a single GPU. It's open source."

---

## Key Messages

1. **Performance**: 40B evals/sec, 800× speedup, consumer hardware
2. **Accessibility**: Python API, open source, clean documentation
3. **Capability**: 100+ parallel worlds, reversible physics, RL integration
4. **Credibility**: Verifiable benchmarks, research-grade architecture

---

*TileUniverse: Built for researchers who need simulation throughput, not simulation overhead.*
