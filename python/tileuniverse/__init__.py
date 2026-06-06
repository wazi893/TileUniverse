"""
TileUniverse: GPU-accelerated parallel universe simulator
with quantum-classical hybrid computation.

Quick Start:
    >>> import tileuniverse as tu
    >>>
    >>> # Classical parallel worlds
    >>> engine = tu.Engine(worlds=5, size=(512, 512), ruleset="gol")
    >>> engine.evolve(100)
    >>> world = engine.get_world(0)
    >>>
    >>> # Quantum-classical hybrid
    >>> from tileuniverse import QuantumSimulation, QGate
    >>> sim = QuantumSimulation()
    >>> sim.register_qdemo(50, 50, 2, [
    ...     QGate.h(0),
    ...     QGate.cnot(0, 1),
    ...     QGate.measure(0),
    ...     QGate.measure(1),
    ... ], seed=42)
    >>> sim.tick_n(4)
    >>> result = sim.get_logic(50, 50)
    >>> print(f"Bell state measurement: {result:02b}")

Modules:
    algorithms: Quantum algorithms (Grover, Bell states, etc.)
    rl: Reinforcement learning integration
"""

__version__ = "0.5.0"  # Added Stabilizer and Hybrid quantum neural networks

# Core extension imports
from tileuniverse._core import (
    # Classical parallel worlds
    Engine,
    BenchmarkResult,
    # Utility functions
    is_cuda_available,
    cuda_device_name,
    benchmark,
    load_config,
    # SPRINT 2.2: Quantum API
    QGate,
    QuantumSimulation,
    grover_circuit,
    optimal_iterations,
    # SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani
    DJOracle,
    deutsch_jozsa_circuit,
    bernstein_vazirani_circuit,
    # SNN bindings: Quantum-SNN hybrid
    InterferenceMode,
    QuantumSNNConfig,
    QuantumSNN,
    # Sprint 59.0: Stabilizer-based quantum neural networks
    FeedbackMode,
    BridgeGate,
    StabilizerSNN,
    HybridQuantumSNN,
    # Fast W state (Vec-based, billions of qubits)
    FastWState,
    WStateVerification,
    create_fast_w_state,
    # Fast GHZ state (O(1) sparse - only 2 amplitudes)
    FastGHZState,
    GHZStateVerification,
    create_fast_ghz_state,
    # E-prop SNN (CPU-based, can learn from scratch)
    EpropSNN,
    SurrogateGradient,
    # Sprint 212: V2 Tile CPU
    V2Cpu,
    V2Trace,
    # Sprint 212: Synth entry point
    SynthResult,
    synthesize,
    # Sprint 215: Multi-output synth + AIGER
    synthesize_multi,
    synthesize_from_aiger,
    synthesize_from_aiger_bytes,
    export_to_aiger,
    export_to_aiger_ascii,
    # Sprint 214: Sequential circuits
    SequentialCircuit,
)

# GPU-only imports (require CUDA build)
try:
    from tileuniverse._core import FusedSNN
except ImportError:
    FusedSNN = None

# Submodules (lazy import to avoid circular deps)
from tileuniverse import algorithms

# Rulesets available
RULESETS = ["gol", "rule110", "wire", "logic"]


def render_ascii(engine, world=0, width=80, height=40, alive_char="#", dead_char="."):
    """
    Render a world state as ASCII art.

    Args:
        engine: Engine instance
        world: World index to render (default: 0)
        width: Output width in characters (default: 80)
        height: Output height in characters (default: 40)
        alive_char: Character for alive cells (default: full block)
        dead_char: Character for dead cells (default: space)

    Returns:
        str: ASCII art representation of the world

    Example:
        >>> engine = tu.Engine(worlds=1, size=(64, 64), ruleset="gol")
        >>> engine.randomize(0, 0.3)
        >>> print(tu.render_ascii(engine))
    """
    world_data = engine.get_world(world)  # Returns 2D numpy array (height, width)
    w, h = engine.size  # (width, height)

    # Calculate scaling
    scale_x = max(1, w // width)
    scale_y = max(1, h // height)
    out_width = min(w, width)
    out_height = min(h, height)

    lines = []
    for y in range(out_height):
        row = ""
        for x in range(out_width):
            sx = x * scale_x
            sy = y * scale_y
            # world_data is (height, width) numpy array
            cell = world_data[sy, sx]
            row += alive_char if cell != 0 else dead_char
        lines.append(row)

    return "\n".join(lines)

def _cli_main():
    """CLI entry point for `tileuniverse` command."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="TileUniverse: GPU-accelerated parallel universe simulator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run performance benchmark")
    bench_parser.add_argument("--worlds", type=int, default=5, help="Number of parallel worlds (default: 5)")
    bench_parser.add_argument("--steps", type=int, default=1000, help="Simulation steps (default: 1000)")
    bench_parser.add_argument("--size", type=int, default=512, help="World size NxN (default: 512)")
    bench_parser.add_argument("--depth", type=int, default=50, help="Steps per kernel launch (default: 50)")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")

    # Visualize command
    viz_parser = subparsers.add_parser("viz", help="Visualize simulation in terminal")
    viz_parser.add_argument("--worlds", type=int, default=1, help="Number of worlds (default: 1)")
    viz_parser.add_argument("--size", type=int, default=64, help="World size NxN (default: 64)")
    viz_parser.add_argument("--ruleset", type=str, default="gol", help="Ruleset: gol, rule110, wire, logic (default: gol)")
    viz_parser.add_argument("--density", type=float, default=0.3, help="Initial random density (default: 0.3)")
    viz_parser.add_argument("--steps", type=int, default=100, help="Steps to simulate (default: 100)")
    viz_parser.add_argument("--fps", type=float, default=10, help="Frames per second (default: 10)")
    viz_parser.add_argument("--world", type=int, default=0, help="Which world to visualize (default: 0)")

    args = parser.parse_args()

    if args.command == "bench":
        print("=" * 70)
        print("  TileUniverse Performance Benchmark")
        print("=" * 70)
        print()

        if not is_cuda_available():
            print("ERROR: CUDA not available. GPU required for benchmarks.")
            sys.exit(1)

        print(f"  GPU: {cuda_device_name()}")
        print(f"  Worlds: {args.worlds}")
        print(f"  Size: {args.size}x{args.size}")
        print(f"  Steps: {args.steps}")
        print(f"  Depth: {args.depth} steps/kernel")
        print()

        result = benchmark(
            worlds=args.worlds,
            width=args.size,
            height=args.size,
            steps=args.steps,
            depth=args.depth,
        )

        evals_b = result.total_evals / 1e9
        rate_b = result.evals_per_sec / 1e9

        print(f"  Total evaluations: {evals_b:.2f}B")
        print(f"  Elapsed time: {result.elapsed_secs:.3f}s")
        print()
        print("  " + "-" * 50)
        print(f"  PERFORMANCE: {rate_b:.1f}B logic evals/sec")
        print("  " + "-" * 50)
        print()

    elif args.command == "info":
        print(f"TileUniverse v{__version__}")
        print(f"CUDA available: {is_cuda_available()}")
        if is_cuda_available():
            print(f"GPU: {cuda_device_name()}")
        print(f"Available rulesets: {', '.join(RULESETS)}")

    elif args.command == "viz":
        import time

        # Characters for visualization (ASCII-safe for Windows)
        ALIVE = "#"
        DEAD = "."

        # Create engine
        engine = Engine(
            worlds=args.worlds,
            size=(args.size, args.size),
            ruleset=args.ruleset,
            reversible=False
        )

        # Initialize with random pattern
        for w in range(args.worlds):
            engine.randomize(w, args.density)

        frame_delay = 1.0 / args.fps
        display_width = min(args.size, 80)  # Terminal width limit
        display_height = min(args.size, 40)  # Terminal height limit

        print(f"\nTileUniverse Visualization")
        print(f"Ruleset: {args.ruleset} | Size: {args.size}x{args.size} | World: {args.world}")
        print(f"Press Ctrl+C to stop\n")
        time.sleep(1)

        try:
            for step in range(args.steps):
                # Get world state
                world = engine.get_world(args.world)

                # Clear screen (ANSI escape)
                print("\033[2J\033[H", end="")

                # Header
                print(f"Step: {engine.step:5d} | Ruleset: {args.ruleset} | World: {args.world}/{args.worlds}")
                print("-" * display_width)

                # Render world (downsampled if needed)
                scale_x = args.size // display_width
                scale_y = args.size // display_height

                for y in range(display_height):
                    row = ""
                    for x in range(display_width):
                        # Sample from scaled position (world is 2D numpy array)
                        sx = x * scale_x
                        sy = y * scale_y
                        cell = world[sy, sx]
                        row += ALIVE if cell != 0 else DEAD
                    print(row)

                print("-" * display_width)

                # Count alive cells (world is 2D numpy array)
                alive = int((world != 0).sum())
                total = world.size
                pct = 100 * alive / total if total > 0 else 0
                print(f"Alive: {alive:6d} / {total} ({pct:.1f}%)")

                # Evolve
                engine.evolve(1)

                # Delay for animation
                time.sleep(frame_delay)

        except KeyboardInterrupt:
            print("\n\nVisualization stopped.")

    else:
        parser.print_help()


__all__ = [
    # Version
    "__version__",
    # Classical
    "Engine",
    "BenchmarkResult",
    # Utilities
    "is_cuda_available",
    "cuda_device_name",
    "benchmark",
    "load_config",
    "render_ascii",
    # Quantum (SPRINT 2.2)
    "QGate",
    "QuantumSimulation",
    "grover_circuit",
    "optimal_iterations",
    # Quantum (SPRINT 2.3)
    "DJOracle",
    "deutsch_jozsa_circuit",
    "bernstein_vazirani_circuit",
    # SNN bindings: Quantum-SNN hybrid
    "InterferenceMode",
    "QuantumSNNConfig",
    "QuantumSNN",
    # Sprint 59.0: Stabilizer-based quantum neural networks
    "FeedbackMode",
    "BridgeGate",
    "StabilizerSNN",
    "HybridQuantumSNN",
    # Fast W state (Vec-based, billions of qubits)
    "FastWState",
    "WStateVerification",
    "create_fast_w_state",
    # Fast GHZ state (O(1) sparse - only 2 amplitudes)
    "FastGHZState",
    "GHZStateVerification",
    "create_fast_ghz_state",
    # GPU Fused SNN (EPIC 97-style, may be None without CUDA)
    "FusedSNN",
    # E-prop SNN (CPU-based, can learn from scratch)
    "EpropSNN",
    "SurrogateGradient",
    # Sprint 212: V2 Tile CPU
    "V2Cpu",
    "V2Trace",
    # Sprint 212: Synth
    "SynthResult",
    "synthesize",
    # Sprint 215: Multi-output synth + AIGER
    "synthesize_multi",
    "synthesize_from_aiger",
    "synthesize_from_aiger_bytes",
    "export_to_aiger",
    "export_to_aiger_ascii",
    # Sprint 214: Sequential circuits
    "SequentialCircuit",
    # Submodules
    "algorithms",
    # Constants
    "RULESETS",
]
