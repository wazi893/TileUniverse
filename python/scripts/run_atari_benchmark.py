#!/usr/bin/env python3
"""
Quick-start script for running Quantum-SNN Atari benchmarks.

This script provides preset configurations for different benchmark scenarios.

Usage:
    # Quick test (1 seed, 100 episodes)
    python run_atari_benchmark.py --preset quick --game Breakout

    # Standard benchmark (3 seeds, 1000 episodes)
    python run_atari_benchmark.py --preset standard --game Pong

    # Full benchmark (5 seeds, 5000 episodes, all games)
    python run_atari_benchmark.py --preset full

    # Custom configuration
    python run_atari_benchmark.py --game SpaceInvaders --episodes 2000 --seeds 42 123 456

    # With curiosity-driven exploration (good for sparse rewards)
    python run_atari_benchmark.py --game MontezumaRevenge --preset standard --curiosity

Requirements:
    pip install gymnasium[atari] ale-py matplotlib
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tileuniverse.rl.atari_benchmark import BenchmarkConfig, run_benchmark, plot_learning_curves


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    "quick": {
        "description": "Quick test run (5-10 minutes)",
        "n_episodes": 100,
        "seeds": [42],
        "modes": ["triggered", "none"],
        "run_epsilon_greedy": True,
        "epsilon_values": [0.1],
    },
    "standard": {
        "description": "Standard benchmark (1-2 hours per game)",
        "n_episodes": 1000,
        "seeds": [42, 123, 456],
        "modes": ["triggered", "anti_grover", "soft_mix", "none"],
        "run_epsilon_greedy": True,
        "epsilon_values": [0.1, 0.05],
    },
    "full": {
        "description": "Full benchmark for publication (8-12 hours per game)",
        "n_episodes": 5000,
        "seeds": [42, 123, 456, 789, 1337],
        "modes": ["triggered", "anti_grover", "grover", "soft_mix", "adaptive", "none"],
        "run_epsilon_greedy": True,
        "epsilon_values": [0.1, 0.05, 0.01],
    },
    "triggered_only": {
        "description": "Focus on triggered mode vs baselines",
        "n_episodes": 2000,
        "seeds": [42, 123, 456, 789, 1337],
        "modes": ["triggered"],
        "run_epsilon_greedy": True,
        "epsilon_values": [0.1, 0.05, 0.01],
    },
    "sparse_reward": {
        "description": "Optimized for sparse reward games (Montezuma, Venture)",
        "n_episodes": 2000,
        "seeds": [42, 123, 456],
        "modes": ["triggered", "anti_grover"],
        "run_epsilon_greedy": True,
        "epsilon_values": [0.1],
        "enable_curiosity": True,
        "curiosity_type": "prediction",
        "curiosity_scale": 1.0,
    },
}

# Popular Atari games for benchmarking
GAME_SUITES = {
    "easy": ["Breakout", "Pong", "SpaceInvaders"],
    "medium": ["Qbert", "Seaquest", "BeamRider", "Enduro"],
    "hard": ["MsPacman", "Asteroids", "Frostbite"],
    "sparse": ["MontezumaRevenge", "Venture", "PrivateEye"],
    "all": [
        "Breakout", "Pong", "SpaceInvaders", "Qbert", "Seaquest",
        "BeamRider", "Enduro", "MsPacman", "Asteroids", "Frostbite",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Quantum-SNN Atari Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  quick           - Fast test (100 episodes, 1 seed)
  standard        - Standard benchmark (1000 episodes, 3 seeds)
  full            - Publication-quality (5000 episodes, 5 seeds)
  triggered_only  - Focus on triggered mode comparison
  sparse_reward   - Optimized for hard exploration games

Game Suites:
  easy            - Breakout, Pong, SpaceInvaders
  medium          - Qbert, Seaquest, BeamRider, Enduro
  hard            - MsPacman, Asteroids, Frostbite
  sparse          - MontezumaRevenge, Venture, PrivateEye
  all             - All of the above

Examples:
  python run_atari_benchmark.py --preset quick --game Breakout
  python run_atari_benchmark.py --preset standard --suite easy
  python run_atari_benchmark.py --game Pong --episodes 500 --seeds 42 123
        """
    )

    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="standard",
                        help="Preset configuration (default: standard)")
    parser.add_argument("--game", type=str, default=None,
                        help="Single game to benchmark")
    parser.add_argument("--suite", choices=list(GAME_SUITES.keys()), default=None,
                        help="Game suite to benchmark")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override random seeds")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="Override interference modes")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                        help="Hidden layer sizes (default: [64])")
    parser.add_argument("--ticks", type=int, default=20,
                        help="SNN ticks per decision (default: 20)")
    parser.add_argument("--curiosity", action="store_true",
                        help="Enable curiosity-driven exploration")
    parser.add_argument("--td-credit", action="store_true",
                        help="Enable TD-style credit assignment")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity (0=quiet, 1=normal, 2=debug)")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")
    parser.add_argument("--list-games", action="store_true",
                        help="List available games and exit")

    args = parser.parse_args()

    # Handle list options
    if args.list_presets:
        print("\nAvailable Presets:")
        print("-" * 60)
        for name, preset in PRESETS.items():
            print(f"  {name:<20} - {preset['description']}")
            print(f"                       Episodes: {preset['n_episodes']}, "
                  f"Seeds: {len(preset['seeds'])}")
        return

    if args.list_games:
        print("\nAvailable Game Suites:")
        print("-" * 60)
        for suite_name, games in GAME_SUITES.items():
            print(f"  {suite_name:<10} - {', '.join(games)}")
        return

    # Get preset configuration
    preset = PRESETS[args.preset]

    # Determine games to run
    if args.game:
        games = [args.game]
    elif args.suite:
        games = GAME_SUITES[args.suite]
    else:
        games = ["Breakout"]  # Default

    print(f"\n{'='*60}")
    print(f"Quantum-SNN Atari Benchmark")
    print(f"{'='*60}")
    print(f"Preset: {args.preset} - {preset['description']}")
    print(f"Games: {', '.join(games)}")
    print(f"{'='*60}\n")

    # Run benchmark for each game
    for game in games:
        print(f"\n{'#'*60}")
        print(f"# Game: {game}")
        print(f"{'#'*60}")

        # Build configuration
        config = BenchmarkConfig(
            game=game,
            n_episodes=args.episodes or preset["n_episodes"],
            seeds=args.seeds or preset["seeds"],
            modes=args.modes or preset["modes"],
            hidden_layers=args.hidden,
            ticks_per_decision=args.ticks,
            run_epsilon_greedy=preset.get("run_epsilon_greedy", True),
            epsilon_values=preset.get("epsilon_values", [0.1]),
            enable_curiosity=args.curiosity or preset.get("enable_curiosity", False),
            curiosity_type=preset.get("curiosity_type", "prediction"),
            curiosity_scale=preset.get("curiosity_scale", 0.5),
            enable_td_credit=args.td_credit or preset.get("enable_td_credit", False),
            output_dir=args.output,
            verbose=args.verbose,
        )

        # Run benchmark
        try:
            results = run_benchmark(config)

            # Generate plots
            output_path = Path(config.output_dir) / game
            plot_learning_curves(results, output_path)

        except Exception as e:
            print(f"\nError benchmarking {game}: {e}")
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All benchmarks complete!")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
