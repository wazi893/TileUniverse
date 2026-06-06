#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# examples/grover_demo.py
# =============================================================================
#
# SPRINT 2.2: Demonstration of Grover's quantum search algorithm
# running on the TileUniverse quantum-classical hybrid simulator.
#
# =============================================================================

"""
Grover's Search Algorithm Demo

This script demonstrates:
1. Building a Grover circuit using the Python API
2. Running the circuit on a QDemo tile
3. Measuring success rate vs theoretical prediction
4. Comparing quantum vs classical search

Run with: python grover_demo.py
"""

import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from tileuniverse import QuantumSimulation, QGate, grover_circuit, optimal_iterations
from tileuniverse.algorithms import (
    run_grover,
    grover_success_rate,
    bell_state,
    ghz_state,
    CircuitBuilder,
)


def demo_basic_grover():
    """Basic Grover's algorithm demonstration."""
    print("=" * 60)
    print("DEMO 1: Basic Grover's Search")
    print("=" * 60)
    
    n_qubits = 3
    marked_state = 0b101  # |101⟩ = 5
    
    print(f"\nSearching for |{marked_state:0{n_qubits}b}⟩ in {2**n_qubits} items")
    print(f"Optimal iterations: {optimal_iterations(n_qubits)}")
    
    # Run with different seeds to show quantum randomness
    results = []
    for seed in range(10):
        result = run_grover(n_qubits, marked_state, seed=seed * 1234)
        results.append(result)
        status = "✓" if result == marked_state else "✗"
        print(f"  Seed {seed}: measured |{result:0{n_qubits}b}⟩ {status}")
    
    success_count = sum(1 for r in results if r == marked_state)
    print(f"\nSuccess rate: {success_count}/10 = {success_count * 10}%")
    print(f"(Theoretical: ~94.5%)")


def demo_manual_circuit():
    """Show how to build circuits manually."""
    print("\n" + "=" * 60)
    print("DEMO 2: Manual Circuit Construction")
    print("=" * 60)
    
    # Create simulation
    sim = QuantumSimulation()
    print(f"\nSimulation: {sim}")
    
    # Build a Bell state manually
    program = [
        QGate.h(0),           # Superposition
        QGate.cnot(0, 1),     # Entanglement
        QGate.measure(0),     # Measure qubit 0
        QGate.measure(1),     # Measure qubit 1
    ]
    
    print("\nBell state circuit:")
    for gate in program:
        print(f"  {gate}")
    
    # Register and run
    sim.register_qdemo(50, 50, n_qubits=2, program=program, seed=42)
    print(f"\nRegistered quantum tile at (50, 50)")
    print(f"Quantum tiles: {sim.quantum_tile_count}")
    
    # Run the circuit (4 ticks for 4 gates)
    sim.tick_n(4)
    
    # Get result
    result = sim.get_logic(50, 50)
    bit0 = (result >> 0) & 1
    bit1 = (result >> 1) & 1
    
    print(f"\nMeasurement result: |{bit1}{bit0}⟩")
    print(f"Qubits are {'correlated' if bit0 == bit1 else 'UNCORRELATED (bug!)'}")


def demo_circuit_builder():
    """Demonstrate the fluent CircuitBuilder API."""
    print("\n" + "=" * 60)
    print("DEMO 3: CircuitBuilder Fluent API")
    print("=" * 60)
    
    # Build a 3-qubit GHZ state using fluent API
    circuit = (CircuitBuilder(3)
        .h(0)                  # Create superposition on q0
        .cnot(0, 1)            # Entangle q0-q1
        .cnot(0, 2)            # Entangle q0-q2
        .barrier()             # Visual separator
        .measure_all()         # Measure all qubits
        .build())
    
    print(f"\nGHZ state circuit ({len(circuit)} gates):")
    for gate in circuit:
        print(f"  {gate}")
    
    # Run it
    sim = QuantumSimulation()
    sim.register_qdemo(25, 25, 3, circuit, seed=12345)
    sim.tick_n(len(circuit))
    
    result = sim.get_logic(25, 25)
    print(f"\nGHZ measurement: |{result:03b}⟩")
    print(f"Expected: |000⟩ or |111⟩ (all correlated)")


def demo_success_rate():
    """Measure and plot success rate."""
    print("\n" + "=" * 60)
    print("DEMO 4: Success Rate Analysis")
    print("=" * 60)
    
    print("\nMeasuring Grover success rate for different qubit counts...")
    print(f"{'Qubits':<8} {'Items':<8} {'Iterations':<12} {'Success Rate':<15} {'Expected'}")
    print("-" * 60)
    
    for n in range(2, 6):
        items = 2 ** n
        iters = optimal_iterations(n)
        
        # Run trials
        rate = grover_success_rate(n, items // 2, iterations=iters, trials=200)
        
        # Theoretical success probability
        # sin^2((2k+1) * arcsin(1/sqrt(N))) where k = iterations
        import math
        theta = math.asin(1 / math.sqrt(items))
        expected = math.sin((2 * iters + 1) * theta) ** 2
        
        print(f"{n:<8} {items:<8} {iters:<12} {rate*100:>6.1f}%{'':>8} {expected*100:.1f}%")


def demo_quantum_speedup():
    """Compare quantum vs classical search."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum Speedup Comparison")
    print("=" * 60)
    
    import random
    
    print("\nComparing quantum (Grover) vs classical (random) search")
    print(f"{'Qubits':<8} {'Items':<8} {'Quantum':<12} {'Classical':<12} {'Speedup'}")
    print("-" * 55)
    
    for n in range(2, 7):
        items = 2 ** n
        marked = random.randint(0, items - 1)
        
        # Quantum: Grover iterations needed
        quantum_queries = optimal_iterations(n)
        
        # Classical: expected N/2 queries for random search
        classical_queries = items / 2
        
        speedup = classical_queries / quantum_queries
        
        print(f"{n:<8} {items:<8} {quantum_queries:<12} {classical_queries:<12.0f} {speedup:.1f}x")
    
    print("\nNote: Speedup is √N, demonstrating quadratic quantum advantage")


def demo_multiple_tiles():
    """Run multiple quantum tiles simultaneously."""
    print("\n" + "=" * 60)
    print("DEMO 6: Multiple Quantum Tiles")
    print("=" * 60)
    
    sim = QuantumSimulation()
    
    # Register 4 tiles searching for different states
    targets = [0b000, 0b011, 0b101, 0b111]
    positions = [(10, 10), (20, 10), (10, 20), (20, 20)]
    
    print("\nRegistering 4 quantum tiles...")
    for target, (x, y) in zip(targets, positions):
        program = grover_circuit(3, target)
        sim.register_qdemo(x, y, 3, program, seed=target * 1234)
        print(f"  Tile at ({x}, {y}): searching for |{target:03b}⟩")
    
    print(f"\nTotal quantum tiles: {sim.quantum_tile_count}")
    
    # Run all circuits (max 50 ticks should be enough)
    sim.tick_n(50)
    
    # Read results
    print("\nResults:")
    for target, (x, y) in zip(targets, positions):
        result = sim.get_logic(x, y)
        status = "✓" if result == target else "✗"
        print(f"  ({x}, {y}): target |{target:03b}⟩ → found |{result:03b}⟩ {status}")


def main():
    """Run all demos."""
    print("\n🔬 TileUniverse Quantum Demo")
    print("   Quantum-Classical Hybrid Computation\n")
    
    demo_basic_grover()
    demo_manual_circuit()
    demo_circuit_builder()
    demo_success_rate()
    demo_quantum_speedup()
    demo_multiple_tiles()
    
    print("\n" + "=" * 60)
    print("✅ All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
