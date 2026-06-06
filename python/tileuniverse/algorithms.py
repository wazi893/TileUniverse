# =============================================================================
# tileuniverse/algorithms.py
# =============================================================================
#
# SPRINT 2.2: High-level quantum algorithm helpers.
#
# This module provides Pythonic wrappers around the Rust algorithm
# implementations, plus additional utilities for experimentation.
#
# =============================================================================

"""
Quantum algorithms for the TileUniverse hybrid simulator.

This module provides high-level functions for building quantum circuits
and running common quantum algorithms.

Example:
    >>> from tileuniverse import QuantumSimulation
    >>> from tileuniverse.algorithms import run_grover, bell_state
    >>>
    >>> # Run Grover's search
    >>> result = run_grover(n_qubits=3, marked_state=5)
    >>> print(f"Found: {result:03b}")
    >>>
    >>> # Create Bell state
    >>> sim = QuantumSimulation()
    >>> program = bell_state()
    >>> sim.register_qdemo(50, 50, 2, program, seed=42)
"""

from typing import List, Optional, Tuple
import math

# Import from the Rust extension
from tileuniverse._core import (
    QGate,
    QuantumSimulation,
    grover_circuit as _grover_circuit,
    optimal_iterations as _optimal_iterations,
    # SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani
    DJOracle,
    deutsch_jozsa_circuit as _dj_circuit,
    bernstein_vazirani_circuit as _bv_circuit,
)


__all__ = [
    # Re-exports from Rust
    'grover_circuit',
    'optimal_iterations',
    # Pure Python helpers
    'run_grover',
    'grover_success_rate',
    'bell_state',
    'ghz_state',
    'uniform_superposition',
    'measure_all',
    # Circuit builders
    'CircuitBuilder',
    # SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani
    'DJOracle',
    'deutsch_jozsa_circuit',
    'bernstein_vazirani_circuit',
    'run_deutsch_jozsa',
    'run_bernstein_vazirani',
    'compare_quantum_speedup',
]


# =============================================================================
# Re-exports with better docstrings
# =============================================================================

def grover_circuit(n_qubits: int, marked_state: int, iterations: Optional[int] = None) -> List[QGate]:
    """
    Build a Grover search circuit.

    Grover's algorithm provides quadratic speedup for unstructured search:
    finding a marked item in N items requires O(√N) queries instead of O(N).

    Args:
        n_qubits: Number of qubits (search space = 2^n_qubits)
        marked_state: The state to find (integer, e.g., 5 for |101⟩)
        iterations: Number of Grover iterations. If None, uses optimal value.

    Returns:
        List of QGate objects implementing Grover's algorithm.

    Example:
        >>> circuit = grover_circuit(3, 5)  # Find |101⟩ in 8 items
        >>> len(circuit)
        42

    Notes:
        - Optimal iterations = floor(π/4 × √N) where N = 2^n_qubits
        - Success probability ≈ sin²((2k+1)θ) where θ = arcsin(1/√N)
        - For 3 qubits: ~94.5% success with 2 iterations
    """
    return _grover_circuit(n_qubits, marked_state, iterations)


def optimal_iterations(n_qubits: int) -> int:
    """
    Calculate optimal number of Grover iterations.

    Args:
        n_qubits: Number of qubits in the search space.

    Returns:
        Optimal iteration count for maximum success probability.

    Example:
        >>> optimal_iterations(3)  # 8 items → √8 × π/4 ≈ 2.2 → 2
        2
        >>> optimal_iterations(10)  # 1024 items → √1024 × π/4 ≈ 25
        25
    """
    return _optimal_iterations(n_qubits)


# =============================================================================
# High-level runners
# =============================================================================

def run_grover(
    n_qubits: int,
    marked_state: int,
    iterations: Optional[int] = None,
    seed: int = 0,
    position: Tuple[int, int] = (50, 50),
) -> int:
    """
    Run Grover's algorithm and return the measured result.

    This is a convenience function that creates a simulation, registers
    a quantum tile with the Grover circuit, runs it, and returns the result.

    Args:
        n_qubits: Number of qubits (search space = 2^n_qubits)
        marked_state: The state to find
        iterations: Grover iterations (default: optimal)
        seed: Random seed for measurement
        position: Tile position (default: (50, 50))

    Returns:
        Measured result as integer (should match marked_state with high probability)

    Example:
        >>> result = run_grover(3, 5, seed=42)
        >>> result == 5  # True with ~94.5% probability
        True
    """
    sim = QuantumSimulation()
    program = grover_circuit(n_qubits, marked_state, iterations)
    x, y = position
    sim.register_qdemo(x, y, n_qubits, program, seed)
    sim.tick_n(len(program))
    return sim.get_logic(x, y)


def grover_success_rate(
    n_qubits: int,
    marked_state: int,
    iterations: Optional[int] = None,
    trials: int = 100,
) -> float:
    """
    Measure Grover's algorithm success rate over multiple trials.

    Args:
        n_qubits: Number of qubits
        marked_state: Target state to find
        iterations: Grover iterations (default: optimal)
        trials: Number of trials to run

    Returns:
        Success rate as float (0.0 to 1.0)

    Example:
        >>> rate = grover_success_rate(3, 5, trials=1000)
        >>> rate > 0.9  # Should be close to 94.5%
        True
    """
    successes = 0
    for trial in range(trials):
        seed = trial * 0x123456789
        result = run_grover(n_qubits, marked_state, iterations, seed)
        if result == marked_state:
            successes += 1
    return successes / trials


# =============================================================================
# Common circuit patterns
# =============================================================================

def bell_state(qubit0: int = 0, qubit1: int = 1, measure: bool = True) -> List[QGate]:
    """
    Create a Bell state (maximally entangled 2-qubit state).

    The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 exhibits perfect correlation:
    measuring one qubit instantly determines the other.

    Args:
        qubit0: First qubit index
        qubit1: Second qubit index
        measure: Whether to include measurement gates

    Returns:
        Circuit creating Bell state

    Example:
        >>> program = bell_state()
        >>> sim = QuantumSimulation()
        >>> sim.register_qdemo(10, 10, 2, program, seed=42)
        >>> sim.tick_n(len(program))
        >>> result = sim.get_logic(10, 10)
        >>> result in [0b00, 0b11]  # Always correlated
        True
    """
    gates = [
        QGate.h(qubit0),
        QGate.cnot(qubit0, qubit1),
    ]
    if measure:
        gates.extend([
            QGate.measure(qubit0),
            QGate.measure(qubit1),
        ])
    return gates


def ghz_state(n_qubits: int, measure: bool = True) -> List[QGate]:
    """
    Create a GHZ state (n-qubit generalization of Bell state).

    The GHZ state |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2 is maximally entangled
    across all qubits.

    Args:
        n_qubits: Number of qubits (must be >= 2)
        measure: Whether to include measurement gates

    Returns:
        Circuit creating GHZ state

    Example:
        >>> program = ghz_state(3)
        >>> # Creates (|000⟩ + |111⟩)/√2
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    gates = [QGate.h(0)]
    for i in range(1, n_qubits):
        gates.append(QGate.cnot(0, i))

    if measure:
        for i in range(n_qubits):
            gates.append(QGate.measure(i))

    return gates


def uniform_superposition(n_qubits: int, measure: bool = False) -> List[QGate]:
    """
    Create uniform superposition over all basis states.

    Applies Hadamard to each qubit: H⊗n|0...0⟩ = (1/√N) Σ|x⟩

    Args:
        n_qubits: Number of qubits
        measure: Whether to include measurement gates

    Returns:
        Circuit creating uniform superposition

    Example:
        >>> program = uniform_superposition(3, measure=True)
        >>> # Each basis state |000⟩ through |111⟩ has equal probability
    """
    gates = [QGate.h(i) for i in range(n_qubits)]
    if measure:
        gates.extend([QGate.measure(i) for i in range(n_qubits)])
    return gates


def measure_all(n_qubits: int) -> List[QGate]:
    """
    Create measurement gates for all qubits.

    Args:
        n_qubits: Number of qubits to measure

    Returns:
        List of Measure gates
    """
    return [QGate.measure(i) for i in range(n_qubits)]


# =============================================================================
# Circuit builder (fluent API)
# =============================================================================

class CircuitBuilder:
    """
    Fluent interface for building quantum circuits.

    Example:
        >>> circuit = (CircuitBuilder(3)
        ...     .h(0).h(1).h(2)
        ...     .cnot(0, 1)
        ...     .cz(1, 2)
        ...     .measure_all()
        ...     .build())
    """

    def __init__(self, n_qubits: int):
        """
        Create a circuit builder.

        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self._gates: List[QGate] = []

    def h(self, qubit: int) -> 'CircuitBuilder':
        """Add Hadamard gate."""
        self._validate_qubit(qubit)
        self._gates.append(QGate.h(qubit))
        return self

    def x(self, qubit: int) -> 'CircuitBuilder':
        """Add Pauli-X gate."""
        self._validate_qubit(qubit)
        self._gates.append(QGate.x(qubit))
        return self

    def z(self, qubit: int) -> 'CircuitBuilder':
        """Add Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._gates.append(QGate.z(qubit))
        return self

    def phase(self, qubit: int, theta: float) -> 'CircuitBuilder':
        """Add phase rotation gate."""
        self._validate_qubit(qubit)
        self._gates.append(QGate.phase(qubit, theta))
        return self

    def cnot(self, control: int, target: int) -> 'CircuitBuilder':
        """Add CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self._gates.append(QGate.cnot(control, target))
        return self

    def cz(self, qubit0: int, qubit1: int) -> 'CircuitBuilder':
        """Add Controlled-Z gate."""
        self._validate_qubit(qubit0)
        self._validate_qubit(qubit1)
        self._gates.append(QGate.cz(qubit0, qubit1))
        return self

    def toffoli(self, ctrl0: int, ctrl1: int, target: int) -> 'CircuitBuilder':
        """Add Toffoli (CCX) gate."""
        self._validate_qubit(ctrl0)
        self._validate_qubit(ctrl1)
        self._validate_qubit(target)
        self._gates.append(QGate.toffoli(ctrl0, ctrl1, target))
        return self

    def ccx(self, ctrl0: int, ctrl1: int, target: int) -> 'CircuitBuilder':
        """Alias for toffoli()."""
        return self.toffoli(ctrl0, ctrl1, target)

    def ccz(self, qubit0: int, qubit1: int, qubit2: int) -> 'CircuitBuilder':
        """Add CCZ gate."""
        self._validate_qubit(qubit0)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self._gates.append(QGate.ccz(qubit0, qubit1, qubit2))
        return self

    def measure(self, qubit: int) -> 'CircuitBuilder':
        """Add measurement gate."""
        self._validate_qubit(qubit)
        self._gates.append(QGate.measure(qubit))
        return self

    def measure_all(self) -> 'CircuitBuilder':
        """Add measurement gates for all qubits."""
        for i in range(self.n_qubits):
            self._gates.append(QGate.measure(i))
        return self

    def hadamard_all(self) -> 'CircuitBuilder':
        """Add Hadamard gates to all qubits."""
        for i in range(self.n_qubits):
            self._gates.append(QGate.h(i))
        return self

    def barrier(self) -> 'CircuitBuilder':
        """
        Visual barrier (no-op, for circuit visualization).

        Note: This doesn't add any gate, just for code organization.
        """
        return self

    def build(self) -> List[QGate]:
        """
        Build and return the circuit.

        Returns:
            List of QGate objects
        """
        return self._gates.copy()

    def _validate_qubit(self, qubit: int) -> None:
        """Check qubit index is valid."""
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.n_qubits - 1}]")

    def __len__(self) -> int:
        """Number of gates in the circuit."""
        return len(self._gates)

    def __repr__(self) -> str:
        return f"CircuitBuilder(n_qubits={self.n_qubits}, gates={len(self._gates)})"


# =============================================================================
# Deutsch-Jozsa (placeholder for Sprint 2.3)
# =============================================================================

# =============================================================================
# SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani Algorithms
# =============================================================================

def deutsch_jozsa_circuit(n_qubits: int, oracle: DJOracle) -> List[QGate]:
    """
    Build a Deutsch-Jozsa circuit.

    The Deutsch-Jozsa algorithm determines whether a function is constant
    (always 0 or always 1) or balanced (half 0, half 1) with a single query.

    Speedup: O(2^(n-1) + 1) classical → O(1) quantum (EXPONENTIAL!)

    Args:
        n_qubits: Number of input qubits (total circuit has n+1 including ancilla)
        oracle: Oracle type created with DJOracle.constant_zero(),
                DJOracle.constant_one(), or DJOracle.balanced(pattern)

    Returns:
        List of QGate objects. After measurement:
        - |0...0⟩ (result = 0) indicates CONSTANT oracle
        - Any other result indicates BALANCED oracle

    Example:
        >>> circuit = deutsch_jozsa_circuit(3, DJOracle.balanced(5))
        >>> sim = QuantumSimulation()
        >>> sim.register_qdemo(50, 50, 4, circuit, seed=42)  # n+1 qubits!
        >>> sim.tick_n(len(circuit))
        >>> result = sim.get_logic(50, 50) & 0b111  # Mask to n bits
        >>> is_constant = (result == 0)
    """
    return _dj_circuit(n_qubits, oracle)


def run_deutsch_jozsa(
    n_qubits: int,
    oracle: DJOracle,
    seed: int = 0,
    position: Tuple[int, int] = (50, 50),
) -> bool:
    """
    Run Deutsch-Jozsa and return whether the oracle is constant.

    Args:
        n_qubits: Number of input qubits
        oracle: Oracle type
        seed: Random seed (doesn't affect deterministic result)
        position: Tile position

    Returns:
        True if oracle is CONSTANT, False if BALANCED

    Example:
        >>> is_const = run_deutsch_jozsa(3, DJOracle.constant_zero())
        >>> print(is_const)  # True
        >>> is_const = run_deutsch_jozsa(3, DJOracle.balanced(5))
        >>> print(is_const)  # False
    """
    sim = QuantumSimulation()
    circuit = deutsch_jozsa_circuit(n_qubits, oracle)
    x, y = position

    # Need n+1 qubits (includes ancilla)
    sim.register_qdemo(x, y, n_qubits + 1, circuit, seed)
    sim.tick_n(len(circuit))

    result = sim.get_logic(x, y)
    # Only check the input qubits (not ancilla)
    input_bits = result & ((1 << n_qubits) - 1)

    return input_bits == 0


def bernstein_vazirani_circuit(n_qubits: int, hidden_string: int) -> List[QGate]:
    """
    Build a Bernstein-Vazirani circuit.

    The Bernstein-Vazirani algorithm finds a hidden string s where
    the oracle computes f(x) = s·x mod 2 (bitwise dot product).

    Speedup: O(n) classical → O(1) quantum (LINEAR to CONSTANT!)

    Args:
        n_qubits: Number of bits in the hidden string
        hidden_string: The secret string encoded in the oracle

    Returns:
        List of QGate objects. After measurement, result equals hidden_string.

    Example:
        >>> circuit = bernstein_vazirani_circuit(4, 0b1010)
        >>> sim = QuantumSimulation()
        >>> sim.register_qdemo(50, 50, 5, circuit, seed=42)  # n+1 qubits!
        >>> sim.tick_n(len(circuit))
        >>> result = sim.get_logic(50, 50) & 0b1111  # Mask to n bits
        >>> print(f"Found: {result:04b}")  # Should be 1010
    """
    return _bv_circuit(n_qubits, hidden_string)


def run_bernstein_vazirani(
    n_qubits: int,
    hidden_string: int,
    seed: int = 0,
    position: Tuple[int, int] = (50, 50),
) -> int:
    """
    Run Bernstein-Vazirani and return the discovered hidden string.

    This is a deterministic algorithm — different seeds give the same result.

    Args:
        n_qubits: Number of bits in the hidden string
        hidden_string: The secret to find
        seed: Random seed (doesn't affect deterministic result)
        position: Tile position

    Returns:
        The hidden string, recovered in a single query

    Example:
        >>> secret = 0b10101
        >>> found = run_bernstein_vazirani(5, secret)
        >>> print(found == secret)  # Always True
    """
    sim = QuantumSimulation()
    circuit = bernstein_vazirani_circuit(n_qubits, hidden_string)
    x, y = position

    # Need n+1 qubits (includes ancilla)
    sim.register_qdemo(x, y, n_qubits + 1, circuit, seed)
    sim.tick_n(len(circuit))

    result = sim.get_logic(x, y)
    # Only return the input qubits (not ancilla)
    return result & ((1 << n_qubits) - 1)


def compare_quantum_speedup(n_qubits: int = 5) -> dict:
    """
    Compare classical vs quantum query complexity for all implemented algorithms.

    Args:
        n_qubits: Number of qubits to use for comparison

    Returns:
        Dictionary with speedup analysis for each algorithm

    Example:
        >>> stats = compare_quantum_speedup(6)
        >>> for algo, data in stats.items():
        ...     print(f"{algo}: {data['speedup']}x speedup")
    """
    n = n_qubits
    N = 2 ** n

    return {
        'grover': {
            'classical_queries': N // 2,  # Expected for random search
            'quantum_queries': int((math.pi / 4) * (N ** 0.5)),
            'speedup_type': 'quadratic',
            'speedup': (N // 2) / max(1, int((math.pi / 4) * (N ** 0.5))),
        },
        'deutsch_jozsa': {
            'classical_queries': (N // 2) + 1,  # Worst case
            'quantum_queries': 1,
            'speedup_type': 'exponential',
            'speedup': (N // 2) + 1,
        },
        'bernstein_vazirani': {
            'classical_queries': n,
            'quantum_queries': 1,
            'speedup_type': 'linear_to_constant',
            'speedup': n,
        },
    }
