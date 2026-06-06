# =============================================================================
# tests/test_quantum_api.py
# =============================================================================
#
# SPRINT 2.2: Python tests for quantum API.
#
# Run with: pytest tests/test_quantum_api.py -v
#
# =============================================================================

"""Tests for TileUniverse quantum Python API."""

import pytest
from tileuniverse import (
    QuantumSimulation,
    QGate,
    grover_circuit,
    optimal_iterations,
)
from tileuniverse.algorithms import (
    run_grover,
    grover_success_rate,
    bell_state,
    ghz_state,
    uniform_superposition,
    measure_all,
    CircuitBuilder,
)


class TestQGate:
    """Tests for QGate class."""

    def test_h_gate(self):
        """Test Hadamard gate creation."""
        gate = QGate.h(0)
        assert str(gate) == "H(0)"

    def test_x_gate(self):
        """Test Pauli-X gate creation."""
        gate = QGate.x(1)
        assert str(gate) == "X(1)"

    def test_z_gate(self):
        """Test Pauli-Z gate creation."""
        gate = QGate.z(2)
        assert str(gate) == "Z(2)"

    def test_phase_gate(self):
        """Test Phase gate creation."""
        gate = QGate.phase(0, 0.5)
        assert "Phase(0" in str(gate)

    def test_cnot_gate(self):
        """Test CNOT gate creation."""
        gate = QGate.cnot(0, 1)
        assert str(gate) == "CNOT(0, 1)"

    def test_cz_gate(self):
        """Test Controlled-Z gate creation."""
        gate = QGate.cz(0, 1)
        assert str(gate) == "CZ(0, 1)"

    def test_toffoli_gate(self):
        """Test Toffoli gate creation."""
        gate = QGate.toffoli(0, 1, 2)
        assert str(gate) == "Toffoli(0, 1, 2)"

    def test_ccx_alias(self):
        """Test CCX is alias for Toffoli."""
        gate = QGate.ccx(0, 1, 2)
        assert str(gate) == "Toffoli(0, 1, 2)"

    def test_ccz_gate(self):
        """Test CCZ gate creation."""
        gate = QGate.ccz(0, 1, 2)
        assert str(gate) == "CCZ(0, 1, 2)"

    def test_measure_gate(self):
        """Test Measure gate creation."""
        gate = QGate.measure(0)
        assert str(gate) == "Measure(0)"


class TestQuantumSimulation:
    """Tests for QuantumSimulation class."""

    def test_create_simulation(self):
        """Test simulation creation."""
        sim = QuantumSimulation()
        assert sim.quantum_tile_count == 0
        assert sim.size == (512, 512)  # Default size

    def test_register_qdemo(self):
        """Test registering a quantum tile."""
        sim = QuantumSimulation()
        program = [QGate.h(0), QGate.measure(0)]
        sim.register_qdemo(50, 50, 1, program, seed=42)
        assert sim.quantum_tile_count == 1

    def test_register_multiple_tiles(self):
        """Test registering multiple quantum tiles."""
        sim = QuantumSimulation()
        for i in range(4):
            program = [QGate.h(0), QGate.measure(0)]
            sim.register_qdemo(i * 10, i * 10, 1, program, seed=i)
        assert sim.quantum_tile_count == 4

    def test_tick(self):
        """Test simulation tick."""
        sim = QuantumSimulation()
        program = [QGate.h(0), QGate.measure(0)]
        sim.register_qdemo(50, 50, 1, program, seed=42)
        sim.tick()  # Should not raise
        assert sim.clock in [True, False]

    def test_tick_n(self):
        """Test multiple ticks."""
        sim = QuantumSimulation()
        program = [QGate.h(0), QGate.measure(0)]
        sim.register_qdemo(50, 50, 1, program, seed=42)
        sim.tick_n(10)  # Should not raise

    def test_get_logic(self):
        """Test reading tile logic."""
        sim = QuantumSimulation()
        program = [QGate.h(0), QGate.measure(0)]
        sim.register_qdemo(50, 50, 1, program, seed=42)
        sim.tick_n(2)
        result = sim.get_logic(50, 50)
        assert result in [0, 1]  # Single qubit measurement

    def test_set_logic(self):
        """Test setting tile logic."""
        sim = QuantumSimulation()
        sim.set_logic(100, 100, 0xDEADBEEF)
        assert sim.get_logic(100, 100) == 0xDEADBEEF

    def test_set_tile(self):
        """Test setting tile type."""
        sim = QuantumSimulation()
        sim.set_tile(100, 100, "and")  # Should not raise

    def test_invalid_qubits(self):
        """Test error on invalid qubit count."""
        sim = QuantumSimulation()
        with pytest.raises(ValueError, match="n_qubits must be 1-12"):
            sim.register_qdemo(50, 50, 0, [QGate.h(0)], seed=0)
        with pytest.raises(ValueError, match="n_qubits must be 1-12"):
            sim.register_qdemo(50, 50, 13, [QGate.h(0)], seed=0)

    def test_invalid_position(self):
        """Test error on out-of-bounds position."""
        sim = QuantumSimulation()
        with pytest.raises(ValueError, match="out of bounds"):
            sim.register_qdemo(1000, 1000, 1, [QGate.h(0)], seed=0)

    def test_empty_program(self):
        """Test error on empty program."""
        sim = QuantumSimulation()
        with pytest.raises(ValueError, match="cannot be empty"):
            sim.register_qdemo(50, 50, 1, [], seed=0)


class TestGroverCircuit:
    """Tests for Grover circuit generation."""

    def test_grover_circuit_basic(self):
        """Test basic Grover circuit generation."""
        circuit = grover_circuit(3, 5, 2)
        assert len(circuit) > 0
        # Should start with H gates
        assert str(circuit[0]) == "H(0)"
        # Should end with Measure gates
        assert "Measure" in str(circuit[-1])

    def test_grover_circuit_default_iterations(self):
        """Test Grover with default iterations."""
        circuit = grover_circuit(3, 5)  # iterations=None
        assert len(circuit) > 0

    def test_optimal_iterations(self):
        """Test optimal iteration calculation."""
        assert optimal_iterations(2) == 1  # √4 × π/4 ≈ 1.57
        assert optimal_iterations(3) == 2  # √8 × π/4 ≈ 2.22
        assert optimal_iterations(4) == 3  # √16 × π/4 ≈ 3.14

    def test_invalid_marked_state(self):
        """Test error on invalid marked state."""
        with pytest.raises(ValueError, match="out of range"):
            grover_circuit(3, 8)  # 8 >= 2^3


class TestGroverExecution:
    """Tests for Grover algorithm execution."""

    def test_run_grover_basic(self):
        """Test basic Grover execution."""
        result = run_grover(3, 5, seed=42)
        assert 0 <= result <= 7  # Valid 3-bit result

    def test_grover_deterministic_seed(self):
        """Test that same seed gives same result."""
        r1 = run_grover(3, 5, seed=12345)
        r2 = run_grover(3, 5, seed=12345)
        assert r1 == r2

    def test_grover_success_rate(self):
        """Test Grover success rate is reasonable."""
        rate = grover_success_rate(3, 5, trials=50)
        # Should succeed most of the time (theoretical: ~94.5%)
        assert rate > 0.7, f"Success rate {rate} too low"

    def test_grover_finds_marked_state(self):
        """Test Grover tends to find marked state."""
        marked = 0b101
        successes = sum(1 for seed in range(20)
                       if run_grover(3, marked, seed=seed * 1000) == marked)
        # Should find it most of the time
        assert successes >= 10, f"Only {successes}/20 successes"


class TestCircuitPatterns:
    """Tests for common circuit patterns."""

    def test_bell_state(self):
        """Test Bell state circuit."""
        circuit = bell_state()
        assert len(circuit) == 4  # H, CNOT, 2x Measure

    def test_bell_state_no_measure(self):
        """Test Bell state without measurement."""
        circuit = bell_state(measure=False)
        assert len(circuit) == 2  # H, CNOT only

    def test_ghz_state(self):
        """Test GHZ state circuit."""
        circuit = ghz_state(3)
        assert len(circuit) == 6  # H, 2x CNOT, 3x Measure

    def test_ghz_state_invalid(self):
        """Test GHZ requires at least 2 qubits."""
        with pytest.raises(ValueError):
            ghz_state(1)

    def test_uniform_superposition(self):
        """Test uniform superposition circuit."""
        circuit = uniform_superposition(3)
        assert len(circuit) == 3  # 3x H

    def test_measure_all(self):
        """Test measure all helper."""
        gates = measure_all(4)
        assert len(gates) == 4
        assert all("Measure" in str(g) for g in gates)


class TestCircuitBuilder:
    """Tests for CircuitBuilder fluent API."""

    def test_builder_basic(self):
        """Test basic circuit building."""
        circuit = CircuitBuilder(2).h(0).cnot(0, 1).measure_all().build()
        assert len(circuit) == 4

    def test_builder_chaining(self):
        """Test method chaining."""
        builder = CircuitBuilder(3)
        result = builder.h(0).x(1).z(2)
        assert result is builder  # Returns self

    def test_builder_hadamard_all(self):
        """Test hadamard_all helper."""
        circuit = CircuitBuilder(4).hadamard_all().build()
        assert len(circuit) == 4

    def test_builder_invalid_qubit(self):
        """Test error on invalid qubit index."""
        builder = CircuitBuilder(2)
        with pytest.raises(ValueError, match="out of range"):
            builder.h(5)

    def test_builder_len(self):
        """Test builder length."""
        builder = CircuitBuilder(2).h(0).h(1)
        assert len(builder) == 2

    def test_builder_repr(self):
        """Test builder string representation."""
        builder = CircuitBuilder(3).h(0)
        assert "n_qubits=3" in repr(builder)
        assert "gates=1" in repr(builder)


class TestBellStateCorrelation:
    """Test Bell state produces correlated measurements."""

    def test_bell_correlation(self):
        """Bell state should give correlated outcomes."""
        sim = QuantumSimulation()
        program = bell_state()
        sim.register_qdemo(50, 50, 2, program, seed=42)
        sim.tick_n(4)

        result = sim.get_logic(50, 50)
        bit0 = (result >> 0) & 1
        bit1 = (result >> 1) & 1

        # Bell state: always |00⟩ or |11⟩
        assert bit0 == bit1, f"Bell state uncorrelated: {bit0} != {bit1}"

    def test_bell_multiple_runs(self):
        """Multiple Bell state runs should all be correlated."""
        for seed in range(10):
            sim = QuantumSimulation()
            program = bell_state()
            sim.register_qdemo(50, 50, 2, program, seed=seed * 1234)
            sim.tick_n(4)

            result = sim.get_logic(50, 50)
            bit0 = (result >> 0) & 1
            bit1 = (result >> 1) & 1

            assert bit0 == bit1, f"Seed {seed}: Bell uncorrelated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
