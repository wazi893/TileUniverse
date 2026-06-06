# =============================================================================
# tests/test_dj_bv.py
# =============================================================================
#
# SPRINT 2.3: Python tests for Deutsch-Jozsa and Bernstein-Vazirani algorithms.
#
# Run with: pytest tests/test_dj_bv.py -v
#
# =============================================================================

"""Tests for Deutsch-Jozsa and Bernstein-Vazirani algorithms."""

import pytest
from tileuniverse import QuantumSimulation, QGate
from tileuniverse._core import (
    DJOracle,
    deutsch_jozsa_circuit,
    bernstein_vazirani_circuit,
)
from tileuniverse.algorithms import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    compare_quantum_speedup,
)


class TestDJOracle:
    """Tests for DJOracle class."""

    def test_constant_zero(self):
        """Test constant-zero oracle creation."""
        oracle = DJOracle.constant_zero()
        assert oracle.is_constant()
        assert not oracle.is_balanced()
        assert "constant" in str(oracle)

    def test_constant_one(self):
        """Test constant-one oracle creation."""
        oracle = DJOracle.constant_one()
        assert oracle.is_constant()
        assert not oracle.is_balanced()

    def test_balanced(self):
        """Test balanced oracle creation."""
        oracle = DJOracle.balanced(5)
        assert oracle.is_balanced()
        assert not oracle.is_constant()
        assert "balanced" in str(oracle)

    def test_repr(self):
        """Test oracle string representation."""
        assert "constant_zero" in repr(DJOracle.constant_zero())
        assert "constant_one" in repr(DJOracle.constant_one())
        assert "balanced(5)" in repr(DJOracle.balanced(5))


class TestDeutschJozsaCircuit:
    """Tests for Deutsch-Jozsa circuit generation."""

    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = deutsch_jozsa_circuit(3, DJOracle.balanced(5))
        assert len(circuit) > 0

    def test_circuit_starts_with_x_on_ancilla(self):
        """Circuit should start with X on ancilla to prepare |1⟩."""
        circuit = deutsch_jozsa_circuit(3, DJOracle.balanced(5))
        # First gate should be X(3) - ancilla is qubit n
        assert str(circuit[0]) == "X(3)"

    def test_circuit_ends_with_measurements(self):
        """Circuit should end with measurements of input qubits."""
        circuit = deutsch_jozsa_circuit(3, DJOracle.balanced(5))
        # Last 3 gates should be Measure(0), Measure(1), Measure(2)
        assert "Measure(0)" in str(circuit[-3])
        assert "Measure(1)" in str(circuit[-2])
        assert "Measure(2)" in str(circuit[-1])

    def test_invalid_qubits(self):
        """Test error on invalid qubit count."""
        with pytest.raises(ValueError):
            deutsch_jozsa_circuit(0, DJOracle.constant_zero())
        with pytest.raises(ValueError):
            deutsch_jozsa_circuit(11, DJOracle.constant_zero())


class TestDeutschJozsaExecution:
    """Tests for Deutsch-Jozsa algorithm execution."""

    def test_constant_zero_detected(self):
        """Constant-zero oracle should be detected as constant."""
        is_constant = run_deutsch_jozsa(3, DJOracle.constant_zero())
        assert is_constant is True

    def test_constant_one_detected(self):
        """Constant-one oracle should be detected as constant."""
        is_constant = run_deutsch_jozsa(3, DJOracle.constant_one())
        assert is_constant is True

    def test_balanced_detected(self):
        """Balanced oracle should be detected as balanced."""
        is_constant = run_deutsch_jozsa(3, DJOracle.balanced(5))
        assert is_constant is False

    def test_all_balanced_patterns(self):
        """All balanced patterns should be detected as balanced."""
        n = 3
        for pattern in range(1, 2**n):
            is_constant = run_deutsch_jozsa(n, DJOracle.balanced(pattern))
            assert is_constant is False, f"Pattern {pattern} should be balanced"

    def test_deterministic(self):
        """DJ is deterministic - same result regardless of seed."""
        r1 = run_deutsch_jozsa(3, DJOracle.balanced(5), seed=111)
        r2 = run_deutsch_jozsa(3, DJOracle.balanced(5), seed=222)
        r3 = run_deutsch_jozsa(3, DJOracle.balanced(5), seed=333)
        assert r1 == r2 == r3 == False

    def test_various_sizes(self):
        """Test DJ works for various qubit counts."""
        for n in range(1, 6):
            # Constant
            assert run_deutsch_jozsa(n, DJOracle.constant_zero()) is True
            assert run_deutsch_jozsa(n, DJOracle.constant_one()) is True
            # Balanced
            assert run_deutsch_jozsa(n, DJOracle.balanced(1)) is False


class TestBernsteinVaziraniCircuit:
    """Tests for Bernstein-Vazirani circuit generation."""

    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = bernstein_vazirani_circuit(4, 0b1010)
        assert len(circuit) > 0

    def test_circuit_structure(self):
        """Circuit should have correct structure."""
        circuit = bernstein_vazirani_circuit(3, 0b101)
        # First gate: X on ancilla
        assert str(circuit[0]) == "X(3)"
        # Then Hadamards
        assert str(circuit[1]) == "H(0)"

    def test_invalid_inputs(self):
        """Test error on invalid inputs."""
        with pytest.raises(ValueError):
            bernstein_vazirani_circuit(0, 0)
        with pytest.raises(ValueError):
            bernstein_vazirani_circuit(3, 8)  # 8 >= 2^3


class TestBernsteinVaziraniExecution:
    """Tests for Bernstein-Vazirani algorithm execution."""

    def test_finds_zero_string(self):
        """Should find hidden string = 0."""
        result = run_bernstein_vazirani(4, 0)
        assert result == 0

    def test_finds_all_ones_string(self):
        """Should find hidden string = all 1s."""
        n = 5
        hidden = (1 << n) - 1  # 11111
        result = run_bernstein_vazirani(n, hidden)
        assert result == hidden

    def test_finds_arbitrary_string(self):
        """Should find arbitrary hidden string."""
        result = run_bernstein_vazirani(6, 0b101010)
        assert result == 0b101010

    def test_exhaustive_small(self):
        """Exhaustively test all strings for small n."""
        n = 4
        for hidden in range(2**n):
            result = run_bernstein_vazirani(n, hidden)
            assert result == hidden, f"Failed for hidden={hidden:04b}"

    def test_deterministic(self):
        """BV is deterministic - same result regardless of seed."""
        hidden = 0b10101
        r1 = run_bernstein_vazirani(5, hidden, seed=111)
        r2 = run_bernstein_vazirani(5, hidden, seed=222)
        r3 = run_bernstein_vazirani(5, hidden, seed=333)
        assert r1 == r2 == r3 == hidden

    def test_various_sizes(self):
        """Test BV works for various qubit counts."""
        for n in range(1, 8):
            # Test a few patterns for each size
            for hidden in [0, 1, (1 << n) - 1, (1 << n) // 2]:
                if hidden < (1 << n):
                    result = run_bernstein_vazirani(n, hidden)
                    assert result == hidden


class TestSpeedupComparison:
    """Tests for speedup comparison utility."""

    def test_returns_all_algorithms(self):
        """Should return data for all algorithms."""
        stats = compare_quantum_speedup(5)
        assert 'grover' in stats
        assert 'deutsch_jozsa' in stats
        assert 'bernstein_vazirani' in stats

    def test_speedup_values(self):
        """Speedup values should be positive."""
        stats = compare_quantum_speedup(5)
        for algo, data in stats.items():
            assert data['speedup'] > 0
            assert data['classical_queries'] >= data['quantum_queries']

    def test_dj_exponential_speedup(self):
        """DJ should show exponential speedup."""
        stats = compare_quantum_speedup(6)
        dj = stats['deutsch_jozsa']
        # For n=6, classical needs 2^5 + 1 = 33 queries, quantum needs 1
        assert dj['classical_queries'] == 33
        assert dj['quantum_queries'] == 1
        assert dj['speedup'] == 33

    def test_bv_linear_speedup(self):
        """BV should show linear speedup."""
        stats = compare_quantum_speedup(10)
        bv = stats['bernstein_vazirani']
        assert bv['classical_queries'] == 10
        assert bv['quantum_queries'] == 1
        assert bv['speedup'] == 10


class TestDJBVOnQDemoTile:
    """Integration tests running on actual QDemo tiles."""

    def test_dj_on_tile(self):
        """Test DJ running on a QDemo tile."""
        sim = QuantumSimulation()
        n = 3
        circuit = deutsch_jozsa_circuit(n, DJOracle.balanced(0b101))
        
        sim.register_qdemo(50, 50, n + 1, circuit, seed=42)
        sim.tick_n(len(circuit))
        
        result = sim.get_logic(50, 50)
        input_bits = result & ((1 << n) - 1)
        
        # Balanced → should NOT be all zeros
        assert input_bits != 0

    def test_bv_on_tile(self):
        """Test BV running on a QDemo tile."""
        sim = QuantumSimulation()
        n = 4
        hidden = 0b1010
        circuit = bernstein_vazirani_circuit(n, hidden)
        
        sim.register_qdemo(50, 50, n + 1, circuit, seed=42)
        sim.tick_n(len(circuit))
        
        result = sim.get_logic(50, 50)
        input_bits = result & ((1 << n) - 1)
        
        assert input_bits == hidden

    def test_multiple_dj_tiles(self):
        """Test multiple DJ tiles simultaneously."""
        sim = QuantumSimulation()
        n = 3
        
        oracles = [
            DJOracle.constant_zero(),
            DJOracle.constant_one(),
            DJOracle.balanced(5),
            DJOracle.balanced(3),
        ]
        positions = [(10, 10), (20, 10), (10, 20), (20, 20)]
        expected = [True, True, False, False]  # is_constant?
        
        max_len = 0
        for oracle, (x, y) in zip(oracles, positions):
            circuit = deutsch_jozsa_circuit(n, oracle)
            max_len = max(max_len, len(circuit))
            sim.register_qdemo(x, y, n + 1, circuit, seed=x * y)
        
        sim.tick_n(max_len)
        
        for i, (x, y) in enumerate(positions):
            result = sim.get_logic(x, y)
            input_bits = result & ((1 << n) - 1)
            is_constant = (input_bits == 0)
            assert is_constant == expected[i], \
                f"Tile {i}: expected is_constant={expected[i]}, got {is_constant}"

    def test_multiple_bv_tiles(self):
        """Test multiple BV tiles simultaneously."""
        sim = QuantumSimulation()
        n = 4
        
        hidden_strings = [0b0000, 0b1111, 0b1010, 0b0101]
        positions = [(30, 30), (40, 30), (30, 40), (40, 40)]
        
        max_len = 0
        for hidden, (x, y) in zip(hidden_strings, positions):
            circuit = bernstein_vazirani_circuit(n, hidden)
            max_len = max(max_len, len(circuit))
            sim.register_qdemo(x, y, n + 1, circuit, seed=hidden * 1234)
        
        sim.tick_n(max_len)
        
        for hidden, (x, y) in zip(hidden_strings, positions):
            result = sim.get_logic(x, y)
            input_bits = result & ((1 << n) - 1)
            assert input_bits == hidden, \
                f"Position ({x},{y}): expected {hidden:04b}, got {input_bits:04b}"


class TestAlgorithmComparison:
    """Compare all three algorithms side by side."""

    def test_all_algorithms_work(self):
        """Verify all three algorithms work on same simulation."""
        from tileuniverse.algorithms import run_grover
        
        # Grover
        grover_result = run_grover(3, 5, seed=42)
        assert 0 <= grover_result <= 7
        
        # DJ
        dj_constant = run_deutsch_jozsa(3, DJOracle.constant_zero())
        dj_balanced = run_deutsch_jozsa(3, DJOracle.balanced(5))
        assert dj_constant is True
        assert dj_balanced is False
        
        # BV
        bv_result = run_bernstein_vazirani(4, 0b1010)
        assert bv_result == 0b1010

    def test_speedup_summary(self):
        """Print speedup summary for all algorithms."""
        stats = compare_quantum_speedup(8)
        
        print("\n=== Quantum Speedup Summary (n=8) ===")
        for algo, data in stats.items():
            print(f"{algo}:")
            print(f"  Classical: {data['classical_queries']} queries")
            print(f"  Quantum:   {data['quantum_queries']} query")
            print(f"  Speedup:   {data['speedup']:.1f}x ({data['speedup_type']})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
