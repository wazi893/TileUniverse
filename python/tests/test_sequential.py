"""Tests for SequentialCircuit Python bindings."""
import pytest
from tileuniverse import SequentialCircuit


# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------

def test_counter_3bit_basic():
    counter = SequentialCircuit.counter(3)
    assert counter.num_state_bits == 3
    assert counter.num_inputs == 0
    assert counter.num_outputs == 0
    assert counter.state_value() == 0

    # Tick through full cycle: 0→1→2→...→7→0
    for expected in range(1, 9):
        counter.tick()
        assert counter.state_value() == expected % 8, (
            f"cycle {expected}: expected {expected % 8}, got {counter.state_value()}"
        )


def test_counter_4bit_wrap():
    counter = SequentialCircuit.counter(4)
    for _ in range(16):
        counter.tick()
    assert counter.state_value() == 0, "should wrap to 0 after 16 ticks"


def test_counter_8bit_full_cycle():
    counter = SequentialCircuit.counter(8)
    for expected in range(256):
        assert counter.state_value() == expected
        counter.tick()
    assert counter.state_value() == 0


def test_counter_set_state():
    counter = SequentialCircuit.counter(3)
    counter.set_state_value(5)
    assert counter.state_value() == 5
    counter.tick()
    assert counter.state_value() == 6
    counter.tick()
    assert counter.state_value() == 7
    counter.tick()
    assert counter.state_value() == 0


def test_counter_set_state_list():
    counter = SequentialCircuit.counter(3)
    counter.set_state([True, False, True])  # 5 = 101
    assert counter.state_value() == 5


def test_counter_cycle_count():
    counter = SequentialCircuit.counter(3)
    assert counter.num_cycles == 0
    counter.tick()
    assert counter.num_cycles == 1
    for _ in range(9):
        counter.tick()
    assert counter.num_cycles == 10


def test_counter_grid_info():
    counter = SequentialCircuit.counter(3)
    info = counter.grid_info()
    assert info is not None
    w, h, layers = info
    assert w > 0 and h > 0 and layers > 0


def test_counter_repr():
    counter = SequentialCircuit.counter(3)
    r = repr(counter)
    assert "SequentialCircuit" in r
    assert "state_bits=3" in r


# ---------------------------------------------------------------------------
# Shift register tests
# ---------------------------------------------------------------------------

def test_shift_register_4bit():
    sr = SequentialCircuit.shift_register(4)
    assert sr.num_state_bits == 4
    assert sr.num_inputs == 1

    # Shift in: 1, 0, 1, 1
    for bit in [True, False, True, True]:
        sr.tick([bit])

    assert sr.state() == [True, False, True, True]


def test_shift_register_alternating():
    sr = SequentialCircuit.shift_register(4)
    for cycle in range(8):
        sr.tick([cycle % 2 == 0])

    # Last 4 bits: cycle 4=True, 5=False, 6=True, 7=False
    assert sr.state() == [True, False, True, False]


def test_shift_register_grid_info():
    sr = SequentialCircuit.shift_register(4)
    # Shift register has no synth grid
    assert sr.grid_info() is None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_counter_invalid_bits():
    with pytest.raises(ValueError):
        SequentialCircuit.counter(0)
    with pytest.raises(ValueError):
        SequentialCircuit.counter(9)


def test_shift_register_invalid_bits():
    with pytest.raises(ValueError):
        SequentialCircuit.shift_register(0)
    with pytest.raises(ValueError):
        SequentialCircuit.shift_register(9)


def test_counter_tick_wrong_inputs():
    counter = SequentialCircuit.counter(3)
    with pytest.raises(ValueError):
        counter.tick([True])  # counter has 0 inputs


def test_shift_register_tick_wrong_inputs():
    sr = SequentialCircuit.shift_register(4)
    with pytest.raises(ValueError):
        sr.tick()  # shift register needs 1 input
    with pytest.raises(ValueError):
        sr.tick([True, False])  # too many


def test_set_state_wrong_size():
    counter = SequentialCircuit.counter(3)
    with pytest.raises(ValueError):
        counter.set_state([True, False])  # need 3 bits


# ---------------------------------------------------------------------------
# from_aig_spec tests
# ---------------------------------------------------------------------------

def test_from_aig_spec_counter():
    """Build a 2-bit counter from truth tables."""
    # 2-bit counter: inputs = [s0, s1], outputs = [ns0, ns1]
    # ns0 = NOT s0:  tt = 0b01 = 1 (for 2 inputs: 0→1, 1→0, 2→1, 3→0 → 0101 = 5)
    # Wait, 2 inputs means 4-entry truth table.
    # s0=0,s1=0 → ns0=1: bit 0 = 1
    # s0=1,s1=0 → ns0=0: bit 1 = 0
    # s0=0,s1=1 → ns0=1: bit 2 = 1
    # s0=1,s1=1 → ns0=0: bit 3 = 0
    # ns0 tt = 0b0101 = 5
    #
    # ns1 = s1 XOR s0:
    # s0=0,s1=0 → ns1=0: bit 0 = 0
    # s0=1,s1=0 → ns1=1: bit 1 = 1
    # s0=0,s1=1 → ns1=1: bit 2 = 1
    # s0=1,s1=1 → ns1=0: bit 3 = 0
    # ns1 tt = 0b0110 = 6
    circuit = SequentialCircuit.from_aig_spec(
        num_latches=2,
        truth_tables=[(5, 2), (6, 2)],
    )
    assert circuit.num_state_bits == 2
    assert circuit.state_value() == 0

    # Should count 0→1→2→3→0
    for expected in [1, 2, 3, 0]:
        circuit.tick()
        assert circuit.state_value() == expected
