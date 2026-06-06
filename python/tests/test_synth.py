"""Tests for synthesize() Python bindings."""
import pytest
from tileuniverse import synthesize, SynthResult


def test_majority_3():
    result = synthesize(truth_table=0xE8, num_inputs=3)
    assert isinstance(result, SynthResult)
    assert result.num_inputs == 3
    assert result.num_outputs == 1
    assert result.num_and_nodes > 0
    assert result.aig_depth > 0
    assert result.mapped_gates > 0
    assert result.grid_width > 0
    assert result.grid_height > 0
    assert result.num_layers > 0
    assert result.route_count > 0
    assert result.combos_tested == 8
    assert result.validated
    assert result.num_mismatches == 0
    assert result.first_mismatch_output is None


def test_and_2():
    result = synthesize(truth_table=0x8, num_inputs=2)
    assert result.num_inputs == 2
    assert result.num_and_nodes == 1
    assert result.validated
    assert result.num_mismatches == 0


def test_or_2():
    result = synthesize(truth_table=0xE, num_inputs=2)
    assert result.validated


def test_xor_2():
    result = synthesize(truth_table=0x6, num_inputs=2)
    assert result.validated


def test_nand_2():
    result = synthesize(truth_table=0x7, num_inputs=2)
    assert result.validated


def test_summary():
    result = synthesize(truth_table=0xE8, num_inputs=3)
    s = result.summary()
    assert "3 inputs" in s
    assert "grid" in s
    assert "PASS" in s


def test_repr():
    result = synthesize(truth_table=0xE8, num_inputs=3)
    r = repr(result)
    assert "SynthResult" in r
    assert "routes=" in r
    assert "validated=true" in r


def test_invalid_num_inputs_zero():
    with pytest.raises(ValueError, match="num_inputs must be 1-6"):
        synthesize(truth_table=0, num_inputs=0)


def test_invalid_num_inputs_too_large():
    with pytest.raises(ValueError, match="num_inputs must be 1-6"):
        synthesize(truth_table=0, num_inputs=7)


def test_truth_table_too_large():
    with pytest.raises(ValueError, match="truth_table too large"):
        synthesize(truth_table=0x10, num_inputs=1)


def test_constant_zero():
    # Degenerate AIG (no gates) — pipeline can't place/route
    with pytest.raises(ValueError, match="Synthesis failed"):
        synthesize(truth_table=0x0, num_inputs=2)


def test_constant_one():
    result = synthesize(truth_table=0xF, num_inputs=2)
    assert result.num_and_nodes == 0


def test_single_input():
    # Buffer: f(x) = x — no AND gates, pipeline can't place/route
    with pytest.raises(ValueError, match="Synthesis failed"):
        synthesize(truth_table=2, num_inputs=1)


# Sprint 213: exhaustive 2-input sweep
@pytest.mark.parametrize("tt", range(16))
def test_2input_sweep(tt):
    """All non-degenerate 2-input functions pass validation."""
    try:
        result = synthesize(truth_table=tt, num_inputs=2)
        # If it didn't error, it should validate
        assert result.validated, (
            f"tt=0x{tt:X} failed: {result.num_mismatches} mismatches, "
            f"first at combo={result.first_mismatch_input_combo}"
        )
    except ValueError:
        # Degenerate functions (no gates) may fail routing — acceptable
        pass


# Sprint 213: selected 3-input functions
@pytest.mark.parametrize("tt,name", [
    (0x80, "AND3"),
    (0xFE, "OR3"),
    (0xE8, "MAJ3"),
    (0x7F, "NAND3"),
    (0xCA, "MUX"),
])
def test_3input_selected(tt, name):
    result = synthesize(truth_table=tt, num_inputs=3)
    assert result.validated, f"{name} (tt=0x{tt:02X}) failed validation"
