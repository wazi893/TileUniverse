"""Tests for Sprint 215: multi-output synthesis and AIGER import/export."""
import os
import tempfile
import pytest
from tileuniverse import (
    synthesize_multi,
    synthesize_from_aiger,
    synthesize_from_aiger_bytes,
    export_to_aiger,
    export_to_aiger_ascii,
    SynthResult,
)


# ---------------------------------------------------------------------------
# D1/D5: synthesize_multi — multi-output combinational synthesis
# ---------------------------------------------------------------------------


def test_synthesize_multi_adder():
    """2-bit adder: 4 inputs → 3 outputs (s0, s1, carry)."""
    # inputs = [a0, a1, b0, b1], A = a0+2*a1, B = b0+2*b1
    result = synthesize_multi(
        truth_tables=[0x5A5A, 0x936C, 0xEC80],
        num_inputs=4,
    )
    assert isinstance(result, SynthResult)
    assert result.num_inputs == 4
    assert result.num_outputs == 3
    assert result.combos_tested == 16
    assert result.validated, (
        f"2-bit adder failed: {result.num_mismatches} mismatches, "
        f"per_output_pass={result.per_output_pass}"
    )
    assert result.per_output_pass == [True, True, True]
    assert result.per_output_mismatches == [0, 0, 0]


def test_synthesize_multi_decoder():
    """3-to-8 decoder: 3 inputs → 8 outputs (one-hot)."""
    # output[j] = 1 iff input_combo == j
    tts = [1 << j for j in range(8)]
    result = synthesize_multi(truth_tables=tts, num_inputs=3)
    assert result.num_inputs == 3
    assert result.num_outputs == 8
    assert result.combos_tested == 8
    # Sprint 216: all 8 outputs must pass
    assert result.validated, (
        f"3-to-8 decoder: {result.num_mismatches} mismatches, "
        f"per_output_pass={result.per_output_pass}"
    )
    assert result.per_output_pass == [True] * 8


def test_synthesize_multi_priority_encoder():
    """4-bit priority encoder: 4 inputs → 2 outputs (encoded position)."""
    # Priority encoder: output = index of highest set bit (MSB priority)
    # For 4 inputs [i0,i1,i2,i3], output = highest set index
    # out0 (LSB of position), out1 (MSB of position)
    # Build truth tables: input combo → priority index
    out0 = 0
    out1 = 0
    for combo in range(16):
        if combo == 0:
            pri = 0  # no input: output 0
        elif combo & 8:
            pri = 3
        elif combo & 4:
            pri = 2
        elif combo & 2:
            pri = 1
        else:
            pri = 0
        if pri & 1:
            out0 |= 1 << combo
        if pri & 2:
            out1 |= 1 << combo

    result = synthesize_multi(truth_tables=[out0, out1], num_inputs=4)
    assert result.num_inputs == 4
    assert result.num_outputs == 2
    assert result.combos_tested == 16
    assert result.validated, (
        f"priority encoder failed: {result.num_mismatches} mismatches"
    )


def test_synthesize_multi_2input_xor_and():
    """Simple 2-output: XOR + AND on same inputs."""
    result = synthesize_multi(truth_tables=[0x6, 0x8], num_inputs=2)
    assert result.num_inputs == 2
    assert result.num_outputs == 2
    assert result.validated
    assert result.per_output_pass == [True, True]


def test_synthesize_multi_per_output_reporting():
    """Per-output pass/mismatch fields are populated correctly."""
    result = synthesize_multi(truth_tables=[0x6, 0x8], num_inputs=2)
    assert len(result.per_output_pass) == 2
    assert len(result.per_output_mismatches) == 2
    assert all(isinstance(p, bool) for p in result.per_output_pass)
    assert all(isinstance(m, int) for m in result.per_output_mismatches)


def test_synthesize_multi_single_output():
    """synthesize_multi works with a single output (degenerates to synthesize)."""
    result = synthesize_multi(truth_tables=[0xE8], num_inputs=3)
    assert result.num_outputs == 1
    assert result.validated


# ---------------------------------------------------------------------------
# D1 error cases
# ---------------------------------------------------------------------------


def test_synthesize_multi_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        synthesize_multi(truth_tables=[], num_inputs=2)


def test_synthesize_multi_bad_num_inputs():
    with pytest.raises(ValueError, match="num_inputs must be 1-6"):
        synthesize_multi(truth_tables=[0x8], num_inputs=0)
    with pytest.raises(ValueError, match="num_inputs must be 1-6"):
        synthesize_multi(truth_tables=[0x8], num_inputs=7)


def test_synthesize_multi_tt_too_large():
    with pytest.raises(ValueError, match="too large"):
        synthesize_multi(truth_tables=[0x10], num_inputs=1)


# ---------------------------------------------------------------------------
# D2/D5: AIGER round-trip
# ---------------------------------------------------------------------------


def test_synthesize_from_aiger_roundtrip():
    """Multi-output: export to AIGER binary → import → synth → validate."""
    # XOR + AND
    aiger_bytes = export_to_aiger(truth_tables=[0x6, 0x8], num_inputs=2)
    assert isinstance(aiger_bytes, bytes)
    assert len(aiger_bytes) > 0

    result = synthesize_from_aiger_bytes(aiger_bytes)
    assert result.num_inputs == 2
    assert result.num_outputs == 2
    assert result.validated


def test_synthesize_from_aiger_bytes():
    """In-memory AIGER: 3-input majority."""
    aiger_bytes = export_to_aiger(truth_tables=[0xE8], num_inputs=3)
    result = synthesize_from_aiger_bytes(aiger_bytes)
    assert result.num_inputs == 3
    assert result.num_outputs == 1
    assert result.validated


def test_export_aiger_ascii():
    """ASCII AIGER export for multi-output circuit."""
    text = export_to_aiger_ascii(truth_tables=[0x6, 0x8], num_inputs=2)
    assert isinstance(text, str)
    assert text.startswith("aag ")
    # Header: aag M I L O A
    header = text.split("\n")[0]
    parts = header.split()
    assert parts[0] == "aag"
    assert parts[2] == "2"  # I = 2 inputs
    assert parts[3] == "0"  # L = 0 latches
    assert parts[4] == "2"  # O = 2 outputs


def test_aiger_roundtrip_multi_output_adder():
    """Full round-trip: 2-bit adder → AIGER binary → import → synth → validate."""
    aiger_bytes = export_to_aiger(
        truth_tables=[0x5A5A, 0x936C, 0xEC80],
        num_inputs=4,
    )
    result = synthesize_from_aiger_bytes(aiger_bytes)
    assert result.num_inputs == 4
    assert result.num_outputs == 3
    assert result.validated, (
        f"AIGER round-trip adder: {result.num_mismatches} mismatches"
    )


def test_aiger_ascii_roundtrip():
    """ASCII AIGER: export → re-import as bytes → synth → validate."""
    text = export_to_aiger_ascii(truth_tables=[0xE8], num_inputs=3)
    result = synthesize_from_aiger_bytes(text.encode("utf-8"))
    assert result.validated


def test_synthesize_from_aiger_file():
    """File-based AIGER import: write temp file → synthesize_from_aiger(path)."""
    aiger_bytes = export_to_aiger(truth_tables=[0x6, 0x8], num_inputs=2)
    with tempfile.NamedTemporaryFile(suffix=".aig", delete=False) as f:
        f.write(aiger_bytes)
        path = f.name
    try:
        result = synthesize_from_aiger(path)
        assert result.num_inputs == 2
        assert result.num_outputs == 2
        assert result.validated
    finally:
        os.unlink(path)


def test_aiger_ascii_roundtrip_multi_output():
    """Multi-output ASCII AIGER round-trip: XOR + AND."""
    text = export_to_aiger_ascii(truth_tables=[0x6, 0x8], num_inputs=2)
    result = synthesize_from_aiger_bytes(text.encode("utf-8"))
    assert result.num_inputs == 2
    assert result.num_outputs == 2
    assert result.validated


# ---------------------------------------------------------------------------
# D2 error cases
# ---------------------------------------------------------------------------


def test_export_aiger_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        export_to_aiger(truth_tables=[], num_inputs=2)


def test_export_aiger_bad_num_inputs():
    with pytest.raises(ValueError, match="num_inputs must be 1-6"):
        export_to_aiger(truth_tables=[0x8], num_inputs=0)


def test_synthesize_from_aiger_bytes_bad_data():
    with pytest.raises(ValueError, match="AIGER parse error"):
        synthesize_from_aiger_bytes(b"not valid aiger data")


def test_export_aiger_rejects_oversized_tt():
    """export_to_aiger must reject oversized truth tables (same as synthesize_multi)."""
    with pytest.raises(ValueError, match="too large"):
        export_to_aiger(truth_tables=[0x10], num_inputs=1)


def test_export_aiger_ascii_rejects_oversized_tt():
    """export_to_aiger_ascii must reject oversized truth tables."""
    with pytest.raises(ValueError, match="too large"):
        export_to_aiger_ascii(truth_tables=[0x10], num_inputs=1)


# ---------------------------------------------------------------------------
# D3: Per-output reporting on single-output synthesize
# ---------------------------------------------------------------------------


def test_single_output_per_output_fields():
    """Existing synthesize() also gets per_output_pass/mismatches."""
    from tileuniverse import synthesize
    result = synthesize(truth_table=0xE8, num_inputs=3)
    assert result.per_output_pass == [True]
    assert result.per_output_mismatches == [0]


# ---------------------------------------------------------------------------
# Sprint 216: Convergence detection + decoder all-pass
# ---------------------------------------------------------------------------


def test_convergence_fields():
    """SynthResult exposes converged and convergence_rounds."""
    from tileuniverse import synthesize
    result = synthesize(truth_table=0xE8, num_inputs=3)
    assert hasattr(result, "converged")
    assert hasattr(result, "convergence_rounds")
    assert result.converged is True
    assert result.convergence_rounds > 0


def test_synthesize_multi_decoder_all_pass():
    """Sprint 216 D5: all 8 decoder outputs pass."""
    tts = [1 << j for j in range(8)]
    result = synthesize_multi(truth_tables=tts, num_inputs=3)
    assert result.validated
    assert all(result.per_output_pass)
    assert result.num_mismatches == 0
