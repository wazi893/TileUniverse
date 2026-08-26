"""Tests for V2Cpu Python bindings."""
import json

import pytest
from tileuniverse import V2Cpu, V2Trace, v2_ground_truth_jsonl


def test_ldi_halt():
    cpu = V2Cpu.from_asm("LDI R0, 42\nHALT")
    cpu.run(100)
    assert cpu.reg(0) == 42
    assert cpu.halted
    assert cpu.pc == 1


def test_arithmetic():
    cpu = V2Cpu.from_asm("LDI R0, 10\nLDI R1, 20\nMOV R2, R0\nADD R2, R1\nHALT")
    cpu.run(1000)
    assert cpu.reg(2) == 30


def test_sub_carry():
    cpu = V2Cpu.from_asm("LDI R0, 5\nLDI R1, 10\nMOV R2, R0\nSUB R2, R1\nHALT")
    cpu.run(1000)
    assert cpu.flags()["carry"]


def test_regs():
    cpu = V2Cpu.from_asm("LDI R0, 1\nLDI R1, 2\nLDI R2, 3\nHALT")
    cpu.run(1000)
    regs = cpu.regs()
    assert len(regs) == 16
    assert regs[0] == 1
    assert regs[1] == 2
    assert regs[2] == 3


def test_ram():
    cpu = V2Cpu.from_asm("LDI R0, 99\nLDI R1, 0\nST [R1], R0\nHALT")
    cpu.run(1000)
    assert cpu.ram(0) == 99


def test_ram_dump():
    cpu = V2Cpu.from_asm("HALT")
    cpu.run(100)
    dump = cpu.ram_dump()
    assert len(dump) == 128


def test_step():
    cpu = V2Cpu.from_asm("LDI R0, 1\nLDI R1, 2\nHALT")
    cpu.step()
    assert not cpu.halted


def test_run_traced():
    cpu = V2Cpu.from_asm("LDI R0, 5\nHALT")
    trace = cpu.run_traced(100)
    assert isinstance(trace, V2Trace)
    assert trace.num_cycles > 0
    assert len(trace.pcs) == trace.num_cycles
    r0_vals = trace.reg_trace(0)
    assert r0_vals[-1] == 5


def test_trace_regs_at():
    cpu = V2Cpu.from_asm("LDI R0, 7\nHALT")
    trace = cpu.run_traced(100)
    regs = trace.regs_at(0)
    assert len(regs) == 16


def test_ground_truth_jsonl():
    doc = v2_ground_truth_jsonl("LDI R0, 41\nLDI R1, 1\nADD R0, R1\nHALT")
    lines = doc.splitlines()
    assert len(lines) >= 4  # header + LDI, LDI, ADD retirements
    header = json.loads(lines[0])
    assert header["type"] == "program"
    assert len(header["program_words"]) == 4
    assert len(header["initial_regs"]) == 16
    steps = [json.loads(line) for line in lines[1:]]
    # Folding reg-write deltas over initial_regs yields absolute state.
    regs = list(header["initial_regs"])
    for step in steps:
        assert "asm" in step
        for write in step["reg_writes"]:
            regs[write["reg"]] = write["value"]
    assert regs[0] == 42
    assert regs[1] == 1


def test_ground_truth_jsonl_non_halting_raises():
    with pytest.raises(RuntimeError):
        v2_ground_truth_jsonl("LDI R0, 1\nLDI R1, 2", max_cycles=16)


def test_cycle_count():
    cpu = V2Cpu.from_asm("NOP\nNOP\nHALT")
    cpu.run(1000)
    assert cpu.cycle_count > 0


def test_from_program():
    # Raw words: LDI R0, 42 ; HALT
    cpu = V2Cpu.from_program([0x182A, 0x0800])
    cpu.run(100)
    assert cpu.reg(0) == 42
    assert cpu.halted


def test_synth_mode_off():
    cpu = V2Cpu.from_asm("LDI R0, 42\nHALT", synth="off")
    cpu.run(100)
    assert cpu.synth_mode == "off"
    assert cpu.reg(0) == 42


def test_assist_counters():
    cpu = V2Cpu.from_asm("HALT")
    counters = cpu.assist_counters()
    assert set(counters) == {
        "stage_f_bank_switches",
        "stage_f_mixed_dual_capture",
        "stage_x_mixed_software",
        "ram_high_bank_read_swaps",
        "rom_upper_bank_group_select",
    }


def test_repr():
    cpu = V2Cpu.from_asm("HALT")
    r = repr(cpu)
    assert "V2Cpu" in r


def test_upper_bank_fetch():
    """Programs spanning both ROM banks execute correctly."""
    nops = "NOP\n" * 62
    asm = nops + "LDI R0, 77\nHALT"
    cpu = V2Cpu.from_asm(asm)
    cpu.run(10000)
    assert cpu.reg(0) == 77
    assert cpu.halted


def test_invalid_register():
    cpu = V2Cpu.from_asm("HALT")
    cpu.run(10)
    with pytest.raises(ValueError):
        cpu.reg(16)


def test_invalid_ram():
    cpu = V2Cpu.from_asm("HALT")
    cpu.run(10)
    with pytest.raises(ValueError):
        cpu.ram(128)


def test_invalid_synth_mode():
    with pytest.raises(ValueError, match="synth must be"):
        V2Cpu.from_asm("HALT", synth="bogus")


def test_assembly_error():
    with pytest.raises(ValueError, match="Assembly error"):
        V2Cpu.from_asm("INVALID_OPCODE R0, R1")
