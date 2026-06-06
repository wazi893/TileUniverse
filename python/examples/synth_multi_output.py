"""Sprint 215 Demo: Multi-output synthesis and AIGER interop.

Shows:
1. 2-bit adder from truth tables (3 outputs)
2. 3-to-8 decoder (8 outputs)
3. AIGER binary round-trip
4. AIGER ASCII export
"""
from tileuniverse import (
    synthesize_multi,
    synthesize_from_aiger_bytes,
    export_to_aiger,
    export_to_aiger_ascii,
)


def demo_adder():
    """2-bit adder: A[1:0] + B[1:0] = {carry, S[1:0]}."""
    print("=== 2-bit Adder (4 inputs, 3 outputs) ===")
    # inputs = [a0, a1, b0, b1]
    # s0 = 0x5A5A, s1 = 0x936C, cout = 0xEC80
    result = synthesize_multi(
        truth_tables=[0x5A5A, 0x936C, 0xEC80],
        num_inputs=4,
    )
    print(f"  {result.summary()}")
    print(f"  Per-output pass: {result.per_output_pass}")
    print(f"  Per-output mismatches: {result.per_output_mismatches}")
    print()
    return result


def demo_decoder():
    """3-to-8 decoder: one-hot output for each input combo."""
    print("=== 3-to-8 Decoder (3 inputs, 8 outputs) ===")
    tts = [1 << j for j in range(8)]
    result = synthesize_multi(truth_tables=tts, num_inputs=3)
    passing = sum(1 for p in result.per_output_pass if p)
    print(f"  {result.summary()}")
    print(f"  Outputs passing: {passing}/8")
    print(f"  Per-output pass: {result.per_output_pass}")
    print(f"  Converged: {result.converged} ({result.convergence_rounds} rounds)")
    print()
    return result


def demo_aiger_roundtrip():
    """Export adder to AIGER binary, re-import, validate."""
    print("=== AIGER Binary Round-Trip ===")
    tts = [0x5A5A, 0x936C, 0xEC80]

    # Export
    aiger_bytes = export_to_aiger(truth_tables=tts, num_inputs=4)
    print(f"  Exported {len(aiger_bytes)} bytes of binary AIGER")

    # Re-import and synth
    result = synthesize_from_aiger_bytes(aiger_bytes)
    print(f"  Re-imported: {result.summary()}")
    print(f"  Round-trip validated: {result.validated}")
    print()
    return result


def demo_aiger_ascii():
    """Export adder to ASCII AIGER (human-readable)."""
    print("=== AIGER ASCII Export ===")
    text = export_to_aiger_ascii(truth_tables=[0x5A5A, 0x936C, 0xEC80], num_inputs=4)
    lines = text.strip().split("\n")
    print(f"  Header: {lines[0]}")
    print(f"  Total lines: {len(lines)}")
    # Show first few lines
    for line in lines[:6]:
        print(f"    {line}")
    if len(lines) > 6:
        print(f"    ... ({len(lines) - 6} more lines)")
    print()


if __name__ == "__main__":
    adder = demo_adder()
    decoder = demo_decoder()
    rt = demo_aiger_roundtrip()
    demo_aiger_ascii()

    print("=== Summary ===")
    print(f"  Adder: {'PASS' if adder.validated else 'FAIL'}")
    passing = sum(1 for p in decoder.per_output_pass if p)
    print(f"  Decoder: {passing}/8 outputs pass")
    print(f"  AIGER round-trip: {'PASS' if rt.validated else 'FAIL'}")
