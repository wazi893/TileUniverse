"""Demonstrate upper-bank ROM fetch (PC >= 64) on the V2 tile CPU."""
from tileuniverse import V2Cpu

# Build a program that uses upper bank ROM (addresses 64+).
# Fill first 63 slots with NOPs, then put real code at address 63+.
nops = "NOP\n" * 62
asm = nops + """
; Now at address 62 — about to cross into upper bank (64+)
LDI R0, 100
LDI R1, 200
MOV R2, R0
ADD R2, R1    ; R2 = 300, executes from upper bank
HALT
"""

cpu = V2Cpu.from_asm(asm)
cpu.run(10000)
counters = cpu.assist_counters()

print("Upper-bank ROM fetch demo")
print(f"R0 = {cpu.reg(0)} (expected 100)")
print(f"R1 = {cpu.reg(1)} (expected 200)")
print(f"R2 = {cpu.reg(2)} (expected 300)")
print(f"Halted: {cpu.halted}")
print(f"Cycles: {cpu.cycle_count}")
print(f"Final PC: {cpu.pc}")
print(f"Upper-bank assists: {counters['rom_upper_bank_group_select']}")

assert cpu.reg(0) == 100
assert cpu.reg(1) == 200
assert cpu.reg(2) == 300
assert cpu.halted
assert counters["rom_upper_bank_group_select"] == 0
print("\nAll assertions passed!")
