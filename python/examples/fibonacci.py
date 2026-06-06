"""Fibonacci sequence on the V2 tile CPU."""
from tileuniverse import V2Cpu

asm = """
; Compute Fibonacci numbers: F(0)..F(10)
; R0 = F(n-2), R1 = F(n-1), R2 = F(n), R3 = counter, R4 = limit
LDI R0, 0       ; F(0)
LDI R1, 1       ; F(1)
LDI R3, 2       ; counter starts at 2
LDI R4, 11      ; compute up to F(10)

loop:
    MOV R2, R0
    ADD R2, R1   ; R2 = F(n-2) + F(n-1)
    MOV R0, R1   ; shift window
    MOV R1, R2
    LDI R5, 1
    ADD R3, R5   ; counter++
    CMP R3, R4
    JNZ loop

HALT
"""

cpu = V2Cpu.from_asm(asm)
trace = cpu.run_traced(10000)

print("Fibonacci on V2 tile CPU")
print(f"Cycles: {trace.num_cycles}")
print(f"F(10) = {cpu.reg(1)}")
print(f"Register trace (R1 = current fib):")
for i, val in enumerate(trace.reg_trace(1)):
    print(f"  cycle {i:3d}: R1 = {val}")
