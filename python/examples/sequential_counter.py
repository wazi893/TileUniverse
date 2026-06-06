"""Sequential counter demo — Sprint 214.

Builds a 4-bit binary counter using the synth pipeline for combinational
next-state logic and software-maintained state feedback. Ticks for 20 cycles
and prints the state trace.
"""

from tileuniverse import SequentialCircuit


def demo_counter():
    print("=== 4-bit Counter ===")
    counter = SequentialCircuit.counter(4)
    print(f"Built: {counter}")
    if counter.grid_info():
        w, h, layers = counter.grid_info()
        print(f"Combinational grid: {w}x{h}x{layers}")
    print()

    print("Cycle | State (bin) | Value")
    print("------+-------------+------")
    print(f"    0 | {counter.state_value():04b}        |     {counter.state_value()}")
    for cycle in range(1, 21):
        counter.tick()
        val = counter.state_value()
        print(f"   {cycle:2d} | {val:04b}        |    {val:2d}")
    print()


def demo_shift_register():
    print("=== 8-bit Shift Register ===")
    sr = SequentialCircuit.shift_register(8)
    print(f"Built: {sr}")
    print()

    # Shift in a pattern: 1,0,1,1,0,1,0,0 (0xB4 reversed = 0x2D)
    pattern = [True, False, True, True, False, True, False, False]
    print("Shifting in pattern:", "".join("1" if b else "0" for b in pattern))
    print()

    print("Cycle | Input | State")
    print("------+-------+----------")
    for i, bit in enumerate(pattern):
        sr.tick([bit])
        state_str = "".join("1" if b else "0" for b in sr.state())
        print(f"    {i+1} |   {'1' if bit else '0'}   | {state_str}")

    # Shift in 8 more zeros to flush
    print("\nFlushing with zeros...")
    for i in range(8):
        sr.tick([False])
        state_str = "".join("1" if b else "0" for b in sr.state())
        print(f"   {i+9:2d} |   0   | {state_str}")
    print()


def demo_from_aig_spec():
    print("=== 2-bit Counter from Truth Tables ===")
    # ns0 = NOT s0: tt = 0b0101 = 5
    # ns1 = s1 XOR s0: tt = 0b0110 = 6
    circuit = SequentialCircuit.from_aig_spec(
        num_latches=2,
        truth_tables=[(5, 2), (6, 2)],
    )
    print(f"Built: {circuit}")
    print()

    print("Cycle | State")
    print("------+------")
    print(f"    0 |     {circuit.state_value()}")
    for cycle in range(1, 9):
        circuit.tick()
        print(f"    {cycle} |     {circuit.state_value()}")
    print()


if __name__ == "__main__":
    demo_counter()
    demo_shift_register()
    demo_from_aig_spec()
    print("All demos completed successfully.")
