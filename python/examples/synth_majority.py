"""Synthesize 3-input majority to tile fabric."""
from tileuniverse import synthesize

result = synthesize(truth_table=0xE8, num_inputs=3)

print("3-input majority synthesis")
print(f"Inputs:      {result.num_inputs}")
print(f"Outputs:     {result.num_outputs}")
print(f"AIG ANDs:    {result.num_and_nodes}")
print(f"Mapped:      {result.mapped_gates} tile-native gates")
print(f"Depth:       {result.aig_depth}")
print(f"Grid:        {result.grid_width} x {result.grid_height} x {result.num_layers}")
print(f"Route tiles: {result.route_count}")
print(f"Validated:   {result.validated}")
print(f"Summary:     {result.summary()}")

assert result.validated
assert result.num_mismatches == 0

print("\nAll assertions passed!")
