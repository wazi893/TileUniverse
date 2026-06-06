# Archived Code

This directory contains archived code that is no longer part of the active codebase.

## tile_cpu/ (archived 2025-02-02)

**Reason:** Early prototype for building CPUs from simulation tiles. Superseded by GPU-native approaches in `cuda_tiles.rs` which achieve 1000x+ better throughput.

**Contents:**
- Gate-level CPU built from TileType primitives (And, Or, Register8, Mux8to1, etc.)
- PC -> ROM -> IR -> Decode -> Execute pipeline
- Cluster experiments (2x2 CPU grids with mailbox communication)

**If you need this functionality:**
- For heterogeneous tile circuits: Use `GpuSparseContext` in `cuda_tiles.rs` (73 Gtiles/sec)
- For homogeneous grids: Use `PackedTileGrid` in `cuda_tiles.rs` (115 Ttiles/sec)
- For combinatorial optimization: Use Ising/QUBO path in `cuda_tiles.rs`

## examples/ (archived with tile_cpu)

CPU simulation examples that depended on the tile_cpu module.

## docs/ (archived with tile_cpu)

`TILE_CPU_ARCHITECTURE.md` - Design document for the tile-based CPU.
