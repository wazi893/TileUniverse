# AlphaFabric SA Route Frontier Benchmark

Run date: 2026-06-19

Command:

```powershell
cargo run --release --example sa_route_probe
```

Configuration: `cases=6:3,8:3,8:4`, `iters=30000`, `seed=0x000000005EEDA1F0`,
`route=no_crossings,max_z=3`, `row_major=claim-baseline`.

Host/toolchain: AMD Ryzen 9 9950X3D, Windows x86_64, `rustc 1.92.0`,
`cargo 1.92.0`, repo HEAD `5dc8724` before local benchmark/report edits.

## Citation Table

| madd width | gates | measured halos | row-major routed frontier | SA verified frontier | claim-ready observation |
|---:|---:|---|---|---|---|
| 6 | 144 | 3 | none in sweep | halo 3 (phys=yes, wire=5,733, vias=318) | row-major fails at halo 3; SA routes and physically verifies at halo 3 |
| 8 | 270 | 3,4 | none in sweep | halo 4 (phys=yes, wire=17,515, vias=803) | SA fails at halo 3 and routes+verifies at halo 4 |

## Raw Rows

| width | gates | inputs | halo | row route | row phys | row ms | SA route | SA phys | SA ms | SA HPWL delta | SA best HPWL | SA wire | accepted |
|---:|---:|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 6 | 144 | 18 | 3 | no | n/a | 10,287 | yes | yes | 22,822 | 44.6% | 3,336 | 5,733 | 951 |
| 8 | 270 | 24 | 3 | no | n/a | 28,732 | no | no | 49,523 | 32.6% | 9,368 | 0 | 774 |
| 8 | 270 | 24 | 4 | skipped | n/a | n/a | yes | yes | 116,321 | 33.8% | 11,500 | 17,515 | 569 |

Notes:

- `phys=yes` is `PlacementEnv::verify_physical`: route, export to the tile fabric, then compare tile evaluation against the AIG on the repository's deterministic verifier vector set.
- The default benchmark skips row-major routing for width 8 halo 4 because the width-8 claim only needs SA fail at halo 3 and SA success at halo 4. Use `--row-major-all` or `--full` for wider baseline sweeps.
- Built-in claim checks passed.
