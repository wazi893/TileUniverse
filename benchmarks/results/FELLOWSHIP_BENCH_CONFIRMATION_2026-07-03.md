# Fellowship Benchmark Confirmation Sweep - 2026-07-03

Purpose: live-confirm the fellowship packet's packed cellular, WMMA quantum, and SIMT parity numbers against the current repo and hardware. Treat these as the current quotable numbers unless a newer sweep supersedes them.

## Environment

- Date/time: 2026-07-03, America/New_York
- Repo HEAD before documentation edits: `4eaca9b`
- GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM
- NVIDIA-SMI: `610.47`; CUDA UMD: `13.3`
- CUDA toolkit: `13.1` (`nvcc V13.1.80`)
- Rust/Cargo: `rustc 1.92.0`, `cargo 1.92.0`
- Pre-sweep GPU state: WDDM desktop load, ~1.0 GiB VRAM in use, ~10% GPU util, 42C, P8, 63.5 W / 575 W

## Summary

| Track | Previous fellowship quote | Current confirmed output | Status |
|---|---:|---:|---|
| Packed cellular, Register V3 full ladder | 115.61T tiles/s | 96.51T tiles/s | Drift down |
| Packed cellular, Register V3 CLI single variant | 115.61T tiles/s | 97.07T tiles/s, 485.3x vs repo u64 baseline | Drift down |
| Quantum PureMMA peak | 15.8 TCOPS @ 24q | 13.13 TCOPS @ 26q | Drift down |
| Quantum PureMMA 32q | 15.4 TCOPS | 12.84 TCOPS | Drift down |
| Honest quantum CUDA raw sanity | not the headline | 6.0316 TCOPS, no fusion multiplier | Confirmed sanity number |
| SIMT 8-lane fabric | 47,354 cyc/s | 53,358 cyc/s warmed; byte-identical oracle | Confirmed / faster |
| SIMT GPU lane-packed settle | 37.9G op-evals/s | 31.8G op-evals/s @ K=16,384; byte-identical oracle | Drift down |

## Packed Cellular

Command:

```powershell
cargo run --release --features cuda,perf-bench --example packed_register_benchmark -- --width 32768 --height 32768 --depth 2000 --steps 20000 --warmup 2000
```

Warmed full-ladder output:

| Variant | Throughput | Elapsed |
|---|---:|---:|
| Shuffle baseline | 23.21T tiles/s | 0.925 s |
| Register V1, 16 rows | 47.21T tiles/s | 0.455 s |
| Register V2, 8 rows + coop | 60.68T tiles/s | 0.354 s |
| Register V3, 32 rows | 96.51T tiles/s | 0.223 s |

Single-variant CLI confirmation:

```powershell
cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --width 32768 --height 32768 --depth 2000 --ticks 20000 --warmup 2000 -r3
```

Key output:

```text
THROUGHPUT: 97.07 T tiles/sec
tiles_per_sec: 9.707e13
improvement_vs_u64: 485.3
logic_eval_ops_per_sec: 97067725447441.55
```

Conclusion: the old 115T/115.61T headline did not reconfirm. Current quote should be "97.07T packed tile-evals/s on RTX 5090" or "96.51T in the full variant ladder"; pair it with the repo CLI's 485.3x vs its 200B u64-per-tile baseline.

Shortened historical-doc command probe:

```powershell
cargo run --release --features cuda,perf-bench --bin bench_engine -- --mode packed --register-v3 --width 32768 --height 32768
```

This command omits `--depth`, `--ticks`, and `--warmup`, so the current `bench_engine` defaults apply:
`depth=100`, `total_steps=5000`, `warmup=256`, `kernel_launches=50`. Local exact-command probes did
not reproduce an external-agent-reported 109.8-112.1T cluster. Results here were 92.95-94.28T over six
initial runs, best 96.34T after a long prewarm/P1 clock ramp, and 92.68-94.55T from direct executable
controls. Checksums stayed deterministic: `world_checksum: 0xe973af9a84882837`,
`logic_checksum: 0xa07a1f07d9845559`.

## Quantum WMMA / PureMMA

Command:

```powershell
cargo run --release --features cuda,perf-bench --example rtx5090_benchmark
```

PureMMA output:

| Qubits | VRAM Used | ILP TCOPS | PureMMA TCOPS | Speedup |
|---:|---:|---:|---:|---:|
| 12 | 32 MB | 6.17 | 12.27 | 1.99x |
| 16 | 128 MB | 6.49 | 13.00 | 2.00x |
| 20 | 512 MB | 6.54 | 13.00 | 1.99x |
| 24 | 2.0 GB | 6.50 | 13.12 | 2.02x |
| 26 | 4.0 GB | 6.50 | 13.13 | 2.02x |
| 28 | 4.0 GB | 6.52 | 13.07 | 2.01x |
| 30 | 8.0 GB | 6.34 | 13.07 | 2.06x |
| 31 | 16.0 GB | 5.77 | 12.98 | 2.25x |
| 32 | 32.0 GB | 3.93 | 12.84 | 3.27x |

Key output:

```text
PEAK PERFORMANCE: 13.13 TCOPS at 26 qubits (PureMMA)
Average PureMMA vs ILP speedup: 2.18x
```

Honest raw sanity command:

```powershell
cargo run --release --features cuda,perf-bench --example honest_quantum_bench
```

Key output:

```text
CPU Quantum Substrate (12 qubits, AVX2): 0.007988 TCOPS
CUDA GPU HONEST SUMMARY: 6.0316 TCOPS (RAW, no fusion)
```

Conclusion: use 13.13 TCOPS as the current RTX 5090 PureMMA headline. Keep the 6.0316 TCOPS figure only when explicitly discussing the helper's ILP raw/no-fusion sanity path.

## SIMT Parity

Fabric command:

```powershell
cargo run --release --example v2_simt_fabric_bench -- 8
```

Warmed key output:

```text
scalar reference: 154 cycles, golden hash 0xed6184d8f176640d
C fabric ORACLE PASSED: all 8 lanes byte-identical to scalar golden
C fabric lockstep: 8 lanes x 154 cycles in 0.023 s = 53358 cyc/s (wide settles: 153, fallback: 0)
B amortized scalar: 15735 cyc/s
C fabric: 53358 cyc/s (3.39x vs B)
```

GPU settle command:

```powershell
cargo run --release --features cuda --example v2_simt_gpu_bench
```

Warmed output:

| Lanes | Iters | Wall ms | us/iter | ns/iter/lane | G op-evals/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 2000 | 2645.1 | 1322.54 | 5166.18 | 1.1 |
| 1024 | 2000 | 2713.2 | 1356.62 | 1324.82 | 4.1 |
| 4096 | 2000 | 5277.2 | 2638.59 | 644.19 | 8.5 |
| 16384 | 2000 | 5636.8 | 2818.38 | 172.02 | 31.8 |

Oracle output:

```text
oracle PASSED: 5693 tiles x 256 lanes byte-identical to scalar propagate_compact
CPU kernel reference: 1.55 us/iter/lane @ 16 (= 3.55 G op-evals/s aggregate peak)
```

Conclusion: correctness still confirms. Current SIMT quote should be "8 lanes byte-identical at 53,358 cyc/s; GPU settle 31.8G op-evals/s at K=16,384, about 9.0x the CPU aggregate peak."
