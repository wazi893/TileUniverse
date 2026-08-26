# Scheduling Optimizer Safety Frontier

Measured on branch `feat/sched-optimizer`, 2026-06-23; extended 2026-06-25 with the derived
safe-operating-point (knee) line and the reproducible frontier CSV.

Command evidence:

```text
cargo run --release --example sched_eval
cargo test sched_eval --lib
```

Focused validation passed: `17 passed; 0 failed; 3563 filtered out`.

## What Changed

The scheduler now reports a second workload, `victim_contention`, alongside the original
`balanced_backfill` case. The point of the new workload is to prevent an overbroad claim:
`balanced_backfill` shows that a smarter placement can sometimes remove adversary/victim
co-location at no throughput cost, while `victim_contention` shows the real tradeoff when
there is not enough trusted work to backfill idle capacity.

Both workloads use real compiled tile-CPU programs and golden cycle costs. The safety metric is
not a heuristic: same-rack adversary/victim overlap windows are replayed through the
`control_eval` thermal covert-channel oracle to measure leaked bits and monitor detection.

## Exact Results

Cluster: 4 cores, 2 racks of 2 cores. One cooling zone per rack.

### Balanced Backfill

Workload:

| job | program | cycles | tenant |
|---:|---|---:|---|
| 0 | fib_recursive | 4255 | trusted |
| 1 | gcd_subtract | 1313 | trusted |
| 2 | sum_squares | 404 | trusted |
| 3 | fact_iterative | 154 | trusted |
| 4 | array_sum_squares | 661 | trusted |
| 5 | print_large | 4538 | victim |
| 6 | print_count | 1807 | victim |
| 7 | sum_loop | 674 | adversary |

Policy results:

| policy | makespan cycles | util | exploitable windows | bits leaked | caught |
|---|---:|---:|---:|---:|---:|
| greedy | 4538 | 76% | 1 | 18 | 1 |
| isolation_aware | 6620 | 52% | 0 | 0 | 0 |
| secure_packing | 6062 | 56% | 0 | 0 | 0 |
| optimized | 4538 | 76% | 0 | 0 | 0 |

Safety price against greedy/leaky baseline:

| safe point | makespan | delta cycles | exact delta | example-reported price |
|---|---:|---:|---:|---:|
| naive isolation | 6620 | +2082 | +45.88% | +45% |
| secure packing | 6062 | +1524 | +33.58% | +33% |
| search-optimized | 4538 | +0 | +0.00% | +0% |

Search frontier:

| lambda | makespan | bits leaked | exploitable windows |
|---:|---:|---:|---:|
| 0 | 4538 | 18 | 1 |
| 1 | 4538 | 0 | 0 |
| 5 | 4538 | 0 | 0 |
| 20 | 4538 | 0 | 0 |
| 100 | 4538 | 0 | 0 |
| 1000 | 4538 | 0 | 0 |

Derived safe operating point (knee): makespan 4538 at +0.00% over greedy, at lambda 1 — the
cheapest safety weight that already closes the channel, 0 bits leaked. Selected by
`safe_operating_point` as the minimum-makespan zero-leak point, so the price is computed from the
frontier rather than hand-entered.

Frontier CSV (`frontier_csv`, the data behind the plot):

```text
lambda,makespan,utilization_pct,bits_leaked,exploitable_windows
0,4538,76,18,1
1,4538,76,0,0
5,4538,76,0,0
20,4538,76,0,0
100,4538,76,0,0
1000,4538,76,0,0
```

Greedy exploit evidence: adversary job 7, `sum_loop`, overlaps victim job 6, `print_count`, in
rack 1 for 494 cycles. The oracle reports 18 leaked secret bits and monitor detection.

Interpretation: this is the "free safety" case. Trusted jobs can backfill around the isolated
adversary, so the optimizer reaches zero leakage without increasing makespan.

### Victim Contention

Workload:

| job | program | cycles | tenant |
|---:|---|---:|---|
| 0 | print_large | 4538 | victim |
| 1 | fib_recursive | 4255 | victim |
| 2 | print_count | 1807 | victim |
| 3 | fib_recursive | 4255 | adversary |

Policy results:

| policy | makespan cycles | util | exploitable windows | bits leaked | caught |
|---|---:|---:|---:|---:|---:|
| greedy | 4538 | 81% | 1 | 18 | 1 |
| isolation_aware | 6062 | 61% | 0 | 0 | 0 |
| secure_packing | 6062 | 61% | 0 | 0 | 0 |
| optimized | 6062 | 61% | 0 | 0 | 0 |

Safety price against greedy/leaky baseline:

| safe point | makespan | delta cycles | exact delta | example-reported price |
|---|---:|---:|---:|---:|
| naive isolation | 6062 | +1524 | +33.58% | +33% |
| secure packing | 6062 | +1524 | +33.58% | +33% |
| search-optimized | 6062 | +1524 | +33.58% | +33% |

Search frontier:

| lambda | makespan | bits leaked | exploitable windows |
|---:|---:|---:|---:|
| 0 | 4538 | 18 | 1 |
| 1 | 6062 | 0 | 0 |
| 5 | 6062 | 0 | 0 |
| 20 | 6062 | 0 | 0 |
| 100 | 6062 | 0 | 0 |
| 1000 | 6062 | 0 | 0 |

Derived safe operating point (knee): makespan 6062 at +33.58% over greedy, at lambda 1, 0 bits
leaked. Unlike the balanced case the knee sits strictly above greedy — the safety price is real,
not an artifact of dumb scheduling.

Frontier CSV (`frontier_csv`, the data behind the plot):

```text
lambda,makespan,utilization_pct,bits_leaked,exploitable_windows
0,4538,81,18,1
1,6062,61,0,0
5,6062,61,0,0
20,6062,61,0,0
100,6062,61,0,0
1000,6062,61,0,0
```

Greedy exploit evidence: adversary job 3, `fib_recursive`, overlaps victim job 2, `print_count`,
in rack 1 for 1807 cycles. The oracle reports 18 leaked secret bits and monitor detection.

Interpretation: this is the visible safety-price case. There is no trusted backfill available, so
zero leakage requires separating adversary and victim work across cooling zones. The measured
safety price is +1524 cycles, or +33.58% makespan relative to greedy.

## Validation Notes

- Cost audit matched live ISS execution for every scheduler input. For each program,
  `modeled_cyc = live_instr + 1` equaled the scheduler's assumed cycle count.
- Live cost-audit rows from the release example:

| program | assumed cycles | live retired instructions | modeled cycles | match |
|---|---:|---:|---:|---|
| fib_recursive | 4255 | 4254 | 4255 | ok |
| gcd_subtract | 1313 | 1312 | 1313 | ok |
| sum_squares | 404 | 403 | 404 | ok |
| fact_iterative | 154 | 153 | 154 | ok |
| array_sum_squares | 661 | 660 | 661 | ok |
| print_large | 4538 | 4537 | 4538 | ok |
| print_count | 1807 | 1806 | 1807 | ok |
| sum_loop | 674 | 673 | 674 | ok |

- `sched_eval::tests::contention_workload_has_real_safety_price` asserts that greedy leaks,
  high-lambda search eliminates leakage, and the zero-leak contention schedule has a larger
  makespan than greedy.
- `sched_eval::tests::contention_costs_trace_to_live_execution` re-measures the contention
  workload live and verifies every assumed cost.
- `sched_eval::tests::optimizer_frontier_is_monotone_and_deterministic` checks deterministic
  seeded optimization and the expected safety-weight behavior.
- The frontier is also exposed as data: `pareto_frontier` returns the lambda sweep as
  `FrontierPoint`s, `safe_operating_point` selects the knee (minimum-makespan zero-leak point),
  and `frontier_csv` emits the tables above. `frontier_report` prints the derived knee line.
  Covered by `pareto_frontier_matches_sweep_and_picks_free_knee`,
  `safe_operating_point_shows_contention_price`, `safe_operating_point_is_none_when_everything_leaks`,
  `frontier_csv_is_wellformed_and_stable`, and the rendered reports are pinned bit-for-bit by
  `frontier_reports_replay_bit_for_bit` (both workloads).
