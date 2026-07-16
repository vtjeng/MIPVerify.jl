# PR #208 — certified LP tightening: WK17a LP paired benchmark

PR #208 certifies LP tightening bounds from row duals — a correctness fix that regresses runtime.
Paired before/after benchmark, analyzed **retroactively** (the paired-analysis tooling post-dates
the merge). Raw per-sample data and dependency snapshots are in `baseline/` and `candidate/`.

- **Baseline** `master` `1bb2f9d82a762ac5f90beca10c7493d2e13d42ea`
- **Candidate** `bugfix/certified-lp-bounds` `f8e02c2c6e947b1b720b89ea9d25a8ba46387380`
- `--samples 1:500 --tightening lp --main-time-limit 120 --norm-order Inf --log-level warn`
- Julia 1.12.6, single thread, sequential runs, identical dependency snapshot
  (`803cbbe702d8ffe785cf653abcba05add104cf30a4c4e79df2d187b1bcbbbed9`); solver HiGHS. Local WSL2
  workstation — not comparable to the CI-hosted `benchmark-results` series.

---

## Summary

- **+29.8% aggregate wall clock**, but the typical sample is worse: **+45.8% per-sample median**. A few large samples pull the aggregate down.
- The cost is entirely in **build + bound tightening**: **+46.6% median, 100% of samples regressed**.
- **Main solve is effectively unaffected** (+0.7% median). Its small aggregate change is ~89% concentrated in the 10 biggest movers — noise.
- **Verdicts:** 1 of 500 semantic outcomes changed, in the candidate's favor (sample 246, `time_limit_unresolved → certified`). Solve status also churned on 7 samples near the 120 s limit (mostly `OPTIMAL ↔ TIME_LIMIT`, net 2 more time-outs) — a symptom of the runtime regression, not new verdicts.

## Detailed statistics

### Per-sample ratio distribution

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 1.09 | 1.30 | 1.39 | 1.47 | 1.56 | 1.65 | 2.61 | 0% | 100% |
| Main solve time | 492 | 0.32 | 0.91 | 0.96 | 1.01 | 1.06 | 1.14 | 2.40 | 40% | 48% |
| Total end-to-end time | 492 | 0.52 | 1.28 | 1.37 | 1.46 | 1.56 | 1.66 | 2.60 | 1% | 99% |

- `Ratio` = candidate ÷ baseline; < 1 = candidate faster.
- `Build` = constructing the MIP model; `tightening` = the LP bound-tightening pass; `main solve` = the final verification MIP.
- `Total` = `build` + `tightening` + `main solve`.

### Aggregate saving and concentration

| series | baseline | candidate | net saved | pooled | top-10 concentration |
|---|--:|--:|--:|--:|--:|
| Build + bound tightening | 2334 s | 3383 s | −1049 s | 1.45 | 4% |
| Main solve time | 1438 s | 1513 s | −74 s | 1.05 | 89% |
| Total end-to-end time | 3772 s | 4895 s | −1123 s | 1.30 | 25% |

- `Net saved` = baseline − candidate; positive = candidate cheaper.
- `Pooled` = candidate total ÷ baseline total.
- `Top-10 concentration` = share of total absolute per-sample change from the 10 biggest movers.

### Solve status and verdict flips

| status | master `1bb2f9d` | PR#208 `f8e02c2` |
|---|--:|--:|
| INFEASIBLE | 475 | 476 |
| OPTIMAL | 11 | 9 |
| SKIPPED_PREDICTED_IN_TARGETED | 8 | 8 |
| TIME_LIMIT | 6 | 7 |

7 samples changed solve status (before → after); only sample 246 also changed its semantic outcome.

| sample | solve status | semantic outcome |
|--:|---|---|
| 150 | `TIME_LIMIT` → `OPTIMAL` | — |
| 242 | `OPTIMAL` → `TIME_LIMIT` | — |
| 246 | `TIME_LIMIT` → `INFEASIBLE` | `time_limit_unresolved` → `certified_no_adversarial_example` |
| 445 | `OPTIMAL` → `TIME_LIMIT` | — |
| 446 | `OPTIMAL` → `TIME_LIMIT` | — |
| 449 | `TIME_LIMIT` → `OPTIMAL` | — |
| 496 | `OPTIMAL` → `TIME_LIMIT` | — |

### Plots

The tightening regression is systematic across the sample set, not outlier-driven; main solve sits on parity.

![ratio ECDF](plots/ratio_ecdf.png)

The extra wall-clock comes from tightening; main solve contributes no offsetting speedup, so the total tracks the tightening penalty.

![absolute runtime ECDF](plots/absolute_runtime_ecdf.png)

Tightening carries a near-constant multiplicative penalty; main solve shows almost no per-sample dispersion. The fix buys correctness (sample 246's certification), not runtime.

![magnitude scatter](plots/magnitude_scatter.png)

## Reproduce

Regenerate the plots and tables from the raw per-sample CSVs:

    analyze_pair --baseline baseline --candidate candidate --out .
