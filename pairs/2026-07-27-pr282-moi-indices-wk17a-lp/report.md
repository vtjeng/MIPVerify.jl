# Performance report: WK17a, LP tightening, 500 samples

#282 reads each affine row of the LP bound certificate through its `MOI.ConstraintIndex`, which removes the `JuMP.ConstraintRef` vector that `JuMP.all_constraints` allocated on every bound solve, and lowers formulation time while leaving every solved model unchanged.

Samples `1:500` of the MNIST test set are verified against `MNIST.WK17a_linf0.1_authors` on the baseline and candidate commits under identical non-objective settings. Both sides use the `feasibility` adversarial-example objective, which searches for any verified witness inside the fixed budget. The ratio distributions, scatter plots, and outcome-flip tables compare each sample's candidate run with its own baseline run; the absolute-runtime distributions summarize each side separately. The `pairs/2026-07-27-pr282-moi-indices-wk17a-lp/` folder on the `benchmark-reports` branch holds the raw per-sample CSVs for both sides, the per-layer ReLU CSVs, the dependency snapshots, the statistics tables, and the plots.

- **Baseline** `master` `4f1ed43` (this PR's merge base); **candidate** `9c12298` (benchmarked commit; `src/` and `test/` byte-identical to PR head `bfca48e`).

| run                               | adversarial-example objective |
| --------------------------------- | ----------------------------- |
| master 4f1ed43 [feasibility]      | `feasibility`                 |
| moi-indices 9c12298 [feasibility] | `feasibility`                 |

- Command:
  `benchmarks/run_pair.sh --base 4f1ed43 --candidate 9c12298 --out <out> --samples 1:500 --tightening lp --main-time-limit 120 --norm-order Inf --base-objective feasibility --candidate-objective feasibility`.
  Julia `1.12.6`, single-threaded; HiGHS (an open-source LP/MIP solver) for all solves; sequential runs on a local WSL2 workstation, identical dependency snapshots on both sides (HiGHS 1.23.0, HiGHS_jll 1.14.0+0). Absolute times are not comparable to the CI-hosted `benchmark-results` series. Analyzed with `uv run analyze_pair.py --baseline <out>/base --candidate <out>/candidate --out <out>/analysis --baseline-label "master 4f1ed43" --candidate-label "moi-indices 9c12298"` from `benchmarks/analysis`.

---

## Summary

- Build + bound tightening, the phase this change targets, fell `608 s → 584 s`, a median ratio of 0.88 over 492 paired samples with 74% improved and 25% regressed. The saving is 12 ± 5 percent; the ± 5 comes from a master-against-master control run of this benchmark measuring a median ratio of 0.95, which fixes the size of the benchmark noise and leaves its direction unknown.
- Total end-to-end time fell `1048 s → 958 s`, pooled ratio 0.91 against a median of 0.89.
- Main solve time, which this change cannot alter, has a median of 1.02. Ten samples hold 94% of its absolute change, so its `440 s → 374 s` aggregate movement measures noise.
- Bound solver calls stay at 99,067 on both sides and match on every one of the 500 samples, so the change alters how the certificate reads each row while the solver receives the same work.
- One sample changed solve status and two changed semantic outcome, both in the candidate's favour. No verdict regressed.
- The formulation pooled ratio of 0.96 sits above the median of 0.88 without a small set of samples explaining the gap: top-10 concentration is 12%, and the ten largest movers sum to +5.4 s against a net of −25.7 s. An earlier pair of a different implementation of the same optimisation measured a formulation pooled ratio of 0.905 against a median of 0.898 on the same benchmark, with baseline formulation totals of 644 s and 597 s across the two sessions on identical master code. The pooled statistic moves between sessions by more than the difference it is being read for.

## Detailed statistics

### Plots

The build-phase curve sits left of the diagonal across most of its range, while the main-solve curve tracks it closely, which is the shape expected from a change confined to formulation.

![Paired ratio distributions](plots/ratio_ecdf.png)

The two sides overlay closely in absolute runtime, with the candidate slightly left in the body of the distribution.

![Absolute runtime distributions](plots/absolute_runtime_ecdf.png)

Most points fall below the `y = x` diagonal in the build phase; the few large excursions are main-solve samples at the time limit.

![Paired runtime scatter](plots/magnitude_scatter.png)

Bound-call counts land exactly on the diagonal, one point per sample.

![Absolute bound-call distributions](plots/absolute_calls_ecdf.png)

![Paired bound-call scatter](plots/calls_scatter.png)

### Per-sample ratio distribution

| series                   |   n |  min |  p10 |  p25 | median |  p75 |  p90 |  max | improved | regressed |
| ------------------------ | --: | ---: | ---: | ---: | -----: | ---: | ---: | ---: | -------: | --------: |
| Build + bound tightening | 492 | 0.40 | 0.77 | 0.84 |   0.88 | 1.02 | 1.31 | 2.46 |      74% |       25% |
| Main solve time          | 492 | 0.10 | 0.92 | 0.97 |   1.02 | 1.10 | 1.38 | 3.80 |      34% |       55% |
| Total end-to-end time    | 492 | 0.11 | 0.76 | 0.84 |   0.89 | 1.04 | 1.32 | 3.12 |      73% |       27% |
| Bound solver calls       | 492 | 1.00 | 1.00 | 1.00 |   1.00 | 1.00 | 1.00 | 1.00 |       0% |        0% |

- `ratio` = candidate ÷ baseline, < 1 = candidate faster; `improved` counts ratio below 0.99, `regressed` counts ratio above 1.01, samples within the ±1% band count as unchanged.
- `build` = constructing the MIP model; `tightening` = the `lp` bound-tightening pass; `main solve` = the final verification MIP.
- `total` = `build` + `tightening` + `main solve`.
- `bound solver calls` = count of HiGHS bound-tightening solves.
- `n` = eligible paired inputs for that series after the C3 filters; use each row's emitted value.

### Aggregate saving and concentration

| series                   |    baseline |   candidate | net saved | pooled ratio | top-10 concentration |
| ------------------------ | ----------: | ----------: | --------: | -----------: | -------------------: |
| Build + bound tightening |       608 s |       584 s |    +24 s |         0.96 |                  12% |
| Main solve time          |       440 s |       374 s |    +66 s |         0.85 |                  94% |
| Total end-to-end time    |      1048 s |       958 s |    +90 s |         0.91 |                  62% |
| Bound solver calls       | 99067 calls | 99067 calls |  +0 calls |         1.00 |                   0% |

- `net saved` = baseline − candidate total; positive = candidate cheaper.
- `pooled ratio` = candidate total ÷ baseline total (aggregate counterpart to the per-sample median).
- `top-10 concentration` = the 10 samples with the largest absolute change account for this share of the total absolute per-sample change (0–100%; higher = a few samples dominate). Bound solver calls has zero total absolute per-sample movement, so its concentration is reported as 0%.

### Solve status and verdict flips

| status                        | master `4f1ed43` [feasibility] | moi-indices `9c12298` [feasibility] |
| ----------------------------- | -----------------------------: | ----------------------------------: |
| INFEASIBLE                    |                            475 |                                 476 |
| OPTIMAL                       |                             15 |                                  15 |
| SKIPPED_PREDICTED_IN_TARGETED |                              8 |                                   8 |
| TIME_LIMIT                    |                              2 |                                   1 |

One sample changed solve status and two changed semantic outcome; the status change is also one of the two outcome changes, so the sets overlap in sample 19.

Solve status:

| transition                  |   n | samples |
| --------------------------- | --: | ------- |
| `TIME_LIMIT` → `INFEASIBLE` |   1 | 19      |

Semantic outcome:

| transition                                                               |   n | samples |
| ------------------------------------------------------------------------ | --: | ------- |
| `time_limit_unresolved` → `certified_no_adversarial_example`             |   1 | 19      |
| `witness_verification_failed` → `adversarial_example_found_or_best_known` |   1 | 212     |

#### Model and outcome audit

The paired raw rows have identical values for:

- dependency snapshot and benchmark arguments;
- model shape: `num_variables`, `num_binary_variables`, `num_structural_constraints`, `num_total_constraints`;
- ReLU classification counts: `relu_total_count`, `relu_stable_count`, `relu_unstable_count`, `relu_zero_output_count`, `relu_linear_in_input_count`, `relu_constant_output_count`, on all 500 samples and on all 1,476 per-layer rows;
- bound accounting: `bound_request_count`, `bound_solver_call_count`, `bound_optimal_count`, `bound_interval_arithmetic_count`;
- `bound_barrier_iterations`, `bound_node_count`, and `objective_bound`.

Objective agreement: 487 inputs carried an objective value on both sides, and all 487 agree within `1e-6`. No optimal/optimal pair disagrees.

## Reproduce

Analyzer command, run from `benchmarks/analysis`:

`uv run analyze_pair.py --baseline <out>/base --candidate <out>/candidate --out <out>/analysis --baseline-label "master 4f1ed43" --candidate-label "moi-indices 9c12298"`
