# PR #233 — input-verified verdict-only mode: WK17a LP paired benchmark

This benchmark compares exact distortion on master with the fixed-budget verdict path in PR #233.
The solve goals differ intentionally: the baseline minimizes adversarial distortion, while the
candidate asks whether a feasible adversarial example exists and independently verifies every
numeric witness. Raw per-sample data, tightening rows, ReLU rows, metrics, and dependency snapshots
are in `baseline/` and `candidate/`.

- Baseline: master `8a455e2756a0d45e224bb1da95f2de8dc2ba3df4`, exact-distortion mode.
- Measured candidate: PR #233 `feacdc84fad376edefc5b273b014e5ea7d5eac80`, verdict-only mode.
- Arguments: `--samples 1:500 --tightening lp --main-time-limit 120 --norm-order Inf`.
- Julia 1.12.6, HiGHS, one Julia thread, identical dependency snapshot
  `1cfef4c977ff08219a888aa479cb94eea0b3dbc16f654d1fc0a09167c8f1c74a`.
- Both runs used the same local workstation. The reusable baseline completed about nine and a half
  hours before the candidate rather than running as an interleaved control.

## Summary

- Whole-run elapsed time fell 43.4%, from 2,434.8 s to 1,379.1 s. Summed per-sample time fell
  43.5%, from 2,431.7 s to 1,374.9 s.
- Summed main-solver time fell 69.9%, from 1,549.0 s to 465.9 s. The 10 largest per-sample main-solve
  changes account for 79% of absolute movement, so the aggregate gain is concentrated in expensive
  inputs.
- Across 492 modeled pairs, the median total-time ratio was 0.94. Using a ±1% band, 62% improved
  and 32% regressed. The pooled ratio was 0.57 because verdict-only shortens several long solves.
- Bound-solver calls were identical at 99,067. Build and bound-tightening time increased 3.0% in
  aggregate, but its median ratio was 0.95. That stage is unchanged in substance, and the
  non-interleaved setup cannot attribute the aggregate difference to verdict-only mode.
- Unresolved timeouts fell from 3 to 2. Three semantic outcomes changed near the 120 s limit: one
  exact robustness certificate became unresolved, one unresolved input gained a verified witness,
  and one unresolved input gained a robustness certificate.
- The candidate had 15 solver-backed witnesses and eight fast-path witnesses. All 23 passed the
  independent target and perturbation-family checks; there were no available-but-unverified
  witnesses.

## Aggregate results

| series | baseline | candidate | candidate / baseline | change |
|---|--:|--:|--:|--:|
| Whole-run elapsed | 2,434.8 s | 1,379.1 s | 0.566 | -43.4% |
| Summed end-to-end time | 2,431.7 s | 1,374.9 s | 0.565 | -43.5% |
| Build + bound tightening | 882.6 s | 909.0 s | 1.030 | +3.0% |
| Main solve time | 1,549.0 s | 465.9 s | 0.301 | -69.9% |
| Bound-solver calls | 99,067 | 99,067 | 1.000 | 0.0% |

## Per-sample distribution

Ratios are candidate divided by baseline; values below 1 are faster. The analyzer excludes the
eight already-misclassified fast-path inputs, leaving 492 modeled pairs. `improved` and `regressed`
use a ±1% band.

| series | n | min | p10 | p25 | median | p75 | p90 | max | improved | regressed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Build + bound tightening | 492 | 0.58 | 0.81 | 0.88 | 0.95 | 1.05 | 1.19 | 7.47 | 61% | 34% |
| Main solve time | 492 | 0.01 | 0.68 | 0.79 | 0.90 | 1.01 | 1.14 | 8.81 | 73% | 25% |
| Total end-to-end time | 492 | 0.02 | 0.77 | 0.87 | 0.94 | 1.04 | 1.17 | 7.74 | 62% | 32% |
| Bound-solver calls | 492 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0% | 0% |

The relative distribution shows the largest shift in main-solve time. Build and tightening remain
centered near parity.

![Paired ratio distributions](plots/ratio_ecdf.png)

The absolute distribution shows verdict-only removing much of the long main-solve tail. Two
candidate inputs still reach the 120 s limit.

![Absolute runtime distributions](plots/absolute_runtime_ecdf.png)

Most expensive main solves lie below the parity line. Sample 19 is the material opposite case: the
exact baseline proved infeasibility, while the candidate reached the time limit.

![Paired runtime scatter](plots/magnitude_scatter.png)

## Witness and outcome audit

| candidate witness check | passed | failed |
|---|--:|--:|
| Numeric candidate available | 23 | — |
| Requested target and margin | 23 | 0 |
| Built-in perturbation-family constraints | 23 | 0 |
| Combined witness verdict | 23 | 0 |

The perturbation check validates shape, finiteness, pixel range, and the L-infinity budget against
the original input. The schema-3 baseline predates numeric witness fields, so the table applies
only to the candidate.

| status | exact distortion | verdict only |
|---|--:|--:|
| `INFEASIBLE` | 475 | 475 |
| `OPTIMAL` | 10 | 15 |
| `SKIPPED_PREDICTED_IN_TARGETED` | 8 | 8 |
| `TIME_LIMIT` | 7 | 2 |

Solve-status changes:

| transition | n | samples |
|---|--:|---|
| `TIME_LIMIT` → `OPTIMAL` | 5 | 212, 242, 321, 446, 480 |
| `INFEASIBLE` → `TIME_LIMIT` | 1 | 19 |
| `TIME_LIMIT` → `INFEASIBLE` | 1 | 407 |

Semantic-outcome changes:

| transition | n | samples |
|---|--:|---|
| robustness certificate → unresolved time limit | 1 | 19 |
| unresolved time limit → verified adversarial witness | 1 | 212 |
| unresolved time limit → robustness certificate | 1 | 407 |

The other four `TIME_LIMIT` → `OPTIMAL` changes already had incumbents in the exact run, so their
adversarial verdict did not change. Objective values are not compared because verdict-only solves a
feasibility problem rather than minimizing distortion.

## Limitations

- This is a cross-mode performance comparison, not a same-objective code regression benchmark.
- The same-day baseline was reused instead of rerun immediately before the candidate. Formulation
  timing differences and individual results near the 120 s limit include fresh-process and
  workstation variation.
- The 500-sample pair is one run per side. It measures the long-tail effect but does not estimate
  run-to-run variance.
- Gurobi was unavailable locally. The benchmark exercises HiGHS. Verdict-only behavior does not
  depend on `SOLUTION_LIMIT` or `OBJECTIVE_LIMIT`; synthetic tests cover those termination statuses.
- The baseline schema does not contain enough numeric witness data to replay its candidates. The
  zero-failure witness claim applies directly to the schema-5 candidate.

## Reproduce

From a checkout containing PR #233's paired-run helper:

```sh
benchmarks/run_pair.sh \
  --base 8a455e2756a0d45e224bb1da95f2de8dc2ba3df4 \
  --candidate feacdc84fad376edefc5b273b014e5ea7d5eac80 \
  --out /tmp/mipv-pr233-verified-input-pair \
  --samples 1:500 \
  --tightening lp \
  --main-time-limit 120 \
  --candidate-mode verdict-only \
  --base-label "exact distortion 8a455e2" \
  --candidate-label "verdict only PR #233"
```

The base mode flag is omitted because commit `8a455e2` predates the mode argument and its benchmark
default is exact distortion.
