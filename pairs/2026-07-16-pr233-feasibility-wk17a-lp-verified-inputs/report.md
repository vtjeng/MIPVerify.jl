# PR #233 — feasibility objective: WK17a first-500 LP paired benchmark

This local paired benchmark compares two intentionally different solve goals. The `closest`
baseline minimizes adversarial distortion; the `feasibility` candidate asks for any adversarial
input within the fixed budget and independently verifies each numeric witness. This measures the
performance tradeoff of using the fixed-budget objective, not a same-objective code regression.

- Baseline: master `ee5fcfa267cd664799a90a7d22c7258546e9df1a`, `closest` objective.
- Candidate: PR #233 implementation commit `edfcbdfaa722dde5ef637f900b4845459ac68e8d`,
  `feasibility` objective.
- Arguments: `--samples 1:500 --tightening lp --main-time-limit 120 --norm-order Inf`.
- Julia 1.12.6, one Julia thread, identical dependency snapshot
  `041d5744a5121dc615cec747e2f55c74663a4ac563bffab69bb5a314939e1c84`.
- Both checkouts used HiGHS.jl 1.23.0 and HiGHS_jll 1.14.0+0. HiGHS.jl is the Julia optimizer
  interface; HiGHS_jll supplies the native HiGHS solver binary. Pinning both packages means the
  comparison uses the same solver interface and executable on each side.
- The runs used the same local workstation. The runner executed the baseline and then the
  candidate in separate fresh processes, about 53 seconds apart; they were sequential rather than
  interleaved.

Raw per-sample data, tightening rows, ReLU rows, metrics, and dependency snapshots are in
`baseline/` and `candidate/`. Generated statistics are in `improvement_stats.md` and
`improvement_stats.csv`.

## Sample accounting

Both runs processed 500 inputs. For 492, each run constructed the MIP, performed LP bound
tightening, and invoked the final verification solve. The other eight inputs were already
misclassified: their original input met the requested attack condition, so the common fast path
returned it without constructing a solver model.

Per-sample ratio distributions use the 492 model-built pairs. This exclusion is solely the eight
fast-path inputs; samples whose solve status or semantic outcome changed are not dropped. All seven
such samples are model-built and remain in the timing population.

The summed all-input end-to-end times are 2,139.591 s and 1,297.437 s. Removing the few
milliseconds spent on the eight fast paths gives 2,139.561 s and 1,297.386 s for the 492
model-built pairs, so both populations round to the same whole-second totals below.

## Timing summary

`total_time_seconds` is the end-to-end per-input measurement: model construction, LP bound
tightening, and the final MIP solve. It is not bound-tightening time alone.

| measure | population | `closest` | `feasibility` | candidate / baseline | change |
|---|---:|---:|---:|---:|---:|
| Whole-run elapsed | all 500 | 2,143.4 s | 1,300.5 s | 0.607 | −39.3% |
| Summed end-to-end time | 492 model-built | 2,139.6 s | 1,297.4 s | 0.606 | −39.4% |
| Non-main-solve time (`total − main solve`) | 492 model-built | 688.6 s | 675.2 s | 0.981 | −1.9% |
| Solver-reported main solve time | 492 model-built | 1,451.0 s | 622.2 s | 0.429 | −57.1% |
| Bound-solver calls | 492 model-built | 99,067 | 99,067 | 1.000 | 0.0% |

The non-main-solve complement includes model construction, LP bound tightening, and benchmark
bookkeeping; it is not the direct bound-tightening field. Direct instrumented bound-tightening
time was 337.5 s → 352.4 s (+4.4%). Neither stage-level measure is used for the headline total.

The pooled end-to-end ratio divides summed candidate time by summed baseline time, so expensive
inputs receive more weight. The median gives each input equal weight. Here the pooled ratio is
0.606 while the median per-sample ratio is 1.031: aggregate time fell 39.4%, but the typical
model-built input was 3.1% slower. The 10 inputs with the largest absolute end-to-end changes
account for 74% of total absolute movement.

Using a ±1% band around parity:

| per-sample end-to-end result | inputs | share |
|---|---:|---:|
| Improved | 186 | 37.8% |
| Within ±1% | 21 | 4.3% |
| Regressed | 285 | 57.9% |

The aggregate gain comes almost entirely from the final MIP solve and is concentrated in expensive
inputs. The combined non-main-solve portion remains near parity.

![Paired ratio distributions](plots/ratio_ecdf.png)

The absolute distribution shows the feasibility objective shortening much of the long main-solve
tail. Four candidate inputs still end at the 120-second limit.

![Absolute runtime distributions](plots/absolute_runtime_ecdf.png)

Most of the expensive baseline solves lie below the parity line. Samples 19 and 46 move the other
way: the baseline proves infeasibility while the candidate reaches the time limit.

![Paired runtime scatter](plots/magnitude_scatter.png)

## Witness audit

The feasibility run produced 22 numeric witnesses: 14 from solver-backed models and eight from the
original-input fast path. No proposed witness failed verification.

| candidate witness check | passed | failed |
|---|---:|---:|
| Numeric candidate available | 22 | — |
| Requested target and margin | 22 | 0 |
| Built-in perturbation-family constraints | 22 | 0 |
| Both independent checks | 22 | 0 |

The perturbation check validates shape, finite values, pixel range, and the L-infinity budget
against the original input. The schema-3 baseline predates numeric witness fields, so the table
applies directly only to the candidate.

## Solve status and semantic outcomes

| solve status | `closest` | `feasibility` |
|---|---:|---:|
| `INFEASIBLE` | 476 | 474 |
| `OPTIMAL` | 10 | 14 |
| `SKIPPED_PREDICTED_IN_TARGETED` | 8 | 8 |
| `TIME_LIMIT` | 6 | 4 |

Six samples changed solve status:

| transition | n | samples |
|---|---:|---|
| `INFEASIBLE` → `TIME_LIMIT` | 2 | 19, 46 |
| `TIME_LIMIT` → `OPTIMAL` | 4 | 150, 242, 446, 480 |

Three samples changed semantic outcome:

| transition | n | samples |
|---|---:|---|
| `certified_no_adversarial_example` → `time_limit_unresolved` | 2 | 19, 46 |
| `adversarial_example_found_or_best_known` → `time_limit_unresolved` | 1 | 212 |

For completeness, this is the per-input before → after audit across the union of both change sets:

| sample | solve status, `closest` → `feasibility` | semantic outcome, `closest` → `feasibility` |
|---:|---|---|
| 19 | `INFEASIBLE` → `TIME_LIMIT` | `certified_no_adversarial_example` → `time_limit_unresolved` |
| 46 | `INFEASIBLE` → `TIME_LIMIT` | `certified_no_adversarial_example` → `time_limit_unresolved` |
| 150 | `TIME_LIMIT` → `OPTIMAL` | `adversarial_example_found_or_best_known` → same |
| 212 | `TIME_LIMIT` → `TIME_LIMIT` | `adversarial_example_found_or_best_known` → `time_limit_unresolved` |
| 242 | `TIME_LIMIT` → `OPTIMAL` | `adversarial_example_found_or_best_known` → same |
| 446 | `TIME_LIMIT` → `OPTIMAL` | `adversarial_example_found_or_best_known` → same |
| 480 | `TIME_LIMIT` → `OPTIMAL` | `adversarial_example_found_or_best_known` → same |

Samples 150, 242, 446, and 480 already had incumbents in the `closest` run, so proving optimality
under `feasibility` did not change their semantic outcome. Sample 212 remains at `TIME_LIMIT`, but
only the baseline has an incumbent.

## Limitations

- This compares two objectives, so it measures the fixed-budget tradeoff between exact distortion
  and feasibility rather than a same-objective implementation regression.
- Each objective ran once, sequentially rather than interleaved. The pair does not estimate
  run-to-run variance.
- Every changed solve status or semantic outcome involves the 120-second limit on at least one
  side. Individual transitions may vary with solver search order and workstation timing near the
  cap.
- The benchmark exercises the repository-pinned HiGHS stack. It does not provide performance
  evidence for other optimizers.

## Reproduce

From a checkout containing PR #233's paired-run helper:

```sh
benchmarks/run_pair.sh \
  --base ee5fcfa267cd664799a90a7d22c7258546e9df1a \
  --candidate edfcbdfaa722dde5ef637f900b4845459ac68e8d \
  --out /tmp/mipv-pr233-feasibility-pair \
  --samples 1:500 \
  --tightening lp \
  --main-time-limit 120 \
  --candidate-objective feasibility \
  --base-label "closest master ee5fcfa" \
  --candidate-label "feasibility PR #233 edfcbdf"
```

The base objective flag is omitted because master `ee5fcfa` predates the objective selector and its
benchmark default is `closest`. The runner creates a worktree for each commit, develops that
checkout's local MIPVerify package, and installs the exact HiGHS versions listed above on both
sides before running them back to back.
