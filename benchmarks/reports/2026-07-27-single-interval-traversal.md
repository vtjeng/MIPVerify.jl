# Single interval traversal benchmark report

- Date: 2026-07-27
- Experimental code:
  [`perf/single-interval-traversal`](https://github.com/vtjeng/MIPVerify.jl/tree/perf/single-interval-traversal)
- Implementation commit:
  [`56d8d4a6b66192baee1431f7eca28243c70b60d8`](https://github.com/vtjeng/MIPVerify.jl/commit/56d8d4a6b66192baee1431f7eca28243c70b60d8)
- Baseline commit:
  [`4f1ed43d8d1ec600e583abc5dd9a7ccb85e3c4ac`](https://github.com/vtjeng/MIPVerify.jl/commit/4f1ed43d8d1ec600e583abc5dd9a7ccb85e3c4ac)
- Related issues: [#267](https://github.com/vtjeng/MIPVerify.jl/issues/267),
  [#276](https://github.com/vtjeng/MIPVerify.jl/issues/276)

## Research idea

Formulating a WK17a verification problem spends more time in Julia than in the solver. Profiling
attributed a share of that time to computing interval bounds on the affine expressions that feed
each ReLU.

`lower_bound(e)` and `upper_bound(e)` in `src/vendor/ConditionalJuMP.jl` are each defined as a call
to `IntervalArithmetic.interval(e)` followed by taking one endpoint. That function walks every term
of the expression and produces both endpoints together, so `upper_bound` computes the lower endpoint
and discards it, and a subsequent `lower_bound` walks the whole expression again to recompute what
the first call already had.

The idea was to compute one interval per ReLU input and read both endpoints from it, removing the
second traversal at no additional cost. The estimate before measuring was a 3% to 6% reduction in
total benchmark wall clock.

## Conclusion

The change produced no measurable improvement. Its median per-sample ratio on Julia-only formulation
time was 1.001 against the shared baseline, and its total Julia-only time was 440.4 s against the
baseline's 435.1 s. The candidate was 1.2% slower in total, the opposite direction from the
predicted saving.

That difference is smaller than this benchmark's run-to-run variation. Three runs in this session
execute behaviorally identical code: the baseline, this change, and a repeat of the baseline. Their
ReLU classifications are byte-identical, so any difference between them is measurement noise. Their
Julia-only totals were 435.1 s, 440.4 s, and 412.1 s; the extremes differ by 6.8% of the smallest.
The measured effect of this change sits inside that spread.

The change is behavior preserving. The full test suite passed (883 pass, 1 pre-existing broken, 0
fail), and the per-layer ReLU classifications over all 492 modeled samples are byte-identical to the
baseline.

## Method

500 MNIST test samples, network `MNIST.WK17a_linf0.1_authors`, L-infinity radius 0.1, `feasibility`
objective, `lp` tightening, 120 s main solve limit, HiGHS with 1 thread, `norm_order = Inf`. 492
samples built a model in every run; the other 8 are already misclassified and skip formulation.

- Command:
  `benchmarks/run_pair.sh --base master --candidate 56d8d4a --out <run-dir> --samples 1:500 --tightening lp --main-time-limit 120 --norm-order Inf --base-objective feasibility --candidate-objective feasibility`.
  Julia `1.12.6`, single-threaded; HiGHS (an open-source LP/MIP solver) for all solves; sequential
  runs on a local WSL2 workstation, identical dependency snapshots on both sides (HiGHS.jl 1.23.0,
  HiGHS_jll 1.14.0+0). Absolute times are not comparable to the CI-hosted `benchmark-results`
  series.

The candidate ran against a baseline run of `master` `4f1ed43` from a session that benchmarked
several candidates against that one baseline. A second run of `master`, taken later in the same
session, is the control: it runs the same commit as the baseline, so it shows what "no change"
measures as on this machine.

The noise floor quoted above is the spread of the three behaviorally identical runs, 412.1 s to
440.4 s, which is 6.8% of the smallest. Two narrower readings of the same data agree with it. The
two `master` runs alone differ by 5.6%. The session's same-commit control measured a median ratio of
0.95 on `Build + bound tightening`, roughly 5 points of movement with no code change; that metric
includes bound-solver time, so it measures a wider quantity at a similar size and in the same
direction. All three figures exceed the 2.5% ceiling derived below.

Issue [#276](https://github.com/vtjeng/MIPVerify.jl/issues/276) asks whether `run_pair.sh` always
running the baseline commit first biases every candidate measured this way. If it does, the
control's 5-point movement has a direction as well as a size, and a reader who wants to reuse this
noise floor should rerun the pair in both orders before trusting its sign.

The raw per-sample data for this pair was not published to the `benchmark-reports` branch, and the
run directory was not retained, so this report holds the only surviving figures. Three other
candidates from the same session share this baseline and do have published pair data, which includes
the dependency snapshots and the benchmark arguments used throughout the session:
[`2026-07-27-cache-affine-constraint-list`](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-27-cache-affine-constraint-list),
[`2026-07-27-hoist-integrality-relaxation`](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-27-hoist-integrality-relaxation),
and
[`2026-07-27-pr277-index-key-wk17a-lp`](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-27-pr277-index-key-wk17a-lp).

The reported metric is Julia-only formulation time: `formulation_time_seconds` minus
`bound_solver_wall_time_seconds`. This isolates Julia work from solver work. End-to-end totals are
not used as the headline because main solve time varied widely between runs of identical code,
driven by a small number of samples that cross the 120 s limit in some runs and not others.

## Why the estimate was wrong

The estimate was never achievable. Interval arithmetic over these expressions is cheap in absolute
terms, which the profiler's relative shares could not show.

Timing the bound calls directly on WK17a sample 2 measures both the cost of a traversal and the
share of units that reach the second one.

Not every unit reaches it. `progressive_relu_bounds` computes all the upper bounds first, then asks
`interval_lowerbound_for_relu` for each lower bound. That helper returns immediately, without
walking the expression, whenever the unit's interval upper bound is nonpositive: such a unit is
fixed to zero, so nothing downstream reads its lower bound. This report calls that early return
**the skip**. The last two columns below show how much work it removes: how many units still need a
lower bound, and what computing only those costs.

| Layer | Expressions | All upper bounds | All lower bounds | Units needing a lower bound | Lower bounds actually computed |
| ----- | ----------: | ---------------: | ---------------: | --------------------------: | -----------------------------: |
| 1     |       3,136 |         0.0119 s |         0.0116 s |                 923 (29.4%) |                       0.0034 s |
| 2     |       1,568 |         0.0334 s |         0.0340 s |                 769 (49.0%) |                       0.0167 s |
| 3     |         100 |         0.0221 s |         0.0254 s |                    9 (9.0%) |                       0.0023 s |

Cost per expression climbs steeply with depth: one upper bound costs 3.8 µs in layer 1, 21 µs in
layer 2, and 221 µs in layer 3. That is why layer 3 costs more in total than layer 1 despite having
31 times fewer expressions. WK17a is two 4 × 4 stride-2 convolutions with 16 and 32 filters followed
by a fully connected layer of 100 units, so a layer-1 ReLU input is an affine combination over 16
terms, a layer-2 input over 256, and a layer-3 input over all 1,568 layer-2 outputs. Traversal cost
climbs with that term count.

The redundant lower-bound work totals 0.022 s on sample 2 and 0.020 s on sample 3. Scaled to the 492
modeled samples, those two measurements extrapolate to 10 s to 11 s; the scaling assumes the
per-sample cost varies little across the cohort. Baseline Julia-only formulation time is 435 s, so
removing this work completely has a ceiling near 2.5%.

That ceiling sits below the 6.8% noise floor, so this benchmark could not resolve the change even if
it were free. The 3% to 6% estimate came from profiler shares without ever checking what one
traversal costs.

The skip matters too, but it is the smaller factor. If every unit ran the second traversal, the
redundant work would cost 0.071 s for each sample, or 8.0% of Julia-only time. The skip is what
brings the available saving down from that 8.0% to 2.5%. It does not remove the work altogether:
roughly a third of all units still reach the second traversal, as the table shows.

## Limitations

- One network and one perturbation family. The benchmark does not exercise MIP tightening,
  `masked_relu`, or blurring perturbations, all of which reach the same bound code.
- One run for each configuration. The noise floor comes from three runs of behaviorally identical
  code. No configuration was repeated, so the candidate's own run-to-run variation is unmeasured.
- One machine. The drift control bounds how much that machine changed during the experiment, but it
  does not generalize to other hardware.
- The 2.5% ceiling extrapolates per-sample redundant work measured on two samples, 2 and 3, to
  all 492. The share of units reaching the second traversal depends on how many ReLUs the
  upper-bound pass fixes to zero, which varies from sample to sample. The two samples agreed
  closely, 0.022 s and 0.020 s, and that agreement is the only evidence here that the extrapolation
  holds.
- The 2.5% ceiling is specific to WK17a. A network whose formulation cost is dominated by interval
  bounds, rather than by certificate and integrality work, would have a higher ceiling and might
  still benefit.
