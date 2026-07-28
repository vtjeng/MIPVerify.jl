# Single interval traversal benchmark report

- Date: 2026-07-27
- Experimental code:
  [`perf/single-interval-traversal`](https://github.com/vtjeng/MIPVerify.jl/tree/perf/single-interval-traversal)
- Implementation commit: `56d8d4a`
- Related issue: [#267](https://github.com/vtjeng/MIPVerify.jl/issues/267)

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
baseline's 435.1 s.

That difference is smaller than this benchmark's run-to-run variation. Three runs here execute
behaviourally identical code: the baseline, this change, and a repeat of the baseline. Their ReLU
classifications are byte-identical, so any difference between them is measurement noise. Their
Julia-only totals were 435.1 s, 440.4 s, and 412.1 s, a spread of 6.8%. The measured effect of this
change sits inside that spread.

The change is behaviour preserving. The full test suite passed (883 pass, 1 pre-existing broken, 0
fail), and the per-layer ReLU classifications over all 492 modelled samples are byte-identical to
the baseline.

## Why the estimate was wrong

The estimate was never achievable. Interval arithmetic over these expressions is cheap in absolute
terms, which the profiler's relative shares could not show.

Timing the bound calls directly on WK17a sample 2 gives both the cost of a traversal and the share
of units that reach the second one.

Not every unit reaches it. `progressive_relu_bounds` computes all the upper bounds first, then asks
`interval_lowerbound_for_relu` for each lower bound. That helper returns immediately, without
walking the expression, whenever the unit's interval upper bound is nonpositive: such a unit is
fixed to zero, so nothing downstream reads its lower bound. This report calls that early return
**the skip**. The last two columns below show how much work it removes — how many units still need a
lower bound, and what computing only those costs.

| Layer | Expressions | All upper bounds | All lower bounds | Units needing a lower bound | Lower bounds actually computed |
| ----- | ----------: | ---------------: | ---------------: | --------------------------: | -----------------------------: |
| 1     |       3,136 |         0.0119 s |         0.0116 s |                 923 (29.4%) |                       0.0034 s |
| 2     |       1,568 |         0.0334 s |         0.0340 s |                 769 (49.0%) |                       0.0167 s |
| 3     |         100 |         0.0221 s |         0.0254 s |                    9 (9.0%) |                       0.0023 s |

The redundant lower-bound work totals 0.022 s for this sample and 0.020 s for sample 3. Across the
492 modelled samples that is 10 s to 11 s. Baseline Julia-only formulation time is 435 s, so
removing this work completely has a ceiling near 2.5%.

That ceiling sits below the 6.8% noise floor, so this benchmark could not resolve the change even if
it were free. The 3% to 6% estimate came from profiler shares without ever checking what one
traversal costs.

The skip matters too, but it is the smaller factor. Without it — that is, if every unit paid for the
second traversal — the redundant work would cost 0.071 s for each sample, or 8.0% of Julia-only
time. The skip is what brings the available saving down from that 8.0% to 2.5%. It does not remove
the work altogether: roughly a third of all units still reach the second traversal, as the table
shows.

## Method

500 MNIST test samples, network `MNIST.WK17a_linf0.1_authors`, L-infinity radius 0.1, `feasibility`
objective, `lp` tightening, 120 s main solve limit, HiGHS with 1 thread, `norm_order = Inf`. 492
samples built a model in every run; the other 8 are already misclassified and skip formulation.

The candidate ran against a baseline run of `master` from the same session. A second run of
`master`, taken later in the session, is the control: it shares that baseline, so it shows what "no
change" measures as on this machine. The two `master` runs differed by 6.8% on Julia-only
formulation time, which is the noise floor quoted above. #276 tracks the benchmark-ordering question
this raises.

The reported metric is Julia-only formulation time: `formulation_time_seconds` minus
`bound_solver_wall_time_seconds`. This isolates Julia work from solver work. End-to-end totals are
not used as the headline because main solve time varied widely between runs of identical code,
driven by a small number of samples that cross the 120 s limit in some runs and not others.

## Limitations

- One network and one perturbation family. The benchmark does not exercise MIP tightening,
  `masked_relu`, or blurring perturbations, all of which reach the same bound code.
- One run for each configuration. The noise floor is estimated from three behaviourally identical
  runs, not from repetitions of each configuration.
- One machine. The drift control bounds how much that machine changed during the experiment, but it
  does not generalise to other hardware.
- The 2.5% ceiling is specific to WK17a. A network whose formulation cost is dominated by interval
  bounds, rather than by certificate and integrality work, would have a higher ceiling and might
  still benefit.
