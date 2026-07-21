# PGD full-start benchmark report

Date: 2026-07-20  
Implementation commit: `f7e2bec107f4155b84a6d69a04cfda9be950217b`

## Conclusion

Warm-starting verification with the strongest non-adversarial PGD candidate did not help this
benchmark. It increased simplex work by 6.8% across the fixed cohort and by 21.6% on the four-case
hard tail. The fixed-cohort 95% bootstrap intervals exclude an improvement. PGD generation and
full-start completion increased end-to-end time by 70.5%.

The random control rules out a generic penalty from merely supplying a complete start as the main
explanation. Its cohort-level simplex ratio was close to one, although it timed out on one hard
sample in all three repetitions. PGD had a better objective margin than the random candidate on all
12 samples, but did more simplex work on five samples, improved none, and tied seven. Incumbent
objective quality did not order branch-and-bound performance.

## Method

The benchmark used MNIST WK17a, an L-infinity radius of `0.1`, LP bound tightening, HiGHS 1.14.x,
one thread, parallel mode off, solver seed zero, and a 30-second solve limit. PGD used 20 restarts,
100 steps, step size `0.01`, and base seed `20260720`. It ran with Julia 1.12.6, HiGHS.jl 1.23.0,
and HiGHS_jll 1.14.0+0. The candidate cache SHA-256 was
`f515fe0bf78e515344b3a2a3c7408e3362ba4a2f8acbfcd67537e0d55c2f6640`. The fixed cohort contained
eight predeclared hard samples and four controls. Each sample used one tightened base model and
three serial, block-rotated copies per treatment:

- `cold`: no start;
- `random_full`: one unselected, uniformly sampled point from the perturbation box, completed to all
  MIP variables; and
- `pgd_full`: the best PGD near-miss, completed through the same mechanism.

The primary metric is the median simplex-iteration count across the three repetitions for each
sample. Aggregate ratios are geometric means over samples; their intervals use a sample-level
bootstrap. End-to-end time charges candidate generation and full-start completion to the
corresponding treatment. The complete protocol and decision thresholds are in
[README.md](README.md#decision-rules-and-definition-of-done).

## Fixed-cohort results

| Metric | `pgd_full / cold` | `random_full / cold` | `pgd_full / random_full` |
| --- | ---: | ---: | ---: |
| Simplex iterations | 1.068 [1.006, 1.160] | 1.013 [0.968, 1.070] | 1.055 [0.965, 1.170] |
| Nodes | 0.911 [0.647, 1.203] | 0.685 [0.351, 1.099] | 1.329 [0.881, 2.605] |
| Main solve time | 1.059 [0.938, 1.224] | 1.018 [0.889, 1.224] | 1.040 [0.874, 1.283] |
| End-to-end time | 1.705 [1.457, 1.982] | 1.063 [0.950, 1.250] | 1.603 [1.253, 1.994] |

Intervals are 95% bootstrap intervals. All 36 cold and 36 PGD solves certified robustness. The
random treatment certified 33 times and reached the common time limit on sample 246 in all three
repetitions. Both full-start treatments were accepted or observed as the initial incumbent in every
run.

| Sample | Stratum | PGD margin | Random margin | Cold simplex | Random simplex | PGD simplex | PGD/cold |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | hard | -1.040 | -3.316 | 2,648 | 2,648 | 2,648 | 1.000 |
| 32 | control | -4.636 | -5.119 | 108 | 108 | 108 | 1.000 |
| 36 | control | -8.972 | -10.421 | 242 | 242 | 242 | 1.000 |
| 46 | hard | -0.982 | -3.576 | 33,491 | 36,417 | 46,667 | 1.393 |
| 233 | hard | -0.674 | -2.915 | 652 | 652 | 660 | 1.012 |
| 246 | hard | -1.138 | -3.550 | 34,290 | 32,745* | 36,541 | 1.066 |
| 313 | control | -8.257 | -9.834 | 285 | 285 | 285 | 1.000 |
| 359 | hard | -0.390 | -2.441 | 15,660 | 20,440 | 16,236 | 1.037 |
| 407 | hard | -0.787 | -2.104 | 28,058 | 24,123 | 39,864 | 1.421 |
| 444 | hard | -0.314 | -1.761 | 609 | 609 | 609 | 1.000 |
| 460 | control | -7.924 | -9.590 | 220 | 220 | 220 | 1.000 |
| 479 | hard | -0.766 | -2.644 | 1,211 | 1,211 | 1,211 | 1.000 |

`*` Sample 246's random value is work at the 30-second cutoff, not completed verification work.
Its final objective bounds were positive, while cold and PGD certified.

## Difficult-case and variance checks

The four hardest non-attack cases by pre-treatment cold work were 46, 246, 359, and 407. Two fresh
Julia processes ran them in forward and reversed treatment order. Resolved treatments reproduced
the same simplex and node counters across both processes. The PGD/cold simplex ratio was 1.216 with
a 95% interval of [1.051, 1.407]. Random/cold was 1.034 in the forward run and 1.046 in the reverse
run; the difference came from the time-limited sample 246.

The main cohort supplied three repetitions per sample. All work counters were identical within a
sample and treatment except:

- sample 46 `pgd_full`: two runs completed with 46,667 simplex iterations and 86 nodes; the third
  reached the time boundary and certified with 45,563 iterations and 25 nodes; and
- sample 246 `random_full`: all runs timed out, with 32,122 to 32,922 simplex iterations.

Wall times varied even when exact work counters matched. The comparable post-fix historical cold
medians overlapped on samples 19, 32, 36, and 46; fresh/historical time ratios ranged from 0.916 to
1.009, so the cold-reference check raised no flag. The fresh hard-tail processes also reproduced
the cold work counters exactly.

Peak recorded process RSS was 2,130 MiB. The lowest recorded system-available memory was 11,016
MiB, above the 4 GiB guard. Runs were serial, and no memory guard or cgroup OOM event fired.

## Validation

- Package suite: 884 passed, one test marked broken, zero failures.
- Benchmark helper suite: 108 passed.
- Focused PGD warm-start suite: 46 passed.
- Julia formatting check: passed.
- Result audit: 12 sample rows, 108 treatment rows, three repetitions for every sample/treatment,
  72 of 72 supplied starts used, and identical copied-model signatures within each sample.

## Interpretation

This non-monotonic behavior is known in branch and bound. Ragsdale and Shapiro reported that a
better initial incumbent can produce a longer search and cautioned that incumbent use is heuristic
([paper abstract](https://www.sciencedirect.com/science/article/pii/0305054895000372)). HiGHS uses a
feasible MIP start to set the primal upper bound and node-queue optimality limit, so the start can
change the subsequent search rather than only provide a passive benchmark
([HiGHS 1.14 source](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/mip/HighsMipSolverData.cpp#L793-L831)).

The results are consistent with that mechanism. A complete start supplies a discrete ReLU and
selector assignment in addition to a scalar margin. PGD improves the scalar margin, but its induced
discrete assignment can steer cuts, heuristics, pruning, and node selection onto a worse path. This
is an inference from the solver mechanism and observed counters; the benchmark does not isolate
which individual HiGHS decision caused each regression.

The earlier sparse-start results are not part of the primary comparison. Partial starts make HiGHS
run a completion procedure, which is a separate source of work
([HiGHS MIP-start documentation](https://ergo-code.github.io/HiGHS/stable/guide/further/)). The
complete random and PGD treatments control for that interface difference.

## Limitations

- The cohort covers one network, perturbation radius, solver generation, and fixed solver seed.
- Three order-rotated repetitions and two fresh hard-tail processes characterize timing and
  fixed-limit variance, but do not sample different solver random seeds. Gurobi's performance
  guidance recommends comparing distributions over multiple seeds for a general solver claim
  ([staff guidance](https://support.gurobi.com/hc/en-us/community/posts/39029910894353-Why-is-it-slower-to-enter-MIP-start-than-not-to-enter-it)).
- Seven of 12 samples solve at the root with identical work across treatments. The hard-tail result
  is more informative about branch-and-bound behavior but has only four samples.
- The random control is one deterministic draw per sample. Multiple random points could measure the
  distribution of discrete start quality, but are not needed to reject the proposed PGD warm start
  under the recorded decision rule.

Under the predeclared definition of done, the solver-level conclusion is **not supported** and the
end-to-end conclusion is **not practical**.
